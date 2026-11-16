"""평가 틀(단계 0-7) — 판 하나의 채점·단계별 표 한 줄·test 막기·덜 찬 달 분리.

계획서 §1-2(지표·threshold·동점), §1-3(보고·test 막기), §3(채점 쌍·덜 찬 달), 부록 A-12(표 칼럼·
동점 많은 판·'같은 validation 으로 고른 횟수'·판별 학습 횟수), A-13(표 주석), B-3(재채점).

- 지표는 `lst.metrics` 만 씀(trapz PR-AUC — AP 함수는 불러오지도 않음).
- 예측 = DataFrame(lims_data_key·h·fold·pred), fold ∈ {fold1, fold2, fold3, refit}(refit = test 쌍).
- CV 정답 = 시료 표에서 d ≤ 2025-06-30 행만 읽은 y(`history.read_samples_blind_test`) — test 기간 y 는
  메모리에 없음. test 정답은 `report_permit()` 로 받은 허가가 있을 때만 `TestTruth.load` 로 읽음.
- test 막기(§1-3): 고르는 코드는 `selection_guard()` 안에서 돔 — 그 안에서 허가 만들기·test 정답 읽기·
  test 칸이 든 후보를 받으면 `TestPeekError` 로 멈춤.
- 덜 찬 달(§3): 단계 0-1 이 정한 (src_route, 'YYYY-MM') 의 d 시료 쌍은 test 채점에서 빼고 '미완 구간' 으로
  따로 보고(쌍 수·검출 수 = 국내·해외 × h 구간, 그 쌍만의 점수).
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import folds as fd
from . import metrics as mt
from .hassign import H_MAX, u_tie_many

PRED_COLS = ("lims_data_key", "h", "fold", "pred")
CV_FOLDS = fd.CV_FOLDS
TEST_FOLD = "refit"
SAMPLE_COLS = ("lims_data_key", "y", "src_route", "sgg", "sido", "p2", "p3", "d", "h")
RESCORE_CLOSE = 0.001          # 부록 A-12: 동점 많은 판이 1위(또는 현재 최고 판)와 0.001 안이면 다시 채점
BAND_NAMES = ("h01_20", "h21_40", "h41_60")
ROUTES = ("domestic", "overseas")


class TestPeekError(RuntimeError):
    """§1-3 test 막기 위반 — 고르는 코드가 test 정답·test 점수를 읽으려 함."""


# ================================================================ test 막기
_SELECTION_DEPTH = 0
_PERMIT_TOKEN = object()


@contextmanager
def selection_guard():
    """이 안에서 도는 코드 = '고르는 코드'(§1-3). test 정답 읽기·허가 만들기 → TestPeekError."""
    global _SELECTION_DEPTH
    _SELECTION_DEPTH += 1
    try:
        yield
    finally:
        _SELECTION_DEPTH -= 1


def in_selection() -> bool:
    return _SELECTION_DEPTH > 0


@dataclass(frozen=True)
class ReportPermit:
    """'보고 단계' 플래그(§1-3: test 정답 채점은 각 단계가 끝난 뒤 보고 때만)."""
    stage: str
    reason: str
    _token: object = field(repr=False, compare=False, default=None)


def report_permit(stage: str, reason: str) -> ReportPermit:
    if in_selection():
        raise TestPeekError("고르는 코드 안에서 test 보고 허가를 만들 수 없음(§1-3)")
    if not stage or not reason:
        raise ValueError("보고 허가에는 단계·이유가 필요")
    return ReportPermit(stage, reason, _PERMIT_TOKEN)


def check_permit(permit) -> ReportPermit:
    if not isinstance(permit, ReportPermit) or permit._token is not _PERMIT_TOKEN:
        raise TestPeekError("test 정답은 report_permit() 로 받은 허가가 있어야 읽음(§1-3)")
    if in_selection():
        raise TestPeekError("고르는 코드가 test 정답을 읽으려 함 — 멈춤(§1-3)")
    return permit


# ================================================================ 시료 표·정답
class SampleTable:
    """단계 0-1 시료 표(행 처리 뒤) + CV 정답(d ≤ refit train 끝 행만).

    df 의 y 는 Int8, d > 2025-06-30 행은 <NA>(`history.read_samples_blind_test` 결과 그대로).
    """

    def __init__(self, df: pd.DataFrame):
        miss = [c for c in SAMPLE_COLS if c not in df.columns]
        if miss:
            raise ValueError(f"시료 표 칼럼 없음: {miss}")
        if df["lims_data_key"].duplicated().any():
            raise ValueError("시료 표 lims_data_key 중복")
        self.df = df.reset_index(drop=True)
        self.keys = self.df["lims_data_key"].astype(str).to_numpy(dtype=object)
        self.index = pd.Index(self.keys)
        self.d_days = fd.to_days(self.df["d"])
        self.src_route = self.df["src_route"].astype(str).to_numpy(dtype=object)
        bad = set(np.unique(self.src_route)) - set(ROUTES)
        if bad:
            raise ValueError(f"src_route 값: {bad}")
        y = self.df["y"]
        known = self.d_days <= fd.SPLITS["refit"].train_end
        if y.isna().to_numpy()[known].any():
            raise ValueError("d ≤ 2025-06-30 행 y 결측")
        if y.notna().to_numpy()[~known].any():
            raise TestPeekError("시료 표에 test 기간 y 가 들어 있음 — read_samples_blind_test 로 읽을 것(§1-3)")
        self._y = y.to_numpy(dtype="float64", na_value=np.nan)
        if not np.isin(self._y[known], (0.0, 1.0)).all():
            raise ValueError("y 는 0/1 만")
        self._expected: dict[str, int] = {}

    @classmethod
    def read(cls, path) -> "SampleTable":
        from .history import read_samples_blind_test

        return cls(read_samples_blind_test(path, SAMPLE_COLS))

    def positions(self, keys) -> np.ndarray:
        pos = self.index.get_indexer(pd.Index(np.asarray(keys, dtype=object)))
        if (pos < 0).any():
            raise KeyError(f"시료 표에 없는 lims_data_key {int((pos < 0).sum())}개")
        return pos

    def meta(self) -> pd.DataFrame:
        """정답 없는 행 표(입력 조립용)."""
        return self.df.drop(columns=["y"])

    def known_frame(self) -> pd.DataFrame:
        """d ≤ 2025-06-30 행(train 행 만들기용, y int8)."""
        m = self.d_days <= fd.SPLITS["refit"].train_end
        out = self.df.loc[m].copy()
        out["y"] = out["y"].astype(np.int8)
        return out

    def cv_y(self, pos: np.ndarray) -> np.ndarray:
        y = self._y[pos]
        if np.isnan(y).any():
            raise TestPeekError("CV 채점 쌍에 test 기간 시료가 들어 있음(§1-3)")
        return y.astype(np.int64)

    def expected_pairs(self, split: str) -> int:
        """그 판 채점 쌍 수(§3: T ∈ 채점 범위인 (시료, h) 쌍 전부)."""
        if split not in self._expected:
            self._expected[split] = int(sum(idx.size for _, idx in
                                            fd.iter_scoring_pairs_by_h(self.d_days, fd.SPLITS[split])))
        return self._expected[split]


class TestTruth:
    """test 기간(d > 2025-06-30) 정답 — 허가가 있을 때만 읽고, 읽을 때마다 허가·고르기 여부를 다시 확인."""

    def __init__(self, keys, y, permit: ReportPermit):
        self._permit = check_permit(permit)
        self._index = pd.Index(np.asarray(keys, dtype=object))
        if self._index.duplicated().any():
            raise ValueError("test 정답 키 중복")
        self._y = np.asarray(y, dtype=np.int64)
        if not np.isin(self._y, (0, 1)).all():
            raise ValueError("test y 는 0/1 만")

    @classmethod
    def load(cls, samples_path, permit: ReportPermit) -> "TestTruth":
        import datetime as _dt

        import pyarrow.parquet as pq

        check_permit(permit)
        cut = _dt.date.fromisoformat(str(fd.as_date([fd.SPLITS["refit"].train_end])[0]))
        t = pq.read_table(samples_path, columns=["lims_data_key", "y"], filters=[("d", ">", cut)]).to_pandas()
        return cls(t["lims_data_key"].astype(str).to_numpy(dtype=object), t["y"].to_numpy(), permit)

    @property
    def permit(self) -> ReportPermit:
        return self._permit

    def y_for(self, keys) -> np.ndarray:
        check_permit(self._permit)
        pos = self._index.get_indexer(pd.Index(np.asarray(keys, dtype=object)))
        if (pos < 0).any():
            raise KeyError(f"test 정답에 없는 시료 {int((pos < 0).sum())}개")
        return self._y[pos]


def load_incomplete(path) -> set[tuple[str, str]]:
    """단계 0-1 덜 찬 달 결과(`12_incomplete_months.json` 의 incomplete = [[src_route, 'YYYY-MM'], …])."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out = set()
    for r, m in obj["incomplete"]:
        if r not in ROUTES or m not in fd.TEST_TAIL_MONTHS:
            raise ValueError(f"덜 찬 달 항목 오류: {r, m}")
        out.add((str(r), str(m)))
    return out


# ================================================================ 쌍 준비
@dataclass
class Pairs:
    split: str
    pos: np.ndarray        # 시료 표 위치
    h: np.ndarray          # int64
    pred: np.ndarray       # float64
    d_days: np.ndarray
    src_route: np.ndarray
    keys: np.ndarray

    def u(self, rescore: bool = False) -> np.ndarray:
        return u_tie_many(self.keys, self.h, rescore=rescore)

    def take(self, m: np.ndarray) -> "Pairs":
        return Pairs(self.split, self.pos[m], self.h[m], self.pred[m], self.d_days[m], self.src_route[m], self.keys[m])


def prepare_pairs(preds: pd.DataFrame, samples: SampleTable, split: str, *, require_complete: bool = True) -> Pairs:
    """예측 표 점검·시료 표와 연결. 채점 쌍이 아닌 행·중복·빠진 쌍이 있으면 멈춤."""
    miss = [c for c in PRED_COLS if c not in preds.columns]
    if miss:
        raise ValueError(f"예측 칼럼 없음: {miss}")
    fold = preds["fold"].astype(str).to_numpy()
    if (fold != split).any():
        raise ValueError(f"예측 fold 칼럼이 {split} 아님")
    keys = preds["lims_data_key"].astype(str).to_numpy(dtype=object)
    h = preds["h"].to_numpy().astype(np.int64)
    if h.min(initial=1) < 1 or h.max(initial=1) > H_MAX:
        raise ValueError("h 는 1~60")
    pred = preds["pred"].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("예측에 NaN·inf")
    if pd.DataFrame({"k": keys, "h": h}).duplicated().any():
        raise ValueError("(lims_data_key, h) 중복 예측")
    pos = samples.positions(keys)
    d = samples.d_days[pos]
    sp = fd.SPLITS[split]
    T = d - h
    if ((T < sp.score_start) | (T > sp.score_end)).any():
        raise ValueError(f"{split}: 채점 범위 밖 T 쌍이 예측에 있음")
    if (d <= sp.train_end).any():
        raise AssertionError(f"{split}: 채점 쌍이 train 기간 시료")
    if require_complete and len(keys) != samples.expected_pairs(split):
        raise ValueError(f"{split}: 채점 쌍 {len(keys)} ≠ 기대 {samples.expected_pairs(split)}(§3 쌍 전부)")
    return Pairs(split, pos, h, pred, d, samples.src_route[pos], keys)


def expand_sample_preds(sample_preds: pd.DataFrame, samples: SampleTable, split: str,
                        value_col: str = "p_hat") -> pd.DataFrame:
    """시료 단위 예측(모든 h 같은 값 — 7-a)을 (시료, h) 채점 쌍 예측으로 펼침."""
    s = sample_preds.drop_duplicates("lims_data_key")
    if len(s) != len(sample_preds):
        raise ValueError("시료 예측 키 중복")
    lut = pd.Series(s[value_col].to_numpy(dtype=np.float64), index=s["lims_data_key"].astype(str))
    parts = []
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, fd.SPLITS[split]):
        if idx.size == 0:
            continue
        k = samples.keys[idx]
        v = lut.reindex(k).to_numpy()
        if np.isnan(v).any():
            raise KeyError(f"{split} h{h}: 시료 예측 없는 채점 시료 {int(np.isnan(v).sum())}개")
        parts.append(pd.DataFrame({"lims_data_key": k, "h": np.full(idx.size, h, np.int16), "fold": split, "pred": v}))
    return pd.concat(parts, ignore_index=True)


# ================================================================ 채점
def _common_d_score(P: Pairs, y: np.ndarray, u: np.ndarray) -> float:
    m = fd.common_d_mask(P.d_days, fd.SPLITS[P.split])
    if not m.any():
        return math.nan
    return mt.pr_auc_band_weighted(y[m], P.pred[m], P.h[m], u[m])


def score_cv(preds: Mapping[str, pd.DataFrame], samples: SampleTable, *,
             thr_preds: Mapping[str, pd.DataFrame] | None = None, rescore: bool = False,
             require_complete: bool = True) -> dict:
    """판 하나의 CV 채점(§1-2): fold 별 지표 전부·CV 평균·threshold(모든 fold validation 예측 합침)·동점 요약.

    preds = {fold1, fold2, fold3: 예측 표}. thr_preds = threshold 용 예측(2차 사슬: 공통 반복 수 N 예측 —
    1차·단계 2 준비는 없음 → CV 예측으로). rescore=True = u_tie 를 md5(key|h|1) 로(부록 A-12 다시 채점).
    """
    if set(preds) != set(CV_FOLDS):
        raise ValueError(f"CV 예측은 {CV_FOLDS} 전부")
    out_folds, ys, ps, tps, fold_of = {}, [], [], [], []
    for f in CV_FOLDS:
        P = prepare_pairs(preds[f], samples, f, require_complete=require_complete)
        y = samples.cv_y(P.pos)
        u = P.u(rescore)
        rep = mt.score_report(y, P.pred, P.h, u)
        rep["common_d_band_weighted"] = _common_d_score(P, y, u)
        out_folds[f] = rep
        ys.append(y)
        ps.append(P.pred)
        fold_of.append(np.full(y.shape[0], f, dtype=object))
        if thr_preds is not None:
            Q = prepare_pairs(thr_preds[f], samples, f, require_complete=require_complete)
            if not (np.array_equal(Q.pos, P.pos) and np.array_equal(Q.h, P.h)):
                raise ValueError(f"{f}: threshold 용 예측의 쌍 순서가 CV 예측과 다름")
            tps.append(Q.pred)
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    t_all = np.concatenate(tps) if thr_preds is not None else p_all
    folds_all = np.concatenate(fold_of)
    fbw = {f: out_folds[f]["pr_auc_band_weighted"] for f in CV_FOLDS}
    cv = mt.cv_mean(fbw)
    thr = mt.select_threshold(y_all, t_all)
    fold_recall = {}
    for f in CV_FOLDS:
        m = folds_all == f
        pos = int(y_all[m].sum())
        fold_recall[f] = math.nan if pos == 0 else float(((t_all[m] >= thr["threshold"]) & (y_all[m] == 1)).sum() / pos)
    return {
        "kind": "cv", "rescore": rescore,
        "cv_band_weighted": cv, "cv_band_weighted_round4": mt.round4(cv),
        "fold_band_weighted": fbw,
        "cv_h_bands": {b: mt.cv_mean([out_folds[f][f"pr_auc_{b}"] for f in CV_FOLDS]) for b in BAND_NAMES},
        "cv_pooled": mt.cv_mean([out_folds[f]["pr_auc_pooled"] for f in CV_FOLDS]),
        "cv_sklearn_default_band_weighted": mt.cv_mean([out_folds[f]["sklearn_default_band_weighted"] for f in CV_FOLDS]),
        "cv_common_d_band_weighted": mt.cv_mean([out_folds[f]["common_d_band_weighted"] for f in CV_FOLDS]),
        "cv_lift_band_weighted": mt.cv_mean([out_folds[f]["lift_band_weighted"] for f in CV_FOLDS]),
        "threshold": thr,
        "threshold_source": "thr_preds" if thr_preds is not None else "cv_preds",
        "fold_recall_at_threshold": fold_recall,
        "ties": mt.tie_summary(p_all, y_all, int(y_all.shape[0])),
        "n_pairs": int(y_all.shape[0]), "n_pos": int(y_all.sum()),
        "folds": out_folds,
    }


def _band_of(h: np.ndarray) -> np.ndarray:
    return np.where(h <= 20, BAND_NAMES[0], np.where(h <= 40, BAND_NAMES[1], BAND_NAMES[2]))


def score_test(test_preds: pd.DataFrame, samples: SampleTable, truth: TestTruth, threshold: float,
               incomplete: Iterable[tuple[str, str]] = (), *, require_complete: bool = True) -> dict:
    """refit test 채점(보고 때만, §1-3) — threshold 는 CV 에서 정한 값 그대로(§1-2).

    덜 찬 달 쌍은 본 점수에서 빼고 `incomplete` 칸에 따로(쌍 수·검출 수 = 국내·해외 × h 구간, 그 쌍만의 점수).
    """
    check_permit(truth.permit)
    P = prepare_pairs(test_preds, samples, TEST_FOLD, require_complete=require_complete)
    y = truth.y_for(P.keys)
    inc = set(incomplete)
    bad = fd.incomplete_pair_mask(P.d_days, P.src_route, inc) if inc else np.zeros(y.shape[0], bool)
    main = ~bad
    Pm, ym = P.take(main), y[main]
    um = Pm.u()
    rep = mt.score_report(ym, Pm.pred, Pm.h, um)
    rep["common_d_band_weighted"] = _common_d_score(Pm, ym, um)
    rep["threshold_metrics"] = mt.threshold_metrics(ym, Pm.pred, Pm.h, threshold)
    # 미완 구간(따로 보고)
    Pb, yb = P.take(bad), y[bad]
    bands = _band_of(Pb.h)
    counts = []
    for r in ROUTES:
        for b in BAND_NAMES:
            m = (Pb.src_route == r) & (bands == b)
            counts.append({"src_route": r, "band": b, "n_pairs": int(m.sum()), "n_pos": int(yb[m].sum())})
    inc_rep = {"months": sorted(inc), "n_pairs": int(bad.sum()), "n_pos": int(yb.sum()), "counts": counts,
               "pr_auc_band_weighted": (mt.pr_auc_band_weighted(yb, Pb.pred, Pb.h, Pb.u()) if bad.any() else math.nan),
               "pr_auc_pooled": (mt.pr_auc_pooled(yb, Pb.pred, Pb.u()) if bad.any() else math.nan)}
    return {"kind": "test", "permit_stage": truth.permit.stage, "report": rep, "incomplete": inc_rep,
            "h_composition_changed": bool(bad.any())}


# ================================================================ 단계별 표(부록 A-12)
TABLE_COLUMNS = (
    "단계·판", "입력·모델", "표시", "CV 구간 가중(평균)", "fold 1", "fold 2", "fold 3",
    "같은 validation 으로 고른 횟수",
    "test 구간 가중", "test h 1~20 평균", "test h 21~40 평균", "test h 41~60 평균", "test 60일 통",
    "공통 d 범위 구간 가중(CV)", "공통 d 범위 구간 가중(test)",
    "배수(구간 가중)", "배수(60일 통)", "threshold",
    "Recall(test 쌍 전체)", "Recall(구간 가중)", "Precision", "F1", "F2", "TP", "FN", "FP", "TN",
    "상위 1% 포착률", "상위 5% 포착률", "상위 10% 포착률", "상위 20% 포착률",
    "합친 threshold 의 fold 별 Recall",
)
MANY_TIES_MARK = "동점 많음(원 예측값 기준)"
INCOMPLETE_MARK = "미완 구간 제외로 h 구성 달라짐"
CAT_AUTO_LR_MARK = "학습률 = 학습마다 자동값(fold·refit 다름)"

TABLE_NOTES_BASE = (
    "test 표: (예보 칸 = 관측값)",
    "해마다 T 05-02 ~ 06-30 은 채점 안 됨",
    "같은 validation 으로 고른 횟수: 반복 수 고르기(판마다 56 시점)는 셈에서 빠짐",
    "test 점수 = 국내 + 해외 전체 하나",
)


def table_row(name: str, desc: str, cv: Mapping | None, test: Mapping | None = None, *,
              selection_count: int | None = None, marks: Sequence[str] = (),
              fold_recall: bool = False) -> dict:
    """단계별 표 한 줄(부록 A-12 칼럼). test 가 없으면 test 칸 비움(보고 전).

    fold_recall=True = 단계 4·5·6 판(합친 threshold 의 fold 별 Recall 칸 채움).
    """
    row = {c: math.nan for c in TABLE_COLUMNS}
    marks = list(marks)
    row["단계·판"] = name
    row["입력·모델"] = desc
    row["같은 validation 으로 고른 횟수"] = selection_count if selection_count is not None else math.nan
    if cv is not None:
        row["CV 구간 가중(평균)"] = cv["cv_band_weighted"]
        for i, f in enumerate(CV_FOLDS, 1):
            row[f"fold {i}"] = cv["fold_band_weighted"][f]
        row["공통 d 범위 구간 가중(CV)"] = cv["cv_common_d_band_weighted"]
        row["threshold"] = cv["threshold"]["threshold"]
        if cv["ties"]["many_ties"] and MANY_TIES_MARK not in marks:
            marks.append(MANY_TIES_MARK)
        if fold_recall:
            row["합친 threshold 의 fold 별 Recall"] = " / ".join(
                f"{cv['fold_recall_at_threshold'][f]:.4f}" for f in CV_FOLDS)
        else:
            row["합친 threshold 의 fold 별 Recall"] = ""
    if test is not None:
        r = test["report"]
        tm = r["threshold_metrics"]
        if cv is not None and not math.isclose(tm["threshold"], cv["threshold"]["threshold"], rel_tol=0, abs_tol=0):
            raise AssertionError("test threshold 가 CV threshold 와 다름(§1-2)")
        row.update({
            "test 구간 가중": r["pr_auc_band_weighted"],
            "test h 1~20 평균": r["pr_auc_h01_20"], "test h 21~40 평균": r["pr_auc_h21_40"],
            "test h 41~60 평균": r["pr_auc_h41_60"], "test 60일 통": r["pr_auc_pooled"],
            "공통 d 범위 구간 가중(test)": r["common_d_band_weighted"],
            "배수(구간 가중)": r["lift_band_weighted"], "배수(60일 통)": r["lift_pooled"],
            "Recall(test 쌍 전체)": tm["recall"], "Recall(구간 가중)": tm["recall_band_weighted"],
            "Precision": tm["precision"], "F1": tm["f1"], "F2": tm["f2"],
            "TP": tm["tp"], "FN": tm["fn"], "FP": tm["fp"], "TN": tm["tn"],
        })
        for q in mt.TOP_Q:
            row[f"상위 {q}% 포착률"] = r["top_q_capture"][q]
        if test["h_composition_changed"] and INCOMPLETE_MARK not in marks:
            marks.append(INCOMPLETE_MARK)
    row["표시"] = "; ".join(marks)
    return row


def table_notes(tests: Sequence[Mapping] = (), extra: Sequence[str] = ()) -> list[str]:
    """표 주석(부록 A-13): 고정 문구 + 덜 찬 달로 뺀 쌍 수·검출 수(국내·해외 × h 구간)."""
    notes = list(TABLE_NOTES_BASE)
    for t in tests:
        inc = t["incomplete"]
        if inc["n_pairs"]:
            parts = ", ".join(f"{c['src_route']} {c['band']} {c['n_pairs']}쌍/검출 {c['n_pos']}"
                              for c in inc["counts"] if c["n_pairs"])
            notes.append(f"덜 찬 달 {inc['months']} 로 test 채점에서 뺀 쌍: {parts}")
            break  # 모든 판이 같은 쌍을 뺌 — 한 번만
    notes.extend(extra)
    return notes


def write_table(rows: Sequence[Mapping], notes: Sequence[str], path) -> Path:
    """단계별 표 csv(utf-8-sig) + 옆에 주석 json. 덮어쓰기 금지."""
    from .config import dump_json

    p = Path(path)
    df = pd.DataFrame(list(rows), columns=list(TABLE_COLUMNS))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", newline="", encoding="utf-8-sig") as fh:
        df.to_csv(fh, index=False)
    dump_json(p.with_suffix(".notes.json"), {"notes": list(notes)})
    return p


# ================================================================ 고르기(§1-2 동점·부록 A-12)
_TEST_KEY_PREFIXES = ("test", "Test", "TEST")


@dataclass(frozen=True)
class Candidate:
    """고르기 대상 한 줄 — CV 값만. extra 에 test 칸이 있으면 멈춤(§1-3)."""
    name: str
    cv: float
    train_count: int          # 판별 학습 횟수(부록 A-12: 트리 판 4·5·6 = 20·앙상블 0·7-a 0 …)
    many_ties: bool
    extra: Mapping[str, Any] = field(default_factory=dict)


def _check_candidates(cands: Sequence[Candidate]) -> None:
    for c in cands:
        if not isinstance(c, Candidate):
            raise TypeError("고르기는 Candidate 만 받음")
        bad = [k for k in c.extra if str(k).startswith(_TEST_KEY_PREFIXES) or "test" in str(k).lower()]
        if bad:
            raise TestPeekError(f"고르는 코드에 test 칸이 들어옴({c.name}: {bad}) — 멈춤(§1-3)")
        if math.isnan(c.cv):
            raise ValueError(f"CV NaN 후보: {c.name}")
    names = [c.name for c in cands]
    if len(set(names)) != len(names):
        raise ValueError("후보 이름 중복")


def rank_candidates(cands: Sequence[Candidate]) -> list[Candidate]:
    """CV 소수 4자리 반올림 큰 순 → 판의 학습 횟수 적은 순 → 판 이름 순(§1-2)."""
    with selection_guard():
        _check_candidates(cands)
        return sorted(cands, key=lambda c: (-mt.round4(c.cv), c.train_count, c.name))


def select_with_tie_check(cands: Sequence[Candidate], rescore_fn: Callable[[str], float] | None) -> dict:
    """고르기 + 동점 많은 판 다시 채점(부록 A-12, 1차·단계 2 준비 고르기).

    동점 많은 판이 CV 1위와 0.001 안이면(1위면 2위와) rescore_fn(판 이름) = u_tie 를 바꾼 CV 로 다시 채점,
    다시 채점한 값으로 순위가 그대로인지 기록(규칙은 그대로 — 고른 판은 원 순위 1위). CV 1위가 동점 많으면 알림.
    """
    with selection_guard():
        ranked = rank_candidates(cands)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        rescored: dict[str, float] = {}
        for c in ranked:
            if not c.many_ties:
                continue
            ref = second if c is top else top
            if ref is None or abs(c.cv - ref.cv) > RESCORE_CLOSE:
                continue
            if rescore_fn is None:
                raise ValueError("다시 채점이 필요한데 rescore_fn 없음")
            for n in (c.name, ref.name):
                if n not in rescored:
                    rescored[n] = float(rescore_fn(n))
        alerts = []
        if top.many_ties:
            alerts.append(f"CV 1위 {top.name} 이 '{MANY_TIES_MARK}' — 확정 전 알림(부록 A-12)")
        ranking_rescore = None
        if rescored:
            alt = [Candidate(c.name, rescored.get(c.name, c.cv), c.train_count, c.many_ties) for c in ranked]
            ranking_rescore = [c.name for c in sorted(alt, key=lambda c: (-mt.round4(c.cv), c.train_count, c.name))]
        return {
            "selected": top.name,
            "ranking": [c.name for c in ranked],
            "cv": {c.name: c.cv for c in ranked},
            "rescored": rescored,
            "ranking_rescore": ranking_rescore,
            "rescore_same_top": None if ranking_rescore is None else ranking_rescore[0] == top.name,
            "alerts": alerts,
        }


def judge_adoption(cand: Candidate, current: Candidate, rescore_fn: Callable[[str], float] | None) -> dict:
    """2차 사슬 채택 판정(§1-2·§6): 4자리 반올림 뒤 오를 때만 채택. 동점 많은 후보가 현재 최고 판과 0.001 안이면
    다시 채점해 기록(규칙은 그대로), 채택된 판이 동점 많으면 알림."""
    with selection_guard():
        _check_candidates([cand, current])
        adopt = mt.is_improvement(cand.cv, current.cv)
        rec = {"candidate": cand.name, "current": current.name, "adopt": adopt, "rescore": None, "alerts": []}
        if (cand.many_ties or current.many_ties) and abs(cand.cv - current.cv) <= RESCORE_CLOSE:
            if rescore_fn is None:
                raise ValueError("다시 채점이 필요한데 rescore_fn 없음")
            a, b = float(rescore_fn(cand.name)), float(rescore_fn(current.name))
            rec["rescore"] = {"candidate": a, "current": b, "adopt_under_rescore": mt.is_improvement(a, b)}
        if adopt and cand.many_ties:
            rec["alerts"].append(f"채택된 판 {cand.name} 이 '{MANY_TIES_MARK}'(부록 A-12)")
        return rec
