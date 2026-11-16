"""2차 사슬 뒷부분(단계 4 다운사이징·5·6 손 교수 방식·8 튜닝·추가 판 ①②·최종 모델) 공통 — 계획서 요약 15~20,
§6 2차 공통('단계 4~6 을 현재 최고 판에 하는 법'·채택 판정·교체 후보·트리 대상 판)·단계 4·5·6·8·추가 판 ①②·
최종 모델 고르기, §1-2(동점·채택), §1-3(판 이름·test 막기), §3(반복 수만 다시), §7-3(블록 OOF·엄격 OOF),
부록 A-6(Theil's U·단계 4~6 범주 숫자 인코딩)·A-8(② 60칸·VAE 입력)·A-9·A-12·A-14·A-15, B-1(디스크 캐시 금지).

이 모듈이 하는 일
- 사슬 상태(현재 최고 판·트리 대상 판의 구성)를 읽고 쓰기(`load_state`·`save_state`) — 형식은 아래 '사슬 상태'.
- 판 구성 → 실행 판 설정(`derive_specs`): 트리 판 = 트리 부분을 줄인·뽑은 자료로, 딥러닝 단독·결합 = 그 모델을,
  앙상블 = 구성마다(+ 평균 판 `write_ensemble`), ④ = 로지스틱 메타 판 + 6 트리 블록 OOF(`Stack4Provider`),
  이상 점수 칼럼이 있으면 VAE 값 모델을 그 자료로 다시 학습(`VAEProvider` — 엄격 OOF·전역 블록 OOF, reuse=False).
- train 행 고르기(`RowPlan`): 줄이기(downsize — Theil's U 1위 칼럼의 검출 0건 값 행 빼기)·5/6 뽑기(`hon_sample`,
  씨앗 0)·상위 5 품목(4-c) + h 붙이기(자기 h / 4-a = 자기 h + 9개(m = 10) / ① = h 1). validation·test 쌍은 늘 전부.
- 단계 4~6 범주 숫자 인코딩(`CatEncoder` — 부록 A-6, 범주 5·season·aw_label → 스무딩 검출률).
- ② 60칸 신경망(`NN60Inputs`·`NN60Model`).
- 판정(`judge_step`): (가) 쌓는 후보 val 1위 → (나) 교체 후보 val 1위, 4자리 반올림(§1-2), 사슬 기록 한 줄.

runner 연결(구현 세부 — runner·stage_cli 를 고치지 않고 이 모듈에서 바꿔 끼움, runner 담당에게 보고함)
 (R1) train 행: runner 는 `folds.make_train_rows`(시료당 h 1개)만 부름 → `install_train_rows(plan)` 가 그 프로세스의
      `runner._train_rows` 를 RowPlan 대로 바꿈(같은 묶음 판은 같은 extra → 같은 RowPlan).
 (R2) 평가 쌍 거르기(① = h 1 쌍만, 4-c = 상위 5 품목 시료 쌍만): `install_pair_filter` 가 `folds.iter_scoring_pairs_by_h`
      를 그 프로세스에서 바꿈(runner·evaluate·tuning 모두 같은 함수를 부르므로 학습·반복 수 곡선·채점·보고가 같은 쌍).
 (R3) 판 이름 약어 'T3'(앙상블 기록 판)·'NN60'(②) 를 `runner.ALG_ABBR` 에 더함(VariantSpec 검사용).
 (R4) 5·6 씨앗 0: stage_cli 가 설정 seed 를 42 로 묶으므로 모델 객체를 `SeedOverride` 로 감싸 fit·학습률 읽기 씨앗만
      0 으로(판 설정 extra.s4.model_seed = 0 에 기록, MLflow param 'seed' 는 42 로 남는 점 표 주석에).

사슬 상태(state json — 3-3 스크립트가 이 꼴로 쓰고(chain_state_3-3), 3-0b(scripts/s3_0b)가 끝 판으로 바꿔 씀
(chain_state_3-0b — 단계 4 가 읽음), 단계 4·5·6·8·최종이 이어 씀)
  {"step": "<마지막 판정 단계>", "current_best": <BestDef>, "tree_target": <BestDef>,
   "tree_top3": [3-0 트리 상위 3 이름…], "tree_rank_3_0b": {알고리즘: 3-0 CV 순위(1~)},   # ②·4-c-L 의 '3-0 CV 1위 트리'
   "history": [사슬 기록 줄…]}
  BestDef = {"name": 판 이름, "kind": "tree"|"dl"|"ensemble"|"stack4"|"nn60", "dir": --inputs 아래 판 폴더(예측·점수),
             "replacement": 교체 후보 여부, "members": [MemberDef…]}   # tree·dl = 1, ensemble = 구성(트리), stack4·nn60 = 1
  MemberDef = {"name": 판 이름, "dir": --inputs 아래 판 폴더(spec.json·n_iter.json), "anomaly": null | AnomalyDef,
               "stack4": null | {"version": LIMS 입력 version(include_ids [])},   # ④ 메타 판
               "row_plan": null | RowPlan.to_dict(),    # 단계 4~6 판(4-a 자료 등) — 그 위 판(5·6 쌓기·8)이 그대로 씀
               "catenc_k": null | k}                    # 단계 4~6 범주 숫자 인코딩 판
  AnomalyDef = {"column": 칼럼 이름, "version": 3-1 VAE 입력 version(트리 대상 판의 현재 입력), "W": W, "mode": 입력 방식,
                "source": 3-1 이상 점수 판 이름}

구현 세부(계획서에 없음 — 담당자 확인):
 (a) 단계 4~6 범주 숫자 인코딩 칼럼 이름 = 'catenc_<칼럼>'(history 구현 세부 a), 원 범주 칼럼 자리에 넣음, float32.
     season·aw_label 의 모집단 값 = 시료 d 의 계절·식품 고정표 aw_label(null → '결측').
 (b) 5·6 뽑기: 해마다(d 의 달력 연도, 오름차순) 한 생성기 `np.random.default_rng(0)` 에서 음성 풀(키 정렬 순)을
     `choice(replace=False)` 로 min(4 × 양성 수, 풀 크기) 개. 양성 0 인 해 = 음성 0. 6 의 칸 = (3단계 RR3 키, sgg).
 (c) 블록 값 모델(VAE·④ 트리)의 학습 시료 = 그 모델 학습 블록 시료(줄이기·뽑기 전) 안에서 같은 단계(줄이기 → 뽑기 →
     상위 5)를 다시 함(줄이기 칼럼·상위 5 품목 = 그 fold(refit) 것). 뽑기 씨앗도 0(모델마다 새 생성기).
 (d) ④ 의 블록 트리(트리 부분)는 4-a 판에서 학습 행에도 h 10개(RowPlan 그대로), VAE 학습 행은 시료당 자기 h 1개.
 (e) 4-a 의 m = 10 고정(공통 규칙 11-08 — 메모리 판정·디스크 기반 학습 안 씀).
 (f) 이상 점수·④ 값은 train 행(d ≤ train 끝)이면 엄격·전역 블록 OOF, 평가 쌍(d > train 끝)이면 판 train 전체 모델 —
     행 d 로 가름. 값 모델은 프로세스 안 메모리에만(디스크 캐시 없음 — B-1), 판 폴더 inputs_<split>/value_models/ 에 저장.
 (g) 앙상블 기록 판(T3): 구성 판 예측 단순 평균(CV = 구성 교차 N 예측 평균, threshold = 공통 N 예측 평균(없으면 CV 예측),
     test = refit 예측 평균) — 학습 횟수 0. 구성 판은 '<앙상블 판>--<알고리즘>' 이름으로 따로 돌림(CV 채점 기록도 남음).
 (h) ② 입력: 행 표의 d·h 를 (T + 1, 1) 로 바꿔 트리 입력을 만들면 E = R = T + 1·window [T − W + 2, T + 1]·g = 0·C 그대로
     (§4-1 정의 그대로 나옴), season 만 T 의 달로 덮어씀. 숫자 = TabularScaler(표준화·결측 0·결측 표시·해외 구조적 결측),
     src_route = 0/1, season·aw_label = 원핫(고정 값 목록), x_cat = sido·sgg·p2·p3 번호(DLPreprocessor 와 같은 번호표 —
     위치 + 1, 0 = OOV). 출력 60칸 중 그 쌍의 h 칸(x_h)이 예측·손실.
 (i) ① 입력 = 그 판 입력에서 h·g 칼럼을 뺌(트리), 딥러닝 판은 g ÷ 50 = 0 그대로 둠(학습 행 h = 1 → g = 0).
 (j) 4-c 의 '코드없음' 3단계(RR3 키 '<p2>|코드없음')는 상위 5 고르기에서 뺌(§2-2 RR3 '4-c 에선 뺌').
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cf
from . import evaluate as ev
from . import folds as fd
from . import history as hi
from . import metrics as mt
from . import registry as rg
from . import runner as rn
from . import scope as scope_
from .hassign import H_MAX
from .paths import resolve

ORDER = "2차"
EXPERIMENT = "listeria-2차"
STAGE4, STAGE56, STAGE8 = "4", "5-6", "8"
ROLE_CAND, ROLE_REPL, ROLE_CMP = "후보", "교체 후보", "비교만"
NA_JUDGE = "해당 없음"
TREE6 = ("XGB", "LGB", "CAT", "RF", "ET", "LR")
TREE_ALGO = {"XGB": "XGB", "LGB": "LGBM", "CAT": "CAT", "RF": "RF", "ET": "ET", "LR": "LR"}
BOOST = ("XGB", "LGB", "CAT")
M_REP = 10                                  # 구현 세부 (e)
HON_RATIO = 4                               # §6 단계 5·6: 음성 = 4 × 양성
HON_SEED = 0                                # §8 씨앗: 단계 5·6 = 0
SELECT_STEPS = ("reduce", "hon5", "hon6", "top5")
REP_MODES = ("own", "4-a", "x1")
BEST_KINDS = ("tree", "dl", "ensemble", "stack4", "nn60")
CATENC_COLS = tuple(rg.CAT5) + ("season", "aw_label")
CATENC_PREFIX = "catenc_"
TOP5_N = 5
STACK4_COLS = tuple(f"l6_{a}" for a in TREE6)
ENS_SEP = "--"
SELECTION_FILE = "selection.json"
STATE_KEYS = ("step", "current_best", "tree_target", "tree_top3", "tree_rank_3_0b", "history")
EXTRA_ALGS = ("T3", "NN60")


class StageError(RuntimeError):
    pass


# ================================================================ (R3) 판 이름 약어
def ensure_runner_algs() -> None:
    missing = tuple(a for a in EXTRA_ALGS if a not in rn.ALG_ABBR)
    if missing:
        rn.ALG_ABBR = tuple(rn.ALG_ABBR) + missing


ensure_runner_algs()


# ================================================================ train 행 고르기(RowPlan)
@dataclass(frozen=True)
class RowPlan:
    """steps = 고르기 단계(차례대로: 'reduce'·'hon5'·'hon6'·'top5' — 빈 = 고르지 않음), rep = h 붙이기
    ('own' 자기 h · '4-a' 자기 h + 9 · 'x1' h 1), sel_dir = 고른 키 폴더(--inputs 아래, steps 가 있을 때)."""
    steps: tuple = ()
    rep: str = "own"
    sel_dir: str | None = None

    def __post_init__(self):
        bad = [s for s in self.steps if s not in SELECT_STEPS]
        if bad:
            raise ValueError(f"고르기 단계: {bad}")
        if self.rep not in REP_MODES:
            raise ValueError(f"rep: {self.rep}")
        if bool(self.steps) != bool(self.sel_dir):
            raise ValueError("steps 와 sel_dir 는 함께")
        if sum(s in ("hon5", "hon6") for s in self.steps) > 1:
            raise ValueError("5·6 뽑기는 하나만")

    def to_dict(self) -> dict:
        return {"steps": list(self.steps), "rep": self.rep, "sel_dir": self.sel_dir}

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "RowPlan":
        if not d:
            return cls()
        bad = set(d) - {"steps", "rep", "sel_dir"}
        if bad:
            raise ValueError(f"RowPlan 모르는 칸: {sorted(bad)}")
        return cls(tuple(d.get("steps", ())), d.get("rep", "own"), d.get("sel_dir"))

    def with_rep(self, rep: str) -> "RowPlan":
        return dataclasses.replace(self, rep=rep)


def replicate(rows: pd.DataFrame, rep: str) -> pd.DataFrame:
    from . import downsize as ds

    if rep == "own":
        return rows
    if rep == "4-a":
        return _replicate_m(rows, M_REP)
    return ds.replicate_h(rows, "x1")


def m_h_values(key: str, h: int, m: int = M_REP) -> np.ndarray:
    """§6 단계 4: 자기 h + `default_rng(md5(f'{key}|m42')[:8]).choice([x ≠ h], m − 1, replace=False)`(오름차순)."""
    import hashlib

    h = int(h)
    rng = np.random.default_rng(int.from_bytes(hashlib.md5(f"{key}|m42".encode("utf-8")).digest()[:8], "big"))
    extra = rng.choice([x for x in range(1, H_MAX + 1) if x != h], m - 1, replace=False)
    return np.sort(np.concatenate([[h], np.asarray(extra, dtype=np.int64)])).astype(np.int64)


def _replicate_m(rows: pd.DataFrame, m: int) -> pd.DataFrame:
    """downsize.replicate_h 와 같은 꼴(행 순서 = 원 순서, 시료 안 h 오름차순) — m = 10(downsize.M_LADDER 밖이라 여기서)."""
    from .hassign import assign_h_many

    if any(c in rows.columns for c in ("E", "S", "g", "R")):
        raise ValueError("E·S·g·R 이 이미 있음 — 복제 뒤 runner.add_bounds 로")
    keys = rows["lims_data_key"].astype(str).to_numpy(dtype=object)
    if pd.Index(keys).has_duplicates:
        raise ValueError("입력은 시료당 1행")
    h_own = assign_h_many(keys.tolist())
    if "h" in rows.columns and not np.array_equal(rows["h"].to_numpy(dtype=np.int64), h_own):
        raise ValueError("입력 h 가 자기 h(부록 A-1)와 다름")
    rep = np.repeat(np.arange(len(rows)), m)
    hs = (np.concatenate([m_h_values(k, h, m) for k, h in zip(keys, h_own)]) if len(rows)
          else np.empty(0, np.int64))
    out = rows.iloc[rep].copy()
    d = fd.day_ints(out["d"])
    out["h"] = hs.astype(np.int16)
    out["T"] = (d - hs).astype("datetime64[D]")
    return out.reset_index(drop=True)


def load_kept(inputs_root: Path, sel_dir: str, split: str) -> np.ndarray:
    p = resolve(inputs_root, sel_dir) / f"kept_keys_{split}.parquet"
    if not p.exists():
        raise StageError(f"고른 키 파일 없음(prep 먼저): {p}")
    return pd.read_parquet(p)["lims_data_key"].astype(str).to_numpy(dtype=object)


def plan_train_rows(known: pd.DataFrame, split: str, plan: RowPlan, inputs_root: Path) -> pd.DataFrame:
    """그 판 train 행 = make_train_rows(시료당 자기 h) → 고른 시료만 → h 붙이기. E·S·g·R 은 부르는 쪽이 add_bounds."""
    rows = fd.make_train_rows(known, fd.SPLITS[split])
    if plan.steps:
        kept = load_kept(inputs_root, plan.sel_dir, split)
        m = pd.Index(kept).get_indexer(pd.Index(rows["lims_data_key"].astype(str).to_numpy(dtype=object))) >= 0
        if int(m.sum()) != len(kept):
            raise StageError(f"{split}: 고른 키 {len(kept)} 중 train 행에 있는 것 {int(m.sum())}")
        rows = rows.loc[m]
    return replicate(rows.reset_index(drop=True), plan.rep)


# ---------------------------------------------------------------- (R1) runner train 행 바꿔 끼우기
_ORIG_TRAIN_ROWS = rn._train_rows


def install_train_rows(plan: RowPlan, inputs_root: Path) -> None:
    """이 프로세스의 runner train 행을 RowPlan 대로(같은 묶음 판 = 같은 RowPlan). 연기 시험 줄이기는 runner 와 같게."""
    inputs_root = Path(inputs_root)

    def _rows(ctx, split, W, smoke, known):
        rows = plan_train_rows(known, split, plan, inputs_root)
        if smoke is not None:
            rows = rn._smoke_train(rows, smoke["max_train_rows"], ctx.seed)
        return rn.add_bounds(rows, W)

    rn._train_rows = _rows


def reset_train_rows() -> None:
    rn._train_rows = _ORIG_TRAIN_ROWS


# ---------------------------------------------------------------- (R2) 평가 쌍 거르기
_ORIG_PAIRS = fd.iter_scoring_pairs_by_h


def install_pair_filter(fn: Callable[[int, np.ndarray, fd.Split], np.ndarray]) -> None:
    """fn(h, idx, split) → 남길 idx(정렬 유지). 모든 h 를 그대로 돌되 거른 쌍만 냄."""
    def _it(d_days, split):
        for h, idx in _ORIG_PAIRS(d_days, split):
            yield h, np.asarray(fn(h, idx, split), dtype=np.int64)

    fd.iter_scoring_pairs_by_h = _it


def reset_pair_filter() -> None:
    fd.iter_scoring_pairs_by_h = _ORIG_PAIRS


def h1_filter(h, idx, split):
    """① 당일 예측: h = 1 쌍만."""
    return idx if h == 1 else idx[:0]


def top5_filter(samples: ev.SampleTable, top5: Mapping[str, Sequence[str]]):
    """4-c·4-c-L: 그 판(fold·refit)의 상위 5 품목(RR3 키) 시료 쌍만. top5 = {판 이름: [p3 키…]}."""
    p3k = hi.p3_key(samples.df["p2"], samples.df["p3"])

    def f(h, idx, split):
        keep = set(top5[split.name])
        return idx[np.fromiter((p3k[i] in keep for i in idx), bool, idx.size)]
    return f


@contextlib.contextmanager
def hooks(*, plan: RowPlan | None = None, inputs_root: Path | None = None, pair_filter=None):
    """시험용: 바꿔 끼운 뒤 원래대로."""
    try:
        if plan is not None:
            install_train_rows(plan, inputs_root)
        if pair_filter is not None:
            install_pair_filter(pair_filter)
        yield
    finally:
        reset_train_rows()
        reset_pair_filter()


# ================================================================ 고르기 계산(prep — 단계 시작 때 한 번)
def hon_sample(keys, d_days, y, cells, kind: str, seed: int = HON_SEED) -> tuple[np.ndarray, pd.DataFrame]:
    """§6 단계 5·6(구현 세부 b): 해마다 양성 전부 + 음성 min(4 × 양성 수, 풀). kind '5' = 그해 음성 전체,
    '6' = 그해 양성이 나온 칸(cells 값)의 그해 음성. 돌려줌 = (남길 bool, 해별 표)."""
    if kind not in ("5", "6"):
        raise ValueError("kind 는 '5'·'6'")
    keys = np.asarray(keys, dtype=object)
    d = np.asarray(d_days, dtype=np.int64)
    yy = np.asarray(y, dtype=np.int64)
    cells = np.asarray(cells, dtype=object)
    n = keys.shape[0]
    if not (d.shape[0] == yy.shape[0] == cells.shape[0] == n):
        raise ValueError("길이 다름")
    year = d.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970
    keep = yy == 1
    rng = np.random.default_rng(int(seed))
    rows = []
    for yr in np.unique(year):
        iy = np.flatnonzero(year == yr)
        pos = iy[yy[iy] == 1]
        neg = iy[yy[iy] == 0]
        if kind == "6":
            pc = set(cells[pos].tolist())
            neg = neg[np.fromiter((c in pc for c in cells[neg]), bool, neg.size)]
        neg = neg[np.argsort(keys[neg].astype(str), kind="stable")]
        want = HON_RATIO * pos.size
        k = min(want, neg.size)
        pick = rng.choice(neg, k, replace=False) if k else np.empty(0, np.int64)
        keep[pick] = True
        rows.append({"year": int(yr), "n_pos": int(pos.size), "pool_neg": int(neg.size), "want_neg": int(want),
                     "drawn_neg": int(k), "short": bool(k < want)})
    return keep, pd.DataFrame(rows)


def top5_codes(p3key, y) -> list[str]:
    """fold train 검출 비중 상위 5 3단계(RR3 키 — '코드없음' 칸 뺌). 동점 = 검사 수 많은 쪽 → 코드 순(§1-2)."""
    s = pd.DataFrame({"p": np.asarray(p3key, dtype=object), "y": np.asarray(y, dtype=np.int64)})
    s = s[~s["p"].astype(str).str.endswith(f"|{hi.NO_CODE}")]
    g = s.groupby("p")["y"].agg(["sum", "count"]).reset_index()
    g = g[g["sum"] > 0].sort_values(["sum", "count", "p"], ascending=[False, False, True], kind="stable")
    if len(g) < TOP5_N:
        raise StageError(f"검출 있는 품목이 {len(g)}개 — 상위 5 못 고름")
    return g["p"].head(TOP5_N).astype(str).tolist()


def _pool(samples: ev.SampleTable) -> pd.DataFrame:
    k = samples.known_frame().reset_index(drop=True)
    k["dd"] = fd.day_ints(k["d"])
    k["p3k"] = hi.p3_key(k["p2"], k["p3"])
    k["cell"] = hi.unit_key(k["p3k"], k["sgg"].astype(str).to_numpy(dtype=object))
    return k


def select_rows(pool: pd.DataFrame, steps: Sequence[str], *, red_col: Mapping[str, str] | None, red_values=None,
                split: str, top5: Mapping[str, Sequence[str]] | None = None, seed: int = HON_SEED,
                base_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """pool(시료 단위, _pool 꼴) 중 base_mask(없으면 그 판 train 시료) 안에서 steps 를 차례로 → 남길 bool·기록.
    줄이기 = red_values[red_col[split]](pool 순서 값) 의 검출 0건 값 행 빼기(downsize.zero_detection_keep 와 같음)."""
    from . import downsize as ds

    m = (pool["dd"].to_numpy() <= fd.SPLITS[split].train_end) if base_mask is None else base_mask.copy()
    rec: dict = {"n_start": int(m.sum()), "pos_start": int(pool["y"].to_numpy()[m].sum()), "steps": []}
    y = pool["y"].to_numpy(dtype=np.int64)
    for st in steps:
        idx = np.flatnonzero(m)
        if st == "reduce":
            col = red_col[split]
            keep, vt = ds.zero_detection_keep(red_values[col].to_numpy()[idx], y[idx])
            info = {"step": st, "column": col, "n_values": int(len(vt)), "n_values_zero_pos": int((vt["pos"] == 0).sum())}
        elif st in ("hon5", "hon6"):
            keep, tab = hon_sample(pool["lims_data_key"].to_numpy()[idx], pool["dd"].to_numpy()[idx], y[idx],
                                   pool["cell"].to_numpy()[idx], st[-1], seed)
            info = {"step": st, "by_year": tab.to_dict(orient="records")}
        else:
            codes = set(top5[split])
            keep = np.fromiter((c in codes for c in pool["p3k"].to_numpy()[idx]), bool, idx.size)
            info = {"step": st, "top5": list(top5[split])}
        nm = np.zeros_like(m)
        nm[idx[keep]] = True
        if int(y[nm].sum()) != int(y[m].sum()) and st != "top5":
            raise AssertionError(f"{split} {st}: 검출 행이 빠짐")
        m = nm
        info.update(n_after=int(m.sum()), pos_after=int(y[m].sum()))
        rec["steps"].append(info)
    return m, rec


def prepare_selection(samples: ev.SampleTable, out_dir: Path, steps: Sequence[str], *, data_root: Path | None = None,
                      theil_targets: Path | None = None, base_sel: Path | None = None) -> dict:
    """판(fold1~3·refit)마다 고른 키 kept_keys_<split>.parquet + selection.json(+ 줄이기 Theil's U 표·상위 5 목록).
    base_sel = 이미 고른 폴더(5·6 쌓기 = 단계 4 줄인 폴더) — 그 키 안에서 steps 를 더 함(기록에 base 단계 포함).
    줄이기: Theil's U 표·1위 칼럼 = downsize.plan_reductions(줄이기 전 판 train), 같은 판 키로 저장."""
    from . import downsize as ds

    out_dir = Path(out_dir)
    if (out_dir / SELECTION_FILE).exists():
        return rn.read_json(out_dir / SELECTION_FILE)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = _pool(samples)
    base_info = rn.read_json(base_sel / SELECTION_FILE) if base_sel is not None else None
    all_steps = list(base_info["steps"]) + list(steps) if base_info else list(steps)
    if base_info and any(st in base_info["steps"] for st in steps):
        raise ValueError(f"이미 한 단계를 또: {steps} ⊂ {base_info['steps']}")
    red_col = dict(base_info["reduce_column"]) if base_info and base_info.get("reduce_column") else None
    top5 = dict(base_info["top5"]) if base_info and base_info.get("top5") else None
    red_values = None
    if "reduce" in steps:
        if data_root is None or theil_targets is None:
            raise ValueError("줄이기에는 data_root·theil_targets 필요")
        cols = ds.load_theil_targets(theil_targets)
        data = ds.ReductionData.load(samples.known_frame(), resolve(data_root, ds.SAMPLE_FILE_REL), cols)
        if list(data.keys) != list(pool["lims_data_key"].astype(str)):
            raise AssertionError("줄이기 자료 키 순서가 시료 표와 다름")
        reds = ds.plan_reductions(data)
        red_col = ds.column_by_split(reds)
        for s, r in reds.items():
            r.table.to_csv(out_dir / f"theil_{s}.csv", index=False, encoding="utf-8-sig")
            cf.dump_json(out_dir / f"reduction_{s}.json", cf.nan_to_null(r.summary()))
        red_values = data.values
    if "top5" in steps:
        top5 = {s: top5_codes(pool["p3k"].to_numpy()[pool["dd"].to_numpy() <= fd.SPLITS[s].train_end],
                              pool["y"].to_numpy()[pool["dd"].to_numpy() <= fd.SPLITS[s].train_end])
                for s in rn.SPLIT_ORDER}
    recs = {}
    for s in rn.SPLIT_ORDER:
        base_mask = None
        if base_info is not None:
            kept = pd.read_parquet(Path(base_sel) / f"kept_keys_{s}.parquet")["lims_data_key"].astype(str)
            base_mask = pd.Index(kept).get_indexer(pd.Index(pool["lims_data_key"].astype(str))) >= 0
            base_mask = np.asarray(base_mask, dtype=bool)
        m, rec = select_rows(pool, steps, red_col=red_col, red_values=red_values, split=s, top5=top5,
                             base_mask=base_mask)
        p = out_dir / f"kept_keys_{s}.parquet"
        if p.exists():
            raise FileExistsError(p)
        pd.DataFrame({"lims_data_key": pool["lims_data_key"].astype(str).to_numpy()[m]}).to_parquet(p, index=False)
        recs[s] = rec
    info = {"steps": all_steps, "own_steps": list(steps), "base_sel": None if base_sel is None else str(base_sel.name),
            "reduce_column": red_col, "top5": top5, "seed": HON_SEED, "per_split": recs,
            "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    cf.dump_json(out_dir / SELECTION_FILE, cf.nan_to_null(info))
    return info


class BlockSelector:
    """블록 값 모델(VAE·④ 트리)의 학습 시료(구현 세부 c): 그 모델 학습 블록 시료 안에서 RowPlan 의 단계를 다시."""

    def __init__(self, samples: ev.SampleTable, plan: RowPlan, inputs_root: Path, data_root: Path | None,
                 theil_targets: Path | None):
        self.plan = plan
        self.pool = _pool(samples)
        self.info = rn.read_json(resolve(inputs_root, plan.sel_dir) / SELECTION_FILE) if plan.steps else None
        self.values = None
        if self.info and "reduce" in self.info["steps"]:
            from . import downsize as ds

            cols = ds.load_theil_targets(theil_targets)
            self.values = ds.ReductionData.load(samples.known_frame(), resolve(data_root, ds.SAMPLE_FILE_REL),
                                                cols).values

    def keys_for(self, model, seed: int = HON_SEED) -> np.ndarray:
        """model = blocks.BlockModel(reuse=False) → 학습 시료 키."""
        bm = model.train_mask(self.pool["dd"].to_numpy())
        if not self.plan.steps:
            return self.pool["lims_data_key"].astype(str).to_numpy(dtype=object)[bm]
        split = model.owner.split
        m, _ = select_rows(self.pool, self.info["steps"], red_col=self.info.get("reduce_column"),
                           red_values=self.values, split=split, top5=self.info.get("top5"), seed=seed, base_mask=bm)
        return self.pool["lims_data_key"].astype(str).to_numpy(dtype=object)[m]


# ================================================================ 단계 4~6 범주 숫자 인코딩(부록 A-6)
class CatEncoder:
    """판(split) 하나: 줄이기·뽑기 전 그 판 train 시료(d ≤ train 끝 — 시료 1개 = 1번)로 모집단, C 규칙(§5-1),
    범주 값 [2015-04-07, C] 검출률 (s + k·p0) ÷ (n + k)(history.category_rate)."""

    def __init__(self, ctx, split, k: int):
        from . import static_feats as sf

        self.split = fd.get_split(split) if isinstance(split, str) else split
        self.k = int(k)
        s = ctx.samples
        m = fd.day_ints(s["d"]) <= self.split.train_end
        sub = s.loc[m, ["lims_data_key", "y", "d", "p2", "p3", "sgg", "sido", "src_route"]].reset_index(drop=True)
        self.pop = hi.Population.from_samples(sub)
        self.pop.add_level("season", sf.season(sub["d"]))
        if ctx.food is not None:
            lab = sf.food_features(ctx.food, sub["p3"].astype(str).to_numpy(dtype=object))["aw_label"].to_numpy(dtype=object)
            self.pop.add_level("aw_label", np.where(pd.isna(lab), rg.AW_MISSING, lab).astype(object))

    def encode(self, X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in CATENC_COLS if c in X.columns]
        if not cols:
            return X
        T = fd.day_ints(meta["d"]) - meta["h"].to_numpy(dtype=np.int64)
        rows = pd.DataFrame({"d": meta["d"].to_numpy(), "C": hi.c_from_T(T).astype("datetime64[D]"),
                             **{c: meta[c].astype(str).to_numpy(dtype=object) for c in ("p2", "p3", "sgg", "sido",
                                                                                         "src_route")}})
        R = hi.prepare_rows(rows)
        enc = {}
        for c in cols:
            vals = R.keys[c] if c in rg.CAT5 else X[c].to_numpy(dtype=object)
            if c not in self.pop.keys:
                raise StageError(f"모집단에 {c} 수준 없음(식품 고정표 필요)")
            enc[c] = hi.category_rate(self.pop, c, vals, R, self.k).astype(np.float32)
        # 제자리 바꿈(큰 X 를 복사하지 않음 — 4-a 는 행 10배): X 는 조립 객체가 이번 호출에 새로 만든 표
        for c, v in enc.items():
            X[c] = v
        X.rename(columns={c: CATENC_PREFIX + c for c in enc}, inplace=True)
        return X


# ================================================================ 블록 OOF 값 모델(VAE·④)
class _OOFProvider:
    """판(split)마다 block_plan(reuse=False) 의 블록 모델 + 판 train 전체 모델을 그 자료로 학습(메모리에만),
    values(meta) = train 행(d ≤ train 끝)은 블록 OOF, 평가 쌍은 전체 모델(구현 세부 f)."""
    columns: tuple = ()

    def __init__(self, w: "Stage4Wiring", pdef: Mapping, plan: RowPlan, seed: int):
        self.w, self.pdef, self.plan, self.seed = w, dict(pdef), plan, int(seed)
        self.fitted: dict[str, dict[int, Any]] = {}
        self.records: dict[str, list] = {}

    def ensure(self, split: str) -> None:
        from . import blocks as bk

        if split in self.fitted:
            return
        plan = bk.block_plan(reuse=False, splits=[split])
        self._plan = plan
        sel = self.w.block_selector(self.plan)
        known = self.w.samples.known_frame()
        out, rec = {}, []
        for i in bk.models_for_split(plan, split):
            bmod = plan[i]
            keys = sel.keys_for(bmod, seed=self.seed)
            rows = bk.model_train_rows(bmod, known, keep_keys=keys)
            out[i] = self.fit_block(rows.reset_index(drop=True), split)
            rec.append({"model": bmod.name, "n": int(len(rows)), "pos": int(rows["y"].sum()),
                        "train_d": bmod.intervals_as_dates()})
            self.w.check()
        self.fitted[split] = {"plan": plan, "models": out}
        self.records[split] = rec

    def values(self, meta: pd.DataFrame, split: str) -> np.ndarray:
        from . import blocks as bk

        self.ensure(split)
        st = self.fitted[split]
        plan, models = st["plan"], st["models"]
        d = fd.day_ints(meta["d"])
        tr = d <= fd.SPLITS[split].train_end
        out = np.full((len(meta), len(self.columns)), np.nan)
        if tr.any():
            who = bk.assemble_oof(plan, split, d[tr])
            pos = np.flatnonzero(tr)
            for i in np.unique(who):
                sel = pos[who == i]
                out[sel] = self.predict_block(models[int(i)], meta.iloc[sel].reset_index(drop=True), split)
        if (~tr).any():
            full = plan.index(bk.full_model(plan, split))
            sel = np.flatnonzero(~tr)
            out[sel] = self.predict_block(models[full], meta.iloc[sel].reset_index(drop=True), split)
        if not np.isfinite(out).all():
            raise StageError("값 모델 출력에 NaN")
        return out

    def save(self, path: Path, split: str) -> None:
        p = Path(path) / "value_models"
        p.mkdir(parents=True, exist_ok=False)
        cf.dump_json(p / "plan.json", {"def": self.pdef, "row_plan": self.plan.to_dict(), "seed": self.seed,
                                       "models": self.records.get(split, [])})
        for i, m in self.fitted.get(split, {}).get("models", {}).items():
            self.save_block(m, p / f"model_{i:02d}")

    # 하위 클래스
    def fit_block(self, rows: pd.DataFrame, split: str):
        raise NotImplementedError

    def predict_block(self, m, meta: pd.DataFrame, split: str) -> np.ndarray:
        raise NotImplementedError

    def save_block(self, m, path: Path) -> None:
        raise NotImplementedError


class VAEProvider(_OOFProvider):
    """3-1 이상 점수(VAE 표 입력) 값 모델을 그 판 자료로 다시 학습(부록 A-8: 입력 = 3-1 때 트리 대상 판 현재 입력의 숫자
    칼럼 − h·g, 표준화·결측 0 + 결측 표시·해외 구조적 결측, train 불검출 행, 원 점수 음의 ELBO → robust z)."""

    def __init__(self, w, pdef, plan, seed):
        super().__init__(w, pdef, plan, seed)
        self.columns = (pdef["column"],)
        self.vs = rg.load_version({"name": f"vae-{pdef['source']}", "order": 2, "W": int(pdef["W"]),
                                   "mode": pdef["mode"], **pdef["version"]})
        self.num_cols = [c for c in rg.tree_columns(self.vs) if c.kind == "numeric"]

    def _X(self, meta, split):
        from . import inputs as ip

        X = ip.build_tree_X(rn.add_bounds(meta[[c for c in meta.columns if c not in ("E", "S", "g", "R")]],
                                          self.vs.W), self.w.ctx, self.vs, split=fd.SPLITS[split])
        return X[[c.name for c in self.num_cols]], meta["src_route"].astype(str).to_numpy(dtype=object)

    def fit_block(self, rows, split):
        from . import inputs as ip
        from . import models_dl as md

        X, route = self._X(rows, split)
        sc = ip.TabularScaler(self.num_cols, self.w.ctx.missing, 2).fit(X)
        xt = sc.transform(X, src_route=route).to_numpy(dtype=np.float32)
        m = md.AnomalyModel("VAE", "tab", device=self.w.device, **self.w.dl_kwargs)
        m.fit({"x_tab": xt}, rows["y"].to_numpy(dtype=np.int64), seed=self.seed, n_jobs=self.w.n_jobs,
              groups=(rows["sgg"].astype(str).to_numpy(), fd.day_ints(rows["d"])))
        return {"scaler": sc, "model": m}

    def predict_block(self, m, meta, split):
        X, route = self._X(meta, split)
        xt = m["scaler"].transform(X, src_route=route).to_numpy(dtype=np.float32)
        return m["model"].score({"x_tab": xt})[:, None]

    def save_block(self, m, path):
        path.mkdir(parents=True)
        m["model"].save(path / "vae.pt")
        cf.dump_json(path / "scaler.json", cf.nan_to_null(m["scaler"].state()))


class Stack4Provider(_OOFProvider):
    """④ LIMS 만 스태킹의 트리 6종(기본 설정·라이브러리 기본 반복 수, 범주 5 + h·g — 단계 4~6 은 범주 숫자 인코딩)을
    블록마다 다시 학습 → 확률 6칼럼(메타 로지스틱 입력). 학습 행 h = RowPlan(구현 세부 d)."""
    columns = STACK4_COLS

    def __init__(self, w, pdef, plan, seed):
        super().__init__(w, pdef, plan, seed)
        self.vs = rg.load_version({"name": "x4-L6", "order": 2, "W": int(pdef["W"]), "mode": pdef["mode"],
                                   **pdef["version"]})
        self.catenc_k = pdef.get("catenc_k")
        self.params = dict(pdef.get("params") or {})      # 단계 8 튜닝 설정(트리마다) — 없으면 기본

    def _asm(self, split, alg):
        return S4TreeInputs(self.w, self.vs, split, catenc_k=self.catenc_k, scale=(alg == "LR"))

    def fit_block(self, rows, split):
        from . import models_tree as mtr

        rows = rn.add_bounds(replicate(rows, self.plan.rep), self.vs.W)
        y = rows["y"].to_numpy(dtype=np.int64)
        out = {}
        for a in TREE6:
            asm = self._asm(split, a).fit(rows)
            X = asm.transform(rows)
            m = mtr.make_model(TREE_ALGO[a], cat_cols=asm.cat_cols,
                               indicator_cols=asm.indicator_cols if a == "LR" else (),
                               max_iter=None, params=self.params.get(a) or None)
            m.fit(X, y, seed=self.seed, n_jobs=self.w.n_jobs)
            out[a] = (asm, m)
        return out

    def predict_block(self, m, meta, split):
        meta = rn.add_bounds(meta[[c for c in meta.columns if c not in ("E", "S", "g", "R")]], self.vs.W)
        return np.column_stack([mm.predict_proba(asm.transform(meta)) for a, (asm, mm) in m.items()])

    def save_block(self, m, path):
        path.mkdir(parents=True)
        for a, (asm, mm) in m.items():
            mm.save(path / f"tree_{a}")


class FrozenVAEProvider(VAEProvider):
    """단계 8(§6 단계 8 '이상 점수 칼럼이 있으면 기존 값을 다시 붙이고 N 다시'): 대상 판이 쓰던 값 모델(판 폴더
    inputs_<split>/value_models — 블록 모델·판 train 전체 모델과 표준화 통계)을 읽어 같은 값을 냄. 학습 없음.
    대상 판도 다시 붙인 판이면(plan.json 의 frozen_from) 원래 학습한 판까지 따라감."""

    def __init__(self, w, pdef, plan, seed):
        super().__init__(w, pdef, plan, seed)
        self.from_dir = pdef["from_dir"]

    def _source(self, split: str) -> tuple[Path, dict]:
        rel = self.from_dir
        for _ in range(20):
            d = resolve(self.w.inputs_root, rel) / f"inputs_{split}" / "value_models"
            if not (d / "plan.json").exists():
                raise StageError(f"대상 판 값 모델 없음(단계 4~6 판 꼴이 아님): {d}")
            pj = rn.read_json(d / "plan.json")
            if not pj.get("frozen_from"):
                return d, pj
            rel = pj["frozen_from"]
        raise StageError("frozen_from 이 너무 깊음")

    def ensure(self, split: str) -> None:
        from . import blocks as bk
        from . import inputs as ip
        from . import models_dl as md

        if split in self.fitted:
            return
        d, pj = self._source(split)
        if pj["def"].get("kind") != "vae" or pj["def"].get("column") != self.columns[0]:
            raise StageError(f"대상 판 값 모델 종류·칼럼이 다름: {pj['def']}")
        plan = bk.block_plan(reuse=False, splits=[split])
        models = {}
        for i in bk.models_for_split(plan, split):
            p = d / f"model_{i:02d}"
            st = rn.read_json(p / "scaler.json")
            sc = ip.TabularScaler(self.num_cols, self.w.ctx.missing, 2)
            f = lambda a: np.array([np.nan if v is None else v for v in a], dtype=np.float64)  # noqa: E731
            sc.columns_in, sc.num, sc.cat, sc.ind, sc.struct = st["columns_in"], st["num"], st["cat"], st["ind"], st["struct"]
            sc.mean, sc.std, sc.zero = f(st["mean"]), f(st["std"]), np.asarray(st["zero"], dtype=bool)
            sc.fitted = True
            models[i] = {"scaler": sc, "model": md.AnomalyModel.load(p / "vae.pt", device=self.w.device)}
        self._plan = plan
        self.fitted[split] = {"plan": plan, "models": models}
        self.records[split] = [{"frozen_from": str(d), "source_def": pj["def"]}]

    def fit_block(self, rows, split):
        raise AssertionError("단계 8 은 기존 이상 점수 값을 다시 붙임 — 값 모델 학습 안 함")

    def save(self, path: Path, split: str) -> None:
        p = Path(path) / "value_models"
        p.mkdir(parents=True, exist_ok=False)
        cf.dump_json(p / "plan.json", {"def": self.pdef, "frozen_from": self.from_dir,
                                       "models": self.records.get(split, [])})


PROVIDERS = {"vae": VAEProvider, "stack4": Stack4Provider, "vae_frozen": FrozenVAEProvider}


# ================================================================ 트리 입력(범주 숫자 인코딩·값 칼럼·h·g 빼기)
class S4TreeInputs:
    """runner InputAssembler(트리): TreeInputs(표준화 없음) → [범주 숫자 인코딩] → [h·g 빼기(①)] → [값 칼럼(이상 점수·
    ④ 6칼럼)] → [로지스틱이면 TabularScaler]. feature_only = ④ 메타(값 칼럼만)."""

    def __init__(self, w: "Stage4Wiring", vs, split, *, catenc_k: int | None = None, scale: bool = False,
                 drop_time: bool = False, provider: _OOFProvider | None = None, feature_only: bool = False):
        from . import inputs as ip

        self.w, self.vs, self.split_name = w, vs, split
        self.base = None if feature_only else ip.TreeInputs(w.ctx, vs, split)
        self.enc = CatEncoder(w.ctx, split, catenc_k) if (catenc_k and not feature_only) else None
        self.scale, self.drop_time, self.provider, self.feature_only = scale, drop_time, provider, feature_only
        self.scaler = None
        self._fit_meta = self._fit_X = None

    @contextlib.contextmanager
    def pair_cache(self):
        with (self.base.pair_cache() if self.base is not None else contextlib.nullcontext()):
            yield

    def _raw(self, meta: pd.DataFrame) -> pd.DataFrame:
        if self.feature_only:
            X = pd.DataFrame(index=pd.RangeIndex(len(meta)))
        else:
            X = self.base.transform(meta)
            if self.enc is not None:
                X = self.enc.encode(X, meta)
            if self.drop_time:
                X.drop(columns=[c for c in ("h", "g") if c in X.columns], inplace=True)
        if self.provider is not None:
            v = self.provider.values(meta, self.split_name)
            for j, c in enumerate(self.provider.columns):
                X[c] = v[:, j].astype(np.float32)
        return X

    def fit(self, train_meta: pd.DataFrame) -> "S4TreeInputs":
        if self.provider is not None:
            self.provider.ensure(self.split_name)
        if self.scale:
            from . import inputs as ip

            X = self._raw(train_meta)
            extra = [c for c in X.columns if c.startswith(CATENC_PREFIX)] + list(
                self.provider.columns if self.provider is not None else ())
            self.scaler = ip.TabularScaler([] if self.feature_only else rg.tree_columns(self.vs),
                                           self.w.ctx.missing, self.vs.order, extra_numeric=extra).fit(X)
            self._fit_meta, self._fit_X = train_meta, X
        return self

    def transform(self, meta: pd.DataFrame) -> pd.DataFrame:
        if self._fit_meta is not None and meta is self._fit_meta:
            X = self._fit_X
            self._fit_meta = self._fit_X = None
        else:
            X = self._raw(meta)
        if self.scaler is not None:
            X = self.scaler.transform(X, src_route=meta["src_route"].astype(str).to_numpy(dtype=object))
        return X

    @property
    def columns(self) -> list[str]:
        if self.scaler is not None:
            s = self.scaler
            return s.num + s.cat + [rg.indicator_name(c) for c in s.ind] + ([rg.STRUCT_COLUMN] if s.struct else [])
        cols = [] if self.feature_only else [c.name for c in rg.tree_columns(self.vs)]
        if self.enc is not None:
            cols = [CATENC_PREFIX + c if c in CATENC_COLS else c for c in cols]
        if self.drop_time:
            cols = [c for c in cols if c not in ("h", "g")]
        return cols + list(self.provider.columns if self.provider is not None else ())

    @property
    def cat_cols(self) -> list[str]:
        if self.enc is not None or self.feature_only:
            return []
        return [c.name for c in rg.tree_columns(self.vs) if c.kind == "category"]

    @property
    def indicator_cols(self) -> list[str]:
        return [c for c in self.columns if c.endswith(rg.NA_SUFFIX) or c == rg.STRUCT_COLUMN]

    def save(self, path) -> None:
        p = Path(path)
        if self.base is not None:
            self.base.save(p)
        else:
            p.mkdir(parents=True, exist_ok=False)
        cf.dump_json(p / "s4_inputs.json", {"catenc_k": None if self.enc is None else self.enc.k,
                                            "drop_time": self.drop_time, "feature_only": self.feature_only,
                                            "scaler": None if self.scaler is None else cf.nan_to_null(self.scaler.state()),
                                            "columns": self.columns})
        if self.provider is not None:
            self.provider.save(p, self.split_name)


# ================================================================ ② 60칸 신경망
class NN60Inputs:
    """② 입력(구현 세부 h): 행 표 → (d, h) = (T + 1, 1) 로 바꾼 트리 입력(h·g 뺌) → 숫자 표준화(TabularScaler)·
    src_route 0/1·season·aw_label 원핫 → x_tab, x_cat = sido·sgg·p2·p3 번호, x_h = 그 쌍의 h(출력 칸)."""

    def __init__(self, w: "Stage4Wiring", vs, split):
        from . import inputs as ip

        self.w, self.vs, self.split = w, vs, split
        self.base = ip.TreeInputs(w.ctx, vs, split)
        self.cols = rg.tree_columns(vs)
        self.num = [c for c in self.cols if c.kind == "numeric"]
        self.scaler = None

    pair_cache = S4TreeInputs.pair_cache

    def _shift(self, meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        h = meta["h"].to_numpy(dtype=np.int64)
        T = fd.day_ints(meta["d"]) - h
        m = meta[[c for c in meta.columns if c not in ("T", "E", "S", "g", "R", "C")]].copy()
        m["d"] = (T + 1).astype("datetime64[D]")
        m["h"] = np.int16(1)
        return rn.add_bounds(m, self.vs.W), h, T

    def _raw(self, meta):
        from . import static_feats as sf

        m, h, T = self._shift(meta)
        X = self.base.transform(m)
        if "season" in X.columns:
            X["season"] = sf.season(T)            # season = T 의 달(부록 A-8 ②)
        return X, h, meta["src_route"].astype(str).to_numpy(dtype=object)

    def fit(self, train_meta):
        from . import inputs as ip

        X, _, route = self._raw(train_meta)
        self.scaler = ip.TabularScaler(self.num, self.w.ctx.missing, self.vs.order).fit(X[[c.name for c in self.num]])
        return self

    def transform(self, meta) -> dict:
        from . import static_feats as sf

        X, h, route = self._raw(meta)
        parts = [self.scaler.transform(X[[c.name for c in self.num]], src_route=route).to_numpy(dtype=np.float32)]
        if not scope_.domestic_only():                     # 국내만 버전: src_route 0/1 없음(lst/scope.py)
            parts.append((route == "overseas").astype(np.float32)[:, None])
        if "season" in X.columns:
            parts.append(_onehot(X["season"].to_numpy(dtype=object), sf.SEASONS))
        if "aw_label" in X.columns:
            parts.append(_onehot(X["aw_label"].to_numpy(dtype=object), rg.AW_LABEL_VALUES))
        voc = self.w.ctx.vocab
        xc = np.column_stack([pd.Index(voc[c]).get_indexer(X[c].astype(str)) + 1 for c in rg.CAT4_DL]).astype(np.int64)
        return {"x_tab": np.ascontiguousarray(np.concatenate(parts, axis=1)), "x_cat": xc, "x_h": h.astype(np.int64)}

    @property
    def cat_sizes(self) -> list[int]:
        return [len(self.w.ctx.vocab[c]) + 1 for c in rg.CAT4_DL]

    @property
    def columns(self) -> list[str]:
        s = self.scaler
        out = ([] if s is None else s.num + [rg.indicator_name(c) for c in s.ind] + ([rg.STRUCT_COLUMN] if s.struct else []))
        if not scope_.domestic_only():
            out = out + ["src_route01"]
        from . import static_feats as sf

        names = {c.name for c in self.cols}
        if "season" in names:
            out += [f"season_{v}" for v in sf.SEASONS]
        if "aw_label" in names:
            out += [f"aw_label_{v}" for v in rg.AW_LABEL_VALUES]
        return out + [f"x_cat:{c}" for c in rg.CAT4_DL] + ["x_h"]

    def save(self, path) -> None:
        p = Path(path)
        self.base.save(p)
        cf.dump_json(p / "nn60_inputs.json", {"scaler": cf.nan_to_null(self.scaler.state()), "columns": self.columns})


def _onehot(v: np.ndarray, values: Sequence[str]) -> np.ndarray:
    idx = pd.Index(list(values)).get_indexer(pd.Index(v.astype(str)))
    if (idx < 0).any():
        raise StageError(f"원핫 모르는 값: {sorted(set(v[idx < 0].tolist()))[:5]}")
    out = np.zeros((v.shape[0], len(values)), np.float32)
    out[np.arange(v.shape[0]), idx] = 1.0
    return out


def _nn60_model_cls():
    import torch
    from torch import nn

    from . import models_dl as md

    class _Net(nn.Module):
        def __init__(self, d: int, cat_sizes):
            super().__init__()
            self.cat = md.CatEmbedding(cat_sizes)
            self.body = nn.Sequential(nn.Linear(d + self.cat.out_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                      nn.Linear(64, H_MAX))

        def forward(self, x_tab, x_cat):
            return self.body(torch.cat([x_tab, self.cat(x_cat)], dim=1))

    class NN60Model(md._TorchModel):
        """부록 A-8 ②: 입력 → Dense 128 → 64 → Dense 60(sigmoid), 손실 = 자기 h 칸 이진 교차엔트로피(나머지 칸 손실에서 뺌),
        학습 설정 = 다른 딥러닝과 같음(Adam 1e-3·배치 1024·최대 20·인내 3·(sgg, d) 10% 조기 종료·최저 에폭 되돌림)."""

        def __init__(self, cat_sizes, *, device=None, max_epochs: int = md.MAX_EPOCHS, batch_size: int = md.BATCH_SIZE):
            super().__init__(device=device, max_epochs=max_epochs, batch_size=batch_size)
            self.cat_sizes = [int(c) for c in cat_sizes]

        def _logit(self, batch):
            out = self.net(batch["x_tab"], batch["x_cat"])
            return out.gather(1, (batch["x_h"] - 1).view(-1, 1)).squeeze(1)

        def _rows(self, batch, yb):
            import torch.nn.functional as F

            bce = F.binary_cross_entropy_with_logits(self._logit(batch), yb, reduction="none")
            return bce, bce

        @staticmethod
        def _check(inputs):
            if set(inputs) != {"x_tab", "x_cat", "x_h"}:
                raise ValueError(f"② 입력 키: {sorted(inputs)}")
            n = inputs["x_tab"].shape[0]
            if inputs["x_tab"].dtype != np.float32 or not np.isfinite(inputs["x_tab"]).all():
                raise ValueError("x_tab float32·유한")
            h = inputs["x_h"]
            if h.shape != (n,) or h.min() < 1 or h.max() > H_MAX:
                raise ValueError("x_h 1~60")
            return n

        def fit(self, inputs, y, *, seed, n_jobs, groups, eval_set=None):
            n = self._check(inputs)
            yv = np.asarray(y)
            if yv.shape != (n,) or not np.isin(yv, (0, 1)).all():
                raise ValueError("y 0/1")
            es = md.split_es_groups(groups, seed)
            tr_idx, es_idx = np.flatnonzero(~es), np.flatnonzero(es)
            ctx = self._prepare_fit(seed, n_jobs, eval_set)
            self.dims = {"d": int(inputs["x_tab"].shape[1])}
            torch.manual_seed(int(seed))
            self.net = _Net(self.dims["d"], self.cat_sizes).to(ctx["device"])
            self.net.cat.set_seen(inputs["x_cat"][tr_idx])
            self.fit_info.update({"n_rows": n, "n_train": int(tr_idx.size), "n_es": int(es_idx.size)})
            self._train(dict(inputs), yv.astype(np.float32), tr_idx, es_idx, seed)
            return self

        @torch.no_grad()
        def predict_proba(self, inputs, n_iter=None):
            if n_iter is not None:
                raise ValueError("② 는 n_iter 없음")
            n = self._check(inputs)
            self.net.eval()
            out = np.empty(n, np.float64)
            for s in range(0, n, md.PRED_BATCH):
                b = md._to_dev(inputs, np.arange(s, min(n, s + md.PRED_BATCH)), self.device)
                out[s:s + md.PRED_BATCH] = torch.sigmoid(self._logit(b).double()).cpu().numpy()
            return out

        def get_config(self):
            return {"arch": "NN60", "layers": [128, 64, H_MAX], "loss": "bce(own h)", "cat_sizes": self.cat_sizes,
                    "dims": self.dims, "lr": md.LR, "batch_size": self.batch_size, "max_epochs": self.max_epochs,
                    "patience": md.PATIENCE, "es_fraction": md.ES_FRACTION}

        def _payload(self):
            return {"cat_sizes": self.cat_sizes, "dims": self.dims, "max_epochs": self.max_epochs,
                    "batch_size": self.batch_size, "fit_info": self.fit_info}

        @classmethod
        def load(cls, path, device=None):
            p = cls._read(path, "NN60Model")
            m = cls(p["cat_sizes"], device=device, max_epochs=p["max_epochs"], batch_size=p["batch_size"])
            m.dims, m.fit_info = dict(p["dims"]), p["fit_info"]
            m.net = _Net(m.dims["d"], m.cat_sizes)
            m.net.load_state_dict(p["state_dict"])
            m.net.to(md.resolve_device(device)).eval()
            return m

    return NN60Model


# ================================================================ (R4) 씨앗 바꾸기
class SeedOverride:
    """fit·read_auto_learning_rate 의 seed 만 바꿈(5·6 씨앗 0) — 나머지는 감싼 모델 그대로."""

    def __init__(self, inner, seed: int):
        self._inner, self._seed = inner, int(seed)

    def fit(self, X, y, *, seed, n_jobs, **kw):
        self._inner.fit(X, y, seed=self._seed, n_jobs=n_jobs, **kw)
        return self

    def read_auto_learning_rate(self, X, y, *, seed, n_jobs):
        return self._inner.read_auto_learning_rate(X, y, seed=self._seed, n_jobs=n_jobs)

    def __getattr__(self, k):
        return getattr(self._inner, k)


# ================================================================ runner 주입(판 설정 extra.s4)
S4_KEYS = ("row_plan", "catenc_k", "drop_time", "provider", "feature_only", "input_order", "model_seed", "role_note",
           "members", "base", "nn60_from")


def s4_of(spec: rn.VariantSpec) -> dict:
    s = dict(spec.extra.get("s4") or {})
    bad = sorted(set(s) - set(S4_KEYS))
    if bad:
        raise ValueError(f"{spec.name}: extra.s4 모르는 칸 {bad}")
    return s


def version_of(spec: rn.VariantSpec):
    """판 설정 → VersionSpec(입력 차수 = extra.s4.input_order(미리보기 = 1) 또는 판 차수)."""
    s = s4_of(spec)
    order = int(s.get("input_order") or {"1차": 1, "2차": 2}[spec.order])
    ver = dict(spec.extra.get("version", {}))
    mode = spec.mode if spec.kind == "tree" else "sum"
    return rg.load_version({"name": spec.name, "order": order, "W": int(spec.W), "mode": mode, **ver})


class Stage4Wiring:
    """runner 주입 함수(make_inputs·make_model·fit_kwargs) — 단계 4~8 판. 딥러닝 판(NN60 제외)은 wiring 그대로."""

    def __init__(self, ctx, samples: ev.SampleTable, inputs_root: Path, *, data_root: Path | None = None,
                 theil_targets: Path | None = None, n_jobs: int = 8, device: str | None = None,
                 dl_kwargs: Mapping | None = None, check: Callable[[], Any] | None = None):
        from . import wiring as wr

        self.ctx, self.samples, self.inputs_root = ctx, samples, Path(inputs_root)
        self.data_root, self.theil_targets = data_root, theil_targets
        self.n_jobs, self.device, self.dl_kwargs = n_jobs, device, dict(dl_kwargs or {})
        self.check = check or (lambda: None)
        self.base = wr.Wiring(ctx, dl_kwargs=self.dl_kwargs, device=device)
        self._providers: dict[str, _OOFProvider] = {}
        self._selectors: dict[str, BlockSelector] = {}

    def block_selector(self, plan: RowPlan) -> BlockSelector:
        k = json.dumps(plan.to_dict(), sort_keys=True)
        if k not in self._selectors:
            self._selectors[k] = BlockSelector(self.samples, plan, self.inputs_root, self.data_root, self.theil_targets)
        return self._selectors[k]

    def provider(self, spec: rn.VariantSpec) -> _OOFProvider | None:
        s = s4_of(spec)
        pdef = s.get("provider")
        if not pdef:
            return None
        seed = int(s.get("model_seed", 42))
        plan = RowPlan.from_dict(s.get("row_plan"))
        key = json.dumps([pdef, plan.to_dict(), seed], sort_keys=True)
        if key not in self._providers:
            self._providers[key] = PROVIDERS[pdef["kind"]](self, pdef, plan, seed)
        return self._providers[key]

    def make_inputs(self, spec: rn.VariantSpec, split: str):
        s = s4_of(spec)
        if spec.algorithm == "NN60":
            return NN60Inputs(self, version_of(spec), split)
        if spec.kind == "dl":
            return self.base.make_inputs(spec, split)
        return S4TreeInputs(self, version_of(spec) if not s.get("feature_only") else _dummy_version(spec), split,
                            catenc_k=s.get("catenc_k"), scale=(spec.algorithm == "LR"),
                            drop_time=bool(s.get("drop_time")), provider=self.provider(spec),
                            feature_only=bool(s.get("feature_only")))

    def make_model(self, spec: rn.VariantSpec, asm):
        s = s4_of(spec)
        if spec.algorithm == "NN60":
            m = _nn60_model_cls()(asm.cat_sizes, device=self.device, **self.dl_kwargs)
        elif spec.kind == "dl":
            m = self.base.make_model(spec, asm)
        else:
            from . import models_tree as mtr

            bad = sorted(set(spec.params) - {"max_iter", "params", "iter_select"})
            if bad:
                raise ValueError(f"트리 판 params 모르는 칸: {bad}")
            m = mtr.make_model(TREE_ALGO[spec.algorithm], cat_cols=asm.cat_cols,
                               indicator_cols=asm.indicator_cols if spec.algorithm == "LR" else (),
                               max_iter=spec.params.get("max_iter"), params=spec.params.get("params"))
        if "model_seed" in s:
            m = SeedOverride(m, int(s["model_seed"]))
        return m

    def fit_kwargs(self, spec, split, train_meta, X) -> dict:
        if spec.kind == "dl":
            return {"groups": (train_meta["sgg"].astype(str).to_numpy(), fd.day_ints(train_meta["d"]))}
        return {}


def _dummy_version(spec):
    return rg.load_version({"name": spec.name, "order": 2, "W": int(spec.W), "mode": spec.mode or "sum",
                            "include_ids": []})


def install_for_group(specs: Sequence[rn.VariantSpec], inputs_root: Path, samples: ev.SampleTable | None = None) -> None:
    """--variant 하위 프로세스: 묶음 첫 판의 RowPlan 으로 train 행(R1), ①·4-c 쌍 거르기(R2)."""
    s = s4_of(specs[0])
    plan = RowPlan.from_dict(s.get("row_plan"))
    for sp in specs[1:]:
        if RowPlan.from_dict(s4_of(sp).get("row_plan")) != plan:
            raise StageError("한 묶음 판의 RowPlan 이 다름")
    install_train_rows(plan, inputs_root)
    flt = pair_filter_for(plan, inputs_root, samples)
    if flt is not None:
        install_pair_filter(flt)


def pair_filter_for(plan: RowPlan, inputs_root: Path, samples: ev.SampleTable | None):
    if plan.rep == "x1":
        return h1_filter
    if "top5" in plan.steps:
        if samples is None:
            raise ValueError("4-c 쌍 거르기에는 시료 표 필요")
        info = rn.read_json(resolve(inputs_root, plan.sel_dir) / SELECTION_FILE)
        return top5_filter(samples, info["top5"])
    return None


# ================================================================ 사슬 상태
def load_state(path: Path) -> dict:
    st = rn.read_json(path)
    miss = [k for k in STATE_KEYS if k not in st]
    if miss:
        raise StageError(f"사슬 상태에 칸 없음: {miss} — {path}")
    for k in ("current_best", "tree_target"):
        check_best(st[k])
    return st


def check_best(b: Mapping) -> None:
    need = ("name", "kind", "dir", "replacement", "members")
    miss = [k for k in need if k not in b]
    if miss or b["kind"] not in BEST_KINDS:
        raise StageError(f"BestDef 칸·종류 이상: {miss} {b.get('kind')}")
    n = len(b["members"])
    if (b["kind"] == "ensemble" and n < 2) or (b["kind"] != "ensemble" and n != 1):
        raise StageError(f"{b['name']}: 구성 수 {n} 이 종류 {b['kind']} 와 안 맞음")
    for m in b["members"]:
        for k in ("name", "dir"):
            if k not in m:
                raise StageError(f"MemberDef 칸 없음: {k}")


def save_state(path: Path, st: Mapping) -> Path:
    for k in ("current_best", "tree_target"):
        check_best(st[k])
    return cf.dump_json(path, cf.nan_to_null(dict(st)))


def member_spec(inputs_root: Path, m: Mapping) -> rn.VariantSpec:
    return rn.VariantSpec(**rn.read_json(resolve(inputs_root, m["dir"]) / "spec.json"))


def name_part(best: Mapping, inputs_root: Path) -> str:
    """부록 A-14: 단계 4~8 판의 알고리즘 자리 = 바탕 판 구조 약어(바탕이 앙상블·스태킹·교체 후보면 바탕 판 이름)."""
    if best["kind"] in ("tree", "dl") and not best["replacement"]:
        return member_spec(inputs_root, best["members"][0]).algorithm
    return best["name"]


def best_cv(inputs_root: Path, best: Mapping) -> ev.Candidate:
    d = resolve(inputs_root, best["dir"])
    _, done, cv = rn.load_done(d.parent, d.name)
    return ev.Candidate(best["name"], float(cv["cv_band_weighted"]), int(rn.read_json(d / "spec.json")["train_count"]),
                        bool(cv["ties"]["many_ties"]))


# ================================================================ 판 구성 → 실행 판 설정
@dataclass
class Derived:
    """derive_specs 결과: run = runner 로 돌릴 판(묶음 순), ensemble = 앙상블 기록 판(없으면 None), best = 새 BestDef 초안."""
    run: list
    ensemble: rn.VariantSpec | None
    best: dict


def _member_variant(src: rn.VariantSpec, *, name: str, stage: str, role: str, s4: Mapping, base_name: str,
                    params: Mapping | None = None, version: Mapping | None = None, algorithm: str | None = None,
                    W: int | None = None, mode: str | None = None) -> rn.VariantSpec:
    extra = {k: v for k, v in dict(src.extra).items() if k in ("version",)}
    if version is not None:
        extra["version"] = dict(version)
    extra["s4"] = {k: v for k, v in dict(s4).items() if v is not None}
    p = dict(src.params if params is None else params)
    alg = algorithm or src.algorithm
    if p.get(rn.ITER_SELECT) and alg not in BOOST:
        p.pop(rn.ITER_SELECT)
    return rn.VariantSpec(name=name, stage=stage, order=ORDER, kind=src.kind if alg != "NN60" else "dl",
                          algorithm=alg, W=W if W is not None else src.W, mode=mode if mode is not None else src.mode,
                          input_set=f"{name}-입력", role=role, train_count=4, params=p, judgement=None,
                          base_variant=base_name, extra=extra)


def chain_params(src: rn.VariantSpec) -> dict:
    """'반복 수만 다시'(§3): 부스팅이면 iter_select, 실행 때 정한 max_iter 는 설정에 안 둠. 튜닝 설정(params.params)은
    그대로(단계 8 뒤 판) — CatBoost learning_rate 가 있으면 그대로 씀(runner)."""
    p = dict(src.params)
    p.pop("max_iter", None)
    if src.kind == "tree" and src.algorithm in BOOST:
        p[rn.ITER_SELECT] = True
    return p


def derive_specs(best: Mapping, *, inputs_root: Path, name: str, stage: str, role: str, plan: RowPlan,
                 catenc_k: int | None, drop_time: bool = False, model_seed: int | None = None,
                 member_params: Mapping[str, Mapping] | None = None, focal_cat: bool = False,
                 anomaly_reuse: bool = False) -> Derived:
    """현재 최고 판(BestDef) 에 단계 4~6·8 기법을 하는 판 설정(§6 2차 공통 '단계 4~6 을 현재 최고 판에 하는 법').
    plan = 트리 부분 RowPlan(딥러닝·② 는 rep 'own' — h 복제 안 함, ① 은 x1 그대로). catenc_k = 트리 범주 숫자 인코딩 k.
    member_params = {구성 이름: 트리 params(단계 8 튜닝 설정)}, focal_cat = CatBoost 손실만 Focal(단계 8),
    anomaly_reuse = 이상 점수 칼럼을 대상 판 값 모델 그대로 다시 붙임(단계 8 — 값 모델 학습 안 함; 단계 4~6 은 False =
    줄인·뽑은 자료로 다시 학습)."""
    from . import tuning as tu

    check_best(best)
    kind = best["kind"]
    dl_plan = plan if plan.rep == "x1" else plan.with_rep("own")
    members, run = [], []

    def tree_params(m, src):
        p = chain_params(src)
        if member_params and m["name"] in member_params:
            lib = dict(member_params[m["name"]])
            p["params"] = lib
        if focal_cat and src.algorithm == "CAT":
            p["params"] = tu.focal_params(p.get("params"))
        return p

    if kind in ("tree", "dl", "ensemble"):
        for m in best["members"]:
            src = member_spec(inputs_root, m)
            mname = name if kind != "ensemble" else f"{name}{ENS_SEP}{src.algorithm}"
            if src.kind == "tree":
                s4 = {"row_plan": plan.to_dict(), "catenc_k": catenc_k, "drop_time": drop_time or None,
                      "provider": _anomaly_pdef(m, reuse_from=m["dir"] if anomaly_reuse else None),
                      "model_seed": model_seed}
                sp = _member_variant(src, name=mname, stage=stage, role=role, s4=s4, base_name=best["name"],
                                     params=tree_params(m, src))
            else:
                s4 = {"row_plan": dl_plan.to_dict(), "model_seed": model_seed}
                sp = _member_variant(src, name=mname, stage=stage, role=role, s4=s4, base_name=best["name"],
                                     params={})
            run.append(sp)
            members.append({"name": sp.name, "dir": None, "anomaly": m.get("anomaly"),
                            "row_plan": RowPlan.from_dict(sp.extra["s4"].get("row_plan")).to_dict(),
                            "catenc_k": sp.extra["s4"].get("catenc_k")})
    elif kind == "stack4":
        m = best["members"][0]
        src = member_spec(inputs_root, m)
        st = dict(m["stack4"])
        pdef = {"kind": "stack4", "version": st["version"], "W": int(src.W), "mode": src.mode, "catenc_k": catenc_k,
                "params": (member_params or {}).get(m["name"]) or None}
        if focal_cat:
            pp = dict(pdef["params"] or {})
            pp["CAT"] = tu.focal_params(pp.get("CAT"))
            pdef["params"] = pp
        s4 = {"row_plan": plan.to_dict(), "provider": pdef, "feature_only": True, "model_seed": model_seed}
        sp = _member_variant(src, name=name, stage=stage, role=role, s4=s4, base_name=best["name"], params={},
                             algorithm="LR")
        run.append(sp)
        members.append({"name": sp.name, "dir": None, "stack4": st, "row_plan": plan.to_dict(), "catenc_k": catenc_k})
    else:   # nn60 — 다시 학습(출력 60칸·자기 h 칸 손실 그대로, §6 단계 5·6)
        m = best["members"][0]
        src = member_spec(inputs_root, m)
        s4 = {"row_plan": dl_plan.to_dict(), "model_seed": model_seed}
        sp = _member_variant(src, name=name, stage=stage, role=role, s4=s4, base_name=best["name"], params={},
                             algorithm="NN60")
        run.append(sp)
        members.append({"name": sp.name, "dir": None, "row_plan": dl_plan.to_dict(), "catenc_k": None})
    ens = None
    if kind == "ensemble":
        ens = rn.VariantSpec(name=name, stage=stage, order=ORDER, kind="tree", algorithm="T3", W=run[0].W,
                             mode=run[0].mode, input_set=f"{name}-구성", role=role, train_count=0, params={},
                             judgement=None, base_variant=best["name"],
                             extra={"s4": {"members": [s.name for s in run]}})
    new_best = {"name": name, "kind": kind, "dir": None, "replacement": kind == "nn60", "members": members}
    return Derived(run=run, ensemble=ens, best=new_best)


def _anomaly_pdef(m: Mapping, reuse_from: str | None = None) -> dict | None:
    a = m.get("anomaly")
    if not a:
        return None
    d = {"kind": "vae", "column": a["column"], "version": dict(a["version"]), "W": int(a["W"]), "mode": a["mode"],
         "source": a.get("source", "3-1")}
    if reuse_from is not None:
        d.update(kind="vae_frozen", from_dir=reuse_from)
    return d


def fill_dirs(best: dict, out_rel: str) -> dict:
    """새 BestDef 의 dir 를 결과 폴더(--inputs 아래) 로."""
    b = copy.deepcopy(best)
    b["dir"] = f"{out_rel.rstrip('/')}/{b['name']}"
    for m in b["members"]:
        m["dir"] = f"{out_rel.rstrip('/')}/{m['name']}"
    return b


# ================================================================ 앙상블 기록 판
def write_ensemble(out_dir: Path, spec: rn.VariantSpec, samples: ev.SampleTable, exp_log: rn.ExperimentLog,
                   code_md5: str, tracker=None, stage_run_id: str | None = None, experiment: str = EXPERIMENT) -> dict:
    """구현 세부 (g): 구성 판 예측 평균 → 판 폴더(spec·pred_*·pred_thr_*·cv_scores·h_metrics·done). 끝났으면 그대로."""
    vdir = rn.variant_dir(out_dir, spec.name)
    if (vdir / "done.json").exists():
        d = rn.read_json(vdir / "done.json")
        if d["spec_md5"] != spec.md5():
            raise rn.ResumeError(f"{spec.name}: 끝난 앙상블 판 설정이 다름")
        return d
    if vdir.exists():
        rn._move_partial(vdir, out_dir, reason="멈춘 앙상블 판 다시")
    mems = list(spec.extra["s4"]["members"])
    mdirs = [rn.variant_dir(out_dir, n) for n in mems]
    for d in mdirs:
        if not (d / "done.json").exists():
            raise StageError(f"구성 판이 안 끝남: {d.name}")
    vdir.mkdir(parents=True)
    cf.dump_json(vdir / "spec.json", spec.to_dict())

    def avg(fname):
        fr = [pd.read_parquet(d / fname) for d in mdirs]
        base = fr[0][["lims_data_key", "h", "fold"]]
        for f in fr[1:]:
            if not (f["lims_data_key"].equals(base["lims_data_key"]) and f["h"].equals(base["h"])):
                raise AssertionError(f"{fname}: 구성 판 쌍 순서가 다름")
        return base.assign(pred=np.mean([f["pred"].to_numpy(dtype=np.float64) for f in fr], axis=0))

    for s in rn.SPLIT_ORDER:
        avg(f"pred_{s}.parquet").to_parquet(rn.pred_path(vdir, s), index=False)
    has_thr = any((d / f"pred_{rn.THR_PREFIX}fold1.parquet").exists() for d in mdirs)
    if has_thr:
        for f in ev.CV_FOLDS:
            fr = []
            for d in mdirs:
                p = d / f"pred_{rn.THR_PREFIX}{f}.parquet"
                fr.append(pd.read_parquet(p if p.exists() else rn.pred_path(d, f)))
            base = fr[0][["lims_data_key", "h", "fold"]]
            base.assign(pred=np.mean([x["pred"].to_numpy(dtype=np.float64) for x in fr], axis=0)).to_parquet(
                rn.pred_path(vdir, rn.THR_PREFIX + f), index=False)
    cv = ev.score_cv(rn.load_preds(vdir, ev.CV_FOLDS), samples, thr_preds=rn.load_thr_preds(vdir))
    count = exp_log.cv_scored(spec.stage, spec.name)
    cf.dump_scores_json(vdir / "cv_scores.json", cv)
    rn.write_h_metrics(vdir, ev.h_metrics_cv(rn.load_preds(vdir, ev.CV_FOLDS), samples,
                                             float(cv["threshold"]["threshold"])))
    run_id = None
    if tracker is not None:
        run_id = tracker.start_variant(experiment, stage_run_id, spec.name,
                                       {"단계": spec.stage, "역할": spec.role, "바탕 판": spec.base_variant or ""},
                                       {"algorithm": "T3", "members": mems, "train_count": 0})
        tracker.log_cv(run_id, cv, selection_count=count)
        tracker.finish(run_id)
    done = {"name": spec.name, "stage": spec.stage, "spec_md5": spec.md5(), "code_md5": code_md5,
            "cv_band_weighted": cv["cv_band_weighted"], "many_ties": cv["ties"]["many_ties"],
            "selection_count": count, "mlflow_run_id": run_id, "members": mems}
    cf.dump_scores_json(vdir / "done.json", done)
    exp_log.append("done", stage=spec.stage, name=spec.name, cv=cv["cv_band_weighted"], count=count)
    return done


# ================================================================ 판정(§6 2차 공통)
def judge_step(*, step: str, state: Mapping, stack: Sequence[tuple[dict, Path]], repl: Sequence[tuple[dict, Path]],
               samples: ev.SampleTable, inputs_root: Path, tree_like: Callable[[dict], bool] | None = None) -> dict:
    """(가) 쌓는 후보 val 1위가 현재 최고 판보다 오르면 채택 → (나) 교체 후보 val 1위가 그때의 현재 최고 판보다 오르면
    채택(§6 2차 공통, 4자리 — §1-2). stack·repl = [(새 BestDef(dir 채움), 판 폴더)]. 돌려줌 = 판정 기록 + 새 state."""
    cur = dict(state["current_best"])
    cur_dir = resolve(inputs_root, cur["dir"])
    dir_of = {cur["name"]: cur_dir}

    def cand(b, d):
        dir_of[b["name"]] = d
        _, done, cv = rn.load_done(d.parent, d.name)
        return ev.Candidate(b["name"], float(cv["cv_band_weighted"]), int(rn.read_json(d / "spec.json")["train_count"]),
                            bool(cv["ties"]["many_ties"]))

    def rescore(n):
        return float(ev.score_cv(rn.load_preds(dir_of[n], ev.CV_FOLDS), samples, rescore=True)["cv_band_weighted"])

    rec = {"step": step, "before": cur["name"], "parts": []}
    new = dict(state)
    tree_target = dict(state["tree_target"])
    with ev.selection_guard():
        for label, group in (("쌓는 후보", stack), ("교체 후보", repl)):
            if not group:
                continue
            cands = [cand(b, d) for b, d in group]
            ranked = ev.rank_candidates(cands)
            top = ranked[0]
            curc = ev.Candidate(cur["name"], *_cv_of(dir_of[cur["name"]]))
            j = ev.judge_adoption(top, curc, rescore)
            bdef = next(b for b, _ in group if b["name"] == top.name)
            rec["parts"].append({"kind": label, "ranking": [c.name for c in ranked], "cv": {c.name: c.cv for c in ranked},
                                 "top": top.name, "against": cur["name"], "judge": j, "adopted": bool(j["adopt"])})
            if j["adopt"]:
                cur = bdef
                if label == "쌓는 후보" and (tree_like(bdef) if tree_like else bdef["kind"] in ("tree", "ensemble")):
                    if not bdef.get("replacement"):
                        tree_target = bdef
    rec["after"] = cur["name"]
    rec["tree_target_after"] = tree_target["name"]
    rec["utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    new["current_best"] = cur
    new["tree_target"] = tree_target
    new["step"] = step
    new["history"] = list(state.get("history", [])) + [{k: rec[k] for k in ("step", "before", "after",
                                                                             "tree_target_after", "utc")}]
    return {"record": rec, "state": new}


def _cv_of(d: Path) -> tuple[float, int, bool]:
    _, done, cv = rn.load_done(d.parent, d.name)
    return float(cv["cv_band_weighted"]), int(rn.read_json(d / "spec.json")["train_count"]), bool(cv["ties"]["many_ties"])


def append_chain_log(inputs_root: Path, rel: str, line: Mapping) -> Path:
    p = resolve(inputs_root, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(cf.nan_to_null(dict(line)), ensure_ascii=False) + "\n")
    return p


def tag_judgements(tracker, rec: Mapping, find: Callable[[str], Any]) -> None:
    """MLflow 판정 tag(부록 A-15): 단계 1위 = 채택/버림(판정 상대), 나머지 = 버림(단계 1위 아님) — 판정 때 한 번."""
    if tracker is None:
        return
    for part in rec["parts"]:
        for n in part["ranking"]:
            r = find(n)
            if r is None or "판정" in r.data.tags:
                continue
            if n == part["top"]:
                tracker.set_judgement(r.info.run_id, "채택" if part["adopted"] else "버림", against=part["against"])
            else:
                tracker.set_judgement(r.info.run_id, "버림(단계 1위 아님)", against=part["against"])


def best_row(inputs_root: Path, best: Mapping, desc: str) -> rn.BaseRow:
    d = resolve(inputs_root, best["dir"])
    done = rn.read_json(d / "done.json")
    return rn.BaseRow(best["name"], desc, rn.load_preds(d, ev.CV_FOLDS), pd.read_parquet(rn.pred_path(d, "refit")),
                      int(done["selection_count"]), done.get("mlflow_run_id"), thr_preds=rn.load_thr_preds(d))


# ================================================================ 앞 단계 결정 읽기(1-3·단계 2 준비)
def load_k(inputs_root: Path, rel: str) -> int:
    s = cf.load_json_strict(resolve(inputs_root, rel))
    if "k" not in s:
        raise StageError("단계 2 준비 고르기에 k 없음")
    return int(s["k"])
