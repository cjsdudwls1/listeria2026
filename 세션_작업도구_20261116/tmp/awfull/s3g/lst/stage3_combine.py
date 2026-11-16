"""단계 3-1 ~ 3-3 결합(트리끼리만) 공통 — 계획서 요약 15, §6 2차 공통(채택 판정·교체 후보·트리 대상 판·현재 입력)·단계
3-1 ~ 3-3·추가 판 ④, §7-3(블록 OOF·엄격 OOF), §1-2(동점·채택), §1-3(판 이름·test 막기), 부록 A-8(3-1 VAE 입력·robust z)·
A-9(3-1 23·3-2 8·④ 118)·A-12(같은 행 모델 재사용·이상 점수 예측 범위·3-3 백분위 평균 대조·판별 학습 횟수)·A-14·A-15.

판(이름 = 부록 A-14):
  3-1-VAEt        이상 점수 단독(비교만) — VAE 19 모델(block_plan 재사용 켬): fold train 행 = fold 안 블록 OOF, fold validation
                  쌍 = fold train 전체 모델, refit train 행 = 전역 블록 OOF, test 쌍 = test 전 전체 모델. 판 예측(pred_*) =
                  validation·test 쌍 값, oof_<split>.parquet = train 행 값 → 이 두 벌이 그대로 이상 점수 칼럼 값.
  3-1-<알고리즘>-VAE  트리 대상 판 + 이상 점수 1칼럼(쌓는 후보)
  3-2-<알고리즘>   3-0 트리 상위 3개마다 현재 입력 + 이상 점수 칼럼(쌓는 후보, 같은 판이면 학습 없이 다시 씀 — 1위 트리 = 3-1 판)
  x4-L6           ④ LIMS 만 스태킹(교체 후보): 트리 6종(기본 설정·기본 반복 수) × 19(블록 OOF·재사용) + 로지스틱 메타 4
  3-3-T3          트리 상위 3개 (시료, h) 확률 평균(쌓는 후보, 학습 없음) + 백분위 평균 대조(기록만)

이상 점수 칼럼(트리 입력): `AnomalyInputs` 가 runner 의 조립 객체를 감싸 칼럼 'anom_vae' 를 맨 뒤에 붙임 — 값은 3-1-VAEt
판 폴더의 저장 값(oof_<split>·pred_<split>)을 (lims_data_key, h) 로 찾음(`AnomalyLookup`). 행 d ≤ 그 판 train 끝이면
OOF 값, 아니면 그 판 전체 모델 값(두 벌은 d 로 갈려 겹치지 않음) — 없는 (시료, h) 면 멈춤(누수·빠짐 방지).

구현 세부(계획서에 없음 — 담당자 확인):
 (a) 이상 점수 칼럼 이름 'anom_vae', 트리 X 맨 뒤(importance.Stack 기본 결합과 같은 자리). 로지스틱은 이 칼럼도 모델 안
     표준화(models_tree 로지스틱 — 숫자 칼럼 train 평균·표준편차)로 들어감(§4-2 로지스틱 규칙).
 (b) VAE 입력 = 트리 대상 판 X 의 숫자 칼럼(등록부 kind 'numeric' — 범주 5·season·aw_label·h·g 뺌) → TabularScaler
     (표준화 → 결측 0 → 결측 표시 + 해외 구조적 결측). 표준화 통계 = 그 VAE 모델의 학습 행 전체(불검출·검출 모두 —
     AnomalyModel 은 그 안 y == 0 행만 학습·조기 종료). 트리 대상 판이 로지스틱이어도 표준화 전 원값 X 에서 시작.
 (c) VAE 학습 행·OOF 대상 행 = (시료, 자기 h) — 입력 X 는 그 행들의 합집합으로 한 번만 만들어(같은 (시료, h) 는 split 과
     무관하게 같은 값 — 이력 집계는 d ≤ C 시료만) 모델마다 잘라 씀. train 행 입력의 이력 모집단 끝 = refit train 끝(모든 행
     C ≤ 그 끝 — 블록 모델이 다른 fold 의 블록 행을 예측할 때 inputs 의 C 점검을 지나게, 값은 같음), ④ 의 기본 트리 조립
     객체도 같음. 평가 쌍은 split 마다 h 별로 메모리에서(B-1).
 (d) 누수 점검(실행 중 assert): OOF 값을 낸 모델의 실제 학습 시료 키에 그 행 시료가 없음, 판 전체 모델의 학습 시료 d 는 모두
     그 판 train 끝 이하(평가 쌍 d 는 그 뒤), train 행마다 OOF 값 정확히 1개이고 그 모델 = blocks.assemble_oof 가 정한 모델.
 (e) 연기 시험(smoke): train 행·평가 쌍 일부는 runner 와 같은 규칙·씨앗(_smoke_train·[seed, h])으로 골라 이상 점수 칼럼
     값이 판의 행과 맞게 함.
 (f) ④ 저장: 판 전체 모델 역할(fold1~3·refit full — fold2-full 은 fold3-b5·전역 b5 와 같은 모델)의 트리 모델·메타 로지스틱
     4개는 pickle 저장(최종 모델 순열 중요도의 다시 추론용), 블록 전용 모델은 설정 덤프만(디스크 — 설정 x4_save_models).
     OOF 6칼럼 표(oof_<split>.parquet)는 저장. 기본 트리 예측은 RF·ET·부스팅 = 같은 입력 행 한 번 예측(runner 와 같은 규칙).
 (g) ④ 메타 = models_tree 로지스틱(max_iter 2,000, 숫자 칼럼 train 평균·표준편차 표준화 — §4-2), 메타 입력 칼럼
     p_XGB·p_LGB·p_CAT·p_RF·p_ET·p_LR(트리 6종 순서).
 (h) 3-3 앙상블: CV = 구성 판 pred_fold*(교차 N) 평균, threshold = 구성 판 pred_thr_fold*(공통 N — 없는 판(RF·ET·로지스틱)은
     pred_fold*) 평균, test = 구성 판 pred_refit 평균. 백분위 = 그 split 채점 쌍 안 순위(같은 값 = 평균 순위) ÷ 쌍 수, test 는
     pred_refit 쌍 전부(덜 찬 달 빼기 전) 기준. 백분위 평균 대조의 threshold 도 같은 규칙(공통 N 예측의 백분위 평균).
 (i) 판 폴더(3-1-VAEt·x4-L6·3-3-T3) = runner 판 폴더 꼴(spec·pred_*·cv_scores·h_metrics·run_meta·done) — x4-L6 의 spec 은
     VariantSpec 꼴(kind tree·algorithm LR = 메타, extra.stack), 3-1-VAEt(kind 'vae')·3-3-T3(kind 'ensemble') 은 VariantSpec
     밖 꼴이라 보고 표에는 BaseRow 로 넣음.
 (j) '같은 validation 으로 고른 횟수': 3-1-VAEt·x4-L6·3-3-T3·3-3 백분위 평균 대조 모두 CV 채점 1회로 셈(부록 A-12 '모든 판·설정').
 (l) 이상 점수 칼럼이 있는 트리 판(3-1·3-2·3-3 학습 판)의 판 폴더 inputs_<split>/value_models = lst/stage4 꼴(plan.json·
     model_XX/vae.pt·scaler.json, XX = block_plan(reuse=False, splits=[split]) 위치) — 3-1-VAEt 의 재사용 계획 모델을 역할대로
     복사(단계 8 FrozenVAEProvider 가 다시 학습 없이 같은 값을 냄). plan.json seed = 42, row_plan = 고르기 없음·자기 h.
 (k) 3-2 같은 판 판정 = (알고리즘·W·입력 방식·변수 묶음·params·이상 점수 칼럼 유무)가 같음 — 같으면 학습 없이 원 판을 가리키는
     기록 폴더(stage3.write_pointer)와 MLflow 새 run(tag '원 판').
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import blocks as bk
from . import config as cf
from . import evaluate as ev
from . import folds as fd
from . import registry as rg
from . import runner as rn
from . import stage3 as s3
from .hassign import H_MAX
from .paths import resolve

ANOM_COL = "anom_vae"
VAE_T = "3-1-VAEt"
X4 = "x4-L6"
T3 = "3-3-T3"
T3_PCT = "3-3-T3-백분위"
DECISION = {"3-1": "3-1_decision.json", "3-2": "3-2_decision.json", "3-3": "3-3_decision.json"}
PLAN = {"3-1": "3-1_plan.json", "3-2": "3-2_plan.json", "3-3": "3-3_plan.json"}
OOF_PREFIX = "oof_"
VAE_MODEL_DIR = "vae_models"
X4_MODEL_DIR = "base_models"
TREE6 = s3.TREE6
X4_COLS = tuple(f"p_{a}" for a in TREE6)
X4_TRAIN_COUNT = 118           # 부록 A-12 '④ = 118'
SPLIT_ORDER = bk.SPLIT_ORDER
CV_FOLDS = ev.CV_FOLDS


class CombineError(RuntimeError):
    pass


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def spec_md5(spec: Mapping) -> str:
    import hashlib

    return hashlib.md5(json.dumps(dict(spec), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# ================================================================ 사슬 상태(현재 최고 판·트리 대상 판·상위 3·현재 입력)
def board_ref(inputs_root: Path, name: str, rel_dir: str, *, replacement: bool = False) -> dict:
    """판정에 쓰는 판 정보 — 예측·점수가 있는 폴더(rel_dir, --inputs 아래) 기준. 끝 판 이름이면 원 판 폴더."""
    d = resolve(inputs_root, rel_dir)
    spec = rn.read_json(d / "spec.json")
    done = rn.read_json(d / "done.json")
    return {"name": name, "dir": rel_dir.rstrip("/"), "source": d.name, "cv": float(done["cv_band_weighted"]),
            "many_ties": bool(done["many_ties"]), "train_count": int(spec["train_count"]), "kind": spec["kind"],
            "algorithm": spec["algorithm"], "replacement": bool(replacement)}


def candidate(ref: Mapping) -> ev.Candidate:
    return ev.Candidate(ref["name"], float(ref["cv"]), int(ref["train_count"]), bool(ref["many_ties"]),
                        {"kind": ref["kind"], "algorithm": ref["algorithm"]})


def read_decision(inputs_root: Path, rel_dir: str, fname: str) -> dict:
    p = resolve(inputs_root, rel_dir) / fname
    if not p.exists():
        raise CombineError(f"앞 단계 판정이 먼저: {p}")
    d = rn.read_json(p)
    d["_md5"] = cf.file_md5(p)
    return d


def state_after_x67(inputs_root: Path, dirs: Mapping[str, str]) -> dict:
    """3-1 시작 상태(v78) = 3-0 순위(트리 대상 판 = 3-0 트리 6종 1위·트리 상위 3개 = 3-0 트리 CV 순·현재 입력 = 변수 전체)
    + ⑥⑦ 판정(현재 최고 판 = 3-0 13 판 중 CV 1위 또는 채택된 ⑥⑦) — 둘 다 x67_decision.json 하나에서."""
    x67 = read_decision(inputs_root, dirs["x67-judge"], s3.X67_DECISION_FILE)
    boards = x67["boards_3_0"]
    cur = board_ref(inputs_root, x67["current_best_after"], x67["current_best_after_dir"],
                    replacement=bool(x67["adopted"]))
    tt = x67["tree_target"]
    if boards[tt]["structure"] not in TREE6:
        raise CombineError(f"트리 대상 판이 트리가 아님: {tt}")
    top3 = [{"name": n, "algorithm": boards[n]["structure"], "dir": boards[n]["dir"]} for n in x67["tree_top3"]]
    if len(top3) != 3 or any(t["algorithm"] not in TREE6 for t in top3) or x67["tree_top3"] != x67["tree_rank_3_0"][:3]:
        raise CombineError(f"트리 상위 3 이상: {x67['tree_top3']}")
    return {
        "after_step": "⑥⑦", "current_best": cur,
        "tree_target": {"name": tt, "algorithm": boards[tt]["structure"], "dir": boards[tt]["dir"], "kind": "tree"},
        "tree_top3": top3,
        "current_inputs": {t["algorithm"]: {"version": boards[t["name"]]["version"], "anomaly": False} for t in top3},
        "anomaly_adopted": False, "adopted_3_2": False, "board_3_1": None, "boards_3_2": None,
        "sources": {"x67_md5": x67["_md5"]},
    }


def state_from(inputs_root: Path, dirs: Mapping[str, str], step: str) -> dict:
    """앞 단계(3-1 → 3-2, 3-2 → 3-3) 판정 뒤 상태."""
    d = read_decision(inputs_root, dirs[step], DECISION[step])
    st = dict(d["state_after"])
    st["sources"] = {**dict(st.get("sources", {})), f"{step}_md5": d["_md5"]}
    return st


# ================================================================ 판 설정
def source_spec(inputs_root: Path, rel_dir: str) -> rn.VariantSpec:
    return rn.VariantSpec(**rn.read_json(resolve(inputs_root, rel_dir) / "spec.json"))


def anomaly_extra(vae_rel: str) -> dict:
    return {"column": ANOM_COL, "source": VAE_T, "dir": vae_rel.rstrip("/")}


def spec_plus_anom(src: rn.VariantSpec, *, name: str, stage: str, vae_rel: str, base_variant: str | None,
                   role: str = s3.ROLE_CAND) -> rn.VariantSpec:
    """트리 판 설정 그대로(부스팅은 반복 수만 다시 — iter_select 그대로) + 이상 점수 1칼럼."""
    if src.kind != "tree" or src.algorithm not in TREE6:
        raise CombineError(f"이상 점수 칼럼은 트리 판에만: {src.name}")
    ext = dict(src.extra)
    if "anomaly" in ext:
        raise CombineError(f"{src.name}: 이미 이상 점수 칼럼이 있음")
    return rn.VariantSpec(name=name, stage=stage, order=s3.ORDER, kind="tree", algorithm=src.algorithm, W=src.W,
                          mode=src.mode, input_set=f"{src.input_set}+VAE", role=role, train_count=4,
                          params=dict(src.params), judgement=None, base_variant=base_variant,
                          extra={"version": dict(ext["version"]), "anomaly": anomaly_extra(vae_rel)})


def content_key(spec: rn.VariantSpec | Mapping) -> str:
    """같은 판 판정(구현 세부 k)."""
    d = spec.to_dict() if isinstance(spec, rn.VariantSpec) else dict(spec)
    ext = dict(d.get("extra") or {})
    return json.dumps([d["kind"], d["algorithm"], d["W"], d["mode"], ext.get("version"), dict(d.get("params") or {}),
                       bool(ext.get("anomaly"))], sort_keys=True, ensure_ascii=False)


def name_3_1(alg: str) -> str:
    return f"3-1-{alg}-VAE"


def name_3_2(alg: str) -> str:
    return f"3-2-{alg}"


# ================================================================ 행·평가 쌍(runner 와 같은 규칙)
def split_train_rows(samples: ev.SampleTable, split: str, W: int, *, smoke: Mapping | None, seed: int,
                     known: pd.DataFrame | None = None) -> pd.DataFrame:
    """그 판 train 행(시료당 자기 h 1개) — runner._train_rows 와 같은 행(연기 시험 부분도 같은 씨앗)."""
    known = samples.known_frame() if known is None else known
    rows = fd.make_train_rows(known, fd.SPLITS[split])
    if smoke is not None:
        rows = rn._smoke_train(rows, int(smoke["max_train_rows"]), seed)
    return rn.add_bounds(rows, W)


def model_train_rows(model: bk.BlockModel, samples: ev.SampleTable, W: int, *, smoke: Mapping | None, seed: int,
                     known: pd.DataFrame | None = None) -> pd.DataFrame:
    known = samples.known_frame() if known is None else known
    rows = bk.model_train_rows(model, known)
    if smoke is not None:
        rows = rn._smoke_train(rows, int(smoke["max_train_rows"]), seed)
    return rn.add_bounds(rows, W)


def iter_eval_meta(samples: ev.SampleTable, split: str, W: int | None, *, max_pairs_per_h: int | None = None,
                   seed: int = 42, check: Callable[[], Any] | None = None):
    """평가 쌍을 h 마다(runner.predict_pairs_multi 와 같은 쌍·순서·연기 시험 부분) → (h, 시료 위치, 행 표)."""
    meta = samples.meta()
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, fd.SPLITS[split]):
        if max_pairs_per_h is not None and idx.size > max_pairs_per_h:
            idx = np.sort(np.random.default_rng([seed, h]).choice(idx, max_pairs_per_h, replace=False))
        if idx.size == 0:
            continue
        if check is not None:
            check()
        m = meta.iloc[idx].copy()
        m["h"] = np.int16(h)
        yield h, idx, rn.add_bounds(m, W)


def _kh(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([df["lims_data_key"].astype(str).to_numpy(dtype=object),
                                      df["h"].to_numpy().astype(np.int64)])


def _pred_frame(keys, h, split: str, pred) -> pd.DataFrame:
    return pd.DataFrame({"lims_data_key": np.asarray(keys, dtype=object), "h": np.asarray(h, dtype=np.int16),
                         "fold": split, "pred": np.asarray(pred, dtype=np.float64)})


# ================================================================ 이상 점수 칼럼(값 찾기·조립 객체 감싸기)
class AnomalyLookup:
    """3-1-VAEt 판 폴더의 저장 값 → (시료, h) 이상 점수. split 마다 (시료 수, 60) float32 표(4 split 까지 메모리 — 실제 자료 split 하나 약 160MB)."""

    def __init__(self, vae_dir: Path, samples: ev.SampleTable, keep: int = 4):
        self.dir = Path(vae_dir)
        if not (self.dir / "done.json").exists():
            raise CombineError(f"3-1-VAEt 판이 끝나지 않음: {self.dir}")
        self.samples = samples
        self.keep = keep
        self._tabs: dict[str, np.ndarray] = {}

    def table(self, split: str) -> np.ndarray:
        if split in self._tabs:
            return self._tabs[split]
        st = self.samples
        sp = fd.SPLITS[split]
        tab = np.full((len(st.keys), H_MAX), np.nan, dtype=np.float32)
        oof = pd.read_parquet(self.dir / f"{OOF_PREFIX}{split}.parquet", columns=["lims_data_key", "h", "value"])
        evp = pd.read_parquet(rn.pred_path(self.dir, split), columns=["lims_data_key", "h", "pred"])
        for df, col, train in ((oof, "value", True), (evp, "pred", False)):
            if _kh(df).duplicated().any():
                raise CombineError(f"{split}: 이상 점수 (시료, h) 중복")
            pos = st.positions(df["lims_data_key"].astype(str).to_numpy(dtype=object))
            d = st.d_days[pos]
            if train and (d > sp.train_end).any():
                raise CombineError(f"{split}: OOF 표에 train 밖 시료")
            if not train and (d <= sp.train_end).any():
                raise CombineError(f"{split}: 평가 쌍 값에 train 기간 시료")
            h = df["h"].to_numpy().astype(np.int64)
            v = df[col].to_numpy(dtype=np.float64)
            if not np.isfinite(v).all():
                raise CombineError(f"{split}: 이상 점수에 NaN·inf")
            tab[pos, h - 1] = v.astype(np.float32)
        while len(self._tabs) >= self.keep:
            self._tabs.pop(next(iter(self._tabs)))
        self._tabs[split] = tab
        return tab

    def values(self, meta: pd.DataFrame, split: str) -> np.ndarray:
        pos = self.samples.positions(meta["lims_data_key"].astype(str).to_numpy(dtype=object))
        h = meta["h"].to_numpy().astype(np.int64)
        if h.size and (h.min() < 1 or h.max() > H_MAX):
            raise CombineError("h 는 1~60")
        v = self.table(split)[pos, h - 1]
        if np.isnan(v).any():
            raise CombineError(f"{split}: 이상 점수가 없는 (시료, h) {int(np.isnan(v).sum())}개 — 3-1-VAEt 범위 밖")
        return v


class AnomalyInputs:
    """runner 조립 객체 감싸기: inner 의 X 맨 뒤에 이상 점수 칼럼(구현 세부 a)."""

    def __init__(self, inner, lookup: AnomalyLookup, split: str):
        self.inner, self.lookup, self.split = inner, lookup, split

    def fit(self, train_meta: pd.DataFrame) -> "AnomalyInputs":
        self.inner.fit(train_meta)
        return self

    def transform(self, meta: pd.DataFrame) -> pd.DataFrame:
        X = self.inner.transform(meta)
        if not isinstance(X, pd.DataFrame):
            raise CombineError("이상 점수 칼럼은 트리 입력(DataFrame)에만")
        if ANOM_COL in X.columns:
            raise CombineError("이상 점수 칼럼이 이미 있음")
        a = self.lookup.values(meta, self.split)
        X[ANOM_COL] = pd.Series(a, index=X.index, dtype=np.float32)
        return X

    def __getattr__(self, name):          # 그 밖 속성(fitted·spec 등)은 감싼 조립 객체 것
        if name in ("inner", "lookup", "split"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def pair_cache(self):
        return self.inner.pair_cache() if hasattr(self.inner, "pair_cache") else contextlib.nullcontext()

    @property
    def columns(self) -> list[str]:
        return list(self.inner.columns) + [ANOM_COL]

    def save(self, path) -> None:
        p = Path(path)
        if hasattr(self.inner, "save"):
            self.inner.save(p)
        else:
            p.mkdir(parents=True, exist_ok=False)
        cf.dump_json(p / "anomaly_column.json", {"column": ANOM_COL, "source_dir": str(self.lookup.dir.name),
                                                 "split": self.split})
        write_frozen_value_models(self.lookup.dir, p, self.split)


def frozen_def(vae_dir: Path) -> dict:
    """lst/stage4 값 모델 정의(VAEProvider 의 pdef 꼴: kind·column·version·W·mode·source)."""
    v = rn.read_json(Path(vae_dir) / "spec.json")
    return {"kind": "vae", "column": ANOM_COL, "version": dict(v["extra"]["version"]), "W": int(v["W"]),
            "mode": v["mode"], "source": VAE_T}


def write_frozen_value_models(vae_dir: Path, inputs_split_dir: Path, split: str) -> Path:
    """판 폴더 inputs_<split>/value_models 를 lst/stage4 꼴로(구현 세부 l — FrozenVAEProvider 가 읽음):
    plan.json(def·row_plan·seed·models) + model_XX/vae.pt·scaler.json, XX = block_plan(reuse=False, splits=[split]) 안
    위치(블록 순 → 판 train 전체). 각 칸 = 3-1-VAEt 가 그 역할((split, 블록)·(split, 전체))로 쓴 재사용 계획의 모델 사본."""
    import shutil

    vae_dir = Path(vae_dir)
    reuse = bk.block_plan(reuse=True)
    by_role = {(r.split, r.kind, r.block): m.name for m in reuse for r in m.roles}
    single = bk.block_plan(reuse=False, splits=[split])
    p = Path(inputs_split_dir) / "value_models"
    p.mkdir(parents=True, exist_ok=False)
    recs = []
    for i in bk.models_for_split(single, split):
        r = single[i].owner
        src = by_role[(r.split, r.kind, r.block)]
        info = rn.read_json(vae_dir / VAE_MODEL_DIR / f"{src}.json")
        q = p / f"model_{i:02d}"
        q.mkdir()
        shutil.copyfile(vae_dir / VAE_MODEL_DIR / f"{src}.pt", q / "vae.pt")
        cf.dump_json(q / "scaler.json", info["scaler"])
        recs.append({"model": single[i].name, "source_model": src, "n": info["n_train_rows"],
                     "pos": info["n_train_pos"], "train_d": info["train_intervals"]})
    cf.dump_json(p / "plan.json", {"def": frozen_def(vae_dir), "row_plan": {"steps": [], "rep": "own", "sel_dir": None},
                                   "seed": 42, "models": recs,
                                   "source": f"{VAE_T} 의 재사용 계획 모델 사본(3-1 — 줄이지 않은 자료)"})
    return p


def anomaly_make_inputs(base_make_inputs: Callable, samples: ev.SampleTable, inputs_root: Path) -> Callable:
    """runner 주입 make_inputs: 판 설정 extra.anomaly 가 있으면 이상 점수 칼럼을 붙인 조립 객체."""
    looks: dict[str, AnomalyLookup] = {}

    def make(spec, split):
        asm = base_make_inputs(spec, split)
        a = (spec.extra or {}).get("anomaly")
        if not a:
            return asm
        if a.get("column") != ANOM_COL or a.get("source") != VAE_T:
            raise CombineError(f"{spec.name}: 이상 점수 칸 설정 이상 {a}")
        if a["dir"] not in looks:
            looks[a["dir"]] = AnomalyLookup(resolve(inputs_root, a["dir"]) / VAE_T, samples)
        return AnomalyInputs(asm, looks[a["dir"]], split)
    return make


def vae_input_md5s(inputs_root: Path, vae_rel: str) -> dict[str, str]:
    d = resolve(inputs_root, vae_rel) / VAE_T
    files = [f"{OOF_PREFIX}{s}.parquet" for s in SPLIT_ORDER] + [f"pred_{s}.parquet" for s in SPLIT_ORDER]
    return {f"vae:{f}": cf.file_md5(d / f) for f in files}


# ================================================================ 판 폴더 공통(시작·마무리)
def begin_board(vdir: Path, out_dir: Path, spec: Mapping, code_md5: str) -> str:
    """'done'(같은 설정·코드면 건너뜀) | 'new'. 멈춘 폴더는 _incomplete 로(B-1). 설정·코드 다르면 멈춤."""
    md5 = spec_md5(spec)
    if (vdir / "done.json").exists():
        d = rn.read_json(vdir / "done.json")
        if d["spec_md5"] != md5:
            raise rn.ResumeError(f"{vdir.name}: 끝난 판의 설정이 지금과 다름")
        if d["code_md5"] != code_md5:
            raise rn.ResumeError(f"{vdir.name}: 끝난 판의 code_md5 ≠ 지금 — 지휘자 결정")
        return "done"
    if vdir.exists():
        rn._move_partial(vdir, out_dir)
    vdir.mkdir(parents=True)
    cf.dump_json(vdir / "spec.json", dict(spec))
    cf.dump_json(vdir / "started.json", {"code_md5": code_md5, "spec_md5": md5, "utc": _utc()})
    return "new"


def finish_board(*, vdir: Path, spec: Mapping, preds: Mapping[str, pd.DataFrame], samples: ev.SampleTable,
                 exp_log: rn.ExperimentLog, code_md5: str, input_md5s: Mapping[str, str], code_root: Path,
                 n_jobs: int, smoke: Mapping | None, thr_preds: Mapping[str, pd.DataFrame] | None = None,
                 tracker=None, stage_run_id: str | None = None, tags: Mapping[str, str] | None = None,
                 params: Mapping[str, Any] | None = None, judgement: str | None = None,
                 artifacts: Sequence[tuple[Path, str | None]] = (), extra_meta: Mapping | None = None,
                 done_extra: Mapping | None = None) -> dict:
    """예측 저장 → CV 채점(test 정답 안 읽음) → 실험 로그('같은 validation 으로 고른 횟수') → MLflow → done.json."""
    name, stage = spec["name"], spec["stage"]
    for s, df in preds.items():
        df.to_parquet(rn.pred_path(vdir, s), index=False)
    for f, df in (thr_preds or {}).items():
        df.to_parquet(rn.pred_path(vdir, rn.THR_PREFIX + f), index=False)
    full = smoke is None
    cv = ev.score_cv({f: preds[f] for f in CV_FOLDS}, samples, thr_preds=thr_preds, require_complete=full)
    count = exp_log.cv_scored(stage, name)
    cf.dump_scores_json(vdir / "cv_scores.json", cv)
    rn.write_h_metrics(vdir, ev.h_metrics_cv({f: preds[f] for f in CV_FOLDS}, samples,
                                             float(cv["threshold"]["threshold"]), require_complete=full))
    cf.save_run_meta(vdir / "run_meta.json", dict(spec), input_md5s, code_root, n_jobs=n_jobs,
                     extra={**dict(extra_meta or {}), "smoke": smoke})
    run_id = None
    if tracker is not None:
        t = {"단계": stage, "역할": spec["role"], "설정 md5": spec_md5(spec), "code_md5": code_md5, **dict(tags or {})}
        run_id = tracker.start_variant(s3.EXPERIMENT, stage_run_id, name, t, dict(params or {}))
        if judgement:
            tracker.set_judgement(run_id, judgement)
        tracker.log_cv(run_id, cv, selection_count=count)
        for f in ("spec.json", "run_meta.json", "cv_scores.json", rn.H_METRICS_FILE):
            tracker.log_artifact(run_id, vdir / f)
        for p, sub in artifacts:
            tracker.log_artifact(run_id, p, sub)
        tracker.finish(run_id)
    done = {"name": name, "stage": stage, "spec_md5": spec_md5(spec), "code_md5": code_md5,
            "cv_band_weighted": cv["cv_band_weighted"], "many_ties": cv["ties"]["many_ties"],
            "selection_count": count, "mlflow_run_id": run_id, "smoke": smoke, **dict(done_extra or {})}
    cf.dump_scores_json(vdir / "done.json", done)
    exp_log.append("done", stage=stage, name=name, cv=cv["cv_band_weighted"], count=count)
    return done


# ================================================================ 3-1 VAE(19 모델 — 블록 OOF·엄격 OOF)
def vae_columns(vs: rg.VersionSpec) -> list[rg.Column]:
    """VAE 입력 칼럼 = 트리 대상 판 트리 칼럼 중 숫자(h·g·범주 뺌 — 부록 A-8)."""
    return [c for c in rg.tree_columns(vs) if c.kind == "numeric"]


class TreeTargetVaeFeatures:
    """실제 자료: 트리 대상 판 X(inputs.build_tree_X) 의 숫자 칼럼 → VAE 입력 표, 전처리 = TabularScaler(구현 세부 b)."""

    def __init__(self, ctx, vs: rg.VersionSpec):
        from . import inputs as ip

        self.ip, self.ctx, self.vs = ip, ctx, vs
        self.cols = vae_columns(vs)
        self.names = [c.name for c in self.cols]
        if not self.names:
            raise CombineError("VAE 입력 숫자 칼럼 없음")

    def train_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = self.ip.TreeInputs(self.ctx, self.vs, "refit").transform(rows)
        return X[self.names]

    @contextlib.contextmanager
    def eval_frames(self, split: str):
        asm = self.ip.TreeInputs(self.ctx, self.vs, split)
        with asm.pair_cache():
            yield lambda meta: asm.transform(meta)[self.names]

    def new_scaler(self):
        return self.ip.TabularScaler(self.cols, self.ctx.missing, self.vs.order)


def _x_tab(scaler, X: pd.DataFrame, src_route) -> np.ndarray:
    Z = scaler.transform(X, src_route=np.asarray(src_route, dtype=object))
    return np.ascontiguousarray(Z.to_numpy(dtype=np.float32))


def _train_keys(rows: pd.DataFrame) -> pd.Index:
    return pd.Index(rows["lims_data_key"].astype(str).unique())


def _check_no_leak(name: str, train_keys: pd.Index, target_keys) -> None:
    hit = pd.Index(np.asarray(target_keys, dtype=object)).isin(train_keys)
    if hit.any():
        raise AssertionError(f"{name}: 예측할 행 시료 {int(hit.sum())}개를 학습함(OOF 누수)")


def _check_assembly(plan, split: str, rows: pd.DataFrame, got: pd.DataFrame) -> None:
    """train 행마다 OOF 값 1개·값을 낸 모델 = blocks.assemble_oof(구현 세부 d)."""
    if len(got) != len(rows):
        raise AssertionError(f"{split}: OOF 값 {len(got)} ≠ train 행 {len(rows)}")
    pos = _kh(got).get_indexer(_kh(rows))
    if (pos < 0).any():
        raise AssertionError(f"{split}: OOF 값이 없는 train 행 {int((pos < 0).sum())}개")
    want = np.array([plan[i].name for i in bk.assemble_oof(plan, split, rows["d"])], dtype=object)
    if not np.array_equal(got["model"].to_numpy(dtype=object)[pos], want):
        raise AssertionError(f"{split}: OOF 값을 낸 모델이 블록 계획과 다름")


def build_vae(*, feats, samples: ev.SampleTable, plan: Sequence[bk.BlockModel], W: int,
              rows_of_model: Callable[[bk.BlockModel], pd.DataFrame], rows_of_split: Callable[[str], pd.DataFrame],
              model_factory: Callable[[], Any], seed: int, n_jobs: int, vdir: Path,
              max_pairs_per_h: int | None = None, check: Callable[[], Any] | None = None,
              save_models: bool = True, log: Callable[[str], Any] = print) -> dict:
    """VAE 블록 모델 학습 → train 행 OOF·평가 쌍 값. 돌려줌 = {'oof': {split: 표}, 'pred': {split: 표}, 'models': [...]}.
    rows_of_model·rows_of_split = 학습 행·OOF 대상 행(행 = (시료, h), T·E·S·g·R 포함) — 단계 4~6 은 줄인·뽑은 행을 넘김."""
    check = check or (lambda: None)
    splits = [s for s in SPLIT_ORDER if any(r.split == s for m in plan for r in m.roles)]
    model_rows = {m.name: rows_of_model(m) for m in plan}
    split_rows = {s: rows_of_split(s) for s in splits}
    allrows = pd.concat([*model_rows.values(), *split_rows.values()], ignore_index=True)
    allrows = allrows.loc[~_kh(allrows).duplicated()].reset_index(drop=True)
    t0 = time.time()
    X_all = feats.train_frame(allrows)
    if len(X_all) != len(allrows):
        raise AssertionError("VAE 입력 행 수가 다름")
    kh_all = _kh(allrows)
    log(f"VAE 입력(train 행 합집합) {X_all.shape} — {time.time() - t0:.0f}s")

    def take(rows):
        pos = kh_all.get_indexer(_kh(rows))
        if (pos < 0).any():
            raise AssertionError("VAE 입력 표에 없는 행")
        X = X_all.iloc[pos]
        X.index = pd.RangeIndex(len(pos))
        return X

    oof_parts: dict[str, list[pd.DataFrame]] = {s: [] for s in splits}
    preds: dict[str, pd.DataFrame] = {}
    infos = []
    mdir = vdir / VAE_MODEL_DIR
    for m in plan:
        check()
        t1 = time.time()
        r = model_rows[m.name]
        if not (r["d"].pipe(fd.day_ints) <= max(b for _, b in m.train_intervals)).all():
            raise AssertionError(f"{m.name}: 학습 행이 학습 구간 밖")
        y = r["y"].to_numpy(dtype=np.int64)
        Xm = take(r)
        scaler = feats.new_scaler().fit(Xm)
        model = model_factory()
        model.fit({"x_tab": _x_tab(scaler, Xm, r["src_route"])}, y, seed=seed, n_jobs=n_jobs,
                  groups=(r["sgg"].astype(str).to_numpy(), fd.day_ints(r["d"])))
        del Xm
        tkeys = _train_keys(r)
        tmax = int(fd.day_ints(r["d"]).max())
        info = {"model": m.name, "aliases": m.aliases, "train_intervals": m.intervals_as_dates(),
                "roles": [x.name for x in m.roles], "n_train_rows": int(len(r)), "n_train_pos": int(y.sum()),
                "fit_seconds": round(time.time() - t1, 2)}
        for role in m.roles:
            if role.kind == "oof":
                t = split_rows[role.split]
                tr = t.loc[bk.oof_target_mask(t["d"], role.split, role.block)]
                _check_no_leak(f"{m.name} → {role.name}", tkeys, tr["lims_data_key"].astype(str))
                v = model.score({"x_tab": _x_tab(scaler, take(tr), tr["src_route"])}) if len(tr) else np.empty(0)
                oof_parts[role.split].append(pd.DataFrame({
                    "lims_data_key": tr["lims_data_key"].astype(str).to_numpy(dtype=object),
                    "h": tr["h"].to_numpy().astype(np.int16), "value": np.asarray(v, dtype=np.float64),
                    "model": m.name}))
            else:
                if tmax > fd.SPLITS[role.split].train_end:
                    raise AssertionError(f"{m.name}: 판 전체 모델이 {role.split} train 끝 뒤 시료를 학습")
                keys_l, h_l, p_l = [], [], []
                with feats.eval_frames(role.split) as frame:
                    for h, idx, meta in iter_eval_meta(samples, role.split, W, max_pairs_per_h=max_pairs_per_h,
                                                       seed=seed, check=check):
                        X = frame(meta)
                        p_l.append(model.score({"x_tab": _x_tab(scaler, X, meta["src_route"])}))
                        keys_l.append(samples.keys[idx])
                        h_l.append(np.full(idx.size, h, np.int16))
                        del X
                preds[role.split] = _pred_frame(np.concatenate(keys_l), np.concatenate(h_l), role.split,
                                                np.concatenate(p_l))
        info["seconds"] = round(time.time() - t1, 2)
        if hasattr(model, "get_config"):
            info["config"] = model.get_config()
        if hasattr(scaler, "state"):
            info["scaler"] = scaler.state()
        if save_models and hasattr(model, "save"):
            mdir.mkdir(parents=True, exist_ok=True)
            model.save(mdir / f"{m.name}.pt")
        mdir.mkdir(parents=True, exist_ok=True)
        cf.dump_json(mdir / f"{m.name}.json", cf.nan_to_null(info))
        infos.append({k: v for k, v in info.items() if k not in ("config", "scaler")})
        log(f"VAE {m.name}: 학습 {info['n_train_rows']}행 — {info['seconds']}s")
        del model, scaler
    oof = {}
    for s in splits:
        got = pd.concat(oof_parts[s], ignore_index=True)
        _check_assembly(plan, s, split_rows[s], got)
        oof[s] = got
    return {"oof": oof, "pred": preds, "models": infos}


def write_oof(vdir: Path, oof: Mapping[str, pd.DataFrame]) -> list[Path]:
    out = []
    for s, df in oof.items():
        p = vdir / f"{OOF_PREFIX}{s}.parquet"
        df.to_parquet(p, index=False)
        out.append(p)
    return out


def vae_spec(target_spec: rn.VariantSpec, tree_target: str, plan_counts: Mapping) -> dict:
    """3-1-VAEt 판 설정(VariantSpec 밖 꼴 — 구현 세부 i)."""
    return {"name": VAE_T, "stage": "3-1", "order": s3.ORDER, "kind": "vae", "algorithm": "VAE",
            "W": target_spec.W, "mode": target_spec.mode,
            "input_set": f"{target_spec.input_set} 숫자 칼럼(h·g·범주 뺌)", "role": s3.ROLE_CMP,
            "train_count": int(plan_counts["total"]), "params": {}, "judgement": s3.NA_JUDGE,
            "base_variant": tree_target,
            "extra": {"version": dict(target_spec.extra["version"]), "vae": "표(tab) VAE, 부록 A-8",
                      "block_plan": dict(plan_counts), "score": "robust z(음의 ELBO, 기준 = 조기 종료 묶음 불검출 행)"}}


# ================================================================ ④ LIMS 만 스태킹
def lims_version(W: int, mode: str = "sum") -> rg.VersionSpec:
    """범주 5 + h·g 만(등록부 tree_columns 가 늘 넣는 칼럼 — 단위 없음)."""
    return rg.load_version({"name": X4, "order": 2, "W": int(W), "mode": mode, "include_ids": []})


class LimsTreeFeatures:
    """실제 자료: ④ 기본 트리 입력 = inputs.TreeInputs(LIMS 만, 로지스틱만 scale) + models_tree.make_model(기본 설정)."""

    def __init__(self, ctx, W: int):
        from . import inputs as ip
        from . import models_tree as mtr
        from . import wiring as wr

        self.ip, self.mtr, self.wr = ip, mtr, wr
        self.ctx = ctx
        self.vs = lims_version(W)
        self.cat_cols = wr.tree_cat_cols(self.vs)

    def assembler(self, alg: str, split: str, rows: pd.DataFrame):
        return self.ip.TreeInputs(self.ctx, self.vs, split, scale=(alg == "LR")).fit(rows)

    def model(self, alg: str, asm):
        ind = ([c for c in asm.columns if c.endswith(rg.NA_SUFFIX) or c == rg.STRUCT_COLUMN] if alg == "LR" else [])
        return self.mtr.make_model(self.wr.TREE_ALGO[alg], cat_cols=self.cat_cols, indicator_cols=ind,
                                   max_iter=None, params=None)

    def meta_model(self):
        return self.mtr.make_model("LR", cat_cols=[], indicator_cols=[], max_iter=None, params=None)


def _predict_rows(model, X, dedupe: bool) -> np.ndarray:
    if dedupe:
        rep, inv = rn.unique_rows(X)
        if rep.size < len(X):
            return np.asarray(model.predict_proba(rn.take_rows(X, rep)), dtype=np.float64)[inv]
    return np.asarray(model.predict_proba(X), dtype=np.float64)


def build_x4(*, feats, samples: ev.SampleTable, plan: Sequence[bk.BlockModel], W: int,
             rows_of_model: Callable[[bk.BlockModel], pd.DataFrame], rows_of_split: Callable[[str], pd.DataFrame],
             seed: int, n_jobs: int, vdir: Path, algs: Sequence[str] = TREE6, max_pairs_per_h: int | None = None,
             check: Callable[[], Any] | None = None, save_full_models: bool = True,
             dedupe_algs: Sequence[str] = rn.ROW_DEDUPE_ALGS, log: Callable[[str], Any] = print) -> dict:
    """④: 트리 6종 × 블록 계획 → OOF 6칼럼(train 행)·판 전체 모델 예측 6칼럼(평가 쌍) → 메타 로지스틱(split 마다).
    돌려줌 = {'pred': {split: 메타 예측 표}, 'oof': {split: OOF 표}, 'models': [...], 'meta': {split: 덤프}}."""
    check = check or (lambda: None)
    splits = [s for s in SPLIT_ORDER if any(r.split == s for m in plan for r in m.roles)]
    split_rows = {s: rows_of_split(s) for s in splits}
    oof = {s: pd.DataFrame({"lims_data_key": split_rows[s]["lims_data_key"].astype(str).to_numpy(dtype=object),
                            "h": split_rows[s]["h"].to_numpy().astype(np.int16)}) for s in splits}
    src = {s: np.full(len(split_rows[s]), "", dtype=object) for s in splits}
    ev_pos: dict[str, np.ndarray] = {}
    ev_h: dict[str, np.ndarray] = {}
    ev_p: dict[str, dict[str, np.ndarray]] = {s: {} for s in splits}
    infos = []
    mdir = vdir / X4_MODEL_DIR
    for alg in algs:
        for s in splits:
            oof[s][f"p_{alg}"] = np.nan
        dd = alg in dedupe_algs
        for m in plan:
            check()
            t1 = time.time()
            r = rows_of_model(m)
            y = r["y"].to_numpy(dtype=np.int64)
            asm = feats.assembler(alg, "refit", r)       # 구현 세부 (c) — 모든 행 C ≤ refit train 끝
            model = feats.model(alg, asm)
            model.fit(asm.transform(r), y, seed=seed, n_jobs=n_jobs)
            tkeys = _train_keys(r)
            tmax = int(fd.day_ints(r["d"]).max())
            info = {"algorithm": alg, "model": m.name, "aliases": m.aliases, "roles": [x.name for x in m.roles],
                    "train_intervals": m.intervals_as_dates(), "n_train_rows": int(len(r)),
                    "n_train_pos": int(y.sum()), "fit_seconds": round(time.time() - t1, 2)}
            for role in m.roles:
                if role.kind == "oof":
                    t = split_rows[role.split]
                    msk = bk.oof_target_mask(t["d"], role.split, role.block)
                    tr = t.loc[msk]
                    _check_no_leak(f"{alg} {m.name} → {role.name}", tkeys, tr["lims_data_key"].astype(str))
                    if len(tr):
                        oof[role.split].loc[msk, f"p_{alg}"] = _predict_rows(model, asm.transform(tr), dd)
                        src[role.split][msk] = src[role.split][msk] + f"{alg}:{m.name};"
                else:
                    if tmax > fd.SPLITS[role.split].train_end:
                        raise AssertionError(f"{alg} {m.name}: 판 전체 모델이 {role.split} train 끝 뒤 시료를 학습")
                    pos_l, h_l, p_l = [], [], []
                    with (asm.pair_cache() if hasattr(asm, "pair_cache") else contextlib.nullcontext()):
                        for h, idx, meta in iter_eval_meta(samples, role.split, W, max_pairs_per_h=max_pairs_per_h,
                                                           seed=seed, check=check):
                            p_l.append(_predict_rows(model, asm.transform(meta), dd))
                            pos_l.append(idx)
                            h_l.append(np.full(idx.size, h, np.int16))
                    pos, hh = np.concatenate(pos_l), np.concatenate(h_l)
                    if role.split in ev_pos:
                        if not (np.array_equal(ev_pos[role.split], pos) and np.array_equal(ev_h[role.split], hh)):
                            raise AssertionError(f"{role.split}: 평가 쌍 순서가 알고리즘마다 다름")
                    else:
                        ev_pos[role.split], ev_h[role.split] = pos, hh
                    ev_p[role.split][alg] = np.concatenate(p_l)
            info["seconds"] = round(time.time() - t1, 2)
            if hasattr(model, "config_dump"):
                info["config"] = model.config_dump()
            mdir.mkdir(parents=True, exist_ok=True)
            if save_full_models and m.is_split_full and hasattr(model, "save"):
                model.save(mdir / f"{alg}__{m.name}.pkl")
            cf.dump_json(mdir / f"{alg}__{m.name}.json", cf.nan_to_null(info))
            infos.append({k: v for k, v in info.items() if k != "config"})
            log(f"④ {alg} {m.name}: 학습 {info['n_train_rows']}행 — {info['seconds']}s")
            del model, asm
    for s in splits:   # train 행마다 6칼럼 모두, 알고리즘마다 정확히 한 모델(그 블록) — 구현 세부 d
        if oof[s][list(f"p_{a}" for a in algs)].isna().any().any():
            raise AssertionError(f"{s}: OOF 값이 빠진 train 행")
        per = pd.Series(src[s]).str.count(";").to_numpy()
        if not (per == len(algs)).all():
            raise AssertionError(f"{s}: train 행의 OOF 값 수가 알고리즘 수와 다름")
        want = [plan[i].name for i in bk.assemble_oof(plan, s, split_rows[s]["d"])]
        exp = np.array(["".join(f"{a}:{w};" for a in algs) for w in want], dtype=object)
        if not np.array_equal(src[s], exp):
            raise AssertionError(f"{s}: OOF 값을 낸 모델이 블록 계획과 다름")
    preds, meta_dump = {}, {}
    for s in splits:
        check()
        cols = [f"p_{a}" for a in algs]
        mm = feats.meta_model()
        Xtr = oof[s][cols].astype(np.float64)
        mm.fit(Xtr, split_rows[s]["y"].to_numpy(dtype=np.int64), seed=seed, n_jobs=n_jobs)
        Xev = pd.DataFrame({c: ev_p[s][a] for c, a in zip(cols, algs)})
        p = np.asarray(mm.predict_proba(Xev), dtype=np.float64)
        preds[s] = _pred_frame(samples.keys[ev_pos[s]], ev_h[s], s, p)
        if hasattr(mm, "save"):
            mm.save(vdir / f"meta_{s}.pkl")
        if hasattr(mm, "config_dump"):
            meta_dump[s] = mm.config_dump()
            cf.dump_json(vdir / f"meta_{s}.json", cf.nan_to_null(meta_dump[s]))
    return {"pred": preds, "oof": oof, "models": infos, "meta": meta_dump}


def x4_spec(save_full_models: bool, W: int) -> rn.VariantSpec:
    """W·mode = LIMS 입력 version 의 W·'sum'(window 칼럼 없음 — 경계 계산용, 단계 4 Stack4Provider 가 int(W) 로 읽음)."""
    return rn.VariantSpec(
        name=X4, stage="3-2", order=s3.ORDER, kind="tree", algorithm="LR", W=int(W), mode="sum",
        input_set="x4-LIMS(범주 5 + h·g) 스태킹", role=s3.ROLE_REPL, train_count=X4_TRAIN_COUNT, params={},
        judgement=None, base_variant=None,
        extra={"stack": {"base": list(TREE6), "base_input": "범주 5 + h·g", "base_params": "라이브러리 기본(반복 수 포함)",
                         "oof": "§7-3 fold 안 블록·전역 블록(같은 행 모델 재사용)", "meta": "로지스틱(입력 p_<트리> 6칼럼, 표준화)",
                         "save_full_models": bool(save_full_models)}, "cat_auto_lr": True})


# ================================================================ 3-3 앙상블(학습 없음)
def _aligned(frames: Sequence[pd.DataFrame], split: str) -> list[np.ndarray]:
    base = frames[0]
    kb = _kh(base)
    out = [base["pred"].to_numpy(dtype=np.float64)]
    for f in frames[1:]:
        if len(f) != len(base):
            raise CombineError(f"{split}: 구성 판 예측 쌍 수가 다름")
        pos = _kh(f).get_indexer(kb)
        if (pos < 0).any():
            raise CombineError(f"{split}: 구성 판 예측 쌍이 다름")
        out.append(f["pred"].to_numpy(dtype=np.float64)[pos])
    return out


def _pct(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(v, method="average") / float(len(v))


def ensemble_preds(member_dirs: Sequence[Path]) -> dict:
    """3-3: {'pred': {split: 평균}, 'thr': {fold: 평균}, 'pct': {...}, 'pct_thr': {...}, 'quantiles': 표}(구현 세부 h)."""
    if len(member_dirs) != 3:
        raise CombineError("3-3 은 트리 3개")
    out = {"pred": {}, "thr": {}, "pct": {}, "pct_thr": {}}
    q_rows = []
    for s in SPLIT_ORDER:
        fr = [pd.read_parquet(rn.pred_path(d, s)) for d in member_dirs]
        vals = _aligned(fr, s)
        keys, h = fr[0]["lims_data_key"].to_numpy(dtype=object), fr[0]["h"].to_numpy()
        out["pred"][s] = _pred_frame(keys, h, s, np.mean(vals, axis=0))
        out["pct"][s] = _pred_frame(keys, h, s, np.mean([_pct(v) for v in vals], axis=0))
        for d, v in zip(member_dirs, vals):
            q = np.percentile(v, [5, 50, 95])
            q_rows.append({"split": "test" if s == "refit" else s, "member": Path(d).name, "q05": q[0], "q50": q[1],
                           "q95": q[2], "n_pairs": int(len(v))})
        if s in CV_FOLDS:
            tf = []
            for d in member_dirs:
                p = rn.pred_path(d, rn.THR_PREFIX + s)
                tf.append(pd.read_parquet(p) if p.exists() else pd.read_parquet(rn.pred_path(d, s)))
            tv = _aligned(tf, s)
            if not _kh(tf[0]).equals(_kh(fr[0])):
                pos = _kh(tf[0]).get_indexer(_kh(fr[0]))
                tv = [x[pos] for x in tv]
            out["thr"][s] = _pred_frame(keys, h, s, np.mean(tv, axis=0))
            out["pct_thr"][s] = _pred_frame(keys, h, s, np.mean([_pct(v) for v in tv], axis=0))
    out["quantiles"] = pd.DataFrame(q_rows)
    return out


def t3_spec(members: Sequence[Mapping]) -> dict:
    return {"name": T3, "stage": "3-3", "order": s3.ORDER, "kind": "ensemble", "algorithm": "T3", "W": None,
            "mode": None, "input_set": "트리 상위 3개 (시료, h) 확률 단순 평균", "role": s3.ROLE_CAND, "train_count": 0,
            "params": {}, "judgement": None, "base_variant": None,
            "extra": {"members": [dict(m) for m in members]}}


# ================================================================ 판정(§6 2차 공통 (가)→(나))
def judge_step(*, stacking: Sequence[ev.Candidate], replacement: Sequence[ev.Candidate], current: ev.Candidate,
               rescore: Callable[[str], float]) -> dict:
    """(가) 쌓는 후보 val 1위 대 현재 최고 판 → (나) 교체 후보 val 1위 대 그때의 현재 최고 판. 소수 4자리, 같으면 안 오름."""
    with ev.selection_guard():
        res = {"current_before": current.name, "stacking": None, "replacement": None}
        cur = current
        for kind, cands in (("stacking", stacking), ("replacement", replacement)):
            if not cands:
                continue
            ranked = ev.rank_candidates(cands)
            top = ranked[0]
            j = ev.judge_adoption(top, cur, rescore)
            res[kind] = {"ranking": [c.name for c in ranked], "cv": {c.name: c.cv for c in ranked}, "top": top.name,
                         "against": cur.name, "judge": j}
            if j["adopt"]:
                cur = top
        res["current_after"] = cur.name
        return res


def judgement_tags(res: Mapping) -> dict[str, tuple[str, str]]:
    """판 이름 → (판정 tag, 판정 상대)."""
    out = {}
    for kind in ("stacking", "replacement"):
        r = res.get(kind)
        if not r:
            continue
        for n in r["ranking"]:
            if n == r["top"]:
                out[n] = ("채택" if r["judge"]["adopt"] else "버림", r["against"])
            else:
                out[n] = ("버림(단계 1위 아님)", r["against"])
    return out


def rescore_from(dir_of: Mapping[str, Path], samples: ev.SampleTable) -> Callable[[str], float]:
    def f(name: str) -> float:
        return float(ev.score_cv(rn.load_preds(dir_of[name], CV_FOLDS), samples, rescore=True,
                                 require_complete=False)["cv_band_weighted"])
    return f


def append_chain(inputs_root: Path, chain_rel: str, line: Mapping, tracker=None) -> None:
    p = resolve(inputs_root, chain_rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(line), ensure_ascii=False) + "\n")
    if tracker is not None:
        tracker.log_artifact(tracker.chain_run(), p)


def set_judgements(tracker, run_ids: Mapping[str, str | None], tags: Mapping[str, tuple[str, str]],
                   rescore: Mapping | None = None) -> None:
    if tracker is None:
        return
    for n, (j, against) in tags.items():
        rid = run_ids.get(n)
        if not rid:
            raise CombineError(f"MLflow 판 run 없음: {n}")
        if "판정" in tracker.client.get_run(rid).data.tags:
            continue
        tracker.set_judgement(rid, j, against=against)
        if rescore and n in rescore and rescore[n] is not None:
            tracker.log_rescore(rid, float(rescore[n]))


def pointer_run(tracker, stage: str, name: str, src_name: str, src_dir: Path, *, role: str, code_reviewed: bool,
                spec: rn.VariantSpec, code_md5: str, selection_count: int) -> str | None:
    """다시 쓰는 판 = 새 run + tag '원 판'(부록 A-15) — 예측은 원 판 run artifact 를 가리킴."""
    if tracker is None:
        return None
    sr = tracker.stage_run(s3.EXPERIMENT, stage)
    r = tracker.find_run(s3.EXPERIMENT, name, sr)
    if r is not None and r.info.status == "FINISHED":
        return r.info.run_id
    tags = {"단계": stage, "역할": role, "원 판": src_name, "코드 검수 완료": "예" if code_reviewed else "아니오",
            "설정 md5": spec.md5(), "code_md5": code_md5}
    rid = tracker.start_variant(s3.EXPERIMENT, sr, name, tags, {"source_variant": src_name})
    tracker.log_cv(rid, rn._load_scores(src_dir / "cv_scores.json"), selection_count=selection_count)
    tracker.finish(rid)
    return rid


# ================================================================ 단계 4 가 읽는 사슬 상태(lst/stage4 '사슬 상태' 꼴)
def _anomaly_def(inputs_root: Path, spec: rn.VariantSpec) -> dict | None:
    """MemberDef.anomaly = {column, version(3-1 VAE 입력 = 트리 대상 판의 현재 입력), W, mode, source}."""
    a = (spec.extra or {}).get("anomaly")
    if not a:
        return None
    v = rn.read_json(resolve(inputs_root, a["dir"]) / VAE_T / "spec.json")
    return {"column": a["column"], "version": dict(v["extra"]["version"]), "W": int(v["W"]), "mode": v["mode"],
            "source": VAE_T}


def _member_def(inputs_root: Path, name: str, rel_dir: str) -> dict:
    sp = source_spec(inputs_root, rel_dir)
    return {"name": name, "dir": rel_dir.rstrip("/"), "anomaly": _anomaly_def(inputs_root, sp), "stack4": None,
            "row_plan": None, "catenc_k": None}


def best_def(inputs_root: Path, ref: Mapping) -> dict:
    """이 모듈의 사슬 상태 판 정보(current_best·tree_target) → lst/stage4 BestDef(구성 판 = spec.json 이 있는 원 판 폴더)."""
    spec = rn.read_json(resolve(inputs_root, ref["dir"]) / "spec.json")
    ext = spec.get("extra") or {}
    if spec["kind"] == "ensemble":
        kind = "ensemble"
        members = [_member_def(inputs_root, m["source"] or m["name"], m["pred_dir"]) for m in ext["members"]]
    elif "stack" in ext:
        kind = "stack4"
        members = [{"name": ref["name"], "dir": ref["dir"].rstrip("/"), "anomaly": None,
                    "stack4": {"version": {"include_ids": []}}, "row_plan": None, "catenc_k": None}]
    elif spec["kind"] in ("tree", "dl"):
        kind = spec["kind"]
        members = [_member_def(inputs_root, ref["name"], ref["dir"])]
    else:
        raise CombineError(f"단계 4 사슬 상태로 못 바꾸는 판 종류: {spec['kind']} ({ref['name']})")
    return {"name": ref["name"], "kind": kind, "dir": ref["dir"].rstrip("/"),
            "replacement": bool(ref.get("replacement", False)), "members": members}


def tree_rank_3_0(inputs_root: Path, dirs: Mapping[str, str]) -> dict[str, int]:
    """3-0 트리 6종 CV 순위(1~, §1-2 동점 규칙) — ⑥⑦ 판정 파일의 tree_rank_3_0. 사슬 상태 칸 이름은 lst/stage4 와 맞춰
    'tree_rank_3_0b' 그대로(뜻 = 3-0 CV 순위 — v78 '앙상블이면 구성 트리 중 3-0 CV 1위', 구현 세부)."""
    x67 = read_decision(inputs_root, dirs["x67-judge"], s3.X67_DECISION_FILE)
    names = list(x67["tree_rank_3_0"])
    if len(names) != len(TREE6):
        raise CombineError("3-0 트리 순위가 6 판이 아님")
    return {x67["boards_3_0"][n]["structure"]: i for i, n in enumerate(names, 1)}


tree_rank_3_0b = tree_rank_3_0          # 옛 이름(stage4 칸 이름과 같은 뜻으로 부르던 곳)


def stage4_state(inputs_root: Path, dirs: Mapping[str, str], state: Mapping, step: str, chain_rel: str) -> dict:
    """판정 뒤 사슬 상태 → lst/stage4 load_state 꼴(step·current_best·tree_target·tree_top3·tree_rank_3_0b·history)."""
    p = resolve(inputs_root, chain_rel)
    hist = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
    return {"step": step, "current_best": best_def(inputs_root, state["current_best"]),
            "tree_target": best_def(inputs_root, state["tree_target"]),
            "tree_top3": [t["name"] for t in state["tree_top3"]],
            "tree_rank_3_0b": tree_rank_3_0(inputs_root, dirs), "history": hist}


def write_stage4_state(inputs_root: Path, dirs: Mapping[str, str], state: Mapping, step: str, chain_rel: str,
                       state_rel: str) -> Path:
    """stage4.save_state(꼴 점검)로 씀. 이미 있으면 같은 내용(history 제외)인지만 확인 — 다르면 멈춤."""
    from . import stage4 as s4

    p = resolve(inputs_root, state_rel)
    st = stage4_state(inputs_root, dirs, state, step, chain_rel)
    if p.exists():
        old = s4.load_state(p)
        new = json.loads(json.dumps(cf.nan_to_null({k: v for k, v in st.items() if k != "history"}), ensure_ascii=False))
        if {k: v for k, v in old.items() if k != "history"} != new:
            raise CombineError(f"이미 있는 사슬 상태 파일이 지금 판정과 다름: {p}")
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    return s4.save_state(p, st)
