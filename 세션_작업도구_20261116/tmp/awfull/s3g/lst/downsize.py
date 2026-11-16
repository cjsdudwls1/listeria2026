"""단계 4 다운사이징 — 계획서 §6 단계 4(줄이기 칼럼·줄이기·4-a·4-a0·미리보기), §6 2차 공통('단계 4~6 을 현재 최고
판에 하는 법' — 줄인 자료로 값 모델 다시 학습), 부록 A-6(Theil's U 대상·고유값 비율), B-8(Theil's U), 부록 A-9(4-a
메모리 판정), B-1·B-2(메모리·디스크 기반 학습 예외 — md5 이름표), §2-1(`prduct_nm` 등 = 행 고르기에만).

줄이기(fold 1~3·refit 마다 따로)
- 대상 칼럼 = 단계 0-1 이 고정한 목록(`s0/0-1/apply/14_theil_target_columns.json` — mnfctur_cmpny_nm 뺌) 중
  그 판 train 에서 한 값뿐·고유값 비율(고유값 수 ÷ 행 수) 0.5 초과인 칼럼을 뺀 것.
- 고유값 비율·Theil's U = 줄이기 전 그 판 train(d ≤ train 끝)의 시료 단위(h 복제 없음), U = U(y | x)(`theil.theils_u`),
  숫자도 값 그대로 범주, NaN 은 한 범주(계산·줄이기 모두).
- 줄이기 칼럼 = 그 판 Theil's U 1위(동점 = 고유값 수 적은 쪽 → 이름 순) — 판마다 다르면 판마다 그 칼럼.
- 줄이기 = 그 판 train 기간에 검출 0건인 칼럼 값의 행을 train 에서만 뺌(검출 1건 이상 값은 남김), validation·test 는
  그대로(이 모듈은 train 행만 다룸). 칼럼 값은 원 시료 파일 값 그대로(제품명 = 문자열 그대로).
- 블록 OOF 값 모델(줄인 자료로 다시 학습 — `blocks.block_plan(reuse=False)`)은 검출 0건 값 판정을 그 모델의 학습 블록
  시료로만 다시 정함(칼럼은 그 판 것 그대로) → `block_model_mask`.

4-a·4-a0·①·미리보기 학습 행(`replicate_h`)
- 4-a = 줄인 train 시료마다 h 1~60 전부(트리 부분만), 메모리 판정으로 m 이 정해지면 시료마다 자기 h + 아래 식의
  m − 1 개:  np.random.default_rng(int.from_bytes(hashlib.md5(f'{lims_data_key}|m42'.encode('utf-8')).digest()[:8],
  'big')).choice([x for x in range(1, 61) if x != h], m - 1, replace=False)
- 4-a0 = 시료당 자기 h 1개(딥러닝 단독·결합 모델 판은 4-a = 4-a0), ① = h 1(T = d − 1).
- 미리보기 4-a 는 같은 함수·같은 규칙(메모리 판정도 따로 한 번).

메모리 판정(4-a, 부록 A-9·§6 단계 4)
- 단계 4 시작 때 fold 1~3·refit 중 줄인 뒤(h 복제 포함) 행 수가 가장 큰 train(`largest_split`)으로 학습 1회 시도 →
  메모리 오류면 다음 단계: float32 → 디스크 기반 학습(XGBoost 만, 아니면 건너뜀) → m = 30 → 20 → 10(앞 단계는 켠 채로
  더함). m = 10 도 모자라면 `MemoryLadderExhausted`(부록 B-1 '학습이 멈추면' — 메모리를 확보해 그 판부터 다시).
- 정한 값은 그 판의 모든 fold·refit·값 모델 추론과 그 위 판(5·6·단계 8)에 같게 씀(다시 정하지 않음).
- 여러 트리 판(앙상블·스태킹·④)은 구성 트리마다 판정 → `combine_settings`(디스크 기반 = XGBoost 구성만, m = 구성 중
  가장 작은 값을 판 전체에). 판 이름 끝에 -m30 등(`name_with_m`).

공개 API(0-9·단계 4 스크립트가 씀)
    THEIL_CANDIDATES, SAMPLE_FILE_REL, M_LADDER
    load_theil_targets(path) -> list[str]
    read_sample_columns(sample_file, keys, columns) -> DataFrame       # 원 시료 칼럼(keys 순서, 값 그대로)
    ReductionData.build(known_frame, cols_frame) / ReductionData.load(known_frame, sample_file, columns)
    theil_table(values: DataFrame, y, columns) -> DataFrame             # 칼럼별 고유값 수·비율·제외 이유·U·순위
    pick_column(table) -> str
    zero_detection_keep(values, y) -> (keep bool, value_stats DataFrame)
    reduce_split(data, split) -> SplitReduction                        # 판 하나(Theil 표·칼럼·남긴 키·셈)
    plan_reductions(data, splits=SPLIT_ORDER) -> dict[str, SplitReduction]
    SplitReduction.keep_mask(keys) -> bool / .summary() -> dict / save(red, out_dir) / load_kept_keys(out_dir, split)
    reduced_train_rows(known_frame, split, red) -> DataFrame            # folds.make_train_rows 뒤 남긴 시료만(h 1개)
    block_model_mask(model, data, reductions) -> bool (N,)             # 줄인 자료 블록 값 모델의 학습 시료
    replicate_h(rows, mode, m=None) -> DataFrame                       # mode '4-a'·'4-a0'·'x1', h·T 다시
    m_h_values(key, h, m) -> int64 (m,)                                # 자기 h + m−1 개(계획서 식)
    replicated_counts(reductions, m=None) -> dict[split, int] / largest_split(reductions) -> str
    MemorySetting(float32, disk, m, level) / memory_ladder(algo, x_is_float32=True) -> list[MemorySetting]
    decide_memory(attempt, *, algo, x_is_float32=True, log=None) -> MemoryDecision          # 같은 프로세스
    decide_memory_in_child(func_ref, kwargs, *, algo, log_path, run_id, x_is_float32=True, timeout=None) -> MemoryDecision
        # func_ref(**kwargs, setting=MemorySetting) 를 하위 프로세스(spawn)로 단계마다 1회 — 시간·최대 RSS 기록
    combine_settings(decisions: Mapping[str, MemoryDecision]) -> dict / name_with_m(name, m) -> str
    to_float32(X) -> DataFrame
    disk_cache_tag(parts: Mapping) -> str
    XGBDiskModel(max_iter=…, params=…).fit_chunks(chunks, *, cache_root, tag, seed, n_jobs) -> self
        # XGBoost 외부 메모리(ExtMemQuantileDMatrix) 학습 — 범주 칼럼 없음(단계 4~6 범주 숫자 인코딩),
        # predict_proba(X, n_iter)·save/load 는 models_tree.XGBModel 그대로

구현 세부(계획서에 없음 — 담당자 확인):
 (1) Theil's U 1위 비교 = 소수 4자리 반올림(metrics.round4) 뒤 비교(§1-2 '모든 점수 비교 … 4자리 반올림 … 같을 때 …
     Theil's U 칼럼 = 고유값 수 적은 쪽 → 이름 순' 을 Theil's U 에도 적용한 것으로 읽음). 원 값도 표에 남김.
 (2) 고유값 수 = NaN 을 한 값으로 셈(A-6 'NaN 이면 한 범주로 셈' 을 고유값 수·한 값뿐 판정에도 적용).
     고유값 비율 0.5 '초과' 만 뺌(= 0.5 는 남김).
 (3) 대상 칼럼 값은 원 시료 파일(`리스테리아ML/data/listeria_data.parquet`) 값 그대로 — src_route 도 원 파일 값(RR1 로
     국내로 옮긴 78행의 'overseas' 그대로), train_prdlst_code 등 빈 칸 = NaN(시료 표의 '코드없음' 아님 — 같은 묶음).
     범주(dictionary) 칼럼은 문자열(object)로 바꿔 읽음(값 그대로), lat·lon 은 float32 값 그대로(등호 묶음).
 (4) 판 train = d ≤ 그 판 train 끝인 시료(단계 0-1 시료 표 — RR 규칙 뒤), test 기간 y 는 읽지 않음(known_frame).
 (5) 복제 행 순서 = 원 행 순서, 시료 안은 h 오름차순. 4-a 입력 행의 h 는 자기 h(부록 A-1)여야 함(아니면 멈춤),
     E·S·g·R 칼럼이 이미 있으면 멈춤(복제 뒤 runner.add_bounds 로 다시).
 (6) 메모리 판정 단계: 트리 입력 X 는 이미 float32(학습 틀 인터페이스)라 '기본 시도' = 'float32' 한 번으로 봄
     (x_is_float32=True — 기록에 float32 = True·'입력이 이미 float32'). x_is_float32=False 면 기본 → float32 따로.
     메모리 오류 = 하위 프로세스 안 MemoryError·메모리 문구 예외(guard 'memory')·신호로 죽음('killed').
     그 밖 실패('error'·'crash'·'timeout')는 판정을 멈춤(TrainingHalted — 메모리 단계로 넘기지 않음).
     단계마다 시도 전 resources.check_resources(가용 RAM 15GB·디스크 10GB 바닥값).
 (7) 최대 RSS = 하위 프로세스의 ru_maxrss(리눅스, GiB) — 윈도는 psutil peak_wset, 없으면 None.
 (8) 디스크 기반: 묶음(chunk)마다 float32 .npy(X)·int8 .npy(y)를 `<cache_root>/xgbext_<md5>/` 에 쓰고(이름표 md5 =
     부르는 쪽이 준 code_md5·입력 md5·판 설정·판(split)·메모리 단계 json), ExtMemQuantileDMatrix(max_bin 등은 sklearn
     XGBClassifier 와 같은 기본) + xgboost.train(XGBClassifier.get_xgb_params() 그대로, 반복 수 = 같은 기본/최대 반복)
     → 학습 뒤 묶음 파일 지움(keep_cache=False). 같은 이름표 폴더가 이미 있으면 멈춤(덮어쓰지 않음).
     메모리 안 학습과 예측이 같은지는 합성 자료로 시험(같음 — 큰 자료에서 분위 경계가 다를 수 있음).
 (9) 가장 큰 train 이 같은 수면 판 순서(fold1 → refit) 앞 판.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import guard as gd
from . import resources as rs
from .blocks import SPLIT_ORDER, BlockModel
from .folds import SPLITS, day_ints, make_train_rows, train_mask
from .hassign import H_MAX, assign_h_many
from .metrics import round4
from .models_tree import BOOSTING, XGBModel, library_versions
from .theil import codes_with_nan, theils_u

# 부록 A-6 Theil's U 대상 후보(계획서 나열 순)
THEIL_CANDIDATES = ("hazard_label", "src_route", "data_source", "expr_iem_nm", "inspct_instt_code", "prdlst_code",
                    "train_prdlst_code", "train_prdlst_nm", "train_prdlst_code_2", "train_prdlst_nm_2", "prduct_nm",
                    "date_basis", "addr", "addr_source", "legal_dong_code", "lat", "lon")
# 대상이 될 수 없는 칼럼(A-6: lims_data_key·정답 관련 5칼럼·mnfctur_de, mnfctur_cmpny_nm — 읽지도 않음)
FORBIDDEN_COLUMNS = ("lims_data_key", "judge_result", "result_value", "result_value_num", "judge_stdr",
                     "judge_stdr_num", "mnfctur_de", "mnfctur_cmpny_nm")
UNIQUE_RATIO_MAX = 0.5
SAMPLE_FILE_REL = "리스테리아ML/data/listeria_data.parquet"
M_LADDER = (30, 20, 10)
M_SALT = "m42"
REPLICATE_MODES = ("4-a", "4-a0", "x1")
BOUND_COLS = ("E", "S", "g", "R")


# ================================================================ 대상 칼럼·원 시료 칼럼
def _check_targets(columns: Sequence[str]) -> list[str]:
    cols = list(columns)
    if not cols or len(set(cols)) != len(cols):
        raise ValueError("Theil's U 대상 칼럼은 비지 않고 겹치지 않게")
    bad = [c for c in cols if c in FORBIDDEN_COLUMNS]
    if bad:
        raise ValueError(f"Theil's U 대상이 될 수 없는 칼럼(부록 A-6): {bad}")
    unk = [c for c in cols if c not in THEIL_CANDIDATES]
    if unk:
        raise ValueError(f"부록 A-6 후보에 없는 칼럼: {unk}")
    return cols


def load_theil_targets(path: str | os.PathLike) -> list[str]:
    """단계 0-1 결과 `14_theil_target_columns.json`({"columns": [...], "note": ...}) → 칼럼 목록(파일 순서)."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "columns" not in obj:
        raise ValueError("columns 칸 없음")
    extra = set(obj) - {"columns", "note"}
    if extra:
        raise ValueError(f"모르는 칸: {sorted(extra)}")
    return _check_targets(obj["columns"])


def read_sample_columns(sample_file: str | os.PathLike, keys: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    """원 시료 파일에서 lims_data_key + 대상 칼럼만 읽어 keys 순서로(조인 many_to_one, 빠진 키면 멈춤 — B-6).
    범주(dictionary) 칼럼은 object 문자열로, 숫자는 값 그대로. 정답 칼럼은 읽지 않음."""
    import pyarrow.parquet as pq

    cols = _check_targets(columns)
    tbl = pq.read_table(sample_file, columns=["lims_data_key", *cols])
    df = tbl.to_pandas()
    for c in cols:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object)
    if df["lims_data_key"].duplicated().any():
        raise ValueError("원 시료 파일 lims_data_key 중복")
    left = pd.DataFrame({"lims_data_key": np.asarray(keys, dtype=object)})
    if left["lims_data_key"].duplicated().any():
        raise ValueError("keys 중복")
    out = left.merge(df, on="lims_data_key", how="left", validate="many_to_one", indicator=True)
    miss = int((out["_merge"] != "both").sum())
    if miss:
        raise ValueError(f"원 시료 파일에 없는 lims_data_key {miss}개")
    return out.drop(columns="_merge").reset_index(drop=True)


@dataclass
class ReductionData:
    """줄이기 계산용 시료 단위 자료(d ≤ 2025-06-30 — test 정답 없음). 순서 = known_frame 순서."""
    keys: np.ndarray            # object (N,)
    d: np.ndarray               # 일수 int64 (N,)
    y: np.ndarray               # int64 0/1 (N,)
    values: pd.DataFrame        # 대상 칼럼 원 값(행 = keys 순서, index 0..N-1)
    columns: list[str]

    @classmethod
    def build(cls, known_frame: pd.DataFrame, cols_frame: pd.DataFrame) -> "ReductionData":
        """known_frame = evaluate.SampleTable.known_frame()(lims_data_key·d·y), cols_frame = read_sample_columns 결과."""
        keys = known_frame["lims_data_key"].astype(str).to_numpy(dtype=object)
        if pd.Index(keys).has_duplicates:
            raise ValueError("lims_data_key 중복")
        d = day_ints(known_frame["d"])
        if (d > SPLITS["refit"].train_end).any():
            raise ValueError("d > 2025-06-30 시료(test 기간)가 들어 있음 — known_frame 만(§1-3)")
        yv = known_frame["y"]
        if yv.isna().any() or not np.isin(yv.to_numpy(), (0, 1)).all():
            raise ValueError("y 는 0/1(결측 없음)")
        if list(cols_frame["lims_data_key"].astype(str)) != list(keys):
            raise ValueError("cols_frame 키 순서가 known_frame 과 다름")
        cols = _check_targets([c for c in cols_frame.columns if c != "lims_data_key"])
        return cls(keys=keys, d=d, y=yv.to_numpy().astype(np.int64),
                   values=cols_frame[cols].reset_index(drop=True), columns=cols)

    @classmethod
    def load(cls, known_frame: pd.DataFrame, sample_file, columns: Sequence[str]) -> "ReductionData":
        return cls.build(known_frame, read_sample_columns(sample_file, known_frame["lims_data_key"].astype(str).tolist(),
                                                          columns))

    def __len__(self) -> int:
        return int(self.keys.shape[0])


# ================================================================ Theil's U 표·1위
def n_unique_with_nan(x) -> int:
    c = codes_with_nan(x)
    return int(c.max()) + 1 if c.size else 0


def theil_table(values: pd.DataFrame, y, columns: Sequence[str]) -> pd.DataFrame:
    """칼럼별 행 수·고유값 수(NaN 한 값)·고유값 비율·제외 이유·U(y | x)·4자리 U·순위(대상만, 1 = 줄이기 칼럼)."""
    cols = _check_targets(columns)
    yy = np.asarray(y, dtype=np.int64)
    n = yy.shape[0]
    if n == 0 or len(values) != n:
        raise ValueError("values·y 길이")
    rows = []
    for c in cols:
        x = values[c].to_numpy()
        nu = n_unique_with_nan(x)
        ratio = nu / n
        if nu <= 1:
            reason = "한 값뿐"
        elif ratio > UNIQUE_RATIO_MAX:
            reason = "고유값 비율 0.5 초과"
        else:
            reason = ""
        u = float(theils_u(yy, x))
        rows.append({"column": c, "n_rows": n, "n_pos": int(yy.sum()), "n_unique": nu, "unique_ratio": ratio,
                     "n_nan": int(pd.isna(pd.Series(x)).sum()), "excluded": reason, "eligible": reason == "",
                     "theil_u": u, "theil_u_r4": round4(u)})
    t = pd.DataFrame(rows)
    el = t[t["eligible"]].sort_values(["theil_u_r4", "n_unique", "column"], ascending=[False, True, True], kind="stable")
    t["rank"] = pd.array([pd.NA] * len(t), dtype="Int64")
    t.loc[el.index, "rank"] = np.arange(1, len(el) + 1)
    return t.sort_values(["rank", "column"], na_position="last", kind="stable").reset_index(drop=True)


def pick_column(table: pd.DataFrame) -> str:
    top = table[table["rank"] == 1]
    if len(top) != 1:
        raise ValueError("Theil's U 대상 칼럼이 없음(모두 한 값뿐·고유값 비율 0.5 초과)")
    return str(top["column"].iloc[0])


# ================================================================ 줄이기
def zero_detection_keep(values, y) -> tuple[np.ndarray, pd.DataFrame]:
    """그 행들 안에서 검출 0건인 값의 행 = False(뺌), 검출 1건 이상 값 = True. NaN = 한 값.
    돌려주는 표 = 값 번호별 행 수·검출 수(값 자체는 안 남김 — 제품명 등은 보고에 안 씀)."""
    codes = codes_with_nan(values)
    yy = np.asarray(y, dtype=np.int64)
    if codes.shape[0] != yy.shape[0]:
        raise ValueError("길이 다름")
    if codes.size == 0:
        return np.zeros(0, dtype=bool), pd.DataFrame({"n": [], "pos": []})
    K = int(codes.max()) + 1
    n = np.bincount(codes, minlength=K)
    pos = np.bincount(codes, weights=yy, minlength=K).astype(np.int64)
    keep = pos[codes] > 0
    return keep, pd.DataFrame({"n": n, "pos": pos})


@dataclass
class SplitReduction:
    split: str
    column: str
    table: pd.DataFrame          # Theil's U 표(theil_table)
    kept_keys: np.ndarray        # 남긴 train 시료 lims_data_key(입력 순서)
    stats: dict                  # 셈(행·검출 전후, 값 수 등)

    def keep_mask(self, keys) -> np.ndarray:
        return pd.Index(self.kept_keys).get_indexer(pd.Index(np.asarray(keys, dtype=object))) >= 0

    def summary(self) -> dict:
        return {"split": self.split, "column": self.column, **self.stats}


def _reduce(values, y) -> tuple[np.ndarray, dict]:
    keep, vt = zero_detection_keep(values, y)
    yy = np.asarray(y, dtype=np.int64)
    st = {"n_before": int(yy.size), "pos_before": int(yy.sum()), "n_after": int(keep.sum()),
          "pos_after": int(yy[keep].sum()), "n_values": int(len(vt)), "n_values_zero_pos": int((vt["pos"] == 0).sum()),
          "n_values_kept": int((vt["pos"] > 0).sum())}
    if st["pos_after"] != st["pos_before"]:
        raise AssertionError("줄이기로 검출 행이 빠짐")
    return keep, st


def reduce_split(data: ReductionData, split: str) -> SplitReduction:
    """판 하나: 줄이기 전 train(시료 단위)으로 Theil's U 표 → 1위 칼럼 → 검출 0건 값 행 빼기."""
    sp = SPLITS[split]
    m = train_mask(data.d, sp)
    idx = np.flatnonzero(m)
    vals = data.values.iloc[idx].reset_index(drop=True)
    y = data.y[idx]
    table = theil_table(vals, y, data.columns)
    col = pick_column(table)
    keep, st = _reduce(vals[col].to_numpy(), y)
    st.update({"theil_u": float(table.loc[table["column"] == col, "theil_u"].iloc[0]),
               "train_end": str(np.datetime64(int(sp.train_end), "D"))})
    return SplitReduction(split=split, column=col, table=table, kept_keys=data.keys[idx[keep]], stats=st)


def plan_reductions(data: ReductionData, splits: Sequence[str] = SPLIT_ORDER) -> dict[str, SplitReduction]:
    out = {}
    for s in splits:
        if s not in SPLIT_ORDER:
            raise ValueError(f"모르는 판: {s}")
        out[s] = reduce_split(data, s)
    return out


def column_by_split(reductions: Mapping[str, SplitReduction]) -> dict[str, str]:
    """판마다 줄이기 칼럼(판마다 다를 수 있음 — 각자 자기 칼럼)."""
    return {s: r.column for s, r in reductions.items()}


def block_model_mask(model: BlockModel, data: ReductionData, reductions: Mapping[str, SplitReduction]) -> np.ndarray:
    """줄인 자료로 다시 학습하는 블록 값 모델의 학습 시료(§6 2차 공통·부록 A-12): 그 모델의 학습 블록 시료(줄이기 전)
    안에서 검출 0건 값 판정을 다시 함(칼럼 = 그 판 것 그대로, 예측할 블록 정답 안 씀). 계획 = block_plan(reuse=False)."""
    if len(model.roles) != 1:
        raise ValueError(f"{model.name}: 재사용 모델 — 줄인 자료는 block_plan(reuse=False) 로(부록 A-12)")
    split = model.owner.split
    red = reductions[split]
    pool = np.flatnonzero(model.train_mask(data.d))
    v = data.values[red.column].iloc[pool]
    keep, _ = _reduce(v.to_numpy(), data.y[pool])
    out = np.zeros(len(data), dtype=bool)
    out[pool[keep]] = True
    if model.owner.kind == "full":  # 판 train 전체 모델 = 그 판 줄인 train 그대로
        if not np.array_equal(np.sort(data.keys[out].astype(str)), np.sort(red.kept_keys.astype(str))):
            raise AssertionError(f"{model.name}: 판 전체 모델 행이 그 판 줄인 train 과 다름")
    return out


def reduced_train_rows(known_frame: pd.DataFrame, split: str, red: SplitReduction) -> pd.DataFrame:
    """그 판 train 행(시료당 자기 h 1개, folds.make_train_rows) 중 남긴 시료만."""
    if red.split != split:
        raise ValueError("다른 판의 줄이기")
    rows = make_train_rows(known_frame, SPLITS[split])
    keep = red.keep_mask(rows["lims_data_key"].astype(str).to_numpy(dtype=object))
    if int(keep.sum()) != red.stats["n_after"]:
        raise AssertionError("남긴 행 수가 줄이기 셈과 다름")
    return rows.loc[keep]


def save_reduction(red: SplitReduction, out_dir: str | os.PathLike) -> dict[str, Path]:
    """판마다 Theil's U 표(csv)·남긴 키(parquet)·셈(json). 있는 파일은 덮어쓰지 않음."""
    from .config import dump_json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_tab = out / f"theil_{red.split}.csv"
    p_keys = out / f"kept_keys_{red.split}.parquet"
    for p in (p_tab, p_keys):
        if p.exists():
            raise FileExistsError(f"이미 있음(덮어쓰지 않음): {p}")
    red.table.to_csv(p_tab, index=False, encoding="utf-8-sig")
    pd.DataFrame({"lims_data_key": red.kept_keys.astype(str)}).to_parquet(p_keys, index=False)
    p_sum = dump_json(out / f"reduction_{red.split}.json", red.summary())
    return {"table": p_tab, "kept_keys": p_keys, "summary": Path(p_sum)}


def load_kept_keys(out_dir: str | os.PathLike, split: str) -> np.ndarray:
    return pd.read_parquet(Path(out_dir) / f"kept_keys_{split}.parquet")["lims_data_key"].astype(str).to_numpy(dtype=object)


# ================================================================ h 복제
def m_h_values(key: str, h: int, m: int) -> np.ndarray:
    """4-a 메모리 대체: 자기 h + 계획서 식의 m − 1 개(오름차순)."""
    if m not in M_LADDER:
        raise ValueError(f"m 은 {M_LADDER}: {m}")
    h = int(h)
    if not 1 <= h <= H_MAX:
        raise ValueError(f"h 범위: {h}")
    rng = np.random.default_rng(int.from_bytes(hashlib.md5(f'{key}|{M_SALT}'.encode('utf-8')).digest()[:8], 'big'))
    extra = rng.choice([x for x in range(1, 61) if x != h], m - 1, replace=False)
    return np.sort(np.concatenate([[h], np.asarray(extra, dtype=np.int64)])).astype(np.int64)


def replicate_h(rows: pd.DataFrame, mode: str, m: int | None = None, *, key_col: str = "lims_data_key",
                d_col: str = "d") -> pd.DataFrame:
    """학습 행 h 붙이기: '4-a' = h 1~60(m 이면 자기 h + m−1 개), '4-a0' = 자기 h, 'x1' = h 1(①).
    입력 = 시료당 자기 h 1개 행(make_train_rows·reduced_train_rows), 출력 h(int16)·T(datetime64[D]) 다시 계산."""
    if mode not in REPLICATE_MODES:
        raise ValueError(f"mode 는 {REPLICATE_MODES}")
    if m is not None and mode != "4-a":
        raise ValueError("m 은 4-a 만")
    if any(c in rows.columns for c in BOUND_COLS):
        raise ValueError("E·S·g·R 이 이미 있음 — 복제 뒤 runner.add_bounds 로(구현 세부 5)")
    keys = rows[key_col].astype(str).to_numpy(dtype=object)
    if pd.Index(keys).has_duplicates:
        raise ValueError("입력은 시료당 1행")
    h_own = assign_h_many(keys.tolist())
    if "h" in rows.columns and not np.array_equal(rows["h"].to_numpy(dtype=np.int64), h_own):
        raise ValueError("입력 h 가 자기 h(부록 A-1)와 다름")
    if mode == "4-a0":
        rep = np.arange(len(rows))
        hs = h_own
    elif mode == "x1":
        rep = np.arange(len(rows))
        hs = np.ones(len(rows), dtype=np.int64)
    elif m is None:
        rep = np.repeat(np.arange(len(rows)), H_MAX)
        hs = np.tile(np.arange(1, H_MAX + 1, dtype=np.int64), len(rows))
    else:
        rep = np.repeat(np.arange(len(rows)), m)
        hs = np.concatenate([m_h_values(k, h, m) for k, h in zip(keys, h_own)]) if len(rows) else np.empty(0, np.int64)
    out = rows.iloc[rep].copy()
    d = day_ints(out[d_col])
    out["h"] = hs.astype(np.int16)
    out["T"] = (d - hs).astype("datetime64[D]")
    return out.reset_index(drop=True)


def replicated_counts(reductions: Mapping[str, SplitReduction], m: int | None = None) -> dict[str, int]:
    """판마다 줄인 뒤 h 복제 포함 train 행 수(4-a)."""
    per = H_MAX if m is None else int(m)
    return {s: int(r.stats["n_after"]) * per for s, r in reductions.items()}


def largest_split(reductions: Mapping[str, SplitReduction]) -> str:
    """줄인 뒤(h 복제 포함) 행 수가 가장 큰 판(복제 배수는 모든 판 같아 m 과 관계없음, 같으면 판 순서 앞)."""
    cnt = replicated_counts(reductions)
    order = [s for s in SPLIT_ORDER if s in cnt]
    best = max(cnt[s] for s in order)
    return next(s for s in order if cnt[s] == best)


# ================================================================ 메모리 판정
@dataclass(frozen=True)
class MemorySetting:
    level: str = "base"
    float32: bool = False
    disk: bool = False
    m: int | None = None      # None = h 1~60 전부

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryLadderExhausted(RuntimeError):
    """m = 10 까지 해도 메모리 부족 — 부록 B-1 '학습이 멈추면': 메모리를 확보한 뒤 그 판부터 다시."""

    def __init__(self, msg: str, attempts: list[dict]):
        super().__init__(msg)
        self.attempts = attempts


@dataclass
class MemoryDecision:
    algo: str
    setting: MemorySetting
    attempts: list[dict] = field(default_factory=list)
    result: Any = None
    seconds: float | None = None
    peak_rss_gb: float | None = None

    def to_dict(self) -> dict:
        return {"algo": self.algo, "setting": self.setting.to_dict(), "attempts": self.attempts,
                "seconds": self.seconds, "peak_rss_gb": self.peak_rss_gb}


def memory_ladder(algo: str, x_is_float32: bool = True) -> list[MemorySetting]:
    """시도 순서(앞 단계는 켠 채로): (기본) → float32 → 디스크 기반(XGBoost 만) → m 30 → 20 → 10."""
    if algo not in ("XGB", "LGBM", "CAT", "RF", "ET", "LR"):
        raise ValueError(f"트리 알고리즘 이름(models_tree.ALGOS): {algo}")
    out = [] if x_is_float32 else [MemorySetting("base", False, False, None)]
    out.append(MemorySetting("float32", True, False, None))
    disk = algo == "XGB"
    if disk:
        out.append(MemorySetting("disk", True, True, None))
    out += [MemorySetting(f"m{m}", True, disk, m) for m in M_LADDER]
    return out


def _attempt_record(algo: str, s: MemorySetting) -> dict:
    return {"algo": algo, **s.to_dict(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")}


def decide_memory(attempt: Callable[[MemorySetting], Any], *, algo: str, x_is_float32: bool = True,
                  log: Callable[[dict], None] | None = None, check_path: str | os.PathLike | None = None
                  ) -> MemoryDecision:
    """같은 프로세스에서 단계마다 attempt(setting) 1회 — MemoryError·메모리 문구 예외면 다음 단계, 그 밖 예외는 그대로."""
    tried = []
    for s in memory_ladder(algo, x_is_float32):
        if check_path is not None:
            rs.check_resources(check_path)
        rec = _attempt_record(algo, s)
        t0 = time.time()
        try:
            res = attempt(s)
        except BaseException as e:  # noqa: BLE001
            if not gd.is_memory_error(e):
                raise
            rec.update(status="memory", message=str(e)[:500], seconds=round(time.time() - t0, 2))
            tried.append(rec)
            if log:
                log(rec)
            continue
        rec.update(status="ok", seconds=round(time.time() - t0, 2))
        tried.append(rec)
        if log:
            log(rec)
        return MemoryDecision(algo, s, tried, res, seconds=rec["seconds"])
    raise MemoryLadderExhausted(f"{algo}: m = 10 까지 메모리 부족 — 메모리를 확보해 그 판부터 다시(부록 B-1)", tried)


def _peak_rss_gb() -> float | None:
    try:
        import resource  # 리눅스

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(kb) / 2 ** 20 if sys.platform.startswith("linux") else float(kb) / 2 ** 30
    except ImportError:
        pass
    try:
        import psutil

        return float(getattr(psutil.Process().memory_info(), "peak_wset")) / 2 ** 30
    except Exception:  # noqa: BLE001
        return None


def timed_attempt(func_ref: str, kwargs: Mapping[str, Any], setting: Mapping[str, Any]) -> dict:
    """하위 프로세스 안: func_ref(**kwargs, setting=MemorySetting) 를 재고 결과·시간·최대 RSS 를 돌려줌."""
    fn = gd._resolve(func_ref)
    t0 = time.time()
    res = fn(**dict(kwargs), setting=MemorySetting(**setting))
    return {"result": res, "seconds": round(time.time() - t0, 2), "peak_rss_gb": _peak_rss_gb()}


MEMORY_CAUSES = ("memory", "killed")


def decide_memory_in_child(func_ref: str, kwargs: Mapping[str, Any], *, algo: str, log_path: str | os.PathLike,
                           run_id: str, x_is_float32: bool = True, timeout: float | None = None) -> MemoryDecision:
    """단계마다 func_ref(**kwargs, setting=…) 를 spawn 하위 프로세스로 1회(guard.run_in_child) — 메모리 오류('memory')·
    신호로 죽음('killed')이면 다음 단계, 그 밖 실패는 TrainingHalted. 시도마다 log_path 에 JSON 한 줄."""
    log_path = Path(log_path)
    gd._resolve(func_ref)
    tried: list[dict] = []
    for s in memory_ladder(algo, x_is_float32):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rs.check_resources(log_path.parent)
        rec = {"run_id": run_id, **_attempt_record(algo, s)}
        try:
            out = gd.run_in_child("lst.downsize:timed_attempt",
                                  {"func_ref": func_ref, "kwargs": dict(kwargs), "setting": s.to_dict()}, timeout=timeout)
        except gd.ChildFailure as e:
            rec.update(status="failed", cause=e.cause, exitcode=e.exitcode, exc_type=e.exc_type, message=e.message[:2000])
            gd._log(log_path, rec)
            tried.append(rec)
            if e.cause in MEMORY_CAUSES:
                continue
            raise gd.TrainingHalted(f"{run_id}: 메모리 판정 중 '{e.cause}' — 원인을 고친 뒤 다시", tried) from e
        rec.update(status="ok", seconds=out["seconds"], peak_rss_gb=out["peak_rss_gb"])
        gd._log(log_path, rec)
        tried.append(rec)
        return MemoryDecision(algo, s, tried, out["result"], out["seconds"], out["peak_rss_gb"])
    raise MemoryLadderExhausted(f"{run_id}: m = 10 까지 메모리 부족 — 메모리를 확보해 그 판부터 다시(부록 B-1)", tried)


def combine_settings(decisions: Mapping[str, MemoryDecision]) -> dict:
    """여러 트리 판: 구성마다 float32·디스크(XGBoost 구성만)는 각자, m = 구성 중 가장 작은 값(없으면 None)을 판 전체에."""
    if not decisions:
        raise ValueError("구성 판정이 없음")
    ms = [d.setting.m for d in decisions.values() if d.setting.m is not None]
    m = min(ms) if ms else None
    comp = {}
    for name, d in decisions.items():
        if d.setting.disk and d.algo != "XGB":
            raise AssertionError("디스크 기반은 XGBoost 만")
        comp[name] = {"algo": d.algo, "float32": d.setting.float32, "disk": d.setting.disk, "own_m": d.setting.m}
    return {"m": m, "components": comp}


def name_with_m(name: str, m: int | None) -> str:
    """판 이름 끝에 -m30 등(부록 A-14: 이미 -m<숫자> 가 있으면 다시 안 붙임)."""
    import re

    if m is None or re.search(r"-m\d+(-|$)", name):
        return name
    return f"{name}-m{int(m)}"


def to_float32(X: pd.DataFrame) -> pd.DataFrame:
    """숫자 칼럼을 float32 로(이미 float32 면 그대로, 범주·문자 칼럼은 안 건드림)."""
    out = X.copy(deep=False)
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype) and s.dtype != np.float32:
            out[c] = s.astype(np.float32)
    return out


# ================================================================ XGBoost 디스크 기반 학습
def disk_cache_tag(parts: Mapping[str, Any]) -> str:
    """B-2: 이름표 md5 — parts 에 code_md5·입력 md5·판 설정 md5·판·메모리 단계를 넣음(빠지면 멈춤)."""
    need = {"code_md5", "input_md5s", "spec_md5", "split", "setting"}
    miss = need - set(parts)
    if miss:
        raise ValueError(f"이름표에 빠진 칸: {sorted(miss)}")
    return hashlib.md5(json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


class XGBDiskModel(XGBModel):
    """XGBoost 외부 메모리 학습(4-a 메모리 판정의 '디스크 기반'). 범주 칼럼 없음(단계 4~6 범주 숫자 인코딩)."""

    def __init__(self, *, max_iter: int | None = None, params: Mapping[str, Any] | None = None):
        super().__init__(cat_cols=(), indicator_cols=(), max_iter=max_iter, params=params)

    def fit(self, X, y, *, seed: int, n_jobs: int, eval_set=None):  # noqa: D401
        raise TypeError("XGBDiskModel 은 fit_chunks(묶음, cache_root=…, tag=…) 로 학습")

    def fit_chunks(self, chunks: Iterable[tuple[pd.DataFrame, Any]], *, cache_root: str | os.PathLike, tag: str,
                   seed: int, n_jobs: int, keep_cache: bool = False) -> "XGBDiskModel":
        import xgboost as xgb

        if self.estimator_ is not None:
            raise RuntimeError("이미 학습한 객체")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed 는 정수")
        n_jobs = rs.check_n_jobs(n_jobs)
        cdir = Path(cache_root) / f"xgbext_{tag}"
        if cdir.exists():
            raise FileExistsError(f"디스크 기반 캐시 폴더가 이미 있음(덮어쓰지 않음): {cdir}")
        cdir.mkdir(parents=True)
        files: list[tuple[Path, Path]] = []
        n_rows = n_pos = 0
        try:
            for i, (X, y) in enumerate(chunks):
                rs.check_resources(cache_root)
                self._check_X(X, fitting=True)
                if self.columns_ is None:
                    self.columns_ = list(X.columns)
                elif list(X.columns) != self.columns_:
                    raise ValueError("묶음마다 칼럼 목록(이름·순서)이 같아야 함")
                yy = np.asarray(y, dtype=np.int64)
                if yy.shape != (len(X),) or not np.isin(yy, (0, 1)).all():
                    raise ValueError("y 는 0/1, X 와 같은 길이")
                fx, fy = cdir / f"x_{i:05d}.npy", cdir / f"y_{i:05d}.npy"
                np.save(fx, X.to_numpy(dtype=np.float32))
                np.save(fy, yy.astype(np.int8))
                files.append((fx, fy))
                n_rows += len(X)
                n_pos += int(yy.sum())
            if not files or n_pos == 0 or n_pos == n_rows:
                raise ValueError("학습 자료가 비었거나 한 부류뿐")
            cols = self.columns_

            class _Iter(xgb.DataIter):
                def __init__(self, fl, prefix):
                    self._fl, self._i = fl, 0
                    super().__init__(cache_prefix=prefix)

                def next(self, input_data):
                    if self._i == len(self._fl):
                        return False
                    fx_, fy_ = self._fl[self._i]
                    input_data(data=np.load(fx_), label=np.load(fy_).astype(np.float32), feature_names=cols)
                    self._i += 1
                    return True

                def reset(self):
                    self._i = 0

            est = self._make_estimator(int(seed), n_jobs)
            params = {k: v for k, v in est.get_xgb_params().items() if v is not None}
            n_rounds = int(est.get_num_boosting_rounds())
            t0 = time.time()
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                dm = xgb.ExtMemQuantileDMatrix(_Iter(files, str(cdir / "cache")), missing=np.nan, nthread=n_jobs)
                bst = xgb.train(params, dm, num_boost_round=n_rounds)
            del dm
            est.load_model(bst.save_raw("ubj"))
            self.estimator_ = est
            self.categories_ = {}
            self.seed_, self.n_jobs_ = int(seed), n_jobs
            self.versions_ = library_versions()
            self.fit_info_ = {"n_rows": int(n_rows), "n_pos": int(n_pos), "fit_seconds": time.time() - t0,
                              "warnings": [f"{w.category.__name__}: {w.message}" for w in wlist][:50],
                              "disk": {"tag": tag, "n_chunks": len(files),
                                       "bytes": int(sum(a.stat().st_size + b.stat().st_size for a, b in files))}}
            self._after_fit(wlist)
        finally:
            if not keep_cache and cdir.exists():
                shutil.rmtree(cdir)  # 내가 만든 이 폴더만
        return self

    def _extra_dump(self) -> dict:
        return {"disk_based": True}


__all__ = [
    "THEIL_CANDIDATES", "FORBIDDEN_COLUMNS", "SAMPLE_FILE_REL", "M_LADDER", "load_theil_targets", "read_sample_columns",
    "ReductionData", "theil_table", "pick_column", "zero_detection_keep", "SplitReduction", "reduce_split",
    "plan_reductions", "column_by_split", "block_model_mask", "reduced_train_rows", "save_reduction", "load_kept_keys",
    "m_h_values", "replicate_h", "replicated_counts", "largest_split", "MemorySetting", "MemoryDecision",
    "MemoryLadderExhausted", "memory_ladder", "decide_memory", "decide_memory_in_child", "timed_attempt",
    "combine_settings", "name_with_m", "to_float32", "disk_cache_tag", "XGBDiskModel",
]
