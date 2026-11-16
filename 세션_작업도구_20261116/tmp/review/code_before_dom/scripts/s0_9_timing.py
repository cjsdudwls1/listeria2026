# -*- coding: utf-8 -*-
"""s0_9_timing.py — 단계 0-9 시간 측정 → 학습 횟수(부록 A-9)에 곱한 예상 시간·예상 끝 날짜(계획서 §6 단계 0-9, §3 최대
반복, §8 일정, 부록 A-9·A-7(AutoKeras 블록 대응)·A-15(MLflow '0-9-<측정 항목>')·B-1(자원·멈춤)·B-2).

측정(fold 3 — fold 중 가장 큰 train, 서버, 0-6·0-8 뒤). 항목마다 하위 프로세스(spawn, `lst.guard`)로 따로 돌려
시간·최대 RSS·GPU 메모리를 잼. 끝난 항목은 건너뛰고 이어 함(멈춘 항목부터).
  1-1   트리 3종(XGB·LGB·CAT) × W 10·100 × 요약·원값, 라이브러리 기본 설정(1차 입력)        id 1-1-<알고>-W<3자리>-<sum|raw>
  1-2   LSTM W 10·100(1차 입력)                                                                id 1-2-LSTM-W<3자리>
        → 가장 느린 트리 = 1-1 W 100 원값 측정 시간이 가장 긴 알고리즘(대리 설정에 씀)
  2차 사슬 부스팅 학습 1회씩(XGB·LGB·CAT, 대리 설정 = W 100 원값·§5 후보 전부·k 50·N_ma 15), 최대 2,000 반복·56 시점
        채점 → fold 3 최적 N 이 마지막 시점이면 최대 반복 2배로 다시(상한 16,000 — §3)           id chain-<알고>-it<5자리>
  그룹 중요도 1회(가장 느린 트리, 그 알고리즘 사슬 모델·fold 3 최적 N)                          id imp-tree-<알고>
  Optuna 시도 1회(가장 느린 트리, 판정한 최대 반복, `tuning.run_optuna(stop_after=1)`)          id optuna-<알고>
  RF·ExtraTrees·로지스틱 1 fold                                                                 id fold-<RF|ET|LR>
  딥러닝·결합 구조마다 1 fold(LSTM·GRU·TRF·LSTMA·TCN·CMB·CMBV, 대리 = 딥러닝 W 100·§5 후보 전부) id dl-<구조>
  딥러닝 1 구조 그룹 중요도 1회 · RF·로지스틱 그룹 중요도 1회(ExtraTrees = RF 측정값)           id imp-dl-<구조>·imp-RF·imp-LR
  딥러닝 1 구조 블록 학습 1회(블록 OOF 모델 중 학습 시료가 가장 많은 것)                        id block-<구조>
  AutoKeras 구조별 대응 확인(부록 A-7 대상 전부 — 대응표 = 설정 autokeras.mapping, 담당자 결정 11-07 =
        `tuning.AK_PROPOSAL` 그대로 고정, 실험 로그에 기록) + 시도 1회 시간                                                   id ak-check-<구조>·ak-trial-<구조>
  (시료, h) 60가지 추론(딥러닝 1 구조)                                                         id infer60-<구조>
  4-a 대리 측정(가장 느린 트리·W 100 원값·§5 후보 전부, fold 1~3·refit 중 줄인 뒤 h 60 복제 행 수가 가장 큰
        train — 메모리 판정 사다리 `downsize.decide_memory_in_child`, 메모리·시간)              id 4a-<알고>
결과(--out): items/<id>/(item.json·result.json·done.json·모델), summary_<시각>/(timing_table.csv·max_iter_decision.csv·
estimate_table.csv·estimate.json·ak_check.csv·run_meta.json), guard_log.jsonl. MLflow = `listeria-준비` 단계 run '0-9' 아래
판 run '0-9-<id>'(시간·RSS·GPU metric, result.json artifact), 요약 표는 단계 run artifact.

실행(서버, 검수 뒤 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_9_timing.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-9 --config configs/s0_9_timing.json
  요약만 다시(측정 끝난 뒤): … --summary-only

구현 세부(계획서에 없음 — 담당자 확인, 반환 보고에 같이):
 (1) 측정 순서: 계획서 나열(1-1·1-2·그룹 중요도·Optuna·사슬 부스팅·…)과 달리 사슬 부스팅을 그룹 중요도·Optuna 앞에 둠 —
     그룹 중요도(3-0b 와 같은 사슬 모델 교차 N 예측)·Optuna 시도(§3 '늘린 값은 … 단계 8 시도에 씀')가 판정한 최대 반복을
     쓰기 때문. 가장 느린 트리 = 1-1 의 W 100 원값 측정(학습 + 입력 + 예측) 시간이 가장 긴 알고리즘.
 (2) '1회 시간' = 그 fold 의 train 입력 조립 + 학습(CatBoost 사슬은 학습률 읽기 포함) + fold 3 validation 쌍 입력 조립·
     예측(사슬 부스팅은 56 시점 채점 포함). 자료 읽기 시간은 따로 적고 빼고 셈.
 (3) 사슬 부스팅 0-9 측정의 CatBoost 학습률 = fold 3 자동 학습률 하나(계획서 = fold 별 평균 — 0-9 는 fold 3 만 학습).
     최대 반복 판정 = fold 3 자기 곡선의 최고 시점(소수 4자리 반올림, 같으면 작은 N)이 마지막 시점인지.
 (4) 그룹 중요도 1회 = fold 3 하나에서 대리 판의 그룹 전부(44)를 M = 10 번 섞기. 부스팅 모델 예측 트리 수(Leaf n_iter) =
     fold 3 자기 곡선 최적 N(교차 N 은 3 fold 가 있어야 정해짐 — 0-9 는 fold 3 하나, n_iter 없이는 멈춤). 예상 = (1회 시간 ÷ 섞은 그룹 수) ×
     섞는 묶음 수 × (fold 1~3 채점 쌍 수 ÷ fold 3 채점 쌍 수).
 (5) Optuna 시도 1회 = TreeFold(fold 3) 하나로 score_tree_config(부분 fold — 반복 수는 자기 곡선 최고 시점),
     TreeFold 평가 쌍 입력 캐시는 설정 optuna_cache_val(기본 false — fold 3 평가 쌍 입력을 메모리에 다 두면 수십 GB).
 (6) 블록 학습 측정 = block_plan(reuse=True) 의 블록 OOF 모델(판 train 전체 역할 없는 모델) 중 학습 시료가 가장 많은 것,
     학습만(추론은 '60가지 추론' 항목). 60가지 추론 = fold 3 채점 쌍에 한 번이라도 들어가는 시료 × h 1~60(시료 전부가
     아님 — 쌍 하나당 시간으로 바꿔 씀).
 (7) AutoKeras 시도 측정 자료 = 단계 8 실제 자료(d ≤ 2021-06-30 시료당 h 1개 — §8 'AutoKeras 시도는 0-9 측정값으로'
     행 수 비례 없음), 구조 = 설정 autokeras.timing_arch. 구조별 대응 확인 = 대응표에서 대응 없음이면 기록만, 있으면
     같은 자료 앞 check_rows 행(씨앗 고정 무작위)·check_epochs 에폭·시도 1회로 그래프가 우리 입력(x_seq·x_static·x_cat)을
     받아 학습되는지 — 'error' 로 끝나면 멈추지 않고 '대응 확인 실패(메시지)' 로 기록(메모리 오류·강제 종료는 멈춤).
 (8) 4-a 대리: 범주 칼럼(범주 5·season·aw_label)은 단계 4~6 범주 숫자 인코딩(부록 A-6, k = 50 — 2-1 의 k 가 정해지기
     전이라 대리 k)으로 `catenc_<칼럼>`. 메모리 판정 단계마다 학습 전에 필요 메모리를 어림(행 × 칼럼 × 4 바이트 ×
     MEM_FACTOR 1.3)해 가용 RAM − 15GB 보다 크면 그 자리에서 MemoryError(실제로 잡았다가 서버 공용 메모리를 바닥내지
     않게), 디스크 기반 단계는 X 크기가 디스크 여유 − 10GB 보다 크면 MemoryError. m = 10 까지 모자라면 항목은 '사다리 끝'
     으로 기록하고 측정은 계속(4-a 와 그 위 판 예상은 '측정 불가' — 담당자 결정). X 는 10만 행 묶음으로 만들어 이어 붙임.
 (9) 예상 셈: 판 하나 = 학습 4회(3 fold + refit)는 fold 3 측정값 × 그 판 train 행 수 ÷ fold 3 train 행 수(§8 '행 수
     비례'), 잰 적 없는 W 는 가까운 측정 W(같으면 큰 W)에서 칼럼 수 비례(딥러닝 = W × 채널 수 비례), 잰 적 없는 판·구조는
     가장 가까운 측정값(표 '근거' 칸)·여럿이면 가장 느린 것(상한 쪽 — 예: 상위 3 딥러닝 = 블록 학습이 가장 느린 3 구조,
     A = 가장 느린 트리, 3-2·3-1 트리·4-c·⑧ = 트리 6종 중 가장 느린 것). GRU + attention(⑥) = LSTM + attention 측정값.
     블록 학습 예상(구조 s) = 블록 학습 측정 × 행 수 비 × (s 의 1 fold 측정 ÷ 대리 구조 1 fold 측정) — 모든 구조에.
     5·6 뽑은 행 = 해마다 양성 × 5(양성 + 음성 4배, 상한) — 판마다 그 판 train 양성 수 × 5. ③ 학습 행 = d ≤ 2025-06-30
     시료의 2/3. 딥러닝 추론 = 값 모델마다 역할의 (시료, h) 쌍 수(블록 OOF = 블록 시료 × 60 — 상한) × 쌍 하나당 시간.
     시각 = 설정 schedule_start(없으면 요약 시각, KST).
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(CODE_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Any, Callable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import guard as gd  # noqa: E402
from lst import registry as rg  # noqa: E402
from lst import resources as rs  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import wiring as wr  # noqa: E402
from lst.hassign import H_MAX  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.windows import MODES, W_GRID  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ================================================================ 계획서 고정값
STAGE = "0-9"
EXPERIMENT = "listeria-준비"
FOLD = "fold3"                                    # §6 0-9: fold 중 가장 큰 train
TREE1 = ("XGB", "LGB", "CAT")                     # 1-1·2차 사슬 부스팅
TREE_OTHER = ("RF", "ET", "LR")
TREE6 = TREE1 + TREE_OTHER
W_MEASURE = (10, 100)                             # 1-1·1-2 측정 W
DL_ARCHS = ("LSTM", "GRU", "TRF", "LSTMA", "TCN", "CMB", "CMBV")   # 3-0 딥러닝 5 + 결합 2
COMBINED = ("CMB", "CMBV")
PROXY = {"W": 100, "mode": "raw", "k": 50, "n_ma": 15, "n_ma_dl": 15}   # 0-9 대리 설정(§6 0-9)
MAX_ITER_START = 2000                             # §3
IMPORTANCE_M = 10                                 # 부록 A-5
SEQ_IDS = ("WEATHER", "WX1", "WX2")               # 시계열 단위(기상·SPI·차감) — ⑥·⑦
# 부록 A-7: 단계 8 대상이 될 수 있는 딥러닝·결합 구조 전부(라벨, 구조, 입력 꼴)
AK_STRUCTS = (("LSTM", "LSTM", "standard"), ("GRU", "GRU", "standard"), ("TRF", "TRF", "standard"),
              ("LSTMA", "LSTMA", "standard"), ("TCN", "TCN", "standard"), ("CMB", "CMB", "standard"),
              ("CMBV", "CMBV", "standard"), ("x6-LSTMA", "LSTMA", "seq_only"), ("x6-GRUA", "GRUA", "seq_only"),
              ("x6-TCN", "TCN", "seq_only"), ("x7-D", "TRF", "seq_only"))
ALL_COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
MEM_FACTOR = 1.3                                  # 구현 세부 (8)
FLOOR = rs.MIN_AVAILABLE_RAM_GB * rs.GB
DISK_FLOOR = rs.MIN_FREE_DISK_GB * rs.GB
CHUNK_ROWS = 100_000
INFER_CHUNK = 50_000
KST = _dt.timezone(_dt.timedelta(hours=9))

# 부록 A-9 학습 횟수 표(3-0b = 13 구조 × 예상 1,068) — 예상(15,415)
A9_COUNTS = (("1-1 트리", 240), ("1-2 LSTM", 40), ("2-0", 0), ("2-1 k", 20), ("2-1b N_ma", 32), ("2-R", 40),
             ("3-0", 97), ("3-0b 변수 빼기", 13884), ("3-1 이상 점수", 88), ("3-2 스태킹", 60),
             ("4-a·4-a0·미리보기", 12), ("4-c·4-c-L", 8), ("5·6", 80), ("추가 판 ①·②", 8), ("추가 판 ③", 210),
             ("추가 판 ④", 118), ("추가 판 ⑥", 57), ("추가 판 ⑦", 8), ("추가 판 ⑧", 4), ("8-T 튜닝", 298),
             ("8 딥러닝(AutoKeras)", 111))
A9_TOTAL = 15415
STRUCTURES13 = TREE6 + DL_ARCHS                   # 3-0·3-0b 13 구조
ROUNDS_0_9 = 3                                    # 부록 A-9 0-9 예상: 그룹·단위 빼기 각 3바퀴

DEFAULT_FUNCS = {k: f"s0_9_timing:measure_{k}" for k in
                 ("tree_fold", "dl_fold", "importance", "optuna", "block", "ak", "infer60", "fa")}

SCHEMA = {
    "plan": cf.Field(str),
    "stage": cf.Field(str, choices=(STAGE,)),
    "experiment": cf.Field(str, choices=(EXPERIMENT,)),
    "samples": cf.Field(str),                      # --inputs 아래
    "theil_targets": cf.Field(str),                # --inputs 아래(단계 0-1 14_theil_target_columns.json)
    "sample_file": cf.Field(str),                  # --root 아래(원 시료 파일 — 4-a 줄이기 칼럼 값)
    "experiment_log": cf.Field(str),               # --inputs 아래 jsonl
    "context": cf.Field(dict),                     # inputs.load_context 설정
    "n_jobs": cf.Field(int),
    "seed": cf.Field(int),
    "gpu_memory_fraction": cf.Field(float),
    "use_mlflow": cf.Field(bool),
    "mlflow_uri": cf.Field(str),
    "code_reviewed": cf.Field(bool),
    "fold": cf.Field(str, choices=(FOLD,)),
    "proxy": {"W": cf.Field(int), "mode": cf.Field(str), "k": cf.Field(int), "n_ma": cf.Field(int),
              "n_ma_dl": cf.Field(int)},
    "max_iter_start": cf.Field(int),
    "importance_M": cf.Field(int),
    "dl_proxy_arch": cf.Field(str, choices=DL_ARCHS),
    "optuna_cache_val": cf.Field(bool),
    "autokeras": {"timing_arch": cf.Field(str), "check_rows": cf.Field(int), "check_epochs": cf.Field(int),
                  "mapping": cf.Field(dict), "decision": cf.Field(str), "approx_note": cf.Field(str)},
    "schedule_start": cf.Field((str, type(None))),
    "child_timeout_s": cf.Field((int, type(None))),
    "smoke": {                                     # 연기 시험만(실제 실행 설정엔 없음)
        "max_train_rows": cf.Field(int), "max_pairs_per_h": cf.Field(int), "max_epochs": cf.Field(int),
        "max_targets": cf.Field(int), "max_samples_infer": cf.Field(int),
    },
}


def load_cfg(path) -> dict:
    raw = cf.load_json_strict(path)
    schema = SCHEMA if "smoke" in raw else {k: v for k, v in SCHEMA.items() if k != "smoke"}
    cfg = dict(cf.validate(raw, schema))
    rs.check_n_jobs(cfg["n_jobs"])
    if cfg["seed"] != 42:
        raise cf.ConfigError("씨앗은 42(§8·부록 B-5)")
    if not (0 < cfg["gpu_memory_fraction"] <= rs.GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {rs.GPU_MEMORY_FRACTION} 이하(공통 규칙)")
    if dict(cfg["proxy"]) != PROXY:
        raise cf.ConfigError(f"proxy 는 {PROXY}(§6 0-9 대리 설정 = W 100 원값·k 50·N_ma 15)")
    if cfg["max_iter_start"] != MAX_ITER_START:
        raise cf.ConfigError("max_iter_start 는 2000(§3)")
    if cfg["importance_M"] != IMPORTANCE_M:
        raise cf.ConfigError("importance_M 은 10(부록 A-5)")
    ak = cfg["autokeras"]
    if ak["timing_arch"] not in {a for _, a, _ in AK_STRUCTS}:
        raise cf.ConfigError(f"autokeras.timing_arch: {ak['timing_arch']}")
    if ak["check_rows"] < 1 or ak["check_epochs"] < 1:
        raise cf.ConfigError("autokeras.check_rows·check_epochs ≥ 1")
    archs = {a for _, a, _ in AK_STRUCTS}
    if set(ak["mapping"]) != archs:
        raise cf.ConfigError(f"autokeras.mapping 은 부록 A-7 구조 전부 {sorted(archs)}: {sorted(ak['mapping'])}")
    if cfg["schedule_start"] is not None:
        _dt.datetime.fromisoformat(cfg["schedule_start"])
    if "smoke" in cfg and cfg["use_mlflow"] and not cfg["mlflow_uri"].startswith("sqlite:"):
        raise cf.ConfigError("연기 시험(smoke) 설정은 로컬 sqlite MLflow 에만")
    return cfg


# ================================================================ 판 설정(VersionSpec dict)
def v_order1(W: int, mode: str) -> dict:
    return {"name": f"0-9-1차-W{W:03d}-{mode}", "order": 1, "W": W, "mode": mode, "include_ids": ["WEATHER"]}


def v_order1_dl(W: int) -> dict:
    return {**v_order1(W, wr.DL_MODE_PLACEHOLDER), "name": f"0-9-1차-딥러닝-W{W:03d}"}


def v_proxy(mode: str = "raw") -> dict:
    return {"name": f"0-9-대리-{mode}", "order": 2, "W": PROXY["W"], "mode": mode, "include_ids": list(rg.ALL_IDS),
            "k": PROXY["k"], "n_ma": PROXY["n_ma"], "n_ma_dl": PROXY["n_ma_dl"]}


def v_proxy_dl() -> dict:
    return {**v_proxy(wr.DL_MODE_PLACEHOLDER), "name": "0-9-대리-딥러닝"}


def v_seq_only() -> dict:
    """⑥·⑦ 입력(시계열 채널만 — 기상·차감·SPI)."""
    return {"name": "0-9-시계열만", "order": 2, "W": PROXY["W"], "mode": "raw", "include_ids": list(SEQ_IDS),
            "n_ma": PROXY["n_ma"], "n_ma_dl": PROXY["n_ma_dl"], "tree_layout": "seq_only", "dl_layout": "seq_only"}


# ================================================================ 하위 프로세스 쪽(측정 함수)
def _resolve(func_ref: str) -> Callable:
    mod, _, name = func_ref.partition(":")
    obj: Any = importlib.import_module(mod)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _peak_rss_gb(children: bool = False) -> float | None:
    try:
        import resource

        who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
        kb = resource.getrusage(who).ru_maxrss
        return float(kb) / 2 ** 20 if sys.platform.startswith("linux") else float(kb) / 2 ** 30
    except ImportError:
        if children:
            return None
    try:
        import psutil

        return float(getattr(psutil.Process().memory_info(), "peak_wset")) / 2 ** 30
    except Exception:  # noqa: BLE001
        return None


def child_entry(func_ref: str, kwargs: Mapping[str, Any], gpu_fraction: float | None = None) -> dict:
    """하위 프로세스 안: GPU 상한(딥러닝 항목) → 측정 함수 → 벽시계·최대 RSS·GPU 최대 메모리를 붙여 돌려줌."""
    gpu = {"limited": False}
    torch = None
    if gpu_fraction is not None:
        gpu["limited"] = rs.limit_gpu_memory(gpu_fraction)
        import torch  # noqa: F811

        if gpu["limited"]:
            torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = dict(_resolve(func_ref)(**dict(kwargs)))
    wall = time.time() - t0
    if torch is not None and gpu["limited"]:
        gpu["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        gpu["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2 ** 30
        gpu["fraction"] = gpu_fraction
    res.update({"wall_seconds": wall, "peak_rss_gb": _peak_rss_gb(), "peak_rss_children_gb": _peak_rss_gb(True),
                "gpu": gpu, "pid": os.getpid()})
    return res


def _check(env) -> Callable[[], Any]:
    d = env["item_dir"]
    return lambda: rs.check_resources(d)


def _load(env, components):
    """시료 표(test 정답 없음) + 입력 자료. 읽기 시간은 측정에서 뺌(구현 세부 2)."""
    from lst import evaluate as ev
    from lst import inputs as ip

    t0 = time.time()
    S = ev.SampleTable.read(resolve(env["inputs"], env["samples"]))
    ctx = ip.load_context(env["context"], env["root"], env["inputs"], components=list(components)) if components else None
    return S, ctx, time.time() - t0


def _smoke_rows(rows: pd.DataFrame, env) -> pd.DataFrame:
    sm = env.get("smoke")
    if not sm or len(rows) <= sm["max_train_rows"]:
        return rows
    y = rows["y"].to_numpy()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y != 1)
    k = max(0, sm["max_train_rows"] - pos.size)
    keep = np.sort(np.concatenate([pos, np.random.default_rng(env["seed"]).choice(neg, min(k, neg.size), replace=False)]))
    return rows.iloc[keep]


def _train_rows(S, env, W) -> pd.DataFrame:
    rows = fd.make_train_rows(S.known_frame(), fd.SPLITS[env["fold"]])
    return rn.add_bounds(_smoke_rows(rows, env), W)


def _iter_pairs(S, env, split: fd.Split, W):
    """평가 쌍을 h 마다 메모리에서(B-1). 연기 시험만 쌍 일부."""
    meta = S.meta()
    sm = env.get("smoke")
    check = _check(env)
    for h, idx in fd.iter_scoring_pairs_by_h(S.d_days, split):
        if sm and idx.size > sm["max_pairs_per_h"]:
            idx = np.sort(np.random.default_rng([env["seed"], h]).choice(idx, sm["max_pairs_per_h"], replace=False))
        if idx.size == 0:
            continue
        check()
        m = meta.iloc[idx].copy()
        m["h"] = np.int16(h)
        yield h, idx, rn.add_bounds(m, W)


def _own_best(curve: Sequence[float], points: Sequence[int]) -> tuple[int, bool]:
    """자기 곡선 최고 시점(4자리 반올림, 같으면 작은 N — models_tree 와 같은 규칙), 그 시점이 마지막인지."""
    from lst.metrics import round4

    r = np.array([round4(float(v)) if np.isfinite(v) else -np.inf for v in curve])
    if not np.isfinite(r).any():
        raise ValueError("곡선이 전부 NaN — 최적 N 을 정할 수 없음")
    i = int(np.flatnonzero(r == r.max())[0])
    return int(points[i]), i == len(points) - 1


def measure_tree_fold(env, algo: str, version: Mapping, max_iter: int | None, params: Mapping, curve: bool,
                      read_cat_lr: bool, save_model: bool, components: Sequence[str]) -> dict:
    """트리 1 fold(fold 3): train 입력 조립 → (CatBoost 학습률 읽기) → 학습 → validation 쌍 예측(사슬 = 56 시점 채점)."""
    from lst import inputs as ip
    from lst import metrics as mtc
    from lst import models_tree as mt
    from lst.hassign import u_tie_many

    S, ctx, t_load = _load(env, components)
    vs = rg.load_version(dict(version))
    split = fd.SPLITS[env["fold"]]
    t0 = time.time()
    rows = _train_rows(S, env, vs.W)
    asm = ip.TreeInputs(ctx, vs, split, scale=(algo == "LR")).fit(rows)
    X = asm.transform(rows)
    y = rows["y"].to_numpy(dtype=np.int64)
    t_build = time.time() - t0
    cat_cols = wr.tree_cat_cols(vs)
    ind = [c for c in asm.columns if c.endswith(rg.NA_SUFFIX) or c == rg.STRUCT_COLUMN] if algo == "LR" else []
    p = dict(params)
    t_lr, lr = 0.0, None
    if read_cat_lr:
        t1 = time.time()
        lr = mt.make_model("CAT", cat_cols=cat_cols, max_iter=max_iter, params=p).read_auto_learning_rate(
            X, y, seed=env["seed"], n_jobs=env["n_jobs"])
        p["learning_rate"] = lr
        t_lr = time.time() - t1
    model = mt.make_model(wr.TREE_ALGO[algo], cat_cols=cat_cols, indicator_cols=ind, max_iter=max_iter, params=p)
    t1 = time.time()
    model.fit(X, y, seed=env["seed"], n_jobs=env["n_jobs"])
    t_fit = time.time() - t1
    n_cols = int(X.shape[1])
    del X
    points = mt.checkpoints(max_iter) if curve else None
    byh = np.full((H_MAX, len(points)) if curve else H_MAX, np.nan)
    t1 = time.time()
    n_pairs = 0
    for h, idx, m in _iter_pairs(S, env, split, vs.W):
        Xh = asm.transform(m)
        n_pairs += idx.size
        if curve:
            P = model.predict_proba_multi(Xh, points)
            yh = S.cv_y(idx)
            u = u_tie_many(S.keys[idx], np.full(idx.size, h, dtype=np.int64))
            for k in range(len(points)):
                byh[h - 1, k] = mtc.pr_auc(yh, P[:, k], u)
        else:
            model.predict_proba(Xh)
    t_pred = time.time() - t1
    out = {"algo": algo, "train_rows": int(len(rows)), "train_pos": int(y.sum()), "n_cols": n_cols, "n_pairs": n_pairs,
           "max_iter": max_iter, "learning_rate": lr, "seconds_load": t_load, "seconds_build_train": t_build,
           "seconds_lr_read": t_lr, "seconds_fit": t_fit, "seconds_predict": t_pred,
           "seconds": t_build + t_lr + t_fit + t_pred, "fit_info": model.config_dump().get("fit_info")}
    if curve:
        cv = [float(mtc.band_weighted(byh[:, k])) for k in range(len(points))]
        best, at_last = _own_best(cv, points)
        out.update(points=list(points), n_points=len(points), curve=cv, own_best_n=best, own_best_at_last=at_last)
    if save_model:
        model.save(Path(env["item_dir"]) / "model.pkl")
    return out


def _dl_groups(rows: pd.DataFrame):
    return rows["sgg"].astype(str).to_numpy(), fd.day_ints(rows["d"])


def _dl_kw(env) -> dict:
    sm = env.get("smoke")
    return {"max_epochs": sm["max_epochs"]} if sm else {}


def measure_dl_fold(env, arch: str, version: Mapping, save_model: bool, components: Sequence[str]) -> dict:
    """딥러닝 1 fold(fold 3): 입력 전처리기 fit·train 입력 → 학습(조기 종료 (sgg, d) 10%) → validation 쌍 예측."""
    from lst import inputs as ip
    from lst import models_dl as md

    S, ctx, t_load = _load(env, components)
    vs = rg.load_version(dict(version))
    split = fd.SPLITS[env["fold"]]
    t0 = time.time()
    rows = _train_rows(S, env, vs.W)
    asm = ip.DLInputs(ctx, vs, split).fit(rows)
    x = asm.transform(rows)
    y = rows["y"].to_numpy(dtype=np.int64)
    t_build = time.time() - t0
    model = md.DLClassifier(arch, asm.cat_sizes, **_dl_kw(env))
    t1 = time.time()
    model.fit(x, y, seed=env["seed"], n_jobs=env["n_jobs"], groups=_dl_groups(rows))
    t_fit = time.time() - t1
    shape = {k: list(v.shape) for k, v in x.items()}
    del x
    t1 = time.time()
    n_pairs = 0
    for h, idx, m in _iter_pairs(S, env, split, vs.W):
        model.predict_proba(asm.transform(m))
        n_pairs += idx.size
    t_pred = time.time() - t1
    if save_model:
        model.save(Path(env["item_dir"]) / "model.pt")
        asm.save(Path(env["item_dir"]) / "inputs")
    return {"arch": arch, "train_rows": int(len(rows)), "train_pos": int(y.sum()), "input_shape": shape,
            "n_pairs": n_pairs, "seconds_load": t_load, "seconds_build_train": t_build, "seconds_fit": t_fit,
            "seconds_predict": t_pred, "seconds": t_build + t_fit + t_pred,
            "epochs_run": model.fit_info.get("epochs_run"), "fit_info": model.fit_info}


def _load_dl(env, ctx, vs, split, model_item: str):
    from lst import inputs as ip
    from lst import models_dl as md

    src = Path(env["items_root"]) / model_item
    asm = ip.DLInputs(ctx, vs, split)
    asm.prep = ip.DLPreprocessor.load(src / "inputs" / "dl_preprocessor.json")
    return asm, md.DLClassifier.load(src / "model.pt")


def measure_importance(env, model_kind: str, algo: str, version: Mapping, model_item: str, n_iter: int | None,
                       M: int, components: Sequence[str]) -> dict:
    """그룹 중요도 1회(fold 3 하나, 대리 판의 그룹 전부, M 회 섞기 — 부록 A-5). 모델 = 앞 항목에서 저장한 것."""
    from lst import importance as im
    from lst import inputs as ip
    from lst import models_tree as mt

    S, ctx, t_load = _load(env, components)
    vs = rg.load_version(dict(version))
    split = fd.SPLITS[env["fold"]]
    t0 = time.time()
    if model_kind == "tree":
        asm = ip.TreeInputs(ctx, vs, split, scale=(algo == "LR"))
        if algo == "LR":                       # 표준화 통계 = 그 모델 train(그 항목과 같은 행)
            rows = _train_rows(S, env, vs.W)
            asm.fit(rows)
            asm.transform(rows)                # fit 캐시 비움
            del rows
        model = mt.TreeModel.load(Path(env["items_root"]) / model_item / "model.pkl")
    elif model_kind == "dl":
        asm, model = _load_dl(env, ctx, vs, split, model_item)
    else:
        raise ValueError(model_kind)
    t_prep = time.time() - t0
    targets = im.targets_groups(vs.units())
    sm = env.get("smoke")
    if sm:
        targets = targets[:sm["max_targets"]]
    if model_kind == "tree" and algo in TREE1 and n_iter is None:   # 검수 주의 (1): 부스팅은 트리 수를 꼭 지정
        raise ValueError("부스팅 중요도에 n_iter 없음 — 최대 반복 전부로 예측됨")
    res = im.permutation_importance({FOLD: im.Leaf("0-9", asm, model, n_iter=n_iter)}, S, targets, M=M,
                                    check=_check(env))
    sec = float(res.seconds[FOLD])
    return {"model_kind": model_kind, "algo": algo, "n_targets": len(targets), "M": M, "n_iter": n_iter,
            "n_pairs": int(res.n_pairs[FOLD]), "seconds_load": t_load, "seconds_prepare": t_prep, "seconds": sec,
            "seconds_per_target": sec / len(targets), "base_score": float(res.base[FOLD])}


def measure_optuna(env, algo: str, version: Mapping, max_iter: int | None, cache_val: bool,
                   components: Sequence[str]) -> dict:
    """Optuna 시도 1회(TPE 첫 시도, fold 3 TreeFold — 구현 세부 5)."""
    from lst import inputs as ip
    from lst import tuning as tu

    S, ctx, t_load = _load(env, components)
    vs = rg.load_version(dict(version))
    split = fd.SPLITS[env["fold"]]
    a = wr.TREE_ALGO[algo]
    t0 = time.time()
    rows = _smoke_rows(fd.make_train_rows(S.known_frame(), split), env)
    tf = tu.TreeFold(env["fold"], S, ip.TreeInputs(ctx, vs, split, scale=(algo == "LR")), train_rows=rows,
                     cache_val=cache_val, check=_check(env))
    t_build = time.time() - t0
    recs = tu.run_optuna(a, lambda p: tu.score_tree_config(a, p, [tf], max_iter=max_iter, seed=env["seed"],
                                                           n_jobs=env["n_jobs"], check=_check(env)),
                         log_path=Path(env["item_dir"]) / "optuna_trials.jsonl", n_trials=tu.N_TPE_TRIALS,
                         seed=tu.TPE_SEED, stop_after=1, label="0-9 시도 1회")
    tr = [r for r in recs if r.get("event") == "trial"][-1]
    return {"algo": algo, "max_iter": max_iter, "train_rows": int(len(rows)), "seconds_load": t_load,
            "seconds_build_train": t_build, "seconds": float(tr["seconds"]), "trial_params": tr["params"],
            "cache_val": cache_val}


def measure_block(env, arch: str, version: Mapping, block_model: str, components: Sequence[str]) -> dict:
    """딥러닝 블록 OOF 모델 학습 1회(구현 세부 6)."""
    from lst import blocks as bl
    from lst import inputs as ip
    from lst import models_dl as md

    S, ctx, t_load = _load(env, components)
    vs = rg.load_version(dict(version))
    bm = next(m for m in bl.block_plan(reuse=True) if m.name == block_model)
    t0 = time.time()
    rows = rn.add_bounds(_smoke_rows(bl.model_train_rows(bm, S.known_frame()), env), vs.W)
    asm = ip.DLInputs(ctx, vs, fd.SPLITS[bm.owner.split]).fit(rows)
    x = asm.transform(rows)
    y = rows["y"].to_numpy(dtype=np.int64)
    t_build = time.time() - t0
    model = md.DLClassifier(arch, asm.cat_sizes, **_dl_kw(env))
    t1 = time.time()
    model.fit(x, y, seed=env["seed"], n_jobs=env["n_jobs"], groups=_dl_groups(rows))
    t_fit = time.time() - t1
    return {"arch": arch, "block_model": block_model, "train_rows": int(len(rows)), "seconds_load": t_load,
            "seconds_build_train": t_build, "seconds_fit": t_fit, "seconds": t_build + t_fit,
            "epochs_run": model.fit_info.get("epochs_run")}


def measure_ak(env, arch: str, version: Mapping, mapping: Mapping | None, rows_limit: int | None, epochs: int,
               components: Sequence[str]) -> dict:
    """AutoKeras 시도 1회(부록 A-7 그래프, 단계 8 자료 = d ≤ 2021-06-30 — 구현 세부 7)."""
    from lst import inputs as ip
    from lst import tuning as tu

    S, ctx, t_load = _load(env, components)
    vs = rg.load_version(dict(version))
    t0 = time.time()
    rows = tu.tune_train_rows(S)
    sm = env.get("smoke")
    lim = rows_limit if not sm else min(rows_limit or sm["max_train_rows"], sm["max_train_rows"])
    if lim is not None and len(rows) > lim:
        rows = rows.iloc[np.sort(np.random.default_rng(env["seed"]).choice(len(rows), lim, replace=False))]
    rows = rows.reset_index(drop=True)
    asm = ip.DLInputs(ctx, vs, tu.TUNE_SPLIT).fit(rows)
    x = asm.transform(rows)
    y = rows["y"].to_numpy(dtype=np.int64)
    t_build = time.time() - t0
    search = tu.autokeras_search(arch, x, y, _dl_groups(rows), directory=Path(env["item_dir"]) / "ak",
                                 project_name="t", cat_sizes=asm.cat_sizes, train_d=rows["d"], max_trials=1,
                                 seed=tu.AK_SEED, mapping=mapping, epochs=epochs)
    info = {k: v for k, v in search.info.items() if isinstance(v, (int, float, str, bool, type(None), dict))}
    return {"arch": arch, "layout": vs.dl_layout, "train_rows": int(len(rows)), "epochs": epochs,
            "seconds_load": t_load, "seconds_build_train": t_build, "seconds": float(search.info["seconds"]),
            "n_trials_done": len(tu.completed_trials(search)), "ak_info": info, "keys": list(search.keys)}


def measure_infer60(env, arch: str, version: Mapping, model_item: str, components: Sequence[str]) -> dict:
    """(시료, h) 60가지 추론: fold 3 채점 쌍 시료 × h 1~60(구현 세부 6)."""
    S, ctx, t_load = _load(env, components)
    vs = rg.load_version(dict(version))
    asm, model = _load_dl(env, ctx, vs, None, model_item)
    pos = np.unique(np.concatenate([idx for _, idx in fd.iter_scoring_pairs_by_h(S.d_days, fd.SPLITS[env["fold"]])]))
    sm = env.get("smoke")
    if sm and pos.size > sm["max_samples_infer"]:
        pos = np.sort(np.random.default_rng(env["seed"]).choice(pos, sm["max_samples_infer"], replace=False))
    meta = S.meta()
    check = _check(env)
    t0 = time.time()
    for h in range(1, H_MAX + 1):
        for s in range(0, pos.size, INFER_CHUNK):
            check()
            m = meta.iloc[pos[s:s + INFER_CHUNK]].copy()
            m["h"] = np.int16(h)
            model.predict_proba(asm.transform(rn.add_bounds(m, vs.W)))
    sec = time.time() - t0
    n = int(pos.size) * H_MAX
    return {"arch": arch, "n_samples": int(pos.size), "n_pairs": n, "seconds_load": t_load, "seconds": sec,
            "seconds_per_pair": sec / n}


# ---------------------------------------------------------------- 4-a 대리(구현 세부 8)
def _population_4a(ctx, train_end: int):
    """단계 4~6 범주 숫자 인코딩 모집단 = 그 판 train 시료(h 복제 없음) + season·aw_label 수준(부록 A-6)."""
    from lst import history as hi
    from lst import static_feats as sf

    s = ctx.samples
    sub = s.loc[fd.day_ints(s["d"]) <= int(train_end), ["lims_data_key", "y", "d", "p2", "p3", "sgg", "sido",
                                                         "src_route"]].reset_index(drop=True)
    pop = hi.Population.from_samples(sub)
    pop.add_level("season", sf.season(sub["d"]))
    aw = sf.food_features(ctx.food, sub["p3"].to_numpy(dtype=object))["aw_label"].to_numpy(dtype=object)
    pop.add_level("aw_label", np.where(pd.isna(aw), rg.AW_MISSING, aw).astype(object))
    return pop


def encode_4a(X: pd.DataFrame, meta: pd.DataFrame, pop, k: int, cat_cols: Sequence[str]) -> pd.DataFrame:
    """범주 칼럼 → 스무딩 검출률 숫자 `catenc_<칼럼>`(그 자리, float32). 행 C = T 직전 06-30(§5-1)."""
    from lst import history as hi

    rows = pd.DataFrame({"d": fd.day_ints(meta["d"]), "C": hi.c_from_T(meta["T"]),
                         **{c: meta[c].astype(str).to_numpy(dtype=object) for c in rg.CAT5}})
    R = hi.prepare_rows(rows)
    enc = {}
    base = hi.category_rates(pop, R, k)
    for c in cat_cols:
        if c in rg.CAT5:
            enc[c] = base[f"catenc_{c}"].to_numpy()
        else:
            enc[c] = hi.category_rate(pop, c, X[c].astype(str).to_numpy(dtype=object), R, k)
    out = {}
    for c in X.columns:
        if c in enc:
            out[f"catenc_{c}"] = np.asarray(enc[c], dtype=np.float32)
        else:
            out[c] = X[c].to_numpy(dtype=np.float32)
    return pd.DataFrame(out)


def _build_4a_chunks(ctx, vs, rows, split, pop, k, cat_cols, check):
    from lst import inputs as ip

    for s in range(0, len(rows), CHUNK_ROWS):
        check()
        part = rows.iloc[s:s + CHUNK_ROWS]
        X = ip.build_tree_X(part, ctx, vs, split=split)
        yield encode_4a(X, part, pop, k, cat_cols), part["y"].to_numpy(dtype=np.int8)


def attempt_4a(env, algo: str, version: Mapping, max_iter: int, params: Mapping, split: str, k: int,
               code_md5: str, setting) -> dict:
    """메모리 판정 한 단계(하위 프로세스): 줄인 train + h 복제 → 입력(범주 숫자 인코딩) → 학습 → 그 판 채점 쌍 예측."""
    from lst import downsize as ds
    from lst import inputs as ip
    from lst import models_tree as mt

    S, ctx, t_load = _load(env, ALL_COMPONENTS)
    vs = rg.load_version(dict(version))
    sp = fd.SPLITS[split]
    check = _check(env)
    t0 = time.time()
    rows = fd.make_train_rows(S.known_frame(), sp)
    keep = pd.Index(ds.load_kept_keys(Path(env["item_dir"]) / "reduction", split))
    rows = rows.loc[keep.get_indexer(pd.Index(rows["lims_data_key"].astype(str))) >= 0]
    rows = rn.add_bounds(_smoke_rows(ds.replicate_h(rows.reset_index(drop=True), "4-a", m=setting.m), env), vs.W)
    cat_cols = wr.tree_cat_cols(vs)
    n_cols = len(rg.tree_columns(vs))
    x_bytes = len(rows) * n_cols * 4
    if setting.disk:
        free = rs.free_disk_bytes(env["item_dir"])
        if x_bytes > free - DISK_FLOOR:
            raise MemoryError(f"디스크 기반: X {x_bytes / rs.GB:.0f}GB > 디스크 여유 {free / rs.GB:.0f}GB − 10GB(구현 세부 8)")
    else:
        avail = rs.available_ram_bytes()
        if x_bytes * MEM_FACTOR > avail - FLOOR:
            raise MemoryError(f"예상 메모리 {x_bytes * MEM_FACTOR / rs.GB:.0f}GB > 가용 RAM {avail / rs.GB:.0f}GB − 15GB"
                              "(구현 세부 8 — 실제로 잡기 전에 멈춤)")
    pop = _population_4a(ctx, sp.train_end)
    a = wr.TREE_ALGO[algo]
    chunks = _build_4a_chunks(ctx, vs, rows, sp, pop, k, cat_cols, check)
    if setting.disk:
        tag = ds.disk_cache_tag({"code_md5": code_md5, "input_md5s": env.get("input_md5s", {}), "spec_md5": env["item_md5"],
                                 "split": split, "setting": setting.to_dict()})
        t1 = time.time()
        model = ds.XGBDiskModel(max_iter=max_iter, params=dict(params)).fit_chunks(
            chunks, cache_root=Path(env["item_dir"]) / "xgbext", tag=tag, seed=env["seed"], n_jobs=env["n_jobs"])
        t_build, t_fit = 0.0, time.time() - t1     # 디스크 기반은 조립·쓰기·학습이 한 흐름
    else:
        X = None
        cols = None
        ys = []
        pos = 0
        for Xc, yc in chunks:
            if X is None:
                cols = list(Xc.columns)
                X = np.empty((len(rows), len(cols)), dtype=np.float32)
            X[pos:pos + len(Xc)] = Xc.to_numpy(dtype=np.float32)
            ys.append(yc)
            pos += len(Xc)
        X = pd.DataFrame(X, columns=cols, copy=False)
        y = np.concatenate(ys).astype(np.int64)
        t_build = time.time() - t0
        model = mt.make_model(a, cat_cols=[], max_iter=max_iter, params=dict(params))
        t1 = time.time()
        model.fit(X, y, seed=env["seed"], n_jobs=env["n_jobs"])
        t_fit = time.time() - t1
        del X
    points = mt.checkpoints(max_iter)
    t1 = time.time()
    n_pairs = 0
    for h, idx, m in _iter_pairs(S, env, sp, vs.W):
        Xh = encode_4a(ip.build_tree_X(m, ctx, vs, split=sp), m, pop, k, cat_cols)
        if split == "refit":
            model.predict_proba(Xh, n_iter=max_iter)
        else:
            model.predict_proba_multi(Xh, points)
        n_pairs += idx.size
    t_pred = time.time() - t1
    return {"train_rows": int(len(rows)), "n_cols": n_cols, "x_gb": x_bytes / rs.GB, "n_pairs": n_pairs,
            "seconds_load": t_load, "seconds_build_train": t_build, "seconds_fit": t_fit, "seconds_predict": t_pred,
            "seconds_work": t_build + t_fit + t_pred}


def measure_fa(env, algo: str, version: Mapping, max_iter: int, params: Mapping, k: int, code_md5: str,
               theil_targets: str, sample_file: str) -> dict:
    """4-a 대리 측정: fold 1~3·refit 줄이기(Theil's U) → 가장 큰 train → 메모리 판정 사다리(단계마다 하위 프로세스)."""
    from lst import downsize as ds

    S, _, t_load = _load(env, ())
    t0 = time.time()
    cols = ds.load_theil_targets(resolve(env["inputs"], theil_targets))
    data = ds.ReductionData.load(S.known_frame(), resolve(env["root"], sample_file), cols)
    reds = ds.plan_reductions(data)
    rdir = Path(env["item_dir"]) / "reduction"
    for r in reds.values():
        ds.save_reduction(r, rdir)
    largest = ds.largest_split(reds)
    t_red = time.time() - t0
    out = {"largest_split": largest, "reduced_rows": {s: int(r.stats["n_after"]) for s, r in reds.items()},
           "reduction": {s: r.summary() for s, r in reds.items()}, "seconds_load": t_load, "seconds_reduction": t_red,
           "algo": algo, "max_iter": max_iter}
    kw = {"env": dict(env), "algo": algo, "version": dict(version), "max_iter": max_iter, "params": dict(params),
          "split": largest, "k": k, "code_md5": code_md5}
    try:
        dec = ds.decide_memory_in_child("s0_9_timing:attempt_4a", kw, algo=wr.TREE_ALGO[algo],
                                        log_path=Path(env["item_dir"]) / "memory_ladder.jsonl", run_id="0-9-4a")
    except ds.MemoryLadderExhausted as e:
        out.update(status="ladder_exhausted", attempts=e.attempts, seconds=float("nan"),
                   note="m = 10 까지 메모리 부족 — 4-a 와 그 위 판 예상 측정 불가(담당자 결정)")
        return out
    m = dec.setting.m
    out.update(status="ok", setting=dec.setting.to_dict(), attempts=dec.attempts, result=dec.result,
               seconds=float(dec.result["seconds_work"]), train_rows=int(dec.result["train_rows"]),
               replicated_rows=ds.replicated_counts(reds, m), peak_rss_gb_attempt=dec.peak_rss_gb,
               name_suffix=ds.name_with_m("4-a", m))
    return out


# ================================================================ 부모 쪽(항목 실행·이어하기·MLflow)
def _md5(obj) -> str:
    return hashlib.md5(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class Timing:
    """항목을 차례로(하위 프로세스) — 끝난 항목은 결과를 읽어 넘어감(B-1 이어하기)."""

    def __init__(self, cfg: dict, root: Path, inputs_root: Path, out: Path, *, overrides: Mapping[str, str] | None = None,
                 tracker=None, isolate: bool = True, check_resources: Callable = rs.check_resources):
        self.cfg, self.root, self.inputs_root, self.out = cfg, Path(root), Path(inputs_root), Path(out)
        self.items_root = self.out / "items"
        self.items_root.mkdir(parents=True, exist_ok=True)
        self.funcs = {**DEFAULT_FUNCS, **dict(overrides or {})}
        self.tracker, self.isolate, self.check_resources = tracker, isolate, check_resources
        self.code_md5 = cf.code_md5(CODE_ROOT)
        self.exp_log = rn.ExperimentLog(resolve(self.inputs_root, cfg["experiment_log"]))
        self.results: dict[str, dict] = {}
        self.specs: dict[str, dict] = {}
        self.order: list[str] = []
        self.ak_rows: list[dict] = []
        self.chain: dict[str, dict] = {}
        self.stage_run_id = tracker.stage_run(EXPERIMENT, STAGE) if tracker is not None else None
        self.env = {"root": str(self.root), "inputs": str(self.inputs_root), "samples": cfg["samples"],
                    "context": cfg["context"], "fold": cfg["fold"], "n_jobs": cfg["n_jobs"], "seed": cfg["seed"],
                    "items_root": str(self.items_root), "smoke": cfg.get("smoke")}

    # ------------------------------------------------------------ 항목 하나
    def item(self, iid: str, kind: str, lane: str, kwargs: Mapping[str, Any], *, allow_error: bool = False) -> dict:
        rn.variant_dir(self.items_root, iid)                 # 이름 검사
        spec = {"id": iid, "kind": kind, "lane": lane, "kwargs": dict(kwargs), "smoke": self.cfg.get("smoke")}
        md5 = _md5(spec)
        d = self.items_root / iid
        if (d / "done.json").exists():
            done = cf.load_json_strict(d / "done.json")
            if done["spec_md5"] != md5:
                raise rn.ResumeError(f"{iid}: 같은 이름 항목의 설정이 다름({done['spec_md5']} ≠ {md5}) — 지휘자 결정")
            if done["code_md5"] != self.code_md5:
                raise rn.ResumeError(f"{iid}: 끝난 항목의 code_md5({done['code_md5']}) ≠ 지금({self.code_md5}) — 지휘자 결정")
            return self._keep(iid, spec, rn._load_scores(d / "result.json"))
        res = None
        if d.exists():
            st = d / "started.json"
            same = st.exists() and cf.load_json_strict(st).get("spec_md5") == md5 and \
                cf.load_json_strict(st).get("code_md5") == self.code_md5
            if same and (d / "result.json").exists():        # 측정은 끝났고 기록(MLflow·done)만 남음
                res = rn._load_scores(d / "result.json")
            else:
                ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                dst = self.out / "_incomplete" / f"{iid}__{ts}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(d, dst)
                self.exp_log.append("rerun", stage=STAGE, name=iid, reason="멈춘 측정 항목 다시", moved_to=dst.name)
        if res is None:
            d.mkdir(parents=True)
            cf.dump_json(d / "item.json", spec)
            cf.dump_json(d / "started.json", {"spec_md5": md5, "code_md5": self.code_md5, "pid": os.getpid(),
                                              "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
            self.exp_log.append("start", stage=STAGE, name=iid, spec_md5=md5, code_md5=self.code_md5)
            self.check_resources(self.out)
            env = {**self.env, "item_dir": str(d), "item_md5": md5}
            ckw = {"func_ref": self.funcs[kind], "kwargs": {"env": env, **dict(kwargs)},
                   "gpu_fraction": self.cfg["gpu_memory_fraction"] if lane == "gpu" else None}
            print(f"[0-9] {iid} 측정 시작 ({kind}, {lane})", flush=True)
            try:
                if self.isolate:
                    res = gd.run_guarded("s0_9_timing:child_entry", ckw, log_path=self.out / "guard_log.jsonl",
                                         run_id=iid, n_jobs_ladder=(self.cfg["n_jobs"],), retries_per_level=1,
                                         before_retry=lambda _r: self.check_resources(self.out),
                                         timeout=self.cfg["child_timeout_s"])
                else:
                    res = child_entry(**ckw)
            except gd.TrainingHalted as e:
                last = e.attempts[-1] if e.attempts else {}
                if allow_error and last.get("cause") == "error":
                    res = {"status": "error", "error": f"{last.get('exc_type')}: {last.get('message', '')[:1500]}",
                           "seconds": float("nan")}
                else:
                    self._fail(d, iid, e, last)
                    raise rn.RunStopped(f"측정 {iid} 멈춤 — {e}. 원인을 없앤 뒤 같은 명령을 다시 실행하면 이 항목부터(B-1)") from e
            except BaseException as e:
                if allow_error and not isinstance(e, (MemoryError, KeyboardInterrupt)) and not self.isolate:
                    res = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:1500]}", "seconds": float("nan")}
                else:
                    self._fail(d, iid, e, {})
                    raise
            res.setdefault("status", "ok")
            cf.dump_scores_json(d / "result.json", res)
            res = rn._load_scores(d / "result.json")
        run_id = self._mlflow(iid, spec, md5, res, d)
        cf.dump_scores_json(d / "done.json", {"id": iid, "spec_md5": md5, "code_md5": self.code_md5, "mlflow_run_id": run_id,
                                       "status": res.get("status"), "seconds": _num(res.get("seconds"))})
        sec = _num(res.get("seconds"))
        self.exp_log.append("done", stage=STAGE, name=iid, seconds=sec if math.isfinite(sec) else None,
                            status=res.get("status"))
        print(f"[0-9] {iid} 끝: {res.get('status')} {_num(res.get('seconds'))}s", flush=True)
        return self._keep(iid, spec, res)

    def _keep(self, iid, spec, res) -> dict:
        self.results[iid] = res
        self.specs[iid] = spec
        if iid not in self.order:
            self.order.append(iid)
        return res

    def _fail(self, d: Path, iid: str, e: BaseException, last: Mapping) -> None:
        try:
            cf.dump_json(d / "failed.json", {"type": type(e).__name__, "message": str(e)[:2000],
                                             "last_attempt": {k: v for k, v in last.items() if k != "traceback"}},
                         overwrite=True)
            self.exp_log.append("stopped", stage=STAGE, name=iid, type=type(e).__name__, message=str(e)[:500])
        except Exception:  # noqa: BLE001 — 기록 실패가 원래 예외를 가리지 않게
            pass

    def _mlflow(self, iid: str, spec: Mapping, md5: str, res: Mapping, d: Path) -> str | None:
        tr = self.tracker
        if tr is None:
            return None
        name = f"0-9-{iid}"
        old = tr.find_run(EXPERIMENT, name, self.stage_run_id)
        if old is not None and old.info.status == "FINISHED":
            return old.info.run_id
        tags = {"단계": STAGE, "코드 검수 완료": "예" if self.cfg["code_reviewed"] else "아니오", "설정 md5": md5,
                "code_md5": self.code_md5}
        kw = spec["kwargs"]
        params = {"kind": spec["kind"], "lane": spec["lane"], "fold": self.cfg["fold"],
                  **{k: kw[k] for k in ("algo", "arch", "max_iter", "n_iter", "block_model", "epochs", "rows_limit", "M")
                     if k in kw},
                  "version": kw["version"].get("name") if isinstance(kw.get("version"), Mapping) else None,
                  "W": kw["version"].get("W") if isinstance(kw.get("version"), Mapping) else None}
        rid = tr.start_variant(EXPERIMENT, self.stage_run_id, name, tags, params)
        m = {k: v for k, v in res.items() if k.startswith("seconds") or k in (
            "wall_seconds", "peak_rss_gb", "peak_rss_children_gb", "train_rows", "n_cols", "n_pairs", "n_points",
            "own_best_n", "n_targets", "learning_rate")}
        m = {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        g = res.get("gpu") or {}
        for k in ("peak_allocated_gb", "peak_reserved_gb"):
            if isinstance(g.get(k), (int, float)):
                m[f"gpu_{k}"] = g[k]
        tr.log_metrics(rid, m)
        tr.log_artifact(rid, d / "result.json")
        tr.finish(rid, "FINISHED" if res.get("status") in ("ok", None) else "FAILED")
        return rid

    # ------------------------------------------------------------ 측정 순서(§6 0-9, 구현 세부 1)
    def run_all(self) -> None:
        cfg = self.cfg
        n1 = list(ALL_COMPONENTS)
        for a in TREE1:                                                              # 1-1
            for W in W_MEASURE:
                for mode in MODES:
                    self.item(f"1-1-{a}-W{W:03d}-{mode}", "tree_fold", "cpu",
                              {"algo": a, "version": v_order1(W, mode), "max_iter": None, "params": {}, "curve": False,
                               "read_cat_lr": False, "save_model": False, "components": ["weather"]})
        for W in W_MEASURE:                                                          # 1-2
            self.item(f"1-2-LSTM-W{W:03d}", "dl_fold", "gpu",
                      {"arch": "LSTM", "version": v_order1_dl(W), "save_model": False, "components": ["weather", "missing"]})
        slow = self.slowest_tree()
        for a in [slow] + [x for x in TREE1 if x != slow]:                           # 2차 사슬 부스팅(§3 최대 반복)
            self.chain_boost(a, n1)
        ch = self.chain[slow]
        self.item(f"imp-tree-{slow}", "importance", "cpu",                          # 그룹 중요도(가장 느린 트리)
                  {"model_kind": "tree", "algo": slow, "version": v_proxy(), "model_item": ch["item"],
                   "n_iter": ch["own_best_n"], "M": cfg["importance_M"], "components": n1})
        self.item(f"optuna-{slow}", "optuna", "cpu",                                # Optuna 시도 1회
                  {"algo": slow, "version": v_proxy(), "max_iter": ch["max_iter"], "cache_val": cfg["optuna_cache_val"],
                   "components": n1})
        for a in TREE_OTHER:                                                         # RF·ExtraTrees·로지스틱 1 fold
            self.item(f"fold-{a}", "tree_fold", "cpu",
                      {"algo": a, "version": v_proxy(), "max_iter": None, "params": {}, "curve": False,
                       "read_cat_lr": False, "save_model": a in ("RF", "LR"), "components": n1})
        px = cfg["dl_proxy_arch"]
        for arch in DL_ARCHS:                                                        # 딥러닝·결합 1 fold
            self.item(f"dl-{arch}", "dl_fold", "gpu",
                      {"arch": arch, "version": v_proxy_dl(), "save_model": arch == px, "components": n1})
        self.item(f"imp-dl-{px}", "importance", "gpu",
                  {"model_kind": "dl", "algo": px, "version": v_proxy_dl(), "model_item": f"dl-{px}", "n_iter": None,
                   "M": cfg["importance_M"], "components": n1})
        for a in ("RF", "LR"):
            self.item(f"imp-{a}", "importance", "cpu",
                      {"model_kind": "tree", "algo": a, "version": v_proxy(), "model_item": f"fold-{a}", "n_iter": None,
                       "M": cfg["importance_M"], "components": n1})
        self.item(f"block-{px}", "block", "gpu",
                  {"arch": px, "version": v_proxy_dl(), "block_model": largest_block_model(), "components": n1})
        self.autokeras(n1)
        self.item(f"infer60-{px}", "infer60", "gpu",
                  {"arch": px, "version": v_proxy_dl(), "model_item": f"dl-{px}", "components": n1})
        p4 = {"learning_rate": ch["learning_rate"]} if slow == "CAT" else {}
        self.item(f"4a-{slow}", "fa", "cpu",
                  {"algo": slow, "version": v_proxy(), "max_iter": ch["max_iter"], "params": p4, "k": PROXY["k"],
                   "code_md5": self.code_md5, "theil_targets": cfg["theil_targets"], "sample_file": cfg["sample_file"]})

    def log_ak_mapping(self, ak: Mapping) -> None:
        """부록 A-7 '블록 대응·범위는 … 정해 실험 로그에 고정(단계 1 전에)' — 같은 대응표 기록이 없을 때만 한 번."""
        from lst import tuning as tu

        h = _md5(ak["mapping"])
        if any(e.get("event") == "ak_mapping_fixed" and e.get("mapping_md5") == h for e in self.exp_log.entries()):
            return
        self.exp_log.append("ak_mapping_fixed", stage=STAGE, mapping_md5=h, mapping=ak["mapping"],
                            same_as_AK_PROPOSAL=json.loads(json.dumps(tu.AK_PROPOSAL)) == ak["mapping"],
                            decision=ak["decision"], approx_note=ak["approx_note"])

    def slowest_tree(self) -> str:
        t = {a: _num(self.results[f"1-1-{a}-W100-raw"]["seconds"]) for a in TREE1}
        return max(TREE1, key=lambda a: (t[a], -TREE1.index(a)))

    def chain_boost(self, a: str, comps) -> dict:
        """§3: 최대 2,000 → fold 3 최적 N 이 마지막 시점이면 2배씩, 상한 16,000(넘으면 '상한')."""
        from lst.models_tree import next_max_iter

        it, hist = self.cfg["max_iter_start"], []
        while True:
            iid = f"chain-{a}-it{it:05d}"
            r = self.item(iid, "tree_fold", "cpu",
                          {"algo": a, "version": v_proxy(), "max_iter": it, "params": {}, "curve": True,
                           "read_cat_lr": a == "CAT", "save_model": True, "components": list(comps)})
            hist.append({"item": iid, "max_iter": it, "n_points": r.get("n_points"), "own_best_n": r.get("own_best_n"),
                         "own_best_at_last": bool(r.get("own_best_at_last")), "seconds": _num(r.get("seconds"))})
            nxt = next_max_iter(it) if r.get("own_best_at_last") else None
            if not r.get("own_best_at_last") or nxt is None:
                self.chain[a] = {"item": iid, "max_iter": it, "own_best_n": int(r["own_best_n"]),
                                 "cap": bool(r.get("own_best_at_last")), "learning_rate": r.get("learning_rate"),
                                 "history": hist}
                return self.chain[a]
            it = nxt

    def autokeras(self, comps) -> None:
        """부록 A-7 구조별 대응 확인 + 시도 1회 시간(구현 세부 7)."""
        from lst import tuning as tu

        ak = self.cfg["autokeras"]
        mapping = table = ak["mapping"]
        self.log_ak_mapping(ak)
        for label, arch, layout in AK_STRUCTS:
            row = {"label": label, "arch": arch, "layout": layout, "mapping_source": "설정(담당자 결정)",
                   "mapping": dict(table.get(arch) or {})}
            try:
                tu.ak_mapping(arch, table)
            except tu.AKUnsupported as e:
                row.update(status="대응 없음(튜닝 안 함)", reason=str(e))
                self.ak_rows.append(row)
                continue
            ver = v_proxy_dl() if layout == "standard" else {**v_seq_only(), "mode": "raw"}
            r = self.item(f"ak-check-{label}", "ak", "gpu",
                          {"arch": arch, "version": ver, "mapping": mapping, "rows_limit": ak["check_rows"],
                           "epochs": ak["check_epochs"], "components": list(comps)}, allow_error=True)
            row.update(status="대응 확인" if r.get("status") == "ok" else "대응 확인 실패",
                       reason=r.get("error"), seconds=_num(r.get("seconds")), keys=r.get("keys"))
            self.ak_rows.append(row)
        arch = ak["timing_arch"]
        tu.ak_mapping(arch, table)                     # 시간 재는 구조는 대응이 있어야 함
        self.item(f"ak-trial-{arch}", "ak", "gpu",
                  {"arch": arch, "version": v_proxy_dl(), "mapping": mapping, "rows_limit": None,
                   "epochs": tu.AK_EPOCHS, "components": list(comps)})


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def largest_block_model() -> str:
    """블록 OOF 모델(판 train 전체 역할 없는 모델) 중 학습 구간 날 수가 가장 긴 것 — 시료 수는 시료 표가 있어야 해서
    부모는 구간 길이로 고름(구현 세부 6; 같은 길이면 계획 순서 앞)."""
    from lst import blocks as bl

    cand = [m for m in bl.block_plan(reuse=True) if not m.is_split_full]
    return max(cand, key=lambda m: (sum(b - a + 1 for a, b in m.train_intervals), -cand.index(m))).name


# ================================================================ 예상(§8·부록 A-9 — 구현 세부 9)
def column_info() -> dict:
    """칼럼·채널 수(등록부) — '칼럼 수 비례' 보간."""
    cnt = lambda v: len(rg.tree_columns(rg.load_version(v)))  # noqa: E731
    base = {"order": 2, "W": PROXY["W"], "mode": "raw"}
    r2 = {"name": "x", **base, "include_ids": ["WEATHER"]}
    out = {
        "order1": {f"W{W:03d}|{m}": cnt(v_order1(W, m)) for W in W_GRID for m in MODES},
        "proxy": cnt(v_proxy()),
        "2-1": cnt({**r2, "include_ids": ["WEATHER", "HIS1", "HIS2", "HIS3", "HIS4"], "k": PROXY["k"]}),
        "2-1b": cnt({**r2, "include_ids": ["WEATHER", "WX2"], "n_ma": PROXY["n_ma"]}),
        "2-R": [cnt({**r2, "include_units": list(rg.RESEARCH_UNITS)})] +
               [cnt({**r2, "include_units": [u]}) for u in rg.RESEARCH_UNITS],
        "seq_tree": cnt(v_seq_only()),
        "lims": len(rg.CAT5) + len(rg.TIME_COLS),       # ④·(e)·4-c-L: 범주 5 + h·g
        "meta": 6,                                        # ④ 메타 로지스틱 입력 = 트리 6종 확률
    }
    seq1 = len(rg.dl_seq_channels(rg.load_version(v_order1_dl(100))))
    out["dl_order1_size"] = {f"W{W:03d}": W * seq1 for W in W_GRID}
    out["dl_C_order1"] = seq1
    out["dl_C_2-1b"] = len(rg.dl_seq_channels(rg.load_version(
        {"name": "x", "order": 2, "W": 100, "mode": "sum", "include_ids": ["WEATHER", "WX2"], "n_ma_dl": PROXY["n_ma_dl"]})))
    from lst import importance as im

    units = [u.name for u in rg.UNITS]
    out["targets"] = {"groups": len(im.targets_groups(units)), "units": len(im.targets_multi_unit(units)),
                      "2r1": len(im.targets_2r1()), "final": len(im.targets_final(units))}
    return out


def row_info(S) -> dict:
    """행 수(시료 표 — 정답은 d ≤ 2025-06-30 만): 판별 train 행·양성·채점 쌍, 블록 모델 학습 시료·역할 쌍."""
    from lst import blocks as bl

    known = S.known_frame()
    rows, pos = {}, {}
    for s in rn.SPLIT_ORDER:
        r = fd.make_train_rows(known, fd.SPLITS[s])
        rows[s], pos[s] = int(len(r)), int(r["y"].sum())
    pairs = {s: S.expected_pairs(s) for s in rn.SPLIT_ORDER}
    d = fd.day_ints(known["d"])
    tune_iv = ((bl.DATA_START, fd.day("2021-06-30")),)
    blocks = []
    for m in bl.block_plan(reuse=True):
        rp = 0
        for r in m.roles:
            rp += int((bl.block_of(d, r.split) == r.block).sum()) * H_MAX if r.kind == "oof" else pairs[r.split]
        blocks.append({"name": m.name, "rows": int(m.train_mask(d).sum()), "full": bool(m.is_split_full),
                       "tune": m.train_intervals == tune_iv, "role_pairs": rp})
    return {"rows": rows, "pos": pos, "pairs": pairs, "n_known": int(len(known)), "blocks": blocks}


class Units:
    """측정 결과 → '1회 시간'(초) 조회."""

    def __init__(self, R: Mapping[str, Mapping], chain: Mapping[str, Mapping], info: Mapping, cfg_proxy_arch: str,
                 ak_arch: str):
        self.R, self.chain, self.I, self.px, self.ak_arch = R, chain, info, cfg_proxy_arch, ak_arch
        self.notes: list[str] = []

    def s(self, iid: str) -> float:
        r = self.R.get(iid)
        return float("nan") if r is None or r.get("status") not in ("ok", None) else _num(r.get("seconds"))

    @staticmethod
    def _near(W: int) -> int:
        return min(W_MEASURE, key=lambda m: (abs(m - W), -m))

    def t11(self, a: str, W: int, mode: str) -> float:
        n = self._near(W)
        c = self.I["cols"]["order1"]
        return self.s(f"1-1-{a}-W{n:03d}-{mode}") * c[f"W{W:03d}|{mode}"] / c[f"W{n:03d}|{mode}"]

    def t12(self, W: int) -> float:
        n = self._near(W)
        c = self.I["cols"]["dl_order1_size"]
        return self.s(f"1-2-LSTM-W{n:03d}") * c[f"W{W:03d}"] / c[f"W{n:03d}"]

    def tree(self, a: str) -> float:
        return self.s(self.chain[a]["item"]) if a in TREE1 else self.s(f"fold-{a}")

    def slow_tree6(self) -> tuple[str, float]:
        v = {a: self.tree(a) for a in TREE6}
        a = max(TREE6, key=lambda x: (v[x] if math.isfinite(v[x]) else -1, -TREE6.index(x)))
        return a, v[a]

    def slow1(self) -> str:
        t = {a: self.s(f"1-1-{a}-W100-raw") for a in TREE1}
        return max(TREE1, key=lambda a: (t[a] if math.isfinite(t[a]) else -1, -TREE1.index(a)))

    def dl(self, arch: str) -> float:
        return self.s(f"dl-{'LSTMA' if arch == 'GRUA' else arch}")

    def blk(self, arch: str, rows: float) -> float:
        r = self.R.get(f"block-{self.px}") or {}
        base = self.s(f"block-{self.px}") * rows / max(1, int(r.get("train_rows") or 1))
        return base * self.dl(arch) / self.dl(self.px)      # 모든 구조: 그 구조 1 fold 측정 ÷ 대리 구조 1 fold 측정

    def top3_dl(self) -> list[str]:
        ref = float(self.I["rows"][FOLD])
        return sorted(DL_ARCHS, key=lambda a: (-(self.blk(a, ref) if math.isfinite(self.blk(a, ref)) else -1),
                                               DL_ARCHS.index(a)))[:3]

    def imp_target(self, kind: str) -> float:
        iid = {"boost": f"imp-tree-{self.slow1()}", "dl": f"imp-dl-{self.px}", "RF": "imp-RF", "LR": "imp-LR"}[kind]
        r = self.R.get(iid)
        return float("nan") if r is None else _num(r.get("seconds_per_target"))

    def imp_kind(self, s: str) -> tuple[str, float]:
        if s in TREE1:
            return "boost", 1.0
        if s in ("RF", "ET"):
            return "RF", 1.0
        if s == "LR":
            return "LR", 1.0
        return "dl", (self.dl(s) / self.dl(self.px) if s in COMBINED else 1.0)


def _parts_row(stage: str, count: int, parts: list[tuple[float, float, str]], lane: str = "seq") -> dict:
    n = sum(p[0] for p in parts)
    sec = sum(p[0] * p[1] for p in parts)
    srcs = []
    for p in parts:
        if p[2] not in srcs:
            srcs.append(p[2])
    return {"단계": stage, "A9_횟수": count, "셈한_횟수": n, "예상_초": sec, "lane": lane, "근거": " / ".join(srcs),
            "parts": [{"count": c, "seconds_each": s, "source": src} for c, s, src in parts]}


def estimate(R: Mapping[str, Mapping], chain: Mapping[str, Mapping], info: Mapping, *, proxy_arch: str, ak_arch: str
             ) -> dict:
    """부록 A-9 표의 줄마다 (횟수, 1회 시간, 근거) 묶음 → 예상 초. 병렬(1-1 CPU·1-2 GPU)은 긴 쪽만(§8)."""
    U = Units(R, chain, info, proxy_arch, ak_arch)
    rows_s, pos_s = info["rows"], info["pos"]
    r3 = float(rows_s[FOLD])
    sf = {s: rows_s[s] / r3 for s in rn.SPLIT_ORDER}
    S4, S3, sref = sum(sf.values()), sum(sf[f] for f in fd.CV_FOLDS), sf["refit"]
    C = info["cols"]
    cols1 = C["order1"]
    slow = U.slow1()
    st_name, st = U.slow_tree6()
    px = proxy_arch
    blocks = info["blocks"]
    non_full = [b for b in blocks if not b["full"]]
    no_tune = [b for b in blocks if not b["tune"]]
    fa = R.get(f"4a-{slow}") or {}
    fa_ok = fa.get("status") == "ok"
    A4 = U.s(f"4a-{slow}") if fa_ok else float("nan")
    rep = fa.get("replicated_rows") or {}
    red = fa.get("reduced_rows") or {}
    largest = fa.get("largest_split")
    m_set = _num((fa.get("setting") or {}).get("m"))     # 저장 json 의 null 은 읽으면 NaN
    per_h = int(m_set) if math.isfinite(m_set) else H_MAX
    rep_l = float(rep.get(largest, float("nan"))) if fa_ok else float("nan")

    def rep_ratio(s):
        return float(rep[s]) / rep_l if fa_ok else float("nan")

    def red_ratio(s):
        return float(red[s]) / r3 if red else float("nan")

    out = []
    # 1-1·1-2(나란히)
    p = [(4, U.t11(a, W, m) * S4 / 4, f"1-1 {a} 측정(W 10·100, 그 밖 W 는 칼럼 수 비례)") for a in TREE1 for W in W_GRID
         for m in MODES]
    out.append(_parts_row("1-1 트리", 240, p, "cpu"))
    out.append(_parts_row("1-2 LSTM", 40, [(4, U.t12(W) * S4 / 4, "1-2 LSTM 측정(W × 채널 수 비례)") for W in W_GRID], "gpu"))
    out.append(_parts_row("2-0", 0, []))
    base = U.s(f"1-1-{slow}-W100-raw") / cols1["W100|raw"] * S4 / 4
    out.append(_parts_row("2-1 k", 20, [(20, base * C["2-1"], f"1-1 {slow} W100 원값 × 칼럼 수 비례(A = 가장 느린 트리)")]))
    out.append(_parts_row("2-1b N_ma", 32, [
        (16, base * C["2-1b"], f"1-1 {slow} W100 원값 × 칼럼 수 비례"),
        (16, U.s("1-2-LSTM-W100") * C["dl_C_2-1b"] / C["dl_C_order1"] * S4 / 4, "1-2 LSTM W100 × 채널 수 비례")]))
    out.append(_parts_row("2-R", 40, [(4, base * c, f"1-1 {slow} W100 원값 × 칼럼 수 비례") for c in C["2-R"]]))
    # 3-0
    top3 = U.top3_dl()
    p = [(4, U.tree(a) * S4 / 4, "사슬 부스팅 측정" if a in TREE1 else f"{a} 1 fold 측정") for a in TREE6]
    p += [(4, U.dl(a) * S4 / 4, "딥러닝·결합 1 fold 측정") for a in DL_ARCHS]
    p += [(1, U.blk(a, b["rows"]), f"블록 학습 측정 × 행 수 비례(상위 3 = 가장 느린 {','.join(top3)})")
          for a in top3 for b in non_full]
    out.append(_parts_row("3-0", 97, p))
    # 3-0b
    G, Ut = C["targets"]["groups"], C["targets"]["units"]
    p = []
    for s in STRUCTURES13:
        u = U.tree(s) if s in TREE6 else U.dl(s)
        src = "사슬 부스팅 측정" if s in TREE1 else ("1 fold 측정" if s in TREE6 else "딥러닝·결합 1 fold 측정")
        p += [(ROUNDS_0_9 * G * 3, u * S3 / 3, src), (ROUNDS_0_9, u * sref, src),
              (ROUNDS_0_9 * Ut * 3, u * S3 / 3, src), (ROUNDS_0_9, u * sref, src)]
    out.append(_parts_row("3-0b 변수 빼기", 13884, p))
    # 3-1
    p = [(1, U.blk(px, b["rows"]), "블록 학습 측정 × 행 수 비례(AE·VAE 잰 적 없음)") for _ in range(4) for b in blocks]
    p += [(12, st * S4 / 4, f"트리 6종 중 가장 느린 {st_name}")]
    out.append(_parts_row("3-1 이상 점수", 88, p))
    out.append(_parts_row("3-2 스태킹", 60, [(60, st * S4 / 4, f"트리 6종 중 가장 느린 {st_name}")]))
    # 4-a·4-a0·미리보기
    p = [(1, A4 * rep_ratio(s), "4-a 대리 측정 × 행 수 비례") for s in rn.SPLIT_ORDER]
    p += [(1, st * red_ratio(s), f"{st_name} × 줄인 행 수 비례(4-a0)") for s in rn.SPLIT_ORDER]
    p += [(1, A4 * rep_ratio(s), "4-a 대리 측정(미리보기 — 상한)") for s in rn.SPLIT_ORDER]
    out.append(_parts_row("4-a·4-a0·미리보기", 12, p))
    out.append(_parts_row("4-c·4-c-L", 8, [(8, st * S4 / 4, f"{st_name}(품목 시료 수 모름 — 상한)")]))
    # 5·6
    p = [(10, st * min(5 * pos_s[s], rows_s[s]) / r3, f"{st_name} × 뽑은 행(양성 × 5) 비례(따로)") for s in rn.SPLIT_ORDER]
    p += [(10, A4 * (min(5 * pos_s[s] * per_h, rep[s]) / rep_l if fa_ok else float("nan")),
           "4-a 대리 × 뽑은 행 비례(쌓기)") for s in rn.SPLIT_ORDER]
    out.append(_parts_row("5·6", 80, p))
    p = [(1, st * red_ratio(s), f"{st_name} × 줄인 행 수(①, h = 1)") for s in rn.SPLIT_ORDER]
    p += [(1, U.dl(px) * red_ratio(s), f"dl-{px} × 줄인 행 수(② — 60칸 신경망 잰 적 없음, 상한)") for s in rn.SPLIT_ORDER]
    out.append(_parts_row("추가 판 ①·②", 8, p))
    r3x = (2 / 3) * info["n_known"] / r3
    p = [(3, U.t11(a, W, m) * r3x, "1-1 측정 × 행 수(시료 2/3)") for a in TREE1 for W in W_GRID for m in MODES]
    p += [(3, U.t12(W) * r3x, "1-2 측정 × 행 수(시료 2/3)") for W in W_GRID]
    out.append(_parts_row("추가 판 ③", 210, p))
    p = []
    for a in TREE6:
        if a in TREE1:
            u = U.s(f"1-1-{a}-W010-sum") * C["lims"] / cols1["W010|sum"]
            src = f"1-1 {a} W10 요약 × 칼럼 수 비례(LIMS 입력)"
        else:
            u = U.s(f"fold-{a}") * C["lims"] / C["proxy"]
            src = f"{a} 1 fold × 칼럼 수 비례(LIMS 입력)"
        p += [(1, u * b["rows"] / r3, src) for b in blocks]
    p += [(4, U.s("fold-LR") * C["meta"] / C["proxy"] * S4 / 4, "로지스틱 1 fold × 칼럼 수 비례(메타)")]
    out.append(_parts_row("추가 판 ④", 118, p))
    p = [(1, U.blk(a, b["rows"]), "블록 학습 측정 × 행 수 비례") for a in ("LSTMA", "GRUA", "TCN") for b in blocks]
    out.append(_parts_row("추가 판 ⑥", 57, p))
    out.append(_parts_row("추가 판 ⑦", 8, [
        (4, U.tree("XGB") * C["seq_tree"] / C["proxy"] * S4 / 4, "사슬 XGB × 칼럼 수 비례(⑦-T)"),
        (4, U.dl("TRF") * S4 / 4, "dl-TRF(⑦-D)")]))
    out.append(_parts_row("추가 판 ⑧", 4, [(4, st * S4 / 4, f"{st_name}(상한)")]))
    opt = U.s(f"optuna-{slow}")
    out.append(_parts_row("8-T 튜닝", 298, [(297, opt * S3 / 3, "Optuna 시도 1회 × 행 수 비례"),
                                          (1, opt * sref, "Optuna 시도 1회 × 행 수 비례(refit)")]))
    akt = U.s(f"ak-trial-{ak_arch}")
    p = [(57, akt, f"AutoKeras 시도 1회({ak_arch})")]
    p += [(1, U.blk(a, b["rows"]), "블록 학습 측정 × 행 수 비례") for a in top3 for b in no_tune]
    out.append(_parts_row("8 딥러닝(AutoKeras)", 111, p))

    # 학습 아닌 계산(부록 A-9)
    P3 = sum(info["pairs"][f] for f in fd.CV_FOLDS) / info["pairs"][FOLD]
    T = C["targets"]
    nt = []
    p = []
    for s in STRUCTURES13:
        kind, ratio = U.imp_kind(s)
        p.append((ROUNDS_0_9 * G + ROUNDS_0_9 * Ut, U.imp_target(kind) * ratio * P3,
                  f"그룹 중요도 1회({kind}) ÷ 섞은 묶음 수 × 섞는 묶음 수 × fold 1~3 쌍 비"))
    nt.append(_parts_row("3-0b 중요도(학습 아님)", 0, p))
    nt.append(_parts_row("2-R1 중요도(학습 아님)", 0, [(T["2r1"], U.imp_target("boost") * P3, "부스팅 그룹 중요도 묶음당")]))
    worst = max((U.imp_target(k) for k in ("boost", "dl", "RF", "LR")), key=lambda v: v if math.isfinite(v) else -1)
    nt.append(_parts_row("최종 모델 안 순열 중요도(학습 아님)", 0, [(T["final"], worst * P3, "중요도 묶음당 가장 느린 종류(상한)")]))
    inf = R.get(f"infer60-{px}") or {}
    tp = _num(inf.get("seconds_per_pair")) if inf.get("status") in ("ok", None) else float("nan")
    p = [(1, tp * b["role_pairs"], "60가지 추론 쌍당 × 역할 쌍 수(3-0 상위 3 OOF)") for _ in range(3) for b in non_full]
    p += [(1, tp * b["role_pairs"], "60가지 추론 쌍당 × 역할 쌍 수(3-1 AE·VAE)") for _ in range(4) for b in blocks]
    p += [(1, tp * b["role_pairs"], "60가지 추론 쌍당 × 역할 쌍 수(⑥)") for _ in range(3) for b in blocks]
    p += [(1, tp * b["role_pairs"], "60가지 추론 쌍당 × 역할 쌍 수(8-TD 값 모델)") for _ in range(3) for b in no_tune]
    nt.append(_parts_row("딥러닝 추론(학습 아님)", 0, p))

    lane = {"cpu": sum(r["예상_초"] for r in out if r["lane"] == "cpu"),
            "gpu": sum(r["예상_초"] for r in out if r["lane"] == "gpu")}
    seq = sum(r["예상_초"] for r in out if r["lane"] == "seq")
    train_total = max(lane["cpu"], lane["gpu"]) + seq
    nt_total = sum(r["예상_초"] for r in nt)
    total = train_total + nt_total
    counts = sum(r["A9_횟수"] for r in out)
    return {"rows": out, "non_training": nt, "lanes_1_1_vs_1_2": lane, "train_seconds": train_total,
            "non_training_seconds": nt_total, "total_seconds": total, "count_total": counts,
            "count_matches_A9": counts == A9_TOTAL and all(r["A9_횟수"] == r["셈한_횟수"] for r in out),
            "slowest_tree_1_1": slow, "slowest_tree6": st_name, "top3_dl": top3, "fold_ratio": sf,
            "fourA_status": fa.get("status"), "nan_rows": [r["단계"] for r in out + nt if not math.isfinite(r["예상_초"])]}


# ================================================================ 요약(측정표·판정·예상·AutoKeras·run_meta)
TIMING_COLS = ("seconds", "seconds_load", "seconds_build_train", "seconds_lr_read", "seconds_fit", "seconds_predict",
               "seconds_prepare", "seconds_reduction", "wall_seconds", "peak_rss_gb", "peak_rss_children_gb",
               "train_rows", "n_cols", "n_pairs", "n_points", "own_best_n", "n_targets", "seconds_per_target",
               "seconds_per_pair")


def write_summary(T: Timing, S, input_md5s: Mapping[str, str]) -> Path:
    cfg = T.cfg
    ts = _dt.datetime.now(KST)
    stem = f"summary_{ts.strftime('%Y%m%d-%H%M%S')}"
    sd = T.out / rn.dedupe_name(stem, [p.name for p in T.out.iterdir()])   # 덮어쓰지 않음
    sd.mkdir(parents=True)
    rows = []
    for iid in T.order:
        r = T.results[iid]
        g = r.get("gpu") or {}
        rows.append({"id": iid, "kind": T.specs[iid]["kind"], "lane": T.specs[iid]["lane"], "status": r.get("status"),
                     **{c: r.get(c) for c in TIMING_COLS}, "gpu_peak_allocated_gb": g.get("peak_allocated_gb"),
                     "gpu_peak_reserved_gb": g.get("peak_reserved_gb"), "error": r.get("error") or r.get("note")})
    pd.DataFrame(rows).to_csv(sd / "timing_table.csv", index=False, encoding="utf-8-sig")
    mx = []
    for a, c in T.chain.items():
        for h in c["history"]:
            mx.append({"algo": a, **h, "decided_max_iter": c["max_iter"], "상한": "상한" if c["cap"] else ""})
    pd.DataFrame(mx).to_csv(sd / "max_iter_decision.csv", index=False, encoding="utf-8-sig")
    cf.dump_scores_json(sd / "max_iter_decision.json", T.chain)
    pd.DataFrame(T.ak_rows).to_csv(sd / "ak_check.csv", index=False, encoding="utf-8-sig")
    info = {**row_info(S), "cols": column_info()}
    est = estimate(T.results, T.chain, info, proxy_arch=cfg["dl_proxy_arch"], ak_arch=cfg["autokeras"]["timing_arch"])
    start = _dt.datetime.fromisoformat(cfg["schedule_start"]) if cfg["schedule_start"] else ts
    end = start + _dt.timedelta(seconds=est["total_seconds"]) if math.isfinite(est["total_seconds"]) else None
    est.update(schedule_start=start.isoformat(timespec="minutes"),
               expected_end=end.isoformat(timespec="minutes") if end else None,
               total_days=est["total_seconds"] / 86400, note_plan_total="부록 A-9 0-9 예상 15,415(§8 본문은 15,419 — 계획서 숫자 다름)")
    tab = [{k: v for k, v in r.items() if k != "parts"} for r in est["rows"] + est["non_training"]]
    tab += [{"단계": "합계(학습 — 1-1·1-2 는 긴 쪽만)", "A9_횟수": est["count_total"], "예상_초": est["train_seconds"]},
            {"단계": "합계(학습 아닌 계산)", "예상_초": est["non_training_seconds"]},
            {"단계": "합계", "예상_초": est["total_seconds"], "근거": f"시작 {est['schedule_start']} → 끝 {est['expected_end']}"}]
    t = pd.DataFrame(tab)
    t["예상_시간"] = t["예상_초"] / 3600
    t.to_csv(sd / "estimate_table.csv", index=False, encoding="utf-8-sig")
    cf.dump_scores_json(sd / "estimate.json", {**est, "info": info})
    ak_map = cfg["autokeras"]
    cf.save_run_meta(sd / "run_meta.json", cfg, input_md5s, CODE_ROOT, n_jobs=cfg["n_jobs"],
                     extra={"items": T.order, "autokeras_mapping_used": ak_map,
                            "lib_versions": rn._lib_versions(), "smoke": cfg.get("smoke")})
    if T.tracker is not None:
        for f in ("timing_table.csv", "max_iter_decision.csv", "estimate_table.csv", "estimate.json", "ak_check.csv",
                  "run_meta.json"):
            T.tracker.log_artifact(T.stage_run_id, sd / f, "tables")
        T.tracker.log_metrics(T.stage_run_id, {"estimate_total_seconds": est["total_seconds"],
                                               "estimate_train_seconds": est["train_seconds"]})
        T.tracker.finish(T.stage_run_id)
    print(f"[0-9] 요약 {sd} — 예상 {est['total_seconds'] / 3600:.1f} 시간({est['total_days']:.1f} 일), 끝 {est['expected_end']}"
          + (f" / 측정 불가 줄: {est['nan_rows']}" if est["nan_rows"] else ""), flush=True)
    return sd


# ================================================================ 실행
def make_tracker(cfg):
    if not cfg["use_mlflow"]:
        return None
    from lst.stage_cli import make_tracker as mk

    return mk(cfg)


def main(argv=None, *, overrides: Mapping[str, str] | None = None, isolate: bool = True,
         check_resources: Callable = rs.check_resources, tracker_fn: Callable = make_tracker):
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="결과 뿌리(단계 0 결과·실험 로그)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--summary-only", action="store_true", help="측정 없이(끝난 항목만으로) 요약·예상만 다시")
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    inputs_root = Path(args.inputs).expanduser().resolve()
    from lst import evaluate as ev

    S = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    md5s = {**cf.inputs_md5(inputs_root, [cfg["samples"], cfg["theil_targets"]]),
            **cf.inputs_md5(rp.root, [cfg["sample_file"]]), **wr.input_md5s(cfg["context"], rp.root, inputs_root)}
    tracker = tracker_fn(cfg)
    T = Timing(cfg, rp.root, inputs_root, rp.out, overrides=overrides, tracker=tracker, isolate=isolate,
               check_resources=check_resources)
    T.env["input_md5s"] = md5s
    if args.summary_only:
        T.funcs = {k: "s0_9_timing:_refuse" for k in T.funcs}
    T.run_all()
    return write_summary(T, S, md5s)


def _refuse(**_kw):
    raise RuntimeError("--summary-only: 끝나지 않은 측정 항목이 있음")


if __name__ == "__main__":
    main()
