# -*- coding: utf-8 -*-
"""3차 설정 로더 + 2차·1차 코드를 고치지 않고 3차 쪽에서 맞추는 감싸기.

반드시 다른 3차 모듈보다 먼저 import (cv.py 가 맨 위에서 import). 순서:
  1) 설정 파일 읽기: 환경변수 EXP3_CONFIG(없으면 exp/3_시간순교차검증/config3.json), EXP3_OVERRIDE(JSON 문자열, 칸 덮어쓰기, 시험 실행·점검용)
  2) 기상 창 길이: 환경변수 EXP2_L 이 이미 있으면 그대로(run3.py --L), 없으면 설정 window_L -> EXP2_L (2차 e2cfg 가 import 때 읽음)
  3) 2차 e2cfg import(1차 src 를 sys.path 에 추가) -> 1차 config 모듈 값을 바꿈:
     - 자료 경로: DATA_DIR·SAMPLE_PATH·WEATHER_PATH = 설정 data.* (2차 e2fg·e2nb 의 파일 경로도 이 DATA_DIR 아래 - 그 모듈들은 이 뒤에 import 됨)
     - 1차 고정 분할 날짜(TRAIN_END 등): 모든 행이 2차 Ctx2 의 'train' 한 분할에 들어가도록 먼 미래로 바꿈.
       3차는 분할 이름을 쓰지 않고 행의 날짜로 직접 고른다(아래 BOUNDS). 1차 config.TRAIN_END 등의 원래 값은 쓰지 않음.
     - 2차 e2data.compute_groups(2차 고위험 품목 묶음, 옛 학습 기간 기준): 3차에서 쓰지 않으므로 빈 결과로 바꿈(새 자료에 버섯 코드가 없어도 멈추지 않게).
       3차 고위험 품목은 겹마다 그 겹 학습 자료로 다시 고름(cv.top_items).
     - 2차 e2nb.NB_N_SAMPLES(옛 시료 행 수 고정값) = 지금 시료 파일 행 수
  4) 날짜 경계 BOUNDS 계산(행 날짜 기준):
     데이터 끝 D = 설정 dates.data_end, 없으면 시료 파일 mnfctur_de 최댓값. D 뒤 행은 어디에도 안 씀.
     시험 시작 = (D + 1일) − 1년, 시험 행 = (행, h) 중 T = d − h ∈ [시험 시작, D − 60일].
     겹 j (n_folds 개, 시험 앞 해를 한 해씩): 그해 시작 = 시험 시작 − (n_folds − j + 1)년, 그해 끝 = 다음 해 시작 − 1일,
       맞혀 보는 행 = T ∈ [그해 시작, 그해 끝 − 60일], 학습 행 = d ≤ 그해 시작 − 1일.
     학습 자료 끝 = 시험 시작 − 1일(최종 재학습).
     dates.folds / dates.test 를 주면 그 값을 그대로 씀(각 {train_end, t_start, t_end} / {t_start, t_end}).

설정 칸 (config3.json)
  data.data_dir / sample_file / weather_file : 자료 폴더·시료 파일·기상 파일(폴더 기준 상대 경로). 환경변수 EXP3_DATA_DIR 이 있으면 폴더를 덮어씀
  data.nb_tfidf_by_train_end : {"겹 학습 끝 날짜": "그 날짜까지 학습 자료로 만든 단위날짜_tfidf 파일 경로", "final": "..."} - 그날 제품명 TF-IDF(n0, 9단계)용.
                               없으면 하나짜리 판(제품명/단위날짜_tfidf.parquet)을 쓰고 한계 기록
  dates.data_end / n_folds / folds / test : 위 4)
  window_L : 기상 창 길이(일, 10의 배수). run3.py --L 이 우선
  seed : 모델 시드(학습 행 h 배정 = default_rng(seed)·default_rng(seed+1))
  smoothing_k : 과거 이력 스무딩 계수 k. select=true 면 run3.py kselect 가 교차검증 3겹 평균(judge)으로 candidates 중 고름(model·featset 으로),
                false 면 fixed 고정 + 한계 기록('옛 검증 해로 고른 값')
  boost : 부스팅 최대 반복·조기 종료 인내
  tree_device : 트리 3종 계산 장치(cpu | gpu) - 한 방식으로 통일(장치 비교 결과로 정함)
  featset_all : '변수 전부' 묶음 목록(cv.BUNDLES 코드)
  threshold : 판정 기준값 규칙 = 60개 h 합친 맞혀 보는 행에서 Recall ≥ recall_min 중 Precision 최대(같으면 Recall 큰 쪽) - 보고 지표는 이 기준값에서
  optuna_objectives : Optuna 다목적 목표(겹 평균, 모두 최대화). 고를 수 있는 값 = pr_auc, f2(60개 h 합친 행에서 F2 최대 - 사양), p_at_r(Recall ≥ recall_min 에서의 최대 Precision),
                      top10 등 겹 점수 칸. 기본 = 사양대로 ["pr_auc", "f2"] (사용자 결정 대기)
  optuna : 시도 수·표본기(tpe)·씨앗·모델별 탐색 범위 [종류, 하한, 상한] (float, float_log, int, int_log, int_or_none, fixed)
  bundle_drop : 묶음 빼기 판단 규칙(no_decrease_any_increase = metrics 의 겹 평균 점수가 모두 낮아지지 않고(허용 tol) 하나라도 오르면 뺀 채로),
                metrics(null 이면 optuna_objectives 와 같은 점수),
                max_passes, n1_judge_folds(제품명 과거 유형 묶음 판단에 쓸 겹 - null 이면 모든 겹, [3] 이면 겹 3 만)
  group_items.top_n : 고위험 품목 수(겹마다 그 겹 학습 자료의 검출 비중 상위)
  smoke : 짧은 시험 실행(시군구 일부·학습 불검출 표본·h 마다 맞혀 보는 행 표본, 검출 행은 모두 유지)
  mem_floor_gb : 시스템 사용 가능 메모리가 이 값 아래면 이 프로세스만 멈춤
"""
import copy
import json
import os
import sys
from pathlib import Path

SRC3 = Path(__file__).resolve().parent
EXP3_DIR = SRC3.parent
EXP_ROOT = EXP3_DIR.parent
EXP2_SRC = EXP_ROOT / "2_고도화" / "src"
for _p in (str(SRC3), str(EXP2_SRC)):
    if _p not in sys.path:
        sys.path.insert(0 if _p == str(SRC3) else 1, _p)


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


CONFIG_PATH = Path(os.environ.get("EXP3_CONFIG") or (EXP3_DIR / "config3.json"))
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
if os.environ.get("EXP3_OVERRIDE"):
    _deep_update(CFG, json.loads(os.environ["EXP3_OVERRIDE"]))
if os.environ.get("EXP3_DATA_DIR"):
    CFG["data"]["data_dir"] = os.environ["EXP3_DATA_DIR"]
if not os.environ.get("EXP2_L"):
    os.environ["EXP2_L"] = str(int(CFG.get("window_L", 100)))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import e2cfg as E    # noqa: E402  (EXP2_L 읽음, 1차 src 를 sys.path 에 추가)
import config as C1  # noqa: E402

# ---------------------------------------------------------------- 1차 config 값 바꾸기(3차 프로세스 안에서만)
C1_ORIGINAL = {k: str(getattr(C1, k)) for k in ("DATA_DIR", "SAMPLE_PATH", "WEATHER_PATH", "TRAIN_END", "VAL_START", "VAL_END",
                                                 "TEST_START", "TEST_END")}
C1.DATA_DIR = Path(CFG["data"]["data_dir"])
C1.SAMPLE_PATH = C1.DATA_DIR / CFG["data"]["sample_file"]
C1.WEATHER_PATH = C1.DATA_DIR / CFG["data"]["weather_file"]
C1.TRAIN_END = "2200-12-31"                      # 모든 행 -> Ctx2 'train' 한 분할 (3차는 행 날짜로 직접 고름)
C1.VAL_START, C1.VAL_END = "2201-01-01", "2201-12-31"
C1.TEST_START, C1.TEST_END = "2202-01-01", "2202-12-31"

import e2data  # noqa: E402

e2data.compute_groups = lambda samples, rows=None: {}      # 2차 고위험 품목 묶음(옛 학습 기간 기준) - 3차에서 안 씀

H_MAX = int(E.H_MAX)
L = int(E.L)
SEED = int(CFG.get("seed", 42))


def sample_rows():
    return pq.ParquetFile(C1.SAMPLE_PATH).metadata.num_rows


import e2nb  # noqa: E402

e2nb.NB_N_SAMPLES = sample_rows()               # 2차 고정값(옛 시료 행 수) -> 지금 시료 파일 행 수


def _d(x):
    return np.datetime64(pd.Timestamp(x).date(), "D")


def _shift_years(d, n):
    return _d(pd.Timestamp(str(d)) - pd.DateOffset(years=n))


def data_end_from_samples():
    t = pq.read_table(C1.SAMPLE_PATH, columns=["mnfctur_de"])
    return _d(pc.max(t.column(0)).as_py())


def compute_bounds(cfg=None):
    cfg = cfg or CFG
    dc = cfg["dates"]
    D = _d(dc["data_end"]) if dc.get("data_end") else data_end_from_samples()
    hm = np.timedelta64(H_MAX, "D")
    one = np.timedelta64(1, "D")
    if dc.get("test"):
        ts, te = _d(dc["test"]["t_start"]), _d(dc["test"]["t_end"])
    else:
        ts = _shift_years(D + one, 1)
        te = D - hm
    folds = []
    if dc.get("folds"):
        for j, f in enumerate(dc["folds"], start=1):
            folds.append({"fold": j, "train_end": str(_d(f["train_end"])), "t_start": str(_d(f["t_start"])), "t_end": str(_d(f["t_end"]))})
    else:
        n = int(dc.get("n_folds", 3))
        for j in range(1, n + 1):
            ys = _shift_years(ts, n - j + 1)
            ye = _shift_years(ts, n - j) - one
            folds.append({"fold": j, "train_end": str(ys - one), "t_start": str(ys), "t_end": str(ye - hm)})
    learn_end = ts - one
    for f in folds:
        if not (_d(f["train_end"]) < _d(f["t_start"]) <= _d(f["t_end"])):
            raise ValueError(f"겹 경계 이상: {f}")
        if _d(f["t_end"]) + hm > learn_end:
            raise ValueError(f"겹 {f['fold']} 맞혀 보는 행의 기준일이 학습 자료 끝 {learn_end} 을 넘음")
    if not (ts <= te <= D - hm):
        raise ValueError(f"시험 경계 이상: {ts} ~ {te}, 데이터 끝 {D}")
    return {"data_end": str(D), "data_end_source": "설정값" if dc.get("data_end") else "시료 파일 mnfctur_de 최댓값",
            "test": {"t_start": str(ts), "t_end": str(te), "d_range": [str(ts + one), str(te + hm)]},
            "learn_end": str(learn_end), "folds": folds, "h_max": H_MAX,
            "h_rule": (f"학습 행 h: 기준일 d <= {folds[-1]['train_end']}(마지막 겹 학습 끝) 행 = default_rng(seed), "
                       f"그 뒤 ~ {learn_end}(학습 자료 끝) 행 = default_rng(seed+1), 각각 단위 키+d 정렬 순서로 integers(1, {H_MAX + 1})"),
            "rule": "시험 = 데이터 마지막 1년(T ∈ [시험 시작, 데이터 끝 − 60일]), 겹 = 그 앞 해를 한 해씩(같은 규칙), 학습 = 각 겹 그해 시작 전"}


BOUNDS = compute_bounds()
FOLDS = BOUNDS["folds"]
FOLD_BY_K = {f["fold"]: f for f in FOLDS}
SMOKE = CFG.get("smoke", {})
K_SMOOTH_FIXED = int(CFG["smoothing_k"]["fixed"])


def cfg_snapshot():
    return {"config_path": str(CONFIG_PATH), "config": copy.deepcopy(CFG), "bounds": BOUNDS, "L": L,
            "c1_original_values_not_used": C1_ORIGINAL,
            "c1_patched": {"DATA_DIR": str(C1.DATA_DIR), "SAMPLE_PATH": str(C1.SAMPLE_PATH), "WEATHER_PATH": str(C1.WEATHER_PATH),
                           "split_dates": "모든 행을 Ctx2 'train' 한 분할로(먼 미래 날짜) - 3차는 행 날짜로 고름"}}
