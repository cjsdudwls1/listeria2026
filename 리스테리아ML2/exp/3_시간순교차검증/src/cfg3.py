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
  4) 날짜 경계 BOUNDS 계산(행 날짜 기준, 사양 1절):
     데이터 끝 D = 설정 dates.data_end, 없으면 시료 파일 mnfctur_de 최댓값. D 뒤 행은 어디에도 안 씀.
     시험 시작 = (D + 1일) − 1년, 시험 행 = (행, h) 중 T = d − h ∈ [시험 시작, D − 60일].
     겹 j (n_folds 개, 시험 앞 해를 한 해씩): 그해 시작 = 시험 시작 − (n_folds − j + 1)년, 그해 끝 = 다음 해 시작 − 1일,
       맞혀 보는 행 = T ∈ [그해 시작, 그해 끝 − 60일], 학습 행 = d ≤ 그해 시작 − 1일(겹 학습 끝).
     최종(리핏) 학습 끝 = 시험 시작 − 1일. 결과 시차는 학습 범위에 반영하지 않음(사양 1절).
     dates.folds / dates.test 를 주면 그 값을 그대로 씀(각 {train_end, t_start, t_end} / {t_start, t_end}).

설정 칸 (config3.json)
  data.data_dir / sample_file / weather_file : 자료 폴더·시료 파일·기상 파일(폴더 기준 상대 경로). 환경변수 EXP3_DATA_DIR 이 있으면 폴더를 덮어씀
  data.nb_tfidf_by_train_end : {"겹 학습 끝 날짜": "그 날짜까지 학습 행으로 만든 단위날짜_tfidf 파일 경로", "final": "..."} - 그날 제품명 TF-IDF(n0)용.
                               없으면 이름 규칙(제품명/단위날짜_tfidf_train<날짜>.parquet), 그것도 없으면 하나짜리 판을 쓰고 한계 기록
  dates.data_end / n_folds / folds / test : 위 4)
  window_L : 기상 창 길이(일, 10의 배수). run3.py --L 이 우선
  seed : 모델 시드(학습 행 h 배정 = default_rng(seed)·default_rng(seed+1))
  smoothing_k : 과거 이력 스무딩 계수 k. run3.py kselect 가 고르기 점수(교차검증 세 겹 평균 구간 가중 PR-AUC)로 candidates 중 고름
                (튜닝 전, 1차 베이스라인 (모델, 입력, 창) + 과거 이력 핵심 칸 hist_core). select=false 로 끄면 fixed 고정(한계 기록)
  threshold : 판정 기준값 규칙 = 60개 h 합친 맞혀 보는 행에서 Recall ≥ recall_min 중 Precision 최대(같으면 Recall 큰 쪽) - 보고 지표는 이 기준값에서
  optuna_objectives : 튜닝 목표(겹 평균, 최대화) = ["wpr_auc"] (사양 3절 고르기 점수 = 구간 가중 PR-AUC 하나)
  optuna_top_n / optuna : 후보 표에 표시할 상위 수 / 시도 수·표본기·씨앗·모델별 탐색 범위
                          [종류, 하한, 상한] (float, float_log, int, int_log, int_or_none, cat, fixed). 튜닝 모델 = 1차 베이스라인 모델 하나
  vanilla_pairs : 바닐라 (모델:입력) 목록 - 입력 = summary(기상 요약)·raw(일별 원값)·seq(BiLSTM 시퀀스), 창은 chain.Ls 전부
  chain : 이어 돌리기 설정(Ls 창 목록, kselect = 바닐라 뒤 k 고르기까지)
  boost : 부스팅 최대 반복·조기 종료 인내
  tree_device : 트리 3종 계산 장치(cpu | gpu) - 한 방식으로 통일
  featset_all / featset_tune : '변수 전부' 묶음 목록 / 튜닝 입력 묶음(그날 제품명 n2·n0 포함 - 운영 전제)
  bundle_drop : 묶음 빼기 - metrics(null 이면 optuna_objectives = 구간 가중 PR-AUC), rule(no_decrease_any_increase = 모두 안 낮아지고 하나라도 오르면 뺌;
                점수 하나면 '구간 가중 PR-AUC 가 오르면 뺌'), importance(permutation = 묶음 섞기 중요도), perm_seed, perm_scope(h_block = h 마다 행끼리 섞음),
                max_passes, n1_judge_folds(null = 모든 겹)
  group_items.top_n : 고위험 품목 수(겹마다 그 겹 학습 행의 검출 비중 상위)
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


class NbData3(e2nb.NbData):
    """2차 e2nb.NbData 와 같은 계산, 뭉치 수만 파일(뭉치_정의.csv 칼럼명 = 시료별_뭉치 nb_* 칸)에서 읽음.
    2차는 '뭉치 21개'를 고정으로 확인하는데, 10-20 새 정의(학습 자료 전체 ≤ 2025-06-30 재선정)는 23개라 3차 쪽에서 그 확인만 바꿈(2차 코드는 그대로).
    N1 칸 수 = 뭉치 수 x 2(비율·스무딩 검출률) x 2(5년·1년), N2 = 뭉치 수 - 이름은 2차 n1_names·n2_names 가 정의 파일에서 읽음."""

    def __init__(self, ctx, need=("n1", "n2", "n0"), log=None):
        import time as _t
        t0 = _t.time()
        self.check = {}
        H = ctx.H
        smp = ctx.samples
        self.clusters = e2nb.cluster_cols()
        pf = pq.ParquetFile(e2nb.NB_SAMPLE_PATH)
        nb_cols = [c for c in pf.schema_arrow.names if c.startswith("nb_")]
        if nb_cols != self.clusters:
            raise ValueError(f"시료별_뭉치 칸 순서가 뭉치_정의.csv 와 다름: {nb_cols} vs {self.clusters}")
        if not self.clusters:
            raise ValueError("뭉치가 하나도 없음")
        t = pq.read_table(e2nb.NB_SAMPLE_PATH).to_pandas()
        key_s = pq.read_table(C1.SAMPLE_PATH, columns=["lims_data_key"]).column(0).to_numpy(zero_copy_only=False).astype(str)
        if len(t) != len(smp) or len(t) != e2nb.NB_N_SAMPLES or len(key_s) != len(t):
            raise ValueError(f"시료별_뭉치 행 수 {len(t)} / 시료 {len(smp)} / 키 {len(key_s)} (시료 파일 {e2nb.NB_N_SAMPLES})")
        same_key = bool(np.array_equal(t["lims_data_key"].to_numpy().astype(str), key_s))
        if not same_key:
            raise ValueError("시료별_뭉치 lims_data_key 가 시료_시군구.parquet 행 순서와 다름")
        Fm = t[self.clusters].to_numpy()
        if Fm.dtype != np.int8 or not np.isin(Fm, [0, 1]).all():
            raise ValueError("뭉치 칸이 int8 0/1 이 아님")
        self.flags = Fm
        self.check["nb_samples"] = {"rows": int(len(t)), "lims_data_key_same_order": same_key, "n_clusters": len(self.clusters),
                                    "flag_sums": {c: int(v) for c, v in zip(self.clusters, Fm.sum(0))}, "file": str(e2nb.NB_SAMPLE_PATH.name)}
        self.s_unit = H.group_ids(smp, "unit")
        self.s_day = H.day_idx(smp["d"].to_numpy())
        rows_full = ctx.rows_full
        r_unit = H.group_ids(rows_full, "unit")
        r_day = H.day_idx(rows_full["d"].to_numpy())
        M = H.Dh + 1
        r_key = r_unit * M + r_day
        if len(np.unique(r_key)) != len(r_key):
            raise ValueError("단위x날짜 행 키 중복")
        self._r_order = np.argsort(r_key, kind="mergesort")
        self._r_sorted = r_key[self._r_order]
        self.n_rows_full = len(rows_full)
        keep_idx = np.flatnonzero(ctx._keep)
        self.frow = {sp: keep_idx[ctx.sp[sp]["idx"]] for sp in C1.SPLITS}
        self.hist = self.N2 = self.N0 = None
        if "n1" in need:
            t1 = _t.time()
            self.hist = e2nb.NbHist(self.s_unit, self.s_day, H.Dh, Fm, smp["pos"].to_numpy())
            self.check["n1_seconds"] = _t.time() - t1
        if "n2" in need:
            ridx = self._row_of(self.s_unit * M + self.s_day)
            N2 = np.zeros((self.n_rows_full, len(self.clusters)), dtype=np.int8)
            np.maximum.at(N2, ridx, Fm)
            cnt = np.bincount(ridx, minlength=self.n_rows_full)
            if (cnt == 0).any():
                raise ValueError("시료가 하나도 없는 단위x날짜 행")
            if not np.array_equal(cnt, rows_full["n_samples"].to_numpy()):
                raise ValueError("행별 시료 수가 rows_full n_samples 와 다름")
            self.N2 = N2
            self.check["n2"] = {"rows_with_any": int(N2.any(1).sum()), "rows": int(len(N2))}
        if "n0" in need:
            self.N0, self.check["n0"] = self._load_tfidf(H, M, rows_full)
            self.check["n0"]["file"] = str(e2nb.TFIDF_PATH.name)
        self.seconds = _t.time() - t0
        if log:
            log(f"  제품명 자료 준비 {self.seconds:.1f}s: {e2nb.NB_SAMPLE_PATH.name} {len(t):,}행(키 순서 같음), 뭉치 {len(self.clusters)}개"
                + (f", N2 행 {len(self.N2):,}" if self.N2 is not None else "")
                + (f", N0 {e2nb.TFIDF_PATH.name} {self.N0.shape} (키 일치 {self.check['n0']['matched']:,}행, n_rows = 시료 수 "
                   f"{self.check['n0']['n_rows_eq_n_samples']})" if self.N0 is not None else ""))


e2nb.NbData = NbData3                            # e2nb.get_nb 가 이 클래스로 만듦


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
    cv_end = ts - one
    for f in folds:
        if not (_d(f["train_end"]) < _d(f["t_start"]) <= _d(f["t_end"])):
            raise ValueError(f"겹 경계 이상: {f}")
        if _d(f["t_end"]) + hm > cv_end:
            raise ValueError(f"겹 {f['fold']} 맞혀 보는 행의 기준일이 시험 시작 전날 {cv_end} 을 넘음")
        if _d(f["train_end"]) > learn_end:
            raise ValueError(f"겹 {f['fold']} 학습 끝이 최종 학습 끝 {learn_end} 뒤")
    if not (ts <= te <= D - hm):
        raise ValueError(f"시험 경계 이상: {ts} ~ {te}, 데이터 끝 {D}")
    return {"data_end": str(D), "data_end_source": "설정값" if dc.get("data_end") else "시료 파일 mnfctur_de 최댓값",
            "test": {"t_start": str(ts), "t_end": str(te), "d_range": [str(ts + one), str(te + hm)]},
            "learn_end": str(learn_end), "cv_end": str(cv_end), "folds": folds, "h_max": H_MAX,
            "h_rule": (f"학습 행 h: 기준일 d <= {folds[-1]['train_end']}(마지막 겹 학습 끝) 행 = default_rng(seed), "
                       f"그 뒤 ~ {learn_end}(최종 학습 끝) 행 = default_rng(seed+1), 각각 단위 키+d 정렬 순서로 integers(1, {H_MAX + 1})"),
            "rule": ("시험 = 데이터 마지막 1년(T ∈ [시험 시작, 데이터 끝 − 60일]), 겹 = 그 앞 해를 한 해씩(같은 규칙), "
                     "학습 = 각 겹 맞혀 보는 해 시작 전날까지(d ≤ 그해 시작 − 1일), 최종 = 시험 시작 전날까지. 결과 시차는 학습 범위에 반영하지 않음")}


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
