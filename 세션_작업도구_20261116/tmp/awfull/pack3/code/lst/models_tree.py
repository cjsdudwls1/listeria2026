"""트리·표 모델 6종(XGBoost·LightGBM·CatBoost·RandomForest·ExtraTrees·로지스틱) — 계획서 §0·§3·§4-2·§6 3-0,
부록 A-7(단계 8 범위 — 지금은 파라미터를 받을 수 있게만)·A-12('범주 처리'·'학습률 읽기'), 부록 B-4·B-5.

공통 API(학습 틀 인터페이스 약속):
    m = make_model(algo, cat_cols=[...], indicator_cols=[...], max_iter=None|int, params=None|{...})
    m.fit(X, y, *, seed, n_jobs, eval_set=None) -> m
    m.predict_proba(X, n_iter=None) -> float64 (N,)        # 양성 확률, 부스팅만 n_iter(트리 수 지정)
    m.predict_proba_multi(X, n_iters) -> float64 (N, K)     # 부스팅 56 시점 한꺼번에(열 k = n_iters[k])
    m.save(path) / TreeModel.load(path)
    m.config_dump() -> dict(JSON 가능)                       # B-4 실제 적용값·범주 목록·표준화 통계 등
    m.non_default_params() -> dict                           # 라이브러리 기본값과 다른 생성 인자

입력 X = DataFrame — 숫자 칼럼(float32 권장, NaN 그대로), 범주 칼럼 = 원 문자열(cat_cols). 범주 변환은 fit 때
그 모델 train 값으로 정해 객체 안에 두고 predict 때 같게 씀(부록 A-12 '범주 처리'):
    XGBoost·LightGBM = pandas category(범주 목록 = train 값, 없던 값 = 결측; XGBoost enable_categorical=True,
                       LightGBM categorical_feature 지정)
    CatBoost         = 문자열 그대로(cat_features 지정)
    RF·ExtraTrees    = 정수 번호 1칼럼(번호 = 코드 문자열 정렬 순, train 값, 없던 범주 = NaN), 밀집 배열(B-4)
    로지스틱          = 원핫(0/1, 없던 범주 = 전부 0) + 숫자 칼럼만 train 평균·표준편차로 표준화 → 결측 0(§4-2),
                       결측 표시 칼럼(indicator_cols)은 입력 쪽에서 만든 0/1 을 그대로 받음
설정: 라이브러리 기본 그대로(§0·§3) — 바꾸는 것은 씨앗·n_jobs(≤ 8)·부스팅 최대 반복(max_iter, 2차 사슬)·
params(2차 사슬 CatBoost 학습률·단계 8 시도·Focal), 로지스틱 max_iter 2,000(§6 3-0).
반복 수(§3): `checkpoints`(56 시점)·`fold_curve`(채점 함수를 인자로)·`select_n_iter`(교차 N·공통 N)·
`next_max_iter`(0-9 최대 반복 2배), CatBoost 학습률 = `CatBoostModel.read_auto_learning_rate`(부록 A-12)·
`mean_learning_rate`.

구현 세부(계획서에 없음 — 담당자 확인):
 (1) 로그·파일만 바꾸는 인자: LightGBM verbose=-1, CatBoost verbose=False·allow_writing_files=False(작업 폴더에
     catboost_info 를 안 만듦). 모델 결과에는 영향 없음.
 (2) 범주 목록 순서 = train 값(str)의 파이썬 문자열 정렬 순(XGBoost·LightGBM category 순서, RF·ET 번호 0부터).
 (3) 범주 칼럼에 결측(None·NaN)이 있으면 멈춤 — 'aw_label' null → '결측', 품목 빈 칸 → '코드없음' 처럼 입력 조립
     쪽에서 문자열 범주로 바꿔 넘김(부록 A-4·RR3). 범주 값은 str 만.
 (4) 로지스틱 표준화: 평균·표준편차(ddof 0, NaN 뺌) float64. 표준편차 0 이면 나누는 값 1(sklearn StandardScaler 와
     같음), train 이 전부 NaN 인 칼럼은 평균 0·나누는 값 1(→ 모두 0) 로 두고 덤프에 기록. 결측 표시 칼럼은
     표준화하지 않음(원핫과 같이 0/1 그대로). 설계 행렬 칼럼 순서 = X 칼럼 순서(범주 칼럼 자리에 원핫 묶음).
 (5) LightGBM 의 부록 A-7 이름은 sklearn API 이름으로 바꿔 넘김: lambda_l2→reg_lambda, bagging_fraction→subsample,
     bagging_freq→subsample_freq, feature_fraction→colsample_bytree(같은 LightGBM 파라미터의 별칭).
 (6) predict 의 칼럼 목록은 fit 때와 이름·순서가 같아야 함(다르면 멈춤 — 칼럼 순서 = 설정 덤프 칼럼 목록 순, B-5).
 (7) LightGBM 이 '더 나눌 잎 없음' 으로 max_iter 전에 멈추면(조기 종료 아님) 만든 트리 수를 기록하고 n_iter 가 그
     보다 크면 만든 트리 전부로 예측(effective n_iter 기록). XGBoost·CatBoost 는 트리 수가 max_iter 와 다르면 멈춤.
 (8) save/load = pickle(같은 라이브러리 판에서만 load — 판이 다르면 멈춤), 있는 파일은 덮어쓰지 않음.
 (9) 반복 수 곡선: 시점마다 predict_proba(n_iter=k) 를 따로 부름(CatBoost staged 예측은 ntree_end 예측과 1e-16
     수준으로 달라 저장 예측 재채점이 어긋날 수 있어 안 씀).
 (10) 곡선 고르기(§3·§1-2): 시점 점수를 소수 4자리 반올림(metrics.round4)해 최댓값, 같으면 작은 N. 곡선에 NaN 이
     있으면 멈춤.
 (11) RF·ExtraTrees 예측(11-08): 나무별 예측은 스레드로 나란히, 더하기는 나무 순서로(`forest_proba_deterministic`) —
     sklearn 기본(n_jobs > 1)은 더하는 순서가 스레드에 따라 바뀌어 같은 입력도 값이 흔들림. 값 = sklearn n_jobs=1 과
     바이트 같음(시험), 학습은 그대로. 행마다 다른 행과 무관 → runner 의 같은 행 한 번 예측(ROW_DEDUPE_ALGS)에 넣음.
알아 둘 라이브러리 성질(로컬 .venv-exp2 = 서버와 같은 판에서 확인):
 - CatBoost 는 범주 칼럼이 있으면 thread_count 에 따라 결과가 달라짐(합성 자료에서 확인) → B-1 의 'n_jobs 줄이기'
   가 결과를 바꿈. `retry_n_jobs_ladder` 는 CatBoost·XGBoost·LightGBM 의 n_jobs 를 줄이지 않음.
 - CatBoost 는 iterations 에 따라 leaf_estimation_iterations 기본값이 바뀜(100 → 1, 200·300 → 10) → max_iter 로
   학습한 모델의 n_iter=k 예측은 'k 반복으로 따로 학습한 모델'과 다를 수 있음(계획서 방식 = 최대 반복까지 학습한
   뒤 트리 수 지정 예측, 그대로 따름). XGBoost·LightGBM 은 같음(시험).
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import resources as rs
from .folds import CV_FOLDS
from .metrics import round4

ALGOS = ("XGB", "LGBM", "CAT", "RF", "ET", "LR")
BOOSTING = ("XGB", "LGBM", "CAT")
LR_MAX_ITER = 2000                 # §6 3-0: 로지스틱 max_iter 2,000(수렴용)
MAX_ITER_DEFAULT_CHAIN = 2000      # §3: 2차 사슬 부스팅 최대 반복
MAX_ITER_CAP = 16000               # §3: 0-9 에서 늘리는 상한
CATBOOST_LR_READ_ITERATIONS = 2000  # 부록 A-12: 학습률 읽기는 최대 반복을 늘려도 2,000 고정

# 부록 A-7 단계 8 탐색 칼럼(+ 2차 사슬 CatBoost 학습률·Focal 손실·bootstrap_type) — 그 밖 이름은 멈춤
ALLOWED_PARAMS = {
    "XGB": ("learning_rate", "max_depth", "reg_lambda", "min_child_weight", "subsample", "colsample_bytree",
            "scale_pos_weight"),
    "LGBM": ("learning_rate", "num_leaves", "lambda_l2", "min_child_samples", "bagging_fraction", "bagging_freq",
             "feature_fraction", "scale_pos_weight"),
    "CAT": ("learning_rate", "depth", "l2_leaf_reg", "random_strength", "subsample", "rsm", "scale_pos_weight",
            "bootstrap_type", "loss_function"),
    "RF": ("max_depth", "min_samples_leaf", "max_features", "class_weight"),
    "ET": ("max_depth", "min_samples_leaf", "max_features", "class_weight"),
    "LR": ("C", "class_weight"),
}
LGBM_ALIASES = {"lambda_l2": "reg_lambda", "bagging_fraction": "subsample", "bagging_freq": "subsample_freq",
                "feature_fraction": "colsample_bytree"}  # 구현 세부 (5)

LIB_MODULES = ("xgboost", "lightgbm", "catboost", "sklearn", "numpy", "pandas")


def library_versions() -> dict[str, str]:
    import importlib

    out = {}
    for m in LIB_MODULES:
        out[m] = importlib.import_module(m).__version__
    return out


# ---------------------------------------------------------------- 반복 수(§3)
def checkpoints(max_iter: int = MAX_ITER_DEFAULT_CHAIN) -> list[int]:
    """채점 시점: 10 ~ 200 은 10 트리마다, 그 뒤 50 트리마다 max_iter 까지(2,000 → 56개)."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError(f"max_iter 는 정수: {max_iter!r}")
    max_iter = int(max_iter)
    if max_iter < 200 or (max_iter - 200) % 50 != 0:
        raise ValueError(f"max_iter 는 200 이상, 200 + 50 배수: {max_iter}")
    return list(range(10, 201, 10)) + list(range(250, max_iter + 1, 50))


def next_max_iter(max_iter: int) -> int | None:
    """0-9: fold 최적 N 이 마지막 시점이면 최대 반복 2배(4,000 → 8,000 …), 상한 16,000 — 넘으면 None('상한' 표시)."""
    nxt = int(max_iter) * 2
    return nxt if nxt <= MAX_ITER_CAP else None


def _argmax_round4(values: np.ndarray, points: Sequence[int]) -> int:
    """4자리 반올림 뒤 최댓값의 시점, 같으면 작은 N(points 오름차순)."""
    r = np.array([round4(float(v)) for v in values])
    best = np.max(r)
    return int(points[int(np.flatnonzero(r == best)[0])])


def fold_curve(pred_matrix: np.ndarray, y, h, u_tie, score_fn: Callable[..., float]) -> np.ndarray:
    """fold validation 쌍의 시점별 점수 곡선. pred_matrix = (쌍 수, 시점 수) float64(`predict_proba_multi`),
    score_fn(y, score, h, u_tie) → float(= metrics.pr_auc_band_weighted)."""
    p = np.asarray(pred_matrix, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("pred_matrix 는 (쌍 수, 시점 수)")
    return np.array([float(score_fn(y, p[:, k], h, u_tie)) for k in range(p.shape[1])], dtype=np.float64)


def select_n_iter(curves: Mapping[str, Sequence[float]], points: Sequence[int]) -> dict:
    """§3 교차 N·공통 N.

    curves = {'fold1': 곡선, 'fold2': …, 'fold3': …}(시점 순 구간 가중 PR-AUC), points = checkpoints(max_iter).
    교차 N(f) = 나머지 두 fold 평균 곡선 최고 시점, 공통 N = 세 fold 평균 곡선 최고 시점(같으면 작은 N, 4자리 반올림),
    fold f 점수 = 곡선 f 의 교차 N 값, 최적 N(f) = 자기 곡선 최고 시점(0-9 최대 반복 판정·기록용).
    """
    pts = [int(x) for x in points]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("points 는 오름차순")
    if set(curves) != set(CV_FOLDS):
        raise ValueError(f"curves 열쇠는 {CV_FOLDS}: {sorted(curves)}")
    c = {f: np.asarray(curves[f], dtype=np.float64) for f in CV_FOLDS}
    for f, a in c.items():
        if a.shape != (len(pts),):
            raise ValueError(f"{f} 곡선 길이 {a.shape} ≠ 시점 수 {len(pts)}")
        if not np.isfinite(a).all():
            raise ValueError(f"{f} 곡선에 NaN·inf — 반복 수를 고를 수 없음")
    idx = {n: i for i, n in enumerate(pts)}
    cross, fold_score, own = {}, {}, {}
    for f in CV_FOLDS:
        others = [g for g in CV_FOLDS if g != f]
        mean_other = np.mean(np.vstack([c[g] for g in others]), axis=0)
        cross[f] = _argmax_round4(mean_other, pts)
        fold_score[f] = float(c[f][idx[cross[f]]])
        own[f] = _argmax_round4(c[f], pts)
    mean_all = np.mean(np.vstack([c[f] for f in CV_FOLDS]), axis=0)
    common = _argmax_round4(mean_all, pts)
    last = pts[-1]
    return {
        "points": pts,
        "cross_n": cross,
        "common_n": common,
        "fold_score": fold_score,
        "cv_mean": float(np.mean([fold_score[f] for f in CV_FOLDS])),
        "own_best_n": own,
        "cross_n_at_cap": {f: cross[f] == last for f in CV_FOLDS},
        "common_n_at_cap": common == last,
        "any_at_cap": common == last or any(cross[f] == last for f in CV_FOLDS),  # 표 "상한" 표시
        "own_best_at_last": {f: own[f] == last for f in CV_FOLDS},             # 0-9 최대 반복 2배 판정
    }


def mean_learning_rate(rates: Sequence[float]) -> float:
    """CatBoost fold 별(5·6 은 씨앗 × fold) 자동 학습률 평균(§3·§6 5·6)."""
    a = np.asarray(list(rates), dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError(f"학습률 목록이 잘못됨: {rates!r}")
    return float(a.mean())


def retry_n_jobs_ladder(algo: str, n_jobs: int) -> tuple[int, ...]:
    """B-1 다시 돌리기 때 쓸 n_jobs 차례 — 결과를 바꾸지 않는 알고리즘만 줄임.

    RF·ExtraTrees(나무마다 씨앗이 n_jobs 와 무관)·로지스틱(sklearn 1.8 부터 n_jobs 효과 없음)만 n → n/2 → … → 1.
    CatBoost 는 범주 칼럼이 있으면 thread_count 로 결과가 달라지고(합성 자료 확인), XGBoost·LightGBM 은 합성 자료에선
    같았으나 보장 문서가 없어 줄이지 않음(같은 n_jobs 로 메모리를 확보한 뒤 다시).
    """
    _check_algo(algo)
    n = rs.check_n_jobs(n_jobs)
    if algo in ("RF", "ET", "LR"):
        out = [n]
        while out[-1] > 1:
            out.append(max(1, out[-1] // 2))
        return tuple(out)
    return (n,)


def _check_algo(algo: str) -> None:
    if algo not in ALGOS:
        raise ValueError(f"모르는 알고리즘: {algo!r} (가능: {ALGOS})")


# ---------------------------------------------------------------- 입력 점검
def _is_str_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)


def _check_cat_series(s: pd.Series, name: str) -> None:
    if not _is_str_dtype(s):
        raise TypeError(f"범주 칼럼 {name!r} 는 문자열(object/string) 이어야 함: {s.dtype}")
    if s.isna().any():
        raise ValueError(f"범주 칼럼 {name!r} 에 결측 {int(s.isna().sum())}행 — 입력 조립에서 문자열 범주로(구현 세부 3)")
    if len(s) and pd.api.types.infer_dtype(s, skipna=False) != "string":
        raise TypeError(f"범주 칼럼 {name!r} 에 str 아닌 값이 있음")


def _check_numeric_series(s: pd.Series, name: str) -> None:
    if pd.api.types.is_bool_dtype(s.dtype) or not pd.api.types.is_numeric_dtype(s.dtype):
        raise TypeError(f"숫자 칼럼 {name!r} 의 형이 숫자가 아님: {s.dtype}")
    if pd.api.types.is_float_dtype(s.dtype):
        v = s.to_numpy()
        if np.isinf(v).any():
            raise ValueError(f"숫자 칼럼 {name!r} 에 inf")


def _check_y(y, n: int) -> np.ndarray:
    a = np.asarray(y)
    if a.shape != (n,):
        raise ValueError(f"y 길이 {a.shape} ≠ X 행 수 {n}")
    if a.dtype == bool:
        a = a.astype(np.int64)
    if not np.isin(a, (0, 1)).all():
        raise ValueError("y 는 0/1 만")
    a = a.astype(np.int64)
    if a.min(initial=1) == a.max(initial=0):
        raise ValueError("y 가 한 값뿐 — 학습 불가")
    return a


def _jsonable(obj: Any) -> Any:
    """설정 덤프용 JSON 변환(numpy 수·NaN → None·dict 열쇠 → str)."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    return repr(obj)


# ---------------------------------------------------------------- 공통 바탕
class TreeModel:
    """트리·표 모델 바탕. 알고리즘별 하위 클래스가 `_make_estimator`·`_encode`·`_fit_estimator`·`_predict` 를 채움."""

    algo: str = ""

    def __init__(self, *, cat_cols: Sequence[str], indicator_cols: Sequence[str] = (),
                 max_iter: int | None = None, params: Mapping[str, Any] | None = None):
        _check_algo(self.algo)
        self.cat_cols = list(cat_cols)
        self.indicator_cols = list(indicator_cols)
        if len(set(self.cat_cols)) != len(self.cat_cols):
            raise ValueError("cat_cols 에 같은 이름")
        if set(self.cat_cols) & set(self.indicator_cols):
            raise ValueError("범주 칼럼과 결측 표시 칼럼이 겹침")
        if self.indicator_cols and self.algo != "LR":
            raise ValueError("indicator_cols 는 로지스틱만(트리 5종은 NaN 그대로 — §4-2)")
        if max_iter is not None:
            if self.algo not in BOOSTING:
                raise ValueError(f"{self.algo} 는 max_iter 를 받지 않음(기본 설정)")
            if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
                raise ValueError(f"max_iter 는 양의 정수: {max_iter!r}")
            max_iter = int(max_iter)
        self.max_iter = max_iter
        params = dict(params or {})
        bad = sorted(set(params) - set(ALLOWED_PARAMS[self.algo]))
        if bad:
            raise ValueError(f"{self.algo} 가 받지 않는 파라미터(계획서 부록 A-7 밖): {bad}")
        self._validate_params(params)
        self.params = params
        # fit 뒤 채움
        self.columns_: list[str] | None = None
        self.categories_: dict[str, list[str]] = {}
        self.seed_: int | None = None
        self.n_jobs_: int | None = None
        self.estimator_ = None
        self.fit_info_: dict[str, Any] = {}
        self.versions_: dict[str, str] = {}

    # ---- 하위 클래스가 덮어씀
    def _validate_params(self, params: dict) -> None:
        pass

    def _make_estimator(self, seed: int, n_jobs: int):
        raise NotImplementedError

    def _fit_estimator(self, est, Xe, y: np.ndarray) -> None:
        est.fit(Xe, y)

    def _encode(self, X: pd.DataFrame):
        raise NotImplementedError

    def _fit_encoders(self, X: pd.DataFrame) -> None:
        """범주 목록 = train 값(str 정렬 순, 구현 세부 2)."""
        self.categories_ = {c: sorted(pd.unique(X[c]).tolist()) for c in self.cat_cols}

    def _predict(self, Xe, n_iter: int | None) -> np.ndarray:
        if n_iter is not None:
            raise ValueError(f"{self.algo} 는 트리 수 지정 예측(n_iter)이 없음")
        return self.estimator_.predict_proba(Xe)[:, 1]

    def _n_trees_built(self) -> int | None:
        return None

    def _applied_params(self) -> dict:
        return {"get_params": self.estimator_.get_params()}

    def _default_estimator(self):
        raise NotImplementedError

    # ---- 점검
    def _check_X(self, X: pd.DataFrame, fitting: bool) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 는 pandas DataFrame")
        cols = list(X.columns)
        if not all(isinstance(c, str) for c in cols):
            raise TypeError("X 칼럼 이름은 str 만")
        if len(set(cols)) != len(cols):
            raise ValueError("X 칼럼 이름 중복")
        if fitting:
            miss = [c for c in self.cat_cols + self.indicator_cols if c not in cols]
            if miss:
                raise ValueError(f"X 에 없는 칼럼: {miss}")
        else:
            if cols != self.columns_:
                raise ValueError("predict X 칼럼 목록(이름·순서)이 fit 때와 다름(구현 세부 6)")
        for c in cols:
            if c in self.cat_cols:
                _check_cat_series(X[c], c)
            else:
                _check_numeric_series(X[c], c)
                if c in self.indicator_cols:
                    v = X[c].to_numpy()
                    if not np.isin(v, (0, 1)).all():
                        raise ValueError(f"결측 표시 칼럼 {c!r} 은 0/1 만(NaN 불가)")

    # ---- 공개 API
    def fit(self, X: pd.DataFrame, y, *, seed: int, n_jobs: int, eval_set=None) -> "TreeModel":
        """train 으로 범주 변환을 정하고 학습. eval_set·조기 종료는 안 씀(B-4) — None 이 아니면 멈춤."""
        if eval_set is not None:
            raise ValueError("트리 모델은 eval_set·조기 종료를 쓰지 않음(부록 B-4)")
        if self.estimator_ is not None:
            raise RuntimeError("이미 학습한 객체 — 새 객체로(make_model)")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed 는 정수: {seed!r}")
        n_jobs = rs.check_n_jobs(n_jobs)
        self._check_X(X, fitting=True)
        yy = _check_y(y, len(X))
        self.columns_ = list(X.columns)
        self._fit_encoders(X)
        Xe = self._encode(X)
        est = self._make_estimator(int(seed), n_jobs)
        t0 = time.time()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            self._fit_estimator(est, Xe, yy)
        self.estimator_ = est
        self.seed_, self.n_jobs_ = int(seed), n_jobs
        self.versions_ = library_versions()
        self.fit_info_ = {
            "n_rows": int(len(X)), "n_pos": int(yy.sum()), "fit_seconds": time.time() - t0,
            "warnings": [f"{w.category.__name__}: {w.message}" for w in wlist][:50],
        }
        self._after_fit(wlist)
        return self

    def _after_fit(self, wlist) -> None:
        n_built = self._n_trees_built()
        if n_built is not None:
            self.fit_info_["n_trees_built"] = int(n_built)

    def _require_fitted(self) -> None:
        if self.estimator_ is None:
            raise RuntimeError("fit 전")

    def predict_proba(self, X: pd.DataFrame, n_iter: int | None = None) -> np.ndarray:
        """양성 확률 float64 (N,). 부스팅만 n_iter(앞 n_iter 개 트리로 예측, B-4)."""
        self._require_fitted()
        if n_iter is not None:
            n_iter = self._check_n_iter(n_iter)
        self._check_X(X, fitting=False)
        Xe = self._encode(X)
        p = np.asarray(self._predict(Xe, n_iter), dtype=np.float64).reshape(-1)
        if p.shape != (len(X),):
            raise RuntimeError("예측 길이가 다름")
        return p

    def predict_proba_multi(self, X: pd.DataFrame, n_iters: Sequence[int]) -> np.ndarray:
        """부스팅 시점별 예측 (N, K) float64 — 열 k = predict_proba(X, n_iter=n_iters[k]) 와 같음(구현 세부 9)."""
        self._require_fitted()
        if self.algo not in BOOSTING:
            raise ValueError(f"{self.algo} 는 트리 수 지정 예측이 없음")
        ks = [self._check_n_iter(k) for k in n_iters]
        self._check_X(X, fitting=False)
        Xe = self._encode(X)
        out = np.empty((len(X), len(ks)), dtype=np.float64)
        for j, k in enumerate(ks):
            out[:, j] = np.asarray(self._predict(Xe, k), dtype=np.float64).reshape(-1)
        return out

    def _check_n_iter(self, n_iter) -> int:
        if self.algo not in BOOSTING:
            raise ValueError(f"{self.algo} 는 트리 수 지정 예측(n_iter)이 없음")
        if isinstance(n_iter, bool) or not isinstance(n_iter, (int, np.integer)) or int(n_iter) < 1:
            raise ValueError(f"n_iter 는 양의 정수: {n_iter!r}")
        n_iter = int(n_iter)
        cap = self.fit_info_.get("n_iter_requested")
        if cap is not None and n_iter > cap:
            raise ValueError(f"n_iter {n_iter} > 학습한 최대 반복 {cap}")
        return n_iter

    def non_default_params(self) -> dict:
        """라이브러리 기본 생성자와 다른 생성 인자(기본 설정 그대로인지 점검용)."""
        self._require_fitted()
        ours = self.estimator_.get_params()
        base = self._default_estimator().get_params()
        out = {}
        for k in sorted(set(ours) | set(base)):
            a, b = ours.get(k, "<없음>"), base.get(k, "<없음>")
            same = (a is b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
            if not same:
                try:
                    same = bool(a == b)
                except Exception:  # noqa: BLE001 — 비교 불가 값은 다름으로
                    same = False
            if not same:
                out[k] = a
        return out

    def config_dump(self) -> dict:
        """설정 덤프(B-2·B-4): 알고리즘·생성 인자·칼럼 목록(순서)·범주 목록·실제 적용 파라미터·라이브러리 판."""
        self._require_fitted()
        d = {
            "algo": self.algo,
            "cat_cols": self.cat_cols,
            "indicator_cols": self.indicator_cols,
            "max_iter": self.max_iter,
            "params": self.params,
            "seed": self.seed_,
            "n_jobs": self.n_jobs_,
            "columns": self.columns_,
            "categories": self.categories_,
            "fit_info": self.fit_info_,
            "non_default_params": self.non_default_params(),
            "applied_params": self._applied_params(),
            "library_versions": self.versions_,
        }
        d.update(self._extra_dump())
        return _jsonable(d)

    def _extra_dump(self) -> dict:
        return {}

    # ---- 저장
    def save(self, path: str | os.PathLike, overwrite: bool = False) -> Path:
        """pickle 저장(구현 세부 8). 있는 파일은 덮어쓰지 않음."""
        self._require_fitted()
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"이미 있음(덮어쓰지 않음): {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=5)
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TreeModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, TreeModel):
            raise TypeError(f"TreeModel 이 아님: {type(obj)}")
        if cls is not TreeModel and not isinstance(obj, cls):
            raise TypeError(f"{cls.__name__} 이 아님: {type(obj).__name__}")
        now = library_versions()
        if obj.versions_ != now:
            raise RuntimeError(f"라이브러리 판이 저장 때와 다름: 저장 {obj.versions_} / 지금 {now}")
        return obj


# ---------------------------------------------------------------- 범주 → pandas category(XGB·LGBM)
class _CategoryEncoded(TreeModel):
    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.cat_cols:
            return X
        Xe = X.copy(deep=False)
        for c in self.cat_cols:
            # 범주 목록 = train 값, 없던 값 = 결측(부록 A-12)
            Xe[c] = pd.Categorical(X[c].to_numpy(), categories=self.categories_[c], ordered=False)
        return Xe


class XGBModel(_CategoryEncoded):
    algo = "XGB"

    def _default_estimator(self):
        from xgboost import XGBClassifier

        return XGBClassifier()

    def _make_estimator(self, seed, n_jobs):
        from xgboost import XGBClassifier

        kw = dict(self.params)
        if self.max_iter is not None:
            kw["n_estimators"] = self.max_iter
        return XGBClassifier(enable_categorical=True, random_state=seed, n_jobs=n_jobs, **kw)

    def _after_fit(self, wlist):
        n_req = self.estimator_.get_params()["n_estimators"]
        n_built = int(self.estimator_.get_booster().num_boosted_rounds())
        self.fit_info_["n_iter_requested"] = n_built if n_req is None else int(n_req)
        self.fit_info_["n_trees_built"] = n_built
        if n_built != self.fit_info_["n_iter_requested"]:
            raise RuntimeError(f"XGBoost 트리 수 {n_built} ≠ 요청 {n_req}")

    def _predict(self, Xe, n_iter):
        if n_iter is None:
            return self.estimator_.predict_proba(Xe)[:, 1]
        return self.estimator_.predict_proba(Xe, iteration_range=(0, n_iter))[:, 1]

    def _applied_params(self):
        cfg = json.loads(self.estimator_.get_booster().save_config())
        return {"get_params": self.estimator_.get_params(), "save_config": cfg}


class LGBMModel(_CategoryEncoded):
    algo = "LGBM"

    def _default_estimator(self):
        from lightgbm import LGBMClassifier

        return LGBMClassifier()

    def _make_estimator(self, seed, n_jobs):
        from lightgbm import LGBMClassifier

        kw = {LGBM_ALIASES.get(k, k): v for k, v in self.params.items()}
        if self.max_iter is not None:
            kw["n_estimators"] = self.max_iter
        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **kw)

    def _fit_estimator(self, est, Xe, y):
        est.fit(Xe, y, categorical_feature=list(self.cat_cols) if self.cat_cols else "auto")

    def _after_fit(self, wlist):
        n_req = int(self.estimator_.get_params()["n_estimators"])
        n_built = int(self.estimator_.booster_.current_iteration())
        self.fit_info_["n_iter_requested"] = n_req
        self.fit_info_["n_trees_built"] = n_built
        self.fit_info_["stopped_no_split"] = n_built < n_req  # 구현 세부 7(조기 종료 아님)

    def _predict(self, Xe, n_iter):
        if n_iter is None:
            return self.estimator_.predict_proba(Xe)[:, 1]
        eff = min(n_iter, self.fit_info_["n_trees_built"])  # 구현 세부 7
        return self.estimator_.predict_proba(Xe, num_iteration=eff)[:, 1]

    def _applied_params(self):
        return {"get_params": self.estimator_.get_params(), "booster_params": dict(self.estimator_.booster_.params)}


# ---------------------------------------------------------------- CatBoost(문자열 범주)
class _StopAfterFirst:
    """첫 반복 뒤 멈추는 콜백(부록 A-12 학습률 읽기)."""

    def after_iteration(self, info) -> bool:
        return False


class CatBoostModel(TreeModel):
    algo = "CAT"

    def _validate_params(self, params):
        lf = params.get("loss_function")
        if lf is not None and not (lf == "Logloss" or (isinstance(lf, str) and lf.startswith("Focal:"))):
            raise ValueError(f"CatBoost loss_function 은 'Logloss' 또는 'Focal:…' 만(§6 단계 8): {lf!r}")

    def _default_estimator(self):
        from catboost import CatBoostClassifier

        return CatBoostClassifier()

    def _base_kwargs(self, seed, n_jobs) -> dict:
        return dict(random_seed=seed, thread_count=n_jobs, verbose=False, allow_writing_files=False)

    def _make_estimator(self, seed, n_jobs):
        from catboost import CatBoostClassifier

        kw = dict(self.params)
        if self.max_iter is not None:
            kw["iterations"] = self.max_iter
        return CatBoostClassifier(**self._base_kwargs(seed, n_jobs), **kw)

    def _fit_estimator(self, est, Xe, y):
        # use_best_model 은 eval_set 이 없으면 False(B-4) — 덤프에서 확인
        est.fit(Xe, y, cat_features=list(self.cat_cols) if self.cat_cols else None)

    def _encode(self, X):
        return X  # 문자열 그대로(부록 A-12)

    def _after_fit(self, wlist):
        allp = self.estimator_.get_all_params()
        n_req = int(allp["iterations"])
        n_built = int(self.estimator_.tree_count_)
        self.fit_info_["n_iter_requested"] = n_req
        self.fit_info_["n_trees_built"] = n_built
        self.fit_info_["learning_rate_applied"] = float(allp["learning_rate"])
        if n_built != n_req:
            raise RuntimeError(f"CatBoost 트리 수 {n_built} ≠ 요청 {n_req}")
        if allp.get("use_best_model") not in (False, None):
            raise RuntimeError("CatBoost use_best_model 이 켜짐(B-4)")

    def _predict(self, Xe, n_iter):
        if n_iter is None:
            return self.estimator_.predict_proba(Xe)[:, 1]
        return self.estimator_.predict_proba(Xe, ntree_end=n_iter)[:, 1]

    def _applied_params(self):
        return {"get_params": self.estimator_.get_params(), "get_all_params": self.estimator_.get_all_params()}

    def read_auto_learning_rate(self, X: pd.DataFrame, y, *, seed: int, n_jobs: int) -> float:
        """부록 A-12 학습률 읽기: iterations = 2,000 으로 학습을 시작해 첫 반복 뒤 콜백으로 멈추고 정해진 learning_rate.

        이 객체(학습 전)의 params(learning_rate 제외)·범주 처리 그대로, 학습 횟수 아님. 이 객체는 바꾸지 않음.
        """
        from catboost import CatBoostClassifier

        if "learning_rate" in self.params:
            raise ValueError("learning_rate 를 이미 정한 설정 — 자동 학습률을 읽을 수 없음")
        n_jobs = rs.check_n_jobs(n_jobs)
        self._check_X(X, fitting=True)
        yy = _check_y(y, len(X))
        kw = dict(self.params)
        kw["iterations"] = CATBOOST_LR_READ_ITERATIONS
        est = CatBoostClassifier(**self._base_kwargs(int(seed), n_jobs), **kw)
        est.fit(X, yy, cat_features=list(self.cat_cols) if self.cat_cols else None, callbacks=[_StopAfterFirst()])
        if int(est.tree_count_) != 1:
            raise RuntimeError(f"학습률 읽기: 첫 반복 뒤 멈추지 않음(트리 {est.tree_count_})")
        return float(est.get_all_params()["learning_rate"])


# ---------------------------------------------------------------- RF·ExtraTrees(정수 번호 1칼럼)
class _CodeEncoded(TreeModel):
    def _validate_params(self, params):
        cw = params.get("class_weight")
        if cw is not None and not (isinstance(cw, Mapping) and set(cw) == {0, 1}):
            raise ValueError("class_weight 는 {0: 1, 1: w} 꼴(부록 A-7 양성 가중)")

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        """밀집 float32 배열(B-4), 범주 칼럼 = 정렬 순 번호(0부터), 없던 범주 = NaN."""
        out = np.empty((len(X), len(self.columns_)), dtype=np.float32)
        for j, c in enumerate(self.columns_):
            if c in self.cat_cols:
                codes = pd.Categorical(X[c].to_numpy(), categories=self.categories_[c]).codes.astype(np.float32)
                codes[codes < 0] = np.nan
                out[:, j] = codes
            else:
                out[:, j] = X[c].to_numpy(dtype=np.float32, na_value=np.nan)
        return out

    def _after_fit(self, wlist):
        cls = list(self.estimator_.classes_)
        if cls != [0, 1]:
            raise RuntimeError(f"classes_ = {cls}")

    def _predict(self, Xe, n_iter):
        if n_iter is not None:
            raise ValueError(f"{self.algo} 는 트리 수 지정 예측(n_iter)이 없음")
        return forest_proba_deterministic(self.estimator_, Xe, self.n_jobs_ or 1)


FOREST_PRED_CHUNK = 65536   # 구현 세부 (11): 행 묶음(결과와 무관 — 메모리만)


def forest_proba_deterministic(est, X: np.ndarray, n_jobs: int, chunk: int = FOREST_PRED_CHUNK) -> np.ndarray:
    """RF·ExtraTrees 양성 확률 — 실행마다 같은 값(구현 세부 11).

    sklearn 의 forest predict_proba 는 n_jobs > 1 이면 나무 예측을 스레드가 끝나는 순서대로 더해(잠금 안 덧셈) 같은 입력도
    마지막 자리가 흔들림. 여기서는 나무마다 예측은 스레드로 나란히 하고, 더하기는 늘 나무 순서(0 → n−1)로 함 →
    sklearn n_jobs=1 결과와 바이트 단위로 같음(n_jobs·행 묶음·다른 행과 무관). 학습 결과는 바꾸지 않음(예측만)."""
    from joblib import Parallel, delayed

    Xv = est._validate_X_predict(X)
    trees = est.estimators_
    n = Xv.shape[0]
    k = int(np.atleast_1d(est.n_classes_)[0])
    out = np.empty(n, dtype=np.float64)
    par = Parallel(n_jobs=max(1, int(n_jobs)), prefer="threads")
    for lo in range(0, n, chunk):
        Xc = Xv[lo:lo + chunk]
        parts = par(delayed(t.predict_proba)(Xc, check_input=False) for t in trees)   # 결과 순서 = 나무 순서
        acc = np.zeros((Xc.shape[0], k), dtype=np.float64)
        for p in parts:
            acc += p
        acc /= len(trees)
        out[lo:lo + Xc.shape[0]] = acc[:, 1]
    return out


class RFModel(_CodeEncoded):
    algo = "RF"

    def _default_estimator(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier()

    def _make_estimator(self, seed, n_jobs):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **self.params)


class ETModel(_CodeEncoded):
    algo = "ET"

    def _default_estimator(self):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier()

    def _make_estimator(self, seed, n_jobs):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs, **self.params)


# ---------------------------------------------------------------- 로지스틱(원핫 + 표준화 + 결측 0)
class LogisticModel(TreeModel):
    algo = "LR"

    def _validate_params(self, params):
        cw = params.get("class_weight")
        if cw is not None and not (isinstance(cw, Mapping) and set(cw) == {0, 1}):
            raise ValueError("class_weight 는 {0: 1, 1: w} 꼴(부록 A-7 양성 가중)")

    def _default_estimator(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression()

    def _make_estimator(self, seed, n_jobs):
        # n_jobs 는 sklearn 1.8 부터 효과 없음(넘기면 FutureWarning) — 넘기지 않음, 스레드는 BLAS 환경변수
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=seed, max_iter=LR_MAX_ITER, **self.params)

    def _fit_encoders(self, X):
        super()._fit_encoders(X)
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols and c not in self.indicator_cols]
        mean, scale, all_nan = {}, {}, []
        for c in self.num_cols_:
            v = X[c].to_numpy(dtype=np.float64, na_value=np.nan)
            ok = ~np.isnan(v)
            if not ok.any():
                mean[c], scale[c] = 0.0, 1.0  # 구현 세부 4
                all_nan.append(c)
                continue
            m = float(v[ok].mean())
            s = float(np.sqrt(np.mean((v[ok] - m) ** 2)))  # ddof 0
            mean[c], scale[c] = m, (s if s > 0 else 1.0)
        self.mean_, self.scale_, self.all_nan_cols_ = mean, scale, all_nan
        names = []
        for c in X.columns:
            if c in self.cat_cols:
                names += [f"{c}=={v}" for v in self.categories_[c]]
            else:
                names.append(c)
        self.design_columns_ = names

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c in self.columns_:
            if c in self.cat_cols:
                cats = self.categories_[c]
                codes = pd.Categorical(X[c].to_numpy(), categories=cats).codes
                oh = np.zeros((len(X), len(cats)), dtype=np.float64)
                ok = codes >= 0  # 없던 범주 = 전부 0
                oh[np.flatnonzero(ok), codes[ok]] = 1.0
                blocks.append(oh)
            elif c in self.indicator_cols:
                blocks.append(X[c].to_numpy(dtype=np.float64).reshape(-1, 1))
            else:
                v = X[c].to_numpy(dtype=np.float64, na_value=np.nan)
                z = (v - self.mean_[c]) / self.scale_[c]
                z[np.isnan(z)] = 0.0  # 표준화 → 결측 0(§4-2)
                blocks.append(z.reshape(-1, 1))
        out = np.hstack(blocks) if blocks else np.empty((len(X), 0))
        if out.shape[1] != len(self.design_columns_):
            raise RuntimeError("설계 행렬 칼럼 수가 다름")
        return out

    def _after_fit(self, wlist):
        from sklearn.exceptions import ConvergenceWarning

        n_iter = int(np.max(self.estimator_.n_iter_))
        conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in wlist)
        self.fit_info_["n_iter_"] = n_iter                       # §6 3-0: 판마다 수렴 여부 기록
        self.fit_info_["converged"] = (not conv_warn) and n_iter < LR_MAX_ITER
        if list(self.estimator_.classes_) != [0, 1]:
            raise RuntimeError(f"classes_ = {list(self.estimator_.classes_)}")

    def _extra_dump(self):
        return {"standardize": {"mean": self.mean_, "scale": self.scale_, "all_nan_cols": self.all_nan_cols_},
                "design_columns": self.design_columns_}


MODEL_CLASSES: dict[str, type[TreeModel]] = {
    "XGB": XGBModel, "LGBM": LGBMModel, "CAT": CatBoostModel, "RF": RFModel, "ET": ETModel, "LR": LogisticModel,
}


def make_model(algo: str, *, cat_cols: Sequence[str], indicator_cols: Sequence[str] = (),
               max_iter: int | None = None, params: Mapping[str, Any] | None = None) -> TreeModel:
    """알고리즘 이름(ALGOS) → 학습 전 모델 객체.

    max_iter: None = 라이브러리 기본 반복 수(1차·단계 2 준비·③·④), 정수 = 2차 사슬 최대 반복(§3, 기본 2,000).
    params: 부록 A-7 이름(+ CatBoost learning_rate·loss_function·bootstrap_type) — 빈 dict = 라이브러리 기본.
    """
    _check_algo(algo)
    return MODEL_CLASSES[algo](cat_cols=cat_cols, indicator_cols=indicator_cols, max_iter=max_iter, params=params)
