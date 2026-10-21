# -*- coding: utf-8 -*-
"""3차 모델 공통 인터페이스 - 트리 3종(XGBoost·LightGBM·CatBoost; 랜덤 포레스트·로지스틱은 3차에서 뺌, 클래스만 남김). 바닐라 = 라이브러리 기본 설정, 불균형 처리 없음.

  m = make_model(name, seed, device, log, params=None)   # params = 라이브러리 파라미터 dict(튜닝·불러오기, set_params 로 넣음)
  m.fit(tr, ev=None, ev_offs=None, rounds=None)          # ev = 그 겹 맞혀 보는 행(부스팅 조기 종료), rounds = 조기 종료 없이 고정 반복(최종 재학습)
  p = m.predict(x)                                        # (n,) float32
  m.importance() -> (n_cat + n_num,) 칸 중요도(범주 칸 먼저, 숫자 칸 다음 - Inputs.all_names() 순서)
  m.params_json() -> JSON 으로 저장할 파라미터(get_params 에서 함수·콜백 등 뺀 것) / load_params(path) -> dict
입력 Inputs: X 숫자 칸 float32(첫 칸 h), cat_str/codes 범주 칸(기본 4칸 sido·sgg·p2·p3 + 식품 묶음이면 aw_label), levels = 범주 고정 목록(전체 행 기준).

범주 처리
- XGBoost: pandas category(고정 목록) + 숫자 칸, enable_categorical=True, tree_method='hist' (파라미터 = sklearn XGBClassifier, 학습 = native, 평가 행 = 일반 DMatrix)
- LightGBM: pandas category + 숫자 칸(categorical_feature='auto')
- CatBoost: 범주 문자열 그대로(2차 Cat2 와 같은 FeaturesData: 숫자 칸 뒤 범주 칸)
- 랜덤 포레스트: 범주 = 정수 코드(순서 있는 숫자로 취급), 결측 = sklearn 결정나무 결측값 지원(NaN 그대로, 채움 없음)
- 로지스틱 회귀: 숫자 칸 학습 행 중앙값 채움 + 표준화, 범주 원-핫(학습 행에 없던 값은 0), 기본 C=1.0, max_iter=10000
조기 종료(부스팅 3종): 최대 반복·인내 = config3 boost(2000·100), 지표 = 맞혀 보는 행 PR-AUC(cv.ap60_fast).
  LightGBM metric='None'(기본 logloss 끔 - 조기 종료가 이 지표 하나만), XGBoost 는 사용자 지표면 기본 지표 꺼짐,
  CatBoost od_type='Iter'·od_wait·use_best_model=True(2차와 같음), GPU 면 metric_period=1(GPU 기본 5).
중요도: XGBoost total_gain, LightGBM gain, CatBoost PredictionValuesChange, 랜덤 포레스트 불순도 감소(MDI),
  로지스틱 회귀 |계수|(숫자 칸 = 표준화 계수, 범주 칸 = 그 칸 원-핫 계수 절댓값 합). 묶음 중요도는 cv 쪽에서 칸 합.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import cv as CV3
import cfg3 as G

MODELS = ["xgb", "lgbm", "cat"]                     # 3차 트리 모델(사용자 확정 10-20: xgb·lgbm·cat + BiLSTM, 랜덤 포레스트·로지스틱은 3차에서 뺌 - 클래스는 코드만 남김)
MODEL_NAMES = {"xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost", "rf": "랜덤 포레스트", "lr": "로지스틱 회귀", "bilstm": "BiLSTM"}
TREE_BOOST = ("xgb", "lgbm", "cat")
MAX_ROUNDS = int(G.CFG["boost"]["max_rounds"])
PATIENCE = int(G.CFG["boost"]["patience"])
N_JOBS = os.cpu_count() or 1
LR_MAX_ITER = 10000
PRED_CHUNK = 500000


class Inputs:
    def __init__(self, X, codes, cat_str, y, names, cat_names, levels):
        self.X, self.codes, self.cat_str, self.y = X, codes, cat_str, y
        self.names, self.cat_names, self.levels = list(names), list(cat_names), levels
        self.n = len(y)

    def all_names(self):
        return self.cat_names + self.names

    def rows(self, a, b):
        return Inputs(self.X[a:b], self.codes[a:b] if self.codes is not None else None,
                      self.cat_str[a:b] if self.cat_str is not None else None, self.y[a:b], self.names, self.cat_names, self.levels)

    def frame_cat(self):
        cdf = pd.DataFrame({c: pd.Categorical.from_codes(self.codes[:, j], categories=self.levels[c]) for j, c in enumerate(self.cat_names)})
        ndf = pd.DataFrame(self.X, columns=self.names, copy=False)
        return pd.concat([cdf, ndf], axis=1)

    def frame_lr(self):
        cdf = pd.DataFrame({c: self.cat_str[:, j].astype(str) for j, c in enumerate(self.cat_names)})
        ndf = pd.DataFrame(self.X, columns=self.names, copy=False)
        return pd.concat([cdf, ndf], axis=1)

    def matrix_rf(self):
        return np.ascontiguousarray(np.hstack([self.codes.astype(np.float32), self.X]), dtype=np.float32)


def _jsonable(v):
    if isinstance(v, (str, bool)) or v is None:
        return v, True
    if isinstance(v, (int, np.integer)):
        return int(v), True
    if isinstance(v, (float, np.floating)):
        return (None, False) if np.isnan(v) else (float(v), True)      # NaN(예: XGBoost missing 기본값)은 저장하지 않고 기본값으로 둠
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            j, ok = _jsonable(x)
            if not ok:
                return None, False
            out.append(j)
        return out, True
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            j, ok = _jsonable(x)
            if not ok:
                return None, False
            out[str(k)] = j
        return out, True
    return None, False


def clean_params(d):
    """get_params 결과에서 JSON 으로 옮길 수 있는 값만(함수·콜백·추정기 객체 제외) + 뺀 칸 이름."""
    out, dropped = {}, []
    for k, v in d.items():
        j, ok = _jsonable(v)
        if ok:
            out[k] = j
        else:
            dropped.append(k)
    return out, dropped


def save_params(path, model, extra=None):
    p, dropped = clean_params(model.est_params())
    obj = {"model": model.name, "params": p, "not_saved": dropped, **(extra or {})}
    CV3.jdump(obj, path)
    return obj


def load_params(path):
    return CV3.jload(path)["params"]


class Base:
    name = ""

    def __init__(self, seed, device, log, params=None):
        self.seed, self.device, self.log = seed, device, log
        self.params = dict(params or {})
        self.best_rounds = self.n_trained = self.fit_seconds = None
        self.es_history = None
        self.notes = {}
        self.est = self.make_est()
        if self.params:
            self.est.set_params(**self.params)
            self.apply_device()

    def apply_device(self):
        """불러온 파라미터의 장치 칸을 지금 실행 장치로(설정 tree_device / --device 로 통일)."""
        if self.name == "xgb":
            self.est.set_params(device=self.device_used())
        elif self.name == "lgbm":
            self.est.set_params(device_type="gpu" if self.device == "gpu" else "cpu")
        elif self.name == "cat":
            self.est.set_params(task_type=self.device_used())

    def device_used(self):
        return "cpu"

    def est_params(self):
        return self.est.get_params()

    def info(self):
        p, dropped = clean_params(self.est_params())
        return {"model": self.name, "device": self.device_used(), "get_params": p, "best_rounds": self.best_rounds,
                "n_trained": self.n_trained, "fit_seconds": self.fit_seconds,
                "es_best_value": (self.es_history[self.best_rounds - 1] if self.es_history and self.best_rounds else None),
                "notes": self.notes}


def _ap60_closure(offs):
    def ap60(y_true, y_score):
        assert len(y_true) == offs[-1, 1]
        return CV3.ap60_fast(np.asarray(y_true), np.asarray(y_score), offs)
    return ap60


# ---------------------------------------------------------------- XGBoost
_XGB_DM = {}                                 # 겹 자료 DMatrix 재사용(튜닝 시도마다 다시 만들지 않음): 키 = (배열 id, 모양, 범주 칸, 종류, max_bin)


def clear_dm_cache():
    """XGBoost DMatrix 재사용 캐시를 비움(설정 하나를 다 돌린 뒤 - 메모리용, 결과와 무관)."""
    _XGB_DM.clear()


def _xgb_dm(x, kind, max_bin=None):
    """학습 = QuantileDMatrix(max_bin), 맞혀 보는 행·예측 = 일반 DMatrix.
    맞혀 보는 행을 QuantileDMatrix(ref) 로 주면(sklearn 래퍼 기본) '변수 전부'에서 반복마다 평가 예측이 3.5초 걸림(일반 DMatrix 0.3초,
    2026-10-20 서버 측정 logs/xgb_probe2.log) -> 평가 행은 일반 DMatrix. 학습에는 영향 없음(평가 행은 나무 만들기에 안 쓰임)."""
    import xgboost as xgb
    key = (id(x.X), x.X.shape, tuple(x.cat_names), kind, max_bin)
    if key in _XGB_DM:
        return _XGB_DM[key][1]
    df = x.frame_cat()
    if kind == "train":
        dm = xgb.QuantileDMatrix(df, label=x.y, enable_categorical=True, **({"max_bin": int(max_bin)} if max_bin else {}))
    else:
        dm = xgb.DMatrix(df, label=x.y, enable_categorical=True)
    del df
    limit = 6 if x.X.nbytes < 2e9 else 2              # 큰 입력(그날 제품명 TF-IDF 포함 등)은 지금 겹의 학습·평가 하나씩만 둠(메모리)
    while len(_XGB_DM) >= limit:
        _XGB_DM.pop(next(iter(_XGB_DM)))
    _XGB_DM[key] = (x.X, dm)                 # 배열 참조를 같이 쥐어 id 재사용 방지
    return dm


class XGB(Base):
    """파라미터는 sklearn XGBClassifier(get_params 저장·set_params 불러오기), 학습은 native xgb.train(get_xgb_params) - 평가 행 일반 DMatrix."""
    name = "xgb"

    def device_used(self):
        return "cuda" if self.device == "gpu" else "cpu"

    def make_est(self):
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=MAX_ROUNDS, enable_categorical=True, tree_method="hist", device=self.device_used(),
                                 random_state=self.seed)

    def est_params(self):
        d = dict(self.est.get_params())
        d.pop("callbacks", None)
        d.pop("eval_metric", None)
        return d

    def fit(self, tr, ev=None, ev_offs=None, rounds=None):
        import xgboost as xgb
        t0 = time.time()
        xp = {k: v for k, v in self.est.get_xgb_params().items() if v is not None}
        mb = xp.get("max_bin")
        dtr = _xgb_dm(tr, "train", mb)
        if rounds is not None:
            self.bst = xgb.train(xp, dtr, num_boost_round=int(rounds))
            self.n_trained = self.best_rounds = int(self.bst.num_boosted_rounds())
        else:
            xp["disable_default_eval_metric"] = 1
            dev = _xgb_dm(ev, "eval")
            f = _ap60_closure(ev_offs)
            yv = np.asarray(ev.y)
            es = xgb.callback.EarlyStopping(rounds=PATIENCE, metric_name="ap60", data_name="validation_0", maximize=True, save_best=False)
            evr = {}
            self.bst = xgb.train(xp, dtr, num_boost_round=MAX_ROUNDS, evals=[(dev, "validation_0")],
                                 custom_metric=lambda pr, dm: ("ap60", f(yv, pr)), callbacks=[es], evals_result=evr, verbose_eval=False)
            hist = evr["validation_0"]
            self.notes["eval_keys"] = list(hist.keys())
            self.es_history = [float(x) for x in hist["ap60"]]
            self.best_rounds = int(self.bst.best_iteration) + 1
            self.n_trained = int(self.bst.num_boosted_rounds())
            if self.es_history[self.best_rounds - 1] != max(self.es_history):
                raise RuntimeError("xgboost 최적 반복의 지표가 기록 최대가 아님")
        self.notes["native_params"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in xp.items()}
        self.fit_seconds = time.time() - t0

    def predict(self, x):
        import xgboost as xgb
        dm = _XGB_DM.get((id(x.X), x.X.shape, tuple(x.cat_names), "eval", None))
        if dm is not None:
            return self.bst.predict(dm[1], iteration_range=(0, self.best_rounds)).astype(np.float32)
        out = np.empty(x.n, dtype=np.float32)
        for a in range(0, x.n, PRED_CHUNK):
            b = min(x.n, a + PRED_CHUNK)
            d = xgb.DMatrix(x.rows(a, b).frame_cat(), enable_categorical=True)
            out[a:b] = self.bst.predict(d, iteration_range=(0, self.best_rounds))
        return out

    def importance(self, x):
        sc = self.bst.get_score(importance_type="total_gain")
        return np.array([float(sc.get(n, 0.0)) for n in x.all_names()])


# ---------------------------------------------------------------- LightGBM
class LGBM(Base):
    name = "lgbm"

    def device_used(self):
        return "gpu(OpenCL)" if self.device == "gpu" else "cpu"

    def make_est(self):
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=MAX_ROUNDS, random_state=self.seed, verbose=-1, metric="None",
                                  device_type="gpu" if self.device == "gpu" else "cpu")

    def fit(self, tr, ev=None, ev_offs=None, rounds=None):
        import lightgbm as lgb
        t0 = time.time()
        Xtr = tr.frame_cat()
        if rounds is not None:
            self.est.set_params(n_estimators=int(rounds))
            self.est.fit(Xtr, tr.y)
            self.n_trained = self.best_rounds = int(self.est.booster_.num_trees())
        else:
            self.est.set_params(n_estimators=MAX_ROUNDS)
            f = _ap60_closure(ev_offs)
            Xev = ev.frame_cat()
            self.est.fit(Xtr, tr.y, eval_set=[(Xev, ev.y)], eval_metric=lambda yt, yp: ("ap60", f(yt, yp), True),
                         callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
            del Xev
            hist = self.est.evals_result_["valid_0"]
            if list(hist.keys()) != ["ap60"]:
                raise RuntimeError(f"lightgbm 평가지표가 ap60 하나가 아님: {list(hist.keys())}")
            self.es_history = [float(x) for x in hist["ap60"]]
            self.best_rounds = int(self.est.best_iteration_)
            self.n_trained = len(self.es_history)
            if self.es_history[self.best_rounds - 1] != max(self.es_history):
                raise RuntimeError("lightgbm 최적 반복의 지표가 기록 최대가 아님")
        self.fit_seconds = time.time() - t0

    def predict(self, x):
        out = np.empty(x.n, dtype=np.float32)
        for a in range(0, x.n, PRED_CHUNK):
            b = min(x.n, a + PRED_CHUNK)
            out[a:b] = self.est.predict_proba(x.rows(a, b).frame_cat(), num_iteration=self.best_rounds)[:, 1]
        return out

    def importance(self, x):
        imp = self.est.booster_.feature_importance(importance_type="gain", iteration=self.best_rounds)
        names = self.est.booster_.feature_name()
        mp = dict(zip(names, imp))
        return np.array([float(mp.get(n, 0.0)) for n in x.all_names()])


# ---------------------------------------------------------------- CatBoost
class AP60Cat:
    def __init__(self, offs):
        self.offs = offs

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        a = np.asarray(approxes[0])
        t = np.asarray(target)
        if len(a) != self.offs[-1, 1]:          # CatBoost 는 학습 자료에도 이 지표를 잼 -> 그때는 합친 AP(기록용, 조기 종료에 안 쓰임)
            return CV3.ap_fast(t, a), 1.0
        return CV3.ap60_fast(t, a, self.offs), 1.0

    def get_final_error(self, error, weight):
        return error


class Cat(Base):
    name = "cat"

    def device_used(self):
        return "GPU" if self.device == "gpu" else "CPU"

    def make_est(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=MAX_ROUNDS, random_seed=self.seed, task_type=self.device_used(), verbose=0,
                                  allow_writing_files=False)

    def est_params(self):
        d = dict(self.est.get_params())
        for k in ("eval_metric", "od_type", "od_wait", "use_best_model", "metric_period"):
            d.pop(k, None)
        return d

    def pool(self, x, with_label=True):
        from catboost import FeaturesData, Pool
        fd = FeaturesData(num_feature_data=np.ascontiguousarray(x.X, dtype=np.float32),
                          cat_feature_data=np.ascontiguousarray(x.cat_str), num_feature_names=list(x.names),
                          cat_feature_names=list(x.cat_names))
        return Pool(fd, label=x.y if with_label else None)

    def fit(self, tr, ev=None, ev_offs=None, rounds=None):
        t0 = time.time()
        ptr = self.pool(tr)
        if rounds is not None:
            # CatBoost 는 learning_rate 말고도 iterations 설정값이 나무 만들기에 영향(selftest3 catboost (b)·(d): 같은 학습률에서
            # iterations=2000 의 앞 31그루 = 2차 E00, iterations=31 은 다름; eval set 유무는 영향 없음) -> 교차검증과 같게
            # iterations = 최대 반복으로 학습한 뒤 앞 rounds 그루만 남김(shrink). 학습률은 튜닝 값, 없으면 교차검증 겹 모델 학습률 평균(fixed_lr).
            self.est.set_params(iterations=MAX_ROUNDS)
            lr = self.params.get("learning_rate", getattr(self, "fixed_lr", None))
            if lr is not None:
                self.est.set_params(learning_rate=float(lr))
            self.est.fit(ptr)
            self.n_trained = int(self.est.tree_count_)
            self.est.shrink(ntree_end=int(rounds))
            self.best_rounds = int(self.est.tree_count_)
            self.notes["final_rule"] = (f"iterations={MAX_ROUNDS} 로 학습 후 앞 {int(rounds)}그루만 남김(shrink) - CatBoost 는 iterations 설정이 나무에 영향")
        else:
            pev = self.pool(ev)
            metric = AP60Cat(ev_offs)
            kw = {"iterations": MAX_ROUNDS, "eval_metric": metric, "od_type": "Iter", "od_wait": PATIENCE, "use_best_model": True}
            if self.device == "gpu":
                kw["metric_period"] = 1
            self.est.set_params(**kw)
            self.est.fit(ptr, eval_set=pev)
            ev_hist = self.est.get_evals_result().get("validation", {})
            key = type(metric).__name__
            if key not in ev_hist:
                raise RuntimeError(f"catboost 평가 기록에 {key} 없음: {list(ev_hist)}")
            self.es_history = [float(x) for x in ev_hist[key]]
            self.best_rounds = int(self.est.get_best_iteration()) + 1
            self.n_trained = len(self.es_history)
            if self.es_history[self.best_rounds - 1] != max(self.es_history):
                raise RuntimeError("catboost 최적 반복의 지표가 기록 최대가 아님")
            if self.est.tree_count_ != self.best_rounds:
                raise RuntimeError(f"catboost tree_count {self.est.tree_count_} != best {self.best_rounds}")
            del pev
        self.notes["learning_rate_used"] = float(self.est.learning_rate_)
        self.fit_seconds = time.time() - t0

    def predict(self, x, ntree_end=0):
        out = np.empty(x.n, dtype=np.float32)
        for a in range(0, x.n, PRED_CHUNK):
            b = min(x.n, a + PRED_CHUNK)
            out[a:b] = self.est.predict(self.pool(x.rows(a, b), with_label=False), prediction_type="Probability",
                                        ntree_end=ntree_end)[:, 1].astype(np.float32)
        return out

    def importance(self, x):
        imp = self.est.get_feature_importance()
        names = self.est.feature_names_
        mp = dict(zip(names, imp))
        return np.array([float(mp.get(n, 0.0)) for n in x.all_names()])


# ---------------------------------------------------------------- 랜덤 포레스트
class RF(Base):
    name = "rf"

    def make_est(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=self.seed, n_jobs=N_JOBS)

    def fit(self, tr, ev=None, ev_offs=None, rounds=None):
        t0 = time.time()
        X = tr.matrix_rf()
        self.notes["input"] = "범주 = 정수 코드(전체 행 기준 고정 목록, 순서 있는 숫자로 취급) + 숫자 칸; 결측 = NaN 그대로(sklearn 결정나무 결측값 지원)"
        self.notes["train_nan_rate"] = float(np.isnan(X).mean())
        self.est.fit(X, tr.y)
        del X
        self.n_trained = self.best_rounds = int(len(self.est.estimators_))
        self.notes["mean_tree_depth"] = float(np.mean([e.get_depth() for e in self.est.estimators_]))
        self.fit_seconds = time.time() - t0

    def predict(self, x):
        out = np.empty(x.n, dtype=np.float32)
        for a in range(0, x.n, PRED_CHUNK):
            b = min(x.n, a + PRED_CHUNK)
            out[a:b] = self.est.predict_proba(x.rows(a, b).matrix_rf())[:, 1]
        return out

    def importance(self, x):
        return np.asarray(self.est.feature_importances_, dtype=np.float64)


# ---------------------------------------------------------------- 로지스틱 회귀
class LR(Base):
    name = "lr"

    def make_est(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=LR_MAX_ITER, random_state=self.seed)

    def _pipe(self, names, cat_names):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        return ColumnTransformer([("num", num, list(names)), ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_names))])

    def fit(self, tr, ev=None, ev_offs=None, rounds=None):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        t0 = time.time()
        self.pre = self._pipe(tr.names, tr.cat_names)
        Z = self.pre.fit_transform(tr.frame_lr())
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always", ConvergenceWarning)
            self.est.fit(Z, tr.y)
        conv_w = any(issubclass(w.category, ConvergenceWarning) for w in wl)
        imp = self.pre.named_transformers_["num"].named_steps["imp"]
        self.n_num_kept = len(imp.get_feature_names_out())
        self.notes.update({"n_iter": int(np.max(self.est.n_iter_)), "convergence_warning": conv_w,
                           "converged": bool(int(np.max(self.est.n_iter_)) < self.est.max_iter and not conv_w),
                           "n_numeric_dropped_all_nan": int(len(tr.names) - self.n_num_kept),
                           "n_design_columns": int(self.est.coef_.shape[1]), "train_nan_rate": float(np.isnan(tr.X).mean()),
                           "input": "숫자 칸: 학습 행 중앙값 채움 + 표준화, 범주: 원-핫(학습 행에 없던 값은 0)"})
        self.n_trained = self.best_rounds = self.notes["n_iter"]
        self.fit_seconds = time.time() - t0

    def predict(self, x):
        out = np.empty(x.n, dtype=np.float32)
        for a in range(0, x.n, PRED_CHUNK // 2):
            b = min(x.n, a + PRED_CHUNK // 2)
            out[a:b] = self.est.predict_proba(self.pre.transform(x.rows(a, b).frame_lr()))[:, 1]
        return out

    def importance(self, x):
        coef = np.abs(self.est.coef_[0])
        imp = self.pre.named_transformers_["num"].named_steps["imp"]
        kept = list(imp.get_feature_names_out())
        mp = dict(zip(kept, coef[:len(kept)]))
        oh = self.pre.named_transformers_["cat"]
        off = len(kept)
        cat_imp = {}
        for c, cats in zip(x.cat_names, oh.categories_):
            cat_imp[c] = float(coef[off:off + len(cats)].sum())
            off += len(cats)
        return np.array([cat_imp.get(n, mp.get(n, 0.0)) for n in x.all_names()], dtype=np.float64)


CLASSES = {"xgb": XGB, "lgbm": LGBM, "cat": Cat, "rf": RF, "lr": LR}


def make_model(name, seed, device, log, params=None):
    return CLASSES[name](seed, device if name in TREE_BOOST else "cpu", log, params)


def suggest(trial, space):
    """config3 optuna.spaces 한 모델 -> 파라미터 dict."""
    p = {}
    for k, spec in space.items():
        kind = spec[0]
        if kind == "float":
            p[k] = trial.suggest_float(k, spec[1], spec[2])
        elif kind == "float_log":
            p[k] = trial.suggest_float(k, spec[1], spec[2], log=True)
        elif kind == "int":
            p[k] = trial.suggest_int(k, spec[1], spec[2])
        elif kind == "int_log":
            p[k] = trial.suggest_int(k, spec[1], spec[2], log=True)
        elif kind == "int_or_none":
            use = trial.suggest_categorical(f"{k}_none", [False, True])
            p[k] = None if use else trial.suggest_int(k, spec[1], spec[2])
        elif kind == "fixed":
            p[k] = spec[1]
        elif kind == "cat":
            p[k] = trial.suggest_categorical(k, list(spec[1:]))
        else:
            raise ValueError(f"탐색 범위 종류 {kind}")
    return p
