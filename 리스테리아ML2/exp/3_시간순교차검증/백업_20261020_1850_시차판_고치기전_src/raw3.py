# -*- coding: utf-8 -*-
"""바닐라 1단계 '일별 원값' 입력(1차 'raw': 창 L일 x 19변수 = 19L칸, lag j = T−j) - 트리 3종을 한 겹에서 같이 돌린다.

맞혀 보는 행 입력이 겹마다 약 170만 (행,h) x 1,901칸 ≈ 13GB(float32)라 캐시도, 메모리에 통째로 두기도 하지 않는다:
  1) 그 겹 학습 행 입력(약 22만 x 1,901 ≈ 1.7GB, 메모리)으로 XGBoost·LightGBM·CatBoost 를 최대 반복(2,000)까지 eval set 없이 학습
     - XGBoost·LightGBM: eval set 은 나무 만들기에 쓰이지 않으므로 '조기 종료로 멈춘 모델'의 앞부분과 같은 나무
     - CatBoost: iterations 설정값이 나무에 영향을 주므로(selftest3 catboost (d)) 조기 종료 때와 같은 iterations=2000, 학습률은 조기 종료 조건
       (eval set + use_best_model)에서 CatBoost 가 자동으로 정하는 값을 미리 구해 고정(같은 조건 = 같은 나무, selftest3 catboost (d)·(e))
  2) 맞혀 보는 행을 h 조각(약 2.8만 행)마다 한 번 만들어 세 모델의 단계별 점수(반복 1..2000 의 누적 점수)를 계산 -> 반복마다 그 h 의 AP
     (XGBoost·LightGBM = 나무별 잎 번호(pred_leaf) -> 잎 값 누적합, CatBoost = staged_predict)
  3) h 1..60 평균 AP 곡선으로 조기 종료(인내 100, 최고 지표 = 처음 나온 최고)를 그대로 재현 -> 최적 반복
  4) 맞혀 보는 행을 h 조각마다 다시 만들어 최적 반복까지의 예측(확률) -> 채점·저장
점수 곡선은 누적 점수(raw margin) 순위로 계산(확률과 순위 같음). 조기 종료 재현 기록 = es_history(반복별 맞혀 보는 행 PR-AUC).
"""
import time

import numpy as np
import pandas as pd

import cfg3 as G
import cv as CV3
import features as F1
import models3 as M3

V = 19


def raw_names(L):
    return ["h"] + F1.feature_names("raw", L)


def raw_X(ctx, rs, L, a=0, b=None):
    """[h, raw 19L] float32 (1차 build_tree_matrix 'raw' 그대로)."""
    b = rs.n if b is None else b
    return F1.build_tree_matrix(ctx.weather.W, rs.s[a:b], rs.t[a:b], rs.h[a:b].astype(np.int16), CV3.L, "raw")


def cat_auto_lr(tr_pool, ev_pool, seed):
    """CatBoost 가 (학습 행 수, iterations=최대 반복, eval set + use_best_model) 조건에서 자동으로 정하는 학습률(몇 반복 만에 멈춰 읽음)."""
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=M3.MAX_ROUNDS, random_seed=seed, od_type="Iter", od_wait=1, use_best_model=True,
                           task_type="CPU", verbose=0, allow_writing_files=False)
    m.fit(tr_pool, eval_set=ev_pool)
    return float(m.learning_rate_)


def _leaf_values_xgb(bst):
    df = bst.trees_to_dataframe()
    T = int(df["Tree"].max()) + 1
    nmax = int(df["Node"].max()) + 1
    LV = np.zeros((T, nmax), dtype=np.float64)
    lf = df[df["Feature"] == "Leaf"]
    LV[lf["Tree"].to_numpy(), lf["Node"].to_numpy()] = lf["Gain"].to_numpy()
    return LV


def _leaf_values_lgb(bst):
    d = bst.dump_model()
    trees = d["tree_info"]
    T = len(trees)
    nl = max(t["num_leaves"] for t in trees)
    LV = np.zeros((T, nl), dtype=np.float64)
    for ti, t in enumerate(trees):
        st = [t["tree_structure"]]
        while st:
            nd = st.pop()
            if "leaf_index" in nd:
                LV[ti, nd["leaf_index"]] = nd["leaf_value"]
            elif "leaf_value" in nd and "split_index" not in nd:      # 잎 하나뿐인 나무
                LV[ti, 0] = nd["leaf_value"]
            else:
                st.append(nd["left_child"])
                st.append(nd["right_child"])
    return LV


def staged_ap(y, M):
    """M (n, T) 누적 점수 -> 반복마다 AP (T,) - cv.ap_fast 와 같은 식(동점 묶음 끝에서 정밀도)을 열 단위로."""
    n, T = M.shape
    P = int(y.sum())
    if P == 0 or P == n:
        return np.full(T, np.nan)
    out = np.empty(T)
    for a in range(0, T, 250):
        b = min(T, a + 250)
        S = M[:, a:b]
        o = np.argsort(-S, axis=0, kind="quicksort")
        Ss = np.take_along_axis(S, o, axis=0)
        ys = y[o].astype(np.int32)
        tp = np.cumsum(ys, axis=0)
        idx = np.arange(1, n + 1)[:, None]
        is_end = np.vstack([np.diff(Ss, axis=0) != 0, np.ones((1, b - a), dtype=bool)])
        # 위치마다 그 동점 묶음 끝의 정밀도
        pos_end = np.where(is_end, np.arange(n)[:, None], n)
        nxt = np.minimum.accumulate(pos_end[::-1], axis=0)[::-1]
        prec_end = np.take_along_axis(tp / idx, nxt, axis=0)
        out[a:b] = (ys * prec_end).sum(axis=0) / P
    return out


def simulate_es(curve, patience):
    """반복별 지표(최대가 좋음) -> 라이브러리 조기 종료와 같은 규칙: 지금까지 최고보다 크면 갱신, 최고 뒤 patience 반복 동안 갱신 없으면 멈춤.
    돌려줌 (최적 반복 수(1부터), 학습한 것으로 볼 반복 수)."""
    best_v, best_t = -np.inf, 0
    for t, v in enumerate(curve):
        if v > best_v:
            best_v, best_t = v, t
        elif t - best_t >= patience:
            return best_t + 1, t + 1
    return best_t + 1, len(curve)


def run_fold(ctx, rs_tr, rs_ev, X_tr, cats_tr, cats_ev, levels, models, seed, log):
    """한 겹: 세 모델(최대 반복) 학습 -> 단계별 AP -> 조기 종료 재현 -> 최적 반복 예측. 돌려줌 {모델: (p, info)}."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier, FeaturesData, Pool
    L = CV3.L
    names = raw_names(L)
    cat_names = list(CV3.BASE_CATS)
    trI = M3.Inputs(X_tr, cats_tr[1], cats_tr[0], rs_tr.y, names, cat_names, levels)
    offs = rs_ev.h_offsets()
    R = M3.MAX_ROUNDS
    fitted, info = {}, {}
    for m in models:
        t0 = time.time()
        if m == "xgb":
            est = M3.make_model("xgb", seed, "cpu", log)
            xp = {k: v for k, v in est.est.get_xgb_params().items() if v is not None}
            dtr = xgb.QuantileDMatrix(trI.frame_cat(), label=trI.y, enable_categorical=True)
            fitted[m] = xgb.train(xp, dtr, num_boost_round=R)
            info[m] = {"get_params": M3.clean_params(est.est_params())[0]}
            del dtr
        elif m == "lgbm":
            est = M3.make_model("lgbm", seed, "cpu", log)
            est.est.set_params(n_estimators=R)
            est.est.fit(trI.frame_cat(), trI.y)
            fitted[m] = est.est.booster_
            info[m] = {"get_params": M3.clean_params(est.est.get_params())[0]}
        elif m == "cat":
            fd = FeaturesData(num_feature_data=np.ascontiguousarray(X_tr), cat_feature_data=np.ascontiguousarray(cats_tr[0]),
                              num_feature_names=names, cat_feature_names=cat_names)
            ptr = Pool(fd, label=rs_tr.y)
            a0, b0 = offs[0]
            Xs = raw_X(ctx, rs_ev, L, a0, min(b0, a0 + 20000))
            pev = Pool(FeaturesData(num_feature_data=Xs, cat_feature_data=np.ascontiguousarray(cats_ev[0][a0:a0 + len(Xs)]),
                                    num_feature_names=names, cat_feature_names=cat_names), label=rs_ev.y[a0:a0 + len(Xs)])
            lr = cat_auto_lr(ptr, pev, seed)
            cm = CatBoostClassifier(iterations=R, learning_rate=lr, random_seed=seed, task_type="CPU", verbose=0, allow_writing_files=False)
            cm.fit(ptr)
            fitted[m] = cm
            info[m] = {"get_params": dict(cm.get_params()), "learning_rate_es_condition": lr}
            del ptr
        info[m]["fit_seconds"] = round(time.time() - t0, 1)
        log(f"    원값 {m}: {R}반복 학습 {info[m]['fit_seconds']:.0f}s")
    LVx = _leaf_values_xgb(fitted["xgb"]) if "xgb" in fitted else None
    LVl = _leaf_values_lgb(fitted["lgbm"]) if "lgbm" in fitted else None
    curves = {m: np.empty((len(offs), R)) for m in models}
    t1 = time.time()
    for hi, (a, b) in enumerate(offs):
        Xb = raw_X(ctx, rs_ev, L, a, b)
        yb = rs_ev.y[a:b]
        cb = (cats_ev[0][a:b], cats_ev[1][a:b])
        bI = M3.Inputs(Xb, cb[1], cb[0], yb, names, cat_names, levels)
        for m in models:
            if m == "xgb":
                dm = xgb.DMatrix(bI.frame_cat(), enable_categorical=True)
                leaf = fitted[m].predict(dm, pred_leaf=True).astype(np.int64)
                Mg = np.cumsum(LVx[np.arange(R)[None, :], leaf], axis=1)
                if hi == 0:                                   # 누적 잎 값 = 라이브러리 raw margin - 기준 상수(점검)
                    ref = fitted[m].predict(dm, output_margin=True, iteration_range=(0, R))
                    info[m]["staged_check_maxdiff"] = float(np.max(np.abs((ref - Mg[:, -1]) - np.median(ref - Mg[:, -1]))))
            elif m == "lgbm":
                df = bI.frame_cat()
                leaf = fitted[m].predict(df, pred_leaf=True).astype(np.int64)
                Mg = np.cumsum(LVl[np.arange(R)[None, :], leaf], axis=1)
                if hi == 0:
                    ref = fitted[m].predict(df, raw_score=True)
                    info[m]["staged_check_maxdiff"] = float(np.max(np.abs(ref - Mg[:, -1])))
            else:
                pool = Pool(FeaturesData(num_feature_data=Xb, cat_feature_data=np.ascontiguousarray(cb[0]),
                                         num_feature_names=names, cat_feature_names=cat_names))
                Mg = np.stack(list(fitted[m].staged_predict(pool, prediction_type="RawFormulaVal", eval_period=1)), axis=1)
            curves[m][hi] = staged_ap(yb, Mg)
            del Mg
        if (hi + 1) % 10 == 0:
            log(f"    원값 단계별 점수 h {hi + 1}/60 ({time.time() - t1:.0f}s)")
    out = {}
    for m in models:
        curve = np.nanmean(curves[m], axis=0)
        best, n_tr = simulate_es(curve, M3.PATIENCE)
        info[m].update({"best_rounds": int(best), "n_trained_es": int(n_tr), "es_history": [float(x) for x in curve[:n_tr]],
                        "es_best_value": float(curve[best - 1]), "staged_seconds": round(time.time() - t1, 1)})
    t2 = time.time()
    preds = {m: np.empty(rs_ev.n, dtype=np.float32) for m in models}
    for a, b in offs:
        Xb = raw_X(ctx, rs_ev, L, a, b)
        cb = (cats_ev[0][a:b], cats_ev[1][a:b])
        bI = M3.Inputs(Xb, cb[1], cb[0], rs_ev.y[a:b], names, cat_names, levels)
        for m in models:
            k = info[m]["best_rounds"]
            if m == "xgb":
                preds[m][a:b] = fitted[m].predict(xgb.DMatrix(bI.frame_cat(), enable_categorical=True), iteration_range=(0, k))
            elif m == "lgbm":
                preds[m][a:b] = fitted[m].predict(bI.frame_cat(), num_iteration=k)
            else:
                pool = Pool(FeaturesData(num_feature_data=Xb, cat_feature_data=np.ascontiguousarray(cb[0]),
                                         num_feature_names=names, cat_feature_names=cat_names))
                preds[m][a:b] = fitted[m].predict(pool, prediction_type="Probability", ntree_end=k)[:, 1]
    for m in models:
        info[m]["predict_seconds"] = round(time.time() - t2, 1)
        out[m] = (preds[m], info[m])
    return out
