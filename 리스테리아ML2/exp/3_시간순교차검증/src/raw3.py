# -*- coding: utf-8 -*-
"""'일별 원값' 입력(1차 'raw': 창 L일 x 19변수 = 19L칸, lag j = T−j) 트리 3종 - 한 겹에서 같이 돌린다(바닐라, k 고르기).

맞혀 보는 행 입력이 겹마다 약 165만 (행,h) x 최대 1,901칸(L=100 이면 약 13GB, float32)이라 캐시도, 메모리에 통째로 두기도 하지 않고
h 조각(약 2.8만 행)마다 그때그때 만든다(조각 예측).
조기 종료 = 라이브러리 조기 종료(지표 = 맞혀 보는 행 h 별 AP 의 h 1..60 평균, 인내 100, 최고 = 처음 나온 최고)를 그대로 재현:
  - XGBoost·LightGBM: 학습 콜백이 check_every(100) 반복마다 새로 생긴 나무들만 맞혀 보는 행 h 조각마다 잎 번호(pred_leaf) -> 잎 값으로
    반복별 누적 점수를 이어 붙여 반복별 지표 곡선을 만들고, 조기 종료 규칙(simulate_es)으로 멈출 때가 지났으면 학습을 멈춤.
    콜백은 나무 만들기에 영향 없음(평가 행은 학습에 안 쓰임) -> 멈춘 모델의 앞부분 = 끝까지 학습한 모델의 앞부분 = 실제 조기 종료 모델.
    (이전 판은 2,000 반복을 다 학습한 뒤 곡선을 만들었음 - 같은 결과, 시간만 줄임)
  - CatBoost: iterations 설정값이 나무에 영향을 주므로(selftest3 catboost (d)) 조기 종료 때와 같은 iterations=2000 으로 학습하고,
    학습률은 조기 종료 조건(eval set + use_best_model)에서 CatBoost 가 자동으로 정하는 값을 미리 구해 고정(selftest3 catboost (d)·(e)).
    곡선은 100그루씩 staged_predict(ntree_start..) 로 이어 붙이고 멈출 때가 지나면 그만 계산.
  - 끝으로 맞혀 보는 행을 h 조각마다 다시 만들어 최적 반복까지의 예측(확률, 라이브러리 predict) -> 채점·저장.
점수 곡선은 누적 점수(raw margin) 순위로 계산(확률과 순위 같음, 상수 차이는 순위에 영향 없음). 조기 종료 재현 기록 = es_history.
extra: 원값 뒤에 붙일 칸(k 고르기의 과거 이력 핵심 칸 등) = {"names": [...], "ev": fn(a, b) -> (b−a, e) float32}; X_tr 에는 이미 붙어 있어야 함.
selftest3 rawes = 이 방식과 실제 조기 종료(models3, 맞혀 보는 행 eval set)의 최적 반복·예측 대조.
wvars: 남길 기상 변수(기상 거르기, None = 19개 전부) - 원값 칸 중 그 변수의 L칸만. perm_wvars: 최적 반복 모델로 기상 변수 섞기 중요도
(h 조각마다 그 변수 L칸을 행끼리 섞어 예측 -> 구간 가중 PR-AUC, 씨앗 = engine3 규칙).
"""
import time

import numpy as np

import cfg3 as G  # noqa: F401
import cv as CV3
import features as F1
import models3 as M3

V = 19
CHECK_EVERY = 100


def raw_cols(wvars=None, L=None):
    """[h, 원값 19L] 중 남길 칸 번호(h 는 늘 포함). 원값 칸은 변수 순서대로 변수마다 lag 1..L (1차 raw_from_window)."""
    L = CV3.L if L is None else L
    V = list(CV3.C1.WEATHER_VARS)
    if wvars is None or set(wvars) == set(V):
        return list(range(1 + len(V) * L))
    keep = set(wvars)
    return [0] + [1 + vi * L + j for vi, v in enumerate(V) if v in keep for j in range(L)]


def raw_names(L, wvars=None):
    nm = ["h"] + F1.feature_names("raw", L)
    return [nm[i] for i in raw_cols(wvars, L)]


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


def _leaf_values_lgb(bst, start=0, num=None):
    d = bst.dump_model(start_iteration=start, num_iteration=num) if num is not None else bst.dump_model()
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


def es_stopped(curve, patience):
    """곡선이 조기 종료 지점(최고 뒤 patience 반복 동안 갱신 없음)까지 왔는지."""
    best, n_tr = simulate_es(curve, patience)
    return n_tr < len(curve) or (len(curve) - best) >= patience


class StagedES:
    """맞혀 보는 행 h 조각별 누적 점수를 반복 묶음 단위로 이어 붙여 반복별 지표 곡선(h 별 AP 의 h 1..60 평균)을 만든다."""

    def __init__(self, y, offs, patience):
        self.y, self.offs, self.patience = y, offs, patience
        self.margin = np.zeros(len(y), dtype=np.float64)
        self.curve = []
        self.done = 0
        self.seconds = 0.0

    def add(self, block_fn, n_new, mode="contrib"):
        """block_fn(a, b) -> (b−a, n_new): mode='contrib' 이면 반복별 나무 기여, 'partial' 이면 이번 묶음 첫 나무부터의 누적 점수.
        돌려줌: 조기 종료 지점을 지났는지."""
        t0 = time.time()
        C = np.empty((len(self.offs), n_new))
        for hi, (a, b) in enumerate(self.offs):
            Tm = np.asarray(block_fn(a, b), dtype=np.float64).reshape(b - a, n_new)
            Mg = self.margin[a:b, None] + (np.cumsum(Tm, axis=1) if mode == "contrib" else Tm)
            C[hi] = staged_ap(self.y[a:b], Mg)
            self.margin[a:b] = Mg[:, -1]
            del Tm, Mg
        self.curve += [float(x) for x in np.nanmean(C, axis=0)]
        self.done += n_new
        self.seconds += time.time() - t0
        return es_stopped(self.curve, self.patience)

    def result(self):
        best, n_tr = simulate_es(self.curve, self.patience)
        return {"best_rounds": int(best), "n_trained_es": int(n_tr), "es_history": self.curve[:n_tr],
                "es_best_value": float(self.curve[best - 1]), "staged_seconds": round(self.seconds, 1), "curve_rounds": len(self.curve)}


def run_fold(ctx, rs_tr, rs_ev, X_tr, cats_tr, cats_ev, levels, models, seed, log, extra=None, check_every=CHECK_EVERY,
             wvars=None, perm_wvars=False, perm_fold=None):
    """한 겹: 세 모델 학습(조기 종료 재현) -> 최적 반복 예측. X_tr = [h, 원값(wvars 칸만), extra 칸]. 돌려줌 {모델: (p, info)}."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier, FeaturesData, Pool
    L = CV3.L
    cols = raw_cols(wvars, L)
    names = raw_names(L, wvars) + (list(extra["names"]) if extra else [])
    assert X_tr.shape[1] == len(names), (X_tr.shape, len(names))
    cat_names = list(CV3.BASE_CATS)
    trI = M3.Inputs(X_tr, cats_tr[1], cats_tr[0], rs_tr.y, names, cat_names, levels)
    offs = rs_ev.h_offsets()
    R, PAT = M3.MAX_ROUNDS, M3.PATIENCE

    def block_X(a, b):
        X = raw_X(ctx, rs_ev, L, a, b)
        if len(cols) != X.shape[1]:
            X = np.ascontiguousarray(X[:, cols])
        if extra:
            X = np.ascontiguousarray(np.hstack([X, np.asarray(extra["ev"](a, b), dtype=np.float32)]), dtype=np.float32)
        return X

    def block_I(a, b):
        return M3.Inputs(block_X(a, b), cats_ev[1][a:b], cats_ev[0][a:b], rs_ev.y[a:b], names, cat_names, levels)

    def cat_pool(X, cs):
        return Pool(FeaturesData(num_feature_data=np.ascontiguousarray(X), cat_feature_data=np.ascontiguousarray(cs),
                                 num_feature_names=names, cat_feature_names=cat_names))

    fitted, info = {}, {}
    for m in models:
        t0 = time.time()
        es = StagedES(rs_ev.y, offs, PAT)
        if m == "xgb":
            est = M3.make_model("xgb", seed, "cpu", log)
            xp = {k: v for k, v in est.est.get_xgb_params().items() if v is not None}
            dtr = xgb.QuantileDMatrix(trI.frame_cat(), label=trI.y, enable_categorical=True)
            chk = {}

            class XgbCB(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    n = epoch + 1
                    if n % check_every and n < R:
                        return False
                    k0, new = es.done, n - es.done
                    sub = model[k0:n]
                    LV = _leaf_values_xgb(sub)

                    def contrib(a, b):
                        dm = xgb.DMatrix(block_I(a, b).frame_cat(), enable_categorical=True)
                        leaf = sub.predict(dm, pred_leaf=True).astype(np.int64).reshape(b - a, new)
                        if not chk and k0 == 0:                     # 누적 잎 값 = 라이브러리 raw margin − 기준 상수(점검, 첫 조각)
                            ref = model.predict(dm, output_margin=True, iteration_range=(0, n))
                            mg = np.cumsum(LV[np.arange(new)[None, :], leaf], axis=1)[:, -1]
                            chk["staged_check_maxdiff"] = float(np.max(np.abs((ref - mg) - np.median(ref - mg))))
                        return LV[np.arange(new)[None, :], leaf]
                    stop = es.add(contrib, new)
                    log(f"    원값 xgb: {n}반복, 곡선 최고 {max(es.curve):.5f} @ {int(np.argmax(es.curve)) + 1}, {time.time() - t0:.0f}s")
                    return stop

            fitted[m] = xgb.train(xp, dtr, num_boost_round=R, callbacks=[XgbCB()])
            info[m] = {"get_params": M3.clean_params(est.est_params())[0], **chk}
            del dtr
        elif m == "lgbm":
            est = M3.make_model("lgbm", seed, "cpu", log)
            est.est.set_params(n_estimators=R)
            chk = {}

            def lgb_cb(env):
                n = env.iteration + 1
                if n % check_every and n < env.end_iteration:
                    return
                bst = env.model
                k0, new = es.done, n - es.done
                LV = _leaf_values_lgb(bst, k0, new)

                def contrib(a, b):
                    df = block_I(a, b).frame_cat()
                    leaf = bst.predict(df, pred_leaf=True, start_iteration=k0, num_iteration=new).astype(np.int64).reshape(b - a, new)
                    if not chk and k0 == 0:
                        ref = bst.predict(df, raw_score=True, start_iteration=0, num_iteration=n)
                        mg = np.cumsum(LV[np.arange(new)[None, :], leaf], axis=1)[:, -1]
                        chk["staged_check_maxdiff"] = float(np.max(np.abs(ref - mg)))
                    return LV[np.arange(new)[None, :], leaf]
                stop = es.add(contrib, new)
                log(f"    원값 lgbm: {n}반복, 곡선 최고 {max(es.curve):.5f} @ {int(np.argmax(es.curve)) + 1}, {time.time() - t0:.0f}s")
                if stop:
                    best, _ = simulate_es(es.curve, PAT)
                    raise lgb.callback.EarlyStopException(best - 1, [("valid_0", "ap60", es.curve[best - 1], True)])

            lgb_cb.order = 30
            est.est.fit(trI.frame_cat(), trI.y, callbacks=[lgb_cb])
            fitted[m] = est.est.booster_
            info[m] = {"get_params": M3.clean_params(est.est.get_params())[0], **chk}
        elif m == "cat":
            ptr = Pool(FeaturesData(num_feature_data=np.ascontiguousarray(X_tr), cat_feature_data=np.ascontiguousarray(cats_tr[0]),
                                    num_feature_names=names, cat_feature_names=cat_names), label=rs_tr.y)
            a0, b0 = offs[0]
            b1 = min(b0, a0 + 20000)
            pev = Pool(FeaturesData(num_feature_data=block_X(a0, b1), cat_feature_data=np.ascontiguousarray(cats_ev[0][a0:b1]),
                                    num_feature_names=names, cat_feature_names=cat_names), label=rs_ev.y[a0:b1])
            lr = cat_auto_lr(ptr, pev, seed)
            cm = CatBoostClassifier(iterations=R, learning_rate=lr, random_seed=seed, task_type="CPU", verbose=0, allow_writing_files=False)
            cm.fit(ptr)
            fitted[m] = cm
            info[m] = {"get_params": dict(cm.get_params()), "learning_rate_es_condition": lr, "fit_full_seconds": round(time.time() - t0, 1)}
            del ptr, pev
            ntree = int(cm.tree_count_)
            for k0 in range(0, ntree, check_every):
                new = min(check_every, ntree - k0)

                def partial(a, b):
                    pool = cat_pool(block_X(a, b), cats_ev[0][a:b])
                    return np.stack(list(cm.staged_predict(pool, prediction_type="RawFormulaVal", ntree_start=k0, ntree_end=k0 + new,
                                                           eval_period=1)), axis=1)
                stop = es.add(partial, new, mode="partial")
                log(f"    원값 cat: {k0 + new}그루 곡선, 최고 {max(es.curve):.5f} @ {int(np.argmax(es.curve)) + 1}, {time.time() - t0:.0f}s")
                if stop:
                    break
        info[m].update(es.result())
        info[m]["fit_seconds"] = round(time.time() - t0, 1)
        info[m]["es_rule"] = (f"조기 종료 재현: {check_every}반복마다 맞혀 보는 행 h 조각별 반복별 AP -> h 1..60 평균 곡선, 인내 {PAT}(최고 = 처음 나온 최고). "
                              "XGBoost·LightGBM 은 멈출 때가 지나면 학습 중단(콜백), CatBoost 는 iterations=2000 학습 뒤 곡선만 중단")
        log(f"    원값 {m}: 최적 반복 {info[m]['best_rounds']} (곡선 {info[m]['curve_rounds']}반복), {info[m]['fit_seconds']:.0f}s")
    t2 = time.time()

    def pred(m, bI):
        k = info[m]["best_rounds"]
        if m == "xgb":
            return fitted[m].predict(xgb.DMatrix(bI.frame_cat(), enable_categorical=True), iteration_range=(0, k))
        if m == "lgbm":
            return fitted[m].predict(bI.frame_cat(), num_iteration=k)
        return fitted[m].predict(cat_pool(bI.X, bI.cat_str), prediction_type="Probability", ntree_end=k)[:, 1]

    preds = {m: np.empty(rs_ev.n, dtype=np.float32) for m in models}
    for a, b in offs:
        bI = block_I(a, b)
        for m in models:
            preds[m][a:b] = pred(m, bI)
    for m in models:
        info[m]["predict_seconds"] = round(time.time() - t2, 1)
    if perm_wvars:                                         # 기상 변수 섞기 중요도(최적 반복 모델, 다시 학습 없음)
        import engine3 as EN
        t3 = time.time()
        pseed, pscope = EN.perm_cfg()
        if pscope != "h_block":
            raise RuntimeError("원값 변수 섞기는 h 조각 안 섞기(perm_scope = h_block)만")
        wl = EN.resolve_wvars(wvars)
        groups = {v: [i for i, n in enumerate(names) if EN.wvar_of(n) == v] for v in wl}
        srcs = {v: EN.perm_source(rs_ev.n, offs, np.random.default_rng([int(pseed), int(perm_fold), EN.wvar_seed_index(v)]), pscope)
                for v in wl}
        pp = {m: {v: np.empty(rs_ev.n, dtype=np.float32) for v in wl} for m in models}
        for a, b in offs:
            bI = block_I(a, b)
            for v in wl:
                gc_ = groups[v]
                orig = bI.X[:, gc_].copy()
                bI.X[:, gc_] = orig[srcs[v][a:b] - a]
                for m in models:
                    pp[m][v][a:b] = pred(m, bI)
                bI.X[:, gc_] = orig
        for m in models:
            info[m]["perm_wvar_score"] = {v: CV3.wpr(rs_ev.y, pp[m][v], offs) for v in wl}
            info[m]["perm_wvar_seconds"] = round(time.time() - t3, 1)
    out = {}
    for m in models:
        out[m] = (preds[m], info[m])
    return out
