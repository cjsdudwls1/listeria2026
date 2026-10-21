# -*- coding: utf-8 -*-
"""겹 자료 만들기와 교차검증 한 번 돌리기(바닐라·k 고르기·튜닝 시도·묶음 빼기·최종이 같이 씀).

FoldData(범위 하나) = 학습 행·점수 낼 행(겹 k: 맞혀 보는 행, 최종: 시험 행)의 숫자 칸(부모 입력 묶음 전부)·범주 칸·묶음 표시.
  view(묶음 목록) -> 그 묶음 칸만 고른 (학습 Inputs, 점수 낼 Inputs). h·범주 4칸은 항상 들어감. 식품 묶음을 빼면 aw_label 범주도 빠짐.
  이상치 점수(ae4)는 부모 묶음 기준으로 한 번 계산해 숫자 칸 끝에 붙임(묶음 빼기 동안 고정 - 2차 단계 선택 '뒤로'와 같음).
run_cv(...) -> 겹별 점수·평균·최적 반복·학습 시간·묶음 중요도(겹마다 칸 중요도 합 1 로 맞춘 뒤 겹 평균, 묶음 = 칸 합)·(선택) 예측.
"""
import time

import numpy as np

import cfg3 as G
import cv as CV3
import models3 as M3


class FoldData:
    def __init__(self, scope, k, rs_tr, rs_ev, X_tr, X_ev, names, cat_names, levels, cs_tr, cd_tr, cs_ev, cd_ev, bundles, tfidf=None):
        self.scope, self.k = scope, k
        self.rs_tr, self.rs_ev = rs_tr, rs_ev
        self.X_tr, self.X_ev = X_tr, X_ev
        self.names, self.cat_names, self.levels = list(names), list(cat_names), levels
        self.cs_tr, self.cd_tr, self.cs_ev, self.cd_ev = cs_tr, cd_tr, cs_ev, cd_ev
        self.bundles = list(bundles)
        self.num_b, self.cat_b = CV3.column_bundles(self.names, self.cat_names, self.bundles)
        self.offs = rs_ev.h_offsets()
        self.tfidf = tfidf
        self.ae_info = None

    def view(self, bundles):
        keep = set(bundles) | {"base"}
        ni = [i for i, b in enumerate(self.num_b) if b in keep]
        ci = [i for i, b in enumerate(self.cat_b) if b in keep]
        nm = [self.names[i] for i in ni]
        cn = [self.cat_names[i] for i in ci]
        lv = {c: self.levels[c] for c in cn}
        sel = (lambda X: X) if len(ni) == len(self.names) else (lambda X: np.ascontiguousarray(X[:, ni]))
        tr = M3.Inputs(sel(self.X_tr), self.cd_tr[:, ci], self.cs_tr[:, ci], self.rs_tr.y, nm, cn, lv)
        ev = M3.Inputs(sel(self.X_ev), self.cd_ev[:, ci], self.cs_ev[:, ci], self.rs_ev.y, nm, cn, lv)
        return tr, ev

    def drop_bundle_columns(self, drop):
        """메모리 줄이기: 숫자 칸 중 drop 묶음 칸을 버림(BiLSTM 튜닝 - 기상 요약은 시퀀스로 들어가므로 표 칸 불필요)."""
        keep = [i for i, b in enumerate(self.num_b) if b not in drop]
        if len(keep) == len(self.names):
            return
        self.X_tr = np.ascontiguousarray(self.X_tr[:, keep])
        self.X_ev = np.ascontiguousarray(self.X_ev[:, keep])
        self.names = [self.names[i] for i in keep]
        self.num_b = [self.num_b[i] for i in keep]

    def bundle_of_all_names(self, x):
        mp = dict(zip(self.names, self.num_b))
        mp.update(dict(zip(self.cat_names, self.cat_b)))
        return [mp[n] for n in x.all_names()]


def build(cvo, scope, bundles, k_smooth, log, learn_cache=None):
    """scope = 겹 번호(int) 또는 'final'. 돌려줌 FoldData. learn_cache = (RowSet, X, fb 지문) 재사용용 dict."""
    bundles = CV3.resolve_featset(bundles)
    tag = "final" if scope == "final" else CV3.fold_def(scope)["train_end"]
    tft = tag if "n0" in bundles else None
    smoke = "smoke_" if cvo.smoke.get("enabled") else ""
    # 학습 행: 학습 자료 전체(시드별, 묶음별 캐시)에서 고름. n0 는 겹마다 TF-IDF 파일이 달라 그 겹 판으로.
    key = (tuple(bundles), int(k_smooth), tft)
    if learn_cache is not None and key in learn_cache:
        rs_all, X_all, fb = learn_cache[key]
    else:
        rs_all = cvo.learn_rows()
        X_all, fb = cvo.assemble_X(rs_all, bundles, k_smooth, f"{smoke}learn_s{cvo.seed}", tft)
        if learn_cache is not None and tft is None:
            learn_cache[key] = (rs_all, X_all, fb)
    tf = getattr(cvo, "tfidf_used", None) if "n0" in bundles else None
    if scope == "final":
        rs_tr, X_tr = rs_all, X_all
        rs_ev = cvo.test_rowset(f"final 시험 입력({','.join(bundles)})")
        X_ev, fb = cvo.assemble_X(rs_ev, bundles, k_smooth, f"{smoke}test", tft)
    else:
        rs_tr = cvo.fold_train(scope)
        m = rs_all.d <= np.datetime64(CV3.fold_def(scope)["train_end"], "D")
        assert rs_tr.n == int(m.sum()) and np.array_equal(rs_tr.pos, rs_all.pos[m]) and np.array_equal(rs_tr.h, rs_all.h[m])
        X_tr = X_all[m]
        rs_ev = cvo.eval_rows(scope)
        X_ev, fb = cvo.assemble_X(rs_ev, bundles, k_smooth, f"{smoke}eval_f{scope}", tft)
    levels = cvo.cat_levels(fb.cat_names, fb)
    cs_tr, cd_tr = cvo.cats(rs_tr, fb, levels)
    cs_ev, cd_ev = cvo.cats(rs_ev, fb, levels)
    names = list(fb.names)
    fd_ae = None
    if "ae4" in bundles:
        import ae3
        S_tr, S_ev, infos = [], [], {}
        sc = f"f{scope}" if scope != "final" else "final"
        nb0, _ = CV3.column_bundles(names, fb.cat_names, [b for b in bundles if b != "ae4"])
        ai = [i for i, b in enumerate(nb0) if b not in ("n2", "n0")]       # 이상치 점수 표 입력 = 그날 제품명 칸 뺀 숫자 칸(2차 ALL_B 규칙)
        Xa_tr = X_tr if len(ai) == len(names) else np.ascontiguousarray(X_tr[:, ai])
        Xa_ev = X_ev if len(ai) == len(names) else np.ascontiguousarray(X_ev[:, ai])
        for kind in ae3.KINDS:
            a, b, inf = ae3.scores(kind, cvo.seed, f"{smoke}{sc}", rs_tr, Xa_tr, cd_tr, rs_ev, Xa_ev, cd_ev, cvo.ctx.weather,
                                   cvo.cache_dir, log)
            S_tr.append(a)
            S_ev.append(b)
            infos[kind] = inf
        del Xa_tr, Xa_ev
        X_tr = np.ascontiguousarray(np.hstack([X_tr, np.stack(S_tr, 1)]), dtype=np.float32)
        X_ev = np.ascontiguousarray(np.hstack([X_ev, np.stack(S_ev, 1)]), dtype=np.float32)
        names += CV3.AE_NAMES
        fd_ae = infos
    fd = FoldData(scope, scope if scope != "final" else None, rs_tr, rs_ev, X_tr, X_ev, names, fb.cat_names, levels,
                  cs_tr, cd_tr, cs_ev, cd_ev, bundles, tfidf=tf)
    fd.ae_info = fd_ae
    fd.window = cvo.check_windows([rs_tr, rs_ev])
    return fd


def _bilstm_fold(k, fd, bundles, params, seed, log):
    """BiLSTM 한 겹: 기상 = 시퀀스(창 L일), 표 입력 = 고른 묶음 중 기상 요약(wx)·h 를 뺀 숫자 칸(dl3 머리말)."""
    import dl3
    keep = set(bundles) - {"wx"}
    ni = [i for i, b in enumerate(fd.num_b) if b in keep]
    tab = (np.ascontiguousarray(fd.X_tr[:, ni]), np.ascontiguousarray(fd.X_ev[:, ni])) if ni else None
    te = CV3.fold_def(k)["train_end"] if fd.scope != "final" else G.BOUNDS["learn_end"]
    p, inf = dl3.run_fold(fd.rs_tr.ctx, fd.rs_tr, fd.rs_ev, fd.cd_tr[:, :4], fd.cd_ev[:, :4], te, seed, log, params, tab)
    if fd.scope == "final":                                # 최종(시험 행): 여기서는 채점하지 않음(기준값은 교차검증 예측에서) - cmd_final 이 score_test
        inf.update({"device": "cuda", "get_params": inf["config"]})
        return {}, None, p, inf
    s, df = CV3.score_fold(fd.rs_ev.y, p, fd.offs)
    s.update({"fold": k, "n_train_rows": int(fd.rs_tr.n), "n_train_pos": int(fd.rs_tr.y.sum()), "best_rounds": inf["best_epoch"],
              "n_trained": inf["n_epochs"], "fit_seconds": inf["fit_seconds"], "predict_seconds": 0.0, "n_num": int(inf["n_tab"]), "n_cat": 4})
    inf.update({"device": "cuda", "get_params": inf["config"], "notes": {"tab_columns": [fd.names[i] for i in ni][:20] + (["..."] if len(ni) > 20 else [])}})
    return s, df, p, inf


MEAN_KEYS = ["pr_auc", "f2", "p_at_r", "r_at_thr", "precision", "recall", "f1", "accuracy", "roc_auc", "inspect_reduction",
             "tp", "fp", "fn", "tn", "n", "top1", "top5", "top10", "top20"]


def run_cv(model, params, fds, bundles, device, seed, log, keep_preds=False, judge_folds=None, quiet=False):
    """fds = {겹: FoldData}. 돌려줌 dict(mean, by_fold, best_rounds, fit_seconds, importance_bundle, preds?)."""
    by_fold, preds, imps, infos = {}, {}, [], {}
    for k, fd in fds.items():
        if model == "bilstm":
            s, df, p, inf = _bilstm_fold(k, fd, bundles, params, seed, log)
            by_fold[k], infos[k] = s, inf
            imps.append({"_error": "BiLSTM 중요도 없음"})
            if keep_preds:
                preds[k] = (p, df)
            if not quiet:
                log(f"    겹 {k} BiLSTM: PR-AUC {s['pr_auc']:.5f}, 최고 에폭 {s['best_rounds']}, {s['fit_seconds']:.0f}s (표 입력 {inf['n_tab']}칸)")
            continue
        tr, ev = fd.view(bundles)
        m = M3.make_model(model, seed, device, log, params)
        use_es = model in M3.TREE_BOOST
        m.fit(tr, ev if use_es else None, fd.offs if use_es else None)
        t1 = time.time()
        p = m.predict(ev)
        tp = time.time() - t1
        s, df = CV3.score_fold(ev.y, p, fd.offs)
        try:
            imp = np.asarray(m.importance(tr), dtype=np.float64)
            imp = imp / imp.sum() if imp.sum() > 0 else imp
            bn = fd.bundle_of_all_names(tr)
            ib = {}
            for b, v in zip(bn, imp):
                ib[b] = ib.get(b, 0.0) + float(v)
            imps.append(ib)
        except Exception as e:  # 중요도가 없으면 기록만
            imps.append({"_error": str(e)})
        s.update({"fold": k, "n_train_rows": int(tr.n), "n_train_pos": int(tr.y.sum()), "best_rounds": m.best_rounds,
                  "n_trained": m.n_trained, "fit_seconds": round(m.fit_seconds, 2), "predict_seconds": round(tp, 2),
                  "n_num": len(tr.names), "n_cat": len(tr.cat_names)})
        by_fold[k] = s
        infos[k] = m.info()
        if keep_preds:
            preds[k] = (p, df)
        if not quiet:
            log(f"    겹 {k}: PR-AUC {s['pr_auc']:.5f}, F2(최대) {s['f2']:.5f}, P@R≥{CV3.RECALL_MIN:.2f} {s['p_at_r']:.5f}, 상위10% {s['top10']:.4f}, 최적 반복 {m.best_rounds}, "
                f"학습 {m.fit_seconds:.1f}s, 예측 {tp:.1f}s ({tr.n:,}행 x {len(tr.names)}+{len(tr.cat_names)}칸)")
        del tr, ev, m
    ks = list(fds.keys()) if judge_folds is None else [k for k in judge_folds if k in fds]
    keys = MEAN_KEYS
    mean = {kk: float(np.mean([by_fold[k][kk] for k in ks])) for kk in keys}
    ib = {}
    good = [d for d in imps if "_error" not in d]
    for d in good:
        for b, v in d.items():
            ib[b] = ib.get(b, 0.0) + v / len(good)
    out = {"model": model, "bundles": list(bundles), "mean": mean, "mean_folds": ks, "by_fold": by_fold,
           "best_rounds": [by_fold[k]["best_rounds"] for k in fds], "fit_seconds": round(sum(by_fold[k]["fit_seconds"] for k in fds), 2),
           "importance_bundle": ib, "model_info": infos}
    if keep_preds:
        out["preds"] = preds
    return out
