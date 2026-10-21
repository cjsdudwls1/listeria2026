# -*- coding: utf-8 -*-
"""자체 점검.
(a) 무작위 행에서 창 값이 기상표의 T-1 .. T-L 과 같고 T 당일이 빠졌는지 (기상표 원본 DataFrame 에서 따로 찾아 비교)
(b) 단위x날짜 묶음 행 수·검출 수·분할별 수 (시료 원본에서 다른 방법으로 다시 셈)
(c) 요약 특성을 단순 반복문으로 다시 계산해 일치
(d) 모든 창(학습 h, 조기종료 h, 평가 h=1..60, L=100/10)이 기상표 안
(e) 모델 래퍼(합성 자료): (e1) LightGBM 학습 Dataset 을 학습 params 로 미리 construct 해도 예전 지연 construct 와
    예측이 비트 단위로 같은지(학습 20만 행 초과 = 구간 경계 표본 추출이 실제로 일어나는 크기),
    (e1)~(e3) 기록한 조기종료 val AP 가 최적 라운드 예측으로 sklearn 이 다시 계산한 AP 및 라운드별 기록 최대값과 같은지
    (CatBoost 는 사용자 지표 이름으로 고르는지 포함)
결과: logs/selftest.txt (UTF-8). 하나라도 실패하면 종료 코드 1.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
import data as D
import features as F

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
OUT = C.EXP_DIR / "logs" / "selftest.txt"
lines, fails = [], []


def say(msg):
    lines.append(msg)
    try:
        print(msg, flush=True)
    except Exception:
        pass


def check(name, ok, detail=""):
    say(f"[{'통과' if ok else '실패'}] {name} {detail}")
    if not ok:
        fails.append(name)


def model_checks():
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    import models as MD

    g = np.random.default_rng(7)

    def synth(n, nf=10):
        cats = np.stack([g.integers(0, k, n) for k in (3, 8, 5, 6)], 1).astype(np.float32)
        Xn = g.normal(size=(n, nf)).astype(np.float32)
        Xn[g.random((n, nf)) < 0.1] = np.nan
        z = -4 + 1.0 * np.nan_to_num(Xn[:, 0]) + 0.8 * (cats[:, 1] == 2) + 0.5 * np.nan_to_num(Xn[:, 1])
        y = (g.random(n) < 1 / (1 + np.exp(-z))).astype(np.int8)
        hcol = g.integers(1, 61, (n, 1)).astype(np.float32)
        return np.ascontiguousarray(np.concatenate([cats, hcol, Xn], 1)), y

    Xtr, ytr = synth(205_000)
    Xva, yva = synth(20_000)
    cols = list(C.CAT_COLS) + ["h"] + [f"f{i}" for i in range(Xtr.shape[1] - 5)]
    sink = lambda m: None

    # (e1) LightGBM
    lm = MD.LGBModel(sink)
    lm.fit(Xtr.copy(), ytr, lambda: Xva.copy(), yva, cols)
    p_new = lm.predict(Xva)
    cat_idx = list(range(len(C.CAT_COLS)))
    d1 = lgb.Dataset(Xtr.copy(), label=ytr, feature_name=cols, categorical_feature=cat_idx, free_raw_data=True)
    d2 = lgb.Dataset(Xva.copy(), label=yva, reference=d1, feature_name=cols, categorical_feature=cat_idx)
    apf = lambda preds, ds: ("ap", float(average_precision_score(yva, preds)), True)
    b_old = lgb.train(dict(lm.params), d1, num_boost_round=C.MAX_ROUNDS, valid_sets=[d2], valid_names=["val"],
                      feval=apf, callbacks=[lgb.early_stopping(C.PATIENCE, verbose=False)])
    p_old = b_old.predict(Xva, num_iteration=b_old.best_iteration).astype(np.float32)
    check("(e1) LightGBM 미리 construct = 예전 지연 construct (예측 비트 단위)",
          int(b_old.best_iteration) == lm.best_rounds and np.array_equal(p_old, p_new),
          f"학습 {len(ytr):,}행, 최적 라운드 {lm.best_rounds} / 예전 방식 {b_old.best_iteration}, "
          f"예측 최대 차이 {float(np.max(np.abs(p_old.astype(np.float64) - p_new))):.1e}")
    ap_re = float(average_precision_score(yva, lm.bst.predict(Xva, num_iteration=lm.best_rounds)))
    check("(e1) LightGBM 기록 AP = 다시 계산 = 라운드별 기록 최대",
          abs(ap_re - lm.best_val_ap) < 1e-12 and lm.best_val_ap == max(lm.val_ap_history)
          and len(lm.val_ap_history) == lm.n_rounds_trained,
          f"기록 {lm.best_val_ap:.6f} / 다시 계산 {ap_re:.6f}, 기록 길이 {len(lm.val_ap_history)}")

    # (e2) CatBoost (CPU, 사용자 지표)
    n2, nv2 = 5000, 2000
    to_str = lambda X: np.stack([X[:, j].astype(int).astype(str) for j in range(4)], 1).astype(object)
    Xc, yc = np.ascontiguousarray(Xtr[:n2, 4:]), ytr[:n2]
    Xcv, ycv = np.ascontiguousarray(Xva[:nv2, 4:]), yva[:nv2]
    cm = MD.CatModel(sink)
    cm.fit(Xc, yc, lambda: Xcv.copy(), ycv, ["h"] + cols[5:], cats_tr=to_str(Xtr[:n2]), cats_va=to_str(Xva[:nv2]))
    raw = cm.model.predict(cm._pool(Xcv, to_str(Xva[:nv2])), prediction_type="RawFormulaVal")
    ap_re = float(average_precision_score(ycv, raw))
    keys = list(cm.model.get_evals_result().get("validation", {}).keys())
    check("(e2) CatBoost 기록 AP = 최적 반복 예측으로 다시 계산 = 기록 최대 (사용자 지표 이름으로 고름)",
          abs(ap_re - cm.best_val_ap) < 1e-9 and cm.metric_key in keys and cm.best_val_ap == max(cm.val_ap_history)
          and cm.n_rounds_trained == len(cm.val_ap_history),
          f"기록 {cm.best_val_ap:.6f} / 다시 계산 {ap_re:.6f}, 최적 {cm.best_rounds}/{cm.n_rounds_trained}, "
          f"평가 기록 이름 {keys}, 고른 이름 {cm.metric_key}")

    # (e3) XGBoost (GPU)
    n3 = 50_000
    xm = MD.XGBModel(sink)
    xm.fit(np.ascontiguousarray(Xtr[:n3]), ytr[:n3], lambda: Xva.copy(), yva, cols)
    p = xm.bst.predict(xgb.DMatrix(Xva, feature_names=cols, feature_types=xm.ft, enable_categorical=True),
                       iteration_range=(0, xm.best_rounds))
    ap_re = float(average_precision_score(yva, p))
    check("(e3) XGBoost 기록 AP = 최적 라운드 예측으로 다시 계산 = 라운드별 기록 최대",
          abs(ap_re - xm.best_val_ap) < 1e-6 and xm.best_val_ap == max(xm.val_ap_history)
          and len(xm.val_ap_history) == xm.n_rounds_trained,
          f"기록 {xm.best_val_ap:.6f} / 다시 계산 {ap_re:.6f}, 최적 {xm.best_rounds}/{xm.n_rounds_trained}")


def main():
    rng = np.random.default_rng(2026)
    # ---------------- (b) 단위x날짜 묶음
    rows = D.assign_h(D.build_unit_rows(D.load_samples()))
    raw = pq.read_table(C.SAMPLE_PATH, columns=C.KEY_COLS + ["mnfctur_de", "judge_result"]).to_pandas()
    k = pd.DataFrame({c: raw[c].astype(object).fillna(C.NULL_CODE).astype(str) for c in C.KEY_COLS})
    k["d"] = pd.to_datetime(raw["mnfctur_de"]).dt.normalize()
    k["pos"] = raw["judge_result"].astype(str).eq("검출")
    check("(b) 시료 수 합계 = 원본 행 수", int(rows["n_samples"].sum()) == len(raw), f"{int(rows['n_samples'].sum()):,} / {len(raw):,}")
    check("(b) 검출 시료 수", int(rows["n_pos_samples"].sum()) == int(k["pos"].sum()), f"{int(rows['n_pos_samples'].sum()):,} / {int(k['pos'].sum()):,}")
    check("(b) 단위x날짜 중복 없음", not rows.duplicated(C.KEY_COLS + ["d"]).any())
    uk = k[C.KEY_COLS + ["d"]].drop_duplicates()
    pk = k.loc[k["pos"], C.KEY_COLS + ["d"]].drop_duplicates()
    def sp_of(d):
        return np.where(d <= pd.Timestamp(C.TRAIN_END), "train", np.where(d <= pd.Timestamp(C.VAL_END), "val", "test"))
    uk = uk.assign(split=sp_of(uk["d"]))
    pk = pk.assign(split=sp_of(pk["d"]))
    for sp in C.SPLITS:
        a = int((rows["split"] == sp).sum()); b = int((uk["split"] == sp).sum())
        ap_ = int(rows.loc[rows["split"] == sp, "y"].sum()); bp = int((pk["split"] == sp).sum())
        check(f"(b) {sp} 행 수", a == b, f"{a:,} / 다시 셈 {b:,}")
        check(f"(b) {sp} 검출 행 수", ap_ == bp, f"{ap_:,} / 다시 셈 {bp:,}")
    check("(b) 분할 기간 밖 행 없음", bool((pd.to_datetime(rows["d"]) <= pd.Timestamp(C.TEST_END)).all()))
    check("(b) y = 묶음 내 검출 1건 이상", bool((rows["y"] == (rows["n_pos_samples"] > 0)).all()))
    ds = D.data_summary(rows)
    say(ds.to_string(index=False))
    # h 배정
    tr = rows[rows["split"] == "train"]; va = rows[rows["split"] == "val"]
    check("(b) train h 범위 1..60", int(tr["h_fit"].min()) == 1 and int(tr["h_fit"].max()) == 60)
    check("(b) 조기종료 val h 범위 1..60", int(va["h_fit"].min()) == 1 and int(va["h_fit"].max()) == 60)
    rows2 = D.assign_h(D.build_unit_rows(D.load_samples()))
    check("(b) h 배정 재현", bool((rows2["h_fit"].to_numpy() == rows["h_fit"].to_numpy()).all()))
    lc, sizes = D.lstm_vocab_codes(rows)
    for j, c in enumerate(C.CAT_COLS):
        unseen = int(((lc[:, j] == 0) & (rows["split"] != "train").to_numpy()).sum())
        tr0 = int(((lc[:, j] == 0) & (rows["split"] == "train").to_numpy()).sum())
        check(f"(b) LSTM 범주 {c}: train 에 0번 없음", tr0 == 0, f"(train 어휘 {sizes[j]}개, val/test 의 학습에 없던 범주 행 {unseen:,})")

    # ---------------- 기상
    wobj = D.Weather()
    W = wobj.W
    wdf = pq.read_table(C.WEATHER_PATH).to_pandas()
    wdf["sgg_code"] = wdf["sgg_code"].astype(str)
    wdf["date"] = pd.to_datetime(wdf["date"])
    wdf = wdf.set_index(["sgg_code", "date"])

    def lookup(sgg, dates):
        ix = wdf.index.get_indexer(pd.MultiIndex.from_arrays([[sgg] * len(dates), pd.to_datetime(dates)]))
        assert (ix >= 0).all(), "기상표에 없는 (시군구, 날짜)"
        return wdf.iloc[ix][C.WEATHER_VARS].to_numpy(dtype=np.float32)

    s_all = wobj.s_idx(rows["sgg_code"].to_numpy())
    d_all = wobj.d_idx(rows["d"].to_numpy())

    # ---------------- (a) 창 값
    pick = rng.choice(len(rows), size=200, replace=False)
    nbad, nT_in = 0, 0
    for L in (100, 10):
        for i in pick:
            r = rows.iloc[i]
            h = int(r["h_fit"]) if r["split"] != "test" else int(rng.integers(1, 61))
            T = pd.Timestamp(r["d"]) - pd.Timedelta(days=h)
            t = np.array([d_all[i] - h])
            win = F.gather(W, s_all[i:i + 1], t, L)[0]
            dates = [T - pd.Timedelta(days=L - j) for j in range(L)]     # T-L .. T-1
            if T in dates:
                nT_in += 1
            exp = lookup(r["sgg_code"], dates)
            if not np.array_equal(win, exp, equal_nan=True):
                nbad += 1
            X = F.build_tree_matrix(W, s_all[i:i + 1], t, np.array([h]), L, "raw")
            names = F.feature_names("raw", L)
            for v_i, v in enumerate(C.WEATHER_VARS[:3]):
                for jlag in (1, L):
                    col = 1 + names.index(f"{v}_lag{jlag:03d}")
                    e = lookup(r["sgg_code"], [T - pd.Timedelta(days=jlag)])[0, v_i]
                    if not (X[0, col] == e or (np.isnan(X[0, col]) and np.isnan(e))):
                        nbad += 1
    check("(a) 창 값 = 기상표 T-L..T-1 (무작위 200행 x L=100,10, raw lag 칸 포함)", nbad == 0, f"불일치 {nbad}")
    check("(a) T 당일이 창에 없음", nT_in == 0)
    # T 당일 값이 lag 칸 어디에도 들어가지 않는지: lag 번호는 1..L 뿐
    check("(a) raw 특성 이름에 lag000(T 당일) 없음", not any(n.endswith("lag000") for n in F.feature_names("raw", 100)))

    # ---------------- (c) 요약 특성 반복문 재계산
    early = np.flatnonzero((rows["split"] == "train").to_numpy())[np.argsort(d_all[(rows["split"] == "train").to_numpy()])[:40]]
    pick_c = np.concatenate([rng.choice(len(rows), size=120, replace=False), early])
    maxdiff, nbad_c, n_partial, n_allnan = 0.0, 0, 0, 0
    for L in (100, 30, 10):
        for i in pick_c:
            r = rows.iloc[i]
            h = int(r["h_fit"]) if r["split"] != "test" else int(rng.integers(1, 61))
            T = pd.Timestamp(r["d"]) - pd.Timedelta(days=h)
            dates = [T - pd.Timedelta(days=L - j) for j in range(L)]
            vals = lookup(r["sgg_code"], dates).astype(np.float64)       # (L, V) 시간순
            X = F.build_tree_matrix(W, s_all[i:i + 1], np.array([d_all[i] - h]), np.array([h]), L, "summary")[0, 1:]
            exp = []
            for v in range(len(C.WEATHER_VARS)):
                y = vals[:, v]
                for kb in range(1, L // 10 + 1):
                    seg = y[L - 10 * kb: L - 10 * kb + 10]
                    exp.append(np.nanmean(seg) if np.isfinite(seg).any() else np.nan)
                ok = np.isfinite(y)
                if 0 < ok.sum() < L:
                    n_partial += 1
                if ok.sum() == 0:
                    n_allnan += 1
                exp += [np.nanmean(y) if ok.any() else np.nan, np.nanstd(y) if ok.any() else np.nan,
                        np.nanmin(y) if ok.any() else np.nan, np.nanmax(y) if ok.any() else np.nan]
                xs = np.arange(L)[ok]
                exp.append(np.polyfit(xs, y[ok], 1)[0] if ok.sum() >= 2 else np.nan)
            exp = np.array(exp)
            same_nan = np.array_equal(np.isnan(exp), np.isnan(X))
            fin = np.isfinite(exp)
            close = np.allclose(X[fin], exp[fin], rtol=1e-4, atol=1e-5)
            if fin.any():
                rel = np.abs(X[fin] - exp[fin]) / np.maximum(1.0, np.abs(exp[fin]))
                maxdiff = max(maxdiff, float(rel.max()))
            if not (same_nan and close):
                nbad_c += 1
    check("(c) 요약 특성 = 반복문 재계산 (160행 x L=100,30,10)", nbad_c == 0,
          f"불일치 행 {nbad_c}, 최대 상대오차 {maxdiff:.2e}, 일부 결측 창 {n_partial}건, 전부 결측 창 {n_allnan}건 포함")

    # ---------------- (c2) 합성 자료: 전부 결측·값 1개·일부 결측 창도 반복문과 일치하는지
    Wf = rng.normal(size=(3, 400, len(C.WEATHER_VARS))).astype(np.float32)
    Wf[0, 100:250, :] = np.nan                          # 시군구 0: 긴 결측 구간(전부 결측 창 생김)
    Wf[1, rng.random(400) < 0.5, :5] = np.nan            # 시군구 1: 무작위 결측
    Wf[2, 150:250, 3] = np.nan; Wf[2, 200, 3] = 1.5      # 시군구 2: 변수 3 값 1개만 있는 창
    nbad_s, n_all, n_one = 0, 0, 0
    for L in (100, 30, 10):
        for si in range(3):
            for tt in (L + 5, 160, 210, 251, 300, 399):
                if tt - L < 0:
                    continue
                X = F.build_tree_matrix(Wf, np.array([si]), np.array([tt]), np.array([1]), L, "summary")[0, 1:]
                vals = Wf[si, tt - L:tt, :].astype(np.float64)
                exp = []
                for v in range(len(C.WEATHER_VARS)):
                    y = vals[:, v]
                    for kb in range(1, L // 10 + 1):
                        seg = y[L - 10 * kb: L - 10 * kb + 10]
                        exp.append(np.nanmean(seg) if np.isfinite(seg).any() else np.nan)
                    ok = np.isfinite(y)
                    n_all += int(ok.sum() == 0); n_one += int(ok.sum() == 1)
                    exp += [np.nanmean(y) if ok.any() else np.nan, np.nanstd(y) if ok.any() else np.nan,
                            np.nanmin(y) if ok.any() else np.nan, np.nanmax(y) if ok.any() else np.nan]
                    exp.append(np.polyfit(np.arange(L)[ok], y[ok], 1)[0] if ok.sum() >= 2 else np.nan)
                exp = np.array(exp)
                fin = np.isfinite(exp)
                if not (np.array_equal(np.isnan(exp), np.isnan(X)) and np.allclose(X[fin], exp[fin], rtol=1e-4, atol=1e-5)):
                    nbad_s += 1
    check("(c2) 합성 자료 요약 특성 = 반복문 (전부 결측·값 1개 창 포함)", nbad_s == 0 and n_all > 0 and n_one > 0,
          f"불일치 {nbad_s}, 전부 결측 변수-창 {n_all}건, 값 1개 변수-창 {n_one}건")

    # ---------------- (d) 모든 창이 기상표 안
    for L in (100, 10):
        lo, hi = [], []
        for sp in C.SPLITS:
            m = (rows["split"] == sp).to_numpy()
            d = d_all[m]
            hf = rows.loc[m, "h_fit"].to_numpy().astype(np.int64)
            ts = {"train": [d - hf], "val": [d - hf, d - 60, d - 1], "test": [d - 60, d - 1]}[sp]
            for t in ts:
                a, b = wobj.check_windows(t, L, f"{sp} L={L}")
                lo.append(a); hi.append(b)
        check(f"(d) 모든 창이 기상표 안 L={L}", min(lo) >= 0 and max(hi) <= wobj.D - 1,
              f"창 날짜 범위 {wobj.date0 + np.timedelta64(min(lo), 'D')} ~ {wobj.date0 + np.timedelta64(max(hi), 'D')}"
              f" (기상표 {wobj.date0} ~ {wobj.date_last})")

    # ---------------- (e) 모델 래퍼(합성 자료)
    model_checks()
    say(f"\n결과: 실패 {len(fails)}개 {fails}")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)