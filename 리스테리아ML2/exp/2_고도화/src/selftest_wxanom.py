# -*- coding: utf-8 -*-
"""M 'wxanom'(이동평균 차감 기상, M1·A3_M1) 자체 점검 (전체 자료). 결과: logs/selftest_wxanom.txt (UTF-8). 하나라도 실패하면 종료 코드 1.
results/·tables/·cache/ 에는 쓰지 않는다(E00 재현 결과는 scratch/wxanom/selftest_e00/, 기상 요약 캐시는 메타가 같을 때만 읽기).

(0) 설정: M1·A3_M1 단계 'm'·시드 42·43·44, STAGES 에서 'm' 이 tables 바로 앞, 조합 ID 정규식과 겹치지 않음, A3_M1 = A3 입력 + wxanom,
    칸 57개(이름 중복 없음), run_exp2.ps1 ValidateSet 에 'm'. 모든 실험 창의 가장 이른 날 − 30일이 기상표 안
(a) 무작위 (행, h) 300쌍(분할마다 100): 기상 parquet 원본(시군구_일별_기상.parquet)을 따로 읽어 반복문으로
    a_t = x_t − (앞 30일 중 결측 아닌 값 평균) 을 창 T−50..T−1 날마다 다시 계산 -> 코드 격자 a_t 와 대조, 요약 57칸(평균·최근 10일 평균·최대) 대조.
    한 쌍씩(h 정수) = 한꺼번에(h 배열, 학습 경로). 대조군: 'T 포함 창(T−49..T)' 또는 '앞 30일에 당일 포함' 으로 만들면 달라지는 쌍이 있음.
    M1·A3_M1 특성 행렬 = E00·A3 특성 행렬(비트 단위) + 57칸
(b) 앞날 자료 영향 없음: 그 시군구의 T 이후(T 당일 포함) 기상 19칸을 무작위 값(결측 섞음)으로 바꿔 격자를 다시 계산해도
    T 전 날의 a_t 와 57칸이 비트 단위로 같음. 대조군: T−1 하루만 바꾸면 57칸이 달라짐
(c) E00 시드 42 를 끝까지 실행해 1차 catboost__summary__L050·2차 results/E00__s42 예측과 비트 단위로 같은지
(d) 기존 F·G 계산 그대로: 고치기 전 e2fg(scratch/e2fg_before_wxanom.py.bak)와 지금 e2fg 의 F·G 칸 값·범주가 같은 300쌍에서 비트 단위로 같음,
    기존 F·G·조합 결과 config.json 의 입력 칸 이름·칸 수 설명 = 지금 코드가 만드는 것
(e) 기록: 57칸 결측 비율(분할별), 변수별 a_t 결측 비율
"""
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import config as C1  # noqa: E402
import e2fg as G  # noqa: E402
from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import Mem, read_preds  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_wxanom.txt"
SCR = E.EXP2_DIR / "scratch" / "wxanom"
OLD_E2FG = E.EXP2_DIR / "scratch" / "e2fg_before_wxanom.py.bak"
K = E.WXANOM_K
V = C1.WEATHER_VARS
NAN = float("nan")
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


def close(a, b, rtol, atol):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=rtol, atol=atol))


def maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = ~np.isnan(a) & ~np.isnan(b)
    if not m.any():
        return 0.0, 0.0
    d = np.abs(a[m] - b[m])
    return float(d.max()), float((d / np.maximum(np.abs(b[m]), 1e-300)).max())


def _day(d, k):
    return str(np.datetime64(d, "D") + np.timedelta64(int(k), "D"))


# ---------------------------------------------------------------- (0) 설정
def t_config(ctx):
    m = [e for e in E.EXPERIMENTS if e["stage"] == "m"]
    ids = [e["id"] for e in m]
    a3 = E.EXP_BY_ID["A3"]["feats"]
    ok_ids = ids == ["M1", "A3_M1"] and all(E.exp_seeds(e, False) == [42, 43, 44] for e in m) and E.exp_seeds(m[0], True) == [42]
    ok_feats = (E.EXP_BY_ID["M1"]["feats"] == dict(E.BASE_FEATS, ext=["wxanom"])
                and E.EXP_BY_ID["A3_M1"]["feats"] == dict(a3, ext=["wxanom"]) and E.EXP_BY_ID["A3_M1"]["feats"]["k"] == "k*")
    st = E.STAGES
    ok_stage = "m" in st and st.index("m") == st.index("tables") - 1 and st[-1] == "tables"
    ok_re = not any(G.is_combo_id(x) for x in ids) and not any(x in ids for x in ("A3_F4", "A3_G1G2G3"))
    names = G.ext_names(["wxanom"])
    ok_names = len(names) == 57 == 19 * 3 and len(set(names)) == 57 and len(V) == 19
    ps1 = (E.EXP2_DIR / "run_exp2.ps1").read_text(encoding="utf-8-sig")
    vs = [ln for ln in ps1.splitlines() if "ValidateSet" in ln]
    ok_ps1 = len(vs) == 1 and "'combo', 'm', 'tables'" in vs[0]
    check("(0) 실험 M1·A3_M1: 단계 'm', 시드 42·43·44(smoke 42), 계획서 대응 칸",
          ok_ids and all(e["plan"] == "1차 보고서 표 9 ② 이동평균 차감" for e in m), f"{ids}")
    check("(0) 입력: M1 = E00 + wxanom, A3_M1 = A3(이력 3묶음, k = k*) + wxanom", ok_feats,
          f"M1 {E.EXP_BY_ID['M1']['feats']}, A3_M1 {E.EXP_BY_ID['A3_M1']['feats']}")
    check("(0) STAGES: 'm' 이 'tables' 바로 앞", ok_stage, f"{st}")
    check("(0) 실험 ID 가 조합 ID 정규식(^A3_(?:F[1-4]|G[1-3])+$)과 겹치지 않음", ok_re)
    check("(0) 칸 57개 = 19변수 x 3, 이름 중복 없음", ok_names, f"{names[:3]} … {names[-1]}")
    check("(0) run_exp2.ps1 ValidateSet 에 'm'(combo 와 tables 사이)", ok_ps1, vs[0].strip() if vs else "ValidateSet 없음")
    # 창 범위: 모든 분할, h 1..60 -> 창 첫날(T−50)의 앞 30일(T−80)도 기상표 안인지
    lo = min(int((ctx.sp[sp]["d"] - E.H_MAX).min()) - E.L for sp in C1.SPLITS)
    first = ctx.weather.date0 + np.timedelta64(lo, "D")
    check(f"(0) 실험 창의 가장 이른 날 − {K}일이 기상표 안(기상표 시작 전 날을 쓰는 a_t 없음)", lo - K >= 0,
          f"가장 이른 창 첫날 {first} (기상표 시작 {ctx.weather.date0} 뒤 {lo}일), 그 앞 {K}일 시작 {first - np.timedelta64(K, 'D')}")


# ---------------------------------------------------------------- (a) 원본 반복문 대조
def pick_pairs(ctx, rng, n_per=100):
    picks = []
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        for j in rng.choice(n, n_per, replace=False):
            picks.append((sp, int(j), int(rng.integers(1, E.H_MAX + 1))))
    return picks


def ref_anom(raw, sgg, day, vi, incl_today=False):
    """반복문: a = x_day − (day−30 .. day−1 중 결측 아닌 값 평균). incl_today 면 day−29 .. day (대조군)."""
    x = raw.get((sgg, day))
    x = NAN if x is None else float(x[vi])
    ks = range(0, K) if incl_today else range(1, K + 1)
    prev = []
    for k in ks:
        r = raw.get((sgg, _day(day, -k)))
        if r is not None and not math.isnan(float(r[vi])):
            prev.append(float(r[vi]))
    if math.isnan(x) or not prev:
        return NAN
    return x - sum(prev) / len(prev)


def ref_stats(a):
    """a: 창 날마다 값(시간순, 마지막 = T−1) -> [평균, 최근 10일 평균, 최대] (결측 빼고, 전부 결측이면 결측)."""
    ok = [x for x in a if not math.isnan(x)]
    last = [x for x in a[-10:] if not math.isnan(x)]
    return [sum(ok) / len(ok) if ok else NAN, sum(last) / len(last) if last else NAN, max(ok) if ok else NAN]


def t_values(ctx, rng):
    picks = pick_pairs(ctx, rng)
    t0 = time.time()
    need, sggs = set(), set()
    info = []
    for sp, j, h in picks:
        row = ctx.rows.iloc[ctx.sp[sp]["idx"][j]]
        sgg = str(row["sgg_code"])
        T = np.datetime64(row["d"], "D") - np.timedelta64(h, "D")
        info.append((sp, j, h, sgg, T))
        sggs.add(sgg)
        for k in range(-(E.L + K + 1), 1):
            need.add(f"{sgg}|{_day(T, k)}")
    t = pq.read_table(C1.WEATHER_PATH, columns=["sgg_code", "date"] + V, filters=[("sgg_code", "in", sorted(sggs))]).to_pandas()
    t["k"] = t["sgg_code"].astype(str) + "|" + pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
    t = t[t["k"].isin(need)]
    vals = t[V].to_numpy(dtype=np.float64)                 # parquet float32 -> float64 (값 그대로)
    raw = {tuple(k.split("|")): vals[i] for i, k in enumerate(t["k"].to_numpy())}
    say(f"    대조 쌍 {len(picks)}개, 기상 parquet 원본에서 필요한 (시군구, 날짜) {len(raw):,}행만 읽음 ({time.time() - t0:.1f}s, 시군구 {len(sggs)}개)")
    AN = G.get_wxanom(ctx)
    names = G.ext_names(["wxanom"])
    by_sp = {}
    for sp, j, h, _, _ in info:
        by_sp.setdefault(sp, []).append((j, h))
    batch = {sp: G.ext_features(ctx, sp, np.array([j for j, _ in v]), np.array([h for _, h in v]), ["wxanom"]) for sp, v in by_sp.items()}
    cnt = {sp: 0 for sp in by_sp}
    bad_grid = bad_sum = bad_batch = 0
    diff_incl_win = diff_incl_today = 0
    md_grid, md_sum = [0.0, 0.0], [0.0, 0.0]
    nan_ref = 0
    first_bad = None
    t1 = time.time()
    for sp, j, h, sgg, T in info:
        got = G.ext_features(ctx, sp, np.array([j]), h, ["wxanom"])[0]
        gb = batch[sp][cnt[sp]]
        cnt[sp] += 1
        bad_batch += int(not np.array_equal(got, gb, equal_nan=True))
        s = int(ctx.sp[sp]["s"][j])
        tT = int(ctx.sp[sp]["d"][j]) - h
        days = [_day(T, k) for k in range(-E.L, 0)]          # T−50 .. T−1
        A_ref = np.array([[ref_anom(raw, sgg, dd, vi) for vi in range(len(V))] for dd in days])     # (50, 19)
        A_got = AN[s, tT - E.L:tT, :]
        if not close(A_got, A_ref, 1e-9, 1e-9):
            bad_grid += 1
            first_bad = first_bad or ("grid", sp, j, h)
        md_grid[0] = max(md_grid[0], maxdiff(A_got, A_ref)[0])
        ref = []
        for vi in range(len(V)):
            ref += ref_stats(list(A_ref[:, vi]))
        nan_ref += int(np.isnan(ref).sum())
        if not close(got, ref, 1e-6, 1e-10):
            bad_sum += 1
            first_bad = first_bad or ("sum", sp, j, h, list(zip(names, got, ref))[:6])
        md = maxdiff(got, ref)
        md_sum = [max(md_sum[0], md[0]), max(md_sum[1], md[1])]
        # 대조군 1: 창을 T−49..T (T 포함)
        days_i = [_day(T, k) for k in range(-E.L + 1, 1)]
        ref_i = []
        for vi in range(len(V)):
            ref_i += ref_stats([ref_anom(raw, sgg, dd, vi) for dd in days_i])
        diff_incl_win += int(not close(got, ref_i, 1e-6, 1e-10))
        # 대조군 2: 앞 30일에 당일 포함
        ref_t = []
        for vi in range(len(V)):
            ref_t += ref_stats([ref_anom(raw, sgg, dd, vi, incl_today=True) for dd in days])
        diff_incl_today += int(not close(got, ref_t, 1e-6, 1e-10))
    n = len(info)
    say(f"    반복문 대조 {time.time() - t1:.1f}s")
    check("(a) 격자 a_t(창 T−50..T−1, 19변수) = 기상 parquet 원본 반복문 계산(앞 30일 결측 빼고 평균, 당일 제외)", bad_grid == 0,
          f"불일치 {bad_grid}/{n} 쌍, 최대 절대 차이 {md_grid[0]:.2e}")
    check("(a) 요약 57칸(변수마다 평균·최근 10일 평균·최대) = 반복문 계산(float32 반올림 허용 rtol 1e-6)", bad_sum == 0,
          f"불일치 {bad_sum}/{n} 쌍, 최대 절대 차이 {md_sum[0]:.2e}, 최대 상대 차이 {md_sum[1]:.2e}, 참조 결측 칸 {nan_ref}/{n * 57}")
    check("(a) 한 쌍씩(h 정수) = 한꺼번에(h 배열, 학습 경로) 비트 단위", bad_batch == 0, f"불일치 {bad_batch}/{n}")
    check("(a) 대조군: 창을 T 포함(T−49..T)으로 하면 달라지는 쌍이 있음", diff_incl_win > 0, f"{diff_incl_win}/{n} 쌍이 달라짐")
    check("(a) 대조군: 앞 30일에 당일을 넣으면(x_t − mean(x_(t−29..t))) 달라지는 쌍이 있음", diff_incl_today > 0, f"{diff_incl_today}/{n} 쌍이 달라짐")
    if first_bad:
        say(f"    [첫 불일치] {first_bad}")


def t_matrix(ctx, rng):
    """M1·A3_M1 특성 행렬 = E00·A3 특성 행렬 + 57칸 (FeatureBuilder 경로, 학습·평가 둘 다)."""
    from e2models import FeatureBuilder
    k_star = json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]
    bad = []
    for base_id, m_id in (("E00", "M1"), ("A3", "A3_M1")):
        fb0 = dict(E.EXP_BY_ID[base_id]["feats"])
        fb1 = dict(E.EXP_BY_ID[m_id]["feats"])
        for f in (fb0, fb1):
            if f.get("k") == "k*":
                f["k"] = k_star
        b0, b1 = FeatureBuilder(ctx, fb0), FeatureBuilder(ctx, fb1)
        ok_names = b1.names == b0.names + G.ext_names(["wxanom"]) and b1.cat_names == b0.cat_names
        for sp in C1.SPLITS:
            pos = rng.choice(len(ctx.sp[sp]["idx"]), 500, replace=False)
            hh = rng.integers(1, E.H_MAX + 1, size=500)
            for h in (hh, int(hh[0])):                         # 학습 경로(h 배열)와 평가 경로(h 정수, 검증·시험은 기상 요약 캐시)
                X0, X1 = b0.num(sp, pos, h), b1.num(sp, pos, h)
                ext = G.ext_features(ctx, sp, pos, h, ["wxanom"])
                ok = (X1.shape[1] == X0.shape[1] + 57 and np.array_equal(X1[:, :X0.shape[1]], X0, equal_nan=True)
                      and np.array_equal(X1[:, X0.shape[1]:], ext, equal_nan=True))
                if not ok:
                    bad.append((m_id, sp, "배열" if not np.isscalar(h) else "정수"))
        say(f"    {m_id}: 숫자 칸 {len(b1.names)} = {base_id} {len(b0.names)} + 57, 범주 {b1.cat_names}")
        if not ok_names:
            bad.append((m_id, "이름"))
    check("(a) M1·A3_M1 특성 행렬 = E00·A3 특성 행렬(비트 단위) + 57칸, 범주 칸 같음 (h 배열·정수 둘 다)", not bad, f"불일치 {bad}")


# ---------------------------------------------------------------- (b) 앞날 자료
def t_future(ctx, rng):
    W = ctx.weather.W
    AN = G.get_wxanom(ctx)
    sd = np.nanstd(W.reshape(-1, W.shape[2]).astype(np.float64), axis=0)
    nbad_past = nbad_feat = n_ctrl = n_eff = 0
    tests = []
    for sp in C1.SPLITS:
        for j in rng.choice(len(ctx.sp[sp]["idx"]), 25, replace=False):
            tests.append((sp, int(j), int(rng.integers(1, E.H_MAX + 1))))
    one_ok = bool(np.array_equal(G.wx_anomaly(W[3:4])[0], AN[3], equal_nan=True))
    for sp, j, h in tests:
        s = int(ctx.sp[sp]["s"][j])
        t = int(ctx.sp[sp]["d"][j]) - h                    # T 의 기상표 날짜 번호
        base = G.ext_features(ctx, sp, np.array([j]), h, ["wxanom"])
        row = AN[s].copy()
        try:
            Wm = W[s].astype(np.float64).copy()
            nfut = Wm[t:].shape
            Wm[t:] = rng.normal(size=nfut) * sd[None, :] * 3 + rng.normal(size=(1, nfut[1])) * 50
            Wm[t:][rng.random(nfut) < 0.1] = np.nan
            Am = G.wx_anomaly(Wm[None].astype(np.float32))[0]
            nbad_past += int(not np.array_equal(Am[:t], row[:t], equal_nan=True))
            n_eff += int(not np.array_equal(Am[t:], row[t:], equal_nan=True))
            AN[s] = Am
            g = G.ext_features(ctx, sp, np.array([j]), h, ["wxanom"])
            nbad_feat += int(not np.array_equal(g, base, equal_nan=True))
            # 대조군: T−1 하루만 바꿈
            Wc = W[s].astype(np.float64).copy()
            Wc[t - 1] = Wc[t - 1] + 10.0 * sd + 1.0
            AN[s] = G.wx_anomaly(Wc[None].astype(np.float32))[0]
            gc_ = G.ext_features(ctx, sp, np.array([j]), h, ["wxanom"])
            n_ctrl += int(not np.array_equal(gc_, base, equal_nan=True))
        finally:
            AN[s] = row
    n = len(tests)
    check("(b) 시군구 하나만 따로 계산한 격자 = 전체 격자(시군구끼리 섞이지 않음)", one_ok)
    check("(b) T 이후(T 당일 포함) 기상 19칸을 무작위(결측 10% 섞음)로 바꿔 다시 계산해도 T 전 날의 a_t 전부 같음", nbad_past == 0 and n_eff == n,
          f"T 전 a_t 다른 쌍 {nbad_past}/{n}, (바꾼 것이 실제로 T 이후 a_t 를 바꾼 쌍 {n_eff}/{n})")
    check("(b) 같은 조건에서 57칸 비트 단위로 같음", nbad_feat == 0, f"불일치 {nbad_feat}/{n}")
    check("(b) 대조군: T−1 하루만 바꾸면 57칸이 달라짐", n_ctrl == n, f"{n_ctrl}/{n} 쌍이 달라짐")
    G_after = np.array_equal(AN, G.wx_anomaly(W), equal_nan=True)
    check("(b) 점검 뒤 격자 원래대로", G_after)


# ---------------------------------------------------------------- (c) E00 재현
def t_e00(ctx, log):
    import shutil
    from run import check_windows, run_one
    from selftest_fg import _cache_meta_ok
    root = SCR / "selftest_e00"
    P = {"results": root / "results", "tables": root / "tables", "logs": root / "logs", "cache": E.EXP2_DIR / "cache",
         "progress": root / "logs" / "progress.log"}
    if (P["results"] / "E00__s42").exists():
        shutil.rmtree(P["results"] / "E00__s42")
    P["results"].mkdir(parents=True, exist_ok=True)
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        say("    기상 요약 캐시 메타가 다름 -> 캐시를 쓰지 않고 그 자리에서 계산(캐시 파일은 건드리지 않음)")
    else:
        say("    기상 요약 캐시(cache/eval_wsum_L050_*.npy) 메타 같음 -> 읽기만 함")
    mem = Mem()
    mem.start()
    win = check_windows(ctx, log)
    t0 = time.time()
    s = run_one(E.EXP_BY_ID["E00"], 42, ctx, P, log, mem, False, win)
    mem.stop()
    d1 = E.EXP1_RESULTS / "catboost__summary__L050"
    d2 = P["results"] / "E00__s42"
    d3 = E.EXP2_DIR / "results" / "E00__s42"
    for sp in ("val", "test"):
        p2, y2, i2 = read_preds(d2 / f"pred_{sp}.parquet")
        for lab, dref in (("1차 catboost__summary__L050", d1), ("2차 results/E00__s42", d3)):
            p1, y1, i1 = read_preds(dref / f"pred_{sp}.parquet")
            same_rows = i1.astype(str).equals(i2.astype(str)) and np.array_equal(y1, y2)
            check(f"(c) E00 시드 42 {sp} 예측 = {lab} (행·y·p 비트 단위)", same_rows and np.array_equal(p1, p2),
                  f"{p1.shape}, 최대 차이 {float(np.max(np.abs(p1.astype(np.float64) - p2))):.1e}")
    m1 = pd.read_csv(d1 / "metrics_by_h.csv")
    m2 = pd.read_csv(d2 / "metrics_by_h.csv")
    cols = ["pr_auc", "roc_auc", "threshold", "tp", "fn", "fp", "tn", "precision", "recall", "f1", "f2"]
    check("(c) E00 시드 42 h 별 지표 = 1차", np.allclose(m1[cols].to_numpy(float), m2[cols].to_numpy(float), rtol=0, atol=1e-12))
    s1 = json.loads((d1 / "summary.json").read_text(encoding="utf-8"))
    check("(c) E00 시드 42 최적 라운드·학습 라운드 = 1차", s["best_rounds_or_epoch"] == s1["best_rounds_or_epoch"]
          and s["n_rounds_trained"] == s1["n_rounds_trained"],
          f"{s['best_rounds_or_epoch']}/{s['n_rounds_trained']} vs {s1['best_rounds_or_epoch']}/{s1['n_rounds_trained']}, 총 {time.time() - t0:.0f}s")
    c2 = json.loads((d2 / "config.json").read_text(encoding="utf-8"))
    c3 = json.loads((d3 / "config.json").read_text(encoding="utf-8"))
    check("(c) E00 config: 입력 칸 이름·개수·설명 = 기존 results/E00__s42, 외부·이동평균 기록 칸 없음",
          c2["feature_columns"] == c3["feature_columns"] and c2["n_features"] == c3["n_features"]
          and c2["n_features_desc"] == c3["n_features_desc"] and "ext" not in c2, f"{c2['n_features_desc']}")
    say(f"    E00 재현 메모리: private {s.get('peak_private_gb')} GB / rss {s.get('peak_rss_gb')} GB")


# ---------------------------------------------------------------- (d) 기존 F·G 계산 그대로
def _load_old():
    ld = importlib.machinery.SourceFileLoader("e2fg_old_wxanom", str(OLD_E2FG))
    spec = importlib.util.spec_from_loader("e2fg_old_wxanom", ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


def t_fg_same(ctx, rng):
    O = _load_old()
    blocks = list(O.BLOCK_ORDER)
    check("(d) F·G 묶음 목록·설명·조합 규칙 = 고치기 전", O.BLOCK_ORDER == G.BLOCK_ORDER and O.FG_BLOCKS == G.FG_BLOCKS
          and all(O.BLOCK_DESC[b] == G.BLOCK_DESC[b] for b in blocks) and O.ext_names(blocks) == G.ext_names(blocks)
          and all(O.combo_plan(s) == G.combo_plan(s) for s in ([], ["F1"], ["F1", "F4"], ["F1", "G1", "G3"], list(O.FG_GROUPS)))
          and O.ext_desc(blocks) == G.ext_desc(blocks))
    picks = pick_pairs(ctx, rng)
    saved = getattr(ctx, "_fg_ext", None)
    bad = badc = 0
    try:
        for sp in C1.SPLITS:
            pp = [(j, h) for s_, j, h in picks if s_ == sp]
            pos = np.array([j for j, _ in pp])
            hh = np.array([h for _, h in pp])
            ctx._fg_ext = None
            old = O.ext_features(ctx, sp, pos, hh, blocks)
            oldc = O.ext_cats(ctx, sp, pos, blocks)
            old1 = O.ext_features(ctx, sp, pos, int(hh[0]), blocks)
            ctx._fg_ext = None
            new = G.ext_features(ctx, sp, pos, hh, blocks)
            newc = G.ext_cats(ctx, sp, pos, blocks)
            new1 = G.ext_features(ctx, sp, pos, int(hh[0]), blocks)
            bad += int(not (np.array_equal(old, new, equal_nan=True) and np.array_equal(old1, new1, equal_nan=True)))
            badc += int(not np.array_equal(oldc, newc))
    finally:
        ctx._fg_ext = saved
    check("(d) F·G 전체 묶음 칸 값(h 배열·정수) = 고치기 전 e2fg (300쌍, 비트 단위)", bad == 0, f"불일치 분할 {bad}/3")
    check("(d) F3 범주 aw_label = 고치기 전", badc == 0)
    # 기존 결과 config.json 의 칸 이름·설명 = 지금 코드
    from e2models import CatRunner
    rows, nb = [], 0
    for d in sorted((E.EXP2_DIR / "results").glob("*__s42")):
        cf = d / "config.json"
        if not cf.exists():
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        if not (c.get("feats") or {}).get("ext") or c.get("model") != "cat":
            continue
        exp = E.EXP_BY_ID.get(c["exp"]) or dict(id=c["exp"], model="cat", feats=c["feats"])
        r = CatRunner(ctx, exp, 42, c["feats"], lambda *_: None, 1, 1)
        ok = r.cols == c["feature_columns"] and r.n_features_desc == c["n_features_desc"] and r.n_features == c["n_features"] \
            and G.ext_info(c["feats"]["ext"])["names"] == c["ext"]["names"]
        nb += int(not ok)
        rows.append(f"{c['exp']}{'' if ok else '(다름)'}")
    check("(d) 기존 F·G·조합 결과(시드 42) config.json 입력 칸 이름·칸 수·설명 = 지금 코드", nb == 0 and len(rows) > 0, f"{len(rows)}개: {', '.join(rows)}")


# ---------------------------------------------------------------- (e) 기록
def t_record(ctx, rng):
    AN = G.get_wxanom(ctx)
    W = ctx.weather.W
    d0 = int((np.datetime64("2014-12-18") - ctx.weather.date0).astype(int))
    d1 = int((np.datetime64(C1.TEST_END) - ctx.weather.date0).astype(int))
    rw = np.isnan(W[:, d0:d1 + 1]).mean(axis=(0, 1))
    ra = np.isnan(AN[:, d0:d1 + 1]).mean(axis=(0, 1))
    say("    [기록] 변수별 결측 비율(시군구 x 날짜 2014-12-18~2026-06-30) 원 기상 -> a_t: "
        + ", ".join(f"{v} {a:.2%}->{b:.2%}" for v, a, b in zip(V, rw, ra)))
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        pos = rng.choice(n, min(n, 20000), replace=False)
        hh = rng.integers(1, E.H_MAX + 1, size=len(pos))
        X = G.ext_features(ctx, sp, pos, hh, ["wxanom"])
        nanc = np.isnan(X)
        worst = np.argsort(-nanc.mean(0))[:4]
        names = G.ext_names(["wxanom"])
        say(f"    [기록] {sp} (행 x 무작위 h) {len(pos):,}쌍: 57칸 중 결측 하나 이상인 쌍 {nanc.any(1).mean():.2%}, 칸 전체 결측 비율 {nanc.mean():.3%}, "
            f"결측 많은 칸 " + ", ".join(f"{names[i]} {nanc[:, i].mean():.2%}" for i in worst))


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    log = Logger(SCR / "progress_selftest_wxanom.log")
    rng = np.random.default_rng(20261019)
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    from selftest_fg import _cache_meta_ok
    if not _cache_meta_ok(ctx):                              # 캐시 파일을 새로 쓰지 않도록(메타가 다르면 그 자리에서 계산)
        ctx.use_cache = False
        say("    기상 요약 캐시 메타가 다름 -> 캐시를 쓰지 않음")
    t_config(ctx)
    G.get_wxanom(ctx)
    t_values(ctx, rng)
    t_matrix(ctx, rng)
    t_future(ctx, rng)
    t_fg_same(ctx, rng)
    t_record(ctx, rng)
    if "--no-e00" not in sys.argv:
        t_e00(ctx, log)
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
