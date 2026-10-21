# -*- coding: utf-8 -*-
"""Q 생장 범위('rgwx'·'rgsoil'·'rgfood', Q1~Q4) 자체 점검 (전체 자료). 결과: logs/selftest_rg.txt (UTF-8). 하나라도 실패하면 종료 코드 1.
results/·tables/·cache/ 에는 쓰지 않는다(기상 요약 캐시는 메타가 같을 때만 읽기).

(0) 설정: Q1~Q4 단계 'q'·시드 42·43·44, 입력 = A3 + 묶음, STAGES 에서 'q' 가 tables 바로 앞, 조합 ID 정규식과 겹치지 않음,
    칸 수 12·3·4·19(이름 중복 없음), run_exp2.ps1 ValidateSet 에 'q'
(a) 무작위 (행, h) 300쌍(분할마다 100): 생장범위 parquet 원본을 따로 읽어 반복문으로 다시 계산 -> 코드 값과 대조
    rgwx = 창 T−50..T−1 칸마다 결측 뺀 평균·최근 10일 평균, rgsoil = 날짜 T 행, rgfood = 그 행 3단계 코드의 4칸(코드없음·표에 없으면 결측).
    한 쌍씩(h 정수) = 한꺼번에(h 배열). 대조군: 창을 T 포함(T−49..T)으로 하면 달라지는 쌍이 있음, 토양을 T+1 행으로 하면 달라지는 쌍이 있음
(b) 앞날 자료 영향 없음: 그 시군구의 T 당일·이후 생장 범위 값(기상·토양)을 무작위로 바꿔도 칸이 비트 단위로 같음. 대조군: T−1(기상)·T(토양 행)를 바꾸면 달라짐
(c) Q1~Q4 특성 행렬 = A3 특성 행렬(비트 단위) + 묶음 칸 (FeatureBuilder, h 배열·정수)
(d) 기존 계산 그대로: 고치기 전 e2fg(scratch/q/e2fg_before_q.py.bak)와 지금 e2fg 의 F·G·M 칸 값·범주가 300쌍에서 비트 단위로 같음,
    이름·설명·칸 수 설명 같음, 기존 결과(시드 42, ext 있는 CatBoost) config.json 의 입력 칸·설명 = 지금 코드
(e) 기록: 분할별 칸 결측 비율
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_rg.txt"
SCR = E.EXP2_DIR / "scratch" / "q"
OLD_E2FG = SCR / "e2fg_before_q.py.bak"
QIDS = ["Q1", "Q2", "Q3", "Q4"]
QBLK = {"Q1": ["rgwx"], "Q2": ["rgsoil"], "Q3": ["rgfood"], "Q4": ["rgwx", "rgsoil", "rgfood"]}
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


def _day(d, k):
    return str(np.datetime64(d, "D") + np.timedelta64(int(k), "D"))


# ---------------------------------------------------------------- (0) 설정
def t_config():
    q = [e for e in E.EXPERIMENTS if e["stage"] == "q"]
    ids = [e["id"] for e in q]
    a3 = E.EXP_BY_ID["A3"]["feats"]
    check("(0) 실험 Q1~Q4: 단계 'q', 시드 42·43·44(smoke 42)",
          ids == QIDS and all(E.exp_seeds(e, False) == [42, 43, 44] for e in q) and E.exp_seeds(q[0], True) == [42], f"{ids}")
    check("(0) 입력 = A3(k = k*) + 생장 범위 묶음",
          all(E.EXP_BY_ID[i]["feats"] == dict(a3, ext=QBLK[i]) for i in QIDS) and a3["k"] == "k*",
          "; ".join(f"{i} {E.EXP_BY_ID[i]['feats'].get('ext')}" for i in QIDS))
    st = E.STAGES
    check("(0) STAGES: 'q' 가 'tables' 바로 앞", "q" in st and st.index("q") == st.index("tables") - 1 and st[-1] == "tables", f"{st}")
    check("(0) 실험 ID 가 조합 ID 정규식과 겹치지 않음", not any(G.is_combo_id(x) for x in ids))
    ns = {i: G.ext_names(QBLK[i]) for i in QIDS}
    check("(0) 칸 수 Q1 12·Q2 3·Q3 4·Q4 19, 이름 중복 없음",
          [len(ns[i]) for i in QIDS] == [12, 3, 4, 19] and len(set(ns["Q4"])) == 19 and ns["Q4"] == ns["Q1"] + ns["Q2"] + ns["Q3"],
          f"{ns['Q4']}")
    ps1 = (E.EXP2_DIR / "run_exp2.ps1").read_text(encoding="utf-8-sig")
    vs = [ln for ln in ps1.splitlines() if "ValidateSet" in ln]
    check("(0) run_exp2.ps1 ValidateSet 에 'q'(n 과 tables 사이)", len(vs) == 1 and "'n', 'q', 'tables'" in vs[0], vs[0].strip() if vs else "")
    check("(0) F·G 조합 규칙(BLOCK_ORDER)에 생장 범위 묶음 없음", not any(b in G.BLOCK_ORDER for b in G.RG_BLOCKS))


# ---------------------------------------------------------------- (a) 원본 반복문 대조
def pick_pairs(ctx, rng, n_per=100):
    picks = []
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        for j in rng.choice(n, n_per, replace=False):
            picks.append((sp, int(j), int(rng.integers(1, E.H_MAX + 1))))
    return picks


def ref_wx(raw, sgg, days):
    """days(시간순) 각 날의 6칸 -> 칸마다 [결측 뺀 평균, 마지막 10일 결측 뺀 평균]."""
    out = []
    for vi in range(len(G.RG_WX_VARS)):
        xs = []
        for dd in days:
            r = raw.get((sgg, dd))
            xs.append(NAN if r is None else float(r[vi]))
        ok = [x for x in xs if not math.isnan(x)]
        last = [x for x in xs[-10:] if not math.isnan(x)]
        out += [sum(ok) / len(ok) if ok else NAN, sum(last) / len(last) if last else NAN]
    return out


def t_values(ctx, rng):
    picks = pick_pairs(ctx, rng)
    t0 = time.time()
    info, sggs, need = [], set(), set()
    for sp, j, h in picks:
        row = ctx.rows.iloc[ctx.sp[sp]["idx"][j]]
        sgg = str(row["sgg_code"])
        T = np.datetime64(row["d"], "D") - np.timedelta64(h, "D")
        info.append((sp, j, h, sgg, T, str(ctx.sp[sp]["p3"][j])))
        sggs.add(sgg)
        for k in range(-E.L - 1, 2):
            need.add(f"{sgg}|{_day(T, k)}")

    def load(path, cols):
        t = pq.read_table(path, columns=["sgg_code", "date"] + cols, filters=[("sgg_code", "in", sorted(sggs))]).to_pandas()
        t["k"] = t["sgg_code"].astype(str) + "|" + pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
        t = t[t["k"].isin(need)]
        vals = t[cols].to_numpy(dtype=np.float64)
        return {tuple(k.split("|")): vals[i] for i, k in enumerate(t["k"].to_numpy())}
    rawW = load(G.RG_WX_PATH, G.RG_WX_VARS)
    rawS = load(G.RG_SOIL_PATH, G.RG_SOIL_VARS)
    ft = pq.read_table(G.RG_FOOD_PATH).to_pandas()
    food = {str(r["train_prdlst_code_2"]): [NAN if pd.isna(r[c]) else float(r[c]) for c in G.RG_FOOD_VARS] for _, r in ft.iterrows()}
    say(f"    대조 쌍 {len(picks)}개, 원본에서 기상 범위 {len(rawW):,}행·토양 pH 범위 {len(rawS):,}행·품목 {len(food)}개 읽음 "
        f"({time.time() - t0:.1f}s, 시군구 {len(sggs)}개)")
    blocks = list(G.RG_BLOCKS)
    by_sp = {}
    for sp, j, h, *_ in info:
        by_sp.setdefault(sp, []).append((j, h))
    batch = {sp: G.ext_features(ctx, sp, np.array([j for j, _ in v]), np.array([h for _, h in v]), blocks) for sp, v in by_sp.items()}
    cnt = {sp: 0 for sp in by_sp}
    bad = {"rgwx": 0, "rgsoil": 0, "rgfood": 0}
    bad_batch = diff_win = diff_soil = 0
    nan_ref = 0
    food_codes = {"표에 있음": 0, "코드없음": 0, "표에 없음": 0}
    first_bad = None
    for sp, j, h, sgg, T, p3 in info:
        got = G.ext_features(ctx, sp, np.array([j]), h, blocks)[0]
        gb = batch[sp][cnt[sp]]
        cnt[sp] += 1
        bad_batch += int(not np.array_equal(got, gb, equal_nan=True))
        days = [_day(T, k) for k in range(-E.L, 0)]
        rw = ref_wx(rawW, sgg, days)
        r = rawS.get((sgg, str(T)))
        rs = [NAN] * 3 if r is None else [float(x) for x in r]
        if p3 in food:
            rf = food[p3]
            food_codes["표에 있음"] += 1
        else:
            rf = [NAN] * 4
            food_codes["코드없음" if p3 == C1.NULL_CODE else "표에 없음"] += 1
        ref = rw + rs + rf
        nan_ref += int(np.isnan(ref).sum())
        for nm, a, b in (("rgwx", 0, 12), ("rgsoil", 12, 15), ("rgfood", 15, 19)):
            if not close(got[a:b], ref[a:b], 1e-6, 1e-7):
                bad[nm] += 1
                first_bad = first_bad or (nm, sp, j, h, sgg, str(T), list(got[a:b]), ref[a:b])
        rw_i = ref_wx(rawW, sgg, [_day(T, k) for k in range(-E.L + 1, 1)])
        diff_win += int(not close(got[:12], rw_i, 1e-6, 1e-7))
        r1 = rawS.get((sgg, _day(T, 1)))
        rs1 = [NAN] * 3 if r1 is None else [float(x) for x in r1]
        diff_soil += int(not close(got[12:15], rs1, 1e-6, 1e-7))
    n = len(info)
    check("(a) rgwx 12칸 = 원본 반복문(창 T−50..T−1 결측 뺀 평균·최근 10일 평균, float32 반올림 허용)", bad["rgwx"] == 0, f"불일치 {bad['rgwx']}/{n}")
    check("(a) rgsoil 3칸 = 원본 날짜 T 행", bad["rgsoil"] == 0, f"불일치 {bad['rgsoil']}/{n}")
    check("(a) rgfood 4칸 = 원본 3단계 코드 행(코드없음·표에 없으면 결측)", bad["rgfood"] == 0, f"불일치 {bad['rgfood']}/{n}, 코드 {food_codes}")
    check("(a) 한 쌍씩(h 정수) = 한꺼번에(h 배열) 비트 단위", bad_batch == 0, f"불일치 {bad_batch}/{n}")
    check("(a) 대조군: 기상 창을 T 포함(T−49..T)으로 하면 달라지는 쌍이 있음", diff_win > 0, f"{diff_win}/{n}")
    check("(a) 대조군: 토양을 T+1 행으로 하면 달라지는 쌍이 있음", diff_soil > 0, f"{diff_soil}/{n}")
    say(f"    참조 결측 칸 {nan_ref}/{n * 19}")
    if first_bad:
        say(f"    [첫 불일치] {first_bad}")


# ---------------------------------------------------------------- (b) 앞날 자료 영향 없음
def t_future(ctx, rng):
    R = G.get_rg(ctx)
    W0, S0 = R.WX, R.SOIL
    picks = pick_pairs(ctx, rng, 10)
    same_f = diff_wx = diff_soil = 0
    try:
        for sp, j, h in picks:
            s = int(ctx.sp[sp]["s"][j])
            t = int(ctx.sp[sp]["d"][j]) - h
            base = G.ext_features(ctx, sp, np.array([j]), h, ["rgwx", "rgsoil"])
            W1, S1 = W0.copy(), S0.copy()
            noise = rng.random(W1[s, t:].shape).astype(np.float32)
            noise[rng.random(noise.shape) < 0.2] = np.nan
            W1[s, t:] = noise
            S1[s, t + 1:] = rng.random(S1[s, t + 1:].shape).astype(np.float32)   # 토양은 T 행을 쓰므로 T+1 부터 바꿈
            R.WX, R.SOIL = W1, S1
            same_f += int(np.array_equal(base, G.ext_features(ctx, sp, np.array([j]), h, ["rgwx", "rgsoil"]), equal_nan=True))
            W2, S2 = W0.copy(), S0.copy()
            W2[s, t - 1] = W2[s, t - 1] + 0.5 if not np.isnan(W2[s, t - 1]).all() else 0.5
            S2[s, t] = np.where(np.isnan(S2[s, t]), 0.5, S2[s, t] + 0.5)
            R.WX, R.SOIL = W2, S2
            x2 = G.ext_features(ctx, sp, np.array([j]), h, ["rgwx", "rgsoil"])
            diff_wx += int(not np.array_equal(base[:, :12], x2[:, :12], equal_nan=True))
            diff_soil += int(not np.array_equal(base[:, 12:], x2[:, 12:], equal_nan=True))
            R.WX, R.SOIL = W0, S0
    finally:
        R.WX, R.SOIL = W0, S0
    n = len(picks)
    check("(b) T 당일·이후 기상 범위, T+1 이후 토양 pH 범위를 바꿔도 칸이 비트 단위로 같음", same_f == n, f"{same_f}/{n}")
    check("(b) 대조군: T−1 기상 범위를 바꾸면 rgwx 가 달라짐", diff_wx == n, f"{diff_wx}/{n}")
    check("(b) 대조군: T 행 토양 pH 범위를 바꾸면 rgsoil 이 달라짐", diff_soil == n, f"{diff_soil}/{n}")


# ---------------------------------------------------------------- (c) 특성 행렬
def t_matrix(ctx, rng):
    from e2models import FeatureBuilder
    k_star = json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]
    f0 = dict(E.EXP_BY_ID["A3"]["feats"], k=k_star)
    b0 = FeatureBuilder(ctx, f0)
    for qid in QIDS:
        f1 = dict(E.EXP_BY_ID[qid]["feats"], k=k_star)
        b1 = FeatureBuilder(ctx, f1)
        blk = QBLK[qid]
        ok_names = b1.names == b0.names + G.ext_names(blk) and b1.cat_names == b0.cat_names
        bad = []
        for sp in C1.SPLITS:
            pos = rng.choice(len(ctx.sp[sp]["idx"]), 500, replace=False)
            hh = rng.integers(1, E.H_MAX + 1, size=500)
            for h in (hh, int(hh[0])):
                X0, X1 = b0.num(sp, pos, h), b1.num(sp, pos, h)
                ext = G.ext_features(ctx, sp, pos, h, blk)
                ok = (X1.shape[1] == X0.shape[1] + len(G.ext_names(blk)) and np.array_equal(X1[:, :X0.shape[1]], X0, equal_nan=True)
                      and np.array_equal(X1[:, X0.shape[1]:], ext, equal_nan=True))
                if not ok:
                    bad.append((sp, "정수" if np.isscalar(h) else "배열"))
        check(f"(c) {qid} 특성 행렬 = A3 행렬(비트 단위) + {blk} 칸, 범주 = A3 범주", ok_names and not bad,
              f"숫자 칸 {len(b1.names)} = A3 {len(b0.names)} + {len(G.ext_names(blk))}, 범주 {b1.cat_names}, 불일치 {bad}")


# ---------------------------------------------------------------- (d) 기존 계산 그대로
def _load_old():
    ld = importlib.machinery.SourceFileLoader("e2fg_old_q", str(OLD_E2FG))
    spec = importlib.util.spec_from_loader("e2fg_old_q", ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


def t_old_same(ctx, rng):
    O = _load_old()
    old_blocks = list(O.EXT_BLOCKS)
    sets = [[b] for b in old_blocks] + [list(O.BLOCK_ORDER), ["spi", "soilmed", "food"], ["spi", "soil", "food"], old_blocks]
    ok_txt = all(O.ext_names(s) == G.ext_names(s) and O.ext_desc(s) == G.ext_desc(s) and O.ext_count_desc(s) == G.ext_count_desc(s)
                 and O.ext_cat_names(s) == G.ext_cat_names(s) for s in sets)
    check("(d) 기존 묶음(F·G·M) 이름·설명·칸 수 설명·범주 = 고치기 전", ok_txt and O.BLOCK_ORDER == G.BLOCK_ORDER and O.FG_BLOCKS == G.FG_BLOCKS)
    check("(d) 설정 시트 M 줄 = 고치기 전", O.wxanom_settings_rows() == G.wxanom_settings_rows())
    picks = pick_pairs(ctx, rng)
    saved = (getattr(ctx, "_fg_ext", None), getattr(ctx, "_wxanom", None))
    bad = badc = 0
    for sp in C1.SPLITS:
        pp = [(j, h) for s_, j, h in picks if s_ == sp]
        pos = np.array([j for j, _ in pp])
        hh = np.array([h for _, h in pp])
        old = O.ext_features(ctx, sp, pos, hh, old_blocks)
        old1 = O.ext_features(ctx, sp, pos, int(hh[0]), old_blocks)
        oldc = O.ext_cats(ctx, sp, pos, old_blocks)
        new = G.ext_features(ctx, sp, pos, hh, old_blocks)
        new1 = G.ext_features(ctx, sp, pos, int(hh[0]), old_blocks)
        newc = G.ext_cats(ctx, sp, pos, old_blocks)
        bad += int(not (np.array_equal(old, new, equal_nan=True) and np.array_equal(old1, new1, equal_nan=True)))
        badc += int(not np.array_equal(oldc, newc))
    ctx._fg_ext, ctx._wxanom = saved
    check("(d) F·G·M 전체 묶음 칸 값(h 배열·정수) = 고치기 전 e2fg (300쌍, 비트 단위)", bad == 0, f"불일치 분할 {bad}/3")
    check("(d) F3 범주 aw_label = 고치기 전", badc == 0)
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
    check("(d) 기존 결과(시드 42, ext 있는 CatBoost) config.json 입력 칸 이름·칸 수·설명 = 지금 코드", nb == 0 and len(rows) > 0,
          f"{len(rows)}개: {', '.join(rows)}")


# ---------------------------------------------------------------- (e) 기록
def t_record(ctx, rng):
    R = G.get_rg(ctx)
    names = G.ext_names(G.RG_BLOCKS)
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        pos = rng.choice(n, min(n, 20000), replace=False)
        hh = rng.integers(1, E.H_MAX + 1, size=len(pos))
        X = G.ext_features(ctx, sp, pos, hh, G.RG_BLOCKS)
        r = np.isnan(X).mean(0)
        say(f"    [기록] {sp} (행 x 무작위 h) {len(pos):,}쌍 칸별 결측 비율: " + ", ".join(f"{nm} {v:.2%}" for nm, v in zip(names, r)))
        say(f"    [기록] {sp} 품목 칸: {R.stats[sp]}")


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    log = Logger(SCR / "progress_selftest_rg.log")
    rng = np.random.default_rng(20261019)
    t_config()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    from selftest_fg import _cache_meta_ok
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        say("    기상 요약 캐시 메타가 다름 -> 캐시를 쓰지 않음")
    G.get_rg(ctx)
    t_values(ctx, rng)
    t_future(ctx, rng)
    t_matrix(ctx, rng)
    t_old_same(ctx, rng)
    t_record(ctx, rng)
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
