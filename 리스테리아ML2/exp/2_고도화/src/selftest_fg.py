# -*- coding: utf-8 -*-
"""F·G 추가분 자체 점검 (전체 자료). 결과: logs/selftest_fg.txt (UTF-8). 하나라도 실패하면 종료 코드 1.
results/·tables/·cache/ 에는 쓰지 않는다(E00 재현 결과는 scratch/fg/selftest_e00/, 기상 요약 캐시는 메타가 같을 때만 읽기).

(0) 생장식 계수: 엑셀 '리스테리아 생장식.xlsx' 를 다시 읽어 쓰는 셀(F6·F19·F26·J6·D15·F15·F21·F28·J15·D41·F41·G41)에 코드의 숫자 글자가
    있는지, 뺀 Wijtes 모델 칸(F5·F18·F23)에 'not mentioned' 가 있는지
(a) F1~F3: 무작위 (행, h) 300쌍(분할마다 100)을 원 parquet 에서 반복문으로 직접 가져와 대조
    SPI = (sgg_code, date) 행을 T−50..T−1 날마다 찾아 순수 파이썬으로 요약, 'T 당일 포함(T−49..T)' 으로 만들면 값이 달라지는 쌍이 실제로 있는지
    토양 = (sgg_code, T) 행 6칸 그대로, T−1 행으로 하면 달라지는 쌍이 있는지 / 식품 = 3단계 코드로 표에서 찾은 값·표기
    한 쌍씩(h 정수) 계산과 한꺼번에(h 배열 = 학습 경로) 계산이 같은지
(b) G1~G3: 엑셀 원문 식대로 따로 짠 스칼라 함수(math)로 같은 300쌍을 다시 계산(기온·강수도 기상 parquet 에서 반복문으로)
    + 정의역 경계·밖(T ≤ Tmin, T ≥ Tmax, T ≥ 48, aw ≤ awmin, aw = 1.00, pH ≤ pHmin, pH ≥ pHmax, 결측)에서 벡터 식 = 스칼라 식
    + 카디널 항 최대 = 1(최적값에서), 0 ≤ f ≤ 1
(c) E00 시드 42 를 끝까지 실행해 1차 catboost__summary__L050·2차 results/E00__s42 예측과 비트 단위로 같은지
(d) 앞날 자료 영향 없음: T 이후(T 당일 포함) 칸을 무작위 값으로 바꿔도 F·G 값이 같음, G3 의 날 t 값 = rain(t−2) 식
(e) 조합 계획 규칙(가짜 선정 결과로): 없음 -> 대체 2개, 1개 -> 각각만, F1·F4 -> 합집합이 F4 와 같아 '전부' 생략, F1·G1·G3 -> 전부 추가
(f) 기록용 수치: 품목 3단계 aw·pH 가 생장식 정의역 밖인 코드·행 수, Tienungoon pH 항 모양, 일최고·최저기온으로 만든 f(T) 비교(보고만)
    + (검증 발견 반영 2026-10-19) 설정 시트 G2 한계 줄의 코드 목록 = 스칼라 식 기울기로 따로 정한 목록, 글에 적은 행 수(e2fg.NOTE_ROWS) = 다시 센 값,
      aw = 1.00 코드는 G2 결측, 설정 시트에 F 한계·G2 한계 줄이 들어감. (0) 에 뺀 Koutsoumanis 2004 문구 근거(F16·F22·F29) 확인 추가
"""
import hashlib
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
OUT = E.EXP2_DIR / "logs" / "selftest_fg.txt"
SCR = E.EXP2_DIR / "scratch" / "fg"
lines, fails = [], []
NAN = float("nan")


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


def close(a, b, rtol=2e-6, atol=2e-6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=rtol, atol=atol))


# ---------------------------------------------------------------- (0) 엑셀 셀
def t_excel():
    import openpyxl
    wb = openpyxl.load_workbook(G.GROWTH_XLSX, data_only=True)
    ws = wb[G.GROWTH_SHEET]
    bad = []
    for cell, toks in G.GROWTH_CELL_TEXT.items():
        txt = str(ws[cell].value or "").replace("−", "-").replace("–", "-")
        miss = [t for t in toks if t not in txt]
        if miss:
            bad.append((cell, miss))
    check("(0) 생장식 계수 숫자 글자가 엑셀 셀에 있음", not bad,
          f"셀 {len(G.GROWTH_CELL_TEXT)}개, 없는 것 {bad}")
    nm = [c for c in ("F5", "F18", "F23") if "not mentioned" in str(ws[c].value or "")]
    check("(0) 뺀 Wijtes 1993 칸에 'not mentioned'", nm == ["F5", "F18", "F23"], f"{nm}")
    # 뺀 Koutsoumanis 2004: broth a0~a8 이 셀에 다 있고 agar a4 = NS (EXCLUDED_MODELS 문구 근거), 문구의 logit 값 = 엑셀 값 그대로 계산
    kt = {"F16": ["Broth", "-605.935", "1.272", "-0.065"], "F22": ["Broth", "-809.232", "-4.411", "-183.674"],
          "F29": ["Broth", "180.260", "0.507", "-11.460", "NS"]}
    kbad = []
    for cell, toks in kt.items():
        txt = str(ws[cell].value or "").replace("−", "-").replace("–", "-")
        kbad += [(cell, t) for t in toks if t not in txt]
    a = dict(a0=-605.935, a1=1.272, a2=180.260, a3=-809.232, a4=0.507, a5=-4.411, a6=-183.674, a7=-0.065, a8=-11.460)

    def klogit(T, aw, ph):
        bw = math.sqrt(1 - aw)
        return (a["a0"] + a["a1"] * T + a["a2"] * ph + a["a3"] * bw + a["a4"] * T * ph + a["a5"] * T * bw + a["a6"] * ph * bw
                + a["a7"] * T * T + a["a8"] * ph * ph)
    kz = [round(klogit(25, aw, 7.0), 1) for aw in (0.993, 0.98, 0.95)]
    why = dict(G.EXCLUDED_MODELS)["Koutsoumanis 2004 로지스틱 (D16·D22·D29)"]
    check("(0) 뺀 Koutsoumanis 2004: broth a0~a8 셀에 있음, agar a4 'NS', 문구 logit 값 = 엑셀 값 계산",
          not kbad and kz == [-10.3, -137.6, -318.9] and all(f"{z:.1f}".replace("-", "−") in why for z in kz), f"없는 것 {kbad}, logit {kz}")
    for c in ("I6", "I15", "I41", "I44"):
        say(f"    {c}: {str(ws[c].value).splitlines()[0]}")
    say("    J6(비고) 식: " + str(ws["J6"].value).replace("\n", " ")[:400])


# ---------------------------------------------------------------- 참조(스칼라, 엑셀 원문 식)
def ref_silva_T(T):
    """엑셀 D6 온도 항: (T−Tmax)·(T−Tmin)^2/{(Topt−Tmin)·[(Topt−Tmin)·(T−Topt)−(Topt−Tmax)(Topt+Tmin−2T)]}"""
    if T is None or math.isnan(T):
        return NAN
    Tmin, Topt, Tmax = -1.273, 37.26, 45.12              # F6
    if not (Tmin < T < Tmax):
        return 0.0
    return (T - Tmax) * (T - Tmin) ** 2 / ((Topt - Tmin) * ((Topt - Tmin) * (T - Topt) - (Topt - Tmax) * (Topt + Tmin - 2 * T)))


def ref_silva_pH(pH):
    """엑셀 D26 pH 항: (pH−pHmax)·(pH−pHmin)/{[(pHopt−pHmin)·(pH−pHopt)−(pHopt−pHmax)(pHmin−pH)]}"""
    if pH is None or math.isnan(pH):
        return NAN
    pHmin, pHopt, pHmax = 4.303, 7.085, 9.483            # F26
    if not (pHmin < pH < pHmax):
        return 0.0
    return (pH - pHmax) * (pH - pHmin) / ((pHopt - pHmin) * (pH - pHopt) - (pHopt - pHmax) * (pHmin - pH))


def ref_silva_aw(aw):
    """엑셀 D19 aw 항: (aw−1.00)·(aw−awmin)^2/{(awopt−awmin).[(awopt−awmin)·(aw−awopt)−(awopt−1.00)(awopt+awmin−2aw)]}"""
    if aw is None or math.isnan(aw):
        return NAN
    awmin, awopt = 0.894, 0.995                          # F19
    if not (awmin < aw < 1.00):
        return 0.0
    return (aw - 1.00) * (aw - awmin) ** 2 / ((awopt - awmin) * ((awopt - awmin) * (aw - awopt) - (awopt - 1.00) * (awopt + awmin - 2 * aw)))


REF_TIEN = {"scottA": dict(b0=-6.023, b1=19.00, b2=-3.049, b3=7514, Tmin=0.4164, b4=4.635, awmin=0.9142, b5=141.0, b6=240.2, pHmin=3.350),
            "L5": dict(b0=-25.36, b1=44.12, b2=-7.022, b3=10257, Tmin=-1.623, b4=8.951, awmin=0.9152, b5=291.8, b6=704.1, pHmin=3.350)}


def ref_tien_P(T, aw, pH, st):
    """엑셀 D15: ln[P/(1−P)] = b₀+b₁·ln(T−Tmin)+b₂·ln²(T−Tmin)+b₃·ln{1−exp[0.536(T−48)]}+b₄·ln(aw−awmin)
    +b₅·ln(1−10^(pHmin−pH))+b₆·ln²(1−10^(pHmin−pH))+b₇·ln(1−{[LAC]/[UMIC(1+10^(pH−pKa))]}), [LAC]=0. P=1/(1+e^(−logit)).
    로그 안이 0 이하면 생장 없음 P=0. 입력 결측이면 결측."""
    if any(v is None or math.isnan(v) for v in (T, aw, pH)):
        return NAN
    c = REF_TIEN[st]
    try:
        lt = math.log(T - c["Tmin"])
        hi = math.log(1 - math.exp(0.536 * (T - 48)))
        la = math.log(aw - c["awmin"])
        lp = math.log(1 - 10 ** (c["pHmin"] - pH))
    except ValueError:
        return 0.0
    lac = math.log(1 - 0.0)                              # [LAC] = 0 -> 0 (b7 값 불필요)
    z = c["b0"] + c["b1"] * lt + c["b2"] * lt ** 2 + c["b3"] * hi + c["b4"] * la + c["b5"] * lp + c["b6"] * lp ** 2 + 0.0 * lac
    if z < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def ref_ivanek(rain2):
    """엑셀 D41: logit(P) = β₀ + β₁·Precipitation.2, F41: intercept=−1.79, β=0.13."""
    if rain2 is None or math.isnan(rain2):
        return NAN
    return 1.0 / (1.0 + math.exp(-(-1.79 + 0.13 * rain2)))


def ref_summary(vals):
    """기상 요약과 같은 정의를 순수 파이썬으로: 10일 구간 평균(구간1 = 가장 최근) 5, 평균, 표준편차(모집단), 최소, 최대, 기울기."""
    L = len(vals)
    out = []
    for k in range(L // 10):
        seg = [x for x in vals[L - 10 * (k + 1): L - 10 * k] if not math.isnan(x)]
        out.append(sum(seg) / len(seg) if seg else NAN)
    ok = [(i, x) for i, x in enumerate(vals) if not math.isnan(x)]
    if not ok:
        return out + [NAN] * 5
    xs = [x for _, x in ok]
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / n)
    if n >= 2:
        sx = sum(i for i, _ in ok)
        sxx = sum(i * i for i, _ in ok)
        sxy = sum(i * x for i, x in ok)
        sy = sum(xs)
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    else:
        slope = NAN
    return out + [mu, sd, min(xs), max(xs), slope]


def ref_g(vals, stats):
    ok = [x for x in vals if not math.isnan(x)]
    last = [x for x in vals[-10:] if not math.isnan(x)]
    r = []
    for s in stats:
        if not ok:
            r.append(NAN)
        elif s == "mean":
            r.append(sum(ok) / len(ok))
        elif s == "mean10":
            r.append(sum(last) / len(last) if last else NAN)
        elif s == "sum":
            r.append(sum(ok))
        elif s == "npos":
            r.append(float(sum(1 for x in ok if x > 0)))
        elif s == "max":
            r.append(max(ok))
    return r


def _table_dict(path, cols):
    t = pq.read_table(path, columns=["sgg_code", "date"] + cols).to_pandas()
    keys = list(zip(t["sgg_code"].astype(str), pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")))
    vals = t[cols].to_numpy(dtype=np.float64)
    return dict(zip(keys, map(tuple, vals)))


def _day(d, k):
    return str(np.datetime64(d, "D") + np.timedelta64(int(k), "D"))


def pick_pairs(ctx, rng, n_per=100):
    picks = []
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        for j in rng.choice(n, n_per, replace=False):
            picks.append((sp, int(j), int(rng.integers(1, E.H_MAX + 1))))
    # 3단계 코드 없는 단위(학습에만 있음) 몇 개 포함
    nc = np.flatnonzero(ctx.sp["train"]["p3"] == C1.NULL_CODE)
    for j in nc[:3]:
        picks.append(("train", int(j), int(rng.integers(1, E.H_MAX + 1))))
    return picks


# ---------------------------------------------------------------- (a)(b)
def t_values(ctx, rng):
    picks = pick_pairs(ctx, rng)
    say(f"    대조 쌍 {len(picks)}개 (분할마다 100 + 3단계 코드 없는 학습 단위 3)")
    t0 = time.time()
    spi = _table_dict(G.SPI_PATH, G.SPI_VARS)
    soil = _table_dict(G.SOIL_PATH, G.SOIL_VARS)
    wx = _table_dict(C1.WEATHER_PATH, ["avg_temp", "rain"])
    food = pq.read_table(G.FOOD_PATH).to_pandas()
    fmap = {r["train_prdlst_code_2"]: r for r in food.to_dict("records")}
    say(f"    원 parquet 사전 만들기 {time.time() - t0:.1f}s (SPI {len(spi):,}, 토양 {len(soil):,}, 기상 {len(wx):,}, 식품 {len(fmap)})")
    blocks = G.BLOCK_ORDER
    names = G.ext_names(blocks)
    ix = {nm: i for i, nm in enumerate(names)}
    bad = {"spi": 0, "soil": 0, "food": 0, "food_cat": 0, "g1": 0, "g2": 0, "g3": 0, "batch": 0}
    spi_incl_diff = soil_prev_diff = 0
    worst = {}
    by_sp = {}
    for sp, j, h in picks:
        by_sp.setdefault(sp, []).append((j, h))
    batch = {sp: G.ext_features(ctx, sp, np.array([j for j, _ in v]), np.array([h for _, h in v]), blocks) for sp, v in by_sp.items()}
    cnt = {sp: 0 for sp in by_sp}
    for sp, j, h in picks:
        v = ctx.sp[sp]
        row = ctx.rows.iloc[v["idx"][j]]
        sgg = str(row["sgg_code"])
        d = np.datetime64(row["d"], "D")
        T = d - np.timedelta64(h, "D")
        got = G.ext_features(ctx, sp, np.array([j]), h, blocks)[0].astype(np.float64)
        gb = batch[sp][cnt[sp]].astype(np.float64)
        cnt[sp] += 1
        bad["batch"] += int(not np.array_equal(got, gb, equal_nan=True))
        # SPI
        days = [_day(T, k) for k in range(-E.L, 0)]
        exp = []
        for vi in range(len(G.SPI_VARS)):
            exp += ref_summary([spi[(sgg, dd)][vi] for dd in days])
        g = got[[ix[n] for n in G.summary_names(G.SPI_VARS)]]
        if not close(g, exp):
            bad["spi"] += 1
            worst.setdefault("spi", (sp, j, h, list(g), exp))
        days_incl = [_day(T, k) for k in range(-E.L + 1, 1)]
        inc = []
        for vi in range(len(G.SPI_VARS)):
            inc += ref_summary([spi[(sgg, dd)][vi] for dd in days_incl])
        spi_incl_diff += int(not close(g, inc))
        # 토양
        e_soil = soil[(sgg, str(T))]
        g_soil = got[[ix[n] for n in G.SOIL_VARS]]
        if not np.array_equal(np.float32(e_soil), np.float32(g_soil), equal_nan=True):
            bad["soil"] += 1
            worst.setdefault("soil", (sp, j, h, list(g_soil), e_soil))
        soil_prev_diff += int(not np.array_equal(np.float32(soil[(sgg, _day(T, -1))]), np.float32(g_soil), equal_nan=True))
        # 식품
        code = str(row["train_prdlst_code_2"])
        r = fmap.get(code)
        e_food = [NAN] * 3 if r is None else [NAN if r[c] is None or pd.isna(r[c]) else float(r[c]) for c in G.FOOD_NUM]
        e_lab = G.FOOD_MISSING if r is None or r["aw_label"] is None or pd.isna(r["aw_label"]) else str(r["aw_label"])
        g_food = got[[ix[f"food_{c}"] for c in G.FOOD_NUM]]
        if not np.array_equal(np.float32(e_food), np.float32(g_food), equal_nan=True):
            bad["food"] += 1
            worst.setdefault("food", (sp, j, h, list(g_food), e_food))
        g_lab = G.ext_cats(ctx, sp, np.array([j]), blocks)[0, 0]
        bad["food_cat"] += int(g_lab != e_lab)
        # G: 기온·강수 parquet 에서 날마다
        temps = [wx[(sgg, dd)][0] for dd in days]
        fT = [ref_silva_T(x) for x in temps]
        e_g1 = ref_g(fT, G.G_STATS["g1"])
        if not close(got[[ix[f"g1_fT_{s}"] for s in G.G_STATS["g1"]]], e_g1):
            bad["g1"] += 1
            worst.setdefault("g1", (sp, j, h, e_g1))
        aw, ph = e_food[0], e_food[1]
        gf = ref_silva_pH(ph) * ref_silva_aw(aw)
        mu = [0.409 * gf * x for x in fT]                 # µopt = 0.409 (J6)
        e_g2 = ref_g(mu, G.G_STATS["g2"])
        for st in G.TIEN_STRAINS:
            e_g2 += ref_g([ref_tien_P(x, aw, ph, st) for x in temps], G.G_STATS["g2"])
        g_g2 = got[[ix[nm] for nm in names if nm.startswith("g2_")]]
        if not close(g_g2, e_g2, rtol=1e-5, atol=1e-6):
            bad["g2"] += 1
            worst.setdefault("g2", (sp, j, h, list(g_g2), e_g2))
        rain2 = [wx[(sgg, _day(T, k - 2))][1] for k in range(-E.L, 0)]   # 날 t 의 Precipitation.2 = rain(t−2)
        e_g3 = ref_g([ref_ivanek(x) for x in rain2], G.G_STATS["g3"])
        if not close(got[[ix[f"g3_P_ivanek_{s}"] for s in G.G_STATS["g3"]]], e_g3):
            bad["g3"] += 1
            worst.setdefault("g3", (sp, j, h, e_g3))
    n = len(picks)
    check("(a) F1 SPI 20칸 = 원 parquet 날마다 찾아 순수 파이썬 요약(창 T−50..T−1)", bad["spi"] == 0, f"불일치 {bad['spi']}/{n}")
    check("(a) F1 SPI 창은 T 당일 제외: 'T 포함(T−49..T)' 으로 만들면 달라지는 쌍이 있음(그래도 위 대조는 일치)", spi_incl_diff > 0,
          f"{spi_incl_diff}/{n} 쌍이 달라짐")
    check("(a) F2 토양 6칸 = 원 parquet (sgg_code, 날짜 T) 행", bad["soil"] == 0, f"불일치 {bad['soil']}/{n}")
    check("(a) F2 토양은 날짜 T 행: T−1 행으로 하면 달라지는 쌍이 있음", soil_prev_diff > 0, f"{soil_prev_diff}/{n} 쌍이 달라짐")
    check("(a) F3 식품 aw·ph·na = 원 parquet 3단계 코드 행(코드 없음·표에 없음 = 결측)", bad["food"] == 0, f"불일치 {bad['food']}/{n}")
    check("(a) F3 aw_label = 원 parquet 표기('결측' 포함)", bad["food_cat"] == 0, f"불일치 {bad['food_cat']}/{n}")
    check("(a) 한 쌍씩(h 정수) = 한꺼번에(h 배열, 학습 경로)", bad["batch"] == 0, f"불일치 {bad['batch']}/{n}")
    check("(b) G1 4칸 = 엑셀 D6 식 스칼라 함수 + 기상 parquet 일평균기온", bad["g1"] == 0, f"불일치 {bad['g1']}/{n}")
    check("(b) G2 9칸 = 엑셀 D6·D19·D26(µmax)·D15(P, 2균주) 식 스칼라 함수", bad["g2"] == 0, f"불일치 {bad['g2']}/{n}")
    check("(b) G3 3칸 = 엑셀 D41 식 스칼라 함수 + 기상 parquet rain(t−2)", bad["g3"] == 0, f"불일치 {bad['g3']}/{n}")
    for k, w in worst.items():
        say(f"    [첫 불일치 {k}] {w}")


def t_edges():
    S = G.SILVA
    T_list = [-30, S["Tmin"] - 1e-9, S["Tmin"], S["Tmin"] + 1e-9, -1.0, 0.0, 0.4164, 0.41641, -1.623, -1.62299, 5, 20, S["Topt"],
              S["Tmax"] - 1e-9, S["Tmax"], S["Tmax"] + 1e-9, 46, 47.99, 47.999999, 48.0, 48.0001, 55, NAN]
    T_grid = list(np.round(np.arange(-20, 60, 0.01), 4)) + T_list
    aw_list = [0.0, 0.5, S["awmin"] - 1e-9, S["awmin"], S["awmin"] + 1e-9, 0.9142, 0.91421, 0.9152, 0.95, S["awopt"], 0.999999, 1.0, 1.0 + 1e-9, 1.1, NAN]
    aw_grid = list(np.round(np.arange(0.80, 1.05, 0.0005), 5)) + aw_list
    ph_list = [2.0, 3.35, 3.350001, 3.5, 3.945, 4.0, S["pHmin"] - 1e-9, S["pHmin"], S["pHmin"] + 1e-9, 6.0, S["pHopt"], S["pHmax"] - 1e-9, S["pHmax"], 10.0, NAN]
    ph_grid = list(np.round(np.arange(2.0, 11.0, 0.005), 4)) + ph_list
    v_T = G.silva_fT(np.array(T_grid))
    r_T = np.array([ref_silva_T(x) for x in T_grid])
    v_p = G.silva_fpH(np.array(ph_grid))
    r_p = np.array([ref_silva_pH(x) for x in ph_grid])
    v_a = G.silva_faw(np.array(aw_grid))
    r_a = np.array([ref_silva_aw(x) for x in aw_grid])
    ok = close(v_T, r_T, 1e-12, 1e-14) and close(v_p, r_p, 1e-12, 1e-14) and close(v_a, r_a, 1e-12, 1e-14)
    check("(b) Silva f(T)·f(pH)·f(aw) 벡터 식 = 스칼라 식 (경계·범위 밖 0·결측 포함)", ok,
          f"T {len(T_grid)}점, pH {len(ph_grid)}점, aw {len(aw_grid)}점")
    one = (abs(float(G.silva_fT(S["Topt"])) - 1) < 1e-12, abs(float(G.silva_fpH(S["pHopt"])) - 1) < 1e-12, abs(float(G.silva_faw(S["awopt"])) - 1) < 1e-12)
    rng01 = all(np.nanmax(x) <= 1 + 1e-12 and np.nanmin(x) >= 0 for x in (v_T, v_p, v_a))
    check("(b) 카디널 항: 최적값에서 1, 전 구간 0 ≤ f ≤ 1", all(one) and rng01,
          f"최적값 1: {one}, 최대 f(T) {np.nanmax(v_T):.6f} (T={T_grid[int(np.nanargmax(v_T))]}), 최대 f(pH) {np.nanmax(v_p):.6f}, 최대 f(aw) {np.nanmax(v_a):.6f}")
    zeros = {"T≤Tmin": float(G.silva_fT(S["Tmin"])), "T≥Tmax": float(G.silva_fT(S["Tmax"])), "aw=1.00": float(G.silva_faw(1.0)),
             "aw≤awmin": float(G.silva_faw(S["awmin"])), "pH≤pHmin": float(G.silva_fpH(S["pHmin"])), "pH≥pHmax": float(G.silva_fpH(S["pHmax"]))}
    check("(b) Silva 정의역 밖·경계 = 0 (생장 없음)", all(v == 0.0 for v in zeros.values()), f"{zeros}")
    # Tienungoon: T x aw x pH 조합
    rng = np.random.default_rng(7)
    Ts = np.array(T_list + list(rng.uniform(-10, 55, 300)))
    aws = np.array(aw_list + list(rng.uniform(0.85, 1.02, 300)))
    phs = np.array(ph_list + list(rng.uniform(2.5, 10, 300)))
    nb, nz, nn = 0, 0, 0
    for st in G.TIEN_STRAINS:
        for T in Ts:
            A = G.tien_A(np.array([T]), st)
            Bv = G.tien_B(aws, phs, st)
            pv = G.sigmoid(A + Bv)
            pr = np.array([ref_tien_P(float(T), float(a), float(p), st) for a, p in zip(aws, phs)])
            if not close(pv, pr, 1e-9, 1e-12):
                nb += 1
            nz += int(np.sum(pr == 0))
            nn += int(np.sum(np.isnan(pr)))
    check("(b) Tienungoon P 벡터 식(온도 부분 + 식품 부분) = D15 스칼라 식 (로그 안 0 이하 -> P 0, 결측 -> 결측)", nb == 0,
          f"균주 2 x T {len(Ts)} x (aw,pH) {len(aws)}: 불일치 T {nb}개, P=0 {nz}개, 결측 {nn}개")
    z = {"T=Tmin(ScottA)": float(G.sigmoid(G.tien_A(np.array([0.4164]), "scottA") + 0.0)[0]),
         "T=48": float(G.sigmoid(G.tien_A(np.array([48.0]), "scottA") + 0.0)[0]),
         "aw=awmin": float(G.sigmoid(G.tien_B(np.array([0.9142]), np.array([6.0]), "scottA"))[0]),
         "pH=pHmin": float(G.sigmoid(G.tien_B(np.array([0.99]), np.array([3.35]), "scottA"))[0]),
         "aw 결측": float(G.sigmoid(G.tien_B(np.array([NAN]), np.array([6.0]), "scottA"))[0])}
    check("(b) Tienungoon 경계: T=Tmin·T=48·aw=awmin·pH=pHmin -> P 0, aw 결측 -> 결측",
          z["T=Tmin(ScottA)"] == 0 and z["T=48"] == 0 and z["aw=awmin"] == 0 and z["pH=pHmin"] == 0 and math.isnan(z["aw 결측"]), f"{z}")
    rr = np.array([0.0, 1.0, 10.0, 50.0, 400.0, NAN])
    check("(b) Ivanek P 벡터 = 스칼라", close(G.ivanek_P(rr), [ref_ivanek(x) for x in rr], 1e-12, 1e-15),
          f"P(0mm)={float(G.ivanek_P(0.0)):.5f}, P(10mm)={float(G.ivanek_P(10.0)):.5f}")
    # 기록: Tienungoon pH 항 모양
    for st in G.TIEN_STRAINS:
        c = G.TIEN[st]
        vals = {p: float(G.tien_B(np.array([0.99]), np.array([p]), st)[0]) for p in (3.4, 3.5, 3.7, 3.945, 4.2, 4.5, 5.0, 6.0, 7.0)}
        umin = -c["b5"] / (2 * c["b6"])
        pmin = c["pHmin"] - math.log10(1 - math.exp(umin))
        say(f"    [기록] Tienungoon {st} 식품 부분 B(aw=0.99, pH): " + ", ".join(f"pH {p}: {v:.2f}" for p, v in vals.items())
            + f" -> b5·u + b6·u² 최소는 pH {pmin:.3f}; 그보다 낮은 pH 에서 다시 커짐(식 그대로 계산, 적용범위 pH 3.4~7.4 - J15)")


# ---------------------------------------------------------------- (c) E00 재현
def _cache_meta_ok(ctx):
    nw = 190
    for sp in ("val", "test"):
        v = ctx.sp[sp]
        key = hashlib.md5(np.ascontiguousarray(np.stack([v["s"], v["d"]])).tobytes()).hexdigest()
        meta = {"n": int(len(v["s"])), "n_weather": nw, "rows_md5": key, "h_max": E.H_MAX,
                "weather_shape": list(ctx.weather.W.shape), "weather_date0": str(ctx.weather.date0)}
        mf = E.EXP2_DIR / "cache" / f"eval_wsum_L{E.L:03d}_{sp}.json"
        if not mf.exists() or json.loads(mf.read_text(encoding="utf-8")) != meta:
            return False
    return True


def t_e00(ctx, log):
    import shutil
    from run import check_windows, run_one
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
    check("(c) E00 config: 입력 칸 이름·개수·설명 = 기존 results/E00__s42, F·G 기록 칸 없음",
          c2["feature_columns"] == c3["feature_columns"] and c2["n_features"] == c3["n_features"]
          and c2["n_features_desc"] == c3["n_features_desc"] and "ext" not in c2,
          f"{c2['n_features_desc']}")
    say(f"    E00 재현 메모리: private {s.get('peak_private_gb')} GB / rss {s.get('peak_rss_gb')} GB")


# ---------------------------------------------------------------- (d) 앞날 자료
def t_future(ctx, rng):
    X = G.get_ext(ctx)
    sp = "val"
    v = ctx.sp[sp]
    n = len(v["idx"])
    pos = rng.choice(n, 400, replace=False)
    h = rng.integers(1, E.H_MAX + 1, size=400)
    T = v["d"][pos] - h
    base = G.ext_features(ctx, sp, pos, h, G.BLOCK_ORDER)
    saved = {k: getattr(X, k) for k in ("SPI", "SOIL", "fT", "Q")}
    savedA = dict(X.A)
    try:
        # 행마다 T 가 달라 행 하나씩: 그 행 시군구의 T 이후(T 포함) 칸을 무작위로
        nbad = 0
        for i in range(len(pos)):
            s, t = int(v["s"][pos[i]]), int(T[i])
            X.SPI = saved["SPI"].copy()
            X.SPI[s, t:] = rng.normal(size=X.SPI[s, t:].shape)
            X.fT = saved["fT"].copy()
            X.fT[s, t:] = rng.uniform(size=X.fT[s, t:].shape)
            X.Q = saved["Q"].copy()
            X.Q[s, t:] = rng.uniform(size=X.Q[s, t:].shape)
            X.A = {k: a.copy() for k, a in savedA.items()}
            for k in X.A:
                X.A[k][s, t:] = rng.normal(size=X.A[k][s, t:].shape)
            X.SOIL = saved["SOIL"].copy()
            X.SOIL[s, t + 1:] = rng.normal(size=X.SOIL[s, t + 1:].shape)      # 토양은 날짜 T 행(값 = T 전날까지 기록)을 씀
            g = G.ext_features(ctx, sp, pos[i:i + 1], int(h[i]), G.BLOCK_ORDER)
            nbad += int(not np.array_equal(g, base[i:i + 1], equal_nan=True))
            if i >= 60:
                break
    finally:
        for k, a in saved.items():
            setattr(X, k, a)
        X.A = savedA
    check("(d) T 이후 칸(SPI·f(T)·P 온도 부분·G3 는 T 당일 포함, 토양은 T+1 부터)을 무작위로 바꿔도 F·G 값 같음", nbad == 0, f"불일치 {nbad}/61")
    W = ctx.weather.W
    iR = C1.WEATHER_VARS.index("rain")
    ss = rng.integers(0, W.shape[0], 500)
    tt = rng.integers(2, W.shape[1], 500)
    q = X.Q[ss, tt]
    ref = np.array([ref_ivanek(float(W[s, t - 2, iR])) for s, t in zip(ss, tt)])
    check("(d) G3 날 t 값 = Ivanek P(rain(t−2)) (2일 전 강수)", close(q, ref, 1e-12, 1e-15), "500칸")


# ---------------------------------------------------------------- (e) 조합 계획
def t_combo():
    cases = {(): ["A3_F4", "A3_G1G2G3"], ("F2",): ["A3_F2"], ("F1", "F4"): ["A3_F1", "A3_F4"],
             ("F1", "G1", "G3"): ["A3_F1", "A3_G1", "A3_G3", "A3_F1G1G3"], ("F1", "F2", "F3"): ["A3_F1", "A3_F2", "A3_F3", "A3_F1F2F3"]}
    bad = []
    for sel, want in cases.items():
        got = [c for c, _, _ in G.combo_plan(list(sel))]
        if got != want:
            bad.append((sel, got, want))
    kinds = [k for _, _, k in G.combo_plan([])]
    blocks = G.blocks_of(["F1", "G1", "G3"])
    check("(e) 조합 계획 규칙(없음 -> 대체 2개 시드 1개, 하나 -> 각각만, 합집합이 한 묶음과 같으면 '전부' 생략)",
          not bad and kinds == ["fallback", "fallback"] and blocks == ["spi", "g1", "g3"], f"불일치 {bad}")
    check("(e) 조합 ID 판별", G.is_combo_id("A3_F1G1G3") and G.is_combo_id("A3_G1G2G3") and not G.is_combo_id("A3")
          and not G.is_combo_id("F1") and not G.is_combo_id("A3_F5"))
    s3 = G.combo_plan(["F3"])
    check("(e) 조합 입력 = A3 이력 + 묶음 블록", G.blocks_of(s3[0][1]) == ["food"])


# ---------------------------------------------------------------- (f) 기록
def t_record(ctx):
    X = G.get_ext(ctx)
    f = X.food_table
    aw, ph = f["aw"].to_numpy(dtype=float), f["ph"].to_numpy(dtype=float)
    say("    [기록] 품목 3단계 표 217개 코드: "
        f"aw 결측 {int(np.isnan(aw).sum())}, ph 결측 {int(np.isnan(ph).sum())}, aw·ph 둘 다 있음 {int((~np.isnan(aw) & ~np.isnan(ph)).sum())} | "
        f"Silva: aw ≤ 0.894 {int((aw <= 0.894).sum())}, aw = 1.00 {int((aw >= 1.0).sum())}, pH ≤ 4.303 {int((ph <= 4.303).sum())} | "
        f"Tienungoon: aw ≤ 0.9142 {int((aw <= 0.9142).sum())}, aw ≤ 0.9152 {int((aw <= 0.9152).sum())}, pH < 3.945(식 모양상 P 가 다시 커지는 쪽) {int((ph < 3.945).sum())}")
    # 설정 시트 G2 한계 줄의 코드 목록: e2fg.food_domain_facts 를 스칼라 식(엑셀 D15 pH 항)의 기울기로 따로 정한 목록과 대조
    fx = G.food_domain_facts(f)
    ref_rise, ref_zero = set(), set()
    for r in f.itertuples(index=False):
        if pd.isna(r.aw) or pd.isna(r.ph):
            continue
        for st, c in REF_TIEN.items():
            def bph(x):
                return c["b5"] * math.log(1 - 10 ** (c["pHmin"] - x)) + c["b6"] * math.log(1 - 10 ** (c["pHmin"] - x)) ** 2
            if r.ph <= c["pHmin"]:
                side, finite = (bph(c["pHmin"] + 1e-9) > bph(c["pHmin"] + 2e-9)), False      # pHmin 이하: 정의역 밖
            else:
                side, finite = bph(r.ph) > bph(r.ph + 1e-6), r.aw > c["awmin"]               # pH 를 낮추면 pH 항이 커지는 쪽
            if side and finite:
                ref_rise.add(r.train_prdlst_code_2)
            elif side:
                ref_zero.add(r.train_prdlst_code_2)
    ref_zero -= ref_rise
    got_rise = set(fx["rising"]["train_prdlst_code_2"])
    got_zero = set(fx["rising_zero"]["train_prdlst_code_2"])
    check("(f) G2 한계 줄 코드 목록 = 스칼라 식 기울기로 정한 목록(P 가 pH 항 때문에 커질 수 있음 / 같은 쪽이지만 정의역 밖 P 0)",
          got_rise == ref_rise and got_zero == ref_zero,
          f"커질 수 있음 {len(got_rise)}개 {sorted(fx['rising']['train_prdlst_nm_2'])}, 정의역 밖 {len(got_zero)}개 {sorted(fx['rising_zero']['train_prdlst_nm_2'])}, "
          f"pH 항 최소 pH {({k: round(v, 4) for k, v in fx['pmin'].items()})}")
    rise_codes = list(got_rise)
    ineq_codes = list(fx["ineq_with_ph"]["train_prdlst_code_2"])
    cnt = {"rising": [], "rising_pos": [], "ineq_with_ph": [], "aw1_g2_all_nan": [], "aw1": []}
    for sp in C1.SPLITS:
        num = X.food_num[sp]
        g = X.gfood[sp]
        ok = ~np.isnan(g)
        b = X.B[sp]["scottA"]
        p3 = ctx.sp[sp]["p3"]
        mr = np.isin(p3, rise_codes)
        mi = np.isin(p3, ineq_codes)
        g2 = np.isnan(G.ext_features(ctx, sp, np.flatnonzero(num[:, 0] >= 1), 1, ["g2"])) if (num[:, 0] >= 1).any() else np.zeros((0, 9), bool)
        cnt["rising"].append(int(mr.sum()))
        cnt["rising_pos"].append(int(ctx.sp[sp]["y"][mr].sum()))
        cnt["ineq_with_ph"].append(int(mi.sum()))
        cnt["aw1_g2_all_nan"].append(int(g2.all(1).sum()))
        cnt["aw1"].append(int((num[:, 0] >= 1).sum()))
        say(f"    [기록] {sp} 단위x날짜 행 {len(num):,}: aw·ph 둘 다 있음 {int(ok.sum()):,} ({ok.mean():.1%}) 중 Silva f(pH)·f(aw)=0 (µmax 항상 0) {int((g[ok] == 0).sum()):,}, "
            f"Tienungoon 식품 부분 −inf(P 항상 0) Scott A {int(np.isneginf(b).sum()):,}, L5 {int(np.isneginf(X.B[sp]['L5']).sum()):,}; "
            f"aw=1.00 행 {int((num[:, 0] >= 1).sum()):,}(G2 9칸 전부 결측 {int(g2.all(1).sum()):,}), pH 항 때문에 P 가 커질 수 있는 코드 행 {int(mr.sum()):,}"
            f"(검출 {int(ctx.sp[sp]['y'][mr].sum())}), 부등호 aw·ph 있음 행 {int(mi.sum()):,}, 3단계 코드 없음 {X.food_stats[sp]['no_code_rows']}")
    check("(f) G2 한계 줄 행 수(e2fg.NOTE_ROWS) = 다시 센 값, aw = 1.00 코드는 ph 가 모두 결측 -> G2 9칸 전부 결측(µmax 0 아님)",
          all(tuple(cnt[k]) == tuple(G.NOTE_ROWS[k]) for k in G.NOTE_ROWS) and len(fx["aw1_with_ph"]) == 0 and cnt["aw1_g2_all_nan"] == cnt["aw1"],
          f"{cnt}, 부등호 aw 코드 {len(fx['ineq'])}개 중 ph 있음 {len(fx['ineq_with_ph'])}")
    try:
        import e2tables
        rows = e2tables.fg_settings_rows({"note": "selftest"})
        keys = [k for k, _ in rows]
        need = [k for k, _ in G.F_NOTES] + [k for k, _ in G.g_notes()] + ["G 한계"]
        check("(f) 설정 시트 F·G 줄에 F 한계 5줄·G2 한계 3줄·G 한계가 들어감", all(k in keys for k in need), f"줄 {len(rows)}개")
    except Exception as ex:
        check("(f) 설정 시트 F·G 줄 만들기", False, repr(ex))
    W = ctx.weather.W
    d0 = int((np.datetime64("2015-01-01") - ctx.weather.date0).astype(int))
    d1 = int((np.datetime64(C1.TEST_END) - ctx.weather.date0).astype(int))
    msg = []
    for v in ("avg_temp", "max_temp", "min_temp"):
        T = W[:, d0:d1 + 1, C1.WEATHER_VARS.index(v)].astype(np.float64)
        fT = G.silva_fT(T)
        msg.append(f"{v}: f>0 인 날 {np.nanmean(fT > 0):.1%}, f 평균 {np.nanmean(fT):.4f}, 결측 {np.isnan(T).mean():.2%}")
    say("    [기록, 보고만] G1 을 일최고·최저기온으로 만들면(시군구 x 날짜 2015-01-01~2026-06-30): " + " | ".join(msg))
    Tm = W[:, d0:d1 + 1, C1.WEATHER_VARS.index("avg_temp")].astype(np.float64)
    say(f"    [기록] 일평균기온 범위 {np.nanmin(Tm):.2f} ~ {np.nanmax(Tm):.2f} °C (Silva Tmax 45.12·Tienungoon 48 넘는 날 없음), "
        f"Tienungoon 적용범위 3~35℃ 밖인 날 {np.nanmean((Tm < 3) | (Tm > 35)):.1%}")
    R = W[:, d0:d1 + 1, C1.WEATHER_VARS.index("rain")]
    S = W[:, d0:d1 + 1, C1.WEATHER_VARS.index("solar_rad")]
    say(f"    [기록] rain(mm) 0 인 날 {np.nanmean(R == 0):.1%}, 결측 {np.isnan(R).mean():.2%} | solar_rad(MJ/m², 일합) > 0.4 인 날(결측 제외) {np.nanmean(S[~np.isnan(S)] > 0.4):.4%}")


def main():
    t0 = time.time()
    log = Logger(E.EXP2_DIR / "logs" / "progress_selftest_fg.log")
    rng = np.random.default_rng(20261019)
    t_excel()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    G.get_ext(ctx)
    t_values(ctx, rng)
    t_edges()
    t_future(ctx, rng)
    t_combo()
    t_record(ctx)
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
