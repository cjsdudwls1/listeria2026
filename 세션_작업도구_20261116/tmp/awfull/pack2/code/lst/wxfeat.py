"""기상 파생변수 WX3~WX6(단계 0-6) — 계획서 §5 표, §4-1, 부록 A-3.

- WX3 생장식 gr1·gr2·gr3 16칼럼: window [S, E](E 끝, W 일), θ = 날마다 avg_temp.
    gr1 = γ_T(θ) 요약 4(mean·mean10·sum·npos)
    gr2 = µmax = µopt·γ_T·γ_pH·γ_aw, P_scottA, P_L5(Tienungoon 2000) 각각 요약 3(mean·mean10·max) = 9
    gr3 = 날 t 의 P = 1 / (1 + e^(−(−1.79 + 0.13·rain[t − 2]))) 요약 3(mean·mean10·max)
- WX4 생장 범위: 기상 6 × (window 평균·최근 10일 평균) = 12칼럼, 토양 pH 3칼럼(SO1 soil_ph 로).
  (품목 4칼럼 rg_aw_lim 등은 새 식품 고정표 담당 — 여기 없음)
- WX5 단기 기상 6칼럼·WX6 rh1·td1: 기준점 R = min(d, E + 1), R − 1 끝, W 와 별개로 자기 길이.
- (추가 판 — 사용자 결정 2026-11-12, 계획서 밖) KO1 Koutsoumanis et al. (2004) 생육확률 6칼럼(`ko_features`):
    Logit(P) = a0 + a1·T + a2·pH + a3·bw + a4·T·pH + a5·T·bw + a6·pH·bw + a7·T² + a8·pH², bw = sqrt(1 − aw),
    P = 1/(1 + exp(−Logit)), 액체배지(broth)·고체배지(agar) 계수 = 식품 연구원 자료(리스테리아 모델적용방안 실험용.pdf
    1쪽) 그대로. T = 날마다 avg_temp(θ, gr 과 같은 window [S, E]), aw·pH = 식품 고정표. 날마다 P → 요약 3(mean·mean10·
    max). 결측 = gr2 와 같음(그날 θ·aw·pH 중 하나라도 결측 → 그날 결측, 값 있는 날 0 → 결측). 구현 세부(계획서에
    없음): exp 넘침만 막음(z ≥ 0 이면 1/(1 + e^(−z)), z < 0 이면 e^z/(1 + e^z) — 같은 값), aw > 1 이면 bw = sqrt(음수)
    = NaN 그대로(값을 바꾸지 않음), 칼럼 이름 gr4_ko_<broth|agar>_<mean|mean10|max>.

window·E·S·R 은 `lst.windows`(window_bounds·DailyCube·gather_window·anomaly_summary_block)를 그대로 씀.
aw·pH(식품 고정표 FD1)·soil_ph(SO1)는 다른 단계가 만든 값을 행마다 배열로 받기만 함.

결측(부록 A-3): 식마다 자기 입력만 봄 — gr1 = θ, gr2 = θ·aw·pH, gr3 = rain[t − 2] 가 그날 하나라도 결측이면
그날 값 결측. 요약은 결측인 날을 빼고(sum = 값 있는 날 합, npos = 값 있는 날 중 γ_T > 0 인 날 수), 값 있는 날이
0 이면(mean10 은 최근 10일 안에서) 결측. WX4 = 경계 포함, 입력 결측이면 결측(rg_t_out_lim 은 둘 중 하나라도).
"""
from __future__ import annotations

import dataclasses
import re
import unicodedata
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import windows as wn

# ---------------------------------------------------------------- 일자료 변수
WEATHER19 = ("avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "max_wind", "sunshine", "solar_rad",
             "soil_t5", "soil_t10", "soil_t20", "soil_t30", "soil_t_nas", "soil_moist",
             "no2", "co", "so2", "wind_dir_sin", "wind_dir_cos")       # §4-3
HUMID = ("rh", "td")                                                    # 습도·이슬점(부록 A-2, WX6 만 씀)
CUBE_VARS = WEATHER19 + HUMID
OVERSEAS_RENAME = {"avg_rhm": "rh", "avg_td": "td"}                     # 해외 표 이름 → 국내 이름
OVERSEAS_PREFIX = "OV"                                                  # §2-3 해외 sgg = 'OV' + alpha-3
VALUE_START = wn.DOMESTIC_WEATHER_START                                 # §4-3 값은 2015-01-01 부터

# ---------------------------------------------------------------- 생장식 계수(부록 A-3 문구 그대로)
SILVA = {"Tmin": -1.273, "Topt": 37.26, "Tmax": 45.12,      # F6
         "pHmin": 4.303, "pHopt": 7.085, "pHmax": 9.483,     # F26
         "awmin": 0.894, "awopt": 0.995, "awmax": 1.00,      # F19, awmax = D6 식 안
         "mu_opt": 0.409}                                     # J6
TIEN = {"scottA": {"b0": -6.023, "b1": 19.00, "b2": -3.049, "b3": 7514.0, "Tmin": 0.4164,
                   "b4": 4.635, "awmin": 0.9142, "b5": 141.0, "b6": 240.2, "pHmin": 3.350},
        "L5": {"b0": -25.36, "b1": 44.12, "b2": -7.022, "b3": 10257.0, "Tmin": -1.623,
               "b4": 8.951, "awmin": 0.9152, "b5": 291.8, "b6": 704.1, "pHmin": 3.350}}
TIEN_FIXED = {"d": 0.536, "Tmax": 48.0}                      # D15·J15
IVANEK = {"b0": -1.79, "b1": 0.13, "lag": 2}                 # F41
GROWTH_SHEET = "(완료) 리스테리아 모노사이토제네스"

# ---------------------------------------------------------------- 칼럼 이름
# 구현 세부(계획서에 없음): 계획서는 'gr1·gr2·gr3 16칼럼' 만 적음 → 이름 꼴 아래처럼 정함.
GR1_STATS = ("mean", "mean10", "sum", "npos")
GR_STATS = ("mean", "mean10", "max")
GR2_PARTS = ("mu", "pscotta", "pl5")
WX3_NAMES = ([f"gr1_{s}" for s in GR1_STATS]
             + [f"gr2_{p}_{s}" for p in GR2_PARTS for s in GR_STATS]
             + [f"gr3_{s}" for s in GR_STATS])
RG_WX_DAILY = ("rg_tmax_opt", "rg_tavg_opt_dist", "rg_t_out_lim", "rg_soilt_opt", "rg_soilt_lim", "rg_soilt_opt_dist")
RG_WX_STATS = ("mean", "mean10")
# 구현 세부(계획서에 없음): 기상 12칼럼 이름 = `<생장범위_정의.csv 칸 이름>_mean`·`_mean10`.
WX4_WX_NAMES = [f"{v}_{s}" for v in RG_WX_DAILY for s in RG_WX_STATS]
WX4_SOIL_NAMES = ["rg_soilph_opt", "rg_soilph_lim", "rg_soilph_opt_dist"]   # 정의 CSV 칸 이름 그대로
WX5_NAMES = ["t3", "rain_lag2", "wind_lag2", "rain_mean25", "solar_23", "trange7"]  # §5 표 이름 그대로
WX6_NAMES = ["rh1", "td1"]
# KO1(추가 판): Koutsoumanis et al. (2004) a0~a8 — 연구원 자료 그대로(고치지 않음)
KOUTSOUMANIS = {"broth": (-605.935, 1.272, 180.260, -809.232, 0.507, -4.411, -183.674, -0.065, -11.460),
                "agar": (-516.908, 3.218, 130.741, -957.558, 0.0, -7.574, -186.298, -0.031, -6.687)}
KO_MEDIA = ("broth", "agar")
KO_NAMES = [f"gr4_ko_{m}_{s}" for m in KO_MEDIA for s in GR_STATS]
GROUP_NAMES = {"WX3": WX3_NAMES, "WX4": WX4_WX_NAMES + WX4_SOIL_NAMES, "WX5": WX5_NAMES, "WX6": WX6_NAMES}
GROUPS = tuple(GROUP_NAMES)
SHORT_LEN = 25        # WX5 가장 긴 구간 R−25 ~ R−1
LAST = wn.BIN         # '최근 10일' = E−9 ~ E

# 생장 범위 경계(생장범위_정의.csv 기본값, 경계 포함)
RG = {"tmax_opt": (30.0, 37.0), "tavg_opt": (30.0, 37.0), "t_lim": (-2.0, 45.0),
      "soilt_opt": (8.0, 15.0), "soilt_lim": (8.0, 30.0),
      "soilph_opt": (6.5, 7.0), "soilph_lim": (5.0, 8.0)}


# ---------------------------------------------------------------- 일자료 표 합치기
def combined_cube(domestic: pd.DataFrame, overseas: pd.DataFrame | None, variables: Sequence[str] = CUBE_VARS,
                  units: Sequence[str] | None = None) -> wn.DailyCube:
    """국내 표(sgg_code·date·19변수·rh·td) + 해외 표(sgg = 'OV' + alpha_3·date·19변수·avg_rhm·avg_td) → DailyCube 하나.

    - 해외 avg_rhm·avg_td → rh·td 로 이름만 바꿈(값 그대로). 해외 alpha_3 칼럼이 있으면 sgg == 'OV' + alpha_3 확인.
    - 단위 코드가 겹치거나, 국내 코드가 'OV' 로 시작하거나, 2015-01-01 전 날에 값이 있으면 멈춤(§4-3 — 채우거나
      지우지 않음, 표가 규칙과 다르면 알림).
    - units: 주면 그 단위만(메모리) — 표에 없는 코드가 있으면 멈춤.
    구현 세부(계획서에 없음): 두 표의 날짜 범위가 달라(국내 2014-04-07 ~ 2026-10-13, 해외 2015-01-01 ~ 2026-06-30)
    합친 날짜 축 = 두 표 합집합, 표에 없는 (단위, 날짜) = NaN(DailyCube.from_long 규칙).
    """
    variables = list(variables)
    dom = domestic.rename(columns={"sgg_code": "_unit"})
    need = ["_unit", "date"] + variables
    miss = [c for c in need if c not in dom.columns]
    if miss:
        raise KeyError(f"국내 표에 없는 칼럼: {miss}")
    parts = [dom[need]]
    if overseas is not None:
        ov = overseas.rename(columns={**OVERSEAS_RENAME, "sgg": "_unit"})
        if any(c in overseas.columns for c in OVERSEAS_RENAME.values()):
            raise ValueError("해외 표에 이미 rh·td 칼럼이 있음 — 이름 맞추기 규칙과 다름")
        miss = [c for c in need if c not in ov.columns]
        if miss:
            raise KeyError(f"해외 표에 없는 칼럼: {miss}")
        if "alpha_3" in ov.columns:
            if not (ov["_unit"].astype(str) == OVERSEAS_PREFIX + ov["alpha_3"].astype(str)).all():
                raise ValueError("해외 sgg 가 'OV' + alpha_3 와 다름")
        parts.append(ov[need])
        du = set(dom["_unit"].astype(str))
        if any(u.startswith(OVERSEAS_PREFIX) for u in du):
            raise ValueError("국내 코드가 'OV' 로 시작함")
        if du & set(ov["_unit"].astype(str)):
            raise ValueError("국내·해외 단위 코드 겹침")
    df = pd.concat(parts, ignore_index=True)
    df["_unit"] = df["_unit"].astype(str)
    if units is not None:
        units = [str(u) for u in units]
        absent = sorted(set(units) - set(df["_unit"]))
        if absent:
            raise KeyError(f"일자료 표에 없는 단위: {absent[:10]}")
        df = df[df["_unit"].isin(set(units))]
    early = pd.to_datetime(df["date"]) < pd.Timestamp(VALUE_START)
    if early.any() and df.loc[early, variables].notna().to_numpy().any():
        raise ValueError(f"{VALUE_START} 전 날에 값이 있음(§4-3: 그 전 NaN) — 표를 확인")
    return wn.DailyCube.from_long(df, "_unit", "date", variables)


# ---------------------------------------------------------------- 생장식(날마다, float64 벡터)
def _cardinal(x, lo, opt, hi, square: bool):
    """카디널 꼴: square=True 면 (x−hi)(x−lo)² / {(opt−lo)[(opt−lo)(x−opt) − (opt−hi)(opt+lo−2x)]}(γ_T·γ_aw),
    False 면 (x−hi)(x−lo) / [(opt−lo)(x−opt) − (opt−hi)(lo−x)](γ_pH). lo < x < hi 밖 0, x 결측 → 결측."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if square:
            f = (x - hi) * (x - lo) ** 2 / ((opt - lo) * ((opt - lo) * (x - opt) - (opt - hi) * (opt + lo - 2.0 * x)))
        else:
            f = (x - hi) * (x - lo) / ((opt - lo) * (x - opt) - (opt - hi) * (lo - x))
    f = np.where((x > lo) & (x < hi), f, 0.0)
    return np.where(np.isnan(x), np.nan, f)


def gamma_T(theta):
    return _cardinal(theta, SILVA["Tmin"], SILVA["Topt"], SILVA["Tmax"], square=True)


def gamma_aw(aw):
    return _cardinal(aw, SILVA["awmin"], SILVA["awopt"], SILVA["awmax"], square=True)


def gamma_pH(ph):
    return _cardinal(ph, SILVA["pHmin"], SILVA["pHopt"], SILVA["pHmax"], square=False)


def mu_max(theta, aw, ph):
    """µmax = µopt·γ_T(θ)·γ_pH·γ_aw(브로드캐스트). 셋 중 하나라도 결측 → 결측."""
    return SILVA["mu_opt"] * gamma_T(theta) * gamma_pH(ph) * gamma_aw(aw)


def _sigmoid(z):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def tien_P(theta, aw, ph, strain: str):
    """Tienungoon 2000 P = 1/(1 + e^(−z)), z = b0 + b1·ln(θ−Tmin) + b2·ln²(θ−Tmin) + b3·ln{1 − e^(0.536(θ−48))}
    + b4·ln(aw−awmin) + b5·v + b6·v², v = ln(1 − 10^(pHmin − pH)), 젖산 항 0. 로그 안이 0 이하면 P = 0.
    θ·aw·pH 중 하나라도 결측 → 결측."""
    c = TIEN[strain]
    th, aw, ph = np.broadcast_arrays(np.asarray(theta, np.float64), np.asarray(aw, np.float64),
                                     np.asarray(ph, np.float64))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        a1 = th - c["Tmin"]
        a2 = 1.0 - np.exp(TIEN_FIXED["d"] * (th - TIEN_FIXED["Tmax"]))
        a3 = aw - c["awmin"]
        a4 = 1.0 - 10.0 ** (c["pHmin"] - ph)
        L = np.log(a1)
        v = np.log(a4)
        z = (c["b0"] + c["b1"] * L + c["b2"] * L ** 2 + c["b3"] * np.log(a2)
             + c["b4"] * np.log(a3) + c["b5"] * v + c["b6"] * v ** 2)
        p = _sigmoid(z)
    ok = (a1 > 0) & (a2 > 0) & (a3 > 0) & (a4 > 0)
    p = np.where(ok, p, 0.0)
    return np.where(np.isnan(th) | np.isnan(aw) | np.isnan(ph), np.nan, p)


def ivanek_P(rain_lag2):
    """Ivanek 2009: P = 1/(1 + e^(−(−1.79 + 0.13·rain[t−2]))). 결측 → 결측."""
    return _sigmoid(IVANEK["b0"] + IVANEK["b1"] * np.asarray(rain_lag2, dtype=np.float64))


# ---------------------------------------------------------------- 요약(결측 날 빼고)
def _mean_mean10_max(x: np.ndarray) -> np.ndarray:
    """(행, W, V) → (행, V, 3) [mean, mean10(E−9 ~ E), max] — windows.anomaly_summary_block 과 같은 계산 그대로 씀."""
    n, W, V = x.shape
    return wn.anomaly_summary_block(x).reshape(n, V, len(wn.ANOM_SUMMARY_STATS))


def _sum_npos(g: np.ndarray):
    """(행, W) → sum(값 있는 날 합), npos(값 있는 날 중 > 0 인 날 수), 값 있는 날 0 이면 결측."""
    ok = ~np.isnan(g)
    cnt = ok.sum(axis=1)
    s = np.where(cnt > 0, np.where(ok, g, 0.0).sum(axis=1), np.nan)
    npos = np.where(cnt > 0, (ok & (g > 0)).sum(axis=1).astype(np.float64), np.nan)
    return s, npos


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    ok = ~np.isnan(x)
    cnt = ok.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, np.where(ok, x, 0.0).sum(axis=axis) / np.maximum(cnt, 1), np.nan)


# ---------------------------------------------------------------- 입력 정리
def _row_array(a, n: int, what: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.shape[0] != n:
        raise ValueError(f"{what} 길이 {a.shape[0]} ≠ 행 수 {n}")
    return a


def _check_w10(W: int) -> int:
    if isinstance(W, bool) or not isinstance(W, (int, np.integer)):
        raise ValueError(f"W 는 정수: {W!r}")
    W = int(W)
    # 구현 세부(계획서에 없음): mean10 = E−9 ~ E 가 window 안에 있어야 해서 W ≥ 10 만 받음(W 후보 10~100).
    if W < LAST:
        raise ValueError(f"WX3·WX4 는 W ≥ {LAST}: {W}")
    return W


def _need(cube: wn.DailyCube, names: Sequence[str]) -> wn.DailyCube:
    miss = [v for v in names if v not in cube.variables]
    if miss:
        raise KeyError(f"일자료 표에 없는 변수: {miss}")
    return cube.select(list(names))


def _tail_bounds(b: wn.Bounds, end: np.ndarray, length: int) -> wn.Bounds:
    """end(포함)에서 끝나는 length 일 구간 — gather_window 가 쓰는 S·W 만 바꿈(T 는 그대로 → 예보 표 이음 위치 유지).
    구현 세부(계획서에 없음): windows.Bounds 를 dataclasses.replace 로 복사(모듈은 안 고침). 바뀐 Bounds 의
    n_obs·n_fc·g 는 원 window 값 그대로라 이 함수 밖에서 쓰지 않음."""
    end = np.asarray(end, dtype=np.int64)
    return dataclasses.replace(b, W=int(length), S=end - int(length) + 1, E=end)


# ---------------------------------------------------------------- WX3
def wx3_block(temp_win: np.ndarray, rain_lag2_win: np.ndarray, aw: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """temp_win (행, W) = window [S, E] 의 avg_temp, rain_lag2_win (행, W) = 같은 날 t 의 rain[t−2],
    aw·ph (행,) → (행, 16) WX3_NAMES 순."""
    gT = gamma_T(temp_win)
    s, npos = _sum_npos(gT)
    a, p = aw[:, None], ph[:, None]
    mu = mu_max(temp_win, a, p)
    ps = tien_P(temp_win, a, p, "scottA")
    pl = tien_P(temp_win, a, p, "L5")
    g3 = ivanek_P(rain_lag2_win)
    st = _mean_mean10_max(np.stack([gT, mu, ps, pl, g3], axis=2))   # (n, 5, 3)
    gr1 = np.stack([st[:, 0, 0], st[:, 0, 1], s, npos], axis=1)
    return np.concatenate([gr1, st[:, 1, :], st[:, 2, :], st[:, 3, :], st[:, 4, :]], axis=1)


def wx3_features(cube: wn.DailyCube, codes, d, h, W: int, aw, ph, *, forecast_cube: wn.DailyCube | None = None,
                 allow_missing_units: bool = False, chunk_rows: int = 10_000) -> pd.DataFrame:
    """WX3 16칼럼. window [S, E](E 끝, W 일); gr3 의 rain[t−2] 는 [S−2, E−2] 날 값(window_bounds(W + 2) 의 앞 W 칸)."""
    W = _check_w10(W)
    b = wn.window_bounds(d, h, W)
    b2 = wn.window_bounds(d, h, W + IVANEK["lag"])          # S − 2 ~ E
    n = b.d.shape[0]
    aw = _row_array(aw, n, "aw")
    ph = _row_array(ph, n, "pH")
    ct = _need(cube, ["avg_temp"])
    cr = _need(cube, ["rain"])
    ft = None if forecast_cube is None else _need(forecast_cube, ["avg_temp"])
    fr = None if forecast_cube is None else _need(forecast_cube, ["rain"])
    out = np.empty((n, len(WX3_NAMES)))
    for lo in range(0, n, chunk_rows):
        sl = slice(lo, min(n, lo + chunk_rows))
        t = wn.gather_window(ct, codes, b, ft, allow_missing_units, rows=sl)[:, :, 0]
        r = wn.gather_window(cr, codes, b2, fr, allow_missing_units, rows=sl)[:, :W, 0]
        out[sl] = wx3_block(t, r, aw[sl], ph[sl])
    return pd.DataFrame(out, columns=WX3_NAMES)


# ---------------------------------------------------------------- KO1(추가 판 — 머리 설명)
def _stable_sigmoid(z) -> np.ndarray:
    """1/(1 + e^(−z)) 를 넘침 없이(z ≥ 0: 1/(1 + e^(−z)), z < 0: e^z/(1 + e^z)) — NaN 은 NaN."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def ko_P(theta, aw, ph, medium: str) -> np.ndarray:
    """Koutsoumanis et al. (2004) 생육확률 P(브로드캐스트). θ·aw·pH 중 하나라도 결측 → 결측(NaN 이 그대로 퍼짐)."""
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = KOUTSOUMANIS[medium]
    T, aw, ph = np.broadcast_arrays(np.asarray(theta, np.float64), np.asarray(aw, np.float64),
                                    np.asarray(ph, np.float64))
    with np.errstate(invalid="ignore"):
        bw = np.sqrt(1.0 - aw)
    z = (a0 + a1 * T + a2 * ph + a3 * bw + a4 * T * ph + a5 * T * bw + a6 * ph * bw + a7 * T ** 2 + a8 * ph ** 2)
    return _stable_sigmoid(z)


def ko_block(temp_win: np.ndarray, aw: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """temp_win (행, W) = window [S, E] 의 avg_temp, aw·ph (행,) → (행, 6) KO_NAMES 순."""
    a, p = aw[:, None], ph[:, None]
    daily = np.stack([ko_P(temp_win, a, p, m) for m in KO_MEDIA], axis=2)     # (n, W, 2)
    st = _mean_mean10_max(daily)                                               # (n, 2, 3)
    return st.reshape(st.shape[0], -1)


def ko_features(cube: wn.DailyCube, codes, d, h, W: int, aw, ph, *, forecast_cube: wn.DailyCube | None = None,
                allow_missing_units: bool = False, chunk_rows: int = 10_000) -> pd.DataFrame:
    """KO1 6칼럼 — window [S, E](E 끝, W 일)·θ = avg_temp(WX3 gr2 와 같은 날짜 규칙)."""
    W = _check_w10(W)
    b = wn.window_bounds(d, h, W)
    n = b.d.shape[0]
    aw = _row_array(aw, n, "aw")
    ph = _row_array(ph, n, "pH")
    ct = _need(cube, ["avg_temp"])
    ft = None if forecast_cube is None else _need(forecast_cube, ["avg_temp"])
    out = np.empty((n, len(KO_NAMES)))
    for lo in range(0, n, chunk_rows):
        sl = slice(lo, min(n, lo + chunk_rows))
        t = wn.gather_window(ct, codes, b, ft, allow_missing_units, rows=sl)[:, :, 0]
        out[sl] = ko_block(t, aw[sl], ph[sl])
    return pd.DataFrame(out, columns=KO_NAMES)


# ---------------------------------------------------------------- WX4
def _in(x, lo, hi):
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isnan(x), np.nan, ((x >= lo) & (x <= hi)).astype(np.float64))


def _dist(x, lo, hi):
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isnan(x), np.nan, np.where(x < lo, lo - x, np.where(x > hi, x - hi, 0.0)))


def rg_weather_daily(avg_temp, max_temp, min_temp, soil_t5) -> np.ndarray:
    """날마다 기상 6칸(RG_WX_DAILY 순, 마지막 축). 경계 포함, 입력 결측 → 결측(rg_t_out_lim 은 둘 중 하나라도)."""
    mx = np.asarray(max_temp, np.float64)
    mn = np.asarray(min_temp, np.float64)
    lo, hi = RG["t_lim"]
    out_lim = np.where(np.isnan(mx) | np.isnan(mn), np.nan, ((mn < lo) | (mx > hi)).astype(np.float64))
    return np.stack([_in(mx, *RG["tmax_opt"]), _dist(avg_temp, *RG["tavg_opt"]), out_lim,
                     _in(soil_t5, *RG["soilt_opt"]), _in(soil_t5, *RG["soilt_lim"]), _dist(soil_t5, *RG["soilt_opt"])],
                    axis=-1)


def wx4_weather_features(cube: wn.DailyCube, codes, d, h, W: int, *, forecast_cube: wn.DailyCube | None = None,
                         allow_missing_units: bool = False, chunk_rows: int = 10_000) -> pd.DataFrame:
    """WX4 기상 12칼럼: 날마다 6칸 → window [S, E] 평균·최근 10일 평균(결측 날 빼고, 값 있는 날 0 이면 결측)."""
    W = _check_w10(W)
    b = wn.window_bounds(d, h, W)
    names = ["avg_temp", "max_temp", "min_temp", "soil_t5"]
    c = _need(cube, names)
    f = None if forecast_cube is None else _need(forecast_cube, names)
    n = b.d.shape[0]
    out = np.empty((n, len(WX4_WX_NAMES)))
    for lo in range(0, n, chunk_rows):
        sl = slice(lo, min(n, lo + chunk_rows))
        x = wn.gather_window(c, codes, b, f, allow_missing_units, rows=sl)
        daily = rg_weather_daily(x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3])   # (n, W, 6)
        st = _mean_mean10_max(daily)[:, :, :2]                                     # mean·mean10
        out[sl] = st.reshape(st.shape[0], -1)
    return pd.DataFrame(out, columns=WX4_WX_NAMES)


def wx4_soil_features(soil_ph) -> pd.DataFrame:
    """WX4 토양 pH 3칼럼(SO1 soil_ph 행마다 값으로). 경계 포함, 결측 → 결측."""
    x = np.asarray(soil_ph, dtype=np.float64).reshape(-1)
    return pd.DataFrame({"rg_soilph_opt": _in(x, *RG["soilph_opt"]), "rg_soilph_lim": _in(x, *RG["soilph_lim"]),
                         "rg_soilph_opt_dist": _dist(x, *RG["soilph_opt"])}, columns=WX4_SOIL_NAMES)


# ---------------------------------------------------------------- WX5·WX6
def wx56_block(x: np.ndarray) -> np.ndarray:
    """x (행, 25, 8) = R−25 ~ R−1 시간순(마지막 칸 = R−1), 변수 순 SHORT_VARS → (행, 8) WX5_NAMES + WX6_NAMES 순.

    구현 세부(계획서에 없음): 평균(t3·rain_mean25·solar_23·trange7)은 값 있는 날만 평균, 다 결측이면 결측
    (§4-2 요약·부록 A-3 WX3 의 'NaN 무시' 와 같게 — 계획서 WX5 칸에는 결측 규칙이 없음). trange7 의 날마다
    max_temp − min_temp 는 둘 중 하나라도 결측이면 그날 결측.
    """
    at, mx, mn, rn, wd, sr, rh, td = (x[:, :, i] for i in range(x.shape[2]))
    r1, r2, r3 = SHORT_LEN - 1, SHORT_LEN - 2, SHORT_LEN - 3       # R−1, R−2, R−3 위치
    t3 = _nanmean(at[:, r3:], 1)                                   # R−3 ~ R−1
    rain_lag2 = rn[:, r2]
    wind_lag2 = wd[:, r2]
    rain_mean25 = _nanmean(rn, 1)                                  # R−25 ~ R−1
    solar_23 = _nanmean(sr[:, r3:r1], 1)                           # R−3, R−2
    trange7 = _nanmean((mx - mn)[:, SHORT_LEN - 7:], 1)            # R−7 ~ R−1
    return np.stack([t3, rain_lag2, wind_lag2, rain_mean25, solar_23, trange7, rh[:, r1], td[:, r1]], axis=1)


SHORT_VARS = ("avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "solar_rad", "rh", "td")


def short_bounds(d, h) -> wn.Bounds:
    """R − 25 ~ R − 1 구간(R = min(d, E + 1), §5). R 은 W 와 무관 — window_bounds 의 W 자리는 25 를 넣고
    S·E 만 R 기준으로 바꿈."""
    b = wn.window_bounds(d, h, SHORT_LEN)
    return _tail_bounds(b, b.R - 1, SHORT_LEN)


def wx56_features(cube: wn.DailyCube, codes, d, h, *, forecast_cube: wn.DailyCube | None = None,
                  allow_missing_units: bool = False, chunk_rows: int = 50_000) -> pd.DataFrame:
    """WX5 6칼럼 + WX6 2칼럼(R − 1 끝)."""
    b = short_bounds(d, h)
    c = _need(cube, SHORT_VARS)
    f = None if forecast_cube is None else _need(forecast_cube, SHORT_VARS)
    n = b.d.shape[0]
    out = np.empty((n, len(WX5_NAMES) + len(WX6_NAMES)))
    for lo in range(0, n, chunk_rows):
        sl = slice(lo, min(n, lo + chunk_rows))
        out[sl] = wx56_block(wn.gather_window(c, codes, b, f, allow_missing_units, rows=sl))
    return pd.DataFrame(out, columns=WX5_NAMES + WX6_NAMES)


# ---------------------------------------------------------------- 묶어서
def wx_derived(cube: wn.DailyCube, codes, d, h, W: int, *, aw=None, ph=None, soil_ph=None,
               groups: Sequence[str] = GROUPS, forecast_cube: wn.DailyCube | None = None,
               allow_missing_units: bool = False, chunk_rows: int = 10_000) -> pd.DataFrame:
    """행((시료, h) 쌍)마다 WX3~WX6 칼럼. aw·ph(FD1)는 WX3 에, soil_ph(SO1)는 WX4 에 필요(행 수와 같은 배열).
    칼럼 순서 = groups 순 → 각 그룹 이름 목록 순."""
    bad = [g for g in groups if g not in GROUP_NAMES]
    if bad:
        raise ValueError(f"모르는 그룹: {bad}")
    kw = dict(forecast_cube=forecast_cube, allow_missing_units=allow_missing_units)
    short = wx56_features(cube, codes, d, h, **kw) if ("WX5" in groups or "WX6" in groups) else None
    parts = []
    for g in groups:
        if g == "WX3":
            if aw is None or ph is None:
                raise ValueError("WX3 에는 aw·ph 배열 필요")
            parts.append(wx3_features(cube, codes, d, h, W, aw, ph, chunk_rows=chunk_rows, **kw))
        elif g == "WX4":
            if soil_ph is None:
                raise ValueError("WX4 에는 soil_ph 배열 필요")
            w = wx4_weather_features(cube, codes, d, h, W, chunk_rows=chunk_rows, **kw)
            s = wx4_soil_features(_row_array(soil_ph, len(w), "soil_ph"))
            parts.append(pd.concat([w, s], axis=1))
        elif g == "WX5":
            parts.append(short[WX5_NAMES])
        elif g == "WX6":
            parts.append(short[WX6_NAMES])
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------- 생장식 엑셀 계수 대조
class GrowthCoefMismatch(ValueError):
    """엑셀 값과 계획서(부록 A-3) 값이 다름 — 고치지 않고 멈춤."""


_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _norm(s: str) -> str:
    """NFKC(수학 기울임 글자 → ASCII) + 아래 첨자 숫자 → 숫자 + 빼기표(−) → '-' + 공백·보이지 않는 곱셈표 제거."""
    t = unicodedata.normalize("NFKC", str(s)).translate(_SUB)
    t = t.replace("−", "-").replace("⁢", "").replace("⁡", "")
    return re.sub(r"\s+", "", t)


def _seg(text: str, strain: str | None) -> str:
    if strain is None:
        return text
    key = {"scottA": "ScottA:", "L5": "L5:"}[strain]
    for part in text.split("/"):
        if part.startswith(key):
            return part
    raise GrowthCoefMismatch(f"균주 '{strain}' 칸을 못 찾음: {text[:80]!r}")


def _labeled(text: str, label: str) -> float:
    m = re.search(re.escape(label) + r"=(-?\d+(?:\.\d+)?)", text)
    if not m:
        raise GrowthCoefMismatch(f"'{label}=' 값을 못 찾음: {text[:120]!r}")
    return float(m.group(1))


# (셀, 균주 또는 None, 엑셀 글자 이름, 계획서 값이 든 사전, 키)
_CHECKS = (
    [("F6", None, k, SILVA, k) for k in ("Tmin", "Topt", "Tmax")]
    + [("F26", None, k, SILVA, k) for k in ("pHmin", "pHopt", "pHmax")]
    + [("F19", None, k, SILVA, k) for k in ("awmin", "awopt")]
    + [("F15", st, k, TIEN[st], k) for st in TIEN for k in ("b0", "b1", "b2", "b3", "Tmin")]
    + [("F21", st, k, TIEN[st], k) for st in TIEN for k in ("b4", "awmin")]
    + [("F28", st, k, TIEN[st], k) for st in TIEN for k in ("b5", "b6", "pHmin")]
    + [("J15", None, "Tmax", TIEN_FIXED, "Tmax"), ("J15", None, "d", TIEN_FIXED, "d"),
       ("F41", None, "intercept", IVANEK, "b0"), ("F41", None, "β(Precipitation.2)", IVANEK, "b1")]
)


def check_growth_cells(cells: Mapping[str, str]) -> list[dict]:
    """셀 글자(셀 주소 → 문자열) ↔ 계획서 값 대조. 다 같으면 대조 목록, 하나라도 다르면 GrowthCoefMismatch."""
    rows, bad = [], []
    norm = {k: _norm(v) for k, v in cells.items() if v is not None}
    for cell, st, label, plan, key in _CHECKS:
        val = _labeled(_seg(norm.get(cell, ""), st), label)
        rows.append({"cell": cell, "strain": st, "name": label, "xlsx": val, "plan": plan[key]})
        if val != plan[key]:
            bad.append(rows[-1])
    # 식 안 상수: D6 awmax 1.00(aw−1.00), J6 µopt(둘째 줄 맨 앞 0.409×), D15 0.536(T−48)
    extra = [("D6", "awmax", r"\(aw-(\d+\.\d+)\)", SILVA["awmax"]),
             ("J6", "mu_opt", r"혼합모델(\d+\.\d+)×", SILVA["mu_opt"]),
             ("D15", "d", r"ln\{1-exp\[(\d+\.\d+)\(T-48\)\]\}", TIEN_FIXED["d"]),
             ("D15", "Tmax", r"exp\[0\.536\(T-(\d+)\)\]", TIEN_FIXED["Tmax"])]
    for cell, name, pat, pv in extra:
        m = re.search(pat, norm.get(cell, ""))
        val = float(m.group(1)) if m else None
        rows.append({"cell": cell, "strain": None, "name": name, "xlsx": val, "plan": pv})
        if val != pv:
            bad.append(rows[-1])
    if bad:
        raise GrowthCoefMismatch(f"생장식 엑셀 값이 계획서와 다름(고치지 않음, 보고): {bad}")
    return rows


GROWTH_CELLS = ("F6", "F26", "F19", "F15", "F21", "F28", "J15", "F41", "D6", "J6", "D15")


def read_growth_cells(xlsx_path, sheet: str = GROWTH_SHEET) -> dict[str, str]:
    import openpyxl  # 늦게 불러옴

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=False)
    try:
        ws = wb[sheet]
        return {c: ws[c].value for c in GROWTH_CELLS}
    finally:
        wb.close()
