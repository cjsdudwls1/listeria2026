"""WX3~WX6 시험 — 부록 A-3 공식·결측 규칙, §5 R 규칙, 일자료 표 합치기, 엑셀 계수 대조.

참값은 날짜를 하나씩 찾아 파이썬 반복문으로 계산(window 모듈을 거치지 않음)."""
import math

import numpy as np
import pandas as pd
import pytest

from lst import windows as wn
from lst import wxfeat as wf
from lst.folds import to_days


def D(s):
    return int(to_days(s)[0])


# ---------------------------------------------------------------- 합성 일자료
START, END = D("2025-01-01"), D("2025-12-31")
UNITS = ("11110", "OVUSA")


def _make_long(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in UNITS:
        for t in range(START, END + 1):
            rows.append({"unit": u, "day": t})
    df = pd.DataFrame(rows)
    n = len(df)
    df["avg_temp"] = rng.uniform(-5, 40, n)
    df["max_temp"] = df["avg_temp"] + rng.uniform(0, 10, n)
    df["min_temp"] = df["avg_temp"] - rng.uniform(0, 10, n)
    df["rain"] = np.where(rng.random(n) < 0.6, 0.0, rng.exponential(10, n))
    df["avg_wind"] = rng.uniform(0, 8, n)
    df["solar_rad"] = rng.uniform(0, 30, n)
    df["soil_t5"] = rng.uniform(-3, 35, n)
    df["rh"] = rng.uniform(20, 100, n)
    df["td"] = rng.uniform(-20, 25, n)
    # 결측 넣기
    for c in ("avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "solar_rad", "soil_t5", "rh", "td"):
        df.loc[rng.random(n) < 0.08, c] = np.nan
    # 경계값 넣기
    df.loc[5, "max_temp"] = 30.0
    df.loc[6, "max_temp"] = 37.0
    df.loc[7, "soil_t5"] = 8.0
    df.loc[8, "soil_t5"] = 15.0
    df["date"] = pd.to_datetime(df["day"].astype("int64").astype("datetime64[D]"))
    return df


VARS = ("avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "solar_rad", "soil_t5", "rh", "td")


@pytest.fixture(scope="module")
def data():
    df = _make_long()
    cube = wn.DailyCube.from_long(df, "unit", "date", VARS)
    look = {(r.unit, r.day): r for r in df.itertuples(index=False)}
    return df, cube, look


def _val(look, u, t, v):
    r = look.get((u, t))
    return np.nan if r is None else float(getattr(r, v))


def _rows():
    # (단위, d, h): h ≤ 10·11·12·60 등, 표 시작 근처(표 밖 날 NaN) 포함
    out = []
    for u in UNITS:
        for d in ("2025-03-15", "2025-07-01", "2025-11-01", "2025-01-20"):
            for h in (1, 9, 10, 11, 12, 13, 30, 60):
                out.append((u, D(d), h))
    return out


def _bounds(d, h, W):
    T = d - h
    E = min(d, T + 10)
    return T, E - W + 1, E, min(d, E + 1)


def _nm(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else np.nan


def _nmax(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.max(xs)) if xs else np.nan


def _close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), (a, b)
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------- 공식 단위 시험
def test_cardinal_values():
    assert wf.gamma_T(wf.SILVA["Topt"]) == pytest.approx(1.0)
    assert wf.gamma_pH(wf.SILVA["pHopt"]) == pytest.approx(1.0)
    assert wf.gamma_aw(wf.SILVA["awopt"]) == pytest.approx(1.0)
    # 범위 밖·경계 = 0, 결측 = 결측
    for f, lo, hi in ((wf.gamma_T, "Tmin", "Tmax"), (wf.gamma_pH, "pHmin", "pHmax"), (wf.gamma_aw, "awmin", "awmax")):
        x = np.array([wf.SILVA[lo], wf.SILVA[hi], wf.SILVA[lo] - 1, wf.SILVA[hi] + 1, np.nan])
        y = f(x)
        assert (y[:4] == 0).all() and np.isnan(y[4])
    # 손계산 한 점(γ_T, θ = 20)
    a, o, b = wf.SILVA["Tmin"], wf.SILVA["Topt"], wf.SILVA["Tmax"]
    th = 20.0
    ref = (th - b) * (th - a) ** 2 / ((o - a) * ((o - a) * (th - o) - (o - b) * (o + a - 2 * th)))
    assert wf.gamma_T(th) == pytest.approx(ref)
    # γ_pH 손계산(pH 5)
    a, o, b = wf.SILVA["pHmin"], wf.SILVA["pHopt"], wf.SILVA["pHmax"]
    x = 5.0
    assert wf.gamma_pH(x) == pytest.approx((x - b) * (x - a) / ((o - a) * (x - o) - (o - b) * (a - x)))
    assert wf.mu_max(th, 0.98, 6.0) == pytest.approx(0.409 * wf.gamma_T(th) * wf.gamma_pH(6.0) * wf.gamma_aw(0.98))
    assert np.isnan(wf.mu_max(th, np.nan, 6.0)) and np.isnan(wf.mu_max(np.nan, 0.98, 6.0))


def _tien_ref(th, aw, ph, st):
    c = wf.TIEN[st]
    args = (th - c["Tmin"], 1 - math.exp(0.536 * (th - 48)), aw - c["awmin"], 1 - 10 ** (c["pHmin"] - ph))
    if min(args) <= 0:
        return 0.0
    L, v = math.log(args[0]), math.log(args[3])
    z = (c["b0"] + c["b1"] * L + c["b2"] * L * L + c["b3"] * math.log(args[1]) + c["b4"] * math.log(args[2])
         + c["b5"] * v + c["b6"] * v * v)
    return 1 / (1 + math.exp(-z))


@pytest.mark.parametrize("st", ["scottA", "L5"])
def test_tienungoon(st):
    for th, aw, ph in [(25, 0.98, 6.5), (10, 0.95, 5.0), (4, 0.99, 7.0), (35, 0.93, 4.0),
                       (48, 0.98, 6.5), (60, 0.98, 6.5), (-5, 0.98, 6.5), (25, 0.90, 6.5), (25, 0.98, 3.0)]:
        assert wf.tien_P(th, aw, ph, st) == pytest.approx(_tien_ref(th, aw, ph, st), abs=1e-15)
    assert np.isnan(wf.tien_P(np.nan, 0.98, 6.5, st)) and np.isnan(wf.tien_P(25, np.nan, 6.5, st))
    assert np.isnan(wf.tien_P(25, 0.98, np.nan, st))


def test_ivanek():
    assert wf.ivanek_P(0.0) == pytest.approx(1 / (1 + math.exp(1.79)))
    assert wf.ivanek_P(10.0) == pytest.approx(1 / (1 + math.exp(-(-1.79 + 1.3))))
    assert np.isnan(wf.ivanek_P(np.nan))


# ---------------------------------------------------------------- WX3
@pytest.mark.parametrize("W", [10, 30, 60])
def test_wx3_matches_bruteforce(data, W):
    df, cube, look = data
    rows = _rows()
    codes = [r[0] for r in rows]
    d = np.array([r[1] for r in rows])
    h = np.array([r[2] for r in rows])
    rng = np.random.default_rng(1)
    aw = rng.uniform(0.88, 1.0, len(rows))
    ph = rng.uniform(3.0, 8.0, len(rows))
    aw[::7] = np.nan
    ph[3::11] = np.nan
    got = wf.wx3_features(cube, codes, d, h, W, aw, ph, allow_missing_units=False, chunk_rows=7)
    assert list(got.columns) == wf.WX3_NAMES and len(wf.WX3_NAMES) == 16
    for i, (u, dd, hh) in enumerate(rows):
        T, S, E, R = _bounds(dd, hh, W)
        days = list(range(S, E + 1))
        th = [_val(look, u, t, "avg_temp") for t in days]
        gT = [float(wf.gamma_T(x)) for x in th]
        ok = [x for x in gT if not math.isnan(x)]
        last = [x for x, t in zip(gT, days) if t >= E - 9]
        exp = [_nm(gT), _nm(last), float(sum(ok)) if ok else np.nan,
               float(sum(1 for x in ok if x > 0)) if ok else np.nan]
        parts = []
        for f in (lambda x: float(wf.mu_max(x, aw[i], ph[i])),
                  lambda x: float(wf.tien_P(x, aw[i], ph[i], "scottA")),
                  lambda x: float(wf.tien_P(x, aw[i], ph[i], "L5"))):
            v = [f(x) for x in th]
            parts.append(v)
        parts.append([float(wf.ivanek_P(_val(look, u, t - 2, "rain"))) for t in days])
        for v in parts:
            exp += [_nm(v), _nm([x for x, t in zip(v, days) if t >= E - 9]), _nmax(v)]
        _close(got.iloc[i].to_numpy(), exp)


def test_wx3_missing_food_makes_gr2_nan(data):
    _, cube, _ = data
    got = wf.wx3_features(cube, ["11110"], [D("2025-07-01")], [5], 30, [np.nan], [6.0])
    assert got.filter(like="gr2_").isna().all(axis=None)
    assert got[["gr1_mean", "gr3_mean"]].notna().all(axis=None)


def test_wx3_rejects_short_w(data):
    _, cube, _ = data
    with pytest.raises(ValueError):
        wf.wx3_features(cube, ["11110"], [D("2025-07-01")], [5], 9, [0.98], [6.0])
    with pytest.raises(ValueError):
        wf.wx3_features(cube, ["11110", "11110"], [D("2025-07-01")] * 2, [5, 6], 30, [0.98], [6.0])


# ---------------------------------------------------------------- WX4
def _rg_ref(at, mx, mn, s5):
    def inn(x, lo, hi):
        return np.nan if math.isnan(x) else float(lo <= x <= hi)

    def dist(x, lo, hi):
        return np.nan if math.isnan(x) else (lo - x if x < lo else (x - hi if x > hi else 0.0))
    out_lim = np.nan if (math.isnan(mx) or math.isnan(mn)) else float(mn < -2 or mx > 45)
    return [inn(mx, 30, 37), dist(at, 30, 37), out_lim, inn(s5, 8, 15), inn(s5, 8, 30), dist(s5, 8, 15)]


def test_rg_daily_boundaries():
    v = wf.rg_weather_daily(np.array([30.0, 37.0, 29.9, np.nan]), np.array([30.0, 37.0, 37.01, 45.0]),
                            np.array([-2.0, -2.01, 0.0, np.nan]), np.array([8.0, 15.0, 30.0, 30.01]))
    _close(v[:, 0], [1, 1, 0, 0])
    _close(v[:, 1], [0, 0, 0.1, np.nan])
    _close(v[:, 2], [0, 1, 0, np.nan])
    _close(v[:, 3], [1, 1, 0, 0])
    _close(v[:, 4], [1, 1, 1, 0])
    _close(v[:, 5], [0, 0, 15, 15.01])


@pytest.mark.parametrize("W", [10, 60, 100])
def test_wx4_weather_matches_bruteforce(data, W):
    _, cube, look = data
    rows = _rows()
    got = wf.wx4_weather_features(cube, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], W,
                                  chunk_rows=5)
    assert list(got.columns) == wf.WX4_WX_NAMES and len(wf.WX4_WX_NAMES) == 12
    for i, (u, dd, hh) in enumerate(rows):
        T, S, E, R = _bounds(dd, hh, W)
        days = list(range(S, E + 1))
        daily = [_rg_ref(*(_val(look, u, t, v) for v in ("avg_temp", "max_temp", "min_temp", "soil_t5"))) for t in days]
        exp = []
        for k in range(6):
            col = [x[k] for x in daily]
            exp += [_nm(col), _nm([x for x, t in zip(col, days) if t >= E - 9])]
        _close(got.iloc[i].to_numpy(), exp)


def test_wx4_soil():
    got = wf.wx4_soil_features([6.5, 7.0, 7.01, 5.0, 8.0, 4.99, np.nan])
    _close(got["rg_soilph_opt"], [1, 1, 0, 0, 0, 0, np.nan])
    _close(got["rg_soilph_lim"], [1, 1, 1, 1, 1, 0, np.nan])
    _close(got["rg_soilph_opt_dist"], [0, 0, 0.01, 1.5, 1.0, 1.51, np.nan])


# ---------------------------------------------------------------- WX5·WX6
def test_short_bounds_R():
    d = D("2025-11-01")
    for h in range(1, 61):
        b = wf.short_bounds([d], [h])
        T = d - h
        E = min(d, T + 10)
        R = min(d, E + 1)
        if h <= 11:
            assert R == d
        assert b.E[0] == R - 1 and b.S[0] == R - 25 and b.W == 25
        assert b.R[0] == R and b.T[0] == T


def test_wx56_matches_bruteforce(data):
    _, cube, look = data
    rows = _rows()
    got = wf.wx56_features(cube, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], chunk_rows=3)
    assert list(got.columns) == wf.WX5_NAMES + wf.WX6_NAMES
    for i, (u, dd, hh) in enumerate(rows):
        _, _, _, R = _bounds(dd, hh, 10)
        g = lambda v, k: _val(look, u, R - k, v)  # noqa: E731
        exp = [_nm([g("avg_temp", k) for k in (1, 2, 3)]), g("rain", 2), g("avg_wind", 2),
               _nm([g("rain", k) for k in range(1, 26)]), _nm([g("solar_rad", 3), g("solar_rad", 2)]),
               _nm([g("max_temp", k) - g("min_temp", k) for k in range(1, 8)]), g("rh", 1), g("td", 1)]
        _close(got.iloc[i].to_numpy(), exp)


def test_wx_derived_and_forecast_cube(data):
    df, cube, look = data
    rows = _rows()[:10]
    codes, d, h = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    n = len(rows)
    out = wf.wx_derived(cube, codes, d, h, 30, aw=np.full(n, 0.98), ph=np.full(n, 6.0), soil_ph=np.full(n, 6.7))
    assert list(out.columns) == wf.WX3_NAMES + wf.WX4_WX_NAMES + wf.WX4_SOIL_NAMES + wf.WX5_NAMES + wf.WX6_NAMES
    # 예보 표: 날짜 ≥ T 칸만 그 표 값 → 관측 표 값 + 100 이면 t3 등은 예보 칸 날만 바뀜
    fc = wn.DailyCube(cube.start, cube.units, cube.variables, cube.values + 100.0)
    a = wf.wx56_features(cube, codes, d, h)
    b = wf.wx56_features(cube, codes, d, h, forecast_cube=fc)
    for i, (u, dd, hh) in enumerate(rows):
        T = dd - hh
        _, _, _, R = _bounds(dd, hh, 10)
        # rh1 = rh[R−1]: R−1 ≥ T 면 예보 표 값
        va, vb = a["rh1"].iloc[i], b["rh1"].iloc[i]
        if math.isnan(va):
            assert math.isnan(vb)
        elif R - 1 >= T:
            assert vb == pytest.approx(va + 100)
        else:
            assert vb == va
    with pytest.raises(ValueError):
        wf.wx_derived(cube, codes, d, h, 30, aw=None, ph=None, soil_ph=None, groups=("WX3",))


# ---------------------------------------------------------------- 표 합치기
def _dom_ov():
    dates = pd.date_range("2014-12-30", "2015-01-03").date
    dom = pd.DataFrame({"sgg_code": ["11110"] * 5, "date": dates})
    ov = pd.DataFrame({"alpha_3": ["USA"] * 3, "sgg": ["OVUSA"] * 3, "date": dates[2:]})
    for c in wf.WEATHER19:
        dom[c] = np.where(pd.to_datetime(dom["date"]) < pd.Timestamp("2015-01-01"), np.nan, 1.0)
        ov[c] = 2.0
    dom["rh"], dom["td"] = 50.0, 5.0
    dom.loc[:1, ["rh", "td"]] = np.nan
    ov["avg_rhm"], ov["avg_td"] = 60.0, np.nan
    ov["src_avg_temp"] = "db"
    return dom, ov


def test_combined_cube():
    dom, ov = _dom_ov()
    c = wf.combined_cube(dom, ov)
    assert c.units == ("11110", "OVUSA") and c.variables == wf.CUBE_VARS
    assert c.start == D("2014-12-30") and c.n_days == 5
    iu = c.unit_index(["OVUSA"])[0]
    assert np.isnan(c.values[iu, :2, :]).all()                     # 해외 표에 없는 날 = NaN
    assert (c.values[iu, 2:, c.variables.index("rh")] == 60).all()  # avg_rhm → rh
    assert np.isnan(c.values[iu, 2:, c.variables.index("td")]).all()
    c2 = wf.combined_cube(dom, ov, units=["OVUSA"])
    assert c2.units == ("OVUSA",)
    with pytest.raises(KeyError):
        wf.combined_cube(dom, ov, units=["OVJPN"])


def test_combined_cube_stops():
    dom, ov = _dom_ov()
    bad = dom.copy()
    bad.loc[0, "avg_temp"] = 3.0                                     # 2015 전 값
    with pytest.raises(ValueError):
        wf.combined_cube(bad, ov)
    bad = ov.copy()
    bad["sgg"] = "OVCAN"
    with pytest.raises(ValueError):
        wf.combined_cube(dom, bad)
    bad = ov.copy()
    bad["rh"] = 1.0
    with pytest.raises(ValueError):
        wf.combined_cube(dom, bad)
    bad = ov.copy()
    bad["sgg"] = "11110"
    bad = bad.drop(columns="alpha_3")
    with pytest.raises(ValueError):
        wf.combined_cube(dom, bad)


# ---------------------------------------------------------------- 엑셀 계수 대조
CELLS = {
    "F6": "Tmin = −1.273, \nTopt = 37.26, \nTmax = 45.12",
    "F19": "aw min = 0.894 \naw opt = 0.995",
    "F26": "pHmin = 4.303, \npHopt = 7.085, \npHmax = 9.483",
    "F15": "Scott A: b₀=−6.023\nb₁=19.00\nb₂=−3.049\nb₃=7514\nTmin=0.4164 / L5: b₀=−25.36\nb₁=44.12\nb₂=−7.022\nb₃=10257\nTmin=−1.623",
    "F21": "Scott A: b₄=4.635, awmin=0.9142 / \nL5: b₄=8.951, awmin=0.9152",
    "F28": "Scott A: b₅=141.0, b₆=240.2, pHmin=3.350 / \nL5: b₅=291.8, b₆=704.1, pHmin=3.350",
    "F41": "intercept=−1.79 (SE 0.19), β(Precipitation.2)=0.13 (SE 0.03)",
    "J15": "온도, aw, pH, 젖산 농도 혼합모델. Tmax=48℃, pHmax=9.5, d=0.536 고정.",
    "D6": "자체개발\n√(𝑎⁢𝑤−1.00)·(𝑎⁢𝑤−𝑎𝑤min)^2",
    "J6": "온도, aw, pH 혼합모델\n0.409×√(𝑇−45.12)",
    "D15": "ln[P/(1−P)] = b₀+b₁·ln(T−Tmin)+b₂·ln²(T−Tmin)+b₃·ln{1−exp[0.536(T−48)]}+b₄·ln(aw−awmin)",
}


def test_growth_cells_ok():
    rows = wf.check_growth_cells(CELLS)
    assert len(rows) == 36 and all(r["xlsx"] == r["plan"] for r in rows)


@pytest.mark.parametrize("cell,old,new", [("F15", "b₃=10257", "b₃=10258"), ("F6", "37.26", "37.27"),
                                          ("J6", "0.409", "0.41"), ("F41", "0.13", "0.14")])
def test_growth_cells_mismatch(cell, old, new):
    c = dict(CELLS)
    c[cell] = c[cell].replace(old, new)
    with pytest.raises(wf.GrowthCoefMismatch):
        wf.check_growth_cells(c)
