"""window 생성기 시험 — 부록 B-7 검산 예시 7개 그대로 + 요약·원값·차감·이음·RR8."""
import numpy as np
import pandas as pd
import pytest

from lst import windows as wn
from lst.folds import as_date, to_days


def D(s):
    return int(to_days(s)[0])


# 부록 B-7: d = 2025-11-01, W = 60 (마지막 줄만 W = 100)
# (W, h, T, S, E, 관측, 예보, 밖(g))
B7 = [
    (60, 1, "2025-10-31", "2025-09-03", "2025-11-01", 58, 2, 0),
    (60, 10, "2025-10-22", "2025-09-03", "2025-11-01", 49, 11, 0),
    (60, 11, "2025-10-21", "2025-09-02", "2025-10-31", 49, 11, 1),
    (60, 12, "2025-10-20", "2025-09-01", "2025-10-30", 49, 11, 2),
    (60, 30, "2025-10-02", "2025-08-14", "2025-10-12", 49, 11, 20),
    (60, 60, "2025-09-02", "2025-07-15", "2025-09-12", 49, 11, 50),
    (100, 13, "2025-10-19", "2025-07-22", "2025-10-29", 89, 11, 3),
]


@pytest.mark.parametrize("W,h,T,S,E,n_obs,n_fc,g", B7)
def test_b7_examples(W, h, T, S, E, n_obs, n_fc, g):
    b = wn.window_bounds("2025-11-01", h, W)
    assert b.T[0] == D(T)
    assert b.S[0] == D(S)
    assert b.E[0] == D(E)
    assert b.n_obs[0] == n_obs and b.n_fc[0] == n_fc
    assert b.g[0] == g
    # 관측·예보 칸 표시도 같은 수
    fc = wn.forecast_day_mask(b)[0]
    assert fc.sum() == n_fc and (~fc).sum() == n_obs
    # 예보 칸은 window 끝쪽에 붙어 있음(T ~ E)
    assert fc[-n_fc:].all() and not fc[:-n_fc].any()
    # 실제 생성 window: 날짜값 표로 S·E 와 lag 이름 확인
    days = np.arange(D("2025-06-01"), D("2025-12-31") + 1)
    cube = wn.DailyCube(int(days[0]), ("U1",), ("x",), days.astype(float)[None, :, None])
    raw = wn.window_features(cube, ["U1"], "2025-11-01", h, W, "raw")
    assert raw["x_lag001"].iloc[0] == D(E)
    assert raw[f"x_lag{W:03d}"].iloc[0] == D(S)
    seq = wn.sequences(cube, ["U1"], "2025-11-01", h, W)
    assert seq.shape == (1, W, 1) and seq[0, 0, 0] == D(S) and seq[0, -1, 0] == D(E)


def test_r_definition():
    # R = min(d, E + 1): h ≤ 11 이면 R = d
    for h in range(1, 61):
        b = wn.window_bounds("2025-11-01", h, 60)
        assert b.R[0] == min(b.d[0], b.E[0] + 1)
        if h <= 11:
            assert b.R[0] == b.d[0]
        else:
            assert b.R[0] == b.d[0] - (h - 10) + 1


def test_bounds_reject_bad_h():
    with pytest.raises(ValueError):
        wn.window_bounds("2025-11-01", 0, 60)
    with pytest.raises(ValueError):
        wn.window_bounds("2025-11-01", 61, 60)


def _cube(rng, U=3, Dn=400, V=2, nan_frac=0.15, start="2024-01-01"):
    vals = rng.normal(10, 5, size=(U, Dn, V))
    vals[rng.random(vals.shape) < nan_frac] = np.nan
    return wn.DailyCube(D(start), tuple(f"U{i}" for i in range(U)), tuple(f"v{j}" for j in range(V)), vals)


def _naive_summary(x, W):
    """한 행·한 변수 x(시간순 W) 의 요약 — 손 계산 기준."""
    out = []
    rev = x[::-1]
    for k in range(W // 10):
        seg = rev[10 * k:10 * (k + 1)]
        seg = seg[~np.isnan(seg)]
        out.append(seg.mean() if seg.size else np.nan)
    ok = ~np.isnan(x)
    v = x[ok]
    if v.size:
        out += [v.mean(), v.std(ddof=0), v.min(), v.max()]
    else:
        out += [np.nan] * 4
    t = np.arange(W)[ok]
    out.append(np.polyfit(t, v, 1)[0] if v.size >= 2 else np.nan)
    return out


def test_summary_matches_naive():
    rng = np.random.default_rng(0)
    cube = _cube(rng)
    n = 40
    codes = rng.choice(cube.units, n)
    d = D("2024-06-01") + rng.integers(0, 150, n)
    h = rng.integers(1, 61, n)
    W = 30
    feats = wn.window_features(cube, codes, d, h, W, "sum", chunk_rows=7)
    x = wn.sequences(cube, codes, d, h, W)
    names = wn.summary_names(cube.variables, W)
    assert list(feats.columns) == names
    for i in range(n):
        for j, v in enumerate(cube.variables):
            exp = _naive_summary(x[i, :, j], W)
            got = feats.loc[i, [c for c in names if c.startswith(v + "_")]].to_numpy()
            np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_summary_all_nan_and_single_value():
    x = np.full((2, 10, 1), np.nan)
    x[1, 3, 0] = 4.0
    blk = wn.summary_block(x)  # b01, mean, std, min, max, slope
    assert np.isnan(blk[0]).all()
    np.testing.assert_allclose(blk[1, :5], [4.0, 4.0, 0.0, 4.0, 4.0])
    assert np.isnan(blk[1, 5])  # 값 하나 → 기울기 NaN


def test_bins_order_b01_is_last_10_days():
    x = np.arange(20, dtype=float)[None, :, None]  # S = 0 … E = 19
    names = wn.summary_names(["v"], 20)
    blk = dict(zip(names, wn.summary_block(x)[0]))
    assert blk["v_b01"] == np.mean(np.arange(10, 20))   # E−9 ~ E
    assert blk["v_b02"] == np.mean(np.arange(0, 10))
    assert blk["v_slope"] == pytest.approx(1.0)


def test_raw_lag_names_and_order():
    x = np.arange(10, dtype=float)[None, :, None] + np.array([0, 100])[None, None, :]
    names = wn.raw_lag_names(["a", "b"], 10)
    blk = dict(zip(names, wn.raw_lag_block(x)[0]))
    assert blk["a_lag001"] == 9 and blk["a_lag010"] == 0
    assert blk["b_lag001"] == 109 and blk["b_lag010"] == 100


def _naive_anomaly(series, N):
    out = np.full(series.shape, np.nan)
    for t in range(series.shape[0]):
        prev = series[max(0, t - N):t]
        prev = prev[~np.isnan(prev)]
        if prev.size and not np.isnan(series[t]):
            out[t] = series[t] - prev.mean()
    return out


@pytest.mark.parametrize("N", [5, 7, 15, 30])
def test_anomaly_matches_naive(N):
    rng = np.random.default_rng(N)
    cube = _cube(rng, U=2, Dn=120, V=2, nan_frac=0.3)
    a = wn.anomaly_cube(cube, N)
    assert a.variables == (f"v0_anom{N}", f"v1_anom{N}")
    for u in range(2):
        for v in range(2):
            np.testing.assert_allclose(a.values[u, :, v], _naive_anomaly(cube.values[u, :, v], N),
                                       rtol=1e-12, atol=1e-12, equal_nan=True)


def test_anomaly_features_sum_and_raw():
    rng = np.random.default_rng(3)
    cube = _cube(rng, U=2, Dn=300, V=1, nan_frac=0.2)
    a = wn.anomaly_cube(cube, 7)
    d = np.array([D("2024-08-01"), D("2024-09-15")])
    h = np.array([5, 40])
    W = 20
    s = wn.window_features(a, ["U0", "U1"], d, h, W, "sum", anomaly=True)
    assert list(s.columns) == ["v0_anom7_mean", "v0_anom7_mean10", "v0_anom7_max"]
    x = wn.sequences(a, ["U0", "U1"], d, h, W)
    for i in range(2):
        xi = x[i, :, 0]
        exp = [np.nanmean(xi), np.nanmean(xi[-10:]), np.nanmax(xi)]
        np.testing.assert_allclose(s.iloc[i].to_numpy(), exp)
    r = wn.window_features(a, ["U0", "U1"], d, h, W, "raw", anomaly=True)
    assert r.columns[0] == "v0_anom7_lag001" and r.columns[-1] == "v0_anom7_lag020"
    np.testing.assert_allclose(r["v0_anom7_lag001"], x[:, -1, 0], equal_nan=True)


def test_window_outside_table_is_nan_and_unknown_unit():
    cube = wn.DailyCube(D("2025-01-01"), ("A",), ("x",), np.ones((1, 30, 1)))
    seq = wn.sequences(cube, ["A"], "2025-01-05", 1, 10)  # S = 2024-12-27
    assert np.isnan(seq[0, :5, 0]).all() and (seq[0, 5:, 0] == 1).all()
    with pytest.raises(KeyError):
        wn.sequences(cube, ["B"], "2025-01-20", 1, 10)
    seq2 = wn.sequences(cube, ["B"], "2025-01-20", 1, 10, allow_missing_units=True)
    assert np.isnan(seq2).all()


def test_forecast_cube_splice_at_T():
    days = 60
    obs = wn.DailyCube(D("2025-01-01"), ("A",), ("spi",), np.zeros((1, days, 1)))
    fcst = wn.DailyCube(D("2025-01-01"), ("A",), ("spi",), np.ones((1, days, 1)))
    for h in (1, 5, 10, 30):
        b = wn.window_bounds("2025-02-20", h, 20)
        x = wn.gather_window(obs, ["A"], b, forecast_cube=fcst)[0, :, 0]
        fc = wn.forecast_day_mask(b)[0]
        assert (x[fc] == 1).all() and (x[~fc] == 0).all()


def test_from_long_roundtrip_and_duplicates():
    df = pd.DataFrame({"sgg_code": ["11010", "11010", "26010"],
                       "date": ["2025-01-01", "2025-01-03", "2025-01-02"],
                       "t": [1.0, 3.0, 2.0]})
    c = wn.DailyCube.from_long(df, "sgg_code", "date", ["t"])
    assert c.units == ("11010", "26010") and c.n_days == 3
    assert np.isnan(c.values[0, 1, 0])  # 표에 없는 날 = NaN(채우지 않음)
    with pytest.raises(ValueError):
        wn.DailyCube.from_long(pd.concat([df, df.iloc[:1]]), "sgg_code", "date", ["t"])


def test_summary_requires_multiple_of_10():
    with pytest.raises(ValueError):
        wn.summary_block(np.zeros((1, 15, 1)))


def test_rr8():
    d = to_days(["2015-04-07", "2015-05-27", "2015-05-29", "2015-05-30", "2015-04-07"])
    route = np.array(["domestic"] * 4 + ["overseas"])
    m = wn.rr8_mask(d, route)
    assert m.tolist() == [True, True, True, False, False]
    b = wn.window_bounds(d, 60, 100)
    assert (b.S == d - 149).all()


def test_from_long_rejects_null_unit_or_date():
    df = pd.DataFrame({"sgg_code": ["11010", None], "date": ["2025-01-01", "2025-01-02"], "t": [1.0, 2.0]})
    with pytest.raises(ValueError, match="null"):
        wn.DailyCube.from_long(df, "sgg_code", "date", ["t"])
    df2 = pd.DataFrame({"sgg_code": ["11010", "11010"], "date": ["2025-01-01", None], "t": [1.0, 2.0]})
    with pytest.raises(ValueError, match="null"):
        wn.DailyCube.from_long(df2, "sgg_code", "date", ["t"])


def test_window_bounds_accepts_datetime64_ns():
    d_ns = pd.to_datetime(["2025-11-01"]).to_numpy()  # datetime64[ns]
    b = wn.window_bounds(d_ns, 1, 60)
    assert b.E[0] == D("2025-11-01") and b.S[0] == D("2025-09-03")


def test_chunk_rows_default():
    import inspect
    assert inspect.signature(wn.window_features).parameters["chunk_rows"].default == 10_000
