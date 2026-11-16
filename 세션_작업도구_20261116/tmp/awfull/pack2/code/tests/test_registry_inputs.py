"""lst.registry·lst.inputs·s0_6_missing·s0_8_groups 시험(합성 자료)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import config as cf
from lst import folds as fd
from lst import history as hi
from lst import inputs as ip
from lst import registry as rg
from lst import static_feats as sf
from lst import windows as wn
from lst import wxfeat as wf

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "scripts"))

DOM = ("11110", "11140", "26110")
OV = ("OVCHN", "OVUSA")
START, END = "2015-01-01", "2017-12-31"


# ---------------------------------------------------------------- 합성 자료
def _weather(rng):
    dates = pd.date_range(START, END, freq="D")
    dom, ov = [], []
    for u in DOM:
        df = pd.DataFrame({"sgg_code": u, "date": dates})
        for v in wf.CUBE_VARS:
            x = rng.normal(10, 5, len(dates))
            if v == "rain":
                x = np.abs(x)
            x[rng.random(len(dates)) < 0.03] = np.nan          # 국내 결측 조금
            df[v] = x
        dom.append(df)
    for u in OV:
        df = pd.DataFrame({"alpha_3": u[2:], "sgg": u, "date": dates})
        for v in rg.WEATHER19:
            df[v] = np.nan if v in rg.STRUCT_VARS else rng.normal(12, 4, len(dates))
        df["avg_rhm"] = rng.normal(60, 10, len(dates))
        df["avg_td"] = np.nan                                   # DB 에 없음
        ov.append(df)
    dom = pd.concat(dom, ignore_index=True)
    dom["soil_t_nas"] = 1.0                                     # 국내 결측 없는 변수(해외는 구조적 결측)
    return dom, pd.concat(ov, ignore_index=True)


def _spi(rng):
    dates = pd.date_range(START, END, freq="D")
    rows = []
    for u in DOM:
        rows.append(pd.DataFrame({"sgg_code": u, "date": dates, "spi3": rng.normal(0, 1, len(dates)),
                                  "spi6": rng.normal(0, 1, len(dates))}))
    obs = pd.concat(rows, ignore_index=True)
    fc = obs.copy()
    fc["spi3"] = fc["spi3"] + 0.5
    ov = pd.concat([pd.DataFrame({"sgg": u, "alpha_3": u[2:], "date": dates, "spi3": rng.normal(0, 1, len(dates)),
                                  "spi6": rng.normal(0, 1, len(dates))}) for u in OV], ignore_index=True)
    return obs, fc, ov


P3 = ("A0101000000000", "A0102000000000", "B0204000000000", "코드없음")
P2 = {"A0101000000000": "A0100000000000", "A0102000000000": "A0100000000000", "B0204000000000": "B0200000000000",
      "코드없음": "A0100000000000"}


def _samples(rng, n=240):
    d = pd.to_datetime("2015-06-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    units = np.array(DOM + OV)[rng.integers(0, 5, n)]
    keys = [f"K{i:05d}" for i in range(n)]
    p3 = np.array(P3)[rng.integers(0, 4, n)]
    s = pd.DataFrame({"lims_data_key": keys, "y": (rng.random(n) < 0.15).astype("int8"),
                      "src_route": np.where(pd.Series(units).str.startswith("OV"), "overseas", "domestic"),
                      "sgg": units, "sido": [u[:2] for u in units], "p2": [P2[x] for x in p3], "p3": p3,
                      "d": d.date})
    from lst.hassign import assign_h_many
    s["h"] = assign_h_many(keys).astype(np.int16)
    s["y"] = s["y"].astype("Int8")
    return s


def _food():
    t = pd.DataFrame({"p3": ["A0101000000000", "A0102000000000", "B0204000000000"],
                      "food_aw": [0.98, np.nan, 0.95], "aw_label": ["단일값", None, "하위중앙"],
                      "food_ph": [6.5, 5.0, np.nan], "food_na": [1.0, 2.0, np.nan]})
    for k, v in sf.rg_item(t["food_aw"].to_numpy(), t["food_ph"].to_numpy()).items():
        t[k] = v
    return t


def _soil(rng):
    dates = pd.date_range("2014-06-01", END, freq="D")
    rows = []
    for u in DOM:
        rows.append(pd.DataFrame({"sgg_code": u, "date": dates, "soil_ph": rng.normal(6.5, 0.3, len(dates)),
                                  "soil_om": rng.normal(20, 3, len(dates)), "soil_vldpha": np.nan}))
    soil = pd.concat(rows, ignore_index=True)
    gph = pd.DataFrame({"iso3": ["CHN"], "ph_mean": [6.8]})
    return sf.SoilLookup(soil, gph)


def _indicator(name):
    units = list(DOM)
    rows = [(u, y, 0.1 * i + 0.01 * (y - 2010), f"{y + 2}-01-01") for i, u in enumerate(units) for y in range(2010, 2017)]
    return sf.YearIndicator.from_long(name, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                                      [r[3] for r in rows])


@pytest.fixture(scope="module")
def ctx():
    rng = np.random.default_rng(7)
    dom, ov = _weather(rng)
    obs, fc, sov = _spi(rng)
    s = _samples(rng)
    c = ip.InputContext(samples=s, y_cut=fd.day("2025-06-30"))
    c.wx = wf.combined_cube(dom, ov)
    c.spi_obs, c.spi_fc = ip.spi_cubes(obs, fc, sov)
    c.food = _food()
    c.livestock = pd.DataFrame({"unit": ["11110", "OVCHN"], "L1": [3.5, 0.2]})
    c.soil = _soil(rng)
    c.eco = _indicator("eco_ratio")
    c.gw = _indicator("gw_pass_rate")
    c.missing = rg.MissingLists({1: frozenset(rg.WEATHER19), 2: frozenset(
        list(rg.WEATHER19) + ["spi3", "food_aw", "food_ph", "hist_rate5y_p3", "td1", "L1"] + list(rg.RG_SOILT))})
    return c


def _meta(ctx, n=None, h=None):
    s = ctx.samples
    m = s[list(ip.META_REQUIRED)].copy()
    if h is not None:
        m["h"] = np.int16(h)
    return m.iloc[:n].reset_index(drop=True) if n else m.reset_index(drop=True)


ALL2 = dict(include_ids=rg.ALL_IDS, k=50, n_ma=15, n_ma_dl=7)


# ---------------------------------------------------------------- 등록부
def test_registry_counts_and_groups():
    assert len(rg.UNITS) == 110 and len(rg.GROUPS) == 44
    g = rg.group_table()
    assert g["weather"].sum() == 9
    assert g.loc[g["weather"], "n_units"].tolist() == [27, 9, 7, 5, 16, 2, 2, 6, 1]
    u = rg.unit_table()
    assert u["n_bundles"].sum() == 113 and (u["n_bundles"] == 2).sum() == 3
    # 비기상 35 = 이력 19·식품 7·토양 6·외부 2·축사 1
    nw = u[u["group_no"] > 9]
    assert nw["id"].isin(["HIS1", "HIS2", "HIS3", "HIS4"]).sum() == 19
    assert nw["unit"].tolist()[19:26] == ["food_aw", "food_ph", "food_na", "rg_aw_lim", "rg_aw_opt_dist",
                                         "rg_ph_lim", "rg_ph_opt_dist"]
    assert nw["unit"].tolist()[-3:] == ["eco_ratio", "gw_pass_rate", "L1"]


def test_registry_names_match_feature_modules():
    spec = rg.VersionSpec("x", 2, 30, "sum", **ALL2)
    names = [c.name for c in rg.tree_columns(spec)]
    for n in wf.WX3_NAMES + wf.WX4_WX_NAMES + wf.WX4_SOIL_NAMES + wf.WX5_NAMES + wf.WX6_NAMES:
        assert n in names
    for c in hi.his_columns([50]):
        assert c["column"] in names
    assert wn.summary_names(["avg_temp"], 30)[0] in names
    assert "avg_temp_anom15_mean" in names and "spi6_slope" in names
    # 1차 칼럼 수: 범주 5 + h·g + 19 × (W/10 + 5)
    s1 = rg.VersionSpec("y", 1, 60, "sum", include_ids=("WEATHER",))
    assert len(rg.tree_columns(s1)) == 7 + 19 * 11
    s1r = rg.VersionSpec("y", 1, 60, "raw", include_ids=("WEATHER",))
    assert len(rg.tree_columns(s1r)) == 7 + 19 * 60


def test_version_validation():
    with pytest.raises(rg.RegistryError):
        rg.VersionSpec("a", 1, 60, "sum", include_ids=("WEATHER", "HIS1"), k=50)
    with pytest.raises(rg.RegistryError):
        rg.VersionSpec("a", 2, 60, "sum", include_ids=("HIS1",))           # k 없음
    with pytest.raises(rg.RegistryError):
        rg.VersionSpec("a", 2, 60, "sum", include_ids=("WX2",), n_ma=6)
    with pytest.raises(rg.RegistryError):
        rg.VersionSpec("a", 2, 60, "sum", include_ids=("WEATHER",), tree_layout="seq_only")
    with pytest.raises(rg.RegistryError):
        rg.VersionSpec("a", 2, 60, "raw", include_ids=("WEATHER", "L1"), dl_layout="seq_only")
    with pytest.raises(cf.ConfigError):
        rg.load_version({"name": "a", "order": 1, "W": 60, "mode": "sum", "include_ids": ["WEATHER"], "bogus": 1})
    v = rg.load_version(CODE / "configs" / "versions" / "v2_all_example.json")
    assert v.k == 50 and len(v.units()) == 110
    for p in (CODE / "configs" / "versions").glob("*.json"):
        rg.load_version(p)


def test_exclusions_and_research():
    s = rg.VersionSpec("a", 2, 60, "sum", exclude_groups=("기온", "계절"), exclude_units=("L1", "wind_dir"), **ALL2)
    names = {u.name for u in s.units()}
    assert "avg_temp" not in names and "season" not in names and "L1" not in names and "wind_dir" not in names
    assert "wind_dir_anom" in names
    r = rg.VersionSpec("b", 2, 60, "sum", include_ids=("WEATHER",), include_units=("t3",))
    assert [u.name for u in r.units() if u.kind == "table"] == ["t3"]
    w8 = rg.VersionSpec("c", 2, 60, "sum", exclude_groups=rg.WEATHER8_GROUPS, **ALL2)
    assert {u.group for u in w8.units() if rg.group(u.group).weather} == {"계절"}


def test_membership_table_and_structural():
    ml = rg.MissingLists({1: frozenset(), 2: frozenset(["avg_temp", "food_aw"])})
    m = rg.membership_table(20, missing=ml)
    assert set(m["where"]) == {"tree_sum", "tree_raw", "dl_seq", "dl_static"}
    assert (m["column"] == "hist_rate5y_p3_{K}").any() and (m["column"] == "avg_temp_anom{N}_mean").any()
    assert (m["column"] == "avg_temp_lag020__na").any() and (m["column"] == "food_aw__na").any()
    assert not (m["column"] == "aw_label__na").any()
    st = m[m["structural"]]
    assert set(st["source"]) == set(rg.STRUCT_VARS) | {"td1"} | set(rg.RG_SOILT)     # rg 지중온도 6 포함(11-07 결정)
    assert m.loc[m["unit"] == "rg_soilt_opt_mean", "structural"].all()
    assert set(rg.structural_sources(2)) == set(rg.STRUCT_VARS) | {"td1"} | set(rg.RG_SOILT)
    assert rg.structural_sources(1) == rg.STRUCT_VARS


# ---------------------------------------------------------------- 트리 입력
def test_tree_X_order1_matches_windows(ctx):
    spec = rg.VersionSpec("1", 1, 30, "sum", include_ids=("WEATHER",))
    meta = _meta(ctx)
    X = ip.build_tree_X(meta, ctx, spec)
    assert list(X.columns) == [c.name for c in rg.tree_columns(spec)]
    assert X["sgg"].dtype == object and X["avg_temp_mean"].dtype == np.float32
    ref = wn.window_features(ctx.wx.select(list(rg.WEATHER19)), meta["sgg"], fd.day_ints(meta["d"]),
                             meta["h"].to_numpy(), 30, "sum")
    np.testing.assert_allclose(X[ref.columns].to_numpy(np.float64), ref.to_numpy(), rtol=1e-6, equal_nan=True)
    b = wn.window_bounds(fd.day_ints(meta["d"]), meta["h"].to_numpy(), 30)
    assert np.array_equal(X["g"].to_numpy(), b.g.astype(np.float32))
    # 해외 행 구조적 결측 변수 = NaN 그대로(채우지 않음)
    ov = (meta["src_route"] == "overseas").to_numpy()
    assert X.loc[ov, "soil_t5_mean"].isna().all()


def test_tree_X_order2_columns_and_values(ctx):
    spec = rg.VersionSpec("2", 2, 20, "raw", **ALL2)
    meta = _meta(ctx, h=5)
    split = fd.SPLITS["fold1"]
    X = ip.build_tree_X(meta, ctx, spec, split=split)
    assert list(X.columns) == [c.name for c in rg.tree_columns(spec)]
    assert set(X["aw_label"]) <= set(rg.AW_LABEL_VALUES)
    nc = (meta["p3"] == "코드없음").to_numpy()
    assert (X.loc[nc, "aw_label"] == "결측").all() and X.loc[nc, "food_aw"].isna().all()
    assert set(X["season"]) <= set(sf.SEASONS)
    # 이력 = history_features 직접 계산과 같음
    pop = hi.Population.from_samples(ctx.samples[fd.day_ints(ctx.samples["d"]) <= split.train_end])
    T = fd.day_ints(meta["d"]) - 5
    hr = pd.DataFrame({"d": fd.day_ints(meta["d"]), "C": hi.c_from_T(T), "p2": meta["p2"], "p3": meta["p3"],
                       "sgg": meta["sgg"], "sido": meta["sido"]})
    H = hi.history_features(pop, hr, ks=[50])
    np.testing.assert_allclose(X[H.columns].to_numpy(np.float64), H.to_numpy(), rtol=1e-6, equal_nan=True)
    # 차감 lag = 차감 일자료 E 날 값
    a = wn.window_features(ctx.anomaly(15).select(["rain_anom15"]), meta["sgg"], fd.day_ints(meta["d"]),
                           meta["h"].to_numpy(), 20, "raw", anomaly=True)
    np.testing.assert_allclose(X["rain_anom15_lag001"].to_numpy(np.float64), a["rain_anom15_lag001"], rtol=1e-6,
                               equal_nan=True)
    # 축사: 표에 없는 단위 NaN
    assert X.loc[(meta["sgg"] == "26110").to_numpy(), "L1"].isna().all()
    # SPI 예보 칸(h 5 → E = d, T ~ E 예보) = 예보 표 값(국내 spi3 예보 = 관측 + 0.5)
    dom = (meta["src_route"] == "domestic").to_numpy()
    b = wn.window_bounds(fd.day_ints(meta["d"]), 5, 20)
    exp = ctx.spi_fc.values[ctx.spi_fc.unit_index(meta["sgg"][dom]), b.E[dom] - ctx.spi_fc.start, 0]
    np.testing.assert_allclose(X.loc[dom, "spi3_lag001"].to_numpy(np.float64), exp, rtol=1e-6)


def test_tree_X_leak_guard_and_exclusion(ctx):
    spec = rg.VersionSpec("2", 2, 20, "sum", **ALL2)
    meta = _meta(ctx, 30)
    with pytest.raises(hi.LeakageError):
        ip.build_tree_X(meta, ctx, spec, split=fd.Split("t", fd.day("2015-06-30"), 0, 0))
    s2 = spec.with_(exclude_groups=("습도",), exclude_units=("food_aw", "soil_ph"))
    X = ip.build_tree_X(meta, ctx, s2)
    assert "rh1" not in X.columns and "aw_label" not in X.columns and "soil_ph" not in X.columns
    assert "rg_soilph_opt" in X.columns and "gr2_mu_mean" in X.columns   # 계산엔 FD1·SO1 값을 씀
    lims = rg.VersionSpec("e", 2, 20, "sum", include_ids=())
    assert list(ip.build_tree_X(meta, ctx, lims).columns) == list(rg.CAT5) + ["h", "g"]
    s7 = rg.VersionSpec("7T", 2, 20, "raw", include_ids=("WEATHER", "WX1", "WX2"), n_ma=7, tree_layout="seq_only")
    X7 = ip.build_tree_X(meta, ctx, s7)
    assert X7.columns[0] == "g" and "h" not in X7.columns and "sgg" not in X7.columns
    assert X7.shape[1] == 1 + (19 + 19 + 2) * 20


def test_meta_bounds_check(ctx):
    spec = rg.VersionSpec("1", 1, 30, "sum", include_ids=("WEATHER",))
    meta = _meta(ctx, 10)
    b = wn.window_bounds(fd.day_ints(meta["d"]), meta["h"].to_numpy(), 30)
    ok = meta.assign(T=fd.as_date(b.T), E=fd.as_date(b.E), S=fd.as_date(b.S), g=b.g.astype(np.int16),
                     R=fd.as_date(b.R))
    ip.build_tree_X(ok, ctx, spec)
    bad = ok.assign(S=fd.as_date(b.S + 1))
    with pytest.raises(ip.InputError):
        ip.build_tree_X(bad, ctx, spec)


# ---------------------------------------------------------------- 딥러닝 입력
def test_dl_order1_layout(ctx):
    spec = rg.VersionSpec("1L", 1, 20, "sum", include_ids=("WEATHER",))
    meta = _meta(ctx)
    raw = ip.build_dl_raw(meta, ctx, spec)
    prep = ip.DLPreprocessor(spec, ctx.missing, ctx.vocab).fit(raw)
    out = prep.transform(raw)
    assert out["x_seq"].shape == (len(meta), 20, 38) and out["x_seq"].dtype == np.float32
    assert prep.static_names == ["g_div50", "src_route01", rg.STRUCT_COLUMN]
    assert out["x_cat"].dtype == np.int64 and out["x_cat"].shape == (len(meta), 4)
    assert not np.isnan(out["x_seq"]).any()
    ov = (meta["src_route"] == "overseas").to_numpy()
    np.testing.assert_array_equal(out["x_static"][:, 1], ov.astype(np.float32))
    np.testing.assert_array_equal(out["x_static"][:, 2], ov.astype(np.float32))
    b = wn.window_bounds(fd.day_ints(meta["d"]), meta["h"].to_numpy(), 20)
    np.testing.assert_allclose(out["x_static"][:, 0], b.g / 50.0, rtol=1e-6)
    names = prep.seq_channel_names
    # 구조적 결측 채널: 해외 행 값 = 0, 표시 = 0 / 국내 결측 = 표시 1
    j = names.index("soil_t5__na")
    assert out["x_seq"][ov][:, :, j].sum() == 0
    jj = names.index("avg_temp__na")
    vi = raw.seq_names.index("avg_temp")
    np.testing.assert_array_equal(out["x_seq"][:, :, jj], np.isnan(raw.seq[:, :, vi]).astype(np.float32))
    # 표준화: train 평균 0
    z = out["x_seq"][:, :, vi][~np.isnan(raw.seq[:, :, vi])]
    assert abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3
    # 전부 결측 채널(해외만 있는 행 묶음이 아니라 전 행 결측은 없음) → 해외만 넣으면 soil_t5 전부 결측 = 0
    rov = ip.build_dl_raw(meta[ov].reset_index(drop=True), ctx, spec)
    p2 = ip.DLPreprocessor(spec, ctx.missing, ctx.vocab).fit(rov)
    assert p2.seq_zero[rov.seq_names.index("soil_t5")]
    assert prep.embed_dims == [min(50, (len(ctx.vocab[c]) + 2) // 2) for c in rg.CAT4_DL]


def test_dl_order2_static_onehot_oov_and_chunks(ctx, tmp_path):
    spec = rg.VersionSpec("3-0-LSTM", 2, 20, "sum", **ALL2)
    meta = _meta(ctx)
    split = fd.SPLITS["fold1"]
    raw = ip.build_dl_raw(meta, ctx, spec, split=split)
    assert raw.seq_names[:19] == list(rg.WEATHER19) and "rain_anom7" in raw.seq_names and raw.seq_names[-2:] == ["spi3", "spi6"]
    tr = meta["sgg"] != "OVUSA"                                    # train 에 OVUSA 없음 → OOV
    rtr = ip.build_dl_raw(meta[tr].reset_index(drop=True), ctx, spec, split=split)
    prep = ip.DLPreprocessor(spec, ctx.missing, ctx.vocab).fit(rtr)
    out = prep.transform(raw)
    sg = out["x_cat"][:, 1]
    assert (sg[(meta["sgg"] == "OVUSA").to_numpy()] == 0).all() and (sg[tr.to_numpy()] > 0).all()
    sn = prep.static_names
    for s in sf.SEASONS:
        assert f"season_{s}" in sn
    oh = out["x_static"][:, [sn.index(f"season_{s}") for s in sf.SEASONS]]
    np.testing.assert_array_equal(oh.sum(1), 1.0)
    aw = out["x_static"][:, [sn.index(f"aw_label_{v}") for v in rg.AW_LABEL_VALUES]]
    np.testing.assert_array_equal(aw.sum(1), 1.0)
    assert "food_aw__na" in sn and "td1__na" in sn and "L1__na" in sn
    ov = (meta["src_route"] == "overseas").to_numpy()
    assert (out["x_static"][ov, sn.index("td1__na")] == 0).all()           # 구조적 → 해외 0
    for c in rg.RG_SOILT:                                                   # rg 지중온도 6 도 구조적(11-07)
        assert (out["x_static"][ov, sn.index(c + "__na")] == 0).all() and (out["x_static"][ov, sn.index(c)] == 0).all()
    assert (out["x_static"][:, sn.index(rg.STRUCT_COLUMN)] == ov).all()
    # 묶음 fit = 한 번 fit
    pc = ip.DLPreprocessor(spec, ctx.missing, ctx.vocab)
    for lo in range(0, int(tr.sum()), 37):
        pc.partial_fit(ip.build_dl_raw(meta[tr].reset_index(drop=True).iloc[lo:lo + 37], ctx, spec, split=split))
    pc.finalize()
    np.testing.assert_allclose(pc.seq_mean, prep.seq_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(pc.seq_std, prep.seq_std, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(pc.st_std, prep.st_std, rtol=1e-9, atol=1e-9)
    # 저장·불러오기
    prep.save(tmp_path / "p.json")
    p2 = ip.DLPreprocessor.load(tmp_path / "p.json")
    o2 = p2.transform(raw)
    for k in out:
        np.testing.assert_array_equal(out[k], o2[k])
    # 조립 객체(runner 꼴)
    asm = ip.DLInputs(ctx, spec, "fold1", chunk_rows=50).fit(meta[tr].reset_index(drop=True))
    o3 = asm.transform(meta)
    np.testing.assert_allclose(o3["x_seq"], out["x_seq"], atol=1e-5)
    assert asm.cat_sizes == prep.cat_cardinalities
    asm.save(tmp_path / "inputs_fold1")


def test_dl_seq_only(ctx):
    spec = rg.VersionSpec("6", 2, 20, "raw", include_ids=("WEATHER", "WX1", "WX2"), n_ma=7, n_ma_dl=7,
                          dl_layout="seq_only")
    meta = _meta(ctx, 50)
    raw = ip.build_dl_raw(meta, ctx, spec)
    prep = ip.DLPreprocessor(spec, ctx.missing, None).fit(raw)
    out = prep.transform(raw)
    assert set(out) == {"x_seq", "x_static"} and prep.static_names == ["g_div50", rg.STRUCT_COLUMN]


# ---------------------------------------------------------------- 표 전처리(로지스틱·표 AE)
def test_tabular_scaler(ctx):
    spec = rg.VersionSpec("2", 2, 20, "sum", **ALL2)
    meta = _meta(ctx)
    X = ip.build_tree_X(meta, ctx, spec)
    sc = ip.TabularScaler(rg.tree_columns(spec), ctx.missing, 2).fit(X)
    Z = sc.transform(X)
    assert not Z[sc.num].isna().any().any()
    assert Z["sgg"].equals(X["sgg"])
    ov = (X["src_route"] == "overseas").to_numpy()
    assert (Z[rg.STRUCT_COLUMN] == ov).all()
    assert (Z.loc[ov, "soil_t5_mean__na"] == 0).all()
    np.testing.assert_array_equal(Z["avg_temp_mean__na"].to_numpy(), X["avg_temp_mean"].isna().to_numpy())
    # rg 지중온도 6칼럼도 구조적 대상 → 해외 행 칼럼별 표시 0, ov_struct_na = 1(해외 행은 실제로 NaN)
    for c in rg.RG_SOILT:
        assert X.loc[ov, c].isna().all()
        assert (Z.loc[ov, c + "__na"] == 0).all() and (Z.loc[ov, c] == 0).all()
        np.testing.assert_array_equal(Z.loc[~ov, c + "__na"].to_numpy(), X.loc[~ov, c].isna().to_numpy())
    assert (Z.loc[ov, rg.STRUCT_COLUMN] == 1).all() and (Z.loc[~ov, rg.STRUCT_COLUMN] == 0).all()
    m = Z["avg_temp_mean"].to_numpy()[X["avg_temp_mean"].notna().to_numpy()]
    assert abs(m.mean()) < 1e-4
    with pytest.raises(ip.InputError):
        ip.TabularScaler(rg.tree_columns(spec), ctx.missing, 2).fit(X.assign(bogus=1.0))
    t = ip.TreeInputs(ctx, spec, None, scale=True).fit(meta)
    Z2 = t.transform(meta)
    assert list(Z2.columns) == t.columns


# ---------------------------------------------------------------- 결측 점검 도구·스크립트
def test_range_nan_counts():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(2, 30, 3))
    v[rng.random(v.shape) < 0.3] = np.nan
    cube = wn.DailyCube(100, ("a", "b"), ("x", "y", "z"), v)
    codes = np.array(["a", "b", "a", "b"])
    lo = np.array([95, 100, 110, 125])
    hi_ = np.array([105, 129, 140, 126])
    got = ip.range_nan_counts(cube, codes, lo, hi_)
    for i in range(4):
        u = 0 if codes[i] == "a" else 1
        for j in range(3):
            exp = sum(1 for t in range(lo[i], hi_[i] + 1)
                      if not (100 <= t < 130) or np.isnan(v[u, t - 100, j]))
            assert got[i, j] == exp


def test_s0_6_missing_compute(ctx, tmp_path):
    import s0_6_missing as sm

    cfg = {"plan": "t", "context": {"y_cut": "2025-06-30"}, "n_ma_values": list(rg.N_MA_VALUES), "h_values": "own",
           "table_W": 10, "table_k": 10, "max_samples": 0, "sample_seed": 0}
    res, det = sm.compute(ctx, cfg, log=lambda *a: None)
    o1, o2 = set(res["order1"]["variables"]), set(res["order2"]["variables"])
    assert o1 <= set(rg.WEATHER19) and o1 <= o2
    assert "avg_temp" in o1                       # 국내 결측 있음
    assert "soil_t_nas" not in o1                 # 국내 결측 없음, 해외 결측은 구조적(목록 판정에서 뺌)
    assert "hist_rate5y_p3" in o2                 # C < 2015-04-07 행
    assert "food_aw" in o2 and "L1" in o2
    assert "td1" in o2 and "spi3" not in o2 and "soil_t_nas" not in o2
    rs = det[det["variable"].isin(rg.RG_SOILT)]
    assert len(rs) == 6 and rs["structural_order2"].all() and (rs["n_ov_nan"] == rs["n_ov"]).all()
    assert res["order2"]["structural_variables"] == list(rg.structural_sources(2))
    cf.dump_json(tmp_path / "m.json", cf.nan_to_null(res))
    ml = rg.MissingLists.from_json(tmp_path / "m.json")        # 구조적 변수 목록 대조 통과
    assert ml.lists[1] == frozenset(o1) and ml.lists[2] == frozenset(o2)


def test_s0_6_structural_mismatch(ctx):
    import s0_6_missing as sm

    bad = ip.InputContext(samples=ctx.samples, y_cut=ctx.y_cut)
    v = ctx.wx.values.copy()
    ou = [i for i, u in enumerate(ctx.wx.units) if u.startswith("OV")]
    v[ou[0], 5, ctx.wx.variables.index("no2")] = 1.0
    bad.wx = wn.DailyCube(ctx.wx.start, ctx.wx.units, ctx.wx.variables, v)
    with pytest.raises(sm.StructuralMismatch):
        sm.structural_check(bad)


def test_s0_8_tables_and_corr(ctx):
    import s0_8_groups as s8

    cfg = {"W_list": [10, 30], "corr": {"W": 30, "k": 50, "n_ma": 15, "split": "fold1", "min_periods": 5,
                                        "max_rows": 0, "seed": 0}}
    g, u, mem, chk = s8.tables(cfg, ctx.missing)
    assert chk["units"] == 110 and set(mem["W"]) == {10, 30}
    corr, grp, rec = s8.correlation(ctx, cfg)
    assert rec["rows"] > 0 and "avg_temp" in corr["unit"].tolist()
    assert len(grp) == 43 * 42 // 2                  # 계절(범주 1단위) 그룹 제외


def test_configs_load():
    import s0_6_missing as sm
    import s0_8_groups as s8

    sm.load_cfg(CODE / "configs" / "s0_6_missing.json")
    s8.load_cfg(CODE / "configs" / "s0_8_groups.json")
