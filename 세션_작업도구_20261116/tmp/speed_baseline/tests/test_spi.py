"""SPI(lst/spi.py)·단계 0-4 SPI 스크립트 시험 — 합성 자료만."""
import csv
import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import stats
from scipy.special import ndtri

from lst import kweather as kw
from lst import spi as sp
from lst import windows as wn
from lst.folds import to_days

FINAL = Path(__file__).resolve().parents[1]


def D(s):
    return int(to_days(s)[0])


def iso(day):
    return str(np.datetime64(int(day), "D"))


# ---------------------------------------------------------------- 날짜 규칙
@pytest.mark.parametrize("t,n,start", [
    ("2015-03-31", 3, "2015-01-01"),   # 계획서 첫 값 날짜(SPI3)
    ("2015-06-30", 6, "2015-01-01"),   # 계획서 첫 값 날짜(SPI6)
    ("2015-03-30", 3, "2014-12-31"),
    ("2015-06-29", 6, "2014-12-30"),
    ("2015-05-30", 3, "2015-02-28"),   # 02-31 없음 → 말일
    ("2016-05-30", 3, "2016-02-29"),   # 윤년
    ("2015-02-28", 3, "2014-12-01"),   # 월말 → 꼭 3개 달
    ("2016-02-29", 6, "2015-09-01"),
    ("2015-12-31", 6, "2015-07-01"),
    ("2015-01-15", 3, "2014-10-16"),
])
def test_window_start(t, n, start):
    assert sp.window_start(D(t), n)[0] == D(start)


def test_calendar_day():
    cases = {"2015-01-01": 1, "2015-02-28": 59, "2015-03-01": 60, "2015-12-31": 365,
             "2016-02-28": 59, "2016-02-29": 59, "2016-03-01": 60, "2016-12-31": 365, "2000-02-29": 59, "1900-03-01": 60}
    got = sp.calendar_day(np.array([D(k) for k in cases]))
    assert got.tolist() == list(cases.values())
    allc = sp.calendar_day(np.arange(D("2015-01-01"), D("2026-12-31") + 1))
    assert allc.min() == 1 and allc.max() == 365 and len(np.unique(allc)) == 365


# ---------------------------------------------------------------- 누적
def test_accumulate_matches_bruteforce():
    rng = np.random.default_rng(1)
    d0 = D("2015-01-01")
    n_days = 500
    P = np.round(rng.gamma(0.4, 8, size=(3, n_days)), 1)
    P[rng.random(P.shape) < 0.5] = 0.0
    P[1, 200] = np.nan
    P[2, 100:110] = 0.0
    for n in (3, 6):
        X = sp.accumulate(P, d0, n)
        for u in range(3):
            for i in range(n_days):
                s = int(sp.window_start(d0 + i, n)[0]) - d0
                if s < 0:
                    ref = np.nan
                else:
                    w = P[u, s:i + 1]
                    ref = np.nan if np.isnan(w).any() else float(w.sum())
                if np.isnan(ref):
                    assert np.isnan(X[u, i]), (n, u, i)
                else:
                    assert X[u, i] == pytest.approx(ref, abs=1e-9), (n, u, i)
                    if ref == 0:
                        assert X[u, i] == 0.0
    # 계획서 첫 값 날짜(강수 2015-01-01 부터)
    assert sp.first_valid_day(sp.accumulate(np.ones((1, 400)), d0, 3), d0) == D("2015-03-31")
    assert sp.first_valid_day(sp.accumulate(np.ones((1, 400)), d0, 6), d0) == D("2015-06-30")


def test_accumulate_exact_zero_and_negative():
    d0 = D("2015-01-01")
    P = np.full((1, 200), 1 / 3)      # 3곳 평균처럼 딱 떨어지지 않는 값
    P[0, 100:] = 0.0
    X = sp.accumulate(P, d0, 3)
    assert (X[0, 195:] == 0.0).all()  # 기간 안 양수 날 없음 → 정확히 0
    with pytest.raises(ValueError):
        sp.accumulate(np.array([[1.0, -0.1]]), d0, 3)


# ---------------------------------------------------------------- 감마 적합·변환
def _synthetic_accum(rng, d0, n_days, q=0.0, shape=2.0, scale=30.0):
    X = rng.gamma(shape, scale, size=(1, n_days))
    X[rng.random(X.shape) < q] = 0.0
    return X


def test_fit_gamma_matches_reference_formula():
    rng = np.random.default_rng(2)
    d0 = D("2015-01-01")
    n_days = D("2026-06-30") - d0 + 1
    days = d0 + np.arange(n_days)
    X = _synthetic_accum(rng, d0, n_days, q=0.1)
    X[0, :80] = np.nan
    fm = sp.fit_mask_of(days, D("2015-01-01"), D("2024-12-31"))
    fit = sp.fit_gamma(X, days, fm, 15)
    cal = sp.calendar_day(days)
    for c in (1, 59, 60, 200, 365):
        dist = np.minimum(np.abs(cal - c), 365 - np.abs(cal - c))
        x = X[0, (dist <= 15) & fm]
        x = x[~np.isnan(x)]
        pos = x[x > 0]
        A = np.log(pos.mean()) - np.log(pos).mean()
        al = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A)
        j = c - 1
        assert fit.n_pool[0, j] == len(x) and fit.n_zero[0, j] == (x == 0).sum() and fit.n_pos[0, j] == len(pos)
        assert fit.q[0, j] == pytest.approx((x == 0).mean())
        assert fit.alpha[0, j] == pytest.approx(al) and fit.beta[0, j] == pytest.approx(pos.mean() / al)
        # Thom 근사가 최우추정과 가까운지(참고)
        a_mle, _, _ = stats.gamma.fit(pos, floc=0)
        assert fit.alpha[0, j] == pytest.approx(a_mle, rel=0.02)
    # 1/1 묶음은 12/17 ~ 1/16(원형), 2/29 는 59 묶음에 듦
    cols = sp.pool_columns(cal, fm, 1, 15)
    assert set(np.unique(cal[cols])) == set(range(351, 366)) | set(range(1, 17))
    cols = sp.pool_columns(cal, np.ones_like(fm), 44, 15)
    assert D("2016-02-29") in set(days[cols])


def test_transform_standard_normal_and_zero():
    rng = np.random.default_rng(3)
    d0 = D("2015-01-01")
    n_days = D("2026-06-30") - d0 + 1
    days = d0 + np.arange(n_days)
    X = _synthetic_accum(rng, d0, n_days, q=0.15)
    fm = np.ones(n_days, bool)
    fit = sp.fit_gamma(X, days, fm, 15)
    Z, n_edge = sp.transform(X, days, fit)
    assert n_edge == 0 and not np.isnan(Z).any()
    # 0 이 없는 감마 자료면 표준정규에 가까움(0 이 있으면 0 날이 모두 Φ⁻¹(q) 한 점이라 평균 0 이 아님)
    X1 = _synthetic_accum(rng, d0, n_days, q=0.0)
    Z1, _ = sp.transform(X1, days, sp.fit_gamma(X1, days, fm, 15))
    assert abs(Z1.mean()) < 0.05 and abs(Z1.std() - 1) < 0.05
    j = sp.calendar_day(days) - 1
    z0 = X[0] == 0
    assert np.allclose(Z[0, z0], ndtri(fit.q[0, j[z0]]))      # 0 강수 = Φ⁻¹(q)
    # 같은 달력일 안에서 순서 보존(누적이 크면 SPI 도 큼)
    m = (j == 100) & ~z0
    o = np.argsort(X[0, m])
    assert (np.diff(Z[0, m][o]) >= 0).all()


def test_degenerate_pools_give_nan():
    d0 = D("2015-01-01")
    n_days = 3 * 365
    days = d0 + np.arange(n_days)
    X = np.zeros((4, n_days))
    X[1] = np.nan                                     # no_data
    X[2, ::400] = 5.0                                 # 양수 너무 적음 → 대부분 degenerate
    X[3] = np.random.default_rng(4).gamma(2, 10, n_days)
    fm = np.ones(n_days, bool)
    fit = sp.fit_gamma(X, days, fm, 15)
    assert (fit.status[0] == sp.STATUS.index("no_positive")).all()
    assert (fit.status[1] == sp.STATUS.index("no_data")).all()
    assert (fit.status[2] == sp.STATUS.index("degenerate")).any()
    assert (fit.status[3] == 0).all()
    Z, n_edge = sp.transform(X, days, fit)
    assert np.isnan(Z[:3]).all() and not np.isnan(Z[3]).any()
    # 적합 기간 밖 0 강수인데 q = 0 → H = 0 → NaN(개수 기록)
    X2 = X[3:4].copy()
    fm2 = days < d0 + 2 * 365
    X2[0, -1] = 0.0
    fit2 = sp.fit_gamma(X2, days, fm2, 15)
    Z2, n_edge2 = sp.transform(X2, days, fit2)
    assert np.isnan(Z2[0, -1]) and n_edge2 == 1


def test_fit_period_only_uses_fit_days():
    rng = np.random.default_rng(5)
    d0 = D("2015-01-01")
    n_days = 6 * 365
    days = d0 + np.arange(n_days)
    X = _synthetic_accum(rng, d0, n_days)
    fm = sp.fit_mask_of(days, d0, d0 + 4 * 365)
    f1 = sp.fit_gamma(X, days, fm, 15)
    X2 = X.copy()
    X2[0, ~fm] *= 10                                   # 적합 기간 밖 값을 바꿔도 모수 그대로
    f2 = sp.fit_gamma(X2, days, fm, 15)
    assert np.array_equal(f1.alpha, f2.alpha) and np.array_equal(f1.q, f2.q)


def test_spi_from_rain_first_valid():
    rng = np.random.default_rng(6)
    d0 = D("2015-01-01")
    n_days = D("2017-12-31") - d0 + 1
    P = np.round(rng.gamma(0.5, 6, size=(2, n_days)) * (rng.random((2, n_days)) < 0.4), 1)
    for n, first in ((3, "2015-03-31"), (6, "2015-06-30")):
        r = sp.spi_from_rain(P, d0, n, d0, D("2017-06-30"))
        assert sp.first_valid_day(r.spi, d0) == D(first)
        assert np.isnan(r.spi[:, :D(first) - d0]).all()


def test_agreement():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.5, 2.0, 3.0, np.nan])
    g = sp.agreement(a, b)
    assert g["n"] == 2 and g["mae"] == pytest.approx(0.25) and g["r"] == pytest.approx(1.0)
    assert sp.agreement([np.nan], [1.0])["n"] == 0


# ---------------------------------------------------------------- 스크립트(합성 뿌리)
D_MIN, D_MAX = "2015-01-01", "2016-12-31"
KMA_MIN = "2014-12-01"
N_ST = 6
SGG = ["11110", "26110", "41590"]
SGG_XY = {"11110": (37.57, 126.98), "26110": (35.10, 129.03), "41590": (37.17, 126.88)}


def _w(path: Path, table: pa.Table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def build_root(root: Path, inputs: Path, rng):
    raw = root / "리스테리아ML2" / "data" / "raw"
    dates = np.arange(np.datetime64(D_MIN), np.datetime64(D_MAX) + np.timedelta64(1, "D"), dtype="datetime64[D]")
    xy = np.column_stack([34.5 + 3.5 * rng.random(N_ST), 126.0 + 3.0 * rng.random(N_ST)])
    rows = {"stn_id": [], "meas_date": [], "sum_rn": []}
    for s in range(N_ST):
        for d in dates:
            if rng.random() < 0.03:
                continue                                              # 행 없는 날
            rows["stn_id"].append(str(90 + s))
            rows["meas_date"].append(d)
            rows["sum_rn"].append(None if rng.random() < 0.6 else float(np.round(rng.gamma(0.6, 8), 1)))  # 빈 값 = 0mm
    _w(raw / "raw_kma_asos_daily_info.parquet",
       pa.table({"stn_id": rows["stn_id"], "meas_date": pa.array(np.array(rows["meas_date"]), pa.date32()),
                 "sum_rn": pa.array(rows["sum_rn"], pa.float64())}))
    _w(raw / "raw_kma_ground_obs_station_info.parquet",
       pa.table({"stn_id": [str(90 + s) for s in range(N_ST)], "lat": xy[:, 0], "lon": xy[:, 1],
                 "oper_start_dt": pa.array([dt.date(2000, 1, 1)] * N_ST, pa.date32())}))
    # 기상청 SPI(앞 5지점만, 2014-12 부터, 빈 칸·빠진 날 포함), 2014 포털 표기 '.766'
    ext = root / "리스테리아ML2" / "data" / "외부_데이터"
    ext.mkdir(parents=True, exist_ok=True)
    kd = np.arange(np.datetime64(KMA_MIN), np.datetime64("2016-10-06") + np.timedelta64(1, "D"), dtype="datetime64[D]")
    with open(ext / "CLM_SPI_DD_합본_20140101_20261006.csv", "w", encoding="cp949", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["지점", "지점명", "일시", "SPI1", "SPI3", "SPI6", "SPI12"])
        for d in kd:
            if str(d) == "2016-03-03":
                continue                                              # 전 지점 빠진 날
            for s in range(N_ST - 1):
                v3 = "" if (str(d) == "2016-04-01" and s == 0) else f"{rng.normal():.3f}".replace("0.", ".", 1)
                wr.writerow([str(90 + s), f"지점{s}", str(d), "0.1", v3, f"{rng.normal():.2f}", "-1"])
    # 단계 0 결과(--inputs)
    _w(inputs / "s0" / "0-4_v2" / "시군구_대표좌표_0-4.parquet",
       pa.table({"sgg_code": SGG, "sgg_nm": ["a", "b", "c"], "lat": [SGG_XY[s][0] for s in SGG],
                 "lon": [SGG_XY[s][1] for s in SGG], "source": ["x"] * 3}))
    a3 = ["JPN", "USA"]
    od = np.arange(np.datetime64(D_MIN), np.datetime64("2016-06-30") + np.timedelta64(1, "D"), dtype="datetime64[D]")
    rain = np.round(rng.gamma(0.5, 6, size=(2, len(od))) * (rng.random((2, len(od))) < 0.5), 1)
    rain[1, :] = np.where(np.arange(len(od)) % 7 == 0, rain[1, :], 0.0)
    _w(inputs / "s0" / "0-4_weather_overseas" / "해외_나라_일별_기상_1차.parquet",
       pa.table({"alpha_3": np.repeat(a3, len(od)), "sgg": np.repeat(["OV" + a for a in a3], len(od)),
                 "date": pa.array(np.tile(od, 2), pa.date32()), "rain": rain.ravel(), "avg_temp": np.zeros(2 * len(od))}))


def _load_script():
    spec = importlib.util.spec_from_file_location("s0_4_spi", FINAL / "scripts" / "s0_4_spi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.check_resources = lambda path: {}   # 로컬 PC 는 RAM 바닥값(15GB) 아래 — 시험에서만 끔
    return mod


def _cfg(tmp_path, **over):
    cfg = json.loads((FINAL / "configs" / "s0_4_spi.json").read_text(encoding="utf-8"))
    cfg.update(over)
    p = tmp_path / "s0_4_spi_test.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    inputs = tmp_path_factory.mktemp("inputs")
    build_root(root, inputs, np.random.default_rng(7))
    return root, inputs


def test_config_matches_plan():
    cfg = json.loads((FINAL / "configs" / "s0_4_spi.json").read_text(encoding="utf-8"))
    assert cfg["fit_start"] == "2015-01-01" and cfg["fit_end"] == "2026-06-30" and cfg["pool_half_days"] == 15
    assert cfg["months"] == [3, 6] and cfg["first_valid"] == {"spi3": "2015-03-31", "spi6": "2015-06-30"}


def test_s0_4_spi_end_to_end(synth, tmp_path):
    root, inputs = synth
    mod = _load_script()
    cfg = _cfg(tmp_path, fit_end="2016-06-30")
    out = tmp_path / "out"
    mod.main(["--root", str(root), "--out", str(out), "--inputs", str(inputs), "--config", str(cfg)])
    obs = pd.read_parquet(out / "SPI_국내_관측_0-4.parquet")
    fc = pd.read_parquet(out / "SPI_국내_예보_0-4.parquet")
    ov = pd.read_parquet(out / "SPI_해외_0-4.parquet")
    assert list(obs.columns) == ["sgg_code", "date", "spi3", "spi6"] == list(fc.columns)
    assert list(ov.columns) == ["sgg", "alpha_3", "date", "spi3", "spi6"]
    c_obs = wn.DailyCube.from_long(obs, "sgg_code", "date", ["spi3", "spi6"])
    c_fc = wn.DailyCube.from_long(fc, "sgg_code", "date", ["spi3", "spi6"])
    c_ov = wn.DailyCube.from_long(ov, "sgg", "date", ["spi3", "spi6"])
    assert c_obs.units == tuple(SGG) == c_fc.units and c_ov.units == ("OVJPN", "OVUSA")
    assert iso(c_obs.start) == KMA_MIN and iso(c_fc.start) == D_MIN and iso(c_fc.end) == D_MAX

    # (1) 기상청 3지점 평균 = 날마다 그날 값 있는 지점 중 가까운 3곳(참조 계산)
    kma = pd.read_csv(root / "리스테리아ML2/data/외부_데이터/CLM_SPI_DD_합본_20140101_20261006.csv", encoding="cp949",
                      dtype={"지점": str})
    st = pq.read_table(root / "리스테리아ML2/data/raw/raw_kma_ground_obs_station_info.parquet").to_pandas().set_index("stn_id")
    for s in SGG:
        for day_s in ("2015-02-10", "2016-04-01", "2016-08-15"):
            k = kma[kma["일시"] == day_s].dropna(subset=["SPI3"])
            dist = [float(kw.haversine_km(*SGG_XY[s], st.loc[x, "lat"], st.loc[x, "lon"])) for x in k["지점"]]
            ref = float(np.mean(k["SPI3"].to_numpy()[np.argsort(dist, kind="stable")[:3]]))
            got = obs[(obs.sgg_code == s) & (obs.date.astype(str) == day_s)]["spi3"].iloc[0]
            assert got == pytest.approx(ref, abs=1e-6)
    gone = obs[obs.date.astype(str) == "2016-03-03"]
    assert gone[["spi3", "spi6"]].isna().all().all()                       # 빠진 날 채우지 않음
    assert (obs.date.astype(str) == "2014-12-15").sum() == len(SGG)        # 2014년 원값도 둠

    # (2) 국내 예보 칸 = ASOS sum_rn(빈 값 0) 3곳 평균 → 우리 식
    asos = pq.read_table(root / "리스테리아ML2/data/raw/raw_kma_asos_daily_info.parquet").to_pandas()
    asos["sum_rn"] = asos["sum_rn"].fillna(0.0)
    days = np.arange(D(D_MIN), D(D_MAX) + 1)
    P = np.full((len(SGG), len(days)), np.nan)
    for i, s in enumerate(SGG):
        for j, dd in enumerate(days):
            a = asos[asos.meas_date == np.datetime64(int(dd), "D").astype(object)]
            dist = [float(kw.haversine_km(*SGG_XY[s], st.loc[x, "lat"], st.loc[x, "lon"])) for x in a["stn_id"]]
            P[i, j] = float(np.mean(a["sum_rn"].to_numpy(np.float32).astype(np.float64)[np.argsort(dist, kind="stable")[:3]])) if len(a) else np.nan
    for n, v in ((3, "spi3"), (6, "spi6")):
        ref = sp.spi_from_rain(P.astype(np.float32).astype(np.float64), D(D_MIN), n, D("2015-01-01"), D("2016-06-30")).spi
        got = c_fc.values[:, :, ["spi3", "spi6"].index(v)]
        assert np.allclose(got, ref, equal_nan=True, rtol=0, atol=1e-7)
    assert np.isnan(c_fc.values[:, :D("2015-03-31") - D(D_MIN), 0]).all()
    assert not np.isnan(c_fc.values[:, D("2015-03-31") - D(D_MIN), 0]).any()
    assert np.isnan(c_fc.values[:, :D("2015-06-30") - D(D_MIN), 1]).all()

    # forecast_cube 로 이음: T 부터 예보 표 값
    d, h, W = np.array([D("2016-09-01")] * 2), np.array([5, 30]), 20
    seq = wn.sequences(c_obs, ["11110", "26110"], d, h, W, forecast_cube=c_fc)
    b = wn.window_bounds(d, h, W)
    for r in range(2):
        for k in range(W):
            dd = b.S[r] + k
            src = c_fc if dd >= b.T[r] else c_obs
            exp = src.values[src.unit_index([["11110", "26110"][r]])[0], dd - src.start]
            assert np.allclose(seq[r, k], exp, equal_nan=True)

    # (3) 해외 = 나라 강수로 나라마다
    raw = pq.read_table(inputs / "s0/0-4_weather_overseas/해외_나라_일별_기상_1차.parquet").to_pandas()
    Pov = raw.pivot(index="sgg", columns="date", values="rain").to_numpy()
    ref = sp.spi_from_rain(Pov, D(D_MIN), 3, D("2015-01-01"), D("2016-06-30")).spi
    assert np.allclose(c_ov.values[:, :, 0], ref, equal_nan=True)
    assert set(ov.drop_duplicates("sgg").set_index("sgg")["alpha_3"].items()) == {("OVJPN", "JPN"), ("OVUSA", "USA")}

    # (4)·(5) 대조·감마 모수·메타
    cmp_ = pd.read_csv(out / "SPI_대조_0-4.csv", encoding="utf-8-sig", dtype={"unit": str})
    assert set(cmp_.level) == {"지점", "시군구"}
    assert set(cmp_[cmp_.level == "지점"].unit) == {str(90 + s) for s in range(N_ST - 1)}
    assert set(cmp_["기간"]) == {"전체", "적합 기간 안", "적합 기간 뒤"}
    gm = pd.read_parquet(out / "SPI_감마모수_0-4.parquet")
    assert set(gm.kind) == {"국내_시군구", "해외_나라", "대조_지점"}
    assert len(gm) == (len(SGG) + 2 + (N_ST - 1)) * 2 * 365
    chk = json.loads((out / "점검.json").read_text(encoding="utf-8"))
    assert chk["국내 우리 식"]["spi3"]["값 있는 첫 날"] == "2015-03-31"
    assert chk["해외 우리 식"]["spi6"]["값 있는 첫 날"] == "2015-06-30"
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert set(meta["stage0_inputs_md5"]) == {"s0/0-4_v2/시군구_대표좌표_0-4.parquet",
                                              "s0/0-4_weather_overseas/해외_나라_일별_기상_1차.parquet"}
    assert len(meta["inputs_md5"]) == 3 and meta["code_md5"]
    md = pq.read_schema(out / "SPI_국내_예보_0-4.parquet").metadata
    assert "감마 적합 기간 2015-01-01 ~ 2016-06-30" in md[b"description"].decode("utf-8")
    # 다시 돌리면 덮지 않음
    with pytest.raises(FileExistsError):
        mod.main(["--root", str(root), "--out", str(out), "--inputs", str(inputs), "--config", str(cfg)])


def test_s0_4_spi_rejects_unknown_config(synth, tmp_path):
    root, inputs = synth
    mod = _load_script()
    from lst.config import ConfigError
    with pytest.raises(ConfigError):
        mod.main(["--root", str(root), "--out", str(tmp_path / "o"), "--inputs", str(inputs),
                  "--config", str(_cfg(tmp_path, extra_knob=1))])
    with pytest.raises(ConfigError):
        mod.main(["--root", str(root), "--out", str(tmp_path / "o2"), "--inputs", str(inputs),
                  "--config", str(_cfg(tmp_path, months=[3, 12]))])
