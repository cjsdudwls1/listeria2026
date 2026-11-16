"""해외 기상(단계 0-4) 시험 — 합성 자료만(네트워크 없음). 계획서 §4-4·부록 A-2."""
import datetime as dt
import importlib.util
import json
import math
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import ovweather as ow

CODE = Path(__file__).resolve().parents[1]
CFG = json.loads((CODE / "configs/s0_4_weather_overseas.json").read_text(encoding="utf-8"))
MODELS = CFG["openmeteo"]["models"]
P = ow.periods_of(CFG)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 계획·무게
def test_split_years_and_plan():
    assert ow.split_years("2013-05-01", "2014-04-06") == [(dt.date(2013, 5, 1), dt.date(2013, 12, 31)),
                                                         (dt.date(2014, 1, 1), dt.date(2014, 4, 6))]
    assert ow.split_years("2015-01-02", "2015-01-01") == []
    ch = ow.plan_chunks("AAA", ow.COORD_DB, P, (dt.date(2015, 1, 1), dt.date(2026, 9, 30)))
    g = {}
    for c in ch:
        g.setdefault(c.group, []).append((c.start, c.end))
    assert g["rain"] == [(dt.date(2013, 5, 1), dt.date(2013, 12, 31)), (dt.date(2014, 1, 1), dt.date(2014, 4, 6))]
    # DB 끝(2026-09-30) > use_end → 끝 뒤 조각 없음, DB 에 없는 기간 = 2014-04-07 ~ 12-31, + 겹치는 2015
    assert g["dbvars"] == [(dt.date(2014, 4, 7), dt.date(2014, 12, 31)), (dt.date(2015, 1, 1), dt.date(2015, 12, 31))]
    assert g["extra"][0] == (dt.date(2014, 4, 7), dt.date(2014, 12, 31)) and g["extra"][-1] == (dt.date(2026, 1, 1), dt.date(2026, 6, 30))
    assert len(g["extra"]) == 13
    # DB 끝이 이르면 끝 뒤도 받음
    ch2 = ow.plan_chunks("AAA", ow.COORD_DB, P, (dt.date(2015, 1, 1), dt.date(2026, 3, 31)))
    assert (dt.date(2026, 4, 1), dt.date(2026, 6, 30)) in [(c.start, c.end) for c in ch2 if c.group == "dbvars"]
    # 겹치는 기간이 DB 밖이면 멈춤
    with pytest.raises(ValueError):
        ow.plan_chunks("AAA", ow.COORD_DB, P, (dt.date(2015, 6, 1), dt.date(2026, 9, 30)))
    # DB 에 없는 나라
    ch3 = ow.plan_chunks("CCC", ow.COORD_POLY, P, None)
    db3 = [(c.start, c.end) for c in ch3 if c.group == "dbvars"]
    assert db3[0] == (dt.date(2014, 4, 7), dt.date(2014, 12, 31)) and db3[-1][1] == dt.date(2026, 6, 30)


def test_cost_matches_observed_headers():
    # 11-07 시험 호출: 15 변수 × 2 모델 × 3일 = 3.0, 10 변수 × 1 모델 × 2지점 × 5일 = 2.0
    assert ow.est_cost(15, 2, 3) == pytest.approx(3.0)
    assert ow.est_cost(10, 1, 5, n_locations=2) == pytest.approx(2.0)
    assert ow.est_cost(1, 1, 365) == pytest.approx(365 / 14)
    v, m = ow.group_request("extra", MODELS)
    assert sorted(m) == ["era5", "era5_land"] and len(v) == 5
    assert ow.est_cost(len(v), len(m), 28) == pytest.approx(2.0)


# ---------------------------------------------------------------- 응답
def _body(c, models, null_key=None, dates=None):
    vars_, ms = ow.group_request(c.group, models)
    t = dates or [x.strftime("%Y-%m-%d") for x in ow.date_range(c.start, c.end)]
    daily = {"time": t}
    for v in vars_:
        for m in ms:
            k = ow.response_key(v, m, ms)
            daily[k] = [None if k == null_key else 1.5] * len(t)
    return json.dumps({"latitude": 1.0, "longitude": 2.0, "timezone": "Asia/Seoul", "utc_offset_seconds": 32400,
                       "daily": daily})


def test_parse_response():
    c = ow.Chunk("AAA", ow.COORD_DB, "extra", dt.date(2015, 1, 1), dt.date(2015, 1, 3))
    df, head = ow.parse_response(_body(c, MODELS, null_key="wind_direction_10m_dominant_era5_land"), c, MODELS)
    assert df.shape == (3, 10) and head["timezone"] == "Asia/Seoul"
    assert df["wind_direction_10m_dominant@era5_land"].isna().all()
    c1 = ow.Chunk("AAA", ow.COORD_DB, "dbvars", dt.date(2015, 1, 1), dt.date(2015, 1, 3))
    df1, _ = ow.parse_response(_body(c1, MODELS), c1, MODELS)
    assert list(df1.columns)[0] == "temperature_2m_mean@era5_seamless"     # 모델 하나면 접미사 없음
    with pytest.raises(ow.ResponseError):
        ow.parse_response(_body(c1, MODELS, dates=["2015-01-01", "2015-01-02"]), c1, MODELS)
    with pytest.raises(ow.ResponseError):
        ow.parse_response(json.dumps({"error": True, "reason": "x"}), c1, MODELS)
    with pytest.raises(ow.ResponseError):
        ow.parse_response(_body(c, MODELS), c, {**MODELS, "dew_point": "ecmwf_ifs"})   # 요청한 칼럼 없음
    # 응답 격자(1.0, 2.0)가 요청 좌표에서 0.5° 안이면 통과, 넘으면 이상
    ow.parse_response(_body(c1, MODELS), c1, MODELS, 1.4, 2.4)
    with pytest.raises(ow.ResponseError, match="옮겨짐"):
        ow.parse_response(_body(c1, MODELS), c1, MODELS, 1.6, 2.0)
    with pytest.raises(ow.ResponseError, match="옮겨짐"):
        ow.parse_response(_body(c1, MODELS), c1, MODELS, 1.0, 2.6)
    assert ow.lon_diff(179.9, -179.9) == pytest.approx(0.2)
    # 요청 칼럼이 모두 null 이면 이상(일부 칼럼만 null 은 정상 — 위 era5_land 풍향)
    j = json.loads(_body(c1, MODELS))
    for k in j["daily"]:
        if k != "time":
            j["daily"][k] = [None] * 3
    with pytest.raises(ow.ResponseError, match="null"):
        ow.parse_response(json.dumps(j), c1, MODELS)


def test_rate_budget():
    b = ow.RateBudget({"minute": 10, "hour": 25, "day": 40, "month": 100})
    assert b.wait_needed(0.0, 5) == 0
    b.add(0.0, 6)
    assert b.wait_needed(1.0, 5) == pytest.approx(59.001)   # 분 창: 0초 기록이 나가야 함
    b.add(61.0, 9)
    b.add(122.0, 9)
    assert b.wait_needed(130.0, 5) == pytest.approx(3600 - 130 + 0.001)   # 시간 창 25 넘음 → 0초 기록이 나갈 때까지
    with pytest.raises(ValueError):
        b.wait_needed(0.0, 11)
    assert ow.classify_429(b'{"reason":"Minutely API request limit exceeded"}') == "minute"
    assert ow.classify_429(b'{"reason":"Hourly API request limit exceeded"}') == "hour"
    assert ow.classify_429(b'{"reason":"Daily API request limit exceeded"}') == "day"
    assert ow.classify_429(b'{"reason":"Monthly API request limit exceeded"}') == "month"
    # 월(31일 창): 이번 호출로 넘으면 True
    assert not b.over_month(130.0, 5) and b.month_used(130.0) == pytest.approx(24)
    b.add(200.0, 70)
    assert b.over_month(300.0, 7) and not b.over_month(300.0 + 31 * 86400, 7)


# ---------------------------------------------------------------- 변환·보정
def test_convert_calibrate_clip():
    idx = pd.date_range("2015-01-01", periods=4)
    om = pd.DataFrame({v: [1.0, 2.0, 3.0, 4.0] for v in ow.CAL_VARS}, index=idx)
    db = om.copy()
    db["avg_temp"] += 2.0
    db["rain"] = om["rain"] * 1.5
    db[ow.HUM] = om[ow.HUM] - 3.0
    db.loc[idx[0], "max_temp"] = np.nan
    cal = ow.calibration(db, om)
    assert cal["avg_temp"]["value"] == pytest.approx(2.0) and cal["rain"]["value"] == pytest.approx(1.5)
    assert cal["max_temp"]["n"] == 3
    zero = om.copy()
    zero["rain"] = 0.0
    assert ow.calibration(db, zero)["rain"]["value"] == 1.0      # Open-Meteo 합 0 → 1
    gap = pd.DataFrame({v: [-1.0, 50.0, 101.0] for v in ow.CAL_VARS}, index=pd.date_range("2014-05-01", periods=3))
    cal2 = {v: {"kind": "diff", "value": 0.0} for v in ow.CAL_VARS}
    cal2["rain"] = {"kind": "ratio", "value": 1.0}
    out, cl = ow.apply_calibration(gap, cal2)
    assert out["rain"].tolist() == [0.0, 50.0, 101.0] and out[ow.HUM].tolist() == [0.0, 50.0, 100.0]
    assert out["avg_wind"].tolist() == [-1.0, 50.0, 101.0]           # 바람은 안 자름
    assert cl["rain"] == 1 and cl[ow.HUM] == 2 and cl["soil_moist"] == 1 and cl["sunshine"] == 1
    c = ow.convert_units(pd.DataFrame({"avg_wind": [36.0], "sunshine": [7200.0], "soil_moist": [0.25], "rain": [3.0]}))
    assert c.iloc[0].tolist() == pytest.approx([10.0, 2.0, 25.0, 3.0])
    assert ow.unit_ratio(np.array([1.0, 2.0, np.nan]), np.array([1.0, 2.0, 5.0])) == pytest.approx(1.0)
    assert ow.unit_ratio(np.array([3.6]), np.array([1.0])) == pytest.approx(3.6)


def test_soil_wind_dew():
    s = ow.interp_soil([10.0, np.nan], [24.0, 1.0], [70.5, 2.0])
    # 5cm = 10 + 1.5/14·14 = 11.5, 10cm = 10 + 6.5/14·14 = 16.5, 20cm = 24 + 2.5/46.5·46.5 = 26.5, 30cm = 36.5
    assert s["soil_t5"][0] == pytest.approx(11.5) and s["soil_t10"][0] == pytest.approx(16.5)
    assert s["soil_t20"][0] == pytest.approx(26.5) and s["soil_t30"][0] == pytest.approx(36.5)
    assert np.isnan(s["soil_t5"][1]) and not np.isnan(s["soil_t20"][1])
    sn, cs = ow.wind_dir_sincos([90.0, np.nan])
    assert sn[0] == pytest.approx(1.0) and abs(cs[0]) < 1e-12 and np.isnan(sn[1])
    assert ow.dewpoint_magnus([20.0], [100.0])[0] == pytest.approx(20.0)
    assert ow.dewpoint_magnus([20.0], [50.0])[0] == pytest.approx(9.26, abs=0.05)
    assert np.isnan(ow.dewpoint_magnus([20.0], [0.0])[0])


def test_decide_coord():
    def e(vals):
        return pd.DataFrame({"mae": vals, "n": 10}, index=list(ow.CAL_VARS))
    per = {"AAA": {ow.COORD_DB: e([0.1] * 10), ow.COORD_COUNTRY: e([0.2] * 10)},
           "BBB": {ow.COORD_DB: e([0.3] * 10), ow.COORD_COUNTRY: e([0.3] * 9 + [0.1])}}
    r = ow.decide_coord(per)
    assert r["winner"] == ow.COORD_DB and r["wins"] == {ow.COORD_DB: 10, ow.COORD_COUNTRY: 1} and r["ties"] == 9
    per["BBB"] = {ow.COORD_DB: e([0.3] * 10), ow.COORD_COUNTRY: e([0.1] * 10)}
    assert ow.decide_coord(per)["winner"] is None


def test_polygon_coords():
    pytest.importorskip("shapely")
    sq = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}
    sq2 = {"type": "Polygon", "coordinates": [[[20, 0], [22, 0], [22, 2], [20, 2], [20, 0]]]}
    geo = {"features": [
        {"properties": {"ISO_A3_EH": "-99", "ADM0_A3": "CCC"}, "geometry": sq},
        {"properties": {"ISO_A3_EH": "-99", "ADM0_A3": "CCC"}, "geometry": sq2},
        {"properties": {"ISO_A3_EH": "DDD", "ADM0_A3": "XXX"}, "geometry": {"type": "Polygon", "coordinates": [[[50, 50], [51, 50], [51, 51], [50, 51], [50, 50]]]}},
    ]}
    lat = np.array([1.0, 2.0, 9.0, 1.0, np.nan, 30.0], dtype=np.float32)
    lon = np.array([1.0, 3.0, 9.0, 21.0, 5.0, 30.0], dtype=np.float32)
    r = ow.polygon_coords(geo, ["CCC", "DDD"], lat, lon)
    assert r["CCC"]["n_samples"] == 4 and r["CCC"]["lat"] == pytest.approx(1.5) and r["CCC"]["lon"] == pytest.approx(6.0)
    assert r["CCC"]["n_polygons"] == 2
    assert r["DDD"]["how"] == "다각형 대표점" and 50 < r["DDD"]["lat"] < 51
    with pytest.raises(KeyError):
        ow.polygon_coords(geo, ["ZZZ"], lat, lon)


def test_column_descriptions():
    d = ow.column_descriptions(MODELS)
    assert set(d) == set(ow.TABLE_VARS)
    assert {k for k, v in d.items() if v["approx"]} == set(ow.APPROX)
    assert d["max_wind"]["description"].startswith("근사") and not d["avg_temp"]["approx"]


# ---------------------------------------------------------------- 끝까지(합성 DB + 가짜 Open-Meteo)
def om_value(key: str, day: pd.Timestamp, lat: float) -> float | None:
    """가짜 Open-Meteo 값(원 단위). key = 응답 칼럼 이름."""
    x = day.dayofyear / 365.25 * 2 * math.pi
    base = {
        "temperature_2m_mean": 15 + 10 * math.sin(x), "temperature_2m_max": 20 + 10 * math.sin(x),
        "temperature_2m_min": 10 + 10 * math.sin(x), "precipitation_sum": 3 * (1 + math.sin(day.toordinal() * 0.7)),
        "wind_speed_10m_mean": 12 + 3 * math.sin(x), "wind_speed_10m_max": 25 + 3 * math.sin(x),
        "sunshine_duration": 30000 + 5000 * math.sin(x), "shortwave_radiation_sum": 15 + 5 * math.sin(x),
        "relative_humidity_2m_mean": 70 + 20 * math.sin(x), "soil_moisture_0_to_7cm_mean": 0.3 + 0.05 * math.sin(x),
        "soil_temperature_0_to_7cm_mean": 10 + 5 * math.sin(x), "soil_temperature_7_to_28cm_mean": 11 + 4 * math.sin(x),
        "soil_temperature_28_to_100cm_mean": 12 + 2 * math.sin(x), "wind_direction_10m_dominant": 180 + 90 * math.sin(x),
        "dew_point_2m_mean": 5 + 5 * math.sin(x),
    }
    for name, v in base.items():
        if key == name or key.startswith(name + "_"):
            model = key[len(name) + 1:] if key != name else None
            if model == "era5_land" and name == "wind_direction_10m_dominant":
                return None
            if model == "era5" and name.startswith("soil_temperature"):
                v += 100.0        # era5 지중온도는 쓰면 안 됨(era5_land 를 골라야 함)
            if name in ("precipitation_sum", "sunshine_duration", "soil_moisture_0_to_7cm_mean"):
                return v * (1 + 0.01 * lat)
            return v + 0.1 * lat
    raise KeyError(key)


DB_OFFSET = {"avg_temp": -0.5, "max_temp": -0.5, "min_temp": -0.5, "avg_wind": -0.5, "max_wind": -0.5,
             "solar_rad": -0.5, ow.HUM: 15.0}
DB_RATIO = {"rain": 1.2, "sunshine": 0.98, "soil_moist": 0.98}


def fake_http_factory(calls):
    def http(url, timeout, ua):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        calls.append(q)
        lat = float(q["latitude"])
        days = pd.date_range(q["start_date"], q["end_date"])
        ms = q["models"].split(",")
        daily = {"time": [x.strftime("%Y-%m-%d") for x in days]}
        for v in q["daily"].split(","):
            for m in ms:
                k = v if len(ms) == 1 else f"{v}_{m}"
                daily[k] = [om_value(k, x, lat) for x in days]
        body = json.dumps({"latitude": round(lat, 1), "longitude": round(float(q["longitude"]), 1),
                           "timezone": "Etc/GMT", "daily": daily}).encode()
        cost = ow.est_cost(len(q["daily"].split(",")), len(ms), len(days))
        return 200, {"X-RateLimit-Cost": str(cost)}, body
    return http


def make_root(tmp_path):
    root = tmp_path / "root"
    ext = root / "리스테리아ML2/data/추출_20261107"
    ext.mkdir(parents=True)
    days = pd.date_range("2015-01-01", "2026-09-30")
    rows = []
    coords = {"AAA": (10.0, 20.0, 11.0, 21.0), "BBB": (-30.0, 150.0, -29.0, 151.0)}
    inv = {v: k for k, v in ow.OM_DB_VARS.items()}
    for a, (la, lo, cla, clo) in coords.items():
        df = pd.DataFrame({"alpha_3": a, "date": days.date})
        for dbc, var in ow.DB_TO_VAR.items():
            raw = np.array([om_value(inv[var], x, la) for x in days])
            if var in DB_RATIO:
                raw = raw * DB_RATIO[var]
            raw = raw + DB_OFFSET.get(var, 0.0)
            df[dbc] = raw
        df.loc[5, "wind_speed_max"] = np.nan
        df["lat"], df["lon"], df["country_lat"], df["country_lon"], df["name_ko"] = la, lo, cla, clo, a
        rows.append(df)
    pd.concat(rows).to_parquet(ext / "global_weather_info.parquet", index=False)
    smp = root / "리스테리아ML/data"
    smp.mkdir(parents=True)
    pd.DataFrame({"lat": np.array([1.0, 3.0, 50.0], dtype=np.float32),
                  "lon": np.array([1.0, 5.0, 50.0], dtype=np.float32)}).to_parquet(smp / "listeria_data.parquet")
    geo = root / "리스테리아ML2/data/외부_수집/축사"
    geo.mkdir(parents=True)
    sq = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}
    (geo / "ne_10m_admin_0_countries_v5.1.2.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"ISO_A3_EH": "CCC", "ADM0_A3": "CCC"},
                                                    "geometry": sq}]}), encoding="utf-8")
    pd.DataFrame({"alpha_3": ["BBB", "AAA", "CCC", "AAA"]}).to_csv(root / "nations.csv", index=False)
    return root


def test_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    fe = _load("s0_4_openmeteo_fetch")
    bu = _load("s0_4_weather_overseas")
    for m in (fe, bu):
        monkeypatch.setattr(m.rs, "check_resources", lambda path: {})
    calls, sleeps = [], []
    clock = [1_000_000.0]
    monkeypatch.setitem(fe.HOOKS, "http", fake_http_factory(calls))
    monkeypatch.setitem(fe.HOOKS, "sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setitem(fe.HOOKS, "clock", lambda: clock[0])
    root = make_root(tmp_path)
    fout = tmp_path / "fetch"
    base = ["--root", str(root), "--out", str(fout), "--nations", "nations.csv"]
    assert fe.main(base + ["--stage", "plan"]) == 0
    assert calls == []
    with pytest.raises(fe.FetchStop):      # coordcheck 전 fetch 금지
        fe.main(base + ["--stage", "fetch"])
    assert fe.main(base + ["--stage", "coordcheck"]) == 0
    dec = json.loads((fout / "coordcheck.json").read_text(encoding="utf-8"))
    assert dec["winner"] == ow.COORD_DB and dec["countries"] == ["AAA", "BBB"]
    n_cc = len(calls)
    assert n_cc == 4                                  # 2나라 × 좌표 2벌 × 2015 1조각
    assert fe.main(base + ["--stage", "fetch"]) == 0
    # DB 나라: rain 2 + dbvars 1(2014) + extra 13 = 16(2015 는 coordcheck 것 다시 씀), CCC: 2 + 13 + 13 = 28
    assert len(calls) - n_cc == 2 * 16 + 28
    assert all(s >= 0 for s in sleeps)
    n_all = len(calls)
    assert fe.main(base + ["--stage", "fetch"]) == 0      # 이어받기: 전부 건너뜀
    assert len(calls) == n_all
    log = [json.loads(x) for x in (fout / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == n_all and all(r["status"] == 200 for r in log)
    # 호출 사이 1초 이상(가짜 시계)
    ts = sorted(r["t"] for r in log)
    assert min(np.diff(ts)) >= 1.0 - 1e-9

    bout = tmp_path / "build"
    assert bu.main(["--root", str(root), "--out", str(bout), "--nations", "nations.csv", "--fetch-dir", str(fout)]) == 0
    t2 = pd.read_parquet(bout / bu.OUT_2)
    t1 = pd.read_parquet(bout / bu.OUT_1)
    assert len(t2) == len(t1) == 3 * len(ow.date_range(P.rain_start, P.use_end))
    t2["date"] = pd.to_datetime(t2["date"])
    t1["date"] = pd.to_datetime(t1["date"])
    a = t2[t2.alpha_3 == "AAA"].set_index("date")
    a1 = t1[t1.alpha_3 == "AAA"].set_index("date")
    la = 10.0
    # DB 기간 = DB 값(단위 변환)
    x = pd.Timestamp("2016-03-01")
    assert a.loc[x, "avg_temp"] == pytest.approx(om_value("temperature_2m_mean", x, la) - 0.5)
    assert a.loc[x, "avg_wind"] == pytest.approx((om_value("wind_speed_10m_mean", x, la) - 0.5) / 3.6)
    assert a.loc[x, "src_avg_temp"] == "db" and a1.loc[x, "avg_temp"] == a.loc[x, "avg_temp"]
    # DB NaN 칸은 안 채움
    assert np.isnan(a.loc[pd.Timestamp("2015-01-06"), "max_wind"]) and a.loc[pd.Timestamp("2015-01-06"), "src_max_wind"] is None
    # DB 에 없는 기간 = Open-Meteo + 보정
    g = pd.Timestamp("2014-06-01")
    assert a.loc[g, "avg_temp"] == pytest.approx(om_value("temperature_2m_mean", g, la) - 0.5)
    assert a.loc[g, "src_avg_temp"] == "om_cal" and np.isnan(a1.loc[g, "avg_temp"]) and a1.loc[g, "src_avg_temp"] is None
    d15 = pd.date_range("2015-01-01", "2015-12-31")
    sdiff = np.mean([om_value("sunshine_duration", t, la) * (0.98 - 1) for t in d15])   # 평균 차이(원 단위)
    assert a.loc[g, "sunshine"] == pytest.approx((om_value("sunshine_duration", g, la) + sdiff) / 3600)
    r0 = pd.Timestamp("2013-06-01")
    assert a.loc[r0, "rain"] == pytest.approx(om_value("precipitation_sum", r0, la) * 1.2)
    assert np.isnan(a.loc[r0, "avg_temp"])           # 2014-04-07 전은 강수만
    assert np.isnan(a.loc[pd.Timestamp("2014-04-06"), "soil_t5"]) and not np.isnan(a.loc[pd.Timestamp("2014-04-07"), "soil_t5"])
    # 습도: 보정(+15) 뒤 100 넘는 칸 자름(om_cal 만), DB 칸은 그대로
    hum_cal = a.loc[a.src_avg_rhm == "om_cal", "avg_rhm"]
    assert hum_cal.max() == 100.0 and (hum_cal == 100.0).sum() > 0
    assert a.loc[a.src_avg_rhm == "db", "avg_rhm"].max() > 100.0
    # DB 에 없는 변수 = Open-Meteo 그대로(지중온도 era5_land 보간·풍향 era5·이슬점 era5), 1차엔 없음
    s0, s1 = om_value("soil_temperature_0_to_7cm_mean_era5_land", x, la), om_value("soil_temperature_7_to_28cm_mean_era5_land", x, la)
    assert a.loc[x, "soil_t5"] == pytest.approx(s0 + 1.5 / 14 * (s1 - s0)) and a.loc[x, "src_soil_t5"] == "om"
    wd = om_value("wind_direction_10m_dominant_era5", x, la)
    assert a.loc[x, "wind_dir_sin"] == pytest.approx(math.sin(math.radians(wd)))
    assert a.loc[x, "avg_td"] == pytest.approx(om_value("dew_point_2m_mean_era5", x, la))
    assert a1["soil_t5"].isna().all() and a1["avg_td"].isna().all()
    for c in ow.ALWAYS_NAN:
        assert t2[c].isna().all() and t2["src_" + c].isna().all()
    # DB 에 없는 나라 = Open-Meteo 그대로, 좌표 = 다각형 안 시료 중앙값(2, 3)
    cc = t2[t2.alpha_3 == "CCC"].set_index("date")
    assert set(cc["src_avg_temp"].dropna()) == {"om"}
    assert cc.loc[x, "avg_temp"] == pytest.approx(om_value("temperature_2m_mean", x, 2.0))
    assert t1.loc[t1.alpha_3 == "CCC", list(ow.TABLE_VARS)].isna().all().all()
    assert set(t2.sgg) == {"OVAAA", "OVBBB", "OVCCC"}
    rep = json.loads((bout / bu.OUT_REP).read_text(encoding="utf-8"))
    assert rep["unit_ratio_db_over_om"]["avg_wind"] == pytest.approx(1.0, abs=0.1)
    assert rep["db_nan_cells_not_filled"]["AAA"] == {"max_wind": 1}
    cal = pd.read_csv(bout / bu.OUT_CAL)
    assert cal.set_index(["alpha_3", "var"]).loc[("BBB", "rain"), "value"] == pytest.approx(1.2)
    assert not (bout / bu.OUT_2).with_suffix(".parquet.part").exists()
    # 다시 만들면 덮지 않고 멈춤
    with pytest.raises(bu.BuildStop):
        bu.main(["--root", str(root), "--out", str(bout), "--nations", "nations.csv", "--fetch-dir", str(fout)])


def test_fetch_stops_on_changed_request(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    fe = _load("s0_4_openmeteo_fetch")
    monkeypatch.setattr(fe.rs, "check_resources", lambda path: {})
    calls = []
    clock = [0.0]
    monkeypatch.setitem(fe.HOOKS, "http", fake_http_factory(calls))
    monkeypatch.setitem(fe.HOOKS, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setitem(fe.HOOKS, "clock", lambda: clock[0])
    root = make_root(tmp_path)
    fout = tmp_path / "fetch"
    base = ["--root", str(root), "--out", str(fout), "--nations", "nations.csv"]
    assert fe.main(base + ["--stage", "coordcheck"]) == 0
    # country 좌표로 강제하면 DB 나라 2015 조각 경로가 달라 새로 받음(덮지 않음) — 같은 경로의 요청 값이 바뀌면 멈춤
    req = next((fout / "raw/AAA/db_latlon").glob("*.req.json"))
    j = json.loads(req.read_text(encoding="utf-8"))
    j["params"]["latitude"] = "99.0"
    req.write_text(json.dumps(j), encoding="utf-8")
    assert fe.main(base + ["--stage", "fetch"]) == 3       # 멈춤(종료 코드 3), 덮지 않음
    assert json.loads(req.read_text(encoding="utf-8"))["params"]["latitude"] == "99.0"


def test_fetch_retry_and_429(tmp_path, monkeypatch):
    fe = _load("s0_4_openmeteo_fetch")
    seq = [(429, {}, b'{"error":true,"reason":"Minutely API request limit exceeded"}'),
           (502, {}, b"bad gateway"), None]
    clock = [0.0]
    slept = []

    def http(url, timeout, ua):
        r = seq.pop(0)
        if r is None:
            return fake_http_factory([])(url, timeout, ua)
        return r

    cfg = ow.load_cfg(CODE / "configs/s0_4_weather_overseas.json")
    f = fe.Fetcher(tmp_path, cfg, "x", http=http, clock=lambda: clock[0],
                   sleep=lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    c = ow.Chunk("AAA", ow.COORD_DB, "rain", dt.date(2013, 5, 1), dt.date(2013, 5, 10))
    assert f.get(c, 1.0, 2.0) == "fetched"
    assert 70 in slept and 10 in slept          # 429(분) 70초, 502 첫 재시도 10초
    assert f.get(c, 1.0, 2.0) == "skip"
    log = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["status"] for x in log] == [429, 502, 200]
    # 400 은 바로 멈춤
    f2 = fe.Fetcher(tmp_path / "b", cfg, "x", http=lambda *a: (400, {}, b'{"reason":"bad var"}'),
                    clock=lambda: clock[0], sleep=lambda s: None)
    (tmp_path / "b").mkdir()
    with pytest.raises(fe.FetchStop):
        f2.get(c, 1.0, 2.0)


def _fetcher(fe, out, http, clock, slept=None):
    cfg = ow.load_cfg(CODE / "configs/s0_4_weather_overseas.json")
    out.mkdir(parents=True, exist_ok=True)
    return fe.Fetcher(out, cfg, "x", http=http, clock=lambda: clock[0],
                      sleep=lambda s: ((slept.append(s) if slept is not None else None), clock.__setitem__(0, clock[0] + s)),
                      log=lambda *a: None)


def test_fetch_quarantine_incomplete(tmp_path):
    fe = _load("s0_4_openmeteo_fetch")
    clock = [0.0]
    calls = []
    f = _fetcher(fe, tmp_path / "o", fake_http_factory(calls), clock)
    c = ow.Chunk("AAA", ow.COORD_DB, "rain", dt.date(2013, 5, 1), dt.date(2013, 5, 10))
    assert f.get(c, 1.0, 2.0) == "fetched"
    p, q = f.paths(c)
    # 조각 파일은 바뀌고 요청 기록 쓰기 전에 끊긴 상태 + 남은 .part
    q.unlink()
    q.with_name(q.name + ".part").write_text("{}", encoding="utf-8")
    assert f.get(c, 1.0, 2.0) == "fetched" and len(calls) == 2      # 다시 받음
    moved = sorted(x.name for x in (tmp_path / "o" / "불완전").rglob("*") if x.is_file())
    assert len(moved) == 2 and any(x.startswith(p.name + ".") for x in moved)   # 지우지 않고 옮김
    assert p.exists() and q.exists() and f.get(c, 1.0, 2.0) == "skip"
    # 요청 기록만 있고 조각 없음 → 멈춤
    p.unlink()
    with pytest.raises(fe.FetchStop):
        f.get(c, 1.0, 2.0)


def test_fetch_stops_on_bad_response_and_month(tmp_path):
    fe = _load("s0_4_openmeteo_fetch")
    clock = [0.0]
    good = fake_http_factory([])
    c = ow.Chunk("AAA", ow.COORD_DB, "rain", dt.date(2013, 5, 1), dt.date(2013, 5, 10))

    def shifted(url, timeout, ua):
        st, h, b = good(url, timeout, ua)
        j = json.loads(b)
        j["latitude"] += 0.8
        return st, h, json.dumps(j).encode()

    f = _fetcher(fe, tmp_path / "a", shifted, clock)
    with pytest.raises(fe.FetchStop, match="저장하지 않고"):
        f.get(c, 1.0, 2.0)
    p, q = f.paths(c)
    assert not p.exists() and not q.exists() and not list((tmp_path / "a").rglob("*.part"))
    log = [json.loads(x) for x in (tmp_path / "a" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "옮겨짐" in log[-1]["error"]
    # 429 월 상한 → 재시도 없이 멈춤
    n = []

    def month429(url, timeout, ua):
        n.append(1)
        return 429, {}, b'{"error":true,"reason":"Monthly API request limit exceeded"}'

    f2 = _fetcher(fe, tmp_path / "b", month429, clock)
    with pytest.raises(fe.FetchStop, match="월 상한"):
        f2.get(c, 1.0, 2.0)
    assert len(n) == 1
    # 기록상 31일 안 무게가 월 상한 근처면 호출 전에 멈춤(호출 기록 이어서 셈)
    (tmp_path / "c").mkdir()
    with open(tmp_path / "c" / "calls.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": -20 * 86400.0, "cost": 249999.5}) + chr(10))
    m = []
    f3 = _fetcher(fe, tmp_path / "c", lambda *a: (m.append(1), good(*a))[1], clock)
    with pytest.raises(fe.FetchStop, match="월 상한"):
        f3.get(c, 1.0, 2.0)
    assert m == []


def test_db_only(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    bu = _load("s0_4_weather_overseas")
    monkeypatch.setattr(bu.rs, "check_resources", lambda path: {})
    root = make_root(tmp_path)
    out = tmp_path / "dbonly"
    with pytest.raises(bu.BuildStop):
        bu.main(["--root", str(root), "--out", str(out), "--db-only", "--fetch-dir", str(tmp_path)])
    assert bu.main(["--root", str(root), "--out", str(out), "--db-only"]) == 0       # 나라 = DB 전부(AAA·BBB)
    t1 = pd.read_parquet(out / bu.OUT_1)
    t2 = pd.read_parquet(out / bu.OUT_2)
    pd.testing.assert_frame_equal(t1, t2)
    days = ow.date_range("2015-01-01", P.use_end)
    assert sorted(t2.alpha_3.unique()) == ["AAA", "BBB"] and len(t2) == 2 * len(days)
    t2["date"] = pd.to_datetime(t2["date"])
    a = t2[t2.alpha_3 == "AAA"].set_index("date")
    x = pd.Timestamp("2016-03-01")
    assert a.loc[x, "avg_temp"] == pytest.approx(om_value("temperature_2m_mean", x, 10.0) - 0.5)
    assert a.loc[x, "avg_wind"] == pytest.approx((om_value("wind_speed_10m_mean", x, 10.0) - 0.5) / 3.6)
    assert a.loc[x, "sunshine"] == pytest.approx(om_value("sunshine_duration", x, 10.0) * 0.98 / 3600)
    assert a.loc[x, "soil_moist"] == pytest.approx(om_value("soil_moisture_0_to_7cm_mean", x, 10.0) * 0.98 * 100)
    assert a.loc[x, ow.HUM] > 0 and a.loc[x, "src_avg_rhm"] == "db"
    assert a.index.max() == pd.Timestamp(P.use_end) and a.index.min() == pd.Timestamp("2015-01-01")
    nan_day = pd.Timestamp("2015-01-06")
    assert np.isnan(a.loc[nan_day, "max_wind"]) and a.loc[nan_day, "src_max_wind"] is None
    for v in ow.DB_ONLY_NAN:
        assert t2[v].isna().all() and t2["src_" + v].isna().all()
    assert set(t2["src_avg_temp"].dropna()) == {"db"}
    assert json.loads(pq.read_schema(out / bu.OUT_2).metadata[b"same_as_1st"]) is True
    rep = json.loads((out / bu.OUT_REP).read_text(encoding="utf-8"))
    assert rep["db_nan_cells_not_filled"]["AAA"] == {"max_wind": 1}
    # 나라 목록을 주면 그 나라만(DB 에 없는 CCC 는 NaN 행)
    out2 = tmp_path / "dbonly2"
    assert bu.main(["--root", str(root), "--out", str(out2), "--db-only", "--nations", "nations.csv"]) == 0
    t3 = pd.read_parquet(out2 / bu.OUT_2)
    assert sorted(t3.alpha_3.unique()) == ["AAA", "BBB", "CCC"]
    assert t3.loc[t3.alpha_3 == "CCC", list(ow.TABLE_VARS)].isna().all().all()
    with pytest.raises(bu.BuildStop):      # 덮지 않음
        bu.main(["--root", str(root), "--out", str(out), "--db-only"])
