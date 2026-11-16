"""국내 기상 규칙(lst/kweather.py)·단계 0-3·0-4 스크립트 시험 — 합성 자료만."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lst import kweather as kw

FINAL = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 참조 구현(칸 하나씩, 03 원자료 대조와 같은 방식)
def ref_cell(lat, lon, day_vals, st_xy, k=3):
    """day_vals: 관측소별 값(NaN = 없음), st_xy: 관측소 좌표. 거리 → 관측소 순서로 정렬해 값 있는 k곳 평균."""
    cand = []
    for j, v in enumerate(day_vals):
        if np.isnan(v):
            continue
        cand.append((float(kw.haversine_km(lat, lon, st_xy[j, 0], st_xy[j, 1])), j, float(v)))
    cand.sort(key=lambda x: (x[0], x[1]))
    top = cand[:k]
    return (float(np.mean([x[2] for x in top])) if top else np.nan), len(top)


def test_knn_mean_matches_reference():
    rng = np.random.default_rng(0)
    n_days, n_st, n_loc = 40, 12, 5
    V = rng.normal(size=(n_days, n_st)).astype(np.float32)
    V[rng.random(V.shape) < 0.4] = np.nan
    V[:, 3] = np.nan  # 값이 전혀 없는 관측소
    V[7, :] = np.nan  # 모든 관측소 결측인 날
    V[8, :] = np.nan
    V[8, 5] = 1.5     # 한 곳만 있는 날
    st1 = np.column_stack([36 + rng.random(n_st), 127 + rng.random(n_st)])
    st2 = st1.copy()
    st2[0] = (36.5, 127.5)  # 20일째부터 관측소 0 위치 바뀜
    loc = np.column_stack([36 + rng.random(n_loc), 127 + rng.random(n_loc)])
    periods = [(0, 20, st1), (20, n_days, st2)]
    for k in (1, 3):
        M, D, C = kw.knn_mean(V, periods, loc, n_days, k=k, block=7)
        for li in range(n_loc):
            for di in range(n_days):
                xy = st1 if di < 20 else st2
                r, n = ref_cell(loc[li, 0], loc[li, 1], V[di].astype(np.float64), xy, k)
                if np.isnan(r):
                    assert np.isnan(M[li, di]) and C[li, di] == 0
                else:
                    assert abs(M[li, di] - r) <= 1e-5 * max(1, abs(r)) and C[li, di] == n
    assert np.isnan(M[:, 7]).all()


def test_tie_breaks_by_station_order():
    V = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    st = np.array([[36.0, 127.1], [36.0, 126.9], [36.0, 127.2], [36.0, 126.8]])  # 0·1 같은 거리, 2·3 같은 거리
    M, _, _ = kw.knn_mean(V, [(0, 1, st)], np.array([[36.0, 127.0]]), 1, k=3)
    assert M[0, 0] == pytest.approx((1 + 2 + 3) / 3)
    M1, _, _ = kw.knn_mean(V, [(0, 1, st)], np.array([[36.0, 127.0]]), 1, k=1)
    assert M1[0, 0] == 1.0


def test_compute_var_fallback_only_where_primary_missing():
    st = np.array([[36.0, 127.0], [36.1, 127.0]])
    nets = {"ASOS": kw.Network("ASOS", ["a", "b"], [(0, 3, st)], {"x": np.array([[1, 2], [np.nan, np.nan], [5, np.nan]], np.float32)}),
            "NAS": kw.Network("NAS", ["c"], [(0, 3, st[:1])], {"y": np.array([[9], [7], [9]], np.float32)})}
    spec = dict(var="v", prim=("ASOS", "x"), fb=("NAS", "y"))
    r = kw.compute_var(nets, spec, np.array([[36.0, 127.0]]), 3)
    assert r["value"][0].tolist() == [1.5, 7.0, 5.0] and r["fb_mask"][0].tolist() == [False, True, False]


def _grid(a, b):
    return kw.DateGrid(np.datetime64(a, "D"), np.datetime64(b, "D"))


def test_daily_table_null_zero_and_duplicates():
    g = _grid("2016-02-27", "2016-03-01")
    t = pa.table({"stn_id": ["1", "1", "2"], "meas_date": pa.array(np.array(["2016-02-28", "2016-02-29", "2016-02-29"], "datetime64[D]"), pa.date32()),
                  "sum_rn": [None, 3.0, None]})
    v, info = kw.daily_table(t, "stn_id", "meas_date", ["sum_rn"], ["1", "2"], "ASOS", g, null_zero={"sum_rn"})
    S = v["sum_rn"]
    assert S[1, 0] == 0.0 and np.isnan(S[1, 1])          # 02-28: 1 행 있음 빈 값 → 0, 2 행 없음 → NaN
    assert S[2].tolist() == [3.0, 0.0]                     # 02-29
    assert np.isnan(S[0]).all() and info["sum_rn_null_as_zero_rows"] == 2
    v2, _ = kw.daily_table(t, "stn_id", "meas_date", ["sum_rn"], ["1", "2"], "ASOS", g)
    assert np.isnan(v2["sum_rn"][1, 0])
    dup = pa.table({"stn_id": ["1", "1"], "meas_date": pa.array(np.array(["2016-02-28"] * 2, "datetime64[D]"), pa.date32()), "sum_rn": [1.0, 2.0]})
    with pytest.raises(RuntimeError):
        kw.daily_table(dup, "stn_id", "meas_date", ["sum_rn"], ["1"], "ASOS", g)


def test_ongjin_coord_rules():
    region = pa.table({"legal_dong_code": ["2800000000", "1100000000", "2872000000"], "level": ["sido", "sido", "sigungu"],
                       "lat": [37.456004, 37.5, 37.95], "lon": [126.705258, 127.0, 124.67]})
    la = [37.2559, 37.2559, 37.9741, 37.456004, None, 37.5, 37.3]
    lo = [126.4833, 126.4833, 124.7186, 126.705258, None, 127.0, 126.0]
    cd = ["28720"] * 6 + ["28110"]
    s = pa.table({"sgg_code": cd, "lat": pa.array(la, pa.float32()), "lon": pa.array(lo, pa.float32())})
    lat, lon, info = kw.ongjin_coord(s, region)
    assert info["rows_all"] == 6 and info["excluded_null_xy"] == 1 and info["excluded_sido_point"] == 2
    assert info["excluded_by_sido_code"] == {"2800000000": 1, "1100000000": 1} and info["used"] == 3
    assert lat == pytest.approx(float(np.float32(37.2559))) and lon == pytest.approx(float(np.float32(126.4833)))
    # 범위 안인데 float32 로 다른 좌표 → 멈춤
    s2 = pa.table({"sgg_code": ["28720"], "lat": pa.array([37.45605], pa.float32()), "lon": pa.array([126.705258], pa.float32())})
    with pytest.raises(ValueError):
        kw.ongjin_coord(s2, region)


def test_compare_matrices():
    a = np.array([[1.0, np.nan, 3.0]], np.float32)
    assert kw.compare_matrices(a, a.copy()) == {"cells": 3, "nan_mismatch": 0, "value_mismatch": 0, "max_abs_diff": 0.0}
    b = np.array([[1.0, 2.0, 3.1]], np.float32)
    r = kw.compare_matrices(a, b)
    assert r["nan_mismatch"] == 1 and r["value_mismatch"] == 1


# ---------------------------------------------------------------- 합성 자료 뿌리(스크립트 시험)
D_MIN, D_MAX = "2016-02-20", "2016-03-05"
SGG = ["11110", "12110", "28720", "41590"]
SGG_XY_OLD = {"11110": (37.57, 126.98), "12110": (34.81, 126.39), "28720": (37.2559, 126.4833), "41590": (37.17, 126.88)}
SAMPLE_ONG = [(37.30, 126.40)] * 5 + [(37.97, 124.72)] * 2 + [(37.456004, 126.705258)] * 4   # 시도청 점 4행 → 중앙값 37.30·126.40


def _w(path: Path, table: pa.Table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def build_root(root: Path, rng):
    dates = np.arange(np.datetime64(D_MIN), np.datetime64(D_MAX) + np.timedelta64(1, "D"), dtype="datetime64[D]")
    R = root / "리스테리아ML2" / "data" / "raw"
    n_as, n_nas, n_air = 8, 10, 9
    as_xy = np.column_stack([34.5 + 3.5 * rng.random(n_as), 125.5 + 2.5 * rng.random(n_as)])
    nas_xy = np.column_stack([34.5 + 3.5 * rng.random(n_nas), 125.5 + 2.5 * rng.random(n_nas)])
    air_xy = np.column_stack([34.5 + 3.5 * rng.random(n_air), 125.5 + 2.5 * rng.random(n_air)])
    # ASOS(행 없는 날 일부, 빈 값 일부)
    asos_cols = ["avg_ta", "max_ta", "min_ta", "avg_ws", "max_ws", "sum_gsr", "avg_cm5_te", "avg_cm10_te", "avg_cm20_tes",
                 "avg_cm30_te", "max_ws1", "avg_rhm", "avg_td", "sum_rn"]
    rows = {c: [] for c in ["stn_id", "meas_date", "sum_sshr"] + asos_cols}
    for s in range(n_as):
        for d in dates:
            if rng.random() < 0.1:
                continue
            rows["stn_id"].append(f"{100 + s}")
            rows["meas_date"].append(d)
            rows["sum_sshr"].append(None if rng.random() < 0.2 else f"{rng.random() * 10:.1f}")
            for c in asos_cols:
                v = float(rng.choice([0, 90, 180, 270, 45])) if c == "max_ws1" else float(rng.normal(10, 5))
                rows[c].append(None if rng.random() < 0.25 else v)
    asos = pa.table({**{k: v for k, v in rows.items() if k != "meas_date"},
                     "meas_date": pa.array(np.array(rows["meas_date"]), pa.date32())})
    _w(R / "raw_kma_asos_daily_info.parquet", asos)
    st = {"stn_id": [], "lat": [], "lon": [], "oper_start_dt": []}
    for s in range(n_as):
        st["stn_id"].append(f"{100 + s}"); st["lat"].append(as_xy[s, 0]); st["lon"].append(as_xy[s, 1])
        st["oper_start_dt"].append(np.datetime64("2010-01-01"))
    st["stn_id"].append("100"); st["lat"].append(37.0); st["lon"].append(126.0); st["oper_start_dt"].append(np.datetime64("2016-02-25"))  # 위치 바뀜
    _w(R / "raw_kma_ground_obs_station_info.parquet",
       pa.table({"stn_id": st["stn_id"], "lat": st["lat"], "lon": st["lon"],
                 "oper_start_dt": pa.array(np.array(st["oper_start_dt"], "datetime64[D]"), pa.date32())}))
    # NAS(2/29 행 없음), 561034A001 포함(빠져야 함)
    nas_cols = ["temp", "hghst_artmp", "lowst_artmp", "hum", "wind", "max_wind", "rn", "srqty", "soil_temp", "soil_wt"]
    rows = {c: [] for c in ["obsr_spot_code", "meas_date"] + nas_cols}
    codes = [f"N{s:02d}" for s in range(n_nas - 1)] + ["561034A001"]
    for s, code in enumerate(codes):
        for d in dates:
            if str(d)[5:] == "02-29" or rng.random() < 0.1:
                continue
            rows["obsr_spot_code"].append(code)
            rows["meas_date"].append(d)
            for c in nas_cols:
                rows[c].append(None if rng.random() < 0.2 else float(rng.normal(20, 5)))
    _w(R / "raw_nas_agri_weather_daily_info.parquet",
       pa.table({**{k: v for k, v in rows.items() if k != "meas_date"}, "meas_date": pa.array(np.array(rows["meas_date"]), pa.date32())}))
    _w(R / "raw_nas_agri_weather_obs_station.parquet",
       pa.table({"obsr_spot_code": codes, "instl_la": nas_xy[:, 0], "instl_lo": nas_xy[:, 1]}))
    # 에어코리아(좌표 없는 측정소 1곳)
    air_cols = ["no2_value", "co_value", "so2_value"]
    rows = {c: [] for c in ["station_nm", "msur_dt"] + air_cols}
    names = [f"측정소{s}" for s in range(n_air)] + ["좌표없음"]
    for s, nm in enumerate(names):
        for d in dates:
            if rng.random() < 0.1:
                continue
            rows["station_nm"].append(nm)
            rows["msur_dt"].append(d)
            for c in air_cols:
                rows[c].append(None if rng.random() < 0.2 else float(rng.random()))
    _w(R / "raw_keco_airkorea_fine_dust_info.parquet",
       pa.table({**{k: v for k, v in rows.items() if k != "msur_dt"}, "msur_dt": pa.array(np.array(rows["msur_dt"]), pa.date32())}))
    _w(R / "raw_keco_airkorea_obs_station.parquet",
       pa.table({"station_nm": names[:-1] + [names[0]], "dm_x": list(air_xy[:, 0]) + [air_xy[0, 0]], "dm_y": list(air_xy[:, 1]) + [air_xy[0, 1]]}))
    # 법정동 표·시료·대표 좌표
    _w(R / "raw_korea_region_code.parquet",
       pa.table({"legal_dong_code": ["2800000000", "2872000000"], "level": ["sido", "sigungu"], "lat": [37.456004, 37.95], "lon": [126.705258, 124.67]}))
    ds = root / "리스테리아ML2" / "data" / "데이터셋"
    la = [x[0] for x in SAMPLE_ONG] + [37.57]
    lo = [x[1] for x in SAMPLE_ONG] + [126.98]
    _w(ds / "시료_시군구.parquet", pa.table({"sgg_code": ["28720"] * len(SAMPLE_ONG) + ["11110"],
                                          "lat": pa.array(la, pa.float32()), "lon": pa.array(lo, pa.float32())}))
    coord = pa.table({"sgg_code": SGG, "sgg_nm": [f"이름{c}" for c in SGG], "lat": [SGG_XY_OLD[c][0] for c in SGG],
                      "lon": [SGG_XY_OLD[c][1] for c in SGG], "source": ["korea_region_code"] * 4})
    _w(ds / "시군구_대표좌표.parquet", coord)
    # 기존 기상표 = 같은 규칙(옛 좌표)으로 만든 표, 2/29 의 NAS 변수는 원래 빈 칸
    grid = _grid(D_MIN, D_MAX)
    paths = {"asos_daily": R / "raw_kma_asos_daily_info.parquet", "asos_station": R / "raw_kma_ground_obs_station_info.parquet",
             "nas_daily": R / "raw_nas_agri_weather_daily_info.parquet", "nas_station": R / "raw_nas_agri_weather_obs_station.parquet",
             "air_daily": R / "raw_keco_airkorea_fine_dust_info.parquet", "air_station": R / "raw_keco_airkorea_obs_station.parquet"}
    nets = kw.load_networks(paths, kw.VARIABLES, grid)
    xy = np.array([SGG_XY_OLD[c] for c in SGG])
    cols = {"sgg_code": np.repeat(np.array(SGG, dtype=object), grid.n).tolist(), "date": pa.array(np.tile(grid.dates, len(SGG)), pa.date32())}
    for spec in kw.VARIABLES:
        cols[spec["var"]] = pa.array(kw.compute_var(nets, spec, xy, grid.n)["value"].ravel(), pa.float32(), from_pandas=True)
    _w(ds / "식정원_db_기상기후" / "시군구_일별_기상.parquet", pa.table(cols))
    # 예보 표(추출본 모양)
    ex = root / "리스테리아ML2" / "data" / "추출_20261107"
    issue = np.datetime64("2016-03-01", "D")
    one = np.timedelta64(1, "D")
    fl = {"sigungu_code": [], "day_offset": [], "forecast_date": []}
    fcols = ["tmprt_150", "tmprt_150Top", "tmprt_150Lwet", "hd_150", "arvlty_300", "arvlty_300Top", "afp", "일조", "solrad_Qy", "soil_Mitr_10", "미세먼지"]
    for c in fcols:
        fl[c] = []
    kcodes = ["11110", "28720", "41590", "46110"]   # 12110 연결 안 됨
    kxy = {"11110": (37.571, 126.989), "28720": (37.537, 125.601), "41590": (37.174, 126.884), "46110": (34.812, 126.387)}
    for c in kcodes:
        for o in range(11):
            fl["sigungu_code"].append(c); fl["day_offset"].append(o); fl["forecast_date"].append(issue + o * one)
            for f in fcols:
                fl[f].append(float(rng.normal(10, 3)))
    flt = pa.table({**{k: v for k, v in fl.items() if k != "forecast_date"},
                    "forecast_date": pa.array(np.array(fl["forecast_date"]), pa.date32()),
                    "pm_src": ["forecast"] * len(fl["day_offset"]), "sun_src": ["station"] * len(fl["day_offset"])})
    _w(ex / "p0830_fcst_long.parquet", flt)
    fi = {k: [] for k in ["predict_seq", "meas_date", "forecast_date", "lat", "lon", "temp", "hghst_artmp", "lowst_artmp", "hum",
                          "wind", "max_wind", "precip_mm", "sun_time", "soil_wt"]}
    seq = 0
    for c in kcodes:
        for o in range(5):
            seq += 1
            fi["predict_seq"].append(seq); fi["meas_date"].append(issue); fi["forecast_date"].append(issue + o * one)
            fi["lat"].append(kxy[c][0]); fi["lon"].append(kxy[c][1])
            for f in ["temp", "hghst_artmp", "lowst_artmp", "hum", "wind", "max_wind", "precip_mm", "sun_time", "soil_wt"]:
                fi[f].append(float(rng.normal(10, 3)))
    fit = pa.table({**{k: v for k, v in fi.items() if k not in ("meas_date", "forecast_date")},
                    "meas_date": pa.array(np.array(fi["meas_date"]), pa.date32()),
                    "forecast_date": pa.array(np.array(fi["forecast_date"]), pa.date32()),
                    "error_msg": pa.array([None] * seq, pa.string()),
                    "created_at": pa.array(np.array(["2016-03-02T00:00:00"] * seq, "datetime64[us]"), pa.timestamp("us"))})
    _w(ex / "kma_weather_predict_info.parquet", fit)
    _w(ex / "kma_sigungu_point.parquet", pa.table({"sigungu_code": kcodes, "sigungu_nm": [f"예보{c}" for c in kcodes],
                                                   "lat": [kxy[c][0] for c in kcodes], "lon": [kxy[c][1] for c in kcodes]}))
    # 공개 지연 원천(SPI CSV·해외)
    ext = root / "리스테리아ML2" / "data" / "외부_데이터"
    ext.mkdir(parents=True, exist_ok=True)
    with open(ext / "CLM_SPI_DD_합본_20140101_20261006.csv", "w", encoding="cp949", newline="") as f:
        w = csv.writer(f)
        w.writerow(["지점", "지점명", "일시", "SPI1", "SPI3", "SPI6", "SPI12"])
        for s in ("90", "100"):
            for d in dates[:-1 if s == "100" else None]:
                w.writerow([s, "속초", str(d), ".1", ".2", "" if str(d) == D_MAX else ".3", ".4"])
    gw_cols = ["temperature_mean", "temperature_max", "temperature_min", "precipitation_sum", "wind_speed_mean", "wind_speed_max",
               "sunshine_duration", "shortwave_radiation_sum", "relative_humidity_mean", "soil_moisture_mean"]
    gw = {"alpha_3": ["CHN"] * len(dates) + ["USA"] * (len(dates) - 3), "date": list(dates) + list(dates[:-3])}
    for c in gw_cols:
        gw[c] = [1.0] * len(gw["alpha_3"])
    _w(ex / "global_weather_info.parquet", pa.table({**{k: v for k, v in gw.items() if k != "date"},
                                                     "date": pa.array(np.array(gw["date"]), pa.date32())}))
    (ex / "global_weather_info.json").write_text(json.dumps({"started_at": "2016-03-08T00:50:56+09:00"}), encoding="utf-8")
    return grid


def _override(root, date="2016-03-01"):
    """합성 ASOS 에서 avg_td 값이 있는 첫 관측소(그 날) 한 칸 — raw_overrides 시험 대상."""
    a = pq.read_table(root / "리스테리아ML2/data/raw/raw_kma_asos_daily_info.parquet").to_pandas()
    r = a[(a.meas_date.astype(str) == date) & a.avg_td.notna()].sort_values("stn_id").iloc[0]
    return [{"input": "asos_daily", "station": r.stn_id, "date": date, "column": "avg_td", "value": None, "reason": "시험"}]


def test_apply_overrides():
    g = _grid("2016-02-27", "2016-03-01")
    V = {"avg_td": np.arange(8, dtype=np.float32).reshape(4, 2)}
    V["avg_td"][0, 1] = np.nan
    o = {"input": "asos_daily", "station": "b", "date": "2016-02-28", "column": "avg_td", "value": None, "reason": "r"}
    done = kw.apply_overrides(V, ["a", "b"], g, [o], "ASOS")
    assert np.isnan(V["avg_td"][1, 1]) and V["avg_td"][1, 0] == 2 and done[0]["원래 값"] == 3.0 and done[0]["적용 칸 수"] == 1
    for bad in ({"station": "x"}, {"date": "2016-03-05"}, {"column": "avg_ta"}, {"value": 1.0},
                {"date": "2016-02-27"},   # 이미 결측 칸 = 대상 칸 없음
                {"date": "2016-02-28"}):  # 위에서 바꿔 이미 결측
        with pytest.raises((ValueError, IndexError)):
            kw.apply_overrides({k: v.copy() for k, v in V.items()}, ["a", "b"], g, [{**o, **bad}], "ASOS")
    with pytest.raises(ValueError):   # 같은 칸 두 번
        kw.apply_overrides({"avg_td": np.ones((4, 2), np.float32)}, ["a", "b"], g, [o, o], "ASOS")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, FINAL / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.check_resources = lambda path: {}   # 로컬 PC 는 RAM 바닥값(15GB) 아래 — 시험에서만 끔
    return mod


def _cfg(tmp_path, name, **over):
    cfg = json.loads((FINAL / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    cfg.update(over)
    p = tmp_path / f"{name}_test.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    grid = build_root(root, np.random.default_rng(42))
    return root, grid


def test_s0_4_end_to_end(synth, tmp_path):
    root, grid = synth
    mod = _load_script("s0_4_weather_domestic")
    cfg = _cfg(tmp_path, "s0_4_weather_domestic", feb29_dates=["2016-02-29"], check_from="2016-02-21", check_cv_end="2016-03-03",
               raw_overrides=_override(root))
    out = tmp_path / "out4"
    mod.main(["--root", str(root), "--out", str(out), "--config", str(cfg)])
    coord = pq.read_table(out / "시군구_대표좌표_0-4.parquet").to_pydict()
    i = coord["sgg_code"].index("28720")
    assert coord["lat"][i] == pytest.approx(float(np.float32(37.30))) and coord["lon"][i] == pytest.approx(float(np.float32(126.40)))
    assert coord["lat"][coord["sgg_code"].index("11110")] == 37.57
    old = pq.read_table(root / "리스테리아ML2/data/데이터셋/식정원_db_기상기후/시군구_일별_기상.parquet").to_pandas()
    mg = pq.read_table(out / "시군구_일별_기상_0-4_합본.parquet").to_pandas()
    assert list(mg.columns) == ["sgg_code", "date", *kw.VARS, "rh", "td"] and len(mg) == len(old)
    # 옹진군 밖·2/29 밖은 기존 표 그대로
    is29 = mg["date"].astype(str) == "2016-02-29"
    keep = (mg.sgg_code != "28720") & ~is29
    for v in kw.VARS:
        a, b = mg.loc[keep, v].to_numpy(), old.loc[keep, v].to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-6)
    # 2/29 채움: 기존 표는 빈 칸
    assert old.loc[is29, "rain"].isna().all() and mg.loc[is29, "rain"].notna().all()
    for c in mg.sgg_code.unique():
        r28 = mg[(mg.sgg_code == c) & (mg.date.astype(str) == "2016-02-28")].iloc[0]
        r29 = mg[(mg.sgg_code == c) & is29].iloc[0]
        assert r29.soil_moist == pytest.approx(r28.soil_moist)
        assert r29.soil_t_nas == pytest.approx(r28.soil_t_nas + (r29.soil_t10 - r28.soil_t10), abs=1e-4)
    # 2/29 rain = ASOS sum_rn(빈 값 0) 가까운 3곳 — 참조 구현으로 한 칸 확인
    asos = pq.read_table(root / "리스테리아ML2/data/raw/raw_kma_asos_daily_info.parquet").to_pandas()
    st = pq.read_table(root / "리스테리아ML2/data/raw/raw_kma_ground_obs_station_info.parquet").to_pandas()
    day = asos[asos.meas_date.astype(str) == "2016-02-29"]
    sxy, vals = [], []
    for sid in sorted(asos.stn_id.unique()):
        rr = st[st.stn_id == sid].sort_values("oper_start_dt")
        ok = rr[rr.oper_start_dt.astype(str) <= "2016-02-29"]
        r = ok.iloc[-1] if len(ok) else rr.iloc[0]
        sxy.append((r.lat, r.lon))
        x = day[day.stn_id == sid]
        vals.append(np.nan if x.empty else (0.0 if np.isnan(x.sum_rn.iloc[0]) else x.sum_rn.iloc[0]))
    ref, _ = ref_cell(37.57, 126.98, np.array(vals, dtype=np.float32).astype(np.float64), np.array(sxy))
    got = mg[(mg.sgg_code == "11110") & is29].rain.iloc[0]
    assert got == pytest.approx(ref, rel=1e-5)
    # 습도·이슬점 파일 = 합본 rh·td
    rt = pq.read_table(out / "시군구_일별_습도이슬점_0-4.parquet").to_pandas()
    assert np.allclose(rt.rh.to_numpy(), mg.rh.to_numpy(), equal_nan=True)
    ong = pq.read_table(out / "옹진군_일별_기상_0-4.parquet").to_pandas()
    assert (ong.sgg_code == "28720").all() and len(ong) == grid.n
    chk = json.loads((out / "점검.json").read_text(encoding="utf-8"))
    assert chk["옹진군 좌표"]["excluded_sido_point"] == 4 and chk["2/29 채움"]["2016-02-29"]["rain"]["채운 칸"] == 4
    assert (out / "run_meta.json").exists() and (out / "점검_변수별.csv").exists()
    # raw_overrides: 한 칸 결측으로 둔 뒤 3곳 평균 — td 를 같은 방식으로 다시 계산해 대조, 적용 칸 수 기록
    ov = _override(root)
    assert chk["원자료 바꾼 칸(raw_overrides)"]["적용 칸 수"] == 1 and chk["원자료 바꾼 칸(raw_overrides)"]["칸"][0]["station"] == ov[0]["station"]
    paths = {k: root / "리스테리아ML2/data/raw" / f for k, f in [("asos_daily", "raw_kma_asos_daily_info.parquet"),
             ("asos_station", "raw_kma_ground_obs_station_info.parquet")]}
    xy = kw.coords_in_order(pq.read_table(out / "시군구_대표좌표_0-4.parquet"), sorted(mg.sgg_code.unique()))
    plain = kw.compute_var(kw.load_networks(paths, [kw.SPECS["td"]], grid), kw.SPECS["td"], xy, grid.n)["value"]
    over = kw.compute_var(kw.load_networks(paths, [kw.SPECS["td"]], grid, ov), kw.SPECS["td"], xy, grid.n)["value"]
    got = rt.td.to_numpy().reshape(len(xy), grid.n)
    assert np.allclose(got, over, equal_nan=True) and not np.allclose(got, plain, equal_nan=True)
    with pytest.raises(FileExistsError):   # 다시 돌리면 덮지 않고 멈춤
        mod.main(["--root", str(root), "--out", str(out), "--config", str(cfg)])


def test_s0_4_stops_when_table_differs(synth, tmp_path):
    """기존 표가 규칙과 다르면(옮긴 규칙 확인) 멈춤."""
    root, _ = synth
    import shutil
    root2 = tmp_path / "root2"
    shutil.copytree(root, root2)
    p = root2 / "리스테리아ML2/data/데이터셋/식정원_db_기상기후/시군구_일별_기상.parquet"
    t = pq.read_table(p).to_pandas()
    t.loc[5, "avg_temp"] = t.loc[5, "avg_temp"] + 1 if not np.isnan(t.loc[5, "avg_temp"]) else 1.0
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), p)
    mod = _load_script("s0_4_weather_domestic")
    cfg = _cfg(tmp_path, "s0_4_weather_domestic", feb29_dates=["2016-02-29"], check_from="2016-02-21", check_cv_end="2016-03-03",
               raw_overrides=_override(root))
    with pytest.raises(RuntimeError):
        mod.main(["--root", str(root2), "--out", str(tmp_path / "o"), "--config", str(cfg)])


def test_s0_4_stops_when_override_target_missing(synth, tmp_path):
    root, _ = synth
    mod = _load_script("s0_4_weather_domestic")
    bad = [{**_override(root)[0], "station": "없는관측소"}]
    cfg = _cfg(tmp_path, "s0_4_weather_domestic", feb29_dates=["2016-02-29"], check_from="2016-02-21", check_cv_end="2016-03-03",
               raw_overrides=bad)
    with pytest.raises(ValueError):
        mod.main(["--root", str(root), "--out", str(tmp_path / "o"), "--config", str(cfg)])
    real = json.loads((FINAL / "configs" / "s0_4_weather_domestic.json").read_text(encoding="utf-8"))["raw_overrides"]
    assert real == [{"input": "asos_daily", "station": "295", "date": "2025-06-14", "column": "avg_td", "value": None,
                     "reason": real[0]["reason"]}]


def test_s0_3_end_to_end(synth, tmp_path):
    root, grid = synth
    mod = _load_script("s0_3_forecast_prep")
    cfg = _cfg(tmp_path, "s0_3_forecast_prep", diff_from="2016-02-21", diff_to="2016-03-03")
    out = tmp_path / "out3"
    mod.main(["--root", str(root), "--out", str(out), "--config", str(cfg)])

    def rd(n):
        with open(out / n, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    mp = rd("예보_변수대응.csv")
    have = {r["변수"]: r["예보에 있음"] for r in mp if r["변수"] != "(우리 변수에 대응 없음)"}
    assert have["td"] == "예보에 없음" and have["soil_t5"] == "예보에 없음" and have["avg_temp"] == "있음" and have["rh"] == "있음"
    assert any(r["p0830_fcst_long 칼럼"] == "미세먼지" for r in mp)
    unl = rd("예보좌표_연결안됨.csv")
    assert [r["sgg_code"] for r in unl] == ["12110"] and unl[0]["예보 좌표 sigungu_code"] == "46110"
    days = json.loads((out / "예보_11일_확인.json").read_text(encoding="utf-8"))
    assert days["fcst_long"]["day_offset 0~10 이 다 있는 시군구 수"] == 4
    # 차이 = 예보 좌표 1곳 − 대표 좌표 3곳(옹진군 = 새 좌표) 평균 — 한 시군구·한 변수 다시 계산
    diff = rd("예보좌표_차이.csv")
    vars_ = {r["변수"] for r in diff}
    assert "td" not in vars_ and "soil_t5" not in vars_ and {"avg_temp", "rain", "rh", "soil_moist"} <= vars_
    paths = {k: root / "리스테리아ML2/data/raw" / f for k, f in [("asos_daily", "raw_kma_asos_daily_info.parquet"),
             ("asos_station", "raw_kma_ground_obs_station_info.parquet"), ("nas_daily", "raw_nas_agri_weather_daily_info.parquet"),
             ("nas_station", "raw_nas_agri_weather_obs_station.parquet"), ("air_daily", "raw_keco_airkorea_fine_dust_info.parquet"),
             ("air_station", "raw_keco_airkorea_obs_station.parquet")]}
    nets = kw.load_networks(paths, [kw.SPECS["avg_temp"]], grid)
    ong_new = (float(np.float32(37.30)), float(np.float32(126.40)))
    m3 = kw.compute_var(nets, kw.SPECS["avg_temp"], np.array([ong_new]), grid.n, 3)["value"][0]
    m1 = kw.compute_var(nets, kw.SPECS["avg_temp"], np.array([(37.537, 125.601)]), grid.n, 1)["value"][0]
    i0, i1 = grid.index("2016-02-21"), grid.index("2016-03-03")
    a = (m1 - m3)[i0:i1 + 1].astype(np.float64)
    row = [r for r in diff if r["sgg_code"] == "28720" and r["변수"] == "avg_temp"][0]
    assert float(row["평균 차이(1곳 − 3곳)"]) == pytest.approx(np.nanmean(a), rel=1e-6, abs=1e-9)
    assert int(row["두 값 다 있는 날 수"]) == int((~np.isnan(a)).sum())
    dl = {r["원천"]: r for r in rd("공개지연_원천별.csv")}
    g = dl["해외 나라 기상(global_weather_info)"]
    assert g["마지막 날"] == D_MAX and g["받은 날 − 마지막 날(일)"] == "3" and g["최근 365일 안 지점: 마지막 날보다 1일 앞에서 끝"] == "0" and g["3~7일"] == "1"
    assert dl["ASOS 일자료"]["받은 시각"] == "" and dl["기상청 SPI 일자료"]["2일"] == "0" and dl["기상청 SPI 일자료"]["최근 365일 안 지점: 마지막 날보다 1일 앞에서 끝"] == "1"
    dc = {(r["원천"], r["칼럼"]): r for r in rd("공개지연_칼럼별.csv")}
    assert dc[("기상청 SPI 일자료", "SPI6")]["원천 마지막 날과 차이(일)"] == "1"
    chk = rd("예보_대응확인_비교.csv")
    assert any(r["예보 칼럼"] == "temp" and int(r["비교 쌍 수"]) > 0 for r in chk)
