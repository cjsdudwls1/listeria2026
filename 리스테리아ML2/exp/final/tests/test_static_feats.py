"""고정표·조회형 파생변수 시험(합성 자료) — §5 표·§5-2·부록 A-3·A-4·B-6."""
import io
import json
import math
import zipfile

import numpy as np
import pandas as pd
import pytest

from lst import static_feats as sf
from lst.folds import to_days


def D(s):
    return int(to_days(s)[0])


# ---------------------------------------------------------------- FD1 원문 해석
@pytest.mark.parametrize("raw,num,label", [
    ("0.95", 0.95, "단일값"), (0.95, 0.95, "단일값"), ("0.90~0.96", 0.93, "중앙"), ("4.0-5.0", 4.5, "중앙"),
    (">0.98", 0.98, "초과"), ("≤ 0.6", 0.6, "이하"), ("0.98<", 0.98, "초과"), ("0.85 이상", 0.85, "이상"),
])
def test_parse_db_values(raw, num, label):
    x, _, lab = sf.parse_db(raw)
    assert x == pytest.approx(num) and lab == label


@pytest.mark.parametrize("raw", ["-0.85", "-", "", None, float("nan"), "약 0.9"])
def test_parse_db_missing(raw):
    x, _, lab = sf.parse_db(raw)
    assert math.isnan(x) and lab == ""


def _cl_aw():
    # 3단계 A(자기값)·B(하위 2개 중앙값)·C(2단계값)·D(결측). B 아래 B1·B2(B2 아래 B21 — 손자도 하위)
    cl = pd.DataFrame({
        "prdlst_cd": ["P", "A", "B", "B1", "B2", "B21", "C", "D", "Q"],
        "hrnk_prdlst_cd": [None, "P", "P", "B", "B", "B2", "Q", "Q", None],
    })
    aw = pd.DataFrame({
        "prdlst_cd": ["A", "B1", "B21", "Q", "P", "C"],
        "aw": ["0.95", "0.90", "0.99", ">0.98", "0.5", "0.97"],
        "ph": ["6.0", "4~5", "7.0", "6.5", None, "5.5"],
        "na": [None, None, None, None, None, None],
        "aw_source": ["original", "original", "original", "original", "original", "estimated"],
    })
    return cl, aw


def test_build_food_table_rules():
    cl, aw = _cl_aw()
    p2_of = {"A": "P", "B": "P", "C": "Q", "D": "Q"}
    ft = sf.build_food_table(cl, aw, ["D", "A", "B", "C"], p2_of)
    t = ft.table.set_index("p3")
    assert list(ft.table.columns) == list(sf.FOOD_TABLE_COLUMNS)
    assert list(ft.table["p3"]) == ["A", "B", "C", "D"]
    # A 자기값
    assert t.loc["A", "food_aw"] == 0.95 and t.loc["A", "aw_label"] == "단일값" and t.loc["A", "aw_rule"] == "자기값"
    # B 하위중앙: B1 0.90, B21 0.99 → 0.945
    assert t.loc["B", "food_aw"] == pytest.approx(0.945) and t.loc["B", "aw_label"] == "하위중앙"
    assert t.loc["B", "aw_n_sub"] == 2
    assert t.loc["B", "food_ph"] == pytest.approx((4.5 + 7.0) / 2)
    # C: 자기 aw 는 aw_source ≠ original 이라 안 씀 → 2단계 Q '>0.98' (초과), ph 는 자기값 5.5
    assert t.loc["C", "food_aw"] == 0.98 and t.loc["C", "aw_label"] == "초과" and t.loc["C", "aw_rule"] == "2단계값"
    assert t.loc["C", "food_ph"] == 5.5 and t.loc["C", "ph_rule"] == "자기값"
    # D: 하위 없음 → 2단계 Q
    assert t.loc["D", "aw_rule"] == "2단계값"
    # na 전부 결측, label null
    assert t["food_na"].isna().all() and t["na_label"].isna().all() and (t["na_rule"] == "결측").all()
    # rg 품목 4
    assert t.loc["A", "rg_aw_lim"] == 1.0 and t.loc["A", "rg_aw_opt_dist"] == pytest.approx(0.02, abs=1e-6)
    assert t.loc["A", "rg_ph_opt_dist"] == pytest.approx(1.0)


def test_build_food_table_rejects():
    cl, aw = _cl_aw()
    with pytest.raises(ValueError):
        sf.build_food_table(cl, aw, ["A", sf.NO_CODE], {"A": "P", sf.NO_CODE: "P"})
    with pytest.raises(ValueError):
        sf.build_food_table(cl, aw, ["Z"], {"Z": "P"})
    with pytest.raises(ValueError):
        sf.build_food_table(cl, aw, ["A"], {})


def test_pair_p2_most_rows_then_code():
    p3 = ["X", "X", "X", "Y", "Y", None]
    p2 = ["b", "a", "b", "c", "a", "z"]
    m = sf.pair_p2(p3, p2)
    assert m == {"X": "b", "Y": "a"}


def test_rg_item_bounds():
    r = sf.rg_item([0.90, 0.997, 0.8999, np.nan], [4.0, 9.6, 9.61, np.nan])
    assert r["rg_aw_lim"].tolist()[:3] == [1.0, 1.0, 0.0] and np.isnan(r["rg_aw_lim"][3])
    assert r["rg_ph_lim"].tolist()[:3] == [1.0, 1.0, 0.0] and np.isnan(r["rg_ph_opt_dist"][3])
    assert r["rg_aw_lim"].dtype == np.float32


def test_food_features_unknown_code():
    cl, aw = _cl_aw()
    t = sf.build_food_table(cl, aw, ["A"], {"A": "P"}).table
    f = sf.food_features(t, ["A", sf.NO_CODE, "A"])
    assert list(f.columns) == list(sf.FD1_COLUMNS + sf.RG_ITEM_COLUMNS)
    assert f["food_aw"].tolist()[0] == 0.95 and np.isnan(f["food_aw"][1])
    assert f["aw_label"].tolist() == ["단일값", None, "단일값"]
    assert np.isnan(f["rg_aw_lim"][1])


# ---------------------------------------------------------------- L1
NAMES = pd.DataFrame({"sgg_code": ["36110", "12730", "41591", "41597", "52111"],
                      "sgg_nm": ["세종특별자치시", "전남광주통합특별시 구례군", "경기도 화성시 만세구", "경기도 화성시 동탄구",
                                 "전북특별자치도 전주시 완산구"]})


def test_match_domestic_farms():
    farm = pd.DataFrame({"시도": ["세종특별자치시", "전남광주통합특별시", "경기도"], "시군구": ["세종특별자치시", "구례군", "화성시 만세구"],
                         "농장수": [600, 487, 1016]})
    m = sf.match_domestic_farms(farm, NAMES)
    assert m["sgg_code"].tolist() == ["36110", "12730", "41591"]
    bad = farm.assign(시군구=["세종특별자치시", "구례군", "화성시"])
    with pytest.raises(ValueError):
        sf.match_domestic_farms(bad, NAMES)


def _molit_json():
    cols = {"data": [{"DATA_DIV_ID": 1, "DATA_DIV_NM": "시도", "UP_DATA_DIV_ID": 0},
                     {"DATA_DIV_ID": 2, "DATA_DIV_NM": "시군구", "UP_DATA_DIV_ID": 0},
                     {"DATA_DIV_ID": 3, "DATA_DIV_NM": "계", "UP_DATA_DIV_ID": 0},
                     {"DATA_DIV_ID": 32, "DATA_DIV_NM": "면적", "UP_DATA_DIV_ID": 3}]}
    rows = [("전국", "전국", 1e11), ("세종", "합계", 4.6e8), ("세종특별자치시", "세종특별자치시", 4.6e8),
            ("전남", "구례군", 4.4e8), ("경남", "구례군", 1.0e8), ("전북", "전주시(계)", 2.0e8), ("전북", "전주시완산구", 9.2e7),
            ("경기", "화성시", 7.0e8)]
    data = {"data": [{"0": "2025", "1": a, "2": b, "32": str(v)} for a, b, v in rows]}
    return json.dumps(cols).encode(), json.dumps(data).encode()


def test_molit_area_match():
    c, d = _molit_json()
    mt = sf.molit_area_table(c, d, 2025)
    assert set(mt["sgg_name"]) == {"세종특별자치시", "구례군", "전주시완산구", "화성시"}
    abbr = {"세종특별자치시": ["세종특별자치시"], "전남광주통합특별시": ["전남", "광주"], "경기도": ["경기"], "전북특별자치도": ["전북"]}
    a = sf.match_domestic_area(NAMES, mt, abbr).set_index("sgg_code")
    assert a.loc["36110", "area_km2"] == pytest.approx(460.0)
    assert a.loc["12730", "area_km2"] == pytest.approx(440.0)   # 경남 구례군(가짜)과 섞이지 않음
    assert a.loc["52111", "area_km2"] == pytest.approx(92.0)
    assert np.isnan(a.loc["41591", "area_km2"]) and np.isnan(a.loc["41597", "area_km2"])  # 화성시 구 = 지적통계에 없음
    with pytest.raises(ValueError):
        sf.molit_area_table(c, d, 2024)


def test_fao_land_area_and_m49():
    csv_txt = ("Area Code,Area Code (M49),Area,Item Code,Item,Element Code,Element,Year Code,Year,Unit,Value,Flag,Note\n"
               "1,'410,Korea,6601,Land area,5110,Area,2024,2024,1000 ha,9760.0,I,\n"
               "2,'158,Taiwan,6601,Land area,5110,Area,2024,2024,1000 ha,3541.0,I,\n"
               "2,'158,Taiwan,6601,Land area,5110,Area,2023,2023,1000 ha,3540.0,I,\n"
               "3,'036,Australia,6610,Agricultural land,5110,Area,2024,2024,1000 ha,1.0,I,\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Inputs_LandUse_E_All_Data_(Normalized).csv", csv_txt.encode("utf-8-sig"))
    nc = pd.DataFrame({"nation_code": ["410", "158", "36 "], "alpha_3": ["KOR", "TWN", "AUS"]})
    t = sf.fao_land_area(buf.getvalue(), sf.m49_map(nc), "6601", "5110", "1000 ha", 2024).set_index("alpha_3")
    assert t.loc["KOR", "area_km2"] == pytest.approx(97600.0) and t.loc["TWN", "area_km2"] == pytest.approx(35410.0)
    assert "AUS" not in t.index


def test_l1_value_and_lookup():
    assert sf.l1_value([600], [460.0])[0] == pytest.approx(600 / 460 * 10)
    t = pd.DataFrame({"unit": ["36110", "OVUSA"], "L1": [1.5, np.nan]})
    v = sf.livestock_lookup(t, ["36110", "11110", "OVUSA", "OVKOR"])
    assert v[0] == 1.5 and np.isnan(v[1:]).all()


# ---------------------------------------------------------------- SO1
def _soil():
    dates = pd.date_range("2020-01-01", "2020-01-05").date
    rows = []
    for u in ("11110", "11140"):
        for i, d in enumerate(dates):
            rows.append((u, d, 6.0 + i if u == "11110" else np.nan, 20.0, 100.0 + i))
    return pd.DataFrame(rows, columns=["sgg_code", "date", "soil_ph", "soil_om", "soil_vldpha"])


def test_soil_lookup():
    so = sf.SoilLookup(_soil(), pd.DataFrame({"iso3": ["USA"], "ph_mean": [6.8]}))
    v = so(["11110", "11110", "11140", "OVUSA", "OVKOR", "99999"],
           [D("2020-01-03"), D("2019-12-31"), D("2020-01-02"), D("2020-01-01"), D("2020-01-01"), D("2020-01-01")])
    assert v["soil_ph"][0] == 8.0 and v["soil_vldpha"][0] == 102.0
    assert np.isnan(v["soil_ph"][1])                   # 표 날짜 밖
    assert np.isnan(v["soil_ph"][2]) and v["soil_om"][2] == 20.0
    assert v["soil_ph"][3] == 6.8 and np.isnan(v["soil_om"][3]) and np.isnan(v["soil_vldpha"][3])
    assert np.isnan(v["soil_ph"][4]) and np.isnan(v["soil_ph"][5])


def test_soil_lookup_requires_full_grid():
    with pytest.raises(ValueError):
        sf.SoilLookup(_soil().iloc[1:], pd.DataFrame({"iso3": [], "ph_mean": []}))


def test_soil_direct_window():
    T = 2000
    days = np.array([T - 1095, T - 1096, T - 1, T, T - 10])
    vals = np.array([1.0, 100.0, 3.0, 100.0, np.nan])
    m, n = sf.soil_direct(days, vals, T)
    assert n == 2 and m == 2.0


# ---------------------------------------------------------------- X1·X2
def _ind():
    # 연도 2018: 사용가능일 2019-01-01(한 행은 2019-01-05 — 연도 사용가능일 = 가장 늦은 값), 2019: 2020-01-01, 2020: 2021-01-01
    return sf.YearIndicator.from_long(
        "x", ["A", "B", "A", "A", "B"], [2018, 2018, 2019, 2020, 2020], [0.1, 0.5, np.nan, 0.3, 0.7],
        ["2019-01-01", "2019-01-05", "2020-01-01", "2021-01-01", "2021-01-01"])


def test_year_indicator_y_star():
    ind = _ind()
    ys = ind.y_star([D("2019-01-05"), D("2019-01-06"), D("2021-01-01"), D("2021-01-02")])
    assert ys.tolist() == [-1, 2018, 2019, 2020]   # 사용가능일 ≤ T−1


def test_year_indicator_mean_and_nan():
    ind = _ind()
    T = [D("2019-01-06"), D("2020-01-02"), D("2021-01-02"), D("2021-01-02"), D("2021-01-02"), D("2019-01-05")]
    v = ind(["A", "A", "A", "B", "OVUSA", "A"], T)
    assert v[0] == pytest.approx(0.1)
    assert v[1] == pytest.approx(0.1)               # 2019 값 없음 → 있는 해(2018)만
    assert v[2] == pytest.approx(0.2)               # (0.1 + 0.3) / 2
    assert v[3] == pytest.approx(0.6)
    assert np.isnan(v[4]) and np.isnan(v[5])


def test_year_indicator_span_five():
    years = list(range(2010, 2017))
    ind = sf.YearIndicator.from_long("x", ["A"] * 7, years, [float(y - 2009) for y in years],
                                     [f"{y + 1}-01-01" for y in years])
    v = ind(["A"], [D("2017-01-02")])               # y* = 2016 → 2012~2016 = 3..7
    assert v[0] == pytest.approx(5.0)


def test_eco_gw_indicator():
    eco = pd.DataFrame({"sgg_code": ["A", "A"], "자료연도": [2016, 2017], "사용가능일": pd.to_datetime(["2018-01-01", "2019-01-01"]).date,
                        "유기_인증면적": [1.0, 2.0], "무농약_인증면적": [1.0, 0.0], "경지면적": [10.0, 0.0], "친환경_비율": [0.2, np.nan]})
    ind, info = sf.eco_indicator(eco)
    assert info["사용가능일_파일과_다른_행"] == 0 and info["경지면적_0_행"] == 1
    assert ind(["A"], [D("2019-01-02")])[0] == pytest.approx(0.2)
    bad = eco.assign(사용가능일=pd.to_datetime(["2018-01-02", "2019-01-01"]).date)
    assert sf.eco_indicator(bad)[1]["사용가능일_파일과_다른_행"] == 1
    gw = pd.DataFrame({"sgg_code": ["A", "A"], "자료연도": [2013, 2014], "사용가능일": pd.to_datetime(["2014-12-31", "2015-12-31"]).date,
                       "검사_건수": [10.0, 0.0], "적합_건수": [9.0, 0.0], "적합률": [0.9, np.nan]})
    g, ginfo = sf.gw_indicator(gw)
    assert ginfo["검사_0_행"] == 1
    assert g(["A"], [D("2016-01-01")])[0] == pytest.approx(0.9)   # 2014 비율 NaN → 2013 만
    assert np.isnan(g(["A"], [D("2014-12-31")])[0])               # 사용가능일 = T → 못 씀


def test_year_indicator_duplicate():
    with pytest.raises(ValueError):
        sf.YearIndicator.from_long("x", ["A", "A"], [2018, 2018], [0.1, 0.2], ["2019-01-01", "2019-01-01"])


# ---------------------------------------------------------------- WX7
def test_season():
    s = sf.season(["2020-03-01", "2020-05-31", "2020-06-01", "2020-08-31", "2020-09-01", "2020-11-30",
                   "2020-12-01", "2021-01-15", "2021-02-28"])
    assert s.tolist() == ["봄", "봄", "여름", "여름", "가을", "가을", "겨울", "겨울", "겨울"]
