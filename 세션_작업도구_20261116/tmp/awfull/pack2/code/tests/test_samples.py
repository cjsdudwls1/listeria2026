"""단계 0-1·0-2 시료 표(lst.samples·lst.geo·lst.theil·scripts/s0_1_samples.py) 시험 — 합성 자료만."""
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lst import folds as fd
from lst import samples as S
from lst.geo import CountryPolygons, haversine_km
from lst.hassign import assign_h
from lst.theil import theils_u

CODE_ROOT = Path(__file__).resolve().parents[1]


# ================================================================ geo
def sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def feat(code, geom, key="ISO_A3_EH"):
    props = {"ISO_A3_EH": "-99", "ADM0_A3": code, "ADMIN": code}
    if key == "ISO_A3_EH":
        props["ISO_A3_EH"] = code
    return {"type": "Feature", "properties": props, "geometry": geom}


FEATURES = [
    feat("KOR", {"type": "Polygon", "coordinates": [sq(126, 35, 128, 38)]}),
    feat("JPN", {"type": "Polygon", "coordinates": [sq(129.5, 31, 131, 34)]}),
    # 구멍 있는 나라 + 같은 코드 다각형 둘(합쳐 한 나라) + ADM0_A3 로 코드
    feat("USA", {"type": "Polygon", "coordinates": [sq(-100, 30, -90, 40), sq(-96, 34, -94, 36)]}, key="ADM0"),
    feat("USA", {"type": "MultiPolygon", "coordinates": [[sq(-150, 60, -140, 65)]]}, key="ADM0"),
    feat("ZZA", {"type": "Polygon", "coordinates": [sq(0, 0, 2, 2)]}),
    feat("ZZB", {"type": "Polygon", "coordinates": [sq(1, 1, 3, 3)]}),  # ZZA 와 겹침 → ambiguous
]


@pytest.fixture(scope="module")
def polys():
    return CountryPolygons.from_features(FEATURES)


def test_geo_inside_hole_multi_nearest_ambiguous(polys):
    lat = [36.5, 32.0, 35.0, 37.0, 62.0, 1.5, 36.0, np.nan]
    lon = [127.0, 130.0, -95.0, -98.0, -145.0, 1.5, 128.05, 127.0]
    code, method, dist = polys.assign(lat, lon)
    assert list(code[:2]) == ["KOR", "JPN"]
    assert method[2] == "nearest" and code[2] == "USA" and 0 < dist[2] < 200  # 구멍 안 → 가장 가까운 나라
    assert code[3] == "USA" and method[3] == "inside" and code[4] == "USA"  # 다각형 둘을 한 나라로
    assert method[5] == "ambiguous" and code[5] is None
    assert code[6] == "KOR" and method[6] == "nearest"
    assert abs(dist[6] - 0.05 * 111.195 * math.cos(math.radians(36.0))) < 0.01
    assert method[7] == "no_coord" and code[7] is None


def test_geo_matches_shapely_on_random_points(polys):
    shapely = pytest.importorskip("shapely.geometry")
    rng = np.random.default_rng(0)
    lat = rng.uniform(25, 45, 3000)
    lon = rng.uniform(-105, -85, 3000)
    hits = polys.inside_codes(lat, lon)
    g = shapely.shape(FEATURES[2]["geometry"])
    exp = [g.contains(shapely.Point(x, y)) for x, y in zip(lon, lat)]
    assert [h == ["USA"] for h in hits] == exp


def test_haversine():
    assert abs(haversine_km(0, 0, 0, 1)[()] - 111.195) < 0.01


# ================================================================ theil
def _nb_conditional_entropy(x, y):  # 부록 B-8 노트북 셀 그대로
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total = sum(y_counter.values())
    e = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total
        p_y = y_counter[xy[1]] / total
        e += p_xy * math.log(p_y / p_xy)
    return e


def _nb_theil_u(x, y):
    s_xy = _nb_conditional_entropy(x, y)
    xc = Counter(x)
    tot = sum(xc.values())
    p = [n / tot for n in xc.values()]
    s_x = -sum(q * math.log(q) for q in p)
    return 1 if s_x == 0 else (s_x - s_xy) / s_x


def test_theil_matches_notebook():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    x = rng.choice(["a", "b", "c", None], 2000).tolist()
    x = ["NA" if v is None else v for v in x]  # 노트북 Counter 는 None 도 한 값 — 같은 범주로 맞춤
    assert abs(theils_u(y, x) - _nb_theil_u(y.tolist(), x)) < 1e-12
    xf = np.where(rng.random(2000) < 0.2, np.nan, rng.integers(0, 5, 2000)).astype(float)
    xs = ["NA" if np.isnan(v) else v for v in xf]
    assert abs(theils_u(y, xf) - _nb_theil_u(y.tolist(), xs)) < 1e-12  # NaN = 한 범주
    assert theils_u(np.zeros(5, int), list("abcde")) == 1.0
    assert abs(theils_u([0, 1, 0, 1], ["a", "b", "a", "b"]) - 1.0) < 1e-12


# ================================================================ 나라 원천·주소·B-6 지표
NC = pd.DataFrame({"name_en": ["Korea, Republic of", "Japan", "United States of America", "Italy"],
                   "name_ko": ["대한민국", "일본", "미국", "이탈리아"],
                   "alpha_2": ["KR", "JP", "US", "IT"], "alpha_3": ["KOR", "JPN", "USA", "ITA"]})


def test_resolve_db_nation_order_and_unmappable():
    a2, a3 = S.nation_maps(NC)
    src = pd.DataFrame({"mnf_nation_cd": [None, "JP", None, None, None],
                        "prdctn_nation_code": [None, "US", None, None, None],
                        "orgplce_nation_code": ["KOREA", None, "ITA", None, None],
                        "import_nation_code": ["US", None, None, None, "XX"]})
    v, name, usage = S.resolve_db_nation(src, S.NATION_SOURCES, a2, a3)
    assert list(v) == ["USA", "JPN", "ITA", None, None]
    assert list(name) == ["import_nation_code", "mnf_nation_cd", "orgplce_nation_code", None, None]
    u = usage.set_index("source")
    assert u.loc["orgplce_nation_code", "unmappable_value"] == 1 and u.loc["import_nation_code", "decided_here"] == 1
    v2, _, _ = S.resolve_db_nation(src, ["mnf_nation_cd", "orgplce_nation_code"], a2, a3)  # 원천을 줄이면
    assert list(v2) == [None, "JPN", "ITA", None, None]


def test_address_countries():
    got = S.address_countries(["1 Main St, Ohio, United States of America", "Via Roma, ITALY", "KOREA? no",
                                "Japan and Italy", None], NC)
    assert got == [["USA"], ["ITA"], ["KOR"], ["ITA", "JPN"], []]


def test_column_check_and_gap_names():
    y = [1, 1, 0, 0, 0, 0]
    x = ["a", None, "b", "b", None, "c"]
    r = S.column_check(x, y)
    assert r["blank_rate_pos"] == 0.5 and r["blank_rate_neg"] == 0.25
    assert r["n_unique"] == 4 and r["pos_only_values"] == 1  # 'a' 만(결측은 불검출에도 있음)
    assert r["pos_only_value_share"] == 0.25 and r["pos_only_row_share_in_pos"] == 0.5
    t21 = pd.DataFrame({"column": ["a", "b", "c"], "blank_gap": [0.1, -0.2, np.nan]})
    t25 = pd.DataFrame({"column": ["a", "b", "c"], "blank_gap": [0.1, -0.3, 0.5]})
    assert S.gap_increased_names(t21, t25, "column") == ["b"]


# ================================================================ 끝에서 끝(합성 뿌리)
STUB_02 = '''
import pyarrow.parquet as pq
class Region:
    def __init__(self, path):
        self.n = pq.read_metadata(path).num_rows
def load_mapping(path):
    return {}, {}
def assign_sgg(c10s, addrs=None, region=None, mapping=None, with_reason=False):
    assert region is not None and mapping is not None
    return [c[:5] if c.startswith("11") else None for c in c10s]
'''

SAMPLE_COLS = ["lims_data_key", "hazard_label", "judge_result", "src_route", "data_source", "expr_iem_nm",
               "result_value", "result_value_num", "judge_stdr", "judge_stdr_num", "inspct_instt_code",
               "prdlst_code", "train_prdlst_code", "train_prdlst_nm", "train_prdlst_code_2", "train_prdlst_nm_2",
               "prduct_nm", "mnfctur_cmpny_nm", "mnfctur_de", "date_basis", "addr", "addr_source",
               "legal_dong_code", "lat", "lon"]


def make_root(root: Path, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2015-07-02", "2026-06-30", freq="D")

    def add(key, route, judge, d, lat, lon, addr="addr", ldc=None, p2="C01", p3="C0101"):
        rows.append({"lims_data_key": key, "hazard_label": "리스테리아", "judge_result": judge, "src_route": route,
                     "data_source": "식약처", "expr_iem_nm": "리스테리아", "result_value": "음성",
                     "result_value_num": np.nan, "judge_stdr": None, "judge_stdr_num": np.nan,
                     "inspct_instt_code": f"I{rng.integers(0, 5)}", "prdlst_code": p3, "train_prdlst_code": p2,
                     "train_prdlst_nm": "n2", "train_prdlst_code_2": p3, "train_prdlst_nm_2": "n3",
                     "prduct_nm": f"제품{rng.integers(0, 50)}", "mnfctur_cmpny_nm": None,
                     "mnfctur_de": pd.Timestamp(d), "date_basis": "제조일", "addr": addr, "addr_source": "업소주소",
                     "legal_dong_code": ldc, "lat": lat, "lon": lon})

    for i in range(400):  # 국내
        add(f"D{i:04d}", "domestic", "검출" if i % 25 == 0 else "불검출", dates[rng.integers(0, len(dates))],
            36.0 + rng.random(), 127.0 + rng.random(), ldc="1111010100")
    add("R2A", "domestic", "불검출", "2020-03-03", 37.0, 127.0, ldc="2811000000")   # RR2
    add("R2B", "domestic", "불검출", "2021-03-03", 37.0, 127.0, ldc="2811000000")   # RR2
    add("R3", "domestic", "불검출", "2020-05-05", 36.5, 127.5, ldc="1111010100", p2=None, p3=None)  # RR3
    add("R3b", "domestic", "불검출", "2020-05-06", 36.5, 127.5, ldc="1111010100", p3=None)          # RR3 한쪽
    add("R7", "domestic", "불검출", "2022-07-01", 36.5, 127.5, ldc="1111010100")                    # RR7
    add("R8", "domestic", "불검출", "2015-04-07", 36.5, 127.5, ldc="1111010100")                    # RR8
    for i in range(60):  # 해외(일본·미국·이탈리아 좌표)
        cc = ["JPN", "USA", "ITA"][i % 3]
        la, lo = {"JPN": (32.5, 130.0), "USA": (37.0, -98.0), "ITA": (43.0, 12.0)}[cc]
        add(f"O{i:04d}", "overseas", "검출" if i % 20 == 0 else "불검출", dates[rng.integers(0, len(dates))], la, lo,
            p2="B02", p3="B0204000000000" if i == 5 else "B0201")
    add("K1", "overseas", "불검출", "2019-01-01", 36.9, 126.9, addr="SEOUL, KOREA")   # RR1 → 11
    add("K2", "overseas", "불검출", "2019-01-02", 35.3, 127.9, addr="BUSAN, KOREA")   # KOR 좌표·02 미배정
    add("U1", "overseas", "불검출", "2020-01-01", 37.0, -98.0)  # DB 원천 없음 → 좌표(USA)
    add("U2", "overseas", "불검출", "2020-01-02", 10.0, 50.0, addr="Rome, Italy")  # 바다 → 가장 가까운 나라
    add("KR1", "overseas", "불검출", "2020-02-02", 32.5, 130.0)  # 제조국 KR(일본 좌표) → OVKOR, RR5
    df = pd.DataFrame(rows, columns=SAMPLE_COLS)
    df["lat"] = df["lat"].astype(np.float32)
    df["lon"] = df["lon"].astype(np.float32)
    for c in ["hazard_label", "judge_result", "src_route", "data_source", "train_prdlst_code"]:
        df[c] = df[c].astype("category")
    p = root / "리스테리아ML/data"
    p.mkdir(parents=True)
    df.to_parquet(p / "listeria_data.parquet", index=False)
    dom = df[(df["src_route"] == "domestic") & ~df["lims_data_key"].isin(["R2A", "R2B"])].copy()
    dom["sgg_code"] = "11110"
    q = root / "리스테리아ML2/data/데이터셋"
    q.mkdir(parents=True)
    dom.to_parquet(q / "시료_시군구.parquet", index=False)
    m = root / "리스테리아ML2/data/중간"
    m.mkdir(parents=True)
    pd.DataFrame({"사유코드": ["A"], "사유": ["x"], "원래_시군구코드": ["28110"], "원래_이름": ["중구"],
                  "행수": [2], "검출수": [0]}).to_csv(m / "시군구_미배정_삭제행_요약.csv", index=False, encoding="utf-8-sig")
    nb = root / "리스테리아ML2/notebook"
    nb.mkdir(parents=True)
    (nb / "02_시군구_배정.py").write_text(STUB_02, encoding="utf-8")
    raw = root / "리스테리아ML2/data/raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"legal_dong_code": ["1111010100", "2611010100", "2611000000"],
                  "legal_dong_nm": ["서울 종로구 청운동", "부산 중구 영주동", "부산 중구"],
                  "molit_use_yn": ["존재", "존재", "존재"], "level": ["eupmyeondong", "eupmyeondong", "sigungu"],
                  "lat": [37.0, 35.2, 36.9], "lon": [127.0, 128.0, 126.9]}).to_parquet(raw / "raw_korea_region_code.parquet")
    (root / "리스테리아ML2/data/시군구_코드_대응표.xlsx").write_bytes(b"stub")
    ex = root / "리스테리아ML2/data/추출_20261107"
    ex.mkdir(parents=True)
    keys = df["lims_data_key"].tolist()
    ov = df["src_route"].astype(str).eq("overseas").to_numpy()
    cntc = [f"R{k}" if o and k not in ("U1", "U2", "K1", "K2") else None for k, o in zip(keys, ov)]
    pd.DataFrame({"lims_data_key": keys, "hazard_label": "리스테리아", "cntc_sys_key2": cntc,
                  "prdctn_nation_code": None, "orgplce_nation_code": ["KOREA" if k == "K1" else None for k in keys],
                  "import_nation_code": None}).to_parquet(ex / "hazard_preprocessed_data_nation.parquet")
    nat = {"JPN": "JP", "USA": "US", "ITA": "IT"}
    so = []
    for k, c in zip(keys, cntc):
        if c is None:
            continue
        code = "KR" if k == "KR1" else nat[["JPN", "USA", "ITA"][int(k[1:]) % 3]]
        so += [{"lims_seq": len(so), "rcept_no": c, "label": "리스테리아", "mnf_nation_cd": code, "mnf_nation_nm": "x"}] * 2
    pd.DataFrame(so).to_parquet(ex / "s_org_lims_data_info_listeria.parquet")
    NC.to_parquet(ex / "nation_code.parquet")
    ne = root / "리스테리아ML2/data/외부_수집/축사"
    ne.mkdir(parents=True)
    feats = FEATURES[:4] + [feat("ITA", {"type": "Polygon", "coordinates": [sq(7, 37, 18, 47)]}),
                            feat("SOM", {"type": "Polygon", "coordinates": [sq(41, 0, 51, 9.9)]})]
    (ne / "ne_10m_admin_0_countries_v5.1.2.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return df


def load_script():
    spec = importlib.util.spec_from_file_location("s0_1_samples", CODE_ROOT / "scripts" / "s0_1_samples.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.check_resources = lambda path: {"note": "시험 — 자원 검사 생략(로컬 PC RAM < 15GB)"}
    return mod


def recorded_from(df):
    """합성 자료의 §3 '기록값'(RR2 제외, 원 src_route) — 대조가 통과하는 경우를 만들려고."""
    d = df[~df["lims_data_key"].isin(["R2A", "R2B"])]
    days = fd.to_days(d["mnfctur_de"])
    y = (d["judge_result"].astype(str) == "검출").astype(int).to_numpy()
    return {n: fd.count_split(days, y, d["src_route"].astype(str).to_numpy(), sp) for n, sp in fd.SPLITS.items()}


@pytest.fixture()
def synth(tmp_path):
    root = tmp_path / "root"
    df = make_root(root)
    return root, df, load_script()


def test_check_stage(synth, tmp_path):
    root, df, mod = synth
    out = tmp_path / "out"
    d = mod.run(["--stage", "check", "--root", str(root), "--out", str(out)], recorded_split=recorded_from(df))
    files = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}
    assert "samples.parquet" not in files and "run_meta.json" in files
    assert any(f.startswith(mod.SEALED_DIR + "/") for f in files)
    names = json.loads((d / "05_b6_names_gap_increased_d_le_2025-06-30.json").read_text(encoding="utf-8"))
    assert set(names) >= {"columns", "nation_sources"}
    assert all(isinstance(x, str) for x in names["columns"] + names["nation_sources"])  # 이름만
    c21 = pd.read_csv(d / "05_b6_columns_d_le_2021-06-30.csv")
    assert list(c21["column"]) == list(S.THEIL_CANDIDATES)
    rr1 = pd.read_csv(d / "04_rr1_coords.csv", dtype={"sgg_by_02_rule": str})
    assert len(rr1) == 2 and set(rr1["n_moved"]) == {0, 1}
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok" and meta["stage"] == "check" and len(meta["inputs_md5"]) == len(S.INPUTS)
    with pytest.raises(FileExistsError):  # 덮어쓰기 금지
        mod.run(["--stage", "check", "--root", str(root), "--out", str(out)], recorded_split=recorded_from(df))


def test_split_mismatch_stops(synth, tmp_path):
    root, df, mod = synth
    rec = recorded_from(df)
    rec["fold1"]["train_n"] += 1
    out = tmp_path / "out"
    with pytest.raises(S.PlanCountMismatch):
        mod.run(["--stage", "check", "--root", str(root), "--out", str(out)], recorded_split=rec)
    t = pd.read_csv(out / "check" / "03_split_counts_vs_plan.csv")
    assert (~t["match"]).sum() == 1
    meta = json.loads((out / "check" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "stopped" and "PlanCountMismatch" in meta["error"]


def _cfg(tmp_path, **kw):
    c = {"nation_sources": list(S.NATION_SOURCES), "theil_columns": ["prduct_nm", "addr"], "confirmed": True,
         "confirmation_note": "시험"}
    c.update(kw)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    return p


def test_apply_stage(synth, tmp_path):
    root, df, mod = synth
    out = tmp_path / "out"
    d = mod.run(["--stage", "apply", "--config", str(_cfg(tmp_path)), "--root", str(root), "--out", str(out)],
                recorded_split=recorded_from(df))
    t = pd.read_parquet(d / "samples.parquet")
    assert list(t.columns) == list(S.SAMPLE_COLUMNS)
    assert pq.read_schema(d / "samples.parquet").field("d").type == pa.date32()
    T = t.set_index("lims_data_key")
    ex = pd.read_parquet(d / "excluded_rows.parquet").set_index("lims_data_key")["rule"].to_dict()
    assert ex == {"R2A": "RR2", "R2B": "RR2", "R8": "RR8"}
    # RR1: KOR 좌표 + 02 규칙 배정 → 국내·sgg 11110, 배정 안 되면 해외로 남음(좌표 나라 KOR)
    assert T.loc["K1", "src_route"] == "domestic" and T.loc["K1", "sgg"] == "11110" and T.loc["K1", "rr1_moved"]
    assert pd.isna(T.loc["K1", "nation_a3"])
    assert T.loc["K2", "src_route"] == "overseas" and T.loc["K2", "sgg"] == "OVKOR"
    assert T.loc["K2", "nation_src"] == "coord_inside"
    # RR4: 좌표 나라 / 바다 → 가장 가까운 나라
    assert T.loc["U1", "sgg"] == "OVUSA" and T.loc["U1", "nation_src"] == "coord_inside"
    assert T.loc["U2", "sgg"] == "OVSOM" and T.loc["U2", "nation_src"] == "coord_nearest"
    # RR5: 제조국 KR 인데 좌표는 일본 → 나라 = 제조국
    assert T.loc["KR1", "sgg"] == "OVKOR" and T.loc["KR1", "rr5_coord_differs"] == True  # noqa: E712
    assert T.loc["O0000", "rr5_coord_differs"] == False and pd.isna(T.loc["D0001", "rr5_coord_differs"])  # noqa: E712
    assert T.loc["O0001", "sgg"] == "OVUSA" and T.loc["O0001", "sido"] == "OV" and T.loc["D0001", "sido"] == "11"
    # RR3·RR6·RR7, h
    assert T.loc["R3", "p2"] == "코드없음" and T.loc["R3", "p3"] == "코드없음" and T.loc["R3b", "p2"] == "C01"
    assert T.loc["R3b", "rr3_p3_nocode"] and not T.loc["R3b", "rr3_p2_nocode"]
    assert T.loc["O0005", "rr6_overseas_only_p3"] and T.loc["R7", "rr7_jul01"]
    assert all(T.loc[k, "h"] == assign_h(k) for k in ["D0001", "K1", "O0003"])
    assert (T["y"] == (df.set_index("lims_data_key").loc[T.index, "judge_result"].astype(str) == "검출")).all()
    rr = pd.read_csv(d / "10_rr_table.csv").set_index("item")
    assert rr.loc["RR1 국내로 옮긴 행", "observed"] == "1" and rr.loc["RR8 제외 행", "observed"] == "1"
    assert rr.loc["OVKOR(RR1 뒤 남은 해외 행 중 나라 KOR)", "observed"] == "2"
    inc = json.loads((d / "12_incomplete_months.json").read_text(encoding="utf-8"))
    assert isinstance(inc["incomplete"], list)
    nt = pd.read_csv(d / "13_nation_table_d_le_2021-06-30.csv")
    assert set(nt["d_range"]) == {"<= 2021-06-30"}
    sub = t[(t["src_route"] == "overseas") & (pd.to_datetime(t["d"]) <= "2021-06-30")]
    assert nt["n_samples"].sum() == len(sub) and nt["n_pos"].sum() == sub["y"].sum()
    assert json.loads((d / "14_theil_target_columns.json").read_text(encoding="utf-8"))["columns"] == ["prduct_nm", "addr"]


def test_apply_with_fewer_sources(synth, tmp_path):
    root, df, mod = synth
    d = mod.run(["--stage", "apply", "--config", str(_cfg(tmp_path, nation_sources=["orgplce_nation_code"])),
                 "--root", str(root), "--out", str(tmp_path / "out")], recorded_split=recorded_from(df))
    T = pd.read_parquet(d / "samples.parquet").set_index("lims_data_key")
    assert T.loc["O0001", "nation_src"] == "coord_inside" and T.loc["KR1", "sgg"] == "OVJPN"


@pytest.mark.parametrize("kw", [
    {"confirmed": False},
    {"confirmation_note": " "},
    {"nation_sources": ["import_nation_code", "mnf_nation_cd"]},   # A-2 순서 아님
    {"theil_columns": ["addr", "prduct_nm"]},                       # A-6 순서 아님
    {"theil_columns": ["prduct_nm", "y"]},                          # 계획에 없는 칼럼
    {"theil_columns": []},
    {"nation_sources": ["mnf_nation_cd", "mnf_nation_cd"]},
])
def test_apply_config_rejects(tmp_path, kw):
    mod = load_script()
    with pytest.raises(Exception):
        mod.load_apply_config(_cfg(tmp_path, **kw))


def test_unconfirmed_config_stops_and_real_config_loads(tmp_path):
    mod = load_script()
    real = CODE_ROOT / "configs" / "s0_1_samples.json"
    cfg = json.loads(real.read_text(encoding="utf-8"))
    assert mod.load_apply_config(real)["confirmed"] is True
    for patch in ({"confirmed": False}, {"confirmation_note": "  "}):
        p = tmp_path / f"cfg_{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps({**cfg, **patch}, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(Exception):
            mod.load_apply_config(p)


def test_judge_result_other_value_stops():
    df = pd.DataFrame({"lims_data_key": ["a", "b"], "judge_result": ["검출", "보류"], "src_route": ["domestic"] * 2,
                       "mnfctur_de": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    with pytest.raises(ValueError):
        S.prepare_basic(df)
    df["judge_result"] = ["검출", "불검출"]
    df.loc[1, "mnfctur_de"] = pd.Timestamp("2020-01-02 13:00")
    with pytest.raises(ValueError):
        S.prepare_basic(df)
