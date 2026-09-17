"""공유용/데이터/LIMS조인_데이터셋.parquet 의 좌표 기반 칼럼이 시료 파일 좌표로 만들어졌는지 전수 대조.

사용자 규칙(2026-09-17): 매칭할 때는 항상 시료 파일(data/listeria_body.parquet) lat·lon 을 쓴다.
방법: 데이터셋에 저장된 관측소 ID·거리(km)를 (1) 시료 파일 좌표 (2) DB 전처리 시료표 좌표(db_lat·db_lon)로
각각 새로 계산해 어느 쪽과 일치하는지 센다. 관측소 후보·관측소 좌표 규칙은 share_build_join.py(커밋 0c5a3ed)와 같게 독립 구현.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
B = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML")
S = B / "공유용" / "데이터"
DBC = S / "DB수집"
OUT = Path(__file__).resolve().parent / "results" / "verify_join_coords.json"
R = {}

# ---------------- 데이터셋 칼럼 목록·설명(파케이 메타데이터)
pf = pq.ParquetFile(S / "LIMS조인_데이터셋.parquet")
cols = [f.name for f in pf.schema_arrow]
R["dataset_rows"], R["dataset_cols"] = pf.metadata.num_rows, len(cols)
R["file_mtime"] = str(pd.Timestamp((S / "LIMS조인_데이터셋.parquet").stat().st_mtime, unit="s", tz="UTC").tz_convert("Asia/Seoul"))
meta = pf.schema_arrow.metadata or {}
dic = None
for k, v in meta.items():
    if k.decode("utf-8", "ignore") == "column_dictionary":
        dic = json.loads(v.decode("utf-8"))
R["dictionary_type"] = type(dic).__name__ if dic is not None else None
if isinstance(dic, list):
    dic = {d.get("컬럼") or d.get("column"): d for d in dic}
kw = ("좌표", "lat", "lon", "위도", "경도", "관측소", "측정소", "거리")
R["dictionary_coord_mentions"] = {c: {k2: v2 for k2, v2 in d.items() if isinstance(v2, str)} for c, d in (dic or {}).items()
                                  if any(w in json.dumps(d, ensure_ascii=False) for w in kw)}

# ---------------- 시료 파일 vs 데이터셋 lat·lon
body = pd.read_parquet(B / "data" / "listeria_body.parquet", columns=["lims_data_key", "lat", "lon"])
need = ["lims_data_key", "src_route", "mnfctur_de", "lat", "lon", "db_lat", "db_lon",
        "asos_stn_id", "asos_dist_km", "nas_obsr_spot_code", "nas_dist_km", "air_station_nm", "air_dist_km"]
X = pd.read_parquet(S / "LIMS조인_데이터셋.parquet", columns=need)
X["src_route"] = X["src_route"].astype("string")
assert (X["lims_data_key"].to_numpy() == body["lims_data_key"].to_numpy()).all(), "행 순서가 시료 파일과 다름"
R["dataset_latlon_equals_body"] = {c: bool(np.array_equal(X[c].to_numpy(), body[c].to_numpy(), equal_nan=True)) for c in ["lat", "lon"]}
R["dataset_latlon_dtype"] = {c: str(X[c].dtype) for c in ["lat", "lon", "db_lat", "db_lon"]}

DOM = (X["src_route"] == "domestic").to_numpy()
N = len(X)
base = pd.to_datetime(X["mnfctur_de"]).dt.normalize().to_numpy()
SAMPLE = (body["lat"].to_numpy(np.float64), body["lon"].to_numpy(np.float64))
DBXY = (X["db_lat"].to_numpy(np.float64), X["db_lon"].to_numpy(np.float64))
exact32 = (X["db_lat"].astype("float32").to_numpy() == body["lat"].to_numpy()) & (X["db_lon"].astype("float32").to_numpy() == body["lon"].to_numpy())
R["coords_differ_domestic_rows"] = int((DOM & ~exact32).sum())


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


def check(prefix, sid, st, daily, dcol):
    st = st.dropna(subset=["lat", "lon"]).drop_duplicates(sid).reset_index(drop=True)
    st[sid] = st[sid].astype(str)
    daily = daily[[sid, dcol]].copy()
    daily[sid] = daily[sid].astype(str)
    daily["_d"] = pd.to_datetime(daily[dcol]).dt.normalize()
    daily = daily[daily[sid].isin(set(st[sid]))].drop_duplicates([sid, "_d"])
    cand = daily.groupby("_d")[sid].apply(lambda s: np.unique(s.to_numpy())).to_dict()
    pos = {s: i for i, s in enumerate(st[sid])}
    SLA, SLO = st["lat"].to_numpy(float), st["lon"].to_numpy(float)
    stored = X[f"{prefix}{sid}"].astype("string").to_numpy(dtype=object)
    stored_d = X[f"{prefix}dist_km"].to_numpy(float)
    res = {}
    for name, (la, lo) in {"시료 파일 좌표": SAMPLE, "DB 좌표(db_lat·db_lon)": DBXY}.items():
        rec = np.full(N, None, dtype=object)
        recd = np.full(N, np.nan)
        ok = DOM & ~np.isnan(la)
        rows_by_day = pd.Series(np.arange(N)[ok]).groupby(base[ok]).apply(lambda s: s.to_numpy()).to_dict()
        for d, rows in rows_by_day.items():
            cs = cand.get(pd.Timestamp(d))
            if cs is None:
                continue
            ci = np.array([pos[s] for s in cs])
            D = hav(la[rows][:, None], lo[rows][:, None], SLA[None, ci], SLO[None, ci])
            j = D.argmin(1)
            rec[rows] = cs[j]
            recd[rows] = D[np.arange(len(rows)), j]
        dom_idx = np.where(DOM)[0]
        same_sid = np.array([(a is None and (b is None or pd.isna(b))) or (a is not None and b is not None and not pd.isna(b) and str(a) == str(b)) for a, b in zip(rec[dom_idx], stored[dom_idx])])
        same_d = np.isclose(np.round(recd[dom_idx], 3), stored_d[dom_idx], atol=1.5e-3, equal_nan=True)
        both = same_sid & same_d
        diff_rows = dom_idx[~both]
        res[name] = {"국내 행": int(len(dom_idx)), "관측소 일치": int(same_sid.sum()), "거리 일치(±0.0015km)": int(same_d.sum()),
                     "둘 다 일치": int(both.sum()), "불일치 행": int(len(diff_rows)),
                     "불일치 행 중 두 좌표가 다른 행": int((~exact32[diff_rows]).sum()) if len(diff_rows) else 0,
                     "불일치 예시": [{"lims_data_key": X["lims_data_key"].iat[i], "저장 관측소": None if pd.isna(stored[i]) else str(stored[i]), "재계산 관측소": rec[i],
                                   "저장 거리": None if np.isnan(stored_d[i]) else float(stored_d[i]), "재계산 거리": None if np.isnan(recd[i]) else round(float(recd[i]), 3)} for i in diff_rows[:3]]}
    ovs_filled = int((~DOM & pd.notna(stored)).sum())
    res["수입 행에 관측소 값 있음"] = ovs_filled
    return res


ast = pd.read_parquet(DBC / "kma_ground_obs_station_info.parquet")
ast["chg_dt"] = pd.to_datetime(ast["chg_dt"], errors="coerce"); ast["reg_dt"] = pd.to_datetime(ast["reg_dt"], errors="coerce")
sA = ast.sort_values(["chg_dt", "reg_dt"], ascending=False, na_position="last").drop_duplicates("stn_id")[["stn_id", "lat", "lon"]].astype({"lat": float, "lon": float})
R["asos"] = check("asos_", "stn_id", sA, pd.read_parquet(DBC / "kma_asos_daily_info.parquet", columns=["stn_id", "meas_date"]), "meas_date")
print("asos", json.dumps(R["asos"], ensure_ascii=False, default=str))

nst = pd.read_parquet(DBC / "nas_agri_weather_obs_station.parquet")
nst["chg_dt"] = pd.to_datetime(nst["chg_dt"], errors="coerce")
sN = nst.sort_values("chg_dt", ascending=False).drop_duplicates("obsr_spot_code")[["obsr_spot_code", "instl_la", "instl_lo"]].rename(columns={"instl_la": "lat", "instl_lo": "lon"}).astype({"lat": float, "lon": float})
R["nas"] = check("nas_", "obsr_spot_code", sN, pd.read_parquet(DBC / "nas_agri_weather_daily_info.parquet", columns=["obsr_spot_code", "meas_date"]), "meas_date")
print("nas", json.dumps(R["nas"], ensure_ascii=False, default=str))

kst = pd.read_parquet(DBC / "keco_airkorea_obs_station.parquet")
kst["chg_dt"] = pd.to_datetime(kst["chg_dt"], errors="coerce"); kst["_u"] = (kst["use_yn"].astype("string") == "1").astype(int)
sK = kst.sort_values(["_u", "chg_dt"], ascending=False).drop_duplicates("station_nm")[["station_nm", "dm_x", "dm_y"]].rename(columns={"dm_x": "lat", "dm_y": "lon"}).astype({"lat": float, "lon": float})
R["air"] = check("air_", "station_nm", sK, pd.read_parquet(DBC / "keco_airkorea_fine_dust_info.parquet", columns=["station_nm", "msur_dt"]), "msur_dt")
print("air", json.dumps(R["air"], ensure_ascii=False, default=str))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: R[k] for k in ["dataset_rows", "dataset_cols", "file_mtime", "dictionary_type", "dataset_latlon_equals_body", "dataset_latlon_dtype", "coords_differ_domestic_rows"]}, ensure_ascii=False, default=str))
print("dictionary coord mentions:", json.dumps(R["dictionary_coord_mentions"], ensure_ascii=False)[:6000])
