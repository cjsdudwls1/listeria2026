"""해외 제조소 좌표 날씨 추출본(raw_global_factory_weather_listeria_sameday)이 시료 파일 좌표로도 같은 격자점에 붙는지 확인.

추출본은 DB 전처리 시료표 좌표(h_lat·h_lon, double)와 global_factory_weather.lat_g·lon_g 등호로 만든 것.
사용자 규칙(2026-09-17): 매칭은 시료 파일(listeria_body.parquet) lat·lon(float32) 으로 해야 함.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
B = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML")
OUT = Path(__file__).resolve().parent / "results" / "verify_factory_coords.json"
R = {}

body = pd.read_parquet(B / "data" / "listeria_body.parquet", columns=["lims_data_key", "src_route", "lat", "lon", "mnfctur_de"])
body["src_route"] = body["src_route"].astype("string")
ovs = body[body["src_route"] == "overseas"].copy()
fw = pd.read_parquet(B / "data" / "collected" / "raw" / "raw_global_factory_weather_listeria_sameday.parquet")
m = ovs.merge(fw, on="lims_data_key", how="left", validate="1:1")
R["rows"] = len(m)
R["h_eq_grid_double"] = int(((m["h_lat"] == m["lat_g"]) & (m["h_lon"] == m["lon_g"])).sum())
R["sample_eq_grid_as_float64"] = int(((m["lat"].astype("float64") == m["lat_g"]) & (m["lon"].astype("float64") == m["lon_g"])).sum())
R["sample_eq_grid_as_float32"] = int(((m["lat"] == m["lat_g"].astype("float32")) & (m["lon"] == m["lon_g"].astype("float32"))).sum())
R["base_date_eq"] = int((m["mnfctur_de"] == m["base_date"]).sum())

g = m[["lat_g", "lon_g"]].drop_duplicates()
g["lat32"], g["lon32"] = g["lat_g"].astype("float32"), g["lon_g"].astype("float32")
R["grid_points_double"] = len(g)
R["grid_points_float32"] = len(g[["lat32", "lon32"]].drop_duplicates())
dup = g[g.duplicated(["lat32", "lon32"], keep=False)].sort_values(["lat32", "lon32"])
R["collapsed_points"] = dup.astype(float).round(8).to_dict("records")
if len(dup):
    key = m["lat"].astype("float32").isin(dup["lat32"]) & m["lon"].astype("float32").isin(dup["lon32"])
    sub = m[key]
    R["collapsed_rows"] = int(len(sub))
    R["collapsed_rows_detail"] = sub.groupby(["lat_g", "lon_g"]).size().rename("rows").reset_index().astype({"lat_g": float, "lon_g": float}).round(8).to_dict("records")
    la1, lo1, la2, lo2 = (np.radians(dup.iloc[i][c]) for i, c in [(0, "lat_g"), (0, "lon_g"), (1, "lat_g"), (1, "lon_g")])
    R["collapsed_distance_m"] = round(float(2 * 6371000 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))), 3)
    vals = ["temperature_mean", "precipitation_sum", "relative_humidity_mean"]
    R["collapsed_same_date_values"] = sub.groupby(["base_date", "lat_g", "lon_g"])[vals].first().reset_index().head(6).astype(str).to_dict("records")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
