"""해외 국가 단위 표(global_weather_info, global_soil_ph)를 시료 파일에 붙였을 때 연결률·결측 점검.

DB 접속 장애(2026-09-17)로 로컬 추출본만 사용한다. 값은 바꾸지 않고(단위 변환·결측 채움 없음) 연결 여부만 센다.
- 시료: data/listeria_body.parquet (676,537행) + data/collected/00_body_db.parquet (국가 코드 3종, DB 몸통 추출본)
- 국가 코드 변환표: data/collected/61_nation_code.parquet (ba_data_info.nation_code 전체)
- global_weather_info: data/collected/raw/raw_global_weather_info.parquet (전체 1,047,294행)
- global_soil_ph: data/collected/raw/raw_global_soil_ph.parquet (전체 235행)
- global_factory_fine_dust, global_fine_dust_info: 로컬 추출본 없음 → DB 탐색 때 받은 열 정의만 출력
조인 규칙은 노트북 3(build_nb3.py 3-3, 6-6)과 같다: 국가 = 생산국 → 원산지 → 수입국 순 첫 값(alpha-2) → alpha-3, 날짜 = 시료 파일 mnfctur_de.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML")
COL = BASE / "data" / "collected"
OUT = Path(__file__).resolve().parent / "results" / "global_join_coverage.json"

R = {}


def hav(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


# ---------------------------------------------------------------- 시료
body = pd.read_parquet(BASE / "data" / "listeria_body.parquet",
                       columns=["lims_data_key", "judge_result", "src_route", "mnfctur_de", "date_basis", "addr_source", "lat", "lon"])
for c in ["judge_result", "src_route", "date_basis", "addr_source"]:
    body[c] = body[c].astype("string")
bdb = pd.read_parquet(COL / "00_body_db.parquet",
                      columns=["lims_data_key", "geo_route", "mnfctur_de_or_fallback", "prdctn_nation_code", "orgplce_nation_code", "import_nation_code"])
assert bdb["lims_data_key"].is_unique
df = body.merge(bdb, on="lims_data_key", how="left", validate="1:1")
assert len(df) == len(body)
R["judge_values"] = df["judge_result"].value_counts(dropna=False).to_dict()
R["route_crosstab"] = pd.crosstab(df["src_route"], df["geo_route"].fillna("(DB 몸통에 없음)")).to_dict()
R["date_same_as_db_fallback"] = int((df["mnfctur_de"] == df["mnfctur_de_or_fallback"]).sum())

ovs = df[df["src_route"] == "overseas"].copy()
N = len(ovs)
POS = "검출"
npos = int((ovs["judge_result"] == POS).sum())
R["overseas_rows"], R["overseas_pos"] = N, npos
R["overseas_date_basis"] = ovs["date_basis"].value_counts(dropna=False).to_dict()
R["overseas_date_null"] = int(ovs["mnfctur_de"].isna().sum())
R["overseas_date_range"] = [str(ovs["mnfctur_de"].min()), str(ovs["mnfctur_de"].max())]

# 국가 코드 3종 채워짐·서로 다름
NC = ["prdctn_nation_code", "orgplce_nation_code", "import_nation_code"]
for c in NC:
    ovs[c] = ovs[c].astype("string").str.strip().replace("", pd.NA)
R["nation_col_filled"] = {c: int(ovs[c].notna().sum()) for c in NC}
R["nation_col_filled_domestic"] = {c: int(df.loc[df["src_route"] == "domestic", c].notna().sum()) for c in NC}
pair = {}
for i, a in enumerate(NC):
    for b in NC[i + 1:]:
        both = ovs[a].notna() & ovs[b].notna()
        pair[f"{a}|{b}"] = {"both": int(both.sum()), "differ": int((both & (ovs[a] != ovs[b])).sum())}
R["nation_pair_differ"] = pair
ovs["nation_a2"] = ovs["prdctn_nation_code"].fillna(ovs["orgplce_nation_code"]).fillna(ovs["import_nation_code"])
ovs["nation_src"] = np.select([ovs["prdctn_nation_code"].notna(), ovs["orgplce_nation_code"].notna(), ovs["import_nation_code"].notna()],
                              ["생산국", "원산지", "수입국"], default="없음")
R["nation_src"] = ovs["nation_src"].value_counts().to_dict()

nation = pd.read_parquet(COL / "61_nation_code.parquet")
for c in nation.columns:
    nation[c] = nation[c].astype("string").str.strip()
assert nation["alpha_2"].is_unique
A2A3 = dict(zip(nation["alpha_2"], nation["alpha_3"]))
ovs["nation_a3"] = ovs["nation_a2"].map(A2A3)
R["a2_not_convertible"] = ovs.loc[ovs["nation_a2"].notna() & ovs["nation_a3"].isna(), "nation_a2"].value_counts().to_dict()
R["not_convertible_by_col"] = {c: ovs.loc[ovs[c].notna() & ~ovs[c].isin(list(A2A3)), c].value_counts().to_dict() for c in NC}
# 국가 코드 3종이 모두 없는 행: 수입 검사 원본(s_org) 제조국 코드로 찾을 수 있는지 (조인 키 rcept_no = cntc_sys_key2 또는 lims_data_key 두 번째 토큰)
sorg = pd.read_parquet(COL / "02e_s_org.parquet", columns=["rcept_no", "mnf_nation_cd"])
sorg["mnf_nation_cd"] = sorg["mnf_nation_cd"].astype("string").str.strip().replace("", pd.NA)
sn = sorg.dropna(subset=["mnf_nation_cd"]).groupby("rcept_no")["mnf_nation_cd"].agg(["first", "nunique"])
bdb_k = pd.read_parquet(COL / "00_body_db.parquet", columns=["lims_data_key", "cntc_sys_key2"])
ovs = ovs.merge(bdb_k, on="lims_data_key", how="left")
ovs["tok2"] = ovs["lims_data_key"].str.split("_").str[1]
ovs["sorg_nation"] = ovs["cntc_sys_key2"].map(sn["first"]).fillna(ovs["tok2"].map(sn["first"]))
nonat = ovs[ovs["nation_a2"].isna()]
R["no_nation_rows"] = {"rows": len(nonat), "pos": int((nonat["judge_result"] == POS).sum()),
                       "date_basis": nonat["date_basis"].value_counts().to_dict(), "addr_source": nonat["addr_source"].value_counts().to_dict(),
                       "has_latlon": int(nonat["lat"].notna().sum()), "sorg_nation_found": int(nonat["sorg_nation"].notna().sum()),
                       "sorg_nation_top": nonat["sorg_nation"].value_counts().head(8).to_dict(),
                       "sorg_nation_found_all_overseas": int(ovs["sorg_nation"].notna().sum()),
                       "sorg_vs_rule_differ": int((ovs["sorg_nation"].notna() & ovs["nation_a2"].notna() & (ovs["sorg_nation"] != ovs["nation_a2"])).sum())}
R["top_countries"] = ovs["nation_a3"].value_counts().head(15).to_dict()
R["n_countries"] = int(ovs["nation_a3"].nunique())

# ---------------------------------------------------------------- global_weather_info
gw = pd.read_parquet(COL / "raw" / "raw_global_weather_info.parquet")
gw["alpha_3"] = gw["alpha_3"].astype("string").str.strip()
VC = ["temperature_mean", "temperature_max", "temperature_min", "precipitation_sum", "relative_humidity_mean", "soil_moisture_mean",
      "sunshine_duration", "wind_speed_mean", "wind_speed_max", "shortwave_radiation_sum"]
R["gw_rows"], R["gw_countries"] = len(gw), int(gw["alpha_3"].nunique())
R["gw_date_range"] = [str(gw["date"].min()), str(gw["date"].max())]
R["gw_dup_key"] = int(gw.duplicated(["alpha_3", "date"]).sum())
R["gw_null_full"] = {c: int(gw[c].isna().sum()) for c in VC}
per = gw.groupby("alpha_3")["date"].agg(["min", "max", "count"])
R["gw_country_end_dates"] = per["max"].astype(str).value_counts().head(10).to_dict()
R["gw_has_KOR"] = bool("KOR" in per.index)
# 국가마다 기간 안 날짜가 모두 있는지
per["span_days"] = (per["max"] - per["min"]).dt.days + 1
R["gw_countries_with_gaps"] = int((per["span_days"] != per["count"]).sum())

gwk = gw.drop_duplicates(["alpha_3", "date"]).set_index(["alpha_3", "date"])
ovs["in_gw_country"] = ovs["nation_a3"].isin(per.index)
key = pd.MultiIndex.from_arrays([ovs["nation_a3"], ovs["mnfctur_de"]])
pos = gwk.index.get_indexer(key)
ovs["gw_hit"] = pos >= 0
for c in VC:
    v = np.full(N, np.nan)
    v[ovs["gw_hit"].to_numpy()] = gwk[c].to_numpy()[pos[pos >= 0]]
    ovs["gw_" + c] = v
ovs = ovs.join(per[["min", "max"]].rename(columns={"min": "gw_cmin", "max": "gw_cmax"}), on="nation_a3")

reason = np.select(
    [ovs["nation_a2"].isna(), ovs["nation_a3"].isna(), ~ovs["in_gw_country"], ovs["mnfctur_de"].isna(),
     ovs["mnfctur_de"] > ovs["gw_cmax"], ovs["mnfctur_de"] < ovs["gw_cmin"], ~ovs["gw_hit"]],
    ["1 국가 코드 3종 모두 없음", "2 alpha-2→alpha-3 변환표에 없음", "3 global_weather_info 에 그 나라 없음", "4 기준일 없음",
     "5 기준일이 그 나라 자료 마지막 날 이후", "6 기준일이 그 나라 자료 첫날 이전", "7 기간 안인데 그날 행 없음"],
    default="0 연결")
ovs["gw_reason"] = reason
R["gw_reason"] = ovs["gw_reason"].value_counts().sort_index().to_dict()
R["gw_reason_pos"] = ovs.loc[ovs["judge_result"] == POS, "gw_reason"].value_counts().sort_index().to_dict()
R["gw_reason_detail_countries"] = {r: ovs.loc[ovs["gw_reason"] == r, "nation_a2"].value_counts(dropna=False).head(8).to_dict()
                                   for r in sorted(set(reason)) if r[0] in "2356"}
late = ovs[ovs["gw_reason"].str.startswith("5")]
R["gw_late_dates"] = [str(late["mnfctur_de"].min()), str(late["mnfctur_de"].max())] if len(late) else None
R["gw_late_by_basis"] = late["date_basis"].value_counts().to_dict()
R["gw_col_filled"] = {c: int(ovs["gw_" + c].notna().sum()) for c in VC}
R["gw_col_filled_pos"] = {c: int(ovs.loc[ovs["judge_result"] == POS, "gw_" + c].notna().sum()) for c in VC}
R["gw_col_zero"] = {c: int((ovs["gw_" + c] == 0).sum()) for c in ["precipitation_sum", "sunshine_duration", "shortwave_radiation_sum"]}
z = ovs[ovs["gw_sunshine_duration"] == 0]
R["gw_sunshine_zero_detail"] = {"countries": z["nation_a3"].value_counts().head(8).to_dict(), "months": z["mnfctur_de"].dt.month.value_counts().sort_index().to_dict(),
                                "radiation_when_zero_median": float(z["gw_shortwave_radiation_sum"].median()) if len(z) else None}
R["gw_col_null_among_hit"] ={c: int((ovs["gw_hit"] & ovs["gw_" + c].isna()).sum()) for c in VC}

# 국가 대표 지점(global_weather_info.lat/lon)과 시료 좌표(시료 파일 lat/lon) 사이 거리
pt = gw.drop_duplicates("alpha_3").set_index("alpha_3")[["lat", "lon", "country_lat", "country_lon"]]
R["gw_point_const_per_country"] = bool((gw.groupby("alpha_3")[["lat", "lon"]].nunique().max() == 1).all())
R["gw_points_top"] = pt.loc[[c for c in ["USA", "ITA", "FRA", "CHN", "ESP", "AUS"] if c in pt.index]].round(2).to_dict("index")
dc = hav(pt["lat"], pt["lon"], pt["country_lat"], pt["country_lon"])
R["gw_point_vs_country_center_km"] = {"median": round(float(np.median(dc)), 1), "max": round(float(dc.max()), 1)}
hit = ovs[ovs["nation_a3"].isin(pt.index) & ovs["lat"].notna()]
p = pt.loc[hit["nation_a3"]]
d = hav(hit["lat"], hit["lon"], p["lat"], p["lon"])
R["dist_km"] = {"n": int(len(d)), "median": round(float(np.median(d)), 1), "p90": round(float(np.percentile(d, 90)), 1),
                "max": round(float(d.max()), 1), "over100": int((d > 100).sum()), "over500": int((d > 500).sum()), "over1000": int((d > 1000).sum())}
hd = hit.assign(d=d)
R["dist_by_country"] = hd.groupby("nation_a3")["d"].agg(["count", "median", "max"]).sort_values("count", ascending=False).head(8).round(0).to_dict("index")

# ---------------------------------------------------------------- global_soil_ph
gs = pd.read_parquet(COL / "raw" / "raw_global_soil_ph.parquet")
gs["iso3"] = gs["iso3"].astype("string").str.strip()
R["gs_rows"], R["gs_iso3_unique"] = len(gs), int(gs["iso3"].nunique())
R["gs_null"] = {c: int(gs[c].isna().sum()) for c in ["ph_mean", "ph_std", "n_pixels", "nation_code"]}
R["gs_has_KOR"] = bool((gs["iso3"] == "KOR").any())
R["gs_ph_range"] = [round(float(gs["ph_mean"].min()), 2), round(float(gs["ph_mean"].max()), 2)]
gsi = gs.drop_duplicates("iso3").set_index("iso3")
ovs["gs_ph_mean"] = ovs["nation_a3"].map(gsi["ph_mean"])
ovs["gs_in"] = ovs["nation_a3"].isin(gsi.index)
R["gs_reason"] = pd.Series(np.select([ovs["nation_a2"].isna(), ovs["nation_a3"].isna(), ~ovs["gs_in"], ovs["gs_ph_mean"].isna()],
                                     ["1 국가 코드 3종 모두 없음", "2 alpha-2→alpha-3 변환표에 없음", "3 global_soil_ph 에 그 나라 없음", "4 나라는 있으나 ph_mean 빈 값"],
                                     default="0 연결(ph_mean 있음)")).value_counts().sort_index().to_dict()
R["gs_missing_countries"] = ovs.loc[ovs["nation_a3"].notna() & ~ovs["gs_in"], "nation_a3"].value_counts().to_dict()
R["gs_ph_null_countries_in_overseas"] = ovs.loc[ovs["gs_in"] & ovs["gs_ph_mean"].isna(), "nation_a3"].value_counts().to_dict()
R["gs_ph_null_countries_all"] = gs.loc[gs["ph_mean"].isna(), ["iso3", "name_ko", "n_pixels"]].astype(str).values.tolist()
R["gs_filled_pos"] =int(ovs.loc[ovs["judge_result"] == POS, "gs_ph_mean"].notna().sum())
R["gw_countries_not_in_gs"] = sorted(set(per.index) - set(gsi.index))[:40]

# ---------------------------------------------------------------- (참고) 제조소 좌표 기상 추출본
fw = pd.read_parquet(COL / "raw" / "raw_global_factory_weather_listeria_sameday.parquet")
R["factory_weather_sameday"] = {"rows": len(fw), "keys_unique": bool(fw["lims_data_key"].is_unique),
                                "in_overseas": int(fw["lims_data_key"].isin(ovs["lims_data_key"]).sum()),
                                "temperature_mean_filled": int(fw["temperature_mean"].notna().sum()),
                                "obsr_eq_base": int((fw["obsr_de"] == fw["base_date"]).sum())}

# ---------------------------------------------------------------- 미세먼지 표 (로컬 추출본 없음)
cols = pd.read_csv(COL / "nb2_db_explore_columns.csv")
tabs = pd.read_csv(COL / "nb2_db_explore_tables.csv")
FD = ["global_factory_fine_dust", "global_fine_dust_info"]
R["fine_dust_tables"] = tabs[tabs["table_name"].isin(FD)].to_dict("records")
R["fine_dust_columns"] = {t: cols[cols["table_name"] == t].iloc[:, 3:].astype(str).values.tolist() for t in FD}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
