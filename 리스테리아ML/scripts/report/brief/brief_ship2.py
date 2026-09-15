# -*- coding: utf-8 -*-
"""선적일 행 후속 확인: 좌표 출처·정밀도(ready_data), 한국 좌표 행 내역, 국경 밖 먼 좌표, 제조국 KR 행, 기상 연결(DB 좌표로)"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import numpy as np, pandas as pd
from db_utils import connect
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"; C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = C + r"\raw"
OUT = {}
conn = connect()
def q(sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
rd = q("""SELECT lims_data_key, geo_route, geo_source, geocode_status, geo_match_level, addr, addr_source, ovsmnfst_nm, mnfctur_cmpny_nm, mnfctur_cmpny_locplc_nm, lat, lon
          FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아' AND date_basis = '선적일'""")
rd['lims_data_key'] = rd['lims_data_key'].astype(str)
rows = pd.read_parquet(T + r"\brief_ship_rows.parquet")
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'judge_result', 'src_route', 'date_basis', 'prduct_nm', 'lat', 'lon'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
sh = body[body['date_basis'].astype(str) == '선적일'].merge(rd.drop(columns=['lat', 'lon']).rename(columns={}), on='lims_data_key', how='left') \
    .merge(pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'lat', 'lon']).rename(columns={'lat': 'lat_db', 'lon': 'lon_db'}), on='lims_data_key', how='left')
rows = rows.drop_duplicates('lims_data_key')
sh = sh.merge(rows[['lims_data_key', 'coord_country', 'coord_country_nm', 'mnf_nation_cd', 'ntncd', 'match_type', 'match_country', 'outside_polygon_km']], on='lims_data_key', how='left')
sh['det'] = sh['judge_result'].astype(str) == '검출'
OUT['rows_in_db_listing'] = int(rd.shape[0]); OUT['body_rows'] = int(len(sh)); OUT['linked'] = int(sh['geo_source'].notna().sum())
la, lo = pd.to_numeric(sh['lat'], errors='coerce'), pd.to_numeric(sh['lon'], errors='coerce')
OUT['parquet_vs_db_coord_maxdiff_deg'] = float(np.nanmax(np.abs(np.r_[la - sh['lat_db'], lo - sh['lon_db']])))
OUT['parquet_vs_db_within_1e-4_pct'] = round(float(((la - sh['lat_db']).abs().lt(1e-4) & (lo - sh['lon_db']).abs().lt(1e-4)).mean() * 100), 2)
pts = sh.drop_duplicates(['lat_db', 'lon_db'])
OUT['geo_source_rows'] = sh['geo_source'].astype(str).value_counts().to_dict(); OUT['geo_source_points'] = pts['geo_source'].astype(str).value_counts().to_dict()
OUT['geocode_status_rows'] = sh['geocode_status'].astype(str).value_counts().to_dict(); OUT['geocode_status_points'] = pts['geocode_status'].astype(str).value_counts().to_dict()
OUT['geo_match_level_rows'] = sh['geo_match_level'].astype(str).value_counts().to_dict()
OUT['det_geocode_status'] = sh.loc[sh.det, 'geocode_status'].astype(str).value_counts().to_dict()
OUT['det_geo_source'] = sh.loc[sh.det, 'geo_source'].astype(str).value_counts().to_dict()
# 같은 좌표에 여러 나라/업체가 몰린 대표점(나라 중심점 등) 의심
grp = sh.groupby(['lat_db', 'lon_db']).agg(rows=('lims_data_key', 'size'), companies=('mnfctur_cmpny_nm', 'nunique'), src=('geo_source', 'first'), st=('geocode_status', 'first'), ctry=('coord_country_nm', 'first'))
OUT['points_with_10plus_companies'] = grp[grp.companies >= 10].sort_values('companies', ascending=False).head(8).reset_index().astype(str).to_dict('records')
# 한국 땅 좌표 행
kr = sh[sh['coord_country'] == 'KR']
OUT['KR_coord_rows'] = kr[['lims_data_key', 'src_route', 'judge_result', 'mnf_nation_cd', 'ntncd', 'geo_source', 'geocode_status', 'addr', 'mnfctur_cmpny_nm', 'prduct_nm', 'lat_db', 'lon_db']].astype(str).to_dict('records')
OUT['KR_coord_src_route'] = kr['src_route'].astype(str).value_counts().to_dict()
# 경로가 국내인 선적일 행
OUT['domestic_route_ship_rows'] = sh[sh.src_route == 'domestic'][['lims_data_key', 'geo_route', 'coord_country', 'mnf_nation_cd', 'addr', 'mnfctur_cmpny_nm', 'lat_db', 'lon_db']].astype(str).to_dict('records')
# 제조국가코드 KR 행
krm = sh[sh['mnf_nation_cd'].astype(str) == 'KR']
OUT['mnf_nation_KR'] = {'rows': int(len(krm)), 'coord_country': krm['coord_country_nm'].astype(str).value_counts().to_dict(), 'addr_top': krm['addr'].astype(str).str.slice(0, 70).value_counts().head(5).to_dict(),
                        'prduct_top': krm['prduct_nm'].astype(str).value_counts().head(5).to_dict()}
# 국경선 밖(해안 등)으로 20km 넘게 떨어진 좌표
far = sh[sh['outside_polygon_km'] > 20].groupby(['lat_db', 'lon_db']).agg(rows=('lims_data_key', 'size'), km=('outside_polygon_km', 'first'), ctry=('coord_country_nm', 'first'),
                                                                        mnf=('mnf_nation_cd', 'first'), addr=('addr', 'first'), st=('geocode_status', 'first')).sort_values('km', ascending=False)
OUT['outside_polygon_gt20km'] = {'points': int(len(far)), 'rows': int(far['rows'].sum()), 'examples': far.head(6).reset_index().astype(str).to_dict('records')}
# 해외 기상 연결(DB 좌표 기준)
rng = pd.read_parquet(T + r"\brief_gfw_coord_range.parquet").rename(columns={'lat': 'lat_db', 'lon': 'lon_db'})
x = sh.merge(rng, on=['lat_db', 'lon_db'], how='left')
OUT['gfw_period_rows'] = {'has_2015_01_01_to_2026_08_08': int(((x['dmin'].astype(str) == '2015-01-01') & (x['dmax'].astype(str) == '2026-08-08')).sum()), 'no_coord_in_range_file': int(x['dmin'].isna().sum())}
d9 = sh[sh.src_route == 'domestic'][['lat_db', 'lon_db']].drop_duplicates()
if len(d9):
    r9 = q("""SELECT c.lat, c.lon, (SELECT count(*) FROM api_data_info.global_factory_weather g WHERE g.lat_g = c.lat AND g.lon_g = c.lon AND g.obsr_de BETWEEN '2015-01-01' AND '2015-01-31') AS n_jan2015
              FROM unnest(%s::float8[], %s::float8[]) AS c(lat, lon)""", (d9['lat_db'].tolist(), d9['lon_db'].tolist()))
    OUT['domestic_route_coords_in_gfw'] = r9.astype(str).to_dict('records')
# 좌표별 기상 날짜 수 표본(30곳): 2015-01-01~2026-08-08 = 4,238일
samp = sh[['lat_db', 'lon_db']].drop_duplicates().sample(30, random_state=1)
cnt = q("""SELECT c.lat, c.lon, (SELECT count(*) FROM api_data_info.global_factory_weather g WHERE g.lat_g = c.lat AND g.lon_g = c.lon) AS n_days
           FROM unnest(%s::float8[], %s::float8[]) AS c(lat, lon)""", (samp['lat_db'].tolist(), samp['lon_db'].tolist()))
OUT['gfw_days_sample30'] = {'min': int(cnt['n_days'].min()), 'max': int(cnt['n_days'].max()), 'expected': 4238}
# 해외제조업소 코드 형식 비교(대장 연결이 낮은 이유)
s02 = pd.read_parquet(C + r"\02e_s_org.parquet", columns=['ovsmnfst_cd'])
OUT['code_format'] = {'s_org_examples': s02['ovsmnfst_cd'].dropna().astype(str).drop_duplicates().head(5).tolist(),
                      's_org_len': s02['ovsmnfst_cd'].dropna().astype(str).str.len().value_counts().head(3).to_dict(),
                      'registry_examples': q("SELECT ovsmnfst_cd FROM ba_data.s_co_ifd_ovsmnfst_regstr_m LIMIT 5")['ovsmnfst_cd'].astype(str).tolist(),
                      'registry_len': q("SELECT length(ovsmnfst_cd) AS l, count(*) AS n FROM ba_data.s_co_ifd_ovsmnfst_regstr_m GROUP BY 1 ORDER BY 2 DESC LIMIT 3").astype(str).to_dict('records')}
conn.close()
json.dump(OUT, open(T + r"\brief_ship2.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
