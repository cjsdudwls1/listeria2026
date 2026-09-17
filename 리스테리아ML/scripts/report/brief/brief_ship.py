# -*- coding: utf-8 -*-
"""기준일 출처 '선적일' 행: 주소 출처·좌표가 진짜 해외인지 + 그 좌표의 해외 기상을 DB에서 받을 수 있는지
- 좌표의 나라: 공개 국경 지도(Natural Earth 1:50m, GeoJSON)를 내려받아 로컬에서 점-다각형 판정(좌표를 외부로 보내지 않음)
- 비교 대상: 수입 LIMS 제조국가코드(mnf_nation_cd), 해외제조업소 대장 국가코드(ntncd), 모델팀 지오코딩 기록(staging_data.s_co_ifd_ovsmnfst_regstr_m_geo)
- DB는 SELECT만"""
import sys, json, os, re, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import numpy as np, pandas as pd
from db_utils import connect
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"; C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = C + r"\raw"
OUT = {}
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet",
                       columns=['lims_data_key', 'judge_result', 'src_route', 'date_basis', 'addr', 'addr_source', 'lat', 'lon', 'mnfctur_de', 'mnfctur_cmpny_nm', 'prduct_nm'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'lat', 'lon', 'cntc_sys_key2', 'prdctn_nation_code', 'orgplce_nation_code', 'import_nation_code', 'addr_source', 'geo_route'])
b = body.merge(db.rename(columns={'lat': 'lat_db', 'lon': 'lon_db', 'addr_source': 'addr_source_db'}), on='lims_data_key', how='left')
sh = b[b['date_basis'].astype(str) == '선적일'].copy()
sh['det'] = sh['judge_result'].astype(str) == '검출'
sh['lat'] = pd.to_numeric(sh['lat'], errors='coerce'); sh['lon'] = pd.to_numeric(sh['lon'], errors='coerce')
OUT['rows'] = int(len(sh)); OUT['det'] = int(sh['det'].sum())
OUT['src_route'] = sh['src_route'].astype(str).value_counts().to_dict()
OUT['addr_source'] = sh['addr_source'].astype(str).value_counts().to_dict()
OUT['geo_route_db'] = sh['geo_route'].astype(str).value_counts().to_dict()
OUT['lat_null'] = int(sh['lat'].isna().sum())
OUT['parquet_eq_db_coord_pct'] = round(float(((sh['lat'] - sh['lat_db']).abs().lt(1e-9) & (sh['lon'] - sh['lon_db']).abs().lt(1e-9)).mean() * 100), 2)
OUT['addr_has_hangul'] = int(sh['addr'].astype('string').str.contains('[가-힣]', regex=True).fillna(False).sum())
OUT['addr_has_korea_word'] = int(sh['addr'].astype('string').str.upper().str.contains('KOREA|대한민국', regex=True).fillna(False).sum())
OUT['coords'] = int(sh[['lat', 'lon']].dropna().drop_duplicates().shape[0])
all_ship = b['date_basis'].astype(str) == '선적일'
OUT['addr_source_overseas_all'] = b.loc[b.src_route == 'overseas', 'addr_source'].astype(str).value_counts().to_dict()

# 수입 LIMS 제조국가·업소코드 (리스테리아 행 우선 1행/접수번호)
s = pd.read_parquet(C + r"\02e_s_org.parquet", columns=['rcept_no', 'label', 'lims_seq', 'mnf_nation_cd', 'ovsmnfst_cd', 'shipng_de'])
s = s.assign(_lis=(s['label'] == '리스테리아')).sort_values(['rcept_no', '_lis', 'lims_seq'], ascending=[True, False, True]).drop_duplicates('rcept_no')
sh = sh.merge(s[['rcept_no', 'mnf_nation_cd', 'ovsmnfst_cd', 'shipng_de']], left_on='cntc_sys_key2', right_on='rcept_no', how='left')
OUT['mnf_nation_cd_present'] = int(sh['mnf_nation_cd'].notna().sum())
OUT['mnf_nation_KR'] = int((sh['mnf_nation_cd'].astype(str) == 'KR').sum())
OUT['mnf_nation_top'] = sh['mnf_nation_cd'].astype(str).value_counts().head(10).to_dict()
OUT['base_eq_shipng_pct'] = round(float((pd.to_datetime(sh['mnfctur_de']).dt.normalize() == pd.to_datetime(sh['shipng_de']).dt.normalize()).mean() * 100), 2)

# 모델팀 지오코딩 기록(해외제조업소 코드 기준)
conn = connect()
codes = sh['ovsmnfst_cd'].dropna().astype(str).unique().tolist()
with conn.cursor() as cur:
    cur.execute("""SELECT ovsmnfst_cd, ntncd, ovsmnfst_addr, la, lo, lat, lon, match_type, match_country, geocode_source, geocode_status
                   FROM staging_data.s_co_ifd_ovsmnfst_regstr_m_geo WHERE ovsmnfst_cd = ANY(%s)""", (codes,))
    geo = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
conn.close()
geo.to_parquet(T + r"\brief_ship_geo.parquet", index=False)
print('지오코딩 기록 행', len(geo), '업소코드', geo['ovsmnfst_cd'].nunique(), '/', len(codes))
g1 = geo.drop_duplicates('ovsmnfst_cd')
for c in ['lat', 'lon', 'la', 'lo']: g1[c] = pd.to_numeric(g1[c], errors='coerce')
sh = sh.merge(g1.rename(columns={'lat': 'geo_lat', 'lon': 'geo_lon', 'la': 'reg_la', 'lo': 'reg_lo'}), on='ovsmnfst_cd', how='left')
OUT['geo_record_rows'] = int(sh['ntncd'].notna().sum())
OUT['row_coord_eq_geo_record_pct_of_linked'] = round(float(((sh['lat'] - sh['geo_lat']).abs().lt(1e-6) & (sh['lon'] - sh['geo_lon']).abs().lt(1e-6))[sh['ntncd'].notna()].mean() * 100), 2)
OUT['match_type_rows'] = sh['match_type'].astype(str).value_counts().to_dict()
OUT['match_country_rows'] = sh['match_country'].astype(str).value_counts().to_dict()
OUT['geocode_source_rows'] = sh['geocode_source'].astype(str).value_counts().to_dict()
OUT['geocode_status_rows'] = sh['geocode_status'].astype(str).value_counts().to_dict()
OUT['ntncd_eq_mnf_nation_pct'] = round(float((sh['ntncd'].astype(str) == sh['mnf_nation_cd'].astype(str))[sh['ntncd'].notna()].mean() * 100), 2)

# 국경 지도로 좌표의 나라 판정
GJ = T + r"\ne_50m_admin_0_countries.geojson"
if not os.path.exists(GJ):
    url = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson'
    urllib.request.urlretrieve(url, GJ)
gj = json.load(open(GJ, encoding='utf-8'))
polys = []
for f in gj['features']:
    p = f['properties']; a2 = p.get('ISO_A2_EH') if p.get('ISO_A2_EH') not in (None, '-99') else p.get('ISO_A2')
    geom = f['geometry']; parts = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    for part in parts:
        rings = [np.asarray(r, dtype=float) for r in part]
        ext = rings[0]
        polys.append({'a2': a2, 'name': p.get('NAME_KO') or p.get('NAME'), 'rings': rings, 'bbox': (ext[:, 0].min(), ext[:, 0].max(), ext[:, 1].min(), ext[:, 1].max())})
pts = sh[['lat', 'lon']].dropna().drop_duplicates().reset_index(drop=True)
px, py_ = pts['lon'].to_numpy(), pts['lat'].to_numpy()
cc = np.array([None] * len(pts), dtype=object); cn = np.array([None] * len(pts), dtype=object)
def inside(x, y, ring):
    xi, yi = ring[:, 0], ring[:, 1]; xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    cond = ((yi[None, :] > y[:, None]) != (yj[None, :] > y[:, None])) & (x[:, None] < (xj - xi)[None, :] * (y[:, None] - yi[None, :]) / ((yj - yi)[None, :] + 1e-300) + xi[None, :])
    return cond.sum(1) % 2 == 1
for pg in polys:
    x0, x1, y0, y1 = pg['bbox']
    cand = np.where((px >= x0) & (px <= x1) & (py_ >= y0) & (py_ <= y1) & (cc == None))[0]
    if len(cand) == 0: continue
    ins = np.zeros(len(cand), dtype=bool)
    for ring in pg['rings']:
        ins ^= inside(px[cand], py_[cand], ring)
    cc[cand[ins]] = pg['a2']; cn[cand[ins]] = pg['name']
# 어느 나라 안에도 안 들어간 점(해안선 단순화 등): 가장 가까운 국경선 꼭짓점의 나라와 거리
allv = np.vstack([np.c_[r, np.full(len(r), i)] for i, pg in enumerate(polys) for r in pg['rings'][:1]])
out_idx = np.where(cc == None)[0]
near_km = np.full(len(pts), np.nan)
for i in out_idx:
    la1, lo1 = np.radians(py_[i]), np.radians(px[i]); la2, lo2 = np.radians(allv[:, 1]), np.radians(allv[:, 0])
    d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    j = d.argmin(); near_km[i] = d[j]; pg = polys[int(allv[j, 2])]
    cc[i] = pg['a2']; cn[i] = pg['name']
pts['coord_country'] = cc; pts['coord_country_nm'] = cn; pts['outside_polygon_km'] = near_km
sh = sh.merge(pts, on=['lat', 'lon'], how='left')
OUT['coord_outside_polygon_points'] = {'points': int(len(out_idx)), 'rows': int(sh['outside_polygon_km'].notna().sum()),
                                       'km_p50': float(np.nanmedian(near_km)) if len(out_idx) else None, 'km_max': float(np.nanmax(near_km)) if len(out_idx) else None}
OUT['coord_country_KR_rows'] = int((sh['coord_country'] == 'KR').sum()); OUT['coord_country_KR_points'] = int((pts['coord_country'] == 'KR').sum())
OUT['coord_country_KP_rows'] = int((sh['coord_country'] == 'KP').sum())
OUT['coord_country_top'] = sh['coord_country_nm'].astype(str).value_counts().head(12).to_dict()
sh['mnf_eq_coord'] = sh['mnf_nation_cd'].astype(str) == sh['coord_country'].astype(str)
lk = sh['mnf_nation_cd'].notna()
OUT['coord_country_eq_mnf_nation'] = {'rows_pct': round(float(sh.loc[lk, 'mnf_eq_coord'].mean() * 100), 2), 'rows_mismatch': int((lk & ~sh['mnf_eq_coord']).sum()),
                                      'det_rows_mismatch': int((lk & ~sh['mnf_eq_coord'] & sh['det']).sum()),
                                      'points_mismatch': int(sh.loc[lk & ~sh['mnf_eq_coord'], ['lat', 'lon']].drop_duplicates().shape[0])}
mm = sh[lk & ~sh['mnf_eq_coord']]
OUT['mismatch_pairs_top'] = (mm['mnf_nation_cd'].astype(str) + '→좌표 ' + mm['coord_country'].astype(str)).value_counts().head(12).to_dict()
ex = mm.groupby(['lat', 'lon']).agg(rows=('lims_data_key', 'size'), mnf=('mnf_nation_cd', 'first'), coord=('coord_country_nm', 'first'), ntncd=('ntncd', 'first'),
                                     match_type=('match_type', 'first'), match_country=('match_country', 'first'), addr=('addr', 'first'), cmp=('mnfctur_cmpny_nm', 'first')).sort_values('rows', ascending=False).head(8)
OUT['mismatch_examples'] = [{k: (str(v)[:90] if isinstance(v, str) else v) for k, v in r.items()} | {'lat': float(i[0]), 'lon': float(i[1])} for i, r in ex.iterrows()]
# 좌표가 대장(원래 la/lo)과 얼마나 떨어져 있나
okr = sh['reg_la'].notna() & sh['reg_lo'].notna() & (sh['reg_la'] != 0)
if okr.any():
    la1, lo1, la2, lo2 = map(np.radians, (sh.loc[okr, 'lat'], sh.loc[okr, 'lon'], sh.loc[okr, 'reg_la'], sh.loc[okr, 'reg_lo']))
    dkm = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    OUT['dist_row_to_registry_la_lo_km'] = {'rows_with_registry_coord': int(okr.sum()), 'p50': float(np.median(dkm)), 'p90': float(np.quantile(dkm, .9)), 'gt50km_rows': int((dkm > 50).sum())}
else:
    OUT['dist_row_to_registry_la_lo_km'] = {'rows_with_registry_coord': 0}

# 해외 기상 DB 연결: 제조소 좌표 기상(당일) + 국가 단위(당일)
gf = pd.read_parquet(R + r"\raw_global_factory_weather_listeria_sameday.parquet", columns=['lims_data_key', 'weather_seq', 'temperature_mean', 'soil_moisture_mean'])
gf['lims_data_key'] = gf['lims_data_key'].astype(str)
sh = sh.merge(gf, on='lims_data_key', how='left')
OUT['gfw_sameday_rows_pct'] = round(float(sh['weather_seq'].notna().mean() * 100), 2)
rng = pd.read_parquet(T + r"\brief_gfw_coord_range.parquet")
sh = sh.merge(rng, on=['lat', 'lon'], how='left')
OUT['gfw_period_rows'] = {'dmin': sh['dmin'].astype(str).value_counts().to_dict(), 'dmax': sh['dmax'].astype(str).value_counts().to_dict()}
bd = pd.to_datetime(sh['mnfctur_de']).dt.normalize()
OUT['ship_date_range'] = {'min': str(bd.min().date()), 'max': str(bd.max().date())}
OUT['ship_date_before_2015_03_02_rows'] = int((bd < pd.Timestamp('2015-03-02')).sum())  # 60일 전이 2015-01-01 이전인 선적일
json.dump(OUT, open(T + r"\brief_ship.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
sh[['lims_data_key', 'lat', 'lon', 'coord_country', 'coord_country_nm', 'mnf_nation_cd', 'ntncd', 'match_type', 'match_country', 'outside_polygon_km']].to_parquet(T + r"\brief_ship_rows.parquet", index=False)
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
