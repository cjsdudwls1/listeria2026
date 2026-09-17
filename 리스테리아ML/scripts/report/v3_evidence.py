# -*- coding: utf-8 -*-
"""보고서 v3 근거 계산 (원자료 측정만, 값 변경 없음)"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import numpy as np, pandas as pd
from db_utils import connect
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = C + r"\raw"
S0, S1 = pd.Timestamp('2015-04-07'), pd.Timestamp('2026-06-30')
OUT = {}

# ---------- 1) ASOS 강수 NULL 의미 근거 ----------
asos = pd.read_parquet(R + r"\raw_kma_asos_daily_info.parquet")
print('ASOS columns:', list(asos.columns))
asos['d'] = pd.to_datetime(asos['meas_date'])
a = asos[(asos.d >= S0) & (asos.d <= S1)].copy()
obs = a['avg_ta'].notna()
a['m'] = a.d.dt.month
mon = a[obs].groupby('m').apply(lambda g: pd.Series({'rn_null_pct': round(g['sum_rn'].isna().mean() * 100, 1), 'rn_zero_pct': round((g['sum_rn'] == 0).mean() * 100, 1), 'rn_pos_pct': round((g['sum_rn'] > 0).mean() * 100, 1)}))
print('\nASOS 월별(기온 관측일 기준) 강수 NULL/0/양수 비율:'); print(mon.T.to_string())
OUT['asos_rn_monthly'] = mon.to_dict()
# 다른 강수 관련 컬럼과의 동시 결측
for c in ['hr_1_max_rn', 'mi_10_max_rn', 'n99_rn', 'sum_rn_dur']:
    if c in a.columns:
        sub = a[obs]
        OUT[f'rn_null_and_{c}'] = {'sum_rn_null_rows': int(sub['sum_rn'].isna().sum()),
                                   f'{c}_notnull_when_rn_null': int((sub['sum_rn'].isna() & sub[c].notna() & (sub[c].astype(str).str.strip() != '')).sum()),
                                   f'{c}_notnull_when_rn_notnull': round(float((sub['sum_rn'].notna() & sub[c].notna()).sum() / max(sub['sum_rn'].notna().sum(), 1) * 100), 1)}
print('\n강수 관련 컬럼 동시 결측:', {k: v for k, v in OUT.items() if k.startswith('rn_null_and')})
# 0.0 값의 시간 강수 컬럼
z = a[obs & (a['sum_rn'] == 0)]
print('sum_rn==0 인 날의 hr_1_max_rn 분포:', z['hr_1_max_rn'].describe().to_dict() if 'hr_1_max_rn' in a.columns else None)
if 'sum_rn_dur' in a.columns:
    print('sum_rn==0 인 날의 강수 계속시간(sum_rn_dur) 예:', z['sum_rn_dur'].astype(str).value_counts().head(5).to_dict())
    print('sum_rn NULL 인 날의 강수 계속시간 예:', a[obs & a['sum_rn'].isna()]['sum_rn_dur'].astype(str).value_counts().head(5).to_dict())
# 일기현상 텍스트 컬럼이 있다면
for c in a.columns:
    if a[c].dtype == object and c not in ('stn_id', 'sum_sshr', 'ss_dur', 'sum_rn_dur', 'max_ta_hrmt', 'min_ta_hrmt') and a[c].astype(str).str.contains('비|강수|rain', regex=True).any():
        print('텍스트 컬럼 후보:', c, a[c].astype(str).value_counts().head(3).to_dict())

# 같은 위치(5km 이내) NAS 관측소와 비교
ast = pd.read_parquet(R + r"\raw_kma_ground_obs_station_info.parquet")
ast['chg_dt'] = pd.to_datetime(ast['chg_dt'], errors='coerce')
ast1 = ast.sort_values('chg_dt', ascending=False).drop_duplicates('stn_id')[['stn_id', 'lat', 'lon']]
nst = pd.read_parquet(R + r"\raw_nas_agri_weather_obs_station.parquet")[['obsr_spot_code', 'instl_la', 'instl_lo']]
def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
pairs = []
for _, s in ast1.iterrows():
    d = hav(float(s.lat), float(s.lon), nst.instl_la.astype(float).values, nst.instl_lo.astype(float).values)
    j = d.argmin()
    if d[j] <= 5: pairs.append((s.stn_id, nst.obsr_spot_code.iloc[j], round(float(d[j]), 2)))
print('\nASOS-NAS 5km 이내 쌍:', len(pairs))
nas = pd.read_parquet(R + r"\raw_nas_agri_weather_daily_info.parquet", columns=['obsr_spot_code', 'meas_date', 'rn', 'temp'])
nas['d'] = pd.to_datetime(nas['meas_date'])
pp = pd.DataFrame(pairs, columns=['stn_id', 'obsr_spot_code', 'km'])
mm = a[obs][['stn_id', 'd', 'sum_rn']].merge(pp, on='stn_id').merge(nas[['obsr_spot_code', 'd', 'rn']], on=['obsr_spot_code', 'd'])
mm = mm[mm['rn'].notna()]
cmp = {
    'pairs': len(pairs), 'days': int(len(mm)),
    'asos_null__nas_rn0_pct': round(float((mm.loc[mm.sum_rn.isna(), 'rn'] == 0).mean() * 100), 1),
    'asos_null__nas_rn_ge1_pct': round(float((mm.loc[mm.sum_rn.isna(), 'rn'] >= 1).mean() * 100), 1),
    'asos_zero__nas_rn0_pct': round(float((mm.loc[mm.sum_rn == 0, 'rn'] == 0).mean() * 100), 1),
    'asos_zero__nas_rn_0to1_pct': round(float(((mm.loc[mm.sum_rn == 0, 'rn'] > 0) & (mm.loc[mm.sum_rn == 0, 'rn'] < 1)).mean() * 100), 1),
    'asos_pos__nas_rn_pos_pct': round(float((mm.loc[mm.sum_rn > 0, 'rn'] > 0).mean() * 100), 1),
    'asos_ge5__nas_rn0_pct': round(float((mm.loc[mm.sum_rn >= 5, 'rn'] == 0).mean() * 100), 1),
}
print('ASOS vs 인근 NAS 같은 날 강수 비교:', cmp); OUT['asos_nas_rain_compare'] = cmp

# ---------- 2) 우리 시료 기준: 강수 0 비율 ----------
S = json.load(open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\rep_stats.json", encoding='utf-8'))
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'judge_result', 'mnfctur_de', 'lat', 'lon'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'lat', 'lon'])
b = body.merge(db.rename(columns={'lat': 'lat_db', 'lon': 'lon_db'}), on='lims_data_key', how='left')
b['lat_use'] = b['lat_db'].fillna(b['lat'].astype(float)); b['lon_use'] = b['lon_db'].fillna(b['lon'].astype(float))
b['base_date'] = pd.to_datetime(b['mnfctur_de']).dt.normalize()
dom = b[b.src_route == 'domestic'].copy()
pts = dom[['lat_use', 'lon_use']].drop_duplicates().reset_index(drop=True)
def nearest(pts, st):
    P = np.radians(pts[['lat_use', 'lon_use']].to_numpy(float)); Sx = np.radians(st[['lat', 'lon']].to_numpy(float))
    ids = np.empty(len(P), dtype=object); dist = np.empty(len(P))
    for i in range(0, len(P), 1000):
        p = P[i:i + 1000]
        aa = np.sin((p[:, None, 0] - Sx[None, :, 0]) / 2) ** 2 + np.cos(p[:, None, 0]) * np.cos(Sx[None, :, 0]) * np.sin((p[:, None, 1] - Sx[None, :, 1]) / 2) ** 2
        dd = 2 * 6371 * np.arcsin(np.sqrt(np.clip(aa, 0, 1))); j = dd.argmin(1)
        ids[i:i + 1000] = st['sid'].to_numpy()[j]; dist[i:i + 1000] = dd[np.arange(len(p)), j]
    return ids, dist
ast2 = ast.copy(); ast2['reg_dt'] = pd.to_datetime(ast2['reg_dt'], errors='coerce')
ast2 = ast2.sort_values(['chg_dt', 'reg_dt'], ascending=False, na_position='last').drop_duplicates('stn_id')
sA = ast2[ast2.stn_id.isin(set(asos.stn_id))][['stn_id', 'lat', 'lon']].rename(columns={'stn_id': 'sid'}).astype({'lat': float, 'lon': float})
ids, dist = nearest(pts, sA)
m = pts.assign(sid=ids, dist=dist)
x = dom.merge(m, on=['lat_use', 'lon_use'], how='left')
ad = asos[['stn_id', 'd', 'sum_rn', 'avg_ta', 'sum_gsr', 'avg_cm10_te']].rename(columns={'stn_id': 'sid', 'd': 'base_date'})
x = x.merge(ad, on=['sid', 'base_date'], how='left', indicator=True)
has_row = x['_merge'] == 'both'
OUT['sample_asos'] = {
    'n': int(len(x)), 'row_exists_pct': round(float(has_row.mean() * 100), 1),
    'sum_rn_present_pct': round(float(x['sum_rn'].notna().mean() * 100), 1),
    'sum_rn_zero_pct_of_all': round(float((x['sum_rn'] == 0).mean() * 100), 1),
    'sum_rn_pos_pct_of_all': round(float((x['sum_rn'] > 0).mean() * 100), 1),
    'sum_rn_null_with_temp_pct_of_all': round(float((x['sum_rn'].isna() & x['avg_ta'].notna()).mean() * 100), 1),
    'avg_ta_missing_no_row_pct': round(float((~has_row).mean() * 100), 2),
    'avg_ta_missing_row_but_null_pct': round(float((has_row & x['avg_ta'].isna()).mean() * 100), 2),
}
x['yr'] = x['base_date'].dt.year
OUT['sample_asos_avg_ta_missing_by_year'] = x.groupby('yr')['avg_ta'].apply(lambda s: round(float(s.isna().mean() * 100), 1)).to_dict()
print('\n우리 국내 시료 × 최근접 ASOS 당일:', OUT['sample_asos'])
print('연도별 avg_ta 빈 비율:', OUT['sample_asos_avg_ta_missing_by_year'])

# ---------- 3) 결측 이유: 해당 항목이 연구기간 내내 빈 관측소가 같은 날 다른 항목은 기록했는지 ----------
def station_gap(df, sid, var, ref):
    d = df[(pd.to_datetime(df['meas_date']) >= S0) & (pd.to_datetime(df['meas_date']) <= S1)]
    has = d.groupby(sid)[var].apply(lambda s: s.notna().any())
    none_st = has[~has].index
    sub = d[d[sid].isin(none_st)]
    return {'stations_total': int(d[sid].nunique()), 'stations_without_value': int(len(none_st)),
            'rows_of_those': int(len(sub)), f'{ref}_present_pct_in_those_rows': round(float(sub[ref].notna().mean() * 100), 1) if len(sub) else None}
gaps = {}
for v in ['sum_gsr', 'avg_cm5_te', 'avg_cm10_te']:
    gaps['ASOS.' + v] = station_gap(asos, 'stn_id', v, 'avg_ta')
nasf = pd.read_parquet(R + r"\raw_nas_agri_weather_daily_info.parquet")
for v, ref in [('temp', 'rn'), ('wind', 'temp'), ('srqty', 'temp'), ('soil_wt', 'temp'), ('gr_temp', 'temp'), ('sun_time', 'temp'), ('soil_temp', 'temp'), ('rn', 'temp')]:
    gaps['NAS.' + v] = station_gap(nasf, 'obsr_spot_code', v, ref)
print('\n항목이 연구기간 내내 빈 관측소:'); [print(' ', k, vv) for k, vv in gaps.items()]
OUT['station_gaps'] = gaps
# NAS 관측소 코드 형태(7번째 글자)별 항목 보유
nasf['kind'] = nasf['obsr_spot_code'].astype(str).str[6]
st = nasf[(pd.to_datetime(nasf.meas_date) >= S0) & (pd.to_datetime(nasf.meas_date) <= S1)]
kind = st.groupby('kind').agg(stations=('obsr_spot_code', 'nunique'), **{f'{v}_pct': (v, lambda s: round(float(s.notna().mean() * 100), 1)) for v in ['temp', 'rn', 'wind', 'srqty', 'sun_time', 'soil_wt', 'soil_temp', 'gr_temp']})
print('\nNAS 지점코드 7번째 글자별 항목 비결측%:'); print(kind.to_string())
OUT['nas_kind'] = kind.reset_index().to_dict('records')

# ---------- 4) 토양 pH 예시값 ----------
conn = connect()
def q(sql):
    with conn.cursor() as cur:
        cur.execute(sql); return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
ex19 = q("SELECT pnu_cd, pnu_nm, legal_dong_code, legal_dong_nm, any_year, exam_day::text, acid, vldpha, om FROM api_data_info.soil_ph_info WHERE any_year = '2020' AND acid IS NOT NULL LIMIT 3")
ex10 = q("SELECT pnu_cd, pnu_nm, legal_dong_code, legal_dong_nm, any_year, exam_day::text, acid FROM api_data_info.soil_ph_info WHERE LENGTH(pnu_cd) = 10 AND any_year = '2024' LIMIT 2")
conn.close()
print('\n토양 pH 필지 예시:'); print(ex19.to_string(index=False)); print(ex10.to_string(index=False))
OUT['soil_examples_19'] = ex19.astype(str).to_dict('records'); OUT['soil_examples_10'] = ex10.astype(str).to_dict('records')
print('우리 시료 legal_dong_code 예:', dom['legal_dong_code'].astype(str).value_counts().head(3).to_dict() if 'legal_dong_code' in dom.columns else '')
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\v3_evidence.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
