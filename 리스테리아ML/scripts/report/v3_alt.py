# -*- coding: utf-8 -*-
"""(참고) 관측소 선택 방식별 국내 시료 당일 값 있음% — 전 변수 (측정만, 값 변경 없음)
현재: DB에 일자료가 한 건이라도 있는 관측소 중 가장 가까운 곳(날짜 무관)
참고 A: 기준일 당일 자료 행이 있는 관측소 중 가장 가까운 곳
+ '행 없음' 원인(개설 전/폐쇄 후/운영 중 빠진 날), 시험용 관측소 확인"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = C + r"\raw"
OUT = {}
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'mnfctur_de', 'lat', 'lon', 'judge_result'] if False else ['lims_data_key', 'src_route', 'mnfctur_de', 'lat', 'lon'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'lat', 'lon'])
b = body.merge(db.rename(columns={'lat': 'lat_db', 'lon': 'lon_db'}), on='lims_data_key', how='left')
b['lat_use'] = b['lat_db'].fillna(b['lat'].astype(float)); b['lon_use'] = b['lon_db'].fillna(b['lon'].astype(float))
b['base_date'] = pd.to_datetime(b['mnfctur_de']).dt.normalize()
dom = b[b.src_route == 'domestic'].copy()

def hav_mat(P, Sx):
    P = np.radians(P); Sx = np.radians(Sx)
    aa = np.sin((P[:, None, 0] - Sx[None, :, 0]) / 2) ** 2 + np.cos(P[:, None, 0]) * np.cos(Sx[None, :, 0]) * np.sin((P[:, None, 1] - Sx[None, :, 1]) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))

def blank_to_na(df, cols):
    for c in cols:
        if df[c].dtype == object or str(df[c].dtype).startswith('string'):
            df[c] = df[c].where(df[c].astype(str).str.strip() != '')
    return df

def run(name, st, daily, did, dcol, vcols):
    dd = daily[[did, dcol] + vcols].rename(columns={did: 'sid', dcol: 'base_date'}).copy()
    dd['base_date'] = pd.to_datetime(dd['base_date']).dt.normalize()
    dd = blank_to_na(dd.drop_duplicates(['sid', 'base_date']), vcols)
    s = st[st['sid'].isin(set(dd['sid']))].reset_index(drop=True)
    pts = dom[['lat_use', 'lon_use']].drop_duplicates().reset_index(drop=True)
    idx = np.empty(len(pts), dtype=int)
    for i in range(0, len(pts), 1000):
        idx[i:i + 1000] = hav_mat(pts.iloc[i:i + 1000].to_numpy(float), s[['lat', 'lon']].to_numpy(float)).argmin(1)
    pts['sid'] = s['sid'].to_numpy()[idx]
    x = dom.merge(pts, on=['lat_use', 'lon_use'], how='left').merge(dd.assign(_row=1), on=['sid', 'base_date'], how='left')
    span = dd.groupby('sid')['base_date'].agg(db_first='min', db_last='max').reset_index()
    x = x.merge(span, on='sid', how='left')
    nr = x['_row'].isna().to_numpy()
    why = np.where(~nr, 'row', np.where(x['base_date'] < x['db_first'], 'before', np.where(x['base_date'] > x['db_last'], 'after', 'gap')))
    q = x.loc[nr, ['lat_use', 'lon_use', 'base_date']].drop_duplicates().reset_index(drop=True)
    day_st = dd.groupby('base_date')['sid'].apply(set)
    alt = []
    for d, g in q.groupby('base_date'):
        cand = s[s['sid'].isin(day_st.get(d, set()))]
        if len(cand) == 0:
            alt.append(g.assign(sid2=None)); continue
        alt.append(g.assign(sid2=cand['sid'].to_numpy()[hav_mat(g[['lat_use', 'lon_use']].to_numpy(float), cand[['lat', 'lon']].to_numpy(float)).argmin(1)]))
    alt = pd.concat(alt) if alt else q.assign(sid2=None)
    y = x[['lims_data_key', 'lat_use', 'lon_use', 'base_date', 'sid']].merge(alt, on=['lat_use', 'lon_use', 'base_date'], how='left')
    y['sid_f'] = np.where(nr, y['sid2'], y['sid'])
    y = y.merge(dd.rename(columns={'sid': 'sid_f'}), on=['sid_f', 'base_date'], how='left')
    res = {'why_pct': {k: round(float(v * 100), 2) for k, v in pd.Series(why).value_counts(normalize=True).items()}}
    for c in vcols:
        res[c] = {'now': round(float(x[c].notna().mean() * 100), 1), 'alt_row_that_day': round(float(y[c].notna().mean() * 100), 1)}
    print(name, json.dumps(res, ensure_ascii=False))
    OUT[name] = res
    return x

ast = pd.read_parquet(R + r"\raw_kma_ground_obs_station_info.parquet")
ast['chg_dt'] = pd.to_datetime(ast['chg_dt'], errors='coerce'); ast['reg_dt'] = pd.to_datetime(ast['reg_dt'], errors='coerce')
sA = ast.sort_values(['chg_dt', 'reg_dt'], ascending=False, na_position='last').drop_duplicates('stn_id')[['stn_id', 'lat', 'lon']].rename(columns={'stn_id': 'sid'}).astype({'lat': float, 'lon': float})
AV = ['avg_ta', 'max_ta', 'min_ta', 'sum_rn', 'avg_ws', 'max_ws', 'max_ins_ws', 'sum_sshr', 'sum_gsr', 'avg_ts', 'min_tg', 'avg_cm5_te', 'avg_cm10_te', 'avg_cm20_tes', 'avg_cm30_te']
run('ASOS', sA, pd.read_parquet(R + r"\raw_kma_asos_daily_info.parquet", columns=['stn_id', 'meas_date'] + AV), 'stn_id', 'meas_date', AV)

nst = pd.read_parquet(R + r"\raw_nas_agri_weather_obs_station.parquet")
nst['chg_dt'] = pd.to_datetime(nst['chg_dt'], errors='coerce')
sN0 = nst.sort_values('chg_dt', ascending=False).drop_duplicates('obsr_spot_code')
sN = sN0[['obsr_spot_code', 'instl_la', 'instl_lo']].rename(columns={'obsr_spot_code': 'sid', 'instl_la': 'lat', 'instl_lo': 'lon'}).astype({'lat': float, 'lon': float})
NV = ['temp', 'hghst_artmp', 'lowst_artmp', 'rn', 'wind', 'max_wind', 'widdir', 'srqty', 'sun_time', 'soil_wt', 'gr_temp', 'soil_temp', 'hum']
nas = pd.read_parquet(R + r"\raw_nas_agri_weather_daily_info.parquet", columns=['obsr_spot_code', 'stn_name', 'meas_date'] + NV)
xn = run('NAS', sN, nas, 'obsr_spot_code', 'meas_date', NV)
# 시험용으로 보이는 관측소
sus = nst[nst['obsr_spot_nm'].astype(str).str.contains('테스트|test|TEST|시험|임시', regex=True)]
print('\n이름에 테스트·시험 등이 들어간 농업기상 관측소:'); print(sus.to_string())
tst = []
for code in sus['obsr_spot_code'].unique():
    dn = nas[nas.obsr_spot_code == code]
    near = xn[xn['sid'] == code]
    tst.append({'code': code, 'name': sus[sus.obsr_spot_code == code]['obsr_spot_nm'].iloc[0], 'addr': str(sus[sus.obsr_spot_code == code]['instl_adres'].iloc[0]),
                'lat': float(sus[sus.obsr_spot_code == code]['instl_la'].iloc[0]), 'lon': float(sus[sus.obsr_spot_code == code]['instl_lo'].iloc[0]),
                'daily_rows': int(len(dn)), 'daily_first': str(pd.to_datetime(dn['meas_date']).min().date()) if len(dn) else None, 'daily_last': str(pd.to_datetime(dn['meas_date']).max().date()) if len(dn) else None,
                'stn_name_in_daily': dn['stn_name'].astype(str).value_counts().head(3).to_dict(),
                'domestic_samples_nearest': int(len(near)), 'of_which_temp_present': int(near['temp'].notna().sum())})
print(json.dumps(tst, ensure_ascii=False, indent=1))
OUT['nas_test_stations'] = tst
for nm, dfst, col in [('ASOS', ast, 'stn_ko')]:
    print(nm, '이름에 (예) 등이 들어간 행:', dfst[dfst[col].astype(str).str.contains(r'\(', regex=True)][['stn_id', col, 'oper_start_dt', 'oper_end_dt', 'use_yn']].head(20).to_string())

kst = pd.read_parquet(R + r"\raw_keco_airkorea_obs_station.parquet")
kst['chg_dt'] = pd.to_datetime(kst['chg_dt'], errors='coerce'); kst['u'] = (kst['use_yn'].astype(str) == '1').astype(int)
sK = kst.sort_values(['u', 'chg_dt'], ascending=False).drop_duplicates('station_nm')[['station_nm', 'dm_x', 'dm_y']].rename(columns={'station_nm': 'sid', 'dm_x': 'lat', 'dm_y': 'lon'}).astype({'lat': float, 'lon': float})
KV = ['no2_value', 'co_value', 'so2_value', 'pm10_value']
run('AIR', sK, pd.read_parquet(R + r"\raw_keco_airkorea_fine_dust_info.parquet", columns=['station_nm', 'msur_dt'] + KV), 'station_nm', 'msur_dt', KV)
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\v3_alt.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('saved')
