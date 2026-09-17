# -*- coding: utf-8 -*-
"""시료 기준 결측 원인 분해: 최근접 관측소에 (a) 그날 행 자체가 없음 (b) 행은 있으나 그 항목만 빈 값 (c) 그 관측소가 연구기간 내내 그 항목을 기록 안 함"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = C + r"\raw"
S0, S1 = pd.Timestamp('2015-04-07'), pd.Timestamp('2026-06-30')
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'mnfctur_de', 'lat', 'lon', 'legal_dong_code'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'lat', 'lon'])
b = body.merge(db.rename(columns={'lat': 'lat_db', 'lon': 'lon_db'}), on='lims_data_key', how='left')
b['lat_use'] = b['lat_db'].fillna(b['lat'].astype(float)); b['lon_use'] = b['lon_db'].fillna(b['lon'].astype(float))
b['base_date'] = pd.to_datetime(b['mnfctur_de']).dt.normalize()
dom = b[b.src_route == 'domestic'].copy()
print('우리 시료 legal_dong_code 예:', dom['legal_dong_code'].astype(str).head(3).tolist())
pts = dom[['lat_use', 'lon_use']].drop_duplicates().reset_index(drop=True)
def nearest(st):
    P = np.radians(pts.to_numpy(float)); Sx = np.radians(st[['lat', 'lon']].to_numpy(float))
    ids = np.empty(len(P), dtype=object)
    for i in range(0, len(P), 1000):
        p = P[i:i + 1000]
        aa = np.sin((p[:, None, 0] - Sx[None, :, 0]) / 2) ** 2 + np.cos(p[:, None, 0]) * np.cos(Sx[None, :, 0]) * np.sin((p[:, None, 1] - Sx[None, :, 1]) / 2) ** 2
        ids[i:i + 1000] = st['sid'].to_numpy()[(2 * 6371 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))).argmin(1)]
    return pts.assign(sid=ids)
OUT = {}
def decompose(name, st, daily, did, dcol, cols):
    s = st[st['sid'].isin(set(daily[did]))]
    m = nearest(s)
    x = dom.merge(m, on=['lat_use', 'lon_use'], how='left')
    dd = daily[[did, dcol] + cols].rename(columns={did: 'sid', dcol: 'base_date'}).copy()
    dd['base_date'] = pd.to_datetime(dd['base_date']).dt.normalize()
    dd = dd.drop_duplicates(['sid', 'base_date'])
    x = x.merge(dd.assign(_row=1), on=['sid', 'base_date'], how='left')
    stud = dd[(dd.base_date >= S0) & (dd.base_date <= S1)]
    for c in cols:
        never = set(stud.groupby('sid')[c].apply(lambda v: v.notna().any()).loc[lambda v: ~v].index)
        v = x[c]
        pres = v.notna()
        no_row = x['_row'].isna()
        never_st = ~pres & ~no_row & x['sid'].isin(never)
        null_other = ~pres & ~no_row & ~x['sid'].isin(never)
        OUT[f'{name}.{c}'] = {'present': round(float(pres.mean() * 100), 1), 'no_row': round(float(no_row.mean() * 100), 1),
                              'station_never_records': round(float(never_st.mean() * 100), 1), 'null_on_that_day': round(float(null_other.mean() * 100), 1),
                              'nearest_stations_used': int(x['sid'].nunique()), 'of_which_never': int(len(set(x['sid'].dropna()) & never))}
        print(name, c, OUT[f'{name}.{c}'])
ast = pd.read_parquet(R + r"\raw_kma_ground_obs_station_info.parquet")
ast['chg_dt'] = pd.to_datetime(ast['chg_dt'], errors='coerce'); ast['reg_dt'] = pd.to_datetime(ast['reg_dt'], errors='coerce')
sA = ast.sort_values(['chg_dt', 'reg_dt'], ascending=False, na_position='last').drop_duplicates('stn_id')[['stn_id', 'lat', 'lon']].rename(columns={'stn_id': 'sid'}).astype({'lat': float, 'lon': float})
asos = pd.read_parquet(R + r"\raw_kma_asos_daily_info.parquet", columns=['stn_id', 'meas_date', 'avg_ta', 'sum_rn', 'avg_ws', 'sum_sshr', 'sum_gsr', 'avg_ts', 'avg_cm5_te', 'avg_cm10_te', 'avg_cm20_tes'])
asos['sum_sshr'] = asos['sum_sshr'].where(asos['sum_sshr'].astype(str).str.strip() != '')
decompose('ASOS', sA, asos, 'stn_id', 'meas_date', ['avg_ta', 'sum_rn', 'avg_ws', 'sum_sshr', 'sum_gsr', 'avg_ts', 'avg_cm5_te'])
nst = pd.read_parquet(R + r"\raw_nas_agri_weather_obs_station.parquet")
nst['chg_dt'] = pd.to_datetime(nst['chg_dt'], errors='coerce')
sN = nst.sort_values('chg_dt', ascending=False).drop_duplicates('obsr_spot_code')[['obsr_spot_code', 'instl_la', 'instl_lo']].rename(columns={'obsr_spot_code': 'sid', 'instl_la': 'lat', 'instl_lo': 'lon'}).astype({'lat': float, 'lon': float})
nas = pd.read_parquet(R + r"\raw_nas_agri_weather_daily_info.parquet", columns=['obsr_spot_code', 'meas_date', 'temp', 'rn', 'wind', 'srqty', 'sun_time', 'soil_wt', 'gr_temp', 'soil_temp'])
nas['sun_time'] = nas['sun_time'].where(nas['sun_time'].astype(str).str.strip() != '')
decompose('NAS', sN, nas, 'obsr_spot_code', 'meas_date', ['temp', 'rn', 'wind', 'srqty', 'sun_time', 'soil_wt', 'gr_temp', 'soil_temp'])
kst = pd.read_parquet(R + r"\raw_keco_airkorea_obs_station.parquet")
kst['chg_dt'] = pd.to_datetime(kst['chg_dt'], errors='coerce'); kst['u'] = (kst['use_yn'].astype(str) == '1').astype(int)
sK = kst.sort_values(['u', 'chg_dt'], ascending=False).drop_duplicates('station_nm')[['station_nm', 'dm_x', 'dm_y']].rename(columns={'station_nm': 'sid', 'dm_x': 'lat', 'dm_y': 'lon'}).astype({'lat': float, 'lon': float})
air = pd.read_parquet(R + r"\raw_keco_airkorea_fine_dust_info.parquet", columns=['station_nm', 'msur_dt', 'no2_value', 'co_value', 'so2_value'])
decompose('AIR', sK, air, 'station_nm', 'msur_dt', ['no2_value', 'co_value', 'so2_value'])
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\v3_evidence2.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
