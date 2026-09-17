# -*- coding: utf-8 -*-
"""'그날 자료 행 없음' 관측소의 실제 개설·종료 정보 대조 + (참고) 그날 자료가 있는 관측소 중 가장 가까운 곳을 쓸 때 연결률 (측정만)"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = C + r"\raw"
OUT = {}
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'mnfctur_de', 'lat', 'lon'])
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

def run(name, st, daily, did, dcol, vcols, stinfo, stkey):
    dd = daily[[did, dcol] + vcols].rename(columns={did: 'sid', dcol: 'base_date'}).copy()
    dd['base_date'] = pd.to_datetime(dd['base_date']).dt.normalize()
    dd = dd.drop_duplicates(['sid', 'base_date'])
    s = st[st['sid'].isin(set(dd['sid']))].reset_index(drop=True)
    pts = dom[['lat_use', 'lon_use']].drop_duplicates().reset_index(drop=True)
    idx = np.empty(len(pts), dtype=int); dist = np.empty(len(pts))
    for i in range(0, len(pts), 1000):
        D = hav_mat(pts.iloc[i:i + 1000].to_numpy(float), s[['lat', 'lon']].to_numpy(float))
        idx[i:i + 1000] = D.argmin(1); dist[i:i + 1000] = D.min(1)
    pts['sid'] = s['sid'].to_numpy()[idx]; pts['km'] = dist
    x = dom.merge(pts, on=['lat_use', 'lon_use'], how='left').merge(dd.assign(_row=1), on=['sid', 'base_date'], how='left')
    span = dd.groupby('sid')['base_date'].agg(db_first='min', db_last='max').reset_index()
    x = x.merge(span, on='sid', how='left')
    nr = x['_row'].isna()
    x['why'] = np.where(~nr, 'row', np.where(x['base_date'] < x['db_first'], 'before', np.where(x['base_date'] > x['db_last'], 'after', 'gap')))
    top = x[nr].groupby(['sid', 'why']).size().rename('samples').reset_index().merge(span, on='sid').sort_values('samples', ascending=False).head(12)
    top = top.merge(stinfo, left_on='sid', right_on=stkey, how='left')
    print(f'\n[{name}] 행 없음 상위 관측소:'); print(top.to_string())
    OUT[f'{name}_norow_top'] = top.astype(str).to_dict('records')
    # (참고) 그날 자료 행이 있는 관측소 중 가장 가까운 곳
    q = x[nr][['lat_use', 'lon_use', 'base_date']].drop_duplicates().reset_index(drop=True)
    day_st = dd.groupby('base_date')['sid'].apply(set)
    alt = []
    for d, g in q.groupby('base_date'):
        ss_ = day_st.get(d, set())
        cand = s[s['sid'].isin(ss_)]
        if len(cand) == 0:
            alt.append(g.assign(sid2=None, km2=np.nan)); continue
        D = hav_mat(g[['lat_use', 'lon_use']].to_numpy(float), cand[['lat', 'lon']].to_numpy(float))
        alt.append(g.assign(sid2=cand['sid'].to_numpy()[D.argmin(1)], km2=D.min(1)))
    alt = pd.concat(alt)
    y = x.merge(alt, on=['lat_use', 'lon_use', 'base_date'], how='left')
    y['sid_f'] = np.where(nr, y['sid2'], y['sid']); y['km_f'] = np.where(nr, y['km2'], y['km'])
    y = y.drop(columns=vcols + ['_row']).merge(dd.rename(columns={'sid': 'sid_f'}).assign(_row=1), on=['sid_f', 'base_date'], how='left')
    res = {'n': int(len(x)), 'no_row_pct_now': round(float(nr.mean() * 100), 2),
           'why_pct': {k: round(float(v * 100), 2) for k, v in x['why'].value_counts(normalize=True).items()},
           'alt_no_row_pct': round(float(y['_row'].isna().mean() * 100), 2),
           'replaced_dist_p50_now': round(float(x.loc[nr, 'km'].median()), 1), 'replaced_dist_p50_alt': round(float(y.loc[nr.to_numpy(), 'km_f'].median()), 1),
           'all_dist_p50_now': round(float(x['km'].median()), 1), 'all_dist_p50_alt': round(float(pd.Series(y['km_f']).median()), 1)}
    for c in vcols:
        res[f'{c}_present_now'] = round(float(x[c].notna().mean() * 100), 1)
        res[f'{c}_present_alt'] = round(float(y[c].notna().mean() * 100), 1)
    print(f'[{name}] 요약:', res)
    OUT[f'{name}_alt'] = res

ast = pd.read_parquet(R + r"\raw_kma_ground_obs_station_info.parquet")
ast['chg_dt'] = pd.to_datetime(ast['chg_dt'], errors='coerce'); ast['reg_dt'] = pd.to_datetime(ast['reg_dt'], errors='coerce')
sA = ast.sort_values(['chg_dt', 'reg_dt'], ascending=False, na_position='last').drop_duplicates('stn_id')
infoA = sA[['stn_id', 'stn_ko', 'oper_start_dt', 'oper_end_dt', 'use_yn']]
sA = sA[['stn_id', 'lat', 'lon']].rename(columns={'stn_id': 'sid'}).astype({'lat': float, 'lon': float})
asos = pd.read_parquet(R + r"\raw_kma_asos_daily_info.parquet", columns=['stn_id', 'meas_date', 'avg_ta', 'sum_rn'])
run('ASOS', sA, asos, 'stn_id', 'meas_date', ['avg_ta', 'sum_rn'], infoA, 'stn_id')
# 같은 관측소 이력 행의 운영 기간(행이 여러 개일 수 있음)
for sid in [r['sid'] for r in OUT['ASOS_norow_top'][:6]]:
    print('ASOS 관측소표 이력', sid, ast[ast.stn_id.astype(str) == str(sid)][['stn_id', 'stn_ko', 'lat', 'lon', 'oper_start_dt', 'oper_end_dt', 'use_yn', 'reg_dt', 'chg_dt']].to_string(index=False))

nst = pd.read_parquet(R + r"\raw_nas_agri_weather_obs_station.parquet")
nst['chg_dt'] = pd.to_datetime(nst['chg_dt'], errors='coerce')
sN = nst.sort_values('chg_dt', ascending=False).drop_duplicates('obsr_spot_code')
infoN = sN[['obsr_spot_code', 'obsr_spot_nm', 'obsr_begin_datetm', 'obsr_end_datetm', 'use_yn']]
sN = sN[['obsr_spot_code', 'instl_la', 'instl_lo']].rename(columns={'obsr_spot_code': 'sid', 'instl_la': 'lat', 'instl_lo': 'lon'}).astype({'lat': float, 'lon': float})
nas = pd.read_parquet(R + r"\raw_nas_agri_weather_daily_info.parquet", columns=['obsr_spot_code', 'meas_date', 'temp', 'rn'])
run('NAS', sN, nas, 'obsr_spot_code', 'meas_date', ['temp', 'rn'], infoN, 'obsr_spot_code')

kst = pd.read_parquet(R + r"\raw_keco_airkorea_obs_station.parquet")
kst['chg_dt'] = pd.to_datetime(kst['chg_dt'], errors='coerce'); kst['u'] = (kst['use_yn'].astype(str) == '1').astype(int)
sK = kst.sort_values(['u', 'chg_dt'], ascending=False).drop_duplicates('station_nm')
infoK = sK[['station_nm', 'year', 'year_end', 'mang_name', 'use_yn']]
sK = sK[['station_nm', 'dm_x', 'dm_y']].rename(columns={'station_nm': 'sid', 'dm_x': 'lat', 'dm_y': 'lon'}).astype({'lat': float, 'lon': float})
air = pd.read_parquet(R + r"\raw_keco_airkorea_fine_dust_info.parquet", columns=['station_nm', 'msur_dt', 'no2_value'])
run('AIR', sK, air, 'station_nm', 'msur_dt', ['no2_value'], infoK, 'station_nm')
# 에어코리아: 행 없음 관측소 중 DB 첫 기록 연도와 측정소표 설치연도(year) 비교
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\v3_check6.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('saved')
