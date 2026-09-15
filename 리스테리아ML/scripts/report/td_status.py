# -*- coding: utf-8 -*-
"""정의서 색 표시 변수 상황 수치 모으기 (기존 측정 JSON + 원자료 파일, 값 변경 없음)"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
R = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected\raw"
S = json.load(open(T + r"\rep_stats.json", encoding='utf-8'))
V = {(x['table'], x['column']): x for g in ['asos', 'nas', 'air'] for x in S[g]}
COV = {(c['source'], c['column']): c for c in S['coverage_weather']}
for t, c, s in [('keco_airkorea_fine_dust_info', 'pm10_value', '에어코리아'), ('kma_asos_daily_info', 'sum_sshr', 'ASOS'), ('kma_asos_daily_info', 'sum_gsr', 'ASOS'),
                ('nas_agri_weather_daily_info', 'temp', 'NAS'), ('nas_agri_weather_daily_info', 'hum', 'NAS'), ('nas_agri_weather_daily_info', 'rn', 'NAS'),
                ('nas_agri_weather_daily_info', 'sun_time', 'NAS'), ('nas_agri_weather_daily_info', 'srqty', 'NAS')]:
    x = V[(t, c)]; cv = COV[(s, c)]
    print(c, {k: x[k] for k in x if k not in ('yearly',)}, '| cov', {k: cv[k] for k in cv})
aw = S['t_aw_cl']
print('\nt_aw_cl keys:', list(aw.keys()))
for c in ['aw', 'ph', 'na', 'mc']:
    print(c, aw[c])
OUT = {}
# 농업기상 관측소 설치 좌표
st = pd.read_parquet(R + r"\raw_nas_agri_weather_obs_station.parquet")
la = pd.to_numeric(st['instl_la'], errors='coerce'); lo = pd.to_numeric(st['instl_lo'], errors='coerce')
daily_codes = set(pd.read_parquet(R + r"\raw_nas_agri_weather_daily_info.parquet", columns=['obsr_spot_code'])['obsr_spot_code'])
dup = st.assign(la=la.round(6), lo=lo.round(6)).groupby(['la', 'lo'])['obsr_spot_code'].nunique()
OUT['nas_station'] = {'rows': int(len(st)), 'codes': int(st['obsr_spot_code'].nunique()), 'la_nonnull': int(la.notna().sum()), 'lo_nonnull': int(lo.notna().sum()),
                      'la_min': float(la.min()), 'la_max': float(la.max()), 'lo_min': float(lo.min()), 'lo_max': float(lo.max()),
                      'outside_korea_box': int(((la < 33) | (la > 39) | (lo < 124) | (lo > 132)).sum()),
                      'coords_shared_by_2plus': int((dup > 1).sum()), 'codes_with_daily': int(len(daily_codes & set(st['obsr_spot_code']))),
                      'use_yn': st['use_yn'].astype(str).value_counts().to_dict(), 'with_end_date': int(st['obsr_end_datetm'].notna().sum()),
                      'instl_la_type': str(st['instl_la'].dtype)}
print('\nNAS 관측소 좌표:', OUT['nas_station'])
# 예보 강수량
p = pd.read_parquet(R + r"\raw_kma_weather_predict_info.parquet")
print('\n예보표 컬럼:', list(p.columns)); print(p.head(3).T.to_string())
pm = pd.to_numeric(p['precip_mm'], errors='coerce')
date_cols = [c for c in p.columns if any(k in c.lower() for k in ('date', 'dt', 'tm', 'de'))]
rng = {c: (str(p[c].min()), str(p[c].max())) for c in date_cols}
OUT['predict'] = {'rows': int(len(p)), 'precip_nonnull': int(p['precip_mm'].notna().sum()), 'precip_num_nonnull': int(pm.notna().sum()),
                  'precip_min': float(pm.min()) if pm.notna().any() else None, 'precip_max': float(pm.max()) if pm.notna().any() else None,
                  'precip_zero': int((pm == 0).sum()), 'date_ranges': rng, 'grids': int(p[['lat', 'lon']].drop_duplicates().shape[0]),
                  'forecast_days': int(p['forecast_date'].nunique())}
print('예보 강수량:', OUT['predict'])
json.dump(OUT, open(T + r"\td_status.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
