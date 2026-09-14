# -*- coding: utf-8 -*-
"""보고서용 원자료 상태 측정 — 값 변경·결측 채움·단위 변환 없음. 개수/분포만 센다."""
import sys, re, json, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, openpyxl
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
R = C + r"\raw"
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
S0, S1 = pd.Timestamp('2015-04-07'), pd.Timestamp('2026-06-30')
NUM = re.compile(r'^\s*-?\d+(\.\d+)?\s*$')
OUT = {}

# ---------- 테이블정의서 한글 설명 ----------
wb = openpyxl.load_workbook(r"C:\Users\NT751\Desktop\천영진\테이블명세서(모델팀)\테이블정의서.xlsx", data_only=True)
SPEC = {}
for ws in wb.worksheets:
    cur = None
    for r in ws.iter_rows(values_only=True):
        a = r[0]
        if isinstance(a, str) and '(' in a and ('TABLE' in a or 'VIEW' in a):
            cur = a.split('(')[0].strip(); continue
        if cur and r[2] is not None and r[1] is not None:
            SPEC[(cur, str(r[2]).strip())] = (str(r[6]).strip() if r[6] is not None else '', str(r[3]) if r[3] is not None else '')
com = pd.read_csv(T + r"\rep_col_comments.csv") if os.path.exists(T + r"\rep_col_comments.csv") else pd.DataFrame(columns=['table', 'column', 'comment', 'type'])
COM = {(t, c): (cm if isinstance(cm, str) else '', ty) for t, c, cm, ty in zip(com['table'], com['column'], com['comment'], com['type'])}
def kname(table, col):
    d = SPEC.get((table, col), ('', ''))[0]
    if not d or d == 'None':
        d = COM.get((table, col), ('', ''))[0]
    return d
def dtype(table, col):
    return COM.get((table, col), ('', ''))[1] or SPEC.get((table, col), ('', ''))[1]

# ---------- 공통 측정 함수 ----------
def measure(df, table, col, id_col, date_col, rules):
    d = df[[id_col, date_col, col]]
    dt_ = pd.to_datetime(d[date_col])
    st = d[(dt_ >= S0) & (dt_ <= S1)]
    stdt = pd.to_datetime(st[date_col])
    s = st[col]
    text = s.dtype == object or str(s.dtype) in ('str', 'string')
    if text:
        ss = s.astype('string')
        present = s.notna() & (ss.str.strip() != '')
        parsable = present & ss.str.match(NUM).fillna(False)
        num = pd.to_numeric(s.where(parsable), errors='coerce')
    else:
        present = s.notna(); parsable = present; num = pd.to_numeric(s, errors='coerce')
    yr = present.groupby(stdt.dt.year).mean() * 100
    st_with = st.loc[present, id_col].nunique()
    rule_counts = {}
    for name, fn in rules.items():
        rule_counts[name] = int(fn(num).sum())
    allnum = pd.to_numeric(df[col].where(df[col].astype('string').str.match(NUM).fillna(False)) if text else df[col], errors='coerce')
    return {
        'table': table, 'column': col, 'korean': kname(table, col), 'db_type': dtype(table, col),
        'raw_period': f"{dt_.min():%Y-%m-%d} ~ {dt_.max():%Y-%m-%d}", 'rows_total': int(len(df)),
        'stations_total': int(d[id_col].nunique()), 'stations_with_value_study': int(st_with),
        'rows_study': int(len(st)), 'nonnull_study': round(float(present.mean() * 100), 1),
        'text_nonnumeric_study': int((present & ~parsable).sum()) if text else 0,
        'yearly_min': round(float(yr.min()), 1), 'yearly_max': round(float(yr.max()), 1),
        'yearly': {int(k): round(float(v), 1) for k, v in yr.items()},
        'min': None if num.dropna().empty else float(num.min()), 'p50': None if num.dropna().empty else float(num.median()),
        'max': None if num.dropna().empty else float(num.max()),
        'zero_pct_of_values': None if num.dropna().empty else round(float((num == 0).sum() / num.notna().sum() * 100), 1),
        'rules': rule_counts,
    }

def out_range(lo, hi):
    return lambda x: (x < lo) | (x > hi)

# ---------- ASOS ----------
asos = pd.read_parquet(R + r"\raw_kma_asos_daily_info.parquet")
A = 'kma_asos_daily_info'
asos_rules = {
    'avg_ta': {'범위밖(-35~45)': out_range(-35, 45)}, 'max_ta': {'범위밖(-35~45)': out_range(-35, 45)}, 'min_ta': {'범위밖(-35~45)': out_range(-35, 45)},
    'sum_rn': {'음수': lambda x: x < 0, '300mm초과': lambda x: x > 300},
    'avg_ws': {'음수·60초과': out_range(0, 60)}, 'max_ws': {'음수·60초과': out_range(0, 60)}, 'max_ins_ws': {'음수·80초과': out_range(0, 80)},
    'sum_sshr': {'14.5시간초과': lambda x: x > 14.5}, 'sum_gsr': {'값0': lambda x: x == 0, '35초과': lambda x: x > 35},
    'avg_ts': {'범위밖(-35~70)': out_range(-35, 70)}, 'min_tg': {'범위밖(-45~45)': out_range(-45, 45)},
    'avg_cm5_te': {'범위밖(-30~45)': out_range(-30, 45)}, 'avg_cm10_te': {'범위밖(-30~45)': out_range(-30, 45)},
    'avg_cm20_tes': {'범위밖(-30~45)': out_range(-30, 45)}, 'avg_cm30_te': {'범위밖(-30~45)': out_range(-30, 45)},
}
OUT['asos'] = [measure(asos, A, c, 'stn_id', 'meas_date', r) for c, r in asos_rules.items()]
stA = asos[(pd.to_datetime(asos.meas_date) >= S0) & (pd.to_datetime(asos.meas_date) <= S1)]
OUT['asos_rn_null_semantics'] = {
    'obs_days': int(stA['avg_ta'].notna().sum()),
    'rn_null_on_obs_days_pct': round(float((stA['sum_rn'].isna() & stA['avg_ta'].notna()).sum() / stA['avg_ta'].notna().sum() * 100), 1),
    'rn_zero_pct_of_values': round(float((stA['sum_rn'] == 0).sum() / stA['sum_rn'].notna().sum() * 100), 1),
}
ast = pd.read_parquet(R + r"\raw_kma_ground_obs_station_info.parquet")
multi = ast.groupby('stn_id').apply(lambda g: g[['lat', 'lon']].drop_duplicates().shape[0])
OUT['asos_station_hist'] = {'rows': int(len(ast)), 'stn_ids': int(ast['stn_id'].nunique()), 'stn_with_multi_coords': int((multi > 1).sum()),
                            'daily_stations': int(asos['stn_id'].nunique())}

# ---------- NAS ----------
nas = pd.read_parquet(R + r"\raw_nas_agri_weather_daily_info.parquet")
N = 'nas_agri_weather_daily_info'
nas_rules = {
    'temp': {'범위밖(-35~45)': out_range(-35, 45)}, 'hghst_artmp': {'±50경계값': lambda x: x.abs() >= 50, '범위밖(-35~45)': out_range(-35, 45)},
    'lowst_artmp': {'±50경계값': lambda x: x.abs() >= 50, '범위밖(-35~45)': out_range(-35, 45)},
    'hum': {'값0': lambda x: x == 0},
    'rn': {'490mm이상': lambda x: x >= 490},
    'wind': {'음수·60초과': out_range(0, 60)}, 'max_wind': {'음수·60초과': out_range(0, 60)}, 'widdir': {'범위밖(0~360)': out_range(0, 360)},
    'sun_time': {'870분초과': lambda x: x > 870}, 'srqty': {'39.5이상': lambda x: x >= 39.5, '값0': lambda x: x == 0},
    'gr_temp': {'범위밖(-35~70)': out_range(-35, 70)}, 'soil_temp': {'범위밖(-30~45)': out_range(-30, 45)},
    'soil_wt': {'음수': lambda x: x < 0, '값0': lambda x: x == 0, '50이상': lambda x: x >= 50},
}
OUT['nas'] = [measure(nas, N, c, 'obsr_spot_code', 'meas_date', r) for c, r in nas_rules.items()]
nst = pd.read_parquet(R + r"\raw_nas_agri_weather_obs_station.parquet")

# ---------- 에어코리아 ----------
air = pd.read_parquet(R + r"\raw_keco_airkorea_fine_dust_info.parquet")
K = 'keco_airkorea_fine_dust_info'
OUT['air'] = [measure(air, K, c, 'station_nm', 'msur_dt', {'음수': lambda x: x < 0, '값0': lambda x: x == 0}) for c in ['no2_value', 'co_value', 'so2_value', 'pm10_value']]
kst = pd.read_parquet(R + r"\raw_keco_airkorea_obs_station.parquet")
OUT['air_station'] = {'rows': int(len(kst)), 'station_nm': int(kst['station_nm'].nunique()), 'daily_station_nm': int(air['station_nm'].nunique()),
                      'daily_not_in_station_table': int(len(set(air['station_nm']) - set(kst['station_nm']))),
                      'dm_x_range': [float(kst['dm_x'].min()), float(kst['dm_x'].max())], 'dm_y_range': [float(kst['dm_y'].min()), float(kst['dm_y'].max())]}

# ---------- 해외 (국가 단위) ----------
if os.path.exists(R + r"\raw_global_weather_info.parquet"):
    gw = pd.read_parquet(R + r"\raw_global_weather_info.parquet")
    G = 'global_weather_info'
    gcols = {'temperature_mean': {}, 'temperature_max': {}, 'temperature_min': {}, 'precipitation_sum': {'음수': lambda x: x < 0},
             'relative_humidity_mean': {'범위밖(0~100)': out_range(0, 100)}, 'wind_speed_mean': {}, 'wind_speed_max': {},
             'sunshine_duration': {'86400초과': lambda x: x > 86400}, 'shortwave_radiation_sum': {}, 'soil_moisture_mean': {'음수': lambda x: x < 0}}
    OUT['global_weather_info'] = [measure(gw, G, c, 'alpha_3', 'date', r) for c, r in gcols.items()]
if os.path.exists(R + r"\raw_global_factory_weather_listeria_sameday.parquet"):
    gf = pd.read_parquet(R + r"\raw_global_factory_weather_listeria_sameday.parquet")
    OUT['global_factory_sameday_cols'] = {c: {'nonnull_pct': round(float(gf[c].notna().mean() * 100), 1), 'min': float(pd.to_numeric(gf[c], errors='coerce').min()), 'max': float(pd.to_numeric(gf[c], errors='coerce').max())}
                                          for c in gf.columns if c not in ('lims_data_key', 'h_lat', 'h_lon', 'base_date', 'obsr_de', 'lat_g', 'lon_g', 'weather_seq', 'created_at')}
    OUT['global_factory_korean'] = {c: kname('global_factory_weather', c) for c in gf.columns}

# ---------- 우리 시료 ----------
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet")
for c in ['lims_data_key', 'src_route', 'judge_result', 'prdlst_code', 'legal_dong_code', 'date_basis']:
    body[c] = body[c].astype(object).where(body[c].notna(), None)
body['base_date'] = pd.to_datetime(body['mnfctur_de']).dt.normalize().astype('datetime64[ms]')
db = pd.read_parquet(C + r"\00_body_db.parquet")
b = body.merge(db[['lims_data_key', 'lat', 'lon', 'mnfctur_de_or_fallback', 'prdlst_mnfctur_sttemnt_no', 'bsn_lcns_ledg_no', 'cntc_sys_key2',
                   'mapped_prdlst_code', 'prdctn_nation_code', 'orgplce_nation_code', 'import_nation_code', 'aw_grade']].rename(columns={'lat': 'lat_db', 'lon': 'lon_db'}), on='lims_data_key', how='left')
OUT['sample'] = {'rows': int(len(b)), 'domestic': int((b.src_route == 'domestic').sum()), 'overseas': int((b.src_route == 'overseas').sum()),
                 'detected': int((b.judge_result == '검출').sum()), 'db_body_rows': int(len(db)),
                 'base_date_eq_fallback': float((pd.to_datetime(b['mnfctur_de_or_fallback']) == pd.to_datetime(b['base_date'])).mean()),
                 'lat_db_null': int(b['lat_db'].isna().sum()), 'legal_dong_null_domestic': int(b.loc[b.src_route == 'domestic', 'legal_dong_code'].isna().sum())}
b['lat_use'] = b['lat_db'].fillna(b['lat'].astype(float)); b['lon_use'] = b['lon_db'].fillna(b['lon'].astype(float))
dom = b[b.src_route == 'domestic']

def nearest(pts, st):
    P = np.radians(pts[['lat', 'lon']].to_numpy(float)); S = np.radians(st[['lat', 'lon']].to_numpy(float))
    ids = np.empty(len(P), dtype=object); dist = np.empty(len(P))
    for i in range(0, len(P), 1000):
        p = P[i:i + 1000]
        a = np.sin((p[:, None, 0] - S[None, :, 0]) / 2) ** 2 + np.cos(p[:, None, 0]) * np.cos(S[None, :, 0]) * np.sin((p[:, None, 1] - S[None, :, 1]) / 2) ** 2
        dd = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1))); j = dd.argmin(1)
        ids[i:i + 1000] = st['sid'].to_numpy()[j]; dist[i:i + 1000] = dd[np.arange(len(p)), j]
    return ids, dist

pts = dom[['lat_use', 'lon_use']].drop_duplicates().rename(columns={'lat_use': 'lat', 'lon_use': 'lon'}).reset_index(drop=True)
def latest(st, sid, la, lo, sort_cols):
    s = st.copy()
    for c in sort_cols:
        if c in s.columns: s[c] = pd.to_datetime(s[c], errors='coerce')
    s = s.sort_values([c for c in sort_cols if c in s.columns], ascending=False, na_position='last')
    return s.drop_duplicates(sid)[[sid, la, lo]].rename(columns={sid: 'sid', la: 'lat', lo: 'lon'}).dropna()

COV = []
def coverage(name, stations, daily, did, dcol, cols):
    s = stations[stations['sid'].isin(set(daily[did]))]
    ids, dist = nearest(pts, s)
    m = pts.assign(sid=ids, dist=dist)
    x = dom.merge(m, left_on=['lat_use', 'lon_use'], right_on=['lat', 'lon'], how='left')
    dd = daily[[did, dcol] + cols].rename(columns={did: 'sid', dcol: 'base_date'}).copy()
    dd['base_date'] = pd.to_datetime(dd['base_date']).astype('datetime64[ms]')
    dd = dd.drop_duplicates(['sid', 'base_date'])
    x = x.merge(dd, on=['sid', 'base_date'], how='left')
    det = x['judge_result'] == '검출'
    for c in cols:
        v = x[c]; pres = v.notna() & (v.astype('string').str.strip() != '').fillna(False)
        dstudy = pd.to_datetime(daily[dcol])
        has = daily.loc[daily[c].notna() & (dstudy >= S0) & (dstudy <= S1), did].unique()
        s2 = s[s['sid'].isin(has)]
        ids2, dist2 = nearest(pts, s2)
        m2 = pts.assign(sid=ids2, dist=dist2)
        x2 = dom[['lat_use', 'lon_use', 'base_date']].merge(m2, left_on=['lat_use', 'lon_use'], right_on=['lat', 'lon'], how='left').merge(dd[['sid', 'base_date', c]], on=['sid', 'base_date'], how='left')
        pres2 = x2[c].notna()
        COV.append({'source': name, 'column': c, 'stations_used': int(len(s)), 'dist_p50': round(float(np.median(x['dist'])), 1), 'dist_max': round(float(x['dist'].max()), 1),
                    'sameday_pct': round(float(pres.mean() * 100), 1), 'sameday_pct_detected': round(float(pres[det].mean() * 100), 1),
                    'var_stations': int(len(s2)), 'var_dist_p50': round(float(np.median(x2['dist'])), 1), 'var_sameday_pct': round(float(pres2.mean() * 100), 1)})

coverage('ASOS', latest(ast, 'stn_id', 'lat', 'lon', ['chg_dt', 'reg_dt']), asos, 'stn_id', 'meas_date', list(asos_rules))
coverage('NAS', latest(nst, 'obsr_spot_code', 'instl_la', 'instl_lo', ['chg_dt']), nas, 'obsr_spot_code', 'meas_date', list(nas_rules))
coverage('에어코리아', latest(kst.assign(use1=(kst['use_yn'].astype(str) == '1').astype(int)), 'station_nm', 'dm_x', 'dm_y', ['use1', 'chg_dt']), air, 'station_nm', 'msur_dt', ['no2_value', 'co_value', 'so2_value', 'pm10_value'])
OUT['coverage_weather'] = COV

# 토양 pH: 법정동+조사연도 필지 존재 여부 (집계값 아님)
if os.path.exists(R + r"\raw_soil_ph_info_dong_year_counts.parquet"):
    sdy = pd.read_parquet(R + r"\raw_soil_ph_info_dong_year_counts.parquet")
    sdy['any_year'] = sdy['any_year'].astype(str).str.strip()
    dy = set(zip(sdy.loc[sdy.n_acid > 0, 'legal_dong_code'], sdy.loc[sdy.n_acid > 0, 'any_year']))
    dong_any = set(sdy.loc[sdy.n_acid > 0, 'legal_dong_code'])
    sgg_y = set(zip(sdy.loc[sdy.n_acid > 0, 'legal_dong_code'].str[:5], sdy.loc[sdy.n_acid > 0, 'any_year']))
    yy = dom['base_date'].dt.year.astype(str)
    k1 = pd.Series([(a, y) in dy for a, y in zip(dom['legal_dong_code'], yy)])
    k2 = dom['legal_dong_code'].isin(dong_any).reset_index(drop=True)
    k3 = pd.Series([((a or '')[:5], y) in sgg_y for a, y in zip(dom['legal_dong_code'], yy)])
    par = pd.Series([n for n in dom[['legal_dong_code']].assign(y=yy.values).merge(sdy[sdy.n_acid > 0].rename(columns={'any_year': 'y'}), on=['legal_dong_code', 'y'], how='left')['n_acid']])
    OUT['coverage_soil'] = {'dong_year_pct': round(float(k1.mean() * 100), 1), 'dong_anyyear_pct': round(float(k2.mean() * 100), 1),
                            'sgg_year_pct': round(float(k3.mean() * 100), 1),
                            'parcels_per_dong_year_p50': float(par.dropna().median()) if par.notna().any() else None,
                            'parcels_per_dong_year_max': float(par.dropna().max()) if par.notna().any() else None,
                            'dong_year_pct_detected': round(float(k1[(dom['judge_result'] == '검출').to_numpy()].mean() * 100), 1),
                            'sgg_level_code_rows': int(dom['legal_dong_code'].astype(str).str.endswith('00000').sum()),
                            'sgg_level_code_pct': round(float(dom['legal_dong_code'].astype(str).str.endswith('00000').mean() * 100), 1)}

# 해외: 국가 단위 / 제조소 좌표
ovs = b[b.src_route == 'overseas'].copy()
nat = pd.read_parquet(C + r"\61_nation_code.parquet")
a2a3 = dict(zip(nat['alpha_2'].str.strip(), nat['alpha_3'].str.strip()))
ovs['a2'] = ovs['prdctn_nation_code'].fillna(ovs['orgplce_nation_code']).fillna(ovs['import_nation_code'])
ovs['a3'] = ovs['a2'].map(a2a3)
cov_o = {'nation_code_present_pct': round(float(ovs['a2'].notna().mean() * 100), 1), 'alpha3_mapped_pct': round(float(ovs['a3'].notna().mean() * 100), 1)}
if os.path.exists(R + r"\raw_global_weather_info.parquet"):
    gcl = ['temperature_mean', 'temperature_max', 'temperature_min', 'precipitation_sum', 'relative_humidity_mean', 'wind_speed_mean', 'wind_speed_max', 'sunshine_duration', 'shortwave_radiation_sum', 'soil_moisture_mean']
    gwd = gw[['alpha_3', 'date'] + gcl].copy(); gwd['date'] = pd.to_datetime(gwd['date']).astype('datetime64[ms]')
    xo = ovs.merge(gwd.rename(columns={'alpha_3': 'a3', 'date': 'base_date'}), on=['a3', 'base_date'], how='left')
    cov_o['global_weather_info_sameday_pct'] = round(float(xo['temperature_mean'].notna().mean() * 100), 1)
    cov_o['global_weather_info_sameday_by_col'] = {c: round(float(xo[c].notna().mean() * 100), 2) for c in gcl}
    cov_o['global_weather_info_countries'] = int(ovs['a3'].dropna().nunique())
if os.path.exists(R + r"\raw_global_factory_weather_listeria_sameday.parquet"):
    hit = ovs['lims_data_key'].isin(set(gf['lims_data_key']))
    cov_o['global_factory_sameday_pct'] = round(float(hit.mean() * 100), 1)
    cov_o['global_factory_sameday_pct_detected'] = round(float(hit[ovs['judge_result'] == '검출'].mean() * 100), 1)
    cov_o['ovs_coords'] = int(ovs[['lat_use', 'lon_use']].drop_duplicates().shape[0])
OUT['coverage_overseas'] = cov_o

# LIMS·대장 원자료 연결 (키로 붙였을 때 원값 존재율)
m_org = pd.read_parquet(C + r"\02a_m_org.parquet")
ledg = pd.read_parquet(C + r"\02b_ledg.parquet"); bsn = pd.read_parquet(C + r"\02c_bsn.parquet"); cl = pd.read_parquet(C + r"\02d_cl.parquet")
s_org = pd.read_parquet(C + r"\02e_s_org.parquet"); co = pd.read_parquet(C + r"\02f_co.parquet")
L = []
def lcov(df, key_left, key_right, cols, table, note=''):
    left = b[list(dict.fromkeys(['lims_data_key', 'src_route', 'judge_result', key_left]))]
    right = df.drop_duplicates(key_right)[[key_right] + cols].assign(_hit=1)
    x = left.merge(right, left_on=key_left, right_on=key_right, how='left')
    keyhit = x['_hit'].notna()
    for c in cols:
        v = x[c]; pres = v.notna() & (v.astype('string').str.strip() != '').fillna(False)
        L.append({'table': table, 'column': c, 'korean': kname(table, c), 'key': f'{key_left} = {key_right}', 'dup_rule': note,
                  'key_match_domestic': round(float(keyhit[x.src_route == 'domestic'].mean() * 100), 1), 'key_match_overseas': round(float(keyhit[x.src_route == 'overseas'].mean() * 100), 1),
                  'value_domestic': round(float(pres[x.src_route == 'domestic'].mean() * 100), 1), 'value_overseas': round(float(pres[x.src_route == 'overseas'].mean() * 100), 1),
                  'value_detected': round(float(pres[x.judge_result == '검출'].mean() * 100), 1),
                  'value_detected_domestic': round(float(pres[(x.judge_result == '검출') & (x.src_route == 'domestic')].mean() * 100), 1),
                  'value_rows_domestic': int(pres[x.src_route == 'domestic'].sum()),
                  'top_values': {str(k)[:30]: int(n) for k, n in v.value_counts().head(4).items()}})
lcov(m_org, 'lims_data_key', 'lims_data_key', ['cnvync_sttus_tp', 'cnvync_sttus_tp_unit_code', 'cnvync_mth_code', 'splore_sttus_code', 'distb_de', 'valid_tmlmt', 'pick_de', 'upper_prdlst_code', 'upper_prdlst_nm', 'expr_se_nm'], 'm_org_lims_data_info')
lr = ledg.assign(_live=(ledg['del_yn'].fillna('N') == 'N')).sort_values(['prdlst_report_no', '_live', 'last_updt_dtm'], ascending=[True, False, False])
lcov(lr, 'prdlst_mnfctur_sttemnt_no', 'prdlst_report_no', ['mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'pog_daycnt', 'qlity_mntnc_tmlmt_daycnt', 'frmlc_mtrqlt'], 't_prdlst_mnf_ledg', '품목보고번호당 여러 행 → 삭제 안 된 최신 행')
lcov(bsn, 'bsn_lcns_ledg_no', 'bsn_lcns_ledg_no', ['wrkshp_pkrm_area', 'domini_sort', 'fish_sort'], 't_bsn_lcns_ledg')
b['prdlst_code_use'] = b['mapped_prdlst_code'].fillna(b['prdlst_code'])
lcov(cl, 'prdlst_code_use', 'prdlst_cd', ['lv', 'hrnk_prdlst_cd', 'htrk_prdlst_cd', 'fdgrp_yn', 'artcl_gun_code', 'food_hf_ls_cl_cd', 'dfn'], 't_prdlst_cl')
so = s_org.assign(_lis=(s_org['label'] == '리스테리아')).sort_values(['rcept_no', '_lis', 'lims_seq'], ascending=[True, False, True])
lcov(so, 'cntc_sys_key2', 'rcept_no', ['prduct_dvs_cd', 'dclr_prduct_dvs', 'hsk_no', 'distb_bgn_de', 'distb_end_de', 'packng_plc_cd', 'ovsmnfst_cd'], 's_org_lims_data_info', '접수번호당 여러 행(시험항목별) → 리스테리아 행 우선')
# 품목제조번호 숫자만 형태 추가 매칭 수
digits = b['prdlst_mnfctur_sttemnt_no'].astype('string').str.replace(r'\D', '', regex=True)
raw_hit = b['prdlst_mnfctur_sttemnt_no'].isin(set(ledg['prdlst_report_no']))
dig_hit = ~raw_hit & digits.isin(set(ledg['prdlst_report_no'])).fillna(False)
OUT['ledg_key_forms'] = {'raw_match_rows': int(raw_hit.sum()), 'digits_only_extra_rows': int(dig_hit.sum()), 'key_present_domestic_pct': round(float(b.loc[b.src_route == 'domestic', 'prdlst_mnfctur_sttemnt_no'].notna().mean() * 100), 1)}
OUT['lims_coverage'] = L
# 품목분류(몸통) 원값
OUT['body_food_type'] = {c: round(float(b[c].notna().mean() * 100), 1) for c in ['prdlst_code', 'train_prdlst_code', 'train_prdlst_nm']}

# t_aw_cl (수분활성도·pH·염도) 원값 — 품목코드 정확 일치만
aw = pd.read_parquet(C + r"\80_t_aw_cl.parquet")
x = b[['prdlst_code_use', 'src_route', 'judge_result']].merge(aw.drop_duplicates('prdlst_cd'), left_on='prdlst_code_use', right_on='prdlst_cd', how='left')
awc = {}
for c in ['aw', 'ph', 'na', 'mc']:
    v = x[c].astype('string')
    pres = x[c].notna() & ~v.str.strip().isin(['', '-']).fillna(True)
    awc[c] = {'domestic': round(float(pres[x.src_route == 'domestic'].mean() * 100), 1), 'overseas': round(float(pres[x.src_route == 'overseas'].mean() * 100), 1),
              'detected': round(float(pres[x.judge_result == '검출'].mean() * 100), 1),
              'detected_domestic': round(float(pres[(x.judge_result == '검출') & (x.src_route == 'domestic')].mean() * 100), 1),
              'table_nonblank': int((aw[c].notna() & ~aw[c].astype('string').str.strip().isin(['', '-']).fillna(True)).sum()),
              'range_text_rows': int(aw[c].astype('string').str.contains(r'~|～|<|>|≤|≥|\d\s*-\s*\d', regex=True).fillna(False).sum())}
awc['code_match_pct'] = round(float(x['prdlst_cd'].notna().mean() * 100), 1)
awc['rows'] = int(len(aw)); awc['aw_source'] = aw['aw_source'].astype('string').str.split('(').str[0].str.strip().value_counts().head(6).to_dict()
awc['aw_nonoriginal'] = int((aw['aw_source'].astype('string').str.strip() != 'original').sum())

import re as _re
def _rng(v):
    t = '' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
    if t in ('', '-', 'nan', '<NA>'): return None
    nums = _re.findall(r'\d+(?:\.\d+)?', t.replace('..', '.'))
    return tuple(float(n) for n in nums) if nums else ('txt', t)
f = pd.read_excel(r"C:\Users\NT751\Desktop\천영진\리서치팀문서\통합_수분활성도_수분함량_식품pH_염도_최종.xlsx", sheet_name='값')
fm = f.merge(aw, left_on='PRDLST_CD', right_on='prdlst_cd', how='left')
fa = fm['수분활성도'].map(_rng); da = fm['aw'].map(_rng)
filled = fm.loc[fa.isna() & da.notna(), 'PRDLST_CD']
changed = fm.loc[fa.notna() & da.notna() & (fa.astype(str) != da.astype(str)), 'PRDLST_CD']
sym = fm.loc[fa.notna() & da.notna() & (fa.astype(str) == da.astype(str)) & (fm['수분활성도'].astype(str).str.strip() != fm['aw'].astype(str).str.strip()), 'PRDLST_CD']
awc['file_codes'] = int(len(f)); awc['db_only_codes'] = int((~aw.prdlst_cd.isin(f.PRDLST_CD)).sum())
awc['file_aw_blank_codes'] = int(fa.isna().sum()); awc['filled_codes'] = int(len(filled))
awc['filled_codes_source'] = fm.loc[fm.PRDLST_CD.isin(filled), 'aw_source'].astype(str).str.split('(').str[0].str.strip().value_counts().to_dict()
awc['changed_codes'] = int(len(changed)); awc['changed_examples'] = fm.loc[fm.PRDLST_CD.isin(changed), ['KOR_NM', '수분활성도', 'aw']].astype(str).head(8).values.tolist()
awc['symbol_only_codes'] = int(len(sym))
awc['sample_rows_filled'] = int(b['prdlst_code_use'].isin(set(filled)).sum()); awc['sample_rows_changed'] = int(b['prdlst_code_use'].isin(set(changed)).sum())
awc['aw_015_rows'] = int((aw['aw'].astype(str).str.strip() == '0.15').sum()); awc['aw_015_db_only'] = int(((aw['aw'].astype(str).str.strip() == '0.15') & ~aw.prdlst_cd.isin(f.PRDLST_CD)).sum())
OUT['t_aw_cl'] = awc

json.dump(OUT, open(T + r"\rep_stats.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: (v if k not in ('asos', 'nas', 'air', 'global_weather_info', 'lims_coverage') else f'{len(v)} items') for k, v in OUT.items()}, ensure_ascii=False, indent=1, default=str)[:6000])
