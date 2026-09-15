# -*- coding: utf-8 -*-
"""v3 보고서 문구 검증 (원자료 측정만)
1) 데이터출처 엑셀 6·9·21행 빨간 글자 원문
2) 농업기상 관측소 코드 7번째 글자별 항목 기록 건수(0건인지 정확히)
3) 국내 시료 '그날 자료 행 없음'의 성격: 관측소 첫 기록 전 / 마지막 기록 후 / 운영 중 빠진 날
4) 관측소표·일자료표에 유형·결측 사유 컬럼이 있는지"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, openpyxl
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = C + r"\raw"
S0, S1 = pd.Timestamp('2015-04-07'), pd.Timestamp('2026-06-30')
OUT = {}

# 1) 빨간 글자
wb = openpyxl.load_workbook(r"C:\Users\NT751\Desktop\천영진\리서치팀문서\위해요소별_중요인자_데이터출처.xlsx", rich_text=True)
ws = wb.worksheets[0]
hdr = [c.value for c in ws[1]]
print('header:', hdr)
for rn in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
    reds = []
    for ci in range(1, ws.max_column + 1):
        cell = ws.cell(rn, ci); v = cell.value
        if v is None: continue
        if hasattr(v, '__iter__') and not isinstance(v, str):
            for part in v:
                if hasattr(part, 'font') and part.font is not None and part.font.color is not None and getattr(part.font.color, 'rgb', None) in ('FFFF0000', 'FF0000'):
                    reds.append((ci, part.text))
        else:
            col = cell.font.color.rgb if cell.font and cell.font.color is not None else None
            if col in ('FFFF0000',):
                reds.append((ci, str(v)))
    print(f'r{rn} 데이터명={ws.cell(rn, 2).value!r} red={reds}')
    OUT[f'red_r{rn}'] = reds

# 2) 농업기상 관측소 유형
nst = pd.read_parquet(R + r"\raw_nas_agri_weather_obs_station.parquet")
print('\nNAS 관측소표 컬럼:', list(nst.columns))
print(nst.head(3).T.to_string())
nas = pd.read_parquet(R + r"\raw_nas_agri_weather_daily_info.parquet")
print('\nNAS 일자료 컬럼:', list(nas.columns))
nas['d'] = pd.to_datetime(nas['meas_date'])
ns = nas[(nas.d >= S0) & (nas.d <= S1)].copy()
for c in ['sun_time']:
    ns[c] = ns[c].where(ns[c].astype(str).str.strip() != '')
ns['kind'] = ns['obsr_spot_code'].astype(str).str[6]
print('코드 예:', ns['obsr_spot_code'].drop_duplicates().head(8).tolist())
kt = ns.groupby('kind').agg(stations=('obsr_spot_code', 'nunique'), rows=('d', 'size'),
                            temp_n=('temp', lambda s: int(s.notna().sum())), sun_time_n=('sun_time', lambda s: int(s.notna().sum())),
                            soil_wt_n=('soil_wt', lambda s: int(s.notna().sum())), soil_temp_n=('soil_temp', lambda s: int(s.notna().sum())), gr_temp_n=('gr_temp', lambda s: int(s.notna().sum())))
print('\n연구기간 유형별 기록 건수:'); print(kt.to_string())
OUT['nas_kind_counts_study'] = kt.reset_index().to_dict('records')
nall = nas.copy(); nall['sun_time'] = nall['sun_time'].where(nall['sun_time'].astype(str).str.strip() != ''); nall['kind'] = nall['obsr_spot_code'].astype(str).str[6]
ka = nall.groupby('kind').agg(sun_time_n=('sun_time', lambda s: int(s.notna().sum())), soil_wt_n=('soil_wt', lambda s: int(s.notna().sum())), soil_temp_n=('soil_temp', lambda s: int(s.notna().sum())), gr_temp_n=('gr_temp', lambda s: int(s.notna().sum())))
print('\nDB 전체 기간 유형별 기록 건수:'); print(ka.to_string())
OUT['nas_kind_counts_all'] = ka.reset_index().to_dict('records')
# 관측소표에 유형 비슷한 컬럼
for c in nst.columns:
    if nst[c].dtype == object and nst[c].nunique() < 30:
        print('관측소표 범주 컬럼 후보:', c, nst[c].value_counts().head(10).to_dict())

# 4) ASOS 일자료 컬럼
asos_cols = pd.read_parquet(R + r"\raw_kma_asos_daily_info.parquet").columns.tolist()
print('\nASOS 일자료 컬럼:', asos_cols)
ast = pd.read_parquet(R + r"\raw_kma_ground_obs_station_info.parquet")
print('ASOS 관측소표 컬럼:', list(ast.columns))

# 3) 국내 시료 '행 없음' 성격
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'mnfctur_de', 'lat', 'lon'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'lat', 'lon'])
b = body.merge(db.rename(columns={'lat': 'lat_db', 'lon': 'lon_db'}), on='lims_data_key', how='left')
b['lat_use'] = b['lat_db'].fillna(b['lat'].astype(float)); b['lon_use'] = b['lon_db'].fillna(b['lon'].astype(float))
b['base_date'] = pd.to_datetime(b['mnfctur_de']).dt.normalize()
dom = b[b.src_route == 'domestic'].copy()
pts = dom[['lat_use', 'lon_use']].drop_duplicates().reset_index(drop=True)
def nearest(st):
    P = np.radians(pts.to_numpy(float)); Sx = np.radians(st[['lat', 'lon']].to_numpy(float))
    ids = np.empty(len(P), dtype=object)
    for i in range(0, len(P), 1000):
        p = P[i:i + 1000]
        aa = np.sin((p[:, None, 0] - Sx[None, :, 0]) / 2) ** 2 + np.cos(p[:, None, 0]) * np.cos(Sx[None, :, 0]) * np.sin((p[:, None, 1] - Sx[None, :, 1]) / 2) ** 2
        ids[i:i + 1000] = st['sid'].to_numpy()[(2 * 6371 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))).argmin(1)]
    return pts.assign(sid=ids)
def norow(name, st, daily, did, dcol):
    dd = daily[[did, dcol]].rename(columns={did: 'sid', dcol: 'base_date'}).copy()
    dd['base_date'] = pd.to_datetime(dd['base_date']).dt.normalize()
    dd = dd.drop_duplicates()
    s = st[st['sid'].isin(set(dd['sid']))]
    x = dom.merge(nearest(s), on=['lat_use', 'lon_use'], how='left')
    x = x.merge(dd.assign(_row=1), on=['sid', 'base_date'], how='left')
    span = dd.groupby('sid')['base_date'].agg(first='min', last='max').reset_index()
    x = x.merge(span, on='sid', how='left')
    nr = x['_row'].isna()
    before = nr & (x['base_date'] < x['first']); after = nr & (x['base_date'] > x['last']); gap = nr & ~before & ~after
    res = {'no_row_pct': round(float(nr.mean() * 100), 2), 'before_first_record_pct': round(float(before.mean() * 100), 2),
           'after_last_record_pct': round(float(after.mean() * 100), 2), 'gap_within_span_pct': round(float(gap.mean() * 100), 2)}
    # 운영 중 빠진 날: 그 관측소가 그 날짜 전후로 기록이 끊긴 연속 일수
    g = x[gap][['sid', 'base_date']].drop_duplicates()
    yrs = x[nr].groupby(x['base_date'].dt.year).size()
    res['no_row_by_year_rows'] = {int(k): int(v) for k, v in yrs.items()}
    # 해당 관측소들의 DB 첫 기록일 예
    fr = x[before].groupby('sid')['first'].first().sort_values(ascending=False).head(5)
    res['before_examples'] = {str(k): str(v.date()) for k, v in fr.items()}
    # 빠진 날 예: 전체 관측소-일 기준 연구기간 누락 일수
    print(name, res)
    OUT[f'norow_{name}'] = res
    return x
sA = ast.copy(); sA['chg_dt'] = pd.to_datetime(sA['chg_dt'], errors='coerce'); sA['reg_dt'] = pd.to_datetime(sA['reg_dt'], errors='coerce')
sA = sA.sort_values(['chg_dt', 'reg_dt'], ascending=False, na_position='last').drop_duplicates('stn_id')[['stn_id', 'lat', 'lon']].rename(columns={'stn_id': 'sid'}).astype({'lat': float, 'lon': float})
asos = pd.read_parquet(R + r"\raw_kma_asos_daily_info.parquet", columns=['stn_id', 'meas_date'])
xa = norow('ASOS', sA, asos, 'stn_id', 'meas_date')
# ASOS 운영 중 빠진 날이 몰린 기간
g = xa[xa['_row'].isna() & (xa['base_date'] >= xa['first']) & (xa['base_date'] <= xa['last'])]
print('ASOS 운영 중 빠진 날 상위 관측소-기간:', g.groupby(['sid', g['base_date'].dt.to_period('M')]).size().sort_values(ascending=False).head(10).to_dict())
# DB 전체 ASOS 관측소-일 누락: 연도별로 관측소 수 × 일수 대비 행 수
asos['d'] = pd.to_datetime(asos['meas_date'])
yr = asos[(asos.d >= S0) & (asos.d <= S1)].groupby(asos['d'].dt.year).agg(rows=('stn_id', 'size'), stations=('stn_id', 'nunique'), days=('d', 'nunique'))
yr['fill_pct'] = (yr['rows'] / (yr['stations'] * yr['days']) * 100).round(1)
print('ASOS 연도별 행/(관측소×일):'); print(yr.to_string())
OUT['asos_year_fill'] = yr.reset_index().to_dict('records')
nst2 = nst.copy(); nst2['chg_dt'] = pd.to_datetime(nst2['chg_dt'], errors='coerce')
sN = nst2.sort_values('chg_dt', ascending=False).drop_duplicates('obsr_spot_code')[['obsr_spot_code', 'instl_la', 'instl_lo']].rename(columns={'obsr_spot_code': 'sid', 'instl_la': 'lat', 'instl_lo': 'lon'}).astype({'lat': float, 'lon': float})
xn = norow('NAS', sN, nas[['obsr_spot_code', 'meas_date']], 'obsr_spot_code', 'meas_date')
kst = pd.read_parquet(R + r"\raw_keco_airkorea_obs_station.parquet")
kst['chg_dt'] = pd.to_datetime(kst['chg_dt'], errors='coerce'); kst['u'] = (kst['use_yn'].astype(str) == '1').astype(int)
sK = kst.sort_values(['u', 'chg_dt'], ascending=False).drop_duplicates('station_nm')[['station_nm', 'dm_x', 'dm_y']].rename(columns={'station_nm': 'sid', 'dm_x': 'lat', 'dm_y': 'lon'}).astype({'lat': float, 'lon': float})
air = pd.read_parquet(R + r"\raw_keco_airkorea_fine_dust_info.parquet", columns=['station_nm', 'msur_dt'])
xk = norow('AIR', sK, air, 'station_nm', 'msur_dt')
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\v3_check5.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('saved')
