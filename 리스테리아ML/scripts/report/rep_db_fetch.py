# -*- coding: utf-8 -*-
"""보고서용 원자료 추가 수집 (SELECT 만, 값 변경 없음). 결과: data/collected/raw/"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import pandas as pd
from db_utils import connect
R = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected\raw"
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
conn = connect()

def q(sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])

def fetch(sql, params=None, name='x'):
    t0 = time.time()
    with conn.transaction():
        with conn.cursor(name=f"c_{name}") as cur:
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            chunks = []
            while True:
                rows = cur.fetchmany(200_000)
                if not rows: break
                chunks.append(pd.DataFrame(rows, columns=cols))
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    print(f"{name}: {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)
    return df

def save_raw(df, name, types):
    for c in df.columns:
        t = types.get(c)
        if t in ('numeric',):
            df[c] = pd.to_numeric(df[c]).astype('float64')
        elif t == 'date':
            df[c] = pd.to_datetime(df[c]).astype('datetime64[ms]')
    df.to_parquet(os.path.join(R, name), index=False)

def types_of(schema, table):
    d = q("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema=%s AND table_name=%s", (schema, table))
    return dict(zip(d.column_name, d.data_type))

# 1) 컬럼 코멘트(한글 설명) — 정의서에 없는 테이블 보완용
com = q("""
SELECT n.nspname AS schema, c.relname AS table, a.attname AS column, col_description(a.attrelid, a.attnum) AS comment,
       format_type(a.atttypid, a.atttypmod) AS type
FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE a.attnum > 0 AND NOT a.attisdropped
  AND (n.nspname, c.relname) IN (('api_data_info','kma_asos_daily_info'),('api_data_info','nas_agri_weather_daily_info'),('api_data_info','keco_airkorea_fine_dust_info'),
       ('api_data_info','soil_ph_info'),('api_data_info','global_weather_info'),('api_data_info','global_factory_weather'),('api_data_info','global_soil_ph'),
       ('ba_data','m_org_lims_data_info'),('ba_data','t_prdlst_mnf_ledg'),('ba_data','t_bsn_lcns_ledg'),('ba_data','t_prdlst_cl'),('ba_data','s_org_lims_data_info'),
       ('ba_data','s_co_ifd_ovsmnfst_regstr_m'),('ba_data','t_aw_cl'),('ready_data','hazard_preprocessed_data'),('api_data_info','kma_ground_obs_station_info'),
       ('api_data_info','nas_agri_weather_obs_station'),('api_data_info','keco_airkorea_obs_station'),('api_data_info','kma_weather_predict_info'))
ORDER BY 1,2,a.attnum""")
com.to_csv(os.path.join(T, 'rep_col_comments.csv'), index=False, encoding='utf-8-sig')
print('comments', len(com))

# 2) 토양 pH 원자료 상태 (필지 단위 테이블 — 건수만 센다)
soil_types = q("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='api_data_info' AND table_name='soil_ph_info' ORDER BY ordinal_position")
print(soil_types.to_string(index=False))
soil_y = q("""SELECT any_year, COUNT(*) AS n, COUNT(acid) AS n_acid, COUNT(vldpha) AS n_vldpha, COUNT(om) AS n_om,
                     COUNT(DISTINCT legal_dong_code) AS n_dong,
                     COUNT(*) FILTER (WHERE acid < 3) AS acid_lt3, COUNT(*) FILTER (WHERE acid > 10) AS acid_gt10,
                     MIN(acid) AS acid_min, MAX(acid) AS acid_max, MIN(vldpha) AS vldpha_min, MAX(vldpha) AS vldpha_max, MIN(om) AS om_min, MAX(om) AS om_max,
                     MIN(exam_day) AS exam_min, MAX(exam_day) AS exam_max
              FROM api_data_info.soil_ph_info GROUP BY any_year ORDER BY any_year""")
soil_y.to_csv(os.path.join(T, 'rep_soil_year.csv'), index=False, encoding='utf-8-sig')
print(soil_y.to_string(index=False))
soil_tot = q("""SELECT COUNT(*) n, COUNT(acid) n_acid, COUNT(vldpha) n_vldpha, COUNT(om) n_om, COUNT(DISTINCT legal_dong_code) n_dong,
                       COUNT(*) FILTER (WHERE acid<3) lt3, COUNT(*) FILTER (WHERE acid>10) gt10, COUNT(*) FILTER (WHERE acid>14) gt14,
                       COUNT(*) FILTER (WHERE vldpha<0) vld_neg, COUNT(*) FILTER (WHERE om<0) om_neg,
                       percentile_disc(0.5) WITHIN GROUP (ORDER BY acid) acid_p50, percentile_disc(0.5) WITHIN GROUP (ORDER BY vldpha) vld_p50, percentile_disc(0.5) WITHIN GROUP (ORDER BY om) om_p50,
                       COUNT(*) FILTER (WHERE LENGTH(legal_dong_code) <> 10) bad_len
                FROM api_data_info.soil_ph_info""")
soil_tot.to_csv(os.path.join(T, 'rep_soil_total.csv'), index=False, encoding='utf-8-sig')
print(soil_tot.to_string(index=False))
# 법정동×연도 필지 수 (집계값 아님, 존재 건수만) — 시료 연결률 계산용
soil_dy = fetch("SELECT legal_dong_code, any_year, COUNT(*) AS n_parcel, COUNT(acid) AS n_acid FROM api_data_info.soil_ph_info GROUP BY 1, 2", name='soil_dong_year_counts')
soil_dy.to_parquet(os.path.join(R, 'raw_soil_ph_info_dong_year_counts.parquet'), index=False)

# 3) 해외 기상 (매핑 테이블) 원자료 전체
t = types_of('api_data_info', 'global_weather_info')
gw = fetch("SELECT * FROM api_data_info.global_weather_info", name='global_weather_info')
save_raw(gw, 'raw_global_weather_info.parquet', t); del gw
t = types_of('api_data_info', 'global_soil_ph')
gs = fetch("SELECT * FROM api_data_info.global_soil_ph", name='global_soil_ph'); save_raw(gs, 'raw_global_soil_ph.parquet', t)

# 4) 해외 제조소 좌표 기상 — 리스테리아 수입 시료 좌표·기준일에 해당하는 행만 (unique index 조회)
t = types_of('api_data_info', 'global_factory_weather')
gf = fetch("""SELECT h.lims_data_key, h.lat AS h_lat, h.lon AS h_lon, h.mnfctur_de_or_fallback AS base_date, w.*
             FROM ready_data.hazard_preprocessed_data h
             JOIN api_data_info.global_factory_weather w ON w.lat_g = h.lat AND w.lon_g = h.lon AND w.obsr_de = h.mnfctur_de_or_fallback
             WHERE h.hazard_label = '리스테리아' AND h.geo_route = 'overseas'""", name='global_factory_weather_listeria')
t.update({'base_date': 'date'})
save_raw(gf, 'raw_global_factory_weather_listeria_sameday.parquet', t)
print(gf.drop(columns=['lims_data_key']).describe(include='all').T.head(30).to_string())
gfr = q("""SELECT MIN(obsr_de) mn, MAX(obsr_de) mx FROM api_data_info.global_factory_weather""")
print('global_factory_weather 기간', gfr.to_dict('records'))

# 5) ASOS 관측소 이력: 같은 stn_id 좌표 여러 개
hist = q("""SELECT stn_id, COUNT(*) n_rows, COUNT(DISTINCT (lat, lon)) n_coord,
                   MAX(6371*acos(LEAST(1.0, cos(radians(lat::float8))*cos(radians(lat::float8))))) AS dummy
            FROM api_data_info.kma_ground_obs_station_info GROUP BY stn_id HAVING COUNT(DISTINCT (lat, lon)) > 1 ORDER BY 2 DESC""")
print('ASOS stn_id with multiple coords:', len(hist))
conn.close()
print('DONE')
