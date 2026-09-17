# -*- coding: utf-8 -*-
"""[작업 1-A] 식정원 DB(TOXIN_MODEL_PRED) 수집 → 공유용/데이터/DB수집/<표>.parquet
- SELECT 만 사용(읽기 전용 접속). 값은 DB 그대로(빈 값 채우기·단위 변환·필터·집계 없음). 타입만 numeric→float64, date→datetime 으로 받음
- 수집 범위: 20260901.테이블정의서 노랑·초록 컬럼 + 결과보고 '1_수집가능여부' 1-1 컬럼(해외 표 global_* 와 우리 시료 값 없는 컬럼 제외) + 연결 키
- 시료 키가 있는 표(전처리 시료표·식약처 검사 원본·수입 검사 원본·품목제조대장)는 우리 시료(listeria_body.parquet)에 해당하는 행만, 나머지 표는 전체 행
"""
import sys, os, time, json, datetime as dt
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from decimal import Decimal
from db_utils import connect

OUT = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\공유용\데이터\DB수집"
os.makedirs(OUT, exist_ok=True)
BODY = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet"
STAMP = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
conn = connect()

def fetch(sql, params=None, label=''):
    """서버측 커서로 청크 fetch (트랜잭션 블록 안에서만 DECLARE 가능)"""
    t0 = time.time(); chunks = []; n = 0
    with conn.transaction():
        with conn.cursor(name=f"c_{int(t0*1000)}") as cur:
            cur.itersize = 100_000
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            while True:
                rows = cur.fetchmany(100_000)
                if not rows: break
                chunks.append(pd.DataFrame(rows, columns=cols)); n += len(rows)
                print(f"   [{label}] {n:,}행 ({time.time()-t0:.0f}초)", flush=True)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    # 타입 정리(값 변경 없음): Decimal→float64, date→datetime64[ms], 그 외 object→string
    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            nn = s.dropna()
            if len(nn) == 0: df[c] = s.astype('string'); continue
            v = nn.iloc[0]
            if isinstance(v, Decimal): df[c] = pd.to_numeric(s, errors='raise').astype('float64')
            elif isinstance(v, dt.datetime): df[c] = pd.to_datetime(s, errors='coerce')
            elif isinstance(v, dt.date): df[c] = pd.to_datetime(s, errors='coerce').astype('datetime64[ms]')
            elif isinstance(v, (int, float, bool)): df[c] = s
            else: df[c] = s.astype('string')
    print(f"  [{label}] 완료 {len(df):,}행 × {df.shape[1]}열 {time.time()-t0:.0f}초", flush=True)
    return df

def q(sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])

manifest = []
def save(name, df, cond, note=''):
    p = os.path.join(OUT, f"{name}.parquet")
    df.to_parquet(p, index=False)
    manifest.append({'파일': f"{name}.parquet", 'DB 표': cond['table'], '행 수': len(df), '열 수': df.shape[1], '추출 조건': cond['where'], '비고': note, '추출 시각': STAMP})
    print(f"  → 저장 {name}.parquet {len(df):,}행 × {df.shape[1]}열", flush=True)

# ---------- 0. DB 컬럼 설명(모든 대상 표) ----------
TABLES = [('ready_data', 'hazard_preprocessed_data'), ('ba_data', 'm_org_lims_data_info'), ('ba_data', 's_org_lims_data_info'), ('ba_data', 't_prdlst_mnf_ledg'), ('ba_data', 't_prdlst_cl'),
          ('ba_data', 't_aw_cl'), ('ba_data_info', 'food_moisture_info'), ('ba_data', 't_cm_cmmn_code'),
          ('api_data_info', 'kma_asos_daily_info'), ('api_data_info', 'kma_ground_obs_station_info'), ('api_data_info', 'nas_agri_weather_daily_info'), ('api_data_info', 'nas_agri_weather_obs_station'),
          ('api_data_info', 'keco_airkorea_fine_dust_info'), ('api_data_info', 'keco_airkorea_obs_station'), ('api_data_info', 'soil_ph_info'), ('api_data_info', 'water_pollution'), ('api_data_info', 'kma_weather_predict_info')]
cc = q("""SELECT n.nspname AS schema, c.relname AS tbl, a.attnum, a.attname AS col, format_type(a.atttypid, a.atttypmod) AS db_type,
                 col_description(c.oid, a.attnum) AS db_comment, obj_description(c.oid, 'pg_class') AS table_comment
          FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace
          WHERE a.attnum > 0 AND NOT a.attisdropped AND (n.nspname, c.relname) IN (%s)""" % ','.join(['(%s,%s)'] * len(TABLES)), [x for t in TABLES for x in t])
cc.to_csv(os.path.join(OUT, '_DB컬럼설명.csv'), index=False, encoding='utf-8-sig')
print('DB 컬럼 설명', len(cc), '칸')

# ---------- 1. 우리 시료 키 ----------
body = pd.read_parquet(BODY, columns=['lims_data_key'])
KEYS = body['lims_data_key'].astype(str).tolist(); KEYSET = set(KEYS)
print('우리 시료 키', len(KEYS))

# ---------- 2. 전처리 시료표 (리스테리아 전체 행 → 우리 키만) ----------
hp = fetch("""SELECT lims_data_key, judge_result, date_basis, mnfctur_de_or_fallback, geo_route, addr_source, geocode_status, geo_match_level, legal_dong_code, lat, lon,
                     prdlst_code, prdlst_nm, mapped_prdlst_code, statistic_prdlst_nm_1, statistic_prdlst_nm_2, statistic_prdlst_nm_3, statistic_prdlst_nm_4,
                     prdlst_mnfctur_sttemnt_no, bsn_lcns_ledg_no, cntc_sys_key2
              FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아'""", label='hazard_preprocessed_data')
n_all = len(hp); hp = hp[hp['lims_data_key'].astype(str).isin(KEYSET)].reset_index(drop=True)
save('hazard_preprocessed_data', hp, {'table': 'ready_data.hazard_preprocessed_data', 'where': f"hazard_label='리스테리아' {n_all:,}행 중 우리 시료 키 {len(hp):,}행"}, '연결 키·좌표·전처리 품목분류(참고)')

# ---------- 3. 식약처 검사 원본 ----------
mo = fetch("""SELECT m.lims_data_key, m.prdlst_nm, m.upper_prdlst_code, m.upper_prdlst_nm, m.cnvync_sttus_tp, m.cnvync_sttus_tp_unit_code, m.cnvync_mth_code, m.splore_sttus_code, m.distb_de, m.valid_tmlmt
              FROM ba_data.m_org_lims_data_info m JOIN ready_data.hazard_preprocessed_data h ON h.lims_data_key = m.lims_data_key WHERE h.hazard_label = '리스테리아'""", label='m_org')
n_all = len(mo); mo = mo[mo['lims_data_key'].astype(str).isin(KEYSET)].reset_index(drop=True)
save('m_org_lims_data_info', mo, {'table': 'ba_data.m_org_lims_data_info', 'where': f"리스테리아 시료 키 일치 {n_all:,}행 중 우리 시료 키 {len(mo):,}행"})

# ---------- 4. 수입 검사 원본 ----------
rc = hp['cntc_sys_key2'].dropna().astype(str).unique().tolist()
so = fetch("""SELECT rcept_no, lims_seq, label, prdlst_nm, prduct_dvs_cd, dclr_prduct_dvs, hsk_no, distb_bgn_de, distb_end_de
              FROM ba_data.s_org_lims_data_info WHERE rcept_no = ANY(%s)""", (rc,), label='s_org')
save('s_org_lims_data_info', so, {'table': 'ba_data.s_org_lims_data_info', 'where': f"rcept_no = 우리 시료 수입접수번호(cntc_sys_key2) {len(rc):,}개"}, '접수번호당 시험항목별 여러 행(label 열로 구분)')

# ---------- 5. 품목제조대장 ----------
nums = hp['prdlst_mnfctur_sttemnt_no'].dropna().astype(str)
digits = nums.str.replace(r'\D', '', regex=True)
cand = sorted(set(nums.tolist()) | set(d for d in digits.tolist() if d))
lg = fetch("""SELECT prdlst_report_no, prdlst_report_ledg_no, prdlst_cd, prdlst_nm, del_yn, last_updt_dtm, prms_dt, end_dt,
                     mnf_cnd_cn, prsrv_distb_stdr_cn, pog_daycnt, qlity_mntnc_tmlmt_daycnt, frmlc_mtrqlt
              FROM ba_data.t_prdlst_mnf_ledg WHERE prdlst_report_no = ANY(%s)""", (cand,), label='ledg')
save('t_prdlst_mnf_ledg', lg, {'table': 'ba_data.t_prdlst_mnf_ledg', 'where': f"prdlst_report_no = 우리 시료 품목제조번호 원문 또는 숫자만 {len(cand):,}개"}, '같은 번호 여러 행(삭제 여부·수정일시 열로 구분)')

# ---------- 6. 식품공전 분류 / 수분활성도 표 / 수분 문헌값 / 공통코드 ----------
cl = fetch("SELECT prdlst_cd, kor_nm, lv, hrnk_prdlst_cd, htrk_prdlst_cd, fdgrp_yn, artcl_gun_code, food_hf_ls_cl_cd, dfn, use_yn FROM ba_data.t_prdlst_cl", label='t_prdlst_cl')
save('t_prdlst_cl', cl, {'table': 'ba_data.t_prdlst_cl', 'where': '전체'})
aw = fetch("SELECT * FROM ba_data.t_aw_cl", label='t_aw_cl')
save('t_aw_cl', aw, {'table': 'ba_data.t_aw_cl', 'where': '전체'}, '리서치팀 파일 적재본(채운 값 섞임, 결과보고 3-6)')
fm = fetch("SELECT * FROM ba_data_info.food_moisture_info", label='food_moisture_info')
save('food_moisture_info', fm, {'table': 'ba_data_info.food_moisture_info', 'where': '전체'})
print('  food_moisture_info 품목명 중복:', int(fm['prdlst_nm'].duplicated().sum()) if 'prdlst_nm' in fm.columns else 'prdlst_nm 없음', list(fm.columns))
cm = fetch("SELECT cmmn_cl_code, cmmn_code, cmmn_code_nm, use_yn FROM ba_data.t_cm_cmmn_code WHERE cmmn_cl_code IN ('IM08','IM11','IM40','IM02','IM03')", label='codes')
save('t_cm_cmmn_code', cm, {'table': 'ba_data.t_cm_cmmn_code', 'where': "cmmn_cl_code IN ('IM08','IM11','IM40','IM02','IM03')"}, '운반방법·시료상태·온도단위·검사목적·검사종류 코드 이름')

# ---------- 7. 기상·대기·토양·수질 (전체 행) ----------
ASOS = ['stn_id', 'stn_nm', 'meas_date', 'avg_ta', 'max_ta', 'min_ta', 'sum_rn', 'avg_ws', 'max_ws', 'max_ins_ws', 'max_ws_wd', 'max_ins_ws_wd', 'max_ws1', 'sum_sshr', 'sum_gsr', 'avg_cm5_te', 'avg_cm10_te', 'avg_cm20_tes', 'avg_cm30_te', 'avg_ts', 'min_tg']
save('kma_asos_daily_info', fetch(f"SELECT {', '.join(ASOS)} FROM api_data_info.kma_asos_daily_info", label='asos'), {'table': 'api_data_info.kma_asos_daily_info', 'where': '전체(수집 범위 컬럼)'})
save('kma_ground_obs_station_info', fetch("SELECT * FROM api_data_info.kma_ground_obs_station_info", label='asos_st'), {'table': 'api_data_info.kma_ground_obs_station_info', 'where': '전체'}, '한 관측소에 이력 행 여러 개')
NAS = ['obsr_spot_code', 'stn_name', 'meas_date', 'temp', 'hghst_artmp', 'lowst_artmp', 'hum', 'rn', 'wind', 'max_wind', 'widdir', 'sun_time', 'srqty', 'soil_temp', 'gr_temp', 'soil_wt']
save('nas_agri_weather_daily_info', fetch(f"SELECT {', '.join(NAS)} FROM api_data_info.nas_agri_weather_daily_info", label='nas'), {'table': 'api_data_info.nas_agri_weather_daily_info', 'where': '전체(수집 범위 컬럼)'})
save('nas_agri_weather_obs_station', fetch("SELECT * FROM api_data_info.nas_agri_weather_obs_station", label='nas_st'), {'table': 'api_data_info.nas_agri_weather_obs_station', 'where': '전체'}, "시험용 관측소 '동방로거테스트' 포함")
save('keco_airkorea_fine_dust_info', fetch("SELECT station_nm, msur_dt, no2_value, co_value, so2_value, pm10_value FROM api_data_info.keco_airkorea_fine_dust_info", label='air'), {'table': 'api_data_info.keco_airkorea_fine_dust_info', 'where': '전체(수집 범위 컬럼)'})
save('keco_airkorea_obs_station', fetch("SELECT * FROM api_data_info.keco_airkorea_obs_station", label='air_st'), {'table': 'api_data_info.keco_airkorea_obs_station', 'where': '전체'}, 'dm_x=위도, dm_y=경도(정의서의 TM 표기는 오기)')
save('soil_ph_info', fetch("SELECT pnu_cd, pnu_nm, legal_dong_code, legal_dong_nm, any_year, exam_day, acid, vldpha, om FROM api_data_info.soil_ph_info", label='soil'), {'table': 'api_data_info.soil_ph_info', 'where': '전체(수집 범위 컬럼)'}, '필지 단위. 2023년 이후 일부 행은 pnu_cd에 법정동코드')
save('water_pollution', fetch("SELECT bplc_nm, lcpmt_ymd, sals_stts_nm, clsbiz_ymd, road_nm_addr, lotno_addr, crd_info_x, crd_info_y FROM api_data_info.water_pollution", label='water'), {'table': 'api_data_info.water_pollution', 'where': '전체(수집 범위 컬럼)'}, '수질오염원 설치시설 목록(수질 측정값 아님). 좌표계 미확인')
save('kma_weather_predict_info', fetch("SELECT * FROM api_data_info.kma_weather_predict_info", label='predict'), {'table': 'api_data_info.kma_weather_predict_info', 'where': '전체'}, '2026-07-01 발표 12일 예보뿐(시료 기준일과 겹치는 날 없음)')
conn.close()
pd.DataFrame(manifest).to_csv(os.path.join(OUT, '_수집목록.csv'), index=False, encoding='utf-8-sig')
print('DONE', STAMP)
