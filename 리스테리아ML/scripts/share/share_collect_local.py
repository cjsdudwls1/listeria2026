# -*- coding: utf-8 -*-
"""[대체 경로] DB 접속이 안 될 때: 2026-09-14 에 같은 DB에서 SELECT 로 받아 둔 추출본(data/collected, raw/)으로
share_collect_db.py 와 같은 파일 구조의 공유용/데이터/DB수집/ 을 만든다. 값은 그대로, 컬럼만 수집 범위로 좁힘.
없는 표: soil_ph_info(전체 행), water_pollution, food_moisture_info / 없는 컬럼: m_org_lims_data_info.prdlst_nm"""
import sys, os, json, datetime as dt
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = os.path.join(C, 'raw'); T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
OUT = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\공유용\데이터\DB수집"; os.makedirs(OUT, exist_ok=True)
STAMP = '2026-09-14 추출본(' + dt.datetime.now().strftime('%Y-%m-%d %H:%M') + ' 정리)'
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key'])
KEYSET = set(body['lims_data_key'].astype(str))
man = []
def save(name, df, table, where, note=''):
    df.to_parquet(os.path.join(OUT, name + '.parquet'), index=False)
    man.append({'파일': name + '.parquet', 'DB 표': table, '행 수': len(df), '열 수': df.shape[1], '추출 조건': where, '비고': note, '추출 시각': STAMP}); print(name, df.shape, flush=True)
hp = pd.read_parquet(os.path.join(C, '00_body_db.parquet')); hp['lims_data_key'] = hp['lims_data_key'].astype(str)
n_all = len(hp); hp = hp[hp['lims_data_key'].isin(KEYSET)][['lims_data_key', 'judge_result', 'date_basis', 'mnfctur_de_or_fallback', 'geo_route', 'addr_source', 'geo_match_level', 'legal_dong_code', 'lat', 'lon', 'prdlst_code', 'prdlst_nm', 'mapped_prdlst_code',
                                                             'statistic_prdlst_nm_1', 'statistic_prdlst_nm_2', 'statistic_prdlst_nm_3', 'statistic_prdlst_nm_4', 'prdlst_mnfctur_sttemnt_no', 'bsn_lcns_ledg_no', 'cntc_sys_key2']].reset_index(drop=True)
save('hazard_preprocessed_data', hp, 'ready_data.hazard_preprocessed_data', f"hazard_label='리스테리아' {n_all:,}행 중 우리 시료 키 {len(hp):,}행", '연결 키·좌표·전처리 품목분류(참고). geocode_status 열은 추출본에 없음')
mo = pd.read_parquet(os.path.join(C, '02a_m_org.parquet')); mo['lims_data_key'] = mo['lims_data_key'].astype(str)
n_all = len(mo); mo = mo[mo['lims_data_key'].isin(KEYSET)][['lims_data_key', 'upper_prdlst_code', 'upper_prdlst_nm', 'cnvync_sttus_tp', 'cnvync_sttus_tp_unit_code', 'cnvync_mth_code', 'splore_sttus_code', 'distb_de', 'valid_tmlmt']].reset_index(drop=True)
save('m_org_lims_data_info', mo, 'ba_data.m_org_lims_data_info', f"리스테리아 시료 키 일치 {n_all:,}행 중 우리 시료 키 {len(mo):,}행", 'prdlst_nm(품목 명)은 추출본에 없어 빠짐 → DB 복구 후 재수집')
so = pd.read_parquet(os.path.join(C, '02e_s_org.parquet'))
ex = pd.read_parquet(os.path.join(T, 'brief_s_org_extra.parquet'), columns=['rcept_no', 'lims_seq', 'prdlst_nm'])
so = so.merge(ex, on=['rcept_no', 'lims_seq'], how='left')[['rcept_no', 'lims_seq', 'label', 'prdlst_nm', 'prduct_dvs_cd', 'dclr_prduct_dvs', 'hsk_no', 'distb_bgn_de', 'distb_end_de']]
save('s_org_lims_data_info', so, 'ba_data.s_org_lims_data_info', 'rcept_no = 우리 시료 수입접수번호(cntc_sys_key2)', f"접수번호당 시험항목별 여러 행(label 열로 구분). prdlst_nm 은 같은 DB 표의 별도 추출본(2026-09-15)을 (rcept_no, lims_seq)로 붙인 열이라 빈 값 {int(so['prdlst_nm'].isna().sum())}행은 추출본에 없던 행")
lg = pd.read_parquet(os.path.join(C, '02b_ledg.parquet'))[['prdlst_report_no', 'prdlst_report_ledg_no', 'prdlst_cd', 'prdlst_nm', 'del_yn', 'last_updt_dtm', 'prms_dt', 'end_dt', 'mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'pog_daycnt', 'qlity_mntnc_tmlmt_daycnt', 'frmlc_mtrqlt']]
save('t_prdlst_mnf_ledg', lg, 'ba_data.t_prdlst_mnf_ledg', 'prdlst_report_no = 우리 시료 품목제조번호 원문 또는 숫자만', '같은 번호 여러 행(삭제 여부·수정일시 열로 구분)')
cl = pd.read_parquet(os.path.join(T, 'brief_t_prdlst_cl.parquet')).merge(pd.read_parquet(os.path.join(C, '02d_cl.parquet'))[['prdlst_cd', 'htrk_prdlst_cd', 'fdgrp_yn', 'artcl_gun_code', 'food_hf_ls_cl_cd']], on='prdlst_cd', how='left')
cl = cl[['prdlst_cd', 'kor_nm', 'lv', 'hrnk_prdlst_cd', 'htrk_prdlst_cd', 'fdgrp_yn', 'artcl_gun_code', 'food_hf_ls_cl_cd', 'dfn', 'use_yn']]
save('t_prdlst_cl', cl, 'ba_data.t_prdlst_cl', '전체')
save('t_aw_cl', pd.read_parquet(os.path.join(C, '80_t_aw_cl.parquet')), 'ba_data.t_aw_cl', '전체', '리서치팀 파일 적재본(채운 값 섞임, 결과보고 3-6)')
cm = pd.read_parquet(os.path.join(C, '70_codes.parquet'))
save('t_cm_cmmn_code', cm, 'ba_data.t_cm_cmmn_code', 'cmmn_cl_code IN (' + ', '.join(sorted(cm['cmmn_cl_code'].astype(str).unique())) + ')', '운반방법(IM08)·시료상태(IM11)·온도단위(IM40)·검사목적(IM02)·검사종류(IM03) 코드 이름')
ASOS = ['stn_id', 'stn_nm', 'meas_date', 'avg_ta', 'max_ta', 'min_ta', 'sum_rn', 'avg_ws', 'max_ws', 'max_ins_ws', 'max_ws_wd', 'max_ins_ws_wd', 'max_ws1', 'sum_sshr', 'sum_gsr', 'avg_cm5_te', 'avg_cm10_te', 'avg_cm20_tes', 'avg_cm30_te', 'avg_ts', 'min_tg']
save('kma_asos_daily_info', pd.read_parquet(os.path.join(R, 'raw_kma_asos_daily_info.parquet'), columns=ASOS), 'api_data_info.kma_asos_daily_info', '전체(수집 범위 컬럼)')
save('kma_ground_obs_station_info', pd.read_parquet(os.path.join(R, 'raw_kma_ground_obs_station_info.parquet')), 'api_data_info.kma_ground_obs_station_info', '전체', '한 관측소에 이력 행 여러 개')
NAS = ['obsr_spot_code', 'stn_name', 'meas_date', 'temp', 'hghst_artmp', 'lowst_artmp', 'hum', 'rn', 'wind', 'max_wind', 'widdir', 'sun_time', 'srqty', 'soil_temp', 'gr_temp', 'soil_wt']
save('nas_agri_weather_daily_info', pd.read_parquet(os.path.join(R, 'raw_nas_agri_weather_daily_info.parquet'), columns=NAS), 'api_data_info.nas_agri_weather_daily_info', '전체(수집 범위 컬럼)')
save('nas_agri_weather_obs_station', pd.read_parquet(os.path.join(R, 'raw_nas_agri_weather_obs_station.parquet')), 'api_data_info.nas_agri_weather_obs_station', '전체', "시험용 관측소 '동방로거테스트' 포함")
save('keco_airkorea_fine_dust_info', pd.read_parquet(os.path.join(R, 'raw_keco_airkorea_fine_dust_info.parquet'), columns=['station_nm', 'msur_dt', 'no2_value', 'co_value', 'so2_value', 'pm10_value']), 'api_data_info.keco_airkorea_fine_dust_info', '전체(수집 범위 컬럼)')
save('keco_airkorea_obs_station', pd.read_parquet(os.path.join(R, 'raw_keco_airkorea_obs_station.parquet')), 'api_data_info.keco_airkorea_obs_station', '전체', 'dm_x=위도, dm_y=경도(정의서의 TM 표기는 오기)')
save('kma_weather_predict_info', pd.read_parquet(os.path.join(R, 'raw_kma_weather_predict_info.parquet')), 'api_data_info.kma_weather_predict_info', '전체', '2026-07-01 발표 12일 예보뿐(시료 기준일과 겹치는 날 없음)')
for name, table, why in [('soil_ph_info', 'api_data_info.soil_ph_info', '추출본 없음(동·연도별 필지 수만 있음) → DB 복구 후 재수집'), ('water_pollution', 'api_data_info.water_pollution', '추출본 없음 → DB 복구 후 재수집'), ('food_moisture_info', 'ba_data_info.food_moisture_info', '추출본 없음 → DB 복구 후 재수집')]:
    man.append({'파일': '(없음)', 'DB 표': table, '행 수': 0, '열 수': 0, '추출 조건': '-', '비고': why, '추출 시각': STAMP})
cc = pd.read_csv(os.path.join(T, 'brief_columns.csv')).rename(columns={'type': 'db_type', 'ccomment': 'db_comment'})
cc = cc[cc['tbl'].isin(['hazard_preprocessed_data', 'm_org_lims_data_info', 's_org_lims_data_info', 't_prdlst_mnf_ledg', 't_prdlst_cl', 't_aw_cl', 'food_moisture_info', 't_cm_cmmn_code', 'kma_asos_daily_info', 'kma_ground_obs_station_info', 'nas_agri_weather_daily_info', 'nas_agri_weather_obs_station', 'keco_airkorea_fine_dust_info', 'keco_airkorea_obs_station', 'soil_ph_info', 'water_pollution', 'kma_weather_predict_info']) & cc['schema'].isin(['ready_data', 'ba_data', 'ba_data_info', 'api_data_info'])]
cc = cc[~((cc['tbl'] == 't_prdlst_cl') & (cc['schema'] == 'ba_data_info')) & ~((cc['tbl'] == 's_org_lims_data_info') & (cc['schema'] == 'ba_data_info'))]
cc['table_comment'] = ''; cc.to_csv(os.path.join(OUT, '_DB컬럼설명.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(man).to_csv(os.path.join(OUT, '_수집목록.csv'), index=False, encoding='utf-8-sig')
print('DONE', STAMP)
