# -*- coding: utf-8 -*-
"""브리핑용 DB 조회 (읽기 전용 SELECT, 임시표·설정 변경 없음, 1천만 행 넘는 표는 전체 스캔 안 함)"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import pandas as pd, numpy as np
from db_utils import connect
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"; C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
conn = connect()
def q(sql, params=None):
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    print(f"  ({len(df):,}행, {time.time() - t0:.1f}초)")
    return df
OUT = {}

# 1) 해외 기상·환경 표: 인덱스, 통계(null 비율·범위·고유값 수), 행 수
STATS_SQL = """SELECT attname, null_frac, n_distinct, (histogram_bounds::text::text[])[1] AS hb_first,
  (histogram_bounds::text::text[])[array_length(histogram_bounds::text::text[], 1)] AS hb_last, left(most_common_vals::text, 160) AS mcv
  FROM pg_stats WHERE schemaname = %s AND tablename = %s"""
for sc, tb in [('api_data_info', 'global_factory_weather'), ('weather_data', 'overseas_weather'), ('api_data_info', 'global_factory_fine_dust'),
               ('api_data_info', 'global_fine_dust_info'), ('ba_data', 't_global_risk_60d_weather')]:
    print(f"\n[{sc}.{tb}]")
    idx = q("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s", (sc, tb))
    for r in idx.itertuples(): print('   index:', r.indexdef)
    st = q(STATS_SQL, (sc, tb)); print(st.to_string(index=False, max_colwidth=60))
    tu = q("SELECT n_live_tup, last_analyze, last_autoanalyze FROM pg_stat_user_tables WHERE schemaname = %s AND relname = %s", (sc, tb))
    print('   ', tu.to_dict('records'))
    OUT[f'stats.{sc}.{tb}'] = {'indexes': idx['indexdef'].tolist(), 'pg_stats': st.astype(str).to_dict('records'), 'tup': tu.astype(str).to_dict('records')}

# 2) 수입 시료 좌표별 global_factory_weather 자료 기간 (좌표·날짜 인덱스가 있을 때만)
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'lat', 'lon', 'cntc_sys_key2'])
ovs = body[body.src_route == 'overseas'].merge(db, on='lims_data_key', how='left')
coords = ovs[['lat', 'lon']].dropna().drop_duplicates()
idxdefs = ' '.join(OUT['stats.api_data_info.global_factory_weather']['indexes'])
print('\n수입 시료 좌표 수:', len(coords))
if 'lat_g, lon_g, obsr_de' in idxdefs:
    sql = """SELECT c.lat, c.lon,
      (SELECT min(g.obsr_de) FROM api_data_info.global_factory_weather g WHERE g.lat_g = c.lat AND g.lon_g = c.lon) AS dmin,
      (SELECT max(g.obsr_de) FROM api_data_info.global_factory_weather g WHERE g.lat_g = c.lat AND g.lon_g = c.lon) AS dmax
      FROM unnest(%s::float8[], %s::float8[]) AS c(lat, lon)"""
    test = coords.head(20)
    print('시험 20개 좌표:'); t0 = time.time(); r20 = q(sql, (test['lat'].tolist(), test['lon'].tolist()))
    if time.time() - t0 < 20:
        rng = q(sql, (coords['lat'].tolist(), coords['lon'].tolist()))
        rng.to_parquet(T + r"\brief_gfw_coord_range.parquet", index=False)
        OUT['gfw_coord_range'] = {'coords': int(len(rng)), 'dmin_min': str(rng['dmin'].min()), 'dmin_max': str(rng['dmin'].max()),
                                  'dmax_min': str(rng['dmax'].min()), 'dmax_max': str(rng['dmax'].max()), 'no_rows': int(rng['dmin'].isna().sum()),
                                  'dmin_counts': rng['dmin'].astype(str).value_counts().head(5).to_dict(), 'dmax_counts': rng['dmax'].astype(str).value_counts().head(5).to_dict()}
        print(OUT['gfw_coord_range'])
else:
    print('좌표·날짜 인덱스 없음 → 좌표별 기간 조회 생략')

# 3) 식약처 LIMS·대장 표 전체 기준 null 비율(통계) — 우리 시료만 비는지, 표 전체가 비는지 비교용
for sc, tb, cols in [('ba_data', 'm_org_lims_data_info', ['cnvync_sttus_tp', 'cnvync_sttus_tp_unit_code', 'splore_sttus_code', 'distb_de', 'valid_tmlmt']),
                     ('ba_data_info', 'lims_rst_nw', ['transport_temp', 'sample_status_code', 'expiry_date', 'valid_days']),
                     ('ba_data', 't_prdlst_mnf_ledg', ['pog_daycnt', 'qlity_mntnc_tmlmt_daycnt', 'frmlc_mtrqlt', 'mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'haccp_appn_no']),
                     ('ba_data', 's_org_lims_data_info', ['prpos_nm', 'distb_bgn_de', 'distb_end_de', 'shipng_de', 'entrpt_de', 'wrhous_carry_de', 'packng_plc_cd']),
                     ('ba_data_info', 'lims_org_info', ['label', 'frght_cstdy_place_nm', 'distb_begin_dt', 'distb_end_dt', 'ship_port_date', 'etrypt_de']),
                     ('ba_data_info', 'distb_step_exam', ['cstdy_tp', 'frmlc_shap', 'reg_dtm']), ('ba_data_info', 'mnf_step_exam', ['mnf_procs', 'reg_dtm'])]:
    print(f"\n[통계] {sc}.{tb}")
    st = q(STATS_SQL, (sc, tb)); st = st[st.attname.isin(cols)]
    print(st.to_string(index=False, max_colwidth=80))
    OUT[f'nullfrac.{sc}.{tb}'] = st.astype(str).to_dict('records')

# 4) 식품코드표 전체(8,864행)
print('\n식품코드표')
cl = q("SELECT prdlst_cd, kor_nm, ncknm, dfn, lv, hrnk_prdlst_cd, use_yn FROM ba_data.t_prdlst_cl")
cl.to_parquet(T + r"\brief_t_prdlst_cl.parquet", index=False)

# 5) 수입 LIMS: 용도·제품명·검사의뢰일 (우리 수입 시료 접수번호만)
rc = ovs['cntc_sys_key2'].dropna().astype(str).unique().tolist()
print('\n수입 LIMS 추가 칸 (접수번호', len(rc), '개)')
so = q("""SELECT rcept_no, lims_seq, label, rcept_de, impt_prpos_cd, prpos_nm, prdlst_cd, prdlst_nm, upper_prdlst_nm, prduct_nm_kor, prduct_nm_eng,
          packng_plc_addr, inspct_reqst_de, inspct_cmptn_de FROM ba_data.s_org_lims_data_info WHERE rcept_no = ANY(%s)""", (rc,))
so.to_parquet(T + r"\brief_s_org_extra.parquet", index=False)

# 6) 해외제조업소 대장: HACCP·식품안전관리시스템 여부 (우리 수입 시료 업소만)
s02 = pd.read_parquet(C + r"\02e_s_org.parquet", columns=['ovsmnfst_cd'])
oc = s02['ovsmnfst_cd'].dropna().astype(str).unique().tolist()
print('\n해외제조업소 (', len(oc), '개)')
co = q("""SELECT ovsmnfst_cd, ntncd, fsms_yn, fsms_crtfc_yn, haccp_oblgapplc_trgt_yn, haccp_crtfc_yn, haccp_crtfc_dtm, sfrprp_food_se_cd
          FROM ba_data.s_co_ifd_ovsmnfst_regstr_m WHERE ovsmnfst_cd = ANY(%s)""", (oc,))
co.to_parquet(T + r"\brief_co_haccp.parquet", index=False)

# 7) 축산물 검사표(lpsms)·림스 원본 정보 키 형식 확인
print('\n키 형식 비교')
OUT['key_samples'] = {
    'ready_listeria_splore_reqest_no': q("SELECT splore_reqest_no FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아' AND splore_reqest_no IS NOT NULL LIMIT 5")['splore_reqest_no'].astype(str).tolist(),
    'lpsms_rqst_mte_rqsts_no': q("SELECT mte_rqsts_no FROM ba_data_info.lpsms_lcf_insp_rqst WHERE mte_rqsts_no IS NOT NULL LIMIT 5")['mte_rqsts_no'].astype(str).tolist(),
    'lpsms_rqst_insp_rqsts_no': q("SELECT insp_rqsts_no FROM ba_data_info.lpsms_lcf_insp_rqst LIMIT 5")['insp_rqsts_no'].astype(str).tolist(),
    'lpsms_smp_example': q("SELECT insp_rqsts_no, insp_smp_nm, pckn_mtrl, pckn_unt, dmst_anm_cd_m_nm, insp_asrt_cd_nm FROM ba_data_info.lpsms_lcf_insp_smp WHERE pckn_mtrl IS NOT NULL LIMIT 5").astype(str).to_dict('records'),
    'lims_org_info_example': q("SELECT splore_reqest_no, label, expr_se_nm, frght_cstdy_place_nm FROM ba_data_info.lims_org_info LIMIT 5").astype(str).to_dict('records'),
}
print(json.dumps(OUT['key_samples'], ensure_ascii=False, indent=1))
conn.close()
json.dump(OUT, open(T + r"\brief_db.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('saved')
