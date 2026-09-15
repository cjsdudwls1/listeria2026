# -*- coding: utf-8 -*-
"""수입 시료 관련 칸 받기 (읽기 전용 SELECT, 우리 수입 시료 키로만 조회)"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import pandas as pd
from db_utils import connect
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"; C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
conn = connect()
def q(sql, params=None, label=''):
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    print(f"[{label}] {len(df):,}행 {time.time() - t0:.1f}초", flush=True)
    return df
for t in ['m_org_lims_data_info', 's_org_lims_data_info', 's_co_ifd_ovsmnfst_regstr_m']:
    print(t, q("SELECT indexdef FROM pg_indexes WHERE schemaname='ba_data' AND tablename=%s", (t,), 'idx')['indexdef'].tolist())
print('lims_org_info', q("SELECT indexdef FROM pg_indexes WHERE schemaname='ba_data_info' AND tablename='lims_org_info'", None, 'idx')['indexdef'].tolist())

body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
keys = body.loc[body.src_route == 'overseas', 'lims_data_key'].tolist()
rd = q("""SELECT lims_data_key, splore_reqest_no, splore_sn, cntc_sys_key2, mnfctur_cmpny_locplc_nm, ovsmnfst_nm
          FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아' AND lims_data_key = ANY(%s)""", (keys,), 'ready_data')
rd.to_parquet(T + r"\imp_ready.parquet", index=False)
mo = q("""SELECT lims_data_key, mnfctur_de, distb_de, orgplce_nation_code, prdctn_nation_code, prdctn_nation_nm, xport_nation_code, xport_nation_nm, import_nation_code,
                 mnfctur_cmpny_locplc_nm, prdlst_eng_nm, prdlst_nm, splore_nm, import_entrps_nm
          FROM ba_data.m_org_lims_data_info WHERE lims_data_key = ANY(%s)""", (keys,), 'm_org')
mo.to_parquet(T + r"\imp_m_org.parquet", index=False)
srq = rd['splore_reqest_no'].dropna().astype(str).unique().tolist()
lo = q("""SELECT splore_reqest_no, label, cntc_sys_key2, mnfctur_year, mnfctur_mt, mnfctur_de, distb_begin_dt, distb_end_dt, ship_port_date, arvl_prearnge_de,
                 prload_nm, etrypt_de, frght_cstdy_place_nm, prdctn_nation_code, xport_nation_code, orgplce_nation_code, expr_se_nm
          FROM ba_data_info.lims_org_info WHERE splore_reqest_no = ANY(%s) AND label = '리스테리아'""", (srq,), 'lims_org_info')
lo.to_parquet(T + r"\imp_lims_org_info.parquet", index=False)
rc = rd['cntc_sys_key2'].dropna().astype(str).unique().tolist()
so = q("SELECT rcept_no, lims_seq, bl_no, custom_cd, bonded_area_cd, wrhous_cd, inspct_carry_plc FROM ba_data.s_org_lims_data_info WHERE rcept_no = ANY(%s)", (rc,), 's_org bl_no')
so.to_parquet(T + r"\imp_s_org_bl.parquet", index=False)
s02 = pd.read_parquet(C + r"\02e_s_org.parquet", columns=['ovsmnfst_cd'])
oc = s02['ovsmnfst_cd'].dropna().astype(str).unique().tolist()
co = q("""SELECT ovsmnfst_cd, ntncd, crtfc_nm, crtfc_instt_nm, fsms_yn, fsms_crtfc_yn, sfrprp_food_se_cd, sfrprp_regist_fclty_no, mprc_regist_fclty_no, food_begin_dtm, marn_begin_dtm, wthdrw_yn
          FROM ba_data.s_co_ifd_ovsmnfst_regstr_m WHERE ovsmnfst_cd = ANY(%s)""", (oc,), 'registry')
co.to_parquet(T + r"\imp_registry.parquet", index=False)
vals = co['sfrprp_food_se_cd'].dropna().astype(str).unique().tolist() + so['bonded_area_cd'].dropna().astype(str).unique().tolist()
cm = q("SELECT cmmn_cl_code, cmmn_code, cmmn_code_nm FROM ba_data.t_cm_cmmn_code WHERE cmmn_code = ANY(%s)", (vals,), 'codes')
cm.to_parquet(T + r"\imp_codes.parquet", index=False)
conn.close()
print('done')
