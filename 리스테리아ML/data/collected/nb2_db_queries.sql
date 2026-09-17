-- 2.ipynb 보조 데이터 추출 쿼리 (읽기 전용, SELECT 만). 추출일: 2026-09-14

-- ===== attr =====
SELECT h.lims_data_key,
       h.judge_result            AS judge_result_db,
       h.geo_route, h.date_basis AS date_basis_db, h.addr_source AS addr_source_db,
       h.dup_count, h.splore_cnt, h.expr_iem_cnt,
       h.mnfctur_de              AS mnfctur_de_raw,
       h.reqest_de, h.inspct_begin_de, h.mnfctur_de_or_fallback,
       h.prdlst_mnfctur_sttemnt_no, h.bsn_lcns_ledg_no, h.cntc_sys_key2,
       h.orgplce_nation_code, h.import_nation_code, h.prdctn_nation_code,
       h.geo_match_level, h.geo_source, h.geocode_status,
       h.prdlst_nm AS prdlst_nm_self, h.prdlst_nm_org,
       h.statistic_prdlst_nm_1, h.statistic_prdlst_nm_2, h.statistic_prdlst_nm_3, h.statistic_prdlst_nm_4,
       h.mapped_prdlst_code, h.aw_grade, h.aw_match_level,
       m.inspct_purps_code, m.inspct_knd_code, m.reqest_sys_code, m.reqest_instt_code,
       m.expr_se_nm, m.upper_prdlst_code, m.upper_prdlst_nm,
       m.splore_sttus_code, m.cnvync_mth_code, m.cnvync_sttus_tp, m.cnvync_sttus_tp_unit_code,
       m.pick_de, m.rcept_de, m.distb_de, m.valid_tmlmt, m.process_tmlmt, m.process_de,
       m.inspct_compt_de, m.inspct_end_de, m.inspct_reqest_dt,
       m.inspct_splore_cnt, m.mxmm_perm_splore_cnt, m.stdr_se_code, m.jdgmnt_fom_code, m.stndrd_value,
       m.xport_nation_code, m.xport_nation_nm, m.prdctn_nation_nm,
       (m.lims_data_key IS NOT NULL) AS in_m_org
FROM ready_data.hazard_preprocessed_data h
LEFT JOIN ba_data.m_org_lims_data_info m ON m.lims_data_key = h.lims_data_key
WHERE h.hazard_label = '리스테리아';

-- ===== s_org =====
SELECT lims_seq, rcept_no, sample_no, inspct_instt_cd, rcept_de, prcs_de, impt_prpos_cd, prpos_nm,
       prdlst_cd, prdlst_nm, upper_prdlst_nm, prduct_nm_kor, hsk_no, prduct_dvs_cd, dclr_prduct_dvs,
       mnf_nation_cd, mnf_nation_nm, impt_ownr_induty, ovsmnfst_cd,
       inspct_kind, inspct_kind_dtl, inspct_kind_cd,
       distb_bgn_de, distb_end_de, shipng_de, entrpt_de, wrhous_carry_de, bonded_area_cd, inspct_carry_plc,
       inspct_reqst_de, inspct_cmptn_de, test_item, result_value, test_result, stdr_value
FROM ba_data.s_org_lims_data_info
WHERE label = '리스테리아';

-- ===== ledg =====
SELECT prdlst_report_no, prdlst_report_ledg_no, prdlst_cd, prdlst_nm, prms_dt, end_dt,
       pog_daycnt, qlity_mntnc_tmlmt_daycnt, pog_cn, frmlc_mtrqlt, haccp_appn_no, del_yn,
       mnf_cnd_cn, prsrv_distb_stdr_cn, food_hf_ls_cl_cd, induty_cd, last_updt_dtm
FROM ba_data.t_prdlst_mnf_ledg
WHERE prdlst_report_no = ANY(%s::text[]);

-- ===== instt =====
WITH nm AS (
    SELECT h.inspct_instt_code, MAX(COALESCE(m.inspct_instt_nm, h.inspct_instt_nm)) AS instt_nm, COUNT(*) AS n_rows
    FROM ready_data.hazard_preprocessed_data h
    LEFT JOIN ba_data.m_org_lims_data_info m ON m.lims_data_key = h.lims_data_key
    WHERE h.hazard_label = '리스테리아'
    GROUP BY h.inspct_instt_code)
SELECT nm.inspct_instt_code, nm.instt_nm, nm.n_rows, i.instt_eng_nm, i.adres
FROM nm LEFT JOIN ba_data.m_sy_inspct_instt i ON i.inspct_instt_code = nm.inspct_instt_code
ORDER BY nm.n_rows DESC;

-- ===== codes =====
SELECT cmmn_cl_code, cmmn_code, cmmn_code_nm, cmmn_code_dc, use_at
FROM ba_data.t_cm_cmmn_code
WHERE cmmn_cl_code IN ('IM02','IM03','IM08','IM11','IM40') OR cmmn_code = ANY(%s::text[])
ORDER BY cmmn_cl_code, cmmn_code;
