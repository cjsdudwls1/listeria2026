-- =====================================================================================
-- 리스테리아 예측 데이터 수집 쿼리  (PostgreSQL / DataGrip, DB: TOXIN_MODEL_PRED)
-- 근거 파일: 리스테리아ML/docs/02_데이터수집/리스테리아_수집변수_매핑.xlsx (2026-09-10 최종)
-- 원칙
--   * 전부 SELECT 만 사용 (DB 변경 없음). 각 섹션을 콘솔에서 따로 실행하고 결과를 CSV/Parquet 로 내보낸다.
--   * 몸통(0번) 은 노트북 df 와 같은 대상(hazard_label='리스테리아', 676,537행 예상).
--   * 1행/lims_data_key 가 보장되어야 하는 조인은 DISTINCT ON 으로 중복을 막았다.
--   * 기준일 = mnfctur_de_or_fallback (df 의 mnfctur_de 는 결측 0 이라 이 컬럼으로 보임).
--   * 국내 기상 결합은 geo_route = 'domestic' 만 (정의서 명시). 수입 건은 6번(해외) 사용.
-- 실행 순서: 1(검증) → 0(몸통) → 2(행 단위 속성) → 3(기상) → 4(토양) → 5(대기) → 6(해외) → 7(코드표)
-- =====================================================================================


-- =====================================================================================
-- 1. 검증 쿼리: 조인 키가 실제로 맞는지 먼저 확인 (각각 몇 초~1분)
-- =====================================================================================

-- 1-1. 몸통 행 수 / 국내·수입 / 기준일 범위  → 노트북 df(676,537행, domestic 628,895 / overseas 47,642) 와 같아야 함
SELECT geo_route, COUNT(*) AS n,
       MIN(mnfctur_de_or_fallback) AS min_base_date, MAX(mnfctur_de_or_fallback) AS max_base_date,
       COUNT(prdlst_mnfctur_sttemnt_no) AS n_prdlst_mnf_no,
       COUNT(bsn_lcns_ledg_no)          AS n_bsn_lcns_no,
       COUNT(cntc_sys_key2)             AS n_cntc_sys_key2,
       COUNT(legal_dong_code)           AS n_legal_dong,
       COUNT(lat)                       AS n_latlon
FROM ready_data.hazard_preprocessed_data
WHERE hazard_label = '리스테리아'
GROUP BY geo_route;

-- 1-2. lims_data_key ↔ ba_data.m_org_lims_data_info 매칭률 (국내 건은 100% 에 가까워야 함)
WITH body AS (
    SELECT lims_data_key, geo_route
    FROM ready_data.hazard_preprocessed_data
    WHERE hazard_label = '리스테리아')
SELECT b.geo_route, COUNT(*) AS n,
       COUNT(m.lims_data_key) AS n_matched,
       ROUND(100.0 * COUNT(m.lims_data_key) / COUNT(*), 2) AS pct_matched
FROM body b
LEFT JOIN ba_data.m_org_lims_data_info m ON m.lims_data_key = b.lims_data_key
GROUP BY b.geo_route;

-- 1-3. 수입 건 ↔ ba_data.s_org_lims_data_info 연결 키 후보 비교
--      후보 A: cntc_sys_key2 (정의서: '연계 키(수입 접수번호)')   후보 B: lims_data_key 의 두 번째 토큰
WITH body AS (
    SELECT lims_data_key, cntc_sys_key2, split_part(lims_data_key, '_', 2) AS key_token2
    FROM ready_data.hazard_preprocessed_data
    WHERE hazard_label = '리스테리아' AND geo_route = 'overseas')
SELECT COUNT(*) AS n_overseas,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM ba_data.s_org_lims_data_info s WHERE s.rcept_no = b.cntc_sys_key2)) AS via_cntc_sys_key2,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM ba_data.s_org_lims_data_info s WHERE s.rcept_no = b.key_token2))    AS via_key_token2
FROM body b;

-- 1-4. 품목제조번호 ↔ ba_data.t_prdlst_mnf_ledg 연결 키 후보 비교 (prdlst_report_no vs prdlst_report_ledg_no)
WITH body AS (
    SELECT DISTINCT prdlst_mnfctur_sttemnt_no AS k
    FROM ready_data.hazard_preprocessed_data
    WHERE hazard_label = '리스테리아' AND prdlst_mnfctur_sttemnt_no IS NOT NULL)
SELECT COUNT(*) AS n_keys,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM ba_data.t_prdlst_mnf_ledg l WHERE l.prdlst_report_no      = b.k)) AS via_report_no,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM ba_data.t_prdlst_mnf_ledg l WHERE l.prdlst_report_ledg_no = b.k)) AS via_report_ledg_no
FROM body b;

-- 1-5. 영업허가번호 ↔ ba_data.t_bsn_lcns_ledg
WITH body AS (
    SELECT DISTINCT bsn_lcns_ledg_no AS k
    FROM ready_data.hazard_preprocessed_data
    WHERE hazard_label = '리스테리아' AND bsn_lcns_ledg_no IS NOT NULL)
SELECT COUNT(*) AS n_keys,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM ba_data.t_bsn_lcns_ledg l WHERE l.bsn_lcns_ledg_no = b.k)) AS matched
FROM body b;

-- 1-6. 공통코드 분류코드 존재 확인 (운반방법 ZIM08 / 시료상태 ZIM11 / 운반온도단위 ZIM40)
SELECT cmmn_cl_code, COUNT(*) AS n_codes
FROM ba_data.t_cm_cmmn_code
WHERE cmmn_cl_code IN ('ZIM08', 'ZIM11', 'ZIM40')
GROUP BY cmmn_cl_code;

-- 1-7. 관측소 ID 형식 / 일자료 보유 관측소 수 (ASOS 관측소표 243개 vs 일자료 보유 관측소는 훨씬 적음)
SELECT 'asos_station' AS src, COUNT(*) AS n, MIN(stn_id) AS sample_min, MAX(stn_id) AS sample_max FROM api_data_info.kma_ground_obs_station_info
UNION ALL
SELECT 'asos_daily', COUNT(DISTINCT stn_id), MIN(stn_id), MAX(stn_id) FROM api_data_info.kma_asos_daily_info
UNION ALL
SELECT 'nas_station', COUNT(*), MIN(obsr_spot_code), MAX(obsr_spot_code) FROM api_data_info.nas_agri_weather_obs_station
UNION ALL
SELECT 'nas_daily', COUNT(DISTINCT obsr_spot_code), MIN(obsr_spot_code), MAX(obsr_spot_code) FROM api_data_info.nas_agri_weather_daily_info;

-- 1-8. 에어코리아 측정소 법정동코드 형식 (11자리로 정의됨 → 앞 5자리가 시군구인지 확인)
SELECT station_nm, legaldong_code, LENGTH(legaldong_code) AS len, use_yn, year, year_end
FROM api_data_info.keco_airkorea_obs_station
ORDER BY station_nm
LIMIT 30;

-- 1-9. 국가코드 체계 대조 (몸통의 수입국 코드 vs 해외 기상 alpha_3 vs global_soil_ph 의 iso3/nation_code)
SELECT import_nation_code, prdctn_nation_code, orgplce_nation_code, COUNT(*) AS n
FROM ready_data.hazard_preprocessed_data
WHERE hazard_label = '리스테리아' AND geo_route = 'overseas'
GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT 30;

SELECT iso3, nation_code, name_ko FROM api_data_info.global_soil_ph ORDER BY iso3 LIMIT 30;


-- =====================================================================================
-- 0. 몸통: 리스테리아 행 전체 (정의서 54컬럼 전부). 결과를 listeria_body_full 로 저장
--    df 의 23컬럼은 모두 여기 포함(src_route≈geo_route). df 에 없던 키 컬럼:
--    mnfctur_de_or_fallback, reqest_de, inspct_begin_de, prdlst_mnfctur_sttemnt_no, bsn_lcns_ledg_no,
--    cntc_sys_key2, geo_match_level, geo_source, statistic_prdlst_nm_1~4, mapped_prdlst_code, *_nation_code
-- =====================================================================================
SELECT *
FROM ready_data.hazard_preprocessed_data
WHERE hazard_label = '리스테리아';

-- (선택) 쓰기 권한이 있는 스키마가 있으면 몸통을 고정해 두면 이후 쿼리가 빨라진다. 권한 없으면 건너뜀.
-- CREATE TABLE train_data.listeria_body AS
-- SELECT * FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아';
-- CREATE INDEX ON train_data.listeria_body (lims_data_key);


-- =====================================================================================
-- 2. 행 단위 속성 (정확히 1행 / lims_data_key). 결과를 listeria_attr 로 저장 → df 와 lims_data_key 로 병합
--    포함: 식약처 LIMS 원본(운반 온도·방법, 시료 상태, 유통일, 채취일 등) / 품목제조대장(제조조건, 보존유통기준,
--          유통기한, 포장재질, HACCP) / 영업허가대장(축종·양식종·포장실 면적) / 식품공전 분류 / 수입 LIMS(물류) /
--          해외제조업소(동물성 식품 종류) / 코드 명칭
-- =====================================================================================
WITH body AS (
    SELECT lims_data_key, geo_route, cntc_sys_key2, prdlst_code, mapped_prdlst_code,
           prdlst_mnfctur_sttemnt_no, bsn_lcns_ledg_no
    FROM ready_data.hazard_preprocessed_data
    WHERE hazard_label = '리스테리아'),
m AS (   -- 식약처 LIMS 원본 (lims_data_key 가 PK)
    SELECT lims_data_key,
           cnvync_sttus_tp, cnvync_sttus_tp_unit_code, cnvync_mth_code, splore_sttus_code,
           distb_de, valid_tmlmt, pick_de, rcept_de, inspct_end_de,
           upper_prdlst_code, upper_prdlst_nm, expr_se_nm
    FROM ba_data.m_org_lims_data_info
    WHERE lims_data_key IN (SELECT lims_data_key FROM body)),
ledg AS (   -- 품목제조대장: 품목제조번호당 1행 (삭제 안 된 최신 행 우선). 1-4 결과에 따라 prdlst_report_no ↔ prdlst_report_ledg_no 교체
    SELECT DISTINCT ON (prdlst_report_no)
           prdlst_report_no,
           frmlc_mtrqlt, mnf_cnd_cn, prsrv_distb_stdr_cn, pog_daycnt, qlity_mntnc_tmlmt_daycnt, haccp_appn_no,
           prms_dt AS ledg_prms_dt, end_dt AS ledg_end_dt
    FROM ba_data.t_prdlst_mnf_ledg
    WHERE prdlst_report_no IN (SELECT prdlst_mnfctur_sttemnt_no FROM body WHERE prdlst_mnfctur_sttemnt_no IS NOT NULL)
    ORDER BY prdlst_report_no, (COALESCE(del_yn, 'N') = 'N') DESC, last_updt_dtm DESC NULLS LAST),
bsn AS (
    SELECT bsn_lcns_ledg_no, domini_sort, fish_sort, wrkshp_pkrm_area
    FROM ba_data.t_bsn_lcns_ledg
    WHERE bsn_lcns_ledg_no IN (SELECT bsn_lcns_ledg_no FROM body WHERE bsn_lcns_ledg_no IS NOT NULL)),
cl AS (   -- 식품공전 품목분류 (품목코드당 1행)
    SELECT DISTINCT ON (prdlst_cd)
           prdlst_cd, lv AS cl_lv, hrnk_prdlst_cd, htrk_prdlst_cd, fdgrp_yn, artcl_gun_code, food_hf_ls_cl_cd, dfn AS cl_dfn
    FROM ba_data.t_prdlst_cl
    ORDER BY prdlst_cd, (COALESCE(use_yn, 'Y') = 'Y') DESC),
s AS (   -- 수입식품 LIMS: 접수번호당 1행 (리스테리아 시험 행 우선). 1-3 결과에 따라 조인 키 교체
    SELECT DISTINCT ON (rcept_no)
           rcept_no,
           mnf_nation_cd, shipng_de, entrpt_de, wrhous_carry_de, bonded_area_cd, wrhous_cd, custom_cd,
           inspct_carry_plc, impt_ownr_induty, distb_bgn_de, distb_end_de,
           prduct_dvs_cd, dclr_prduct_dvs, hsk_no, packng_plc_cd, packng_plc_addr, ovsmnfst_cd
    FROM ba_data.s_org_lims_data_info
    WHERE rcept_no IN (SELECT cntc_sys_key2 FROM body WHERE cntc_sys_key2 IS NOT NULL)
    ORDER BY rcept_no, (test_item ILIKE '%리스테리아%') DESC, lims_seq),
co AS (
    SELECT ovsmnfst_cd, sfrprp_food_se_cd, sfrprp_regist_fclty_no
    FROM ba_data.s_co_ifd_ovsmnfst_regstr_m),
code AS (   -- 공통코드 (분류코드+코드 당 1행)
    SELECT DISTINCT ON (cmmn_cl_code, cmmn_code) cmmn_cl_code, cmmn_code, cmmn_code_nm
    FROM ba_data.t_cm_cmmn_code
    WHERE cmmn_cl_code IN ('ZIM08', 'ZIM11', 'ZIM40')
    ORDER BY cmmn_cl_code, cmmn_code, (COALESCE(use_at, 'Y') = 'Y') DESC)
SELECT b.lims_data_key,
       -- 식약처 LIMS 원본
       m.cnvync_sttus_tp, m.cnvync_sttus_tp_unit_code, c40.cmmn_code_nm AS cnvync_tp_unit_nm,
       m.cnvync_mth_code, c08.cmmn_code_nm AS cnvync_mth_nm,
       m.splore_sttus_code, c11.cmmn_code_nm AS splore_sttus_nm,
       m.distb_de, m.valid_tmlmt, m.pick_de, m.rcept_de, m.inspct_end_de,
       m.upper_prdlst_code, m.upper_prdlst_nm, m.expr_se_nm,
       -- 품목제조대장
       ledg.frmlc_mtrqlt, ledg.mnf_cnd_cn, ledg.prsrv_distb_stdr_cn, ledg.pog_daycnt, ledg.qlity_mntnc_tmlmt_daycnt,
       ledg.haccp_appn_no, ledg.ledg_prms_dt, ledg.ledg_end_dt,
       -- 영업허가대장
       bsn.domini_sort, bsn.fish_sort, bsn.wrkshp_pkrm_area,
       -- 식품공전 분류
       cl.cl_lv, cl.hrnk_prdlst_cd, cl.htrk_prdlst_cd, cl.fdgrp_yn, cl.artcl_gun_code, cl.food_hf_ls_cl_cd, cl.cl_dfn,
       -- 수입 LIMS + 해외제조업소
       s.mnf_nation_cd, s.shipng_de, s.entrpt_de, s.wrhous_carry_de, s.bonded_area_cd, s.wrhous_cd, s.custom_cd,
       s.inspct_carry_plc, s.impt_ownr_induty, s.distb_bgn_de, s.distb_end_de,
       s.prduct_dvs_cd, s.dclr_prduct_dvs, s.hsk_no, s.packng_plc_cd, s.packng_plc_addr, s.ovsmnfst_cd,
       co.sfrprp_food_se_cd, co.sfrprp_regist_fclty_no
FROM body b
LEFT JOIN m    ON m.lims_data_key = b.lims_data_key
LEFT JOIN ledg ON ledg.prdlst_report_no = b.prdlst_mnfctur_sttemnt_no
LEFT JOIN bsn  ON bsn.bsn_lcns_ledg_no = b.bsn_lcns_ledg_no
LEFT JOIN cl   ON cl.prdlst_cd = COALESCE(b.mapped_prdlst_code, b.prdlst_code)
LEFT JOIN s    ON s.rcept_no = b.cntc_sys_key2
LEFT JOIN co   ON co.ovsmnfst_cd = s.ovsmnfst_cd
LEFT JOIN code c40 ON c40.cmmn_cl_code = 'ZIM40' AND c40.cmmn_code = m.cnvync_sttus_tp_unit_code
LEFT JOIN code c08 ON c08.cmmn_cl_code = 'ZIM08' AND c08.cmmn_code = m.cnvync_mth_code
LEFT JOIN code c11 ON c11.cmmn_cl_code = 'ZIM11' AND c11.cmmn_code = m.splore_sttus_code;


-- =====================================================================================
-- 3. 국내 기상 (ASOS + 농업기상 NAS). 국내(domestic) 건만.
--    방식: 3-1 좌표→최근접 관측소 표 (좌표 중복 제거라 약 7천 행) + 3-2/3-3 일자료 내보내기 → 파이썬에서 기준일 창(D0, 7일, 30일) 집계
--    또는 3-4 처럼 SQL 에서 바로 창 집계 (무거움, 676k행 × 30일 조인)
-- =====================================================================================

-- 3-1. 좌표별 최근접 ASOS / NAS 관측소 (일자료를 실제 보유한 관측소만 후보로). 결과를 stn_map 으로 저장 → df 와 (lat, lon) 으로 병합
WITH pts AS (
    SELECT DISTINCT lat::double precision AS lat, lon::double precision AS lon
    FROM ready_data.hazard_preprocessed_data
    WHERE hazard_label = '리스테리아' AND geo_route = 'domestic' AND lat IS NOT NULL AND lon IS NOT NULL),
asos AS (
    SELECT st.stn_id, st.lat::double precision AS lat, st.lon::double precision AS lon, st.ht, st.basin, st.fct_id, st.legaldong_code
    FROM api_data_info.kma_ground_obs_station_info st
    WHERE st.stn_id::text IN (SELECT DISTINCT stn_id FROM api_data_info.kma_asos_daily_info)),
nas AS (
    SELECT st.obsr_spot_code, st.instl_la::double precision AS lat, st.instl_lo::double precision AS lon, st.clmt_zone_code, st.legaldong_code
    FROM api_data_info.nas_agri_weather_obs_station st
    WHERE st.obsr_spot_code IN (SELECT DISTINCT obsr_spot_code FROM api_data_info.nas_agri_weather_daily_info))
SELECT p.lat, p.lon,
       a.stn_id        AS asos_stn_id,  ROUND(a.dist_km::numeric, 1) AS asos_dist_km, a.ht AS asos_ht, a.basin AS asos_basin, a.fct_id AS asos_fct_id,
       n.obsr_spot_code AS nas_stn_id,  ROUND(n.dist_km::numeric, 1) AS nas_dist_km,  n.clmt_zone_code
FROM pts p
CROSS JOIN LATERAL (
    SELECT s.stn_id, s.ht, s.basin, s.fct_id,
           6371 * acos(LEAST(1.0, cos(radians(p.lat)) * cos(radians(s.lat)) * cos(radians(s.lon) - radians(p.lon))
                                + sin(radians(p.lat)) * sin(radians(s.lat)))) AS dist_km
    FROM asos s ORDER BY dist_km LIMIT 1) a
CROSS JOIN LATERAL (
    SELECT s.obsr_spot_code, s.clmt_zone_code,
           6371 * acos(LEAST(1.0, cos(radians(p.lat)) * cos(radians(s.lat)) * cos(radians(s.lon) - radians(p.lon))
                                + sin(radians(p.lat)) * sin(radians(s.lat)))) AS dist_km
    FROM nas s ORDER BY dist_km LIMIT 1) n;

-- 3-2. ASOS 일자료 (기준일 범위 - 30일 ~ 최대 기준일). 402k행 전체라 그냥 다 받아도 됨. 결과를 asos_daily 로 저장
SELECT stn_id, meas_date,
       avg_ta, max_ta, min_ta,                                   -- 기온
       sum_rn, n99_rn, hr_1_max_rn, mi_10_max_rn,                 -- 강수
       avg_ws, max_ws, max_ws_wd, max_ins_ws, max_ins_ws_wd, max_ws1, hr_24_sum_rws,   -- 바람
       CASE WHEN sum_sshr ~ '^\s*-?[0-9]+(\.[0-9]+)?\s*$' THEN sum_sshr::numeric END AS sum_sshr_num,   -- 일조 (text→숫자)
       CASE WHEN ss_dur   ~ '^\s*-?[0-9]+(\.[0-9]+)?\s*$' THEN ss_dur::numeric   END AS ss_dur_num,
       sum_gsr, hr_1_max_icsr,                                    -- 일사
       avg_ts, min_tg, avg_cm5_te, avg_cm10_te, avg_cm20_tes, avg_cm30_te,            -- 지면·지중온도
       avg_m05_te, avg_m10_te, avg_m15_te, avg_m30_te, avg_m50_te
FROM api_data_info.kma_asos_daily_info
WHERE meas_date BETWEEN (SELECT MIN(mnfctur_de_or_fallback) - 30 FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아')
                    AND (SELECT MAX(mnfctur_de_or_fallback)      FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아');

-- 3-3. 농업기상(NAS) 일자료. 결과를 nas_daily 로 저장
SELECT obsr_spot_code, meas_date,
       temp, hghst_artmp, lowst_artmp,                            -- 기온
       rn,                                                        -- 강수 (0 명시 기록 → NULL 은 진짜 결측)
       wind, max_wind, widdir,                                    -- 바람
       CASE WHEN sun_time ~ '^\s*-?[0-9]+(\.[0-9]+)?\s*$' THEN sun_time::numeric END AS sun_time_num,   -- 일조 (관측소 91/219)
       srqty,                                                     -- 일사 (관측소 180/219)
       gr_temp, soil_temp, soil_wt                                -- 지면온도·지중온도·토양수분
FROM api_data_info.nas_agri_weather_daily_info
WHERE meas_date BETWEEN (SELECT MIN(mnfctur_de_or_fallback) - 30 FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아')
                    AND (SELECT MAX(mnfctur_de_or_fallback)      FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아');

-- 3-4. (옵션) SQL 에서 바로 창 집계: ASOS 예시. 1행/lims_data_key. 수 분 이상 걸릴 수 있음. NAS 도 같은 틀로 작성
WITH body AS (
    SELECT lims_data_key, lat::double precision AS lat, lon::double precision AS lon, mnfctur_de_or_fallback AS base_date
    FROM ready_data.hazard_preprocessed_data
    WHERE hazard_label = '리스테리아' AND geo_route = 'domestic' AND lat IS NOT NULL AND mnfctur_de_or_fallback IS NOT NULL),
pts AS (SELECT DISTINCT lat, lon FROM body),
asos AS (
    SELECT st.stn_id::text AS stn_id, st.lat::double precision AS lat, st.lon::double precision AS lon
    FROM api_data_info.kma_ground_obs_station_info st
    WHERE st.stn_id::text IN (SELECT DISTINCT stn_id FROM api_data_info.kma_asos_daily_info)),
stn_map AS (
    SELECT p.lat, p.lon, a.stn_id
    FROM pts p
    CROSS JOIN LATERAL (
        SELECT s.stn_id,
               6371 * acos(LEAST(1.0, cos(radians(p.lat)) * cos(radians(s.lat)) * cos(radians(s.lon) - radians(p.lon))
                                    + sin(radians(p.lat)) * sin(radians(s.lat)))) AS dist_km
        FROM asos s ORDER BY dist_km LIMIT 1) a),
bm AS (
    SELECT b.lims_data_key, b.base_date, sm.stn_id
    FROM body b JOIN stn_map sm ON sm.lat = b.lat AND sm.lon = b.lon)
SELECT bm.lims_data_key, bm.stn_id AS asos_stn_id,
       -- 당일(D0)
       MAX(w.avg_ta)  FILTER (WHERE w.meas_date = bm.base_date) AS ta_d0,
       MAX(w.sum_rn)  FILTER (WHERE w.meas_date = bm.base_date) AS rn_d0,
       MAX(w.avg_ws)  FILTER (WHERE w.meas_date = bm.base_date) AS ws_d0,
       MAX(w.sum_gsr) FILTER (WHERE w.meas_date = bm.base_date) AS gsr_d0,
       MAX(w.avg_cm10_te) FILTER (WHERE w.meas_date = bm.base_date) AS soil10_d0,
       -- 7일 창 (D-6 ~ D0)
       AVG(w.avg_ta)  FILTER (WHERE w.meas_date >= bm.base_date - 6) AS ta_mean_7d,
       MAX(w.max_ta)  FILTER (WHERE w.meas_date >= bm.base_date - 6) AS ta_max_7d,
       MIN(w.min_ta)  FILTER (WHERE w.meas_date >= bm.base_date - 6) AS ta_min_7d,
       SUM(w.sum_rn)  FILTER (WHERE w.meas_date >= bm.base_date - 6) AS rn_sum_7d,
       AVG(w.avg_ws)  FILTER (WHERE w.meas_date >= bm.base_date - 6) AS ws_mean_7d,
       AVG(CASE WHEN w.sum_sshr ~ '^\s*-?[0-9]+(\.[0-9]+)?\s*$' THEN w.sum_sshr::numeric END)
                      FILTER (WHERE w.meas_date >= bm.base_date - 6) AS sshr_mean_7d,
       AVG(w.sum_gsr) FILTER (WHERE w.meas_date >= bm.base_date - 6) AS gsr_mean_7d,
       AVG(w.avg_cm5_te)   FILTER (WHERE w.meas_date >= bm.base_date - 6) AS soil5_mean_7d,
       AVG(w.avg_cm10_te)  FILTER (WHERE w.meas_date >= bm.base_date - 6) AS soil10_mean_7d,
       AVG(w.avg_cm20_tes) FILTER (WHERE w.meas_date >= bm.base_date - 6) AS soil20_mean_7d,
       -- 30일 창 (D-29 ~ D0)
       AVG(w.avg_ta)  AS ta_mean_30d,
       SUM(w.sum_rn)  AS rn_sum_30d,
       AVG(w.avg_ws)  AS ws_mean_30d,
       AVG(w.sum_gsr) AS gsr_mean_30d,
       AVG(w.avg_cm10_te) AS soil10_mean_30d,
       COUNT(w.meas_date) AS n_days_30d
FROM bm
JOIN api_data_info.kma_asos_daily_info w
  ON w.stn_id = bm.stn_id AND w.meas_date BETWEEN bm.base_date - 29 AND bm.base_date
GROUP BY bm.lims_data_key, bm.stn_id;


-- =====================================================================================
-- 4. 토양 (soil_ph_info): 법정동 × 조사연도 집계 + 시군구 fallback. 결과를 soil_dong, soil_sgg 로 저장
--    조인: df.legal_dong_code(10자리) + 기준일 연도 → 없으면 시군구(앞 5자리) + 연도 → 없으면 시군구 전체 연도
-- =====================================================================================
-- 4-1. 법정동 × 연도
SELECT legal_dong_code, any_year::int AS yr, COUNT(*) AS n_samples,
       AVG(acid) AS soil_ph_mean, STDDEV(acid) AS soil_ph_std,
       AVG(vldpha) AS vldpha_mean, AVG(om) AS om_mean
FROM api_data_info.soil_ph_info
WHERE acid IS NOT NULL
GROUP BY legal_dong_code, any_year;

-- 4-2. 시군구 × 연도 (fallback)
SELECT LEFT(legal_dong_code, 5) AS sgg_cd, any_year::int AS yr, COUNT(*) AS n_samples,
       AVG(acid) AS soil_ph_mean, AVG(vldpha) AS vldpha_mean, AVG(om) AS om_mean
FROM api_data_info.soil_ph_info
WHERE acid IS NOT NULL
GROUP BY LEFT(legal_dong_code, 5), any_year;

-- 4-3. (해외) 국가별 토양 pH. 수입 건에 국가코드로 결합
SELECT iso3, nation_code, name_ko, sub_region, ph_mean, ph_std
FROM api_data_info.global_soil_ph;


-- =====================================================================================
-- 5. 대기 오염물질 (에어코리아 NO2·CO·SO2): 시군구 × 일 평균. 결과를 air_sgg_daily 로 저장
--    조인: LEFT(df.legal_dong_code, 5) = sgg_cd, 기준일 창 집계는 파이썬에서 (기상과 같은 방식)
--    1-8 에서 legaldong_code 앞 5자리가 시군구가 아니면 dm_x/dm_y(TM) 를 pyproj 로 위경도 변환해 3-1 방식으로 최근접 매칭
-- =====================================================================================
WITH st AS (
    SELECT DISTINCT ON (station_nm) station_nm, legaldong_code
    FROM api_data_info.keco_airkorea_obs_station
    WHERE legaldong_code IS NOT NULL
    ORDER BY station_nm, (COALESCE(use_yn, 'Y') = 'Y') DESC, chg_dt DESC NULLS LAST)
SELECT LEFT(st.legaldong_code, 5) AS sgg_cd, f.msur_dt,
       AVG(f.no2_value) AS no2_mean, AVG(f.co_value) AS co_mean, AVG(f.so2_value) AS so2_mean,
       COUNT(*) AS n_stations
FROM api_data_info.keco_airkorea_fine_dust_info f
JOIN st ON st.station_nm = f.station_nm
WHERE f.msur_dt BETWEEN (SELECT MIN(mnfctur_de_or_fallback) - 30 FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아')
                    AND (SELECT MAX(mnfctur_de_or_fallback)      FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아')
GROUP BY LEFT(st.legaldong_code, 5), f.msur_dt;


-- =====================================================================================
-- 6. 해외 (수입 건 47,642행 전용): 국가 × 일 기상. 결과를 global_daily 로 저장
--    조인: 몸통 국가코드 → alpha_3 (1-9 로 코드 체계 확인. 몸통 코드가 숫자 3자리면 global_soil_ph.nation_code→iso3 로 변환)
--    주의: 테이블목록에 '재수집 대상, 현재 미사용' 표시 → 품질 확인 후 사용
-- =====================================================================================
SELECT alpha_3, name_ko, date,
       temperature_mean, temperature_max, temperature_min, precipitation_sum,
       relative_humidity_mean, soil_moisture_mean, sunshine_duration, wind_speed_mean, shortwave_radiation_sum
FROM api_data_info.global_weather_info
WHERE date BETWEEN (SELECT MIN(mnfctur_de_or_fallback) - 30 FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아' AND geo_route = 'overseas')
               AND (SELECT MAX(mnfctur_de_or_fallback)      FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아' AND geo_route = 'overseas');


-- =====================================================================================
-- 7. 코드표 (명칭 확인용). 결과를 code_zim 으로 저장
-- =====================================================================================
SELECT cmmn_cl_code, cmmn_code, cmmn_code_nm, cmmn_code_dc, use_at
FROM ba_data.t_cm_cmmn_code
WHERE cmmn_cl_code IN ('ZIM08', 'ZIM11', 'ZIM40')
ORDER BY cmmn_cl_code, sort_ordr, cmmn_code;
