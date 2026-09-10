# -*- coding: utf-8 -*-
"""리스테리아 수집 파이프라인: DB → parquet (리스테리아ML/data/collected/).
- 세션 전용 TEMP 테이블(body)만 만들고 실제 테이블은 건드리지 않는다.
- 각 단계 결과가 이미 있으면 건너뛴다 (재실행 안전).
- 조인 키가 불확실한 두 곳(수입 접수번호, 품목제조번호)은 매칭 수를 비교해 자동 선택하고 로그에 남긴다.
사용: py collect_all.py            (전체)
      py collect_all.py 2 3        (특정 단계만)
"""
import os
import sys
import time
import json
import decimal
import datetime as dt
import pandas as pd
from db_utils import connect

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
os.makedirs(OUT, exist_ok=True)
LOG = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_parquet(df: pd.DataFrame, name: str):
    """psycopg 결과(Decimal/date 객체)를 parquet 친화 타입으로 바꿔 저장."""
    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            nn = s.dropna()
            if len(nn) == 0:
                continue
            v = nn.iloc[0]
            if isinstance(v, decimal.Decimal):
                df[c] = pd.to_numeric(s, errors="coerce").astype("float64")
            elif isinstance(v, dt.datetime):
                df[c] = pd.to_datetime(s, errors="coerce")
            elif isinstance(v, dt.date):
                df[c] = pd.to_datetime(s, errors="coerce")
            elif isinstance(v, (list, dict)):
                df[c] = s.astype(str)
    path = os.path.join(OUT, name)
    df.to_parquet(path, index=False)
    log(f"saved {name}: {len(df):,} rows x {df.shape[1]} cols")
    return path


def fetch(conn, sql, label, params=None) -> pd.DataFrame:
    t0 = time.time()
    with conn.transaction():  # 서버측 커서는 트랜잭션 블록 안에서만 동작
        with conn.cursor(name=f"cur_{label}_{int(t0)}") as cur:
            cur.itersize = 100_000
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            chunks, n = [], 0
            while True:
                rows = cur.fetchmany(100_000)
                if not rows:
                    break
                chunks.append(pd.DataFrame(rows, columns=cols))
                n += len(rows)
                if n % 500_000 < 100_000:
                    log(f"  {label}: {n:,} rows ...")
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    LOG[label] = {"rows": len(df), "sec": round(time.time() - t0)}
    log(f"  {label}: {len(df):,} rows in {time.time() - t0:,.0f}s")
    return df


def scalar(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()


def exists(name):
    return os.path.exists(os.path.join(OUT, name))


# ------------------------------------------------------------------------------------
def step_body(conn):
    """TEMP body 생성 (+인덱스) 및 00_body.parquet 저장"""
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("SET work_mem = '512MB'")
        cur.execute("SET max_parallel_workers_per_gather = 4")
        cur.execute("DROP TABLE IF EXISTS body")
        cur.execute("CREATE TEMP TABLE body AS SELECT * FROM ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아'")
        cur.execute("CREATE INDEX ON body (lims_data_key)")
        cur.execute("CREATE INDEX ON body (prdlst_mnfctur_sttemnt_no)")
        cur.execute("CREATE INDEX ON body (bsn_lcns_ledg_no)")
        cur.execute("CREATE INDEX ON body (cntc_sys_key2)")
        cur.execute("ANALYZE body")
    n = scalar(conn, "SELECT COUNT(*) FROM body")[0]
    log(f"TEMP body ready: {n:,} rows ({time.time() - t0:,.0f}s)")
    if not exists("00_body.parquet"):
        df = fetch(conn, "SELECT * FROM body", "00_body")
        to_parquet(df, "00_body.parquet")
    # 기준일 범위
    r = scalar(conn, "SELECT MIN(mnfctur_de_or_fallback), MAX(mnfctur_de_or_fallback) FROM body")
    LOG["base_date_range"] = [str(r[0]), str(r[1])]
    log(f"base_date range: {r[0]} ~ {r[1]}")


def step_attr(conn):
    """2. 행 단위 속성 (1행/lims_data_key)"""
    if exists("02_attr.parquet"):
        log("skip 02_attr (exists)"); return
    # 조인 키 자동 선택
    s_a = scalar(conn, "SELECT COUNT(*) FROM body b WHERE geo_route='overseas' AND EXISTS (SELECT 1 FROM ba_data.s_org_lims_data_info s WHERE s.rcept_no = b.cntc_sys_key2)")[0]
    s_b = scalar(conn, "SELECT COUNT(*) FROM body b WHERE geo_route='overseas' AND EXISTS (SELECT 1 FROM ba_data.s_org_lims_data_info s WHERE s.rcept_no = split_part(b.lims_data_key,'_',2))")[0]
    s_key = "b.cntc_sys_key2" if s_a >= s_b else "split_part(b.lims_data_key,'_',2)"
    LOG["s_org_key"] = {"via_cntc_sys_key2": s_a, "via_token2": s_b, "chosen": s_key}
    log(f"s_org key: cntc_sys_key2={s_a:,} / token2={s_b:,} → {s_key}")

    l_a = scalar(conn, "SELECT COUNT(DISTINCT b.prdlst_mnfctur_sttemnt_no) FROM body b WHERE EXISTS (SELECT 1 FROM ba_data.t_prdlst_mnf_ledg l WHERE l.prdlst_report_no = b.prdlst_mnfctur_sttemnt_no)")[0]
    l_b = scalar(conn, "SELECT COUNT(DISTINCT b.prdlst_mnfctur_sttemnt_no) FROM body b WHERE EXISTS (SELECT 1 FROM ba_data.t_prdlst_mnf_ledg l WHERE l.prdlst_report_ledg_no = b.prdlst_mnfctur_sttemnt_no)")[0]
    l_key = "prdlst_report_no" if l_a >= l_b else "prdlst_report_ledg_no"
    LOG["ledg_key"] = {"via_report_no": l_a, "via_report_ledg_no": l_b, "chosen": l_key}
    log(f"ledg key: report_no={l_a:,} / report_ledg_no={l_b:,} → {l_key}")

    sql = f"""
WITH m AS (
    SELECT lims_data_key,
           cnvync_sttus_tp, cnvync_sttus_tp_unit_code, cnvync_mth_code, splore_sttus_code,
           distb_de, valid_tmlmt, pick_de, rcept_de, inspct_end_de,
           upper_prdlst_code, upper_prdlst_nm, expr_se_nm
    FROM ba_data.m_org_lims_data_info
    WHERE lims_data_key IN (SELECT lims_data_key FROM body)),
ledg AS (
    SELECT DISTINCT ON ({l_key})
           {l_key} AS ledg_key,
           frmlc_mtrqlt, mnf_cnd_cn, prsrv_distb_stdr_cn, pog_daycnt, qlity_mntnc_tmlmt_daycnt, haccp_appn_no,
           prms_dt AS ledg_prms_dt, end_dt AS ledg_end_dt
    FROM ba_data.t_prdlst_mnf_ledg
    WHERE {l_key} IN (SELECT prdlst_mnfctur_sttemnt_no FROM body WHERE prdlst_mnfctur_sttemnt_no IS NOT NULL)
    ORDER BY {l_key}, (COALESCE(del_yn, 'N') = 'N') DESC, last_updt_dtm DESC NULLS LAST),
bsn AS (
    SELECT bsn_lcns_ledg_no, domini_sort, fish_sort, wrkshp_pkrm_area
    FROM ba_data.t_bsn_lcns_ledg
    WHERE bsn_lcns_ledg_no IN (SELECT bsn_lcns_ledg_no FROM body WHERE bsn_lcns_ledg_no IS NOT NULL)),
cl AS (
    SELECT DISTINCT ON (prdlst_cd)
           prdlst_cd, lv AS cl_lv, hrnk_prdlst_cd, htrk_prdlst_cd, fdgrp_yn, artcl_gun_code, food_hf_ls_cl_cd, dfn AS cl_dfn
    FROM ba_data.t_prdlst_cl
    ORDER BY prdlst_cd, (COALESCE(use_yn, 'Y') = 'Y') DESC),
bkey AS (SELECT lims_data_key, {s_key} AS s_key FROM body b),
s AS (
    SELECT DISTINCT ON (rcept_no)
           rcept_no,
           mnf_nation_cd, shipng_de, entrpt_de, wrhous_carry_de, bonded_area_cd, wrhous_cd, custom_cd,
           inspct_carry_plc, impt_ownr_induty, distb_bgn_de, distb_end_de,
           prduct_dvs_cd, dclr_prduct_dvs, hsk_no, packng_plc_cd, packng_plc_addr, ovsmnfst_cd
    FROM ba_data.s_org_lims_data_info
    WHERE rcept_no IN (SELECT s_key FROM bkey WHERE s_key IS NOT NULL)
    ORDER BY rcept_no, (test_item ILIKE '%리스테리아%') DESC, lims_seq),
co AS (
    SELECT ovsmnfst_cd, sfrprp_food_se_cd, sfrprp_regist_fclty_no
    FROM ba_data.s_co_ifd_ovsmnfst_regstr_m),
code AS (
    SELECT DISTINCT ON (cmmn_cl_code, cmmn_code) cmmn_cl_code, cmmn_code, cmmn_code_nm
    FROM ba_data.t_cm_cmmn_code
    WHERE cmmn_cl_code IN ('ZIM08', 'ZIM11', 'ZIM40')
    ORDER BY cmmn_cl_code, cmmn_code, (COALESCE(use_at, 'Y') = 'Y') DESC)
SELECT b.lims_data_key,
       m.cnvync_sttus_tp, m.cnvync_sttus_tp_unit_code, c40.cmmn_code_nm AS cnvync_tp_unit_nm,
       m.cnvync_mth_code, c08.cmmn_code_nm AS cnvync_mth_nm,
       m.splore_sttus_code, c11.cmmn_code_nm AS splore_sttus_nm,
       m.distb_de, m.valid_tmlmt, m.pick_de, m.rcept_de, m.inspct_end_de,
       m.upper_prdlst_code, m.upper_prdlst_nm, m.expr_se_nm,
       ledg.frmlc_mtrqlt, ledg.mnf_cnd_cn, ledg.prsrv_distb_stdr_cn, ledg.pog_daycnt, ledg.qlity_mntnc_tmlmt_daycnt,
       ledg.haccp_appn_no, ledg.ledg_prms_dt, ledg.ledg_end_dt,
       bsn.domini_sort, bsn.fish_sort, bsn.wrkshp_pkrm_area,
       cl.cl_lv, cl.hrnk_prdlst_cd, cl.htrk_prdlst_cd, cl.fdgrp_yn, cl.artcl_gun_code, cl.food_hf_ls_cl_cd, cl.cl_dfn,
       s.mnf_nation_cd, s.shipng_de, s.entrpt_de, s.wrhous_carry_de, s.bonded_area_cd, s.wrhous_cd, s.custom_cd,
       s.inspct_carry_plc, s.impt_ownr_induty, s.distb_bgn_de, s.distb_end_de,
       s.prduct_dvs_cd, s.dclr_prduct_dvs, s.hsk_no, s.packng_plc_cd, s.packng_plc_addr, s.ovsmnfst_cd,
       co.sfrprp_food_se_cd, co.sfrprp_regist_fclty_no
FROM body b
LEFT JOIN bkey ON bkey.lims_data_key = b.lims_data_key
LEFT JOIN m    ON m.lims_data_key = b.lims_data_key
LEFT JOIN ledg ON ledg.ledg_key = b.prdlst_mnfctur_sttemnt_no
LEFT JOIN bsn  ON bsn.bsn_lcns_ledg_no = b.bsn_lcns_ledg_no
LEFT JOIN cl   ON cl.prdlst_cd = COALESCE(b.mapped_prdlst_code, b.prdlst_code)
LEFT JOIN s    ON s.rcept_no = bkey.s_key
LEFT JOIN co   ON co.ovsmnfst_cd = s.ovsmnfst_cd
LEFT JOIN code c40 ON c40.cmmn_cl_code = 'ZIM40' AND c40.cmmn_code = m.cnvync_sttus_tp_unit_code
LEFT JOIN code c08 ON c08.cmmn_cl_code = 'ZIM08' AND c08.cmmn_code = m.cnvync_mth_code
LEFT JOIN code c11 ON c11.cmmn_cl_code = 'ZIM11' AND c11.cmmn_code = m.splore_sttus_code
"""
    df = fetch(conn, sql, "02_attr")
    assert df["lims_data_key"].is_unique, "02_attr: lims_data_key 중복 발생"
    to_parquet(df, "02_attr.parquet")


HAV = ("6371 * acos(LEAST(1.0, cos(radians(p.lat)) * cos(radians(s.lat)) * cos(radians(s.lon) - radians(p.lon))"
       " + sin(radians(p.lat)) * sin(radians(s.lat))))")


def step_weather(conn):
    """3-1 최근접 관측소 / 3-2 ASOS 일자료 / 3-3 NAS 일자료"""
    if not exists("31_stn_map.parquet"):
        sql = f"""
WITH pts AS (
    SELECT DISTINCT lat::double precision AS lat, lon::double precision AS lon
    FROM body WHERE geo_route = 'domestic' AND lat IS NOT NULL AND lon IS NOT NULL),
asos AS (
    SELECT st.stn_id::text AS stn_id, st.lat::double precision AS lat, st.lon::double precision AS lon,
           st.ht, st.basin, st.fct_id, st.legaldong_code
    FROM api_data_info.kma_ground_obs_station_info st
    WHERE st.stn_id::text IN (SELECT DISTINCT stn_id FROM api_data_info.kma_asos_daily_info)),
nas AS (
    SELECT st.obsr_spot_code, st.instl_la::double precision AS lat, st.instl_lo::double precision AS lon,
           st.clmt_zone_code, st.legaldong_code
    FROM api_data_info.nas_agri_weather_obs_station st
    WHERE st.obsr_spot_code IN (SELECT DISTINCT obsr_spot_code FROM api_data_info.nas_agri_weather_daily_info))
SELECT p.lat, p.lon,
       a.stn_id AS asos_stn_id, ROUND(a.dist_km::numeric, 2) AS asos_dist_km, a.ht AS asos_ht, a.basin AS asos_basin,
       a.fct_id AS asos_fct_id, a.legaldong_code AS asos_legaldong_code,
       n.obsr_spot_code AS nas_stn_id, ROUND(n.dist_km::numeric, 2) AS nas_dist_km, n.clmt_zone_code
FROM pts p
CROSS JOIN LATERAL (SELECT s.stn_id, s.ht, s.basin, s.fct_id, s.legaldong_code, {HAV} AS dist_km FROM asos s ORDER BY dist_km LIMIT 1) a
CROSS JOIN LATERAL (SELECT s.obsr_spot_code, s.clmt_zone_code, {HAV} AS dist_km FROM nas s ORDER BY dist_km LIMIT 1) n
"""
        to_parquet(fetch(conn, sql, "31_stn_map"), "31_stn_map.parquet")
    else:
        log("skip 31_stn_map (exists)")

    rng = "(SELECT MIN(mnfctur_de_or_fallback) - 30 FROM body) AND (SELECT MAX(mnfctur_de_or_fallback) FROM body)"
    num = "~ '^\\s*-?[0-9]+(\\.[0-9]+)?\\s*$'"
    if not exists("32_asos_daily.parquet"):
        sql = f"""
SELECT stn_id, meas_date,
       avg_ta, max_ta, min_ta,
       sum_rn, n99_rn, hr_1_max_rn, mi_10_max_rn,
       avg_ws, max_ws, max_ws_wd, max_ins_ws, max_ins_ws_wd, max_ws1, hr_24_sum_rws,
       CASE WHEN sum_sshr {num} THEN sum_sshr::numeric END AS sum_sshr_num,
       CASE WHEN ss_dur   {num} THEN ss_dur::numeric   END AS ss_dur_num,
       sum_gsr, hr_1_max_icsr,
       avg_ts, min_tg, avg_cm5_te, avg_cm10_te, avg_cm20_tes, avg_cm30_te,
       avg_m05_te, avg_m10_te, avg_m15_te, avg_m30_te, avg_m50_te
FROM api_data_info.kma_asos_daily_info
WHERE meas_date BETWEEN {rng}
"""
        to_parquet(fetch(conn, sql, "32_asos_daily"), "32_asos_daily.parquet")
    else:
        log("skip 32_asos_daily (exists)")

    if not exists("33_nas_daily.parquet"):
        sql = f"""
SELECT obsr_spot_code, meas_date,
       temp, hghst_artmp, lowst_artmp, rn, wind, max_wind, widdir,
       CASE WHEN sun_time {num} THEN sun_time::numeric END AS sun_time_num,
       srqty, gr_temp, soil_temp, soil_wt
FROM api_data_info.nas_agri_weather_daily_info
WHERE meas_date BETWEEN {rng}
"""
        to_parquet(fetch(conn, sql, "33_nas_daily"), "33_nas_daily.parquet")
    else:
        log("skip 33_nas_daily (exists)")


def step_soil(conn):
    if not exists("41_soil_dong_year.parquet"):
        sql = """
SELECT legal_dong_code, NULLIF(TRIM(any_year), '')::int AS yr, COUNT(*) AS n_samples,
       AVG(acid) AS soil_ph_mean, STDDEV(acid) AS soil_ph_std, AVG(vldpha) AS vldpha_mean, AVG(om) AS om_mean
FROM api_data_info.soil_ph_info
WHERE acid IS NOT NULL
GROUP BY legal_dong_code, any_year"""
        to_parquet(fetch(conn, sql, "41_soil_dong_year"), "41_soil_dong_year.parquet")
    if not exists("42_soil_sgg_year.parquet"):
        sql = """
SELECT LEFT(legal_dong_code, 5) AS sgg_cd, NULLIF(TRIM(any_year), '')::int AS yr, COUNT(*) AS n_samples,
       AVG(acid) AS soil_ph_mean, AVG(vldpha) AS vldpha_mean, AVG(om) AS om_mean
FROM api_data_info.soil_ph_info
WHERE acid IS NOT NULL
GROUP BY LEFT(legal_dong_code, 5), any_year"""
        to_parquet(fetch(conn, sql, "42_soil_sgg_year"), "42_soil_sgg_year.parquet")
    if not exists("43_global_soil_ph.parquet"):
        to_parquet(fetch(conn, "SELECT iso3, nation_code, name_ko, sub_region, ph_mean, ph_std FROM api_data_info.global_soil_ph", "43_global_soil_ph"),
                   "43_global_soil_ph.parquet")


def step_air(conn):
    if exists("50_air_sgg_daily.parquet"):
        log("skip 50_air (exists)"); return
    # 측정소 법정동코드 형식 기록
    fmt = fetch(conn, "SELECT LENGTH(legaldong_code) AS len, COUNT(*) AS n FROM api_data_info.keco_airkorea_obs_station GROUP BY 1 ORDER BY 1", "keco_code_len")
    LOG["keco_legaldong_len"] = fmt.to_dict("records")
    sql = """
WITH st AS (
    SELECT DISTINCT ON (station_nm) station_nm, legaldong_code
    FROM api_data_info.keco_airkorea_obs_station
    WHERE legaldong_code IS NOT NULL
    ORDER BY station_nm, (COALESCE(use_yn, 'Y') = 'Y') DESC, chg_dt DESC NULLS LAST)
SELECT LEFT(st.legaldong_code, 5) AS sgg_cd, f.msur_dt,
       AVG(f.no2_value) AS no2_mean, AVG(f.co_value) AS co_mean, AVG(f.so2_value) AS so2_mean, COUNT(*) AS n_stations
FROM api_data_info.keco_airkorea_fine_dust_info f
JOIN st ON st.station_nm = f.station_nm
WHERE f.msur_dt BETWEEN (SELECT MIN(mnfctur_de_or_fallback) - 30 FROM body) AND (SELECT MAX(mnfctur_de_or_fallback) FROM body)
GROUP BY LEFT(st.legaldong_code, 5), f.msur_dt"""
    to_parquet(fetch(conn, sql, "50_air_sgg_daily"), "50_air_sgg_daily.parquet")


def step_global(conn):
    if not exists("60_global_daily.parquet"):
        sql = """
SELECT alpha_3, name_ko, date,
       temperature_mean, temperature_max, temperature_min, precipitation_sum,
       relative_humidity_mean, soil_moisture_mean, sunshine_duration, wind_speed_mean, shortwave_radiation_sum
FROM api_data_info.global_weather_info
WHERE date BETWEEN (SELECT MIN(mnfctur_de_or_fallback) - 30 FROM body WHERE geo_route = 'overseas')
              AND (SELECT MAX(mnfctur_de_or_fallback) FROM body WHERE geo_route = 'overseas')"""
        to_parquet(fetch(conn, sql, "60_global_daily"), "60_global_daily.parquet")
    if not exists("61_nation_codes.parquet"):
        sql = """
SELECT import_nation_code, prdctn_nation_code, orgplce_nation_code, COUNT(*) AS n
FROM body WHERE geo_route = 'overseas'
GROUP BY 1, 2, 3 ORDER BY n DESC"""
        to_parquet(fetch(conn, sql, "61_nation_codes"), "61_nation_codes.parquet")


def step_codes(conn):
    if not exists("70_code_zim.parquet"):
        sql = """
SELECT cmmn_cl_code, cmmn_code, cmmn_code_nm, cmmn_code_dc, use_at
FROM ba_data.t_cm_cmmn_code
WHERE cmmn_cl_code IN ('ZIM08', 'ZIM11', 'ZIM40')
ORDER BY cmmn_cl_code, sort_ordr, cmmn_code"""
        to_parquet(fetch(conn, sql, "70_code_zim"), "70_code_zim.parquet")


STEPS = {"0": step_body, "2": step_attr, "3": step_weather, "4": step_soil, "5": step_air, "6": step_global, "7": step_codes}

if __name__ == "__main__":
    want = sys.argv[1:] or list(STEPS)
    if "0" not in want:
        want = ["0"] + want  # body TEMP 테이블은 항상 필요
    conn = connect(readonly=False)  # TEMP 테이블 생성을 위해 read-only 해제 (실제 테이블은 SELECT 만)
    t0 = time.time()
    for k in want:
        log(f"===== step {k} =====")
        STEPS[k](conn)
    conn.close()
    LOG["total_sec"] = round(time.time() - t0)
    with open(os.path.join(OUT, "collect_log.json"), "w", encoding="utf-8") as f:
        json.dump(LOG, f, ensure_ascii=False, indent=1, default=str)
    log(f"ALL DONE in {time.time() - t0:,.0f}s")
