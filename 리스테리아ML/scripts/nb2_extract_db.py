# -*- coding: utf-8 -*-
"""2.ipynb 보조 데이터 추출 (TOXIN_MODEL_PRED, 읽기 전용 SELECT 만).

결과: 리스테리아ML/data/collected/nb2_db_*
  - nb2_db_attr.parquet            ready_data(리스테리아) + 식약처 원본 LIMS(m_org) 행 단위 컬럼
  - nb2_db_s_org_listeria.parquet  수입식품 LIMS 의 리스테리아 행 (선적·입항·검사종류·제조국)
  - nb2_db_ledg.parquet            품목제조대장 (유통기한·보존유통기준·제조조건·포장재질)
  - nb2_db_instt.csv               검사기관 코드별 이름·주소
  - nb2_db_codes.csv               공통코드 (IM02 검사목적, IM03 검사종류, IM08 운반방법, IM11 시료상태, IM40 온도단위, CI07 의뢰시스템 등)
  - nb2_db_queries.sql             사용한 SQL 원문

규칙: 회사 공용 계정이므로 SELECT 만 실행한다. TEMP 테이블·설정 변경 없음.
접속은 db_utils.connect() (세션 옵션 default_transaction_read_only=on, 비밀번호는 pgpass.conf).
사용: py nb2_extract_db.py
"""
import datetime as dt
import decimal
import os
import re
import sys
import time

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from db_utils import connect  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "data", "collected")
os.makedirs(OUT, exist_ok=True)
SQLS = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def clean(df):
    """psycopg 의 Decimal/date 를 parquet 친화 타입으로."""
    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            nn = s.dropna()
            if len(nn) == 0:
                continue
            v = nn.iloc[0]
            if isinstance(v, decimal.Decimal):
                df[c] = pd.to_numeric(s, errors="coerce").astype("float64")
            elif isinstance(v, (dt.datetime, dt.date)):
                df[c] = pd.to_datetime(s, errors="coerce")
    return df


def fetch(conn, label, sql, params=None):
    SQLS[label] = sql
    t0 = time.time()
    with conn.transaction():  # 서버측 커서는 트랜잭션 블록 안에서만 동작 (읽기 전용 트랜잭션)
        with conn.cursor(name=f"c_{label}_{int(t0)}") as cur:
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            chunks = []
            while True:
                rows = cur.fetchmany(100_000)
                if not rows:
                    break
                chunks.append(pd.DataFrame(rows, columns=cols))
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    log(f"{label}: {len(df):,} rows x {df.shape[1]} cols ({time.time() - t0:.0f}s)")
    return clean(df)


SQL_ATTR = """
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
WHERE h.hazard_label = '리스테리아'
"""

SQL_SORG = """
SELECT lims_seq, rcept_no, sample_no, inspct_instt_cd, rcept_de, prcs_de, impt_prpos_cd, prpos_nm,
       prdlst_cd, prdlst_nm, upper_prdlst_nm, prduct_nm_kor, hsk_no, prduct_dvs_cd, dclr_prduct_dvs,
       mnf_nation_cd, mnf_nation_nm, impt_ownr_induty, ovsmnfst_cd,
       inspct_kind, inspct_kind_dtl, inspct_kind_cd,
       distb_bgn_de, distb_end_de, shipng_de, entrpt_de, wrhous_carry_de, bonded_area_cd, inspct_carry_plc,
       inspct_reqst_de, inspct_cmptn_de, test_item, result_value, test_result, stdr_value
FROM ba_data.s_org_lims_data_info
WHERE label = '리스테리아'
"""

SQL_LEDG = """
SELECT prdlst_report_no, prdlst_report_ledg_no, prdlst_cd, prdlst_nm, prms_dt, end_dt,
       pog_daycnt, qlity_mntnc_tmlmt_daycnt, pog_cn, frmlc_mtrqlt, haccp_appn_no, del_yn,
       mnf_cnd_cn, prsrv_distb_stdr_cn, food_hf_ls_cl_cd, induty_cd, last_updt_dtm
FROM ba_data.t_prdlst_mnf_ledg
WHERE prdlst_report_no = ANY(%s::text[])
"""

SQL_INSTT = """
WITH nm AS (
    SELECT h.inspct_instt_code, MAX(COALESCE(m.inspct_instt_nm, h.inspct_instt_nm)) AS instt_nm, COUNT(*) AS n_rows
    FROM ready_data.hazard_preprocessed_data h
    LEFT JOIN ba_data.m_org_lims_data_info m ON m.lims_data_key = h.lims_data_key
    WHERE h.hazard_label = '리스테리아'
    GROUP BY h.inspct_instt_code)
SELECT nm.inspct_instt_code, nm.instt_nm, nm.n_rows, i.instt_eng_nm, i.adres
FROM nm LEFT JOIN ba_data.m_sy_inspct_instt i ON i.inspct_instt_code = nm.inspct_instt_code
ORDER BY nm.n_rows DESC
"""

SQL_CODE = """
SELECT cmmn_cl_code, cmmn_code, cmmn_code_nm, cmmn_code_dc, use_at
FROM ba_data.t_cm_cmmn_code
WHERE cmmn_cl_code IN ('IM02','IM03','IM08','IM11','IM40') OR cmmn_code = ANY(%s::text[])
ORDER BY cmmn_cl_code, cmmn_code
"""


def main():
    conn = connect()  # readonly 기본
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '900s'")  # 세션 한정

    attr = fetch(conn, "attr", SQL_ATTR)
    assert attr["lims_data_key"].is_unique, "attr: lims_data_key 중복"
    attr.to_parquet(os.path.join(OUT, "nb2_db_attr.parquet"), index=False)

    fetch(conn, "s_org", SQL_SORG).to_parquet(os.path.join(OUT, "nb2_db_s_org_listeria.parquet"), index=False)

    keys = sorted(set(attr["prdlst_mnfctur_sttemnt_no"].dropna().astype(str)))
    keys = sorted(set(re.sub(r"\D", "", k) for k in keys) | set(keys))  # 원문 + 숫자만 남긴 번호
    fetch(conn, "ledg", SQL_LEDG, (keys,)).to_parquet(os.path.join(OUT, "nb2_db_ledg.parquet"), index=False)

    fetch(conn, "instt", SQL_INSTT).to_csv(os.path.join(OUT, "nb2_db_instt.csv"), index=False, encoding="utf-8-sig")

    vals = sorted(set(attr["reqest_sys_code"].dropna().astype(str)) | set(attr["stdr_se_code"].dropna().astype(str))
                  | set(attr["jdgmnt_fom_code"].dropna().astype(str)))
    fetch(conn, "codes", SQL_CODE, (vals,)).to_csv(os.path.join(OUT, "nb2_db_codes.csv"), index=False, encoding="utf-8-sig")
    conn.close()

    with open(os.path.join(OUT, "nb2_db_queries.sql"), "w", encoding="utf-8") as f:
        f.write("-- 2.ipynb 보조 데이터 추출 쿼리 (읽기 전용, SELECT 만). 추출일: " + time.strftime("%Y-%m-%d") + "\n")
        for k, v in SQLS.items():
            f.write(f"\n-- ===== {k} =====\n{v.strip()};\n")
    log("done")


if __name__ == "__main__":
    main()
