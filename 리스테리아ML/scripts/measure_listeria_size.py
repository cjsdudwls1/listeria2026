import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path
import psycopg
from dotenv import dotenv_values

# ready_data.hazard_preprocessed_data 의 hazard_label='리스테리아' 행과 그 키로 연결되는 행만 골라
# 행 수와 DB 행 크기 합(sum(pg_column_size(t.*)))을 잰다. 읽기 전용 세션, SELECT 만.
# 사용: py measure_listeria_size.py
ENV_PATH = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\.env")
H = "ready_data.hazard_preprocessed_data"
KEYS = "(select distinct {col} from " + H + " where hazard_label = '리스테리아' and {col} is not null)"

QUERIES = [
    (H + " (hazard_label='리스테리아')",
     f"select count(*), sum(pg_column_size(t.*)) from {H} t where hazard_label = '리스테리아'"),
    ("staging_data.export_body_listeria (전체)",
     "select count(*), sum(pg_column_size(t.*)) from staging_data.export_body_listeria t"),
    ("ba_data.m_org_lims_data_info (lims_data_key)",
     "select count(*), sum(pg_column_size(t.*)) from ba_data.m_org_lims_data_info t "
     "where t.lims_data_key in " + KEYS.format(col="lims_data_key")),
    ("ba_data.s_org_lims_data_info (rcept_no = cntc_sys_key2)",
     "select count(*), sum(pg_column_size(t.*)) from ba_data.s_org_lims_data_info t "
     "where t.rcept_no in " + KEYS.format(col="cntc_sys_key2")),
    ("ba_data.s_org_lims_data_info (위 + label='리스테리아')",
     "select count(*), sum(pg_column_size(t.*)) from ba_data.s_org_lims_data_info t "
     "where t.label = '리스테리아' and t.rcept_no in " + KEYS.format(col="cntc_sys_key2")),
    ("ba_data.t_prdlst_mnf_ledg (prdlst_report_no 또는 prdlst_report_ledg_no = prdlst_mnfctur_sttemnt_no, 원문 일치)",
     "select count(*), sum(pg_column_size(t.*)) from ba_data.t_prdlst_mnf_ledg t "
     "where t.prdlst_report_no in " + KEYS.format(col="prdlst_mnfctur_sttemnt_no") +
     " or t.prdlst_report_ledg_no in " + KEYS.format(col="prdlst_mnfctur_sttemnt_no")),
    ("ba_data.t_bsn_lcns_ledg (bsn_lcns_ledg_no)",
     "select count(*), sum(pg_column_size(t.*)) from ba_data.t_bsn_lcns_ledg t "
     "where t.bsn_lcns_ledg_no in " + KEYS.format(col="bsn_lcns_ledg_no")),
]

e = dotenv_values(ENV_PATH)
with psycopg.connect(host=e["DB_HOST"], port=int(e.get("DB_PORT") or 5432), dbname=e["DB_NAME"],
                     user=e["DB_USER"], password=e.get("DB_PASSWORD") or None,
                     options="-c default_transaction_read_only=on -c statement_timeout=480000",
                     application_name="listeria_measure_size", connect_timeout=15) as conn:
    with conn.cursor() as cur:
        cur.execute("show default_transaction_read_only")
        assert cur.fetchone()[0] == "on", "읽기 전용 세션이 아님"
        for name, sql in QUERIES:
            t0 = time.time()
            try:
                cur.execute(sql)
                n, b = cur.fetchone()
                print(f"{name}\n    행 {n:,} | {(b or 0)/1024**2:,.1f} MB | {time.time()-t0:.0f}초", flush=True)
            except Exception as ex:
                conn.rollback()
                print(f"{name}\n    실패: {type(ex).__name__}: {str(ex).splitlines()[0]}", flush=True)
