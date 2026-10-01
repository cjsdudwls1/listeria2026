# -*- coding: utf-8 -*-
"""staging_data.export_body_listeria 의 result_value_num / judge_stdr / judge_stdr_num 정의 확인 (읽기 전용).

2026-09-22 작업기록 참조. 실행: py 리스테리아ML/scripts/db_export_3cols_정의확인.py
- 세 컬럼은 정의서(테이블명세서(모델팀)/테이블정의서.xlsx)에 없고 DB 코멘트도 비어 있다.
- 정의는 DB 함수 staging_data.build_export_qual(리스테리아 등 정성 항목용 export 표 생성 함수) 소스에 있다:
    result_value_num = round(판정 뷰 "수치" 또는 ba_data.jz_num(result_value,'micro'), 4)
    judge_stdr       = staging_data.hazard_preprocessed_data.mxmm_value
    judge_stdr_num   = round(ba_data.jz_stdr(mxmm_value), 4)
- 아래는 그 정의로 다시 계산해 실제 값과 전수 대조한다.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import psycopg

PGPASS = os.path.join(os.environ.get("APPDATA", ""), "postgresql", "pgpass.conf")
HOST = "10.31.72.23"


def pgpass_entry(host=HOST):
    with open(PGPASS, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace("\\:", "\x00").split(":")
            if len(parts) < 5:
                continue
            h, p, d, u = (x.replace("\x00", ":") for x in parts[:4])
            if h in (host, "*"):
                return {"host": host, "port": 5432 if p == "*" else int(p), "dbname": d, "user": u}
    raise RuntimeError("pgpass.conf 항목 없음")


e = pgpass_entry()
conn = psycopg.connect(host=e["host"], port=e["port"], dbname=e["dbname"], user=e["user"],
                       options="-c default_transaction_read_only=on", application_name="listeria_colcheck",
                       connect_timeout=15)
conn.autocommit = True


def q(sql, title, raw=False, limit=None):
    print("\n### " + title, flush=True)
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    if raw:
        for r in rows:
            print(r[0])
        return rows
    print(" | ".join(cols))
    for r in (rows if limit is None else rows[:limit]):
        print(" | ".join("" if v is None else str(v) for v in r))
    if limit is not None and len(rows) > limit:
        print(f"... ({len(rows)} rows total)")
    return rows


q("show default_transaction_read_only", "read-only 확인")

q("""
select table_schema, table_name, column_name, data_type
from information_schema.columns
where column_name in ('result_value_num','judge_stdr','judge_stdr_num') order by 1,2,3
""", "1. 세 컬럼을 가진 테이블 (staging_data.export_body_* 뿐)")

q("""
select n.nspname||'.'||c.relname as tbl, a.attname, coalesce(col_description(c.oid,a.attnum),'(코멘트 없음)') as comment
from pg_attribute a join pg_class c on c.oid=a.attrelid join pg_namespace n on n.oid=c.relnamespace
where a.attnum>0 and not a.attisdropped and a.attname in ('result_value_num','judge_stdr','judge_stdr_num')
order by 1,2
""", "2. 컬럼 코멘트")

q("""
select pg_get_functiondef(p.oid)
from pg_proc p join pg_namespace n on n.oid=p.pronamespace
where n.nspname='staging_data' and p.proname='build_export_qual'
""", "3. staging_data.build_export_qual 정의 (세 컬럼의 계산식)", raw=True)

for fn in ("jz_num", "jz_stdr", "jz_norm"):
    q(f"""
    select pg_get_functiondef(p.oid)
    from pg_proc p join pg_namespace n on n.oid=p.pronamespace
    where n.nspname='ba_data' and p.proname='{fn}'
    """, f"4. ba_data.{fn} 정의", raw=True)

q("""
select src_route, count(*) as n,
       count(result_value_num) as result_value_num_nn, count(judge_stdr) as judge_stdr_nn, count(judge_stdr_num) as judge_stdr_num_nn
from staging_data.export_body_listeria group by 1 order by 1
""", "5. 값 있는 행 수")

q("""
select count(*) as n_rows,
  sum(case when result_value_num is not distinct from round(ba_data.jz_num(result_value,'micro'),4) then 1 else 0 end) as same,
  sum(case when result_value_num is distinct from round(ba_data.jz_num(result_value,'micro'),4) then 1 else 0 end) as differ
from staging_data.export_body_listeria
""", "6. result_value_num = round(ba_data.jz_num(result_value,'micro'),4) 전수 재현")

q("""
select count(*) as joined,
  sum(case when e.judge_stdr is not distinct from h.mxmm_value then 1 else 0 end) as same,
  sum(case when e.judge_stdr is distinct from h.mxmm_value then 1 else 0 end) as differ
from staging_data.export_body_listeria e
join staging_data.hazard_preprocessed_data h on h.lims_data_key=e.lims_data_key and h.hazard_label='리스테리아'
""", "7. judge_stdr = staging_data.hazard_preprocessed_data.mxmm_value 전수 재현")

q("""
select count(*) as n_judge_stdr,
  sum(case when judge_stdr_num is not distinct from round(ba_data.jz_stdr(judge_stdr),4) then 1 else 0 end) as same,
  sum(case when judge_stdr_num is distinct from round(ba_data.jz_stdr(judge_stdr),4) then 1 else 0 end) as differ
from staging_data.export_body_listeria where judge_stdr is not null
""", "8. judge_stdr_num = round(ba_data.jz_stdr(judge_stdr),4) 재현")

q("""
select result_value, result_value_num, count(*) as n
from staging_data.export_body_listeria where result_value_num is not null
group by 1,2 order by n desc
""", "9. result_value_num 이 있는 행의 (result_value, result_value_num)")

q("""
select judge_stdr, judge_stdr_num, src_route, count(*) as n
from staging_data.export_body_listeria where judge_stdr is not null
group by 1,2,3 order by n desc
""", "10. judge_stdr 값 분포")

q("""
select m.stndrd_value, m.stdr_se_code, m.jdgmnt_fom_code, m.mxmm_value_se_code, m.mxmm_value, e.judge_stdr, e.judge_stdr_num, count(*) as n
from staging_data.export_body_listeria e
join ba_data.m_org_lims_data_info m on m.lims_data_key=e.lims_data_key
where e.src_route='domestic' and e.judge_stdr is not null
group by 1,2,3,4,5,6,7 order by n desc
""", "11. 국내 judge_stdr 64행의 ba_data.m_org_lims_data_info 규격 컬럼")

q("""
with x as (
  select e.judge_stdr, h.cntc_sys_key2
  from staging_data.export_body_listeria e
  join ready_data.hazard_preprocessed_data h on h.lims_data_key=e.lims_data_key and h.hazard_label='리스테리아'
  where e.src_route='overseas')
select count(*) as overseas_rows,
  sum(case when judge_stdr is null then 1 else 0 end) as judge_stdr_null,
  sum(case when judge_stdr is not null and exists (select 1 from ba_data.s_org_lims_data_info s where s.rcept_no=x.cntc_sys_key2 and s.label='리스테리아' and s.stdr_value=x.judge_stdr) then 1 else 0 end) as judge_stdr_eq_sorg_stdr_value,
  sum(case when judge_stdr is null and exists (select 1 from ba_data.s_org_lims_data_info s where s.rcept_no=x.cntc_sys_key2 and s.label='리스테리아' and nullif(btrim(s.stdr_value),'') is not null) then 1 else 0 end) as null_but_sorg_stdr_value_exists
from x
""", "12. 수입 행: judge_stdr 와 ba_data.s_org_lims_data_info.stdr_value")

conn.close()
