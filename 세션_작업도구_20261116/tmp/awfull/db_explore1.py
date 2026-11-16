# 읽기 전용 DB 탐색 1: t_aw_cl·t_aw_cl_full·t_prdlst_cl 설명, t_aw_cl 을 쓰는 뷰·함수, aw·품목 코드 칼럼이 있는 표
import importlib.util, sys
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(r"C:\Users\NT751\Desktop\천영진")
spec = importlib.util.spec_from_file_location("ex", ROOT / "리스테리아ML2/exp/final/scripts/s0_db_extract.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
ex.STATEMENT_TIMEOUT_MS = 120000
conn = ex.connect(dotenv_values(ROOT / "리스테리아ML/.env"))
cur = conn.cursor()


def q(sql, args=None, limit=200):
    cur.execute(sql, args)
    rows = cur.fetchall()
    return rows[:limit]


print("== 표 설명")
for s, t in (("ba_data", "t_aw_cl"), ("ba_data", "t_aw_cl_full"), ("ba_data", "t_prdlst_cl")):
    r = q("select obj_description(c.oid, 'pg_class'), c.relkind from pg_class c join pg_namespace n on n.oid = c.relnamespace "
          "where n.nspname = %s and c.relname = %s", (s, t))
    print(f"{s}.{t}:", r)
    cols = q("select a.attname, col_description(a.attrelid, a.attnum) from pg_attribute a join pg_class c on c.oid = a.attrelid "
             "join pg_namespace n on n.oid = c.relnamespace where n.nspname = %s and c.relname = %s and a.attnum > 0 "
             "and not a.attisdropped order by a.attnum", (s, t))
    print("   칼럼 설명:", cols)

print("== 't_aw_cl' 이 들어간 뷰·구체화 뷰 정의")
for r in q("select schemaname, viewname from pg_views where definition ilike '%%t_aw_cl%%' "
           "union all select schemaname, matviewname from pg_matviews where definition ilike '%%t_aw_cl%%'"):
    print("  ", r)
print("== 't_aw_cl' 이 들어간 함수")
for r in q("select n.nspname, p.proname from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
           "where p.prosrc ilike '%%t_aw_cl%%'"):
    print("  ", r)
print("== 외래 키(t_prdlst_cl·t_aw_cl 참조)")
for r in q("select conrelid::regclass::text, confrelid::regclass::text, pg_get_constraintdef(oid) from pg_constraint "
           "where contype = 'f' and (confrelid::regclass::text ilike '%%t_prdlst_cl%%' or confrelid::regclass::text ilike '%%t_aw_cl%%' "
           "or conrelid::regclass::text ilike '%%t_aw_cl%%')"):
    print("  ", r)

print("== aw 값 칼럼이 있는 표(칼럼 이름 aw·aflatoxin_aw·*_aw·aw_*·water_activity)")
for r in q("select table_schema, table_name, column_name, data_type from information_schema.columns "
           "where table_schema not in ('pg_catalog','information_schema') and (column_name in ('aw','aflatoxin_aw','water_activity') "
           "or column_name ilike '%%\\_aw' or column_name ilike 'aw\\_%%') order by 1,2,3"):
    print("  ", r)

print("== 품목 코드 칼럼(이름에 prdlst 또는 prdlst_c) 있는 표")
for r in q("select table_schema, table_name, string_agg(column_name, ', ' order by column_name) from information_schema.columns "
           "where table_schema not in ('pg_catalog','information_schema') and column_name ilike '%%prdlst%%' "
           "group by 1,2 order by 1,2", limit=400):
    print("  ", r)
conn.rollback(); conn.close()
