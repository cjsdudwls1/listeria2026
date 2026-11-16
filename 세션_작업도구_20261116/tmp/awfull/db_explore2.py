# 읽기 전용 DB 탐색 2: aw_grade·aw_match_level(모델팀 전처리 표), guard 함수, 품목 계층 보조 표
import importlib.util
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(r"C:\Users\NT751\Desktop\천영진")
spec = importlib.util.spec_from_file_location("ex", ROOT / "리스테리아ML2/exp/final/scripts/s0_db_extract.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
ex.STATEMENT_TIMEOUT_MS = 180000
conn = ex.connect(dotenv_values(ROOT / "리스테리아ML/.env"))
cur = conn.cursor()


def q(sql, args=None):
    cur.execute(sql, args)
    return cur.fetchall()


def comments(s, t):
    return q("select a.attname, col_description(a.attrelid, a.attnum) from pg_attribute a join pg_class c on c.oid = a.attrelid "
             "join pg_namespace n on n.oid = c.relnamespace where n.nspname = %s and c.relname = %s and a.attnum > 0 "
             "and not a.attisdropped order by a.attnum", (s, t))


H = "staging_data.hazard_preprocessed_data"
print("== hazard_preprocessed_data 표 설명:", q("select obj_description('staging_data.hazard_preprocessed_data'::regclass, 'pg_class')"))
cm = comments("staging_data", "hazard_preprocessed_data")
print("   칼럼(설명):", [(a, b) for a, b in cm])
print("== aw_match_level × aw_grade 행 수")
for r in q(f"select aw_match_level, aw_grade, count(*) from {H} group by 1,2 order by 1,2"):
    print("  ", r)
print("== guard 함수 소스(앞부분)")
for nm in ("guard_pre", "guard_post"):
    src = q("select p.prosrc from pg_proc p join pg_namespace n on n.oid = p.pronamespace where n.nspname='staging_ref' and p.proname=%s", (nm,))
    print(f"-- {nm}:", (src[0][0][:1500] if src else None))
for t in ("prdlst_hierarchy", "prdlst_hierarchy_fallback", "prdlst_prefix_fallback", "kc_food_type_map"):
    print(f"== staging_data.{t}: 설명", q(f"select obj_description('staging_data.{t}'::regclass, 'pg_class')"),
          "| 칼럼", comments("staging_data", t), "| 행", q(f"select count(*) from staging_data.{t}")[0][0])
    print("   예:", q(f"select * from staging_data.{t} limit 3"))
conn.rollback(); conn.close()
