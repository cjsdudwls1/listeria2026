# 읽기 전용 DB 탐색 3: aw_grade·aw_match_level 을 만드는 함수/규칙 재현
import importlib.util
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(r"C:\Users\NT751\Desktop\천영진")
spec = importlib.util.spec_from_file_location("ex", ROOT / "리스테리아ML2/exp/final/scripts/s0_db_extract.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
ex.STATEMENT_TIMEOUT_MS = 300000
conn = ex.connect(dotenv_values(ROOT / "리스테리아ML/.env"))
cur = conn.cursor()


def q(sql, args=None):
    cur.execute(sql, args)
    return cur.fetchall()


print("== aw_match_level·aw_grade·prdlst_hierarchy 를 쓰는 함수")
for r in q("select n.nspname, p.proname, length(p.prosrc) from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
           "where p.prosrc ilike '%%aw_match_level%%' or p.prosrc ilike '%%aw_grade%%' or p.prosrc ilike '%%prdlst_hierarchy%%'"):
    print("  ", r)
print("== 뷰 정의에 aw_match_level")
for r in q("select schemaname, viewname from pg_views where definition ilike '%%aw_match_level%%'"):
    print("  ", r)
print("== hazard_label 값")
print("  ", q("select hazard_label, count(*) from staging_data.hazard_preprocessed_data group by 1 order by 2 desc"))

H = "staging_data.hazard_preprocessed_data"
# 규칙 재현: 시료 코드(후보 칼럼)의 계층 조상 중 t_aw_cl 에 행이 있는 가장 깊은 수준 → aflatoxin_aw
for key in ("prdlst_code", "mapped_prdlst_code", "train_prdlst_code"):
    sql = f"""
    with s as (
      select h.aw_match_level, h.aw_grade, h.{key} as k from {H} h where h.hazard_label ilike '%%listeria%%' or true
    ), j as (
      select s.*, ph.own_lv, ph.cd_lv1, ph.cd_lv2, ph.cd_lv3, ph.cd_lv4,
        (select a.aflatoxin_aw from ba_data.t_aw_cl a where a.prdlst_cd = s.k) as g_own,
        (select a.aflatoxin_aw from ba_data.t_aw_cl a where a.prdlst_cd = ph.cd_lv4) as g4,
        (select a.aflatoxin_aw from ba_data.t_aw_cl a where a.prdlst_cd = ph.cd_lv3) as g3,
        (select a.aflatoxin_aw from ba_data.t_aw_cl a where a.prdlst_cd = ph.cd_lv2) as g2
      from s left join staging_data.prdlst_hierarchy ph on ph.prdlst_cd = s.k
    )
    select aw_match_level, count(*) as n,
      count(*) filter (where ph_found) as in_hier,
      count(*) filter (where aw_grade::text = g_own) as eq_own,
      count(*) filter (where aw_grade::text = g4) as eq_l4,
      count(*) filter (where aw_grade::text = g3) as eq_l3,
      count(*) filter (where aw_grade::text = g2) as eq_l2
    from (select j.*, (own_lv is not null) as ph_found from j) z group by 1 order by 1
    """
    print(f"== 키 {key}: aw_match_level 별 행 · 계층표에 있음 · aw_grade = t_aw_cl.aflatoxin_aw(자기·4·3·2단계 조상)")
    for r in q(sql):
        print("  ", r)
conn.rollback(); conn.close()
