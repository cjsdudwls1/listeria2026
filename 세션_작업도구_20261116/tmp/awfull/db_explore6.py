# 읽기 전용 DB 탐색 6: DB 방식(자기 코드 → 4·3단계 조상 → 2단계)으로 지금 t_aw_cl 의 aw·ph·na 를 붙이면 연결률(리스테리아 본 표 행)
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


H = "staging_data.hazard_preprocessed_data"
E = "staging_data.export_body_listeria"
cols = [r[0] for r in q("select column_name from information_schema.columns where table_schema='staging_data' "
                        "and table_name='export_body_listeria' order by ordinal_position")]
print("export_body_listeria 칼럼:", cols)
print("judge_result 값(리스테리아):", q(f"select judge_result, count(*) from {H} where hazard_label='리스테리아' group by 1 order by 2 desc"))
A = "ba_data.t_aw_cl"
for c in ("aw", "ph", "na"):
    v = lambda t: f"nullif(nullif(trim({t}.{c}), ''), '-')"
    sql = f"""
    with s as (select h.mapped_prdlst_code k, h.train_prdlst_code t2, h.judge_result jr from {H} h
               join {E} e on e.lims_data_key = h.lims_data_key where h.hazard_label = '리스테리아'),
    hh as (select s.*, coalesce(ph.prdlst_cd, fb.base_cd) kk from s
           left join staging_data.prdlst_hierarchy ph on ph.prdlst_cd = s.k
           left join staging_data.prdlst_hierarchy_fallback fb on fb.prdlst_cd = s.k),
    j as (select hh.*, ph.own_lv, {v('a0')} g0, {v('a4')} g4, {v('a3')} g3, {v('at2')} gt2
          from hh left join staging_data.prdlst_hierarchy ph on ph.prdlst_cd = hh.kk
          left join {A} a0 on a0.prdlst_cd = hh.kk
          left join {A} a4 on a4.prdlst_cd = ph.cd_lv4
          left join {A} a3 on a3.prdlst_cd = ph.cd_lv3
          left join {A} at2 on at2.prdlst_cd = hh.t2)
    select jr,
      count(*) n,
      count(*) filter (where g0 is not null) own,
      count(*) filter (where g0 is null and g4 is not null) l4,
      count(*) filter (where g0 is null and g4 is null and g3 is not null) l3,
      count(*) filter (where g0 is null and g4 is null and g3 is null and gt2 is not null) l2,
      count(*) filter (where coalesce(g0, g4, g3, gt2) is null) miss
    from j group by 1 order by 1
    """
    print(f"== {c}: 판정 · 행 · 자기 코드 값 · 4단계 조상 · 3단계 조상 · 2단계(train_prdlst_code) · 못 붙음")
    for r in q(sql):
        print("  ", r, f"| 붙음 {(r[1]-r[6])/r[1]:.1%}")
conn.rollback(); conn.close()
