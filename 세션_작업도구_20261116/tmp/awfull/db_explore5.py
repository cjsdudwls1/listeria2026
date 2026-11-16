# 읽기 전용 DB 탐색 5: 재현 안 맞는 행 살피기 — 2단계 대신 train_prdlst_code, 계층표 밖 코드는 fallback 표
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
BAK = "staging_data.t_aw_cl_bak_20261112"
base = f"""
with s as (select aw_match_level, aw_grade, mapped_prdlst_code k, train_prdlst_code t2, train_prdlst_code_2 t3, load_batch
           from {H} where hazard_label = '리스테리아'),
h as (select s.*, coalesce(ph.prdlst_cd, fb.base_cd) as kk from s
      left join staging_data.prdlst_hierarchy ph on ph.prdlst_cd = s.k
      left join staging_data.prdlst_hierarchy_fallback fb on fb.prdlst_cd = s.k),
j as (select h.*, ph.own_lv, a0.aflatoxin_aw g0, a4.aflatoxin_aw g4, a3.aflatoxin_aw g3, a2.aflatoxin_aw g2,
             at2.aflatoxin_aw gt2, at3.aflatoxin_aw gt3
      from h left join staging_data.prdlst_hierarchy ph on ph.prdlst_cd = h.kk
      left join {BAK} a0 on a0.prdlst_cd = h.kk
      left join {BAK} a4 on a4.prdlst_cd = ph.cd_lv4
      left join {BAK} a3 on a3.prdlst_cd = ph.cd_lv3
      left join {BAK} a2 on a2.prdlst_cd = ph.cd_lv2
      left join {BAK} at2 on at2.prdlst_cd = h.t2
      left join {BAK} at3 on at3.prdlst_cd = h.t3),
p as (select j.*,
        case when g0 is not null then 'lv' || own_lv when g4 is not null then 'lv4' when g3 is not null then 'lv3'
             when g2 is not null then 'lv2' else '미매칭' end as lev_a,
        coalesce(g0, g4, g3, g2, '1') as grd_a,
        case when g0 is not null then 'lv' || own_lv when g4 is not null then 'lv4' when g3 is not null then 'lv3'
             when gt2 is not null then 'lv2' else '미매칭' end as lev_b,
        coalesce(g0, g4, g3, gt2, '1') as grd_b
      from j)
"""
print("== 안(a) 계층 2단계 · 안(b) 2단계 대신 train_prdlst_code (둘 다 계층표 밖 코드는 fallback base_cd)")
r = q(base + """select count(*),
  count(*) filter (where lev_a = aw_match_level and grd_a = aw_grade::text),
  count(*) filter (where lev_b = aw_match_level and grd_b = aw_grade::text),
  count(*) filter (where grd_b = aw_grade::text) from p""")[0]
print(f"   {r[0]:,}행: (a) 둘 다 일치 {r[1]:,} ({r[1]/r[0]:.2%}) · (b) 둘 다 일치 {r[2]:,} ({r[2]/r[0]:.2%}) · (b) 등급 일치 {r[3]:,} ({r[3]/r[0]:.2%})")
print("== (b) 안 맞는 행: 실제 단계 × 재현 단계 × load_batch (많은 순 15)")
for row in q(base + """select aw_match_level, lev_b, load_batch, count(*) from p
               where not (lev_b = aw_match_level and grd_b = aw_grade::text) group by 1,2,3 order by 4 desc limit 15"""):
    print("  ", row)
print("== 안 맞는 예(실제 lv2 인데 재현은 더 깊은 단계) 5개")
for row in q(base + """select k, kk, t2, t3, own_lv, g0, g4, g3, g2, gt2, aw_match_level, aw_grade, lev_b, grd_b, load_batch
               from p where aw_match_level = 'lv2' and lev_b <> 'lv2' limit 5"""):
    print("  ", row)
print("== load_batch 분포:", q(f"select load_batch, count(*) from {H} where hazard_label='리스테리아' group by 1 order by 1"))
conn.rollback(); conn.close()
