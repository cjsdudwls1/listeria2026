# 읽기 전용 DB 탐색 4: 갱신 전 t_aw_cl 백업들로 aw_grade·aw_match_level 규칙 재현(리스테리아 행)
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


for t in ("staging_data.t_aw_cl_bak_20261112", "staging_ref.ba_data__t_aw_cl", "staging_ref.ba_data__t_aw_cl_20261112_0941",
          "staging_data.t_aw_cl_xlsx_202611"):
    print(t, "행", q(f"select count(*) from {t}")[0][0], "| 설명", q(f"select obj_description('{t}'::regclass, 'pg_class')")[0][0])

H = "staging_data.hazard_preprocessed_data"
for bak in ("staging_data.t_aw_cl_bak_20261112", "staging_ref.ba_data__t_aw_cl", "ba_data.t_aw_cl"):
    for key in ("mapped_prdlst_code", "prdlst_code"):
        sql = f"""
        with s as (select aw_match_level, aw_grade, {key} as k from {H} where hazard_label = '리스테리아'),
        j as (
          select s.*, ph.own_lv,
            a0.aflatoxin_aw g0, a4.aflatoxin_aw g4, a3.aflatoxin_aw g3, a2.aflatoxin_aw g2
          from s left join staging_data.prdlst_hierarchy ph on ph.prdlst_cd = s.k
          left join {bak} a0 on a0.prdlst_cd = s.k
          left join {bak} a4 on a4.prdlst_cd = ph.cd_lv4
          left join {bak} a3 on a3.prdlst_cd = ph.cd_lv3
          left join {bak} a2 on a2.prdlst_cd = ph.cd_lv2
        ), p as (
          select aw_match_level, aw_grade,
            case when g0 is not null then 'lv' || own_lv when g4 is not null then 'lv4' when g3 is not null then 'lv3'
                 when g2 is not null then 'lv2' else '미매칭' end as lev,
            coalesce(g0, g4, g3, g2, '1') as grd
          from j
        )
        select count(*), count(*) filter (where lev = aw_match_level), count(*) filter (where grd = aw_grade::text),
               count(*) filter (where lev = aw_match_level and grd = aw_grade::text)
        from p
        """
        n, lv_ok, g_ok, both = q(sql)[0]
        print(f"[{bak} · 키 {key}] 리스테리아 {n:,}행: 단계 일치 {lv_ok:,} ({lv_ok/n:.2%}) · 등급 일치 {g_ok:,} ({g_ok/n:.2%}) · 둘 다 {both:,} ({both/n:.2%})")
print("== 리스테리아 행 aw_match_level 분포:", q(f"select aw_match_level, count(*) from {H} where hazard_label='리스테리아' group by 1 order by 1"))
conn.rollback(); conn.close()
