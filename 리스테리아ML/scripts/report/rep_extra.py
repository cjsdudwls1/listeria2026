# -*- coding: utf-8 -*-
"""보고서 보강용: LIMS 원본 품목코드·품목명, 수질오염원·식품수분정보·유통단계조사 테이블 상태 (SELECT 만, 값 변경 없음)"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import pandas as pd
from db_utils import connect
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
conn = connect()
def q(sql, p=None):
    with conn.cursor() as cur:
        cur.execute(sql, p)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'judge_result'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'cntc_sys_key2', 'mapped_prdlst_code', 'prdlst_code'])
b = body.merge(db, on='lims_data_key', how='left')
OUT = {}
m = q("SELECT lims_data_key, prdlst_code, prdlst_nm FROM ba_data.m_org_lims_data_info WHERE lims_data_key = ANY(%s)", (b['lims_data_key'].tolist(),))
x = b.merge(m, on='lims_data_key', how='left', suffixes=('', '_m'))
def pres(s): return s.notna() & (s.astype('string').str.strip() != '').fillna(False)
for c in ['prdlst_code_m', 'prdlst_nm']:
    p = pres(x[c])
    OUT[f'm_org.{c}'] = {'domestic': round(float(p[x.src_route == 'domestic'].mean() * 100), 1), 'overseas': round(float(p[x.src_route == 'overseas'].mean() * 100), 1),
                         'detected': round(float(p[x.judge_result == '검출'].mean() * 100), 1)}
keys = sorted(set(b['cntc_sys_key2'].dropna().astype(str)))
so = q("SELECT rcept_no, BOOL_OR(NULLIF(TRIM(prdlst_cd), '') IS NOT NULL) AS has_cd, BOOL_OR(NULLIF(TRIM(prdlst_nm), '') IS NOT NULL) AS has_nm FROM ba_data.s_org_lims_data_info WHERE rcept_no = ANY(%s) GROUP BY rcept_no", (keys,))
y = b.merge(so, left_on='cntc_sys_key2', right_on='rcept_no', how='left')
for c in ['has_cd', 'has_nm']:
    p = y[c].fillna(False).astype(bool)
    OUT[f's_org.{c}'] = {'domestic': round(float(p[y.src_route == 'domestic'].mean() * 100), 1), 'overseas': round(float(p[y.src_route == 'overseas'].mean() * 100), 1),
                         'detected': round(float(p[y.judge_result == '검출'].mean() * 100), 1)}
cl = q("SELECT prdlst_cd, kor_nm FROM ba_data.t_prdlst_cl")
b['code_use'] = b['mapped_prdlst_code'].fillna(b['prdlst_code'])
z = b.merge(cl.drop_duplicates('prdlst_cd'), left_on='code_use', right_on='prdlst_cd', how='left')
p = pres(z['kor_nm'])
OUT['t_prdlst_cl.kor_nm'] = {'domestic': round(float(p[z.src_route == 'domestic'].mean() * 100), 1), 'overseas': round(float(p[z.src_route == 'overseas'].mean() * 100), 1), 'detected': round(float(p[z.judge_result == '검출'].mean() * 100), 1)}
OUT['mapped_code_null_rows'] = int(db.loc[db.lims_data_key.isin(body.lims_data_key), 'mapped_prdlst_code'].isna().sum())
wp = q("""SELECT COUNT(*) n, COUNT(crd_info_x) n_xy, COUNT(clsbiz_ymd) n_close, MIN(lcpmt_ymd)::text lc_min, MAX(lcpmt_ymd)::text lc_max, MAX(collected_at)::text col,
                 COUNT(*) FILTER (WHERE lcpmt_ymd <= DATE '2026-06-30' AND (clsbiz_ymd IS NULL OR clsbiz_ymd >= DATE '2015-04-07')) n_study,
                 COUNT(NULLIF(TRIM(road_nm_addr), '')) n_road, COUNT(NULLIF(TRIM(lotno_addr), '')) n_lot
          FROM api_data_info.water_pollution""")
OUT['water_pollution'] = wp.iloc[0].to_dict()
OUT['water_pollution_status'] = dict(q("SELECT sals_stts_nm, COUNT(*) n FROM api_data_info.water_pollution GROUP BY 1 ORDER BY 2 DESC").values.tolist())
fmi = q("SELECT COUNT(*) n, COUNT(NULLIF(TRIM(water_activate), '')) n_aw, COUNT(NULLIF(TRIM(moisture_content), '')) n_mc, COUNT(NULLIF(TRIM(water_activate_source), '')) n_aw_src FROM ba_data_info.food_moisture_info")
OUT['food_moisture_info'] = fmi.iloc[0].to_dict()
dse = q("""SELECT COUNT(*) n, COUNT(NULLIF(TRIM(cstdy_tp), '')) n_temp, COUNT(NULLIF(TRIM(frmlc_shap), '')) n_pack,
                  COUNT(*) FILTER (WHERE frmlc_shap ILIKE '%%진공%%') n_vac, MIN(reg_dtm)::date::text mn, MAX(reg_dtm)::date::text mx
           FROM ba_data_info.distb_step_exam""")
OUT['distb_step_exam'] = dse.iloc[0].to_dict()
OUT['soil_pnu'] = q("SELECT COUNT(DISTINCT pnu_cd) n_pnu FROM api_data_info.soil_ph_info").iloc[0].to_dict()
OUT['pick_de_domestic_rows'] = None
m2 = pd.read_parquet(C + r"\02a_m_org.parquet", columns=['lims_data_key', 'pick_de'])
xx = body.merge(m2, on='lims_data_key', how='left')
OUT['pick_de_domestic_rows'] = int(xx.loc[xx.src_route == 'domestic', 'pick_de'].notna().sum())
bd = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'date_basis'])
OUT['date_basis'] = body.merge(bd, on='lims_data_key')['date_basis'].value_counts().to_dict()
conn.close()
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\rep_extra.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
