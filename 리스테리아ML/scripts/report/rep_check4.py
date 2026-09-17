import sys, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import numpy as np, pandas as pd
from db_utils import connect
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
OUT = {}
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'judge_result', 'legal_dong_code'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'mapped_prdlst_code', 'prdlst_nm', 'legal_dong_code'])
b = body.merge(db, on='lims_data_key', how='left', suffixes=('_pq', '_db'))
aw = pd.read_parquet(C + r"\80_t_aw_cl.parquet")
f = pd.read_excel(r"C:\Users\NT751\Desktop\천영진\리서치팀문서\통합_수분활성도_수분함량_식품pH_염도_최종.xlsx", sheet_name='값')
def blank(s): return s.isna() | s.astype(str).str.strip().isin(['', '-', 'nan', '<NA>'])
fm = f.merge(aw, left_on='PRDLST_CD', right_on='prdlst_cd', how='left')
filled = set(fm.loc[blank(fm['수분활성도']) & ~blank(fm['aw']), 'PRDLST_CD'])
x = b.merge(aw[['prdlst_cd', 'aw']], left_on='mapped_prdlst_code', right_on='prdlst_cd', how='left')
has_db = ~blank(x['aw'])
file_basis = has_db & ~x['mapped_prdlst_code'].isin(filled)
dom, ovs, det_dom = x.src_route == 'domestic', x.src_route == 'overseas', (x.src_route == 'domestic') & (x.judge_result == '검출')
OUT['aw_db_basis'] = [round(has_db[dom].mean() * 100, 1), round(has_db[ovs].mean() * 100, 1), round(has_db[det_dom].mean() * 100, 1)]
OUT['aw_file_basis'] = [round(file_basis[dom].mean() * 100, 1), round(file_basis[ovs].mean() * 100, 1), round(file_basis[det_dom].mean() * 100, 1)]
# 부등호 방향 반전
inv = fm[fm['수분활성도'].astype(str).str.strip().str.match(r'^\d+(\.\d+)?\s*<$') & fm['aw'].astype(str).str.strip().str.startswith('<')]
OUT['inverted_codes'] = inv[['KOR_NM', '수분활성도', 'aw']].astype(str).values.tolist()
OUT['inverted_rows'] = int(b['mapped_prdlst_code'].isin(set(inv['PRDLST_CD'])).sum())
# 품목명 매칭
conn = connect()
def q(sql, p=None):
    with conn.cursor() as cur:
        cur.execute(sql, p); return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
fmi = q("SELECT prdlst_nm, water_activate FROM ba_data_info.food_moisture_info")
names = set(fmi['prdlst_nm'].astype(str).str.strip()); names_aw = set(fmi.loc[~blank(fmi['water_activate']), 'prdlst_nm'].astype(str).str.strip())
nm = b['prdlst_nm'].astype(str).str.strip()
OUT['fmi_name_match'] = [round(nm[b.src_route == 'domestic'].isin(names).mean() * 100, 1), round(nm[b.src_route == 'overseas'].isin(names).mean() * 100, 1),
                         round(nm[b.src_route == 'domestic'].isin(names_aw).mean() * 100, 1), round(nm[b.src_route == 'overseas'].isin(names_aw).mean() * 100, 1)]
OUT['pnu'] = q("""SELECT LENGTH(pnu_cd) len, COUNT(*) n_rows, COUNT(DISTINCT pnu_cd) n_distinct, MIN(any_year) y_min, MAX(any_year) y_max FROM api_data_info.soil_ph_info GROUP BY 1 ORDER BY 1""").astype(str).values.tolist()
OUT['pnu_2023'] = q("""SELECT LENGTH(pnu_cd) len, COUNT(*) n, COUNT(DISTINCT pnu_cd) d FROM api_data_info.soil_ph_info WHERE any_year = '2023' GROUP BY 1""").astype(str).values.tolist()
OUT['exam_2026'] = q("SELECT MIN(exam_day)::text, MAX(exam_day)::text FROM api_data_info.soil_ph_info WHERE any_year='2026'").values.tolist()
OUT['pnu_comment'] = q("SELECT col_description(a.attrelid, a.attnum) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='api_data_info' AND c.relname='soil_ph_info' AND a.attname='pnu_cd'").iloc[0, 0]
OUT['dse_temp_meaningful'] = int(q(r"SELECT COUNT(*) FROM ba_data_info.distb_step_exam WHERE cstdy_tp ~ '[0-9]|냉장|냉동|실온|상온|℃|온도'").iloc[0, 0])
OUT['dse_old_docs'] = int(q(r"SELECT COUNT(*) FROM ba_data_info.distb_step_exam WHERE substring(doc_no from '(20[0-9]{2})') < '2015'").iloc[0, 0])
conn.close()
ld = b[b.src_route == 'domestic']
OUT['legal_dong_diff_rows'] = int((ld['legal_dong_code_pq'].astype(str) != ld['legal_dong_code_db'].astype(str)).sum())
OUT['sgg_code_db'] = int(ld['legal_dong_code_db'].astype(str).str.endswith('00000').sum())
kst = pd.read_parquet(C + r"\raw\raw_keco_airkorea_fine_dust_info.parquet", columns=['station_nm', 'msur_dt'])
yr = pd.to_datetime(kst['msur_dt']).dt.year
OUT['air_stations_2015_2026'] = [int(kst.loc[yr == 2015, 'station_nm'].nunique()), int(kst.loc[yr == 2026, 'station_nm'].nunique())]
print(json.dumps(OUT, ensure_ascii=False, indent=1))
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\rep_check4.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
