# -*- coding: utf-8 -*-
"""포장 재질 칸 '진공·가스' 글자 실제 예시 + 축산물 검사표 포장재 칸 통계(pg_stats, SELECT만)"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import pandas as pd
from db_utils import connect
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'judge_result'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'prdlst_mnfctur_sttemnt_no'])
ledg = pd.read_parquet(C + r"\02b_ledg.parquet")
lr = ledg.assign(_live=(ledg['del_yn'].fillna('N') == 'N')).sort_values(['prdlst_report_no', '_live', 'last_updt_dtm'], ascending=[True, False, False]).drop_duplicates('prdlst_report_no')
y = body.merge(db, on='lims_data_key', how='left').merge(lr, left_on='prdlst_mnfctur_sttemnt_no', right_on='prdlst_report_no', how='left')
d = y[y.src_route == 'domestic']
fm = d['frmlc_mtrqlt'].astype('string')
OUT = {'진공_top': fm[fm.str.contains('진공', regex=False).fillna(False)].value_counts().head(8).to_dict(),
       '진공_det_rows': int((fm.str.contains('진공', regex=False).fillna(False) & (d['judge_result'].astype(str) == '검출')).sum()),
       '가스질소_top': fm[fm.str.contains('가스|질소|MAP', regex=True).fillna(False)].value_counts().head(6).to_dict(),
       '진공_products': int(d.loc[fm.str.contains('진공', regex=False).fillna(False), 'prdlst_mnfctur_sttemnt_no'].nunique())}
conn = connect()
with conn.cursor() as cur:
    cur.execute("""SELECT tablename, attname, null_frac, n_distinct, left(most_common_vals::text, 120) AS mcv FROM pg_stats
                   WHERE schemaname = 'ba_data_info' AND ((tablename = 'lpsms_lcf_insp_smp' AND attname IN ('pckn_mtrl', 'pckn_unt', 'insp_smp_nm'))
                      OR (tablename = 'lpsms_lcf_insp_rqst' AND attname IN ('mte_rqsts_no', 'insp_rqst_dt', 'wrkshp_gubn_cd_nm')))""")
    OUT['lpsms_stats'] = [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]
conn.close()
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\brief_check_pack.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
