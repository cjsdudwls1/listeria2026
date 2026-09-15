# -*- coding: utf-8 -*-
"""보고서에 남길 컬럼이 우리 시료 파일(listeria_body.parquet) 컬럼과 같은 값인지 대조 (측정만)"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet")
body['lims_data_key'] = body['lims_data_key'].astype(str)
OUT = {}
def norm(s): return s.astype(str).str.strip()
for f in ['02a_m_org', '02e_s_org', '00_body_db']:
    df = pd.read_parquet(C + rf"\{f}.parquet")
    print(f, df.shape, list(df.columns)[:60])
m = pd.read_parquet(C + r"\02a_m_org.parquet"); m['lims_data_key'] = m['lims_data_key'].astype(str)
x = body.merge(m, on='lims_data_key', how='inner', suffixes=('', '_m'))
for bc in ['prduct_nm', 'train_prdlst_nm', 'prdlst_code', 'mnfctur_cmpny_nm']:
    for mc in ['prdlst_nm', 'upper_prdlst_nm', 'prduct_nm_m', 'prdlst_code_m', 'upper_prdlst_code']:
        if bc in x.columns and mc in x.columns:
            both = x[bc].notna() & x[mc].notna()
            eq = (norm(x.loc[both, bc]) == norm(x.loc[both, mc])).mean() * 100 if both.any() else None
            print(f'm_org {mc} == body {bc}: 둘 다 있음 {int(both.sum())}, 같은 값 {eq}')
            OUT[f'm_org.{mc}=={bc}'] = eq
s = pd.read_parquet(C + r"\02e_s_org.parquet")
print('s_org columns:', list(s.columns))
bd = pd.read_parquet(C + r"\00_body_db.parquet"); bd['lims_data_key'] = bd['lims_data_key'].astype(str)
print('body_db columns:', list(bd.columns))
key = 'cntc_sys_key2' if 'cntc_sys_key2' in bd.columns else None
if key:
    y = body.merge(bd[['lims_data_key', key]], on='lims_data_key').merge(s, left_on=key, right_on='rcept_no', how='inner', suffixes=('', '_s'))
    for bc in ['prduct_nm', 'train_prdlst_nm', 'prdlst_code', 'mnfctur_cmpny_nm']:
        for sc in ['prdlst_nm', 'prduct_nm_s', 'prdlst_cd', 'hsk_no', 'dclr_prduct_dvs']:
            if bc in y.columns and sc in y.columns:
                both = y[bc].notna() & y[sc].notna()
                eq = (norm(y.loc[both, bc]) == norm(y.loc[both, sc])).mean() * 100 if both.any() else None
                print(f's_org {sc} == body {bc}: 둘 다 있음 {int(both.sum())}, 같은 값 {eq}')
                OUT[f's_org.{sc}=={bc}'] = eq
    print(y[['prduct_nm', 'train_prdlst_nm'] + [c for c in ['prdlst_nm'] if c in y.columns]].head(8).to_string())
print(x[['prduct_nm', 'train_prdlst_nm', 'prdlst_code'] + [c for c in ['prdlst_nm', 'upper_prdlst_nm'] if c in x.columns]].head(8).to_string())
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\v3_dupcheck.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
