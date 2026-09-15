# -*- coding: utf-8 -*-
"""수입식품 관련 칸 찾기(로컬 카탈로그 CSV): 재배·도축·어획 날짜·장소, 생산·제조일, 운송·항만·창고, HS 코드, 인증"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
cols = pd.read_csv(T + r"\brief_columns.csv"); tabs = pd.read_csv(T + r"\brief_tables.csv")
cols = cols.merge(tabs[['schema', 'tbl', 'approx_rows', 'tcomment']], on=['schema', 'tbl'], how='left')
cols = cols[~cols.schema.isin(['staging_bak', 'train_bak', 'train_data', 'predict_data', 'weather_data'])]
cols = cols[~cols.tbl.str.contains('aflatoxin|_judge|^api|wx', regex=True)]
PAT = {
    '재배·수확·도축·어획': r'도축|어획|수확|재배|포획|양식|해역|농장|산란|착유|slaughter|catch|harvest|farm|fishing|fish_area|abattoir',
    '생산·제조·가공일': r'생산일|제조일|제조 일|제조일자|가공일|포장일|냉동일|생산 일|mnfctur_de|mnf_de|prdctn_de|produc.*date|manufacture_date|mnfctur_year|mnfctur_mt',
    '원산지·생산국': r'원산|산지|생산국|생산 국가|수출국|수출 국가|제조국|제조 국가|origin|prdctn_nation|xport_nation|mnf_nation|orgplce',
    '운송·항만': r'운송|수송|항공|선박|해상|선적|입항|도착|항 명|선적항|port|vessel|flight|bl_no|blno|awb|컨테이너|container|reefer|arvl|prload|shipng|entrpt|etrypt',
    '창고·보세': r'창고|보세|반입|세관|wrhous|bonded|bnded|custom|warehouse|cstdy|tkin',
    'HS·품목': r'hsk|hs_|관세|tariff',
    '인증·안전관리': r'haccp|fsms|인증|crtfc|안전관리',
}
for k, p in PAT.items():
    m = cols[cols.apply(lambda r: bool(re.search(p, str(r.col), re.I) or re.search(p, str(r.ccomment) if isinstance(r.ccomment, str) else '', re.I)), axis=1)]
    print(f"\n===== {k} ({len(m)}칸) =====")
    for (sc, tb), g in m.groupby(['schema', 'tbl']):
        print(f"- {sc}.{tb} (약 {int(g.approx_rows.iloc[0]):,}행): " + ' | '.join(f"{x.col}{'=' + ' '.join(str(x.ccomment).split())[:30] if isinstance(x.ccomment, str) else ''}" for x in g.sort_values('attnum').itertuples()))
g = cols[(cols.schema == 'ba_data') & (cols.tbl == 'm_org_lims_data_info')].sort_values('attnum')
print('\n[ba_data.m_org_lims_data_info 전체 칸]', ' | '.join(f"{x.col}={' '.join(str(x.ccomment).split())[:24] if isinstance(x.ccomment, str) else ''}" for x in g.itertuples()))
