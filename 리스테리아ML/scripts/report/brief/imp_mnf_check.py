# -*- coding: utf-8 -*-
"""림스 원본정보(lims_org_info) 제조일·유통시작/종료·선적일이 수입 시료에 맞게 붙는지 검증: (시료 의뢰번호 + 수입 접수번호) 둘 다 일치, 선적일과의 앞뒤"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"; C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
def dt(s):
    x = s.astype('string').str.slice(0, 10); ok = x.str.match(r'^(19|20)\d\d-\d\d-\d\d$').fillna(False)
    return pd.to_datetime(x.where(ok), errors='coerce', format='%Y-%m-%d')
def qs(d):
    d = pd.Series(d).dropna()
    return {'n': int(len(d)), 'p10': float(d.quantile(.1)), 'p50': float(d.median()), 'p90': float(d.quantile(.9)), 'after_ship_pct': round(float((d > 0).mean() * 100), 1)} if len(d) else {'n': 0}
body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'src_route', 'judge_result'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
o = body[body.src_route == 'overseas'].merge(pd.read_parquet(T + r"\imp_ready.parquet")[['lims_data_key', 'splore_reqest_no', 'cntc_sys_key2']].astype({'lims_data_key': str}), on='lims_data_key', how='left')
s = pd.read_parquet(C + r"\02e_s_org.parquet", columns=['rcept_no', 'label', 'lims_seq', 'shipng_de', 'distb_bgn_de', 'distb_end_de', 'dclr_prduct_dvs'])
s = s.assign(_lis=(s['label'] == '리스테리아')).sort_values(['rcept_no', '_lis', 'lims_seq'], ascending=[True, False, True]).drop_duplicates('rcept_no')
o = o.merge(s, left_on='cntc_sys_key2', right_on='rcept_no', how='left')
lo = pd.read_parquet(T + r"\imp_lims_org_info.parquet")
lo['k1'] = lo['splore_reqest_no'].astype(str); lo['k2'] = lo['cntc_sys_key2'].astype(str)
OUT = {'lims_org_rows': int(len(lo)), 'key2_present_pct': round(float(lo['cntc_sys_key2'].notna().mean() * 100), 1)}
# 같은 (의뢰번호, 접수번호) 안에서 제조일이 여러 개인지
g = lo.dropna(subset=['mnfctur_de']).groupby(['k1', 'k2'])['mnfctur_de'].nunique()
OUT['pairs_with_mnf'] = int(len(g)); OUT['pairs_multi_mnf_pct'] = round(float((g > 1).mean() * 100), 1) if len(g) else None
agg = lo.groupby(['k1', 'k2']).agg(mnf=('mnfctur_de', lambda v: v.dropna().min() if v.notna().any() else None), mnf_n=('mnfctur_de', 'nunique'),
                                   dbg=('distb_begin_dt', lambda v: v.dropna().min() if v.notna().any() else None), den=('distb_end_dt', lambda v: v.dropna().max() if v.notna().any() else None),
                                   shp=('ship_port_date', lambda v: v.dropna().min() if v.notna().any() else None)).reset_index()
x = o.assign(k1=o['splore_reqest_no'].astype(str), k2=o['cntc_sys_key2'].astype(str)).merge(agg, on=['k1', 'k2'], how='left')
x['det'] = x['judge_result'].astype(str) == '검출'
dvs = x['dclr_prduct_dvs'].fillna('(없음)').astype(str)
sh = dt(x['shipng_de']); mn = dt(x['mnf']); b0 = dt(x['distb_bgn_de']); lsh = dt(x['shp'])
OUT['link_both_keys_pct'] = round(float(x['mnf_n'].notna().mean() * 100), 1)
OUT['mnf_present_pct'] = round(float(mn.notna().mean() * 100), 1)
OUT['mnf_present_det'] = int((mn.notna() & x.det).sum())
OUT['ship_port_date_eq_shipng_pct'] = round(float((lsh == sh)[lsh.notna() & sh.notna()].mean() * 100), 1)
OUT['by_dvs'] = {}
for k in dvs.value_counts().index[:4]:
    mk = dvs == k
    OUT['by_dvs'][k] = {'rows': int(mk.sum()), 'mnf_present_pct': round(float(mn[mk].notna().mean() * 100), 1), 'mnf_minus_ship_days': qs((mn - sh).dt.days[mk]),
                        'mnf_eq_distb_bgn_pct': round(float((mn == b0)[mk & mn.notna() & b0.notna()].mean() * 100), 1) if (mk & mn.notna() & b0.notna()).any() else None,
                        'distb_bgn_minus_ship_days': qs((b0 - sh).dt.days[mk])}
# 둘 중 하나라도 선적 전 날짜(생산일 후보)가 있는 행
cand = ((mn.notna() & (mn <= sh)) | (b0.notna() & (b0 < sh) & dvs.isin(['수산물', '가공식품'])))
OUT['production_date_candidate_pct'] = round(float(cand.mean() * 100), 1)
OUT['production_date_candidate_by_dvs'] = {k: round(float(cand[dvs == k].mean() * 100), 1) for k in dvs.value_counts().index[:4]}
OUT['production_date_candidate_det'] = int((cand & x.det).sum())
OUT['production_to_ship_days'] = qs((sh - mn.where(mn.notna() & (mn <= sh), b0)).dt.days[cand])
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
json.dump(OUT, open(T + r"\imp_mnf_check.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
