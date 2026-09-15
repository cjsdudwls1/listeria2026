# -*- coding: utf-8 -*-
"""수입 시료(47,642행) 기준 — 원자료 값을 바꾸지 않고 세기·기간 계산만"""
import sys, json, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"; C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"
OUT = {}
def pct(m): m = pd.Series(m).fillna(False).astype(bool); return round(float(m.mean() * 100), 1) if len(m) else None
def has(s): return s.notna() & (s.astype('string').str.strip().fillna('') != '')
def dt(s):
    x = s.astype('string').str.slice(0, 10)
    ok = x.str.match(r'^(19|20)\d\d-\d\d-\d\d$').fillna(False)
    return pd.to_datetime(x.where(ok), errors='coerce', format='%Y-%m-%d')
def qs(d):
    d = pd.Series(d).dropna()
    return {'n': int(len(d)), 'p10': float(d.quantile(.1)), 'p50': float(d.median()), 'p90': float(d.quantile(.9)), 'neg_pct': round(float((d < 0).mean() * 100), 1)} if len(d) else {'n': 0}

body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'judge_result', 'src_route', 'prduct_nm', 'date_basis'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'statistic_prdlst_nm_1', 'statistic_prdlst_nm_2', 'prdlst_nm', 'prdctn_nation_code', 'orgplce_nation_code', 'import_nation_code', 'cntc_sys_key2', 'reqest_de'])
o = body[body.src_route == 'overseas'].merge(db, on='lims_data_key', how='left')
o['det'] = o['judge_result'].astype(str) == '검출'
N = len(o); OUT['n'] = N; OUT['det'] = int(o.det.sum())
# 수입 LIMS(리스테리아 행 우선 1행/접수번호)
s = pd.read_parquet(C + r"\02e_s_org.parquet").merge(pd.read_parquet(T + r"\brief_s_org_extra.parquet").drop(columns=['label']), on=['rcept_no', 'lims_seq'], how='left') \
      .merge(pd.read_parquet(T + r"\imp_s_org_bl.parquet")[['rcept_no', 'lims_seq', 'bl_no']], on=['rcept_no', 'lims_seq'], how='left')
s = s.assign(_lis=(s['label'] == '리스테리아')).sort_values(['rcept_no', '_lis', 'lims_seq'], ascending=[True, False, True]).drop_duplicates('rcept_no')
o = o.merge(s.drop(columns=['_lis']), left_on='cntc_sys_key2', right_on='rcept_no', how='left')
assert len(o) == N
dvs = o['dclr_prduct_dvs'].fillna('(없음)').astype(str)

# A. 구성
OUT['A_stat1'] = {k: {'rows': int(v), 'det': int(o.loc[o['statistic_prdlst_nm_1'].astype(str) == k, 'det'].sum())} for k, v in o['statistic_prdlst_nm_1'].astype(str).value_counts().items()}
OUT['A_dvs'] = {k: {'rows': int(v), 'det': int(o.loc[dvs == k, 'det'].sum())} for k, v in dvs.value_counts().items()}
OUT['A_top_items_by_dvs'] = {k: {str(a)[:40]: int(c) for a, c in o.loc[dvs == k, 'prdlst_nm_y' if 'prdlst_nm_y' in o.columns else 'prdlst_nm'].value_counts().head(6).items()} for k in dvs.value_counts().index[:4]}
OUT['A_det_items'] = {str(a)[:40]: int(c) for a, c in o.loc[o.det, 'prdlst_nm_y' if 'prdlst_nm_y' in o.columns else 'prdlst_nm'].value_counts().head(10).items()}
item = o['prdlst_nm_y' if 'prdlst_nm_y' in o.columns else 'prdlst_nm'].astype('string')

# B. 날짜: 생산·제조일
mo = pd.read_parquet(T + r"\imp_m_org.parquet"); mo['lims_data_key'] = mo['lims_data_key'].astype(str)
x = o[['lims_data_key']].merge(mo, on='lims_data_key', how='left'); x.index = o.index
OUT['B_m_org_link_pct'] = pct(x['mnfctur_cmpny_locplc_nm'].notna() | x['prdlst_nm'].notna() | x['splore_nm'].notna())
OUT['B_m_org_mnfctur_de_pct'] = pct(dt(x['mnfctur_de']).notna())
OUT['B_m_org_distb_de_pct'] = pct(dt(x['distb_de']).notna())
for c in ['orgplce_nation_code', 'prdctn_nation_code', 'xport_nation_code']:
    OUT[f'B_m_org_{c}_pct'] = pct(has(x[c]))
rd = pd.read_parquet(T + r"\imp_ready.parquet"); rd['lims_data_key'] = rd['lims_data_key'].astype(str)
lo = pd.read_parquet(T + r"\imp_lims_org_info.parquet")
y = o[['lims_data_key', 'cntc_sys_key2']].merge(rd[['lims_data_key', 'splore_reqest_no']], on='lims_data_key', how='left')
lo['_k'] = lo['splore_reqest_no'].astype(str)
agg = lo.groupby('_k').agg(**{c: (c, lambda v: v.dropna().iloc[0] if v.notna().any() else None) for c in ['mnfctur_year', 'mnfctur_mt', 'mnfctur_de', 'distb_begin_dt', 'distb_end_dt', 'ship_port_date', 'arvl_prearnge_de', 'prload_nm', 'etrypt_de', 'frght_cstdy_place_nm', 'cntc_sys_key2']})
y = y.merge(agg, left_on=y['splore_reqest_no'].astype(str), right_index=True, how='left', suffixes=('', '_lo')); y.index = o.index
OUT['B_lims_org_info_link_pct'] = pct(y['splore_reqest_no'].astype(str).isin(set(lo['_k'])))
OUT['B_lims_org_info_present_pct'] = {c: pct(has(y[c])) for c in ['mnfctur_year', 'mnfctur_mt', 'mnfctur_de', 'distb_begin_dt', 'distb_end_dt', 'ship_port_date', 'arvl_prearnge_de', 'prload_nm', 'etrypt_de', 'frght_cstdy_place_nm']}
OUT['B_prload_top'] = {str(k)[:30]: int(v) for k, v in y['prload_nm'].value_counts().head(10).items()}
# 수입 유통시작일·종료일 뜻 확인(신고제품구분별)
sh, b0, e0 = dt(o['shipng_de']), dt(o['distb_bgn_de']), dt(o['distb_end_de'])
same = (b0 == e0) & b0.notna()
OUT['B_distb_same_pct_of_bgn_present'] = pct(same[b0.notna()])
res = {}
for k in dvs.value_counts().index[:4]:
    mk = dvs == k
    res[k] = {'rows': int(mk.sum()), 'bgn_present': pct(b0[mk].notna()), 'end_present': pct(e0[mk].notna()), 'same_date_pct': pct(same[mk & b0.notna()]),
              'same_date_minus_ship': qs((b0 - sh).dt.days[mk & same]), 'diff_bgn_minus_ship': qs((b0 - sh).dt.days[mk & ~same & b0.notna() & e0.notna()]),
              'diff_end_minus_ship': qs((e0 - sh).dt.days[mk & ~same & b0.notna() & e0.notna()]), 'diff_end_minus_bgn': qs((e0 - b0).dt.days[mk & ~same & b0.notna() & e0.notna()])}
OUT['B_distb_by_dvs'] = res
# 선적 전 날짜(생산일 후보)가 있는 행
OUT['B_any_date_before_ship_pct'] = pct(((b0 < sh) | (e0 < sh)).fillna(False))

# C. 운송·창고
bl = o['bl_no'].astype('string').str.replace(r'\s', '', regex=True).str.upper()
cls = np.select([bl.isna() | (bl == ''), bl.str.match(r'^\d{3}-?\d{8}$').fillna(False), bl.str.match(r'^\d+$').fillna(False), bl.str.match(r'^[A-Z]{3,4}').fillna(False)],
                ['(없음)', '숫자 11자리(항공화물번호 형태)', '그 밖의 숫자만', '영문으로 시작'], default='그 밖의 형태')
o['bl_cls'] = cls
en = dt(o['entrpt_de']); days = (en - sh).dt.days
OUT['C_bl_present_pct'] = pct(o['bl_cls'] != '(없음)')
OUT['C_bl_class'] = {k: {'rows': int(v), 'ship_to_arrival': qs(days[o.bl_cls == k]), 'det': int(o.loc[o.bl_cls == k, 'det'].sum())} for k, v in o['bl_cls'].value_counts().items()}
OUT['C_bl_examples'] = {k: o.loc[o.bl_cls == k, 'bl_no'].astype(str).head(3).tolist() for k in o['bl_cls'].unique()}
OUT['C_ship_to_arrival_by_dvs'] = {k: qs(days[dvs == k]) for k in dvs.value_counts().index[:4]}
cp = o['inspct_carry_plc'].astype('string')
OUT['C_carry_plc_present_pct'] = pct(has(o['inspct_carry_plc']))
OUT['C_carry_plc_cold_word_pct'] = pct(cp.str.contains('냉동|냉장|저온|콜드|COLD|FREEZ|REEFER', regex=True, case=False).fillna(False))
OUT['C_carry_plc_top'] = {str(k)[:40]: int(v) for k, v in cp.value_counts().head(8).items()}
OUT['C_custom_present_pct'] = pct(has(o['custom_cd'])); OUT['C_wrhous_present_pct'] = pct(has(o['wrhous_cd']))
cm = pd.read_parquet(T + r"\imp_codes.parquet")
ba = o['bonded_area_cd'].astype(str).value_counts().head(8)
OUT['C_bonded_area_top'] = {k: int(v) for k, v in ba.items()}
OUT['C_bonded_code_classes'] = cm[cm['cmmn_code'].isin(ba.index)].groupby('cmmn_cl_code').size().sort_values(ascending=False).head(5).to_dict()

# D. 상태(냉동·냉장 등)
m2 = pd.read_parquet(C + r"\02a_m_org.parquet", columns=['lims_data_key', 'splore_sttus_code']); m2['lims_data_key'] = m2['lims_data_key'].astype(str)
codes = pd.read_parquet(C + r"\70_codes.parquet"); CN = dict(zip(codes['cmmn_code'].astype(str), codes['cmmn_code_nm'].astype(str)))
st = o[['lims_data_key']].merge(m2.drop_duplicates('lims_data_key'), on='lims_data_key', how='left')['splore_sttus_code'].map(lambda v: CN.get(str(v), None) if pd.notna(v) else None); st.index = o.index
OUT['D_sample_state_by_dvs'] = {k: {'present': pct(st[dvs == k].notna()), **{str(a): int(c) for a, c in st[dvs == k].value_counts().items()}} for k in dvs.value_counts().index[:4]}
W = {'냉동': '냉동', '냉장': '냉장', '신선': '신선', '건조': '건조|건', '훈제': '훈제', '염장·염건': '염장|염건|염수', '자숙(삶음)': '자숙', '살균·멸균': '살균|멸균', '횟감·생식': '횟감|생식'}
OUT['D_item_words_by_dvs'] = {k: {w: int(item[dvs == k].str.contains(p, regex=True).fillna(False).sum()) for w, p in W.items()} for k in dvs.value_counts().index[:4]}
hs = o['hsk_no'].astype('string').str.replace(r'\D', '', regex=True)
OUT['D_hsk_present_pct'] = pct(hs.str.len().fillna(0) >= 4)
h4 = hs.str.slice(0, 4)
tab = []
for code, n in h4.value_counts().head(15).items():
    mk = h4 == code
    tab.append({'hs4': code, 'rows': int(n), 'det': int(o.loc[mk, 'det'].sum()), '품명_냉동%': pct(item[mk].str.contains('냉동', regex=False).fillna(False)),
                '품명_냉장%': pct(item[mk].str.contains('냉장', regex=False).fillna(False)), 'top_item': {str(a)[:30]: int(c) for a, c in item[mk].value_counts().head(2).items()}})
OUT['D_hs4_top'] = tab
# 품명에 냉동이 있는 행의 HS 앞 4자리, 냉장이 있는 행의 HS 앞 4자리
OUT['D_hs4_when_item_frozen'] = h4[item.str.contains('냉동', regex=False).fillna(False)].value_counts().head(6).to_dict()
OUT['D_hs4_when_item_chilled'] = h4[item.str.contains('냉장', regex=False).fillna(False)].value_counts().head(6).to_dict()
en_nm = o['prduct_nm_eng'].astype('string').str.upper()
EW = {'FROZEN': 'FROZEN', 'CHILLED': 'CHILLED', 'FRESH': 'FRESH', 'SMOKED': 'SMOKED', 'COOKED·BOILED': 'COOKED|BOILED', 'RAW': r'\bRAW\b', 'PASTEURI': 'PASTEURI', 'UNPASTEURI·RAW MILK': 'UNPASTEURI|RAW MILK', 'CURED·DRIED': 'CURED|DRIED', 'FERMENT': 'FERMENT'}
OUT['D_eng_name_present_pct'] = pct(has(o['prduct_nm_eng']))
OUT['D_eng_words'] = {w: {'rows': int(en_nm.str.contains(p, regex=True).fillna(False).sum()), 'det': int((en_nm.str.contains(p, regex=True).fillna(False) & o.det).sum())} for w, p in EW.items()}

# E. 가공식품 공정 단서 · 해외제조업소 대장
pk = dvs == '가공식품'
OUT['E_processed_rows'] = int(pk.sum()); OUT['E_processed_det'] = int(o.loc[pk, 'det'].sum())
OUT['E_processed_eng_words'] = {w: int((en_nm.str.contains(p, regex=True).fillna(False) & pk).sum()) for w, p in EW.items()}
reg = pd.read_parquet(T + r"\imp_registry.parquet")
r = o[['ovsmnfst_cd']].merge(reg.drop_duplicates('ovsmnfst_cd'), on='ovsmnfst_cd', how='left'); r.index = o.index
lk = r['ntncd'].notna()
OUT['E_registry_link_pct'] = pct(lk); OUT['E_registry_link_by_dvs'] = {k: pct(lk[dvs == k]) for k in dvs.value_counts().index[:4]}
OUT['E_fsms_yn'] = r.loc[lk, 'fsms_yn'].astype(str).value_counts().to_dict()
OUT['E_crtfc_nm_present_pct_of_linked'] = pct(has(r.loc[lk, 'crtfc_nm']))
OUT['E_crtfc_nm_top'] = {str(k)[:40]: int(v) for k, v in r.loc[lk, 'crtfc_nm'].value_counts().head(8).items()}
sf = r.loc[lk, 'sfrprp_food_se_cd'].astype(str).value_counts().head(8)
OUT['E_sfrprp_food_se_top'] = {k: int(v) for k, v in sf.items()}
OUT['E_sfrprp_code_names'] = cm[cm['cmmn_code'].isin(sf.index)][['cmmn_cl_code', 'cmmn_code', 'cmmn_code_nm']].head(20).astype(str).to_dict('records')

# F. 나라 코드(생산·원산지·수출·제조)
OUT['F_codes_present_pct'] = {'생산국(prdctn)': pct(has(o['prdctn_nation_code'])), '원산지(orgplce)': pct(has(o['orgplce_nation_code'])), '수출국(xport, m_org)': pct(has(x['xport_nation_code'])),
                              '제조국(수입 LIMS mnf_nation_cd)': pct(has(o['mnf_nation_cd']))}
both = has(o['prdctn_nation_code']) & has(o['mnf_nation_cd'])
OUT['F_prdctn_ne_mnf_pct'] = pct((o['prdctn_nation_code'].astype(str) != o['mnf_nation_cd'].astype(str))[both])
OUT['F_prdctn_ne_mnf_by_dvs'] = {k: pct((o['prdctn_nation_code'].astype(str) != o['mnf_nation_cd'].astype(str))[both & (dvs == k)]) for k in dvs.value_counts().index[:4]}
json.dump(OUT, open(T + r"\imp_local.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
