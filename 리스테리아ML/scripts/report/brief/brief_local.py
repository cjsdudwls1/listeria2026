# -*- coding: utf-8 -*-
"""브리핑 수치: 우리 시료(listeria_body 676,537행) 기준 — 원자료 값을 바꾸지 않고 세기·기간 계산만"""
import sys, json, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"; C = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected"; R = C + r"\raw"
OUT = {}
def pct(m, base=None):
    m = pd.Series(m).fillna(False).astype(bool)
    return round(float(m.mean() * 100), 1) if len(m) else None
def has(s): return s.notna() & (s.astype('string').str.strip().fillna('') != '')
def dt(s):
    x = s.astype('string').str.slice(0, 10)
    ok = x.str.match(r'^(19|20)\d\d-\d\d-\d\d$').fillna(False)
    return pd.to_datetime(x.where(ok), errors='coerce', format='%Y-%m-%d')
def q(days):
    d = pd.Series(days).dropna()
    return {'n': int(len(d)), 'p10': float(d.quantile(.1)), 'p25': float(d.quantile(.25)), 'p50': float(d.median()), 'p75': float(d.quantile(.75)), 'p90': float(d.quantile(.9)),
            'neg': int((d < 0).sum())} if len(d) else {'n': 0}

body = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet", columns=['lims_data_key', 'judge_result', 'src_route', 'prdlst_code', 'prduct_nm', 'mnfctur_de', 'date_basis'])
body['lims_data_key'] = body['lims_data_key'].astype(str)
db = pd.read_parquet(C + r"\00_body_db.parquet", columns=['lims_data_key', 'geo_route', 'geo_match_level', 'addr_source', 'mnfctur_de', 'reqest_de', 'inspct_begin_de', 'prdlst_mnfctur_sttemnt_no',
                                                        'cntc_sys_key2', 'mapped_prdlst_code', 'prdlst_nm', 'prdctn_nation_code', 'orgplce_nation_code', 'import_nation_code', 'lat', 'lon'])
b = body.merge(db.rename(columns={'mnfctur_de': 'mnf_orig'}), on='lims_data_key', how='left')
b['det'] = b['judge_result'].astype(str) == '검출'
DOM, OVS = b.src_route == 'domestic', b.src_route == 'overseas'
def by(mask_series):
    m = pd.Series(mask_series, index=b.index).fillna(False).astype(bool)
    return {'domestic': pct(m[DOM]), 'overseas': pct(m[OVS]), 'det_domestic': pct(m[DOM & b.det]), 'det_overseas': pct(m[OVS & b.det])}
OUT['n'] = {'all': int(len(b)), 'domestic': int(DOM.sum()), 'overseas': int(OVS.sum()), 'det_domestic': int((DOM & b.det).sum()), 'det_overseas': int((OVS & b.det).sum())}

# ===== 1. 해외 기상: 수입 시료 기준일 구성·좌표 정밀도·나라 =====
o = b[OVS].copy()
OUT['ovs_date_basis'] = o['date_basis'].astype(str).value_counts().to_dict()
OUT['ovs_date_basis_det'] = o.loc[o.det, 'date_basis'].astype(str).value_counts().to_dict()
OUT['ovs_geo_match_level'] = o['geo_match_level'].astype(str).value_counts().to_dict()
OUT['ovs_addr_source'] = o['addr_source'].astype(str).value_counts().to_dict()
o['nation'] = o['prdctn_nation_code'].fillna(o['orgplce_nation_code']).fillna(o['import_nation_code'])
OUT['ovs_nation_top'] = o['nation'].astype(str).value_counts().head(12).to_dict()
OUT['ovs_coords'] = int(o[['lat', 'lon']].dropna().drop_duplicates().shape[0])
cc = o.dropna(subset=['lat', 'lon']).groupby('nation')[['lat', 'lon']].apply(lambda g: g.drop_duplicates().shape[0])
OUT['ovs_coords_per_nation_top'] = cc.sort_values(ascending=False).head(8).to_dict()
# 같은 좌표를 쓰는 시료 수(좌표가 나라 중심점 하나로 몰렸는지)
cnt = o.groupby(['lat', 'lon']).size().sort_values(ascending=False)
OUT['ovs_rows_top_coords'] = [{'lat': float(i[0]), 'lon': float(i[1]), 'rows': int(v), 'nation': str(o[(o.lat == i[0]) & (o.lon == i[1])]['nation'].iloc[0])} for i, v in cnt.head(5).items()]
gf = pd.read_parquet(R + r"\raw_global_factory_weather_listeria_sameday.parquet")
VARS = ['temperature_mean', 'temperature_max', 'temperature_min', 'relative_humidity_mean', 'relative_humidity_max', 'relative_humidity_min', 'precipitation_sum',
        'wind_speed_mean', 'wind_speed_max', 'wind_speed_min', 'radiation_sum', 'sunshine_duration', 'soil_moisture_mean']
OUT['gfw_sameday'] = {'rows': int(len(gf)), 'matched': int(gf['weather_seq'].notna().sum()), 'nonnull_pct': {c: pct(gf[c].notna()) for c in VARS},
                      'min': {c: float(gf[c].min()) for c in VARS}, 'max': {c: float(gf[c].max()) for c in VARS}}
gwi = pd.read_parquet(R + r"\raw_global_weather_info.parquet")
gd = pd.to_datetime(gwi['date'])
st = gwi[(gd >= '2015-04-07') & (gd <= '2026-06-30')]
OUT['gwi'] = {'rows': int(len(gwi)), 'date_min': str(gd.min().date()), 'date_max': str(gd.max().date()), 'countries': int(gwi['alpha_3'].nunique()),
              'nonnull_study_pct': {c: pct(st[c].notna()) for c in ['temperature_mean', 'temperature_max', 'temperature_min', 'relative_humidity_mean', 'precipitation_sum', 'wind_speed_mean', 'wind_speed_max', 'shortwave_radiation_sum', 'sunshine_duration', 'soil_moisture_mean']},
              'points_per_country_max': int(gwi.groupby('alpha_3')[['lat', 'lon']].apply(lambda g: g.drop_duplicates().shape[0]).max())}

# ===== 2. 제조~섭취 =====
codes = pd.read_parquet(C + r"\70_codes.parquet"); CN = dict(zip(codes['cmmn_code'].astype(str), codes['cmmn_code_nm'].astype(str)))
m = pd.read_parquet(C + r"\02a_m_org.parquet"); m['lims_data_key'] = m['lims_data_key'].astype(str)
x = b.merge(m.drop_duplicates('lims_data_key'), on='lims_data_key', how='left')
assert len(x) == len(b)
# 2-1 보관 상태·운반 온도
OUT['splore_sttus'] = {'present': by(has(x['splore_sttus_code'])),
                       'values_domestic': {CN.get(k, k): int(v) for k, v in x.loc[DOM, 'splore_sttus_code'].astype(str).value_counts().items() if k not in ('None', 'nan', '<NA>')},
                       'values_overseas': {CN.get(k, k): int(v) for k, v in x.loc[OVS, 'splore_sttus_code'].astype(str).value_counts().items() if k not in ('None', 'nan', '<NA>')}}
OUT['cnvync_sttus_tp'] = {'present': by(has(x['cnvync_sttus_tp'])), 'values': {str(k): int(v) for k, v in x['cnvync_sttus_tp'].value_counts().head(8).items()},
                          'unit_present': by(has(x['cnvync_sttus_tp_unit_code']))}
OUT['cnvync_mth'] = {'present': by(has(x['cnvync_mth_code'])), 'values': {CN.get(str(k), str(k)): int(v) for k, v in x['cnvync_mth_code'].value_counts().items()}}
# 2-2 기간: 국내 원본 제조일 → 의뢰일, 유통 일(distb_de)
mo, rq, dd = dt(x['mnf_orig']), dt(x['reqest_de']), dt(x['distb_de'])
OUT['dom_mnf_present'] = by(mo.notna())
OUT['dom_mnf_to_request_days'] = q((rq - mo).dt.days[DOM])
OUT['distb_de_present'] = by(dd.notna())
OUT['distb_de_minus_mnf_days_domestic'] = q((dd - mo).dt.days[DOM])
OUT['distb_de_minus_request_days_domestic'] = q((dd - rq).dt.days[DOM])
OUT['valid_tmlmt'] = {'present': by(has(x['valid_tmlmt'])), 'values': {str(k): int(v) for k, v in x['valid_tmlmt'].value_counts().head(6).items()}}
# 2-3 품목제조대장(삭제 안 된 최신 행, 원문 번호 일치)
ledg = pd.read_parquet(C + r"\02b_ledg.parquet")
lr = ledg.assign(_live=(ledg['del_yn'].fillna('N') == 'N')).sort_values(['prdlst_report_no', '_live', 'last_updt_dtm'], ascending=[True, False, False]).drop_duplicates('prdlst_report_no')
y = b[['lims_data_key', 'prdlst_mnfctur_sttemnt_no']].merge(lr, left_on='prdlst_mnfctur_sttemnt_no', right_on='prdlst_report_no', how='left')
assert len(y) == len(b)
y.index = b.index
OUT['ledger_key'] = by(y['prdlst_report_no'].notna())
for c in ['pog_daycnt', 'qlity_mntnc_tmlmt_daycnt', 'frmlc_mtrqlt', 'mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'haccp_appn_no']:
    OUT[f'ledger.{c}'] = {'present': by(has(y[c])), 'top': {str(k)[:40]: int(v) for k, v in y.loc[DOM, c].value_counts().head(8).items()}}
pog = y['pog_daycnt'].astype('string')
OUT['ledger.pog_patterns_domestic'] = {'제조일로부터 N개월': int(pog[DOM].str.contains(r'제조일로부터\s*\d+\s*개월', regex=True).fillna(False).sum()),
                                       '제조일로부터 N일': int(pog[DOM].str.contains(r'제조일로부터\s*\d+\s*일', regex=True).fillna(False).sum()),
                                       '제조일로부터 N년': int(pog[DOM].str.contains(r'제조일로부터\s*\d+\s*년', regex=True).fillna(False).sum()),
                                       '숫자만·개월만 등 기타': int((has(y['pog_daycnt'])[DOM] & ~pog[DOM].str.contains('제조일로부터', regex=False).fillna(False)).sum())}
mc = y['mnf_cnd_cn'].astype('string'); ps = y['prsrv_distb_stdr_cn'].astype('string'); fm = y['frmlc_mtrqlt'].astype('string')
OUT['ledger_keywords_domestic_rows'] = {
    '제조조건: 가열·살균·멸균': int(mc[DOM].str.contains('가열|살균|멸균', regex=True).fillna(False).sum()),
    '제조조건: 냉동·냉장': int(mc[DOM].str.contains('냉동|냉장', regex=True).fillna(False).sum()),
    '보존유통기준: 냉장': int(ps[DOM].str.contains('냉장', regex=False).fillna(False).sum()),
    '보존유통기준: 냉동': int(ps[DOM].str.contains('냉동', regex=False).fillna(False).sum()),
    '보존유통기준: 실온·상온': int(ps[DOM].str.contains('실온|상온', regex=True).fillna(False).sum()),
    '포장재질: 진공': int(fm[DOM].str.contains('진공', regex=False).fillna(False).sum()),
    '포장재질: 가스·질소': int(fm[DOM].str.contains('가스|질소|MAP', regex=True).fillna(False).sum())}
# 2-4 수입 LIMS: 리스테리아 행 우선 1행/접수번호
s02 = pd.read_parquet(C + r"\02e_s_org.parquet")
sx = pd.read_parquet(T + r"\brief_s_org_extra.parquet")
s = s02.merge(sx.drop(columns=['label']), on=['rcept_no', 'lims_seq'], how='left')
s = s.assign(_lis=(s['label'] == '리스테리아')).sort_values(['rcept_no', '_lis', 'lims_seq'], ascending=[True, False, True]).drop_duplicates('rcept_no')
z = b[['lims_data_key', 'cntc_sys_key2', 'reqest_de']].merge(s, left_on='cntc_sys_key2', right_on='rcept_no', how='left'); z.index = b.index
OUT['s_org_key'] = by(z['rcept_no'].notna())
OUT['s_org_prpos_nm'] = {'present': by(has(z['prpos_nm'])), 'values_overseas': z.loc[OVS, 'prpos_nm'].astype(str).value_counts().to_dict(),
                         'values_det_overseas': z.loc[OVS & b.det, 'prpos_nm'].astype(str).value_counts().to_dict()}
sh, en, wc, rq2 = dt(z['shipng_de']), dt(z['entrpt_de']), dt(z['wrhous_carry_de']), dt(z['reqest_de'])
db0, de0 = dt(z['distb_bgn_de']), dt(z['distb_end_de'])
OUT['import_dates_present'] = {c: by(has(z[c])) for c in ['shipng_de', 'entrpt_de', 'wrhous_carry_de', 'distb_bgn_de', 'distb_end_de', 'packng_plc_cd', 'packng_plc_addr']}
OUT['import_placeholder_9999'] = {c: int(z.loc[OVS, c].astype('string').str.startswith('9999').fillna(False).sum()) for c in ['distb_bgn_de', 'distb_end_de']}
OUT['import_days'] = {'선적→입항': q((en - sh).dt.days[OVS]), '입항→보관창고반입': q((wc - en).dt.days[OVS]), '보관창고반입→검사의뢰': q((rq2 - wc).dt.days[OVS]),
                      '선적→검사의뢰': q((rq2 - sh).dt.days[OVS]), '유통시작일−선적일': q((db0 - sh).dt.days[OVS]), '유통종료일−선적일': q((de0 - sh).dt.days[OVS]),
                      '유통종료일−유통시작일': q((de0 - db0).dt.days[OVS])}
nm = z['prdlst_nm'].astype('string'); pk = z['prduct_nm_kor'].astype('string')
OUT['import_name_keywords_overseas_rows'] = {
    '품명: 냉동': int(nm[OVS].str.contains('냉동', regex=False).fillna(False).sum()), '품명: 냉장': int(nm[OVS].str.contains('냉장', regex=False).fillna(False).sum()),
    '품명: 횟감·생식': int(nm[OVS].str.contains('횟감|생식', regex=True).fillna(False).sum()), '품명: 가열': int(nm[OVS].str.contains('가열', regex=False).fillna(False).sum()),
    '품명 예시(냉동·냉장 포함)': nm[OVS & nm.str.contains('냉동|냉장', regex=True).fillna(False)].value_counts().head(8).to_dict()}
co = pd.read_parquet(T + r"\brief_co_haccp.parquet")
zc = z[['ovsmnfst_cd']].merge(co.drop_duplicates('ovsmnfst_cd'), on='ovsmnfst_cd', how='left'); zc.index = b.index
OUT['ovs_manufacturer'] = {'key': by(zc['ntncd'].notna()), 'haccp_crtfc_yn_overseas': zc.loc[OVS, 'haccp_crtfc_yn'].astype(str).value_counts().to_dict(),
                           'fsms_yn_overseas': zc.loc[OVS, 'fsms_yn'].astype(str).value_counts().to_dict(), 'fsms_crtfc_yn_overseas': zc.loc[OVS, 'fsms_crtfc_yn'].astype(str).value_counts().to_dict()}
# 2-5 식품코드 이름(식품공전 분류)으로 본 공정·보관·섭취 표시
cl = pd.read_parquet(T + r"\brief_t_prdlst_cl.parquet").drop_duplicates('prdlst_cd')
code_use = b['mapped_prdlst_code'].fillna(b['prdlst_code']).astype('string')
k = pd.DataFrame({'code': code_use}).merge(cl[['prdlst_cd', 'kor_nm']], left_on='code', right_on='prdlst_cd', how='left'); k.index = b.index
kn = k['kor_nm'].astype('string')
GROUPS = {'가열·살균·멸균': '가열|살균|멸균', '비가열·생(生)': '비가열|생햄|생식|^생', '냉동': '냉동', '냉장': '냉장', '즉석섭취·신선편의': '즉석섭취|신선편의', '즉석조리': '즉석조리', '훈제·발효·건조': '훈제|발효|건조'}
OUT['type_name_keywords'] = {}
for g, p in GROUPS.items():
    mm = kn.str.contains(p, regex=True).fillna(False)
    OUT['type_name_keywords'][g] = {'rows': by(mm), 'n_domestic': int((mm & DOM).sum()), 'n_overseas': int((mm & OVS).sum()),
                                    'top_names': {str(a): int(c) for a, c in kn[mm].value_counts().head(6).items()}}
defs = cl[cl['kor_nm'].astype(str).str.contains('즉석섭취식품|신선편의식품|즉석조리식품|냉동식품|생햄|알가열제품', regex=True)][['prdlst_cd', 'kor_nm', 'dfn']]
OUT['type_definitions'] = [{'code': r.prdlst_cd, 'name': r.kor_nm, 'dfn': ' '.join(str(r.dfn).split())[:220]} for r in defs.itertuples()]
# 2-6 제품명 글자(참고)
pn = b['prduct_nm'].astype('string')
KW = {'냉동': '냉동', '냉장': '냉장', '진공': '진공', '훈제': '훈제', '횟감·회': '횟감|사시미|생선회', '살균': '살균|멸균', '가열': '가열'}
OUT['product_name_keywords'] = {g: by(pn.str.contains(p, regex=True).fillna(False)) for g, p in KW.items()}
json.dump(OUT, open(T + r"\brief_local.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
