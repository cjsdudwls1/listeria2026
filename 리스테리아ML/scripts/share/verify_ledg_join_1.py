# -*- coding: utf-8 -*-
"""[검증 1] 공유용 LIMS조인_데이터셋 ledg_* 9개 컬럼(품목제조대장) 조인 검증 — 2026-09-17
입력(읽기만): 공유용/데이터/LIMS조인_데이터셋.parquet, 공유용/데이터/DB수집/{hazard_preprocessed_data,t_prdlst_mnf_ledg}.parquet, data/collected/02b_ledg.parquet
출력: 화면만 — [1] 독립 재구현과 값 일치 [2] 번호 형식·미연결 [3] 제품명 대조 [4] 같은 번호 여러 행 [5] 보고신고일·수정일시 vs 기준일 [6] 판정별 연결률
결과 정리: docs/00_보고서/작업기록.md 의 2026-09-17 '품목제조대장 조인 검증'
실행: py 리스테리아ML/scripts/share/verify_ledg_join_1.py"""
import sys, re
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML"
S = D + r"\공유용\데이터"
LGC = ['mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'pog_daycnt', 'qlity_mntnc_tmlmt_daycnt', 'frmlc_mtrqlt', 'prdlst_report_ledg_no', 'del_yn', 'last_updt_dtm']
share = pd.read_parquet(S + r"\LIMS조인_데이터셋.parquet", columns=['lims_data_key', 'src_route', 'judge_result', 'mnfctur_de', 'prduct_nm', 'mnfctur_cmpny_nm', 'prdlst_code', 'db_prdlst_mnfctur_sttemnt_no', 'db_mapped_prdlst_code'] + ['ledg_' + c for c in LGC] + ['ledg_match_type'])
body = pd.read_parquet(D + r"\data\listeria_body.parquet", columns=['lims_data_key'])
hp = pd.read_parquet(S + r"\DB수집\hazard_preprocessed_data.parquet", columns=['lims_data_key', 'prdlst_mnfctur_sttemnt_no'])
lg = pd.read_parquet(S + r"\DB수집\t_prdlst_mnf_ledg.parquet")
lg02 = pd.read_parquet(D + r"\data\collected\02b_ledg.parquet")
print("share", share.shape, "ledger extract", lg.shape, "02b", lg02.shape)
print("추출본 = 02b 같은가:", lg.reset_index(drop=True).equals(lg02[lg.columns].reset_index(drop=True)))
print("ledger dtypes:", lg.dtypes.to_dict())

# ---------- 1. 독립 재구현 ----------
X = share[['lims_data_key']].merge(hp, on='lims_data_key', how='left')
assert (X['lims_data_key'].values == share['lims_data_key'].values).all()
raw = X['prdlst_mnfctur_sttemnt_no'].astype('string')
dig = raw.str.replace(r'[^0-9]', '', regex=True)
L = lg.copy()
L['live'] = (L['del_yn'].astype('string').fillna('N') != 'Y').astype(int)
L['upd'] = pd.to_datetime(L['last_updt_dtm'])
L['ledg_num'] = pd.to_numeric(L['prdlst_report_ledg_no'], errors='coerce')
print("대장번호 숫자 변환 실패:", L['ledg_num'].isna().sum(), "/ 길이 분포:", L['prdlst_report_ledg_no'].astype(str).str.len().value_counts().to_dict())
# 행마다 순위 매겨 1등
L = L.sort_values(['prdlst_report_no', 'live', 'upd', 'ledg_num'], ascending=[True, False, False, False], na_position='last', kind='mergesort')
best = L.groupby('prdlst_report_no', sort=False).head(1).set_index('prdlst_report_no')
keys = set(best.index)
k = raw.where(raw.isin(keys), dig.where(dig.isin(keys) & (dig != ''), pd.NA))
mt = np.where(raw.isin(keys).fillna(False), '원문 일치', np.where(k.notna(), '숫자만 일치', None))
diff = {}
for c in LGC:
    mine = k.map(best[c]) if True else None
    theirs = share['ledg_' + c]
    a = mine.astype('string'); b = theirs.astype('string')
    if c == 'last_updt_dtm':
        a = pd.to_datetime(mine).astype('string'); b = pd.to_datetime(theirs).astype('string')
    neq = ~((a == b).fillna(False) | (a.isna() & b.isna()))
    diff[c] = int(neq.sum())
mt_s = pd.Series(mt, dtype='string'); neq = ~((mt_s == share['ledg_match_type'].astype('string')).fillna(False) | (mt_s.isna() & share['ledg_match_type'].isna()))
diff['match_type'] = int(neq.sum())
print("\n[1] 재구현 vs 공유 parquet 불일치 행 수:", diff)

# ---------- 2. 번호 형식 ----------
dom = share['src_route'].astype(str) == 'domestic'
has = raw.notna() & (raw.str.strip() != '')
print("\n[2] 품목제조번호 있음: 국내", int((has & dom).sum()), "/", int(dom.sum()), " 수입", int((has & ~dom).sum()))
print("   시료 번호에 숫자 아닌 글자 있는 행:", int((has & raw.str.contains(r'[^0-9]', regex=True)).sum()), "고유", raw[has & raw.str.contains(r'[^0-9]', regex=True)].nunique())
nd = raw[has].str.findall(r'[^0-9]').explode().dropna()
print("   숫자 아닌 글자 종류:", nd.value_counts().head(10).to_dict())
print("   시료 번호 길이(고유 번호 기준):", raw[has].drop_duplicates().str.len().value_counts().sort_index().to_dict())
print("   대장 번호 숫자 아닌 글자 있는 행:", int(lg['prdlst_report_no'].astype(str).str.contains(r'[^0-9]').sum()), " 길이:", lg['prdlst_report_no'].astype(str).str.len().value_counts().sort_index().to_dict())
un = has & k.isna()
print("   연결 안 된 행:", int(un.sum()), "고유 번호", raw[un].nunique(), "(국내", int((un & dom).sum()), ")")
print("   연결 안 된 번호 길이:", raw[un].drop_duplicates().str.len().value_counts().sort_index().to_dict())
print("   연결 안 된 번호 예:", raw[un].value_counts().head(12).to_dict())
dg = mt_s == '숫자만 일치'
print("   숫자만 일치 행:", int(dg.sum()), "고유 원문", raw[dg].nunique())
print("   숫자만 일치 원문 예:", raw[dg].value_counts().head(12).to_dict())

# ---------- 3. 제품명 대조 ----------
def norm(s):
    return s.astype('string').str.lower().str.replace(r'[\s\(\)\[\]\{\}<>·,.\-_/\\\'"~!@#$%^&*+=:;?|]', '', regex=True)
ln = k.map(best['prdlst_nm'])
lc = k.map(best['prdlst_cd'])
pn = norm(share['prduct_nm']); lnn = norm(ln)
linked = k.notna()
exact = (pn == lnn).fillna(False)
contain = pd.Series([(isinstance(a, str) and isinstance(b, str) and len(a) > 0 and len(b) > 0 and (a in b or b in a)) for a, b in zip(pn.fillna(pd.NA).astype(object), lnn.astype(object))], index=share.index)
def bigram(s):
    return set(s[i:i+2] for i in range(len(s)-1)) if isinstance(s, str) and len(s) > 1 else set()
sim = pd.Series([ (len(bigram(a) & bigram(b)) / max(1, len(bigram(a) | bigram(b)))) if (isinstance(a, str) and isinstance(b, str)) else np.nan for a, b in zip(pn.astype(object), lnn.astype(object))], index=share.index)
for nm, m in [('원문 일치', mt_s == '원문 일치'), ('숫자만 일치', mt_s == '숫자만 일치')]:
    m = m.fillna(False)
    n = int(m.sum())
    print(f"\n[3] {nm} {n:,}행: 제품명 정규화 후 같음 {exact[m].mean():.1%} · 한쪽이 다른 쪽을 포함 {(exact|contain)[m].mean():.1%} · 2글자 조각 유사도 ≥0.3 {((exact|contain)|(sim>=0.3))[m].mean():.1%} · 유사도<0.1 이고 포함 아님 {((~(exact|contain)) & (sim<0.1))[m].mean():.1%}")
bad = linked & ~(exact | contain) & (sim < 0.1)
print("   이름이 전혀 다른 행 예(시료 제품명 | 대장 품목명 | 번호):")
ex = pd.DataFrame({'시료': share['prduct_nm'], '대장': ln, '번호': raw, 'type': mt_s})[bad]
print(ex.drop_duplicates('번호').head(25).to_string())
print("   이름이 전혀 다른 행 고유 번호:", ex['번호'].nunique(), "행", len(ex))
# 식품유형 코드 대조
print("   대장 prdlst_cd 예:", lg['prdlst_cd'].astype(str).value_counts().head(5).to_dict(), "/ 시료 prdlst_code 예:", share.loc[linked, 'prdlst_code'].astype(str).value_counts().head(5).to_dict())
code_eq = (lc.astype('string') == share['prdlst_code'].astype('string')).fillna(False)
code_eq14 = (lc.astype('string').str[:14] == share['prdlst_code'].astype('string').str[:14]).fillna(False)
code_eq_m = (lc.astype('string') == share['db_mapped_prdlst_code'].astype('string')).fillna(False)
print(f"   식품유형 코드 같음: 시료 prdlst_code {code_eq[linked].mean():.1%} · 앞 14자리 {code_eq14[linked].mean():.1%} · DB mapped 코드 {code_eq_m[linked].mean():.1%}")

# ---------- 4. 같은 번호 여러 행 ----------
g = lg.groupby('prdlst_report_no')
cnt = g.size()
dupn = set(cnt[cnt > 1].index)
print(f"\n[4] 번호 {len(cnt):,}개 중 여러 행인 번호 {len(dupn):,}개 (행 수 분포 {cnt.value_counts().sort_index().to_dict()})")
k_dup = k.isin(dupn)
print(f"   그 번호가 붙은 시료 행 {int(k_dup.sum()):,} (연결 행의 {k_dup[linked].mean():.2%})")
sub = lg[lg['prdlst_report_no'].isin(dupn)]
print("   del_yn 분포(여러 행 번호):", sub['del_yn'].astype('string').fillna('<NA>').value_counts().to_dict(), " 전체:", lg['del_yn'].astype('string').fillna('<NA>').value_counts().to_dict())
def ndiff(df, col):
    return int((df.groupby('prdlst_report_no')[col].nunique(dropna=False) > 1).sum())
for col in ['prdlst_nm', 'prdlst_cd', 'prms_dt', 'end_dt', 'pog_daycnt', 'frmlc_mtrqlt', 'mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'qlity_mntnc_tmlmt_daycnt', 'del_yn']:
    print(f"   여러 행 번호 중 '{col}' 값이 행마다 다른 번호: {ndiff(sub, col)}")
sub_live = sub[sub['del_yn'].astype('string').fillna('N') != 'Y']
lc2 = sub_live.groupby('prdlst_report_no').size(); ml = set(lc2[lc2 > 1].index)
print(f"   삭제 안 된 행이 2개 이상인 번호: {len(ml)} (시료 행 {int(k.isin(ml).sum())})")
for col in ['prdlst_nm', 'prdlst_cd', 'prms_dt', 'pog_daycnt', 'frmlc_mtrqlt']:
    print(f"     그중 '{col}' 다른 번호: {ndiff(sub_live[sub_live['prdlst_report_no'].isin(ml)], col)}")
print("   여러 행 번호 예시 3개:")
for no in list(sorted(dupn, key=lambda x: -int(k.eq(x).sum())))[:4]:
    print(lg[lg['prdlst_report_no'] == no][['prdlst_report_no', 'prdlst_report_ledg_no', 'prdlst_nm', 'prms_dt', 'end_dt', 'del_yn', 'last_updt_dtm', 'pog_daycnt', 'frmlc_mtrqlt']].to_string(), "\n   붙은 시료 행", int(k.eq(no).sum()))

# ---------- 5. 날짜 대조 ----------
print("\n[5] prms_dt 형식 예:", lg['prms_dt'].astype(str).str.len().value_counts().to_dict(), lg['prms_dt'].head(3).tolist(), " end_dt 예:", lg['end_dt'].dropna().astype(str).head(3).tolist(), "end_dt 값 있음", int(lg['end_dt'].notna().sum()))
def pdt(s):
    s = s.astype('string').str.replace(r'\D', '', regex=True).str[:8]
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')
L['prms'] = pdt(L['prms_dt']); L['end'] = pdt(L['end_dt'])
best = L.groupby('prdlst_report_no', sort=False).head(1).set_index('prdlst_report_no')
base = pd.to_datetime(share['mnfctur_de']).dt.normalize()
prms = k.map(best['prms']); end = k.map(best['end']); upd = pd.to_datetime(k.map(best['upd']))
print(f"   붙은 행 prms_dt 날짜 변환 실패: {int((linked & prms.isna()).sum())}")
print(f"   보고 신고일자 > 시료 기준일(제품 보고 전에 만든 시료?): {int((linked & (prms > base)).sum()):,}행 ({(prms > base)[linked].mean():.2%})")
d = (prms - base).dt.days[linked & (prms > base)]
if len(d): print("     그 차이(일) 분위:", d.quantile([.1, .5, .9]).to_dict())
print(f"   생산종료일자 < 시료 기준일: {int((linked & (end < base)).sum()):,}행 / 생산종료일자 있음 {int((linked & end.notna()).sum()):,}행")
print(f"   최종 수정 일시 > 시료 기준일: {(upd > base)[linked].mean():.1%}")
print("   최종 수정 일시 상위(대장 행 기준):", L['upd'].dt.date.value_counts().head(5).to_dict())
# 시료 기준일에 유효했던 행(보고 신고일자 <= 기준일 중 가장 최근, 삭제 안 된 행 우선)과 고른 행 비교 — 여러 행 번호만
rows = pd.DataFrame({'i': np.arange(len(share)), 'no': k, 'base': base})[k_dup]
cand = rows.merge(L[['prdlst_report_no', 'prdlst_report_ledg_no', 'live', 'prms', 'upd', 'ledg_num'] + ['pog_daycnt', 'frmlc_mtrqlt', 'mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'qlity_mntnc_tmlmt_daycnt']], left_on='no', right_on='prdlst_report_no')
cand['valid'] = (cand['prms'] <= cand['base']).astype(int)
tv = cand.sort_values(['i', 'valid', 'prms', 'live', 'upd', 'ledg_num'], ascending=[True, False, False, False, False, False], na_position='last').groupby('i').head(1).set_index('i')
chosen_ledg = share['ledg_prdlst_report_ledg_no'].astype('string')
same_row = (tv['prdlst_report_ledg_no'].astype('string') == chosen_ledg.iloc[tv.index].values)
print(f"   여러 행 번호가 붙은 시료 {len(tv):,}행: '기준일 시점 유효 행' 규칙과 고른 행이 다른 시료 {int((~same_row).sum()):,}행")
cont_diff = pd.Series(False, index=tv.index)
for c in ['pog_daycnt', 'frmlc_mtrqlt', 'mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'qlity_mntnc_tmlmt_daycnt']:
    a = tv[c].astype('string'); b = share['ledg_' + c].astype('string').iloc[tv.index]
    b.index = tv.index
    dd = ~((a == b).fillna(False) | (a.isna() & b.isna()))
    print(f"     그중 '{c}' 값까지 달라지는 시료 행: {int(dd.sum()):,}")
    cont_diff |= dd
print(f"     5개 내용 컬럼 중 하나라도 달라지는 시료 행: {int(cont_diff.sum()):,}  (검출 {int((share['judge_result'].astype(str).iloc[tv.index][cont_diff.values] == '검출').sum())})")

# ---------- 6. 판정별 연결률 ----------
det = share['judge_result'].astype(str) == '검출'
for nm, m in [('국내 전체', dom), ('국내 검출', dom & det), ('국내 불검출', dom & ~det)]:
    print(f"\n[6] {nm}: 번호 있음 {has[m].mean():.1%} · 연결 {linked[m].mean():.1%} · 번호 있는 행 중 연결 {linked[m & has].mean():.2%}")
