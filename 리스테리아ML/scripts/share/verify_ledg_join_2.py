# -*- coding: utf-8 -*-
"""[검증 2] 품목제조대장 조인 — 같은 번호 여러 행일 때 행 고르는 규칙 비교(제품명 일치율), 제품명 불일치 원인, 품목코드 불일치, 숫자만 일치, 미연결 번호 — 2026-09-17
입력(읽기만): 공유용/데이터/LIMS조인_데이터셋.parquet, 공유용/데이터/DB수집/t_prdlst_mnf_ledg.parquet
결과 정리: docs/00_보고서/작업기록.md 의 2026-09-17 '품목제조대장 조인 검증'
실행: py 리스테리아ML/scripts/share/verify_ledg_join_2.py"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, pyarrow.parquet as pq
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML"; S = D + r"\공유용\데이터"
share = pd.read_parquet(S + r"\LIMS조인_데이터셋.parquet", columns=['lims_data_key', 'judge_result', 'mnfctur_de', 'date_basis', 'prduct_nm', 'mnfctur_cmpny_nm', 'prdlst_code', 'db_prdlst_nm', 'db_prdlst_mnfctur_sttemnt_no', 'ledg_prdlst_report_ledg_no', 'ledg_match_type'])
lg = pd.read_parquet(S + r"\DB수집\t_prdlst_mnf_ledg.parquet")
print("last_updt_dtm NaT:", int(lg['last_updt_dtm'].isna().sum()), " del_yn NA:", int(lg['del_yn'].isna().sum()))
print("nb2_db_attr cols:", pq.read_schema(D + r"\data\collected\nb2_db_attr.parquet").names)
raw = share['db_prdlst_mnfctur_sttemnt_no'].astype('string'); dig = raw.str.replace(r'[^0-9]', '', regex=True)
keys = set(lg['prdlst_report_no'])
k = raw.where(raw.isin(keys), dig.where(dig.isin(keys) & (dig != ''), pd.NA))
linked = k.notna()
def norm(s): return s.astype('string').str.lower().str.replace(r'[^0-9a-z가-힣]', '', regex=True)
def bg(s): return set(s[i:i+2] for i in range(len(s)-1)) if isinstance(s, str) and len(s) > 1 else set()
def sim(a, b):
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b: return np.nan
    if a == b or a in b or b in a: return 1.0
    A, B = bg(a), bg(b); return len(A & B) / max(1, len(A | B))
lg['nm_n'] = norm(lg['prdlst_nm']); share['pn_n'] = norm(share['prduct_nm'])
lg['prms'] = pd.to_datetime(lg['prms_dt'], format='%Y%m%d', errors='coerce'); lg['end'] = pd.to_datetime(lg['end_dt'], format='%Y%m%d', errors='coerce')
lg['live'] = (lg['del_yn'].fillna('N') != 'Y').astype(int); lg['ln'] = lg['prdlst_report_ledg_no'].astype('int64')
base = pd.to_datetime(share['mnfctur_de']).dt.normalize()

# ---- A. 여러 행 번호: 규칙별 제품명 일치 ----
cnt = lg.groupby('prdlst_report_no').size(); dupn = set(cnt[cnt > 1].index)
R = pd.DataFrame({'i': np.arange(len(share)), 'no': k, 'base': base, 'pn': share['pn_n'], 'chosen': share['ledg_prdlst_report_ledg_no'], 'judge': share['judge_result'].astype(str)})[k.isin(dupn)]
C = R.merge(lg[['prdlst_report_no', 'prdlst_report_ledg_no', 'nm_n', 'prdlst_nm', 'prms', 'end', 'live', 'last_updt_dtm', 'ln', 'pog_daycnt', 'frmlc_mtrqlt']], left_on='no', right_on='prdlst_report_no')
C['s'] = [sim(a, b) for a, b in zip(C['pn'], C['nm_n'])]
C['valid'] = ((C['prms'] <= C['base']) & (C['end'].isna() | (C['end'] >= C['base']))).astype(int)
C['started'] = (C['prms'] <= C['base']).astype(int)
cur = C[C['prdlst_report_ledg_no'] == C['chosen']].set_index('i'); print('완전 동점 중복 i:', int(cur.index.duplicated().sum())); cur = cur[~cur.index.duplicated()]
fd = lg[lg.duplicated('prdlst_report_ledg_no', keep=False)]; print('대장번호 중복 행:', len(fd), '/ 모든 열까지 같은 행:', int(lg.drop(columns=[c for c in lg.columns if c in ('nm_n',)]).duplicated(keep=False).sum())); 
tv = C.sort_values(['i', 'started', 'prms', 'live', 'last_updt_dtm', 'ln'], ascending=[True, False, False, False, False, False]).groupby('i').head(1).set_index('i')
tv2 = C.sort_values(['i', 'valid', 'started', 'prms', 'live', 'last_updt_dtm', 'ln'], ascending=[True, False, False, False, False, False, False]).groupby('i').head(1).set_index('i')
orc = C.sort_values(['i', 's'], ascending=[True, False]).groupby('i').head(1).set_index('i')
n = len(cur)
print(f"\n[A] 여러 행 번호가 붙은 시료 {n:,}행 — 제품명 유사도(1=같거나 포함)")
for nm, t in [('현재 규칙(삭제X→최신수정→대장번호)', cur), ('보고신고일 ≤ 기준일 중 최근', tv), ('보고신고일 ≤ 기준일 & 생산종료 없음/이후 우선', tv2), ('(상한) 이름 제일 비슷한 행', orc)]:
    print(f"   {nm}: 같거나 포함 {(t['s'] == 1).mean():.1%} · 유사도≥0.3 {(t['s'] >= .3).mean():.1%}")
diff_i = cur.index[cur['prdlst_report_ledg_no'].values != tv.loc[cur.index, 'prdlst_report_ledg_no'].values]
print(f"   현재 규칙 ≠ 보고신고일 규칙: {len(diff_i)}행 → 그중 이름 일치가 현재 규칙 쪽 {(cur.loc[diff_i,'s'] > tv.loc[diff_i,'s']).sum()} · 보고신고일 쪽 {(cur.loc[diff_i,'s'] < tv.loc[diff_i,'s']).sum()} · 같음 {(cur.loc[diff_i,'s'] == tv.loc[diff_i,'s']).sum()}")
diff2 = cur.index[cur['prdlst_report_ledg_no'].values != tv2.loc[cur.index, 'prdlst_report_ledg_no'].values]
print(f"   현재 규칙 ≠ 생산종료 반영 규칙: {len(diff2)}행 → 이름 일치 현재 쪽 {(cur.loc[diff2,'s'] > tv2.loc[diff2,'s']).sum()} · 생산종료 반영 쪽 {(cur.loc[diff2,'s'] < tv2.loc[diff2,'s']).sum()} · 같음 {(cur.loc[diff2,'s'] == tv2.loc[diff2,'s']).sum()}")
diff3 = cur.index[(cur['s'].values < orc.loc[cur.index, 's'].values)]
print(f"   현재 규칙이 고른 행보다 이름이 더 맞는 다른 행이 있는 시료: {len(diff3)}행 (번호 {R.set_index('i').loc[diff3,'no'].nunique()}개, 검출 {(R.set_index('i').loc[diff3,'judge']=='검출').sum()})")
ex = pd.DataFrame({'시료제품명': share['prduct_nm'].iloc[diff3].values, '번호': R.set_index('i').loc[diff3, 'no'].values, '기준일': base.iloc[diff3].dt.date.values,
                   '현재규칙 대장명': cur.loc[diff3, 'prdlst_nm'].values, '현재 del': cur.loc[diff3, 'live'].values, '현재 보고일': cur.loc[diff3, 'prms'].dt.date.values,
                   '이름맞는 대장명': orc.loc[diff3, 'prdlst_nm'].values, '맞는행 del': orc.loc[diff3, 'live'].values, '맞는행 보고일': orc.loc[diff3, 'prms'].dt.date.values, '맞는행 종료': orc.loc[diff3, 'end'].dt.date.values})
print('   (이하 예시 생략)')
print("   그 행들에서 유통기한 값 달라지는 행:", int(np.sum(np.asarray(cur.loc[diff3, 'pog_daycnt'].astype(object).fillna('<NA>')) != np.asarray(orc.loc[diff3, 'pog_daycnt'].astype(object).fillna('<NA>')))))

# ---- B. 이름 전혀 다른 행: 원인 추적 ----
lg1 = lg.drop_duplicates('prdlst_report_ledg_no').set_index('prdlst_report_ledg_no')
cn = share['ledg_prdlst_report_ledg_no'].map(lg1['nm_n'])
s_all = pd.Series([sim(a, b) if isinstance(b, str) else np.nan for a, b in zip(share['pn_n'].astype(object), cn.astype(object))])
bad = linked & (s_all < 0.1)
print(f"\n[B] 연결 {int(linked.sum()):,}행 중 제품명 유사도<0.1: {int(bad.sum()):,}행 (검출 {int((bad & (share['judge_result'].astype(str)=='검출')).sum())})")
# 번호 하나에 시료 제품명이 여러 개인가
g = pd.DataFrame({'no': k, 'pn': share['pn_n']})[linked].groupby('no')['pn'].nunique()
bad_no = set(k[bad])
print(f"   이름 다른 행의 번호 {len(bad_no)}개 중: 같은 번호에 시료 제품명 3개 이상 {int((g.loc[list(bad_no)] >= 3).sum())} · 1개 {int((g.loc[list(bad_no)] == 1).sum())}")
print(f"   전체 연결 번호 중 시료 제품명 3개 이상인 번호 {int((g>=3).sum())} / {len(g)}")
many = g.sort_values(ascending=False).head(5)
for no, v in many.items():
    ss = share.loc[k == no, 'prduct_nm'].value_counts().head(6).to_dict()
    print(f"   번호 {no}: 시료 제품명 {v}종 {ss} | 대장명 {lg.loc[lg['prdlst_report_no']==no, 'prdlst_nm'].tolist()}")
# 시료 제품명이 일반명(식품유형 이름과 같음)인 경우
generic = (share['pn_n'] == norm(share['db_prdlst_nm'])).fillna(False)
print(f"   이름 다른 행 중 시료 제품명 = 품목(식품유형)명: {int((bad & generic).sum())}")
# 날짜
bp = share['ledg_prdlst_report_ledg_no'].map(lg1['prms'])
print(f"   이름 다른 행 중 보고신고일 > 기준일: {int((bad & (bp > base)).sum())} (이름 맞는 연결 행에서는 {((bp > base) & linked & ~bad).sum()} / {int((linked & ~bad).sum())})")

# ---- C. 식품유형 코드 불일치 ----
lc = share['ledg_prdlst_report_ledg_no'].map(lg1['prdlst_cd'])
neq = linked & (lc.astype('string') != share['prdlst_code'].astype('string')).fillna(True)
print(f"\n[C] 대장 품목코드 ≠ 시료 품목코드: {int(neq.sum()):,}행 / {int(linked.sum()):,}")
pairs = pd.DataFrame({'시료': share['prdlst_code'], '시료품목명': share['db_prdlst_nm'], '대장': lc})[neq].value_counts().head(12)
print(pairs.to_string())
print(f"   코드 불일치 행 중 제품명 유사도≥0.3: {(s_all[neq] >= 0.3).mean():.1%} / 코드 일치 행: {(s_all[linked & ~neq] >= 0.3).mean():.1%}")
print("   시료 코드 첫 3글자 분포(불일치):", share.loc[neq, 'prdlst_code'].astype(str).str[:3].value_counts().head(5).to_dict(), " 대장:", lc[neq].astype(str).str[:3].value_counts().head(5).to_dict())

# ---- D. 숫자만 일치 이름 다른 행 ----
dg = (share['ledg_match_type'].astype('string') == '숫자만 일치').fillna(False)
print("\n[D] 숫자만 일치 & 이름 유사도<0.3:")
print(pd.DataFrame({'시료': share['prduct_nm'], '번호원문': raw, '대장': share['ledg_prdlst_report_ledg_no'].map(lg1['prdlst_nm']), 's': s_all})[dg & (s_all < 0.3)].drop_duplicates('번호원문').to_string())
# 하이픈 형식: 앞 11자리 + '-' + 일련
hy = raw[dg].str.extract(r'^(\d+)-(\d+)$')
print("   '-' 앞 자릿수:", hy[0].str.len().value_counts().to_dict(), " 뒤 자릿수:", hy[1].str.len().value_counts().to_dict())

# ---- E. 연결 안 된 번호 ----
un = raw.notna() & (raw.str.strip() != '') & ~linked
u = raw[un]
print(f"\n[E] 연결 안 된 {int(un.sum())}행: 숫자 아닌 글자만/너무 짧음(<11자리 숫자) {int(((dig[un].str.len()) < 11).sum())} · 11자리 이상 숫자 {int(((dig[un].str.len()) >= 11).sum())}")
print("   11자리 이상 예:", u[dig[un].str.len() >= 11].value_counts().head(8).to_dict())
