# HACCP 원자료 점검(가공 전): 행 수, 겹침, 날짜 분포, 같은 업소 여러 행, 주소 꼴. 결과는 화면과 텍스트 파일로.
# 실행: <venv>\python.exe -I haccp_check.py <svc: mfds|cert> <출력 txt>
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from haccp_load import load

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 50); pd.set_option('display.max_colwidth', 80)
svc, outp = sys.argv[1], sys.argv[2]
lines = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)

df, files, totals = load(svc)
P(f'== {svc}: 파일 {len(files)}개, 행 {len(df):,}, 응답 totalCount {totals}')
P('칼럼:', list(df.columns))
cols = [c for c in df.columns if not c.startswith('_')]
P('모든 칼럼이 같은 행(완전 중복):', int(df.duplicated(cols).sum()))
for c in cols:
    P(f'  {c}: 빈 값 {int(df[c].isna().sum() + (df[c].fillna("").str.strip() == "").sum() - df[c].isna().sum())}, 고유 {df[c].nunique():,}')

def year_dist(s, name):
    d = pd.to_datetime(s, format='%Y%m%d' if s.dropna().str.fullmatch(r'\d{8}').all() else '%Y-%m-%d', errors='coerce')
    P(f'-- {name}: 날짜로 못 읽음 {int(d.isna().sum() - s.isna().sum())}, 빈 값 {int(s.isna().sum())}, 최소 {d.min()}, 최대 {d.max()}')
    P(d.dt.year.value_counts().sort_index().to_string())
    return d

if svc == 'mfds':
    d = year_dist(df['HACCP_APPN_DT'], 'HACCP_APPN_DT 연도별 행 수')
    P('-- 오늘(2026-10-19) 이후 날짜 행:', int((d > '2026-10-19').sum()))
    P('-- INDUTY_CD_NM:'); P(df['INDUTY_CD_NM'].value_counts(dropna=False).to_string())
    P('-- SSIZESCALE_BSSH_YN:'); P(df['SSIZESCALE_BSSH_YN'].value_counts(dropna=False).to_string())
    # 같은 업소 여러 행
    g = df.groupby('LCNS_NO')
    n = g.size()
    P(f'-- LCNS_NO 고유 {len(n):,}, 2행 이상인 LCNS_NO {int((n>1).sum()):,} (그 행 합 {int(n[n>1].sum()):,}), 최대 {n.max()}')
    P('   행 수별 LCNS_NO 개수:', n.value_counts().sort_index().head(15).to_dict())
    k = df.groupby('LCNS_NO').agg(appn_no=('HACCP_APPN_NO', 'nunique'), appn_dt=('HACCP_APPN_DT', 'nunique'), prd=('PRDLST_NM', 'nunique'), nm=('BSSH_NM', 'nunique'), addr=('SITE_ADDR', 'nunique'))
    m = k[n > 1]
    P('   2행 이상 LCNS_NO 중: 지정번호 2개 이상', int((m.appn_no > 1).sum()), '/ 지정일 2개 이상', int((m.appn_dt > 1).sum()), '/ 품목명 2개 이상', int((m.prd > 1).sum()), '/ 업체명 2개 이상', int((m.nm > 1).sum()), '/ 주소 2개 이상', int((m.addr > 1).sum()))
    P('   (LCNS_NO, PRDLST_NM) 겹치는 행:', int(df.duplicated(['LCNS_NO', 'PRDLST_NM']).sum()))
    P('   (LCNS_NO, HACCP_APPN_NO, PRDLST_NM) 겹치는 행:', int(df.duplicated(['LCNS_NO', 'HACCP_APPN_NO', 'PRDLST_NM']).sum()))
    a = df.groupby('HACCP_APPN_NO').agg(lc=('LCNS_NO', 'nunique'), r=('LCNS_NO', 'size'))
    P(f'   HACCP_APPN_NO 고유 {len(a):,}, 여러 LCNS_NO 에 걸친 지정번호 {int((a.lc>1).sum()):,}')
    # 예시: 품목명 여러 개·지정일 여러 개인 업소
    ex = m[(m.appn_dt > 1)].head(5).index
    P('   예시(지정일 여러 개인 업소, 업체명 제외):')
    P(df[df.LCNS_NO.isin(ex)][['LCNS_NO', 'INDUTY_CD_NM', 'SITE_ADDR', 'HACCP_APPN_DT', 'HACCP_APPN_NO', 'PRDLST_NM']].sort_values(['LCNS_NO', 'HACCP_APPN_DT']).to_string())
    # 지정번호 앞 4자리(연도)와 지정일 연도 비교
    yr_no = df['HACCP_APPN_NO'].str.extract(r'^(\d{4})')[0]
    P('-- HACCP_APPN_NO 앞 4자리 = HACCP_APPN_DT 연도 인 행:', int((yr_no == df['HACCP_APPN_DT'].str[:4]).sum()), '/ 다른 행:', int(((yr_no != df['HACCP_APPN_DT'].str[:4]) & yr_no.notna()).sum()), '/ 앞 4자리 숫자 아님:', int(yr_no.isna().sum()))
    P('   HACCP_APPN_NO 꼴 예:', df['HACCP_APPN_NO'].str.replace(r'\d', '9', regex=True).value_counts().head(10).to_dict())
    # 주소
    s = df['SITE_ADDR'].fillna('')
    P('-- SITE_ADDR 낱말 수별 행:', s.str.split().str.len().value_counts().sort_index().to_dict())
    P('-- SITE_ADDR 고유값(행 수):'); P(s.value_counts().to_string())
else:
    d1 = year_dist(df['issuedate'], 'issuedate 연도별 행 수')
    d2 = year_dist(df['issueenddate'], 'issueenddate 연도별 행 수')
    P('-- issueenddate 값 상위:', df['issueenddate'].value_counts().head(10).to_dict())
    P('-- 오늘(2026-10-19)보다 이른 issueenddate 행:', int((d2 < '2026-10-19').sum()))
    P('-- issuedate > issueenddate 행:', int((d1 > d2).sum()))
    for c in ['productGb', 'companykindNm', 'businesstypeNm']:
        P(f'-- {c}:'); P(df[c].value_counts(dropna=False).head(60).to_string())
    P('-- area1:'); P(df['area1'].value_counts(dropna=False).to_string())
    n = df.groupby('appointno').size()
    P(f'-- appointno 고유 {len(n):,}, 2행 이상 {int((n>1).sum()):,} (그 행 합 {int(n[n>1].sum()):,}), 최대 {n.max()}')
    P('   (appointno, businessitem) 겹침:', int(df.duplicated(['appointno', 'businessitem']).sum()))
    ex = n[n > 1].head(3).index
    P(df[df.appointno.isin(ex)][[c for c in cols if c != 'company']].to_string())
    c2 = df.groupby(['company', 'area1', 'area2']).agg(r=('appointno', 'size'), ap=('appointno', 'nunique'), d=('issuedate', 'nunique'))
    P(f'-- (company, area1, area2) 묶음 {len(c2):,}, 인증번호 2개 이상인 묶음 {int((c2.ap>1).sum()):,}, 인증일 2개 이상 {int((c2.d>1).sum()):,}')
    P('   appointno 꼴 예:', df['appointno'].str.replace(r'\d', '9', regex=True).value_counts().head(15).to_dict())
    P('-- area2 낱말 수별:', df['area2'].fillna('').str.split().str.len().value_counts().sort_index().to_dict())
    P('-- (area1, area2) 고유값(행 수):'); P((df['area1'].fillna('') + ' | ' + df['area2'].fillna('')).value_counts().to_string())

open(outp, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
