# I0580 점검: 받은 행 수·겹침/빠짐, 날짜 칼럼(채워진 비율·연도 분포·같은 값 몰림), 영업상태, 15058303·15033311 과 맞대기.
# 출력: 중간\외부지표\_haccp_I0580_점검.txt. venv 파이썬(pandas), 네트워크 안 씀.
# 실행: <리스테리아ML\.venv-exp2>\python.exe -I i0580_check.py
import sys, os, re, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from i0580_load import load, COLS
import haccp_load

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
MID = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\중간\외부지표'
OUT = os.path.join(MID, '_haccp_I0580_점검.txt')
L = []
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); L.append(s)
def pct(a, b):
    return f'{a:,}/{b:,} ({(a / b * 100 if b else 0):.2f}%)'

df, files, info = load()
N = len(df)
P('# I0580 점검', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
P('\n## 1. 받은 파일·행 수')
P('파일', len(files), '개, 행', f'{N:,}')
P('total_count 값들:', info.total_count.value_counts().to_dict())
P('구간별 행 수 값들:', info.n.value_counts().to_dict())
P('구간 끝-시작+1 과 행 수가 다른 파일:', info[(info.end - info.start + 1) != info.n].to_dict('records'))
P('칼럼:', [c for c in df.columns if not c.startswith('_')])
P('응답에 없는(빈) 칼럼 확인 — 칼럼별 None 수:', {c: int(df[c].isna().sum()) for c in COLS if c in df})
dup_all = df.duplicated(COLS, keep=False)
P('완전 중복(13칼럼 모두 같은) 행:', int(dup_all.sum()), '/ 처음 것 빼고:', int(df.duplicated(COLS).sum()))
# 구간 경계 겹침: 앞 구간 마지막 5행과 뒤 구간 첫 5행이 같은지
keyf = lambda r: tuple(r[c] for c in COLS)
bnd = 0
for s in sorted(df._start.unique())[1:]:
    prev = df[df._start < s]._start.max()
    a = set(df[(df._start == prev)].tail(20).apply(keyf, axis=1))
    b = set(df[(df._start == s)].head(20).apply(keyf, axis=1))
    bnd += len(a & b)
P('구간 경계(앞 구간 끝 20행 × 뒤 구간 첫 20행)에서 같은 행 수:', bnd)
if dup_all.any():
    d = df[dup_all].sort_values(COLS)
    P('완전 중복 행의 구간 시작 번호 분포:', d._start.value_counts().sort_index().to_dict())
    P(d.head(10)[COLS + ['_rownum']].to_string())

P('\n## 2. 날짜 칼럼')
DATES = ['HACCP_APPN_DT', 'CRTFC_ENDDT', 'ASGN_CANCL_DT', 'CRTFC_RETN_DT', 'CLSBIZ_DT']
def to_date(s):
    if s is None or not str(s).strip():
        return None
    s = str(s).strip()
    if re.fullmatch(r'\d{8}', s):
        try:
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))
        except ValueError:
            return 'bad'
    return 'bad'
for c in DATES:
    v = df[c]
    filled = v.notna() & (v.str.strip() != '')
    dd = v.map(to_date)
    bad = int((dd == 'bad').sum())
    P(f'\n### {c}')
    P('  채워진 행:', pct(int(filled.sum()), N), '/ 날짜로 못 읽은 값:', bad, (v[dd == 'bad'].value_counts().head(10).to_dict() if bad else ''))
    if filled.any():
        vc = v[filled].value_counts()
        P('  고유값 수:', len(vc))
        P('  가장 많은 값 10개(행, 채워진 행 중 %):', ', '.join(f'{k}: {n:,} ({n / filled.sum() * 100:.2f}%)' for k, n in vc.head(10).items()))
        P('  20261019(받은 날) 인 행:', pct(int((v == '20261019').sum()), N), '/ 20291018 인 행:', pct(int((v == '20291018').sum()), N))
        ok = dd[dd.map(lambda x: isinstance(x, datetime.date))]
        P('  가장 이른·늦은 값:', min(ok), '~', max(ok))
        y = ok.map(lambda x: x.year).value_counts().sort_index()
        P('  연도별 행:', ', '.join(f'{k}: {n:,}' for k, n in y.items()))

P('\n## 3. 칼럼끼리 비교')
fill = lambda c: df[c].notna() & (df[c].str.strip() != '')
both = fill('ASGN_CANCL_DT') & fill('CRTFC_ENDDT')
P('ASGN_CANCL_DT 와 CRTFC_ENDDT 둘 다 채워진 행:', pct(int(both.sum()), N), '/ 그중 두 값이 같은 행:', pct(int((both & (df.ASGN_CANCL_DT == df.CRTFC_ENDDT)).sum()), int(both.sum())))
P('ASGN_CANCL_DT 만 채워짐:', int((fill('ASGN_CANCL_DT') & ~fill('CRTFC_ENDDT')).sum()), '/ CRTFC_ENDDT 만:', int((~fill('ASGN_CANCL_DT') & fill('CRTFC_ENDDT')).sum()), '/ 둘 다 빔:', int((~fill('ASGN_CANCL_DT') & ~fill('CRTFC_ENDDT')).sum()))
diff = df[both & (df.ASGN_CANCL_DT != df.CRTFC_ENDDT)]
if len(diff):
    P('두 값이 다른 행 예(최대 15):'); P(diff.head(15)[['LCNS_NO', 'HACCP_APPN_NO', 'HACCP_APPN_DT', 'CRTFC_ENDDT', 'ASGN_CANCL_DT', 'CRTFC_RETN_DT', 'CLSBIZ_DT', 'CLSBIZ_DVS_CD_NM']].to_string())
# 지정일과 종료일 차이
a = df.HACCP_APPN_DT.map(to_date); e = df.CRTFC_ENDDT.map(to_date)
m = a.map(lambda x: isinstance(x, datetime.date)) & e.map(lambda x: isinstance(x, datetime.date))
gap = pd.Series([(y - x).days for x, y in zip(a[m], e[m])], index=a[m].index)
P('CRTFC_ENDDT − HACCP_APPN_DT (일) 값별 행 상위 10:', gap.value_counts().head(10).to_dict())
P('HACCP_APPN_DT 가 받은 날(20261019)이 아닌 행:', pct(int(((df.HACCP_APPN_DT != '20261019') & fill('HACCP_APPN_DT')).sum()), N))
# 폐업일보다 지정일이 늦은 행
c = df.CLSBIZ_DT.map(to_date)
mc = m & c.map(lambda x: isinstance(x, datetime.date))
late = pd.Series([x > y for x, y in zip(a[mc], c[mc])], index=a[mc].index)
P('CLSBIZ_DT 와 HACCP_APPN_DT 둘 다 있는 행:', int(mc.sum()), '/ 그중 지정일이 폐업일보다 늦은 행:', pct(int(late.sum()), int(mc.sum())))
r = df.CRTFC_RETN_DT.map(to_date)
mr = a.map(lambda x: isinstance(x, datetime.date)) & r.map(lambda x: isinstance(x, datetime.date))
late_r = pd.Series([x > y for x, y in zip(a[mr], r[mr])], index=a[mr].index)
P('CRTFC_RETN_DT 와 HACCP_APPN_DT 둘 다 있는 행:', int(mr.sum()), '/ 그중 지정일이 반납일보다 늦은 행:', pct(int(late_r.sum()), int(mr.sum())))

P('\n## 4. 영업상태 CLSBIZ_DVS_CD_NM')
vc = df.CLSBIZ_DVS_CD_NM.value_counts(dropna=False)
P(vc.to_string())
t = pd.DataFrame({'상태': df.CLSBIZ_DVS_CD_NM.fillna('(None)'), '폐업일있음': fill('CLSBIZ_DT'), '반납일있음': fill('CRTFC_RETN_DT')})
P('영업상태 × 폐업일자 채워짐:'); P(pd.crosstab(t.상태, t.폐업일있음).to_string())
P('영업상태 × 인증반납일자 채워짐:'); P(pd.crosstab(t.상태, t.반납일있음).to_string())
P('\n업종 INDUTY_CD_NM:'); P(df.INDUTY_CD_NM.value_counts(dropna=False).to_string())
P('\n## 5. 업소·번호')
P('LCNS_NO 고유:', df.LCNS_NO.nunique(), '/ 빈 값:', int((~fill('LCNS_NO')).sum()))
P('HACCP_APPN_NO 고유:', df.HACCP_APPN_NO.nunique(), '/ 빈 값:', int((~fill('HACCP_APPN_NO')).sum()))
P('(LCNS_NO, HACCP_APPN_NO) 고유:', len(df[['LCNS_NO', 'HACCP_APPN_NO']].drop_duplicates()))
P('(LCNS_NO, HACCP_APPN_NO, PRDLST_NM) 고유:', len(df[['LCNS_NO', 'HACCP_APPN_NO', 'PRDLST_NM']].drop_duplicates()))
P('(LCNS_NO, HACCP_APPN_NO, PRDLST_NM, SITE_ADDR) 고유:', len(df[['LCNS_NO', 'HACCP_APPN_NO', 'PRDLST_NM', 'SITE_ADDR']].drop_duplicates()))
pat = df.HACCP_APPN_NO.fillna('').str.strip().map(lambda s: 'YYYY-N-NNNN' if re.fullmatch(r'\d{4}-\d-\d{4}', s) else ('YYYY-NNN' if re.fullmatch(r'\d{4}-\d+', s) else ('빈 값' if not s else '그 밖')))
P('HACCP_APPN_NO 꼴:', pat.value_counts().to_dict())
P('HACCP_APPN_NO 앞 4자리(연도로 보임) 분포:', df.HACCP_APPN_NO.fillna('').str.strip().str[:4].value_counts().sort_index().to_dict())
g = df.groupby('HACCP_APPN_NO').agg(n=('LCNS_NO', 'size'), nl=('LCNS_NO', 'nunique'), ns=('CLSBIZ_DVS_CD_NM', 'nunique'), nd=('HACCP_APPN_DT', 'nunique'))
P('지정번호당 행 수 분포:', g.n.value_counts().sort_index().head(15).to_dict())
P('지정번호 하나에 LCNS_NO 2개 이상:', int((g.nl > 1).sum()), '/ 영업상태 2개 이상:', int((g.ns > 1).sum()), '/ 지정일 2개 이상:', int((g.nd > 1).sum()))
P('PRDLST_NM 빈 값·"-":', int((~fill('PRDLST_NM') | (df.PRDLST_NM.str.strip() == '-')).sum()))
P('PRSDNT_NM 채워진 행:', int(fill('PRSDNT_NM').sum()))
P('SITE_ADDR 빈 값:', int((~fill('SITE_ADDR')).sum()))
P('같은 LCNS_NO·HACCP_APPN_NO·PRDLST_NM 인데 영업상태가 다른 묶음 수:',
  int((df.groupby(['LCNS_NO', 'HACCP_APPN_NO', 'PRDLST_NM']).CLSBIZ_DVS_CD_NM.nunique() > 1).sum()))

P('\n## 6. 15058303(식약처 지정현황, 지정일 2021-01 까지)과 맞대기')
m3, _, _ = haccp_load.load('mfds')
m3['no'] = m3.HACCP_APPN_NO.fillna('').str.strip(); df['no'] = df.HACCP_APPN_NO.fillna('').str.strip()
P('15058303 행', len(m3), '/ HACCP_APPN_NO 고유', m3.no.nunique(), '/ LCNS_NO 고유', m3.LCNS_NO.nunique())
s_i = set(df.no) - {''}; s_m = set(m3.no) - {''}
P('지정번호: I0580 고유', len(s_i), '/ 15058303 고유', len(s_m), '/ 둘 다', len(s_i & s_m),
  f'(15058303 번호 중 {len(s_i & s_m) / len(s_m) * 100:.2f}% 가 I0580 에 있음, I0580 번호 중 {len(s_i & s_m) / len(s_i) * 100:.2f}% 가 15058303 에 있음)')
l_i = set(df.LCNS_NO.dropna()) - {''}; l_m = set(m3.LCNS_NO.dropna()) - {''}
P('LCNS_NO: I0580 고유', len(l_i), '/ 15058303 고유', len(l_m), '/ 둘 다', len(l_i & l_m), f'(15058303 의 {len(l_i & l_m) / len(l_m) * 100:.2f}%)')
# 같은 번호+같은 LCNS_NO 끼리 날짜 비교
mi = df[['no', 'LCNS_NO', 'HACCP_APPN_DT']].drop_duplicates()
mm = m3[['no', 'LCNS_NO', 'HACCP_APPN_DT']].drop_duplicates()
j = mi.merge(mm, on=['no', 'LCNS_NO'], suffixes=('_I0580', '_15058303'))
j = j[j.no != '']
P('(지정번호, LCNS_NO) 가 같은 짝:', len(j), '/ 그중 지정일이 같은 짝:', pct(int((j.HACCP_APPN_DT_I0580 == j.HACCP_APPN_DT_15058303).sum()), len(j)))
P('같은 짝에서 I0580 지정일 값 상위:', j.HACCP_APPN_DT_I0580.value_counts().head(5).to_dict())
P('같은 짝에서 15058303 지정일 연도 상위:', j.HACCP_APPN_DT_15058303.str[:4].value_counts().head(10).to_dict())
# I0580 에서 지정일이 받은 날이 아닌 행만
j2 = j[j.HACCP_APPN_DT_I0580 != '20261019']
P('그중 I0580 지정일이 20261019 가 아닌 짝:', len(j2), '/ 15058303 지정일과 같은 짝:', int((j2.HACCP_APPN_DT_I0580 == j2.HACCP_APPN_DT_15058303).sum()))
if len(j2):
    P(j2.head(15).to_string())
# 15058303 에 있는데 I0580 에 없는 번호의 성격
gone = m3[(m3.no != '') & ~m3.no.isin(s_i)]
P('15058303 에만 있는 지정번호의 행', len(gone), '/ 고유 번호', gone.no.nunique(), '/ 그 지정일 연도 분포:', gone.HACCP_APPN_DT.str[:4].value_counts().sort_index().to_dict())

P('\n## 7. 15033311(인증원, 지금 유효한 인증만)과 맞대기')
c3, _, _ = haccp_load.load('cert')
c3['no'] = c3.appointno.fillna('').str.strip()
s_c = set(c3.no) - {''}
P('15033311 appointno 고유', len(s_c), '/ I0580 지정번호와 같은 것', len(s_i & s_c),
  f'(I0580 번호 중 {len(s_i & s_c) / len(s_i) * 100:.2f}%, 15033311 번호 중 {len(s_i & s_c) / len(s_c) * 100:.2f}%)')
P('15033311 productGb 별 I0580 에 있는 비율:', c3.assign(hit=c3.no.isin(s_i)).groupby('productGb').hit.mean().round(4).to_dict())
jc = df[['no', 'HACCP_APPN_DT', 'CRTFC_ENDDT', 'ASGN_CANCL_DT', 'CLSBIZ_DVS_CD_NM']].drop_duplicates().merge(
    c3[['no', 'issuedate', 'issueenddate']], on='no')
jc['issuedate8'] = jc.issuedate.str.replace('-', ''); jc['issueenddate8'] = jc.issueenddate.str.replace('-', '')
P('번호가 같은 짝:', len(jc))
P('  HACCP_APPN_DT = issuedate:', pct(int((jc.HACCP_APPN_DT == jc.issuedate8).sum()), len(jc)))
P('  CRTFC_ENDDT = issueenddate:', pct(int((jc.CRTFC_ENDDT == jc.issueenddate8).sum()), len(jc)))
P('  ASGN_CANCL_DT = issueenddate:', pct(int((jc.ASGN_CANCL_DT == jc.issueenddate8).sum()), len(jc)))
P('  issuedate 값 예(상위):', jc.issuedate.value_counts().head(5).to_dict())
P('  issueenddate 값 예(상위):', jc.issueenddate.value_counts().head(5).to_dict())
P('  짝의 I0580 영업상태:', jc.CLSBIZ_DVS_CD_NM.value_counts().to_dict())
P('I0580 영업상태별, 15033311 에 번호가 있는 비율:', df.assign(hit=df.no.isin(s_c)).groupby('CLSBIZ_DVS_CD_NM').hit.agg(['size', 'mean']).round(4).to_string())
P('I0580 업종별, 15033311 에 번호가 있는 비율:', df.assign(hit=df.no.isin(s_c)).groupby('INDUTY_CD_NM').hit.agg(['size', 'mean']).round(4).to_string())
P('I0580 업종별, 15058303 에 번호가 있는 비율:', df.assign(hit=df.no.isin(s_m)).groupby('INDUTY_CD_NM').hit.agg(['size', 'mean']).round(4).to_string())

open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('저장:', OUT)
