# I0580 점검 4: 2017-08 전에 끝난 지정이 빠졌는지 가늠 — 15058303 에는 있고 I0580 에는 없는 업소·품목.
# 다른 자료에 날짜가 없는 번호(4,644개)의 HACCP_APPN_DT 가 최초 지정일인지. 출력: 중간\외부지표\_haccp_I0580_점검4.txt
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from i0580_load import load
import haccp_load

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
MID = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\중간\외부지표'
OUT = os.path.join(MID, '_haccp_I0580_점검4.txt')
L = []
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); L.append(s)
def pct(a, b):
    return f'{a:,}/{b:,} ({(a / b * 100 if b else 0):.2f}%)'
D = lambda s: pd.to_datetime(s.where(s.fillna('').str.strip() != ''), format='%Y%m%d', errors='coerce')

df, _, _ = load(); df['no'] = df.HACCP_APPN_NO.str.strip(); df['appn'] = D(df.HACCP_APPN_DT); df['end'] = D(df.CRTFC_ENDDT)
df['noyr'] = pd.to_numeric(df.no.str[:4], errors='coerce')
m3, _, _ = haccp_load.load('mfds'); m3['no'] = m3.HACCP_APPN_NO.fillna('').str.strip(); m3['d'] = D(m3.HACCP_APPN_DT)
c3, _, _ = haccp_load.load('cert'); c3['no'] = c3.appointno.fillna('').str.strip()

P('## 1. 15058303 에는 있고 I0580 에는 없는 업소(LCNS_NO)')
miss = m3[~m3.LCNS_NO.isin(set(df.LCNS_NO))]
P('업소', miss.LCNS_NO.nunique(), '/ 15058303 업소', m3.LCNS_NO.nunique(), f'({miss.LCNS_NO.nunique() / m3.LCNS_NO.nunique() * 100:.2f}%)', '/ 행', len(miss))
fy = miss.groupby('LCNS_NO').d.min().dt.year.value_counts().sort_index()
P('  그 업소의 가장 이른 지정일 연도:', fy.to_dict())
P('  그 업소의 업종:', miss.drop_duplicates('LCNS_NO').INDUTY_CD_NM.value_counts(dropna=False).to_dict())
P('  그 업소 행의 번호가 15033311(지금 유효) 에 있음:', pct(int(miss.no.isin(set(c3.no)).sum()), len(miss)))
allfy = m3.groupby('LCNS_NO').d.min().dt.year.value_counts().sort_index()
P('  (비교) 15058303 전체 업소의 가장 이른 지정일 연도:', allfy.to_dict())
P('  연도별 빠진 비율(%):', {int(y): round(fy.get(y, 0) / allfy[y] * 100, 1) for y in allfy.index})

P('\n## 2. 옛 꼴 번호 행 중 같은 LCNS_NO·PRDLST_NM 이 I0580 에 없는 행')
old = m3[~m3.no.str.fullmatch(r'\d{4}-\d-\d{4}') & (m3.no != '')]
key_i = set(zip(df.LCNS_NO, df.PRDLST_NM))
old = old.assign(hit=[(a, b) in key_i for a, b in zip(old.LCNS_NO, old.PRDLST_NM)])
nh = old[~old.hit]
P('행', len(nh), '/ 옛 꼴 행', len(old), '/ 품목명 "-"·빈 값', int(nh.PRDLST_NM.fillna('').str.strip().isin(['', '-']).sum()))
P('  지정일 연도:', nh.d.dt.year.value_counts().sort_index().to_dict())
P('  품목명 상위 15:', nh.PRDLST_NM.value_counts().head(15).to_dict())
# 같은 LCNS_NO 의 I0580 품목명과 글자 일부가 겹치는지(이름만 바뀐 경우 가늠)
pi = df.groupby('LCNS_NO').PRDLST_NM.apply(set)
def near(r):
    s = pi.get(r.LCNS_NO)
    if not s or not isinstance(r.PRDLST_NM, str) or len(r.PRDLST_NM) < 2:
        return False
    a = re.sub(r'\W', '', r.PRDLST_NM)
    return any(a and (a in re.sub(r'\W', '', b) or re.sub(r'\W', '', b) in a) for b in s if b)
nh2 = nh[nh.PRDLST_NM.fillna('').str.strip().ne('-')]
P('  ("-" 빼고) 같은 LCNS_NO 의 I0580 품목명과 글자가 한쪽에 들어가는 행:', pct(int(nh2.apply(near, axis=1).sum()), len(nh2)))

P('\n## 3. 다른 두 자료에 날짜가 없는 번호의 HACCP_APPN_DT')
s58 = set(m3.no); s33 = set(c3.no)
num = df.groupby('no').agg(appn=('appn', 'min'), noyr=('noyr', 'first'), end=('end', 'max'))
nn = num[~num.index.isin(s58) & ~num.index.isin(s33)]
P('번호', len(nn), '/ 번호 연도 = 가장 이른 HACCP_APPN_DT 연도:', pct(int((nn.noyr == nn.appn.dt.year).sum()), len(nn)))
P('  (가장 이른 HACCP_APPN_DT 연도 − 번호 연도) 분포:', (nn.appn.dt.year - nn.noyr).value_counts().sort_index().to_dict())
P('  번호 연도 분포:', nn.noyr.value_counts().sort_index().to_dict())
P('  그 번호의 끝(CRTFC_ENDDT 최대)이 받은 날 이전:', pct(int((nn.end < pd.Timestamp('2026-10-19')).sum()), len(nn)))
ok = num[num.index.isin(s58) | num.index.isin(s33)]
P('(비교) 다른 자료에 날짜가 있는 번호: 번호 연도 = 가장 이른 HACCP_APPN_DT 연도:', pct(int((ok.noyr == ok.appn.dt.year).sum()), len(ok)))

open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('저장:', OUT)
