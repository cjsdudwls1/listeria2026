# I0580 추가 점검: 지정일이 '마지막 연장일'인지(3년 단위), 지정번호 앞 4자리 연도와 최초 지정일, 끝난 지정이 들어 있는지,
# ASGN_CANCL_DT 가 CRTFC_ENDDT 와 다른 행, 2017-08-03 몰림, 15058303 에만 있는 번호의 성격.
# 출력: 중간\외부지표\_haccp_I0580_점검2.txt. venv 파이썬(pandas), 네트워크 안 씀.
import sys, os, re, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from i0580_load import load, COLS
import haccp_load

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
MID = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\중간\외부지표'
OUT = os.path.join(MID, '_haccp_I0580_점검2.txt')
TODAY = pd.Timestamp('2026-10-19')
L = []
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); L.append(s)
def pct(a, b):
    return f'{a:,}/{b:,} ({(a / b * 100 if b else 0):.2f}%)'
def D(s):
    return pd.to_datetime(s.where(s.fillna('').str.strip() != ''), format='%Y%m%d', errors='coerce')

df, files, info = load()
df['no'] = df.HACCP_APPN_NO.str.strip()
for c in ['HACCP_APPN_DT', 'CRTFC_ENDDT', 'ASGN_CANCL_DT', 'CRTFC_RETN_DT', 'CLSBIZ_DT']:
    df[c + '_d'] = D(df[c])
N = len(df)

P('## A. 원자료 행 순서')
a = df.HACCP_APPN_DT_d
P('행 순서대로 HACCP_APPN_DT 가 늘지 않는(같거나 줄어드는) 곳 비율:', pct(int((a.diff().dropna() <= pd.Timedelta(0)).sum()), N - 1))
P('처음 5행 / 마지막 5행 HACCP_APPN_DT:', df.HACCP_APPN_DT.head(5).tolist(), df.HACCP_APPN_DT.tail(5).tolist())

def years_between(x, y):
    """x→y 가 몇 년 몇 일 뒤인지: (같은 월일이면 년 차이, 아니면 None)"""
    return [(b.year - a_.year) if (a_.month, a_.day) == (b.month, b.day) else None for a_, b in zip(x, y)]

P('\n## B. 15058303 지정일과 비교 (지정번호·LCNS_NO 가 같은 짝)')
m3, _, _ = haccp_load.load('mfds')
m3['no'] = m3.HACCP_APPN_NO.fillna('').str.strip()
m3['d'] = D(m3.HACCP_APPN_DT)
j = (df[['no', 'LCNS_NO', 'HACCP_APPN_DT_d']].drop_duplicates()
     .merge(m3[['no', 'LCNS_NO', 'd']].drop_duplicates(), on=['no', 'LCNS_NO']))
j = j[j.no != '']
j['yd'] = years_between(j.d, j.HACCP_APPN_DT_d)
j['gap'] = (j.HACCP_APPN_DT_d - j.d).dt.days
P('짝', len(j))
P('I0580 지정일 = 15058303 지정일과 월·일이 같은 짝:', pct(int(j.yd.notna().sum()), len(j)))
P('  그 짝의 년 차이 분포:', j.yd.value_counts().sort_index().to_dict())
P('  년 차이가 3의 배수인 짝:', pct(int(j.yd.dropna().map(lambda v: v % 3 == 0).sum()), int(j.yd.notna().sum())))
nm = j[j.yd.isna()]
P('월·일이 다른 짝', len(nm), '— I0580 지정일 − 15058303 지정일(일) 분포 요약:', nm.gap.describe().round(1).to_dict())
P('  I0580 지정일이 15058303 지정일보다 이른 짝:', pct(int((j.gap < 0).sum()), len(j)))
# 하루 차이까지 보아 3년 주기에 맞는지
def cyc(g):
    # g 일이 3년 배수(1095~1096일 × k) ± 3일 안인지
    if pd.isna(g) or g < 0:
        return None
    k = round(g / 1095.75)
    return k if abs(g - k * 1095.75) <= 3 else -1
j['cyc'] = j.gap.map(cyc)
P('지정일 차이가 3년의 배수(±3일)인 짝:', pct(int((j.cyc >= 0).sum()), len(j)), '/ k 분포:', j.cyc.value_counts().sort_index().to_dict())

P('\n## C. 15033311 issuedate 와 비교 (번호가 같은 짝)')
c3, _, _ = haccp_load.load('cert')
c3['no'] = c3.appointno.fillna('').str.strip()
c3['iss'] = pd.to_datetime(c3.issuedate, errors='coerce'); c3['iend'] = pd.to_datetime(c3.issueenddate, errors='coerce')
jc = df[['no', 'HACCP_APPN_DT_d', 'CRTFC_ENDDT_d']].drop_duplicates().merge(c3[['no', 'iss', 'iend']], on='no')
jc['gap'] = (jc.HACCP_APPN_DT_d - jc.iss).dt.days
jc['cyc'] = jc.gap.map(cyc)
P('짝', len(jc), '/ 지정일 = issuedate:', pct(int((jc.gap == 0).sum()), len(jc)), '/ 지정일이 issuedate 보다 이름:', pct(int((jc.gap < 0).sum()), len(jc)))
P('지정일 − issuedate 가 3년의 배수(±3일):', pct(int((jc.cyc >= 0).sum()), len(jc)), '/ k 분포:', jc.cyc.value_counts().sort_index().to_dict())

P('\n## D. 지정번호 앞 4자리 연도와 최초 지정일 연도')
df['noyr'] = pd.to_numeric(df.no.str[:4], errors='coerce')
j['noyr'] = pd.to_numeric(j.no.str[:4], errors='coerce')
P('15058303 짝: 번호 앞 4자리 = 15058303 지정일 연도:', pct(int((j.noyr == j.d.dt.year).sum()), len(j)),
  '/ 차이(지정일 연도 − 번호 연도) 분포:', (j.d.dt.year - j.noyr).value_counts().sort_index().head(15).to_dict())
jc['noyr'] = pd.to_numeric(jc.no.str[:4], errors='coerce')
P('15033311 짝: 번호 앞 4자리 = issuedate 연도:', pct(int((jc.noyr == jc.iss.dt.year).sum()), len(jc)),
  '/ 차이(issuedate 연도 − 번호 연도) 분포:', (jc.iss.dt.year - jc.noyr).value_counts().sort_index().head(15).to_dict())
P('I0580 전체: 번호 앞 4자리 ≤ 지정일 연도:', pct(int((df.noyr <= df.HACCP_APPN_DT_d.dt.year).sum()), N))
# 15033311 짝에서 번호 앞 4자리 연도 vs issuedate 연도: 번호 연도가 늦은 경우 예
x = jc[jc.noyr != jc.iss.dt.year].head(10)
P('번호 연도 ≠ issuedate 연도 예:'); P(x.to_string())

P('\n## E. 끝난 지정이 들어 있는지 (CRTFC_ENDDT 가 받은 날 2026-10-19 보다 이른 행)')
ended = df.CRTFC_ENDDT_d < TODAY
P('CRTFC_ENDDT < 2026-10-19 인 행:', pct(int(ended.sum()), N))
P('  그 행의 영업상태:', df[ended].CLSBIZ_DVS_CD_NM.value_counts().to_dict())
P('  그 행 중 CRTFC_RETN_DT 있음:', int(df[ended].CRTFC_RETN_DT_d.notna().sum()), '/ CLSBIZ_DT 있음:', int(df[ended].CLSBIZ_DT_d.notna().sum()),
  '/ 둘 다 없음:', int((df[ended].CRTFC_RETN_DT_d.isna() & df[ended].CLSBIZ_DT_d.isna()).sum()))
P('  그 행의 CRTFC_ENDDT 연도:', df[ended].CRTFC_ENDDT_d.dt.year.value_counts().sort_index().to_dict())
P('CRTFC_ENDDT ≥ 2026-10-19 인 행의 영업상태:', df[~ended].CLSBIZ_DVS_CD_NM.value_counts().to_dict(),
  '/ 그중 반납일 있음:', int(df[~ended].CRTFC_RETN_DT_d.notna().sum()))
# 반납일과 종료일 관계
r = df[df.CRTFC_RETN_DT_d.notna()]
P('반납일 있는 행', len(r), ': 반납일 ≤ CRTFC_ENDDT', int((r.CRTFC_RETN_DT_d <= r.CRTFC_ENDDT_d).sum()), '/ 반납일 > CRTFC_ENDDT', int((r.CRTFC_RETN_DT_d > r.CRTFC_ENDDT_d).sum()),
  '/ CRTFC_ENDDT − 반납일(일) 요약:', (r.CRTFC_ENDDT_d - r.CRTFC_RETN_DT_d).dt.days.describe().round(0).to_dict())
P('반납일 = CRTFC_ENDDT + 1일:', int(((r.CRTFC_RETN_DT_d - r.CRTFC_ENDDT_d).dt.days == 1).sum()), '/ 반납일 = CRTFC_ENDDT:', int((r.CRTFC_RETN_DT_d == r.CRTFC_ENDDT_d).sum()))
P('반납일 있는 행의 CRTFC_ENDDT 가 받은 날 이후:', int((r.CRTFC_ENDDT_d >= TODAY).sum()))
P('가장 늦은 반납일:', r.CRTFC_RETN_DT_d.max(), '/ 월별 반납 행(2025-07 이후):', r[r.CRTFC_RETN_DT_d >= '2025-07-01'].CRTFC_RETN_DT_d.dt.to_period('M').value_counts().sort_index().to_dict())
# 지정번호 단위
g = df.groupby('no').agg(end=('CRTFC_ENDDT_d', 'max'), retn=('CRTFC_RETN_DT_d', 'min'), retn_n=('CRTFC_RETN_DT_d', 'count'), n=('no', 'size'),
                         normal=('CLSBIZ_DVS_CD_NM', lambda s: (s == '정상').any()))
P('지정번호', len(g), '개: CRTFC_ENDDT(최대) < 받은 날', pct(int((g.end < TODAY).sum()), len(g)), '/ 반납일 있는 번호', int(g.retn_n.gt(0).sum()),
  '/ 영업상태 정상 행이 하나라도 있는 번호', int(g.normal.sum()))

P('\n## F. 2017-08-03 몰림')
b = df[df.CRTFC_ENDDT == '20170803']
P('CRTFC_ENDDT=20170803 행', len(b), '/ 지정일 분포(연도):', b.HACCP_APPN_DT_d.dt.year.value_counts().sort_index().to_dict())
P('  반납일:', b.CRTFC_RETN_DT.replace('', '(빈 값)').value_counts().head(5).to_dict(), '/ 영업상태:', b.CLSBIZ_DVS_CD_NM.value_counts().to_dict())
P('  업종:', b.INDUTY_CD_NM.value_counts().to_dict())
P('  (종료일−지정일) 일 분포 상위:', (b.CRTFC_ENDDT_d - b.HACCP_APPN_DT_d).dt.days.value_counts().head(8).to_dict())
P('CRTFC_ENDDT 가 2017-08-03 ~ 2017-12-31 인 행:', int(((df.CRTFC_ENDDT_d >= '2017-08-03') & (df.CRTFC_ENDDT_d <= '2017-12-31')).sum()),
  '/ 2018년:', int((df.CRTFC_ENDDT_d.dt.year == 2018).sum()))
P('CLSBIZ_DT < 2017-08-03 인 행', int((df.CLSBIZ_DT_d < '2017-08-03').sum()), '— 그 행의 CRTFC_ENDDT 연도:', df[df.CLSBIZ_DT_d < '2017-08-03'].CRTFC_ENDDT_d.dt.year.value_counts().sort_index().to_dict(),
  '/ 영업상태:', df[df.CLSBIZ_DT_d < '2017-08-03'].CLSBIZ_DVS_CD_NM.value_counts().to_dict())
P('CRTFC_RETN_DT < 2017-08-03 인 행', int((df.CRTFC_RETN_DT_d < '2017-08-03').sum()), '— CRTFC_ENDDT 연도:', df[df.CRTFC_RETN_DT_d < '2017-08-03'].CRTFC_ENDDT_d.dt.year.value_counts().sort_index().to_dict())

P('\n## G. ASGN_CANCL_DT ≠ CRTFC_ENDDT 인 행')
dff = df[df.ASGN_CANCL_DT != df.CRTFC_ENDDT]
P('행', len(dff), '/ ASGN_CANCL_DT < CRTFC_ENDDT:', int((dff.ASGN_CANCL_DT_d < dff.CRTFC_ENDDT_d).sum()), '/ > :', int((dff.ASGN_CANCL_DT_d > dff.CRTFC_ENDDT_d).sum()))
P('  ASGN_CANCL_DT 연도·월:', dff.ASGN_CANCL_DT_d.dt.to_period('M').value_counts().sort_index().to_dict())
P('  ASGN_CANCL_DT < 받은 날:', int((dff.ASGN_CANCL_DT_d < TODAY).sum()))
P('  영업상태:', dff.CLSBIZ_DVS_CD_NM.value_counts().to_dict(), '/ 반납일 있음:', int(dff.CRTFC_RETN_DT_d.notna().sum()))
P('  (ASGN_CANCL_DT − HACCP_APPN_DT) 일 요약:', (dff.ASGN_CANCL_DT_d - dff.HACCP_APPN_DT_d).dt.days.describe().round(0).to_dict())
P('  같은 번호의 다른 행에서도 ASGN_CANCL_DT 가 다른지: 번호', dff.no.nunique(), '개, 그 번호의 전체 행', int(df.no.isin(dff.no).sum()))
P('  반납일 있는 행에서 ASGN_CANCL_DT = 반납일:', int((dff.ASGN_CANCL_DT == dff.CRTFC_RETN_DT).sum()))
P('  15033311(지금 유효한 인증)에 번호가 있는 비율:', pct(int(dff.no.isin(set(c3.no)).sum()), len(dff)))
P('  ASGN_CANCL_DT < 받은 날 인 행 중 15033311 에 번호 있음:', pct(int(dff[dff.ASGN_CANCL_DT_d < TODAY].no.isin(set(c3.no)).sum()), int((dff.ASGN_CANCL_DT_d < TODAY).sum())))
P('  예:'); P(dff[['LCNS_NO', 'no', 'PRDLST_NM', 'HACCP_APPN_DT', 'CRTFC_ENDDT', 'ASGN_CANCL_DT', 'CRTFC_RETN_DT', 'CLSBIZ_DT', 'CLSBIZ_DVS_CD_NM']].head(12).to_string())

P('\n## H. 15058303 에만 있는 지정번호')
m3['fmt'] = m3.no.map(lambda s: 'YYYY-N-NNNN' if re.fullmatch(r'\d{4}-\d-\d{4}', s) else ('빈 값' if not s else '그 밖'))
sI = set(df.no)
m3['inI'] = m3.no.isin(sI)
P('15058303 번호 꼴별 I0580 에 있는 비율(행):'); P(m3.groupby('fmt').inI.agg(['size', 'mean']).round(4).to_string())
only = m3[~m3.inI & (m3.no != '')]
P('15058303 에만 있는 행', len(only), ': 그 LCNS_NO 가 I0580 에 있음', pct(int(only.LCNS_NO.isin(set(df.LCNS_NO)).sum()), len(only)))
P('  꼴별·지정일 연도:'); P(pd.crosstab(only.d.dt.year, only.fmt).to_string())
yy = only[only.fmt == 'YYYY-N-NNNN']
P('  YYYY-N-NNNN 꼴인데 I0580 에 없는 행', len(yy), '/ 15033311 에 있음', int(yy.no.isin(set(c3.no)).sum()))
P('  YYYY-N-NNNN 꼴, I0580 에 없는 번호의 LCNS_NO 가 I0580 에 있고, 같은 LCNS_NO 의 I0580 행이 같은 품목명:',
  int(yy.merge(df[['LCNS_NO', 'PRDLST_NM']].drop_duplicates(), on=['LCNS_NO', 'PRDLST_NM']).no.nunique()), '번호')
# 15058303 의 YYYY-N-NNNN 꼴 번호 중 2017~2020 지정인데 I0580 에 없는 것
P('  15058303 지정일 2017~2020, YYYY-N-NNNN 꼴, I0580 에 없음:', int(((yy.d.dt.year >= 2017)).sum()), '행')

P('\n## I. 지정번호 단위로 줄였을 때 (참고: 한 지정번호의 여러 행)')
multi = df[df.no.duplicated(keep=False)].sort_values(['no', 'SITE_ADDR'])
P('2행 이상인 지정번호의 행', len(multi))
k = multi.groupby('no').agg(addr=('SITE_ADDR', 'nunique'), prd=('PRDLST_NM', 'nunique'), st=('CLSBIZ_DVS_CD_NM', 'nunique'),
                            ap=('HACCP_APPN_DT', 'nunique'), en=('CRTFC_ENDDT', 'nunique'), rt=('CRTFC_RETN_DT', 'nunique'))
P('  번호 안에서 서로 다른 값이 있는 번호 수 — SITE_ADDR:', int((k.addr > 1).sum()), '/ PRDLST_NM:', int((k.prd > 1).sum()), '/ 영업상태:', int((k.st > 1).sum()),
  '/ HACCP_APPN_DT:', int((k.ap > 1).sum()), '/ CRTFC_ENDDT:', int((k.en > 1).sum()), '/ CRTFC_RETN_DT:', int((k.rt > 1).sum()), '/ 번호 수', len(k))
P('  예(번호 3개):')
for no in k[(k.addr > 1) & (k.st > 1)].index[:3]:
    P(df[df.no == no][['LCNS_NO', 'no', 'PRDLST_NM', 'SITE_ADDR', 'HACCP_APPN_DT', 'CRTFC_ENDDT', 'CRTFC_RETN_DT', 'CLSBIZ_DT', 'CLSBIZ_DVS_CD_NM']].to_string())
for no in k[(k.ap > 1)].index[:2]:
    P(df[df.no == no][['LCNS_NO', 'no', 'PRDLST_NM', 'SITE_ADDR', 'HACCP_APPN_DT', 'CRTFC_ENDDT', 'CRTFC_RETN_DT', 'CLSBIZ_DT', 'CLSBIZ_DVS_CD_NM']].to_string())

open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('저장:', OUT)
