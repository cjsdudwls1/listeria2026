# I0580 점검 3: 옛 꼴 번호(15058303)와 번호 바뀜, 폐업일 뒤 연장된 지정, 칼럼 조합별 '연도 말 지정 유효' 참고 수.
# 참고 수는 보고용(사건표에는 넣지 않음). 출력: 중간\외부지표\_haccp_I0580_점검3.txt
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from i0580_load import load
import haccp_load

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
MID = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\중간\외부지표'
OUT = os.path.join(MID, '_haccp_I0580_점검3.txt')
TODAY = pd.Timestamp('2026-10-19')
L = []
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); L.append(s)
def pct(a, b):
    return f'{a:,}/{b:,} ({(a / b * 100 if b else 0):.2f}%)'
def D(s):
    return pd.to_datetime(s.where(s.fillna('').str.strip() != ''), format='%Y%m%d', errors='coerce')

df, _, _ = load()
df['no'] = df.HACCP_APPN_NO.str.strip()
for c, n in [('HACCP_APPN_DT', 'appn'), ('CRTFC_ENDDT', 'end'), ('ASGN_CANCL_DT', 'cancl'), ('CRTFC_RETN_DT', 'retn'), ('CLSBIZ_DT', 'clsbiz')]:
    df[n] = D(df[c])
df['noyr'] = pd.to_numeric(df.no.str[:4], errors='coerce')

m3, _, _ = haccp_load.load('mfds')
m3['no'] = m3.HACCP_APPN_NO.fillna('').str.strip(); m3['d'] = D(m3.HACCP_APPN_DT)
c3, _, _ = haccp_load.load('cert')
c3['no'] = c3.appointno.fillna('').str.strip(); c3['iss'] = pd.to_datetime(c3.issuedate, format='%Y-%m-%d', errors='coerce')

P('## 1. 15058303 옛 꼴 번호(YYYY-N-NNNN 아님)와 I0580')
old = m3[~m3.no.str.fullmatch(r'\d{4}-\d-\d{4}') & (m3.no != '')]
P('옛 꼴 행', len(old), '/ 고유 번호', old.no.nunique(), '/ 꼴 예:', old.no.value_counts().head(8).index.tolist())
x = old.merge(df[['LCNS_NO', 'PRDLST_NM', 'no', 'noyr', 'appn']].rename(columns={'no': 'no_I'}), on=['LCNS_NO', 'PRDLST_NM'])
P('옛 꼴 행 중 같은 LCNS_NO·PRDLST_NM 이 I0580 에 있는 행:', pct(int(old.set_index(['LCNS_NO', 'PRDLST_NM']).index.isin(df.set_index(['LCNS_NO', 'PRDLST_NM']).index).sum()), len(old)))
if len(x):
    x1 = x.sort_values('noyr').drop_duplicates(['no', 'LCNS_NO', 'PRDLST_NM'])
    P('  그때 I0580 번호 앞 4자리 − 15058303 지정일 연도 분포:', (x1.noyr - x1.d.dt.year).value_counts().sort_index().head(20).to_dict())
P('옛 꼴 행의 PRDLST_NM "-"·빈 값:', int(old.PRDLST_NM.fillna('').str.strip().isin(['', '-']).sum()))
P('옛 꼴 행 중 같은 LCNS_NO 가 I0580 에 있음:', pct(int(old.LCNS_NO.isin(set(df.LCNS_NO)).sum()), len(old)))
# 같은 LCNS_NO 의 옛 꼴 행 지정일 → 같은 LCNS_NO 의 I0580 가장 이른 번호 연도
fy = df.groupby('LCNS_NO').noyr.min()
o2 = old.assign(fy=old.LCNS_NO.map(fy)).dropna(subset=['fy'])
P('  같은 LCNS_NO 의 I0580 가장 이른 번호 연도 − 옛 꼴 지정일 연도 분포:', (o2.fy - o2.d.dt.year).value_counts().sort_index().head(25).to_dict())

P('\n## 2. 폐업일자와 지정')
cl = df[df.clsbiz.notna()]
P('폐업일자 있는 행', len(cl), '/ 지정일(HACCP_APPN_DT)이 폐업일보다 늦은 행', int((cl.appn > cl.clsbiz).sum()))
P('  그 늦은 행의 영업상태:', cl[cl.appn > cl.clsbiz].CLSBIZ_DVS_CD_NM.value_counts().to_dict())
g = df.groupby('no').agg(allcl=('clsbiz', lambda s: s.notna().all()), anyalive=('CLSBIZ_DVS_CD_NM', lambda s: (s == '정상').any()))
late_no = set(cl[cl.appn > cl.clsbiz].no)
P('  늦은 행이 있는 번호', len(late_no), '/ 그중 같은 번호에 영업상태 정상 행이 있는 번호', int(g.loc[list(late_no)].anyalive.sum()),
  '/ 모든 행이 폐업인 번호', int(g.loc[list(late_no)].allcl.sum()))
P('모든 행이 폐업(폐업일자 있음)인 번호', int(g.allcl.sum()), '/ 번호', len(g))

P('\n## 3. 칼럼 조합별 연도 말(12-31) 지정 유효 수 (참고)')
# 행 단위 끝
cancl_real = df.cancl.where(df.cancl != df.end)  # ASGN_CANCL_DT 가 CRTFC_ENDDT 와 다를 때만 취소일로 봄
df['end_E1'] = pd.concat([df.end, df.retn, cancl_real], axis=1).min(axis=1)
cl_use = df.clsbiz.where(df.CLSBIZ_DVS_CD_NM != '폐업(관할변경)')
df['end_E2'] = pd.concat([df.end, df.retn, cancl_real, cl_use], axis=1).min(axis=1)
# 번호 단위: 시작·끝
num = df.groupby('no').agg(LCNS_NO=('LCNS_NO', 'first'), induty=('INDUTY_CD_NM', 'first'), noyr=('noyr', 'first'),
                           appn_min=('appn', 'min'), E1=('end_E1', 'max'), E2=('end_E2', 'max'))
f58 = m3[m3.no != ''].groupby('no').d.min()
f33 = c3[c3.no != ''].groupby('no').iss.min()
num['d58'] = num.index.map(f58); num['d33'] = num.index.map(f33)
num['S_noyr'] = pd.to_datetime(num.noyr.astype('Int64').astype(str) + '-01-01', errors='coerce')
num['S_first'] = pd.concat([num.d58, num.d33, num.appn_min], axis=1).min(axis=1)
num['S_appn'] = num.appn_min
P('번호 수', len(num), '/ 15058303 날짜 있음', int(num.d58.notna().sum()), '/ 15033311 날짜 있음', int(num.d33.notna().sum()),
  '/ 둘 다 없음', int((num.d58.isna() & num.d33.isna()).sum()))
P('S_first 가 15058303·15033311 에서 온 번호:', int((num.S_first < num.appn_min).sum()), '/ I0580 지정일 그대로:', int((num.S_first == num.appn_min).sum()))
P('번호 앞 4자리 연도 − S_first 연도 분포:', (num.noyr - num.S_first.dt.year).value_counts().sort_index().head(15).to_dict())
P('번호 끝(E1) 연도 분포(받은 날 이전만):', num[num.E1 < TODAY].E1.dt.year.value_counts().sort_index().to_dict())
P('E1 이 2017-08-03·04 인 번호:', int(num.E1.isin([pd.Timestamp('2017-08-03'), pd.Timestamp('2017-08-04')]).sum()))

years = list(range(2008, 2026))
rows = []
for y in years + ['2026-10-19']:
    T = pd.Timestamp(f'{y}-12-31') if isinstance(y, int) else pd.Timestamp(y)
    r = {'기준일': T.date()}
    for s in ['S_noyr', 'S_first', 'S_appn']:
        for e in ['E1', 'E2']:
            v = num[(num[s] <= T) & (num[e] >= T)]
            r[f'{s}+{e} 번호'] = len(v)
            r[f'{s}+{e} 업소'] = v.LCNS_NO.nunique()
    v = num[(num.S_first <= T) & (num.E1 >= T)]
    r['S_first+E1 식품제조가공업 업소'] = v[v.induty == '식품제조가공업'].LCNS_NO.nunique()
    # 끝 정보 없이(생존자만): 지금 유효(E1 ≥ 받은 날)인 번호를 시작일만으로 셈
    alive = num[num.E1 >= TODAY]
    r['지금 유효한 번호만, S_first 기준'] = int((alive.S_first <= T).sum())
    rows.append(r)
ref = pd.DataFrame(rows)
P(ref[['기준일', 'S_first+E1 번호', 'S_first+E1 업소', 'S_first+E2 번호', 'S_first+E2 업소', 'S_noyr+E1 번호', 'S_noyr+E1 업소',
       'S_appn+E1 번호', 'S_appn+E1 업소', 'S_first+E1 식품제조가공업 업소', '지금 유효한 번호만, S_first 기준']].to_string(index=False))
ref.to_csv(os.path.join(MID, '_haccp_I0580_연도말_유효수_참고.csv'), index=False, encoding='utf-8-sig')

P('\n## 4. 15058303(2021-01 무렵 목록)과 비교: 2020-12-31 기준')
T = pd.Timestamp('2020-12-31')
v = num[(num.S_first <= T) & (num.E1 >= T)]
P('I0580 S_first+E1 로 2020-12-31 유효 번호', len(v), '/ 업소', v.LCNS_NO.nunique())
P('15058303 번호(고유)', m3[m3.no != ''].no.nunique(), '/ LCNS_NO', m3.LCNS_NO.nunique(), '/ 그중 YYYY-N-NNNN 꼴 번호', m3[m3.no.str.fullmatch(r'\d{4}-\d-\d{4}')].no.nunique())
s58 = set(m3[m3.no.str.fullmatch(r'\d{4}-\d-\d{4}')].no)
P('2020-12-31 유효(I0580) 번호 중 15058303 에 있음:', pct(len(set(v.index) & s58), len(v)))
P('15058303 YYYY-N-NNNN 번호 중 I0580 에서 2020-12-31 유효:', pct(len(set(v.index) & s58), len(s58)))
nn = num[num.index.isin(s58)]
P('15058303 YYYY-N-NNNN 번호이면서 I0580 에 있는데 2020-12-31 유효 아님:', int((~((nn.S_first <= T) & (nn.E1 >= T))).sum()),
  '— 그 E1 분포(연도):', nn[~((nn.S_first <= T) & (nn.E1 >= T))].E1.dt.year.value_counts().sort_index().to_dict())

open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('저장:', OUT)
