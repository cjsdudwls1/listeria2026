# 두 자료 맞대기(점검용): 15033311 종료일 분포·인증일과 종료일 차이, 15058303 지정번호가 15033311 인증번호에 있는지.
import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from haccp_load import load
sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
m, _, _ = load('mfds'); c, _, _ = load('cert')
c['end'] = c.issueenddate; c['iss'] = c.issuedate
print('15033311 issueenddate 가장 이른 값:', c.end.min(), '/ 오늘 이전 값 행:', int((c.end < '2026-10-19').sum()), '/ 2500-12-31 행:', int((c.end == '2500-12-31').sum()))
print('  2500-12-31 행의 companykindNm:', c[c.end == '2500-12-31'].companykindNm.value_counts().to_dict())
ok = c[c.end != '2500-12-31']
dy = (pd.to_datetime(ok.end) - pd.to_datetime(ok.iss)).dt.days / 365.25
print('  (issueenddate - issuedate) 년 분포(사분위):', dy.describe().round(2).to_dict())
print('  issuedate 연도별 (issueenddate - issuedate) 중앙값(년):', ok.assign(y=ok.iss.str[:4], d=dy).groupby('y').d.median().round(1).to_dict())
print('  issueenddate 연-월 상위:', c.end.str[:7].value_counts().head(8).to_dict())
mn = m.HACCP_APPN_NO.str.strip(); cn = set(c.appointno.str.strip())
inn = mn.isin(cn)
print(f'15058303 지정번호가 15033311 인증번호에 있는 행: {int(inn.sum()):,} / {len(m):,} ({inn.mean()*100:.1f}%)')
print('  지정번호 꼴별 있는 비율:', m.assign(p=mn.str.replace(r"\d", "9", regex=True), i=inn).groupby('p').i.agg(['size', 'mean']).sort_values('size', ascending=False).head(6).round(3).to_dict('index'))
print('  지정일 연도별 있는 비율:', m.assign(y=m.HACCP_APPN_DT.str[:4], i=inn).groupby('y').i.mean().round(3).to_dict())
mm = m.assign(no=mn).merge(c.assign(no=c.appointno.str.strip())[['no', 'issuedate', 'businessitemNm', 'area1', 'area2']], on='no')
same = (mm.HACCP_APPN_DT == mm.issuedate.str.replace('-', ''))
print(f'  번호가 같은 행 {len(mm):,} 중 날짜(HACCP_APPN_DT = issuedate) 같은 행 {int(same.sum()):,} ({same.mean()*100:.1f}%)')
print('  예시:'); print(mm[~same][['no', 'HACCP_APPN_DT', 'issuedate', 'PRDLST_NM', 'businessitemNm']].head(8).to_string())
