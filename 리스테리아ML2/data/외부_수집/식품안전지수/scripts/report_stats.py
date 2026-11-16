# 연도표 작성 보고용 수치 세기(읽기만). 결과를 txt 로 저장.
# 사용: python -I report_stats.py <친환경 parquet> <지하수 parquet> <시군구_대표좌표.parquet> <친환경 원자료 csv(20251231)> <출력 txt>
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding='utf-8')
pe, pg, ps, praw, out = sys.argv[1:6]
L = []
P = L.append
s = pd.read_parquet(ps)[['sgg_code', 'sgg_nm']]
e = pd.read_parquet(pe).merge(s, on='sgg_code')
g = pd.read_parquet(pg).merge(s, on='sgg_code')
P('## 스키마')
P('친환경: ' + ', '.join(f'{f.name}:{f.type}' for f in pq.read_schema(pe)))
P('지하수: ' + ', '.join(f'{f.name}:{f.type}' for f in pq.read_schema(pg)))

P('\n## 친환경')
P(f'행 {len(e)}, 시군구 {e.sgg_code.nunique()}, 연도 {e.자료연도.min()}~{e.자료연도.max()}')
t = e.groupby('자료연도').agg(값있음=('친환경_비율', lambda x: int(x.notna().sum())), 빈칸=('친환경_비율', lambda x: int(x.isna().sum())),
                           영채움=('0채움_여부', 'sum'), 단위전체영채움=('계산단위_전체_0채움', 'sum'),
                           결정대기=('결정대기_사유', lambda x: int(x.notna().sum())))
P(t.to_string())
P('빈칸 시군구: ' + '; '.join(f'{n}({",".join(map(str, v))})' for n, v in e[e.친환경_비율.isna()].groupby('sgg_nm')['자료연도'].apply(list).items()))
P('0채움 시군구별: ' + '; '.join(f'{n}({",".join(map(str, v))})' for n, v in e[e['0채움_여부']].groupby('sgg_nm')['자료연도'].apply(list).items()))
P('분모 단위별 시군구 수: ' + str(e.drop_duplicates('sgg_code')['비율_계산단위'].value_counts().to_dict()))
P('사용가능일 근거별 행 수: ' + str(e.groupby(['사용가능일_근거']).size().to_dict()))
P('연도별 사용가능일: ' + str(e.drop_duplicates('자료연도').set_index('자료연도')['사용가능일'].astype(str).to_dict()))
P('결정대기 시군구: ' + '; '.join(f'{n}({",".join(map(str, v))})' for n, v in e[e.결정대기_사유.notna()].groupby('sgg_nm')['자료연도'].apply(list).items()))
P('비율 분포: ' + e['친환경_비율'].describe().round(4).to_dict().__str__())
P('비율 0 인 행: ' + str(int((e['친환경_비율'] == 0).sum())))
P('비율 > 0.3: ' + '; '.join(f"{r.sgg_nm} {r.자료연도} {r.친환경_비율:.3f}" for r in e[e['친환경_비율'] > 0.3].itertuples()))
# 원자료 계 vs 유기+무농약
r = pd.read_csv(praw, encoding='cp949', dtype=str)
r.columns = ['연도', '시도', '시군구', '구분', '계', '유기', '무농약']
r = r[r['구분'].str.startswith('면적')]
for c in ['계', '유기', '무농약']:
    r[c] = pd.to_numeric(r[c])
dd = (r['계'] - r['유기'] - r['무농약']).abs()
P(f'원자료 면적 줄 {len(r)}개 중 계 ≠ 유기+무농약 {int((dd > 1e-6).sum())}개, 최대 차 {dd.max():.4f}ha')
# 단위 합의 계 vs 유기+무농약
d2 = (e['인증면적_합'] - e['유기_인증면적'] - e['무농약_인증면적']).abs()
P(f'결과 행 중 인증면적_합 ≠ 유기+무농약 {int((d2 > 1e-6).sum())}행, 최대 차 {d2.max():.4f}ha')

P('\n## 지하수')
P(f'행 {len(g)}, 시군구 {g.sgg_code.nunique()}, 연도 {g.자료연도.min()}~{g.자료연도.max()}')
t = g.groupby('자료연도').agg(판=('연보_판', 'first'), 값있음=('검사_건수', lambda x: int(x.notna().sum())),
                           빈칸=('검사_건수', lambda x: int(x.isna().sum())), 줄없음=('연보_줄없음', 'sum'),
                           적합률빈칸=('적합률', lambda x: int(x.isna().sum())),
                           검사0건=('검사_건수', lambda x: int((x == 0).sum())),
                           결정대기=('결정대기_사유', lambda x: int(x.notna().sum())))
P(t.to_string())
P('빈칸 시군구: ' + '; '.join(f'{n}({",".join(map(str, v))})' for n, v in g[g.검사_건수.isna()].groupby('sgg_nm')['자료연도'].apply(list).items()))
P('검사 0건 시군구: ' + '; '.join(f'{n}({",".join(map(str, v))})' for n, v in g[g.검사_건수 == 0].groupby('sgg_nm')['자료연도'].apply(list).items()))
P('값 단위별 행 수: ' + str(g['값_단위'].value_counts(dropna=False).to_dict()))
P('값 단위별 시군구 수(2024): ' + str(g[g.자료연도 == 2024]['값_단위'].value_counts(dropna=False).to_dict()))
P('사용가능일 근거별 행 수: ' + str(g.groupby('사용가능일_근거').size().to_dict()))
P('연도별 사용가능일: ' + str(g.drop_duplicates('자료연도').set_index('자료연도')['사용가능일'].astype(str).to_dict()))
m = g[g['미기재_건수'] > 0]
P(f'미기재 > 0 인 행(2013~2015 자료): {len(m)}행, 시군구 {m.sgg_code.nunique()}개, 미기재 합 {m.drop_duplicates(["원자료_시군구_이름","자료연도"])["미기재_건수"].sum():g}')
P('결정대기 시군구: ' + '; '.join(f'{n}({",".join(map(str, v))})' for n, v in g[g.결정대기_사유.notna()].groupby('sgg_nm')['자료연도'].apply(list).items()))
P('적합률 분포: ' + g['적합률'].describe().round(4).to_dict().__str__())
# 인쇄 적합률과 계산 적합률 비교(2016~2021 자료는 정수 %, 2022~ 소수 첫째)
x = g[g['연보_적합률_인쇄값'].notna() & g['적합률'].notna()].copy()
x['계산%'] = x['적합률'] * 100
x['차'] = (x['계산%'] - x['연보_적합률_인쇄값']).abs()
P(f"인쇄 적합률과 계산 적합률 차 최대 {x['차'].max():.3f}%p (인쇄값 반올림 자리: 2016~2021 자료 정수, 2022~ 소수 첫째)")
bad = x[x['차'] > 0.5]
P('인쇄 적합률 차 > 0.5%p 행: ' + '; '.join(f"{n} {y} 인쇄 {pv} 계산 {cv:.2f}" for n, y, pv, cv in zip(bad['sgg_nm'], bad['자료연도'], bad['연보_적합률_인쇄값'], bad['계산%'])))
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
