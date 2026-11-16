# 친환경 시도별통계: 같은 해 '시' 행과 '시 ○○구' 행의 관계 확인(읽기만).
# 사용: python -I eco_city_check.py <2025파일 csv> <다른 파일 csv(20241231)> <출력txt>
import sys
import pandas as pd

p25, p24, out = sys.argv[1], sys.argv[2], sys.argv[3]


def read(p):
    for e in ('utf-8-sig', 'cp949'):
        try:
            d = pd.read_csv(p, encoding=e, dtype=str)
            d.columns = ['연도', '시도', '시군구', '구분', '계', '유기', '무농약']
            return d
        except UnicodeDecodeError:
            continue


d = read(p25)
for c in ['계', '유기', '무농약']:
    d[c + '_n'] = pd.to_numeric(d[c], errors='coerce')
d['구분'] = d['구분'].str.replace('ton', '톤')
L = []
P = L.append
P('비숫자 값: ' + str({c: d.loc[d[c + '_n'].isna() & d[c].notna(), c].unique().tolist() for c in ['계', '유기', '무농약']}))
P('빈 값 수: ' + str({c: int(d[c].isna().sum()) for c in ['계', '유기', '무농약']}))
# 계 = 유기 + 무농약 ?
dd = (d['계_n'] - d['유기_n'] - d['무농약_n']).abs()
P(f'계 != 유기+무농약 (차 >1e-6) 행 수: {(dd > 1e-6).sum()}')
P(d.loc[dd > 1e-6, ['연도', '시도', '시군구', '구분', '계', '유기', '무농약']].head(30).to_string())

cities = sorted({s.split(' ')[0] for s in d['시군구'].dropna() if ' ' in s})
P('구 행이 있는 시: ' + ', '.join(cities))
rows = []
for city in cities:
    sub = d[d['시군구'].fillna('').str.startswith(city)]
    for (yr, gb), g in sub.groupby(['연도', '구분']):
        crow = g[g['시군구'] == city]
        grows = g[g['시군구'] != city]
        r = {'시': city, '연도': yr, '구분': gb, '시행': len(crow), '구행수': len(grows),
             '구행들': ';'.join(grows['시군구'].str.replace(city + ' ', '').tolist())}
        for c in ['계', '유기', '무농약']:
            r['시_' + c] = crow[c + '_n'].sum() if len(crow) else None
            r['구합_' + c] = grows[c + '_n'].sum() if len(grows) else None
        rows.append(r)
t = pd.DataFrame(rows)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 60)
P('\n== 면적(ha) 줄: 시 행 vs 구 행 합')
P(t[t['구분'].str.startswith('면적')].to_string())
P('\n== 건수(건) 줄: 시 행 vs 구 행 합')
P(t[t['구분'].str.startswith('건수')].to_string())

# 출장소 행
P('\n== 출장소로 보이는 행(면적·건수)')
m = d['시군구'].fillna('').str.contains('출')
P(d.loc[m & d['구분'].str.match('면적|건수'), ['연도', '시도', '시군구', '구분', '계', '유기', '무농약']].to_string())

# 20241231 과 20251231 차이 전체
e = read(p24); e['구분'] = e['구분'].str.replace('ton', '톤')
keys = ['연도', '시도', '시군구', '구분']
m2 = e.merge(d[keys + ['계', '유기', '무농약']], on=keys, how='outer', indicator=True, suffixes=('_24', '_25'))
m2 = m2[m2['연도'].astype(int) <= 2024]
P('\n== 20241231 vs 20251231 (2016~2024) 한쪽에만 있는 행 전체')
P(m2.loc[m2['_merge'] != 'both', keys + ['계_24', '계_25', '_merge']].to_string())

# 시군구 이름 빈칸 행
P('\n== 시군구 빈칸 행: ' + str(d['시군구'].isna().sum()))
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('ok')
