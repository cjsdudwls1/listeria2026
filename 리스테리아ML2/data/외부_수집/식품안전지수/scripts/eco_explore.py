# 친환경 시도별통계(농산물) 파일 4개 탐색(읽기만).
# 사용: python -I eco_explore.py <원자료폴더> <출력txt>
import sys, os, glob
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(src, '친환경_시도별통계_농산물_*.csv')))


def read(p):
    for e in ('utf-8-sig', 'cp949'):
        try:
            d = pd.read_csv(p, encoding=e, dtype=str)
            return d, e
        except UnicodeDecodeError:
            continue
    raise RuntimeError(p)


lines = []
P = lines.append
frames = {}
for p in files:
    d, e = read(p)
    d.columns = ['연도', '시도', '시군구', '구분', '계', '유기', '무농약']
    frames[os.path.basename(p)] = d
    P(f'== {os.path.basename(p)} enc={e} rows={len(d)}')
    P('  구분 값: ' + str(d['구분'].value_counts().to_dict()))
    P('  연도별 행: ' + str(d['연도'].value_counts().sort_index().to_dict()))

last = frames[os.path.basename(files[-1])]
P('\n== 2025 파일: 시도 값')
P(str(last['시도'].value_counts().to_dict()))
# 시군구 이름 전체 목록(시도별)
P('\n== 시도·시군구 조합별 등장 연도 (2025 파일)')
a = last[last['구분'].str.startswith('면적')]
g = a.groupby(['시도', '시군구'])['연도'].apply(lambda s: ','.join(sorted(s.unique())))
for (sd, sg), yrs in g.items():
    P(f'  {sd} | {sg} | {yrs}')

# 같은 해, 같은 시도·시군구·구분 중복 행
dup = last[last.duplicated(['연도', '시도', '시군구', '구분'], keep=False)]
P(f'\n== 2025 파일 중복 키 행 수: {len(dup)}')
P(dup.head(40).to_string())

# 파일 간 같은 해 값 비교
P('\n== 파일 간 같은 해 값 비교(면적 줄)')
keys = ['연도', '시도', '시군구', '구분']
names = list(frames)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        x = frames[names[i]]; y = frames[names[j]]
        x = x.assign(구분=x['구분'].str.replace('ton', '톤'))
        y = y.assign(구분=y['구분'].str.replace('ton', '톤'))
        yrs = sorted(set(x['연도']) & set(y['연도']))
        m = x[x['연도'].isin(yrs)].merge(y[y['연도'].isin(yrs)], on=keys, how='outer', suffixes=('_a', '_b'), indicator=True)
        only_a = (m['_merge'] == 'left_only').sum(); only_b = (m['_merge'] == 'right_only').sum()
        both = m[m['_merge'] == 'both']
        diff = 0
        dl = []
        for c in ['계', '유기', '무농약']:
            va = pd.to_numeric(both[c + '_a'], errors='coerce'); vb = pd.to_numeric(both[c + '_b'], errors='coerce')
            dd = ~((va - vb).abs() < 1e-9) & ~(va.isna() & vb.isna())
            diff += int(dd.sum())
            if dd.any():
                dl.append(both.loc[dd, keys + [c + '_a', c + '_b']].head(10).to_string())
        P(f'  {names[i]} vs {names[j]} 겹친 연도 {yrs[0]}~{yrs[-1]}: 한쪽만 {only_a}/{only_b}, 값 다른 칸 {diff}')
        for t in dl:
            P(t)
        if only_a or only_b:
            P(m.loc[m['_merge'] != 'both', keys + ['_merge']].head(20).to_string())

open(out, 'w', encoding='utf-8').write('\n'.join(lines))
print('written', out)
