# -*- coding: utf-8 -*-
"""보고용 숫자 계산 (파일목록.csv 만 읽음)."""
import csv, collections, sys

G, M = 1024 ** 3, 1024 ** 2


def h(n):
    return f'{n / G:.1f} GB' if n >= G else (f'{n / M:.1f} MB' if n >= M else f'{n / 1024:.1f} KB')


P = 'C:\\Users\\NT751\\'
KEYS = ['Documents\\Amaranth10', 'Documents\\%I9F9\'T', 'Downloads', 'Pictures\\Screenshots', 'DataGripProjects',
        'Desktop\\천영진', '.ssh', '.ipython', '.claude', 'Desktop\\천영진\\리스테리아ML2\\exp',
        'Desktop\\천영진\\리스테리아ML2\\exp\\final_results', 'Desktop\\천영진\\리스테리아ML2\\data',
        'Desktop\\천영진\\리스테리아ML\\data', 'Desktop\\천영진\\문서', 'Desktop\\천영진\\toxin_2026',
        'Desktop\\천영진\\Listeria', 'Desktop\\천영진\\보관', 'Desktop\\천영진\\리스테리아ML2\\notebook',
        'Desktop\\천영진\\리스테리아ML', 'Desktop\\천영진\\리스테리아ML2', '.claude\\jobs', '.claude\\projects']
T = {k: [0, 0, 0, 0, 0] for k in KEYS + ['ROOT .claude.json*', 'ROOT files']}
tot = [0, 0]; new = [0, 0]; user = [0, 0]; usernew = [0, 0]; sens = 0
cat = collections.Counter(); loc = collections.Counter(); catloc = collections.defaultdict(collections.Counter)
with open(sys.argv[1], encoding='utf-8-sig') as f:
    r = csv.reader(f); next(r)
    for row in r:
        p, s = row[0], int(row[1]); n = row[6] == '이후'
        tot[0] += 1; tot[1] += s
        if n:
            new[0] += 1; new[1] += s
        if not row[8]:
            user[0] += 1; user[1] += s
            if n:
                usernew[0] += 1; usernew[1] += s
        rel = p[len(P):] if p.startswith(P) else None
        if row[7]:
            sens += 1
            parts = (rel or p).split('\\')
            if rel is None:
                g = p
            elif len(parts) == 1:
                g = '(사용자 폴더 바로 아래)'
            elif parts[0] in ('Desktop', '.claude', 'Documents', 'Pictures'):
                g = '\\'.join(parts[:min(3, len(parts) - 1)])
            else:
                g = parts[0]
            loc[g] += 1
            for c in row[7].split('; '):
                cat[c] += 1
                catloc[c][g] += 1
        if rel is None:
            continue
        hits = [k for k in KEYS if rel.startswith(k + '\\')]
        if '\\' not in rel:
            hits.append('ROOT files')
            if rel.startswith('.claude.json'):
                hits.append('ROOT .claude.json*')
        for k in hits:
            t = T[k]; t[0] += 1; t[1] += s
            if row[8]:
                t[2] += 1; t[3] += s
            if row[7]:
                t[4] += 1
print('total', tot[0], h(tot[1]), tot[1])
print('after0901', new[0], h(new[1]), new[1])
print('user(non-regen)', user[0], h(user[1]), 'after', usernew[0], h(usernew[1]))
print('sens files', sens)
for c, v in cat.items():
    print('CAT', v, c)
    for g, x in catloc[c].most_common(12):
        print('      ', x, g)
print('--- by location')
for g, v in loc.most_common(40):
    print('  LOC', v, g)
print('--- key folders: files size | regen files size | sens')
for k, t in T.items():
    print(f'{k}\t{t[0]}\t{h(t[1])}\t| regen {t[2]} {h(t[3])}\t| sens {t[4]}')
