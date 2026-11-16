import sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
cur = None
bad = 0
for i, l in enumerate(lines, 1):
    if l.startswith('|'):
        n = l.count('|')
        if cur is None:
            cur = n
        elif n != cur:
            print('mismatch line', i, n, cur); bad += 1
    else:
        cur = None
print('bad', bad)
s = [i for i, l in enumerate(lines, 1) if l.startswith('## ')]
print('section starts', s)
