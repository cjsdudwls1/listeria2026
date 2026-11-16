import re, sys
from collections import Counter

D = r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\concise'
PAT = {
    'dq': re.compile(r'"([^"\n]*)"'),
    'cq': re.compile(r'“([^”\n]*)”'),
    'bt': re.compile(r'`([^`\n]+)`'),
    'num': re.compile(r'\d+(?:[.,]\d+)*'),
    'dec': re.compile(r'결정 \d+(?:-[a-z0-9]+)?'),
    'tag': re.compile(r'\[(?:확정|권고|대기)'),
    'line': re.compile(r'\d+(?:~\d+)?줄'),
    'sec': re.compile(r'§\d+(?:-\d+)?'),
}
PIPE = re.compile(r'(?<!\\)\|')
ks = [int(a) for a in sys.argv[1:]] or range(1, 9)
for k in ks:
    try:
        a = open(f'{D}\\in_{k}.md', encoding='utf-8').read().replace('\r\n', '\n')
        b = open(f'{D}\\out_{k}.md', encoding='utf-8').read().replace('\r\n', '\n')
    except FileNotFoundError:
        print(k, 'missing out');
        continue
    la, lb = a.split('\n'), b.split('\n')
    print(f'== chunk {k}: lines {len(la)} -> {len(lb)}, chars {len(a)} -> {len(b)} ({100*len(b)/len(a):.0f}%)')
    ha = [l for l in la if l.startswith('#')]
    hb = [l for l in lb if l.startswith('#')]
    if ha != hb:
        print('  HEADINGS DIFFER', [x for x in ha if x not in hb][:5], [x for x in hb if x not in ha][:5])
    ta = [l for l in la if l.startswith('|')]
    tb = [l for l in lb if l.startswith('|')]
    if len(ta) != len(tb):
        print('  TABLE ROWS', len(ta), '->', len(tb))
    else:
        bad = [i for i, (x, y) in enumerate(zip(ta, tb)) if len(PIPE.findall(x)) != len(PIPE.findall(y))]
        if bad:
            print('  PIPE COUNT DIFF rows', bad[:10])
    for name, p in PAT.items():
        ca, cb = Counter(p.findall(a)), Counter(p.findall(b))
        miss, extra = ca - cb, cb - ca
        if miss or extra:
            ms = list(miss.elements())
            es = list(extra.elements())
            print(f'  {name}: missing {len(ms)} {ms[:8]} | extra {len(es)} {es[:8]}')
