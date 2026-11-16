import re
P = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'
t = open(P, encoding='utf-8').read().replace('\r\n', '\n')
lines = t.split('\n')
# 확정된 결정 번호
sec = t[t.index('### 12-1'):t.index('\n## 13')]
m = re.search(r'권고대로 확정: ([0-9·\-a-z~]+)', sec)
conf = set()
for tok in m.group(1).split('·'):
    if '~' in tok:
        a, b = tok.split('~'); base = a.split('-')[0]
        for k in range(int(a.split('-')[1]), int(b.split('-')[1]) + 1):
            conf.add(f'{base}-{k}')
    else:
        conf.add(tok)
PIPE = re.compile(r'(?<!\\)\|')
for l in sec.split('\n'):
    if re.match(r'^\| \d', l):
        c = [x.strip() for x in PIPE.split(l)[1:-1]]
        if '확정' in c[3]:
            conf.add(c[0])
out = []
for i, l in enumerate(lines, 1):
    if l.startswith('| ') and re.match(r'^\| \d', l) and i > t[:t.index('### 12-1')].count('\n'):
        continue  # 결정 표 행 자체는 제외
    for mm in re.finditer(r'\[권고([^\]]*)\]', l):
        inner = mm.group(1)
        d = re.search(r'결정 (\d+(?:-[a-z0-9]+)?)', inner)
        dn = d.group(1) if d else None
        if dn and dn in conf:
            continue
        s = max(0, mm.start() - 110)
        out.append(f'{i}\t결정 {dn}\t{l[s:mm.end()]}')
open(r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\gwongo_scan.txt', 'w', encoding='utf-8').write('\n'.join(out))
print(len(out))
