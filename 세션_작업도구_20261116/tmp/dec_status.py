import re
P = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'
t = open(P, encoding='utf-8').read().replace('\r\n', '\n')
s = t.index('### 12-1')
e = t.index('\n## 13')
sec = t[s:e]
m = re.search(r'권고대로 확정: ([0-9·\-a-z~]+)', sec)
bulk = set()
if m:
    for tok in m.group(1).split('·'):
        if '~' in tok:  # 55-2~55-15
            a, b = tok.split('~')
            base = a.split('-')[0]
            for k in range(int(a.split('-')[1]), int(b.split('-')[1]) + 1):
                bulk.add(f'{base}-{k}')
        else:
            bulk.add(tok)
PIPE = re.compile(r'(?<!\\)\|')
conf, prof, ext, pend = [], [], [], []
for l in sec.split('\n'):
    if not re.match(r'^\| \d', l):
        continue
    c = [x.strip() for x in PIPE.split(l)[1:-1]]
    no, rec, who = c[0], c[3], (c[5] if len(c) > 5 else '')
    if '교수님 확인' in who or ('교수님 확인' in rec and '확정' not in rec):
        prof.append(no)
    elif '식정원' in who:
        ext.append(no)
    elif '확정' in rec or no in bulk:
        conf.append(no)
    else:
        pend.append((no, c[1][:30], rec[:80]))
print('rows', len(conf) + len(prof) + len(ext) + len(pend), 'confirmed', len(conf))
print('professor', prof)
print('external', ext)
print('pending', len(pend))
for p in pend:
    print(' ', p)
