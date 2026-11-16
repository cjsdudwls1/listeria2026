import re
from collections import Counter

P = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'
B = r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\plan_before_concise.md'
t = open(P, 'rb').read().decode('utf-8').replace('\r\n', '\n')
lines = t.split('\n')
idx = [i for i, l in enumerate(lines) if l.startswith('  - 낙농장(L1): 국내는 받은 축사 엑셀로')]
assert len(idx) == 1
lines.insert(idx[0] + 1, '  - 계획서 전체 말투를 간결체로 다시 씀(사용자 "계획서 말투 간결체로 수정" — 내용·인용·숫자는 그대로)')
t = '\n'.join(lines)
open(P, 'wb').write(t.replace('\n', '\r\n').encode('utf-8'))

a = open(B, encoding='utf-8').read().replace('\r\n', '\n')
PAT = {'dq': r'"([^"\n]*)"', 'bt': r'`([^`\n]+)`', 'num': r'\d+(?:[.,]\d+)*', 'dec': r'결정 \d+(?:-[a-z0-9]+)?', 'line': r'\d+(?:~\d+)?줄'}
for k, p in PAT.items():
    ca, cb = Counter(re.findall(p, a)), Counter(re.findall(p, t))
    m, e = ca - cb, cb - ca
    print(k, 'missing', sum(m.values()), list(m)[:5], 'extra', sum(e.values()), list(e)[:5])
print('lines', len(a.split('\n')), '->', len(lines), 'chars', len(a), '->', len(t))
