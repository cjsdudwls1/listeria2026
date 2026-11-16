import re
P = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'
L = open(P, encoding='utf-8').read().replace('\r\n', '\n').split('\n')
out = []
for key in ('[권고', '지시 뒤', '대기'):
    out.append(f'##### {key}')
    for i, l in enumerate(L):
        if key in l and not l.startswith('- 11-0') and not l.startswith('  - '+'11-0'):
            j = l.find(key)
            s = max(0, j - 170); e = min(len(l), j + 110)
            out.append(f'{i+1}: …{l[s:e]}…')
open(r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\open_items.txt', 'w', encoding='utf-8').write('\n'.join(out))
