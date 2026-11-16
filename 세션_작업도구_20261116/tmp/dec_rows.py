import re
P = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'
t = open(P, encoding='utf-8').read().replace('\r\n', '\n')
sec = t[t.index('### 12-1'):t.index('\n## 13')]
out = []
for l in sec.split('\n'):
    m = re.match(r'^\| (1|4|8|20-a|21|27|41|52|55|55-1) \|', l)
    if m:
        c = [x.strip() for x in l.split(' | ')]
        out.append(f'[{m.group(1)}] {c[1][:40]} || 권고: {c[3][:220]} || 묻는 곳: {c[5] if len(c) > 5 else ""}')
open(r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\dec_rows.txt', 'w', encoding='utf-8').write('\n'.join(out))
