import re, sys
sys.stdout.reconfigure(encoding='utf-8')
L = open(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v61.md", encoding='utf-8', newline='').read().split('\r\n')
a = [i for i, l in enumerate(L) if l.startswith('## 0. 고정 조건')][0]
b = [i for i, l in enumerate(L) if l.startswith('## 부록 A')][0]
print('body', b - a)
for pat in ['3-4', '11 구조', '11종', '11 판', '구조 11', 'LIMS 갈래', 'LIMS Dense', '딥러닝 끝 판', '끝 판 전부', '11 끝',
            '13,212', '127,007', '149,841', '15,356', '11,748', '125,543', '(1등)', '시계열 갈래', '결합 모델', '결합-']:
    for i, l in enumerate(L):
        for m in re.finditer(re.escape(pat), l):
            s = max(0, m.start() - 45)
            print(pat, '|', i + 1, '|', l[s:m.end() + 45])
# A-9 sum
st = [i for i, l in enumerate(L) if l.startswith('### A-9')][0]
tot = 0
for l in L[st:st + 30]:
    c = l.split('|')
    if len(c) > 3 and re.fullmatch(r'[\d,]+', c[2].strip()):
        tot += int(c[2].strip().replace(',', ''))
        print('row', c[1].strip(), c[2].strip())
print('fixed sum', tot, 'trees only', tot + 298, 'tree+DL3', tot + 298 + 111, '6', tot + 298 + 74, 'DL single', tot + 23)
print(13 * 1068, 1464 + 13 * 1068, 13 * 11413, 1464 + 13 * 11413, 15 * 1068, 1472 + 15 * 1068, 15 * 11413, 1472 + 15 * 11413)
print(24 + 20 + 8 + 45)
