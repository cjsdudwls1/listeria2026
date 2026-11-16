# 연보 4.2(수질측정망 자료 현황) 쪽에서 '지점번호처럼 보이는 줄' 개수를 세어 대략적인 행 수를 봄(읽기만, 추정치).
# 사용: py -3.14 -I count_wells.py <pdf> <시작쪽> <끝쪽>
import sys, re, fitz
sys.stdout.reconfigure(encoding="utf-8")
doc = fitz.open(sys.argv[1]); a, b = int(sys.argv[2]), int(sys.argv[3])
pat = re.compile(r'^[A-Za-z0-9가-힣]{1,4}(-[A-Za-z0-9]{1,4}){2,5}\s*$')
n = 0; ex = []
heads = {}
for i in range(a - 1, b):
    t = doc[i].get_text()
    for ln in t.splitlines():
        if pat.match(ln.strip()):
            n += 1
            if len(ex) < 5: ex.append(ln.strip())
    for h in re.findall(r'4\.2\.\d+\s*[^\n]{0,30}', t):
        heads.setdefault(h.strip(), i + 1)
print('지점번호 모양 줄', n, '예', ex)
for h, p in list(heads.items())[:40]:
    print('  절', h, 'p', p)
