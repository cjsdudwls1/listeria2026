# 저장한 파일에 인증키(FOODSAFETY_KEY·DATA_GO_KR_KEY 의 원래 꼴·URL 인코딩 꼴·앞 12글자)가 들어 있는지 검사. 키 값은 출력하지 않음.
# 실행: py -I i0580_keycheck.py <폴더 또는 파일> ...
import sys, os, urllib.parse
sys.stdout.reconfigure(encoding='utf-8')
ENV = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\.env"
keys = {}
for line in open(ENV, encoding='utf-8'):
    for name in ('FOODSAFETY_KEY', 'DATA_GO_KR_KEY'):
        if line.startswith(name + '='):
            keys[name] = line.split('=', 1)[1].strip().strip('"').strip("'")
pats = {}
for name, raw in keys.items():
    forms = {raw, urllib.parse.unquote(raw), urllib.parse.quote(raw, safe=''), urllib.parse.quote_plus(raw),
             urllib.parse.quote(urllib.parse.unquote(raw), safe='')}
    forms |= {f[:12] for f in list(forms) if len(f) >= 12}
    pats[name] = {f.encode() for f in forms if f}
n = 0; hit = 0
for root in sys.argv[1:]:
    files = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
    for f in files:
        b = open(f, 'rb').read(); n += 1
        found = [name for name, ps in pats.items() if any(p in b for p in ps)]
        if found:
            hit += 1; print('키 발견:', f, found)
print(f'검사한 키 {sorted(keys)} / 검사한 파일 {n}개, 키가 들어 있는 파일 {hit}개')
