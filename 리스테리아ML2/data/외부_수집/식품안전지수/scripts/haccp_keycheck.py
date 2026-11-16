# 저장한 파일에 인증키(원래 꼴·URL 인코딩 꼴)가 들어 있는지 검사. 키 값은 출력하지 않음.
# 실행: py -I haccp_keycheck.py <폴더 또는 파일> ...
import sys, os, urllib.parse
sys.stdout.reconfigure(encoding='utf-8')
ENV = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\.env"
key = None
for line in open(ENV, encoding='utf-8'):
    if line.startswith('DATA_GO_KR_KEY='):
        raw = line.split('=', 1)[1].strip().strip('"').strip("'")
        key = urllib.parse.unquote(raw)
pats = {key.encode(), urllib.parse.quote(key, safe='').encode(), urllib.parse.quote_plus(key).encode(), raw.encode()}
pats |= {p[:16] for p in list(pats)}  # 앞 16글자만 들어간 경우도
n = hit = 0
for root in sys.argv[1:]:
    files = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
    for f in files:
        b = open(f, 'rb').read(); n += 1
        if any(p in b for p in pats):
            hit += 1; print('키 발견:', f)
print(f'검사한 파일 {n}개, 키가 들어 있는 파일 {hit}개')
