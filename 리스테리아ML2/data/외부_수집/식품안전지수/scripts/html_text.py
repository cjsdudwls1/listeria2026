# 받아 둔 html 을 글자만 뽑아 출력(읽기만). 태그·스크립트 제거. 키워드 주면 그 주변만.
# 사용: python -I html_text.py <html> [시작키워드] [글자수]
import sys, re, html
sys.stdout.reconfigure(encoding="utf-8")
p = sys.argv[1]
b = open(p, 'rb').read()
try:
    raw = b.decode('utf-8')
except UnicodeDecodeError:
    raw = b.decode('cp949', errors='replace')
raw = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', raw)
t = html.unescape(re.sub(r'(?s)<[^>]+>', ' ', raw))
t = re.sub(r'[ \t\r\f\v]+', ' ', t)
t = re.sub(r'\n\s*\n+', '\n', t)
if len(sys.argv) > 2 and sys.argv[2]:
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    i = t.find(sys.argv[2])
    print(t[i:i + n] if i >= 0 else '(키워드 없음)')
else:
    print(t[:int(sys.argv[3]) if len(sys.argv) > 3 else 6000])
