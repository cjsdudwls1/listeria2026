# data.go.kr 검색 결과 html 에서 데이터셋 제목·링크·기관을 뽑아 출력(읽기만).
# 사용: python -I dg_search_parse.py <html>
import sys, re, html
sys.stdout.reconfigure(encoding="utf-8")
raw = open(sys.argv[1], encoding='utf-8', errors='replace').read()
# 결과 항목: <a href="/data/15050403/fileData.do..."> 다음에 <span class="title">...</span>
for m in re.finditer(r'href="(/data/(\d+)/(fileData|openapi|standard)\.do)[^"]*"[^>]*>\s*(?:<span[^>]*>[^<]*</span>\s*)*<span class="title">\s*(.*?)\s*</span>', raw, re.S):
    title = html.unescape(re.sub(r'<[^>]+>', '', m.group(4))).strip()
    title = re.sub(r'\s+', ' ', title)
    print(m.group(2), m.group(3), '|', title)
