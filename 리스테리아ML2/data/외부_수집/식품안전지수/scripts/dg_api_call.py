# 공공데이터포털 API 1회 호출 → 받은 그대로 파일 저장. 인증키는 .env 에서 읽고 절대 출력·저장하지 않음.
# 사용: python -I dg_api_call.py <env파일> <요청주소> <저장파일> [키이름=ServiceKey] [이름=값 ...]
#   예: ... https://apis.data.go.kr/B553748/CertCompanyListService2/getCertCompanyListService2 out.json ServiceKey returnType=json pageNo=1 numOfRows=1
import sys, json, re, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
envp, url, outp = sys.argv[1], sys.argv[2], sys.argv[3]
keyname = sys.argv[4] if len(sys.argv) > 4 else 'ServiceKey'
params = dict(a.split('=', 1) for a in sys.argv[5:])
key = None
for line in open(envp, encoding='utf-8'):
    if line.startswith('DATA_GO_KR_KEY='):
        key = urllib.parse.unquote(line.split('=', 1)[1].strip().strip('"').strip("'"))
if not key:
    print('키 없음'); sys.exit(1)
q = urllib.parse.urlencode({keyname: key, **params})
req = urllib.request.Request(url + '?' + q, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read(); status = r.status
except urllib.error.HTTPError as e:
    body = e.read(); status = e.code
# 혹시 응답에 키가 들어 있으면 지움(인코딩 꼴 포함)
for k in (key, urllib.parse.quote(key, safe=''), urllib.parse.quote_plus(key)):
    body = body.replace(k.encode(), b'[KEY]')
open(outp, 'wb').write(body)
txt = body.decode('utf-8', errors='replace')
m = {t: re.findall(rf'"?{t}"?\s*[:>]\s*"?([^",<}}]*)', txt)[:1] for t in ('resultCode', 'resultMsg', 'resultMessage', 'totalCount', 'returnReasonCode', 'returnAuthMsg', 'errMsg')}
print('HTTP', status, '| 크기', len(body), '|', {k: v for k, v in m.items() if v})
print(txt[:300].replace('\n', ' '))
