# HACCP 공공데이터포털 API 를 쪽(page)마다 받아 받은 그대로 JSON 파일로 저장.
# 인증키는 .env 의 DATA_GO_KR_KEY 를 읽어 쓰고, 출력·파일·로그에 쓰지 않음(응답에 들어 있으면 [KEY] 로 지움).
# 시스템 py(3.14, 표준 라이브러리만)로 실행:  py -I haccp_fetch.py <svc> <outdir> <numOfRows> <startPage> [endPage|auto] [이름=값 ...]
#   svc: mfds = 15058303 식품의약품안전처_HACCP 지정현황, cert = 15033311 한국식품안전관리인증원_HACCP 인증업체정보
# 이미 받은(정상 응답) 쪽 파일은 건너뜀. 실패한 쪽은 간격을 두고 다시 시도.
import sys, os, json, time, urllib.parse, urllib.request, urllib.error

sys.stdout.reconfigure(encoding="utf-8")
ENV = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\.env"
SVC = {
    'mfds': ('https://apis.data.go.kr/1471000/HaccpAppnSttusService02/getHaccpAppnSttusList02', 'serviceKey', {'type': 'json'}),
    'cert': ('https://apis.data.go.kr/B553748/CertCompanyListService2/getCertCompanyListService2', 'ServiceKey', {'returnType': 'json'}),
}

def load_key():
    for line in open(ENV, encoding='utf-8'):
        if line.startswith('DATA_GO_KR_KEY='):
            return urllib.parse.unquote(line.split('=', 1)[1].strip().strip('"').strip("'"))
    raise SystemExit('키 없음')

def scrub(body, key):
    for k in (key, urllib.parse.quote(key, safe=''), urllib.parse.quote_plus(key)):
        body = body.replace(k.encode(), b'[KEY]')
    return body

def find_items(obj):
    """응답 안의 행 목록과 totalCount 를 찾음(구조가 서비스마다 조금 다름)."""
    total = None; items = None
    def walk(o):
        nonlocal total, items
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'totalCount' and total is None:
                    total = v
                if k in ('items', 'item') and items is None:
                    if isinstance(v, list):
                        items = [x.get('item', x) if isinstance(x, dict) else x for x in v]
                    elif isinstance(v, dict) and 'item' in v:
                        it = v['item']
                        items = it if isinstance(it, list) else [it]
                walk(v) if isinstance(v, (dict, list)) else None
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(obj)
    return total, items

def call(url, keyname, key, params, timeout):
    q = urllib.parse.urlencode({keyname: key, **params})
    req = urllib.request.Request(url + '?' + q, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # 시간 초과·연결 끊김
        return -1, repr(e).encode()

def main():
    svc, outdir, rows, start = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    end = sys.argv[5] if len(sys.argv) > 5 else 'auto'
    extra = dict(a.split('=', 1) for a in sys.argv[6:])
    url, keyname, base = SVC[svc]
    key = load_key()
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, '_호출기록.tsv'), 'a', encoding='utf-8')
    page = start; last = None if end == 'auto' else int(end)
    tries_max = int(extra.pop('_tries', '6')); timeout = int(extra.pop('_timeout', '90')); wait = int(extra.pop('_wait', '10'))
    while last is None or page <= last:
        fn = os.path.join(outdir, f'p{page:04d}_n{rows}.json')
        if os.path.exists(fn):
            try:
                t, it = find_items(json.load(open(fn, encoding='utf-8')))
                if it is not None:
                    if last is None and t is not None:
                        last = -(-int(t) // rows)
                    page += 1; continue
            except Exception:
                pass
        ok = False
        for attempt in range(1, tries_max + 1):
            params = {**base, 'pageNo': str(page), 'numOfRows': str(rows), **extra}
            t0 = time.time()
            status, body = call(url, keyname, key, params, timeout)
            body = scrub(body, key)
            dt = time.time() - t0
            total = items = None
            try:
                total, items = find_items(json.loads(body.decode('utf-8')))
            except Exception:
                pass
            n = len(items) if items is not None else None
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{svc}\tpage={page}\trows={rows}\ttry={attempt}\tHTTP={status}\tsec={dt:.1f}\tbytes={len(body)}\ttotal={total}\tn={n}\n"); log.flush()
            print(f'page {page} try {attempt}: HTTP {status}, {dt:.1f}s, {len(body)}B, total={total}, n={n}', flush=True)
            if status == 200 and items is not None:
                open(fn, 'wb').write(body)
                if last is None and total is not None:
                    last = -(-int(total) // rows)
                    print('마지막 쪽 =', last, flush=True)
                ok = True
                break
            # 실패 응답도 확인용으로 남김(덮어씀)
            open(os.path.join(outdir, f'_실패_p{page:04d}.bin'), 'wb').write(body[:5000])
            time.sleep(min(wait * attempt, 60))
        if not ok:
            print(f'page {page} 실패 — 중단', flush=True)
            sys.exit(2)
        if os.path.exists(os.path.join(outdir, f'_실패_p{page:04d}.bin')):
            os.remove(os.path.join(outdir, f'_실패_p{page:04d}.bin'))
        page += 1
        time.sleep(0.3)
    print('끝', flush=True)

if __name__ == '__main__':
    main()
