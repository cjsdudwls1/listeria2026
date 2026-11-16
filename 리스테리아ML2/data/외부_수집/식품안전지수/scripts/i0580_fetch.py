# 식품안전나라 오픈API I0580(HACCP 적용업소 지정 현황)을 구간(start~end)마다 받아 받은 그대로 JSON 파일로 저장.
# 인증키는 .env 의 FOODSAFETY_KEY 를 읽어 쓰고, 출력·파일·로그에 쓰지 않음(응답에 들어 있으면 [KEY] 로 지움).
# 시스템 py(3.14, 표준 라이브러리만)로 실행:
#   py -I i0580_fetch.py <outdir> <logfile> <한 번 행 수> <첫 시작 번호> [마지막 끝 번호|auto]
#   py -I i0580_fetch.py <outdir> <logfile> once <start> <end>      (한 구간만, 시험용)
# 이미 받은(정상 응답) 구간 파일은 건너뜀. 실패한 구간은 간격을 두고 다시 시도. 호출 사이 0.5초 이상.
import sys, os, json, time, urllib.parse, urllib.request, urllib.error

sys.stdout.reconfigure(encoding="utf-8")
ENV = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\.env"
BASE = "http://openapi.foodsafetykorea.go.kr/api/{key}/I0580/json/{s}/{e}"


def load_key():
    for line in open(ENV, encoding='utf-8'):
        if line.startswith('FOODSAFETY_KEY='):
            return line.split('=', 1)[1].strip().strip('"').strip("'")
    raise SystemExit('키 없음')


def scrub(body, key):
    for k in {key, urllib.parse.quote(key, safe=''), urllib.parse.quote_plus(key), urllib.parse.unquote(key)}:
        if k:
            body = body.replace(k.encode(), b'[KEY]')
    return body


def parse(body):
    try:
        o = json.loads(body.decode('utf-8'))
    except Exception:
        return None, None, None, None
    svc = o.get('I0580') if isinstance(o, dict) else None
    if not isinstance(svc, dict):
        res = o.get('RESULT', {}) if isinstance(o, dict) else {}
        return None, None, res.get('CODE'), res.get('MSG')
    res = svc.get('RESULT', {}) or {}
    rows = svc.get('row')
    return svc.get('total_count'), rows, res.get('CODE'), res.get('MSG')


def call(key, s, e, timeout):
    url = BASE.format(key=urllib.parse.quote(key, safe=''), s=s, e=e)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as ex:
        return ex.code, ex.read()
    except Exception as ex:  # 시간 초과·연결 끊김 (메시지에 주소가 들어갈 수 있어 종류만 남김)
        return -1, type(ex).__name__.encode()


def fetch_one(key, outdir, log, s, e, tries_max=6, timeout=120, wait=10):
    fn = os.path.join(outdir, f'r{s:06d}_{e:06d}.json')
    for attempt in range(1, tries_max + 1):
        t0 = time.time()
        status, body = call(key, s, e, timeout)
        body = scrub(body, key)
        dt = time.time() - t0
        total, rows, code, msg = parse(body)
        n = len(rows) if isinstance(rows, list) else None
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tI0580\tstart={s}\tend={e}\ttry={attempt}\tHTTP={status}"
                f"\tsec={dt:.1f}\tbytes={len(body)}\ttotal_count={total}\tn={n}\tCODE={code}\tMSG={msg}")
        log.write(line + '\n'); log.flush()
        print(line, flush=True)
        if status == 200 and isinstance(rows, list) and code == 'INFO-000':
            open(fn, 'wb').write(body)
            fail = os.path.join(outdir, f'_실패_r{s:06d}.bin')
            if os.path.exists(fail):
                os.remove(fail)
            return total, n, body
        open(os.path.join(outdir, f'_실패_r{s:06d}.bin'), 'wb').write(body[:5000])
        if code is not None and code != 'INFO-000' and status == 200:
            # 오류 코드(행 수 초과 등)는 다시 해도 같으므로 바로 끝냄
            return total, None, body
        time.sleep(min(wait * attempt, 60))
    return None, None, None


def main():
    outdir, logfile, rows = sys.argv[1], sys.argv[2], sys.argv[3]
    key = load_key()
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    log = open(logfile, 'a', encoding='utf-8')
    if rows == 'once':
        s, e = int(sys.argv[4]), int(sys.argv[5])
        fetch_one(key, outdir, log, s, e, tries_max=1)
        return
    rows = int(rows); s = int(sys.argv[4])
    last = sys.argv[5] if len(sys.argv) > 5 else 'auto'
    last = None if last == 'auto' else int(last)
    totals = set()
    while last is None or s <= last:
        e = s + rows - 1 if last is None else min(s + rows - 1, last)
        fn = os.path.join(outdir, f'r{s:06d}_{e:06d}.json')
        if os.path.exists(fn):
            total, r, code, _ = parse(open(fn, 'rb').read())
            if isinstance(r, list) and code == 'INFO-000':
                totals.add(total)
                if last is None and total is not None:
                    last = int(total)
                s = e + 1
                continue
        total, n, body = fetch_one(key, outdir, log, s, e)
        if n is None:
            print(f'{s}~{e} 실패 — 중단', flush=True)
            sys.exit(2)
        totals.add(total)
        if last is None and total is not None:
            last = int(total)
            print('마지막 번호 =', last, flush=True)
        s = e + 1
        time.sleep(0.5)
    print('끝. total_count 값들 =', sorted(totals), flush=True)


if __name__ == '__main__':
    main()
