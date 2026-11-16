# -*- coding: utf-8 -*-
"""보안 점검 준비용 파일 목록 (읽기 전용: 디렉터리 메타데이터만 읽음, 파일 내용은 열지 않음).

사용: py -3 scan_inventory.py <출력폴더> <로그파일> <PowerShell 목록 TSV> [제외폴더 ...]
- 기본 목록은 Python os.scandir (junction·symlink 는 따라가지 않음).
- 보안 프로그램이 Python 에게 숨기는 폴더(%I9F9'T 등)는 PowerShell 목록(TSV)으로 보충.
"""
import os, sys, csv, re
from datetime import datetime
from collections import defaultdict

OUT_DIR, LOG_PATH, PS_TSV = sys.argv[1], sys.argv[2], sys.argv[3]
EXCLUDE = {os.path.normcase(os.path.abspath(p)) for p in sys.argv[4:]}
EXCLUDE.add(os.path.normcase(os.path.abspath(OUT_DIR)))

PROFILE = r'C:\Users\NT751'
CUTOFF = datetime(2026, 9, 1).timestamp()
PREFIX = '\\\\?\\'
CHEON = 'Desktop\\천영진'
DECOY_NOTE = '보안 프로그램 감시용 파일(추정) - 암호화·삭제 금지'


def lp(p):
    return p if p.startswith(PREFIX) else PREFIX + p


def sp(p):
    return p[len(PREFIX):] if p.startswith(PREFIX) else p


def excluded(path):
    n = os.path.normcase(path)
    return any(n == e or n.startswith(e + '\\') for e in EXCLUDE)


# ---------------- PowerShell 목록 읽기 ----------------
ps_files, ps_err, ps_link = {}, [], []
with open(PS_TSV, encoding='utf-8-sig') as f:
    for line in f:
        t = line.rstrip('\r\n').split('\t')
        if t[0] == 'F' and len(t) >= 5:
            if not excluded(t[1]):
                ps_files[os.path.normcase(t[1])] = (t[1], int(t[2]), t[3], t[4])
        elif t[0] == 'ERR':
            ps_err.append(t[1:])
        elif t[0] == 'LINK':
            ps_link.append(t[1])

hidden_roots = sorted({v[0].split('\\')[0] + '\\' + v[0].split('\\')[1] for v in ps_files.values()
                       if v[0].split('\\')[1].startswith('%')})
FILE_ROOTS = [PROFILE, r'C:\Users\Public', r'C:\tmp', r'C:\Temp'] + hidden_roots
SUMMARY_ROOTS = [r'C:\GreenT', r'C:\KCopa', r'C:\SecuYouSB_MFDS', r'C:\CKBS_FTP', r'C:\ESD',
                 r'C:\inetpub', r'C:\MSOCache', r'C:\PerfLogs', r'C:\BIZET.bin']

# ---------------- 분류 규칙 (이름·경로·확장자만) ----------------
IP_RE = re.compile(r'(?<![\d.])(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)')
TOKEN_RE = re.compile(r'(?:^|[^a-z])tokens?(?:[^a-z]|$)')
CONN_TOKEN_RE = re.compile(r'(?:^|[^a-z])(?:ssh|scp|ip|host|hosts|dsn)(?:[^a-z]|$)')
SYNC_RE = re.compile(r'(?:^|[^a-z])(?:sync|deploy)(?:[^a-z]|$)')

ACCT_EXT = {'.pem', '.key', '.ppk', '.pfx', '.p12', '.jks', '.keystore', '.kdbx', '.kdb', '.ovpn', '.rdp'}
ACCT_NAMES = {'.env', 'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519', '.git-credentials', '.netrc', '_netrc',
              '.pgpass', '.pypirc', '.npmrc', 'credentials', 'kaggle.json', 'authorized_keys', 'token'}
ACCT_KW = ['credential', 'secret', 'password', 'passwd', '비밀번호', '비번', '계정', '패스워드', '인증키',
           'apikey', 'api_key', 'api-key', 'private_key', 'privatekey', '접속정보', '접속 정보',
           '로그인정보', '로그인 정보']
BROWSER_LOGIN = {'login data', 'login data for account', 'cookies', 'web data'}
CONN_KW = ['db_conn', 'dbconn', 'database', 'datasource', 'connection', 'connect', '접속', '서버', 'server',
           'sftp', 'ftp', 'rsync', 'tunnel', 'known_hosts']
NO_NAME_RULE_EXT = {'.exe', '.msi', '.dll', '.lnk', '.searchconnector-ms', '.ini'}
SCRIPT_EXT = {'.sh', '.ps1', '.bat', '.cmd', '.py'}
HIST_NAMES = {'.lesshst', 'history.sqlite', 'consolehost_history.txt', '.python_history', '.node_repl_history',
              '.bash_history'}
DOC_EXT = {'.hwp', '.hwpx', '.doc', '.docx', '.xls', '.xlsx', '.xlsm', '.csv', '.pdf', '.ppt', '.pptx', '.txt',
           '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.zip', '.7z', '.alz', '.egg', '.rtf',
           '.odt', '.ods', '.md', '.msg', '.eml'}
PI_KW = ['명단', '주민', '연락처', '전화번호', '휴대폰', '핸드폰', '주소록', '개인정보', '인적', '이력서', '자기소개서',
         '신상', '회원', '급여', '계좌', '통장', '신분증', '여권', '등본', '초본', '가족관계', '비상연락', '인사기록',
         '성명', '생년월일', '증명서', '자격증', '면허증', '근로계약', '연봉계약', '원천징수', '소득금액', '납세',
         '건강검진', '진단서', '신청서', '지원서', '동의서', '참석자', '서약서', '위임장', '사업자등록', '명함',
         '조직도', 'resume', '주소']
DATA_EXT = {'.parquet', '.csv', '.tsv', '.xlsx', '.xls', '.xlsm', '.feather', '.pkl', '.pickle', '.db',
            '.sqlite', '.h5', '.hdf5'}
RAW_KW = ['시료', 'lims', '축사', '농장', '업체', '주소', 'addr', '인허가', '영업', '조인', '전달', 'sample',
          '대장', 'mnfctur', 'cmpny']
RAW_NAMES = {'listeria_data.parquet', 'listeria_body.parquet'}
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.pdf'}

LIB_PARTS = {'site-packages', 'dist-packages', 'node_modules', 'anaconda3', 'miniconda3', '.conda', 'pkgs',
             '.vscode', '.vscode-shared', '.cache', '.cupy', '.keras', '.matplotlib', '.continuum', '.anaconda',
             '.git', '__pycache__'}
SENS_ORDER = ['계정·키', '접속 정보(서버·DB·IP) 가능', '명령 기록(접속 정보 가능)', '개인정보 가능 문서',
              '원자료(업체·주소 포함 가능)', '스캔·화면 캡처', 'Claude 대화·메모 기록(내부 IP·계정 언급 가능)',
              '경로에 내부 IP 형태']


def is_lib(root_is_profile, dpl):
    if root_is_profile and dpl[:2] == ['.claude', 'plugins']:
        return True   # Claude 플러그인(외부 배포 코드)
    for p in dpl:
        if p in LIB_PARTS or p.startswith('.venv') or p == 'venv':
            return True
    return False


def regen_of(dpl):
    """재생성 가능 구분과 그 기준 폴더 깊이(index)."""
    for i, p in enumerate(dpl):
        if p.startswith('.venv') or p == 'venv' or p in ('site-packages', 'node_modules'):
            return '가상환경', i
        if p in ('anaconda3', 'miniconda3', '.conda'):
            return '프로그램(재설치 가능)', i
        if p == '.git':
            return 'git 기록', i
        if p in ('__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.ipynb_checkpoints'):
            return '캐시', i
        if i == 0 and p == '.local':
            return '프로그램(재설치 가능)', i
        if i == 0 and p in ('.cache', '.cupy', '.matplotlib', '.keras', '.vscode', '.vscode-shared'):
            return '캐시·프로그램(재생성 가능)', i
    return '', -1


def sensitive_of(root_is_profile, parts, full_path):
    nl = parts[-1].lower()
    ext = os.path.splitext(nl)[1]
    dpl = [p.lower() for p in parts[:-1]]
    l1 = dpl[0] if dpl else ''
    pdir = dpl[-1] if dpl else ''
    r = set()
    # 특정 저장소(라이브러리 폴더 안이어도 표시)
    if root_is_profile and dpl[:2] == ['.cache', 'huggingface'] and nl == 'token':
        r.add('계정·키')
    if root_is_profile and dpl[:2] == ['.config', 'gh'] and nl == 'hosts.yml':
        r.add('계정·키')
    browser = any(p in ('chrome_profile', 'user data') for p in dpl)
    if browser:
        if nl in BROWSER_LOGIN:
            r.add('계정·키')       # 브라우저 로그인·쿠키 저장 파일
    elif not is_lib(root_is_profile, dpl):
        named = ext not in NO_NAME_RULE_EXT
        # A 계정·키
        if (ext in ACCT_EXT or nl in ACCT_NAMES or nl.startswith('.env.') or nl.endswith('.env')
                or (root_is_profile and l1 == '.ssh')
                or (named and (any(k in nl for k in ACCT_KW) or TOKEN_RE.search(nl)
                               or ('토큰' in nl and '토큰화' not in nl) or nl.endswith('가이드.txt')
                               or ('서버' in nl and '사용법' in nl)))):
            r.add('계정·키')
        # B 접속 정보
        if named and (any(k in nl for k in CONN_KW) or CONN_TOKEN_RE.search(nl)
                      or (ext in SCRIPT_EXT and SYNC_RE.search(nl))):
            r.add('접속 정보(서버·DB·IP) 가능')
        # C 명령 기록
        if nl in HIST_NAMES or nl.endswith('_history'):
            r.add('명령 기록(접속 정보 가능)')
        # D 개인정보 가능 문서
        if ext in DOC_EXT and any(k in nl or k in pdir for k in PI_KW):
            r.add('개인정보 가능 문서')
        # E 원자료(업체·주소)
        if ext in DATA_EXT and (nl in RAW_NAMES or any(k in nl or k in pdir for k in RAW_KW)):
            r.add('원자료(업체·주소 포함 가능)')
        # F 스캔·화면 캡처
        if ext in IMG_EXT and (any(p in ('scan', '스캔', 'screenshots') for p in dpl)
                               or any(k in nl for k in ('스크린샷', 'screenshot', '캡처'))):
            r.add('스캔·화면 캡처')
        # G Claude 대화·메모 기록
        if root_is_profile and (
                (l1 == '.claude' and (ext == '.jsonl'
                                      or ('memory' in dpl and ext == '.md')
                                      or (len(dpl) > 1 and dpl[1] == 'projects' and 'tool-results' in dpl)))
                or (len(parts) == 1 and nl.startswith('.claude.json'))):
            r.add('Claude 대화·메모 기록(내부 IP·계정 언급 가능)')
    # H 경로에 IP 형태
    if IP_RE.search(full_path):
        r.add('경로에 내부 IP 형태')
    return '; '.join(x for x in SENS_ORDER if x in r)


def note_of(parts):
    if any(p.lower().startswith('%i9') for p in parts):
        return DECOY_NOTE
    return ''


# ---------------- 집계 ----------------
def new_agg():
    return [0, 0, 0, 0, 0, 0, 0.0]   # 파일 수, 크기, 9월 이후 수, 9월 이후 크기, 민감 수, 재생성 크기, 최근 고친 시각


agg = defaultdict(new_agg)          # (단계, 경로) -> 집계
deep = defaultdict(new_agg)         # 천영진 아래 깊은 폴더(로그용)
regen_roots = defaultdict(new_agg)  # 재생성 가능 폴더(로그용)
sens_hits = []                      # (경로, 크기, 이유, 9월 이후)
errors, skipped_links, ps_only, ps_skip = [], [], [], []
seen = set()
listed = [0, 0, 0, 0]               # 수, 크기, 9월 이후 수, 9월 이후 크기
appdata_tot = [0, 0, 0, 0]


def add(key, size, is_new, sens, regen, mtime):
    a = agg[key]
    a[0] += 1; a[1] += size
    if is_new:
        a[2] += 1; a[3] += size
    if sens:
        a[4] += 1
    if regen:
        a[5] += size
    if mtime > a[6]:
        a[6] = mtime


def fmt(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')


out_csv = os.path.join(OUT_DIR, '파일목록.csv')
fh = open(out_csv, 'w', newline='', encoding='utf-8-sig')
w = csv.writer(fh)
w.writerow(['경로', '크기(바이트)', '만든 날', '고친 날', '최상위 폴더', '2단계 폴더', '9월 이후 여부',
            '민감 의심', '재생성 가능', '비고'])


def handle(root, path, size, ct, mt, source):
    root_is_profile = (root == PROFILE)
    rel = path[len(root) + 1:]
    parts = rel.split('\\')
    is_new = (ct >= CUTOFF) or (mt >= CUTOFF)
    dpl = [p.lower() for p in parts[:-1]]
    if root_is_profile and parts[0].lower() == 'appdata':
        l2 = 'AppData\\' + parts[1] if len(parts) > 2 else 'AppData\\(바로 아래 파일)'
        l3 = (l2 + '\\' + parts[2]) if len(parts) > 3 else ((l2 + '\\(바로 아래 파일)') if len(parts) == 3 else None)
        add((1, PROFILE + '\\AppData'), size, is_new, '', '', mt)
        add((2, PROFILE + '\\' + l2), size, is_new, '', '', mt)
        if l3:
            add((3, PROFILE + '\\' + l3), size, is_new, '', '', mt)
        appdata_tot[0] += 1; appdata_tot[1] += size
        if is_new:
            appdata_tot[2] += 1; appdata_tot[3] += size
        return
    sens = sensitive_of(root_is_profile, parts, path)
    regen, ridx = regen_of(dpl)
    note = note_of(parts)
    if source == 'ps':
        note = (note + '; ' if note else '') + 'Python 목록에 안 보여 PowerShell 로 확인'
    if root_is_profile:
        if len(parts) == 1:
            top1 = '(사용자 폴더 바로 아래 파일)'; top2 = top1
            k1 = PROFILE + '\\' + top1; k2 = None
        else:
            top1 = parts[0]
            top2 = parts[0] + '\\' + (parts[1] if len(parts) > 2 else '(바로 아래 파일)')
            k1 = PROFILE + '\\' + top1; k2 = PROFILE + '\\' + top2
    else:
        top1 = root
        top2 = root + '\\' + (parts[0] if len(parts) > 1 else '(바로 아래 파일)')
        k1 = root; k2 = top2
    add((1, k1), size, is_new, sens, regen, mt)
    if k2:
        add((2, k2), size, is_new, sens, regen, mt)
    if root_is_profile and top2 == CHEON:
        k3 = PROFILE + '\\' + CHEON + '\\' + (parts[2] if len(parts) > 3 else '(바로 아래 파일)')
        add((3, k3), size, is_new, sens, regen, mt)
        for depth in range(3, min(len(parts), 7)):
            d = deep[PROFILE + '\\' + '\\'.join(parts[:depth])]
            d[0] += 1; d[1] += size
            if regen:
                d[5] += size
            if is_new:
                d[2] += 1; d[3] += size
            if sens:
                d[4] += 1
    if regen:
        r = regen_roots[(regen, root + '\\' + '\\'.join(parts[:ridx + 1]))]
        r[0] += 1; r[1] += size
    if sens:
        sens_hits.append((path, size, sens, is_new))
    listed[0] += 1; listed[1] += size
    if is_new:
        listed[2] += 1; listed[3] += size
    w.writerow([path, size, fmt(ct), fmt(mt), top1, top2, '이후' if is_new else '이전', sens, regen, note])


def walk(root, top):
    try:
        it = os.scandir(lp(top))
    except OSError as e:
        errors.append((top, type(e).__name__))
        return
    files, dirs = [], []
    with it:
        for e in it:
            try:
                if e.is_symlink() or e.is_junction():
                    skipped_links.append(sp(e.path))
                    continue
                if e.is_dir(follow_symlinks=False):
                    dirs.append(e)
                else:
                    files.append(e)
            except OSError as ex:
                errors.append((sp(e.path), type(ex).__name__))
    files.sort(key=lambda x: x.name.casefold())
    dirs.sort(key=lambda x: x.name.casefold())
    for f in files:
        p = sp(f.path)
        try:
            st = f.stat(follow_symlinks=False)
        except OSError as ex:
            errors.append((p, type(ex).__name__))
            continue
        seen.add(os.path.normcase(p))
        handle(root, p, st.st_size, getattr(st, 'st_birthtime', st.st_ctime), st.st_mtime, 'py')
    for d in dirs:
        p = sp(d.path)
        if excluded(p):
            continue
        walk(root, p)


for root in FILE_ROOTS:
    if os.path.isdir(root):
        walk(root, root)

# PowerShell 에만 보이는 파일 보충
roots_sorted = sorted(FILE_ROOTS, key=len, reverse=True)
for key in sorted(ps_files, key=lambda k: ps_files[k][0].casefold()):
    if key in seen:
        continue
    path, size, cts, mts = ps_files[key]
    if not any(x.lower().startswith('%i9') for x in path.split('\\')):
        ps_skip.append(path)   # 두 목록 사이에 생기거나 지워진 파일(시차) - 목록에 넣지 않음
        continue
    root = next((r for r in roots_sorted if path.lower().startswith(r.lower() + '\\')), None)
    if root is None:
        continue
    ct = datetime.strptime(cts, '%Y-%m-%d %H:%M').timestamp()
    mt = datetime.strptime(mts, '%Y-%m-%d %H:%M').timestamp()
    ps_only.append(path)
    handle(root, path, size, ct, mt, 'ps')
fh.close()
py_only = len(seen - set(ps_files))

# 프로그램 폴더 등: 요약만
summary_only = {}
for root in SUMMARY_ROOTS:
    tot = [0, 0, 0, 0, 0.0]
    if os.path.isdir(root):
        for dp, dn, fn in os.walk(root):
            for n in fn:
                try:
                    st = os.stat(os.path.join(dp, n))
                except OSError:
                    continue
                ct, mt = getattr(st, 'st_birthtime', st.st_ctime), st.st_mtime
                tot[0] += 1; tot[1] += st.st_size
                if ct >= CUTOFF or mt >= CUTOFF:
                    tot[2] += 1; tot[3] += st.st_size
                tot[4] = max(tot[4], mt)
    elif os.path.isfile(root):
        st = os.stat(root)
        tot = [1, st.st_size, 0, 0, st.st_mtime]
    else:
        continue
    summary_only[root] = tot


# ---------------- 폴더별 요약 ----------------
def human(n):
    for u, d in (('GB', 1024 ** 3), ('MB', 1024 ** 2), ('KB', 1024)):
        if n >= d:
            return f'{n / d:.1f} {u}'
    return f'{n} B'


NOTE_APPDATA = '프로그램 설정·캐시(파일 단위 목록 없음, 요약만)'
sum_csv = os.path.join(OUT_DIR, '폴더별_요약.csv')
with open(sum_csv, 'w', newline='', encoding='utf-8-sig') as sh:
    sw = csv.writer(sh)
    sw.writerow(['단계', '폴더', '파일 수', '전체 크기(바이트)', '전체 크기', '9월 이후 파일 수', '9월 이후 크기(바이트)',
                 '민감 의심 파일 수', '재생성 가능 크기(바이트)', '최근 고친 날', '비고'])

    def row(k, note=''):
        a = agg[k]
        app = '\\AppData' in k[1]
        if not note and any(p.lower().startswith('%i9') for p in k[1].split('\\')):
            note = DECOY_NOTE
        sw.writerow([k[0], k[1], a[0], a[1], human(a[1]), a[2], a[3], '' if app else a[4], '' if app else a[5],
                     fmt(a[6]) if a[6] else '', note])

    for k1 in sorted([k for k in agg if k[0] == 1], key=lambda k: k[1].casefold()):
        isapp = k1[1] == PROFILE + '\\AppData'
        row(k1, NOTE_APPDATA if isapp else '')
        for k2 in sorted([k for k in agg if k[0] == 2 and k[1].startswith(k1[1] + '\\')], key=lambda k: -agg[k][1]):
            row(k2, NOTE_APPDATA if isapp else '')
            for k3 in sorted([k for k in agg if k[0] == 3 and k[1].startswith(k2[1] + '\\')],
                             key=lambda k: -agg[k][1]):
                row(k3, NOTE_APPDATA if isapp else '')
    for root, t in summary_only.items():
        sw.writerow([1, root, t[0], t[1], human(t[1]), t[2], t[3], '', '', fmt(t[4]) if t[4] else '',
                     '사용자 폴더 밖(C:\\ 바로 아래) 프로그램·보안 프로그램 폴더 - 요약만'])

# ---------------- 로그(내 작업용) ----------------
with open(LOG_PATH, 'w', encoding='utf-8') as lg:
    P = lambda *a: print(*a, file=lg)
    P('FILE_ROOTS', FILE_ROOTS)
    P('listed files', listed[0], 'bytes', listed[1], human(listed[1]))
    P('listed after0901', listed[2], 'bytes', listed[3], human(listed[3]))
    P('appdata', appdata_tot, human(appdata_tot[1]), 'after', human(appdata_tot[3]))
    P('summary_only', {k: (v[0], human(v[1]), v[2]) for k, v in summary_only.items()})
    P('ps files', len(ps_files), 'ps_only', len(ps_only), 'py_only(not in ps)', py_only)
    P('ps err', len(ps_err), ps_err[:20])
    P('ps_skip (timing)', len(ps_skip), ps_skip)
    for p in ps_only:
        P('  PS_ONLY', p)
    P('py errors', len(errors), errors[:40])
    P('skipped links', len(skipped_links))
    P('\n== L1/L2/L3 (non-AppData) ==')
    for k in sorted(agg, key=lambda k: (k[1].casefold(), k[0])):
        if '\\AppData' in k[1] and k[0] == 3:
            continue
        a = agg[k]
        P(f'L{k[0]}\t{human(a[1]):>10}\t{a[0]}\tnew {a[2]} {human(a[3])}\tsens {a[4]}\tregen {human(a[5])}\t{k[1]}')
    P('\n== 천영진 deep (>=20MB or sens) ==')
    for k in sorted(deep, key=lambda k: k.casefold()):
        d = deep[k]
        if d[1] >= 20 * 1024 ** 2 or d[4]:
            P(f'{human(d[1]):>10}\t{d[0]}\tregen {human(d[5])}\tnew {d[2]}\tsens {d[4]}\t{k}')
    P('\n== regen roots ==')
    for (cat, k), r in sorted(regen_roots.items(), key=lambda x: -x[1][1]):
        P(f'{human(r[1]):>10}\t{r[0]}\t{cat}\t{k}')
    P('\n== sensitive hits ==', len(sens_hits))
    cnt = defaultdict(int)
    for p, s, rs, n in sens_hits:
        for x in rs.split('; '):
            cnt[x] += 1
    for k, v in cnt.items():
        P('  ', k, v)
    for p, s, rs, n in sens_hits:
        P(f'{rs}\t{"이후" if n else "이전"}\t{s}\t{p}')
print('done', listed, appdata_tot, 'ps_only', len(ps_only))
