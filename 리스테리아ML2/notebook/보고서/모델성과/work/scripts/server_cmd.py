# -*- coding: utf-8 -*-
"""팀 서버 읽기 전용 명령·복사 (접속 정보 값은 출력하지 않음).

사용:
  server_cmd.py ls  <원격 경로(~/cyj_listeria/root3 아래 상대)>
  server_cmd.py get <원격 파일(상대)> <로컬 폴더>
- 원격에서는 ls/cat 계열 읽기와 scp 복사만 한다(쓰기·삭제·실행 없음).
- 출력에서 ip·계정 값은 <host>·<user> 로 가린다.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = r'C:\Users\NT751\Desktop\천영진'
INFO = os.path.join(BASE, 'GPU 서버 사용법.md')
KEY = os.path.join(BASE, '보관', 'gpu_server.pem')
REMOTE_ROOT = '~/cyj_listeria/root3'

txt = open(INFO, encoding='utf-8').read()
host = re.search(r'^ip:\s*(\S+)', txt, re.M).group(1)
user = re.search(r'^계정:\s*(\S+)', txt, re.M).group(1)


def mask(s):
    return s.replace(host, '<host>').replace(user, '<user>')


def run(args):
    p = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = mask(p.stdout)
    err = mask(p.stderr)
    if out:
        print(out, end='' if out.endswith('\n') else '\n')
    if err.strip():
        print('[stderr]', err.strip())
    return p.returncode


SSH = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
cmd = sys.argv[1]
if cmd == 'ls':
    rel = sys.argv[2]
    # 경로에 위험 문자 금지(읽기 전용 ls 만)
    assert re.fullmatch(r'[A-Za-z0-9_./*\-]*', rel), rel
    rc = run(SSH + [f'{user}@{host}', f'ls -la {REMOTE_ROOT}/{rel}'])
    sys.exit(rc)
elif cmd == 'get':
    rel = sys.argv[2]
    local = sys.argv[3]
    assert re.fullmatch(r'[A-Za-z0-9_./*\-]*', rel), rel
    os.makedirs(local, exist_ok=True)
    rc = run(['scp', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
              f'{user}@{host}:{REMOTE_ROOT}/{rel}', local + os.sep])
    sys.exit(rc)
else:
    raise SystemExit('unknown cmd')
