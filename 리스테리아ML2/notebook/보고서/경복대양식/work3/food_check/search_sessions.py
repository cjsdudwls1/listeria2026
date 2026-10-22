# -*- coding: utf-8 -*-
"""세션 기록(jsonl)에서 사용자·어시스턴트 글 중 키워드가 든 부분을 찾아 출력 (읽기만)
사용: python search_sessions.py <패턴(정규식)> [세션ID 앞부분 ...] [--ctx N] [--max N]
"""
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

D = Path(r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----')
args = sys.argv[1:]
ctx = 300
mx = 40
if '--ctx' in args:
    i = args.index('--ctx'); ctx = int(args[i + 1]); del args[i:i + 2]
if '--max' in args:
    i = args.index('--max'); mx = int(args[i + 1]); del args[i:i + 2]
pat = re.compile(args[0])
ids = args[1:]
files = sorted(D.glob('*.jsonl'))
if ids:
    files = [f for f in files if any(f.name.startswith(s) for s in ids)]


def texts(msg):
    c = msg.get('content')
    if isinstance(c, str):
        yield c
    elif isinstance(c, list):
        for b in c:
            if isinstance(b, dict):
                if b.get('type') == 'text':
                    yield b.get('text', '')
                elif b.get('type') == 'tool_use':
                    yield json.dumps(b.get('input', {}), ensure_ascii=False)
                elif b.get('type') == 'tool_result':
                    cc = b.get('content')
                    if isinstance(cc, str):
                        yield cc
                    elif isinstance(cc, list):
                        for x in cc:
                            if isinstance(x, dict) and x.get('type') == 'text':
                                yield x.get('text', '')


n = 0
for f in files:
    with open(f, encoding='utf-8', errors='replace') as fh:
        for ln, line in enumerate(fh):
            try:
                o = json.loads(line)
            except Exception:
                continue
            m = o.get('message')
            if not isinstance(m, dict):
                continue
            role = m.get('role')
            ts = o.get('timestamp', '')
            for t in texts(m):
                for mm in pat.finditer(t):
                    s = max(0, mm.start() - ctx)
                    e = min(len(t), mm.end() + ctx)
                    print(f'=== {f.name[:8]} line{ln} {role} {ts}')
                    print(t[s:e].replace('\r', ''))
                    print()
                    n += 1
                    if n >= mx:
                        sys.exit(0)
                    break
print(f'[찾은 수 {n}]')
