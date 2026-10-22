# -*- coding: utf-8 -*-
"""세션 기록 jsonl 의 특정 줄(메시지) 글 전체를 출력 (읽기만)
사용: python print_session_line.py <세션ID 앞부분> <줄 번호> [<줄 번호> ...]
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

D = Path(r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----')
sid = sys.argv[1]
want = {int(x) for x in sys.argv[2:]}
f = next(p for p in sorted(D.glob('*.jsonl')) if p.name.startswith(sid))
with open(f, encoding='utf-8', errors='replace') as fh:
    for ln, line in enumerate(fh):
        if ln not in want:
            continue
        o = json.loads(line)
        m = o.get('message', {})
        print(f'===== {f.name[:8]} line{ln} {m.get("role")} {o.get("timestamp")}')
        c = m.get('content')
        if isinstance(c, str):
            print(c)
        else:
            for b in c or []:
                if not isinstance(b, dict):
                    continue
                if b.get('type') == 'text':
                    print(b.get('text', ''))
                elif b.get('type') == 'tool_use':
                    print('[tool_use]', json.dumps(b.get('input', {}), ensure_ascii=False)[:3000])
                elif b.get('type') == 'tool_result':
                    cc = b.get('content')
                    if isinstance(cc, list):
                        cc = '\n'.join(x.get('text', '') for x in cc if isinstance(x, dict))
                    print('[tool_result]', str(cc)[:3000])
        print()
