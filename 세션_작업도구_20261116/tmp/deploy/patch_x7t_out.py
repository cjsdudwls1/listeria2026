# -*- coding: utf-8 -*-
"""얼린 사본의 scripts/s3_extra.py 에서 ⑦-T(x7-T)를 뺌(담당자 결정 2026-11-10) — 로컬 원본과 같은 세 곳만 바꿈.
사용: python patch_x7t_out.py <s3_extra.py 경로>   (각 바꿀 곳이 정확히 한 번 있어야 함, 아니면 아무것도 안 쓰고 멈춤)
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
EDITS = [
    ('REPL_NAMES = ("x6-LSTMA", "x7-T", "x7-D")',
     'REPL_NAMES = ("x6-LSTMA", "x7-D")   # ⑦-T(x7-T, 시계열만 XGBoost)는 제외(담당자 결정 2026-11-10 — 시간만 듦)'),
    ('    where = {"x6-LSTMA": "x678-G", "x7-D": "x678-G", "x7-T": "x678-C"}',
     '    where = {"x6-LSTMA": "x678-G", "x7-D": "x678-G"}'),
    ('    if sub not in ("G", "C", "E"):\n        raise cf.ConfigError(f"sub 는 G·C·E·judge: {sub!r}")',
     '    if sub == "C":\n        raise SystemExit("⑦-T(x7-T, 시계열만 XGBoost)는 제외(담당자 결정 2026-11-10) — sub \'C\' 는 돌리지 않음")\n'
     '    if sub not in ("G", "E"):\n        raise cf.ConfigError(f"sub 는 G·E·judge: {sub!r}")'),
]
for old, new in EDITS:
    n = s.count(old)
    if n != 1:
        sys.exit(f"바꿀 곳이 {n}번 있음(1번이어야 함): {old[:60]!r}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print(f"고침: {p}")
