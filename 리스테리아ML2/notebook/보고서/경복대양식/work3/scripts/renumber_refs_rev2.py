# -*- coding: utf-8 -*-
"""rev2(2026-10-22) 원고 글 속 참조 '표[n]'·'그림[n]' 을 옛 번호 -> 새 번호로 한 번에 바꿈(연쇄 치환 없음).
바뀌는 까닭: Ⅰ부 2장 EDA 끝에 표 1개·그림 1개를 새로 넣고(새 표[5]·그림[5]), 8장 결과 표 3개(옛 표[17]·[18]·[19])를 글로 바꿈.
옛 표[17]~[19] 참조는 바꾸지 않고 남은 곳을 알림(그 문장은 손으로 고침). 쓰는 법: py renumber_refs_rev2.py <원고 폴더>"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
SEC = sys.argv[1]
T_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
         20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23}
F_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
RE = re.compile(r"(표|그림)\[(\d+)\]")
for fn in sorted(os.listdir(SEC)):
    if not fn.endswith(".md"):
        continue
    p = os.path.join(SEC, fn)
    with io.open(p, encoding="utf-8") as f:
        s = f.read()
    left, changed = [], [0]

    def sub(m):
        kind, n = m.group(1), int(m.group(2))
        mp = T_MAP if kind == "표" else F_MAP
        if n not in mp:
            left.append("%s[%d]" % (kind, n))
            return m.group(0)
        if mp[n] != n:
            changed[0] += 1
        return "%s[%d]" % (kind, mp[n])

    out = RE.sub(sub, s)
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print("%s: 바꾼 참조 %d곳, 대응 없는 번호 %s" % (fn, changed[0], sorted(set(left)) or "없음"))
