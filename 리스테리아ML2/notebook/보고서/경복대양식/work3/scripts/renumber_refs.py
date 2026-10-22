# -*- coding: utf-8 -*-
"""원고 글 속 표 참조 번호 '표[n]' 를 파일마다 주어진 대응(옛 번호 -> 새 번호)으로 한 번에 바꿈(연쇄 치환 없음).
그림 참조는 바꾸지 않음. 바꾼 곳을 찍고, 대응에 없는 번호가 남으면 알림.
쓰는 법: py renumber_refs.py  (대응은 아래 MAP — 2026-10-22 rev1 원고 구조 기준)"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
SEC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sections3")
MAP = {
    "10_II1-3_서론_설계_베이스라인.md": {8: 6, 11: 7, 27: 23, 5: 5},
    "14_II4_특성공학.md": {26: 22, 12: 8, 13: 9, 14: 10, 15: 11, 29: 25, 16: 12},
    "15_II5-9_튜닝_선택_결과_소결.md": {17: 13, 18: 14, 19: 15, 28: 24, 20: 16, 21: 17, 22: 18, 23: 19, 24: 20, 26: 22},
    "30_부록_입력변수기록표.md": {26: 22, 27: 23, 28: 24, 29: 25},
}
RE = re.compile(r"표\[(\d+)\]")
for fn, mp in MAP.items():
    p = os.path.join(SEC, fn)
    with io.open(p, encoding="utf-8") as f:
        s = f.read()
    left = []

    def sub(m):
        n = int(m.group(1))
        if n not in mp:
            left.append(n)
            return m.group(0)
        return "표[%d]" % mp[n]

    out = RE.sub(sub, s)
    changed = sum(1 for a, b in zip(RE.findall(s), RE.findall(out)) if a != b)
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print("%s: 바꾼 참조 %d곳, 대응 없는 번호 %s" % (fn, changed, sorted(set(left)) or "없음"))
