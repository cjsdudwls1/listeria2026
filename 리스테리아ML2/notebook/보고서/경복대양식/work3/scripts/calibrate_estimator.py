# -*- coding: utf-8 -*-
"""쪽 수 어림(estimate_pages.py)의 간격 상수를 이전 조립(45쪽, 2026-10-22 09:05 판, 원고 = work3/rev1_전/sections3)의
실제 제목 쪽 번호에 맞춰 고름(격자 탐색) — 맞춘 상수로 새 원고 쪽 수를 어림. 읽기 전용.
쓰는 법: py calibrate_estimator.py <새 원고 폴더>"""
import itertools
import os
import sys

import estimate_pages as E

sys.stdout.reconfigure(encoding="utf-8")
W3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(W3, "rev1_전", "sections3")
# 이전 조립 로그(work3/build_logs/03_build_figdpi.log) '== 제목 쪽' 의 부문·대제목 쪽(차례대로) + 전체 45쪽
ACTUAL_H1 = [2, 2, 6, 10, 12, 12, 12, 15, 18, 31, 33, 34, 37, 40, 42]
ACTUAL_PAGES = 45


def run(blocks, og, hg, gb1, pitch, gap):
    E.OBJ_GAP, E.HEAD_GAP, E.PITCH = og, hg, pitch
    E.GAP_BEFORE.update({"b1": gb1, "plain": gb1})
    pages, heads, notes, nt, nf = E.simulate(blocks, 37193, gap, 0.75)
    return pages, [h[0] for h in heads if h[1] in ("", "h1")]


old = E.blocks_of(OLD)
best = []
for og, hg, gb1, pitch in itertools.product((8, 14, 22), (0, 6, 13), (8.0, 12.3), (21.0, 21.6)):
    pages, hp = run(old, og, hg, gb1, pitch, 0.40)
    err = sum(abs(a - b) for a, b in zip(hp, ACTUAL_H1)) + 2 * abs(pages - ACTUAL_PAGES)
    best.append((err, pages, og, hg, gb1, pitch, hp))
best.sort(key=lambda x: x[0])
for b in best[:5]:
    print("오차 %d 쪽 %d  OBJ_GAP %s HEAD_GAP %s b1 간격 %s 줄 %s  %s" % b)
err, pages, og, hg, gb1, pitch, hp = best[0]
new = E.blocks_of(sys.argv[1])
for gap in (0.40, 0.25):
    p, hp2 = run(new, og, hg, gb1, pitch, gap)
    print("새 원고(gap %.2f): 어림 %d쪽, 대제목 쪽 %s" % (gap, p, hp2))
