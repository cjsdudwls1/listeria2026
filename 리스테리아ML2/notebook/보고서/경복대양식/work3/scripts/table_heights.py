# -*- coding: utf-8 -*-
"""원고의 표마다 조립기(build_kb_hwp)와 같은 열 폭·어림 높이를 찍음(본문 높이 대비 %) — 쪽 수 줄일 곳 찾기용, 읽기만.
쓰는 법: py table_heights.py <원고 폴더>"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import build_kb_hwp as B  # noqa: E402

files, _ = B.read_order(sys.argv[1])
n = 0
tot = 0.0
for p in files:
    for blk in B.parse_md(p, B.default_base_dirs):
        if blk.kind != "table":
            continue
        n += 1
        tb = blk.table
        w = B.col_widths(tb)
        h = B.table_height([tb.header] + tb.rows, w)
        tot += h
        print("표 %2d  %5.1f%%  열 %d  폭 %s  %s" % (n, 100.0 * h / B.BODY_H, len(tb.header),
                                              [round(100.0 * x / sum(w), 1) for x in w], (tb.caption or "")[:40]))
print("합 %.2f쪽" % (tot / B.BODY_H))
