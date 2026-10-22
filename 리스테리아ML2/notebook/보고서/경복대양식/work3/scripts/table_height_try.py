# -*- coding: utf-8 -*-
"""md 파일 안의 표(들)의 조립기 어림 높이(pt, 제목 포함)·열 폭(%)을 찍음 — 표 모양을 바꿔 볼 때 쓰는 도구.
쓰는 법: py table_height_try.py <md 파일>"""
import sys

import estimate_pages as E

B = E.B
for b in B.parse_md(sys.argv[1], B.default_base_dirs):
    if b.kind != "table":
        continue
    tb = b.table
    w = B.col_widths(tb)
    est = (850 + 1920 + 282 + B.table_height([tb.header] + tb.rows, w)) / 100.0
    print("%5.0fpt(본문의 %2.0f%%) 열 %s  %s" % (est, 100 * est / E.H, [round(100.0 * x / B.TABLE["max_width"]) for x in w],
                                          (tb.caption or "")[:40]))
