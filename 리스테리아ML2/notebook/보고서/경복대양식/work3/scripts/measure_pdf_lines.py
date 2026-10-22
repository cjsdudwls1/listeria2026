# -*- coding: utf-8 -*-
"""조립 PDF 한 쪽의 글 줄 위치·글자 크기·줄 폭(pt)을 찍음 — 쪽 수 어림(estimate_pages.py) 보정용. 읽기 전용.
쓰는 법: py measure_pdf_lines.py <pdf> <쪽 번호(1부터)> [<쪽 번호> ...]"""
import sys

import pymupdf

sys.stdout.reconfigure(encoding="utf-8")
doc = pymupdf.open(sys.argv[1])
for pno in [int(x) for x in sys.argv[2:]]:
    pg = doc[pno - 1]
    print("== p%d" % pno)
    for b in pg.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            sp = [s for s in ln["spans"] if s["text"].strip()]
            if not sp:
                continue
            t = "".join(s["text"] for s in sp)
            x0, y0, x1, y1 = ln["bbox"]
            print("y %6.1f-%6.1f x %5.1f-%5.1f size %4.1f n %3d  %s" % (y0, y1, x0, x1, sp[0]["size"], len(t), t[:60]))
