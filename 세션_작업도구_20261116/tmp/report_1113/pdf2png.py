# -*- coding: utf-8 -*-
"""PDF -> 쪽별 PNG. 사용: py pdf2png.py <pdf> <outdir> [dpi] [first] [last]"""
import os, sys
import pymupdf

pdf, out = sys.argv[1], sys.argv[2]
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 80
os.makedirs(out, exist_ok=True)
doc = pymupdf.open(pdf)
first = int(sys.argv[4]) if len(sys.argv) > 4 else 1
last = int(sys.argv[5]) if len(sys.argv) > 5 else doc.page_count
for i in range(first - 1, min(last, doc.page_count)):
    doc[i].get_pixmap(dpi=dpi).save(os.path.join(out, f"p{i+1:02d}.png"))
print("pages", doc.page_count)
