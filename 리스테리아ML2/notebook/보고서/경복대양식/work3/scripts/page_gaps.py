# -*- coding: utf-8 -*-
"""조립 PDF 쪽마다 본문 아래 빈 높이(pt·mm·비율)와 그림 위치 — 배치 점검용(build_kb_hwp.py 와 같은 좌표 기준)"""
import sys
import pymupdf
sys.stdout.reconfigure(encoding="utf-8")
PDF_SHIFT = 14.45
TOP = (2834 + 4251) / 100.0 - PDF_SHIFT
BOT = 842 - (1417 + 4251) / 100.0 - PDF_SHIFT
FOOT_Y = 795.0
H = BOT - TOP
doc = pymupdf.open(sys.argv[1])
for i, pg in enumerate(doc):
    ys = [b[3] for b in pg.get_text("blocks") if b[1] < FOOT_Y]
    ys += [x["bbox"][3] for x in pg.get_image_info() if x["bbox"][1] < FOOT_Y]
    ys += [d["rect"].y1 for d in pg.get_drawings() if d["rect"].y0 < FOOT_Y]
    bot = max(ys) if ys else TOP
    blank = BOT - bot
    imgs = ["%.0fx%.0fmm@y%.0f-%.0f" % ((x["bbox"][2] - x["bbox"][0]) / 72 * 25.4, (x["bbox"][3] - x["bbox"][1]) / 72 * 25.4,
                                       x["bbox"][1], x["bbox"][3]) for x in pg.get_image_info()]
    print("p%02d 아래 빈 %5.1fpt(%5.1fmm, %4.1f%%) %s" % (i + 1, blank, blank / 72 * 25.4, 100 * blank / H, " ".join(imgs)))
