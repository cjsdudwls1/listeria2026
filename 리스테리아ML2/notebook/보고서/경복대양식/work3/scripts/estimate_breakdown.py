# -*- coding: utf-8 -*-
"""원고 요소별 어림 높이를 장('##')·절('###') 단위로 합쳐 찍음(쪽 나눔 손실 빼고 순수 높이) — 어디를 줄일지 정하는 용도.
쓰는 법: py estimate_breakdown.py <원고 폴더> [--detail]"""
import math
import sys

import estimate_pages as E

B = E.B
CAP = 37193


def block_h(b, counters):
    if b.kind in ("b1", "b2", "b3", "plain"):
        n = max(1, math.ceil(B.text_width(b.text) / (CAP - (600 if b.kind in ("b2", "b3") else 0))))
        return n * E.PITCH + E.GAP_BEFORE[b.kind], "text"
    if b.kind == "table":
        tb = b.table
        w = B.col_widths(tb)
        return (850 + 1920 + 282 + B.table_height([tb.header] + tb.rows, w)) / 100.0 + E.OBJ_GAP, "table"
    if b.kind == "fig":
        w, h = E.fig_size_pt(b)
        return h + E.OBJ_GAP + 8 + E.PITCH, "fig"
    if b.kind in E.HEAD_H:
        return E.HEAD_H[b.kind] + E.HEAD_GAP, "head"
    return 0.0, "other"


def main():
    folder = sys.argv[1]
    detail = "--detail" in sys.argv
    bl = E.blocks_of(folder)
    cur_h1, cur_h2 = "(앞)", ""
    agg = {}
    order = []
    total = 0.0
    for b in bl:
        if b.kind in ("part", "h1"):
            cur_h1, cur_h2 = b.text, ""
        elif b.kind == "h2":
            cur_h2 = b.text
        key = (cur_h1, cur_h2)
        if key not in agg:
            agg[key] = {"text": 0.0, "table": 0.0, "fig": 0.0, "head": 0.0, "other": 0.0}
            order.append(key)
        h, kind = block_h(b, None)
        agg[key][kind] += h
        total += h
        if detail and kind in ("table", "fig"):
            print("   %-5s %4.0fpt  %s" % (kind, h, (b.table.caption if b.kind == "table" else b.fig["title"])[:40]))
    for key in order:
        a = agg[key]
        s = sum(a.values())
        print("%5.2f쪽  글 %4.0f 표 %4.0f 그림 %4.0f  %s / %s" % (s / E.H, a["text"], a["table"], a["fig"], key[0][:22], key[1][:26]))
    print("합 %.2f쪽(쪽 나눔 손실 빼고)" % (total / E.H))


if __name__ == "__main__":
    main()
