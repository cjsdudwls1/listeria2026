# -*- coding: utf-8 -*-
"""원고(sections 폴더)의 조립 쪽 수 어림 — 한글(COM) 없이 쪽 배치를 흉내 냄(계획·점검용, 실제 쪽 수는 조립 PDF 로 확인).

조립기(build_kb_hwp.py)를 모듈로 불러 같은 원고 해석(parse_md)·표 열 폭·표 어림 높이(col_widths·table_height)를 쓰고,
글 문단 줄 수는 본문 한 줄에 드는 글 폭(이전 조립 PDF 의 꽉 찬 줄에서 잼)으로 셈.
쪽 배치: 문단은 줄 단위로 쪽을 넘김, 표는 어림 높이가 본문의 60% 넘으면 쪽 경계에서 나누고(머리 행 반복) 아니면 통째로,
그림은 통째로. 통째로 넘어가 앞 쪽이 본문의 40%(--gap) 넘게 비면 표는 나누고 그림은 75%(--fig-min)까지 줄임(조립기와 같은 규칙).
쓰는 법: py estimate_pages.py <원고 폴더> [--calib-pdf <이전 조립 pdf>] [--gap 0.4] [--fig-min 0.75]
"""
import argparse
import math
import os
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, KB)
import build_kb_hwp as B  # noqa: E402

H = 714.5           # 본문 높이(pt) = BODY_BOTTOM_PT - BODY_TOP_PT
PITCH = 21.6        # 12pt 본문 줄 간격(pt, 조립 PDF 실측)
GAP_BEFORE = {"b1": 12.3, "plain": 12.3, "b2": 6.0, "b3": 6.0}
HEAD_H = {"part": 46.0, "h1": 36.0, "h2": 36.0, "box": 34.0}
HEAD_GAP = 13.0     # 제목 앞 추가 간격(쪽 맨 위면 없음)
OBJ_GAP = 22.0      # 표·그림 앞뒤 간격 합(pt)


def full_line_cap(pdf):
    """이전 조립 PDF 의 12pt 본문 줄 가운데 오른쪽 끝까지 찬 줄의 글 폭(조립기 text_width 단위) 중앙값 -> 한 줄 폭"""
    import pymupdf
    doc = pymupdf.open(pdf)
    ws = []
    for pg in doc:
        for b in pg.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                sp = [s for s in ln["spans"] if s["text"].strip()]
                if not sp or abs(sp[0]["size"] - 12.0) > 0.2:
                    continue
                x0, y0, x1, y1 = ln["bbox"]
                if x1 < 520 or x0 > 80:
                    continue
                t = "".join(s["text"] for s in ln["spans"]).replace("¡", "").strip()
                if t.startswith("[표") or t.startswith("[그림"):
                    continue
                ws.append((B.text_width(t), x0))
    cont = [w for w, x0 in ws if x0 > 60]
    return statistics.median(cont) if cont else 38000.0


def fig_size_pt(blk):
    data = open(blk.fig["path"], "rb").read()
    pw, ph, dpi = B.png_info(data, blk.fig["path"])
    P = B.PICTURE
    w = min(int(round(pw * 7200.0 / (dpi or P["default_dpi"]))), P["max_w"])
    h = int(round(w * ph / float(pw)))
    if h > P["max_h"]:
        h = P["max_h"]
        w = int(round(h * pw / float(ph)))
    return w / 100.0, h / 100.0


def blocks_of(sections):
    files, _ = B.read_order(sections)
    out = []
    for p in files:
        for blk in B.parse_md(p, B.default_base_dirs):
            blk.src_file = os.path.basename(p)
            out.append(blk)
    return out


def simulate(blocks, cap, gap_ratio, fig_min, verbose=False):
    page, used = 1, H           # 1쪽 = 표지, 본문은 2쪽부터
    heads, notes = [], []
    pending_head = 0.0
    n_tbl = n_fig = 0

    def new_page():
        nonlocal page, used
        page += 1
        used = 0.0

    for bi, blk in enumerate(blocks):
        k = blk.kind
        if k == "pagebreak":
            new_page()
            continue
        if k == "part":
            new_page()
            used += HEAD_H["part"]
            heads.append((page, "", blk.text))
            continue
        if k in ("h1", "h2", "box"):
            h = HEAD_H[k] + (HEAD_GAP if used > 0 else 0)
            # 다음 문단과 함께(keepWithNext): 제목 + 다음 요소 첫 두 줄이 안 들어가면 새 쪽
            if used + h + 2 * PITCH > H:
                new_page()
                h = HEAD_H[k]
            used += h
            heads.append((page, k, blk.text))
            continue
        if k in ("b1", "b2", "b3", "plain"):
            cap_k = cap - (600 if k in ("b2", "b3") else 0)
            n = max(1, math.ceil(B.text_width(blk.text) / cap_k))
            g = GAP_BEFORE[k] if used > 0 else 0.0
            for i in range(n):
                need = PITCH + (g if i == 0 else 0)
                if used + need > H:
                    new_page()
                    need = PITCH
                used += need
            continue
        if k == "table":
            n_tbl += 1
            tb = blk.table
            widths = B.col_widths(tb)
            rows = [tb.header] + tb.rows
            row_h = [(B.ROW_PAD + B.LINE_H * max(B.cell_lines(t, widths[c], ri == 0) for c, t in enumerate(row))) / 100.0
                     for ri, row in enumerate(rows)]
            cap_h = (850 + 1920 + 2 * B.TABLE["out_margin"]) / 100.0
            cap_lines = max(1, math.ceil(B.text_width("[표 %d] %s" % (n_tbl, tb.caption or "")) * 1.2 / (sum(widths) * 0.95)))
            cap_h += (cap_lines - 1) * PITCH
            est = sum(row_h) + cap_h
            split = est > B.TABLE["split_ratio"] * H * 1.0
            total = est + OBJ_GAP
            if used + total <= H:
                used += total
                continue
            remaining = H - used
            if not split and not (remaining / H > gap_ratio and len(tb.rows) >= 3):
                notes.append("표 %d 다음 쪽으로(앞 쪽 %.0f%% 빔)" % (n_tbl, 100 * remaining / H))
                new_page()
                used = total
                continue
            # 쪽 경계에서 나눔: 머리 행 반복
            used += OBJ_GAP / 2
            ri = 0
            first = True
            while ri < len(rows):
                if first:
                    if used + row_h[0] + row_h[min(1, len(rows) - 1)] > H:
                        new_page()
                    used += row_h[0]
                    ri = 1
                    first = False
                    continue
                if used + row_h[ri] > H:
                    new_page()
                    used += row_h[0]
                used += row_h[ri]
                ri += 1
            if used + cap_h + OBJ_GAP / 2 > H:
                new_page()
            used += cap_h + OBJ_GAP / 2
            continue
        if k == "fig":
            n_fig += 1
            w, h = fig_size_pt(blk)
            cap_lines = max(1, math.ceil(B.text_width("[그림 %d] %s" % (n_fig, blk.fig["title"])) * 1.2 / (w * 100 * 0.95)))
            total = h + OBJ_GAP + 8 + cap_lines * PITCH
            if used + total <= H:
                used += total
                continue
            remaining = H - used
            if remaining / H > gap_ratio:
                s = (remaining - OBJ_GAP - 8 - cap_lines * PITCH) / h
                if s >= fig_min:
                    notes.append("그림 %d %.0f%% 로 줄여 앞 쪽에" % (n_fig, 100 * s))
                    used = H
                    continue
            notes.append("그림 %d 다음 쪽으로(앞 쪽 %.0f%% 빔)" % (n_fig, 100 * remaining / H))
            new_page()
            used = total
            continue
    return page, heads, notes, n_tbl, n_fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections")
    ap.add_argument("--calib-pdf", default=os.path.join(KB, "out3", "리스테리아_예측모델_개발_연구보고서.pdf"))
    ap.add_argument("--cap", type=float, default=0, help="본문 한 줄 글 폭(조립기 text_width 단위), 0 = PDF 에서 잼")
    ap.add_argument("--gap", type=float, default=0.40)
    ap.add_argument("--fig-min", type=float, default=0.75)
    a = ap.parse_args()
    cap = a.cap or full_line_cap(a.calib_pdf)
    blocks = blocks_of(a.sections)
    pages, heads, notes, nt, nf = simulate(blocks, cap, a.gap, a.fig_min)
    print("한 줄 글 폭 %.0f, 표 %d, 그림 %d, 어림 쪽 수 %d(표지 포함)" % (cap, nt, nf, pages))
    for p, k, t in heads:
        print("  p%-3d %s%s" % (p, {"": "", "h1": "  ", "h2": "    ", "box": "      "}.get(k, ""), t[:46]))
    for n in notes:
        print("  -", n)


if __name__ == "__main__":
    main()
