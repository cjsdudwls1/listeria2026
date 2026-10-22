# -*- coding: utf-8 -*-
"""두 원고 폴더의 표·그림이 같은지 대조(본문 절반 판 작업, 2026-10-22)

쓰는 법: py 09_compare_tables_figs.py <원래 원고 폴더> <새 원고 폴더>
  조립기 parse_md 로 요소를 나눠, 표(머리 행·정렬·본문 행·캡션)와 그림(제목·경로)·쪽 나눔·제목 줄의
  차례와 내용이 같은지 봄. 본문 문단(○·-·•·일반)은 대조하지 않음.
  본문 글에 남은 표·그림 참조 번호('표[n]'·'그림[n]')도 모아 새 폴더 쪽 번호가 실제 표·그림 수 안에 있는지 봄.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, KB)
import build_kb_hwp as kb                             # noqa: E402

BODY = ("b1", "b2", "b3", "plain")


def objects(folder):
    files, _ = kb.read_order(folder)
    out, body = [], []
    for p in files:
        for b in kb.parse_md(p, kb.default_base_dirs):
            if b.kind == "table":
                t = b.table
                out.append(("table", tuple(t.header), tuple(tuple(r) for r in t.rows), tuple(t.aligns), t.caption))
            elif b.kind == "fig":
                out.append(("fig", b.fig["title"], b.fig["rel"]))
            elif b.kind in BODY:
                body.append((os.path.basename(p), b.text))
            else:
                out.append((b.kind, getattr(b, "text", None)))
    return out, body


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = sys.argv[1], sys.argv[2]
    oa, _ = objects(a)
    ob, body_b = objects(b)
    print("요소(표·그림·제목·쪽 나눔) 수: %d / %d" % (len(oa), len(ob)))
    same = oa == ob
    print("표·그림·제목·쪽 나눔 차례와 내용:", "같음" if same else "다름")
    if not same:
        for i, (x, y) in enumerate(zip(oa, ob)):
            if x != y:
                print("  첫 차이 #%d\n    원래: %r\n    새것: %r" % (i, x[:2], y[:2]))
                break
    n_tbl = sum(1 for x in ob if x[0] == "table")
    n_fig = sum(1 for x in ob if x[0] == "fig")
    print("표 %d개, 그림 %d개" % (n_tbl, n_fig))
    refs_t, refs_f = set(), set()
    for f, t in body_b:
        refs_t.update(int(x) for x in re.findall(r"표\[(\d+)\]", t))
        refs_f.update(int(x) for x in re.findall(r"그림\[(\d+)\]", t))
    print("본문이 부르는 표 번호:", sorted(refs_t), "범위 밖:", sorted(x for x in refs_t if not 1 <= x <= n_tbl))
    print("본문이 부르는 그림 번호:", sorted(refs_f), "범위 밖:", sorted(x for x in refs_f if not 1 <= x <= n_fig))
    print("본문이 부르지 않는 표:", sorted(set(range(1, n_tbl + 1)) - refs_t))
    print("본문이 부르지 않는 그림:", sorted(set(range(1, n_fig + 1)) - refs_f))


if __name__ == "__main__":
    main()
