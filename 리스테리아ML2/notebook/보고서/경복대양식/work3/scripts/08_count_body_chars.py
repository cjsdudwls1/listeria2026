# -*- coding: utf-8 -*-
"""본문 글자 수 세기(본문 절반 판 작업, 2026-10-22)

쓰는 법: py 08_count_body_chars.py <원고 폴더> [<원고 폴더2> ...]
  조립기(build_kb_hwp.py)의 parse_md 로 원고를 요소로 나눈 뒤
  본문 글 = ○(b1)·-(b2)·•(b3)·일반 문단(plain) 글만 셈(표 칸·표 제목·그림 제목·제목 줄(#·##·###·□)·주석 제외).
  공백 제외 글자 수와 공백 포함 글자 수를 원고 파일별·합계로 찍음. 제목 줄·표·캡션 글자 수도 참고로 찍음.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(os.path.dirname(HERE))          # 경복대양식/
sys.path.insert(0, KB)
import build_kb_hwp as kb                             # noqa: E402

BODY = ("b1", "b2", "b3", "plain")
HEAD = ("part", "h1", "h2", "box")


def nospace(s):
    return len("".join(s.split()))


def count(folder):
    files, _ = kb.read_order(folder)
    rows = []
    tot = dict(body=0, body_sp=0, n_body=0, head=0, cap=0, cell=0)
    for p in files:
        bl = kb.parse_md(p, kb.default_base_dirs)
        r = dict(file=os.path.basename(p), body=0, body_sp=0, n_body=0, head=0, cap=0, cell=0)
        for b in bl:
            if b.kind in BODY:
                r["body"] += nospace(b.text)
                r["body_sp"] += len(b.text)
                r["n_body"] += 1
            elif b.kind in HEAD:
                r["head"] += nospace(b.text)
            elif b.kind == "table":
                t = b.table
                r["cap"] += nospace(t.caption or "")
                r["cell"] += sum(nospace(c) for c in t.header) + sum(nospace(c) for row in t.rows for c in row)
            elif b.kind == "fig":
                r["cap"] += nospace(b.fig["title"])
        for k in tot:
            tot[k] += r[k]
        rows.append(r)
    return rows, tot


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for folder in sys.argv[1:]:
        rows, tot = count(folder)
        print("== %s" % folder)
        print("  %-40s %8s %8s %6s | %6s %6s %7s" % ("파일", "본문(공백X)", "본문(공백O)", "문단", "제목", "캡션", "표 칸"))
        for r in rows:
            print("  %-40s %8d %8d %6d | %6d %6d %7d" % (r["file"], r["body"], r["body_sp"], r["n_body"],
                                                       r["head"], r["cap"], r["cell"]))
        print("  %-40s %8d %8d %6d | %6d %6d %7d" % ("합계", tot["body"], tot["body_sp"], tot["n_body"],
                                                   tot["head"], tot["cap"], tot["cell"]))


if __name__ == "__main__":
    main()
