# -*- coding: utf-8 -*-
"""원고의 표·그림 번호(조립 순서대로 매겨지는 번호)와 글 속 참조 '표[n]'·'그림[n]' 을 나란히 찍음 — 번호 고치기 점검용.
쓰는 법: py check_refs.py <원고 폴더>"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import build_kb_hwp as B  # noqa: E402

files, _ = B.read_order(sys.argv[1])
nt = nf = 0
RE = re.compile(r"(Ⅰ부\s*|부록\s*)?(표|그림)\[(\d+)\]")
for p in files:
    name = os.path.basename(p)
    for blk in B.parse_md(p, B.default_base_dirs):
        if blk.kind == "table":
            if blk.table.caption is not None:
                nt += 1
                print("  [표 %d] %s  (%s:%d)" % (nt, (blk.table.caption or "")[:50], name[:6], blk.src[1]))
            texts = [blk.table.caption or ""] + [c for row in [blk.table.header] + blk.table.rows for c in row]
        elif blk.kind == "fig":
            nf += 1
            print("  [그림 %d] %s  (%s:%d)" % (nf, blk.fig["title"][:50], name[:6], blk.src[1]))
            texts = [blk.fig["title"]]
        else:
            texts = [blk.text or ""]
        for t in texts:
            for m in RE.finditer(t):
                i = m.start()
                print("      참조 %s%s[%s]  (%s:%d) …%s…" % (m.group(1) or "", m.group(2), m.group(3), name[:6], blk.src[1],
                                                    t[max(0, i - 18):i + 12]))
print("표 %d개, 그림 %d개" % (nt, nf))
