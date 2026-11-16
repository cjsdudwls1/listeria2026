# -*- coding: utf-8 -*-
"""section xml 의 최상위 문단을 (paraPr, styleID, charPr 목록, 글, 표/그림 여부)로 나열."""
import sys, re
from lxml import etree

NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
      "hs": "http://www.hancom.co.kr/hwpml/2011/section"}
path = sys.argv[1]
out = sys.argv[2]
tree = etree.parse(path)
root = tree.getroot()
HP = "{%s}" % NS["hp"]
lines = []
for i, p in enumerate(root.findall("hp:p", NS)):
    runs = p.findall("hp:run", NS)
    cps = [r.get("charPrIDRef") for r in runs]
    texts = []
    for r in runs:
        for t in r.findall("hp:t", NS):
            texts.append("".join(t.itertext()))
    txt = "".join(texts)
    kinds = []
    for r in runs:
        for c in r:
            tag = c.tag.replace(HP, "")
            if tag not in ("t",):
                kinds.append(tag)
    extra = ""
    tbl = p.find(".//hp:tbl", NS)
    if tbl is not None:
        extra = f" TBL rows={tbl.get('rowCnt')} cols={tbl.get('colCnt')} bf={tbl.get('borderFillIDRef')}"
    pic = p.find(".//hp:pic", NS)
    if pic is not None:
        extra += " PIC"
    lines.append(f"{i}\tpara={p.get('paraPrIDRef')} style={p.get('styleIDRef')} pb={p.get('pageBreak')} cp={cps} kinds={kinds}{extra}\t{txt[:80]}")
open(out, "w", encoding="utf-8").write("\n".join(lines))
print(len(lines))
