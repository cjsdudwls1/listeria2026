# -*- coding: utf-8 -*-
"""section xml 의 n 번째 최상위 문단 XML 을 파일로."""
import sys
from lxml import etree

NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}
path, idxs, out = sys.argv[1], sys.argv[2], sys.argv[3]
root = etree.parse(path).getroot()
ps = root.findall("hp:p", NS)
chunks = []
for s in idxs.split(","):
    i = int(s)
    x = etree.tostring(ps[i], encoding="unicode")
    chunks.append(f"===== {i} ({len(x)} chars)\n{x}\n")
open(out, "w", encoding="utf-8").write("\n".join(chunks))
