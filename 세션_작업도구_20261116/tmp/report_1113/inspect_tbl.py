# -*- coding: utf-8 -*-
"""표·그림 구조 요약을 UTF-8 파일로."""
import sys, re
from lxml import etree

NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
      "hc": "http://www.hancom.co.kr/hwpml/2011/core"}
path, out = sys.argv[1], sys.argv[2]
root = etree.parse(path).getroot()
ps = root.findall("hp:p", NS)
L = []
def txt(el):
    return "".join(t.text or "" for t in el.iter("{%s}t" % NS["hp"]))
for i, p in enumerate(ps):
    for tbl in p.iter("{%s}tbl" % NS["hp"]):
        sz = tbl.find("hp:sz", NS)
        cap = tbl.find("hp:caption", NS)
        capinfo = ""
        if cap is not None:
            cp = cap.find(".//hp:p", NS)
            capinfo = f"cap side={cap.get('side')} w={cap.get('width')} gap={cap.get('gap')} para={cp.get('paraPrIDRef')} style={cp.get('styleIDRef')} cps={[r.get('charPrIDRef') for r in cp.findall('hp:run', NS)]} text={txt(cap)[:60]}"
        inm = tbl.find("hp:inMargin", NS)
        L.append(f"[{i}] TBL rows={tbl.get('rowCnt')} cols={tbl.get('colCnt')} bf={tbl.get('borderFillIDRef')} w={sz.get('width')} h={sz.get('height')} pageBreak={tbl.get('pageBreak')} repeatHeader={tbl.get('repeatHeader')} inMargin={dict(inm.attrib) if inm is not None else None}")
        L.append("    " + capinfo)
        trs = tbl.findall("hp:tr", NS)
        for ri, tr in enumerate(trs[:3]):
            for tc in tr.findall("hp:tc", NS):
                cp = tc.find(".//hp:p", NS)
                cm = tc.find("hp:cellMargin", NS)
                csz = tc.find("hp:cellSz", NS)
                span = tc.find("hp:cellSpan", NS)
                sl = tc.find("hp:subList", NS)
                L.append(f"    r{ri} bf={tc.get('borderFillIDRef')} hasMargin={tc.get('hasMargin')} header={tc.get('header')} para={cp.get('paraPrIDRef')} style={cp.get('styleIDRef')} cps={[r.get('charPrIDRef') for r in cp.findall('hp:run', NS)]} va={sl.get('vertAlign')} margin={dict(cm.attrib)} sz={dict(csz.attrib)} span={dict(span.attrib)} t={txt(tc)[:30]}")
    for pic in p.iter("{%s}pic" % NS["hp"]):
        x = etree.tostring(pic, encoding="unicode")
        x = re.sub(r' xmlns:\w+="[^"]+"', '', x)
        L.append(f"[{i}] PIC len={len(x)}\n{x[:6000]}")
open(out, "w", encoding="utf-8").write("\n".join(L))
print(len(L))
