# -*- coding: utf-8 -*-
"""식품안전지수 PPT·엑셀에서 기관·과제명·지표 정의 글자 찾기(읽기만)."""
import re, sys, zipfile
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
D = Path(r"C:\Users\NT751\Desktop\천영진\문서\영진문서")
ppt = D / "식품안전지수 자료 찾기_지수산정 길라잡이_2018.11.14.pptx"
xls = D / "식품안전지수_protocol_20190103.xlsx"
KW = re.compile(r"정보원|과제|연구|식약처|식품의약품|용역|기관|센터|대학|조상구|지수|Index|index|발주|사업")
def xml_text(b):
    s = b.decode("utf-8", "ignore")
    return re.sub(r"<[^>]+>", " | ", s)
with zipfile.ZipFile(ppt) as z:
    for n in z.namelist():
        if n.startswith("docProps/"):
            print("==", n); print(xml_text(z.read(n))[:1500])
    # notes & all slide texts with keywords
    for n in sorted(z.namelist()):
        if (n.startswith("ppt/notesSlides/") or n.startswith("ppt/slides/") or n.startswith("ppt/slideMasters/") or n.startswith("ppt/slideLayouts/")) and n.endswith(".xml"):
            t = re.sub(r"\s*\|\s*(\|\s*)+", " | ", xml_text(z.read(n)))
            t = re.sub(r"<[^>]+>", "", t)
            words = re.findall(r"<a:t>([^<]*)</a:t>", z.read(n).decode("utf-8","ignore"))
            txt = "".join(words)
            if n.startswith("ppt/notesSlides/") and txt.strip():
                print("== NOTE", n, txt[:600])
            elif KW.search(txt) and not n.startswith("ppt/slides/slide"):
                print("== KW", n, txt[:400])
print("\n===== XLSX")
with zipfile.ZipFile(xls) as z:
    for n in z.namelist():
        if n.startswith("docProps/"):
            print("==", n); print(xml_text(z.read(n))[:1500])
    if "xl/sharedStrings.xml" in z.namelist():
        ss = re.findall(r"<t[^>]*>([^<]*)</t>", z.read("xl/sharedStrings.xml").decode("utf-8","ignore"))
        print("shared strings", len(ss))
        for s in ss:
            if KW.search(s):
                print("  KW:", s[:200])
    wb = z.read("xl/workbook.xml").decode("utf-8","ignore")
    print("sheets:", re.findall(r'<sheet [^>]*name="([^"]+)"', wb))
    for n in z.namelist():
        if "comments" in n or "threadedComments" in n:
            print("== comments file", n, xml_text(z.read(n))[:800])
