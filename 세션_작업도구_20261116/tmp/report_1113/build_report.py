# -*- coding: utf-8 -*-
"""양식 hwpx(사본)의 header·표지(section0)는 그대로 두고 본문(section1)만 content.BLOCKS 로 새로 만들어 HWP·PDF 로 저장.

사용: py build_report.py            (조립 + 한글 변환 + PDF 쪽 PNG)
      py build_report.py --no-hwp   (hwpx 조립만)
"""
import html
import os
import re
import shutil
import stat
import sys
import zipfile

from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
TPLX = os.path.join(W, "tpl", "x")
BUILD = os.path.join(W, "build")
OUT = os.path.join(W, "out")
sys.path.insert(0, W)
import content  # noqa: E402

NAME = "리스테리아_예측모델_개발_연구보고서_20261113"

# ------------------------------------------------------------------ 양식 스타일 ID(양식 본문에서 확인한 값)
STY = {
    "part": (32, 0, 29),   # 부 제목(I. / II.)
    "h1": (47, 2, 13),     # 1. 제목
    "h2": (47, 3, 13),     # 1.1 제목
    "h3": (48, 6, 7),      # □ 제목
    "o": (49, 7, 1),       # ○ 본문
    "d": (50, 8, 5),       # - 본문
    "b": (52, 9, 1),       # • 본문
}
BOLD_CP = 50               # header 에 새로 넣는 굵은 본문 글자(charPr 1 + 굵게)
CELL_HDR_CP, CELL_CP = 48, 23
CELL_PP = {"L": 42, "C": 2, "R": 3}
TBL_W = 47411
PAGE_TEXT_W = 48360


def esc(s):
    return html.escape(s, quote=False)


def runs(text, cp, bold_cp):
    parts = text.split("**")
    out = []
    for i, part in enumerate(parts):
        if part == "":
            continue
        c = bold_cp if i % 2 == 1 else cp
        out.append(f'<hp:run charPrIDRef="{c}"><hp:t>{esc(part)}</hp:t></hp:run>')
    return "".join(out) if out else f'<hp:run charPrIDRef="{cp}"/>'


def para(text, ppr, style, cp, pb=0, bold_cp=BOLD_CP):
    return (f'<hp:p id="0" paraPrIDRef="{ppr}" styleIDRef="{style}" pageBreak="{pb}" columnBreak="0" merged="0">'
            f'{runs(text, cp, bold_cp)}</hp:p>')


SUBLIST = ('<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="{va}" linkListIDRef="0" '
           'linkListNextIDRef="0" textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">')


def caption(kind, num, text, width):
    nt = "TABLE" if kind == "표" else "PICTURE"
    return (f'<hp:caption side="BOTTOM" fullSz="0" width="8503" gap="850" lastWidth="{width}">'
            + SUBLIST.format(va="TOP")
            + f'<hp:p id="0" paraPrIDRef="2" styleIDRef="10" pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="8"><hp:t>[{kind} </hp:t><hp:ctrl><hp:autoNum num="{num}" numType="{nt}">'
            f'<hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" suffixChar="" supscript="0"/></hp:autoNum>'
            f'</hp:ctrl><hp:t>] {esc(text)}</hp:t></hp:run></hp:p></hp:subList></hp:caption>')


def est_lines(text, width_hu, pt=10):
    """칸 안 글 줄 수 어림(한글 1자 ≈ 글자 크기, 영문·숫자 ≈ 0.55)."""
    usable = max(1000, width_hu - 1800)
    per_char = pt * 100
    lines = 0
    for seg in text.replace("**", "").split("\n"):
        w = sum(per_char * (1.0 if ord(ch) > 0x2E80 else 0.55) for ch in seg)
        lines += max(1, int(w // usable) + 1)
    return lines


def table_xml(spec, num, tid, z):
    header, rows, widths, aligns = spec["header"], spec["rows"], spec["widths"], spec["aligns"]
    assert sum(widths) == TBL_W, (spec["key"], sum(widths))
    ncol = len(header)
    allrows = [header] + rows
    nrow = len(allrows)
    # 행 병합: None = 위 칸과 병합
    span = [[1] * ncol for _ in range(nrow)]
    covered = [[False] * ncol for _ in range(nrow)]
    for r in range(1, nrow):
        for c in range(ncol):
            if allrows[r][c] is None:
                rr = r - 1
                while covered[rr][c]:
                    rr -= 1
                span[rr][c] += 1
                covered[r][c] = True
    row_h = []
    for r, row in enumerate(allrows):
        h = 0
        for c in range(ncol):
            if row[c] is None:
                continue
            h = max(h, est_lines(str(row[c]), widths[c]) * 1600 + 1200)
        row_h.append(max(h, 2800))
    trs = []
    for r, row in enumerate(allrows):
        tcs = []
        for c in range(ncol):
            if covered[r][c]:
                continue
            txt = "" if row[c] is None else str(row[c])
            hdr = r == 0
            cp = CELL_HDR_CP if hdr else CELL_CP
            pp = 2 if hdr else CELL_PP[aligns[c]]
            ps = "".join(
                f'<hp:p id="0" paraPrIDRef="{pp}" styleIDRef="11" pageBreak="0" columnBreak="0" merged="0">'
                f'{runs(line, cp, CELL_HDR_CP)}</hp:p>' for line in txt.split("\n"))
            h = sum(row_h[r:r + span[r][c]])
            tcs.append(
                f'<hp:tc name="" header="{1 if hdr else 0}" hasMargin="1" protect="0" editable="0" dirty="0" '
                f'borderFillIDRef="7">' + SUBLIST.format(va="CENTER") + ps + '</hp:subList>'
                f'<hp:cellAddr colAddr="{c}" rowAddr="{r}"/><hp:cellSpan colSpan="1" rowSpan="{span[r][c]}"/>'
                f'<hp:cellSz width="{widths[c]}" height="{h}"/>'
                f'<hp:cellMargin left="900" right="900" top="600" bottom="600"/></hp:tc>')
        trs.append("<hp:tr>" + "".join(tcs) + "</hp:tr>")
    total_h = sum(row_h)
    pb = spec.get("pb", "CELL")
    return (f'<hp:tbl id="{tid}" zOrder="{z}" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" '
            f'lock="0" dropcapstyle="None" pageBreak="{pb}" repeatHeader="1" rowCnt="{nrow}" colCnt="{ncol}" '
            f'cellSpacing="0" borderFillIDRef="4" noAdjust="0">'
            f'<hp:sz width="{TBL_W}" widthRelTo="ABSOLUTE" height="{total_h}" heightRelTo="ABSOLUTE" protect="0"/>'
            + (f'<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
               f'vertRelTo="PARA" horzRelTo="PARA" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
               if spec.get("inline", True) else
               f'<hp:pos treatAsChar="0" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
               f'vertRelTo="PARA" horzRelTo="COLUMN" vertAlign="TOP" horzAlign="CENTER" vertOffset="0" horzOffset="0"/>')
            + f'<hp:outMargin left="141" right="141" top="141" bottom="141"/>'
            + caption("표", num, spec["caption"], TBL_W)
            + '<hp:inMargin left="510" right="510" top="141" bottom="141"/>' + "".join(trs) + '</hp:tbl>')


def pic_xml(spec, num, bin_id, pid, inst, z):
    im = Image.open(spec["png"])
    pw, ph = im.size
    ow, oh = pw * 75, ph * 75                      # 96 dpi 기준 HWPUNIT
    cw = spec["width"]
    ch = round(cw * ph / pw)
    sx = cw / ow
    sy = ch / oh
    return (f'<hp:pic id="{pid}" zOrder="{z}" numberingType="PICTURE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" '
            f'lock="0" dropcapstyle="None" href="" groupLevel="0" instid="{inst}" reverse="0">'
            f'<hp:offset x="0" y="0"/><hp:orgSz width="{ow}" height="{oh}"/><hp:curSz width="{cw}" height="{ch}"/>'
            f'<hp:flip horizontal="0" vertical="0"/>'
            f'<hp:rotationInfo angle="0" centerX="{cw // 2}" centerY="{ch // 2}" rotateimage="1"/>'
            f'<hp:renderingInfo><hc:transMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
            f'<hc:scaMatrix e1="{sx:.6f}" e2="0" e3="0" e4="0" e5="{sy:.6f}" e6="0"/>'
            f'<hc:rotMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/></hp:renderingInfo>'
            f'<hp:imgRect><hc:pt0 x="0" y="0"/><hc:pt1 x="{ow}" y="0"/><hc:pt2 x="{ow}" y="{oh}"/>'
            f'<hc:pt3 x="0" y="{oh}"/></hp:imgRect><hp:imgClip left="0" right="{ow}" top="0" bottom="{oh}"/>'
            f'<hp:inMargin left="0" right="0" top="0" bottom="0"/><hp:imgDim dimwidth="{ow}" dimheight="{oh}"/>'
            f'<hc:img binaryItemIDRef="{bin_id}" bright="0" contrast="0" effect="REAL_PIC" alpha="0"/><hp:effects/>'
            f'<hp:sz width="{cw}" widthRelTo="ABSOLUTE" height="{ch}" heightRelTo="ABSOLUTE" protect="0"/>'
            f'<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
            f'vertRelTo="PARA" horzRelTo="PARA" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
            f'<hp:outMargin left="0" right="0" top="0" bottom="0"/>'
            + caption("그림", num, spec["caption"], cw) + '</hp:pic>')


def resolve_refs(text, nums):
    def rep(m):
        kind, key = m.group(1), m.group(2)
        if (kind, key) not in nums:
            raise KeyError(f"없는 참조 {kind}:{key}")
        return f"{kind}[{nums[(kind, key)]}]"
    return re.sub(r"\{(표|그림):(\w+)\}", rep, text)


def build_section(first_para_xml, root_open):
    nums = {}
    ti = fi = 0
    for kind, spec, _ in content.BLOCKS:
        if kind == "table":
            ti += 1
            nums[("표", spec["key"])] = ti
        elif kind == "figure":
            fi += 1
            nums[("그림", spec["key"])] = fi
    body = []
    images = []
    tnum = fnum = 0
    z = 0
    for kind, item, kw in content.BLOCKS:
        if kind in STY:
            text = resolve_refs(item, nums)
            if kw.get("first"):
                assert ">I. 데이터 전처리 부문<" in first_para_xml
                body.append(first_para_xml.replace(">I. 데이터 전처리 부문<", f">{esc(text)}<"))
                continue
            ppr, st, cp = STY[kind]
            body.append(para(text, ppr, st, cp, pb=1 if kw.get("pagebreak") else 0))
        elif kind == "table":
            tnum += 1
            z += 1
            spec = dict(item)
            spec["caption"] = resolve_refs(spec["caption"], nums)
            spec["rows"] = [[None if c is None else resolve_refs(str(c), nums) for c in row] for row in spec["rows"]]
            tbl = table_xml(spec, tnum, 1900000000 + tnum, z)
            if spec.get("inline", True):
                body.append('<hp:p id="0" paraPrIDRef="2" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
                            f'<hp:run charPrIDRef="5">{tbl}<hp:t/></hp:run></hp:p>')
            else:
                body.append('<hp:p id="0" paraPrIDRef="46" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
                            f'<hp:run charPrIDRef="49">{tbl}<hp:t/></hp:run></hp:p>')
                body.append('<hp:p id="0" paraPrIDRef="46" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
                            '<hp:run charPrIDRef="49"/></hp:p>')
        elif kind == "figure":
            fnum += 1
            z += 1
            spec = dict(item)
            spec["caption"] = resolve_refs(spec["caption"], nums)
            bin_id = f"image{fnum}"
            images.append((bin_id, spec["png"]))
            pic = pic_xml(spec, fnum, bin_id, 2000000000 + fnum, 926258177 + fnum, z)
            body.append('<hp:p id="0" paraPrIDRef="2" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
                        f'<hp:run charPrIDRef="5">{pic}<hp:t/></hp:run></hp:p>')
        else:
            raise ValueError(kind)
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>' + root_open + "".join(body) + "</hs:sec>"
    return xml, images, nums


def main(do_hwp=True):
    def _rw(func, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(BUILD):
        shutil.rmtree(BUILD, onerror=_rw)
    bx = os.path.join(BUILD, "x")
    shutil.copytree(TPLX, bx)
    for root, _, files in os.walk(bx):
        for f in files:
            os.chmod(os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD)
    os.makedirs(OUT, exist_ok=True)

    # header: 굵은 본문 글자 모양 추가(charPr 1 복사 + 굵게)
    hp = os.path.join(bx, "Contents", "header.xml")
    hx = open(hp, encoding="utf-8").read()
    m = re.search(r'<hh:charPr id="1".*?</hh:charPr>', hx, re.S)
    cp1 = m.group(0)
    assert "<hh:bold/>" not in cp1
    cp50 = cp1.replace('<hh:charPr id="1"', f'<hh:charPr id="{BOLD_CP}"', 1)
    cp50 = re.sub(r"(<hh:offset [^>]*/>)", r"\1<hh:bold/>", cp50, count=1)
    assert "<hh:bold/>" in cp50
    last = list(re.finditer(r"</hh:charPr>", hx))[-1]
    hx = hx[:last.end()] + cp50 + hx[last.end():]
    assert '<hh:charProperties itemCnt="50">' in hx
    hx = hx.replace('<hh:charProperties itemCnt="50">', '<hh:charProperties itemCnt="51">')
    open(hp, "w", encoding="utf-8").write(hx)

    # 본문
    sp = os.path.join(bx, "Contents", "section1.xml")
    sx = open(sp, encoding="utf-8").read()
    root_open = re.search(r"<hs:sec [^>]*>", sx).group(0)
    p0s = sx.index("<hp:p ")
    p0e = sx.index("</hp:p>", p0s) + len("</hp:p>")
    p0 = sx[p0s:p0e]
    assert p0.count("<hp:p ") == 1 and "secPr" in p0
    p0 = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", p0, flags=re.S)
    xml, images, nums = build_section(p0, root_open)
    open(sp, "w", encoding="utf-8").write(xml)

    # 그림 파일·목록
    bd = os.path.join(bx, "BinData")
    for f in os.listdir(bd):
        os.remove(os.path.join(bd, f))
    for bin_id, png in images:
        shutil.copyfile(png, os.path.join(bd, bin_id + ".png"))
    cpath = os.path.join(bx, "Contents", "content.hpf")
    cx = open(cpath, encoding="utf-8").read()
    cx = re.sub(r'<opf:item id="image\d+"[^>]*/>', "", cx)
    items = "".join(f'<opf:item id="{b}" href="BinData/{b}.png" media-type="image/png" isEmbeded="1"/>'
                    for b, _ in images)
    assert '<opf:item id="section1"' in cx
    cx = cx.replace('<opf:item id="section1"', items + '<opf:item id="section1"', 1)
    open(cpath, "w", encoding="utf-8").write(cx)
    # 미리보기 글·커서
    open(os.path.join(bx, "Preview", "PrvText.txt"), "w", encoding="utf-8").write(
        "리스테리아 위해요소 예측모델 개발 연구 보고서\r\n\r\n식품안전정보원\r\n")
    st = os.path.join(bx, "settings.xml")
    s2 = open(st, encoding="utf-8").read()
    s2 = re.sub(r'paraIDRef="\d+" pos="\d+"', 'paraIDRef="0" pos="0"', s2)
    open(st, "w", encoding="utf-8").write(s2)

    # zip(mimetype 첫 항목·무압축)
    hwpx = os.path.join(BUILD, NAME + ".hwpx")
    with zipfile.ZipFile(hwpx, "w") as z:
        z.write(os.path.join(bx, "mimetype"), "mimetype", compress_type=zipfile.ZIP_STORED)
        for root, _, files in os.walk(bx):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, bx).replace("\\", "/")
                if rel == "mimetype":
                    continue
                z.write(full, rel, compress_type=zipfile.ZIP_DEFLATED)
    print("hwpx", hwpx, "tables", sum(1 for k in nums if k[0] == "표"), "figures", len(images))
    if not do_hwp:
        return

    from pyhwpx import Hwp
    hwp = Hwp(new=True, visible=False)
    com = hwp.hwp
    try:
        ok = com.Open(hwpx, "HWPX", "forceopen:true")
        print("open", ok)
        out_hwp = os.path.join(OUT, NAME + ".hwp")
        out_pdf = os.path.join(OUT, NAME + ".pdf")
        for p in (out_hwp, out_pdf):
            if os.path.exists(p):
                os.remove(p)
        com.SaveAs(out_hwp, "HWP", "")
        com.SaveAs(out_pdf, "PDF", "")
        txt = com.GetTextFile("TEXT", "")
        open(os.path.join(OUT, NAME + ".txt"), "w", encoding="utf-8").write(txt)
        print("pages", com.PageCount)
    finally:
        com.Clear(1)
        com.Quit()


if __name__ == "__main__":
    main(do_hwp="--no-hwp" not in sys.argv)
