# -*- coding: utf-8 -*-
"""
리스테리아 개별 보고서 HWP 생성기
===================================

쓰는 법
  py build_report_hwp.py            report_content.py 의 BLOCKS 로
                                    out/04_개별 보고서_리스테리아_v1.hwp, 같은 이름 .pdf, out/png/p01.png … 를 만든다
  py build_report_hwp.py --no-com   한글(COM) 없이 out/_build/report.hwpx 까지만 만든다(서식 점검용)
  다른 보고서를 같은 서식으로 만들 때(옵션을 주지 않으면 위와 똑같이 동작)
    --content <내용 py 경로>      BLOCKS 를 읽을 파일(기본 report_content.py)
    --out-dir <폴더>              결과 폴더(기본 out). HWPX 는 <폴더>/_build/, PNG 는 <폴더>/png/
    --out-name <파일 이름>        확장자 없는 결과 파일 이름(기본 '04_개별 보고서_리스테리아_v1').
                                  기본 이름이 아니면 HWPX 도 <폴더>/_build/<파일 이름>.hwpx 로 만든다
    예) py build_report_hwp.py --content report_1차_baseline.py --out-dir out_1차_baseline --out-name "리스테리아_1차_베이스라인_실험_보고서"

내용을 바꿀 때
  report_content.py 의 BLOCKS 만 고치고 이 파일을 다시 돌린다. 같은 BLOCKS 면 같은 결과가 나온다
  (HWPX 는 바이트까지 같음. HWP/PDF 는 한글이 저장 시각 같은 정보를 넣으므로 내용만 같음).

서식 원본 (이 폴더의 양식/ 사본만 읽는다. 원본 개별보고서 폴더는 건드리지 않는다)
  양식/shellfish_v12.hwpx   06_개별 보고서 _패류독소_v1.2.hwp 를 HWPX 로 바꾼 것
                            -> Contents/header.xml(글꼴·글자 모양·문단 모양·테두리/배경) 을 그대로 쓴다
                            -> 블록별 문단 모양·글자 모양 ID 와 표 XML 모양은 이 파일 section0.xml 에서 실제 쓰인 것을 옮겼다(STYLE, TABLE)
  양식/listeria_v1.hwpx     04_개별 보고서_리스테리아_v1.hwp(빈 양식) 를 HWPX 로 바꾼 것
                            -> 패키지 구성, 쪽 설정(secPr: 용지·여백·단), settings.xml(인쇄 설정: 모아찍기 없음),
                               절(제n절) 앞 쪽 나눔을 이 파일에서 가져온다

방법(검증된 절차)
  1) HWPX(zip) 안의 Contents/section0.xml 만 새로 만들고 header.xml 은 패류독소 것을 그대로 넣는다
  2) hp:linesegarray(줄 배치 캐시) 는 넣지 않는다 — 넣으면 여러 줄 문단이 한 줄로 겹쳐 찍힌다. 빼면 한글이 열 때 다시 계산한다
  3) zip 은 mimetype 을 첫 항목·무압축으로
  4) pyhwpx(COM, 한컴오피스 2020) 로 열어 SaveAs "HWP" 와 "PDF", PDF 는 pymupdf 로 쪽별 PNG(dpi 110)
  숨은 한글 창이 멈추면 이 스크립트가 띄운 '창 제목 없는 Hwp' 프로세스만 종료할 것(사용자 창은 제목이 있음)
  5) 문서 정보 제목(Contents/content.hpf 의 opf:title)은 BLOCKS 의 title 블록 글로 바꾼다(양식 제목이 남지 않게)

표 칸 글
  칸 글의 '\\n' 은 같은 칸 안의 여러 문단(셀 문단 모양 그대로)으로 넣는다. 셀 문단 모양은 한글 낱말을 글자 단위로
  나누므로(breakNonLatinWord=BREAK_WORD, 패류독소 값 그대로) 낱말이 끊기지 않게 BLOCKS 쪽에서 줄을 나눠 둔다.
  PDF 를 만든 뒤 칸 줄마다 PDF 에서 한 줄로 찍혔는지(check_cell_lines) 확인해 출력한다.
"""
import argparse
import importlib.util
import os
import re
import sys
import zipfile
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT_PY = os.path.join(HERE, "report_content.py")
FORM_DIR = os.path.join(HERE, "양식")
SHELLFISH_HWPX = os.path.join(FORM_DIR, "shellfish_v12.hwpx")
LISTERIA_HWPX = os.path.join(FORM_DIR, "listeria_v1.hwpx")
OUT_DIR = os.path.join(HERE, "out")
BUILD_DIR = os.path.join(OUT_DIR, "_build")
PNG_DIR = os.path.join(OUT_DIR, "png")
OUT_NAME = "04_개별 보고서_리스테리아_v1"
PNG_DPI = 110

# ---------------------------------------------------------------------------
# 블록별 서식 (패류독소 v1.2 section0.xml 에서 실제 쓰인 조합; ID 는 패류독소 header.xml 기준)
#   pp = paraPrIDRef, st = styleIDRef, cp = charPrIDRef, prefix = 문장 앞에 글자로 붙는 기호
#   ❍ 는 문단 글머리표가 아니라 글자 U+F0A6, · 는 글자 U+F09E (패류독소 원문 그대로)
# ---------------------------------------------------------------------------
STYLE = {
    "title": dict(pp=20, st=0, cp=9, prefix=""),            # [ 패류독소 개별 보고서 ]
    "h1": dict(pp=21, st=22, cp=7, prefix=""),              # 제1절 …
    "h2": dict(pp=24, st=23, cp=11, prefix=""),             # 1. 원시 데이터 수집
    "h3": dict(pp=27, st=22, cp=12, prefix=""),             # (1) …
    "b1": dict(pp=26, st=22, cp=8, prefix="\uf0a6 "),       # ❍ 문장
    "b2": dict(pp=28, st=22, cp=8, prefix="  - "),          # - 문장 (앞 공백 2칸)
    "b3": dict(pp=28, st=22, cp=8, prefix="   \uf09e "),    # · 문장 (앞 공백 3칸)
}
# 빈 줄: 한글에서 Enter 를 친 것처럼 바로 앞 문단의 문단 모양·글자 모양을 물려받는다(패류독소 원문과 같은 방식).
#        표 바로 뒤의 빈 줄은 패류독소처럼 ❍ 문단 모양(pp26/cp8).
BLANK_AFTER_TABLE = dict(pp=26, st=22, cp=8)

# 표 (패류독소 [표 *] 두 개와 같은 모양)
TABLE = dict(
    anchor_pp=31, anchor_st=21, anchor_cp=18,   # 표를 담는 문단(캡션 스타일, 가운데)
    cap_pp=31, cap_st=21, cap_cp=18,            # 캡션 '[ 표 n ] …' : 표 위(side=TOP), 함초롬바탕 10pt 굵게
    cell_pp=25, cell_st=0,                      # 셀 문단: 가운데 정렬(숫자 칸 포함, 패류독소 관례)
    head_bf=6, head_cp=18,                      # 머리 행: 실선 0.12mm + 배경 #F0F0F0, 함초롬바탕 10pt 굵게
    body_bf=5, body_cp=0,                       # 본문 행: 실선 0.12mm 배경 없음, 함초롬바탕 10pt
    total_bf=5, total_cp=18,                    # 합계 행(total_row=True): 패류독소에 합계 행이 없어 음영 없이 굵게만
    tbl_bf=5,
    width=42189,                                # 표 폭(HWPUNIT) = 패류독소 첫 표 폭
    row_h=1282,                                 # 한 줄 행 높이(패류독소 본문 행). 여러 줄이면 한글이 늘린다
    cell_margin=(510, 510, 141, 141),           # 셀 안 여백 좌우상하
    out_margin=141,
    caption_gap=850, caption_width=8504,
    vert_offset=352,
)
H1_PAGE_BREAK = True   # 리스테리아 양식처럼 제2절부터 각 절을 새 쪽에서 시작(패류독소 v1.2 는 쪽 나눔 없음)

# 그림 블록 ("img", dict(path=그림 파일 절대 경로, caption="[ 그림 n ] …", width_mm=150)):
#   HWPX 에는 그 자리에 표시 글자(IMG_MARK % n, 캡션 문단 모양 = 가운데 정렬)를 넣고, 바로 아래에 캡션 문단을 둔다.
#   한글(COM)로 연 뒤 표시 글자를 찾아 지우고 그 자리에 그림을 '글자처럼 취급'·문서에 포함으로 넣는다
#   (폭 width_mm, 높이는 그림 가로세로 비율대로). 그림은 출력한 이미지 파일을 그대로 쓴다(다시 그리지 않음).
IMG_MARK = "@@IMG%02d@@"
IMG_LIST = []          # build_section 이 채움: [(표시 글자, 경로, 폭 mm, 캡션)]
# 한글 COM 잠금(여러 세션이 한글 자동화를 동시에 쓰지 않게): 변환 전에 이 파일이 있으면 없어질 때까지 기다리고,
# 없으면 만들고(세션 이름·시각) 끝나면 지운다.
HWP_LOCK = os.path.join(HERE, "hwp_build.lock")
LOCK_OWNER = "[1번 베이스라인 실험 다시]"

# 쪽 경계 처리(글꼴·크기·간격 등 보이는 서식은 그대로, 쪽 나눔 동작만 바꿈 — 패류독소 v1.2 와 다른 점)
#   KEEP_TOGETHER : 제목(h1·h2·h3) 문단 모양을 복제해 '다음 문단과 함께'(keepWithNext) 를 켜고,
#                   본문(b1·b2·b3) 문단 모양을 복제해 '외톨이줄 보호'(widowOrphan) 를 켠다.
#                   복제본은 header.xml 끝에 새 ID 로 붙이며 원래 ID 의 다른 값은 모두 같다.
#                   (패류독소 v1.2 는 둘 다 꺼져 있어 쪽 끝에 제목만 남는 경우가 있음)
#   TABLE_SPLIT   : 표가 쪽 경계에 걸릴 때. "TABLE" = 행 경계에서 나눔(셀 내용은 안 쪼갬), 패류독소 원래 값은 "CELL"(셀 안 줄 단위로 나눔)
#   SMALL_TABLE_NO_SPLIT : 어림 높이가 본문 높이의 이 비율 이하인 작은 표는 '글자처럼 취급' + 나누지 않음("NONE") 으로 두어
#                          문단과 함께 통째로 다음 쪽으로 넘어가게 함(머리 행만 쪽 끝에 남는 일, 표만 넘어가고 뒤 빈 줄이 앞 쪽에 남는 일을 막음).
#                          더 긴 표는 패류독소처럼 '글자처럼 취급' 아님(위아래 배치) + TABLE_SPLIT 방식으로 행 경계에서 나뉨
#                          (글자처럼 취급한 표는 한 쪽보다 길면 잘려 찍히므로 긴 표에는 쓰지 않음 — 60행 시험으로 확인)
#   REPEAT_HEADER : 쪽을 넘긴 표에 머리 행을 다시 찍음(머리 행 셀 header="1"; 패류독소 표는 tbl repeatHeader="1" 이지만 셀이 header="0")
#   BLANK_AFTER_TABLE_AUTO : 표 바로 뒤에 빈 줄 블록이 없으면 빈 줄 하나를 넣음(패류독소 표 두 개는 모두 뒤에 빈 줄이 있음)
KEEP_TOGETHER = True
TABLE_SPLIT = "TABLE"
SMALL_TABLE_NO_SPLIT = 0.7
REPEAT_HEADER = True
BLANK_AFTER_TABLE_AUTO = True
BODY_HEIGHT = 84186 - 5668 - 4252 - 4252 - 4252   # 쪽 본문 높이(용지 - 위·아래 여백 - 머리말·꼬리말 여백), HWPUNIT

# 열 폭 자동 보정: BLOCKS 의 widths 비율로 나눈 뒤, 숫자(예: 6,280,656 · 2015-04-07 · 628,878행)와
# 머리 행·합계 행 낱말이 셀 안에서 쪼개지지 않을 만큼만 그 열을 넓히고, 여유 있는 다른 열에서 비율대로 덜어낸다.
# widths 합이 표 폭(TABLE["width"])과 같은 정수 목록이면 HWPUNIT 로 직접 정한 폭으로 보고 보정하지 않는다
# (어림 글자 폭이 실제보다 넓어, 실측으로 맞춘 폭을 흐트러뜨리므로).
AUTO_FIT_COLUMNS = True

# header.xml 에서 확인할 값(양식 파일이 바뀌면 여기서 멈춘다): charPr id -> (크기pt, 색, 굵게, 한글 글꼴)
EXPECT_CHAR = {
    9: (17.0, "#0000FF", True, "휴먼명조"),
    7: (15.0, "#000000", True, "휴먼명조"),
    11: (14.0, "#000000", False, "HY견고딕"),
    12: (12.0, "#000000", True, "휴먼명조"),
    8: (12.0, "#000000", False, "휴먼명조"),
    18: (10.0, "#000000", True, "함초롬바탕"),
    0: (10.0, "#000000", False, "함초롬바탕"),
}


# ---------------------------------------------------------------------------
def load_blocks(content_py=None):
    sys.dont_write_bytecode = True   # 보고서 폴더에 __pycache__ 를 만들지 않음
    spec = importlib.util.spec_from_file_location("report_content", content_py or CONTENT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BLOCKS


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        return [(i, z.read(i.filename)) for i in z.infolist()]


def zget(entries, name):
    for i, d in entries:
        if i.filename == name:
            return d
    raise KeyError(name)


# ---------------------------------------------------------------------------
# header.xml 해석(보고·점검용)
def parse_header(xml):
    fonts = {}
    for fm in re.finditer(r'<hh:fontface lang="(\w+)"[^>]*>(.*?)</hh:fontface>', xml, re.S):
        fonts[fm.group(1)] = {int(m.group(1)): m.group(2) for m in re.finditer(r'<hh:font id="(\d+)" face="([^"]+)"', fm.group(2))}
    chars = {}
    for m in re.finditer(r'<hh:charPr id="(\d+)"(.*?)</hh:charPr>', xml, re.S):
        b = m.group(2)
        fr = re.search(r'<hh:fontRef hangul="(\d+)" latin="(\d+)"', b)
        chars[int(m.group(1))] = dict(
            size=int(re.search(r'height="(\d+)"', b).group(1)) / 100,
            color=re.search(r'textColor="([^"]+)"', b).group(1),
            bold="<hh:bold/>" in b,
            hangul=fonts["HANGUL"][int(fr.group(1))],
            latin=fonts["LATIN"][int(fr.group(2))],
            spacing=int(re.search(r'<hh:spacing hangul="(-?\d+)"', b).group(1)),
        )
    paras = {}
    for m in re.finditer(r'<hh:paraPr id="(\d+)"(.*?)</hh:paraPr>', xml, re.S):
        b = m.group(2)
        case = re.search(r'<hp:case[^>]*>(.*?)</hp:case>', b, re.S)
        mb = case.group(1) if case else b

        def g(tag):
            x = re.search(r'<hc:%s value="(-?\d+)"' % tag, mb)
            return int(x.group(1)) if x else None
        ls = re.search(r'<hh:lineSpacing type="(\w+)" value="(\d+)"', mb) or re.search(r'<hh:lineSpacing type="(\w+)" value="(\d+)"', b)
        paras[int(m.group(1))] = dict(
            align=re.search(r'<hh:align horizontal="(\w+)"', b).group(1),
            intent=g("intent"), left=g("left"), prev=g("prev"), next=g("next"),
            line="%s %s" % (ls.group(1), ls.group(2)))
    fills = {}
    for m in re.finditer(r'<hh:borderFill id="(\d+)"(.*?)</hh:borderFill>', xml, re.S):
        b = m.group(2)
        lb = re.search(r'<hh:leftBorder type="(\w+)" width="([^"]+)"', b)
        fc = re.search(r'faceColor="([^"]+)"', b)
        fills[int(m.group(1))] = dict(border="%s %s" % lb.groups(), face=fc.group(1) if fc else "none")
    styles = {int(m.group(1)): m.group(2) for m in re.finditer(r'<hh:style id="(\d+)" type="\w+" name="([^"]*)"', xml)}
    return dict(chars=chars, paras=paras, fills=fills, styles=styles)


def check_header(h):
    bad = []
    for cid, (size, color, bold, face) in EXPECT_CHAR.items():
        c = h["chars"].get(cid)
        if not c or (c["size"], c["color"], c["bold"], c["hangul"]) != (size, color, bold, face):
            bad.append("charPr %d: 기대 %s, 실제 %s" % (cid, (size, color, bold, face), c))
    for k, s in list(STYLE.items()) + [("blank_after_table", BLANK_AFTER_TABLE)]:
        if s["pp"] not in h["paras"] or s["st"] not in h["styles"] or s["cp"] not in h["chars"]:
            bad.append("%s: ID 없음 %s" % (k, s))
    for key in ("head_bf", "body_bf", "total_bf", "tbl_bf"):
        if TABLE[key] not in h["fills"]:
            bad.append("borderFill %s 없음" % key)
    if bad:
        raise SystemExit("header.xml 이 예상과 다름(양식 파일 확인):\n  " + "\n  ".join(bad))


def style_map_lines(h):
    def desc(pp, cp, st):
        c, p = h["chars"][cp], h["paras"][pp]
        return ("paraPr %d(%s, 내어쓰기 %s, 왼쪽 %s, 위 %s/아래 %s, 줄간격 %s) / charPr %d(%s·%s %.0fpt %s%s, 자간 %d) / style %d(%s)"
                % (pp, p["align"], p["intent"], p["left"], p["prev"], p["next"], p["line"],
                   cp, c["hangul"], c["latin"], c["size"], c["color"], " 굵게" if c["bold"] else "", c["spacing"],
                   st, h["styles"].get(st, "?")))
    out = []
    for k, s in STYLE.items():
        extra = ""
        if s["pp"] in PP_REMAP:
            extra = "  [쪽 나눔용 복제 paraPr %d 사용]" % PP_REMAP[s["pp"]]
        out.append("%-6s %s  앞글자=%r%s" % (k, desc(s["pp"], s["cp"], s["st"]), s["prefix"], extra))
    out.append("blank  바로 앞 문단 모양 물려받음 / 표 뒤: " + desc(BLANK_AFTER_TABLE["pp"], BLANK_AFTER_TABLE["cp"], BLANK_AFTER_TABLE["st"]))
    T = TABLE
    out.append("caption " + desc(T["cap_pp"], T["cap_cp"], T["cap_st"]) + " side=TOP gap=%d" % T["caption_gap"])
    out.append("tbl-anchor " + desc(T["anchor_pp"], T["anchor_cp"], T["anchor_st"]))
    for nm, bf, cp in (("head", T["head_bf"], T["head_cp"]), ("body", T["body_bf"], T["body_cp"]), ("total", T["total_bf"], T["total_cp"])):
        out.append("cell-%s borderFill %d(%s, 배경 %s) / %s" % (nm, bf, h["fills"][bf]["border"], h["fills"][bf]["face"], desc(T["cell_pp"], cp, T["cell_st"])))
    return out


PP_REMAP = {}   # KEEP_TOGETHER 로 복제한 문단 모양: 원래 ID -> 새 ID


def patch_header(xml):
    """KEEP_TOGETHER: 제목·본문 문단 모양을 복제해 쪽 나눔 설정 하나만 켠 새 ID 를 header.xml 끝에 붙인다"""
    if not KEEP_TOGETHER:
        return xml, {}
    plan = [(STYLE["h1"]["pp"], "keepWithNext"), (STYLE["h2"]["pp"], "keepWithNext"), (STYLE["h3"]["pp"], "keepWithNext"),
            (STYLE["b1"]["pp"], "widowOrphan"), (STYLE["b2"]["pp"], "widowOrphan")]
    if STYLE["b3"]["pp"] != STYLE["b2"]["pp"]:
        plan.append((STYLE["b3"]["pp"], "widowOrphan"))
    nxt = max(int(x) for x in re.findall(r'<hh:paraPr id="(\d+)"', xml)) + 1
    clones, remap = [], {}
    for pid, attr in plan:
        if pid in remap:
            continue
        blk = re.search(r'<hh:paraPr id="%d" .*?</hh:paraPr>' % pid, xml, re.S).group(0)
        blk = blk.replace('<hh:paraPr id="%d" ' % pid, '<hh:paraPr id="%d" ' % nxt, 1)
        blk, k = re.subn(r'\b%s="0"' % attr, '%s="1"' % attr, blk, count=1)
        if k != 1:
            raise SystemExit("paraPr %d 에 %s 설정이 없음" % (pid, attr))
        clones.append(blk)
        remap[pid] = nxt
        nxt += 1
    xml = xml.replace("</hh:paraProperties>", "".join(clones) + "</hh:paraProperties>", 1)
    xml = re.sub(r'<hh:paraProperties itemCnt="(\d+)"',
                 lambda m: '<hh:paraProperties itemCnt="%d"' % (int(m.group(1)) + len(clones)), xml, count=1)
    return xml, remap


def PP(pid):
    return PP_REMAP.get(pid, pid)


# ---------------------------------------------------------------------------
# section0.xml 생성
def run_xml(cp, text):
    if text == "":
        return '<hp:run charPrIDRef="%d"/>' % cp
    return '<hp:run charPrIDRef="%d"><hp:t>%s</hp:t></hp:run>' % (cp, escape(text))


def para_xml(pp, st, cp, text, page_break=False, pid="2147483648"):
    return ('<hp:p id="%s" paraPrIDRef="%d" styleIDRef="%d" pageBreak="%d" columnBreak="0" merged="0">%s</hp:p>'
            % (pid, pp, st, 1 if page_break else 0, run_xml(cp, text)))


def split_widths(ratios, total):
    s = float(sum(ratios))
    w = [int(round(total * r / s)) for r in ratios]
    w[-1] += total - sum(w)          # 반올림 오차는 마지막 열에
    return w


# 함초롬바탕 10pt 글자 폭 어림값(HWPUNIT, 1000 = 10pt). 한글이 만든 PDF 의 글자 위치 간격으로 잰 값(2026-10-16)에 맞춤
#   한글 한 자 972, 띄어쓰기 504(한글은 글꼴 공백 대신 자체 공백 폭을 씀), 숫자 552, 가운뎃점 324
def text_width(s, bold=False):
    w = 0
    for ch in s:
        o = ord(ch)
        if ch.isdigit() and o < 128:
            w += 552
        elif ch in "MW":
            w += 860
        elif ch in "IJ":
            w += 310
        elif "A" <= ch <= "Z":
            w += 700
        elif ch == "m":
            w += 900
        elif ch == "w":
            w += 720
        elif ch in "ijlft":
            w += 330
        elif ch in "rs":
            w += 490
        elif "a" <= ch <= "z":
            w += 590
        elif ch in ",.:;'\"|!·²°`()[]{}":
            w += 324
        elif ch in "/-~_=+*<>":
            w += 552
        elif ch == "%":
            w += 828
        elif ch == "×":
            w += 612
        elif ch == " ":
            w += 504
        elif o < 0x2000 and o >= 0xA0:
            w += 600
        else:
            w += 972                       # 한글·전각·화살표·원문자
    return int(w * (1.02 if bold else 1.0))


NUMERIC_TOKEN = re.compile(r"^[\d,.\-~%:/()+]*\d[\d,.\-~%:/()+]*[가-힣]{0,2}$")


def table_margin(tb):
    """표의 셀 안 여백(좌, 우, 위, 아래). BLOCKS 표에 cell_margin_lr 이 있으면 그 표만 좌우 여백을 그 값으로"""
    ml, mr, mt, mb = TABLE["cell_margin"]
    if tb.get("cell_margin_lr") is not None:
        ml = mr = int(tb["cell_margin_lr"])
    return (ml, mr, mt, mb)


def fit_widths(widths, header, rows, total_row, margin=None):
    """숫자 낱말과 머리·합계 행 낱말이 쪼개지지 않게 최소 폭을 보장(가능한 범위에서)"""
    T = TABLE
    margin = margin or T["cell_margin"]
    pad = margin[0] + margin[1] + 200
    ncol = len(widths)
    need = [0] * ncol
    lines = [(header, True)] + [(r, total_row and i == len(rows) - 1) for i, r in enumerate(rows)]
    for cells, bold in lines:
        for ci, c in enumerate(cells):
            for tok in str(c).split():
                tw0 = text_width(tok, bold)
                if bold or NUMERIC_TOKEN.match(tok) or tw0 <= 3600:   # 3600 = 전각 3.5자 정도의 짧은 낱말(예: MJ/m²)
                    tw = tw0 + pad
                    if tw <= T["width"] * 0.3:
                        need[ci] = max(need[ci], tw)
    w = list(widths)
    deficit = sum(max(0, need[i] - w[i]) for i in range(ncol))
    if deficit == 0:
        return w, False
    slack = [max(0, w[i] - need[i]) if need[i] <= w[i] else 0 for i in range(ncol)]
    total_slack = sum(slack)
    if total_slack <= 0:
        return w, False
    take = min(deficit, total_slack)
    scale = take / float(deficit)
    for i in range(ncol):
        if need[i] > w[i]:
            w[i] += int((need[i] - w[i]) * scale)
    for i in range(ncol):
        if slack[i] > 0:
            w[i] -= int(take * slack[i] / float(total_slack))
    w[max(range(ncol), key=lambda i: slack[i])] += T["width"] - sum(w)
    return w, True


def cell_xml(text, col, row, width, height, bf, cp, is_header=False, margin=None):
    """칸 글의 '\\n' 은 같은 칸 안의 새 문단(같은 셀 문단 모양·글자 모양)으로 넣는다.
    셀 문단 모양(패류독소 paraPr 25)은 한글 낱말을 글자 단위로 나누므로(breakNonLatinWord=BREAK_WORD),
    낱말이 중간에서 끊기지 않게 하려면 BLOCKS 의 칸 글에 자연스러운 경계에서 '\\n' 을 넣어 줄마다 칸 폭 안에 들게 한다."""
    T = TABLE
    paras = "".join(para_xml(T["cell_pp"], T["cell_st"], cp, line) for line in str(text).split("\n"))
    ml, mr, mt, mb = margin or T["cell_margin"]
    return ('<hp:tc name="" header="%d" hasMargin="0" protect="0" editable="0" dirty="0" borderFillIDRef="%d">'
            % (1 if is_header else 0, bf) +
            '<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="CENTER" linkListIDRef="0" '
            'linkListNextIDRef="0" textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">%s</hp:subList>'
            '<hp:cellAddr colAddr="%d" rowAddr="%d"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
            '<hp:cellSz width="%d" height="%d"/><hp:cellMargin left="%d" right="%d" top="%d" bottom="%d"/></hp:tc>'
            % (paras, col, row, width, height, ml, mr, mt, mb))


WIDTH_LOG = []   # (캡션, 비율 폭, 보정 폭) — 보고용
AUTO_BLANK_LOG = []   # 빈 줄을 자동으로 넣은 표 캡션 — 보고용
SPLIT_LOG = []   # (캡션, 어림 높이 비율, pageBreak 값) — 보고용
MARGIN_LOG = []   # (캡션, 셀 안 여백) — 기본과 다른 여백을 쓴 표, 보고용


def estimate_table_height(allrows, widths, margin=None):
    """셀 글 폭으로 줄 수를 어림해 표 높이(캡션 포함, HWPUNIT)를 계산. 한 줄 행 1282, 줄이 늘 때마다 1600(10pt×160%)"""
    T = TABLE
    margin = margin or T["cell_margin"]
    pad = margin[0] + margin[1]
    h = 1500 + T["caption_gap"]
    for cells, bf, cp in allrows:
        bold = cp == T["head_cp"]
        n = 1
        for ci, c in enumerate(cells):
            inner = max(1, widths[ci] - pad)
            lines = sum(max(1, -(-int(text_width(part, bold) * 1.03) // inner)) for part in str(c).split("\n"))
            n = max(n, lines)
        h += T["row_h"] + (n - 1) * 1600
    return h


def table_xml(tb, n):
    T = TABLE
    header, rows = tb["header"], tb["rows"]
    ncol = len(header)
    for r in rows:
        if len(r) != ncol:
            raise SystemExit("표 '%s': 열 수가 머리 행(%d)과 다른 행 %r" % (tb["caption"], ncol, r))
    total = bool(tb.get("total_row"))
    margin = table_margin(tb)
    if margin != tuple(T["cell_margin"]):
        MARGIN_LOG.append((tb["caption"], margin))
    spec = tb.get("widths") or [1] * ncol
    exact = sum(spec) == T["width"] and all(float(x).is_integer() for x in spec)   # HWPUNIT 로 직접 정한 열 폭
    widths = split_widths(spec, T["width"])
    if AUTO_FIT_COLUMNS and not exact:
        fitted, changed = fit_widths(widths, header, rows, total, margin)
        if changed:
            WIDTH_LOG.append((tb["caption"], widths, fitted))
            widths = fitted
    allrows = [(header, T["head_bf"], T["head_cp"])]
    for i, r in enumerate(rows):
        last = i == len(rows) - 1
        if total and last:
            allrows.append((r, T["total_bf"], T["total_cp"]))
        else:
            allrows.append((r, T["body_bf"], T["body_cp"]))
    trs = []
    for ri, (cells, bf, cp) in enumerate(allrows):
        hd = REPEAT_HEADER and ri == 0
        tcs = "".join(cell_xml(c, ci, ri, widths[ci], T["row_h"], bf, cp, hd, margin) for ci, c in enumerate(cells))
        trs.append("<hp:tr>%s</hp:tr>" % tcs)
    height = T["row_h"] * len(allrows)
    est = estimate_table_height(allrows, widths, margin)
    small = est <= SMALL_TABLE_NO_SPLIT * BODY_HEIGHT
    split = "NONE" if small else TABLE_SPLIT
    SPLIT_LOG.append((tb["caption"], est / float(BODY_HEIGHT), split + (" 글자처럼 취급" if small else " 위아래 배치")))
    om = T["out_margin"]
    ml, mr, mt, mb = margin
    caption = para_xml(T["cap_pp"], T["cap_st"], T["cap_cp"], tb["caption"])
    tbl = ('<hp:tbl id="%d" zOrder="%d" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" lock="0" '
           'dropcapstyle="None" pageBreak="%s" repeatHeader="1" rowCnt="%d" colCnt="%d" cellSpacing="0" borderFillIDRef="%d" noAdjust="1">'
           '<hp:sz width="%d" widthRelTo="ABSOLUTE" height="%d" heightRelTo="ABSOLUTE" protect="0"/>'
           '<hp:pos treatAsChar="%d" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" vertRelTo="PARA" '
           'horzRelTo="PARA" vertAlign="TOP" horzAlign="LEFT" vertOffset="%d" horzOffset="0"/>'
           '<hp:outMargin left="%d" right="%d" top="%d" bottom="%d"/>'
           '<hp:caption side="TOP" fullSz="0" width="%d" gap="%d" lastWidth="%d"><hp:subList id="" textDirection="HORIZONTAL" '
           'lineWrap="BREAK" vertAlign="TOP" linkListIDRef="0" linkListNextIDRef="0" textWidth="0" textHeight="0" hasTextRef="0" '
           'hasNumRef="0">%s</hp:subList></hp:caption>'
           '<hp:inMargin left="%d" right="%d" top="%d" bottom="%d"/>%s</hp:tbl>'
           % (1900000001 + n, n, split, len(allrows), ncol, T["tbl_bf"], T["width"], height,
              1 if small else 0, 0 if small else T["vert_offset"], om, om, om, om,
              T["caption_width"], T["caption_gap"], T["width"], caption, ml, mr, mt, mb, "".join(trs)))
    return ('<hp:p id="2147483648" paraPrIDRef="%d" styleIDRef="%d" pageBreak="0" columnBreak="0" merged="0">'
            '<hp:run charPrIDRef="%d">%s<hp:t/></hp:run></hp:p>' % (T["anchor_pp"], T["anchor_st"], T["anchor_cp"], tbl))


def build_section(blocks, listeria_sec):
    m = re.search(r'^(.*?<hs:sec [^>]*>)', listeria_sec, re.S)
    head = m.group(1)
    secpr = re.search(r'<hp:secPr .*?</hp:secPr>', listeria_sec, re.S).group(0)
    colpr = re.search(r'<hp:ctrl><hp:colPr [^>]*/></hp:ctrl>', listeria_sec).group(0)

    out = [head]
    prev = None          # 바로 앞 문단의 (pp, st, cp) — 빈 줄이 물려받음
    prev_kind = None
    n_h1 = 0
    n_tbl = 0
    n_img = 0
    del IMG_LIST[:]
    first = True
    for bi, blk in enumerate(blocks):
        kind = blk[0]
        if first and kind != "title":
            raise SystemExit("BLOCKS 첫 블록은 title 이어야 함(쪽 설정을 그 문단에 넣음)")
        if kind == "title":
            s = STYLE["title"]
            out.append('<hp:p id="0" paraPrIDRef="%d" styleIDRef="%d" pageBreak="0" columnBreak="0" merged="0">'
                       '<hp:run charPrIDRef="%d">%s%s</hp:run>%s</hp:p>'
                       % (s["pp"], s["st"], s["cp"], secpr if first else "", colpr if first else "", run_xml(s["cp"], blk[1])))
            prev = (s["pp"], s["st"], s["cp"])
        elif kind in STYLE:
            s = STYLE[kind]
            pb = False
            if kind == "h1":
                n_h1 += 1
                pb = H1_PAGE_BREAK and n_h1 > 1
            out.append(para_xml(PP(s["pp"]), s["st"], s["cp"], s["prefix"] + blk[1], page_break=pb))
            prev = (PP(s["pp"]), s["st"], s["cp"])
        elif kind == "blank":
            if prev_kind == "table" or prev is None:
                b = BLANK_AFTER_TABLE
                pp, st, cp = PP(b["pp"]), b["st"], b["cp"]
            else:
                pp, st, cp = prev
            out.append(para_xml(pp, st, cp, ""))
            prev = (pp, st, cp)
        elif kind == "table":
            n_tbl += 1
            out.append(table_xml(blk[1], n_tbl))
            prev = (PP(BLANK_AFTER_TABLE["pp"]), BLANK_AFTER_TABLE["st"], BLANK_AFTER_TABLE["cp"])
            nxt = blocks[bi + 1][0] if bi + 1 < len(blocks) else None
            if BLANK_AFTER_TABLE_AUTO and nxt not in ("blank", "h1"):
                out.append(para_xml(prev[0], prev[1], prev[2], ""))
                AUTO_BLANK_LOG.append(blk[1]["caption"])
                kind = "blank"
        elif kind == "img":
            n_img += 1
            im = blk[1]
            if not os.path.isfile(im["path"]):
                raise SystemExit("그림 파일 없음: %s" % im["path"])
            mark = IMG_MARK % n_img
            IMG_LIST.append((mark, os.path.abspath(im["path"]), float(im.get("width_mm", 150)), im["caption"]))
            T = TABLE
            out.append(para_xml(T["cap_pp"], T["cap_st"], T["cap_cp"], mark))          # 그림 자리(한글에서 그림으로 바꿈)
            out.append(para_xml(T["cap_pp"], T["cap_st"], T["cap_cp"], im["caption"]))  # 캡션(그림 아래)
            prev = (PP(BLANK_AFTER_TABLE["pp"]), BLANK_AFTER_TABLE["st"], BLANK_AFTER_TABLE["cp"])
            nxt = blocks[bi + 1][0] if bi + 1 < len(blocks) else None
            if BLANK_AFTER_TABLE_AUTO and nxt not in ("blank", "h1"):
                out.append(para_xml(prev[0], prev[1], prev[2], ""))
                AUTO_BLANK_LOG.append(im["caption"])
            kind = "blank"
        else:
            raise SystemExit("모르는 블록 종류: %r" % (blk,))
        prev_kind = kind
        first = False
    out.append("</hs:sec>")
    xml = "".join(out)
    assert "linesegarray" not in xml
    return xml


def preview_text(blocks):
    lines = []
    for b in blocks:
        k = b[0]
        if k in STYLE:
            lines.append((STYLE[k]["prefix"] + b[1]).replace("\uf0a6", "❍").replace("\uf09e", "·"))
        elif k in ("table", "img"):
            lines.append(b[1]["caption"])
    txt = "\r\n".join(lines)
    return txt[:1000]


# ---------------------------------------------------------------------------
def build_hwpx(blocks, out_path):
    sf = read_zip(SHELLFISH_HWPX)
    li = read_zip(LISTERIA_HWPX)
    header_xml = zget(sf, "Contents/header.xml").decode("utf-8")
    h = parse_header(header_xml)
    check_header(h)
    header_xml, remap = patch_header(header_xml)
    PP_REMAP.clear()
    PP_REMAP.update(remap)
    del WIDTH_LOG[:], SPLIT_LOG[:], AUTO_BLANK_LOG[:], MARGIN_LOG[:]
    header = header_xml.encode("utf-8")
    section = build_section(blocks, zget(li, "Contents/section0.xml").decode("utf-8"))
    settings = zget(li, "settings.xml").decode("utf-8")
    settings = re.sub(r'<ha:CaretPosition [^>]*/>', '<ha:CaretPosition listIDRef="0" paraIDRef="0" pos="0"/>', settings)
    # 문서 정보 제목: 양식에 남은 제목('제3절 위해요소 예측모델 개발 방안') 대신 BLOCKS 의 title 블록 글
    hpf = zget(li, "Contents/content.hpf").decode("utf-8")
    title = next(b[1] for b in blocks if b[0] == "title")
    hpf, k = re.subn(r'<opf:title>.*?</opf:title>|<opf:title/>', lambda m: '<opf:title>%s</opf:title>' % escape(title), hpf, count=1)
    if k != 1:
        raise SystemExit("content.hpf 에 opf:title 이 없음(양식 파일 확인)")
    replace = {
        "Contents/header.xml": header,
        "Contents/section0.xml": section.encode("utf-8"),
        "Contents/content.hpf": hpf.encode("utf-8"),
        "settings.xml": settings.encode("utf-8"),
        "Preview/PrvText.txt": preview_text(blocks).encode("utf-8"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, "w") as z:
        names = [i.filename for i, _ in li]
        assert names[0] == "mimetype"
        for info, data in li:
            zi = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED if info.filename in ("mimetype", "version.xml") else zipfile.ZIP_DEFLATED
            zi.external_attr = info.external_attr
            z.writestr(zi, replace.get(info.filename, data))
    return h, section


def _hwp_processes():
    """실행 중인 Hwp.exe: {PID: 창 제목}. 창 제목이 없으면 'N/A'"""
    import csv
    import io
    import subprocess
    r = subprocess.run(["tasklist", "/V", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq Hwp.exe"],
                       capture_output=True, encoding="cp949", errors="replace")
    out = {}
    for row in csv.reader(io.StringIO(r.stdout)):
        if len(row) >= 9 and row[0].lower() == "hwp.exe":
            out[int(row[1])] = row[8]
    return out


def _img_size_mm(path, width_mm):
    import pymupdf
    pm = pymupdf.Pixmap(path)
    w, h = pm.width, pm.height
    return width_mm, round(width_mm * h / float(w), 1)


def hwp_lock_acquire(owner=LOCK_OWNER, wait_s=10, max_wait_s=3600):
    """HWP_LOCK 이 있으면 없어질 때까지 기다린 뒤 만든다(내용 = 세션 이름·시각·PID)."""
    import time
    t0 = time.time()
    while os.path.exists(HWP_LOCK):
        if time.time() - t0 > max_wait_s:
            raise SystemExit("한글 잠금 파일이 %d초 넘게 남아 있음: %s" % (max_wait_s, HWP_LOCK))
        try:
            print("한글 잠금 대기:", open(HWP_LOCK, encoding="utf-8").read().strip().replace("\n", " | "))
        except OSError:
            pass
        time.sleep(wait_s)
    with open(HWP_LOCK, "x", encoding="utf-8") as f:
        f.write("owner=%s\nstarted=%s\npid=%d\n" % (owner, time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid()))


def hwp_lock_release():
    try:
        txt = open(HWP_LOCK, encoding="utf-8").read()
        if "pid=%d" % os.getpid() in txt:
            os.remove(HWP_LOCK)
    except OSError:
        pass


def convert_with_hangul(hwpx_path, hwp_path, pdf_path, images=()):
    import subprocess
    import time
    from pyhwpx import Hwp
    before = set(_hwp_processes())
    hwp = Hwp(new=True, visible=False)
    mine = [p for p in _hwp_processes() if p not in before]   # 이번에 띄운 숨은 한글
    com = hwp.hwp
    try:
        ok = com.Open(hwpx_path, "HWPX", "forceopen:true")
        if not ok:
            raise RuntimeError("HWPX 열기 실패: " + hwpx_path)
        for mark, path, wmm, _cap in images:                    # 표시 글자 -> 그림(글자처럼 취급, 문서에 포함)
            hwp.MoveDocBegin()
            if not hwp.find(mark, direction="Forward"):
                raise RuntimeError("그림 자리 표시 글자를 찾지 못함: %s" % mark)
            hwp.Delete()
            w, h = _img_size_mm(path, wmm)
            hwp.insert_picture(path, treat_as_char=True, embedded=True, sizeoption=1, width=w, height=h)
        pages = com.PageCount
        if not com.SaveAs(hwp_path, "HWP", ""):
            raise RuntimeError("HWP 저장 실패")
        if not com.SaveAs(pdf_path, "PDF", ""):
            raise RuntimeError("PDF 저장 실패")
    finally:
        try:
            com.Clear(1)
        except Exception:
            pass
        com.Quit()
        # Quit 뒤에도 남은 경우, 이번에 띄운 PID 중 창 제목이 없는 것만 종료(사용자 한글 창은 건드리지 않음)
        time.sleep(2)
        now = _hwp_processes()
        for p in mine:
            if p in now and now[p] in ("N/A", ""):
                subprocess.run(["taskkill", "/PID", str(p), "/F"], capture_output=True)
                print("Quit 뒤 남은 숨은 한글 프로세스 종료: PID", p)
    return pages


def render_png(pdf_path, png_dir, dpi=PNG_DPI):
    import pymupdf
    if os.path.isdir(png_dir):
        for f in os.listdir(png_dir):
            if re.fullmatch(r"p\d+\.png", f):
                os.remove(os.path.join(png_dir, f))
    os.makedirs(png_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    sizes = set()
    for i, pg in enumerate(doc):
        sizes.add((round(pg.rect.width), round(pg.rect.height)))
        pg.get_pixmap(dpi=dpi).save(os.path.join(png_dir, "p%02d.png" % (i + 1)))
    text = "".join(pg.get_text() for pg in doc)
    n = doc.page_count
    doc.close()
    return n, sizes, text


def check_pdf_text(blocks, pdf_text):
    """모든 블록 글이 PDF 에 들어갔는지(공백 무시) 확인"""
    norm = lambda s: re.sub(r"\s+", "", s)
    hay = norm(pdf_text)
    missing = []
    for b in blocks:
        if b[0] in STYLE:
            if norm(b[1]) not in hay:
                missing.append(b[1][:40])
        elif b[0] == "table":
            for t in [b[1]["caption"]] + list(b[1]["header"]) + [c for r in b[1]["rows"] for c in r]:
                if t and norm(str(t)) not in hay:
                    missing.append("표: " + str(t)[:40])
        elif b[0] == "img":
            if norm(b[1]["caption"]) not in hay:
                missing.append("그림: " + b[1]["caption"][:40])
    if "@@IMG" in pdf_text:
        missing.append("그림으로 바뀌지 않은 표시 글자: %s" % re.findall(r"@@IMG\d+@@", pdf_text))
    return missing


def check_cell_lines(blocks, pdf_path):
    """표 칸 글의 줄('\\n' 으로 나눈 것)마다 PDF 에서 한 줄로 찍혔는지 확인(한글이 줄을 더 나눴으면 낱말이 끊긴 것).
    돌려주는 값: [(표 캡션, 칸 줄 글)] — 한 줄로 찍히지 않은 것"""
    import pymupdf
    norm = lambda s: re.sub(r"\s+", "", s)
    doc = pymupdf.open(pdf_path)
    vis = []
    for pg in doc:
        for b in pg.get_text("rawdict")["blocks"]:
            for ln in b.get("lines", []):
                # 표 칸 글(함초롬바탕 10pt)만 본다 — 본문(휴먼명조 12pt)에 같은 글이 있어도 칸이 끊긴 것을 놓치지 않게
                sp = [s for s in ln["spans"] if "HCRBatang" in s["font"] and abs(s["size"] - 10) < 0.3]
                if sp:
                    vis.append(norm("".join(c["c"] for s in sp for c in s["chars"])))
    doc.close()
    bad = []
    for b in blocks:
        if b[0] != "table":
            continue
        tb = b[1]
        for c in list(tb["header"]) + [c for r in tb["rows"] for c in r]:
            for line in str(c).split("\n"):
                t = norm(line)
                if len(t) >= 2 and not any(t in v for v in vis):
                    bad.append((tb["caption"], line))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-com", action="store_true", help="HWPX 까지만 만든다")
    ap.add_argument("--content", default=CONTENT_PY, help="BLOCKS 를 읽을 내용 py 경로(기본 report_content.py)")
    ap.add_argument("--out-dir", default=OUT_DIR, help="결과 폴더(기본 out)")
    ap.add_argument("--out-name", default=OUT_NAME, help="확장자 없는 결과 파일 이름(기본 '%s')" % OUT_NAME)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    content_py = os.path.abspath(a.content)
    out_dir = os.path.abspath(a.out_dir)
    out_name = a.out_name
    build_dir = os.path.join(out_dir, "_build")
    png_dir = os.path.join(out_dir, "png")
    if content_py != CONTENT_PY or out_dir != OUT_DIR or out_name != OUT_NAME:
        print("내용:", content_py, "| 결과 폴더:", out_dir, "| 파일 이름:", out_name)

    blocks = load_blocks(content_py)
    hwpx = os.path.join(build_dir, "report.hwpx" if out_name == OUT_NAME else out_name + ".hwpx")
    h, section = build_hwpx(blocks, hwpx)
    print("HWPX:", hwpx, "(section0.xml %d bytes, 블록 %d개, 표 %d개)" % (
        len(section.encode("utf-8")), len(blocks), sum(1 for b in blocks if b[0] == "table")))
    print("== style_map")
    for line in style_map_lines(h):
        print("  " + line)
    if PP_REMAP:
        print("== 쪽 나눔용 복제 문단 모양(원래 -> 새 ID):", PP_REMAP)
    pct = lambda ws: "[" + ", ".join("%.1f" % (100.0 * x / TABLE["width"]) for x in ws) + "]%"
    for cap, w0, w1 in WIDTH_LOG:
        print("== 열 폭 보정 %s: %s -> %s" % (cap, pct(w0), pct(w1)))
    for cap, r, sp in SPLIT_LOG:
        print("== 쪽 경계 %s: 어림 높이 본문의 %.0f%% -> pageBreak=%s" % (cap, 100 * r, sp))
    for cap in AUTO_BLANK_LOG:
        print("== 표 뒤 빈 줄 자동 추가: %s" % cap)
    for cap, m in MARGIN_LOG:
        print("== 셀 안 여백 %s: 좌우 %d·위아래 %d HWPUNIT (기본 좌우 %d)" % (cap, m[0], m[2], TABLE["cell_margin"][0]))
    if a.no_com:
        return
    hwp_path = os.path.join(out_dir, out_name + ".hwp")
    pdf_path = os.path.join(out_dir, out_name + ".pdf")
    if IMG_LIST:
        print("== 그림 %d개: %s" % (len(IMG_LIST), ", ".join("%s(%gmm)" % (os.path.basename(p), w) for _, p, w, _ in IMG_LIST)))
    hwp_lock_acquire()
    try:
        pages = convert_with_hangul(hwpx, hwp_path, pdf_path, list(IMG_LIST))
    finally:
        hwp_lock_release()
    n, sizes, text = render_png(pdf_path, png_dir)
    print("HWP:", hwp_path, "(한글 쪽 수 %s)" % pages)
    print("PDF:", pdf_path, "(%d쪽, 쪽 크기 pt %s)" % (n, sorted(sizes)))
    print("PNG:", png_dir, "p01.png ~ p%02d.png" % n)
    missing = check_pdf_text(blocks, text)
    print("PDF 글 대조:", "빠진 글 없음" if not missing else "빠진 글 %d건 %s" % (len(missing), missing[:10]))
    broken = check_cell_lines(blocks, pdf_path)
    print("표 칸 줄 점검:", "모든 칸 줄이 한 줄로 찍힘(한글이 더 나눈 줄 없음)" if not broken else
          "한 줄로 찍히지 않은 칸 줄 %d건" % len(broken))
    for cap, line in broken:
        print("   %s  %r" % (cap, line))


if __name__ == "__main__":
    main()
