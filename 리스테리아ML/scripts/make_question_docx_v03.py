# -*- coding: utf-8 -*-
"""질문 리스트 워드 v0.3 생성 — 타겟 범위 · 기상 × 가공식품 연결

v0.2 와 같은 형식(질문 1건 = 1쪽: 제목 → 질문 상자 2줄 → 근거 네 문장 + 표 1개, 부록, 답변 기록지)을 쓰되,
표와 수치는 노트북 실행 출력에서 그대로 가져온다 (계산을 이 스크립트에 다시 쓰지 않음).
- 표: 노트북 코드 셀 첫 줄이 '# [Q-태그]' 인 셀의 마지막 표 출력(text/html)을 캡처
- 수치: '# [Q-NUMS]' 셀이 print 한 JSON 을 읽어 질문 문장의 {이름} 자리에 넣음
- 질문 내용: question_spec_v03.py (QUESTIONS, APPENDIX)

사용: py make_question_docx_v03.py <노트북.ipynb> [spec.py] [out.docx]
필요: python-docx, Pillow, Microsoft Edge 또는 Chrome (표 캡처용 헤드리스 실행)
"""
import html
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops
from docx import Document
from docx.enum.table import WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # 리스테리아ML
NB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebook" / "4_기상_가공식품_타겟범위.ipynb"
SPEC_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "question_spec_v03.py"
OUT_DIR = ROOT / "docs" / "03_질문리스트"
CAP_DIR = OUT_DIR / "캡처_v0.3"
DOCX = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT_DIR / "리스테리아_질문리스트_v0.3.docx"

# ============================================================ 1. 노트북 출력 읽기
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
TAG_RE = re.compile(r"^\s*#\s*\[(Q-[A-Za-z0-9_-]+)\]")


def tagged_cells():
    out = {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        m = TAG_RE.match(src.splitlines()[0] if src else "")
        if m:
            out[m.group(1)] = c
    return out


CELLS = tagged_cells()


def table_html(tag):
    '''태그 셀의 마지막 표 출력(text/html). 에러 출력이 있으면 멈춤'''
    c = CELLS.get(tag)
    if c is None:
        raise KeyError(f"노트북에 '# [{tag}]' 셀이 없음")
    htmls = []
    for o in c.get("outputs", []):
        if o["output_type"] == "error":
            raise RuntimeError(f"{tag} 셀에 에러 출력: {o.get('evalue')}")
        if "data" in o and "text/html" in o["data"]:
            htmls.append("".join(o["data"]["text/html"]))
    if not htmls:
        raise ValueError(f"{tag} 셀에 표 출력이 없음")
    return htmls[-1]


def numbers():
    '''[Q-NUMS] 셀의 print 출력 JSON'''
    c = CELLS.get("Q-NUMS")
    if c is None:
        return {}
    text = "".join("".join(o["text"]) for o in c.get("outputs", []) if o["output_type"] == "stream")
    start = text.find("{")
    return json.loads(text[start:]) if start >= 0 else {}


NUMS = numbers()


class _Missing(dict):
    def __missing__(self, key):
        raise KeyError(f"질문 문장의 {{{key}}} 값이 [Q-NUMS] 출력에 없음")


def fill(text):
    return text.format_map(_Missing(NUMS)) if "{" in text else text


spec_mod = importlib.util.spec_from_file_location("qspec", SPEC_PATH)
qspec = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(qspec)
QUESTIONS, APPENDIX = qspec.QUESTIONS, getattr(qspec, "APPENDIX", [])
META = getattr(qspec, "META", {})

# ============================================================ 2. 표 캡처 (노트북 표 HTML → 헤드리스 브라우저 → PNG)
CSS = """
body{margin:0;padding:10px;background:#fff;font-family:'Malgun Gothic','맑은 고딕',sans-serif}
table{border-collapse:collapse;font-size:14px;color:#222}
thead th{border-bottom:2px solid #333;padding:5px 12px;text-align:right;font-weight:700;white-space:nowrap;vertical-align:bottom}
thead th .h{display:inline-block;max-width:7.5em;white-space:normal;word-break:keep-all;line-height:1.3}
thead tr:first-child th:first-child{text-align:left}
tbody th{padding:4px 12px;text-align:left;font-weight:700;white-space:nowrap;vertical-align:top}
tbody td{padding:4px 12px;text-align:right;white-space:nowrap}
tbody tr:nth-child(odd){background:#f4f4f4}
tbody tr.hl{background:#fff0b3}
td.l,th.l{text-align:left}
"""
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
BROWSER = next(p for p in BROWSERS if os.path.exists(p))
TMP = Path(tempfile.mkdtemp(prefix="qdoc3_"))


def prepare_table(raw, highlight=(), left_cols=0):
    '''pandas to_html 표를 캡처용으로 정리: 테두리 속성 제거, 강조 행 표시, 앞쪽 글자 열 왼쪽 정렬'''
    t = re.sub(r'<table[^>]*>', '<table>', raw)
    t = re.sub(r'<tr style="text-align: right;">', '<tr>', t)
    # 인덱스 이름만 있는 둘째 머리행(pandas 기본 모양)을 첫 머리행 첫 칸으로 합침 — 열 머리가 한 줄일 때만
    thead = re.search(r'<thead>(.*?)</thead>', t, flags=re.S)
    if thead:
        trs = re.findall(r'<tr>.*?</tr>', thead.group(1), flags=re.S)
        if len(trs) == 2:
            cells2 = re.findall(r'<th[^>]*>(.*?)</th>', trs[1], flags=re.S)
            if cells2 and cells2[0].strip() and all(not x.strip() for x in cells2[1:]):
                first_row = re.sub(r'<th[^>]*>\s*</th>', f'<th>{cells2[0]}</th>', trs[0], count=1)
                t = t.replace(thead.group(0), f'<thead>{first_row}</thead>')
    # 긴 열 머리는 낱말 사이에서 줄바꿈 (표 폭이 줄어 워드에서 글자가 작아지지 않게)
    thead = re.search(r'<thead>(.*?)</thead>', t, flags=re.S)
    if thead:
        seen = [0]

        def wrap_th(m):
            seen[0] += 1
            if seen[0] == 1 or not m.group(2).strip():          # 첫 칸(행 이름 머리)은 행 이름 폭을 따르므로 줄바꿈하지 않음
                return m.group(0)
            return f'{m.group(1)}<span class="h">{m.group(2)}</span>{m.group(3)}'
        wrapped = re.sub(r'(<th[^>]*>)(.*?)(</th>)', wrap_th, thead.group(1), flags=re.S)
        t = t.replace(thead.group(0), f'<thead>{wrapped}</thead>')
    if highlight:
        def mark(m):
            row = m.group(0)
            first = re.search(r'<th[^>]*>(.*?)</th>', row, flags=re.S)
            label = html.unescape(re.sub(r'<[^>]+>', '', first.group(1))).strip() if first else ''
            return row.replace('<tr>', '<tr class="hl">', 1) if label in highlight else row
        t = re.sub(r'<tr>.*?</tr>', mark, t, flags=re.S)
    if left_cols:
        def left(m):
            row = m.group(0)
            n = [0]

            def cell(mm):
                n[0] += 1
                return mm.group(0).replace('<td', '<td class="l"', 1) if n[0] <= left_cols else mm.group(0)
            return re.sub(r'<td[^>]*>', cell, row)
        t = re.sub(r'<tr[^>]*>.*?</tr>', left, t, flags=re.S)
        # 머리행도 같은 열을 왼쪽 정렬 (첫 칸 = 행 이름 머리 다음부터 left_cols 개)
        thead = re.search(r'<thead>(.*?)</thead>', t, flags=re.S)
        if thead:
            k = [0]

            def head(mm):
                k[0] += 1
                return mm.group(0).replace('<th', '<th class="l"', 1) if 2 <= k[0] <= left_cols + 1 else mm.group(0)
            t = t.replace(thead.group(0), '<thead>' + re.sub(r'<th[^>]*>', head, thead.group(1)) + '</thead>')
    return t


def capture(tag, name, highlight=(), left_cols=0):
    body = prepare_table(table_html(tag), highlight, left_cols)
    rows = body.count("<tr")
    page = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    n = len(list(TMP.glob("t*.html")))
    hp, shot, png = TMP / f"t{n}.html", TMP / f"t{n}.png", CAP_DIR / f"{name}.png"
    hp.write_text(page, encoding="utf-8")
    subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size=1800,{160 + 46 * (rows + 2)}", f"--user-data-dir={TMP / 'profile'}", f"--screenshot={shot}", hp.as_uri()],
                   capture_output=True, timeout=120, check=True)
    im = Image.open(shot).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    pad = 12
    im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height))).save(png)
    return png


# ============================================================ 3. 워드 도우미 (v0.2 와 같은 모양)
INK, BLUE, ORANGE, GRAY = "1F2937", "1F5FAF", "B45309", "6B7280"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(1.8)
TEXT_W = 16.6


def set_rfonts(rpr, name, east="Malgun Gothic"):
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), name)
    rf.set(qn("w:eastAsia"), east)


def style_font(name, size, bold=None, color=INK, before=None, after=None, line=None):
    st = doc.styles[name]
    set_rfonts(st.element.get_or_add_rPr(), "Malgun Gothic")
    st.font.size = Pt(size)
    if bold is not None:
        st.font.bold = bold
    st.font.color.rgb = RGBColor.from_string(color)
    pf = st.paragraph_format
    if before is not None:
        pf.space_before = Pt(before)
    if after is not None:
        pf.space_after = Pt(after)
    if line is not None:
        pf.line_spacing = line
    return st


style_font("Normal", 10.5, after=4, line=1.3)
style_font("Heading 1", 15, bold=True, color="111827", before=0, after=4)
style_font("List Bullet", 10.5, after=3, line=1.3)
TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def rich(p, text, size=None, color=None, bold=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1])
            set_rfonts(r._element.get_or_add_rPr(), "Consolas")
            r.font.size = Pt((size or 10.5) - 1)
            r.font.color.rgb = RGBColor.from_string("7C2D12")
            continue
        strong = part.startswith("**") and part.endswith("**")
        r = p.add_run(part[2:-2] if strong else part)
        r.bold = True if (strong or bold) else None
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p


def para(text="", size=None, color=None, bold=False, align=None, before=None, after=None):
    p = doc.add_paragraph()
    rich(p, text, size, color, bold)
    if align is not None:
        p.alignment = align
    if before is not None:
        p.paragraph_format.space_before = Pt(before)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullet(text):
    return rich(doc.add_paragraph(style="List Bullet"), text)


def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def cell_borders(cell, **edges):
    bd = OxmlElement("w:tcBorders")
    for e in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{e}")
        if e in edges:
            sz, color = edges[e]
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), color)
        else:
            el.set(qn("w:val"), "nil")
        bd.append(el)
    cell._tc.get_or_add_tcPr().append(bd)


def table_setup(t, widths, margins=(70, 120, 70, 120)):
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = t._tbl.tblPr
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:w"), str(int(sum(widths) * 567)))
    tw.set(qn("w:type"), "dxa")
    tblPr.append(tw)
    mar = OxmlElement("w:tblCellMar")
    for k, v in zip(("top", "left", "bottom", "right"), margins):
        el = OxmlElement(f"w:{k}")
        el.set(qn("w:w"), str(v))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)
    for i, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(widths[i] * 567)))
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = Cm(widths[i])


def box(fill, bar, fn, margins=(110, 220, 110, 180)):
    t = doc.add_table(rows=1, cols=1)
    table_setup(t, [TEXT_W], margins=margins)
    c = t.rows[0].cells[0]
    shade(c, fill)
    cell_borders(c, left=(24, bar))
    first = c.paragraphs[0]
    fn(c)
    if not first.text and len(c.paragraphs) > 1:
        first._element.getparent().remove(first._element)
    para(after=2)


def question_box(items):
    def fillbox(c):
        for who, color, text in items:
            p = c.add_paragraph()
            r = p.add_run(who)
            r.bold = True
            r.font.color.rgb = RGBColor.from_string(color)
            p.add_run("   ")
            rich(p, text)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.3
    box("EEF4FB", BLUE, fillbox)


def code_box(text):
    def fillbox(c):
        for line in text.strip("\n").splitlines():
            p = c.add_paragraph()
            r = p.add_run(line if line else " ")
            set_rfonts(r._element.get_or_add_rPr(), "Consolas")
            r.font.size = Pt(8.5)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
    box("F3F4F6", "9CA3AF", fillbox, margins=(90, 160, 90, 120))


def step(n, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.55)
    p.paragraph_format.first_line_indent = Cm(-0.55)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(f"{n}. ")
    r.bold = True
    r = p.add_run(label)
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(BLUE)
    p.add_run(" — ")
    rich(p, text)
    return p


def figure(png, caption, max_w=TEXT_W - 0.6, max_h=6.8):
    '''표 그림: 원래 크기(2배 캡처 기준)를 넘지 않게, 쪽 폭과 높이 상한(cm) 안에 맞춤 — 질문 1건이 한 쪽을 넘지 않게'''
    im = Image.open(png)
    w = min(max_w, im.width / 2 / 96 * 2.54)
    w = min(w, max_h * im.width / im.height)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(str(png), width=Cm(w))
    para(caption, size=8.5, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)


def label(text, before=10, size=11):
    return para(text, size=size, bold=True, before=before, after=4)


def new_page_heading(text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.page_break_before = True
    return h


def footer(title):
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run(f"{title} · ")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string(GRAY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    rr = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "16")
    rpr.append(sz)
    rr.append(rpr)
    tt = OxmlElement("w:t")
    tt.text = "1"
    rr.append(tt)
    fld.append(rr)
    fp._p.append(fld)


# ============================================================ 4. 문서 조립
def build():
    CAP_DIR.mkdir(parents=True, exist_ok=True)
    footer(META.get("footer", "리스테리아 예측 모델 질문 리스트 v0.3"))
    who = {"data": ("데이터 담당자", BLUE), "expert": ("도메인 전문가", ORANGE)}

    # 첫 쪽
    para(META.get("title", "리스테리아 예측 모델 질문 리스트"), size=20, bold=True, color="111827", after=2)
    para(fill(META.get("subtitle", "")), color=GRAY, after=14)
    label("이 문서는", before=0)
    for b in META.get("intro", []):
        bullet(fill(b))
    groups = []
    for q in QUESTIONS:
        if q["group"] not in groups:
            groups.append(q["group"])
    for g in groups:
        label(g, before=12)
        for q in QUESTIONS:
            if q["group"] == g:
                bullet(fill(q["title"]))

    # 질문 쪽
    for q in QUESTIONS:
        new_page_heading(fill(q["title"]))
        para("결정할 것: " + fill(q["decide"]), color=GRAY, after=8)
        items = []
        if q.get("q_data"):
            items.append((*who["data"], fill(q["q_data"])))
        if q.get("q_expert"):
            items.append((*who["expert"], fill(q["q_expert"])))
        question_box(items)
        label("근거", before=4)
        for i, (lab, text) in enumerate(q["steps"], start=1):
            step(i, lab, fill(text))
            if i == 2 and q.get("table"):
                t = q["table"]
                png = capture(t["tag"], f"{q['id']}_{t['tag']}", tuple(t.get("highlight", ())), t.get("left_cols", 0))
                figure(png, f"표 {q['id']}. " + fill(t["caption"]))

    # 부록
    if APPENDIX:
        new_page_heading("부록. 사용한 방법과 규칙")
        for block in APPENDIX:
            if block.get("label"):
                label(fill(block["label"]), before=6)
            if block.get("text"):
                para(fill(block["text"]), after=4)
            if block.get("code"):
                code_box(block["code"])
            for b in block.get("bullets", []):
                bullet(fill(b))

    # 답변 기록지
    new_page_heading("답변 기록지")
    para("답변자·일자와 함께 적어 주시면 모델링 결정의 근거로 보관합니다.", after=8)
    rows = []
    for q in QUESTIONS:
        if q.get("q_data"):
            rows.append((q["id"], "데이터 담당자"))
        if q.get("q_expert"):
            rows.append((q["id"], "도메인 전문가"))
    t = doc.add_table(rows=1 + len(rows), cols=3)
    t.style = doc.styles["Table Grid"]
    table_setup(t, [4.2, 3.0, 9.4], margins=(80, 120, 80, 120))
    for j, h in enumerate(["질문", "답변자 · 일자", "답변"]):
        c = t.rows[0].cells[j]
        rich(c.paragraphs[0], h, size=10, bold=True)
        shade(c, "E5E7EB")
    for i, (qid, whom) in enumerate(rows, start=1):
        rich(t.rows[i].cells[0].paragraphs[0], f"**{qid}** · {whom}", size=10)
        t.rows[i].height = Cm(2.2)
        t.rows[i].height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if ri == 0:
            trPr.append(OxmlElement("w:tblHeader"))
        for c in row.cells:
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            cell_borders(c, top=(4, "C7CCD4"), left=(4, "C7CCD4"), bottom=(4, "C7CCD4"), right=(4, "C7CCD4"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(DOCX)
    print("saved:", DOCX, "/ questions:", len(QUESTIONS), "/ captures:", len(list(CAP_DIR.glob("*.png"))))


if __name__ == "__main__":
    build()
