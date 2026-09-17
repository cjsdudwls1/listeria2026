# -*- coding: utf-8 -*-
"""데이터 담당자·도메인 전문가 질문 리스트(워드) 생성 — 예시 3건, v0.2 (단순화판)

질문 1건 = 1쪽: 제목 → 질문 상자(담당자 1줄·전문가 1줄) → 근거 네 문장(데이터·작업·결과·그래서) + 표 1개.
SQL·임의 분류 규칙은 부록, 답변은 마지막 쪽 답변 기록지.
2.ipynb 와 같은 입력·같은 로직으로 표를 다시 계산하고 핵심 수치는 assert 로 대조한다 (원자료 가공 없음).

입력 (읽기만): data/listeria_body.parquet, data/collected/nb2_db_attr.parquet,
               nb2_db_codes.csv, nb2_db_s_org_listeria.parquet  (DB 추출: scripts/nb2_extract_db.py, SELECT 만)
출력: docs/03_질문리스트/이전버전/리스테리아_질문리스트_예시3건_v0.2.docx, docs/03_질문리스트/이전버전/캡처/표*.png
필요: python-docx, Pillow, Microsoft Edge 또는 Chrome (표 캡처용 헤드리스 실행)
사용: py make_question_docx.py
"""
import html
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from docx import Document
from docx.enum.table import WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]          # 리스테리아ML
DATA, COL = ROOT / "data", ROOT / "data" / "collected"
OUT_DIR = ROOT / "docs" / "03_질문리스트" / "이전버전"
CAP_DIR = OUT_DIR / "캡처"
DOCX = OUT_DIR / "리스테리아_질문리스트_예시3건_v0.2.docx"
VERSION = "v0.2"
TODAY = "2026-09-14"

# ============================================================ 1. 데이터 (2.ipynb 1-3절과 같은 결합)
df = pd.read_parquet(DATA / "listeria_body.parquet")
attr = pd.read_parquet(COL / "nb2_db_attr.parquet")
codes = pd.read_csv(COL / "nb2_db_codes.csv", dtype=str)
sorg = pd.read_parquet(COL / "nb2_db_s_org_listeria.parquet", columns=["lims_seq", "rcept_no", "shipng_de"])

keep = ["lims_data_key", "judge_result_db", "mnfctur_de_raw", "reqest_de", "mnfctur_de_or_fallback", "cntc_sys_key2",
        "addr_source_db", "inspct_purps_code", "reqest_sys_code", "in_m_org"]
dfx = df.merge(attr[keep], on="lims_data_key", how="left", validate="one_to_one", indicator=True)
assert len(dfx) == 676_537 and (dfx["_merge"] == "both").all()
dfx["pos"] = dfx["judge_result"].astype(str).eq("검출").astype(int)
N, K = len(dfx), int(dfx["pos"].sum())
assert K == 1_969 and (dfx["judge_result"].astype(str) == dfx["judge_result_db"].astype(str)).all()
code_nm = codes[codes["cmmn_cl_code"].isin(["IM02", "CI07"])].set_index("cmmn_code")["cmmn_code_nm"].to_dict()
dfx["검사목적"] = dfx["inspct_purps_code"].map(code_nm).fillna("(원본LIMS 없음)")
dfx["의뢰시스템"] = dfx["reqest_sys_code"].map(code_nm).fillna("(결측)")
dfx["국내해외"] = dfx["src_route"].astype(str).map({"domestic": "국내", "overseas": "해외"})
for c in ["mnfctur_de", "mnfctur_de_raw", "reqest_de", "mnfctur_de_or_fallback"]:
    dfx[c] = pd.to_datetime(dfx[c])
m_link = float(dfx["in_m_org"].astype(bool).mean())
not_in = attr[~attr["lims_data_key"].isin(df["lims_data_key"])]
n_db, n_not_in, n_not_in_lab = len(attr), len(not_in), int((not_in["addr_source_db"].astype(str) == "검사기관").sum())

# ---------- 표 1 (Q1): 검사목적별 — 주요 4개 + 그 외 + 합계
g1 = dfx.groupby("검사목적").agg(건수=("pos", "size"), 검출=("pos", "sum"), 자체=("의뢰시스템", lambda s: int((s == "자체의뢰").sum())))
MAIN = ["기타", "자가품질위탁검사", "수입검사", "수거검사"]
rest = g1.drop(index=MAIN)
t1 = g1.loc[MAIN].copy()
t1.loc[f"그 외 {len(rest)}개 합산"] = rest.sum()
t1.loc["합계"] = g1.sum()
T1 = pd.DataFrame({"검사 건수": t1["건수"], "검출": t1["검출"], "검출 비중(%)": t1["검출"] / K * 100,
                   "의뢰시스템 '자체의뢰' 비율(%)": t1["자체"] / t1["건수"] * 100})
T1.index.name = "검사목적"
assert T1.loc["기타", "검출"] == 1_619 and T1.loc["수거검사", "검출"] == 66 and T1.loc["합계", "검사 건수"] == N
Q1 = dict(k_etc=int(T1.loc["기타", "검출"]), share_etc=T1.loc["기타", "검출 비중(%)"], self_etc=T1.loc["기타", "의뢰시스템 '자체의뢰' 비율(%)"],
          k_col=int(T1.loc["수거검사", "검출"]), share_col=T1.loc["수거검사", "검출 비중(%)"])

# ---------- 표 2 (Q2): 두 기준규격외 코드 × 작성자 임의 키워드 분류


def has_kw(s, pattern):
    x = s.astype("string").str.lower().str.replace(r"\s+", "", regex=True)
    return x.str.contains(pattern, regex=True, na=False).astype(bool)


PAT = {  # 2.ipynb 1-3절과 같은 규칙. 위에서부터 먼저 걸리는 분류 하나
    "환경검체": r"바닥|벽체|벽면|표면|장갑|도마|칼날|컨베이어|컨베어|배수|대차|선반|작업대|손잡이|스왑|swab|공중낙하|낙하균|천장|환경",
    "팽이버섯": r"팽이|enoki|flammulina",
    "공정시료": r"세척전|세척후|가열전|가열후|살균전|살균후|입고|원료육|반제품|공정|숙성전|숙성후|포장전|포장후|해동|절단전|절단후|냉각",
    "완제품 표기": r"완제품",
}
flags = {k: has_kw(dfx["prduct_nm"], p) for k, p in PAT.items()}
dfx["임의 분류"] = np.select([flags["환경검체"], flags["팽이버섯"], flags["공정시료"], flags["완제품 표기"]],
                          ["환경검체", "팽이버섯", "공정시료", "완제품 표기"], "그 외 제품명")
Z27, Z38 = "정액,수정란,종란", "기타기준규격외"
zsub = dfx[dfx["train_prdlst_nm"].astype(str).isin([Z27, Z38])]
assert zsub.loc[zsub["train_prdlst_nm"].astype(str) == Z27, "prdlst_code"].astype(str).eq("Z2700400000000").all()
assert zsub.loc[zsub["train_prdlst_nm"].astype(str) == Z38, "prdlst_code"].astype(str).eq("Z3800100000000").all()
ct = zsub.groupby(["임의 분류", zsub["train_prdlst_nm"].astype(str)])["pos"].agg(["size", "sum"])
ORDER2 = ["팽이버섯", "환경검체", "공정시료", "완제품 표기", "그 외 제품명"]


def cell2(kind, code):
    n, k = (int(ct.loc[(kind, code), "size"]), int(ct.loc[(kind, code), "sum"])) if (kind, code) in ct.index else (0, 0)
    return f"{n:,}행 / {k:,}건"


T2 = pd.DataFrame({f"'{Z27}' 코드": [cell2(k, Z27) for k in ORDER2], f"'{Z38}' 코드": [cell2(k, Z38) for k in ORDER2]}, index=ORDER2)
z_tot = zsub.groupby(zsub["train_prdlst_nm"].astype(str))["pos"].agg(["size", "sum"])
T2.loc["합계"] = [f"{int(z_tot.loc[Z27, 'size']):,}행 / {int(z_tot.loc[Z27, 'sum']):,}건", f"{int(z_tot.loc[Z38, 'size']):,}행 / {int(z_tot.loc[Z38, 'sum']):,}건"]
T2.index.name = "임의 분류 (제품명 키워드)"
assert int(ct.loc[("팽이버섯", Z27), "sum"]) == 415 and int(ct.loc[("환경검체", Z27), "sum"]) == 94
assert int(z_tot.loc[Z27, "sum"]) == 640 and int(z_tot.loc[Z38, "sum"]) == 791
top27 = zsub.loc[zsub["train_prdlst_nm"].astype(str) == Z27, "prduct_nm"].value_counts().head(15).index
top38 = zsub.loc[zsub["train_prdlst_nm"].astype(str) == Z38, "prduct_nm"].value_counts().head(15).index
for nm in ["수출용 팽이버섯 (2-1번)", "포장실 바닥", "수확실 벽체"]:
    assert nm in top27, nm
for nm in ["원료육", "배추(세척전)"]:
    assert nm in top38, nm
kind_k = zsub.groupby("임의 분류")["pos"].sum()
Q2 = dict(k_both=int(z_tot["sum"].sum()), share_both=z_tot["sum"].sum() / K * 100,
          k_enoki=int(kind_k.get("팽이버섯", 0)), k_env=int(kind_k.get("환경검체", 0)), k_proc=int(kind_k.get("공정시료", 0)))

# ---------- 표 3 (Q3): 기준일 출처별 — DB 날짜와 일치율, 주말 비율
S = sorg.sort_values("lims_seq").drop_duplicates("rcept_no").set_index("rcept_no")
assert sorg.groupby("rcept_no")["shipng_de"].nunique(dropna=True).le(1).all()      # 같은 접수번호 안의 선적일은 하나
b = dfx["date_basis"].astype(str)
ov_ship = dfx[(dfx["국내해외"] == "해외") & (b == "선적일")].copy()
ov_ship["s_shipng_de"] = pd.to_datetime(ov_ship["cntc_sys_key2"].map(S["shipng_de"]))
weekend = dfx["mnfctur_de"].dt.dayofweek.ge(5)
rows3 = [
    ("제조일", int((b == "제조일").sum()), "원본 제조일 (mnfctur_de)",
     (dfx.loc[b == "제조일", "mnfctur_de"] == dfx.loc[b == "제조일", "mnfctur_de_raw"]).mean() * 100, weekend[b == "제조일"].mean() * 100),
    ("의뢰일", int((b == "의뢰일").sum()), "의뢰일 (reqest_de)",
     (dfx.loc[b == "의뢰일", "mnfctur_de"] == dfx.loc[b == "의뢰일", "reqest_de"]).mean() * 100, weekend[b == "의뢰일"].mean() * 100),
    ("선적일", int((b == "선적일").sum()), "수입 LIMS 선적일 (shipng_de)",
     (ov_ship["mnfctur_de"] == ov_ship["s_shipng_de"]).mean() * 100, weekend[b == "선적일"].mean() * 100),
]
T3 = pd.DataFrame(rows3, columns=["기준일 출처 (date_basis)", "행 수", "값이 같은 DB 날짜", "일치율(%)", "주말 비율(%)"]).set_index("기준일 출처 (date_basis)")
same_fallback = (dfx["mnfctur_de"] == dfx["mnfctur_de_or_fallback"]).mean() * 100
assert (T3["일치율(%)"] == 100).all() and same_fallback == 100, T3
dom = dfx["국내해외"] == "국내"
Q3 = dict(n_ov_ship=len(ov_ship), dom_req_share=((b == "의뢰일") & dom).sum() / dom.sum(),
          wk_mfg=T3.loc["제조일", "주말 비율(%)"], wk_req=T3.loc["의뢰일", "주말 비율(%)"], wk_ship=T3.loc["선적일", "주말 비율(%)"])

# ============================================================ 2. 결과 표 캡처 (HTML → 헤드리스 브라우저 → PNG)
CSS = """
body{margin:0;padding:10px;background:#fff;font-family:'Malgun Gothic','맑은 고딕',sans-serif}
table{border-collapse:collapse;font-size:15px;color:#222}
thead th{border-bottom:2px solid #333;padding:7px 14px;text-align:right;font-weight:700;white-space:nowrap}
thead th.l{text-align:left}
tbody th{padding:7px 14px;text-align:left;font-weight:700;white-space:nowrap}
tbody td{padding:7px 14px;text-align:right;white-space:nowrap}
tbody td.l{text-align:left}
tbody tr:nth-child(odd){background:#f4f4f4}
tbody tr.hl{background:#fff0b3}
tbody tr.total th,tbody tr.total td{border-top:1.5px solid #999;font-weight:700;background:#fff}
"""
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
BROWSER = next(p for p in BROWSERS if os.path.exists(p))
TMP = Path(tempfile.mkdtemp(prefix="qdoc_"))


def capture(frame, name, fmts=None, highlight=(), left_cols=(), total_row="합계"):
    """DataFrame → 표 이미지. highlight: 강조할 행, total_row: 굵게 구분할 합계 행"""
    fmts = fmts or {}
    head = f'<th class="l">{html.escape(str(frame.index.name or ""))}</th>'
    head += "".join(f'<th class="{"l" if c in left_cols else ""}">{html.escape(str(c))}</th>' for c in frame.columns)
    rows = []
    for i in range(len(frame)):
        key = frame.index[i]
        tds = []
        for j, c in enumerate(frame.columns):
            v = frame.iat[i, j]
            txt = v if isinstance(v, str) else (fmts[c].format(v) if c in fmts else f"{v:,}")
            tds.append(f'<td class="{"l" if c in left_cols else ""}">{html.escape(txt)}</td>')
        cls = "total" if key == total_row else ("hl" if key in highlight else "")
        rows.append(f'<tr class="{cls}"><th>{html.escape(str(key))}</th>{"".join(tds)}</tr>')
    page = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"
    n = len(list(TMP.glob("t*.html")))
    hp, shot, png = TMP / f"t{n}.html", TMP / f"t{n}.png", CAP_DIR / f"{name}.png"   # 브라우저에는 영문 경로만 넘김
    hp.write_text(page, encoding="utf-8")
    subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size=1600,{120 + 44 * (len(frame) + 2)}", f"--user-data-dir={TMP / 'profile'}", f"--screenshot={shot}", hp.as_uri()],
                   capture_output=True, timeout=120, check=True)
    im = Image.open(shot).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    pad = 12
    im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height))).save(png)
    return png


CAP_DIR.mkdir(parents=True, exist_ok=True)
IMG = {
    1: capture(T1, "표1_검사목적별_검출", {"검사 건수": "{:,.0f}", "검출": "{:,.0f}", "검출 비중(%)": "{:.1f}",
                                      "의뢰시스템 '자체의뢰' 비율(%)": "{:.1f}"}, highlight=("기타", "수거검사")),
    2: capture(T2, "표2_기준규격외코드_임의분류"),
    3: capture(T3, "표3_기준일출처_DB날짜_일치", {"행 수": "{:,}", "일치율(%)": "{:.1f}", "주말 비율(%)": "{:.1f}"},
               left_cols=("값이 같은 DB 날짜",)),
}

# ============================================================ 3. 워드
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


def para(text="", size=None, color=None, bold=False, align=None, before=None, after=None, container=None):
    p = (container or doc).add_paragraph()
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
    def fill(c):
        for who, color, text in items:
            p = c.add_paragraph()
            r = p.add_run(who)
            r.bold = True
            r.font.color.rgb = RGBColor.from_string(color)
            p.add_run("   ")
            rich(p, text)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.3
    box("EEF4FB", BLUE, fill)


def sql_box(sql):
    def fill(c):
        for line in sql.strip("\n").splitlines():
            p = c.add_paragraph()
            r = p.add_run(line if line else " ")
            set_rfonts(r._element.get_or_add_rPr(), "Consolas")
            r.font.size = Pt(8.5)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
    box("F3F4F6", "9CA3AF", fill, margins=(90, 160, 90, 120))


def step(n, label, text):
    """근거 한 문장: '1. 데이터 — …' (내어쓰기)"""
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


def figure(no, caption, max_w=TEXT_W - 0.6):
    im = Image.open(IMG[no])
    w = min(max_w, im.width / 2 / 96 * 2.54)       # 캡처 배율 2 기준 원래 크기, 넓으면 줄임
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(str(IMG[no]), width=Cm(w))
    para(f"표 {no}. {caption}", size=8.5, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)


def label(text, before=10):
    return para(text, size=11, bold=True, before=before, after=4)


def new_page_heading(text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.page_break_before = True
    return h


def footer():
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run(f"리스테리아 예측 모델 질문 리스트 (예시 {VERSION}) · ")
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


footer()
DATA_OWNER, EXPERT = ("데이터 담당자", BLUE), ("도메인 전문가", ORANGE)
TITLE = {1: f"Q1. 검출의 {Q1['share_etc']:.0f}%가 검사목적 '기타' — 예측 대상에 포함하나?",
         2: "Q2. 공정·환경·팽이버섯 검체로 보이는 행 — 학습에서 빼나?",
         3: "Q3. 기준일에 날짜 세 종류가 섞여 보임 — 어느 날짜를 쓰나?"}

# ---------------------------------------------------------------- 첫 쪽
para("리스테리아 예측 모델 질문 리스트", size=20, bold=True, color="111827", after=2)
para(f"예시 3건 · {VERSION} · {TODAY} · 근거: 2.ipynb", color=GRAY, after=14)
label("이 문서는", before=0)
bullet("모델링 결정(학습 대상, 뺄 데이터, 날짜 사용) 전에 **데이터 담당자·도메인 전문가의 공식 의견**을 받기 위한 질문")
bullet("문서 안의 해석은 모두 작성자의 **추정**. 데이터를 붙이거나 센 방법이 틀렸다면 지적 부탁")
bullet("답변은 마지막 쪽 **답변 기록지**에 적어 주시면 모델링 근거로 보관")
label("질문", before=14)
for i in (1, 2, 3):
    bullet(TITLE[i])

# ---------------------------------------------------------------- Q1
new_page_heading(TITLE[1])
para("결정할 것: 학습 대상 범위 (전체 검사 / 정부 검사만 / 나눠서 학습)", color=GRAY, after=8)
question_box([
    (*DATA_OWNER, "검사목적 코드 '기타'(`IM02000007`)는 어떤 경우에 입력되나요?"),
    (*EXPERT, "'기타'가 업체가 스스로 맡긴 검사라면 예측 대상에 넣어도 되나요? 정부 수거·수입 검사와 한 모델로 섞어도 되나요?"),
])
label("근거", before=4)
step(1, "데이터", f"학습용 데이터(`listeria_body.parquet`, {N:,}행)에 원본 LIMS 테이블(`ba_data.m_org_lims_data_info`)을 `lims_data_key`로 붙여 "
                 f"검사목적·의뢰시스템 코드를 가져옴 ({m_link:.1%} 연결). 코드 이름은 공통코드 테이블(`ba_data.t_cm_cmmn_code`)에서 붙임")
step(2, "작업", "검사목적별로 검사 건수, 검출 수, 의뢰시스템이 '자체의뢰'인 비율을 셈 → 표 1")
figure(1, "검사목적별 검사 건수·검출 (2.ipynb 2-2절)")
step(3, "결과", f"'기타'에서 검출 {Q1['k_etc']:,}건(전체의 {Q1['share_etc']:.1f}%)이 나오고, '기타'의 {Q1['self_etc']:.1f}%는 의뢰시스템이 '자체의뢰'. "
               f"공무원 수거검사는 검출 {Q1['k_col']}건({Q1['share_col']:.1f}%)")
step(4, "그래서", "**(추정)** '기타'는 업체가 스스로 맡긴 검사로 보임. 그대로 학습하면 모델이 '어떤 식품이 위험한가'보다 "
                 "'누가 왜 검사했나'를 배울 수 있어, 학습 대상 범위를 먼저 정해야 함")

# ---------------------------------------------------------------- Q2
new_page_heading(TITLE[2])
para("결정할 것: 기준규격외 코드 안에서 학습에서 뺄 행 / 따로 둘 행", color=GRAY, after=8)
question_box([
    (*DATA_OWNER, "팽이버섯·공정 시료·작업장 검체가 '정액,수정란,종란'(`Z2700400000000`)·'기타기준규격외'(`Z3800100000000`) 코드로 들어온 입력 기준은 무엇인가요?"),
    (*EXPERT, "이런 시료를 예측 대상에서 빼야 하나요, 따로 둬야 하나요? 표 2의 임의 분류 기준이 맞는지도 봐 주세요."),
])
label("근거", before=4)
step(1, "데이터", "조인 없음. 학습용 데이터의 품목코드(`prdlst_code`)·학습용 품목명(`train_prdlst_nm`)·제품명(`prduct_nm`)·판정(`judge_result`)만 사용")
step(2, "작업", "검출이 가장 많은 두 코드의 제품명을 작성자가 **임의 키워드**로 분류함 "
               "(예: '바닥·장갑·스왑' → 환경검체, '팽이·enoki' → 팽이버섯, '세척전·가열후·원료육' → 공정시료. 전체 규칙은 부록) → 표 2")
figure(2, "두 기준규격외 코드의 제품명 임의 분류 — 칸 = 행 수 / 검출 (2.ipynb 3-3·3-4절)", max_w=13.0)
step(3, "결과", f"두 코드에서 전체 검출의 {Q2['share_both']:.1f}%({Q2['k_both']:,}건)가 나옴. 제품명은 '수출용 팽이버섯', '포장실 바닥', '원료육', "
               f"'배추(세척전)' 같은 이름이 많고, 임의 분류로는 팽이버섯 {Q2['k_enoki']}건·환경검체 {Q2['k_env']}건·공정시료 {Q2['k_proc']}건이 검출됨")
step(4, "그래서", "**(추정)** 식품 완제품이 아닌 수출용 팽이버섯·공정 시료·작업장 검체가 많이 섞인 것으로 보임. "
                 "그대로 학습하면 식품의 위험보다 공장 점검·수출검사 관행을 배울 수 있음")

# ---------------------------------------------------------------- Q3
new_page_heading(TITLE[3])
para("결정할 것: 월·계절 변수와 학습/검증 기간 나누기에 쓸 날짜 (기준일 = `mnfctur_de`)", color=GRAY, after=8)
question_box([
    (*DATA_OWNER, "`mnfctur_de`(기준일)는 제조일이 없으면 의뢰일·수입 선적일로 채운 값이 맞나요? `date_basis` 최종 규칙은 무엇인가요? "
                  "(정의서: 제조일/의뢰일/수거일, 실제 값: 제조일/의뢰일/선적일)"),
    (*EXPERT, f"리스테리아 검출을 시기(월·계절)로 볼 때 기준 날짜는 제조일과 검사 의뢰일 중 무엇이 맞나요? "
              f"제조일이 없는 행(국내 행의 {Q3['dom_req_share']:.0%})은 의뢰일로 대신해도 되나요?"),
])
label("근거", before=4)
step(1, "데이터", "학습용 데이터의 기준일을 DB `ready_data.hazard_preprocessed_data`의 원본 제조일·의뢰일과 `lims_data_key`로 붙여 비교. "
                 "수입 행은 수입 LIMS `ba_data.s_org_lims_data_info`의 선적일과 접수번호로 붙여 비교 (컬럼은 부록)")
step(2, "작업", "기준일 출처(`date_basis`)별로 기준일이 어느 DB 날짜와 같은지, 주말 비율을 셈 → 표 3")
figure(3, f"기준일 출처별 DB 날짜 일치율·주말 비율 (2.ipynb 4-1·4-2·4-5절, 선적일 일치율은 연결된 해외 {Q3['n_ov_ship']:,}행 기준)")
step(3, "결과", f"세 값이 각각 원본 제조일·의뢰일·수입 선적일과 100% 같음. 주말 비율도 제조일 {Q3['wk_mfg']:.1f}%, 의뢰일 {Q3['wk_req']:.1f}%, "
               f"선적일 {Q3['wk_ship']:.1f}%로 서로 다름")
step(4, "그래서", "**(추정)** 한 컬럼에 서로 다른 세 사건의 날짜가 섞여 있음. 같은 '7월'이라도 행마다 제조한 달·검사를 맡긴 달·선적한 달로 "
                 "뜻이 달라, 어느 날짜를 쓸지 정해야 함")

# ---------------------------------------------------------------- 부록
new_page_heading("부록. 사용한 SQL과 임의 분류 규칙")
para(f"데이터 담당자 확인용. DB `TOXIN_MODEL_PRED` 에서 SELECT 로만 조회 ({TODAY}, `scripts/nb2_extract_db.py`, 질문에 쓴 컬럼만 표시)", after=6)
sql_box(f"""
-- 1) 학습용 데이터 행에 원본 LIMS 코드와 DB 날짜 붙이기 (Q1, Q3)
SELECT h.lims_data_key,
       h.mnfctur_de, h.reqest_de, h.mnfctur_de_or_fallback, h.cntc_sys_key2,
       m.inspct_purps_code, m.reqest_sys_code
FROM ready_data.hazard_preprocessed_data h
LEFT JOIN ba_data.m_org_lims_data_info m ON m.lims_data_key = h.lims_data_key
WHERE h.hazard_label = '리스테리아';                          -- {n_db:,}행

-- 2) 코드 이름 (Q1)
SELECT cmmn_cl_code, cmmn_code, cmmn_code_nm
FROM ba_data.t_cm_cmmn_code
WHERE cmmn_cl_code IN ('IM02','IM03','IM08','IM11','IM40')
   OR cmmn_code = ANY(:실제 의뢰시스템 코드값);

-- 3) 수입 선적일 (Q3) — 접수번호마다 lims_seq 가 가장 작은 행 1개 사용
SELECT lims_seq, rcept_no, shipng_de
FROM ba_data.s_org_lims_data_info
WHERE label = '리스테리아';                                    -- {len(sorg):,}행
""")
para(f"학습용 데이터 {N:,}행과는 `lims_data_key`로 1:1 연결. DB 에만 있는 {n_not_in:,}행은 집계에서 빠짐 (이 중 {n_not_in_lab:,}행은 주소 출처가 '검사기관')", after=10)
label("Q2 임의 분류 규칙", before=4)
para("제품명의 공백을 없애고 소문자로 바꾼 뒤, 아래 단어가 들어 있으면 분류 (여러 개가 걸리면 위쪽 분류 하나)", after=4)
for k, p in PAT.items():
    para(f"**{k}** : " + p.replace("|", ", "), after=2)
para("**그 외 제품명** : 위 단어 없음", after=2)

# ---------------------------------------------------------------- 답변 기록지
new_page_heading("답변 기록지")
para("답변자·일자와 함께 적어 주시면 모델링 결정의 근거로 보관합니다.", after=8)
ANS = [("Q1", "데이터 담당자"), ("Q1", "도메인 전문가"), ("Q2", "데이터 담당자"), ("Q2", "도메인 전문가"), ("Q3", "데이터 담당자"), ("Q3", "도메인 전문가")]
t = doc.add_table(rows=1 + len(ANS), cols=3)
t.style = doc.styles["Table Grid"]
widths = [4.2, 3.0, 9.4]
table_setup(t, widths, margins=(80, 120, 80, 120))
for j, h in enumerate(["질문", "답변자 · 일자", "답변"]):
    c = t.rows[0].cells[j]
    rich(c.paragraphs[0], h, size=10, bold=True)
    shade(c, "E5E7EB")
for i, (q, who) in enumerate(ANS, start=1):
    rich(t.rows[i].cells[0].paragraphs[0], f"**{q}** · {who}", size=10)
    t.rows[i].height = Cm(2.6)
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
print("saved:", DOCX)
for k, v in IMG.items():
    print(f"  표{k}: {v.name} {Image.open(v).size}")
print(T1.round(2).to_string())
print(T2.to_string())
print(T3.round(2).to_string())
print("Q1", {k: round(float(v), 3) for k, v in Q1.items()})
print("Q2", {k: round(float(v), 3) for k, v in Q2.items()})
print("Q3", {k: round(float(v), 4) for k, v in Q3.items()})
