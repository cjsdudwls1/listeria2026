# -*- coding: utf-8 -*-
"""데이터 담당자·도메인 전문가 질문 리스트(워드) 생성 — 예시 3건 (검수용 초안 v0.1)

2.ipynb 와 같은 입력·같은 로직으로 표를 다시 계산하고, 결과 표를 캡처 이미지(PNG)로 만들어 워드에 넣는다.
핵심 수치는 2.ipynb 출력값과 assert 로 대조한다.

입력 (읽기만): data/listeria_body.parquet, data/collected/nb2_db_attr.parquet,
               nb2_db_codes.csv, nb2_db_s_org_listeria.parquet  (DB 추출: scripts/nb2_extract_db.py, SELECT 만)
출력: docs/질문리스트/리스테리아_질문리스트_예시3건_v0.1.docx, docs/질문리스트/캡처/*.png
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
OUT_DIR = ROOT / "docs" / "질문리스트"
CAP_DIR = OUT_DIR / "캡처"
DOCX = OUT_DIR / "리스테리아_질문리스트_예시3건_v0.1.docx"
VERSION = "v0.1 (예시 3건, 검수용 초안)"
TODAY = "2026-09-14"

# ============================================================ 1. 데이터 (2.ipynb 1-3절과 같은 결합)
df = pd.read_parquet(DATA / "listeria_body.parquet")
attr = pd.read_parquet(COL / "nb2_db_attr.parquet")
codes = pd.read_csv(COL / "nb2_db_codes.csv", dtype=str)
sorg = pd.read_parquet(COL / "nb2_db_s_org_listeria.parquet", columns=["lims_seq", "rcept_no", "shipng_de"])

keep = ["lims_data_key", "judge_result_db", "mnfctur_de_raw", "reqest_de", "mnfctur_de_or_fallback", "cntc_sys_key2",
        "addr_source_db", "inspct_purps_code", "reqest_sys_code", "in_m_org"]
dfx = df.merge(attr[keep], on="lims_data_key", how="left", validate="one_to_one", indicator=True)
assert len(dfx) == len(df) == 676_537 and (dfx["_merge"] == "both").all()
dfx["pos"] = dfx["judge_result"].astype(str).eq("검출").astype(int)
assert dfx["pos"].sum() == 1_969
same_judge = (dfx["judge_result"].astype(str) == dfx["judge_result_db"].astype(str)).mean()
assert same_judge == 1.0
code_nm = codes[codes["cmmn_cl_code"].isin(["IM02", "IM03", "IM08", "IM11", "CI07"])].set_index("cmmn_code")["cmmn_code_nm"].to_dict()
dfx["검사목적"] = dfx["inspct_purps_code"].map(code_nm).fillna("(원본LIMS 없음)")
dfx["의뢰시스템"] = dfx["reqest_sys_code"].map(code_nm).fillna("(결측)")
dfx["국내해외"] = dfx["src_route"].astype(str).map({"domestic": "국내", "overseas": "해외"})
for c in ["mnfctur_de", "mnfctur_de_raw", "reqest_de", "mnfctur_de_or_fallback"]:
    dfx[c] = pd.to_datetime(dfx[c])
N, K = len(dfx), int(dfx["pos"].sum())
m_link = float(dfx["in_m_org"].astype(bool).mean())
n_db = len(attr)
not_in = attr[~attr["lims_data_key"].isin(df["lims_data_key"])]
n_not_in, n_not_in_lab = len(not_in), int((not_in["addr_source_db"].astype(str) == "검사기관").sum())


def rate_table(frame, by, sort="pos"):
    g = frame.groupby(by, observed=True, dropna=False)["pos"].agg(건수="size", 검출="sum")
    g["검출률(%)"] = g["검출"] / g["건수"] * 100
    g["검출비중(%)"] = g["검출"] / frame["pos"].sum() * 100
    return g.sort_values(["검출", "건수"], ascending=False) if sort == "pos" else g.sort_index()


# ---------- Q1: 검사목적
q1_rate = rate_table(dfx, "검사목적")
prof = dfx.groupby("검사목적", observed=True).agg(
    건수=("pos", "size"), 검출=("pos", "sum"),
    **{"국내 행(%)": ("국내해외", lambda s: (s == "국내").mean() * 100),
       "자체의뢰(%)": ("의뢰시스템", lambda s: (s == "자체의뢰").mean() * 100),
       "통합망(%)": ("의뢰시스템", lambda s: (s == "통합망").mean() * 100)}
).sort_values(["검출", "건수"], ascending=False)
etc = q1_rate.loc["기타"]
assert int(etc["건수"]) == 326_892 and int(etc["검출"]) == 1_619 and int(q1_rate.loc["수거검사", "검출"]) == 66
nolink = dfx[dfx["검사목적"] == "(원본LIMS 없음)"]
Q1 = dict(n_etc=int(etc["건수"]), k_etc=int(etc["검출"]), share_etc=etc["검출비중(%)"],
          etc_self=prof.loc["기타", "자체의뢰(%)"], etc_dom=prof.loc["기타", "국내 행(%)"],
          k_col=int(q1_rate.loc["수거검사", "검출"]), share_col=q1_rate.loc["수거검사", "검출비중(%)"],
          col_int=prof.loc["수거검사", "통합망(%)"], n_nolink=len(nolink),
          nolink_ov=(nolink["국내해외"] == "해외").mean() * 100)

# ---------- Q2: 기준규격외 코드와 제품명 (작성자 임의 키워드 규칙)


def has_kw(s, pattern):
    x = s.astype("string").str.lower().str.replace(r"\s+", "", regex=True)
    return x.str.contains(pattern, regex=True, na=False).astype(bool)


PAT = {  # 2.ipynb 1-3절과 같은 규칙. 위에서부터 먼저 걸리는 분류 하나
    "환경검체": r"바닥|벽체|벽면|표면|장갑|도마|칼날|컨베이어|컨베어|배수|대차|선반|작업대|손잡이|스왑|swab|공중낙하|낙하균|천장|환경",
    "팽이버섯": r"팽이|enoki|flammulina",
    "공정시료": r"세척전|세척후|가열전|가열후|살균전|살균후|입고|원료육|반제품|공정|숙성전|숙성후|포장전|포장후|해동|절단전|절단후|냉각",
    "완제품표기": r"완제품",
}
flags = {k: has_kw(dfx["prduct_nm"], p) for k, p in PAT.items()}
dfx["검체성격(임의 분류)"] = np.select([flags["환경검체"], flags["팽이버섯"], flags["공정시료"], flags["완제품표기"]],
                                ["환경검체", "팽이버섯", "공정시료", "완제품표기"], "일반 제품명")
q2_kind = rate_table(dfx, "검체성격(임의 분류)")


def top_names(sub, k=15):
    a = sub["prduct_nm"].value_counts().head(k)
    b = sub.loc[sub["pos"] == 1, "prduct_nm"].value_counts().head(k)
    n = max(len(a), len(b))
    pad = lambda s: [f"{i} ({v:,})" for i, v in s.items()] + [""] * (n - len(s))
    out = pd.DataFrame({"전체 행 상위 제품명 (건수)": pad(a), "검출 행 상위 제품명 (검출 수)": pad(b)})
    out.index = range(1, n + 1)
    out.index.name = "순위"
    return out, list(a.index)


Z = {}
for name in ["정액,수정란,종란", "기타기준규격외"]:
    sub = dfx[dfx["train_prdlst_nm"].astype(str) == name]
    tbl, names = top_names(sub)
    Z[name] = dict(n=len(sub), k=int(sub["pos"].sum()), codes=sub["prdlst_code"].astype(str).value_counts(normalize=True).to_dict(),
                   table=tbl, names=names)
assert Z["기타기준규격외"]["k"] == 791 and Z["정액,수정란,종란"]["k"] == 640
assert Z["정액,수정란,종란"]["codes"] == {"Z2700400000000": 1.0} and Z["기타기준규격외"]["codes"] == {"Z3800100000000": 1.0}
for nm in ["수출용 팽이버섯 (2-1번)", "ENOKI MUSHROOM", "포장실 바닥", "수확실 벽체"]:
    assert nm in Z["정액,수정란,종란"]["names"], nm
for nm in ["완제품", "원료육", "배추(세척전)", "배추(세척후)"]:
    assert nm in Z["기타기준규격외"]["names"], nm
assert int(q2_kind.loc["팽이버섯", "검출"]) == 445

# ---------- Q3: 기준일
q3_basis = rate_table(dfx, ["국내해외", "date_basis"], sort="index")
S = sorg.sort_values("lims_seq").drop_duplicates("rcept_no").set_index("rcept_no")
ship_multi = sorg.groupby("rcept_no")["shipng_de"].nunique(dropna=True).gt(1).mean()   # 같은 접수번호 안의 선적일 종류
ov = dfx[dfx["국내해외"] == "해외"].copy()
ov["s_shipng_de"] = pd.to_datetime(ov["cntc_sys_key2"].map(S["shipng_de"]))
ov_link = ov["cntc_sys_key2"].isin(S.index).mean()
b = dfx["date_basis"].astype(str)
raw = dfx["mnfctur_de_raw"].notna()
sb = ov[ov["date_basis"].astype(str) == "선적일"]


def same_row(cond, what, eq):
    eq = pd.Series(eq)
    return {"parquet 행": cond, "비교한 DB 값": what, "비교 행 수": len(eq), "다른 행 수": int((~eq).sum()),
            "같은 비율(%)": eq.mean() * 100}


match = pd.DataFrame([
    same_row("전체", "ready_data.mnfctur_de_or_fallback", dfx["mnfctur_de"] == dfx["mnfctur_de_or_fallback"]),
    same_row("date_basis=제조일", "ready_data.mnfctur_de (원본 제조일)",
             dfx.loc[b == "제조일", "mnfctur_de"] == dfx.loc[b == "제조일", "mnfctur_de_raw"]),
    same_row("date_basis=의뢰일", "ready_data.reqest_de (의뢰일)",
             dfx.loc[b == "의뢰일", "mnfctur_de"] == dfx.loc[b == "의뢰일", "reqest_de"]),
    same_row("해외·date_basis=선적일", "s_org.shipng_de (선적일)", sb["mnfctur_de"] == sb["s_shipng_de"]),
    same_row("전체 (날짜 대신 여부)", "원본 제조일 있음 ⇔ date_basis=제조일", (b == "제조일") == raw),
]).set_index("parquet 행")
print(match.to_string())
wd = pd.crosstab(dfx["date_basis"], dfx["mnfctur_de"].dt.dayofweek, normalize="index").mul(100)
wd.columns = ["월", "화", "수", "목", "금", "토", "일"]
wd["주말 합계(%)"] = wd["토"] + wd["일"]
dom = dfx[dfx["국내해외"] == "국내"]
gap = (dom["reqest_de"] - dom["mnfctur_de_raw"]).dt.days.dropna()
Q3 = dict(raw_share=raw.mean(), dom_n=len(dom),
          dom_mfg=int(q3_basis.loc[("국내", "제조일"), "건수"]), dom_req=int(q3_basis.loc[("국내", "의뢰일"), "건수"]),
          ov_ship=int(q3_basis.loc[("해외", "선적일"), "건수"]),
          dom_req_share=q3_basis.loc[("국내", "의뢰일"), "건수"] / len(dom),
          rate_req=q3_basis.loc[("국내", "의뢰일"), "검출률(%)"], rate_mfg=q3_basis.loc[("국내", "제조일"), "검출률(%)"],
          wk_req=wd.loc["의뢰일", "주말 합계(%)"], wk_mfg=wd.loc["제조일", "주말 합계(%)"], wk_ship=wd.loc["선적일", "주말 합계(%)"],
          gap_n=len(gap), gap_med=gap.median(), gap_p75=gap.quantile(.75), gap_p95=gap.quantile(.95),
          n_ov=len(ov), ov_link=ov_link, ship_multi=ship_multi)
assert Q3["dom_mfg"] == 295_907 and Q3["dom_req"] == 332_979 and Q3["ov_ship"] == 47_063 and round(Q3["gap_med"]) == 2

# ============================================================ 2. 결과 표 캡처 (HTML → 헤드리스 브라우저 → PNG)
CSS = """
body{margin:0;padding:10px;background:#fff;font-family:'Malgun Gothic','맑은 고딕',sans-serif}
table{border-collapse:collapse;font-size:13px;color:#222}
thead th{border-bottom:1.5px solid #333;padding:6px 12px;text-align:right;font-weight:700;white-space:nowrap;vertical-align:bottom}
thead th.l{text-align:left}
tbody th{padding:5px 12px;text-align:left;font-weight:700;white-space:nowrap}
tbody td{padding:5px 12px;text-align:right;white-space:nowrap}
tbody td.l{text-align:left}
tbody tr:nth-child(odd){background:#f4f4f4}
tbody tr.hl{background:#fff0b3}
"""
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
BROWSER = next(p for p in BROWSERS if os.path.exists(p))
TMP = Path(tempfile.mkdtemp(prefix="qdoc_"))


def fmt_value(col, v, fmts):
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "-"
    f = fmts.get(col)
    if f:
        return f.format(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:,.1f}"


def capture(frame, name, fmts=None, highlight=(), left_cols=()):
    """DataFrame → Jupyter 출력과 비슷한 표 이미지. highlight: 강조할 행 인덱스(첫 단계 값)."""
    fmts = fmts or {}
    idx_names = [n if n is not None else "" for n in frame.index.names]
    head = "".join(f'<th class="l">{html.escape(str(n))}</th>' for n in idx_names)
    head += "".join(f'<th class="{"l" if c in left_cols else ""}">{html.escape(str(c))}</th>' for c in frame.columns)
    rows, prev = [], None
    for i in range(len(frame)):                       # iat: 열 dtype 유지 (정수가 실수로 바뀌지 않게)
        key = frame.index[i]
        keys = key if isinstance(key, tuple) else (key,)
        ths = "".join(f"<th>{'' if (lv == 0 and len(keys) > 1 and prev is not None and prev[0] == k) else html.escape(str(k))}</th>"
                      for lv, k in enumerate(keys))
        prev = keys
        tds = "".join(f'<td class="{"l" if c in left_cols else ""}">{html.escape(fmt_value(c, frame.iat[i, j], fmts))}</td>'
                      for j, c in enumerate(frame.columns))
        cls = ' class="hl"' if (key in highlight or keys[0] in highlight) else ""
        rows.append(f"<tr{cls}>{ths}{tds}</tr>")
    page = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"
    n = len(list(TMP.glob("t*.html")))
    hp, shot, png = TMP / f"t{n}.html", TMP / f"t{n}.png", CAP_DIR / f"{name}.png"   # 브라우저에는 영문 경로만 넘김
    hp.write_text(page, encoding="utf-8")
    height = 90 + 34 * (len(frame) + 2)
    subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size=1800,{height}", f"--user-data-dir={TMP / 'profile'}", f"--screenshot={shot}", hp.as_uri()],
                   capture_output=True, timeout=120, check=True)
    im = Image.open(shot).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    pad = 12
    im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height))).save(png)
    return png


CAP_DIR.mkdir(parents=True, exist_ok=True)
RATE_FMT = {"건수": "{:,.0f}", "검출": "{:,.0f}", "검출률(%)": "{:.3f}", "검출비중(%)": "{:.1f}"}
IMG = {
    "Q1-1": capture(q1_rate, "Q1-1_검사목적별_검출", RATE_FMT, highlight=("기타", "수거검사")),
    "Q1-2": capture(prof, "Q1-2_검사목적별_의뢰시스템", {"건수": "{:,.0f}", "검출": "{:,.0f}", "국내 행(%)": "{:.2f}",
                                                       "자체의뢰(%)": "{:.2f}", "통합망(%)": "{:.2f}"}, highlight=("기타", "수거검사")),
    "Q2-1": capture(Z["정액,수정란,종란"]["table"], "Q2-1_정액수정란종란_제품명", left_cols=tuple(Z["정액,수정란,종란"]["table"].columns)),
    "Q2-2": capture(Z["기타기준규격외"]["table"], "Q2-2_기타기준규격외_제품명", left_cols=tuple(Z["기타기준규격외"]["table"].columns)),
    "Q2-3": capture(q2_kind, "Q2-3_임의분류별_검출", RATE_FMT),
    "Q3-1": capture(q3_basis, "Q3-1_국내해외_기준일출처", RATE_FMT),
    "Q3-2": capture(match, "Q3-2_기준일_DB날짜_일치율", {"비교 행 수": "{:,.0f}", "다른 행 수": "{:,.0f}", "같은 비율(%)": "{:.3f}"},
                    left_cols=("비교한 DB 값",)),
    "Q3-3": capture(wd, "Q3-3_기준일출처별_요일분포", {c: "{:.1f}" for c in wd.columns}),
}

# ============================================================ 3. 워드
INK, BLUE, ORANGE, GRAYTXT = "1F2937", "1F5FAF", "B45309", "6B7280"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
TEXT_W = 17.0


def set_rfonts(rpr, name, east="Malgun Gothic"):
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    rf.set(qn("w:ascii"), name)
    rf.set(qn("w:hAnsi"), name)
    rf.set(qn("w:cs"), name)
    rf.set(qn("w:eastAsia"), east)


def style_font(name, size, bold=None, color=INK, space_before=None, space_after=None):
    st = doc.styles[name]
    set_rfonts(st.element.get_or_add_rPr(), "Malgun Gothic")
    st.font.size = Pt(size)
    if bold is not None:
        st.font.bold = bold
    st.font.color.rgb = RGBColor.from_string(color)
    if space_before is not None:
        st.paragraph_format.space_before = Pt(space_before)
    if space_after is not None:
        st.paragraph_format.space_after = Pt(space_after)
    return st


style_font("Normal", 9.5, space_after=2).paragraph_format.line_spacing = 1.2
style_font("Heading 1", 15, bold=True, color="111827", space_before=0, space_after=6)
style_font("Heading 2", 11.5, bold=True, color=BLUE, space_before=10, space_after=3)
style_font("Heading 3", 10, bold=True, color="111827", space_before=8, space_after=2)
for s in ("List Bullet", "List Bullet 2"):
    style_font(s, 9.5, space_after=1)

TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def rich(p, text, size=None, color=None, bold=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1])
            set_rfonts(r._element.get_or_add_rPr(), "Consolas")
            r.font.size = Pt((size or 9.5) - 0.5)
            r.font.color.rgb = RGBColor.from_string("7C2D12")
            continue
        else:
            r = p.add_run(part)
            r.bold = bold or None
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p


def para(text="", size=None, color=None, bold=False, align=None, after=None, container=None):
    p = (container or doc).add_paragraph()
    rich(p, text, size, color, bold)
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullet(text, level=1, container=None):
    p = (container or doc).add_paragraph(style="List Bullet" if level == 1 else "List Bullet 2")
    return rich(p, text)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def cell_borders(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
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
    tcPr.append(bd)


def table_setup(t, widths, margins=(60, 110, 60, 110)):
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = t._tbl.tblPr
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:w"), str(int(sum(widths) * 567)))
    tw.set(qn("w:type"), "dxa")
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
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


def grid_table(header, rows, widths, header_fill="E5E7EB", size=9, first_col_bold=False):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = doc.styles["Table Grid"]
    table_setup(t, widths)
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.paragraphs[0].text = ""
        rich(c.paragraphs[0], h, size=size, bold=True)
        shade(c, header_fill)
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r):
            c = t.rows[i].cells[j]
            rich(c.paragraphs[0], str(v), size=size, bold=(first_col_bold and j == 0))
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
            cell_borders(c, top=(4, "C7CCD4"), left=(4, "C7CCD4"), bottom=(4, "C7CCD4"), right=(4, "C7CCD4"))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def box(fill, bar, fn):
    """왼쪽 굵은 선이 있는 강조 상자. fn(cell) 로 내용 채움."""
    t = doc.add_table(rows=1, cols=1)
    table_setup(t, [TEXT_W], margins=(90, 200, 90, 160))
    c = t.rows[0].cells[0]
    shade(c, fill)
    cell_borders(c, left=(24, bar))
    c.paragraphs[0].text = ""
    fn(c)
    first = c.paragraphs[0]
    if not first.text and len(c.paragraphs) > 1:
        first._element.getparent().remove(first._element)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def sql_box(sql):
    def fill(c):
        for line in sql.strip("\n").splitlines():
            p = c.add_paragraph()
            r = p.add_run(line if line else " ")
            set_rfonts(r._element.get_or_add_rPr(), "Consolas")
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor.from_string("111827")
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
    box("F3F4F6", "9CA3AF", fill)


def figure(key, caption, max_w=TEXT_W - 0.5, max_h=None):
    im = Image.open(IMG[key])
    natural_cm = im.width / 2 / 96 * 2.54      # 캡처 배율 2
    w = min(max_w, natural_cm)
    if max_h:
        w = min(w, max_h * im.width / im.height)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(str(IMG[key]), width=Cm(w))
    para(f"그림 {key}. {caption}", size=8.5, color=GRAYTXT, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)


def question_box(items):
    """items: [(대상, 색, [질문 문장, ...]), ...]"""
    def fill(c):
        for who, color, qs in items:
            p = c.add_paragraph()
            r = p.add_run(who)
            r.bold = True
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor.from_string(color)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(1)
            for q in qs:
                bp = c.add_paragraph(style="List Bullet")
                rich(bp, q)
                bp.paragraph_format.space_after = Pt(2)
    box("EEF4FB", "1F5FAF", fill)


def row_flags(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    cs = OxmlElement("w:cantSplit")          # 행이 쪽 사이에서 잘리지 않게
    trPr.append(cs)
    if header:
        th = OxmlElement("w:tblHeader")      # 다음 쪽에서 머리행 반복
        trPr.append(th)


def answer_sheet(rows):
    """rows: [(질문 번호, 대상), ...] — 문서 끝의 답변 기록지 한 장"""
    t = grid_table(["질문", "대상", "답변자 · 일자", "답변 (공식 의견)", "모델링 반영 결정"], [[q, who, "", "", ""] for q, who in rows],
                   [1.4, 2.4, 2.6, 6.4, 4.2], header_fill="E5E7EB", size=8.5, first_col_bold=True)
    row_flags(t.rows[0], header=True)
    for row in t.rows[1:]:
        row.height = Cm(1.45)
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        row_flags(row)


def label_line(label, text):
    p = doc.add_paragraph()
    r = p.add_run(label + "  ")
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(BLUE)
    rich(p, text)
    p.paragraph_format.space_after = Pt(2)
    return p


def new_page_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.page_break_before = True   # 빈 쪽이 생기지 않도록 별도 쪽나눔 문단 대신 사용
    return h


def add_footer():
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run(f"리스테리아 예측 모델 질문 리스트 {VERSION} · ")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string(GRAYTXT)
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


add_footer()
WHO_DATA, WHO_DOMAIN = ("데이터 담당자께", BLUE), ("도메인 전문가께", ORANGE)

# ---------------------------------------------------------------- 표지·개요
doc.add_heading("리스테리아 예측 모델 — 데이터 담당자 · 도메인 전문가 질문 리스트", level=1)
para(f"{VERSION} · 작성일 {TODAY}", color=GRAYTXT, after=8)
grid_table(["항목", "내용"], [
    ["목적", "ML/DL 실험·보고서의 의사결정(타겟 범위, 데이터 한계, 제외할 데이터)을 **담당자·전문가 답변**으로 근거화"],
    ["근거 자료", "`리스테리아ML/notebook/2.ipynb` (git 커밋 `efff1cc`)"],
    ["기준 데이터", f"`data/listeria_body.parquet` {N:,}행 · 검출 {K:,}건 / DB `TOXIN_MODEL_PRED` ({TODAY} SELECT 조회)"],
    ["답변 대상", "데이터 담당자 (DB·테이블 설계) · 도메인 전문가 (식품·위해요소)"],
], [3.0, 14.0], first_col_bold=True)

doc.add_heading("작성 원칙", level=2)
bullet("문서의 수치는 작성자가 2.ipynb 에서 **임의로 집계·분류한 결과**. 해석은 모두 **추정**이며, 답변 전에는 모델링 결정에 쓰지 않음")
bullet("원자료는 고치지 않음 — 결측 채움·재분류·행 삭제 없이 DB 값 그대로 집계")
bullet("질문마다 근거를 **① 데이터·조인 → ② 작업·결과 → ③ 상황 정리(추정) → ④ 의문** 순서로 적음")
bullet("답변은 마지막 쪽 **답변 기록지**에 답변자·일자와 함께 기록. 조인·집계 방법이 틀렸다면 같은 칸에 지적 요청")

doc.add_heading("질문 요약", level=2)
grid_table(["No", "결정할 사항", "질문 요지", "대상"], [
    ["Q1", "타겟 범위", f"검출의 {Q1['share_etc']:.1f}%가 검사목적 '기타'. 이 검사를 예측 대상에 포함하는가", "데이터 담당자 · 도메인 전문가"],
    ["Q2", "제외할 데이터", "기준규격외 코드 속 팽이버섯 수출검사·공정시료·환경검체로 보이는 행을 학습에서 빼는가", "데이터 담당자 · 도메인 전문가"],
    ["Q3", "데이터 한계", "기준일 컬럼에 제조일·의뢰일·선적일이 섞인 것으로 보임. 날짜 변수·기간 분할에 어느 날짜를 쓰는가", "데이터 담당자 · 도메인 전문가"],
], [1.1, 2.6, 9.6, 3.7], first_col_bold=True)

# ---------------------------------------------------------------- Q1
new_page_heading(f"Q1. [타겟 범위] 검출의 {Q1['share_etc']:.0f}%가 검사목적 '기타' — 예측 대상에 포함하는가?", level=1)
label_line("결정할 사항", "학습·평가 대상 범위 — 전체 검사 / 정부 감시검사(수거·수입·모니터링)만 / 검사목적별 층 분리")
question_box([
    (*WHO_DATA, ["**Q1-1.** 원본 LIMS `inspct_purps_code` 값 `IM02000007`('기타')은 어떤 검사일 때 입력되는지?",
                 "**Q1-2.** 아래 ①의 조인·코드명 매핑이 맞는지? (컬럼 설명에는 `ZIM02`, 공통코드 테이블 분류 코드는 `IM02` 로 되어 있음)"]),
    (*WHO_DOMAIN, ["**Q1-3.** ③의 추정처럼 '기타'가 업체가 스스로 맡긴 검사라면, 예측 대상에 포함하는 것이 맞는지?",
                   "**Q1-4.** 업체 자체 의뢰 검사와 정부 수거·수입 검사를 한 모델로 학습해도 되는지, 나눠야 하는지?"]),
])
doc.add_heading("근거 ① 사용 데이터 · 조인", level=3)
bullet(f"기준 행: `data/listeria_body.parquet` {N:,}행 (`judge_result` = '검출' {K:,}건)")
bullet("DB `TOXIN_MODEL_PRED` 에서 아래 SQL 로 추출 (`scripts/nb2_extract_db.py`, 이 질문에 쓴 컬럼만 표시)")
sql_box(f"""
SELECT h.lims_data_key,
       m.inspct_purps_code,      -- 검사목적 코드
       m.reqest_sys_code         -- 의뢰시스템 코드
FROM ready_data.hazard_preprocessed_data h
LEFT JOIN ba_data.m_org_lims_data_info m
       ON m.lims_data_key = h.lims_data_key
WHERE h.hazard_label = '리스테리아';          -- {n_db:,}행

SELECT cmmn_cl_code, cmmn_code, cmmn_code_nm  -- 코드명
FROM ba_data.t_cm_cmmn_code
WHERE cmmn_cl_code IN ('IM02','IM03','IM08','IM11','IM40')
   OR cmmn_code = ANY(:실제 의뢰시스템 코드값);
""")
bullet("parquet 와 첫 번째 추출 결과를 `lims_data_key` 로 1:1 LEFT JOIN")
bullet(f"parquet {N:,}행이 모두 DB 에 있음 · 원본 LIMS 값이 붙은 행 {m_link:.1%}", 2)
bullet(f"원본 LIMS 가 없는 {Q1['n_nolink']:,}행은 '(원본LIMS 없음)'으로 따로 표시 (이 중 해외 행 {Q1['nolink_ov']:.0f}%)", 2)
bullet(f"DB 에만 있고 parquet 에 없는 {n_not_in:,}행은 집계에서 빠짐 (이 중 {n_not_in_lab:,}행은 주소 출처가 '검사기관')", 2)
bullet("코드명: 공통코드 `cmmn_code` = 코드값으로 매핑 — 검사목적은 분류 `IM02`, 의뢰시스템은 분류 `CI07` 의 이름 사용")

doc.add_heading("근거 ② 작업 · 결과", level=3)
bullet("작업 1: 검사목적 코드명별 건수·검출 수 집계 (필터·보정 없음)")
figure("Q1-1", "검사목적별 건수·검출 (2.ipynb 2-2절과 같은 집계, 노란 줄 = 본문에서 언급)", max_h=7.6)
bullet("작업 2: 검사목적별로 국내 행 비율과, 의뢰시스템 코드명이 '자체의뢰'·'통합망'인 행의 비율 집계")
figure("Q1-2", "검사목적별 국내 행·의뢰시스템 비율 (%) (2.ipynb 2-2절)", max_h=7.6)

doc.add_heading("근거 ③ 상황 정리 (작성자 추정 — 확인 필요)", level=3)
bullet(f"'기타' {Q1['n_etc']:,}행에서 검출 {Q1['k_etc']:,}건 → 전체 검출의 {Q1['share_etc']:.1f}%")
bullet(f"'기타' 행: 의뢰시스템 '자체의뢰' {Q1['etc_self']:.2f}%, 국내 행 {Q1['etc_dom']:.2f}%")
bullet(f"'수거검사': 검출 {Q1['k_col']}건({Q1['share_col']:.1f}%), 의뢰시스템 '통합망' {Q1['col_int']:.1f}%")
bullet("**추정**: '기타'는 업체가 스스로 맡긴 검사로 보임 — 의뢰시스템 코드명만 보고 추정했으며, 코드 입력 기준은 확인하지 못함")

doc.add_heading("근거 ④ 그래서 생긴 의문", level=3)
bullet("추정이 맞다면 검출 대부분이 업체가 고른 시료에서 나옴")
bullet("그대로 학습하면 모델이 '어떤 식품이 오염되는가'보다 '누가 왜 검사를 맡겼는가'를 배울 수 있음 → 예측 대상 범위를 먼저 정해야 함 (Q1-3, Q1-4)")

# ---------------------------------------------------------------- Q2
z27, z38 = Z["정액,수정란,종란"], Z["기타기준규격외"]
kind = q2_kind
new_page_heading("Q2. [제외할 데이터] 기준규격외 코드 속 팽이버섯 수출검사·공정시료·환경검체로 보이는 행을 학습에서 빼는가?", level=1)
label_line("결정할 사항", "학습에서 뺄 행 / 별도 층으로 둘 행 / 품목코드를 변수로 쓸 때 기준규격외 코드 처리")
question_box([
    (*WHO_DATA, ["**Q2-1.** 품목코드 `Z2700400000000`(학습용 품목명 '정액,수정란,종란') 행의 제품명이 수출용 팽이버섯·재배장 바닥·벽체로 보임(그림 Q2-1). 이 코드로 들어온 경위는? (매핑 규칙인지, 입력 관행인지)",
                 "**Q2-2.** `Z3800100000000`('기타기준규격외')은 어떤 경우에 입력되는지? (세척 전후 배추·원료육 같은 제품명이 많음, 그림 Q2-2)"]),
    (*WHO_DOMAIN, ["**Q2-3.** ② 작업 3의 키워드 분류(작성자 임의 규칙)가 검체 성격을 제대로 나누는지? 틀리거나 빠진 키워드 지적 요청",
                   "**Q2-4.** 공정 중 채취 시료·작업장 환경 검체·수출용 팽이버섯 검사를 예측 대상에서 빼야 하는지, 별도 층으로 둬야 하는지?"]),
])
doc.add_heading("근거 ① 사용 데이터 · 조인", level=3)
bullet("조인 없음 — `data/listeria_body.parquet` 컬럼만 사용")
bullet("`prdlst_code` 원본 품목코드 · `train_prdlst_nm` 학습용 품목명 · `prduct_nm` 제품명 · `judge_result` 판정", 2)
bullet(f"위 컬럼의 출처는 DB `ready_data.hazard_preprocessed_data` — parquet 판정값은 DB `judge_result` 와 행 단위로 {same_judge:.0%} 같음")

doc.add_heading("근거 ② 작업 · 결과", level=3)
bullet("작업 1: `train_prdlst_nm` 별 검출 수 집계 → 검출이 가장 많은 두 품목")
bullet(f"'기타기준규격외' {z38['n']:,}행 · 검출 {z38['k']}건 (원본 품목코드 전부 `Z3800100000000`)", 2)
bullet(f"'정액,수정란,종란' {z27['n']:,}행 · 검출 {z27['k']}건 (원본 품목코드 전부 `Z2700400000000`)", 2)
bullet("작업 2: 두 품목 안에서 `prduct_nm` 빈도 상위 15개 — 전체 행 / 검출 행")
figure("Q2-1", "'정액,수정란,종란'(Z2700400000000) 제품명 상위 15 (2.ipynb 3-3절)")
figure("Q2-2", "'기타기준규격외'(Z3800100000000) 제품명 상위 15 (2.ipynb 3-3절)")
bullet("작업 3 (**작성자 임의 규칙**): 제품명의 공백을 없애고 소문자로 바꾼 뒤, 아래 키워드가 들어 있으면 분류. 여러 개가 걸리면 표의 위쪽 분류 하나만 붙임")
grid_table(["임의 분류", "키워드 (정규식 OR)"], [
    ["환경검체", "바닥, 벽체, 벽면, 표면, 장갑, 도마, 칼날, 컨베이어, 컨베어, 배수, 대차, 선반, 작업대, 손잡이, 스왑, swab, 공중낙하, 낙하균, 천장, 환경"],
    ["팽이버섯", "팽이, enoki, flammulina"],
    ["공정시료", "세척전, 세척후, 가열전, 가열후, 살균전, 살균후, 입고, 원료육, 반제품, 공정, 숙성전, 숙성후, 포장전, 포장후, 해동, 절단전, 절단후, 냉각"],
    ["완제품표기", "완제품"],
    ["일반 제품명", "위 키워드 없음"],
], [2.6, 14.4], size=8.5, first_col_bold=True)
figure("Q2-3", "임의 분류별 건수·검출 (2.ipynb 1-3절 · 3-4절)")

doc.add_heading("근거 ③ 상황 정리 (작성자 추정 — 확인 필요)", level=3)
bullet(f"두 코드의 검출 {z27['k'] + z38['k']:,}건 = 전체 검출의 {(z27['k'] + z38['k']) / K:.1%}")
bullet("'정액,수정란,종란' 상위 제품명: '수출용 팽이버섯 (2-1번)', 'ENOKI MUSHROOM', '포장실 바닥', '수확실 벽체' 등 → 가축 정액·수정란과 관계없어 보임")
bullet("'기타기준규격외' 상위 제품명: '완제품', '원료육', '배추(세척전)', '배추(세척후)' 등 → 제조 공정 중에 채취한 시료로 보임")
bullet(f"임의 분류 결과: 팽이버섯 {int(kind.loc['팽이버섯', '건수']):,}행 · 검출 {int(kind.loc['팽이버섯', '검출'])}건({kind.loc['팽이버섯', '검출률(%)']:.2f}%), "
       f"환경검체 {int(kind.loc['환경검체', '건수']):,}행 · {int(kind.loc['환경검체', '검출'])}건({kind.loc['환경검체', '검출률(%)']:.2f}%), "
       f"공정시료 {int(kind.loc['공정시료', '건수']):,}행 · {int(kind.loc['공정시료', '검출'])}건")
bullet("**추정**: 완제품이 아닌 공정·환경 검체와 수출검사 팽이버섯이 적지 않게 섞여 있음 — 키워드 규칙이라 오탐·누락이 있고, 전문가 검증을 받지 않음")

doc.add_heading("근거 ④ 그래서 생긴 의문", level=3)
bullet("이 행을 그대로 학습하면 식품의 오염 위험보다 공장 위생 점검·수출검사 관행을 배울 수 있음")
bullet("품목코드를 변수로 쓰면 '정액,수정란,종란'이 고위험 식품처럼 학습될 수 있음")
bullet("→ 제외·층 분리 기준이 필요 (Q2-3, Q2-4)")

# ---------------------------------------------------------------- Q3
new_page_heading("Q3. [데이터 한계] 기준일 컬럼에 제조일·의뢰일·선적일이 섞인 것으로 보임 — 날짜 변수·기간 분할에 어느 날짜를 쓰는가?", level=1)
label_line("결정할 사항", "연·월·계절 등 날짜 변수와 학습/검증 기간 분할에 쓸 날짜 / 날짜 컬럼을 사건별(제조·의뢰·선적)로 나눠 쓸지")
question_box([
    (*WHO_DATA, ["**Q3-1.** parquet `mnfctur_de` 는 이름은 제조일인데, 값은 DB `mnfctur_de_or_fallback`(기준일)과 같아 보임(그림 Q3-2). 의도한 것인지?",
                 "**Q3-2.** `date_basis` 설명이 테이블정의서('제조일/의뢰일/수거일')와 실제 값('제조일/의뢰일/선적일')이 다름. 최종 규칙은? '선적일'은 수입 LIMS `shipng_de` 가 맞는지?"]),
    (*WHO_DOMAIN, ["**Q3-3.** 리스테리아 검출을 시기(월·계절)와 연결해 해석할 때, 기준 시점은 제조일과 검사 의뢰일 중 무엇이 맞는지?",
                   f"**Q3-4.** 제조일이 없는 행(국내 행의 약 {Q3['dom_req_share']:.0%})을 의뢰일로 대신해도 되는지? 제조→의뢰 간격이 긴 행도 있어 보임(③)"]),
])
doc.add_heading("근거 ① 사용 데이터 · 조인", level=3)
bullet("parquet 컬럼: `mnfctur_de`(기준일로 보이는 날짜) · `date_basis`(기준일 출처) · `src_route`(국내/해외)")
sql_box(f"""
SELECT lims_data_key,
       mnfctur_de,               -- 원본 제조일 (분석에서는 mnfctur_de_raw 로 이름 바꿈)
       reqest_de,                -- 의뢰일
       mnfctur_de_or_fallback,   -- 기준일
       cntc_sys_key2             -- 수입 접수번호
FROM ready_data.hazard_preprocessed_data
WHERE hazard_label = '리스테리아';           -- {n_db:,}행

SELECT lims_seq, rcept_no, shipng_de         -- 선적일
FROM ba_data.s_org_lims_data_info
WHERE label = '리스테리아';                  -- {len(sorg):,}행
""")
bullet("parquet 와 ready_data 추출 결과: `lims_data_key` 1:1 LEFT JOIN")
bullet("수입 LIMS 는 접수번호(`rcept_no`)당 여러 행 → `lims_seq` 가 가장 작은 행 1개만 사용")
bullet(f"같은 접수번호 안에서 선적일이 2개 이상인 접수번호: {Q3['ship_multi']:.2%}", 2)
bullet(f"parquet 해외 행의 `cntc_sys_key2` = `rcept_no` 로 LEFT JOIN → 해외 {Q3['n_ov']:,}행 중 {Q3['ov_link']:.1%} 연결", 2)

doc.add_heading("근거 ② 작업 · 결과", level=3)
bullet("작업 1: 국내/해외 × `date_basis` 별 건수·검출 집계")
figure("Q3-1", "국내/해외 × 기준일 출처(date_basis)별 건수·검출 (2.ipynb 4-1절)")
bullet("작업 2: `date_basis` 값마다 parquet `mnfctur_de` 를 DB 날짜와 행 단위로 비교 (마지막 줄은 날짜가 아니라 '원본 제조일 있음' 여부를 비교, `s_org` = `ba_data.s_org_lims_data_info`)")
figure("Q3-2", "parquet mnfctur_de 와 DB 날짜 비교 (2.ipynb 4-1절 · 4-5절)")
bullet("작업 3: `date_basis` 별 `mnfctur_de` 요일 분포 (행 비율 %)")
figure("Q3-3", "기준일 출처별 요일 분포 (2.ipynb 4-2절)")

doc.add_heading("근거 ③ 상황 정리 (작성자 추정 — 확인 필요)", level=3)
bullet(f"원본 제조일이 있는 행 {Q3['raw_share']:.1%} — 국내: 제조일 {Q3['dom_mfg']:,}행 / 의뢰일 {Q3['dom_req']:,}행, 해외: 선적일 {Q3['ov_ship']:,}행")
bullet(f"요일: 주말 비율이 의뢰일 {Q3['wk_req']:.1f}%, 제조일 {Q3['wk_mfg']:.1f}%, 선적일 {Q3['wk_ship']:.1f}% → 세 값이 서로 다른 사건 날짜로 보임")
bullet(f"원본 제조일이 있는 국내 {Q3['gap_n']:,}행의 제조→의뢰 간격: 중앙값 {Q3['gap_med']:.0f}일, 75% 지점 {Q3['gap_p75']:.0f}일, 95% 지점 {Q3['gap_p95']:.0f}일")
bullet(f"국내 검출률: 기준일이 의뢰일인 행 {Q3['rate_req']:.3f}%, 제조일인 행 {Q3['rate_mfg']:.3f}% → 제조일이 비어 있는 것이 검체 종류와 관련 있을 가능성")
bullet("**추정**: parquet `mnfctur_de` 한 컬럼에 제조·의뢰·선적 세 사건의 날짜가 섞여 있음")

doc.add_heading("근거 ④ 그래서 생긴 의문", level=3)
bullet("같은 '7월'이라도 행마다 제조한 달·검사를 맡긴 달·해외에서 선적한 달로 뜻이 다름 → 월·계절 변수와 기간 분할의 의미가 흐려짐")
bullet("수입 행은 해외 선적 날짜라 국내 행과 같은 시간축으로 보기 어려워 보임")
bullet("→ 날짜 변수에 쓸 기준 날짜와, 제조일이 없는 행의 처리 기준이 필요 (Q3-1 ~ Q3-4)")

# ---------------------------------------------------------------- 답변 기록지
new_page_heading("답변 기록지", level=1)
bullet("답변자 · 일자와 공식 의견을 적어 주시면, 모델링 결정의 근거로 보관함")
bullet("조인·집계 방법이 틀린 곳이 있으면 해당 질문 칸에 함께 적어 주시기 바람")
doc.add_paragraph().paragraph_format.space_after = Pt(2)
answer_sheet([("Q1-1", "데이터 담당자"), ("Q1-2", "데이터 담당자"), ("Q1-3", "도메인 전문가"), ("Q1-4", "도메인 전문가"),
              ("Q2-1", "데이터 담당자"), ("Q2-2", "데이터 담당자"), ("Q2-3", "도메인 전문가"), ("Q2-4", "도메인 전문가"),
              ("Q3-1", "데이터 담당자"), ("Q3-2", "데이터 담당자"), ("Q3-3", "도메인 전문가"), ("Q3-4", "도메인 전문가")])

OUT_DIR.mkdir(parents=True, exist_ok=True)
doc.save(DOCX)
print("saved:", DOCX)
for k, v in IMG.items():
    print(f"  {k}: {v.name} {Image.open(v).size}")
print("Q1:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in Q1.items()})
print("Q3:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in Q3.items()})
print("m_link", round(m_link, 4), "not_in", n_not_in, n_not_in_lab, "same_judge", same_judge)
