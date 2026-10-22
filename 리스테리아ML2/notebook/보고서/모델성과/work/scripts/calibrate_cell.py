# -*- coding: utf-8 -*-
"""표 칸 안에서 줄바꿈 없이 들어가는 최소 칸 폭 재기(칸 여백·칸 간격 15 dxa, 양식과 같은 표 속성).

문자열마다 칸 폭을 0.5pt 씩 늘린 1칸 표를 여러 개 만들고, Word(내가 띄운 인스턴스)로 열어 칸 안 줄 수를 잼.
결과: 문자열, 글자 크기, Word 문단 폭(calibrate_width 와 같은 방식), 줄바꿈 없는 최소 칸 폭 → 차이(= 칸 덧폭).
"""
import os

import pythoncom
import win32com.client as win32
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm

TMP = os.path.join(os.environ['TEMP'], 'listeria_perf_check', 'calib_cell.docx')
TEMPLATE = r'C:\Users\NT751\Desktop\천영진\모델 성과 발표 예시.docx'
import sys  # noqa: E402

SIZE = 9.0
TESTS = ['기상 시퀀스 90일', '27,554.1', 'Accuracy', '1차 베이스라인']
BREAK_FIRST = len(sys.argv) > 1 and sys.argv[1] == 'break'  # 칸 첫 줄에 'BiLSTM' + 줄바꿈을 넣고 잼
STEPS = [x / 2 for x in range(0, 31)]  # 0 ~ 15pt 덧폭


def el(tag, **a):
    e = OxmlElement(tag)
    for k, v in a.items():
        e.set(qn(k), str(v))
    return e


def run_font(r, bold=False):
    rPr = r._element.get_or_add_rPr()
    rf = OxmlElement('w:rFonts')
    for a in ('w:ascii', 'w:eastAsia', 'w:hAnsi', 'w:cs'):
        rf.set(qn(a), '굴림')
    rPr.insert(0, rf)
    r.font.size = Pt(SIZE)
    r.font.bold = bold
    k = OxmlElement('w:kern')
    k.set(qn('w:val'), '0')
    rPr.append(k)


doc = Document(TEMPLATE)
body = doc.element.body
for ch in list(body):
    if ch.tag != qn('w:sectPr'):
        body.remove(ch)
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(1.0)
# 1) 문단 폭 재기용
for s in TESTS:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    run_font(p.add_run(s))
# 2) 칸 폭별 표: 문자열마다 기준 폭(문단 폭 추정치 - 2pt)부터
from PIL import ImageFont  # noqa: E402

G = ImageFont.truetype(r'C:\Windows\Fonts\gulim.ttc', 1000, index=0)
plan = []
for s in TESTS:
    base = G.getlength(s) / 1000 * SIZE + s.count(' ') * (0.5 - 0.333) * SIZE - 2
    for add in STEPS:
        wpt = base + add
        tw = int(round(wpt * 20))
        t = doc.add_table(rows=1, cols=2)
        tblPr = t._tbl.tblPr
        for ch in list(tblPr):
            tblPr.remove(ch)
        tblPr.append(el('w:tblW', **{'w:w': tw + 2000, 'w:type': 'dxa'}))
        tblPr.append(el('w:tblCellSpacing', **{'w:w': 15, 'w:type': 'dxa'}))
        tblPr.append(el('w:tblLayout', **{'w:type': 'fixed'}))
        mar = el('w:tblCellMar')
        for side in ('top', 'left', 'bottom', 'right'):
            mar.append(el(f'w:{side}', **{'w:w': 15, 'w:type': 'dxa'}))
        tblPr.append(mar)
        gcs = t._tbl.tblGrid.findall(qn('w:gridCol'))
        gcs[0].set(qn('w:w'), str(tw))
        gcs[1].set(qn('w:w'), '2000')
        for ci, cell in enumerate(t.rows[0].cells):
            tcPr = cell._tc.get_or_add_tcPr()
            for ch in list(tcPr):
                tcPr.remove(ch)
            tcPr.append(el('w:tcW', **{'w:w': tw if ci == 0 else 2000, 'w:type': 'dxa'}))
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if ci == 0 and BREAK_FIRST:
                run_font(p.add_run('BiLSTM'))
                rb = p.add_run()
                rb.add_break()
                run_font(rb)
            run_font(p.add_run(s if ci == 0 else 'x'))
        doc.add_paragraph()
        plan.append((s, wpt))
doc.save(TMP)

pythoncom.CoInitialize()
word = win32.DispatchEx('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
d = None
out = {}
try:
    d = word.Documents.Open(TMP, ReadOnly=True, AddToRecentFiles=False)
    d.Repaginate()
    para_w = {}
    for i, s in enumerate(TESTS, start=1):
        rng = d.Paragraphs(i).Range
        para_w[s] = d.Range(rng.End - 1, rng.End - 1).Information(5) - rng.Characters(1).Information(5)
    for ti, (s, wpt) in enumerate(plan, start=1):
        cell = d.Tables(ti).Cell(1, 1).Range
        first = cell.Characters(1).Information(10)  # wdFirstCharacterLineNumber
        last = d.Range(cell.End - 2, cell.End - 1).Information(10)
        lines = last - first + 1
        if lines == (2 if BREAK_FIRST else 1) and s not in out:
            out[s] = wpt
finally:
    if d is not None:
        d.Close(SaveChanges=0)
    word.Quit(SaveChanges=0)
    pythoncom.CoUninitialize()

for s in TESTS:
    mw = out.get(s)
    print(f'{s}: 문단 폭 {para_w[s]:.2f}pt, 한 줄 최소 칸 폭 {mw}pt, 덧폭 {None if mw is None else round(mw - para_w[s], 2)}pt')
