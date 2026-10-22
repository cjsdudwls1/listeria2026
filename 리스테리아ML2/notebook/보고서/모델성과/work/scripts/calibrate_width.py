# -*- coding: utf-8 -*-
"""Word 가 굴림 글자를 실제로 얼마나 넓게 놓는지 재서 PIL 측정값과 견줌(칸 폭 계산 보정용).

임시 docx(%TEMP%)에 시험 문자열을 문단마다 넣고, 내가 띄운 Word(DispatchEx)로 열어
문단 첫 글자 왼쪽 위치와 문단 기호 위치의 차(pt)를 잰다. 결과는 화면에만 출력.
"""
import os
import sys

import pythoncom
import win32com.client as win32
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm
from PIL import ImageFont

TMP = os.path.join(os.environ['TEMP'], 'listeria_perf_check', 'calib.docx')
TEMPLATE = r'C:\Users\NT751\Desktop\천영진\모델 성과 발표 예시.docx'
G = ImageFont.truetype(r'C:\Windows\Fonts\gulim.ttc', 1000, index=0)
SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 8.5

TESTS = ['0.0718', '27,554.1', '19,204.3', 'Accuracy', 'Precision', 'PR-AUC', '기상 시퀀스 90일', '기상 시퀀스',
         '1차 베이스라인', '튜닝 18번 후보', '80일(채택)', '학습 2015-04-07~2022-06-30 → 맞혀 봄 2022-07-01~2023-06-30',
         '가가가가가가가가가가', '0000000000', 'aaaaaaaaaa', '가0가0가0가0가0', '가 0 가 0 가 0', '검정일 T−1,095 ~ T−1',
         'DB 표 api_data_info 의 kma_asos_daily_info(ASOS)·', 'KOSIS_시군구_논밭면적_2008_2025_20251231.csv']


def run_font(r, bold):
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
for bold in (False, True):
    for s in TESTS:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(s)
        run_font(r, bold)
doc.save(TMP)

pythoncom.CoInitialize()
word = win32.DispatchEx('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
d = None
try:
    d = word.Documents.Open(TMP, ReadOnly=True, AddToRecentFiles=False)
    d.Repaginate()
    res = []
    for i in range(1, d.Paragraphs.Count + 1):
        rng = d.Paragraphs(i).Range
        x0 = rng.Characters(1).Information(5)
        x1 = d.Range(rng.End - 1, rng.End - 1).Information(5)
        res.append((rng.Text.rstrip('\r'), x1 - x0))
finally:
    if d is not None:
        d.Close(SaveChanges=0)
    word.Quit(SaveChanges=0)
    pythoncom.CoUninitialize()

n = len(TESTS)
for i, (s, w) in enumerate(res):
    bold = i >= n
    pil = G.getlength(s) / 1000 * SIZE
    print(f"{'B' if bold else 'R'} word {w:7.2f}pt  pil {pil:7.2f}pt  ratio {w / pil:5.3f}  diff {w - pil:6.2f}  {s}")
