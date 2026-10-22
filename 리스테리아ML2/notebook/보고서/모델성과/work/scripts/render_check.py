# -*- coding: utf-8 -*-
"""Word COM 으로 쪽 수 확인 + 임시 PDF → PNG(검수용).

사용: render_check.py 문서.docx 임시폴더 [dpi]
- 내가 띄운 Word 인스턴스(DispatchEx)만 쓰고, 그 인스턴스만 닫는다.
- 문서는 읽기 전용으로 연다(원본 저장 안 함).
- PDF·PNG 는 임시폴더(결과 폴더 밖, %TEMP% 아래)에만 만든다. PDF 는 PNG 를 만든 뒤 지운다.
"""
import os
import sys

import pythoncom
import win32com.client as win32

docx = os.path.abspath(sys.argv[1])
outdir = os.path.abspath(sys.argv[2])
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 110
assert 'Temp' in outdir or 'TEMP' in outdir or 'tmp' in outdir.lower(), '임시폴더만 허용'
os.makedirs(outdir, exist_ok=True)
pdf = os.path.join(outdir, 'check.pdf')

pythoncom.CoInitialize()
word = win32.DispatchEx('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = None
try:
    doc = word.Documents.Open(docx, ConfirmConversions=False, ReadOnly=True, AddToRecentFiles=False)
    doc.Repaginate()
    print('WORD_STAT_PAGES', doc.ComputeStatistics(2))  # wdStatisticPages
    doc.ExportAsFixedFormat(pdf, 17)  # wdExportFormatPDF
    print('WORD_INFO_PAGES', doc.Content.Information(4))  # wdNumberOfPagesInDocument(내보내기 뒤 배치 기준)
finally:
    if doc is not None:
        doc.Close(SaveChanges=0)
    word.Quit(SaveChanges=0)
    pythoncom.CoUninitialize()

import fitz  # noqa: E402

d = fitz.open(pdf)
print('PDF_PAGES', d.page_count)
for i, page in enumerate(d):
    pix = page.get_pixmap(dpi=dpi)
    p = os.path.join(outdir, f'page-{i + 1:02d}.png')
    pix.save(p)
    print(p)
d.close()
os.remove(pdf)
