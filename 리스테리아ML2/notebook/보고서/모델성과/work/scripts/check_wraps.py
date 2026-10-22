# -*- coding: utf-8 -*-
"""Word 로 열어 표 칸마다 실제 줄 수가 넣은 줄 수(줄바꿈 + 1)보다 많은 칸(뜻하지 않은 줄바꿈)을 찾음. 읽기 전용.
사용: check_wraps.py 문서.docx [무시할 표 번호들(쉼표)]
"""
import sys

import pythoncom
import win32com.client as win32

path = sys.argv[1]
skip = {int(x) for x in sys.argv[2].split(',')} if len(sys.argv) > 2 and sys.argv[2] else set()

pythoncom.CoInitialize()
word = win32.DispatchEx('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
d = None
bad = 0
try:
    d = word.Documents.Open(path, ReadOnly=True, AddToRecentFiles=False)
    d.Repaginate()
    for ti in range(1, d.Tables.Count + 1):
        if ti in skip:
            continue
        t = d.Tables(ti)
        for c in t.Range.Cells:
            r = c.Range
            txt = r.Text[:-2] if r.Text.endswith('\r\x07') else r.Text
            if not txt.strip():
                continue
            want = txt.count('\x0b') + 1
            first = r.Characters(1).Information(10)
            last = d.Range(r.End - 2, r.End - 1).Information(10)
            got = last - first + 1
            if got != want:
                bad += 1
                print(f'표 {ti} 칸({c.RowIndex},{c.ColumnIndex}): 줄 {got} (넣은 줄 {want}) | {txt!r}')
    print('표 수', d.Tables.Count, '뜻하지 않은 줄바꿈 칸', bad)
finally:
    if d is not None:
        d.Close(SaveChanges=0)
    word.Quit(SaveChanges=0)
    pythoncom.CoUninitialize()
