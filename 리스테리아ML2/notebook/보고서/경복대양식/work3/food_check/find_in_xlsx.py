# -*- coding: utf-8 -*-
"""엑셀 파일들에서 글자 찾기 (읽기만). 사용: python find_in_xlsx.py <정규식> <xlsx 경로> [<xlsx 경로> ...]"""
import re
import sys

import openpyxl

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

pat = re.compile(sys.argv[1], re.I)
for path in sys.argv[2:]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        head = None
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                head = row
            if any(v is not None and pat.search(str(v)) for v in row):
                print(f'[{path.split(chr(92))[-1]} / {ws.title} / {i + 1}행]', [v for v in row if v is not None])
    wb.close()
