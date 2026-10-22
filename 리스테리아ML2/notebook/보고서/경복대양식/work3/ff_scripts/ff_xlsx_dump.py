# -*- coding: utf-8 -*-
"""식품안전지수 protocol 엑셀: 시트별로 값 있는 칸을 모두 찍기(읽기만)."""
import sys
from pathlib import Path
import openpyxl
sys.stdout.reconfigure(encoding="utf-8")
p = Path(r"C:\Users\NT751\Desktop\천영진\문서\영진문서\식품안전지수_protocol_20190103.xlsx")
wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
want = sys.argv[1:] or wb.sheetnames
for ws in wb.worksheets:
    if ws.title not in want:
        continue
    print("=" * 20, ws.title, ws.max_row, ws.max_column)
    for row in ws.iter_rows():
        cells = [(c.coordinate, c.value) for c in row if c.value is not None and str(c.value).strip() != ""]
        if cells:
            print(" ; ".join(f"{a}={str(v)[:60]}" for a, v in cells))
