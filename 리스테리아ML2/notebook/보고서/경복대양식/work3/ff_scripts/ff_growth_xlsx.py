# -*- coding: utf-8 -*-
"""리스테리아 생장식.xlsx 시트 '(완료) 리스테리아 모노사이토제네스' 에서 쓴 행(6·15·19·21·26·28·41)과 머리 행 글자(읽기만)."""
import sys
from pathlib import Path
import openpyxl
sys.stdout.reconfigure(encoding="utf-8")
p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋\리스테리아 생장식.xlsx")
wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
print(wb.sheetnames)
ws = wb["(완료) 리스테리아 모노사이토제네스"]
for r in [1, 2, 3, 4, 6, 15, 19, 21, 26, 28, 34, 41]:
    vals = []
    for c in ws.iter_rows(min_row=r, max_row=r):
        for cell in c:
            if cell.value is not None and str(cell.value).strip():
                vals.append(f"{cell.coordinate}={str(cell.value)[:300]}")
    print(f"--- 행 {r}: " + " ; ".join(vals))
