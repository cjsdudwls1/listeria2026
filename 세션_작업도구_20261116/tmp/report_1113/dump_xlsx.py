# -*- coding: utf-8 -*-
import sys, openpyxl
path, out = sys.argv[1], sys.argv[2]
wb = openpyxl.load_workbook(path, data_only=True)
L = []
for ws in wb.worksheets:
    L.append(f"===== SHEET {ws.title} dims={ws.dimensions} max_row={ws.max_row} max_col={ws.max_column}")
    for row in ws.iter_rows(values_only=True):
        if all(v is None for v in row):
            continue
        L.append(" | ".join("" if v is None else str(v) for v in row))
open(out, "w", encoding="utf-8").write("\n".join(L))
print("ok")
