# -*- coding: utf-8 -*-
"""리서치팀 '위해요소별_중요인자_데이터출처.xlsx' 리스테리아 행 출력(빨간 글자 표시, 읽기만)."""
import sys
from pathlib import Path
import openpyxl
sys.stdout.reconfigure(encoding="utf-8")
p = Path(r"C:\Users\NT751\Desktop\천영진\문서\리서치팀문서\위해요소별_중요인자_데이터출처.xlsx")
wb = openpyxl.load_workbook(p, rich_text=True)
for ws in wb.worksheets:
    print("=" * 10, ws.title, ws.max_row, ws.max_column)
    for row in ws.iter_rows():
        vals = []
        for c in row:
            v = c.value
            if v is None:
                continue
            if hasattr(v, "__iter__") and not isinstance(v, str):
                parts = []
                for part in v:
                    if isinstance(part, str):
                        parts.append(part)
                    else:
                        col = getattr(getattr(part.font, "color", None), "rgb", None) if part.font else None
                        t = part.text
                        parts.append(f"[RED:{t}]" if col and str(col).upper().endswith("FF0000") else t)
                s = "".join(parts)
            else:
                col = getattr(getattr(c.font, "color", None), "rgb", None)
                s = str(v)
                if col and str(col).upper().endswith("FF0000"):
                    s = f"[RED:{s}]"
            vals.append(f"{c.coordinate}={s[:200]}")
        if vals:
            print(" ; ".join(vals))
