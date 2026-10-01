import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import openpyxl

# 테이블정의서 xlsx 2개(모델팀본, 20260901본)의 모든 시트에서 낱말이 들어 있는 행을 그대로 출력한다.
# 사용: py find_in_table_spec.py t_aw_cl [낱말 ...]
FILES = [Path(r"C:\Users\NT751\Desktop\천영진\테이블명세서(모델팀)\테이블정의서.xlsx"),
         Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\docs\참고자료\20260901.테이블정의서 (1).xlsx")]
words = [w.lower() for w in sys.argv[1:]]

for f in FILES:
    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
    print(f"\n==== {f.name} | 시트 {wb.sheetnames}")
    for ws in wb.worksheets:
        for i, row in enumerate(ws.iter_rows(values_only=True), 1):
            cells = [str(c) for c in row if c is not None]
            if any(w in " ".join(cells).lower() for w in words):
                print(f"[{ws.title}] {i}행: {cells}")
