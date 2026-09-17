# -*- coding: utf-8 -*-
"""20260901.테이블정의서 (1).xlsx: 시트 구조와 색이 칠해진 셀(채우기 색) 추출 — 읽기만"""
import sys, json
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')
import openpyxl
P = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\docs\참고자료\20260901.테이블정의서 (1).xlsx"
wb = openpyxl.load_workbook(P)
print('시트:', [(ws.title, ws.max_row, ws.max_column) for ws in wb.worksheets])
def color_key(c):
    f = c.fill
    if f is None or f.fill_type in (None, 'none'):
        return None
    fg = f.fgColor
    if fg is None: return None
    if fg.type == 'rgb': return f"rgb:{fg.rgb}"
    if fg.type == 'theme': return f"theme:{fg.theme}:tint{round(fg.tint or 0, 2)}"
    if fg.type == 'indexed': return f"indexed:{fg.indexed}"
    return f"{fg.type}"
OUT = {}
for ws in wb.worksheets:
    cnt = Counter(); rows_by_color = defaultdict(set)
    for row in ws.iter_rows():
        for c in row:
            k = color_key(c)
            if k:
                cnt[k] += 1; rows_by_color[k].add(c.row)
    print(f'\n[{ws.title}] 채우기 색 분포:', dict(cnt))
    for k, rs in rows_by_color.items():
        rs = sorted(rs)
        print('  ', k, '행 수', len(rs), '예:', rs[:15])
    # 헤더 추정: 처음 6행 출력
    for r in range(1, min(ws.max_row, 6) + 1):
        print('   row', r, [ws.cell(r, cc).value for cc in range(1, min(ws.max_column, 12) + 1)])
