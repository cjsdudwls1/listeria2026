# -*- coding: utf-8 -*-
"""색칠된 행 상세: 소속 표(위쪽 표 머리행), 컬럼, 타입, 설명, 칠해진 열 — 읽기만"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import openpyxl
P = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\docs\20260901.테이블정의서 (1).xlsx"
wb = openpyxl.load_workbook(P)
def ck(c):
    f = c.fill
    if f is None or f.fill_type in (None, 'none'): return None
    fg = f.fgColor
    if fg.type == 'rgb': return fg.rgb
    if fg.type == 'theme': return f"theme{fg.theme}"
    return str(fg.type)
TARGET = {'FFFFFF00': '노랑', 'FF92D050': '초록', 'FFFF0000': '빨강', 'FFFFF3D6': '연노랑(크림)', 'theme0': '흰색(theme0)'}
OUT = []
for ws in wb.worksheets:
    header_rows = [r for r in range(2, ws.max_row + 1) if isinstance(ws.cell(r, 1).value, str) and ws.cell(r, 2).value is None and '(' in ws.cell(r, 1).value]
    def table_of(r):
        hs = [h for h in header_rows if h <= r]
        return ws.cell(hs[-1], 1).value.split('(')[0].strip() if hs else None
    for r in range(1, ws.max_row + 1):
        cols = {}
        for cc in range(1, ws.max_column + 1):
            k = ck(ws.cell(r, cc))
            if k in TARGET: cols.setdefault(TARGET[k], []).append(cc)
        if not cols or r == 1 or r in header_rows: continue
        if set(cols) == {'흰색(theme0)'} and ws.title != 'api_data_info': continue
        vals = [ws.cell(r, cc).value for cc in range(1, 8)]
        rec = {'sheet': ws.title, 'row': r, 'table': table_of(r), 'seq': vals[1], 'column': vals[2], 'type': vals[3], 'len': vals[4], 'null': vals[5], 'desc': vals[6], 'colors': cols,
               'font_colors': {cc: (ws.cell(r, cc).font.color.rgb if ws.cell(r, cc).font and ws.cell(r, cc).font.color is not None and ws.cell(r, cc).font.color.type == 'rgb' else None) for cc in range(1, 8)}}
        OUT.append(rec)
for x in OUT:
    if x['sheet'] == 'ba_data 뷰': continue
    print(f"{x['sheet']} r{x['row']} [{x['table']}] {x['seq']} {x['column']} | {x['type']} | {x['desc']} | 색 {x['colors']}")
print('\n[ba_data 뷰 크림색 행 요약]')
for x in OUT:
    if x['sheet'] == 'ba_data 뷰':
        print(f"  r{x['row']} [{x['table']}] {x['column']} | {x['desc']} | 색 {x['colors']}")
# 표 머리행 목록(api_data_info, ba_data 테이블)
for ws in wb.worksheets:
    print('\n', ws.title, '표 목록:', [ws.cell(r, 1).value for r in range(2, ws.max_row + 1) if isinstance(ws.cell(r, 1).value, str) and ws.cell(r, 2).value is None])
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\td_colored_rows.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
