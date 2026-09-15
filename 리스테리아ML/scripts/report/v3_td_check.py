# -*- coding: utf-8 -*-
"""추가 섹션 점검: 기존 행·시트가 직전 버전과 같은지, 새 섹션 전문, 금지어"""
import sys, openpyxl
sys.stdout.reconfigure(encoding='utf-8')
old = openpyxl.load_workbook(sys.argv[1]); new = openpyxl.load_workbook(sys.argv[2])
print('시트:', new.sheetnames)
for name in old.sheetnames:
    wo, wn = old[name], new[name]
    diffs = []
    for r in range(1, wo.max_row + 1):
        for c in range(1, wo.max_column + 1):
            if wo.cell(r, c).value != wn.cell(r, c).value:
                diffs.append((r, c))
    extra = [(r, wn.cell(r, 1).value) for r in range(wo.max_row + 1, wn.max_row + 1) if wn.cell(r, 1).value is not None]
    print(f'[{name}] 기존 {wo.max_row}행 중 달라진 칸 {len(diffs)}개, 새로 붙은 행 {len(extra)}개(첫 행 {extra[0][0] if extra else "-"})')
ws = new['0_문자보고']
BAN = ['출처 r', '부분·대체', "{'", 'nan', 'None', '센서']
import re
for r in range(134, ws.max_row + 1):
    v = ws.cell(r, 1).value
    if v is None: continue
    print(f'{r:3d} {v}')
    for b in BAN:
        if b in v: print('   [금지어]', b)
    for m in re.findall(r'[a-z]+_[a-z0-9_]+', v):
        print('   [영문 식별자]', m)
