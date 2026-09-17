# -*- coding: utf-8 -*-
"""v3 보고서 점검: 시트명, 금지 문구, 0번 시트 영문 컬럼명 잔존, 판정 값, 0번 시트 전문 출력"""
import sys, re, openpyxl
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
P = sys.argv[1]
wb = openpyxl.load_workbook(P)
print('시트:', wb.sheetnames)
BAN = ['출처 r', '부분·대체', '부분,대체', 'aflatoxin', 'train_prdlst', 'wx160', "{'", 'nan', 'None', '센서', '해당 변수 관측 지점', 'NULL·0 혼재', '결정 필요:']
for ws in wb.worksheets:
    for row in ws.iter_rows():
        for c in row:
            if isinstance(c.value, str):
                for b in BAN:
                    if b in c.value:
                        print(f'[금지어 {b!r}] {ws.title}!{c.coordinate}: {c.value[:160]}')
                if '아플라톡신' in c.value:
                    print(f'[아플라톡신 언급] {ws.title}!{c.coordinate}: {c.value[:160]}')
ws0 = wb['0_문자보고']
col_pat = re.compile(r'[a-z]+_[a-z0-9_]+')
print('\n=== 0번 시트 영문 식별자(표·파일명 제외 확인용) ===')
toks = Counter()
for row in ws0.iter_rows():
    for c in row:
        if isinstance(c.value, str):
            for m in col_pat.findall(c.value):
                toks[m] += 1
print(dict(toks))
print('\n=== 1번 시트 판정(G열) 분포 ===')
ws1 = wb['1_수집가능여부']
print(Counter(ws1.cell(r, 7).value for r in range(7, ws1.max_row + 1) if ws1.cell(r, 1).value and ws1.cell(r, 7).value and ws1.cell(r, 5).value))
print('B열 근거 예:', [ws1.cell(r, 3).value for r in (7, 12, 25)])
print('\n=== 0번 시트 전문 ===')
for r in range(1, ws0.max_row + 1):
    v = ws0.cell(r, 1).value
    if v is not None:
        print(f'{r:3d} {v}')
