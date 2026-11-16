import sys
from collections import Counter
import openpyxl

wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
ws = wb.worksheets[0]
rows = ws.iter_rows(values_only=True)
h = next(rows)
isp = h.index('가축종')
TOK = ('젖소', '산양', '염소', '면양')
c = Counter(); n = 0; n_any = 0; n_cow = 0
for r in rows:
    s = str(r[isp] or '')
    hit = [t for t in TOK if t in s]
    if hit:
        n_any += 1
        for t in hit:
            c[t] += 1
        if '젖소' in s:
            n_cow += 1
    n += 1
open(sys.argv[2], 'w', encoding='utf-8').write(f'전체 {n}, 낙농장(젖소·산양·염소·면양 하나라도) {n_any}, 젖소만 기준 {n_cow}, 축종별 {dict(c)}')
