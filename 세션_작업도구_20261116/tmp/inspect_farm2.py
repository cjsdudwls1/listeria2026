import sys
from collections import Counter
import openpyxl

xlsx = sys.argv[1]
wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
ws = wb.worksheets[0]
rows = ws.iter_rows(values_only=True)
header = next(rows)
ia, iy, ib, isp = header.index('소재지 주소'), header.index('조사연도'), header.index('데이터기준일'), header.index('가축종')
reg, yrs, bases, dairy, empty_addr = Counter(), Counter(), Counter(), 0, 0
n = 0
for r in rows:
    n += 1
    a = (r[ia] or '').split()
    if len(a) >= 2:
        reg[(a[0], a[1])] += 1
    else:
        empty_addr += 1
    yrs[str(r[iy])[:10]] += 1
    bases[str(r[ib])[:10]] += 1
    if r[isp] and '젖소' in str(r[isp]):
        dairy += 1
out = [f'rows={n} empty_or_short_addr={empty_addr} dairy_rows(젖소 포함)={dairy}',
       f'distinct 시도+시군구 = {len(reg)}',
       f'조사연도 = {dict(yrs.most_common(10))}',
       f'데이터기준일 = {dict(bases.most_common(10))}',
       '시도별 시군구 수·농장 수 = ' + str({s: (len([1 for (a, b) in reg if a == s]), sum(v for (a, b), v in reg.items() if a == s)) for s in sorted({a for a, b in reg})})]
open(sys.argv[2], 'w', encoding='utf-8').write('\n'.join(out))
