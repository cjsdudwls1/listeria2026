# 엑셀(.xlsx)에서 키워드가 들어간 칸과 메모를 찾아 출력(읽기만).
# 사용: python -I xlsx_search.py <xlsx> <키워드1,키워드2,...> [행 앞뒤 칸 출력 여부 1]
import sys, openpyxl
sys.stdout.reconfigure(encoding="utf-8")

path, kws = sys.argv[1], sys.argv[2].split(',')
ctx = len(sys.argv) > 3 and sys.argv[3] == '1'
wb = openpyxl.load_workbook(path, data_only=False)
print('SHEETS:', [(ws.title, ws.max_row, ws.max_column) for ws in wb.worksheets])
for ws in wb.worksheets:
    for row in ws.iter_rows():
        for c in row:
            v = c.value
            hit = isinstance(v, str) and any(k in v for k in kws)
            com = c.comment.text if c.comment else None
            chit = com and any(k in com for k in kws)
            if hit or chit:
                print(f'[{ws.title}!{c.coordinate}] {repr(v)[:300]}' + (f'  || 메모: {com[:300]!r}' if com else ''))
                if ctx:
                    vals = [ws.cell(row=c.row, column=j).value for j in range(1, min(ws.max_column, 30) + 1)]
                    print('   ROW:', [x for x in vals if x is not None][:30])
