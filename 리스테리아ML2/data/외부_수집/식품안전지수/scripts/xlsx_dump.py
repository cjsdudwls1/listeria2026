# 엑셀 시트의 지정 범위 값을 출력(읽기만). 메모가 있으면 함께.
# 사용: python -I xlsx_dump.py <xlsx> <시트이름> <범위 예: AL1:AQ20>
import sys, openpyxl
sys.stdout.reconfigure(encoding="utf-8")
path, sheet, rng = sys.argv[1], sys.argv[2], sys.argv[3]
wb = openpyxl.load_workbook(path, data_only=False)
ws = wb[sheet]
for row in ws[rng]:
    out = []
    for c in row:
        if c.value is not None or c.comment:
            s = f'{c.coordinate}={c.value!r}'
            if c.comment:
                s += f' [메모:{c.comment.text!r}]'
            out.append(s)
    if out:
        print(' | '.join(out))
