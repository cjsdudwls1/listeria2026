# -*- coding: utf-8 -*-
"""CSV 를 칸 이름: 값 형태로 보기 (읽기 전용). 사용: peek_csv.py 파일 [칸1,칸2,...] [최대행]"""
import csv
import sys

path = sys.argv[1]
cols = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] else None
maxrows = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
with open(path, encoding='utf-8-sig', newline='') as f:
    rd = csv.DictReader(f)
    print('COLUMNS:', rd.fieldnames)
    for i, row in enumerate(rd):
        if i >= maxrows:
            break
        if cols:
            print(' | '.join(f'{c}={row.get(c)}' for c in cols))
        else:
            print(i, row)
