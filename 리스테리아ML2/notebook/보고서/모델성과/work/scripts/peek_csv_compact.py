# -*- coding: utf-8 -*-
"""CSV 를 줄인 표로 보기 (숫자 반올림). 사용: peek_csv_compact.py 파일 칸1,칸2,... [자릿수]"""
import csv
import sys

path = sys.argv[1]
cols = sys.argv[2].split(',')
nd = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def fmt(v):
    try:
        f = float(v)
        if f.is_integer() and abs(f) > 999:
            return str(int(f))
        return f'{f:.{nd}f}'
    except Exception:
        return v


with open(path, encoding='utf-8-sig', newline='') as f:
    rd = csv.DictReader(f)
    missing = [c for c in cols if c not in rd.fieldnames]
    if missing:
        print('MISSING', missing)
        print('COLUMNS', rd.fieldnames)
    print('\t'.join(cols))
    for row in rd:
        print('\t'.join(fmt(row.get(c, '')) for c in cols))
