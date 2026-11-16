# -*- coding: utf-8 -*-
"""파일목록_MD5_SHA256.csv → md5sum/sha256sum -c 형식 두 파일(독립 대조용).
사용: python3.12 -I csv_to_sums.py <csv> <out_md5> <out_sha256>"""
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
with open(sys.argv[2], "w", encoding="utf-8") as fm, open(sys.argv[3], "w", encoding="utf-8") as fs:
    for r in rows:
        fm.write(f"{r['MD5']}  {r['경로']}\n")
        fs.write(f"{r['SHA256']}  {r['경로']}\n")
print(len(rows), "rows; total bytes", sum(int(r["크기(바이트)"]) for r in rows))
