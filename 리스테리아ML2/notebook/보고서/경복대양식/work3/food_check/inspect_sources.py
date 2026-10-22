# -*- coding: utf-8 -*-
"""엑셀 원본·DB 추출본의 겉모습 확인 (읽기만): 시트 목록·머리말·파일 속성(작성자 등)·parquet 메타데이터"""
import sys
import zipfile
from pathlib import Path

import openpyxl
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

XLSX = Path(r'C:\Users\NT751\Desktop\천영진\문서\리서치팀문서\통합_수분활성도_수분함량_식품pH_염도_최종.xlsx')
RAW = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\raw\raw_t_aw_cl.parquet')

print('=== 엑셀 파일', XLSX, XLSX.stat().st_size, 'bytes')
import datetime as dt
print('파일 수정 시각(로컬):', dt.datetime.fromtimestamp(XLSX.stat().st_mtime))
with zipfile.ZipFile(XLSX) as z:
    names = z.namelist()
    print('zip 안 파일:', names)
    for n in ('docProps/core.xml', 'docProps/app.xml', 'docProps/custom.xml'):
        if n in names:
            print(f'--- {n}')
            print(z.read(n).decode('utf-8', errors='replace'))

wb = openpyxl.load_workbook(XLSX, read_only=False, data_only=True)
print('\n시트:', wb.sheetnames)
p = wb.properties
print('properties: creator=%r lastModifiedBy=%r created=%r modified=%r title=%r subject=%r description=%r keywords=%r category=%r company?' % (
    p.creator, p.lastModifiedBy, p.created, p.modified, p.title, p.subject, p.description, p.keywords, p.category))
for ws in wb.worksheets:
    print(f'\n=== 시트 {ws.title!r} 크기 {ws.max_row}행 × {ws.max_column}열, dims={ws.dimensions}')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i >= 8:
            break
        print('  ', row)
    # 메모(comment)·병합 셀
    cm = [(c.coordinate, c.comment.text, c.comment.author) for r in ws.iter_rows() for c in r if c.comment]
    if cm:
        print('  메모:', cm[:20])
    if ws.merged_cells.ranges:
        print('  병합:', list(ws.merged_cells.ranges)[:10])
wb.close()

print('\n=== DB 추출본', RAW, RAW.stat().st_size, 'bytes', dt.datetime.fromtimestamp(RAW.stat().st_mtime))
pf = pq.ParquetFile(RAW)
print('행 수:', pf.metadata.num_rows)
print('스키마:', pf.schema_arrow)
md = pf.schema_arrow.metadata or {}
for k, v in md.items():
    if k == b'pandas':
        continue
    print('  메타', k.decode(), '=', v.decode('utf-8', errors='replace')[:2000])
