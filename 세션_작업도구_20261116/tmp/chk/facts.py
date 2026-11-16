import os, json
import pyarrow.parquet as pq
import openpyxl, csv
base = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data'
t = pq.read_table(os.path.join(base, r'데이터셋\시군구_대표좌표.parquet')).to_pylist()
print('coord rows', len(t), list(t[0].keys()))
nm = {r['sgg_nm']: r['sgg_code'] for r in t}
for k in ['전남광주통합특별시 구례군', '세종특별자치시']:
    print(k, nm.get(k))
print('codes starting 46/29:', [r['sgg_code'] for r in t if str(r['sgg_code'])[:2] in ('46', '29')][:5])
print('28720', [r for r in t if str(r['sgg_code']) == '28720'])
d = os.path.join(base, r'외부_수집\축사')
print(os.listdir(d))
wb = openpyxl.load_workbook(os.path.join(d, '축사_시군구별_농장수.xlsx'), read_only=True, data_only=True)
print(wb.sheetnames)
ws = wb['시군구별_농장수']
rows = list(ws.iter_rows(values_only=True))
print(rows[0])
data = [r for r in rows[1:] if r and r[0]]
print('n', len(data))
miss = []
for r in data:
    key = f'{r[0]} {r[1]}'
    code = nm.get(key)
    if code is None and r[0] == r[1]:
        code = nm.get(r[0])
    print(r[:6], '->', code)
