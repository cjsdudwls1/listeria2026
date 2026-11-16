import pyarrow.parquet as pq, pyarrow.compute as pc, openpyxl, csv, json, datetime as dt, collections
B=r'C:\Users\NT751\Desktop\천영진'
t=pq.read_table(B+r'\리스테리아ML\data\listeria_data.parquet',columns=['src_route','mnfctur_de','judge_result'])
d=t.to_pydict()
from collections import Counter
c=Counter()
for r,m in zip(d['src_route'],d['mnfctur_de']):
    if r=='domestic':
        s=str(m)[:10]
        if s<='2015-06-05': c[s]+=1
print(sorted(c.items()))
s=pq.read_table(B+r'\리스테리아ML2\data\데이터셋\시군구_대표좌표.parquet').to_pydict()
print(list(s.keys()), len(s['sgg_code']))
nm=dict(zip(s['sgg_nm'],s['sgg_code']))
wb=openpyxl.load_workbook(B+r'\리스테리아ML2\data\외부_수집\축사\축사_시군구별_농장수.xlsx',read_only=True,data_only=True)
print(wb.sheetnames)
ws=wb['시군구별_농장수']; rows=list(ws.iter_rows(values_only=True)); print(rows[0]); print(len(rows)-1)
miss=[]
for r in rows[1:]:
    key = r[0] if r[0]==r[1] else f'{r[0]} {r[1]}'
    if key not in nm: miss.append((r[0],r[1]))
    if '구례' in str(r[1]) or '세종' in str(r[0]) or '동해' in str(r[1]) or '화성' in str(r[1]): print(r, nm.get(key))
print('miss',miss)
with open(B+r'\리스테리아ML2\data\외부_수집\축사\국가리스트_1_농장전체.csv',encoding='utf-8-sig') as f:
    rr=list(csv.DictReader(f))
print(list(rr[0].keys()), len(rr), sum(1 for x in rr if float(x['farm_total'])==0))
g=json.load(open(B+r'\리스테리아ML2\data\외부_수집\축사\ne_10m_admin_0_countries_v5.1.2.geojson',encoding='utf-8'))
cc=Counter()
for ft in g['features']:
    p=ft['properties']; k=p.get('ISO_A3_EH');
    if not k or k=='-99': k=p.get('ADM0_A3')
    cc[k]+=1
print([k for k,v in cc.items() if v>1])
