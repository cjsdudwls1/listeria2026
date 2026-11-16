import pyarrow.parquet as pq
from collections import Counter
B=r'C:\Users\NT751\Desktop\천영진'
s=pq.read_table(B+r'\리스테리아ML2\data\데이터셋\시군구_대표좌표.parquet').to_pydict()
print(Counter(s['source']))
for i,c in enumerate(s['sgg_code']):
    if c in ('28720','12730','36110') or s['source'][i]!=Counter(s['source']).most_common(1)[0][0]: print(c, s['sgg_nm'][i], s['source'][i])
print(sorted(set(x[:2] for x in s['sgg_code'])))
