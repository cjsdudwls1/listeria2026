"""확인용(읽기 전용): 시료 파일 src_route 값, 뭉치 정의 '학습 행'이 시료 수인지, 식품 속성 표 기간 단서."""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

B = 'C:/Users/NT751/Desktop/천영진/리스테리아ML2/data'
S = os.path.join(B, '데이터셋', '시료_시군구.parquet')
s = pq.read_table(S, columns=['lims_data_key', 'src_route', 'mnfctur_de', 'judge_result']).to_pandas()
print('src_route', s['src_route'].astype(str).value_counts(dropna=False).to_dict())
nb = pq.read_table(os.path.join(B, '데이터셋', '제품명', '시료별_뭉치.parquet')).to_pandas()
print('nb cols', list(nb.columns)[:5], len(nb))
m = s.merge(nb, on='lims_data_key', how='inner')
print('merged', len(m))
d = pd.to_datetime(m['mnfctur_de'])
lm = d <= pd.Timestamp('2025-06-30')
pos = (m['judge_result'].astype(str) == '검출')
for c in ['nb_enoki', 'nb_preheat', 'nb_extract', 'nb_raw_meat']:
    print(c, int((m[c] == 1)[lm].sum()), int(((m[c] == 1) & pos)[lm].sum()))
f = os.path.join(B, '데이터셋', '식정원_db_식품', '품목3단계_식품속성.parquet')
pf = pq.ParquetFile(f)
print('food schema', pf.schema_arrow.names)
md = pf.schema_arrow.metadata or {}
for k, v in md.items():
    kk = k.decode('utf-8', 'replace')
    if kk != 'pandas':
        print('meta', kk, v.decode('utf-8', 'replace')[:500])
g = os.path.join(B, '중간', '품목3단계_식품속성_근거.csv')
gd = pd.read_csv(g, encoding='utf-8-sig', nrows=3)
print('근거 cols', list(gd.columns))
