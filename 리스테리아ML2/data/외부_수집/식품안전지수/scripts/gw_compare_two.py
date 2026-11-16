# 그림 PDF 두 번 옮긴 결과를 칸마다 대조(읽기만).
# 사용: python -I gw_compare_two.py <1차csv> <2차csv> <출력csv>
# 짝짓기: (판, PDF쪽, 쪽안순번). 이름·숫자 칸 모두 글자 그대로 비교(앞뒤 공백만 무시).
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
a = pd.read_csv(sys.argv[1], dtype=str, encoding='utf-8-sig', keep_default_na=False)
b = pd.read_csv(sys.argv[2], dtype=str, encoding='utf-8-sig', keep_default_na=False)
key = ['판', 'PDF쪽', '쪽안순번']
cols = ['원문_시도', '원문_시군구'] + [f'{x}_{y}' for x in ['검사건수', '적합', '부적합', '적합률'] for y in ['총계', '음용', '비음용']]
for d in (a, b):
    for c in key + cols:
        d[c] = d[c].astype(str).str.strip()
m = a.merge(b, on=key, how='outer', suffixes=('_1차', '_2차'), indicator=True)
print('1차 행', len(a), '2차 행', len(b), '짝 맞음', (m['_merge'] == 'both').sum(),
      '1차만', (m['_merge'] == 'left_only').sum(), '2차만', (m['_merge'] == 'right_only').sum())
rows = []
for _, r in m.iterrows():
    if r['_merge'] != 'both':
        rows.append({**{k: r[k] for k in key}, '칸': '(행 짝 없음)', '1차': r.get('원문_시군구_1차', ''), '2차': r.get('원문_시군구_2차', '')})
        continue
    for c in cols:
        v1, v2 = r[c + '_1차'], r[c + '_2차']
        if v1 != v2:
            rows.append({**{k: r[k] for k in key}, '원문_시도': r['원문_시도_1차'], '원문_시군구': r['원문_시군구_1차'], '칸': c, '1차': v1, '2차': v2})
out = pd.DataFrame(rows)
n_cells = (m['_merge'] == 'both').sum() * len(cols)
print('비교한 칸', n_cells, '다른 칸', len(out))
if len(out):
    print(out.to_string())
out.to_csv(sys.argv[3], index=False, encoding='utf-8-sig')
