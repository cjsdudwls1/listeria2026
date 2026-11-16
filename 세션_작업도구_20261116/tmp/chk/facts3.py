import pyarrow.parquet as pq
t = pq.read_table(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋\시군구_대표좌표.parquet').to_pylist()
with open(r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\chk\facts3.txt', 'w', encoding='utf-8') as o:
    for r in t:
        if '화성' in r['sgg_nm'] or '세종' in r['sgg_nm'] or '옹진' in r['sgg_nm']:
            o.write(repr(r) + '\n')
    s = pq.read_table(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋\시료_시군구.parquet', columns=['sgg_code']).column('sgg_code').to_pylist()
    o.write('sample rows %d uniq %d, coord codes %d, same %s\n' % (len(s), len(set(s)), len(t), set(s) == set(r['sgg_code'] for r in t)))
