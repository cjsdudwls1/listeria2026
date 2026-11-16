import pyarrow.parquet as pq, statistics, datetime
b = r'C:\Users\NT751\Desktop\천영진'
s = pq.read_table(b + r'\리스테리아ML\data\listeria_data.parquet', columns=['lims_data_key', 'lat', 'lon', 'mnfctur_de']).to_pylist()
g = pq.read_table(b + r'\리스테리아ML2\data\데이터셋\시료_시군구.parquet').to_pylist()
keys = {r['lims_data_key'] for r in g if r['sgg_code'] == '28720'}
rows = [r for r in s if r['lims_data_key'] in keys]
out = open(r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\chk\ongjin.txt', 'w', encoding='utf-8')
out.write('cols %s n %d\n' % (list(g[0].keys()), len(rows)))
def med(rs):
    la = [r['lat'] for r in rs if r['lat'] is not None]; lo = [r['lon'] for r in rs if r['lon'] is not None]
    return (statistics.median(la) if la else None, statistics.median(lo) if lo else None, len(la))
out.write('all %s\n' % (med(rows),))
for cut in ['2022-06-30', '2023-06-30', '2024-06-30', '2025-06-30']:
    c = datetime.date.fromisoformat(cut)
    rs = [r for r in rows if r['mnfctur_de'] is not None and (r['mnfctur_de'] if isinstance(r['mnfctur_de'], datetime.datetime) and False else datetime.date.fromisoformat(str(r['mnfctur_de'])[:10])) <= c]
    out.write('%s %s\n' % (cut, med(rs)))
out.close()
