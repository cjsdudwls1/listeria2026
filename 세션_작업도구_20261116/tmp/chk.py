import pyarrow.parquet as pq, json, collections
t = pq.read_table(r'리스테리아ML2\data\데이터셋\시군구_대표좌표.parquet').to_pandas()
out=[]
out.append(str(t.columns.tolist())); out.append(str(len(t)))
out.append(t[t.astype(str).apply(lambda r: '구례' in ' '.join(r), axis=1)].to_string())
out.append(str(sorted(set(t['sgg_code'].astype(str).str[:2]))))
g = json.load(open(r'리스테리아ML2\data\외부_수집\축사\ne_10m_admin_0_countries_v5.1.2.geojson', encoding='utf-8'))
c = collections.Counter(f['properties']['ISO_A3_EH'] for f in g['features'])
out.append(str(len(g['features'])))
out.append(str({k:v for k,v in c.items() if v>1}))
open('C:/Users/NT751/.claude/jobs/a19471fc/tmp/chk_out.txt','w',encoding='utf-8').write('\n'.join(out))
