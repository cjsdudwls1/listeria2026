import json, sys, zipfile, os
sys.path.insert(0, r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook')
from db_conn import connect

SRC = sys.argv[1]
OUT = sys.argv[2]
REP = sys.argv[3]
g = json.load(open(SRC, encoding='utf-8'))
feats = []
for f in g['features']:
    p = f['properties']
    iso = p.get('ISO_A3_EH') or '-99'
    if iso in ('-99', None, ''):
        iso = p.get('ADM0_A3')
    feats.append({'type': 'Feature',
                  'properties': {'iso3': iso, 'adm0_a3': p.get('ADM0_A3'), 'name_en': p.get('NAME_EN') or p.get('NAME'),
                                 'name_ko': p.get('NAME_KO'), 'sovereignt': p.get('SOVEREIGNT')},
                  'geometry': f['geometry']})
out = {'type': 'FeatureCollection', 'name': 'country_borders_naturalearth_v5_1_2_10m',
       'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
       'features': feats}
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
zp = OUT[:-8] + '.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(OUT, os.path.basename(OUT))

with connect() as conn, conn.cursor() as cur:
    cur.execute("SELECT DISTINCT alpha_3, name_ko FROM api_data_info.global_weather_info ORDER BY 1")
    db = cur.fetchall()
ne = {f['properties']['iso3'] for f in feats}
dup = sorted({c for c in [f['properties']['iso3'] for f in feats] if [f['properties']['iso3'] for f in feats].count(c) > 1})
miss = [(a, n) for a, n in db if a not in ne]
lines = [f'경계 {len(feats)}개(나라·지역), 파일 {os.path.getsize(OUT)/1e6:.1f}MB, zip {os.path.getsize(zp)/1e6:.1f}MB',
         f'DB global_weather_info 나라 {len(db)}개 중 경계 파일에 없는 코드 {len(miss)}개: {miss}',
         f'경계 파일에서 코드가 겹치는 것: {dup}',
         'iso3 가 ADM0_A3 로 대체된 것: ' + str([(f['properties']['iso3'], f['properties']['name_en']) for f in feats if f['properties']['iso3'] == f['properties']['adm0_a3'] and False])]
open(REP, 'w', encoding='utf-8').write('\n'.join(lines))
