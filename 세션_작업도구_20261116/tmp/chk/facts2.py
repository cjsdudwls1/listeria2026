import os, json, csv, collections
import openpyxl
base = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data'
out = open(r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\chk\facts2.txt', 'w', encoding='utf-8')
d = os.path.join(base, r'외부_수집\축사')
out.write(repr(os.listdir(d)) + '\n')
wb = openpyxl.load_workbook(os.path.join(d, '축사_시군구별_농장수.xlsx'), read_only=True, data_only=True)
ws = wb['시군구별_농장수']
rows = list(ws.iter_rows(values_only=True))
out.write(repr(rows[0]) + '\n')
for r in rows[1:6]:
    out.write(repr(r) + '\n')
for r in rows[1:]:
    if r and r[1] and ('화성' in str(r[1]) or '동해' in str(r[1]) or '세종' in str(r[1])):
        out.write('special ' + repr(r) + '\n')
out.write('n data %d\n' % len([r for r in rows[1:] if r and r[0]]))
# csv
for enc in ('utf-8-sig', 'cp949'):
    try:
        with open(os.path.join(d, '국가리스트_1_농장전체.csv'), encoding=enc) as f:
            rr = list(csv.DictReader(f))
        break
    except Exception as e:
        out.write('enc fail %s\n' % enc)
out.write(repr(list(rr[0].keys())) + '\n')
out.write('n countries %d, zero %d\n' % (len(rr), sum(1 for r in rr if float(r['farm_total'] or 0) == 0)))
g = json.load(open(os.path.join(d, 'ne_10m_admin_0_countries_v5.1.2.geojson'), encoding='utf-8'))
cnt = collections.Counter()
for ft in g['features']:
    p = ft['properties']
    c = p.get('ISO_A3_EH')
    if not c or c == '-99':
        c = p.get('ADM0_A3')
    cnt[c] += 1
out.write('dups ' + repr([k for k, v in cnt.items() if v > 1]) + '\n')
iso = set(r['iso3'] for r in rr)
out.write('csv iso3 not in geojson ' + repr(sorted(iso - set(cnt))) + '\n')
out.close()
