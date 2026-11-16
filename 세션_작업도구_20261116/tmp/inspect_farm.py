import sys
import pyarrow.parquet as pq
import pyarrow.compute as pc
import openpyxl

xlsx, dom, ovs = sys.argv[1], sys.argv[2], sys.argv[3]
out = []

wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
for ws in wb.worksheets:
    rows = ws.iter_rows(values_only=True)
    header = next(rows, None)
    n = 0
    cols = {}
    for r in rows:
        n += 1
        for i, v in enumerate(r):
            if v is not None:
                cols.setdefault(i, set())
                if len(cols[i]) < 40:
                    cols[i].add(str(v)[:20])
    out.append(f'[xlsx] sheet={ws.title} rows={n} header={header}')
    for i, h in enumerate(header or []):
        vals = cols.get(i, set())
        hs = str(h)
        if any(k in hs for k in ('축종', '시도', '시군구', '구분', '종류', '국가', '나라', 'country', 'species')):
            out.append(f'   col {i} {hs}: distinct(<=40) {sorted(vals)[:40]}')

for name, p in (('domestic', dom), ('overseas', ovs)):
    f = pq.ParquetFile(p)
    t = f.read()
    out.append(f'[{name}] rows={t.num_rows} cols={t.num_columns}')
    out.append('   ' + ', '.join(f'{c}:{t.schema.field(c).type}' for c in t.column_names))
    if 'farm_unmapped' in t.column_names:
        vc = pc.value_counts(t['farm_unmapped']).to_pylist()
        out.append(f'   farm_unmapped {vc}')
    cnt_cols = [c for c in t.column_names if any(k in c for k in ('cattle', 'pig', 'poultry'))]
    for c in cnt_cols[:15]:
        col = t[c]
        nz = pc.sum(pc.greater(pc.fill_null(col, 0), 0)).as_py()
        out.append(f'   {c}: nonnull={len(col) - col.null_count} >0={nz}')

open(sys.argv[4], 'w', encoding='utf-8').write('\n'.join(out))
