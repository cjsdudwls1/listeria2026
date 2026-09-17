# -*- coding: utf-8 -*-
"""브리핑용 DB 카탈로그 탐색 (시스템 카탈로그 SELECT만, 데이터 표는 읽지 않음)
A. 해외 기상 관련 표 / B. 제조~섭취(유통·보관 기간·온도, 제조공정, 섭취형태, 유통기한, 포장) 관련 컬럼
+ 리서치팀 데이터현황 엑셀 키워드 검색"""
import sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import pandas as pd, openpyxl
from db_utils import connect
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
conn = connect()
def read_sql(sql, conn, params=None, label=''):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
tabs = read_sql("""
SELECT n.nspname AS schema, c.relname AS tbl, c.relkind, c.reltuples::bigint AS approx_rows,
       pg_total_relation_size(c.oid) AS bytes, obj_description(c.oid, 'pg_class') AS tcomment
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r','v','m','p','f') AND n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%'
""", conn, label='tables')
cols = read_sql("""
SELECT n.nspname AS schema, c.relname AS tbl, a.attnum, a.attname AS col, format_type(a.atttypid, a.atttypmod) AS type,
       col_description(c.oid, a.attnum) AS ccomment
FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE a.attnum > 0 AND NOT a.attisdropped AND c.relkind IN ('r','v','m','p','f')
  AND n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%'
""", conn, label='columns')
conn.close()
tabs.to_csv(T + r"\brief_tables.csv", index=False, encoding='utf-8-sig'); cols.to_csv(T + r"\brief_columns.csv", index=False, encoding='utf-8-sig')
print('스키마별 표 수:', tabs.groupby('schema').size().to_dict(), '| 컬럼 수', len(cols))
for sc, g in tabs.sort_values(['schema', 'tbl']).groupby('schema'):
    print(f"\n[{sc}] " + ' / '.join(f"{r.tbl}({r.relkind},{int(r.approx_rows):,})" for r in g.itertuples()))

# A. 해외 기상
KA = re.compile(r'global|overseas|foreign|era5|meteo|nation|country|해외|국외|제조국|국가|수출국|weather|climate|wx|기상|날씨', re.I)
ta = tabs[tabs.apply(lambda r: bool(KA.search(str(r.tbl)) or KA.search(str(r.tcomment or ''))), axis=1)]
print('\n\n===== A. 해외·기상 관련 표 =====')
for r in ta.sort_values(['schema', 'tbl']).itertuples():
    print(f"- {r.schema}.{r.tbl} ({r.relkind}) 약 {int(r.approx_rows):,}행 {r.bytes/1e6:,.0f}MB | 표 설명: {r.tcomment}")
    cc = cols[(cols.schema == r.schema) & (cols.tbl == r.tbl)].sort_values('attnum')
    print('   ', ' | '.join(f"{x.col}[{x.type}]{'=' + str(x.ccomment) if x.ccomment else ''}" for x in cc.itertuples()))

# B. 제조~섭취 관련 컬럼 (기상 표 제외)
KB = re.compile(r'유통|보관|보존|저장|냉장|냉동|실온|상온|해동|온도|운반|운송|배송|수송|입고|출고|선적|입항|반입|통관|도착|가열|살균|멸균|조리|공정|제조방법|제조 ?조건|가공방법|훈제|발효|건조|레토르트|haccp|섭취|용법|용도|즉석|유통기한|소비기한|품질유지|유효|기한|포장|진공|가스|질소|용기|재질|밀봉|성상|형태|원재료|원료|'
                r'prsrv|distb|cstdy|frmlc|packng|pack|mnf_cnd|mthd|heat|steril|frozen|chill|shelf|expir|pog_|qlity|shipng|entrpt|wrhous|trnsp|cnvync|splore|tmpr|temp|_tp$|storage|package|cook|intake', re.I)
WX = set(ta['schema'] + '.' + ta['tbl'])
cb = cols[cols.apply(lambda r: (r.schema + '.' + r.tbl) not in WX and bool(KB.search(str(r.col)) or KB.search(str(r.ccomment or ''))), axis=1)]
cb = cb.merge(tabs[['schema', 'tbl', 'relkind', 'approx_rows', 'tcomment']], on=['schema', 'tbl'], how='left')
cb.to_csv(T + r"\brief_lifecycle_cols.csv", index=False, encoding='utf-8-sig')
print('\n\n===== B. 제조~섭취 관련 컬럼(기상 표 제외) =====', len(cb))
for (sc, tb), g in cb.sort_values(['schema', 'tbl', 'attnum']).groupby(['schema', 'tbl']):
    r0 = g.iloc[0]
    print(f"- {sc}.{tb} ({r0.relkind}) 약 {int(r0.approx_rows):,}행 | 표 설명: {r0.tcomment}")
    print('   ', ' | '.join(f"{x.col}[{x.type}]{'=' + str(x.ccomment) if x.ccomment else ''}" for x in g.itertuples()))

# 리서치팀 데이터현황 엑셀
P = r"C:\Users\NT751\Desktop\천영진\리서치팀문서\[2026] 위해요소별 데이터 현황_0707.xlsx"
wb = openpyxl.load_workbook(P, data_only=True, read_only=True)
KX = re.compile(r'유통|보관|보존|저장|냉장|냉동|실온|온도|운반|가열|살균|멸균|조리|공정|섭취|즉석|유통기한|소비기한|포장|진공|해외|수입|제조국|기상')
print('\n\n===== 데이터현황_0707 엑셀 =====', [ws.title for ws in wb.worksheets])
for ws in wb.worksheets:
    hits = []
    for row in ws.iter_rows():
        for c in row:
            if isinstance(c.value, str) and KX.search(c.value):
                hits.append(f"{c.coordinate}:{' '.join(c.value.split())[:90]}")
    print(f"[{ws.title}] {ws.max_row}행 · 키워드 칸 {len(hits)}개")
    for h in hits[:40]: print('   ', h)
