# -*- coding: utf-8 -*-
"""지정 표의 전체 컬럼(이름·타입·설명) 출력 — 로컬 카탈로그 CSV 사용"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
cols = pd.read_csv(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\brief_columns.csv")
for full in sys.argv[1:]:
    sc, tb = full.split('.')
    g = cols[(cols.schema == sc) & (cols.tbl == tb)].sort_values('attnum')
    print(f"\n{full} ({len(g)}열)")
    print('  ' + ' | '.join(f"{x.col}{'=' + ' '.join(str(x.ccomment).split())[:40] if isinstance(x.ccomment, str) else ''}" for x in g.itertuples()))
