# -*- coding: utf-8 -*-
"""B(제조~섭취) 후보 컬럼을 표별로 짧게 보기 — ba_data·ba_data_info·ready_data·staging_data(시료 연결 가능성 있는 원본성 표)"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
cb = pd.read_csv(T + r"\brief_lifecycle_cols.csv")
cb = cb[cb.schema.isin(['ba_data', 'ba_data_info', 'ready_data', 'staging_data', 'predict_data'])]
cb = cb[~cb.tbl.str.contains('aflatoxin|_bak|_judge')]
# 약한 패턴(용기·형태·원료 등) 제외하고 핵심 주제 표시
TOP = {
    '기간·날짜': r'유통|기한|선적|입항|반입|통관|도착|입고|출고|shipng|entrpt|wrhous|distb|pog_|qlity|valid|expir',
    '온도·보관': r'보관|보존|저장|냉장|냉동|실온|상온|해동|온도|운반|cnvync|splore|prsrv|cstdy|tmpr|_tp$',
    '제조공정': r'가열|살균|멸균|조리|공정|제조방법|제조 ?조건|가공방법|훈제|발효|건조|레토르트|haccp|mnf_cnd|mthd|steril|heat',
    '섭취형태': r'섭취|용법|용도|즉석|그대로|intake',
    '포장': r'포장|진공|가스|질소|밀봉|frmlc|packng|pack',
}
def topic(r):
    s = f"{r.col} {r.ccomment if isinstance(r.ccomment, str) else ''}"
    return ','.join(k for k, p in TOP.items() if re.search(p, s, re.I))
cb['topic'] = cb.apply(topic, axis=1)
cb = cb[cb.topic != '']
for (sc, tb), g in cb.sort_values(['schema', 'tbl', 'attnum']).groupby(['schema', 'tbl'], sort=False):
    rows = int(g.approx_rows.iloc[0])
    print(f"\n{sc}.{tb} (약 {rows:,}행) {g.tcomment.iloc[0] if isinstance(g.tcomment.iloc[0], str) else ''}")
    for x in g.itertuples():
        cm = ' '.join(str(x.ccomment).split())[:70] if isinstance(x.ccomment, str) else ''
        print(f"   [{x.topic}] {x.col} ({x.type}) {cm}")
