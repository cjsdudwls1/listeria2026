# -*- coding: utf-8 -*-
"""종관기상 일강수량 0.0mm 날 vs 빈 날: 1시간 최다강수량·강수 계속시간 기록 비율 (연구기간, 기온 관측일, 측정만)"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
R = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected\raw"
S0, S1 = pd.Timestamp('2015-04-07'), pd.Timestamp('2026-06-30')
a = pd.read_parquet(R + r"\raw_kma_asos_daily_info.parquet", columns=['stn_id', 'meas_date', 'avg_ta', 'sum_rn', 'hr_1_max_rn', 'sum_rn_dur', 'mi_10_max_rn'])
a['d'] = pd.to_datetime(a['meas_date']); a = a[(a.d >= S0) & (a.d <= S1) & a['avg_ta'].notna()]
def has(s): return s.notna() & (s.astype(str).str.strip() != '')
OUT = {}
for lab, mask in [('빈 값', a['sum_rn'].isna()), ('0.0', a['sum_rn'] == 0), ('0.1~1 미만', (a['sum_rn'] > 0) & (a['sum_rn'] < 1)), ('1 이상', a['sum_rn'] >= 1)]:
    g = a[mask]
    OUT[lab] = {'days': int(len(g)), 'hr_1_max_rn_pct': round(float(has(g['hr_1_max_rn']).mean() * 100), 1), 'sum_rn_dur_pct': round(float(has(g['sum_rn_dur']).mean() * 100), 1),
                'mi_10_max_rn_pct': round(float(has(g['mi_10_max_rn']).mean() * 100), 1)}
    if lab == '0.0':
        OUT[lab]['hr_1_max_rn_values'] = g['hr_1_max_rn'].astype(str).value_counts().head(5).to_dict()
        OUT[lab]['sum_rn_dur_values'] = g['sum_rn_dur'].astype(str).value_counts().head(5).to_dict()
print(json.dumps(OUT, ensure_ascii=False, indent=1))
json.dump(OUT, open(r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp\v3_rain0.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
