# 그림 PDF 에서 손으로 옮긴 지하수 표(1차·2차) 점검(읽기만).
# 소계 = 시군구 합, 전국 = 소계 합, 검사 = 적합 + 부적합, 총계 = 음용 + 비음용, 적합률 = 적합/검사×100 (소수 첫째 자리 반올림 비교)
# 사용: python -I gw_check_image.py <옮긴csv>
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

d = pd.read_csv(sys.argv[1], dtype=str, encoding='utf-8-sig')
d['원문_시도'] = d['원문_시도'].fillna('')
cnt = [f'{a}_{b}' for a in ['검사건수', '적합', '부적합'] for b in ['총계', '음용', '비음용']]
for c in cnt + [f'적합률_{b}' for b in ['총계', '음용', '비음용']]:
    d[c + '_n'] = pd.to_numeric(d[c].str.replace(',', ''), errors='coerce')
bad = []
nonnum = d[[c + '_n' for c in cnt]].isna().any(axis=1)
for _, r in d[nonnum].iterrows():
    bad.append(f'숫자 아닌 칸: {r["PDF쪽"]}쪽 {r["원문_시도"]} {r["원문_시군구"]}')
is_sub = (d['원문_시군구'].str.replace(' ', '') == '소계') | ((d['원문_시군구'].str.strip() == '계') & (d['원문_시도'] != '전국'))  # 2025년판은 시도 합계 행 이름이 '계'
is_tot = d['원문_시도'] == '전국'
body = d[~is_sub & ~is_tot]
print(f'행 {len(d)}: 시군구 {len(body)}, 소계 {is_sub.sum()}, 전국 {is_tot.sum()}')
for sd, g in body.groupby('원문_시도'):
    sub = d[is_sub & (d['원문_시도'] == sd)]
    if len(sub) != 1:
        bad.append(f'{sd} 소계 행 {len(sub)}개'); continue
    for c in cnt:
        if abs(g[c + '_n'].sum() - sub[c + '_n'].iloc[0]) > 1e-9:
            bad.append(f'{sd} {c}: 시군구합 {g[c + "_n"].sum():g} vs 소계 {sub[c + "_n"].iloc[0]:g}')
tot = d[is_tot]
for c in cnt:
    s1 = d.loc[is_sub, c + '_n'].sum(); s2 = tot[c + '_n'].iloc[0]
    if abs(s1 - s2) > 1e-9:
        bad.append(f'전국 {c}: 소계합 {s1:g} vs 전국 {s2:g}')
for b in ['총계', '음용', '비음용']:
    x = d[f'검사건수_{b}_n'] - d[f'적합_{b}_n'] - d[f'부적합_{b}_n']
    for _, r in d[x.abs() > 1e-9].iterrows():
        bad.append(f'행안 검사!=적합+부적합 ({b}): {r["PDF쪽"]}쪽 {r["원문_시도"]} {r["원문_시군구"]}')
    # 적합률 비교 (검사 0 이면 인쇄값 0 인지)
    rate = np.floor(d[f'적합_{b}_n'] / d[f'검사건수_{b}_n'] * 1000 + 0.5) / 10  # 반올림(0.05 올림)
    pr = d[f'적합률_{b}_n']
    z = d[f'검사건수_{b}_n'] == 0
    m = (~z & ((rate - pr).abs() > 0.051)) | (z & (pr != 0))
    for _, r in d[m].iterrows():
        bad.append(f'적합률 다름 ({b}): {r["PDF쪽"]}쪽 {r["원문_시도"]} {r["원문_시군구"]} 인쇄 {r[f"적합률_{b}"]} 계산 {(r[f"적합_{b}_n"]/r[f"검사건수_{b}_n"]*100) if r[f"검사건수_{b}_n"] else 0:.2f}')
for a in ['검사건수', '적합', '부적합']:
    x = d[f'{a}_총계_n'] - d[f'{a}_음용_n'] - d[f'{a}_비음용_n']
    for _, r in d[x.abs() > 1e-9].iterrows():
        bad.append(f'행안 총계!=음용+비음용 ({a}): {r["PDF쪽"]}쪽 {r["원문_시도"]} {r["원문_시군구"]}')
print('문제 없음' if not bad else '\n'.join(bad))
