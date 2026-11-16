# 옮긴 지하수 표의 합계 점검(읽기만): 시도 소계 = 시군구 합, 전국 = 소계 합, 적합+부적합(+미기재) = 합계.
# 사용: python -I gw_check_sums.py <옮긴표csv> <출력txt>
import sys
import pandas as pd

p, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(p, dtype=str, encoding='utf-8-sig')
# 시도 칸이 빈 행은 앞 행 시도로 묶어 점검(원문은 바꾸지 않음, 점검용)
d['원문_시도'] = d.groupby('판')['원문_시도'].ffill()
L = []
P = L.append


def num(s):
    if pd.isna(s):
        return None
    s = str(s).replace(',', '').strip()
    if s in ('-', ''):
        return None
    return float(s)


for ed, g in d.groupby('판'):
    ed = int(ed)
    if ed >= 2023:
        cnt = ['검사건수_총계', '검사건수_음용', '검사건수_비음용', '적합_총계', '적합_음용', '적합_비음용',
               '부적합_총계', '부적합_음용', '부적합_비음용']
    else:
        cnt = ['합계', '적합', '부적합'] + (['미기재'] if ed <= 2016 else [])
    g = g.copy()
    for c in cnt:
        g[c + '_n'] = g[c].map(num)
    is_sub = g['원문_시군구'].str.replace(' ', '').isin(['소계'])
    is_tot = g['원문_시도'] == '전국'
    body = g[~is_sub & ~is_tot]
    P(f'== 판 {ed}: 행 {len(g)} (시군구 {len(body)}, 소계 {is_sub.sum()}, 전국 {is_tot.sum()})')
    # 소계 점검
    bad = []
    for sd, gg in body.groupby('원문_시도'):
        sub = g[is_sub & (g['원문_시도'] == sd)]
        if len(sub) != 1:
            bad.append(f'{sd}: 소계 행 {len(sub)}개')
            continue
        for c in cnt:
            s1 = gg[c + '_n'].sum(); s2 = sub[c + '_n'].iloc[0]
            if abs(s1 - s2) > 1e-9:
                bad.append(f'{sd} {c}: 시군구합 {s1:g} vs 소계 {s2:g}')
    # 소계가 있는데 시군구 행이 없는 시도
    for sd in set(g.loc[is_sub, '원문_시도']) - set(body['원문_시도']):
        bad.append(f'{sd}: 소계만 있고 시군구 행 없음')
    tot = g[is_tot]
    if len(tot) == 1:
        for c in cnt:
            s1 = g.loc[is_sub, c + '_n'].sum(); s2 = tot[c + '_n'].iloc[0]
            s3 = body[c + '_n'].sum()
            if abs(s1 - s2) > 1e-9 or abs(s3 - s2) > 1e-9:
                bad.append(f'전국 {c}: 소계합 {s1:g}, 시군구합 {s3:g} vs 전국 {s2:g}')
    else:
        bad.append(f'전국 행 {len(tot)}개')
    # 행 안 점검
    if ed >= 2023:
        for grp in ['총계', '음용', '비음용']:
            x = g['검사건수_' + grp + '_n'] - g['적합_' + grp + '_n'] - g['부적합_' + grp + '_n']
            for _, r in g[x.abs() > 1e-9].iterrows():
                bad.append(f'행안 {grp}: {r["원문_시도"]} {r["원문_시군구"]} 검사 {r["검사건수_"+grp]} != 적합 {r["적합_"+grp]} + 부적합 {r["부적합_"+grp]}')
        for c in ['검사건수', '적합', '부적합']:
            x = g[c + '_총계_n'] - g[c + '_음용_n'] - g[c + '_비음용_n']
            for _, r in g[x.abs() > 1e-9].iterrows():
                bad.append(f'행안 {c}: {r["원문_시도"]} {r["원문_시군구"]} 총계 != 음용+비음용')
    else:
        x = g['합계_n'] - g['적합_n'] - g['부적합_n'] - (g['미기재_n'] if ed <= 2016 else 0)
        for _, r in g[x.abs() > 1e-9].iterrows():
            bad.append(f'행안: {r["원문_시도"]} {r["원문_시군구"]} 합계 {r["합계"]} != 적합 {r["적합"]} + 부적합 {r["부적합"]}' + (f' + 미기재 {r["미기재"]}' if ed <= 2016 else ''))
    P('  문제 없음' if not bad else '  ' + '\n  '.join(bad))
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
