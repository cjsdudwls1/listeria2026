# 지하수 정기 수질검사(지하수조사연보 '시군구별 지하수 수질검사 현황') 연도표 만들기. 원자료는 읽기만.
# 사용: python -I build_groundwater.py <중간/외부지표 폴더> <시군구_대표좌표.parquet> <출력 parquet>
# 입력: <중간>/_지하수_글자/_기계추출_1차.csv (2014~2023년판, 글자 층 기계 추출, 합계 점검 통과)
#       <중간>/_지하수_대조/1차_옮김_2024.csv, 1차_옮김_2025.csv (그림 PDF, 2차 독립 옮김과 칸마다 같음)
# 규칙: 원값 그대로(천 단위 쉼표만 뺌). 적합률 = 적합 ÷ 검사(검사 0건이면 빈칸). 연보에 줄이 없으면 빈칸(채우지 않음).
#   시군구 대응: 그 시군구를 통째로 담는 가장 작은 옛 단위의 값을 넣음(일반시 → 구들, 나뉜 단위 → 나뉜 시군구들).
#   통째로 담는 옛 단위가 하나도 없으면(옛 단위 여러 개에 걸침) 옛 단위들의 건수를 더한 값을 넣고 결정대기 표시.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import ext_sgg_map as M

sys.stdout.reconfigure(encoding='utf-8')
mid, sgg_path, out_path = sys.argv[1:4]
sgg = M.load_sgg(sgg_path)
codes = sorted(sgg['sgg_code'])
nmof = dict(zip(sgg['sgg_code'], sgg['sgg_nm']))


def num(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).replace(',', '').strip()
    if s in ('', '-'):
        return np.nan
    return float(s)


# ---------- 1. 옮긴 원표 합치기 ----------
a = pd.read_csv(os.path.join(mid, '_지하수_글자', '_기계추출_1차.csv'), dtype=str, encoding='utf-8-sig')
U = []
for _, r in a.iterrows():
    ed = int(r['판'])
    d = {'판': ed, '자료연도': ed - 1, 'PDF쪽': int(r['PDF쪽']), '인쇄쪽': int(float(r['인쇄쪽'])) if pd.notna(r['인쇄쪽']) else None,
         '표안순번': int(r['순번']), '원문_시도': r['원문_시도'] if pd.notna(r['원문_시도']) else '',
         '원문_시군구': r['원문_시군구'], '옮긴_방법': 'PDF 글자 층 기계 추출(합계 점검 통과)', '메모': ''}
    if r['칸수확인'] != 'ok':
        d['메모'] = r['칸수확인']
    if ed >= 2023:
        d['원표_칸이름'] = '검사건수(총계·음용·비음용)/적합(…)/부적합(…)/적합율(%)(…)'
        for x, y in [('검사건수', '검사건수'), ('적합', '적합'), ('부적합', '부적합')]:
            for g in ['총계', '음용', '비음용']:
                d[f'{y}_{g}'] = r[f'{x}_{g}']
        for g in ['총계', '음용', '비음용']:
            d[f'적합률_{g}_인쇄'] = r[f'적합율(%)_{g}']
    else:
        d['원표_칸이름'] = '합계/적합/부적합/' + ('미기재' if ed <= 2016 else '적합률(%)')
        d['검사건수_총계'] = r['합계']; d['적합_총계'] = r['적합']; d['부적합_총계'] = r['부적합']
        if ed <= 2016:
            d['미기재'] = r['미기재']
        else:
            d['적합률_총계_인쇄'] = r['적합률(%)']
    U.append(d)
for ed in (2024, 2025):
    b = pd.read_csv(os.path.join(mid, '_지하수_대조', f'1차_옮김_{ed}.csv'), dtype=str, encoding='utf-8-sig', keep_default_na=False)
    for _, r in b.iterrows():
        d = {'판': ed, '자료연도': ed - 1, 'PDF쪽': int(r['PDF쪽']), '인쇄쪽': int(r['PDF쪽']) - 24, '표안순번': None,
             '원문_시도': r['원문_시도'], '원문_시군구': r['원문_시군구'],
             '옮긴_방법': '그림 PDF 를 보고 두 번 독립으로 옮겨 칸마다 대조(다른 칸 0)', '메모': r['메모'],
             '원표_칸이름': '검사건수(총계·음용·비음용)/적합(…)/부적합(…)/적합률(…)'}
        for x in ['검사건수', '적합', '부적합']:
            for g in ['총계', '음용', '비음용']:
                d[f'{x}_{g}'] = r[f'{x}_{g}']
        for g in ['총계', '음용', '비음용']:
            d[f'적합률_{g}_인쇄'] = r[f'적합률_{g}']
        U.append(d)
T = pd.DataFrame(U)
T['표안순번'] = T.groupby('판').cumcount() + 1
for c in [f'{x}_{g}' for x in ['검사건수', '적합', '부적합'] for g in ['총계', '음용', '비음용']] + ['미기재']:
    T[c] = T[c].map(lambda v: (str(v).replace(',', '').strip() if pd.notna(v) else v))
cols = ['판', '자료연도', 'PDF쪽', '인쇄쪽', '표안순번', '원문_시도', '원문_시군구', '원표_칸이름',
        '검사건수_총계', '적합_총계', '부적합_총계', '미기재', '적합률_총계_인쇄',
        '검사건수_음용', '적합_음용', '부적합_음용', '적합률_음용_인쇄',
        '검사건수_비음용', '적합_비음용', '부적합_비음용', '적합률_비음용_인쇄', '옮긴_방법', '메모']
T = T[cols]
T.to_csv(os.path.join(mid, '지하수_연보_옮긴표.csv'), index=False, encoding='utf-8-sig')

# ---------- 2. 대응 ----------
T['시도_점검용'] = T.groupby('판')['원문_시도'].transform(lambda s: s.replace('', np.nan).ffill())
body = T[~T['원문_시군구'].str.replace(' ', '').isin(['소계', '계', '총합계'])].copy()
body = body[body['원문_시도'] != '전국']
maprows = {}
body['대응'] = None
for i, r in body.iterrows():
    m = M.map_name(r['시도_점검용'], r['원문_시군구'], r['자료연도'], sgg)
    assert m['codes'], (r['판'], r['시도_점검용'], r['원문_시군구'])
    body.at[i, '대응'] = tuple(m['codes'])
    k = (r['시도_점검용'], r['원문_시군구'], tuple(m['codes']), m['유형'])
    maprows.setdefault(k, {'m': m, 'eds': []})['eds'].append(int(r['자료연도']))
MR = []
for (sd, nm, cs, _t), v in maprows.items():
    m = v['m']; ys = sorted(v['eds'])
    MR.append({'자료': '지하수조사연보 시군구별 수질검사 현황', '원문_시도': sd, '원문_이름': nm, '법정동코드': '',
               '나오는_해(자료연도)': f'{ys[0]}~{ys[-1]}' if ys == list(range(ys[0], ys[-1] + 1)) else ','.join(map(str, ys)),
               '대응_sgg_code': ';'.join(cs), '대응_sgg_nm': ';'.join(nmof[c] for c in cs),
               '대응_유형': m['유형'], '이름바뀜': m['이름바뀜'], '나뉨': m['나뉨'], '합쳐짐': m['합쳐짐'],
               '시단위값': m['시단위값'], '설명': m['설명']})
pd.DataFrame(MR).to_csv(os.path.join(mid, '_대응_지하수.csv'), index=False, encoding='utf-8-sig')

# 시군구가 여러 옛 단위에 걸치는 경우: 필요한 옛 단위 목록(모두 있어야 더함)
NEEDS = {
    '28125': lambda y: [('인천', '중구'), ('인천', '동구')],  # 제물포구 = 옛 중구 내륙 + 옛 동구
}
for c in ['43111', '43112', '43113', '43114']:
    NEEDS[c] = (lambda y: [('충북', '청주시'), ('충북', '청원군')] if y <= 2014 else None)

# ---------- 3. 사용가능일 ----------
# 사용자 확정(2026-10-19): 사용가능일 = max(판권면 'Y년 12월' → Y-12-31, PDF 파일 속 만든 날짜 creationDate)
# PDF 날짜는 gw_pdf_dates.py 가 만든 <중간>/_지하수_PDF날짜.csv 에서 읽음
pdfd = pd.read_csv(os.path.join(mid, '_지하수_PDF날짜.csv'), dtype=str, encoding='utf-8-sig').set_index('판')
PUB = {}
for ed in range(2014, 2026):
    colo = f'{ed}-12-31'
    made = pdfd.loc[str(ed), '만든날짜']
    detail = f"판권면 '발행일 {ed}년 12월' → {colo}(일이 없어 그 달 마지막 날) / PDF 파일 속 만든 날짜 {made} 중 늦은 날"
    if made > colo:
        PUB[ed] = (made, 'PDF 만든 날짜가 정함(판권면 발행월보다 늦음)', detail)
    else:
        PUB[ed] = (colo, '판권면 발행일이 정함(Y-12-31)', detail)

# ---------- 4. 시군구 × 자료연도 ----------
rows = []
cnt_cols = {'검사_건수': '검사건수_총계', '적합_건수': '적합_총계', '부적합_건수': '부적합_총계', '미기재_건수': '미기재',
            '검사_건수_음용': '검사건수_음용', '적합_건수_음용': '적합_음용', '부적합_건수_음용': '부적합_음용',
            '검사_건수_비음용': '검사건수_비음용', '적합_건수_비음용': '적합_비음용', '부적합_건수_비음용': '부적합_비음용'}
rate_cols = {'연보_적합률_인쇄값': '적합률_총계_인쇄', '연보_적합률_음용_인쇄값': '적합률_음용_인쇄', '연보_적합률_비음용_인쇄값': '적합률_비음용_인쇄'}
for ed in range(2014, 2026):
    yr = ed - 1
    bd = body[body['판'] == ed]
    for c in codes:
        rec = {'sgg_code': c, '자료연도': yr, '사용가능일': PUB[ed][0], '사용가능일_근거': PUB[ed][1], '사용가능일_근거_상세': PUB[ed][2],
               '연보_판': ed}
        need = NEEDS.get(c, lambda y: None)(yr)
        cand = bd[[c in cs for cs in bd['대응']]]
        flag_split = flag_merge = flag_city = flag_rename = False
        pending = None
        if need:
            # 여러 옛 단위에 걸침 → 다 있으면 건수 합
            got = []
            for sd, nm in need:
                x = bd[(bd['시도_점검용'] == sd) & (bd['원문_시군구'] == nm)]
                if len(x) == 1:
                    got.append(x.iloc[0])
            names = '+'.join(f'{sd} {nm}' for sd, nm in need)
            if len(got) == len(need):
                src = pd.DataFrame(got)
                unit_kind = '옛 단위 여러 개의 합'
                pending = (f'사용자 확정(10-19): {nmof[c]} 은 옛 단위 {names} 에 걸쳐 있어(통째로 담는 옛 단위 없음) '
                           '두 단위 건수를 더해 적합률 계산')
            else:
                src = pd.DataFrame()
                unit_kind = None
                pending = f'사용자 확정(10-19): {nmof[c]} 은 옛 단위 {names} 에 걸쳐 있는데 이 판에 그중 일부 줄이 없어 빈칸(연보에 줄이 없으면 빈칸)'
            flag_merge = True
        else:
            if len(cand) == 0:
                src = pd.DataFrame(); unit_kind = None
            else:
                # 통째로 담는 단위 중 가장 작은 것(대응 시군구 수가 가장 적은 줄)
                cand = cand.assign(_n=[len(cs) for cs in cand['대응']]).sort_values('_n')
                best = cand.iloc[[0]]
                if (cand['_n'] == best['_n'].iloc[0]).sum() > 1:
                    raise RuntimeError(f'같은 크기 후보 여러 개: {ed} {c} {cand[["시도_점검용","원문_시군구"]].values.tolist()}')
                src = best
                n = best['_n'].iloc[0]
                unit_kind = '시군구' if n == 1 else '옛 단위(여러 시군구 공통)'
        if len(src):
            for k2, col in cnt_cols.items():
                vals = src[col].map(num)
                rec[k2] = float(vals.sum()) if vals.notna().all() else np.nan
            if len(src) == 1:
                for k2, col in rate_cols.items():
                    rec[k2] = num(src.iloc[0][col])
            else:
                for k2 in rate_cols:
                    rec[k2] = np.nan
            rec['원자료_시군구_이름'] = '+'.join(f"{r['시도_점검용']} {r['원문_시군구']}" for _, r in src.iterrows())
            rec['연보_PDF쪽'] = ';'.join(str(int(p)) for p in src['PDF쪽'])
            rec['연보_인쇄쪽'] = ';'.join(str(int(p)) if pd.notna(p) else '' for p in src['인쇄쪽'])
            rec['값_단위'] = unit_kind
            # 표시: 줄 하나가 여러 시군구에 같은 값으로 들어가면 나뉨(행정 구역이 나뉨) 또는 시단위값(일반시 자료)
            ms = [M.map_name(r['시도_점검용'], r['원문_시군구'], yr, sgg) for _, r in src.iterrows()]
            flag_rename = any(m['이름바뀜'] for m in ms)
            one_to_many = len(src) == 1 and len(src.iloc[0]['대응']) > 1
            flag_split = one_to_many and ms[0]['나뉨']
            flag_city = one_to_many and ms[0]['시단위값']
            rec['연보_줄없음'] = False
        else:
            for k2 in list(cnt_cols) + list(rate_cols):
                rec[k2] = np.nan
            rec['원자료_시군구_이름'] = None; rec['연보_PDF쪽'] = None; rec['연보_인쇄쪽'] = None
            rec['값_단위'] = None
            rec['연보_줄없음'] = True
        rec['이름바뀜'] = bool(flag_rename)
        rec['나뉨_옛단위값'] = bool(flag_split)
        rec['합쳐짐_옛단위합'] = bool(flag_merge)
        rec['시단위값'] = bool(flag_city)
        rec['결정대기_사유'] = pending
        t = rec['검사_건수']
        rec['적합률'] = rec['적합_건수'] / t if pd.notna(t) and t > 0 else np.nan
        rows.append(rec)
O = pd.DataFrame(rows)
order = ['sgg_code', '자료연도', '사용가능일', '사용가능일_근거', '사용가능일_근거_상세',
         '검사_건수', '적합_건수', '부적합_건수', '적합률', '미기재_건수', '연보_적합률_인쇄값',
         '검사_건수_음용', '적합_건수_음용', '부적합_건수_음용', '연보_적합률_음용_인쇄값',
         '검사_건수_비음용', '적합_건수_비음용', '부적합_건수_비음용', '연보_적합률_비음용_인쇄값',
         '원자료_시군구_이름', '연보_판', '연보_PDF쪽', '연보_인쇄쪽', '값_단위',
         '이름바뀜', '나뉨_옛단위값', '합쳐짐_옛단위합', '시단위값', '연보_줄없음', '결정대기_사유']
O = O[order]
O['사용가능일'] = pd.to_datetime(O['사용가능일']).dt.date
import pyarrow as pa, pyarrow.parquet as pq
tbl = pa.Table.from_pandas(O, preserve_index=False)
for name, typ in {'사용가능일': pa.date32(), '자료연도': pa.int64(), '연보_판': pa.int64()}.items():
    i = tbl.schema.get_field_index(name)
    tbl = tbl.set_column(i, pa.field(name, typ), tbl.column(name).cast(typ))
pq.write_table(tbl, out_path)
print('옮긴표 행', len(T), '/ 결과 행', len(O), '자료연도', O['자료연도'].min(), O['자료연도'].max())
print('값 있는 행', int(O['검사_건수'].notna().sum()), '빈칸 행', int(O['검사_건수'].isna().sum()),
      '적합률 빈칸', int(O['적합률'].isna().sum()), '결정대기', int(O['결정대기_사유'].notna().sum()))
