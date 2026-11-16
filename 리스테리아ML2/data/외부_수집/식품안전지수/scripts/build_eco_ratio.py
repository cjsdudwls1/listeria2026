# 친환경(유기·무농약) 인증 면적 비율 연도표 만들기 (원자료는 읽기만).
# 사용: python -I build_eco_ratio.py <식품안전지수 폴더> <시군구_대표좌표.parquet> <출력 parquet> <중간 폴더>
# 규칙(사용자 승인 2026-10-19): 비율 = 인증 면적 ÷ 경지면적, 분자·분모가 다 있는 가장 작은 단위에서 계산해 그 단위의 우리 시군구에 같은 값,
#   그 해 표가 있는데 줄이 없는 시군구는 0, 결측 채움·단위 변환은 하지 않음.
# 사용자 확정(2026-10-19): 사용가능일 = max(친환경 자료, 경지면적), 시 행·구 행 결정대기 칸은 '값 영향 없음'.
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import ext_sgg_map as M

sys.stdout.reconfigure(encoding='utf-8')
base, sgg_path, out_path, mid = sys.argv[1:5]
sgg = M.load_sgg(sgg_path)
codes = sorted(sgg['sgg_code'])

# ---------- 친환경 원자료: 20251231 파일(2016~2025 누적, 앞 파일들과 같은 해 값 차이 0) ----------
eco_file = os.path.join(base, '친환경_인증', '원자료', '친환경_시도별통계_농산물_20251231.csv')
e = pd.read_csv(eco_file, encoding='cp949', dtype=str)
e.columns = ['연도', '시도', '시군구', '구분', '계', '유기', '무농약']
e['연도'] = e['연도'].astype(int)
area = e[e['구분'].str.startswith('면적')].copy()
for c in ['계', '유기', '무농약']:
    area[c] = pd.to_numeric(area[c])
years = sorted(area['연도'].unique())

# 이름 대응
maprows = []
rowmap = {}
for (sd, nm), g in area.groupby(['시도', '시군구']):
    ys = sorted(g['연도'].unique())
    # 청주시는 2016년 뒤라 year 조건 무관. 연도별로 달라지는 규칙은 없음(친환경 2016~2025)
    r = M.map_name(sd, nm, ys[0], sgg)
    rowmap[(sd, nm)] = r['codes']
    maprows.append({'자료': '친환경 시도별통계(농산물)', '원문_시도': sd, '원문_이름': nm, '법정동코드': '',
                    '나오는_해(자료연도)': f'{ys[0]}~{ys[-1]}' if ys == list(range(ys[0], ys[-1] + 1)) else ','.join(map(str, ys)),
                    '대응_sgg_code': ';'.join(r['codes']),
                    '대응_sgg_nm': ';'.join(sgg.set_index('sgg_code').loc[r['codes'], 'sgg_nm']) if r['codes'] else '',
                    '대응_유형': r['유형'], '이름바뀜': r['이름바뀜'], '나뉨': r['나뉨'], '합쳐짐': r['합쳐짐'],
                    '시단위값': r['시단위값'], '설명': r['설명']})
unmapped = [k for k, v in rowmap.items() if not v]
assert not unmapped, unmapped
area['대응'] = [tuple(rowmap[(a, b)]) for a, b in zip(area['시도'], area['시군구'])]

# 같은 해 시 행 + 구 행 공존 확인(결정 대기 표시용)
full = e.copy()
for c in ['계', '유기', '무농약']:
    full[c] = pd.to_numeric(full[c])
pending = {}  # (sgg, year) -> 사유
pending_detail = []
cities = sorted({s.split(' ')[0] for s in area['시군구'] if ' ' in s})
for city in cities:
    for yr in years:
        sub = full[(full['연도'] == yr) & full['시군구'].str.startswith(city)]
        crow = sub[sub['시군구'] == city]; grow = sub[sub['시군구'] != city]
        if len(crow) and len(grow):
            ca = crow[crow['구분'].str.startswith('면적')]['계'].sum()
            ga = grow[grow['구분'].str.startswith('면적')]['계'].sum()
            call0 = bool((crow[['계', '유기', '무농약']] == 0).all().all())
            gall0 = bool((grow[grow['구분'].str.startswith('면적')][['계', '유기', '무농약']] == 0).all().all())
            # 구분 4개(건수·농가수·면적·출하량) 각각 시 행 vs 구 행 합
            cmp = {}
            for gb in sub['구분'].unique():
                cv = crow[crow['구분'] == gb]['계'].sum(); gv = grow[grow['구분'] == gb]['계'].sum()
                cmp[gb] = (cv, gv)
            is_sum = all(abs(cv - gv) < 1e-9 for cv, gv in cmp.values())
            def fmt(v):
                return f'{v:g}'
            parts = []
            for gb, (cv, gv) in cmp.items():
                parts.append(f"{gb.split('(')[0]} {fmt(cv)}/{fmt(gv)}")
            if is_sum:
                why = None  # 합이면 구 행을 씀(결정 대기 아님)
            else:
                head = f'{yr}년 원자료에 {city} 행과 구 행이 함께 있고 시 행이 구 행 합이 아님 (시 행/구 행 합: ' + ', '.join(parts) + ')'
                if call0:
                    tail = ' — 시 행은 네 구분 모두 0'
                elif gall0 and ca == 0:
                    tail = ' — 면적은 시 행·구 행 모두 0'
                elif gall0:
                    tail = ' — 구 행 면적은 모두 0, 값은 시 행에만 있음'
                else:
                    tail = ''
                why = head + tail + '. 면적은 한쪽이 0 이라 시 행을 더하든 빼든 시 단위 비율 같음'
            pending_detail.append({'시': city, '연도': yr, '시행_면적계': ca, '구행합_면적계': ga, '시행_모두0': call0,
                                   '구행_면적모두0': gall0, '구분별(시행,구행합)': json.dumps({k: [float(a), float(b)] for k, (a, b) in cmp.items()}, ensure_ascii=False),
                                   '구행합과_같음': is_sum})
            if why:
                tg = set()
                for cs in area[(area['연도'] == yr) & area['시군구'].str.startswith(city)]['대응']:
                    tg |= set(cs)
                for c in tg:
                    pending[(c, yr)] = why
pd.DataFrame(pending_detail).to_csv(os.path.join(mid, '_친환경_시행구행_점검.csv'), index=False, encoding='utf-8-sig')

# ---------- 경지면적 ----------
k = pd.read_csv(os.path.join(base, '경지면적', '원자료', 'KOSIS_시군구_논밭면적_2008_2025_20251231.csv'),
                encoding='cp949', dtype={'법정동코드': str})
k = k.set_index('법정동코드')

# 끝자리 719 줄 점검: 광역시 합계 = 군 줄 + 719 줄 ?
chk719 = []
for tot, parts in [('2600000000', ['2671000000', '2671900000']), ('2700000000', ['2771000000', '2771900000']),
                   ('2700000000', ['2771000000', '2771900000', '2772000000']),
                   ('2800000000', ['2871000000', '2871900000']), ('2900000000', ['2920000000', '2920900000']),
                   ('3100000000', ['3171000000', '3171900000'])]:
    for yr in range(2008, 2026):
        for t in ['논', '밭']:
            c = f'{t}_{yr}_면적'
            s = k.loc[parts, c].sum(); v = k.loc[tot, c]
            chk719.append({'합계코드': tot, '더한코드': '+'.join(parts), '연도': yr, '논밭': t, '합계': v, '더한값': s, '차': v - s})
chk719 = pd.DataFrame(chk719)
chk719.to_csv(os.path.join(mid, '_경지면적_719줄_점검.csv'), index=False, encoding='utf-8-sig')

# ---------- 사용가능일 ----------
# 친환경 공표일: 사이트에 '등록일'로 적힌 날짜만 공표일로 봄(파일 이름 날짜는 추출 기준일로 봄: 20251231 파일 등록일이 2026-08-06 이라서)
eco_pub = {
    2022: ('2023-08-08', '공표일 확인', 'data.mafra.go.kr 데이터 등록일 2023.08.08 (2022년까지 담긴 첫 파일 …_20230802.csv)'),
    2025: ('2026-08-06', '공표일 확인', 'data.go.kr 파일데이터 …_20251231 등록일 2026-08-06 (data.mafra 수정일 2026.08.06)'),
}


def rule2y(y):
    return f'{y + 2}-01-01'


def eco_avail(y):
    if y in eco_pub:
        return eco_pub[y]
    if y == 2023:
        return (rule2y(y), '자료연도+2년 규칙', '2023년까지 담긴 첫 파일 이름 날짜 20240805(추출 기준일로 봄), 사이트 등록일 확인 못 함')
    if y == 2024:
        return (rule2y(y), '자료연도+2년 규칙', '2024년까지 담긴 첫 파일 이름 날짜 20241231(추출 기준일로 봄), 사이트 등록일 확인 못 함')
    return (rule2y(y), '자료연도+2년 규칙', '이 해 값이 처음 공표된 날 확인 못 함(받은 가장 이른 파일 …_20230802, data.mafra 등록일 2023.08.08)')


# ---------- 계산 ----------
def unit_note(c, u, yr):
    if c == '28720':
        return '원자료에 인천 옹진군 줄이 없음(인천 합계 − 강화군 = 2871900000 줄(차 0~1ha), 이 줄은 설명 없음) → 인천광역시 전체 비율'
    if c == '12330':
        return '옛 광주광역시 자치구 중 광산구만 원자료에 자기 줄(2920000000)이 있어 자기 비율'
    if u == '2900000000':
        return '옛 광주광역시 전체(5개 구) 비율. 광산구는 자기 줄이 있음'
    if u == '2700000000' and yr <= 2023:
        return '대구광역시 합계에 2023년까지 군위군 면적 없음 → 분자도 군위군 빼고 셈'
    if u == '2700000000':
        return '대구광역시 합계에 2024년부터 군위군 포함 → 분자도 군위군 넣고 셈'
    return None


rows = []
unit_log = []
for yr in years:
    units, ratio_unit = M.kosis_units(sgg, yr)
    udict = {u[0]: u for u in units}
    ay = area[area['연도'] == yr]
    covered = set()
    for cs in ay['대응']:
        covered |= set(cs)
    # 분자: 단위 구성원 안에 대응이 다 들어가는 줄의 합
    unum = {}
    for code, kind, name, mem in units:
        ms = set(mem)
        inside = ay[[set(cs) <= ms for cs in ay['대응']]]
        straddle = ay[[bool(set(cs) & ms) and not set(cs) <= ms for cs in ay['대응']]]
        assert straddle.empty, (yr, code, straddle[['시도', '시군구']].values.tolist())
        ndl = len([m for m in mem if m not in covered])
        unum[code] = (inside['유기'].sum(), inside['무농약'].sum(), inside['계'].sum(), len(inside), ndl)
        non = k.loc[code, f'논_{yr}_면적']; bat = k.loc[code, f'밭_{yr}_면적']
        unit_log.append({'자료연도': yr, '계산단위_코드': code, '단위종류': kind, '단위이름': name, '구성원수': len(mem),
                         '구성원': ';'.join(mem), '원자료줄수': len(inside), '줄없는_구성원수': ndl,
                         '유기': inside['유기'].sum(), '무농약': inside['무농약'].sum(), '계': inside['계'].sum(),
                         '논': non, '밭': bat})
    ea = eco_avail(yr)
    # 경지면적(분모) 공표일: 확인된 날짜 없음 → 자료연도+2년 1월 1일 (사용자 확정 10-19: 같은 규칙)
    farm_avail = (rule2y(yr), '자료연도+2년 규칙', "data.go.kr 설명은 'KOSIS … 연말에 공시'뿐 날짜 없음(받은 파일 등록일 2026-08-14)")
    # 사용가능일 = max(친환경 자료, 경지면적) (사용자 확정 10-19)
    if ea[0] > farm_avail[0]:
        fin = (ea[0], f'친환경 자료가 정함: {ea[1]}')
    elif ea[0] < farm_avail[0]:
        fin = (farm_avail[0], f'경지면적이 정함: {farm_avail[1]}')
    else:
        fin = (ea[0], f'둘 다 같음: 친환경 {ea[1]}, 경지면적 {farm_avail[1]}')
    fin_detail = f'친환경 자료 {ea[0]}({ea[1]}; {ea[2]}) / 경지면적 {farm_avail[0]}({farm_avail[1]}) 중 늦은 날'
    for c in codes:
        u = ratio_unit[c]
        code, kind, name, mem = udict[u]
        org, nop, tot, nrow, ndl = unum[u]
        non = k.loc[u, f'논_{yr}_면적']; bat = k.loc[u, f'밭_{yr}_면적']
        farm = non + bat if (pd.notna(non) and pd.notna(bat)) else np.nan
        ratio = tot / farm if pd.notna(farm) and farm != 0 else np.nan
        rows.append({
            'sgg_code': c, '자료연도': int(yr),
            '사용가능일': fin[0], '사용가능일_근거': fin[1], '사용가능일_근거_상세': fin_detail,
            '친환경자료_사용가능일': ea[0], '친환경자료_사용가능일_근거': ea[1] + ' — ' + ea[2],
            '경지면적_사용가능일': farm_avail[0], '경지면적_사용가능일_근거': farm_avail[1] + ' — ' + farm_avail[2],
            '유기_인증면적': org, '무농약_인증면적': nop, '인증면적_합': tot,
            '경지면적': farm, '논_면적': non, '밭_면적': bat, '면적_단위': 'ha',
            '비율_계산단위': kind, '계산단위_코드': u, '계산단위_이름': name, '계산단위_시군구': ';'.join(mem),
            '친환경_비율': ratio,
            '0채움_여부': c not in covered,
            '계산단위_전체_0채움': nrow == 0,
            '계산단위_줄없는_시군구수': ndl,
            '결정대기_사유': ('값 영향 없음' if (c, yr) in pending else None),
            '시행구행_메모': pending.get((c, yr)),
            '계산단위_메모': unit_note(c, u, yr),
        })
out = pd.DataFrame(rows)
out['사용가능일'] = pd.to_datetime(out['사용가능일']).dt.date
out['경지면적_사용가능일'] = pd.to_datetime(out['경지면적_사용가능일']).dt.date
out['친환경자료_사용가능일'] = pd.to_datetime(out['친환경자료_사용가능일']).dt.date
import pyarrow as pa, pyarrow.parquet as pq
schema_fix = {'사용가능일': pa.date32(), '친환경자료_사용가능일': pa.date32(), '경지면적_사용가능일': pa.date32(), '자료연도': pa.int64()}
tbl = pa.Table.from_pandas(out, preserve_index=False)
for name, typ in schema_fix.items():
    i = tbl.schema.get_field_index(name)
    tbl = tbl.set_column(i, pa.field(name, typ), tbl.column(name).cast(typ))
pq.write_table(tbl, out_path)
pd.DataFrame(unit_log).to_csv(os.path.join(mid, '_친환경_계산단위_기록.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(maprows).to_csv(os.path.join(mid, '_대응_친환경.csv'), index=False, encoding='utf-8-sig')
print('rows', len(out), 'years', years[0], years[-1], 'sgg', out['sgg_code'].nunique())
print('비율 빈칸', int(out['친환경_비율'].isna().sum()), '0채움', int(out['0채움_여부'].sum()), '결정대기', int(out['결정대기_사유'].notna().sum()))
print('719 점검 차 0 아닌 칸:', int((chk719['차'].abs() > 1e-9).sum()), '/', len(chk719))
