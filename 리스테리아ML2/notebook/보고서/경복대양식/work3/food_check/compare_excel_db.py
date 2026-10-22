# -*- coding: utf-8 -*-
"""리서치팀 엑셀('값' 시트) vs ba_data.t_aw_cl 추출본 — 품목코드별 aw·ph·na 값 비교 (읽기만)

입력(읽기만)
  - 엑셀 : 천영진/문서/리서치팀문서/통합_수분활성도_수분함량_식품pH_염도_최종.xlsx '값' 시트
  - DB   : 리스테리아ML2/data/raw/raw_t_aw_cl.parquet (2026-10-16 17:18, select * from ba_data.t_aw_cl)
  - 참고 : 리스테리아ML2/data/데이터셋/시료_시군구.parquet (다른 코드에 걸린 시료 행 수만 셈)
           리스테리아ML2/data/중간/품목3단계_식품속성_근거.csv (대표값 자료가 엑셀인 3단계)
출력(이 폴더에만)
  - 비교_품목코드별.csv  : 품목코드 × 항목 1줄 (원문·숫자·표기·분류)
  - 비교_요약.json       : 분류별 품목코드 수
같음의 기준: 06_품목_식품속성.py 의 parse_food 규칙(아래에 그대로 복사)으로 원문을 (숫자, 표기)로 읽은 뒤
  숫자가 math.isclose(rel_tol=1e-9, abs_tol=1e-12) 이면 '숫자 같음', 표기(단일값·중앙·이상·이하·초과·미만)까지 같으면 '같음'.
  원문 글자 그대로 같은지도 따로 셈(엑셀 숫자 칸은 글자로 바꿔 비교).
"""
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

OUT = Path(__file__).resolve().parent
XLSX = Path(r'C:\Users\NT751\Desktop\천영진\문서\리서치팀문서\통합_수분활성도_수분함량_식품pH_염도_최종.xlsx')
RAW = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\raw\raw_t_aw_cl.parquet')
SAMPLE = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋\시료_시군구.parquet')
EVID = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\중간\품목3단계_식품속성_근거.csv')
FOOD_COLS = ['aw', 'ph', 'na']
FOOD_KO = {'aw': '수분활성도', 'ph': '식품pH', 'na': '염도'}
XL_COL = {'aw': '수분활성도', 'ph': '식품 pH', 'na': '염도'}

# ---------------- 06_품목_식품속성.py 의 parse_food / parse_food_n 그대로 (복사)
NUM = r'-?\d+(?:\.\d+)?'
RANGE_SEP = r'[~～∼\-–]'
SIGN_LABEL = {'<': '미만', '>': '초과', '≤': '이하', '≥': '이상', '<=': '이하', '>=': '이상'}
SIGN_FLIP = {'미만': '초과', '초과': '미만', '이하': '이상', '이상': '이하'}


def parse_food(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return math.nan, '빈칸', ''
    if isinstance(s, (int, float, np.integer, np.floating)) and not isinstance(s, bool):
        return float(s), '숫자 그대로', '단일값'
    t = str(s).strip()
    if t == '':
        return math.nan, '빈칸', ''
    if re.fullmatch(NUM, t):
        return float(t), '숫자 그대로', '단일값'
    m = re.fullmatch(rf'({NUM})\s*{RANGE_SEP}\s*({NUM})', t)
    if m:
        return (float(m[1]) + float(m[2])) / 2, '범위 → 가운데 값', '중앙'
    m = re.fullmatch(rf'([<>≤≥]=?)\s*({NUM})', t)
    if m:
        return float(m[2]), '부등호 → 경계값', SIGN_LABEL[m[1]]
    m = re.fullmatch(rf'({NUM})\s*([<>≤≥]=?)', t)
    if m:
        return float(m[1]), '부등호 → 경계값', SIGN_FLIP[SIGN_LABEL[m[2]]]
    m = re.fullmatch(rf'({NUM})\s*(이상|이하|초과|미만)', t)
    if m:
        return float(m[1]), '부등호 → 경계값', m[2]
    return math.nan, '해석 못 함 → 결측', ''


def parse_food_n(s):
    x, rule, label = parse_food(s)
    if math.isnan(x):
        return x, rule, label
    return round(x, 12), rule, label


def raw_txt(v):
    """원문을 글자로 (엑셀 숫자 칸은 파이썬 기본 글자; 정수 모양 float 는 정수 글자)"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


# ---------------- 읽기
db = pq.read_table(RAW).to_pydict()
assert pq.read_schema(RAW).metadata[b'source_table'] == b'ba_data.t_aw_cl'
DB = {c: {k: db[k][i] for k in db} for i, c in enumerate(db['prdlst_cd'])}
assert len(DB) == len(db['prdlst_cd'])

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
rows = list(wb['값'].iter_rows(values_only=True))
wb.close()
H = list(rows[0])
assert H == ['PRDLST_CD', 'KOR_NM', 'LV', '수분활성도', '수분함량', '식품 pH', '염도'], H
XL = {}
for r in rows[1:]:
    if all(v is None for v in r):
        continue
    assert r[0] not in XL
    XL[r[0]] = dict(zip(H, r))

codes_both = sorted(set(XL) & set(DB))
codes_xl_only = sorted(set(XL) - set(DB))
codes_db_only = sorted(set(DB) - set(XL))
print(f'엑셀 품목코드 {len(XL)} · DB 품목코드 {len(DB)} · 둘 다 {len(codes_both)} · 엑셀에만 {len(codes_xl_only)} · DB 에만 {len(codes_db_only)}')

# 같은 코드의 이름·레벨이 같은지 (같은 품목인지)
name_diff = [(c, XL[c]['KOR_NM'], DB[c]['kor_nm']) for c in codes_both if str(XL[c]['KOR_NM']).strip() != str(DB[c]['kor_nm']).strip()]
lv_diff = [(c, XL[c]['LV'], DB[c]['lv']) for c in codes_both if float(XL[c]['LV']) != float(DB[c]['lv'])]
print(f'둘 다 있는 코드 중 이름 다름 {len(name_diff)} · 레벨 다름 {len(lv_diff)}', name_diff[:10], lv_diff[:10])
# 이름 기준: DB 에만 있는 코드의 이름이 엑셀 이름과 겹치는지
xl_names = defaultdict(list)
for c in XL:
    xl_names[str(XL[c]['KOR_NM']).strip()].append(c)
db_only_name_hit = [(c, DB[c]['kor_nm'], xl_names[str(DB[c]['kor_nm']).strip()]) for c in codes_db_only if str(DB[c]['kor_nm']).strip() in xl_names]
print(f'DB 에만 있는 코드 중 이름이 엑셀 이름과 같은 코드 {len(db_only_name_hit)}', db_only_name_hit[:10])
print('DB 에만 있는 코드의 lv 분포:', dict(Counter(DB[c]['lv'] for c in codes_db_only)), '· 엑셀 코드 lv 분포:', dict(Counter(XL[c]['LV'] for c in XL)))

# ---------------- 품목코드 × 항목 비교
OUTROWS = []
SUM = {}
for f in FOOD_COLS:
    k = Counter()
    for c in sorted(set(XL) | set(DB)):
        xr = XL[c][XL_COL[f]] if c in XL else None
        dr = DB[c][f] if c in DB else None
        xn, xrule, xlab = parse_food_n(xr)
        dn, drule, dlab = parse_food_n(dr)
        xv, dv = not math.isnan(xn), not math.isnan(dn)
        if xv and dv:
            if not math.isclose(xn, dn, rel_tol=1e-9, abs_tol=1e-12):
                cat = '(가) 둘 다 값 · 숫자 다름'
            elif xlab != dlab:
                cat = '(나) 둘 다 값 · 숫자 같고 표기 다름'
            else:
                cat = '(다) 둘 다 값 · 같음'
        elif xv:
            cat = '(라) 엑셀에만 값' + ('' if c in DB else ' (DB 에 코드 없음)')
        elif dv:
            cat = '(마) DB 에만 값' + (' (엑셀에 코드 없음)' if c not in XL else f' (엑셀 원문 {raw_txt(xr) or "빈칸"!r})')
        else:
            cat = '(바) 둘 다 값 없음'
        # 엑셀 원문 '-' 등은 묶어서 셈
        cat_k = re.sub(r" \(엑셀 원문 '.*'\)", ' (엑셀에 코드 있음, 값 없음)', cat)
        k[cat_k] += 1
        raw_same = (c in XL and c in DB and raw_txt(xr) == raw_txt(dr))
        OUTROWS.append({
            '항목': f, '항목명': FOOD_KO[f], '품목코드': c,
            '이름_엑셀': XL[c]['KOR_NM'] if c in XL else '', '이름_DB': DB[c]['kor_nm'] if c in DB else '',
            'lv_DB': DB[c]['lv'] if c in DB else '', 'aw_source_DB': DB[c]['aw_source'] if c in DB else '',
            '원문_엑셀': raw_txt(xr), '원문_DB': raw_txt(dr), '원문_글자같음': '예' if raw_same else '아니오',
            '숫자_엑셀': '' if not xv else f'{xn:.10g}', '표기_엑셀': xlab,
            '숫자_DB': '' if not dv else f'{dn:.10g}', '표기_DB': dlab,
            '분류': cat_k,
        })
    SUM[f] = dict(sorted(k.items()))

with open(OUT / '비교_품목코드별.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(OUTROWS[0].keys()))
    w.writeheader()
    w.writerows(OUTROWS)

print('\n== 분류별 품목코드 수 ==')
for f in FOOD_COLS:
    print(f'[{FOOD_KO[f]} {f}]')
    for kk, v in SUM[f].items():
        print(f'   {kk}: {v}')

# 둘 다 코드가 있는 653개 안에서만: 둘 다 값 / 다름 / 한쪽만
print('\n== 두 자료에 다 있는 품목코드(653개) 안에서 ==')
SUM2 = {}
for f in FOOD_COLS:
    sub = [r for r in OUTROWS if r['항목'] == f and r['품목코드'] in XL and r['품목코드'] in DB]
    both_v = [r for r in sub if r['숫자_엑셀'] and r['숫자_DB']]
    num_diff = [r for r in both_v if r['분류'].startswith('(가)')]
    lab_diff = [r for r in both_v if r['분류'].startswith('(나)')]
    raw_diff_but_same = [r for r in both_v if r['분류'].startswith('(다)') and r['원문_글자같음'] == '아니오']
    xl_only = [r for r in sub if r['숫자_엑셀'] and not r['숫자_DB']]
    db_only = [r for r in sub if r['숫자_DB'] and not r['숫자_엑셀']]
    none = [r for r in sub if not r['숫자_DB'] and not r['숫자_엑셀']]
    SUM2[f] = dict(둘다값=len(both_v), 숫자다름=len(num_diff), 숫자같고표기다름=len(lab_diff), 같음=len(both_v) - len(num_diff) - len(lab_diff),
                   같음중_원문글자다름=len(raw_diff_but_same), 엑셀에만값=len(xl_only), DB에만값=len(db_only), 둘다없음=len(none))
    print(f'[{FOOD_KO[f]}]', SUM2[f])
    print('   DB 에만 값인 코드의 엑셀 원문:', dict(Counter(r['원문_엑셀'] or '빈칸' for r in db_only)))
    print('   DB 에만 값인 코드의 aw_source:', dict(Counter((r['aw_source_DB'] or '').split('(')[0] for r in db_only)) if f == 'aw' else '')
    print('   숫자 다름 전부:')
    for r in num_diff:
        print(f"     {r['품목코드']} {r['이름_엑셀']} | 엑셀 {r['원문_엑셀']!r} → {r['숫자_엑셀']}({r['표기_엑셀']}) | DB {r['원문_DB']!r} → {r['숫자_DB']}({r['표기_DB']}) | aw_source {r['aw_source_DB']!r}")
    print('   숫자 같고 표기 다름 (원문 짝별 코드 수):', dict(Counter((r['원문_엑셀'], r['원문_DB']) for r in lab_diff)))
    print('   같음인데 원문 글자 다름 (원문 짝 예 10):', Counter((r['원문_엑셀'], r['원문_DB']) for r in raw_diff_but_same).most_common(10))

# DB 에만 있는 663 코드: 값 있음 수
print('\n== DB 에만 있는 품목코드 ==')
for f in FOOD_COLS:
    v = sum(1 for c in codes_db_only if not math.isnan(parse_food_n(DB[c][f])[0]))
    print(f'[{FOOD_KO[f]}] DB 에만 있는 코드 {len(codes_db_only)}개 중 값 있음 {v}')
print('DB 에만 있는 코드의 aw_source:', dict(Counter((DB[c]['aw_source'] or '').split('(')[0] for c in codes_db_only)))

# ---------------- 참고: 다른 코드에 걸린 시료 행 (시료_시군구, prdlst_code 정확 일치 → 앞 14자리)
smp = pq.read_table(SAMPLE, columns=['prdlst_code', 'judge_result']).to_pydict()
pc_cnt = Counter(zip(smp['prdlst_code'], smp['judge_result']))
REF = {}
for f in FOOD_COLS:
    diff_codes = {r['품목코드'] for r in OUTROWS if r['항목'] == f and r['분류'].startswith(('(가)', '(나)'))}
    numd_codes = {r['품목코드'] for r in OUTROWS if r['항목'] == f and r['분류'].startswith('(가)')}
    n_rows = Counter()
    for (p, j), n in pc_cnt.items():
        key = p if (p in XL or p in DB) else p[:14]
        if key in numd_codes:
            n_rows['숫자다름_행'] += n
            n_rows['숫자다름_검출'] += n * (j == '검출')
        if key in diff_codes:
            n_rows['숫자또는표기다름_행'] += n
            n_rows['숫자또는표기다름_검출'] += n * (j == '검출')
    REF[f] = dict(n_rows)
print('\n== 참고: 시료_시군구 628,517행 중 다른 코드에 걸린 행(prdlst_code 정확 일치 → 앞 14자리) ==', REF)

# 3단계 대표값 표에서 엑셀 값을 쓴 줄
ev = list(csv.DictReader(open(EVID, encoding='utf-8-sig')))
for f in FOOD_COLS:
    sub = [r for r in ev if r['항목'] == f]
    src = Counter(r['대표값_자료'] for r in sub)
    print(f'[{FOOD_KO[f]}] 3단계 대표값 자료별 줄 수:', dict(src))

json.dump({'코드수': dict(엑셀=len(XL), DB=len(DB), 둘다=len(codes_both), 엑셀에만=len(codes_xl_only), DB에만=len(codes_db_only)),
           '이름다름': len(name_diff), '레벨다름': len(lv_diff),
           '분류별_전체코드': SUM, '두자료공통코드_안에서': SUM2, '참고_시료행': REF},
          open(OUT / '비교_요약.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n저장:', OUT / '비교_품목코드별.csv', OUT / '비교_요약.json')
