# -*- coding: utf-8 -*-
"""
build_food_db_table.py — 품목 3단계(train_prdlst_code_2)마다 식품 속성(수분활성도 aw · 식품pH ph · 염도 na) 'DB 고정표'

3차 표(식정원_db_식품/품목3단계_식품속성.parquet = 리서치팀 엑셀 우선 + 2024-06-30 까지 시료 최빈값)를 대신하는 표.
규칙은 사용자 2026-10-27 확정(exp/다음실험_계획_4차.md 1-1, 2절 미결 3 '정함'). 기존 파일은 읽기만 하고 덮어쓰지 않는다.

입력 (읽기만)
  - data/raw/raw_t_prdlst_cl.parquet          (ba_data.t_prdlst_cl, 2026-10-26 16:30 조회)  — 품목 분류표
  - data/raw/raw_t_aw_cl_20261026.parquet     (ba_data.t_aw_cl, 2026-10-26 16:30 조회)     — 식품 속성 원문
  - data/데이터셋/시료_시군구.parquet           — 3단계 코드 목록·2단계 짝·검출 수
  - data/데이터셋/식정원_db_식품/품목3단계_식품속성.parquet (3차 표, 비교용)
  - data/데이터셋/생장범위/생장범위_품목3단계.parquet (3차 생장 범위 표, 칸·메타데이터 형식 참고)

출력 (data/데이터셋/)
  - 품목3단계_식품속성_DB고정표_20261027.parquet : 3차 표와 같은 칸(train_prdlst_code_2, train_prdlst_nm_2, train_prdlst_code,
        aw, aw_label, ph, ph_label, na, na_label — 같은 자료형) + 덧붙인 칸 {aw,ph,na}_rule(string), {aw,ph,na}_n_sub(int32)
  - 생장범위_품목3단계_DB고정표_20261027.parquet : 3차 생장범위_품목3단계.parquet 와 같은 칸·형식(make_ranges.py 와 같은 계산)
  - 품목3단계_식품속성_DB고정표_20261027.md      : 규칙·건수·결측률·3차 비교

규칙
  1) 원문 → (숫자, 표기): 06_품목_식품속성.py 의 parse_food / parse_food_n 그대로(아래에 복사).
     범위 = 가운데 값('중앙'), 부등호·이상·이하·초과·미만 = 경계값, 단일 숫자 = '단일값', '-' 등 해석 못 하는 글 = 값 없음.
     음수로 읽히는 원문도 해석 못 함 → 값 없음(aw·pH·염도는 음수가 될 수 없음 — 사용자 2026-10-27, 영양강화 밀가루 aw '-0.85').
  2) t_aw_cl 의 aw 는 aw_source = 'original' 인 행만 씀. ph·na 는 출처 칸이 없어 aw_source 와 관계없이 표 값 그대로.
  3) 하위 코드 = t_prdlst_cl 에서 hrnk_prdlst_cd(상위 코드)를 따라 내려가 그 코드 아래에 있는 모든 코드(자기 자신 제외, lv 무관,
     vald_end_dt 지난 코드 포함).
  4) 대표값(항목마다 따로): ① 3단계 코드 자체 값('자기값') → ② 하위 코드 중 값이 있는 코드들의 값 중앙값('하위중앙', 코드당 한 번)
     → ③ 시료에서 짝지어진 2단계 코드(train_prdlst_code)의 DB 값('2단계값') → ④ 결측.
     3단계 코드 하나가 시료에서 2단계 코드 여럿과 짝지어지면 시료 수가 가장 많은 짝(같으면 코드 순 앞)을 씀.
     표기({c}_label): ①·③ 은 그 값의 표기(단일값·중앙·이상·이하·초과·미만), ② 는 '하위중앙', ④ 는 null(3차 표와 같게 null —
     모델에 넣을 때 '결측' 범주로 바꾸는 것도 3차와 같음).
     중앙값은 numpy median(값 개수가 짝수면 가운데 두 값의 평균) 후 소수 12자리로 맞춤(부동소수 오차만 없앰, 3차 parse_food_n 과 같은 이유).
  값은 원문 그대로(결측 채움·단위 변환·이상값 삭제 없음).

실행: py "리스테리아ML2\\data\\데이터셋\\build_food_db_table.py"   (numpy·pyarrow 만, 무작위 없음 — 다시 돌리면 같은 결과)
"""
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

T0 = time.time()
BASE = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2')
DS = BASE / 'data' / '데이터셋'
RAW_CL = BASE / 'data' / 'raw' / 'raw_t_prdlst_cl.parquet'
RAW_AW = BASE / 'data' / 'raw' / 'raw_t_aw_cl_20261026.parquet'
SAMPLE = DS / '시료_시군구.parquet'
OLD_FOOD = DS / '식정원_db_식품' / '품목3단계_식품속성.parquet'
OLD_RG = DS / '생장범위' / '생장범위_품목3단계.parquet'
TAG = '20261027'
OUT_FOOD = DS / f'품목3단계_식품속성_DB고정표_{TAG}.parquet'
OUT_RG = DS / f'생장범위_품목3단계_DB고정표_{TAG}.parquet'
OUT_MD = DS / f'품목3단계_식품속성_DB고정표_{TAG}.md'
DB_DATE = '2026-10-26 16:30 (ba_data.t_prdlst_cl · ba_data.t_aw_cl 조회 시각, data/raw/출처_t_prdlst_cl_t_aw_cl_20261026.md)'
SCRIPT = '리스테리아ML2/data/데이터셋/build_food_db_table.py'

for p in (OUT_FOOD, OUT_RG, OUT_MD):
    assert not p.exists(), f'이미 있음 — 덮어쓰지 않음: {p}'

FOOD_COLS = ['aw', 'ph', 'na']
FOOD_KO = {'aw': '수분활성도', 'ph': '식품pH', 'na': '염도'}
FOOD_LABELS = ['단일값', '중앙', '이상', '이하', '초과', '미만']
RULES = ['자기값', '하위중앙', '2단계값', '결측']
RULE_TXT = {'자기값': '① 3단계 코드 자체 값', '하위중앙': '② 하위 코드 값 중앙값', '2단계값': '③ 2단계 코드 값', '결측': '④ 결측'}


def log(*a):
    print(f'[{time.time() - T0:5.1f}초]', *a, flush=True)


# ---------------------------------------------------------------- 1) 원문 → 숫자 — 06_품목_식품속성.py parse_food / parse_food_n 그대로 복사
NUM = r'-?\d+(?:\.\d+)?'
RANGE_SEP = r'[~～∼\-–]'
SIGN_LABEL = {'<': '미만', '>': '초과', '≤': '이하', '≥': '이상', '<=': '이하', '>=': '이상'}
SIGN_FLIP = {'미만': '초과', '초과': '미만', '이하': '이상', '이상': '이하'}


def parse_food(s):
    """원문 → (숫자, 적용 규칙, 값 표기). 값 표기: 단일값 · 중앙 · 이상 · 이하 · 초과 · 미만. 값이 없으면 숫자 nan, 표기 ''"""
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


NUM_ROUND = 12


def parse_food_n(s):
    x, rule, label = parse_food(s)
    if math.isnan(x):
        return x, rule, label
    y = round(x, NUM_ROUND)
    assert y == x or rule == '범위 → 가운데 값', (s, x, y)
    assert abs(y - x) < 1e-12
    return y, rule, label


def parse_db(s):
    """parse_food_n + 음수 원문은 해석 못 함 → 결측(사용자 2026-10-27 — 영양강화 밀가루 aw '-0.85')"""
    x, rule, label = parse_food_n(s)
    if not math.isnan(x) and x < 0:
        return math.nan, '해석 못 함 → 결측', ''
    return x, rule, label


def num_txt(x):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.10g}'


# ---------------------------------------------------------------- 분류표
cl = pq.read_table(RAW_CL)
assert cl.schema.metadata[b'source_table'] == b'ba_data.t_prdlst_cl'
CL = cl.select(['prdlst_cd', 'kor_nm', 'lv', 'hrnk_prdlst_cd', 'vald_end_dt']).to_pydict()
CL_CODES = CL['prdlst_cd']
assert None not in CL_CODES and len(set(CL_CODES)) == len(CL_CODES)
CL_LV = dict(zip(CL_CODES, CL['lv']))
CL_NM = dict(zip(CL_CODES, CL['kor_nm']))
CL_END = dict(zip(CL_CODES, CL['vald_end_dt']))
CHILDREN = defaultdict(list)
for code, up in zip(CL_CODES, CL['hrnk_prdlst_cd']):
    if up is not None and up != code:
        CHILDREN[up].append(code)


def descendants(code):
    """hrnk_prdlst_cd 를 따라 code 아래의 모든 코드(자기 제외). 순환이 있으면 멈춤"""
    out, stack, seen = [], list(CHILDREN.get(code, [])), {code}
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        stack.extend(CHILDREN.get(c, []))
    return sorted(out)


log(f't_prdlst_cl {len(CL_CODES):,}행 · 상위 코드가 있는 코드 {sum(len(v) for v in CHILDREN.values()):,}')

# ---------------------------------------------------------------- 식품 속성표
awt = pq.read_table(RAW_AW)
assert awt.schema.metadata[b'source_table'] == b'ba_data.t_aw_cl'
AWD = awt.to_pydict()
assert None not in AWD['prdlst_cd'] and len(set(AWD['prdlst_cd'])) == len(AWD['prdlst_cd'])
DB = {code: {k: AWD[k][i] for k in AWD} for i, code in enumerate(AWD['prdlst_cd'])}
N_AW = len(DB)
NON_ORIG = sorted(c for c in DB if DB[c]['aw_source'] != 'original')
NON_ORIG_SRC = Counter(DB[c]['aw_source'] for c in NON_ORIG)
NON_ORIG_PH = [c for c in NON_ORIG if not math.isnan(parse_db(DB[c]['ph'])[0])]
NON_ORIG_NA = [c for c in NON_ORIG if not math.isnan(parse_db(DB[c]['na'])[0])]
AW_NOT_IN_CL = sorted(c for c in DB if c not in CL_LV)

# 항목 → 코드 → (숫자, 표기, 원문)  (값이 있는 코드만)
VAL = {c: {} for c in FOOD_COLS}
PARSE_RULES = {c: Counter() for c in FOOD_COLS}
UNPARSED = {c: Counter() for c in FOOD_COLS}
for code, row in DB.items():
    for c in FOOD_COLS:
        x, rule, label = parse_db(row[c])
        if c == 'aw' and row['aw_source'] != 'original':
            PARSE_RULES[c]['(aw_source ≠ original → 쓰지 않음)'] += 1
            continue
        PARSE_RULES[c][rule] += 1
        if rule == '해석 못 함 → 결측':
            UNPARSED[c][str(row[c]).strip()] += 1
        if not math.isnan(x):
            assert label in FOOD_LABELS
            VAL[c][code] = (x, label, str(row[c]).strip())
log('t_aw_cl', N_AW, '행 · aw_source ≠ original', len(NON_ORIG), '행 · 값 있는 코드 수',
    {c: len(VAL[c]) for c in FOOD_COLS})

# ---------------------------------------------------------------- 시료
smp = pq.read_table(SAMPLE, columns=['lims_data_key', 'judge_result', 'train_prdlst_code', 'train_prdlst_nm',
                                     'train_prdlst_code_2', 'train_prdlst_nm_2'])
N = smp.num_rows
S = smp.to_pydict()
assert set(S['judge_result']) <= {'검출', '불검출'}
det = np.array([j == '검출' for j in S['judge_result']])
tp2 = S['train_prdlst_code_2']
has3 = np.array([x is not None for x in tp2])
PAIR = Counter((k3, k2) for k3, k2 in zip(tp2, S['train_prdlst_code']) if k3 is not None)
NM2 = {}
NM1 = {}
for k3, n3, k2, n2 in zip(tp2, S['train_prdlst_nm_2'], S['train_prdlst_code'], S['train_prdlst_nm']):
    if k3 is not None:
        assert NM2.setdefault(k3, n3) == n3
        NM1.setdefault(k2, n2)
CODES3 = sorted(NM2)
BY3 = defaultdict(list)
for (k3, k2), n in PAIR.items():
    BY3[k3].append((k2, n))
MULTI2 = {k3: sorted(v, key=lambda t: (-t[1], t[0] or '')) for k3, v in BY3.items() if len(v) > 1}
K2_OF = {k3: sorted(v, key=lambda t: (-t[1], t[0] or ''))[0][0] for k3, v in BY3.items()}
assert all(K2_OF[k] is not None for k in CODES3)
N3_ROWS = Counter(k for k in tp2 if k is not None)
N3_DET = Counter(k for k, d in zip(tp2, det) if k is not None and d)
log(f'시료 {N:,}행 (검출 {int(det.sum()):,}) · 3단계 코드 {len(CODES3)}개 · 3단계 빈 행 {int((~has3).sum())} · 2단계 여럿과 짝인 3단계 {len(MULTI2)}개')
NOT_IN_CL3 = [k for k in CODES3 if k not in CL_LV]
NOT_IN_CL2 = sorted({K2_OF[k] for k in CODES3 if K2_OF[k] not in CL_LV})
assert not NOT_IN_CL3, NOT_IN_CL3

# ---------------------------------------------------------------- 대표값
DESC = {k: descendants(k) for k in CODES3}
REP = {c: {} for c in FOOD_COLS}
for c in FOOD_COLS:
    for k3 in CODES3:
        sub = [d for d in DESC[k3] if d in VAL[c]]
        r = dict(n_desc=len(DESC[k3]), n_sub_val=len(sub), src='', n_sub=None)
        if k3 in VAL[c]:
            x, lab, raw = VAL[c][k3]
            r.update(rule='자기값', num=x, label=lab, src=f'{k3} "{raw}"')
        elif sub:
            vals = np.array([VAL[c][d][0] for d in sub], dtype=np.float64)
            r.update(rule='하위중앙', num=round(float(np.median(vals)), NUM_ROUND), label='하위중앙', n_sub=len(sub),
                     src=f'하위 {len(sub)}개 ({num_txt(vals.min())}~{num_txt(vals.max())})')
        elif K2_OF[k3] in VAL[c]:
            k2 = K2_OF[k3]
            x, lab, raw = VAL[c][k2]
            r.update(rule='2단계값', num=x, label=lab, src=f'{k2} "{raw}"')
        else:
            r.update(rule='결측', num=None, label=None)
        REP[c][k3] = r

# ---------------------------------------------------------------- 결과 표 저장 (3차 표와 같은 칸·자료형 + 덧붙인 칸)
old = pq.read_table(OLD_FOOD)
OLD_COLS = old.schema.names
cols = {
    'train_prdlst_code_2': pa.array(CODES3, type=pa.string()),
    'train_prdlst_nm_2': pa.array([NM2[k] for k in CODES3], type=pa.string()),
    'train_prdlst_code': pa.array([K2_OF[k] for k in CODES3], type=pa.string()),
}
for c in FOOD_COLS:
    cols[c] = pa.array([REP[c][k]['num'] for k in CODES3], type=pa.float64())
    cols[f'{c}_label'] = pa.array([REP[c][k]['label'] for k in CODES3], type=pa.string())
for c in FOOD_COLS:
    cols[f'{c}_rule'] = pa.array([REP[c][k]['rule'] for k in CODES3], type=pa.string())
    cols[f'{c}_n_sub'] = pa.array([REP[c][k]['n_sub'] for k in CODES3], type=pa.int32())
OUT = pa.table(cols)
assert OUT.schema.names[:len(OLD_COLS)] == OLD_COLS
assert OUT.select(OLD_COLS).schema.equals(old.schema, check_metadata=False)
COL_DEF = {
    'train_prdlst_code_2': '품목 3단계 코드(키) — 시료_시군구.parquet 에 나오는 3단계 코드 전부',
    'train_prdlst_nm_2': '3단계 이름(시료 파일 값)',
    'train_prdlst_code': "'2단계값' 규칙에 쓴 2단계 코드(시료에서 이 3단계와 짝지어진 2단계 중 시료 수 최다)",
    **{c: f'{FOOD_KO[c]} 대표값(원문을 숫자로 바꾼 값; 결측 null)' for c in FOOD_COLS},
    **{f'{c}_label': f'{FOOD_KO[c]} 표기: 자기값·2단계값이면 그 값의 표기(단일값·중앙·이상·이하·초과·미만), 하위중앙이면 "하위중앙", 결측이면 null'
       for c in FOOD_COLS},
    **{f'{c}_rule': f'{FOOD_KO[c]} 대표값을 정한 규칙: 자기값 / 하위중앙 / 2단계값 / 결측' for c in FOOD_COLS},
    **{f'{c}_n_sub': f'{FOOD_KO[c]} 하위중앙일 때 중앙값에 쓴 하위 코드 수(그 밖은 null)' for c in FOOD_COLS},
}
OUT = OUT.replace_schema_metadata({
    'script': SCRIPT,
    'db_date': DB_DATE,
    'rule': ("3단계(train_prdlst_code_2)마다 항목별: ① 3단계 코드 자체의 t_aw_cl 값 -> ② t_prdlst_cl hrnk_prdlst_cd 로 찾은 하위 코드 전부"
             "(lv 무관, 유효 종료 포함) 중 값이 있는 코드 값의 중앙값 -> ③ 2단계 코드(train_prdlst_code) 값 -> ④ 결측(null). "
             "aw 는 aw_source='original' 행만, ph·na 는 표 값 그대로. 원문->숫자는 06_품목_식품속성.py parse_food 그대로"),
    'columns': json.dumps(COL_DEF, ensure_ascii=False),
})
pq.write_table(OUT, OUT_FOOD)
log(f'저장: {OUT_FOOD.name} {OUT.num_rows}행 × {OUT.num_columns}칸')

# ---------------------------------------------------------------- 생장 범위 표 (make_ranges.py 와 같은 계산)
AW_OPT, AW_LIM, PH_OPT, PH_LIM = 0.97, (0.90, 0.997), 7.0, (4.0, 9.6)


def in_range(x, lo, hi):
    out = ((x >= lo) & (x <= hi)).astype(np.float64)
    out[np.isnan(x)] = np.nan
    return out


def to_np(col):
    return pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False)


def to_pa32(v):
    return pa.array(v.astype(np.float32), type=pa.float32(), from_pandas=True)


aw, ph = to_np(OUT['aw']), to_np(OUT['ph'])
RG = {
    'rg_aw_lim': in_range(aw, *AW_LIM),
    'rg_aw_opt_dist': np.abs(aw - AW_OPT),
    'rg_ph_lim': in_range(ph, *PH_LIM),
    'rg_ph_opt_dist': np.abs(ph - PH_OPT),
}
old_rg = pq.read_table(OLD_RG)
meta = {k.decode(): v.decode() for k, v in old_rg.schema.metadata.items()}
meta['inputs'] = json.dumps([OUT_FOOD.name], ensure_ascii=False)
meta['script'] = SCRIPT + ' (리스테리아ML2/data/데이터셋/생장범위/scripts/make_ranges.py 3) 과 같은 계산)'
meta['description'] = meta['description'] + f' — DB 고정표({OUT_FOOD.name}) 값으로 다시 계산'
meta['columns'] = meta['columns'].replace('부등호 표기 품목은 생장범위_부등호표기.csv', f'부등호 표기 품목은 {OUT_MD.name}')
out_rg = pa.table({'train_prdlst_code_2': OUT['train_prdlst_code_2'], **{k: to_pa32(v) for k, v in RG.items()}})
out_rg = out_rg.replace_schema_metadata(meta)
assert out_rg.schema.equals(old_rg.schema, check_metadata=False)
pq.write_table(out_rg, OUT_RG)
log(f'저장: {OUT_RG.name} {out_rg.num_rows}행 × {out_rg.num_columns}칸')

# ---------------------------------------------------------------- 요약 수치
OLDD = old.to_pydict()
OLD = {k: {c: (OLDD[c][i], OLDD[f'{c}_label'][i]) for c in FOOD_COLS} for i, k in enumerate(OLDD['train_prdlst_code_2'])}
assert set(OLD) == set(CODES3), '3차 표와 3단계 코드 목록이 다름'
OLD_RG_D = old_rg.to_pydict()


def pct(a, b):
    return f'{100.0 * a / b:.2f}' if b else '-'


def same_num(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


ROW3 = np.array([N3_ROWS[k] for k in CODES3])
DET3 = np.array([N3_DET[k] for k in CODES3])
NDET = int(det.sum())
lines = []
W = lines.append

W(f'# 품목 3단계 식품 속성 DB 고정표 ({TAG})')
W('')
W(f'- 만든 날: 2026-10-27 · DB 기준일: {DB_DATE}')
W(f'- 만든 스크립트: `{SCRIPT}` (DB 접속 없음, 받아 둔 parquet 만 읽음. 무작위 없음)')
W(f'- 결과: `{OUT_FOOD.name}` ({OUT.num_rows}행 × {OUT.num_columns}칸), `{OUT_RG.name}` ({out_rg.num_rows}행 × {out_rg.num_columns}칸)')
W('- 대신하는 3차 파일(그대로 둠): `식정원_db_식품/품목3단계_식품속성.parquet`, `생장범위/생장범위_품목3단계.parquet`')
W('- 규칙 근거: `리스테리아ML2/exp/다음실험_계획_4차.md` 1-1, 2절 미결 3(사용자 2026-10-27 정함)')
W('')
W('## 1. 규칙')
W('')
W('1. **원문 → 숫자**: 3차(`notebook/06_품목_식품속성.py` parse_food)와 같음. 단일 숫자 = 그 값(표기 `단일값`), 범위(`~`, `-` 등) = 가운데 값(`중앙`), '
  '`<`·`>`·`≤`·`≥`·이상·이하·초과·미만 = 경계값(표기 `미만`·`초과`·`이하`·`이상`), 숫자 뒤 부등호(`0.98<`)는 방향을 뒤집어 읽음, '
  '`-` 처럼 숫자로 읽히지 않는 글 = 값 없음. 음수로 읽히는 원문도 값 없음(aw·pH·염도는 음수가 될 수 없음 — 사용자 2026-10-27, 영양강화 밀가루 aw `-0.85`). '
  '범위 가운데 값은 소수 12자리로 맞춤(부동소수 오차만 없앰).')
W(f'2. **aw 는 `aw_source` = `original` 행만**: `ba_data.t_aw_cl` {N_AW:,}행 중 {len(NON_ORIG)}행 제외(아래 3절). '
  'ph·na 는 출처 칸이 없어 `aw_source` 와 관계없이 표 값 그대로 씀.')
W('3. **하위 코드**: `ba_data.t_prdlst_cl` 의 `hrnk_prdlst_cd`(상위_품목_코드)를 따라 내려가 그 코드 아래에 있는 모든 코드(자기 자신 제외). '
  '`lv` 무관, `vald_end_dt` 가 지난 코드도 포함.')
W('4. **대표값**(aw·ph·na 마다 따로):')
W('   - ① `자기값`: 3단계 코드(`train_prdlst_code_2`) 자체의 `t_aw_cl` 값')
W('   - ② `하위중앙`: ①이 없으면, 하위 코드 중 값이 있는 코드들의 값의 중앙값(코드마다 한 번 — `t_aw_cl` 은 코드당 1행). 개수가 짝수면 가운데 두 값의 평균. 소수 12자리로 맞춤.')
W('   - ③ `2단계값`: ②도 없으면, 그 3단계와 시료에서 짝지어진 2단계 코드(`train_prdlst_code`) 자체의 `t_aw_cl` 값(2단계의 하위 코드는 보지 않음). '
  f'한 3단계가 2단계 여럿과 짝이면 시료 수 최다 짝 — 이번 시료에서는 그런 3단계 **{len(MULTI2)}개**.')
W('   - ④ `결측`: 셋 다 없으면 null.')
W('5. **표기 칸 `aw_label`·`ph_label`·`na_label`**: ①·③ 은 쓴 값의 표기(`단일값`·`중앙`·`이상`·`이하`·`초과`·`미만`), ② 는 새 범주 `하위중앙`, '
  '④ 는 null(3차 표와 같이 null 로 저장 — 모델에 넣을 때 `결측` 범주로 바꾸는 것도 3차와 같음). '
  '`하위중앙` 은 중앙값을 낸 하위 코드 값들의 표기가 섞여 있어도(부등호·범위 포함) 숫자만으로 중앙값을 냄.')
W('6. **덧붙인 칸**: `{aw,ph,na}_rule` = 위 ①~④ 중 쓴 규칙(`자기값`·`하위중앙`·`2단계값`·`결측`), `{aw,ph,na}_n_sub` = `하위중앙` 일 때 중앙값에 쓴 하위 코드 수(그 밖은 null). '
  '앞 9칸은 3차 표와 칸 이름·자료형·순서가 같아 바꿔 끼울 수 있음.')
W('7. **생장 범위 표**: `make_ranges.py` 3) 과 같은 계산(그림 값 aw 한계 0.90~0.997·최적 0.97, pH 한계 4.0~9.6·최적 7.0, 경계 포함, 결측 → null, float32). '
  '표기가 중앙·부등호·하위중앙이어도 숫자 그대로 씀. 칸·자료형·메타데이터 형식은 3차와 같고 `inputs`·`script`·`description` 만 바꿈.')
W('8. 값은 원문 그대로: 결측 채움·단위 변환·이상값 삭제 없음.')
W('')
W('## 2. 단계별 건수 (3단계 코드 217개 기준, 괄호 = 그 3단계의 시료 행 / 검출)')
W('')
W('| 항목 | ① 자기값 | ② 하위중앙 | ③ 2단계값 | ④ 결측 |')
W('|---|---:|---:|---:|---:|')
STAGE_CNT = {}
for c in FOOD_COLS:
    cells = []
    for rl in RULES:
        ks = [i for i, k in enumerate(CODES3) if REP[c][k]['rule'] == rl]
        STAGE_CNT[(c, rl)] = len(ks)
        cells.append(f'{len(ks)} ({int(ROW3[ks].sum()):,} / {int(DET3[ks].sum()):,})')
    W(f'| {FOOD_KO[c]} {c} | ' + ' | '.join(cells) + ' |')
W('')
W('하위중앙에 쓴 하위 코드 수(3단계별):')
W('')
for c in FOOD_COLS:
    ns = [REP[c][k]['n_sub'] for k in CODES3 if REP[c][k]['rule'] == '하위중앙']
    if ns:
        W(f'- {FOOD_KO[c]}: {len(ns)}개 3단계 · 하위 코드 수 중앙값 {np.median(ns):g} · 최소 {min(ns)} · 최대 {max(ns)} · 1개뿐인 3단계 {sum(n == 1 for n in ns)}개')
    else:
        W(f'- {FOOD_KO[c]}: 없음')
W('')
W('표기(`{c}_label`)별 3단계 수:')
W('')
for c in FOOD_COLS:
    cnt = Counter(REP[c][k]['label'] or '(null=결측)' for k in CODES3)
    W(f'- {FOOD_KO[c]}: ' + ' · '.join(f'{k} {v}' for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))
W('')
W('## 3. `t_aw_cl` 에서 뺀 행·원문 해석')
W('')
W(f'- `aw_source` ≠ `original` 인 행 **{len(NON_ORIG)}행** — aw 를 쓰지 않음. 그중 aw 원문이 숫자로 읽히는 행 '
  f'{sum(not math.isnan(parse_db(DB[x]["aw"])[0]) for x in NON_ORIG)}행(나머지는 aw 빈칸).')
grp = Counter()
for s, n in NON_ORIG_SRC.items():
    key = s.split('(')[0].split(':')[0].split(' ')[0]
    grp[key] += n
W('  - 종류별 행 수: ' + ' · '.join(f'`{k}` {v}' for k, v in sorted(grp.items(), key=lambda kv: -kv[1])))
W(f'- 같은 {len(NON_ORIG)}행 중 ph 값이 있는 행 {len(NON_ORIG_PH)}행({", ".join(NON_ORIG_PH) or "-"}), na 값이 있는 행 {len(NON_ORIG_NA)}행'
  f'({", ".join(NON_ORIG_NA) or "-"}) — 규칙 2 대로 ph·na 는 씀.')
used_nonorig = {c: [k for k in CODES3 if any(code in NON_ORIG for code in ([k] if REP[c][k]['rule'] == '자기값' else
                                                                               [K2_OF[k]] if REP[c][k]['rule'] == '2단계값' else
                                                                               [d for d in DESC[k] if d in VAL[c]] if REP[c][k]['rule'] == '하위중앙' else []))]
                for c in ('ph', 'na')}
W(f'  - 그 행 값이 대표값에 들어간 3단계 수: ph {len(used_nonorig["ph"])}개, na {len(used_nonorig["na"])}개')
for c in FOOD_COLS:
    pr = PARSE_RULES[c]
    W(f'- {FOOD_KO[c]} 원문 해석(행 수): ' + ' · '.join(f'{k} {v}' for k, v in sorted(pr.items())) +
      (f' — 해석 못 한 원문: ' + ', '.join(f'`{k}` {v}' for k, v in UNPARSED[c].most_common()) if UNPARSED[c] else ''))
W(f'- 값이 있는 코드 수(규칙 1·2 적용 뒤): aw {len(VAL["aw"]):,} · ph {len(VAL["ph"]):,} · na {len(VAL["na"]):,}')
W(f'- `t_aw_cl` 코드 중 분류표에 없는 코드 {len(AW_NOT_IN_CL)}개' + (f' ({", ".join(AW_NOT_IN_CL[:10])}{" …" if len(AW_NOT_IN_CL) > 10 else ""}) — 하위 코드로 잡히지 않음(자기값·2단계값으로는 쓰임)' if AW_NOT_IN_CL else ''))
neg = [(k, c) for c in FOOD_COLS for k in CODES3 if REP[c][k]['num'] is not None and REP[c][k]['num'] < 0]
W(f'- **음수 대표값 {len(neg)}건**' + (': ' + '; '.join(
    f'{k} {NM2[k]} {c} = {num_txt(REP[c][k]["num"])} ({REP[c][k]["rule"]}, ' +
    ', '.join(f'{d} "{VAL[c][d][2]}"' for d in DESC[k] if d in VAL[c]) + ')' for k, c in neg) +
    ' — 원문 `-0.85` 를 3차와 같은 규칙(맨 앞 `-` = 음수 부호)으로 −0.85 로 읽어 중앙값에 들어감. 오기(`~0.85` 등)일 수 있으나 '
    '규칙 8(원문 그대로)에 따라 고치지 않음 — 확인 필요' if neg else ''))
big = [k for k in CODES3 if REP['na'][k]['num'] is not None and REP['na'][k]['num'] > 20]
W(f'- 염도 대표값이 20 보다 큰 3단계 {len(big)}개' + (': ' + ', '.join(f'{k} {NM2[k]} {num_txt(REP["na"][k]["num"])} ({REP["na"][k]["rule"]})' for k in big) if big else ''))
c839 = [k for k in CODES3 if REP['aw'][k]['num'] is not None and abs(REP['aw'][k]['num'] - 0.839) < 1e-12]
W(f'- aw 대표값이 0.839 인 3단계 {len(c839)}개(3차에서 출처 확인 필요로 표시한 값 — `t_aw_cl` 원문 \'0.839\', aw_source original){": " + ", ".join(c839) if c839 else ""}')
W('')
W('## 4. 결측률')
W('')
W('기준: 3단계 코드 = 217개 중 결측 코드 수 · 시료 = `시료_시군구.parquet` 전체 행(3단계 코드가 빈 130행은 붙일 수 없어 결측으로 셈) · 검출 시료 = 그중 검출 행. 괄호 안은 3차 표.')
W('')
W('| 항목 | 3단계 코드 | 시료 | 검출 시료 |')
W('|---|---:|---:|---:|')
MISS = {}
for c in FOOD_COLS:
    newmiss3 = {k: REP[c][k]['num'] is None for k in CODES3}
    oldmiss3 = {k: OLD[k][c][0] is None for k in CODES3}
    nm = np.array([newmiss3[k] if k is not None else True for k in tp2])
    om = np.array([oldmiss3[k] if k is not None else True for k in tp2])
    a3, o3 = sum(newmiss3.values()), sum(oldmiss3.values())
    MISS[c] = dict(n3=a3, rows=int(nm.sum()), det=int((nm & det).sum()))
    W(f'| {FOOD_KO[c]} {c} | {a3}/217 = {pct(a3, 217)}% ({o3} = {pct(o3, 217)}%) | '
      f'{int(nm.sum()):,}/{N:,} = {pct(int(nm.sum()), N)}% ({pct(int(om.sum()), N)}%) | '
      f'{int((nm & det).sum()):,}/{NDET:,} = {pct(int((nm & det).sum()), NDET)}% ({pct(int((om & det).sum()), NDET)}%) |')
W('')
W('## 5. 3차 표와 비교')
W('')
W('3단계 217개 중 값(숫자) 또는 표기가 바뀐 코드 수:')
W('')
W('| 항목 | 같음 | 숫자 바뀜 | 숫자 같고 표기만 바뀜 | 3차 결측 → 새 값 | 3차 값 → 새 결측 | 바뀐 코드 합계 (시료 행 / 검출) |')
W('|---|---:|---:|---:|---:|---:|---:|')
CHG = {}
for c in FOOD_COLS:
    k = Counter()
    chg_codes = []
    for i, code in enumerate(CODES3):
        on, ol = OLD[code][c]
        nn, nl = REP[c][code]['num'], REP[c][code]['label']
        if on is None and nn is None:
            k['same'] += 1
        elif on is None:
            k['fill'] += 1; chg_codes.append(i)
        elif nn is None:
            k['lose'] += 1; chg_codes.append(i)
        elif not same_num(on, nn):
            k['num'] += 1; chg_codes.append(i)
        elif ol != nl:
            k['lab'] += 1; chg_codes.append(i)
        else:
            k['same'] += 1
    CHG[c] = (k, chg_codes)
    W(f'| {FOOD_KO[c]} {c} | {k["same"]} | {k["num"]} | {k["lab"]} | {k["fill"]} | {k["lose"]} | '
      f'{len(chg_codes)} ({int(ROW3[chg_codes].sum()):,} / {int(DET3[chg_codes].sum()):,}) |')
W('')
W('"숫자 같고 표기만 바뀜" 내역(3차 표기 → 새 표기: 코드 수):')
W('')
for c in FOOD_COLS:
    lk = Counter((OLD[k][c][1], REP[c][k]['label']) for k in CODES3
                 if OLD[k][c][0] is not None and REP[c][k]['num'] is not None and same_num(OLD[k][c][0], REP[c][k]['num'])
                 and OLD[k][c][1] != REP[c][k]['label'])
    W(f'- {FOOD_KO[c]}: ' + (' · '.join(f'{a} → {b}: {n}' for (a, b), n in lk.most_common()) or '없음'))
W('')
W('생장 범위 칸이 3차와 달라진 3단계 수(결측 여부 포함):')
W('')
nrg = out_rg.to_pydict()
for col in RG:
    a, b = OLD_RG_D[col], nrg[col]
    assert OLD_RG_D['train_prdlst_code_2'] == nrg['train_prdlst_code_2']
    d = sum(1 for x, y in zip(a, b) if not ((x is None and y is None) or (x is not None and y is not None and abs(x - y) < 1e-6)))
    W(f'- `{col}`: {d}개')
W('')
W('### 검출이 많은 3단계 상위 15개 — 3차 값 → 새 값(규칙)')
W('')
W('표기 줄임: 단=단일값, 중=중앙(범위 가운데), 하위=하위중앙, 결=결측. 부등호 표기(이상·이하·초과·미만)는 그대로.')
W('')
W('| 3단계 코드 | 이름 | 2단계 | 시료 / 검출 | aw 3차 → 새 | ph 3차 → 새 | na 3차 → 새 |')
W('|---|---|---|---:|---|---|---|')
AB = {'단일값': '단', '중앙': '중', '하위중앙': '하위', None: '결'}


def vtxt(x, lab):
    if x is None:
        return '결측'
    return f'{num_txt(x)}({AB.get(lab, lab)})'


top15 = sorted(CODES3, key=lambda k: (-N3_DET[k], -N3_ROWS[k], k))[:15]
for k in top15:
    cells = []
    for c in FOOD_COLS:
        on, ol = OLD[k][c]
        r = REP[c][k]
        mark = '' if (same_num(on, r['num']) and (ol == r['label'])) else ' ◀'
        cells.append(f'{vtxt(on, ol)} → {vtxt(r["num"], r["label"])} [{r["rule"]}]{mark}')
    W(f'| {k} | {NM2[k]} | {K2_OF[k]} {NM1.get(K2_OF[k], "")} | {N3_ROWS[k]:,} / {N3_DET[k]:,} | ' + ' | '.join(cells) + ' |')
W('')
W('◀ = 숫자나 표기가 3차와 다름. 하위중앙의 하위 코드 수는 parquet `*_n_sub` 칸.')
W('')
W('### 부등호 표기 품목 (make_ranges.py 의 생장범위_부등호표기.csv 대신)')
W('')
sign = [(k, c) for c in ('aw', 'ph') for k in CODES3 if REP[c][k]['label'] in ('초과', '이상', '미만', '이하')]
if sign:
    W('| 3단계 코드 | 이름 | 항목 | 값 | 표기 | 규칙 |')
    W('|---|---|---|---:|---|---|')
    for k, c in sign:
        W(f'| {k} | {NM2[k]} | {c} | {num_txt(REP[c][k]["num"])} | {REP[c][k]["label"]} | {REP[c][k]["rule"]} |')
else:
    W('없음')
W('')
W('## 6. 확인 못 한 점·한계')
W('')
W(f'- 시료의 3단계·2단계 칸이 분류표 `lv` 와 맞지 않음(3단계 칸 217종 중 lv 2·3·4 가 섞임 — 출처 md 실측). 그래서 "3단계 자체 값"·"하위 코드"·"2단계 값"은 분류표의 lv 가 아니라 시료 칸에 적힌 코드 기준임.')
same23 = [k for k in CODES3 if K2_OF[k] == k]
W(f'- 3단계 코드와 2단계 코드가 같은 3단계 {len(same23)}개({", ".join(same23)}) — ③ 2단계값은 ① 과 같은 값이라 ① 이 없으면 ③ 도 없음.')
W(f'- 시료 2단계 코드 중 분류표에 없는 코드: {len(NOT_IN_CL2)}개{(" " + ", ".join(NOT_IN_CL2)) if NOT_IN_CL2 else ""}.')
expired = [k for k in CODES3 if CL_END.get(k) not in (None, '99991231')]
W(f'- 유효 종료일이 지난 3단계 코드 {len(expired)}개도 그대로 표에 넣음(시료에 나오므로). 하위 코드도 유효 종료 포함.')
W('- 염도(na) 단위를 문서로 확인하지 못함(3차와 같음, DB 코멘트 "염도"만). 원문에 `95.5~96.6` 처럼 다른 단위로 보이는 값이 섞여 있으나 그대로 둠. 생장 범위 표에 염도 칸은 3차처럼 만들지 않음.')
W('- ph·na 는 `t_aw_cl` 에 출처 칸이 없어 원측정값인지 가공값인지 구분하지 못함(aw_source 는 aw 에 대한 칸으로 보고 ph·na 에는 적용하지 않음 — 사용자 규칙).')
W('- 하위중앙은 하위 코드 수와 무관하게 같은 무게로 셈(하위 코드 1개뿐인 경우도 그 값 하나가 중앙값). 하위 코드끼리 단계(lv)가 달라도 같이 셈.')
W('- 하위 코드 값의 표기(부등호·범위)는 중앙값을 낼 때 숫자만 씀 — 예: `>0.98` 은 0.98 로 셈.')
W('- 0.839(2단계 9개 코드에 똑같은 aw 원문)의 출처는 3차 때처럼 확인 못 함 — 규칙대로 씀.')

OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
log(f'저장: {OUT_MD.name}')
print('\n'.join(lines))
log('끝')
