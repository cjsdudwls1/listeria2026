# -*- coding: utf-8 -*-
"""
06_품목_식품속성.py — 품목군 3단계(train_prdlst_code_2)마다 식품 속성(수분활성도 aw · 식품pH ph · 염도 na) 값 하나인 표

왜 3단계 표인가 (사용자 결정 2026-10-16, 세션 '[식품데이터-품목 별 시료들의 최빈값으로 조인. 모델 입력 시 결측행 그대로(품목군의 성질로 보고)]')
  예측 단위가 시군구 × 품목군 2단계(train_prdlst_code) × 3단계(train_prdlst_code_2)이므로, 식품 값은 3단계 코드 하나당 값 하나로 정해
  3단계 코드로 붙인다. 시료 단위 값(시료 자기 품목코드 값)은 대표값을 계산할 때만 쓰고 모델에는 넣지 않는다.

입력 (읽기만 함)
  - ba_data.t_aw_cl 원자료 : 리스테리아ML2/data/raw/raw_t_aw_cl.parquet (01_원자료_받기.py raw_t_aw_cl 이 받은 SELECT * 그대로)
        prdlst_cd(품목코드), kor_nm, lv, aw(DB 코멘트 "수분활성도"), ph("식품pH"), na("염도"), aw_source(코멘트 없음)
  - 리서치팀 엑셀          : 문서/리서치팀문서/통합_수분활성도_수분함량_식품pH_염도_최종.xlsx 의 '값' 시트
        PRDLST_CD · KOR_NM · LV · 수분활성도 · 수분함량 · 식품 pH · 염도 (수분함량은 쓰지 않음)
  - 시료                   : 리스테리아ML2/data/데이터셋/시료_시군구.parquet (국내, 시군구를 정한 행만 — 628,517행)
        prdlst_code · train_prdlst_code · train_prdlst_nm · train_prdlst_code_2 · train_prdlst_nm_2 · mnfctur_de · judge_result

출력
  - 리스테리아ML2/data/데이터셋/식정원_db_식품/품목3단계_식품속성.parquet  (모델 입력용)
        train_prdlst_code_2, train_prdlst_nm_2, train_prdlst_code, aw, aw_label, ph, ph_label, na, na_label
        3단계 코드 1개당 1행(시료_시군구.parquet 에 나오는 3단계 코드 전부), train_prdlst_code_2 순 정렬.
        숫자는 float64(원문을 숫자로 바꾼 값 그대로 — float32 로 줄이지 않음), 표기는 string. 결측은 null(채우지 않음).
        3단계 코드가 빈 시료 행(130행)은 이 표에 없다 — 붙이면 결측.
  - 리스테리아ML2/data/중간/품목3단계_식품속성_근거.csv  (보고서용 점검 표, 3단계 × 항목 1줄)

규칙
  1) 원문 → (숫자, 표기) — 예전 구축 노트북(final_기상조인코드_사본.ipynb) 6절 parse_food 그대로:
     숫자는 그대로(단일값), 범위(~ ～ ∼ - –)는 가운데 값(중앙), 부등호 < > ≤ ≥ 와 '이상·이하·초과·미만'은 경계값(미만·초과·이하·이상),
     숫자 뒤 부등호(0.98<)는 방향을 뒤집어 읽음(0.98 초과), 숫자 맨 앞 '-' 는 음수 부호, 그 밖('-' 등)은 값 없음.
     (노트북과 다른 점 하나: 범위 가운데 값의 부동소수 계산 오차를 소수 12자리로 없앤다 — (0.94+0.97)/2 = 0.9550000000000001 → 0.955.
      같은 가운데 값이 오차 때문에 서로 다른 상태로 세지지 않게 하려는 것. 단일값·경계값은 바뀌지 않음을 확인한다)
  2) 두 자료 합치기 — 노트북 6절 그대로 '엑셀 우선': 품목코드·항목마다 엑셀 값이 숫자로 읽히면 엑셀 값, 아니면 t_aw_cl 값.
  3) 시료 단위 조인(계산용) — 시료마다 자기 prdlst_code 로: 정확 일치 → (20자리 코드) 앞 14자리. 노트북 6절의 앞 두 키만 쓴다.
     이 단계에서는 2·3단계 코드로 물려받지 않는다(사용자 결정 2026-10-16).
  4) 3단계 대표값 — 항목마다 따로. train_prdlst_code_2 마다 학습 기간(mnfctur_de 의 날짜 ≤ 2024-06-30) 시료의 '상태' 최빈값.
     상태 = (숫자, 표기) 짝. 원자료에 그 품목코드 값이 없거나 '-' 등이면 '값 없음' 상태 하나.
       - 최빈 상태가 하나이고 값이면 → 그 숫자·표기가 대표값 (단계 1)
       - 최빈 상태가 '값 없음', 최빈 동률(값 vs 값 없음, 값 vs 값), 학습 기간 시료 없음 → 물려받기:
         원자료에 그 3단계 코드로 적힌 값(단계 2) → 원자료에 그 2단계 코드로 적힌 값(단계 3) → 결측(단계 4)
         ('원자료 값'은 2)의 엑셀 우선으로 합친 품목코드 값)
     2단계 값 0.839(t_aw_cl 의 수분활성도 원문 '0.839', 2단계 품목군 9개에 똑같이 있음, aw_source 'original')는 지우지 않고 쓰되
     근거 표에 '출처확인필요_0839' 로 표시한다(값을 준 품목코드가 이 9개 중 하나이면 '예').
  값은 원자료 그대로: 결측 채움·단위 변환·이상값 삭제 없음(수분활성도 원문 '-0.85' 는 규칙대로 −0.85).

모델에 넣을 때(참고, 이 스크립트는 하지 않음): aw·ph·na 숫자는 결측 그대로, aw_label 은 범주(결측은 '결측' 범주). ph_label·na_label 은 넣지 않음.

다시 돌리면 같은 결과가 나온다(무작위 없음, 동률 판정·정렬 순서 고정).
실행: py "리스테리아ML2\\notebook\\06_품목_식품속성.py"   (py 3.14 — numpy·pyarrow·openpyxl 사용, pandas 쓰지 않음)
"""
import csv
import datetime as dt
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

T0 = time.time()
BASE = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2')
RAW_AW = BASE / 'data' / 'raw' / 'raw_t_aw_cl.parquet'
XLSX = Path(r'C:\Users\NT751\Desktop\천영진\문서\리서치팀문서\통합_수분활성도_수분함량_식품pH_염도_최종.xlsx')
XLSX_SHEET = '값'
SAMPLE_SGG = BASE / 'data' / '데이터셋' / '시료_시군구.parquet'
OUT_PATH = BASE / 'data' / '데이터셋' / '식정원_db_식품' / '품목3단계_식품속성.parquet'
CHECK_PATH = BASE / 'data' / '중간' / '품목3단계_식품속성_근거.csv'

TRAIN_END = dt.date(2024, 6, 30)       # 학습 기간: mnfctur_de 의 날짜 ≤ 2024-06-30
FOOD_COLS = ['aw', 'ph', 'na']
FOOD_KO = {'aw': '수분활성도', 'ph': '식품pH', 'na': '염도'}            # DB 칼럼 코멘트 그대로
XL_COL = {'aw': '수분활성도', 'ph': '식품 pH', 'na': '염도'}            # 엑셀 '값' 시트 머리글
FOOD_LABELS = ['단일값', '중앙', '이상', '이하', '초과', '미만']
AW839_RAW = '0.839'                    # 출처 확인이 필요한 2단계 수분활성도 원문 (t_aw_cl)
STAGE_TXT = {1: '3단계 시료 최빈값', 2: '원자료 3단계 코드 값', 3: '원자료 2단계 코드 값', 4: '결측'}


def log(*a):
    print(f'[{time.time() - T0:5.1f}초]', *a, flush=True)


# ---------------------------------------------------------------- 1) 원문 → (숫자, 적용 규칙, 표기) — 노트북 6절 parse_food 그대로
NUM = r'-?\d+(?:\.\d+)?'
RANGE_SEP = r'[~～∼\-–]'
SIGN_LABEL = {'<': '미만', '>': '초과', '≤': '이하', '≥': '이상', '<=': '이하', '>=': '이상'}
SIGN_FLIP = {'미만': '초과', '초과': '미만', '이하': '이상', '이상': '이하'}   # 부등호가 숫자 뒤에 올 때 (예: 0.98< = 0.98 초과)


def parse_food(s):
    """원문 → (숫자, 적용 규칙, 값 표기). 값 표기: 단일값 · 중앙 · 이상 · 이하 · 초과 · 미만. 값이 없으면 숫자 nan, 표기 ''"""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return math.nan, '빈칸', ''
    if isinstance(s, (int, float, np.integer, np.floating)) and not isinstance(s, bool):
        return float(s), '숫자 그대로', '단일값'       # 엑셀의 숫자 칸
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


NUM_ROUND = 12      # 범위 가운데 값의 부동소수 오차만 없애는 자릿수 (예: (0.94+0.97)/2 = 0.9550000000000001 → 0.955)


def parse_food_n(s):
    """parse_food 결과의 숫자를 소수 12자리로 맞춘다 — 같은 가운데 값이 계산 오차 때문에 서로 다른 상태로 세지지 않게.
    범위 가운데 값에서만 숫자가 바뀌고(1e-12 보다 작은 오차), 단일값·경계값은 바뀌지 않아야 한다(확인)."""
    x, rule, label = parse_food(s)
    if math.isnan(x):
        return x, rule, label
    y = round(x, NUM_ROUND)
    assert y == x or rule == '범위 → 가운데 값', (s, x, y)
    assert abs(y - x) < 1e-12
    return y, rule, label


def raw_text(s):
    """원문을 보여 줄 때 쓰는 글자 (빈칸은 None)"""
    return None if s is None or (isinstance(s, float) and math.isnan(s)) or str(s).strip() == '' else str(s).strip()


def num_txt(x):
    return f'{x:.10g}'


def state_txt(st):
    return '값 없음' if st is None else f'{num_txt(st[0])} ({st[1]})'


# ---------------------------------------------------------------- 원자료 두 개 읽기
aw_t = pq.read_table(RAW_AW)
assert aw_t.schema.metadata[b'source_table'] == b'ba_data.t_aw_cl'
AWD = aw_t.to_pydict()
DB_CODES = AWD['prdlst_cd']
assert None not in DB_CODES and len(set(DB_CODES)) == len(DB_CODES), 't_aw_cl prdlst_cd 가 비었거나 겹침'
DB = {code: {k: AWD[k][i] for k in AWD} for i, code in enumerate(DB_CODES)}

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ws = wb[XLSX_SHEET]
xrows = list(ws.iter_rows(values_only=True))
wb.close()
XH = list(xrows[0])
assert XH == ['PRDLST_CD', 'KOR_NM', 'LV', '수분활성도', '수분함량', '식품 pH', '염도'], XH
XROWS = [r for r in xrows[1:] if not all(v is None for v in r)]     # 엑셀 끝의 완전히 빈 줄만 뺌 (pandas read_excel 과 같게)
XL = {}
for r in XROWS:
    code = r[0]
    assert isinstance(code, str) and code.strip() == code and code not in XL, f'엑셀 PRDLST_CD 이상: {code!r}'
    XL[code] = dict(zip(XH, r))
NOT_A_CODE = {'', 'nan', 'NaN', 'None', '<NA>'}
assert not (set(DB) | set(XL)) & NOT_A_CODE
log(f'ba_data.t_aw_cl 품목코드 {len(DB):,}개 · 엑셀 값 시트 품목코드 {len(XL):,}개 '
    f'(엑셀 코드 중 t_aw_cl 에도 있는 코드 {sum(c in DB for c in XL):,}개, 엑셀에만 {sum(c not in DB for c in XL)}개)')

# ---------------------------------------------------------------- 2) 품목코드별 값 (엑셀 우선)
ALL_CODES = sorted(set(DB) | set(XL))
PARSED = {c: {} for c in FOOD_COLS}         # 항목 → 품목코드 → {'db': (숫자, 규칙, 표기, 원문), 'xl': (...)}
VAL = {c: {} for c in FOOD_COLS}            # 항목 → 품목코드 → (숫자, 표기, 자료, 원문, 규칙) — 값이 있는 코드만
PARSE_RULES = Counter()
UNPARSED = Counter()
for c in FOOD_COLS:
    for code in ALL_CODES:
        db_raw = DB[code][c] if code in DB else None
        xl_raw = XL[code][XL_COL[c]] if code in XL else None
        pd_ = parse_food_n(db_raw) + (raw_text(db_raw),)
        px_ = parse_food_n(xl_raw) + (raw_text(xl_raw),)
        if code in DB:
            PARSE_RULES[(c, 't_aw_cl', pd_[1])] += 1
            if pd_[1] == '해석 못 함 → 결측':
                UNPARSED[(c, 't_aw_cl', pd_[3])] += 1
        if code in XL:
            PARSE_RULES[(c, '엑셀', px_[1])] += 1
            if px_[1] == '해석 못 함 → 결측':
                UNPARSED[(c, '엑셀', px_[3])] += 1
        PARSED[c][code] = {'db': pd_, 'xl': px_}
        if not math.isnan(px_[0]):          # 엑셀에 값이 있으면 엑셀
            VAL[c][code] = (px_[0], px_[2], '엑셀', px_[3], px_[1])
        elif not math.isnan(pd_[0]):        # 없으면 t_aw_cl
            VAL[c][code] = (pd_[0], pd_[2], 't_aw_cl', pd_[3], pd_[1])
        if code in VAL[c]:
            assert VAL[c][code][1] in FOOD_LABELS

# 같은 숫자를 같은 상태로 셀 수 있는지: 서로 다른 숫자인데 거의 같은(부동소수 오차 수준) 짝이 없어야 한다
for c in FOOD_COLS:
    nums = sorted({v[0] for v in VAL[c].values()})
    close = [(a, b) for a, b in zip(nums, nums[1:]) if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)]
    assert not close, f'{c}: 부동소수 오차로만 다른 숫자 {close}'

CMP = {}
for c in FOOD_COLS:
    k = Counter()
    for code in ALL_CODES:
        d_, x_ = PARSED[c][code]['db'], PARSED[c][code]['xl']
        dn, xn = not math.isnan(d_[0]), not math.isnan(x_[0])
        if dn and xn:
            k['(가) 숫자가 다름' if not math.isclose(d_[0], x_[0], rel_tol=1e-9, abs_tol=1e-12)
              else ('(나) 숫자 같고 표기가 다름' if d_[2] != x_[2] else '같음')] += 1
        elif dn:
            k['(다) t_aw_cl 에만 값'] += 1
        elif xn:
            k['(다) 엑셀에만 값'] += 1
        else:
            k['둘 다 값 없음'] += 1
    CMP[c] = k
log('원문 → 숫자 규칙별 품목코드 수:', dict(sorted(PARSE_RULES.items())))
log('해석 못 해 값 없음으로 둔 원문:', dict(UNPARSED))
log('두 자료 비교(품목코드 수):', {FOOD_KO[c]: dict(CMP[c]) for c in FOOD_COLS})
log('값이 있는 품목코드 수(엑셀 우선):', {FOOD_KO[c]: len(VAL[c]) for c in FOOD_COLS},
    '그중 엑셀 값:', {FOOD_KO[c]: sum(v[2] == '엑셀' for v in VAL[c].values()) for c in FOOD_COLS})

# 0.839 — t_aw_cl 수분활성도 원문이 '0.839' 인 코드
C839 = sorted(code for code in DB if DB[code]['aw'] == AW839_RAW)
assert len(C839) == 9, C839
assert all(DB[code]['lv'] == 3.0 and DB[code]['aw_source'] == 'original' for code in C839)
assert all(VAL['aw'][code][2] == 't_aw_cl' and VAL['aw'][code][0] == 0.839 for code in C839), '0.839 코드에 엑셀 값이 있음'
n839_other = sum(1 for code, v in VAL['aw'].items() if v[0] == 0.839 and code not in C839)
log(f"t_aw_cl 수분활성도 원문 '0.839' 코드 {len(C839)}개 (lv 3, aw_source 'original', 엑셀 값 없음) · 그 밖에 숫자 0.839 인 코드 {n839_other}개")

# ---------------------------------------------------------------- 시료 읽기
smp = pq.read_table(SAMPLE_SGG, columns=['lims_data_key', 'prdlst_code', 'train_prdlst_code', 'train_prdlst_nm',
                                         'train_prdlst_code_2', 'train_prdlst_nm_2', 'mnfctur_de', 'judge_result'])
N = smp.num_rows
S = {k: pc.cast(smp[k], pa.string()).to_pylist() for k in
     ['prdlst_code', 'train_prdlst_code', 'train_prdlst_nm', 'train_prdlst_code_2', 'train_prdlst_nm_2', 'judge_result']}
assert len(set(smp['lims_data_key'].to_pylist())) == N
assert None not in S['prdlst_code'] and not set(S['prdlst_code']) & NOT_A_CODE
for k in ('train_prdlst_code', 'train_prdlst_code_2'):
    assert not {x for x in S[k] if x is not None} & NOT_A_CODE
det = np.array([j == '검출' for j in S['judge_result']])
assert set(S['judge_result']) <= {'검출', '불검출'}
mn = smp['mnfctur_de']
assert mn.null_count == 0
mn_day = pc.cast(pc.floor_temporal(mn, unit='day'), pa.date32()).to_pylist()
train = np.array([d <= TRAIN_END for d in mn_day])
z_row = np.array([p.startswith('Z') for p in S['prdlst_code']])
tp2 = S['train_prdlst_code_2']
has3 = np.array([x is not None for x in tp2])
log(f'시료 {N:,}행 (검출 {int(det.sum()):,}) · 학습 기간 {int(train.sum()):,}행 (검출 {int((train & det).sum()):,}) · '
    f'3단계 코드 빈 행 {int((~has3).sum())} (검출 {int((~has3 & det).sum())}) · Z 행 {int(z_row.sum()):,} (검출 {int((z_row & det).sum()):,})')
log('prdlst_code 길이별 행 수:', dict(Counter(len(p) for p in S['prdlst_code'])))

# 3단계 → 이름·2단계 (1:1 이어야 함)
INFO3 = {}
for i in range(N):
    k3 = tp2[i]
    if k3 is None:
        assert S['train_prdlst_code'][i] is None
        continue
    v = (S['train_prdlst_nm_2'][i], S['train_prdlst_code'][i], S['train_prdlst_nm'][i])
    assert INFO3.setdefault(k3, v) == v, f'3단계 {k3} 에 이름·2단계가 여럿'
CODES3 = sorted(INFO3)
assert all(v[1] is not None for v in INFO3.values())
log(f'3단계 코드 {len(CODES3)}개 (학습 기간 시료가 있는 코드 {len({tp2[i] for i in np.flatnonzero(train & has3)})}개)')

# ---------------------------------------------------------------- 3) 시료 단위 조인 (계산용): prdlst_code 정확 일치 → 앞 14자리
P_CODES = sorted(set(S['prdlst_code']))
HIT = {c: {} for c in FOOD_COLS}            # 항목 → 시료 prdlst_code → (값을 준 품목코드, 어느 키) 또는 None
for c in FOOD_COLS:
    for p in P_CODES:
        if p in VAL[c]:
            HIT[c][p] = (p, 'prdlst_code')
        elif p[:14] in VAL[c]:
            HIT[c][p] = (p[:14], 'prdlst_code 앞 14자리')
        else:
            HIT[c][p] = None


def own_state(c, p):
    h = HIT[c][p]
    return None if h is None else VAL[c][h[0]][:2]


KEY_ROWS = {c: Counter() for c in FOOD_COLS}
for c in FOOD_COLS:
    for p, n in Counter(S['prdlst_code']).items():
        h = HIT[c][p]
        KEY_ROWS[c]['값 없음' if h is None else h[1]] += n
log('시료 단위 조인 — 키별 시료 행 수:', {FOOD_KO[c]: dict(KEY_ROWS[c]) for c in FOOD_COLS})

# ---------------------------------------------------------------- 4) 3단계 대표값
idx3 = defaultdict(list)
for i in range(N):
    if tp2[i] is not None:
        idx3[tp2[i]].append(i)
P_ARR = S['prdlst_code']
REP = {c: {} for c in FOOD_COLS}            # 항목 → 3단계 → dict
for c in FOOD_COLS:
    for k3 in CODES3:
        rows = idx3[k3]
        tr_rows = [i for i in rows if train[i]]
        st_cnt = Counter(own_state(c, P_ARR[i]) for i in tr_rows)
        # 상태 순서: 시료 수 많은 순 → (같으면) 값 없음 먼저 → 숫자·표기 순. 동률 판정은 시료 수로만 한다
        ranked = sorted(st_cnt.items(), key=lambda kv: (-kv[1], kv[0] is not None, kv[0] or (0.0, '')))
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        tie = second is not None and second[1] == top[1]
        tied = [s for s, n in ranked if top is not None and n == top[1]]
        if not tr_rows:
            reason = '학습 기간 시료 없음'
        elif tie:
            reason = '최빈 동률(값 vs 값 없음)' if None in tied else '최빈 동률(값 vs 값)'
        elif top[0] is None:
            reason = '최빈 상태가 값 없음'
        else:
            reason = ''
        r = dict(n_train=len(tr_rows), n_train_det=int(det[tr_rows].sum()) if tr_rows else 0, n_states=len(st_cnt),
                 top=top, second=second, tie=tie, reason=reason, n_all=len(rows), n_all_det=int(det[rows].sum()))
        if not reason:                                              # 단계 1
            st = top[0]
            src_codes = Counter(HIT[c][P_ARR[i]][0] for i in tr_rows if own_state(c, P_ARR[i]) == st)
            r.update(stage=1, num=st[0], label=st[1], src_codes=src_codes)
        elif k3 in VAL[c]:                                          # 단계 2: 원자료 3단계 코드 값
            v = VAL[c][k3]
            r.update(stage=2, num=v[0], label=v[1], src_codes=Counter({k3: 0}))
        elif INFO3[k3][1] in VAL[c]:                                # 단계 3: 원자료 2단계 코드 값
            v = VAL[c][INFO3[k3][1]]
            r.update(stage=3, num=v[0], label=v[1], src_codes=Counter({INFO3[k3][1]: 0}))
        else:                                                       # 단계 4: 결측
            r.update(stage=4, num=None, label=None, src_codes=Counter())
        r['flag839'] = c == 'aw' and any(code in C839 and VAL['aw'][code][2] == 't_aw_cl' for code in r['src_codes'])
        if r['flag839']:
            assert r['num'] == 0.839 and r['label'] == '단일값'
        REP[c][k3] = r

# ---------------------------------------------------------------- 결과 파일 (모델 입력용)
cols = {
    'train_prdlst_code_2': pa.array(CODES3, type=pa.string()),
    'train_prdlst_nm_2': pa.array([INFO3[k][0] for k in CODES3], type=pa.string()),
    'train_prdlst_code': pa.array([INFO3[k][1] for k in CODES3], type=pa.string()),
}
for c in FOOD_COLS:
    cols[c] = pa.array([REP[c][k]['num'] for k in CODES3], type=pa.float64())
    cols[f'{c}_label'] = pa.array([REP[c][k]['label'] for k in CODES3], type=pa.string())
OUT = pa.table(cols)
OUT = OUT.replace_schema_metadata({
    'script': '리스테리아ML2/notebook/06_품목_식품속성.py',
    'rule': '3단계(train_prdlst_code_2)마다 학습 기간(mnfctur_de<=2024-06-30) 시료 상태((숫자,표기) 또는 값 없음) 최빈값; '
            '값 없음·동률·학습 기간 시료 없음이면 원자료 3단계 코드 값 -> 2단계 코드 값 -> 결측(null). 원자료=엑셀 우선, 없으면 ba_data.t_aw_cl',
})
assert OUT.num_rows == len(CODES3) and pc.count_distinct(OUT['train_prdlst_code_2']).as_py() == len(CODES3)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(OUT, OUT_PATH)
log(f'결과 저장: {OUT_PATH} ({OUT.num_rows}행 × {OUT.num_columns}칼럼)')

# ---------------------------------------------------------------- 시료 행에 대표값을 붙였을 때 (요약 수치용)
REP_ROW = {}                                # 항목 → 시료 행마다 대표 상태 (None = 결측)
for c in FOOD_COLS:
    lut = {k: (None if REP[c][k]['num'] is None else (REP[c][k]['num'], REP[c][k]['label'])) for k in CODES3}
    REP_ROW[c] = [lut[k] if k is not None else None for k in tp2]


def pct(a, b):
    return round(100.0 * a / b, 3) if b else None


SUMMARY = {}
for c in FOOD_COLS:
    miss = np.array([s is None for s in REP_ROW[c]])
    own = [own_state(c, p) for p in P_ARR]
    own_has = np.array([s is not None for s in own])
    rep_has = ~miss
    diff = np.array([a != b for a, b in zip(own, REP_ROW[c])])
    fill = ~own_has & rep_has                    # 자기 값 없음 → 대표값 있음
    lose = own_has & ~rep_has                    # 자기 값 있음 → 대표값 결측
    chg = own_has & rep_has & diff               # 둘 다 값인데 숫자나 표기가 다름
    st = {s: dict(n3=0, rows=0, det=0) for s in (1, 2, 3, 4)}
    for k in CODES3:
        r = REP[c][k]
        st[r['stage']]['n3'] += 1
        st[r['stage']]['rows'] += r['n_all']
        st[r['stage']]['det'] += r['n_all_det']
    reasons = Counter(REP[c][k]['reason'] or '단계 1' for k in CODES3)
    f839 = [k for k in CODES3 if REP[c][k]['flag839']]
    SUMMARY[c] = dict(
        stages=st, reasons=reasons,
        miss_all=(int(miss.sum()), N, pct(int(miss.sum()), N)),
        miss_nonz=(int((miss & ~z_row).sum()), int((~z_row).sum()), pct(int((miss & ~z_row).sum()), int((~z_row).sum()))),
        miss_z=(int((miss & z_row).sum()), int(z_row.sum()), pct(int((miss & z_row).sum()), int(z_row.sum()))),
        miss_det=(int((miss & det).sum()), int(det.sum()), pct(int((miss & det).sum()), int(det.sum()))),
        miss_train=(int((miss & train).sum()), int(train.sum()), pct(int((miss & train).sum()), int(train.sum()))),
        miss_after=(int((miss & ~train).sum()), int((~train).sum()), pct(int((miss & ~train).sum()), int((~train).sum()))),
        own_miss_all=(int((~own_has).sum()), pct(int((~own_has).sum()), N)),
        own_miss_det=(int((~own_has & det).sum()), pct(int((~own_has & det).sum()), int(det.sum()))),
        f839_n3=len(f839), f839_rows=sum(REP[c][k]['n_all'] for k in f839), f839_det=sum(REP[c][k]['n_all_det'] for k in f839),
        f839_codes=f839,
        diff=(int(diff.sum()), pct(int(diff.sum()), N)), diff_has3=(int((diff & has3).sum()), pct(int((diff & has3).sum()), int(has3.sum()))),
        diff_det=(int((diff & det).sum()), pct(int((diff & det).sum()), int(det.sum()))),
        fill=(int(fill.sum()), int((fill & det).sum())), lose=(int(lose.sum()), int((lose & det).sum())),
        chg=(int(chg.sum()), int((chg & det).sum())),
        diff_rows=diff,
    )

# ---------------------------------------------------------------- 근거 표 (3단계 × 항목)
ROWS = []
for k3 in CODES3:
    nm2, k2, nm1 = INFO3[k3]
    rows3 = idx3[k3]
    for c in FOOD_COLS:
        r = REP[c][k3]
        n_diff = int(SUMMARY[c]['diff_rows'][rows3].sum())
        if r['stage'] == 1:
            src = ', '.join(f'{code}({n})' for code, n in sorted(r['src_codes'].items(), key=lambda kv: (-kv[1], kv[0])))
        else:
            src = ', '.join(r['src_codes'])
        codes = list(r['src_codes'])
        srcs = sorted({VAL[c][code][2] for code in codes})
        raws = sorted({VAL[c][code][3] for code in codes})
        aws = sorted({DB[code]['aw_source'] or '' for code in codes if VAL[c][code][2] == 't_aw_cl'}) if c == 'aw' else []
        top, second = r['top'], r['second']
        ROWS.append({
            'train_prdlst_code_2': k3, 'train_prdlst_nm_2': nm2, 'train_prdlst_code': k2, 'train_prdlst_nm': nm1,
            '항목': c, '항목명': FOOD_KO[c],
            '학습기간_시료수': r['n_train'], '학습기간_검출수': r['n_train_det'], '학습기간_상태수': r['n_states'],
            '최빈상태': state_txt(top[0]) if top else '', '최빈상태_시료수': top[1] if top else '',
            '최빈상태_비율(%)': pct(top[1], r['n_train']) if top else '',
            '2위상태': state_txt(second[0]) if second else '', '2위상태_시료수': second[1] if second else '',
            '2위상태_비율(%)': pct(second[1], r['n_train']) if second else '',
            '최빈_동률': '예' if r['tie'] else '아니오',
            '단계': r['stage'], '단계_설명': STAGE_TXT[r['stage']], '물려받은_이유': r['reason'],
            '대표값': '' if r['num'] is None else num_txt(r['num']), '대표값_표기': r['label'] or '',
            '대표값_품목코드(학습기간_시료수)': src, '대표값_자료': ' / '.join(srcs), '대표값_원문': ' / '.join(raws),
            'aw_source(t_aw_cl)': ' / '.join(aws),
            '출처확인필요_0839': ('예' if r['flag839'] else '아니오') if c == 'aw' else '',
            '전체기간_시료수': r['n_all'], '전체기간_검출수': r['n_all_det'],
            '자기값과_대표값이_다른_시료수': n_diff, '자기값과_대표값이_다른_비율(%)': pct(n_diff, r['n_all']),
        })
CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CHECK_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
    w.writeheader()
    w.writerows(ROWS)
log(f'근거 표 저장: {CHECK_PATH} ({len(ROWS)}줄 = 3단계 {len(CODES3)}개 × 항목 3)')

# ---------------------------------------------------------------- 요약 수치 출력
print('\n== 요약 ==')
print(f'시료_시군구 {N:,}행 · 검출 {int(det.sum()):,} · Z 아닌 행 {int((~z_row).sum()):,} · 3단계 코드 {len(CODES3)}개 (코드 빈 행 {int((~has3).sum())})')
for c in FOOD_COLS:
    s = SUMMARY[c]
    print(f'\n[{FOOD_KO[c]} {c}]')
    for k in (1, 2, 3, 4):
        v = s['stages'][k]
        print(f'  단계 {k} ({STAGE_TXT[k]}): 3단계 {v["n3"]}개 · 시료 {v["rows"]:,}행 · 검출 {v["det"]:,}')
    print('  단계를 정한 이유별 3단계 수:', dict(s['reasons']))
    print(f'  시료 행 결측 — 전체 {s["miss_all"][0]:,}/{s["miss_all"][1]:,} ({s["miss_all"][2]}%) · '
          f'Z 아닌 행 {s["miss_nonz"][0]:,}/{s["miss_nonz"][1]:,} ({s["miss_nonz"][2]}%) · '
          f'Z 행 {s["miss_z"][0]:,}/{s["miss_z"][1]:,} ({s["miss_z"][2]}%) · 검출 행 {s["miss_det"][0]:,}/{s["miss_det"][1]:,} ({s["miss_det"][2]}%)')
    print(f'    학습 기간 {s["miss_train"][2]}% · 그 뒤 {s["miss_after"][2]}%  '
          f'(참고: 시료 자기 값 기준 결측 전체 {s["own_miss_all"][1]}% · 검출 행 {s["own_miss_det"][1]}%)')
    if c == 'aw':
        print(f'  0.839 가 대표값인 3단계 {s["f839_n3"]}개 · 시료 {s["f839_rows"]:,}행 · 검출 {s["f839_det"]:,} — {s["f839_codes"]}')
    print(f'  시료 자기 값과 대표값이 다른 행 {s["diff"][0]:,} ({s["diff"][1]}% of {N:,}; 3단계 코드 있는 행 기준 {s["diff_has3"][1]}%) · '
          f'검출 행 중 {s["diff_det"][0]:,} ({s["diff_det"][1]}%)')
    print(f'    내역: 자기 값 없음→대표값 있음 {s["fill"][0]:,}(검출 {s["fill"][1]}) · 자기 값 있음→대표값 결측 {s["lose"][0]:,}(검출 {s["lose"][1]}) · '
          f'둘 다 값인데 다름 {s["chg"][0]:,}(검출 {s["chg"][1]})')
log('끝')
