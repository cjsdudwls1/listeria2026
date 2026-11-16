# -*- coding: utf-8 -*-
"""
04_시군구_토양.py — 시군구 x 날짜 토양표 (산도·유기물·유효인산의 지난 3년 토양검정 중앙값과 기록 수)

입력 (읽기만 함)
  - 토양 원자료 : 리스테리아ML2/data/raw/raw_soil_ph_info.parquet  (01_원자료_받기.py 가 받은 api_data_info.soil_ph_info)
                  쓰는 칼럼: legal_dong_code(법정동코드), legal_dong_nm(법정동명, 점검 표에만), exam_day(토양검정일),
                             acid(산도), om(유기물), vldpha(유효인산)
  - 기상표      : 리스테리아ML2/data/데이터셋/식정원_db_기상기후/시군구_일별_기상.parquet (sgg_code·date 만 읽음 — 날짜 범위·시군구 목록)
  - 시군구 배정 : 리스테리아ML2/notebook/02_시군구_배정.py 의 Region·load_mapping·assign_sgg (같은 함수를 불러 씀)
  - 시료        : 리스테리아ML2/data/데이터셋/시료_시군구.parquet (sgg_code·mnfctur_de·judge_result — 참고 수치만)

출력
  - 리스테리아ML2/data/데이터셋/식정원_db_토양/시군구_일별_토양.parquet
      sgg_code(string), date(date32), soil_ph, soil_ph_n, soil_om, soil_om_n, soil_vldpha, soil_vldpha_n
      (값 float32·결측 null, 건수 int32). 정렬: sgg_code, date. 시군구 256곳 x 날짜 전부.
  - 리스테리아ML2/data/중간/시군구_토양_점검.csv (점검 표)

규칙 (사용자 결정 2026-10-16)
  - 토양 기록의 시군구: 기록마다 legal_dong_code(10자리)를 02_시군구_배정.py 의 assign_sgg 로 오늘 기준 시군구(256개 중 하나)에
    매긴다. 토양 원자료에는 주소가 없으므로 주소는 넘기지 않는다(addrs=None). 시군구를 못 정한 기록은 빼고 그 수를 점검 표에 적는다.
  - 값: 시군구마다, 날짜 t 마다, exam_day 가 [t−1095일, t−1일] 안인 그 시군구 기록의 항목 값 중앙값과,
    그 항목 값이 있는(null 아닌) 기록 수. 당일(t) 검정은 넣지 않는다.
    중앙값은 numpy median (기록 수가 짝수면 가운데 두 값의 평균). float64 로 계산한 뒤 float32 로 저장한다.
    기록이 0건이면 값은 null, 건수는 0.
  - 값은 원자료 그대로: 이상값·중복 행을 지우지 않고, exam_type(경지구분코드)도 가리지 않는다. 결측 채움·단위 변환 없음.
  - 날짜 범위: 기상표의 date 최소 ~ 최대(이어진 날짜 전부). 시군구 목록: 02 의 Region.targets (기상표 시군구 목록과 같은지 확인).
  - 시료 참고 수치: 시료_시군구.parquet 의 (sgg_code, mnfctur_de 의 날짜) 로 붙였을 때 항목별 결측 행 수·검출 수.

다시 돌리면 같은 결과가 나온다(무작위 없음 — 점검용 표본 칸도 고정 시드).
실행: py "리스테리아ML2\\notebook\\04_시군구_토양.py"   (py 3.14 — numpy·pyarrow 사용, pandas 쓰지 않음)
"""
import csv
import importlib.util
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

T0 = time.time()
BASE = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2')
RAW_SOIL = BASE / 'data' / 'raw' / 'raw_soil_ph_info.parquet'
WEATHER = BASE / 'data' / '데이터셋' / '식정원_db_기상기후' / '시군구_일별_기상.parquet'
SAMPLE_SGG = BASE / 'data' / '데이터셋' / '시료_시군구.parquet'
SGG02 = BASE / 'notebook' / '02_시군구_배정.py'
OUT_PATH = BASE / 'data' / '데이터셋' / '식정원_db_토양' / '시군구_일별_토양.parquet'
CHECK_PATH = BASE / 'data' / '중간' / '시군구_토양_점검.csv'

WINDOW_DAYS = 1095            # exam_day ∈ [t − 1095일, t − 1일]
SINCE = np.datetime64('2018-01-01', 'D')   # 점검 표 '2018년 이후' 기준
N_VERIFY = 200                # 항목마다 원자료에서 직접 다시 계산해 대조하는 칸 수
VERIFY_SEED = 20261016

# (원자료 칼럼, 결과 칼럼, 한글 이름) — 한글 이름은 DB 칼럼 코멘트 그대로
ITEMS = [('acid', 'soil_ph', '산도'), ('om', 'soil_om', '유기물'), ('vldpha', 'soil_vldpha', '유효인산')]


def log(*a):
    print(f'[{time.time() - T0:6.0f}초]', *a, flush=True)


# ---------------------------------------------------------------- 02 의 시군구 배정 함수
spec = importlib.util.spec_from_file_location('sgg02', str(SGG02))
sgg02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sgg02)
region = sgg02.Region(sgg02.KRC)
mapping = sgg02.load_mapping(sgg02.XLSX)
TARGETS = list(region.targets)            # 오늘 기준 시군구 256곳 (정렬됨)
assert TARGETS == sorted(TARGETS) and len(TARGETS) == 256, len(TARGETS)
S_IDX = {c: i for i, c in enumerate(TARGETS)}

# ---------------------------------------------------------------- 날짜 범위·시군구 목록 (기상표)
wx = pq.read_table(WEATHER, columns=['sgg_code', 'date'])
wd = pc.min_max(wx['date'])
DATE_MIN = np.datetime64(wd['min'].as_py(), 'D')
DATE_MAX = np.datetime64(wd['max'].as_py(), 'D')
DATES = np.arange(DATE_MIN, DATE_MAX + np.timedelta64(1, 'D'), dtype='datetime64[D]')
ND = len(DATES)
assert pc.count_distinct(wx['date']).as_py() == ND, '기상표 날짜가 이어져 있지 않음'
WX_SGG = sorted(pc.unique(wx['sgg_code']).to_pylist())
assert WX_SGG == TARGETS, '기상표 시군구 목록과 02 Region.targets 가 다름'
del wx
D = DATES.astype(np.int64)                # 1970-01-01 부터 며칠째
log(f'날짜 {DATE_MIN} ~ {DATE_MAX} ({ND:,}일), 시군구 {len(TARGETS)}곳 → {ND * len(TARGETS):,}칸')

# ---------------------------------------------------------------- 토양 원자료 + 시군구 배정
raw = pq.read_table(RAW_SOIL, columns=['legal_dong_code', 'exam_day'] + [c for c, _, _ in ITEMS])
N_RAW = raw.num_rows
assert raw['legal_dong_code'].null_count == 0 and raw['exam_day'].null_count == 0
enc = pc.dictionary_encode(raw['legal_dong_code']).combine_chunks()
codes_u = enc.dictionary.to_pylist()       # 서로 다른 legal_dong_code
codes_i = enc.indices.to_numpy()
sgg_u, how_u, assign_info = sgg02.assign_sgg(codes_u, None, region=region, mapping=mapping, with_reason=True)
bad = {c for c in sgg_u if c is not None and c not in S_IDX}
assert not bad, f'256곳 밖 코드: {bad}'
lut = np.array([-1 if c is None else S_IDX[c] for c in sgg_u], dtype=np.int32)
row_s = lut[codes_i]                        # 기록마다 시군구 번호 (-1 = 못 정함)
code_rows = np.bincount(codes_i, minlength=len(codes_u))
UNASSIGNED = [(codes_u[k], int(code_rows[k]), how_u[k]) for k in range(len(codes_u)) if sgg_u[k] is None]
UNASSIGNED.sort(key=lambda x: (-x[1], x[0]))
n_unassigned = int((row_s < 0).sum())
assert n_unassigned == sum(x[1] for x in UNASSIGNED)
how_rows = Counter()
for k in range(len(codes_u)):
    how_rows[how_u[k].split(':')[0]] += int(code_rows[k])
log(f'토양 기록 {N_RAW:,}행, 서로 다른 legal_dong_code {len(codes_u):,}개, 시군구 못 정한 기록 {n_unassigned:,}행 '
    f'({len(UNASSIGNED)}개 코드)')

row_day = pc.cast(raw['exam_day'], pa.int32()).to_numpy().astype(np.int64)
row_val = {c: raw[c].to_numpy(zero_copy_only=False).astype(np.float64) for c, _, _ in ITEMS}   # null → NaN
for c, _, _ in ITEMS:
    assert int(np.isnan(row_val[c]).sum()) == raw[c].null_count, f'{c}: null 이 아닌 NaN 값이 있음'
EXAM_MIN = str(np.datetime64(int(row_day.min()), 'D'))
EXAM_MAX = str(np.datetime64(int(row_day.max()), 'D'))
RAW_NONNULL = {c: int((~np.isnan(row_val[c])).sum()) for c, _, _ in ITEMS}
RAW_NONNULL_ASSIGNED = {c: int(((~np.isnan(row_val[c])) & (row_s >= 0)).sum()) for c, _, _ in ITEMS}
del raw, enc

# 시군구·검정일 순으로 정렬 (못 정한 기록은 뺌)
keep = np.nonzero(row_s >= 0)[0]
order = keep[np.lexsort((row_day[keep], row_s[keep]))]
s_sorted = row_s[order]
d_sorted = row_day[order]
v_sorted = {c: row_val[c][order] for c, _, _ in ITEMS}
bounds = np.searchsorted(s_sorted, np.arange(len(TARGETS) + 1), side='left')
del keep

# ---------------------------------------------------------------- 시군구 x 날짜 계산
NS = len(TARGETS)
MED = {c: np.full(NS * ND, np.nan, dtype=np.float64) for c, _, _ in ITEMS}
CNT = {c: np.zeros(NS * ND, dtype=np.int32) for c, _, _ in ITEMS}
lo_key = D - WINDOW_DAYS                  # exam_day >= t − 1095
hi_key = D                                # exam_day <  t  (= exam_day <= t − 1)
for s in range(NS):
    a, b = bounds[s], bounds[s + 1]
    base = s * ND
    for c, _, _ in ITEMS:
        v = v_sorted[c][a:b]
        d = d_sorted[a:b]
        ok = ~np.isnan(v)
        v, d = v[ok], d[ok]
        lo = np.searchsorted(d, lo_key, side='left')
        hi = np.searchsorted(d, hi_key, side='left')
        n = hi - lo
        CNT[c][base:base + ND] = n
        med = MED[c]
        prev = (-1, -1)
        last = np.nan
        for j in range(ND):
            if n[j] == 0:
                continue
            key = (lo[j], hi[j])
            if key != prev:
                last = float(np.median(v[lo[j]:hi[j]]))
                prev = key
            med[base + j] = last
    if (s + 1) % 32 == 0 or s == NS - 1:
        log(f'시군구 {s + 1}/{NS}')

# ---------------------------------------------------------------- 저장
sgg_col = pa.array(np.repeat(np.array(TARGETS, dtype=object), ND).tolist(), type=pa.string())
date_col = pa.array(np.tile(DATES, NS), type=pa.date32())
cols = {'sgg_code': sgg_col, 'date': date_col}
for c, out, _ in ITEMS:
    m = CNT[c] == 0
    assert np.array_equal(m, np.isnan(MED[c])), c
    cols[out] = pa.array(MED[c].astype(np.float32), type=pa.float32(), mask=m)
    cols[out + '_n'] = pa.array(CNT[c], type=pa.int32())
table = pa.table(cols)
assert table.num_rows == NS * ND
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(table, OUT_PATH)
log(f'저장 {OUT_PATH} ({table.num_rows:,}행 x {table.num_columns}칼럼)')

# ---------------------------------------------------------------- 원자료에서 직접 다시 계산해 대조 (표본 칸)
rng = np.random.default_rng(VERIFY_SEED)
VERIFY = {}
for c, out, _ in ITEMS:
    cells = rng.choice(NS * ND, size=N_VERIFY, replace=False)
    ok_v = ok_n = 0
    for cell in cells:
        s, j = divmod(int(cell), ND)
        t = int(D[j])
        msk = (row_s == s) & (row_day >= t - WINDOW_DAYS) & (row_day <= t - 1) & ~np.isnan(row_val[c])
        vals = row_val[c][msk].tolist()
        n_ref = len(vals)
        got_n = int(CNT[c][cell])
        got_v = table[out][int(cell)].as_py()
        ok_n += (got_n == n_ref)
        if n_ref == 0:
            ok_v += got_v is None
        else:
            ref = float(np.float32(statistics.median(vals)))
            ok_v += (got_v is not None and got_v == ref)
    VERIFY[c] = (ok_v, ok_n)
    log(f'대조 {out}: 값 {ok_v}/{N_VERIFY}, 건수 {ok_n}/{N_VERIFY}')
    assert ok_v == N_VERIFY and ok_n == N_VERIFY, f'{out} 직접 계산과 다름'

# ---------------------------------------------------------------- 시료 참고 수치 (sgg_code, mnfctur_de 날짜)
smp = pq.read_table(SAMPLE_SGG, columns=['sgg_code', 'mnfctur_de', 'judge_result'])
N_SMP = smp.num_rows
s_code = smp['sgg_code'].to_pylist()
det = np.array([j == '검출' for j in pc.cast(smp['judge_result'], pa.string()).to_pylist()])
ts = smp['mnfctur_de']
n_time = int(pc.sum(pc.not_equal(pc.cast(pc.floor_temporal(ts, unit='day'), pa.int64()),
                                 pc.cast(ts, pa.int64()))).as_py() or 0)
s_day = pc.cast(pc.cast(pc.floor_temporal(ts, unit='day'), pa.date32()), pa.int32())   # mnfctur_de 의 날짜 (일 번호)
s_day_null = np.array(s_day.is_null().to_pylist(), dtype=bool)
s_day_f = s_day.to_numpy(zero_copy_only=False).astype(np.float64)                   # null 은 NaN — 아래에서 따로 처리
s_idx = np.array([S_IDX.get(x, -1) if x is not None else -1 for x in s_code])
j_idx = np.full(N_SMP, -1, dtype=np.int64)
j_idx[~s_day_null] = s_day_f[~s_day_null].astype(np.int64) - int(D[0])
in_range = (~s_day_null) & (j_idx >= 0) & (j_idx < ND)
has_sgg = s_idx >= 0
joinable = has_sgg & in_range
cell_idx = np.where(joinable, s_idx * ND + np.clip(j_idx, 0, ND - 1), 0)
SAMPLE_REF = {}
for c, out, _ in ITEMS:
    n_at = np.where(joinable, CNT[c][cell_idx], 0)
    miss = n_at == 0
    SAMPLE_REF[c] = dict(
        rows=N_SMP, det=int(det.sum()),
        miss_rows=int(miss.sum()), miss_det=int((miss & det).sum()),
        miss_no_sgg=int((~has_sgg).sum()), miss_no_sgg_det=int(((~has_sgg) & det).sum()),
        miss_no_date=int((has_sgg & s_day_null).sum()), miss_no_date_det=int((has_sgg & s_day_null & det).sum()),
        miss_out_range=int((has_sgg & ~s_day_null & ~in_range).sum()),
        miss_out_range_det=int((has_sgg & ~s_day_null & ~in_range & det).sum()),
        miss_zero=int((joinable & miss).sum()), miss_zero_det=int((joinable & miss & det).sum()),
    )
log('시료 참고 수치', {c: (v['miss_rows'], v['miss_det']) for c, v in SAMPLE_REF.items()})

# ---------------------------------------------------------------- 점검 표
ROWS = []


def add(section, item, target, metric, value, note=''):
    ROWS.append({'구분': section, '항목': item, '대상': target, '지표': metric, '값': value, '설명': note})


def pct(a, b):
    return round(100.0 * a / b, 3) if b else ''


add('범위', '', '전체', '날짜 범위', f'{DATE_MIN} ~ {DATE_MAX}', '기상표(시군구_일별_기상.parquet) date 최소~최대')
add('범위', '', '전체', '날짜 수', ND)
add('범위', '', '전체', '시군구 수', NS, '02_시군구_배정.py Region.targets (기상표 시군구 목록과 같음)')
add('범위', '', '전체', '전체 칸', NS * ND)
add('범위', '', '전체', '창', f'exam_day ∈ [t−{WINDOW_DAYS}일, t−1일]', '당일 검정 제외')
add('원자료', '', '전체', '토양 기록 수', N_RAW, 'api_data_info.soil_ph_info 전체 행')
add('원자료', '', '전체', 'exam_day 범위', f'{EXAM_MIN} ~ {EXAM_MAX}')
add('원자료', '', '전체', '서로 다른 legal_dong_code 수', len(codes_u))
for k in sorted(how_rows):
    add('원자료', '', '배정 근거별 기록 수', k, how_rows[k], '02_시군구_배정.py assign() 근거 문자열(콜론 뒤 생략)')
for c, out, ko in ITEMS:
    add('원자료', f'{out} ({c} {ko})', '전체', '값 있는 기록 수', RAW_NONNULL[c])
    add('원자료', f'{out} ({c} {ko})', '전체', 'null 기록 수', N_RAW - RAW_NONNULL[c])
    add('원자료', f'{out} ({c} {ko})', '시군구 정한 기록', '값 있는 기록 수', RAW_NONNULL_ASSIGNED[c])

add('시군구 못 정함', '', '전체', '기록 수', n_unassigned, '계산에서 뺌')
add('시군구 못 정함', '', '전체', 'legal_dong_code 수', len(UNASSIGNED))
nm_tbl = pq.read_table(RAW_SOIL, columns=['legal_dong_code', 'legal_dong_nm'],
                       filters=[('legal_dong_code', 'in', [x[0] for x in UNASSIGNED])]) if UNASSIGNED else None
for code, nrow, how in UNASSIGNED:
    nms = sorted(set(nm_tbl.filter(pc.equal(nm_tbl['legal_dong_code'], code))['legal_dong_nm'].to_pylist()), key=str)
    st = region.K.get(code, {}).get('molit_use_yn', '법정동 표에 없음')
    add('시군구 못 정함', '', code, '기록 수', nrow,
        f'legal_dong_nm={"/".join(map(str, nms))}; korea_region_code molit_use_yn={st}; 근거={how}')

sgg_nm = {c: region.sg_nm[c] for c in TARGETS}
since_mask = np.tile(DATES >= SINCE, NS)
n_since = int(since_mask.sum())
for c, out, ko in ITEMS:
    lab = f'{out} ({c} {ko})'
    n = CNT[c]
    miss = n == 0
    add('결측 칸', lab, '전체', '결측 칸', int(miss.sum()))
    add('결측 칸', lab, '전체', '결측 칸 비율(%)', pct(int(miss.sum()), NS * ND))
    add('결측 칸', lab, '2018-01-01 이후', '결측 칸', int((miss & since_mask).sum()))
    add('결측 칸', lab, '2018-01-01 이후', '결측 칸 비율(%)', pct(int((miss & since_mask).sum()), n_since))
    allmiss = miss.reshape(NS, ND).all(axis=0)
    add('결측 칸', lab, '전체', '모든 시군구가 결측인 날 수', int(allmiss.sum()))
    add('결측 칸', lab, '2018-01-01 이후', '모든 시군구가 결측인 날 수', int((allmiss & (DATES >= SINCE)).sum()))
    # 시군구별
    M = miss.reshape(NS, ND)
    r_all = M.mean(axis=1) * 100
    r_since = M[:, DATES >= SINCE].mean(axis=1) * 100
    add('시군구별 결측', lab, '전체', '결측 칸이 하나라도 있는 시군구 수(2018-01-01 이후)', int((r_since > 0).sum()))
    add('시군구별 결측', lab, '전체', '모든 칸이 결측인 시군구 수', int((r_all == 100).sum()))
    top = sorted(range(NS), key=lambda i: (-r_since[i], -r_all[i], TARGETS[i]))
    for i in top:
        if r_since[i] == 0:
            break
        add('시군구별 결측', lab, f'{TARGETS[i]} {sgg_nm[TARGETS[i]]}', '결측 칸 비율(%) 2018-01-01 이후 / 전체',
            f'{r_since[i]:.2f} / {r_all[i]:.2f}')
    # 건수 분포
    for scope, sel in (('전체', np.ones(NS * ND, bool)), ('2018-01-01 이후', since_mask)):
        x = n[sel]
        q = np.percentile(x, [0, 1, 5, 10, 25, 50, 75, 90, 99, 100], method='lower')
        add('건수 분포', lab, scope, '칸별 기록 수 백분위 0/1/5/10/25/50/75/90/99/100', '/'.join(str(int(v)) for v in q))
        for lab_b, lo_b, hi_b in (('0건', 0, 0), ('1~4건', 1, 4), ('5~9건', 5, 9), ('10~29건', 10, 29),
                                  ('30~99건', 30, 99), ('100~999건', 100, 999), ('1000건 이상', 1000, 1 << 40)):
            k = int(((x >= lo_b) & (x <= hi_b)).sum())
            add('건수 분포', lab, scope, f'칸 수 {lab_b}', k, f'{pct(k, len(x))}%')
    # 값 범위
    vv = MED[c][~miss]
    q = np.percentile(vv, [0, 1, 50, 99, 100], method='lower')
    add('값 범위', lab, '칸 값(중앙값)', '최솟값/1%/중앙값/99%/최댓값', '/'.join(f'{float(np.float32(v)):g}' for v in q))
    rv = row_val[c][(row_s >= 0) & ~np.isnan(row_val[c])]
    q = np.percentile(rv, [0, 1, 50, 99, 100], method='lower')
    add('값 범위', lab, '원자료 기록 값(시군구 정한 기록)', '최솟값/1%/중앙값/99%/최댓값', '/'.join(f'{v:g}' for v in q))
    add('대조', lab, f'표본 {N_VERIFY}칸', '원자료 직접 계산과 같은 칸 수(값/건수)', f'{VERIFY[c][0]}/{VERIFY[c][1]}',
        f'고정 시드 {VERIFY_SEED}, statistics.median 으로 다시 계산')

add('시료 참고', '', '시료_시군구.parquet', '행 수', N_SMP)
add('시료 참고', '', '시료_시군구.parquet', '검출 수', int(det.sum()))
add('시료 참고', '', '시료_시군구.parquet', 'mnfctur_de 에 시각(0시 아님)이 있는 행', n_time, '날짜 부분만 써서 붙임')
for c, out, ko in ITEMS:
    lab = f'{out} ({c} {ko})'
    r = SAMPLE_REF[c]
    add('시료 참고', lab, '결측 행 전체', '행 수 / 검출 수', f'{r["miss_rows"]} / {r["miss_det"]}')
    add('시료 참고', lab, '  sgg_code 없음', '행 수 / 검출 수', f'{r["miss_no_sgg"]} / {r["miss_no_sgg_det"]}')
    add('시료 참고', lab, '  mnfctur_de 없음', '행 수 / 검출 수', f'{r["miss_no_date"]} / {r["miss_no_date_det"]}')
    add('시료 참고', lab, '  날짜가 표 범위 밖', '행 수 / 검출 수', f'{r["miss_out_range"]} / {r["miss_out_range_det"]}')
    add('시료 참고', lab, '  붙었으나 지난 3년 기록 0건', '행 수 / 검출 수', f'{r["miss_zero"]} / {r["miss_zero_det"]}')

CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CHECK_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
    w.writeheader()
    w.writerows(ROWS)
log(f'점검 표: {CHECK_PATH} ({len(ROWS)}줄)')
log('배정 요약', assign_info)
for c, out, _ in ITEMS:
    miss = CNT[c] == 0
    log(f'{out}: 결측 칸 {pct(int(miss.sum()), NS * ND)}% (2018-01-01 이후 {pct(int((miss & since_mask).sum()), n_since)}%), '
        f'시료 결측 행 {SAMPLE_REF[c]["miss_rows"]:,} (검출 {SAMPLE_REF[c]["miss_det"]})')
log('끝')
