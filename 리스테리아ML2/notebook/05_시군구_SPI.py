# -*- coding: utf-8 -*-
"""
05_시군구_SPI.py — 시군구 x 날짜 표준강수지수(SPI) 표 (SPI3 · SPI6 두 지수)

지수 선택: 사용자 결정 2026-10-16 — SPI3·SPI6 두 지수만.
  근거(사용자에게 보고된 내용): 기상청 공식 기상가뭄 판단 기준이 SPI6, 2026-05 부터 보조 지표 SPI3.

입력 (읽기만 함)
  - 표준강수지수 일자료 : 리스테리아ML2/data/외부_데이터/CLM_SPI_DD_합본_20140101_20261006.csv
      cp949, 머리줄 '지점,지점명,일시,SPI1,SPI3,SPI6,SPI12', 지점·날짜마다 1행.
      기상청 기상자료개방포털(data.kma.go.kr) > 기후 > 가뭄 > 표준강수지수 에서 받은 파일의 합본
      (보관/리스테리아ML2_20261016/final_기상조인코드_사본.ipynb 4-6절 설명).
  - 지점 위치 이력 : 리스테리아ML2/data/외부_데이터/META_관측지점정보.csv
      cp949, 첫 줄과 머리줄 다음 줄이 빈 줄. 칼럼 지점·시작일·종료일·지점명·…·위도·경도·…
  - 시군구 대표 좌표 : 리스테리아ML2/data/데이터셋/시군구_대표좌표.parquet (sgg_code, lat, lon)
  - 날짜 범위 : 리스테리아ML2/data/데이터셋/식정원_db_기상기후/시군구_일별_기상.parquet 의 date 최소~최대
      (기상표와 같은 날짜 범위로 맞춘다)

출력
  - 리스테리아ML2/data/데이터셋/외부_수집/시군구_일별_SPI.parquet
      sgg_code(string), date(date32), spi3, spi6(float32, 결측은 null). 정렬: sgg_code, date. 256 시군구 × 날짜 전부
  - 리스테리아ML2/data/중간/시군구_SPI_점검.csv (지수별 점검 표)

규칙 — 보관/리스테리아ML2_20261016/final_기상조인코드_사본.ipynb 4-6절(표준강수지수)과 같다. 바뀐 것은 좌표뿐이다:
시료 좌표(lat·lon) → 시군구 대표 좌표. 03_시군구_기상.py 와 같은 계산 함수다.
  - 지수 값은 파일 글자를 숫자로만 바꾼다(빈칸은 결측). 결측 채움·이상값 처리 없음. 지점 번호는 글자로 다룬다.
  - 지점 좌표: 날짜마다 "META 의 시작일이 그날 이전(같은 날 포함)인 행 중 가장 최근 행"의 위도·경도
    (그런 행이 없으면 가장 이른 행). 종료일은 좌표를 고르는 데 쓰지 않는다.
  - 시군구마다 날짜마다, 그날 그 지수 값이 있는 지점 중 구면 거리(지구 반지름 6371.0088km)로 가까운 3곳의 단순 평균.
    3곳이 안 되면 있는 곳만 평균, 한 곳도 없으면 결측. 거리 상한 없음. 대체 칼럼 없음.
    같은 거리면 지점 번호(글자) 순서가 앞인 곳.
  - 지점 값은 float32 로 담는다(노트북 4-6절과 같다 — 예: 파일의 −3.09 는 float32 로 −3.0899999…). 평균은 float64 로
    계산한 뒤 float32 로 저장한다.
  - CSV 에 없는 날(파일 기간 밖인 날, 그날 행이 없는 날, 그 지수가 전 지점 빈칸인 날)은 결측이다.
  - 같은 지점·같은 날 행이 둘 이상이면 멈춘다. META 에 없는 지점이 있으면 멈춘다.
    날짜 범위 안에서 위치 이력의 빈틈·겹침이 있으면(어느 좌표를 쓸지 정해야 하므로) 멈춘다.

다시 돌리면 같은 결과가 나온다(무작위 없음 — 점검용 표본 칸도 고정 시드).
실행: py "리스테리아ML2\\notebook\\05_시군구_SPI.py"   (py 3.14 — numpy·pyarrow 사용, pandas 쓰지 않음)
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

T0 = time.time()
BASE = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2')
SPI_PATH = BASE / 'data' / '외부_데이터' / 'CLM_SPI_DD_합본_20140101_20261006.csv'
META_PATH = BASE / 'data' / '외부_데이터' / 'META_관측지점정보.csv'
SGG_COORD = BASE / 'data' / '데이터셋' / '시군구_대표좌표.parquet'
WX_PATH = BASE / 'data' / '데이터셋' / '식정원_db_기상기후' / '시군구_일별_기상.parquet'
OUT_PATH = BASE / 'data' / '데이터셋' / '외부_수집' / '시군구_일별_SPI.parquet'
CHECK_PATH = BASE / 'data' / '중간' / '시군구_SPI_점검.csv'

K_NEAREST = 3                 # 가까운 지점 3곳 평균
EARTH_R_KM = 6371.0088        # 구면 거리 계산에 쓰는 지구 반지름(km)
BLOCK = 32                    # 한 번에 계산하는 날짜 수 (메모리 때문에). 값에는 영향이 없다
N_VERIFY = 400                # 지수마다 원자료(CSV 행)에서 직접 다시 계산해 대조하는 칸 수
VERIFY_SEED = 20261016

CSV_HEADER = ['지점', '지점명', '일시', 'SPI1', 'SPI3', 'SPI6', 'SPI12']
VARIABLES = [   # 결과 칼럼 · 파일 칼럼 · 한글 이름
    dict(var='spi3', src='SPI3', ko='표준강수지수 3개월'),
    dict(var='spi6', src='SPI6', ko='표준강수지수 6개월'),
]
VARS = [v['var'] for v in VARIABLES]
SRC = [v['src'] for v in VARIABLES]


def haversine_km(lat1, lon1, lat2, lon2):
    """두 지점 사이의 구면 거리(km). 위도·경도는 도 단위 (노트북 1절과 같은 식)"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return EARTH_R_KM * 2 * np.arcsin(np.sqrt(a))


def day_ranges(days):
    """날짜(datetime64[D]) 목록을 이어진 구간으로 묶어 글로 적는다 — 예: '2025-05-08~2025-05-21(14일)'"""
    days = np.sort(np.asarray(days, dtype='datetime64[D]'))
    out, i = [], 0
    while i < len(days):
        j = i
        while j + 1 < len(days) and (days[j + 1] - days[j]).astype(int) == 1:
            j += 1
        out.append(f'{days[i]}' if i == j else f'{days[i]}~{days[j]}({j - i + 1}일)')
        i = j + 1
    return out


def to_num(s, where):
    """파일 글자 → 숫자. 빈칸은 NaN. 숫자로 안 바뀌는 글자가 있으면 멈춘다"""
    if s == '':
        return np.nan
    try:
        return float(s)
    except ValueError:
        raise ValueError(f'{where}: 숫자로 안 바뀌는 값 {s!r}')


# ── 1. 시군구 좌표와 날짜 범위 ───────────────────────────────────────────────
sgg = pq.read_table(SGG_COORD, columns=['sgg_code', 'lat', 'lon'])
codes_all = sgg['sgg_code'].to_pylist()
order = np.argsort(np.array(codes_all, dtype=object), kind='stable')
SGG_CODES = [codes_all[i] for i in order]
LOC_XY = np.column_stack([sgg['lat'].to_numpy()[order], sgg['lon'].to_numpy()[order]]).astype(np.float64)
assert len(set(SGG_CODES)) == len(SGG_CODES), 'sgg_code 가 겹침'
assert all(c is not None and len(c) == 5 for c in SGG_CODES), 'sgg_code 가 5자리 글자가 아님'
assert not np.isnan(LOC_XY).any(), '좌표가 빈 시군구가 있음'
N_LOC = len(SGG_CODES)

wx = pq.read_table(WX_PATH, columns=['sgg_code', 'date'])
DATE_MIN = np.datetime64(pc.min(wx['date']).as_py(), 'D')
DATE_MAX = np.datetime64(pc.max(wx['date']).as_py(), 'D')
wx_codes = set(pc.unique(wx['sgg_code']).to_pylist())
del wx
DATES = np.arange(DATE_MIN, DATE_MAX + np.timedelta64(1, 'D'), dtype='datetime64[D]')
N_DATES = len(DATES)
print(f'시군구 {N_LOC}곳 · 날짜 범위(기상표 {WX_PATH.name} 의 date) {DATE_MIN} ~ {DATE_MAX} ({N_DATES:,}일) → {N_LOC} × {N_DATES:,} = {N_LOC * N_DATES:,}행')
print(f'  기상표의 sgg_code 와 대표 좌표 파일의 sgg_code 가 같은가: {wx_codes == set(SGG_CODES)}'
      + ('' if wx_codes == set(SGG_CODES) else f' (대표 좌표에만 {sorted(set(SGG_CODES) - wx_codes)} · 기상표에만 {sorted(wx_codes - set(SGG_CODES))})'))


# ── 2. 표준강수지수 파일 읽기 ─────────────────────────────────────────────────
with open(SPI_PATH, encoding='cp949', newline='') as f:
    rows = list(csv.reader(f))
assert rows[0] == CSV_HEADER, rows[0]
rows = rows[1:]
assert all(len(r) == len(CSV_HEADER) for r in rows), '칼럼 수가 다른 줄이 있음'
assert all(r[0].isdigit() and r[1] != '' and len(r[2]) == 10 for r in rows), '지점 번호가 숫자 글자가 아니거나 지점명·일시가 빈 줄이 있음'
csv_sid = np.array([r[0] for r in rows], dtype=object)                       # 지점 번호는 글자로 둔다
csv_day = np.array([r[2] for r in rows], dtype='datetime64[D]')
csv_val = {c: np.array([to_num(r[CSV_HEADER.index(c)], f'{r[0]} {r[2]} {c}') for r in rows], dtype=np.float64) for c in SRC}
pair = np.char.add(np.char.add(csv_sid.astype(str), '|'), csv_day.astype(str))
n_dup = len(pair) - len(np.unique(pair))
if n_dup:
    raise RuntimeError(f'같은 지점·같은 날 행이 {n_dup}건 있음 — 어느 값을 쓸지 정해야 함')
name_of = {}
for r in rows:
    name_of.setdefault(r[0], set()).add(r[1])
SPI_START, SPI_END = csv_day.min(), csv_day.max()
SIDS = sorted(set(csv_sid.tolist()))                                          # 지점 번호(글자)를 글자 순으로 — 노트북 4-6절과 같다
print(f'파일 {SPI_PATH.name}: {len(rows):,}행 · 기간 {SPI_START} ~ {SPI_END} · 지점 {len(SIDS)}곳 · 지점명이 둘 이상인 지점 {sum(len(v) > 1 for v in name_of.values())}곳')
for c in SRC:
    print(f'  {c}: 빈칸 {int(np.isnan(csv_val[c]).sum())}칸 / {len(rows):,}칸')
del rows


# ── 3. 지점 좌표 (날짜별 위치) ────────────────────────────────────────────────
with open(META_PATH, encoding='cp949', newline='') as f:
    mrows = [r for r in csv.reader(f) if any(x.strip() for x in r)]           # 빈 줄은 건너뛴다
mh = mrows[0]
assert {'지점', '시작일', '종료일', '지점명', '위도', '경도'} <= set(mh), mh
mi = {k: mh.index(k) for k in ('지점', '시작일', '종료일', '지점명', '위도', '경도')}
meta = []
for r in mrows[1:]:
    assert len(r) == len(mh), f'META 칼럼 수가 다른 줄: {r}'
    meta.append({'sid': r[mi['지점']], 'name': r[mi['지점명']], 'start': np.datetime64(r[mi['시작일']], 'D'),
                 'end': np.datetime64(r[mi['종료일']], 'D') if r[mi['종료일']] else None,
                 'lat': float(r[mi['위도']]), 'lon': float(r[mi['경도']])})
st_rows = {}
for m in meta:
    st_rows.setdefault(m['sid'], []).append(m)
missing_meta = [s for s in SIDS if s not in st_rows]
assert not missing_meta, f'META 파일에 없는 지점: {missing_meta}'
for s in SIDS:
    st_rows[s].sort(key=lambda m: m['start'])
    starts = [m['start'] for m in st_rows[s]]
    assert len(set(starts)) == len(starts), f'{s}: 시작일이 같은 META 행이 두 개 이상'
print(f'META 파일 {META_PATH.name}: {len(meta)}행 · 지점 {len(st_rows)}곳 — 표준강수지수 지점 {len(SIDS)}곳이 모두 있다')

# 위치 이력 점검 (종료일은 좌표 고르기에 쓰지 않는다 — 날짜 범위 안에 빈틈·겹침이 있으면 멈추는 데만 쓴다)
gaps, overlaps = [], []
for s in SIDS:
    g = st_rows[s]
    for a, b in zip(g[:-1], g[1:]):
        if a['end'] is None or a['end'] > b['start']:
            overlaps.append((s, a['end'], b['start']))
        elif a['end'] < b['start']:
            gaps.append((s, a['end'], b['start']))
gap_in = [x for x in gaps if x[1] <= DATE_MAX and x[2] > DATE_MIN]
ovl_in = [x for x in overlaps if x[2] <= DATE_MAX and (x[1] is None or x[1] > DATE_MIN)]
assert not gap_in and not ovl_in, f'날짜 범위 안에 위치 이력의 빈틈·겹침이 있음 — 그 날짜에 어느 좌표를 쓸지 정해야 한다: {gap_in + ovl_in}'
print(f'  위치 이력 빈틈(전체 기간) {[(s, str(e), str(n)) for s, e, n in gaps] or "없음"} · 날짜 범위 안 빈틈·겹침 0건')
no_row_yet = [s for s in SIDS if st_rows[s][0]['start'] > DATE_MIN]


def xy_on(s, day):
    """지점 s 의 그날 좌표 — 시작일이 그날 이전(같은 날 포함)인 행 중 가장 최근 행, 없으면 가장 이른 행"""
    ok = [m for m in st_rows[s] if m['start'] <= day]
    m = ok[-1] if ok else st_rows[s][0]
    return (m['lat'], m['lon'])


changes = sorted({m['start'] for s in SIDS for m in st_rows[s] if DATE_MIN < m['start'] <= DATE_MAX})
bounds = [DATE_MIN] + changes + [DATE_MAX + np.timedelta64(1, 'D')]
PERIODS = []
for a, b in zip(bounds[:-1], bounds[1:]):
    PERIODS.append((int((a - DATE_MIN).astype(int)), int((b - DATE_MIN).astype(int)), np.array([xy_on(s, a) for s in SIDS], dtype=np.float64)))
print(f'  날짜 범위 안에서 지점 위치가 바뀌는 날 {len(changes)}번 → 기간 {len(PERIODS)}개: '
      + ', '.join(f'{d} ' + '·'.join(f'{s}' for s in SIDS if any(m['start'] == d for m in st_rows[s])) for d in changes))
print(f'  날짜 범위 첫날에 시작일이 그날 이전인 행이 없는 지점(가장 이른 행의 좌표를 쓰게 된다): {no_row_yet or "없음"}')


# ── 4. (날짜 × 지점) 표 ───────────────────────────────────────────────────────
sid_pos = {s: i for i, s in enumerate(SIDS)}
ci = np.array([sid_pos[s] for s in csv_sid], dtype=np.int64)
di_all = (csv_day - DATE_MIN).astype(np.int64)
keep = (di_all >= 0) & (di_all < N_DATES)
VALUES = {}
for c in SRC:
    V = np.full((N_DATES, len(SIDS)), np.nan, dtype=np.float32)
    V[di_all[keep], ci[keep]] = csv_val[c][keep].astype(np.float32)
    VALUES[c] = V
    assert int((~np.isnan(V)).sum()) == int((~np.isnan(csv_val[c][keep])).sum())
rows_day = np.zeros(N_DATES, dtype=np.int64)
np.add.at(rows_day, di_all[keep], 1)                                          # 날마다 CSV 행 수 (그날 행이 있는 지점 수)
print(f'날짜 범위 가운데 CSV 기간 밖인 날: 앞쪽 {max(int((SPI_START - DATE_MIN).astype(int)), 0)}일 · 뒤쪽 {max(int((DATE_MAX - SPI_END).astype(int)), 0)}일')


# ── 5. 날짜마다 가까운 지점 3곳 평균 (03_시군구_기상.py 와 같은 함수) ──────────────
def knn_mean(V, periods, loc_xy, k=K_NEAREST, block=BLOCK):
    """V: (날짜 × 지점) 값 표. 반환: 평균값 (좌표 × 날짜) float32, 고른 지점까지 평균 거리 km (좌표 × 날짜) float32, 고른 지점 수 (좌표 × 날짜) int8.
    날짜마다 그날 값 있는 지점 중 거리순 k 곳의 평균 (k 곳이 안 되면 있는 곳만, 없으면 NaN). 같은 거리면 지점 순서(SIDS)가 앞인 곳."""
    have = ~np.isnan(V).all(axis=0)              # 기간 중 값이 한 번이라도 있는 지점만 후보
    Vh = V[:, have]
    n_loc = len(loc_xy)
    M = np.full((n_loc, N_DATES), np.nan, dtype=np.float32)
    Dm = np.full((n_loc, N_DATES), np.nan, dtype=np.float32)
    Cn = np.zeros((n_loc, N_DATES), dtype=np.int8)
    if Vh.shape[1] == 0:
        return M, Dm, Cn
    for a, b, st_xy in periods:
        xy = st_xy[have]
        dist = haversine_km(loc_xy[:, None, 0], loc_xy[:, None, 1], xy[None, :, 0], xy[None, :, 1])   # (좌표, 지점) km
        order = np.argsort(dist, axis=1, kind='stable')
        dsort = np.take_along_axis(dist, order, axis=1)
        for s0 in range(a, b, block):
            s1 = min(s0 + block, b)
            vals = Vh[s0:s1][:, order]                        # (날짜, 좌표, 지점 거리순)
            ok = ~np.isnan(vals)
            sel = ok & (np.cumsum(ok, axis=2) <= k)           # 값 있는 곳 중 가까운 k 곳
            cnt = sel.sum(axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.where(sel, vals, 0).sum(axis=2, dtype=np.float64) / cnt
                dmean = (sel * dsort[None]).sum(axis=2) / cnt
            M[:, s0:s1] = mean.T
            Dm[:, s0:s1] = dmean.T
            Cn[:, s0:s1] = cnt.T
    return M, Dm, Cn


out_of_file = (DATES < SPI_START) | (DATES > SPI_END)
RESULT, CHECK_ROWS, CNT = {}, [], {}
for spec in VARIABLES:
    t1 = time.time()
    var, c = spec['var'], spec['src']
    M, D, Cn = knn_mean(VALUES[c], PERIODS, LOC_XY)
    RESULT[var], CNT[var] = M, Cn
    has = ~np.isnan(M)
    n_st = (~np.isnan(VALUES[c])).sum(axis=1)                                 # 날마다 그 지수 값이 있는 지점 수
    no_day = (~has).all(axis=0)                                               # 전 시군구가 결측인 날
    part_day = (~has).any(axis=0) & has.any(axis=0)                           # 일부 시군구만 결측인 날 (거리 상한이 없으므로 0 이어야 한다)
    assert not part_day.any(), f'{var}: 일부 시군구만 결측인 날이 있음 {DATES[part_day][:10]}'
    assert (no_day == (n_st == 0)).all(), f'{var}: 결측인 날 ≠ 값 있는 지점이 0곳인 날'
    in_file_no_row = no_day & ~out_of_file & (rows_day == 0)                   # CSV 기간 안인데 그날 행이 한 줄도 없는 날
    in_file_blank = no_day & ~out_of_file & (rows_day > 0)                     # 행은 있는데 이 지수가 전 지점 빈칸인 날
    dd = D[has].astype(np.float64)
    vv = M[has].astype(np.float64)
    cn = Cn[has]
    n_st_has = n_st[n_st > 0]
    CHECK_ROWS.append({
        '칼럼': var, '변수': spec['ko'], '원자료': f'{SPI_PATH.name} 의 {c}',
        '시군구 수': N_LOC, '날짜 수': N_DATES, '날짜 범위': f'{DATE_MIN} ~ {DATE_MAX}',
        '전체 칸': int(M.size), '값 있는 칸': int(has.sum()), '결측 칸': int((~has).sum()),
        '결측 칸 비율(%)': round(float((~has).mean()) * 100, 3),
        '전 시군구가 결측인 날 수': int(no_day.sum()),
        '  그 가운데 CSV 기간 밖인 날 수': int((no_day & out_of_file).sum()),
        '  그 가운데 CSV 기간 안인데 그날 행이 없는 날 수': int(in_file_no_row.sum()),
        '  그 가운데 행은 있는데 이 지수가 전 지점 빈칸인 날 수': int(in_file_blank.sum()),
        '일부 시군구만 결측인 날 수': int(part_day.sum()),
        '전 시군구가 결측인 날 목록(이어진 구간)': ' / '.join(day_ranges(DATES[no_day])),
        '그날 값 있는 지점 수 (값 있는 날 기준) 최소~최대': f'{int(n_st_has.min())}~{int(n_st_has.max())}' if n_st_has.size else '',
        '고른 지점이 3곳 미만인 칸': int((cn < K_NEAREST).sum()),
        '고른 지점까지 평균 거리 중앙값(km)': round(float(np.median(dd)), 2),
        '고른 지점까지 평균 거리 90%(km)': round(float(np.percentile(dd, 90)), 2),
        '고른 지점까지 평균 거리 최대(km)': round(float(dd.max()), 2),
        '평균 거리가 최대인 시군구': SGG_CODES[int(np.nanargmax(np.where(has, D, -1).max(axis=1)))],
        '평균 거리 50km 넘는 칸(%)': round(float((dd > 50).mean()) * 100, 2),
        '최솟값': round(float(vv.min()), 4), '중앙값': round(float(np.median(vv)), 4), '최댓값': round(float(vv.max()), 4),
        '원자료 지점 값 범위(날짜 범위 안)': f'{float(np.nanmin(VALUES[c])):.4f} ~ {float(np.nanmax(VALUES[c])):.4f}',
    })
    r = CHECK_ROWS[-1]
    print(f"{var}: 결측 {r['결측 칸 비율(%)']:.3f}% · 전 시군구 결측인 날 {r['전 시군구가 결측인 날 수']}일 "
          f"(CSV 기간 밖 {r['  그 가운데 CSV 기간 밖인 날 수']} · 그날 행 없음 {r['  그 가운데 CSV 기간 안인데 그날 행이 없는 날 수']} · 지수만 전 지점 빈칸 {r['  그 가운데 행은 있는데 이 지수가 전 지점 빈칸인 날 수']}) "
          f"· 거리 중앙값 {r['고른 지점까지 평균 거리 중앙값(km)']}km · 90% {r['고른 지점까지 평균 거리 90%(km)']}km · 최대 {r['고른 지점까지 평균 거리 최대(km)']}km "
          f"· 값 {r['최솟값']} ~ {r['최댓값']} ({time.time() - t1:.1f}초)")
    print(f"  전 시군구 결측인 날: {r['전 시군구가 결측인 날 목록(이어진 구간)']}")
    del D, has, dd, vv, cn


# ── 6. 원자료(CSV 행·META 행)에서 직접 다시 계산해 대조 — 표를 거치지 않는다 ─────────────
o = np.argsort(csv_day, kind='stable')
R_day, R_sid = csv_day[o], csv_sid[o]
R_val = {c: csv_val[c][o] for c in SRC}


def recompute(lat, lon, di, c):
    day = DATES[di]
    lo_, hi_ = np.searchsorted(R_day, day, 'left'), np.searchsorted(R_day, day, 'right')
    cand = []
    for j in range(lo_, hi_):
        v = R_val[c][j]
        if np.isnan(v):
            continue
        y, x = xy_on(R_sid[j], day)
        cand.append((float(haversine_km(np.float64(lat), np.float64(lon), np.float64(y), np.float64(x))), R_sid[j], float(np.float32(v))))
    cand.sort(key=lambda t: (t[0], t[1]))
    top = cand[:K_NEAREST]
    return float(np.mean([t[2] for t in top])) if top else np.nan


rng = np.random.default_rng(VERIFY_SEED)
n_cell = n_same = 0
bad = []
for spec in VARIABLES:
    var, c = spec['var'], spec['src']
    li = rng.integers(0, N_LOC, N_VERIFY)
    dix = rng.integers(0, N_DATES, N_VERIFY)
    # 위치가 바뀌는 날 앞뒤 · 결측인 날 · 첫날·끝날도 넣는다
    extra = sorted({0, N_DATES - 1} | {int((d - DATE_MIN).astype(int)) + k for d in changes for k in (-1, 0)}
                   | set(np.nonzero(np.isnan(RESULT[var]).all(axis=0))[0][:5].tolist()))
    li = np.concatenate([li, rng.integers(0, N_LOC, len(extra))])
    dix = np.concatenate([dix, np.array(extra, dtype=np.int64)])
    for l, d in zip(li.tolist(), dix.tolist()):
        v = recompute(LOC_XY[l, 0], LOC_XY[l, 1], d, c)
        s = float(RESULT[var][l, d])
        same = (np.isnan(v) and np.isnan(s)) or (not np.isnan(v) and not np.isnan(s) and abs(np.float32(v) - s) <= 1e-5 * max(1, abs(v)))
        n_cell += 1
        n_same += bool(same)
        if not same:
            bad.append((var, SGG_CODES[l], str(DATES[d]), v, s))
print(f'원자료 대조: {n_cell:,}칸 중 같은 칸 {n_same:,} · 다른 칸 {len(bad)}')
assert not bad, bad[:10]


# ── 7. 저장 ─────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
meta_out = {
    'description': '시군구 x 날짜 표준강수지수(SPI3·SPI6) — 시군구 대표 좌표에서 그날 값 있는 지점 중 구면 거리 가까운 3곳 단순 평균. 규칙은 05_시군구_SPI.py 문서 글 참고',
    'source_files': json.dumps([SPI_PATH.name, META_PATH.name, SGG_COORD.name], ensure_ascii=False),
    'date_min': str(DATE_MIN), 'date_max': str(DATE_MAX), 'spi_file_start': str(SPI_START), 'spi_file_end': str(SPI_END),
    'variables': json.dumps(VARIABLES, ensure_ascii=False), 'k_nearest': str(K_NEAREST), 'earth_r_km': str(EARTH_R_KM),
}
schema = pa.schema([pa.field('sgg_code', pa.string()), pa.field('date', pa.date32())] + [pa.field(v, pa.float32()) for v in VARS],
                   metadata={k: v.encode('utf-8') for k, v in meta_out.items()})
cols = {'sgg_code': pa.array(np.repeat(np.array(SGG_CODES, dtype=object), N_DATES).tolist(), pa.string()),
        'date': pa.array(np.tile(DATES, N_LOC), pa.date32())}
for v in VARS:
    cols[v] = pa.array(RESULT[v].ravel(), type=pa.float32(), from_pandas=True)     # (시군구 × 날짜) 를 행 순서대로 펼침 — NaN 은 null 로
table = pa.table(cols, schema=schema)
tmp = OUT_PATH.with_suffix('.parquet.part')
pq.write_table(table, tmp)
tmp.replace(OUT_PATH)

with open(CHECK_PATH, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(CHECK_ROWS[0].keys()))
    w.writeheader()
    w.writerows(CHECK_ROWS)

# 저장한 파일을 다시 열어 꼴 확인
chk = pq.read_table(OUT_PATH)
assert chk.num_rows == N_LOC * N_DATES and chk.schema.names == ['sgg_code', 'date'] + VARS
k = pc.binary_join_element_wise(chk['sgg_code'], pc.cast(chk['date'], pa.string()), '|')
assert pc.all(pc.greater(k[1:], k[:-1])).as_py(), 'sgg_code, date 순서가 아님 또는 겹치는 행이 있음'
for v in VARS:
    assert chk[v].null_count == int(np.isnan(RESULT[v]).sum())
    assert pc.sum(pc.is_nan(chk[v].drop_null())).as_py() == 0, f'{v}: NaN 이 null 로 바뀌지 않음'
print(f'저장: {OUT_PATH} ({chk.num_rows:,}행 × {chk.num_columns}칼럼, {OUT_PATH.stat().st_size / 1e6:.2f}MB)')
print(f'점검 표: {CHECK_PATH}')
print(f'걸린 시간 {time.time() - T0:.0f}초')
