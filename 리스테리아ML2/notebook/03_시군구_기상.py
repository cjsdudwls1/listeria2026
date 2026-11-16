# -*- coding: utf-8 -*-
"""
03_시군구_기상.py — 시군구 x 날짜 기상표 (기상·대기 17개 + 풍향 sin·cos 2개 = 19개 변수)

입력 (읽기만 함)
  - 시군구 대표 좌표 : 리스테리아ML2/data/데이터셋/시군구_대표좌표.parquet (sgg_code, lat, lon)
  - 시료(날짜 범위용) : 리스테리아ML2/data/데이터셋/시료_시군구.parquet (mnfctur_de 만 읽음)
  - 원자료 (01_원자료_받기.py 가 받은 것, 리스테리아ML2/data/raw/)
      raw_kma_asos_daily_info          ← api_data_info.kma_asos_daily_info
      raw_kma_ground_obs_station_info  ← api_data_info.kma_ground_obs_station_info
      raw_nas_agri_weather_daily_info  ← api_data_info.nas_agri_weather_daily_info
      raw_nas_agri_weather_obs_station ← api_data_info.nas_agri_weather_obs_station
      raw_keco_airkorea_fine_dust_info ← api_data_info.keco_airkorea_fine_dust_info
      raw_keco_airkorea_obs_station    ← api_data_info.keco_airkorea_obs_station

출력
  - 리스테리아ML2/data/데이터셋/식정원_db_기상기후/시군구_일별_기상.parquet
      sgg_code(string), date(date32), 변수 19개(float32, 결측은 null). 정렬: sgg_code, date
  - 리스테리아ML2/data/중간/시군구_기상_점검.csv (변수별 점검 표)

규칙 — 보관/리스테리아ML2_20261016/final_기상조인코드_사본.ipynb 4-1·4-2·4-3·4-5절, 5절(풍향)과 같다.
바뀐 것은 좌표뿐이다: 시료 좌표(lat·lon) → 시군구 대표 좌표.
  - 날짜마다, 그날 그 칼럼 값이 있는 관측소 중 구면 거리(지구 반지름 6371.0088km)로 가까운 3곳의 단순 평균.
    3곳이 안 되면 있는 곳만 평균, 한 곳도 없으면 결측. 거리 상한 없음. 같은 거리면 관측소 번호(이름) 순서가 앞인 곳.
  - 기본 칼럼 결과가 결측인 칸에만 대체 칼럼 결과를 넣는다.
  - 값은 원자료 그대로(결측 채움·단위 변환·이상값 처리 없음). ASOS sum_sshr 는 DB 에 텍스트라 숫자로만 바꾼다.
  - ASOS 좌표: kma_ground_obs_station_info 의 lat·lon. 날짜마다 "oper_start_dt 가 그날 이전인 행 중 가장 최근 행"
    (그런 행이 없으면 가장 이른 행).
  - NAS 좌표: nas_agri_weather_obs_station 의 instl_la·instl_lo (관측소당 1행). 동방로거테스트 561034A001 은 쓰지 않는다.
  - 에어코리아 좌표: 일자료 station_nm 으로 keco_airkorea_obs_station 의 dm_x(위도)·dm_y(경도). 같은 이름 여러 행은
    좌표가 같을 때만 한 행으로 쓰고(다르면 멈춤), 측정소 정보에 없는 측정소는 뺀다.
  - 같은 관측소·같은 날 행이 둘 이상이면 멈춘다.
  - 풍향: ASOS max_ws1(최다풍향, 도)을 관측소·날짜마다 sin·cos 로 바꾼 뒤 각각 가까운 3곳 평균(평균 뒤 길이 1로 맞추지 않음).
  - 날짜 범위: DATE_MIN = 시료_시군구.parquet 의 가장 이른 mnfctur_de − 365일, DATE_MAX = 세 관측망 일자료 중 가장 늦은 날짜.

다시 돌리면 같은 결과가 나온다(무작위 없음 — 점검용 표본 칸도 고정 시드).
실행: py "리스테리아ML2\\notebook\\03_시군구_기상.py"   (py 3.14 — numpy·pyarrow 사용, pandas 쓰지 않음)
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
RAW_DIR = BASE / 'data' / 'raw'
SGG_COORD = BASE / 'data' / '데이터셋' / '시군구_대표좌표.parquet'
SAMPLE_SGG = BASE / 'data' / '데이터셋' / '시료_시군구.parquet'
OUT_PATH = BASE / 'data' / '데이터셋' / '식정원_db_기상기후' / '시군구_일별_기상.parquet'
CHECK_PATH = BASE / 'data' / '중간' / '시군구_기상_점검.csv'

K_NEAREST = 3                 # 가까운 관측소 3곳 평균
EARTH_R_KM = 6371.0088        # 구면 거리 계산에 쓰는 지구 반지름(km)
DAYS_BEFORE_MIN = 365         # DATE_MIN = 가장 이른 mnfctur_de − 365일 (긴 window 실험 대비)
EXCLUDE_NAS = ['561034A001']  # NAS 동방로거테스트 — 쓰지 않는다 (사용자 결정 2026-10-06)
BLOCK = 32                    # 한 번에 계산하는 날짜 수 (메모리 때문에). 값에는 영향이 없다
N_VERIFY = 200                # 변수마다 원자료에서 직접 다시 계산해 대조하는 칸 수
VERIFY_SEED = 20261016

# 칼럼 이름 · 한글 이름 · 단위 · 기본 (관측망, 원자료 칼럼) · 대체 또는 None — 노트북 4-1절 VARIABLES + 5절 WIND_DIR_VARIABLES
VARIABLES = [
    dict(var='avg_temp',   ko='평균기온',      unit='°C',    prim=('ASOS', 'avg_ta'),       fb=('NAS', 'temp')),
    dict(var='max_temp',   ko='최고기온',      unit='°C',    prim=('ASOS', 'max_ta'),       fb=('NAS', 'hghst_artmp')),
    dict(var='min_temp',   ko='최저기온',      unit='°C',    prim=('ASOS', 'min_ta'),       fb=('NAS', 'lowst_artmp')),
    dict(var='rain',       ko='강수량',        unit='mm',    prim=('NAS', 'rn'),            fb=None),
    dict(var='avg_wind',   ko='평균풍속',      unit='m/s',   prim=('ASOS', 'avg_ws'),       fb=('NAS', 'wind')),
    dict(var='max_wind',   ko='최대풍속',      unit='m/s',   prim=('ASOS', 'max_ws'),       fb=('NAS', 'max_wind')),
    dict(var='sunshine',   ko='일조시간',      unit='시간',  prim=('ASOS', 'sum_sshr'),     fb=None),
    dict(var='solar_rad',  ko='일사량',        unit='MJ/m²', prim=('ASOS', 'sum_gsr'),      fb=('NAS', 'srqty')),
    dict(var='soil_t5',    ko='지중온도 5cm',  unit='°C',    prim=('ASOS', 'avg_cm5_te'),   fb=None),
    dict(var='soil_t10',   ko='지중온도 10cm', unit='°C',    prim=('ASOS', 'avg_cm10_te'),  fb=None),
    dict(var='soil_t20',   ko='지중온도 20cm', unit='°C',    prim=('ASOS', 'avg_cm20_tes'), fb=None),
    dict(var='soil_t30',   ko='지중온도 30cm', unit='°C',    prim=('ASOS', 'avg_cm30_te'),  fb=None),
    dict(var='soil_t_nas', ko='지중온도(NAS, 깊이 설명 없음)', unit='°C', prim=('NAS', 'soil_temp'), fb=None),
    dict(var='soil_moist', ko='토양수분',      unit='%',     prim=('NAS', 'soil_wt'),       fb=None),
    dict(var='no2',        ko='이산화질소',    unit='ppm',   prim=('AIR', 'no2_value'),     fb=None),
    dict(var='co',         ko='일산화탄소',    unit='ppm',   prim=('AIR', 'co_value'),      fb=None),
    dict(var='so2',        ko='아황산가스',    unit='ppm',   prim=('AIR', 'so2_value'),     fb=None),
    dict(var='wind_dir_sin', ko='최다풍향 sin', unit='(−1~1)', prim=('ASOS', 'max_ws1_sin'), fb=None),
    dict(var='wind_dir_cos', ko='최다풍향 cos', unit='(−1~1)', prim=('ASOS', 'max_ws1_cos'), fb=None),
]
VARS = [v['var'] for v in VARIABLES]
assert len(set(VARS)) == len(VARS) == 19
NET_TABLE = {'ASOS': 'api_data_info.kma_asos_daily_info', 'NAS': 'api_data_info.nas_agri_weather_daily_info',
             'AIR': 'api_data_info.keco_airkorea_fine_dust_info'}
NET_FILE = {'ASOS': ('raw_kma_asos_daily_info', 'stn_id', 'meas_date'),
            'NAS': ('raw_nas_agri_weather_daily_info', 'obsr_spot_code', 'meas_date'),
            'AIR': ('raw_keco_airkorea_fine_dust_info', 'station_nm', 'msur_dt')}


def haversine_km(lat1, lon1, lat2, lon2):
    """두 지점 사이의 구면 거리(km). 위도·경도는 도 단위 (노트북 1절과 같은 식)"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return EARTH_R_KM * 2 * np.arcsin(np.sqrt(a))


def day_num(arr):
    """date32 배열 → 1970-01-01 부터 며칠째 (int64)"""
    return pc.cast(arr, pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64)


def raw_cols(net):
    return sorted({v['prim'][1] for v in VARIABLES if v['prim'][0] == net and not v['prim'][1].startswith('max_ws1_')}
                  | {v['fb'][1] for v in VARIABLES if v['fb'] and v['fb'][0] == net})


def to_float(col, name):
    """원자료 칼럼 → float64 numpy (null 은 NaN). 텍스트 칼럼(sum_sshr)은 숫자로만 바꾼다 — 숫자로 안 바뀌는 값이 있으면 멈춘다"""
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        uniq = pc.unique(col.drop_null()).to_pylist()
        conv = {}
        for s in uniq:
            try:
                conv[s] = float(s)
            except ValueError:
                raise ValueError(f'{name}: 숫자로 안 바뀌는 값 {s!r}')
        vals = col.to_pylist()
        return np.array([np.nan if s is None else conv[s] for s in vals], dtype=np.float64), len(uniq)
    return col.to_numpy(zero_copy_only=False).astype(np.float64), None


# ── 1. 시군구 좌표와 날짜 범위 ───────────────────────────────────────────────
sgg = pq.read_table(SGG_COORD, columns=['sgg_code', 'lat', 'lon'])
order = np.argsort(np.array(sgg['sgg_code'].to_pylist(), dtype=object), kind='stable')
SGG_CODES = [sgg['sgg_code'].to_pylist()[i] for i in order]
LOC_XY = np.column_stack([sgg['lat'].to_numpy()[order], sgg['lon'].to_numpy()[order]]).astype(np.float64)
assert len(set(SGG_CODES)) == len(SGG_CODES), 'sgg_code 가 겹침'
assert all(c is not None and len(c) == 5 for c in SGG_CODES), 'sgg_code 가 5자리 글자가 아님'
assert not np.isnan(LOC_XY).any(), '좌표가 빈 시군구가 있음'
N_LOC = len(SGG_CODES)

mn = pq.read_table(SAMPLE_SGG, columns=['mnfctur_de', 'sgg_code'])
de = mn['mnfctur_de']
assert de.null_count == 0
de_min = pc.min(de).as_py()
assert (de_min.hour, de_min.minute, de_min.second, de_min.microsecond) == (0, 0, 0, 0), '가장 이른 mnfctur_de 에 시각이 있음'
de_min_sgg = pc.min(pc.filter(de, pc.is_valid(mn['sgg_code']))).as_py()
DATE_MIN = np.datetime64(de_min.date(), 'D') - np.timedelta64(DAYS_BEFORE_MIN, 'D')
net_max = {}
for net, (fname, _, dcol) in NET_FILE.items():
    net_max[net] = pc.max(pq.read_table(RAW_DIR / f'{fname}.parquet', columns=[dcol])[dcol]).as_py()
DATE_MAX = np.datetime64(max(net_max.values()), 'D')
DATES = np.arange(DATE_MIN, DATE_MAX + np.timedelta64(1, 'D'), dtype='datetime64[D]')
N_DATES = len(DATES)
D0 = int(DATE_MIN.astype(np.int64))       # 1970-01-01 부터 며칠째 (DATE_MIN)
print(f'시군구 {N_LOC}곳 · 시료_시군구 가장 이른 mnfctur_de {de_min.date()} (sgg_code 있는 행만 보면 {de_min_sgg.date()})')
print(f'일자료 가장 늦은 날짜: ' + ', '.join(f'{k} {v}' for k, v in net_max.items()))
print(f'DATE_MIN {DATE_MIN} (= {de_min.date()} − {DAYS_BEFORE_MIN}일) · DATE_MAX {DATE_MAX} · 날짜 {N_DATES:,}일 → {N_LOC} × {N_DATES:,} = {N_LOC * N_DATES:,}행')


# ── 2. 관측망별 (날짜 × 관측소) 표와 관측소 좌표 (노트북 4-2절) ─────────────────
def daily_table(tbl, sid_col, date_col, value_cols, sids, label):
    """(날짜 × 관측소) float32 표 — DATE_MIN~DATE_MAX 전체, 관측소 순서 sids. 같은 관측소·같은 날 행이 둘 이상이면 멈춘다(원자료 전체에서 확인)"""
    sid_all = tbl[sid_col]
    days = day_num(tbl[date_col])
    codes = pc.index_in(sid_all, value_set=pc.unique(sid_all)).to_numpy(zero_copy_only=False).astype(np.int64)
    key = codes * 1_000_000 + (days - days.min())
    n_dup = len(key) - len(np.unique(key))
    if n_dup:
        raise RuntimeError(f'{label}: 같은 관측소·같은 날 행이 {n_dup}건 있음 — 어느 값을 쓸지 정해야 함')
    pos = pc.index_in(sid_all, value_set=pa.array(sids, pa.string())).to_numpy(zero_copy_only=False)
    keep = (~np.isnan(pos.astype(np.float64))) & (days >= D0) & (days <= D0 + N_DATES - 1)
    ri = (days[keep] - D0).astype(np.int64)
    ci = pos[keep].astype(np.int64)
    out, info = {}, {}
    for c in value_cols:
        v64, n_text = to_float(tbl[c], f'{label}.{c}')
        V = np.full((N_DATES, len(sids)), np.nan, dtype=np.float32)
        V[ri, ci] = v64[keep].astype(np.float32)
        out[c] = V
        if n_text is not None:
            info[c] = n_text
    return out, info


NETS = {}   # 관측망 → {'sids', 'periods': [(시작 날짜번호, 끝 날짜번호, 좌표 배열)], 'values': {칼럼: 표}}

# ASOS — 날짜별 위치
asos_cols = raw_cols('ASOS') + ['max_ws1']
asos = pq.read_table(RAW_DIR / 'raw_kma_asos_daily_info.parquet', columns=['stn_id', 'meas_date'] + asos_cols)
asos_st = pq.read_table(RAW_DIR / 'raw_kma_ground_obs_station_info.parquet', columns=['stn_id', 'lat', 'lon', 'oper_start_dt']).to_pylist()
sids = sorted(pc.unique(asos['stn_id']).to_pylist())
st_rows = {}
for r in asos_st:
    if r['stn_id'] in set(sids):
        st_rows.setdefault(r['stn_id'], []).append(r)
assert set(st_rows) == set(sids), 'ASOS 좌표가 없는 관측소가 있음'
for s in sids:
    starts = [r['oper_start_dt'] for r in st_rows[s]]
    assert None not in starts and len(set(starts)) == len(starts), f'ASOS {s}: 운영 시작일이 비었거나 겹침'
    st_rows[s].sort(key=lambda r: r['oper_start_dt'])
d_min_py, d_max_py = DATE_MIN.astype(object), DATE_MAX.astype(object)
changes = sorted({r['oper_start_dt'] for s in sids for r in st_rows[s] if d_min_py < r['oper_start_dt'] <= d_max_py})
bounds = [d_min_py] + changes + [d_max_py + (DATES[1] - DATES[0]).astype(object)]
periods = []
for a, b in zip(bounds[:-1], bounds[1:]):
    xy = []
    for s in sids:
        ok = [r for r in st_rows[s] if r['oper_start_dt'] <= a]
        r = ok[-1] if ok else st_rows[s][0]
        xy.append((float(r['lat']), float(r['lon'])))
    periods.append(((np.datetime64(a, 'D') - DATE_MIN).astype(int), min((np.datetime64(b, 'D') - DATE_MIN).astype(int), N_DATES), np.array(xy)))
vals, info = daily_table(asos, 'stn_id', 'meas_date', asos_cols, sids, 'ASOS')
NETS['ASOS'] = {'sids': sids, 'periods': periods, 'values': vals}
print(f'ASOS: 관측소 {len(sids)}곳 · 위치가 바뀌는 날 {len(changes)}번 → 기간 {len(periods)}개 · sum_sshr 텍스트 값 종류 {info.get("sum_sshr")}개 모두 숫자로 바뀜')
del asos

# 풍향 — max_ws1(도) → sin·cos (관측소·날짜마다, NaN 은 NaN 그대로) (노트북 5절)
V_deg = NETS['ASOS']['values']['max_ws1']
deg_vals = np.unique(V_deg[~np.isnan(V_deg)])
rad = np.deg2rad(V_deg.astype(np.float64))
NETS['ASOS']['values']['max_ws1_sin'] = np.sin(rad).astype(np.float32)
NETS['ASOS']['values']['max_ws1_cos'] = np.cos(rad).astype(np.float32)
del rad
print(f'  max_ws1 값 종류(도): {deg_vals.astype(int).tolist()}')

# NAS — 관측소당 좌표 1개
nas_cols = raw_cols('NAS')
nas = pq.read_table(RAW_DIR / 'raw_nas_agri_weather_daily_info.parquet', columns=['obsr_spot_code', 'meas_date'] + nas_cols)
nas_st = pq.read_table(RAW_DIR / 'raw_nas_agri_weather_obs_station.parquet', columns=['obsr_spot_code', 'instl_la', 'instl_lo']).to_pylist()
nas_xy = {}
for r in nas_st:
    assert r['obsr_spot_code'] not in nas_xy, 'NAS 관측지점 정보에 같은 지점이 두 행'
    nas_xy[r['obsr_spot_code']] = (r['instl_la'], r['instl_lo'])
sids = [s for s in sorted(pc.unique(nas['obsr_spot_code']).to_pylist()) if s not in EXCLUDE_NAS]
assert set(sids) <= set(nas_xy), 'NAS 좌표가 없는 관측소가 있음'
xy = np.array([nas_xy[s] for s in sids], dtype=np.float64)
assert not np.isnan(xy).any()
vals, _ = daily_table(nas, 'obsr_spot_code', 'meas_date', nas_cols, sids, 'NAS')
NETS['NAS'] = {'sids': sids, 'periods': [(0, N_DATES, xy)], 'values': vals}
print(f'NAS: 관측소 {len(sids)}곳 (561034A001 제외 — 포함 여부 {"561034A001" in sids})')
del nas

# 에어코리아 — 측정소 이름으로 좌표 연결
air_cols = raw_cols('AIR')
air = pq.read_table(RAW_DIR / 'raw_keco_airkorea_fine_dust_info.parquet', columns=['station_nm', 'msur_dt'] + air_cols)
air_st = pq.read_table(RAW_DIR / 'raw_keco_airkorea_obs_station.parquet', columns=['station_nm', 'dm_x', 'dm_y']).to_pylist()
air_xy = {}
for r in air_st:
    assert r['dm_x'] is not None and r['dm_y'] is not None and 33 <= r['dm_x'] <= 39 and 124 <= r['dm_y'] <= 132, '에어코리아 좌표 범위가 예상(dm_x 위도·dm_y 경도)과 다름'
    xy_r = (r['dm_x'], r['dm_y'])
    if r['station_nm'] in air_xy:
        assert air_xy[r['station_nm']] == xy_r, f"같은 이름인데 좌표가 다른 측정소: {r['station_nm']}"
    air_xy[r['station_nm']] = xy_r
all_air = sorted(pc.unique(air['station_nm']).to_pylist())
sids = [s for s in all_air if s in air_xy]
no_xy = [s for s in all_air if s not in air_xy]
vals, _ = daily_table(air, 'station_nm', 'msur_dt', air_cols, sids, 'AIR')
NETS['AIR'] = {'sids': sids, 'periods': [(0, N_DATES, np.array([air_xy[s] for s in sids], dtype=np.float64))], 'values': vals}
print(f'에어코리아: 일자료 측정소 {len(all_air)}곳 중 좌표 있는 {len(sids)}곳 사용, 좌표 없어 뺀 {len(no_xy)}곳: {no_xy}')
del air
print(f'원자료 읽기 끝 ({time.time() - T0:.0f}초)')


# ── 3. 날짜마다 가까운 관측소 3곳 평균 (노트북 4-3절) ───────────────────────────
def knn_mean(V, periods, loc_xy, k=K_NEAREST, block=BLOCK):
    """V: (날짜 × 관측소) 값 표. 반환: 평균값 (좌표 × 날짜) float32, 고른 관측소까지 평균 거리 km (좌표 × 날짜) float32.
    날짜마다 그날 값 있는 관측소 중 거리순 k 곳의 평균 (k 곳이 안 되면 있는 곳만, 없으면 NaN). 같은 거리면 관측소 순서(sids)가 앞인 곳.
    노트북은 속도 때문에 거리순 60곳만 먼저 보고 모자라면 전체에서 다시 골랐다 — 여기서는 처음부터 전체 관측소를 보므로 결과가 같다"""
    have = ~np.isnan(V).all(axis=0)              # 기간 중 값이 한 번이라도 있는 관측소만 후보
    Vh = V[:, have]
    n_loc = len(loc_xy)
    M = np.full((n_loc, N_DATES), np.nan, dtype=np.float32)
    Dm = np.full((n_loc, N_DATES), np.nan, dtype=np.float32)
    if Vh.shape[1] == 0:
        return M, Dm
    for a, b, st_xy in periods:
        xy = st_xy[have]
        dist = haversine_km(loc_xy[:, None, 0], loc_xy[:, None, 1], xy[None, :, 0], xy[None, :, 1])   # (좌표, 관측소) km
        order = np.argsort(dist, axis=1, kind='stable')
        dsort = np.take_along_axis(dist, order, axis=1)
        for s0 in range(a, b, block):
            s1 = min(s0 + block, b)
            vals = Vh[s0:s1][:, order]                        # (날짜, 좌표, 관측소 거리순)
            ok = ~np.isnan(vals)
            sel = ok & (np.cumsum(ok, axis=2) <= k)           # 값 있는 곳 중 가까운 k 곳
            cnt = sel.sum(axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.where(sel, vals, 0).sum(axis=2, dtype=np.float64) / cnt
                dmean = (sel * dsort[None]).sum(axis=2) / cnt
            M[:, s0:s1] = mean.T
            Dm[:, s0:s1] = dmean.T
    return M, Dm


RESULT, CHECK_ROWS, FB_MASK = {}, [], {}
i2015 = int((np.datetime64('2015-01-01') - DATE_MIN).astype(int))
for spec in VARIABLES:
    t1 = time.time()
    var, prim, fb = spec['var'], spec['prim'], spec['fb']
    net = NETS[prim[0]]
    M, D = knn_mean(net['values'][prim[1]], net['periods'], LOC_XY)
    n_fb = 0
    use_fb = np.zeros_like(M, dtype=bool)
    if fb:
        netf = NETS[fb[0]]
        Mf, Df = knn_mean(netf['values'][fb[1]], netf['periods'], LOC_XY)
        use_fb = np.isnan(M) & ~np.isnan(Mf)      # 기본 칼럼 결과가 빈 칸에만 대체 칼럼 결과
        n_fb = int(use_fb.sum())
        M[use_fb] = Mf[use_fb]
        D[use_fb] = Df[use_fb]
        del Mf, Df
    FB_MASK[var] = use_fb
    RESULT[var] = M
    has = ~np.isnan(M)
    dd = D[has].astype(np.float64)
    vv = M[has].astype(np.float64)
    no_day = (~has).all(axis=0)
    fb_days = DATES[use_fb.any(axis=0)]
    CHECK_ROWS.append({
        '칼럼': var, '변수': spec['ko'], '단위': spec['unit'],
        '기본': f'{NET_TABLE[prim[0]]}.{prim[1] if not prim[1].startswith("max_ws1_") else "max_ws1 (" + prim[1][-3:] + ")"}',
        '대체': f'{NET_TABLE[fb[0]]}.{fb[1]}' if fb else '없음',
        '전체 칸': int(M.size), '값 있는 칸': int(has.sum()),
        '값 있는 칸 비율(%)': round(float(has.mean()) * 100, 3),
        '2015-01-01 이후 값 있는 칸 비율(%)': round(float(has[:, i2015:].mean()) * 100, 3),
        '모든 시군구가 결측인 날 수': int(no_day.sum()),
        '2015-01-01 이후 모든 시군구가 결측인 날 수': int(no_day[i2015:].sum()),
        '대체 칼럼 값이 들어간 칸': n_fb,
        '대체 칼럼 값이 들어간 날': f'{len(fb_days)}일 ({fb_days.min()} ~ {fb_days.max()})' if len(fb_days) else '0일',
        '고른 관측소까지 평균 거리 중앙값(km)': round(float(np.median(dd)), 2) if dd.size else None,
        '고른 관측소까지 평균 거리 90%(km)': round(float(np.percentile(dd, 90)), 2) if dd.size else None,
        '고른 관측소까지 평균 거리 최대(km)': round(float(dd.max()), 2) if dd.size else None,
        '평균 거리 50km 넘는 칸(%)': round(float((dd > 50).mean()) * 100, 2) if dd.size else None,
        '최솟값': round(float(vv.min()), 4) if vv.size else None,
        '중앙값': round(float(np.median(vv)), 4) if vv.size else None,
        '최댓값': round(float(vv.max()), 4) if vv.size else None,
    })
    del D, has, dd, vv
    r = CHECK_ROWS[-1]
    print(f"{var:13s} 값 있음 {r['값 있는 칸 비율(%)']:7.3f}% (2015~ {r['2015-01-01 이후 값 있는 칸 비율(%)']:7.3f}%) · 대체 {n_fb:,}칸 · "
          f"거리 중앙값 {r['고른 관측소까지 평균 거리 중앙값(km)']}km · 90% {r['고른 관측소까지 평균 거리 90%(km)']}km · 최대 {r['고른 관측소까지 평균 거리 최대(km)']}km "
          f"· 범위 {r['최솟값']} ~ {r['최댓값']} ({time.time() - t1:.1f}초)")


# ── 4. 원자료에서 직접 다시 계산해 대조 (표를 거치지 않고 원자료 행에서 바로) ───────────────
def build_raw_by_date(net):
    fname, sid_col, dcol = NET_FILE[net]
    cols = raw_cols(net) + (['max_ws1'] if net == 'ASOS' else [])
    t = pq.read_table(RAW_DIR / f'{fname}.parquet', columns=[sid_col, dcol] + cols)
    days = day_num(t[dcol])
    o = np.argsort(days, kind='stable')
    sid = np.array(t[sid_col].to_pylist(), dtype=object)[o]
    vals = {c: to_float(t[c], c)[0][o] for c in cols}
    return {'days': days[o], 'sid': sid, 'vals': vals}


RAW = {net: build_raw_by_date(net) for net in NET_FILE}
ASOS_ST_ALL = {}
for r in asos_st:
    ASOS_ST_ALL.setdefault(r['stn_id'], []).append(r)


def station_xy_raw(net, sid, day):
    """원자료 관측소 정보에서 그날의 좌표 (없으면 None)"""
    if net == 'ASOS':
        rows = sorted(ASOS_ST_ALL.get(sid, []), key=lambda r: r['oper_start_dt'])
        if not rows:
            return None
        ok = [r for r in rows if r['oper_start_dt'] <= day]
        r = ok[-1] if ok else rows[0]
        return (r['lat'], r['lon'])
    if net == 'NAS':
        return None if sid in EXCLUDE_NAS else nas_xy.get(sid)
    return air_xy.get(sid)


def recompute(lat, lon, di, net, col):
    """한 칸 (좌표, 날짜번호) 을 원자료에서 직접 계산 — 그날 값 있는 관측소 중 구면 거리 가까운 3곳 평균"""
    R = RAW[net]
    dayn = D0 + di
    lo_, hi_ = np.searchsorted(R['days'], dayn, 'left'), np.searchsorted(R['days'], dayn, 'right')
    src = 'max_ws1' if col.startswith('max_ws1_') else col
    day = DATES[di].astype(object)
    cand = []
    for j in range(lo_, hi_):
        v = R['vals'][src][j]
        if np.isnan(v):
            continue
        xy_ = station_xy_raw(net, R['sid'][j], day)
        if xy_ is None:
            continue
        if col.endswith('_sin'):
            v = float(np.float32(np.sin(np.deg2rad(v))))
        elif col.endswith('_cos'):
            v = float(np.float32(np.cos(np.deg2rad(v))))
        else:
            v = float(np.float32(v))
        cand.append((float(haversine_km(np.float64(lat), np.float64(lon), np.float64(xy_[0]), np.float64(xy_[1]))), R['sid'][j], v))
    cand.sort(key=lambda x: (x[0], x[1]))
    top = cand[:K_NEAREST]
    return (float(np.mean([x[2] for x in top])) if top else np.nan), len(top)


rng = np.random.default_rng(VERIFY_SEED)
n_cell = n_same = 0
bad = []
for spec in VARIABLES:
    var = spec['var']
    li = rng.integers(0, N_LOC, N_VERIFY)
    di = rng.integers(0, N_DATES, N_VERIFY)
    fbm = np.argwhere(FB_MASK[var])                          # 대체 칼럼 값이 들어간 칸도 몇 개 넣는다
    if len(fbm):
        pick = fbm[rng.integers(0, len(fbm), min(20, len(fbm)))]
        li, di = np.concatenate([li, pick[:, 0]]), np.concatenate([di, pick[:, 1]])
    for l, d in zip(li.tolist(), di.tolist()):
        v, _ = recompute(LOC_XY[l, 0], LOC_XY[l, 1], d, *spec['prim'])
        if np.isnan(v) and spec['fb']:
            v, _ = recompute(LOC_XY[l, 0], LOC_XY[l, 1], d, *spec['fb'])
        s = float(RESULT[var][l, d])
        same = (np.isnan(v) and np.isnan(s)) or (not np.isnan(v) and not np.isnan(s) and abs(np.float32(v) - s) <= 1e-5 * max(1, abs(v)))
        n_cell += 1
        n_same += bool(same)
        if not same:
            bad.append((var, SGG_CODES[l], str(DATES[d]), v, s))
print(f'원자료 대조: {n_cell:,}칸 중 같은 칸 {n_same:,} · 다른 칸 {len(bad)}')
assert not bad, bad[:10]
del RAW


# ── 5. 저장 ─────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
meta = {
    'description': '시군구 x 날짜 기상표 — 시군구 대표 좌표에서 그날 값 있는 관측소 중 구면 거리 가까운 3곳 단순 평균. 규칙은 03_시군구_기상.py 문서 글 참고',
    'date_min': str(DATE_MIN), 'date_max': str(DATE_MAX),
    'variables': json.dumps([{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()} for s in VARIABLES], ensure_ascii=False),
    'exclude_nas': json.dumps(EXCLUDE_NAS), 'k_nearest': str(K_NEAREST), 'earth_r_km': str(EARTH_R_KM),
}
schema = pa.schema([pa.field('sgg_code', pa.string()), pa.field('date', pa.date32())] + [pa.field(v, pa.float32()) for v in VARS],
                   metadata={k: v.encode('utf-8') for k, v in meta.items()})
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
print(f'저장: {OUT_PATH} ({chk.num_rows:,}행 × {chk.num_columns}칼럼, {OUT_PATH.stat().st_size / 1e6:.1f}MB)')
print(f'점검 표: {CHECK_PATH}')
print(f'걸린 시간 {time.time() - T0:.0f}초')
