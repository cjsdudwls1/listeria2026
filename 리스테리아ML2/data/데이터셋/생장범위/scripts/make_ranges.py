# -*- coding: utf-8 -*-
"""
make_ranges.py — 리스테리아 모노사이토제네스 '생장 최적·한계 범위'를 변수(rg_*)로 만드는 스크립트

근거 (값은 그림에 적힌 그대로 씀)
  - 리서치팀 예측인자 정리 ppt(0.3 리스테리아 모노사이토제네스)
      = 문서/리서치팀문서/예측인자 정리 ppt.jpg 의 표 '리스테리아 모노사이토제네스 예측인자 생장 최적·한계 범위'
        온도(기온) 최적 30~37°C · 한계 −2~45°C / 토양 온도 최적 8~15°C · 한계 8~30°C /
        토양 pH 최적 6.5~7.0 · 한계 5.0~8.0 / 수분활성도 최적 0.97 · 한계 0.90~0.997 /
        pH 최적 7.0 · 한계 4.0~9.6 / 염도 최적 '-' · 한계 ≤16%
  - 값의 원래 논문 출처: 그림에 출처 없음.

입력 (읽기만 함, 원본 수정 없음)
  - 리스테리아ML2/data/데이터셋/식정원_db_기상기후/시군구_일별_기상.parquet : sgg_code, date, avg_temp, max_temp, min_temp, soil_t5
  - 리스테리아ML2/data/데이터셋/식정원_db_토양/시군구_일별_토양.parquet     : sgg_code, date, soil_ph (soil_ph_n 등 검정 건수 칸은 쓰지 않음)
  - 리스테리아ML2/data/데이터셋/식정원_db_식품/품목3단계_식품속성.parquet   : train_prdlst_code_2, train_prdlst_nm_2, train_prdlst_code, aw, aw_label, ph, ph_label

출력 (모두 리스테리아ML2/data/데이터셋/생장범위/)
  - 생장범위_일별_기상.parquet   : sgg_code, date + rg_tmax_opt, rg_tavg_opt_dist, rg_t_out_lim, rg_soilt_opt, rg_soilt_lim, rg_soilt_opt_dist
  - 생장범위_일별_토양pH.parquet : sgg_code, date + rg_soilph_opt, rg_soilph_lim, rg_soilph_opt_dist
  - 생장범위_품목3단계.parquet   : train_prdlst_code_2 + rg_aw_lim, rg_aw_opt_dist, rg_ph_lim, rg_ph_opt_dist
  - 생장범위_정의.csv (utf-8-sig) : 칸마다 뜻·입력 칼럼·범위 값·근거·결측 규칙·선택 사항
  - 생장범위_부등호표기.csv (utf-8-sig) : 품목 3단계 대표값의 표기가 부등호(초과·이상·미만·이하)인 품목 목록
  일별 두 표의 행 순서는 시군구_일별_기상.parquet 과 같다(sgg_code, date 순). rg_* 칸은 모두 float32, 결측은 null.

규칙
  - 결측 입력이면 결과도 결측(null). 0 으로 채우지 않는다.
  - 범위 경계값은 포함(이상·이하). 0/1 칸은 범위 안이면 1, 밖이면 0.
  - '거리' = 범위 안이면 0, 밖이면 가까운 경계까지의 절댓값 차. 최적값이 한 점(수분활성도 0.97, 식품 pH 7.0)이면 |값 − 최적값|.
  - 토양 온도는 5cm 깊이(soil_t5) 하나만 씀 — 그림에 깊이가 없어서 표면에 가장 가까운 깊이를 고름.
  - rg_t_out_lim 은 min_temp·max_temp 두 칸을 함께 보므로, 둘 중 하나라도 결측이면 결측.
  - 염도(na) 변수(rg_na_lim)는 만들지 않음: na 칼럼의 단위를 문서로 확인하지 못함(DB 코멘트·테이블정의서·리서치팀 엑셀 머리글 모두 '염도'만 있고 단위 없음).
  - 식품 값은 품목3단계_식품속성.parquet 의 숫자를 그대로 씀(표기가 '중앙'·'초과'·'이상'·'미만'·'이하'여도 숫자 그대로).
  - 창 요약(T−50..T−1 등)은 하지 않음 — 일별 값만 만든다(창 요약은 실험 세션이 함).

실행: "리스테리아ML\\.venv-exp2\\Scripts\\python.exe" "리스테리아ML2\\data\\데이터셋\\생장범위\\scripts\\make_ranges.py"
      (numpy·pyarrow 만 사용, 단일 프로세스, 무작위 없음 — 다시 돌리면 같은 결과)
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
DS = BASE / 'data' / '데이터셋'
IN_WX = DS / '식정원_db_기상기후' / '시군구_일별_기상.parquet'
IN_SOIL = DS / '식정원_db_토양' / '시군구_일별_토양.parquet'
IN_FOOD = DS / '식정원_db_식품' / '품목3단계_식품속성.parquet'
OUT_DIR = DS / '생장범위'
OUT_WX = OUT_DIR / '생장범위_일별_기상.parquet'
OUT_SOIL = OUT_DIR / '생장범위_일별_토양pH.parquet'
OUT_FOOD = OUT_DIR / '생장범위_품목3단계.parquet'
OUT_DEF = OUT_DIR / '생장범위_정의.csv'
OUT_SIGN = OUT_DIR / '생장범위_부등호표기.csv'

FIG = '리서치팀 예측인자 정리 ppt(0.3 리스테리아 모노사이토제네스)'
FIG_FILE = '문서/리서치팀문서/예측인자 정리 ppt.jpg'
FIG_TABLE = "표 '리스테리아 모노사이토제네스 예측인자 생장 최적·한계 범위'"
ORIG_SRC = '그림에 출처 없음'
SCRIPT = '리스테리아ML2/data/데이터셋/생장범위/scripts/make_ranges.py'

# 그림 값 (그대로)
T_OPT = (30.0, 37.0)        # 온도(기온) 최적 30~37°C
T_LIM = (-2.0, 45.0)        # 온도(기온) 한계 −2~45°C
ST_OPT = (8.0, 15.0)        # 토양 온도 최적 8~15°C
ST_LIM = (8.0, 30.0)        # 토양 온도 한계 8~30°C
SPH_OPT = (6.5, 7.0)        # 토양 pH 최적 6.5~7.0
SPH_LIM = (5.0, 8.0)        # 토양 pH 한계 5.0~8.0
AW_OPT = 0.97               # 수분활성도 최적 0.97
AW_LIM = (0.90, 0.997)      # 수분활성도 한계 0.90~0.997
PH_OPT = 7.0                # pH(식품) 최적 7.0
PH_LIM = (4.0, 9.6)         # pH(식품) 한계 4.0~9.6
NA_LIM_MAX = 16.0           # 염도 한계 ≤16% (단위 미확인으로 쓰지 않음)

SIGN_LABELS = ('초과', '이상', '미만', '이하')


def log(*a):
    print(f'[{time.time() - T0:5.1f}초]', *a, flush=True)


def to_np(col):
    """pyarrow 숫자 칸 → float64 numpy (null 은 NaN). float32 값은 float64 로 정확히 옮겨진다."""
    return pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False)


def in_range(x, lo, hi):
    """lo ≤ x ≤ hi 이면 1, 아니면 0, x 결측이면 NaN"""
    out = ((x >= lo) & (x <= hi)).astype(np.float64)
    out[np.isnan(x)] = np.nan
    return out


def dist_range(x, lo, hi):
    """범위 안 0, 밖이면 가까운 경계까지의 절댓값 차, x 결측이면 NaN"""
    out = np.where(x < lo, lo - x, np.where(x > hi, x - hi, 0.0))
    out[np.isnan(x)] = np.nan
    return out


def to_pa32(v):
    """float64 numpy → float32 pyarrow (NaN → null)"""
    return pa.array(v.astype(np.float32), type=pa.float32(), from_pandas=True)


def n_nan_not_null(col):
    """null 이 아닌데 NaN 인 값 수 (입력 점검용)"""
    return int(pc.sum(pc.is_nan(pc.drop_null(col))).as_py() or 0)


# ---------------------------------------------------------------- 1) 일별 기상 범위 표
wx = pq.read_table(IN_WX, columns=['sgg_code', 'date', 'avg_temp', 'max_temp', 'min_temp', 'soil_t5'])
N = wx.num_rows
assert wx.schema.field('date').type == pa.date32()
log(f'시군구_일별_기상 {N:,}행 · 시군구 {pc.count_distinct(wx["sgg_code"]).as_py()}곳 · '
    f'날짜 {pc.min(wx["date"]).as_py()} ~ {pc.max(wx["date"]).as_py()}')
# 키: 빈 값 없음·중복 없음·(sgg_code, date) 순 정렬
assert wx['sgg_code'].null_count == 0 and wx['date'].null_count == 0
key_sorted = wx.select(['sgg_code', 'date']).sort_by([('sgg_code', 'ascending'), ('date', 'ascending')])
assert key_sorted.equals(wx.select(['sgg_code', 'date'])), '기상표가 sgg_code, date 순이 아님'
n_keys = wx.group_by(['sgg_code', 'date']).aggregate([]).num_rows
assert n_keys == N, f'기상표 키 중복 {N - n_keys}'
for c in ['avg_temp', 'max_temp', 'min_temp', 'soil_t5']:
    nn = n_nan_not_null(wx[c])
    log(f'  입력 {c}: null {wx[c].null_count:,} ({100 * wx[c].null_count / N:.3f}%) · null 아닌 NaN {nn}')
    assert nn == 0, f'{c} 에 null 이 아닌 NaN 이 있음 — 결측 규칙 다시 정해야 함'

tavg, tmax, tmin, st5 = (to_np(wx[c]) for c in ['avg_temp', 'max_temp', 'min_temp', 'soil_t5'])

t_out = ((tmin < T_LIM[0]) | (tmax > T_LIM[1])).astype(np.float64)
one_missing = np.isnan(tmin) ^ np.isnan(tmax)
t_out[np.isnan(tmin) | np.isnan(tmax)] = np.nan
log(f'  rg_t_out_lim: min_temp·max_temp 중 하나만 결측인 행 {int(one_missing.sum()):,} '
    f'(그중 남은 한 칸만으로 한계 밖인 행 {int((one_missing & ((tmin < T_LIM[0]) | (tmax > T_LIM[1]))).sum()):,}) → 규칙대로 결측')

WX_COLS = {
    'rg_tmax_opt': in_range(tmax, *T_OPT),
    'rg_tavg_opt_dist': dist_range(tavg, *T_OPT),
    'rg_t_out_lim': t_out,
    'rg_soilt_opt': in_range(st5, *ST_OPT),
    'rg_soilt_lim': in_range(st5, *ST_LIM),
    'rg_soilt_opt_dist': dist_range(st5, *ST_OPT),
}
# 결측 그대로 옮겨졌는지
assert np.array_equal(np.isnan(WX_COLS['rg_tmax_opt']), np.isnan(tmax))
assert np.array_equal(np.isnan(WX_COLS['rg_tavg_opt_dist']), np.isnan(tavg))
for c in ['rg_soilt_opt', 'rg_soilt_lim', 'rg_soilt_opt_dist']:
    assert np.array_equal(np.isnan(WX_COLS[c]), np.isnan(st5))

out_wx = pa.table({'sgg_code': wx['sgg_code'], 'date': wx['date'], **{k: to_pa32(v) for k, v in WX_COLS.items()}})


# ---------------------------------------------------------------- 2) 일별 토양 pH 범위 표
soil = pq.read_table(IN_SOIL, columns=['sgg_code', 'date', 'soil_ph'])
assert soil.num_rows == N, f'토양표 행 수 {soil.num_rows:,} ≠ 기상표 {N:,}'
assert soil.select(['sgg_code', 'date']).equals(wx.select(['sgg_code', 'date'])), '토양표 키·순서가 기상표와 다름'
nn = n_nan_not_null(soil['soil_ph'])
log(f'시군구_일별_토양 {soil.num_rows:,}행 (키·순서 기상표와 같음) · soil_ph null {soil["soil_ph"].null_count:,} '
    f'({100 * soil["soil_ph"].null_count / N:.3f}%) · null 아닌 NaN {nn}')
assert nn == 0
sph = to_np(soil['soil_ph'])
SOIL_COLS = {
    'rg_soilph_opt': in_range(sph, *SPH_OPT),
    'rg_soilph_lim': in_range(sph, *SPH_LIM),
    'rg_soilph_opt_dist': dist_range(sph, *SPH_OPT),
}
for c in SOIL_COLS:
    assert np.array_equal(np.isnan(SOIL_COLS[c]), np.isnan(sph))
out_soil = pa.table({'sgg_code': soil['sgg_code'], 'date': soil['date'], **{k: to_pa32(v) for k, v in SOIL_COLS.items()}})


# ---------------------------------------------------------------- 3) 품목 3단계 식품 범위 표
food = pq.read_table(IN_FOOD)
NF = food.num_rows
codes = food['train_prdlst_code_2'].to_pylist()
assert None not in codes and len(set(codes)) == NF, '품목3단계 코드가 비었거나 겹침'
for c in ['aw', 'ph']:
    assert n_nan_not_null(food[c]) == 0
aw, ph = to_np(food['aw']), to_np(food['ph'])
FOOD_COLS = {
    'rg_aw_lim': in_range(aw, *AW_LIM),
    'rg_aw_opt_dist': np.abs(aw - AW_OPT),           # NaN 은 NaN 그대로
    'rg_ph_lim': in_range(ph, *PH_LIM),
    'rg_ph_opt_dist': np.abs(ph - PH_OPT),
}
assert np.array_equal(np.isnan(FOOD_COLS['rg_aw_lim']), np.isnan(aw))
assert np.array_equal(np.isnan(FOOD_COLS['rg_ph_lim']), np.isnan(ph))
out_food = pa.table({'train_prdlst_code_2': food['train_prdlst_code_2'], **{k: to_pa32(v) for k, v in FOOD_COLS.items()}})
aw_lab, ph_lab = food['aw_label'].to_pylist(), food['ph_label'].to_pylist()
log(f'품목3단계_식품속성 {NF}행 · aw 결측 {int(np.isnan(aw).sum())} · ph 결측 {int(np.isnan(ph).sum())} · '
    f'aw 표기 {dict(sorted({l: aw_lab.count(l) for l in set(aw_lab) if l}.items()))} · '
    f'ph 표기 {dict(sorted({l: ph_lab.count(l) for l in set(ph_lab) if l}.items()))}')


# ---------------------------------------------------------------- 정의 (정의 csv·parquet 메타데이터 공통)
def fig_cell(factor, kind, txt):
    return f'{FIG} — {FIG_TABLE} 의 {factor} {kind} 칸 "{txt}"'


MISS = '입력 칼럼이 결측(null)이면 결과도 결측(null). 0 으로 채우지 않음'
F_WX = '생장범위_일별_기상.parquet'
F_SOIL = '생장범위_일별_토양pH.parquet'
F_FOOD = '생장범위_품목3단계.parquet'
I_WX = '식정원_db_기상기후/시군구_일별_기상.parquet'
I_SOIL = '식정원_db_토양/시군구_일별_토양.parquet'
I_FOOD = '식정원_db_식품/품목3단계_식품속성.parquet'
DEFS = [
    dict(칸='sgg_code', 파일=f'{F_WX}, {F_SOIL}', 뜻='시군구 코드(키). 입력 표 값 그대로', 입력파일=f'{I_WX} / {I_SOIL}', 입력칼럼='sgg_code',
         범위값='', 계산='', 근거='', 원래출처='', 결측규칙='빈 값 없음', 선택사항='', 자료형='string'),
    dict(칸='date', 파일=f'{F_WX}, {F_SOIL}', 뜻='날짜(키). 입력 표 값 그대로. 그날 하루의 값', 입력파일=f'{I_WX} / {I_SOIL}', 입력칼럼='date',
         범위값='', 계산='', 근거='', 원래출처='', 결측규칙='빈 값 없음', 선택사항='', 자료형='date32'),
    dict(칸='rg_tmax_opt', 파일=F_WX, 뜻='그날 최고기온이 생장 최적 온도 범위 안에 들면 1, 아니면 0', 입력파일=I_WX, 입력칼럼='max_temp(최고기온, °C)',
         범위값='30 ≤ max_temp ≤ 37', 계산='30 ≤ max_temp ≤ 37 이면 1, 아니면 0 (경계 포함)', 근거=fig_cell('온도(기온)', '최적', '30~37°C'),
         원래출처=ORIG_SRC, 결측규칙=MISS, 선택사항='그림의 "온도(기온)"를 기상 기온으로 봄. 최적 판정은 최고기온으로 함(지시)', 자료형='float32 (0/1, 결측 가능)'),
    dict(칸='rg_tavg_opt_dist', 파일=F_WX, 뜻='그날 평균기온과 생장 최적 온도 범위(30~37°C) 사이의 거리(°C). 범위 안이면 0', 입력파일=I_WX, 입력칼럼='avg_temp(평균기온, °C)',
         범위값='30~37', 계산='avg_temp < 30 이면 30 − avg_temp, avg_temp > 37 이면 avg_temp − 37, 그 밖은 0', 근거=fig_cell('온도(기온)', '최적', '30~37°C'),
         원래출처=ORIG_SRC, 결측규칙=MISS, 선택사항='거리는 평균기온으로 잼(지시)', 자료형='float32'),
    dict(칸='rg_t_out_lim', 파일=F_WX, 뜻='그날 기온이 생장 한계 범위(−2~45°C) 밖으로 나갔으면 1(최저기온 < −2 또는 최고기온 > 45), 아니면 0', 입력파일=I_WX,
         입력칼럼='min_temp(최저기온, °C), max_temp(최고기온, °C)', 범위값='−2 ≤ 기온 ≤ 45 (밖으로 나간 날 = min_temp < −2 또는 max_temp > 45)',
         계산='min_temp < −2 또는 max_temp > 45 이면 1, 아니면 0 (경계값 −2·45 는 한계 안으로 봄)', 근거=fig_cell('온도(기온)', '한계', '-2~45°C'),
         원래출처=ORIG_SRC, 결측규칙='min_temp·max_temp 중 하나라도 결측이면 결측(남은 한 칸만으로 1 이 될 수 있어도 결측으로 둠)',
         선택사항='다른 칸과 달리 1 = 한계 밖(나쁜 쪽)', 자료형='float32 (0/1, 결측 가능)'),
    dict(칸='rg_soilt_opt', 파일=F_WX, 뜻='그날 5cm 지중온도가 토양 온도 최적 범위 안에 들면 1, 아니면 0', 입력파일=I_WX, 입력칼럼='soil_t5(지중온도 5cm, °C)',
         범위값='8 ≤ soil_t5 ≤ 15', 계산='8 ≤ soil_t5 ≤ 15 이면 1, 아니면 0 (경계 포함)', 근거=fig_cell('토양 온도', '최적', '8~15°C'),
         원래출처=ORIG_SRC, 결측규칙=MISS,
         선택사항='토양 온도는 5cm 깊이(soil_t5) 하나만 씀 — 그림에 깊이가 없어서 표면에 가장 가까운 깊이를 고름. soil_t10·soil_t20·soil_t30·soil_t_nas(깊이 설명 없음)는 쓰지 않음',
         자료형='float32 (0/1, 결측 가능)'),
    dict(칸='rg_soilt_lim', 파일=F_WX, 뜻='그날 5cm 지중온도가 토양 온도 한계 범위 안에 들면 1, 아니면 0', 입력파일=I_WX, 입력칼럼='soil_t5(지중온도 5cm, °C)',
         범위값='8 ≤ soil_t5 ≤ 30', 계산='8 ≤ soil_t5 ≤ 30 이면 1, 아니면 0 (경계 포함)', 근거=fig_cell('토양 온도', '한계', '8~30°C'),
         원래출처=ORIG_SRC, 결측규칙=MISS, 선택사항='5cm 깊이(soil_t5) — rg_soilt_opt 와 같은 이유. 그림의 한계 하한(8°C)이 최적 하한과 같음(그림 그대로 씀)',
         자료형='float32 (0/1, 결측 가능)'),
    dict(칸='rg_soilt_opt_dist', 파일=F_WX, 뜻='그날 5cm 지중온도와 토양 온도 최적 범위(8~15°C) 사이의 거리(°C). 범위 안이면 0', 입력파일=I_WX,
         입력칼럼='soil_t5(지중온도 5cm, °C)', 범위값='8~15', 계산='soil_t5 < 8 이면 8 − soil_t5, soil_t5 > 15 이면 soil_t5 − 15, 그 밖은 0',
         근거=fig_cell('토양 온도', '최적', '8~15°C'), 원래출처=ORIG_SRC, 결측규칙=MISS, 선택사항='5cm 깊이(soil_t5) — rg_soilt_opt 와 같은 이유', 자료형='float32'),
    dict(칸='rg_soilph_opt', 파일=F_SOIL, 뜻='그 시군구의 토양 산도(soil_ph)가 토양 pH 최적 범위 안이면 1, 아니면 0', 입력파일=I_SOIL,
         입력칼럼='soil_ph(그날 이전 3년 [t−1095일, t−1일] 토양검정 기록의 산도 중앙값 — 04_시군구_토양.py)', 범위값='6.5 ≤ soil_ph ≤ 7.0',
         계산='6.5 ≤ soil_ph ≤ 7.0 이면 1, 아니면 0 (경계 포함)', 근거=fig_cell('토양 pH', '최적', 'pH 6.5~7.0'), 원래출처=ORIG_SRC, 결측규칙=MISS,
         선택사항='soil_ph 를 그날 행 값 그대로 씀. 검정 건수 칸(soil_ph_n)은 쓰지 않음', 자료형='float32 (0/1, 결측 가능)'),
    dict(칸='rg_soilph_lim', 파일=F_SOIL, 뜻='그 시군구의 토양 산도(soil_ph)가 토양 pH 한계 범위 안이면 1, 아니면 0', 입력파일=I_SOIL, 입력칼럼='soil_ph',
         범위값='5.0 ≤ soil_ph ≤ 8.0', 계산='5.0 ≤ soil_ph ≤ 8.0 이면 1, 아니면 0 (경계 포함)', 근거=fig_cell('토양 pH', '한계', 'pH 5.0~8.0'), 원래출처=ORIG_SRC,
         결측규칙=MISS, 선택사항='rg_soilph_opt 와 같음', 자료형='float32 (0/1, 결측 가능)'),
    dict(칸='rg_soilph_opt_dist', 파일=F_SOIL, 뜻='토양 산도(soil_ph)와 토양 pH 최적 범위(6.5~7.0) 사이의 거리(pH). 범위 안이면 0', 입력파일=I_SOIL, 입력칼럼='soil_ph',
         범위값='6.5~7.0', 계산='soil_ph < 6.5 이면 6.5 − soil_ph, soil_ph > 7.0 이면 soil_ph − 7.0, 그 밖은 0', 근거=fig_cell('토양 pH', '최적', 'pH 6.5~7.0'),
         원래출처=ORIG_SRC, 결측규칙=MISS, 선택사항='rg_soilph_opt 와 같음', 자료형='float32'),
    dict(칸='train_prdlst_code_2', 파일=F_FOOD, 뜻='품목군 3단계 코드(키). 입력 표 값 그대로', 입력파일=I_FOOD, 입력칼럼='train_prdlst_code_2',
         범위값='', 계산='', 근거='', 원래출처='', 결측규칙='빈 값 없음', 선택사항='', 자료형='string'),
    dict(칸='rg_aw_lim', 파일=F_FOOD, 뜻='품목 3단계 대표 수분활성도(aw)가 생장 한계 범위 안이면 1, 아니면 0', 입력파일=I_FOOD, 입력칼럼='aw(수분활성도 대표값)',
         범위값='0.90 ≤ aw ≤ 0.997', 계산='0.90 ≤ aw ≤ 0.997 이면 1, 아니면 0 (경계 포함)', 근거=fig_cell('수분활성도', '한계', '0.90~0.997'), 원래출처=ORIG_SRC,
         결측규칙=MISS, 선택사항='aw_label 이 중앙·초과·이상·미만·이하여도 aw 숫자 그대로 씀. 부등호 표기 품목은 생장범위_부등호표기.csv',
         자료형='float32 (0/1, 결측 가능)'),
    dict(칸='rg_aw_opt_dist', 파일=F_FOOD, 뜻='품목 3단계 대표 수분활성도(aw)와 최적값 0.97 의 차이(절댓값)', 입력파일=I_FOOD, 입력칼럼='aw(수분활성도 대표값)',
         범위값='0.97 (한 점)', 계산='|aw − 0.97|', 근거=fig_cell('수분활성도', '최적', '0.97'), 원래출처=ORIG_SRC, 결측규칙=MISS,
         선택사항='그림의 최적이 한 점이라 최적 0/1 칸은 만들지 않고 거리만 만듦. aw 숫자 그대로(부등호 표기 포함)', 자료형='float32'),
    dict(칸='rg_ph_lim', 파일=F_FOOD, 뜻='품목 3단계 대표 식품 pH(ph)가 생장 한계 범위 안이면 1, 아니면 0', 입력파일=I_FOOD, 입력칼럼='ph(식품pH 대표값)',
         범위값='4.0 ≤ ph ≤ 9.6', 계산='4.0 ≤ ph ≤ 9.6 이면 1, 아니면 0 (경계 포함)', 근거=fig_cell('pH', '한계', 'pH 4.0~9.6'), 원래출처=ORIG_SRC,
         결측규칙=MISS, 선택사항='그림의 "pH"를 식품 pH 로 봄(토양 pH 는 따로 있음). ph_label 이 중앙이어도 숫자 그대로', 자료형='float32 (0/1, 결측 가능)'),
    dict(칸='rg_ph_opt_dist', 파일=F_FOOD, 뜻='품목 3단계 대표 식품 pH(ph)와 최적값 7.0 의 차이(절댓값)', 입력파일=I_FOOD, 입력칼럼='ph(식품pH 대표값)',
         범위값='7.0 (한 점)', 계산='|ph − 7.0|', 근거=fig_cell('pH', '최적', 'pH 7.0'), 원래출처=ORIG_SRC, 결측규칙=MISS,
         선택사항='그림의 최적이 한 점이라 최적 0/1 칸은 만들지 않고 거리만 만듦', 자료형='float32'),
    dict(칸='rg_na_lim (만들지 않음)', 파일='(없음)', 뜻='염도가 한계(≤16%) 안이면 1 — 만들지 않음', 입력파일=I_FOOD, 입력칼럼='na(염도 대표값)',
         범위값='na ≤ 16 (%)', 계산='', 근거=fig_cell('염도', '한계', '≤16%') + ' (최적 칸은 "-")', 원래출처=ORIG_SRC, 결측규칙='',
         선택사항=('na 칼럼의 단위가 %인지 문서로 확인하지 못해 만들지 않음. 확인한 문서: ba_data.t_aw_cl 칼럼 na 의 DB 코멘트 "염도", '
               '테이블정의서(문서/테이블명세서(모델팀)/테이블정의서.xlsx) na 설명 "염도", 리서치팀 엑셀(통합_수분활성도_수분함량_식품pH_염도_최종.xlsx) '
               "'값'·'출처' 시트 머리글 \"염도\" — 셋 다 단위 없음. 단위가 확인되면 추가"),
         자료형=''),
]
DEF_BY = {d['칸']: d for d in DEFS}


def meta_for(cols, desc, inputs):
    return {
        'description': desc,
        'basis': f'{FIG} ({FIG_FILE}) {FIG_TABLE}; 원래 논문 출처: {ORIG_SRC}',
        'inputs': json.dumps(inputs, ensure_ascii=False),
        'script': SCRIPT,
        'rule_common': '결측 입력 → null(0 채움 없음), 경계 포함, 거리 = 범위 안 0·밖이면 가까운 경계까지 절댓값 차',
        'columns': json.dumps({c: {k: DEF_BY[c][k] for k in ('뜻', '입력칼럼', '범위값', '계산', '결측규칙', '선택사항')} for c in cols}, ensure_ascii=False),
    }


# ---------------------------------------------------------------- 저장
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_wx = out_wx.replace_schema_metadata(meta_for(list(WX_COLS), '시군구 x 날짜 리스테리아 생장 범위 변수(기온·5cm 지중온도). 행 순서 = 시군구_일별_기상.parquet',
                                                 [I_WX]))
out_soil = out_soil.replace_schema_metadata(meta_for(list(SOIL_COLS), '시군구 x 날짜 리스테리아 생장 범위 변수(토양 산도). 행 순서 = 시군구_일별_기상.parquet',
                                                     [I_SOIL]))
out_food = out_food.replace_schema_metadata(meta_for(list(FOOD_COLS), '품목군 3단계 리스테리아 생장 범위 변수(수분활성도·식품 pH). 염도는 단위 미확인으로 없음',
                                                     [I_FOOD]))
pq.write_table(out_wx, OUT_WX)
pq.write_table(out_soil, OUT_SOIL)
pq.write_table(out_food, OUT_FOOD)
log(f'저장: {OUT_WX.name} {out_wx.num_rows:,}행 × {out_wx.num_columns}칸 · {OUT_SOIL.name} {out_soil.num_rows:,}행 × {out_soil.num_columns}칸 · '
    f'{OUT_FOOD.name} {out_food.num_rows}행 × {out_food.num_columns}칸')

DEF_HEAD = {'칸': '칸 이름', '파일': '파일', '뜻': '뜻', '입력파일': '입력 파일(data/데이터셋/ 아래)', '입력칼럼': '입력 칼럼',
            '범위값': '범위 값', '계산': '계산', '근거': '근거(그림)', '원래출처': '값의 원래 출처', '결측규칙': '결측 규칙',
            '선택사항': '선택 사항·참고', '자료형': '자료형'}
with open(OUT_DEF, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(list(DEF_HEAD.values()))
    for d in DEFS:
        w.writerow([d[k] for k in DEF_HEAD])
    w.writerow([])
    w.writerow(['(참고) 부등호 표기 품목', '생장범위_부등호표기.csv', '품목 3단계 대표 aw 의 표기가 초과·이상·미만·이하인 품목 목록(숫자는 그대로 씀)',
                I_FOOD, 'aw, aw_label', '', '', '', '', '', '', ''])
log(f'저장: {OUT_DEF.name} ({len(DEFS)}줄 + 참고 1줄)')

# 부등호 표기 품목 (aw·ph)
sign_rows = []
nm2, code2 = food['train_prdlst_nm_2'].to_pylist(), food['train_prdlst_code'].to_pylist()
for item, vals, labs, lim, dist in (('aw', aw, aw_lab, FOOD_COLS['rg_aw_lim'], FOOD_COLS['rg_aw_opt_dist']),
                                    ('ph', ph, ph_lab, FOOD_COLS['rg_ph_lim'], FOOD_COLS['rg_ph_opt_dist'])):
    for i in range(NF):
        if labs[i] in SIGN_LABELS:
            sign_rows.append({'train_prdlst_code_2': codes[i], 'train_prdlst_nm_2': nm2[i], 'train_prdlst_code': code2[i], '항목': item,
                              '값': f'{vals[i]:.10g}', '표기': labs[i],
                              f'rg_{item}_lim': int(lim[i]), f'rg_{item}_opt_dist': f'{dist[i]:.6g}'})
SIGN_HEAD = ['train_prdlst_code_2', 'train_prdlst_nm_2', 'train_prdlst_code', '항목', '값', '표기', 'rg_aw_lim', 'rg_aw_opt_dist', 'rg_ph_lim', 'rg_ph_opt_dist']
with open(OUT_SIGN, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=SIGN_HEAD, restval='')
    w.writeheader()
    w.writerows(sign_rows)
log(f'저장: {OUT_SIGN.name} ({len(sign_rows)}줄 — 항목별 {dict((it, sum(r["항목"] == it for r in sign_rows)) for it in ("aw", "ph"))})')
log('끝')
