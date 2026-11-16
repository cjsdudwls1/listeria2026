# -*- coding: utf-8 -*-
"""
check_ranges.py — make_ranges.py 결과 점검 (읽기만 함, 파일을 쓰지 않음 — 결과는 화면 출력)

  1) 행 수·키: 일별 두 표의 행 수 = 시군구_일별_기상.parquet 행 수, (sgg_code, date) 중복 0, 키·순서 같음
  2) 칸마다 결측 비율(전체·연도별), 0/1 칸의 1 비율(전체·월별), 거리 칸의 평균(전체·월별)
  3) 입력과 다시 대조: 무작위가 아닌 고정 간격 표본 행에서 규칙을 다시 계산해 같은지
  4) (참고) 학습 기간 관계: 시료_시군구.parquet 중 mnfctur_de 의 날짜 ≤ 2024-06-30 인 행만.
     일별 칸은 창 T−50..T−1(기준일 = mnfctur_de 의 날짜) 평균을 검출/불검출로 나눠 비교, 식품 칸은 값별 시료·검출 수.
     2024-06-30 뒤 행의 judge_result 는 읽은 뒤 바로 버리고 어디에도 쓰지 않는다(2025-07-01 이후 포함).

실행: "리스테리아ML\\.venv-exp2\\Scripts\\python.exe" "리스테리아ML2\\data\\데이터셋\\생장범위\\scripts\\check_ranges.py"
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DS = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋')
IN_WX = DS / '식정원_db_기상기후' / '시군구_일별_기상.parquet'
IN_SOIL = DS / '식정원_db_토양' / '시군구_일별_토양.parquet'
IN_FOOD = DS / '식정원_db_식품' / '품목3단계_식품속성.parquet'
SAMPLE = DS / '시료_시군구.parquet'
OUT = DS / '생장범위'
TRAIN_END = dt.date(2024, 6, 30)
WIN = (50, 1)          # T−50 .. T−1

FLAG_COLS = ['rg_tmax_opt', 'rg_t_out_lim', 'rg_soilt_opt', 'rg_soilt_lim', 'rg_soilph_opt', 'rg_soilph_lim']
DIST_COLS = ['rg_tavg_opt_dist', 'rg_soilt_opt_dist', 'rg_soilph_opt_dist']


def f64(col):
    return pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False)


wx_in = pq.read_table(IN_WX, columns=['sgg_code', 'date', 'avg_temp', 'max_temp', 'min_temp', 'soil_t5'])
soil_in = pq.read_table(IN_SOIL, columns=['sgg_code', 'date', 'soil_ph'])
rw = pq.read_table(OUT / '생장범위_일별_기상.parquet')
rs = pq.read_table(OUT / '생장범위_일별_토양pH.parquet')
rf = pq.read_table(OUT / '생장범위_품목3단계.parquet')
N = wx_in.num_rows

print('== 1) 행 수·키')
for name, t in (('생장범위_일별_기상', rw), ('생장범위_일별_토양pH', rs)):
    dup = t.num_rows - t.group_by(['sgg_code', 'date']).aggregate([]).num_rows
    same = t.select(['sgg_code', 'date']).equals(wx_in.select(['sgg_code', 'date']))
    print(f'  {name}: {t.num_rows:,}행 (기상 입력 {N:,}행, 같음={t.num_rows == N}) · 키 중복 {dup} · 키·순서 기상 입력과 같음={same}')
    print(f'    칸·자료형: {[(f.name, str(f.type)) for f in t.schema]}')
print(f'  생장범위_품목3단계: {rf.num_rows}행 · 코드 중복 {rf.num_rows - pc.count_distinct(rf["train_prdlst_code_2"]).as_py()} · '
      f'칸: {[(f.name, str(f.type)) for f in rf.schema]}')

R = {c: f64(rw[c]) for c in rw.column_names if c.startswith('rg_')}
R.update({c: f64(rs[c]) for c in rs.column_names if c.startswith('rg_')})
dates = rw['date'].to_numpy(zero_copy_only=False).astype('datetime64[D]')
years = dates.astype('datetime64[Y]').astype(int) + 1970
months = dates.astype('datetime64[M]').astype(int) % 12 + 1

print('\n== 2) 결측·1 비율·거리 평균')
print('  칸 | 결측 비율(%) | 1 비율 또는 평균(결측 뺀 값)')
for c in FLAG_COLS + DIST_COLS:
    v = R[c]
    ok = ~np.isnan(v)
    kind = '1 비율(%)' if c in FLAG_COLS else '평균'
    val = 100 * v[ok].mean() if c in FLAG_COLS else v[ok].mean()
    extra = '' if c in FLAG_COLS else f' · 0(범위 안) 비율 {100 * (v[ok] == 0).mean():.2f}% · 최대 {v[ok].max():.2f}'
    print(f'  {c} | {100 * (~ok).mean():.3f} | {kind} {val:.3f}{extra}')

print('\n  월별 (결측 뺀 값의 1 비율 % / 거리 평균)')
print('  칸 \\ 월 | ' + ' | '.join(f'{m:>5d}' for m in range(1, 13)))
for c in FLAG_COLS + DIST_COLS:
    v = R[c]
    row = []
    for m in range(1, 13):
        sel = (months == m) & ~np.isnan(v)
        x = v[sel].mean() if sel.any() else np.nan
        row.append(f'{100 * x:5.1f}' if c in FLAG_COLS else f'{x:5.2f}')
    print(f'  {c} | ' + ' | '.join(row))

print('\n  연도별 결측 비율(%)')
yrs = sorted(set(years.tolist()))
print('  칸 \\ 연도 | ' + ' | '.join(str(y) for y in yrs))
for c in ['rg_tmax_opt', 'rg_soilt_opt', 'rg_soilph_opt']:
    v = R[c]
    print(f'  {c} | ' + ' | '.join(f'{100 * np.isnan(v[years == y]).mean():.1f}' for y in yrs))

# 결측 날 모양: 기상 결측이 날짜 전체(256 시군구 모두)인지
S = 256
D = N // S
assert S * D == N
g = np.isnan(R['rg_tmax_opt']).reshape(S, D)
day_all = g.all(axis=0)
day_any = g.any(axis=0)
d0 = dates.reshape(S, D)[0]
print(f'\n  기상 결측: 256곳 모두 결측인 날 {int(day_all.sum())}일 · 일부만 결측인 날 {int((day_any & ~day_all).sum())}일 · '
      f'모두 결측인 날의 행 {int(day_all.sum()) * S:,} (전체 결측 행 {int(g.sum()):,})')
if day_all.any():
    idx = np.flatnonzero(day_all)
    runs, start = [], idx[0]
    for a, b in zip(idx, np.append(idx[1:], -1)):
        if b != a + 1:
            runs.append((str(d0[start]), str(d0[a]), int(a - start + 1)))
            start = b
    print(f'    모두 결측인 날 구간(앞 15개): {runs[:15]}{" …" if len(runs) > 15 else ""} (구간 {len(runs)}개)')
gp = np.isnan(R['rg_soilph_opt']).reshape(S, D)
print(f'  토양 pH 결측: 시군구별 결측 비율 — 0% {int((gp.mean(axis=1) == 0).sum())}곳 · 100% {int((gp.mean(axis=1) == 1).sum())}곳 · '
      f'그 밖 {int(((gp.mean(axis=1) > 0) & (gp.mean(axis=1) < 1)).sum())}곳')
col_ok = (~gp).any(axis=0)
print(f'    한 곳이라도 토양 pH 가 있는 첫날 {d0[np.flatnonzero(col_ok)[0]]} · 그날 이후 256곳 모두 결측인 날 {int((~col_ok[np.flatnonzero(col_ok)[0]:]).sum())}일')

print('\n== 3) 입력과 다시 대조 (7행마다 1행 표본)')
idx = np.arange(0, N, 7)
tavg, tmax, tmin, st5 = (f64(wx_in[c])[idx] for c in ['avg_temp', 'max_temp', 'min_temp', 'soil_t5'])
sph = f64(soil_in['soil_ph'])[idx]


def chk(name, got, want):
    got = got[idx] if got.shape[0] == N else got
    both_nan = np.isnan(got) & np.isnan(want)
    eq = both_nan | np.isclose(got, want, rtol=0, atol=1e-5)
    print(f'  {name}: 표본 {len(idx):,}행 중 다른 행 {int((~eq).sum())}')


def rin(x, lo, hi):
    return np.where(np.isnan(x), np.nan, ((x >= lo) & (x <= hi)).astype(float))


def rdist(x, lo, hi):
    return np.where(np.isnan(x), np.nan, np.maximum(np.maximum(lo - x, x - hi), 0.0))


chk('rg_tmax_opt', R['rg_tmax_opt'], rin(tmax, 30, 37))
chk('rg_tavg_opt_dist', R['rg_tavg_opt_dist'], rdist(tavg, 30, 37))
chk('rg_t_out_lim', R['rg_t_out_lim'], np.where(np.isnan(tmin) | np.isnan(tmax), np.nan, ((tmin < -2) | (tmax > 45)).astype(float)))
chk('rg_soilt_opt', R['rg_soilt_opt'], rin(st5, 8, 15))
chk('rg_soilt_lim', R['rg_soilt_lim'], rin(st5, 8, 30))
chk('rg_soilt_opt_dist', R['rg_soilt_opt_dist'], rdist(st5, 8, 15))
chk('rg_soilph_opt', R['rg_soilph_opt'], rin(sph, 6.5, 7.0))
chk('rg_soilph_lim', R['rg_soilph_lim'], rin(sph, 5.0, 8.0))
chk('rg_soilph_opt_dist', R['rg_soilph_opt_dist'], rdist(sph, 6.5, 7.0))
fd = pq.read_table(IN_FOOD)
assert fd['train_prdlst_code_2'].equals(rf['train_prdlst_code_2'])
aw, ph = f64(fd['aw']), f64(fd['ph'])
idx = np.arange(rf.num_rows)
chk('rg_aw_lim', f64(rf['rg_aw_lim']), rin(aw, 0.90, 0.997))
chk('rg_aw_opt_dist', f64(rf['rg_aw_opt_dist']), np.abs(aw - 0.97))
chk('rg_ph_lim', f64(rf['rg_ph_lim']), rin(ph, 4.0, 9.6))
chk('rg_ph_opt_dist', f64(rf['rg_ph_opt_dist']), np.abs(ph - 7.0))
# 경계값 그대로 포함되는지: 입력이 정확히 경계값인 행
for name, x, lo, hi, col in (('max_temp', f64(wx_in['max_temp']), 30, 37, 'rg_tmax_opt'), ('soil_t5', f64(wx_in['soil_t5']), 8, 15, 'rg_soilt_opt'),
                             ('soil_ph', f64(soil_in['soil_ph']), 6.5, 7.0, 'rg_soilph_opt')):
    on = (x == lo) | (x == hi)
    print(f'  경계값 행 {name} ∈ {{{lo}, {hi}}}: {int(on.sum()):,}행 · 그 행의 {col} 값 {sorted(set(R[col][on].tolist()))}')
for c in ['rg_aw_lim', 'rg_aw_opt_dist', 'rg_ph_lim', 'rg_ph_opt_dist']:
    v = f64(rf[c])
    ok = ~np.isnan(v)
    s = f'1 비율 {100 * v[ok].mean():.1f}% ({int(v[ok].sum())}/{int(ok.sum())})' if c.endswith('_lim') else \
        f'평균 {v[ok].mean():.3f} · 최소 {v[ok].min():.3f} · 최대 {v[ok].max():.3f}'
    print(f'  {c}: 결측 {int((~ok).sum())}/{rf.num_rows} · {s}')

print('\n== 4) (참고) 학습 기간(mnfctur_de 의 날짜 ≤ 2024-06-30) 관계')
smp = pq.read_table(SAMPLE, columns=['sgg_code', 'mnfctur_de', 'train_prdlst_code_2', 'judge_result'])
day = pc.cast(pc.floor_temporal(smp['mnfctur_de'], unit='day'), pa.date32())
keep = pc.and_(pc.less_equal(day, pa.scalar(TRAIN_END, pa.date32())), pc.is_valid(smp['sgg_code']))
smp = smp.filter(keep)                        # 2024-06-30 뒤 행은 여기서 버림 — 그 뒤 judge_result 는 쓰지 않음
day = pc.cast(pc.floor_temporal(smp['mnfctur_de'], unit='day'), pa.date32())
assert pc.max(day).as_py() <= TRAIN_END
det = np.array([j == '검출' for j in pc.cast(smp['judge_result'], pa.string()).to_pylist()])
print(f'  학습 기간 시료 {smp.num_rows:,}행 (sgg_code 있는 행) · 검출 {int(det.sum()):,}')

sgg_list = rw['sgg_code'].to_pylist()[::D]
s_idx = {s: i for i, s in enumerate(sgg_list)}
si = np.array([s_idx.get(s, -1) for s in smp['sgg_code'].to_pylist()])
d_first = d0[0]
di = (day.to_numpy(zero_copy_only=False).astype('datetime64[D]') - d_first).astype(int)
ok_row = (si >= 0) & (di - WIN[0] >= 0) & (di < D)
print(f'  창 T−{WIN[0]}..T−{WIN[1]} 이 기상 기간 안에 드는 시료 {int(ok_row.sum()):,}행 (검출 {int((ok_row & det).sum()):,})')
print('  칸 | 창 평균의 평균: 검출 / 불검출 | 창 평균 결측 행 비율 검출 / 불검출')
for c in FLAG_COLS + DIST_COLS:
    v = R[c].reshape(S, D)
    val = np.where(np.isnan(v), 0.0, v)
    cnt = (~np.isnan(v)).astype(np.int32)
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(val, axis=1)], axis=1)
    cc = np.concatenate([np.zeros((S, 1), np.int32), np.cumsum(cnt, axis=1)], axis=1)
    m = np.full(len(si), np.nan)
    r = np.flatnonzero(ok_row)
    a, b = di[r] - WIN[0], di[r] - WIN[1] + 1           # 날짜 a .. b−1 = T−50 .. T−1
    n = cc[si[r], b] - cc[si[r], a]
    with np.errstate(invalid='ignore', divide='ignore'):
        m[r] = np.where(n > 0, (cs[si[r], b] - cs[si[r], a]) / n, np.nan)
    md, mn_ = m[ok_row & det], m[ok_row & ~det]
    fmt = (lambda x: f'{100 * x:.2f}%') if c in FLAG_COLS else (lambda x: f'{x:.3f}')
    print(f'  {c} | {fmt(np.nanmean(md))} / {fmt(np.nanmean(mn_))} | {100 * np.isnan(md).mean():.2f}% / {100 * np.isnan(mn_).mean():.2f}%')

fl = {c: dict(zip(rf['train_prdlst_code_2'].to_pylist(), f64(rf[c]).tolist())) for c in ['rg_aw_lim', 'rg_ph_lim']}
k3 = pc.cast(smp['train_prdlst_code_2'], pa.string()).to_pylist()
tot_det = int(det.sum())
print('  식품 칸 값별: 시료 수 · 검출 수 · 검출 비중(전체 검출 중 %) · 그룹 내 검출률(%)')
for c in fl:
    v = np.array([fl[c].get(k, np.nan) if k is not None else np.nan for k in k3])
    for lab, sel in (('1', v == 1), ('0', v == 0), ('결측', np.isnan(v))):
        n, nd = int(sel.sum()), int((sel & det).sum())
        print(f'    {c}={lab}: {n:,} · {nd:,} · {100 * nd / tot_det:.1f}% · {100 * nd / n if n else float("nan"):.3f}%')
