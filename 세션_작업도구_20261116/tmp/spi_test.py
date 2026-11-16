"""SPI 직접 계산 시험: 받아 둔 ASOS 일강수로 SPI3·SPI6 를 계산해 기상청 값과 비교.
방법: McKee(1993)/WMO(2012) — n개월 누적 강수 → 달력일별 감마 적합(Thom MLE 근사, 0 강수는 확률 q 로 따로)
      → H = q + (1-q)G(x) → 표준정규 역함수. 기상청 정의: 2019-02 까지 30n일, 2019-03-01 부터 달력 n개월(그날 포함).
"""
import csv, datetime as dt, sys
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.special import gammainc, ndtri

ASOS = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\raw\raw_kma_asos_daily_info.parquet'
SPI = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\외부_데이터\CLM_SPI_DD_합본_20140101_20261006.csv'
OUT = sys.argv[1]
log = []
D0 = dt.date(2015, 1, 1)
D1 = dt.date(2026, 10, 6)
N = (D1 - D0).days + 1
dates = [D0 + dt.timedelta(i) for i in range(N)]

t = pq.read_table(ASOS, columns=['stn_id', 'meas_date', 'sum_rn'])
stn = t['stn_id'].to_numpy(zero_copy_only=False).astype(int)
md = t['meas_date']
md = pc.cast(md, 'date32') if str(md.type) != 'date32[day]' else md
md = md.to_numpy(zero_copy_only=False).astype('datetime64[D]')
rn = t['sum_rn'].to_numpy(zero_copy_only=False)
rn = np.where(np.isnan(rn.astype(float)), 0.0, rn.astype(float))  # 빈 값 = 0mm(결정 61)
idx = (md - np.datetime64(D0)).astype(int)

# 기상청 SPI
kma = {}
with open(SPI, encoding='cp949') as f:
    r = csv.reader(f)
    h = next(r)
    for row in r:
        try:
            s = int(row[0]); d = dt.date.fromisoformat(row[2][:10])
        except Exception:
            continue
        if d < D0 or d > D1:
            continue
        v3 = float(row[4]) if row[4].strip() not in ('', '-') else np.nan
        v6 = float(row[5]) if row[5].strip() not in ('', '-') else np.nan
        kma.setdefault(s, [np.full(N, np.nan), np.full(N, np.nan)])
        i = (d - D0).days
        kma[s][0][i] = v3; kma[s][1][i] = v6
stations = sorted(set(kma) & set(stn.tolist()))
log.append(f'기상청 SPI 지점 {len(kma)}, ASOS 원자료에 있는 지점 {len(stations)}')

# 날짜 보조값
doy = np.array([min(d.timetuple().tm_yday, 365) if not (d.month == 2 and d.day == 29) else 59 for d in dates])
doy = np.array([(d.timetuple().tm_yday - (1 if (d.month > 2 and (d.year % 4 == 0)) else 0)) if not (d.month == 2 and d.day == 29) else 59 for d in dates])
year = np.array([d.year for d in dates])
post = np.array([d >= dt.date(2019, 3, 1) for d in dates])


def month_start_idx(n):
    out = np.empty(N, dtype=int)
    for i, d in enumerate(dates):
        m = d.month - n; y = d.year
        while m < 1:
            m += 12; y -= 1
        last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(1)).day
        s = dt.date(y, m, min(d.day, last)) + dt.timedelta(1)
        out[i] = (s - D0).days
    return out


MS = {3: month_start_idx(3), 6: month_start_idx(6)}


def accum(P, n):
    C = np.concatenate([[0.0], np.cumsum(np.nan_to_num(P))])
    Cn = np.concatenate([[0], np.cumsum(np.isnan(P))])
    i = np.arange(N)
    a30 = np.full(N, np.nan); L = 30 * n
    ok = i - L + 1 >= 0
    j = i[ok] - L + 1
    a30[ok] = np.where(Cn[i[ok] + 1] - Cn[j] == 0, C[i[ok] + 1] - C[j], np.nan)
    am = np.full(N, np.nan); j = MS[n]
    ok = j >= 0
    am[ok] = np.where(Cn[i[ok] + 1] - Cn[j[ok]] == 0, C[i[ok] + 1] - C[j[ok]], np.nan)
    return np.where(post, am, a30)  # 기상청 정의대로 기간별로 고름


def spi(X, fit_mask, half=15):
    out = np.full(N, np.nan)
    for c in range(1, 366):
        dd = np.minimum(np.abs(doy - c), 365 - np.abs(doy - c))
        pool = (dd <= half) & fit_mask & ~np.isnan(X)
        x = X[pool]
        if len(x) < 30:
            continue
        q = np.mean(x <= 0)
        nz = x[x > 0]
        A = np.log(nz.mean()) - np.mean(np.log(nz))
        a = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A); b = nz.mean() / a
        tgt = (doy == c) & ~np.isnan(X)
        Hh = q + (1 - q) * gammainc(a, np.maximum(X[tgt], 0) / b)
        out[tgt] = ndtri(np.clip(Hh, 1e-10, 1 - 1e-10))
    return out


def stats(a, b, m):
    k = m & ~np.isnan(a) & ~np.isnan(b)
    if k.sum() < 10:
        return 'n<10'
    return f'r {np.corrcoef(a[k], b[k])[0, 1]:.3f}·MAE {np.mean(np.abs(a[k] - b[k])):.3f}·n {k.sum()}'


res = {}
for n, col in ((3, 0), (6, 1)):
    A_all, B_all, A_oos, B_oos, PST = [], [], [], [], []
    for s in stations:
        P = np.full(N, np.nan)
        k = (stn == s) & (idx >= 0) & (idx < N)
        P[idx[k]] = rn[k]
        X = accum(P, n)
        ours_in = spi(X, year <= 2025)                     # 2015~2025 전부로 적합(표본 안)
        ours_oos = spi(X, year <= 2021)                    # 2015~2021 로만 적합 → 2022~ 비교(표본 밖)
        A_all.append(ours_in); B_all.append(kma[s][col]); A_oos.append(ours_oos); PST.append(post)
    A = np.concatenate(A_all); B = np.concatenate(B_all); O = np.concatenate(A_oos); PS = np.concatenate(PST)
    yr = np.tile(year, len(stations))
    log.append(f'SPI{n} 표본 안(2015~2025 적합): 전체 {stats(A, B, np.ones_like(PS))} | 2019-02 까지(30n일) {stats(A, B, ~PS)} | 2019-03 부터(달력 n개월) {stats(A, B, PS)}')
    log.append(f'SPI{n} 표본 밖(2015~2021 적합, 2022~ 비교): {stats(O, B, yr >= 2022)}')
    res[n] = (A, B)

# 예보 11일 붙이기 예: 서울(108) 2025-08-01 을 T 로, T~T+10 강수를 (가) 관측 그대로 (나) 0mm 로 넣어 SPI3 계산
s = 108 if 108 in stations else stations[0]
P = np.full(N, np.nan)
k = (stn == s) & (idx >= 0) & (idx < N)
P[idx[k]] = rn[k]
T = (dt.date(2025, 8, 1) - D0).days
X = accum(P, 3)
base = spi(X, year <= 2024)
P2 = P.copy(); P2[T:T + 11] = 0.0
X2 = accum(P2, 3)
dry = spi(X2, year <= 2024)
log.append(f'예보 붙이기 예(지점 {s}, T = 2025-08-01, SPI3, 2015~2024 적합): T−1 {base[T-1]:.2f}')
log.append('  T~T+10 강수 관측 그대로: ' + ' '.join(f'{v:.2f}' for v in base[T:T + 11]))
log.append('  T~T+10 강수 0mm 로 바꿈:   ' + ' '.join(f'{v:.2f}' for v in dry[T:T + 11]))
log.append(f'  같은 기간 기상청 SPI3: ' + ' '.join(f'{v:.2f}' for v in kma[s][0][T:T + 11]))
log.append(f'  T~T+10 관측 강수 합 {np.nansum(P[T:T+11]):.1f}mm')
open(OUT, 'w', encoding='utf-8').write('\n'.join(log))
