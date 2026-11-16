"""국내 SPI: 과거 = 기상청 값, 예보 11일 = 우리 식으로 이을 때 이음매 크기 시험.
학습 땐 예보 = 관측이라, 예보 날 '정답'은 그 날의 기상청 값으로 봄.
(가) 그냥 잇기: ours[T+k]   (나) 옮겨 잇기: ours[T+k] + (kma[T-1] - ours[T-1])
"""
import csv, datetime as dt, sys
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.special import gammainc, ndtri

ASOS = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\raw\raw_kma_asos_daily_info.parquet'
SPI = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\외부_데이터\CLM_SPI_DD_합본_20140101_20261006.csv'
D0, D1 = dt.date(2015, 1, 1), dt.date(2026, 10, 6)
N = (D1 - D0).days + 1
dates = [D0 + dt.timedelta(i) for i in range(N)]
t = pq.read_table(ASOS, columns=['stn_id', 'meas_date', 'sum_rn'])
stn = t['stn_id'].to_numpy(zero_copy_only=False).astype(int)
md = t['meas_date']
md = pc.cast(md, 'date32') if str(md.type) != 'date32[day]' else md
md = md.to_numpy(zero_copy_only=False).astype('datetime64[D]')
rn = t['sum_rn'].to_numpy(zero_copy_only=False).astype(float)
rn = np.where(np.isnan(rn), 0.0, rn)
idx = (md - np.datetime64(D0)).astype(int)
kma = {}
with open(SPI, encoding='cp949') as f:
    r = csv.reader(f); next(r)
    for row in r:
        try:
            s = int(row[0]); d = dt.date.fromisoformat(row[2][:10])
        except Exception:
            continue
        if d < D0 or d > D1:
            continue
        kma.setdefault(s, [np.full(N, np.nan), np.full(N, np.nan)])
        i = (d - D0).days
        for c, j in ((0, 4), (1, 5)):
            v = row[j].strip()
            kma[s][c][i] = float(v) if v not in ('', '-') else np.nan
stations = sorted(set(kma) & set(stn.tolist()))
doy = np.array([(d.timetuple().tm_yday - (1 if (d.month > 2 and d.year % 4 == 0) else 0)) if not (d.month == 2 and d.day == 29) else 59 for d in dates])
fit = np.array([d <= dt.date(2025, 6, 30) for d in dates])


def month_start_idx(n):
    out = np.empty(N, dtype=int)
    for i, d in enumerate(dates):
        m, y = d.month - n, d.year
        while m < 1:
            m += 12; y -= 1
        last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(1)).day
        out[i] = (dt.date(y, m, min(d.day, last)) + dt.timedelta(1) - D0).days
    return out


MS = {3: month_start_idx(3), 6: month_start_idx(6)}


def accum(P, n):  # 달력 n개월(그날 포함) — 우리 식은 한 정의로 통일
    C = np.concatenate([[0.0], np.cumsum(np.nan_to_num(P))]); Cn = np.concatenate([[0], np.cumsum(np.isnan(P))])
    i = np.arange(N); j = MS[n]; out = np.full(N, np.nan); ok = j >= 0
    out[ok] = np.where(Cn[i[ok] + 1] - Cn[j[ok]] == 0, C[i[ok] + 1] - C[j[ok]], np.nan)
    return out


def spi(X, half=15):
    out = np.full(N, np.nan)
    for c in range(1, 366):
        dd = np.minimum(np.abs(doy - c), 365 - np.abs(doy - c))
        x = X[(dd <= half) & fit & ~np.isnan(X)]
        if len(x) < 30:
            continue
        q = np.mean(x <= 0); nz = x[x > 0]
        A = np.log(nz.mean()) - np.mean(np.log(nz)); a = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A); b = nz.mean() / a
        tg = (doy == c) & ~np.isnan(X)
        out[tg] = ndtri(np.clip(q + (1 - q) * gammainc(a, np.maximum(X[tg], 0) / b), 1e-10, 1 - 1e-10))
    return out


Ts = np.array([i for i, d in enumerate(dates) if dt.date(2019, 4, 1) <= d <= dt.date(2025, 6, 19)])
log = [f'지점 {len(stations)}, T {len(Ts)}일(2019-04-01 ~ 2025-06-19, 기상청이 달력 n개월 정의인 기간)']
for n, col in ((3, 0), (6, 1)):
    nv = {k: [] for k in range(11)}; an = {k: [] for k in range(11)}; jump_n, jump_k = [], []
    for s in stations:
        P = np.full(N, np.nan); m = (stn == s) & (idx >= 0) & (idx < N); P[idx[m]] = rn[m]
        o = spi(accum(P, n)); kv = kma[s][col]
        off = kv[Ts - 1] - o[Ts - 1]
        for k in range(11):
            nv[k].append(np.abs(o[Ts + k] - kv[Ts + k])); an[k].append(np.abs(o[Ts + k] + off - kv[Ts + k]))
        jump_n.append(np.abs(o[Ts] - kv[Ts - 1])); jump_k.append(np.abs(kv[Ts] - kv[Ts - 1]))
    f = lambda L: np.nanmean(np.concatenate(L))
    log.append(f'SPI{n}: T 에서 튀는 크기 — 그냥 잇기 {f(jump_n):.3f}, 기상청 값의 하루 변화 {f(jump_k):.3f}')
    log.append(f'SPI{n}: 예보 날 기상청 값과 평균 차이(k = 0·5·10일째) — 그냥 잇기 ' + '·'.join(f'{f(nv[k]):.3f}' for k in (0, 5, 10)) + ' / 옮겨 잇기 ' + '·'.join(f'{f(an[k]):.3f}' for k in (0, 5, 10)))
open(sys.argv[1], 'w', encoding='utf-8').write('\n'.join(log))
