"""SPI(SPI3·SPI6) 우리 식 — 계획서 §4-5, §5 WX1, 부록 A-2(칼럼 설명에 감마 적합 기간).

계획서 문구(§4-5): "직전 3·6개월(달력, 그날 포함) 누적 강수 → 달력일 ±15일을 묶어 감마 적합(Thom 근사,
0 강수 확률 따로) → 표준정규. 감마 적합 기간 = 2015-01-01 ~ 2026-06-30 하나로 고정", "SPI3 2015-03-31·
SPI6 2015-06-30 부터(그 전 NaN)".

순수 함수만(자료 읽기 없음). 날짜는 1970-01-01 부터 일수(int64). 강수 행렬 P 는 (단위, 날짜) — 열 i = d0 + i.

식:
  누적 X_n(t) = Σ P[s .. t], s = (t + 1일) − n 개월(달력, 그날 포함).
  달력일 c(t) ∈ 1..365, 묶음 = 원형 거리 |c(t) − c| ≤ 15 이고 t 가 적합 기간 안이며 X 가 있는 날.
  묶음마다: q = 0 인 날 비율, 양수만으로 Thom 근사
    A = ln(mean(x)) − mean(ln x), α = (1 + √(1 + 4A/3)) / (4A), β = mean(x) / α.
  H(x) = q + (1 − q)·G(x; α, β)(x > 0), H(0) = q → SPI = Φ⁻¹(H).

구현 세부(계획서에 없음 — 담당자 확인):
  - '직전 n 개월(달력, 그날 포함)' 의 시작 = (t + 1일) 에서 n 개월 뺀 날(그 달에 그 날짜가 없으면 그 달 말일).
    월말 t 는 꼭 n 개 달력 달(예: 2015-06-30 → 2015-01-01 ~ 06-30). 이렇게 해야 계획서의 첫 값 날짜
    (SPI3 2015-03-31·SPI6 2015-06-30, 강수 2015-01-01 부터)가 나옴 — 't − n 개월 + 1일' 로 하면 SPI6 첫 값이
    2015-07-01 이 되어 계획서와 다름.
  - 누적 기간 안에 강수 결측(NaN) 날이 하루라도 있으면 그 누적은 NaN(채우지 않음). 기간 시작이 자료 시작 전이면 NaN.
  - 누적이 정확히 0 = 기간 안 양수 강수 날이 없음(합을 직접 더해 반올림 오차 없음).
  - 달력일: 365일 달력, 윤년 3/1 이후는 하루 당김, 2/29 는 2/28 과 같은 달력일(59).
  - '적합 기간' = 누적이 끝나는 날 t 가 그 기간 안인 값으로 적합(기간 밖 날도 같은 모수로 변환).
  - 감마를 못 맞추는 묶음(값 없음·양수 없음·양수 2개 미만·A ≤ 0) 은 그 달력일 값 전부 NaN.
  - H 가 0 또는 1(Φ⁻¹ 가 ±무한)이면 NaN 으로 두고 개수 기록(자르기·바꾸기 없음 — 계획서에 자르기 없음).
  - 묶음 최소 크기 두지 않음(묶음 크기는 모수 표에 기록).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc, ndtri

MONTHS = (3, 6)
HALF_DAYS = 15
N_CAL = 365
FEB29_CAL = 59
STATUS = ("ok", "no_data", "no_positive", "degenerate")


# ---------------------------------------------------------------- 날짜
def _ymd(days: np.ndarray):
    d = np.asarray(days, dtype=np.int64).astype("datetime64[D]")
    y = d.astype("datetime64[Y]").astype(np.int64) + 1970
    m = d.astype("datetime64[M]").astype(np.int64) % 12 + 1
    dd = (d - d.astype("datetime64[M]")).astype(np.int64) + 1
    return y, m, dd


def _is_leap(y):
    y = np.asarray(y)
    return ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)


def _days_in_month(y, m):
    first = (np.asarray(y) - 1970) * 12 + (np.asarray(m) - 1)
    a = first.astype("datetime64[M]")
    return ((a + np.timedelta64(1, "M")).astype("datetime64[D]") - a.astype("datetime64[D]")).astype(np.int64)


def _from_ymd(y, m, d) -> np.ndarray:
    mon = ((np.asarray(y) - 1970) * 12 + (np.asarray(m) - 1)).astype("datetime64[M]")
    return (mon.astype("datetime64[D]") + (np.asarray(d) - 1)).astype(np.int64)


def calendar_day(days) -> np.ndarray:
    """365일 달력의 날 번호 1..365(윤년 3/1 이후 하루 당김, 2/29 = 59)."""
    days = np.atleast_1d(np.asarray(days, dtype=np.int64))
    y, m, d = _ymd(days)
    jan1 = _from_ymd(y, np.ones_like(m), np.ones_like(d))
    doy = days - jan1 + 1
    leap = _is_leap(y)
    c = np.where(leap & (m > 2), doy - 1, doy)
    c = np.where(leap & (m == 2) & (d == 29), FEB29_CAL, c)
    assert c.min(initial=1) >= 1 and c.max(initial=1) <= N_CAL
    return c.astype(np.int64)


def window_start(days, n_months: int) -> np.ndarray:
    """직전 n 개월(달력, 그날 포함) 누적의 첫 날 = (t + 1일) − n 개월(없는 날짜면 그 달 말일)."""
    if isinstance(n_months, bool) or not isinstance(n_months, (int, np.integer)) or n_months < 1:
        raise ValueError(f"n_months 는 양의 정수: {n_months!r}")
    u = np.atleast_1d(np.asarray(days, dtype=np.int64)) + 1
    y, m, d = _ymd(u)
    k = (y * 12 + (m - 1)) - int(n_months)
    y2, m2 = k // 12, k % 12 + 1
    d2 = np.minimum(d, _days_in_month(y2, m2))
    return _from_ymd(y2, m2, d2)


# ---------------------------------------------------------------- 누적
def accumulate(P: np.ndarray, d0: int, n_months: int) -> np.ndarray:
    """P (단위, 날짜) 일강수 → 직전 n 개월(그날 포함) 누적 (단위, 날짜). NaN 규칙은 모듈 설명."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("P 는 (단위, 날짜) 2차원")
    if (P < 0).any():
        raise ValueError(f"음수 강수 {int((P < 0).sum())}칸 — 원자료 확인 필요")
    U, D = P.shape
    t = d0 + np.arange(D, dtype=np.int64)
    L = t - window_start(t, n_months) + 1          # 누적 일수(그날 포함)
    first = np.arange(D) - L + 1                    # 첫 날 열 위치
    ok_start = first >= 0
    nan = np.isnan(P)
    P0 = np.where(nan, 0.0, P)
    s = np.zeros((U, D))
    bad = np.zeros((U, D), dtype=bool)
    for k in range(int(L.max(initial=1))):
        use = (k < L)
        if k:
            sh = np.zeros((U, D))
            sh[:, k:] = P0[:, :-k]
            nb = np.zeros((U, D), dtype=bool)
            nb[:, k:] = nan[:, :-k]
        else:
            sh, nb = P0, nan
        s += np.where(use[None, :], sh, 0.0)
        bad |= nb & use[None, :]
    s[bad | ~ok_start[None, :]] = np.nan
    return s


# ---------------------------------------------------------------- 감마 적합
@dataclass(frozen=True)
class GammaFit:
    """(단위, 달력일 1..365 → 열 0..364) 모수."""
    n_pool: np.ndarray
    n_zero: np.ndarray
    n_pos: np.ndarray
    q: np.ndarray
    A: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    status: np.ndarray   # STATUS 위치(int8)

    @property
    def ok(self) -> np.ndarray:
        return self.status == 0


def pool_columns(cal: np.ndarray, fit_mask: np.ndarray, c: int, half: int = HALF_DAYS) -> np.ndarray:
    """달력일 c 의 묶음에 드는 열(cal = calendar_day(날짜), 원형 거리 ≤ half, 적합 기간 안)."""
    dist = np.abs(np.asarray(cal) - c)
    dist = np.minimum(dist, N_CAL - dist)
    return np.nonzero((dist <= half) & fit_mask)[0]


def fit_gamma(X: np.ndarray, days, fit_mask, half: int = HALF_DAYS) -> GammaFit:
    """누적 X (단위, 날짜)·날짜(일수)·적합 기간 표시 → 달력일마다 감마 모수(Thom 근사, 0 은 q 로 따로)."""
    X = np.asarray(X, dtype=np.float64)
    days = np.asarray(days, dtype=np.int64)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if X.shape[1] != days.shape[0] or fit_mask.shape != days.shape:
        raise ValueError("X 날짜 수와 days·fit_mask 길이가 다름")
    U = X.shape[0]
    shape = (U, N_CAL)
    n_pool = np.zeros(shape, np.int64)
    n_zero = np.zeros(shape, np.int64)
    n_pos = np.zeros(shape, np.int64)
    q = np.full(shape, np.nan)
    A = np.full(shape, np.nan)
    alpha = np.full(shape, np.nan)
    beta = np.full(shape, np.nan)
    status = np.zeros(shape, np.int8)
    cal = calendar_day(days)
    for c in range(1, N_CAL + 1):
        cols = pool_columns(cal, fit_mask, c, half)
        Xc = X[:, cols]
        valid = ~np.isnan(Xc)
        pos = valid & (Xc > 0)
        npool = valid.sum(axis=1)
        nzero = (valid & (Xc == 0)).sum(axis=1)
        npos = pos.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(pos, Xc, 0.0).sum(axis=1) / npos
            mlog = np.where(pos, np.log(np.where(pos, Xc, 1.0)), 0.0).sum(axis=1) / npos
            a_ = np.log(mean) - mlog
            al = (1.0 + np.sqrt(1.0 + 4.0 * a_ / 3.0)) / (4.0 * a_)
            be = mean / al
            qq = nzero / npool
        st = np.zeros(U, np.int8)
        st[npos < 2] = 3
        st[(npos >= 2) & ~(a_ > 0)] = 3
        st[(npool > 0) & (npos == 0)] = 2
        st[npool == 0] = 1
        j = c - 1
        n_pool[:, j], n_zero[:, j], n_pos[:, j] = npool, nzero, npos
        q[:, j] = np.where(npool > 0, qq, np.nan)
        good = st == 0
        A[:, j] = np.where(npos >= 2, a_, np.nan)
        alpha[:, j] = np.where(good, al, np.nan)
        beta[:, j] = np.where(good, be, np.nan)
        status[:, j] = st
    return GammaFit(n_pool, n_zero, n_pos, q, A, alpha, beta, status)


def transform(X: np.ndarray, days, fit: GammaFit) -> tuple[np.ndarray, int]:
    """누적 X → SPI. 반환 (SPI (단위, 날짜), H 가 0·1 이라 NaN 으로 둔 칸 수)."""
    X = np.asarray(X, dtype=np.float64)
    j = calendar_day(days) - 1
    Q, AL, BE, OK = fit.q[:, j], fit.alpha[:, j], fit.beta[:, j], fit.ok[:, j]
    have = ~np.isnan(X) & OK
    H = np.full(X.shape, np.nan)
    pos = have & (X > 0)
    zero = have & (X == 0)
    H[pos] = Q[pos] + (1.0 - Q[pos]) * gammainc(AL[pos], X[pos] / BE[pos])
    H[zero] = Q[zero]
    edge = have & ((H <= 0) | (H >= 1))
    H[edge] = np.nan
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Z = ndtri(H)
    Z[~np.isfinite(Z)] = np.nan
    return Z, int(edge.sum())


def fit_mask_of(days, fit_start: int, fit_end: int) -> np.ndarray:
    days = np.asarray(days, dtype=np.int64)
    return (days >= fit_start) & (days <= fit_end)


@dataclass(frozen=True)
class SpiResult:
    months: int
    X: np.ndarray        # 누적 강수 (단위, 날짜)
    spi: np.ndarray      # (단위, 날짜)
    fit: GammaFit
    n_edge: int          # H 0·1 → NaN 칸 수


def spi_from_rain(P: np.ndarray, d0: int, n_months: int, fit_start: int, fit_end: int,
                  half: int = HALF_DAYS) -> SpiResult:
    """일강수 P (단위, 날짜, 첫 열 = d0) → 우리 식 SPI_n. 적합 기간 = [fit_start, fit_end](일수)."""
    D = np.asarray(P).shape[1]
    days = d0 + np.arange(D, dtype=np.int64)
    X = accumulate(P, d0, n_months)
    fit = fit_gamma(X, days, fit_mask_of(days, fit_start, fit_end), half)
    Z, n_edge = transform(X, days, fit)
    return SpiResult(int(n_months), X, Z, fit, n_edge)


# ---------------------------------------------------------------- 대조
def agreement(a, b) -> dict:
    """같은 칸 대조: 둘 다 값 있는 칸 수·피어슨 상관·평균 절대 차(MAE)·평균 차(a − b)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    k = ~np.isnan(a) & ~np.isnan(b)
    n = int(k.sum())
    if n == 0:
        return {"n": 0, "r": None, "mae": None, "mean_diff": None}
    x, y = a[k], b[k]
    r = None
    if n >= 2 and x.std() > 0 and y.std() > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    return {"n": n, "r": r, "mae": float(np.abs(x - y).mean()), "mean_diff": float((x - y).mean())}


def first_valid_day(Z: np.ndarray, d0: int):
    """값이 있는 첫 날(일수), 없으면 None."""
    anyv = ~np.isnan(Z).all(axis=0)
    return int(d0 + anyv.argmax()) if anyv.any() else None
