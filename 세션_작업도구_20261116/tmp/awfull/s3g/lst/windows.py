"""window 생성기(단계 0-5) — 계획서 §1-1, §4-1, §4-2, §2-2 RR8, 부록 B-7.

정의(§1-1):
  T = d − h, E = min(d, T + 10), S = E − W + 1, g = d − E = max(0, h − 10),
  R = min(d, E + 1), 관측 칸 = S ~ T−1, 예보 칸 = T ~ E(d 당일도 예보 칸),
  관측 일수 = max(0, W − min(h + 1, 11)).
요약(§4-2, 변수마다): E 부터 거꾸로 10일 구간 평균 b01(E−9 ~ E)·b02(E−19 ~ E−10)…,
  window 평균·표준편차(ddof 0)·최소·최대·기울기(날짜(일)에 대한 최소제곱, 단위/일),
  NaN 무시, 관측·예보 칸을 한 줄로.
원값: lag001 = E, lag002 = E−1, …, lagW = S. 시퀀스: S ~ E 시간순 W일.
이동평균 차감: a_t = x_t − mean([t − N_ma, t − 1] 의 값 있는 날), 다 결측이거나 x_t 결측이면
  결측 → 요약 판 = window 평균·최근 10일 평균·최댓값, 원값 판 = `<변수>_anom{N_ma}_lag001..`.

이 모듈은 일자료 표(날짜 × 단위 × 변수, `DailyCube`)를 받는 순수 함수만 둠(자료 읽기 없음).
예보 칸 값: 기본은 같은 표(완전 예보 가정 — 예보 칸 = 관측값, §0). 국내 SPI 처럼 T 부터 다른
값을 이어 붙일 때는 `forecast_cube` 를 줌(§4-5·§7-1: 이음 위치 T).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .folds import as_date, day_ints, to_days
from .hassign import H_MAX

LEAD = 10                 # E = min(d, T + 10)
FORECAST_LEN = LEAD + 1   # 예보 길이 = 발표 당일 포함 11일
BIN = 10                  # 요약 10일 구간
W_GRID = tuple(range(10, 101, 10))
RR8_W, RR8_H = 100, 60    # RR8 최악 기준(S = d − 149)
DOMESTIC_WEATHER_START = "2015-01-01"  # §4-3: 국내 값·차감은 2015-01-01 부터(그 전 NaN)
SUMMARY_STATS = ("mean", "std", "min", "max", "slope")
ANOM_SUMMARY_STATS = ("mean", "mean10", "max")


def _days(x) -> np.ndarray:
    """정수(일수)는 그대로, 날짜는 일수로(folds.day_ints)."""
    return day_ints(x)


def _check_h(h) -> np.ndarray:
    h = np.atleast_1d(np.asarray(h))
    hi = h.astype(np.int64)
    if not (hi == h).all() or hi.min(initial=1) < 1 or hi.max(initial=1) > H_MAX:
        raise ValueError("h 는 1~60 정수")
    return hi


def _check_w(W: int) -> int:
    if isinstance(W, bool) or not isinstance(W, (int, np.integer)) or W < 1:
        raise ValueError(f"W 는 양의 정수: {W!r}")
    return int(W)


# ---------------------------------------------------------------- 경계
@dataclass(frozen=True)
class Bounds:
    d: np.ndarray
    h: np.ndarray
    W: int
    T: np.ndarray
    E: np.ndarray
    S: np.ndarray
    R: np.ndarray
    g: np.ndarray
    n_obs: np.ndarray   # 관측 칸 일수(S ~ T−1)
    n_fc: np.ndarray    # 예보 칸 일수(T ~ E)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"d": as_date(self.d), "h": self.h, "T": as_date(self.T), "S": as_date(self.S),
                             "E": as_date(self.E), "R": as_date(self.R), "g": self.g,
                             "n_obs": self.n_obs, "n_fc": self.n_fc})


def window_bounds(d, h, W: int) -> Bounds:
    """§1-1 경계. d·h 는 같은 길이(또는 스칼라 브로드캐스트)."""
    W = _check_w(W)
    d = _days(d)
    h = _check_h(h)
    d, h = np.broadcast_arrays(d, h)
    d = d.astype(np.int64).copy()
    h = h.astype(np.int64).copy()
    T = d - h
    E = np.minimum(d, T + LEAD)
    S = E - W + 1
    g = d - E
    R = np.minimum(d, E + 1)
    n_fc = np.minimum(W, E - T + 1)
    n_obs = W - n_fc
    # §1-1·§4-1 식과 같은지 확인
    assert (g == np.maximum(0, h - LEAD)).all()
    assert (n_obs == np.maximum(0, W - np.minimum(h + 1, FORECAST_LEN))).all()
    return Bounds(d, h, W, T, E, S, R, g, n_obs, n_fc)


def forecast_day_mask(b: Bounds) -> np.ndarray:
    """(행, W) — window 의 i 번째 날(S + i)이 예보 칸(≥ T)이면 True."""
    dates = b.S[:, None] + np.arange(b.W)[None, :]
    return dates >= b.T[:, None]


# ---------------------------------------------------------------- 일자료 표
@dataclass(frozen=True)
class DailyCube:
    """일자료 표: values[단위, 날짜, 변수], 날짜는 start 부터 하루 간격 연속."""
    start: int                       # 첫 날(일수)
    units: tuple[str, ...]
    variables: tuple[str, ...]
    values: np.ndarray               # float64 (n_units, n_days, n_vars)

    def __post_init__(self):
        v = self.values
        if v.ndim != 3 or v.shape[0] != len(self.units) or v.shape[2] != len(self.variables):
            raise ValueError("values 모양이 단위·변수 수와 다름")
        if len(set(self.units)) != len(self.units) or len(set(self.variables)) != len(self.variables):
            raise ValueError("단위·변수 이름 중복")

    @property
    def n_days(self) -> int:
        return self.values.shape[1]

    @property
    def end(self) -> int:
        return self.start + self.n_days - 1

    @classmethod
    def from_long(cls, df: pd.DataFrame, unit_col: str, date_col: str,
                  var_cols: Sequence[str]) -> "DailyCube":
        """긴 표(단위·날짜·변수 칼럼) → DailyCube. (단위, 날짜) 중복이면 멈춤.

        구현 세부(계획서에 없음): 표에 없는 (단위, 날짜) 는 NaN(값을 채우지 않음 — 원표에
        그 행이 없다는 뜻 그대로), 날짜 축은 표의 최소 ~ 최대 날짜.
        """
        if df[unit_col].isna().any() or df[date_col].isna().any():
            raise ValueError("단위·날짜 칼럼에 null")
        if df.duplicated([unit_col, date_col]).any():
            raise ValueError("(단위, 날짜) 중복")
        units = tuple(sorted(df[unit_col].astype(str).unique()))
        dd = to_days(df[date_col])
        start, end = int(dd.min()), int(dd.max())
        ui = pd.Index(units).get_indexer(df[unit_col].astype(str))
        vals = np.full((len(units), end - start + 1, len(var_cols)), np.nan)
        vals[ui, dd - start, :] = df[list(var_cols)].to_numpy(dtype=np.float64)
        return cls(start, units, tuple(var_cols), vals)

    def unit_index(self, codes, allow_missing: bool = False) -> np.ndarray:
        """단위 코드 → 위치. 표에 없는 코드: allow_missing 이면 −1(그 행 전부 NaN), 아니면 멈춤."""
        idx = pd.Index(self.units).get_indexer(pd.Index(np.asarray(codes).astype(str)))
        if not allow_missing and (idx < 0).any():
            bad = sorted(set(np.asarray(codes).astype(str)[idx < 0]))[:10]
            raise KeyError(f"일자료 표에 없는 단위: {bad}")
        return idx.astype(np.int64)

    def select(self, variables: Sequence[str]) -> "DailyCube":
        vi = [self.variables.index(v) for v in variables]
        return DailyCube(self.start, self.units, tuple(variables), self.values[:, :, vi])


def _gather(cube: DailyCube, uidx: np.ndarray, first: np.ndarray, W: int) -> np.ndarray:
    """(행, W, 변수): 날짜 first ~ first + W − 1. 표 밖 날짜·없는 단위 = NaN."""
    n = uidx.shape[0]
    di = (first - cube.start)[:, None] + np.arange(W)[None, :]
    valid = (di >= 0) & (di < cube.n_days) & (uidx >= 0)[:, None]
    out = np.full((n, W, len(cube.variables)), np.nan)
    ub = np.broadcast_to(uidx[:, None], (n, W))
    out[valid] = cube.values[ub[valid], di[valid]]
    return out


def gather_window(cube: DailyCube, codes, b: Bounds, forecast_cube: DailyCube | None = None,
                  allow_missing_units: bool = False, rows: slice | np.ndarray | None = None) -> np.ndarray:
    """window 값 (행, W, 변수), S ~ E 시간순. forecast_cube 가 있으면 날짜 ≥ T 칸은 그 표 값."""
    sel = slice(None) if rows is None else rows
    codes = np.asarray(codes)[sel]
    S = b.S[sel]
    x = _gather(cube, cube.unit_index(codes, allow_missing_units), S, b.W)
    if forecast_cube is not None:
        if forecast_cube.variables != cube.variables:
            raise ValueError("예보 표 변수가 관측 표와 다름")
        xf = _gather(forecast_cube, forecast_cube.unit_index(codes, allow_missing_units), S, b.W)
        fc = (S[:, None] + np.arange(b.W)[None, :]) >= b.T[sel][:, None]
        x[fc] = xf[fc]
    return x


# ---------------------------------------------------------------- 요약·원값
def _nan_stats(x: np.ndarray, axis: int):
    ok = ~np.isnan(x)
    cnt = ok.sum(axis=axis)
    s = np.where(ok, x, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return ok, cnt, mean


def _nan_minmax(x: np.ndarray, axis: int, cnt: np.ndarray):
    mn = np.where(cnt > 0, np.where(np.isnan(x), np.inf, x).min(axis=axis), np.nan)
    mx = np.where(cnt > 0, np.where(np.isnan(x), -np.inf, x).max(axis=axis), np.nan)
    return mn, mx


def summary_names(var_names: Sequence[str], W: int) -> list[str]:
    """요약 칼럼 이름(변수마다 b01..bK, mean, std, min, max, slope).

    구현 세부(계획서에 없음): 이름 꼴 `<변수>_b01`·`_mean`·`_std`·`_min`·`_max`·`_slope`
    (계획서는 b01 과 `<변수>_mean` 만 적음), 칼럼 순서 = 변수 순 → 위 순.
    """
    K = W // BIN
    names = []
    for v in var_names:
        names += [f"{v}_b{k:02d}" for k in range(1, K + 1)]
        names += [f"{v}_{s}" for s in SUMMARY_STATS]
    return names


def summary_block(x: np.ndarray) -> np.ndarray:
    """x (행, W, 변수) 시간순 → (행, 변수 × (K + 5)), summary_names 순."""
    n, W, V = x.shape
    if W % BIN:
        raise ValueError(f"요약 판 W 는 10 의 배수: {W}")
    K = W // BIN
    rev = x[:, ::-1, :]                                  # 0 = E
    bins = rev.reshape(n, K, BIN, V)
    _, _, bmean = _nan_stats(bins, axis=2)               # (n, K, V)
    ok, cnt, mean = _nan_stats(x, axis=1)                # (n, V)
    dev = np.where(ok, x - np.where(np.isnan(mean), 0.0, mean)[:, None, :], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.where(cnt > 0, np.sqrt((dev ** 2).sum(axis=1) / np.maximum(cnt, 1)), np.nan)
    mn, mx = _nan_minmax(x, 1, cnt)
    t = np.arange(W, dtype=np.float64)[None, :, None]
    tbar = (np.where(ok, t, 0.0).sum(axis=1) / np.maximum(cnt, 1))[:, None, :]
    tdev = np.where(ok, t - tbar, 0.0)
    sxx = (tdev ** 2).sum(axis=1)
    sxy = (tdev * dev).sum(axis=1)
    # 구현 세부(계획서에 없음): 값 있는 날이 2일 미만이면 기울기 NaN(정의 안 됨)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where((cnt >= 2) & (sxx > 0), sxy / np.where(sxx > 0, sxx, 1.0), np.nan)
    cols = []
    for v in range(V):
        cols.append(bmean[:, :, v])
        cols.append(np.stack([mean[:, v], std[:, v], mn[:, v], mx[:, v], slope[:, v]], axis=1))
    return np.concatenate(cols, axis=1)


def raw_lag_names(var_names: Sequence[str], W: int) -> list[str]:
    """원값 칼럼 이름: `<변수>_lag001`(= E) … `<변수>_lag{W:03d}`(= S)."""
    return [f"{v}_lag{j:03d}" for v in var_names for j in range(1, W + 1)]


def raw_lag_block(x: np.ndarray) -> np.ndarray:
    """x (행, W, 변수) 시간순 → (행, 변수 × W), lag001 = E."""
    n, W, V = x.shape
    rev = x[:, ::-1, :]                                  # 0 = E
    return np.ascontiguousarray(rev.transpose(0, 2, 1)).reshape(n, V * W)


# ---------------------------------------------------------------- 이동평균 차감
def anomaly_name(var: str, n_ma: int) -> str:
    """구현 세부(계획서가 모호해 정함): `{N_ma}` = 정수 그대로(예: anom5·anom15)."""
    return f"{var}_anom{int(n_ma)}"


def anomaly_cube(cube: DailyCube, n_ma: int) -> DailyCube:
    """§4-2 차감 일자료: a_t = x_t − ([t − N_ma, t − 1] 값 있는 날 평균).

    표 시작 전 날은 결측으로 봄(평균에서 빠짐). 변수 이름 = `<변수>_anom{N_ma}`.
    """
    if isinstance(n_ma, bool) or not isinstance(n_ma, (int, np.integer)) or n_ma < 1:
        raise ValueError(f"N_ma 는 양의 정수: {n_ma!r}")
    x = cube.values
    U, D, V = x.shape
    s = np.zeros_like(x)
    c = np.zeros(x.shape, dtype=np.int64)
    for k in range(1, int(n_ma) + 1):
        if k >= D:
            break
        prev = x[:, :D - k, :]
        ok = ~np.isnan(prev)
        s[:, k:, :] += np.where(ok, prev, 0.0)
        c[:, k:, :] += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        base = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    a = x - base  # x_t 결측 또는 base 결측이면 NaN
    names = tuple(anomaly_name(v, n_ma) for v in cube.variables)
    return DailyCube(cube.start, cube.units, names, a)


def anomaly_summary_names(anom_var_names: Sequence[str]) -> list[str]:
    """구현 세부(계획서에 없음): `<변수>_anom{N}_mean`·`_mean10`·`_max`."""
    return [f"{v}_{s}" for v in anom_var_names for s in ANOM_SUMMARY_STATS]


def anomaly_summary_block(xa: np.ndarray) -> np.ndarray:
    """차감 window (행, W, 변수) → window 평균·최근 10일(E−9 ~ E) 평균·최댓값(NaN 무시, 값 있는 날 0 이면 결측)."""
    n, W, V = xa.shape
    _, cnt, mean = _nan_stats(xa, axis=1)
    _, _, mean10 = _nan_stats(xa[:, max(0, W - BIN):, :], axis=1)
    _, mx = _nan_minmax(xa, 1, cnt)
    return np.stack([mean, mean10, mx], axis=2).reshape(n, V * len(ANOM_SUMMARY_STATS))


# ---------------------------------------------------------------- 묶어서
MODES = ("sum", "raw")  # 판 이름의 입력 방식(요약 = sum, 원값 = raw)


def window_feature_names(var_names: Sequence[str], W: int, mode: str, anomaly: bool = False) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"입력 방식은 {MODES}: {mode!r}")
    if mode == "raw":
        return raw_lag_names(var_names, W)
    return anomaly_summary_names(var_names) if anomaly else summary_names(var_names, W)


def window_features(cube: DailyCube, codes, d, h, W: int, mode: str, *, anomaly: bool = False,
                    forecast_cube: DailyCube | None = None, allow_missing_units: bool = False,
                    chunk_rows: int = 10_000, dtype=np.float64) -> pd.DataFrame:
    """행((시료, h) 쌍)마다 window 칼럼(트리 입력).

    - mode 'sum' = 요약 칼럼만, 'raw' = lag001~lagW 만(§4-2).
    - anomaly=True 면 cube 는 `anomaly_cube` 결과(요약 판 = 3칼럼, 원값 판 = lag).
    - chunk_rows: 메모리 묶음 크기(결과 불변). dtype: 기본 float64(4-a float32 대체 때만 float32).
    구현 세부(계획서에 없음): chunk_rows 기본 10,000(공용 서버 메모리).
    """
    b = window_bounds(d, h, W)
    names = window_feature_names(cube.variables, W, mode, anomaly)
    n = b.d.shape[0]
    out = np.empty((n, len(names)), dtype=dtype)
    for lo in range(0, n, chunk_rows):
        sl = slice(lo, min(n, lo + chunk_rows))
        x = gather_window(cube, codes, b, forecast_cube, allow_missing_units, rows=sl)
        if mode == "raw":
            blk = raw_lag_block(x)
        else:
            blk = anomaly_summary_block(x) if anomaly else summary_block(x)
        out[sl] = blk
    return pd.DataFrame(out, columns=names)


def sequences(cube: DailyCube, codes, d, h, W: int, *, forecast_cube: DailyCube | None = None,
              allow_missing_units: bool = False, dtype=np.float64) -> np.ndarray:
    """딥러닝 시퀀스 (쌍, W, 채널), S ~ E 시간순 — (시료, h) 쌍마다(§7-1). 표준화는 여기서 안 함."""
    b = window_bounds(d, h, W)
    return gather_window(cube, codes, b, forecast_cube, allow_missing_units).astype(dtype, copy=False)


# ---------------------------------------------------------------- RR8
def rr8_mask(d, src_route, weather_start: str = DOMESTIC_WEATHER_START) -> np.ndarray:
    """§2-2 RR8: W 100·h 60 최악 기준 S(= d − 149)가 국내 기상 값 시작(2015-01-01) 전이면
    True(모든 판에서 제외). 국내('domestic') 행만."""
    b = window_bounds(d, RR8_H, RR8_W)
    start = int(to_days(weather_start)[0])
    return (np.asarray(src_route) == "domestic") & (b.S < start)
