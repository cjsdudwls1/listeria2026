# -*- coding: utf-8 -*-
"""기상 창 꺼내기와 트리용 특성(요약 / 일별 원값).

정렬: 목표일 d, 앞당김 h -> 예측하는 날 T = d - h. 창 = T-L .. T-1 (L일, T 당일 제외).
창 배열은 시간순: 위치 0 = T-L, 위치 L-1 = T-1.
"""

import numpy as np

import config as C

V = len(C.WEATHER_VARS)


def feature_names(mode: str, L: int) -> list:
    nb = L // C.BLOCK
    names = []
    for v in C.WEATHER_VARS:
        if mode == "summary":
            names += [f"{v}_b{k:02d}" for k in range(1, nb + 1)]          # b01 = T-10..T-1
            names += [f"{v}_mean", f"{v}_std", f"{v}_min", f"{v}_max", f"{v}_slope"]
        elif mode == "raw":
            names += [f"{v}_lag{j:03d}" for j in range(1, L + 1)]          # lag j = T-j
        else:
            raise ValueError(mode)
    return names


def tree_columns(mode: str, L: int, with_cats: bool) -> list:
    lead = list(C.CAT_COLS) if with_cats else []
    return lead + ["h"] + feature_names(mode, L)


def gather(W: np.ndarray, s: np.ndarray, t: np.ndarray, L: int) -> np.ndarray:
    """(n, L, V) 시간순 창. t = T 의 날짜 인덱스."""
    idx = t[:, None] - L + np.arange(L)[None, :]
    return W[s[:, None], idx]


def summary_from_window(win: np.ndarray, L: int) -> np.ndarray:
    """변수마다 [최근부터 10일 구간 평균 L/10개, 평균, 표준편차(ddof=0), 최소, 최대, 기울기]. NaN 무시."""
    n = win.shape[0]
    nb = L // C.BLOCK
    m = ~np.isnan(win)
    v0 = np.where(m, win, 0.0).astype(np.float64)
    mf = m.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cnt_b = mf.reshape(n, nb, C.BLOCK, V).sum(2)
        sum_b = v0.reshape(n, nb, C.BLOCK, V).sum(2)
        bmean = (sum_b / cnt_b)[:, ::-1, :]                    # 구간1 = 가장 최근(T-10..T-1)
        cnt = cnt_b.sum(1)
        s1 = sum_b.sum(1)
        mean = s1 / cnt
        dev = np.where(m, win - mean[:, None, :], 0.0)
        std = np.sqrt((dev * dev).sum(1) / cnt)
        mn = np.where(m, win, np.inf).min(1).astype(np.float64)
        mx = np.where(m, win, -np.inf).max(1).astype(np.float64)
        mn[cnt == 0] = np.nan
        mx[cnt == 0] = np.nan
        x = np.arange(L, dtype=np.float64)[None, :, None]         # 시간순 일 번호
        sx = (mf * x).sum(1)
        sxx = (mf * x * x).sum(1)
        sxy = (v0 * x).sum(1)
        slope = (cnt * sxy - sx * s1) / (cnt * sxx - sx * sx)
        slope[cnt < 2] = np.nan
    parts = np.concatenate(
        [bmean.transpose(0, 2, 1), mean[..., None], std[..., None], mn[..., None], mx[..., None], slope[..., None]],
        axis=2)                                                      # (n, V, nb+5)
    return parts.reshape(n, V * (nb + 5)).astype(np.float32)


def raw_from_window(win: np.ndarray) -> np.ndarray:
    """변수마다 lag1(T-1) .. lagL(T-L)."""
    n, L, _ = win.shape
    return np.ascontiguousarray(win[:, ::-1, :].transpose(0, 2, 1)).reshape(n, V * L)


def n_weather_features(mode: str, L: int) -> int:
    return V * (L // C.BLOCK + 5) if mode == "summary" else V * L


def build_tree_matrix(W, s, t, h, L, mode, cats=None, chunk=4096) -> np.ndarray:
    """트리 입력 행렬 float32: [범주 코드 4개(cats 가 있을 때)] + [h] + [기상 특성]. 결측은 NaN 그대로."""
    n = len(s)
    lead = 0 if cats is None else cats.shape[1]
    F = n_weather_features(mode, L)
    X = np.empty((n, lead + 1 + F), dtype=np.float32)
    if lead:
        X[:, :lead] = cats
    X[:, lead] = h
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        win = gather(W, s[a:b], t[a:b], L)
        X[a:b, lead + 1:] = summary_from_window(win, L) if mode == "summary" else raw_from_window(win)
    return X
