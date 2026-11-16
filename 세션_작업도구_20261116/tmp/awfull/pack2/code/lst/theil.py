"""Theil's U = U(y | x) — 계획서 §6 단계 4, 부록 A-6, B-6, B-8.

부록 B-8: `Listeria\\alone_in_the_woods_using_theil_s_u_for_survival.ipynb` 의
`conditional_entropy`·`theil_u` 셀을 옮김 — `theil_u(y, x)`:
  s_xy = H(y | x), s_y = H(y), U = (s_y − s_xy) / s_y, H(y) = 0 이면 1.
부록 A-6: 숫자도 값 그대로 범주, NaN 은 한 범주로 셈.
노트북은 Counter 로 세고, 여기서는 같은 식을 factorize + bincount 로 계산(같은 값 — 시험에서 대조).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def codes_with_nan(x) -> np.ndarray:
    """범주 번호(NaN·None = 한 범주). 숫자·문자 모두 값 그대로."""
    s = pd.Series(x)
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    codes, _ = pd.factorize(s, use_na_sentinel=False)
    return codes.astype(np.int64)


def entropy_counts(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def conditional_entropy(y_codes: np.ndarray, x_codes: np.ndarray) -> float:
    """H(y | x) = Σ p(x, y) · log(p(x) / p(x, y))."""
    n = y_codes.shape[0]
    ny = int(y_codes.max()) + 1
    joint = np.bincount(x_codes * ny + y_codes)
    px = np.bincount(x_codes)
    nz = np.flatnonzero(joint)
    pxy = joint[nz] / n
    p_x = px[nz // ny] / n
    return float((pxy * np.log(p_x / pxy)).sum())


def theils_u(y, x) -> float:
    """U(y | x): x 가 y 를 설명하는 정도(0~1)."""
    yc = codes_with_nan(y)
    xc = codes_with_nan(x)
    if yc.shape[0] != xc.shape[0]:
        raise ValueError("길이 다름")
    if yc.shape[0] == 0:
        raise ValueError("빈 자료")
    s_y = entropy_counts(np.bincount(yc))
    if s_y == 0:
        return 1.0
    return (s_y - conditional_entropy(yc, xc)) / s_y
