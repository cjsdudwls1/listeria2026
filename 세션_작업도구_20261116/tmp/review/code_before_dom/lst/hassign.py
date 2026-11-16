"""h 배정·u_tie — 계획서 부록 A-1, A-12, §3.

- h = int.from_bytes(md5(lims_data_key)[:8], 'big') % 60 + 1
- u_tie = int.from_bytes(md5(f'{lims_data_key}|{h}')[:8], 'big') / 2**64
- 동점 많은 판 다시 채점(부록 A-12) = md5(`lims_data_key`|h|1) → `rescore=True`
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

H_MAX = 60


def _md5_u64(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def _check_key(key) -> str:
    if not isinstance(key, str) or key == "":
        raise TypeError(f"lims_data_key 는 빈 값이 아닌 문자열: {key!r}")
    return key


def _check_h(h) -> int:
    hi = int(h)
    if hi != h or not (1 <= hi <= H_MAX):
        raise ValueError(f"h 는 1~{H_MAX} 정수: {h!r}")
    return hi


def assign_h(key: str) -> int:
    return _md5_u64(_check_key(key)) % H_MAX + 1


def assign_h_many(keys: Iterable[str]) -> np.ndarray:
    return np.fromiter((assign_h(k) for k in keys), dtype=np.int64)


def _tie_string(key: str, h: int, rescore: bool) -> str:
    return f"{key}|{h}|1" if rescore else f"{key}|{h}"


def u_tie_int(key: str, h: int, rescore: bool = False) -> int:
    """u_tie 의 분자(정수, 0 ≤ · < 2**64). 정렬엔 이 값이 u_tie 와 같은 순서(정확)."""
    return _md5_u64(_tie_string(_check_key(key), _check_h(h), rescore))


def u_tie(key: str, h: int, rescore: bool = False) -> float:
    return u_tie_int(key, h, rescore) / 2 ** 64


def u_tie_many(keys: Iterable[str], hs: Iterable[int], rescore: bool = False,
               exact: bool = True) -> np.ndarray:
    """(시료, h) 쌍마다 u_tie.

    기본 exact=True: 분자 정수(uint64) — 순서는 u_tie 와 같고 float64 로 바꿀 때 생길 수
    있는 같은 값(53비트 반올림)이 없음(채점용). exact=False 면 u_tie(float64). metrics 는 둘 다 받음.
    """
    vals = [u_tie_int(k, h, rescore) for k, h in zip(keys, hs, strict=True)]
    arr = np.array(vals, dtype=np.uint64)
    return arr if exact else arr.astype(np.float64) / 2.0 ** 64
