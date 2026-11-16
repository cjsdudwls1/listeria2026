"""지표 — 계획서 §1-2, 부록 A-12, 부록 B-3.

PR-AUC = sklearn `precision_recall_curve`(drop_intermediate=False) → `auc`(trapz).
AP(평균 정밀도) 함수는 이 모듈·패키지 어디에서도 쓰지 않음(불러오지도 않음).

- 동점 풀기(부록 A-12): 원 예측값(float64) 큰 순 → 같으면 u_tie 작은 순으로 정렬,
  순위를 점수(쌍 수 − 순위)로 바꿔 곡선 계산 → 쌍마다 한 점. 상위 q% 도 같은 정렬.
- sklearn 기본(같은 값 묶음) PR-AUC 는 기록용으로 따로(`pr_auc_sklearn_default`).
- threshold(§1-2) 는 원 예측값, 같은 값 묶음을 함께.
- 검출 0건·전부 검출인 묶음(h 등)은 NaN, 구간 가중은 NaN 재가중(§1-2 표).
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Mapping

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

H_MAX = 60
BANDS = ((1, 20, 0.5), (21, 40, 0.3), (41, 60, 0.2))  # (h 시작, h 끝, 가중치)
MIN_RECALL = 0.8           # threshold: Recall ≥ 0.8 중 Precision 최대
TOP_Q = (1, 5, 10, 20)     # 상위 q% 포착률
MANY_TIES_RATIO = 0.01     # 가장 큰 동점 묶음 ≥ CV validation 쌍의 1% → '동점 많음'
ROUND_DIGITS = 4           # 점수 비교 = 소수 4자리 반올림 뒤


# ---------------------------------------------------------------- 입력 점검
def _y(y) -> np.ndarray:
    a = np.asarray(y)
    if a.dtype == bool:
        return a.astype(np.int64)
    if a.size and not np.isin(a, (0, 1)).all():
        raise ValueError("y 는 0/1 만")
    return a.astype(np.int64)


def _score(s) -> np.ndarray:
    a = np.asarray(s, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("예측값에 NaN·inf")
    return a


def _u(u, n) -> np.ndarray:
    a = np.asarray(u)
    if a.shape != (n,):
        raise ValueError("u_tie 길이가 다름")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise ValueError("u_tie 에 NaN")
    return a


def _h(h, n) -> np.ndarray:
    a = np.asarray(h)
    if a.shape != (n,):
        raise ValueError("h 길이가 다름")
    ai = a.astype(np.int64)
    if not (ai == a).all() or ai.min(initial=1) < 1 or ai.max(initial=1) > H_MAX:
        raise ValueError("h 는 1~60 정수")
    return ai


def _check_len(y, s):
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y·예측 길이가 다름")


# ---------------------------------------------------------------- 동점 풀기
def tie_broken_order(score, u_tie) -> np.ndarray:
    """정렬 순서(위 = 먼저): 원 예측값 큰 순 → u_tie 작은 순.

    구현 세부(계획서에 없음): 예측값·u_tie 가 모두 같으면 입력 순서(안정 정렬).
    """
    s = _score(score)
    u = _u(u_tie, s.shape[0])
    return np.lexsort((u, -s))


def tie_broken_scores(score, u_tie) -> np.ndarray:
    """순위(1 = 맨 위)를 점수(쌍 수 − 순위)로 바꿈(부록 A-12) — 동점 없는 점수."""
    order = tie_broken_order(score, u_tie)
    n = order.shape[0]
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    return (n - ranks).astype(np.float64)


# ---------------------------------------------------------------- PR-AUC
def _degenerate(y: np.ndarray) -> bool:
    n_pos = int(y.sum())
    return y.shape[0] == 0 or n_pos == 0 or n_pos == y.shape[0]


def _trapz_pr(y: np.ndarray, s: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y, s, drop_intermediate=False)
    return float(auc(recall, precision))


def pr_auc(y, score, u_tie) -> float:
    """trapz PR-AUC(동점은 u_tie 로 풀어 쌍마다 한 점). 검출 0건·전부 검출·빈 묶음 = NaN."""
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    if _degenerate(y):
        return math.nan
    return _trapz_pr(y, tie_broken_scores(s, u_tie))


def pr_auc_sklearn_default(y, score) -> float:
    """sklearn 기본(같은 값 묶음) trapz PR-AUC — 기록용, 고르기엔 안 씀."""
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    if _degenerate(y):
        return math.nan
    return _trapz_pr(y, s)


def _by_h(fn, y, h, *arrays) -> np.ndarray:
    out = np.full(H_MAX, np.nan)
    order = np.argsort(h, kind="stable")
    hs = h[order]
    bounds = np.searchsorted(hs, np.arange(1, H_MAX + 2))
    for k in range(H_MAX):
        idx = order[bounds[k]:bounds[k + 1]]
        if idx.size:
            out[k] = fn(y[idx], *(a[idx] for a in arrays))
    return out


def pr_auc_by_h(y, score, h, u_tie) -> np.ndarray:
    """h 별 PR-AUC, 길이 60(인덱스 0 = h 1). 검출 0건·전부 검출·쌍 없음 = NaN."""
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    h = _h(h, y.shape[0])
    u = _u(u_tie, y.shape[0])
    return _by_h(pr_auc, y, h, s, u)


def pr_auc_sklearn_default_by_h(y, score, h) -> np.ndarray:
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    h = _h(h, y.shape[0])
    return _by_h(pr_auc_sklearn_default, y, h, s)


def band_means(values_by_h) -> tuple[float, float, float]:
    """구간(h 1~20·21~40·41~60) 평균, NaN 인 h 는 뺌, 구간 통째 NaN 이면 NaN."""
    v = np.asarray(values_by_h, dtype=np.float64)
    if v.shape != (H_MAX,):
        raise ValueError("h 별 값은 길이 60")
    out = []
    for lo, hi, _ in BANDS:
        seg = v[lo - 1:hi]
        ok = ~np.isnan(seg)
        out.append(float(seg[ok].mean()) if ok.any() else math.nan)
    return tuple(out)


def band_weighted(values_by_h) -> float:
    """0.5·0.3·0.2 가중, 통째 NaN 구간은 빼고 남은 가중치를 합 1 로 다시 나눔. 전부 NaN = NaN."""
    means = band_means(values_by_h)
    num = 0.0
    den = 0.0
    for m, (_, _, w) in zip(means, BANDS):
        if not math.isnan(m):
            num += w * m
            den += w
    return num / den if den > 0 else math.nan


def pr_auc_band_weighted(y, score, h, u_tie) -> float:
    """주 지표: 구간 가중 PR-AUC."""
    return band_weighted(pr_auc_by_h(y, score, h, u_tie))


def pr_auc_pooled(y, score, u_tie) -> float:
    """60일 통 PR-AUC: 모든 (시료, h) 쌍을 이어 한 번."""
    return pr_auc(y, score, u_tie)


def positive_rate_by_h(y, h) -> np.ndarray:
    y = _y(y)
    h = _h(h, y.shape[0])
    n = np.bincount(h, minlength=H_MAX + 1)[1:].astype(np.float64)
    pos = np.bincount(h, weights=y, minlength=H_MAX + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, pos / n, np.nan)


def lift_band_weighted(pr_by_h, y, h) -> float:
    """구간 가중 PR-AUC ÷ (h 별 검출 비율을 같은 가중으로 합친 값).

    구현 세부(계획서가 모호해 정함): '같은 가중' = PR-AUC 가 NaN 인 h 는 검출 비율에서도
    빼고 같은 NaN 재가중을 적용(분자·분모가 같은 h 집합).
    """
    pr = np.asarray(pr_by_h, dtype=np.float64)
    rate = positive_rate_by_h(y, h)
    rate = np.where(np.isnan(pr), np.nan, rate)
    base = band_weighted(rate)
    num = band_weighted(pr)
    if math.isnan(num) or math.isnan(base) or base == 0:
        return math.nan
    return num / base


def lift_pooled(pooled_pr_auc: float, y) -> float:
    """60일 통 PR-AUC ÷ 전체 쌍 검출 비율."""
    y = _y(y)
    if y.size == 0 or y.sum() == 0 or math.isnan(pooled_pr_auc):
        return math.nan
    return pooled_pr_auc / float(y.mean())


# ---------------------------------------------------------------- 상위 q%
def top_q_capture(y, score, u_tie, qs: Iterable[int] = TOP_Q) -> dict[int, float]:
    """상위 q% 쌍(정렬 위에서 ⌈q% × 쌍 수⌉)이 잡은 검출 ÷ 전체 검출(동점은 u_tie 정렬)."""
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    n = y.shape[0]
    n_pos = int(y.sum())
    order = tie_broken_order(s, u_tie)
    cum = np.concatenate([[0], np.cumsum(y[order])])
    out = {}
    for q in qs:
        if isinstance(q, bool) or not isinstance(q, int) or not (0 < q <= 100):
            raise ValueError(f"q 는 1~100 정수: {q!r}")
        k = -(-q * n // 100)  # 정수 올림
        out[q] = math.nan if n_pos == 0 else float(cum[k]) / n_pos
    return out


# ---------------------------------------------------------------- 반올림 비교
def round4(x: float) -> float:
    """소수 4자리 반올림(§1-2).

    구현 세부(계획서에 없음): 사사오입(ROUND_HALF_UP), float 의 최단 10진 표현 기준.
    """
    if math.isnan(x):
        return math.nan
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-ROUND_DIGITS), rounding=ROUND_HALF_UP))


def is_improvement(new: float, old: float) -> bool:
    """§1-2 동점: 4자리 반올림 뒤 비교, 같으면 상승 아님. NaN 이면 상승 아님."""
    a, b = round4(new), round4(old)
    if math.isnan(a) or math.isnan(b):
        return False
    return a > b


# ---------------------------------------------------------------- threshold
def _ratio_round4_units(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num/den 을 소수 4자리 사사오입한 값 × 10^4(정수, 정확 계산)."""
    num = num.astype(np.int64)
    den = den.astype(np.int64)
    scale = 10 ** ROUND_DIGITS
    return (2 * scale * num + den) // (2 * den)


def threshold_candidates(y, score):
    """같은 값 묶음별(예측 ≥ t 를 양성) 후보: (thresholds 내림차순, TP, FP)."""
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    ys = y[order]
    last = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])  # 같은 값 묶음의 마지막 위치
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    return ss[last], tp, fp


def select_threshold(y, score) -> dict:
    """§1-2: Recall ≥ 0.8 인 값 중 Precision 최대 → 같으면 Recall → 큰 값.

    - Recall ≥ 0.8 판정은 반올림 없이 정확히(5·TP ≥ 4·검출 수).
    - 구현 세부(계획서가 모호해 정함): Precision·Recall 비교는 §1-2 '소수 4자리 반올림 뒤
      비교' 를 적용(그래야 '큰 값' 단계가 뜻을 가짐).
    """
    y = _y(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("검출 0건 — threshold 를 정할 수 없음")
    thr, tp, fp = threshold_candidates(y, score)
    ok = 5 * tp >= 4 * n_pos
    idx = np.flatnonzero(ok)  # 늘 하나 이상(가장 낮은 값 = 전부 양성 → Recall 1)
    p4 = _ratio_round4_units(tp[idx], tp[idx] + fp[idx])
    r4 = _ratio_round4_units(tp[idx], np.full(idx.shape, n_pos))
    # 정렬 키: Precision(4자리) 큰 → Recall(4자리) 큰 → threshold 큰
    best = np.lexsort((thr[idx], r4, p4))[-1]
    i = idx[best]
    return {
        "threshold": float(thr[i]),
        "tp": int(tp[i]),
        "fp": int(fp[i]),
        "recall": float(tp[i] / n_pos),
        "precision": float(tp[i] / (tp[i] + fp[i])),
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    """구현 세부(계획서에 없음): 예측 양성 0 이면 Precision NaN, P·R 중 NaN 이면 F NaN,
    P = R = 0 이면 F = 0."""
    recall = tp / (tp + fn) if tp + fn > 0 else math.nan
    precision = tp / (tp + fp) if tp + fp > 0 else math.nan
    if math.isnan(recall) or math.isnan(precision):
        return recall, precision, math.nan, math.nan
    if precision + recall == 0:
        return recall, precision, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    f2 = 5 * precision * recall / (4 * precision + recall)
    return recall, precision, f1, f2


def recall_by_h(y, score, h, threshold: float) -> np.ndarray:
    """h 별 Recall(같은 threshold), 검출 0건 h = NaN."""
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    h = _h(h, y.shape[0])
    pred = s >= threshold
    pos = np.bincount(h, weights=y, minlength=H_MAX + 1)[1:]
    tp = np.bincount(h, weights=((y == 1) & pred).astype(np.float64), minlength=H_MAX + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pos > 0, tp / pos, np.nan)


def threshold_metrics(y, score, h, threshold: float) -> dict:
    """§1-2 threshold 지표: 쌍 전체 Recall·Precision·F1·F2·혼동행렬 + 구간 가중 Recall."""
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    pred = s >= threshold
    tp = int(((y == 1) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    recall, precision, f1, f2 = _prf(tp, fp, fn)
    rbh = recall_by_h(y, s, h, threshold)
    return {
        "threshold": float(threshold),
        "recall": recall,
        "recall_band_weighted": band_weighted(rbh),
        "precision": precision,
        "f1": f1,
        "f2": f2,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


# ---------------------------------------------------------------- 동점 많음
def tie_summary(score, y, n_cv_pairs: int | None = None) -> dict:
    """원 예측값(float64) 같은 값 묶음 중 가장 큰 묶음의 크기·검출 비율,
    크기 ÷ CV fold validation 쌍 전체 ≥ 1% 면 '동점 많음'(부록 A-12).

    score·y = 세 fold validation 예측을 합친 것. n_cv_pairs 를 안 주면 len(score).
    구현 세부(계획서에 없음): 가장 큰 묶음이 여럿이면 예측값이 큰 묶음의 검출 비율을 적음.
    """
    s = _score(score)
    y = _y(y)
    _check_len(y, s)
    n = int(n_cv_pairs) if n_cv_pairs is not None else s.shape[0]
    if s.size == 0:
        return {"max_tie_size": 0, "max_tie_pos_rate": math.nan, "max_tie_ratio": math.nan, "many_ties": False}
    vals, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    m = int(counts.max())
    g = int(np.flatnonzero(counts == m)[-1])  # 가장 큰 값 쪽
    in_g = inv == g
    ratio = m / n
    return {
        "max_tie_size": m,
        "max_tie_value": float(vals[g]),
        "max_tie_pos_rate": float(y[in_g].mean()),
        "max_tie_ratio": ratio,
        "many_ties": bool(m * 100 >= n),  # ≥ MANY_TIES_RATIO(1%) — 정수 비교
    }


# ---------------------------------------------------------------- 묶음 보고
def score_report(y, score, h, u_tie, qs: Iterable[int] = TOP_Q) -> dict:
    """곡선 지표 전부(threshold 제외). 고르기·보고 모두 같은 함수(부록 B-3)."""
    y = _y(y)
    s = _score(score)
    _check_len(y, s)
    hh = _h(h, y.shape[0])
    u = _u(u_tie, y.shape[0])
    by_h = pr_auc_by_h(y, s, hh, u)
    pooled = pr_auc_pooled(y, s, u)
    sk_by_h = pr_auc_sklearn_default_by_h(y, s, hh)
    b1, b2, b3 = band_means(by_h)
    return {
        "pr_auc_band_weighted": band_weighted(by_h),
        "pr_auc_h01_20": b1, "pr_auc_h21_40": b2, "pr_auc_h41_60": b3,
        "pr_auc_pooled": pooled,
        "pr_auc_by_h": by_h.tolist(),
        "sklearn_default_band_weighted": band_weighted(sk_by_h),
        "sklearn_default_pooled": pr_auc_sklearn_default(y, s),
        "lift_band_weighted": lift_band_weighted(by_h, y, hh),
        "lift_pooled": lift_pooled(pooled, y),
        "top_q_capture": top_q_capture(y, s, u, qs),
        "n_pairs": int(y.shape[0]), "n_pos": int(y.sum()),
    }


def cv_mean(fold_scores: Mapping | Iterable[float]) -> float:
    """CV 평균(fold 1~3 점수 평균). NaN 이 있으면 NaN(조용히 빼지 않음)."""
    vals = list(fold_scores.values()) if isinstance(fold_scores, Mapping) else list(fold_scores)
    if not vals:
        return math.nan
    return float(np.mean(np.asarray(vals, dtype=np.float64)))
