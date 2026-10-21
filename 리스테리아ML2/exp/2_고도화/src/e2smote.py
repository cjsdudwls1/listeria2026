# -*- coding: utf-8 -*-
"""D2. SMOTENC 변형 직접 구현: '시도 안 이웃 + 기준 행 범주' (사용자 결정 2026-10-19, 추천안).
(imbalanced-learn 이 설치돼 있지 않고 설치하지 말라는 지시. 또 imbalanced-learn 의 SMOTENC 는 결측(NaN)이 있으면 오류)

절차 (imbalanced-learn 0.12 SMOTENC 를 바탕으로, 이웃 후보와 범주 규칙만 바꿈)
  1) 만들 개수 = int(불검출 수 x 0.1 - 검출 수)  (imblearn 의 float sampling_strategy 규칙)
  2) 연속 변수(h + 기상 요약 190)는 원래 단위 그대로(척도 맞춤 없음, imblearn 과 같음)
  3) 범주 4개는 원-핫으로 바꾸고 1 대신 (전체 학습 검출 행 연속 변수 표준편차들의 중앙값)/2 를 넣음
  4) 이웃 k=5 (자기 자신 제외): 기준 검출 행과 같은 시도(sido = sgg_code 앞 2자리)의 검출 행 중에서만 찾음.
     거리는 3) 의 같은 행렬로 그 시도 검출 행끼리만 계산.
     그 시도의 학습 검출 행이 k+1(=6)개 미만(자기 + 이웃 5 가 안 됨)이면 전국 검출 행에서 찾음(전국 대체).
  5) 새 표본: 검출 행 하나(기준 행)와 그 이웃 하나를 무작위로 골라 연속 변수 = x + u x (이웃 - x), u ~ U(0,1)
     범주 4칸(sido, sgg, p2, p3) = 기준 검출 행의 값 그대로 -> 합성 행은 실제 검출 단위와 같은 단위
표준 SMOTENC(imbalanced-learn) 와 다른 점은 DIFF_FROM_STANDARD 에 적음(표 비고·설정 시트에도 그대로 씀).
"""
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances

METHOD = "시도 안 이웃 + 기준 행 범주"
DIFF_FROM_STANDARD = [
    "이웃 후보: 표준 SMOTENC 는 전체 검출 행에서 k=5 이웃 -> 여기서는 기준 검출 행과 같은 시도(sgg_code 앞 2자리) 검출 행 중에서만"
    "(그 시도 학습 검출 행이 6개 미만이면 전국 검출 행에서)",
    "범주: 표준 SMOTENC 는 칸마다 이웃 5개의 최빈값(같으면 무작위) -> 여기서는 기준 검출 행의 범주 4칸 값 그대로"
    "(합성 행 단위 = 실제 검출 단위, 칸끼리 어긋난 조합 없음)",
    "결측(이전부터 같음): 거리 = sklearn nan_euclidean_distances(결측 칸 빼고 계산 후 칸 수 비율로 키움, 공통 칸 없으면 무한대), "
    "표준편차·중앙값은 결측 무시, 보간 식 그대로라 기준 행·이웃 중 한쪽이 결측이면 새 값도 결측 (표준은 결측이 있으면 오류)",
    "그 밖(이전부터 같음): 자기 자신은 거리 무한대로 두고 뺌, 난수 = numpy default_rng(시드)(표준은 RandomState)",
]


def smotenc(Xc: np.ndarray, codes: np.ndarray, y: np.ndarray, ratio: float, k: int, seed: int, group):
    """Xc (n, p) 연속 변수(NaN 허용), codes (n, q) 범주 번호, y (n,) 0/1, group (n,) 이웃 찾기 묶음(시도 문자열).
    돌려줌: (새 연속 변수 (m, p) float32, 새 범주 번호 (m, q), 정보 dict, 계획(검사용) dict)"""
    y = np.asarray(y)
    mi = np.flatnonzero(y == 1)
    n_min, n_maj = len(mi), int((y == 0).sum())
    n_new = int(n_maj * ratio - n_min)
    if n_new <= 0:
        raise ValueError(f"SMOTENC: 만들 개수 {n_new} <= 0 (불검출 {n_maj}, 검출 {n_min}, 비율 {ratio})")
    if n_min <= k:
        raise ValueError(f"SMOTENC: 검출 행 {n_min} <= k {k}")
    group = np.asarray(group)
    if group.shape != (len(y),):
        raise ValueError(f"SMOTENC: group 모양 {group.shape} != ({len(y)},)")
    Xm = Xc[mi].astype(np.float64)
    Cm = codes[mi]
    gm = group[mi].astype(str)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(Xm, axis=0)
    med = float(np.nanmedian(sd))
    blocks = [Xm]
    for j in range(Cm.shape[1]):
        u, inv = np.unique(Cm[:, j], return_inverse=True)
        oh = np.zeros((n_min, len(u)))
        oh[np.arange(n_min), inv] = med / 2.0
        blocks.append(oh)
    Z = np.hstack(blocks)

    # ---- 이웃: 시도 안(그 시도 검출 행끼리만 거리), 작은 시도는 전국
    glev, gcnt = np.unique(gm, return_counts=True)
    by_group = {str(g): int(c) for g, c in zip(glev, gcnt)}
    small = [str(g) for g, c in zip(glev, gcnt) if c < k + 1]
    fallback = np.isin(gm, small)                          # (n_min,) 전국에서 이웃을 찾는 기준 행
    nn = np.full((n_min, k), -1, dtype=np.int64)
    nn_dist = np.full((n_min, k), np.nan)
    n_nan_dist = 0
    for g in glev:
        if str(g) in small:
            continue
        idx = np.flatnonzero(gm == g)
        D = nan_euclidean_distances(Z[idx], Z[idx])
        n_nan_dist += int(np.isnan(D).sum())
        D[np.isnan(D)] = np.inf
        np.fill_diagonal(D, np.inf)
        loc = np.argsort(D, axis=1, kind="stable")[:, :k]
        nn[idx] = idx[loc]
        nn_dist[idx] = np.take_along_axis(D, loc, axis=1)
    fidx = np.flatnonzero(fallback)
    if len(fidx):
        D = nan_euclidean_distances(Z[fidx], Z)
        n_nan_dist += int(np.isnan(D).sum())
        D[np.isnan(D)] = np.inf
        D[np.arange(len(fidx)), fidx] = np.inf             # 자기 자신
        loc = np.argsort(D, axis=1, kind="stable")[:, :k]
        nn[fidx] = loc
        nn_dist[fidx] = np.take_along_axis(D, loc, axis=1)
    if (nn < 0).any():
        raise RuntimeError("SMOTENC: 이웃을 정하지 못한 검출 행이 있음")
    self_nn = int((nn == np.arange(n_min)[:, None]).sum())
    if self_nn:
        raise RuntimeError(f"SMOTENC: 자기 자신이 이웃으로 뽑힌 경우 {self_nn}개(공통 칸 없는 거리 무한대가 많음)")

    # ---- 새 표본 (난수 순서는 이전과 같음: 기준 행·이웃 칸, 비율)
    rng = np.random.default_rng(seed)
    sidx = rng.integers(0, n_min * k, size=n_new)
    rows, cols = sidx // k, sidx % k
    steps = rng.uniform(size=n_new)[:, None]
    base = Xm[rows]
    nbr = nn[rows, cols]
    Xnew = base + steps * (Xm[nbr] - base)
    Cnew = Cm[rows].astype(codes.dtype, copy=True)          # 범주 = 기준 검출 행 그대로

    in_grp = ~fallback[rows]
    info = {"method": METHOD,
            "neighbor_rule": f"기준 검출 행과 같은 시도(sido = sgg_code 앞 2자리) 학습 검출 행 중 거리 가까운 {k}개(자기 제외). "
                             f"그 시도 학습 검출 행이 {k + 1}개 미만이면 전국 학습 검출 행에서",
            "category_rule": "범주 4칸(sido, sgg, p2, p3) = 기준 검출 행의 값 그대로(이웃 최빈값 쓰지 않음)",
            "numeric_rule": "연속 변수(h + 기상 요약) = 기준 행 + U(0,1) x (이웃 - 기준 행)",
            "n_minority": n_min, "n_majority": n_maj, "n_new": n_new, "ratio_after": (n_min + n_new) / n_maj,
            "k": k, "group_min_rows": k + 1, "median_std": med, "n_onehot_cols": int(Z.shape[1] - Xm.shape[1]),
            "minority_by_group": by_group,
            "fallback_groups": {g: by_group[g] for g in small},
            "n_minority_fallback": int(fallback.sum()),
            "n_new_from_fallback": int((~in_grp).sum()),
            "neighbor_same_group_rate": float(np.mean(gm[nbr[in_grp]] == gm[rows[in_grp]])) if in_grp.any() else None,
            "n_inf_neighbor_dist": int(np.isinf(nn_dist).sum()),
            "nan_distance_pairs": n_nan_dist,
            "nan_rate_minority": float(np.isnan(Xm).mean()), "nan_rate_synthetic": float(np.isnan(Xnew).mean()),
            "distance": "nan_euclidean (sklearn), 같은 시도 검출 행끼리만 계산(전국 대체 시도의 기준 행은 전국 검출 행과)",
            "onehot_weight": "원-핫 1 대신 (전체 학습 검출 행 연속 변수 표준편차의 중앙값)/2 - 시도마다 다시 구하지 않음",
            "rng": f"numpy default_rng({seed}): 기준 행·이웃 칸 integers, 비율 uniform (이전 방식과 같은 순서)",
            "diff_from_standard_smotenc": DIFF_FROM_STANDARD}
    plan = {"minority_index": mi, "nn": nn, "rows": rows, "cols": cols, "steps": steps[:, 0], "median_std": med,
            "group": gm, "fallback": fallback}
    return Xnew.astype(np.float32), Cnew, info, plan
