"""지표 시험 — 손으로 만든 작은 예제(trapz 값·동점 규칙·NaN 재가중·threshold 동점),
sklearn auc(recall, precision) 와 직접 비교, AP 함수 미사용 소스 검사."""
import math
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve

from lst import metrics as mt
from lst.hassign import u_tie_many

FINAL = Path(__file__).resolve().parents[1]
AP_NAME = "average_" + "precision"  # 이 파일에도 그 이름이 붙어 나오지 않게


def naive_trapz(y, order):
    """정렬 순서대로 한 쌍씩 내려가며 (R, P) 점 → (R=0, P=1) 에서 시작해 사다리꼴 합."""
    y = np.asarray(y)[order]
    n_pos = y.sum()
    tp = fp = 0
    r_prev, p_prev, area = 0.0, 1.0, 0.0
    for yi in y:
        tp += yi
        fp += 1 - yi
        r, p = tp / n_pos, tp / (tp + fp)
        area += (r - r_prev) * (p + p_prev) / 2
        r_prev, p_prev = r, p
    return area


# ---------------------------------------------------------------- trapz 손 계산
def test_trapz_hand_example():
    # 순서 1,0,1,0: 점 (0.5,1)(0.5,0.5)(1,2/3)(1,0.5) → 0.5 + 0.5·(0.5+2/3)/2
    y = [1, 0, 1, 0]
    s = [0.9, 0.8, 0.7, 0.1]
    exp = 0.5 + 0.5 * (0.5 + 2 / 3) / 2
    assert mt.pr_auc(y, s, [0.5] * 4) == pytest.approx(exp, abs=1e-15)
    assert mt.pr_auc_sklearn_default(y, s) == pytest.approx(exp, abs=1e-15)


def test_ties_broken_by_u_tie():
    y = [1, 0, 1, 0]
    s = [0.5, 0.5, 0.5, 0.5]
    # u 작은 순 → 원래 순서 1,0,1,0 → 위 손 계산 값
    assert mt.pr_auc(y, s, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.5 + 0.5 * (0.5 + 2 / 3) / 2)
    # u 를 뒤집으면 순서 0,1,0,1: 점 (0,0)(0.5,0.5)(0.5,1/3)(1,0.5)
    exp_rev = 0 + 0.5 * (0 + 0.5) / 2 + 0 + 0.5 * (1 / 3 + 0.5) / 2
    assert mt.pr_auc(y, s, [0.4, 0.3, 0.2, 0.1]) == pytest.approx(exp_rev)
    # sklearn 기본(같은 값 묶음) = 점 하나 (1, 0.5) → (0,1)~(1,0.5) 사다리꼴 0.75
    assert mt.pr_auc_sklearn_default(y, s) == pytest.approx(0.75)


def test_tie_broken_scores_rank_rule():
    s = np.array([0.2, 0.9, 0.2, 0.5])
    u = np.array([0.7, 0.1, 0.3, 0.0])
    # 순서: 0.9 → 0.5 → 0.2(u .3) → 0.2(u .7)
    assert mt.tie_broken_order(s, u).tolist() == [1, 3, 2, 0]
    assert mt.tie_broken_scores(s, u).tolist() == [0.0, 3.0, 1.0, 2.0]  # 쌍 수 − 순위


def test_matches_sklearn_auc_and_naive_random():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(20, 300))
        y = (rng.random(n) < 0.2).astype(int)
        if y.sum() in (0, n):
            continue
        s = rng.random(n)  # 동점 없음
        u = rng.random(n)
        p, r, _ = precision_recall_curve(y, s, drop_intermediate=False)
        ref = auc(r, p)
        assert mt.pr_auc(y, s, u) == pytest.approx(ref, abs=1e-12)
        assert mt.pr_auc_sklearn_default(y, s) == pytest.approx(ref, abs=1e-12)
        assert mt.pr_auc(y, s, u) == pytest.approx(naive_trapz(y, np.lexsort((u, -s))), abs=1e-12)


def test_heavy_ties_equal_naive_with_u_tie_from_keys():
    rng = np.random.default_rng(2)
    n = 500
    keys = [f"K{i:05d}" for i in range(n)]
    h = rng.integers(1, 61, n)
    y = (rng.random(n) < 0.1).astype(int)
    s = np.round(rng.random(n), 1)  # 동점 많음
    for exact in (False, True):
        u = u_tie_many(keys, h, exact=exact)
        assert mt.pr_auc(y, s, u) == pytest.approx(naive_trapz(y, np.lexsort((u, -s))), abs=1e-12)
    # float u 와 정수 u 의 순서가 같음
    uf, ui = u_tie_many(keys, h, exact=False), u_tie_many(keys, h)
    assert np.array_equal(np.lexsort((uf, -s)), np.lexsort((ui, -s)))


def test_degenerate_is_nan():
    assert math.isnan(mt.pr_auc([0, 0, 0], [0.1, 0.2, 0.3], [0, 0, 0]))
    assert math.isnan(mt.pr_auc([1, 1], [0.1, 0.2], [0, 0]))
    assert math.isnan(mt.pr_auc([], [], []))


def test_rejects_nan_prediction_and_bad_y():
    with pytest.raises(ValueError):
        mt.pr_auc([0, 1], [np.nan, 0.2], [0, 0])
    with pytest.raises(ValueError):
        mt.pr_auc([0, 2], [0.1, 0.2], [0, 0])


# ---------------------------------------------------------------- h 별·구간 가중
def test_band_weighted_rules():
    v = np.full(60, np.nan)
    v[0:20] = 0.4
    v[20:40] = 0.2
    v[40:60] = 0.1
    assert mt.band_weighted(v) == pytest.approx(0.5 * 0.4 + 0.3 * 0.2 + 0.2 * 0.1)
    # NaN 인 h 는 구간 평균에서 뺌
    v2 = v.copy()
    v2[0] = np.nan
    v2[1] = 1.0  # 구간 1 = (1.0 + 0.4·18)/19
    b1 = (1.0 + 0.4 * 18) / 19
    assert mt.band_means(v2)[0] == pytest.approx(b1)
    # 구간 1 통째 NaN → 0.3·0.2 를 합 1 로 다시 나눔
    v3 = v.copy()
    v3[0:20] = np.nan
    assert mt.band_weighted(v3) == pytest.approx((0.3 * 0.2 + 0.2 * 0.1) / 0.5)
    # 구간 1·3 NaN → 구간 2 값 그대로
    v4 = np.full(60, np.nan)
    v4[25] = 0.7
    assert mt.band_weighted(v4) == pytest.approx(0.7)
    assert math.isnan(mt.band_weighted(np.full(60, np.nan)))


def test_by_h_nan_for_no_or_all_positive():
    rng = np.random.default_rng(5)
    n = 2000
    h = rng.integers(1, 61, n)
    y = (rng.random(n) < 0.3).astype(int)
    y[h == 3] = 0      # 검출 0건
    y[h == 50] = 1     # 전부 검출
    s = rng.random(n)
    u = rng.random(n)
    by = mt.pr_auc_by_h(y, s, h, u)
    assert by.shape == (60,)
    assert math.isnan(by[2]) and math.isnan(by[49])
    for k in (0, 10, 59):
        m = h == k + 1
        assert by[k] == pytest.approx(mt.pr_auc(y[m], s[m], u[m]))
    bw = mt.pr_auc_band_weighted(y, s, h, u)
    assert bw == pytest.approx(mt.band_weighted(by))
    rep = mt.score_report(y, s, h, u)
    assert rep["pr_auc_band_weighted"] == pytest.approx(bw)
    assert rep["pr_auc_pooled"] == pytest.approx(mt.pr_auc(y, s, u))


def test_lift():
    h = np.repeat(np.arange(1, 61), 10)
    y = np.zeros(600, int)
    y[::10] = 1          # h 마다 검출 1/10
    y[h == 2] = 0        # h 2 검출 0 → PR-AUC NaN, 검출 비율도 뺌
    s = np.linspace(1, 0, 600)
    u = np.zeros(600)
    by = mt.pr_auc_by_h(y, s, h, u)
    lift = mt.lift_band_weighted(by, y, h)
    assert lift == pytest.approx(mt.band_weighted(by) / 0.1)
    pooled = mt.pr_auc_pooled(y, s, u)
    assert mt.lift_pooled(pooled, y) == pytest.approx(pooled / y.mean())


# ---------------------------------------------------------------- threshold
def test_threshold_basic_and_groups():
    y = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    s = [.9, .8, .8, .7, .6, .5, .4, .3, .2, .1]
    r = mt.select_threshold(y, s)
    # 검출 5: Recall ≥ 0.8 → TP ≥ 4 → 후보 t=.5(TP4,FP2,P .6667)·.2(TP5,FP4,.5556)·… → .5
    assert r["threshold"] == 0.5 and r["tp"] == 4 and r["fp"] == 2
    # 같은 값 묶음 함께: t = .8 이면 두 쌍(.8) 모두 양성
    thr, tp, fp = mt.threshold_candidates(y, s)
    assert thr.tolist()[:3] == [.9, .8, .7] and tp.tolist()[:3] == [1, 2, 3] and fp.tolist()[:3] == [0, 1, 1]


def test_threshold_tie_precision_then_recall():
    # Precision 같고(0.8) Recall 다름 → Recall 큰 쪽
    # 정렬: 0×4, 1×16 (TP16 FP4: P .8 R .8) → 0, 1×4 (TP20 FP5: P .8 R 1), 그 사이 점은 P < .8
    y = [0] * 4 + [1] * 16 + [0] + [1] * 4
    s = np.linspace(1.0, 0.04, 25)
    r = mt.select_threshold(y, s)
    assert r["tp"] == 20 and r["fp"] == 5 and r["threshold"] == s[24]


def test_threshold_precision_compared_after_rounding():
    # 검출 20000. t=.9: TP16000 FP4000 (P .8 → .8000, R .8 → .8000)
    #             t=.5: TP16001 FP4001 (P .79997 → .8000, R .80005 → .8001)
    # 반올림 없이면 t=.9(P 큼), 4자리 반올림 뒤엔 Precision 같고 Recall 큼 → t=.5
    y = np.r_[np.ones(16000), np.zeros(4000), [1, 0], np.zeros(100000), np.ones(3999)].astype(int)
    s = np.r_[np.full(20000, 0.9), [0.5, 0.5], np.full(100000, 0.3), np.full(3999, 0.2)]
    assert y.sum() == 20000
    r = mt.select_threshold(y, s)
    assert r["threshold"] == 0.5 and r["tp"] == 16001 and r["fp"] == 4001


def test_threshold_same_rounded_pr_picks_larger_value():
    # t=.9: TP16000 FP4000 (P .8000 R .8000), t=.5: TP16000 FP4001 (P .79996 → .8000, R .8000) → 큰 값
    y = np.r_[np.ones(16000), np.zeros(4000), [0], np.zeros(100000), np.ones(4000)].astype(int)
    s = np.r_[np.full(20000, 0.9), [0.5], np.full(100000, 0.3), np.full(4000, 0.2)]
    r = mt.select_threshold(y, s)
    assert r["threshold"] == 0.9 and r["fp"] == 4000


def test_threshold_recall_exact_08_boundary():
    # 검출 5, TP 4 = 정확히 0.8 → 허용
    y = [1, 1, 1, 1, 0, 1]
    s = [.9, .8, .7, .6, .5, .4]
    assert mt.select_threshold(y, s)["threshold"] == 0.6


def test_threshold_metrics_values():
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    s = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
    h = np.array([1, 1, 1, 1, 30, 30, 50, 50])
    m = mt.threshold_metrics(y, s, h, 0.6)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 3)
    P, R = 3 / 4, 3 / 4
    assert m["precision"] == pytest.approx(P) and m["recall"] == pytest.approx(R)
    assert m["f1"] == pytest.approx(2 * P * R / (P + R))
    assert m["f2"] == pytest.approx(5 * P * R / (4 * P + R))
    # h 별 Recall: h1 = 3/3, h30 = NaN(검출 0), h50 = 0/1 → 구간 1 = 1, 구간 2 NaN, 구간 3 = 0
    assert m["recall_band_weighted"] == pytest.approx((0.5 * 1 + 0.2 * 0) / 0.7)


def test_top_q_capture():
    n = 200
    y = np.zeros(n, int)
    y[[0, 1, 50, 199]] = 1
    s = np.linspace(1, 0, n)
    out = mt.top_q_capture(y, s, np.zeros(n))
    # 상위 1% = ⌈2⌉ = 2 쌍 → 2/4, 5% = 10 → 2/4, 10% = 20 → 2/4, 20% = 40 → 2/4
    assert out == {1: 0.5, 5: 0.5, 10: 0.5, 20: 0.5}
    # 동점: 전부 같은 값이면 u_tie 순서로 위에서 자름
    u = np.arange(n)[::-1].astype(float)  # 마지막 쌍이 맨 위
    out2 = mt.top_q_capture(y, np.zeros(n), u, qs=(1,))
    assert out2[1] == 0.25  # 위 2 쌍 = 199(검출), 198


def test_tie_summary_flag():
    s = np.r_[np.full(10, 0.3), np.linspace(0, 0.2, 990)]
    y = np.r_[np.ones(2), np.zeros(998)].astype(int)
    t = mt.tie_summary(s, y)
    assert t["max_tie_size"] == 10 and t["many_ties"] is True  # 10/1000 = 1%
    assert t["max_tie_pos_rate"] == pytest.approx(0.2)
    t2 = mt.tie_summary(s, y, n_cv_pairs=1001)
    assert t2["many_ties"] is False


def test_round4_and_improvement():
    assert mt.round4(0.12345) == 0.1235   # 사사오입
    assert mt.round4(0.12344999) == 0.1234
    assert mt.is_improvement(0.12346, 0.12344) is True   # .1235 > .1234
    assert mt.is_improvement(0.123449, 0.12341) is False  # 둘 다 .1234 → 상승 아님
    assert mt.is_improvement(float("nan"), 0.1) is False


# ---------------------------------------------------------------- AP 미사용 소스 검사
def test_no_ap_function_anywhere():
    bad = []
    for p in FINAL.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        if AP_NAME in text or re.search(r"from\s+sklearn\.metrics\s+import\s+\*", text):
            bad.append(str(p.relative_to(FINAL)))
    assert not bad, f"AP 함수 이름·와일드카드 import 발견: {bad}"
    assert not any(AP_NAME in name for name in dir(mt))
