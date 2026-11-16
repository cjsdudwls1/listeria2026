"""h 배정·u_tie(부록 A-1)와 평가 틀(§3) 시험 — 합성 자료만."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from lst import folds as fd
from lst import hassign as hs


def D(s):
    return int(fd.to_days(s)[0])


# ---------------------------------------------------------------- h·u_tie
def test_assign_h_formula():
    for key in ["L000001", "가나다-123", "20150407ABC"]:
        exp = int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "big") % 60 + 1
        assert hs.assign_h(key) == exp
    hs_ = hs.assign_h_many([f"K{i}" for i in range(6000)])
    assert hs_.min() == 1 and hs_.max() == 60 and len(set(hs_.tolist())) == 60


def test_u_tie_formula_and_rescore():
    key, h = "L000001", 17
    exp = int.from_bytes(hashlib.md5(f"{key}|{h}".encode("utf-8")).digest()[:8], "big") / 2 ** 64
    assert hs.u_tie(key, h) == exp
    exp1 = int.from_bytes(hashlib.md5(f"{key}|{h}|1".encode("utf-8")).digest()[:8], "big") / 2 ** 64
    assert hs.u_tie(key, h, rescore=True) == exp1
    arr = hs.u_tie_many([key, key], [h, h + 1], exact=False)
    assert arr[0] == exp and 0 <= arr[1] < 1
    ints = hs.u_tie_many([key], [h])  # 기본 = 정수(exact)
    assert ints.dtype == np.uint64 and int(ints[0]) == hs.u_tie_int(key, h)


def test_u_tie_rejects_bad_h_and_key():
    with pytest.raises(ValueError):
        hs.u_tie("K", 0)
    with pytest.raises(ValueError):
        hs.u_tie("K", 2.5)
    with pytest.raises(TypeError):
        hs.assign_h(None)
    assert hs.u_tie("K", np.int64(5)) == hs.u_tie("K", 5)  # numpy 정수도 '5' 로


# ---------------------------------------------------------------- 분할 정의
def test_split_table_matches_plan():
    exp = {"fold1": ("2022-06-30", "2022-07-01", "2023-05-01", 305),
           "fold2": ("2023-06-30", "2023-07-01", "2024-05-01", 306),
           "fold3": ("2024-06-30", "2024-07-01", "2025-05-01", 305),
           "refit": ("2025-06-30", "2025-07-01", "2026-05-01", 305)}
    for name, (te, ss, se, days) in exp.items():
        sp = fd.SPLITS[name]
        assert (sp.train_end, sp.score_start, sp.score_end) == (D(te), D(ss), D(se))
        assert sp.score_days == days == fd.RECORDED_COUNTS[name]["score_days"]
    assert (fd.SPLITS["refit"].test_start, fd.SPLITS["refit"].test_end) == (D("2025-07-01"), D("2026-06-30"))
    assert fd.get_split(2) is fd.SPLITS["fold2"]


def test_common_d_range():
    lo, hi = fd.common_d_range(fd.SPLITS["fold1"])
    assert (lo, hi) == (D("2022-08-30"), D("2023-05-02"))  # 부록 A-12 '08-30 ~ 05-02'


def _samples(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = D("2021-01-01") + rng.integers(0, 365 * 6, n)
    return pd.DataFrame({"lims_data_key": [f"K{i:06d}" for i in range(n)],
                         "d": fd.as_date(d),
                         "y": (rng.random(n) < 0.01).astype(int),
                         "src_route": np.where(rng.random(n) < 0.1, "overseas", "domestic")})


@pytest.mark.parametrize("name", ["fold1", "fold2", "fold3", "refit"])
def test_scoring_pairs_rules(name):
    sp = fd.SPLITS[name]
    s = _samples()
    d = fd.to_days(s["d"])
    pairs = fd.scoring_pairs(d, sp)
    T = fd.to_days(pairs["T"])
    pd_ = fd.to_days(pairs["d"])
    assert ((T >= sp.score_start) & (T <= sp.score_end)).all()
    assert (pd_ - T == pairs["h"].to_numpy()).all()
    assert (pd_ > sp.train_end).all()
    assert not pairs.duplicated(["row", "h"]).any()
    # 브루트 포스와 같은 쌍
    brute = {(i, h) for i in range(len(d)) for h in range(1, 61) if sp.score_start <= d[i] - h <= sp.score_end}
    assert set(zip(pairs["row"], pairs["h"])) == brute
    # 채점 시료 = 쌍 하나라도 있는 시료
    m = fd.scoring_sample_mask(d, sp)
    assert set(np.flatnonzero(m)) == set(pairs["row"])


def test_rr7_and_test_boundary():
    d = fd.to_days(["2022-07-01", "2025-07-01", "2025-07-02", "2026-06-30"])
    f1, f2, rf = fd.SPLITS["fold1"], fd.SPLITS["fold2"], fd.SPLITS["refit"]
    # 그해 07-01 시료: 그 fold train·채점 밖, 다음 fold train 안
    assert not fd.train_mask(d, f1)[0] and not fd.scoring_sample_mask(d, f1)[0]
    assert fd.train_mask(d, f2)[0]
    # d = 2025-07-01: refit train·test 채점 둘 다 밖
    assert not fd.train_mask(d, rf)[1] and not fd.scoring_sample_mask(d, rf)[1]
    assert fd.scoring_sample_mask(d, rf)[2] and fd.scoring_sample_mask(d, rf)[3]


def test_make_train_rows():
    s = _samples()
    sp = fd.SPLITS["fold2"]
    tr = fd.make_train_rows(s, sp)
    assert (fd.to_days(tr["d"]) <= sp.train_end).all()
    assert len(tr) == int((fd.to_days(s["d"]) <= sp.train_end).sum())
    assert tr["h"].tolist() == [hs.assign_h(k) for k in tr["lims_data_key"]]
    assert (fd.to_days(tr["d"]) - fd.to_days(tr["T"]) == tr["h"].to_numpy()).all()
    bad = s.copy()
    bad.loc[1, "lims_data_key"] = bad.loc[0, "lims_data_key"]
    with pytest.raises(ValueError):
        fd.make_train_rows(bad, sp)
    wrong_h = s.assign(h=1)
    with pytest.raises(ValueError):
        fd.make_train_rows(wrong_h, sp)


def test_count_and_check_split_counts():
    s = _samples()
    sp = fd.SPLITS["refit"]
    c = fd.count_split(fd.to_days(s["d"]), s["y"], s["src_route"], sp)
    assert c["score_n"] == c["score_n_domestic"] + c["score_n_overseas"]
    with pytest.raises(AssertionError):
        fd.check_split_counts(c, sp)  # 합성 자료라 기록값과 다름 → 멈춤
    fd.check_split_counts(dict(fd.RECORDED_COUNTS["fold1"]), fd.SPLITS["fold1"])


def test_incomplete_month_table_and_mask():
    rows = []
    for route, base in (("domestic", 100), ("overseas", 10)):
        for y in (2023, 2024, 2025):
            for m in ("04", "05", "06"):
                rows.append((route, f"{y}-{m}", base))
    rows += [("domestic", "2026-04", 100), ("domestic", "2026-05", 70), ("domestic", "2026-06", 69),
             ("overseas", "2026-04", 6)]  # 해외 05·06 은 표에 없음 → 0건
    t = fd.incomplete_month_table(pd.DataFrame(rows, columns=["src_route", "ym", "n"]))
    flag = {(r, ym): v for r, ym, v in t[["src_route", "ym", "incomplete"]].itertuples(index=False)}
    assert flag[("domestic", "2026-04")] is False
    assert flag[("domestic", "2026-05")] is False   # 70% 는 '미만' 아님
    assert flag[("domestic", "2026-06")] is True
    assert flag[("overseas", "2026-04")] is True and flag[("overseas", "2026-06")] is True
    inc = {k for k, v in flag.items() if v}
    pdays = fd.to_days(["2026-06-03", "2026-04-10", "2026-05-31", "2026-03-31"])
    route = np.array(["domestic", "domestic", "overseas", "overseas"])
    assert fd.incomplete_pair_mask(pdays, route, inc).tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        fd.incomplete_pair_mask(pdays, route, {("domestic", "2026-03")})


def test_to_days_rejects_ints_and_nat():
    with pytest.raises(TypeError):
        fd.to_days(np.array([1, 2]))
    with pytest.raises(ValueError):
        fd.to_days(pd.Series([pd.NaT]))


def test_date_inputs_datetime64_same_as_day_ints():
    # datetime64[ns]·datetime64[D]·문자열·Series 모두 정수 일수와 같은 결과(조용히 0쌍 금지)
    s = _samples(500, seed=4)
    d_int = fd.to_days(s["d"])
    sp = fd.SPLITS["fold1"]
    ref = fd.scoring_pairs(d_int, sp)
    assert len(ref) > 0
    for d_in in (s["d"], s["d"].to_numpy(), s["d"].to_numpy().astype("datetime64[D]"),
                 s["d"].dt.strftime("%Y-%m-%d").to_numpy()):
        got = fd.scoring_pairs(d_in, sp)
        assert got[["row", "h"]].equals(ref[["row", "h"]])
        assert np.array_equal(fd.train_mask(d_in, sp), fd.train_mask(d_int, sp))
        assert np.array_equal(fd.scoring_sample_mask(d_in, sp), fd.scoring_sample_mask(d_int, sp))
        assert np.array_equal(fd.common_d_mask(d_in, sp), fd.common_d_mask(d_int, sp))
        assert fd.count_split(d_in, s["y"], s["src_route"], sp) == fd.count_split(d_int, s["y"], s["src_route"], sp)
    with pytest.raises(TypeError):
        fd.day_ints(np.array([True, False]))
