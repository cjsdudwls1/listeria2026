"""과거 이력(HIS1~HIS4)·7-a·단계 4~6 범주 숫자 인코딩 시험 — 합성 자료만(계획서 §5-1·부록 A-3·A-6·B-6).

가장 중요한 것 = 누수: 자기 시료·d > C 시료가 집계에 안 들어가는지, validation·test 행은 그 fold train 끝까지만 보는지.
"""
import math

import numpy as np
import pandas as pd
import pytest

from lst import folds as fd
from lst import history as hi
from lst.hassign import assign_h_many
from lst.samples import NO_CODE


def D(s):
    return int(fd.to_days(s)[0])


def synth(n=6000, seed=0, d_lo="2015-04-07", d_hi="2026-06-30", pos=0.04):
    rng = np.random.default_rng(seed)
    d = rng.integers(D(d_lo), D(d_hi) + 1, n)
    p2s = ["A0100000000000", "B0200000000000", "C0300000000000"]
    p2 = rng.choice(p2s, n)
    sub = rng.integers(1, 4, n)
    p3 = np.array([f"{a[:3]}{b:02d}00000000{'0'}" for a, b in zip(p2, sub)], dtype=object)
    p3 = np.array([s[:14] for s in p3], dtype=object)
    # RR3: 일부는 3단계 '코드없음'(p2 는 그대로 — (p2, '코드없음') 로 나눠 셈), 일부는 둘 다 '코드없음'
    nc = rng.random(n) < 0.02
    p3[nc] = NO_CODE
    both = rng.random(n) < 0.005
    p2 = p2.astype(object)
    p2[both] = NO_CODE
    p3[both] = NO_CODE
    sgg_dom = ["11110", "11140", "26110", "41110"]
    ov = rng.random(n) < 0.1
    sgg = np.where(ov, rng.choice(["OVCHN", "OVUSA"], n), rng.choice(sgg_dom, n)).astype(object)
    sido = np.array([s[:2] if not s.startswith("OV") else "OV" for s in sgg], dtype=object)
    y = (rng.random(n) < pos).astype(np.int8)
    keys = [f"K{seed}_{i:06d}" for i in range(n)]
    df = pd.DataFrame({"lims_data_key": keys, "y": y, "src_route": np.where(ov, "overseas", "domestic"),
                       "sgg": sgg, "sido": sido, "p2": p2, "p3": p3,
                       "d": pd.to_datetime(fd.as_date(d)).date, "h": assign_h_many(keys).astype(np.int16),
                       "rr1_moved": False})
    return df


@pytest.fixture(scope="module")
def S():
    return synth()


def train_rows(S, split):
    tr = fd.make_train_rows(S, split)
    tr["C"] = fd.as_date(hi.c_for_train_rows(tr["T"], split))
    return tr


def eval_rows(S, split):
    """채점 시료(쌍 하나 이상) + C = train 끝(c_for_eval_rows 가 쌍마다 확인)."""
    pairs = fd.scoring_pairs(fd.to_days(S["d"]), split)
    hi.c_for_eval_rows(pairs["T"], split)
    rows = np.unique(pairs["row"].to_numpy())
    ev = S.iloc[rows].copy()
    ev["C"] = fd.as_date(np.full(len(ev), split.train_end))
    return ev


def pop_of(S, split=None):
    if split is not None:
        S = S[fd.to_days(S["d"]) <= split.train_end]
    return hi.Population.from_samples(S)


def assert_frame_close(a, b):
    assert list(a.columns) == list(b.columns)
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-12, atol=1e-14, equal_nan=True)


# ================================================================ C 규칙
@pytest.mark.parametrize("T,C", [("2023-07-01", "2023-06-30"), ("2023-06-30", "2022-06-30"),
                                 ("2024-01-15", "2023-06-30"), ("2023-12-31", "2023-06-30"),
                                 ("2024-06-29", "2023-06-30"), ("2016-02-29", "2015-06-30"),
                                 ("2015-05-01", "2014-06-30")])
def test_c_from_T(T, C):
    assert hi.c_from_T([T])[0] == D(C)


@pytest.mark.parametrize("name", list(fd.SPLITS))
def test_eval_C_is_train_end_for_every_scoring_T(name):
    sp = fd.SPLITS[name]
    T = np.arange(sp.score_start, sp.score_end + 1)
    assert (hi.c_for_eval_rows(T, sp) == sp.train_end).all()
    assert (hi.c_from_T(T) == sp.train_end).all()  # §5-1 두 표현이 같음
    with pytest.raises(ValueError):
        hi.c_for_eval_rows([sp.score_end + 1], sp)


def test_train_C_before_T_and_le_train_end(S):
    for sp in fd.SPLITS.values():
        tr = train_rows(S, sp)
        C = fd.to_days(tr["C"])
        T = fd.to_days(tr["T"])
        d = fd.to_days(tr["d"])
        assert (C < T).all() and (T < d).all() and (C <= sp.train_end).all()
        assert set(fd.as_date(np.unique(C)).astype(str)) <= {f"{y}-06-30" for y in range(2014, 2026)}


# ================================================================ 정의 대조
@pytest.mark.parametrize("name", ["fold1", "refit"])
def test_features_match_reference(S, name):
    sp = fd.SPLITS[name]
    pop_df = S[fd.to_days(S["d"]) <= sp.train_end]
    pop = hi.Population.from_samples(pop_df)
    ref = hi.Reference(pop_df)
    rng = np.random.default_rng(1)
    for rows in (train_rows(S, sp), eval_rows(S, sp)):
        F = hi.history_features(pop, rows)
        pick = rng.choice(len(rows), 60, replace=False)
        for i in pick:
            r = rows.iloc[i]
            exp = ref.row(r)
            got = F.iloc[i]
            for c, v in exp.items():
                if math.isnan(v):
                    assert math.isnan(got[c]), c
                else:
                    assert got[c] == pytest.approx(v, rel=1e-12, abs=1e-15), c


def test_column_list_and_count():
    cols = hi.his_columns()
    names = [c["column"] for c in cols]
    assert len(names) == len(set(names))
    # 단위 19개(부록 A-5 이력 19) = HIS1 6 + HIS2 6 + HIS3 4 + HIS4 3, k 5값이면 칼럼 6·5 + 6 + 4 + 3·5 = 55
    assert len(names) == 55
    assert [c for c in names if c.startswith("woe_")] == ["woe_sido", "woe_sgg", "woe_p2", "woe_p3"]
    assert all(c["window_end"] == "C" for c in cols)


# ================================================================ 누수(부록 B-6)
def test_future_samples_never_used_train_rows(S):
    """C 마다: d > C 시료의 y 를 뒤집거나 아예 지워도 그 C 행의 값이 그대로."""
    sp = fd.SPLITS["refit"]
    tr = train_rows(S, sp)
    base = hi.history_features(pop_of(S), tr)
    Cs = fd.to_days(tr["C"])
    d_all = fd.to_days(S["d"])
    for c in np.unique(Cs):
        sel = Cs == c
        flipped = S.copy()
        fut = d_all > c
        flipped.loc[fut, "y"] = (1 - flipped.loc[fut, "y"]).astype(np.int8)
        F1 = hi.history_features(hi.Population.from_samples(flipped), tr[sel])
        assert_frame_close(F1, base[sel])
        F2 = hi.history_features(hi.Population.from_samples(S[~fut]), tr[sel])
        assert_frame_close(F2, base[sel])


def test_past_samples_are_used(S):
    """반대 확인(시험이 헛돌지 않게): C 이전 창 안 시료를 바꾸면 값이 바뀜."""
    sp = fd.SPLITS["fold2"]
    ev = eval_rows(S, sp)
    base = hi.history_features(pop_of(S, sp), ev)
    d_all = fd.to_days(S["d"])
    inwin = (d_all <= sp.train_end) & (d_all > sp.train_end - 300)
    mod = S.copy()
    mod.loc[inwin, "y"] = 1
    F = hi.history_features(pop_of(mod, sp), ev)
    assert not np.allclose(F.to_numpy(float), base.to_numpy(float), equal_nan=True)


def test_self_sample_not_in_aggregate(S):
    """자기 시료를 모집단에서 빼도 자기 행 값이 같음(자기 시료는 d > C 라 원래 안 들어감)."""
    sp = fd.SPLITS["fold3"]
    tr = train_rows(S, sp)
    pop = pop_of(S, sp)
    rng = np.random.default_rng(2)
    for i in rng.choice(len(tr), 25, replace=False):
        r = tr.iloc[[i]]
        base = hi.history_features(pop, r)
        minus = S[S["lims_data_key"] != r["lims_data_key"].iloc[0]]
        assert_frame_close(hi.history_features(pop_of(minus, sp), r), base)
        # 자기 y 를 뒤집어도 같음
        flip = S.copy()
        j = flip.index[flip["lims_data_key"] == r["lims_data_key"].iloc[0]]
        flip.loc[j, "y"] = 1 - flip.loc[j, "y"]
        assert_frame_close(hi.history_features(pop_of(flip, sp), r), base)


@pytest.mark.parametrize("name", list(fd.SPLITS))
def test_eval_rows_see_only_train_end(S, name):
    """validation·test 행: train 끝 뒤 시료(그 fold 채점 기간 시료 포함)를 뒤집어도 같음, 모든 행 같은 C."""
    sp = fd.SPLITS[name]
    ev = eval_rows(S, sp)
    base = hi.history_features(hi.Population.from_samples(S), ev)
    flip = S.copy()
    after = fd.to_days(flip["d"]) > sp.train_end
    flip.loc[after, "y"] = (1 - flip.loc[after, "y"]).astype(np.int8)
    assert_frame_close(hi.history_features(hi.Population.from_samples(flip), ev), base)
    # train 끝까지만 자른 모집단과도 같음
    assert_frame_close(hi.history_features(pop_of(S, sp), ev), base)


def test_d_le_C_rows_rejected(S):
    r = S.iloc[[0]].copy()
    r["C"] = r["d"]
    with pytest.raises(hi.LeakageError):
        hi.history_features(pop_of(S), r)
    with pytest.raises(hi.LeakageError):
        hi.Reference(S).row(r.iloc[0])
    sp = fd.SPLITS["fold1"]
    with pytest.raises(hi.LeakageError):
        hi.c_for_train_rows([sp.train_end + 400], sp)


def test_audit_d_used_le_C(S):
    sp = fd.SPLITS["fold1"]
    pop = pop_of(S, sp)
    hi.history_features(pop, train_rows(S, sp))
    assert pop.audit and all(a["d_max_used"] <= a["C"] for a in pop.audit)


# ================================================================ NaN·환산·수준
def test_window_before_data_start_is_nan(S):
    """T 가 2015-04-07 ~ 06-30 이면 C = 2014-06-30 → 창 전체가 자료 시작 전 → 전부 NaN."""
    tr = train_rows(S, fd.SPLITS["fold1"])
    early = tr[fd.to_days(tr["C"]) == D("2014-06-30")]
    assert len(early) > 0
    F = hi.history_features(pop_of(S), early)
    assert F.isna().all().all()


def test_n_all_zero_is_nan():
    df = synth(200, seed=3, d_lo="2015-04-07", d_hi="2016-03-31")
    rows = df.iloc[:5][["p2", "p3", "sgg", "sido"]].copy()
    rows["d"] = pd.Timestamp("2018-03-01").date()
    rows["C"] = pd.Timestamp("2017-06-30").date()  # 1년 창 [2016-07-01, 2017-06-30] 시료 0
    F = hi.history_features(hi.Population.from_samples(df), rows)
    one = [c for c in F.columns if "1y" in c]
    five = [c for c in F.columns if "5y" in c or c.startswith("woe_")]
    assert F[one].isna().all().all()
    assert F[five].notna().all().all()


def test_his2_window_scaling():
    """C = 2015-06-30: 실제 일수 = 2015-04-07 ~ 06-30 = 85일(계획서 '1년도 약 85일치') → n × 365/85, n × 1826/85."""
    assert D("2015-06-30") - hi.DATA_START + 1 == 85
    df = synth(400, seed=4, d_lo="2015-04-07", d_hi="2015-06-30")
    row = df.iloc[[0]][["p2", "p3", "sgg", "sido"]].copy()
    row["d"] = pd.Timestamp("2015-08-01").date()
    row["C"] = pd.Timestamp("2015-06-30").date()
    F = hi.history_features(hi.Population.from_samples(df), row).iloc[0]
    n_sgg = int((df["sgg"] == row["sgg"].iloc[0]).sum())
    assert F["hist_logn1y_sgg"] == pytest.approx(math.log1p(n_sgg * 365 / 85))
    assert F["hist_logn5y_sgg"] == pytest.approx(math.log1p(n_sgg * 1826 / 85))
    # 창이 자료 시작 뒤에 다 들어가면 환산 없음
    row2 = row.copy()
    row2["d"] = pd.Timestamp("2021-08-01").date()
    row2["C"] = pd.Timestamp("2021-06-30").date()
    df2 = pd.concat([df, synth(400, seed=5, d_lo="2020-07-01", d_hi="2021-06-30")], ignore_index=True)
    F2 = hi.history_features(hi.Population.from_samples(df2), row2).iloc[0]
    d2 = fd.to_days(df2["d"])
    n1 = int(((df2["sgg"] == row["sgg"].iloc[0]) & (d2 >= D("2020-07-01"))).sum())
    assert F2["hist_logn1y_sgg"] == pytest.approx(math.log1p(n1))


def test_his4_n0_uses_p0_and_his1_unseen_is_p0(S):
    sp = fd.SPLITS["fold2"]
    r = S.iloc[[0]][["p2", "p3", "sgg", "sido"]].copy()
    r["p3"] = "Z9999999999999"
    r["sgg"] = "99999"
    r["d"] = pd.Timestamp("2023-08-01").date()
    r["C"] = pd.Timestamp("2023-06-30").date()
    pop = pop_of(S, sp)
    F = hi.history_features(pop, r).iloc[0]
    d = fd.to_days(S["d"])
    w = (d >= sp.train_end - 364) & (d <= sp.train_end)
    p0 = S["y"][w].sum() / w.sum()
    for k in hi.K_VALUES:
        assert F[f"hist_rate1y_u_k{k:03d}"] == pytest.approx(p0)
        assert F[f"hist_unc1y_u_k{k:03d}"] == pytest.approx(math.sqrt(p0 * (1 - p0) / (k + 1)))
    assert F["hist_logn1y_u"] == 0.0


def test_rr3_nocode_split_by_p2():
    """3단계 '코드없음' 은 (p2, '코드없음') 로 나눠 셈 — p2 가 다른 '코드없음' 끼리 안 섞임."""
    p3k = hi.p3_key(["A0100000000000", "B0200000000000", NO_CODE, "A0100000000000"],
                    [NO_CODE, NO_CODE, NO_CODE, "A0101000000000"])
    assert list(p3k) == [f"A0100000000000|{NO_CODE}", f"B0200000000000|{NO_CODE}", f"{NO_CODE}|{NO_CODE}",
                         "A0101000000000"]
    base = dict(y=0, src_route="domestic", sgg="11110", sido="11", d=pd.Timestamp("2020-01-01").date())
    df = pd.DataFrame([dict(base, lims_data_key="a", p2="A0100000000000", p3=NO_CODE, y=1),
                       dict(base, lims_data_key="b", p2="B0200000000000", p3=NO_CODE),
                       dict(base, lims_data_key="c", p2="B0200000000000", p3=NO_CODE)])
    rows = df[["p2", "p3", "sgg", "sido"]].copy()
    rows["d"] = pd.Timestamp("2021-01-01").date()
    rows["C"] = pd.Timestamp("2020-06-30").date()
    F = hi.history_features(hi.Population.from_samples(df), rows)
    assert np.exp(F["hist_logn1y_p3"]).round().tolist() == [2.0, 3.0, 3.0]  # n = 1, 2, 2
    assert F["hist_rate1y_p3_k010"].iloc[0] > F["hist_rate1y_p3_k010"].iloc[1]


def test_overseas_levels_and_national_p3():
    """해외: sgg = 나라, WOE sido = 'OV' 한 묶음, 3단계 전국은 국내·해외 함께."""
    base = dict(y=0, p2="A0100000000000", p3="A0101000000000", d=pd.Timestamp("2020-01-01").date())
    df = pd.DataFrame([dict(base, lims_data_key="a", src_route="overseas", sgg="OVCHN", sido="OV", y=1),
                       dict(base, lims_data_key="b", src_route="overseas", sgg="OVUSA", sido="OV"),
                       dict(base, lims_data_key="c", src_route="domestic", sgg="11110", sido="11")])
    rows = df[["p2", "p3", "sgg", "sido"]].copy()
    rows["d"] = pd.Timestamp("2021-01-01").date()
    rows["C"] = pd.Timestamp("2020-06-30").date()
    F = hi.history_features(hi.Population.from_samples(df), rows)
    s_all, n_all = 1, 3
    w_ov = math.log((1 + .5) / (s_all + .5)) - math.log((2 - 1 + .5) / (n_all - s_all + .5))
    assert F["woe_sido"].iloc[0] == pytest.approx(w_ov) == pytest.approx(F["woe_sido"].iloc[1])
    assert F["hist_logn1y_p3"].tolist() == pytest.approx([math.log1p(3)] * 3)
    assert F["hist_logn1y_sgg"].tolist() == pytest.approx([math.log1p(1)] * 3)


def test_domestic_sgg_unit_counts_independent_of_overseas(S):
    """부록 B-6: 해외 행을 빼도 국내 행의 시군구·단위 검사 수(HIS2)는 같음(전국·p0 칼럼 제외)."""
    sp = fd.SPLITS["fold1"]
    tr = train_rows(S, sp)
    dom = tr[tr["src_route"] == "domestic"]
    a = hi.history_features(pop_of(S, sp), dom)
    b = hi.history_features(pop_of(S[S["src_route"] == "domestic"], sp), dom)
    cols = [c for c in a.columns if c.startswith("hist_logn") and (c.endswith("_sgg") or c.endswith("_u"))]
    assert_frame_close(a[cols], b[cols])


def test_population_checks(S):
    with pytest.raises(ValueError):
        hi.Population.from_samples(pd.concat([S.iloc[:3], S.iloc[:1]]))
    bad = S.iloc[:3].copy()
    bad["d"] = pd.Timestamp("2015-04-06").date()
    with pytest.raises(ValueError):
        hi.Population.from_samples(bad)
    bad = S.iloc[:3].copy()
    bad.loc[bad.index[0], "sgg"] = None
    with pytest.raises(ValueError):
        hi.Population.from_samples(bad)


# ================================================================ 7-a
def test_te7a_formula_and_name(S):
    sp = fd.SPLITS["fold1"]
    ev = eval_rows(S, sp)
    pop = pop_of(S, sp)
    for L in hi.TE_LENGTHS:
        g = hi.te7a_grid(pop, ev, L, hi.K_VALUES, hi.K_VALUES)
        assert len(g) == 25
        d = fd.to_days(S["d"])
        lo = hi.DATA_START if L == "all" else sp.train_end - hi.PERIOD_DAYS[L] + 1
        w = (d >= lo) & (d <= sp.train_end)
        p0 = S["y"][w].sum() / w.sum()
        p3k = hi.p3_key(S["p2"], S["p3"])
        for i in (0, 7, 33):
            r = ev.iloc[i]
            rk = hi.p3_key([r["p2"]], [r["p3"]])[0]
            sel3 = w & (p3k == rk)
            selu = sel3 & (S["sgg"].to_numpy() == r["sgg"])
            for ku, kp in [(10, 200), (50, 30), (200, 10)]:
                pp3 = (S["y"][sel3].sum() + kp * p0) / (sel3.sum() + kp)
                pu = (S["y"][selu].sum() + ku * pp3) / (selu.sum() + ku)
                assert g[(ku, kp)][i] == pytest.approx(pu, rel=1e-12)
    assert hi.te7a_name("5y", 10, 200) == "7-a-5y-ku010-kp200"
    assert hi.te7a_name("all", 50, 30) == "7-a-all-ku050-kp030"


def test_te7a_eval_future_invariant(S):
    sp = fd.SPLITS["fold3"]
    ev = eval_rows(S, sp)
    a = hi.te7a_grid(hi.Population.from_samples(S), ev, "all", [30], [30])[(30, 30)]
    flip = S.copy()
    after = fd.to_days(flip["d"]) > sp.train_end
    flip.loc[after, "y"] = (1 - flip.loc[after, "y"]).astype(np.int8)
    b = hi.te7a_grid(hi.Population.from_samples(flip), ev, "all", [30], [30])[(30, 30)]
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()


# ================================================================ 부록 A-6
def test_category_rate_formula_window_all_and_custom_level(S):
    sp = fd.SPLITS["fold2"]
    tr = train_rows(S, sp)
    pop = pop_of(S, sp)
    k = 50
    E = hi.category_rates(pop, tr, k)
    assert list(E.columns) == [f"catenc_{c}" for c in hi.CAT5]
    d = fd.to_days(S["d"])
    rng = np.random.default_rng(3)
    for i in rng.choice(len(tr), 20, replace=False):
        r = tr.iloc[i]
        c = int(fd.to_days(pd.Series([r["C"]]))[0])
        w = d <= c  # [2015-04-07, C]
        if c < hi.DATA_START:
            assert E.iloc[i].isna().all()
            continue
        p0 = S["y"][w].sum() / w.sum()
        sel = w & (S["src_route"].to_numpy() == r["src_route"])
        assert E["catenc_src_route"].iloc[i] == pytest.approx((S["y"][sel].sum() + k * p0) / (sel.sum() + k))
        rk = hi.p3_key([r["p2"]], [r["p3"]])[0]
        sel = w & (hi.p3_key(S["p2"], S["p3"]) == rk)
        assert E["catenc_p3"].iloc[i] == pytest.approx((S["y"][sel].sum() + k * p0) / (sel.sum() + k))
    # 사용자 범주(season 꼴): 모집단 = 시료 d 의 달, 행 = 행 d 의 달
    Sp = S[d <= sp.train_end]
    month = pd.to_datetime(Sp["d"]).dt.month.astype(str).to_numpy()
    pop.add_level("month", month)
    rv = pd.to_datetime(tr["d"]).dt.month.astype(str).to_numpy()
    v = hi.category_rate(pop, "month", rv, tr, k)
    assert np.isnan(v[fd.to_days(tr["C"]) < hi.DATA_START]).all()
    assert np.isfinite(v[fd.to_days(tr["C"]) >= hi.DATA_START]).all()
    with pytest.raises(ValueError):
        pop.add_level("bad", np.array([None] * len(Sp), dtype=object))


def test_category_rate_future_invariant(S):
    sp = fd.SPLITS["refit"]
    tr = train_rows(S, sp)
    a = hi.category_rates(hi.Population.from_samples(S), tr, 30)
    Cs = fd.to_days(tr["C"])
    for c in np.unique(Cs)[::3]:
        sel = Cs == c
        keep = S[fd.to_days(S["d"]) <= c]
        b = hi.category_rates(hi.Population.from_samples(keep), tr[sel], 30)
        assert_frame_close(b, a[sel])


def test_reduced_rows_use_full_population(S):
    """단계 4·5·6: 줄이기 전 시료로 계산한 뒤 행을 줄임 = 모집단은 그대로, 행만 줄여도 남은 행 값이 같음."""
    sp = fd.SPLITS["fold1"]
    tr = train_rows(S, sp)
    pop = pop_of(S, sp)
    full = hi.history_features(pop, tr)
    keep = (np.arange(len(tr)) % 3) == 0
    assert_frame_close(hi.history_features(pop, tr[keep]), full[keep])
    # h 1~60 복제 행(4-a): 행마다 자기 T 로 C — 모집단은 시료 1번
    rep = tr.iloc[:50].copy()
    rep = pd.concat([rep.assign(h=hh) for hh in (1, 30, 60)], ignore_index=True)
    rep["T"] = fd.as_date(fd.to_days(rep["d"]) - rep["h"].to_numpy())
    rep["C"] = fd.as_date(hi.c_for_train_rows(rep["T"], sp))
    hi.history_features(pop, rep)  # 돌기만 하면 됨(d > C 확인 포함)
