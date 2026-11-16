"""lst.importance 시험(합성 자료) — 부록 A-5 I·ψ_f·π, B-5 씨앗·h_block, 스태킹·앙상블 같은 순열, 실제 조립 객체 소속."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import folds as fd
from lst import importance as im
from lst import metrics as mt
from lst import registry as rg
from lst.hassign import u_tie_many

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _imp_tune_synth import (LOOKUP, SPLITS, DictAsm, DictModel, FrameAsm, FrameModel,  # noqa: E402
                             StackModel, make_samples)


@pytest.fixture(scope="module")
def data():
    return make_samples()


def frame_leaf(feats, model=None, name="T"):
    asm = FrameAsm(feats)
    return im.Leaf(name, asm, model or FrameModel(), layout=im.layout_from_names(asm.columns, LOOKUP))


def dict_leaf(feats, model=None, name="D"):
    return im.Leaf(name, DictAsm(feats), model or DictModel(),
                   layout=im.layout_from_names({"x_seq": ["avg_temp", "avg_temp__na"], "x_static": ["t3", "g_div50"]},
                                               LOOKUP))


def manual_I(samples, feats, split, fold_no, target, M, cols=("a_col", "a_col__na")):
    """계획서 문구대로 따로 계산: h 오름차순·src_route 순 칸, 칸 안 key 순, rng.permutation, 그 칼럼만 바꿈."""
    asm, model = FrameAsm(feats), FrameModel()
    from lst.runner import add_bounds
    rngs = {r: np.random.default_rng([2026, fold_no, *target.seed_key, r]) for r in range(1, M + 1)}
    bh = np.full(60, np.nan)
    ph = np.full((M, 60), np.nan)
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, split):
        if not idx.size:
            continue
        m = samples.meta().iloc[idx].copy()
        m["h"] = np.int16(h)
        X = asm.transform(add_bounds(m.reset_index(drop=True), None))
        y = samples.cv_y(idx)
        u = u_tie_many(samples.keys[idx], np.full(idx.size, h))
        bh[h - 1] = mt.pr_auc(y, model.predict_proba(X), u)
        routes = samples.src_route[idx]
        keys = samples.keys[idx]
        for r in range(1, M + 1):
            Xp = X.copy()
            for rt in sorted(set(routes)):
                pos = np.flatnonzero(routes == rt)
                pos = pos[np.argsort(keys[pos].astype(str), kind="stable")]
                p = rngs[r].permutation(pos.size)
                for c in cols:
                    v = X[c].to_numpy()
                    Xp.loc[pos, c] = v[pos[p]]
            ph[r - 1, h - 1] = mt.pr_auc(y, model.predict_proba(Xp), u)
    b = mt.band_weighted(bh)
    return b, np.array([b - mt.band_weighted(ph[r]) for r in range(M)])


def test_seeds_b5():
    g = im.group_target("기온")
    assert g.seed_key == (1000 + rg.group("기온").no,)
    assert np.array_equal(g.rng(2, 3).random(5), np.random.default_rng([2026, 2, 1000 + g.no, 3]).random(5))
    u = im.unit_target("rain")
    assert u.seed_key == (5000 + rg.unit("rain").no,)
    t2 = im.targets_2r1()
    cols = [t for t in t2 if t.kind == "column"]
    assert [t.name for t in cols] == list(rg.RESEARCH_UNITS) and [t.seed_key for t in cols] == [(9000 + k,) for k in
                                                                                                  range(1, 10)]
    assert [t for t in t2 if t.kind == "bundle"][0].seed_key == (9000,)
    assert {t.name for t in t2 if t.kind == "group"} == {"기온", "강수", "바람", "햇빛", "습도"}
    fin = im.targets_final({"t3", "L1", "avg_temp", "season"})
    assert [t.seed_key for t in fin if t.kind == "column"] == [(20000, 1), (20000, 9)]
    assert [t.seed_key for t in fin if t.kind == "bundle"] == [(20000, 0)]
    assert {t.name for t in fin if t.kind == "group"} == {"기온", "계절", "L1"}
    assert [t.seed_key for t in fin if t.kind == "group"] == [(21000, rg.group(n).no) for n in ("기온", "계절", "L1")]
    gf = [t for t in fin if t.kind == "group"][0]
    assert np.array_equal(gf.rng(1, 2).random(3), np.random.default_rng([2026, 1, 21000, gf.no, 2]).random(3))
    # 단위 빼기 대상 = 남은 단위 2개 이상 그룹만
    mu = im.targets_multi_unit(["avg_temp", "max_temp", "rain", "season"])
    assert [t.name for t in mu] == ["avg_temp", "max_temp"]


def test_single_leaf_matches_manual_and_metric(data):
    samples, feats = data
    t = im.unit_target("t3")
    M = 3
    res = im.permutation_importance({"fold1": frame_leaf(feats)}, samples, [t], M=M, splits=SPLITS)
    b, I = manual_I(samples, feats, SPLITS["fold1"], 1, t, M)
    assert res.base["fold1"] == b
    np.testing.assert_array_equal(res.records.sort_values("r")["I"].to_numpy(), I)
    # 섞기 전 점수 = 쌍 전부를 이어 metrics.pr_auc_band_weighted(구현 세부 b)
    ys, ps, hs, us = [], [], [], []
    asm, model = FrameAsm(feats), FrameModel()
    from lst.runner import add_bounds
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, SPLITS["fold1"]):
        m = samples.meta().iloc[idx].copy()
        m["h"] = np.int16(h)
        ps.append(model.predict_proba(asm.transform(add_bounds(m.reset_index(drop=True), None))))
        ys.append(samples.cv_y(idx))
        hs.append(np.full(idx.size, h))
        us.append(u_tie_many(samples.keys[idx], np.full(idx.size, h)))
    assert mt.pr_auc_band_weighted(np.concatenate(ys), np.concatenate(ps), np.concatenate(hs),
                                   np.concatenate(us)) == pytest.approx(b, abs=1e-15)
    assert res.n_pairs["fold1"] == sum(len(y) for y in ys)


def test_only_target_columns_change_within_cells(data):
    samples, feats = data
    model = FrameModel()
    leaf = frame_leaf(feats, model)
    im.permutation_importance({"fold2": leaf}, samples, [im.unit_target("t3")], M=2, splits=SPLITS)
    base, big = model.seen[0], model.seen[1]          # h 1: 섞기 전, 섞은 뒤(M 2 이어 붙임)
    n = len(base)
    assert len(big) == 2 * n
    for r in range(2):
        part = big.iloc[r * n:(r + 1) * n].reset_index(drop=True)
        for c in ("sido", "h", "g", "b_col", "ov_struct_na"):
            pd.testing.assert_series_equal(part[c], base[c], check_names=False)
        # a_col·a_col__na 는 같은 순열(행 짝 유지), 칸(src_route) 안에서만
        pair_base = set(zip(base["a_col"].fillna(-99), base["a_col__na"], base["ov_struct_na"]))
        pair_perm = set(zip(part["a_col"].fillna(-99), part["a_col__na"], part["ov_struct_na"]))
        assert pair_perm <= pair_base
        assert not part["a_col"].fillna(-99).equals(base["a_col"].fillna(-99))
    assert list(big.columns) == list(base.columns) and list(big.dtypes) == list(base.dtypes)


def test_r_batch_and_seed_mean_same(data):
    samples, feats = data
    tg = [im.unit_target("t3"), im.unit_target("L1")]
    a = im.permutation_importance({"fold1": frame_leaf(feats)}, samples, tg, M=3, splits=SPLITS)
    b = im.permutation_importance({"fold1": frame_leaf(feats)}, samples, tg, M=3, splits=SPLITS, r_batch=1)
    pd.testing.assert_frame_equal(a.records, b.records)
    two = im.Mean("seed", [frame_leaf(feats, name="s0"), frame_leaf(feats, name="s1")])
    c = im.permutation_importance({"fold1": two}, samples, tg, M=3, splits=SPLITS)
    np.testing.assert_allclose(c.records["I"], a.records["I"], atol=1e-12)


def test_stack_feeder_same_permutation(data):
    samples, feats = data
    dlm, stm = DictModel(), StackModel()
    feeder = dict_leaf(feats, dlm)
    base = frame_leaf(feats, stm, name="B")
    node = im.Stack("S", base, {"dl_p": feeder})
    t = im.unit_target("t3")
    res = im.permutation_importance({"fold1": node}, samples, [t], M=2, splits=SPLITS)
    xb, xp = stm.seen[1], dlm.seen[1]                 # h 1 섞은 입력(바탕·feeder)
    # 같은 순열: 바탕 a_col(NaN → 0) == feeder x_static t3 칼럼
    np.testing.assert_array_equal(np.nan_to_num(xb["a_col"].to_numpy()), xp["x_static"][:, 0])
    # feeder 의 x_seq avg_temp 채널(다른 단위)·g_div50 은 안 섞임
    base_in = dlm.seen[0]
    n = len(base_in["x_static"])
    np.testing.assert_array_equal(xp["x_seq"], np.tile(base_in["x_seq"], (2, 1, 1)))
    np.testing.assert_array_equal(xp["x_static"][:, 1], np.tile(base_in["x_static"][:, 1], 2))
    # 붙인 칼럼 dl_p 는 feeder 의 섞은 예측
    np.testing.assert_allclose(xb["dl_p"].to_numpy(), DictModel().predict_proba(xp))
    assert len(xb) == 2 * n and (res.records["I"].abs() > 0).any()
    # 그룹 기온 = avg_temp 채널(+ 결측 채널)·t3 둘 다
    sel = feeder.get_layout().select(im.group_target("기온"))
    assert sel["x_seq"].tolist() == [0, 1] and sel["x_static"].tolist() == [0]


def test_absent_target_and_summary_rules(data):
    samples, feats = data
    with pytest.raises(im.ImportanceError):
        im.permutation_importance({"fold1": frame_leaf(feats)}, samples, [im.group_target("대기질")], M=1,
                                  splits=SPLITS)
    tg = [im.group_target("기온"), im.group_target("강수"), im.group_target("계절")]
    rows = []
    vals = {"기온": [0.01] * 10, "강수": [0.01] * 7 + [-0.01] * 3, "계절": [0.0] * 10}
    for t in tg:
        for f in fd.CV_FOLDS:
            for r in range(1, 11):
                v = vals[t.name][r - 1]
                if t.name == "강수" and f == "fold3":
                    v = 0.02
                rows.append({"fold": f, "target_kind": t.kind, "target_no": t.no, "target": t.name, "r": r,
                             "score_base": 0.1, "score_perm": 0.1 - v, "I": v})
    s = im.summarize(pd.DataFrame(rows), fd.CV_FOLDS, tg, 10).set_index("target")
    assert s.loc["기온", "skip"] and s.loc["기온", "pi"] == 1.0
    assert not s.loc["강수", "skip"] and s.loc["강수", "psi_fold1"] == 0.7 and s.loc["강수", "psi_fold3"] == 1.0
    assert s.loc["계절", "pi"] == 0.0 and not s.loc["계절", "skip"]          # I = 0 은 I > 0 아님
    res = im.ImportanceResult(pd.DataFrame(rows), s.reset_index(), {}, {}, 10, fd.CV_FOLDS, {}, {})
    assert res.removal_order() == [("group", rg.group("계절").no), ("group", rg.group("강수").no)]


def test_full_three_folds_and_save(data, tmp_path):
    samples, feats = data
    models = {f: frame_leaf(feats) for f in fd.CV_FOLDS}
    tg = [im.unit_target("t3"), im.unit_target("L1")]
    res = im.permutation_importance(models, samples, tg, M=2, splits=SPLITS,
                                    expect_base=None)
    assert set(res.records["fold"]) == set(fd.CV_FOLDS) and len(res.records) == 2 * 3 * 2
    assert res.summary["skip"].map(lambda v: isinstance(v, (bool, np.bool_))).all()
    paths = res.save(tmp_path)
    assert all(p.exists() for p in paths)
    with pytest.raises(FileExistsError):
        res.save(tmp_path)
    with pytest.raises(im.ImportanceError):
        im.permutation_importance(models, samples, tg, M=2, splits=SPLITS, expect_base={"fold1": 0.0})


# ---------------------------------------------------------------- 실제 조립 객체(registry 소속)
def _registry_ctx():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_registry_inputs as tri
    from lst import inputs as ip
    from lst import wxfeat as wf

    rng = np.random.default_rng(7)                       # test_registry_inputs.ctx 와 같은 합성 자료
    dom, ov = tri._weather(rng)
    obs, fc, sov = tri._spi(rng)
    s = tri._samples(rng)
    c = ip.InputContext(samples=s, y_cut=fd.day("2025-06-30"))
    c.wx = wf.combined_cube(dom, ov)
    c.spi_obs, c.spi_fc = ip.spi_cubes(obs, fc, sov)
    c.food = tri._food()
    c.livestock = pd.DataFrame({"unit": ["11110", "OVCHN"], "L1": [3.5, 0.2]})
    c.soil = tri._soil(rng)
    c.eco = tri._indicator("eco_ratio")
    c.gw = tri._indicator("gw_pass_rate")
    c.missing = rg.MissingLists({1: frozenset(rg.WEATHER19), 2: frozenset(
        list(rg.WEATHER19) + ["spi3", "food_aw", "food_ph", "hist_rate5y_p3", "td1", "L1"] + list(rg.RG_SOILT))})
    return c


def test_real_tree_and_dl_layouts():
    from lst import evaluate as ev
    from lst import inputs as ip
    from lst import models_dl as md
    from lst import models_tree as mtr
    from lst.runner import add_bounds

    ctx = _registry_ctx()
    spec = rg.VersionSpec("t", 2, 20, "sum", include_ids=rg.ALL_IDS, k=50, n_ma=15, n_ma_dl=7)
    split = fd.Split("fold1", fd.day("2016-06-30"), fd.day("2016-07-01"), fd.day("2017-05-01"))
    samples = ev.SampleTable(ctx.samples)
    rows = add_bounds(fd.make_train_rows(samples.known_frame(), split).reset_index(drop=True), None)
    targets = [im.group_target("기온"), im.group_target("계절"), im.unit_target("food_aw"),
               im.research_column_target("td1")]
    # 로지스틱(표준화 + 결측 표시)
    asm = ip.TreeInputs(ctx, spec, split, scale=True).fit(rows)
    X = asm.transform(rows)
    ind = [c for c in X.columns if c.endswith(rg.NA_SUFFIX) or c == rg.STRUCT_COLUMN]
    lr = mtr.make_model("LR", cat_cols=[c for c in X.columns if X[c].dtype == object], indicator_cols=ind)
    lr.fit(X, rows["y"].to_numpy(np.int64), seed=42, n_jobs=1)
    leaf = im.Leaf("LR", asm, lr)
    lay = leaf.get_layout()
    sel = [lay.slots[None][i].name for i in lay.select(im.group_target("기온"))[None]]
    assert "avg_temp_mean" in sel and "avg_temp_mean__na" in sel and "t3" in sel and rg.STRUCT_COLUMN not in sel
    assert "td1" not in sel
    sel_td = [lay.slots[None][i].name for i in lay.select(im.research_column_target("td1"))[None]]
    assert set(sel_td) == {"td1", "td1__na"}
    assert [lay.slots[None][i].name for i in lay.select(im.unit_target("food_aw"))[None]][:2] == ["food_aw", "aw_label"]
    res = im.permutation_importance({"fold1": leaf}, samples, targets, M=2, splits={"fold1": split})
    assert np.isfinite(res.records["I"]).all()
    # 딥러닝(x_seq 채널·x_static 원핫)
    dasm = ip.DLInputs(ctx, spec, split).fit(rows)
    Xd = dasm.transform(rows)
    net = md.DLClassifier("LSTM", dasm.cat_sizes, device="cpu", max_epochs=1)
    net.fit(Xd, rows["y"].to_numpy(np.int64), seed=42, n_jobs=1, groups=(rows["sgg"].to_numpy(), fd.day_ints(rows["d"])))
    dl = im.Leaf("LSTM", dasm, net)
    dlay = dl.get_layout()
    s = dlay.select(im.group_target("기온"))
    seq_names = [dlay.slots["x_seq"][i].name for i in s["x_seq"]]
    assert "avg_temp" in seq_names and "avg_temp__na" in seq_names and "avg_temp_anom7" in seq_names
    st = [dlay.slots["x_static"][i].name for i in dlay.select(im.group_target("계절"))["x_static"]]
    assert st == [f"season_{v}" for v in ("봄", "여름", "가을", "겨울")] or set(st) == {c for c in dasm.prep.static_names
                                                                              if c.startswith("season_")}
    res2 = im.permutation_importance({"fold1": im.Mean("ens", [leaf, dl])}, samples, targets, M=2,
                                     splits={"fold1": split})
    assert len(res2.summary) == len(targets)
