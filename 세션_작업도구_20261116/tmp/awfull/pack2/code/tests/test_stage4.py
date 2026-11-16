"""lst/stage4 시험(합성 자료·가짜 모델): 줄이기·5/6 뽑기·h 붙이기가 train 에만(validation·test 그대로)·검출 행은 늘 남음,
5·6 음성 뽑기 규칙(씨앗 0·해마다 4배·6 = 양성 칸), 4-a h 10개 식, ① h 1 쌍만·4-c 상위 5, 범주 숫자 인코딩 값,
블록 값 모델 OOF 배정, 앙상블 평균, 판정 규칙((가)→(나), 4자리), ② 60칸 모델."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import evaluate as ev
from lst import folds as fd
from lst import history as hi
from lst import runner as rn
from lst import stage4 as s4
from tests.test_runner import STUB, FakeAssembler, FakeModel, env  # noqa: F401
from tests.test_chain import make_model as chain_model


# ---------------------------------------------------------------- 5·6 뽑기
def test_hon_sample_rules():
    rng = np.random.default_rng(0)
    n = 3000
    keys = np.array([f"K{i:05d}" for i in range(n)], dtype=object)
    d = rng.integers(fd.day("2016-01-01"), fd.day("2019-12-31"), n)
    y = (rng.random(n) < 0.03).astype(int)
    cells = np.array([f"c{i}" for i in rng.integers(0, 40, n)], dtype=object)
    year = d.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    for kind in ("5", "6"):
        keep, tab = s4.hon_sample(keys, d, y, cells, kind)
        assert keep[y == 1].all()                                   # 양성 전부
        for r in tab.itertuples():
            iy = year == r.year
            assert r.n_pos == int(y[iy].sum())
            assert r.drawn_neg == int((keep & iy & (y == 0)).sum()) == min(4 * r.n_pos, r.pool_neg)
            if kind == "6":                                          # 음성은 그해 양성이 나온 칸에서만
                pc = set(cells[iy & (y == 1)])
                assert set(cells[keep & iy & (y == 0)]) <= pc
        keep2, _ = s4.hon_sample(keys, d, y, cells, kind)           # 씨앗 0 결정적
        assert np.array_equal(keep, keep2)
        keep3, _ = s4.hon_sample(keys, d, y, cells, kind, seed=1)
        assert not np.array_equal(keep, keep3)
    # 모자라면 그해 음성 전부
    k, t = s4.hon_sample(keys[:10], np.full(10, fd.day("2017-03-01")), np.r_[np.ones(3, int), np.zeros(7, int)],
                         np.array(["a"] * 10, dtype=object), "5")
    assert k.all() and bool(t["short"].iloc[0])


def test_m10_replication_formula():
    rows = fd.make_train_rows(pd.DataFrame({"lims_data_key": ["A1", "B2", "C3"], "d": pd.to_datetime(
        ["2020-05-01", "2021-01-01", "2019-07-07"]).date, "y": [0, 1, 0]}), fd.SPLITS["refit"])
    out = s4.replicate(rows, "4-a")
    assert len(out) == 30
    for k, h in zip(rows["lims_data_key"], rows["h"]):
        hs = out.loc[out["lims_data_key"] == k, "h"].to_numpy()
        import hashlib
        rng = np.random.default_rng(int.from_bytes(hashlib.md5(f"{k}|m42".encode("utf-8")).digest()[:8], "big"))
        want = np.sort(np.r_[h, rng.choice([x for x in range(1, 61) if x != h], 9, replace=False)])
        assert np.array_equal(np.sort(hs), want) and len(set(hs)) == 10 and h in hs
    assert (fd.day_ints(out["d"]) - out["h"].to_numpy() == fd.day_ints(out["T"])).all()
    x1 = s4.replicate(rows, "x1")
    assert (x1["h"] == 1).all() and len(x1) == 3


def test_select_rows_reduce_keeps_positives():
    n = 400
    rng = np.random.default_rng(1)
    pool = pd.DataFrame({"lims_data_key": [f"K{i:04d}" for i in range(n)], "y": (rng.random(n) < 0.1).astype(int),
                         "dd": rng.integers(fd.day("2016-01-01"), fd.day("2025-06-30"), n)})
    pool["p3k"] = np.array(["A", "B", "C", "D", "E", "F", "G"], dtype=object)[rng.integers(0, 7, n)]
    pool["cell"] = pool["p3k"]
    vals = pd.DataFrame({"prduct_nm": [f"p{i % 50}" for i in range(n)]})
    for s in rn.SPLIT_ORDER:
        m, rec = s4.select_rows(pool, ["reduce"], red_col={s: "prduct_nm"}, red_values=vals, split=s)
        tr = pool["dd"].to_numpy() <= fd.SPLITS[s].train_end
        assert not (m & ~tr).any()                                  # train 밖(validation·test) 은 다루지 않음
        assert (m | ~(tr & (pool["y"] == 1))).all()                 # 검출 행은 늘 남음
        v = vals["prduct_nm"].to_numpy()
        zero = {x for x in set(v[tr]) if pool["y"].to_numpy()[tr & (v == x)].sum() == 0}
        assert set(v[m]).isdisjoint(zero) and set(v[tr]) - zero <= set(v[m])
        m2, _ = s4.select_rows(pool, ["reduce", "hon6"], red_col={s: "prduct_nm"}, red_values=vals, split=s)
        assert not (m2 & ~m).any() and (m2 | ~(m & (pool["y"] == 1))).all()


def test_top5_codes_tie_and_no_code():
    p = np.array(["A", "A", "B", "B", "C", "D", "E", "F", "F", f"X|{hi.NO_CODE}", f"X|{hi.NO_CODE}"], dtype=object)
    y = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
    top = s4.top5_codes(p, y)
    assert top[0] == "A" and f"X|{hi.NO_CODE}" not in top
    assert top[1] in ("B", "F")                                     # 검출 1·검사 2 → 검사 수 많은 쪽, 같으면 코드 순
    assert top[1:3] == ["B", "F"]


# ---------------------------------------------------------------- runner 와 이어서(합성 시료)
def _ctx(env, out, log, model=FakeModel, asm=FakeAssembler):
    return rn.RunContext(samples=env["st"], out_dir=out, code_root=Path(__file__).resolve().parents[1],
                         input_md5s={"x": "0"}, exp_log=log, make_inputs=asm, make_model=model,
                         n_jobs=8, check_resources=STUB, require_7a_logged=False)


def _spec(name, **kw):
    base = dict(name=name, stage="4", order="2차", kind="tree", algorithm="XGB", W=20, mode="sum",
                input_set="합성", role="후보", train_count=4, extra={"version": {"include_ids": ["WEATHER"]}})
    base.update(kw)
    return rn.VariantSpec(**base)


def test_train_only_selection_and_h10(env, tmp_path):
    st = env["st"]
    sel = env["inputs"] / "s4" / "sel-hon5"
    info = s4.prepare_selection(st, sel, ["hon5"])
    assert set(info["per_split"]) == set(rn.SPLIT_ORDER)
    plan = s4.RowPlan(("hon5",), "4-a", "s4/sel-hon5")
    known = st.known_frame()
    for s in rn.SPLIT_ORDER:
        rows = s4.plan_train_rows(known, s, plan, env["inputs"])
        kept = s4.load_kept(env["inputs"], "s4/sel-hon5", s)
        tr = known[fd.day_ints(known["d"]) <= fd.SPLITS[s].train_end]
        assert len(rows) == 10 * len(kept)
        assert set(tr.loc[tr["y"] == 1, "lims_data_key"]) <= set(kept)       # 검출 행 남음
        assert (fd.day_ints(rows["d"]) <= fd.SPLITS[s].train_end).all()
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    sp = _spec("4-a-XGB", algorithm="XGB")
    with s4.hooks(plan=plan, inputs_root=env["inputs"]):
        r = rn.run_group([sp], _ctx(env, tmp_path / "out", log))[0]
    assert r["status"] == "done"
    tm = rn.read_json(tmp_path / "out" / sp.name / "run_meta.json")["timings"]
    for s in rn.SPLIT_ORDER:
        assert tm[s]["train_rows"] == 10 * len(s4.load_kept(env["inputs"], "s4/sel-hon5", s))
        want = st.expected_pairs(s)                                 # validation·test 쌍은 전부
        assert tm[s]["n_pairs"] == want
    assert rn._train_rows is s4._ORIG_TRAIN_ROWS                     # 원래대로


def test_x1_and_top5_pair_filters(env, tmp_path):
    st = env["st"]
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    with s4.hooks(plan=s4.RowPlan((), "x1"), inputs_root=env["inputs"], pair_filter=s4.h1_filter):
        rn.run_group([_spec("x1-XGB")], _ctx(env, tmp_path / "o1", log))
        p = pd.read_parquet(tmp_path / "o1" / "x1-XGB" / "pred_fold1.parquet")
        assert set(p["h"]) == {1}
        assert len(p) == st.expected_pairs("fold1")
    assert st.expected_pairs("fold1") > 0
    sel = env["inputs"] / "s4" / "sel-top5"
    info = s4.prepare_selection(ev.SampleTable(st.df.copy()), sel, ["top5"])
    flt = s4.top5_filter(st, info["top5"])
    st2 = ev.SampleTable(st.df.copy())
    with s4.hooks(plan=s4.RowPlan(("top5",), "own", "s4/sel-top5"), inputs_root=env["inputs"], pair_filter=flt):
        rn.run_group([_spec("4-c-XGB")], _ctx({**env, "st": st2}, tmp_path / "o2", log))
        p = pd.read_parquet(tmp_path / "o2" / "4-c-XGB" / "pred_refit.parquet")
        pos = st2.positions(p["lims_data_key"])
        keys = hi.p3_key(st2.df["p2"], st2.df["p3"])[pos]
        assert set(keys) <= set(info["top5"]["refit"])


def test_catenc_values():
    from tests.test_registry_inputs import ctx as ctx_fixture  # noqa: F401
    from lst import inputs as ip
    from lst import registry as rg
    from tests import test_registry_inputs as tri

    rng = np.random.default_rng(7)
    dom, ov = tri._weather(rng)
    s = tri._samples(rng, n=400)
    c = ip.InputContext(samples=s, y_cut=fd.day("2025-06-30"))
    c.food = tri._food()
    sp = fd.Split("t", fd.day("2016-06-30"), fd.day("2016-07-01"), fd.day("2017-05-01"))
    enc = s4.CatEncoder(c, sp, 30)
    meta = s[s["d"].map(lambda x: pd.Timestamp(x) > pd.Timestamp("2016-07-20"))].head(30).copy()
    meta["h"] = np.int16(5)
    X = pd.DataFrame({"sgg": meta["sgg"].to_numpy(), "p3": meta["p3"].to_numpy(), "season": "여름",
                      "aw_label": "결측", "z": 1.0})
    out = s4.CatEncoder.encode(enc, X, meta.reset_index(drop=True))
    assert list(out.columns) == ["catenc_sgg", "catenc_p3", "catenc_season", "catenc_aw_label", "z"]
    # 손 계산: sgg 칸 = (s + k·p0) ÷ (n + k), 기간 [2015-04-07, C], C = T 직전 06-30
    T = fd.day_ints(meta["d"]) - 5
    C = hi.c_from_T(T)
    pop = s[fd.day_ints(s["d"]) <= sp.train_end]
    for i in range(5):
        pm = pop[(fd.day_ints(pop["d"]) <= C[i])]
        p0 = pm["y"].astype(int).mean()
        g = pm[pm["sgg"] == meta["sgg"].iloc[i]]
        want = (g["y"].astype(int).sum() + 30 * p0) / (len(g) + 30)
        assert out["catenc_sgg"].iloc[i] == pytest.approx(want, rel=1e-6)
    assert out["catenc_sgg"].dtype == np.float32


# ---------------------------------------------------------------- 블록 값 모델 OOF 배정
class _ConstProvider(s4._OOFProvider):
    columns = ("v",)

    def fit_block(self, rows, split):
        return {"keys": set(rows["lims_data_key"]), "val": float(len(rows))}

    def predict_block(self, m, meta, split):
        assert not (set(meta["lims_data_key"]) & m["keys"]) or fd.day_ints(meta["d"]).min() > fd.SPLITS[split].train_end
        return np.full((len(meta), 1), m["val"])

    def save_block(self, m, path):
        path.mkdir(parents=True)


def test_oof_provider_assignment(env):
    from lst import blocks as bk

    st = env["st"]

    class W:
        samples = st
        check = staticmethod(lambda: None)

        def block_selector(self, plan):
            return s4.BlockSelector(st, plan, env["inputs"], None, None)

    pr = _ConstProvider(W(), {"kind": "x"}, s4.RowPlan(), 42)
    for split in ("fold1", "refit"):
        rows = fd.make_train_rows(st.known_frame(), fd.SPLITS[split])
        v = pr.values(rows, split)
        plan = pr.fitted[split]["plan"]
        who = bk.assemble_oof(plan, split, fd.day_ints(rows["d"]))
        sizes = {i: int(plan[i].train_mask(fd.day_ints(st.known_frame()["d"])).sum()) for i in set(who)}
        assert np.array_equal(v[:, 0], np.array([sizes[i] for i in who], dtype=float))
        pairs = next(iter(fd.iter_scoring_pairs_by_h(st.d_days, fd.SPLITS[split])))[1]
        ev_meta = st.meta().iloc[pairs]
        full = plan.index(bk.full_model(plan, split))
        assert (pr.values(ev_meta, split)[:, 0] == pr.fitted[split]["models"][full]["val"]).all()


# ---------------------------------------------------------------- 앙상블·판정
def _mk(env, tmp_path, names, algs, seedshift=0):
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    out = tmp_path / "s4"
    for n, a in zip(names, algs):
        rn.run_group([_spec(n, algorithm=a, input_set=f"합성-{n}")], _ctx(env, out, log))
    return out, log


def test_write_ensemble_mean(env, tmp_path):
    out, log = _mk(env, tmp_path, ["E--XGB", "E--LGB"], ["XGB", "LGB"])
    sp = rn.VariantSpec(name="E", stage="4", order="2차", kind="tree", algorithm="T3", W=20, mode="sum",
                        input_set="E-구성", role="후보", train_count=0, extra={"s4": {"members": ["E--XGB", "E--LGB"]}})
    d = s4.write_ensemble(out, sp, env["st"], log, "c" * 32)
    a, b, e = (pd.read_parquet(out / n / "pred_refit.parquet") for n in ("E--XGB", "E--LGB", "E"))
    assert np.allclose(e["pred"], (a["pred"] + b["pred"]) / 2)
    assert d["cv_band_weighted"] == rn.load_done(out, "E")[2]["cv_band_weighted"]
    assert s4.write_ensemble(out, sp, env["st"], log, "c" * 32)["spec_md5"] == sp.md5()   # 이어하기


def test_judge_step_order(env, tmp_path):
    out, log = _mk(env, tmp_path, ["cur", "a", "b", "r"], ["XGB", "LGB", "CAT", "RF"])
    cv = {n: rn.load_done(out, n)[2]["cv_band_weighted"] for n in ("cur", "a", "b", "r")}
    inputs = tmp_path
    best = lambda n, kind="tree", repl=False: {"name": n, "kind": kind, "dir": f"s4/{n}", "replacement": repl,  # noqa
                                               "members": [{"name": n, "dir": f"s4/{n}"}]}
    state = {"step": "3-3", "current_best": best("cur"), "tree_target": best("cur"), "tree_top3": [],
             "tree_rank_3_0b": {}, "history": []}
    res = s4.judge_step(step="4", state=state, stack=[(best("a"), out / "a"), (best("b"), out / "b")],
                        repl=[(best("r", "nn60", True), out / "r")], samples=env["st"], inputs_root=inputs)
    rec = res["record"]
    top = max(("a", "b"), key=lambda n: (round(cv[n], 4), -("a", "b").index(n)))
    assert rec["parts"][0]["top"] == top
    adopt1 = round(cv[top], 4) > round(cv["cur"], 4)
    assert rec["parts"][0]["adopted"] == adopt1
    cur_after1 = top if adopt1 else "cur"
    assert rec["parts"][1]["against"] == cur_after1                 # (나) 는 (가) 뒤 현재 최고 판과
    adopt2 = round(cv["r"], 4) > round(cv[cur_after1], 4)
    assert rec["after"] == ("r" if adopt2 else cur_after1)
    assert res["state"]["tree_target"]["name"] == (top if adopt1 else "cur")   # 교체 후보는 트리 대상 판 안 바꿈


def test_s4_names_and_seed_override():
    class M:
        def fit(self, X, y, *, seed, n_jobs, eval_set=None):
            self.seed = seed
            return self

        def predict_proba(self, X, n_iter=None):
            return np.zeros(len(X))
    m = s4.SeedOverride(M(), 0).fit([1], [0], seed=42, n_jobs=1)
    assert m._inner.seed == 0 and m.predict_proba([1, 2]).shape == (2,)
    assert "T3" in rn.ALG_ABBR and "NN60" in rn.ALG_ABBR


def test_nn60_model_fit_predict(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    NN = s4._nn60_model_cls()
    rng = np.random.default_rng(0)
    n = 600
    X = {"x_tab": rng.normal(size=(n, 5)).astype(np.float32), "x_cat": rng.integers(0, 4, (n, 4)).astype(np.int64),
         "x_h": rng.integers(1, 61, n).astype(np.int64)}
    y = (X["x_tab"][:, 0] + 0.1 * X["x_h"] / 60 > 0.8).astype(int)
    groups = (np.array([f"s{i % 30}" for i in range(n)]), np.arange(n) % 17)
    m = NN([5, 5, 5, 5], device="cpu", max_epochs=2).fit(X, y, seed=42, n_jobs=1, groups=groups)
    p = m.predict_proba(X)
    assert p.shape == (n,) and np.isfinite(p).all()
    X2 = dict(X, x_h=np.where(X["x_h"] == 60, 1, X["x_h"] + 1))
    assert not np.allclose(p, m.predict_proba(X2))                  # h 칸마다 다른 출력
    m.save(tmp_path / "m.pt")
    assert np.allclose(NN.load(tmp_path / "m.pt", device="cpu").predict_proba(X), p)


# ---------------------------------------------------------------- 실제 입력 조립 경로(합성 InputContext)
def _syn_ctx():
    from lst import inputs as ip
    from lst import registry as rg
    from lst import wxfeat as wf
    from tests import test_registry_inputs as tri

    rng = np.random.default_rng(7)
    dom, ov = tri._weather(rng)
    obs, fc, sov = tri._spi(rng)
    s = tri._samples(rng, n=500)
    c = ip.InputContext(samples=s, y_cut=fd.day("2025-06-30"))
    c.wx = wf.combined_cube(dom, ov)
    c.spi_obs, c.spi_fc = ip.spi_cubes(obs, fc, sov)
    c.food = tri._food()
    c.livestock = pd.DataFrame({"unit": ["11110", "OVCHN"], "L1": [3.5, 0.2]})
    c.soil = tri._soil(rng)
    c.eco = tri._indicator("eco_ratio")
    c.gw = tri._indicator("gw_pass_rate")
    c.missing = rg.MissingLists({1: frozenset(rg.WEATHER19), 2: frozenset(list(rg.WEATHER19) + ["food_aw"])})
    st = ev.SampleTable(s.assign(d=pd.to_datetime(s["d"]).dt.date))
    return c, st


def test_s4_inputs_catenc_vae_lr_nn60(tmp_path):
    pytest.importorskip("torch")
    ctx, st = _syn_ctx()
    w = s4.Stage4Wiring(ctx, st, tmp_path, n_jobs=1, device="cpu", dl_kwargs={"max_epochs": 1})
    ver = {"include_ids": ["WEATHER", "HIS1", "WX7", "FD1"], "k": 30}
    vae = {"kind": "vae", "column": "anomaly_vae", "version": ver, "W": 10, "mode": "sum", "source": "3-1-VAEt"}
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), 10)
    for alg in ("XGB", "LR"):
        sp = rn.VariantSpec(name=f"4-a0-{alg}", stage="4", order="2차", kind="tree", algorithm=alg, W=10, mode="sum",
                            input_set="x", role="후보", train_count=4,
                            extra={"version": ver, "s4": {"catenc_k": 30, "provider": vae,
                                                          "row_plan": s4.RowPlan().to_dict()}})
        asm = w.make_inputs(sp, "fold1").fit(rows)
        X = asm.transform(rows)
        assert list(X.columns) == asm.columns
        assert asm.cat_cols == [] and not any(X[c].dtype == object for c in X.columns)
        assert {"catenc_sgg", "catenc_p3", "catenc_season", "catenc_aw_label"} <= set(X.columns)
        assert "anomaly_vae" in X.columns and np.isfinite(X["anomaly_vae"]).all()
        if alg == "LR":
            assert abs(float(X["catenc_sgg"].mean())) < 1e-4            # 표준화(로지스틱)
            m = w.make_model(sp, asm)
            m.fit(X, rows["y"].to_numpy(dtype=int), seed=42, n_jobs=1)
        asm.save(tmp_path / f"inp_{alg}")
        assert (tmp_path / f"inp_{alg}" / "value_models" / "plan.json").exists()
    assert len(w._providers) == 1                                   # 값 모델은 판 사이에 한 벌(메모리)
    # ① h·g 빼기
    sp1 = rn.VariantSpec(name="x1-XGB", stage="4", order="2차", kind="tree", algorithm="XGB", W=10, mode="sum",
                         input_set="x", role="비교만", train_count=4,
                         extra={"version": ver, "s4": {"catenc_k": 30, "drop_time": True}})
    r1 = rn.add_bounds(s4.replicate(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), "x1"), 10)
    a1 = w.make_inputs(sp1, "fold1").fit(r1)
    assert "h" not in a1.transform(r1).columns and "g" not in a1.columns
    # ② 입력
    nn = rn.VariantSpec(name="x2-NN60", stage="4", order="2차", kind="dl", algorithm="NN60", W=10, mode="sum",
                        input_set="x", role="교체 후보", train_count=4, extra={"version": ver, "s4": {}})
    an = w.make_inputs(nn, "fold1").fit(rows)
    Xn = an.transform(rows)
    assert Xn["x_tab"].dtype == np.float32 and np.isfinite(Xn["x_tab"]).all()
    assert np.array_equal(Xn["x_h"], rows["h"].to_numpy()) and Xn["x_cat"].min() >= 1
    assert Xn["x_tab"].shape[1] + 4 + 1 == len(an.columns)
    mdl = w.make_model(nn, an)
    mdl.fit(Xn, rows["y"].to_numpy(dtype=int), seed=42, n_jobs=1, **w.fit_kwargs(nn, "fold1", rows, Xn))
    assert mdl.predict_proba(Xn).shape == (len(rows),)


def test_stack4_meta_inputs(tmp_path):
    ctx, st = _syn_ctx()
    w = s4.Stage4Wiring(ctx, st, tmp_path, n_jobs=1, device="cpu")
    pdef = {"kind": "stack4", "version": {"include_ids": []}, "W": 10, "mode": "sum", "catenc_k": 30, "params": None}
    sp = rn.VariantSpec(name="4-a0-x4-L6", stage="4", order="2차", kind="tree", algorithm="LR", W=10, mode="sum",
                        input_set="x", role="후보", train_count=4,
                        extra={"version": {"include_ids": []}, "s4": {"provider": pdef, "feature_only": True,
                                                                      "row_plan": s4.RowPlan().to_dict()}})
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold2"]), 10)
    asm = w.make_inputs(sp, "fold2").fit(rows)
    X = asm.transform(rows)
    assert list(X.columns) == list(s4.STACK4_COLS) == asm.columns
    m = w.make_model(sp, asm).fit(X, rows["y"].to_numpy(dtype=int), seed=42, n_jobs=1)
    assert m.predict_proba(X).shape == (len(rows),)


def test_stage8_reattach_anomaly_no_training(tmp_path, monkeypatch):
    """단계 8: 이상 점수 칼럼 = 대상 판 값 모델 그대로(학습 없음, 값 같음) — derive_specs(anomaly_reuse=True)."""
    pytest.importorskip("torch")
    from lst import models_dl as md

    ctx, st = _syn_ctx()
    w = s4.Stage4Wiring(ctx, st, tmp_path, n_jobs=1, device="cpu", dl_kwargs={"max_epochs": 1})
    ver = {"include_ids": ["WEATHER", "HIS1", "WX7", "FD1"], "k": 30}
    an = {"column": "anomaly_vae", "version": ver, "W": 10, "mode": "sum", "source": "3-1-VAEt"}
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), 10)
    tgt = rn.VariantSpec(name="4-a0-XGB", stage="4", order="2차", kind="tree", algorithm="XGB", W=10, mode="sum",
                         input_set="x", role="후보", train_count=4, params={rn.ITER_SELECT: True},
                         extra={"version": ver, "s4": {"catenc_k": 30, "provider": s4._anomaly_pdef({"anomaly": an}),
                                                       "row_plan": s4.RowPlan().to_dict()}})
    vdir = tmp_path / "s4" / "4-a0-XGB"
    vdir.mkdir(parents=True)
    (vdir / "spec.json").write_text(json.dumps(tgt.to_dict()), encoding="utf-8")
    a0 = w.make_inputs(tgt, "fold1").fit(rows)
    v0 = a0.transform(rows)["anomaly_vae"].to_numpy()
    a0.save(vdir / "inputs_fold1")                                  # runner 가 판 폴더에 저장하는 꼴
    best = {"name": "4-a0-XGB", "kind": "tree", "dir": "s4/4-a0-XGB", "replacement": False,
            "members": [{"name": "4-a0-XGB", "dir": "s4/4-a0-XGB", "anomaly": an,
                         "row_plan": s4.RowPlan().to_dict(), "catenc_k": 30}]}
    d = s4.derive_specs(best, inputs_root=tmp_path, name="8-T-XGB", stage="8", role="후보", plan=s4.RowPlan(),
                        catenc_k=30, member_params={"4-a0-XGB": {"max_depth": 3}}, anomaly_reuse=True)
    pd8 = s4.s4_of(d.run[0])["provider"]
    assert pd8["kind"] == "vae_frozen" and pd8["from_dir"] == "s4/4-a0-XGB"
    assert d.run[0].params == {rn.ITER_SELECT: True, "params": {"max_depth": 3}}   # 반복 수만 다시
    d4 = s4.derive_specs(best, inputs_root=tmp_path, name="5-x", stage="5-6", role="후보", plan=s4.RowPlan(),
                         catenc_k=30)
    assert s4.s4_of(d4.run[0])["provider"]["kind"] == "vae"          # 단계 4~6 은 다시 학습 그대로

    def boom(*a, **k):
        raise AssertionError("값 모델 학습이 일어남")
    monkeypatch.setattr(md.AnomalyModel, "fit", boom)
    w8 = s4.Stage4Wiring(ctx, st, tmp_path, n_jobs=1, device="cpu")
    a8 = w8.make_inputs(d.run[0], "fold1").fit(rows)
    v8 = a8.transform(rows)["anomaly_vae"].to_numpy()
    assert np.allclose(v8, v0, rtol=0, atol=1e-5)
    # 8-T 위의 Focal(대상 = 8-T 판, 그 판도 다시 붙인 판) → frozen_from 을 따라 원래 값
    v8dir = tmp_path / "s8" / "8-T-XGB"
    v8dir.mkdir(parents=True)
    a8.save(v8dir / "inputs_fold1")
    pf = {**pd8, "from_dir": "s8/8-T-XGB"}
    spf = rn.VariantSpec(**{**d.run[0].to_dict(), "name": "8-F-8-T-XGB",
                            "extra": {**d.run[0].extra, "s4": {**s4.s4_of(d.run[0]), "provider": pf}}})
    vf = w8.make_inputs(spf, "fold1").fit(rows).transform(rows)["anomaly_vae"].to_numpy()
    assert np.allclose(vf, v0, rtol=0, atol=1e-5)
