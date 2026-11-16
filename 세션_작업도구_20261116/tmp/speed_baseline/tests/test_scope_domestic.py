"""국내만 버전(lst.scope) 시험 — 합성 자료.

1. 기본(scope 칸 없음 = 지금 판): 국내만 버전을 넣기 전 코드(사본)로 얻은 지문 GOLDEN 과 바이트 단위로 같음
   (칼럼 목록·입력 행렬·단계 4 입력·runner 로 끝까지 돈 판의 fold·refit 예측과 CV 점수 — tests/_scope_golden.py).
2. 국내만: 해외 정보 칼럼(src_route·src_route01·ov_struct_na)만 빠지고 나머지 칼럼 값은 같은 국내 행에서 지금 판과 같음.
3. 흐름: 국내 시료만으로 runner 판(트리·로지스틱·LSTM) 끝까지 + MLflow experiment '-국내'.
4. 분리: 결과 뿌리 표지·--out 대조, Tracker experiment 이름, 설정 묶음 configs_dom(만들기·읽기).
5. 단계 0 국내 준비(s0_dom_prep)·두 버전 비교(s_dom_rescore).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from lst import config as cf
from lst import evaluate as ev
from lst import folds as fd
from lst import inputs as ip
from lst import registry as rg
from lst import runner as rn
from lst import scope
from lst import wiring as wr
from tests import _scope_golden as gold

CODE = Path(__file__).resolve().parents[1]

# 국내만 버전 코드를 넣기 전 코드(사본)로 tests/_scope_golden.digest_all 을 두 번 돌려 같은 값을 확인한 지문(2026-11-09)
GOLDEN = {
    "dl-o1-None": "dd45b1f3a4da86164501f15c9294bd92",
    "dl-o2-None": "467ba505f9e3363276c1db4bc549d083",
    "dl-o2-seq": "8b773be9743f52a221ffe53bbf25b74b",
    "lims": "7cc41d2ae9ede870ac40da3359955643",
    "membership": "f0cfda03317dbd5913b9264008ecb49d",
    "run-LR-o2": "ec7056b291d1c634ce7ab7d53f5459de",
    "run-LSTM-o1": "49a1da66e9274eaf82f82a2e4dd04edd",
    "run-XGB-o1": "ab92d03ecdf945a6d97ce7c878e0d729",
    "s4-LR": "8fce9c9061aa0f4b7b19d99ab7fb335a",
    "s4-NN60": "21ff467b064e323708a1a72b232a3f09",
    "s4-XGB": "ae230d2d48fe43b695f9412bdb373330",
    "seqD": "559c21411bfc054498ec66e9df3e6b64",
    "seqT": "0bc6ab21d77c6a96bd4c940854b11eb5",
    "tree-CAT-10-sum-o2": "5c1121ccd9af36350af1fd291f8bb726",
    "tree-LR-20-sum-o2": "c8b2aae4753290bf4a369e91fe7e7e57",
    "tree-XGB-20-raw-o2": "65ef0c7b7828518600c90634cadcb589",
    "tree-XGB-20-sum-o1": "0c6c8a7ead439457a69f192d578006f4",
    "v1-10-raw": "92cd9995f4833c707193831f7772f009",
    "v1-10-sum": "c5fb5ed05c67a22eb74eb15904fe594c",
    "v1-100-raw": "7f4d19a4e6a8e90bf963401e2bd17f24",
    "v1-100-sum": "16223d63a4cf2202d5b585d9cb1052ee",
    "v1-60-raw": "287ebb2993aa404952aa9b46b0679bd1",
    "v1-60-sum": "6e9de0da54e1a64cff01af16e085f16a",
    "v2-10-raw": "af751f70d5f4eb3083d5569569afbac6",
    "v2-10-sum": "1ad3e9070d2c5ddc1c8ff5560dea7b1c",
    "v2-100-raw": "aa35a8cab172de1b0f97c8f3c9f5bf52",
    "v2-100-sum": "06eee08bf504fda3f4b3be483bacc371",
    "v2-60-raw": "019f53b1f8d8d0534530849b54ce5eb5",
    "v2-60-sum": "3a8d2ba4473cde9918d8bc2805979527",
}
OV_COLS = ("src_route", "src_route01", rg.STRUCT_COLUMN)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"_tdom_{name}", CODE / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# ================================================================ 1. 기본 = 이전 코드와 바이트 동일
def test_default_bytes_identical_to_code_before_domestic(tmp_path):
    pytest.importorskip("torch")
    assert scope.current() is None and not scope.domestic_only()
    got = gold.digest_all(tmp_path)
    assert got == GOLDEN


def test_default_mode_after_explicit_none_activation(tmp_path):
    """설정을 읽어 scope = None 으로 정한 뒤에도(지금 판 설정) 칼럼·입력 지문이 같음."""
    scope.activate(None)
    got = gold.registry_digest()
    assert got == {k: GOLDEN[k] for k in got}


# ================================================================ 범위 규칙
def test_activation_rules_and_names():
    assert scope.experiment_name("listeria-1차") == "listeria-1차"
    scope.activate(None)
    scope.activate(None)                       # 같은 값 다시 = 그대로
    with pytest.raises(scope.ScopeError, match="바뀜"):
        scope.activate("domestic")
    scope._reset_for_tests()
    with pytest.raises(scope.ScopeError, match="모르는"):
        scope.activate("overseas")
    scope.activate("domestic")
    assert scope.domestic_only() and scope.experiment_name("listeria-2차") == "listeria-2차-국내"
    with pytest.raises(scope.ScopeError):
        scope.with_field({"scope": cf.Field(str)})
    sch = scope.with_field({"plan": cf.Field(str)})
    cf.validate({"plan": "x", "scope": "domestic"}, sch)
    with pytest.raises(cf.ConfigError):
        cf.validate({"plan": "x", "scope": "all"}, sch)


def test_roots_guard(tmp_path):
    full, dom = tmp_path / "results", tmp_path / "results_dom"
    full.mkdir()
    dom.mkdir()
    scope.write_marker(dom, "domestic")
    # 지금 판 설정(None) → 지금 판 뿌리 OK, 국내판 뿌리 멈춤
    scope.apply({}, full, full / "s1" / "1-1")
    with pytest.raises(scope.ScopeError, match="표지"):
        scope.note_roots(dom, dom / "s1")
    scope._reset_for_tests()
    # 국내판 설정 → 지금 판 뿌리 멈춤(쓰기 전 — 인자 먼저 받아 둔 뒤 설정 읽을 때)
    scope.note_roots(full, full / "s1" / "1-1")
    with pytest.raises(scope.ScopeError, match="표지"):
        scope.activate("domestic")
    scope._reset_for_tests()
    # 국내판 --out 은 국내판 뿌리 안
    scope.apply({"scope": "domestic"}, dom, dom / "s1" / "1-1")
    with pytest.raises(scope.ScopeError, match="--out"):
        scope.note_roots(dom, tmp_path / "elsewhere")
    scope.write_marker(dom, "domestic")        # 같은 값이면 그대로
    (dom / scope.MARKER).write_text('{"scope": "zz"}', encoding="utf-8")
    with pytest.raises(scope.ScopeError, match="이상함"):
        scope.read_marker(dom)


def test_parse_paths_notes_roots(tmp_path):
    from lst.paths import build_parser, parse_paths

    dom = tmp_path / "rd"
    dom.mkdir()
    scope.write_marker(dom, "domestic")
    ap = build_parser("x")
    ap.add_argument("--inputs")
    parse_paths(["--root", str(tmp_path), "--out", str(dom / "s0"), "--inputs", str(dom)], ap)
    with pytest.raises(scope.ScopeError):
        scope.activate(None)                   # 지금 판 설정으로 국내판 뿌리 → 멈춤


# ================================================================ 2. 국내만 칼럼
def test_registry_domestic_columns():
    specs = [rg.load_version(s) for s in (
        {"name": "a", "order": 1, "W": 20, "mode": "sum", "include_ids": ["WEATHER"]},
        {"name": "b", "order": 2, "W": 20, "mode": "raw", "include_ids": list(rg.ALL_IDS), "k": 50, "n_ma": 15,
         "n_ma_dl": 7},
        {"name": "c", "order": 2, "W": 20, "mode": "sum", "include_ids": []})]
    base = [(rg.tree_columns(v), rg.dl_seq_channels(v) if v.include_ids else [], rg.dl_fixed_static(v),
             rg.dl_static_columns(v)) for v in specs]
    scope.activate("domestic")
    assert rg.cat_columns() == ("sido", "sgg", "p2", "p3")
    for v, (t0, s0, f0, st0) in zip(specs, base):
        t1 = rg.tree_columns(v)
        assert [c.name for c in t1] == [c.name for c in t0 if c.name != "src_route"]
        assert not any(c.structural for c in t1)
        assert [c.name for c in rg.dl_fixed_static(v)] == ["g_div50"] and [c.name for c in f0] == ["g_div50", "src_route01"]
        if v.include_ids:
            assert [c.name for c in rg.dl_seq_channels(v)] == [c.name for c in s0]
        assert [c.name for c in rg.dl_static_columns(v)] == [c.name for c in st0]
        assert not rg.has_structural(t1 + rg.dl_static_columns(v))
    m = rg.membership_table(20, 50, 15, 7)
    assert not m["structural"].any()


@pytest.fixture(scope="module")
def dom_world():
    """test_dedup 합성 자료의 국내 행만(해외 일자료표는 그대로 — 실제처럼 읽기만 하고 안 씀)."""
    from tests import test_dedup as td

    rng = np.random.default_rng(11)
    dom, ov = td._wx(rng)
    obs, fc, sov = td._spi(rng)
    s = td._samples(rng)
    s = s[s["src_route"] == "domestic"].reset_index(drop=True)
    c, st = gold._world(dom, ov, obs, fc, sov, s, rng)
    return c, st


def _rows(st, split, W):
    return rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS[split]), W)


def _drop(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in OV_COLS if c in df.columns])


def test_tree_inputs_domestic_equal_minus_overseas_columns(dom_world):
    from tests import test_dedup as td

    c, st = dom_world
    for alg, W, mode, order, ver in (("XGB", 20, "sum", "1차", None), ("LR", 20, "sum", "2차", td.ALL2),
                                     ("XGB", 20, "raw", "2차", td.ALL2), ("CAT", 10, "sum", "2차", {"include_ids": []})):
        sp = td.vspec(alg, W, mode, order, ver)
        rows = _rows(st, "fold2", W)
        scope._reset_for_tests()
        a0 = wr.Wiring(c).make_inputs(sp, "fold2").fit(rows)
        x0 = a0.transform(rows)
        scope._reset_for_tests()
        scope.activate("domestic")
        a1 = wr.Wiring(c).make_inputs(sp, "fold2").fit(rows)
        x1 = a1.transform(rows)
        assert not set(OV_COLS) & set(x1.columns) and list(x1.columns) == list(a1.columns)
        assert "src_route" in x0.columns
        if alg == "LR" and ver:
            assert rg.STRUCT_COLUMN in x0.columns and float(x0[rg.STRUCT_COLUMN].sum()) == 0.0
            assert any(k.startswith("soil_t5") and k.endswith(rg.NA_SUFFIX) for k in x1.columns)
        pd.testing.assert_frame_equal(x1, _drop(x0), check_exact=True)


def test_dl_inputs_domestic_equal_minus_overseas_columns(dom_world):
    from tests import test_dedup as td

    c, st = dom_world
    for ver, order in (({"include_ids": ["WEATHER"]}, "1차"), (td.ALL2, "2차"),
                       ({"include_ids": ["WEATHER", "WX1", "WX2"], "n_ma_dl": 7, "dl_layout": "seq_only"}, "2차")):
        sp = td.vspec("LSTM", 20, None, order, ver, kind="dl")
        rows = _rows(st, "fold2", 20)
        scope._reset_for_tests()
        a0 = wr.Wiring(c).make_inputs(sp, "fold2").fit(rows)
        x0 = a0.transform(rows)
        n0 = a0.prep.static_names
        scope._reset_for_tests()
        scope.activate("domestic")
        a1 = wr.Wiring(c).make_inputs(sp, "fold2").fit(rows)
        x1 = a1.transform(rows)
        n1 = a1.prep.static_names
        keep = [i for i, n in enumerate(n0) if n not in OV_COLS]
        assert n1 == [n0[i] for i in keep] and not set(OV_COLS) & set(n1)
        assert np.array_equal(x1["x_static"], x0["x_static"][:, keep])
        assert np.array_equal(x1["x_seq"], x0["x_seq"]) and a1.prep.seq_channel_names == a0.prep.seq_channel_names
        if "x_cat" in x0:
            assert np.array_equal(x1["x_cat"], x0["x_cat"])
        # 저장한 전처리기는 같은 범위에서만 다시 읽힘(국내판 전처리기를 지금 판 코드 경로로 읽으면 멈춤)
        st_ = a1.prep.state()
        assert ip.DLPreprocessor.from_state(st_).static_names == n1
        if order == "2차" and "dl_layout" not in ver:
            scope._reset_for_tests()
            with pytest.raises(ip.InputError):
                ip.DLPreprocessor.from_state(st_)


def test_nn60_and_catenc_domestic(tmp_path):
    from lst import stage4 as s4
    from tests import test_stage4 as t4

    ctx, st = t4._syn_ctx()
    keep = ctx.samples["src_route"] == "domestic"
    ctx.samples = ctx.samples[keep].reset_index(drop=True)
    ctx._pop.clear()
    ctx._vocab = None
    st = ev.SampleTable(st.df[st.df["src_route"] == "domestic"].reset_index(drop=True))
    ver = {"include_ids": ["WEATHER", "HIS1", "WX7", "FD1", "SO1"], "k": 30}
    rows = _rows(st, "fold1", 10)
    nn = rn.VariantSpec(name="x2-NN60", stage="4", order="2차", kind="dl", algorithm="NN60", W=10, mode="sum",
                        input_set="x", role="교체 후보", train_count=4, extra={"version": ver, "s4": {}})
    tr = rn.VariantSpec(name="4-a0-XGB", stage="4", order="2차", kind="tree", algorithm="XGB", W=10, mode="sum",
                        input_set="x", role="후보", train_count=4,
                        extra={"version": ver, "s4": {"catenc_k": 30, "row_plan": s4.RowPlan().to_dict()}})
    out = {}
    for mode in (None, "domestic"):
        scope._reset_for_tests()
        scope.activate(mode)
        w = s4.Stage4Wiring(ctx, st, tmp_path / str(mode), n_jobs=1, device="cpu")
        an = w.make_inputs(nn, "fold1").fit(rows)
        at = w.make_inputs(tr, "fold1").fit(rows)
        out[mode] = (an.columns, an.transform(rows), at.columns, at.transform(rows))
    c0, x0, tc0, t0 = out[None]
    c1, x1, tc1, t1 = out["domestic"]
    tab0 = [n for n in c0 if not n.startswith("x_cat:") and n != "x_h"]
    keep = [i for i, n in enumerate(tab0) if n not in OV_COLS]
    assert [n for n in c1 if not n.startswith("x_cat:") and n != "x_h"] == [tab0[i] for i in keep]
    assert "src_route01" in c0 and "src_route01" not in c1
    assert np.array_equal(x1["x_tab"], x0["x_tab"][:, keep]) and np.array_equal(x1["x_cat"], x0["x_cat"])
    assert "catenc_src_route" in tc0 and "catenc_src_route" not in tc1
    pd.testing.assert_frame_equal(t1, t0.drop(columns=["catenc_src_route"]), check_exact=True)


# ================================================================ 3. 국내만 흐름(runner + MLflow)
def test_domestic_run_flow_and_mlflow(dom_world, tmp_path):
    pytest.importorskip("torch")
    from lst import tracking as tk
    from tests import test_dedup as td

    c, st = dom_world
    scope.activate("domestic")
    tr = tk.Tracker(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", artifact_root=(tmp_path / "art").as_uri())
    w = wr.Wiring(c, dl_kwargs={"max_epochs": 1}, device="cpu")
    log = rn.ExperimentLog(tmp_path / "logs" / "experiment_log.jsonl")
    for alg, W, mode, order, ver, kind, exp in (("XGB", 10, "sum", "1차", None, "tree", "listeria-1차"),
                                                 ("LR", 10, "sum", "2차", td.ALL2, "tree", "listeria-2차"),
                                                 ("LSTM", 10, None, "1차", None, "dl", "listeria-1차")):
        sp = td.vspec(alg, W, mode, order, ver, kind=kind)
        sp = rn.VariantSpec(**{**sp.to_dict(), "name": f"d-{alg}", "stage": "1-1"})
        od = tmp_path / "rd" / f"run_{alg}"
        ctx = rn.RunContext(samples=st, out_dir=od, code_root=CODE, input_md5s={"x": "0"}, exp_log=log,
                            make_inputs=w.make_inputs, make_model=w.make_model, fit_kwargs=w.fit_kwargs, n_jobs=2,
                            tracker=tr, experiment=exp, stage_run_id=tr.stage_run(exp, "1-1"),
                            check_resources=lambda p: {"note": "test"}, require_7a_logged=False,
                            smoke={"max_train_rows": 400, "max_pairs_per_h": 25})
        rn.run_variant(sp, ctx)
        vd = rn.variant_dir(od, sp.name)
        keys = set(pd.read_parquet(rn.pred_path(vd, "refit"))["lims_data_key"])
        assert keys <= set(st.keys)
        for f in ev.CV_FOLDS:
            assert set(pd.read_parquet(rn.pred_path(vd, f))["lims_data_key"]) <= set(st.keys)
        assert (vd / "cv_scores.json").exists()
    names = {e.name for e in tr.client.search_experiments()}
    assert {"listeria-1차-국내", "listeria-2차-국내"} <= names and not names & {"listeria-1차", "listeria-2차"}


def test_tracker_experiments_separate(tmp_path):
    from lst import tracking as tk

    uri = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    art = (tmp_path / "art").as_uri()
    t0 = tk.Tracker(uri, artifact_root=art)
    r_full = t0.stage_run("listeria-1차", "1-1")
    c_full = t0.chain_run()
    scope.activate("domestic")
    t1 = tk.Tracker(uri, artifact_root=art)
    r_dom = t1.stage_run("listeria-1차", "1-1")
    assert r_dom != r_full and t1.find_run("listeria-1차", "1-1", None).info.run_id == r_dom
    assert t1.chain_run() != c_full
    e = t1.client.get_run(r_dom).info.experiment_id
    assert t1.client.get_experiment(e).name == "listeria-1차-국내"
    assert t1.client.get_run(r_dom).data.tags["차수"] == "1차"
    with pytest.raises(tk.TrackingError):
        t1.experiment_id("listeria-1차-국내")   # 논리 이름만 받음(실제 이름으로 직접 못 씀)


# ================================================================ 4. 설정 묶음
def test_make_configs_dom_rules():
    mk = load_script("make_configs_dom")
    built = mk.build()
    src = {p.name for p in (CODE / "configs").glob("*.json")}
    assert set(built) == (src - set(mk.EXCLUDE)) | {"s0_dom_prep.json", "s_dom_rescore.json"}
    for n, c in built.items():
        assert c["scope"] == "domestic", n
        assert not set(mk.DROP_KEYS) & set(c), n
        if "code_reviewed" in c:
            assert c["code_reviewed"] is False
        txt = json.dumps(c, ensure_ascii=False)
        assert '"configs/' not in txt, n
        if n in src:
            orig = json.loads((CODE / "configs" / n).read_text(encoding="utf-8"))
            same = {k for k in orig if k not in ("plan", "code_reviewed", *mk.DROP_KEYS)}
            for k in same:
                if k == "search_dirs":
                    assert c[k] == [d for d in orig[k] if d != "s3/3-0b"]
                elif not json.dumps(orig[k]).count('"configs/'):
                    assert c[k] == orig[k], (n, k)
    assert built["s1_3_decide.json"]["config_1_1"] == "configs_dom/s1_1_trees.json"
    assert built["s2_select.json"]["configs"]["2-R"] == "configs_dom/s2_R_research.json"


def test_configs_dom_up_to_date_and_loadable():
    mk = load_script("make_configs_dom")
    assert mk.check() == [], "configs_dom 이 configs 와 다름 — scripts/make_configs_dom.py 를 다시 돌릴 것"
    from lst import stage_cli as sc

    D = CODE / "configs_dom"
    m = {n: load_script(n) for n in ("s0_6_missing", "s0_6_history", "s0_8_groups", "s0_10_te", "s0_10_log_mlflow",
                                     "s1_1_trees", "s1_2_lstm", "s1_3_decide", "s_x3", "s2_prep", "s3_0", "s3_0b",
                                     "s3_1", "s3_2", "s3_3", "s3_extra", "s4", "s5_6", "s8", "s8_d", "s_final",
                                     "s0_dom_prep", "s_dom_rescore")}

    def load(name):
        p = D / name
        raw = json.loads(p.read_text(encoding="utf-8"))
        stem = name[:-5]
        sub = raw.get("sub")
        if stem in ("s0_6_missing", "s0_6_history", "s0_8_groups", "s0_10_te", "s_x3", "s3_0b", "s3_1", "s3_2",
                    "s3_3", "s_final", "s0_dom_prep", "s_dom_rescore"):
            return m[stem].load_cfg(p)
        if stem == "s0_10_log_mlflow":
            return cf.load_config(p, m[stem].SCHEMA)
        if stem in ("s1_1_trees", "s1_2_lstm"):
            return sc.load_cfg(p, m[stem].SCHEMA_EXTRA)
        if stem == "s1_3_decide":
            return cf.load_config(p, m[stem].SCHEMA)
        if stem.startswith("s2_"):
            mod = m["s2_prep"]
            sch = {"2-0": mod.BASE_SCHEMA, "select": mod.SELECT_SCHEMA}.get(sub, mod.TRAIN_SCHEMA)
            return sc.load_cfg(p, sch) if sch is mod.TRAIN_SCHEMA else cf.load_config(p, sch)
        if stem.startswith("s3_0_"):
            return sc.load_cfg(p, m["s3_0"].SCHEMA)
        if stem.startswith("s3_extra"):
            mod = m["s3_extra"]
            return cf.load_config(p, mod.JUDGE_SCHEMA) if sub == "judge" else sc.load_cfg(p, mod.TRAIN_SCHEMA)
        if stem.startswith("s4_") or stem.startswith("s5_6"):
            mod = m["s4" if stem.startswith("s4_") else "s5_6"]
            if sub in ("prep", "judge"):
                return cf.load_config(p, mod.PREP_SCHEMA if sub == "prep" else mod.JUDGE_SCHEMA)
            return sc.load_cfg(p, mod.TRAIN_SCHEMA)
        if stem.startswith("s8_D"):
            mod = m["s8_d"]
            return cf.load_config(p, mod.JUDGE_SCHEMA) if sub == "judge" else mod.load_cfg(p)
        if stem.startswith("s8_"):
            mod = m["s8"]
            return cf.load_config(p, mod.JUDGE_SCHEMA) if sub in ("judgeT", "judgeF") else mod.load_cfg(p)
        raise AssertionError(f"읽는 법 모름: {name}")

    names = sorted(p.name for p in D.glob("*.json"))
    assert len(names) == len(mk.build())
    for n in names:
        scope._reset_for_tests()
        cfg = load(n)
        assert cfg["scope"] == "domestic", n


# ================================================================ 5. 단계 0 국내 준비·두 버전 비교
def _full_samples(n=3000, seed=5) -> pd.DataFrame:
    from lst import samples as S
    from tests.test_evaluate import make_samples

    s = make_samples(n=n, seed=seed)
    ov = s["src_route"] == "overseas"
    s["p3"] = np.where(s["p3"] == "C01", S.NO_CODE, s["p3"])
    s["nation_a3"] = np.where(ov, s["sgg"].str[2:], None)
    s["nation_src"] = np.where(ov, "mnf_nation_cd", None)
    s["rr1_moved"] = False
    s["rr3_p2_nocode"] = False
    s["rr3_p3_nocode"] = s["p3"] == S.NO_CODE
    s["rr5_coord_differs"] = None
    s["rr6_overseas_only_p3"] = False
    s["rr7_jul01"] = False
    return s[list(S.SAMPLE_COLUMNS)]


@pytest.fixture()
def two_roots(tmp_path, monkeypatch):
    from lst import samples as S
    from lst import static_feats as sf

    full, dom = tmp_path / "results", tmp_path / "results_dom"
    a = full / "s0" / "0-1" / "apply"
    a.mkdir(parents=True)
    s = _full_samples()
    s.to_parquet(a / "samples.parquet", index=False)
    _, inc = S.incomplete_months(s)
    (a / "12_incomplete_months.json").write_text(json.dumps({"incomplete": inc}), encoding="utf-8")
    (a / "14_theil_target_columns.json").write_text(json.dumps(
        {"columns": ["hazard_label", "src_route", "prduct_nm"], "note": "합성"}, ensure_ascii=False), encoding="utf-8")
    for d in ("0-4_v2", "0-4_spi", "0-4_weather_overseas", "0-6_static"):
        (full / "s0" / d).mkdir(parents=True)
        (full / "s0" / d / "x.txt").write_text(d, encoding="utf-8")
    real = s["p3"] != S.NO_CODE
    pair = sf.pair_p2(s.loc[real, "p3"].to_numpy(dtype=object), s.loc[real, "p2"].to_numpy(dtype=object))
    pd.DataFrame({"p3": list(pair), "p2_used": list(pair.values())}).to_parquet(
        full / "s0" / "0-6_static" / "식품_고정표.parquet", index=False)
    dom_n = int((s["src_route"] == "domestic").sum())
    dom_pos = int(s.loc[s["src_route"] == "domestic", "y"].sum())
    mk = load_script("make_configs_dom")
    cfg = {**mk.PREP_CFG, "expected": {"rows": dom_n, "pos": dom_pos}}
    cp = tmp_path / "prep.json"
    cp.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    prep = load_script("s0_dom_prep")
    monkeypatch.setattr(prep, "check_resources", lambda path: {"note": "test stub"})   # 로컬 PC RAM 바닥값 대신
    if os.name == "nt":                       # 로컬 Windows 는 심볼릭 링크 권한이 없음 → 디렉터리 정션(같은 뜻)
        import _winapi

        monkeypatch.setattr(prep, "_make_link", lambda src, dst: _winapi.CreateJunction(str(src), str(dst)))
        monkeypatch.setattr(prep, "_is_link", lambda p: os.path.islink(p) or os.path.isjunction(p))
    return {"full": full, "dom": dom, "s": s, "cfg": cp, "prep": prep, "tmp": tmp_path}


def _prep_argv(r):
    return ["--root", str(r["tmp"]), "--inputs", str(r["dom"]), "--source", str(r["full"]),
            "--out", str(r["dom"] / "s0" / "0-1"), "--config", str(r["cfg"])]


def test_s0_dom_prep(two_roots):
    r = two_roots
    out = r["prep"].main(_prep_argv(r))
    assert scope.read_marker(r["dom"]) == "domestic" and scope.read_marker(r["full"]) is None
    got = pq.read_table(out / "samples.parquet")
    src = pq.read_table(r["full"] / "s0" / "0-1" / "apply" / "samples.parquet")
    want = src.filter(pc.equal(src["src_route"], "domestic"))
    assert got.equals(want) and got.schema.names == src.schema.names
    th = json.loads((out / "14_theil_target_columns.json").read_text(encoding="utf-8"))
    assert th["columns"] == ["hazard_label", "prduct_nm"]
    inc = json.loads((out / "12_incomplete_months.json").read_text(encoding="utf-8"))["incomplete"]
    assert all(rr == "domestic" for rr, _ in inc)
    for d in ("0-4_v2", "0-4_spi", "0-4_weather_overseas", "0-6_static"):
        p = r["dom"] / "s0" / d
        assert p.resolve() == (r["full"] / "s0" / d).resolve() and (p / "x.txt").exists()
    sc_ = pd.read_csv(out / "15_domestic_split_counts.csv", encoding="utf-8-sig")
    assert set(sc_["split"]) == {"fold1", "fold2", "fold3", "refit"}
    assert (sc_["score_pos_overseas"] == 0).all()
    meta = json.loads((r["dom"] / "s0" / "0-1" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["config"]["scope"] == "domestic" and meta["rows"]["overseas_dropped"] > 0
    with pytest.raises(FileExistsError):       # 덮어쓰기 금지
        scope._reset_for_tests()
        r["prep"].main(_prep_argv(r))


def test_s0_dom_prep_stops(two_roots):
    r = two_roots
    bad = json.loads(r["cfg"].read_text(encoding="utf-8"))
    bad["expected"]["rows"] += 1
    r["cfg"].write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(r["prep"].PrepStop, match="기록"):
        r["prep"].main(_prep_argv(r))
    scope._reset_for_tests()
    with pytest.raises(r["prep"].PrepStop, match="서로 안"):
        r["prep"].main(["--root", str(r["tmp"]), "--inputs", str(r["full"] / "x"), "--source", str(r["full"]),
                        "--out", str(r["full"] / "x" / "s0"), "--config", str(r["cfg"])])


def test_s0_dom_prep_food_pairing_stop(two_roots):
    r = two_roots
    good = json.loads(r["cfg"].read_text(encoding="utf-8"))
    fp = r["full"] / "s0" / "0-6_static" / "식품_고정표.parquet"
    t = pd.read_parquet(fp)
    t.loc[0, "p2_used"] = "ZZ"
    t.to_parquet(fp, index=False)
    r["cfg"].write_text(json.dumps(good, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(r["prep"].PrepStop, match="식품 고정표"):
        r["prep"].main(_prep_argv(r))


def test_s_dom_rescore(two_roots):
    """지금 판 판 하나(가짜 입력·가짜 모델, 국내 + 해외)를 국내 시료로만 다시 채점 = 국내 쌍만 남긴 예측의 직접 채점."""
    from tests.test_runner import STUB, FakeAssembler, FakeModel, spec_of

    r = two_roots
    r["prep"].main(_prep_argv(r))
    scope._reset_for_tests()
    full_samples = ev.SampleTable.read(r["full"] / "s0" / "0-1" / "apply" / "samples.parquet")
    od = r["full"] / "s1" / "1-1"
    sp = spec_of()
    ctx = rn.RunContext(samples=full_samples, out_dir=od, code_root=CODE, input_md5s={"x": "0"},
                        exp_log=rn.ExperimentLog(r["full"] / "logs" / "experiment_log.jsonl"),
                        make_inputs=FakeAssembler, make_model=FakeModel, n_jobs=8, check_resources=STUB,
                        require_7a_logged=False)
    rn.run_variant(sp, ctx)
    scope._reset_for_tests()
    mk = load_script("make_configs_dom")
    cfg = {**mk.RESCORE_CFG, "stage_dirs": ["s1/1-1"]}
    cp = r["tmp"] / "rescore.json"
    cp.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    rs = load_script("s_dom_rescore")
    rs.check_resources = lambda path: {"note": "test stub"}
    out = r["dom"] / "compare" / "full_on_domestic"
    argv = ["--root", str(r["tmp"]), "--inputs", str(r["dom"]), "--source", str(r["full"]), "--out", str(out),
            "--config", str(cp)]
    rs.main(argv)
    dom_st = ev.SampleTable.read(r["dom"] / "s0" / "0-1" / "apply" / "samples.parquet")
    vd = rn.variant_dir(od, sp.name)
    preds = {f: rs.keep_domestic(p, dom_st) for f, p in rn.load_preds(vd, ev.CV_FOLDS).items()}
    direct = ev.score_cv(preds, dom_st)
    t = pd.read_csv(out / "compare_cv.csv", encoding="utf-8-sig")
    row = t[t["판"] == sp.name].iloc[0]
    js0 = [x for x in json.loads((out / "compare_scores.json").read_text(encoding="utf-8")) if x["name"] == sp.name][0]
    assert js0["cv"]["cv_band_weighted"] == direct["cv_band_weighted"]          # 같은 함수·같은 쌍 = 같은 값
    assert row["국내 다시 채점 CV"] == pytest.approx(direct["cv_band_weighted"], rel=1e-12)
    assert direct["n_pairs"] == sum(dom_st.expected_pairs(f) for f in ev.CV_FOLDS)
    full_cv = rn._load_scores(vd / "cv_scores.json")["cv_band_weighted"]
    assert js0["cv_full"] == full_cv and js0["cv_full"] != js0["cv"]["cv_band_weighted"]
    sk = json.loads((out / "skipped.json").read_text(encoding="utf-8"))
    assert any("7-a" in str(x) for x in sk)                       # 합성 지금 판에 0-10 없음 → 기록만
    assert "국내 test 구간 가중" not in t.columns                   # 보고 허가 없으면 test 안 읽음
    scope._reset_for_tests()
    rs.main(argv + ["--report", "--reason", "시험"])
    tr = pd.read_csv(out / "compare_cv_report.csv", encoding="utf-8-sig")
    assert np.isfinite(tr.loc[tr["판"] == sp.name, "국내 test 구간 가중"].iloc[0])
    js = json.loads((out / "compare_scores_report.json").read_text(encoding="utf-8"))
    one = [x for x in js if x["name"] == sp.name][0]
    assert one["test_threshold_full"]["report"]["threshold_metrics"]["threshold"] == pytest.approx(one["threshold_full"])
    scope._reset_for_tests()
    with pytest.raises(FileExistsError):
        rs.main(argv)
    scope._reset_for_tests()
    with pytest.raises(scope.ScopeError):     # 국내판 설정 → 지금 판 뿌리에 쓰기 금지
        rs.main(["--root", str(r["tmp"]), "--inputs", str(r["full"]), "--source", str(r["full"]),
                 "--out", str(r["full"] / "cmp"), "--config", str(cp)])


def test_script_flow_s0_10_domestic(two_roots, monkeypatch):
    """국내판 단계 0 흐름(스크립트 그대로): s0_dom_prep → s0_10_te → s0_10_log_mlflow(configs_dom 설정, MLflow 만 로컬
    sqlite) — 7-a 가 국내 시료만으로, MLflow 는 listeria-준비-국내, 실험 로그는 국내판 뿌리에. 지금 판 설정으로 국내판
    뿌리를 돌리면 멈춤."""
    from lst import tracking as tk

    r = two_roots
    r["prep"].main(_prep_argv(r))
    D = CODE / "configs_dom"
    te = load_script("s0_10_te")
    monkeypatch.setattr(te, "check_resources", lambda path: {"note": "test stub"})   # 로컬 PC RAM 바닥값 대신
    scope._reset_for_tests()
    with pytest.raises(scope.ScopeError):                     # 기본 설정(configs/) → 국내판 뿌리 멈춤
        te.main(["--root", str(r["tmp"]), "--inputs", str(r["dom"]), "--out", str(r["dom"] / "s0" / "0-10x")])
    scope._reset_for_tests()
    te.main(["--root", str(r["tmp"]), "--inputs", str(r["dom"]), "--out", str(r["dom"] / "s0" / "0-10"),
             "--config", str(D / "s0_10_te.json")])
    sel = json.loads((r["dom"] / "s0" / "0-10" / "te7a_selected.json").read_text(encoding="utf-8"))
    cvp = pd.read_parquet(r["dom"] / "s0" / "0-10" / f"te7a_cv_pred_{sel['selected']}.parquet")
    dom_keys = set(r["s"].loc[r["s"]["src_route"] == "domestic", "lims_data_key"])
    assert set(cvp["lims_data_key"]) <= dom_keys
    lc = json.loads((D / "s0_10_log_mlflow.json").read_text(encoding="utf-8"))
    lc["mlflow_uri"] = f"sqlite:///{(r['tmp'] / 'm.db').as_posix()}"
    cp = r["tmp"] / "log10.json"
    cp.write_text(json.dumps(lc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.chdir(r["tmp"])                                # sqlite 기본 artifact 자리(./mlruns)를 임시 폴더로
    scope._reset_for_tests()
    lm = load_script("s0_10_log_mlflow")
    lm.main(["--root", str(r["tmp"]), "--inputs", str(r["dom"]), "--out", str(r["dom"] / "s0" / "0-10_mlflow"),
             "--config", str(cp)])
    assert (r["dom"] / "logs" / "experiment_log.jsonl").exists() and not (r["full"] / "logs").exists()
    t = tk.Tracker(lc["mlflow_uri"])
    names = {e.name for e in t.client.search_experiments()}
    assert "listeria-준비-국내" in names and "listeria-준비" not in names


def test_dom_final_cfg(tmp_path):
    """최종 단계 설정 채우기: dir_2_0 = 국내판 1-3 의 A, x1_dir = 국내판 4-X1 의 끝난 판 하나."""
    fc = load_script("dom_final_cfg")
    rd = tmp_path / "rd"
    rd.mkdir()
    scope.write_marker(rd, "domestic")
    (rd / "s1" / "1-3").mkdir(parents=True)
    (rd / "s1" / "1-3" / "1-3_decision.json").write_text(json.dumps({"A": "XGB"}), encoding="utf-8")
    (rd / "s2" / "2-0" / "2-0-XGB").mkdir(parents=True)
    (rd / "s2" / "2-0" / "2-0-XGB" / "done.json").write_text("{}", encoding="utf-8")
    argv = ["--inputs", str(rd), "--config", str(CODE / "configs_dom" / "s_final.json"), "--write", "logs/f.json"]
    p = fc.main(argv)
    got = json.loads(p.read_text(encoding="utf-8"))
    assert got["dir_2_0"] == "s2/2-0/2-0-XGB" and "x1_dir" not in got and got["scope"] == "domestic"
    for n in ("x1-LGB", "x1-T3"):
        (rd / "s4" / "4-X1" / n).mkdir(parents=True)
    (rd / "s4" / "4-X1" / "x1-LGB" / "done.json").write_text("{}", encoding="utf-8")
    scope._reset_for_tests()
    with pytest.raises(FileExistsError):                   # 이미 있고 내용이 다름(x1_dir 생김)
        fc.main(argv)
    scope._reset_for_tests()
    got2 = json.loads(fc.main(argv[:-1] + ["logs/g.json"]).read_text(encoding="utf-8"))
    assert got2["x1_dir"] == "s4/4-X1/x1-LGB"
    (rd / "s4" / "4-X1" / "x1-T3" / "done.json").write_text("{}", encoding="utf-8")
    scope._reset_for_tests()
    with pytest.raises(RuntimeError, match="여럿"):
        fc.main(argv[:-1] + ["logs/h.json"])
    scope._reset_for_tests()
    with pytest.raises(scope.ScopeError):                  # 지금 판 뿌리(표지 없음)에선 멈춤
        fc.main(["--inputs", str(tmp_path), "--config", str(CODE / "configs_dom" / "s_final.json"), "--write", "x.json"])
