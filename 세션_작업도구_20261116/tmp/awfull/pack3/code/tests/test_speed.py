"""속도 개선 스위치(lst/speed — 사용자 승인 11-12) 켬·끔 결과가 바이트 단위로 같은지(합성 자료).

- dl_keyed_inputs: 딥러닝 입력(train·평가 쌍, pair_cache 안·밖, 묶음 경계)·트리 입력(트리·로지스틱) 바이트 같음.
- fast_predict: DLClassifier.predict_proba·AnomalyModel.raw_score 바이트 같음(CPU, CUDA 가 있으면 CUDA 도).
- gpu_importance: 순열 중요도(딥러닝 잎·이상 점수 잎·트리와 평균) records 바이트 같음 + 섞은 예측 바이트 같음.
- parallel_groups: 묶음 동시 실행 차례(사슬 판 차례)·한 번씩·멈춤 처리, 설정 점검, 코어 묶음.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import config as cf
from lst import folds as fd
from lst import inputs as ip
from lst import registry as rg
from lst import runner as rn
from lst import speed
from lst import wiring as wr
from tests.test_dedup import ALL2, CASES, _bytes, _pair_inputs, vspec, world  # noqa: F401

ON = {"dl_keyed_inputs": True, "fast_predict": True, "gpu_importance": True}


def _devices():
    import torch

    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------- 입력
@pytest.mark.parametrize("alg,W,mode,order,ver", CASES)
@pytest.mark.parametrize("dedup", [True, False])
def test_tree_inputs_speed_same_bytes(world, monkeypatch, alg, W, mode, order, ver, dedup):
    c, st = world
    monkeypatch.setattr(ip, "DEDUP", dedup)
    sp = vspec(alg, W, mode, order, ver)
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), W)
    res = {}
    for on in (False, True):
        speed.configure(ON if on else None)
        a = wr.Wiring(c).make_inputs(sp, "fold1").fit(rows)
        res[on] = (_bytes(a.transform(rows)), [_bytes(x) for x in _pair_inputs(a, st, "fold1", W)],
                   [_bytes(x) for x in _pair_inputs(a, st, "fold1", W, scope=False)])
    speed.configure(None)
    assert res[True][0] == res[False][0]
    assert len(res[True][1]) > 10
    assert res[True][1] == res[False][1] and res[True][2] == res[False][2]


def test_dl_inputs_speed_same_bytes(world, monkeypatch):
    c, st = world
    for ver, W in (({"include_ids": ["WEATHER"]}, 20), ({**ALL2}, 20), ({**ALL2}, 30)):
        sp = vspec("LSTM", W, None, "1차" if len(ver) == 1 else "2차", ver, kind="dl")
        rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold2"]), W)
        res = {}
        for on in (False, True):
            speed.configure(ON if on else None)
            a = wr.Wiring(c).make_inputs(sp, "fold2").fit(rows)
            a.chunk_rows = 97                                  # 묶음 경계를 넘는 캐시·키 표도 시험
            res[on] = (a.prep.state(), _bytes(a.transform(rows)),
                       [_bytes(x) for x in _pair_inputs(a, st, "fold2", W)],
                       [_bytes(x) for x in _pair_inputs(a, st, "fold2", W, scope=False)])
        speed.configure(None)
        assert res[True][0] == res[False][0]                   # 전처리기 통계(행마다 fit) 같음
        assert res[True][1] == res[False][1]
        assert res[True][2] == res[False][2] and res[True][3] == res[False][3]


def test_dl_keyed_raw_really_keyed(world):
    """키 표가 실제로 행보다 작음(쌍 입력에서 (sgg, E, T, src_route) 고유값으로)."""
    c, st = world
    sp = vspec("LSTM", 20, None, "2차", ALL2, kind="dl")
    spec = wr.Wiring(c).make_inputs(sp, "fold1").spec
    meta = st.meta()
    m = rn.add_bounds(meta.assign(h=np.int16(10)), 20)
    raw = ip.build_dl_raw(m, c, spec, keyed_seq=True)
    full = ip.build_dl_raw(m, c, spec)
    assert raw.seq_index is not None and raw.seq.shape[0] < full.seq.shape[0]
    assert np.array_equal(raw.seq[raw.seq_index], full.seq, equal_nan=True)
    assert raw.n == full.n and np.array_equal(raw.static, full.static, equal_nan=True)
    prep = ip.DLPreprocessor(spec, c.missing, c.vocab)
    with pytest.raises(ip.InputError):
        prep.partial_fit(raw)                                  # fit 은 행마다만


def test_dl_dedup_off_falls_back(world, monkeypatch):
    c, st = world
    monkeypatch.setattr(ip, "DEDUP", False)
    sp = vspec("LSTM", 20, None, "2차", ALL2, kind="dl")
    spec = wr.Wiring(c).make_inputs(sp, "fold1").spec
    m = rn.add_bounds(st.meta().assign(h=np.int16(5)).iloc[:300], 20)
    raw = ip.build_dl_raw(m, c, spec, keyed_seq=True)
    assert raw.seq_index is None and raw.seq.shape[0] == 300


# ---------------------------------------------------------------- 예측(fast_predict)
def _small_dl(world, device, arch="LSTM"):
    from lst import models_dl as md

    c, st = world
    sp = vspec(arch, 20, None, "2차", ALL2, kind="dl")
    w = wr.Wiring(c, dl_kwargs={"max_epochs": 1}, device=device)
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), 20)
    asm = w.make_inputs(sp, "fold1").fit(rows)
    X = asm.transform(rows)
    model = w.make_model(sp, asm).fit(X, rows["y"].to_numpy(np.int64), seed=42, n_jobs=2,
                                      **w.fit_kwargs(sp, "fold1", rows, X))
    an = md.AnomalyModel("VAE", "seq", device=device, max_epochs=1)
    an.fit({"x_seq": X["x_seq"]}, rows["y"].to_numpy(np.int64), seed=42, n_jobs=2,
           groups=(rows["sgg"].to_numpy(), fd.day_ints(rows["d"])))
    return asm, model, an, X


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fast_predict_same_bytes(world, monkeypatch, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from lst import models_dl as md

    monkeypatch.setattr(md, "PRED_BATCH", 300)                 # 묶음 여러 개(마지막 묶음 작게)
    asm, model, an, X = _small_dl(world, device)
    out = {}
    for on in (False, True):
        speed.configure({"fast_predict": on})
        out[on] = (model.predict_proba(X).tobytes(), an.raw_score({"x_seq": X["x_seq"]}).tobytes(),
                   an.score({"x_seq": X["x_seq"]}).tobytes())
    speed.configure(None)
    assert out[True] == out[False]


# ---------------------------------------------------------------- 순열 중요도(gpu_importance)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gpu_importance_same_bytes(monkeypatch, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    from lst import evaluate as ev
    from lst import importance as im
    from lst import models_dl as md
    from lst import models_tree as mtr
    from tests.test_importance import _registry_ctx

    monkeypatch.setattr(md, "PRED_BATCH", 256)                 # 이어 붙인 R × n 행이 묶음 여러 개
    ctx = _registry_ctx()
    spec = rg.VersionSpec("t", 2, 20, "sum", include_ids=rg.ALL_IDS, k=50, n_ma=15, n_ma_dl=7)
    split = fd.Split("fold1", fd.day("2016-06-30"), fd.day("2016-07-01"), fd.day("2017-05-01"))
    samples = ev.SampleTable(ctx.samples)
    rows = rn.add_bounds(fd.make_train_rows(samples.known_frame(), split).reset_index(drop=True), None)
    y = rows["y"].to_numpy(np.int64)
    targets = [im.group_target("기온"), im.group_target("계절"), im.unit_target("food_aw"),
               im.research_column_target("td1")]
    tasm = ip.TreeInputs(ctx, spec, split).fit(rows)
    Xt = tasm.transform(rows)
    tree = mtr.make_model("XGB", cat_cols=[c for c in Xt.columns if Xt[c].dtype == object], max_iter=20)
    tree.fit(Xt, y, seed=42, n_jobs=2)
    dasm = ip.DLInputs(ctx, spec, split).fit(rows)
    Xd = dasm.transform(rows)
    groups = (rows["sgg"].to_numpy(), fd.day_ints(rows["d"]))
    net = md.DLClassifier("LSTM", dasm.cat_sizes, device=device, max_epochs=1)
    net.fit(Xd, y, seed=42, n_jobs=1, groups=groups)
    an = md.AnomalyModel("AE", "seq", device=device, max_epochs=1)
    an.fit(Xd, y, seed=42, n_jobs=1, groups=groups)
    an_lay = im.layout_from_names({"x_seq": dasm.prep.seq_channel_names},
                                  im.column_lookup(list(dasm.prep.seq_cols) + list(dasm.prep.st_cols)))
    nodes = {
        "dl": im.Leaf("LSTM", dasm, net),
        "mean": im.Mean("ens", [im.Leaf("XGB", tasm, tree), im.Leaf("LSTM2", dasm, net),
                                im.Leaf("AE", dasm, an, output="score", layout=an_lay)]),
    }
    used = []
    real = im._GPUInput.predict

    def spy(self, leaf, sel, bt, st_):
        p = real(self, leaf, sel, bt, st_)
        used.append(leaf.name)
        return p
    monkeypatch.setattr(im._GPUInput, "predict", spy)
    for key, node in nodes.items():
        res = {}
        for on in (False, True):
            speed.configure({"gpu_importance": on, "fast_predict": on})
            r = im.permutation_importance({"fold1": node}, samples, targets, M=3, splits={"fold1": split}, r_batch=2)
            res[on] = r
        speed.configure(None)
        a, b = res[False], res[True]
        assert a.records.drop(columns=[]).to_numpy().tolist() == b.records.to_numpy().tolist()
        for c in ("score_base", "score_perm", "I"):
            assert a.records[c].to_numpy().tobytes() == b.records[c].to_numpy().tobytes()
        assert a.base == b.base
        assert np.array_equal(a.base_by_h["fold1"], b.base_by_h["fold1"], equal_nan=True)
    assert {"LSTM", "LSTM2", "AE"} <= set(used) and "XGB" not in used


def test_gpu_importance_batches_match_cpu_take(monkeypatch):
    """GPU 색인으로 만든 묶음 = 예전 _take_dict 로 만든 이어 붙인 입력의 같은 구간(바이트)."""
    torch = pytest.importorskip("torch")
    from lst import importance as im

    rng = np.random.default_rng(3)
    n, R = 50, 3
    X = {"x_seq": rng.normal(size=(n, 7, 5)).astype(np.float32), "x_static": rng.normal(size=(n, 4)).astype(np.float32),
         "x_cat": rng.integers(0, 3, size=(n, 4)).astype(np.int64)}
    sel = {"x_seq": np.array([1, 3]), "x_static": np.array([0])}
    bt = np.tile(np.arange(n), R)
    stake = np.concatenate([rng.permutation(n) for _ in range(R)])
    ref = im._take_dict(X, bt, sel, stake)

    class M:
        device = torch.device("cpu")

        def check_source_inputs(self, X):
            return n

        def source_keys(self):
            return ["x_seq", "x_static", "x_cat"]

    leaf = im.Leaf("L", None, M(), layout=im.Layout("dict", {}))
    g = im._GPUInput(leaf, X)
    fetch = g.fetch_fn(sel, bt, stake)
    for s, e in ((0, 40), (40, 120), (120, 150)):
        b = fetch(s, e)
        for k in X:
            assert b[k].numpy().tobytes() == ref[k][s:e].tobytes()


# ---------------------------------------------------------------- 설정·동시 실행
def test_speed_config_check():
    assert speed.check(None)["parallel_groups"] == 1 and not speed.check({})["gpu_importance"]
    with pytest.raises(cf.ConfigError):
        speed.check({"bogus": True})
    with pytest.raises(cf.ConfigError):
        speed.check({"parallel_groups": 2})                    # cpu_cores 필요
    with pytest.raises(cf.ConfigError):
        speed.check({"parallel_groups": 2, "cpu_cores": 13})
    with pytest.raises(cf.ConfigError):
        speed.check({"gpu_fraction_cap": 0.5})
    with pytest.raises(cf.ConfigError):
        speed.check({"dl_keyed_inputs": 1})
    st = speed.configure({"parallel_groups": 3, "cpu_cores": 12, "gpu_fraction_cap": 0.25})
    from lst import resources as rs

    assert st["parallel_groups"] == 3 and rs.gpu_cap() == 0.25
    speed.reset_for_tests()
    assert rs.gpu_cap() is None


def test_core_slots(monkeypatch):
    import os

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    s = speed.core_slots(2, 12)
    assert s == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    s = speed.core_slots(1, 12)
    assert s == [list(range(8))]                               # 프로세스당 8 이하
    s = speed.core_slots(6, 12)
    assert [len(x) for x in s] == [2] * 6 and len({c for x in s for c in x}) == 12
    assert speed.core_slots(2, None) is None


class _FakeProc:
    def __init__(self, name, log, rc=0, steps=2):
        self.name, self.rc, self.steps, self.pid = name, rc, steps, 0
        log.append(("start", name))
        self.log = log

    def poll(self):
        self.steps -= 1
        if self.steps > 0:
            return None
        if ("end", self.name) not in self.log:
            self.log.append(("end", self.name))
        return self.rc

    def terminate(self):  # pragma: no cover
        pass


def test_parallel_groups_order_and_once(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    log = []
    groups = [["a1", "a2"], ["c1"], ["b1"], ["c2"], ["b2"]]
    serial = {"c1", "c2"}                                      # 사슬 판(최대 반복 기록 차례)

    def launch(cmd, lp, cores):
        name = cmd[-1]
        return _FakeProc(name, log, steps={"c1": 6}.get(name, 2))
    el = rn.ExperimentLog(tmp_path / "log.jsonl")
    done = rn.run_variants_isolated("s.py", groups, ["--x"], el, "t", processes=3, cpu_cores=6, serial=serial,
                                    log_dir=tmp_path, launch=launch)
    starts = [n for e, n in log if e == "start"]
    assert sorted(starts) == sorted(",".join(g) for g in groups) and len(starts) == len(set(starts))
    assert sorted(done) == sorted(n for g in groups for n in g)
    i_end_c1 = log.index(("end", "c1"))
    assert log.index(("start", "c2")) > i_end_c1               # 사슬 판 묶음은 차례대로
    assert log.index(("start", "b1")) < i_end_c1               # 다른 묶음은 동시에


def test_parallel_groups_stop_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    log = []

    def launch(cmd, lp, cores):
        return _FakeProc(cmd[-1], log, rc=1 if cmd[-1] == "x" else 0, steps=2 if cmd[-1] == "x" else 5)
    el = rn.ExperimentLog(tmp_path / "log.jsonl")
    with pytest.raises(rn.RunStopped):
        rn.run_variants_isolated("s.py", [["x"], ["y"], ["z"], ["w"]], [], el, "t", processes=2, cpu_cores=4,
                                 log_dir=tmp_path, launch=launch)
    starts = [n for e, n in log if e == "start"]
    assert starts == ["x", "y"]                                # 멈춘 뒤 새로 안 띄우고, 돌던 y 는 끝까지
    assert ("end", "y") in log
    assert any(e.get("event") == "stopped" and e["name"] == "x" for e in el.entries())


def test_stage_lock(tmp_path):
    p = tmp_path / rn.PARALLEL_LOCK
    with rn._StageLock(p):
        with pytest.raises(rn.RunStopped):
            with rn._StageLock(p):
                pass
    with rn._StageLock(p):
        pass


def test_stage_cfg_speed_field(tmp_path):
    from lst import stage_cli as sc

    code = Path(__file__).resolve().parents[1]
    raw = cf.load_json_strict(code / "configs" / "s1_2_lstm.json")
    raw["speed"] = {"dl_keyed_inputs": True, "parallel_groups": 2, "cpu_cores": 12, "gpu_fraction_cap": 0.25}
    p = tmp_path / "c.json"
    import json

    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    sys.path.insert(0, str(code / "scripts"))
    import s1_2_lstm

    cfg = sc.load_cfg(p, s1_2_lstm.SCHEMA_EXTRA)
    assert cfg["speed"]["parallel_groups"] == 2
    assert speed.on("dl_keyed_inputs") and speed.get("parallel_groups") == 2
    raw["speed"] = {"nope": 1}
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(cf.ConfigError):
        sc.load_cfg(p, s1_2_lstm.SCHEMA_EXTRA)


def test_gpu_cap_applies(monkeypatch):
    from lst import resources as rs

    torch = pytest.importorskip("torch")
    got = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: got.append(f))
    rs.limit_gpu_memory(0.4)
    speed.configure({"gpu_fraction_cap": 0.25})
    rs.limit_gpu_memory(0.4)
    rs.limit_gpu_memory(0.1)
    assert got == [0.4, 0.25, 0.1]


def test_speed_check_functions(monkeypatch):
    """scripts/speed_check 의 대조 함수(합성 자료 — 서버 대조 명령이 도는지)."""
    pytest.importorskip("torch")
    from lst import evaluate as ev
    from lst import importance as im
    from lst import models_dl as md
    from tests.test_importance import _registry_ctx

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import speed_check as sck

    monkeypatch.setattr(md, "PRED_BATCH", 256)
    ctx = _registry_ctx()
    spec = rg.VersionSpec("t", 2, 20, "sum", include_ids=rg.ALL_IDS, k=50, n_ma=15, n_ma_dl=7)
    split = fd.Split("fold1", fd.day("2016-06-30"), fd.day("2016-07-01"), fd.day("2017-05-01"))
    samples = ev.SampleTable(ctx.samples)
    rows = rn.add_bounds(fd.make_train_rows(samples.known_frame(), split).reset_index(drop=True), None)
    dasm = ip.DLInputs(ctx, spec, split).fit(rows)
    Xd = dasm.transform(rows)
    net = md.DLClassifier("GRU", dasm.cat_sizes, device="cpu", max_epochs=1)
    net.fit(Xd, rows["y"].to_numpy(np.int64), seed=42, n_jobs=1, groups=(rows["sgg"].to_numpy(), fd.day_ints(rows["d"])))
    leaf = im.Leaf("GRU", dasm, net)
    hs = [h for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, split) if idx.size]
    r1 = sck.check_inputs(dasm, samples, split, 20, None)
    assert r1["same_bytes"] and r1["n_h"] == len(hs)
    r2 = sck.check_predict(leaf, samples, split, 20, hs[0])
    assert r2["same_bytes"]
    r3 = sck.check_importance(leaf, samples, split, "fold1", hs[0], 2)
    assert r3["base_same"] and r3["perm_same_bytes"] and r3["perm_same_round4"]
