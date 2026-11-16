"""중요도 나눠 돌리기·이어 하기(lst.importance 구현 세부 h·scripts/s3_0b importance_parts) — 합성 자료·가짜 노드.

  - (fold, h 구간) 부분으로 나눠(차례도 섞어) 돌린 뒤 합친 결과 = 한 번에 돌린 permutation_importance(records·summary
    CSV 바이트, meta 는 seconds 만 빼고 같음).
  - 중간에 멈춘 뒤 이어 한 결과도 같음, 끝난 h 는 다시 예측하지 않음, 서명이 다르면 멈춤, 빠진 h 가 있으면 합치기 멈춤.
  - s3_0b: importance_parts 로 돌린 3-0b_importance/ = 기본(하나) 결과(가짜 노드), 부분 하나가 멈춰도 나머지는 끝까지·
    다시 하면 남은 h 만, 실제 하위 프로세스 띄우기(launch_parts), 설정 점검.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import folds as fd
from lst import importance as im

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _imp_tune_synth import (LOOKUP, SPLITS, DictAsm, DictModel, FrameAsm, FrameModel,  # noqa: E402
                             make_samples)


@pytest.fixture(scope="module")
def data():
    return make_samples()


class CountModel(FrameModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict_proba(self, X, n_iter=None):
        self.calls += 1
        return super().predict_proba(X, n_iter)


def graph(feats):
    """fold 마다 Mean(트리 꼴 Leaf, 딥러닝 꼴 Leaf) — 칼럼·채널 둘 다 섞임."""
    out = {}
    for f in fd.CV_FOLDS:
        fa = FrameAsm(feats)
        tree = im.Leaf(f"T-{f}", fa, CountModel(), layout=im.layout_from_names(fa.columns, LOOKUP))
        dl = im.Leaf(f"D-{f}", DictAsm(feats), DictModel(),
                     layout=im.layout_from_names({"x_seq": ["avg_temp", "avg_temp__na"],
                                                  "x_static": ["t3", "g_div50"]}, LOOKUP))
        out[f] = im.Mean(f"M-{f}", [tree, dl])
    return out


TARGETS = [im.unit_target("t3"), im.unit_target("L1"), im.unit_target("avg_temp")]
M = 3


def _files(res, d: Path) -> dict:
    res.save(d)
    out = {n: (d / f"importance_{n}.csv").read_bytes() for n in ("records", "summary")}
    meta = json.loads((d / "importance_meta.json").read_text(encoding="utf-8"))
    meta.pop("seconds")
    out["meta"] = meta
    return out


def _run_parts(models, samples, state, parts, tag="t", check=None):
    for f, lo, hi in parts:
        im.importance_part(models, samples, TARGETS, fold=f, h_from=lo, h_to=hi, M=M, state_dir=state,
                           splits=SPLITS, tag=tag, check=check)


def _merge(samples, state, tag="t"):
    tg = [im.target_from_dict(d) for d in im.state_targets(state)]
    assert [t.key for t in tg] == [t.key for t in TARGETS]
    return im.merge_parts(samples, tg, state_dir=state, M=M, splits=SPLITS, tag=tag)


def test_parts_merge_equals_sequential(data, tmp_path):
    samples, feats = data
    seq = im.permutation_importance(graph(feats), samples, TARGETS, M=M, splits=SPLITS)
    parts = [("fold3", 41, 60), ("fold2", 18, 60), ("fold1", 1, 60), ("fold3", 1, 9), ("fold2", 1, 17),
             ("fold3", 10, 40)]                                                  # 차례 섞음·구간 고르지 않음
    _run_parts(graph(feats), samples, tmp_path / "state", parts)
    assert all(not im.pending_h(samples, f, 1, 60, tmp_path / "state", SPLITS) for f in fd.CV_FOLDS)
    mer = _merge(samples, tmp_path / "state")
    assert mer.base == seq.base and mer.n_pairs == seq.n_pairs
    for f in fd.CV_FOLDS:
        assert np.array_equal(mer.base_by_h[f], seq.base_by_h[f], equal_nan=True)
    a, b = _files(seq, tmp_path / "a"), _files(mer, tmp_path / "b")
    assert a["records"] == b["records"] and a["summary"] == b["summary"] and a["meta"] == b["meta"]
    assert (seq.records["I"] != 0).any()                                         # 섞기가 실제로 점수를 바꿈


def test_resume_after_stop_same_and_skips_done(data, tmp_path):
    samples, feats = data
    seq = im.permutation_importance(graph(feats), samples, TARGETS, M=M, splits=SPLITS)
    state = tmp_path / "state"

    class Stop(RuntimeError):
        pass

    n = {"k": 0}

    def check():                                                                 # 7번째 h 앞에서 멈춤(자원 하한 흉내)
        n["k"] += 1
        if n["k"] == 7:
            raise Stop()

    g = graph(feats)
    with pytest.raises(Stop):
        im.importance_part(g, samples, TARGETS, fold="fold2", h_from=1, h_to=60, M=M, state_dir=state,
                           splits=SPLITS, tag="t", check=check)
    done = sorted(int(p.stem[1:]) for p in (state / "fold2").glob("h*.npz"))
    assert len(done) == 6 and not list((state / "fold2").glob(".*tmp"))
    g2 = graph(feats)
    r = im.importance_part(g2, samples, TARGETS, fold="fold2", h_from=1, h_to=60, M=M, state_dir=state,
                           splits=SPLITS, tag="t")
    assert r["resumed"] == done and set(r["computed"]).isdisjoint(done)
    calls = g2["fold2"].members[0].model.calls
    assert calls == len(r["computed"]) * (1 + 2)          # 끝난 h 는 예측 안 함(트리 입력에 있는 대상 = t3·L1)
    # 부분 구간으로 나눠 이어 해도(앞 구간 끝난 상태 + 다른 나눔) 같음
    _run_parts(graph(feats), samples, state, [("fold1", 1, 25), ("fold3", 1, 60)])
    _run_parts(graph(feats), samples, state, [("fold1", 1, 60)])               # 1~25 는 이어받기, 26~ 계산
    mer = _merge(samples, state)
    a, b = _files(seq, tmp_path / "a"), _files(mer, tmp_path / "b")
    assert a == b


def test_signature_and_missing_h(data, tmp_path):
    samples, feats = data
    state = tmp_path / "state"
    _run_parts(graph(feats), samples, state, [("fold1", 1, 20)])
    with pytest.raises(im.ImportanceError, match="서명"):
        _run_parts(graph(feats), samples, state, [("fold1", 1, 20)], tag="다른 판")
    with pytest.raises(im.ImportanceError, match="서명"):
        im.importance_part(graph(feats), samples, TARGETS[:2], fold="fold1", h_from=1, h_to=5, M=M,
                           state_dir=state, splits=SPLITS, tag="t")
    with pytest.raises(im.ImportanceError, match="안 끝난 h"):
        _merge(samples, state)
    with pytest.raises(im.ImportanceError, match="h 구간"):
        im.importance_part(graph(feats), samples, TARGETS, fold="fold1", h_from=5, h_to=61, M=M, state_dir=state,
                           splits=SPLITS)
    with pytest.raises(im.ImportanceError, match="모델 입력에 없는 대상"):
        im.importance_part(graph(feats), samples, [im.group_target("대기질")], fold="fold1", M=M, state_dir=state,
                           splits=SPLITS)


def test_target_from_dict_roundtrip():
    from lst import registry as rg

    units = {u for g in rg.GROUPS for u in g.units}
    ts = (im.targets_groups(sorted(units)) + [im.unit_target("t3")] + im.targets_2r1()
          + im.targets_final(units | set(im.RESEARCH_COLUMNS)))
    assert len(ts) > 50
    for t in ts:
        d = json.loads(json.dumps(im._target_dict(t)))
        assert im.target_from_dict(d) == t


# ---------------------------------------------------------------- s3_0b(가짜 노드 — tests/test_s3_0b 와 같은 환경)
from tests.test_chain import DEC, SEL  # noqa: E402,F401
from tests.test_runner import STUB, FakeAssembler, env  # noqa: E402,F401
from tests.test_s3_0b import _fixed_code_md5, _state_first, b_cfg, fake_member_nodes  # noqa: E402,F401
from tests.test_s3_scripts import CODE, load_script, stage3_env  # noqa: E402,F401


def _first_lstm(e):
    from lst import runner as rn

    src = rn.VariantSpec(**json.loads((e["inputs"] / "s3" / "3-0-D" / "3-0-LSTM" / "spec.json").read_text("utf-8")))
    _state_first(e, src, "D")


def _setup(m, e, tmp_path, tag, **kw):
    from lst.paths import resolve

    inputs = e["inputs"]
    cfg = m.load_cfg(b_cfg(tmp_path, tag, chain_state_in="logs/chain_state_3-3_old.json", **kw))
    out = resolve(inputs, cfg["dirs"]["3-0b"])
    m.choose_target(cfg, inputs, out)
    return cfg, out


def test_s3_0b_parts_equal_default_and_resume(stage3_env, tmp_path):
    from lst import runner as rn
    from lst import stage3b as sb

    e = stage3_env
    inputs = e["inputs"]
    _first_lstm(e)
    m = load_script("s3_0b")
    nodes = fake_member_nodes(inputs)
    cfg1, out1 = _setup(m, e, tmp_path, "one")
    m.run_importance(cfg1, tmp_path, inputs, out1, member_nodes_fn=nodes, check=lambda: None)

    cfg2, out2 = _setup(m, e, tmp_path, "parts", importance_parts=["fold1", "fold2:1-30", "fold2:31-60", "fold3"])
    k = {"n": 0}

    def stop_once():
        k["n"] += 1
        if k["n"] == 40:
            raise rn.RunStopped("자원 하한(흉내)")
    with pytest.raises(rn.RunStopped):
        m.run_importance_parts(cfg2, [], tmp_path, inputs, out2, member_nodes_fn=nodes, check=stop_once,
                               in_process=True)
    assert not m.importance_done(out2) and (out2 / m.STATE_DIR).exists()
    assert m.pending_parts(cfg2, inputs, out2) and list((out2 / m.STATE_DIR).glob("fold*/h*.npz"))
    # 다시(같은 명령) — 남은 h 만, 합친 결과 = 기본 결과
    m.run_importance_parts(cfg2, [], tmp_path, inputs, out2, member_nodes_fn=nodes, check=lambda: None,
                           in_process=True)
    a, b = out1 / sb.IMPORTANCE_DIR, out2 / sb.IMPORTANCE_DIR
    for f in ("importance_records.csv", "importance_summary.csv"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    ma, mb = (json.loads((d / "importance_meta.json").read_text(encoding="utf-8")) for d in (a, b))
    ma.pop("seconds"), mb.pop("seconds")
    assert ma == mb and ma["drop_groups"]
    assert not (out2 / m.STATE_DIR).exists()                                    # 합친 뒤 상태 폴더 지움
    assert m.drop_groups(out2) == m.drop_groups(out1)
    log = [json.loads(x) for x in (inputs / "logs" / "experiment_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(x.get("message") == "3-0b 중요도 나눠 돌리기" for x in log)


def test_s3_0b_launch_partial_failure(stage3_env, tmp_path):
    """본 프로세스 흐름: 부분 하나가 멈춰도 나머지는 끝까지 → RunStopped·실험 로그 → 다시 하면 그 부분만."""
    from lst import runner as rn
    from lst import stage3b as sb

    e = stage3_env
    inputs = e["inputs"]
    _first_lstm(e)
    m = load_script("s3_0b")
    nodes = fake_member_nodes(inputs)
    cfg, out = _setup(m, e, tmp_path, "launch")
    seen = []

    def fake_launch(fail):
        def launch(cmds, logs):
            rcs = []
            for c, lp in zip(cmds, logs):
                part = c[c.index("--part") + 1]
                assert c[c.index("--part") - 1] == "--importance" and lp.parent.name == m.PART_LOG_DIR
                seen.append(part)
                if part == fail:
                    rcs.append(3)
                    continue
                m.run_importance_part(cfg, tmp_path, inputs, out, part, member_nodes_fn=nodes, check=lambda: None)
                rcs.append(0)
            return rcs
        return launch

    with pytest.raises(rn.RunStopped, match="fold2:1-60"):
        m.run_importance_parts(cfg, ["--x"], tmp_path, inputs, out, launch=fake_launch("fold2:1-60"))
    assert seen == ["fold1:1-60", "fold2:1-60", "fold3:1-60"]
    assert m.pending_parts(cfg, inputs, out) == ["fold2:1-60"]
    log = [json.loads(x) for x in (inputs / "logs" / "experiment_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["event"] == "stopped" and log[-1]["name"] == "3-0b 중요도 부분 fold2:1-60" and log[-1]["returncode"] == 3
    seen.clear()
    m.run_importance_parts(cfg, ["--x"], tmp_path, inputs, out, launch=fake_launch(None))
    assert seen == ["fold2:1-60"] and m.importance_done(out)
    assert (out / sb.IMPORTANCE_DIR / "importance_summary.csv").exists()


def test_launch_parts_real_subprocesses(tmp_path):
    m = load_script("s3_0b")
    code = "import sys, time; print('part', sys.argv[1], flush=True); time.sleep(0.3); sys.exit(int(sys.argv[1]))"
    cmds = [[sys.executable, "-c", code, str(rc)] for rc in (0, 3, 0)]
    logs = [tmp_path / "logs" / f"p{i}.log" for i in range(3)]
    assert m.launch_parts(cmds, logs) == [0, 3, 0]
    assert all(lp.read_bytes().strip() for lp in logs)
    assert m.launch_parts(cmds[:1], logs[:1]) == [0]                           # 로그 이어 씀
    assert logs[0].read_text(encoding="ascii").count("part") == 2


def test_parts_config():
    from lst import config as cf

    m = load_script("s3_0b")
    assert m.parse_parts(["fold1", "fold2", "fold3:1-30", "fold3:31-60"])[-1] == ("fold3", 31, 60)
    for bad in ([], ["fold1", "fold2"], ["fold1", "fold2", "fold3:1-30"], ["fold1", "fold2", "fold3", "fold3:1-2"],
                ["fold1", "fold2", "fold3:1-30", "fold3:30-60"], ["fold1", "fold2", "refit"], ["fold1", "fold1"],
                ["fold1", "fold2", "fold3:0-60"], ["fold1", "fold2", "fold3:1-61"], ["fold1", "fold2", "fold3:a-b"]):
        with pytest.raises(cf.ConfigError):
            m.parse_parts(bad)
    c = m.load_cfg(CODE / "configs" / "s3_0b.json")
    assert c["importance_parts"] == ["fold1", "fold2", "fold3"] and "reuse_3_0b_old" not in c


def test_s3_0b_main_part_argument(stage3_env, tmp_path):
    """하위 프로세스가 받는 명령(--importance --part)을 main 으로 — 그 부분 h 만 상태 파일."""
    e = stage3_env
    inputs = e["inputs"]
    _first_lstm(e)
    m = load_script("s3_0b")
    cfgp = b_cfg(tmp_path, "main", chain_state_in="logs/chain_state_3-3_old.json",
                 importance_parts=["fold1", "fold2", "fold3:1-30", "fold3:31-60"])
    cfg, out = _setup(m, e, tmp_path, "main", importance_parts=["fold1", "fold2", "fold3:1-30", "fold3:31-60"])
    argv = ["--root", str(tmp_path), "--inputs", str(inputs), "--out", str(out), "--config", str(cfgp)]
    with pytest.raises(SystemExit):
        m.main(argv + ["--part", "fold3:31-60"], member_nodes_fn=fake_member_nodes(inputs))
    from lst import config as cf

    with pytest.raises(cf.ConfigError):
        m.main(argv + ["--importance", "--part", "fold3"], member_nodes_fn=fake_member_nodes(inputs),
               check_resources_fn=lambda: None)
    r = m.main(argv + ["--importance", "--part", "fold3:31-60"], member_nodes_fn=fake_member_nodes(inputs),
               check_resources_fn=lambda: None)
    hs = sorted(int(p.stem[1:]) for p in (out / m.STATE_DIR / "fold3").glob("h*.npz"))
    assert hs and min(hs) >= 31 and hs == r["computed"]
    assert not (out / m.STATE_DIR / "fold1").exists()
    assert m.pending_parts(cfg, inputs, out) == ["fold1:1-60", "fold2:1-60", "fold3:1-30"]


# ---------------------------------------------------------------- s_final --importance(가짜 노드 — tests/test_s_final 과 같은 환경)
from tests.test_s_final import NoisyModel, ResearchAsm, _argv, _cfg, _state, sf, world  # noqa: E402,F401


def _sf_nodes(spec):
    from lst import evaluate as ev
    from lst import runner as rn

    def nodes(cfg, root, inputs_root, decision, samples):
        out = {}
        known = samples.known_frame()
        lk = {"t3": "t3", "L1": "L1", "h": None, "g": None}
        for f in ev.CV_FOLDS:
            rows = rn.add_bounds(fd.make_train_rows(known, fd.SPLITS[f]), spec.W)
            asm = ResearchAsm(spec, f).fit(rows)
            mdl = NoisyModel(spec, asm).fit(asm.transform(rows), rows["y"].to_numpy(), seed=42, n_jobs=8)
            out[f] = im.Leaf(f"{spec.name}-{f}", asm, mdl, layout=im.layout_from_names(list(asm.columns), lk))
        return out
    return nodes


def _sf_cfg(w, tag, st, **kw):
    p = _cfg(w, tag, st)
    c = json.loads(p.read_text(encoding="utf-8"))
    c.update(kw)
    p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    return p


def test_s_final_parts_equal_default_and_resume(world):
    from lst import runner as rn

    w = world
    st = _state(w, "s4/state_parts.json", "4-a-CAT", "s4")
    nodes = _sf_nodes(w["V"]["4-a-CAT"][1])
    c1 = _sf_cfg(w, "p1", st)
    sf.main(_argv(w, "p1", c1))
    out1 = w["inputs"] / "final" / "p1"
    a = sf.run_importance(sf.load_cfg(c1), w["tmp"], w["inputs"], out1, nodes_fn=nodes, M=2, base_tol=1e-4)

    c2 = _sf_cfg(w, "p2", st, importance_parts=["fold1:1-40", "fold1:41-60", "fold2", "fold3"])
    sf.main(_argv(w, "p2", c2))
    out2 = w["inputs"] / "final" / "p2"
    cfg2 = sf.load_cfg(c2)
    k = {"n": 0}

    def stop_once():
        k["n"] += 1
        if k["n"] == 25:
            raise rn.RunStopped("자원 하한(흉내)")
    with pytest.raises(rn.RunStopped):
        sf.run_importance_parts(cfg2, [], w["tmp"], w["inputs"], out2, nodes_fn=nodes, M=2, base_tol=1e-4,
                                check=stop_once, in_process=True)
    assert not sf.importance_done(out2) and sf.pending_parts(cfg2, w["inputs"], out2)
    b = sf.run_importance_parts(cfg2, [], w["tmp"], w["inputs"], out2, nodes_fn=nodes, M=2, base_tol=1e-4,
                                check=lambda: None, in_process=True)
    for f in ("importance_records.csv", "importance_summary.csv"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    ma, mb = (json.loads((d / "importance_meta.json").read_text(encoding="utf-8")) for d in (a, b))
    ma.pop("seconds"), mb.pop("seconds")
    assert ma == mb
    assert {t["kind"] for t in ma["targets"]} == {"column", "bundle", "group"}
    assert all(t["seed_key"][0] in (im.SEED_FINAL_COL, im.SEED_FINAL_GROUP) for t in ma["targets"])
    assert not (out2 / sf.STATE_DIR).exists()
    assert sf.run_importance_parts(cfg2, [], w["tmp"], w["inputs"], out2, in_process=True) == b   # 이미 있음


def test_s_final_parts_unsupported_and_args(world):
    from lst import config as cf

    w = world
    st = _state(w, "s4/state_parts_u.json", "4-a-CAT", "s4")
    c = _sf_cfg(w, "pu", st, importance_parts=["fold1", "fold2", "fold3"])
    sf.main(_argv(w, "pu", c))
    out = w["inputs"] / "final" / "pu"

    def unsup(*a, **k):
        raise sf.Unsupported("가짜 — 지원 안 함")
    with pytest.raises(SystemExit):
        sf.main(_argv(w, "pu", c, "--part", "fold1"))
    with pytest.raises(cf.ConfigError):
        sf.main(_argv(w, "pu", c, "--importance", "--part", "fold1:1-30"), nodes_fn=unsup)
    sf.run_importance_parts(sf.load_cfg(c), [], w["tmp"], w["inputs"], out, nodes_fn=unsup, in_process=True)
    u = json.loads((out / sf.IMPORTANCE_DIR / "unsupported.json").read_text(encoding="utf-8"))
    assert u["reason"] == "가짜 — 지원 안 함" and not (out / sf.STATE_DIR).exists()
    with pytest.raises(cf.ConfigError):
        sf.load_cfg(_sf_cfg(w, "bad", st, importance_parts=["fold1", "fold2"]))
    # 하위 프로세스가 받는 명령 꼴(main --importance --part): 그 부분 h 만 상태 파일·부분 정보
    c2 = _sf_cfg(w, "pm", st, importance_parts=["fold1", "fold2", "fold3:1-30", "fold3:31-60"])
    sf.main(_argv(w, "pm", c2))
    out2 = w["inputs"] / "final" / "pm"
    r = sf.main(_argv(w, "pm", c2, "--importance", "--part", "fold3:31-60"), nodes_fn=_sf_nodes(w["V"]["4-a-CAT"][1]),
                importance_M=2, check_resources_fn=lambda: None)
    hs = sorted(int(q.stem[1:]) for q in (out2 / sf.STATE_DIR / "fold3").glob("h*.npz"))
    assert hs and min(hs) >= 31 and hs == r["computed"]
    assert sf.pending_parts(sf.load_cfg(c2), w["inputs"], out2) == ["fold1:1-60", "fold2:1-60", "fold3:1-30"]
    info = json.loads((out2 / sf.STATE_DIR / "parts" / "fold3_h31-60.json").read_text(encoding="utf-8"))
    assert info["unsupported"] is None and set(info["expect_base"]) == set(fd.CV_FOLDS)


def test_s_final_config_values():
    c = sf.load_cfg(CODE / "configs" / "s_final.json")
    assert c["importance_parts"] == ["fold1", "fold2", "fold3"] and c["gpu_memory_fraction"] == 0.25
