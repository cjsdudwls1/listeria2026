"""추가 판 ③(s_x3.py) 시험 — 합성 자료·가짜 입력·가짜 모델: 겹 나누기 결정성·겹침 0·validation 쌍 = 시료 × 60·보조 채점
월일 거르기·threshold·이어하기·test 막기(--report 없이 test 정답 안 읽음)·보고."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import evaluate as ev
from lst import folds as fd
from lst import metrics as mt
from lst import runner as rn
from lst import stage_cli as sc
from tests.test_evaluate import fake_preds
from tests.test_runner import STUB, FakeAssembler, FakeModel, env  # noqa: F401

CODE = Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location("s_x3", CODE / "scripts" / "s_x3.py")
x3 = importlib.util.module_from_spec(spec_)
sys.modules["s_x3"] = x3
spec_.loader.exec_module(x3)

BASE = "1-CAT-W020-sum"


def _setup(env, smoke_pairs=None):
    inputs = env["inputs"]
    st = env["st"]
    bdir = inputs / "s1" / "1-1" / BASE
    bdir.mkdir(parents=True)
    spec = rn.VariantSpec(name=BASE, stage="1-1", order="1차", kind="tree", algorithm="CAT", W=20, mode="sum",
                          input_set="1차-트리", role="후보", train_count=4, judgement="해당 없음",
                          extra={"version": {"include_ids": ["WEATHER"]}})
    (bdir / "spec.json").write_text(json.dumps(spec.to_dict(), ensure_ascii=False), encoding="utf-8")
    (bdir / "done.json").write_text("{}", encoding="utf-8")
    cvp = {f: fake_preds(st, f, seed=i) for i, f in enumerate(ev.CV_FOLDS)}
    from lst import config as cf

    cf.dump_scores_json(bdir / "cv_scores.json", ev.score_cv(cvp, st))
    fake_preds(st, "refit", seed=9).to_parquet(bdir / "pred_refit.parquet", index=False)
    d13 = inputs / "s1" / "1-3"
    d13.mkdir(parents=True)
    (d13 / "1-3_decision.json").write_text(json.dumps({"baseline": BASE}), encoding="utf-8")
    cfg = json.loads((CODE / "configs" / "s_x3.json").read_text(encoding="utf-8"))
    cfg["use_mlflow"] = False
    if smoke_pairs:
        cfg["smoke"] = {"max_pairs_per_h": smoke_pairs}
    p = env["tmp"] / "cfg.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return p


def _fac(cfg, rp, inputs_root, spec):
    return sc.Factories(make_inputs=FakeAssembler, make_model=FakeModel, input_md5s={"x": "0"})


def _argv(env, cfgp, *extra):
    return ["--root", str(env["tmp"]), "--out", str(env["inputs"] / "s1" / "x3"), "--inputs", str(env["inputs"]),
            "--config", str(cfgp), *extra]


def _run(env, cfgp, **kw):
    return x3.main(_argv(env, cfgp), factories_fn=_fac, tracker_fn=lambda c: None, check_resources_fn=STUB,
                   before_fit_fn=lambda c: None, **kw)


def test_assign_folds_deterministic_balanced():
    keys = np.array([f"K{i}" for i in range(1001)], dtype=object)
    a, b = x3.assign_folds(keys), x3.assign_folds(keys)
    assert np.array_equal(a, b)
    perm = np.random.default_rng(42).permutation(1001)
    assert a[perm[0]] == 1 and a[perm[1]] == 2 and a[perm[2]] == 3
    c = np.bincount(a)[1:]
    assert c.max() - c.min() <= 1


def test_aux_mask():
    days = fd.day_ints(pd.to_datetime(["2023-07-01", "2023-05-01", "2023-05-02", "2023-06-30", "2024-02-29",
                                       "2023-12-31"]).values)
    assert x3._aux_mask(days).tolist() == [True, True, False, False, True, True]


def test_end_to_end_resume_and_test_block(env, monkeypatch):
    cfgp = _setup(env)

    def boom(*a, **k):
        raise AssertionError("test 정답을 읽으려 함")
    monkeypatch.setattr(ev.TestTruth, "load", classmethod(boom))
    done = _run(env, cfgp)
    vroot = env["inputs"] / "s1" / "x3" / f"x3-{BASE}"
    st = env["st"]
    known = st.known_frame()
    assign = pd.read_parquet(vroot / "fold_assign.parquet")
    assert len(assign) == len(known)
    # 겹 train/validation 겹침 0, validation 쌍 = 시료 × 60
    for f in x3.FOLDS:
        p = pd.read_parquet(vroot / f / "pred.parquet")
        vk = set(assign.loc[assign["fold"] == f, "lims_data_key"])
        assert set(p["lims_data_key"]) == vk and len(p) == len(vk) * 60
        rows = x3.train_rows(known.reset_index(drop=True), assign, f, 20)
        assert not (set(rows["lims_data_key"]) & vk)
        assert (rows["h"].to_numpy() == known.set_index("lims_data_key").loc[rows["lims_data_key"], "h"].to_numpy()).all()
    sc_ = json.loads((vroot / "scores.json").read_text(encoding="utf-8"))
    preds = {f: pd.read_parquet(vroot / f / "pred.parquet") for f in x3.FOLDS}
    allp = pd.concat(preds.values(), ignore_index=True)
    y = st.cv_y(st.positions(allp["lims_data_key"]))
    assert sc_["threshold"]["threshold"] == mt.select_threshold(y, allp["pred"].to_numpy())["threshold"]
    assert abs(sc_["cv_band_weighted"] - np.mean(list(sc_["fold_band_weighted"].values()))) < 1e-12
    assert done["cv_band_weighted"] == sc_["cv_band_weighted"]
    assert (vroot / "table_cv.csv").exists() and (vroot / "run_meta.json").exists()
    # 실험 로그: '무작위 겹' 따로
    e = [x for x in env["log"].entries() if x["event"] == "cv_scored"]
    assert len(e) == 1 and e[0]["kind"] == "무작위 겹" and e[0]["stage"] == "③"
    # 다시 돌리면 건너뜀
    again = _run(env, cfgp)
    assert again["cv_band_weighted"] == done["cv_band_weighted"]


def test_resume_partial_fold(env):
    cfgp = _setup(env)
    calls = {"n": 0}

    class Boom(FakeModel):
        def fit(self, X, y, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise MemoryError("가짜")
            return super().fit(X, y, **kw)

    def fac(cfg, rp, inputs_root, spec):
        return sc.Factories(make_inputs=FakeAssembler, make_model=Boom, input_md5s={"x": "0"})

    with pytest.raises(MemoryError):
        x3.main(_argv(env, cfgp), factories_fn=fac, tracker_fn=lambda c: None, check_resources_fn=STUB,
                before_fit_fn=lambda c: None)
    vroot = env["inputs"] / "s1" / "x3" / f"x3-{BASE}"
    assert (vroot / "x3f1" / "done.json").exists() and (vroot / "x3f2" / "failed.json").exists()
    t1 = (vroot / "x3f1" / "done.json").read_bytes()
    _run(env, cfgp)
    assert (vroot / "x3f1" / "done.json").read_bytes() == t1          # 끝난 겹은 다시 안 돌림
    assert list((vroot / "_incomplete").iterdir())                      # 멈춘 겹은 옮기고 그 겹부터
    assert (vroot / "done.json").exists()


def test_report_two_thresholds(env):
    cfgp = _setup(env)
    _run(env, cfgp)
    with pytest.raises(SystemExit):
        x3.main(_argv(env, cfgp, "--report"), tracker_fn=lambda c: None, check_resources_fn=STUB)
    tab = x3.main(_argv(env, cfgp, "--report", "--reason", "시험 보고"), tracker_fn=lambda c: None,
                  check_resources_fn=STUB)
    t = pd.read_csv(tab, encoding="utf-8-sig")
    assert list(t["threshold 출처"]) == ["무작위 겹 threshold", "시간순 threshold"]
    vroot = env["inputs"] / "s1" / "x3" / f"x3-{BASE}"
    js = json.loads((vroot / "test_scores.json").read_text(encoding="utf-8"))
    bcv = json.loads((env["inputs"] / "s1" / "1-1" / BASE / "cv_scores.json").read_text(encoding="utf-8"))
    assert js["threshold_chronological"] == bcv["threshold"]["threshold"]


def test_code_change_stops(env):
    cfgp = _setup(env)
    out = env["inputs"] / "s1" / "x3"
    rn.snapshot_code(CODE, out)
    (out / "_code" / "code_md5.txt").write_text("0" * 32, encoding="utf-8")
    with pytest.raises(rn.ResumeError):
        _run(env, cfgp)


def test_config_strict(env, tmp_path):
    cfg = json.loads((CODE / "configs" / "s_x3.json").read_text(encoding="utf-8"))
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    assert x3.load_cfg(p)["stage"] == "③"
    cfg["모르는칸"] = 1
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception):
        x3.load_cfg(p)


def test_mlflow_run_and_report(env):
    cfgp = _setup(env)
    tr = env["tracker"]
    done = x3.main(_argv(env, cfgp), factories_fn=_fac, tracker_fn=lambda c: tr, check_resources_fn=STUB,
                   before_fit_fn=lambda c: None)
    r = tr.client.get_run(done["mlflow_run_id"])
    assert r.data.tags["판정"] == "해당 없음" and r.data.tags["역할"] == "비교만" and r.data.tags["바탕 판"] == BASE
    assert "cv_band_weighted" in r.data.metrics
    x3.main(_argv(env, cfgp, "--report", "--reason", "보고"), tracker_fn=lambda c: tr, check_resources_fn=STUB)
    m = tr.client.get_run(done["mlflow_run_id"]).data.metrics
    assert "test_band_weighted" in m and "test_chrono_threshold" in m
