"""단계 3-0·추가 판 스크립트(v78) — 판 목록(--list), 3-0 순위(첫 현재 최고 판·트리 대상 판·상위 3개)·⑥·⑦ 판정·⑧ 바탕.
가짜 입력·가짜 모델로 3-0 13 판을 만들어(runner 그대로) 그 위에서 돈다(중요도 계산 자체는 test_importance)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import evaluate as ev
from lst import registry as rg
from lst import runner as rn
from lst import stage3 as s3
from tests.test_chain import DEC, SEL, make_model
from tests.test_runner import STUB, FakeAssembler, env  # noqa: F401

CODE = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", CODE / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def write_decisions(inputs: Path):
    p = inputs / "s1" / "1-3"
    p.mkdir(parents=True, exist_ok=True)
    (p / "1-3_decision.json").write_text(json.dumps(DEC), encoding="utf-8")
    q = inputs / "s2" / "2-select"
    q.mkdir(parents=True, exist_ok=True)
    (q / "2_prep_selection.json").write_text(json.dumps(SEL), encoding="utf-8")


def cfg_files(tmp: Path):
    """실제 설정 파일을 그대로 쓰되 MLflow 끔."""
    out = {}
    for n in ("s3_0_T", "s3_0_D", "s3_0b", "s3_extra_G", "s3_extra_C", "s3_extra_E", "s3_extra_judge"):
        c = json.loads((CODE / "configs" / f"{n}.json").read_text(encoding="utf-8"))
        c["use_mlflow"] = False
        p = tmp / f"{n}.json"
        p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
        out[n] = p
    return out


def test_list_specs(tmp_path, capsys):
    inputs = tmp_path / "results"
    write_decisions(inputs)
    cfgs = cfg_files(tmp_path)
    m = load_script("s3_0")
    a = ["--root", str(tmp_path), "--out", str(tmp_path / "o"), "--inputs", str(inputs)]
    assert m.main(a + ["--config", str(cfgs["s3_0_T"]), "--list"]) == [f"3-0-{x}" for x in s3.TREE6]
    assert m.main(a + ["--config", str(cfgs["s3_0_D"]), "--list"]) == [f"3-0-{x}" for x in s3.DL7]
    x = load_script("s3_extra")
    assert x.main(a + ["--config", str(cfgs["s3_extra_G"]), "--list"]) == ["x6-LSTMA", "x7-D"]
    with pytest.raises(SystemExit):   # ⑦-T(x7-T) 제외(2026-11-10) — sub 'C' 는 멈춤
        x.main(a + ["--config", str(cfgs["s3_extra_C"]), "--list"])


def _ctx(env, out, log):
    return rn.RunContext(samples=env["st"], out_dir=out, code_root=CODE, input_md5s={"x": "0"}, exp_log=log,
                         make_inputs=FakeAssembler, make_model=make_model, n_jobs=8, check_resources=STUB,
                         require_7a_logged=False)


@pytest.fixture()
def stage3_env(env, tmp_path, monkeypatch):
    """가짜로 끝낸 1차 베이스라인·3-0 13 판(최대 반복은 2,000 에서 정해지게 — 판정 규칙 시험은 test_chain)."""
    real = rn.max_iter_step
    monkeypatch.setattr(rn, "max_iter_step", lambda c, p, mi: real(np.r_[1.0, np.zeros(len(p) - 1)], p, mi))
    inputs = env["inputs"]
    write_decisions(inputs)
    log = rn.ExperimentLog(inputs / "logs" / "experiment_log.jsonl")
    b = rn.VariantSpec(name=DEC["baseline"], stage="1-1", order="1차", kind="tree", algorithm="CAT", W=60, mode="sum",
                       input_set="1차-트리", role="후보", train_count=4, judgement="해당 없음")
    rn.run_variant(b, _ctx(env, inputs / "s1" / "1-1", log))
    for sub in ("T", "D"):
        for g in rn.group_specs(s3.specs_3_0(sub, DEC, SEL)):
            rn.run_group(g, _ctx(env, inputs / "s3" / f"3-0-{sub}", log))
    cfgs = cfg_files(tmp_path)
    return {**env, "cfgs": cfgs, "log": log}


def _cfg(m, path, out):
    c = m.load_cfg(path)
    c["_out"] = str(out)
    return c


def run_extra_judge(e, tmp_path):
    """⑥⑦(가짜) 판 → 판정(3-0 순위 포함)."""
    x = load_script("s3_extra")
    inputs = e["inputs"]
    for sub, dname in (("G", "x678-G"),):   # ⑦-T(x7-T) 제외 — x678-C 판 없음
        for g in rn.group_specs(s3.specs_extra(sub, DEC, SEL)):
            rn.run_group(g, _ctx(e, inputs / "s3" / dname, e["log"]))
    jo = inputs / "s3" / "x67-judge"
    return x.main(["--root", str(tmp_path), "--out", str(jo), "--inputs", str(inputs), "--config",
                   str(e["cfgs"]["s3_extra_judge"])], tracker_fn=lambda c: None)


def test_extra_judge_from_3_0_and_x8(stage3_env, tmp_path):
    """v78: 3-0 13 판 CV 1위 = 첫 현재 최고 판(출발), 트리 6종 CV 순 = 트리 대상 판·상위 3개, ⑥⑦ 판정 상대 = 3-0 1위,
    ⑧ 바탕 = 3-0 트리 6종 1위(변수 전체), ⑧ 단계 = '⑧'."""
    e = stage3_env
    inputs = e["inputs"]
    cvs, alg_of = {}, {}
    for sub in ("T", "D"):
        for a in s3.SUB_ALGS[sub]:
            cvs[f"3-0-{a}"] = rn.load_done(inputs / "s3" / f"3-0-{sub}", f"3-0-{a}")[2]["cv_band_weighted"]
            alg_of[f"3-0-{a}"] = a
    ranked = sorted(cvs, key=lambda n: (-ev.mt.round4(cvs[n]), n))
    trees = [n for n in ranked if alg_of[n] in s3.TREE6]
    r = run_extra_judge(e, tmp_path)
    assert r["first_3_0"] == ranked[0] == r["current_best_before"]
    assert r["tree_rank_3_0"] == trees and r["tree_top3"] == trees[:3] and r["tree_target"] == trees[0]
    assert set(r["current_inputs"]) == {alg_of[n] for n in trees[:3]}
    cv = {n: rn.read_json(inputs / "s3" / "x678-G" / n / "done.json")["cv_band_weighted"]
          for n in ("x6-LSTMA", "x7-D")}
    top = sorted(cv, key=lambda n: (-ev.mt.round4(cv[n]), n))[0]
    assert r["top"] == top and r["adopted"] == ev.mt.is_improvement(cv[top], cvs[ranked[0]])
    assert r["current_best_after"] == (top if r["adopted"] else ranked[0])
    chain = [json.loads(x) for x in (inputs / "logs" / "chain_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["step"] for c in chain] == ["3-0", "⑥⑦"] and chain[0]["judgement"] == "출발"
    assert chain[0]["after"] == ranked[0] and chain[1]["after"] == r["current_best_after"]
    # ⑧
    x = load_script("s3_extra")
    rank = x.ranking_3_0(inputs, json.loads(e["cfgs"]["s3_extra_E"].read_text(encoding="utf-8")))
    base = x.x8_base_spec(inputs, rank)
    assert base.name == trees[0] and "exclude_groups" not in base.extra["version"]
    sp = s3.specs_extra("E", DEC, SEL, base)
    assert [s.name for s in sp] == [f"x8-{alg_of[trees[0]]}"] and sp[0].stage == "⑧"
    assert x.main(["--root", str(tmp_path), "--out", str(tmp_path / "o"), "--inputs", str(inputs), "--config",
                   str(e["cfgs"]["s3_extra_E"]), "--list"]) == [sp[0].name]
