"""단계 4·5·6 스크립트 — 판 목록·판 설정(이름·RowPlan·범주 인코딩·씨앗), 단계 4 판정 → 사슬 상태(stage4 칸) → 5·6 뽑기·판
목록 → 5·6 판정. 가짜 입력·가짜 모델(runner 그대로)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import runner as rn
from lst import stage4 as s4
from tests.test_chain import DEC, SEL, make_model
from tests.test_runner import STUB, FakeAssembler, env  # noqa: F401
from tests.test_s3_scripts import load_script, write_decisions

CODE = Path(__file__).resolve().parents[1]


def _cfgs(tmp: Path):
    out = {}
    for n in ("s4_prep", "s4_pre", "s4_T", "s4_X1", "s4_NN", "s4_C", "s4_judge", "s5_6_prep", "s5_6_T",
              "s5_6_judge"):
        c = json.loads((CODE / "configs" / f"{n}.json").read_text(encoding="utf-8"))
        if "use_mlflow" in c:
            c["use_mlflow"] = False
        p = tmp / f"{n}.json"
        p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
        out[n] = p
    return out


def _ctx(env, out, log):
    return rn.RunContext(samples=env["st"], out_dir=out, code_root=CODE, input_md5s={"x": "0"}, exp_log=log,
                         make_inputs=FakeAssembler, make_model=make_model, n_jobs=8, check_resources=STUB,
                         require_7a_logged=False)


def _best(name, d, kind="tree", repl=False, **m):
    return {"name": name, "kind": kind, "dir": d, "replacement": repl, "members": [{"name": name, "dir": d, **m}]}


@pytest.fixture()
def s4env(env, tmp_path, monkeypatch):
    real = rn.max_iter_step
    monkeypatch.setattr(rn, "max_iter_step", lambda c, p, mi: real(np.r_[1.0, np.zeros(len(p) - 1)], p, mi))
    inputs = env["inputs"]
    write_decisions(inputs)
    log = rn.ExperimentLog(inputs / "logs" / "experiment_log.jsonl")
    b = rn.VariantSpec(name=DEC["baseline"], stage="1-1", order="1차", kind="tree", algorithm="CAT", W=60, mode="sum",
                       input_set="1차-트리", role="후보", train_count=4, judgement="해당 없음",
                       extra={"version": {"include_ids": ["WEATHER"]}})
    rn.run_variant(b, _ctx(env, inputs / "s1" / "1-1", log))
    cur = rn.VariantSpec(name="3-0b-XGB-src", stage="3-0", order="2차", kind="tree", algorithm="XGB", W=60, mode="sum",
                         input_set="3-0-전체", role="후보", train_count=4, params={rn.ITER_SELECT: True},
                         extra={"version": {"include_ids": ["WEATHER", "HIS1"], "k": 50}})
    rn.run_variant(cur, _ctx(env, inputs / "s3" / "3-0-T", log))
    best = _best("3-0b-XGB-end", "s3/3-0-T/3-0b-XGB-src")
    state = {"step": "3-0b", "current_best": best, "tree_target": best, "tree_top3": ["3-0b-XGB-end"],
             "tree_rank_3_0b": {"XGB": 1}, "history": []}
    s4.save_state(inputs / "logs" / "chain_state_3-0b.json", state)      # v78: 단계 4 는 3-0b 판정 뒤 사슬 상태를 읽음
    return {**env, "cfgs": _cfgs(tmp_path), "log": log, "best": best}


def _a(env, out):
    return ["--root", str(env["tmp"]), "--out", str(out), "--inputs", str(env["inputs"])]


def test_s4_lists_and_specs(s4env):
    m = load_script("s4")
    c = s4env["cfgs"]
    assert m.main(_a(s4env, "x") + ["--config", str(c["s4_T"]), "--list"]) == ["4-a-XGB", "4-a0-XGB"]
    assert m.main(_a(s4env, "x") + ["--config", str(c["s4_X1"]), "--list"]) == ["x1-XGB"]
    assert m.main(_a(s4env, "x") + ["--config", str(c["s4_NN"]), "--list"]) == ["x2-NN60"]
    assert m.main(_a(s4env, "x") + ["--config", str(c["s4_C"]), "--list"]) == ["4-c-XGB", "4-c-L-XGB"]
    assert m.main(_a(s4env, "x") + ["--config", str(c["s4_pre"]), "--list"]) == ["4-a-pre-CAT"]
    cfg = json.loads(c["s4_T"].read_text(encoding="utf-8"))
    b = m.build(cfg, s4env["inputs"])
    a, a0 = b["run"]
    for sp, rep in ((a, "4-a"), (a0, "own")):
        x = s4.s4_of(sp)
        assert s4.RowPlan.from_dict(x["row_plan"]) == s4.RowPlan(("reduce",), rep, "s4/sel-reduce")
        assert x["catenc_k"] == SEL["k"] and sp.params == {rn.ITER_SELECT: True}
        assert sp.extra["version"] == {"include_ids": ["WEATHER", "HIS1"], "k": 50} and sp.role == "후보"
    x1 = m.build(json.loads(c["s4_X1"].read_text(encoding="utf-8")), s4env["inputs"])["run"][0]
    assert s4.s4_of(x1)["drop_time"] is True and s4.s4_of(x1)["row_plan"]["rep"] == "x1" and x1.judgement == "해당 없음"
    cc = m.build(json.loads(c["s4_C"].read_text(encoding="utf-8")), s4env["inputs"])["run"]
    assert "catenc_k" not in s4.s4_of(cc[0])                       # 4-c = 범주 처리 그대로
    assert cc[1].extra["version"] == {"include_ids": []} and s4.s4_of(cc[1])["row_plan"]["steps"] == ["top5"]
    nn = m.build(json.loads(c["s4_NN"].read_text(encoding="utf-8")), s4env["inputs"])
    assert nn["run"][0].kind == "dl" and nn["bests"]["x2-NN60"]["replacement"] is True
    pre = m.build(json.loads(c["s4_pre"].read_text(encoding="utf-8")), s4env["inputs"])["run"][0]
    assert s4.s4_of(pre)["input_order"] == 1 and pre.params == {rn.ITER_SELECT: True}


def test_s4_dl_best_only_a0(s4env):
    m = load_script("s4")
    inputs = s4env["inputs"]
    dl = rn.VariantSpec(name="3-0-LSTM", stage="3-0", order="2차", kind="dl", algorithm="LSTM", W=60, mode=None,
                        input_set="3-0-전체-DL", role="후보", train_count=4, extra={"version": {"include_ids": ["WEATHER"]}})
    d = inputs / "s3" / "3-0-D" / "3-0-LSTM"
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps(dl.to_dict()), encoding="utf-8")
    st = s4.load_state(inputs / "logs" / "chain_state_3-0b.json")
    st["current_best"] = _best("3-0b-LSTM-end", "s3/3-0-D/3-0-LSTM", kind="dl")
    (inputs / "logs" / "chain_state_3-0b.json").unlink()
    s4.save_state(inputs / "logs" / "chain_state_3-0b.json", st)
    b = m.build(json.loads(s4env["cfgs"]["s4_T"].read_text(encoding="utf-8")), inputs)
    assert [s.name for s in b["run"]] == ["4-a0-LSTM"]
    assert s4.s4_of(b["run"][0])["row_plan"]["rep"] == "own" and "catenc_k" not in s4.s4_of(b["run"][0])


def test_ensemble_derive(s4env):
    inputs = s4env["inputs"]
    log = s4env["log"]
    for a in ("LGB", "CAT"):
        sp = rn.VariantSpec(name=f"3-0-{a}", stage="3-0", order="2차", kind="tree", algorithm=a, W=60, mode="sum",
                            input_set="3-0-전체", role="후보", train_count=4, params={rn.ITER_SELECT: True},
                            extra={"version": {"include_ids": ["WEATHER"]}})
        (inputs / "s3" / "3-0-T" / sp.name).mkdir(parents=True)
        (inputs / "s3" / "3-0-T" / sp.name / "spec.json").write_text(json.dumps(sp.to_dict()), encoding="utf-8")
    ens = {"name": "3-3-T3", "kind": "ensemble", "dir": "s3/3-3/3-3-T3", "replacement": False,
           "members": [{"name": "3-0b-XGB-end", "dir": "s3/3-0-T/3-0b-XGB-src"},
                       {"name": "3-0b-LGB-end", "dir": "s3/3-0-T/3-0-LGB",
                        "anomaly": {"column": "anomaly_vae", "version": {"include_ids": ["WEATHER"]}, "W": 60,
                                    "mode": "sum", "source": "3-1-VAEt"}},
                       {"name": "3-0b-CAT-end", "dir": "s3/3-0-T/3-0-CAT"}]}
    d = s4.derive_specs(ens, inputs_root=inputs, name="4-a-3-3-T3", stage="4", role="후보",
                        plan=s4.RowPlan(("reduce",), "4-a", "s4/sel-reduce"), catenc_k=50)
    assert [s.name for s in d.run] == ["4-a-3-3-T3--XGB", "4-a-3-3-T3--LGB", "4-a-3-3-T3--CAT"]
    assert s4.s4_of(d.run[1])["provider"]["kind"] == "vae" and "provider" not in s4.s4_of(d.run[0])
    assert d.ensemble.algorithm == "T3" and d.ensemble.extra["s4"]["members"] == [s.name for s in d.run]
    # 단계 8 꼴: 튜닝 설정·Focal(CatBoost 만)
    d8 = s4.derive_specs(ens, inputs_root=inputs, name="8-F-x", stage="8", role="후보",
                         plan=s4.RowPlan(("reduce",), "4-a", "s4/sel-reduce"), catenc_k=50,
                         member_params={"3-0b-XGB-end": {"max_depth": 3}}, focal_cat=True)
    assert d8.run[0].params["params"] == {"max_depth": 3}
    assert d8.run[2].params["params"]["loss_function"].startswith("Focal") and "params" not in d8.run[1].params
    # 5·6 씨앗 0
    d56 = s4.derive_specs(ens, inputs_root=inputs, name="5-x", stage="5-6", role="후보",
                          plan=s4.RowPlan(("hon5",), "own", "s5_6/sel-hon5"), catenc_k=50, model_seed=0)
    assert all(s4.s4_of(s)["model_seed"] == 0 for s in d56.run)


def _fake_run(env, log, out_dir, specs):
    """판 설정 그대로 가짜 모델로(이름·폴더만 맞추면 판정 시험엔 충분)."""
    for sp in specs:
        rn.run_variant(sp, _ctx(env, out_dir, log))


def test_judge4_then_5_6(s4env):
    m4, m56 = load_script("s4"), load_script("s5_6")
    inputs, c = s4env["inputs"], s4env["cfgs"]
    for sub, key in (("T", "4-T"), ("NN", "4-NN")):
        b = m4.build(json.loads(c[f"s4_{sub}"].read_text(encoding="utf-8")), inputs)
        _fake_run(s4env, s4env["log"], inputs / "s4" / key, [rn.VariantSpec(**{**sp.to_dict(), "algorithm": "XGB",
                                                                                   "kind": "tree"}) for sp in b["run"]])
    dec = m4.main(_a(s4env, inputs / "s4" / "4-judge") + ["--config", str(c["s4_judge"])])
    st = s4.load_state(inputs / "logs" / "chain_state_4.json")
    assert st["stage4"]["start_best"]["name"] == "3-0b-XGB-end"
    assert st["current_best"]["name"] == dec["record"]["after"]
    assert st["stage4"]["adopted"] == (st["current_best"]["name"] != "3-0b-XGB-end")
    lines = (inputs / "logs" / "chain_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["step"] == "4"
    # 줄인 폴더(가짜 — 단계 4 줄이기 결과 꼴)
    sel = inputs / "s4" / "sel-reduce"
    sel.mkdir(parents=True, exist_ok=True)
    known = s4env["st"].known_frame()
    for s in rn.SPLIT_ORDER:
        k = known[rn.fd.day_ints(known["d"]) <= rn.fd.SPLITS[s].train_end]["lims_data_key"]
        pd.DataFrame({"lims_data_key": k.iloc[::2].tolist() + k[known.loc[k.index, "y"] == 1].tolist()}).drop_duplicates(
        ).to_parquet(sel / f"kept_keys_{s}.parquet", index=False)
    (sel / "selection.json").write_text(json.dumps({"steps": ["reduce"], "reduce_column": {s: "prduct_nm" for s in
                                                                                             rn.SPLIT_ORDER},
                                                     "top5": None}), encoding="utf-8")
    out = m56.main(_a(s4env, inputs / "s5_6" / "sel") + ["--config", str(c["s5_6_prep"])])
    assert ("sep", "5") in out and ("sep", "6") in out
    assert (("stk", "5") in out) == st["stage4"]["adopted"]
    names = m56.main(_a(s4env, "x") + ["--config", str(c["s5_6_T"]), "--list"])
    want = ["5-XGB", "6-XGB"] + ([f"5-{st['current_best']['name']}", f"6-{st['current_best']['name']}"]
                                 if st["stage4"]["adopted"] else [])
    assert names == want
    b = m56.build(json.loads(c["s5_6_T"].read_text(encoding="utf-8")), inputs)
    for sp in b["run"]:
        x = s4.s4_of(sp)
        assert x["model_seed"] == 0
        assert (x.get("catenc_k") == SEL["k"]) == (sp.kind == "tree")
        plan = s4.RowPlan.from_dict(x["row_plan"])
        if not sp.name.startswith(("5-XGB", "6-XGB")):
            assert plan.steps[0] == "reduce"
        # 뽑은 키 안에 그 판 train 검출 행이 모두 있음
        for s in rn.SPLIT_ORDER:
            kept = set(s4.load_kept(inputs, plan.sel_dir, s))
            if plan.steps[0] != "reduce":
                tr = known[rn.fd.day_ints(known["d"]) <= rn.fd.SPLITS[s].train_end]
                assert set(tr.loc[tr["y"] == 1, "lims_data_key"]) <= kept
    _fake_run(s4env, s4env["log"], inputs / "s5_6" / "5-6-T",
              [rn.VariantSpec(**{**sp.to_dict(), "algorithm": "XGB", "kind": "tree"}) for sp in b["run"]])
    d56 = m56.main(_a(s4env, inputs / "s5_6" / "5-6-judge") + ["--config", str(c["s5_6_judge"])])
    st2 = s4.load_state(inputs / "logs" / "chain_state_5-6.json")
    part = d56["record"]["parts"][0]
    assert set(part["ranking"]) == set(names) and part["against"] == st["current_best"]["name"]
    assert st2["current_best"]["name"] == (part["top"] if part["adopted"] else st["current_best"]["name"])


def test_report_rows_same_pairs_x1(s4env):
    """①·4-c 보고: 1차 베이스라인·현재 최고 판 줄도 같은 평가 쌍(h 1)으로 거름 — 채점 쌍 수 검사 통과."""
    from lst import evaluate as ev

    m = load_script("s4")
    inputs = s4env["inputs"]
    j = inputs / "s4" / "4-judge"
    j.mkdir(parents=True)
    (j / "4_decision.json").write_text("{}", encoding="utf-8")
    cfg = json.loads(s4env["cfgs"]["s4_X1"].read_text(encoding="utf-8"))
    st = ev.SampleTable(s4env["st"].df.copy())       # 보고 프로세스처럼 새 시료 표(기대 쌍 수 캐시 없음)
    with s4.hooks(pair_filter=s4.h1_filter):
        rows = m.report_rows_for(inputs, {"flt": s4.h1_filter})(cfg, inputs, st, s4env["log"])
        assert len(rows) == 2
        for r in rows:
            for f, p in r.cv_preds.items():
                assert set(p["h"]) == {1}
            assert set(r.test_preds["h"]) == {1}
            cv = ev.score_cv(r.cv_preds, st, thr_preds=r.thr_preds)        # 쌍 수 검사(require_complete) 통과
            assert cv["n_pairs"] == sum(st.expected_pairs(f) for f in ev.CV_FOLDS)
