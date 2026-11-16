"""실행 틀 시험 — 가짜 입력·가짜 모델로 판 하나가 끝까지(fold 1~3·refit·채점·MLflow·건너뛰기·멈춘 판 다시·보고),
1차 스크립트 판 목록·1-3 정하기·0-10 MLflow 올리기. MLflow 는 로컬 임시 sqlite."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst import tracking as tk  # noqa: E402
from tests.test_evaluate import make_samples  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[1]
STUB = lambda path: {"note": "test stub"}  # noqa: E731  로컬 PC RAM 바닥값 대신


# ---------------------------------------------------------------- 가짜 입력·모델(인터페이스 약속 꼴)
class FakeAssembler:
    columns = ["h", "g", "p3_code", "ov"]

    def __init__(self, spec, split):
        self.spec, self.split, self.fitted = spec, split, False

    def fit(self, train_meta):
        assert "y" in train_meta.columns and (fd.to_days(train_meta["d"]) <= fd.SPLITS[self.split].train_end).all()
        for c in ("T", "E", "g", "R"):
            assert c in train_meta.columns
        self.codes = {v: i for i, v in enumerate(sorted(train_meta["p3"].unique()))}
        self.fitted = True
        return self

    def transform(self, meta):
        assert self.fitted
        return pd.DataFrame({"h": meta["h"].astype(np.float32), "g": meta["g"].astype(np.float32),
                             "p3_code": meta["p3"].map(self.codes).astype(np.float32),
                             "ov": (meta["src_route"] == "overseas").astype(np.float32)})


class LeakCheckAssembler(FakeAssembler):
    def transform(self, meta):
        if "y" in meta.columns and not getattr(self, "_train_pass", False):
            self._train_pass = True          # 첫 호출 = train 행(y 있음), 그 뒤 평가 쌍엔 y 가 없어야 함
        elif "y" in meta.columns:
            raise AssertionError("평가 쌍 행 표에 y 가 들어옴")
        return super().transform(meta)


class FakeModel:
    def __init__(self, spec, asm=None):
        assert asm is None or asm.fitted          # make_model(spec, fit 한 조립 객체)
        self.spec = spec

    def fit(self, X, y, *, seed, n_jobs, eval_set=None):
        from sklearn.linear_model import LogisticRegression

        assert seed == 42 and 1 <= n_jobs <= 8
        self.m = LogisticRegression(max_iter=200).fit(X.fillna(0).to_numpy(), y)
        return self

    def predict_proba(self, X, n_iter=None):
        return self.m.predict_proba(X.fillna(0).to_numpy())[:, 1].astype(np.float64)

    def save(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "coef.json").write_text(json.dumps(self.m.coef_.tolist()), encoding="utf-8")

    def applied_params(self):
        return self.m.get_params()


class BoomModel(FakeModel):
    def fit(self, X, y, **kw):
        raise MemoryError("가짜 메모리 오류")


@pytest.fixture()
def env(tmp_path):
    inputs = tmp_path / "results"
    p = inputs / "s0" / "0-1" / "apply"
    p.mkdir(parents=True)
    make_samples(n=1800, seed=3).to_parquet(p / "samples.parquet", index=False)
    (p / "12_incomplete_months.json").write_text(json.dumps({"incomplete": [["overseas", "2026-06"]]}), encoding="utf-8")
    tracker = tk.Tracker(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", artifact_root=(tmp_path / "art").as_uri())
    st = ev.SampleTable.read(p / "samples.parquet")
    return {"tmp": tmp_path, "inputs": inputs, "samples_path": p / "samples.parquet", "st": st, "tracker": tracker,
            "log": rn.ExperimentLog(inputs / "logs" / "experiment_log.jsonl")}


def spec_of(name="1-CAT-W020-sum", W=20, mode="sum", alg="CAT"):
    return rn.VariantSpec(name=name, stage="1-1", order="1차", kind="tree", algorithm=alg, W=W, mode=mode,
                          input_set="1차-트리", role="후보", train_count=4, judgement="해당 없음")


def ctx_of(env, out, asm=FakeAssembler, model=FakeModel, require_7a=False):
    tr = env["tracker"]
    return rn.RunContext(samples=env["st"], out_dir=out, code_root=CODE_ROOT, input_md5s={"x": "0"},
                         exp_log=env["log"], make_inputs=asm, make_model=model, n_jobs=8, tracker=tr,
                         experiment="listeria-1차", stage_run_id=tr.stage_run("listeria-1차", "1-1"),
                         check_resources=STUB, require_7a_logged=require_7a)


def test_code_changed_mid_stage_stops(env):
    """단계 시작 때 남긴 코드 사본(_code/code_md5.txt)과 지금 코드가 다르면 판을 시작하지 않음."""
    out = env["tmp"] / "s1" / "1-1"
    rn.snapshot_code(CODE_ROOT, out)
    (out / "_code" / "code_md5.txt").write_text("0" * 32, encoding="utf-8")
    with pytest.raises(rn.ResumeError, match="code_md5"):
        rn.run_variant(spec_of(), ctx_of(env, out))
    assert not (out / spec_of().name).exists()


def test_requires_prior_7a_log(env):
    """CV 채점 단계는 실험 로그에 0-10(7-a 격자) 기록이 있어야 시작(부록 A-12 셈)."""
    out = env["tmp"] / "s1" / "1-1"
    with pytest.raises(rn.ResumeError, match="0-10"):
        rn.run_variant(spec_of(), ctx_of(env, out, require_7a=True))
    env["log"].cv_scored(rn.TE7A_STAGE, "7-a-5y-ku050-kp050", 75, rn.TE7A_KIND)
    r = rn.run_variant(spec_of(), ctx_of(env, out, require_7a=True))
    assert r["status"] == "done" and r["selection_count"] == 75


def test_run_variant_end_to_end(env):
    out = env["tmp"] / "s1" / "1-1"
    s1, s2 = spec_of(), spec_of("1-XGB-W010-raw", 10, "raw", "XGB")
    r1 = rn.run_variant(s1, ctx_of(env, out, asm=LeakCheckAssembler))
    r2 = rn.run_variant(s2, ctx_of(env, out))
    assert r1["status"] == "done" and r1["selection_count"] == 0 and r2["selection_count"] == 1   # 그 판 앞까지
    v = out / s1.name
    for f in ("spec.json", "run_meta.json", "cv_scores.json", "done.json", "columns_fold1.json",
              "applied_params_refit.json", "pred_fold1.parquet", "pred_refit.parquet"):
        assert (v / f).exists(), f
    assert (v / "model_refit" / "coef.json").exists()
    pr = pd.read_parquet(v / "pred_refit.parquet")
    assert list(pr.columns) == list(ev.PRED_COLS) and len(pr) == env["st"].expected_pairs("refit")
    assert "y" not in pr.columns
    meta = cf.load_json_strict(v / "run_meta.json")
    assert meta["code_md5"] == cf.code_md5(CODE_ROOT) and "OMP_NUM_THREADS" in meta["threads"]
    # 저장 예측으로 재채점 = 같은 값(B-3)
    cv = ev.score_cv(rn.load_preds(v, ev.CV_FOLDS), env["st"])
    assert cv["cv_band_weighted"] == r1["cv_band_weighted"]
    # 끝난 판은 건너뜀(B-1)
    assert rn.run_variant(s1, ctx_of(env, out))["status"] == "skipped"
    # 같은 이름 다른 설정 → 멈춤
    with pytest.raises(rn.ResumeError):
        rn.run_variant(spec_of(W=30), ctx_of(env, out))
    # MLflow: 판 run tag·metric·artifact
    run = env["tracker"].client.get_run(r1["mlflow_run_id"])
    assert run.data.tags["판정"] == "해당 없음" and run.data.tags["역할"] == "후보" and run.info.status == "FINISHED"
    assert run.data.metrics["selection_count"] == 0 and "cv_band_weighted" in run.data.metrics
    assert "test_band_weighted" not in run.data.metrics          # test 는 보고 때만
    arts = [a.path for a in env["tracker"].client.list_artifacts(r1["mlflow_run_id"], "predictions")]
    assert "predictions/pred_refit.parquet" in arts

    # 보고(허가): test 채점·표·MLflow test 지표, 덜 찬 달 분리
    with ev.selection_guard(), pytest.raises(ev.TestPeekError):
        rn.report_stage(out_dir=out, stage="1-1", names=[s1.name, s2.name], samples=env["st"],
                        samples_path=env["samples_path"], permit=None, incomplete=set(), exp_log=env["log"])
    permit = ev.report_permit("1-1", "시험 보고")
    inc = ev.load_incomplete(env["inputs"] / "s0/0-1/apply/12_incomplete_months.json")
    p = rn.report_stage(out_dir=out, stage="1-1", names=[s1.name, s2.name], samples=env["st"],
                        samples_path=env["samples_path"], permit=permit, incomplete=inc, exp_log=env["log"],
                        tracker=env["tracker"], stage_run_id=ctx_of(env, out).stage_run_id)
    t = pd.read_csv(p, encoding="utf-8-sig")
    assert list(t.columns) == list(ev.TABLE_COLUMNS) and len(t) == 2
    assert t["test 구간 가중"].notna().all() and t["같은 validation 으로 고른 횟수"].tolist() == [0, 1]
    assert ev.INCOMPLETE_MARK in t.loc[0, "표시"] and ev.CAT_AUTO_LR_MARK in t.loc[0, "표시"]
    ts = rn._load_scores(v / "test_scores.json")
    assert ts["incomplete"]["n_pairs"] > 0
    run = env["tracker"].client.get_run(r1["mlflow_run_id"])
    assert "test_band_weighted" in run.data.metrics and run.data.metrics["test_incomplete_n_pairs"] > 0
    assert any(e["event"] == "test_reported" for e in env["log"].entries())


def test_stopped_variant_rerun(env):
    out = env["tmp"] / "s1" / "1-1"
    s = spec_of()
    with pytest.raises(MemoryError):
        rn.run_variant(s, ctx_of(env, out, model=BoomModel))
    assert (out / s.name / "failed.json").exists() and not (out / s.name / "done.json").exists()
    r = rn.run_variant(s, ctx_of(env, out))
    assert r["status"] == "done"
    moved = list((out / "_incomplete").iterdir())
    assert len(moved) == 1 and moved[0].name.startswith(s.name + "__")
    reruns = [e for e in env["log"].entries() if e["event"] == "rerun"]
    assert reruns and "MemoryError" in reruns[0]["reason"]
    runs = env["tracker"].client.search_runs([env["tracker"].experiment_id("listeria-1차")])
    assert sorted(r.info.status for r in runs if r.data.tags.get("판 이름") == s.name) == ["FAILED", "FINISHED"]


def test_isolated_stops_on_failure(env, tmp_path):
    script = tmp_path / "dummy.py"
    script.write_text("import sys\nname = sys.argv[sys.argv.index('--variant') + 1]\nsys.exit(3 if name == 'b' else 0)\n",
                      encoding="utf-8")
    with pytest.raises(rn.RunStopped, match="판 b"):
        rn.run_variants_isolated(script, ["a", "b", "c"], [], env["log"], "1-1")
    st = [e for e in env["log"].entries() if e["event"] == "stopped"]
    assert st[0]["name"] == "b" and st[0]["returncode"] == 3


def test_names_and_cv_count_idempotent(env):
    assert rn.name_stage1("CAT", 60, "sum") == "1-CAT-W060-sum" and rn.name_stage1("LSTM", 100) == "1-LSTM-W100"
    assert rn.dedupe_name("3-2a-CAT", ["3-2a-CAT", "3-2a-CAT-2"]) == "3-2a-CAT-3"
    log = env["log"]
    assert log.cv_scored("0-10", "7-a-5y-ku050-kp050", 75, "7-a 격자") == 0
    assert log.cv_scored("1-1", "x") == 75 and log.cv_scored("1-1", "x") == 75   # 1-1 첫 판 = 75
    assert log.cv_scored("1-1", "y") == 76


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, CODE_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage1_specs_and_list(env, capsys):
    m11, m12 = load_script("s1_1_trees"), load_script("s1_2_lstm")
    c11 = sc.load_cfg(CODE_ROOT / "configs/s1_1_trees.json", m11.SCHEMA_EXTRA)
    c12 = sc.load_cfg(CODE_ROOT / "configs/s1_2_lstm.json", m12.SCHEMA_EXTRA)
    n11 = [s.name for s in m11.specs(c11)]
    n12 = [s.name for s in m12.specs(c12)]
    assert len(n11) == 60 == len(set(n11)) and len(n12) == 10
    assert "1-XGB-W100-raw" in n11 and n12[0] == "1-LSTM-W010"
    bad = dict(c11, model_params={"depth": 3})
    with pytest.raises(cf.ConfigError):
        m11.specs(bad)
    names = m11.main(["--root", str(env["tmp"]), "--out", str(env["tmp"] / "o"), "--inputs", str(env["inputs"]),
                      "--config", str(CODE_ROOT / "configs/s1_1_trees.json"), "--list"])
    assert names == n11


def test_stage_main_variant_and_report(env, monkeypatch):
    """stage_main 으로 판 하나(--variant) → 보고(--report, 7-a 기준 줄 포함) — 가짜 입력·모델 주입."""
    m11 = load_script("s1_1_trees")
    tr = env["tracker"]
    cfgp = env["tmp"] / "cfg.json"
    c = json.loads((CODE_ROOT / "configs/s1_1_trees.json").read_text(encoding="utf-8"))
    cfgp.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    two = lambda cfg: m11.specs(cfg)[:2]  # noqa: E731
    fac = lambda cfg, rp, ir: sc.Factories(FakeAssembler, FakeModel, {"inputs.py": "0"})  # noqa: E731
    # 0-10 결과(가짜) + MLflow 올리기
    d010 = make_fake_0_10(env)
    m010 = load_script("s0_10_log_mlflow")
    res = m010.log_0_10(d010, env["st"], tr, env["log"], grid_size=75)
    assert res["selection_count"] == 0
    out = env["tmp"] / "s1" / "1-1"
    base = ["--root", str(env["tmp"]), "--out", str(out), "--inputs", str(env["inputs"]), "--config", str(cfgp)]
    kw = dict(script=CODE_ROOT / "scripts" / "s1_1_trees.py", description="t", schema_extra=m11.SCHEMA_EXTRA,
              specs_fn=two, factories_fn=fac, tracker_fn=lambda cfg: tr, check_resources=STUB)
    for n in [s.name for s in two(sc.load_cfg(cfgp, m11.SCHEMA_EXTRA))]:
        sc.stage_main(base + ["--variant", n], **kw)
    with pytest.raises(SystemExit):
        sc.stage_main(base + ["--report"], **kw)          # 이유 없이 보고 안 됨
    p = sc.stage_main(base + ["--report", "--reason", "시험"], **kw)
    t = pd.read_csv(p, encoding="utf-8-sig")
    assert t.loc[0, "단계·판"] == res["name"] and t.loc[0, "같은 validation 으로 고른 횟수"] == 0
    assert t["같은 validation 으로 고른 횟수"].tolist()[1:] == [75, 76]
    vr = tr.client.get_run(res["run_id"])
    assert "test_band_weighted" in vr.data.metrics            # 7-a test 는 1-1 보고 때 같은 run 에


def make_fake_0_10(env) -> Path:
    """0-10 산출물 꼴(te7a_selected.json·te7a_grid_cv.csv·te7a_cv_pred·te7a_test_pred)을 합성."""
    st = env["st"]
    d = env["inputs"] / "s0" / "0-10"
    d.mkdir(parents=True)
    name = "7-a-5y-ku050-kp050"
    rng = np.random.default_rng(9)
    parts, reports = [], {}
    for f in ev.CV_FOLDS:
        m = fd.scoring_sample_mask(st.d_days, fd.SPLITS[f])
        parts.append(pd.DataFrame({"fold": f, "lims_data_key": st.keys[m], "d": st.df["d"].to_numpy()[m],
                                   "p_hat": rng.random(int(m.sum()))}))
    cvp = pd.concat(parts, ignore_index=True)
    cvp.to_parquet(d / f"te7a_cv_pred_{name}.parquet", index=False)
    cv = ev.score_cv({f: ev.expand_sample_preds(cvp[cvp["fold"] == f], st, f) for f in ev.CV_FOLDS}, st)
    m = fd.scoring_sample_mask(st.d_days, fd.SPLITS["refit"])
    pd.DataFrame({"lims_data_key": st.keys[m], "p_hat": rng.random(int(m.sum()))}).to_parquet(
        d / f"te7a_test_pred_{name}.parquet", index=False)
    reports = {f: {"pr_auc_band_weighted": cv["fold_band_weighted"][f]} for f in ev.CV_FOLDS}
    cf.dump_json(d / "te7a_selected.json", {"selected": name, "length": "5y", "k_u": 50, "k_p3": 50,
                                             "fold_reports": reports, "test_scored": False})
    pd.DataFrame({"name": [f"g{i}" for i in range(75)]}).to_csv(d / "te7a_grid_cv.csv", index=False, encoding="utf-8-sig")
    return d


def test_decide_1_3():
    m13 = load_script("s1_3_decide")
    trees = [ev.Candidate(rn.name_stage1(a, W, mo), 0.01 + (0.001 if (a, W, mo) == ("LGB", 30, "raw") else 0), 4, False,
                          {"algorithm": a, "W": W, "mode": mo, "kind": "tree"})
             for a in ("XGB", "LGB", "CAT") for W in range(10, 101, 10) for mo in ("sum", "raw")]
    lstms = [ev.Candidate(rn.name_stage1("LSTM", W), 0.02 if W == 40 else 0.005, 4, False,
                          {"algorithm": "LSTM", "W": W, "mode": None, "kind": "dl"}) for W in range(10, 101, 10)]
    # W 40 트리 중 CAT-raw 가 근소 1위가 되게
    trees = [ev.Candidate(c.name, 0.0125 if c.name == "1-CAT-W040-raw" else c.cv, 4, False, c.extra) for c in trees]
    dec = m13.decide_1_3(trees, lstms, rescore=None)
    assert dec["baseline"] == "1-LSTM-W040" and dec["W"] == 40 and dec["W_DL"] == 40
    assert dec["tree_mode"] == "raw" and dec["tree_mode_from"] == "1-CAT-W040-raw" and dec["A"] == "CAT"
    tags = m13.selection_tags(dec)
    assert tags["1-LSTM-W040"].startswith("1차 베이스라인") and "A = CAT" in tags["1-CAT-W040-raw"]
    with pytest.raises(ev.TestPeekError):
        m13.decide_1_3([ev.Candidate(c.name, c.cv, 4, False, {**c.extra, "test_pooled": 0.1}) for c in trees], lstms, None)
