"""단계 2 준비(scripts/s2_prep.py) 시험 — 가짜 입력·가짜 모델·작은 합성 시료로 끝까지:
1차 결과(가짜) → 2-0 복사 → 2-1·2-1b-T·2-1b-L·2-R 판(stage_cli --variant) → 2-R1 중요도 → 고르기(동점 규칙) → 보고.
MLflow 는 로컬 임시 sqlite(서버·실제 자료 안 씀)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from lst import config as cf
from lst import evaluate as ev
from lst import importance as im
from lst import registry as rg
from lst import runner as rn
from lst import stage_cli as sc
from tests.test_evaluate import make_samples
from tests.test_runner import CODE_ROOT, STUB, env, load_script, make_fake_0_10  # noqa: F401  (env = fixture)

S2 = load_script("s2_prep")
CFG = {p: CODE_ROOT / "configs" / f for p, f in (("2-0", "s2_0_base.json"), ("2-1", "s2_1_k.json"),
                                                   ("2-1b-T", "s2_1b_tree.json"), ("2-1b-L", "s2_1b_lstm.json"),
                                                   ("2-R", "s2_R_research.json"), ("select", "s2_select.json"))}
DEC = {"baseline": "1-LSTM-W020", "baseline_cv": 0.1, "W": 20, "W_DL": 20, "tree_mode": "sum",
       "tree_mode_from": "1-XGB-W020-sum", "A": "XGB", "A_variant": "1-XGB-W020-sum"}

# 판마다 신호 세기(클수록 CV 높음 — 같은 잡음이라 세기에 단조): 2-1 은 k030 1위, 2-1b-T 는 N05 = N15 동점 1위(이름 순 N05),
# 2-1b-L 은 N30 1위
STRENGTH = {"1-XGB-W020-sum": 0.9, "1-LSTM-W020": 0.8,
            "2-1-k010": 0.6, "2-1-k030": 1.4, "2-1-k050": 1.0, "2-1-k100": 0.8, "2-1-k200": 0.4,
            "2-1b-T-N05": 1.2, "2-1b-T-N07": 0.6, "2-1b-T-N15": 1.2, "2-1b-T-N30": 0.9,
            "2-1b-L-N05": 0.5, "2-1b-L-N07": 0.7, "2-1b-L-N15": 0.9, "2-1b-L-N30": 1.3}
_FULL = make_samples(n=1800, seed=3)          # env 시료와 같은 것(정답 전 기간 — 가짜 신호 만들기에만)
YMAP = dict(zip(_FULL["lims_data_key"], _FULL["y"].astype(float)))


def _noise(keys, h, col) -> np.ndarray:
    s = pd.Series([f"{k}|{int(x)}|{col}" for k, x in zip(keys, h)])
    u = (pd.util.hash_pandas_object(s, index=False).to_numpy().astype(np.float64) + 0.5) / 2.0 ** 64
    return ndtri(u)


class SigAsm:
    """판 설정에 따라 칼럼이 정해지는 가짜 조립 객체: 2-R 판 = h·g + 그 판의 리서치 칼럼(t3 에만 신호),
    그 밖 = h·g + 'sig'(신호 세기 = STRENGTH)."""

    def __init__(self, spec, split=None):
        self.spec, self.split = spec, split
        units = spec.extra.get("version", {}).get("include_units", [])
        self.feats = list(units) if units else ["sig"]
        self.columns = ["h", "g"] + self.feats

    def fit(self, meta=None):
        return self

    def transform(self, meta):
        keys = meta["lims_data_key"].astype(str).to_numpy()
        h = meta["h"].to_numpy()
        y = np.array([YMAP[k] for k in keys])
        out = {"h": meta["h"].astype(np.float32).to_numpy(), "g": meta["g"].astype(np.float32).to_numpy()}
        for c in self.feats:
            s = STRENGTH.get(self.spec.name, 1.0) if c == "sig" else (1.0 if c == "t3" else 0.0)
            out[c] = (s * y + _noise(keys, h, c)).astype(np.float32)
        return pd.DataFrame(out)


class SumModel:
    def fit(self, X, y, *, seed, n_jobs, eval_set=None):
        assert seed == 42
        return self

    def predict_proba(self, X, n_iter=None):
        z = X.drop(columns=["h", "g"]).to_numpy(dtype=np.float64).sum(axis=1)
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path):
        Path(path).write_text("{}", encoding="utf-8")


FAC = lambda cfg, rp, ir: sc.Factories(SigAsm, lambda sp, asm: SumModel(), {"fake": "0"})  # noqa: E731
LOOKUP = {**{u: u for u in rg.RESEARCH_UNITS}, "h": None, "g": None}


def fake_nodes(cfg, root, inputs_root, vdir, spec):
    out = {}
    for f in ev.CV_FOLDS:
        a = SigAsm(spec, f)
        out[f] = im.Leaf(f"{spec.name}-{f}", a, SumModel(), layout=im.layout_from_names(a.columns, LOOKUP))
    return out


def _stage1(env):
    """1차 결과(가짜): 1-1 A 판·1-2 LSTM 판 + 1-3 결정 json + 0-10(7-a) 기록."""
    tr = env["tracker"]
    env["log"].cv_scored(rn.TE7A_STAGE, "7-a-5y-ku050-kp050", 75, rn.TE7A_KIND)
    make_fake_0_10(env)
    for stage, sub, sp in (("1-1", "s1/1-1", rn.VariantSpec(name="1-XGB-W020-sum", stage="1-1", order="1차", kind="tree",
                                                            algorithm="XGB", W=20, mode="sum", input_set="1차-트리",
                                                            role="후보", train_count=4, judgement="해당 없음",
                                                            extra={"version": {"include_ids": ["WEATHER"]}})),
                           ("1-2", "s1/1-2", rn.VariantSpec(name="1-LSTM-W020", stage="1-2", order="1차", kind="dl",
                                                            algorithm="LSTM", W=20, mode=None, input_set="1차-딥러닝",
                                                            role="후보", train_count=4, judgement="해당 없음",
                                                            extra={"version": {"include_ids": ["WEATHER"]}}))):
        ctx = rn.RunContext(samples=env["st"], out_dir=env["inputs"] / sub, code_root=CODE_ROOT, input_md5s={"x": "0"},
                            exp_log=env["log"], make_inputs=SigAsm, make_model=lambda s, a: SumModel(), n_jobs=8,
                            tracker=tr, experiment="listeria-1차", stage_run_id=tr.stage_run("listeria-1차", stage),
                            check_resources=STUB, require_7a_logged=False)
        assert rn.run_variant(sp, ctx)["status"] == "done"
    p = env["inputs"] / "s1" / "1-3"
    p.mkdir(parents=True)
    (p / "1-3_decision.json").write_text(json.dumps(DEC), encoding="utf-8")


def _argv(env, part, *extra):
    out = env["inputs"] / json.loads(CFG["select"].read_text(encoding="utf-8"))["dirs"][part]
    return ["--root", str(env["tmp"]), "--inputs", str(env["inputs"]), "--out", str(out),
            "--config", str(CFG[part]), *extra]


INJ = dict(factories_fn=FAC, check_resources_fn=STUB, before_variant_fn=lambda cfg: None)


def test_specs_names_counts():
    dec = dict(DEC)
    n = {}
    for part in S2.TRAIN_SUBS:
        cfg = sc.load_cfg(CFG[part], S2.TRAIN_SCHEMA)
        sp = S2.part_specs(part, cfg, dec)
        n[part] = [s.name for s in sp]
        assert all(s.role == "비교만" and s.judgement == "해당 없음" and s.params == {} and s.train_count == 4 for s in sp)
        assert all(s.order == "2차" for s in sp)
    assert n["2-1"] == ["2-1-k010", "2-1-k030", "2-1-k050", "2-1-k100", "2-1-k200"]
    assert n["2-1b-T"] == ["2-1b-T-N05", "2-1b-T-N07", "2-1b-T-N15", "2-1b-T-N30"]
    assert n["2-1b-L"] == ["2-1b-L-N05", "2-1b-L-N07", "2-1b-L-N15", "2-1b-L-N30"]
    assert n["2-R"] == [f"2-R{i}" for i in range(1, 11)]
    total = sum(len(v) for v in n.values())
    assert total * 4 == 20 + 32 + 40          # 부록 A-9 학습 횟수(2-1 20·2-1b 32·2-R 40)
    cfg = sc.load_cfg(CFG["2-R"], S2.TRAIN_SCHEMA)
    r = {s.name: s for s in S2.part_specs("2-R", cfg, dec)}
    assert r["2-R1"].extra["version"]["include_units"] == list(rg.RESEARCH_UNITS)
    assert r["2-R10"].extra["version"]["include_units"] == ["L1"] and r["2-R2"].extra["version"]["include_units"] == ["t3"]
    lt = S2.part_specs("2-1b-L", sc.load_cfg(CFG["2-1b-L"], S2.TRAIN_SCHEMA), dec)
    assert all(s.kind == "dl" and s.algorithm == "LSTM" and s.W == 20 and s.mode is None for s in lt)
    # 판 설정 → 등록부 칼럼: 2-1 = HIS 칼럼(k 이름), 2-1b-T = 차감 칼럼(N 이름), 2-R1 = 리서치 9칼럼
    from lst import wiring as wr
    k30 = next(s for s in S2.part_specs("2-1", sc.load_cfg(CFG["2-1"], S2.TRAIN_SCHEMA), dec) if s.name == "2-1-k030")
    cols = [c.name for c in rg.tree_columns(wr.version_spec(k30))]
    assert "hist_rate5y_unit_k030" in cols or any(c.endswith("_k030") for c in cols)
    assert not any("anom" in c for c in cols)
    r1 = [c.name for c in rg.tree_columns(wr.version_spec(r["2-R1"]))]
    assert set(rg.RESEARCH_UNITS) <= set(r1) and not any(c.startswith("hist_") for c in r1)
    # 같은 입력 판이 없음 → 묶음 = 판 하나씩(구현 세부 8)
    assert all(len(g) == 1 for g in rn.group_specs([s for p in S2.TRAIN_SUBS for s in
                                                   S2.part_specs(p, sc.load_cfg(CFG[p], S2.TRAIN_SCHEMA), dec)]))


def test_config_checks():
    cfg = dict(sc.load_cfg(CFG["2-1"], S2.TRAIN_SCHEMA))
    for bad in ({"k_values": [10, 30]}, {"model_params": {"depth": 3}}, {"n_jobs": 4}, {"stage": "2-1"}):
        with pytest.raises(cf.ConfigError):
            S2.part_specs("2-1", {**cfg, **bad}, DEC)
    c2 = dict(sc.load_cfg(CFG["2-1b-L"], S2.TRAIN_SCHEMA))
    with pytest.raises(cf.ConfigError):
        S2.part_specs("2-1b-L", {**c2, "gpu_memory_fraction": 0.5}, DEC)
    with pytest.raises(cf.ConfigError):
        S2.part_specs("2-1b-T", {**dict(sc.load_cfg(CFG["2-1b-T"], S2.TRAIN_SCHEMA)), "n_ma_values": [5, 7, 15, 31]}, DEC)


def test_choose_tie_rules():
    c = lambda n, v, tc=4, mt=False: ev.Candidate(n, v, tc, mt, {})  # noqa: E731
    # 소수 4자리 반올림 뒤 같음 → 학습 횟수 → 이름 순(작은 N·k 가 앞)
    s = S2.choose([c("2-1b-T-N15", 0.123449), c("2-1b-T-N05", 0.12345), c("2-1b-T-N30", 0.1200)], None)
    assert s["selected"] == "2-1b-T-N05"
    s = S2.choose([c("2-1-k030", 0.20001), c("2-1-k010", 0.2)], None)
    assert s["selected"] == "2-1-k010"
    s = S2.choose([c("2-1-k030", 0.2001), c("2-1-k010", 0.2)], None)
    assert s["selected"] == "2-1-k030"
    # 동점 많은 판이 1위와 0.001 안 → 다시 채점 기록, 1위가 동점 많으면 알림
    s = S2.choose([c("2-1-k030", 0.2005, mt=True), c("2-1-k010", 0.2)], lambda n: {"2-1-k030": 0.19, "2-1-k010": 0.2}[n])
    assert s["selected"] == "2-1-k030" and s["rescored"] and s["rescore_same_top"] is False and s["alerts"]
    with pytest.raises(ev.TestPeekError):     # test 칸이 들어오면 멈춤(§1-3)
        S2.choose([ev.Candidate("2-1-k010", 0.1, 4, False, {"test_band_weighted": 0.2})], None)


def test_end_to_end(env):
    tr = env["tracker"]
    tf = lambda cfg: tr  # noqa: E731
    _stage1(env)

    # ---- 2-0: 학습 없이 복사
    d = S2.main(_argv(env, "2-0"), tracker_fn=tf)
    v20 = env["inputs"] / "s2/2-0/2-0-XGB"
    src = env["inputs"] / "s1/1-1/1-XGB-W020-sum"
    assert d["cv_band_weighted"] == rn.read_json(src / "done.json")["cv_band_weighted"]
    assert (v20 / "cv_scores.json").read_bytes() == (src / "cv_scores.json").read_bytes()
    assert not list(v20.glob("pred_*.parquet"))                     # 예측은 위치만
    s = rn.read_json(v20 / "source.json")
    assert s["dir"] == "s1/1-1/1-XGB-W020-sum" and set(s["files_md5"]) == set(S2.SOURCE_FILES)
    run = tr.client.get_run(d["mlflow_run_id"])
    assert run.data.tags["바탕 판"] == "1-XGB-W020-sum" and run.data.tags["판정"] == "해당 없음"
    assert run.data.tags["역할"] == "비교만" and run.data.tags["단계"] == "2-0"
    assert run.data.params["source_variant"] == "1-XGB-W020-sum"
    assert run.data.metrics["cv_band_weighted"] == pytest.approx(d["cv_band_weighted"], abs=0)
    parent = tr.client.get_run(run.data.tags["mlflow.parentRunId"])
    assert parent.info.run_name == "2" and parent.info.experiment_id == tr.experiment_id("listeria-2차")
    assert S2.main(_argv(env, "2-0"), tracker_fn=tf)["mlflow_run_id"] == d["mlflow_run_id"]   # 다시 = 건너뜀
    n_before = len([e for e in env["log"].entries() if e["event"] == "cv_scored"])

    # ---- 2-1·2-1b·2-R: 판마다(--variant)
    names = {}
    for part in S2.TRAIN_SUBS:
        names[part] = S2.main(_argv(env, part, "--list"), tracker_fn=tf, **INJ)
        for n in names[part]:
            r = S2.main(_argv(env, part, "--variant", n), tracker_fn=tf, **INJ)
            assert r["status"] == "done"
    assert len([e for e in env["log"].entries() if e["event"] == "cv_scored"]) == n_before + 23
    r = tr.client.get_run(rn.read_json(env["inputs"] / "s2/2-1/2-1-k030/done.json")["mlflow_run_id"])
    assert r.data.tags["단계"] == "2-1" and r.data.tags["역할"] == "비교만"
    assert tr.client.get_run(r.data.tags["mlflow.parentRunId"]).info.run_name == "2"
    # 2-1b-T 의 N05·N15 는 같은 예측(신호 세기 같음) → CV 같음
    cvs = {n: rn.read_json(env["inputs"] / "s2/2-1b-T" / n / "done.json")["cv_band_weighted"] for n in names["2-1b-T"]}
    assert cvs["2-1b-T-N05"] == cvs["2-1b-T-N15"]

    # ---- 2-R1 중요도(가짜 fold 모델 — 섞기 전 점수 = 판 fold 점수 확인 포함)
    cfg = sc.load_cfg(CFG["2-R"], S2.TRAIN_SCHEMA)
    out2r = env["inputs"] / "s2/2-R"
    specs = S2.part_specs("2-R", cfg, DEC)
    p = S2.run_2r1_importance(cfg, env["tmp"], env["inputs"], out2r, specs, nodes_fn=fake_nodes,
                              tracker=S2.PartTracker(tr, "2-R"), M=2, check=lambda: None)
    summ = pd.read_csv(p / "importance_summary.csv", encoding="utf-8-sig")
    assert len(summ) == 9 + 1 + 5                                  # 칼럼 9·리서치 전체·기상 그룹 5(기온·강수·바람·햇빛·습도)
    assert set(summ.loc[summ["target_kind"] == "group", "target"]) == {"기온", "강수", "바람", "햇빛", "습도"}
    t3 = summ[(summ["target_kind"] == "column") & (summ["target"] == "t3")].iloc[0]
    assert t3["mean_I"] > 0                                        # 신호 칼럼
    assert S2.run_2r1_importance(cfg, env["tmp"], env["inputs"], out2r, specs, nodes_fn=fake_nodes, M=2,
                                 check=lambda: None) == p          # 다시 = 건너뜀
    rid = rn.read_json(out2r / "2-R1" / "done.json")["mlflow_run_id"]
    assert any(a.path.startswith("importance") for a in tr.client.list_artifacts(rid))

    # ---- 보고는 고르기 전엔 안 됨(§1-3)
    with pytest.raises(SystemExit):
        S2.main(_argv(env, "select", "--report", "--reason", "시험"), tracker_fn=tf)

    # ---- 고르기(CV 만)
    sel = S2.main(_argv(env, "select"), tracker_fn=tf)
    assert sel["k"] == 30 and sel["k_variant"] == "2-1-k030"
    assert sel["n_ma"] == 5 and sel["n_ma_variant"] == "2-1b-T-N05"        # N05 = N15 동점 → 이름 순
    assert sel["n_ma_dl"] == 30 and sel["n_ma_dl_variant"] == "2-1b-L-N30"
    assert sel["W"] == 20 and sel["A"] == "XGB"
    saved = rn.read_json(env["inputs"] / "s2/2-select" / S2.SELECTION_FILE)
    assert saved["k"] == 30 and saved["n_ma"] == 5 and saved["n_ma_dl"] == 30
    sr = tr.find_run("listeria-2차", "2", None).info.run_id
    tag = lambda n: tr.find_run("listeria-2차", n, sr).data.tags.get("선택")  # noqa: E731
    assert tag("2-1-k030") == "k = 30" and tag("2-1b-T-N05") == "N_ma(트리) = 5" and tag("2-1b-L-N30") == "N_ma(딥러닝) = 30"
    assert tag("2-1-k010") is None
    with pytest.raises(FileExistsError):
        S2.main(_argv(env, "select"), tracker_fn=tf)
    assert not any(e["event"] == "test_reported" for e in env["log"].entries())

    # ---- 보고(test 채점, 하위 단계마다 표)
    rep = S2.main(_argv(env, "select", "--report", "--reason", "시험 보고"), tracker_fn=tf)
    t21 = pd.read_csv(rep["2-1"], encoding="utf-8-sig")
    assert t21["단계·판"].tolist() == ["7-a-5y-ku050-kp050", "1-LSTM-W020", "2-0-XGB"] + names["2-1"]
    assert t21["test 구간 가중"].notna().all()
    tl = pd.read_csv(rep["2-1b-L"], encoding="utf-8-sig")
    assert tl["단계·판"].tolist()[:2] == ["7-a-5y-ku050-kp050", "1-LSTM-W020"] and "2-0-XGB" not in tl["단계·판"].tolist()
    dl = pd.read_csv(rep["2-R_vs_2-0"], encoding="utf-8-sig")
    assert dl["판"].tolist() == [f"2-R{i}" for i in range(1, 11)]
    t2r = pd.read_csv(rep["2-R"], encoding="utf-8-sig").set_index("단계·판")
    assert dl.loc[0, "CV 변화(2-0 대비)"] == pytest.approx(t2r.loc["2-R1", "CV 구간 가중(평균)"]
                                                        - t2r.loc["2-0-XGB", "CV 구간 가중(평균)"])
    run = tr.client.get_run(d["mlflow_run_id"])
    assert "test_band_weighted" in run.data.metrics                  # 2-0 판 run 에 test 지표(한 번)
    hist = tr.client.get_metric_history(d["mlflow_run_id"], "test_band_weighted")
    assert len(hist) == 1
    arts = [a.path for a in tr.client.list_artifacts(sr, "tables")]
    assert {"tables/2-1", "tables/2-1b-T", "tables/2-1b-L", "tables/2-R"} <= set(arts)
    with pytest.raises(FileExistsError):                             # 보고는 한 번
        S2.main(_argv(env, "select", "--report", "--reason", "또"), tracker_fn=tf)
