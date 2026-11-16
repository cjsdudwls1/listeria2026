"""s_final(최종 모델 고르기·최종 보고) 시험 — 합성 시료·가짜 입력·가짜 모델로 판 폴더를 만들고 가짜 사슬 상태·사슬 기록으로.
최종 고르기(1차 베이스라인 이김·사슬 이김·동점), 변화 표 줄 순서·변화 값·fold 1 알림, --report 없이 test 정답 안 읽음,
보조 표(품목 묶음·3분위), 최종 모델 안 순열 중요도 호출 경로(가짜 노드)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from lst import evaluate as ev  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import runner as rn  # noqa: E402
from tests.test_runner import STUB, FakeAssembler, FakeModel, env  # noqa: E402,F401

CODE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s_final", CODE / "scripts" / "s_final.py")
sf = importlib.util.module_from_spec(_spec)
sys.modules["s_final"] = sf
_spec.loader.exec_module(sf)


class ResearchAsm(FakeAssembler):
    columns = ["h", "g", "t3", "L1"]

    def transform(self, meta):
        X = super().transform(meta)
        return X.rename(columns={"p3_code": "t3", "ov": "L1"})


class NoisyModel(FakeModel):
    """split 마다 정한 만큼 행 함수(입력만으로 정해짐) 잡음을 섞음 — fold 점수를 일부러 바꿈."""

    def __init__(self, spec, asm=None):
        super().__init__(spec, asm)
        v = dict(spec.extra.get("noise", {})).get(asm.split, 0.0) if asm is not None else 0.0
        self.const = v == "const"
        self.a = 0.0 if self.const else float(v)

    def predict_proba(self, X, n_iter=None):
        p = super().predict_proba(X)
        if self.const:                      # 상수 예측 = 그 fold 점수를 검출 비율 근처로 떨어뜨림
            return np.full(len(X), 0.5)
        z = X.fillna(0).to_numpy(dtype=np.float64).sum(axis=1)
        r = np.abs(np.sin(z * 12.9898 + 0.5)) % 1.0
        return (1 - self.a) * p + self.a * r


def _spec(name, stage, order="2차", noise=None, train_count=4):
    return rn.VariantSpec(name=name, stage=stage, order=order, kind="tree", algorithm="CAT", W=20, mode="sum",
                          input_set="합성", role="후보", train_count=train_count, judgement=None,
                          extra={"version": {"include_ids": ["WEATHER"]}, "noise": dict(noise or {})})


def _run(env, out, spec):
    ctx = rn.RunContext(samples=env["st"], out_dir=out, code_root=CODE, input_md5s={"x": "0"}, exp_log=env["log"],
                        make_inputs=ResearchAsm, make_model=NoisyModel, n_jobs=8, tracker=env["tracker"],
                        experiment="listeria-1차", stage_run_id=env["tracker"].stage_run("listeria-1차", spec.stage),
                        check_resources=STUB, require_7a_logged=False)
    return rn.run_group([spec], ctx)[0]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """판 폴더·사슬 상태·사슬 기록·설정 — 모듈에서 한 번."""
    import lst.tracking as tk
    from tests.test_evaluate import make_samples

    tmp = tmp_path_factory.mktemp("final")
    inputs = tmp / "results"
    p = inputs / "s0" / "0-1" / "apply"
    p.mkdir(parents=True)
    make_samples(n=1800, seed=3).to_parquet(p / "samples.parquet", index=False)
    (p / "12_incomplete_months.json").write_text(json.dumps({"incomplete": [["overseas", "2026-06"]]}), encoding="utf-8")
    tracker = tk.Tracker(f"sqlite:///{(tmp / 'mlflow.db').as_posix()}", artifact_root=(tmp / "art").as_uri())
    e = {"tmp": tmp, "inputs": inputs, "st": ev.SampleTable.read(p / "samples.parquet"), "tracker": tracker,
         "log": rn.ExperimentLog(inputs / "logs" / "experiment_log.jsonl"), "samples_path": p / "samples.parquet"}
    V = {
        "1-CAT-W020-sum": ("s1/1-1", _spec("1-CAT-W020-sum", "1-1", "1차", {"fold3": 0.1})),
        "3-0b-CAT-end": ("s3/3-0b", _spec("3-0b-CAT-end", "3-0b")),
        "4-a-CAT": ("s4", _spec("4-a-CAT", "4")),
        "8-T-CAT": ("s8", _spec("8-T-CAT", "8", noise={"fold1": "const"})),
        "2-0-CAT": ("s2/2-0", _spec("2-0-CAT", "2-0")),
        "2-R1": ("s2/2-R", _spec("2-R1", "2", noise={"fold2": 0.2})),
        "2-R2": ("s2/2-R", _spec("2-R2", "2", noise={"fold2": 0.1})),
    }
    V["tie"] = ("s8b", _spec("8-T-CAT", "8b", "1차", {"fold3": 0.1}))  # 베이스라인과 같은 예측(이름만 다름)
    for k, (rel, sp) in V.items():
        _run(e, inputs / rel, sp)
    (inputs / "s1" / "1-3").mkdir(parents=True)
    (inputs / "s1" / "1-3" / "1-3_decision.json").write_text(json.dumps(
        {"baseline": "1-CAT-W020-sum", "W": 20, "W_DL": 20, "tree_mode": "sum", "A": "CAT",
         "A_variant": "1-CAT-W020-sum"}), encoding="utf-8")
    (inputs / "s3" / "3-0b" / "3-0b_decision.json").write_text(json.dumps(
        {"target_structure": "CAT", "adopted": True, "drop_groups": ["습도"]}), encoding="utf-8")
    imp = inputs / "s2" / "2-R" / "2-R1_importance"
    imp.mkdir(parents=True)
    pd.DataFrame({"target_kind": ["column", "bundle"], "target": ["t3", "리서치 변수 전체"], "mean_I": [0.01, 0.02],
                  "pi": [0.7, 0.9]}).to_csv(imp / "importance_summary.csv", index=False, encoding="utf-8-sig")
    lines = [{"step": "3-0b", "judgement": "출발", "before": None, "after": "옛-순서-줄"},     # 옛 순서 줄 — 안 읽음
             {"step": "3-0", "judgement": "출발", "before": None, "after": "3-0b-CAT-end"},
             {"step": "⑥⑦", "judgement": "버림", "candidate": "x6-LSTMA", "before": "3-0b-CAT-end",
              "after": "3-0b-CAT-end"},
             {"step": "4", "before": "3-0b-CAT-end", "after": "4-a-CAT",
              "parts": [{"kind": "쌓는 후보", "top": "4-a-CAT", "adopted": True}]},
             {"step": "5-6", "before": "4-a-CAT", "after": "4-a-CAT",
              "parts": [{"kind": "쌓는 후보", "top": "5-4-a-CAT", "adopted": False}]},
             {"step": "8-T", "before": "4-a-CAT", "after": "8-T-CAT",
              "parts": [{"kind": "8-T", "top": "8-T-CAT", "adopted": True}]}]
    (inputs / "logs" / "chain_log.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                                                     encoding="utf-8")
    e["V"] = V
    return e


def _best(name, rel):
    return {"name": name, "kind": "tree", "dir": f"{rel}/{name}", "replacement": False,
            "members": [{"name": name, "dir": f"{rel}/{name}", "anomaly": None}]}


def _state(w, path, name, rel):
    st = {"step": "8", "current_best": _best(name, rel), "tree_target": _best(name, rel), "tree_top3": [],
          "tree_rank_3_0b": {}, "history": []}
    p = w["inputs"] / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    return path


def _cfg(w, tag, state_rel):
    cfg = {"plan": "시험", "stage": "최종", "order": "2차", "experiment": "listeria-2차",
           "samples": "s0/0-1/apply/samples.parquet", "incomplete_months": "s0/0-1/apply/12_incomplete_months.json",
           "experiment_log": "logs/experiment_log.jsonl", "use_mlflow": True, "mlflow_uri": "sqlite:///x",
           "code_reviewed": False, "n_jobs": 8, "seed": 42, "decision_1_3": "s1/1-3/1-3_decision.json",
           "dir_1_1": "s1/1-1", "dir_1_2": "s1/1-2", "chain_state": state_rel, "chain_log": "logs/chain_log.jsonl",
           "decision_3_0b": "s3/3-0b/3-0b_decision.json", "dir_2_0": "s2/2-0/2-0-CAT", "dir_2_R": "s2/2-R",
           "search_dirs": ["s3/3-0b", "s4", "s8"], "importance_M": 2, "context": {}}
    p = w["tmp"] / f"cfg_{tag}.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return p


def _argv(w, tag, cfgp, *extra):
    return ["--root", str(w["tmp"]), "--out", str(w["inputs"] / "final" / tag), "--inputs", str(w["inputs"]),
            "--config", str(cfgp), *extra]


def _cv(w, rel, name):
    return rn._load_scores(w["inputs"] / rel / name / "cv_scores.json")


def _no_truth(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("test 정답을 읽음")
    monkeypatch.setattr(ev.TestTruth, "load", classmethod(lambda cls, *a, **k: boom()))


def test_select_baseline_wins_and_no_test(world, monkeypatch):
    w = world
    _no_truth(monkeypatch)
    st = _state(w, "s8/state_a.json", "8-T-CAT", "s8")
    d = sf.main(_argv(w, "a", _cfg(w, "a", st)))
    b, c = _cv(w, "s1/1-1", "1-CAT-W020-sum")["cv_band_weighted"], _cv(w, "s8", "8-T-CAT")["cv_band_weighted"]
    exp = "1-CAT-W020-sum" if round(b, 4) >= round(c, 4) else "8-T-CAT"
    assert exp == "1-CAT-W020-sum"              # 8-T 는 fold 1 상수 예측 — 베이스라인이 이김
    assert d["final"] == exp and d["final_source"] == "1차 베이스라인"
    assert "test" not in json.dumps(d["selection"]).lower()
    # 순열 중요도 모드도 test 정답 안 읽음 — 지원 안 되는 판 경로(가짜): unsupported.json
    def nodes_unsup(*a, **k):
        raise sf.Unsupported("가짜 — 지원 안 함")
    sf.main(_argv(w, "a", _cfg(w, "a", st), "--importance"), nodes_fn=nodes_unsup)
    assert (w["inputs"] / "final" / "a" / "importance" / "unsupported.json").exists()


def test_select_chain_wins(world):
    w = world
    st = _state(w, "s4/state_b.json", "4-a-CAT", "s4")
    d = sf.main(_argv(w, "b", _cfg(w, "b", st)))
    b, c = _cv(w, "s1/1-1", "1-CAT-W020-sum")["cv_band_weighted"], _cv(w, "s4", "4-a-CAT")["cv_band_weighted"]
    assert round(c, 4) > round(b, 4)            # 베이스라인은 fold 3 잡음
    assert d["final"] == "4-a-CAT" and d["final_source"] == "사슬 마지막 현재 최고 판"
    assert d["final_def"]["members"][0]["dir"] == "s4/4-a-CAT"


def test_select_tie_name_order(world):
    w = world
    st = _state(w, "s8b/state_c.json", "8-T-CAT", "s8b")
    assert _cv(w, "s8b", "8-T-CAT")["cv_band_weighted"] == _cv(w, "s1/1-1", "1-CAT-W020-sum")["cv_band_weighted"]
    d = sf.main(_argv(w, "c", _cfg(w, "c", st)))
    assert d["final"] == "1-CAT-W020-sum"       # 같으면 학습 횟수(4 = 4) → 이름 순(§1-2)
    assert d["selection"]["ranking"] == ["1-CAT-W020-sum", "8-T-CAT"]


def test_report_change_table_aux_research_mlflow(world):
    w = world
    st = _state(w, "s8/state_d.json", "8-T-CAT", "s8")
    cfgp = _cfg(w, "d", st)
    sf.main(_argv(w, "d", cfgp))
    tr = w["tracker"]
    rdir = sf.main(_argv(w, "d", cfgp, "--report", "--reason", "시험 보고"), tracker_fn=lambda c: tr)
    ch = pd.read_csv(rdir / "change_table.csv", encoding="utf-8-sig")
    assert list(ch["구분"]) == ["1차 베이스라인", "3-0 1위(첫 현재 최고 판)", "채택한 기법(4)", "튜닝(8-T)", "최종 모델"]
    assert list(ch["판"]) == ["1-CAT-W020-sum", "3-0b-CAT-end", "4-a-CAT", "8-T-CAT", "1-CAT-W020-sum"]
    v = ch["val(CV 구간 가중)"].to_numpy()
    assert np.allclose(ch["val 변화(직전 대비)"].to_numpy()[1:], np.diff(v))
    assert np.allclose(ch["test 변화(직전 대비)"].to_numpy()[1:], np.diff(ch["test 구간 가중"].to_numpy()))
    alerts = ch["fold 1 알림"].fillna("").tolist()
    assert alerts[3] == "채택했는데 fold 1 단독으로 안 오름"   # 8-T: fold 1 잡음
    assert alerts[0] == alerts[1] == alerts[4] == ""
    assert ch.loc[3, "fold 1"] < ch.loc[2, "fold 1"]
    # 최종 test 전체 지표·h 별 표·보조 표
    t = json.loads((rdir / "final_test_scores.json").read_text(encoding="utf-8"))
    assert t["final"] == "1-CAT-W020-sum" and "threshold_metrics" in t["test"]["report"]
    hm = pd.read_csv(rdir / "final_h_metrics.csv", encoding="utf-8-sig")
    assert (hm["split"] == "test").all() and len(hm) == 60
    tier = pd.read_csv(rdir / "aux_unit_tertile.csv", encoding="utf-8-sig")
    prod = pd.read_csv(rdir / "aux_product_route.csv", encoding="utf-8-sig")
    tot = prod[(prod["국내·해외"] == "전체")]["n_pairs"].sum()
    assert tier["n_pairs"].sum() == tot > 0
    rt = pd.read_csv(rdir / "research_table.csv", encoding="utf-8-sig")
    r3 = rt[rt["칼럼"] == "t3"].iloc[0]
    b20 = _cv(w, "s2/2-0", "2-0-CAT")["cv_band_weighted"]
    assert np.isclose(r3["CV 변화(기준 대비)"], _cv(w, "s2/2-R", "2-R2")["cv_band_weighted"] - b20)
    assert r3["변수별 중요도(모든 리서치 칼럼 넣은 판)"] == 0.01 and r3["최종 모델에 남음"] == "예"
    assert "안 거친" in r3["최종 모델 구조의 변수 빼기 결과"]            # 최종 = 1차 베이스라인(v78 A-10)
    notes = json.loads((rdir / "research_notes.json").read_text(encoding="utf-8"))
    assert len(notes["sentences"]) == 9 and "Murphy 2023" in notes["sentences"][0]
    iv = pd.read_csv(rdir / "input_variables.csv", encoding="utf-8-sig")
    assert set(iv["칼럼"]) == {"h", "g", "t3", "L1"}
    assert iv.set_index("칼럼").loc["h", "ID"] == "TIME"   # 가짜 칼럼 t3·L1 은 판 version(WEATHER 만)에 없어 ID 빈칸
    # MLflow: 최종 판 run tag '최종', 사슬 run 에 변화 표
    rid = rn.read_json(w["inputs"] / "s1" / "1-1" / "1-CAT-W020-sum" / "done.json")["mlflow_run_id"]
    assert tr.client.get_run(rid).data.tags["최종"] == "예"
    with pytest.raises(sf.FinalError, match="이미 있음"):
        sf.main(_argv(w, "d", cfgp, "--report", "--reason", "다시"), tracker_fn=lambda c: tr)


def test_research_drop_result_for_chain_final(world):
    w = world
    st = _state(w, "s4/state_e.json", "4-a-CAT", "s4")
    cfgp = _cfg(w, "e", st)
    sf.main(_argv(w, "e", cfgp))
    rdir = sf.main(_argv(w, "e", cfgp, "--report", "--reason", "시험"), tracker_fn=lambda c: None)
    rt = pd.read_csv(rdir / "research_table.csv", encoding="utf-8-sig").set_index("칼럼")
    assert rt.loc["t3", "최종 모델 구조의 변수 빼기 결과"] == "남음"
    assert rt.loc["rh1", "최종 모델 구조의 변수 빼기 결과"] == "그룹째 빠짐"
    assert rt.loc["rh1", "최종 모델에 남음"] == "아니오"


def test_product_bundles():
    cl = pd.DataFrame({
        "prdlst_cd": ["A0000000000000", "A0100000000000", "B0000000000000", "B0100000000000", "B0200000000000",
                      "B0300000000000", "C0000000000000", "C0300000000000", "Z0000000000000", "Z3800000000000",
                      "D0000000000000", "D0100000000000", "B0201000000000"],
        "hrnk_prdlst_cd": ["", "A0000000000000", "", "B0000000000000", "B0000000000000", "B0000000000000", "",
                           "C0000000000000", None, "Z0000000000000", "", "D0000000000000", "B0200000000000"],
        "kor_nm": ["식물성", "곡류", "동물성", "축산물", "수산물", "기타동물", "가공식품", "과자", "기준규격외", "기타",
                   "첨가물", "x", "어류"]})
    p2 = np.array(["A0100000000000", "B0100000000000", "B0200000000000", "B0201000000000", "B0300000000000",
                   "C0300000000000", "Z3800000000000", "D0100000000000", "Q9", "코드없음"], dtype=object)
    got = sf.product_bundles(p2, cl).tolist()
    assert got == ["농산물", "축산물", "수산물", "수산물", "그 밖", "가공식품", "기준규격외", "그 밖", "그 밖", "코드없음"]
    assert sf.product_bundles(p2, None).tolist() == ["그 밖"] * 9 + ["코드없음"]


def test_aux_tertile_partition(world):
    w = world
    st = w["st"]
    rng = np.random.default_rng(0)
    parts = []
    for h, idx in fd.iter_scoring_pairs_by_h(st.d_days, fd.SPLITS["refit"]):
        if idx.size:
            parts.append(pd.DataFrame({"lims_data_key": st.keys[idx], "h": np.int16(h), "fold": "refit",
                                       "pred": rng.random(idx.size)}))
    P = ev.prepare_pairs(pd.concat(parts, ignore_index=True), st, "refit")
    y = (rng.random(P.pred.size) < 0.1).astype(np.int64)
    prod, tier = sf.aux_tables(P, y, st, 0.5, None)
    assert tier["n_pairs"].sum() == P.pred.size
    assert list(tier["단위 시료 수 구간"]) == ["하위", "중간", "상위"]
    allr = prod[prod["국내·해외"] == "전체"]
    assert allr["n_pairs"].sum() == P.pred.size
    dom = prod[prod["국내·해외"] == "domestic"]["n_pairs"].sum()
    assert dom == int((P.src_route == "domestic").sum())


def test_importance_path_fake_nodes(world):
    """최종 모델 안 순열 중요도: importance.permutation_importance 를 가짜 노드로 — 대상 = targets_final(리서치 칼럼·묶음·
    그룹), 섞기 전 점수 = 판 fold 점수(expect_base)."""
    from lst import importance as im
    from lst import registry as rg

    w = world
    st = _state(w, "s4/state_f.json", "4-a-CAT", "s4")
    cfgp = _cfg(w, "f", st)
    sf.main(_argv(w, "f", cfgp))
    spec = w["V"]["4-a-CAT"][1]

    def nodes(cfg, root, inputs_root, decision, samples):
        out = {}
        known = samples.known_frame()
        lk = {"t3": "t3", "L1": "L1", "h": None, "g": None}
        for f in ev.CV_FOLDS:
            rows = rn.add_bounds(fd.make_train_rows(known, fd.SPLITS[f]), spec.W)
            asm = ResearchAsm(spec, f).fit(rows)
            m = NoisyModel(spec, asm).fit(asm.transform(rows), rows["y"].to_numpy(), seed=42, n_jobs=8)
            out[f] = im.Leaf(f"{spec.name}-{f}", asm, m, layout=im.layout_from_names(list(asm.columns), lk))
        return out

    # 가짜 로지스틱은 예측 행 수에 따라 마지막 자리가 달라 순위 동점 풀이가 조금 다름 — 시험에서만 허용 오차를 넓힘
    dst = sf.run_importance(sf.load_cfg(cfgp), w["tmp"], w["inputs"], w["inputs"] / "final" / "f", nodes_fn=nodes, M=2,
                            base_tol=1e-4)
    s = pd.read_csv(dst / "importance_summary.csv", encoding="utf-8-sig")
    kinds = s.groupby("target_kind")["target"].apply(list).to_dict()
    assert kinds["column"] == ["t3", "L1"] and kinds["bundle"] == ["리서치 변수 전체"]
    assert set(kinds["group"]) == {rg.group(rg.unit("t3").group_no).name, rg.group(rg.unit("L1").group_no).name}
    meta = json.loads((dst / "importance_meta.json").read_text(encoding="utf-8"))
    cv = _cv(w, "s4", "4-a-CAT")
    for f in ev.CV_FOLDS:
        assert abs(meta["base"][f] - cv["fold_band_weighted"][f]) < 1e-4


def test_x1_compare(world):
    """① 당일 예측 비교: h = 1 쌍만으로 fold 마다 PR-AUC → 평균, '①_better' 기록(최종 고르기와 무관), 보고 때 표 한 줄씩."""
    import shutil

    from lst import metrics as mt

    w = world
    src = w["inputs"] / "s8" / "8-T-CAT"
    xd = w["inputs"] / "s4x" / "x1-CAT"
    xd.mkdir(parents=True)
    for f in ("spec.json", "done.json", "cv_scores.json"):
        shutil.copyfile(src / f, xd / f)
    for s in rn.SPLIT_ORDER:
        pr = pd.read_parquet(rn.pred_path(src, s))
        pr[pr["h"] == 1].to_parquet(rn.pred_path(xd, s), index=False)
    st = _state(w, "s4/state_g.json", "4-a-CAT", "s4")
    cfgp = _cfg(w, "g", st)
    raw = json.loads(cfgp.read_text(encoding="utf-8"))
    raw["x1_dir"] = "s4x/x1-CAT"
    cfgp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    d = sf.main(_argv(w, "g", cfgp))
    assert d["final"] == "4-a-CAT"
    x = d["x1"]
    for name, key in (("4-a-CAT", "final_h1"), ("x1-CAT", "x1_h1")):
        dd = w["inputs"] / ("s4" if name == "4-a-CAT" else "s4x") / name
        exp = []
        for f in ev.CV_FOLDS:
            pr = pd.read_parquet(rn.pred_path(dd, f))
            pr = pr[pr["h"] == 1]
            P = ev.prepare_pairs(pr, w["st"], f, require_complete=False)
            exp.append(mt.pr_auc(w["st"].cv_y(P.pos), P.pred, P.u()))
        assert np.isclose(x[key]["cv"], np.mean(exp))
    assert d["①_better"] == mt.is_improvement(x["x1_h1"]["cv"], x["final_h1"]["cv"])
    rdir = sf.main(_argv(w, "g", cfgp, "--report", "--reason", "시험"), tracker_fn=lambda c: None)
    xc = pd.read_csv(rdir / "x1_compare.csv", encoding="utf-8-sig")
    assert list(xc["구분"]) == ["최종 모델", "① 당일 예측"] and xc["test h = 1 PR-AUC"].notna().all()


def test_chain_steps_cancel_line_drops_adopted_step():
    """'채택 취소' 줄(담당자 결정)은 그 단계의 앞선 채택 줄만 변화 표에서 뺌 — 다른 단계 채택은 그대로."""
    lines = [
        {"step": "3-0", "judgement": "출발", "after": "3-0-LSTM"},
        {"step": "5-6", "judgement": "채택", "before": "3-0-LSTM", "after": "5-LSTM"},
        {"step": "8-T", "judgement": "튜닝 안 함", "before": "5-LSTM", "after": "5-LSTM"},
        {"step": "8-D", "judgement": "채택", "before": "5-LSTM", "after": "8-D-LSTM"},
        {"step": "8-D", "judgement": "채택 취소", "before": "8-D-LSTM", "after": "5-LSTM", "reason": "test 확인 뒤 담당자 결정"},
    ]
    st = sf.chain_steps(lines)
    assert [s["name"] for s in st] == ["3-0-LSTM", "5-LSTM"]
    assert [s["adopted"] for s in st] == [False, True]
    # 취소 줄이 없으면 8-D 채택 줄이 남음(예전과 같음)
    assert [s["name"] for s in sf.chain_steps(lines[:-1])] == ["3-0-LSTM", "5-LSTM", "8-D-LSTM"]
