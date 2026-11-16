"""단계 3-0b(v78 — 1등 = 3-3 판정 뒤 현재 최고 판) — 합성 자료·가짜 모델.

  - 새 순서 흐름: 3-0 → ⑥⑦ 판정(3-0 순위) → 3-1 → 3-2 → 3-3 → 3-0b → 단계 4 가 사슬 상태를 읽음.
  - 1등이 앙상블일 때: 중요도 = 구성 모델 전부의 평균 예측(섞기 전 점수 = 3-3-T3 fold 점수), 구성마다 같은 그룹을 빼고 다시
    학습 → 다시 평균 → 채택 규칙 → 끝 판 3-0b-3-3-T3-end → 사슬 상태(lst/stage4 derive_specs 가 읽음).
  - 1등이 ④ 면 학습 없이 유지.
  - 옛 순서 3-0b 결과 이어받기(같은 판·같은 규칙·코드 바이트 같음 → 학습·중요도 없이), 코드가 다르면 안 이어받음, 옛 후보 판이
    안 끝났으면 멈춤.
중요도 노드는 가짜(구성 판의 저장 예측을 그대로 내는 칼럼 + 예측에 안 쓰는 칼럼 — 부록 A-5 그룹 단위에 붙임).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import evaluate as ev
from lst import importance as im
from lst import metrics as mt
from lst import registry as rg
from lst import runner as rn
from lst import stage3 as s3
from lst import stage3_combine as c3
from lst import stage3b as sb
from lst import stage4 as s4
from lst.paths import resolve
from tests.test_chain import DEC, SEL  # noqa: F401
from tests.test_runner import STUB, FakeAssembler, env  # noqa: F401
from tests.test_s3_scripts import CODE, _ctx, load_script, stage3_env  # noqa: F401
from tests.test_stage3_combine import FakeLims, FakeVae, FakeVaeFeats, _actx, _cfg_file, _upto_x67

KEEP_UNIT = rg.group("기온").units[0]
DROP_UNIT = rg.group("계절").units[0]


@pytest.fixture(autouse=True)
def _fixed_code_md5(monkeypatch):
    from lst import config as cf

    monkeypatch.setattr(cf, "code_md5", lambda *a, **k: "f" * 32)


# ---------------------------------------------------------------- 가짜 중요도 노드
class PredAsm:
    """구성 판 저장 fold 예측(교차 N)을 칼럼 c_keep 으로, 예측에 안 쓰는 c_drop."""
    columns = ["c_keep", "c_drop"]

    def __init__(self, vdir: Path, split: str):
        df = pd.read_parquet(rn.pred_path(vdir, split))
        self.s = pd.Series(df["pred"].to_numpy(dtype=np.float64), index=c3._kh(df))

    def transform(self, meta):
        p = self.s.reindex(c3._kh(meta)).to_numpy()
        assert not np.isnan(p).any()
        return pd.DataFrame({"c_keep": p, "c_drop": np.zeros(len(p))})


class PassModel:
    def predict_proba(self, X, n_iter=None):
        return X["c_keep"].to_numpy(dtype=np.float64)


def fake_member_nodes(inputs):
    lay = im.layout_from_names(["c_keep", "c_drop"], {"c_keep": KEEP_UNIT, "c_drop": DROP_UNIT})

    def f(m):
        vdir = resolve(inputs, m["dir"])
        return {fo: im.Leaf(f"{m['name']}-{fo}", PredAsm(vdir, fo), PassModel(), layout=lay) for fo in ev.CV_FOLDS}
    return f


# ---------------------------------------------------------------- 3-3 까지(조건 없이 끝까지 — 판정 시험은 test_stage3_combine)
def flow_to_3_3(e, tmp_path):
    inputs, st = e["inputs"], e["st"]
    x67 = _upto_x67(e, tmp_path)
    m1 = load_script("s3_1")
    out1 = inputs / "s3" / "3-1"
    cfg1 = m1.load_cfg(_cfg_file(tmp_path, "s3_1"))
    m1.write_plan(cfg1, inputs, out1)
    reg = []
    m1.run_vae(cfg1, tmp_path, inputs, out1, feats=FakeVaeFeats(st), model_factory=lambda: FakeVae(reg),
               check=lambda: None)
    rn.run_group([m1.board_spec(cfg1, inputs)], _actx(e, out1))
    m1.judge(cfg1, inputs, out1, rn.snapshot_code(CODE, out1), None)
    m2 = load_script("s3_2")
    out2 = inputs / "s3" / "3-2"
    cfg2 = m2.load_cfg(_cfg_file(tmp_path, "s3_2"))
    m2.write_plan(cfg2, inputs, out2)
    code2 = rn.snapshot_code(CODE, out2)
    m2.write_pointers(cfg2, inputs, out2, code2, None)
    for g in rn.group_specs(m2.train_specs(cfg2, inputs)):
        rn.run_group(g, _actx(e, out2))
    m2.run_x4(cfg2, tmp_path, inputs, out2, feats=FakeLims(st), check=lambda: None)
    m2.judge(cfg2, inputs, out2, code2, None)
    m3 = load_script("s3_3")
    out3 = inputs / "s3" / "3-3"
    cfg3 = m3.load_cfg(_cfg_file(tmp_path, "s3_3"))
    assert m3.train_specs(cfg3, inputs) == []                                   # v78: 3-3 새 학습 없음
    m3.write_plan(cfg3, inputs, out3)
    code3 = rn.snapshot_code(CODE, out3)
    m3.write_member_pointers(cfg3, inputs, out3, code3, None)
    m3.build_t3(cfg3, inputs, out3, code3, None)
    j3 = m3.judge(cfg3, inputs, out3, code3, None)
    return {"x67": x67, "j3": j3}


def b_cfg(tmp_path, tag, **kw):
    c = json.loads((CODE / "configs" / "s3_0b.json").read_text(encoding="utf-8"))
    c["use_mlflow"] = False
    c.pop("reuse_3_0b_old", None)
    c["dirs"]["3-0b"] = f"s3/3-0b-{tag}"
    c["chain_state_out"] = f"logs/chain_state_3-0b_{tag}.json"
    c.update(kw)
    p = tmp_path / f"s3_0b_{tag}.json"
    p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    return p


def run_3_0b(e, tmp_path, cfgp, nodes_fn):
    """기본 실행과 같은 차례(학습은 같은 프로세스의 runner — 가짜 입력·모델)."""
    m = load_script("s3_0b")
    inputs = e["inputs"]
    cfg = m.load_cfg(cfgp)
    out = resolve(inputs, cfg["dirs"]["3-0b"])
    info = m.choose_target(cfg, inputs, out)
    m.reuse(cfg, inputs, out)
    m.run_importance(cfg, tmp_path, inputs, out, member_nodes_fn=nodes_fn, check=lambda: None)
    for g in rn.group_specs(m.train_specs(cfg, inputs, out)):
        rn.run_group(g, _actx(e, out))
    code = rn.snapshot_code(CODE, out)
    m.build_ensemble(cfg, inputs, out, code, None)
    d = m.finalize(cfg, inputs, out, code, None)
    return m, cfg, out, info, d


def _cv(d: Path):
    return rn.load_done(d.parent, d.name)[2]


# ---------------------------------------------------------------- 흐름 + 1등이 앙상블
def test_flow_and_ensemble_first(stage3_env, tmp_path):
    e = stage3_env
    inputs = e["inputs"]
    f = flow_to_3_3(e, tmp_path)
    st33 = s4.load_state(inputs / "logs" / "chain_state_3-3.json")
    assert st33["current_best"]["name"] == f["j3"]["state_after"]["current_best"]["name"]

    # (1) 사슬 그대로의 1등
    m, cfg, out, info, d = run_3_0b(e, tmp_path, b_cfg(tmp_path, "real", chain_state_in="logs/chain_state_3-3.json"),
                                    fake_member_nodes(inputs))
    assert info["name"] == st33["current_best"]["name"]
    if st33["current_best"]["kind"] == "stack4":                                   # ④ = 학습 없이 유지
        assert d["candidate"] is None and not d["adopted"]
    st = s4.load_state(inputs / cfg["chain_state_out"])
    assert st["step"] == "3-0b" and st["current_best"]["name"] == info["end"] == d["end"]["name"]
    assert st["history"][-1]["step"] == "3-0b"

    # (2) 1등 = 3-3-T3 앙상블(강제)
    ens = c3.best_def(inputs, {"name": c3.T3, "dir": "s3/3-3/3-3-T3"})
    st_e = {**st33, "current_best": ens}
    s4.save_state(inputs / "logs" / "chain_state_3-3_ens.json", st_e)
    m, cfg, out, info, d = run_3_0b(e, tmp_path, b_cfg(tmp_path, "ens", chain_state_in="logs/chain_state_3-3_ens.json"),
                                    fake_member_nodes(inputs))
    assert info["part"] == c3.T3 and info["candidate"] == "3-0b-3-3-T3" and info["end"] == "3-0b-3-3-T3-end"
    imp = pd.read_csv(out / sb.IMPORTANCE_DIR / "importance_summary.csv", encoding="utf-8-sig")
    assert set(imp["target"]) == {"기온", "계절"}                                  # 1등 입력에 있는 그룹만
    assert float(imp.set_index("target").loc["계절", "mean_I"]) == 0.0            # 예측에 안 쓰는 칼럼 → I = 0 → 뺌
    meta = json.loads((out / sb.IMPORTANCE_DIR / "importance_meta.json").read_text(encoding="utf-8"))
    t3cv = _cv(inputs / "s3" / "3-3" / c3.T3)
    for fo in ev.CV_FOLDS:                                                        # 섞기 전 = 3-3-T3 fold 점수(평균 예측)
        assert abs(meta["base"][fo] - t3cv["fold_band_weighted"][fo]) <= 1e-9
    drop = list(d["drop_groups"])
    assert "계절" in drop and drop == s3.drop_groups_from_summary(imp)
    mem = d["members_cv"]
    assert set(mem) == {c3.source_spec(inputs, x["dir"]).algorithm for x in ens["members"]} and len(mem) == 3
    for alg, v in mem.items():                                                    # 구성마다 같은 그룹을 빼고 다시 학습
        assert v["after"] == f"3-0b-3-3-T3--{alg}"
        sp = json.loads((out / v["after"] / "spec.json").read_text(encoding="utf-8"))
        assert sp["stage"] == "3-0b" and sp["extra"]["version"]["exclude_groups"] == drop
        src = json.loads((resolve(inputs, next(x["dir"] for x in ens["members"]
                                              if c3.source_spec(inputs, x["dir"]).algorithm == alg))
                          / "spec.json").read_text(encoding="utf-8"))
        assert bool(sp["extra"].get("anomaly")) == bool(src["extra"].get("anomaly"))   # 이상 점수 칸 그대로(VAE 안 바꿈)
    cdir = out / "3-0b-3-3-T3"
    for s in rn.SPLIT_ORDER:                                                      # 후보 = 구성 다시 평균
        mean = np.mean([pd.read_parquet(rn.pred_path(out / v["after"], s))["pred"].to_numpy() for v in mem.values()],
                       axis=0)
        assert np.allclose(pd.read_parquet(rn.pred_path(cdir, s))["pred"], mean)
    ccv = _cv(cdir)
    rule = s3.adopt_3_0b({"cv": ccv["cv_band_weighted"], "fold": ccv["fold_band_weighted"]},
                         {"cv": t3cv["cv_band_weighted"], "fold": t3cv["fold_band_weighted"]})
    assert d["adopted"] == rule["adopt"]
    src_rel = "s3/3-0b-ens/3-0b-3-3-T3" if rule["adopt"] else "s3/3-3/3-3-T3"
    assert d["end"]["source_dir"] == src_rel
    end = json.loads((out / "3-0b-3-3-T3-end" / "source.json").read_text(encoding="utf-8"))
    assert end["dir"] == src_rel and not (out / "3-0b-3-3-T3-end" / "pred_fold1.parquet").exists()
    st = s4.load_state(inputs / cfg["chain_state_out"])
    cb = st["current_best"]
    assert cb["name"] == "3-0b-3-3-T3-end" and cb["kind"] == "ensemble" and cb["dir"] == src_rel
    assert st["tree_target"]["name"] == cb["name"]                                # 끝 판(앙상블·교체 후보 아님)
    if rule["adopt"]:
        assert sorted(x["dir"] for x in cb["members"]) == sorted(f"s3/3-0b-ens/{v['after']}" for v in mem.values())
    chain = [json.loads(x) for x in (inputs / "logs" / "chain_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert chain[-1]["step"] == "3-0b" and chain[-1]["after"] == cb["name"] and chain[-1]["after_dir"] == src_rel
    assert [c["step"] for c in chain][:2] == ["3-0", "⑥⑦"]
    assert m.finalize(cfg, inputs, out, rn.snapshot_code(CODE, out), None) == json.loads(
        (out / sb.DECISION_FILE).read_text(encoding="utf-8"))                     # 두 번째는 그대로

    # (3) 단계 4 읽기: s4 build(사슬 상태 = 3-0b 판정 뒤)
    s4m = load_script("s4")
    c4 = json.loads((CODE / "configs" / "s4_T.json").read_text(encoding="utf-8"))
    assert c4["chain_state_in"] == "logs/chain_state_3-0b.json"
    c4["chain_state_in"] = cfg["chain_state_out"]
    b = s4m.build(c4, inputs)
    assert b["state"]["current_best"]["name"] == "3-0b-3-3-T3-end"
    assert [s.algorithm for s in b["run"]] == [c3.source_spec(inputs, x["dir"]).algorithm for x in cb["members"]] * 2   # 4-a·4-a0 구성마다
    assert all(s.base_variant == "3-0b-3-3-T3-end" for s in b["run"])
    tm = s4m.target_member(b["state"], inputs)                                   # 앙상블 → 3-0 CV 1위 트리
    rank = b["state"]["tree_rank_3_0b"]
    assert rank[s4.member_spec(inputs, tm).algorithm] == min(rank[s4.member_spec(inputs, x).algorithm]
                                                             for x in cb["members"])


def test_x4_first_no_training(stage3_env, tmp_path):
    """1등이 ④ 면 빼는 대상 그룹이 없어 학습 없이 유지(끝 판 = ④, 교체 후보 물려받음)."""
    e = stage3_env
    inputs = e["inputs"]
    x67 = _upto_x67(e, tmp_path)
    vdir = inputs / "s3" / "3-2" / c3.X4
    vdir.mkdir(parents=True)
    from tests.test_evaluate import fake_preds

    for s in rn.SPLIT_ORDER:
        fake_preds(e["st"], s, seed=7).to_parquet(rn.pred_path(vdir, s), index=False)
    sp = c3.x4_spec(False, 20).to_dict()
    (vdir / "spec.json").write_text(json.dumps(sp, ensure_ascii=False), encoding="utf-8")
    cv = ev.score_cv(rn.load_preds(vdir, ev.CV_FOLDS), e["st"])
    from lst import config as cf

    cf.dump_scores_json(vdir / "cv_scores.json", cv)
    cf.dump_scores_json(vdir / "done.json", {"name": c3.X4, "cv_band_weighted": cv["cv_band_weighted"],
                                             "many_ties": cv["ties"]["many_ties"], "selection_count": 1,
                                             "spec_md5": "x", "code_md5": "f" * 32})
    best = c3.best_def(inputs, {"name": c3.X4, "dir": f"s3/3-2/{c3.X4}", "replacement": True})
    tt = c3.best_def(inputs, {"name": x67["tree_target"], "dir": x67["boards_3_0"][x67["tree_target"]]["dir"]})
    st = {"step": "3-3", "current_best": best, "tree_target": tt, "tree_top3": x67["tree_top3"],
          "tree_rank_3_0b": {}, "history": []}
    s4.save_state(inputs / "logs" / "chain_state_3-3_x4.json", st)
    m, cfg, out, info, d = run_3_0b(e, tmp_path, b_cfg(tmp_path, "x4", chain_state_in="logs/chain_state_3-3_x4.json"),
                                    lambda mm: pytest.fail("④ 는 중요도 노드를 만들지 않음"))
    assert info["end"] == f"3-0b-{c3.X4}-end" and d["candidate"] is None and not d["adopted"] and d["drop_groups"] == []
    out_st = s4.load_state(inputs / cfg["chain_state_out"])
    assert out_st["current_best"]["kind"] == "stack4" and out_st["current_best"]["replacement"]
    assert out_st["tree_target"]["name"] == x67["tree_target"]                    # 교체 후보 → 트리 대상 판 그대로
    assert json.loads((out / info["end"] / "spec.json").read_text(encoding="utf-8"))["role"] == "교체 후보"


# ---------------------------------------------------------------- 옛 순서 3-0b 결과 이어받기
def _old_3_0b(e, alg="LSTM", drop=("계절",), run_candidate=True):
    """옛 순서 3-0b 폴더(s3/3-0b-old): 대상·중요도(요약·meta)·후보 판(설정 = stage3.spec_3_0b)·코드 사본."""
    from lst import config as cf
    from lst import wiring as wr

    inputs = e["inputs"]
    sub = "T" if alg in s3.TREE6 else "D"
    src = rn.VariantSpec(**json.loads((inputs / "s3" / f"3-0-{sub}" / f"3-0-{alg}" / "spec.json").read_text("utf-8")))
    old = inputs / "s3" / "3-0b-old"
    old.mkdir(parents=True)
    cf.dump_json(old / sb.TARGET_FILE, {"selected": src.name, "algorithm": alg, "dir": f"s3/3-0-{sub}/{src.name}"})
    tg = im.targets_groups(wr.version_spec(src).units())
    imp = old / sb.IMPORTANCE_DIR
    imp.mkdir()
    pd.DataFrame({"target_kind": "group", "target_no": [t.no for t in tg], "target": [t.name for t in tg],
                  "mean_I": [(-0.001 if t.name in drop else 0.002) for t in tg]}).to_csv(
        imp / "importance_summary.csv", index=False, encoding="utf-8-sig")
    cf.dump_json(imp / "importance_meta.json", {"variant": src.name, "M": 10, "perm_seed": im.PERM_SEED,
                                                "targets": [{"kind": t.kind, "no": t.no, "name": t.name,
                                                             "seed_key": list(t.seed_key)} for t in tg]})
    cand = s3.spec_3_0b(src, list(drop))
    if run_candidate:
        rn.run_group([cand], _ctx(e, old, e["log"]))
    else:
        rn.snapshot_code(CODE, old)
    return src, cand, old


def _state_first(e, src, sub):
    inputs = e["inputs"]
    best = c3.best_def(inputs, {"name": src.name, "dir": f"s3/3-0-{sub}/{src.name}"})
    st = {"step": "3-3", "current_best": best, "tree_target": best if src.kind == "tree" else best,
          "tree_top3": [], "tree_rank_3_0b": {}, "history": []}
    if src.kind != "tree":
        st["tree_target"] = c3.best_def(inputs, {"name": "3-0-CAT", "dir": "s3/3-0-T/3-0-CAT"})
    s4.save_state(inputs / "logs" / "chain_state_3-3_old.json", st)


def test_reuse_old_3_0b(stage3_env, tmp_path):
    e = stage3_env
    inputs = e["inputs"]
    src, cand, old = _old_3_0b(e)
    _state_first(e, src, "D")
    cfgp = b_cfg(tmp_path, "reuse", chain_state_in="logs/chain_state_3-3_old.json", reuse_3_0b_old="s3/3-0b-old")
    m, cfg, out, info, d = run_3_0b(e, tmp_path, cfgp, lambda mm: pytest.fail("이어받으면 중요도를 다시 안 셈"))
    r = json.loads((out / sb.REUSE_FILE).read_text(encoding="utf-8"))
    assert r["ok"] and r["drop"] == ["계절"] and r["candidate"] == cand.name
    assert info["part"] == "LSTM" and info["candidate"] == cand.name == "3-0b-LSTM" and info["end"] == "3-0b-LSTM-end"
    assert (out / sb.IMPORTANCE_DIR / "importance_summary.csv").read_bytes() == \
        (old / sb.IMPORTANCE_DIR / "importance_summary.csv").read_bytes()
    assert not (out / cand.name).exists()                                         # 후보 판 다시 학습 안 함
    ccv, bcv = _cv(old / cand.name), _cv(inputs / "s3" / "3-0-D" / src.name)
    rule = s3.adopt_3_0b({"cv": ccv["cv_band_weighted"], "fold": ccv["fold_band_weighted"]},
                         {"cv": bcv["cv_band_weighted"], "fold": bcv["fold_band_weighted"]})
    assert d["adopted"] == rule["adopt"] and d["reused_from"] == "s3/3-0b-old"
    assert d["end"]["source_dir"] == (f"s3/3-0b-old/{cand.name}" if rule["adopt"] else f"s3/3-0-D/{src.name}")
    st = s4.load_state(inputs / cfg["chain_state_out"])
    assert st["current_best"]["name"] == "3-0b-LSTM-end" and st["current_best"]["kind"] == "dl"
    assert st["tree_target"]["name"] == "3-0-CAT"                                 # 딥러닝 1등 → 트리 대상 판 그대로


def test_reuse_refused_when_code_differs_or_waits(stage3_env, tmp_path):
    e = stage3_env
    inputs = e["inputs"]
    src, cand, old = _old_3_0b(e)
    _state_first(e, src, "D")
    p = old / "_code" / "lst" / "metrics.py"
    p.write_bytes(p.read_bytes() + b"\n# changed\n")
    info = sb.target_info(s4.load_state(inputs / "logs" / "chain_state_3-3_old.json")["current_best"], inputs)
    r = sb.reuse_check(old_rel="s3/3-0b-old", info=info, inputs_root=inputs, code_root=CODE, M=10)
    assert not r["ok"] and not r["wait"] and "lst/metrics.py" in r["code_diff"]
    p.write_bytes(p.read_bytes()[: -len(b"\n# changed\n")])
    assert sb.reuse_check(old_rel="s3/3-0b-old", info=info, inputs_root=inputs, code_root=CODE, M=10)["ok"]
    # 옛 후보 판이 아직 안 끝남 → 멈춤(같은 판을 두 번 학습하지 않음)
    import shutil

    shutil.rmtree(old / cand.name)
    r = sb.reuse_check(old_rel="s3/3-0b-old", info=info, inputs_root=inputs, code_root=CODE, M=10)
    assert r["wait"] and not r["ok"]
    cfgp = b_cfg(tmp_path, "wait", chain_state_in="logs/chain_state_3-3_old.json", reuse_3_0b_old="s3/3-0b-old")
    m = load_script("s3_0b")
    cfg = m.load_cfg(cfgp)
    out = resolve(inputs, cfg["dirs"]["3-0b"])
    m.choose_target(cfg, inputs, out)
    with pytest.raises(rn.RunStopped, match="아직 안 끝남"):
        m.reuse(cfg, inputs, out)
    # 1등이 옛 대상과 다르면 이어받지 않음(새로 계산)
    other = sb.target_info(c3.best_def(inputs, {"name": "3-0-CAT", "dir": "s3/3-0-T/3-0-CAT"}), inputs)
    r = sb.reuse_check(old_rel="s3/3-0b-old", info=other, inputs_root=inputs, code_root=CODE, M=10)
    assert not r["ok"] and not r["wait"] and "옛 3-0b 대상" in r["reasons"][0]


def test_candidate_plan_names(stage3_env, tmp_path):
    """구조 자리(부록 A-14): 1등이 3-0 판이면 구조 약어(설정 = stage3.spec_3_0b — 옛 판과 같은 md5), 그 밖이면 1등 판 이름."""
    e = stage3_env
    inputs = e["inputs"]
    b = c3.best_def(inputs, {"name": "3-0-XGB", "dir": "s3/3-0-T/3-0-XGB"})
    info = sb.target_info(b, inputs)
    p = sb.candidate_plan(info, ["계절", "습도"], inputs, "s3/3-0b-best")
    src = c3.source_spec(inputs, "s3/3-0-T/3-0-XGB")
    assert p["train"][0].md5() == s3.spec_3_0b(src, ["계절", "습도"]).md5() and info["end"] == "3-0b-XGB-end"
    assert sb.candidate_plan(info, [], inputs, "s3/3-0b-best") is None
    b2 = {**b, "name": "3-1-XGB-VAE"}
    info2 = sb.target_info(b2, inputs)
    assert info2["part"] == "XGB"                                                 # 판 설정 기준(3-0 판 폴더)
    b3 = {**b, "replacement": True}
    info3 = sb.target_info(b3, inputs)
    assert info3["part"] == "3-0-XGB" and info3["candidate"] == "3-0b-3-0-XGB"
    sp = sb.candidate_plan(info3, ["계절"], inputs, "s3/3-0b-best")["train"][0]
    assert sp.name == "3-0b-3-0-XGB" and sp.params.get(rn.ITER_SELECT) and sp.base_variant == "3-0-XGB"
    assert sp.extra["version"]["exclude_groups"] == ["계절"]
