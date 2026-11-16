"""s8(단계 8) 시험 — 합성 자료·가짜 점수·가짜 모델: Optuna 시도 수 20(첫 시도 + 19)·이어하기, 기본 유지 판정(4자리),
8-T·Focal 판 params(튜닝 설정·learning_rate 규칙·Focal 손실), 판 이름, dl·nn60 대상 판 없음, 앙상블 구성마다 튜닝,
④ 6종(learning_rate 탐색 안 함), 판정 순서 8-T → Focal."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from lst import config as cf
from lst import runner as rn
from lst import stage4 as s4
from lst import tuning as tu
from tests.test_runner import STUB, FakeAssembler, FakeModel, env  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import s8  # noqa: E402

pytest.importorskip("optuna")


# ---------------------------------------------------------------- 가짜 점수
class Counter:
    def __init__(self, fail_at=None, base=0.30):
        self.n, self.fail_at, self.base = 0, fail_at, base

    def __call__(self, algo, params, folds, *, max_iter, seed, n_jobs):
        self.n += 1
        if self.fail_at is not None and self.n == self.fail_at:
            raise MemoryError("가짜 메모리 오류")
        lr = float(params.get("learning_rate", 0.1))
        md = float(params.get("max_depth", 6))
        return {"cv": self.base + 0.01 * np.sin(lr * 30 + md), "fold_scores": {"fold1": 0.1}}


def _spec(name, alg="XGB", params=None, kind="tree", extra=None):
    return rn.VariantSpec(name=name, stage="5-6", order="2차", kind=kind, algorithm=alg, W=20, mode="sum",
                          input_set="x", role="후보", train_count=4,
                          params=params if params is not None else ({rn.ITER_SELECT: True} if alg in s4.BOOST else {}),
                          extra=extra or {"version": {"include_ids": ["WEATHER"]}})


def _part(alg="XGB", from_target=False, rep="own"):
    return s8.Part(label=alg, alg=alg, member={"name": "m", "dir": "x"}, src=_spec("m", alg),
                   plan=s4.RowPlan((), rep) if rep == "own" else s4.RowPlan(("reduce",), rep, "s4/sel"),
                   catenc_k=None, seed=42, base_params={}, max_iter=2000 if alg in s4.BOOST else None,
                   t000_from_target=from_target, tune_lr=True)


def test_tune_20_trials_and_resume(tmp_path):
    c = Counter()
    p = _part("XGB")
    res = s8.tune_part(p, tmp_path, folds_fn=lambda: ["f"], base_cv=lambda: 0.3, score_config=c)
    assert c.n == 20 and res["n_trials_total"] == 20                      # 첫 시도 + 19
    lines = [json.loads(x) for x in (tmp_path / "optuna_XGB.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(r["event"] == "trial" for r in lines) == 19 and lines[0]["n_trials"] == 19 and lines[0]["seed"] == 42
    assert len(res["selection"]["top"]) == 1 + 5                           # t000 + 상위 5
    c2 = Counter()
    assert s8.tune_part(p, tmp_path, folds_fn=lambda: ["f"], base_cv=lambda: 0.3, score_config=c2) == res and c2.n == 0
    # 멈춤 뒤 이어하기: 시도 7 에서 멈추면 그 시도부터(앞 시도 다시 안 매김)
    d2 = tmp_path / "b"
    c3 = Counter(fail_at=8)
    with pytest.raises(MemoryError):
        s8.tune_part(p, d2, folds_fn=lambda: ["f"], base_cv=lambda: 0.3, score_config=c3)
    c4 = Counter()
    r2 = s8.tune_part(p, d2, folds_fn=lambda: ["f"], base_cv=lambda: 0.3, score_config=c4)
    assert c4.n == 20 - 7 and r2["selection"]["best_cv"] == res["selection"]["best_cv"]


def test_t000_from_target_no_training(tmp_path):
    c = Counter()
    res = s8.tune_part(_part("CAT", from_target=True), tmp_path, folds_fn=lambda: pytest.fail("학습 없어야 함")
                       if c.n < 0 else ["f"], base_cv=lambda: 0.25, score_config=c)
    assert c.n == 19 and res["t000"]["cv"] == 0.25 and "학습 없음" in res["t000"]["source"]


def test_default_keep_round4(tmp_path):
    class Flat(Counter):
        def __call__(self, *a, **k):
            self.n += 1
            return {"cv": 0.30004}
    r = s8.tune_part(_part("LGB", from_target=True), tmp_path, folds_fn=lambda: ["f"], base_cv=lambda: 0.3,
                     score_config=Flat())
    assert r["adopt"] is False and r["selection"]["judgement"] == "기본 유지" and r["params"] is None
    sel = tu.select_tuned(0.3, [{"event": "trial", "number": 1, "value": 0.30006, "params": {}}])
    assert sel["adopt"] is True


def test_rep_10_space(tmp_path):
    p = _part("LGB", rep="4-a")
    assert p.rep == 10
    r = s8.tune_part(p, tmp_path, folds_fn=lambda: ["f"], base_cv=lambda: 0.3, score_config=Counter())
    assert r["space"]["min_child_samples"]["low"] == 50 and r["space"]["min_child_samples"]["high"] == 5000


# ---------------------------------------------------------------- 판 설정(합성 판 폴더)
def _variant(root: Path, rel: str, spec: rn.VariantSpec, lr=None):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    cf.dump_json(d / "spec.json", spec.to_dict(), overwrite=True)
    cf.dump_json(d / "done.json", {"n_iter": {"learning_rate": lr}} if lr is not None else {}, overwrite=True)
    return rel


def _best(name, kind, members):
    return {"name": name, "kind": kind, "dir": f"s5/{name}", "replacement": kind == "nn60", "members": members}


def _state(root, best, path="logs/state_in.json"):
    st = {"step": "5-6", "current_best": best, "tree_target": best, "tree_top3": [], "tree_rank_3_0b": {},
          "history": []}
    (root / path).unlink(missing_ok=True)
    s4.save_state(root / path, st)
    return path


def _cfg(root, state_in, mid="logs/state_mid.json"):
    return {"chain_state_in": state_in, "chain_state_mid": mid,
            "dirs": {"8-tune": "s8/8-tune", "8-T": "s8/8-T", "8-F": "s8/8-F", "8-judge": "s8/8-judge"}}


def _tune_file(root, label, adopt, params, t000=0.3):
    d = root / "s8" / "8-tune"
    d.mkdir(parents=True, exist_ok=True)
    cf.dump_json(d / f"tune_{label}.json", {"label": label, "adopt": adopt, "params": params, "t000": {"cv": t000},
                                           "selection": {"best_cv": 0.31, "best_trial": "t003", "top": []}},
                 overwrite=True)


def _log(tmp_path, algs=("XGB", "LGB", "CAT")):
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    for a in algs:
        log.record_max_iter(a, max_iter=4000)
    return log


def test_ensemble_parts_and_8T_params(tmp_path):
    root = tmp_path
    rows = {"row_plan": s4.RowPlan(("reduce", "hon5"), "4-a", "s5/sel").to_dict(), "catenc_k": 50}
    ms = []
    for a in ("CAT", "XGB", "RF"):
        rel = _variant(root, f"s5/5-E--{a}", _spec(f"5-E--{a}", a, extra={"version": {"include_ids": ["WEATHER"]},
                                                                            "s4": {"model_seed": 0}}),
                       lr=0.07 if a == "CAT" else None)
        ms.append({"name": f"5-E--{a}", "dir": rel, **rows})
    best = _best("5-E", "ensemble", ms)
    cfg = _cfg(root, _state(root, best))
    parts = s8.tune_parts(rn.read_json(root / cfg["chain_state_in"]), root, _log(tmp_path))
    assert [p.label for p in parts] == ["CAT", "XGB", "RF"]                # 구성마다
    assert all(p.seed == 0 and p.rep == 10 and not p.t000_from_target for p in parts)
    assert parts[0].base_params == {"learning_rate": 0.07} and parts[0].max_iter == 4000 and parts[2].max_iter is None
    _tune_file(root, "CAT", False, None)
    _tune_file(root, "XGB", True, {"learning_rate": 0.05, "max_depth": 4})
    _tune_file(root, "RF", False, None)
    b = s8.build_T(cfg, root, _log(tmp_path / "x"))
    assert b["status"] == "판" and b["name"] == "8-T-5-E"
    by = {s.algorithm: s for s in b["run"]}
    assert by["XGB"].params["params"] == {"learning_rate": 0.05, "max_depth": 4} and by["XGB"].params[rn.ITER_SELECT]
    assert by["CAT"].params["params"] == {"learning_rate": 0.07}         # 기본 유지 CatBoost = 대상 판 학습률
    assert by["RF"].params.get("params") is None
    assert all(s.extra["s4"]["model_seed"] == 0 and s.extra["s4"]["catenc_k"] == 50 for s in b["run"])
    assert b["ens"][0].algorithm == "T3" and b["best"]["dir"] == "s8/8-T/8-T-5-E"
    for lab in ("XGB",):
        _tune_file(root, lab, False, None)
    assert s8.build_T(cfg, root, _log(tmp_path / "y"))["status"] == "기본 유지"


def test_focal_params_and_names(tmp_path):
    root = tmp_path
    rel = _variant(root, "s5/5-CAT", _spec("5-CAT", "CAT"), lr=0.09)
    best = _best("5-CAT", "tree", [{"name": "5-CAT", "dir": rel}])
    _state(root, best, "logs/state_mid.json")
    cfg = _cfg(root, "logs/state_mid.json")
    f = s8.build_F(cfg, root)
    assert f["status"] == "판" and f["name"] == "8-F-5-CAT"
    lib = f["run"][0].params["params"]
    assert lib["loss_function"] == tu.FOCAL_LOSS and lib["learning_rate"] == 0.09   # 8-T 미채택: 대상 판 학습률
    # 8-T 채택 뒤(바탕 = 8-T 판): Optuna 학습률 그대로
    rel2 = _variant(root, "s8/8-T/8-T-CAT", _spec("8-T-CAT", "CAT", params={rn.ITER_SELECT: True, "params": {
        "learning_rate": 0.02, "depth": 4, "bootstrap_type": "Bernoulli"}}))
    _state(root, _best("8-T-CAT", "tree", [{"name": "8-T-CAT", "dir": rel2}]), "logs/state_mid.json")
    f2 = s8.build_F(cfg, root)
    assert f2["name"] == "8-F-8-T-CAT"
    assert f2["run"][0].params["params"] == {"learning_rate": 0.02, "depth": 4, "bootstrap_type": "Bernoulli",
                                             "loss_function": tu.FOCAL_LOSS}
    rel3 = _variant(root, "s5/5-XGB", _spec("5-XGB", "XGB"))
    _state(root, _best("5-XGB", "tree", [{"name": "5-XGB", "dir": rel3}]), "logs/state_mid.json")
    assert s8.build_F(cfg, root)["status"] == "CatBoost 구성 없음"


def test_dl_nn60_no_tuning(tmp_path):
    root = tmp_path
    for kind, alg in (("dl", "LSTM"), ("nn60", "NN60")):
        rel = _variant(root, f"s5/x-{alg}", _spec(f"x-{alg}", alg, kind="dl", params={}))
        cfg = _cfg(root, _state(root, _best(f"x-{alg}", kind, [{"name": f"x-{alg}", "dir": rel}])))
        assert s8.tune_parts(rn.read_json(root / cfg["chain_state_in"]), root, None) == []
        assert s8.build_T(cfg, root, None)["status"] == "튜닝 안 함" and s8.build_T(cfg, root, None)["run"] == []
        _state(root, _best(f"x-{alg}", kind, [{"name": f"x-{alg}", "dir": rel}]), "logs/state_mid.json")
        assert s8.build_F(cfg, root)["status"] == "튜닝 안 함"


def test_stack4_parts(tmp_path):
    root = tmp_path
    rel = _variant(root, "s3/x4-L6", _spec("x4-L6", "LR", params={}))
    best = _best("x4-L6", "stack4", [{"name": "x4-L6", "dir": rel, "stack4": {"version": {"include_ids": []}}}])
    cfg = _cfg(root, _state(root, best))
    parts = s8.tune_parts(rn.read_json(root / cfg["chain_state_in"]), root, None)
    assert [p.label for p in parts] == list(s4.TREE6)
    assert all(not p.tune_lr and p.max_iter is None for p in parts)
    for p in parts:
        _tune_file(root, p.label, p.label == "CAT", {"depth": 3, "bootstrap_type": "Bernoulli"} if p.label == "CAT" else None)
    b = s8.build_T(cfg, root, None)
    pdef = b["run"][0].extra["s4"]["provider"]
    assert b["run"][0].algorithm == "LR" and pdef["kind"] == "stack4" and pdef["params"] == {
        "CAT": {"depth": 3, "bootstrap_type": "Bernoulli"}}


# ---------------------------------------------------------------- 판정 순서(8-T → Focal) — 가짜 학습
class NoiseModel(FakeModel):
    """나쁜 대상 판(무작위 예측) — 8-T(가짜 로지스틱)가 이기게."""
    def fit(self, X, y, *, seed, n_jobs, eval_set=None):
        return self

    def predict_proba(self, X, n_iter=None):
        return np.random.default_rng(len(X)).random(len(X))

    def save(self, path):
        Path(path).mkdir(parents=True)

    def applied_params(self):
        return {}


def _run(env, out, specs, model=FakeModel):
    log = rn.ExperimentLog(env["tmp"] / "runlog.jsonl")
    ctx = rn.RunContext(samples=env["st"], out_dir=out, code_root=Path(__file__).resolve().parents[1],
                        input_md5s={"x": "0"}, exp_log=log, make_inputs=FakeAssembler,
                        make_model=lambda sp, asm=None: model(sp, asm), n_jobs=8, check_resources=STUB,
                        require_7a_logged=False)
    for sp in specs:
        try:
            rn.run_group([sp], ctx)
        except rn.ResumeError:          # 다른 작업자가 코드 폴더에 파일을 쓰는 중 — 코드 사본 다시
            import shutil
            shutil.rmtree(out / "_code")
            shutil.rmtree(out / sp.name, ignore_errors=True)
            rn.run_group([sp], ctx)


def test_judge_order_T_then_F(env):
    root = env["inputs"]
    tgt = _spec("3-0b-CAT-end", "CAT", params={rn.ITER_SELECT: True, "params": {"learning_rate": 0.05}})
    tgt = rn.VariantSpec(**{**tgt.to_dict(), "params": {}})                 # 대상 판(가짜 — 반복 수 고르기 없이)
    _run(env, root / "s3", [tgt], model=NoiseModel)
    rel = "s3/3-0b-CAT-end"
    sp = rn.read_json(root / rel / "spec.json")
    sp["params"] = {"params": {"learning_rate": 0.05}}
    cf.dump_json(root / rel / "spec.json", sp, overwrite=True)
    best = {"name": "3-0b-CAT-end", "kind": "tree", "dir": rel, "replacement": False,
            "members": [{"name": "3-0b-CAT-end", "dir": rel}]}
    cfg = _cfg(root, _state(root, best))
    _tune_file(root, "CAT", True, {"learning_rate": 0.03, "depth": 4, "bootstrap_type": "Bernoulli"})
    log = _log(env["tmp"])
    b = s8.build_T(cfg, root, log)
    assert b["name"] == "8-T-CAT"
    run = [rn.VariantSpec(**{**s.to_dict(), "params": {}}) for s in b["run"]]     # 가짜 학습(반복 수 고르기 없음)
    _run(env, root / "s8" / "8-T", run)
    jt = s8.judge_T_core(cfg, root, env["st"], log)
    assert jt["record"]["parts"][0]["against"] == "3-0b-CAT-end" and jt["record"]["parts"][0]["adopted"]
    assert jt["state"]["current_best"]["name"] == "8-T-CAT" and jt["record"]["same_score_comparison"]
    s4.save_state(root / cfg["chain_state_mid"], jt["state"])
    # 8-T 판 spec 에 튜닝 params 를 되돌려(가짜 학습 때 비웠음) Focal 바탕으로
    p = root / "s8/8-T/8-T-CAT/spec.json"
    d = rn.read_json(p)
    d["params"] = b["run"][0].params
    cf.dump_json(p, d, overwrite=True)
    f = s8.build_F(cfg, root)
    assert f["name"] == "8-F-8-T-CAT" and f["run"][0].params["params"]["learning_rate"] == 0.03
    _run(env, root / "s8" / "8-F", [rn.VariantSpec(**{**s.to_dict(), "params": {}}) for s in f["run"]])
    jf = s8.judge_F_core(cfg, root, env["st"])
    assert jf["record"]["parts"][0]["against"] == "8-T-CAT"               # Focal 상대 = 8-T 판정 뒤 현재 최고 판
    assert jf["record"]["parts"][0]["adopted"] is False                    # 같은 가짜 모델 → 같은 CV → 안 오름
    assert [h["step"] for h in jf["state"]["history"]] == ["8-T", "8-F"]
