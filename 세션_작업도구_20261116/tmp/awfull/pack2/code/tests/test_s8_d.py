"""s8_d(단계 8-D 딥러닝 튜닝) 시험 — 합성 자료·가짜 모델·가짜 점수: 설정 기본값이면 모델이 그대로(층·예측 같음)·손실 식,
범위·raw → params, 묶어 묻기 20회(t000 + 19)·앞 10회 = 차례 탐색과 같은 값·이어하기, fold 단위 기록·이어하기(평가 쌍 입력
묶음마다 한 번), fold 점수 = score_cv, 기본 유지·채택 판정(fold 2개 이상 규칙), 딥러닝 없음이면 상태 그대로, 판 설정."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lst import config as cf
from lst import evaluate as ev
from lst import folds as fd
from lst import models_dl as md
from lst import runner as rn
from lst import stage4 as s4
from lst import tuning as tu
from lst import tuning_dl as tdl
from lst import wiring as wr
from tests.test_runner import STUB, FakeAssembler, FakeModel, env  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import s8_d  # noqa: E402

pytest.importorskip("optuna")


# ---------------------------------------------------------------- 모델 설정(models_dl 구현 세부 16)
def _dl_data(n=240, W=12, C=3, S=2, seed=0):
    rng = np.random.default_rng(seed)
    x = {"x_seq": rng.normal(size=(n, W, C)).astype(np.float32), "x_static": rng.normal(size=(n, S)).astype(np.float32),
         "x_cat": np.stack([rng.integers(0, m, n) for m in (3, 5, 4, 6)], 1).astype(np.int64)}
    y = (x["x_seq"][:, -1, 0] + rng.normal(scale=0.5, size=n) > 0.8).astype(np.int64)
    groups = (np.array([f"s{i % 7}" for i in range(n)]), np.arange(n) // 3)
    return x, y, groups


@pytest.mark.parametrize("arch", md.CLASSIFIER_ARCHS)
def test_default_hparams_same_model(arch):
    """설정 없음 = 기본값을 다 적은 설정: 층 이름·모양·예측이 같음(기본 판 동작 그대로)."""
    x, y, g = _dl_data()
    a = md.DLClassifier(arch, (3, 5, 4, 6), device="cpu", max_epochs=2).fit(x, y, seed=0, n_jobs=1, groups=g)
    hp = {k: v for k, v in md.default_hparams(arch).items() if v is not None}
    b = md.DLClassifier(arch, (3, 5, 4, 6), device="cpu", max_epochs=2, hparams=hp).fit(x, y, seed=0, n_jobs=1,
                                                                                      groups=g)
    sa, sb = a.net.state_dict(), b.net.state_dict()
    assert list(sa) == list(sb) and all(sa[k].shape == sb[k].shape for k in sa)
    assert np.array_equal(a.predict_proba(x), b.predict_proba(x))
    assert "hparams" not in a.get_config() and b.get_config()["hparams"]["hidden"] == 64


def test_hparams_change_model_and_save_load(tmp_path):
    x, y, g = _dl_data()
    hp = {"lr": 3e-3, "hidden": 32, "layers": 2, "dropout": 0.2, "pos_weight": 3.0, "loss": "focal",
          "focal_gamma": 2, "focal_alpha": 0.25}
    m = md.DLClassifier("LSTM", (3, 5, 4, 6), device="cpu", max_epochs=2, batch_size=64, hparams=hp)
    m.fit(x, y, seed=0, n_jobs=1, groups=g)
    assert m.net.rnn.hidden_size == 32 and m.net.rnn.num_layers == 2 and m.lr == 3e-3
    cfg = m.get_config()
    assert cfg["lr"] == 3e-3 and cfg["batch_size"] == 64 and cfg["loss"].startswith("focal(gamma=2.0,alpha=0.25)")
    p = m.save(tmp_path / "m.pt")
    m2 = md.DLClassifier.load(p, device="cpu")
    assert m2.hparams == m.hparams and np.array_equal(m2.predict_proba(x), m.predict_proba(x))
    with pytest.raises(ValueError, match="layers"):
        md.normalize_hparams("TCN", {"layers": 2})
    with pytest.raises(ValueError, match="4 의 배수"):
        md.normalize_hparams("TRF", {"hidden": 34})
    with pytest.raises(ValueError, match="focal_gamma"):
        md.normalize_hparams("LSTM", {"loss": "focal"})
    with pytest.raises(ValueError, match="모르는 칸"):
        md.normalize_hparams("LSTM", {"momentum": 0.9})


def test_loss_formula():
    logit = torch.tensor([2.0, -1.0, 0.5, -3.0])
    y = torch.tensor([1.0, 0.0, 0.0, 1.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    assert md.train_rows_loss(logit, y, bce, None) is bce
    base = md.normalize_hparams("LSTM", {})
    assert md.train_rows_loss(logit, y, bce, base) is bce                       # 기본(bce·가중 1) = 같은 텐서
    w = md.train_rows_loss(logit, y, bce, md.normalize_hparams("LSTM", {"pos_weight": 3.0}))
    assert torch.allclose(w, bce * torch.tensor([3.0, 1.0, 1.0, 3.0]))
    f = md.train_rows_loss(logit, y, bce, md.normalize_hparams("LSTM", {"loss": "focal", "focal_gamma": 2,
                                                                        "focal_alpha": 0.25, "pos_weight": 2.0}))
    p = torch.sigmoid(logit)
    pt = torch.where(y == 1, p, 1 - p)
    at = torch.where(y == 1, torch.tensor(0.25), torch.tensor(0.75))
    assert torch.allclose(f, at * (1 - pt) ** 2 * bce * torch.where(y == 1, 2.0, 1.0))


def test_gpu_fraction_setting():
    try:
        md.set_gpu_fraction(0.25)
        assert md.gpu_fraction() == 0.25
        with pytest.raises(ValueError):
            md.set_gpu_fraction(0.5)
    finally:
        md.set_gpu_fraction(None)
    assert md.gpu_fraction() == 0.4


def test_wiring_dl_params():
    sp = rn.VariantSpec(name="8-D-LSTM", stage="8", order="2차", kind="dl", algorithm="LSTM", W=20, mode=None,
                        input_set="x", role="후보", train_count=4, params={})
    assert wr.dl_model_kwargs(sp, {"max_epochs": 2}) == {"max_epochs": 2}
    sp2 = rn.VariantSpec(**{**sp.to_dict(), "params": {"dl": {"batch_size": 512, "lr": 1e-4, "hidden": 32}}})
    assert wr.dl_model_kwargs(sp2) == {"batch_size": 512, "hparams": {"lr": 1e-4, "hidden": 32}}
    with pytest.raises(ValueError, match="모르는 칸"):
        wr.dl_model_kwargs(rn.VariantSpec(**{**sp.to_dict(), "params": {"max_iter": 3}}))


# ---------------------------------------------------------------- 범위·묶어 묻기(tuning_dl d2·d3·d4)
def test_space_and_params():
    import optuna

    assert "layers" not in tdl.search_space("TCN") and tdl.search_space("TRF")["layers"]["high"] == 3
    st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    for arch in md.CLASSIFIER_ARCHS:
        for _ in range(15):
            raw = tdl.suggest(st.ask(), arch)
            p = tdl.dl_params(arch, raw)["dl"]
            assert p["batch_size"] in tdl.BATCH_CHOICES and 1e-4 <= p["lr"] <= 3e-3 and 1 <= p["pos_weight"] <= 4
            assert ("focal_gamma" in p) == (p["loss"] == "focal")
            md.normalize_hparams(arch, {k: v for k, v in p.items() if k != "batch_size"})


class FakeBatch:
    """가짜 묶음 점수: lr·hidden 에 따라 다른 CV, fail_batch 에서 한 번 멈춤."""
    def __init__(self, fail_batch=None):
        self.calls, self.fail_batch = [], fail_batch

    def __call__(self, items, bi):
        if self.fail_batch == bi:
            self.fail_batch = None
            raise MemoryError("가짜 메모리 오류")
        self.calls.append((bi, [n for n, _ in items]))
        return {n: {"cv": 0.03 + 0.001 * np.sin(p["dl"]["lr"] * 3000 + p["dl"]["hidden"]),
                    "fold_scores": {"fold1": 0.1, "fold2": 0.1, "fold3": 0.1}} for n, p in items}


def test_run_batched_counts_resume_and_startup(tmp_path):
    f = FakeBatch()
    recs = tdl.run_batched("LSTM", f, log_path=tmp_path / "a.jsonl", batches=(10, 3, 3, 3), label="LSTM")
    assert len(recs) == 19 and [len(c[1]) for c in f.calls] == [10, 3, 3, 3]
    lines = [json.loads(x) for x in (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event"] == "study" and lines[0]["seed"] == 42 and lines[0]["batches"] == [10, 3, 3, 3]
    assert sum(r["event"] == "ask" for r in lines) == 19 and sum(r["event"] == "trial" for r in lines) == 19
    # 다시 부르면 점수 계산 없이 같은 기록
    f2 = FakeBatch()
    assert tdl.run_batched("LSTM", f2, log_path=tmp_path / "a.jsonl", batches=(10, 3, 3, 3), label="LSTM") == recs
    assert f2.calls == []
    # 3번째 묶음에서 멈춤 → 이어 하면 남은 묶음만, 결과 같음
    f3 = FakeBatch(fail_batch=3)
    with pytest.raises(MemoryError):
        tdl.run_batched("LSTM", f3, log_path=tmp_path / "b.jsonl", batches=(10, 3, 3, 3), label="LSTM")
    f4 = FakeBatch()
    r2 = tdl.run_batched("LSTM", f4, log_path=tmp_path / "b.jsonl", batches=(10, 3, 3, 3), label="LSTM")
    assert [c[0] for c in f4.calls] == [3, 4] and [r["raw"] for r in r2] == [r["raw"] for r in recs]
    # 앞 10회(TPE 무작위 시작)는 차례로 물어도 같은 값(구현 세부 d2)
    seq = tdl.run_batched("LSTM", FakeBatch(), log_path=tmp_path / "c.jsonl", batches=(1,) * 19, label="LSTM")
    assert [r["raw"] for r in seq[:10]] == [r["raw"] for r in recs[:10]]
    # 탐색 설정이 바뀌면 멈춤
    with pytest.raises(tdl.DLTuningError, match="탐색 설정"):
        tdl.run_batched("LSTM", FakeBatch(), log_path=tmp_path / "a.jsonl", batches=(10, 9), label="LSTM")
    with pytest.raises(tdl.DLTuningError, match="batches"):
        tdl.check_batches((10, 3), 19)


# ---------------------------------------------------------------- fold 단위 점수·이어하기(DLTrialScorer)
class ParamModel(FakeModel):
    """params 에 따라 예측이 달라지는 가짜 모델(lr 이 클수록 잡음 큼). fail_on = (fold, 번호 params lr) 학습에서 한 번 멈춤."""
    fail: list = []

    def fit(self, X, y, *, seed, n_jobs, eval_set=None, groups=None):
        if ParamModel.fail and ParamModel.fail[0] == (self.split, self.spec.params.get("dl", {}).get("lr")):
            ParamModel.fail.clear()
            raise MemoryError("가짜")
        return super().fit(X, y, seed=seed, n_jobs=n_jobs)

    def predict_proba(self, X, n_iter=None):
        p = super().predict_proba(X)
        lr = float(self.spec.params.get("dl", {}).get("lr", 0.0))
        return np.clip(p + lr * np.random.default_rng(len(X)).random(len(X)), 0, 1)


def _scorer(env, out, counter):
    spec = rn.VariantSpec(name="8-D-LSTM", stage="8", order="2차", kind="dl", algorithm="LSTM", W=20, mode=None,
                          input_set="x", role="후보", train_count=4, params={})
    known = env["st"].known_frame()

    def make_inputs(sp, split):
        counter["inputs"] += 1
        return FakeAssembler(sp, split)

    def make_model(sp, asm):
        m = ParamModel(sp, asm)
        m.split = asm.split
        return m

    return tdl.DLTrialScorer(spec=spec, samples=env["st"], make_inputs=make_inputs, make_model=make_model,
                             fit_kwargs=lambda sp, f, rows, X: {"groups": None},
                             train_rows_fn=lambda f: rn.add_bounds(fd.make_train_rows(known, fd.SPLITS[f]), 20),
                             fold_log=out / "folds.jsonl", n_jobs=8, seed=42, check=lambda: None)


def test_scorer_fold_resume_and_score_cv(env):
    out = env["tmp"] / "tune"
    out.mkdir()
    cnt = {"inputs": 0}
    items = [(0, {}), (1, {"dl": {"lr": 0.01}}), (2, {"dl": {"lr": 0.2}})]
    ParamModel.fail = [("fold2", 0.2)]
    sc1 = _scorer(env, out, cnt)
    with pytest.raises(MemoryError):
        sc1(items, 1)
    done = sc1.done()
    assert set(done) == {(0, "fold1"), (1, "fold1"), (2, "fold1")}       # fold1 은 기록됨, fold2 는 통째로 다시
    sc2 = _scorer(env, out, cnt)
    res = sc2(items, 1)
    assert sc2.n_fits == 6 and sc2.n_passes == 2                          # 남은 fold 2·3 만(시도 3개 × 2)
    passes = [json.loads(x) for x in (out / "folds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(r["event"] == "pass" for r in passes) == 3                 # fold 마다 평가 쌍 입력 한 번(묶음 공유)
    # fold 점수 = evaluate.score_cv 의 fold 값(같은 예측으로)
    preds = {}
    for f in ev.CV_FOLDS:
        rows = rn.add_bounds(fd.make_train_rows(env["st"].known_frame(), fd.SPLITS[f]), 20)
        asm = FakeAssembler(None, f).fit(rows)
        sp = rn.VariantSpec(name="x", stage="8", order="2차", kind="dl", algorithm="LSTM", W=20, mode=None,
                            input_set="x", role="후보", train_count=4, params={"dl": {"lr": 0.01}})
        m = ParamModel(sp, asm)
        m.split = f
        m.fit(asm.transform(rows), rows["y"].to_numpy(), seed=42, n_jobs=8)
        preds[f] = rn.predict_pairs(m, asm, env["st"], f, 20)
    cv = ev.score_cv(preds, env["st"])
    assert res[1]["fold_scores"] == pytest.approx(cv["fold_band_weighted"], abs=1e-12)
    assert res[1]["cv"] == pytest.approx(cv["cv_band_weighted"], abs=1e-12)
    assert res[0]["cv"] != res[2]["cv"]


# ---------------------------------------------------------------- tune_part: 기본 유지·채택·t000 대조
def _part(seed=0):
    sp = rn.VariantSpec(name="5-LSTM", stage="5-6", order="2차", kind="dl", algorithm="LSTM", W=20, mode=None,
                        input_set="x", role="후보", train_count=4, params={})
    return s8_d.DLPart(label="LSTM", member={"name": "5-LSTM", "dir": "x"}, src=sp, plan=s4.RowPlan(), seed=seed,
                       base_params={})


class FakeScore:
    def __init__(self, value, t000=0.03):
        self.value, self.t000, self.calls = value, t000, []

    def __call__(self, items, bi):
        self.calls.append([n for n, _ in items])
        return {n: {"cv": (self.t000 if n == 0 else self.value(n)), "fold_scores": {f: 0.03 for f in ev.CV_FOLDS}}
                for n, _ in items}


def test_tune_keep_default(tmp_path):
    f = FakeScore(lambda n: 0.03004)                    # 4자리 반올림 0.0300 = t000 → 기본 유지
    r = s8_d.tune_part(_part(), tmp_path, score_fn=f, base_cv=lambda: 0.03, batches=(10, 3, 3, 3))
    assert r["adopt"] is False and r["selection"]["judgement"] == "기본 유지" and r["params"] is None
    assert r["n_trials_total"] == 20 and f.calls[0][0] == 0 and len(f.calls[0]) == 11     # 첫 묶음 = t000 대조 + 10
    assert r["t000_check"]["abs_diff"] == 0 and not r["t000_check"]["alert"]
    assert (tmp_path / "top5_LSTM.csv").exists() and len(pd.read_csv(tmp_path / "trials_LSTM.csv")) == 19
    f2 = FakeScore(lambda n: 0.5)
    assert s8_d.tune_part(_part(), tmp_path, score_fn=f2, base_cv=lambda: 0.03, batches=(10, 3, 3, 3)) == r
    assert f2.calls == []                                # 끝난 튜닝은 그대로


def test_tune_adopt_best_and_alert(tmp_path):
    f = FakeScore(lambda n: 0.03 + 0.0001 * (n % 7), t000=0.0312)
    r = s8_d.tune_part(_part(), tmp_path, score_fn=f, base_cv=lambda: 0.03, batches=(10, 3, 3, 3))
    assert r["adopt"] and r["selection"]["best_trial"] == "t006" and r["params"]["dl"]["batch_size"] in tdl.BATCH_CHOICES
    assert r["t000_check"]["alert"] is True              # 다시 학습한 t000 이 대상 판 CV 와 다름 → 알림(고르기는 그대로)
    assert r["selection"]["baseline_cv"] == 0.03


# ---------------------------------------------------------------- 대상·판 설정·판정
def _variant(root: Path, rel: str, spec: rn.VariantSpec, cv=None, folds=None):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    cf.dump_json(d / "spec.json", spec.to_dict(), overwrite=True)
    cf.dump_json(d / "done.json", {"name": spec.name}, overwrite=True)
    if cv is not None:
        cf.dump_json(d / "cv_scores.json", {"cv_band_weighted": cv, "fold_band_weighted": folds,
                                            "ties": {"many_ties": False}}, overwrite=True)
    return rel


def _dl_spec(name, alg="LSTM", extra=None, params=None):
    return rn.VariantSpec(name=name, stage="5-6", order="2차", kind="dl", algorithm=alg, W=20, mode=None,
                          input_set="x", role="후보", train_count=4, params=params or {},
                          extra=extra or {"version": {"include_ids": ["WEATHER"]},
                                          "s4": {"row_plan": s4.RowPlan(("hon5",), "own", "s5_6/sel-hon5").to_dict(),
                                                 "model_seed": 0}})


def _state(root, best, path="logs/chain_state_8.json"):
    tt = best if best["kind"] == "tree" else {"name": "3-0-RF", "kind": "tree", "dir": "s3/3-0-T/3-0-RF",
                                              "replacement": False, "members": [{"name": "3-0-RF", "dir": "s3/3-0-T/3-0-RF"}]}
    st = {"step": "8-F", "current_best": best, "tree_target": tt, "tree_top3": [], "tree_rank_3_0b": {},
          "history": [{"step": "8-F"}]}
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).unlink(missing_ok=True)
    s4.save_state(root / path, st)
    return path


def _cfg(state_in="logs/chain_state_8.json", fold_majority=True):
    return {"chain_state_in": state_in, "chain_state_out": "logs/chain_state_8D.json", "fold_majority": fold_majority,
            "dirs": {"8-D-tune": "s8/8-D-tune", "8-D": "s8/8-D", "8-D-judge": "s8/8-D-judge"}}


def _tune_file(root, adopt, params=None):
    d = root / "s8" / "8-D-tune"
    d.mkdir(parents=True, exist_ok=True)
    cf.dump_json(d / "tune_LSTM.json", {"label": "LSTM", "adopt": adopt, "params": params, "t000": {"cv": 0.0346},
                                        "t000_check": None,
                                        "selection": {"best_cv": 0.0361, "best_trial": "t004", "top": []}},
                 overwrite=True)


def _target(root):
    rel = _variant(root, "s5_6/5-6-T/5-LSTM", _dl_spec("5-LSTM"), cv=0.0346,
                   folds={"fold1": 0.03, "fold2": 0.035, "fold3": 0.0388})
    m = {"name": "5-LSTM", "dir": rel, "anomaly": None,
         "row_plan": s4.RowPlan(("hon5",), "own", "s5_6/sel-hon5").to_dict(), "catenc_k": None}
    return {"name": "5-LSTM", "kind": "dl", "dir": rel, "replacement": False, "members": [m]}


def test_parts_and_build_D(tmp_path):
    root = tmp_path
    _state(root, _target(root))
    cfg = _cfg()
    parts = s8_d.dl_parts(rn.read_json(root / cfg["chain_state_in"]), root)
    assert len(parts) == 1 and parts[0].label == "LSTM" and parts[0].seed == 0 and parts[0].plan.steps == ("hon5",)
    with pytest.raises(s4.StageError, match="tune 먼저"):
        s8_d.build_D(cfg, root)
    _tune_file(root, False)
    assert s8_d.build_D(cfg, root)["status"] == "기본 유지" and s8_d.build_D(cfg, root)["run"] == []
    hp = {"dl": {"batch_size": 512, "lr": 5e-4, "hidden": 128, "layers": 2, "dropout": 0.1, "pos_weight": 2.0,
                 "loss": "bce"}}
    _tune_file(root, True, hp)
    b = s8_d.build_D(cfg, root)
    sp = b["run"][0]
    assert b["status"] == "판" and b["name"] == "8-D-LSTM" and sp.params == hp and sp.kind == "dl"
    assert sp.extra["s4"] == {"row_plan": s4.RowPlan(("hon5",), "own", "s5_6/sel-hon5").to_dict(), "model_seed": 0}
    assert sp.extra["version"] == {"include_ids": ["WEATHER"]} and b["best"]["dir"] == "s8/8-D/8-D-LSTM"
    assert wr.dl_model_kwargs(sp)["hparams"]["hidden"] == 128


def test_no_dl_part_passes_state(tmp_path):
    root = tmp_path
    for kind, alg in (("tree", "RF"), ("nn60", "NN60")):
        sp = rn.VariantSpec(name=f"x-{alg}", stage="5-6", order="2차", kind="tree" if kind == "tree" else "dl",
                            algorithm=alg, W=20, mode="sum" if kind == "tree" else None, input_set="x", role="후보",
                            train_count=4)
        rel = _variant(root, f"s5/x-{alg}", sp, cv=0.03, folds={f: 0.03 for f in ev.CV_FOLDS})
        best = {"name": f"x-{alg}", "kind": kind, "dir": rel, "replacement": kind == "nn60",
                "members": [{"name": f"x-{alg}", "dir": rel}]}
        _state(root, best)
        assert s8_d.dl_parts(rn.read_json(root / "logs/chain_state_8.json"), root) == []
        r = s8_d.judge_core(_cfg(), root, None)
        assert r["built"]["status"] == "튜닝 안 함" and r["state"]["current_best"] == best
        assert r["state"]["history"][-1]["judgement"] == "튜닝 안 함" and r["record"]["after"] == f"x-{alg}"
    # 앙상블 구성에 딥러닝 → 멈춤(지원 안 함)
    rel = _variant(root, "s5/x-LSTM", _dl_spec("x-LSTM"))
    rel2 = _variant(root, "s5/x-RF2", rn.VariantSpec(name="x-RF2", stage="5-6", order="2차", kind="tree",
                                                     algorithm="RF", W=20, mode="sum", input_set="x", role="후보",
                                                     train_count=4))
    _state(root, {"name": "E", "kind": "ensemble", "dir": rel, "replacement": False,
                  "members": [{"name": "x-LSTM", "dir": rel}, {"name": "x-RF2", "dir": rel2}]})
    with pytest.raises(s4.StageError, match="앙상블"):
        s8_d.dl_parts(rn.read_json(root / "logs/chain_state_8.json"), root)


def test_judge_keep_and_adopt_rules(tmp_path):
    root = tmp_path
    tgt = _target(root)
    _state(root, tgt)
    _tune_file(root, False)
    r = s8_d.judge_core(_cfg(), root, None)
    assert r["built"]["status"] == "기본 유지" and r["state"]["current_best"]["name"] == "5-LSTM"
    assert r["record"]["tune"]["best_trial"] == "t004"
    hp = {"dl": {"batch_size": 512, "lr": 5e-4, "hidden": 128, "layers": 2, "dropout": 0.1, "pos_weight": 2.0,
                 "loss": "bce"}}
    _tune_file(root, True, hp)
    # CV 는 오르지만 fold 1개만 오름 → fold_majority 면 버림, 아니면 채택
    _variant(root, "s8/8-D/8-D-LSTM", s8_d.build_D(_cfg(), root)["run"][0], cv=0.0361,
             folds={"fold1": 0.0299, "fold2": 0.0349, "fold3": 0.0435})
    r1 = s8_d.judge_core(_cfg(fold_majority=True), root, None)
    assert r1["record"]["adopted"] is False and r1["state"]["current_best"]["name"] == "5-LSTM"
    assert r1["record"]["parts"][0]["fold_rule"]["n_fold_up"] == 1 and r1["record"]["parts"][0]["judge"]["adopt"]
    r2 = s8_d.judge_core(_cfg(fold_majority=False), root, None)
    assert r2["record"]["adopted"] is True and r2["state"]["current_best"]["name"] == "8-D-LSTM"
    assert r2["state"]["current_best"]["dir"] == "s8/8-D/8-D-LSTM" and r2["state"]["tree_target"] == r1["state"]["tree_target"]
    assert r2["state"]["history"][-1]["judgement"] == "채택"
    # fold 2개 이상 오르면 fold_majority 여도 채택
    _variant(root, "s8/8-D/8-D-LSTM", s8_d.build_D(_cfg(), root)["run"][0], cv=0.0361,
             folds={"fold1": 0.0301, "fold2": 0.0351, "fold3": 0.0431})
    r3 = s8_d.judge_core(_cfg(fold_majority=True), root, None)
    assert r3["record"]["adopted"] is True and r3["record"]["trial_vs_variant_cv"]["abs_diff"] == 0
    # CV 가 4자리로 같으면 안 오름
    _variant(root, "s8/8-D/8-D-LSTM", s8_d.build_D(_cfg(), root)["run"][0], cv=0.03464,
             folds={"fold1": 0.0301, "fold2": 0.0351, "fold3": 0.0388})
    assert s8_d.judge_core(_cfg(fold_majority=True), root, None)["record"]["adopted"] is False


# ---------------------------------------------------------------- judge 진입점(설정 파일·사슬 기록·결정 파일)
def _write_cfg(root, name, body):
    p = root / name
    p.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return p


def test_judge_main_pass_through(env):
    root = env["inputs"]
    sp = rn.VariantSpec(name="x-RF", stage="5-6", order="2차", kind="tree", algorithm="RF", W=20, mode="sum",
                        input_set="x", role="후보", train_count=4)
    rel = _variant(root, "s5/x-RF", sp, cv=0.03, folds={f: 0.03 for f in ev.CV_FOLDS})
    best = {"name": "x-RF", "kind": "tree", "dir": rel, "replacement": False, "members": [{"name": "x-RF", "dir": rel}]}
    _state(root, best)
    base = json.loads((Path(s8_d.CODE_ROOT) / "configs" / "s8_D_judge.json").read_text(encoding="utf-8"))
    base["use_mlflow"] = False
    cp = _write_cfg(env["tmp"], "j.json", base)
    out = root / "s8" / "8-D-judge"
    d = s8_d.main(["--root", str(env["tmp"]), "--out", str(out), "--inputs", str(root), "--config", str(cp)])
    assert d["record"]["status"] == "튜닝 안 함" and d["record"]["after"] == "x-RF"
    st = rn.read_json(root / "logs" / "chain_state_8D.json")
    assert st["current_best"] == best and st["step"] == "8-D"
    line = json.loads((root / "logs" / "chain_log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert line["step"] == "8-D" and line["judgement"] == "튜닝 안 함"
    assert s8_d.main(["--root", str(env["tmp"]), "--out", str(out), "--inputs", str(root), "--config", str(cp)]) == d


def test_configs_8D():
    code = Path(s8_d.CODE_ROOT)
    for sub in ("tune", "D"):
        name = "s8_D_tune.json" if sub == "tune" else "s8_D.json"
        cfg = s8_d.load_cfg(code / "configs" / name)
        assert cfg["sub"] == sub and cfg["chain_state_in"] == "logs/chain_state_8.json"
        assert cfg["gpu_memory_fraction"] == 0.25 and cfg["dirs"]["8-D"] == "s8/8-D"
    assert s8_d.load_cfg(code / "configs" / "s8_D_tune.json")["tpe_batches"] == [10, 3, 3, 3]
    j = cf.load_config(code / "configs" / "s8_D_judge.json", s8_d.JUDGE_SCHEMA)
    assert j["chain_state_out"] == "logs/chain_state_8D.json" and j["fold_majority"] is False
    fin = cf.load_json_strict(code / "configs" / "s_final.json")
    assert fin["chain_state"] == "logs/chain_state_8D.json" and "s8/8-D" in fin["search_dirs"]
    # 국내판 설정(configs_dom) 대조는 뺌 — 국내판 폐기(사용자 2026-11-12 "멈추고 다 버려"), configs_dom 폴더 없음


def test_make_configs_dom_keeps_reviewed(tmp_path):
    import make_configs_dom as mcd

    (tmp_path / "configs").mkdir()
    for n, plan in (("a.json", "가"), ("b.json", "나")):
        (tmp_path / "configs" / n).write_text(json.dumps({"plan": plan, "code_reviewed": True, "x": 1}),
                                              encoding="utf-8")
    mcd.write(tmp_path)
    assert json.loads((tmp_path / "configs_dom" / "a.json").read_text(encoding="utf-8"))["code_reviewed"] is False
    for n in ("a.json", "b.json"):                      # 지휘자가 검수 뒤 true 로
        p = tmp_path / "configs_dom" / n
        d = json.loads(p.read_text(encoding="utf-8"))
        d["code_reviewed"] = True
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "configs" / "b.json").write_text(json.dumps({"plan": "나", "code_reviewed": True, "x": 2}),
                                                 encoding="utf-8")
    mcd.write(tmp_path)
    a = json.loads((tmp_path / "configs_dom" / "a.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "configs_dom" / "b.json").read_text(encoding="utf-8"))
    assert a["code_reviewed"] is True and b["code_reviewed"] is False and b["x"] == 2   # 바뀐 파일만 false
    assert mcd.check(tmp_path) == []


# ---------------------------------------------------------------- tune 진입점(설정·train 행·실험 로그) — 가짜 wiring
class _FakeWiring:
    def __init__(self):
        self.n_inputs = 0

    def make_inputs(self, sp, split):
        self.n_inputs += 1
        return FakeAssembler(sp, split)

    def make_model(self, sp, asm):
        m = ParamModel(sp, asm)
        m.split = asm.split
        return m

    def fit_kwargs(self, sp, split, rows, X):
        assert {"T", "E", "g", "R"} <= set(rows.columns)
        return {"groups": (rows["sgg"].to_numpy(), rows["d"].to_numpy())}


def test_tune_main_end_to_end(env, monkeypatch):
    root = env["inputs"]
    sp = _dl_spec("3-0-LSTM", extra={"version": {"include_ids": ["WEATHER"]}})
    rel = _variant(root, "s3/3-0-D/3-0-LSTM", sp, cv=0.031, folds={f: 0.031 for f in ev.CV_FOLDS})
    _state(root, {"name": "3-0-LSTM", "kind": "dl", "dir": rel, "replacement": False,
                  "members": [{"name": "3-0-LSTM", "dir": rel, "anomaly": None}]})
    cfg = json.loads((Path(s8_d.CODE_ROOT) / "configs" / "s8_D_tune.json").read_text(encoding="utf-8"))
    cfg["use_mlflow"] = False
    cp = _write_cfg(env["tmp"], "t.json", cfg)
    monkeypatch.setattr(s8_d, "check_resources", STUB)
    fw = _FakeWiring()
    out = root / "s8" / "8-D-tune"
    argv = ["--root", str(env["tmp"]), "--out", str(out), "--inputs", str(root), "--config", str(cp)]
    r = s8_d.main(argv, wiring_fn=lambda *a: fw)
    assert r["n_trials_total"] == 20 and r["t000"]["cv"] == 0.031 and r["batches"] == [10, 3, 3, 3]
    assert r["t000_check"] is not None and fw.n_inputs == 3                     # fold 마다 train 입력 한 번
    folds = [json.loads(x) for x in (out / "dl_folds_LSTM.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(x["event"] == "pass" for x in folds) == 4 * 3                    # 묶음 4 × fold 3
    assert sum(x["event"] == "fold" for x in folds) == (19 + 1) * 3
    log = rn.ExperimentLog(root / "logs" / "experiment_log.jsonl")
    assert log.count_for("8", "8-D-tune-LSTM", kind="시도") == 0
    assert [e for e in log.entries() if e.get("event") == "cv_scored"][-1]["n_settings"] == 19
    assert (out / "_code" / "code_md5.txt").exists()
    assert s8_d.main(argv, wiring_fn=lambda *a: pytest.fail("다시 안 함")) == r     # 끝난 튜닝은 그대로
    # D 판 목록: 기본 유지면 판 없음, 채택이면 8-D-LSTM 하나
    dcfg = json.loads((Path(s8_d.CODE_ROOT) / "configs" / "s8_D.json").read_text(encoding="utf-8"))
    dcfg["use_mlflow"] = False
    dp = _write_cfg(env["tmp"], "d.json", dcfg)
    names = s8_d.main(["--root", str(env["tmp"]), "--out", str(root / "s8" / "8-D"), "--inputs", str(root),
                       "--config", str(dp), "--list"])
    assert names == (["8-D-LSTM"] if r["adopt"] else [])
