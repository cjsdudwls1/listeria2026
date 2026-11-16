"""lst.tuning 시험(합성 자료·작은 모델) — 부록 A-7 범위, Optuna 이어하기, 시도 점수(§3 교차 N), Focal, AutoKeras(가짜 모듈)."""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from lst import folds as fd
from lst import metrics as mt
from lst import models_tree as mtree
from lst import tuning as tu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _imp_tune_synth import SPLITS, FrameAsm, make_samples  # noqa: E402

optuna = pytest.importorskip("optuna")


@pytest.fixture(scope="module")
def data():
    return make_samples(n=700, seed=3)


@pytest.fixture(scope="module")
def folds(data):
    samples, feats = data
    return [tu.TreeFold(f, samples, FrameAsm(feats), split=SPLITS[f]) for f in fd.CV_FOLDS]


# ---------------------------------------------------------------- 범위
def _sample(algo, n=60, **kw):
    st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    out = []
    for _ in range(n):
        t = st.ask()
        out.append(tu.suggest_params(t, algo, **kw))
        st.tell(t, 0.0)
    return out


def test_space_matches_a7():
    sp = tu.search_space("CAT")
    assert sp["learning_rate"] == {"kind": "float", "low": 0.005, "high": 0.3, "log": True}
    assert sp["depth"] == {"kind": "int", "low": 2, "high": 10, "log": False}
    assert sp["bootstrap_type"] == {"kind": "fixed", "value": "Bernoulli"}
    assert sp["scale_pos_weight"]["low"] == 1 and sp["scale_pos_weight"]["high"] == 300
    assert tu.search_space("LGBM")["num_leaves"] == {"kind": "int", "low": 4, "high": 1024, "log": True}
    assert tu.search_space("LGBM")["bagging_freq"]["value"] == 1
    assert tu.search_space("LR")["C"]["low"] == 1e-4 and tu.search_space("LR")["C"]["high"] == 1e2
    # 4-a: 행 수 기준 범위 × 복제 수
    assert tu.search_space("LGBM", rep=60)["min_child_samples"] == {"kind": "int", "low": 300, "high": 30000,
                                                                     "log": True}
    assert tu.search_space("XGB", rep=30)["min_child_weight"]["high"] == 3000
    assert tu.search_space("RF", rep=60)["min_samples_leaf"]["high"] == 12000
    assert tu.search_space("XGB", rep=60)["max_depth"]["high"] == 10          # 행 수 기준 아님
    assert "learning_rate" not in tu.search_space("CAT", tune_lr=False)       # ④


@pytest.mark.parametrize("algo", tu.TREE_ALGOS)
def test_suggest_in_range_and_model_accepts(algo):
    sp = tu.search_space(algo)
    seen_unl = set()
    for raw in _sample(algo):
        for k, v in raw.items():
            d = sp[k]
            if d["kind"] == "bool":
                seen_unl.add(v)
                continue
            assert d["low"] <= v <= d["high"]
            assert isinstance(v, int) == (d["kind"] == "int")
        if algo in ("RF", "ET"):
            assert ("max_depth" in raw) == (not raw["max_depth_unlimited"])
        p = tu.model_params(algo, raw)
        mtree.make_model(algo, cat_cols=[], max_iter=200 if algo in mtree.BOOSTING else None, params=p)
        if algo in ("RF", "ET"):
            assert p["class_weight"][0] == 1.0 and (p["max_depth"] is None) == raw["max_depth_unlimited"]
    if algo in ("RF", "ET"):
        assert seen_unl == {True, False}


# ---------------------------------------------------------------- Optuna 이어하기
def _fake_score(p):
    v = sum(math.log1p(abs(float(x))) for k, x in p.items() if isinstance(x, (int, float)) and not isinstance(x, bool))
    return {"cv": v / 10.0, "fold_scores": {"fold1": v}}


def test_optuna_resume_is_identical(tmp_path):
    a = tu.run_optuna("LGBM", _fake_score, log_path=tmp_path / "a.jsonl", n_trials=14)
    b1 = tu.run_optuna("LGBM", _fake_score, log_path=tmp_path / "b.jsonl", n_trials=14, stop_after=5)
    assert len(b1) == 5
    calls = []

    def score2(p):
        calls.append(p)
        return _fake_score(p)

    b = tu.run_optuna("LGBM", score2, log_path=tmp_path / "b.jsonl", n_trials=14)
    assert len(calls) == 9                                                  # 끝난 5 시도는 다시 안 돎
    assert [r["raw"] for r in a] == [r["raw"] for r in b] and [r["value"] for r in a] == [r["value"] for r in b]
    assert [r["number"] for r in b] == list(range(1, 15))
    # 기록을 건드리면 되풀이 불일치로 멈춤
    lines = (tmp_path / "b.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[3])
    rec["raw"]["num_leaves"] = rec["raw"]["num_leaves"] + 1
    lines[3] = json.dumps(rec, ensure_ascii=False)
    (tmp_path / "c.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(tu.TuningError):
        tu.run_optuna("LGBM", _fake_score, log_path=tmp_path / "c.jsonl", n_trials=14)
    with pytest.raises(tu.TuningError):                                     # 탐색 설정이 다르면 멈춤
        tu.run_optuna("LGBM", _fake_score, log_path=tmp_path / "a.jsonl", n_trials=14, rep=60)
    with pytest.raises(tu.TuningError):                                     # NaN 은 실패로 넘기지 않음
        tu.run_optuna("CAT", lambda p: {"cv": float("nan")}, log_path=tmp_path / "d.jsonl", n_trials=2)


def test_select_tuned_rounding_and_keep_default():
    recs = [{"event": "trial", "number": 1, "value": 0.12341, "params": {}},
            {"event": "trial", "number": 2, "value": 0.12344, "params": {}},
            {"event": "trial", "number": 3, "value": 0.10, "params": {}}]
    s = tu.select_tuned(0.1234, recs)
    assert not s["adopt"] and s["judgement"] == "기본 유지" and s["best_trial"] == "t001"   # 반올림 같음 → 작은 번호
    s = tu.select_tuned(0.1233, recs)
    assert s["adopt"] and s["selected"]["number"] == 1 and s["top"][0]["trial"] == "t000"
    assert [r["trial"] for r in s["top"][1:]] == ["t001", "t002", "t003"]


# ---------------------------------------------------------------- 시도 점수(§3)
def test_score_boosting_cross_n(folds):
    r = tu.score_tree_config("XGB", {"max_depth": 3, "scale_pos_weight": 5.0}, folds, max_iter=200, n_jobs=1)
    assert r["points"] == mtree.checkpoints(200) and not r["partial"]
    curves = {f: np.array(c) for f, c in r["curves"].items()}
    sel = mtree.select_n_iter(curves, r["points"])
    assert r["cv"] == sel["cv_mean"] and r["cross_n"] == sel["cross_n"] and r["common_n"] == sel["common_n"]
    # 곡선 한 점 = 그 트리 수 예측의 구간 가중 PR-AUC
    fo = folds[0]
    m = mtree.make_model("XGB", cat_cols=fo.cat_cols, max_iter=200, params={"max_depth": 3, "scale_pos_weight": 5.0})
    m.fit(fo.X, fo.y, seed=42, n_jobs=1)
    byh = np.full(60, np.nan)
    for h, Xh, yh, uh in fo.iter_val():
        byh[h - 1] = mt.pr_auc(yh, m.predict_proba(Xh, n_iter=50), uh)
    assert curves["fold1"][r["points"].index(50)] == pytest.approx(mt.band_weighted(byh), abs=1e-15)
    assert "save_config" in r["applied"]["fold1"]["applied_params"]
    # 0-9: fold 하나 = partial(자기 곡선 최고 시점)
    p = tu.score_tree_config("XGB", {"max_depth": 3}, folds[2:], max_iter=200, n_jobs=1)
    assert p["partial"] and set(p["own_best_n"]) == {"fold3"}


def test_score_plain_rf_lr(folds):
    r = tu.score_tree_config("RF", tu.model_params("RF", {"max_depth_unlimited": False, "max_depth": 5,
                                                          "min_samples_leaf": 3, "max_features": 0.5,
                                                          "pos_weight": 10.0}), folds, max_iter=None, n_jobs=1)
    assert np.isfinite(r["cv"]) and r["cv"] == pytest.approx(np.mean(list(r["fold_scores"].values())))
    lr = tu.score_tree_config("LR", tu.model_params("LR", {"C": 0.5, "pos_weight": 3.0}), folds, max_iter=None,
                              n_jobs=1)
    assert np.isfinite(lr["cv"])
    with pytest.raises(tu.TuningError):
        tu.score_tree_config("RF", {}, folds, max_iter=200)


def test_focal(folds):
    assert tu.FOCAL_LOSS == "Focal:focal_alpha=0.75;focal_gamma=2"
    p = tu.focal_params({"learning_rate": 0.05, "depth": 4})
    assert p == {"learning_rate": 0.05, "depth": 4, "loss_function": tu.FOCAL_LOSS}
    with pytest.raises(tu.TuningError):
        tu.focal_params({"loss_function": "CrossEntropy"})
    s = tu.focal_supported()
    assert s["supported"], s
    r = tu.score_tree_config("CAT", tu.focal_params({"learning_rate": 0.1}), folds[:1], max_iter=200, n_jobs=1)
    assert np.isfinite(r["cv"]) and "Focal" in json.dumps(r["applied"], ensure_ascii=False)


def test_tune_split_before_all_folds():
    s = tu.TUNE_SPLIT
    assert fd.as_date([s.train_end])[0] == np.datetime64("2021-06-30")
    assert fd.as_date([s.score_start, s.score_end]).tolist() == [np.datetime64("2021-07-01").astype(object),
                                                                 np.datetime64("2022-05-01").astype(object)]
    assert s.score_end < min(fd.SPLITS[f].score_start for f in fd.CV_FOLDS)


# ---------------------------------------------------------------- AutoKeras(가짜 모듈 — 실제는 서버 0-9)
class _Blk:
    log = []

    def __init__(self, **kw):
        self.kw = kw
        _Blk.log.append((type(self).__name__, kw))

    def __call__(self, x):
        return (type(self).__name__, self.kw, x)


def fake_ak():
    _Blk.log = []
    mod = types.SimpleNamespace()
    for nm in ("Input", "RNNBlock", "ConvBlock", "Embedding", "Merge", "DenseBlock", "ClassificationHead"):
        setattr(mod, nm, type(nm, (_Blk,), {}))

    class AutoModel:
        last = None

        def __init__(self, **kw):
            self.kw = kw
            self.tuner = types.SimpleNamespace(oracle=types.SimpleNamespace(trials={}))
            AutoModel.last = self

        def fit(self, **kw):
            self.fit_kw = kw

    mod.AutoModel = AutoModel
    return mod


def fake_keras():
    class ES:
        def __init__(self, **kw):
            self.kw = kw

    class HP:
        def __init__(self):
            self.fixed = {}

        def Fixed(self, k, v):
            self.fixed[k] = v

    keras = types.SimpleNamespace(callbacks=types.SimpleNamespace(EarlyStopping=ES),
                                  utils=types.SimpleNamespace(set_random_seed=lambda s: None))
    return keras, types.SimpleNamespace(HyperParameters=HP)


def test_ak_graph_mapping():
    ak = fake_ak()
    nodes, out, keys = tu.build_ak_graph("LSTM", has_cat=True, cat_sizes=[3, 5, 4, 6], ak=ak)
    assert keys == ("x_seq", "x_static", "x_cat") and len(nodes) == 3
    names = [n for n, _ in _Blk.log]
    assert ("RNNBlock", {"layer_type": "lstm"}) in _Blk.log and ("Embedding", {"max_features": 18}) in _Blk.log
    assert names.count("DenseBlock") == 2
    assert out[0] == "ClassificationHead" and out[2][0] == "DenseBlock" and out[2][2][0] == "Merge"
    seq_branch, tab_branch = out[2][2][2]
    assert seq_branch[0] == "RNNBlock" and tab_branch[0] == "DenseBlock" and tab_branch[2][0] == "Merge"
    assert all(kw == {"merge_type": "concatenate"} for n, kw in _Blk.log if n == "Merge")
    _, _, keys = tu.build_ak_graph("TCN", has_cat=False, ak=fake_ak())
    assert keys == ("x_seq", "x_static") and _Blk.log[1][0] == "ConvBlock"
    for a in ("TRF", "CMBV"):
        with pytest.raises(tu.AKUnsupported):
            tu.build_ak_graph(a, has_cat=True, cat_sizes=[1, 1, 1, 1], ak=fake_ak())
    with pytest.raises(tu.AKUnsupported):
        tu.ak_mapping("LSTM", {"LSTM": {"supported": False, "reason": "담당자 결정"}})


def test_ak_prepare_inputs_offsets():
    xc = np.array([[0, 1, 2, 3], [2, 4, 0, 5]], np.int64)
    out = tu.ak_prepare_inputs({"x_cat": xc, "x_seq": np.zeros((2, 3, 1), np.float32)}, [3, 5, 4, 6])
    assert out["x_cat"].tolist() == [[0, 4, 10, 15], [2, 7, 8, 17]]
    with pytest.raises(tu.TuningError):
        tu.ak_prepare_inputs({"x_cat": np.array([[3, 0, 0, 0]], np.int64)}, [3, 5, 4, 6])


def test_autokeras_search_settings(tmp_path):
    ak = fake_ak()
    keras, kt = fake_keras()
    n = 400
    rng = np.random.default_rng(0)
    inputs = {"x_seq": rng.normal(size=(n, 5, 2)).astype(np.float32), "x_static": rng.normal(size=(n, 3)).astype(np.float32),
              "x_cat": rng.integers(0, 3, size=(n, 4)).astype(np.int64)}
    y = (rng.random(n) < 0.2).astype(int)
    groups = (np.array(["11110", "26110"])[rng.integers(0, 2, n)], rng.integers(16000, 16100, n))
    d = np.full(n, fd.day("2021-06-30"))
    s = tu.autokeras_search("GRU", inputs, y, groups, directory=tmp_path, project_name="p", cat_sizes=[3, 3, 3, 3],
                            train_d=d, ak=ak, keras=keras, keras_tuner=kt)
    am = ak.AutoModel.last
    assert am.kw["max_trials"] == 19 and am.kw["seed"] == 42 and am.kw["tuner"] == "greedy"
    assert am.kw["objective"] == "val_loss" and am.kw["hyperparameters"].fixed == {"optimizer": "adam"}
    assert am.kw["overwrite"] is False
    fk = am.fit_kw
    assert fk["batch_size"] == 1024 and fk["epochs"] == 20
    assert fk["callbacks"][0].kw == {"monitor": "val_loss", "patience": 3, "restore_best_weights": True}
    n_tr, n_va = len(fk["y"]), len(fk["validation_data"][1])
    assert n_tr + n_va == n and 0 < n_va < n_tr
    assert s.keys == ("x_seq", "x_static", "x_cat") and fk["x"][2].max() <= 11
    with pytest.raises(tu.TuningError):
        tu.autokeras_search("GRU", inputs, y, groups, directory=tmp_path, project_name="p", cat_sizes=[3, 3, 3, 3],
                            train_d=d + 1, ak=ak, keras=keras, keras_tuner=kt)


def test_disable_adaptive_batch():
    calls = []

    def adaptive(batch_size, func, **kw):  # AutoKeras 원본 흉내(메모리 부족이면 절반)
        raise AssertionError("원본이 불리면 안 됨")

    mod = types.SimpleNamespace(run_with_adaptive_batch_size=adaptive)
    assert tu.disable_adaptive_batch_size(mod) is True
    assert tu.disable_adaptive_batch_size(mod) is False

    def fit(**kw):
        calls.append(kw)
        raise MemoryError("oom")

    with pytest.raises(MemoryError):
        mod.run_with_adaptive_batch_size(1024, fit, x=1, validation_data=(2, 3))
    assert calls == [{"validation_data": (2, 3), "batch_size": 1024, "x": 1}]


def test_ak_trial_scoring_and_select(data):
    samples, feats = data

    class KM:
        def __init__(self, w):
            self.w = w

        def predict(self, xs, batch_size, verbose):
            return 1 / (1 + np.exp(-self.w * xs[1][:, 0:1].astype(np.float64)))

    class Trial:
        def __init__(self, tid, w, status="COMPLETED"):
            self.trial_id, self.w, self.status = tid, w, status
            self.hyperparameters = types.SimpleNamespace(values={"w": w})
            self.score = 0.5

    trials = {"01": Trial("01", -1.0), "00": Trial("00", 2.0)}
    tuner = types.SimpleNamespace(oracle=types.SimpleNamespace(trials=trials), load_model=lambda t: KM(t.w))
    s = tu.AKSearch("LSTM", ("x_seq", "x_static"), None, None, tuner, {})
    rng = np.random.default_rng(1)
    val = []
    for h in (1, 2, 30):
        n = 80
        st = rng.normal(size=(n, 2)).astype(np.float32)
        yv = (st[:, 0] + rng.normal(size=n) > 1).astype(int)
        val.append((h, {"x_seq": np.zeros((n, 2, 1), np.float32), "x_static": st}, yv, np.arange(n, dtype=np.uint64)))
    sc = tu.score_ak_trials(s, val)
    assert [x["trial"] for x in sc] == ["t01", "t02"] and [x["trial_id"] for x in sc] == ["00", "01"]
    direct = tu.score_predict_fn(lambda X: 1 / (1 + np.exp(-2.0 * X["x_static"][:, 0].astype(np.float64))), val)
    assert sc[0]["score"] == pytest.approx(direct["score"], abs=1e-12) and sc[0]["score"] > sc[1]["score"]
    sel = tu.select_ak(sc, default_score=round(sc[0]["score"], 4))          # 반올림 같음 → 기본 유지
    assert not sel["adopt"] and sel["judgement"] == "기본 유지"
    sel = tu.select_ak(sc, default_score=sc[1]["score"])
    assert sel["adopt"] and sel["selected"]["trial"] == "t01"
    trials["02"] = Trial("02", 1.0, status="FAILED")
    with pytest.raises(tu.TuningError):
        tu.score_ak_trials(s, val)
