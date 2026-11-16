"""2차 사슬 반복 수 고르기(§3·§1-2) — runner 의 교차 N·공통 N·threshold 용 공통 N 예측·refit 공통 N·CatBoost 학습률 평균·
최대 반복 판정(fold 3 먼저, 2배, 상한 16,000, 실험 로그로 다음 판에 이어 씀), RF·ExtraTrees 예측 결정성,
3-0b 채택 규칙·빼는 그룹, 3-0·3-0b·추가 판 판 설정. 합성 자료·가짜 모델."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import evaluate as ev
from lst import folds as fd
from lst import metrics as mt
from lst import models_tree as mtr
from lst import registry as rg
from lst import runner as rn
from lst import stage3 as s3
from tests.test_runner import STUB, FakeAssembler, FakeModel, env  # noqa: F401  (env = fixture)


# ---------------------------------------------------------------- 가짜 부스팅(행마다 결정적, n_iter 로 순위가 바뀜)
class FakeBoost:
    LR_BASE = 0.01

    def __init__(self, spec, asm=None):
        self.spec = spec
        self.max_iter = spec.params.get("max_iter")
        self.lib = dict(spec.params.get("params") or {})

    def fit(self, X, y, *, seed, n_jobs, eval_set=None):
        assert self.max_iter is not None and seed == 42
        self.n_train = len(X)
        self.phase = 0.37 * (self.n_train % 13) + {"XGB": 0.0, "LGB": 1.1, "CAT": 2.3}[self.spec.algorithm]
        return self

    def _s(self, X):
        return X.fillna(0).to_numpy(dtype=np.float64).sum(axis=1)

    def predict_proba_multi(self, X, n_iters):
        s = self._s(X)
        out = np.empty((len(X), len(n_iters)))
        for j, k in enumerate(n_iters):
            assert 1 <= k <= self.max_iter
            out[:, j] = 0.5 + 0.45 * np.sin(s * 1.7 + k / 400.0 + self.phase)
        return out

    def predict_proba(self, X, n_iter=None):
        return self.predict_proba_multi(X, [n_iter or self.max_iter])[:, 0]

    def read_auto_learning_rate(self, X, y, *, seed, n_jobs):
        assert self.spec.algorithm == "CAT" and "learning_rate" not in self.lib
        return self.LR_BASE + len(X) / 1e6

    def save(self, path):
        Path(path).write_bytes(pickle.dumps({"max_iter": self.max_iter, "lib": self.lib, "n_train": self.n_train,
                                             "phase": self.phase, "alg": self.spec.algorithm}))


def load_fake(path, spec):
    d = pickle.loads(Path(path).read_bytes())
    m = FakeBoost(spec)
    m.max_iter, m.lib, m.n_train, m.phase = d["max_iter"], d["lib"], d["n_train"], d["phase"]
    return m


def make_model(spec, asm=None):
    return FakeBoost(spec, asm) if spec.params.get(rn.ITER_SELECT) else FakeModel(spec, asm)


def chain_specs(algs=("XGB", "LGB", "CAT", "RF"), input_set="2차-합성", stage="3-0", prefix="3-0"):
    return [rn.VariantSpec(name=f"{prefix}-{a}", stage=stage, order="2차", kind="tree", algorithm=a, W=20, mode="sum",
                           input_set=input_set, role="후보", train_count=4, judgement="판정 없음",
                           params={rn.ITER_SELECT: True} if a in rn.BOOST_ALGS else {},
                           extra={"version": {"include_ids": ["WEATHER"]}}) for a in algs]


def ctx_of(env, out, log):
    return rn.RunContext(samples=env["st"], out_dir=out, code_root=Path(__file__).resolve().parents[1],
                         input_md5s={"x": "0"}, exp_log=log, make_inputs=FakeAssembler, make_model=make_model,
                         n_jobs=8, check_resources=STUB, require_7a_logged=False)


def force_until(monkeypatch, target):
    """fold 3 자기 곡선을 max_iter < target 이면 늘 오르는 곡선으로 바꿔 판정(실제 max_iter_step 그대로)."""
    real = rn.max_iter_step

    def f(curve, points, max_iter):
        if max_iter < target:
            curve = np.linspace(0.1, 0.9, len(points))
        return real(curve, points, max_iter)
    monkeypatch.setattr(rn, "max_iter_step", f)


def _argmax4(v, pts):
    r = np.array([mt.round4(float(x)) for x in v])
    return int(pts[int(np.flatnonzero(r == r.max())[0])])


def _pairs_pred(env, model, spec, split, n_iter):
    st = env["st"]
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS[split]), spec.W)
    asm = FakeAssembler(spec, split).fit(rows)

    class W:
        def predict_proba(self, X, n_iter_=None):
            return model.predict_proba(X, n_iter=n_iter)
    return rn.predict_pairs(W(), asm, st, split, spec.W)


def test_chain_group_cross_common_threshold_refit_lr(env, tmp_path, monkeypatch):
    force_until(monkeypatch, 4000)                 # 2,000 → 4,000 에서 정해짐(fold 3 먼저)
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    out = tmp_path / "s3"
    sp = chain_specs()
    res = rn.run_group(sp, ctx_of(env, out, log))
    assert [r["status"] for r in res] == ["done"] * 4
    recs = {e["algorithm"]: e for e in log.entries() if e["event"] == rn.MAX_ITER_EVENT}
    assert set(recs) == {"XGB", "LGB", "CAT"}
    for a, r in recs.items():
        assert r["max_iter"] == 4000 and not r["at_cap"] and [h["max_iter"] for h in r["history"]] == [2000, 4000]
    st = env["st"]
    for s in sp[:3]:
        vdir = out / s.name
        nit = json.loads((vdir / rn.N_ITER_FILE).read_text(encoding="utf-8"))
        assert nit["max_iter"] == 4000 and nit["points"] == mtr.checkpoints(4000) and len(nit["points"]) == 96
        cur = pd.read_csv(vdir / rn.N_ITER_CURVES_FILE, encoding="utf-8-sig")
        pts = cur["n_iter"].tolist()
        sel = nit["selection"]
        for f in ev.CV_FOLDS:                      # 교차 N = 나머지 두 fold 평균 곡선 최고(같으면 작은 N)
            others = [g for g in ev.CV_FOLDS if g != f]
            assert sel["cross_n"][f] == _argmax4(cur[others].mean(axis=1).to_numpy(), pts)
        assert sel["common_n"] == _argmax4(cur[list(ev.CV_FOLDS)].mean(axis=1).to_numpy(), pts)
        cv = json.loads((vdir / "cv_scores.json").read_text(encoding="utf-8"))
        assert cv["threshold_source"] == "thr_preds"
        for f in ev.CV_FOLDS:                      # fold 점수 = 곡선의 교차 N 값, 저장 예측 = 그 트리 수 예측
            k = pts.index(sel["cross_n"][f])
            assert cv["fold_band_weighted"][f] == pytest.approx(cur.loc[k, f], abs=1e-15)
            m = load_fake(vdir / f"model_{f}", s)
            assert m.max_iter == 4000
            got = pd.read_parquet(vdir / f"pred_{f}.parquet")
            exp = _pairs_pred(env, m, s, f, sel["cross_n"][f])
            assert np.array_equal(got["pred"].to_numpy(), exp["pred"].to_numpy())
            thr = pd.read_parquet(vdir / f"pred_thr_{f}.parquet")
            exp = _pairs_pred(env, m, s, f, sel["common_n"])
            assert np.array_equal(thr["pred"].to_numpy(), exp["pred"].to_numpy())
        thr_all = {f: pd.read_parquet(vdir / f"pred_thr_{f}.parquet") for f in ev.CV_FOLDS}
        cvp = rn.load_preds(vdir, ev.CV_FOLDS)
        want = ev.score_cv(cvp, st, thr_preds=thr_all)["threshold"]["threshold"]
        assert cv["threshold"]["threshold"] == want   # threshold = 공통 N fold 예측으로
        m = load_fake(vdir / "model_refit", s)        # refit = 최대 반복까지 학습 → 공통 N 예측
        assert m.max_iter == 4000
        got = pd.read_parquet(vdir / "pred_refit.parquet")
        exp = _pairs_pred(env, m, s, "refit", sel["common_n"])
        assert np.array_equal(got["pred"].to_numpy(), exp["pred"].to_numpy())
        done = json.loads((vdir / "done.json").read_text(encoding="utf-8"))
        assert done["n_iter"]["common_n"] == sel["common_n"] and done["n_iter"]["max_iter_decided_here"]
    # CatBoost 학습률 = fold 별 자동 학습률 평균 → 모든 fold·refit
    nit = json.loads((out / "3-0-CAT" / rn.N_ITER_FILE).read_text(encoding="utf-8"))
    rows = {f: len(fd.make_train_rows(st.known_frame(), fd.SPLITS[f])) for f in ev.CV_FOLDS}
    want = np.mean([FakeBoost.LR_BASE + rows[f] / 1e6 for f in ev.CV_FOLDS])
    assert nit["learning_rate"]["mean"] == pytest.approx(want, rel=0, abs=1e-15)
    for f in rn.SPLIT_ORDER:
        assert load_fake(out / "3-0-CAT" / f"model_{f}", sp[2]).lib == {"learning_rate": nit["learning_rate"]["mean"]}
    assert "learning_rate" not in json.loads((out / "3-0-XGB" / rn.N_ITER_FILE).read_text(encoding="utf-8"))
    # 기본 설정 판(RF)은 반복 수 고르기 없음
    rf = out / "3-0-RF"
    assert not (rf / rn.N_ITER_FILE).exists() and not (rf / "pred_thr_fold1.parquet").exists()
    assert json.loads((rf / "cv_scores.json").read_text(encoding="utf-8"))["threshold_source"] == "cv_preds"

    # 같은 알고리즘의 다음 2차 사슬 판 = 실험 로그의 최대 반복 그대로(다시 정하지 않음)
    sp2 = chain_specs(("XGB",), input_set="다른-입력", stage="3-0b", prefix="3-0b")
    rn.run_group(sp2, ctx_of(env, out, log))
    nit = json.loads((out / "3-0b-XGB" / rn.N_ITER_FILE).read_text(encoding="utf-8"))
    assert nit["max_iter"] == 4000 and not nit["max_iter_source"]["decided_here"]
    assert len([e for e in log.entries() if e["event"] == rn.MAX_ITER_EVENT]) == 3


def test_chain_max_iter_cap(env, tmp_path, monkeypatch):
    force_until(monkeypatch, 10 ** 9)              # 늘 마지막 시점 → 16,000 에서 '상한'
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    out = tmp_path / "s3"
    rn.run_group(chain_specs(("LGB",)), ctx_of(env, out, log))
    r = log.max_iter_record("LGB")
    assert r["max_iter"] == 16000 and r["at_cap"]
    assert [h["max_iter"] for h in r["history"]] == [2000, 4000, 8000, 16000]
    assert any(e["event"] == "alert" and "상한" in e["message"] for e in log.entries())
    nit = json.loads((out / "3-0-LGB" / rn.N_ITER_FILE).read_text(encoding="utf-8"))
    assert len(nit["points"]) == 336 and nit["points"][-1] == 16000
    with pytest.raises(rn.ResumeError):            # 두 번 정하지 않음
        log.record_max_iter("LGB", max_iter=2000)


def test_max_iter_step_rules():
    pts = mtr.checkpoints(2000)
    up = np.linspace(0.1, 0.2, len(pts))
    st = rn.max_iter_step(up, pts, 2000)
    assert st["own_best_at_last"] and not st["decided"] and st["next"] == 4000
    st = rn.max_iter_step(np.linspace(0, 1, 336), mtr.checkpoints(16000), 16000)
    assert st["decided"] and st["at_cap"] and st["next"] is None
    peak = np.r_[np.linspace(0.1, 0.3, 30), np.linspace(0.3, 0.1, len(pts) - 30)]
    st = rn.max_iter_step(peak, pts, 2000)
    assert st["decided"] and not st["at_cap"] and st["own_best_n"] == pts[29]
    tie = np.full(len(pts), 0.12341)               # 4자리 반올림 같으면 작은 N
    tie[-1] = 0.123414
    assert rn.argmax_round4(tie, pts) == pts[0]
    tie[-1] = 0.12346
    assert rn.argmax_round4(tie, pts) == pts[-1]


def test_chain_spec_rules():
    s = chain_specs(("RF",))[0]
    bad = rn.VariantSpec(**{**s.to_dict(), "params": {rn.ITER_SELECT: True}})
    with pytest.raises(ValueError, match="부스팅"):
        rn.is_chain_boost(bad)
    x = chain_specs(("XGB",))[0]
    with pytest.raises(ValueError, match="max_iter"):
        rn.is_chain_boost(rn.VariantSpec(**{**x.to_dict(), "params": {rn.ITER_SELECT: True, "max_iter": 500}}))
    m = rn.model_spec(x, 4000, None)
    assert m.params == {rn.ITER_SELECT: True, "max_iter": 4000} and m.md5() != x.md5()
    c = rn.model_spec(chain_specs(("CAT",))[0], 2000, 0.05)
    assert c.params["params"] == {"learning_rate": 0.05}


# ---------------------------------------------------------------- RF·ExtraTrees 예측 결정성
@pytest.fixture(scope="module")
def forest_data():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f"x{i}": rng.normal(size=n).astype(np.float32) for i in range(8)})
    X.loc[rng.random(n) < 0.1, "x3"] = np.nan
    X["cat"] = rng.choice(["a", "b", "c"], n)
    y = (rng.random(n) < 0.1 + 0.2 * (X["x0"].fillna(0) > 0)).astype(int)
    return X, y


@pytest.mark.parametrize("algo", ["RF", "ET"])
def test_forest_predict_deterministic(algo, forest_data):
    from sklearn.base import clone

    X, y = forest_data
    m = mtr.make_model(algo, cat_cols=["cat"]).fit(X, y, seed=42, n_jobs=4)
    p1, p2 = m.predict_proba(X), m.predict_proba(X)
    assert np.array_equal(p1, p2)                                      # 두 번 예측 = 같은 값
    est1 = clone(m.estimator_).set_params(n_jobs=1)                     # sklearn n_jobs=1(나무 순서 더하기)과 바이트 같음
    est1.estimators_, est1.n_classes_, est1.classes_ = m.estimator_.estimators_, m.estimator_.n_classes_, m.estimator_.classes_
    est1.n_features_in_, est1.n_outputs_ = m.estimator_.n_features_in_, m.estimator_.n_outputs_
    ref = est1.predict_proba(m._encode(X))[:, 1]
    assert np.array_equal(p1, ref)
    idx = np.r_[5:40, 900:950]                                          # 행 일부 = 전체의 그 행(같은 행 한 번 예측 가능)
    assert np.array_equal(m.predict_proba(X.iloc[idx].reset_index(drop=True)), p1[idx])
    small = mtr.forest_proba_deterministic(m.estimator_, m._encode(X), 2, chunk=97)   # 묶음·n_jobs 와 무관
    assert np.array_equal(small, p1)
    m2 = mtr.make_model(algo, cat_cols=["cat"]).fit(X, y, seed=42, n_jobs=1)          # 학습은 그대로(n_jobs 무관)
    assert np.array_equal(m2.predict_proba(X), p1)
    assert algo in rn.ROW_DEDUPE_ALGS


# ---------------------------------------------------------------- 3-0b 채택·빼는 그룹
def _sc(cv, f1, f2, f3):
    return {"cv": cv, "fold": {"fold1": f1, "fold2": f2, "fold3": f3}}


def test_adopt_3_0b_rule():
    base = _sc(0.0500, 0.0400, 0.0500, 0.0600)
    assert s3.adopt_3_0b(_sc(0.0510, 0.0410, 0.0510, 0.0590), base)["adopt"]          # 평균↑ + 2 fold↑
    r = s3.adopt_3_0b(_sc(0.0510, 0.0450, 0.0490, 0.0590), base)                       # 평균↑ 이지만 1 fold↑
    assert not r["adopt"] and r["cv_up"] and r["n_fold_up"] == 1
    assert not s3.adopt_3_0b(_sc(0.05004, 0.0410, 0.0510, 0.0610), base)["adopt"]     # 4자리 같음 = 상승 아님
    assert not s3.adopt_3_0b(_sc(0.0499, 0.0410, 0.0510, 0.0577), base)["adopt"]      # 3 fold 중 2↑ 이지만 평균↓
    r = s3.adopt_3_0b(_sc(0.0510, 0.04004, 0.0510, 0.0620), base)                      # fold 1 은 4자리 같음
    assert r["fold_up"] == {"fold1": False, "fold2": True, "fold3": True} and r["adopt"]


def test_drop_groups_from_summary():
    s = pd.DataFrame({"target_kind": ["group"] * 4 + ["unit"], "target": ["계절", "기온", "L1", "강수", "t3"],
                      "mean_I": [0.0, 0.001, -0.002, 1e-12, -1.0]})
    assert s3.drop_groups_from_summary(s) == ["계절", "L1"]          # ≤ 0(반올림 없이), 그룹 번호 순


# ---------------------------------------------------------------- 판 설정
DEC = {"baseline": "1-CAT-W060-sum", "W": 60, "W_DL": 60, "tree_mode": "sum", "A": "CAT", "A_variant": "1-CAT-W060-sum"}
SEL = {"k": 50, "n_ma": 15, "n_ma_dl": 7, "W": 60, "W_DL": 60, "tree_mode": "sum"}


def test_specs_3_0_and_groups():
    t = s3.specs_3_0("T", DEC, SEL)
    d = s3.specs_3_0("D", DEC, SEL)
    assert [s.name for s in t + d] == [f"3-0-{a}" for a in s3.STRUCTS13]
    assert [[s.name for s in g] for g in rn.group_specs(t)] == [["3-0-XGB", "3-0-LGB", "3-0-CAT", "3-0-RF", "3-0-ET"],
                                                                 ["3-0-LR"]]
    assert [[s.name for s in g] for g in rn.group_specs(d)] == [[s.name for s in d]]     # 딥러닝 7 = 같은 입력 한 묶음
    assert [rn.is_chain_boost(s) for s in t] == [True, True, True, False, False, False]
    assert all(s.W == 60 and s.mode == "sum" and s.judgement == "판정 없음" for s in t)
    assert all(s.mode is None and s.kind == "dl" for s in d)
    assert t[0].extra["version"] == {"include_ids": list(rg.IDS), "k": 50, "n_ma": 15, "n_ma_dl": 7}
    assert rn.row_dedupe_ok(rn.group_specs(t)[0])


def test_spec_3_0b_and_end_names():
    base = s3.spec_3_0("CAT", DEC, SEL)
    c = s3.spec_3_0b(base, ["L1", "계절", "기온"])
    assert c.name == "3-0b-CAT" and c.stage == "3-0b" and c.params == base.params
    assert c.extra["version"]["exclude_groups"] == ["기온", "계절", "L1"]
    from lst import wiring as wr

    names = {u.group for u in wr.version_spec(c).units()}
    assert not names & {"기온", "계절", "L1"}
    assert s3.end_name("CAT") == "3-0b-CAT-end"
    with pytest.raises(ValueError):
        s3.spec_3_0b(base, [])


def test_specs_extra():
    g = s3.specs_extra("G", DEC, SEL)
    assert [(s.name, s.algorithm, s.kind, s.role) for s in g] == [("x6-LSTMA", "LSTMA", "dl", "교체 후보"),
                                                                   ("x7-D", "TRF", "dl", "교체 후보")]
    assert len(rn.group_specs(g)) == 1 and g[0].extra["version"]["dl_layout"] == "seq_only"
    base = s3.spec_3_0("XGB", DEC, SEL)
    c = s3.specs_extra("C", DEC, SEL) + s3.specs_extra("E", DEC, SEL, base)
    assert [s.name for s in c] == ["x7-T", "x8-XGB"]
    with pytest.raises(ValueError):
        s3.specs_extra("E", DEC, SEL)
    x7 = c[0]
    assert x7.mode == "raw" and rn.is_chain_boost(x7) and x7.extra["version"]["n_ma"] == SEL["n_ma_dl"]
    from lst import wiring as wr

    cols = [col.name for col in rg.tree_columns(wr.version_spec(x7))]
    assert cols[0] == "g" and "h" not in cols and "sgg" not in cols
    x8 = c[1]
    assert x8.role == "비교만" and x8.judgement == "해당 없음" and rn.is_chain_boost(x8)
    left = {u.group for u in wr.version_spec(x8).units()}
    assert not left & set(rg.WEATHER8_GROUPS) and "계절" in left
    assert s3.spec_x8(x8) is None                              # 기상 8그룹이 이미 다 빠짐 → 학습 없음
    assert s3.specs_extra("E", DEC, SEL, x8) == []


def test_importance_dedupe_predict_fn(forest_data):
    """3-0b 중요도의 같은 행 한 번 예측(predict_fn) = 모든 행 예측과 바이트 같음(부스팅 교차 N·RF)."""
    X, y = forest_data
    Xd = pd.concat([X, X.iloc[:500], X.iloc[100:300]], ignore_index=True)    # 같은 행 섞음
    for algo, n_iter in (("XGB", 7), ("RF", None)):
        kw = {"max_iter": 20} if algo == "XGB" else {}
        m = mtr.make_model(algo, cat_cols=["cat"], **kw).fit(X, y, seed=42, n_jobs=4)
        f = s3.dedupe_predict_fn(m, n_iter)
        ref = m.predict_proba(Xd, n_iter=n_iter) if n_iter else m.predict_proba(Xd)
        assert np.array_equal(f(Xd), ref)
