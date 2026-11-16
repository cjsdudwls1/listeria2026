"""단계 3-1 ~ 3-3(lst/stage3_combine·scripts/s3_1·s3_2·s3_3) — 합성 자료·가짜 모델.

핵심 점검(§7-3 엄격 OOF·블록 OOF):
  - 예측하는 행(시료)을 학습한 모델이 그 행 값을 내지 않음(VAE·④ 모두, 가짜 모델이 학습·예측한 시료를 기록).
  - fold validation 쌍·test 쌍 값 = 그 판 train 전체 모델(fold train 전체 = d ≤ train 끝 전부), 블록 모델은 그 판 train 안만 학습.
  - 같은 행 모델 재사용(19 = 전역 6 + 엄격 13), 이상 점수 칼럼이 runner 판의 train 행 = OOF·평가 쌍 = 전체 모델 값.
그리고 3-3 평균·백분위, 판정 (가)→(나), 스크립트 흐름(v78: 3-0 → ⑥⑦ 판정 → 3-1 → 3-2 → 3-3).
"""
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import blocks as bk
from lst import evaluate as ev
from lst import folds as fd
from lst import metrics as mt
from lst import runner as rn
from lst import stage3 as s3
from lst import stage3_combine as c3
from tests.test_chain import make_model
from tests.test_evaluate import fake_preds
from tests.test_runner import STUB, FakeAssembler, FakeModel, env  # noqa: F401
from tests.test_s3_scripts import CODE, _ctx, load_script, stage3_env  # noqa: F401

W = 20


@pytest.fixture(autouse=True)
def _fixed_code_md5(monkeypatch):
    """다른 담당이 같은 폴더 코드를 고치는 중이라 시험 도중 code_md5 가 바뀌어 멈추지 않게(시험에서만 고정)."""
    from lst import config as cf

    monkeypatch.setattr(cf, "code_md5", lambda *a, **k: "f" * 32)


# ---------------------------------------------------------------- 가짜 VAE 입력·모델(학습·예측한 시료를 기록)
class FakeVaeFeats:
    """x_tab 첫 칸 = 시료 번호(가짜 모델이 학습·예측 시료를 알게), 둘째 = h."""
    names = ["kid", "hh"]

    def __init__(self, st: ev.SampleTable):
        self.kid = pd.Series(np.arange(len(st.keys), dtype=np.float32), index=st.keys)

    def train_frame(self, rows):
        return pd.DataFrame({"kid": self.kid.reindex(rows["lims_data_key"].astype(str)).to_numpy(),
                             "hh": rows["h"].to_numpy(dtype=np.float32)})

    @contextlib.contextmanager
    def eval_frames(self, split):
        yield self.train_frame

    def new_scaler(self):
        return PassScaler()


class PassScaler:
    def fit(self, X):
        self.cols = list(X.columns)
        return self

    def transform(self, X, src_route=None):
        assert list(X.columns) == self.cols and src_route is not None and len(src_route) == len(X)
        return X

    def state(self):
        return {"cols": self.cols}


class FakeVae:
    def __init__(self, registry):
        self.id = len(registry)
        registry.append(self)
        self.scored = set()

    def fit(self, inputs, y, *, seed, n_jobs, groups):
        assert seed == 42 and set(inputs) == {"x_tab"} and inputs["x_tab"].dtype == np.float32
        self.train_kids = set(inputs["x_tab"][:, 0].astype(int).tolist())
        return self

    def save(self, path):
        Path(path).write_text("fake", encoding="utf-8")

    def score(self, inputs):
        k = inputs["x_tab"][:, 0].astype(int)
        self.scored |= set(k.tolist())
        return np.full(len(k), float(self.id)) + inputs["x_tab"][:, 1] / 1000.0


def _rows_fns(st):
    known = st.known_frame()
    return (lambda m: c3.model_train_rows(m, st, W, smoke=None, seed=42, known=known),
            lambda s: c3.split_train_rows(st, s, W, smoke=None, seed=42, known=known))


def _vae(st, vdir):
    reg = []
    rm, rs = _rows_fns(st)
    plan = bk.block_plan(reuse=True)
    vdir.mkdir(parents=True, exist_ok=True)
    res = c3.build_vae(feats=FakeVaeFeats(st), samples=st, plan=plan, W=W, rows_of_model=rm, rows_of_split=rs,
                       model_factory=lambda: FakeVae(reg), seed=42, n_jobs=8, vdir=vdir, log=lambda s: None)
    return plan, reg, res


def test_vae_strict_oof_no_leak(env, tmp_path):
    st = env["st"]
    plan, reg, res = _vae(st, tmp_path / "v")
    assert len(reg) == 19 == bk.plan_counts(plan)["total"]                      # 부록 A-9: 전역 6 + 엄격 13
    kid = pd.Series(np.arange(len(st.keys)), index=st.keys)
    by_name = {m.name: reg[i] for i, m in enumerate(plan)}
    for s in bk.SPLIT_ORDER:
        sp = fd.SPLITS[s]
        rows = c3.split_train_rows(st, s, W, smoke=None, seed=42)
        o = res["oof"][s]
        assert len(o) == len(rows) and not c3._kh(o).duplicated().any()
        ids = np.floor(o["value"].to_numpy()).astype(int)
        for mid in np.unique(ids):                                              # 값을 낸 모델이 그 시료를 학습 안 함
            k = kid.reindex(o.loc[ids == mid, "lims_data_key"]).to_numpy()
            assert not (set(k.tolist()) & reg[mid].train_kids)
            assert max(reg[mid].train_kids) is not None
        for name in o["model"].unique():                                        # 블록 모델 = 그 판 train 안만(엄격)
            m = by_name[name]
            dmax = st.d_days[sorted(m.train_kids)].max()
            assert dmax <= sp.train_end
        p = res["pred"][s]
        full = by_name[bk.full_model(plan, s).name]
        assert set(np.floor(p["pred"].to_numpy()).astype(int).tolist()) == {full.id}   # 평가 쌍 = 판 전체 모델
        allk = set(kid.reindex(rows["lims_data_key"]).tolist())
        assert full.train_kids == allk                                           # = 그 판 train 전체
        pk = set(kid.reindex(p["lims_data_key"]).tolist())
        assert not (pk & full.train_kids) and (st.d_days[sorted(pk)] > sp.train_end).all()
        assert len(p) == st.expected_pairs(s)
    # 같은 행 모델 재사용: fold2 블록 4 = fold1 넷째 블록 모델, fold3 블록 5·전역 블록 5 = fold2 전체 모델
    o2 = res["oof"]["fold2"]
    blk = bk.block_of(st.d_days[st.positions(o2["lims_data_key"])], "fold2")
    assert set(o2.loc[blk == 4, "model"]) == {"fold1-b4"}
    o3 = res["oof"]["fold3"]
    assert "fold2-full" in set(o3["model"]) and "fold2-full" in set(res["oof"]["refit"]["model"])


def test_vae_leak_guard_stops(env, tmp_path, monkeypatch):
    """OOF 대상 행 고르기가 틀리면(자기 블록 아닌 행) 학습 시료를 예측하게 돼 멈춤."""
    monkeypatch.setattr(bk, "oof_target_mask", lambda d, split, block: np.ones(len(d), dtype=bool))
    with pytest.raises(AssertionError, match="OOF 누수"):
        _vae(env["st"], tmp_path / "v")


def _write_vae_board(st, root: Path, rel: str, res=None):
    vdir = root / rel / c3.VAE_T
    if res is None:
        plan, reg, res = _vae(st, vdir)
    c3.write_oof(vdir, res["oof"])
    for s, df in res["pred"].items():
        df.to_parquet(rn.pred_path(vdir, s), index=False)
    (vdir / "spec.json").write_text(json.dumps({"W": W, "mode": "sum", "extra": {"version": {"include_ids": []}}}),
                                    encoding="utf-8")
    (vdir / "done.json").write_text("{}", encoding="utf-8")
    return vdir, res


class RealScalerFeats(FakeVaeFeats):
    def new_scaler(self):
        from lst import inputs as ip

        return ip.TabularScaler([], None, 2, extra_numeric=self.names)


def test_frozen_value_models_roundtrip(env, tmp_path):
    """3-1 판(이상 점수 칼럼) 폴더 inputs_<split>/value_models → lst/stage4 FrozenVAEProvider 가 다시 학습 없이 같은 값."""
    from types import SimpleNamespace

    from lst import models_dl as md
    from lst import stage4 as s4

    st = env["st"]
    root = tmp_path / "results"
    vdir = root / "s3" / "3-1" / c3.VAE_T
    vdir.mkdir(parents=True)
    rm, rs = _rows_fns(st)
    feats = RealScalerFeats(st)
    res = c3.build_vae(feats=feats, samples=st, plan=bk.block_plan(reuse=True), W=W, rows_of_model=rm,
                       rows_of_split=rs, model_factory=lambda: md.AnomalyModel("VAE", "tab", device="cpu", max_epochs=1),
                       seed=42, n_jobs=1, vdir=vdir, log=lambda s: None)
    _write_vae_board(st, root, "s3/3-1", res)
    sp = rn.VariantSpec(name="3-1-LR-VAE", stage="3-1", order="2차", kind="tree", algorithm="LR", W=W, mode="sum",
                        input_set="x+VAE", role="후보", train_count=4,
                        extra={"version": {"include_ids": ["WEATHER"]}, "anomaly": c3.anomaly_extra("s3/3-1")})
    ctx = rn.RunContext(samples=st, out_dir=root / "s3" / "3-1", code_root=CODE, input_md5s={"x": "0"},
                        exp_log=env["log"], make_inputs=c3.anomaly_make_inputs(FakeAssembler, st, root),
                        make_model=FakeModel, n_jobs=8, check_resources=STUB, require_7a_logged=False)
    assert rn.run_variant(sp, ctx)["status"] == "done"
    look = c3.AnomalyLookup(vdir, st)
    w = SimpleNamespace(inputs_root=root, device="cpu", ctx=SimpleNamespace(missing=None), check=lambda: None,
                        n_jobs=1, dl_kwargs={})
    pdef = {**c3.frozen_def(vdir), "kind": "vae_frozen", "from_dir": f"s3/3-1/{sp.name}"}
    prov = s4.FrozenVAEProvider(w, pdef, s4.RowPlan(), 42)
    prov._X = lambda meta, split: (feats.train_frame(meta), meta["src_route"].astype(str).to_numpy(dtype=object))
    for s in bk.SPLIT_ORDER:
        pj = json.loads((root / "s3" / "3-1" / sp.name / f"inputs_{s}" / "value_models" / "plan.json").read_text("utf-8"))
        assert pj["def"]["kind"] == "vae" and pj["def"]["column"] == c3.ANOM_COL
        rows = c3.split_train_rows(st, s, W, smoke=None, seed=42)
        assert np.allclose(prov.values(rows, s)[:, 0], look.values(rows, s), atol=1e-5)          # train 행 = 블록 OOF
        for h, idx, meta in c3.iter_eval_meta(st, s, W):
            if h in (1, 37):
                assert np.allclose(prov.values(meta, s)[:, 0], look.values(meta, s), atol=1e-5)  # 평가 쌍 = 전체 모델


def test_anomaly_lookup_and_runner(env, tmp_path):
    st = env["st"]
    root = tmp_path / "results"
    vdir, res = _write_vae_board(st, root, "s3/3-1")
    look = c3.AnomalyLookup(vdir, st)
    rows = c3.split_train_rows(st, "fold1", W, smoke=None, seed=42)
    v = look.values(rows, "fold1")
    o = res["oof"]["fold1"].set_index(["lims_data_key", "h"])["value"]
    assert np.allclose(v, o.reindex(c3._kh(rows)).to_numpy().astype(np.float32))
    bad = rows.copy()
    bad["h"] = (bad["h"] % 60) + 1                                                # 자기 h 가 아닌 train 행 → 값 없음
    with pytest.raises(c3.CombineError, match="없는"):
        look.values(bad, "fold1")
    # runner 판(이상 점수 칼럼): train 행 = OOF, 평가 쌍 = 판 전체 모델 값
    seen = []

    class Spy(FakeModel):
        def fit(self, X, y, **kw):
            seen.append(("fit", X[c3.ANOM_COL].to_numpy().copy()))
            assert list(X.columns)[-1] == c3.ANOM_COL
            return super().fit(X, y, **kw)

        def predict_proba(self, X, n_iter=None):
            seen.append(("pred", X[c3.ANOM_COL].to_numpy().copy()))
            return super().predict_proba(X, n_iter)

    sp = rn.VariantSpec(name="3-1-LR-VAE", stage="3-1", order="2차", kind="tree", algorithm="LR", W=W, mode="sum",
                        input_set="x+VAE", role="후보", train_count=4,
                        extra={"version": {"include_ids": ["WEATHER"]}, "anomaly": c3.anomaly_extra("s3/3-1")})
    ctx = rn.RunContext(samples=st, out_dir=root / "s3" / "3-1", code_root=CODE, input_md5s={"x": "0"},
                        exp_log=env["log"], make_inputs=c3.anomaly_make_inputs(FakeAssembler, st, root),
                        make_model=Spy, n_jobs=8, check_resources=STUB, require_7a_logged=False)
    r = rn.run_variant(sp, ctx)
    assert r["status"] == "done"
    fits = [a for k, a in seen if k == "fit"]
    assert len(fits) == 4
    for s, a in zip(bk.SPLIT_ORDER, fits):
        assert np.array_equal(np.sort(a), np.sort(res["oof"][s]["value"].to_numpy().astype(np.float32)))
    preds = np.concatenate([a for k, a in seen if k == "pred"])
    allp = np.concatenate([res["pred"][s]["pred"].to_numpy() for s in bk.SPLIT_ORDER]).astype(np.float32)
    assert np.array_equal(np.sort(preds), np.sort(allp))
    cols = json.loads((root / "s3" / "3-1" / sp.name / "columns_fold1.json").read_text(encoding="utf-8"))["columns"]
    assert cols[-1] == c3.ANOM_COL


# ---------------------------------------------------------------- ④
class FakeLims:
    def __init__(self, st):
        self.kid = pd.Series(np.arange(len(st.keys)), index=st.keys)
        self.models = []

    def assembler(self, alg, split, rows):
        return FakeAssembler(None, split).fit(rows)

    def model(self, alg, asm):
        lims = self

        class M(FakeModel):
            def fit(self, X, y, **kw):
                self.alg = alg
                lims.models.append(self)
                return super().fit(X, y, **kw)
        return M(None, asm)

    def meta_model(self):
        class Meta(FakeModel):
            def fit(self, X, y, **kw):
                assert list(X.columns) == list(c3.X4_COLS) and not X.isna().any().any()
                return super().fit(X, y, **kw)
        return Meta(None)


def test_x4_blocks_and_meta(env, tmp_path):
    st = env["st"]
    rm, rs = _rows_fns(st)
    plan = bk.block_plan(reuse=True)
    fl = FakeLims(st)
    vdir = tmp_path / "x4"
    vdir.mkdir()
    res = c3.build_x4(feats=fl, samples=st, plan=plan, W=W, rows_of_model=rm, rows_of_split=rs, seed=42, n_jobs=8,
                      vdir=vdir, log=lambda s: None)
    assert len(fl.models) == 6 * 19                                              # 부록 A-9: 6 × 19
    for s in bk.SPLIT_ORDER:
        assert len(res["pred"][s]) == st.expected_pairs(s)
        p = res["pred"][s]["pred"].to_numpy()
        assert ((p >= 0) & (p <= 1)).all()
        o = res["oof"][s]
        assert len(o) == len(rs(s)) and not o[list(c3.X4_COLS)].isna().any().any()
    full_saved = sorted(p.name for p in (vdir / c3.X4_MODEL_DIR).iterdir() if p.suffix == ".pkl")
    assert len(full_saved) == 6 * 4 and all("-full" in n or "fold2-full" in n for n in full_saved)
    sc_ = ev.score_cv({f: res["pred"][f] for f in ev.CV_FOLDS}, st)
    assert 0 <= sc_["cv_band_weighted"] <= 1


def test_x4_leak_guard(env, tmp_path, monkeypatch):
    monkeypatch.setattr(bk, "oof_target_mask", lambda d, split, block: np.ones(len(d), dtype=bool))
    rm, rs = _rows_fns(env["st"])
    with pytest.raises(AssertionError, match="OOF 누수"):
        c3.build_x4(feats=FakeLims(env["st"]), samples=env["st"], plan=bk.block_plan(), W=W, rows_of_model=rm,
                    rows_of_split=rs, seed=42, n_jobs=8, vdir=tmp_path, log=lambda s: None)


# ---------------------------------------------------------------- 3-3 평균·백분위
def test_ensemble_mean_and_percentile(env, tmp_path):
    st = env["st"]
    dirs = []
    for i in range(3):
        d = tmp_path / f"m{i}"
        d.mkdir()
        for s in bk.SPLIT_ORDER:
            fake_preds(st, s, seed=10 * i + bk.SPLIT_ORDER.index(s)).to_parquet(rn.pred_path(d, s), index=False)
        if i == 0:                                                                # 부스팅 구성만 공통 N 예측
            for f in ev.CV_FOLDS:
                fake_preds(st, f, seed=99).to_parquet(rn.pred_path(d, rn.THR_PREFIX + f), index=False)
        dirs.append(d)
    e = c3.ensemble_preds(dirs)
    for s in bk.SPLIT_ORDER:
        ms = [pd.read_parquet(rn.pred_path(d, s)) for d in dirs]
        assert np.allclose(e["pred"][s]["pred"], np.mean([m["pred"] for m in ms], axis=0))
        q = e["pct"][s]["pred"].to_numpy()
        assert (q > 0).all() and (q <= 1).all()
    t0 = pd.read_parquet(rn.pred_path(dirs[0], rn.THR_PREFIX + "fold1"))["pred"].to_numpy()
    t1 = pd.read_parquet(rn.pred_path(dirs[1], "fold1"))["pred"].to_numpy()
    t2 = pd.read_parquet(rn.pred_path(dirs[2], "fold1"))["pred"].to_numpy()
    assert np.allclose(e["thr"]["fold1"]["pred"], (t0 + t1 + t2) / 3)
    assert set(e["quantiles"]["split"]) == {"fold1", "fold2", "fold3", "test"}


# ---------------------------------------------------------------- 판정 (가)→(나)
def test_judge_step_order():
    C = ev.Candidate
    cur = C("cur", 0.10001, 4, False)
    r = c3.judge_step(stacking=[C("a", 0.10004, 4, False), C("b", 0.1002, 4, False)],
                      replacement=[C("x4", 0.1003, 118, False)], current=cur, rescore=None)
    assert r["stacking"]["top"] == "b" and r["stacking"]["judge"]["adopt"]
    assert r["replacement"]["against"] == "b" and r["replacement"]["judge"]["adopt"] and r["current_after"] == "x4"
    t = c3.judge_step(stacking=[C("a", 0.10004, 4, False)], replacement=[], current=cur, rescore=None)
    assert not t["stacking"]["judge"]["adopt"] and t["current_after"] == "cur"       # 4자리 같으면 안 오름
    tags = c3.judgement_tags(r)
    assert tags["b"][0] == "채택" and tags["a"][0] == "버림(단계 1위 아님)" and tags["x4"] == ("채택", "b")


# ---------------------------------------------------------------- 스크립트 흐름(3-0b·⑥⑦ → 3-1 → 3-2 → 3-3)
def _cfg_file(tmp: Path, name: str) -> Path:
    c = json.loads((CODE / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    c["use_mlflow"] = False
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    return p


def _upto_x67(e, tmp_path):
    """v78: 3-0 → ⑥⑦ 판정(3-0 순위 포함). 돌려줌 = x67_decision."""
    from tests.test_s3_scripts import run_extra_judge

    return run_extra_judge(e, tmp_path)


def _actx(e, out):
    return rn.RunContext(samples=e["st"], out_dir=out, code_root=CODE, input_md5s={"x": "0"}, exp_log=e["log"],
                         make_inputs=c3.anomaly_make_inputs(FakeAssembler, e["st"], e["inputs"]), make_model=make_model,
                         n_jobs=8, check_resources=STUB, require_7a_logged=False)


def test_scripts_flow(stage3_env, tmp_path):
    e = stage3_env
    inputs, st = e["inputs"], e["st"]
    x67 = _upto_x67(e, tmp_path)
    # ---- 3-1
    m1 = load_script("s3_1")
    out1 = inputs / "s3" / "3-1"
    cfg1 = m1.load_cfg(_cfg_file(tmp_path, "s3_1"))
    a = ["--root", str(tmp_path), "--inputs", str(inputs)]
    sp1 = m1.board_spec(cfg1, inputs)
    talg = x67["boards_3_0"][x67["tree_target"]]["structure"]
    assert sp1.name == f"3-1-{talg}-VAE" and sp1.extra["anomaly"]["column"] == c3.ANOM_COL
    assert m1.main(a + ["--out", str(out1), "--config", str(_cfg_file(tmp_path, "s3_1")), "--list"]) == [sp1.name]
    m1.write_plan(cfg1, inputs, out1)
    reg = []
    v = m1.run_vae(cfg1, tmp_path, inputs, out1, feats=FakeVaeFeats(st), model_factory=lambda: FakeVae(reg),
                   check=lambda: None)
    assert len(reg) == 19 and (out1 / c3.VAE_T / "oof_refit.parquet").exists() and v["cv_band_weighted"] >= 0
    rn.run_group([sp1], _actx(e, out1))
    j1 = m1.judge(cfg1, inputs, out1, rn.snapshot_code(CODE, out1), None)
    cv1 = rn.load_done(out1, sp1.name)[2]["cv_band_weighted"]
    cur0 = x67["current_best_after"]
    assert j1["state_before"]["current_best"]["name"] == cur0
    assert j1["adopted"] == mt.is_improvement(cv1, j1["state_before"]["current_best"]["cv"])
    assert j1["state_after"]["anomaly_adopted"] == j1["adopted"]
    assert j1["state_after"]["tree_target"]["name"] == (sp1.name if j1["adopted"] else x67["tree_target"])
    # ---- 3-2
    m2 = load_script("s3_2")
    out2 = inputs / "s3" / "3-2"
    cfg2 = m2.load_cfg(_cfg_file(tmp_path, "s3_2"))
    bs = m2.boards(cfg2, inputs)
    assert [b["algorithm"] for b in bs] == [x67["boards_3_0"][n]["structure"] for n in x67["tree_top3"]]
    reused = [b for b in bs if b["reuse"]]
    assert [b["algorithm"] for b in reused] == [talg]                              # 트리 대상 판의 3-2 판 = 3-1 판
    m2.write_plan(cfg2, inputs, out2)
    code2 = rn.snapshot_code(CODE, out2)
    m2.write_pointers(cfg2, inputs, out2, code2, None)
    for g in rn.group_specs(m2.train_specs(cfg2, inputs)):
        rn.run_group(g, _actx(e, out2))
    m2.run_x4(cfg2, tmp_path, inputs, out2, feats=FakeLims(st), check=lambda: None)
    j2 = m2.judge(cfg2, inputs, out2, code2, None)
    cvs = {b["name"]: b["cv"] for b in j2["boards_3_2"].values()}
    assert cvs[f"3-2-{talg}"] == cv1
    top = sorted(cvs, key=lambda n: (-mt.round4(cvs[n]), n))[0]
    cur1 = j1["state_after"]["current_best"]
    assert j2["judge"]["stacking"]["top"] == top
    assert j2["adopted_stacking"] == mt.is_improvement(cvs[top], cur1["cv"])
    after_a = cvs[top] if j2["adopted_stacking"] else cur1["cv"]
    assert j2["adopted_replacement"] == mt.is_improvement(j2["x4"]["cv"], after_a)
    # ---- 3-3
    m3 = load_script("s3_3")
    out3 = inputs / "s3" / "3-3"
    cfg3 = m3.load_cfg(_cfg_file(tmp_path, "s3_3"))
    ms = m3.members(cfg3, inputs)
    assert all(x["source"] is not None for x in ms) and m3.train_specs(cfg3, inputs) == []   # 학습 없음
    if j2["adopted_stacking"]:
        assert [x["source"] for x in ms] == [f"3-2-{x['algorithm']}" for x in ms]
    elif j1["adopted"]:
        assert all(x["spec"].extra.get("anomaly") for x in ms)
    else:
        assert [x["source"] for x in ms] == x67["tree_top3"]
    m3.write_plan(cfg3, inputs, out3)
    code3 = rn.snapshot_code(CODE, out3)
    m3.write_member_pointers(cfg3, inputs, out3, code3, None)
    m3.build_t3(cfg3, inputs, out3, code3, None)
    for s in bk.SPLIT_ORDER:
        mean = np.mean([pd.read_parquet(rn.pred_path(resolve_dir(inputs, x["pred_dir"]), s))["pred"].to_numpy()
                        for x in ms], axis=0)
        assert np.allclose(pd.read_parquet(rn.pred_path(out3 / c3.T3, s))["pred"], mean)
    j3 = m3.judge(cfg3, inputs, out3, code3, None)
    t3cv = rn.read_json(out3 / c3.T3 / "done.json")["cv_band_weighted"]
    cur2 = j2["state_after"]["current_best"]
    assert j3["adopted"] == mt.is_improvement(t3cv, cur2["cv"])
    assert (out3 / c3.T3 / "pct_cv_scores.json").exists() and (out3 / c3.T3 / "member_quantiles.csv").exists()
    chain = [json.loads(x) for x in (inputs / "logs" / "chain_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["step"] for c in chain][-4:] == ["3-1", "3-2", "3-2 ④", "3-3"]
    assert chain[-1]["after"] == j3["state_after"]["current_best"]["name"]
    assert m3.judge(cfg3, inputs, out3, code3, None) == json.loads((out3 / c3.DECISION["3-3"]).read_text("utf-8"))
    # ---- 단계 4 연결: chain_state_3-3.json → lst/stage4 읽기 함수
    from lst import stage4 as s4

    s4st = s4.load_state(inputs / "logs" / "chain_state_3-3.json")
    after = j3["state_after"]
    assert s4st["step"] == "3-3" and s4st["current_best"]["name"] == after["current_best"]["name"]
    assert s4st["tree_target"]["name"] == after["tree_target"]["name"] and s4st["tree_top3"] == x67["tree_top3"]
    assert sorted(s4st["tree_rank_3_0b"].values()) == [1, 2, 3, 4, 5, 6]
    assert s4st["tree_rank_3_0b"][talg] == 1                                    # 값 = 3-0 트리 CV 순위(v78)
    assert [h["step"] for h in s4st["history"]][-4:] == ["3-1", "3-2", "3-2 ④", "3-3"]
    # 판 종류마다(앙상블·④·이상 점수 트리 판) BestDef → stage4 derive_specs 가 읽음
    plan = s4.RowPlan(("reduce",), "4-a", "s4/sel-reduce")
    ens = c3.best_def(inputs, {"name": c3.T3, "dir": "s3/3-3/3-3-T3"})
    assert ens["kind"] == "ensemble" and len(ens["members"]) == 3
    d = s4.derive_specs(ens, inputs_root=inputs, name="4-a-3-3-T3", stage="4", role="후보", plan=plan, catenc_k=50)
    assert [s.algorithm for s in d.run] == [x["algorithm"] for x in ms] and d.ensemble.algorithm == "T3"
    b31 = c3.best_def(inputs, {"name": sp1.name, "dir": f"s3/3-1/{sp1.name}"})
    a31 = b31["members"][0]["anomaly"]
    assert b31["kind"] == "tree" and a31["column"] == c3.ANOM_COL and a31["W"] == sp1.W and a31["source"] == c3.VAE_T
    assert a31["version"] == rn.read_json(out1 / c3.VAE_T / "spec.json")["extra"]["version"]
    d = s4.derive_specs(b31, inputs_root=inputs, name=f"4-a-{talg}", stage="4", role="후보", plan=plan, catenc_k=50)
    assert s4.s4_of(d.run[0])["provider"]["kind"] == "vae" and s4.s4_of(d.run[0])["provider"]["column"] == c3.ANOM_COL
    x4b = c3.best_def(inputs, {"name": c3.X4, "dir": f"s3/3-2/{c3.X4}", "replacement": True})
    assert x4b["kind"] == "stack4" and x4b["replacement"] and x4b["members"][0]["stack4"]["version"]["include_ids"] == []
    d = s4.derive_specs(x4b, inputs_root=inputs, name="4-a-x4-L6", stage="4", role="후보", plan=plan, catenc_k=50)
    assert s4.s4_of(d.run[0])["provider"]["kind"] == "stack4"
    s4.check_best(ens), s4.check_best(b31), s4.check_best(x4b)


def resolve_dir(inputs, rel):
    from lst.paths import resolve

    return resolve(inputs, rel)


# ---------------------------------------------------------------- 실제 부품(합성 입력 묶음): VAE 입력·TabularScaler·AnomalyModel, ④ 트리 6종·메타
from tests.test_registry_inputs import ctx  # noqa: E402,F401  (module fixture)


def test_real_components(ctx):  # noqa: F811
    from lst import registry as rg
    from lst import models_dl as md

    s = ctx.samples
    rows = rn.add_bounds(s.loc[fd.day_ints(s["d"]) <= fd.day("2017-06-30")].copy(), W)
    rows["y"] = rows["y"].astype(np.int8)
    vs = rg.load_version({"name": "t", "order": 2, "W": W, "mode": "sum", "include_ids": ["WEATHER", "WX1", "FD1"]})
    f = c3.TreeTargetVaeFeatures(ctx, vs)
    assert "h" not in f.names and "g" not in f.names and "aw_label" not in f.names and "food_aw" in f.names
    X = f.train_frame(rows)
    assert list(X.columns) == f.names and len(X) == len(rows)
    sc_ = f.new_scaler().fit(X)
    Z = c3._x_tab(sc_, X, rows["src_route"])
    assert Z.dtype == np.float32 and np.isfinite(Z).all() and Z.shape[1] > len(f.names)   # 결측 표시·구조적 결측 칼럼
    m = md.AnomalyModel("VAE", "tab", device="cpu", max_epochs=2)
    m.fit({"x_tab": Z}, rows["y"].to_numpy(dtype=np.int64), seed=42, n_jobs=1,
          groups=(rows["sgg"].astype(str).to_numpy(), fd.day_ints(rows["d"])))
    z = m.score({"x_tab": Z})
    assert z.shape == (len(rows),) and np.isfinite(z).all()
    with f.eval_frames("refit") as frame:
        assert list(frame(rows.iloc[:5]).columns) == f.names
    # ④ 실제 트리 6종(기본 설정)·메타
    lf = c3.LimsTreeFeatures(ctx, W)
    cols = {}
    for alg in c3.TREE6:
        asm = lf.assembler(alg, "refit", rows)
        mm = lf.model(alg, asm).fit(asm.transform(rows), rows["y"].to_numpy(dtype=np.int64), seed=42, n_jobs=2)
        assert mm.max_iter is None and mm.params == {}                         # 라이브러리 기본 설정
        cols[f"p_{alg}"] = mm.predict_proba(asm.transform(rows))
    meta = lf.meta_model().fit(pd.DataFrame(cols), rows["y"].to_numpy(dtype=np.int64), seed=42, n_jobs=1)
    p = meta.predict_proba(pd.DataFrame(cols))
    assert ((p >= 0) & (p <= 1)).all() and meta.config_dump()["standardize"]["mean"]
