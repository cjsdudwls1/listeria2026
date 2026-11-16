"""딥러닝 모델 묶음 시험(합성 자료) — 계획서 §7, 부록 A-8, 단계 3-1, B-1·B-5.

로컬(GPU 없으면 CPU)·서버 공용. torch 가 없으면 건너뜀.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lst import models_dl as md  # noqa: E402

N, W, C, S = 400, 12, 5, 3
CAT_SIZES = (4, 9, 6, 7)   # OOV(0) 포함 번호표 크기
FAST = dict(max_epochs=1, batch_size=128)


def make_data(n=N, w=W, c=C, s=S, seed=0, cat=True):
    rng = np.random.default_rng(seed)
    x_seq = rng.normal(size=(n, w, c)).astype(np.float32)
    x_static = rng.normal(size=(n, s)).astype(np.float32)
    inputs = {"x_seq": x_seq, "x_static": x_static}
    if cat:
        inputs["x_cat"] = np.stack([rng.integers(1, m, size=n) for m in CAT_SIZES], axis=1).astype(np.int64)
    y = (x_seq[:, -3:, 0].mean(axis=1) + 0.5 * rng.normal(size=n) > 0.8).astype(np.int8)
    sgg = rng.integers(0, 20, size=n).astype(str)
    d = rng.integers(0, 15, size=n)
    return inputs, y, (sgg, d)


@pytest.fixture(scope="module")
def data():
    return make_data()


# ---------------------------------------------------------------- 고정 규칙
def test_embedding_dim_rule():
    assert md.embedding_dim(4) == 2 and md.embedding_dim(9) == 5 and md.embedding_dim(257) == 50
    assert md.embedding_dim(1) == 1 and md.embedding_dim(99) == 50 and md.embedding_dim(98) == 49


def test_tcn_receptive_field_125():
    assert md.tcn_receptive_field() == 125


def test_es_groups_whole_and_ten_percent():
    rng = np.random.default_rng(1)
    sgg = rng.integers(0, 30, size=5000).astype(str)
    d = rng.integers(0, 40, size=5000)
    es = md.split_es_groups((sgg, d), seed=42)
    key = np.char.add(np.char.add(sgg, "|"), d.astype(str))
    n_groups = len(np.unique(key))
    for g in np.unique(key):  # 묶음은 통째로 한쪽
        v = es[key == g]
        assert v.all() or (~v).all()
    assert len(np.unique(key[es])) == round(0.1 * n_groups)
    # 행 순서를 바꿔도 같은 묶음이 뽑힘(정렬 뒤 고름)
    perm = rng.permutation(5000)
    es2 = md.split_es_groups((sgg[perm], d[perm]), seed=42)
    assert set(key[perm][es2]) == set(key[es])
    assert not np.array_equal(es, md.split_es_groups((sgg, d), seed=43))


def test_robust_z_rules():
    p = md.robust_z_params(np.array([1.0, 2, 3, 4, 5]))
    assert p["median"] == 3 and p["scale"] == 2 and p["scale_rule"] == "iqr"
    assert np.allclose(md.robust_z(np.array([3.0, 5.0]), p), [0, 1])
    p = md.robust_z_params(np.array([0.0, 0, 0, 0, 0, 0, 0, 10]))  # IQR 0 → 표준편차
    assert p["scale_rule"] == "std" and np.isclose(p["scale"], np.std([0] * 7 + [10]))
    p = md.robust_z_params(np.array([2.0, 2, 2]))
    assert p["scale_rule"] == "one" and p["scale"] == 1.0


# ---------------------------------------------------------------- 구조별
@pytest.mark.parametrize("arch", md.CLASSIFIER_ARCHS)
def test_arch_fit_predict_save_load(arch, data, tmp_path):
    inputs, y, groups = data
    m = md.DLClassifier(arch, CAT_SIZES, device="cpu", **FAST).fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    p = m.predict_proba(inputs)
    assert p.dtype == np.float64 and p.shape == (N,) and np.isfinite(p).all()
    assert ((p > 0) & (p < 1)).all()
    assert m.fit_info["epochs_run"] == 1 and m.fit_info["n_train"] + m.fit_info["n_es"] == N
    f = m.save(tmp_path / f"{arch}.pt")
    with pytest.raises(FileExistsError):
        m.save(f)
    m2 = md.DLClassifier.load(f, device="cpu")
    assert np.array_equal(m2.predict_proba(inputs), p)
    assert m2.get_config()["arch"] == arch and m2.fit_info == m.fit_info


@pytest.mark.parametrize("arch", ["LSTMA", "GRUA", "TCN", "TRF"])
def test_no_cat_variant(arch):
    """⑥·⑦-D: x_cat 없음(x_static = g ÷ 50·해외 구조적 결측 2칼럼)."""
    inputs, y, groups = make_data(s=2, cat=False)
    m = md.DLClassifier(arch, None, device="cpu", **FAST).fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    assert m.predict_proba(inputs).shape == (N,)
    with pytest.raises(KeyError):  # 받지 않는 키
        m.predict_proba({**inputs, "x_cat": np.zeros((N, 4), np.int64)})


def test_combined_needs_structured_input():
    inputs, y, groups = make_data(s=0, cat=False)
    with pytest.raises(ValueError):
        md.DLClassifier("CMB", None, device="cpu", **FAST).fit(inputs, y, seed=42, n_jobs=2, groups=groups)


def test_cmbv_head_uses_mu_and_detached_error(data):
    inputs, y, groups = data
    m = md.DLClassifier("CMBV", CAT_SIZES, device="cpu", **FAST).fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    assert m.net.head[0].in_features == 64 + 64 + md.LATENT + 1
    # 추론 결정적(z = μ): 두 번 같은 값
    assert np.array_equal(m.predict_proba(inputs), m.predict_proba(inputs))


@pytest.mark.parametrize("w", [10, 13, 100])
def test_seq_vae_length_crop(w):
    core = md.SeqVAECore(7, w, variational=True)
    x = torch.randn(3, w, 7)
    mu, lv = core.encode(x)
    assert mu.shape == (3, md.LATENT) and lv.shape == (3, md.LATENT)
    assert core.decode(mu).shape == (3, w, 7)


# ---------------------------------------------------------------- 씨앗·재현
def test_seed_reproducible_cpu(data):
    inputs, y, groups = data
    kw = dict(device="cpu", max_epochs=2, batch_size=64)
    a = md.DLClassifier("LSTM", CAT_SIZES, **kw).fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    b = md.DLClassifier("LSTM", CAT_SIZES, **kw).fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    c = md.DLClassifier("LSTM", CAT_SIZES, **kw).fit(inputs, y, seed=7, n_jobs=2, groups=groups)
    assert np.array_equal(a.predict_proba(inputs), b.predict_proba(inputs))
    assert a.fit_info["es_rows_md5"] == b.fit_info["es_rows_md5"]
    assert not np.array_equal(a.predict_proba(inputs), c.predict_proba(inputs))
    assert a.fit_info["determinism"]["deterministic_algorithms"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 없음")
@pytest.mark.parametrize("arch", md.CLASSIFIER_ARCHS)
def test_seed_reproducible_gpu(arch, data, tmp_path):
    inputs, y, groups = data
    kw = dict(device="cuda", max_epochs=2, batch_size=64)
    a = md.DLClassifier(arch, CAT_SIZES, **kw).fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    b = md.DLClassifier(arch, CAT_SIZES, **kw).fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    pa = a.predict_proba(inputs)
    assert np.array_equal(pa, b.predict_proba(inputs))
    f = a.save(tmp_path / "g.pt")
    assert np.array_equal(md.DLClassifier.load(f, device="cuda").predict_proba(inputs), pa)


def test_early_stopping_restores_best(data):
    inputs, y, groups = data
    m = md.DLClassifier("GRU", CAT_SIZES, device="cpu", max_epochs=6, batch_size=64).fit(
        inputs, y, seed=42, n_jobs=2, groups=groups)
    es = m.fit_info["es_loss"]
    assert m.fit_info["best_epoch"] == int(np.argmin(es)) + 1
    # 되돌린 가중치로 다시 잰 조기 종료 손실 = 최저 손실
    es_idx = np.flatnonzero(md.split_es_groups(groups, 42))
    again = m._eval_rows(dict(inputs), y.astype(np.float32), es_idx, which=1).mean()
    assert np.isclose(again, min(es), rtol=1e-6)
    # 인내 3: 최저 뒤 3 에폭 넘게 돌지 않음
    assert m.fit_info["epochs_run"] <= m.fit_info["best_epoch"] + md.PATIENCE


# ---------------------------------------------------------------- 결측·OOV·입력 점검
def test_nan_and_dtype_rejected(data):
    inputs, y, groups = data
    bad = {**inputs, "x_seq": inputs["x_seq"].copy()}
    bad["x_seq"][3, 2, 1] = np.nan
    m = md.DLClassifier("LSTM", CAT_SIZES, device="cpu", **FAST)
    with pytest.raises(ValueError, match="NaN"):
        m.fit(bad, y, seed=42, n_jobs=2, groups=groups)
    with pytest.raises(TypeError):
        m.fit({**inputs, "x_static": inputs["x_static"].astype(np.float64)}, y, seed=42, n_jobs=2, groups=groups)
    with pytest.raises(TypeError):
        m.fit({**inputs, "x_cat": inputs["x_cat"].astype(np.int32)}, y, seed=42, n_jobs=2, groups=groups)
    with pytest.raises(ValueError):  # 번호표 밖
        xc = inputs["x_cat"].copy()
        xc[0, 1] = CAT_SIZES[1]
        m.fit({**inputs, "x_cat": xc}, y, seed=42, n_jobs=2, groups=groups)
    with pytest.raises(ValueError):  # eval_set 안 받음
        m.fit(inputs, y, seed=42, n_jobs=2, groups=groups, eval_set=(inputs, y))
    with pytest.raises(ValueError):  # n_jobs ≤ 8
        m.fit(inputs, y, seed=42, n_jobs=16, groups=groups)


def test_missing_zero_and_indicator_channels_are_plain_inputs(data):
    """결측 0 + 표시 채널은 조립 쪽이 만든 값 — 모델은 그대로 받아 씀(0 행도 정상 예측)."""
    inputs, y, groups = data
    x = inputs["x_seq"].copy()
    x[:50, :, 0] = 0.0          # 결측 → 0
    ind = np.zeros((N, W, 1), np.float32)
    ind[:50, :, 0] = 1.0        # 결측 표시 채널
    inp = {**inputs, "x_seq": np.concatenate([x, ind], axis=2)}
    m = md.DLClassifier("TCN", CAT_SIZES, device="cpu", **FAST).fit(inp, y, seed=42, n_jobs=2, groups=groups)
    p = m.predict_proba(inp)
    assert np.isfinite(p).all() and m.dims["C"] == C + 1
    with pytest.raises(ValueError):  # 채널 수가 다르면 멈춤
        m.predict_proba(inputs)


def test_oov_and_unseen_categories(data):
    inputs, y, groups = data
    xc = inputs["x_cat"].copy()
    xc[:, 1] = np.where(xc[:, 1] == 8, 7, xc[:, 1])   # sgg 번호 8 은 train 에 없음
    inp = {**inputs, "x_cat": xc}
    m = md.DLClassifier("LSTM", CAT_SIZES, device="cpu", **FAST).fit(inp, y, seed=42, n_jobs=2, groups=groups)
    emb = m.net.tail.cat.embs[1]
    assert torch.count_nonzero(emb.weight[0]) == 0           # OOV 행 0 고정(학습 뒤에도)
    assert not bool(m.net.tail.cat.seen1[8]) and not bool(m.net.tail.cat.seen1[0])
    unseen = {**inp, "x_cat": inp["x_cat"].copy()}
    unseen["x_cat"][:, 1] = 8
    oov = {**inp, "x_cat": inp["x_cat"].copy()}
    oov["x_cat"][:, 1] = 0
    assert np.array_equal(m.predict_proba(unseen), m.predict_proba(oov))


def test_es_only_category_is_oov():
    """조기 종료 10% 에만 있는 범주 = 가중치 학습에 안 쓰였으므로 OOV(구현 세부 3)."""
    inputs, y, groups = make_data()
    es = md.split_es_groups(groups, 42)
    xc = inputs["x_cat"].copy()
    xc[es, 2] = 5
    xc[~es & (xc[:, 2] == 5), 2] = 4
    m = md.DLClassifier("GRU", CAT_SIZES, device="cpu", **FAST).fit(
        {**inputs, "x_cat": xc}, y, seed=42, n_jobs=2, groups=groups)
    assert not bool(m.net.tail.cat.seen2[5]) and bool(m.net.tail.cat.seen2[4])


# ---------------------------------------------------------------- 메모리 오류 → 배치 나눔
def test_grad_accumulation_equals_full_batch(data):
    """나눠 계산한 기울기 = 한 배치 기울기(dropout 끈 상태에서 비교)."""
    inputs, y, groups = data
    m = md.DLClassifier("LSTMA", CAT_SIZES, device="cpu", **FAST).fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    b = np.arange(200)
    yf = y.astype(np.float32)
    grads = []
    for splits in (1, 4):
        m.net.eval()  # dropout 끔
        m.net.zero_grad()
        for chunk in np.array_split(b, splits):
            batch = md._to_dev(inputs, chunk, torch.device("cpu"))
            rows, _ = m._rows(batch, torch.from_numpy(yf[chunk]))
            (rows.sum() / len(b)).backward()
        grads.append([p.grad.clone() for p in m.net.parameters() if p.grad is not None])
    for g1, g4 in zip(*grads):
        assert torch.allclose(g1, g4, atol=1e-6, rtol=1e-4)


def test_oom_splits_same_batch(data, monkeypatch):
    inputs, y, groups = data
    m = md.DLClassifier("GRU", CAT_SIZES, device="cpu", max_epochs=1, batch_size=256)
    orig = md.DLClassifier._rows
    calls = {"n": 0}

    def flaky(self, batch, yb):
        if self.net.training and batch["x_seq"].shape[0] > 128:
            calls["n"] += 1
            raise torch.cuda.OutOfMemoryError("가짜 메모리 오류")
        return orig(self, batch, yb)

    monkeypatch.setattr(md.DLClassifier, "_rows", flaky)
    m.fit(inputs, y, seed=42, n_jobs=2, groups=groups)
    assert calls["n"] >= 1 and m.fit_info["final_splits"] == 2
    assert m.fit_info["oom_events"][0] == {"epoch": 1, "splits": 2}
    assert np.isfinite(m.predict_proba(inputs)).all()


def test_oom_with_batchnorm_not_split(data, monkeypatch):
    inputs, y, groups = data
    m = md.DLClassifier("GRU", CAT_SIZES, device="cpu", max_epochs=1, batch_size=256)
    orig_build = md.DLClassifier._build
    orig_rows = md.DLClassifier._rows

    def build_with_bn(self):
        net = orig_build(self)
        net.bn = torch.nn.BatchNorm1d(3)
        return net

    def oom(self, batch, yb):
        if self.net.training:
            raise torch.cuda.OutOfMemoryError("가짜")
        return orig_rows(self, batch, yb)

    monkeypatch.setattr(md.DLClassifier, "_build", build_with_bn)
    monkeypatch.setattr(md.DLClassifier, "_rows", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        m.fit(inputs, y, seed=42, n_jobs=2, groups=groups)


# ---------------------------------------------------------------- AE·VAE 이상 점수
@pytest.mark.parametrize("kind", md.ANOMALY_KINDS)
@pytest.mark.parametrize("mode", md.ANOMALY_MODES)
def test_anomaly_models(kind, mode, data, tmp_path):
    inputs, y, groups = data
    if mode == "tab":
        rng = np.random.default_rng(3)
        inp = {"x_tab": rng.normal(size=(N, 11)).astype(np.float32)}
    else:
        inp = {"x_seq": inputs["x_seq"]}
    m = md.AnomalyModel(kind, mode, device="cpu", max_epochs=2, batch_size=64).fit(
        inp, y, seed=42, n_jobs=2, groups=groups)
    n_neg = int((y == 0).sum())
    assert m.fit_info["n_train"] + m.fit_info["n_es"] == n_neg   # 불검출 행만
    raw = m.raw_score(inp)
    z = m.score(inp)
    assert raw.dtype == np.float64 and raw.shape == (N,) and np.isfinite(raw).all()
    if kind == "AE":
        assert (raw >= 0).all()
    assert np.allclose(z, (raw - m.z_params["median"]) / m.z_params["scale"])
    assert np.array_equal(raw, m.raw_score(inp))                  # 추론 결정적(VAE z = μ)
    f = m.save(tmp_path / f"{kind}_{mode}.pt")
    m2 = md.AnomalyModel.load(f, device="cpu")
    assert np.array_equal(m2.score(inp), z) and m2.z_params == m.z_params


def test_anomaly_baseline_is_es_negative_rows(data):
    inputs, y, groups = data
    inp = {"x_seq": inputs["x_seq"]}
    m = md.AnomalyModel("AE", "seq", device="cpu", **FAST).fit(inp, y, seed=42, n_jobs=2, groups=groups)
    neg = np.flatnonzero(y == 0)
    sgg, d = groups
    es_neg = md.split_es_groups((sgg[neg], d[neg]), 42)
    base = m.raw_score({"x_seq": inp["x_seq"][neg[es_neg]]})
    assert np.isclose(m.z_params["median"], np.median(base))
    assert m.z_params["n_baseline"] == int(es_neg.sum())
