"""AutoKeras 호환(부록 A-7, 브리프 '환경 확인(11-07)') 시험. keras 가 없으면 keras 시험만 건너뜀."""
import numpy as np
import pytest

from lst import ak_compat as ak


class FakeDense:
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation


def test_dense_patch_converts_numpy_int_and_is_idempotent():
    class D(FakeDense):
        pass

    assert ak.patch_keras_dense_units(D) is True
    assert ak.patch_keras_dense_units(D) is False          # 두 번 감싸지 않음
    a = D(np.int64(1))
    assert type(a.units) is int and a.units == 1
    b = D(units=np.int32(64), activation="relu")
    assert type(b.units) is int and b.units == 64 and b.activation == "relu"
    c = D(7)
    assert c.units == 7
    t = D(True)                                             # bool 은 그대로(값 바꾸지 않음)
    assert t.units is True


class FakeKerasModel:
    def __init__(self, out):
        self.out = out
        self.seen = None

    def predict(self, xs, batch_size, verbose):
        self.seen = (xs, batch_size)
        return self.out


class FakeAutoModel:
    def __init__(self, km):
        self.km = km

    def export_model(self):
        return self.km

    def predict(self, *a, **k):  # pragma: no cover — 라벨을 돌려주므로 쓰면 안 됨
        raise AssertionError("AutoModel.predict 는 쓰지 않음")


def _inputs(n=5):
    rng = np.random.default_rng(0)
    return {"x_seq": rng.normal(size=(n, 4, 3)).astype(np.float32),
            "x_static": rng.normal(size=(n, 2)).astype(np.float32),
            "x_cat": rng.integers(0, 3, size=(n, 4)).astype(np.int64)}


def test_export_predict_proba_uses_export_model():
    p = np.linspace(0.1, 0.9, 5, dtype=np.float32).reshape(5, 1)
    km = FakeKerasModel(p)
    out = ak.export_predict_proba(FakeAutoModel(km), _inputs())
    assert out.dtype == np.float64 and out.shape == (5,) and np.allclose(out, p[:, 0])
    xs, bs = km.seen
    assert [x.shape for x in xs] == [(5, 4, 3), (5, 2), (5, 4)] and bs == 1024


def test_proba_shape_and_range_checks():
    with pytest.raises(ValueError):
        ak.keras_predict_proba(FakeKerasModel(np.zeros((5, 2))), _inputs())
    with pytest.raises(ValueError):
        ak.keras_predict_proba(FakeKerasModel(np.full((5, 1), 1.5)), _inputs())


def test_ak_inputs_order_and_checks():
    inp = _inputs()
    xs = ak.ak_inputs(inp, ("x_cat", "x_seq", "x_static"))
    assert xs[0] is inp["x_cat"] and xs[1] is inp["x_seq"]
    with pytest.raises(KeyError):
        ak.ak_inputs(inp, ("x_seq", "x_static"))           # 안 쓰는 키가 남음
    bad = dict(inp)
    bad["x_seq"] = bad["x_seq"].copy()
    bad["x_seq"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ak.ak_inputs(bad)
    with pytest.raises(TypeError):
        ak.ak_inputs({**inp, "x_cat": inp["x_cat"].astype(np.int32)})


def test_real_keras_dense_patch(monkeypatch):
    """서버(keras 3.15.1, KERAS_BACKEND=torch)에서만 실제로 돎."""
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    keras = pytest.importorskip("keras")
    ak.patch_keras_dense_units()
    layer = keras.layers.Dense(np.int64(1), activation="sigmoid")
    assert type(layer.units) is int and layer.get_config()["units"] == 1


def test_real_autokeras_prepare(monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    pytest.importorskip("autokeras")
    info = ak.prepare_autokeras()
    assert info["backend"] == "torch" and info["dense_units_patch"] is True
