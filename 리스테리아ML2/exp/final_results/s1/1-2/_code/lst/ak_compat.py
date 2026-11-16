"""AutoKeras 3.0.0 + Keras 3.15.1(KERAS_BACKEND=torch) 호환 — 계획서 부록 A-7(단계 8 딥러닝 튜닝), 공통 규칙 '환경 확인(11-07)'.

지금 범위: 입력을 받는 래퍼와 Dense 패치만(탐색 그래프·튜너 실행은 단계 8 에서).

- `patch_keras_dense_units()`: AutoKeras 출력층 Dense 에 units 가 numpy.int64 로 넘어와 멈추는 문제 —
  `keras.layers.Dense.__init__` 에서 units 를 파이썬 int 로 바꿈(값 그대로, 구현 세부 — 브리프).
  AutoKeras 를 쓰기 전에 한 번 부름(여러 번 불러도 한 번만 감쌈).
- 확률: `AutoModel.predict` 는 분류 라벨(0/1)을 돌려주므로 쓰지 않고 `export_model().predict` 로
  (`export_predict_proba`). `hypermodel.build(hp)` 로 다시 학습한 Keras 모델은 `keras_predict_proba`.
- 입력 순서: 우리 딥러닝 입력 dict(x_seq·x_static·x_cat) → AutoKeras 다중 입력 list(그래프를 만든
  순서 그대로 `keys` 로 넘김).

구현 세부(계획서에 없음): Keras 백엔드 = 환경변수 KERAS_BACKEND=torch(keras import 전),
이미 다른 백엔드로 불렸으면 멈춤. 예측 배치 = 1024(부록 A-8 배치와 같게, 결과 영향 없음).
"""
from __future__ import annotations

import functools
import inspect
import numbers
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_KEYS = ("x_seq", "x_static", "x_cat")
PRED_BATCH = 1024
_PATCH_FLAG = "_lst_units_int_patch"


def ensure_torch_backend() -> str:
    """KERAS_BACKEND=torch 확인(keras import 전이면 설정). 반환 = 백엔드 이름."""
    if "keras" in sys.modules:
        import keras

        b = keras.backend.backend()
        if b != "torch":
            raise RuntimeError(f"keras 가 이미 '{b}' 백엔드로 불림 — KERAS_BACKEND=torch 로 다시 시작")
        return b
    cur = os.environ.get("KERAS_BACKEND")
    if cur not in (None, "torch"):
        raise RuntimeError(f"KERAS_BACKEND={cur} — torch 여야 함")
    os.environ["KERAS_BACKEND"] = "torch"
    return "torch"


def _as_int_units(units: Any) -> Any:
    if isinstance(units, numbers.Integral) and not isinstance(units, bool) and type(units) is not int:
        v = int(units)
        if v != units:  # pragma: no cover — Integral 이면 같음
            raise ValueError(f"units 변환 값이 다름: {units!r}")
        return v
    return units


def patch_keras_dense_units(dense_cls: type | None = None) -> bool:
    """Dense.__init__ 의 units(첫 위치 인자 또는 units=) 를 int 로. 새로 감쌌으면 True, 이미면 False.

    dense_cls: 시험용(기본 = keras.layers.Dense).
    """
    if dense_cls is None:
        ensure_torch_backend()
        import keras

        dense_cls = keras.layers.Dense
    orig = dense_cls.__init__
    if getattr(orig, _PATCH_FLAG, False):
        return False

    @functools.wraps(orig)
    def __init__(self, *args, **kwargs):
        if args:
            args = (_as_int_units(args[0]),) + tuple(args[1:])
        elif "units" in kwargs:
            kwargs["units"] = _as_int_units(kwargs["units"])
        return orig(self, *args, **kwargs)

    # Keras 의 자동 설정(Operation — inspect.getfullargspec)이 감싼 함수의 *args 를 보고 멈추지 않게 원래 서명을 붙임
    __init__.__signature__ = inspect.signature(orig)
    setattr(__init__, _PATCH_FLAG, True)
    dense_cls.__init__ = __init__
    return True


def prepare_autokeras() -> dict:
    """AutoKeras 쓰기 전 한 번: 백엔드 확인 + Dense 패치. 판 기록용 판 정보 반환."""
    backend = ensure_torch_backend()
    patched = patch_keras_dense_units()
    import autokeras
    import keras

    return {"keras": keras.__version__, "autokeras": autokeras.__version__, "backend": backend,
            "dense_units_patch": True, "patched_now": patched}


def ak_inputs(inputs: Mapping[str, np.ndarray], keys: Sequence[str] = DEFAULT_KEYS) -> list[np.ndarray]:
    """우리 입력 dict → AutoKeras 다중 입력 list(keys 순서). dtype·행 수·결측 점검."""
    unknown = set(inputs) - set(keys)
    if unknown:
        raise KeyError(f"keys 에 없는 입력: {sorted(unknown)} — 그래프 입력 순서(keys)에 넣거나 빼고 넘길 것")
    out, n = [], None
    for k in keys:
        if k not in inputs:
            raise KeyError(f"입력 키 없음: {k}")
        a = inputs[k]
        want = np.int64 if k == "x_cat" else np.float32
        if not isinstance(a, np.ndarray) or a.dtype != want:
            raise TypeError(f"{k}: {np.dtype(want)} numpy 배열이어야 함")
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError(f"행 수 다름: {k}")
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise ValueError(f"{k}: NaN·inf 있음")
        out.append(a)
    return out


def _proba_from(pred: Any, n: int) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    if p.shape == (n, 1):
        p = p[:, 0]
    if p.shape != (n,):
        raise ValueError(f"확률 모양 {p.shape} — 시그모이드 1칸((N,1)·(N,))만 받음")
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ValueError("확률이 0~1 밖이거나 결측")
    return p


def keras_predict_proba(keras_model: Any, inputs: Mapping[str, np.ndarray],
                        keys: Sequence[str] = DEFAULT_KEYS, batch_size: int = PRED_BATCH) -> np.ndarray:
    """Keras 모델(export_model 결과 또는 hypermodel.build(hp) 로 학습한 모델) → float64 (N,) 양성 확률."""
    xs = ak_inputs(inputs, keys)
    pred = keras_model.predict(xs if len(xs) > 1 else xs[0], batch_size=batch_size, verbose=0)
    return _proba_from(pred, xs[0].shape[0])


def export_predict_proba(auto_model: Any, inputs: Mapping[str, np.ndarray],
                         keys: Sequence[str] = DEFAULT_KEYS, batch_size: int = PRED_BATCH) -> np.ndarray:
    """AutoKeras AutoModel → export_model().predict 로 확률(AutoModel.predict 는 라벨이라 안 씀)."""
    return keras_predict_proba(auto_model.export_model(), inputs, keys, batch_size)
