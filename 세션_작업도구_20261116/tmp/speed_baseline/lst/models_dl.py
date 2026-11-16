"""딥러닝 모델 묶음(PyTorch) — 계획서 §7(7-1·7-2·7-3), 부록 A-8, 단계 3-1(AE·VAE), 부록 B-1·B-5.

구조(부록 A-8 표, 판 이름 약어는 부록 A-14):
  LSTM·GRU / TRF(트랜스포머 인코더) / TCN / LSTMA·GRUA(+ attention) / CMB(결합-기본) /
  CMBV(결합-VAE) — 분류기 `DLClassifier`
  AE·VAE × 표·시퀀스 — 이상 점수 `AnomalyModel`(원 점수 = AE 복원 MSE·VAE 음의 ELBO,
  이상 점수 = robust z, 기준 = 조기 종료용 묶음 10% 의 불검출 행)

입력(학습 틀 인터페이스 약속): dict
  x_seq    float32 (N, W, C)  — 기상 시퀀스(표준화·결측 0·결측 표시 채널은 입력 조립 쪽이 만듦)
  x_static float32 (N, S)     — 정적 칼럼(g ÷ 50 등)
  x_cat    int64   (N, 4)     — sido·sgg·p2·p3 번호(없는 구조 ⑥·⑦-D 는 키를 안 넣음)
  x_tab    float32 (N, d)     — 표 AE·VAE 전용(구현 세부: 약속에 없는 키 — 아래 목록)
모델은 받은 배열만 씀(NaN·inf 가 있으면 멈춤 — 결측 0 처리는 조립 쪽 책임).

학습(부록 A-8 학습 줄): 에폭마다 학습 행 전체 무작위 순서, Adam 1e-3, 배치 1024, 최대 20 에폭,
인내 3, 조기 종료 = train 의 (sgg, d) 묶음 단위 무작위 10%, 감시 = 검증 손실(분류 = 이진
교차엔트로피, AE·VAE = 자기 손실, CMBV = 이진 교차엔트로피), 끝나면 검증 손실 최저 에폭 가중치로
되돌림. 씨앗 42(B-5 — 호출하는 쪽이 seed 로 넘김). GPU 메모리 오류 = 같은 배치를 나눠 계산
(gradient accumulation, BatchNormalization 이 있는 구조는 나누지 않고 멈춤 — B-1).
GPU 메모리 상한 = `resources.limit_gpu_memory()`(0.4, 공통 규칙).

구현 세부(계획서에 없음) — 반환 보고에 같이 적음:
 1. 출력층은 로짓 1개 + `BCEWithLogits`(= Dense 1(sigmoid) + 이진 교차엔트로피와 수식 같음, 수치 안정).
    predict_proba = float64 sigmoid.
 2. 층 초기화·LayerNorm eps·TransformerEncoderLayer 기본(post-norm, FF 활성 relu)·nn.Embedding
    초기화(N(0,1)) 등은 PyTorch 기본값.
 3. 범주 OOV 번호 = `oov_index`(기본 0 — 입력 조립 쪽 번호표와 맞춰야 함). '그 모델 train 에 없는
    범주' = 가중치 학습에 쓴 행(조기 종료 10% 제외)에 없는 번호 → OOV(임베딩 0 고정 —
    padding_idx 로 0 벡터·기울기 0).
 4. 조기 종료 10% = 고유 (sgg, d) 묶음을 정렬한 뒤 `np.random.default_rng(seed)` 순열 앞
    round(0.1 × 묶음 수)개(최소 1) — 행 10% 가 아니라 묶음 10%.
 5. 씨앗 쓰임: 층 초기화·dropout·VAE 잡음 = `torch.manual_seed(seed)`, 에폭 순서 = 별도
    `torch.Generator().manual_seed(seed)`, 묶음 고르기 = numpy `default_rng(seed)`.
 6. 결정적 실행: `CUBLAS_WORKSPACE_CONFIG=:4096:8`(import 때 없으면 설정),
    `torch.use_deterministic_algorithms(True)`, cudnn deterministic·benchmark False. TF32 등
    정밀도 설정은 PyTorch 기본값 그대로(기록만).
 7. 메모리 오류 때 나눔 수는 2배씩(같은 배치, 손실 = 배치 평균과 같게 행 수로 가중), 한 번 늘면
    그 학습 끝까지 유지, 나눈 사실은 fit_info['oom_events'] 에 기록. 나누면 dropout 난수 소비가
    달라져 결과가 바이트 단위로는 다를 수 있음(수식은 같음). 예측(추론)은 묶음 크기만 절반씩.
 8. 추론 묶음 = 8192 행(학습 배치와 무관, 결과에 영향 없음 — GPU 커널 선택에 따른 1e-7 수준 차이 가능).
 9. TCN: 블록 = [인과 합성곱 → ReLU → dropout] × 2, 출력 = ReLU(블록 출력 + 잔차)(Bai et al. 2018 꼴),
    첫 블록 입력 채널 ≠ 64 이면 잔차에 1×1 합성곱.
10. 트랜스포머 위치 인코딩 = Vaswani 사인·코사인(위치 0..W−1, 투영 뒤 더함, dropout 없음). 결합-기본의
    트랜스포머 1블록은 LSTM 출력 위라 위치 인코딩 안 더함.
11. LSTMA·GRUA: MultiheadAttention(64, 헤드 2 → 헤드당 32 = key 32, 출력 투영 64), 어텐션 내부
    dropout 없음.
12. AE·VAE 활성화: 표 = 숨은 층 ReLU, 잠재·출력 선형. 시퀀스 = 각 합성곱 뒤 ReLU → Dropout 0.2,
    디코더 Linear 뒤 활성 없음, 전치 합성곱 사이 ReLU, 출력 선형.
13. VAE 손실 = 재구성 MSE(행마다 원소 평균) + KL(잠재 16차원 합), 배치 평균. 음의 ELBO 점수도 같은 꼴
    (z = μ). 결합-VAE 머리의 복원 오차 칼럼 = decode(μ) 의 행별 MSE(학습·추론 모두, 기울기 막음),
    VAE 갈래 손실은 재매개변수 표본으로.
14. AE·VAE 학습 행 = 받은 행 중 y == 0(fit 안에서 거름), 조기 종료 묶음도 그 행들의 묶음에서 고름.
    robust z 의 분위 = numpy 선형 보간, 표준편차 ddof 0.
15. predict 시 입력 W·C·S 가 학습 때와 다르면 멈춤.
16. (단계 8-D) 분류기 설정 hparams(없으면 위 부록 A-8 기본 그대로 — 층·키·계산 모두 같음): lr·hidden·layers·dropout·
    pos_weight·loss('bce'|'focal')·focal_gamma·focal_alpha(배치 크기는 생성자 batch_size). 쓰는 자리(`normalize_hparams`):
    hidden = 시퀀스 갈래 폭(LSTM·GRU 은닉, TRF d_model(FF = 2 × d_model), TCN 채널, LSTMA·GRUA 은닉·어텐션·FF 폭(잔차라
    같은 폭), CMB·CMBV LSTM 은닉·트랜스포머 d_model) — Dense 64 머리·정형 갈래·VAE 갈래는 그대로. layers = RNN 층 수
    (LSTM·GRU·LSTMA·GRUA·CMB·CMBV)·TRF 인코더 층 수, TCN 은 블록 5개 고정(받는 범위 125일 — 부록 A-8). dropout = LSTM·GRU
    (기본 0) 마지막 은닉 뒤·머리 Dense 64 뒤 + 2층이면 층 사이, TRF·CMB·CMBV 인코더 dropout(기본 0.1, CMB 2층 LSTM 층 사이도),
    TCN 블록 dropout(기본 0.1), LSTMA·GRUA 세 자리(기본 0.3·0.3·0.4)를 한 값으로. 손실(행마다): bce = BCE × (양성이면
    pos_weight), focal = α_t·(1 − p_t)^γ·BCE × (양성이면 pos_weight)(α_t = 양성 α·음성 1 − α). 조기 종료 감시는 늘 가중 없는
    BCE(부록 A-8 감시 그대로). CMBV 는 이 손실 + VAE 손실.
17. GPU 상한: `set_gpu_fraction(f)` 로 낮추면 학습·불러오기 때 그 값(없으면 공통 0.4) — 스크립트의 gpu_memory_fraction 이
    모델 만들 때 0.4 로 다시 올라가지 않게.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # CUDA 문맥 전에 있어야 결정적 cuBLAS

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .resources import GPU_MEMORY_FRACTION, check_n_jobs, limit_gpu_memory

# ---------------------------------------------------------------- 고정값(부록 A-8 학습 줄)
LR = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 20
PATIENCE = 3
ES_FRACTION = 0.10
PRED_BATCH = 8192           # 구현 세부 8
HIDDEN = 64
LATENT = 16
CAT_COLUMNS = ("sido", "sgg", "p2", "p3")
EMB_MAX = 50
FORMAT = "lst_dl_v1"

CLASSIFIER_ARCHS = ("LSTM", "GRU", "TRF", "TCN", "LSTMA", "GRUA", "CMB", "CMBV")
ANOMALY_KINDS = ("AE", "VAE")
ANOMALY_MODES = ("tab", "seq")
TCN_DILATIONS = (1, 2, 4, 8, 16)
TCN_KERNEL = 3

_FLOAT_KEYS = ("x_seq", "x_static", "x_tab")

# ---------------------------------------------------------------- 단계 8-D 분류기 설정(구현 세부 16)
HP_KEYS = ("lr", "hidden", "layers", "dropout", "pos_weight", "loss", "focal_gamma", "focal_alpha")
LOSSES = ("bce", "focal")
NO_LAYERS = ("TCN",)                                  # 블록 수 고정(받는 범위)
TRF_LIKE = ("TRF", "CMB", "CMBV")                     # 헤드 4 → hidden 4 의 배수
ATTN_LIKE = ("LSTMA", "GRUA")                         # 헤드 2 → hidden 2 의 배수
_DEF_LAYERS = {"TRF": 2, "TCN": None}
_DEF_DROPOUT = {"LSTM": 0.0, "GRU": 0.0, "TRF": 0.1, "TCN": 0.1, "LSTMA": None, "GRUA": None, "CMB": 0.1, "CMBV": 0.1}
ATTN_DROPOUTS = (0.3, 0.3, 0.4)                       # LSTMA·GRUA 기본(부록 A-8) — dropout None


def default_hparams(arch: str) -> dict:
    """부록 A-8 기본(구조마다)."""
    if arch not in CLASSIFIER_ARCHS:
        raise ValueError(f"모르는 구조 {arch!r}")
    return {"lr": LR, "hidden": HIDDEN, "layers": _DEF_LAYERS.get(arch, 1), "dropout": _DEF_DROPOUT[arch],
            "pos_weight": 1.0, "loss": "bce", "focal_gamma": None, "focal_alpha": None}


def normalize_hparams(arch: str, hp: Mapping[str, Any] | None) -> dict:
    """hp(일부 칸) → 기본을 채운 전체 설정(점검). 모르는 칸·구조에 없는 칸(TCN layers)·범위 밖이면 멈춤."""
    out = default_hparams(arch)
    hp = dict(hp or {})
    bad = sorted(set(hp) - set(HP_KEYS))
    if bad:
        raise ValueError(f"딥러닝 설정 모르는 칸: {bad}")
    if arch in NO_LAYERS and hp.get("layers") is not None:
        raise ValueError(f"{arch}: layers 는 고정(블록 5개 — 부록 A-8)")
    out.update(hp)
    lr, hid = float(out["lr"]), out["hidden"]
    if not (lr > 0 and math.isfinite(lr)):
        raise ValueError(f"lr: {lr}")
    if isinstance(hid, bool) or int(hid) != hid or int(hid) < 2:
        raise ValueError(f"hidden: {hid}")
    hid = int(hid)
    if arch in TRF_LIKE and hid % 4:
        raise ValueError(f"{arch}: hidden 은 4 의 배수(헤드 4)")
    if (arch in ATTN_LIKE or arch == "TRF") and hid % 2:
        raise ValueError(f"{arch}: hidden 은 짝수")
    lay = out["layers"]
    if lay is not None and (isinstance(lay, bool) or int(lay) != lay or not 1 <= int(lay) <= 4):
        raise ValueError(f"layers: {lay}")
    d = out["dropout"]
    if d is not None and not (0.0 <= float(d) < 1.0):
        raise ValueError(f"dropout: {d}")
    pw = float(out["pos_weight"])
    if not (pw > 0 and math.isfinite(pw)):
        raise ValueError(f"pos_weight: {pw}")
    if out["loss"] not in LOSSES:
        raise ValueError(f"loss: {out['loss']!r} — {LOSSES}")
    if out["loss"] == "focal":
        if out["focal_gamma"] is None or out["focal_alpha"] is None:
            raise ValueError("focal 에는 focal_gamma·focal_alpha")
        if float(out["focal_gamma"]) < 0 or not 0.0 < float(out["focal_alpha"]) < 1.0:
            raise ValueError("focal_gamma ≥ 0·0 < focal_alpha < 1")
        out["focal_gamma"], out["focal_alpha"] = float(out["focal_gamma"]), float(out["focal_alpha"])
    elif out["focal_gamma"] is not None or out["focal_alpha"] is not None:
        raise ValueError("bce 에는 focal_gamma·focal_alpha 없음")
    out.update(lr=lr, hidden=hid, layers=None if lay is None else int(lay),
               dropout=None if d is None else float(d), pos_weight=pw)
    return out


def _hp(hp: Mapping | None, arch: str) -> dict:
    return dict(hp) if hp is not None else default_hparams(arch)


def train_rows_loss(logit: torch.Tensor, y: torch.Tensor, bce: torch.Tensor, hp: Mapping | None) -> torch.Tensor:
    """학습 손실 행(구현 세부 16). hp None·기본(bce·pos_weight 1) = bce 그대로(같은 텐서)."""
    if hp is None:
        return bce
    w = float(hp["pos_weight"])
    if hp["loss"] == "bce":
        return bce if w == 1.0 else bce * (1.0 + (w - 1.0) * y)
    p = torch.sigmoid(logit)
    p_t = p * y + (1.0 - p) * (1.0 - y)
    a = float(hp["focal_alpha"])
    a_t = a * y + (1.0 - a) * (1.0 - y)
    rows = a_t * (1.0 - p_t).pow(float(hp["focal_gamma"])) * bce
    return rows if w == 1.0 else rows * (1.0 + (w - 1.0) * y)


_GPU_FRACTION: list = [None]


def set_gpu_fraction(fraction: float | None) -> None:
    """구현 세부 17 — 이 프로세스의 딥러닝 GPU 상한(None = 공통 0.4)."""
    if fraction is not None and not (0 < float(fraction) <= GPU_MEMORY_FRACTION):
        raise ValueError(f"GPU 메모리 비율은 0 초과 {GPU_MEMORY_FRACTION} 이하: {fraction}")
    _GPU_FRACTION[0] = None if fraction is None else float(fraction)


def gpu_fraction() -> float:
    return _GPU_FRACTION[0] if _GPU_FRACTION[0] is not None else GPU_MEMORY_FRACTION


# ---------------------------------------------------------------- 공통 설정
def embedding_dim(n_categories: int) -> int:
    """부록 A-8: min(50, (범주 수 + 1) // 2), 범주 수 = 번호표 크기(OOV 포함)."""
    n = int(n_categories)
    if n < 1:
        raise ValueError(f"범주 수 ≥ 1: {n}")
    return min(EMB_MAX, (n + 1) // 2)


def tcn_receptive_field(kernel: int = TCN_KERNEL, dilations: Sequence[int] = TCN_DILATIONS,
                        convs_per_block: int = 2) -> int:
    """받는 범위 = 1 + Σ 블록 합성곱 (k−1)·d. 부록 A-8: 125일."""
    return 1 + convs_per_block * (kernel - 1) * sum(dilations)


def set_determinism(seed: int) -> dict:
    """결정적 실행 설정 + 전역 씨앗(구현 세부 5·6). 적용 값 dict 반환(기록용)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed 는 정수: {seed!r}")
    seed = int(seed)
    torch.manual_seed(seed)  # CPU·CUDA 모두
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch": str(torch.__version__),
    }


def resolve_device(device: str | None) -> torch.device:
    """None = CUDA 있으면 cuda, 없으면 cpu(로컬 시험). cuda 면 메모리 상한(set_gpu_fraction 값, 없으면 0.4) 적용."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 없음")
        limit_gpu_memory(gpu_fraction())
    return dev


def split_es_groups(groups, seed: int, fraction: float = ES_FRACTION) -> np.ndarray:
    """조기 종료용 (sgg, d) 묶음 단위 무작위 10%(구현 세부 4). 반환 = 조기 종료 행 bool (N,).

    groups: 1차원 배열(묶음 이름) 또는 (sgg 배열, d 배열) 두 개짜리 tuple/list.
    """
    codes = _group_codes(groups)
    n_groups = int(codes.max()) + 1 if len(codes) else 0
    if n_groups < 2:
        raise ValueError(f"조기 종료 묶음을 나누려면 묶음 ≥ 2: {n_groups}")
    k = max(1, int(round(fraction * n_groups)))
    rng = np.random.default_rng(int(seed))
    chosen = np.zeros(n_groups, dtype=bool)
    chosen[rng.permutation(n_groups)[:k]] = True
    return chosen[codes]


def _group_codes(groups) -> np.ndarray:
    if isinstance(groups, (tuple, list)) and len(groups) == 2 and not np.isscalar(groups[0]):
        a, b = (np.asarray(g) for g in groups)
        if a.shape != b.shape or a.ndim != 1:
            raise ValueError("groups (sgg, d) 두 배열의 모양이 다름")
        mi = pd.MultiIndex.from_arrays([pd.Index(a), pd.Index(b)])
        codes, _ = pd.factorize(mi, sort=True)
    else:
        arr = np.asarray(groups)
        if arr.ndim != 1:
            raise ValueError("groups 는 1차원")
        codes, _ = pd.factorize(pd.Index(arr), sort=True)
    codes = np.asarray(codes)
    if (codes < 0).any():
        raise ValueError("groups 에 결측")
    return codes


def robust_z_params(baseline: np.ndarray) -> dict:
    """부록 A-8 이상 점수: 중앙값·IQR(0 이면 표준편차, 그것도 0 이면 1)."""
    b = np.asarray(baseline, dtype=np.float64)
    if b.ndim != 1 or len(b) == 0 or not np.isfinite(b).all():
        raise ValueError("기준 점수는 유한한 1차원 배열(1개 이상)")
    med = float(np.median(b))
    q75, q25 = np.percentile(b, [75, 25])
    scale = float(q75 - q25)
    rule = "iqr"
    if scale == 0.0:
        scale = float(np.std(b, ddof=0))
        rule = "std"
        if scale == 0.0:
            scale, rule = 1.0, "one"
    return {"median": med, "scale": scale, "scale_rule": rule, "n_baseline": int(len(b))}


def robust_z(raw: np.ndarray, params: Mapping) -> np.ndarray:
    return (np.asarray(raw, dtype=np.float64) - params["median"]) / params["scale"]


# ---------------------------------------------------------------- 입력 점검
def _check_inputs(inputs: Mapping[str, np.ndarray], required: Sequence[str],
                  optional: Sequence[str] = ()) -> int:
    if not isinstance(inputs, Mapping):
        raise TypeError("입력은 dict")
    unknown = set(inputs) - set(required) - set(optional)
    if unknown:
        raise KeyError(f"모르는 입력 키: {sorted(unknown)}")
    missing = [k for k in required if k not in inputs]
    if missing:
        raise KeyError(f"입력 키 없음: {missing}")
    n = None
    for k, a in inputs.items():
        if not isinstance(a, np.ndarray):
            raise TypeError(f"{k}: numpy 배열이어야 함")
        want = np.int64 if k == "x_cat" else np.float32
        if a.dtype != want:
            raise TypeError(f"{k}: dtype {a.dtype} — {np.dtype(want)} 이어야 함")
        nd = {"x_seq": 3, "x_static": 2, "x_cat": 2, "x_tab": 2}[k]
        if a.ndim != nd:
            raise ValueError(f"{k}: {nd}차원이어야 함(지금 {a.shape})")
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError(f"행 수 다름: {k} {a.shape[0]} vs {n}")
        if k in _FLOAT_KEYS:
            step = max(1, (1 << 24) // max(1, int(np.prod(a.shape[1:]))))
            for s in range(0, a.shape[0], step):
                if not np.isfinite(a[s:s + step]).all():
                    raise ValueError(f"{k}: NaN·inf 있음 — 결측 0·표시는 입력 조립 쪽에서(§4-2)")
    if not n:
        raise ValueError("행 0개")
    return int(n)


def _check_y(y, n: int) -> np.ndarray:
    y = np.asarray(y)
    if y.shape != (n,):
        raise ValueError(f"y 모양 {y.shape} ≠ ({n},)")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y 는 0/1")
    return y.astype(np.float32)


def _to_dev(inputs: Mapping[str, np.ndarray], idx: np.ndarray, dev: torch.device) -> dict:
    nb = dev.type == "cuda"
    return {k: torch.from_numpy(np.ascontiguousarray(a[idx])).to(dev, non_blocking=nb)
            for k, a in inputs.items()}


def _state_md5(idx: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(np.sort(idx).astype(np.int64)).tobytes()).hexdigest()


def _py(obj: Any) -> Any:
    """torch.save(weights_only) 로 불러올 수 있게 파이썬 기본형으로."""
    if isinstance(obj, str):
        return str(obj)  # TorchVersion 등 str 하위형 → str
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    return obj


# ---------------------------------------------------------------- 층 조각
class CatEmbedding(nn.Module):
    """sido·sgg·p2·p3 임베딩. OOV 번호 행 = 0 벡터 고정(padding_idx), train 에 없는 번호 → OOV."""

    def __init__(self, cat_sizes: Sequence[int], oov_index: int = 0):
        super().__init__()
        self.cat_sizes = [int(n) for n in cat_sizes]
        self.oov_index = int(oov_index)
        for n in self.cat_sizes:
            if not (0 <= self.oov_index < n):
                raise ValueError(f"oov_index {oov_index} 가 번호표 크기 {n} 밖")
        self.embs = nn.ModuleList(nn.Embedding(n, embedding_dim(n), padding_idx=self.oov_index)
                                  for n in self.cat_sizes)
        for j, n in enumerate(self.cat_sizes):
            self.register_buffer(f"seen{j}", torch.ones(n, dtype=torch.bool))
        self.out_dim = sum(embedding_dim(n) for n in self.cat_sizes)

    def set_seen(self, x_cat: np.ndarray) -> None:
        for j, n in enumerate(self.cat_sizes):
            m = np.zeros(n, dtype=bool)
            m[np.unique(x_cat[:, j])] = True
            getattr(self, f"seen{j}").copy_(torch.from_numpy(m))

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        outs = []
        for j, emb in enumerate(self.embs):
            idx = x_cat[:, j]
            seen = getattr(self, f"seen{j}")
            idx = torch.where(seen[idx], idx, torch.full_like(idx, self.oov_index))
            outs.append(emb(idx))
        return torch.cat(outs, dim=1)


class _Tail(nn.Module):
    """정적 칼럼 + 임베딩을 이어 붙이는 부분(없으면 폭 0)."""

    def __init__(self, static_dim: int, cat_sizes: Sequence[int] | None, oov_index: int):
        super().__init__()
        self.static_dim = int(static_dim)
        self.cat = CatEmbedding(cat_sizes, oov_index) if cat_sizes is not None else None
        self.out_dim = self.static_dim + (self.cat.out_dim if self.cat is not None else 0)

    def forward(self, x_static: torch.Tensor | None, x_cat: torch.Tensor | None) -> torch.Tensor | None:
        parts = []
        if self.static_dim:
            parts.append(x_static)
        if self.cat is not None:
            parts.append(self.cat(x_cat))
        return torch.cat(parts, dim=1) if parts else None


def _cat(a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    return a if b is None else torch.cat([a, b], dim=1)


def sinusoidal_encoding(length: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _encoder_layer(d_model: int = HIDDEN, dropout: float = 0.1) -> nn.TransformerEncoderLayer:
    """부록 A-8: 헤드 4·FF 128(= 2 × d_model)·dropout 0.1(기본)."""
    return nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=2 * d_model, dropout=dropout,
                                      batch_first=True)


def _rnn(cell: str, C: int, H: int, layers: int, inter_dropout: float) -> nn.Module:
    """RNN(층 1 = 부록 A-8 그대로). 층 사이 dropout 은 2층 이상일 때만."""
    return (nn.LSTM if cell == "LSTM" else nn.GRU)(C, H, num_layers=layers,
                                                   dropout=inter_dropout if layers > 1 else 0.0, batch_first=True)


def _drop(p: float) -> nn.Module:
    return nn.Dropout(p) if p > 0 else nn.Identity()


class _CausalConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):  # (B, C, L)
        return self.conv(F.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv(cin, cout, TCN_KERNEL, dilation), nn.ReLU(), nn.Dropout(dropout),
            _CausalConv(cout, cout, TCN_KERNEL, dilation), nn.ReLU(), nn.Dropout(dropout))
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.res(x))


class SeqVAECore(nn.Module):
    """시퀀스 AE·VAE 틀(부록 A-8): Conv1d(32·16, k7, s2, p3, dropout 0.2) → 평탄화 → Linear 16,
    디코더 Linear → (16 × L2) → ConvT(16→32→C, k7, s2, p3, op1) → W 길이로 자름."""

    def __init__(self, channels: int, seq_len: int, variational: bool, latent: int = LATENT):
        super().__init__()
        self.C, self.W, self.variational = int(channels), int(seq_len), bool(variational)
        l1 = (self.W + 2 * 3 - 7) // 2 + 1
        self.L2 = (l1 + 2 * 3 - 7) // 2 + 1
        if self.L2 < 1:
            raise ValueError(f"시퀀스가 너무 짧음: W={seq_len}")
        flat = 16 * self.L2
        self.enc = nn.Sequential(
            nn.Conv1d(self.C, 32, 7, stride=2, padding=3), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 16, 7, stride=2, padding=3), nn.ReLU(), nn.Dropout(0.2), nn.Flatten())
        self.mu = nn.Linear(flat, latent)
        self.logvar = nn.Linear(flat, latent) if self.variational else None
        self.dec_fc = nn.Linear(latent, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, self.C, 7, stride=2, padding=3, output_padding=1))

    def encode(self, x):  # x (B, W, C)
        h = self.enc(x.transpose(1, 2))
        return self.mu(h), (self.logvar(h) if self.variational else None)

    def decode(self, z):
        h = self.dec_fc(z).view(z.shape[0], 16, self.L2)
        return self.dec(h)[:, :, :self.W].transpose(1, 2)


class TabAECore(nn.Module):
    """표 AE·VAE(부록 A-8): Linear d → 128 → 64 → 16 → 64 → 128 → d."""

    def __init__(self, d: int, variational: bool, latent: int = LATENT):
        super().__init__()
        self.variational = bool(variational)
        self.enc = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent)
        self.logvar = nn.Linear(64, latent) if self.variational else None
        self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, d))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), (self.logvar(h) if self.variational else None)

    def decode(self, z):
        return self.dec(z)


def _kl_rows(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def _mse_rows(recon, x):
    return (recon - x).pow(2).flatten(1).mean(dim=1)


def _vae_rows(core, x, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(손실 행 = 재구성 MSE (+ KL), μ, logvar). sample=False 면 z = μ(점수·추론)."""
    mu, logvar = core.encode(x)
    if core.variational:
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        rows = _mse_rows(core.decode(z), x) + _kl_rows(mu, logvar)
    else:
        rows = _mse_rows(core.decode(mu), x)
    return rows, mu, logvar


# ---------------------------------------------------------------- 분류 구조
class _RNNNet(nn.Module):
    """LSTM·GRU: RNN(64) 마지막 은닉 → 정적·임베딩 concat → Dense 64(relu) → Dense 1.
    hp(구현 세부 16): RNN 폭·층, dropout = 마지막 은닉 뒤·Dense 64 뒤(기본 0 = 없음)."""

    def __init__(self, cell: str, C: int, W: int, tail: _Tail, hp: Mapping | None = None):
        super().__init__()
        hp = _hp(hp, cell)
        H, d = hp["hidden"], hp["dropout"]
        self.rnn = _rnn(cell, C, H, hp["layers"], d)
        self.tail = tail
        self.head = nn.Sequential(nn.Linear(H + tail.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.drop = _drop(d)

    def forward(self, x_seq, x_static, x_cat):
        _, hn = self.rnn(x_seq)
        h = (hn[0] if isinstance(hn, tuple) else hn)[-1]
        z = _cat(self.drop(h), self.tail(x_static, x_cat))
        z = self.drop(self.head[1](self.head[0](z)))
        return {"logit": self.head[2](z).squeeze(1)}


class _TRFNet(nn.Module):
    """트랜스포머 인코더: 선형 투영 64 + 사인 위치 인코딩 → 인코더 2층(헤드 4·FF 128·dropout 0.1)
    → 평균 풀링 → concat → Dense 64(relu) → Dense 1. hp: d_model·층 수·인코더 dropout."""

    def __init__(self, C: int, W: int, tail: _Tail, hp: Mapping | None = None):
        super().__init__()
        hp = _hp(hp, "TRF")
        H = hp["hidden"]
        self.proj = nn.Linear(C, H)
        self.register_buffer("pe", sinusoidal_encoding(W, H), persistent=False)
        self.enc = nn.TransformerEncoder(_encoder_layer(H, hp["dropout"]), num_layers=hp["layers"],
                                         enable_nested_tensor=False)
        self.tail = tail
        self.head = nn.Sequential(nn.Linear(H + tail.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_seq, x_static, x_cat):
        h = self.enc(self.proj(x_seq) + self.pe[: x_seq.shape[1]]).mean(dim=1)
        return {"logit": self.head(_cat(h, self.tail(x_static, x_cat))).squeeze(1)}


class _TCNNet(nn.Module):
    """TCN: 잔차 블록 5개(커널 3·채널 64·dilation 1~16·dropout 0.1) → 마지막 시점 → concat
    → Dense 64(relu) → Dense 1. hp: 채널 폭·블록 dropout(블록 수 고정)."""

    def __init__(self, C: int, W: int, tail: _Tail, hp: Mapping | None = None):
        super().__init__()
        hp = _hp(hp, "TCN")
        H = hp["hidden"]
        blocks, cin = [], C
        for d in TCN_DILATIONS:
            blocks.append(_TCNBlock(cin, H, d, hp["dropout"]))
            cin = H
        self.tcn = nn.Sequential(*blocks)
        self.tail = tail
        self.head = nn.Sequential(nn.Linear(H + tail.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_seq, x_static, x_cat):
        h = self.tcn(x_seq.transpose(1, 2))[:, :, -1]
        return {"logit": self.head(_cat(h, self.tail(x_static, x_cat))).squeeze(1)}


class _AttnRNNNet(nn.Module):
    """LSTM·GRU + attention: RNN(64, 시퀀스) → Dropout 0.3 → MHA(헤드 2, key 32) + 잔차 + LayerNorm
    → Dense 64(relu) → Dropout 0.3 → 잔차 + LayerNorm → 평균 풀링 → Dropout 0.4 → concat → Dense 1.
    hp: 폭(RNN·어텐션·Dense — 잔차라 같은 폭)·RNN 층 수·dropout 한 값(세 자리 — None 이면 0.3·0.3·0.4)."""

    def __init__(self, cell: str, C: int, W: int, tail: _Tail, hp: Mapping | None = None):
        super().__init__()
        hp = _hp(hp, cell + "A")
        H = hp["hidden"]
        d1, d2, d3 = ATTN_DROPOUTS if hp["dropout"] is None else (hp["dropout"],) * 3
        self.rnn = _rnn(cell, C, H, hp["layers"], d1)
        self.drop1 = nn.Dropout(d1)
        self.mha = nn.MultiheadAttention(H, num_heads=2, batch_first=True)
        self.ln1 = nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Dropout(d2))
        self.ln2 = nn.LayerNorm(H)
        self.drop2 = nn.Dropout(d3)
        self.tail = tail
        self.out = nn.Linear(H + tail.out_dim, 1)

    def forward(self, x_seq, x_static, x_cat):
        a, _ = self.rnn(x_seq)
        a = self.drop1(a)
        z = self.ln1(a + self.mha(a, a, a, need_weights=False)[0])
        z = self.ln2(z + self.ff(z))
        h = self.drop2(z.mean(dim=1))
        return {"logit": self.out(_cat(h, self.tail(x_static, x_cat))).squeeze(1)}


class _CombinedNet(nn.Module):
    """결합-기본: 정형 Dense 128→64(x_static + 임베딩) + 시계열 LSTM(64) → 트랜스포머 인코더 1블록
    (헤드 4·FF 128·dropout 0.1) → 평균 풀링 → Dense 64, concat → Dense 64 → Dense 1.
    결합-VAE(vae=True): + 시퀀스 VAE 갈래, 머리 concat 에 μ 16·복원 오차 1칼럼(기울기 막음).
    hp: 시계열 갈래 폭(LSTM·d_model)·LSTM 층 수·트랜스포머 dropout(LSTM 층 사이도)."""

    def __init__(self, C: int, W: int, tail: _Tail, vae: bool, hp: Mapping | None = None):
        super().__init__()
        if tail.out_dim == 0:
            raise ValueError("결합 모델 정형 갈래에 넣을 x_static·x_cat 이 없음")
        hp = _hp(hp, "CMBV" if vae else "CMB")
        H, d = hp["hidden"], hp["dropout"]
        self.tail = tail
        self.struct = nn.Sequential(nn.Linear(tail.out_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.rnn = _rnn("LSTM", C, H, hp["layers"], d)
        self.enc = nn.TransformerEncoder(_encoder_layer(H, d), num_layers=1, enable_nested_tensor=False)
        self.ts_out = nn.Sequential(nn.Linear(H, 64), nn.ReLU())
        self.vae = SeqVAECore(C, W, variational=True) if vae else None
        head_in = 64 + 64 + ((LATENT + 1) if vae else 0)
        self.head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_seq, x_static, x_cat):
        s = self.struct(self.tail(x_static, x_cat))
        a, _ = self.rnn(x_seq)
        t = self.ts_out(self.enc(a).mean(dim=1))
        parts = [s, t]
        out = {}
        if self.vae is not None:
            vae_rows, mu, _ = _vae_rows(self.vae, x_seq, sample=self.training)
            with torch.no_grad():
                rec_err = _mse_rows(self.vae.decode(mu), x_seq)
            parts += [mu, rec_err.detach().unsqueeze(1)]
            out["vae_rows"] = vae_rows
        out["logit"] = self.head(torch.cat(parts, dim=1)).squeeze(1)
        return out


def _build_classifier(arch: str, C: int, W: int, tail: _Tail, hp: Mapping | None = None) -> nn.Module:
    if arch in ("LSTM", "GRU"):
        return _RNNNet(arch, C, W, tail, hp)
    if arch == "TRF":
        return _TRFNet(C, W, tail, hp)
    if arch == "TCN":
        return _TCNNet(C, W, tail, hp)
    if arch in ("LSTMA", "GRUA"):
        return _AttnRNNNet(arch[:-1], C, W, tail, hp)
    if arch in ("CMB", "CMBV"):
        return _CombinedNet(C, W, tail, vae=(arch == "CMBV"), hp=hp)
    raise ValueError(f"모르는 구조: {arch}")


class _AnomalyNet(nn.Module):
    def __init__(self, kind: str, mode: str, dims: Mapping[str, int]):
        super().__init__()
        var = kind == "VAE"
        self.core = (TabAECore(dims["d"], var) if mode == "tab"
                     else SeqVAECore(dims["C"], dims["W"], var))

    def forward(self, x):
        rows, _, _ = _vae_rows(self.core, x, sample=self.training)
        return rows


# ---------------------------------------------------------------- 학습 틀
class _TorchModel:
    """공통: 학습 고리(조기 종료·최저 에폭 되돌림·메모리 오류 나눔), 추론, 저장·불러오기."""

    max_epochs: int
    batch_size: int

    def __init__(self, *, device: str | None = None, max_epochs: int = MAX_EPOCHS,
                 batch_size: int = BATCH_SIZE):
        if not (isinstance(max_epochs, int) and max_epochs >= 1):
            raise ValueError("max_epochs ≥ 1")
        if not (isinstance(batch_size, int) and batch_size >= 1):
            raise ValueError("batch_size ≥ 1")
        self.device_request = device
        self.max_epochs = max_epochs  # 계획서 = 20(시험에서만 줄임)
        self.batch_size = batch_size  # 계획서 = 1024
        self.net: nn.Module | None = None
        self.dims: dict[str, int] = {}
        self.fit_info: dict[str, Any] = {}
        self._splits = 1

    # -- 하위 클래스가 채움
    def _rows(self, batch: dict, yb: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    # -- 한 배치 학습(메모리 오류면 같은 배치를 나눠 기울기 누적)
    def _step(self, opt, data: Mapping[str, np.ndarray], y: np.ndarray | None, b: np.ndarray,
              epoch: int) -> float:
        dev = self.device
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
        while True:
            try:
                opt.zero_grad(set_to_none=True)
                total = 0.0
                for chunk in np.array_split(b, self._splits):
                    batch = _to_dev(data, chunk, dev)
                    yb = torch.from_numpy(y[chunk]).to(dev) if y is not None else None
                    rows, _ = self._rows(batch, yb)
                    (rows.sum() / len(b)).backward()
                    total += float(rows.detach().sum())
                opt.step()
                return total
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.net.modules()):
                    raise  # B-1: BatchNormalization 구조는 나누지 않음 — GPU 메모리를 확보한 뒤 다시
                if self._splits >= len(b):
                    raise
                self._splits = min(len(b), self._splits * 2)
                self.fit_info.setdefault("oom_events", []).append(
                    {"epoch": epoch, "splits": self._splits})
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

    @torch.no_grad()
    def _eval_rows(self, data: Mapping[str, np.ndarray], y: np.ndarray | None, idx: np.ndarray,
                   which: int) -> np.ndarray:
        """eval 모드 행별 값(which 0 = 학습 손실 행, 1 = 감시 손실 행)."""
        self.net.eval()
        dev = self.device
        out = np.empty(len(idx), dtype=np.float64)
        step = PRED_BATCH
        s = 0
        while s < len(idx):
            c = idx[s:s + step]
            try:
                batch = _to_dev(data, c, dev)
                yb = torch.from_numpy(y[c]).to(dev) if y is not None else None
                out[s:s + len(c)] = self._rows(batch, yb)[which].double().cpu().numpy()
                s += len(c)
            except torch.cuda.OutOfMemoryError:
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if step == 1:
                    raise
                step = max(1, step // 2)
        return out

    def _train(self, data: Mapping[str, np.ndarray], y: np.ndarray | None, tr_idx: np.ndarray,
               es_idx: np.ndarray, seed: int) -> None:
        gen = torch.Generator().manual_seed(int(seed))
        opt = torch.optim.Adam(self.net.parameters(), lr=getattr(self, "lr", LR))
        best, best_state, best_ep, bad = math.inf, None, 0, 0
        tr_hist, es_hist = [], []
        for ep in range(1, self.max_epochs + 1):
            self.net.train()
            perm = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
            total = 0.0
            for s in range(0, len(perm), self.batch_size):
                total += self._step(opt, data, y, perm[s:s + self.batch_size], ep)
            tr_hist.append(total / len(perm))
            es_loss = float(self._eval_rows(data, y, es_idx, which=1).mean())
            es_hist.append(es_loss)
            if es_loss < best:  # Keras EarlyStopping(min_delta 0) 와 같은 '줄어야 개선'
                best, best_ep, bad = es_loss, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is None:
            raise FloatingPointError(f"조기 종료 손실이 한 번도 유한하지 않음: {es_hist}")
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.fit_info.update({"train_loss": tr_hist, "es_loss": es_hist, "best_epoch": best_ep,
                              "best_es_loss": best, "epochs_run": len(es_hist),
                              "final_splits": self._splits})

    def _prepare_fit(self, seed: int, n_jobs: int, eval_set) -> dict:
        if eval_set is not None:
            raise ValueError("딥러닝 조기 종료는 train 의 (sgg, d) 묶음 10% 로만(부록 A-8) — eval_set 안 받음")
        torch.set_num_threads(check_n_jobs(n_jobs))
        det = set_determinism(seed)
        dev = resolve_device(self.device_request)
        self._splits = 1
        self.fit_info = {"determinism": det, "device": str(dev), "n_jobs": n_jobs, "oom_events": []}
        return {"device": dev}

    # -- 저장·불러오기
    def _payload(self) -> dict:
        raise NotImplementedError

    def save(self, path: str | os.PathLike, overwrite: bool = False) -> Path:
        if self.net is None:
            raise RuntimeError("학습 전 모델은 저장 안 함")
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"덮어쓰기 금지: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": FORMAT, "class": type(self).__name__, **_py(self._payload()),
                   "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()}}
        tmp = p.with_name(p.name + ".tmp")
        try:
            torch.save(payload, tmp)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, p)
        return p

    @staticmethod
    def _read(path, cls_name: str) -> dict:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != FORMAT or payload.get("class") != cls_name:
            raise ValueError(f"다른 형식의 파일: {payload.get('format')}/{payload.get('class')}")
        return payload


class DLClassifier(_TorchModel):
    """분류 딥러닝(LSTM·GRU·TRF·TCN·LSTMA·GRUA·CMB·CMBV).

    DLClassifier(arch, cat_sizes=(n_sido, n_sgg, n_p2, n_p3) | None, *, oov_index=0, device=None, hparams=None)
    .fit(inputs, y, *, seed, n_jobs, groups, eval_set=None) -> self
    .predict_proba(inputs, n_iter=None) -> float64 (N,) 양성 확률
    .save(path) / DLClassifier.load(path, device=None)
    cat_sizes = 범주 번호표 크기(OOV 포함, 부록 A-8) — None 이면 x_cat 을 안 받음(⑥·⑦-D).
    hparams = 단계 8-D 설정(구현 세부 16 — None·빈 dict = 부록 A-8 기본 그대로), 배치 크기는 batch_size.
    """

    def __init__(self, arch: str, cat_sizes: Sequence[int] | None = None, *, oov_index: int = 0,
                 device: str | None = None, max_epochs: int = MAX_EPOCHS, batch_size: int = BATCH_SIZE,
                 hparams: Mapping[str, Any] | None = None):
        super().__init__(device=device, max_epochs=max_epochs, batch_size=batch_size)
        if arch not in CLASSIFIER_ARCHS:
            raise ValueError(f"모르는 구조 {arch!r} — {CLASSIFIER_ARCHS}")
        if cat_sizes is not None:
            cat_sizes = [int(n) for n in cat_sizes]
            if len(cat_sizes) != len(CAT_COLUMNS):
                raise ValueError(f"cat_sizes 는 {CAT_COLUMNS} 4개")
        self.arch = arch
        self.cat_sizes = cat_sizes
        self.oov_index = int(oov_index)
        self.hparams = normalize_hparams(arch, hparams) if hparams else None
        self.lr = self.hparams["lr"] if self.hparams else LR

    @property
    def _keys(self) -> tuple[list[str], list[str]]:
        req = ["x_seq", "x_static"] + (["x_cat"] if self.cat_sizes is not None else [])
        return req, []

    def _check(self, inputs) -> int:
        req, opt = self._keys
        n = _check_inputs(inputs, req, opt)
        if self.cat_sizes is not None:
            xc = inputs["x_cat"]
            if xc.shape[1] != len(self.cat_sizes):
                raise ValueError(f"x_cat 칼럼 {xc.shape[1]} ≠ {len(self.cat_sizes)}")
            for j, m in enumerate(self.cat_sizes):
                lo, hi = int(xc[:, j].min()), int(xc[:, j].max())
                if lo < 0 or hi >= m:
                    raise ValueError(f"x_cat[{CAT_COLUMNS[j]}] 번호 {lo}~{hi} 가 번호표 0~{m - 1} 밖")
        return n

    def _dims_of(self, inputs) -> dict[str, int]:
        _, W, C = inputs["x_seq"].shape
        return {"W": int(W), "C": int(C), "S": int(inputs["x_static"].shape[1])}

    def _build(self) -> nn.Module:
        tail = _Tail(self.dims["S"], self.cat_sizes, self.oov_index)
        return _build_classifier(self.arch, self.dims["C"], self.dims["W"], tail, self.hparams)

    def _rows(self, batch, yb):
        out = self.net(batch["x_seq"], batch["x_static"], batch.get("x_cat"))
        bce = F.binary_cross_entropy_with_logits(out["logit"], yb, reduction="none")
        train = train_rows_loss(out["logit"], yb, bce, self.hparams)
        train = train + out["vae_rows"] if "vae_rows" in out else train
        return train, bce

    def fit(self, inputs: Mapping[str, np.ndarray], y, *, seed: int, n_jobs: int, groups,
            eval_set=None) -> "DLClassifier":
        n = self._check(inputs)
        yv = _check_y(y, n)
        es = split_es_groups(groups, seed)
        if len(es) != n:
            raise ValueError(f"groups 길이 {len(es)} ≠ 행 수 {n}")
        tr_idx, es_idx = np.flatnonzero(~es), np.flatnonzero(es)
        if len(tr_idx) == 0:
            raise ValueError("학습 행 0개")
        ctx = self._prepare_fit(seed, n_jobs, eval_set)
        self.dims = self._dims_of(inputs)
        torch.manual_seed(int(seed))  # 층 초기화
        self.net = self._build().to(ctx["device"])
        if self.cat_sizes is not None:
            self.net.tail.cat.set_seen(inputs["x_cat"][tr_idx])
        self.fit_info.update({"n_rows": n, "n_train": int(len(tr_idx)), "n_es": int(len(es_idx)),
                              "n_pos_train": int(yv[tr_idx].sum()), "n_pos_es": int(yv[es_idx].sum()),
                              "es_rows_md5": _state_md5(es_idx)})
        self._train(dict(inputs), yv, tr_idx, es_idx, seed)
        return self

    @torch.no_grad()
    def predict_proba(self, inputs: Mapping[str, np.ndarray], n_iter=None) -> np.ndarray:
        if n_iter is not None:
            raise ValueError("딥러닝은 n_iter 없음(최저 에폭 가중치 하나)")
        if self.net is None:
            raise RuntimeError("학습 전")
        n = self._check(inputs)
        if self._dims_of(inputs) != self.dims:
            raise ValueError(f"입력 모양 {self._dims_of(inputs)} ≠ 학습 때 {self.dims}")
        self.net.eval()
        dev = self.device
        out = np.empty(n, dtype=np.float64)
        step, s = PRED_BATCH, 0
        while s < n:
            c = np.arange(s, min(n, s + step))
            try:
                b = _to_dev(inputs, c, dev)
                logit = self.net(b["x_seq"], b["x_static"], b.get("x_cat"))["logit"]
                out[s:s + len(c)] = torch.sigmoid(logit.double()).cpu().numpy()
                s += len(c)
            except torch.cuda.OutOfMemoryError:
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if step == 1:
                    raise
                step = max(1, step // 2)
        return out

    def get_config(self) -> dict:
        """B-4 기록용 구조·학습 설정(hparams 가 있을 때만 'hparams' 칸 — 기본 판 기록은 그대로)."""
        hp = self.hparams
        loss = "binary_crossentropy"
        if hp is not None:
            loss = ("focal(gamma={focal_gamma},alpha={focal_alpha})".format(**hp) if hp["loss"] == "focal" else loss)
            if hp["pos_weight"] != 1.0:
                loss += f"*pos_weight({hp['pos_weight']})"
        out = {"arch": self.arch, "cat_columns": list(CAT_COLUMNS) if self.cat_sizes else None,
               "cat_sizes": self.cat_sizes,
               "emb_dims": [embedding_dim(m) for m in self.cat_sizes] if self.cat_sizes else None,
               "oov_index": self.oov_index, "dims": dict(self.dims), "lr": self.lr,
               "optimizer": "Adam", "batch_size": self.batch_size, "max_epochs": self.max_epochs,
               "patience": PATIENCE, "es_fraction": ES_FRACTION, "es_unit": "(sgg, d)",
               "loss": loss + ("+vae(mse+kl)" if self.arch == "CMBV" else ""),
               "monitor": "es_binary_crossentropy", "restore_best": True,
               "n_params": int(sum(p.numel() for p in self.net.parameters())) if self.net else None}
        if hp is not None:
            out["hparams"] = dict(hp)
        return out

    def _payload(self) -> dict:
        out = {"arch": self.arch, "cat_sizes": self.cat_sizes, "oov_index": self.oov_index,
               "dims": self.dims, "max_epochs": self.max_epochs, "batch_size": self.batch_size,
               "fit_info": self.fit_info}
        if self.hparams is not None:      # 기본 판 파일 꼴은 그대로
            out["hparams"] = dict(self.hparams)
        return out

    @classmethod
    def load(cls, path, device: str | None = None) -> "DLClassifier":
        p = cls._read(path, cls.__name__)
        m = cls(p["arch"], p["cat_sizes"], oov_index=p["oov_index"], device=device,
                max_epochs=p["max_epochs"], batch_size=p["batch_size"], hparams=p.get("hparams"))
        m.dims = dict(p["dims"])
        m.fit_info = p["fit_info"]
        m.net = m._build()
        m.net.load_state_dict(p["state_dict"])
        m.net.to(resolve_device(device)).eval()
        return m


class AnomalyModel(_TorchModel):
    """AE·VAE 이상 점수(단계 3-1). kind ∈ {AE, VAE}, mode ∈ {tab(x_tab), seq(x_seq)}.

    .fit(inputs, y, *, seed, n_jobs, groups, eval_set=None) -> self   (y == 0 행만 학습)
    .raw_score(inputs) -> float64 (N,)   AE = 복원 MSE, VAE = 음의 ELBO(z = μ)
    .score(inputs)     -> float64 (N,)   robust z(기준 = 조기 종료 묶음 10% 의 불검출 행)
    .save(path) / AnomalyModel.load(path, device=None)
    """

    def __init__(self, kind: str, mode: str, *, device: str | None = None,
                 max_epochs: int = MAX_EPOCHS, batch_size: int = BATCH_SIZE):
        super().__init__(device=device, max_epochs=max_epochs, batch_size=batch_size)
        if kind not in ANOMALY_KINDS or mode not in ANOMALY_MODES:
            raise ValueError(f"kind {ANOMALY_KINDS}·mode {ANOMALY_MODES}: {kind}, {mode}")
        self.kind, self.mode = kind, mode
        self.key = "x_tab" if mode == "tab" else "x_seq"
        self.z_params: dict | None = None

    def _dims_of(self, inputs) -> dict[str, int]:
        a = inputs[self.key]
        return {"d": int(a.shape[1])} if self.mode == "tab" else {"W": int(a.shape[1]), "C": int(a.shape[2])}

    def _rows(self, batch, yb):
        rows = self.net(batch[self.key])
        return rows, rows

    def fit(self, inputs: Mapping[str, np.ndarray], y, *, seed: int, n_jobs: int, groups,
            eval_set=None) -> "AnomalyModel":
        n = _check_inputs(inputs, [self.key], [k for k in ("x_seq", "x_static", "x_cat", "x_tab")
                                                if k != self.key])
        yv = _check_y(y, n)
        neg = np.flatnonzero(yv == 0)
        if isinstance(groups, (tuple, list)) and len(groups) == 2 and not np.isscalar(groups[0]):
            g_neg = tuple(np.asarray(g)[neg] for g in groups)
        else:
            g_neg = np.asarray(groups)[neg]
        es_neg = split_es_groups(g_neg, seed)
        tr_idx, es_idx = neg[~es_neg], neg[es_neg]
        ctx = self._prepare_fit(seed, n_jobs, eval_set)
        self.dims = self._dims_of(inputs)
        torch.manual_seed(int(seed))
        self.net = _AnomalyNet(self.kind, self.mode, self.dims).to(ctx["device"])
        self.fit_info.update({"n_rows": n, "n_neg": int(len(neg)), "n_train": int(len(tr_idx)),
                              "n_es": int(len(es_idx)), "es_rows_md5": _state_md5(es_idx)})
        data = {self.key: inputs[self.key]}
        self._train(data, None, tr_idx, es_idx, seed)
        base = self._eval_rows(data, None, es_idx, which=0)  # 최저 에폭 가중치로, eval(z = μ)
        self.z_params = robust_z_params(base)
        return self

    def raw_score(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("학습 전")
        n = _check_inputs(inputs, [self.key], [k for k in ("x_seq", "x_static", "x_cat", "x_tab")
                                                if k != self.key])
        if self._dims_of(inputs) != self.dims:
            raise ValueError(f"입력 모양 {self._dims_of(inputs)} ≠ 학습 때 {self.dims}")
        return self._eval_rows({self.key: inputs[self.key]}, None, np.arange(n), which=0)

    def score(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        return robust_z(self.raw_score(inputs), self.z_params)

    def get_config(self) -> dict:
        return {"kind": self.kind, "mode": self.mode, "input": self.key, "dims": dict(self.dims),
                "latent": LATENT, "lr": LR, "optimizer": "Adam", "batch_size": self.batch_size,
                "max_epochs": self.max_epochs, "patience": PATIENCE, "es_fraction": ES_FRACTION,
                "es_unit": "(sgg, d)", "train_rows": "y == 0",
                "loss": "mse" if self.kind == "AE" else "mse+kl",
                "score": "recon_mse" if self.kind == "AE" else "neg_elbo(z=mu)",
                "robust_z": self.z_params,
                "n_params": int(sum(p.numel() for p in self.net.parameters())) if self.net else None}

    def _payload(self) -> dict:
        return {"kind": self.kind, "mode": self.mode, "dims": self.dims, "max_epochs": self.max_epochs,
                "batch_size": self.batch_size, "fit_info": self.fit_info, "z_params": self.z_params}

    @classmethod
    def load(cls, path, device: str | None = None) -> "AnomalyModel":
        p = cls._read(path, cls.__name__)
        m = cls(p["kind"], p["mode"], device=device, max_epochs=p["max_epochs"], batch_size=p["batch_size"])
        m.dims = dict(p["dims"])
        m.fit_info = p["fit_info"]
        m.z_params = p["z_params"]
        m.net = _AnomalyNet(m.kind, m.mode, m.dims)
        m.net.load_state_dict(p["state_dict"])
        m.net.to(resolve_device(device)).eval()
        return m
