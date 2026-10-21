# -*- coding: utf-8 -*-
"""AE·VAE 이상 탐지 점수 (사용자 지시 2026-10-19, 단계 'ae': 실험 S_*·A3_AE*·A3_*_p). 설정 = e2cfg.AE.
[점수 규칙 v2 - 사용자 승인 2026-10-19, SCORE_RULE] 학습 행은 검출·불검출 모두 5겹에 무작위 배정(default_rng([seed, 3]))해 '그 행을 안 본 겹 모델'
점수, 검증·시험 = 겹 모델 5개 점수 평균. 각 겹 모델은 자기 겹 밖의 학습 구간 불검출 행으로 학습·정규화. 전체 정상 행 모델은 쓰지 않음.
(v1 = 불검출 행은 겹 모델, 검출 행·검증·시험은 전체 정상 행 모델 -> 학습 구간에서 점수 출처가 정답과 엮여 시드에 따라 학습이 그 차이를 잡음,
 FS12 시드 42 확인 logs/fs12_s42_check_server.txt.) 아래 '교차 적합(학습 행 점수)'·'검증·시험 = 같은 전체 정상 행 모델' 문단은 v1 설명.
아래 칸 수·길이는 기상 창 L = 50 일 때 값. 창 길이가 다르면(e2cfg.L, 환경변수 EXP2_L) 표 숫자 칸 = 1 + 기상 요약 NW + 9 = N_TAB,
시퀀스 길이 = L (합성곱 길이 L -> SEQ_LENS) - 코드는 이 값을 계산해서 씀(L=100: 표 295칸, 시퀀스 100 -> 50 -> 25 -> 50 -> 100).

모델(kind) 4개
  AEt  표 오토인코더   : 입력 = A3 숫자 200칸(h 1 + 기상 요약 190 + 이력 9, k*; A3 와 같은 FeatureBuilder) 표준화 + 범주 4칸(sido, sgg, p2, p3)
                        임베딩 각 8차원. MLP 인코더 Linear(200+32 -> 128) ReLU Linear(-> 64) ReLU Linear(-> 16),
                        디코더 Linear(16 -> 64) ReLU Linear(-> 128) ReLU Linear(-> 200). 복원 대상 = 표준화된 숫자 200칸(범주는 인코더 입력만).
  VAEt 표 VAE          : 같은 뼈대에서 인코더 끝을 잠재 평균·로그분산 Linear(64 -> 16) 둘로.
  AEs  시퀀스 오토인코더: 입력 = DL 실험과 같은 50일 x 38채널(e2dl_impl.build_seq 그대로 호출: 표준화 19(결측 0) + 결측 표시 19),
                        창 T−50 … T−1 (T = d − h, T 당일 제외). 범주·h 입력 없음.
                        keras 시계열 이상 탐지 예제 구조(Conv1D 인코더 -> Conv1DTranspose 디코더):
                        Conv1D(38->32, 7, 보폭 2, same) ReLU Dropout(0.2) -> Conv1D(32->16, 7, 보폭 2) ReLU ->
                        ConvTranspose1D(16->16, 7, 보폭 2) ReLU Dropout(0.2) -> ConvTranspose1D(16->32, 7, 보폭 2) ReLU -> ConvTranspose1D(32->38, 7)
                        (길이 50 -> 25 -> 13 -> 25 -> 50, 전치 합성곱 출력 길이는 인코더 길이에 맞춤). 복원 대상 = 입력 38채널 전부.
                        계산은 nn.Conv2d 커널 (7, 1) 꼴(Conv1d 와 같은 계산 - SeqNet 설명 참고).
  VAEs 시퀀스 VAE      : 같은 뼈대에서 두 번째 Conv1D 를 잠재 평균·로그분산 Conv1D(32->16, 7, 보폭 2) 둘로(잠재 16채널 x 13, 활성화 없음).

정상 = 학습 구간(기준일 <= 2024-06-30) 불검출(y=0) 행만. 검출 행·검증·시험 행은 학습·정규화·어휘에 들어가지 않음.
  정규화(모델마다 그 모델에 배정된 정상 행으로 - 조기 종료용 일부 포함):
    표   : 숫자 200칸 칸별 평균·표준편차(결측 무시, ddof=0; 표준편차 0/전부 결측 칸은 1, 평균 결측은 0) -> 표준화 뒤 결측 0(= 평균, B2 와 같은 규칙).
           임베딩 어휘 = 그 행들에 있던 범주만 1..n, 없던 범주 = 0번 = 0 벡터(padding_idx=0, 학습 안 됨).
    시퀀스: 19변수별 평균·표준편차 = 그 행들의 창(T−50 … T−1)에 들어가는 (시군구, 날짜) 칸을 창 겹침 수로 가중(결측 무시, ddof=0)
           -> 그 값으로 build_seq(1차 LSTM 과 같은 식)를 호출(DL 실험은 학습 기간 기상표 전체 통계 - 여기만 다름).
점수(클수록 이상 = '검출 쪽'으로 고정):
  AE  = 복원 오차 MSE (행마다 복원 대상 칸 평균)
  VAE = 음의 ELBO = 0.5·Σ(복원 오차²) + KL(q(z|x) || N(0, I)) (가우스 복호기 분산 1, 상수항 뺌). 점수 계산 때 z = 잠재 평균(결정적).
       학습 손실은 같은 식에 z = 평균 + 표준편차·ε(재매개화) 표본 1개.
행과 h: CatBoost 학습 행은 시드마다 h 하나(ctx.h_fit(seed)['train']) -> 학습 행 점수는 그 (행, h) 입력으로. AE 학습 입력도 그 h.
  검증·시험 = h 1..60 전부 (행, h) 마다 점수.
교차 적합(학습 행 점수): 학습 구간 불검출 행을 default_rng([seed, 1]) 순열로 5겹(크기 차이 1 이하) -> 겹 f 행 점수 = 겹 f 를 뺀 정상 행으로 학습한 모델.
  학습 구간 검출 행 점수 = 전체 정상 행으로 학습한 모델. 검증·시험 점수 = 같은 전체 정상 행 모델.
학습: Adam lr 1e-3, 배치 1024, 최대 50에폭, 조기 종료 = 그 모델 정상 행 중 무작위 10%(default_rng([seed, 2, 모델 번호]))의 평균 점수,
  인내 5, 최고 에폭 가중치 복원. 초기값·dropout·재매개화 = torch.manual_seed(seed x 10 + 모델 번호)(겹 0~4, 전체 = 5),
  에폭 셔플 = torch.Generator(같은 값). GPU 필수. 재현성: torch.use_deterministic_algorithms(True)·cudnn.deterministic=True·
  cudnn.benchmark=False(계산 동안만, 끝나면 되돌림), CUBLAS_WORKSPACE_CONFIG=:4096:8(이 모듈을 처음 import 할 때 없으면 넣음).
위약판(_p): 점수를 분할마다 예측일 T 의 달력 연도 안에서 행끼리 섞음(default_rng(20261019) 하나로 차례대로: 학습 = 그 시드 h 의 T 연도,
  검증 -> 시험 = h 1..60 마다 그 h 의 T 연도, 연도 오름차순). 값 모음은 연도마다 그대로.
캐시: (kind, 시드)마다 한 번 계산해 ctx 메모리 + 실행 결과 폴더 아래 _ae_scores/<kind>_s<시드>_<full|smoke>.npz(+ .json 메타)에 저장.
  메타(행·h·설정·코드 md5·torch 버전 등)가 지금과 같을 때만 다시 읽음(다르면 새로 계산해 덮어씀). 같은 시드의 S_·A3_·_p 가 같은 점수를 씀.
"""
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

_CUBLAS_PRESET = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # 결정적 cuBLAS (CUDA 를 처음 쓰기 전에 있어야 함)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

# 이 모듈보다 먼저 같은 프로세스에서 CUDA 를 쓴 경우(예: 같은 실행에서 DL 단계가 먼저 돎) cuBLAS 설정이 늦게 들어가 재현성이 보장되지 않을 수 있음 -> 기록·경고
_CUDA_INIT_BEFORE = bool(torch.cuda.is_initialized()) and _CUBLAS_PRESET is None

import e2cfg as E  # noqa: E402  (1차 src 를 sys.path 에 추가하므로 config 보다 먼저)
import config as C1  # noqa: E402

AEC = E.AE
KINDS = list(E.AE_KINDS)
PLACEBO = "_p"
N_CH = 2 * len(C1.WEATHER_VARS)              # 38 (build_seq 와 같음)
SCORE_DIR = "_ae_scores"
SCORE_RULE = ("v2: 학습 행은 검출·불검출 모두 5겹에 무작위 배정(default_rng([seed, 3]))해 그 행을 안 본 겹 모델 점수, 검증·시험 = 겹 모델 5개 점수 평균"
              "(float64 로 더해 5 로 나눈 뒤 float32). 겹 모델 = 자기 겹 밖 학습 구간 불검출 행으로 학습·정규화. 전체 정상 행 모델은 쓰지 않음")
SCORE_RULE_TAG = "v2"
# 표 AE 숫자 칸 수 = A3 숫자 칸(h 1 + 기상 요약 NW + A3 이력 칸). L=50 -> 1 + 190 + 9 = 200
N_TAB = 1 + E.NW + sum(len(E.HIST_SETS[x]) for x in E.EXP_BY_ID["A3"]["feats"]["hist"])


def _conv_len(n):
    """Conv(커널 k, 보폭 s, 패딩 k//2) 출력 길이 = floor((n + 2·(k//2) − k) / s) + 1."""
    k, st = AEC["seq"]["kernel"], AEC["seq"]["stride"]
    return (n + 2 * (k // 2) - k) // st + 1


_SL1 = _conv_len(AEC["seq"]["L"])
_SL2 = _conv_len(_SL1)
SEQ_LENS = f"{AEC['seq']['L']} -> {_SL1} -> {_SL2} -> {_SL1} -> {AEC['seq']['L']}"     # L=50: 50 -> 25 -> 13 -> 25 -> 50
CODE_FILES2 = ["e2ae.py", "e2dl_impl.py", "e2models.py", "e2hist.py", "e2data.py"]   # 점수 계산에 쓰는 2차 모듈(캐시 메타)
CODE_FILES1 = ["data.py", "features.py"]                                           # 1차 모듈


# ---------------------------------------------------------------- 이름·설명
def parse_block(b):
    """'AEt' -> ('AEt', False), 'AEt_p' -> ('AEt', True)."""
    plc = b.endswith(PLACEBO)
    kind = b[: -len(PLACEBO)] if plc else b
    if kind not in KINDS:
        raise ValueError(f"모르는 AE 묶음: {b}")
    return kind, plc


def is_tab(kind):
    return kind in ("AEt", "VAEt")


def is_var(kind):
    return kind in ("VAEt", "VAEs")


def ae_names(blocks):
    return [f"ae_score_{k}" + ("_placebo" if p else "") for k, p in map(parse_block, blocks)]


def tab_cols(tab_feats):
    """주어진 표 AE 입력(해석된 입력 dict)의 (숫자 칸 이름, 범주 칸 이름) - FeatureBuilder 와 같은 규칙(자료 없이)."""
    from e2models import num_feature_names
    import e2fg
    cats = (list(C1.CAT_COLS) if tab_feats.get("cats") else []) + (e2fg.ext_cat_names(tab_feats["ext"]) if tab_feats.get("ext") else [])
    return num_feature_names(tab_feats), cats


def tab_sig(tab_feats):
    """표 AE 입력 구분값(입력 칸 목록 + 입력 설정의 md5 앞 12자리). tab_feats 없으면(A3 기본) None."""
    if tab_feats is None:
        return None
    n, c = tab_cols(tab_feats)
    return hashlib.md5(json.dumps({"num": n, "cat": c, "feats": tab_feats}, sort_keys=True, ensure_ascii=False,
                                  default=str).encode("utf-8")).hexdigest()[:12]


def input_text(kind, tab_feats=None):
    if is_tab(kind):
        if tab_feats is not None:
            n, c = tab_cols(tab_feats)
            return (f"기준 모델 입력 숫자 {len(n)}칸 표준화 + 범주 {len(c)}칸({', '.join(c)}) 임베딩(각 {AEC['tab']['emb_dim']}) "
                    f"- 입력 칸 목록 md5 {tab_sig(tab_feats)}")
        return f"A3 숫자 {N_TAB}칸(h + 기상 요약 {E.NW} + 이력 {N_TAB - 1 - E.NW}, k*) 표준화 + 범주 4칸 임베딩(각 {AEC['tab']['emb_dim']})"
    return f"기상 {E.L}일 x {N_CH}채널(build_seq: 표준화 19 + 결측 표시 19, 창 T−{E.L}…T−1)"


def ae_desc(blocks):
    """표 '입력' 칸(A3 + 점수)."""
    return "이상 점수[" + ", ".join(E.AE_KIND_DESC[k] + (" 위약판" if p else "") + " 1" for k, p in map(parse_block, blocks)) + "]"


def ae_count_desc(blocks):
    """config.json n_features_desc 용."""
    return "이상 점수 " + str(len(blocks)) + "(" + ", ".join(k + ("_p" if p else "") for k, p in map(parse_block, blocks)) + ")"


def score_only_desc(kind):
    """표 '입력' 칸(점수 단독 S_*)."""
    return f"이상 점수 단독: {E.AE_KIND_DESC[kind]} (입력 {input_text(kind)}) - CatBoost 없음, 예측값 p = 점수(확률 아님)"


def arch_tab(n, n_cat=4):
    return (f"MLP: [숫자 {n} + 임베딩 {n_cat}x{AEC['tab']['emb_dim']}] -> Linear({AEC['tab']['hidden'][0]}) ReLU -> Linear({AEC['tab']['hidden'][1]}) ReLU -> "
            f"잠재 {AEC['tab']['latent']}(AE: Linear, VAE: 평균·로그분산 Linear 둘) -> Linear({AEC['tab']['hidden'][1]}) ReLU -> "
            f"Linear({AEC['tab']['hidden'][0]}) ReLU -> Linear({n}). 복원 대상 = 표준화 숫자 {n}칸")


ARCH_TEXT = {
    "tab": arch_tab(N_TAB),
    "seq": (f"1D 합성곱(keras 시계열 이상 탐지 예제 구조): Conv1D(38->{AEC['seq']['filters'][0]}, {AEC['seq']['kernel']}, 보폭 {AEC['seq']['stride']}, same) ReLU "
            f"Dropout({AEC['seq']['dropout']}) -> Conv1D(->{AEC['seq']['filters'][1]}, {AEC['seq']['kernel']}, 보폭 {AEC['seq']['stride']}) ReLU(VAE: 평균·로그분산 "
            f"Conv1D 둘, 활성화 없음) -> ConvTranspose1D(->{AEC['seq']['filters'][1]}) ReLU Dropout({AEC['seq']['dropout']}) -> "
            f"ConvTranspose1D(->{AEC['seq']['filters'][0]}) ReLU -> ConvTranspose1D(->38, {AEC['seq']['kernel']}, 보폭 1). 길이 {SEQ_LENS}. "
            "복원 대상 = 입력 38채널 전부. 계산은 Conv2d 커널 (7, 1) 꼴(같은 계산, 이 GPU 에서 빠름)"),
}
RULE_NORMAL = ("정상 = 학습 구간(기준일 <= 2024-06-30) 불검출(y=0) 행만 학습에 씀. 검출 행·검증·시험 행은 학습·조기 종료·정규화 통계·임베딩 어휘에 들어가지 않음. "
               "정규화는 모델마다 그 모델에 배정된 정상 행으로: 표 = 칸별 평균·표준편차(결측 무시, ddof=0; 표준편차 0/전부 결측 칸 1) 뒤 결측 0, "
               f"임베딩 어휘 = 그 행에 있던 범주만(없던 범주 = 0 벡터); 시퀀스 = 그 행들의 창(T−{AEC['seq']['L']}…T−1)에 든 (시군구, 날짜) 칸을 창 겹침 수로 가중한 "
               "19변수 평균·표준편차로 build_seq 호출(DL 실험은 학습 기간 기상표 전체 통계 - 여기만 다름)")
RULE_SCORE = ("점수: AE = 복원 오차 MSE(행마다 복원 대상 칸 평균), VAE = 음의 ELBO = 0.5·Σ(복원 오차²) + KL(q(z|x)||N(0,I)) "
              "(가우스 복호기 분산 1, 상수 뺌, 점수 계산 때 z = 잠재 평균). 클수록 이상 = '검출 쪽'으로 고정(방향을 결과 보고 바꾸지 않음)")
RULE_ROWS = ("행과 h: CatBoost 학습 행은 시드마다 h 하나(ctx.h_fit(seed)) -> 학습 행 점수와 AE 학습 입력은 그 (행, h). 검증·시험은 h 1..60 전부 (행, h) 마다. "
             f"[점수 규칙 v2] 학습 행 전부(검출·불검출)를 default_rng([seed, 3]) 순열로 {AEC['n_folds']}겹에 배정 -> 겹 f 행의 점수 = 겹 f 밖 학습 구간 불검출 행으로 "
             "학습한 겹 f 모델(검출 행도 같은 방식 - 점수 출처가 정답과 무관). 검증·시험 점수 = 겹 모델 5개 점수 평균. 전체 정상 행 모델은 쓰지 않음")
RULE_TRAIN = (f"학습: Adam lr {AEC['lr']}, 배치 {AEC['batch']}, 최대 {AEC['max_epochs']}에폭, 조기 종료 = 그 모델 정상 행 중 무작위 "
              f"{AEC['es_frac']:.0%}(default_rng([seed, 2, 모델 번호]))의 평균 점수, 인내 {AEC['patience']}, 최고 에폭 가중치 복원. "
              "초기값·dropout·재매개화 = torch.manual_seed(seed x 10 + 모델 번호: 겹 0~4, 전체 정상 행 모델 5), 셔플 = torch.Generator(같은 값). "
              "크기·학습률·에폭 = 기본값(하이퍼파라미터 조정 안 함 - 사용자 지시). GPU, 재현성 설정 = use_deterministic_algorithms(True)·"
              "cudnn.deterministic=True·benchmark=False(점수 계산 동안만)·CUBLAS_WORKSPACE_CONFIG=:4096:8")
RULE_PLACEBO = (f"위약판(_p): 점수를 분할마다 예측일 T 의 달력 연도 안에서 행끼리 섞음(default_rng({AEC['placebo_seed']}) 하나로 차례대로 - "
                "학습 행 = 그 시드 h 의 T 연도 묶음, 검증 -> 시험 = h 1..60 마다 그 h 의 T 연도 묶음, 연도 오름차순). 연도마다 값 모음은 그대로, "
                "행과 점수의 짝만 끊음(조기 종료용 검증 칸 = 같은 위약 행렬의 그 행 h 값)")
RULE_CACHE = ("점수는 (kind, 시드)마다 한 번 계산해 실행 결과 폴더 아래 _ae_scores/<kind>_s<시드>_<full|smoke>[_t<표 입력 구분값>]_v2.npz 에 저장(메타 .json: "
              "점수 규칙·행·h·설정·코드 md5·torch 버전). 메타가 같을 때만 다시 읽음 -> 같은 시드의 S_·A3_·_p 가 같은 점수를 씀. cache/ 폴더는 쓰지 않음. "
              "v1(옛 규칙) 점수 파일은 이름이 달라 섞이지 않음")


def ae_info(blocks, tab_feats=None):
    """config.json 기록용(A3 + 점수 실험). tab_feats = 표 AE 입력(단계 선택 ae4: 그 시점 기준 모델 입력, 없으면 A3 - 이전과 같은 글)."""
    out = {"blocks": list(blocks), "names": ae_names(blocks), "rule_normal": RULE_NORMAL, "rule_score": RULE_SCORE,
           "rule_rows": RULE_ROWS, "rule_train": RULE_TRAIN, "cache": RULE_CACHE, "config": AEC}
    if tab_feats is not None:
        n, c = tab_cols(tab_feats)
        out["tab_input_feats"] = tab_feats
        out["tab_input_sig"] = tab_sig(tab_feats)
        out["tab_input_rule"] = ("표 AE·VAE 입력 = 단계 선택에서 그 시점 기준 모델의 입력(숫자 칸 표준화 + 범주 칸 임베딩, aw_label 같은 추가 범주는 "
                                 "전체 행의 값 목록으로 번호를 매긴 뒤 같은 어휘 규칙). 점수 캐시 파일 이름에 입력 구분값(tab_sig)을 붙여 A3 입력 점수와 섞이지 않게 함")
    for k, p in map(parse_block, blocks):
        if tab_feats is not None and is_tab(k):
            n, c = tab_cols(tab_feats)
            out[k] = {"input": input_text(k, tab_feats), "arch": arch_tab(len(n), len(c)), "variational": is_var(k)}
        else:
            out[k] = {"input": input_text(k), "arch": ARCH_TEXT["tab" if is_tab(k) else "seq"], "variational": is_var(k)}
        if p:
            out["placebo"] = RULE_PLACEBO
    return out


def settings_rows():
    """표 '설정' 시트 줄(단계 'ae')."""
    return [("AE 모델(S_·A3_·_p 공통)", f"AEt·VAEt = 표({input_text('AEt')}), {ARCH_TEXT['tab']}. "
                                       f"AEs·VAEs = 시퀀스({input_text('AEs')}), {ARCH_TEXT['seq']}"),
            ("AE 정상 행·정규화", RULE_NORMAL),
            ("AE 점수", RULE_SCORE),
            ("AE 행과 h·교차 적합", RULE_ROWS),
            ("AE 학습", RULE_TRAIN),
            ("AE 위약판(A3_*_p)", RULE_PLACEBO),
            ("AE 점수 단독(S_*)", "CatBoost 없음. 점수를 그대로 예측값 p 로 저장·채점(PR-AUC·ROC-AUC = 순위만 씀, 판정 기준값 = 검증 h 1~60 합친 점수의 "
                                "F2 최대 점수값 -> 시험에 그대로, 다른 실험과 같은 e2eval). p 는 확률이 아니라 점수 단위. 방향이 반대로 나오면"
                                "(PR-AUC < 기저율) 그대로 둠 - 실험기록 비고에 기저율과 함께 적음"),
            ("AE 실험", "S_AEt·S_VAEt·S_AEs·S_VAEs = 점수 단독, A3_AEt·A3_VAEt·A3_AEs·A3_VAEs = A3(이력 9칸, k*) + 점수 1칸, "
                       "A3_*_p = A3 + 위약 점수 1칸. 시드 42·43·44, 단계 'ae'"),
            ("AE 점수 저장", RULE_CACHE)]


# ---------------------------------------------------------------- 재현성
@contextmanager
def deterministic_torch():
    """점수 계산 동안만 결정적 알고리즘을 켜고 끝나면 되돌린다(다른 실험에 영향 없게)."""
    prev = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev[2], prev[3]


# ---------------------------------------------------------------- 행 나누기
def fold_assign(n, seed):
    """정상 행 n 개의 겹 번호(0..n_folds-1): default_rng([seed, 1]) 순열 위치 % n_folds (겹 크기 차이 1 이하)."""
    f = np.empty(n, dtype=np.int64)
    f[np.random.default_rng([seed, 1]).permutation(n)] = np.arange(n) % AEC["n_folds"]
    return f


def fold_assign_all(n, seed):
    """학습 행 전부(검출·불검출) n 개의 겹 번호(0..n_folds-1): default_rng([seed, 3]) 순열 위치 % n_folds. 정답과 무관(y 를 쓰지 않음)."""
    f = np.empty(n, dtype=np.int64)
    f[np.random.default_rng([seed, 3]).permutation(n)] = np.arange(n) % AEC["n_folds"]
    return f


def es_split(rows, seed, mi):
    """모델 행 -> (학습 행, 조기 종료용 행) 둘 다 오름차순. 조기 종료용 = default_rng([seed, 2, mi]) 순열 앞 es_frac."""
    p = np.random.default_rng([seed, 2, mi]).permutation(len(rows))
    n_es = int(round(len(rows) * AEC["es_frac"]))
    return np.sort(rows[p[n_es:]]), np.sort(rows[p[:n_es]])


def a3_feats(k):
    """표 AE 입력 = A3 숫자 칸(A3 와 같은 FeatureBuilder, AE 칸 없음)."""
    f = dict(E.EXP_BY_ID["A3"]["feats"])
    f["k"] = k
    return f


# ---------------------------------------------------------------- 입력: 표
class TabInput:
    """표 AE 입력. 학습 행 숫자 칸 = A3 FeatureBuilder(그 시드의 h), 범주 = ctx.sp[sp]['cats'](전체 기준 고정 목록 번호).
    feats(단계 선택 ae4) 를 주면 숫자 칸 = 그 입력의 FeatureBuilder, 범주 = 기본 4칸 + 그 입력의 추가 범주(aw_label 등: 전체 행 값 목록 정렬 순 번호)."""

    def __init__(self, ctx, k, htr, feats=None):
        from e2models import FeatureBuilder
        t0 = time.time()
        self.ctx = ctx
        self.fb = FeatureBuilder(ctx, a3_feats(k) if feats is None else dict(feats))
        if feats is None and len(self.fb.names) != N_TAB:
            raise RuntimeError(f"표 AE 입력 칸 수 {len(self.fb.names)} != {N_TAB}")
        self.n_num = len(self.fb.names)
        self.X = self.fb.num("train", np.arange(len(ctx.sp["train"]["idx"])), htr)
        self._codes = None
        extra = [c for c in self.fb.cat_names if c not in C1.CAT_COLS]
        if feats is not None and extra:
            j = [self.fb.cat_names.index(c) for c in extra]
            strs = {sp: np.asarray(self.fb.cats(sp, np.arange(len(ctx.sp[sp]["idx"]))))[:, j].astype(str) for sp in C1.SPLITS}
            levels = [np.unique(np.concatenate([strs[sp][:, q] for sp in C1.SPLITS])) for q in range(len(extra))]
            self._codes = {sp: np.ascontiguousarray(np.hstack([ctx.sp[sp]["cats"].astype(np.int64)] +
                                                              [np.searchsorted(levels[q], strs[sp][:, q]).astype(np.int64)[:, None]
                                                               for q in range(len(extra))])) for sp in C1.SPLITS}
        self.extra_cats = extra if self._codes is not None else []
        self.seconds = time.time() - t0

    def codes(self, sp):
        """범주 번호 (n, 4[+추가])."""
        return self.ctx.sp[sp]["cats"] if self._codes is None else self._codes[sp]

    @staticmethod
    def prep(X, cats):
        """정규화 통계·어휘(그 모델 행으로). X (n, N_TAB) float32, cats (n, 4) int."""
        Xd = X.astype(np.float64)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(Xd, axis=0)
            sd = np.nanstd(Xd, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
        vocab = [np.unique(cats[:, j]) for j in range(cats.shape[1])]
        return {"mu": mu, "sd": sd, "vocab": vocab}

    @staticmethod
    def z(X, prep):
        Z = (X.astype(np.float64) - prep["mu"]) / prep["sd"]
        Z[np.isnan(Z)] = 0.0
        return Z.astype(np.float32)

    @staticmethod
    def cat_idx(cats, prep):
        """(n, 4) int64: 그 모델 행에 있던 범주 1..n(어휘 순서), 없던 범주 0."""
        out = np.zeros(cats.shape, dtype=np.int64)
        for j, voc in enumerate(prep["vocab"]):
            i = np.clip(np.searchsorted(voc, cats[:, j]), 0, len(voc) - 1)
            out[:, j] = np.where(voc[i] == cats[:, j], i + 1, 0)
        return out

    def model_prep(self, rows):
        return self.prep(self.X[rows], self.codes("train")[rows])

    def cat_sizes(self, prep):
        return [len(v) for v in prep["vocab"]]

    def device_train(self, prep, dev):
        return {"Z": torch.from_numpy(self.z(self.X, prep)).to(dev),
                "C": torch.from_numpy(self.cat_idx(self.codes("train"), prep)).to(dev)}

    @staticmethod
    def batch(dd, idx):
        return dd["Z"][idx], dd["C"][idx]

    def device_eval(self, prep, dev, sp, h, C_cache):
        """검증·시험 h 하나: 숫자 칸 A3 FeatureBuilder(sp, 전체 행, h)."""
        n = len(self.ctx.sp[sp]["idx"])
        if sp not in C_cache:
            C_cache[sp] = torch.from_numpy(self.cat_idx(self.codes(sp), prep)).to(dev)
        X = self.fb.num(sp, np.arange(n), h)
        return {"Z": torch.from_numpy(self.z(X, prep)).to(dev), "C": C_cache[sp]}


# ---------------------------------------------------------------- 입력: 시퀀스
def window_weight(S, D, s, T, L):
    """(S, D) float64: 칸 (시군구, 날짜) 가 행들의 창 T−L … T−1 에 들어간 횟수(차분 배열 + 누적합)."""
    dC = np.zeros((S, D + 1), dtype=np.int64)
    np.add.at(dC, (s, T - L), 1)
    np.add.at(dC, (s, T), -1)
    return np.cumsum(dC[:, :D], axis=1).astype(np.float64)


def seq_stats(W, s, T, L):
    """창 겹침 수로 가중한 변수별 평균·표준편차(NaN 무시, ddof=0), float64 (V,) 둘."""
    S, D, V = W.shape
    Cw = window_weight(S, D, s, T, L)
    mu, sd = np.empty(V), np.empty(V)
    for v in range(V):
        x = W[:, :, v].astype(np.float64)
        ok = ~np.isnan(x)
        w = np.where(ok, Cw, 0.0)
        x0 = np.where(ok, x, 0.0)
        tw = w.sum()
        mu[v] = (w * x0).sum() / tw
        sd[v] = np.sqrt((w * (x0 - mu[v]) ** 2).sum() / tw)
    return mu, sd


def seq_table(W, mu32, sd32):
    """build_seq(1차 LSTM 과 같은 식)를 그대로 호출 - 표준화 통계만 바꿔 넣는다(ctx 대신 W·통계만 가진 묶음)."""
    from e2dl_impl import build_seq
    shim = SimpleNamespace(weather=SimpleNamespace(W=W, train_period_stats=lambda: (mu32, sd32)))
    seq, _, _ = build_seq(shim)
    return seq


class SeqInput:
    """시퀀스 AE 입력: 창 T−L … T−1 (T = d − h), (B, 38, L) 로 넘김(위치 0 = T−L, L−1 = T−1)."""

    def __init__(self, ctx, htr):
        self.ctx = ctx
        self.L = AEC["seq"]["L"]
        v = ctx.sp["train"]
        self.s_tr = v["s"].astype(np.int64)
        self.T_tr = v["d"].astype(np.int64) - np.asarray(htr, dtype=np.int64)
        self.seconds = 0.0

    def model_prep(self, rows):
        mu, sd = seq_stats(self.ctx.weather.W, self.s_tr[rows], self.T_tr[rows], self.L)
        return {"mu64": mu, "sd64": sd, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}

    def table(self, prep, dev):
        return torch.from_numpy(seq_table(self.ctx.weather.W, prep["mu"], prep["sd"])).to(dev)

    def device_train(self, prep, dev, table=None):
        tb = self.table(prep, dev) if table is None else table
        return {"tb": tb, "s": torch.from_numpy(self.s_tr).to(dev), "T": torch.from_numpy(self.T_tr).to(dev),
                "ar": torch.arange(self.L, device=dev)}

    @staticmethod
    def batch(dd, idx):
        s, T, ar = dd["s"][idx], dd["T"][idx], dd["ar"]
        x = dd["tb"][s[:, None], T[:, None] - ar.numel() + ar[None, :]]          # (B, L, 38)
        return x.permute(0, 2, 1).contiguous(), None

    def device_eval(self, dd_train, sp, h):
        v = self.ctx.sp[sp]
        dev = dd_train["tb"].device
        T = v["d"].astype(np.int64) - int(h)
        return {"tb": dd_train["tb"], "s": torch.from_numpy(v["s"].astype(np.int64)).to(dev),
                "T": torch.from_numpy(T).to(dev), "ar": dd_train["ar"]}


# ---------------------------------------------------------------- 신경망
class TabNet(nn.Module):
    def __init__(self, n_num, cat_sizes, variational):
        super().__init__()
        c = AEC["tab"]
        h1, h2 = c["hidden"]
        self.variational = variational
        self.embs = nn.ModuleList([nn.Embedding(n + 1, c["emb_dim"], padding_idx=0) for n in cat_sizes])
        self.enc = nn.Sequential(nn.Linear(n_num + c["emb_dim"] * len(cat_sizes), h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        if variational:
            self.mu = nn.Linear(h2, c["latent"])
            self.lv = nn.Linear(h2, c["latent"])
        else:
            self.z = nn.Linear(h2, c["latent"])
        self.dec = nn.Sequential(nn.Linear(c["latent"], h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, n_num))

    def forward(self, x, c, sample=False):
        h = self.enc(torch.cat([x] + [e(c[:, j]) for j, e in enumerate(self.embs)], dim=1))
        if not self.variational:
            return self.dec(self.z(h)), None, None
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if sample else mu
        return self.dec(z), mu, lv


class SeqNet(nn.Module):
    """시간 축 하나의 1D 합성곱 오토인코더. 계산은 nn.Conv2d·ConvTranspose2d 커널 (k, 1)·보폭 (s, 1)·패딩 (p, 0) 에 입력 (B, C, L, 1) 꼴
    - Conv1d 와 같은 계산·같은 기본 초기값(fan_in = C x k). 1D 꼴로 두면 이 GPU(cuDNN, 결정적 설정)가 가중치 기울기에 FFT 알고리즘을 골라
    배치마다 약 26ms·작업 공간 약 2.5GB 를 써서(2D (k,1) 꼴은 약 4ms·0.07GB, scratch/ae/prof_seq4.py) 이 꼴로 계산한다."""

    def __init__(self, n_ch, variational):
        super().__init__()
        c = AEC["seq"]
        f1, f2 = c["filters"]
        k, st = (c["kernel"], 1), (c["stride"], 1)
        p = (c["kernel"] // 2, 0)
        self.variational = variational
        self.c1 = nn.Conv2d(n_ch, f1, k, stride=st, padding=p)
        self.drop1 = nn.Dropout(c["dropout"])
        if variational:
            self.c2_mu = nn.Conv2d(f1, f2, k, stride=st, padding=p)
            self.c2_lv = nn.Conv2d(f1, f2, k, stride=st, padding=p)
        else:
            self.c2 = nn.Conv2d(f1, f2, k, stride=st, padding=p)
        self.d1 = nn.ConvTranspose2d(f2, f2, k, stride=st, padding=p)
        self.drop2 = nn.Dropout(c["dropout"])
        self.d2 = nn.ConvTranspose2d(f2, f1, k, stride=st, padding=p)
        self.out = nn.ConvTranspose2d(f1, n_ch, k, stride=(1, 1), padding=p)

    def forward(self, x, c=None, sample=False):
        x4 = x.unsqueeze(-1)                                   # (B, 38, L) -> (B, 38, L, 1)
        L0 = x4.shape[2]
        h = self.drop1(torch.relu(self.c1(x4)))
        L1 = h.shape[2]
        if self.variational:
            mu, lv = self.c2_mu(h), self.c2_lv(h)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if sample else mu
        else:
            mu = lv = None
            z = torch.relu(self.c2(h))
        u = self.drop2(torch.relu(self.d1(z, output_size=[L1, 1])))
        u = torch.relu(self.d2(u, output_size=[L0, 1]))
        return self.out(u).squeeze(-1), mu, lv


def make_net(kind, n_num=None, cat_sizes=None):
    if is_tab(kind):
        return TabNet(n_num, cat_sizes, is_var(kind))
    return SeqNet(N_CH, is_var(kind))


def row_score(net, x, c, sample=False):
    """행마다 점수 (B,): AE = 복원 오차 MSE, VAE = 0.5·Σ 복원 오차² + KL."""
    rec, mu, lv = net(x, c, sample)
    err = (rec - x).pow(2).flatten(1)
    if mu is None:
        return err.mean(1)
    kl = 0.5 * (mu.pow(2) + lv.exp() - 1.0 - lv).flatten(1).sum(1)
    return 0.5 * err.sum(1) + kl


def score_rows(net, batch_fn, dd, rows, dev):
    """rows (numpy 행 번호) 의 점수 float32 (평가 모드, z = 잠재 평균, eval_batch 씩)."""
    net.eval()
    out = np.empty(len(rows), dtype=np.float32)
    r = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(dev)
    with torch.no_grad():
        for a in range(0, len(rows), AEC["eval_batch"]):
            x, c = batch_fn(dd, r[a:a + AEC["eval_batch"]])
            out[a:a + len(x)] = row_score(net, x, c).float().cpu().numpy()
    return out


def fit_net(kind, net_args, batch_fn, dd, fit_rows, es_rows, model_seed, dev, log, tag):
    """학습(조기 종료 = es_rows 평균 점수, 인내, 최고 에폭 복원). 돌려줌 (net, 정보)."""
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    net = make_net(kind, *net_args).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=AEC["lr"])
    g = torch.Generator().manual_seed(model_seed)
    fr = torch.from_numpy(np.asarray(fit_rows, dtype=np.int64)).to(dev)
    n = len(fit_rows)
    best, best_ep, wait, best_state, hist = float("inf"), 0, 0, None, []
    t0 = time.time()
    for ep in range(1, AEC["max_epochs"] + 1):
        te = time.time()
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot, nb = 0.0, 0
        for a in range(0, n, AEC["batch"]):
            x, c = batch_fn(dd, fr[perm[a:a + AEC["batch"]]])
            loss = row_score(net, x, c, sample=True).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        es = float(np.mean(score_rows(net, batch_fn, dd, es_rows, dev), dtype=np.float64))
        if not (np.isfinite(tot) and np.isfinite(es)):
            raise RuntimeError(f"AE {tag}: 손실이 유한하지 않음(에폭 {ep}, 학습 손실 합 {tot}, 조기 종료 점수 {es})")
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "es_score": es, "seconds": time.time() - te})
        if es < best:
            best, best_ep, wait = es, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= AEC["patience"]:
                break
    net.load_state_dict(best_state)
    info = {"model_seed": model_seed, "n_fit": int(n), "n_es": int(len(es_rows)), "best_epoch": best_ep, "epochs": len(hist),
            "best_es_score": best, "seconds": time.time() - t0, "n_params": int(sum(p.numel() for p in net.parameters())),
            "history": hist}
    log(f"      {tag}: 학습 {n:,}행·조기종료 {len(es_rows):,}행, 최고 에폭 {best_ep}/{len(hist)}, 조기종료 평균 점수 {best:.5f}, {info['seconds']:.1f}s")
    return net, info, best_state


# ---------------------------------------------------------------- 점수 계산
def _md5(*arrs):
    m = hashlib.md5()
    for a in arrs:
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()


def code_md5():
    src2 = Path(__file__).resolve().parent
    d = {f: hashlib.md5((src2 / f).read_bytes()).hexdigest() for f in CODE_FILES2}
    d.update({f"1차/{f}": hashlib.md5((E.EXP1_SRC / f).read_bytes()).hexdigest() for f in CODE_FILES1})
    return d


def meta(ctx, kind, seed, k, tab_feats=None):
    hf = ctx.h_fit(seed)
    tf = tab_feats if is_tab(kind) else None
    extra = {"tab_feats": tf, "tab_sig": tab_sig(tf)} if tf is not None else {}
    return {"kind": kind, "seed": int(seed), "k": k, "smoke": bool(ctx.smoke), "L": E.L, "config": AEC, "score_rule": SCORE_RULE,
            "a3_feats": a3_feats(k) if is_tab(kind) and tf is None else None, **extra,
            "rows_md5": {sp: _md5(v["s"], v["d"], v["hd"], v["y"], v["cats"]) for sp, v in ctx.sp.items()},
            "h_md5": _md5(hf["train"], hf["val"]), "weather_shape": list(ctx.weather.W.shape),
            "weather_date0": str(ctx.weather.date0), "code_md5": code_md5(), "torch": torch.__version__}


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return None
    q = np.percentile(x, [10, 50, 90])
    return {"n": int(len(x)), "mean": float(x.mean()), "p10": float(q[0]), "median": float(q[1]), "p90": float(q[2])}


def compute_scores(ctx, kind, seed, k, log=None, debug=False, tab_feats=None):
    """(kind, seed) 점수 계산. 돌려줌 (S, info[, dbg]): S = {'train': (n_tr,), 'val': (60, n_va), 'test': (60, n_te)} float32.
    debug=True 면 dbg = 모델마다 행 번호·정규화·가중치(점검용)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가 - AE 점수는 GPU 로 계산(재현성 설정 포함)")
    log = log or (lambda *_: None)
    t0 = time.time()
    dev = torch.device("cuda")
    tab = is_tab(kind)
    y = ctx.sp["train"]["y"]
    neg, pos1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    htr = ctx.h_fit(seed)["train"].astype(np.int64)
    fold = fold_assign_all(len(y), seed)                        # 학습 행 전부(검출 포함) 겹 배정 - 정답과 무관
    inp = (TabInput(ctx, k, htr) if tab_feats is None else TabInput(ctx, k, htr, feats=tab_feats)) if tab else SeqInput(ctx, htr)
    batch_fn = inp.batch
    log(f"    AE {kind} 시드 {seed}: 학습 구간 정상 {len(neg):,}행 · 검출 {len(pos1):,}행, 학습 행 전부 {AEC['n_folds']}겹(점수 규칙 v2), "
        f"입력 준비 {inp.seconds:.1f}s")
    if _CUDA_INIT_BEFORE:
        log("    [주의] 이 프로세스에서 AE 전에 CUDA 를 이미 써서 CUBLAS_WORKSPACE_CONFIG 가 늦게 들어감 - 점수 재현성이 보장되지 않을 수 있음"
            "(-Stage ae 를 따로 실행 권장)")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    S = {"train": np.full(len(y), np.nan, dtype=np.float32)}
    models, dbg = {}, {"fold_all": fold, "neg": neg, "pos": pos1, "models": {}}
    kept = []
    t_eval = 0.0
    with deterministic_torch():
        for mi in range(AEC["n_folds"]):
            tag = f"겹 {mi} 모델"
            rows = neg[fold[neg] != mi]                          # 학습 = 겹 mi 밖 학습 구간 불검출 행
            target = np.flatnonzero(fold == mi)                  # 점수 = 겹 mi 의 학습 행 전부(검출 포함)
            if not np.all(y[rows] == 0) or np.intersect1d(rows, target).size:
                raise RuntimeError(f"AE {tag}: 학습 행 검사 실패")
            fit_rows, es_rows = es_split(rows, seed, mi)
            prep = inp.model_prep(rows)
            dd = inp.device_train(prep, dev)
            net_args = (inp.n_num, inp.cat_sizes(prep)) if tab else ()
            net, minfo, state = fit_net(kind, net_args, batch_fn, dd, fit_rows, es_rows, seed * 10 + mi, dev, log, tag)
            S["train"][target] = score_rows(net, batch_fn, dd, target, dev)
            minfo.update({"rows": int(len(rows)), "target_rows": int(len(target)), "target_pos": int(y[target].sum()),
                          "target": f"겹 {mi} 학습 행 전부(검출 {int(y[target].sum())} 포함)",
                          "prep": ({"vocab_sizes": inp.cat_sizes(prep), "mu_first5": prep["mu"][:5].tolist(),
                                    "sd_first5": prep["sd"][:5].tolist()} if tab else
                                   {"mu": prep["mu64"].tolist(), "sd": prep["sd64"].tolist()})})
            models[str(mi)] = minfo
            kept.append((prep, state, net_args))
            if debug:
                dbg["models"][mi] = {"rows": rows, "fit": fit_rows, "es": es_rows, "target": target,
                                     "prep": prep, "state": state, "net_args": net_args}
            del net, dd
            torch.cuda.empty_cache()
        te = time.time()                                         # 검증·시험: h 1..60 전부, 겹 모델 5개 점수 평균
        nets = []
        for prep, state, net_args in kept:
            nm = make_net(kind, *net_args).to(dev)
            nm.load_state_dict(state)
            nets.append(nm)
        for sp in ("val", "test"):
            n = len(ctx.sp[sp]["idx"])
            acc = np.zeros((E.H_MAX, n), dtype=np.float64)
            if tab:                                              # 입력(숫자 칸)은 h 마다 한 번 만들고 모델마다 자기 정규화로
                Cm = [torch.from_numpy(inp.cat_idx(inp.codes(sp), kp[0])).to(dev) for kp in kept]
                for h in range(1, E.H_MAX + 1):
                    Xh = inp.fb.num(sp, np.arange(n), h)
                    for m, (prep, _, _) in enumerate(kept):
                        de = {"Z": torch.from_numpy(inp.z(Xh, prep)).to(dev), "C": Cm[m]}
                        acc[h - 1] += score_rows(nets[m], batch_fn, de, np.arange(n), dev)
                        del de
                del Cm
            else:                                                # 시퀀스: 모델마다 자기 정규화 표
                for m, (prep, _, _) in enumerate(kept):
                    dtr = inp.device_train(prep, dev)
                    for h in range(1, E.H_MAX + 1):
                        de = inp.device_eval(dtr, sp, h)
                        acc[h - 1] += score_rows(nets[m], batch_fn, de, np.arange(n), dev)
                        del de
                    del dtr
                    torch.cuda.empty_cache()
            S[sp] = (acc / len(kept)).astype(np.float32)
        t_eval = time.time() - te
        del nets
        torch.cuda.empty_cache()
    if np.isnan(S["train"]).any():
        raise RuntimeError(f"AE {kind}: 점수 없는 학습 행 {int(np.isnan(S['train']).sum())}")
    from sklearn.metrics import average_precision_score, roc_auc_score
    hv = ctx.h_fit(seed)["val"].astype(np.int64)
    yv = ctx.sp["val"]["y"]
    info = {"kind": kind, "seed": int(seed), "k": k,
            "input": input_text(kind) if tab_feats is None or not tab else input_text(kind, tab_feats),
            "arch": ARCH_TEXT["tab" if tab else "seq"] if tab_feats is None or not tab else arch_tab(inp.n_num, len(inp.cat_sizes(prep))),
            "variational": is_var(kind), "n_normal_train": int(len(neg)), "n_pos_train": int(len(pos1)),
            "score_rule": SCORE_RULE, "fold_sizes": np.bincount(fold, minlength=AEC["n_folds"]).tolist(),
            "fold_label_table": [[int(((fold == f) & (y == 0)).sum()), int(((fold == f) & (y == 1)).sum())] for f in range(AEC["n_folds"])],
            "models": models,
            "input_seconds": inp.seconds, "eval_seconds": t_eval, "seconds_total": time.time() - t0,
            "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "gpu_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "deterministic": "use_deterministic_algorithms(True), cudnn.deterministic=True, benchmark=False, "
                                                         f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
                                                         + (" [주의: CUDA 를 이 모듈 전에 이미 씀 - 재현성 보장 안 될 수 있음]" if _CUDA_INIT_BEFORE else ""),
            "score_stats": {"train_neg_crossfit": _stats(S["train"][neg]), "train_pos_crossfit": _stats(S["train"][pos1]),
                            **{f"{sp}_y{c}_all_h": _stats(S[sp][:, ctx.sp[sp]["y"] == c].ravel()) for sp in ("val", "test") for c in (0, 1)}},
            "record_only": {"train_roc_auc": float(roc_auc_score(y, S["train"])), "train_ap": float(average_precision_score(y, S["train"])),
                            "train_base_rate": float(y.mean()),
                            "val_ap_at_es_h": float(average_precision_score(yv, S["val"][hv - 1, np.arange(len(yv))])),
                            "val_base_rate": float(yv.mean()),
                            "note": "기록만(선정·방향 결정에 쓰지 않음). 학습 행 점수 = 겹 모델(검출·불검출 같은 방식), 검증 = 겹 모델 5개 평균"}}
    log(f"    AE {kind} 시드 {seed} 끝: {info['seconds_total']:.0f}s (검증·시험 점수 {t_eval:.0f}s), GPU 최대 할당 "
        f"{info['gpu_peak_allocated_gb']:.2f} GB, 기록용 학습 ROC-AUC {info['record_only']['train_roc_auc']:.4f}")
    return (S, info, dbg) if debug else (S, info)


# ---------------------------------------------------------------- 캐시
def _cache_files(ctx, kind, seed, tab_feats=None):
    d = getattr(ctx, "ae_dir", None)
    if d is None:
        return None, None
    d = Path(d)
    stem = f"{kind}_s{seed}_{'smoke' if ctx.smoke else 'full'}"
    if tab_feats is not None and is_tab(kind):                 # 표 AE 입력이 A3 가 아니면 입력 구분값을 붙임(A3 점수와 섞이지 않게)
        stem += f"_t{tab_sig(tab_feats)}"
    stem += f"_{SCORE_RULE_TAG}"                                 # 점수 규칙 v2(옛 v1 파일과 섞이지 않게)
    return d / f"{stem}.npz", d / f"{stem}.json"


def get_scores(ctx, kind, seed, k, log=None, tab_feats=None):
    """(S, info, 출처) - ctx 메모리 -> 디스크(메타 같을 때) -> 새로 계산(디스크에 저장) 순서.
    tab_feats: 표 AE 입력(없으면 A3). 시퀀스 AE·VAE 는 표 입력과 무관하므로 무시(같은 점수를 같이 씀)."""
    log = log or getattr(ctx, "log", None) or (lambda *_: None)
    mem = ctx.__dict__.setdefault("_ae_scores", {})
    tf = tab_feats if is_tab(kind) else None
    key = (kind, int(seed), k) if tf is None else (kind, int(seed), k, tab_sig(tf))
    if key in mem:
        return mem[key]
    f, mf = _cache_files(ctx, kind, seed, tf)
    want = meta(ctx, kind, seed, k, tf)
    got = None
    if f is not None and f.exists() and mf.exists():
        try:
            j = json.loads(mf.read_text(encoding="utf-8"))
            if j.get("meta") == json.loads(json.dumps(want, default=str)):
                z = np.load(f)
                S = {sp: z[sp] for sp in ("train", "val", "test")}
                ok = (S["train"].shape == (len(ctx.sp["train"]["idx"]),)
                      and all(S[sp].shape == (E.H_MAX, len(ctx.sp[sp]["idx"])) for sp in ("val", "test")))
                if ok:
                    got = (S, j["info"], f"디스크에서 읽음: {f}")
                    log(f"    AE {kind} 시드 {seed}: 저장된 점수 읽음 ({f.name})")
            else:
                log(f"    AE {kind} 시드 {seed}: 저장된 점수의 메타가 지금과 다름 -> 새로 계산")
        except Exception as e:
            log(f"    AE {kind} 시드 {seed}: 저장된 점수를 읽지 못함({e}) -> 새로 계산")
    if got is None:
        S, info = compute_scores(ctx, kind, seed, k, log) if tf is None else compute_scores(ctx, kind, seed, k, log, tab_feats=tf)
        src = "새로 계산"
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.stem + f".tmp{os.getpid()}.npz")
            np.savez(tmp, **S)
            os.replace(tmp, f)
            tmpj = mf.with_name(mf.stem + f".tmp{os.getpid()}.json")
            tmpj.write_text(json.dumps({"meta": want, "info": info}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            os.replace(tmpj, mf)
            src += f" -> 저장 {f}"
        got = (S, info, src)
    while len(mem) >= 4:                                       # 메모리: 최근 4개만(나머지는 디스크에서 다시 읽음)
        mem.pop(next(iter(mem)))
    mem[key] = got
    return got


def calendar_year(ctx, t):
    """기상표 날짜 번호 -> 달력 연도."""
    return (ctx.weather.date0 + np.asarray(t, dtype=np.int64).astype("timedelta64[D]")).astype("datetime64[Y]").astype(np.int64) + 1970


def _shuffle_within(x, g, rng):
    out = x.copy()
    for yv in np.unique(g):                                    # 연도 오름차순
        idx = np.flatnonzero(g == yv)
        out[idx] = rng.permutation(x[idx])
    return out


def placebo_scores(ctx, S, seed):
    """위약판: 분할마다 예측일 T 의 달력 연도 안에서 행끼리 섞음(default_rng(placebo_seed) 하나로 학습 -> 검증 h1..60 -> 시험 h1..60)."""
    rng = np.random.default_rng(AEC["placebo_seed"])
    ht = ctx.h_fit(seed)["train"].astype(np.int64)
    out = {"train": _shuffle_within(S["train"], calendar_year(ctx, ctx.sp["train"]["d"].astype(np.int64) - ht), rng)}
    for sp in ("val", "test"):
        d = ctx.sp[sp]["d"].astype(np.int64)
        M = np.empty_like(S[sp])
        for h in range(1, E.H_MAX + 1):
            M[h - 1] = _shuffle_within(S[sp][h - 1], calendar_year(ctx, d - h), rng)
        out[sp] = M
    return out


def get_placebo(ctx, kind, seed, k, tab_feats=None):
    S, _, _ = get_scores(ctx, kind, seed, k) if tab_feats is None else get_scores(ctx, kind, seed, k, tab_feats=tab_feats)
    pm = ctx.__dict__.setdefault("_ae_placebo", {})
    key = (kind, int(seed), k, tab_sig(tab_feats if is_tab(kind) else None))
    if key not in pm:
        pm.clear()                                             # 위약판은 다시 만들기 쉬움 -> 마지막 1개만
        pm[key] = placebo_scores(ctx, S, seed)
    return pm[key]


# ---------------------------------------------------------------- 특성 칸·점수 단독 실행기
def ae_features(ctx, sp, pos, h, blocks, seed, k, tab_feats=None):
    """[AE 점수 칸] float32 (n, len(blocks)). 학습 행은 그 시드의 h(ctx.h_fit(seed)['train'])에서만 점수가 있음.
    tab_feats: 표 AE 입력(단계 선택 ae4, 없으면 A3 - 이전과 같음)."""
    if seed is None:
        raise ValueError("AE 점수 칸: 시드가 없음(FeatureBuilder(seed=...) 필요)")
    pos = np.asarray(pos)
    out = np.empty((len(pos), len(blocks)), dtype=np.float32)
    for j, b in enumerate(blocks):
        kind, plc = parse_block(b)
        if tab_feats is None:
            S = get_placebo(ctx, kind, seed, k) if plc else get_scores(ctx, kind, seed, k)[0]
        else:
            S = get_placebo(ctx, kind, seed, k, tab_feats) if plc else get_scores(ctx, kind, seed, k, tab_feats=tab_feats)[0]
        if sp == "train":
            hs = ctx.h_fit(seed)["train"][pos].astype(np.int64)
            hh = np.full(len(pos), int(h), dtype=np.int64) if np.isscalar(h) else np.asarray(h, dtype=np.int64)
            if not np.array_equal(hh, hs):
                raise ValueError("AE 점수 칸: 학습 행 점수는 그 시드의 h(ctx.h_fit(seed)['train'])에서만 계산됨 - 다른 h 로 요청됨")
            out[:, j] = S["train"][pos]
        elif np.isscalar(h):
            out[:, j] = S[sp][int(h) - 1, pos]
        else:
            out[:, j] = S[sp][np.asarray(h, dtype=np.int64) - 1, pos]
    return out


class AEScoreRunner:
    """점수 단독(S_*): 이상 점수를 그대로 예측값 p 로(클수록 검출). CatBoost 없음. run.py 실행기 약속(fit·predict·info·close) 그대로."""

    def __init__(self, ctx, exp, seed, feats, log, smoke):
        self.ctx, self.exp, self.seed, self.feats, self.log = ctx, exp, seed, feats, log
        blocks = feats["ae"]
        if len(blocks) != 1:
            raise ValueError(f"점수 단독 실험은 묶음 1개: {blocks}")
        self.kind, plc = parse_block(blocks[0])
        if plc:
            raise ValueError("점수 단독 실험에 위약판 없음")
        self.pos = {sp: np.arange(len(ctx.sp[sp]["idx"])) for sp in C1.SPLITS}
        self.cols = ae_names([self.kind])
        self.n_features = (N_TAB + len(C1.CAT_COLS)) if is_tab(self.kind) else E.L * N_CH
        self.n_features_desc = score_only_desc(self.kind)
        self.feature_seconds = 0.0

    def fit(self):
        t0 = time.time()
        self.S, self.ae_info, self.source = get_scores(self.ctx, self.kind, self.seed, self.feats["k"], self.log)
        self.fit_seconds = time.time() - t0
        ms = list(self.ae_info["models"].values())               # 점수 규칙 v2: 겹 모델 5개(전체 정상 행 모델 없음)
        self.best_rounds = int(round(float(np.mean([m["best_epoch"] for m in ms]))))
        self.n_rounds_trained = int(max(m["epochs"] for m in ms))
        self.n_train_rows, self.n_train_pos = int(self.ae_info["n_normal_train"]), 0
        from sklearn.metrics import average_precision_score
        hv = self.ctx.h_fit(self.seed)["val"].astype(np.int64)
        yv = self.ctx.sp["val"]["y"]
        self.best_val_ap = float(average_precision_score(yv, self.S["val"][hv - 1, np.arange(len(yv))]))
        self.log(f"    점수 단독 {self.kind}: {self.source}, {self.fit_seconds:.1f}s")

    def predict(self, sp, h):
        return np.ascontiguousarray(self.S[sp][int(h) - 1][self.pos[sp]], dtype=np.float32)

    def info(self):
        return {"score_as_prediction": "예측값 p = 이상 점수 그대로(확률 아님, 클수록 검출 쪽). PR-AUC·ROC-AUC 는 순위만, 판정 기준값은 점수 단위",
                "best_val_ap_note": "조기 종료용 검증 h(default_rng(seed+1)) 배정의 점수 AP - 기록만(AE 조기 종료는 학습 구간 정상 행으로 함)",
                "n_train_rows_note": "n_train_rows = 학습 구간 불검출 행 수(겹 모델마다 자기 겹 밖 행으로 학습), 최적 에폭 = 겹 모델 최고 에폭 평균, "
                                     "학습한 에폭 = 겹 모델 최대. 검출 행은 학습에 안 씀(점수 규칙 v2)",
                "source": self.source, "rules": ae_info([self.kind]), "ae": self.ae_info}

    def close(self):
        pass
