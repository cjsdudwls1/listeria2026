# -*- coding: utf-8 -*-
"""2차(고도화) 실험 설정. 사용자 승인 사양 그대로. 값을 바꾸면 결과가 달라지므로 임의로 고치지 말 것.

1차 코드(1_베이스라인_lims기상/src)는 고치지 않고 import 해서 쓴다(sys.path 뒤쪽에 추가).
2차 모듈 이름은 모두 e2 로 시작해 1차 모듈(config, data, features, metrics, models, common)과 겹치지 않게 했다.
"""
import os
import sys
from pathlib import Path

EXP2_DIR = Path(__file__).resolve().parents[1]
EXP1_DIR = EXP2_DIR.parent / "1_베이스라인_lims기상"
EXP1_SRC = EXP1_DIR / "src"
EXP1_RESULTS = EXP1_DIR / "results"
if str(EXP1_SRC) not in sys.path:
    sys.path.append(str(EXP1_SRC))          # 2차 src 가 앞, 1차 src 가 뒤

import config as C1                          # noqa: E402  (1차 설정: 자료 경로, 분할, 기상 변수, 범주, H_MAX, 트리 라운드)

# ---------------------------------------------------------------- 공통
# 기상 창 길이(사용자 결정 2026-10-19: 2차 실험을 100일 위에서 다시 - 환경변수 EXP2_L, 없으면 50 = 이전과 같음).
# 요약 = 10일 구간 평균 L/10 + 평균·표준편차·최소·최대·기울기 5. 기상 창을 쓰는 곳(기상 요약·이동평균 차감·생장식·생장 범위·SPI 창 요약,
# DL 시퀀스 DL["L"], 시퀀스 AE·VAE 입력 AE["seq"]["L"], 창 확인, 평가 캐시 이름 eval_wsum_L{L})이 모두 이 값을 따른다.
L_DEFAULT = 50
L = int(os.environ.get("EXP2_L", "") or L_DEFAULT)
if L < 10 or L % C1.BLOCK:
    raise ValueError(f"EXP2_L={L}: 10 이상이고 {C1.BLOCK} 의 배수여야 함(10일 구간 요약·최근 10일 칸)")
NW = len(C1.WEATHER_VARS) * (L // C1.BLOCK + 5)   # 기상 요약 칸 수 (L=50 -> 190, L=100 -> 285)
NSPI = 2 * (L // C1.BLOCK + 5)                    # SPI3·SPI6 창 요약 칸 수 (L=50 -> 20)
# 고르는 규칙(사용자 확정 원칙 2026-10-19): 환경변수 EXP2_SELECT = old(기본, 이전과 같음: 검증 구간 가중 PR-AUC) | tint
#   tint = 검증 T 통합 60일평균 PR-AUC: 예측하는 날 T = d − h 가 VAL_START ~ VAL_END − H_MAX 인 저장 검증 예측 행만, h 마다 PR-AUC(average precision)
#          -> h 1~60 단순 평균 -> 시드 평균. k*(select_k)·A 최고(select_A)·F·G 조합 선정(e2fg.combo_experiments)에 적용. 계산 = e2eval.val_tint_pr_auc
SELECT = (os.environ.get("EXP2_SELECT", "") or "old").strip().lower()
if SELECT not in ("old", "tint"):
    raise ValueError(f"EXP2_SELECT={SELECT}: old 또는 tint")
SEEDS = [42, 43, 44]
H_MAX = C1.H_MAX                             # 60
MAX_ROUNDS = C1.MAX_ROUNDS                   # 2000
PATIENCE = C1.PATIENCE                       # 100
# 구간 가중: 0.5 x (H1~20 평균) + 0.3 x (H21~40 평균) + 0.2 x (H41~60 평균)
H_BANDS = [(1, 20, 0.5), (21, 40, 0.3), (41, 60, 0.2)]
SUMMARY_METRICS = ["pr_auc", "roc_auc", "recall", "precision", "f2", "f1", "accuracy"]
WEIGHTED_METRICS = ["pr_auc", "roc_auc", "recall", "precision", "f2"]   # 사양: 이 5개는 60일 평균과 구간 가중 모두

# ---------------------------------------------------------------- A. 과거 검출 이력
P5, P1 = 1826, 365                           # 5년, 1년 (T 전날까지의 일 수: T-P .. T-1)
K_GRID = [10, 30, 50, 100, 200]
HIST_LEVELS = ["p3", "sgg", "unit"]          # ① 품목군 3단계(전국) ② 시군구(전 품목) ③ 단위(시군구 x 2단계 x 3단계)
WOE_LEVELS = ["sido", "sgg", "p2", "p3"]     # D3: 범주 4개 각각의 WOE (5년 창)
HIST_SETS = {                                # 이름 -> 이력 변수 묶음
    "rate5": [("rate", P5, lv) for lv in HIST_LEVELS],
    "logn5": [("logn", P5, lv) for lv in HIST_LEVELS],
    "rate1": [("rate", P1, lv) for lv in HIST_LEVELS],
}
# HX 추가 묶음(사용자 지시 2026-10-19, 단계 'hx'): feats["hist"] 에 이 이름이 있을 때만 칸이 붙음(기존 실험은 그대로)
#   logn1 = 1년 창 log(1+검사 수), sd1 = 1년 스무딩 검출률의 베타 사후 표준편차 sqrt(m(1-m)/(n+k+1)) (m = rate1 과 같은 식, 같은 p0·k)
HIST_SETS["logn1"] = [("logn", P1, lv) for lv in HIST_LEVELS]
HIST_SETS["sd1"] = [("sd", P1, lv) for lv in HIST_LEVELS]

# ---------------------------------------------------------------- B. 로지스틱 회귀
LR_PARAMS = {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000}   # l1_ratio=0 = L2 (sklearn 1.9 에서 penalty='l2' 는 폐지 예정 경고)

# ---------------------------------------------------------------- C. 고위험 품목
GROUP_MUSHROOM = ["Z2700400000000", "A0700600000000"]
GROUP_RATE_MIN, GROUP_POS_MIN = 0.005, 5     # (나) 학습 구간 품목 내 검출률 >= 0.5% 이고 검출 >= 5건
GROUP_SHARE_TOP = 5                          # (다) 학습 구간 전체 검출 중 비중 상위 5
GROUP_EXPECTED = {
    "C1": ["Z2700400000000", "A0700600000000"],
    "C2": ["Z2700400000000", "A0700600000000", "C0316080000000", "B0101000000000", "C0319060000000"],
    "C3": ["Z2700400000000", "Z3800100000000", "C0316050000000", "C0316010000000", "C0316080000000"],
}
GROUP_DESC = {"C1": "(가) 버섯 2종", "C2": "(나) 품목 내 검출률 >=0.5% & 검출 >=5건",
              "C3": "(다) 전체 검출 중 비중 상위 5종"}
# C4 다운사이징(사용자 지시 2026-10-19 "검출된 품목만 취급하는 다운사이징도 ㄱㄱ"): 학습 구간(기준일 <= 2024-06-30) 시료에서
# 검출이 1건 이상인 품목군 3단계 코드 전부. 목록은 e2data.compute_groups 가 시료에서 다시 계산(아래는 2026-10-19 잰 값과 대조용)
GROUP_C4_POS_MIN = 1
GROUP_EXPECTED["C4"] = ["Z2700400000000", "Z3800100000000", "C0316050000000", "C0316080000000", "C0316010000000",
                        "C0322020000000", "C0319060000000", "C0316020000000", "A0700600000000", "B0101000000000",
                        "C0318090000000", "C0316030000000", "C0316040000000", "C0323020000000", "A1217900000000",
                        "C0319010000000", "A0400100000000", "B0203000000000", "B0205000000000", "B0209000000000",
                        "C0309010000000", "C0316060000000", "C0316070000000", "C0318010000000", "C0319030000000",
                        "C0322030000000"]
GROUP_C4_EXPECTED_TRAIN_SAMPLES = 356955
GROUP_DESC["C4"] = "(라) 학습 구간 검출 1건 이상 품목 전부(다운사이징)"
GROUPS = ["C1", "C2", "C3", "C4"]

# ---------------------------------------------------------------- D. 불균형·WOE
SMOTE_RATIO = 0.1
SMOTE_K = 5
WOE_EPS = 0.5

# ---------------------------------------------------------------- H. 학습 행에 h 여러 개
H10_N = 10
H60_N = 60
H60_RAM_LIMIT_GB = 12.0
H60_RAM_NEED_GB = 32.0                       # 단계 'h60' 시작 조건(시스템 사용 가능 메모리, H30 측정 최대 13.7GB x 2 + 여유)
HBIG_N = 30                                  # H60 대체(메모리 추정 초과 -> 사양 '가능한 최대 개수(예: 30)'), h 는 행마다 중복 없는 무작위 30개
MULTI_H_MEMMAP_MIN = 20                      # 학습 행에 h 를 이 개수 이상 붙이면 디스크 열 우선 파일 + quantize 방식(H10 은 메모리 방식)

# ---------------------------------------------------------------- DL (다음 에이전트가 구현)
DL = {"L": L, "emb_dim": 8, "lr": 1e-3, "batch": 1024, "max_epochs": 30, "patience": 5,
      "lstm_hidden": 64, "lstm_layers": 1,
      "trf": {"d_model": 64, "layers": 2, "heads": 4, "ff": 128, "dropout": 0.1, "pos_emb": L, "head_hidden": 64}}   # 위치 임베딩 수 = 창 길이

# ---------------------------------------------------------------- 시험 실행(--smoke)
SMOKE_NEG = {"train": 20000, "val": 1500, "test": 1500}   # 음성 일부 + 양성 전부 (train 은 SMOTENC 비율 0.1 이 성립하도록 2만)
SMOKE_SEED = 0
SMOKE_SEEDS = [42]
SMOKE_MAX_ROUNDS = 100
SMOKE_PATIENCE = 20

# ---------------------------------------------------------------- 실험 목록
BASE_FEATS = {"weather": True, "cats": True, "hist": [], "woe": False}


def _f(**kw):
    d = dict(BASE_FEATS)
    d.update(kw)
    return d


EXPERIMENTS = [
    dict(id="E00", stage="e00", plan="기준(baseline)", model="cat", feats=_f(),
         desc=f"1차 CatBoost 요약 {L}일 재현: 범주 4 + h + 기상 요약 {NW}, 불균형 처리 없음"),
]
for _k in K_GRID:
    EXPERIMENTS.append(dict(id=f"A1_k{_k}", stage="a_k", plan="E01 (k 비교)", model="cat", seeds=[42],
                            feats=_f(hist=["rate5"], k=_k),
                            desc=f"E00 + 5년 스무딩 검출률 3수준(k={_k}), 시드 42 만"))
EXPERIMENTS += [
    dict(id="A1", stage="a", plan="E01", model="cat", feats=_f(hist=["rate5"], k="k*"),
         desc="E00 + 5년 스무딩 검출률 3수준 (k = k*)"),
    dict(id="A2", stage="a", plan="E02", model="cat", feats=_f(hist=["rate5", "logn5"], k="k*"),
         desc="A1 + 5년 log(1+검사 수) 3수준"),
    dict(id="A3", stage="a", plan="E03", model="cat", feats=_f(hist=["rate5", "logn5", "rate1"], k="k*"),
         desc="A2 + 1년 스무딩 검출률 3수준"),
    dict(id="B1", stage="b", plan="사양 B (계획서 번호 없음)", model="lr",
         feats={"weather": False, "cats": False, "hist": ["rate5", "logn5", "rate1"], "woe": False, "k": "k*"},
         desc="로지스틱 회귀: A3 이력 변수 9 + h (표준화)"),
    dict(id="B2", stage="b", plan="사양 B (계획서 번호 없음)", model="lr",
         feats={"weather": True, "cats": False, "hist": ["rate5", "logn5", "rate1"], "woe": False, "k": "k*"},
         desc=f"로지스틱 회귀: B1 + 기상 요약 {NW} (표준화, 결측 0=학습 평균)"),
]
for _g in ("C1", "C2", "C3"):
    EXPERIMENTS.append(dict(id=f"{_g}_base", stage="c", plan="E05", model="cat", group=_g, feats=_f(),
                            desc=f"고위험 품목 전용 {GROUP_DESC[_g]}, E00 입력"))
    EXPERIMENTS.append(dict(id=f"{_g}_hist", stage="c", plan="E05", model="cat", group=_g, feats="A*",
                            desc=f"고위험 품목 전용 {GROUP_DESC[_g]}, A 최고 설정 입력"))
EXPERIMENTS += [                             # C4 다운사이징(사용자 지시 2026-10-19) - C1~C3 와 같은 틀, 단계 'c'(끝난 C1~C3 는 건너뜀)
    dict(id="C4_base", stage="c", plan="E05 (C4 다운사이징, 사용자 지시 2026-10-19)", model="cat", group="C4", feats=_f(),
         desc=f"검출 품목 전용 {GROUP_DESC['C4']}, E00 입력"),
    dict(id="C4_hist", stage="c", plan="E05 (C4 다운사이징, 사용자 지시 2026-10-19)", model="cat", group="C4", feats="A*",
         desc=f"검출 품목 전용 {GROUP_DESC['C4']}, A 최고 설정 입력"),
]
EXPERIMENTS += [
    dict(id="D1", stage="d", plan="E04", model="cat", feats=_f(), spw=True,
         desc="E00 + CatBoost scale_pos_weight = 학습 불검출 행 수 / 검출 행 수"),
    dict(id="D2", stage="d", plan="E04", model="cat", feats=_f(), smote=True,
         desc="E00 + SMOTENC(학습 행렬만, 검출 = 불검출의 10%, k=5) - 직접 구현"),
    dict(id="D3", stage="d", plan="E04", model="cat", feats=_f(cats=False, woe=True),
         desc="E00 의 범주 4개를 WOE 숫자로 바꿈(5년 창, T 전날까지)"),
    dict(id="H10", stage="h10", plan="사양 H (계획서 번호 없음)", model="cat", feats=_f(), multi_h=H10_N,
         desc="E00 + 학습 행마다 서로 다른 h 10개"),
    dict(id="H30", stage="hbig", plan="사양 H (H60 대체)", model="cat", feats=_f(), multi_h=HBIG_N, seeds=[42],
         desc="E00 + 학습 행마다 서로 다른 h 30개(H60 은 메모리 추정 초과로 대체, 시드 42 만, 디스크 열 우선 파일 + quantize)"),
    # H60: 이 PC 메모리로는 안 돼 H30 으로 대체했던 것을 팀 공용 서버(RAM 94GB)에서 실행(사용자 지시 2026-10-19 "안돌릴 이유가 없다면 ㄱㄱ").
    # 단계 'h60' 은 시스템 사용 가능 메모리가 H60_RAM_NEED_GB 미만이면 run.py 가 실행하지 않음(이 PC 보호).
    dict(id="H60", stage="h60", plan="사양 H", model="cat", feats=_f(), multi_h=H60_N,
         desc="E00 + 학습 행마다 h 1~60 전부(학습 행 1,337만, 팀 공용 서버에서 실행 - 이 PC 메모리 부족, 디스크 열 우선 파일 + quantize)"),
    dict(id="DL_lstmA", stage="dl", plan="사양 DL", model="dl", desc=f"LSTM, 방식 A(h 입력), {L}일"),
    dict(id="DL_trfA", stage="dl", plan="사양 DL", model="dl", desc=f"트랜스포머 인코더, 방식 A(h 입력), {L}일"),
    dict(id="DL_lstmB", stage="dl", plan="사양 DL", model="dl", desc=f"LSTM, 방식 B(출력 60칸), {L}일"),
    dict(id="DL_trfB", stage="dl", plan="사양 DL", model="dl", desc=f"트랜스포머 인코더, 방식 B(출력 60칸), {L}일"),
]
# ---------------------------------------------------------------- F·G (사용자 승인 2026-10-19 "싹다 ㄱ" - 2차 추가분, 계산은 e2fg.py)
# 입력 = E00(범주 4 + h + 기상 요약 L일) + 묶음. 조합(combo)은 F·G 결과를 본 뒤 e2fg.combo_experiments 가 만든다(이 목록에 없음).
EXPERIMENTS += [
    dict(id="F1", stage="f", plan="1차 보고서 표 9 ①", model="cat", feats=_f(ext=["spi"]),
         desc=f"E00 + 표준강수지수 SPI3·SPI6 창 T-{L}..T-1 요약 {NSPI}칸(기상과 같은 요약)"),
    dict(id="F2", stage="f", plan="1차 보고서 표 9 ①", model="cat", feats=_f(ext=["soil"]),
         desc="E00 + 토양(산도·유기물·유효인산 지난 3년 중앙값과 기록 수) 날짜 T 행 6칸"),
    dict(id="F3", stage="f", plan="1차 보고서 표 9 ①", model="cat", feats=_f(ext=["food"]),
         desc="E00 + 식품 속성(품목 3단계 대표 aw·ph·na 숫자 3칸 + aw_label 범주 1칸)"),
    dict(id="F4", stage="f", plan="1차 보고서 표 9 ①", model="cat", feats=_f(ext=["spi", "soil", "food"]),
         desc="E00 + F1 + F2 + F3"),
    dict(id="G1", stage="g", plan="1차 보고서 표 9 ② 생장식", model="cat", feats=_f(ext=["g1"]),
         desc="E00 + Silva 2020 온도 생장 요인 f(일평균기온) 창 요약 4칸(평균·최근 10일 평균·합·f>0 날 수)"),
    dict(id="G2", stage="g", plan="1차 보고서 표 9 ② 생장식", model="cat", feats=_f(ext=["g2"]),
         desc="E00 + µmax(Silva 2020, 기온 x 3단계 aw·pH)·생장 확률 P(Tienungoon 2000 Scott A·L5) 창 요약 9칸"),
    dict(id="G3", stage="g", plan="1차 보고서 표 9 ② 생장식", model="cat", feats=_f(ext=["g3"]),
         desc="E00 + 환경 검출 확률(Ivanek 2009 강수 모델, 2일 전 강수량) 창 요약 3칸"),
]
# 토양 검정 건수(*_n) 제외 확인(사용자 승인 2026-10-19): 건수가 날짜 대리 변수일 수 있어 중앙값 3칸만 넣은 판
_A3 = next(e for e in EXPERIMENTS if e["id"] == "A3")
EXPERIMENTS += [
    dict(id="F2b", stage="f", plan="1차 보고서 표 9 ① (토양 건수 제외 확인)", model="cat", feats=_f(ext=["soilmed"]),
         desc="E00 + 토양 중앙값 3칸(산도·유기물·유효인산 지난 3년 중앙값, 검정 건수 제외) 날짜 T 행"),
    dict(id="A3_F4b", stage="f", plan="1차 보고서 표 9 ① (토양 건수 제외 확인)", model="cat",
         feats=dict(_A3["feats"], ext=["spi", "soilmed", "food"]),
         desc="A3 + F1 + 토양 중앙값 3칸 + F3 (A3_F4 에서 토양 검정 건수 3칸만 뺌)"),
]
# ---------------------------------------------------------------- M. 이동평균 차감 기상 (사용자 승인 2026-10-19, 1차 보고서 표 9 ② - 10-12 교수님 통화의 이동평균 차감 피처)
# 기상 19개 변수마다 날 t 의 a_t = x_t − mean(x_{t−K} … x_{t−1}) (앞 K 일 평균, 당일 제외, 결측은 빼고 평균) -> 창 T−L … T−1 요약 57칸.
# 계산은 e2fg.py 의 'wxanom' 묶음. K = 30 은 제안 기본값(사용자 승인 흐름 2026-10-19).
WXANOM_K = 30
EXPERIMENTS += [
    dict(id="M1", stage="m", plan="1차 보고서 표 9 ② 이동평균 차감", model="cat", feats=_f(ext=["wxanom"]),
         desc=f"E00 + 이동평균 차감 기상(19변수 a_t = x_t − 앞 {WXANOM_K}일 평균, 당일 제외) 창 T-{L}..T-1 요약 57칸(평균·최근 10일 평균·최대)"),
    dict(id="A3_M1", stage="m", plan="1차 보고서 표 9 ② 이동평균 차감", model="cat",
         feats=dict(_A3["feats"], ext=["wxanom"]),
         desc="A3 + 이동평균 차감 기상 57칸(M1 과 같은 칸, A3 의 k* 그대로)"),
]
# ---------------------------------------------------------------- N. 제품명 변수 (사용자 승인 2026-10-19, 교수님 계획서 4·5절 - 계산은 e2nb.py)
# 입력 = A3(범주 4 + h + 기상 요약 NW + 이력 9, k*) + 제품명 묶음. feats["nb"] = 'n1'(과거 뭉치 이력 84)·'n2'(그날 뭉치 멀티핫 21)·'n0'(TF-IDF 500).
# 실험 ID 는 조합 ID 정규식(^A3_(?:F[1-4]|G[1-3])+$)과 겹치지 않음. 예측 때 제품명 입력 여부는 보고서 한계로 적음.
NB_PLAN = "제품명 변수(교수님 계획서 4·5절)"
EXPERIMENTS += [
    dict(id="A3_N1", stage="n", plan=NB_PLAN, model="cat", feats=dict(_A3["feats"], nb=["n1"]),
         desc="A3 + N1 단위 과거 뭉치 이력 84칸(21뭉치 x 비율·스무딩 검출률 x 5년·1년, 창 T-P..T-1, k*)"),
    dict(id="A3_N2", stage="n", plan=NB_PLAN, model="cat", feats=dict(_A3["feats"], nb=["n2"]),
         desc="A3 + N2 그날 그 단위 시료 뭉치 멀티핫 21칸(목표일 시료 제품명 - 운영에서 제품명을 입력받는다는 전제)"),
    dict(id="A3_N1N2", stage="n", plan=NB_PLAN, model="cat", feats=dict(_A3["feats"], nb=["n1", "n2"]),
         desc="A3 + N1 84칸 + N2 21칸"),
    dict(id="A3_N0", stage="n", plan=NB_PLAN, model="cat", feats=dict(_A3["feats"], nb=["n0"]),
         desc="A3 + N0 그날 그 단위 제품명 TF-IDF 500칸(n_rows 제외, 목표일 시료 제품명 - N2 와 같은 전제)"),
]
# ---------------------------------------------------------------- Q. 생장 범위 (리서치팀 생장 최적·한계 범위, 사용자 지시 - [질문] 세션 전달 2026-10-19)
# 파일 = data/데이터셋/생장범위/ ([질문] 세션이 만듦, 정의 = 생장범위_정의.csv). 계산은 e2fg.py 'rgwx'·'rgsoil'·'rgfood' 묶음.
RG_PLAN = "생장 최적·한계 범위(리서치팀 예측인자 표)"
EXPERIMENTS += [
    dict(id="Q1", stage="q", plan=RG_PLAN, model="cat", feats=dict(_A3["feats"], ext=["rgwx"]),
         desc=f"A3 + 일별 기상 범위 6칸(기온 최적·한계, 5cm 지중온도 최적·한계) 창 T-{L}..T-1 평균·최근 10일 평균 12칸"),
    dict(id="Q2", stage="q", plan=RG_PLAN, model="cat", feats=dict(_A3["feats"], ext=["rgsoil"]),
         desc="A3 + 토양 pH 범위 3칸(최적 6.5~7.0·한계 5.0~8.0 0/1, 최적까지 거리) 날짜 T 행"),
    dict(id="Q3", stage="q", plan=RG_PLAN, model="cat", feats=dict(_A3["feats"], ext=["rgfood"]),
         desc="A3 + 식품 범위 4칸(3단계 대표 aw 한계 0/1·0.97 까지 거리, 식품 pH 한계 0/1·7.0 까지 거리)"),
    dict(id="Q4", stage="q", plan=RG_PLAN, model="cat", feats=dict(_A3["feats"], ext=["rgwx", "rgsoil", "rgfood"]),
         desc="A3 + Q1 + Q2 + Q3 (생장 범위 19칸)"),
]
# 묶음 단독 추가(베이스라인 기준, 사용자 결정 2026-10-19 [종합] 전달): E00 + 생장 범위. 기존 Q1~Q4(A3 위에 더한 판)는 그대로
SOLO_PLAN = "묶음 단독 추가(베이스라인 기준)"
EXPERIMENTS += [
    dict(id="E00_Q", stage="q", plan=SOLO_PLAN, model="cat", feats=_f(ext=["rgwx", "rgsoil", "rgfood"]),
         desc="베이스라인(E00) + 생장 범위 19칸(기온·토양온도·토양 pH·식품의 최적·한계 범위)"),
]
# ---------------------------------------------------------------- P. 외부 지표 (사용자 승인·확정 2026-10-19 - 교수님 식품안전지수 지역 지표, [질문] 세션이 연도표 작성)
# 파일 = data/데이터셋/외부_지표/ (친환경_인증비율·지하수_정기수질검사). 계산은 e2fg.py 'eco'·'gw'·'eco_p'·'gw_p' 묶음. 위약판(p) = 자료연도 안 시군구 섞기.
P_PLAN = "외부 지표(식품안전지수 지역 지표)"
EXPERIMENTS += [
    dict(id="P1", stage="p", plan=P_PLAN, model="cat", feats=dict(_A3["feats"], ext=["eco"]),
         desc="A3 + 친환경 인증 면적 비율 1칸(T−1 까지 사용 가능한 가장 최근 자료연도)"),
    dict(id="P2", stage="p", plan=P_PLAN, model="cat", feats=dict(_A3["feats"], ext=["gw"]),
         desc="A3 + 지하수 정기 수질검사 적합률 1칸(T−1 까지 사용 가능한 가장 최근 자료연도)"),
    dict(id="P3", stage="p", plan=P_PLAN, model="cat", feats=dict(_A3["feats"], ext=["eco", "gw"]),
         desc="A3 + P1 + P2 (2칸)"),
    dict(id="P1p", stage="p", plan=P_PLAN + " 위약판", model="cat", feats=dict(_A3["feats"], ext=["eco_p"]),
         desc="A3 + 친환경 비율 위약판 1칸(자료연도마다 시군구끼리 값을 섞음 - P1 과 비교용)"),
    dict(id="P2p", stage="p", plan=P_PLAN + " 위약판", model="cat", feats=dict(_A3["feats"], ext=["gw_p"]),
         desc="A3 + 지하수 적합률 위약판 1칸(자료연도마다 시군구끼리 값을 섞음 - P2 와 비교용)"),
    dict(id="P4", stage="p", plan=P_PLAN, model="cat", feats=dict(_A3["feats"], ext=["haccp"]),
         desc="A3 + HACCP 지정 비율 1칸(T−1 에 유효한 그 시군구·업종묶음 지정 수 ÷ 전국 수, T < 2017-08-03 결측, 업종묶음에 없는 품목 결측)"),
    dict(id="P4p", stage="p", plan=P_PLAN + " 위약판", model="cat", feats=dict(_A3["feats"], ext=["haccp_p"]),
         desc="A3 + HACCP 지정 비율 위약판 1칸(달력 연도마다 시군구 순서를 섞음 - P4 와 비교용)"),
]
# 묶음 단독 추가(베이스라인 기준, 사용자 결정 2026-10-19 [종합] 전달): E00 + 외부 지표 3칸(위약판 없음). 기존 P1~P4(A3 위에 더한 판)는 그대로
EXPERIMENTS += [
    dict(id="E00_P", stage="p", plan=SOLO_PLAN, model="cat", feats=_f(ext=["eco", "gw", "haccp"]),
         desc="베이스라인(E00) + 외부 지표 3칸(친환경 인증 비율·지하수 수질 적합률·HACCP 지정 비율)"),
]
# ---------------------------------------------------------------- HX. 과거 이력 추가 칸 (사용자 지시 2026-10-19 - 계산은 e2hist.py 'logn'·'sd')
# 입력 = A3(범주 4 + h + 기상 요약 NW + 이력 9, k*) + 1년 창 칸. 창 = 기준일 ∈ [T−365, T−1] 시료(e2hist 그대로, T 당일·이후 제외).
HX_PLAN = "과거 이력 파생변수"
EXPERIMENTS += [
    dict(id="A3_L1", stage="hx", plan=HX_PLAN, model="cat", feats=dict(_A3["feats"], hist=_A3["feats"]["hist"] + ["logn1"]),
         desc="A3 + 1년 log(1+검사 수) 3수준(품목군 3단계·시군구·단위)"),
    dict(id="A3_U1", stage="hx", plan=HX_PLAN, model="cat", feats=dict(_A3["feats"], hist=_A3["feats"]["hist"] + ["sd1"]),
         desc="A3 + 1년 스무딩 검출률의 베타 사후 표준편차 sqrt(m(1−m)/(n+k+1)) 3수준(m·p0·k = 1년 스무딩 검출률과 같음)"),
    dict(id="A3_L1U1", stage="hx", plan=HX_PLAN, model="cat", feats=dict(_A3["feats"], hist=_A3["feats"]["hist"] + ["logn1", "sd1"]),
         desc="A3 + 1년 log(1+검사 수) 3 + 1년 사후 표준편차 3 (6칸)"),
]
# ---------------------------------------------------------------- AE. 이상 탐지 AE·VAE (사용자 지시 2026-10-19 - 계산은 e2ae.py)
# 모델 4개: AEt·VAEt(표: A3 숫자 1 + NW + 9 + 범주 4 임베딩, MLP), AEs·VAEs(시퀀스: DL 과 같은 L일 x 38채널, 1D 합성곱).
# 정상 = 학습 구간 불검출 행만(정규화 통계·임베딩 어휘도 그 행에서). 점수: AE = 복원 오차 MSE, VAE = 음의 ELBO. 클수록 이상(= 검출 쪽).
# 학습 행 점수 = 학습 행 전부(검출 포함) 5겹 배정 -> 그 행을 안 본 겹 모델, 검증·시험 = 겹 모델 5개 평균으로 h 1..60 전부(점수 규칙 v2, 사용자 승인 2026-10-19 - e2ae.SCORE_RULE). 시드마다 한 번 계산해 실행 결과 폴더
# 아래 _ae_scores/ 에 저장(같은 시드의 S_·A3_·_p 가 같은 점수를 씀). 크기·학습률·에폭은 기본값(하이퍼파라미터 조정 안 함 - 사용자 지시).
AE = {"batch": 1024, "lr": 1e-3, "max_epochs": 50, "patience": 5, "es_frac": 0.1, "n_folds": 5, "eval_batch": 8192,
      "tab": {"emb_dim": 8, "hidden": [128, 64], "latent": 16},
      "seq": {"L": L, "filters": [32, 16], "kernel": 7, "stride": 2, "dropout": 0.2},
      "placebo_seed": 20261019}
AE_KINDS = ["AEt", "VAEt", "AEs", "VAEs"]
AE_KIND_DESC = {"AEt": "표 AE", "VAEt": "표 VAE", "AEs": "시퀀스 AE", "VAEs": "시퀀스 VAE"}
AE_PLAN = "이상 탐지(AE·VAE)"
for _kd in AE_KINDS:                         # (a) 점수 단독 - CatBoost 없음, 점수를 예측값(순위)으로 그대로 채점
    EXPERIMENTS.append(dict(id=f"S_{_kd}", stage="ae", plan=AE_PLAN, model="ae", feats=dict(_A3["feats"], ae=[_kd]),
                            desc=f"{AE_KIND_DESC[_kd]} 이상 점수 단독(클수록 검출로 보고 그대로 채점, 판정 기준값 = 검증 F2 최대 점수)"))
for _kd in AE_KINDS:                         # (b) A3 + 점수 1칸
    EXPERIMENTS.append(dict(id=f"A3_{_kd}", stage="ae", plan=AE_PLAN, model="cat", feats=dict(_A3["feats"], ae=[_kd]),
                            desc=f"A3 + {AE_KIND_DESC[_kd]} 이상 점수 1칸(학습 행 = 5겹 교차 적합 점수)"))
for _kd in AE_KINDS:                         # (c) 위약판 - 점수를 분할·h 마다 예측일 T 의 달력 연도 안에서 행끼리 섞음
    EXPERIMENTS.append(dict(id=f"A3_{_kd}_p", stage="ae", plan=AE_PLAN + " 위약판", model="cat", feats=dict(_A3["feats"], ae=[f"{_kd}_p"]),
                            desc=f"A3 + {AE_KIND_DESC[_kd]} 점수 위약판 1칸(예측일 T 의 달력 연도 안에서 행끼리 섞음 - A3_{_kd} 와 비교용)"))
# ---------------------------------------------------------------- XS. 작은 실험 3개 (사용자 지시 2026-10-19 - 교수님 계획서 11절 3순위·6절 실험 A 빈칸 채우기, 단계 'xs')
# 기존 설정 그대로(CatBoost, 시드 42·43·44). 계산은 기존 FeatureBuilder 그대로: feats["weather"] False 면 숫자 칸 = h(+ 이력)만, 'rate1' = A3 의 1년 스무딩 검출률과 같은 e2hist 계산.
_E00 = next(e for e in EXPERIMENTS if e["id"] == "E00")
EXPERIMENTS += [
    dict(id="E00_R1", stage="xs", plan="계획서 11절 3순위(기준 + 최근 1년 검출률)", model="cat", feats=dict(_E00["feats"], hist=["rate1"], k="k*"),
         desc="E00 + 1년 스무딩 검출률 3수준(rate1: 품목군 3단계·시군구·단위, k = k*, A3 의 rate1 칸과 같은 계산)"),
    dict(id="E00_NW", stage="xs", plan="계획서 6절 실험 A(정형 데이터만)", model="cat", feats=dict(_E00["feats"], weather=False),
         desc=f"E00 에서 기상 요약 {NW}칸을 뺀 판(범주 4 + h)"),
    dict(id="A3_NW", stage="xs", plan="계획서 6절 실험 A(정형 데이터만)", model="cat", feats=dict(_A3["feats"], weather=False),
         desc=f"A3 에서 기상 요약 {NW}칸을 뺀 판(범주 4 + h + 이력 9, k*)"),
]
# ---------------------------------------------------------------- STK. E08 스태킹 (사용자 지시 2026-10-19 - 교수님 계획서 7절 방식 C, 기록표 E08 '스태킹 = 정형 모델 + 시계열 모델')
# 재학습 없음: 시드마다 기본 모델 3개의 저장 예측(results/<ID>__s<시드>/pred_val·pred_test.parquet)을 읽어 로지스틱 메타 모델. 계산은 e2stk.py.
# 메타 모델을 검증 구간 저장 예측으로 학습하므로 검증 점수는 메타 학습 구간 점수(비교에는 시험 점수만). 단계 'stk' 는 특성 자료(Ctx2)를 만들지 않음.
STK = {"clip": 1e-6, "max_iter": 10000}      # 입력 = logit(clip(p, 1e-6, 1 − 1e-6)), 메타 = sklearn LogisticRegression(기본값, max_iter, random_state = 시드)
STK_BASE_DESC = {"E00": "CatBoost E00", "A3": "CatBoost A3", "DL_lstmA": "LSTM 방식 A", "DL_trfA": "트랜스포머 방식 A"}
STK_PLAN = "E08 스태킹"
for _b in ("E00", "A3"):
    EXPERIMENTS.append(dict(id=f"STK_{_b}", stage="stk", plan=STK_PLAN, model="stack", bases=[_b, "DL_lstmA", "DL_trfA"],
                            desc=f"스태킹: {_b}(CatBoost) + DL_lstmA(LSTM 방식 A) + DL_trfA(트랜스포머 방식 A) 저장 예측의 logit 3칸 -> "
                                 "로지스틱 회귀 메타 모델(검증 구간 저장 예측 전체로 학습, 재학습 없음)"))
# ---------------------------------------------------------------- XALL. 추가 변수 전부 (사용자 지시 2026-10-19 [종합] 전달 - "LIMS+기상 이외에 추가한 모든 변수에 대한 결과",
# 단계 'xall', 시드 42·43·44, CatBoost = E00 설정, k = k*). ALL_A = E00 + 단계 선택 12묶음 전부(운영 때 쓸 수 있는 것), ALL_B = ALL_A + 그날 제품명(n2·n0).
# AE·VAE 점수 4개의 표 입력 = ALL_A 의 나머지 입력(숫자 칸 + 범주 임베딩, FS12 와 같은 규칙 - 'ae_tab'), 시퀀스 = L일 기상. ALL_B 도 같은 표 입력(그날 제품명 칸은 AE 입력에 안 넣음).
XALL_PLAN = "추가 변수 전부(LIMS·기상 밖 변수 모두)"
NB_DAY_PREMISE = "운영 전제: 그날 검사할 제품명을 알아야 함"
_ALL_TAB = dict(_E00["feats"], ext=["spi", "soilmed", "food", "eco", "gw", "haccp", "rgwx", "rgsoil", "rgfood", "g1", "g2", "g3", "wxanom"],
                hist=["rate5", "logn5", "rate1", "logn1", "sd1"], woe=True, k="k*", nb=["n1"])
_ALL_A_DESC = "베이스라인 + 운영 때 쓸 수 있는 추가 변수 전부(외부 수집 데이터 + 파생변수 + 이상 점수)"
EXPERIMENTS += [
    dict(id="ALL_A", stage="xall", plan=XALL_PLAN, model="cat", feats=dict(_ALL_TAB, ae=list(AE_KINDS), ae_tab=dict(_ALL_TAB)),
         desc=_ALL_A_DESC),
    dict(id="ALL_A0", stage="xall", plan=XALL_PLAN, model="cat", feats=dict(_ALL_TAB),          # ALL_A 에서 AE·VAE 점수 4칸만 뺀 판(사용자 승인 2026-10-19)
         desc="베이스라인 + 운영 때 쓸 수 있는 추가 변수 전부(이상 점수 제외)"),
    dict(id="ALL_B", stage="xall", plan=XALL_PLAN, model="cat",
         feats=dict(_ALL_TAB, nb=["n1", "n2", "n0"], ae=list(AE_KINDS), ae_tab=dict(_ALL_TAB)),
         desc=_ALL_A_DESC + f" + 그날 검사 제품명 정보까지 전부({NB_DAY_PREMISE})"),
    dict(id="E00_N2", stage="xall", plan=SOLO_PLAN, model="cat", feats=_f(nb=["n2"], k="k*"),
         desc=f"베이스라인(E00) + 그날 제품명 유형 멀티핫 21칸({NB_DAY_PREMISE})"),
    dict(id="E00_N0", stage="xall", plan=SOLO_PLAN, model="cat", feats=_f(nb=["n0"], k="k*"),
         desc=f"베이스라인(E00) + 그날 제품명 TF-IDF 500칸({NB_DAY_PREMISE})"),
]
# ---------------------------------------------------------------- FS·BW. 양방향 단계 선택 (사용자 결정 2026-10-19 [종합] 전달 - 100일 위 2차 실험 A단계, 단계 'fs'·'bw')
# 판단 점수 = 검증 T 통합 60일평균 PR-AUC 시드 42·43·44 평균(e2eval.val_tint_pr_auc). CatBoost = E00 설정 그대로, k = k*.
# 전진('fs'): 기준 = E00 에서 시작, 아래 순서로 '지금 기준 + 묶음 하나'(FS01~FS12) -> 후보 점수 > 기준 점수면 남김(새 기준), 아니면 뺌.
# 뒤로('bw'): 전진 최종 기준에서 남긴 묶음을 하나씩 뺀 후보(BW01, BW02, … 차례 번호) -> 오른 후보가 있으면 가장 많이 오른 것을 빼고 반복, 없으면 끝.
# 실험은 실행 중에 정해짐(e2fs.py, combo 처럼 EXPERIMENTS 에 없음). 결정 기록 = results/selection_FS.json.
# 'ae4' 의 표 AE·VAE 입력 = FS12 시점 기준 모델 입력(숫자 칸 + 범주 임베딩), 뒤로 단계에서는 FS12 때 계산한 점수를 고정 묶음으로 씀(다시 계산 안 함).
# (이전 안 '누적 실험 CUM01~CUM14' 는 사용자 결정으로 접음 - 정의 삭제, 서버 root_L100 에 CUM01~CUM03 결과만 남아 있음)
FS_PLAN = {"forward": "양방향 단계 선택(전진)", "backward": "양방향 단계 선택(뒤로)"}
FS_BUNDLES = [   # code = 묶음 코드(selection_FS.json 'bundle'), add = 입력에 더할 것, n = 칸 수(숫자 + 범주)
    dict(code="spi", add={"ext": ["spi"]}, n=NSPI, name="표준강수지수(SPI3·SPI6)",
         desc=f"창 T-{L}..T-1 요약, 기상과 같은 요약"),
    dict(code="soilmed", add={"ext": ["soilmed"]}, n=3, name="토양 중앙값 3칸",
         desc="산도·유기물·유효인산 지난 3년 중앙값, 검정 건수 제외"),
    dict(code="food", add={"ext": ["food"]}, n=4, name="식품 속성",
         desc="수분활성도·식품 pH·염도 3 + 수분활성도 표기 범주 1"),
    dict(code="eco_gw_haccp", add={"ext": ["eco", "gw", "haccp"]}, n=3, name="외부 지표",
         desc="친환경 인증 면적 비율·지하수 수질검사 적합률·HACCP 지정 비율"),
    dict(code="rg", add={"ext": ["rgwx", "rgsoil", "rgfood"]}, n=19, name="생장 범위",
         desc=f"기상 범위 12(창 T-{L}..T-1 평균·최근 10일 평균)·토양 pH 범위 3·식품 범위 4"),
    dict(code="g123", add={"ext": ["g1", "g2", "g3"]}, n=16, name="생장식 지수",
         desc="온도 생장 요인 4·최대 생장 속도와 생장 확률(2균주) 9·강수 검출 확률 3"),
    dict(code="wxanom", add={"ext": ["wxanom"]}, n=57, name="이동평균 차감 기상",
         desc=f"19변수 x 평균·최근 10일 평균·최대, a_t = x_t − 앞 {WXANOM_K}일 평균"),
    dict(code="hist_core", add={"hist": ["rate5", "logn5", "rate1"]}, n=9, name="과거 검출 이력",
         desc="5년 스무딩 검출률·5년 로그 검사 건수·1년 스무딩 검출률 x 품목군 3단계·시군구·단위, k = k*"),
    dict(code="hist_1y", add={"hist": ["logn1", "sd1"]}, n=6, name="1년 검사 건수·검출률 불확실성",
         desc="1년 log(1+검사 수) 3·1년 스무딩 검출률의 베타 사후 표준편차 3"),
    dict(code="woe", add={"woe": True}, n=4, name="WOE",
         desc="시도·시군구·품목 2단계·3단계, 5년 창 - 범주 4칸은 그대로"),
    dict(code="n1", add={"nb": ["n1"]}, n=84, name="제품명 과거 유형",
         desc="단위 과거 21뭉치 x 비율·스무딩 검출률 x 5년·1년, T 전날까지"),
    dict(code="ae4", add={"ae": ["AEt", "VAEt", "AEs", "VAEs"]}, n=4, name="AE·VAE 이상 점수 4개",
         desc=f"표 AE·표 VAE(입력 = 그 시점 기준 모델 숫자 칸 + 범주 임베딩)·시퀀스 AE·시퀀스 VAE({L}일 기상)"),
]
FS_BY_CODE = {b["code"]: b for b in FS_BUNDLES}
FS_K_NEED = {"hist_core", "hist_1y", "n1", "ae4"}      # 이 묶음이 있으면 입력에 k = k*
EXP_BY_ID = {e["id"]: e for e in EXPERIMENTS}
STAGES =["e00", "prev1", "a_k", "a", "b", "c", "d", "h10", "hbig", "h60", "dl", "f", "g", "combo", "m", "n", "q", "p", "hx", "ae", "xs", "stk", "xall", "fs", "bw", "tables"]
# 사양에 있으나 실행하지 않는 실험(표에 이유와 함께 한 줄). H60 은 2026-10-19 부터 서버에서 실행(EXPERIMENTS 의 H60) - 옛 이유 기록:
# NOT_RUN = [dict(id="H60", plan="사양 H", model="cat",
#                 desc="E00 + 학습 행마다 h 60개 전부(학습 행 1,337만)",
#                 reason="실행 안 함: 학습까지 개인 메모리(private) 추정 약 14~15GB > 12GB. 측정: 양자화·원값 놓기까지 최대 5.9GB, 그 뒤 5.0~5.2GB(logs/h60_memtest.json). 학습 단계 증가분 측정 E00 0.58GB(22만 행, 100라운드)·H10 1.95GB(223만 행, 50라운드)·H30 4.6GB(668만 행, 5라운드)(logs/timing_*.json) -> 1,337만 행 약 9GB. 이 PC 사용 가능 메모리 약 10~12GB. 사양에 따라 H30(측정 최대 8.2GB)으로 대체")]
NOT_RUN = []


def exp_seeds(exp, smoke):
    if smoke:
        return list(SMOKE_SEEDS)
    return list(exp.get("seeds", SEEDS))


def run_dir_name(exp_id, seed):
    return f"{exp_id}__s{seed}"


def paths(smoke):
    res = EXP2_DIR / "results" / ("_smoke" if smoke else "")
    tab = EXP2_DIR / "tables" / ("_smoke" if smoke else "")
    logs = EXP2_DIR / "logs"
    return {"results": res, "tables": tab, "logs": logs, "cache": EXP2_DIR / "cache",
            "progress": logs / ("progress_smoke.log" if smoke else "progress.log")}
