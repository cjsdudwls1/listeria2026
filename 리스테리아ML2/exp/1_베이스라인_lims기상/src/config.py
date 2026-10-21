# -*- coding: utf-8 -*-
"""과업 2 실험 설정 (사용자 확정 사양). 값을 바꾸면 결과가 달라지므로 임의로 고치지 말 것."""
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋")
SAMPLE_PATH = DATA_DIR / "시료_시군구.parquet"
WEATHER_PATH = DATA_DIR / "식정원_db_기상기후" / "시군구_일별_기상.parquet"

WEATHER_VARS = [
    "avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "max_wind",
    "sunshine", "solar_rad", "soil_t5", "soil_t10", "soil_t20", "soil_t30",
    "soil_t_nas", "soil_moist", "no2", "co", "so2", "wind_dir_sin", "wind_dir_cos",
]

# 단위 = (시군구, 2단계, 3단계). 코드 빈 값은 '코드없음'
KEY_COLS = ["sgg_code", "train_prdlst_code", "train_prdlst_code_2"]
NULL_CODE = "코드없음"
CAT_COLS = ["sido", "sgg", "p2", "p3"]          # 모델 범주 입력 4개

# 분할 (기준일 d)
TRAIN_END = "2024-06-30"
VAL_START, VAL_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
SPLITS = ["train", "val", "test"]

H_MAX = 60                      # 앞당김 h = 1..60
SEED = 42                       # 모델 시드, train h 배정
SEED_H_VAL_ES = 43              # 조기 종료용 val h 배정

# 실행 구성
TREE_MODELS = ["xgboost", "lightgbm", "catboost"]
TREE_INPUTS = ["summary", "raw"]
STEP1_L = 100
STEP2_LS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BLOCK = 10                      # 요약 특성의 구간 길이(일)

# 트리 공통
MAX_ROUNDS = 2000
PATIENCE = 100

# LSTM
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_EMB_DIM = 8                # 사양에 없음 -> 범주 4개 모두 8차원(가장 단순)
LSTM_HEAD = 64
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_BATCH = 1024
LSTM_MAX_EPOCHS = 30
LSTM_PATIENCE = 5
LSTM_EVAL_BATCH = 1024          # 예측만 할 때의 배치(사양에 없음 -> 학습 배치와 같게. 8192 는 cuDNN 이 GPU 5GB 넘게 써서 줄임)

# 시험 실행(--smoke)
SMOKE_L = 20
SMOKE_STEP2_LS = [10, 20]
SMOKE_NEG = {"train": 3000, "val": 1500, "test": 1500}   # 음성 일부 + 양성 전부
SMOKE_SEED = 0


def run_name(model: str, inp: str, L: int) -> str:
    return f"{model}__{inp}__L{L:03d}"


def parse_run_name(name: str):
    model, inp, l = name.split("__")
    assert l.startswith("L"), name
    L = int(l[1:])
    if model == "lstm":
        assert inp == "seq", name
    else:
        assert model in TREE_MODELS and inp in TREE_INPUTS, name
    assert L % BLOCK == 0 and 10 <= L <= 100, name
    return model, inp, L


def step1_runs(smoke: bool):
    L = SMOKE_L if smoke else STEP1_L
    order = [("lightgbm", "summary"), ("xgboost", "summary"), ("catboost", "summary"),
             ("lightgbm", "raw"), ("xgboost", "raw"), ("catboost", "raw"), ("lstm", "seq")]
    return [run_name(m, i, L) for m, i in order]


def step2_Ls(smoke: bool):
    return SMOKE_STEP2_LS if smoke else STEP2_LS


def paths(smoke: bool, results_root=None, tables_root=None):
    sfx = "_smoke" if smoke else ""
    res = Path(results_root) if results_root else EXP_DIR / f"results{sfx}"
    tab = Path(tables_root) if tables_root else EXP_DIR / f"tables{sfx}"
    logs = EXP_DIR / "logs"
    prog = logs / (f"progress_{res.name}.log" if results_root else f"progress{sfx}.log")
    return {"results": res, "tables": tab, "logs": logs, "progress": prog}
