# -*- coding: utf-8 -*-
"""3차 연구 보고서 Ⅰ부 2장 EDA 를 시료 단위가 아니라 예측 단위 행(단위 × 날짜) 기준으로 다시 셈 (2026-10-22).

사용자 지시(10-22): "그림 2, 표 6 등 우리 예측 단위 (시군구x2차x3차)로 정제한 데이터 기준으로 보여줘야할듯?
  다른것도 시료단위가 아니라 우리 예측 단위 기준으로."

자료(읽기만): 리스테리아ML2/data/데이터셋/시료_시군구.parquet (국내·시군구 배정 시료 628,517건, 검출 1,844),
             시군구_대표좌표.parquet(시도·시군구 이름), 식정원_db_기상기후/시군구_일별_기상.parquet(평균기온 avg_temp).
행 정의 = 실험 1차 src/data.py 의 load_samples + build_unit_rows 와 같음(2·3차 실험도 이 함수를 씀):
  예측 단위 = (sgg_code, train_prdlst_code, train_prdlst_code_2), 코드 빈 값 = '코드없음'
  행 = 단위 × 기준일 d(mnfctur_de 를 날짜로), y = 그 행 시료 중 judge_result '검출' 이 1건 이상이면 1
  -> 이 스크립트가 만든 행을 실험 build_unit_rows 결과(1차 src 를 읽기 전용으로 import)와 한 행씩 견줌
     (키·시료 수·검출 시료 수·y 가 모두 같아야 진행) + 실험·원고 표 6 값과 견줌(다르면 멈춤).
기간: 기준일 전체(2015-04-07 ~ 2026-06-30, 자료 설명용 — 모델·변수 고르기에 쓰지 않음).
  표 6 의 train = 기준일 ~ 2025-06-30, test = 기준일 2025-07-01 ~ 2026-06-30.
지표(행 기준, 그림·표):
  검출 비중(%) = 그 범주의 검출 행 ÷ 전체 검출 행(1,141) × 100   (파란 막대)
  행 비중(%)   = 그 범주의 행 ÷ 전체 행(291,812) × 100            (회색 막대)
  검출률(%)    = 그 범주의 검출 행 ÷ 그 범주의 행 × 100           (기온 구간·품목 예시)
같은 계산을 시료 기준(시료 1건 = 1)으로도 해서 원고의 지금 숫자(02·03·06·07 스크립트 결과)와 같은지 먼저 확인함
  -> 정의·계산이 기존 스크립트와 같고 셈 단위만 바뀌었다는 근거(다르면 멈춤).
품목 3단계 = train_prdlst_code_2(빈 값 = '코드없음' 한 품목, 3단계 코드마다 2단계는 하나). 이름은 코드별 첫 값(코드와 1:1).
'팽이' = 제품명(prduct_nm)에 '팽이' 글자가 있음. 행 기준은 (가) 그 행(같은 단위·같은 날) 시료 중 하나라도 제품명에 '팽이',
  (나) 그 행의 검출 시료 중 하나라도 제품명에 '팽이' 두 가지를 셈.
그림 모양: 02_eda_figs.py·11_eda_item_test_bins.py 와 같음(Malgun Gothic, 폭 6.3인치, 200dpi, 글자 9.5pt, 같은 rcParams,
  BLUE = 검출, REF = 전체 행, 그림 5 왼쪽 판 NEUTRAL, 그림 안 제목 없음). 기존 그림·표 파일은 건드리지 않음(새 이름만 씀).
출력: figs3/I_1u_year_rows.png·I_2u_item3_share_top10.png·I_4u_month_share.png·I_5u_item3_row_bins.png,
      work3/out_facts_unit/eda_unit_summary.json·eda_u_*.csv
실행: <리스테리아ML/.venv-exp2/Scripts/python.exe> -B 12_eda_unit_basis.py
"""
import sys

sys.dont_write_bytecode = True

import importlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2")
DS = BASE / "data" / "데이터셋"
SAMPLE = DS / "시료_시군구.parquet"
COORD = DS / "시군구_대표좌표.parquet"
WX = DS / "식정원_db_기상기후" / "시군구_일별_기상.parquet"
EXP1_SRC = BASE / "exp" / "1_베이스라인_lims기상" / "src"
REP = BASE / "notebook" / "보고서" / "경복대양식"
FIG = REP / "figs3"
OUT = REP / "work3" / "out_facts_unit"
OUT.mkdir(parents=True, exist_ok=True)

FIG_NAMES = {"year": "I_1u_year_rows.png", "item": "I_2u_item3_share_top10.png",
             "month": "I_4u_month_share.png", "bins": "I_5u_item3_row_bins.png"}
OLD_FIGS = {"I_1_year_detections.png", "I_2_item3_share_top10.png", "I_3_sido_share.png", "I_4_month_share.png",
            "I_5_item3_count_rate.png", "I_5_item3_test_bins.png", "I_5_temp_bin_rate.png", "I_0_T_h_d.png"}
assert not (set(FIG_NAMES.values()) & OLD_FIGS)

NULL_CODE = "코드없음"
KEYS = ["sgg_code", "p2", "p3"]
TRAIN_END = pd.Timestamp("2025-06-30")
TEST_START, TEST_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-06-30")
Z27, Z38 = "Z2700400000000", "Z3800100000000"      # 정액,수정란,종란 > 기타 / 기타기준규격외
EOMNAMU = "A1217900000000"                         # 기타식물류 > 엄나무(원고 예시)

# ---------------------------------------------------------------- 모양(02_eda_figs.py·11_eda_item_test_bins.py 와 같음)
BLUE = "#2a78d6"        # 검출
REF = "#c9c7bf"         # 전체 행(회색)
NEUTRAL = "#4f5d6e"     # 품목 수(그림 5 왼쪽 판)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#6f6e69"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FS = 9.5
W = 6.3
DPI = 200
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": FS,
    "axes.labelsize": FS, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": FS,
})
LEG_DET = "검출 비중(전체 검출 행 중 %)"
LEG_ROW = "행 비중(전체 행 중 %)"


def style(ax, axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


LAYOUT = {}


def layout_problems(fig, name):
    """저장 해상도에서 글자 상자를 재어 (1) 그림 밖으로 나간 글자(잘림) (2) 같은 판 안 꼬리표·범례끼리 겹침을 찾음."""
    fig.set_dpi(DPI)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    wpx, hpx = fig.canvas.get_width_height()
    probs = []
    drawn = []                      # 실제로 그려지는 글자만(눈금 범위 밖 눈금 글자는 matplotlib 이 만들기만 하고 그리지 않음)
    for ax in fig.axes:
        drawn += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            drawn += list(ax.get_legend().get_texts())
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            drawn += [tk.label1 for tk in axis.get_major_ticks() if lo - 1e-9 <= tk.get_loc() <= hi + 1e-9]
    for t in drawn:
        if not (t.get_visible() and t.get_text().strip()):
            continue
        bb = t.get_window_extent(rend)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > wpx + 0.5 or bb.y1 > hpx + 0.5:
            probs.append(f"잘림: '{t.get_text()}' ({bb.x0:.0f},{bb.y0:.0f})-({bb.x1:.0f},{bb.y1:.0f}) / 그림 {wpx}×{hpx}")
    for ax in fig.axes:
        boxes = [(t.get_text(), t.get_window_extent(rend)) for t in ax.texts if t.get_visible() and t.get_text().strip()]
        if ax.get_legend() is not None:
            boxes.append(("[범례]", ax.get_legend().get_window_extent(rend)))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][1], boxes[j][1]
                ox, oy = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
                if ox > 0 and oy > STACK_PX:
                    probs.append(f"겹침: '{boxes[i][0]}' · '{boxes[j][0]}' {ox:.0f}×{oy:.0f}px")
                elif ox > 0 and oy > 0:
                    STACKED.append(f"{name}: '{boxes[i][0]}' · '{boxes[j][0]}' 줄 상자만 {oy:.0f}px")
    return probs


# 위아래로 쌓인 두 꼬리표(가로 막대 그림의 검출·행 비중 짝)는 글자 상자(줄 높이 = 글자 + 위아래 여백)가 조금 겹쳐도 됨.
# 상자 세로 겹침이 STACK_PX 이하면 '참고'로만 따로 적음. 다만 상자 기준으로는 획이 닿는지 알 수 없어(10-22 잉크 점검에서
# 02 와 같은 배치일 때 '(44행)'·'9.3%' 등 3쌍의 획이 닿았음), 그림 3 은 짝 꼬리표를 위·아래로 PAIR_NUDGE_PT 씩 벌려 그림.
STACK_PX = 6
STACKED = []


def save(fig, key):
    p = FIG / FIG_NAMES[key]
    LAYOUT[FIG_NAMES[key]] = layout_problems(fig, key)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("그림", p, "| 글자 점검:", LAYOUT[FIG_NAMES[key]] or "잘림·겹침 없음")
    return str(p)


def r1(v):
    return float(f"{v:.1f}")


def r2(v):
    return float(f"{v:.2f}")


# ---------------------------------------------------------------- 시료(실험 load_samples 와 같은 규칙 + 이름·제품명)
raw = pq.read_table(SAMPLE, columns=["judge_result", "mnfctur_de", "sgg_code", "train_prdlst_code", "train_prdlst_nm",
                                     "train_prdlst_code_2", "train_prdlst_nm_2", "prduct_nm"]).to_pandas()
jr = raw["judge_result"].astype(object)
bad = set(jr.unique()) - {"검출", "불검출"}
if bad:
    raise SystemExit(f"judge_result 에 예상 밖 값: {bad}")
pnm = raw["prduct_nm"].astype(object)
S = pd.DataFrame({
    "sgg_code": raw["sgg_code"].astype(str),
    "p2": raw["train_prdlst_code"].astype(object).where(raw["train_prdlst_code"].notna(), NULL_CODE),
    "p3": raw["train_prdlst_code_2"].astype(object).where(raw["train_prdlst_code_2"].notna(), NULL_CODE),
    "d": pd.to_datetime(raw["mnfctur_de"].values.astype("datetime64[D]")),
    "det": (jr == "검출").to_numpy().astype(np.int64),
    "enoki": pnm.where(pnm.notna(), "").astype(str).str.contains("팽이", regex=False).to_numpy().astype(np.int64),
})
S["enoki_det"] = S["enoki"] * S["det"]
S["cnt"] = 1
N_S, P_S = len(S), int(S["det"].sum())
if (N_S, P_S) != (628517, 1844):
    raise SystemExit(f"시료 수가 다름: {N_S}·{P_S} (기대 628,517·1,844)")
if (S["sgg_code"].str.len() != 5).any():
    raise SystemExit("sgg_code 5자리 아님")
n_p2_per_p3 = S.groupby("p3")["p2"].nunique()
if (n_p2_per_p3 > 1).any():
    raise SystemExit(f"3단계 코드 중 2단계가 둘 이상인 코드 {int((n_p2_per_p3 > 1).sum())}개 — 품목 3단계 = 코드 하나로 볼 수 없음")
NAMES = (raw.assign(p3=S["p3"]).astype({"train_prdlst_nm": object, "train_prdlst_nm_2": object})
         .groupby("p3")[["train_prdlst_nm", "train_prdlst_nm_2"]].first())
NAMES.columns = ["nm2", "nm3"]


def item_label(code):
    if code == NULL_CODE or pd.isna(NAMES.loc[code, "nm2"]):
        return NULL_CODE
    a, b = str(NAMES.loc[code, "nm2"]), str(NAMES.loc[code, "nm3"])
    return a if a == b == "기타기준규격외" else f"{a} > {b}"


coord = pq.read_table(COORD).to_pandas()
SGG_NM = dict(zip(coord["sgg_code"].astype(str), coord["sgg_nm"]))
SIDO_NM = {c[:2]: n.split()[0] for c, n in SGG_NM.items()}
SHORT = {"서울특별시": "서울", "전남광주통합특별시": "전남광주", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
         "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "충청북도": "충북", "충청남도": "충남",
         "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주", "강원특별자치도": "강원", "전북특별자치도": "전북"}


# ---------------------------------------------------------------- 단위 × 날짜 행(실험 build_unit_rows 와 같은 규칙)
def build_rows(s):
    g = (s.groupby(KEYS + ["d"], sort=False, observed=True)
         .agg(n=("det", "size"), npos=("det", "sum"), n_enoki=("enoki", "sum"), n_enoki_det=("enoki_det", "sum"))
         .reset_index())
    g["det"] = (g["npos"] > 0).astype(np.int64)
    g = g.sort_values(KEYS + ["d"], kind="mergesort").reset_index(drop=True)
    g["cnt"] = 1
    # 팽이 표시: 행 안 시료 중 하나라도(enoki) / 행 안 검출 시료 중 하나라도(enoki_det)
    g["enoki"] = (g["n_enoki"] > 0).astype(np.int64)
    g["enoki_det"] = (g["n_enoki_det"] > 0).astype(np.int64)
    return g


R = build_rows(S)


def check_against_exp1(rows):
    """1차 src 의 data.load_samples + build_unit_rows 를 읽기 전용으로 불러 같은 행인지 한 행씩 견줌."""
    sys.path.insert(0, str(EXP1_SRC))
    try:
        for m in ("config", "data"):
            if m in sys.modules:
                raise SystemExit(f"모듈 이름 '{m}' 이 이미 올라와 있음 — 1차 모듈과 헷갈릴 수 있어 멈춤")
        C1 = importlib.import_module("config")
        D1 = importlib.import_module("data")
    finally:
        sys.path.pop(0)
    for mod in (C1, D1):
        if Path(mod.__file__).resolve().parent != EXP1_SRC.resolve():
            raise SystemExit(f"다른 모듈을 읽음: {mod.__file__}")
    if C1.KEY_COLS != ["sgg_code", "train_prdlst_code", "train_prdlst_code_2"] or C1.NULL_CODE != NULL_CODE \
            or Path(C1.SAMPLE_PATH).resolve() != SAMPLE.resolve():
        raise SystemExit("1차 config 의 단위 키·빈 코드·시료 경로가 이 스크립트와 다름")
    e = D1.build_unit_rows(D1.load_samples())
    res = {"exp1_rows": int(len(e)), "exp1_pos_rows": int(e["y"].sum())}
    same = len(e) == len(rows)
    if same:
        pairs = [("sgg_code", "sgg_code"), ("train_prdlst_code", "p2"), ("train_prdlst_code_2", "p3")]
        for a, b in pairs:
            same &= bool(np.array_equal(e[a].astype(str).to_numpy(), rows[b].astype(str).to_numpy()))
        same &= bool(np.array_equal(e["d"].to_numpy().astype("datetime64[D]"), rows["d"].to_numpy().astype("datetime64[D]")))
        for a, b in (("n_samples", "n"), ("n_pos_samples", "npos"), ("y", "det")):
            same &= bool(np.array_equal(e[a].to_numpy().astype(np.int64), rows[b].to_numpy().astype(np.int64)))
    res["same_rows_as_exp1_build_unit_rows"] = bool(same)
    if not same:
        raise SystemExit(f"실험 build_unit_rows 결과와 행이 다름: {res}")
    return res


EXP1_CHECK = check_against_exp1(R)
N_R, P_R = len(R), int(R["det"].sum())

for df in (S, R):
    df["year"] = df["d"].dt.year
    df["month"] = df["d"].dt.month
    df["ym"] = df["d"].dt.to_period("M")
    df["sido"] = df["sgg_code"].str[:2]
    df["unit"] = df["sgg_code"] + "|" + df["p2"].astype(str) + "|" + df["p3"].astype(str)


# ---------------------------------------------------------------- 표 6(단위 × 날짜·예측 단위 줄) + 실험·원고 값 대조
def table6():
    out = []
    for name, a, b in (("전체 기간", R["d"].min(), TEST_END), ("train", R["d"].min(), TRAIN_END), ("test", TEST_START, TEST_END)):
        r = R[(R["d"] >= a) & (R["d"] <= b)]
        u_all = r["unit"].nunique()
        u_pos = r.loc[r["det"] == 1, "unit"].nunique()
        out.append({"기간": name, "기준일 처음": str(r["d"].min().date()), "기준일 끝": str(r["d"].max().date()),
                    "단위x날짜 행 수": int(len(r)), "단위x날짜 검출 행": int(r["det"].sum()),
                    "단위x날짜 검출률(%)": round(100.0 * r["det"].sum() / len(r), 3),
                    "행 하나의 시료 수 평균": round(float(r["n"].mean()), 2),
                    "예측 단위 수(그 기간에 행이 있는)": int(u_all), "검출이 한 번이라도 있었던 단위 수": int(u_pos),
                    "검출 단위 비율(%)": round(100.0 * u_pos / u_all, 1),
                    "(참고) 시료 수": int(r["n"].sum()), "(참고) 검출 시료 수": int(r["npos"].sum())})
    return pd.DataFrame(out)


T6 = table6()
EXPECT_T6 = {"전체 기간": (291812, 1141, 0.391, 2.15, 6631, 359), "train": (257042, 973, 0.379, 2.11, 6369, 329),
             "test": (34770, 168, 0.483, 2.50, 2627, 83)}
bad = []
for p, exp in EXPECT_T6.items():
    r = T6[T6["기간"] == p].iloc[0]
    got = (int(r["단위x날짜 행 수"]), int(r["단위x날짜 검출 행"]), float(r["단위x날짜 검출률(%)"]), float(r["행 하나의 시료 수 평균"]),
           int(r["예측 단위 수(그 기간에 행이 있는)"]), int(r["검출이 한 번이라도 있었던 단위 수"]))
    if got != exp:
        bad.append(f"{p}: 계산 {got}, 기대 {exp}")
if bad:
    raise SystemExit("표 6 값과 다름(멈춤):\n  " + "\n  ".join(bad))
T6.to_csv(OUT / "eda_u_table6.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 기온(02 와 같은 자료·구간): 기준일 달 시군구 월평균 / 참고 d−30 ~ d−1 평균
w_ = pq.read_table(WX, columns=["sgg_code", "date", "avg_temp"]).to_pandas()
w_["sgg_code"] = w_["sgg_code"].astype(str)
w_["date"] = pd.to_datetime(w_["date"])
w_["ym"] = w_["date"].dt.to_period("M")
MT = w_.groupby(["sgg_code", "ym"])["avg_temp"].mean().rename("t_month").reset_index()
SGGS = sorted(w_["sgg_code"].unique())
DATES = np.array(sorted(w_["date"].unique()), dtype="datetime64[D]")
w_ = w_.sort_values(["sgg_code", "date"])
if len(w_) != len(SGGS) * len(DATES):
    raise SystemExit("기상표가 시군구 × 날짜 격자가 아님")
A = w_["avg_temp"].to_numpy(dtype=np.float64).reshape(len(SGGS), len(DATES))
OK = ~np.isnan(A)
CS = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(np.where(OK, A, 0.0), axis=1)], axis=1)
CN = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(OK, axis=1)], axis=1)
SGG_I = {s: i for i, s in enumerate(SGGS)}
EDGES = [-np.inf, 0, 5, 10, 15, 20, 25, np.inf]
T_LB = ["0 미만", "0~5", "5~10", "10~15", "15~20", "20~25", "25 이상"]


def add_temp(df):
    df = df.merge(MT, on=["sgg_code", "ym"], how="left")
    if df["t_month"].isna().any():
        raise SystemExit(f"월평균 기온이 안 붙은 행 {int(df['t_month'].isna().sum())}개")
    si = df["sgg_code"].map(SGG_I).to_numpy()
    di = (df["d"].to_numpy().astype("datetime64[D]") - DATES[0]).astype(np.int64)
    if (di < 30).any():
        raise SystemExit("기준일 전 30일이 기상표 밖")
    with np.errstate(invalid="ignore", divide="ignore"):
        df["t_prev30"] = (CS[si, di] - CS[si, di - 30]) / (CN[si, di] - CN[si, di - 30])
    return df


S = add_temp(S)
R = add_temp(R)

# ---------------------------------------------------------------- 공통 계산(시료 기준·행 기준 같은 코드, cnt = 1 건/행, det = 검출 시료/검출 행)
BINS = [1, 10, 100, 1000, 10000, np.inf]
BIN_LB = ["1 ~ 9", "10 ~ 99", "100 ~ 999", "1,000 ~ 9,999", "10,000 이상"]


def share(df, key, N, P):
    g = df.groupby(key, sort=True, observed=True).agg(n=("cnt", "sum"), k=("det", "sum")).reset_index()
    g["검출 비중(%)"] = g["k"] / P * 100
    g["행 비중(%)"] = g["n"] / N * 100
    g["검출률(%)"] = g["k"] / g["n"] * 100
    return g


def count_bins(g, N, P):
    g = g.copy()
    g["구간"] = pd.cut(g["n"], BINS, right=False, labels=BIN_LB)
    rows = []
    for lb in BIN_LB:
        s = g[g["구간"] == lb]
        rows.append({"구간": lb, "개수": int(len(s)), "개수 비율(%)": len(s) / len(g) * 100,
                     "n 합": int(s["n"].sum()), "k 합": int(s["k"].sum()),
                     "k 비중(%)": s["k"].sum() / P * 100, "n 비중(%)": s["n"].sum() / N * 100,
                     "검출률 최소(%)": float(s["검출률(%)"].min()) if len(s) else np.nan,
                     "검출률 최대(%)": float(s["검출률(%)"].max()) if len(s) else np.nan,
                     "검출 0 개수": int((s["k"] == 0).sum())})
    b = pd.DataFrame(rows)
    assert b["개수"].sum() == len(g) and b["n 합"].sum() == N and b["k 합"].sum() == P
    return b


def temp_bins(df, col, N, P):
    b = pd.cut(df[col], EDGES, right=False, labels=T_LB)
    g = df.assign(bin=b).groupby("bin", observed=False).agg(n=("cnt", "sum"), k=("det", "sum")).reset_index()
    g["검출률(%)"] = g["k"] / g["n"] * 100
    g["검출 비중(%)"] = g["k"] / P * 100
    g["행 비중(%)"] = g["n"] / N * 100
    return g


def facts(df):
    N, P = int(df["cnt"].sum()), int(df["det"].sum())
    F, T = {"N": N, "P": P}, {}
    # 1) 연도
    yr = share(df, "year", N, P)
    T["year"] = yr
    F["year_max_det"] = int(yr.loc[yr["k"].idxmax(), "year"])
    F["year_max_det_k"] = int(yr["k"].max())
    full = yr[(yr["year"] >= 2015) & (yr["year"] <= 2025)]
    F["year_n_up_every_year_2015_2025"] = bool(np.all(np.diff(full["n"].to_numpy()) > 0))
    # 2) 품목 3단계
    it = share(df, "p3", N, P)
    it["label"] = it["p3"].map(item_label)
    it = it.sort_values(["k", "p3"], ascending=[False, True]).reset_index(drop=True)
    it["순위"] = np.arange(1, len(it) + 1)
    T["item"] = it
    k_of = dict(zip(it["p3"], it["k"]))
    n_of = dict(zip(it["p3"], it["n"]))
    F["top2_items"] = [item_label(c) for c in it["p3"].head(2)]
    F["top2_share"] = float(it["k"].head(2).sum() / P * 100)
    F["z38_z27_det_share"] = float((k_of[Z38] + k_of[Z27]) / P * 100)
    F["top10_det_share"] = float(it["k"].head(10).sum() / P * 100)
    F["n_items_with_det"] = int((it["k"] > 0).sum())
    z27 = df[df["p3"] == Z27]
    F["z27_det"] = int(z27["det"].sum())
    F["z27_det_enoki_any"] = int(((z27["det"] == 1) & (z27["enoki"] == 1)).sum())       # (가) 행 안 시료 중 하나라도 '팽이'
    F["z27_det_enoki_detsample"] = int(((z27["det"] == 1) & (z27["enoki_det"] == 1)).sum())  # (나) 행 안 검출 시료 중 '팽이'
    F["z27_n"] = int(z27["cnt"].sum())
    F["z27_n_enoki_any"] = int(z27["enoki"].sum())
    F["z38_n"] = int(n_of[Z38])
    F["z38_n_share"] = float(n_of[Z38] / N * 100)
    F["z38_det"] = int(k_of[Z38])
    # 3) 월
    mo = share(df, "month", N, P)
    T["month"] = mo
    F["months_det_share_gt_n_share"] = [int(m) for m in mo.loc[mo["검출 비중(%)"] > mo["행 비중(%)"], "month"]]
    # 4) 시도
    sd = share(df, "sido", N, P)
    sd["시도"] = sd["sido"].map(SIDO_NM)
    sd["약칭"] = sd["시도"].map(SHORT)
    assert sd["약칭"].notna().all(), sd
    sd = sd.sort_values(["k", "n"], ascending=[False, False]).reset_index(drop=True)
    T["sido"] = sd
    F["sido_top2"] = list(sd["약칭"].head(2))
    F["sido_41_47_det_share"] = float(sd.loc[sd["sido"].isin(["41", "47"]), "k"].sum() / P * 100)
    # 5) 기온 구간
    tb = temp_bins(df, "t_month", N, P)
    T["temp_month"] = tb
    T["temp_prev30"] = temp_bins(df, "t_prev30", N, P)
    rate = dict(zip(tb["bin"].astype(str), tb["검출률(%)"]))
    F["temp_rate"] = {k: float(v) for k, v in rate.items()}
    F["temp_rate_monotonic_up"] = bool(np.all(np.diff(tb["검출률(%)"].to_numpy()) > 0))
    zmask = df["p2"].astype(str).str.startswith("Z")
    sub = []
    for nm, m in (("Z로 시작하는 2단계 코드", zmask), ("그 밖의 2단계 코드(코드없음 포함)", ~zmask)):
        g = temp_bins(df[m], "t_month", N, P)
        for r in g.itertuples(index=False):
            sub.append({"품목 묶음": nm, "기온 구간(°C)": str(r.bin), "n": int(r.n), "k": int(r.k),
                        "검출률(%)": (r.k / r.n * 100) if r.n else np.nan})
    T["temp_month_by_Z"] = pd.DataFrame(sub)
    # 6) 품목 3단계의 n(검사 수 / 행 수) 구간
    ib = count_bins(it, N, P)
    T["item_bins"] = ib
    F["n_items"] = int(len(it))
    F["items_n_lt100"] = int(ib["개수"].iloc[:2].sum())
    F["items_n_lt100_pct"] = float(ib["개수"].iloc[:2].sum() / len(it) * 100)
    F["items_n_ge10000"] = int(ib["개수"].iloc[4])
    F["items_n_ge10000_n_share"] = float(ib["n 비중(%)"].iloc[4])
    F["items_n_ge10000_k_share"] = float(ib["k 비중(%)"].iloc[4])
    F["items_empty_bins"] = [lb for lb, c in zip(ib["구간"], ib["개수"]) if c == 0]
    F["item_n_min"], F["item_n_max"] = int(it["n"].min()), int(it["n"].max())
    F["item_n_max_label"] = item_label(it.loc[it["n"].idxmax(), "p3"])
    e = it[it["p3"] == EOMNAMU].iloc[0]
    F["eomnamu"] = {"label": item_label(EOMNAMU), "n": int(e["n"]), "k": int(e["k"]), "rate": float(e["검출률(%)"]),
                    "rate_if_k_minus_1": float((e["k"] - 1) / e["n"] * 100)}
    # 검사(행)가 적은데 검출이 있는 품목(예시 후보): n < 100, k >= 1
    few = it[(it["n"] < 100) & (it["k"] >= 1)].sort_values("n")
    F["few_n_items_with_det"] = [{"label": lb, "n": int(n), "k": int(k), "rate": float(k / n * 100),
                                  "rate_if_k_minus_1": float((k - 1) / n * 100)}
                                 for lb, n, k in zip(few["label"], few["n"], few["k"])]
    # 7) 예측 단위의 n 구간
    un = share(df, "unit", N, P)
    ub = count_bins(un, N, P)
    T["unit_bins"] = ub
    F["n_units"] = int(len(un))
    F["units_n_1_9"] = int(ub["개수"].iloc[0])
    F["units_n_1_9_pct"] = float(ub["개수 비율(%)"].iloc[0])
    F["unit_n_median"] = float(un["n"].median())
    F["unit_n_max"] = int(un["n"].max())
    # 8) 검출 집중(단위)
    un = un.sort_values(["k", "n"], ascending=[False, False]).reset_index(drop=True)
    un["누적 검출 비중(%)"] = un["k"].cumsum() / P * 100
    sg, p2, p3 = zip(*[u.split("|") for u in un["unit"]])
    un["sgg_code"], un["p2"], un["p3"] = sg, p2, p3
    un["시군구"] = un["sgg_code"].map(SGG_NM)
    un["품목"] = un["p3"].map(item_label)
    T["units_sorted"] = un
    F["units_with_det"] = int((un["k"] > 0).sum())
    F["units_top_n_for_50pct"] = int((un["누적 검출 비중(%)"] < 50).sum() + 1)
    t1 = un.iloc[0]
    F["top1_unit"] = {"시군구": t1["시군구"], "sgg_code": t1["sgg_code"], "품목": t1["품목"], "n": int(t1["n"]), "k": int(t1["k"]),
                      "share": float(t1["k"] / P * 100)}
    return F, T


FS_, TS_ = facts(S)     # 시료 기준(원고 지금 값 재현용)
FR_, TR_ = facts(R)     # 행 기준(새 값)

# 시료 기준이 원고 지금 숫자와 같은지(정의가 기존 스크립트와 같다는 확인)
EXPECT_SAMPLE = [
    ("2020년 검출 최대", FS_["year_max_det"], 2020),
    ("검사 시료 2015~2025 해마다 늘어남", FS_["year_n_up_every_year_2015_2025"], True),
    ("두 품목 검출 77.6%", r1(FS_["z38_z27_det_share"]), 77.6),
    ("두 품목 = 상위 2", sorted(FS_["top2_items"]), sorted(["기타기준규격외", "정액,수정란,종란 > 기타"])),
    ("정액 기타 검출 640", FS_["z27_det"], 640),
    ("그중 팽이 254", FS_["z27_det_enoki_any"], 254),
    ("기타기준규격외 291,571건", FS_["z38_n"], 291571),
    ("기타기준규격외 46.4%", r1(FS_["z38_n_share"]), 46.4),
    ("검출 비중 > 검사 비중 달", FS_["months_det_share_gt_n_share"], [7, 8, 9, 11, 12]),
    ("경기·경북 55.0%", r1(FS_["sido_41_47_det_share"]), 55.0),
    ("25°C 이상 0.37%", r2(FS_["temp_rate"]["25 이상"]), 0.37),
    ("10~15°C 0.20%", r2(FS_["temp_rate"]["10~15"]), 0.20),
    ("0~5°C 0.31%", r2(FS_["temp_rate"]["0~5"]), 0.31),
    ("품목 218개", FS_["n_items"], 218),
    ("100건 미만 131개", FS_["items_n_lt100"], 131),
    ("100건 미만 60.1%", r1(FS_["items_n_lt100_pct"]), 60.1),
    ("1만 건 이상 10개", FS_["items_n_ge10000"], 10),
    ("1만 건 이상 검사 84.5%", r1(FS_["items_n_ge10000_n_share"]), 84.5),
    ("1만 건 이상 검출 90.1%", r1(FS_["items_n_ge10000_k_share"]), 90.1),
    ("품목 검사 최소 1", FS_["item_n_min"], 1),
    ("품목 검사 최대 291,571", FS_["item_n_max"], 291571),
    ("엄나무 32건·2건", (FS_["eomnamu"]["n"], FS_["eomnamu"]["k"]), (32, 2)),
    ("단위 6,631개", FS_["n_units"], 6631),
    ("단위 1~9건 56.0%", r1(FS_["units_n_1_9_pct"]), 56.0),
    ("상위 21개 단위", FS_["units_top_n_for_50pct"], 21),
    ("1위 단위 청도군", FS_["top1_unit"]["sgg_code"], "47820"),
    ("1위 단위 품목", FS_["top1_unit"]["품목"], "정액,수정란,종란 > 기타"),
    ("1위 단위 17.5%", r1(FS_["top1_unit"]["share"]), 17.5),
]
bad = [f"{nm}: 계산 {got}, 원고 {exp}" for nm, got, exp in EXPECT_SAMPLE if got != exp]
if bad:
    raise SystemExit("시료 기준 재계산이 원고 숫자와 다름(정의가 기존 스크립트와 다를 수 있어 멈춤):\n  " + "\n  ".join(bad))
print("시료 기준 재계산: 원고 숫자 %d개 모두 같음" % len(EXPECT_SAMPLE))
if (FR_["N"], FR_["P"]) != (291812, 1141):
    raise SystemExit(f"행 수가 다름: {FR_['N']}·{FR_['P']}")

# ---------------------------------------------------------------- 행 기준 표 저장(CSV)
REN = {"n": "행 수", "k": "검출 행 수"}
yr = TR_["year"].rename(columns=REN)
yr = yr.merge(R.groupby("year").agg(**{"(참고) 시료 수": ("n", "sum"), "(참고) 검출 시료 수": ("npos", "sum")}).reset_index(), on="year")
yr.to_csv(OUT / "eda_u_year.csv", index=False, encoding="utf-8-sig")
it = TR_["item"].copy()
it = it.join(NAMES, on="p3")
it = it.merge(R.groupby("p3").agg(**{"(참고) 시료 수": ("n", "sum"), "(참고) 검출 시료 수": ("npos", "sum")}).reset_index(), on="p3")
it = it.sort_values("순위").rename(columns=REN)
it.to_csv(OUT / "eda_u_item3.csv", index=False, encoding="utf-8-sig")
TR_["month"].rename(columns=REN).to_csv(OUT / "eda_u_month.csv", index=False, encoding="utf-8-sig")
TR_["sido"].rename(columns=REN).to_csv(OUT / "eda_u_sido.csv", index=False, encoding="utf-8-sig")
for k_, f_ in (("temp_month", "eda_u_temp_month_bins.csv"), ("temp_prev30", "eda_u_temp_prev30_bins.csv")):
    TR_[k_].rename(columns={"bin": "기온 구간(°C, 이상~미만)", **REN}).to_csv(OUT / f_, index=False, encoding="utf-8-sig")
TR_["temp_month_by_Z"].rename(columns=REN).to_csv(OUT / "eda_u_temp_month_bins_by_Z.csv", index=False, encoding="utf-8-sig")
BIN_REN = {"구간": "행 수 구간(행)", "n 합": "행 수 합", "k 합": "검출 행 수 합", "k 비중(%)": "검출 비중(%)", "n 비중(%)": "행 비중(%)"}
TR_["item_bins"].rename(columns={**BIN_REN, "개수": "품목 수", "개수 비율(%)": "품목 비율(%)", "검출률 최소(%)": "품목별 검출률 최소(%)",
                                 "검출률 최대(%)": "품목별 검출률 최대(%)", "검출 0 개수": "검출 행 0 품목 수"}) \
    .to_csv(OUT / "eda_u_item3_row_bins.csv", index=False, encoding="utf-8-sig")
TR_["unit_bins"].rename(columns={**BIN_REN, "개수": "단위 수", "개수 비율(%)": "단위 비율(%)", "검출률 최소(%)": "단위별 검출률 최소(%)",
                                 "검출률 최대(%)": "단위별 검출률 최대(%)", "검출 0 개수": "검출 행 0 단위 수"}) \
    .to_csv(OUT / "eda_u_unit_row_bins.csv", index=False, encoding="utf-8-sig")
TR_["units_sorted"].head(40)[["시군구", "sgg_code", "품목", "p2", "p3", "n", "k", "누적 검출 비중(%)"]].rename(columns=REN) \
    .to_csv(OUT / "eda_u_top_units.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 그림 2(연도별): 검출 행 수(위)·전체 행 수(아래)
yr = TR_["year"]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.0), sharex=True, gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.18})
x = np.arange(len(yr))
a1.bar(x, yr["k"], width=0.62, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
for xi, v in zip(x, yr["k"]):
    a1.text(xi, v + max(yr["k"]) * 0.03, f"{v:,}", ha="center", va="bottom", fontsize=FS, color=INK)
a1.set_ylabel("검출 행 수")
a1.set_ylim(0, max(yr["k"]) * 1.22)
style(a1)
a2.bar(x, yr["n"] / 1000, width=0.62, color=REF, edgecolor="white", linewidth=1.0, zorder=3)
for xi, v in zip(x, yr["n"]):
    a2.text(xi, v / 1000 + max(yr["n"]) / 1000 * 0.03, f"{v / 1000:.1f}" if v >= 1000 else f"{v}행", ha="center", va="bottom",
            fontsize=FS, color=MUTED)
a2.set_ylabel("전체 행 수\n(천 행)")
a2.set_ylim(0, max(yr["n"]) / 1000 * 1.3)
a2.set_xticks(x, [str(v) for v in yr["year"]])
style(a2)
fig.align_ylabels([a1, a2])
fig.subplots_adjust(left=0.115, right=0.99, top=0.98, bottom=0.09)
FIGS = {"I_1u": save(fig, "year")}

# ---------------------------------------------------------------- 그림 3(품목 2단계 > 3단계 검출 비중 상위 10)
top = TR_["item"].head(10).copy()
fig, ax = plt.subplots(figsize=(W, 3.5))
yy = np.arange(len(top))[::-1]
hgt = 0.38
ax.barh(yy + hgt / 2, top["검출 비중(%)"], height=hgt, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3, label=LEG_DET)
ax.barh(yy - hgt / 2, top["행 비중(%)"], height=hgt, color=REF, edgecolor="white", linewidth=0.8, zorder=3, label=LEG_ROW)
xmax = max(top["검출 비중(%)"].max(), top["행 비중(%)"].max())
# 위아래로 쌓인 짝 꼬리표(검출 비중 위·행 비중 아래)는 막대 가운데에 그대로 두면 '(44행)' 의 괄호 아래 끝과 아래 줄
# '9.3%' 의 윗부분 획이 닿음(10-22 잉크 점검: 3쌍 겹침, 회색 꼬리표 2개는 위 파란 막대에 붙음) -> 위 꼬리표는 PAIR_NUDGE_PT 만큼 위로,
# 아래 꼬리표는 같은 만큼 아래로 옮김(막대 가운데에서 1pt 씩만 벗어남, 나머지 모양은 02 와 같음)
PAIR_NUDGE_PT = 1.0
for yi, v, n in zip(yy, top["검출 비중(%)"], top["k"]):
    ax.annotate(f"{v:.1f}% ({n:,}행)", (v + xmax * 0.012, yi + hgt / 2), xytext=(0, PAIR_NUDGE_PT), textcoords="offset points",
                va="center", fontsize=FS, color=INK)
for yi, v in zip(yy, top["행 비중(%)"]):
    ax.annotate(f"{v:.2f}%" if v < 0.1 else f"{v:.1f}%", (v + xmax * 0.012, yi - hgt / 2), xytext=(0, -PAIR_NUDGE_PT),
                textcoords="offset points", va="center", fontsize=FS, color=MUTED)
ax.set_yticks(yy, top["label"])
ax.set_xlim(0, xmax * 1.3)
ax.set_xlabel("비중(%)")
ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2)
style(ax, "x")
fig.subplots_adjust(left=0.375, right=0.99, top=0.99, bottom=0.125)
# 가장 긴 막대의 꼬리표('49.1% (560행)' 등)가 판 오른쪽 끝을 넘지 않게 가로 눈금 끝을 조금씩 늘림(02 의 1.3 배에서 시작)
fig.set_dpi(DPI)
for _ in range(20):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    if max(t.get_window_extent(rend).x1 for t in ax.texts) <= ax.get_window_extent(rend).x1 - 2:
        break
    ax.set_xlim(0, ax.get_xlim()[1] * 1.02)
FIGS["I_2u"] = save(fig, "item")

# ---------------------------------------------------------------- 그림 4(월별: 검출 비중 vs 행 비중)
mo = TR_["month"]
fig, ax = plt.subplots(figsize=(W, 2.35))
x = np.arange(12)
w = 0.38
ax.bar(x - w / 2, mo["검출 비중(%)"], width=w, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3, label=LEG_DET)
ax.bar(x + w / 2, mo["행 비중(%)"], width=w, color=REF, edgecolor="white", linewidth=0.8, zorder=3, label=LEG_ROW)
ymax = max(mo["검출 비중(%)"].max(), mo["행 비중(%)"].max())
for xi, v in zip(x, mo["검출 비중(%)"]):
    ax.text(xi - w / 2, v + ymax * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=FS, color=INK)
ax.set_xticks(x, [f"{m}월" for m in range(1, 13)])
ax.set_ylim(0, ymax * 1.32)
ax.set_ylabel("비중(%)")
ax.legend(loc="upper left", ncol=2, handlelength=1.2, borderaxespad=0.1)
style(ax)
fig.subplots_adjust(left=0.085, right=0.99, top=0.98, bottom=0.125)
FIGS["I_4u"] = save(fig, "month")


# ---------------------------------------------------------------- 그림 5(품목 3단계의 행 수 구간별 품목 수 / 검출·행 비중)
def fmt(v):
    """비중 꼬리표: 0 은 '0', 0.1 미만은 소수 둘째 자리, 나머지는 첫째 자리(11 과 같음)."""
    if v == 0:
        return "0"
    return f"{v:.2f}" if v < 0.1 else f"{v:.1f}"


ib = TR_["item_bins"]
N_ITEM = int(ib["개수"].sum())
n_item = ib["개수"].to_numpy(dtype=int)
item_pct = n_item / N_ITEM * 100
det = ib["k 비중(%)"].to_numpy(dtype=float)
rws = ib["n 비중(%)"].to_numpy(dtype=float)
TICK = ["1 ~ 9", "10 ~ 99", "100 ~\n999", "1,000 ~\n9,999", "10,000\n이상"]
XLABEL = "품목의 행 수(행)"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.22})
x = np.arange(len(BIN_LB))
a1.bar(x, n_item, width=0.62, color=NEUTRAL, edgecolor="white", linewidth=0.8, zorder=3)
ymax1 = float(n_item.max())
for xi, n, p in zip(x, n_item, item_pct):
    a1.text(xi, n + ymax1 * 0.03, f"{n}개\n({p:.1f}%)", ha="center", va="bottom", fontsize=FS, color=INK)
a1.set_ylim(0, ymax1 * 1.3)
a1.set_xlim(-0.6, len(BIN_LB) - 0.4)
a1.set_xticks(x, TICK)
a1.set_ylabel(f"품목 수\n(개, 괄호 = 전체 {N_ITEM}개 중 %)")
a1.set_xlabel(XLABEL)
style(a1)
w = 0.4
bars_det = a2.bar(x - w / 2, det, width=w, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3, label=LEG_DET)
bars_row = a2.bar(x + w / 2, rws, width=w, color=REF, edgecolor="white", linewidth=0.8, zorder=3, label=LEG_ROW)
ymax2 = float(max(det.max(), rws.max()))
lab_d = [a2.annotate(fmt(v), (xi - w / 2, v + ymax2 * 0.02), xytext=(0, 0), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS, color=INK) for xi, v in zip(x, det)]
lab_i = [a2.annotate(fmt(v), (xi + w / 2, v + ymax2 * 0.02), xytext=(0, 0), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS, color=MUTED) for xi, v in zip(x, rws)]
a2.set_ylim(0, ymax2 * 1.16)
a2.set_xlim(-0.6, len(BIN_LB) - 0.4)
a2.set_xticks(x, TICK)
a2.set_ylabel("비중(%)")
a2.set_xlabel(XLABEL)
a2.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)
style(a2)
fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.255)
fig.set_dpi(DPI)
# 범례 글이 길어져('… 검출 행 중 %') 가장 큰 막대 꼬리표와 범례 상자가 닿으면 세로 눈금 끝을 조금씩 늘림(11 의 1.16 배에서 시작)
for _ in range(30):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lb = a2.get_legend().get_window_extent(rend)
    hit = False
    for t in lab_d + lab_i:
        bb = t.get_window_extent(rend)
        if min(lb.x1, bb.x1) - max(lb.x0, bb.x0) > 0 and min(lb.y1, bb.y1) - max(lb.y0, bb.y0) > -2:
            hit = True
            break
    if not hit:
        break
    a2.set_ylim(0, a2.get_ylim()[1] * 1.03)
# 짝 막대의 두 꼬리표가 붙지 않게 비킴(11 과 같은 규칙)
MIN_GAP_PT = 3.5
fig.canvas.draw()
rend = fig.canvas.get_renderer()
px_per_pt = DPI / 72
for td, ti in zip(lab_d, lab_i):
    b1, b2 = td.get_window_extent(rend), ti.get_window_extent(rend)
    if min(b1.y1, b2.y1) - max(b1.y0, b2.y0) <= 0:
        continue
    gap_pt = (b2.x0 - b1.x1) / px_per_pt
    if gap_pt < MIN_GAP_PT:
        s = (MIN_GAP_PT - gap_pt) / 2
        td.xyann = (-s, 0)
        ti.xyann = (s, 0)
# 꼬리표가 옆(다른 계열)의 더 높은 막대에 붙지 않게 바깥쪽으로 비킴: 행 비중 꼬리표는 왼쪽 검출 막대에서 오른쪽으로,
# 검출 비중 꼬리표는 오른쪽 행 비중 막대에서 왼쪽으로 BAR_GAP_PT 이상 띄움(10-22 잉크 점검: '19.2' 가 왼쪽 '29.0' 막대에 붙음)
BAR_GAP_PT = 1.5
fig.canvas.draw()
rend = fig.canvas.get_renderer()
for td, ti, bd, bi in zip(lab_d, lab_i, bars_det.patches, bars_row.patches):
    tb, bb = ti.get_window_extent(rend), bd.get_window_extent(rend)
    gap_pt = (tb.x0 - bb.x1) / px_per_pt
    if tb.y0 < bb.y1 and gap_pt < BAR_GAP_PT:
        ti.xyann = (ti.xyann[0] + BAR_GAP_PT - gap_pt, ti.xyann[1])
    tb, bb = td.get_window_extent(rend), bi.get_window_extent(rend)
    gap_pt = (bb.x0 - tb.x1) / px_per_pt
    if tb.y0 < bb.y1 and gap_pt < BAR_GAP_PT:
        td.xyann = (td.xyann[0] - (BAR_GAP_PT - gap_pt), td.xyann[1])
FIGS["I_5u"] = save(fig, "bins")

# ---------------------------------------------------------------- 항목 1)~9) 시료 기준(원고) → 행 기준(새 값) 대조표
def pct(v, nd=1):
    return f"{v:.{nd}f}%"


tr_s, tr_r = FS_["temp_rate"], FR_["temp_rate"]
few = FR_["few_n_items_with_det"]
CMP = [
    ("1) 그림 2", "검출 최대 해", f"{FS_['year_max_det']}년 {FS_['year_max_det_k']:,}건", f"{FR_['year_max_det']}년 {FR_['year_max_det_k']:,}행"),
    ("1) 그림 2", "2015~2025 해마다 늘어남(검사 시료 → 행)", str(FS_["year_n_up_every_year_2015_2025"]), str(FR_["year_n_up_every_year_2015_2025"])),
    ("2) 품목", "상위 2 품목", " / ".join(FS_["top2_items"]), " / ".join(FR_["top2_items"])),
    ("2) 품목", "기타기준규격외 + 정액,수정란,종란 > 기타 검출 비중", pct(FS_["z38_z27_det_share"]), pct(FR_["z38_z27_det_share"])),
    ("2) 품목", "정액,수정란,종란 > 기타 검출", f"{FS_['z27_det']:,}건", f"{FR_['z27_det']:,}행"),
    ("2) 품목", "그중 제품명에 '팽이'", f"{FS_['z27_det_enoki_any']:,}건",
     f"(가) 행 안 시료 중 하나라도 {FR_['z27_det_enoki_any']:,}행 / (나) 행 안 검출 시료 중 {FR_['z27_det_enoki_detsample']:,}행"),
    ("2) 품목", "기타기준규격외 몫", f"{pct(FS_['z38_n_share'])}({FS_['N']:,}건 중 {FS_['z38_n']:,}건)",
     f"{pct(FR_['z38_n_share'])}({FR_['N']:,}행 중 {FR_['z38_n']:,}행)"),
    ("3) 월", "검출 비중 > 검사(행) 비중인 달", ",".join(map(str, FS_["months_det_share_gt_n_share"])),
     ",".join(map(str, FR_["months_det_share_gt_n_share"]))),
    ("4) 시도", "경기·경북 검출 비중 합", pct(FS_["sido_41_47_det_share"]), pct(FR_["sido_41_47_det_share"]) + f"(상위 2 = {'·'.join(FR_['sido_top2'])})"),
    ("5) 기온", "25°C 이상 / 10~15°C / 0~5°C 검출률",
     f"{tr_s['25 이상']:.2f}% / {tr_s['10~15']:.2f}% / {tr_s['0~5']:.2f}%", f"{tr_r['25 이상']:.2f}% / {tr_r['10~15']:.2f}% / {tr_r['0~5']:.2f}%"),
    ("5) 기온", "구간 따라 한 방향으로 늘어남", str(FS_["temp_rate_monotonic_up"]), str(FR_["temp_rate_monotonic_up"])),
    ("6) 그림 5", "100 미만 품목", f"{FS_['items_n_lt100']}개({FS_['items_n_lt100_pct']:.1f}%, 검사 100건 미만)",
     f"{FR_['items_n_lt100']}개({FR_['items_n_lt100_pct']:.1f}%, 행 100개 미만)"),
    ("6) 그림 5", "1만 이상 품목 수·검사(행) 비중·검출 비중",
     f"{FS_['items_n_ge10000']}개·{FS_['items_n_ge10000_n_share']:.1f}%·{FS_['items_n_ge10000_k_share']:.1f}%",
     f"{FR_['items_n_ge10000']}개·{FR_['items_n_ge10000_n_share']:.1f}%·{FR_['items_n_ge10000_k_share']:.1f}%"),
    ("6) 그림 5", "품목마다 검사 수(행 수) 범위", f"{FS_['item_n_min']:,} ~ {FS_['item_n_max']:,}건", f"{FR_['item_n_min']:,} ~ {FR_['item_n_max']:,}행"),
    ("6) 그림 5", "엄나무 예시", f"검사 {FS_['eomnamu']['n']}건 중 검출 {FS_['eomnamu']['k']}건 = {FS_['eomnamu']['rate']:.2f}%(1건이면 {FS_['eomnamu']['rate_if_k_minus_1']:.2f}%)",
     f"행 {FR_['eomnamu']['n']}개 중 검출 행 {FR_['eomnamu']['k']}개 = {FR_['eomnamu']['rate']:.2f}%(검출 행이 없으면 {FR_['eomnamu']['rate_if_k_minus_1']:.2f}%)"),
    ("7) 단위", "1 ~ 9 인 단위 비율", f"{FS_['units_n_1_9_pct']:.1f}%({FS_['units_n_1_9']:,}개, 검사 1 ~ 9건)",
     f"{FR_['units_n_1_9_pct']:.1f}%({FR_['units_n_1_9']:,}개, 행 1 ~ 9개)"),
    ("8) 집중", "검출 50% 이상이 나온 상위 단위 수", f"{FS_['units_top_n_for_50pct']}개", f"{FR_['units_top_n_for_50pct']}개"),
    ("8) 집중", "1위 단위·비중", f"{FS_['top1_unit']['시군구']} × {FS_['top1_unit']['품목']} {FS_['top1_unit']['share']:.1f}%",
     f"{FR_['top1_unit']['시군구']} × {FR_['top1_unit']['품목']} {FR_['top1_unit']['share']:.1f}%"),
]
cmp_df = pd.DataFrame(CMP, columns=["항목", "내용", "시료 기준(원고 지금 값)", "행 기준(새 값)"])
cmp_df.to_csv(OUT / "eda_u_compare.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 요약 JSON
def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


summary = {
    "기준": "단위 × 날짜 행(예측 단위 = sgg_code × train_prdlst_code × train_prdlst_code_2, 빈 코드 '코드없음') × 기준일 mnfctur_de, "
            "y = 그 행 시료 중 검출 1건 이상, 기준일 전체 기간",
    "실험 build_unit_rows 대조": EXP1_CHECK,
    "표 6(단위 × 날짜·예측 단위 줄)": T6.to_dict("records"),
    "시료 기준 재계산 = 원고 숫자": len(EXPECT_SAMPLE),
    "행 기준": {k: v for k, v in FR_.items()},
    "시료 기준": {k: v for k, v in FS_.items() if k != "few_n_items_with_det"},
    "대조표": cmp_df.to_dict("records"),
    "그림": FIGS,
    "그림 글자 점검(잘림·겹침, 저장 해상도 상자 기준)": LAYOUT,
    "그림 글자 점검 참고(쌓인 꼬리표 줄 상자만 닿음)": STACKED,
    "표 파일": sorted(p.name for p in OUT.glob("eda_u_*.csv")),
}
(OUT / "eda_unit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=jsonable), encoding="utf-8")

# ---------------------------------------------------------------- 화면
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 80)
print("실험 build_unit_rows 대조:", EXP1_CHECK)
print(T6.T.to_string())
print("== 대조표 ==")
print(cmp_df.to_string(index=False))
print("== 연도 ==")
print(TR_["year"].to_string(index=False))
print("== 품목 상위 12 ==")
print(TR_["item"].head(12)[["순위", "label", "n", "k", "검출 비중(%)", "행 비중(%)", "검출률(%)"]].to_string(index=False))
print("== 월 ==")
print(TR_["month"].to_string(index=False))
print("== 시도 ==")
print(TR_["sido"][["약칭", "n", "k", "검출 비중(%)", "행 비중(%)", "검출률(%)"]].to_string(index=False))
print("== 기온(기준일 달 시군구 월평균) ==")
print(TR_["temp_month"].to_string(index=False))
print("== 기온(d−30 ~ d−1, 참고) ==")
print(TR_["temp_prev30"].to_string(index=False))
print("== 기온 × Z 묶음(참고) ==")
print(TR_["temp_month_by_Z"].to_string(index=False))
print("== 품목 행 수 구간 ==")
print(TR_["item_bins"].to_string(index=False))
print("빈 구간:", FR_["items_empty_bins"])
print("== 단위 행 수 구간 ==")
print(TR_["unit_bins"].to_string(index=False))
print("== 행 100개 미만·검출 있는 품목(예시 후보) ==")
for r in FR_["few_n_items_with_det"]:
    print("  ", r)
print("== 검출 행 상위 단위 25 ==")
print(TR_["units_sorted"].head(25)[["시군구", "품목", "n", "k", "누적 검출 비중(%)"]].to_string(index=False))
print("저장:", OUT)
