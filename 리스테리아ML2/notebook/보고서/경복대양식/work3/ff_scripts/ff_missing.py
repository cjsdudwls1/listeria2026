# -*- coding: utf-8 -*-
"""3차 실험 입력 칸의 결측 비율을 로컬 자료 파일로 다시 센다(읽기만, 결과는 work3 아래).

목적: 보고서 'II.4 추가 데이터와 특성공학'·부록 '입력 변수 기록표'의 '결측과 이유' 칸에 넣을 3차 기준 수치.
실험 코드(exp/3_시간순교차검증/src, exp/2_고도화/src, exp/1_베이스라인_lims기상/src)는 고치지도 부르지도 않고,
그 코드의 결측 규칙(아래)을 그대로 옮겨 센다. 값을 만들지 않고 '결측인지'만 센다.

행 묶음(사양_20261020.md 1절, final_cat/config.json bounds 와 같음)
  - 단위 x 날짜 행 = 시료_시군구.parquet 의 (sgg_code, train_prdlst_code('코드없음'), train_prdlst_code_2('코드없음'), mnfctur_de 날짜) 고유 행,
    정렬 = 단위 키 + d (1차 data.build_unit_rows 와 같음). y = 검출 시료 1건 이상.
  - 최종 학습 행 = d <= 2025-06-30, 행마다 h 하나: d <= 2024-06-30 행 = default_rng(42).integers(1, 61), 그 뒤 = default_rng(43) (cv._h_assign 과 같은 순서)
  - 겹 k 맞혀 보는 행 = (행, h) 중 T = d − h 가 [t_start, t_end] (h = 1..60), 학습 끝 값 고정 anchor = 겹 학습 끝
  - 시험 행 = T ∈ [2025-07-01, 2026-05-01], anchor = 2025-06-30. 시험 행의 정답(judge_result)은 쓰지 않는다(행 수만).
결측 규칙(코드 그대로)
  - 기상 요약·SPI 요약(창 T−80 … T−1): 10일 구간 8개 중 관측 0인 구간의 구간 평균 = 결측, 창 전체 관측 0 이면 평균·표준편차·최소·최대 결측, 관측 < 2 면 기울기 결측
  - 평균·최근 10일 평균·최대·합·양수일 수 요약(생장 범위 기상·생장식·이동평균 차감): 창(또는 최근 10일) 관측 0 이면 결측
  - 이동평균 차감 a_t: x_t 결측이거나 앞 30일 관측 0 이면 결측
  - 날짜 T 행 값(토양 중앙값·생장 범위 토양 pH): 시군구_일별_토양 / 생장범위_일별_토양pH 의 (시군구, T) 칸이 결측이면 결측
  - 친환경·지하수: 사용가능일 <= T−1 인 가장 최근 자료연도 값, 없거나 빈칸이면 결측
  - HACCP: 단위 2단계 코드가 업종묶음 코드 목록에 없음, T < 사용시작일(2017-08-03), 그날 전국 수 0 이면 결측
  - 식품 속성·생장 범위 식품: 단위 3단계 코드로 붙임, 표에 값 없으면 결측(aw_label 은 '결측' 범주 - 숫자 결측과 같은 행)
  - 생장식 G2: 3단계 aw·pH 중 하나라도 없으면 결측(9칸 모두), G1·G3: 창 안 기온·2일 전 강수 관측 0 이면 결측
  - 과거 이력 스무딩 검출률·1년 불확실성: 같은 창 전국 시료 0건이면 결측. log 검사 수·WOE: 결측 없음
  - 제품명 과거 유형 N1: 비율 = 창 안 그 단위 시료 0건이면 결측, 스무딩 검출률 = 같은 창 전국 그 뭉치 시료 0건이면 결측
  - 그날 제품명 N2·TF-IDF N0·이상치 점수: 결측 없음(N0 파일 nan 칸 0 - results/nb_preflight.json)
실행: <리스테리아ML>/.venv-exp2/Scripts/python.exe -I ff_missing.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
T0 = time.time()
BASE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2")
DS = BASE / "data" / "데이터셋"
OUT = BASE / "notebook" / "보고서" / "경복대양식" / "work3"
NULL = "코드없음"
KEY = ["sgg_code", "train_prdlst_code", "train_prdlst_code_2"]
L, BLOCK, H_MAX, K_ANOM = 80, 10, 60, 30
P5, P1 = 1826, 365
WV = ["avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "max_wind", "sunshine", "solar_rad", "soil_t5", "soil_t10",
      "soil_t20", "soil_t30", "soil_t_nas", "soil_moist", "no2", "co", "so2", "wind_dir_sin", "wind_dir_cos"]
WV18 = [v for v in WV if v != "co"]
FOLDS = [(1, "2022-06-30", "2022-07-01", "2023-05-01"), (2, "2023-06-30", "2023-07-01", "2024-05-01"),
         (3, "2024-06-30", "2024-07-01", "2025-05-01")]
TEST = ("2025-06-30", "2025-07-01", "2026-05-01")
LEARN_END = np.datetime64("2025-06-30", "D")
LAST_TE = np.datetime64("2024-06-30", "D")


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def D(x):
    return np.datetime64(x, "D")


# ---------------------------------------------------------------- 1. 단위 x 날짜 행
S = pq.read_table(DS / "시료_시군구.parquet", columns=KEY + ["mnfctur_de", "judge_result"]).to_pandas()
samples = pd.DataFrame({
    "sgg_code": S["sgg_code"].astype(str),
    "train_prdlst_code": S["train_prdlst_code"].astype(object).where(S["train_prdlst_code"].notna(), NULL),
    "train_prdlst_code_2": S["train_prdlst_code_2"].astype(object).where(S["train_prdlst_code_2"].notna(), NULL),
    "d": S["mnfctur_de"].values.astype("datetime64[D]"),
    "pos": (S["judge_result"].astype(object) == "검출").to_numpy().astype(np.int8),
})
rows = (samples.groupby(KEY + ["d"], sort=False, observed=True).agg(n_samples=("pos", "size"), n_pos=("pos", "sum")).reset_index())
rows["y"] = (rows["n_pos"] > 0).astype(np.int8)
rows = rows.sort_values(KEY + ["d"], kind="mergesort").reset_index(drop=True)
N = len(rows)
log(f"단위 x 날짜 행 {N:,}, 시료 {len(samples):,}")

# 기상 격자(시군구·날짜 번호)
wx = pq.read_table(DS / "식정원_db_기상기후" / "시군구_일별_기상.parquet").to_pandas()
wx["sgg_code"] = wx["sgg_code"].astype(str)
wx["date"] = pd.to_datetime(wx["date"]).values.astype("datetime64[D]")
wx = wx.sort_values(["sgg_code", "date"], kind="mergesort")
SGGS = np.array(sorted(wx["sgg_code"].unique()))
DATES = np.array(sorted(wx["date"].unique())).astype("datetime64[D]")
NS, ND = len(SGGS), len(DATES)
DATE0 = DATES[0]
W = wx[WV].to_numpy(dtype=np.float32).reshape(NS, ND, len(WV))
sidx = {s: i for i, s in enumerate(SGGS)}
r_s = rows["sgg_code"].map(sidx).to_numpy().astype(np.int64)
r_d = rows["d"].to_numpy().astype("datetime64[D]")
r_di = (r_d - DATE0).astype(np.int64)
r_y = rows["y"].to_numpy()
r_p2 = rows["train_prdlst_code"].astype(str).to_numpy()
r_p3 = rows["train_prdlst_code_2"].astype(str).to_numpy()
unit_codes, r_unit = np.unique(rows["sgg_code"] + "|" + rows["train_prdlst_code"].astype(str) + "|" + rows["train_prdlst_code_2"].astype(str),
                               return_inverse=True)
log(f"기상 격자 {W.shape}, {DATE0} ~ {DATES[-1]}, 단위 {len(unit_codes):,}")

# 학습 행 h (cv._h_assign)
hfit = np.zeros(N, dtype=np.int64)
ma = r_d <= LAST_TE
mb = (r_d > LAST_TE) & (r_d <= LEARN_END)
hfit[ma] = np.random.default_rng(42).integers(1, H_MAX + 1, size=int(ma.sum()))
hfit[mb] = np.random.default_rng(43).integers(1, H_MAX + 1, size=int(mb.sum()))


class Pairs:
    def __init__(self, name, idx, h, anchor):
        self.name, self.idx, self.h = name, np.asarray(idx, dtype=np.int64), np.asarray(h, dtype=np.int64)
        self.anchor = None if anchor is None else D(anchor)
        self.s = r_s[self.idx]
        self.t = r_di[self.idx] - self.h                  # T 의 기상 날짜 번호
        self.T = r_d[self.idx] - self.h.astype("timedelta64[D]")
        self.y = r_y[self.idx]
        self.n = len(self.idx)


def t_pairs(name, ts, te, anchor):
    ii, hh = [], []
    for h in range(1, H_MAX + 1):
        T = r_d - np.timedelta64(h, "D")
        m = (T >= D(ts)) & (T <= D(te))
        ii.append(np.flatnonzero(m))
        hh.append(np.full(int(m.sum()), h))
    return Pairs(name, np.concatenate(ii), np.concatenate(hh), anchor)


learn_idx = np.flatnonzero(r_d <= LEARN_END)
SETS = [Pairs("최종 학습 행", learn_idx, hfit[learn_idx], None)]
for k, te_, ts, te in FOLDS:
    SETS.append(t_pairs(f"겹{k} 맞혀 보는 행", ts, te, te_))
SETS.append(t_pairs("시험 행", TEST[1], TEST[2], TEST[0]))
for p in SETS:
    lo = int(p.t.min()) - L
    assert lo >= 0, p.name
    log(f"{p.name}: {p.n:,} (행, h) / h 당 {p.n / (1 if p.anchor is None else H_MAX):,.1f}"
        + ("" if p.name == "시험 행" else f", 검출 {int(p.y.sum()):,}"))

# ---------------------------------------------------------------- 2. 결측 세기 도구
RES = []          # (묶음, 칸, 행 묶음, 칸 수, 결측 칸 비율, 행 기준(하나라도) 결측 비율, 검출 행 중 행 기준 결측 비율, 비고)


def add(bundle, col, p, ncol, nan_cells, any_nan, note=""):
    """nan_cells = (n,) 그 행의 결측 칸 수, any_nan = (n,) bool."""
    use_y = p.name != "시험 행"
    pos = p.y == 1
    RES.append({"묶음": bundle, "칸": col, "행 묶음": p.name, "칸 수": ncol, "(행,h) 수": p.n,
                "결측 칸 비율(%)": round(100.0 * float(nan_cells.sum()) / (p.n * ncol), 3),
                "칸 하나라도 결측인 (행,h) 비율(%)": round(100.0 * float(any_nan.mean()), 3),
                "검출 (행,h) 중 칸 하나라도 결측(%)": (round(100.0 * float(any_nan[pos].mean()), 3) if use_y and pos.any() else None),
                "비고": note})


def prefix(obs):
    """obs (NS, ND) bool -> (NS, ND+1) int32 누적(날짜 번호 < t 인 관측 수)."""
    P = np.zeros((obs.shape[0], obs.shape[1] + 1), dtype=np.int32)
    np.cumsum(obs, axis=1, out=P[:, 1:])
    return P


def wcount(P, s, t, a, b):
    """날짜 [t−a, t−b−1] 관측 수 (a > b ≥ 0)."""
    return P[s, t - b] - P[s, t - a]


def summary_nan(P, p):
    """기상식 요약 13칸(구간 8 + 평균·표준편차·최소·최대·기울기)의 결측 칸 수."""
    nb = L // BLOCK
    empty = np.zeros(p.n, dtype=np.int64)
    for k in range(1, nb + 1):
        empty += (wcount(P, p.s, p.t, BLOCK * k, BLOCK * (k - 1)) == 0)
    cnt = wcount(P, p.s, p.t, L, 0)
    return empty + 4 * (cnt == 0) + (cnt < 2)


def stat_nan(P, p, stats):
    """평균·최근 10일 평균·최대·합·양수일 수 요약의 결측 칸 수."""
    cnt = wcount(P, p.s, p.t, L, 0)
    c10 = wcount(P, p.s, p.t, 10, 0)
    out = np.zeros(p.n, dtype=np.int64)
    for st in stats:
        out += (c10 == 0) if st == "mean10" else (cnt == 0)
    return out


# ---------------------------------------------------------------- 3. 기상 요약(1차 베이스라인 입력, 18개 변수)
OBS = {v: ~np.isnan(W[:, :, j]) for j, v in enumerate(WV)}
PW = {v: prefix(OBS[v]) for v in WV}
for p in SETS:
    tot = np.zeros(p.n, dtype=np.int64)
    for v in WV18:
        c = summary_nan(PW[v], p)
        tot += c
        add("wx 기상 요약", v, p, 13, c, c > 0)
    add("wx 기상 요약", "(18개 변수 234칸 전체)", p, 234, tot, tot > 0)
log("기상 요약 끝")

# ---------------------------------------------------------------- 4. SPI
spi = pq.read_table(DS / "외부_수집" / "시군구_일별_SPI.parquet").to_pandas()
spi["sgg_code"] = spi["sgg_code"].astype(str)
spi = spi.sort_values(["sgg_code", "date"], kind="mergesort")
assert len(spi) == NS * ND
SPI = {v: spi[v].to_numpy(dtype=np.float32).reshape(NS, ND) for v in ["spi3", "spi6"]}
for p in SETS:
    tot = np.zeros(p.n, dtype=np.int64)
    for v in ["spi3", "spi6"]:
        c = summary_nan(prefix(~np.isnan(SPI[v])), p)
        tot += c
        add("spi 표준강수지수", v, p, 13, c, c > 0)
    add("spi 표준강수지수", "(26칸 전체)", p, 26, tot, tot > 0)
# 전 시군구 결측 날(그리드 기준)
spi_all_nan_days = {v: int(np.isnan(SPI[v]).all(axis=0).sum()) for v in SPI}
log(f"SPI 끝, 전 시군구 결측 날 {spi_all_nan_days}")

# ---------------------------------------------------------------- 5. 토양 중앙값(날짜 T 행)
soil = pq.read_table(DS / "식정원_db_토양" / "시군구_일별_토양.parquet").to_pandas()
soil["sgg_code"] = soil["sgg_code"].astype(str)
soil = soil.sort_values(["sgg_code", "date"], kind="mergesort")
SOIL = {v: soil[v].to_numpy(dtype=np.float32).reshape(NS, ND) for v in ["soil_ph", "soil_om", "soil_vldpha"]}
for p in SETS:
    tot = np.zeros(p.n, dtype=np.int64)
    for v, a in SOIL.items():
        c = np.isnan(a[p.s, p.t]).astype(np.int64)
        tot += c
        add("soilmed 토양 중앙값", v, p, 1, c, c > 0)
    add("soilmed 토양 중앙값", "(3칸 전체)", p, 3, tot, tot > 0)
log("토양 끝")

# ---------------------------------------------------------------- 6. 식품 속성(3단계 코드)
food = pq.read_table(DS / "식정원_db_식품" / "품목3단계_식품속성.parquet").to_pandas()
fmap = {r.train_prdlst_code_2: r for r in food.itertuples(index=False)}


def food_val(col):
    return np.array([np.nan if (c not in fmap or pd.isna(getattr(fmap[c], col))) else float(getattr(fmap[c], col)) for c in r_p3])


F_AW, F_PH, F_NA = food_val("aw"), food_val("ph"), food_val("na")
for p in SETS:
    tot = np.zeros(p.n, dtype=np.int64)
    for nm, a in (("food_aw (aw_label '결측' 범주와 같은 행)", F_AW), ("food_ph", F_PH), ("food_na", F_NA)):
        c = np.isnan(a[p.idx]).astype(np.int64)
        tot += c
        add("food 식품 속성", nm, p, 1, c, c > 0)
    add("food 식품 속성", "(숫자 3칸 전체)", p, 3, tot, tot > 0)
    zrow = np.char.startswith(r_p2[p.idx].astype(str), "Z")
    add("food 식품 속성", "참고: 2단계 코드가 Z(기준규격외 등)인 (행,h)", p, 1, zrow.astype(np.int64), zrow, "결측이 아니라 해당 비율")
log("식품 끝")

# ---------------------------------------------------------------- 7. 외부 지표: 친환경·지하수(연도표)


def extidx_grid(path, col):
    t = pq.read_table(path, columns=["sgg_code", "자료연도", "사용가능일", col]).to_pandas()
    t["sgg_code"] = t["sgg_code"].astype(str)
    t["사용가능일"] = pd.to_datetime(t["사용가능일"]).values.astype("datetime64[D]")
    G = np.full((NS, ND), np.nan, dtype=np.float32)
    days = np.arange(ND)
    for sg, g in t.groupby("sgg_code", sort=False):
        g = g.sort_values(["사용가능일", "자료연도"], kind="mergesort")
        a = (g["사용가능일"].to_numpy().astype("datetime64[D]") - DATE0).astype(np.int64)
        k = np.searchsorted(a, days - 1, side="right") - 1
        v = g[col].to_numpy(dtype=np.float64)
        G[sidx[sg]] = np.where(k >= 0, v[np.maximum(k, 0)], np.nan)
    return G, t


ECO, eco_t = extidx_grid(DS / "외부_지표" / "친환경_인증비율.parquet", "친환경_비율")
GW, gw_t = extidx_grid(DS / "외부_지표" / "지하수_정기수질검사.parquet", "적합률")

# HACCP (HaccpData 와 같은 규칙)
hc = pq.read_table(DS / "외부_지표" / "haccp_유효구간.parquet",
                   columns=["sgg_code", "업종묶음", "우리_2단계_코드목록", "시작일", "끝일", "사용시작일"]).to_pandas()
use0 = sorted(set(pd.to_datetime(hc["사용시작일"]).values.astype("datetime64[D]")))
assert len(use0) == 1
USE0 = use0[0]
code2b, hbundles = {}, []
for b, g in hc.groupby("업종묶음", sort=True):
    codes = {c.strip() for Lc in g["우리_2단계_코드목록"].dropna() for c in str(Lc).split(",") if c.strip() and c.strip() != "None"}
    if b == "대응 없음" or not codes:
        continue
    hbundles.append(b)
    for c in codes:
        code2b[c] = len(hbundles) - 1
st = (pd.to_datetime(hc["시작일"]).values.astype("datetime64[D]") - DATE0).astype(np.int64)
en = (pd.to_datetime(hc["끝일"]).values.astype("datetime64[D]") - DATE0).astype(np.int64)
lo_ = np.clip(st + 1, 0, ND)
hi_ = np.clip(en + 2, 0, ND)
hs = np.array([sidx.get(str(x), -1) for x in hc["sgg_code"]], dtype=np.int64)
u0 = int((USE0 - DATE0).astype(np.int64))
HR = []
hb_all = hc["업종묶음"].to_numpy()
for bi, b in enumerate(hbundles):
    m = (hb_all == b) & (lo_ < hi_)
    dC = np.zeros((NS + 1, ND + 1), dtype=np.int64)
    rs_ = np.where(hs[m] >= 0, hs[m], NS)
    np.add.at(dC, (rs_, lo_[m]), 1)
    np.add.at(dC, (rs_, hi_[m]), -1)
    C = np.cumsum(dC[:, :ND], axis=1)
    Ntot = C.sum(0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(Ntot > 0, C[:NS] / Ntot, np.nan)
    R[:, :u0] = np.nan
    HR.append(R.astype(np.float32))
r_ub = np.array([code2b.get(c, -1) for c in r_p2], dtype=np.int64)
log(f"HACCP 업종묶음 {hbundles}, 사용시작일 {USE0}, 묶음 코드 {len(code2b)}개")


def haccp_vals(p):
    out = np.full(p.n, np.nan, dtype=np.float32)
    ub = r_ub[p.idx]
    for bi in range(len(HR)):
        m = ub == bi
        out[m] = HR[bi][p.s[m], p.t[m]]
    return out, ub


ZERO = []   # 값 0 비율
for p in SETS:
    tot = np.zeros(p.n, dtype=np.int64)
    ve = ECO[p.s, p.t]
    vg = GW[p.s, p.t]
    vh, ub = haccp_vals(p)
    for nm, v in (("eco_ratio 친환경 인증 비율", ve), ("gw_pass_rate 지하수 정기수질검사 적합률", vg), ("haccp_ratio HACCP 지정 비율", vh)):
        c = np.isnan(v).astype(np.int64)
        tot += c
        add("extidx 외부 지표", nm, p, 1, c, c > 0)
        ok = ~np.isnan(v)
        ZERO.append({"칸": nm, "행 묶음": p.name, "값 있는 (행,h)": int(ok.sum()), "값 0 비율(값 있는 것 중, %)": round(100.0 * float((v[ok] == 0).mean()), 3) if ok.any() else None,
                     "값 1 비율(값 있는 것 중, %)": round(100.0 * float((v[ok] == 1).mean()), 3) if ok.any() else None,
                     "중앙값": float(np.median(v[ok])) if ok.any() else None})
    # HACCP 결측 이유 나눔
    no_b = ub < 0
    early = p.t < u0
    add("extidx 외부 지표", "haccp 결측 이유: 2단계 코드가 업종묶음 밖", p, 1, no_b.astype(np.int64), no_b)
    add("extidx 외부 지표", "haccp 결측 이유: T < 2017-08-03(사용시작일)", p, 1, early.astype(np.int64), early)
    early_eco = p.T < D("2018-01-01")
    add("extidx 외부 지표", "eco 결측 이유: T <= 2018-01-01(첫 사용가능일 이전·당일)", p, 1, (p.T <= D("2018-01-01")).astype(np.int64), p.T <= D("2018-01-01"))
    add("extidx 외부 지표", "(3칸 전체)", p, 3, tot, tot > 0)
log("외부 지표 끝")

# ---------------------------------------------------------------- 8. 생장 범위
rgw = pq.read_table(DS / "생장범위" / "생장범위_일별_기상.parquet").to_pandas()
rgw["sgg_code"] = rgw["sgg_code"].astype(str)
rgw = rgw.sort_values(["sgg_code", "date"], kind="mergesort")
RGWX = ["rg_tmax_opt", "rg_tavg_opt_dist", "rg_t_out_lim", "rg_soilt_opt", "rg_soilt_lim", "rg_soilt_opt_dist"]
PRG = {v: prefix(~np.isnan(rgw[v].to_numpy(dtype=np.float32).reshape(NS, ND))) for v in RGWX}
rgs = pq.read_table(DS / "생장범위" / "생장범위_일별_토양pH.parquet").to_pandas()
rgs["sgg_code"] = rgs["sgg_code"].astype(str)
rgs = rgs.sort_values(["sgg_code", "date"], kind="mergesort")
RGS = {v: rgs[v].to_numpy(dtype=np.float32).reshape(NS, ND) for v in ["rg_soilph_opt", "rg_soilph_lim", "rg_soilph_opt_dist"]}
rgf = pq.read_table(DS / "생장범위" / "생장범위_품목3단계.parquet").to_pandas()
rgfm = {r[0]: r[1:] for r in rgf.itertuples(index=False, name=None)}
RGF_COLS = ["rg_aw_lim", "rg_aw_opt_dist", "rg_ph_lim", "rg_ph_opt_dist"]
RGF = {c: np.array([np.nan if (x not in rgfm or pd.isna(rgfm[x][j])) else float(rgfm[x][j]) for x in r_p3]) for j, c in enumerate(RGF_COLS)}
for p in SETS:
    tot = np.zeros(p.n, dtype=np.int64)
    sub = np.zeros(p.n, dtype=np.int64)
    for v in RGWX:
        c = stat_nan(PRG[v], p, ("mean", "mean10"))
        sub += c
    add("rg 생장 범위", "rgwx 기상 범위 12칸", p, 12, sub, sub > 0)
    tot += sub
    sub = np.zeros(p.n, dtype=np.int64)
    for v, a in RGS.items():
        sub += np.isnan(a[p.s, p.t])
    add("rg 생장 범위", "rgsoil 토양 pH 범위 3칸", p, 3, sub, sub > 0)
    tot += sub
    sub = np.zeros(p.n, dtype=np.int64)
    for c_ in RGF_COLS:
        sub += np.isnan(RGF[c_][p.idx])
    add("rg 생장 범위", "rgfood 식품 범위 4칸", p, 4, sub, sub > 0)
    tot += sub
    add("rg 생장 범위", "(19칸 전체)", p, 19, tot, tot > 0)
log("생장 범위 끝")

# ---------------------------------------------------------------- 9. 생장식 G1·G2·G3
obsT = OBS["avg_temp"]
PT = PW["avg_temp"]
rain = W[:, :, WV.index("rain")]
obsR2 = np.zeros_like(obsT)
obsR2[:, 2:] = ~np.isnan(rain[:, :-2])
PR2 = prefix(obsR2)
gfood_nan = np.isnan(F_AW) | np.isnan(F_PH)
for p in SETS:
    g1 = stat_nan(PT, p, ("mean", "mean10", "sum", "npos"))
    add("g123 생장식 지수", "g1 온도 요인 4칸", p, 4, g1, g1 > 0)
    base = stat_nan(PT, p, ("mean", "mean10", "max"))
    gf = gfood_nan[p.idx]
    g2 = np.where(gf, 9, 3 * base)
    add("g123 생장식 지수", "g2 µmax·생장확률 P(2균주) 9칸", p, 9, g2, g2 > 0)
    add("g123 생장식 지수", "g2 결측 이유: 3단계 aw 또는 pH 없음", p, 1, gf.astype(np.int64), gf)
    g3 = stat_nan(PR2, p, ("mean", "mean10", "max"))
    add("g123 생장식 지수", "g3 강수 검출확률 3칸", p, 3, g3, g3 > 0)
    tot = g1 + g2 + g3
    add("g123 생장식 지수", "(16칸 전체)", p, 16, tot, tot > 0)
log("생장식 끝")

# ---------------------------------------------------------------- 10. 이동평균 차감 기상(19개 변수 x 3)
for p in SETS:
    tot = np.zeros(p.n, dtype=np.int64)
    for v in WV:
        o = OBS[v]
        Pv = PW[v]
        prev = np.zeros_like(o)
        t_ = np.arange(ND)
        lo = np.clip(t_ - K_ANOM, 0, None)
        prev = (Pv[:, t_] - Pv[:, lo]) > 0                # 앞 30일(당일 제외) 관측 있음
        oa = o & prev
        c = stat_nan(prefix(oa), p, ("mean", "mean10", "max"))
        tot += c
        add("wxanom 이동평균 차감 기상", v, p, 3, c, c > 0)
    add("wxanom 이동평균 차감 기상", "(57칸 전체)", p, 57, tot, tot > 0)
log("이동평균 차감 끝")

# ---------------------------------------------------------------- 11. 과거 이력·N1 (창 규칙)
base = samples["d"].to_numpy().astype("datetime64[D]").min()
s_day = (samples["d"].to_numpy().astype("datetime64[D]") - base).astype(np.int64)
Dh = int(s_day.max()) + 1
CN = np.zeros(Dh + 1, dtype=np.int64)
np.cumsum(np.bincount(s_day, minlength=Dh), out=CN[1:])
nbf = pq.read_table(DS / "제품명" / "시료별_뭉치.parquet").to_pandas()
key_s = pq.read_table(DS / "시료_시군구.parquet", columns=["lims_data_key"]).column(0).to_numpy(zero_copy_only=False).astype(str)
assert np.array_equal(nbf["lims_data_key"].to_numpy().astype(str), key_s)
CL = [c for c in nbf.columns if c.startswith("nb_")]
FL = nbf[CL].to_numpy().astype(np.int64)
CNc = np.zeros((Dh + 1, len(CL)), dtype=np.int64)
ndc = np.zeros((Dh, len(CL)), dtype=np.int64)
np.add.at(ndc, s_day, FL)
np.cumsum(ndc, axis=0, out=CNc[1:])
# 시료의 단위 번호
s_unit_codes = samples["sgg_code"] + "|" + samples["train_prdlst_code"].astype(str) + "|" + samples["train_prdlst_code_2"].astype(str)
umap = {u: i for i, u in enumerate(unit_codes)}
s_unit = s_unit_codes.map(umap).to_numpy().astype(np.int64)
M = Dh + 1
o = np.lexsort((s_day, s_unit))
skey = s_unit[o] * M + s_day[o]


def past_hi(p):
    """창 끝 다음 날의 이력 날짜 번호(hi) - 학습 행: T, 맞혀 보는 행·시험 행: anchor + 1일."""
    if p.anchor is None:
        Tn = (p.T - base).astype(np.int64)
    else:
        Tn = np.full(p.n, int(((p.anchor + np.timedelta64(1, "D")) - base).astype(np.int64)))
    return Tn


for p in SETS:
    Tn = past_hi(p)
    u = r_unit[p.idx]
    for P_, tag in ((P5, "5년"), (P1, "1년")):
        hi = np.clip(Tn, 0, Dh)
        lo = np.clip(Tn - P_, 0, Dh)
        Nn = CN[hi] - CN[lo]
        rate_nan = Nn == 0
        if tag == "5년":
            add("hist_core 과거 검출 이력", "hist_rate5y_* 3칸(전국 창 시료 0건이면 결측)", p, 3, 3 * rate_nan, rate_nan)
        else:
            add("hist_core 과거 검출 이력", "hist_rate1y_* 3칸", p, 3, 3 * rate_nan, rate_nan)
            add("sd1 1년 검출률 불확실성", "hist_sd1y_* 3칸", p, 3, 3 * rate_nan, rate_nan)
        nu = np.searchsorted(skey, u * M + hi, side="left") - np.searchsorted(skey, u * M + lo, side="left")
        share_nan = nu == 0
        Nc = CNc[hi] - CNc[lo]                          # (n, 23)
        rate_c_nan = (Nc == 0)
        cells = 23 * share_nan + rate_c_nan.sum(1)
        add("n1 제품명 과거 유형", f"{tag} 비율 23칸(창 안 그 단위 시료 0건이면 결측)", p, 23, 23 * share_nan, share_nan)
        add("n1 제품명 과거 유형", f"{tag} 스무딩 검출률 23칸(전국 그 뭉치 시료 0건이면 결측)", p, 23, rate_c_nan.sum(1), rate_c_nan.any(1))
log("과거 이력·N1 끝")

# ---------------------------------------------------------------- 12. 저장
res = pd.DataFrame(RES)
res.to_csv(OUT / "ff_missing_rates.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(ZERO).to_csv(OUT / "ff_extidx_zero_rates.csv", index=False, encoding="utf-8-sig")
meta = {"단위x날짜 행": int(N), "시료": int(len(samples)), "기상 격자": [int(NS), int(ND), str(DATE0), str(DATES[-1])],
        "행 묶음": {p.name: {"(행,h)": int(p.n), "h 당 평균": round(p.n / (1 if p.anchor is None else H_MAX), 3),
                          "검출": (None if p.name == "시험 행" else int(p.y.sum())), "anchor": None if p.anchor is None else str(p.anchor),
                          "T 범위": [str(p.T.min()), str(p.T.max())]} for p in SETS},
        "SPI 전 시군구 결측 날(격자 전체)": spi_all_nan_days, "HACCP 업종묶음": hbundles, "HACCP 묶음 2단계 코드 수": len(code2b),
        "HACCP 사용시작일": str(USE0), "이력 base": str(base), "뭉치 수": len(CL), "seconds": round(time.time() - T0, 1)}
(OUT / "ff_missing_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
log("저장 끝")
