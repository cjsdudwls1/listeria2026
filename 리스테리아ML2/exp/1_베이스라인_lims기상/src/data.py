# -*- coding: utf-8 -*-
"""시료 -> 단위x날짜 행, 기상표 -> W[시군구, 날짜, 19] 배열. 원자료는 읽기만 한다."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C


def load_samples() -> pd.DataFrame:
    cols = C.KEY_COLS + ["mnfctur_de", "judge_result"]
    df = pq.read_table(C.SAMPLE_PATH, columns=cols).to_pandas()
    jr = df["judge_result"].astype(object)
    bad = set(jr.unique()) - {"검출", "불검출"}
    if bad:
        raise ValueError(f"judge_result 에 예상 밖 값: {bad}")
    out = pd.DataFrame({
        "sgg_code": df["sgg_code"].astype(str),
        "train_prdlst_code": df["train_prdlst_code"].astype(object).where(df["train_prdlst_code"].notna(), C.NULL_CODE),
        "train_prdlst_code_2": df["train_prdlst_code_2"].astype(object).where(df["train_prdlst_code_2"].notna(), C.NULL_CODE),
        "d": df["mnfctur_de"].values.astype("datetime64[D]"),
        "pos": (jr == "검출").to_numpy().astype(np.int8),
    })
    if out["d"].isna().any():
        raise ValueError("mnfctur_de 빈 값 있음")
    if (out["sgg_code"].str.len() != 5).any():
        raise ValueError("sgg_code 5자리 아님")
    return out


def assign_split(d: pd.Series) -> np.ndarray:
    d = pd.to_datetime(d)
    sp = np.full(len(d), "", dtype=object)
    sp[(d <= pd.Timestamp(C.TRAIN_END)).to_numpy()] = "train"
    sp[((d >= pd.Timestamp(C.VAL_START)) & (d <= pd.Timestamp(C.VAL_END))).to_numpy()] = "val"
    sp[((d >= pd.Timestamp(C.TEST_START)) & (d <= pd.Timestamp(C.TEST_END))).to_numpy()] = "test"
    if (sp == "").any():
        raise ValueError(f"분할 기간 밖 행 {int((sp == '').sum())}개")
    return sp


def build_unit_rows(samples: pd.DataFrame) -> pd.DataFrame:
    """단위 x 기준일 묶음. y = 검출 1건 이상이면 1. n_samples 는 보고용(모델 입력 금지)."""
    g = (samples.groupby(C.KEY_COLS + ["d"], sort=False, observed=True)
         .agg(n_samples=("pos", "size"), n_pos_samples=("pos", "sum"))
         .reset_index())
    g["y"] = (g["n_pos_samples"] > 0).astype(np.int8)
    g = g.sort_values(C.KEY_COLS + ["d"], kind="mergesort").reset_index(drop=True)
    g["split"] = assign_split(g["d"])
    g["sido"] = g["sgg_code"].str[:2]
    g["sgg"] = g["sgg_code"]
    g["p2"] = g["train_prdlst_code"]
    g["p3"] = g["train_prdlst_code_2"]
    return g


def assign_h(rows: pd.DataFrame) -> pd.DataFrame:
    """train: 행마다 h 하나(default_rng(42)), 조기 종료용 val: 행마다 h 하나(default_rng(43)).
    rows 는 이미 단위 키 + d 로 정렬돼 있음. test 와 최종 평가용 val 은 h=1..60 전부를 따로 예측."""
    rows = rows.copy()
    rows["h_fit"] = 0
    for sp, seed in (("train", C.SEED), ("val", C.SEED_H_VAL_ES)):
        m = (rows["split"] == sp).to_numpy()
        rng = np.random.default_rng(seed)
        rows.loc[m, "h_fit"] = rng.integers(1, C.H_MAX + 1, size=int(m.sum()))
    rows["h_fit"] = rows["h_fit"].astype(np.int16)
    return rows


def category_levels(rows: pd.DataFrame) -> dict:
    """트리용 범주 목록(전체 행 기준 고정 목록 -> 분할 간 코드 일치)."""
    return {c: sorted(rows[c].unique().tolist()) for c in C.CAT_COLS}


def cat_codes(rows: pd.DataFrame, levels: dict) -> np.ndarray:
    out = np.empty((len(rows), len(C.CAT_COLS)), dtype=np.int32)
    for j, c in enumerate(C.CAT_COLS):
        out[:, j] = pd.Categorical(rows[c], categories=levels[c]).codes
    if (out < 0).any():
        raise ValueError("범주 코드 -1 발생")
    return out


def lstm_vocab_codes(rows: pd.DataFrame) -> tuple:
    """LSTM 임베딩 번호: 학습(train) 행에 있던 범주만 1..n, 그 밖은 0(0 벡터 고정)."""
    tr = rows[rows["split"] == "train"]
    sizes, out = [], np.zeros((len(rows), len(C.CAT_COLS)), dtype=np.int64)
    for j, c in enumerate(C.CAT_COLS):
        voc = sorted(tr[c].unique().tolist())
        mp = {v: i + 1 for i, v in enumerate(voc)}
        out[:, j] = rows[c].map(mp).fillna(0).astype(np.int64).to_numpy()
        sizes.append(len(voc))
    return out, sizes


def smoke_subset(rows: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(C.SMOKE_SEED)
    keep = np.zeros(len(rows), dtype=bool)
    for sp in C.SPLITS:
        m = (rows["split"] == sp).to_numpy()
        keep |= m & (rows["y"].to_numpy() == 1)
        neg = np.flatnonzero(m & (rows["y"].to_numpy() == 0))
        keep[rng.choice(neg, size=min(C.SMOKE_NEG[sp], len(neg)), replace=False)] = True
    return rows[keep].reset_index(drop=True)


class Weather:
    """기상표를 float32 배열 W[시군구, 날짜, 변수] 로. 값은 그대로(채움·변환 없음)."""

    def __init__(self):
        t = pq.read_table(C.WEATHER_PATH, columns=["sgg_code", "date"] + C.WEATHER_VARS)
        w = t.to_pandas()
        w["sgg_code"] = w["sgg_code"].astype(str)
        w["date"] = pd.to_datetime(w["date"]).values.astype("datetime64[D]")
        self.sggs = np.array(sorted(w["sgg_code"].unique()))
        dates = np.array(sorted(w["date"].unique())).astype("datetime64[D]")
        self.date0 = dates[0]
        self.D = int((dates[-1] - dates[0]).astype(int)) + 1
        self.S = len(self.sggs)
        if len(dates) != self.D:
            raise ValueError("기상표 날짜가 연속이 아님")
        if len(w) != self.S * self.D or w.duplicated(["sgg_code", "date"]).any():
            raise ValueError("기상표가 시군구x날짜 완전 격자가 아님")
        w = w.sort_values(["sgg_code", "date"], kind="mergesort")
        self.W = np.ascontiguousarray(
            w[C.WEATHER_VARS].to_numpy(dtype=np.float32).reshape(self.S, self.D, len(C.WEATHER_VARS)))
        self.sgg_index = {s: i for i, s in enumerate(self.sggs)}
        self.date_last = dates[-1]

    def s_idx(self, sgg_codes) -> np.ndarray:
        idx = pd.Series(sgg_codes).map(self.sgg_index)
        if idx.isna().any():
            raise ValueError("기상표에 없는 sgg_code")
        return idx.to_numpy().astype(np.int64)

    def d_idx(self, d) -> np.ndarray:
        return (np.asarray(d).astype("datetime64[D]") - self.date0).astype(np.int64)

    def check_windows(self, t: np.ndarray, L: int, what: str = ""):
        """창 = t-L .. t-1 (t = 예측하는 날 T 의 인덱스, T 당일 제외)이 기상표 안인지 확인."""
        lo, hi = int(t.min()) - L, int(t.max()) - 1
        assert lo >= 0, f"{what}: 창 시작이 기상표 이전 (idx {lo})"
        assert hi <= self.D - 1, f"{what}: 창 끝이 기상표 이후 (idx {hi})"
        return lo, hi

    def train_period_stats(self):
        """LSTM 표준화용: 학습 기간(날짜 <= TRAIN_END) 기상표 값의 변수별 평균·표준편차(NaN 무시, ddof=0)."""
        end = int((np.datetime64(C.TRAIN_END) - self.date0).astype(int))
        sub = self.W[:, : end + 1, :].reshape(-1, self.W.shape[2]).astype(np.float64)
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0)
        return mu.astype(np.float32), sd.astype(np.float32)


def data_summary(rows: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for sp in C.SPLITS + ["전체"]:
        r = rows if sp == "전체" else rows[rows["split"] == sp]
        recs.append({
            "구분": {"train": "학습", "val": "검증", "test": "테스트", "전체": "전체"}[sp],
            "split": sp,
            "기간": f"{pd.Timestamp(r['d'].min()).date()}~{pd.Timestamp(r['d'].max()).date()}",
            "단위x날짜_행수": len(r),
            "검출_행수": int(r["y"].sum()),
            "검출_%": round(100.0 * r["y"].mean(), 4),
            "묶인_시료수": int(r["n_samples"].sum()),
            "검출_시료수": int(r["n_pos_samples"].sum()),
            "단위수": int(r[C.KEY_COLS].drop_duplicates().shape[0]),
        })
    return pd.DataFrame(recs)


def load_all(smoke: bool = False):
    rows = assign_h(build_unit_rows(load_samples()))
    levels = category_levels(rows)          # 전체 기준 고정 목록
    if smoke:
        rows = smoke_subset(rows)
    return rows, levels
