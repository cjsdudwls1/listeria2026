"""importance·tuning 시험용 합성 자료(시료 표·fold 범위·가짜 조립 객체·가짜 모델)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lst import evaluate as ev
from lst import folds as fd
from lst.hassign import assign_h_many

SPLITS = {
    "fold1": fd.Split("fold1", fd.day("2016-06-30"), fd.day("2016-07-01"), fd.day("2016-12-31")),
    "fold2": fd.Split("fold2", fd.day("2016-12-31"), fd.day("2017-01-01"), fd.day("2017-06-30")),
    "fold3": fd.Split("fold3", fd.day("2017-06-30"), fd.day("2017-07-01"), fd.day("2017-11-30")),
}
LOOKUP = {"a_col": "t3", "b_col": "L1", "sido": None, "h": None, "g": None, "avg_temp": "avg_temp",
          "t3": "t3", "g_div50": None}


def make_samples(n: int = 900, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    keys = [f"S{i:05d}" for i in range(n)]
    route = np.where(rng.random(n) < 0.3, "overseas", "domestic")
    sgg = np.where(route == "overseas", "OVCHN", rng.choice(["11110", "26110"], n))
    sido = np.array(["OV" if s.startswith("OV") else s[:2] for s in sgg], dtype=object)
    p3 = rng.choice(["A0101000000000", "B0204000000000"], n)
    p2 = np.where(p3 == "A0101000000000", "A0100000000000", "B0200000000000")
    sig = rng.normal(size=n)
    y = (sig + rng.normal(scale=0.8, size=n) > 1.0).astype("int8")
    df = pd.DataFrame({"lims_data_key": keys, "y": pd.array(y, dtype="Int8"), "src_route": route, "sgg": sgg,
                       "sido": sido, "p2": p2, "p3": p3, "d": d.date})
    df["h"] = assign_h_many(keys).astype(np.int16)
    a = sig.copy()
    a_na = rng.random(n) < 0.1
    a[a_na] = np.nan
    feats = pd.DataFrame({"a": a, "a_na": a_na.astype(np.float32), "b": rng.normal(size=n)}, index=keys)
    return ev.SampleTable(df), feats


class FrameAsm:
    """트리 입력 꼴(DataFrame): sido(문자열)·h·g·a_col·a_col__na·b_col·ov_struct_na."""
    cols = ["sido", "h", "g", "a_col", "a_col__na", "b_col", "ov_struct_na"]

    def __init__(self, feats: pd.DataFrame):
        self.feats = feats
        self.fit_rows = None

    def fit(self, rows):
        self.fit_rows = len(rows)
        return self

    def transform(self, meta: pd.DataFrame) -> pd.DataFrame:
        f = self.feats.loc[meta["lims_data_key"].astype(str).to_numpy()]
        return pd.DataFrame({
            "sido": meta["sido"].astype(str).to_numpy(dtype=object),
            "h": meta["h"].to_numpy(dtype=np.float32),
            "g": meta["g"].to_numpy(dtype=np.float32),
            "a_col": f["a"].to_numpy(dtype=np.float32),
            "a_col__na": f["a_na"].to_numpy(dtype=np.float32),
            "b_col": f["b"].to_numpy(dtype=np.float32),
            "ov_struct_na": (meta["src_route"].to_numpy() == "overseas").astype(np.float32),
        })

    @property
    def columns(self):
        return list(self.cols)


class DictAsm:
    """딥러닝 입력 꼴: x_seq(채널 avg_temp·avg_temp__na)·x_static(t3·g_div50)·x_cat."""

    def __init__(self, feats):
        self.feats = feats

    def fit(self, rows):
        return self

    def transform(self, meta):
        f = self.feats.loc[meta["lims_data_key"].astype(str).to_numpy()]
        a = np.nan_to_num(f["a"].to_numpy(dtype=np.float32))
        n = len(meta)
        seq = np.zeros((n, 3, 2), np.float32)
        seq[:, :, 0] = a[:, None] * np.array([0.5, 1.0, 1.5], np.float32)
        seq[:, :, 1] = f["a_na"].to_numpy(dtype=np.float32)[:, None]
        st = np.stack([a, meta["g"].to_numpy(dtype=np.float32) / 50], axis=1).astype(np.float32)
        return {"x_seq": seq, "x_static": st, "x_cat": np.zeros((n, 4), np.int64)}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class FrameModel:
    def __init__(self):
        self.seen = []

    def predict_proba(self, X, n_iter=None):
        self.seen.append(X)
        a = X["a_col"].fillna(0).to_numpy(np.float64)
        return sig(2 * a + 0.3 * X["b_col"].to_numpy(np.float64) + 0.2 * X["a_col__na"].to_numpy(np.float64))


class DictModel:
    def __init__(self):
        self.seen = []

    def predict_proba(self, X, n_iter=None):
        self.seen.append(X)
        return sig(X["x_static"][:, 0].astype(np.float64) + 0.5 * X["x_seq"][:, 2, 0].astype(np.float64))


class StackModel:
    """바탕 칼럼 + 붙인 딥러닝 확률 칼럼 'dl_p'."""

    def __init__(self):
        self.seen = []

    def predict_proba(self, X, n_iter=None):
        self.seen.append(X)
        return sig(X["b_col"].to_numpy(np.float64) + 3 * (X["dl_p"].to_numpy(np.float64) - 0.5))
