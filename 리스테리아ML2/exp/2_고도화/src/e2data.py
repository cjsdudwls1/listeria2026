# -*- coding: utf-8 -*-
"""2차 데이터 문맥. 1차 data.py(시료 -> 단위x날짜 행, 기상 W 배열)를 그대로 쓰고 다음을 더한다.
- 시드별 h 배정: 학습 행 default_rng(s), 조기 종료용 검증 행 default_rng(s+1). 전체 행에서 배정한 뒤 부분집합(smoke·C)을 고른다.
  시드 42 = 1차(42/43)와 같음 (selftest 확인).
- 여러 h 배정(H10: 학습 행마다 서로 다른 h 10개, H60: 60개 전부)
- 과거 검출 이력(e2hist.History), 고위험 품목 묶음(학습 구간 시료로 다시 계산)
- 검증·시험 행의 기상 요약 특성 캐시(h=1..60, 디스크 메모리맵). 1차 features.build_tree_matrix 와 같은 함수로 만든 값.
- DL(다음 에이전트)용: 단위 번호, (단위, 날짜) -> y 찾기
원자료는 읽기만 한다.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd

import config as C1
import data as D
import features as F
import e2cfg as E
from e2hist import History


def smoke_keep(rows: pd.DataFrame) -> np.ndarray:
    """1차 smoke_subset 과 같은 방식(양성 전부 + 음성 일부), 음성 수만 2차 설정."""
    rng = np.random.default_rng(E.SMOKE_SEED)
    keep = np.zeros(len(rows), dtype=bool)
    y = rows["y"].to_numpy()
    for sp in C1.SPLITS:
        m = (rows["split"] == sp).to_numpy()
        keep |= m & (y == 1)
        neg = np.flatnonzero(m & (y == 0))
        keep[rng.choice(neg, size=min(E.SMOKE_NEG[sp], len(neg)), replace=False)] = True
    return keep


def _split_of_dates(d) -> np.ndarray:
    d = np.asarray(d).astype("datetime64[D]")
    sp = np.full(len(d), "", dtype=object)
    sp[d <= np.datetime64(C1.TRAIN_END)] = "train"
    sp[(d >= np.datetime64(C1.VAL_START)) & (d <= np.datetime64(C1.VAL_END))] = "val"
    sp[(d >= np.datetime64(C1.TEST_START)) & (d <= np.datetime64(C1.TEST_END))] = "test"
    return sp


def group_outside(codes, samples: pd.DataFrame, rows: pd.DataFrame = None) -> dict:
    """묶음 품목 밖(3단계 코드가 codes 에 없음) 검증·시험 검출: 단위x날짜 행(rows, y=1)과 시료(pos=1) 수·비율.
    C4(학습 때 검출 있던 품목)의 밖 = 그 전용 모델이 다루지 않는 범위(사용자 지시 2026-10-19 추가 보고)."""
    cs = set(codes)
    out = {}
    s_sp = _split_of_dates(samples["d"].to_numpy())
    s_in = samples["train_prdlst_code_2"].astype(str).isin(cs).to_numpy()
    s_pos = samples["pos"].to_numpy() == 1
    for sp in ("val", "test"):
        m = s_sp == sp
        r = {"samples": int(m.sum()), "samples_outside": int((m & ~s_in).sum()),
             "pos_samples": int((m & s_pos).sum()), "pos_samples_outside": int((m & s_pos & ~s_in).sum())}
        if rows is not None:
            rm = (rows["split"] == sp).to_numpy()
            r_in = rows["train_prdlst_code_2"].astype(str).isin(cs).to_numpy()
            ry = rows["y"].to_numpy() == 1
            r.update({"rows": int(rm.sum()), "rows_outside": int((rm & ~r_in).sum()),
                      "pos_rows": int((rm & ry).sum()), "pos_rows_outside": int((rm & ry & ~r_in).sum())})
            r["pos_rows_outside_rate"] = r["pos_rows_outside"] / r["pos_rows"] if r["pos_rows"] else float("nan")
            r["rows_outside_rate"] = r["rows_outside"] / r["rows"] if r["rows"] else float("nan")
            out_codes = rows.loc[rm & ry & ~r_in, "train_prdlst_code_2"].astype(str).value_counts()
            r["pos_rows_outside_codes"] = {k: int(v) for k, v in out_codes.items()}
        r["pos_samples_outside_rate"] = r["pos_samples_outside"] / r["pos_samples"] if r["pos_samples"] else float("nan")
        out[sp] = r
    return out


def compute_groups(samples: pd.DataFrame, rows: pd.DataFrame = None) -> dict:
    """학습 구간(d <= 2024-06-30) 시료 기준 품목군 3단계(train_prdlst_code_2)별 검사 수·검출 수로 묶음을 고른다.
    C1~C3 = 이전과 같음. C4(사용자 지시 2026-10-19) = 학습 구간 검출 1건 이상 품목 전부(검출 수 내림차순, 같으면 코드 순)
    + 그 밖 검증·시험 검출 수(rows 를 주면 단위x날짜 행 기준도)."""
    tr = samples[samples["d"] <= pd.Timestamp(C1.TRAIN_END)]
    g = tr.groupby("train_prdlst_code_2", observed=True).agg(n=("pos", "size"), s=("pos", "sum"))
    g["s"] = g["s"].astype(int)
    g["rate"] = g["s"] / g["n"]
    g["share"] = g["s"] / g["s"].sum()
    g = g.reset_index().rename(columns={"train_prdlst_code_2": "code"})
    out = {}
    missing = [c for c in E.GROUP_MUSHROOM if c not in set(g["code"])]
    if missing:
        raise ValueError(f"버섯 코드가 학습 구간 시료에 없음: {missing}")
    c1 = g[g["code"].isin(E.GROUP_MUSHROOM)].set_index("code").loc[E.GROUP_MUSHROOM].reset_index()
    c2 = g[(g["rate"] >= E.GROUP_RATE_MIN) & (g["s"] >= E.GROUP_POS_MIN)].sort_values(["rate", "code"], ascending=[False, True])
    c3 = g.sort_values(["s", "code"], ascending=[False, True]).head(E.GROUP_SHARE_TOP)
    c4 = g[g["s"] >= E.GROUP_C4_POS_MIN].sort_values(["s", "code"], ascending=[False, True])
    for name, df in (("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4)):
        df = df.reset_index(drop=True)
        out[name] = {"codes": df["code"].tolist(), "desc": E.GROUP_DESC[name],
                     "train_samples": int(df["n"].sum()), "train_pos": int(df["s"].sum()),
                     "train_pos_share": float(df["s"].sum() / g["s"].sum()),
                     "table": df[["code", "n", "s", "rate", "share"]].to_dict("records"),
                     "matches_expected": sorted(df["code"].tolist()) == sorted(E.GROUP_EXPECTED[name])}
    out["C4"]["rule"] = f"학습 구간(기준일 <= {C1.TRAIN_END}) 시료에서 검출 >= {E.GROUP_C4_POS_MIN}건인 품목군 3단계 코드 전부"
    out["C4"]["matches_expected_train_samples"] = out["C4"]["train_samples"] == E.GROUP_C4_EXPECTED_TRAIN_SAMPLES
    out["C4"]["outside"] = group_outside(out["C4"]["codes"], samples, rows)
    out["_train_total"] = {"samples": int(g["n"].sum()), "pos": int(g["s"].sum())}
    return out


class Ctx2:
    def __init__(self, smoke: bool, log, use_cache: bool = True, cache_dir=None):
        t0 = time.time()
        self.smoke = smoke
        self.log = log
        self.samples = D.load_samples()
        rows_full = D.build_unit_rows(self.samples)
        self.levels = D.category_levels(rows_full)               # 전체 기준 고정 목록(1차와 같음)
        split_full = rows_full["split"].to_numpy()
        self._full_pos = {sp: np.flatnonzero(split_full == sp) for sp in C1.SPLITS}
        self._keep = smoke_keep(rows_full) if smoke else np.ones(len(rows_full), dtype=bool)
        self.rows_full = rows_full
        rows = rows_full[self._keep].reset_index(drop=True)
        self.rows = rows
        self.weather = D.Weather()
        self.H = History(self.samples)
        self.groups = compute_groups(self.samples, rows_full)
        codes = D.cat_codes(rows, self.levels)
        lcodes, self.lstm_cat_sizes = D.lstm_vocab_codes(rows)
        s = self.weather.s_idx(rows["sgg_code"].to_numpy())
        d = self.weather.d_idx(rows["d"].to_numpy())
        hd = self.H.day_idx(rows["d"].to_numpy())
        grp = {lv: self.H.group_ids(rows, lv) for lv in ("p3", "sgg", "unit", "sido", "p2")}
        unit_full, _ = pd.factorize(rows_full["sgg_code"].astype(str) + "|" + rows_full["train_prdlst_code"].astype(str)
                                    + "|" + rows_full["train_prdlst_code_2"].astype(str), sort=True)
        self._unit_full = unit_full.astype(np.int64)
        unit = self._unit_full[self._keep]
        cat_str = np.stack([rows[c].astype(str).to_numpy(dtype=object) for c in C1.CAT_COLS], axis=1)
        p3 = rows["train_prdlst_code_2"].astype(str).to_numpy(dtype=object)
        self.sp = {}
        for sp in C1.SPLITS:
            idx = np.flatnonzero((rows["split"] == sp).to_numpy())
            self.sp[sp] = {"idx": idx, "s": s[idx], "d": d[idx], "hd": hd[idx],
                           "y": rows["y"].to_numpy()[idx].astype(np.int8), "cats": codes[idx],
                           "lstm_cats": lcodes[idx], "cat_str": np.ascontiguousarray(cat_str[idx]),
                           "grp": {k: v[idx] for k, v in grp.items()}, "p3": p3[idx], "unit": unit[idx]}
        self.use_cache = use_cache and not smoke
        self.cache_dir = cache_dir
        self._cache = {}
        self._yl = None
        log(f"데이터 준비 {time.time() - t0:.1f}s: " + ", ".join(
            f"{sp} {len(v['idx']):,}행/양성 {int(v['y'].sum()):,}" for sp, v in self.sp.items())
            + f" | 기상 W {self.weather.W.shape} {self.weather.date0}~{self.weather.date_last}"
            + f" | 이력 시료 {self.H.n_samples:,}(검출 {self.H.n_pos:,}), 날짜 {self.H.base}~ {self.H.Dh}일")

    # ---------------------------------------------------------------- h 배정
    def _sub(self, sp, full_arr):
        return full_arr[self._keep[self._full_pos[sp]]]

    def h_fit(self, seed: int) -> dict:
        """학습: default_rng(seed).integers(1,61) / 조기 종료용 검증: default_rng(seed+1). 전체 행(단위 키+d 정렬) 기준 배정."""
        out = {}
        for sp, sd in (("train", seed), ("val", seed + 1)):
            rng = np.random.default_rng(sd)
            full = rng.integers(1, E.H_MAX + 1, size=len(self._full_pos[sp])).astype(np.int16)
            out[sp] = self._sub(sp, full)
        return out

    def h_multi(self, seed: int, k: int) -> np.ndarray:
        """학습 행마다 서로 다른 h k 개 (n, k) int16. k=60 은 1..60 전부(무작위 없음).
        k<60: default_rng(seed).random((전체 학습 행, 60)) 의 행별 argsort 앞 k 개 + 1 (중복 없는 무작위 순서)."""
        n_full = len(self._full_pos["train"])
        if k == E.H_MAX:
            full = np.tile(np.arange(1, E.H_MAX + 1, dtype=np.int16), (n_full, 1))
        else:
            rng = np.random.default_rng(seed)
            full = (np.argsort(rng.random((n_full, E.H_MAX)), axis=1)[:, :k] + 1).astype(np.int16)
        return self._sub("train", full)

    # ---------------------------------------------------------------- 묶음(C) 필터
    def positions(self, sp: str, group=None) -> np.ndarray:
        n = len(self.sp[sp]["idx"])
        if group is None:
            return np.arange(n)
        codes = set(self.groups[group]["codes"])
        return np.flatnonzero(np.isin(self.sp[sp]["p3"], list(codes)))

    # ---------------------------------------------------------------- 기상 요약
    def weather_block(self, sp, pos, h) -> np.ndarray:
        """[h, 기상 요약 190] float32 = 1차 build_tree_matrix(..., 'summary', cats=None) 와 같은 값.
        h 가 정수이고 캐시를 쓰면 캐시에서, 아니면 그 자리에서 계산."""
        v = self.sp[sp]
        if np.isscalar(h) and self.use_cache and sp in ("val", "test"):
            wc = self._eval_cache(sp)
            X = np.empty((len(pos), 1 + wc.shape[2]), dtype=np.float32)
            X[:, 0] = h
            X[:, 1:] = wc[int(h) - 1][pos]
            return X
        hh = np.full(len(pos), int(h), dtype=np.int16) if np.isscalar(h) else np.asarray(h, dtype=np.int16)
        t = v["d"][pos] - hh.astype(np.int64)
        return F.build_tree_matrix(self.weather.W, v["s"][pos], t, hh, E.L, "summary")

    def _eval_cache(self, sp):
        if sp in self._cache:
            return self._cache[sp]
        v = self.sp[sp]
        nw = F.n_weather_features("summary", E.L)
        key = hashlib.md5(np.ascontiguousarray(np.stack([v["s"], v["d"]])).tobytes()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f = self.cache_dir / f"eval_wsum_L{E.L:03d}_{sp}.npy"
        mf = self.cache_dir / f"eval_wsum_L{E.L:03d}_{sp}.json"
        meta = {"n": int(len(v["s"])), "n_weather": nw, "rows_md5": key, "h_max": E.H_MAX,
                "weather_shape": list(self.weather.W.shape), "weather_date0": str(self.weather.date0)}
        ok = False
        if f.exists() and mf.exists():
            try:
                ok = json.loads(mf.read_text(encoding="utf-8")) == meta
            except Exception:
                ok = False
        if not ok:
            t0 = time.time()
            if mf.exists():
                mf.unlink()
            mm = np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=(E.H_MAX, len(v["s"]), nw))
            for h in range(1, E.H_MAX + 1):
                hh = np.full(len(v["s"]), h, dtype=np.int16)
                mm[h - 1] = F.build_tree_matrix(self.weather.W, v["s"], v["d"] - h, hh, E.L, "summary")[:, 1:]
            mm.flush()
            del mm
            mf.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            self.log(f"  기상 요약 캐시 생성 {sp}: {f.name} ({time.time() - t0:.1f}s)")
        self._cache[sp] = np.load(f, mmap_mode="r")
        return self._cache[sp]

    def release_eval_cache(self) -> int:
        """검증·시험 기상 요약 캐시 메모리맵을 놓는다(다음 평가 때 같은 파일을 다시 염 - 값 같음).
        큰 학습(H10·H30) 전에 앞 실행 평가로 작업 집합에 올라온 캐시 파일 페이지(약 2~3GB)를 내려 시스템 사용 가능 메모리를 늘리려고."""
        n = len(self._cache)
        self._cache.clear()
        import gc
        gc.collect()
        return n

    # ---------------------------------------------------------------- DL 용
    def lookup_y(self, unit: np.ndarray, wday: np.ndarray) -> np.ndarray:
        """(단위 번호, 기상 날짜 번호) 의 단위x날짜 행 y. 행이 없으면 -1. 전체 행(모든 분할, smoke 와 무관) 기준."""
        if self._yl is None:
            r = self.rows_full
            wd = self.weather.d_idx(r["d"].to_numpy())
            key = self._unit_full * (self.weather.D + 1) + wd
            o = np.argsort(key, kind="mergesort")
            self._yl = (key[o], r["y"].to_numpy().astype(np.int8)[o])
        keys, ys = self._yl
        q = np.asarray(unit, dtype=np.int64) * (self.weather.D + 1) + np.asarray(wday, dtype=np.int64)
        i = np.searchsorted(keys, q)
        i = np.clip(i, 0, len(keys) - 1)
        hit = keys[i] == q
        return np.where(hit, ys[i], -1).astype(np.int8)

    def data_summary(self):
        return D.data_summary(self.rows)
