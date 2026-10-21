# -*- coding: utf-8 -*-
"""N. 제품명 변수(사용자 승인 2026-10-19, 교수님 계획서 4·5절). 실험 A3_N1·A3_N2·A3_N1N2·A3_N0 (단계 'n').

자료(다른 세션 [질문] 이 만든 파일, 읽기만): data/데이터셋/제품명/
  시료별_뭉치.parquet   628,517행 = 시료_시군구.parquet 행 순서, lims_data_key + nb_* int8 21칸(제품명 정규식 뭉치, 뭉치_정의.csv)
  단위날짜_tfidf.parquet 단위(sgg_code·train_prdlst_code·train_prdlst_code_2) x 날짜 d 마다 n_rows(시료 수, 입력 금지)
                         + tf_c_000~299(글자 2~4그램)·tf_w_000~199(낱말) float32. 어휘·idf = 단위날짜_tfidf_어휘.csv
  train_prdlst_code 빈 값(null)은 우리 키처럼 '코드없음'으로 맞춘다(1차 data.load_samples 와 같은 규칙).

정렬: 목표일 d, 앞당김 h -> 예측하는 날 T = d − h.
  N1 'n1' 단위(시군구 x 2단계 x 3단계)의 과거 뭉치 이력 - 뭉치 c 마다, 기간 P = 5년(1,826일)·1년(365일),
          창 = 기준일 ∈ [T−P, T−1] 시료(T 당일과 그 뒤 제외, A3 이력 e2hist 와 같은 창·같은 날짜 번호·같은 자르기 [0, Dh]):
          share_c = n_c / n (단위 시료 n 중 뭉치 c 시료 n_c 비율, n = 0 이면 결측)
          rate_c  = (s_c + k·p0_c) / (n_c + k)  (s_c = 단위 뭉치 c 시료 중 검출, k = k*,
                    p0_c = 같은 창 전국 뭉치 c 시료의 검출률 S_c / N_c, N_c = 0 이면 결측 -> rate_c 결측)
          -> 21 x 2(비율·스무딩 검출률) x 2(5년·1년) = 84칸. 건수(n, n_c) 칸은 넣지 않음(날짜 시계 교훈).
  N2 'n2' 그날 그 단위 시료들의 뭉치 멀티핫(뭉치 c 시료가 하나라도 있으면 1) 21칸 - 목표일 시료 자체 정보(h 와 무관).
          운영에서 예측 때 그날 검사할 제품명을 입력받는다는 전제.
  N0 'n0' 단위날짜_tfidf 의 tf_* 500칸 그대로(n_rows 제외) - 목표일 시료 자체 정보(h 와 무관), N2 와 같은 전제.
원자료는 읽기만 한다. 값 채움·단위 변환 없음(특성 행렬에 넣을 때 float32).
"""
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C1
import e2cfg as E

NB_DIR = C1.DATA_DIR / "제품명"
NB_SAMPLE_PATH = NB_DIR / "시료별_뭉치.parquet"
NB_DEF_PATH = NB_DIR / "뭉치_정의.csv"
TFIDF_PATH = NB_DIR / "단위날짜_tfidf.parquet"
TFIDF_VOCAB_PATH = NB_DIR / "단위날짜_tfidf_어휘.csv"
NB_N_SAMPLES = 628517                          # 사양: 시료별_뭉치 행 수 = 시료_시군구 행 수
TFIDF_N_ROWS_COL = "n_rows"                   # 그 단위·날짜 시료 수(입력 금지 - 날짜 시계·건수)
NB_ORDER = ["n1", "n2", "n0"]
NB_PERIODS = [(E.P5, "5y"), (E.P1, "1y")]
NB_KINDS = ["share", "rate"]
CHUNK = 65536


def cluster_cols():
    """뭉치_정의.csv 의 칼럼명 순서(nb_*). 시료별_뭉치.parquet 칸 순서와 같은지는 NbData 가 확인."""
    d = pd.read_csv(NB_DEF_PATH, encoding="utf-8-sig")
    return d["칼럼명"].astype(str).tolist()


def cluster_names():
    """{nb_* 칼럼명: 한글 이름} (뭉치_정의.csv)."""
    d = pd.read_csv(NB_DEF_PATH, encoding="utf-8-sig")
    return dict(zip(d["칼럼명"].astype(str), d["한글 이름"].astype(str)))


def tfidf_cols():
    """단위날짜_tfidf.parquet 의 tf_* 칸(파일 순서, n_rows·키 칸 제외)."""
    names = pq.ParquetFile(TFIDF_PATH).schema_arrow.names
    return [c for c in names if c.startswith("tf_")]


def n1_names(clusters=None):
    cl = clusters or cluster_cols()
    return [f"n1_{kind}{pl}_{c[3:]}" for _, pl in NB_PERIODS for kind in NB_KINDS for c in cl]


def n2_names(clusters=None):
    cl = clusters or cluster_cols()
    return [f"n2_{c[3:]}" for c in cl]


def nb_names(blocks):
    bad = set(blocks) - set(NB_ORDER)
    if bad:
        raise ValueError(f"모르는 제품명 묶음: {bad}")
    names = []
    for b in NB_ORDER:
        if b not in blocks:
            continue
        if b == "n1":
            names += n1_names()
        elif b == "n2":
            names += n2_names()
        elif b == "n0":
            names += tfidf_cols()
    return names


BLOCK_DESC = {"n1": "N1 단위 과거 뭉치 이력 84(21뭉치 x 비율·스무딩 검출률 x 5년·1년)",
              "n2": "N2 그날 단위 시료 뭉치 멀티핫 21",
              "n0": "N0 그날 단위 제품명 TF-IDF 500(글자 300 + 낱말 200)"}


def nb_desc(blocks):
    """표 '입력' 칸."""
    return "제품명[" + ", ".join(BLOCK_DESC[b] for b in NB_ORDER if b in blocks) + "]"


def nb_count_desc(blocks):
    """config.json n_features_desc 용."""
    return "제품명 " + " + ".join(f"{b.upper()} {len(nb_names([b]))}" for b in NB_ORDER if b in blocks)


N1_RULE = ("단위(시군구 x 2단계 x 3단계)의 과거 뭉치 이력. 예측하는 날 T = d − h, 창 = 기준일 ∈ [T−P, T−1] 시료(T 당일·이후 제외, "
           "A3 이력(e2hist)과 같은 창·날짜 번호·자르기). 뭉치 c(21개)마다 P = 5년(1,826일)·1년(365일)에 대해 "
           "share_c = n_c / n (단위 시료 n 중 뭉치 c 시료 n_c 비율, n = 0 이면 결측), "
           "rate_c = (s_c + k·p0_c) / (n_c + k) (s_c = 단위 뭉치 c 시료 중 검출 수, k = k*, p0_c = 같은 창 전국 뭉치 c 시료 검출률, "
           "전국 뭉치 c 시료가 0 이면 결측) -> 21 x 2 x 2 = 84칸. 건수 칸(n, n_c)은 넣지 않음(날짜 시계 교훈). "
           "한 시료가 여러 뭉치에 들 수 있음(뭉치끼리 겹침)")
N2_RULE = ("그날(목표일 d) 그 단위 시료들의 뭉치 멀티핫: 뭉치 c 시료가 하나라도 있으면 1, 없으면 0 -> 21칸(h 와 무관). "
           "목표일 시료 자체의 제품명 정보이므로 운영에서 예측할 때 그날 검사할 제품명을 입력받는다는 전제(보고서 한계)")
N0_RULE = ("단위날짜_tfidf.parquet 의 tf_c_000~299(글자 2~4그램 300)·tf_w_000~199(낱말 200) = 500칸 그대로(h 와 무관), "
           "n_rows(그 단위·날짜 시료 수)는 입력에서 뺌. 키 = (sgg_code, train_prdlst_code, train_prdlst_code_2, d), "
           "train_prdlst_code 빈 값은 '코드없음'으로 맞춤. 목표일 시료 자체의 제품명 정보이므로 N2 와 같은 전제(보고서 한계). "
           "어휘·idf 를 어느 기간 자료로 정했는지는 파일(단위날짜_tfidf_어휘.csv: 열이름·종류·어휘·idf)에 적혀 있지 않음 - 확인 필요")


def nb_info(blocks):
    """config.json 기록용."""
    d = {"blocks": list(blocks), "n_names": len(nb_names(blocks))}
    if "n1" in blocks:
        d["n1"] = {"file": NB_SAMPLE_PATH.name, "clusters": cluster_cols(), "periods": [p for p, _ in NB_PERIODS],
                   "window": "기준일 ∈ [T−P, T−1] (T = d − h, 당일 제외)", "rule": N1_RULE, "names": n1_names()}
    if "n2" in blocks:
        d["n2"] = {"file": NB_SAMPLE_PATH.name, "rule": N2_RULE, "names": n2_names()}
    if "n0" in blocks:
        d["n0"] = {"file": TFIDF_PATH.name, "vocab": TFIDF_VOCAB_PATH.name, "excluded": [TFIDF_N_ROWS_COL], "rule": N0_RULE,
                   "n_cols": len(tfidf_cols())}
    return d


def _fill_code(s):
    s = s.astype(object)
    return s.where(s.notna(), C1.NULL_CODE).astype(str)


class NbHist:
    """N1 계산용: 시료를 (단위 번호, 날짜 번호)로 정렬해 누적합을 두고, 창 경계를 searchsorted 로 찾는다.
    unit (N,) 시료의 단위 번호, day (N,) 기준일 날짜 번호(History base), Dh = 날짜 수, flags (N, C) 0/1, pos (N,) 0/1."""

    def __init__(self, unit, day, Dh, flags, pos):
        unit = np.asarray(unit, dtype=np.int64)
        day = np.asarray(day, dtype=np.int64)
        flags = np.asarray(flags).astype(np.int32)
        pos = np.asarray(pos).astype(np.int32)
        if day.min() < 0 or day.max() >= Dh:
            raise ValueError("시료 날짜 번호가 [0, Dh) 밖")
        self.Dh = int(Dh)
        self.M = self.Dh + 1
        o = np.lexsort((day, unit))
        self.key = unit[o] * self.M + day[o]
        N, C = flags.shape
        self.C = C
        fo = flags[o]
        so = fo * pos[o][:, None]
        self.cn = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(np.ones(N, dtype=np.int64), out=self.cn[1:])
        self.cnc = np.zeros((N + 1, C), dtype=np.int32)
        np.cumsum(fo, axis=0, out=self.cnc[1:])
        self.csc = np.zeros((N + 1, C), dtype=np.int32)
        np.cumsum(so, axis=0, out=self.csc[1:])
        # 전국 뭉치 c 일별 누적: CN[t, c] = 날짜 번호 < t 인 뭉치 c 시료 수
        nd = np.zeros((self.Dh, C), dtype=np.int64)
        sd = np.zeros((self.Dh, C), dtype=np.int64)
        np.add.at(nd, day, flags.astype(np.int64))
        np.add.at(sd, day, (flags * pos[:, None]).astype(np.int64))
        self.CN = np.zeros((self.Dh + 1, C), dtype=np.int64)
        self.CS = np.zeros((self.Dh + 1, C), dtype=np.int64)
        np.cumsum(nd, axis=0, out=self.CN[1:])
        np.cumsum(sd, axis=0, out=self.CS[1:])

    def window(self, g, t, P):
        """창 T−P .. T−1 의 단위 (n, n_c, s_c) 와 전국 (N_c, S_c). t = T 의 날짜 번호(History base), g = 단위 번호."""
        g = np.asarray(g, dtype=np.int64)
        t = np.asarray(t, dtype=np.int64)
        hi = np.clip(t, 0, self.Dh)
        lo = np.clip(t - P, 0, self.Dh)
        ih = np.searchsorted(self.key, g * self.M + hi, side="left")
        il = np.searchsorted(self.key, g * self.M + lo, side="left")
        n = (self.cn[ih] - self.cn[il]).astype(np.float64)
        nc = (self.cnc[ih] - self.cnc[il]).astype(np.float64)
        sc = (self.csc[ih] - self.csc[il]).astype(np.float64)
        Nc = (self.CN[hi] - self.CN[lo]).astype(np.float64)
        Sc = (self.CS[hi] - self.CS[lo]).astype(np.float64)
        return n, nc, sc, Nc, Sc

    def features(self, g, t, k):
        """(n, 84) float64: P 마다 [share 21, rate 21] (NB_PERIODS 순서)."""
        parts = []
        for P, _ in NB_PERIODS:
            n, nc, sc, Nc, Sc = self.window(g, t, P)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(n[:, None] > 0, nc / np.where(n > 0, n, 1.0)[:, None], np.nan)
                p0 = np.where(Nc > 0, Sc / np.where(Nc > 0, Nc, 1.0), np.nan)
                rate = (sc + k * p0) / (nc + k)
            parts += [share, rate]
        return np.hstack(parts)


class NbData:
    """제품명 자료(N1·N2·N0). ctx(Ctx2)의 시료 순서·단위 번호(History unit)·날짜 번호와 맞춘다.
    N2·N0 는 전체 단위x날짜 행(rows_full) 순서 배열로 만들고, 분할 행은 smoke 와 무관하게 전체 행 번호로 찾는다."""

    def __init__(self, ctx, need=("n1", "n2", "n0"), log=None):
        t0 = time.time()
        self.check = {}
        H = ctx.H
        smp = ctx.samples
        self.clusters = cluster_cols()
        # ---- 시료별_뭉치: 행 수·키·칸 확인
        pf = pq.ParquetFile(NB_SAMPLE_PATH)
        cols = pf.schema_arrow.names
        nb_cols = [c for c in cols if c.startswith("nb_")]
        if nb_cols != self.clusters:
            raise ValueError(f"시료별_뭉치 칸 순서가 뭉치_정의.csv 와 다름: {nb_cols} vs {self.clusters}")
        if len(self.clusters) != 21:
            raise ValueError(f"뭉치 수 {len(self.clusters)} != 21")
        t = pq.read_table(NB_SAMPLE_PATH).to_pandas()
        key_s = pq.read_table(C1.SAMPLE_PATH, columns=["lims_data_key"]).column(0).to_numpy(zero_copy_only=False).astype(str)
        if len(t) != len(smp) or len(t) != NB_N_SAMPLES or len(key_s) != len(t):
            raise ValueError(f"시료별_뭉치 행 수 {len(t)} / 시료 {len(smp)} / 키 {len(key_s)} (사양 {NB_N_SAMPLES})")
        same_key = bool(np.array_equal(t["lims_data_key"].to_numpy().astype(str), key_s))
        if not same_key:
            raise ValueError("시료별_뭉치 lims_data_key 가 시료_시군구.parquet 행 순서와 다름")
        F = t[self.clusters].to_numpy()
        if F.dtype != np.int8 or not np.isin(F, [0, 1]).all():
            raise ValueError("뭉치 칸이 int8 0/1 이 아님")
        self.flags = F
        self.check["nb_samples"] = {"rows": int(len(t)), "lims_data_key_same_order": same_key, "n_clusters": len(self.clusters),
                                    "flag_sums": {c: int(v) for c, v in zip(self.clusters, F.sum(0))}}
        # ---- 시료 단위 번호·날짜 번호 (History 와 같은 번호)
        self.s_unit = H.group_ids(smp, "unit")
        self.s_day = H.day_idx(smp["d"].to_numpy())
        rows_full = ctx.rows_full
        r_unit = H.group_ids(rows_full, "unit")
        r_day = H.day_idx(rows_full["d"].to_numpy())
        M = H.Dh + 1
        r_key = r_unit * M + r_day
        if len(np.unique(r_key)) != len(r_key):
            raise ValueError("단위x날짜 행 키 중복")
        self._r_order = np.argsort(r_key, kind="mergesort")
        self._r_sorted = r_key[self._r_order]
        self.n_rows_full = len(rows_full)
        # 분할 행 -> 전체 행 번호
        keep_idx = np.flatnonzero(ctx._keep)
        self.frow = {sp: keep_idx[ctx.sp[sp]["idx"]] for sp in C1.SPLITS}
        self.hist = None
        self.N2 = None
        self.N0 = None
        if "n1" in need:
            t1 = time.time()
            self.hist = NbHist(self.s_unit, self.s_day, H.Dh, F, smp["pos"].to_numpy())
            self.check["n1_seconds"] = time.time() - t1
        if "n2" in need:
            ridx = self._row_of(self.s_unit * M + self.s_day)
            N2 = np.zeros((self.n_rows_full, len(self.clusters)), dtype=np.int8)
            np.maximum.at(N2, ridx, F)
            cnt = np.bincount(ridx, minlength=self.n_rows_full)
            if (cnt == 0).any():
                raise ValueError("시료가 하나도 없는 단위x날짜 행")
            if not np.array_equal(cnt, rows_full["n_samples"].to_numpy()):
                raise ValueError("행별 시료 수가 rows_full n_samples 와 다름")
            self.N2 = N2
            self.check["n2"] = {"rows_with_any": int(N2.any(1).sum()), "rows": int(len(N2))}
        if "n0" in need:
            self.N0, self.check["n0"] = self._load_tfidf(H, M, rows_full)
        self.seconds = time.time() - t0
        if log:
            log(f"  제품명 자료 준비 {self.seconds:.1f}s: 시료별_뭉치 {len(t):,}행(키 순서 같음), 뭉치 {len(self.clusters)}개"
                + (f", N2 행 {len(self.N2):,}" if self.N2 is not None else "")
                + (f", N0 {self.N0.shape} ({self.N0.nbytes / 1e9:.2f} GB, 키 일치 {self.check['n0']['matched']:,}행, "
                   f"n_rows = 시료 수 {self.check['n0']['n_rows_eq_n_samples']})" if self.N0 is not None else ""))

    def _row_of(self, keys):
        i = np.searchsorted(self._r_sorted, keys)
        i = np.clip(i, 0, len(self._r_sorted) - 1)
        if not np.array_equal(self._r_sorted[i], keys):
            raise ValueError("단위x날짜 행에 없는 키")
        return self._r_order[i]

    def _load_tfidf(self, H, M, rows_full):
        names = pq.ParquetFile(TFIDF_PATH).schema_arrow.names
        keycols = C1.KEY_COLS + ["d"]
        tcols = [c for c in names if c.startswith("tf_")]
        other = [c for c in names if c not in tcols and c not in keycols]
        if other != [TFIDF_N_ROWS_COL] or len(tcols) != 500:
            raise ValueError(f"단위날짜_tfidf 칸이 예상과 다름: 키·tf 밖 {other}, tf {len(tcols)}")
        voc = pd.read_csv(TFIDF_VOCAB_PATH, encoding="utf-8-sig")
        if voc["열이름"].astype(str).tolist() != tcols:
            raise ValueError("단위날짜_tfidf_어휘.csv 열이름 순서가 parquet tf 칸과 다름")
        k = pq.read_table(TFIDF_PATH, columns=keycols + [TFIDF_N_ROWS_COL]).to_pandas()
        n_null_p2 = int(k["train_prdlst_code"].isna().sum())
        n_null_p3 = int(k["train_prdlst_code_2"].isna().sum())
        kd = pd.DataFrame({"sgg_code": k["sgg_code"].astype(str), "train_prdlst_code": _fill_code(k["train_prdlst_code"]),
                           "train_prdlst_code_2": _fill_code(k["train_prdlst_code_2"])})
        u = H.group_ids(kd, "unit")
        dd = H.day_idx(pd.to_datetime(k["d"]).values.astype("datetime64[D]"))
        tk = u * M + dd
        if len(np.unique(tk)) != len(tk) or len(tk) != self.n_rows_full:
            raise ValueError(f"단위날짜_tfidf 키 중복 또는 행 수 {len(tk)} != 단위x날짜 행 {self.n_rows_full}")
        ridx = self._row_of(tk)                                  # tfidf 행 i -> 전체 행 번호
        if len(np.unique(ridx)) != self.n_rows_full:
            raise ValueError("단위날짜_tfidf 가 단위x날짜 행과 1:1 이 아님")
        nr_eq = bool(np.array_equal(k[TFIDF_N_ROWS_COL].to_numpy(), rows_full["n_samples"].to_numpy()[ridx]))
        X = np.empty((self.n_rows_full, len(tcols)), dtype=np.float32)
        nnan = 0
        for a in range(0, len(tcols), 50):
            cs = tcols[a:a + 50]
            tb = pq.read_table(TFIDF_PATH, columns=cs)
            for j, c in enumerate(cs):
                col = tb.column(c).to_numpy(zero_copy_only=False)
                if col.dtype != np.float32:
                    raise ValueError(f"{c}: float32 아님 ({col.dtype})")
                nnan += int(np.isnan(col).sum())
                X[ridx, a + j] = col
            del tb
        chk = {"rows": int(len(tk)), "matched": int(len(ridx)), "n_cols": len(tcols), "null_train_prdlst_code_filled": n_null_p2,
               "null_train_prdlst_code_2_filled": n_null_p3, "n_rows_eq_n_samples": nr_eq, "nan_cells": nnan,
               "excluded": [TFIDF_N_ROWS_COL]}
        return X, chk

    # ---------------------------------------------------------------- 특성
    def features(self, ctx, sp, pos, h, blocks, k):
        pos = np.asarray(pos)
        n = len(pos)
        out = np.empty((n, len(nb_names(blocks))), dtype=np.float32)
        fr = self.frow[sp][pos]
        v = ctx.sp[sp]
        hh = np.full(n, int(h), dtype=np.int64) if np.isscalar(h) else np.asarray(h, dtype=np.int64)
        col = 0
        for b in NB_ORDER:
            if b not in blocks:
                continue
            if b == "n1":
                if self.hist is None:
                    raise RuntimeError("N1 자료 없음")
                w = len(NB_PERIODS) * len(NB_KINDS) * len(self.clusters)
                g = v["grp"]["unit"][pos]
                t = v["hd"][pos] - hh                              # T 의 날짜 번호(History base)
                for a in range(0, n, CHUNK):
                    c = slice(a, min(n, a + CHUNK))
                    out[c, col:col + w] = self.hist.features(g[c], t[c], float(k))
                col += w
            elif b == "n2":
                w = len(self.clusters)
                out[:, col:col + w] = self.N2[fr]
                col += w
            elif b == "n0":
                w = self.N0.shape[1]
                out[:, col:col + w] = self.N0[fr]
                col += w
        assert col == out.shape[1]
        return out


def get_nb(ctx, blocks):
    """ctx 에 제품명 자료를 한 번 만들어 둔다(필요한 묶음만 - N0 500칸은 쓸 때만 읽음)."""
    X = getattr(ctx, "_nb", None)
    need = set(blocks)
    if X is not None:
        have = {"n1": X.hist is not None, "n2": X.N2 is not None, "n0": X.N0 is not None}
        if all(have[b] for b in need):
            return X
        need |= {b for b, v in have.items() if v}
    X = NbData(ctx, need=tuple(b for b in NB_ORDER if b in need), log=getattr(ctx, "log", None))
    ctx._nb = X
    return X


def nb_features(ctx, sp, pos, h, blocks, k):
    """[제품명 칸] float32 (n, len(nb_names(blocks)))."""
    return get_nb(ctx, blocks).features(ctx, sp, pos, h, blocks, k)


def nb_settings_rows(ctx_check=None):
    """표 '설정' 시트 줄."""
    return [("N 제품명 변수 자료", f"data/데이터셋/제품명/{NB_SAMPLE_PATH.name}(시료 {NB_N_SAMPLES:,}행 = 시료_시군구.parquet 행 순서, lims_data_key 로 확인, "
                                 f"nb_* 21칸 = {NB_DEF_PATH.name}), {TFIDF_PATH.name}(단위x날짜 291,812행, tf_* 500칸 + n_rows). "
                                 "다른 세션([질문])이 만든 파일(사용자 승인 2026-10-19), 읽기만 함"),
            ("N1 과거 뭉치 이력(A3_N1·A3_N1N2)", N1_RULE),
            ("N2 그날 뭉치 멀티핫(A3_N2·A3_N1N2)", N2_RULE),
            ("N0 그날 제품명 TF-IDF(A3_N0)", N0_RULE),
            ("N 전제·한계(보고서)", "N2·N0 는 목표일 d 의 그 단위 시료 제품명으로 만든 값 -> 운영에서 예측할 때 그날 검사할 제품명을 입력받아야 쓸 수 있음. "
                                 "제품명을 입력받지 않는 운영이면 N1(과거 시료 제품명만 씀)만 해당. 예측 때 제품명 입력 여부는 보고서 한계로 적음(사용자 지시 2026-10-19). "
                                 "뭉치 정규식은 학습 기간(mnfctur_de <= 2024-06-30) 행으로만 골랐다고 뭉치_정의.csv '선정 기간' 칸에 적혀 있음"),
            ("N 실험", "A3_N1 = A3(이력 9칸, k*) + N1 84칸, A3_N2 = A3 + N2 21칸, A3_N1N2 = A3 + N1 + N2 105칸, A3_N0 = A3 + N0 500칸. "
                       "시드 42·43·44, 단계 'n'. N1 의 k = A3 와 같은 k*")]
