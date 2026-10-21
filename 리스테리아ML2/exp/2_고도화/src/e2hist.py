# -*- coding: utf-8 -*-
"""A. 과거 검출 이력 변수와 D3 WOE.

시료 단위 건수(검사 수 n, 검출 수 s)를 무리(level)별 일별 배열로 만들고 누적합으로 창 합계를 구한다.
창 = 예측하는 날 T 의 전날까지 P 일: 날짜 T-P .. T-1 (T 당일과 그 뒤 시료는 들어가지 않음 - 사용자 승인).
누적합 Cn[g, t] = 날짜 번호 < t 인 시료 수  ->  창 합계 = Cn[g, T] - Cn[g, T-P]  (번호는 [0, Dh] 로 자름. 자료 밖은 0건).

무리:
  p3   = 품목군 3단계(전국, train_prdlst_code_2)
  sgg  = 시군구(전 품목)
  unit = 단위(시군구 x 2단계 x 3단계)
  sido, p2 = WOE 용(시도 = sgg_code 앞 2자리, 2단계 전국)
코드 빈 값은 1차와 같이 '코드없음' 하나의 무리.
"""
import numpy as np
import pandas as pd

import config as C1
import e2cfg as E

LEVEL_COLS = {
    "p3": ["train_prdlst_code_2"],
    "sgg": ["sgg_code"],
    "unit": C1.KEY_COLS,
    "sido": ["sido"],
    "p2": ["train_prdlst_code"],
}


def _key(df: pd.DataFrame, lv: str) -> np.ndarray:
    cols = LEVEL_COLS[lv]
    if len(cols) == 1:
        return df[cols[0]].astype(str).to_numpy(dtype=object)
    return (df[cols[0]].astype(str) + "|" + df[cols[1]].astype(str) + "|" + df[cols[2]].astype(str)).to_numpy(dtype=object)


class History:
    """samples: 1차 data.load_samples() 결과(sgg_code, train_prdlst_code, train_prdlst_code_2, d, pos)."""

    def __init__(self, samples: pd.DataFrame, levels=("p3", "sgg", "unit", "sido", "p2")):
        s = samples.copy()
        s["sido"] = s["sgg_code"].astype(str).str[:2]
        d = s["d"].to_numpy().astype("datetime64[D]")
        self.base = d.min()
        sd = (d - self.base).astype(np.int64)
        self.Dh = int(sd.max()) + 1
        pos = s["pos"].to_numpy().astype(np.int64)
        self.n_samples, self.n_pos = len(s), int(pos.sum())
        self.Cn, self.Cs, self.index = {}, {}, {}
        for lv in levels:
            codes, uniq = pd.factorize(_key(s, lv), sort=True)
            G = len(uniq)
            flat = codes.astype(np.int64) * self.Dh + sd
            n = np.bincount(flat, minlength=G * self.Dh).reshape(G, self.Dh)
            k = np.bincount(flat, weights=pos, minlength=G * self.Dh).reshape(G, self.Dh).astype(np.int64)
            Cn = np.zeros((G, self.Dh + 1), dtype=np.int32)
            Cs = np.zeros((G, self.Dh + 1), dtype=np.int32)
            np.cumsum(n, axis=1, out=Cn[:, 1:])
            np.cumsum(k, axis=1, out=Cs[:, 1:])
            self.Cn[lv], self.Cs[lv] = Cn, Cs
            self.index[lv] = {u: i for i, u in enumerate(uniq)}
        tn = np.bincount(sd, minlength=self.Dh)
        ts = np.bincount(sd, weights=pos, minlength=self.Dh).astype(np.int64)
        self.CN = np.zeros(self.Dh + 1, dtype=np.int64)
        self.CS = np.zeros(self.Dh + 1, dtype=np.int64)
        np.cumsum(tn, out=self.CN[1:])
        np.cumsum(ts, out=self.CS[1:])

    # ---------------------------------------------------------------- 번호
    def group_ids(self, rows: pd.DataFrame, lv: str) -> np.ndarray:
        df = rows if "sido" in rows.columns else rows.assign(sido=rows["sgg_code"].astype(str).str[:2])
        g = pd.Series(_key(df, lv)).map(self.index[lv])
        if g.isna().any():
            raise ValueError(f"이력 무리 {lv} 에 없는 키 {int(g.isna().sum())}개")
        return g.to_numpy().astype(np.int64)

    def day_idx(self, d) -> np.ndarray:
        return (np.asarray(d).astype("datetime64[D]") - self.base).astype(np.int64)

    # ---------------------------------------------------------------- 창 합계
    def counts(self, lv, g, t, P):
        """창 T-P .. T-1 의 (n, s). t = T 의 날짜 번호(base 기준)."""
        hi = np.clip(t, 0, self.Dh)
        lo = np.clip(t - P, 0, self.Dh)
        Cn, Cs = self.Cn[lv], self.Cs[lv]
        return (Cn[g, hi] - Cn[g, lo]).astype(np.float64), (Cs[g, hi] - Cs[g, lo]).astype(np.float64)

    def totals(self, t, P):
        hi = np.clip(t, 0, self.Dh)
        lo = np.clip(t - P, 0, self.Dh)
        return (self.CN[hi] - self.CN[lo]).astype(np.float64), (self.CS[hi] - self.CS[lo]).astype(np.float64)


def hist_names(sets):
    names = []
    for st in sets:
        for kind, P, lv in E.HIST_SETS[st]:
            names.append(f"hist_{kind}{'5y' if P == E.P5 else '1y'}_{lv}")
    return names


def hist_features(H: History, grp: dict, hd: np.ndarray, h, sets, k) -> np.ndarray:
    """(n, 변수 수) float32. grp[lv] = 행의 무리 번호, hd = 목표일 d 의 날짜 번호, h = 앞당김(정수 또는 행별 배열).
    rate = (s + k*p0)/(n + k), p0 = 같은 창 전체 시료 검출률(창에 시료가 하나도 없으면 NaN -> rate NaN)
    logn = log(1 + n)
    sd   = sqrt(m(1-m)/(n + k + 1)), m = rate 와 같은 식(같은 창·p0·k) - 베타 사후 Beta(s + k*p0, n - s + k*(1-p0)) 의 표준편차
           (HX 'sd1', 사용자 지시 2026-10-19. p0 NaN 이면 NaN)"""
    t = hd - np.asarray(h, dtype=np.int64)
    items = [it for st in sets for it in E.HIST_SETS[st]]
    out = np.empty((len(t), len(items)), dtype=np.float32)
    p0 = {}
    for j, (kind, P, lv) in enumerate(items):
        n, s = H.counts(lv, grp[lv], t, P)
        if kind in ("rate", "sd"):
            if P not in p0:
                N, S = H.totals(t, P)
                with np.errstate(invalid="ignore", divide="ignore"):
                    p0[P] = np.where(N > 0, S / np.where(N > 0, N, 1.0), np.nan)
            if kind == "rate":
                out[:, j] = (s + k * p0[P]) / (n + k)
            else:                                            # sd: rate 칸은 위 식 그대로(기존 계산 불변)
                m = (s + k * p0[P]) / (n + k)
                out[:, j] = np.sqrt(m * (1.0 - m) / (n + k + 1.0))
        elif kind == "logn":
            out[:, j] = np.log1p(n)
        else:
            raise ValueError(kind)
    return out


# ---------------------------------------------------------------- HX (사용자 지시 2026-10-19): 설명·날짜 시계 확인 결과
HX_RULE = ("A3 + 1년 창 칸(창 = 기준일 ∈ [T−365, T−1] 시료, T = d − h, e2hist 그대로 T 당일·이후 제외, 3수준 = 품목군 3단계(전국)·시군구(전 품목)·"
           "단위(시군구 x 2단계 x 3단계)). logn1 = log(1 + n) (n = 그 수준 창 안 시료 수). sd1 = sqrt(m(1−m)/(n+k+1)), m = (s + k·p0)/(n + k) "
           "(1년 스무딩 검출률 rate1 과 같은 m, p0 = 같은 창 전국 검출률, k = k*) - 베타 사후 Beta(s + k·p0, n − s + k(1 − p0)) 의 표준편차, "
           "p0 가 없으면(창에 전국 시료 0) 결측. A3_L1 = A3 + logn1 3칸, A3_U1 = A3 + sd1 3칸, A3_L1U1 = A3 + 6칸. 칸 이름 hist_logn1y_<수준>·hist_sd1y_<수준>")
HX_DATECLOCK_CSV = E.EXP2_DIR / "logs" / "selftest_hx_dateclock.csv"     # selftest_hx.py 가 씀(예측일 T 연도별 평균)


def hx_dateclock_text():
    """표 '설정' 시트용: selftest_hx 의 날짜 시계 확인표 요약(학습 연도별 평균의 처음·끝 해와 해마다 늘어난 횟수, 검증·시험 평균)."""
    if not HX_DATECLOCK_CSV.exists():
        return f"확인표 없음({HX_DATECLOCK_CSV.name} - src/selftest_hx.py 를 돌리면 생김)"
    d = pd.read_csv(HX_DATECLOCK_CSV, encoding="utf-8-sig")
    cols = [c for c in d.columns if c.startswith("hist_logn1y_") or c.startswith("hist_sd1y_")]
    tr = d[d["split"] == "train"].sort_values("year")
    parts = []
    for c in cols:
        v = tr[c].to_numpy(dtype=np.float64)
        up = int(np.sum(np.diff(v) > 0))
        t = f"{c}: 학습 {int(tr['year'].iloc[0])}년 {v[0]:.4g} -> {int(tr['year'].iloc[-1])}년 {v[-1]:.4g} (해마다 늘어난 해 {up}/{len(v) - 1})"
        for sp, lab in (("val", "검증"), ("test", "시험")):
            s = d[d["split"] == sp]
            if len(s):
                t += f", {lab} {np.average(s[c], weights=s['n']):.4g}"
        parts.append(t)
    return f"{HX_DATECLOCK_CSV.parent.name}/{HX_DATECLOCK_CSV.name} (학습 = 시드 42 h, 검증·시험 = h 1~60 전부의 예측일 T 연도별 평균): " + "; ".join(parts)


def woe_names():
    return [f"woe_{lv}" for lv in E.WOE_LEVELS]


def woe_features(H: History, grp: dict, hd: np.ndarray, h) -> np.ndarray:
    """WOE = ln((s+0.5)/(S+0.5)) - ln((n-s+0.5)/(N-S+0.5)), 5년 창(T 전날까지). (n, 4) float32."""
    t = hd - np.asarray(h, dtype=np.int64)
    N, S = H.totals(t, E.P5)
    e = E.WOE_EPS
    out = np.empty((len(t), len(E.WOE_LEVELS)), dtype=np.float32)
    for j, lv in enumerate(E.WOE_LEVELS):
        n, s = H.counts(lv, grp[lv], t, E.P5)
        out[:, j] = np.log((s + e) / (S + e)) - np.log((n - s + e) / (N - S + e))
    return out
