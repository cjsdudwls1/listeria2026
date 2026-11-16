"""과거 이력(HIS1~HIS4)·7-a 타겟 인코딩·단계 4~6 범주 숫자 인코딩 — 공통 집계.

계획서 §5 표(HIS1~HIS4), §5-1(계산 규칙), 부록 A-3(HIS1~HIS4·7-a 공식), 부록 A-6 끝 항목(단계 4~6
범주 숫자 인코딩), §6 단계 7(7-a), 단계 0-6·0-10, §2-2 RR3('코드없음'), 부록 B-6(누수 assert).

규칙(§5-1 그대로)
- 기준일 C = 그 행의 T 직전 06-30(T 가 07-01 ~ 12-31 이면 그해 06-30, 01-01 ~ 06-30 이면 전해 06-30).
  train 행 = 자기 T 로, validation·test 행 = 그 fold(refit) train 끝(채점 T 범위에선 둘이 같음 — 확인함).
- 집계 = d ≤ C 인 시료만(자기 시료는 d > C 라 안 들어감 — 행마다 d > C 를 assert).
- 기간: 5년 = [C−1825, C](1,826일), 1년 = [C−364, C](365일), 7-a·A-6 의 '전체' = [2015-04-07, C].
- p0 = 같은 기간 전체 검출률, m = 1년 원 검출률(n = 0 이면 p0), s_all·n_all = 같은 기간 전체 검출·검사.
- 창 전체가 자료 시작(2015-04-07) 전이거나 그 기간 n_all = 0 이면 그 칼럼 NaN.
- 3수준 = 3단계 전국(국내·해외 전체)·시군구 전 품목(해외는 나라 = 'OV'+alpha-3 `sgg`)·단위 (3단계, `sgg`).
  HIS3 WOE 수준 = `sido`(해외 'OV' 한 묶음)·`sgg`·`p2`·`p3`.
- RR3: '코드없음' 3단계 칸은 (p2, '코드없음') 으로 나눠 셈(3단계 수준·단위·7-a·A-6 의 p3).
- 단계 4·5·6: 줄이기·뽑기 전 시료(모집단)로 계산한 뒤 행을 줄임 → 모집단(Population)과 행(rows)을 따로 받음.

공식(부록 A-3)
- HIS1 = (s + k·p0) ÷ (n + k)
- HIS2 = log(1 + n), 창이 자료 시작 전으로 넘어가면 n × 기간 일수 ÷ 실제 일수
- HIS3 = ln((s + 0.5)/(s_all + 0.5)) − ln((n − s + 0.5)/(n_all − s_all + 0.5)), 5년
- HIS4 = √(m(1−m) ÷ (n + k + 1))
- 7-a: p̂_u = (s_u + k_u·p̂_p3) ÷ (n_u + k_u), p̂_p3 = (s_p3 + k_p3·p0) ÷ (n_p3 + k_p3)
- A-6: 범주 값마다 [2015-04-07, C] 의 (s + k·p0) ÷ (n + k)

구현 세부(계획서에 없음 — 담당자 확인):
 (a) 칼럼 이름: HIS1 `hist_rate{5y,1y}_{p3,sgg,u}_k{010…200}`, HIS2 `hist_logn{5y,1y}_{p3,sgg,u}`,
     HIS3 `woe_{sido,sgg,p2,p3}`(계획서 그대로), HIS4 `hist_unc1y_{p3,sgg,u}_k{010…200}`,
     A-6 `catenc_<칼럼>`. 수준 접미사 p3 = 3단계 전국, sgg = 시군구(나라) 전 품목, u = (3단계, sgg).
 (b) HIS2 '실제 일수' = 창 [lo, C] 중 자료 시작(2015-04-07) 이후 일수 = C − 2015-04-07 + 1
     (lo < 2015-04-07 일 때만 환산, 환산 뒤 log(1 + n × 기간 일수 ÷ 실제 일수)).
 (c) '코드없음' 3단계 키 = 문자열 '<p2>|코드없음'(실제 품목 코드와 겹치지 않음).
 (d) 모집단 = 주어진 시료 표(단계 0-1 결과 — RR2·RR4·RR8 로 뺀 행은 없음) 그대로.
 (e) 범주 값에 결측(null)이 있으면 멈춤 — aw_label null 은 부르는 쪽이 '결측' 범주 문자열로 바꿔 넘김(부록 A-4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .folds import Split, as_date, day, day_ints
from .samples import NO_CODE

DATA_START_STR = "2015-04-07"
DATA_START = day(DATA_START_STR)               # §5-1 자료 시작
K_VALUES = (10, 30, 50, 100, 200)              # 단계 2-1 k 후보 = 7-a k_u·k_p3 후보(§6 단계 7)
PERIOD_DAYS = {"5y": 1826, "1y": 365}          # §5-1
HIS_LENGTHS = ("5y", "1y")
TE_LENGTHS = ("1y", "5y", "all")               # 7-a 길이(§6 단계 7)
HIS_LEVELS = ("p3", "sgg", "u")                # §5-1 3수준
WOE_LEVELS = ("sido", "sgg", "p2", "p3")       # §5-1 HIS3
CAT5 = ("sido", "sgg", "p2", "p3", "src_route")  # 범주 5(§6 1-1 입력)
WOE_EPS = 0.5
KEY_COLS = ("p2", "p3", "sgg", "sido")
_UNIT_SEP = "\x1f"


class LeakageError(AssertionError):
    """이력·인코딩 집계에 d > C 시료·자기 시료가 들어갈 수 있는 입력(부록 B-6)."""


# ---------------------------------------------------------------- 기준일 C
def c_from_T(T) -> np.ndarray:
    """§1-1·§5-1: T 직전 06-30(일수 int64). T = 07-01 → 그해 06-30, T = 06-30 → 전해 06-30."""
    t = as_date(day_ints(T))
    year = t.astype("datetime64[Y]").astype(np.int64) + 1970
    month = t.astype("datetime64[M]").astype(np.int64) % 12 + 1
    cy = np.where(month >= 7, year, year - 1)
    jun1 = ((cy - 1970).astype("datetime64[Y]").astype("datetime64[M]") + np.timedelta64(5, "M")).astype("datetime64[D]")
    return (jun1 + np.timedelta64(29, "D")).astype(np.int64)


def c_for_train_rows(T, split: Split) -> np.ndarray:
    """train 행 C = 자기 T 직전 06-30. C ≤ train 끝인지 확인."""
    C = c_from_T(T)
    if (C > split.train_end).any():
        raise LeakageError(f"{split.name}: train 행 C 가 train 끝을 넘음")
    return C


def c_for_eval_rows(T, split: Split) -> np.ndarray:
    """validation·test 행 C = 그 fold(refit) train 끝(§5-1). T 직전 06-30 과도 같은지 확인."""
    T = day_ints(T)
    if ((T < split.score_start) | (T > split.score_end)).any():
        raise ValueError(f"{split.name}: 채점 범위 밖 T")
    if (c_from_T(T) != split.train_end).any():
        raise LeakageError(f"{split.name}: 채점 T 의 'T 직전 06-30' 이 train 끝과 다름")
    return np.full(T.shape, split.train_end, dtype=np.int64)


# ---------------------------------------------------------------- 키
def _str_array(x, name: str) -> np.ndarray:
    a = np.asarray(pd.Series(x).to_numpy(dtype=object), dtype=object)
    bad = pd.isna(pd.Series(a)).to_numpy() | ~np.fromiter((isinstance(v, str) for v in a), bool, a.shape[0])
    if bad.any():
        raise ValueError(f"{name} 에 결측·문자열 아닌 값 {int(bad.sum())}개")
    return a


def p3_key(p2, p3) -> np.ndarray:
    """RR3: 3단계 '코드없음' → (p2, '코드없음') 키 '<p2>|코드없음'."""
    p2 = _str_array(p2, "p2")
    p3 = _str_array(p3, "p3")
    out = p3.copy()
    nc = p3 == NO_CODE
    if nc.any():
        out[nc] = np.array([f"{a}|{NO_CODE}" for a in p2[nc]], dtype=object)
    return out


def unit_key(p3k, sgg) -> np.ndarray:
    p3k = _str_array(p3k, "p3 키")
    sgg = _str_array(sgg, "sgg")
    if any(_UNIT_SEP in s for s in sgg):
        raise ValueError("sgg 에 구분 문자")
    return np.array([a + _UNIT_SEP + b for a, b in zip(p3k, sgg)], dtype=object)


def level_values(df: pd.DataFrame, need_src_route: bool = False) -> dict[str, np.ndarray]:
    """수준별 키: p3(=RR3 키)·sgg·u·sido·p2(+src_route)."""
    miss = [c for c in KEY_COLS if c not in df.columns]
    if miss:
        raise KeyError(f"칼럼 없음: {miss}")
    p3k = p3_key(df["p2"], df["p3"])
    sgg = _str_array(df["sgg"], "sgg")
    out = {"p3": p3k, "sgg": sgg, "u": unit_key(p3k, sgg),
           "sido": _str_array(df["sido"], "sido"), "p2": _str_array(df["p2"], "p2")}
    if need_src_route or "src_route" in df.columns:
        if "src_route" in df.columns:
            out["src_route"] = _str_array(df["src_route"], "src_route")
        else:
            raise KeyError("src_route 칼럼 없음")
    return out


# ---------------------------------------------------------------- 모집단·집계
def _y01(y) -> np.ndarray:
    a = pd.Series(y)
    if a.isna().any():
        raise ValueError("y 결측")
    v = a.to_numpy()
    if not np.isin(v, (0, 1)).all():
        raise ValueError("y 는 0/1 만")
    return v.astype(np.int64)


@dataclass
class Population:
    """집계 대상 시료(시료 1개 = 1번, h 복제 없음). keys = 수준 → 시료별 값."""
    d: np.ndarray
    y: np.ndarray
    keys: dict[str, np.ndarray]
    _codes: dict[str, tuple[np.ndarray, pd.Index]] = field(default_factory=dict, repr=False)
    audit: list[dict] = field(default_factory=list, repr=False)  # C 마다 실제로 모은 시료 d 최대(부록 B-6)

    @classmethod
    def from_samples(cls, samples: pd.DataFrame, d_col: str = "d") -> "Population":
        if "lims_data_key" in samples.columns and samples["lims_data_key"].duplicated().any():
            raise ValueError("모집단 lims_data_key 중복(h 복제 행을 넣지 말 것)")
        d = day_ints(samples[d_col])
        if (d < DATA_START).any():
            raise ValueError(f"자료 시작({DATA_START_STR}) 전 d 시료가 있음")
        keys = level_values(samples, need_src_route="src_route" in samples.columns)
        return cls(d=d, y=_y01(samples["y"]), keys=keys)

    def add_level(self, name: str, values) -> None:
        """사용자 범주(season·aw_label 등) 수준 추가 — 시료별 값(결측 금지)."""
        v = _str_array(values, name)
        if v.shape[0] != self.d.shape[0]:
            raise ValueError(f"{name} 길이가 모집단과 다름")
        if name in self.keys:
            raise ValueError(f"이미 있는 수준: {name}")
        self.keys[name] = v

    def _level(self, level: str) -> tuple[np.ndarray, pd.Index]:
        if level not in self._codes:
            codes, uniq = pd.factorize(self.keys[level])
            self._codes[level] = (codes.astype(np.int64), pd.Index(uniq))
        return self._codes[level]

    def window_mask(self, c: int, length: str) -> np.ndarray:
        lo = window_lo(np.array([c]), length)[0]
        m = (self.d >= lo) & (self.d <= c)
        if m.any():
            dmax = int(self.d[m].max())
            if dmax > c:  # 부록 B-6: 집계 시료 d ≤ C
                raise LeakageError("집계 시료 d > C")
            self.audit.append({"C": int(c), "length": length, "n": int(m.sum()), "d_max_used": dmax})
        return m

    def totals(self, row_C: np.ndarray, length: str) -> tuple[np.ndarray, np.ndarray]:
        """행마다 그 기간 전체 검출 s_all·검사 n_all."""
        uc, inv = np.unique(row_C, return_inverse=True)
        s = np.empty(uc.shape[0])
        n = np.empty(uc.shape[0])
        for j, c in enumerate(uc):
            m = self.window_mask(int(c), length)
            s[j] = self.y[m].sum()
            n[j] = m.sum()
        return s[inv], n[inv]

    def counts(self, level: str, row_values, row_C: np.ndarray, length: str) -> tuple[np.ndarray, np.ndarray]:
        """행마다 (그 수준 값, 그 기간) 의 검출 s·검사 n. 모집단에 없는 값 = 0·0."""
        codes, uniq = self._level(level)
        rv = _str_array(row_values, level)
        rc = uniq.get_indexer(rv)
        K = len(uniq)
        uc, inv = np.unique(row_C, return_inverse=True)
        GS = np.zeros((uc.shape[0], K + 1))
        GN = np.zeros((uc.shape[0], K + 1))
        for j, c in enumerate(uc):
            m = self.window_mask(int(c), length)
            GS[j, :K] = np.bincount(codes[m], weights=self.y[m], minlength=K)
            GN[j, :K] = np.bincount(codes[m], minlength=K)
        col = np.where(rc >= 0, rc, K)  # 마지막 칸 = 0(없던 값)
        return GS[inv, col], GN[inv, col]


def window_lo(C: np.ndarray, length: str) -> np.ndarray:
    if length == "all":
        return np.full(np.shape(C), DATA_START, dtype=np.int64)
    if length not in PERIOD_DAYS:
        raise ValueError(f"모르는 기간: {length!r}")
    return np.asarray(C, dtype=np.int64) - PERIOD_DAYS[length] + 1


@dataclass(frozen=True)
class Rows:
    d: np.ndarray
    C: np.ndarray
    keys: dict[str, np.ndarray]
    index: pd.Index


def prepare_rows(rows: pd.DataFrame, d_col: str = "d", c_col: str = "C") -> Rows:
    """행(train 행·채점 시료·운영 행): d·C·단위 키. 자기 시료가 집계에 못 들어가게 d > C 를 assert."""
    if c_col not in rows.columns:
        raise KeyError("행에 C 칼럼 없음(c_for_train_rows·c_for_eval_rows 로 만들 것)")
    d = day_ints(rows[d_col])
    C = day_ints(rows[c_col])
    if (d <= C).any():
        raise LeakageError(f"d ≤ C 인 행 {int((d <= C).sum())}개 — 자기 시료가 집계에 들어갈 수 있음")
    return Rows(d=d, C=C, keys=level_values(rows), index=rows.index)


def _as_rows(rows) -> Rows:
    return rows if isinstance(rows, Rows) else prepare_rows(rows)


def _valid(C: np.ndarray, n_all: np.ndarray) -> np.ndarray:
    """§5-1: 창 전체가 자료 시작 전(C < 2015-04-07)이거나 n_all = 0 이면 NaN."""
    return (C >= DATA_START) & (n_all > 0)


# ---------------------------------------------------------------- 이름
def k_tag(k: int) -> str:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not (1 <= int(k) <= 999):
        raise ValueError(f"k 는 1~999 정수: {k!r}")
    return f"k{int(k):03d}"


def his_columns(ks: Sequence[int] = K_VALUES) -> list[dict]:
    """칼럼 목록(§5 표 순서: HIS1 → HIS2 → HIS3 → HIS4) — 칼럼·ID·기간·수준·k·window 끝."""
    out = []
    for L in HIS_LENGTHS:
        for lv in HIS_LEVELS:
            for k in ks:
                out.append({"column": f"hist_rate{L}_{lv}_{k_tag(k)}", "id": "HIS1", "length": L, "level": lv, "k": int(k)})
    for L in HIS_LENGTHS:
        for lv in HIS_LEVELS:
            out.append({"column": f"hist_logn{L}_{lv}", "id": "HIS2", "length": L, "level": lv, "k": None})
    for lv in WOE_LEVELS:
        out.append({"column": f"woe_{lv}", "id": "HIS3", "length": "5y", "level": lv, "k": None})
    for lv in HIS_LEVELS:
        for k in ks:
            out.append({"column": f"hist_unc1y_{lv}_{k_tag(k)}", "id": "HIS4", "length": "1y", "level": lv, "k": int(k)})
    for r in out:
        r["window_end"] = "C"
    return out


def te7a_name(length: str, k_u: int, k_p3: int) -> str:
    """부록 A-12 판 이름 꼴: 7-a-<1y·5y·all>-ku<k_u>-kp<k_p3>(3자리)."""
    if length not in TE_LENGTHS:
        raise ValueError(f"7-a 길이: {length!r}")
    return f"7-a-{length}-ku{k_tag(k_u)[1:]}-kp{k_tag(k_p3)[1:]}"


# ---------------------------------------------------------------- HIS1~HIS4
def history_features(pop: Population, rows, ks: Sequence[int] = K_VALUES) -> pd.DataFrame:
    """HIS1~HIS4 칼럼 전부(k 값마다). rows = DataFrame(d·C·p2·p3·sgg·sido) 또는 Rows."""
    R = _as_rows(rows)
    ks = [int(k) for k in ks]
    for k in ks:
        k_tag(k)
    C = R.C
    cols: dict[str, np.ndarray] = {}
    for L in HIS_LENGTHS:
        s_all, n_all = pop.totals(C, L)
        ok = _valid(C, n_all)
        with np.errstate(invalid="ignore", divide="ignore"):
            p0 = np.where(ok, s_all / np.where(n_all > 0, n_all, 1), np.nan)
        lo = window_lo(C, L)
        actual = C - np.maximum(lo, DATA_START) + 1
        scale = np.where(lo < DATA_START, PERIOD_DAYS[L] / np.maximum(actual, 1), 1.0)
        for lv in HIS_LEVELS:
            s, n = pop.counts(lv, R.keys[lv], C, L)
            for k in ks:
                cols[f"hist_rate{L}_{lv}_{k_tag(k)}"] = np.where(ok, (s + k * p0) / (n + k), np.nan)
            cols[f"hist_logn{L}_{lv}"] = np.where(ok, np.log1p(n * scale), np.nan)
            if L == "1y":
                with np.errstate(invalid="ignore", divide="ignore"):
                    m = np.where(n > 0, s / np.where(n > 0, n, 1), p0)
                for k in ks:
                    cols[f"hist_unc1y_{lv}_{k_tag(k)}"] = np.where(ok, np.sqrt(m * (1 - m) / (n + k + 1)), np.nan)
        if L == "5y":
            for lv in WOE_LEVELS:
                s, n = pop.counts(lv, R.keys[lv], C, L)
                with np.errstate(invalid="ignore", divide="ignore"):
                    woe = (np.log((s + WOE_EPS) / (s_all + WOE_EPS))
                           - np.log((n - s + WOE_EPS) / (n_all - s_all + WOE_EPS)))
                cols[f"woe_{lv}"] = np.where(ok, woe, np.nan)
    order = [c["column"] for c in his_columns(ks)]
    return pd.DataFrame({c: cols[c] for c in order}, index=R.index)


# ---------------------------------------------------------------- 7-a
def te7a_grid(pop: Population, rows, length: str, k_u_values: Iterable[int],
              k_p3_values: Iterable[int]) -> dict[tuple[int, int], np.ndarray]:
    """7-a 점수 p̂_u(부록 A-3) — (k_u, k_p3) 마다 행 길이 배열. 창이 자료 시작 전·n_all = 0 이면 NaN."""
    if length not in TE_LENGTHS:
        raise ValueError(f"7-a 길이: {length!r}")
    R = _as_rows(rows)
    C = R.C
    s_all, n_all = pop.totals(C, length)
    ok = _valid(C, n_all)
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = np.where(ok, s_all / np.where(n_all > 0, n_all, 1), np.nan)
    s_p3, n_p3 = pop.counts("p3", R.keys["p3"], C, length)
    s_u, n_u = pop.counts("u", R.keys["u"], C, length)
    out = {}
    kps = [int(k) for k in k_p3_values]
    for kp in kps:
        k_tag(kp)
    for ku in k_u_values:
        k_tag(ku)
        for kp in kps:
            p_p3 = (s_p3 + kp * p0) / (n_p3 + kp)
            out[(int(ku), kp)] = np.where(ok, (s_u + ku * p_p3) / (n_u + ku), np.nan)
    return out


# ---------------------------------------------------------------- 부록 A-6 끝 항목
def category_rate(pop: Population, level: str, row_values, rows, k: int) -> np.ndarray:
    """범주 값 [2015-04-07, C] 검출률 (s + k·p0) ÷ (n + k). level = 모집단 수준 이름(add_level 로 더한 것 포함)."""
    k_tag(k)
    R = _as_rows(rows)
    C = R.C
    s_all, n_all = pop.totals(C, "all")
    ok = _valid(C, n_all)
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = np.where(ok, s_all / np.where(n_all > 0, n_all, 1), np.nan)
    s, n = pop.counts(level, row_values, C, "all")
    return np.where(ok, (s + k * p0) / (n + k), np.nan)


def category_rates(pop: Population, rows, k: int, columns: Sequence[str] = CAT5) -> pd.DataFrame:
    """범주 5(sido·sgg·p2·p3·src_route) 숫자 인코딩 `catenc_<칼럼>` — p3 는 RR3 키.
    season·aw_label 은 pop.add_level(…) 뒤 category_rate 로(행 값은 부르는 쪽이 줌)."""
    R = _as_rows(rows)
    out = {}
    for c in columns:
        if c not in R.keys:
            raise KeyError(f"행에 {c} 없음")
        out[f"catenc_{c}"] = category_rate(pop, c, R.keys[c], R, k)
    return pd.DataFrame(out, index=R.index)


# ---------------------------------------------------------------- 시료 표 읽기(test 막기)
def read_samples_blind_test(path, columns: Sequence[str], cut: int | None = None) -> pd.DataFrame:
    """시료 표(단계 0-1 결과)를 읽되 y 는 d ≤ cut(기본 refit train 끝 2025-06-30) 행만 읽음(§1-3 test 막기).
    d > cut 행의 y = <NA>(Int8) — 그 행이 모집단(Population)에 들어가면 'y 결측' 으로 멈춤."""
    import datetime as _dt

    import pyarrow.parquet as pq

    from .folds import SPLITS

    cut = SPLITS["refit"].train_end if cut is None else int(cut)
    cut_date = _dt.date.fromisoformat(str(as_date([cut])[0]))
    cols = [c for c in columns if c != "y"]
    if "lims_data_key" not in cols or "d" not in cols:
        raise ValueError("columns 에 lims_data_key·d 필요")
    df = pq.read_table(path, columns=cols).to_pandas()
    if df["lims_data_key"].duplicated().any():
        raise ValueError("시료 표 lims_data_key 중복")
    if "y" in columns:
        yk = pq.read_table(path, columns=["lims_data_key", "y"], filters=[("d", "<=", cut_date)]).to_pandas()
        df = df.merge(yk, on="lims_data_key", how="left", validate="one_to_one")
        df["y"] = df["y"].astype("Int8")
        known = day_ints(df["d"]) <= cut
        if df.loc[known, "y"].isna().any() or df.loc[~known, "y"].notna().any():
            raise AssertionError("y 읽기 범위가 d ≤ cut 과 다름")
        df = df[list(columns)]
    return df


# ---------------------------------------------------------------- 정의 그대로 다시 계산(점검용, 느림)
def _one_day(x) -> int:
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return int(x)
    return int(day_ints(pd.Series([x]))[0])


class Reference:
    """HIS1~HIS4 를 정의대로(행마다 표를 직접 거름) 다시 계산 — 단위 시험·실행 중 표본 대조용.
    Population·counts 와 다른 코드 경로(bincount·factorize 안 씀)."""

    def __init__(self, samples: pd.DataFrame):
        self.d = day_ints(samples["d"])
        self.y = samples["y"].to_numpy().astype(np.int64)
        self.vals = {"p3": p3_key(samples["p2"], samples["p3"]), "sgg": samples["sgg"].to_numpy(dtype=object),
                     "sido": samples["sido"].to_numpy(dtype=object), "p2": samples["p2"].to_numpy(dtype=object)}

    def row(self, row: Mapping, ks: Sequence[int] = K_VALUES) -> dict[str, float]:
        """row = d·C·p2·p3·sgg·sido."""
        d, y, V = self.d, self.y, self.vals
        C, rd = _one_day(row["C"]), _one_day(row["d"])
        if rd <= C:
            raise LeakageError("d ≤ C")
        r = {"p3": p3_key([row["p2"]], [row["p3"]])[0], "sgg": row["sgg"], "sido": row["sido"], "p2": row["p2"]}
        out: dict[str, float] = {}
        nan = float("nan")
        for L in HIS_LENGTHS:
            lo = C - PERIOD_DAYS[L] + 1
            w = (d >= lo) & (d <= C)
            n_all, s_all = int(w.sum()), int(y[w].sum())
            ok = C >= DATA_START and n_all > 0
            p0 = s_all / n_all if ok else nan
            scale = PERIOD_DAYS[L] / (C - DATA_START + 1) if lo < DATA_START else 1.0
            for lv in HIS_LEVELS:
                sel = w & (V["p3"] == r["p3"]) & (V["sgg"] == r["sgg"]) if lv == "u" else w & (V[lv] == r[lv])
                n, s = int(sel.sum()), int(y[sel].sum())
                for k in ks:
                    out[f"hist_rate{L}_{lv}_{k_tag(k)}"] = (s + k * p0) / (n + k) if ok else nan
                out[f"hist_logn{L}_{lv}"] = float(np.log(1 + n * scale)) if ok else nan
                if L == "1y":
                    m = s / n if n > 0 else p0
                    for k in ks:
                        out[f"hist_unc1y_{lv}_{k_tag(k)}"] = float(np.sqrt(m * (1 - m) / (n + k + 1))) if ok else nan
            if L == "5y":
                for lv in WOE_LEVELS:
                    sel = w & (V[lv] == r[lv])
                    n, s = int(sel.sum()), int(y[sel].sum())
                    out[f"woe_{lv}"] = (float(np.log((s + 0.5) / (s_all + 0.5))
                                              - np.log((n - s + 0.5) / (n_all - s_all + 0.5))) if ok else nan)
        return out
