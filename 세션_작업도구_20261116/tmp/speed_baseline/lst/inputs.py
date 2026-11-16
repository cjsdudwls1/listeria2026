"""입력 조립 — 계획서 §1-1, §4-1·§4-2(요약·원값·시퀀스·차감·결측·'해외 구조적 결측'), §5 표, §6 1차·단계 2 준비·
2차 공통, 부록 A-8(딥러닝 입력 x_seq·x_static·x_cat), 부록 A-12 '범주 처리', 부록 B-6.

학습 틀 인터페이스(공통 규칙 '추가(11-07 저녁)'):
- 행 표(meta): DataFrame, 행 = (시료, h) — lims_data_key·src_route·sido·sgg·p2·p3·d·h (T·E·S·g·R·C 가 있으면 대조).
- 트리 입력 X = `build_tree_X(meta, ctx, spec, split=…)`: 숫자 칼럼 float32(NaN 그대로), 범주 칼럼(범주 5·season·
  aw_label) = 원 문자열(aw_label null = '결측'). 알고리즘별 범주 변환(부록 A-12)은 트리 모델 객체가 함.
- 딥러닝 입력 = `build_dl_raw(...)` → `DLPreprocessor`(그 모델 train 으로 fit) → `transform` →
  dict(x_seq float32 (N, W, C), x_static float32 (N, S), x_cat int64 (N, 4)) — seq_only 판(⑥·⑦-D)은 x_cat 없음.
- 로지스틱·표 AE·VAE 숫자 칼럼 전처리 = `TabularScaler`(표준화 → 결측 0 → 결측 표시 + '해외 구조적 결측').
큰 입력 행렬은 파일로 만들지 않음(부록 B-1) — 이 모듈은 메모리에서만 만듦.

자료(`load_context`): 국내·해외 일자료 = `wxfeat.combined_cube`, SPI = 국내 관측 표 + 해외 표(관측 칸 cube) /
국내 예보 표 + 해외 표(예보 칸 forecast_cube — 이음 위치 T, §4-5·§7-1), 식품 고정표·축사 표(단계 0-6),
토양(SO1)·친환경(X1)·지하수(X2)는 `static_feats` 조회 함수, 이력은 `history`(C = T 직전 06-30, §5-1).

구현 세부(계획서에 없음 — 담당자 확인):
 (a) 표준화 = 평균·표준편차(ddof 0, NaN 뺌). 시퀀스 채널은 그 채널의 (행 × 날) 값 전부로 한 쌍. 원핫·0/1·g ÷ 50 은
     표준화 안 함. '전부 결측·분산 0' 칼럼·채널 = 0(부록 A-8).
 (b) 결측 표시 = 그 칸이 NaN 이면 1. '해외 구조적 결측' 대상 칼럼·채널은 해외 행이면 칼럼별 표시 0, 합친
     'ov_struct_na' = 해외 행 1·국내 행 0(그 판에 대상 칼럼·채널이 하나라도 있을 때만 그 칼럼이 있음 — 부록 A-5).
 (c) x_static 순서 = g_div50 → src_route01(standard, 국내만 버전은 없음) → ov_struct_na(국내만 버전은 없음) → 표 칼럼(단위 번호 순, 숫자는 표준화·
     원핫은 그대로) → 결측 표시 칼럼. x_seq 채널 순서 = 값 채널(기상 19 → 차감 19 → SPI, registry.ordered_units) → 결측 표시 채널.
 (d) x_cat 번호 = 범주 번호표(시료 표 전체 범주, 코드 문자열 정렬 순) 위치 + 1, 0 = OOV(그 모델 train 에 없던 범주).
     임베딩 크기 = 번호표 크기 + 1(OOV 포함), 차원 = min(50, (크기 + 1) // 2).
 (e) 트리 X 칼럼 순서 = 등록부 `registry.tree_columns` 순서(범주 5 → h·g → 기상 window → 차감 → SPI → 표 칼럼).
 (f) 이력 칼럼의 모집단 = 시료 표 중 d ≤ (split 의 train 끝, 없으면 y 읽기 끝) — C ≤ train 끝 이라 fold 마다 자른
     모집단과 같은 값. split 을 주면 모든 행 C ≤ train 끝 을 assert(부록 B-6).
 (g) 계산 중복 없애기(결과 불변 — 담당자 승인 11-08 "같은 결과면 계산을 공유해 절약"): 칼럼 묶음마다 그 묶음이 실제로
     기대는 키의 고유값으로 한 번만 계산하고 행으로 펼침(`_keyed`). 키 — 기상 window 요약·원값·차감·시퀀스 = (sgg, E)
     (W·N_ma 고정, S = E − W + 1, 예보 칸 = 같은 표), SPI = (sgg, E, T)(예보 표 이음 위치 T — 해외도 같은 키),
     WX3 = (sgg, E, p3)(aw·pH = p3 의 식품값), WX4 = (sgg, E, T)(soil_ph = SO1(sgg, T)), WX5·WX6 = (sgg, R),
     SO1·X1·X2 = (sgg, T), L1 = sgg, 식품 = p3, season = d, 이력 = (p2, p3, sgg, sido, C)(d > C 는 모든 행 assert).
     `PairMemo` = 같은 조립 객체의 transform 사이에 이어 쓰는 메모리 캐시(디스크 없음, B-1) — runner 가 평가 쌍 h 묶음
     동안만 켬(`pair_cache()`), train 행 입력은 그 호출 안에서만 중복 제거(학습 행 수·가중 그대로).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cf
from . import folds as fd
from . import history as hi
from . import registry as rg
from . import static_feats as sf
from . import windows as wn
from . import wxfeat as wf
from .paths import resolve

META_REQUIRED = ("lims_data_key", "src_route", "sido", "sgg", "p2", "p3", "d", "h")
SAMPLE_COLS = ("lims_data_key", "y", "src_route", "sgg", "sido", "p2", "p3", "d", "h")
OVERSEAS = "overseas"
DOMESTIC = "domestic"


class InputError(ValueError):
    pass


# ======================================================================== 자료 묶음
CONTEXT_SCHEMA = {
    "results": {                                         # --inputs(단계 0 결과 뿌리) 아래 상대 경로
        "samples": cf.Field(str),
        "domestic_weather": cf.Field(str, required=False),
        "overseas_weather": cf.Field(str, required=False),
        "spi_domestic_obs": cf.Field(str, required=False),
        "spi_domestic_fc": cf.Field(str, required=False),
        "spi_overseas": cf.Field(str, required=False),
        "food_table": cf.Field(str, required=False),
        "livestock": cf.Field(str, required=False),
        "missing_lists": cf.Field(str, required=False),
    },
    "data": {                                            # --root(자료 뿌리) 아래 상대 경로
        "soil": cf.Field(str, required=False),
        "global_soil_ph": cf.Field(str, required=False),
        "eco": cf.Field(str, required=False),
        "gw": cf.Field(str, required=False),
    },
    "y_cut": cf.Field(str),                              # y 를 읽는 끝(d ≤ 2025-06-30 — §1-3 test 막기)
}


@dataclass
class InputContext:
    samples: pd.DataFrame                    # SAMPLE_COLS, y = Int8(d > y_cut 은 <NA>)
    y_cut: int
    wx: wn.DailyCube | None = None           # 국내 + 해외 일자료(WEATHER19 + rh·td)
    spi_obs: wn.DailyCube | None = None
    spi_fc: wn.DailyCube | None = None
    food: pd.DataFrame | None = None
    livestock: pd.DataFrame | None = None
    soil: sf.SoilLookup | None = None
    eco: sf.YearIndicator | None = None
    gw: sf.YearIndicator | None = None
    missing: rg.MissingLists | None = None
    _anom: dict = field(default_factory=dict, repr=False)
    _pop: dict = field(default_factory=dict, repr=False)
    _vocab: dict | None = field(default=None, repr=False)

    # ---- 차감 일자료(N_ma 마다 한 번 계산해 둠)
    def anomaly(self, n_ma: int) -> wn.DailyCube:
        if self.wx is None:
            raise InputError("기상 일자료 없음")
        if n_ma not in rg.N_MA_VALUES:
            raise InputError(f"N_ma 는 {rg.N_MA_VALUES} 중: {n_ma}")
        if n_ma not in self._anom:
            self._anom[n_ma] = wn.anomaly_cube(self.wx.select(list(rg.WEATHER19)), n_ma)
        return self._anom[n_ma]

    def drop_anomaly_cache(self) -> None:
        self._anom.clear()

    # ---- 이력 모집단
    def population(self, train_end: int) -> hi.Population:
        train_end = int(train_end)
        if train_end > self.y_cut:
            raise InputError("모집단 끝이 y 읽기 끝보다 뒤(§1-3 test 막기)")
        if train_end not in self._pop:
            s = self.samples
            m = fd.day_ints(s["d"]) <= train_end
            self._pop[train_end] = hi.Population.from_samples(s.loc[m, ["lims_data_key", "y", "d", "p2", "p3",
                                                                        "sgg", "sido", "src_route"]])
        return self._pop[train_end]

    # ---- 범주 번호표(부록 A-8: 전체 범주로 고정)
    @property
    def vocab(self) -> dict[str, tuple]:
        if self._vocab is None:
            self._vocab = {c: tuple(sorted(self.samples[c].astype(str).unique().tolist())) for c in rg.CAT4_DL}
        return self._vocab


def _read(path, columns=None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pandas()


def spi_cubes(dom_obs: pd.DataFrame, dom_fc: pd.DataFrame, ov: pd.DataFrame | None) -> tuple[wn.DailyCube, wn.DailyCube]:
    """SPI 관측 칸 cube(국내 기상청 값 + 해외 우리 식)·예보 칸 cube(국내 우리 식 + 해외 우리 식 — 해외는 관측·예보 같은 표)."""
    var = list(rg.SPI_VARS)
    parts_o = [dom_obs.rename(columns={"sgg_code": "_u"})[["_u", "date"] + var]]
    parts_f = [dom_fc.rename(columns={"sgg_code": "_u"})[["_u", "date"] + var]]
    if ov is not None:
        if "alpha_3" in ov.columns and (ov["sgg"].astype(str) != wf.OVERSEAS_PREFIX + ov["alpha_3"].astype(str)).any():
            raise InputError("해외 SPI sgg 가 'OV' + alpha_3 와 다름")
        o = ov.rename(columns={"sgg": "_u"})[["_u", "date"] + var]
        dom_u = set(parts_o[0]["_u"].astype(str)) | set(parts_f[0]["_u"].astype(str))
        if dom_u & set(o["_u"].astype(str)) or any(u.startswith(wf.OVERSEAS_PREFIX) for u in dom_u):
            raise InputError("국내·해외 SPI 단위 코드 겹침")
        parts_o.append(o)
        parts_f.append(o)
    obs = wn.DailyCube.from_long(pd.concat(parts_o, ignore_index=True), "_u", "date", var)
    fc = wn.DailyCube.from_long(pd.concat(parts_f, ignore_index=True), "_u", "date", var)
    return obs, fc


def load_context(cfg: Mapping, data_root, inputs_root, *, components: Iterable[str] | None = None) -> InputContext:
    """CONTEXT_SCHEMA 설정 → InputContext. components 를 주면 그 이름만 읽음
    ('weather'·'spi'·'food'·'livestock'·'soil'·'eco'·'gw'·'missing'), 없으면 설정에 있는 것 전부."""
    cf.validate(cfg, CONTEXT_SCHEMA)
    R, D = cfg["results"], cfg["data"]
    want = None if components is None else set(components)

    def on(name, *keys, where=R):
        ok = all(k in where for k in keys)
        if want is not None and name in want and not ok:
            raise InputError(f"'{name}' 자료 경로가 설정에 없음: {keys}")
        return ok and (want is None or name in want)

    y_cut = fd.day(cfg["y_cut"])
    samples = hi.read_samples_blind_test(resolve(inputs_root, R["samples"]), list(SAMPLE_COLS), cut=y_cut)
    for c in ("src_route", "sgg", "sido", "p2", "p3"):
        if samples[c].isna().any():
            raise InputError(f"시료 표 {c} 결측")
    ctx = InputContext(samples=samples, y_cut=y_cut)
    if on("weather", "domestic_weather", "overseas_weather"):
        dom = _read(resolve(inputs_root, R["domestic_weather"]), ["sgg_code", "date"] + list(wf.CUBE_VARS))
        ov = _read(resolve(inputs_root, R["overseas_weather"]),
                   ["alpha_3", "sgg", "date"] + list(rg.WEATHER19) + list(wf.OVERSEAS_RENAME))
        ctx.wx = wf.combined_cube(dom, ov)
        del dom, ov
    if on("spi", "spi_domestic_obs", "spi_domestic_fc", "spi_overseas"):
        ctx.spi_obs, ctx.spi_fc = spi_cubes(_read(resolve(inputs_root, R["spi_domestic_obs"])),
                                            _read(resolve(inputs_root, R["spi_domestic_fc"])),
                                            _read(resolve(inputs_root, R["spi_overseas"])))
    if on("food", "food_table"):
        ctx.food = _read(resolve(inputs_root, R["food_table"]))
    if on("livestock", "livestock"):
        ctx.livestock = _read(resolve(inputs_root, R["livestock"]), ["unit", "L1"])
    if on("soil", "soil", "global_soil_ph", where=D):
        soil = _read(resolve(data_root, D["soil"]), ["sgg_code", "date"] + list(sf.SO1_COLUMNS))
        ctx.soil = sf.SoilLookup(soil, _read(resolve(data_root, D["global_soil_ph"]), ["iso3", "ph_mean"]))
    if on("eco", "eco", where=D):
        ctx.eco, info = sf.eco_indicator(_read(resolve(data_root, D["eco"])))
        if info["사용가능일_파일과_다른_행"]:
            raise InputError("X1 파일 사용가능일이 '자료연도 + 2년 1월 1일' 과 다름")
    if on("gw", "gw", where=D):
        ctx.gw, _ = sf.gw_indicator(_read(resolve(data_root, D["gw"])))
    if on("missing", "missing_lists"):
        ctx.missing = rg.MissingLists.from_json(resolve(inputs_root, R["missing_lists"]))
    return ctx


# ======================================================================== 행 표
@dataclass(frozen=True)
class Rows:
    n: int
    key: np.ndarray
    route: np.ndarray        # 'domestic'·'overseas'
    sgg: np.ndarray
    cats: dict               # 범주 5 문자열
    d: np.ndarray            # 일수
    h: np.ndarray
    b: wn.Bounds             # §1-1 경계(W)
    T: np.ndarray
    C: np.ndarray            # 이력 기준일(§5-1)

    @property
    def overseas(self) -> np.ndarray:
        return self.route == OVERSEAS


def _str_col(meta: pd.DataFrame, c: str) -> np.ndarray:
    a = meta[c].to_numpy(dtype=object)
    if a.shape[0] and pd.api.types.infer_dtype(a, skipna=False) == "string":   # 모두 str(결측 없음) — 빠른 길
        return a
    bad = pd.isna(meta[c]).to_numpy() | ~np.fromiter((isinstance(v, str) for v in a), bool, a.shape[0])
    if bad.any():
        raise InputError(f"meta {c} 에 결측·문자열 아닌 값 {int(bad.sum())}개")
    return a


def prepare_rows(meta: pd.DataFrame, W: int, split: fd.Split | None = None) -> Rows:
    """meta 점검 + 경계. T·E·S·g·R·C 칼럼이 있으면 계산값과 같은지 확인(다르면 멈춤)."""
    miss = [c for c in META_REQUIRED if c not in meta.columns]
    if miss:
        raise InputError(f"meta 칼럼 없음: {miss}")
    d = fd.day_ints(meta["d"])
    h = meta["h"].to_numpy()
    b = wn.window_bounds(d, h, W)
    for c, v in (("T", b.T), ("E", b.E), ("S", b.S), ("R", b.R)):
        if c in meta.columns and not np.array_equal(fd.day_ints(meta[c]), v):
            raise InputError(f"meta {c} 가 §1-1 정의(W = {W})와 다름")
    if "g" in meta.columns and not np.array_equal(meta["g"].to_numpy(dtype=np.int64), b.g):
        raise InputError("meta g 가 §1-1 정의와 다름")
    C = hi.c_from_T(b.T)
    if "C" in meta.columns and not np.array_equal(fd.day_ints(meta["C"]), C):
        raise InputError("meta C 가 'T 직전 06-30' 과 다름")
    if split is not None and (C > split.train_end).any():
        raise hi.LeakageError(f"{split.name}: C 가 train 끝을 넘는 행 {int((C > split.train_end).sum())}개")
    cats = {c: _str_col(meta, c) for c in rg.CAT5}
    route = cats["src_route"]
    if not np.isin(route, (DOMESTIC, OVERSEAS)).all():
        raise InputError("src_route 는 domestic·overseas 만")
    return Rows(len(meta), meta["lims_data_key"].to_numpy(dtype=object), route, cats["sgg"], cats, d,
                b.h, b, b.T, C)


def _take(rows: Rows, pos: np.ndarray) -> Rows:
    """행 표의 일부(대표 행) — 경계·범주 그대로 골라 냄."""
    b = rows.b
    nb = wn.Bounds(b.d[pos], b.h[pos], b.W, b.T[pos], b.E[pos], b.S[pos], b.R[pos], b.g[pos], b.n_obs[pos],
                   b.n_fc[pos])
    return Rows(int(pos.shape[0]), rows.key[pos], rows.route[pos], rows.sgg[pos],
                {c: v[pos] for c, v in rows.cats.items()}, rows.d[pos], rows.h[pos], nb, rows.T[pos], rows.C[pos])


# ======================================================================== 계산 중복 없애기(구현 세부 (g))
DEDUP = True    # 시험용 스위치(False = 예전처럼 모든 행으로 계산 — 같은 결과 대조용), 실제 실행은 늘 True


def _group(keys: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """키 칼럼들(행마다 튜플)이 같은 행 묶기 → (행 → 묶음 번호(첫 등장 순), 묶음마다 첫 행 위치)."""
    n = len(keys[0])
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    code = None
    for k in keys:
        c, u = pd.factorize(np.asarray(k))
        if (c < 0).any():
            raise InputError("중복 제거 키에 결측")
        c = c.astype(np.int64)
        code = c if code is None else pd.factorize(code * np.int64(len(u)) + c)[0].astype(np.int64)
    return code, first_positions(code)


def first_positions(code: np.ndarray) -> np.ndarray:
    """번호가 첫 등장 순(0, 1, 2, …)일 때 번호마다 첫 행 위치(번호 순)."""
    cm = np.maximum.accumulate(code)
    new = np.empty(code.shape[0], dtype=bool)
    new[:1] = True
    new[1:] = cm[1:] > cm[:-1]
    first = np.flatnonzero(new).astype(np.int64)
    assert first.shape[0] == int(cm[-1]) + 1 if code.shape[0] else True
    return first


class _Entry:
    """키 튜플 → 위치, 값 배열(칼럼 이름마다, 첫 축 = 키) — 용량을 두 배씩 늘림."""

    def __init__(self):
        self.pos: dict[tuple, int] = {}
        self.vals: dict[str, np.ndarray] = {}
        self.n = 0
        self.cols: list | None = None     # DataFrame 블록의 칼럼 이름(_keyed_frame)

    def add(self, tuples: list, vals: Mapping[str, np.ndarray]) -> None:
        m = len(tuples)
        if not self.vals:
            self.vals = {k: np.empty((max(m, 1024),) + v.shape[1:], dtype=v.dtype) for k, v in vals.items()}
        if set(vals) != set(self.vals):
            raise InputError("캐시 칼럼이 다름")
        cap = next(iter(self.vals.values())).shape[0]
        if self.n + m > cap:
            new = max(self.n + m, 2 * cap)
            for k, a in self.vals.items():
                b = np.empty((new,) + a.shape[1:], dtype=a.dtype)
                b[:self.n] = a[:self.n]
                self.vals[k] = b
        for k, v in vals.items():
            a = self.vals[k]
            if v.dtype != a.dtype or v.shape[1:] != a.shape[1:]:
                raise InputError(f"캐시 칼럼 {k} 꼴이 다름")
            a[self.n:self.n + m] = v
        for i, t in enumerate(tuples):
            self.pos[t] = self.n + i
        self.n += m

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.vals.values()))


class PairMemo:
    """같은 조립 객체의 transform 사이에 이어 쓰는 메모리 캐시(구현 세부 (g), 디스크 없음 — B-1).
    블록 이름(블록 종류·변수·W·mode·N_ma·dtype·모집단 끝 등 계산을 정하는 값 전부) → (키 튜플 → 값). 값은 대표 행으로
    계산한 그대로(같은 키 = 같은 값)."""

    def __init__(self):
        self.store: dict[tuple, _Entry] = {}
        self.cubes: dict[tuple, wn.DailyCube] = {}

    def clear(self) -> None:
        self.store.clear()
        self.cubes.clear()

    def nbytes(self) -> int:
        return sum(e.nbytes() for e in self.store.values())

    def n_keys(self) -> dict:
        return {k: e.n for k, e in self.store.items()}


def _keyed(memo: PairMemo | None, name: tuple, keys: Sequence[np.ndarray], n: int,
           compute) -> dict[str, np.ndarray]:
    """키가 같은 행은 한 번만 계산: compute(대표 행 위치 배열) → {칼럼: (대표 수, …) 배열}, 행으로 펼쳐 돌려줌.
    memo 가 있으면 앞 호출에서 계산한 키는 다시 계산하지 않음."""
    if n == 0 or not DEDUP:
        return compute(np.arange(n, dtype=np.int64))
    code, first = _group(keys)
    if memo is None:
        vals = compute(first)
        return {k: v[code] for k, v in vals.items()}
    ent = memo.store.get(name)
    if ent is None:
        ent = memo.store[name] = _Entry()
    tup = list(zip(*[np.asarray(k)[first].tolist() for k in keys]))
    loc = np.fromiter((ent.pos.get(t, -1) for t in tup), dtype=np.int64, count=len(tup))
    new = np.flatnonzero(loc < 0)
    if new.size:
        base = ent.n
        ent.add([tup[i] for i in new], compute(first[new]))
        loc[new] = base + np.arange(new.size)
    ri = loc[code]
    return {k: a[ri] for k, a in ent.vals.items()}


def _cube(memo: PairMemo | None, tag: str, get, variables: Sequence[str]) -> wn.DailyCube:
    """일자료 표에서 변수 고르기(DailyCube.select 는 표 전체 복사) — memo 가 있으면 한 번만."""
    if memo is None:
        return get().select(list(variables))
    k = (tag, tuple(variables))
    if k not in memo.cubes:
        memo.cubes[k] = get().select(list(variables))
    return memo.cubes[k]


def _keyed_frame(memo, name, keys, rows: Rows, fn, as_array: bool = False):
    """fn(대표 행 Rows) → DataFrame 인 블록을 키로 중복 제거(칼럼 이름·dtype 그대로). as_array 면 (칼럼 이름, (행, 칼럼)
    C 연속 배열) — 트리 X 에 넣을 때 DataFrame 을 거치며 생기는 전치 복사를 피함(같은 값)."""
    cols: list = []

    def compute(pos):
        df = fn(_take(rows, pos))
        cols[:] = list(df.columns)
        return {"_v": df.to_numpy(copy=False)}

    v = _keyed(memo, name, keys, rows.n, compute)["_v"]
    ent = memo.store.get(name) if memo is not None else None
    if ent is not None:
        if cols:
            ent.cols = list(cols)
        cols = list(ent.cols)          # 캐시만 썼을 때 — 처음 계산 때 이름
    if as_array:
        return cols, v
    return pd.DataFrame(v, columns=cols, copy=False)


# ======================================================================== 칼럼 계산(블록)
def _need(obj, what):
    if obj is None:
        raise InputError(f"자료 묶음에 {what} 없음(load_context 설정 확인)")
    return obj


def weather_window_block(rows: Rows, ctx: InputContext, variables: Sequence[str], W: int, mode: str,
                         dtype=np.float32, memo: PairMemo | None = None, as_array: bool = False):
    """1차·2차 기상 window(§4-2): 요약 또는 원값, 예보 칸 = 관측값(완전 예보 가정 — §0). 키 (sgg, E)."""
    get = lambda: _cube(memo, "wx", lambda: _need(ctx.wx, "기상 일자료"), variables)  # noqa: E731
    return _keyed_frame(memo, ("wx", tuple(variables), W, mode, np.dtype(dtype).str), [rows.sgg, rows.b.E], rows,
                        lambda r: wn.window_features(get(), r.sgg, r.d, r.h, W, mode, dtype=dtype), as_array)


def spi_window_block(rows: Rows, ctx: InputContext, variables: Sequence[str], W: int, mode: str,
                     dtype=np.float32, memo: PairMemo | None = None, as_array: bool = False):
    """WX1: 관측 칸(S ~ T−1) = 관측 표, 예보 칸(T ~ E) = 예보 표(§4-5, 국내 이음 위치 T). 키 (sgg, E, T)."""
    v = list(variables)
    obs = lambda: _cube(memo, "spi_obs", lambda: _need(ctx.spi_obs, "SPI"), v)  # noqa: E731
    fc = lambda: _cube(memo, "spi_fc", lambda: _need(ctx.spi_fc, "SPI 예보"), v)  # noqa: E731
    return _keyed_frame(memo, ("spi", tuple(v), W, mode, np.dtype(dtype).str), [rows.sgg, rows.b.E, rows.T], rows,
                        lambda r: wn.window_features(obs(), r.sgg, r.d, r.h, W, mode, forecast_cube=fc(), dtype=dtype),
                        as_array)


def anomaly_window_block(rows: Rows, ctx: InputContext, variables: Sequence[str], n_ma: int, W: int, mode: str,
                         dtype=np.float32, memo: PairMemo | None = None, as_array: bool = False):
    """WX2: 차감 일자료의 window(요약 판 3칼럼·원값 판 lag). 키 (sgg, E)."""
    names = [wn.anomaly_name(v, n_ma) for v in variables]
    get = lambda: _cube(memo, f"anom{n_ma}", lambda: ctx.anomaly(n_ma), names)  # noqa: E731
    return _keyed_frame(memo, ("anom", tuple(names), W, mode, np.dtype(dtype).str), [rows.sgg, rows.b.E], rows,
                        lambda r: wn.window_features(get(), r.sgg, r.d, r.h, W, mode, anomaly=True, dtype=dtype),
                        as_array)


def _seq_block(memo, name, keys, rows: Rows, fn) -> np.ndarray:
    """시퀀스 (행, W, 채널) 를 키로 중복 제거 — fn(대표 행 Rows) → 배열."""
    return _keyed(memo, name, keys, rows.n, lambda pos: {"_s": fn(_take(rows, pos))})["_s"]


def table_features(rows: Rows, ctx: InputContext, units: Sequence[rg.Unit], W: int, k: int | None,
                   split: fd.Split | None = None, memo: PairMemo | None = None) -> dict[str, np.ndarray]:
    """표 칼럼 단위(HIS·WX3~WX7·FD1·SO1·X1·X2·L1·rg 품목·rg 토양 pH) → 칼럼 이름 → 배열.
    숫자 = float64(NaN 그대로), season·aw_label = 문자열(aw_label null → '결측', 부록 A-4).
    칼럼 묶음마다 기대는 키의 고유값으로 한 번만 계산(구현 세부 (g))."""
    if any(u.kind != "table" for u in units):
        raise InputError("table_features 는 표 칼럼 단위만")
    ids = {u.id for u in units}
    if ids & {"HIS1", "HIS4"} and k is None:
        raise InputError("HIS1·HIS4 에는 k 필요")
    names = {c.name for u in units for c in rg.unit_tree_columns(u, W, "sum", k, None)}
    out: dict[str, np.ndarray] = {}
    stems = {u.name for u in units}
    n = rows.n
    sgg, p3, E, T = rows.sgg, rows.cats["p3"], rows.b.E, rows.T
    # wx_derived 그룹: WX3·WX5·WX6 단위, WX4 = 기상 12(rg 기온·지중온도) 또는 토양 pH 3 단위가 있을 때(품목 4 는 식품표)
    groups = sorted({u.id for u in units if u.id in ("WX3", "WX5", "WX6")}
                    | ({"WX4"} if stems & set(rg.RG_TEMP + rg.RG_SOILT + rg.RG_SOILPH) else set()))
    need_food = bool(ids & {"FD1"}) or bool(stems & set(rg.RG_ITEM)) or "WX3" in groups   # WX3 은 aw·pH 필요
    need_soil = "SO1" in ids or "WX4" in groups                                            # WX4 는 soil_ph 필요
    if need_food:
        def _food(pos):
            food = sf.food_features(_need(ctx.food, "식품 고정표"), p3[pos])
            o = {c: food[c].to_numpy(dtype=np.float64) for c in rg.FD1_NUM + rg.RG_ITEM}
            lab = food["aw_label"].to_numpy(dtype=object)
            o["aw_label"] = np.where(pd.isna(lab), rg.AW_MISSING, lab).astype(object)
            return o
        out.update(_keyed(memo, ("food",), [p3], n, _food))
        bad = set(out["aw_label"]) - set(rg.AW_LABEL_VALUES)
        if bad:
            raise InputError(f"aw_label 에 모르는 값: {sorted(bad)}")
    if need_soil:
        def _soil(pos):
            so = _need(ctx.soil, "토양 조회")(sgg[pos], T[pos])
            return {c: so[c].to_numpy(dtype=np.float64) for c in rg.SO1}
        out.update(_keyed(memo, ("soil",), [sgg, T], n, _soil))
    if groups:
        cube = _need(ctx.wx, "기상 일자료")

        def _wxd(gs, aw=None, ph=None, sph=None):
            def f(pos):
                r = _take(rows, pos)
                df = wf.wx_derived(cube, r.sgg, r.d, r.h, W, groups=gs,
                                   aw=None if aw is None else aw[pos], ph=None if ph is None else ph[pos],
                                   soil_ph=None if sph is None else sph[pos])
                return {c: df[c].to_numpy(dtype=np.float64) for c in df.columns}
            return f
        # 그룹마다 따로(같은 함수 — 칼럼은 이름으로 넣음): WX3 키 (sgg, E, p3), WX4 (sgg, E, T), WX5·WX6 (sgg, R)
        if "WX3" in groups:
            out.update(_keyed(memo, ("WX3", W), [sgg, E, p3], n, _wxd(["WX3"], out["food_aw"], out["food_ph"])))
        if "WX4" in groups:
            out.update(_keyed(memo, ("WX4", W), [sgg, E, T], n, _wxd(["WX4"], sph=out["soil_ph"])))
        g56 = [g for g in groups if g in ("WX5", "WX6")]
        if g56:
            out.update(_keyed(memo, ("WX56", tuple(g56)), [sgg, rows.b.R], n, _wxd(g56)))
    if "WX7" in ids:
        out.update(_keyed(memo, ("season",), [rows.d], n, lambda pos: {"season": sf.season(rows.d[pos])}))
    if "X1" in ids:
        out.update(_keyed(memo, ("X1",), [sgg, T], n,
                          lambda pos: {"eco_ratio": _need(ctx.eco, "친환경 지표")(sgg[pos], T[pos])}))
    if "X2" in ids:
        out.update(_keyed(memo, ("X2",), [sgg, T], n,
                          lambda pos: {"gw_pass_rate": _need(ctx.gw, "지하수 지표")(sgg[pos], T[pos])}))
    if "L1" in ids:
        out.update(_keyed(memo, ("L1",), [sgg], n,
                          lambda pos: {"L1": sf.livestock_lookup(_need(ctx.livestock, "축사 표"), sgg[pos])}))
    if ids & {"HIS1", "HIS2", "HIS3", "HIS4"}:
        if (rows.d <= rows.C).any():   # history.prepare_rows 와 같은 점검을 모든 행에(대표 행만 넘기므로)
            raise hi.LeakageError(f"d ≤ C 인 행 {int((rows.d <= rows.C).sum())}개 — 자기 시료가 집계에 들어갈 수 있음")
        train_end = split.train_end if split is not None else ctx.y_cut
        pop = ctx.population(train_end)
        kk = k if k is not None else hi.K_VALUES[0]
        his_keys = [rows.cats["p2"], rows.cats["p3"], sgg, rows.cats["sido"], rows.C]

        def _his(pos):
            hr = pd.DataFrame({"d": rows.d[pos], "C": rows.C[pos],
                               **{c: rows.cats[c][pos] for c in ("p2", "p3", "sgg", "sido")}})
            his = hi.history_features(pop, hr, ks=[kk])
            return {c: his[c].to_numpy(dtype=np.float64) for c in his.columns}
        out.update(_keyed(memo, ("his", int(train_end), int(kk)), his_keys, n, _his))
    miss = names - set(out)
    if miss:
        raise InputError(f"표 칼럼 계산 빠짐: {sorted(miss)[:5]}")
    return {c: out[c] for c in names}


# ======================================================================== 트리 입력
def build_tree_X(meta: pd.DataFrame, ctx: InputContext, spec: rg.VersionSpec, *, split: fd.Split | None = None,
                 dtype=np.float32, memo: PairMemo | None = None) -> pd.DataFrame:
    """트리 입력 X(그 판 칼럼 전부, 등록부 순서). 숫자 float32·NaN 그대로, 범주 = 원 문자열.
    split 을 주면 이력 모집단 = d ≤ split train 끝·C ≤ train 끝 assert(부록 B-6).
    memo = 앞 호출과 계산을 나눠 쓰는 메모리 캐시(구현 세부 (g), 결과 같음) — 없으면 이 호출 안에서만 중복 제거."""
    rows = prepare_rows(meta, spec.W, split)
    cols = rg.tree_columns(spec)
    units = spec.units()
    num_cols = [c for c in cols if c.kind in ("numeric", "time")]
    pos = {c.name: i for i, c in enumerate(num_cols)}
    X = np.full((rows.n, len(num_cols)), np.nan, dtype=dtype)
    filled = np.zeros(len(num_cols), dtype=bool)

    def put(name, arr):
        i = pos[name]
        X[:, i] = arr
        filled[i] = True

    def put_df(block):
        cols, arr = block                                            # (칼럼 이름, (행, 칼럼) 배열)
        idx = [pos[c] for c in cols]
        arr = np.asarray(arr).astype(dtype, copy=False)
        # 칼럼 순서가 이어진 토막마다 slice 로 넣음(같은 값 — 칼럼 번호 목록 대입보다 빠름)
        ia = np.asarray(idx, dtype=np.int64)
        cut = np.flatnonzero(np.diff(ia) != 1) + 1
        for lo, hi in zip(np.r_[0, cut], np.r_[cut, ia.size]):
            X[:, ia[lo]:ia[lo] + (hi - lo)] = arr[:, lo:hi]
        filled[idx] = True

    if "h" in pos:
        put("h", rows.h)
    put("g", rows.b.g)
    wv = [s for u in units if u.kind == "window" for s in u.sources]
    if wv:
        put_df(weather_window_block(rows, ctx, wv, spec.W, spec.mode, dtype, memo=memo, as_array=True))
    sv = [s for u in units if u.kind == "spi" for s in u.sources]
    if sv:
        put_df(spi_window_block(rows, ctx, sv, spec.W, spec.mode, dtype, memo=memo, as_array=True))
    av = [s for u in units if u.kind == "anom" for s in u.sources]
    if av:
        put_df(anomaly_window_block(rows, ctx, av, spec.n_ma, spec.W, spec.mode, dtype, memo=memo, as_array=True))
    tu = [u for u in units if u.kind == "table"]
    cat_extra = {}
    if tu:
        tab = table_features(rows, ctx, tu, spec.W, spec.k, split, memo=memo)
        for name, arr in tab.items():
            if name in pos:
                put(name, arr)
            else:
                cat_extra[name] = arr
    if not filled.all():
        raise InputError(f"채우지 못한 칼럼: {[num_cols[i].name for i in np.flatnonzero(~filled)][:5]}")
    df = pd.DataFrame(X, columns=[c.name for c in num_cols], copy=False)
    # 범주 칼럼을 등록부 순서 자리에 끼움(숫자 블록은 복사 안 함)
    for i, c in enumerate(cols):
        if c.kind == "category":
            v = rows.cats[c.name] if c.name in rows.cats else cat_extra[c.name]
            df.insert(i, c.name, pd.Series(v, dtype=object, index=df.index))
    if list(df.columns) != [c.name for c in cols]:
        raise InputError("트리 X 칼럼 순서가 등록부와 다름")
    return df


# ======================================================================== 딥러닝 입력
@dataclass
class DLRaw:
    """표준화 전 딥러닝 입력(행 = (시료, h) 쌍). seq NaN 그대로, static = 단위 칼럼(숫자 NaN·원핫 0/1)."""
    seq: np.ndarray                  # float32 (N, W, C)
    seq_names: list
    static: np.ndarray               # float32 (N, S_unit)
    static_names: list
    g: np.ndarray                    # (N,) 일수
    overseas: np.ndarray             # (N,) bool
    cat: dict | None                 # 'sido'·'sgg'·'p2'·'p3' → 문자열 배열(seq_only 판 None)
    keys: np.ndarray | None = None
    h: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.seq.shape[0]


def build_dl_raw(meta: pd.DataFrame, ctx: InputContext, spec: rg.VersionSpec, *,
                 split: fd.Split | None = None, memo: PairMemo | None = None) -> DLRaw:
    """부록 A-8 입력(표준화 전): x_seq = 기상·차감(n_ma_dl)·SPI 채널(그 판 단위에 든 것), x_static 단위 칼럼 =
    표 칼럼(window 요약 칼럼은 W_DL = W 로 계산)·계절 원핫·aw_label 원핫, x_cat = sido·sgg·p2·p3 문자열."""
    rows = prepare_rows(meta, spec.W, split)
    seq_cols = rg.dl_seq_channels(spec)
    units = spec.units()
    parts = {}
    W = spec.W
    kE = [rows.sgg, rows.b.E]                    # 기상·차감 시퀀스 키 (sgg, E), SPI (sgg, E, T) — 구현 세부 (g)
    wv = [s for u in units if u.kind == "window" for s in u.sources]
    if wv:
        x = _seq_block(memo, ("seq_wx", tuple(wv), W), kE, rows,
                       lambda r: wn.sequences(_cube(memo, "wx", lambda: _need(ctx.wx, "기상 일자료"), wv),
                                              r.sgg, r.d, r.h, W, dtype=np.float32))
        parts.update({v: x[:, :, i] for i, v in enumerate(wv)})
    av = [s for u in units if u.kind == "anom" for s in u.sources]
    if av:
        names = [wn.anomaly_name(v, spec.n_ma_dl) for v in av]
        x = _seq_block(memo, ("seq_anom", tuple(names), W), kE, rows,
                       lambda r: wn.sequences(_cube(memo, f"anom{spec.n_ma_dl}", lambda: ctx.anomaly(spec.n_ma_dl),
                                                    names), r.sgg, r.d, r.h, W, dtype=np.float32))
        parts.update({v: x[:, :, i] for i, v in enumerate(names)})
    sv = [s for u in units if u.kind == "spi" for s in u.sources]
    if sv:
        x = _seq_block(memo, ("seq_spi", tuple(sv), W), kE + [rows.T], rows,
                       lambda r: wn.sequences(_cube(memo, "spi_obs", lambda: _need(ctx.spi_obs, "SPI"), sv),
                                              r.sgg, r.d, r.h, W,
                                              forecast_cube=_cube(memo, "spi_fc", lambda: _need(ctx.spi_fc, "SPI 예보"), sv),
                                              dtype=np.float32))
        parts.update({v: x[:, :, i] for i, v in enumerate(sv)})
    seq = (np.stack([parts[c.name] for c in seq_cols], axis=2).astype(np.float32, copy=False) if seq_cols
           else np.zeros((rows.n, spec.W, 0), np.float32))
    st_cols = rg.dl_static_columns(spec)
    static = np.zeros((rows.n, len(st_cols)), dtype=np.float32)
    if st_cols:
        tu = [u for u in units if u.kind == "table"]
        tab = table_features(rows, ctx, tu, spec.W, spec.k, split, memo=memo)
        for j, c in enumerate(st_cols):
            if c.kind == "onehot":                       # 'season_<값>'·'aw_label_<값>'
                val = c.name[len(c.source) + 1:]
                static[:, j] = (tab[c.source] == val).astype(np.float32)
            else:
                static[:, j] = tab[c.name]
    cat = None if spec.dl_layout == "seq_only" else {c: rows.cats[c] for c in rg.CAT4_DL}
    return DLRaw(seq, [c.name for c in seq_cols], static, [c.name for c in st_cols], rows.b.g.astype(np.float32),
                 rows.overseas, cat, rows.key, rows.h)


class _Moments:
    """칼럼별 (개수, 평균, M2) — 묶음마다 합침(Chan), NaN 뺌."""

    def __init__(self, k: int):
        self.n = np.zeros(k)
        self.mean = np.zeros(k)
        self.m2 = np.zeros(k)

    def update(self, x2d: np.ndarray) -> None:
        x = np.asarray(x2d, dtype=np.float64)
        ok = ~np.isnan(x)
        nb = ok.sum(axis=0).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mb = np.where(nb > 0, np.where(ok, x, 0.0).sum(axis=0) / np.maximum(nb, 1), 0.0)
        dev = np.where(ok, x - mb[None, :], 0.0)
        m2b = (dev ** 2).sum(axis=0)
        n = self.n + nb
        delta = mb - self.mean
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(n > 0, nb / np.maximum(n, 1), 0.0)
            self.mean = self.mean + delta * frac
            self.m2 = self.m2 + m2b + np.where(n > 0, delta ** 2 * self.n * nb / np.maximum(n, 1), 0.0)
        self.n = n

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(평균, 표준편차, 0 으로 둘 칼럼) — 전부 결측·분산 0 = 0(부록 A-8)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            std = np.sqrt(np.where(self.n > 0, self.m2 / np.maximum(self.n, 1), np.nan))
        zero = (self.n == 0) | ~(std > 0)
        return self.mean.copy(), std, zero


def _standardize(x: np.ndarray, mean, std, zero) -> np.ndarray:
    """(x − 평균) ÷ 표준편차 → 결측 0, 0 으로 둘 칼럼 = 0. 마지막 축 = 칼럼."""
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - mean) / np.where(zero, 1.0, std)
    z = np.where(np.isnan(z) | zero, 0.0, z)
    return z.astype(np.float32)


class DLPreprocessor:
    """딥러닝 전처리기(§4-2·부록 A-8): 그 모델 train 으로 partial_fit(묶음 여러 번 가능) → finalize → transform.
    표준화 통계는 모델마다(이 객체를 모델과 함께 보관). 결측 표시 채널·칼럼 = 결측 표시 목록(그 판 차수)."""

    def __init__(self, spec: rg.VersionSpec, missing: rg.MissingLists, vocab: Mapping[str, Sequence[str]] | None):
        self.spec = spec
        self.seq_cols = rg.dl_seq_channels(spec)
        self.st_cols = rg.dl_static_columns(spec)
        self.fixed = rg.dl_fixed_static(spec)
        self.route01 = any(c.name == "src_route01" for c in self.fixed)   # 국내만 버전은 없음(registry 구현 세부 (h))
        self.structural = rg.has_structural(self.seq_cols + self.st_cols)
        o = spec.order
        self.seq_ind = np.array([missing.needs_indicator(c, o) for c in self.seq_cols], dtype=bool)
        self.seq_struct = np.array([c.structural for c in self.seq_cols], dtype=bool)
        self.st_num = np.array([c.kind == "numeric" for c in self.st_cols], dtype=bool)
        self.st_ind = np.array([missing.needs_indicator(c, o) for c in self.st_cols], dtype=bool)
        self.st_struct = np.array([c.structural for c in self.st_cols], dtype=bool)
        self.vocab = None if spec.dl_layout == "seq_only" else {c: tuple(vocab[c]) for c in rg.CAT4_DL}
        self._vidx = None if self.vocab is None else {c: {v: i + 1 for i, v in enumerate(self.vocab[c])}
                                                      for c in rg.CAT4_DL}
        self._seq_m = _Moments(len(self.seq_cols))
        self._st_m = _Moments(int(self.st_num.sum()))
        self.seen = None if self.vocab is None else {c: set() for c in rg.CAT4_DL}
        self.fitted = False

    # ---- 이름
    @property
    def seq_channel_names(self) -> list[str]:
        return ([c.name for c in self.seq_cols]
                + [rg.indicator_name(c.name) for c, m in zip(self.seq_cols, self.seq_ind) if m])

    @property
    def static_names(self) -> list[str]:
        return ([c.name for c in self.fixed] + ([rg.STRUCT_COLUMN] if self.structural else [])
                + [c.name for c in self.st_cols]
                + [rg.indicator_name(c.name) for c, m in zip(self.st_cols, self.st_ind) if m])

    @property
    def cat_cardinalities(self) -> list[int] | None:
        """범주 칼럼마다 임베딩 크기(번호표 + OOV)."""
        return None if self.vocab is None else [len(self.vocab[c]) + 1 for c in rg.CAT4_DL]

    @property
    def embed_dims(self) -> list[int] | None:
        """부록 A-8: min(50, (범주 수 + 1) // 2), 범주 수 = 번호표 크기(OOV 포함)."""
        cc = self.cat_cardinalities
        return None if cc is None else [min(50, (k + 1) // 2) for k in cc]

    def _check(self, raw: DLRaw) -> None:
        if raw.seq_names != [c.name for c in self.seq_cols] or raw.static_names != [c.name for c in self.st_cols]:
            raise InputError("DLRaw 칼럼이 이 판 등록부와 다름")
        if raw.seq.shape[1] != self.spec.W:
            raise InputError("시퀀스 길이 ≠ W")
        if (raw.cat is None) != (self.vocab is None):
            raise InputError("x_cat 유무가 판 layout 과 다름")

    # ---- fit
    def partial_fit(self, raw: DLRaw) -> "DLPreprocessor":
        if self.fitted:
            raise InputError("이미 finalize 됨")
        self._check(raw)
        n, W, C = raw.seq.shape
        if C:
            self._seq_m.update(raw.seq.reshape(n * W, C))
        if self.st_num.any():
            self._st_m.update(raw.static[:, self.st_num])
        if self.seen is not None:
            for c in rg.CAT4_DL:
                self.seen[c].update(np.unique(raw.cat[c]).tolist())
        return self

    def finalize(self) -> "DLPreprocessor":
        self.seq_mean, self.seq_std, self.seq_zero = self._seq_m.finalize()
        self.st_mean, self.st_std, self.st_zero = self._st_m.finalize()
        self.fitted = True
        return self

    def fit(self, raw: DLRaw) -> "DLPreprocessor":
        return self.partial_fit(raw).finalize()

    # ---- transform
    def _cat_codes(self, raw: DLRaw) -> np.ndarray:
        out = np.zeros((raw.n, len(rg.CAT4_DL)), dtype=np.int64)
        for j, c in enumerate(rg.CAT4_DL):
            idx = self._vidx[c]
            v = raw.cat[c]
            codes = np.fromiter((idx.get(s, -1) for s in v), dtype=np.int64, count=v.shape[0])
            if (codes < 0).any():
                raise InputError(f"{c}: 범주 번호표에 없는 값 {sorted(set(v[codes < 0]))[:5]}")
            seen = np.fromiter((s in self.seen[c] for s in v), dtype=bool, count=v.shape[0])
            out[:, j] = np.where(seen, codes, 0)            # train 에 없던 범주 → OOV 0
        return out

    def transform(self, raw: DLRaw) -> dict:
        if not self.fitted:
            raise InputError("finalize 전")
        self._check(raw)
        ov = raw.overseas
        z = _standardize(raw.seq, self.seq_mean, self.seq_std, self.seq_zero)
        if self.seq_ind.any():
            ind = np.isnan(raw.seq[:, :, self.seq_ind])
            st = self.seq_struct[self.seq_ind]
            ind[:, :, st] &= ~ov[:, None, None]
            z = np.concatenate([z, ind.astype(np.float32)], axis=2)
        fixed = [raw.g.astype(np.float32)[:, None] / np.float32(50.0)]
        if self.route01:
            fixed.append(ov.astype(np.float32)[:, None])       # src_route 0/1(해외 = 1)
        if self.structural:
            fixed.append(ov.astype(np.float32)[:, None])
        stat = raw.static.astype(np.float32, copy=True)
        if self.st_num.any():
            stat[:, self.st_num] = _standardize(raw.static[:, self.st_num], self.st_mean, self.st_std, self.st_zero)
        parts = fixed + [stat]
        if self.st_ind.any():
            ind = np.isnan(raw.static[:, self.st_ind])
            st = self.st_struct[self.st_ind]
            ind[:, st] &= ~ov[:, None]
            parts.append(ind.astype(np.float32))
        x_static = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        out = {"x_seq": np.ascontiguousarray(z, dtype=np.float32), "x_static": x_static}
        if self.vocab is not None:
            out["x_cat"] = self._cat_codes(raw)
        assert out["x_static"].shape[1] == len(self.static_names)
        assert out["x_seq"].shape[2] == len(self.seq_channel_names)
        return out

    # ---- 보관
    def state(self) -> dict:
        if not self.fitted:
            raise InputError("finalize 전")
        f = lambda a: [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=np.float64)]  # noqa: E731
        return {"spec": self.spec.to_dict(), "seq_names": self.seq_channel_names, "static_names": self.static_names,
                "seq_mean": f(self.seq_mean), "seq_std": f(self.seq_std), "seq_zero": self.seq_zero.tolist(),
                "st_mean": f(self.st_mean), "st_std": f(self.st_std), "st_zero": self.st_zero.tolist(),
                "seq_ind": self.seq_ind.tolist(), "st_ind": self.st_ind.tolist(), "structural": self.structural,
                "vocab": None if self.vocab is None else {c: list(v) for c, v in self.vocab.items()},
                "seen": None if self.seen is None else {c: sorted(v) for c, v in self.seen.items()}}

    def save(self, path) -> None:
        cf.dump_json(path, self.state())

    @classmethod
    def from_state(cls, st: Mapping) -> "DLPreprocessor":
        spec = rg.load_version(st["spec"])

        class _ML:  # 저장된 결측 표시 여부를 그대로 씀
            def __init__(self, seq_ind, st_ind, seq_cols, st_cols):
                self.m = {(c.role, c.name): bool(v) for c, v in zip(seq_cols, seq_ind)}
                self.m.update({(c.role, c.name): bool(v) for c, v in zip(st_cols, st_ind)})

            def needs_indicator(self, c, order):
                return self.m.get((c.role, c.name), False)

        ml = _ML(st["seq_ind"], st["st_ind"], rg.dl_seq_channels(spec), rg.dl_static_columns(spec))
        obj = cls(spec, ml, st["vocab"])
        a = lambda v: np.array([np.nan if x is None else x for x in v], dtype=np.float64)  # noqa: E731
        obj.seq_mean, obj.seq_std, obj.seq_zero = a(st["seq_mean"]), a(st["seq_std"]), np.array(st["seq_zero"], bool)
        obj.st_mean, obj.st_std, obj.st_zero = a(st["st_mean"]), a(st["st_std"]), np.array(st["st_zero"], bool)
        obj.seen = None if st["seen"] is None else {c: set(v) for c, v in st["seen"].items()}
        obj.fitted = True
        if obj.seq_channel_names != st["seq_names"] or obj.static_names != st["static_names"]:
            raise InputError("저장된 전처리기 칼럼이 등록부와 다름")
        return obj

    @classmethod
    def load(cls, path) -> "DLPreprocessor":
        return cls.from_state(cf.load_json_strict(path))


def fit_dl_preprocessor(meta_train: pd.DataFrame, ctx: InputContext, spec: rg.VersionSpec, *,
                        missing: rg.MissingLists | None = None, split: fd.Split | None = None,
                        chunk_rows: int = 50_000) -> DLPreprocessor:
    """그 모델 train 행(meta)으로 전처리기 fit(묶음마다 build_dl_raw → partial_fit, 큰 배열을 한꺼번에 안 만듦).
    구현 세부(계획서에 없음): chunk_rows 기본 50,000(결과 불변)."""
    prep = DLPreprocessor(spec, missing if missing is not None else _need(ctx.missing, "결측 표시 목록"),
                          None if spec.dl_layout == "seq_only" else ctx.vocab)
    for lo in range(0, len(meta_train), chunk_rows):
        prep.partial_fit(build_dl_raw(meta_train.iloc[lo:lo + chunk_rows], ctx, spec, split=split))
    return prep.finalize()


def iter_dl_inputs(meta: pd.DataFrame, ctx: InputContext, prep: DLPreprocessor, *, split: fd.Split | None = None,
                   chunk_rows: int = 50_000, memo: PairMemo | None = None):
    """(행 위치 slice, dict(x_seq·x_static·x_cat)) 를 묶음마다 — 예측·학습 배치용(디스크 캐시 없음, 부록 B-1)."""
    for lo in range(0, len(meta), chunk_rows):
        sl = slice(lo, min(len(meta), lo + chunk_rows))
        yield sl, prep.transform(build_dl_raw(meta.iloc[sl], ctx, prep.spec, split=split, memo=memo))


# ======================================================================== 로지스틱·표 AE·VAE 전처리
class TabularScaler:
    """§4-2 로지스틱·딥러닝 규칙을 표(트리 X 꼴)에: 숫자 칼럼 = 그 모델 train 평균·표준편차(NaN 뺌, ddof 0)로 표준화
    → 결측 0 → 결측 표시 칼럼('<칼럼>__na', 원 변수가 목록에 있을 때) + 'ov_struct_na'(구조적 결측 칼럼이 있을 때).
    범주 칼럼은 그대로 둠(원핫은 모델 쪽 — 부록 A-12). extra_numeric = 등록부 밖 숫자 칼럼(이상 점수 등, 표시 없음).
    h·g 를 빼는 것(표 AE·VAE)은 부르는 쪽이 X 에서 뺌.
    출력 칼럼 순서 = 숫자(입력 순, float32) → 범주(입력 순, 원 문자열) → 결측 표시 → ov_struct_na."""

    def __init__(self, columns: Sequence[rg.Column], missing: rg.MissingLists, order: int,
                 extra_numeric: Sequence[str] = ()):
        self.info = {c.name: c for c in columns}
        self.extra = tuple(extra_numeric)
        self.missing = missing
        self.order = order
        self.fitted = False

    def _plan(self, X: pd.DataFrame):
        unknown = [c for c in X.columns if c not in self.info and c not in self.extra]
        if unknown:
            raise InputError(f"등록부에 없는 칼럼(부록 B-2): {unknown[:5]}")
        num = [c for c in X.columns if c in self.extra or self.info[c].kind in ("numeric", "time")]
        cat = [c for c in X.columns if c not in num]
        ind = [c for c in num if c in self.info and self.missing.needs_indicator(self.info[c], self.order)]
        struct = any(c in self.info and self.info[c].structural for c in num)
        return num, cat, ind, struct

    def fit(self, X: pd.DataFrame) -> "TabularScaler":
        self.num, self.cat, self.ind, self.struct = self._plan(X)
        m = _Moments(len(self.num))
        m.update(X[self.num].to_numpy(dtype=np.float64))
        self.mean, self.std, self.zero = m.finalize()
        self.columns_in = list(X.columns)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame, src_route=None) -> pd.DataFrame:
        if not self.fitted:
            raise InputError("fit 전")
        if list(X.columns) != self.columns_in:
            raise InputError("칼럼이 fit 때와 다름")
        raw = X[self.num].to_numpy(dtype=np.float64)
        z = _standardize(raw, self.mean, self.std, self.zero)
        out = pd.DataFrame(z, columns=self.num, index=X.index)
        for c in self.cat:
            out[c] = X[c]
        need_ov = self.struct
        ov = None
        if need_ov:
            r = X["src_route"].to_numpy(dtype=object) if "src_route" in X.columns else src_route
            if r is None:
                raise InputError("구조적 결측 처리에 src_route 필요")
            ov = np.asarray(r, dtype=object) == OVERSEAS
        if self.ind:
            pos = {c: i for i, c in enumerate(self.num)}
            ind = np.isnan(raw[:, [pos[c] for c in self.ind]])
            if ov is not None:
                st = np.array([self.info[c].structural for c in self.ind], dtype=bool)
                ind[:, st] &= ~ov[:, None]
            out = pd.concat([out, pd.DataFrame(ind.astype(np.float32), index=X.index,
                                               columns=[rg.indicator_name(c) for c in self.ind])], axis=1)
        if self.struct:
            out[rg.STRUCT_COLUMN] = ov.astype(np.float32)
        return out


    # ---- 보관
    def state(self) -> dict:
        if not self.fitted:
            raise InputError("fit 전")
        f = lambda a: [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=np.float64)]  # noqa: E731
        return {"columns_in": self.columns_in, "num": self.num, "cat": self.cat, "ind": self.ind,
                "struct": self.struct, "mean": f(self.mean), "std": f(self.std), "zero": self.zero.tolist()}


# ======================================================================== runner 주입용 조립 객체
def _split(split) -> fd.Split | None:
    if split is None or isinstance(split, fd.Split):
        return split
    return fd.get_split(split)


class TreeInputs:
    """runner InputAssembler(트리): fit(train 행 표) → transform(행 표) = 트리 입력 X(DataFrame).
    scale=True(로지스틱 — §4-2) 면 TabularScaler 를 train X 로 fit 해 표준화·결측 0·결측 표시를 붙임
    (범주 원핫은 모델 쪽)."""

    def __init__(self, ctx: InputContext, spec: rg.VersionSpec, split=None, *, scale: bool = False,
                 missing: rg.MissingLists | None = None):
        self.ctx, self.spec, self.split, self.scale = ctx, spec, _split(split), scale
        self.cols = rg.tree_columns(spec)
        self.scaler = None
        self._missing = missing
        self._fit_meta = None
        self._fit_X = None
        self._memo: PairMemo | None = None

    @contextmanager
    def pair_cache(self):
        """평가 쌍 h 묶음 동안 블록 계산을 키 고유값으로 이어 씀(구현 세부 (g), 메모리만 — 끝나면 비움). runner 가 씀."""
        if self._memo is not None:
            raise InputError("pair_cache 겹침")
        self._memo = PairMemo()
        try:
            yield self._memo
        finally:
            self._memo = None

    def fit(self, train_meta: pd.DataFrame) -> "TreeInputs":
        if self.scale:
            X = build_tree_X(train_meta, self.ctx, self.spec, split=self.split)
            ml = self._missing if self._missing is not None else _need(self.ctx.missing, "결측 표시 목록")
            self.scaler = TabularScaler(self.cols, ml, self.spec.order).fit(X)
            self._fit_meta, self._fit_X = train_meta, X           # 바로 이어지는 transform(train 행)에 다시 씀
        return self

    def transform(self, meta: pd.DataFrame) -> pd.DataFrame:
        if self._fit_meta is not None and meta is self._fit_meta:
            X = self._fit_X
            self._fit_meta = self._fit_X = None
        else:
            X = build_tree_X(meta, self.ctx, self.spec, split=self.split, memo=self._memo)
        return self.scaler.transform(X) if self.scaler is not None else X

    @property
    def columns(self) -> list[str]:
        if self.scaler is not None:
            s = self.scaler
            return (s.num + s.cat + [rg.indicator_name(c) for c in s.ind] + ([rg.STRUCT_COLUMN] if s.struct else []))
        return [c.name for c in self.cols]

    def save(self, path) -> None:
        p = os.fspath(path)
        os.makedirs(p, exist_ok=False)
        cf.dump_json(os.path.join(p, "tree_inputs.json"),
                     {"spec": self.spec.to_dict(), "scale": self.scale,
                      "scaler": None if self.scaler is None else self.scaler.state()})


class DLInputs:
    """runner InputAssembler(딥러닝): fit(train 행 표) = DLPreprocessor fit(묶음마다), transform(행 표) =
    dict(x_seq·x_static·x_cat). cat_sizes = 범주 번호표 크기(OOV 포함) — models_dl 의 cat_sizes."""

    def __init__(self, ctx: InputContext, spec: rg.VersionSpec, split=None, *,
                 missing: rg.MissingLists | None = None, chunk_rows: int = 50_000):
        self.ctx, self.spec, self.split, self.chunk_rows = ctx, spec, _split(split), chunk_rows
        self._missing = missing
        self.prep: DLPreprocessor | None = None
        self._memo: PairMemo | None = None

    pair_cache = TreeInputs.pair_cache

    def fit(self, train_meta: pd.DataFrame) -> "DLInputs":
        self.prep = fit_dl_preprocessor(train_meta, self.ctx, self.spec, missing=self._missing, split=self.split,
                                        chunk_rows=self.chunk_rows)
        return self

    def transform(self, meta: pd.DataFrame) -> dict:
        if self.prep is None:
            raise InputError("fit 전")
        n = len(meta)
        out = None
        for sl, part in iter_dl_inputs(meta, self.ctx, self.prep, split=self.split, chunk_rows=self.chunk_rows,
                                       memo=self._memo):
            if out is None:
                out = {k: np.empty((n,) + v.shape[1:], dtype=v.dtype) for k, v in part.items()}
            for k, v in part.items():
                out[k][sl] = v
        if out is None:   # 0 행
            S = len(self.prep.static_names)
            out = {"x_seq": np.zeros((0, self.spec.W, len(self.prep.seq_channel_names)), np.float32),
                   "x_static": np.zeros((0, S), np.float32)}
            if self.prep.vocab is not None:
                out["x_cat"] = np.zeros((0, len(rg.CAT4_DL)), np.int64)
        return out

    @property
    def cat_sizes(self) -> list[int] | None:
        return None if self.prep is None else self.prep.cat_cardinalities

    @property
    def columns(self) -> list[str]:
        if self.prep is None:
            return []
        return ([f"x_seq:{c}" for c in self.prep.seq_channel_names] + [f"x_static:{c}" for c in self.prep.static_names]
                + ([f"x_cat:{c}" for c in rg.CAT4_DL] if self.prep.vocab is not None else []))

    def save(self, path) -> None:
        p = os.fspath(path)
        os.makedirs(p, exist_ok=False)
        self.prep.save(os.path.join(p, "dl_preprocessor.json"))


# ======================================================================== 결측 점검 도구(단계 0-6)
def range_nan_counts(cube: wn.DailyCube, codes, lo, hi) -> np.ndarray:
    """행마다 [lo, hi](일수, 양 끝 포함) 구간 변수별 NaN 날 수 (행, 변수). 표 밖 날짜도 NaN 으로 셈."""
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    ui = cube.unit_index(codes)
    nan = np.isnan(cube.values)
    U, D, V = nan.shape
    cs = np.zeros((U, D + 1, V), dtype=np.int32)
    np.cumsum(nan, axis=1, out=cs[:, 1:, :])
    a = np.clip(lo - cube.start, 0, D)
    b = np.clip(hi - cube.start + 1, 0, D)
    b = np.maximum(a, b)
    inside = cs[ui, b, :] - cs[ui, a, :]
    outside = (hi - lo + 1) - (b - a)
    return inside + outside[:, None]
