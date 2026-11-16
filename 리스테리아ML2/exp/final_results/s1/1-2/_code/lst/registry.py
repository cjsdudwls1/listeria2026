"""칼럼 등록부 — 계획서 §5 표, §4-2, 부록 A-5(빼기 단위 110·그룹 44), 부록 A-8(딥러닝 입력), §6 단계 0-8.

빼기 단위(부록 A-5): 원 변수 하나의 window 칼럼 묶음 = 1단위(SPI3·SPI6 도 각 1단위), 이동평균 차감은 원 변수마다
1단위(풍향 sin·cos 차감 함께 1단위), 그 밖 파생 칼럼 하나 = 1단위, 풍향 sin·cos 한 짝 = 1단위, aw + aw_label = 1단위
→ 113 칼럼 묶음에서 짝 3개로 110 단위. 그룹 = 기상 9(기온·강수·바람·햇빛·지중온도·토양수분·습도·대기질·계절)
+ 비기상 35(§5 표 순서대로 단위 하나씩: 이력 19·식품 7·토양 6·외부 지표 2·축사 1). 후보 아님 = 범주 5·h·g.

판 설정(VersionSpec) → 그 판의 트리 칼럼 목록·딥러닝 채널(x_seq)·x_static 칼럼 목록.
결측 표시 목록(단계 0-6, `MissingLists`) → 칼럼마다 결측 표시 칼럼·채널이 붙는지, '해외 구조적 결측' 1칼럼(§4-2).

구현 세부(계획서에 없음 — 담당자 확인):
 (a) 1차 기상 19변수 window 묶음의 ID 이름 = 'WEATHER'(§5 표에 ID 없음). 범주 5 = 'CAT5', h·g = 'TIME'.
 (b) 그룹 번호 = 부록 A-5 기상 9 순서(1~9) → 비기상 35 = §5 표 순서(10~44). 단위 번호 = 그룹 순 → 그룹 안 순서
     (기상 그룹 안: 부록 A-5 문구 순 — window 묶음 → 차감 묶음 → 그 밖 파생 칼럼).
 (c) 단위 이름(설정·기록용, W·k·N_ma 와 무관): window = 변수 이름(풍향 짝 = 'wind_dir'), 차감 = '<변수>_anom'
     (풍향 짝 = 'wind_dir_anom'), SPI = 'spi3'·'spi6', 그 밖 = 칼럼 이름(HIS1·HIS4 는 k 접미사 뺀 줄기,
     aw + aw_label = 'food_aw').
 (d) 축사 칼럼 이름 = 'L1'(§5 표에 칼럼 이름 없음, 단계 0-6 축사 표 칼럼 이름 그대로).
 (e) 계절·aw_label 딥러닝 원핫 칼럼 이름 = 'season_<봄…>'·'aw_label_<값>', aw_label 원핫 칸 = 단일값·중앙·이상·이하·
     초과·미만·하위중앙·결측 8칸(부록 A-4 값 목록 + '결측').
 (f) 결측 표시 칼럼 이름 = '<칼럼>__na', 채널 이름 = '<채널>__na', '해외 구조적 결측' 칼럼 = 'ov_struct_na'.
 (g) '해외 구조적 결측' 대상 = §4-2 가 이름으로 든 변수의 window·차감 칼럼·채널(soil_t5~t30·soil_t_nas·no2·co·so2·
     풍향 sin·cos, 2차는 + td1)과 거기서 나온 칼럼 — soil_t5 에서 나온 rg 지중온도 6칼럼도 넣음(§4-2 '그 변수들에서
     나온 칼럼별 결측 표시 = 0'; 부록 A-5 '모두 빠지면 같이 뺌' 은 이 묶음 전체로 — 계획서 문구 맞춤 예정, 11-07).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from . import config as cf
from . import windows as wn
from .history import HIS_LENGTHS, HIS_LEVELS, K_VALUES, WOE_LEVELS, k_tag
from .static_feats import FOOD_LABELS, SEASONS

# ---------------------------------------------------------------- 상수
WEATHER19 = ("avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "max_wind", "sunshine", "solar_rad",
             "soil_t5", "soil_t10", "soil_t20", "soil_t30", "soil_t_nas", "soil_moist",
             "no2", "co", "so2", "wind_dir_sin", "wind_dir_cos")          # §4-3 (wxfeat.WEATHER19 와 같음)
WIND_PAIR = ("wind_dir_sin", "wind_dir_cos")
SPI_VARS = ("spi3", "spi6")
N_MA_VALUES = (5, 7, 15, 30)                                              # §6 단계 2-1b
CAT5 = ("sido", "sgg", "p2", "p3", "src_route")                           # §6 1-1
CAT4_DL = ("sido", "sgg", "p2", "p3")                                     # 부록 A-8 x_cat
TIME_COLS = ("h", "g")
IDS = ("WEATHER", "HIS1", "HIS2", "HIS3", "HIS4", "WX1", "WX2", "WX3", "WX4", "WX5", "WX6", "WX7",
       "FD1", "SO1", "X1", "X2", "L1")
ORDER1_IDS = ("WEATHER",)
ALL_IDS = IDS
STRUCT_VARS = ("soil_t5", "soil_t10", "soil_t20", "soil_t30", "soil_t_nas", "no2", "co", "so2",
               "wind_dir_sin", "wind_dir_cos")                            # §4-2 (1차·2차)
STRUCT_TABLE_ORDER2 = ("td1",)                                            # §4-2 (2차)
STRUCTURAL_INCLUDE_RG_SOILT = True                                        # 구현 세부 (g) — 지휘자 결정 11-07
STRUCT_COLUMN = "ov_struct_na"
NA_SUFFIX = "__na"
AW_LABEL_VALUES = tuple(FOOD_LABELS) + ("하위중앙", "결측")              # 부록 A-4 + '결측'
AW_MISSING = "결측"
RESEARCH_UNITS = ("t3", "rain_lag2", "wind_lag2", "rain_mean25", "solar_23", "trange7", "rh1", "td1", "L1")  # 부록 A-10
WEATHER8_GROUPS = ("기온", "강수", "바람", "햇빛", "지중온도", "토양수분", "습도", "대기질")             # ⑧
K_PLACE, N_PLACE = "{K}", "{N}"                                           # 0-8 표의 이름 틀

# 생장식·생장 범위 칼럼(wxfeat 이름 그대로)
GR1 = ("gr1_mean", "gr1_mean10", "gr1_sum", "gr1_npos")
GR2 = tuple(f"gr2_{p}_{s}" for p in ("mu", "pscotta", "pl5") for s in ("mean", "mean10", "max"))
GR3 = ("gr3_mean", "gr3_mean10", "gr3_max")
RG_TEMP = tuple(f"{v}_{s}" for v in ("rg_tmax_opt", "rg_tavg_opt_dist", "rg_t_out_lim") for s in ("mean", "mean10"))
RG_SOILT = tuple(f"{v}_{s}" for v in ("rg_soilt_opt", "rg_soilt_lim", "rg_soilt_opt_dist") for s in ("mean", "mean10"))
RG_SOILPH = ("rg_soilph_opt", "rg_soilph_lim", "rg_soilph_opt_dist")
RG_ITEM = ("rg_aw_lim", "rg_aw_opt_dist", "rg_ph_lim", "rg_ph_opt_dist")
WX5 = ("t3", "rain_lag2", "wind_lag2", "rain_mean25", "solar_23", "trange7")
WX6 = ("rh1", "td1")
SO1 = ("soil_ph", "soil_om", "soil_vldpha")
FD1_NUM = ("food_aw", "food_ph", "food_na")
HIS1_STEMS = tuple(f"hist_rate{L}_{lv}" for L in HIS_LENGTHS for lv in HIS_LEVELS)
HIS2_STEMS = tuple(f"hist_logn{L}_{lv}" for L in HIS_LENGTHS for lv in HIS_LEVELS)
HIS3_STEMS = tuple(f"woe_{lv}" for lv in WOE_LEVELS)
HIS4_STEMS = tuple(f"hist_unc1y_{lv}" for lv in HIS_LEVELS)


class RegistryError(ValueError):
    pass


# ---------------------------------------------------------------- 단위·그룹
@dataclass(frozen=True)
class Unit:
    no: int
    name: str
    id: str                  # §5 ID(또는 WEATHER)
    kind: str                # 'window' | 'anom' | 'spi' | 'table'
    sources: tuple           # 원 변수(일자료 변수 또는 표 칼럼 줄기)
    group_no: int = 0
    group: str = ""
    window_end: str = ""     # 부록 B-6 칼럼마다 window 끝(E·R−1·T−1·C·d·고정)

    @property
    def n_bundles(self) -> int:
        """부록 A-5 '칼럼 묶음' 수(짝 = 2)."""
        if self.kind in ("window", "anom"):
            return len(self.sources)
        return 2 if self.name == "food_aw" else 1


@dataclass(frozen=True)
class Group:
    no: int
    name: str
    weather: bool
    units: tuple             # 단위 이름


def _w(v):  # window 단위
    return ("wind_dir" if v == "wind_dir" else v, "WEATHER", "window", WIND_PAIR if v == "wind_dir" else (v,), "E")


def _a(v):  # 차감 단위
    return (f"{v}_anom", "WX2", "anom", WIND_PAIR if v == "wind_dir" else (v,), "E")


def _t(name, id_, end):  # 표 칼럼 단위
    return (name, id_, "table", (name,), end)


_WEATHER_GROUPS = (
    ("기온", [_w("avg_temp"), _w("max_temp"), _w("min_temp"), _a("avg_temp"), _a("max_temp"), _a("min_temp"),
             _t("t3", "WX5", "R-1"), _t("trange7", "WX5", "R-1")]
     + [_t(c, "WX3", "E") for c in GR1 + GR2] + [_t(c, "WX4", "E") for c in RG_TEMP]),
    ("강수", [_w("rain"), _a("rain"), ("spi3", "WX1", "spi", ("spi3",), "E"), ("spi6", "WX1", "spi", ("spi6",), "E"),
             _t("rain_lag2", "WX5", "R-1"), _t("rain_mean25", "WX5", "R-1")] + [_t(c, "WX3", "E") for c in GR3]),
    ("바람", [_w("avg_wind"), _w("max_wind"), _a("avg_wind"), _a("max_wind"), _w("wind_dir"), _a("wind_dir"),
             _t("wind_lag2", "WX5", "R-1")]),
    ("햇빛", [_w("sunshine"), _w("solar_rad"), _a("sunshine"), _a("solar_rad"), _t("solar_23", "WX5", "R-1")]),
    ("지중온도", [_w(v) for v in ("soil_t5", "soil_t10", "soil_t20", "soil_t30", "soil_t_nas")]
     + [_a(v) for v in ("soil_t5", "soil_t10", "soil_t20", "soil_t30", "soil_t_nas")]
     + [_t(c, "WX4", "E") for c in RG_SOILT]),
    ("토양수분", [_w("soil_moist"), _a("soil_moist")]),
    ("습도", [_t("rh1", "WX6", "R-1"), _t("td1", "WX6", "R-1")]),
    ("대기질", [_w("no2"), _w("co"), _w("so2"), _a("no2"), _a("co"), _a("so2")]),
    ("계절", [_t("season", "WX7", "d")]),
)
_NONWEATHER_UNITS = (
    [_t(s, "HIS1", "C") for s in HIS1_STEMS] + [_t(s, "HIS2", "C") for s in HIS2_STEMS]
    + [_t(s, "HIS3", "C") for s in HIS3_STEMS] + [_t(s, "HIS4", "C") for s in HIS4_STEMS]
    + [_t(c, "FD1", "고정") for c in FD1_NUM] + [_t(c, "WX4", "고정") for c in RG_ITEM]
    + [_t(c, "SO1", "T-1") for c in SO1] + [_t(c, "WX4", "T-1") for c in RG_SOILPH]
    + [_t("eco_ratio", "X1", "T-1"), _t("gw_pass_rate", "X2", "T-1"), _t("L1", "L1", "고정")]
)


def _build():
    units, groups = [], []
    no = 0
    for gi, (gname, items) in enumerate(_WEATHER_GROUPS, start=1):
        names = []
        for (name, id_, kind, src, end) in items:
            no += 1
            units.append(Unit(no, name, id_, kind, tuple(src), gi, gname, end))
            names.append(name)
        groups.append(Group(gi, gname, True, tuple(names)))
    gi = len(groups)
    for (name, id_, kind, src, end) in _NONWEATHER_UNITS:
        gi += 1
        no += 1
        units.append(Unit(no, name, id_, kind, tuple(src), gi, name, end))
        groups.append(Group(gi, name, False, (name,)))
    return tuple(units), tuple(groups)


UNITS, GROUPS = _build()
_UNIT_BY_NAME = {u.name: u for u in UNITS}
_GROUP_BY_NAME = {g.name: g for g in GROUPS}
_GROUP_BY_NO = {g.no: g for g in GROUPS}


def _self_check():
    assert len(UNITS) == 110, len(UNITS)                                    # 부록 A-5
    assert sum(u.n_bundles for u in UNITS) == 113
    assert len(GROUPS) == 44 and sum(g.weather for g in GROUPS) == 9
    assert sum(len(g.units) for g in GROUPS if len(g.units) > 1) == 74      # 부록 A-9 단위 빼기 첫 바퀴 74
    nonw = [u for u in UNITS if not GROUPS[u.group_no - 1].weather]
    assert len(nonw) == 35
    assert len({u.name for u in UNITS}) == 110
    srcs = [s for u in UNITS if u.kind == "window" for s in u.sources]
    assert tuple(sorted(srcs)) == tuple(sorted(WEATHER19))
    srcs = [s for u in UNITS if u.kind == "anom" for s in u.sources]
    assert tuple(sorted(srcs)) == tuple(sorted(WEATHER19))
    for r in RESEARCH_UNITS:
        assert r in _UNIT_BY_NAME
    for g in WEATHER8_GROUPS:
        assert g in _GROUP_BY_NAME


_self_check()


def unit(name: str) -> Unit:
    try:
        return _UNIT_BY_NAME[name]
    except KeyError:
        raise RegistryError(f"모르는 단위: {name!r}") from None


def group(key) -> Group:
    if isinstance(key, int) and not isinstance(key, bool):
        if key not in _GROUP_BY_NO:
            raise RegistryError(f"모르는 그룹 번호: {key}")
        return _GROUP_BY_NO[key]
    if key not in _GROUP_BY_NAME:
        raise RegistryError(f"모르는 그룹: {key!r}")
    return _GROUP_BY_NAME[key]


# ---------------------------------------------------------------- 판 설정
VERSION_SCHEMA = {
    "name": cf.Field(str),
    "order": cf.Field(int, choices=(1, 2)),
    "W": cf.Field(int, choices=wn.W_GRID),
    "mode": cf.Field(str, choices=wn.MODES),
    "include_ids": cf.Field(list, item_types=str),
    "include_units": cf.Field(list, item_types=str, required=False),
    "exclude_units": cf.Field(list, item_types=str, required=False),
    "exclude_groups": cf.Field(list, item_types=(str, int), required=False),
    "k": cf.Field((int, type(None)), required=False),
    "n_ma": cf.Field((int, type(None)), required=False),
    "n_ma_dl": cf.Field((int, type(None)), required=False),
    "tree_layout": cf.Field(str, choices=("standard", "seq_only"), required=False),
    "dl_layout": cf.Field(str, choices=("standard", "seq_only"), required=False),
}


@dataclass(frozen=True)
class VersionSpec:
    """판 설정. order = 차수(1·2 — 결측 표시 목록·허용 ID), W = window(트리·딥러닝 같음 — §0),
    mode = 트리 입력 방식('sum' 요약·'raw' 원값), include_ids = §5 ID(+ WEATHER), include_units = ID 밖에서 더할 단위
    (2-R 하나씩 추가 판), exclude_units·exclude_groups = 3-0b 빼기 결과, k = 이력 k(2-1), n_ma = 트리 차감 N_ma,
    n_ma_dl = 딥러닝 차감 채널 N_ma(2-1b), tree_layout 'seq_only' = ⑦-T(시계열 채널 원값 lag + g),
    dl_layout 'seq_only' = ⑥·⑦-D(x_seq + g ÷ 50·해외 구조적 결측, x_cat 없음)."""
    name: str
    order: int
    W: int
    mode: str
    include_ids: tuple = ()
    include_units: tuple = ()
    exclude_units: tuple = ()
    exclude_groups: tuple = ()
    k: int | None = None
    n_ma: int | None = None
    n_ma_dl: int | None = None
    tree_layout: str = "standard"
    dl_layout: str = "standard"

    def __post_init__(self):
        if self.order not in (1, 2):
            raise RegistryError("order 는 1·2")
        if self.W not in wn.W_GRID:
            raise RegistryError(f"W 는 {wn.W_GRID} 중: {self.W}")
        if self.mode not in wn.MODES:
            raise RegistryError(f"mode 는 {wn.MODES}: {self.mode}")
        bad = [i for i in self.include_ids if i not in IDS]
        if bad:
            raise RegistryError(f"모르는 ID: {bad}")
        if len(set(self.include_ids)) != len(self.include_ids):
            raise RegistryError("include_ids 중복")
        if self.order == 1 and (set(self.include_ids) - set(ORDER1_IDS) or self.include_units):
            raise RegistryError("1차 판은 WEATHER 만(§6 1-1)")
        for n in self.include_units + self.exclude_units:
            unit(n)
        for g in self.exclude_groups:
            group(g)
        if self.k is not None and self.k not in K_VALUES:
            raise RegistryError(f"k 는 {K_VALUES} 중(§6 2-1): {self.k}")
        for nm, v in (("n_ma", self.n_ma), ("n_ma_dl", self.n_ma_dl)):
            if v is not None and v not in N_MA_VALUES:
                raise RegistryError(f"{nm} 는 {N_MA_VALUES} 중(§6 2-1b): {v}")
        ids = {u.id for u in self.units()}
        if ids & {"HIS1", "HIS4"} and self.k is None:
            raise RegistryError("HIS1·HIS4 가 있으면 k 필요")
        if self.tree_layout not in ("standard", "seq_only") or self.dl_layout not in ("standard", "seq_only"):
            raise RegistryError("layout 은 standard·seq_only")
        if self.tree_layout == "seq_only" and self.mode != "raw":
            raise RegistryError("⑦-T(seq_only) 는 원값 lag(§6 추가 판 ⑦)")
        if self.tree_layout == "seq_only" or self.dl_layout == "seq_only":
            other = [u.name for u in self.units() if u.kind not in ("window", "anom", "spi")]
            if other:
                raise RegistryError(f"seq_only 판엔 시계열 단위만: {other[:5]}")

    def units(self) -> list[Unit]:
        """이 판의 단위(단위 번호 순)."""
        drop_g = {group(g).no for g in self.exclude_groups}
        keep = {u.name for u in UNITS if u.id in self.include_ids} | set(self.include_units)
        keep -= set(self.exclude_units)
        return [u for u in UNITS if u.name in keep and u.group_no not in drop_g]

    def has_ids(self, *ids) -> bool:
        s = {u.id for u in self.units()}
        return any(i in s for i in ids)

    def to_dict(self) -> dict:
        return {"name": self.name, "order": self.order, "W": self.W, "mode": self.mode,
                "include_ids": list(self.include_ids), "include_units": list(self.include_units),
                "exclude_units": list(self.exclude_units), "exclude_groups": list(self.exclude_groups),
                "k": self.k, "n_ma": self.n_ma, "n_ma_dl": self.n_ma_dl,
                "tree_layout": self.tree_layout, "dl_layout": self.dl_layout}

    def with_(self, **kw) -> "VersionSpec":
        return replace(self, **kw)


def load_version(obj: str | os.PathLike | Mapping) -> VersionSpec:
    """판 설정 JSON(경로 또는 dict) → VersionSpec(모르는 칸이면 멈춤 — 부록 B-2)."""
    cfg = cf.load_config(obj, VERSION_SCHEMA) if not isinstance(obj, Mapping) else cf.validate(obj, VERSION_SCHEMA)
    kw = dict(cfg)
    for k in ("include_ids", "include_units", "exclude_units", "exclude_groups"):
        if k in kw:
            kw[k] = tuple(kw[k])
    return VersionSpec(**kw)


# ---------------------------------------------------------------- 칼럼
@dataclass(frozen=True)
class Column:
    name: str
    role: str                 # 'tree' | 'seq' | 'static'
    kind: str                 # 'category' | 'numeric' | 'time' | 'fixed' | 'onehot'
    unit: str | None = None   # 단위 이름(후보 아님 = None)
    unit_no: int | None = None
    group: str | None = None
    group_no: int | None = None
    id: str | None = None
    source: str | None = None  # 원 변수(결측 표시 목록의 단위)
    window_end: str | None = None
    structural: bool = False   # '해외 구조적 결측' 대상(§4-2)


def structural_sources(order: int) -> tuple:
    """그 차수의 '해외 구조적 결측' 원 변수(일자료 변수·표 칼럼) — 1차 = STRUCT_VARS, 2차 = + td1 (+ rg 지중온도 6,
    STRUCTURAL_INCLUDE_RG_SOILT 일 때). 단계 0-6 목록 판정(국내 행으로만)과 MissingLists 대조에 씀."""
    if order == 1:
        return STRUCT_VARS
    return STRUCT_VARS + STRUCT_TABLE_ORDER2 + (RG_SOILT if STRUCTURAL_INCLUDE_RG_SOILT else ())


def _structural(u: Unit, src: str, order: int) -> bool:
    if u.kind in ("window", "anom") and src in STRUCT_VARS:
        return True
    if order == 2 and u.kind == "table" and src in STRUCT_TABLE_ORDER2:
        return True
    if STRUCTURAL_INCLUDE_RG_SOILT and src in RG_SOILT:
        return True
    return False


def _fmt_k(k) -> str:
    return K_PLACE if k is None else k_tag(k)


def _fmt_n(n) -> str:
    return N_PLACE if n is None else str(int(n))


def table_column_name(stem: str, k: int | None) -> str:
    """표 칼럼 이름: HIS1·HIS4 = 줄기 + '_k###'(lst.history 이름), 그 밖 = 줄기 그대로."""
    if stem in HIS1_STEMS or stem in HIS4_STEMS:
        return f"{stem}_{_fmt_k(k)}"
    return stem


def anomaly_var(var: str, n_ma: int | None) -> str:
    """차감 일자료 변수 이름(windows.anomaly_name 과 같음)."""
    return wn.anomaly_name(var, n_ma) if n_ma is not None else f"{var}_anom{N_PLACE}"


def unit_tree_columns(u: Unit, W: int, mode: str, k: int | None, n_ma: int | None, order: int = 2) -> list[Column]:
    """단위 하나의 트리 칼럼(이름 틀: k·n_ma 가 None 이면 {K}·{N} 자리표시)."""
    base = dict(role="tree", unit=u.name, unit_no=u.no, group=u.group, group_no=u.group_no, id=u.id,
                window_end=u.window_end)
    out = []
    if u.kind in ("window", "spi"):
        for v in u.sources:
            for c in wn.window_feature_names([v], W, mode):
                out.append(Column(c, kind="numeric", source=v, structural=_structural(u, v, order), **base))
    elif u.kind == "anom":
        for v in u.sources:
            for c in wn.window_feature_names([anomaly_var(v, n_ma)], W, mode, anomaly=True):
                out.append(Column(c, kind="numeric", source=v, structural=_structural(u, v, order), **base))
    else:
        s = u.sources[0]
        if u.name == "season":
            out.append(Column("season", kind="category", source="season", **base))
        elif u.name == "food_aw":
            out.append(Column("food_aw", kind="numeric", source="food_aw", **base))
            out.append(Column("aw_label", kind="category", source="aw_label", **base))
        else:
            out.append(Column(table_column_name(s, k), kind="numeric", source=s,
                              structural=_structural(u, s, order), **base))
    return out


_KIND_ORDER = {"window": 0, "anom": 1, "spi": 2, "table": 3}


def ordered_units(units: Sequence[Unit]) -> list[Unit]:
    """입력 칼럼·채널 순서(부록 A-8 '기상 19 + 차감 19 + SPI3·SPI6' 순): window(WEATHER19 순) → 차감(WEATHER19 순) →
    SPI → 표 칼럼(단위 번호 순 = §5 표·부록 A-5 순). 구현 세부(계획서에 없음)."""
    def key(u):
        if u.kind in ("window", "anom"):
            return (_KIND_ORDER[u.kind], WEATHER19.index(u.sources[0]))
        return (_KIND_ORDER[u.kind], u.no)
    return sorted(units, key=key)


def tree_columns(spec: VersionSpec) -> list[Column]:
    """그 판의 트리 입력 칼럼(순서 = 범주 5 → h·g → ordered_units 순). ⑦-T(seq_only) = 시계열 단위 원값 lag + g
    (차감 채널은 n_ma 값 — ⑦-T 설정에는 딥러닝 N_ma 를 넣음)."""
    out = []
    if spec.tree_layout == "standard":
        out += [Column(c, role="tree", kind="category", source=c) for c in CAT5]
        out += [Column(c, role="tree", kind="time") for c in TIME_COLS]
    else:
        out += [Column("g", role="tree", kind="time")]
    for u in ordered_units(spec.units()):
        if u.kind == "anom" and spec.n_ma is None:
            raise RegistryError("차감(WX2) 단위가 있으면 n_ma 필요")
        out += unit_tree_columns(u, spec.W, spec.mode, spec.k, spec.n_ma, spec.order)
    _no_dup(out)
    return out


def unit_seq_channels(u: Unit, n_ma_dl: int | None, order: int = 2) -> list[Column]:
    base = dict(role="seq", kind="numeric", unit=u.name, unit_no=u.no, group=u.group, group_no=u.group_no, id=u.id,
                window_end=u.window_end)
    if u.kind in ("window", "spi"):
        return [Column(v, source=v, structural=_structural(u, v, order), **base) for v in u.sources]
    if u.kind == "anom":
        return [Column(anomaly_var(v, n_ma_dl), source=v, structural=_structural(u, v, order), **base)
                for v in u.sources]
    return []


def unit_static_columns(u: Unit, k: int | None, order: int = 2) -> list[Column]:
    """x_static 의 단위 칼럼(표 칼럼만 — 부록 A-8): 계절 = 원핫 4칸, aw + aw_label = food_aw + aw_label 원핫 8칸."""
    if u.kind != "table":
        return []
    base = dict(role="static", unit=u.name, unit_no=u.no, group=u.group, group_no=u.group_no, id=u.id,
                window_end=u.window_end)
    if u.name == "season":
        return [Column(f"season_{s}", kind="onehot", source="season", **base) for s in SEASONS]
    if u.name == "food_aw":
        return ([Column("food_aw", kind="numeric", source="food_aw", **base)]
                + [Column(f"aw_label_{v}", kind="onehot", source="aw_label", **base) for v in AW_LABEL_VALUES])
    s = u.sources[0]
    return [Column(table_column_name(s, k), kind="numeric", source=s, structural=_structural(u, s, order), **base)]


def dl_seq_channels(spec: VersionSpec) -> list[Column]:
    """x_seq 값 채널(결측 표시 채널 제외), ordered_units 순(기상 → 차감 → SPI). 차감 채널 = n_ma_dl(2-1b 딥러닝 값)."""
    out = []
    for u in ordered_units(spec.units()):
        if u.kind == "anom" and spec.n_ma_dl is None:
            raise RegistryError("차감 채널이 있으면 n_ma_dl 필요")
        out += unit_seq_channels(u, spec.n_ma_dl, spec.order)
    _no_dup(out)
    return out


def dl_fixed_static(spec: VersionSpec) -> list[Column]:
    """x_static 고정 칼럼(후보 아님): standard = g ÷ 50·src_route(0/1), seq_only = g ÷ 50. (구조적 결측 칼럼은
    has_structural 일 때 전처리기가 붙임)."""
    out = [Column("g_div50", role="static", kind="fixed")]
    if spec.dl_layout == "standard":
        out.append(Column("src_route01", role="static", kind="fixed", source="src_route"))
    return out


def dl_static_columns(spec: VersionSpec) -> list[Column]:
    """x_static 단위 칼럼(표 칼럼, 단위 번호 순) — seq_only 판은 없음."""
    if spec.dl_layout == "seq_only":
        return []
    out = []
    for u in spec.units():
        out += unit_static_columns(u, spec.k, spec.order)
    _no_dup(out)
    return out


def has_structural(cols: Iterable[Column]) -> bool:
    return any(c.structural for c in cols)


def _no_dup(cols: Sequence[Column]) -> None:
    names = [c.name for c in cols]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise RegistryError(f"칼럼 이름 중복: {dup[:5]}")


# ---------------------------------------------------------------- 결측 표시 목록(단계 0-6)
@dataclass(frozen=True)
class MissingLists:
    """차수(1·2)마다 결측 표시 대상 원 변수(일자료 변수·표 칼럼 줄기) 집합 — §4-2, 단계 0-6 결과."""
    lists: Mapping[int, frozenset]
    source: str = ""

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "MissingLists":
        obj = cf.load_json_strict(path)
        try:
            lists = {1: frozenset(obj["order1"]["variables"]), 2: frozenset(obj["order2"]["variables"])}
            if tuple(obj["order1"]["structural_variables"]) != structural_sources(1):
                raise RegistryError("결측 표시 목록의 1차 구조적 결측 변수가 등록부와 다름")
            if tuple(obj["order2"]["structural_variables"]) != structural_sources(2):
                raise RegistryError("결측 표시 목록의 2차 구조적 결측 변수가 등록부와 다름")
        except KeyError as e:
            raise RegistryError(f"결측 표시 목록 JSON 칸 없음: {e}") from None
        return cls(lists, str(path))

    def needs_indicator(self, col: Column, order: int) -> bool:
        """그 칼럼·채널에 결측 표시가 붙는지(원 변수가 그 차수 목록에 있으면). 범주·원핫·고정 칼럼은 안 붙음."""
        if col.kind not in ("numeric",) or col.source is None:
            return False
        return col.source in self.lists[order]


def indicator_name(name: str) -> str:
    return name + NA_SUFFIX


# ---------------------------------------------------------------- 0-8 표
def group_table() -> pd.DataFrame:
    return pd.DataFrame([{"group_no": g.no, "group": g.name, "weather": g.weather, "n_units": len(g.units),
                          "units": "|".join(g.units)} for g in GROUPS])


def unit_table() -> pd.DataFrame:
    return pd.DataFrame([{"unit_no": u.no, "unit": u.name, "id": u.id, "kind": u.kind, "sources": "|".join(u.sources),
                          "group_no": u.group_no, "group": u.group, "n_bundles": u.n_bundles,
                          "window_end": u.window_end, "research": u.name in RESEARCH_UNITS} for u in UNITS])


def membership_table(W: int, k: int | None = None, n_ma: int | None = None, n_ma_dl: int | None = None,
                     missing: MissingLists | None = None, order: int = 2) -> pd.DataFrame:
    """단계 0-8: 단위마다 소속 칼럼 전부 — 트리 요약·원값, 딥러닝 x_seq 채널·x_static 칼럼(+ 결측 표시 칼럼·채널,
    missing 이 있을 때). k·n_ma·n_ma_dl 이 None 이면 이름에 {K}·{N} 자리표시."""
    rows = []

    def add(u, c: Column, where: str, indicator: bool = False):
        rows.append({"group_no": u.group_no, "group": u.group, "unit_no": u.no, "unit": u.name, "id": u.id,
                     "where": where, "column": indicator_name(c.name) if indicator else c.name,
                     "kind": "indicator" if indicator else c.kind, "source": c.source, "window_end": c.window_end,
                     "structural": c.structural, "W": W})

    for u in UNITS:
        for mode in wn.MODES:
            for c in unit_tree_columns(u, W, mode, k, n_ma, order):
                add(u, c, f"tree_{mode}")
                if missing is not None and missing.needs_indicator(c, order):
                    add(u, c, f"tree_{mode}", indicator=True)
        for c in unit_seq_channels(u, n_ma_dl, order):
            add(u, c, "dl_seq")
            if missing is not None and missing.needs_indicator(c, order):
                add(u, c, "dl_seq", indicator=True)
        for c in unit_static_columns(u, k, order):
            add(u, c, "dl_static")
            if missing is not None and missing.needs_indicator(c, order):
                add(u, c, "dl_static", indicator=True)
    return pd.DataFrame(rows)


def representative_columns(W: int, k: int, n_ma: int) -> dict[str, str]:
    """단계 0-8 그룹 사이 상관용 단위 대표 칼럼(요약 판): window·SPI = `<변수>_mean`, 차감 = `<변수>_anom{N}_mean`,
    풍향 짝 = sin 쪽, 표 칼럼 = 그 칼럼(범주 season·aw_label 제외 — Spearman 대상 아님).
    구현 세부(계획서 '`<변수>_mean` Spearman' 을 단위 대표 칼럼으로 넓힘)."""
    out = {}
    for u in UNITS:
        v = u.sources[0]
        if u.kind in ("window", "spi"):
            out[u.name] = f"{v}_mean"
        elif u.kind == "anom":
            out[u.name] = f"{anomaly_var(v, n_ma)}_mean"
        elif u.name == "season":
            continue
        else:
            out[u.name] = table_column_name(v, k)
    return out
