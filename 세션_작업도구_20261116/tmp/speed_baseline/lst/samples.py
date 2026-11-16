"""시료 표(단계 0-1·0-2) — 계획서 §2-1·§2-2(RR1~RR8)·§2-3·§3·§6 단계 0-1·0-2,
부록 A-1(h)·A-2(해외 나라 코드 순서·나라 경계 파일)·A-6(Theil's U 대상 후보)·B-6(점검).

흐름
  1. 읽기·다시 셈(§2-1) → §3 기록값 대조(시료 파일 − RR2 361행, 그 밖 규칙 전 — 다르면 멈춤).
  2. RR1(해외 행 중 좌표가 국내 → 국내로 옮기고 `02_시군구_배정.py` 규칙으로 시군구) → RR2(제외)
     → RR3('코드없음') → RR4(DB 나라 코드 못 정한 행: 좌표·주소, 그래도 못 정하면 제외)
     → RR5(좌표 나라 ≠ 제조국 → 제조국) → RR6(표시) → RR7(표시) → RR8(제외, `lst.windows.rr8_mask`).
  3. 운영 단위 칼럼(§2-3): sgg·sido·p2·p3·src_route, h(부록 A-1), d = mnfctur_de.
  4. 점검표: 적용 후 수, 덜 찬 달(§3), 0-2 나라별 표, 부록 B-6.

'담당자 확인'(나라 코드 원천·Theil's U 대상 칼럼 목록)은 이 모듈이 정하지 않음 — check 단계에서
확인용 표를 만들고, apply 단계는 담당자가 확인한 목록(설정 파일)을 받아서만 씀.

구현 세부(계획서에 없음 — 담당자 확인 필요) — 반환 보고에도 같은 목록:
 (a) RR1 '좌표가 국내' = 나라 경계 파일(부록 A-2)로 정한 좌표 나라가 KOR(다각형 안, 어느 다각형에도
     안 들면 가장 가까운 나라 — `lst.geo`). `02_시군구_배정.py` 는 법정동 코드(+주소) 규칙이라 좌표 입력이
     없음 → 좌표에서 가장 가까운 `raw_korea_region_code` '존재' 읍면동·리 대표점(대권 거리)의 10자리 코드를
     그 규칙(`assign_sgg`, 주소 = 시료 `addr`)에 넣음. 점검용으로 가장 가까운 국내 시료 좌표의 `sgg_code` 도 표에.
 (b) A-2 원천 값 → alpha-3: `nation_code.alpha_2` 와 같으면 그 `alpha_3`, `alpha_3` 와 같으면 그대로(공백 제거·
     대소문자 바꿈 없음). 원천 값이 있어도 이 표로 못 맞추면(예: 'KOREA'·'U.S.A') 다음 원천으로 넘어감(표에 셈).
     s_org 는 `rcept_no` 마다 나라 1개여야 함(아니면 멈춤) — (`rcept_no`, `mnf_nation_cd`) 중복 행은 하나로.
 (c) RR4 남은 행: 좌표 나라(a 와 같은 방법) → 좌표가 없거나 두 나라 안에 들면 주소의 나라 이름
     (`nation_code.name_en` 전체 또는 쉼표 앞 부분이 낱말로 하나만 나올 때) → 그래도 없으면 제외.
     좌표 나라 코드가 `nation_code.alpha_3` 에 없어도(Natural Earth 전용 코드) 그대로 씀(표에 표시).
 (d) RR5 '좌표 나라' = (a) 의 좌표 나라(가장 가까운 나라 포함), 비교 대상 = DB 원천으로 정한 행만.
 (e) 'OVKOR' 셈 = RR1 뒤 해외 행 중 최종 나라 = KOR.
 (f) 빈 칸 = 결측(null/NaN). 시료 파일에 빈 문자열은 없음(구조 확인) — 빈 문자열이 나오면 멈춤.
 (g) d = `mnfctur_de` 날짜(시각이 00:00 이 아니면 멈춤).
 (h) 부록 B-6 수치표의 행 = RR1 적용·RR2·RR8 제외 뒤 시료(RR4 제외는 원천 확인 뒤라 반영 안 함 — check 단계),
     `src_route` 는 RR1 뒤 값. '검출 행에만 나오는 값' 비율 = 두 가지(고유값 중 비율·검출 행 중 그런 값 행 비율).
     d ≤ 2025-06-30 은 같은 지표를 재 봉인 파일로만 쓰고, 이름 목록 = |검출 빈 칸 비율 − 불검출 빈 칸 비율| 이
     d ≤ 2021-06-30 때보다 큰(반올림 없이) 칼럼·원천.
 (i) 0-2 나라별 표 = d ≤ 2021-06-30 시료 수·검출 수만(그 뒤 기간 집계는 최종 보고 때 — 시료 수도 안 냄),
     제조국 수 = 최종 해외 행의 나라 코드 고유값 수(전체 기간·d ≤ 2021-06-30 둘 다, 정답 안 씀).
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from types import ModuleType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import folds as fd
from .geo import CountryPolygons, haversine_km
from .hassign import assign_h_many
from .theil import codes_with_nan, theils_u
from .windows import rr8_mask

# ---------------------------------------------------------------- 입력(자료 뿌리 아래 상대 경로)
INPUTS = {
    "sample": "리스테리아ML/data/listeria_data.parquet",                     # §2-1
    "sgg": "리스테리아ML2/data/데이터셋/시료_시군구.parquet",                  # §2-1
    "rr2_summary": "리스테리아ML2/data/중간/시군구_미배정_삭제행_요약.csv",     # §2-2 RR2
    "sgg02": "리스테리아ML2/notebook/02_시군구_배정.py",                      # §2-1·RR1 규칙
    "krc": "리스테리아ML2/data/raw/raw_korea_region_code.parquet",           # 02 규칙의 법정동 표
    "sgg_xlsx": "리스테리아ML2/data/시군구_코드_대응표.xlsx",                 # 02 규칙의 대응표
    "s_org": "리스테리아ML2/data/추출_20261107/s_org_lims_data_info_listeria.parquet",   # A-2 1순위
    "hazard": "리스테리아ML2/data/추출_20261107/hazard_preprocessed_data_nation.parquet",  # A-2 2~4순위
    "nation_code": "리스테리아ML2/data/추출_20261107/nation_code.parquet",    # A-2 alpha 맞춤
    "ne": "리스테리아ML2/data/외부_수집/축사/ne_10m_admin_0_countries_v5.1.2.geojson",   # A-2 나라 경계
}

# ---------------------------------------------------------------- 계획서 기록값
RECORDED_BASIC = {  # §2-1(단계 0 에서 다시 셈 — 다르면 보고)
    "n_rows": 676_520, "n_cols": 25, "n_domestic": 628_878, "n_overseas": 47_642,
    "pos": 1_969, "pos_domestic": 1_844, "pos_overseas": 125,
    "d_min": "2015-04-07", "d_max": "2026-06-30", "d_min_overseas": "2015-05-21", "d_max_overseas": "2026-06-20",
    "sgg_rows": 628_517,
}
RECORDED_RR = {  # §2-2·§2-3(다르면 보고, 멈추지 않음)
    "RR1_rows": 78, "RR1_coords": 9, "RR1_pos": 0, "RR2_rows": 361, "RR2_pos": 0,
    "RR3_both_empty": 130, "RR4_rows": 211, "RR5_rows": 4_194, "RR6_rows": 14, "RR8_rows": 9,
    "OVKOR_rows": 123,
}
RR8_RECORDED_D = ("2015-04-07", "2015-05-27")
JUDGE = {"검출": 1, "불검출": 0}           # §2-1: 그 밖 값이면 멈춤
NO_CODE = "코드없음"                       # §2-2 RR3
RR6_CODES = ("B0204000000000", "B0208000000000", "C0310020000000")
NATION_SOURCES = ("mnf_nation_cd", "prdctn_nation_code", "orgplce_nation_code", "import_nation_code")  # A-2 순서
THEIL_CANDIDATES = (  # 부록 A-6 대상 후보(순서 그대로)
    "hazard_label", "src_route", "data_source", "expr_iem_nm", "inspct_instt_code", "prdlst_code",
    "train_prdlst_code", "train_prdlst_nm", "train_prdlst_code_2", "train_prdlst_nm_2", "prduct_nm",
    "mnfctur_cmpny_nm", "date_basis", "addr", "addr_source", "legal_dong_code", "lat", "lon",
)
B6_JUDGE_CUT = "2021-06-30"    # 부록 B-6·§6 0-2: 판단용 = d ≤ 2021-06-30
B6_SEALED_CUT = "2025-06-30"   # 부록 B-6: 수치는 최종 보고 전까지 안 보여 줌
KOREA = "KOR"
OV = "OV"

# 시료 표 칼럼(계획서에 있는 것만 — 부록 B-2 check_columns)
SAMPLE_COLUMNS = (
    "lims_data_key",   # §2-1
    "y",               # §1-1 judge_result 검출 1·불검출 0
    "src_route",       # §2-3 (RR1 뒤)
    "sgg", "sido", "p2", "p3",  # §2-3
    "d",               # §2-1 mnfctur_de
    "h",               # 부록 A-1
    "nation_a3",       # §2-3 해외 나라 alpha-3(국내 null)
    "nation_src",      # §6 0-1·부록 B-6 나라 코드 원천
    "rr1_moved",       # RR1 국내로 옮김
    "rr3_p2_nocode", "rr3_p3_nocode",  # RR3 빈 칸 → '코드없음'
    "rr5_coord_differs",  # RR5 좌표 나라 ≠ 제조국(해외 DB 원천 행만, 그 밖 null)
    "rr6_overseas_only_p3",  # RR6
    "rr7_jul01",       # RR7 d = 그해 07-01
)
NATION_SRC_COORD_INSIDE = "coord_inside"
NATION_SRC_COORD_NEAREST = "coord_nearest"
NATION_SRC_ADDR = "addr"


class PlanCountMismatch(AssertionError):
    """§3 기록값과 다름 — 멈춤(계획서 §3)."""


# ---------------------------------------------------------------- 읽기
def _to_plain(df: pd.DataFrame) -> pd.DataFrame:
    """범주(dictionary) 칼럼을 object 로(값 그대로)."""
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def read_parquet(path, columns=None) -> pd.DataFrame:
    return _to_plain(pq.read_table(path, columns=columns).to_pandas())


def load_sgg02(path) -> ModuleType:
    """`02_시군구_배정.py` 를 모듈로 불러옴(main() 은 안 돔 — 그 파일 설명대로)."""
    spec = importlib.util.spec_from_file_location("sgg02_rule", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("Region", "load_mapping", "assign_sgg"):
        if not hasattr(mod, name):
            raise AttributeError(f"02 규칙 파일에 {name} 없음")
    return mod


def check_no_empty_strings(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """구현 세부 (f): 빈 문자열이 있으면 멈춤(빈 칸 = 결측으로만 정의했으므로)."""
    bad = {}
    for c in columns:
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            m = s.map(lambda v: isinstance(v, str) and v.strip() == "")
            if m.any():
                bad[c] = int(m.sum())
    if bad:
        raise ValueError(f"빈 문자열 칸(정의 밖 — 담당자 확인): {bad}")


def prepare_basic(sample: pd.DataFrame) -> pd.DataFrame:
    """y·d·h 만들기(§2-1·§1-1·부록 A-1). 원 칼럼은 그대로 둠."""
    if sample["lims_data_key"].isna().any() or sample["lims_data_key"].duplicated().any():
        raise ValueError("lims_data_key 결측·중복")
    jr = sample["judge_result"]
    bad = sorted(set(jr.dropna().unique()) - set(JUDGE)) + (["<결측>"] if jr.isna().any() else [])
    if bad:
        raise ValueError(f"judge_result 에 '검출'·'불검출' 밖 값(§2-1 — 멈춤): {bad}")
    out = sample.copy()
    out["y"] = jr.map(JUDGE).astype(np.int8)
    ts = pd.to_datetime(out["mnfctur_de"])
    if ts.isna().any():
        raise ValueError("mnfctur_de 결측")
    if (ts != ts.dt.normalize()).any():
        raise ValueError("mnfctur_de 에 시각(00:00 아님) — 구현 세부 (g), 멈춤")
    out["d"] = ts.dt.normalize().to_numpy().astype("datetime64[D]")
    out["h"] = assign_h_many(out["lims_data_key"].tolist()).astype(np.int16)
    sr = out["src_route"]
    if not set(sr.dropna().unique()) <= {"domestic", "overseas"} or sr.isna().any():
        raise ValueError(f"src_route 값 밖: {sorted(set(sr.dropna().unique()))}")
    return out


# ---------------------------------------------------------------- §2-1 다시 셈·§3 대조
def _ds(x) -> str:
    return str(np.datetime64(x, "D"))


def basic_counts(s: pd.DataFrame, n_cols: int, n_sgg: int) -> pd.DataFrame:
    ov = s["src_route"] == "overseas"
    obs = {
        "n_rows": len(s), "n_cols": n_cols, "n_domestic": int((~ov).sum()), "n_overseas": int(ov.sum()),
        "pos": int(s["y"].sum()), "pos_domestic": int(s.loc[~ov, "y"].sum()), "pos_overseas": int(s.loc[ov, "y"].sum()),
        "d_min": _ds(s["d"].min()), "d_max": _ds(s["d"].max()),
        "d_min_overseas": _ds(s.loc[ov, "d"].min()), "d_max_overseas": _ds(s.loc[ov, "d"].max()),
        "sgg_rows": n_sgg,
    }
    return pd.DataFrame([{"item": k, "recorded": RECORDED_BASIC[k], "observed": obs[k],
                          "match": RECORDED_BASIC[k] == obs[k], "plan": "§2-1"} for k in RECORDED_BASIC])


def split_count_table(d, y, src_route, recorded: Mapping | None) -> pd.DataFrame:
    """§3 표의 셈(lst.folds.count_split) — recorded 가 있으면 기록값과 나란히."""
    days = fd.to_days(pd.Series(d))
    rows = []
    for name, sp in fd.SPLITS.items():
        obs = fd.count_split(days, y, src_route, sp)
        for k, v in obs.items():
            r = {"split": name, "item": k, "observed": v}
            if recorded is not None:
                rec = recorded[name].get(k)
                r.update({"recorded": rec, "match": rec == v})
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- RR2
def rr2_mask(s: pd.DataFrame, sgg_keys: set) -> np.ndarray:
    """국내(원 src_route) 행 중 `시료_시군구.parquet` 에 없는 행."""
    return ((s["src_route"] == "domestic") & ~s["lims_data_key"].isin(sgg_keys)).to_numpy()


def rr2_check(s: pd.DataFrame, mask: np.ndarray, summary: pd.DataFrame) -> pd.DataFrame:
    """RR2 행을 요약 CSV(사유 × 원래 시군구 코드별 행 수·검출 수)와 대조."""
    need = {"원래_시군구코드", "행수", "검출수"}
    if not need <= set(summary.columns):
        raise ValueError(f"RR2 요약 칼럼 없음: {need - set(summary.columns)}")
    sub = s.loc[mask]
    obs = (sub.assign(o=sub["legal_dong_code"].astype(str).str[:5])
              .groupby("o").agg(n=("y", "size"), pos=("y", "sum")))
    rec = summary.assign(o=summary["원래_시군구코드"].astype(str).str.zfill(5)).groupby("o")[["행수", "검출수"]].sum()
    t = obs.join(rec, how="outer").fillna(0).astype(np.int64).reset_index().rename(columns={"o": "원래_시군구코드"})
    t["match"] = (t["n"] == t["행수"]) & (t["pos"] == t["검출수"])
    return t


# ---------------------------------------------------------------- 나라 코드 원천(A-2)
def nation_maps(nc: pd.DataFrame) -> tuple[dict, set]:
    a2 = nc["alpha_2"]
    a3 = nc["alpha_3"]
    if a2.isna().any() or a3.isna().any() or a2.duplicated().any() or a3.duplicated().any():
        raise ValueError("nation_code alpha_2·alpha_3 결측·중복")
    return dict(zip(a2, a3)), set(a3)


def map_alpha(values: pd.Series, a2_to_a3: dict, a3_set: set) -> pd.Series:
    """구현 세부 (b): alpha-2 → alpha-3, alpha-3 는 그대로, 못 맞추면 None."""
    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if v in a2_to_a3:
            return a2_to_a3[v]
        if v in a3_set:
            return v
        return None
    return values.map(f).astype(object)


def load_nation_sources(keys: pd.Series, hazard: pd.DataFrame, s_org: pd.DataFrame) -> pd.DataFrame:
    """시료 행 순서대로 A-2 원천 4칸(원값). 조인 키: lims_data_key(hazard), cntc_sys_key2 = rcept_no(s_org)."""
    if hazard["lims_data_key"].duplicated().any():
        raise ValueError("hazard lims_data_key 중복(멈춤 — B-6)")
    so = s_org.drop_duplicates(["rcept_no", "mnf_nation_cd"])
    if so["rcept_no"].duplicated().any():
        raise ValueError("s_org rcept_no 하나에 나라 코드 여럿(멈춤)")
    left = pd.DataFrame({"lims_data_key": keys.to_numpy()})
    m = left.merge(hazard[["lims_data_key", "cntc_sys_key2", "prdctn_nation_code", "orgplce_nation_code",
                           "import_nation_code"]], on="lims_data_key", how="left", validate="many_to_one",
                   indicator=True)
    miss = int((m["_merge"] != "both").sum())
    if miss:
        raise ValueError(f"hazard 에 없는 시료 {miss}행(멈춤 — B-6 조인 뒤 결측)")
    m = m.drop(columns="_merge").merge(so[["rcept_no", "mnf_nation_cd"]], left_on="cntc_sys_key2",
                                       right_on="rcept_no", how="left", validate="many_to_one")
    if len(m) != len(keys):
        raise AssertionError("조인 뒤 행 수 바뀜")
    return _to_plain(m[["lims_data_key", "cntc_sys_key2", *NATION_SOURCES]].reset_index(drop=True))


def resolve_db_nation(src: pd.DataFrame, sources: Sequence[str], a2_to_a3: dict, a3_set: set
                      ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """sources 순서대로 첫 '맞춘' 값(구현 세부 b). 돌려줌: (alpha-3, 원천 이름, 원천별 셈 표)."""
    n = len(src)
    a3 = np.full(n, None, dtype=object)
    name = np.full(n, None, dtype=object)
    rows = []
    for s_ in sources:
        if s_ not in NATION_SOURCES:
            raise ValueError(f"모르는 원천: {s_}")
        raw = src[s_]
        mapped = map_alpha(raw, a2_to_a3, a3_set).to_numpy()
        has = raw.notna().to_numpy()
        ok = np.array([v is not None for v in mapped])
        take = ok & np.array([v is None for v in a3])
        rows.append({"source": s_, "has_value": int(has.sum()), "mapped": int(ok.sum()),
                     "unmappable_value": int((has & ~ok).sum()), "decided_here": int(take.sum())})
        a3[take] = mapped[take]
        name[take] = s_
    return a3, name, pd.DataFrame(rows)


_WORD = r"(?<![A-Z0-9]){}(?![A-Z0-9])"


def address_countries(addrs: Sequence, nc: pd.DataFrame) -> list[list[str]]:
    """구현 세부 (c): 주소에 `name_en`(전체 또는 쉼표 앞)이 낱말로 든 나라 alpha-3 목록."""
    pats = []
    for name_en, a3 in zip(nc["name_en"], nc["alpha_3"]):
        if not isinstance(name_en, str):
            continue
        names = {name_en.strip().upper(), name_en.split(",")[0].strip().upper()}
        for nm in names:
            if len(nm) >= 3:
                pats.append((re.compile(_WORD.format(re.escape(nm))), a3))
    out = []
    for a in addrs:
        u = a.upper() if isinstance(a, str) else ""
        out.append(sorted({a3 for p, a3 in pats if u and p.search(u)}))
    return out


# ---------------------------------------------------------------- 좌표 나라
def coord_countries(lat, lon, polys: CountryPolygons) -> pd.DataFrame:
    """고유 좌표마다 나라를 정하고 행으로 펼침(점검 표용 method·거리 포함)."""
    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    key = pd.DataFrame({"lat": la, "lon": lo})
    u = key.drop_duplicates().reset_index(drop=True)
    code, method, dist = polys.assign(u["lat"].to_numpy(), u["lon"].to_numpy())
    u["coord_cc"], u["coord_method"], u["coord_dist_km"] = code, method, dist
    out = key.merge(u, on=["lat", "lon"], how="left", validate="many_to_one")
    if len(out) != len(key):
        raise AssertionError("좌표 펼침 행 수 바뀜")
    return out


def coord_countries_for(s: pd.DataFrame, mask: np.ndarray, polys: CountryPolygons) -> pd.DataFrame:
    """mask 행(해외)만 좌표 나라를 정하고, 그 밖 행은 None(시료 파일 행 순서 그대로)."""
    out = pd.DataFrame({"coord_cc": np.full(len(s), None, dtype=object),
                        "coord_method": np.full(len(s), None, dtype=object),
                        "coord_dist_km": np.full(len(s), np.nan)})
    idx = np.flatnonzero(mask)
    if idx.size:
        c = coord_countries(s["lat"].to_numpy()[idx], s["lon"].to_numpy()[idx], polys)
        for k in ("coord_cc", "coord_method"):
            v = out[k].to_numpy(dtype=object).copy()
            v[idx] = _na_to_none(c[k].to_numpy(dtype=object))
            out[k] = v
        dk = out["coord_dist_km"].to_numpy().copy()
        dk[idx] = c["coord_dist_km"].to_numpy()
        out["coord_dist_km"] = dk
    return out


def check_rows_mask(s: pd.DataFrame, route: np.ndarray, rr2: np.ndarray) -> np.ndarray:
    """구현 세부 (h): 부록 B-6 check 단계 행 = RR2·RR8 제외(RR1 뒤 src_route)."""
    d = s["d"].to_numpy().astype("datetime64[D]")
    return ~rr2 & ~rr8_mask(d, route)


# ---------------------------------------------------------------- RR1
@dataclass
class RR1Result:
    moved: np.ndarray            # 시료 행 위치 기준 bool
    sgg: np.ndarray              # 옮긴 행의 sgg_code(그 밖 None)
    coord_table: pd.DataFrame    # 좌표마다 점검 표(정답 없음)
    candidates: np.ndarray       # 좌표 나라 KOR 인 해외 행


def nearest_points(lat, lon, plat, plon) -> tuple[np.ndarray, np.ndarray]:
    """점마다 가장 가까운 기준점 위치·거리(km, 대권). 같으면 앞 위치."""
    idx = np.empty(len(lat), dtype=np.int64)
    dist = np.empty(len(lat))
    for i, (a, b) in enumerate(zip(lat, lon)):
        dk = haversine_km(a, b, plat, plon)
        j = int(np.argmin(dk))
        idx[i], dist[i] = j, float(dk[j])
    return idx, dist


def apply_rr1(s: pd.DataFrame, cc: pd.DataFrame, sgg02: ModuleType, region, mapping,
              krc: pd.DataFrame, dom_sgg: pd.DataFrame) -> RR1Result:
    """구현 세부 (a). s·cc 는 같은 행 순서(cc = 해외 행만 값, 국내 None 가능)."""
    ov = (s["src_route"] == "overseas").to_numpy()
    cand = ov & (cc["coord_cc"].to_numpy() == KOREA)
    pts = krc[(krc["molit_use_yn"] == "존재") & krc["level"].isin(["eupmyeondong", "ri"])
              & krc["lat"].notna() & krc["lon"].notna()].reset_index(drop=True)
    sub = s.loc[cand, ["lims_data_key", "addr", "lat", "lon"]].copy()
    la = sub["lat"].to_numpy(dtype=np.float64)
    lo = sub["lon"].to_numpy(dtype=np.float64)
    sgg = np.full(len(s), None, dtype=object)
    if len(sub):
        j, dk = nearest_points(la, lo, pts["lat"].to_numpy(np.float64), pts["lon"].to_numpy(np.float64))
        c10 = pts["legal_dong_code"].astype(str).to_numpy()[j]
        codes = sgg02.assign_sgg(list(c10), list(sub["addr"]), region=region, mapping=mapping)
        sgg[np.flatnonzero(cand)] = np.array(codes, dtype=object)
        sub["near_legal_dong_code"] = c10
        sub["near_legal_dong_nm"] = pts["legal_dong_nm"].to_numpy()[j]
        sub["near_legal_dong_km"] = dk
        sub["sgg_by_02_rule"] = codes
        # 점검용: 가장 가까운 국내 시료 좌표(시료 파일 lat·lon)의 sgg_code
        dsu = dom_sgg[["lat", "lon", "sgg_code"]].dropna(subset=["lat", "lon"])
        dsu = (dsu.assign(lat=dsu["lat"].astype(np.float64), lon=dsu["lon"].astype(np.float64))
                  .groupby(["lat", "lon"])["sgg_code"].agg(lambda v: ",".join(sorted(set(v)))).reset_index())
        jj, dd = nearest_points(la, lo, dsu["lat"].to_numpy(), dsu["lon"].to_numpy())
        sub["near_domestic_sample_sgg"] = dsu["sgg_code"].to_numpy()[jj]
        sub["near_domestic_sample_km"] = dd
    moved = cand.copy()
    moved[cand] = np.array([c is not None for c in sgg[cand]], dtype=bool)
    sub["moved"] = moved[cand] if len(sub) else []
    ccs = cc.loc[cand, ["coord_method", "coord_dist_km"]].reset_index(drop=True)
    sub = sub.reset_index(drop=True).join(ccs)
    if len(sub):
        grp = ["lat", "lon"]
        table = (sub.groupby(grp, sort=True)
                    .agg(n_rows=("lims_data_key", "size"), addr_example=("addr", "first"),
                         coord_method=("coord_method", "first"), coord_dist_km=("coord_dist_km", "first"),
                         near_legal_dong_code=("near_legal_dong_code", lambda v: ",".join(sorted(set(v)))),
                         near_legal_dong_nm=("near_legal_dong_nm", "first"),
                         near_legal_dong_km=("near_legal_dong_km", "first"),
                         sgg_by_02_rule=("sgg_by_02_rule", lambda v: ",".join(sorted({str(x) for x in v}))),
                         near_domestic_sample_sgg=("near_domestic_sample_sgg", "first"),
                         near_domestic_sample_km=("near_domestic_sample_km", "first"),
                         n_moved=("moved", "sum"))
                    .reset_index())
        table["sgg_agrees_with_near_domestic_sample"] = [
            a in set(str(b).split(",")) for a, b in zip(table["sgg_by_02_rule"], table["near_domestic_sample_sgg"])]
    else:
        table = pd.DataFrame(columns=["lat", "lon", "n_rows"])
    return RR1Result(moved=moved, sgg=sgg, coord_table=table, candidates=cand)


# ---------------------------------------------------------------- 시료 표
@dataclass
class Built:
    table: pd.DataFrame                 # 시료 표(남는 행)
    excluded: pd.DataFrame              # lims_data_key·rule
    rr_table: pd.DataFrame              # RR 셈·기록값
    rr4_rows: pd.DataFrame              # RR4 행 점검 표(정답 없음)
    nation_usage: pd.DataFrame          # 원천별 셈
    rr7_by_year: pd.DataFrame
    rr6_observed_codes: list = field(default_factory=list)


def _isnone(a) -> np.ndarray:
    """None·NaN 모두 결측."""
    return pd.isna(pd.Series(list(a), dtype=object)).to_numpy(dtype=bool)


def _na_to_none(a) -> np.ndarray:
    return np.array([None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in a], dtype=object)


def _rr_row(item, observed, plan, recorded_key=None, note=""):
    rec = RECORDED_RR.get(recorded_key) if recorded_key else None
    return {"item": item, "plan": plan, "recorded": rec, "observed": observed,
            "match": (rec == observed) if rec is not None else None, "note": note}


def build_samples(s: pd.DataFrame, cc: pd.DataFrame, rr1: RR1Result, rr2: np.ndarray, sgg_map: pd.Series,
                  nsrc: pd.DataFrame, sources: Sequence[str], nc: pd.DataFrame) -> Built:
    """RR1~RR8 적용·운영 단위 칼럼(§2-2·§2-3). s·cc·nsrc 는 시료 파일 행 순서."""
    n = len(s)
    a2_to_a3, a3_set = nation_maps(nc)
    ov0 = (s["src_route"] == "overseas").to_numpy()
    route = np.where(rr1.moved, "domestic", s["src_route"].to_numpy()).astype(object)
    ov = route == "overseas"

    # ---- 나라 코드(A-2): RR1 뒤 해외 행
    db_a3, db_src, usage = resolve_db_nation(nsrc, sources, a2_to_a3, a3_set)
    pre_rr4 = ov0 & _isnone(db_a3)          # RR1 전 해외 행 중 DB 로 못 정함
    rr4 = ov & _isnone(db_a3)                # RR1 뒤 남은 해외 행
    nation = np.where(ov, db_a3, None).astype(object)
    nsrc_name = np.where(ov, db_src, None).astype(object)
    idx4 = np.flatnonzero(rr4)
    addr_cc = address_countries(s["addr"].to_numpy()[idx4], nc) if len(idx4) else []
    for k, i in enumerate(idx4):
        c, m = cc["coord_cc"].iat[i], cc["coord_method"].iat[i]
        if c is not None and m in ("inside", "nearest"):
            nation[i] = c
            nsrc_name[i] = NATION_SRC_COORD_INSIDE if m == "inside" else NATION_SRC_COORD_NEAREST
        elif len(addr_cc[k]) == 1:
            nation[i] = addr_cc[k][0]
            nsrc_name[i] = NATION_SRC_ADDR
    rr4_unresolved = rr4 & _isnone(nation)
    rr4_rows = pd.DataFrame({
        "lims_data_key": s["lims_data_key"].to_numpy()[idx4], "addr": s["addr"].to_numpy()[idx4],
        "lat": s["lat"].to_numpy()[idx4], "lon": s["lon"].to_numpy()[idx4],
        **{f"raw_{c}": nsrc[c].to_numpy()[idx4] for c in NATION_SOURCES},
        "coord_cc": cc["coord_cc"].to_numpy()[idx4], "coord_method": cc["coord_method"].to_numpy()[idx4],
        "coord_dist_km": cc["coord_dist_km"].to_numpy()[idx4],
        "coord_cc_in_nation_code": [c in a3_set for c in cc["coord_cc"].to_numpy()[idx4]],
        "addr_countries": [",".join(v) for v in addr_cc],
        "nation_a3": nation[idx4], "nation_src": nsrc_name[idx4],
    })
    rr4_rows["addr_disagrees"] = [bool(a) and (c not in a.split(","))
                                  for a, c in zip(rr4_rows["addr_countries"], rr4_rows["nation_a3"])]

    # ---- RR5: 좌표 나라 ≠ 제조국(DB 원천 행) → 나라 = 제조국(이미 그렇게 정함, 표시만)
    from_db = ov & np.isin(nsrc_name.astype(str), list(NATION_SOURCES))
    coord_cc = cc["coord_cc"].to_numpy()
    rr5 = np.full(n, None, dtype=object)
    rr5[from_db] = np.array([c is not None and c != v for c, v in zip(coord_cc[from_db], nation[from_db])], dtype=bool)
    rr5_n = int(np.sum(rr5[from_db].astype(bool))) if from_db.any() else 0
    inside = cc["coord_method"].to_numpy() == "inside"
    rr5_inside_n = int(np.sum([bool(rr5[i]) for i in np.flatnonzero(from_db & inside)]))

    # ---- 단위 칼럼(§2-3)
    sgg = np.full(n, None, dtype=object)
    dom = ~ov
    dom_keys = s["lims_data_key"].to_numpy()
    sgg_dom = _na_to_none(pd.Series(dom_keys).map(sgg_map).to_numpy(dtype=object))
    sgg[dom & ~rr1.moved] = sgg_dom[dom & ~rr1.moved]
    sgg[rr1.moved] = rr1.sgg[rr1.moved]
    sgg[ov] = np.array([None if v is None else OV + v for v in nation[ov]], dtype=object)

    # ---- RR3
    p2_raw, p3_raw = s["train_prdlst_code"], s["train_prdlst_code_2"]
    p2_empty, p3_empty = p2_raw.isna().to_numpy(), p3_raw.isna().to_numpy()
    p2 = np.where(p2_empty, NO_CODE, p2_raw.astype(object).to_numpy()).astype(object)
    p3 = np.where(p3_empty, NO_CODE, p3_raw.astype(object).to_numpy()).astype(object)

    # ---- RR6·RR7
    rr6 = np.isin(p3, list(RR6_CODES))
    d = s["d"].to_numpy().astype("datetime64[D]")
    md = pd.Series(d).dt.strftime("%m-%d").to_numpy()
    rr7 = md == "07-01"

    # ---- 제외(RR2 → RR4 → RR8 순서, 행마다 첫 규칙)
    rule = np.full(n, None, dtype=object)
    rule[rr2] = "RR2"
    rule[_isnone(rule) & rr4_unresolved] = "RR4"
    alive = _isnone(rule)
    if (alive & dom & _isnone(sgg)).any():
        raise AssertionError("남는 국내 행에 sgg 없음(RR2·RR1 셈 확인)")
    rr8 = rr8_mask(d, route) & alive
    rule[rr8] = "RR8"
    keep = _isnone(rule)

    tab = pd.DataFrame({
        "lims_data_key": s["lims_data_key"].to_numpy(), "y": s["y"].to_numpy(np.int8),
        "src_route": route, "sgg": sgg, "sido": [None if v is None else (OV if v.startswith(OV) else v[:2]) for v in sgg],
        "p2": p2, "p3": p3, "d": d, "h": s["h"].to_numpy(np.int16),
        "nation_a3": nation, "nation_src": nsrc_name,
        "rr1_moved": rr1.moved, "rr3_p2_nocode": p2_empty, "rr3_p3_nocode": p3_empty,
        "rr5_coord_differs": rr5, "rr6_overseas_only_p3": rr6, "rr7_jul01": rr7,
    })
    tab = tab.loc[keep].reset_index(drop=True)
    check_sample_table(tab)
    excluded = pd.DataFrame({"lims_data_key": s["lims_data_key"].to_numpy()[~keep], "rule": rule[~keep]})

    # ---- 셈 표
    y = s["y"].to_numpy()
    rr1_coords = (s.loc[rr1.moved, ["lat", "lon"]].drop_duplicates().shape[0])
    obs_ov_p3 = sorted(set(p3[ov & keep]) - set(p3[~ov & keep]) - {NO_CODE})
    kor = ov & keep & (nation == KOREA)
    rr8_d = d[rr8]
    rows = [
        _rr_row("RR1 국내로 옮긴 행", int(rr1.moved.sum()), "§2-2 RR1", "RR1_rows"),
        _rr_row("RR1 좌표 곳 수", int(rr1_coords), "§2-2 RR1", "RR1_coords"),
        _rr_row("RR1 검출", int(y[rr1.moved].sum()), "§2-2 RR1", "RR1_pos"),
        _rr_row("RR1 후보(좌표 나라 KOR) 중 02 규칙 미배정 행", int((rr1.candidates & ~rr1.moved).sum()), "§2-2 RR1",
                note="옮기지 않고 해외로 둠"),
        _rr_row("RR2 제외 행", int(rr2.sum()), "§2-2 RR2", "RR2_rows"),
        _rr_row("RR2 검출", int(y[rr2].sum()), "§2-2 RR2", "RR2_pos"),
        _rr_row("RR3 2·3단계 함께 빈 행", int((p2_empty & p3_empty).sum()), "§2-2 RR3", "RR3_both_empty"),
        _rr_row("RR3 2단계만 빈 행", int((p2_empty & ~p3_empty).sum()), "§2-2 RR3", note="단계 0 셈"),
        _rr_row("RR3 3단계만 빈 행", int((~p2_empty & p3_empty).sum()), "§2-2 RR3", note="단계 0 셈"),
        _rr_row("RR4 DB 원천으로 나라 못 정한 해외 행(RR1 전)", int(pre_rr4.sum()), "§2-2 RR4", "RR4_rows"),
        _rr_row("RR4 DB 원천으로 나라 못 정한 해외 행(RR1 뒤)", int(rr4.sum()), "§2-2 RR4"),
        _rr_row("RR4 좌표(다각형 안)로 정함", int(np.sum(nsrc_name[rr4] == NATION_SRC_COORD_INSIDE)), "§2-2 RR4"),
        _rr_row("RR4 좌표(가장 가까운 나라)로 정함", int(np.sum(nsrc_name[rr4] == NATION_SRC_COORD_NEAREST)), "§2-2 RR4",
                note="구현 세부 (a)(c)"),
        _rr_row("RR4 주소로 정함", int(np.sum(nsrc_name[rr4] == NATION_SRC_ADDR)), "§2-2 RR4"),
        _rr_row("RR4 그래도 못 정해 제외", int(rr4_unresolved.sum()), "§2-2 RR4"),
        _rr_row("RR5 좌표 나라 ≠ 제조국(DB 원천 행)", rr5_n, "§2-2 RR5", "RR5_rows", note="구현 세부 (d)"),
        _rr_row("RR5 중 좌표가 다각형 안인 행", rr5_inside_n, "§2-2 RR5"),
        _rr_row("RR6 해외 전용 3단계 3종 행(시료 파일 전체)", int(rr6.sum()), "§2-2 RR6", "RR6_rows"),
        _rr_row("RR6 그중 남는 행", int((rr6 & keep).sum()), "§2-2 RR6"),
        _rr_row("RR6 남는 행에서 센 해외에만 있는 3단계 코드", ",".join(obs_ov_p3), "§2-2 RR6",
                note="기록: " + ",".join(RR6_CODES)),
        _rr_row("RR7 d = 07-01 행(남는 행)", int((rr7 & keep).sum()), "§2-2 RR7", note="점검용 셈만"),
        _rr_row("RR8 제외 행", int(rr8.sum()), "§2-2 RR8", "RR8_rows"),
        _rr_row("RR8 d 범위", f"{_ds(rr8_d.min())}~{_ds(rr8_d.max())}" if rr8_d.size else "", "§2-2 RR8",
                note="기록: " + "~".join(RR8_RECORDED_D)),
        _rr_row("OVKOR(RR1 뒤 남은 해외 행 중 나라 KOR)", int(kor.sum()), "§2-3", "OVKOR_rows", note="구현 세부 (e)"),
        _rr_row("OVKOR 중 mnf_nation_cd 원천", int(np.sum(kor & (nsrc_name == "mnf_nation_cd"))), "§2-3"),
    ]
    rr_table = pd.DataFrame(rows)
    rr7_year = (pd.DataFrame({"year": pd.Series(d).dt.year.to_numpy(), "rr7": rr7, "keep": keep})
                .query("rr7 and keep").groupby("year").size().rename("n_rows").reset_index())
    usage = usage.assign(sources_order=",".join(sources))
    return Built(table=tab, excluded=excluded, rr_table=rr_table, rr4_rows=rr4_rows, nation_usage=usage,
                 rr7_by_year=rr7_year, rr6_observed_codes=obs_ov_p3)


def check_sample_table(t: pd.DataFrame) -> None:
    """부록 B-6 일부: 키 중복 0, sgg 5자리(국내)·'OV'+3(해외), 품목 코드 문자형, h 1~60."""
    if list(t.columns) != list(SAMPLE_COLUMNS):
        raise AssertionError("시료 표 칼럼이 계획 목록과 다름")
    if t["lims_data_key"].duplicated().any():
        raise AssertionError("키 중복")
    dom = t["src_route"] == "domestic"
    sg = t["sgg"]
    if sg.isna().any():
        raise AssertionError("sgg 결측")
    if not sg[dom].map(lambda v: isinstance(v, str) and len(v) == 5 and v.isdigit()).all():
        raise AssertionError("국내 sgg 가 5자리 숫자 문자가 아님")
    if not sg[~dom].map(lambda v: isinstance(v, str) and len(v) == 5 and v.startswith(OV)).all():
        raise AssertionError("해외 sgg 가 'OV' + 3글자가 아님")
    if not (t.loc[dom, "nation_a3"].isna().all() and t.loc[~dom, "nation_a3"].notna().all()):
        raise AssertionError("나라 코드 칸 이상")
    for c in ("p2", "p3"):
        if not t[c].map(lambda v: isinstance(v, str)).all():
            raise AssertionError(f"{c} 문자형 아님")
    if t["h"].min() < 1 or t["h"].max() > 60:
        raise AssertionError("h 범위")


# ---------------------------------------------------------------- 적용 후 수·덜 찬 달·0-2
def post_counts(t: pd.DataFrame) -> pd.DataFrame:
    ov = t["src_route"] == "overseas"
    tot = pd.DataFrame([
        {"split": "all", "item": "n_rows", "observed": len(t)},
        {"split": "all", "item": "n_domestic", "observed": int((~ov).sum())},
        {"split": "all", "item": "n_overseas", "observed": int(ov.sum())},
        {"split": "all", "item": "pos", "observed": int(t["y"].sum())},
        {"split": "all", "item": "pos_domestic", "observed": int(t.loc[~ov, "y"].sum())},
        {"split": "all", "item": "pos_overseas", "observed": int(t.loc[ov, "y"].sum())},
    ])
    return pd.concat([tot, split_count_table(t["d"], t["y"], t["src_route"], None)], ignore_index=True)


def month_counts(t: pd.DataFrame) -> pd.DataFrame:
    """(src_route, d 의 달)별 시료 수 — 정답 안 씀."""
    ym = pd.Series(t["d"].to_numpy().astype("datetime64[D]")).dt.strftime("%Y-%m")
    return (pd.DataFrame({"src_route": t["src_route"].to_numpy(), "ym": ym.to_numpy()})
            .groupby(["src_route", "ym"]).size().rename("n").reset_index())


def incomplete_months(t: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
    tab = fd.incomplete_month_table(month_counts(t))
    tab["prior_counts"] = tab["prior_counts"].map(lambda v: ",".join(map(str, v)))
    bad = [[r, m] for r, m, inc in tab[["src_route", "ym", "incomplete"]].itertuples(index=False) if inc]
    return tab, bad


def nation_table(t: pd.DataFrame, nc: pd.DataFrame, cut: str = B6_JUDGE_CUT) -> tuple[pd.DataFrame, dict]:
    """0-2: 나라별 시료·검출(d ≤ 2021-06-30 만), 제조국 수."""
    ov = t[t["src_route"] == "overseas"]
    cutd = np.datetime64(cut, "D")
    sub = ov[ov["d"].to_numpy().astype("datetime64[D]") <= cutd]
    tab = (sub.groupby("nation_a3").agg(n_samples=("y", "size"), n_pos=("y", "sum"),
                                         nation_src=("nation_src", lambda v: ",".join(sorted(set(v)))))
              .reset_index())
    names = nc.set_index("alpha_3")[["name_en", "name_ko"]]
    tab = tab.join(names, on="nation_a3").sort_values(["n_samples", "nation_a3"], ascending=[False, True])
    tab.insert(1, "d_range", f"<= {cut}")
    info = {"n_nations_all_d": int(ov["nation_a3"].nunique()), f"n_nations_d_le_{cut}": int(sub["nation_a3"].nunique()),
            "nations_not_in_nation_code": sorted(set(ov["nation_a3"]) - set(nc["alpha_3"]))}
    return tab.reset_index(drop=True), info


# ---------------------------------------------------------------- 부록 B-6
def column_check(x, y) -> dict:
    """빈 칸 비율(검출·불검출), 고유값, '검출 행에만 나오는 값' 비율 2종, U(y | x)."""
    yv = np.asarray(y, dtype=np.int64)
    blank = pd.isna(pd.Series(x)).to_numpy()
    npos, nneg = int(yv.sum()), int((1 - yv).sum())
    codes = codes_with_nan(x)
    nu = int(codes.max()) + 1 if codes.size else 0
    pos_c = np.bincount(codes, weights=yv, minlength=nu)
    neg_c = np.bincount(codes, weights=1 - yv, minlength=nu)
    pos_only = (pos_c > 0) & (neg_c == 0)
    br_pos = float(blank[yv == 1].mean()) if npos else np.nan
    br_neg = float(blank[yv == 0].mean()) if nneg else np.nan
    return {
        "n_rows": int(yv.size), "n_pos": npos, "n_neg": nneg,
        "blank_rate_pos": br_pos, "blank_rate_neg": br_neg, "blank_gap": br_pos - br_neg,
        "n_unique": nu, "unique_ratio": nu / yv.size if yv.size else np.nan,
        "pos_only_values": int(pos_only.sum()),
        "pos_only_value_share": float(pos_only.sum() / nu) if nu else np.nan,
        "pos_only_row_share_in_pos": float(pos_c[pos_only].sum() / npos) if npos else np.nan,
        "theils_u": theils_u(yv, codes) if yv.size else np.nan,
    }


def blank_check(x, y) -> dict:
    yv = np.asarray(y, dtype=np.int64)
    blank = pd.isna(pd.Series(x)).to_numpy()
    npos, nneg = int(yv.sum()), int((1 - yv).sum())
    br_pos = float(blank[yv == 1].mean()) if npos else np.nan
    br_neg = float(blank[yv == 0].mean()) if nneg else np.nan
    return {"n_rows": int(yv.size), "n_pos": npos, "n_neg": nneg,
            "blank_rate_pos": br_pos, "blank_rate_neg": br_neg, "blank_gap": br_pos - br_neg}


def b6_tables(s: pd.DataFrame, route: np.ndarray, rows_mask: np.ndarray, nsrc: pd.DataFrame,
              a2_to_a3: dict, a3_set: set, cut: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """d ≤ cut 행으로 (Theil 후보 칼럼 표, 나라 코드 원천 표)."""
    d = s["d"].to_numpy().astype("datetime64[D]")
    m = rows_mask & (d <= np.datetime64(cut, "D"))
    y = s["y"].to_numpy()[m]
    cols = []
    for c in THEIL_CANDIDATES:
        x = route[m] if c == "src_route" else s[c].to_numpy()[m]
        cols.append({"column": c, "d_range": f"<= {cut}", **column_check(x, y)})
    ov = m & (route == "overseas")
    yo = s["y"].to_numpy()[ov]
    srcs = []
    for c in NATION_SOURCES:
        raw = nsrc[c][ov]
        r = blank_check(raw.to_numpy(), yo)
        mapped = map_alpha(raw.reset_index(drop=True), a2_to_a3, a3_set).to_numpy()
        unm = raw.notna().to_numpy() & np.array([v is None for v in mapped])
        r["unmappable_rate_pos"] = float(unm[yo == 1].mean()) if r["n_pos"] else np.nan
        r["unmappable_rate_neg"] = float(unm[yo == 0].mean()) if r["n_neg"] else np.nan
        srcs.append({"source": c, "d_range": f"<= {cut}", **r})
    for c, x in (("lat·lon(남은 행 좌표)", s["lat"].to_numpy()[ov]), ("addr(남은 행 주소)", s["addr"].to_numpy()[ov])):
        srcs.append({"source": c, "d_range": f"<= {cut}", **blank_check(x, yo)})
    return pd.DataFrame(cols), pd.DataFrame(srcs)


def gap_increased_names(t21: pd.DataFrame, t25: pd.DataFrame, key: str) -> list[str]:
    """부록 B-6: d ≤ 2021-06-30 때보다 검출·불검출 빈 칸 비율 차이(절댓값)가 커진 이름(수치 없음)."""
    a = t21.set_index(key)["blank_gap"].abs()
    b = t25.set_index(key)["blank_gap"].abs()
    both = a.index.intersection(b.index)
    return [k for k in t21[key] if k in both and np.isfinite(a[k]) and np.isfinite(b[k]) and b[k] > a[k]]
