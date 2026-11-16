"""고정표·조회형 파생변수 — 계획서 §5 표(FD1·WX4 품목 4·SO1·X1·X2·L1·WX7), §5-2, 부록 A-3·A-4, 부록 B-6.

- FD1 식품 고정표: `리스테리아ML2/data/데이터셋/build_food_db_table.py` 규칙 그대로(원문 → 숫자 parse 함수 복사,
  자기값 → 하위 코드 중앙값 → 2단계 코드 값 → 결측), 코드 목록만 국내 3단계 + 해외 전용 3종(§5-2).
- WX4 품목 4칼럼: 같은 스크립트(= make_ranges.py 3))와 같은 계산을 새 고정표 값으로(부록 A-3).
- L1 축사: 농장 수 ÷ 면적(km²) × 10, 국내 33 시군구·해외 나라, 자료 없는 곳 NaN(부록 A-4).
- SO1 토양: 국내 = 시군구_일별_토양.parquet 의 date = T 행, 해외 = global_soil_ph.ph_mean(om·vldpha NaN)(§5 표).
- X1·X2: y* = '사용가능일 ≤ T−1' 인 가장 큰 자료연도(전국 공통, 연도 사용가능일 = 그 연도 행 중 가장 늦은 값),
  [y*−4, y*] 중 그 단위에 값 있는 해의 비율 평균(없으면 NaN, 0 채움 없음)(§5 표 아래).
- WX7 season: 달 → 봄 3~5·여름 6~8·가을 9~11·겨울 12~2(남반구 보정 없음).

조회 함수는 (단위, 날짜) 배열을 받아 칼럼을 돌려줌 — 날짜는 일수(int, 1970-01-01 기준) 또는 날짜형.
단위 = 시료 표 `sgg`(국내 5자리 시군구 코드, 해외 'OV' + alpha-3).
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .folds import day_ints

OV = "OV"
NO_CODE = "코드없음"  # §2-2 RR3 — 고정표에 없음 → FD1 결측

# ======================================================================== FD1 원문 → 숫자 (build_food_db_table.py 그대로 복사)
FOOD_COLS = ("aw", "ph", "na")
FOOD_LABELS = ("단일값", "중앙", "이상", "이하", "초과", "미만")
RULES = ("자기값", "하위중앙", "2단계값", "결측")
NUM = r"-?\d+(?:\.\d+)?"
RANGE_SEP = r"[~～∼\-–]"
SIGN_LABEL = {"<": "미만", ">": "초과", "≤": "이하", "≥": "이상", "<=": "이하", ">=": "이상"}
SIGN_FLIP = {"미만": "초과", "초과": "미만", "이하": "이상", "이상": "이하"}
NUM_ROUND = 12


def parse_food(s):
    """원문 → (숫자, 적용 규칙, 값 표기). 값 표기: 단일값 · 중앙 · 이상 · 이하 · 초과 · 미만. 값이 없으면 숫자 nan, 표기 ''"""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return math.nan, "빈칸", ""
    if isinstance(s, (int, float, np.integer, np.floating)) and not isinstance(s, bool):
        return float(s), "숫자 그대로", "단일값"
    t = str(s).strip()
    if t == "":
        return math.nan, "빈칸", ""
    if re.fullmatch(NUM, t):
        return float(t), "숫자 그대로", "단일값"
    m = re.fullmatch(rf"({NUM})\s*{RANGE_SEP}\s*({NUM})", t)
    if m:
        return (float(m[1]) + float(m[2])) / 2, "범위 → 가운데 값", "중앙"
    m = re.fullmatch(rf"([<>≤≥]=?)\s*({NUM})", t)
    if m:
        return float(m[2]), "부등호 → 경계값", SIGN_LABEL[m[1]]
    m = re.fullmatch(rf"({NUM})\s*([<>≤≥]=?)", t)
    if m:
        return float(m[1]), "부등호 → 경계값", SIGN_FLIP[SIGN_LABEL[m[2]]]
    m = re.fullmatch(rf"({NUM})\s*(이상|이하|초과|미만)", t)
    if m:
        return float(m[1]), "부등호 → 경계값", m[2]
    return math.nan, "해석 못 함 → 결측", ""


def parse_food_n(s):
    x, rule, label = parse_food(s)
    if math.isnan(x):
        return x, rule, label
    y = round(x, NUM_ROUND)
    assert y == x or rule == "범위 → 가운데 값", (s, x, y)
    assert abs(y - x) < 1e-12
    return y, rule, label


def parse_db(s):
    """parse_food_n + 음수 원문은 해석 못 함 → 결측(부록 A-4 '음수 원문(예: '-0.85') = 결측')"""
    x, rule, label = parse_food_n(s)
    if not math.isnan(x) and x < 0:
        return math.nan, "해석 못 함 → 결측", ""
    return x, rule, label


# ======================================================================== FD1 고정표
def descendants(children: Mapping[str, list], code: str) -> list[str]:
    """hrnk_prdlst_cd 를 따라 code 아래의 모든 코드(자기 제외, lv 무관, 유효 종료 포함). 순환이면 멈춤."""
    out, stack, seen = [], list(children.get(code, [])), {code}
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        stack.extend(children.get(c, []))
    return sorted(out)


def _null(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pair_p2(p3: Sequence, p2: Sequence) -> dict[str, str]:
    """3단계 → '2단계값' 규칙에 쓸 2단계 코드: 시료에서 짝지어진 2단계 중 시료 수 최다(같으면 코드 순 앞)
    (build_food_db_table.py 의 K2_OF 그대로)."""
    cnt = Counter((a, b) for a, b in zip(p3, p2) if not _null(a))
    by3 = defaultdict(list)
    for (a, b), n in cnt.items():
        by3[a].append((b, n))
    return {a: sorted(v, key=lambda t: (-t[1], t[0] or ""))[0][0] for a, v in by3.items()}


@dataclass
class FoodTable:
    table: pd.DataFrame          # 칼럼 FOOD_TABLE_COLUMNS
    parse_rules: dict            # 항목 → 규칙별 행 수
    unparsed: dict               # 항목 → 해석 못 한 원문 수
    n_aw_rows: int
    n_non_original: int


FOOD_TABLE_COLUMNS = (
    "p3", "p2_used",
    "food_aw", "aw_label", "food_ph", "ph_label", "food_na", "na_label",
    "aw_rule", "aw_n_sub", "ph_rule", "ph_n_sub", "na_rule", "na_n_sub",
    "rg_aw_lim", "rg_aw_opt_dist", "rg_ph_lim", "rg_ph_opt_dist",
)
FD1_COLUMNS = ("food_aw", "food_ph", "food_na", "aw_label")           # §5 표 FD1
RG_ITEM_COLUMNS = ("rg_aw_lim", "rg_aw_opt_dist", "rg_ph_lim", "rg_ph_opt_dist")  # 부록 A-3 WX4 품목 4
AW_OPT, AW_LIM, PH_OPT, PH_LIM = 0.97, (0.90, 0.997), 7.0, (4.0, 9.6)   # 부록 A-3


def in_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """경계 포함, 입력 결측이면 결측."""
    out = ((x >= lo) & (x <= hi)).astype(np.float64)
    out[np.isnan(x)] = np.nan
    return out


def rg_item(aw: np.ndarray, ph: np.ndarray) -> dict[str, np.ndarray]:
    """부록 A-3 품목 4칼럼(make_ranges.py 3) 과 같은 계산, 저장은 float32 — build_food_db_table.py 와 같음)."""
    aw = np.asarray(aw, dtype=np.float64)
    ph = np.asarray(ph, dtype=np.float64)
    out = {
        "rg_aw_lim": in_range(aw, *AW_LIM),
        "rg_aw_opt_dist": np.abs(aw - AW_OPT),
        "rg_ph_lim": in_range(ph, *PH_LIM),
        "rg_ph_opt_dist": np.abs(ph - PH_OPT),
    }
    return {k: v.astype(np.float32) for k, v in out.items()}


def build_food_table(cl: pd.DataFrame, aw_cl: pd.DataFrame, codes3: Sequence[str],
                     p2_of: Mapping[str, str]) -> FoodTable:
    """cl = ba_data.t_prdlst_cl(prdlst_cd·hrnk_prdlst_cd), aw_cl = ba_data.t_aw_cl(prdlst_cd·aw·ph·na·aw_source),
    codes3 = 고정표 3단계 코드 목록, p2_of = pair_p2 결과(codes3 전부 있어야 함)."""
    codes_cl = cl["prdlst_cd"].tolist()
    if any(_null(c) for c in codes_cl) or len(set(codes_cl)) != len(codes_cl):
        raise ValueError("t_prdlst_cl prdlst_cd 결측·중복")
    children = defaultdict(list)
    for code, up in zip(codes_cl, cl["hrnk_prdlst_cd"].tolist()):
        if not _null(up) and up != code:
            children[up].append(code)
    in_cl = set(codes_cl)
    codes_aw = aw_cl["prdlst_cd"].tolist()
    if any(_null(c) for c in codes_aw) or len(set(codes_aw)) != len(codes_aw):
        raise ValueError("t_aw_cl prdlst_cd 결측·중복")
    codes3 = sorted(codes3)
    if len(set(codes3)) != len(codes3) or NO_CODE in codes3:
        raise ValueError("3단계 코드 목록 중복 또는 '코드없음' 포함")
    missing_cl = [k for k in codes3 if k not in in_cl]
    if missing_cl:
        raise ValueError(f"분류표에 없는 3단계 코드: {missing_cl}")
    missing_p2 = [k for k in codes3 if _null(p2_of.get(k))]
    if missing_p2:
        raise ValueError(f"2단계 짝 없는 3단계 코드: {missing_p2}")

    val = {c: {} for c in FOOD_COLS}
    parse_rules = {c: Counter() for c in FOOD_COLS}
    unparsed = {c: Counter() for c in FOOD_COLS}
    n_non_orig = 0
    for row in aw_cl[["prdlst_cd", "aw", "ph", "na", "aw_source"]].itertuples(index=False):
        code = row.prdlst_cd
        n_non_orig += row.aw_source != "original"
        for c in FOOD_COLS:
            raw = getattr(row, c)
            x, rule, label = parse_db(raw)
            if c == "aw" and row.aw_source != "original":
                parse_rules[c]["(aw_source ≠ original → 쓰지 않음)"] += 1
                continue
            parse_rules[c][rule] += 1
            if rule == "해석 못 함 → 결측":
                unparsed[c][str(raw).strip()] += 1
            if not math.isnan(x):
                assert label in FOOD_LABELS
                val[c][code] = (x, label)

    rows = {k: {"p3": k, "p2_used": p2_of[k]} for k in codes3}
    for c in FOOD_COLS:
        for k3 in codes3:
            r = rows[k3]
            sub = [d for d in descendants(children, k3) if d in val[c]]
            if k3 in val[c]:
                x, lab = val[c][k3]
                rule, n_sub = "자기값", None
            elif sub:
                x = round(float(np.median(np.array([val[c][d][0] for d in sub], dtype=np.float64))), NUM_ROUND)
                lab, rule, n_sub = "하위중앙", "하위중앙", len(sub)
            elif p2_of[k3] in val[c]:
                x, lab = val[c][p2_of[k3]]
                rule, n_sub = "2단계값", None
            else:
                x, lab, rule, n_sub = np.nan, None, "결측", None
            r[f"food_{c}"] = x
            r[f"{c}_label"] = lab
            r[f"{c}_rule"] = rule
            r[f"{c}_n_sub"] = n_sub
    t = pd.DataFrame([rows[k] for k in codes3])
    for c in FOOD_COLS:
        t[f"food_{c}"] = t[f"food_{c}"].astype(np.float64)
        t[f"{c}_n_sub"] = t[f"{c}_n_sub"].astype("Int32")
        t[f"{c}_label"] = t[f"{c}_label"].astype(object).where(t[f"{c}_label"].notna(), None)
    for k, v in rg_item(t["food_aw"].to_numpy(), t["food_ph"].to_numpy()).items():
        t[k] = v
    t = t[list(FOOD_TABLE_COLUMNS)]
    return FoodTable(t, {c: dict(v) for c, v in parse_rules.items()}, {c: dict(v) for c, v in unparsed.items()},
                     len(aw_cl), int(n_non_orig))


def food_features(table: pd.DataFrame, p3: Sequence) -> pd.DataFrame:
    """3단계 코드 배열 → FD1 4칼럼 + 품목 rg 4칼럼. 표에 없는 코드('코드없음' 포함)는 결측(aw_label = None —
    모델 입력 때 '결측' 범주, 부록 A-4)."""
    if table["p3"].duplicated().any():
        raise ValueError("고정표 p3 중복")
    idx = pd.Index(table["p3"]).get_indexer(pd.Index(np.asarray(p3, dtype=object)))
    ok = idx >= 0
    out = {}
    for c in FD1_COLUMNS + RG_ITEM_COLUMNS:
        src = table[c].to_numpy()
        if c == "aw_label":
            v = np.full(len(idx), None, dtype=object)
            v[ok] = src[idx[ok]]
        else:
            v = np.full(len(idx), np.nan, dtype=src.dtype if src.dtype.kind == "f" else np.float64)
            v[ok] = src[idx[ok]]
        out[c] = v
    return pd.DataFrame(out)


# ======================================================================== L1 축사
def farm_unit_name(sido: str, sgg: str) -> str:
    """부록 A-4: '시도 + 공백 + 시군구' — 시도·시군구 이름이 같은(세종) 경우 시도 이름 하나."""
    sido, sgg = str(sido).strip(), str(sgg).strip()
    return sido if sido == sgg else f"{sido} {sgg}"


def match_domestic_farms(farm: pd.DataFrame, sgg_names: pd.DataFrame) -> pd.DataFrame:
    """farm = 엑셀 시트(시도·시군구·농장수 …), sgg_names = 시군구_대표좌표(sgg_code·sgg_nm).
    이름 그대로 맞춤(부록 A-4) — 못 맞추면 멈춤(담당자 보고 대상)."""
    nm = sgg_names[["sgg_code", "sgg_nm"]]
    if nm["sgg_nm"].duplicated().any() or nm["sgg_code"].duplicated().any():
        raise ValueError("시군구_대표좌표 이름·코드 중복")
    look = dict(zip(nm["sgg_nm"], nm["sgg_code"]))
    out = farm.copy()
    out["unit_name"] = [farm_unit_name(a, b) for a, b in zip(farm["시도"], farm["시군구"])]
    out["sgg_code"] = out["unit_name"].map(look)
    bad = out.loc[out["sgg_code"].isna(), "unit_name"].tolist()
    if bad:
        raise ValueError(f"축사 엑셀 이름을 sgg_nm 과 못 맞춤(담당자 보고): {bad}")
    if out["sgg_code"].duplicated().any():
        raise ValueError("축사 엑셀 두 줄이 같은 sgg_code")
    return out


MOLIT_TOTAL_AREA_COL = "32"   # 지적통계 formId 2300: '계' 아래 '면적'(㎡) — 칼럼 정의 JSON 으로 확인


def molit_area_table(columns_json: bytes | str, data_json: bytes | str, year: int) -> pd.DataFrame:
    """국토교통부 지적통계(시군구) JSON → (sido_abbr, sgg_name, area_m2). '합계'·'(계)'·'전국' 줄은 뺌."""
    cols = json.loads(columns_json)["data"]
    byid = {str(c["DATA_DIV_ID"]): c for c in cols}
    c32 = byid.get(MOLIT_TOTAL_AREA_COL)
    up = byid.get(str(c32["UP_DATA_DIV_ID"])) if c32 else None
    if not (c32 and c32["DATA_DIV_NM"] == "면적" and up and up["DATA_DIV_NM"] == "계"
            and byid["1"]["DATA_DIV_NM"] == "시도" and byid["2"]["DATA_DIV_NM"] == "시군구"):
        raise ValueError("지적통계 칼럼 정의가 예상(1 시도·2 시군구·32 = 계/면적)과 다름")
    rows = []
    for r in json.loads(data_json)["data"]:
        if str(r["0"]) != str(year):
            raise ValueError(f"기준 연도 다른 줄: {r['0']}")
        sido, sgg = r["1"], r["2"]
        if sido == "전국" or sgg == "합계" or sgg.endswith("(계)"):
            continue
        rows.append((sido, sgg, float(r[MOLIT_TOTAL_AREA_COL])))
    t = pd.DataFrame(rows, columns=["sido_abbr", "sgg_name", "area_m2"])
    if t.duplicated(["sido_abbr", "sgg_name"]).any():
        raise ValueError("지적통계 (시도, 시군구) 중복")
    return t


def match_domestic_area(units: pd.DataFrame, molit: pd.DataFrame, sido_abbr: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """units = (sgg_code, sgg_nm). 지적통계 줄을 '우리 시도 이름 → 지적통계 시도 약칭(설정)' + 시군구 이름(공백 뺌)으로 맞춤.
    sgg_nm 이 한 낱말(세종)이면 시도 = 시군구 = 그 이름. 돌려줌: sgg_code·area_km2(못 맞추면 NaN)·molit_name."""
    key = defaultdict(list)
    for i, r in molit.iterrows():
        key[(r["sido_abbr"], r["sgg_name"].replace(" ", ""))].append(i)
    out = []
    for code, nm in zip(units["sgg_code"], units["sgg_nm"]):
        parts = str(nm).split()
        sido, sgg = (parts[0], parts[0]) if len(parts) == 1 else (parts[0], "".join(parts[1:]))
        if sido not in sido_abbr:
            raise ValueError(f"설정 molit_sido_abbr 에 없는 시도: {sido}")
        hits = [i for a in sido_abbr[sido] for i in key.get((a, sgg), [])]
        if len(hits) > 1:
            raise ValueError(f"지적통계 줄 여럿과 맞음: {nm}")
        if hits:
            r = molit.loc[hits[0]]
            out.append((code, r["area_m2"] / 1e6, f"{r['sido_abbr']} {r['sgg_name']}"))
        else:
            out.append((code, np.nan, None))
    return pd.DataFrame(out, columns=["sgg_code", "area_km2", "area_src_name"])


def fao_land_area(zip_bytes: bytes, m49_to_a3: Mapping[str, str], item_code: str, element_code: str,
                  unit: str, year: int) -> pd.DataFrame:
    """FAOSTAT Land Use 일괄 zip → (alpha_3, area_km2, fao_area, fao_m49, flag). 1000 ha × 10 = km².
    M49 → alpha-3 = DB nation_code 표(nation_code 칸 = M49 숫자). 같은 alpha-3 두 줄이면 멈춤."""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = [n for n in z.namelist() if n.endswith("_All_Data_(Normalized).csv")]
    if len(names) != 1:
        raise ValueError(f"zip 안 자료 파일: {names}")
    rows = []
    with z.open(names[0]) as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")):
            if r["Item Code"] != item_code or r["Element Code"] != element_code or r["Year"] != str(year):
                continue
            if r["Unit"] != unit:
                raise ValueError(f"단위 다름: {r['Unit']}")
            m49 = r["Area Code (M49)"].lstrip("'")
            a3 = m49_to_a3.get(m49.zfill(3))
            if a3 is None or r["Value"] == "":
                continue
            rows.append((a3, float(r["Value"]) * 10.0, r["Area"], m49, r["Flag"]))
    t = pd.DataFrame(rows, columns=["alpha_3", "area_km2", "fao_area", "fao_m49", "flag"])
    if t["alpha_3"].duplicated().any():
        raise ValueError(f"FAOSTAT alpha-3 중복: {t.loc[t['alpha_3'].duplicated(), 'alpha_3'].tolist()}")
    return t


def m49_map(nation_code: pd.DataFrame) -> dict[str, str]:
    m = {}
    for c, a in zip(nation_code["nation_code"], nation_code["alpha_3"]):
        if _null(c) or _null(a) or str(c).strip() == "":
            continue
        k = str(c).strip().zfill(3)
        if k in m and m[k] != a:
            raise ValueError(f"M49 {k} 가 alpha-3 둘: {m[k]}, {a}")
        m[k] = a
    return m


LIVESTOCK_COLUMNS = ("unit", "src_route", "unit_name", "farms", "area_km2", "area_source", "area_base_year",
                     "area_src_name", "L1", "nan_reason")


def l1_value(farms, area_km2):
    """부록 A-4: 농장 수 ÷ 면적(km²) × 10."""
    return np.asarray(farms, dtype=np.float64) / np.asarray(area_km2, dtype=np.float64) * 10.0


def livestock_lookup(table: pd.DataFrame, units: Sequence) -> np.ndarray:
    """단위 배열 → L1. 표에 없는 단위 = NaN(0 아님, 부록 A-4)."""
    if table["unit"].duplicated().any():
        raise ValueError("축사 표 unit 중복")
    idx = pd.Index(table["unit"]).get_indexer(pd.Index(np.asarray(units, dtype=object)))
    v = np.full(len(idx), np.nan)
    ok = idx >= 0
    v[ok] = table["L1"].to_numpy(dtype=np.float64)[idx[ok]]
    return v


# ======================================================================== SO1 토양
SO1_COLUMNS = ("soil_ph", "soil_om", "soil_vldpha")


class SoilLookup:
    """국내 = 시군구_일별_토양.parquet 의 (sgg_code, date = T) 행, 해외('OV' + alpha-3) = global_soil_ph.ph_mean,
    om·vldpha NaN. 표에 없는 시군구·날짜·나라 = NaN."""

    def __init__(self, soil: pd.DataFrame, global_ph: pd.DataFrame):
        if soil.duplicated(["sgg_code", "date"]).any():
            raise ValueError("토양 표 (sgg_code, date) 중복")
        days = day_ints(soil["date"])
        self.d0, self.d1 = int(days.min()), int(days.max())
        self.units = sorted(soil["sgg_code"].unique().tolist())
        nd = self.d1 - self.d0 + 1
        if len(soil) != nd * len(self.units):
            raise ValueError("토양 표가 시군구 × 이어진 날짜 꽉 찬 꼴이 아님")
        ui = pd.Index(self.units).get_indexer(soil["sgg_code"])
        di = days - self.d0
        self.cube = {}
        for c in SO1_COLUMNS:
            a = np.full((len(self.units), nd), np.nan, dtype=np.float64)
            a[ui, di] = soil[c].to_numpy(dtype=np.float64, na_value=np.nan)
            self.cube[c] = a
        if global_ph["iso3"].duplicated().any():
            raise ValueError("global_soil_ph iso3 중복")
        self.gph = dict(zip(global_ph["iso3"], global_ph["ph_mean"].astype(float)))

    def __call__(self, units: Sequence, T) -> pd.DataFrame:
        units = np.asarray(units, dtype=object)
        t = day_ints(T)
        if len(t) != len(units):
            raise ValueError("단위·날짜 길이 다름")
        n = len(units)
        out = {c: np.full(n, np.nan) for c in SO1_COLUMNS}
        ov = np.array([isinstance(u, str) and u.startswith(OV) for u in units], dtype=bool)
        ui = pd.Index(self.units).get_indexer(pd.Index(units))
        dom = (~ov) & (ui >= 0) & (t >= self.d0) & (t <= self.d1)
        for c in SO1_COLUMNS:
            out[c][dom] = self.cube[c][ui[dom], t[dom] - self.d0]
        out["soil_ph"][ov] = [self.gph.get(u[len(OV):], np.nan) for u in units[ov]]
        return pd.DataFrame(out)


def soil_direct(exam_days: np.ndarray, values: np.ndarray, T: int, window: int = 1095) -> tuple[float, int]:
    """부록 B-6 점검용 직접 계산: 검정일 ∈ [T−window, T−1] 인 값(결측 뺌) 중앙값·건수(§5 표 SO1)."""
    m = (exam_days >= T - window) & (exam_days <= T - 1) & ~np.isnan(values)
    n = int(m.sum())
    return (float(np.median(values[m])) if n else math.nan), n


# ======================================================================== X1·X2
@dataclass
class YearIndicator:
    """단위 × 자료연도 비율 표 + 연도별 사용가능일(전국 공통)."""
    name: str
    units: list
    years: np.ndarray            # 정렬된 자료연도
    avail: np.ndarray            # 연도별 사용가능일(일수) = 그 연도 행 중 가장 늦은 값
    values: np.ndarray           # (단위, 연도) 비율, 없으면 NaN
    span: int = 5                # [y* − 4, y*]

    @classmethod
    def from_long(cls, name: str, unit: Sequence, year: Sequence, value: Sequence, avail_day: Sequence,
                  span: int = 5) -> "YearIndicator":
        df = pd.DataFrame({"u": list(unit), "y": np.asarray(year, dtype=np.int64),
                           "v": np.asarray(value, dtype=np.float64), "a": day_ints(pd.Series(list(avail_day)))})
        if df.duplicated(["u", "y"]).any():
            raise ValueError(f"{name}: (단위, 자료연도) 중복")
        years = np.sort(df["y"].unique())
        avail = df.groupby("y")["a"].max().reindex(years).to_numpy()
        units = sorted(df["u"].unique().tolist())
        vals = np.full((len(units), len(years)), np.nan)
        vals[pd.Index(units).get_indexer(df["u"]), np.searchsorted(years, df["y"].to_numpy())] = df["v"].to_numpy()
        return cls(name, units, years, avail, vals, span)

    def y_star(self, T) -> np.ndarray:
        """y* = 사용가능일 ≤ T−1 인 가장 큰 자료연도(없으면 −1)."""
        t = day_ints(T)
        ok = self.avail[None, :] <= (t[:, None] - 1)
        yy = np.where(ok, self.years[None, :], -1)
        return yy.max(axis=1)

    def __call__(self, units: Sequence, T) -> np.ndarray:
        units = np.asarray(units, dtype=object)
        t = day_ints(T)
        if len(t) != len(units):
            raise ValueError("단위·날짜 길이 다름")
        ys = self.y_star(t)
        ui = pd.Index(self.units).get_indexer(pd.Index(units))
        out = np.full(len(units), np.nan)
        # y* 마다 단위별 평균을 한 번 계산
        for y in np.unique(ys):
            if y < 0:
                continue
            cols = (self.years >= y - (self.span - 1)) & (self.years <= y)
            block = self.values[:, cols]
            cnt = (~np.isnan(block)).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnt > 0, np.nansum(block, axis=1) / np.maximum(cnt, 1), np.nan)
            sel = (ys == y) & (ui >= 0)
            out[sel] = mean[ui[sel]]
        return out


def eco_indicator(df: pd.DataFrame) -> tuple[YearIndicator, dict]:
    """X1: (유기 + 무농약 면적) ÷ 경지면적, 사용가능일 = 자료연도 + 2년 1월 1일(§5 표) — 파일 '사용가능일' 과 같은지 점검."""
    avail = pd.to_datetime((df["자료연도"].astype(int) + 2).astype(str) + "-01-01")
    file_avail = pd.to_datetime(df["사용가능일"])
    n_avail_diff = int((avail != file_avail).sum())
    area = df["경지면적"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (df["유기_인증면적"].astype(float) + df["무농약_인증면적"].astype(float)) / area.where(area != 0)
    diff = (v - df["친환경_비율"].astype(float)).abs()
    info = {"사용가능일_파일과_다른_행": n_avail_diff,
            "경지면적_0_행": int((area == 0).sum()), "비율_NaN_행": int(v.isna().sum()),
            "파일_친환경_비율과_차이_최대": float(diff.max()) if diff.notna().any() else None}
    return YearIndicator.from_long("eco_ratio", df["sgg_code"], df["자료연도"], v, avail), info


def gw_indicator(df: pd.DataFrame) -> tuple[YearIndicator, dict]:
    """X2: 적합 ÷ 검사(검사 0 이면 NaN), 사용가능일 = 파일 칼럼 그대로(부록 A-3)."""
    n = df["검사_건수"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = df["적합_건수"].astype(float) / n.where(n != 0)
    diff = (v - df["적합률"].astype(float)).abs()
    info = {"검사_0_행": int((n == 0).sum()), "검사_결측_행": int(n.isna().sum()), "비율_NaN_행": int(v.isna().sum()),
            "파일_적합률과_차이_최대": float(diff.max()) if diff.notna().any() else None}
    return YearIndicator.from_long("gw_pass_rate", df["sgg_code"], df["자료연도"], v, df["사용가능일"]), info


# ======================================================================== WX7 season
SEASONS = ("봄", "여름", "가을", "겨울")
_MONTH_SEASON = {3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름",
                 9: "가을", 10: "가을", 11: "가을", 12: "겨울", 1: "겨울", 2: "겨울"}


def season(dates) -> np.ndarray:
    """§5 표 WX7: 날짜(d — 단계 4 ② 는 T)의 달 → 봄·여름·가을·겨울(남반구 보정 없음)."""
    t = day_ints(dates)
    months = t.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    return np.array([_MONTH_SEASON[int(m)] for m in months], dtype=object)
