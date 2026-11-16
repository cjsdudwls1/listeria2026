# -*- coding: utf-8 -*-
"""s0_6_static.py — 단계 0-6 고정표(FD1 식품·WX4 품목 4·L1 축사)와 조회형 파생변수(SO1·X1·X2·WX7) 점검.

계획서: §5 표(FD1·WX4·WX7·SO1·X1·X2·L1)·표 아래 X1·X2 규칙, §5-2, §6 단계 0-6, 부록 A-3(WX4·X2 사용가능일),
부록 A-4(식품 고정표·축사), 부록 B-6(축사 NaN assert·SO1 원천 재계산).

만드는 것(--out, 기존 파일 안 덮음):
- 식품_고정표.parquet: p3·p2_used·food_aw·aw_label·food_ph·ph_label·food_na·na_label·{aw,ph,na}_rule·_n_sub·rg 품목 4
  (3단계 = 시료 표 p3 전부 − '코드없음' = 국내 217 + 해외 전용 3종).
- 축사_L1.parquet(+ .csv): unit(국내 sgg_code·해외 'OV'+alpha-3)·농장 수·면적 km²·면적 출처·L1·NaN 이유.
- 면적_출처.json: 받은 원본 출처 기록(출처.json 그대로) + 단위별로 쓴 통계 줄.
- 점검.json: 키 맞음·결측 수·aw_label 분포·규칙별 수·이전 DB 고정표(20261027)와 차이·SO1/X1/X2/season 시료 표 결측·범위·
  SO1 원천 재계산 대조.
- run_meta.json: code_md5·입력 md5·스레드·설정(B-2).
SO1·X1·X2·season 은 T(또는 d)마다 달라 표로 만들지 않음 — lst.static_feats 의 조회 함수를 학습 때 행마다 부름.
여기서는 시료 표(h 그대로, T = d − h)에 한 번 불러 결측 수·범위만 기록.

구현 세부(계획서에 없음 — 담당자 확인 필요):
- 2단계값 규칙의 2단계 짝 = 단계 0-1 시료 표(국내 + 해외, '코드없음' 뺀 행)의 (p3, p2) 최다 짝(build_food_db_table.py 는
  국내 시료_시군구.parquet 로 셈). 이전 DB 고정표의 train_prdlst_code 와 다른 3단계 수를 점검.json 에 적음.
- 식품 고정표 이전판(품목3단계_식품속성_DB고정표_20261027.parquet — 지운 3차 표 아님)과 값 비교를 점검용으로만 기록.
- rg 품목 4칼럼 float32(build_food_db_table.py 와 같음). 표에 없는 코드('코드없음') → FD1·rg 결측, aw_label null.
- 설정 aw_source_filter(없으면 'original' = 옛 부록 A-4 'aw 는 aw_source = original 만'): 'all' = aw·ph·na 모두 출처와
  상관없이 표 값 그대로(사용자 결정 2026-11-12 — 식품 고정표 v2, configs/s0_6_static_food2.json). 보충 규칙(하위 중앙값 →
  2단계 값 → 결측)·범위·부등호·음수 원문 처리는 그대로.
- 식품_값출처_세기.csv: 항목(aw·ph·na) × 값 출처(자기값·하위중앙·2단계값·결측) 3단계 코드 수(+ 자기값의 aw_source 분포는
  점검.json FD1.self_value_aw_source, 이전 0-6 고정표와 코드별 비교는 FD1.vs_prev_static — 설정 results_inputs.prev_food_static).
- 설정 missing_check(선택): 이 고정표로 d ≤ y_cut 시료의 식품 칼럼(food_aw·ph·na, rg 품목 4)에 결측이 남는지 보고 기존
  결측 표시 목록(missing_lists.json order2)과 대조(gr2 9칼럼은 aw·pH 결측 행이 남으면 목록에 그대로 — 결측 규칙 부록 A-3).
  달라지면 결과 파일을 다 쓴 뒤 멈춤(0-6_missing 을 새 고정표로 다시 만들어야 함 — 대기열이 멈춤).
- 국내 면적 = 국토교통부 지적통계 2025(시군구 '계' 면적 ㎡ ÷ 10⁶ = km²), 시도 이름 → 지적통계 시도 약칭은 설정
  molit_sido_abbr, 시군구 이름은 공백 빼고 맞춤. 지적통계에 없는 단위(화성시 3구 — 2026-02 신설)는 설정
  domestic_area_manual(값·출처) 또는 domestic_area_nan_ok(NaN 허용, 담당자 결정)에 없으면 멈춤.
- 해외 면적 = FAOSTAT 'Land area' 2024(1000 ha × 10 = km²), M49 → alpha-3 = DB nation_code 표.
  farm 목록에 없는 나라(OVKOR) = NaN.
- X1 비율 = (유기_인증면적 + 무농약_인증면적) ÷ 경지면적(경지면적 0 → NaN), 파일 '친환경_비율'(인증면적_합 ÷ 경지면적)과
  차이 최대를 기록. X2 비율 = 적합_건수 ÷ 검사_건수(검사 0 → NaN). season 값 = '봄'·'여름'·'가을'·'겨울'.

실행(서버, 검수 뒤 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_6_static.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-6_static
  (면적 원본은 먼저 로컬 `py -I scripts/s0_6_area_fetch.py --root <천영진>` 로 받아 자료 뿌리에 올림)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst import resources as rs  # noqa: E402

rs.set_thread_env(rs.THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import paths as ph  # noqa: E402
from lst import samples as S  # noqa: E402
from lst import static_feats as sf  # noqa: E402
from lst.folds import day_ints  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CONFIG = str(CODE_ROOT / "configs" / "s0_6_static.json")
INPUT_KEYS = ("prdlst_cl", "aw_cl", "prev_food_table", "farm_domestic_xlsx", "farm_overseas_csv", "sgg_names",
              "sample_sgg", "area_domestic_columns", "area_domestic_data", "area_overseas_zip", "area_provenance",
              "nation_code", "soil_sgg", "global_soil_ph", "eco", "gw", "raw_soil", "sgg02", "krc", "sgg_xlsx")
SCHEMA = {
    "inputs": {k: cf.Field(str) for k in INPUT_KEYS},
    "results_inputs": {"samples": cf.Field(str), "prev_food_static": cf.Field(str, required=False)},
    "aw_source_filter": cf.Field(str, required=False, choices=sf.AW_SOURCE_FILTERS),
    "missing_check": {"missing_lists": cf.Field(str), "y_cut": cf.Field(str)},
    "farm_sheet": cf.Field(str),
    "molit_year": cf.Field(int),
    "molit_sido_abbr": cf.Field(dict),
    "domestic_area_manual": cf.Field(list),
    "domestic_area_nan_ok": cf.Field(list, item_types=str),
    "fao": {"item_code": cf.Field(str), "element_code": cf.Field(str), "unit": cf.Field(str), "year": cf.Field(int)},
    "so1_verify": {"n_cells": cf.Field(int), "seed": cf.Field(int), "window_days": cf.Field(int)},
    "area_fetch": cf.Field(dict),  # s0_6_area_fetch.py 가 씀
}
MANUAL_SCHEMA = {"sgg_code": cf.Field(str), "area_km2": cf.Field((int, float)), "source": cf.Field(str),
                 "url": cf.Field(str), "base_year": cf.Field(int), "fetched": cf.Field(str)}
OUT = {"food": "식품_고정표.parquet", "l1": "축사_L1.parquet", "l1_csv": "축사_L1.csv",
       "area": "면적_출처.json", "check": "점검.json", "meta": "run_meta.json", "food_src": "식품_값출처_세기.csv"}
OPTIONAL = ("missing_check",)
# 결측 표시 목록 대조 대상(식품 고정표에서 바로 나오는 표 칼럼) — gr2 는 aw·pH 결측이면 결측(부록 A-3)
MISSING_FOOD_VARS = sf.FD1_COLUMNS[:3] + sf.RG_ITEM_COLUMNS
GR2_VARS = tuple(f"gr2_{p}_{s}" for p in ("mu", "pscotta", "pl5") for s in ("mean", "mean10", "max"))
FARM_COL = "농장수"  # 부록 A-4: D 칼럼 '농장수'(전체 농장 수)


class BuildStop(RuntimeError):
    pass


def build_parser():
    ap = ph.build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="단계 0 결과 뿌리(서버 ~/cyj_listeria_final/results)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    return ap


def load_cfg(path):
    raw = cf.load_json_strict(path)
    cfg = cf.validate(raw, {k: v for k, v in SCHEMA.items() if k not in OPTIONAL or k in raw})
    for i, m in enumerate(cfg["domestic_area_manual"]):
        cf.validate(m, MANUAL_SCHEMA, where=f"domestic_area_manual[{i}].")
    for k, v in cfg["molit_sido_abbr"].items():
        if not (isinstance(v, list) and v and all(isinstance(x, str) for x in v)):
            raise cf.ConfigError(f"molit_sido_abbr.{k} 는 문자열 목록")
    for p in cfg["results_inputs"].values():
        ph.to_rel(p)
    return cfg


def _miss_split(v: np.ndarray, s: pd.DataFrame) -> dict:
    """결측 수: 전체·국내·해외·검출 행."""
    m = pd.isna(v) if v.dtype == object else np.isnan(v)
    dom = (s["src_route"] == "domestic").to_numpy()
    y = (s["y"] == 1).to_numpy()
    return {"rows": int(len(m)), "nan": int(m.sum()), "nan_domestic": int((m & dom).sum()),
            "nan_overseas": int((m & ~dom).sum()), "nan_pos": int((m & y).sum()),
            "nan_pos_domestic": int((m & y & dom).sum()), "nan_pos_overseas": int((m & y & ~dom).sum())}


def _range(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    v = v[~np.isnan(v)]
    if not len(v):
        return {"n": 0}
    q = np.percentile(v, [0, 1, 50, 99, 100])
    return {"n": int(len(v)), "min": float(q[0]), "p1": float(q[1]), "p50": float(q[2]), "p99": float(q[3]),
            "max": float(q[4])}


# ---------------------------------------------------------------- FD1
def run_food(P, samples, chk, cfg=None, prev_static=None):
    p3 = samples["p3"].to_numpy(dtype=object)
    p2 = samples["p2"].to_numpy(dtype=object)
    dom = (samples["src_route"] == "domestic").to_numpy()
    real = p3 != S.NO_CODE
    codes_dom = set(p3[dom & real])
    codes_ov_only = set(p3[~dom & real]) - codes_dom
    if codes_ov_only != set(S.RR6_CODES):
        raise BuildStop(f"해외 전용 3단계가 RR6 3종과 다름: {sorted(codes_ov_only)}")
    codes3 = sorted(codes_dom | codes_ov_only)
    p2_of = sf.pair_p2(p3[real], p2[real])
    cl = S.read_parquet(P["prdlst_cl"], columns=["prdlst_cd", "hrnk_prdlst_cd"])
    awt = pq.read_table(P["aw_cl"])
    if awt.schema.metadata.get(b"source_table") != b"ba_data.t_aw_cl":
        raise BuildStop("aw_cl 원천 표 메타가 ba_data.t_aw_cl 아님")
    if pq.read_schema(P["prdlst_cl"]).metadata.get(b"source_table") != b"ba_data.t_prdlst_cl":
        raise BuildStop("prdlst_cl 원천 표 메타가 ba_data.t_prdlst_cl 아님")
    awf = (cfg or {}).get("aw_source_filter", "original")
    aw_df = S._to_plain(awt.to_pandas())
    ft = sf.build_food_table(cl, aw_df, codes3, p2_of, aw_source_filter=awf)
    t = ft.table
    # 점검
    rows_feat = sf.food_features(t, p3)
    prev = S.read_parquet(P["prev_food_table"])
    pv = prev.set_index("train_prdlst_code_2")
    cmp = {}
    common = [k for k in t["p3"] if k in pv.index]
    tt = t.set_index("p3")
    for c in sf.FOOD_COLS:
        a = tt.loc[common, f"food_{c}"].to_numpy(dtype=float)
        b = pv.loc[common, c].to_numpy(dtype=float)
        same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=False)
        lab_same = [x == y or (x is None and pd.isna(y)) for x, y in
                    zip(tt.loc[common, f"{c}_label"], pv.loc[common, f"{c}_label"])]
        cmp[c] = {"값_다른_코드": [k for k, s_ in zip(common, same) if not s_],
                  "표기_다른_코드": [k for k, s_ in zip(common, lab_same) if not s_]}
    cmp["2단계짝_다른_코드"] = [k for k in common if tt.loc[k, "p2_used"] != pv.loc[k, "train_prdlst_code"]]
    chk["FD1"] = {
        "codes": len(codes3), "codes_domestic": len(codes_dom), "codes_overseas_only": sorted(codes_ov_only),
        "sample_rows_no_code": int((~real).sum()),
        "key_match": {"시료_p3_중_표에_없는_코드": sorted(set(p3) - set(codes3)),
                      "표_코드_중_시료에_없는_코드": sorted(set(codes3) - set(p3))},
        "rules": {c: dict(Counter(t[f"{c}_rule"])) for c in sf.FOOD_COLS},
        "aw_label_codes": {str(k): int(v) for k, v in Counter(t["aw_label"].fillna("(null)")).items()},
        "aw_label_rows": {str(k): int(v) for k, v in Counter(pd.Series(rows_feat["aw_label"]).fillna("(null)")).items()},
        "missing_codes": {c: int(t[f"food_{c}"].isna().sum()) for c in sf.FOOD_COLS},
        "missing_rows": {c: _miss_split(rows_feat[c].to_numpy(), samples) for c in sf.FD1_COLUMNS + sf.RG_ITEM_COLUMNS},
        "range_codes": {c: _range(t[c].to_numpy()) for c in ("food_aw", "food_ph", "food_na") + sf.RG_ITEM_COLUMNS},
        "parse_rules": ft.parse_rules, "unparsed": ft.unparsed,
        "t_aw_cl_rows": ft.n_aw_rows, "aw_source_not_original_rows": ft.n_non_original,
        "vs_prev_db_table_20261027": {"common_codes": len(common), **cmp},
        "aw_source_filter": awf,
        "codes_not_in_t_aw_cl": sorted(set(codes3) - set(aw_df["prdlst_cd"])),
        "self_value_aw_source": self_value_sources(t, aw_df),
    }
    if prev_static is not None:
        chk["FD1"]["vs_prev_static"] = compare_static(t, prev_static)
    return t


def source_counts(t: pd.DataFrame) -> pd.DataFrame:
    """항목 × 값 출처 → 3단계 코드 수(출처 4종 전부 — 0 포함)."""
    rows = [{"항목": c, "값 출처": r, "3단계 코드 수": int((t[f"{c}_rule"] == r).sum())}
            for c in sf.FOOD_COLS for r in sf.RULES]
    out = pd.DataFrame(rows)
    tot = out.groupby("항목")["3단계 코드 수"].sum()
    if (tot != len(t)).any():
        raise BuildStop(f"값 출처 합이 코드 수와 다름: {tot.to_dict()}")
    return out


def self_value_sources(t: pd.DataFrame, aw_df: pd.DataFrame) -> dict:
    """자기값을 쓴 3단계 코드의 t_aw_cl.aw_source 분포(항목마다)."""
    src = dict(zip(aw_df["prdlst_cd"], aw_df["aw_source"]))
    return {c: {str(k): int(v) for k, v in Counter(src.get(k3) for k3 in t.loc[t[f"{c}_rule"] == "자기값", "p3"]).items()}
            for c in sf.FOOD_COLS}


def compare_static(t: pd.DataFrame, prev: pd.DataFrame) -> dict:
    """이전 단계 0-6 고정표(결과 폴더)와 코드마다 값·출처 비교(점검용)."""
    pv = prev.set_index("p3")
    tt = t.set_index("p3")
    if set(pv.index) != set(tt.index):
        return {"same_codes": False, "only_new": sorted(set(tt.index) - set(pv.index)),
                "only_prev": sorted(set(pv.index) - set(tt.index))}
    out = {"same_codes": True}
    for c in sf.FOOD_COLS:
        a = tt[f"food_{c}"].to_numpy(dtype=float)
        b = pv.loc[tt.index, f"food_{c}"].to_numpy(dtype=float)
        same = (np.isnan(a) & np.isnan(b)) | (a == b)
        out[c] = {"값_다른_코드_수": int((~same).sum()),
                  "결측→값": int((np.isnan(b) & ~np.isnan(a)).sum()), "값→결측": int((~np.isnan(b) & np.isnan(a)).sum()),
                  "이전_출처": {str(k): int(v) for k, v in Counter(pv.loc[tt.index, f"{c}_rule"]).items()},
                  "지금_출처": {str(k): int(v) for k, v in Counter(tt[f"{c}_rule"]).items()}}
    return out


def missing_list_check(food: pd.DataFrame, samples: pd.DataFrame, lists: dict, y_cut: str) -> dict:
    """d ≤ y_cut 시료에서 식품 칼럼 결측이 남는지 → 기존 2차 결측 표시 목록과 같은지(설명은 머리 구현 세부)."""
    cut = int(np.datetime64(y_cut, "D").astype(np.int64))
    m = day_ints(samples["d"]) <= cut
    sub = samples.loc[m]
    f = sf.food_features(food, sub["p3"].to_numpy(dtype=object))
    old = set(lists["order2"]["variables"])
    rec, diff = {}, []
    for c in MISSING_FOOD_VARS:
        n = int(np.isnan(f[c].to_numpy(dtype=np.float64)).sum())
        now, before = n > 0, c in old
        rec[c] = {"nan_rows_d_le_cut": n, "in_list_now": now, "in_list_before": before}
        if now != before:
            diff.append(c)
    aw_ph_nan = bool(np.isnan(f["food_aw"].to_numpy(dtype=np.float64)).any()
                     or np.isnan(f["food_ph"].to_numpy(dtype=np.float64)).any())
    for c in GR2_VARS:
        before = c in old
        if aw_ph_nan:   # aw·pH 결측 행이 있으면 gr2 결측 → 목록에 있음
            now = True
        elif before:    # 식품 결측이 없어졌으면 기상 결측으로만 정해짐 — 여기서 못 가림
            now = None
        else:
            now = False
        rec[c] = {"in_list_now": now, "in_list_before": before}
        if now is None or now != before:
            diff.append(c)
    return {"y_cut": y_cut, "rows_d_le_cut": int(m.sum()), "variables": rec, "changed": diff, "same": not diff}


# ---------------------------------------------------------------- L1
def run_l1(P, cfg, samples, chk):
    names = S.read_parquet(P["sgg_names"], columns=["sgg_code", "sgg_nm"])
    farm = pd.read_excel(P["farm_domestic_xlsx"], sheet_name=cfg["farm_sheet"], dtype={"시도": str, "시군구": str})
    if list(farm.columns[:4]) != ["시도", "시군구", "모집지역", FARM_COL]:
        raise BuildStop(f"축사 엑셀 칼럼이 부록 A-4 와 다름: {list(farm.columns)}")
    fd = sf.match_domestic_farms(farm, names)
    # 키 점검: 시료_시군구.parquet sgg_code 256 과 같은지
    ss = set(S.read_parquet(P["sample_sgg"], columns=["sgg_code"])["sgg_code"].unique())
    if set(names["sgg_code"]) != ss:
        raise BuildStop("시군구_대표좌표 sgg_code 와 시료_시군구 sgg_code 가 다름")
    molit = sf.molit_area_table(Path(P["area_domestic_columns"]).read_bytes(),
                                Path(P["area_domestic_data"]).read_bytes(), cfg["molit_year"])
    units = names.loc[names["sgg_code"].isin(fd["sgg_code"])]
    da = sf.match_domestic_area(units, molit, cfg["molit_sido_abbr"]).set_index("sgg_code")
    prov = json.loads(Path(P["area_provenance"]).read_text(encoding="utf-8"))
    pmap = {it["file"]: it for it in prov["items"]}
    molit_src = pmap[Path(P["area_domestic_data"]).name]
    fao_src = pmap[Path(P["area_overseas_zip"]).name]
    manual = {m["sgg_code"]: m for m in cfg["domestic_area_manual"]}
    nan_ok = set(cfg["domestic_area_nan_ok"])
    rows = []
    unresolved = []
    for r in fd.itertuples(index=False):
        code = r.sgg_code
        farms = float(getattr(r, FARM_COL))
        if code in manual:
            m = manual[code]
            area, src, yr, srcnm = float(m["area_km2"]), f"{m['source']} {m['url']} (받은 날 {m['fetched']})", m["base_year"], None
        elif not np.isnan(da.loc[code, "area_km2"]):
            area, src, yr, srcnm = float(da.loc[code, "area_km2"]), f"국토교통부 지적통계 {molit_src['url']}", molit_src["base_year"], da.loc[code, "area_src_name"]
        elif code in nan_ok:
            area, src, yr, srcnm = np.nan, "면적 없음(담당자 결정 domestic_area_nan_ok)", None, None
        else:
            unresolved.append(f"{code} {r.unit_name}")
            continue
        rows.append({"unit": code, "src_route": "domestic", "unit_name": r.unit_name, "farms": farms, "area_km2": area,
                     "area_source": src, "area_base_year": yr, "area_src_name": srcnm})
    if unresolved:
        raise BuildStop("국내 면적을 지적통계 2025 에서 못 찾은 시군구(부록 A-4 '못 맞추면 담당자에게 보고') — "
                        f"설정 domestic_area_manual 또는 domestic_area_nan_ok 에 담당자 결정을 넣어야 함: {unresolved}")
    # 해외
    oc = pd.read_csv(P["farm_overseas_csv"], encoding="utf-8-sig", keep_default_na=False, dtype={"farm_total": int})
    if list(oc.columns) != ["iso3", "iso2", "country_en", "country_ko", "farm_total"] or oc["iso3"].duplicated().any():
        raise BuildStop("해외 농장 CSV 칼럼·iso3 중복")
    nc = S.read_parquet(P["nation_code"], columns=["nation_code", "alpha_3"])
    fao = sf.fao_land_area(Path(P["area_overseas_zip"]).read_bytes(), sf.m49_map(nc), cfg["fao"]["item_code"],
                           cfg["fao"]["element_code"], cfg["fao"]["unit"], cfg["fao"]["year"]).set_index("alpha_3")
    no_area = [a for a in oc["iso3"] if a not in fao.index]
    if no_area:
        raise BuildStop(f"FAOSTAT {cfg['fao']['year']} 면적 없는 나라: {no_area}")
    for r in oc.itertuples(index=False):
        rows.append({"unit": sf.OV + r.iso3, "src_route": "overseas", "unit_name": r.country_ko, "farms": float(r.farm_total),
                     "area_km2": float(fao.loc[r.iso3, "area_km2"]), "area_source": f"FAOSTAT Land area {fao_src['url']}",
                     "area_base_year": cfg["fao"]["year"], "area_src_name": f"{fao.loc[r.iso3, 'fao_area']} (M49 {fao.loc[r.iso3, 'fao_m49']}, flag {fao.loc[r.iso3, 'flag']})"})
    t = pd.DataFrame(rows)
    ov = t["src_route"] == "overseas"
    zero = ov & (t["farms"] == 0)  # 부록 A-4: farm_total 0 = 자료 없음 → NaN
    t["L1"] = sf.l1_value(t["farms"], t["area_km2"])
    t.loc[zero, "L1"] = np.nan
    t["nan_reason"] = None
    t.loc[zero, "nan_reason"] = "farm_total 0 = 축사 자료 없음(부록 A-4)"
    t.loc[(~ov) & t["area_km2"].isna(), "nan_reason"] = "면적 없음(담당자 결정)"
    t["area_base_year"] = t["area_base_year"].astype("Int64")
    t = t[list(sf.LIVESTOCK_COLUMNS)]
    # 부록 B-6 assert: 자료 없는 곳 NaN
    sgg = samples["sgg"].to_numpy(dtype=object)
    l1 = sf.livestock_lookup(t, sgg)
    listed_dom = set(t.loc[~ov, "unit"])
    dom_rows = (samples["src_route"] == "domestic").to_numpy()
    assert np.isnan(l1[dom_rows & ~np.isin(sgg, list(listed_dom))]).all(), "33곳 밖 시군구가 NaN 아님"
    assert np.isnan(sf.livestock_lookup(t, ["41597"])).all(), "화성시 동탄구가 NaN 아님"
    assert int(zero.sum()) == 19 and np.isnan(t.loc[zero, "L1"]).all(), "farm_total 0 나라 19곳 NaN 아님"
    ov_units = set(samples.loc[~dom_rows, "sgg"])
    not_listed_ov = sorted(ov_units - set(t.loc[ov, "unit"]))
    assert np.isnan(sf.livestock_lookup(t, not_listed_ov)).all()
    if len(fd) != 33:
        raise BuildStop(f"국내 축사 시군구 {len(fd)}곳(부록 A-4 = 33)")
    chk["L1"] = {
        "domestic_units": len(fd), "overseas_units": int(ov.sum()),
        "key_match": {"국내_33곳_모두_256_안": bool(set(fd["sgg_code"]) <= ss),
                      "해외_목록_중_시료에_없는_나라": sorted(set(t.loc[ov, "unit"]) - ov_units),
                      "시료_해외_나라_중_목록에_없는_나라(NaN)": not_listed_ov},
        "nan_units": t.loc[t["L1"].isna(), ["unit", "unit_name", "nan_reason"]].to_dict("records"),
        "overseas_farm_total_0": int(zero.sum()),
        "missing_rows": _miss_split(l1, samples), "range_units": _range(t["L1"].to_numpy()),
        "farm_xlsx_데이터기준일": {"min": str(farm["데이터기준일_최소"].min()), "max": str(farm["데이터기준일_최대"].max())}
        if "데이터기준일_최소" in farm.columns else None,
    }
    area_rec = {"출처": prov, "fao": cfg["fao"], "molit_year": cfg["molit_year"],
                "domestic_area_manual": cfg["domestic_area_manual"], "domestic_area_nan_ok": cfg["domestic_area_nan_ok"],
                "units": t[["unit", "unit_name", "area_km2", "area_source", "area_base_year", "area_src_name"]]
                .astype(object).where(t.notna(), None).to_dict("records")}
    return t, area_rec


# ---------------------------------------------------------------- SO1·X1·X2·season
def run_lookups(P, cfg, samples, chk):
    units = samples["sgg"].to_numpy(dtype=object)
    d = day_ints(samples["d"])
    T = d - samples["h"].to_numpy(dtype=np.int64)
    soil = S.read_parquet(P["soil_sgg"], columns=["sgg_code", "date"] + list(sf.SO1_COLUMNS))
    gph = S.read_parquet(P["global_soil_ph"], columns=["iso3", "ph_mean"])
    so = sf.SoilLookup(soil, gph)
    del soil
    v = so(units, T)
    chk["SO1"] = {c: {"missing": _miss_split(v[c].to_numpy(), samples), "range": _range(v[c].to_numpy())}
                  for c in sf.SO1_COLUMNS}
    chk["SO1"]["table_dates"] = [str(np.datetime64(so.d0, "D")), str(np.datetime64(so.d1, "D"))]
    chk["SO1"]["T_range"] = [str(np.datetime64(int(T.min()), "D")), str(np.datetime64(int(T.max()), "D"))]
    ov_a3 = sorted({u[2:] for u in units if u.startswith(sf.OV)})
    chk["SO1"]["overseas_nations_no_global_ph"] = [a for a in ov_a3 if a not in so.gph]
    chk["SO1"]["verify_raw"] = verify_so1(P, cfg, so, units, T)
    eco, eco_info = sf.eco_indicator(S.read_parquet(P["eco"]))
    gw, gw_info = sf.gw_indicator(S.read_parquet(P["gw"]))
    if eco_info["사용가능일_파일과_다른_행"]:
        raise BuildStop(f"X1 파일 사용가능일이 '자료연도 + 2년 1월 1일' 과 다른 행 {eco_info['사용가능일_파일과_다른_행']}")
    for name, ind, info in (("X1_eco_ratio", eco, eco_info), ("X2_gw_pass_rate", gw, gw_info)):
        x = ind(units, T)
        ys = ind.y_star(T)
        chk[name] = {"missing": _miss_split(x, samples), "range": _range(x), "table": info,
                     "year_avail": {int(y): str(np.datetime64(int(a), "D")) for y, a in zip(ind.years, ind.avail)},
                     "y_star_rows": {str(int(k)): int(n) for k, n in zip(*np.unique(ys, return_counts=True))},
                     "units_in_table": len(ind.units),
                     "domestic_units_not_in_table": sorted({u for u in units if not u.startswith(sf.OV)} - set(ind.units))}
    se = sf.season(d)
    chk["WX7_season"] = {"rows": {str(k): int(n) for k, n in Counter(se).items()}}


def verify_so1(P, cfg, so, units, T):
    """부록 B-6: 원천 soil_ph_info 로 시군구·날짜 몇 칸 다시 계산 → 표(date = T 행) 값·검정일 ≤ T−1 대조."""
    vc = cfg["so1_verify"]
    dom = np.array([not u.startswith(sf.OV) for u in units])
    pairs = pd.DataFrame({"u": units[dom], "t": T[dom]}).drop_duplicates().sort_values(["u", "t"]).reset_index(drop=True)
    rng = np.random.default_rng(vc["seed"])
    pick = pairs.iloc[np.sort(rng.choice(len(pairs), size=min(vc["n_cells"], len(pairs)), replace=False))]
    raw = pq.read_table(P["raw_soil"], columns=["legal_dong_code", "exam_day", "acid", "om", "vldpha"])
    enc = pc.dictionary_encode(raw["legal_dong_code"]).combine_chunks()
    codes_u = enc.dictionary.to_pylist()
    sgg02 = S.load_sgg02(P["sgg02"])
    region = sgg02.Region(str(P["krc"]))
    mapping = sgg02.load_mapping(str(P["sgg_xlsx"]))
    sgg_u = sgg02.assign_sgg(codes_u, None, region=region, mapping=mapping)
    row_sgg = np.array(sgg_u, dtype=object)[enc.indices.to_numpy()]
    days = pc.cast(raw["exam_day"], pa.int32()).to_numpy().astype(np.int64)
    vals = {c: raw[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in ("acid", "om", "vldpha")}
    del raw, enc
    want = set(pick["u"])
    sel = np.isin(row_sgg, list(want))
    row_sgg, days = row_sgg[sel], days[sel]
    vals = {c: v[sel] for c, v in vals.items()}
    got = so(pick["u"].to_numpy(dtype=object), pick["t"].to_numpy())
    ok = {c: 0 for c in sf.SO1_COLUMNS}
    bad = []
    for i, (u, t) in enumerate(zip(pick["u"], pick["t"])):
        m = row_sgg == u
        for raw_c, col in (("acid", "soil_ph"), ("om", "soil_om"), ("vldpha", "soil_vldpha")):
            ref, n = sf.soil_direct(days[m], vals[raw_c][m], int(t), vc["window_days"])
            g = got[col].iloc[i]
            same = (np.isnan(ref) and np.isnan(g)) or (not np.isnan(ref) and float(np.float32(ref)) == float(g))
            ok[col] += bool(same)
            if not same:
                bad.append({"sgg": u, "T": str(np.datetime64(int(t), "D")), "col": col, "table": g, "raw": ref, "n": n})
    res = {"cells": len(pick), "match": ok, "mismatch": bad[:20]}
    assert not bad, f"SO1 원천 재계산과 표가 다름: {bad[:5]}"
    return res


def main(argv=None) -> Path:
    rp, args = ph.parse_paths(argv, build_parser())
    inputs_root = Path(args.inputs).expanduser().resolve()
    cfg = load_cfg(args.config)
    for k in OUT.values():
        if rp.result(k).exists():
            raise FileExistsError(f"덮어쓰기 금지: {rp.result(k)}")
    P = {k: rp.data(v) for k, v in cfg["inputs"].items()}
    samples_p = ph.resolve(inputs_root, cfg["results_inputs"]["samples"])
    rs.check_resources(rp.out)
    samples = S.read_parquet(samples_p, columns=["lims_data_key", "y", "src_route", "sgg", "p2", "p3", "d", "h"])
    chk = {"samples": {"rows": len(samples), "pos": int(samples["y"].sum())}}
    prev_rel = cfg["results_inputs"].get("prev_food_static")
    prev_static = S.read_parquet(ph.resolve(inputs_root, prev_rel)) if prev_rel else None
    food = run_food(P, samples, chk, cfg, prev_static)
    mchk = None
    if "missing_check" in cfg:
        lists = json.loads(ph.resolve(inputs_root, cfg["missing_check"]["missing_lists"]).read_text(encoding="utf-8"))
        mchk = chk["missing_list_check"] = missing_list_check(food, samples, lists, cfg["missing_check"]["y_cut"])
    l1, area_rec = run_l1(P, cfg, samples, chk)
    run_lookups(P, cfg, samples, chk)
    pq.write_table(pa.Table.from_pandas(food, preserve_index=False), rp.result(OUT["food"]))
    with open(rp.result(OUT["food_src"]), "x", newline="", encoding="utf-8-sig") as fh:
        source_counts(food).to_csv(fh, index=False)
    pq.write_table(pa.Table.from_pandas(l1, preserve_index=False), rp.result(OUT["l1"]))
    l1.to_csv(rp.result(OUT["l1_csv"]), index=False, encoding="utf-8-sig")
    cf.dump_json(rp.result(OUT["area"]), cf.nan_to_null(area_rec))
    cf.dump_json(rp.result(OUT["check"]), cf.nan_to_null(chk))
    md5 = cf.inputs_md5(rp.root, cfg["inputs"].values())
    md5.update({f"results/{v}": cf.file_md5(ph.resolve(inputs_root, v)) for v in cfg["results_inputs"].values()})
    if "missing_check" in cfg:
        v = cfg["missing_check"]["missing_lists"]
        md5[f"results/{v}"] = cf.file_md5(ph.resolve(inputs_root, v))
    cf.save_run_meta(rp.result(OUT["meta"]), cfg, md5, CODE_ROOT, n_jobs=None)
    print("식품 값 출처(3단계 코드 수):", {c: dict(Counter(food[f"{c}_rule"])) for c in sf.FOOD_COLS})
    if mchk is not None and not mchk["same"]:
        raise BuildStop(f"결측 표시 목록이 바뀜(식품 칼럼 {mchk['changed']}) — 결과는 썼음. 0-6_missing 을 새 고정표로 "
                        f"다시 만들어야 함(지휘자 확인): {rp.result(OUT['check'])}")
    if mchk is not None:
        print("결측 표시 목록 대조: 같음 — 기존 0-6_missing 그대로 씀")
    print(f"끝: {rp.out}")
    return rp.out


if __name__ == "__main__":
    main()
