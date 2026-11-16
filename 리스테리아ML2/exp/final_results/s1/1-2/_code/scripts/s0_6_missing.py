# -*- coding: utf-8 -*-
"""s0_6_missing.py — 단계 0-6 결측 표시 목록(계획서 §4-2, §6 단계 0-6, 부록 A-5·A-8).

계획서 문구(§4-2): "결측 표시 칼럼·채널 = 0-6 에서 원 변수(일자료 변수·표 칼럼) 단위로 정함 — 그 차수 입력 표(1차 표·
2차 표 각 1개)의 d ≤ 2025-06-30 자료에 결측이 있는 원 변수 → 그 변수에서 나온 window·요약·차감 칼럼·채널 전부에 표시
… soil_t5~t30·soil_t_nas·no2·co·so2·풍향 sin·cos(1차·2차)와 이슬점 td1(2차)은 해외 행에서 늘 결측 … '해외 구조적 결측'
1칼럼으로 합침 … 그 결측 표시 목록(원 변수 단위) 기록".

하는 일(d ≤ y_cut(2025-06-30) 시료, 행 처리 뒤 단계 0-1 시료 표):
 1. 일자료 변수(1차·2차 기상 19): 시료마다 그 단위의 [d − 149, d](W 100·h 60 최악 S ~ d — 모든 W·h 의 window 합집합)에
    NaN 날이 있는 시료 수(국내·해외). 2차는 같은 구간의 차감 일자료(N_ma 4값 전부)도 봄 — 원 변수 하나라도 NaN 이면 목록.
 2. SPI3·SPI6(2차): 관측 칸 표·예보 칸 표 둘 다 같은 구간.
 3. 표 칼럼(2차: HIS1~HIS4·WX3~WX6·FD1·SO1·X1·X2·L1·rg 품목·rg 토양 pH): 시료 × h(설정 h_values: 'all' = 1~60 전부,
    'own' = 시료 자기 h)로 lst.inputs.table_features(학습 때와 같은 함수)를 계산해 NaN 행 수. WX3·WX4 window 칼럼은
    W = table_W(10 — W 10 window 는 더 긴 W 의 window 안에 있어(E 같음) NaN 행이 가장 많음).
 4. 구조적 결측 변수(§4-2 목록): 해외 일자료 표에서 늘 NaN 인지 assert(아니면 멈춤 — 계획서와 다름) → 목록 판정은
    국내 행으로만(해외 행 결측은 'ov_struct_na' 1칼럼이 맡음).
출력(<out>/, 이미 있으면 멈춤): missing_lists.json(lst.registry.MissingLists 가 읽는 꼴) · missing_detail.csv · run_meta.json

실행(서버, 검수 뒤 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_6_missing.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-6_missing

구현 세부(계획서에 없음 — 담당자 확인):
 - '입력 표의 d ≤ 2025-06-30 자료' = 위 1~3(일자료는 시료별 window 합집합 구간, 표 칼럼은 시료 × h 실제 계산값).
   목록에 더 넣는 쪽(늘 0 인 결측 표시 칼럼)은 해가 없고 빠뜨리는 쪽은 결측이 0 으로 숨으므로 넓게 잡음.
 - 차감 칼럼의 NaN(앞 N_ma 일 전부 결측 등)도 그 원 변수의 결측으로 셈.
 - 표 칼럼 HIS1·HIS4 의 NaN 은 k 와 무관(§5-1: C·n_all 로만 NaN) → table_k 하나로 셈. 이력 모집단 = d ≤ 2025-06-30.
 - rg 지중온도 6칼럼(soil_t5 에서 나옴)도 구조적 결측 대상(registry 구현 세부 (g), 지휘자 결정 11-07) → 2차 목록 판정은
   국내 행으로만(registry.structural_sources(2)).
   rg 지중온도 해외 결측의 근거 = soil_t5 가 해외 일자료에서 늘 NaN(structural_check).
 - max_samples > 0 이면 시료를 그만큼만 뽑아 봄(로컬 구조 확인용, 실제 실행은 0).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import inputs as ip  # noqa: E402
from lst import registry as rg  # noqa: E402
from lst import windows as wn  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = {
    "plan": cf.Field(str),
    "context": ip.CONTEXT_SCHEMA,
    "n_ma_values": cf.Field(list, item_types=int),
    "h_values": cf.Field(str, choices=("all", "own")),
    "table_W": cf.Field(int),
    "table_k": cf.Field(int),
    "max_samples": cf.Field(int),
    "sample_seed": cf.Field(int),
}
OUT_NAMES = ("missing_lists.json", "missing_detail.csv", "run_meta.json")
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw")


class StructuralMismatch(AssertionError):
    """§4-2 구조적 결측 변수가 해외 일자료 표에서 늘 NaN 이 아님 — 계획서와 다름(고치지 않고 멈춤)."""


def load_cfg(path) -> dict:
    cfg = dict(cf.load_config(path, SCHEMA))
    if tuple(cfg["n_ma_values"]) != rg.N_MA_VALUES:
        raise cf.ConfigError(f"n_ma_values 는 계획서 단계 2-1b 의 {list(rg.N_MA_VALUES)} 그대로")
    if cfg["table_W"] != min(wn.W_GRID):
        raise cf.ConfigError(f"table_W 는 {min(wn.W_GRID)}(가장 짧은 W — NaN 이 가장 많음)")
    if cfg["table_k"] not in rg.K_VALUES:
        raise cf.ConfigError("table_k 는 k 후보 중 하나")
    if cfg["context"]["y_cut"] != str(fd.as_date([fd.SPLITS["refit"].train_end])[0]):
        raise cf.ConfigError("y_cut 은 refit train 끝(2025-06-30)")
    if cfg["max_samples"] < 0:
        raise cf.ConfigError("max_samples ≥ 0")
    return cfg


# ---------------------------------------------------------------- 1·2. 일자료
def structural_check(ctx: ip.InputContext) -> dict:
    """구조적 결측 변수(+ td)가 해외 단위 일자료에서 전부 NaN 인지."""
    cube = ctx.wx
    ov_u = np.array([u.startswith("OV") for u in cube.units])
    rec = {}
    for v in rg.STRUCT_VARS + ("td",):
        vi = cube.variables.index(v)
        n_val = int((~np.isnan(cube.values[ov_u, :, vi])).sum())
        rec[v] = n_val
    bad = {v: n for v, n in rec.items() if n}
    if bad:
        raise StructuralMismatch(f"해외 일자료에 값이 있는 구조적 결측 변수(칸 수): {bad}")
    return {"overseas_units": int(ov_u.sum()), "non_nan_cells": rec}


def daily_counts(cube: wn.DailyCube, S: pd.DataFrame, lo, hi) -> pd.DataFrame:
    """변수마다 [lo, hi] 에 NaN 날이 있는 시료 수(국내·해외)."""
    cnt = ip.range_nan_counts(cube, S["sgg"].to_numpy(dtype=object), lo, hi) > 0
    ov = (S["src_route"] == ip.OVERSEAS).to_numpy()
    return pd.DataFrame({"variable": list(cube.variables), "n_dom": int((~ov).sum()), "n_dom_nan": cnt[~ov].sum(0),
                         "n_ov": int(ov.sum()), "n_ov_nan": cnt[ov].sum(0)})


def _flag(row, structural: bool) -> bool:
    return bool(row["n_dom_nan"] > 0 or (not structural and row["n_ov_nan"] > 0))


# ---------------------------------------------------------------- 3. 표 칼럼
def table_counts(ctx: ip.InputContext, S: pd.DataFrame, h_values: str, W: int, k: int) -> pd.DataFrame:
    units = [u for u in rg.UNITS if u.kind == "table" and u.name != "season"]
    cols = [c for u in units for c in rg.unit_tree_columns(u, W, "sum", k, None) if c.kind == "numeric"]
    ov = (S["src_route"] == ip.OVERSEAS).to_numpy()
    acc = {c.name: [0, 0, 0, 0] for c in cols}   # n_dom, n_dom_nan, n_ov, n_ov_nan (행 = (시료, h))
    split = fd.SPLITS["refit"]
    hs = range(1, 61) if h_values == "all" else [None]
    base = S[list(ip.META_REQUIRED)].reset_index(drop=True)
    for h in hs:
        meta = base if h is None else base.assign(h=np.int64(h))
        rows = ip.prepare_rows(meta, W, split)
        tab = ip.table_features(rows, ctx, units, W, k, split)
        for c in cols:
            nan = np.isnan(tab[c.name])
            a = acc[c.name]
            a[0] += int((~ov).sum())
            a[1] += int(nan[~ov].sum())
            a[2] += int(ov.sum())
            a[3] += int(nan[ov].sum())
    return pd.DataFrame([{"column": c.name, "variable": c.source, "unit": c.unit, "id": c.id, "structural": c.structural,
                          "n_dom": acc[c.name][0], "n_dom_nan": acc[c.name][1],
                          "n_ov": acc[c.name][2], "n_ov_nan": acc[c.name][3]} for c in cols])


# ---------------------------------------------------------------- 묶어서
def compute(ctx: ip.InputContext, cfg: dict, log=print) -> tuple[dict, pd.DataFrame]:
    cut = fd.day(cfg["context"]["y_cut"])
    S = ctx.samples
    S = S[fd.day_ints(S["d"]) <= cut].reset_index(drop=True)
    if cfg["max_samples"]:
        rng = np.random.default_rng(cfg["sample_seed"])
        S = S.iloc[np.sort(rng.choice(len(S), min(cfg["max_samples"], len(S)), replace=False))].reset_index(drop=True)
    d = fd.day_ints(S["d"])
    lo = wn.window_bounds(d, wn.RR8_H, wn.RR8_W).S          # d − 149
    hi_ = d
    detail = []
    struct = structural_check(ctx)

    t = time.time()
    wx = daily_counts(ctx.wx.select(list(rg.WEATHER19)), S, lo, hi_)
    for _, r in wx.iterrows():
        detail.append({"variable": r["variable"], "kind": "daily", "what": "기상 일자료", "column": r["variable"],
                       **{k: int(r[k]) for k in ("n_dom", "n_dom_nan", "n_ov", "n_ov_nan")}})
    log(f"일자료 19: {time.time() - t:.0f}초")
    for n in cfg["n_ma_values"]:
        t = time.time()
        an = daily_counts(ctx.anomaly(n), S, lo, hi_)
        for v, (_, r) in zip(rg.WEATHER19, an.iterrows()):
            detail.append({"variable": v, "kind": "daily", "what": f"차감 N_ma {n}", "column": r["variable"],
                           **{k: int(r[k]) for k in ("n_dom", "n_dom_nan", "n_ov", "n_ov_nan")}})
        ctx.drop_anomaly_cache()
        log(f"차감 {n}: {time.time() - t:.0f}초")
    for nm, cube in (("SPI 관측 칸", ctx.spi_obs), ("SPI 예보 칸", ctx.spi_fc)):
        sp = daily_counts(cube, S, lo, hi_)
        for _, r in sp.iterrows():
            detail.append({"variable": r["variable"], "kind": "spi", "what": nm, "column": r["variable"],
                           **{k: int(r[k]) for k in ("n_dom", "n_dom_nan", "n_ov", "n_ov_nan")}})
    t = time.time()
    tb = table_counts(ctx, S, cfg["h_values"], cfg["table_W"], cfg["table_k"])
    for _, r in tb.iterrows():
        detail.append({"variable": r["variable"], "kind": "table", "what": f"표 칼럼({r['id']})", "column": r["column"],
                       **{k: int(r[k]) for k in ("n_dom", "n_dom_nan", "n_ov", "n_ov_nan")}})
    log(f"표 칼럼: {time.time() - t:.0f}초")

    det = pd.DataFrame(detail)
    struct2 = set(rg.structural_sources(2))
    det["structural_order1"] = det["variable"].isin(rg.structural_sources(1)) & (det["kind"] == "daily")
    det["structural_order2"] = det["variable"].isin(struct2) & det["kind"].isin(["daily", "table"])
    det["has_nan_order1"] = [_flag(r, r["structural_order1"]) for _, r in det.iterrows()]
    det["has_nan_order2"] = [_flag(r, r["structural_order2"]) for _, r in det.iterrows()]
    o1 = det[(det["kind"] == "daily") & (det["what"] == "기상 일자료")]
    v1 = sorted(set(o1.loc[o1["has_nan_order1"], "variable"]))
    v2 = sorted(set(det.loc[det["has_nan_order2"], "variable"]))
    rows_rec = {"samples_d_le_cut": int(len(S)), "domestic": int((S["src_route"] == ip.DOMESTIC).sum()),
                "overseas": int((S["src_route"] == ip.OVERSEAS).sum())}
    res = {
        "plan": cfg["plan"], "y_cut": cfg["context"]["y_cut"], "rows": rows_rec,
        "daily_range": "[d − 149, d] (W 100·h 60 최악 S ~ d, 모든 W·h window 합집합)",
        "h_values": cfg["h_values"], "table_W": cfg["table_W"], "table_k": cfg["table_k"],
        "n_ma_values": cfg["n_ma_values"], "max_samples": cfg["max_samples"],
        "structural_check": struct,
        "order1": {"variables": v1, "structural_variables": list(rg.structural_sources(1)),
                   "structural_column": rg.STRUCT_COLUMN,
                   "candidates": list(rg.WEATHER19)},
        "order2": {"variables": v2, "structural_variables": list(rg.structural_sources(2)),
                   "structural_column": rg.STRUCT_COLUMN,
                   "candidates": sorted(set(det["variable"]))},
    }
    return res, det


def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="단계 0 결과 뿌리(서버 ~/cyj_listeria_final/results)")
    ap.add_argument("--config", default=str(CODE_ROOT / "configs" / "s0_6_missing.json"))
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = {n: rp.result(n) for n in OUT_NAMES}
    exist = [str(p) for p in out.values() if p.exists()]
    if exist:
        raise FileExistsError(f"이미 있는 결과 — 덮지 않음: {exist}")
    res0 = check_resources(rp.out)
    ctx = ip.load_context(cfg["context"], rp.root, inputs_root, components=COMPONENTS)
    check_resources(rp.out)
    res, det = compute(ctx, cfg)
    res["seconds"] = round(time.time() - t0, 1)
    with open(out["missing_detail.csv"], "x", newline="", encoding="utf-8-sig") as f:
        det.to_csv(f, index=False)
    cf.dump_json(out["missing_lists.json"], cf.nan_to_null(res))
    C, D = cfg["context"]["results"], cfg["context"]["data"]
    md5 = {f"inputs:{k}": v for k, v in cf.inputs_md5(inputs_root, [C[k] for k in C if k != "missing_lists"]).items()}
    md5.update({f"root:{k}": v for k, v in cf.inputs_md5(rp.root, list(D.values())).items()})
    cf.save_run_meta(out["run_meta.json"], cfg, md5, CODE_ROOT, n_jobs=None,
                     extra={"outputs": list(OUT_NAMES[:-1]), "config_path": Path(args.config).name,
                            "inputs_root_arg": "--inputs", "resources_at_start": res0})
    print(f"1차 목록 {len(res['order1']['variables'])}개 · 2차 목록 {len(res['order2']['variables'])}개")
    print(f"끝: {rp.out} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
