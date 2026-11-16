# -*- coding: utf-8 -*-
"""s0_8_groups.py — 단계 0-8 그룹 번호·소속 단위 전체 표 + 그룹 사이 상관(계획서 §6 단계 0-8, 부록 A-5·A-8).

계획서 문구(§6 0-8): "단계 3-0b 그룹 번호·소속 단위 전체 표를 만들어 고정(부록 A-5, 소속 칼럼 목록은 요약·원값 두 방식과
딥러닝 채널·x_static 칼럼 다), 그룹 사이 상관 기록(fold 1 train, `<변수>_mean` Spearman, 해석용)".

하는 일:
 1. lst.registry 의 단위 110·그룹 44(기상 9 + 비기상 35) 표 — 짝 3개·칼럼 묶음 113·다단위 그룹 단위 74 를 assert.
 2. 소속 표: W(설정 W_list)마다 단위별 트리 칼럼(요약·원값), 딥러닝 x_seq 채널·x_static 칼럼, 결측 표시 칼럼·채널
    (단계 0-6 결측 표시 목록 2차). k·N_ma 는 단계 2 에서 정해지므로 이름에 {K}(= k###)·{N}(= N_ma) 자리표시.
 3. 상관: fold 1 train 행(시료마다 h 1개, §3)으로 2차 요약 판(W·k·N_ma = 설정 corr)을 만들어 단위 대표 칼럼(registry.
    representative_columns — window·SPI·차감 = `_mean`, 표 칼럼 = 그 칼럼, 범주 제외) Spearman(쌍마다 둘 다 값 있는 행) →
    단위 × 단위 표, 그룹 쌍마다 |ρ| 최대 단위 쌍. 해석용(고르기에 안 씀).
출력(<out>/, 이미 있으면 멈춤): 0-8_groups.csv · 0-8_units.csv · 0-8_membership.csv · 0-8_corr_units.csv ·
  0-8_corr_groups.csv · 0-8_check.json · run_meta.json

실행(서버, 0-6 결측 표시 목록 뒤, 검수 뒤 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_8_groups.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-8_groups

구현 세부(계획서에 없음 — 담당자 확인):
 - 상관 판의 W·k·N_ma = 0-9 대리 설정과 같은 W 100·k 50·N_ma 15(window 가 정해지기 전이라) — 설정 corr.
 - '`<변수>_mean`' 을 단위 대표 칼럼으로 넓힘(비기상 단위·WX3~WX6 은 칼럼 하나라 그 칼럼). season·aw_label(범주) 제외.
 - Spearman = pandas corr(method='spearman', min_periods = 설정), corr.max_rows > 0 이면 그만큼 무작위 행만(로컬 확인용).
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
from lst import scope as lst_scope  # noqa: E402
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
    "W_list": cf.Field(list, item_types=int),
    "corr": {
        "W": cf.Field(int),
        "k": cf.Field(int),
        "n_ma": cf.Field(int),
        "split": cf.Field(str, choices=("fold1",)),
        "min_periods": cf.Field(int),
        "max_rows": cf.Field(int),
        "seed": cf.Field(int),
    },
}
SCHEMA = lst_scope.with_field(SCHEMA)   # 국내만 버전 scope 칸(없으면 지금 판)
OUT_NAMES = ("0-8_groups.csv", "0-8_units.csv", "0-8_membership.csv", "0-8_corr_units.csv", "0-8_corr_groups.csv",
             "0-8_check.json", "run_meta.json")
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")


def load_cfg(path) -> dict:
    cfg = dict(cf.load_config(path, SCHEMA))
    bad = [w for w in cfg["W_list"] if w not in wn.W_GRID]
    if bad or len(set(cfg["W_list"])) != len(cfg["W_list"]) or not cfg["W_list"]:
        raise cf.ConfigError(f"W_list 는 {wn.W_GRID} 중(중복 없이): {cfg['W_list']}")
    c = cfg["corr"]
    if c["W"] not in wn.W_GRID or c["k"] not in rg.K_VALUES or c["n_ma"] not in rg.N_MA_VALUES:
        raise cf.ConfigError("corr W·k·n_ma 는 계획서 후보 값 중")
    if c["max_rows"] < 0 or c["min_periods"] < 2:
        raise cf.ConfigError("corr.max_rows ≥ 0, min_periods ≥ 2")
    if "missing_lists" not in cfg["context"]["results"]:
        raise cf.ConfigError("context.results.missing_lists(단계 0-6 결과) 필요")
    return cfg


def tables(cfg: dict, missing: rg.MissingLists) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    g = rg.group_table()
    u = rg.unit_table()
    mem = pd.concat([rg.membership_table(W, missing=missing, order=2) for W in cfg["W_list"]], ignore_index=True)
    check = {
        "units": int(len(u)), "groups": int(len(g)), "weather_groups": int(g["weather"].sum()),
        "nonweather_groups": int((~g["weather"]).sum()), "column_bundles": int(u["n_bundles"].sum()),
        "pairs": int((u["n_bundles"] == 2).sum()),
        "multi_unit_group_units": int(g.loc[g["n_units"] > 1, "n_units"].sum()),
        "research_units": list(rg.RESEARCH_UNITS), "weather8_groups": list(rg.WEATHER8_GROUPS),
        "structural_column": None if lst_scope.domestic_only() else rg.STRUCT_COLUMN, "membership_rows": {
            f"W{W:03d}": {w: int(n) for w, n in mem[mem["W"] == W]["where"].value_counts().sort_index().items()}
            for W in cfg["W_list"]},
    }
    assert check["units"] == 110 and check["groups"] == 44 and check["weather_groups"] == 9
    assert check["column_bundles"] == 113 and check["pairs"] == 3 and check["multi_unit_group_units"] == 74
    return g, u, mem, check


def correlation(ctx: ip.InputContext, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    c = cfg["corr"]
    split = fd.SPLITS[c["split"]]
    S = ctx.samples
    S = S[fd.day_ints(S["d"]) <= split.train_end].reset_index(drop=True)
    meta = fd.make_train_rows(S[list(ip.META_REQUIRED)], split)
    if c["max_rows"]:
        rng = np.random.default_rng(c["seed"])
        meta = meta.iloc[np.sort(rng.choice(len(meta), min(c["max_rows"], len(meta)), replace=False))]
    meta = meta.reset_index(drop=True)
    spec = rg.VersionSpec("0-8-corr", 2, c["W"], "sum", include_ids=rg.ALL_IDS, k=c["k"], n_ma=c["n_ma"])
    X = ip.build_tree_X(meta, ctx, spec, split=split)
    rep = rg.representative_columns(c["W"], c["k"], c["n_ma"])
    D = pd.DataFrame({u: X[col].to_numpy(dtype=np.float64) for u, col in rep.items()})
    corr = D.corr(method="spearman", min_periods=c["min_periods"])
    unit_group = {u.name: (u.group_no, u.group) for u in rg.UNITS}
    rows = []
    gnames = {g.no: g.name for g in rg.GROUPS}
    names = list(D.columns)
    by_g: dict[int, list[str]] = {}
    for n in names:
        by_g.setdefault(unit_group[n][0], []).append(n)
    gs = sorted(by_g)
    for i, a in enumerate(gs):
        for b in gs[i + 1:]:
            sub = corr.loc[by_g[a], by_g[b]]
            arr = sub.to_numpy()
            if np.isnan(arr).all():
                rows.append({"group_a_no": a, "group_a": gnames[a], "group_b_no": b, "group_b": gnames[b],
                             "max_abs_rho": np.nan, "rho": np.nan, "unit_a": None, "unit_b": None})
                continue
            k = np.nanargmax(np.abs(arr))
            ia, ib = np.unravel_index(k, arr.shape)
            rows.append({"group_a_no": a, "group_a": gnames[a], "group_b_no": b, "group_b": gnames[b],
                         "max_abs_rho": float(abs(arr[ia, ib])), "rho": float(arr[ia, ib]),
                         "unit_a": sub.index[ia], "unit_b": sub.columns[ib]})
    grp = pd.DataFrame(rows).sort_values(["max_abs_rho", "group_a_no", "group_b_no"], ascending=[False, True, True],
                                         na_position="last")
    corr.index.name = "unit"
    rec = {"split": c["split"], "rows": int(len(D)), "W": c["W"], "k": c["k"], "n_ma": c["n_ma"],
           "units_in_corr": len(names), "representative": rep,
           "non_null_rows": {n: int(D[n].notna().sum()) for n in names}}
    return corr.reset_index(), grp, rec


def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="단계 0 결과 뿌리(서버 ~/cyj_listeria_final/results)")
    ap.add_argument("--config", default=str(CODE_ROOT / "configs" / "s0_8_groups.json"))
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = {n: rp.result(n) for n in OUT_NAMES}
    exist = [str(p) for p in out.values() if p.exists()]
    if exist:
        raise FileExistsError(f"이미 있는 결과 — 덮지 않음: {exist}")
    res0 = check_resources(rp.out)
    ctx = ip.load_context(cfg["context"], rp.root, inputs_root, components=COMPONENTS)
    g, u, mem, check = tables(cfg, ctx.missing)
    check_resources(rp.out)
    corr, grp, rec = correlation(ctx, cfg)
    check["correlation"] = rec
    check["seconds"] = round(time.time() - t0, 1)
    for name, df in (("0-8_groups.csv", g), ("0-8_units.csv", u), ("0-8_membership.csv", mem),
                     ("0-8_corr_units.csv", corr), ("0-8_corr_groups.csv", grp)):
        with open(out[name], "x", newline="", encoding="utf-8-sig") as f:
            df.to_csv(f, index=False)
    cf.dump_json(out["0-8_check.json"], cf.nan_to_null(check))
    C, D = cfg["context"]["results"], cfg["context"]["data"]
    md5 = {f"inputs:{k}": v for k, v in cf.inputs_md5(inputs_root, list(C.values())).items()}
    md5.update({f"root:{k}": v for k, v in cf.inputs_md5(rp.root, list(D.values())).items()})
    cf.save_run_meta(out["run_meta.json"], cfg, md5, CODE_ROOT, n_jobs=None,
                     extra={"outputs": list(OUT_NAMES[:-1]), "config_path": Path(args.config).name,
                            "inputs_root_arg": "--inputs", "resources_at_start": res0})
    print(f"끝: {rp.out} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
