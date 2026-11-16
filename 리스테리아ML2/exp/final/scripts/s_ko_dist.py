# -*- coding: utf-8 -*-
"""s_ko_dist.py — 추가 판 KO1(Koutsoumanis 생육확률 6칼럼) 시료 기준 분포(학습 없음, 보고용 — 사용자 결정 2026-11-12).

시료 표(단계 0-1, 행 처리 뒤) 시료마다 자기 h(부록 A-1)·설정 W 로 KO 6칼럼(lst/inputs.table_features — 학습 때와 같은
함수, 식품 고정표 = 설정 context food_table)을 계산해 칼럼 × 묶음(전체·국내·해외·d ≤ y_cut·d > y_cut)마다
시료 수·결측 수·0 비율(값 = 0)·1e-6 미만 비율·0.5 초과 비율·평균·중앙값·최대를 적음(결측 뺀 값으로).
출력(--out, 이미 있으면 멈춤): ko_dist.csv · run_meta.json

실행(서버 — 검수 뒤 지휘자):
  $P scripts/s_ko_dist.py --root $B/data --inputs $B/results --out $B/results/s5_6/ko_dist --config configs/s_ko_dist.json

구현 세부(계획서에 없음 — 추가 판): 묶음·통계 항목은 위 그대로(지휘자 요청 '0 비율·평균·최대'에 결측·작은 값 비율 더함).
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
from lst.paths import build_parser, parse_paths  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = {"plan": cf.Field(str), "context": ip.CONTEXT_SCHEMA, "W": cf.Field(int, choices=wn.W_GRID),
          "chunk_rows": cf.Field(int)}
OUT_NAMES = ("ko_dist.csv", "run_meta.json")
COMPONENTS = ("weather", "food")
SMALL = 1e-6


def ko_values(ctx: ip.InputContext, S: pd.DataFrame, W: int, chunk_rows: int, log=print) -> pd.DataFrame:
    units = [u for u in rg.EXTRA_UNITS if u.id == "KO1"]
    split = fd.SPLITS["refit"]
    parts = []
    for lo in range(0, len(S), chunk_rows):
        meta = S.iloc[lo:lo + chunk_rows][list(ip.META_REQUIRED)].reset_index(drop=True)
        rows = ip.prepare_rows(meta, W, split)
        tab = ip.table_features(rows, ctx, units, W, None, split)
        parts.append(pd.DataFrame({c: tab[c] for c in rg.KO}))
        log(f"{min(len(S), lo + chunk_rows)}/{len(S)}")
    return pd.concat(parts, ignore_index=True)


def dist_table(v: pd.DataFrame, S: pd.DataFrame, y_cut: str) -> pd.DataFrame:
    dom = (S["src_route"] == ip.DOMESTIC).to_numpy()
    le = fd.day_ints(S["d"]) <= fd.day(y_cut)
    groups = {"전체": np.ones(len(S), bool), "국내": dom, "해외": ~dom, f"d ≤ {y_cut}": le, f"d > {y_cut}": ~le}
    rows = []
    for c in rg.KO:
        x = v[c].to_numpy(dtype=np.float64)
        for g, m in groups.items():
            a = x[m]
            ok = a[~np.isnan(a)]
            n = int(ok.size)
            rows.append({"칼럼": c, "묶음": g, "시료 수": int(m.sum()), "결측 수": int(np.isnan(a).sum()),
                         "0 비율": float((ok == 0).mean()) if n else np.nan,
                         "1e-6 미만 비율": float((ok < SMALL).mean()) if n else np.nan,
                         "0.5 초과 비율": float((ok > 0.5).mean()) if n else np.nan,
                         "평균": float(ok.mean()) if n else np.nan, "중앙값": float(np.median(ok)) if n else np.nan,
                         "최대": float(ok.max()) if n else np.nan})
    return pd.DataFrame(rows)


def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", default=str(CODE_ROOT / "configs" / "s_ko_dist.json"))
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, SCHEMA))
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = {n: rp.result(n) for n in OUT_NAMES}
    exist = [str(p) for p in out.values() if p.exists()]
    if exist:
        raise FileExistsError(f"이미 있는 결과 — 덮지 않음: {exist}")
    check_resources(rp.out)
    ctx = ip.load_context(cfg["context"], rp.root, inputs_root, components=COMPONENTS)
    S = ctx.samples.reset_index(drop=True)
    v = ko_values(ctx, S, cfg["W"], cfg["chunk_rows"])
    tab = dist_table(v, S, cfg["context"]["y_cut"])
    with open(out["ko_dist.csv"], "x", newline="", encoding="utf-8-sig") as fh:
        tab.to_csv(fh, index=False, float_format="%.6g")
    C = cfg["context"]["results"]
    md5 = {f"inputs:{k}": x for k, x in cf.inputs_md5(inputs_root, [C["samples"], C["domestic_weather"],
                                                                     C["overseas_weather"], C["food_table"]]).items()}
    cf.save_run_meta(out["run_meta.json"], cfg, md5, CODE_ROOT, n_jobs=None,
                     extra={"seconds": round(time.time() - t0, 1), "config_path": Path(args.config).name})
    print(tab[tab["묶음"] == "전체"].to_string(index=False))
    print(f"끝: {rp.out} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
