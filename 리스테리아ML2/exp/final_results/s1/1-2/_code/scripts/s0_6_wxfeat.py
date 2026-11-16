# -*- coding: utf-8 -*-
"""s0_6_wxfeat.py — 단계 0-6 기상 파생 WX3~WX6 점검(fold 1 train 행 일부: 칼럼 결측 수·범위만, 큰 파일 안 만듦).

계획서: §5 표(WX3·WX4·WX5·WX6), §4-1(E 끝·R − 1 끝), 부록 A-3(생장식·생장 범위 공식·결측 규칙), §6 단계 0-6.

하는 일:
1. 생장식 계수: `리스테리아 생장식.xlsx` 시트 '(완료) 리스테리아 모노사이토제네스' 셀(F6·F26·F19·F15·F21·F28·J15·F41·
   D6·J6·D15)을 읽어 계획서 부록 A-3 값과 대조 — 하나라도 다르면 멈춤(고치지 않음).
2. fold 1 train 행(§3, 시료마다 h 1개 — lst.folds.make_train_rows) 중 일부(n_rows, row_seed)로 W 마다
   lst.wxfeat.wx_derived(WX3·WX4·WX5·WX6) 계산 → 칼럼마다 행 수·결측 수(국내·해외)·최소·최대·평균.
   aw·pH(FD1)는 설정 food_table 이 있을 때만(없으면 NaN 배열 → gr2 전부 결측으로 나옴),
   soil_ph(SO1)는 설정 soil 이 있을 때 점검용으로 찾아 넣음(아래 구현 세부).

만드는 것(--out, 기존 파일 안 덮음, --dry-run 이면 화면에만): 0-6_wxfeat_점검.csv, 0-6_생장식_계수_대조.json, run_meta.json.

실행(서버, 검수 뒤 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_6_wxfeat.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-6_wxfeat_check

구현 세부(계획서에 없음 — 담당자 확인 필요):
- 점검 행 고르기 = fold 1 train 행에서 np.random.default_rng(row_seed).choice(n_rows, 비복원) 후 원 순서 정렬.
- soil_ph 점검용 찾기(정식 SO1 칼럼은 SO1 담당 단계가 만듦): 국내 = 시군구_일별_토양.parquet 의 (sgg_code, date = T)
  행 soil_ph(§5 SO1 문구), 해외 = global_soil_ph.ph_mean(iso3 = sgg 의 'OV' 뒤 3글자), 없으면 NaN.
- food_table = {path, key, aw, ph}: path 는 --inputs(단계 0 결과) 아래 상대 경로(0-6 식품 고정표), 시료 p3 를 key 칼럼과 그대로 맞춤, 없는 코드 = NaN.
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst import resources as rs  # noqa: E402

rs.set_thread_env(rs.THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import paths as ph  # noqa: E402
from lst import windows as wn  # noqa: E402
from lst import wxfeat as wf  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CONFIG = "configs/s0_6_wxfeat.json"
SCHEMA = {
    "plan": cf.Field(str),
    "inputs": {"samples": cf.Field(str), "domestic_weather": cf.Field(str), "overseas_weather": cf.Field(str)},
    "growth_xlsx": cf.Field(str),
    "split": cf.Field(str, choices=tuple(fd.SPLITS)),
    "n_rows": cf.Field(int),
    "row_seed": cf.Field(int),
    "W": cf.Field(list, item_types=int),
    "food_table": cf.Field((dict, type(None))),
    "soil": cf.Field((dict, type(None))),
}
FOOD_SCHEMA = {"path": cf.Field(str), "key": cf.Field(str), "aw": cf.Field(str), "ph": cf.Field(str)}
SOIL_SCHEMA = {"domestic": cf.Field(str), "overseas": cf.Field(str)}
SAMPLE_COLS = ["lims_data_key", "y", "src_route", "sgg", "p3", "d", "h"]


def build_parser():
    ap = ph.build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="단계 0 결과 뿌리(서버 ~/cyj_listeria_final/results)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true", help="파일을 만들지 않고 화면에만")
    return ap


def pick_rows(samples: pd.DataFrame, split: str, n_rows: int, seed: int) -> pd.DataFrame:
    tr = fd.make_train_rows(samples, fd.get_split(split))
    rr8 = wn.rr8_mask(tr["d"], tr["src_route"])
    if rr8.any():
        print(f"[0-6] RR8 해당 {int(rr8.sum())}행 뺌(§2-2)")
        tr = tr.loc[~rr8]
    if n_rows < len(tr):
        idx = np.sort(np.random.default_rng(seed).choice(len(tr), n_rows, replace=False))
        tr = tr.iloc[idx]
    return tr.reset_index(drop=True)


def soil_lookup(root: Path, soil_cfg, rows: pd.DataFrame) -> np.ndarray:
    T = fd.day_ints(rows["T"])
    out = np.full(len(rows), np.nan)
    dom = rows["src_route"].to_numpy() == "domestic"
    st = pq.read_table(ph.resolve(root, soil_cfg["domestic"]), columns=["sgg_code", "date", "soil_ph"]).to_pandas()
    st["day"] = fd.to_days(st["date"])
    if st.duplicated(["sgg_code", "day"]).any():
        raise ValueError("토양 표 (sgg_code, date) 중복")
    key = pd.MultiIndex.from_arrays([st["sgg_code"].astype(str), st["day"]])
    s = pd.Series(st["soil_ph"].to_numpy(np.float64), index=key)
    q = pd.MultiIndex.from_arrays([rows.loc[dom, "sgg"].astype(str), T[dom]])
    out[dom] = s.reindex(q).to_numpy()
    g = pq.read_table(ph.resolve(root, soil_cfg["overseas"]), columns=["iso3", "ph_mean"]).to_pandas()
    if g["iso3"].duplicated().any():
        raise ValueError("global_soil_ph iso3 중복")
    gm = pd.Series(g["ph_mean"].to_numpy(np.float64), index=g["iso3"].astype(str))
    a3 = rows.loc[~dom, "sgg"].astype(str).str[len(wf.OVERSEAS_PREFIX):]
    out[~dom] = gm.reindex(a3).to_numpy()
    return out


def food_lookup(root: Path, food_cfg, rows: pd.DataFrame):
    t = pq.read_table(ph.resolve(root, food_cfg["path"]), columns=[food_cfg["key"], food_cfg["aw"], food_cfg["ph"]]).to_pandas()
    if t[food_cfg["key"]].duplicated().any():
        raise ValueError("식품 고정표 키 중복")
    t = t.set_index(t[food_cfg["key"]].astype(str))
    k = rows["p3"].astype(str)
    return (t[food_cfg["aw"]].astype(np.float64).reindex(k).to_numpy(),
            t[food_cfg["ph"]].astype(np.float64).reindex(k).to_numpy())


def col_stats(X: pd.DataFrame, route: np.ndarray, W: int) -> pd.DataFrame:
    recs = []
    dom = route == "domestic"
    for c in X.columns:
        v = X[c].to_numpy(np.float64)
        na = np.isnan(v)
        recs.append({"W": W, "column": c, "n": len(v), "n_nan": int(na.sum()),
                     "n_nan_domestic": int(na[dom].sum()), "n_nan_overseas": int(na[~dom].sum()),
                     "min": float(np.nanmin(v)) if (~na).any() else np.nan,
                     "max": float(np.nanmax(v)) if (~na).any() else np.nan,
                     "mean": float(np.nanmean(v)) if (~na).any() else np.nan})
    return pd.DataFrame(recs)


def run(argv=None) -> int:
    rp, args = ph.parse_paths(argv, build_parser())
    inputs_root = Path(args.inputs).expanduser().resolve()
    if not inputs_root.is_dir():
        raise FileNotFoundError(f"--inputs 폴더 없음: {inputs_root}")
    cfg = cf.load_config(CODE_ROOT / args.config if not Path(args.config).is_absolute() else args.config, SCHEMA)
    if cfg["food_table"] is not None:
        cf.validate(cfg["food_table"], FOOD_SCHEMA, where="food_table.")
    if cfg["soil"] is not None:
        cf.validate(cfg["soil"], SOIL_SCHEMA, where="soil.")
    if not args.dry_run:
        rs.check_resources(rp.out)

    # 1. 생장식 계수 대조(다르면 GrowthCoefMismatch 로 멈춤)
    xlsx = rp.data(cfg["growth_xlsx"])
    coef_rows = wf.check_growth_cells(wf.read_growth_cells(xlsx))
    print(f"[0-6] 생장식 계수 {len(coef_rows)}개 엑셀 = 계획서")

    # 2. 행·입력
    samples = pq.read_table(ph.resolve(inputs_root, cfg["inputs"]["samples"]), columns=SAMPLE_COLS).to_pandas()
    rows = pick_rows(samples, cfg["split"], cfg["n_rows"], cfg["row_seed"])
    n = len(rows)
    units = sorted(rows["sgg"].astype(str).unique())
    dom = pq.read_table(ph.resolve(inputs_root, cfg["inputs"]["domestic_weather"]),
                        columns=["sgg_code", "date"] + list(wf.CUBE_VARS)).to_pandas()
    ov = pq.read_table(ph.resolve(inputs_root, cfg["inputs"]["overseas_weather"]),
                       columns=["alpha_3", "sgg", "date"] + list(wf.WEATHER19) + list(wf.OVERSEAS_RENAME)).to_pandas()
    cube = wf.combined_cube(dom, ov, units=units)
    del dom, ov
    aw, phv = (np.full(n, np.nan), np.full(n, np.nan)) if cfg["food_table"] is None else food_lookup(inputs_root, cfg["food_table"], rows)
    soil = np.full(n, np.nan) if cfg["soil"] is None else soil_lookup(rp.root, cfg["soil"], rows)
    route = rows["src_route"].to_numpy()
    print(f"[0-6] 행 {n:,}(국내 {int((route == 'domestic').sum()):,}·해외 {int((route == 'overseas').sum()):,}), "
          f"aw 결측 {int(np.isnan(aw).sum()):,}, soil_ph 결측 {int(np.isnan(soil).sum()):,}")

    stats = []
    for W in cfg["W"]:
        X = wf.wx_derived(cube, rows["sgg"].astype(str).to_numpy(), rows["d"], rows["h"].to_numpy(), W,
                          aw=aw, ph=phv, soil_ph=soil)
        stats.append(col_stats(X, route, W))
    st = pd.concat(stats, ignore_index=True)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(st.to_string(index=False))

    if not args.dry_run:
        p = rp.out / "0-6_wxfeat_점검.csv"
        if p.exists():
            raise FileExistsError(f"덮어쓰기 금지: {p}")
        st.to_csv(p, index=False, encoding="utf-8-sig")
        cf.dump_json(rp.out / "0-6_생장식_계수_대조.json", cf.nan_to_null(coef_rows))
        md5 = {f"inputs/{k}": cf.file_md5(ph.resolve(inputs_root, v)) for k, v in cfg["inputs"].items()}
        if cfg["food_table"] is not None:
            md5["inputs/food_table"] = cf.file_md5(ph.resolve(inputs_root, cfg["food_table"]["path"]))
        rels = [cfg["growth_xlsx"]] + ([] if cfg["soil"] is None else list(cfg["soil"].values()))
        md5.update({f"root/{k}": v for k, v in cf.inputs_md5(rp.root, rels).items()})
        cf.save_run_meta(rp.out / "run_meta.json", cfg, md5, CODE_ROOT, n_jobs=None,
                         extra={"n_rows_used": n, "dry_run": False})
    return 0


if __name__ == "__main__":
    sys.exit(run())
