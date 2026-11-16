# -*- coding: utf-8 -*-
"""s0_6_history.py — 단계 0-6 과거 이력(HIS1~HIS4) 칼럼 점검(계획서 §5 표·§5-1·부록 A-3·B-6, §6 단계 0-6).

fold 1~3·refit 마다 train 행(`lst.folds.make_train_rows`, C = T 직전 06-30)과 채점 쌍(`lst.folds.scoring_pairs`,
C = 그 fold(refit) train 끝)에 HIS1~HIS4 칼럼(k 5값 전부)을 만들어 결측 수·범위·누수 점검만 기록한다.
큰 행렬 파일(칼럼 값)은 만들지 않음 — 칼럼은 학습 때 `lst.history.history_features` 로 다시 만듦.

점검(부록 B-6 의 이력 관련 항목):
  - 집계 시료 d ≤ C(C 마다 실제로 모은 시료 d 최대), 모든 행 d > C(자기 시료 제외), train 행 C ≤ train 끝,
    채점 쌍 C = train 끝 = 'T 직전 06-30'.
  - 모집단을 그 fold(refit) train(d ≤ train 끝)으로 잘라 씀(validation·test 정답은 집계에 못 들어감).
  - 정의대로 다시 계산(`lst.history.Reference`, 다른 코드 경로)한 표본 행과 같은지(다르면 멈춤).
  - 'train 엔 NaN 없고 평가에만 있는 칼럼'·'train 에서 전부 NaN 인데 평가에는 값이 있는 칼럼'(평가 = 채점 쌍).
  - 해외 행을 뺀 모집단으로도 국내 행의 시군구·단위 검사 수(HIS2 sgg·u)가 같은지(전국·p0 칼럼 제외).
채점 쌍 칼럼은 시료마다 같은 값(C 가 모든 h 같음)이라 채점 시료마다 한 번 계산하고, 쌍 수는 시료별 쌍 수로 셈.
정답(y)은 집계(d ≤ train 끝)에만 쓰고 채점은 하지 않음 — test 기간(d > 2025-06-30) y 는 읽지도 않음.

실행(서버, 검수 뒤 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_6_history.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-6_history
출력(<out>/, 이미 있으면 멈춤): history_columns.csv · history_stats.csv · history_check.json · run_meta.json

구현 세부(계획서에 없음 — 담당자 확인): 표본 대조 행 수·씨앗(설정 spot_check_rows·spot_check_seed),
  범위 통계 = 최소·최대·평균(train 행 단위, 채점 쌍은 쌍 가중), 그 밖 lst/history.py docstring.
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
from lst import history as hi  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = {
    "plan": cf.Field(str),
    "samples": cf.Field(str),            # --inputs 아래 상대 경로(단계 0-1 결과)
    "k_values": cf.Field(list, item_types=int),
    "splits": cf.Field(list, item_types=str),
    "spot_check_rows": cf.Field(int),
    "spot_check_seed": cf.Field(int),
}
SAMPLE_COLS = ("lims_data_key", "y", "src_route", "sgg", "sido", "p2", "p3", "d", "h")
OUT_NAMES = ("history_columns.csv", "history_stats.csv", "history_check.json", "run_meta.json")


def load_cfg(path) -> dict:
    cfg = dict(cf.load_config(path, SCHEMA))
    if tuple(cfg["k_values"]) != hi.K_VALUES:
        raise cf.ConfigError(f"k_values 는 계획서 단계 2-1 의 {list(hi.K_VALUES)} 그대로")
    cf.check_columns(cfg["splits"], fd.SPLITS, "판")
    if len(set(cfg["splits"])) != len(cfg["splits"]) or cfg["spot_check_rows"] < 1:
        raise cf.ConfigError("splits 중복 또는 spot_check_rows < 1")
    return cfg


def read_samples(path: Path) -> pd.DataFrame:
    """y 는 d ≤ 2025-06-30 행만 읽음(test 정답 안 읽음 — test 행 y = <NA>, 모집단에 들어가면 멈춤)."""
    return hi.read_samples_blind_test(path, SAMPLE_COLS).reset_index(drop=True)


def col_stats(F: pd.DataFrame, w: np.ndarray | None = None) -> pd.DataFrame:
    """칼럼마다 NaN 수(가중이면 쌍 수)·최소·최대·평균."""
    X = F.to_numpy(float)
    nan = np.isnan(X)
    ww = np.ones(X.shape[0]) if w is None else w.astype(float)
    with np.errstate(invalid="ignore"):
        mn = np.where(nan.all(0), np.nan, np.nanmin(np.where(nan, np.inf, X), 0))
        mx = np.where(nan.all(0), np.nan, np.nanmax(np.where(nan, -np.inf, X), 0))
        Xz = np.where(nan, 0.0, X)
        wsum = (ww[:, None] * ~nan).sum(0)
        mean = np.where(wsum > 0, (ww[:, None] * Xz).sum(0) / np.where(wsum > 0, wsum, 1), np.nan)
    return pd.DataFrame({"column": F.columns, "n": float(ww.sum()), "n_nan": (ww[:, None] * nan).sum(0),
                         "n_nan_rows": nan.sum(0), "min": mn, "max": mx, "mean": mean})


def spot_check(ref: hi.Reference, rows: pd.DataFrame, F: pd.DataFrame, n: int, rng) -> dict:
    pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
    worst = 0.0
    for i in pick:
        exp = ref.row(rows.iloc[i])
        got = F.iloc[i]
        for c, v in exp.items():
            g = got[c]
            if np.isnan(v) != np.isnan(g):
                raise AssertionError(f"표본 대조 NaN 다름: 행 {i} {c} 기대 {v} 계산 {g}")
            if not np.isnan(v):
                diff = abs(g - v) / max(1.0, abs(v))
                worst = max(worst, diff)
                if diff > 1e-10:
                    raise AssertionError(f"표본 대조 다름: 행 {i} {c} 기대 {v} 계산 {g}")
    return {"rows_checked": int(len(pick)), "max_rel_diff": worst}


def run_split(S: pd.DataFrame, sp: fd.Split, ks, n_spot: int, rng) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    d_all = fd.to_days(S["d"])
    pop_df = S[d_all <= sp.train_end].reset_index(drop=True)
    pop = hi.Population.from_samples(pop_df)
    ref = hi.Reference(pop_df)
    rec: dict = {"population_n": int(len(pop_df)), "population_d_max": str(fd.as_date([fd.to_days(pop_df["d"]).max()])[0])}

    # ---- train 행
    tr = fd.make_train_rows(S, sp)
    Ctr = hi.c_for_train_rows(tr["T"], sp)
    tr["C"] = fd.as_date(Ctr)
    d_tr = fd.to_days(tr["d"])
    F_tr = hi.history_features(pop, tr, ks)
    st_tr = col_stats(F_tr).assign(split=sp.name, part="train")
    cvals, ccnt = np.unique(Ctr, return_counts=True)
    rec["train"] = {"rows": int(len(tr)), "C_counts": {str(fd.as_date([c])[0]): int(n) for c, n in zip(cvals, ccnt)},
                    "min_d_minus_C": int((d_tr - Ctr).min()), "C_max": str(fd.as_date([Ctr.max()])[0]),
                    "C_le_train_end": bool((Ctr <= sp.train_end).all()), "d_gt_C": bool((d_tr > Ctr).all()),
                    "spot_check": spot_check(ref, tr, F_tr, n_spot, rng)}

    # ---- 채점 쌍(시료마다 한 번 계산, 쌍 수로 가중)
    pairs = fd.scoring_pairs(d_all, sp)
    Cp = hi.c_for_eval_rows(pairs["T"], sp)  # 쌍마다 C = train 끝 = T 직전 06-30 확인
    rows, w = np.unique(pairs["row"].to_numpy(), return_counts=True)
    ev = S.iloc[rows].copy()
    ev["C"] = fd.as_date(np.full(len(ev), sp.train_end))
    F_ev = hi.history_features(pop, ev, ks)
    st_ev = col_stats(F_ev, w).assign(split=sp.name, part="eval_pairs")
    d_ev = fd.to_days(ev["d"])
    rec["eval"] = {"pairs": int(len(pairs)), "samples": int(len(ev)), "C": str(fd.as_date([sp.train_end])[0]),
                   "all_pairs_C_eq_train_end": bool((Cp == sp.train_end).all()),
                   "min_d_minus_C": int((d_ev - sp.train_end).min()), "d_gt_C": bool((d_ev > sp.train_end).all()),
                   "spot_check": spot_check(ref, ev, F_ev, n_spot, rng)}

    # ---- 집계 시료 d ≤ C(Population.audit)
    au = pd.DataFrame(pop.audit)
    g = au.groupby("C")["d_max_used"].max()
    rec["aggregate_d_max_by_C"] = {str(fd.as_date([c])[0]): str(fd.as_date([v])[0]) for c, v in g.items()}
    rec["aggregate_d_le_C"] = bool((au["d_max_used"] <= au["C"]).all())
    rec["aggregate_d_le_train_end"] = bool((au["d_max_used"] <= sp.train_end).all())

    # ---- B-6: train 엔 NaN 없고 평가에만 있음 / train 전부 NaN 인데 평가엔 값
    tr_nan = F_tr.isna().to_numpy()
    ev_nan = F_ev.isna().to_numpy()
    ev_nan_pairs = (w[:, None] * ev_nan).sum(0)
    only_eval = [{"column": c, "eval_nan_pairs": int(ev_nan_pairs[j])} for j, c in enumerate(F_tr.columns)
                 if not tr_nan[:, j].any() and ev_nan[:, j].any()]
    tr_allnan = [{"column": c, "eval_value_pairs": int((w * ~ev_nan[:, j]).sum())} for j, c in enumerate(F_tr.columns)
                 if tr_nan[:, j].all() and (~ev_nan[:, j]).any()]
    rec["b6_nan_only_in_eval"] = only_eval
    rec["b6_all_nan_in_train_but_eval_has_values"] = tr_allnan

    # ---- B-6: 해외 행을 빼도 국내 행 시군구·단위 검사 수 같음
    dom_cols = [c for c in F_tr.columns if c.startswith("hist_logn") and (c.endswith("_sgg") or c.endswith("_u"))]
    pop_dom = hi.Population.from_samples(pop_df[pop_df["src_route"] == "domestic"])
    same = {}
    excluded = {}
    for part, rows_df, F, Crow in (("train", tr, F_tr, np.asarray(Ctr)),
                                   ("eval", ev, F_ev, np.full(len(ev), sp.train_end))):
        m = (rows_df["src_route"] == "domestic").to_numpy()
        Fd = hi.history_features(pop_dom, rows_df[m], ks)[dom_cols]
        a, b = Fd.to_numpy(float), F[dom_cols].to_numpy(float)[m]
        # §5-1 'n_all = 0 이면 NaN' 은 전체(국내 + 해외) 검사 수에 기대는 규칙 — 부록 B-6 대상('해외에 안 기대는 입력만') 밖.
        # 국내만 모집단의 n_all = 0 인 칸(예: C = 2015-06-30 — 그때까지 국내 시료 0, 해외 24)은 비교에서 빼고 수만 기록.
        keep = np.ones(a.shape, dtype=bool)
        cm = Crow[m]
        for j, c in enumerate(dom_cols):
            _, n_dom = pop_dom.totals(cm, "5y" if "5y" in c else "1y")
            keep[:, j] = n_dom > 0
        excluded[part] = int((~keep).sum())
        a2, b2 = np.where(keep, a, 0.0), np.where(keep, b, 0.0)
        same[part] = bool(np.array_equal(np.isnan(a2), np.isnan(b2)) and np.allclose(a2, b2, rtol=0, atol=0, equal_nan=True))
    rec["b6_domestic_sgg_u_counts_without_overseas_same"] = same
    rec["b6_excluded_cells_domestic_n_all_zero"] = excluded
    if not all(same.values()):
        raise AssertionError(f"{sp.name}: 해외 행을 빼면 국내 시군구·단위 검사 수가 바뀜 {same}")
    for key in ("C_le_train_end", "d_gt_C"):
        if not rec["train"][key]:
            raise AssertionError(f"{sp.name}: train {key} 실패")
    if not (rec["eval"]["d_gt_C"] and rec["eval"]["all_pairs_C_eq_train_end"] and rec["aggregate_d_le_C"]
            and rec["aggregate_d_le_train_end"]):
        raise AssertionError(f"{sp.name}: 누수 점검 실패 {rec}")
    rec["seconds"] = round(time.time() - t0, 1)
    return pd.concat([st_tr, st_ev], ignore_index=True), rec


def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="단계 0 결과 뿌리(서버 ~/cyj_listeria_final/results)")
    ap.add_argument("--config", default=str(CODE_ROOT / "configs" / "s0_6_history.json"))
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = {n: rp.result(n) for n in OUT_NAMES}
    exist = [str(p) for p in out.values() if p.exists()]
    if exist:
        raise FileExistsError(f"이미 있는 결과 — 덮지 않음: {exist}")
    res0 = check_resources(rp.out)
    S = read_samples(resolve(inputs_root, cfg["samples"]))
    ks = tuple(cfg["k_values"])
    rng = np.random.default_rng(cfg["spot_check_seed"])
    stats, check = [], {"samples_rows": int(len(S)), "k_values": list(ks), "splits": {}}
    for name in cfg["splits"]:
        check_resources(rp.out)
        st, rec = run_split(S, fd.SPLITS[name], ks, cfg["spot_check_rows"], rng)
        stats.append(st)
        check["splits"][name] = rec
        print(f"{name}: train {rec['train']['rows']}행 · 채점 쌍 {rec['eval']['pairs']} · {rec['seconds']}초")
    cols = pd.DataFrame(hi.his_columns(ks))
    cols["formula"] = cols["id"].map({"HIS1": "(s + k·p0) ÷ (n + k)", "HIS2": "log(1 + n), 창이 자료 시작 전이면 n × 기간 일수 ÷ 실제 일수",
                                      "HIS3": "ln((s+0.5)/(s_all+0.5)) − ln((n−s+0.5)/(n_all−s_all+0.5))",
                                      "HIS4": "√(m(1−m) ÷ (n + k + 1)), m = 1년 원 검출률(n = 0 이면 p0)"})
    with open(out["history_columns.csv"], "x", newline="", encoding="utf-8-sig") as f:
        cols.to_csv(f, index=False)
    st = pd.concat(stats, ignore_index=True)[["split", "part", "column", "n", "n_nan", "n_nan_rows", "min", "max", "mean"]]
    with open(out["history_stats.csv"], "x", newline="", encoding="utf-8-sig") as f:
        st.to_csv(f, index=False)
    check["seconds"] = round(time.time() - t0, 1)
    cf.dump_json(out["history_check.json"], cf.nan_to_null(check))
    cf.save_run_meta(out["run_meta.json"], cfg, cf.inputs_md5(inputs_root, [cfg["samples"]]), CODE_ROOT, n_jobs=None,
                     extra={"outputs": list(OUT_NAMES[:-1]), "config_path": Path(args.config).name,
                            "inputs_root_arg": "--inputs", "resources_at_start": res0})
    print(f"끝: {rp.out} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
