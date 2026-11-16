# -*- coding: utf-8 -*-
"""s0_10_te.py — 단계 0-10 타겟 인코딩 단독 판(7-a) 미리 계산, CV 만(계획서 §6 단계 0-10·단계 7, §5-1, 부록 A-3·A-12).

7-a(학습 없음): 단위 (3단계, `sgg`) 스무딩 검출률 p̂_u = (s_u + k_u·p̂_p3) ÷ (n_u + k_u),
p̂_p3 = (s_p3 + k_p3·p0) ÷ (n_p3 + k_p3) 를 그대로 점수로. 격자 = 길이 1년·5년·전체 × k_u × k_p3 ∈ {10, 30, 50, 100, 200}
= 75 칸, 7-a CV(fold 1~3 validation 구간 가중 PR-AUC 평균, `lst.metrics`)로 고름. C 규칙(§5-1)이라 validation 쌍은 모든 h
같은 값(C = 그 fold train 끝).

test: 채점하지 않음(0-10 은 CV 만 — test 채점은 1-1 보고 때). 고른 칸의 test 예측(refit C = 2025-06-30, d ≤ 2025-06-30
시료로 계산)만 시료마다 저장(정답 칼럼 없음). test 막기: 시료 표의 y 는 d ≤ 2025-06-30 행만 읽고(parquet 행 거름),
CV 정답은 d ≤ 2025-06-30 행에서만 읽음(fold 3 채점 시료 d ≤ 2025-05-01 + 60 = 2025-06-30).

고르기(§1-2·부록 A-12): CV 평균 소수 4자리 반올림 뒤 큰 쪽, 같으면 판의 학습 횟수(7-a = 0, 모두 같음) → 판 이름 순.
동점 많음(부록 A-12): 칸마다 세 fold validation 예측을 합쳐 가장 큰 같은 값 묶음 ÷ CV 쌍 ≥ 1% 면 표시(`metrics.tie_summary`),
다시 채점 = u_tie 를 md5(key|h|1) 로 바꾼 점수도 칸마다 기록(1위와 0.001 안 표시), 고른 칸이 동점 많으면 알림.

실행(서버, 검수 뒤 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_10_te.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-10
출력(<out>/, 이미 있으면 멈춤):
  te7a_grid_cv.csv            75 칸 × fold 별 구간 가중 PR-AUC·CV 평균·반올림·순위·다시 채점·동점 요약
  te7a_selected.json          고른 칸·fold 별 score_report(지표 전부)·CV threshold(세 fold validation 합침)·알림
  te7a_cv_pred_<판>.parquet   고른 칸 fold 별 validation 시료 예측(lims_data_key·fold·d·p_hat, 정답 없음)
  te7a_test_pred_<판>.parquet 고른 칸 test 시료 예측(lims_data_key·d·src_route·sgg·p2·p3·n_pairs·p_hat, 정답 없음)
  run_meta.json

구현 세부(계획서에 없음 — 담당자 확인):
  - '1위와 0.001 안' = |CV − CV 1위| ≤ 0.001(반올림 없이). 다시 채점은 모든 칸에 해 둠(기록만, 고르기 규칙은 그대로).
  - test 예측 저장은 고른 칸만(격자의 고르지 않은 값은 test 를 안 매김 — §0). MLflow 기록은 이 스크립트에서 안 함
    (공통 MLflow 모듈이 아직 없음 — 지휘자가 정한 뒤 붙임).
  - 그 밖 lst/history.py docstring(키·'코드없음'·모집단).
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
from lst import metrics as mt  # noqa: E402
from lst.hassign import u_tie_many  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = {
    "plan": cf.Field(str),
    "samples": cf.Field(str),                 # --inputs 아래 상대 경로(단계 0-1 결과)
    "lengths": cf.Field(list, item_types=str),
    "k_u_values": cf.Field(list, item_types=int),
    "k_p3_values": cf.Field(list, item_types=int),
    "cv_folds": cf.Field(list, item_types=str),
    "save_test_predictions": cf.Field(bool),
}
SAMPLE_COLS = ("lims_data_key", "y", "src_route", "sgg", "sido", "p2", "p3", "d", "h")
RESCORE_CLOSE = 0.001


def load_cfg(path) -> dict:
    cfg = dict(cf.load_config(path, SCHEMA))
    if tuple(cfg["lengths"]) != hi.TE_LENGTHS:
        raise cf.ConfigError(f"lengths 는 계획서 단계 7 의 {list(hi.TE_LENGTHS)} 그대로")
    for k in ("k_u_values", "k_p3_values"):
        if tuple(cfg[k]) != hi.K_VALUES:
            raise cf.ConfigError(f"{k} 는 계획서 단계 7 의 {list(hi.K_VALUES)} 그대로")
    if tuple(cfg["cv_folds"]) != fd.CV_FOLDS:
        raise cf.ConfigError(f"cv_folds 는 {list(fd.CV_FOLDS)}")
    return cfg


def split_known_test(S: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """test 막기: y 는 d ≤ refit train 끝 행만 읽었음(read_samples_blind_test) — test 행은 y 칼럼 없는 표로."""
    cut = fd.SPLITS["refit"].train_end
    d = fd.to_days(S["d"])
    known = S[d <= cut].reset_index(drop=True)
    known["y"] = known["y"].astype(np.int8)  # <NA> 가 있으면 여기서 멈춤
    test = S[d > cut].drop(columns=["y"]).reset_index(drop=True)
    return known, test


class FoldPairs:
    """한 fold 의 채점 쌍: 시료 위치·h·y·u_tie(원·다시 채점)."""

    def __init__(self, known: pd.DataFrame, sp: fd.Split):
        d = fd.to_days(known["d"])
        if sp.score_end + 60 > fd.SPLITS["refit"].train_end:
            raise AssertionError("CV 채점 시료가 test 기간에 닿음")
        pairs = fd.scoring_pairs(d, sp)
        hi.c_for_eval_rows(pairs["T"], sp)  # 쌍마다 C = train 끝 확인
        self.rows, self.pos = np.unique(pairs["row"].to_numpy(), return_inverse=True)
        self.samples = known.iloc[self.rows].copy()
        self.samples["C"] = fd.as_date(np.full(len(self.samples), sp.train_end))
        keys = known["lims_data_key"].to_numpy()[pairs["row"].to_numpy()]
        self.h = pairs["h"].to_numpy()
        self.y = known["y"].to_numpy().astype(np.int64)[pairs["row"].to_numpy()]
        self.u = u_tie_many(keys, self.h)
        self.u_rs = u_tie_many(keys, self.h, rescore=True)
        self.n = len(pairs)


def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="단계 0 결과 뿌리(서버 ~/cyj_listeria_final/results)")
    ap.add_argument("--config", default=str(CODE_ROOT / "configs" / "s0_10_te.json"))
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    inputs_root = Path(args.inputs).expanduser().resolve()
    fixed = {"grid": "te7a_grid_cv.csv", "sel": "te7a_selected.json", "meta": "run_meta.json"}
    exist = [str(rp.result(v)) for v in fixed.values() if rp.result(v).exists()] + \
            [str(p) for p in rp.out.glob("te7a_*_pred_*.parquet")]
    if exist:
        raise FileExistsError(f"이미 있는 결과 — 덮지 않음: {exist}")
    res0 = check_resources(rp.out)
    S = hi.read_samples_blind_test(resolve(inputs_root, cfg["samples"]), SAMPLE_COLS)
    known, test = split_known_test(S)
    del S
    L_, KU, KP = cfg["lengths"], cfg["k_u_values"], cfg["k_p3_values"]
    cells = [(L, ku, kp) for L in L_ for ku in KU for kp in KP]
    names = {c: hi.te7a_name(*c) for c in cells}

    # ---- fold 별 점수
    score = {c: {} for c in cells}
    score_rs = {c: {} for c in cells}
    sample_vals: dict[str, dict] = {}
    fps: dict[str, FoldPairs] = {}
    for f in cfg["cv_folds"]:
        check_resources(rp.out)
        sp = fd.SPLITS[f]
        fp = FoldPairs(known, sp)
        fps[f] = fp
        pop = hi.Population.from_samples(known[fd.to_days(known["d"]) <= sp.train_end])
        vals = {}
        for L in L_:
            g = hi.te7a_grid(pop, fp.samples, L, KU, KP)
            for (ku, kp), v in g.items():
                if not np.isfinite(v).all():
                    raise AssertionError(f"{f} {L} ku{ku} kp{kp}: 7-a 점수에 NaN")
                vals[(L, ku, kp)] = v
        if any(a["d_max_used"] > sp.train_end for a in pop.audit):
            raise AssertionError("집계 시료가 train 끝을 넘음")
        sample_vals[f] = vals
        for c in cells:
            s = vals[c][fp.pos]
            score[c][f] = mt.pr_auc_band_weighted(fp.y, s, fp.h, fp.u)
            score_rs[c][f] = mt.pr_auc_band_weighted(fp.y, s, fp.h, fp.u_rs)
        print(f"{f}: 채점 쌍 {fp.n} · 시료 {len(fp.rows)} · 검출 쌍 {int(fp.y.sum())} ({time.time() - t0:.0f}초)")

    # ---- 격자 표
    n_cv = sum(fp.n for fp in fps.values())
    y_pool = np.concatenate([fps[f].y for f in cfg["cv_folds"]])
    recs = []
    for c in cells:
        pooled = np.concatenate([sample_vals[f][c][fps[f].pos] for f in cfg["cv_folds"]])
        ts = mt.tie_summary(pooled, y_pool, n_cv)
        cv = mt.cv_mean(score[c])
        cv_rs = mt.cv_mean(score_rs[c])
        recs.append({"name": names[c], "length": c[0], "k_u": c[1], "k_p3": c[2],
                     **{f"{f}_band_weighted": score[c][f] for f in cfg["cv_folds"]},
                     "cv_mean": cv, "cv_mean_round4": mt.round4(cv),
                     **{f"{f}_band_weighted_rescore": score_rs[c][f] for f in cfg["cv_folds"]},
                     "cv_mean_rescore": cv_rs, "cv_mean_rescore_round4": mt.round4(cv_rs),
                     "train_count": 0,
                     "max_tie_size": ts["max_tie_size"], "max_tie_ratio": ts["max_tie_ratio"],
                     "max_tie_pos_rate": ts["max_tie_pos_rate"], "many_ties": ts["many_ties"]})
    grid = pd.DataFrame(recs)
    if grid["cv_mean"].isna().any():
        raise AssertionError(f"CV 평균 NaN 칸: {grid.loc[grid['cv_mean'].isna(), 'name'].tolist()}")
    grid = grid.sort_values(["cv_mean_round4", "train_count", "name"], ascending=[False, True, True], kind="mergesort")
    grid["rank"] = np.arange(1, len(grid) + 1)
    top = grid.iloc[0]
    grid["within_0.001_of_top"] = (grid["cv_mean"] - top["cv_mean"]).abs() <= RESCORE_CLOSE
    grid["rank_rescore"] = grid["cv_mean_rescore_round4"].rank(ascending=False, method="min").astype(int)
    top = grid.iloc[0]
    with open(rp.result(fixed["grid"]), "x", newline="", encoding="utf-8-sig") as fh:
        grid.to_csv(fh, index=False)

    # ---- 고른 칸: 지표 전부·threshold
    sel = (top["length"], int(top["k_u"]), int(top["k_p3"]))
    sel_name = names[sel]
    reports = {}
    for f in cfg["cv_folds"]:
        fp = fps[f]
        reports[f] = mt.score_report(fp.y, sample_vals[f][sel][fp.pos], fp.h, fp.u)
    pooled_sel = np.concatenate([sample_vals[f][sel][fps[f].pos] for f in cfg["cv_folds"]])
    thr = mt.select_threshold(y_pool, pooled_sel)
    alerts = []
    if bool(top["many_ties"]):
        alerts.append("CV 1위 칸이 '동점 많음(원 예측값 기준)' — 확정 전 알림(부록 A-12)")
    second = grid.iloc[1] if len(grid) > 1 else None
    if second is not None and abs(second["cv_mean"] - top["cv_mean"]) <= RESCORE_CLOSE and (top["many_ties"] or second["many_ties"]):
        alerts.append(f"1위·2위({second['name']}) CV 차 ≤ 0.001 이고 동점 많은 칸 — 다시 채점 순위 기록(rank_rescore)")
    selected = {"selected": sel_name, "length": sel[0], "k_u": sel[1], "k_p3": sel[2],
                "cv_mean_band_weighted": float(top["cv_mean"]), "cv_mean_round4": float(top["cv_mean_round4"]),
                "fold_reports": reports, "cv_threshold_pooled_validation": thr,
                "tie_summary_pooled_validation": mt.tie_summary(pooled_sel, y_pool, n_cv),
                "rank_rescore_of_selected": int(top["rank_rescore"]),
                "selection_count_same_validation": len(cells), "alerts": alerts,
                "test_scored": False, "note": "test 채점은 1-1 보고 때(계획서 §6 단계 0-10)"}
    cf.dump_json(rp.result(fixed["sel"]), cf.nan_to_null(selected))

    # ---- 예측 저장(정답 없음)
    files = [fixed["grid"], fixed["sel"]]
    cvp = pd.concat([pd.DataFrame({"fold": f, "lims_data_key": fps[f].samples["lims_data_key"].to_numpy(),
                                   "d": fps[f].samples["d"].to_numpy(), "p_hat": sample_vals[f][sel]})
                     for f in cfg["cv_folds"]], ignore_index=True)
    p = rp.result(f"te7a_cv_pred_{sel_name}.parquet")
    if p.exists():
        raise FileExistsError(p)
    cvp.to_parquet(p, index=False)
    files.append(p.name)
    if cfg["save_test_predictions"]:
        check_resources(rp.out)
        rf = fd.SPLITS["refit"]
        if "y" in test.columns:
            raise AssertionError("test 표에 y 가 남아 있음")
        pairs = fd.scoring_pairs(fd.to_days(test["d"]), rf)
        hi.c_for_eval_rows(pairs["T"], rf)
        rows, n_pairs = np.unique(pairs["row"].to_numpy(), return_counts=True)
        tv = test.iloc[rows].copy()
        tv["C"] = fd.as_date(np.full(len(tv), rf.train_end))
        pop = hi.Population.from_samples(known)  # d ≤ 2025-06-30 = train+val
        v = hi.te7a_grid(pop, tv, sel[0], [sel[1]], [sel[2]])[(sel[1], sel[2])]
        if not np.isfinite(v).all():
            raise AssertionError("test 7-a 점수에 NaN")
        out_t = tv[["lims_data_key", "d", "src_route", "sgg", "p2", "p3"]].assign(n_pairs=n_pairs, p_hat=v)
        p = rp.result(f"te7a_test_pred_{sel_name}.parquet")
        if p.exists():
            raise FileExistsError(p)
        out_t.to_parquet(p, index=False)
        files.append(p.name)
    cf.save_run_meta(rp.result(fixed["meta"]), cfg, cf.inputs_md5(inputs_root, [cfg["samples"]]), CODE_ROOT, n_jobs=None,
                     extra={"outputs": files, "config_path": Path(args.config).name, "inputs_root_arg": "--inputs",
                            "resources_at_start": res0, "seconds": round(time.time() - t0, 1)})
    print(f"고른 칸 {sel_name}: CV {top['cv_mean']:.4f} · 알림 {alerts} · 끝 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
