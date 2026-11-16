# -*- coding: utf-8 -*-
"""s_te_raw.py — 스무딩 없는 과거 검출률 기준 줄(비교만 — 담당자 요청 2026-11-09).

점수 = 7-a(부록 A-3) 식에서 k_u·k_p3 → 0 극한:
  단위 u = (3단계 품목, 시군구 sgg) 의 기간 안 검출 수 ÷ 검사 수(원 검출률, 스무딩 없음).
  그 기간에 그 단위 검사가 없으면 3단계 전국 원 검출률, 그것도 없으면 전체 원 검출률 p0(극한 값 그대로).
기간·기준일 = §5-1(7-a 와 같음): 1년·5년·전체, CV 쌍의 C = 그 fold train 끝, test 쌍의 C = 2025-06-30.
채점 = 0-10·1-1 보고와 같은 함수(trapz PR-AUC·구간 가중·threshold = 합친 CV 에서 Recall ≥ 0.8 중 Precision 최대).
고르기에 쓰지 않는 비교 줄 — test 정답은 보고 허가(--reason)로만 읽고 실험 로그에 test_viewed 를 남김.

결과(--out, 덮지 않음): raw_table.csv, te7a_cv_pred_<판>.parquet·te7a_test_pred_<판>.parquet(정답 없음),
h_metrics_<판>.csv(h 1~60 × 지표, CV·test), run_meta.json.
실행: python scripts/s_te_raw.py --root <천영진> --inputs <results> --out <결과 폴더> --reason "<이유>" [--lengths 5y]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
for p in (str(CODE_ROOT), str(CODE_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import history as hi  # noqa: E402
from lst import metrics as mt  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import check_resources  # noqa: E402
from lst.runner import ExperimentLog  # noqa: E402

import s0_10_te as te  # noqa: E402  (FoldPairs·split_known_test·SAMPLE_COLS 그대로 씀)

SAMPLES = "s0/0-1/apply/samples.parquet"
INCOMPLETE = "s0/0-1/apply/12_incomplete_months.json"
EXP_LOG = "logs/experiment_log.jsonl"


def raw_rate(pop: hi.Population, rows: pd.DataFrame, length: str) -> tuple[np.ndarray, np.ndarray]:
    """원 검출률(스무딩 없음) — 단위 → 3단계 전국 → 전체 순으로 검사 있는 첫 수준. 반환 (값, 출처 'u'/'p3'/'p0')."""
    R = hi._as_rows(rows)
    C = R.C
    s_all, n_all = pop.totals(C, length)
    ok = hi._valid(C, n_all)
    s3, n3 = pop.counts("p3", R.keys["p3"], C, length)
    su, nu = pop.counts("u", R.keys["u"], C, length)
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = s_all / np.where(n_all > 0, n_all, 1)
        p3 = np.where(n3 > 0, s3 / np.where(n3 > 0, n3, 1), p0)
        pu = np.where(nu > 0, su / np.where(nu > 0, nu, 1), p3)
    src = np.where(nu > 0, "u", np.where(n3 > 0, "p3", "p0"))
    return np.where(ok, pu, np.nan), src


def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--reason", required=True, help="test 채점 이유(실험 로그에 남김)")
    ap.add_argument("--lengths", default="1y,5y,all")
    rp, args = parse_paths(argv, ap)
    inputs_root = Path(args.inputs).expanduser().resolve()
    lengths = [x for x in args.lengths.split(",") if x]
    if not set(lengths) <= set(hi.TE_LENGTHS):
        raise ValueError(f"lengths ⊆ {hi.TE_LENGTHS}")
    out_table = rp.result("raw_table.csv")
    if out_table.exists():
        raise FileExistsError(out_table)
    res0 = check_resources(rp.out)
    samples_path = resolve(inputs_root, SAMPLES)
    S = hi.read_samples_blind_test(samples_path, te.SAMPLE_COLS)
    known, test = te.split_known_test(S)
    del S
    if "y" in test.columns:
        raise AssertionError("test 표에 y 가 남아 있음")
    names = {L: f"7-a-raw-{L}" for L in lengths}

    # ---- CV fold 별 시료 예측(정답은 fold 채점 쌍 것만 — FoldPairs)
    cvp = {L: [] for L in lengths}
    src_cnt = {L: {} for L in lengths}
    for f in fd.CV_FOLDS:
        check_resources(rp.out)
        sp = fd.SPLITS[f]
        fp = te.FoldPairs(known, sp)
        pop = hi.Population.from_samples(known[fd.to_days(known["d"]) <= sp.train_end])
        for L in lengths:
            v, src = raw_rate(pop, fp.samples, L)
            if not np.isfinite(v).all():
                raise AssertionError(f"{f} {L}: 값에 NaN")
            cvp[L].append(pd.DataFrame({"fold": f, "lims_data_key": fp.samples["lims_data_key"].to_numpy(),
                                        "d": fp.samples["d"].to_numpy(), "p_hat": v}))
            u, c = np.unique(src, return_counts=True)
            src_cnt[L][f] = {str(a): int(b) for a, b in zip(u, c)}
        if any(a["d_max_used"] > sp.train_end for a in pop.audit):
            raise AssertionError("집계 시료가 train 끝을 넘음")
        print(f"{f}: 시료 {len(fp.rows)} ({time.time() - t0:.0f}초)", flush=True)

    # ---- test 시료 예측(C = 2025-06-30, 모집단 = d ≤ 2025-06-30)
    rf = fd.SPLITS["refit"]
    pairs = fd.scoring_pairs(fd.to_days(test["d"]), rf)
    hi.c_for_eval_rows(pairs["T"], rf)
    rows, n_pairs = np.unique(pairs["row"].to_numpy(), return_counts=True)
    tv = test.iloc[rows].copy()
    tv["C"] = fd.as_date(np.full(len(tv), rf.train_end))
    pop_all = hi.Population.from_samples(known)
    testp = {}
    for L in lengths:
        v, src = raw_rate(pop_all, tv, L)
        if not np.isfinite(v).all():
            raise AssertionError(f"test {L}: 값에 NaN")
        testp[L] = tv[["lims_data_key", "d", "src_route", "sgg", "p2", "p3"]].assign(n_pairs=n_pairs, p_hat=v)
        u, c = np.unique(src, return_counts=True)
        src_cnt[L]["test"] = {str(a): int(b) for a, b in zip(u, c)}

    # ---- 채점(0-10·1-1 보고와 같은 evaluate 함수)
    st = ev.SampleTable.read(samples_path)
    permit = ev.report_permit("7-a-raw", args.reason)
    truth = ev.TestTruth.load(samples_path, permit)
    inc = ev.load_incomplete(resolve(inputs_root, INCOMPLETE))
    recs, files = [], []
    for L in lengths:
        name = names[L]
        cv_df = pd.concat(cvp[L], ignore_index=True)
        for kind, df in (("cv", cv_df), ("test", testp[L])):
            p = rp.result(f"te7a_{kind}_pred_{name}.parquet")
            if p.exists():
                raise FileExistsError(p)
            df.to_parquet(p, index=False)
            files.append(p.name)
        cv_preds = {f: ev.expand_sample_preds(cv_df[cv_df["fold"] == f], st, f) for f in ev.CV_FOLDS}
        test_preds = ev.expand_sample_preds(testp[L], st, "refit")
        cv = ev.score_cv(cv_preds, st)
        thr = cv["threshold"]["threshold"]
        t = ev.score_test(test_preds, st, truth, thr, inc)
        row = ev.table_row(name, f"스무딩 없는 과거 검출률(단위 → 3단계 전국 → 전체, {L}) — 비교만", cv, t,
                           selection_count=None, marks=(), fold_recall=False)
        row["출처 칸 수(단위/3단계/전체)"] = str(src_cnt[L])
        recs.append(row)
        hm = pd.concat([ev.h_metrics_cv(cv_preds, st, thr), ev.h_metrics_test(test_preds, st, truth, thr, inc)],
                       ignore_index=True)
        p = rp.result(f"h_metrics_{name}.csv")
        hm.to_csv(p, index=False, encoding="utf-8-sig")
        files.append(p.name)
        print(f"{name}: CV {cv['cv_band_weighted']:.4f}", flush=True)
    pd.DataFrame(recs).to_csv(out_table, index=False, encoding="utf-8-sig")
    files.append(out_table.name)
    ExperimentLog(resolve(inputs_root, EXP_LOG)).append(
        "test_viewed", stage="7-a-raw", names=list(names.values()), reason=args.reason, tool="s_te_raw.py")
    cf.save_run_meta(rp.result("run_meta.json"), {"lengths": lengths, "reason": args.reason},
                     cf.inputs_md5(inputs_root, [SAMPLES, INCOMPLETE]), CODE_ROOT, n_jobs=None,
                     extra={"outputs": files, "resources_at_start": res0, "seconds": round(time.time() - t0, 1)})
    print(f"끝 {time.time() - t0:.0f}초 — {out_table}")


if __name__ == "__main__":
    main()
