# -*- coding: utf-8 -*-
"""과업 2 실행.

python src/run.py --stage step1|step2|tables|all [--smoke] [--only 실행이름]
  step1 : 기상 100일(시험 실행은 20일)로 XGBoost/LightGBM/CatBoost x (summary, raw) + LSTM = 7회
  step2 : step1 에서 val 60일 평균 PR-AUC 최고 (모델, 입력 방식) 으로 L = 10..100 (L=100 은 재사용)
  tables: 비교표·baseline 1~60일 상세표·selection.json·xlsx
  all   : step1 -> step2 -> tables
결과가 완전한 실행(summary.json complete=True + 파일 4개)은 건너뛴다(이어하기).
test 는 기록만 하고 선정에 쓰지 않는다.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config as C
import data as D
import metrics as M
from common import Ctx, Logger, MemSampler, install_tee, make_runner

RUN_FILES = ["config.json", "metrics_by_h.csv", "pred_val.parquet", "pred_test.parquet", "summary.json"]
MODEL_DISPLAY = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "lstm": "LSTM"}
INPUT_DISPLAY = {"summary": "요약(summary)", "raw": "일별 원값(raw)", "seq": "시간순 시퀀스(seq)"}
# 지침 1-1(불균형 문제를 어떻게 했는지) 표기
IMBALANCE_NOTE = "불균형 처리 없음 - 가중치·샘플링·pos_weight 모두 쓰지 않음(사용자 지시)"
TABLE_NOTE = ("모델비교·window비교의 test Precision·Recall·F2·TP·FN·FP·TN 은 h 1~60 예측을 모두 합친 혼동행렬 기준"
              "(n = test 행 수 x 60). baseline_1_60일 은 h 별 값")


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_complete(rdir):
    if not all((rdir / f).exists() for f in RUN_FILES):
        return False
    try:
        return bool(jload(rdir / "summary.json").get("complete"))
    except Exception:
        return False


def write_preds(path, ctx, sp, P):
    v = ctx.sp[sp]
    idx, n = v["idx"], len(v["idx"])
    rows = ctx.rows.iloc[idx]
    cols = {}
    for name, cidx, lev in (("sgg_code", 1, "sgg"), ("train_prdlst_code", 2, "p2"), ("train_prdlst_code_2", 3, "p3")):
        codes = np.tile(v["cats"][:, cidx], C.H_MAX).astype(np.int32)
        cols[name] = pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(ctx.levels[lev], type=pa.string()))
    cols["d"] = pa.array(np.tile(rows["d"].to_numpy().astype("datetime64[D]"), C.H_MAX), type=pa.date32())
    cols["h"] = pa.array(np.repeat(np.arange(1, C.H_MAX + 1, dtype=np.int8), n))
    cols["y"] = pa.array(np.tile(v["y"].astype(np.int8), C.H_MAX))
    cols["p"] = pa.array(P.reshape(-1).astype(np.float32))
    pq.write_table(pa.table(cols), path, compression="zstd")


def run_one(name, ctx, P, log):
    model, inp, L = C.parse_run_name(name)
    rdir = P["results"] / name
    if is_complete(rdir):
        log(f"[건너뜀] {name}: 결과 완전")
        return jload(rdir / "summary.json")
    if rdir.exists():
        log(f"  {name}: 불완전한 이전 결과 삭제 후 다시 실행")
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    log(f"[시작] {name}")
    t_all = time.time()
    win_lo, win_hi = ctx.check_all_windows(L)
    log(f"  창 확인 통과: 모든 창이 기상표 안 ({win_lo} ~ {win_hi})")
    mem = MemSampler()
    mem.start()
    runner = make_runner(ctx, model, inp, L, log)
    unit = "에폭" if model == "lstm" else "라운드"
    try:
        runner.fit()
        log(f"  학습 끝: 최적 {unit} {runner.best_rounds} (학습한 수 {runner.n_rounds_trained}), "
            f"조기종료 val AP {runner.best_val_ap:.5f}, 학습 {runner.fit_seconds:.1f}s")
        t_ev = time.time()
        preds = {}
        for sp in ("val", "test"):
            n = len(ctx.sp[sp]["idx"])
            Pm = np.empty((C.H_MAX, n), dtype=np.float32)
            for h in range(1, C.H_MAX + 1):
                Pm[h - 1] = runner.predict(sp, h)
                if h % 20 == 0:
                    log(f"    평가 {sp} h={h}/{C.H_MAX} ({time.time() - t_ev:.0f}s)")
            preds[sp] = Pm
        eval_seconds = time.time() - t_ev
        info = runner.info()
    finally:
        runner.close()
    memd = mem.stop()

    # ---- 지표
    recs = []
    for sp in ("val", "test"):
        y = ctx.sp[sp]["y"]
        for h in range(1, C.H_MAX + 1):
            p = preds[sp][h - 1]
            recs.append({"split": sp, "h": h, "n": len(y), "pos": int(y.sum()),
                         "pr_auc": M.pr_auc(y, p), "roc_auc": M.roc_auc(y, p)})
    yv_all = np.tile(ctx.sp["val"]["y"], C.H_MAX)
    thr = M.f2_threshold(yv_all, preds["val"].reshape(-1))
    for r in recs:
        y = ctx.sp[r["split"]]["y"]
        cm = M.confusion(y, preds[r["split"]][r["h"] - 1], thr["threshold"])
        r.update({"threshold": thr["threshold"], **{k: cm[k] for k in
                  ("tp", "fn", "fp", "tn", "accuracy", "precision", "recall", "f1", "f2")}})
    mdf = pd.DataFrame(recs)
    overall = {sp: M.confusion(np.tile(ctx.sp[sp]["y"], C.H_MAX), preds[sp].reshape(-1), thr["threshold"])
               for sp in ("val", "test")}
    mean = {f"{sp}_{k}_mean60": float(mdf.loc[mdf.split == sp, k].mean())
            for sp in ("val", "test") for k in ("pr_auc", "roc_auc")}

    cfg = {"run": name, "model": model, "input": inp, "L": L, "smoke": ctx.smoke,
           "alignment": "목표일 d, 앞당김 h -> T = d - h, 기상 창 = T-L .. T-1 (T 당일 제외)",
           "splits": {"train": f"d <= {C.TRAIN_END}", "val": f"{C.VAL_START}~{C.VAL_END}",
                      "test": f"{C.TEST_START}~{C.TEST_END}"},
           "rows": {sp: {"n": len(v["idx"]), "pos": int(v["y"].sum())} for sp, v in ctx.sp.items()},
           "h_assignment": "train: 행(단위 키+d 정렬)마다 default_rng(42).integers(1,61); 조기종료 val: default_rng(43); 최종 평가 val/test: h=1..60 전부",
           "inputs": "범주 sido, sgg, p2, p3 + h + 기상 특성 (그 밖의 입력 없음)",
           "n_features": runner.n_features, "n_features_desc": runner.n_features_desc,
           "feature_columns": getattr(runner, "cols", None), "window_range": [win_lo, win_hi],
           "model_info": info, "imbalance_handling": "없음(가중치·샘플링·pos_weight 없음)", "seed": C.SEED}
    jdump(cfg, rdir / "config.json")
    mdf.to_csv(rdir / "metrics_by_h.csv", index=False, encoding="utf-8-sig")
    for sp in ("val", "test"):
        write_preds(rdir / f"pred_{sp}.parquet", ctx, sp, preds[sp])
    summ = {"run": name, "model": model, "input": inp, "L": L, "smoke": ctx.smoke,
            **mean, "threshold_rule": "val 예측(h 1~60 합침)에서 F2 최대 확률값, 판정 p >= 기준값",
            **thr, "test_overall": overall["test"], "val_overall": overall["val"],
            "best_rounds_or_epoch": runner.best_rounds, "n_rounds_trained": runner.n_rounds_trained,
            "early_stop_val_ap": runner.best_val_ap, "fit_seconds": round(runner.fit_seconds, 2),
            "feature_seconds": round(runner.feature_seconds, 2), "eval_seconds": round(eval_seconds, 2),
            "total_seconds": round(time.time() - t_all, 2), "n_features": runner.n_features, **memd,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": True}
    jdump(summ, rdir / "summary.json")
    log(f"[끝] {name}: val PR-AUC(60일 평균) {mean['val_pr_auc_mean60']:.5f}, test {mean['test_pr_auc_mean60']:.5f}, "
        f"기준값 {thr['threshold']:.5f}, 총 {summ['total_seconds']:.0f}s, 최대 RSS {memd['peak_rss_gb']} GB")
    return summ


def select_step1(P, smoke, log):
    names = C.step1_runs(smoke)
    missing = [n for n in names if not is_complete(P["results"] / n)]
    if missing:
        raise RuntimeError(f"step1 결과가 완전하지 않음: {missing}")
    sums = [jload(P["results"] / n / "summary.json") for n in names]
    # 같은 값이면 고정 실행 순서 앞쪽 (sorted 는 안정 정렬)
    order = sorted(range(len(sums)), key=lambda i: -sums[i]["val_pr_auc_mean60"])
    best = sums[order[0]]
    sel = {"rule": "step1 7회 중 val 60일 평균 PR-AUC 최대 (test 미사용). 같은 값이면 실행 순서 앞쪽",
           "ranking": [{"run": sums[i]["run"], "val_pr_auc_mean60": sums[i]["val_pr_auc_mean60"]} for i in order],
           "selected_model": best["model"], "selected_input": best["input"], "selected_run": best["run"]}
    jdump(sel, P["results"] / "selection_step1.json")
    log(f"step1 선정: {best['model']} / {best['input']} (val PR-AUC {best['val_pr_auc_mean60']:.5f})")
    return sel


def table_row(s):
    t = s["test_overall"]
    return {"모델": MODEL_DISPLAY[s["model"]], "기상_입력_방식": INPUT_DISPLAY[s["input"]], "L": s["L"],
            "특성_수": s["n_features"],
            "val_PR-AUC_60일평균": s["val_pr_auc_mean60"], "test_PR-AUC_60일평균": s["test_pr_auc_mean60"],
            "val_ROC-AUC_60일평균": s["val_roc_auc_mean60"], "test_ROC-AUC_60일평균": s["test_roc_auc_mean60"],
            # test 지표·혼동행렬은 h 1~60 예측을 모두 합친 것(행 수 x 60). 양식 docx 의 '60일 평균' 과 단위가 다름
            "기준값(val_F2최대)": s["threshold"], "test_Precision(h1~60 전체)": t["precision"],
            "test_Recall(h1~60 전체)": t["recall"], "test_F2(h1~60 전체)": t["f2"],
            "test_TP(h1~60 합계)": t["tp"], "test_FN(h1~60 합계)": t["fn"], "test_FP(h1~60 합계)": t["fp"],
            "test_TN(h1~60 합계)": t["tn"],
            "최적_라운드/에폭": s["best_rounds_or_epoch"], "학습_시간_초": s["fit_seconds"], "실행이름": s["run"]}


def build_tables(P, smoke, log):
    res, tab = P["results"], P["tables"]
    tab.mkdir(parents=True, exist_ok=True)
    sel1 = select_step1(P, smoke, log)
    step1 = pd.DataFrame([table_row(jload(res / n / "summary.json")) for n in C.step1_runs(smoke)])
    m, i = sel1["selected_model"], sel1["selected_input"]
    wnames = [C.run_name(m, i, L) for L in C.step2_Ls(smoke)]
    missing = [n for n in wnames if not is_complete(res / n)]
    if missing:
        raise RuntimeError(f"step2 결과가 완전하지 않음: {missing}")
    wsum = [jload(res / n / "summary.json") for n in wnames]
    window = pd.DataFrame([table_row(s) for s in wsum])
    # 같은 값이면 짧은 L (np.argmax 첫 번째)
    best_i = int(np.argmax([s["val_pr_auc_mean60"] for s in wsum]))
    Lstar = wsum[best_i]["L"]
    base = wsum[best_i]["run"]
    mdf = pd.read_csv(res / base / "metrics_by_h.csv")
    tdf = mdf[mdf.split == "test"].sort_values("h")
    detail = pd.DataFrame({"예측 거리": [f"{h}일" for h in tdf.h], "accuracy": tdf.accuracy.values,
                           "precision": tdf.precision.values, "recall": tdf.recall.values, "f1": tdf.f1.values,
                           "roc-auc": tdf.roc_auc.values, "pr-auc": tdf.pr_auc.values, "tp": tdf.tp.values,
                           "fn": tdf.fn.values, "fp": tdf.fp.values, "tn": tdf.tn.values, "n": tdf.n.values})
    ds_path = res / "data_summary.csv"
    ds = pd.read_csv(ds_path) if ds_path.exists() else D.data_summary(D.load_all(smoke)[0])
    sel = {"stage1": sel1,
           "stage2": {"rule": "선정 (모델, 입력 방식) 의 L=10..100 중 val 60일 평균 PR-AUC 최대 (test 미사용). 같은 값이면 짧은 L",
                      "ranking": sorted([{"run": s["run"], "L": s["L"], "val_pr_auc_mean60": s["val_pr_auc_mean60"]}
                                         for s in wsum], key=lambda r: -r["val_pr_auc_mean60"]),
                      "L_star": Lstar},
           "baseline": {"run": base, "model": MODEL_DISPLAY[m], "input": INPUT_DISPLAY[i], "L": Lstar,
                        "val_pr_auc_mean60": wsum[best_i]["val_pr_auc_mean60"],
                        "test_pr_auc_mean60": wsum[best_i]["test_pr_auc_mean60"],
                        "threshold": wsum[best_i]["threshold"]},
           "imbalance_handling": IMBALANCE_NOTE, "table_note": TABLE_NOTE,
           "note": "선정은 val 만 사용. test 는 기록용.", "smoke": smoke}
    step1.to_csv(tab / "모델비교_100일.csv", index=False, encoding="utf-8-sig")
    window.to_csv(tab / "window비교.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(tab / "baseline_1_60일.csv", index=False, encoding="utf-8-sig")
    jdump(sel, tab / "selection.json")
    flat = [("1단계 규칙", sel1["rule"]), ("1단계 선정 모델", MODEL_DISPLAY[m]), ("1단계 선정 입력", INPUT_DISPLAY[i])]
    flat += [(f"1단계 순위 {k + 1}", "{} (val PR-AUC {:.6f})".format(r["run"], r["val_pr_auc_mean60"]))
             for k, r in enumerate(sel1["ranking"])]
    flat += [("2단계 규칙", sel["stage2"]["rule"]), ("L*", Lstar), ("baseline", base)]
    flat += [(f"2단계 순위 {k + 1}", "{} (val PR-AUC {:.6f})".format(r["run"], r["val_pr_auc_mean60"]))
             for k, r in enumerate(sel["stage2"]["ranking"])]
    flat += [("불균형 처리", IMBALANCE_NOTE), ("표 test 지표 기준", TABLE_NOTE), ("비고", sel["note"])]
    with pd.ExcelWriter(tab / "task2_결과표.xlsx", engine="openpyxl") as xw:
        ds.to_excel(xw, sheet_name="데이터요약", index=False)
        step1.to_excel(xw, sheet_name="모델비교_100일", index=False)
        window.to_excel(xw, sheet_name="window비교", index=False)
        detail.to_excel(xw, sheet_name="baseline_1_60일", index=False)
        pd.DataFrame(flat, columns=["항목", "값"]).astype(str).to_excel(xw, sheet_name="선정", index=False)
    log(f"표 작성 끝: L* = {Lstar}, baseline = {base} -> {tab}")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["step1", "step2", "tables", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="이 실행 이름만 (해당 단계 실행 목록 안에서)")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--tables-root", default=None)
    ap.add_argument("--console-log", default=None, help="콘솔 출력을 UTF-8 파일에도 씀")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if args.console_log:
        install_tee(args.console_log)
    P = C.paths(args.smoke, args.results_root, args.tables_root)
    for k in ("results", "tables", "logs"):
        P[k].mkdir(parents=True, exist_ok=True)
    log = Logger(P["progress"])
    log(f"===== run.py {' '.join(sys.argv[1:])} | python {sys.version.split()[0]} | pid {os.getpid()}")
    stages = ["step1", "step2", "tables"] if args.stage == "all" else [args.stage]
    ctx = None
    try:
        for st in stages:
            if st in ("step1", "step2"):
                if st == "step1":
                    names = C.step1_runs(args.smoke)
                else:
                    sel = select_step1(P, args.smoke, log)
                    names = [C.run_name(sel["selected_model"], sel["selected_input"], L) for L in C.step2_Ls(args.smoke)]
                if args.only:
                    if args.only not in names:
                        raise ValueError(f"--only {args.only} 는 {st} 실행 목록에 없음: {names}")
                    names = [args.only]
                todo = [n for n in names if not is_complete(P["results"] / n)]
                log(f"--- {st}: 실행 {len(names)}개 중 남은 것 {len(todo)}개: {todo}")
                if ctx is None and (todo or not (P["results"] / "data_summary.csv").exists()):
                    ctx = Ctx(args.smoke, log)
                    D.data_summary(ctx.rows).to_csv(P["results"] / "data_summary.csv", index=False, encoding="utf-8-sig")
                for n in names:
                    run_one(n, ctx, P, log)
            else:
                build_tables(P, args.smoke, log)
        log(f"===== 완료: {args.stage}")
    except Exception:
        log("!!!!! 오류로 멈춤:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()