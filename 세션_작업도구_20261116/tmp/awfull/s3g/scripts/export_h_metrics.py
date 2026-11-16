# -*- coding: utf-8 -*-
"""export_h_metrics.py — 단계 판들의 h 1~60 별·지표별 표를 한 파일로(담당자 보고서용).

저장된 예측(pred_*.parquet)과 그 판의 합친 CV threshold(cv_scores.json)만으로 다시 계산(학습 없음 — 이전 코드로 끝난
판도 됨). 지표는 lst.metrics 만(trapz PR-AUC·u_tie 동점 풀기, sklearn 기본 묶음 PR-AUC 는 기록용).
test 칸은 --with-test(보고 허가 — §1-3 test 막기, --reason 필요)일 때만, 덜 찬 달 쌍은 본 점수처럼 뺌.

출력(--out 아래, 있으면 멈춤): h_metrics_<단계>.xlsx — 시트 'long'(판·split·h·지표 긴 표), 'pr_auc_wide'(행 = 판·split,
칼럼 = h01~h60 PR-AUC), 'notes'. openpyxl 이 없으면 같은 이름의 csv 2개(long·wide).

실행(서버):
  ~/cyj_listeria_final/venv/bin/python scripts/export_h_metrics.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/exports --stage 1-1 [--only 판,…] \
      [--with-test --reason "1-1 보고"]
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import pandas as pd  # noqa: E402

from lst import evaluate as ev  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_SAMPLES = "s0/0-1/apply/samples.parquet"
DEFAULT_INCOMPLETE = "s0/0-1/apply/12_incomplete_months.json"


def stage_dir_default(stage: str) -> str:
    """구현 세부: 단계 결과 폴더 = s<단계 앞 번호>/<단계>(예: 1-1 → s1/1-1). 다르면 --stage-dir."""
    return f"s{stage.split('-')[0]}/{stage}"


def variant_tables(stage_dir: Path, names, samples: ev.SampleTable, truth: ev.TestTruth | None,
                   incomplete) -> pd.DataFrame:
    parts = []
    for n in names:
        vdir = rn.variant_dir(stage_dir, n)
        spec, done, cv = rn.load_done(stage_dir, n)
        thr = float(cv["threshold"]["threshold"])
        full = not done.get("smoke")
        t = ev.h_metrics_cv(rn.load_preds(vdir, ev.CV_FOLDS), samples, thr, require_complete=full)
        if truth is not None:
            t = pd.concat([t, ev.h_metrics_test(pd.read_parquet(rn.pred_path(vdir, "refit")), samples, truth, thr,
                                                incomplete, require_complete=full)], ignore_index=True)
        t.insert(0, "variant", n)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def wide_pr_auc(long: pd.DataFrame) -> pd.DataFrame:
    w = long.pivot_table(index=["variant", "split"], columns="h", values="pr_auc", aggfunc="first", dropna=False)
    w.columns = [f"h{int(h):02d}" for h in w.columns]
    return w.reset_index()


def main(argv=None):
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--stage-dir", help="--inputs 아래 단계 결과 폴더(기본 s<앞 번호>/<단계>)")
    ap.add_argument("--only")
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--reason")
    ap.add_argument("--samples", default=DEFAULT_SAMPLES)
    ap.add_argument("--incomplete", default=DEFAULT_INCOMPLETE)
    ap.add_argument("--experiment-log", default="logs/experiment_log.jsonl", help="--inputs 아래 실험 로그")
    rp, args = parse_paths(argv, ap)
    inputs_root = Path(args.inputs).expanduser().resolve()
    sdir = resolve(inputs_root, args.stage_dir or stage_dir_default(args.stage))
    names = sorted(p.name for p in sdir.iterdir() if (p / "done.json").exists())
    if args.only:
        want = [n.strip() for n in args.only.split(",") if n.strip()]
        miss = [n for n in want if n not in names]
        if miss:
            raise KeyError(f"끝난 판이 아님: {miss}")
        names = want
    if not names:
        raise SystemExit("끝난 판 없음")
    samples_path = resolve(inputs_root, args.samples)
    samples = ev.SampleTable.read(samples_path)
    truth, inc = None, set()
    if args.with_test:
        if not args.reason:
            raise SystemExit("--with-test 에는 --reason 이 필요(보고 허가 — §1-3)")
        permit = ev.report_permit(args.stage, args.reason)
        truth = ev.TestTruth.load(samples_path, permit)
        inc = ev.load_incomplete(resolve(inputs_root, args.incomplete))
        rn.ExperimentLog(resolve(inputs_root, args.experiment_log)).append(
            "test_viewed", stage=args.stage, names=list(names), reason=args.reason, tool="export_h_metrics")
    long = variant_tables(sdir, names, samples, truth, inc)
    wide = wide_pr_auc(long)
    notes = pd.DataFrame({"note": [
        f"단계 {args.stage}, 판 {len(names)}개, test {'포함(보고 허가: ' + args.reason + ')' if truth is not None else '없음'}",
        "pr_auc = trapz PR-AUC(동점 u_tie, 주 지표), pr_auc_sklearn_default = 같은 값 묶음(기록용)",
        "lift = 그 h 의 PR-AUC ÷ 그 h 검출 비율, threshold = 그 판의 합친 CV threshold(test 도 같은 값)",
        "검출 0건·전부 검출인 h 의 PR-AUC = NaN, test 는 덜 찬 달 쌍을 뺌",
        "AP(평균 정밀도)는 쓰지 않음"]})
    stem = f"h_metrics_{args.stage}" + ("_with_test" if truth is not None else "")
    try:
        import openpyxl  # noqa: F401

        p = rp.result(stem + ".xlsx")
        if p.exists():
            raise FileExistsError(p)
        with pd.ExcelWriter(p, engine="openpyxl", mode="w") as xw:
            long.to_excel(xw, sheet_name="long", index=False)
            wide.to_excel(xw, sheet_name="pr_auc_wide", index=False)
            notes.to_excel(xw, sheet_name="notes", index=False)
        out = [p]
    except ImportError:
        out = []
        for nm, df in (("long", long), ("pr_auc_wide", wide)):
            p = rp.result(f"{stem}_{nm}.csv")
            with open(p, "x", newline="", encoding="utf-8-sig") as fh:
                df.to_csv(fh, index=False)
            out.append(p)
        print("openpyxl 없음 — csv 로 저장")
    print(f"판 {len(names)}개 → {[str(x) for x in out]}")
    return out


if __name__ == "__main__":
    main()
