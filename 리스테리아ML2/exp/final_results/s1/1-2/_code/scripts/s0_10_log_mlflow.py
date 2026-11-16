# -*- coding: utf-8 -*-
"""s0_10_log_mlflow.py — 단계 0-10(7-a 타겟 인코딩 단독) 결과를 MLflow `listeria-준비` 에 올림(계획서 부록 A-15, A-12, B-3).

A-15: 7-a = `listeria-준비` 의 단계 run '0-10'(격자는 run 하나 — 격자 표 artifact, 고른 칸만 판 run). 판 run tag =
역할 '기준 줄'·판정 '해당 없음'. CV 지표만(test 채점은 1-1 보고 때 같은 run 에 — s1_1_trees.py --report).
B-3: 저장 예측(te7a_cv_pred)을 펼쳐 다시 채점한 CV 가 te7a_selected.json 의 fold 별 값과 같은지 assert.
실험 로그: '같은 validation 으로 고른 횟수' 에 7-a 격자 75 칸을 한 기록으로 셈(부록 A-12).

실행(서버, 0-10 뒤 — MLflow 서버가 떠 있을 때):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_10_log_mlflow.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-10_mlflow --config configs/s0_10_log_mlflow.json
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = {
    "plan": cf.Field(str),
    "samples": cf.Field(str),
    "dir_0_10": cf.Field(str),
    "experiment_log": cf.Field(str),
    "mlflow_uri": cf.Field(str),
    "grid_size": cf.Field(int),
}
EXPERIMENT = "listeria-준비"
STAGE = "0-10"


def log_0_10(dir_0_10: Path, samples: ev.SampleTable, tracker, exp_log: rn.ExperimentLog, grid_size: int = 75,
             code_root: Path = CODE_ROOT) -> dict:
    """0-10 결과 → MLflow(단계 run 0-10 + 고른 칸 판 run). 돌려줌 = {name, run_id, cv, selection_count}."""
    sel = rn.read_json(dir_0_10 / "te7a_selected.json")
    name = sel["selected"]
    grid = pd.read_csv(dir_0_10 / "te7a_grid_cv.csv", encoding="utf-8-sig")
    if len(grid) != grid_size:
        raise AssertionError(f"7-a 격자 {len(grid)} ≠ {grid_size}")
    if sel.get("test_scored"):
        raise AssertionError("0-10 은 CV 만 — test 가 채점돼 있음")
    base = rn.te7a_baseline(dir_0_10, samples)
    cv = ev.score_cv(base.cv_preds, samples)
    for f, rep in sel["fold_reports"].items():   # B-3 재채점 assert
        a, b = cv["fold_band_weighted"][f], rep["pr_auc_band_weighted"]
        if not (math.isclose(a, b, rel_tol=0, abs_tol=1e-12)):
            raise AssertionError(f"{f}: 저장 예측 재채점 {a} ≠ 0-10 기록 {b}")
    sr = tracker.stage_run(EXPERIMENT, STAGE)
    if tracker.find_run(EXPERIMENT, name, sr) is not None:
        raise FileExistsError(f"이미 올린 판 run: {name}")
    count = exp_log.cv_scored(STAGE, name, n_settings=grid_size, kind="7-a 격자")
    for f in ("te7a_grid_cv.csv", "te7a_selected.json", "run_meta.json"):
        if (dir_0_10 / f).exists():
            tracker.log_artifact(sr, dir_0_10 / f, "grid")
    rid = tracker.start_variant(EXPERIMENT, sr, name, {"단계": STAGE, "역할": "기준 줄"},
                                {"algorithm": "7-a", "length": sel["length"], "k_u": sel["k_u"], "k_p3": sel["k_p3"],
                                 "train_count": 0, "seed": 42})
    tracker.set_judgement(rid, "해당 없음")
    tracker.log_cv(rid, cv, selection_count=count)
    for p in sorted(dir_0_10.glob(f"te7a_*_pred_{name}.parquet")):
        tracker.log_artifact(rid, p, "predictions")
    tracker.finish(rid)
    tracker.finish(sr)
    exp_log.append("mlflow_logged", stage=STAGE, name=name, run_id=rid)
    return {"name": name, "run_id": rid, "cv": cv["cv_band_weighted"], "selection_count": count}


def main(argv=None):
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, SCHEMA))
    inputs_root = Path(args.inputs).expanduser().resolve()
    from lst.tracking import Tracker

    res = log_0_10(resolve(inputs_root, cfg["dir_0_10"]), ev.SampleTable.read(resolve(inputs_root, cfg["samples"])),
                   Tracker(cfg["mlflow_uri"]), rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])),
                   cfg["grid_size"])
    cf.dump_scores_json(rp.result("mlflow_logged.json"), res)
    print(res)
    return res


if __name__ == "__main__":
    main()
