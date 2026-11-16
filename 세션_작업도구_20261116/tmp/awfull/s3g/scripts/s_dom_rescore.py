# -*- coding: utf-8 -*-
"""s_dom_rescore.py — 두 버전 비교: 지금 판(국내 + 해외)의 판들을 학습 없이 국내 시료로만 다시 채점(lst/scope.py 머리 참고).

- 지금 판 결과 뿌리(--source, 읽기만)의 설정 stage_dirs 아래 끝난 판 폴더(done.json)마다 저장 예측(pred_fold1~3·
  pred_thr_fold*·pred_refit)을 국내판 시료 표(--inputs 의 s0/0-1 — 지금 판 시료 표의 국내 행 그대로)에 있는 시료의
  쌍만 남겨 `lst.evaluate.score_cv`·`score_test` 로 채점(지표·u_tie·threshold 규칙·덜 찬 달 모두 같은 함수).
  국내 쌍은 지금 판 채점 쌍의 부분집합이라 남긴 쌍 수가 국내 시료 표의 기대 쌍 수와 같아야 함(require_complete —
  다르면 그 판은 '건너뜀' 으로 이유와 함께 기록: 4-c(상위 5 품목 쌍만)·①(h = 1 쌍만) 등).
- threshold: (가) 국내 CV 예측으로 §1-2 규칙을 다시 적용한 값(본 칸) — test 지표는 이 값으로, (나) 지금 판 CV(국내 +
  해외)에서 정한 원 threshold 로 test 지표도 따로(threshold_full_* 칸).
- 최종 모델: --source 의 final_decision.json 이 있으면 그 final_def.dir 판을 '최종' 줄로 더 넣음. 7-a 기준 줄 = 지금 판
  0-10 고른 칸의 시료 단위 예측을 국내 시료 쌍으로 펼친 것.
- test 정답은 --report --reason 보고 허가가 있을 때만(§1-3) — 없으면 CV 칸만.
- 예측이 판 폴더에 없으면 source.json·spec.extra.source 의 dir 를 따라감(s_final 구현 세부 (1) 과 같은 규칙).
출력(--out, 국내판 뿌리 안): compare_cv.csv(지금 판 원 CV·국내 다시 채점 CV) · compare_table[_report].csv(부록 A-12
  칼럼, 국내 채점) · compare_scores[_report].json · skipped.json · run_meta.json — 이미 있으면 멈춤(보고는 _report 이름).

실행(서버 — 지금 판이 끝난 뒤 아무 때, 국내판 단계 0 준비 뒤):
  B=~/cyj_listeria_final; P=$B/venv/bin/python
  $P -I scripts/s_dom_rescore.py --root $B/data --inputs $B/results_dom --source $B/results \
      --out $B/results_dom/compare/full_on_domestic --config configs_dom/s_dom_rescore.json
  보고(test): 같은 명령 + --report --reason "두 버전 비교 보고"
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import scope as sp  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = sp.with_field({
    "plan": cf.Field(str),
    "samples": cf.Field(str),                    # --inputs 아래(국내판 시료 표)
    "incomplete_months": cf.Field(str),          # --inputs 아래(국내 덜 찬 달)
    "source_samples": cf.Field(str),             # --source 아래(대조용 — 국내 행이 같은지)
    "source_te7a_dir": cf.Field(str),            # --source 아래
    "source_final_decision": cf.Field(str),      # --source 아래(없으면 최종 줄 없음)
    "stage_dirs": cf.Field(list, item_types=str),
})
STAGE = "국내 비교"


def load_cfg(path) -> dict:
    cfg = dict(cf.load_config(path, SCHEMA))
    if cfg.get("scope") != sp.DOMESTIC:
        raise cf.ConfigError("s_dom_rescore 는 scope = domestic 설정만")
    return cfg


def pred_dir(source_root: Path, vdir: Path) -> Path:
    if rn.pred_path(vdir, "fold1").exists() and rn.pred_path(vdir, "refit").exists():
        return vdir
    if (vdir / "source.json").exists():
        return resolve(source_root, rn.read_json(vdir / "source.json")["dir"])
    spec = rn.read_json(vdir / "spec.json")
    src = (spec.get("extra") or {}).get("source")
    if isinstance(src, Mapping) and src.get("dir"):
        return resolve(source_root, src["dir"])
    raise FileNotFoundError(f"{vdir.name}: 예측 위치를 못 찾음")


def keep_domestic(df: pd.DataFrame, samples: ev.SampleTable) -> pd.DataFrame:
    m = samples.index.get_indexer(pd.Index(df["lims_data_key"].astype(str).to_numpy(dtype=object))) >= 0
    return df.loc[m].reset_index(drop=True)


def check_same_domestic_rows(source_samples: Path, samples_path: Path) -> dict:
    """국내판 시료 표 = 지금 판 시료 표의 국내 행(키·d·h·범주·d ≤ 2025-06-30 정답 — 칼럼 전부, 같은 순서) 확인.
    test 기간 정답은 읽지 않음(history.read_samples_blind_test — §1-3)."""
    import pyarrow.parquet as pq

    from lst import history as hi

    cols = pq.read_schema(source_samples).names
    if pq.read_schema(samples_path).names != cols:
        raise AssertionError("국내판 시료 표 칼럼이 지금 판과 다름")
    a = hi.read_samples_blind_test(source_samples, cols)
    a = a[a["src_route"] == "domestic"].reset_index(drop=True)
    b = hi.read_samples_blind_test(samples_path, cols)
    if not a.equals(b):
        raise AssertionError("국내판 시료 표가 지금 판 시료 표의 국내 행과 다름")
    return {"rows": len(b)}


def variant_dirs(source_root: Path, rels) -> list[tuple[str, Path]]:
    out = []
    for rel in rels:
        d = resolve(source_root, rel)
        if not d.is_dir():
            continue
        for v in sorted(p for p in d.iterdir() if p.is_dir() and (p / "done.json").exists()):
            out.append((rel, v))
    return out


def rescore_one(vdir: Path, source_root: Path, samples: ev.SampleTable, truth, inc) -> dict:
    spec, done, cv_full = rn.load_done(vdir.parent, vdir.name)
    if done.get("smoke"):
        raise ValueError("연기 시험 판")
    pdir = pred_dir(source_root, vdir)
    preds = {f: keep_domestic(p, samples) for f, p in rn.load_preds(pdir, ev.CV_FOLDS).items()}
    thr = rn.load_thr_preds(pdir)
    thr = None if thr is None else {f: keep_domestic(p, samples) for f, p in thr.items()}
    cv = ev.score_cv(preds, samples, thr_preds=thr)
    rec = {"name": vdir.name, "desc": rn.VariantSpec(**spec).describe(), "pred_dir": str(pdir),
           "cv_full": float(cv_full["cv_band_weighted"]), "threshold_full": float(cv_full["threshold"]["threshold"]),
           "cv": cv, "test": None, "test_threshold_full": None, "selection_count": done.get("selection_count")}
    if truth is not None:
        tp = keep_domestic(pd.read_parquet(rn.pred_path(pdir, "refit")), samples)
        rec["test"] = ev.score_test(tp, samples, truth, float(cv["threshold"]["threshold"]), inc)
        rec["test_threshold_full"] = ev.score_test(tp, samples, truth, rec["threshold_full"], inc)
    return rec


def te7a_row(source_root: Path, rel: str, samples: ev.SampleTable, truth, inc) -> dict:
    b = rn.te7a_baseline(resolve(source_root, rel), samples)
    cv = ev.score_cv(b.cv_preds, samples)
    t = None
    if truth is not None and b.test_preds is not None:
        t = ev.score_test(b.test_preds, samples, truth, float(cv["threshold"]["threshold"]), inc)
    return {"name": f"{b.name}(지금 판)", "desc": b.desc, "pred_dir": rel, "cv_full": None, "threshold_full": None,
            "cv": cv, "test": t, "test_threshold_full": None, "selection_count": None}


def main(argv=None) -> Path:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="국내판 결과 뿌리")
    ap.add_argument("--source", required=True, help="지금 판 결과 뿌리(읽기만)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--report", action="store_true", help="test 채점(보고 허가)")
    ap.add_argument("--reason")
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    inputs_root = Path(args.inputs).expanduser().resolve()
    source_root = Path(args.source).expanduser().resolve()
    sp.apply(cfg, inputs_root, rp.out)
    if sp.read_marker(source_root) is not None:
        raise RuntimeError("--source 가 국내판 뿌리")
    if source_root == rp.out or source_root in rp.out.parents:
        raise RuntimeError("--out 이 지금 판 뿌리 안")
    tag = "_report" if args.report else ""
    names = (f"compare_cv{tag}.csv", f"compare_table{tag}.csv", f"compare_scores{tag}.json", f"skipped{tag}.json",
             f"run_meta{tag}.json")
    for n in names:
        if (rp.out / n).exists():
            raise FileExistsError(f"덮어쓰기 금지: {rp.out / n}")
    check_resources(rp.out)             # 가용 RAM 15GB·디스크 10GB 바닥값(부록 B-1) — 판마다 다시
    samples_path = resolve(inputs_root, cfg["samples"])
    same = check_same_domestic_rows(resolve(source_root, cfg["source_samples"]), samples_path)
    samples = ev.SampleTable.read(samples_path)
    inc = ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"]))
    truth = None
    if args.report:
        if not args.reason:
            raise SystemExit("--report 에는 --reason 이 필요")
        truth = ev.TestTruth.load(samples_path, ev.report_permit(STAGE, args.reason))

    recs, skipped = [], []
    fin_dir = None
    fdp = resolve(source_root, cfg["source_final_decision"])
    if fdp.exists():
        fd_ = rn.read_json(fdp)
        fin_dir = resolve(source_root, fd_["final_def"]["dir"]).resolve()
    try:
        recs.append(te7a_row(source_root, cfg["source_te7a_dir"], samples, truth, inc))
    except Exception as e:  # 기준 줄이 없으면 기록만
        skipped.append({"name": "7-a(지금 판)", "reason": f"{type(e).__name__}: {e}"})
    for rel, v in variant_dirs(source_root, cfg["stage_dirs"]):
        check_resources(rp.out)
        try:
            r = rescore_one(v, source_root, samples, truth, inc)
        except Exception as e:  # 판 하나가 안 돼도 나머지 표는 남김(건너뜀 기록)
            skipped.append({"dir": f"{rel}/{v.name}", "reason": f"{type(e).__name__}: {e}"})
            continue
        r["dir"] = f"{rel}/{v.name}"
        r["final"] = fin_dir is not None and v.resolve() == fin_dir
        recs.append(r)
    if fin_dir is not None and not any(r.get("final") for r in recs):
        # 지금 판 최종 모델 폴더가 stage_dirs 목록 밖이면(예: 다른 단계 폴더) 그 폴더를 직접 다시 채점
        rel_fin = fin_dir.relative_to(source_root.resolve()).as_posix() if fin_dir.is_relative_to(source_root.resolve()) \
            else str(fin_dir)
        try:
            r = rescore_one(fin_dir, source_root, samples, truth, inc)
            r["dir"], r["final"] = rel_fin, True
            recs.append(r)
        except Exception as e:
            skipped.append({"dir": rel_fin, "reason": f"최종 모델 폴더: {type(e).__name__}: {e}"})

    cvrows = [{"판": r["name"], "폴더": r.get("dir", r["pred_dir"]), "최종": "예" if r.get("final") else "",
               "지금 판 CV(국내 + 해외)": r["cv_full"], "국내 다시 채점 CV": r["cv"]["cv_band_weighted"],
               **{f"국내 {f}": r["cv"]["fold_band_weighted"][f] for f in ev.CV_FOLDS},
               "국내 threshold": r["cv"]["threshold"]["threshold"], "지금 판 threshold": r["threshold_full"],
               **({"국내 test 구간 가중": r["test"]["report"]["pr_auc_band_weighted"]} if r["test"] is not None else {})}
              for r in recs]
    pd.DataFrame(cvrows).to_csv(rp.out / names[0], index=False, encoding="utf-8-sig")
    rows = [ev.table_row(r["name"] + (" [최종]" if r.get("final") else ""), r["desc"], r["cv"], r["test"],
                         selection_count=r["selection_count"]) for r in recs]
    notes = ev.table_notes([r["test"] for r in recs if r["test"] is not None],
                           ("국내만 다시 채점: 지금 판(국내 + 해외 학습) 저장 예측 중 국내 시료 쌍만 — 학습 없음",
                            "threshold = 국내 CV 예측으로 §1-2 규칙을 다시 적용(지금 판 원 threshold 의 test 지표는 json)"))
    ev.write_table(rows, notes, rp.out / names[1])
    cf.dump_scores_json(rp.out / names[2], recs)
    cf.dump_json(rp.out / names[3], skipped)
    md5 = {"inputs:" + cfg["samples"]: cf.file_md5(samples_path),
           "source:" + cfg["source_samples"]: cf.file_md5(resolve(source_root, cfg["source_samples"]))}
    cf.save_run_meta(rp.out / names[4], cfg, md5, CODE_ROOT, n_jobs=None,
                     extra={"source_root": str(source_root), "report": bool(args.report), "reason": args.reason,
                            "n_scored": len(recs), "n_skipped": len(skipped), "same_domestic_rows": same})
    print(f"[국내 비교] 채점 {len(recs)}판, 건너뜀 {len(skipped)} → {rp.out}")
    return rp.out


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
