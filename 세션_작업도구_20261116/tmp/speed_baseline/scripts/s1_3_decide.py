# -*- coding: utf-8 -*-
"""s1_3_decide.py — 단계 1-3 정하기(계획서 §6 1-3, §1-2 동점, 부록 A-12 동점 많은 판, A-15 '선택' tag).

- window W = 1차 70 설정(1-1 트리 60 + 1-2 LSTM 10) 중 CV 1위 설정의 W(트리·딥러닝 모든 판 — W_DL = W)
- 트리 입력 방식 = 1위가 트리면 그 방식, LSTM 이면 그 W 의 트리 설정 중 CV 1위 방식
- 1차 베이스라인 = 70 설정 중 CV 1위
- 기준 알고리즘 A = 그 W·입력 방식의 트리 설정 중 CV 1위 알고리즘
고르기 = CV 평균 구간 가중 PR-AUC(소수 4자리) → 판의 학습 횟수 → 판 이름(§1-2). 동점 많은 판이 1위와 0.001 안이면
다시 채점(u_tie = md5(key|h|1)) 해 순위가 그대로인지 기록, CV 1위가 동점 많으면 확정 전 알림(부록 A-12).
고르는 코드는 test 를 읽지 않음(evaluate.selection_guard — 후보는 CV 값만).

실행(1-1·1-2 가 끝난 뒤):
  python scripts/s1_3_decide.py --root … --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s1/1-3 \
      --config configs/s1_3_decide.json            # 정하기(CV 만)
  … --table                                         # 정한 뒤: 알고리즘 × W × 입력 표(70 설정 CV·test — 보고 표 합침)
출력: 1-3_decision.json·1-3_cv_pivot.csv(·1-3_table.csv), MLflow '선택' tag.
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

from lst import config as cf  # noqa: E402
from lst import scope as lst_scope  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = {
    "plan": cf.Field(str),
    "samples": cf.Field(str),
    "experiment_log": cf.Field(str),
    "dir_1_1": cf.Field(str),     # --inputs 아래 1-1 결과 폴더
    "dir_1_2": cf.Field(str),
    "config_1_1": cf.Field(str),  # 코드 뿌리 아래 1-1·1-2 설정(판 목록)
    "config_1_2": cf.Field(str),
    "use_mlflow": cf.Field(bool),
    "mlflow_uri": cf.Field(str),
}
SCHEMA = lst_scope.with_field(SCHEMA)   # 국내만 버전 scope 칸(없으면 지금 판)


def decide_1_3(trees: list[ev.Candidate], lstms: list[ev.Candidate], rescore) -> dict:
    """§6 1-3 정하기. trees·lstms = CV 후보(extra 에 algorithm·W·mode·kind)."""
    if len(trees) != 60 or len(lstms) != 10:
        raise ValueError(f"1차 = 트리 60 + LSTM 10 (받은 수 {len(trees)}·{len(lstms)})")
    with ev.selection_guard():
        allc = {c.name: c for c in trees + lstms}
        s_all = ev.select_with_tie_check(list(allc.values()), rescore)
        top = allc[s_all["selected"]]
        W = int(top.extra["W"])
        s_mode = None
        if top.extra["kind"] == "tree":
            mode, mode_from = top.extra["mode"], top.name
        else:
            s_mode = ev.select_with_tie_check([c for c in trees if c.extra["W"] == W], rescore)
            mode_from = s_mode["selected"]
            mode = allc[mode_from].extra["mode"]
        s_a = ev.select_with_tie_check([c for c in trees if c.extra["W"] == W and c.extra["mode"] == mode], rescore)
        a_var = s_a["selected"]
        alerts = s_all["alerts"] + (s_mode["alerts"] if s_mode else []) + s_a["alerts"]
        return {
            "baseline": top.name, "baseline_cv": top.cv, "W": W, "W_DL": W,
            "tree_mode": mode, "tree_mode_from": mode_from,
            "A": allc[a_var].extra["algorithm"], "A_variant": a_var,
            "select_all": s_all, "select_mode": s_mode, "select_A": s_a, "alerts": alerts,
        }


def selection_tags(dec: dict) -> dict[str, str]:
    """판 이름 → '선택' tag 값(A-15: 1차 베이스라인·window W·트리 입력 방식·A)."""
    tags: dict[str, list[str]] = {}
    tags.setdefault(dec["baseline"], []).extend(["1차 베이스라인", f"window W = {dec['W']}"])
    tags.setdefault(dec["tree_mode_from"], []).append(f"트리 입력 방식 = {dec['tree_mode']}")
    tags.setdefault(dec["A_variant"], []).append(f"A = {dec['A']}")
    return {k: ", ".join(dict.fromkeys(v)) for k, v in tags.items()}


def cv_pivot(cands: list[ev.Candidate]) -> pd.DataFrame:
    rows = [{"algorithm": c.extra["algorithm"], "W": c.extra["W"], "mode": c.extra["mode"] or "-", "name": c.name,
             "cv_band_weighted": c.cv, "many_ties": c.many_ties} for c in cands]
    return pd.DataFrame(rows).sort_values(["algorithm", "mode", "W"]).reset_index(drop=True)


def _names(script: str, cfg_rel: str) -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(script, CODE_ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from lst import stage_cli as sc

    return [s.name for s in mod.specs(sc.load_cfg(CODE_ROOT / cfg_rel, mod.SCHEMA_EXTRA))]


def main(argv=None):
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--table", action="store_true", help="정한 뒤 70 설정 CV·test 표(보고 표 합침)")
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, SCHEMA))
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    inputs_root = Path(args.inputs).expanduser().resolve()
    d11, d12 = resolve(inputs_root, cfg["dir_1_1"]), resolve(inputs_root, cfg["dir_1_2"])
    dec_path = rp.result("1-3_decision.json")
    if args.table:
        if not dec_path.exists():
            raise SystemExit("1-3 정하기를 먼저(test 는 정한 뒤에만 봄 — §1-3)")
        t = pd.concat([pd.read_csv(d / "stage_table_report.csv", encoding="utf-8-sig") for d in (d11, d12)],
                      ignore_index=True)
        with open(rp.result("1-3_table.csv"), "x", newline="", encoding="utf-8-sig") as fh:
            t.to_csv(fh, index=False)
        return t
    if dec_path.exists():
        raise FileExistsError(f"이미 정함: {dec_path}")
    n11 = _names("s1_1_trees", cfg["config_1_1"])
    n12 = _names("s1_2_lstm", cfg["config_1_2"])
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    trees, lstms = rn.candidates(d11, n11), rn.candidates(d12, n12)
    r11, r12 = rn.rescore_fn(d11, samples), rn.rescore_fn(d12, samples)
    dec = decide_1_3(trees, lstms, lambda n: (r11 if n in set(n11) else r12)(n))
    tags = selection_tags(dec)
    dec["selection_tags"] = tags
    cf.dump_scores_json(dec_path, dec)
    with open(rp.result("1-3_cv_pivot.csv"), "x", newline="", encoding="utf-8-sig") as fh:
        cv_pivot(trees + lstms).to_csv(fh, index=False)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    log.append("decision", stage="1-3", baseline=dec["baseline"], W=dec["W"], tree_mode=dec["tree_mode"], A=dec["A"],
               alerts=dec["alerts"])
    if cfg["use_mlflow"]:
        from lst.tracking import Tracker

        tr = Tracker(cfg["mlflow_uri"])
        for name, val in tags.items():
            stage = "1-1" if name in set(n11) else "1-2"
            sr = tr.find_run("listeria-1차", stage, None)
            vr = tr.find_run("listeria-1차", name, sr.info.run_id)
            tr.set_tags(vr.info.run_id, {"선택": val})
        sr = tr.stage_run("listeria-1차", "1-3")
        tr.log_artifact(sr, dec_path)
        tr.log_artifact(sr, rp.result("1-3_cv_pivot.csv"))
        tr.finish(sr)
    print(f"1차 베이스라인 {dec['baseline']} · W {dec['W']} · 트리 입력 {dec['tree_mode']} · A {dec['A']} · 알림 {dec['alerts']}")
    return dec


if __name__ == "__main__":
    main()
