# -*- coding: utf-8 -*-
"""s5_6.py — 단계 5·6 손 교수 방식(해마다 양성 전부 + 음성 4배, 씨앗 0) — 계획서 요약 17, §6 2차 공통 표 5·6 줄·단계 5·6,
부록 A-6·A-9·A-12·A-14·A-15. 공통 = lst/stage4.

판(최대 4):
  따로  = 단계 4 시작 때 현재 최고 판(사슬 상태 stage4.start_best)의 train 에서 뽑음 → 5-<구조>·6-<구조>
  쌓기  = 단계 4 가 끝난 뒤 현재 최고 판이 단계 4 에서 채택된 판(4-a·4-a0·②)일 때만, 그 판의 줄인 train 에서 뽑음 →
          5-<그 판 이름>·6-<그 판 이름>(예: 5-4-a-CAT) — 아니면 쌓기 = 따로라 한 벌(판 없음)
  5 = 해마다 양성 전부 + 그해 음성 무작위 4 × 양성 수(모자라면 그해 음성 전부), 6 = 같은 수를 그해 양성이 나온
  (3단계 × 시군구, 해외 3단계 × 나라) 칸들의 그해 음성 풀에서. 뽑기는 시료 단위 → 그 판의 h 규칙(4-a 트리 부분이면 h 10개).
  모두: 바탕 판 설정(반복 수만 다시), 트리 = 단계 4~6 범주 숫자 인코딩, 씨앗 0(뽑기·모델 학습), 이상 점수(VAE)·④ 블록
  트리·② 도 뽑은 자료로 다시 학습, validation·test 전부 예측.
설정 sub:
  prep   학습 없음: 뽑은 키(kept_keys_<판>·selection.json — 해마다 뽑은 양성·음성 수) — dirs 의 sel-* 폴더.
  T      판 학습(트리·딥러닝 섞이면 gpu_memory_fraction).
  judge  학습 없음: 4판 중 val 1위 대 현재 최고 판(§6 2차 공통 (가)) → 사슬 상태 chain_state_out·사슬 기록·5-6_decision.json.
보고: T 설정 + --report --reason "단계 5·6 보고"(judge 뒤).

실행(서버 — 검수 뒤 지휘자, 단계 4 판정 뒤):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s5_6.py $A --out ~/cyj_listeria_final/results/s5_6/sel --config configs/s5_6_prep.json
  $P scripts/s5_6.py $A --out ~/cyj_listeria_final/results/s5_6/5-6-T --config configs/s5_6_T.json
  $P scripts/s5_6.py $A --out ~/cyj_listeria_final/results/s5_6/5-6-judge --config configs/s5_6_judge.json

구현 세부(계획서에 없음 — 담당자 확인): lst/stage4 머리 (b)·(c)·(R4), 그리고
 (u1) 따로 판의 바탕이 단계 4 시작 때 판이고 그 판 설정 그대로(트리 부분은 범주 숫자 인코딩만 바뀜) — h 는 시료당 자기 h 1개.
 (u2) 쌓기 바탕 = 단계 4 판정 뒤 현재 최고 판이 4-a·4-a0·x2-NN60 중 하나일 때(사슬 상태 stage4.adopted).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

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
from lst import stage4 as s4  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, check_resources, limit_gpu_memory  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = s4.STAGE56
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
SEL = {("sep", "5"): "sel-hon5", ("sep", "6"): "sel-hon6", ("stk", "5"): "sel-stk-hon5", ("stk", "6"): "sel-stk-hon6"}
DIRS_SCHEMA = {k: cf.Field(str) for k in (*SEL.values(), "5-6-T", "5-6-judge")}
TRAIN_SCHEMA = {
    "sub": cf.Field(str, choices=("T",)),
    "decision_1_3": cf.Field(str),
    "selection_2": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "chain_state_in": cf.Field(str),       # 단계 4 판정 뒤 사슬 상태
    "theil_targets": cf.Field(str),
    "dirs": DIRS_SCHEMA,
    "gpu_memory_fraction": cf.Field(float, required=False),
    "device": cf.Field(str, required=False),
}
PREP_SCHEMA = {"plan": cf.Field(str), "sub": cf.Field(str, choices=("prep",)), "stage": cf.Field(str),
               "samples": cf.Field(str), "chain_state_in": cf.Field(str), "dirs": DIRS_SCHEMA}
PREP_SCHEMA = lst_scope.with_field(PREP_SCHEMA)   # 국내만 버전 scope 칸(없으면 지금 판)
JUDGE_SCHEMA = {"plan": cf.Field(str), "sub": cf.Field(str, choices=("judge",)), "stage": cf.Field(str),
                "order": cf.Field(str), "experiment": cf.Field(str), "samples": cf.Field(str),
                "experiment_log": cf.Field(str), "use_mlflow": cf.Field(bool), "mlflow_uri": cf.Field(str),
                "code_reviewed": cf.Field(bool), "selection_2": cf.Field(str), "chain_state_in": cf.Field(str),
                "chain_state_out": cf.Field(str), "chain_log": cf.Field(str), "dirs": DIRS_SCHEMA}
JUDGE_SCHEMA = lst_scope.with_field(JUDGE_SCHEMA)   # 국내만 버전 scope 칸(없으면 지금 판)


def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run"):
        ap.add_argument(a, action="store_true")
    ns = ap.parse_known_args(list(argv))[0]
    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)
    return ns


def bases(state: Mapping) -> dict:
    """{'sep': 단계 4 시작 때 현재 최고 판, 'stk': 단계 4 채택 판(없으면 없음)}."""
    if "stage4" not in state:
        raise s4.StageError("사슬 상태에 stage4 칸 없음 — 단계 4 판정(s4 judge) 뒤 상태여야 함")
    out = {"sep": state["stage4"]["start_best"]}
    if state["stage4"]["adopted"]:
        out["stk"] = state["current_best"]
    return out


def base_plan(best: Mapping) -> s4.RowPlan:
    """바탕 판의 RowPlan(구성 모두 같아야 함 — 트리 부분 기준, 딥러닝 구성은 own)."""
    plans = {json_key(s4.RowPlan.from_dict(m.get("row_plan"))) for m in best["members"]
             if m.get("row_plan") is not None}
    if len(plans) > 1:
        raise s4.StageError(f"{best['name']}: 구성마다 RowPlan 이 다름")
    m = next((m for m in best["members"] if m.get("row_plan")), None)
    return s4.RowPlan.from_dict(m["row_plan"]) if m else s4.RowPlan()


def json_key(p: s4.RowPlan) -> str:
    import json

    return json.dumps(p.to_dict(), sort_keys=True)


def plan_for(cfg, which: str, kind: str, best: Mapping) -> s4.RowPlan:
    bp = base_plan(best)
    rep = bp.rep if which == "stk" else "own"
    if which == "stk" and best["kind"] in ("dl", "nn60"):
        rep = "own"
    return s4.RowPlan(tuple(bp.steps if which == "stk" else ()) + (f"hon{kind}",), rep, cfg["dirs"][SEL[(which, kind)]])


def build(cfg: Mapping, inputs_root: Path) -> dict:
    state = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
    k = s4.load_k(inputs_root, cfg["selection_2"])
    out_rel = cfg["dirs"]["5-6-T"]
    run, ens, bests = [], [], {}
    for which, best in bases(state).items():
        part = s4.name_part(best, inputs_root) if which == "sep" else best["name"]
        for kind in ("5", "6"):
            d = s4.derive_specs(best, inputs_root=inputs_root, name=f"{kind}-{part}", stage=STAGE, role=s4.ROLE_CAND,
                                plan=plan_for(cfg, which, kind, best), catenc_k=k, model_seed=s4.HON_SEED)
            run.extend(d.run)
            if d.ensemble is not None:
                ens.append(d.ensemble)
            b = s4.fill_dirs(d.best, out_rel)
            if best["kind"] == "nn60":
                b["replacement"] = True       # ② 를 다시 학습한 판도 ② 처럼(단계 8 튜닝 안 함 — 이름 조각 = 판 이름)
            bests[b["name"]] = b
    return {"run": run, "ens": ens, "bests": bests, "state": state}


def check_cfg(cfg) -> None:
    if cfg["stage"] != STAGE or cfg["order"] != s4.ORDER or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("stage '5-6' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
    f = cfg.get("gpu_memory_fraction")
    if f is not None and not (0 < f <= GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")


def make_specs_fn(inputs_root: Path):
    holder = {}

    def specs_fn(cfg):
        check_cfg(cfg)
        if "b" not in holder:
            holder["b"] = build(cfg, inputs_root)
        return holder["b"]["run"]
    specs_fn.holder = holder
    return specs_fn


def factories(cfg, rp, inputs_root: Path) -> sc.Factories:
    from lst import inputs as ip
    from lst import wiring as wr

    dl_kw = {"max_epochs": cfg["smoke"]["max_epochs"]} if "smoke" in cfg and "max_epochs" in cfg["smoke"] else {}
    ctx = ip.load_context(cfg["context"], rp.root, inputs_root, components=list(COMPONENTS))
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    w = s4.Stage4Wiring(ctx, samples, inputs_root, data_root=rp.root,
                        theil_targets=resolve(inputs_root, cfg["theil_targets"]), n_jobs=cfg["n_jobs"],
                        device=cfg.get("device"), dl_kwargs=dl_kw, check=lambda: check_resources(rp.out))
    return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                        input_md5s=wr.input_md5s(cfg["context"], rp.root, inputs_root), fit_kwargs=w.fit_kwargs)


def before_variant(cfg) -> None:
    if cfg.get("gpu_memory_fraction"):
        limit_gpu_memory(cfg["gpu_memory_fraction"])


def report_rows_for(inputs_root: Path):
    def report_rows(cfg, _ir, samples, exp_log):
        from lst import stage3 as s3

        jd = resolve(inputs_root, cfg["dirs"]["5-6-judge"]) / "5-6_decision.json"
        if not jd.exists():
            raise SystemExit("단계 5·6 판정(judge)을 먼저 — test 는 판정 뒤에만(§1-3)")
        dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
        st = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
        rows = [s3.baseline_1_row(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"]),
                s4.best_row(inputs_root, st["current_best"], "단계 5·6 시작 때 현재 최고 판(판정 상대)")]
        if st["stage4"]["start_best"]["name"] != st["current_best"]["name"]:
            rows.append(s4.best_row(inputs_root, st["stage4"]["start_best"], "단계 4 시작 때 현재 최고 판(따로 판 바탕)"))
        out = resolve(inputs_root, cfg["dirs"]["5-6-T"])
        for e in build(cfg, inputs_root)["ens"]:
            d = out / e.name
            done = rn.read_json(d / "done.json")
            rows.append(rn.BaseRow(e.name, f"앙상블(구성 {', '.join(e.extra['s4']['members'])} 평균)",
                                   rn.load_preds(d, ev.CV_FOLDS), pd.read_parquet(rn.pred_path(d, "refit")),
                                   done["selection_count"], done.get("mlflow_run_id"), thr_preds=rn.load_thr_preds(d)))
        return rows
    return report_rows


NOTES = ("5·6 판의 트리 = 단계 4~6 범주 숫자 인코딩(부록 A-6) — 바탕이 단계 4~6 판이 아니면 차이에 범주 인코딩 변경이 섞임",
         "씨앗 0 한 번(뽑기·학습) — MLflow param 'seed' 는 실행 틀 값 42 로 남고 실제 학습 씨앗은 extra.s4.model_seed = 0",
         "해마다 뽑은 양성·음성 수 = 뽑은 키 폴더 selection.json",
         "부스팅 판: CV = 교차 N 예측, threshold = 공통 N 예측, test = refit 공통 N(§3·§1-2)")


# ================================================================ prep·judge
def prep(argv) -> dict:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, PREP_SCHEMA))
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    if cfg["stage"] != STAGE:
        raise cf.ConfigError("stage '5-6'")
    inputs_root = Path(args.inputs).expanduser().resolve()
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    state = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
    out = {}
    for which, best in bases(state).items():
        bp = base_plan(best)
        if which == "sep" and bp.steps:
            raise s4.StageError("따로 판 바탕(단계 4 시작 때 판)에 이미 고르기가 있음")
        base_sel = resolve(inputs_root, bp.sel_dir) if (which == "stk" and bp.steps) else None
        for kind in ("5", "6"):
            info = s4.prepare_selection(samples, resolve(inputs_root, cfg["dirs"][SEL[(which, kind)]]), [f"hon{kind}"],
                                        base_sel=base_sel)
            out[(which, kind)] = info
            for s in rn.SPLIT_ORDER:
                st = info["per_split"][s]["steps"][-1]
                print(f"{which}-{kind} {s}: {info['per_split'][s]['n_start']} → {st['n_after']}(검출 {st['pos_after']})")
    return out


def judge(argv, *, tracker_fn=sc.make_tracker) -> dict:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    if cfg["stage"] != STAGE or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("judge 설정: stage '5-6' · experiment listeria-2차")
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = rp.out
    if out.resolve() != resolve(inputs_root, cfg["dirs"]["5-6-judge"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[5-6-judge]")
    p = out / "5-6_decision.json"
    if p.exists():
        print(f"단계 5·6 판정 이미 있음: {p}")
        return rn.read_json(p)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    b = build({**cfg}, inputs_root)
    state = b["state"]
    stack = [(bd, resolve(inputs_root, bd["dir"])) for bd in b["bests"].values()]
    res = s4.judge_step(step=STAGE, state=state, stack=stack, repl=[], samples=samples, inputs_root=inputs_root)
    rec, new = res["record"], res["state"]
    tr = tracker_fn(cfg)
    if tr is not None:
        sr = tr.stage_run(s4.EXPERIMENT, STAGE)
        s4.tag_judgements(tr, rec, lambda n: tr.find_run(s4.EXPERIMENT, n, sr))
    line = {"step": STAGE, "judgement": "채택" if rec["after"] != rec["before"] else "버림", "before": rec["before"],
            "after": rec["after"], "tree_target": new["tree_target"]["name"], "parts": rec["parts"], "utc": rec["utc"]}
    cpath = s4.append_chain_log(inputs_root, cfg["chain_log"], line)
    if tr is not None:
        tr.log_artifact(tr.chain_run(), cpath)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    for part in rec["parts"]:
        for a in part["judge"]["alerts"]:
            log.append("alert", stage=STAGE, message=a)
            print(f"[알림] {a}")
    log.append("decision", stage=STAGE, before=rec["before"], after=rec["after"])
    s4.save_state(resolve(inputs_root, cfg["chain_state_out"]), new)
    out.mkdir(parents=True, exist_ok=True)
    cf.dump_scores_json(p, {"record": rec, "state_out": cfg["chain_state_out"]})
    if tr is not None:
        tr.log_artifact(tr.stage_run(s4.EXPERIMENT, STAGE), p, "decision")
    print(f"단계 5·6: {rec['before']} → {rec['after']}")
    return rn.read_json(p)


# ================================================================ 진입점
def main(argv=None, *, factories_fn=None, tracker_fn=sc.make_tracker, check_resources_fn=None,
         before_variant_fn=before_variant):
    argv = list(sys.argv[1:] if argv is None else argv)
    ns = _pre(argv)
    if not ns.config or not ns.inputs:
        raise SystemExit("--config·--inputs 필요")
    raw = cf.load_json_strict(ns.config)
    sub = raw.get("sub")
    if sub == "prep":
        return prep(argv)
    if sub == "judge":
        return judge(argv, tracker_fn=tracker_fn)
    if sub != "T":
        raise cf.ConfigError(f"sub: {sub!r}")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    if not (ns.list or ns.verify_run):
        want = resolve(inputs_root, raw["dirs"]["5-6-T"]).resolve()
        if Path(ns.out).expanduser().resolve() != want:
            raise cf.ConfigError(f"--out 은 --inputs 아래 dirs[5-6-T] = {want}")
    specs_fn = make_specs_fn(inputs_root)
    if ns.variant:
        cfg = sc.load_cfg(ns.config, TRAIN_SCHEMA)
        by = {s.name: s for s in specs_fn(cfg)}
        group = [by[n.strip()] for n in ns.variant.split(",") if n.strip()]
        s4.install_for_group(group, inputs_root)
    res = sc.stage_main(argv, script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=TRAIN_SCHEMA,
                        specs_fn=specs_fn, factories_fn=factories_fn or factories, before_variant=before_variant_fn,
                        fold_recall=True, tracker_fn=tracker_fn, check_resources=check_resources_fn,
                        report_rows_fn=report_rows_for(inputs_root), report_notes=NOTES)
    default = not (ns.list or ns.variant or ns.verify_run or ns.report or ns.only)
    if default and specs_fn.holder["b"]["ens"]:
        cfg = sc.load_cfg(ns.config, TRAIN_SCHEMA)
        samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
        log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
        tr = tracker_fn(cfg)
        sr = tr.stage_run(cfg["experiment"], STAGE) if tr is not None else None
        outp = Path(ns.out).expanduser().resolve()
        code = rn.snapshot_code(CODE_ROOT, outp)
        for e in specs_fn.holder["b"]["ens"]:
            d = s4.write_ensemble(outp, e, samples, log, code, tracker=tr, stage_run_id=sr)
            print(f"{e.name}: 앙상블 CV {d['cv_band_weighted']}")
    return res


if __name__ == "__main__":
    main()
