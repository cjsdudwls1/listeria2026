# -*- coding: utf-8 -*-
"""s3_1.py — 단계 3-1 이상 점수(VAE) — 계획서 요약 15, §6 2차 공통(채택 판정·트리 대상 판·현재 입력)·단계 3-1, §7-3(블록
OOF·엄격 OOF), §1-2·§1-3, 부록 A-8(VAE 입력·robust z)·A-9(3-1 = 23)·A-12(같은 행 모델 재사용·이상 점수 예측 범위)·A-14·A-15.

차례(기본 실행 한 번 — 끝난 차례는 건너뜀, ⑥⑦ 판정(x67-judge — 3-0 순위 포함) 뒤, v78 순서 3-0 → ⑥⑦ → 3-1):
  0. 3-1_plan.json: 판 목록·고르기 규칙(§1-3 'test 채점 전에 적어 둠') — 트리 대상 판(= 3-0 트리 6종 1위)·현재 최고 판
  1. 3-1-VAEt(하위 프로세스 `--vae`, GPU): 표 VAE 19 모델(blocks.block_plan 재사용 켬 — 전역 6 + 엄격 13), 입력 = 트리 대상
     판 X 의 숫자 칼럼(h·g·범주 뺌) → 표준화·결측 0 + 결측 표시, 불검출 행만 학습, 이상 점수 = robust z.
     fold train 행 = fold 안 블록 OOF, fold validation 쌍 = fold train 전체 모델, refit train 행 = 전역 블록 OOF,
     test 쌍 = test 전 전체 모델 → oof_<split>.parquet·pred_<split>.parquet, 단독 판(비교만) CV = fold train 전체 모델 예측.
  2. 3-1-<알고리즘>-VAE(하위 프로세스 `--variant`): 트리 대상 판 설정 그대로 + 이상 점수 1칼럼 'anom_vae'(부스팅은 §3 반복 수
     고르기 켠 채 — 교차 N·공통 N·CatBoost 학습률 평균 다시), 3 fold + refit·test
  3. 판정: 3-1 판(쌓는 후보) val 이 현재 최고 판보다 오르면(4자리) 채택 → 새 현재 최고 판·트리 대상 판, 현재 입력에 이상 점수
     칼럼. → 3-1_decision.json(state_after = 다음 단계가 읽음) + 사슬 기록 한 줄 + MLflow 판정 tag
  보고(--report, 판정 뒤): 3-1 판 test + 7-a·1차 베이스라인·현재 최고 판(판정 전)·트리 대상 판·3-1-VAEt 줄 표(부록 A-12)

실행(서버 — 검수 뒤 지휘자):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s3_1.py $A --out ~/cyj_listeria_final/results/s3/3-1 --config configs/s3_1.json
  $P scripts/s3_1.py $A --out ~/cyj_listeria_final/results/s3/3-1 --config configs/s3_1.json --report --reason "단계 3-1 보고"

구현 세부(계획서에 없음 — 담당자 확인): lst/stage3_combine 머리 (a)~(e)·(i)·(j). VAE 단계는 하위 프로세스 하나에서 한 번(멈추면
그 판 폴더를 _incomplete 로 옮기고 처음부터 — 블록 모델 이어하기 없음). 3-1 판 이름 = '3-1-<알고리즘>-VAE'(부록 A-14).
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()


from lst import blocks as bk  # noqa: E402
from lst import config as cf  # noqa: E402
from lst import scope as lst_scope  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage3 as s3  # noqa: E402
from lst import stage3_combine as c3  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.paths import resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, check_resources, limit_gpu_memory  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = "3-1"
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
DIRS_SCHEMA = {"x67-judge": cf.Field(str), "3-1": cf.Field(str)}
SCHEMA = {
    "decision_1_3": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "dirs": DIRS_SCHEMA,
    "chain_log": cf.Field(str),
    "gpu_memory_fraction": cf.Field(float),
    "vae_device": cf.Field(str, choices=("cuda", "cpu")),
}


def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run", "--vae"):
        ap.add_argument(a, action="store_true")
    ns = ap.parse_known_args(list(argv))[0]
    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)
    return ns


def load_cfg(path) -> dict:
    cfg = sc.load_cfg(path, SCHEMA)
    if cfg["stage"] != STAGE or cfg["order"] != s3.ORDER or cfg["experiment"] != s3.EXPERIMENT:
        raise cf.ConfigError(f"stage '{STAGE}' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
    if cfg["n_jobs"] != 8:
        raise cf.ConfigError("n_jobs 8")
    if not (0 < cfg["gpu_memory_fraction"] <= GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    return cfg


def _smoke(cfg) -> dict | None:
    return {k: cfg["smoke"][k] for k in rn.SMOKE_KEYS} if "smoke" in cfg else None


# ================================================================ 판 설정
def state0(cfg, inputs_root: Path) -> dict:
    return c3.state_after_x67(inputs_root, cfg["dirs"])


def board_spec(cfg, inputs_root: Path, st: Mapping | None = None) -> rn.VariantSpec:
    """3-1 판 = 트리 대상 판(그 판의 현재 입력) + 이상 점수 1칼럼."""
    st = st or state0(cfg, inputs_root)
    tt = st["tree_target"]
    src = c3.source_spec(inputs_root, tt["dir"])
    if dict(src.extra["version"]) != dict(st["current_inputs"].get(src.algorithm, {}).get("version",
                                                                                         src.extra["version"])):
        raise c3.CombineError("트리 대상 판 설정과 현재 입력이 다름")
    return c3.spec_plus_anom(src, name=c3.name_3_1(src.algorithm), stage=STAGE, vae_rel=cfg["dirs"]["3-1"],
                             base_variant=tt["name"])


def write_plan(cfg, inputs_root: Path, out: Path) -> dict:
    p = out / c3.PLAN[STAGE]
    st = state0(cfg, inputs_root)
    sp = board_spec(cfg, inputs_root, st)
    plan = {"stage": STAGE, "boards": [c3.VAE_T, sp.name],
            "roles": {c3.VAE_T: "비교만(이상 점수 단독)", sp.name: "후보(쌓는 후보)"},
            "rule": "3-1 판 val(CV 평균 구간 가중 PR-AUC)이 현재 최고 판보다 소수 4자리에서 오르면 채택(같으면 안 오름) — §6 2차 공통",
            "current_best": st["current_best"]["name"], "tree_target": st["tree_target"]["name"],
            "board_spec_md5": sp.md5(), "utc": c3._utc()}
    if p.exists():
        old = rn.read_json(p)
        if old["board_spec_md5"] != plan["board_spec_md5"] or old["current_best"] != plan["current_best"]:
            raise c3.CombineError("3-1_plan.json 이 지금 판 설정·현재 최고 판과 다름")
        return old
    out.mkdir(parents=True, exist_ok=True)
    cf.dump_json(p, plan)
    return plan


# ================================================================ 3-1-VAEt(하위 프로세스)
def run_vae(cfg, root: Path, inputs_root: Path, out: Path, *, feats=None, model_factory=None, check=None,
            tracker=None) -> dict:
    vdir = out / c3.VAE_T
    st = state0(cfg, inputs_root)
    tt = st["tree_target"]
    src = c3.source_spec(inputs_root, tt["dir"])
    plan = bk.block_plan(reuse=True)
    counts = bk.plan_counts(plan)
    if counts["total"] != 19:
        raise c3.CombineError(f"VAE 모델 수 {counts['total']} ≠ 19(부록 A-9)")
    spec = c3.vae_spec(src, tt["name"], counts)
    code_md5 = rn.snapshot_code(CODE_ROOT, out)
    if c3.begin_board(vdir, out, spec, code_md5) == "done":
        print(f"{c3.VAE_T}: 이미 끝남")
        return rn.read_json(vdir / "done.json")
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    known = samples.known_frame()
    smoke = _smoke(cfg)
    W = int(src.W)
    ctx_md5s: dict = {}
    if feats is None:
        from lst import inputs as ip
        from lst import wiring as wr

        ctx = ip.load_context(cfg["context"], root, inputs_root, components=list(COMPONENTS))
        feats = c3.TreeTargetVaeFeatures(ctx, wr.version_spec(src))
        ctx_md5s = wr.input_md5s(cfg["context"], root, inputs_root)
    if model_factory is None:
        from lst import models_dl as md

        if cfg["vae_device"] == "cuda":
            limit_gpu_memory(cfg["gpu_memory_fraction"])
        kw = {"max_epochs": cfg["smoke"]["max_epochs"]} if smoke and "max_epochs" in cfg["smoke"] else {}

        def model_factory():
            return md.AnomalyModel("VAE", "tab", device=cfg["vae_device"], **kw)
    res = c3.build_vae(
        feats=feats, samples=samples, plan=plan, W=W,
        rows_of_model=lambda m: c3.model_train_rows(m, samples, W, smoke=smoke, seed=cfg["seed"], known=known),
        rows_of_split=lambda s: c3.split_train_rows(samples, s, W, smoke=smoke, seed=cfg["seed"], known=known),
        model_factory=model_factory, seed=cfg["seed"], n_jobs=cfg["n_jobs"], vdir=vdir,
        max_pairs_per_h=smoke["max_pairs_per_h"] if smoke else None, check=check or (lambda: check_resources(out)))
    oof_paths = c3.write_oof(vdir, res["oof"])
    ptab = vdir / "block_plan.csv"
    bk.plan_table(plan).to_csv(ptab, index=False, encoding="utf-8-sig")
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    md5s = {**cf.inputs_md5(inputs_root, [cfg["samples"], cfg["incomplete_months"]]), **ctx_md5s}
    sr = tracker.stage_run(s3.EXPERIMENT, STAGE) if tracker is not None else None
    arts = [(ptab, "vae"), *[(p, "vae/oof") for p in oof_paths],
            *[(p, "vae/models") for p in sorted((vdir / c3.VAE_MODEL_DIR).glob("*.json"))]]
    done = c3.finish_board(
        vdir=vdir, spec=spec, preds=res["pred"], samples=samples, exp_log=log, code_md5=code_md5, input_md5s=md5s,
        code_root=CODE_ROOT, n_jobs=cfg["n_jobs"], smoke=smoke, tracker=tracker, stage_run_id=sr,
        tags={"코드 검수 완료": "예" if cfg["code_reviewed"] else "아니오", "바탕 판": tt["name"]},
        params={"algorithm": "VAE", "W": W, "mode": src.mode, "input_set_id": spec["input_set"], "seed": cfg["seed"],
                "n_models": counts["total"], "tree_target": tt["name"]},
        judgement=s3.NA_JUDGE, artifacts=arts,
        extra_meta={"models": res["models"], "block_plan": counts, "tree_target": tt["name"]})
    print(f"{c3.VAE_T}: CV {done['cv_band_weighted']}")
    return done


# ================================================================ 판정
def judge(cfg, inputs_root: Path, out: Path, code_md5: str, tracker=None) -> dict:
    dpath = out / c3.DECISION[STAGE]
    if dpath.exists():
        print(f"3-1 판정 이미 있음: {dpath}")
        return rn.read_json(dpath)
    st = state0(cfg, inputs_root)
    sp = board_spec(cfg, inputs_root, st)
    _, done, cv = rn.load_done(out, sp.name)
    if done["spec_md5"] != sp.md5():
        raise c3.CombineError("끝난 3-1 판 설정이 지금과 다름")
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    rel = f"{cfg['dirs']['3-1'].rstrip('/')}/{sp.name}"
    cand = c3.board_ref(inputs_root, sp.name, rel)
    cur = st["current_best"]
    dir_of = {sp.name: out / sp.name, cur["name"]: resolve(inputs_root, cur["dir"])}
    res = c3.judge_step(stacking=[c3.candidate(cand)], replacement=[], current=c3.candidate(cur),
                        rescore=c3.rescore_from(dir_of, samples))
    adopted = bool(res["stacking"]["judge"]["adopt"])
    after = copy.deepcopy(st)
    after.pop("sources", None)
    after["after_step"] = STAGE
    after["board_3_1"] = {"name": sp.name, "algorithm": sp.algorithm, "dir": rel, "cv": cand["cv"]}
    if adopted:
        after["current_best"] = cand
        after["tree_target"] = {"name": sp.name, "algorithm": sp.algorithm, "dir": rel, "kind": "tree"}
        after["anomaly_adopted"] = True
        for a in after["current_inputs"]:
            after["current_inputs"][a]["anomaly"] = True       # §6 2차 공통 '현재 입력 = 변수 묶음 + 채택한 칼럼'
    tt = st["tree_target"]
    tt_ref = c3.board_ref(inputs_root, tt["name"], tt["dir"])
    vdone = rn.read_json(out / c3.VAE_T / "done.json")
    alerts = list(res["stacking"]["judge"]["alerts"])
    decision = {
        "stage": STAGE, "candidate": sp.name, "adopted": adopted, "judge": res,
        "tree_target_before_after": {"tree_target": tt["name"], "tree_target_cv": tt_ref["cv"],
                                     "with_vae": sp.name, "with_vae_cv": cand["cv"],
                                     "tree_target_is_current_best": tt["name"] == cur["name"]},
        "vae_standalone": {"name": c3.VAE_T, "cv": vdone["cv_band_weighted"], "role": "비교만"},
        "state_before": st, "state_after": after, "alerts": alerts, "code_md5": code_md5, "utc": c3._utc(),
    }
    tags = c3.judgement_tags(res)
    rs = res["stacking"]["judge"]["rescore"]
    c3.set_judgements(tracker, {sp.name: done.get("mlflow_run_id")}, tags,
                      {sp.name: rs["candidate"]} if rs else None)
    c3.append_chain(inputs_root, cfg["chain_log"],
                    {"step": STAGE, "judgement": "채택" if adopted else "버림", "candidate": sp.name,
                     "before": cur["name"], "after": after["current_best"]["name"],
                     "tree_target": after["tree_target"]["name"], "current_inputs": after["current_inputs"],
                     "utc": decision["utc"]}, tracker)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    for a in alerts:
        log.append("alert", stage=STAGE, message=a)
        print(f"[알림] {a}")
    log.append("decision", stage=STAGE, candidate=sp.name, adopted=adopted, current_best=after["current_best"]["name"])
    cf.dump_scores_json(dpath, decision)
    if tracker is not None:
        tracker.log_artifact(tracker.stage_run(s3.EXPERIMENT, STAGE), dpath, "decision")
    print(f"3-1: {sp.name} {'채택' if adopted else '버림'} — 현재 최고 판 = {after['current_best']['name']}")
    return decision


# ================================================================ 보고(test — 판정 뒤)
def report(cfg, inputs_root: Path, out: Path, reason: str, tracker=None) -> Path:
    dpath = out / c3.DECISION[STAGE]
    if not dpath.exists():
        raise SystemExit("3-1 판정을 먼저(test 는 단계가 끝난 뒤 — §1-3)")
    d = rn.read_json(dpath)
    permit = ev.report_permit(STAGE, reason)
    samples_path = resolve(inputs_root, cfg["samples"])
    samples = ev.SampleTable.read(samples_path)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    rows = []
    if cfg.get("te7a_dir"):
        rows.append(rn.te7a_baseline(resolve(inputs_root, cfg["te7a_dir"]), samples, log))
    rows.append(s3.baseline_1_row(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"]))
    st = d["state_before"]
    cur, tt = st["current_best"], st["tree_target"]
    rows.append(s3.base_row_from(resolve(inputs_root, cur["dir"]), cur["name"],
                                 f"현재 최고 판(3-1 판정 상대) = {cur['source']}"))
    if tt["name"] != cur["name"]:
        rows.append(s3.base_row_from(resolve(inputs_root, tt["dir"]), tt["name"], "트리 대상 판(3-1 바탕)"))
    vd = rn.read_json(out / c3.VAE_T / "done.json")
    rows.append(s3.base_row_from(out / c3.VAE_T, c3.VAE_T, "이상 점수 단독(비교만) — fold train 전체·test 전 전체 VAE",
                                 vd.get("mlflow_run_id")))
    sr = tracker.stage_run(s3.EXPERIMENT, STAGE) if tracker is not None else None
    notes = ["이상 점수 칼럼 = 엄격 OOF(train 행 = 블록 OOF, validation·test 쌍 = 그 판 전체 VAE) — §7-3",
             "부스팅 판: CV = 교차 N 예측, threshold = 공통 N 예측, test = refit 공통 N(§3·§1-2)"]
    return rn.report_stage(out_dir=out, stage=STAGE, names=[d["candidate"]], samples=samples,
                           samples_path=samples_path, permit=permit,
                           incomplete=ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"])),
                           exp_log=log, tracker=tracker, stage_run_id=sr, base_rows=rows, extra_notes=notes)


# ================================================================ 진입점
def factories(cfg, rp, inputs_root: Path) -> sc.Factories:
    from lst import wiring as wr

    w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    return sc.Factories(make_inputs=c3.anomaly_make_inputs(w.make_inputs, samples, inputs_root),
                        make_model=w.make_model,
                        input_md5s={**wr.input_md5s(cfg["context"], rp.root, inputs_root),
                                    **c3.vae_input_md5s(inputs_root, cfg["dirs"]["3-1"])},
                        fit_kwargs=w.fit_kwargs)


def main(argv=None, *, factories_fn=None, tracker_fn=sc.make_tracker, check_resources_fn=None, vae_feats=None,
         vae_model_factory=None, subprocess_vae: bool = True):
    argv = list(sys.argv[1:] if argv is None else argv)
    ns = _pre(argv)
    if not ns.inputs or not ns.config or not ns.out:
        raise SystemExit("--inputs·--config·--out 필요")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    out = Path(ns.out).expanduser().resolve()
    cfg = load_cfg(ns.config)
    if not ns.verify_run and out != resolve(inputs_root, cfg["dirs"]["3-1"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[3-1]")
    root = Path(ns.root).expanduser().resolve()
    if ns.vae:
        return run_vae(cfg, root, inputs_root, out, feats=vae_feats, model_factory=vae_model_factory,
                       check=check_resources_fn, tracker=tracker_fn(cfg))
    if ns.report:
        if not ns.reason:
            raise SystemExit("--report 에는 --reason 이 필요(실험 로그)")
        p = report(cfg, inputs_root, out, ns.reason, tracker_fn(cfg))
        print(f"보고 표: {p}")
        return p

    def specs_fn(c):
        return [board_spec(c, inputs_root)]

    kw = dict(script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=SCHEMA, specs_fn=specs_fn,
              factories_fn=factories_fn or factories, tracker_fn=tracker_fn, check_resources=check_resources_fn)
    if ns.variant or ns.list or ns.verify_run:
        return sc.stage_main(argv, **kw)
    # 기본: 판 목록 → VAE → 3-1 판 → 판정
    write_plan(cfg, inputs_root, out)
    if not (out / c3.VAE_T / "done.json").exists():
        if subprocess_vae:
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), *argv, "--vae"])
            if rc != 0:
                rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])).append(
                    "stopped", stage=STAGE, name=c3.VAE_T, returncode=rc, reason=rn._rc_reason(rc))
                raise rn.RunStopped(f"{c3.VAE_T} 멈춤 — {rn._rc_reason(rc)}. 원인을 없앤 뒤 같은 명령을 다시(B-1)")
        else:
            run_vae(cfg, root, inputs_root, out, feats=vae_feats, model_factory=vae_model_factory,
                    check=check_resources_fn, tracker=tracker_fn(cfg))
    sc.stage_main(argv, **kw)
    code_md5 = rn.snapshot_code(CODE_ROOT, out)
    return judge(cfg, inputs_root, out, code_md5, tracker_fn(cfg))


if __name__ == "__main__":
    main()
