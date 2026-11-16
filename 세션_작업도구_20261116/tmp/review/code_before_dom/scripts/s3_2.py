# -*- coding: utf-8 -*-
"""s3_2.py — 단계 3-2 스태킹(트리 입력 + 이상 점수 판·④ LIMS 만 스태킹) — 계획서 요약 15, §6 2차 공통(채택 판정 (가)→(나)·
교체 후보·트리 대상 판)·단계 3-2·추가 판 ④, §7-3(블록 OOF·같은 행 모델 재사용), §1-2·§1-3, 부록 A-9(3-2 12·④ 118)·A-12·
A-14·A-15.

차례(기본 실행 한 번 — 끝난 차례는 건너뜀, 3-1 판정 뒤 — 1위 트리의 3-2 판 = 3-1 판이라 학습 3-2 = 2판 × 4 = 8):
  0. 3-2_plan.json: 판 목록·고르기 규칙(§1-3)
  1. 3-2-<알고리즘>: 3-0 트리 상위 3개마다 그 트리 판 설정 그대로(부스팅은 §3 반복 수만 다시) + 현재 입력 + 이상 점수 칼럼
     (3-1 VAE, 채택 여부와 무관). 3-1 판과 같은 판(트리 대상 판의 3-2 판)은 학습 없이 다시 씀 → 원 판을 가리키는 기록 폴더 +
     MLflow 새 run(tag '원 판'). 나머지는 하위 프로세스 `--variant`(runner — 이상 점수 칼럼 = 3-1-VAEt 저장 값).
  2. x4-L6 ④(하위 프로세스 `--x4`, CPU): 트리 6종(라이브러리 기본 설정·기본 반복 수, 범주 5 + h·g) × 19(전역 6 + 엄격 13 —
     fold 1·2 = 4블록, fold 3 = 5블록, 같은 행 모델 재사용) → OOF 6칼럼 → 로지스틱 메타(표준화) — fold validation 메타 입력 =
     fold train 전체 트리 예측, refit 메타 train = 전역 블록 OOF, test 메타 입력 = test 전 전체 트리 예측
  3. 판정: (가) 3-2 판 셋 중 val 1위 대 현재 최고 판 → (나) ④ 대 그때의 현재 최고 판(교체 후보 — 이기면 현재 최고 판만 바뀜).
     → 3-2_decision.json(state_after) + 사슬 기록 + MLflow 판정 tag
  보고(--report, 판정 뒤): 3-2 판(학습한 것)·x4-L6 test + 7-a·1차 베이스라인·현재 최고 판(판정 전)·다시 쓴 판 줄

실행(서버 — 검수 뒤 지휘자):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s3_2.py $A --out ~/cyj_listeria_final/results/s3/3-2 --config configs/s3_2.json
  $P scripts/s3_2.py $A --out ~/cyj_listeria_final/results/s3/3-2 --config configs/s3_2.json --report --reason "단계 3-2 보고"

구현 세부(계획서에 없음 — 담당자 확인): lst/stage3_combine 머리 (c)·(d)·(f)·(g)·(i)~(k). ④ 는 하위 프로세스 하나에서 한 번(멈추면
판 폴더를 _incomplete 로 옮기고 처음부터). 3-2 판 '바탕 판' = 그 트리의 3-0 판 이름.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
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
from lst import evaluate as ev  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage3 as s3  # noqa: E402
from lst import stage3_combine as c3  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.paths import resolve  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = "3-2"
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
X4_COMPONENTS = ("missing",)
DIRS_SCHEMA = {"x67-judge": cf.Field(str), "3-1": cf.Field(str), "3-2": cf.Field(str)}
SCHEMA = {
    "decision_1_3": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "dirs": DIRS_SCHEMA,
    "chain_log": cf.Field(str),
    "x4_save_models": cf.Field(bool),
}


def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run", "--x4"):
        ap.add_argument(a, action="store_true")
    return ap.parse_known_args(list(argv))[0]


def load_cfg(path) -> dict:
    cfg = sc.load_cfg(path, SCHEMA)
    if cfg["stage"] != STAGE or cfg["order"] != s3.ORDER or cfg["experiment"] != s3.EXPERIMENT:
        raise cf.ConfigError(f"stage '{STAGE}' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
    if cfg["n_jobs"] != 8:
        raise cf.ConfigError("n_jobs 8")
    return cfg


def _smoke(cfg) -> dict | None:
    return {k: cfg["smoke"][k] for k in rn.SMOKE_KEYS} if "smoke" in cfg else None


# ================================================================ 판 설정
def state1(cfg, inputs_root: Path) -> dict:
    return c3.state_from(inputs_root, cfg["dirs"], "3-1")


def boards(cfg, inputs_root: Path, st: Mapping | None = None) -> list[dict]:
    """상위 트리 3개마다 3-2 판(설정 + 다시 쓸 원 판). reuse = 같은 판인 3-1 판 정보(없으면 None)."""
    st = st or state1(cfg, inputs_root)
    b31 = st["board_3_1"]
    spec31 = c3.source_spec(inputs_root, b31["dir"])
    out = []
    for t in st["tree_top3"]:
        src = c3.source_spec(inputs_root, t["dir"])
        cur = st["current_inputs"][src.algorithm]["version"]
        if dict(src.extra["version"]) != dict(cur):
            raise c3.CombineError(f"{t['name']}: 판 설정 변수 묶음과 현재 입력이 다름")
        sp = c3.spec_plus_anom(src, name=c3.name_3_2(src.algorithm), stage=STAGE, vae_rel=cfg["dirs"]["3-1"],
                               base_variant=t["name"])
        same = c3.content_key(sp) == c3.content_key(spec31)
        out.append({"algorithm": src.algorithm, "spec": sp, "end_name": t["name"], "reuse": b31 if same else None})
    return out


def train_specs(cfg, inputs_root: Path) -> list[rn.VariantSpec]:
    return [b["spec"] for b in boards(cfg, inputs_root) if b["reuse"] is None]


def pred_dir_rel(cfg, b: Mapping) -> str:
    return b["reuse"]["dir"] if b["reuse"] else f"{cfg['dirs']['3-2'].rstrip('/')}/{b['spec'].name}"


def write_plan(cfg, inputs_root: Path, out: Path) -> dict:
    p = out / c3.PLAN[STAGE]
    st = state1(cfg, inputs_root)
    bs = boards(cfg, inputs_root, st)
    plan = {"stage": STAGE, "stacking": [b["spec"].name for b in bs], "replacement": [c3.X4],
            "reused": {b["spec"].name: b["reuse"]["name"] for b in bs if b["reuse"]},
            "rule": "(가) 3-2 판 셋 중 val 1위가 현재 최고 판보다 소수 4자리에서 오르면 채택 → (나) ④ 가 그때의 현재 최고 판보다 "
                    "오르면 채택(교체 후보 — 현재 최고 판만 바뀜) — §6 2차 공통",
            "current_best": st["current_best"]["name"], "tree_target": st["tree_target"]["name"],
            "spec_md5": {b["spec"].name: b["spec"].md5() for b in bs}, "utc": c3._utc()}
    if p.exists():
        old = rn.read_json(p)
        if old["spec_md5"] != plan["spec_md5"] or old["current_best"] != plan["current_best"]:
            raise c3.CombineError("3-2_plan.json 이 지금 판 설정·현재 최고 판과 다름")
        return old
    out.mkdir(parents=True, exist_ok=True)
    cf.dump_json(p, plan)
    return plan


def write_pointers(cfg, inputs_root: Path, out: Path, code_md5: str, tracker=None) -> list[str]:
    """3-1 판과 같은 3-2 판 = 학습 없이 다시 씀(원 판 기록 폴더 + MLflow 새 run 'tag 원 판')."""
    names = []
    for b in boards(cfg, inputs_root):
        if not b["reuse"]:
            continue
        src_dir = resolve(inputs_root, b["reuse"]["dir"])
        sp = dataclasses.replace(b["spec"], extra={**dict(b["spec"].extra), "source": b["reuse"]["name"]})
        src_done = rn.read_json(src_dir / "done.json")
        rid = c3.pointer_run(tracker, STAGE, sp.name, b["reuse"]["name"], src_dir, role=s3.ROLE_CAND,
                             code_reviewed=cfg["code_reviewed"], spec=sp, code_md5=code_md5,
                             selection_count=int(src_done["selection_count"]))
        s3.write_pointer(out, sp, src_dir, b["reuse"]["dir"], code_md5, mlflow_run_id=rid)
        names.append(sp.name)
    return names


# ================================================================ ④(하위 프로세스)
def run_x4(cfg, root: Path, inputs_root: Path, out: Path, *, feats=None, check=None, tracker=None) -> dict:
    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    W = int(dec["W"])
    sp = c3.x4_spec(cfg["x4_save_models"], W)
    spec = sp.to_dict()
    vdir = out / c3.X4
    code_md5 = rn.snapshot_code(CODE_ROOT, out)
    if c3.begin_board(vdir, out, spec, code_md5) == "done":
        print(f"{c3.X4}: 이미 끝남")
        return rn.read_json(vdir / "done.json")
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    known = samples.known_frame()
    smoke = _smoke(cfg)
    ctx_md5s: dict = {}
    if feats is None:
        from lst import inputs as ip
        from lst import wiring as wr

        ctx = ip.load_context(cfg["context"], root, inputs_root, components=list(X4_COMPONENTS))
        feats = c3.LimsTreeFeatures(ctx, W)
        ctx_md5s = wr.input_md5s({**cfg["context"], "results": {k: v for k, v in cfg["context"]["results"].items()
                                                                if k in ("samples", "missing_lists")},
                                  "data": {}}, root, inputs_root)
    plan = bk.block_plan(reuse=True)
    if bk.plan_counts(plan)["total"] != 19:
        raise c3.CombineError("④ 블록 모델 수 ≠ 19(부록 A-9)")
    res = c3.build_x4(
        feats=feats, samples=samples, plan=plan, W=W,
        rows_of_model=lambda m: c3.model_train_rows(m, samples, W, smoke=smoke, seed=cfg["seed"], known=known),
        rows_of_split=lambda s: c3.split_train_rows(samples, s, W, smoke=smoke, seed=cfg["seed"], known=known),
        seed=cfg["seed"], n_jobs=cfg["n_jobs"], vdir=vdir, max_pairs_per_h=smoke["max_pairs_per_h"] if smoke else None,
        check=check or (lambda: check_resources(out)), save_full_models=cfg["x4_save_models"])
    oof_paths = c3.write_oof(vdir, res["oof"])
    ptab = vdir / "block_plan.csv"
    bk.plan_table(plan).to_csv(ptab, index=False, encoding="utf-8-sig")
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    md5s = {**cf.inputs_md5(inputs_root, [cfg["samples"], cfg["incomplete_months"]]), **ctx_md5s}
    sr = tracker.stage_run(s3.EXPERIMENT, STAGE) if tracker is not None else None
    arts = [(ptab, "stack"), *[(p, "stack/oof") for p in oof_paths],
            *[(p, "stack/meta") for p in sorted(vdir.glob("meta_*.json"))],
            *[(p, "stack/base_models") for p in sorted((vdir / c3.X4_MODEL_DIR).glob("*.json"))]]
    done = c3.finish_board(
        vdir=vdir, spec=spec, preds=res["pred"], samples=samples, exp_log=log, code_md5=code_md5, input_md5s=md5s,
        code_root=CODE_ROOT, n_jobs=cfg["n_jobs"], smoke=smoke, tracker=tracker, stage_run_id=sr,
        tags={"코드 검수 완료": "예" if cfg["code_reviewed"] else "아니오"},
        params={"algorithm": "stack-LR", "base": ",".join(c3.TREE6), "input_set_id": sp.input_set,
                "seed": cfg["seed"], "train_count": c3.X4_TRAIN_COUNT},
        artifacts=arts, extra_meta={"models": res["models"], "W_for_bounds": W})
    print(f"{c3.X4}: CV {done['cv_band_weighted']}")
    return done


# ================================================================ 판정
def judge(cfg, inputs_root: Path, out: Path, code_md5: str, tracker=None) -> dict:
    dpath = out / c3.DECISION[STAGE]
    if dpath.exists():
        print(f"3-2 판정 이미 있음: {dpath}")
        return rn.read_json(dpath)
    st = state1(cfg, inputs_root)
    bs = boards(cfg, inputs_root, st)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    stack, dir_of, run_ids, info32 = [], {}, {}, {}
    for b in bs:
        name = b["spec"].name
        pd_rel = pred_dir_rel(cfg, b)
        own = rn.read_json(out / name / "done.json")          # 학습한 판·다시 쓴 판 기록 폴더 모두
        if own["spec_md5"] != b["spec"].md5() and not b["reuse"]:
            raise c3.CombineError(f"끝난 {name} 설정이 지금과 다름")
        ref = c3.board_ref(inputs_root, name, pd_rel)
        ref["train_count"] = 4                                   # 부록 A-12 트리 판 = 4(다시 쓴 판도 같은 판)
        stack.append(ref)
        dir_of[name] = resolve(inputs_root, pd_rel)
        run_ids[name] = own.get("mlflow_run_id")
        info32[b["algorithm"]] = {"name": name, "pred_dir": pd_rel, "reused_from": b["reuse"]["name"] if b["reuse"] else None,
                                  "end_name": b["end_name"], "cv": ref["cv"]}
    x4_rel = f"{cfg['dirs']['3-2'].rstrip('/')}/{c3.X4}"
    x4 = c3.board_ref(inputs_root, c3.X4, x4_rel, replacement=True)
    dir_of[c3.X4] = out / c3.X4
    run_ids[c3.X4] = rn.read_json(out / c3.X4 / "done.json").get("mlflow_run_id")
    cur = st["current_best"]
    dir_of[cur["name"]] = resolve(inputs_root, cur["dir"])
    res = c3.judge_step(stacking=[c3.candidate(r) for r in stack], replacement=[c3.candidate(x4)],
                        current=c3.candidate(cur), rescore=c3.rescore_from(dir_of, samples))
    after = copy.deepcopy(st)
    after.pop("sources", None)
    after["after_step"] = STAGE
    after["boards_3_2"] = info32
    a_stack = bool(res["stacking"]["judge"]["adopt"])
    a_repl = bool(res["replacement"]["judge"]["adopt"])
    if a_stack:
        top = next(r for r in stack if r["name"] == res["stacking"]["top"])
        alg = next(a for a, v in info32.items() if v["name"] == top["name"])
        after["current_best"] = top
        after["tree_target"] = {"name": top["name"], "algorithm": alg, "dir": top["dir"], "kind": "tree"}
        after["adopted_3_2"] = True
    if a_repl:
        after["current_best"] = x4                              # 교체 후보 — 트리 대상 판·현재 입력 그대로
    alerts = list(res["stacking"]["judge"]["alerts"]) + list(res["replacement"]["judge"]["alerts"])
    decision = {"stage": STAGE, "judge": res, "adopted_stacking": a_stack, "adopted_replacement": a_repl,
                "boards_3_2": info32, "x4": {"name": c3.X4, "dir": x4_rel, "cv": x4["cv"]},
                "state_before": st, "state_after": after, "alerts": alerts, "code_md5": code_md5, "utc": c3._utc()}
    resc = {}
    for kind in ("stacking", "replacement"):
        rs = res[kind]["judge"]["rescore"]
        if rs:
            resc[res[kind]["top"]] = rs["candidate"]
    c3.set_judgements(tracker, run_ids, c3.judgement_tags(res), resc)
    for kind, flag in (("stacking", a_stack), ("replacement", a_repl)):
        r = res[kind]
        c3.append_chain(inputs_root, cfg["chain_log"],
                        {"step": STAGE if kind == "stacking" else f"{STAGE} ④", "judgement": "채택" if flag else "버림",
                         "candidate": r["top"], "before": r["against"],
                         "after": r["top"] if flag else r["against"], "tree_target": after["tree_target"]["name"],
                         "current_inputs": after["current_inputs"], "utc": decision["utc"]}, tracker)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    for a in alerts:
        log.append("alert", stage=STAGE, message=a)
        print(f"[알림] {a}")
    log.append("decision", stage=STAGE, stacking_top=res["stacking"]["top"], adopted_stacking=a_stack,
               adopted_x4=a_repl, current_best=after["current_best"]["name"])
    cf.dump_scores_json(dpath, decision)
    if tracker is not None:
        tracker.log_artifact(tracker.stage_run(s3.EXPERIMENT, STAGE), dpath, "decision")
    print(f"3-2: 3-2 1위 {res['stacking']['top']} {'채택' if a_stack else '버림'}, ④ {'채택' if a_repl else '버림'} — "
          f"현재 최고 판 = {after['current_best']['name']}")
    return decision


# ================================================================ 보고(test — 판정 뒤)
def report(cfg, inputs_root: Path, out: Path, reason: str, tracker=None) -> Path:
    dpath = out / c3.DECISION[STAGE]
    if not dpath.exists():
        raise SystemExit("3-2 판정을 먼저(test 는 단계가 끝난 뒤 — §1-3)")
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
    cur = d["state_before"]["current_best"]
    rows.append(s3.base_row_from(resolve(inputs_root, cur["dir"]), cur["name"],
                                 f"현재 최고 판(3-2 판정 전) = {cur['source']}"))
    names = []
    for alg, b in d["boards_3_2"].items():
        if b["reused_from"]:
            own = rn.read_json(out / b["name"] / "done.json")
            rows.append(s3.base_row_from(resolve(inputs_root, b["pred_dir"]), b["name"],
                                         f"3-2 판 = {b['reused_from']} 다시 씀(학습 없음)", own.get("mlflow_run_id")))
        else:
            names.append(b["name"])
    names.append(c3.X4)
    sr = tracker.stage_run(s3.EXPERIMENT, STAGE) if tracker is not None else None
    notes = ["3-2 판 = 3-0 트리 상위 3개 + 이상 점수 칼럼(쌓는 후보), x4-L6 = ④ LIMS 만 스태킹(교체 후보)",
             "x4-L6 기본 트리 = 라이브러리 기본 설정(반복 수·CatBoost 학습률 = 학습마다 자동값)",
             "부스팅 판: CV = 교차 N 예측, threshold = 공통 N 예측, test = refit 공통 N(§3·§1-2)"]
    return rn.report_stage(out_dir=out, stage=STAGE, names=names, samples=samples, samples_path=samples_path,
                           permit=permit, incomplete=ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"])),
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


def main(argv=None, *, factories_fn=None, tracker_fn=sc.make_tracker, check_resources_fn=None, x4_feats=None,
         subprocess_x4: bool = True):
    argv = list(sys.argv[1:] if argv is None else argv)
    ns = _pre(argv)
    if not ns.inputs or not ns.config or not ns.out:
        raise SystemExit("--inputs·--config·--out 필요")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    out = Path(ns.out).expanduser().resolve()
    cfg = load_cfg(ns.config)
    if not ns.verify_run and out != resolve(inputs_root, cfg["dirs"]["3-2"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[3-2]")
    root = Path(ns.root).expanduser().resolve()
    if ns.x4:
        return run_x4(cfg, root, inputs_root, out, feats=x4_feats, check=check_resources_fn, tracker=tracker_fn(cfg))
    if ns.report:
        if not ns.reason:
            raise SystemExit("--report 에는 --reason 이 필요(실험 로그)")
        p = report(cfg, inputs_root, out, ns.reason, tracker_fn(cfg))
        print(f"보고 표: {p}")
        return p

    def specs_fn(c):
        return train_specs(c, inputs_root)

    kw = dict(script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=SCHEMA, specs_fn=specs_fn,
              factories_fn=factories_fn or factories, tracker_fn=tracker_fn, check_resources=check_resources_fn)
    if ns.variant or ns.list or ns.verify_run:
        return sc.stage_main(argv, **kw)
    # 기본: 판 목록 → 다시 쓰는 판 → 3-2 판 학습 → ④ → 판정
    write_plan(cfg, inputs_root, out)
    code_md5 = rn.snapshot_code(CODE_ROOT, out)
    write_pointers(cfg, inputs_root, out, code_md5, tracker_fn(cfg))
    if train_specs(cfg, inputs_root):
        sc.stage_main(argv, **kw)
    if not (out / c3.X4 / "done.json").exists():
        if subprocess_x4:
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), *argv, "--x4"])
            if rc != 0:
                rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])).append(
                    "stopped", stage=STAGE, name=c3.X4, returncode=rc, reason=rn._rc_reason(rc))
                raise rn.RunStopped(f"{c3.X4} 멈춤 — {rn._rc_reason(rc)}. 원인을 없앤 뒤 같은 명령을 다시(B-1)")
        else:
            run_x4(cfg, root, inputs_root, out, feats=x4_feats, check=check_resources_fn, tracker=tracker_fn(cfg))
    return judge(cfg, inputs_root, out, code_md5, tracker_fn(cfg))


if __name__ == "__main__":
    main()
