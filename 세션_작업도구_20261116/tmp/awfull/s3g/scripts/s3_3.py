# -*- coding: utf-8 -*-
"""s3_3.py — 단계 3-3 앙상블(트리 상위 3개 평균, 학습 없음) — 계획서 요약 15, §6 2차 공통(채택 판정·트리 대상 판)·단계 3-3,
§1-2·§1-3, 부록 A-12(3-3 백분위 평균 대조·구성 모델별 예측 분위수·판별 학습 횟수 앙상블 = 0)·A-14·A-15.

차례(기본 실행 한 번 — 3-2 판정 뒤):
  0. 3-3_plan.json: 구성 판·고르기 규칙(§1-3)
  1. 구성 트리 3개: 3-2 판이 채택됐으면 3-2 트리 3개(다시 씀), 아니면 3-0 트리 상위 3개를 각자 현재 입력으로 — 같은 판
     (3-0 판·3-1 판·3-2 판)이 있으면 다시 씀(기록 폴더 '3-3-<알고리즘>' + MLflow 새 run 'tag 원 판'), 없으면 학습
     (하위 프로세스 `--variant`, runner — 계획서 규칙상 보통 없음)
  2. 3-3-T3: (시료, h) 쌍마다 구성 3개 확률 단순 평균 — CV = 교차 N 예측 평균, threshold = 공통 N 예측 평균, test = refit 평균.
     대조(기록만) = 백분위 평균 CV(pct_cv_scores.json), 구성 모델별 예측 분위수(member_quantiles.csv)
  3. 판정: 3-3-T3(쌓는 후보) val 이 현재 최고 판보다 오르면 채택 → 새 현재 최고 판·트리 대상 판(앙상블)
     → 3-3_decision.json(state_after) + 사슬 기록 + MLflow 판정 tag
     + 단계 4 가 읽는 사슬 상태 logs/chain_state_3-3.json(lst/stage4 '사슬 상태' 꼴 — BestDef·MemberDef(이상 점수 칼럼
       AnomalyDef·④ stack4)·tree_top3·tree_rank_3_0b(칸 이름 그대로, 값 = 3-0 트리 6 CV 순위)·history(사슬 기록 줄))
       — 이 파일은 3-0b(1등 변수 빼기, scripts/s3_0b.py)가 읽고, 단계 4 는 3-0b 가 쓰는 logs/chain_state_3-0b.json 을 읽음
  보고(--report, 판정 뒤): 3-3-T3·백분위 평균 대조·구성 판·7-a·1차 베이스라인·현재 최고 판(판정 전) 줄 표

실행(서버 — 검수 뒤 지휘자, CPU 몇 분):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s3_3.py $A --out ~/cyj_listeria_final/results/s3/3-3 --config configs/s3_3.json
  $P scripts/s3_3.py $A --out ~/cyj_listeria_final/results/s3/3-3 --config configs/s3_3.json --report --reason "단계 3-3 보고"

구현 세부(계획서에 없음 — 담당자 확인): lst/stage3_combine 머리 (h)~(k). 새로 학습하는 구성 판 이름 = '3-3-<알고리즘>'(부록
A-14 에 없음), 다시 쓰는 구성 판 기록 폴더도 같은 이름.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
from pathlib import Path
from typing import Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

from lst import config as cf  # noqa: E402
from lst import scope as lst_scope  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage3 as s3  # noqa: E402
from lst import stage3_combine as c3  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.paths import resolve  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = "3-3"
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
DIRS_SCHEMA = {"x67-judge": cf.Field(str), "3-1": cf.Field(str), "3-2": cf.Field(str),
               "3-3": cf.Field(str)}
SCHEMA = {
    "decision_1_3": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "dirs": DIRS_SCHEMA,
    "chain_log": cf.Field(str),
    "chain_state": cf.Field(str),          # --inputs 아래 단계 4 가 읽는 사슬 상태(lst/stage4 '사슬 상태' 꼴)
}
QUANTILE_FILE = "member_quantiles.csv"
PCT_FILE = "pct_cv_scores.json"


def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run"):
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
    return cfg


# ================================================================ 구성 판
def state2(cfg, inputs_root: Path) -> dict:
    return c3.state_from(inputs_root, cfg["dirs"], "3-2")


def members(cfg, inputs_root: Path, st: Mapping | None = None) -> list[dict]:
    """구성 3개 = {algorithm, name(3-3-<알고리즘>), source(원 판 이름 또는 None), pred_dir(예측 폴더 rel), spec}."""
    st = st or state2(cfg, inputs_root)
    out_rel = cfg["dirs"]["3-3"].rstrip("/")
    res = []
    for t in st["tree_top3"]:
        alg = t["algorithm"]
        name = f"3-3-{alg}"
        if st["adopted_3_2"]:
            b = st["boards_3_2"][alg]
            res.append({"algorithm": alg, "name": name, "source": b["name"], "pred_dir": b["pred_dir"],
                        "spec": c3.source_spec(inputs_root, b["pred_dir"])})
            continue
        src = c3.source_spec(inputs_root, t["dir"])
        if st["anomaly_adopted"]:
            want = c3.spec_plus_anom(src, name=name, stage=STAGE, vae_rel=cfg["dirs"]["3-1"], base_variant=t["name"])
        else:
            want = dataclasses.replace(src, name=name, stage=STAGE, role=s3.ROLE_CAND, judgement=None,
                                       base_variant=t["name"], extra={"version": dict(src.extra["version"])})
        pool = [(t["name"], t["dir"])]
        if st.get("board_3_1"):
            pool.append((st["board_3_1"]["name"], st["board_3_1"]["dir"]))
        for b in (st.get("boards_3_2") or {}).values():
            pool.append((b["name"], b["pred_dir"]))
        hit = next(((n, d) for n, d in pool if c3.content_key(c3.source_spec(inputs_root, d)) == c3.content_key(want)),
                   None)
        if hit is not None:
            res.append({"algorithm": alg, "name": name, "source": hit[0], "pred_dir": hit[1],
                        "spec": c3.source_spec(inputs_root, hit[1])})
        else:
            res.append({"algorithm": alg, "name": name, "source": None, "pred_dir": f"{out_rel}/{name}", "spec": want})
    return res


def train_specs(cfg, inputs_root: Path) -> list[rn.VariantSpec]:
    return [m["spec"] for m in members(cfg, inputs_root) if m["source"] is None]


def write_plan(cfg, inputs_root: Path, out: Path) -> dict:
    p = out / c3.PLAN[STAGE]
    st = state2(cfg, inputs_root)
    ms = members(cfg, inputs_root, st)
    plan = {"stage": STAGE, "stacking": [c3.T3], "members": {m["name"]: m["source"] or "새로 학습" for m in ms},
            "rule": "3-3-T3 val 이 현재 최고 판보다 소수 4자리에서 오르면 채택 — §6 2차 공통. 백분위 평균은 대조(기록만)",
            "current_best": st["current_best"]["name"], "utc": c3._utc()}
    if p.exists():
        old = rn.read_json(p)
        if old["members"] != plan["members"] or old["current_best"] != plan["current_best"]:
            raise c3.CombineError("3-3_plan.json 이 지금 구성·현재 최고 판과 다름")
        return old
    out.mkdir(parents=True, exist_ok=True)
    cf.dump_json(p, plan)
    return plan


def write_member_pointers(cfg, inputs_root: Path, out: Path, code_md5: str, tracker=None) -> None:
    for m in members(cfg, inputs_root):
        if m["source"] is None:
            continue
        src_dir = resolve(inputs_root, m["pred_dir"])
        sp = dataclasses.replace(m["spec"], name=m["name"], stage=STAGE, role=s3.ROLE_CAND, judgement=None,
                                 extra={**dict(m["spec"].extra), "source": m["source"]})
        src_done = rn.read_json(src_dir / "done.json")
        rid = c3.pointer_run(tracker, STAGE, sp.name, m["source"], src_dir, role=s3.ROLE_CAND,
                             code_reviewed=cfg["code_reviewed"], spec=sp, code_md5=code_md5,
                             selection_count=int(src_done["selection_count"]))
        s3.write_pointer(out, sp, src_dir, m["pred_dir"], code_md5, mlflow_run_id=rid)


# ================================================================ 3-3-T3(학습 없음)
def build_t3(cfg, inputs_root: Path, out: Path, code_md5: str, tracker=None) -> dict:
    ms = members(cfg, inputs_root)
    mem = [{"algorithm": m["algorithm"], "name": m["name"], "source": m["source"], "pred_dir": m["pred_dir"]}
           for m in ms]
    spec = c3.t3_spec(mem)
    vdir = out / c3.T3
    if c3.begin_board(vdir, out, spec, code_md5) == "done":
        return rn.read_json(vdir / "done.json")
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    smoke = any(rn.read_json(resolve(inputs_root, m["pred_dir"]) / "done.json").get("smoke") for m in ms) or None
    ens = c3.ensemble_preds([resolve(inputs_root, m["pred_dir"]) for m in ms])
    has_thr = any(rn.load_thr_preds(resolve(inputs_root, m["pred_dir"])) is not None for m in ms)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    qp = vdir / QUANTILE_FILE
    ens["quantiles"].to_csv(qp, index=False, encoding="utf-8-sig", float_format="%.17g")
    pct = ev.score_cv({f: ens["pct"][f] for f in c3.CV_FOLDS}, samples,
                      thr_preds=ens["pct_thr"] if has_thr else None, require_complete=not smoke)
    pct_count = log.cv_scored(STAGE, c3.T3_PCT)
    cf.dump_scores_json(vdir / PCT_FILE, {"name": c3.T3_PCT, "selection_count": pct_count, "cv": pct,
                                          "rule": "부록 A-12 백분위 평균(기준 분포 = 채점하는 쌍 자신의 예측) — 대조, 기록만"})
    md5s = {f"member:{m['name']}:{f}": cf.file_md5(resolve(inputs_root, m["pred_dir"]) / f)
            for m in ms for f in [f"pred_{s}.parquet" for s in rn.SPLIT_ORDER]}
    sr = tracker.stage_run(s3.EXPERIMENT, STAGE) if tracker is not None else None
    done = c3.finish_board(
        vdir=vdir, spec=spec, preds=ens["pred"], thr_preds=ens["thr"] if has_thr else None, samples=samples,
        exp_log=log, code_md5=code_md5, input_md5s=md5s, code_root=CODE_ROOT, n_jobs=cfg["n_jobs"],
        smoke={"from_members": True} if smoke else None, tracker=tracker, stage_run_id=sr,
        tags={"코드 검수 완료": "예" if cfg["code_reviewed"] else "아니오"},
        params={"algorithm": "T3", "members": ",".join(m["name"] for m in ms),
                "member_sources": ",".join(str(m["source"] or m["name"]) for m in ms), "train_count": 0},
        artifacts=[(qp, "ensemble"), (vdir / PCT_FILE, "ensemble")],
        done_extra={"members": mem, "pct_cv_band_weighted": pct["cv_band_weighted"]})
    if tracker is not None and done.get("mlflow_run_id"):
        tracker.log_metrics(done["mlflow_run_id"], {"pct_cv_band_weighted": pct["cv_band_weighted"]})
    print(f"{c3.T3}: CV {done['cv_band_weighted']} (백분위 평균 대조 {pct['cv_band_weighted']})")
    return done


# ================================================================ 판정
def judge(cfg, inputs_root: Path, out: Path, code_md5: str, tracker=None) -> dict:
    dpath = out / c3.DECISION[STAGE]
    if dpath.exists():
        print(f"3-3 판정 이미 있음: {dpath}")
        d = rn.read_json(dpath)
        write_chain_state(cfg, inputs_root, d)          # 없으면 씀(있으면 같은지만 확인)
        return d
    st = state2(cfg, inputs_root)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    rel = f"{cfg['dirs']['3-3'].rstrip('/')}/{c3.T3}"
    t3 = c3.board_ref(inputs_root, c3.T3, rel)
    cur = st["current_best"]
    dir_of = {c3.T3: out / c3.T3, cur["name"]: resolve(inputs_root, cur["dir"])}
    res = c3.judge_step(stacking=[c3.candidate(t3)], replacement=[], current=c3.candidate(cur),
                        rescore=c3.rescore_from(dir_of, samples))
    adopted = bool(res["stacking"]["judge"]["adopt"])
    tdone = rn.read_json(out / c3.T3 / "done.json")
    after = copy.deepcopy(st)
    after.pop("sources", None)
    after["after_step"] = STAGE
    after["board_3_3"] = {"name": c3.T3, "dir": rel, "cv": t3["cv"], "members": tdone["members"]}
    if adopted:
        after["current_best"] = t3
        after["tree_target"] = {"name": c3.T3, "algorithm": "T3", "dir": rel, "kind": "ensemble",
                                "members": tdone["members"]}
    alerts = list(res["stacking"]["judge"]["alerts"])
    decision = {"stage": STAGE, "candidate": c3.T3, "adopted": adopted, "judge": res,
                "pct_cv_band_weighted": tdone["pct_cv_band_weighted"], "members": tdone["members"],
                "state_before": st, "state_after": after, "alerts": alerts, "code_md5": code_md5, "utc": c3._utc()}
    rs = res["stacking"]["judge"]["rescore"]
    c3.set_judgements(tracker, {c3.T3: tdone.get("mlflow_run_id")}, c3.judgement_tags(res),
                      {c3.T3: rs["candidate"]} if rs else None)
    c3.append_chain(inputs_root, cfg["chain_log"],
                    {"step": STAGE, "judgement": "채택" if adopted else "버림", "candidate": c3.T3,
                     "before": cur["name"], "after": after["current_best"]["name"],
                     "tree_target": after["tree_target"]["name"], "current_inputs": after["current_inputs"],
                     "utc": decision["utc"]}, tracker)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    for a in alerts:
        log.append("alert", stage=STAGE, message=a)
        print(f"[알림] {a}")
    log.append("decision", stage=STAGE, candidate=c3.T3, adopted=adopted, current_best=after["current_best"]["name"])
    cf.dump_scores_json(dpath, decision)
    if tracker is not None:
        tracker.log_artifact(tracker.stage_run(s3.EXPERIMENT, STAGE), dpath, "decision")
    print(f"3-3: {c3.T3} {'채택' if adopted else '버림'} — 현재 최고 판 = {after['current_best']['name']}")
    write_chain_state(cfg, inputs_root, decision)
    return decision


def write_chain_state(cfg, inputs_root: Path, decision: Mapping) -> Path:
    """단계 4 가 읽는 사슬 상태 파일(state_after 를 lst/stage4 꼴로 — 3-3_decision.json 은 그대로)."""
    p = c3.write_stage4_state(inputs_root, cfg["dirs"], decision["state_after"], STAGE, cfg["chain_log"],
                              cfg["chain_state"])
    print(f"단계 4 사슬 상태: {p}")
    return p


# ================================================================ 보고(test — 판정 뒤)
def report(cfg, inputs_root: Path, out: Path, reason: str, tracker=None) -> Path:
    dpath = out / c3.DECISION[STAGE]
    if not dpath.exists():
        raise SystemExit("3-3 판정을 먼저(test 는 단계가 끝난 뒤 — §1-3)")
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
                                 f"현재 최고 판(3-3 판정 전) = {cur['source']}"))
    names = []
    for m in d["members"]:
        if m["source"] is None:
            names.append(m["name"])
        else:
            own = rn.read_json(out / m["name"] / "done.json")
            rows.append(s3.base_row_from(resolve(inputs_root, m["pred_dir"]), m["name"],
                                         f"구성 판 = {m['source']} 다시 씀", own.get("mlflow_run_id")))
    tdone = rn.read_json(out / c3.T3 / "done.json")
    rows.append(s3.base_row_from(out / c3.T3, c3.T3, "트리 상위 3개 확률 단순 평균(앙상블)", tdone.get("mlflow_run_id")))
    ens = c3.ensemble_preds([resolve(inputs_root, m["pred_dir"]) for m in d["members"]])
    has_thr = (out / c3.T3 / f"pred_{rn.THR_PREFIX}fold1.parquet").exists()
    rows.append(rn.BaseRow(c3.T3_PCT, "백분위 평균 대조(기록만 — 기준 분포 = 채점하는 쌍 자신의 예측)",
                           {f: ens["pct"][f] for f in c3.CV_FOLDS}, ens["pct"]["refit"],
                           log.count_for(STAGE, c3.T3_PCT), None,
                           thr_preds=({f: ens["pct_thr"][f] for f in c3.CV_FOLDS} if has_thr else None)))
    sr = tracker.stage_run(s3.EXPERIMENT, STAGE) if tracker is not None else None
    notes = ["3-3-T3 = 구성 트리 3개 (시료, h) 확률 단순 평균 — CV = 교차 N 예측 평균, threshold = 공통 N 예측 평균",
             f"구성 모델별 예측 분위수(5·50·95%) = {c3.T3}/{QUANTILE_FILE}",
             "백분위 평균 = 운영에서 기준 분포를 만들 수 없어 대조로만(부록 A-12)"]
    return rn.report_stage(out_dir=out, stage=STAGE, names=names, samples=samples, samples_path=samples_path,
                           permit=permit, incomplete=ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"])),
                           exp_log=log, tracker=tracker, stage_run_id=sr, base_rows=rows, extra_notes=notes)


# ================================================================ 진입점
def factories(cfg, rp, inputs_root: Path) -> sc.Factories:
    from lst import wiring as wr

    w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    md5s = dict(wr.input_md5s(cfg["context"], rp.root, inputs_root))
    if any("anomaly" in s.extra for s in train_specs(cfg, inputs_root)):
        md5s.update(c3.vae_input_md5s(inputs_root, cfg["dirs"]["3-1"]))
    return sc.Factories(make_inputs=c3.anomaly_make_inputs(w.make_inputs, samples, inputs_root),
                        make_model=w.make_model, input_md5s=md5s, fit_kwargs=w.fit_kwargs)


def main(argv=None, *, factories_fn=None, tracker_fn=sc.make_tracker, check_resources_fn=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ns = _pre(argv)
    if not ns.inputs or not ns.config or not ns.out:
        raise SystemExit("--inputs·--config·--out 필요")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    out = Path(ns.out).expanduser().resolve()
    cfg = load_cfg(ns.config)
    if not ns.verify_run and out != resolve(inputs_root, cfg["dirs"]["3-3"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[3-3]")
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
    write_plan(cfg, inputs_root, out)
    code_md5 = rn.snapshot_code(CODE_ROOT, out)
    write_member_pointers(cfg, inputs_root, out, code_md5, tracker_fn(cfg))
    if train_specs(cfg, inputs_root):
        sc.stage_main(argv, **kw)
    build_t3(cfg, inputs_root, out, code_md5, tracker_fn(cfg))
    return judge(cfg, inputs_root, out, code_md5, tracker_fn(cfg))


if __name__ == "__main__":
    main()
