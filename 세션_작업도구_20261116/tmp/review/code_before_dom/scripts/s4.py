# -*- coding: utf-8 -*-
"""s4.py — 단계 4 다운사이징(4-a·4-a0·미리보기·①·②·4-c·4-c-L)·판정 — 계획서 요약 16·19, §6 2차 공통·단계 4·추가 판
①②, 부록 A-6·A-8·A-9·A-12·A-14·A-15. 공통 = lst/stage4.

설정 sub(한 설정 = 결과 폴더 하나 — --out 은 --inputs 아래 dirs[<sub 폴더>]):
  prep   학습 없음: 줄이기(fold 1~3·refit 마다 Theil's U 표·1위 칼럼·검출 0건 값 행 빼기 → kept_keys·theil_<판>.csv·
         reduction_<판>.json — dirs['sel-reduce']) + 상위 5 품목(4-c — dirs['sel-top5']).
  pre    미리보기 4-a(2-1 뒤 — A 의 1차 (W, 입력 방식) 판 설정, 1차 입력 그대로 + 단계 4~6 범주 숫자 인코딩(k = 2-1),
         줄인 train·h 10개, 반복 수 = 2차 사슬 규칙) — 4-a-pre-<A>, 비교만(고르기에 안 씀), test 는 단계 4 보고 때.
  T      단계 시작 때 현재 최고 판(사슬 상태 chain_state_in = 3-0b 판정 뒤 logs/chain_state_3-0b.json — v78)에 4-a(줄인 train·트리 부분 h 10개)·4-a0(h 1개) — 쌓는 후보.
         딥러닝 단독·결합·② 면 4-a0 한 판(4-a = 4-a0). 앙상블이면 구성마다 학습 뒤 평균 판(T3). 이상 점수 칼럼이 있으면
         VAE 값 모델을 줄인 자료로 다시(엄격·전역 블록 OOF). GPU 가 필요한 판(딥러닝·VAE)은 gpu_memory_fraction.
  X1     ① 당일 예측(줄인 train·h 1 = T = d − 1·h·g 칼럼 없음, 평가 = h 1 쌍만) — 비교만(h = 1 비교, 최종 모델과).
  NN     ② 60칸 신경망(GPU — 줄인 자료, 입력 = 단계 시작 때 트리 대상 판 입력을 한 T 기준으로) — 교체 후보.
  C      4-c(현재 최고 판을 fold train 검출 비중 상위 5 품목 시료로만 — 줄이기 없음·범주 처리 그대로)·4-c-L(트리 대상 판
         알고리즘으로 LIMS 입력만) — 비교만, 채점 = 그 품목 쌍만.
  judge  학습 없음(T·NN 끝난 뒤): (가) 4-a·4-a0 중 val 1위 대 현재 최고 판 → (나) ② 대 그때의 현재 최고 판(§6 2차 공통) →
         사슬 상태(chain_state_out)·사슬 기록 한 줄·4_decision.json, MLflow 판정 tag.
보고(--report --reason, judge 뒤): pre·T·X1·NN·C 설정마다(7-a·1차 베이스라인·단계 시작 때 현재 최고 판 줄 + 앙상블 판 줄,
C 는 같은 품목 쌍으로 거른 현재 최고 판 줄).

실행(서버 — 검수 뒤 지휘자):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/sel --config configs/s4_prep.json
  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/4-pre --config configs/s4_pre.json     # 2-1 뒤 아무 때
  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/4-T --config configs/s4_T.json         # 3-0b 판정 뒤
  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/4-X1 --config configs/s4_X1.json       # T 와 나란히 가능
  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/4-NN --config configs/s4_NN.json       # GPU
  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/4-C --config configs/s4_C.json
  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/4-judge --config configs/s4_judge.json
  보고: pre·T·X1·NN·C 명령 + --report --reason "단계 4 보고"

구현 세부(계획서에 없음 — 담당자 확인): lst/stage4 머리 (a)~(j)·(R1)~(R4), 그리고
 (s1) ① 설정은 4-a·4-a0 어느 쪽이 1위여도 같음(둘은 h 붙이기만 다름 — ① 은 h 1) → T 판정을 기다리지 않고 나란히 돌림.
 (s2) ① 의 반복 수 곡선·threshold·CV = h 1 쌍만(평가 쌍 거르기 R2 — 구간 가중 = h 1 PR-AUC 하나로 다시 나눔).
 (s3) 4-c 비교 줄 = 현재 최고 판 예측을 그 fold(refit) 상위 5 품목 쌍으로 거른 것(같은 시료), 4-c-L 의 '앙상블이면 구성
      트리 중 3-0 CV 1위'(v78) = 사슬 상태 tree_rank_3_0b(칸 이름 그대로, 값 = 3-0 트리 CV 순위).
 (s4) 미리보기는 실험 로그의 그 알고리즘 최대 반복 기록을 씀 — 3-0 보다 먼저 돌면 미리보기가 A 의 첫 2차 사슬 부스팅 판이
      되어 최대 반복을 정함(§3 문구 '첫 2차 사슬 부스팅 판' 그대로) — 지휘자 확인 사항.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import history as hi  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage4 as s4  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, check_resources, limit_gpu_memory  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = s4.STAGE4
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
TRAIN_SUBS = ("pre", "T", "X1", "NN", "C")
SUB_DIR = {"prep": "sel-reduce", "pre": "4-pre", "T": "4-T", "X1": "4-X1", "NN": "4-NN", "C": "4-C",
           "judge": "4-judge"}
DIRS_SCHEMA = {k: cf.Field(str) for k in ("sel-reduce", "sel-top5", "4-pre", "4-T", "4-X1", "4-NN", "4-C",
                                           "4-judge")}
TRAIN_SCHEMA = {
    "sub": cf.Field(str, choices=TRAIN_SUBS),
    "decision_1_3": cf.Field(str),
    "selection_2": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "chain_state_in": cf.Field(str, required=False),     # 3-0b 판정 뒤 사슬 상태(pre 는 없음)
    "theil_targets": cf.Field(str),
    "dirs": DIRS_SCHEMA,
    "gpu_memory_fraction": cf.Field(float, required=False),
    "device": cf.Field(str, required=False),
}
PREP_SCHEMA = {"plan": cf.Field(str), "sub": cf.Field(str, choices=("prep",)), "stage": cf.Field(str),
               "samples": cf.Field(str), "theil_targets": cf.Field(str), "dirs": DIRS_SCHEMA}
JUDGE_SCHEMA = {"plan": cf.Field(str), "sub": cf.Field(str, choices=("judge",)), "stage": cf.Field(str),
                "order": cf.Field(str), "experiment": cf.Field(str), "samples": cf.Field(str),
                "experiment_log": cf.Field(str), "use_mlflow": cf.Field(bool), "mlflow_uri": cf.Field(str),
                "code_reviewed": cf.Field(bool), "decision_1_3": cf.Field(str), "selection_2": cf.Field(str),
                "chain_state_in": cf.Field(str), "chain_state_out": cf.Field(str), "chain_log": cf.Field(str),
                "theil_targets": cf.Field(str), "dirs": DIRS_SCHEMA}


def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run"):
        ap.add_argument(a, action="store_true")
    return ap.parse_known_args(list(argv))[0]


# ================================================================ 판 설정
def _red_plan(cfg, rep: str) -> s4.RowPlan:
    return s4.RowPlan(("reduce",), rep, cfg["dirs"]["sel-reduce"])


def target_member(state: Mapping, inputs_root: Path, which: str = "tree_target") -> dict:
    """트리 대상 판의 트리 하나(앙상블이면 구성 트리 중 3-0 CV 1위 — tree_rank_3_0b 칸, v78)."""
    b = state[which]
    if b["kind"] not in ("tree", "ensemble"):
        raise s4.StageError(f"트리 대상 판 종류가 트리·앙상블이 아님: {b['kind']}")
    if b["kind"] == "tree":
        return b["members"][0]
    rank = state["tree_rank_3_0b"]
    return min(b["members"], key=lambda m: rank[s4.member_spec(inputs_root, m).algorithm])


def build(cfg: Mapping, inputs_root: Path) -> dict:
    """sub → {'run': 판 설정(runner), 'ens': 앙상블 기록 판, 'bests': {판: BestDef(dir 채움)}}."""
    sub = cfg["sub"]
    k = s4.load_k(inputs_root, cfg["selection_2"])
    out_rel = cfg["dirs"][SUB_DIR[sub]]
    if sub == "pre":
        dec = cf.load_json_strict(resolve(inputs_root, cfg["decision_1_3"]))
        A, W, mode = dec["A"], int(dec["W"]), dec["tree_mode"]
        src = rn.VariantSpec(**rn.read_json(resolve(inputs_root, cfg["dir_1_1"]) / rn.name_stage1(A, W, mode) /
                                            "spec.json"))
        sp = s4._member_variant(src, name=f"4-a-pre-{A}", stage=STAGE, role=s4.ROLE_CMP,
                                s4={"row_plan": _red_plan(cfg, "4-a").to_dict(), "catenc_k": k, "input_order": 1},
                                base_name=src.name, params={rn.ITER_SELECT: True})
        sp = rn.VariantSpec(**{**sp.to_dict(), "judgement": s4.NA_JUDGE})
        return {"run": [sp], "ens": [], "bests": {}}
    state = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
    best = state["current_best"]
    part = s4.name_part(best, inputs_root)
    run, ens, bests = [], [], {}

    def add(d: s4.Derived):
        run.extend(d.run)
        if d.ensemble is not None:
            ens.append(d.ensemble)
        b = s4.fill_dirs(d.best, out_rel)
        bests[b["name"]] = b

    if sub == "T":
        if best["kind"] in ("dl", "nn60"):
            add(s4.derive_specs(best, inputs_root=inputs_root, name=f"4-a0-{part}", stage=STAGE, role=s4.ROLE_CAND,
                                plan=_red_plan(cfg, "own"), catenc_k=k))
        else:
            for nm, rep in ((f"4-a-{part}", "4-a"), (f"4-a0-{part}", "own")):
                add(s4.derive_specs(best, inputs_root=inputs_root, name=nm, stage=STAGE, role=s4.ROLE_CAND,
                                    plan=_red_plan(cfg, rep), catenc_k=k))
    elif sub == "X1":
        add(s4.derive_specs(best, inputs_root=inputs_root, name=f"x1-{part}", stage=STAGE, role=s4.ROLE_CMP,
                            plan=_red_plan(cfg, "x1"), catenc_k=k, drop_time=True))
    elif sub == "NN":
        m = target_member(state, inputs_root)
        src = s4.member_spec(inputs_root, m)
        sp = s4._member_variant(src, name="x2-NN60", stage=STAGE, role=s4.ROLE_REPL,
                                s4={"row_plan": _red_plan(cfg, "own").to_dict(), "nn60_from": m["name"]},
                                base_name=state["tree_target"]["name"], params={}, algorithm="NN60")
        run.append(sp)
        bests[sp.name] = s4.fill_dirs({"name": sp.name, "kind": "nn60", "dir": None, "replacement": True,
                                       "members": [{"name": sp.name, "dir": None,
                                                    "row_plan": _red_plan(cfg, "own").to_dict(), "catenc_k": None}]},
                                      out_rel)
    elif sub == "C":
        top = s4.RowPlan(("top5",), "own", cfg["dirs"]["sel-top5"])
        if best["kind"] != "stack4":                             # ④ 는 4-c 안 함(§6 단계 4)
            add(s4.derive_specs(best, inputs_root=inputs_root, name=f"4-c-{part}", stage=STAGE, role=s4.ROLE_CMP,
                                plan=top, catenc_k=None))
        m = target_member(state, inputs_root)
        src = s4.member_spec(inputs_root, m)
        ver = {"include_ids": []}
        sp = s4._member_variant(src, name=f"4-c-L-{src.algorithm}", stage=STAGE, role=s4.ROLE_CMP,
                                s4={"row_plan": top.to_dict()}, base_name=state["tree_target"]["name"],
                                params=s4.chain_params(rn.VariantSpec(**{**src.to_dict(), "params": {}})),
                                version=ver)
        run.append(sp)
    for i, sp in enumerate(run):
        if sp.role == s4.ROLE_CMP and sp.judgement is None:
            run[i] = rn.VariantSpec(**{**sp.to_dict(), "judgement": s4.NA_JUDGE})
    return {"run": run, "ens": ens, "bests": bests, "state": state}


def check_cfg(cfg) -> None:
    if cfg["stage"] != STAGE or cfg["order"] != s4.ORDER or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("stage '4' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
    f = cfg.get("gpu_memory_fraction")
    if f is not None and not (0 < f <= GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    if cfg["sub"] != "pre" and "chain_state_in" not in cfg:
        raise cf.ConfigError("chain_state_in 필요(pre 제외)")


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


# ================================================================ 보고 비교 줄
def _filter_best_row(row: rn.BaseRow, samples: ev.SampleTable, flt) -> rn.BaseRow:
    """현재 최고 판 예측을 평가 쌍 거르기(4-c 상위 5 품목·① h 1)와 같은 쌍으로."""
    def f(df, split):
        if df is None:
            return None
        pos = samples.positions(df["lims_data_key"])
        keep = np.zeros(len(df), bool)
        sp = rn.fd.SPLITS[split]
        h = df["h"].to_numpy()
        for hh in np.unique(h):
            m = np.flatnonzero(h == hh)
            ok = set(flt(int(hh), pos[m], sp).tolist())
            keep[m] = np.fromiter((p in ok for p in pos[m]), bool, m.size)
        return df.loc[keep].reset_index(drop=True)
    cv = {s: f(v, s) for s, v in row.cv_preds.items()}
    thr = None if row.thr_preds is None else {s: f(v, s) for s, v in row.thr_preds.items()}
    return rn.BaseRow(row.name + " (같은 쌍)", row.desc, cv, f(row.test_preds, "refit"), row.selection_count,
                      None, thr_preds=thr)


def report_rows_for(inputs_root: Path, flt_holder: dict):
    def report_rows(cfg, _inputs_root, samples, exp_log):
        from lst import stage3 as s3

        dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
        base = s3.baseline_1_row(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"])
        if flt_holder.get("flt") is not None:      # ①·4-c: 비교 줄도 같은 평가 쌍으로(7-a 줄은 expand_sample_preds 가 이미)
            base = _filter_best_row(base, samples, flt_holder["flt"])
        rows = [base]
        if cfg["sub"] == "pre":
            return rows
        jd = resolve(inputs_root, cfg["dirs"]["4-judge"]) / "4_decision.json"
        if not jd.exists():
            raise SystemExit("단계 4 판정(judge)을 먼저 — test 는 판정 뒤에만(§1-3)")
        st = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
        r = s4.best_row(inputs_root, st["current_best"], "단계 4 시작 때 현재 최고 판(판정 상대)")
        if flt_holder.get("flt") is not None:
            r = _filter_best_row(r, samples, flt_holder["flt"])
        rows.append(r)
        out = resolve(inputs_root, cfg["dirs"][SUB_DIR[cfg["sub"]]])
        for e in build(cfg, inputs_root)["ens"]:
            d = out / e.name
            done = rn.read_json(d / "done.json")
            rows.append(rn.BaseRow(e.name, f"앙상블(구성 {', '.join(e.extra['s4']['members'])} 평균)",
                                   rn.load_preds(d, ev.CV_FOLDS), pd.read_parquet(rn.pred_path(d, "refit")),
                                   done["selection_count"], done.get("mlflow_run_id"), thr_preds=rn.load_thr_preds(d)))
        return rows
    return report_rows


NOTES = ("단계 4 판의 트리 = 단계 4~6 범주 숫자 인코딩(부록 A-6) — 현재 최고 판이 단계 4~6 판이 아니고 트리 부분이 있으면 "
         "현재 최고 판과의 차이에 범주 인코딩 변경이 섞임(부록 A-12)",
         "부스팅 판: CV = 교차 N 예측, threshold = 공통 N 예측, test = refit 공통 N(§3·§1-2)",
         "줄이기·h 붙이기는 train 에만 — validation·test 쌍은 전부(§6 단계 4)")


# ================================================================ prep·judge
def prep(argv) -> dict:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, PREP_SCHEMA))
    if cfg["stage"] != STAGE:
        raise cf.ConfigError("stage '4'")
    inputs_root = Path(args.inputs).expanduser().resolve()
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    red = s4.prepare_selection(samples, resolve(inputs_root, cfg["dirs"]["sel-reduce"]), ["reduce"],
                               data_root=rp.root, theil_targets=resolve(inputs_root, cfg["theil_targets"]))
    top = s4.prepare_selection(samples, resolve(inputs_root, cfg["dirs"]["sel-top5"]), ["top5"])
    for s in rn.SPLIT_ORDER:
        st = red["per_split"][s]["steps"][0]
        print(f"{s}: 줄이기 칼럼 {st['column']} — 행 {red['per_split'][s]['n_start']} → {st['n_after']}, "
              f"검출 {st['pos_after']}; 상위 5 = {top['top5'][s]}")
    return {"reduce": red, "top5": top}


def judge(argv, *, tracker_fn=sc.make_tracker) -> dict:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))
    if cfg["stage"] != STAGE or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("judge 설정: stage '4' · experiment listeria-2차")
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = rp.out
    if out.resolve() != resolve(inputs_root, cfg["dirs"]["4-judge"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[4-judge]")
    p = out / "4_decision.json"
    if p.exists():
        print(f"단계 4 판정 이미 있음: {p}")
        return rn.read_json(p)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    state = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
    stack, repl = [], []
    for sub, lst in (("T", stack), ("NN", repl)):
        c = {**cfg, "sub": sub, "dirs": cfg["dirs"]}
        b = build(c, inputs_root)
        for name, bd in b["bests"].items():
            lst.append((bd, resolve(inputs_root, bd["dir"])))
    res = s4.judge_step(step=STAGE, state=state, stack=stack, repl=repl, samples=samples, inputs_root=inputs_root)
    rec, new = res["record"], res["state"]
    adopted_names = {b["name"] for b, _ in stack + repl}
    new["stage4"] = {"start_best": state["current_best"], "start_tree_target": state["tree_target"],
                     "adopted": new["current_best"]["name"] in adopted_names}
    # ① 비교용 기록: 단계 4 1위(4-a·4-a0 중 CV 1위)
    with ev.selection_guard():
        cands = [ev.Candidate(b["name"], *s4._cv_of(d)) for b, d in stack]
        new["stage4"]["stage4_first"] = ev.rank_candidates(cands)[0].name if cands else None
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
    cf.dump_scores_json(p, {"record": rec, "state_out": cfg["chain_state_out"], "stage4": new["stage4"]})
    if tr is not None:
        tr.log_artifact(tr.stage_run(s4.EXPERIMENT, STAGE), p, "decision")
    print(f"단계 4: {rec['before']} → {rec['after']}(트리 대상 판 {new['tree_target']['name']})")
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
    if sub not in TRAIN_SUBS:
        raise cf.ConfigError(f"sub: {sub!r}")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    if not (ns.list or ns.verify_run):
        want = resolve(inputs_root, raw["dirs"][SUB_DIR[sub]]).resolve()
        if Path(ns.out).expanduser().resolve() != want:
            raise cf.ConfigError(f"--out 은 --inputs 아래 dirs[{SUB_DIR[sub]}] = {want}")
    specs_fn = make_specs_fn(inputs_root)
    flt_holder: dict = {}
    if ns.variant or ns.report:
        cfg = sc.load_cfg(ns.config, TRAIN_SCHEMA)
        specs = specs_fn(cfg)
        samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
        if ns.variant:
            by = {s.name: s for s in specs}
            group = [by[n.strip()] for n in ns.variant.split(",") if n.strip()]
            s4.install_for_group(group, inputs_root, samples)
        else:                                       # 보고: 같은 쌍으로 채점(①·4-c)
            plans = {json_key(s4.RowPlan.from_dict(s4.s4_of(s).get("row_plan"))) for s in specs}
            if len(plans) > 1 and sub in ("X1", "C"):
                raise s4.StageError("보고 설정 안 판의 평가 쌍 거르기가 다름")
            flt = s4.pair_filter_for(s4.RowPlan.from_dict(s4.s4_of(specs[0]).get("row_plan")), inputs_root, samples)
            if flt is not None:
                s4.install_pair_filter(flt)
                flt_holder["flt"] = flt
    res = sc.stage_main(argv, script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=TRAIN_SCHEMA,
                        specs_fn=specs_fn, factories_fn=factories_fn or factories, before_variant=before_variant_fn,
                        fold_recall=True, tracker_fn=tracker_fn, check_resources=check_resources_fn,
                        report_rows_fn=report_rows_for(inputs_root, flt_holder), report_notes=NOTES)
    default = not (ns.list or ns.variant or ns.verify_run or ns.report or ns.only)
    if default and specs_fn.holder["b"]["ens"]:
        cfg = sc.load_cfg(ns.config, TRAIN_SCHEMA)
        samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
        log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
        tr = tracker_fn(cfg)
        sr = tr.stage_run(cfg["experiment"], STAGE) if tr is not None else None
        code = rn.snapshot_code(CODE_ROOT, Path(ns.out).expanduser().resolve())
        for e in specs_fn.holder["b"]["ens"]:
            d = s4.write_ensemble(Path(ns.out).expanduser().resolve(), e, samples, log, code, tracker=tr,
                                  stage_run_id=sr)
            print(f"{e.name}: 앙상블 CV {d['cv_band_weighted']}")
    return res


def json_key(plan: s4.RowPlan) -> str:
    import json

    return json.dumps(plan.to_dict(), sort_keys=True)


if __name__ == "__main__":
    main()
