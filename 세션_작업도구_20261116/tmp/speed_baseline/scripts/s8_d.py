# -*- coding: utf-8 -*-
"""s8_d.py — 단계 8-D 딥러닝 튜닝(단계 8 판정 뒤 현재 최고 판의 딥러닝 부분) — 담당자 결정 11-11(계획서 단계 8-D 절 초안),
§6 단계 8·2차 공통(채택 판정), 부록 A-8(딥러닝 학습)·A-12·A-15·B-1. 공통 = lst/tuning_dl·lst/stage4·lst/tuning.

대상 = 단계 8 판정 뒤 사슬 상태(chain_state_in = logs/chain_state_8.json)의 현재 최고 판(단계 안에서 안 바꿈). 딥러닝 부분:
  dl        딥러닝 단독·결합 모델(⑥·⑦-D 포함, 5·6 판 포함) — 그 모델 설정을 탐색
  그 밖     트리·앙상블(트리 구성)·④·② = 딥러닝 부분 없음 → 판 없이 '튜닝 안 함'(상태 그대로 넘김). 앙상블 구성에 딥러닝이
            있으면 멈춤(지금 사슬엔 없음 — 지원 안 함).
자료·입력·h 규칙·에폭·조기 종료 규칙·평가 쌍·채점·threshold·학습 씨앗은 대상 판 그대로, 설정만 바꿈(씨앗 하나, 음성 4배 고정).

설정 sub(한 설정 = 결과 폴더 하나 — --out 은 --inputs 아래 dirs[<sub 폴더>]):
  tune   t000(대상 판 설정 = 대상 판 CV, 학습 없음) + TPE(씨앗 42) 19회(tuning_dl.run_batched — 묶어 묻기 tpe_batches,
         시도·fold 기록 jsonl 이어하기) → tuning.select_tuned(top 5) = tune_<구조>.json·top5_<구조>.csv·trials_<구조>.csv.
         첫 묶음에서 t000 도 다시 학습해 대상 판 CV 와 대조(기록만).
  D      8-D-<바탕> 판(3 fold + refit): 고른 설정(params {"dl": …}) — 기본 유지·튜닝 안 함이면 학습 없음.
  judge  학습 없음: 8-D(쌓는 후보) 대 현재 최고 판 → 채택 = CV 평균 4자리 상승(+ fold_majority 면 3 fold 중 2개 이상 상승)
         → chain_state_out(logs/chain_state_8D.json)·8D_decision.json·사슬 기록, MLflow 8-D 판 run(기본 유지면 판 run 만)
         + 시도 run(자식, <판 이름>-t000 ~ t019, 점수만).
보고(judge 뒤): D 설정 + --report --reason "단계 8-D 보고"(1차 베이스라인·대상 판 줄 + 8-D 판 — test 는 여기서만).

실행(서버 — 검수 뒤 지휘자, 단계 8 판정 뒤):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s8_d.py $A --out ~/cyj_listeria_final/results/s8/8-D-tune --config configs/s8_D_tune.json
  $P scripts/s8_d.py $A --out ~/cyj_listeria_final/results/s8/8-D --config configs/s8_D.json
  $P scripts/s8_d.py $A --out ~/cyj_listeria_final/results/s8/8-D-judge --config configs/s8_D_judge.json
  보고: D 명령 + --report --reason "단계 8-D 보고"

구현 세부(계획서에 없음 — 담당자 확인): lst/tuning_dl 머리 (d1)~(d5), lst/models_dl 구현 세부 16·17, 그리고
 (e1) 판 이름 = 8-D-<바탕 판 구조 약어>(바탕이 교체 후보면 바탕 판 이름 — 부록 A-14 단계 8 규칙), MLflow 단계 = '8'.
 (e2) 8-D 판 3 fold 도 runner 로 다시 학습(시도와 같은 설정·씨앗 — 같은 값 기대) → judge 에서 고른 시도 점수와 대조해 기록.
 (e3) 시도 수는 실험 로그 '같은 validation 으로 고른 횟수'에 19(kind '시도', 이름 8-D-tune-<구조>)로 셈(부록 A-12 'Optuna
      시도 포함') — t000 은 대상 판 자신이라 안 셈, t000 대조 학습도 안 셈.
 (e4) 채택 = 2차 공통(CV 4자리 상승) — 설정 fold_majority 기본 false(계획서 2차 공통). true 로 두면 3 fold 중 2개 이상 상승도(stage3.adopt_3_0b
      와 같은 계산). 동점 많은 판 다시 채점 기록·알림 = evaluate.judge_adoption.
 (e5) GPU 상한 = 설정 gpu_memory_fraction(0.25) — models_dl.set_gpu_fraction 으로 모델 만들 때도 그 값.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as _dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import metrics as mt  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import scope as lst_scope  # noqa: E402
from lst import stage4 as s4  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst import tuning as tu  # noqa: E402
from lst import tuning_dl as tdl  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, check_resources, limit_gpu_memory  # noqa: E402

import s8  # noqa: E402  (같은 scripts 폴더 — wiring_for·factories 를 같이 씀)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = s4.STAGE8
STEP = "8-D"
TOP = 5
SUB_DIR = {"tune": "8-D-tune", "D": "8-D", "judge": "8-D-judge"}
DECISION = "8D_decision.json"
DIRS_SCHEMA = {k: cf.Field(str) for k in SUB_DIR.values()}
COMMON_EXTRA = {
    "sub": cf.Field(str, choices=("tune", "D")),
    "decision_1_3": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "chain_state_in": cf.Field(str),
    "theil_targets": cf.Field(str),
    "dirs": DIRS_SCHEMA,
    "tpe_batches": cf.Field(list, required=False, item_types=int),
    "gpu_memory_fraction": cf.Field(float, required=False),
    "device": cf.Field(str, required=False),
}
JUDGE_SCHEMA = {"plan": cf.Field(str), "sub": cf.Field(str, choices=("judge",)), "stage": cf.Field(str),
                "order": cf.Field(str), "experiment": cf.Field(str), "samples": cf.Field(str),
                "experiment_log": cf.Field(str), "use_mlflow": cf.Field(bool), "mlflow_uri": cf.Field(str),
                "code_reviewed": cf.Field(bool), "chain_state_in": cf.Field(str), "chain_state_out": cf.Field(str),
                "chain_log": cf.Field(str), "fold_majority": cf.Field(bool), "dirs": DIRS_SCHEMA}
JUDGE_SCHEMA = lst_scope.with_field(JUDGE_SCHEMA)   # 국내만 버전 scope 칸(없으면 지금 판)
NOTES = ("단계 8-D 시도·판 점수 = 대상 판과 같은 방식(같은 train 행·입력·씨앗·조기 종료·평가 쌍) — 바뀐 것은 딥러닝 설정만",
         "5·6 판이 대상이면 학습 씨앗 0(MLflow param seed 는 42 로 남음 — 판 설정 extra.s4.model_seed = 0)")


def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run"):
        ap.add_argument(a, action="store_true")
    ns = ap.parse_known_args(list(argv))[0]
    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)
    return ns


# ================================================================ 딥러닝 부분
@dataclass
class DLPart:
    label: str                 # 구조 약어(파일 이름)
    member: dict               # MemberDef
    src: rn.VariantSpec        # 구성 판 설정
    plan: s4.RowPlan
    seed: int                  # 학습 씨앗(5·6 판 = 0)
    base_params: dict          # 대상 판 params({} = 부록 A-8 기본)

    @property
    def arch(self) -> str:
        return self.src.algorithm


def dl_parts(state: Mapping, inputs_root: Path) -> list[DLPart]:
    """대상(현재 최고 판)의 딥러닝 부분 — 딥러닝 단독·결합 = 1개, 그 밖 = 빈 목록."""
    best = state["current_best"]
    kind = best["kind"]
    if kind == "ensemble":
        for m in best["members"]:
            if s4.member_spec(inputs_root, m).kind == "dl":
                raise s4.StageError(f"{best['name']}: 앙상블 구성에 딥러닝 — 단계 8-D 지원 안 함(지휘자 확인)")
        return []
    if kind != "dl":
        return []
    m = best["members"][0]
    src = s4.member_spec(inputs_root, m)
    if src.kind != "dl" or src.algorithm not in tdl.md.CLASSIFIER_ARCHS:
        raise s4.StageError(f"딥러닝 판 구성이 딥러닝 분류기가 아님: {src.name}({src.algorithm})")
    if m.get("anomaly"):
        raise s4.StageError(f"{src.name}: 딥러닝 판에 이상 점수 칼럼 — 지원 안 함")
    e = dict(src.extra.get("s4") or {})
    plan = s4.RowPlan.from_dict(m.get("row_plan") if m.get("row_plan") is not None else e.get("row_plan"))
    return [DLPart(label=src.algorithm, member=dict(m), src=src, plan=plan, seed=int(e.get("model_seed", 42)),
                   base_params=dict(src.params))]


def variant_name(state: Mapping, inputs_root: Path) -> str:
    return f"{STEP}-{s4.name_part(state['current_best'], inputs_root)}"


def base_spec(state: Mapping, part: DLPart, inputs_root: Path, name: str, out_rel: str | None = None):
    """대상 판을 그대로 다시 만드는 판 설정(stage4.derive_specs — RowPlan·씨앗) + 새 BestDef 초안."""
    d = s4.derive_specs(state["current_best"], inputs_root=inputs_root, name=name, stage=STAGE, role=s4.ROLE_CAND,
                        plan=part.plan, catenc_k=None, model_seed=(0 if part.seed == 0 else None))
    if len(d.run) != 1 or d.ensemble is not None:
        raise s4.StageError("딥러닝 판 설정이 하나가 아님")
    sp = dataclasses.replace(d.run[0], params=dict(part.base_params))
    best = s4.fill_dirs(d.best, out_rel) if out_rel else d.best
    return sp, best


def target_cv(inputs_root: Path, state: Mapping) -> float:
    return s4._cv_of(resolve(inputs_root, state["current_best"]["dir"]))[0]


# ================================================================ 설정·자원
def check_cfg(cfg) -> None:
    if cfg["stage"] != STAGE or cfg["order"] != s4.ORDER or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("stage '8' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
    f = cfg.get("gpu_memory_fraction")
    if f is not None and not (0 < f <= GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    if cfg["n_jobs"] != 8:
        raise cf.ConfigError("n_jobs 8")
    if cfg.get("sub") == "tune":
        if "tpe_batches" not in cfg:
            raise cf.ConfigError("tune 설정에는 tpe_batches")
        tdl.check_batches(cfg["tpe_batches"], tdl.N_TPE)


def load_cfg(path) -> dict:
    cfg = sc.load_cfg(path, COMMON_EXTRA)
    check_cfg(cfg)
    return cfg


def before_variant(cfg) -> None:
    f = cfg.get("gpu_memory_fraction")
    if f:
        from lst import models_dl as md

        limit_gpu_memory(f)
        md.set_gpu_fraction(f)


# ================================================================ tune
def tune_part(part: DLPart, out: Path, *, score_fn, base_cv, batches, t000_check: bool = True) -> dict:
    """t000 + TPE 19 → 고르기(select_tuned, top 5). 끝났으면(tune_<구조>.json) 그대로.
    score_fn([(번호, params)…], 묶음) = DLTrialScorer(번호 0 = t000 대조 학습)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res_p = out / f"tune_{part.label}.json"
    if res_p.exists():
        return rn.read_json(res_p)
    t0_p = out / f"t000_{part.label}.json"
    if t0_p.exists():
        t0 = rn.read_json(t0_p)
    else:
        t0 = {"cv": float(base_cv()), "source": "대상 판 CV(학습 없음)", "params": part.base_params}
        cf.dump_scores_json(t0_p, t0)
    check: dict = {}

    def score_batch(items, bi):
        extra = [(0, part.base_params)] if (t000_check and bi == 1) else []
        r = dict(score_fn(extra + list(items), bi))
        if extra:
            check.update(r.pop(0))
        return r

    recs = tdl.run_batched(part.arch, score_batch, log_path=out / f"dl_trials_{part.label}.jsonl", n_trials=tdl.N_TPE,
                           seed=tdl.TPE_SEED, batches=batches, label=part.label)
    if len(recs) != tdl.N_TPE:
        raise s4.StageError(f"{part.label}: TPE 시도 {len(recs)} ≠ {tdl.N_TPE}")
    sel = tu.select_tuned(float(t0["cv"]), recs, top=TOP)
    if t000_check and not check:      # 첫 묶음이 이전 실행에서 끝났으면 fold 기록에서
        fo = [r for r in tdl._read(out / f"dl_folds_{part.label}.jsonl") if r.get("event") == "fold" and r["number"] == 0]
        if len(fo) == len(ev.CV_FOLDS):
            fs = {r["fold"]: tdl._num(r["score"]) for r in fo}
            check.update(cv=mt.cv_mean(fs), fold_scores=fs)
    t000_chk = None
    if check:
        diff = abs(float(check["cv"]) - float(t0["cv"]))
        t000_chk = {"retrained_cv": float(check["cv"]), "fold_scores": check.get("fold_scores"),
                    "target_cv": float(t0["cv"]), "abs_diff": diff, "alert": bool(diff > tdl.T000_ALERT_TOL)}
    pd.DataFrame(sel["top"]).assign(params=lambda d: d["params"].map(lambda v: json.dumps(v, ensure_ascii=False))) \
        .to_csv(out / f"top5_{part.label}.csv", index=False, encoding="utf-8-sig")
    tdl.trial_table(recs).to_csv(out / f"trials_{part.label}.csv", index=False, encoding="utf-8-sig")
    res = {"label": part.label, "arch": part.arch, "member": part.member["name"], "t000": t0, "t000_check": t000_chk,
           "selection": sel, "adopt": bool(sel["adopt"]), "params": (sel["selected"] or {}).get("params"),
           "seed": part.seed, "batches": list(batches), "space": tdl.search_space(part.arch),
           "n_trials_total": 1 + len(recs), "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    cf.dump_scores_json(res_p, res)
    return res


def tune_main(argv, *, wiring_fn=None) -> dict:
    ap = sc.parser(__doc__.splitlines()[0])
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    if cfg["sub"] != "tune":
        raise cf.ConfigError("tune 설정이 아님")
    inputs_root = Path(args.inputs).expanduser().resolve()
    if rp.out.resolve() != resolve(inputs_root, cfg["dirs"]["8-D-tune"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[8-D-tune]")
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    state = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
    parts = dl_parts(state, inputs_root)
    rp.out.mkdir(parents=True, exist_ok=True)
    if not parts:
        cf.dump_json(rp.out / "no_tuning.json", {"target": state["current_best"]["name"],
                                                 "kind": state["current_best"]["kind"],
                                                 "reason": "딥러닝 부분이 없는 판 — 튜닝 안 함(단계 8-D)"}, overwrite=True)
        print("단계 8-D: 대상에 딥러닝 부분이 없음 — 튜닝 안 함")
        return {}
    part = parts[0]
    if (rp.out / f"tune_{part.label}.json").exists():
        res = rn.read_json(rp.out / f"tune_{part.label}.json")
        log.cv_scored(STAGE, f"{STEP}-tune-{part.label}", n_settings=tdl.N_TPE, kind="시도")   # 이미 있으면 그대로
        print(f"단계 8-D 튜닝 이미 있음: {part.label} {res['selection']['judgement']}")
        return res
    before_variant(cfg)
    rn.snapshot_code(CODE_ROOT, rp.out)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    w = (wiring_fn or s8.wiring_for)(cfg, rp.root, inputs_root, rp.out)
    spec, _ = base_spec(state, part, inputs_root, variant_name(state, inputs_root))
    plan = s4.RowPlan.from_dict(spec.extra["s4"].get("row_plan"))
    flt = s4.pair_filter_for(plan, inputs_root, samples)
    if flt is not None:
        s4.install_pair_filter(flt)
    smoke = {k: cfg["smoke"][k] for k in rn.SMOKE_KEYS} if "smoke" in cfg else None
    known = samples.known_frame()

    def train_rows(split):      # runner._train_rows 를 stage4.install_train_rows 로 바꿔 끼운 것과 같은 계산
        rows = s4.plan_train_rows(known, split, plan, inputs_root)
        if smoke is not None:
            rows = rn._smoke_train(rows, smoke["max_train_rows"], cfg["seed"])
        return rn.add_bounds(rows, spec.W)

    scorer = tdl.DLTrialScorer(spec=spec, samples=samples, make_inputs=w.make_inputs, make_model=w.make_model,
                               fit_kwargs=w.fit_kwargs, train_rows_fn=train_rows,
                               fold_log=rp.out / f"dl_folds_{part.label}.jsonl", n_jobs=cfg["n_jobs"], seed=cfg["seed"],
                               check=lambda: check_resources(rp.out),
                               max_pairs_per_h=smoke["max_pairs_per_h"] if smoke else None)
    res = tune_part(part, rp.out, score_fn=scorer, base_cv=lambda: target_cv(inputs_root, state),
                    batches=cfg["tpe_batches"])
    log.cv_scored(STAGE, f"{STEP}-tune-{part.label}", n_settings=tdl.N_TPE, kind="시도")
    log.append("tuning", stage=STEP, part=part.label, adopt=res["adopt"], best_cv=res["selection"]["best_cv"],
               baseline_cv=res["selection"]["baseline_cv"], n_trials=res["n_trials_total"])
    if res.get("t000_check") and res["t000_check"]["alert"]:
        msg = (f"8-D t000 다시 학습 CV {res['t000_check']['retrained_cv']:.6f} ≠ 대상 판 CV "
               f"{res['t000_check']['target_cv']:.6f}(차이 > {tdl.T000_ALERT_TOL}) — 규칙은 그대로(대상 판 CV 로 고름)")
        log.append("alert", stage=STEP, message=msg)
        print(f"[알림] {msg}")
    print(f"{part.label}: {res['selection']['judgement']} (t000 {res['selection']['baseline_cv']:.4f}, "
          f"1위 {res['selection']['best_trial']} {res['selection']['best_cv']:.4f})")
    return res


# ================================================================ D 판
def load_tune(inputs_root: Path, cfg: Mapping, part: DLPart) -> dict:
    f = resolve(inputs_root, cfg["dirs"]["8-D-tune"]) / f"tune_{part.label}.json"
    if not f.exists():
        raise s4.StageError(f"튜닝 결과 없음(tune 먼저): {f}")
    return rn.read_json(f)


def build_D(cfg: Mapping, inputs_root: Path) -> dict:
    """8-D 판 설정. status = '튜닝 안 함'(딥러닝 부분 없음)·'기본 유지'(t000 이 1위 — 학습 없음)·'판'."""
    state = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
    parts = dl_parts(state, inputs_root)
    name = variant_name(state, inputs_root)
    base = {"run": [], "best": None, "name": name, "parts": parts, "state": state, "tune": None}
    if not parts:
        return {**base, "status": "튜닝 안 함"}
    t = load_tune(inputs_root, cfg, parts[0])
    if not t["adopt"]:
        return {**base, "status": "기본 유지", "tune": t}
    sp, best = base_spec(state, parts[0], inputs_root, name, cfg["dirs"]["8-D"])
    sp = dataclasses.replace(sp, params=dict(t["params"]))
    tdl.dl_params(parts[0].arch, {**t["params"]["dl"]})      # 점검(모르는 칸·범위)
    return {**base, "status": "판", "run": [sp], "best": best, "tune": t}


def make_specs_fn(inputs_root: Path):
    holder: dict = {}

    def specs_fn(cfg):
        check_cfg(cfg)
        if "b" not in holder:
            holder["b"] = build_D(cfg, inputs_root)
        return holder["b"]["run"]
    specs_fn.holder = holder
    return specs_fn


def report_rows_for(inputs_root: Path):
    def report_rows(cfg, _ir, samples, exp_log):
        from lst import stage3 as s3

        jd = resolve(inputs_root, cfg["dirs"]["8-D-judge"]) / DECISION
        if not jd.exists():
            raise SystemExit("단계 8-D 판정(judge)을 먼저 — test 는 판정 뒤에만(§1-3)")
        dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
        rows = [s3.baseline_1_row(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"])]
        st = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
        rows.append(s4.best_row(inputs_root, st["current_best"], "단계 8-D 대상 판(단계 8 판정 뒤 현재 최고 판)"))
        return rows
    return report_rows


# ================================================================ judge
def _pred_dir(inputs_root: Path, d: Path) -> Path:
    if rn.pred_path(d, "refit").exists():
        return d
    if (d / "source.json").exists():
        return resolve(inputs_root, rn.read_json(d / "source.json")["dir"])
    src = (rn.read_json(d / "spec.json").get("extra") or {}).get("source")
    if isinstance(src, Mapping) and src.get("dir"):
        return resolve(inputs_root, src["dir"])
    return d


def judge_core(cfg: Mapping, inputs_root: Path, samples: ev.SampleTable) -> dict:
    from lst import stage3 as s3

    b = build_D(cfg, inputs_root)
    state = b["state"]
    cur = dict(state["current_best"])
    utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    t = b["tune"]
    rec = {"step": STEP, "status": b["status"], "target": cur["name"], "name": b["name"], "parts": [], "utc": utc,
           "fold_majority": bool(cfg["fold_majority"]),
           "tune": None if t is None else {"adopt": t["adopt"], "t000": t["t000"]["cv"],
                                           "t000_check": t.get("t000_check"), "best": t["selection"]["best_cv"],
                                           "best_trial": t["selection"]["best_trial"], "top": t["selection"]["top"],
                                           "params": t.get("params")}}
    new = copy.deepcopy(dict(state))
    new["step"] = STEP
    if b["status"] != "판":
        new["history"] = list(state.get("history", [])) + [{"step": STEP, "before": cur["name"], "after": cur["name"],
                                                            "judgement": b["status"], "utc": utc}]
        rec.update(before=cur["name"], after=cur["name"], adopted=False)
        return {"record": rec, "state": new, "built": b}
    cdir = resolve(inputs_root, b["best"]["dir"])
    curdir = resolve(inputs_root, cur["dir"])
    _, _, ccv = rn.load_done(cdir.parent, cdir.name)
    _, _, kcv = rn.load_done(curdir.parent, curdir.name)
    dirs = {b["name"]: cdir, cur["name"]: _pred_dir(inputs_root, curdir)}

    def rescore(n):
        return float(ev.score_cv(rn.load_preds(dirs[n], ev.CV_FOLDS), samples, rescore=True)["cv_band_weighted"])

    with ev.selection_guard():
        cand = ev.Candidate(b["name"], *s4._cv_of(cdir))
        curc = ev.Candidate(cur["name"], *s4._cv_of(curdir))
        j = ev.judge_adoption(cand, curc, rescore)
        rule = (s3.adopt_3_0b({"cv": ccv["cv_band_weighted"], "fold": ccv["fold_band_weighted"]},
                              {"cv": kcv["cv_band_weighted"], "fold": kcv["fold_band_weighted"]})
                if cfg["fold_majority"] else None)
    adopt = bool(j["adopt"] and (rule is None or rule["adopt"]))
    rec["parts"] = [{"kind": "쌓는 후보", "ranking": [b["name"]], "cv": {b["name"]: cand.cv}, "top": b["name"],
                     "against": cur["name"], "judge": j, "fold_rule": rule, "adopted": adopt}]
    rec["trial_vs_variant_cv"] = {"trial": t["selection"]["best_cv"], "variant": cand.cv,
                                  "abs_diff": abs(t["selection"]["best_cv"] - cand.cv)}
    after = b["best"] if adopt else cur
    new["current_best"] = after
    if adopt and after["kind"] in ("tree", "ensemble") and not after.get("replacement"):
        new["tree_target"] = after
    new["history"] = list(state.get("history", [])) + [{"step": STEP, "before": cur["name"], "after": after["name"],
                                                        "judgement": "채택" if adopt else "버림",
                                                        "tree_target_after": new["tree_target"]["name"], "utc": utc}]
    rec.update(before=cur["name"], after=after["name"], adopted=adopt)
    return {"record": rec, "state": new, "built": b}


def _trial_runs(tr, parent_id: str, name: str, tune: Mapping, tune_dir: Path) -> None:
    """시도 run(부록 A-15 꼴): <판 이름>-t000 ~ t019, 점수만."""
    rid = tr.child_run(s4.EXPERIMENT, parent_id, f"{name}-t000", "시도", {"단계": STAGE})
    tr.log_metrics(rid, {"cv_band_weighted": tune["t000"]["cv"]})
    tr.finish(rid)
    for r in tdl._read(tune_dir / f"dl_trials_{tune['label']}.jsonl"):
        if r.get("event") != "trial":
            continue
        rid = tr.child_run(s4.EXPERIMENT, parent_id, f"{name}-{tu.trial_label(r['number'])}", "시도", {"단계": STAGE})
        tr.log_params(rid, r["raw"])
        tr.log_metrics(rid, {"cv_band_weighted": r["value"],
                             **{f"{f}_band_weighted": v for f, v in
                                ((r.get("result") or {}).get("fold_scores") or {}).items() if v is not None}})
        tr.finish(rid)


def judge(argv, *, tracker_fn=sc.make_tracker) -> dict:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    if cfg["stage"] != STAGE or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("judge 설정: stage '8' · experiment listeria-2차")
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = rp.out
    if out.resolve() != resolve(inputs_root, cfg["dirs"]["8-D-judge"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[8-D-judge]")
    p = out / DECISION
    if p.exists():
        print(f"판정 이미 있음: {p}")
        return rn.read_json(p)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    r = judge_core(cfg, inputs_root, samples)
    rec, new, b = r["record"], r["state"], r["built"]
    tr = tracker_fn(cfg)
    if tr is not None:
        sr = tr.stage_run(s4.EXPERIMENT, STAGE)
        if b["status"] == "판":
            s4.tag_judgements(tr, rec, lambda n: tr.find_run(s4.EXPERIMENT, n, sr))
        if b["status"] in ("판", "기본 유지"):
            run = tr.find_run(s4.EXPERIMENT, b["name"], sr)
            if run is None:      # 기본 유지 — 판 run 만 만듦(부록 A-15)
                rid = tr.start_variant(s4.EXPERIMENT, sr, b["name"], {"단계": STAGE, "역할": s4.ROLE_CAND},
                                       {"parts": [x.label for x in b["parts"]]})
                tr.set_judgement(rid, "기본 유지", against=rec["target"])
                tr.finish(rid)
            else:
                rid = run.info.run_id
            _trial_runs(tr, rid, b["name"], b["tune"], resolve(inputs_root, cfg["dirs"]["8-D-tune"]))
    judgement = "채택" if rec.get("adopted") else (b["status"] if b["status"] != "판" else "버림")
    line = {"step": STEP, "judgement": judgement, "before": rec["before"], "after": rec["after"],
            "tree_target": new["tree_target"]["name"], "utc": rec["utc"]}
    if rec.get("adopted"):
        line["after_dir"] = new["current_best"]["dir"]
    cpath = s4.append_chain_log(inputs_root, cfg["chain_log"], line)
    if tr is not None:
        tr.log_artifact(tr.chain_run(), cpath)
    for part in rec.get("parts", []):
        for a in part["judge"]["alerts"]:
            log.append("alert", stage=STEP, message=a)
            print(f"[알림] {a}")
    log.append("decision", stage=STEP, before=rec["before"], after=rec["after"], status=b["status"])
    s4.save_state(resolve(inputs_root, cfg["chain_state_out"]), new)
    out.mkdir(parents=True, exist_ok=True)
    cf.dump_scores_json(p, {"record": rec, "state_out": cfg["chain_state_out"]})
    if tr is not None:
        tr.log_artifact(tr.stage_run(s4.EXPERIMENT, STAGE), p, "decision")
    print(f"{STEP}: {judgement} — {rec['before']} → {rec['after']}")
    return rn.read_json(p)


# ================================================================ 진입점
def main(argv=None, *, factories_fn=None, tracker_fn=sc.make_tracker, check_resources_fn=None,
         before_variant_fn=before_variant, wiring_fn=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ns = _pre(argv)
    if not ns.config or not ns.inputs:
        raise SystemExit("--config·--inputs 필요")
    raw = cf.load_json_strict(ns.config)
    sub = raw.get("sub")
    if sub == "judge":
        return judge(argv, tracker_fn=tracker_fn)
    if sub == "tune":
        return tune_main(argv, wiring_fn=wiring_fn)
    if sub != "D":
        raise cf.ConfigError(f"sub: {sub!r}")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    if not (ns.list or ns.verify_run):
        want = resolve(inputs_root, raw["dirs"]["8-D"]).resolve()
        if Path(ns.out).expanduser().resolve() != want:
            raise cf.ConfigError(f"--out 은 --inputs 아래 dirs[8-D] = {want}")
    specs_fn = make_specs_fn(inputs_root)
    cfg = load_cfg(ns.config)
    specs = specs_fn(cfg)
    built = specs_fn.holder["b"]
    if not specs:
        msg = f"단계 8-D: 판 없음 — {built['status']}"
        print(msg)
        if not (ns.list or ns.report):
            rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])).append("note", stage=STEP, message=msg)
        return []
    if ns.variant:
        by = {s.name: s for s in specs}
        s4.install_for_group([by[n.strip()] for n in ns.variant.split(",") if n.strip()], inputs_root, None)
    return sc.stage_main(argv, script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=COMMON_EXTRA,
                         specs_fn=specs_fn, factories_fn=factories_fn or s8.factories, before_variant=before_variant_fn,
                         fold_recall=True, tracker_fn=tracker_fn, check_resources=check_resources_fn,
                         report_rows_fn=report_rows_for(inputs_root), report_notes=NOTES)


if __name__ == "__main__":
    main()
