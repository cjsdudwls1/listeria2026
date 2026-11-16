# -*- coding: utf-8 -*-
"""s8.py — 단계 8 튜닝·불균형(8-T Optuna 20회 → Focal) — 계획서 요약 18, §6 단계 8·2차 공통 표 단계 8 줄, §3(학습률·
최대 반복), 부록 A-7·A-9·A-12('단계 8 점수')·A-14·A-15·B-1. 공통 = lst/stage4·lst/tuning.

대상 = 단계 시작 때 현재 최고 판(chain_state_in — 5·6 판정 뒤 사슬 상태, 단계 안에서 안 바꿈). 튜닝할 부분(구성):
  tree     단일 트리 1개(이상 점수 칼럼이 있으면 칼럼을 뺀 입력으로 튜닝 → 판에서 다시 붙임)
  ensemble 구성 트리마다 따로(그 트리 단독 CV 로 시도 고르기)
  stack4   ④ 기본 트리 6종마다(반복 수 = 라이브러리 기본·learning_rate 탐색 안 함·그 트리 단독 fold validation CV,
           메타 로지스틱은 튜닝 안 함 — 고른 설정으로 OOF 블록·메타를 다시 만듦)
  dl·nn60  튜닝 안 함(딥러닝 단독·결합·⑥·⑦-D·② 와 그 5·6 판) — 판 없음, 사슬 상태 그대로.

설정 sub(한 설정 = 결과 폴더 하나 — --out 은 --inputs 아래 dirs[<sub 폴더>]):
  tune    구성마다(하위 프로세스 '--part <구성 약어>') 첫 시도 t000(대상 판 설정 그대로 — 단일 트리·이상 점수 칼럼
          없음이면 대상 판 CV(학습 없음), 아니면 그 구성 트리 단독 3 fold) + TPE(씨앗 42) 19회(tuning.run_optuna —
          jsonl 이어하기) → tuning.select_tuned(top 5) = tune_<약어>.json·top5_<약어>.csv.
  T       8-T 판(3 fold + refit): 채택 후보 구성만 튜닝 설정(다른 구성은 기본 유지), 이상 점수 칼럼 다시 붙임 — 모든 구성이
          기본 유지면 학습 없음.
  judgeT  학습 없음: 8-T(쌓는 후보) 대 현재 최고 판 → chain_state_mid·8T_decision.json·사슬 기록, MLflow 8-T 판 run
          (기본 유지면 판 run 만 만들고 '기본 유지') + 시도 run(자식, <판 이름>-<약어>-t000~t019, 점수만).
  F       Focal 판: 그 앞까지 채택한 판(chain_state_mid 의 현재 최고 판)의 CatBoost 부분만 손실 Focal(γ 2·α 0.75) —
          CatBoost 구성이 없으면 판 없음.
  judgeF  학습 없음: Focal(쌓는 후보) 대 그때의 현재 최고 판 → chain_state_out·8_decision.json·사슬 기록.
보고(--report --reason, judgeF 뒤): T·F 설정(7-a·1차 베이스라인·대상 판 줄, 앙상블이면 구성별 기법 전후 CV 주석).

실행(서버 — 검수 뒤 지휘자, 5·6 판정 뒤):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s8.py $A --out ~/cyj_listeria_final/results/s8/8-tune --config configs/s8_tune.json
  $P scripts/s8.py $A --out ~/cyj_listeria_final/results/s8/8-T --config configs/s8_T.json
  $P scripts/s8.py $A --out ~/cyj_listeria_final/results/s8/8-judge --config configs/s8_judgeT.json
  $P scripts/s8.py $A --out ~/cyj_listeria_final/results/s8/8-F --config configs/s8_F.json
  $P scripts/s8.py $A --out ~/cyj_listeria_final/results/s8/8-judge --config configs/s8_judgeF.json
  보고: T·F 명령 + --report --reason "단계 8 보고"

구현 세부(계획서에 없음 — 담당자 확인):
 (t1) 8-T 판 3 fold 도 runner 로 다시 학습(시도와 같은 설정·씨앗 — 같은 값; 부록 A-9 셈 '19 × 3 + refit 1' 보다 3 학습 더).
      대상이 단일 트리·이상 점수 칼럼 없음이면 8-T 판 CV 와 고른 시도 점수가 같아야 함 → judgeT 에서 대조해 기록.
 (t2) 첫 시도 t000 의 CatBoost 학습률 = 대상 판(그 구성) 의 학습률(done.json n_iter.learning_rate, 없으면 params 의
      learning_rate) — '대상 판 설정 그대로'. 8-T 판·Focal 판에서 기본 유지 CatBoost 구성도 이 값을 params 에 넣어 runner
      가 다시 읽지 않게(§3 '대상 판의 학습률 그대로'). ④ 의 CatBoost 는 라이브러리 기본(자동값) 그대로.
 (t3) 시도 기록 파일 = 결과 폴더 optuna_<약어>.jsonl, 첫 시도 = t000_<약어>.json(이어하기 — 있으면 다시 안 매김).
 (t4) 5·6 판 대상이면 시도·판 학습 씨앗 0(score_tree_config seed=0·derive_specs model_seed=0), TPE 씨앗 42 그대로.
 (t5) 앙상블 대상: judgeT 기록에 '튜닝 설정 앙상블 대 기본 설정 앙상블' CV(= 8-T 앙상블 CV 대 대상 판 CV)와 구성별 전후 CV.
 (t6) 시도 run 의 tag 역할은 비움(부록 A-15 역할 값에 '시도' 가 없음), metric = cv_band_weighted·fold 별 점수.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import metrics as mt  # noqa: E402
from lst import registry as rg  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage4 as s4  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst import tuning as tu  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, check_resources, limit_gpu_memory  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = s4.STAGE8
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
N_TPE = 19                       # 첫 시도 + 19 = 20(§6 단계 8)
TOP = 5
TRAIN_SUBS = ("T", "F")
SUB_DIR = {"tune": "8-tune", "T": "8-T", "F": "8-F", "judgeT": "8-judge", "judgeF": "8-judge"}
DIRS_SCHEMA = {k: cf.Field(str) for k in ("8-tune", "8-T", "8-F", "8-judge")}
COMMON_EXTRA = {
    "sub": cf.Field(str, choices=("tune", "T", "F", "judgeT", "judgeF")),
    "decision_1_3": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "chain_state_in": cf.Field(str),
    "chain_state_mid": cf.Field(str),
    "theil_targets": cf.Field(str),
    "dirs": DIRS_SCHEMA,
    "gpu_memory_fraction": cf.Field(float, required=False),
    "device": cf.Field(str, required=False),
}
JUDGE_SCHEMA = {"plan": cf.Field(str), "sub": cf.Field(str, choices=("judgeT", "judgeF")), "stage": cf.Field(str),
                "order": cf.Field(str), "experiment": cf.Field(str), "samples": cf.Field(str),
                "experiment_log": cf.Field(str), "use_mlflow": cf.Field(bool), "mlflow_uri": cf.Field(str),
                "code_reviewed": cf.Field(bool), "chain_state_in": cf.Field(str), "chain_state_mid": cf.Field(str),
                "chain_state_out": cf.Field(str, required=False), "chain_log": cf.Field(str), "dirs": DIRS_SCHEMA}


def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason", "--part"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run"):
        ap.add_argument(a, action="store_true")
    return ap.parse_known_args(list(argv))[0]


# ================================================================ 튜닝할 부분(구성)
@dataclass
class Part:
    label: str                     # 구성 약어(시도 run 이름·파일 이름)
    alg: str                       # 판 이름 약어(XGB·LGB·CAT·RF·ET·LR)
    member: dict                   # MemberDef
    src: rn.VariantSpec            # 구성 판 설정
    plan: s4.RowPlan
    catenc_k: int | None
    seed: int                      # 학습 씨앗(5·6 판 = 0)
    base_params: dict              # 첫 시도(대상 판 설정)의 라이브러리 params
    max_iter: int | None           # 부스팅 최대 반복(실험 로그) — ④·RF·ET·LR = None
    t000_from_target: bool         # 단일 트리·이상 점수 칼럼 없음 → 대상 판 CV 를 첫 시도 점수로
    tune_lr: bool
    stack4: dict | None = None     # ④ 기본 트리면 {'version','W','mode'}

    @property
    def algo(self) -> str:
        return s4.TREE_ALGO[self.alg]

    @property
    def rep(self) -> int:
        return s4.M_REP if self.plan.rep == "4-a" else 1


def _member_s4(src: rn.VariantSpec) -> dict:
    return dict(src.extra.get("s4") or {})


def member_plan(m: Mapping, src: rn.VariantSpec) -> tuple[s4.RowPlan, int | None, int]:
    """구성의 RowPlan·범주 숫자 인코딩 k·학습 씨앗(MemberDef 칸 우선, 없으면 판 설정 extra.s4)."""
    e = _member_s4(src)
    plan = s4.RowPlan.from_dict(m.get("row_plan") if m.get("row_plan") is not None else e.get("row_plan"))
    k = m.get("catenc_k") if m.get("catenc_k") is not None else e.get("catenc_k")
    seed = int(e.get("model_seed", 42))
    return plan, (int(k) if k is not None else None), seed


def member_lr(inputs_root: Path, m: Mapping, src: rn.VariantSpec) -> float | None:
    """구성 판의 CatBoost 학습률(구현 세부 t2): params 의 learning_rate → done.json n_iter.learning_rate → n_iter.json."""
    lib = dict(src.params.get("params") or {})
    if "learning_rate" in lib:
        return float(lib["learning_rate"])
    d = resolve(inputs_root, m["dir"])
    done = rn.read_json(d / "done.json")
    lr = (done.get("n_iter") or {}).get("learning_rate")
    if lr is None:
        ni = rn.load_n_iter(d)
        lr = None if ni is None else (ni.get("summary") or {}).get("learning_rate")
    return None if lr is None else float(lr)


def tune_parts(state: Mapping, inputs_root: Path, exp_log: rn.ExperimentLog | None) -> list[Part]:
    """대상(현재 최고 판)의 튜닝할 트리 부분 — 딥러닝 단독·결합·② 는 빈 목록."""
    best = state["current_best"]
    kind = best["kind"]
    if kind in ("dl", "nn60"):
        return []

    def max_iter(alg):
        if alg not in s4.BOOST:
            return None
        rec = exp_log.max_iter_record(alg) if exp_log is not None else None
        if rec is None:
            raise s4.StageError(f"{alg} 최대 반복 기록이 실험 로그에 없음(§3 — 첫 2차 사슬 부스팅 판 뒤)")
        return int(rec["max_iter"])

    out = []
    if kind in ("tree", "ensemble"):
        for m in best["members"]:
            src = s4.member_spec(inputs_root, m)
            if src.kind != "tree":
                raise s4.StageError(f"트리 판 구성이 트리가 아님: {src.name}")
            plan, k, seed = member_plan(m, src)
            lib = dict(src.params.get("params") or {})
            if src.algorithm == "CAT" and "learning_rate" not in lib:
                lr = member_lr(inputs_root, m, src)
                if lr is None:
                    raise s4.StageError(f"{src.name}: CatBoost 학습률 기록 없음")
                lib["learning_rate"] = lr
            out.append(Part(label=src.algorithm, alg=src.algorithm, member=dict(m), src=src, plan=plan, catenc_k=k,
                            seed=seed, base_params=lib, max_iter=max_iter(src.algorithm),
                            t000_from_target=(kind == "tree" and not m.get("anomaly")), tune_lr=True))
        if len({p.label for p in out}) != len(out):
            raise s4.StageError("앙상블 구성 알고리즘이 겹침")
        return out
    # stack4 — 기본 트리 6종
    m = best["members"][0]
    src = s4.member_spec(inputs_root, m)
    plan, k, seed = member_plan(m, src)
    st = dict(m["stack4"])
    prev = dict((_member_s4(src).get("provider") or {}).get("params") or {})
    for a in s4.TREE6:
        out.append(Part(label=a, alg=a, member=dict(m), src=src, plan=plan, catenc_k=k, seed=seed,
                        base_params=dict(prev.get(a) or {}), max_iter=None, t000_from_target=False, tune_lr=False,
                        stack4={"version": st["version"], "W": int(st.get("W", src.W)),
                                "mode": st.get("mode", src.mode)}))
    return out


def is_hon(parts) -> bool:
    return any(p.seed == 0 for p in parts)


# ================================================================ 시도
def part_assembler(w: s4.Stage4Wiring, part: Part, split: str):
    """튜닝용 조립 객체(이상 점수 칼럼 뺌 — §6 단계 8)."""
    if part.stack4 is not None:
        vs = rg.load_version({"name": f"x4-L6-{part.alg}", "order": 2, "W": part.stack4["W"],
                              "mode": part.stack4["mode"], **part.stack4["version"]})
        return s4.S4TreeInputs(w, vs, split, catenc_k=part.catenc_k, scale=(part.alg == "LR"))
    sp = s4._member_variant(part.src, name=f"8-tune-{part.label}", stage=STAGE, role=s4.ROLE_CAND,
                            s4={"row_plan": part.plan.to_dict(), "catenc_k": part.catenc_k}, base_name=part.src.name,
                            params={})
    return w.make_inputs(sp, split)


def make_folds(w: s4.Stage4Wiring, part: Part, samples: ev.SampleTable, inputs_root: Path, *, smoke=None,
               check=None) -> list:
    known = samples.known_frame()
    out = []
    for f in ev.CV_FOLDS:
        rows = s4.plan_train_rows(known, f, part.plan, inputs_root)
        if smoke is not None:
            rows = rn._smoke_train(rows, smoke["max_train_rows"], 42)
        out.append(tu.TreeFold(f, samples, part_assembler(w, part, f), train_rows=rows, cache_val=True, check=check))
    return out


def target_cv(inputs_root: Path, state: Mapping) -> float:
    d = resolve(inputs_root, state["current_best"]["dir"])
    return float(rn.load_done(d.parent, d.name)[2]["cv_band_weighted"])


def tune_part(part: Part, out: Path, *, folds_fn: Callable[[], list] | None, base_cv: Callable[[], float],
              score_config: Callable = tu.score_tree_config, n_jobs: int = 8) -> dict:
    """첫 시도 t000 + TPE 19 → 고르기(select_tuned, top 5). 끝났으면(tune_<약어>.json) 그대로."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res_p = out / f"tune_{part.label}.json"
    if res_p.exists():
        return rn.read_json(res_p)
    folds_holder: dict = {}

    def folds():
        if "f" not in folds_holder:
            folds_holder["f"] = folds_fn()
        return folds_holder["f"]

    t0_p = out / f"t000_{part.label}.json"
    if t0_p.exists():
        t0 = rn.read_json(t0_p)
    else:
        if part.t000_from_target:
            t0 = {"cv": float(base_cv()), "source": "대상 판 CV(학습 없음)", "params": part.base_params}
        else:
            r = score_config(part.algo, dict(part.base_params), folds(), max_iter=part.max_iter, seed=part.seed,
                             n_jobs=n_jobs)
            t0 = {"cv": float(r["cv"]), "source": "그 구성 트리 단독(이상 점수 칼럼 뺀 입력) 3 fold",
                  "params": part.base_params, "fold_scores": r.get("fold_scores"), "cross_n": r.get("cross_n")}
        cf.dump_scores_json(t0_p, t0)
    recs = tu.run_optuna(part.algo, lambda p: score_config(part.algo, p, folds(), max_iter=part.max_iter,
                                                          seed=part.seed, n_jobs=n_jobs),
                         log_path=out / f"optuna_{part.label}.jsonl", n_trials=N_TPE, seed=tu.TPE_SEED,
                         rep=part.rep, tune_lr=part.tune_lr, label=part.label)
    trials = [r for r in recs if r.get("event", "trial") == "trial"]
    if len(trials) != N_TPE:
        raise s4.StageError(f"{part.label}: TPE 시도 {len(trials)} ≠ {N_TPE}")
    sel = tu.select_tuned(float(t0["cv"]), trials, top=TOP)
    pd.DataFrame(sel["top"]).assign(params=lambda d: d["params"].map(lambda v: json.dumps(v, ensure_ascii=False))) \
        .to_csv(out / f"top5_{part.label}.csv", index=False, encoding="utf-8-sig")
    res = {"label": part.label, "alg": part.alg, "member": part.member["name"], "t000": t0, "selection": sel,
           "adopt": bool(sel["adopt"]), "params": (sel["selected"] or {}).get("params"),
           "max_iter": part.max_iter, "rep": part.rep, "seed": part.seed, "tune_lr": part.tune_lr,
           "space": tu.search_space(part.algo, rep=part.rep, tune_lr=part.tune_lr),
           "n_trials_total": 1 + len(trials),
           "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    cf.dump_scores_json(res_p, res)
    return res


def load_tune(inputs_root: Path, cfg: Mapping, parts: list[Part]) -> dict[str, dict]:
    d = resolve(inputs_root, cfg["dirs"]["8-tune"])
    out = {}
    for p in parts:
        f = d / f"tune_{p.label}.json"
        if not f.exists():
            raise s4.StageError(f"튜닝 결과 없음(tune 먼저): {f}")
        out[p.label] = rn.read_json(f)
    return out


# ================================================================ 판 설정(8-T·Focal)
def _pin_lr(spec: rn.VariantSpec, lr: float | None) -> rn.VariantSpec:
    """CatBoost params 에 learning_rate 가 없으면 넣음(구현 세부 t2)."""
    if spec.algorithm != "CAT":
        return spec
    p = dict(spec.params)
    lib = dict(p.get("params") or {})
    if "learning_rate" in lib:
        return spec
    if lr is None:
        raise s4.StageError(f"{spec.name}: CatBoost 학습률을 정할 수 없음")
    lib["learning_rate"] = float(lr)
    p["params"] = lib
    return dataclasses.replace(spec, params=p)



def build_T(cfg: Mapping, inputs_root: Path, exp_log: rn.ExperimentLog | None) -> dict:
    """8-T 판 설정. status = '튜닝 안 함'(트리 부분 없음)·'기본 유지'(모든 구성 기본 유지 — 학습 없음)·'판'."""
    state = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
    target = state["current_best"]
    parts = tune_parts(state, inputs_root, exp_log)
    name = f"8-T-{s4.name_part(target, inputs_root)}"
    base = {"run": [], "ens": [], "best": None, "name": name, "parts": parts, "state": state}
    if not parts:
        return {**base, "status": "튜닝 안 함"}
    tunes = load_tune(inputs_root, cfg, parts)
    if not any(t["adopt"] for t in tunes.values()):
        return {**base, "status": "기본 유지", "tunes": tunes}
    plan, k, seed = parts[0].plan, parts[0].catenc_k, parts[0].seed
    for p in parts[1:]:
        if (p.plan, p.catenc_k, p.seed) != (plan, k, seed):
            raise s4.StageError("구성마다 RowPlan·범주 인코딩·씨앗이 다름")
    if target["kind"] == "stack4":
        mp = {target["members"][0]["name"]: {p.alg: (tunes[p.label]["params"] if tunes[p.label]["adopt"]
                                                     else (p.base_params or None)) for p in parts}}
        mp = {kk: {a: v for a, v in d.items() if v} for kk, d in mp.items()}
    else:
        mp = {p.member["name"]: tunes[p.label]["params"] for p in parts if tunes[p.label]["adopt"]}
    d = s4.derive_specs(target, inputs_root=inputs_root, name=name, stage=STAGE, role=s4.ROLE_CAND, plan=plan,
                        catenc_k=k, model_seed=(0 if seed == 0 else None), member_params=mp,
                        anomaly_reuse=True)
    run = d.run
    if target["kind"] in ("tree", "ensemble"):
        by = {p.member["name"]: p for p in parts}
        run = [_pin_lr(sp, by[m["name"]].base_params.get("learning_rate")) for sp, m in zip(run, target["members"])]
    best = s4.fill_dirs(d.best, cfg["dirs"]["8-T"])
    return {**base, "status": "판", "run": run, "ens": [d.ensemble] if d.ensemble is not None else [], "best": best,
            "tunes": tunes}


def has_catboost(best: Mapping, inputs_root: Path) -> bool:
    if best["kind"] == "stack4":
        return True
    if best["kind"] in ("dl", "nn60"):
        return False
    return any(s4.member_spec(inputs_root, m).algorithm == "CAT" for m in best["members"])


def build_F(cfg: Mapping, inputs_root: Path) -> dict:
    """Focal 판 = 그 앞까지 채택한 판(chain_state_mid 현재 최고 판)의 CatBoost 손실만 Focal."""
    mid = s4.load_state(resolve(inputs_root, cfg["chain_state_mid"]))
    b = mid["current_best"]
    base = {"run": [], "ens": [], "best": None, "state": mid, "base": b["name"], "name": f"8-F-{b['name']}"}
    if b["kind"] in ("dl", "nn60"):
        return {**base, "status": "튜닝 안 함"}
    if not has_catboost(b, inputs_root):
        return {**base, "status": "CatBoost 구성 없음"}
    m0 = b["members"][0]
    plan, k, seed = member_plan(m0, s4.member_spec(inputs_root, m0))
    d = s4.derive_specs(b, inputs_root=inputs_root, name=base["name"], stage=STAGE, role=s4.ROLE_CAND, plan=plan,
                        catenc_k=k, model_seed=(0 if seed == 0 else None), focal_cat=True,
                        anomaly_reuse=True)
    run = d.run
    if b["kind"] in ("tree", "ensemble"):
        run = [_pin_lr(sp, member_lr(inputs_root, m, s4.member_spec(inputs_root, m)))
               for sp, m in zip(run, b["members"])]
    return {**base, "status": "판", "run": run, "ens": [d.ensemble] if d.ensemble is not None else [],
            "best": s4.fill_dirs(d.best, cfg["dirs"]["8-F"])}


# ================================================================ 실행 연결
def check_cfg(cfg) -> None:
    if cfg["stage"] != STAGE or cfg["order"] != s4.ORDER or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("stage '8' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
    f = cfg.get("gpu_memory_fraction")
    if f is not None and not (0 < f <= GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    if cfg["n_jobs"] != 8:
        raise cf.ConfigError("n_jobs 8")


def make_specs_fn(inputs_root: Path):
    holder: dict = {}

    def specs_fn(cfg):
        check_cfg(cfg)
        if "b" not in holder:
            log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
            holder["b"] = build_T(cfg, inputs_root, log) if cfg["sub"] == "T" else build_F(cfg, inputs_root)
        return holder["b"]["run"]
    specs_fn.holder = holder
    return specs_fn


def wiring_for(cfg, root: Path, inputs_root: Path, out: Path) -> s4.Stage4Wiring:
    from lst import inputs as ip

    dl_kw = {"max_epochs": cfg["smoke"]["max_epochs"]} if "smoke" in cfg and "max_epochs" in cfg["smoke"] else {}
    ctx = ip.load_context(cfg["context"], root, inputs_root, components=list(COMPONENTS))
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    return s4.Stage4Wiring(ctx, samples, inputs_root, data_root=root,
                           theil_targets=resolve(inputs_root, cfg["theil_targets"]), n_jobs=cfg["n_jobs"],
                           device=cfg.get("device"), dl_kwargs=dl_kw, check=lambda: check_resources(out))


def factories(cfg, rp, inputs_root: Path) -> sc.Factories:
    from lst import wiring as wr

    w = wiring_for(cfg, rp.root, inputs_root, rp.out)
    return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                        input_md5s=wr.input_md5s(cfg["context"], rp.root, inputs_root), fit_kwargs=w.fit_kwargs)


def before_variant(cfg) -> None:
    if cfg.get("gpu_memory_fraction"):
        limit_gpu_memory(cfg["gpu_memory_fraction"])


def report_rows_for(inputs_root: Path):
    def report_rows(cfg, _ir, samples, exp_log):
        from lst import stage3 as s3

        jd = resolve(inputs_root, cfg["dirs"]["8-judge"]) / "8_decision.json"
        if not jd.exists():
            raise SystemExit("단계 8 판정(judgeF)을 먼저 — test 는 판정 뒤에만(§1-3)")
        dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
        rows = [s3.baseline_1_row(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"])]
        key = "chain_state_in" if cfg["sub"] == "T" else "chain_state_mid"
        st = s4.load_state(resolve(inputs_root, cfg[key]))
        rows.append(s4.best_row(inputs_root, st["current_best"],
                                "단계 8 대상 판(단계 시작 때 현재 최고 판)" if cfg["sub"] == "T"
                                else "Focal 바탕 판(그 앞까지 채택한 판)"))
        out = resolve(inputs_root, cfg["dirs"][SUB_DIR[cfg["sub"]]])
        b = make_specs_fn(inputs_root)
        b(cfg)
        for e in b.holder["b"]["ens"]:
            d = out / e.name
            done = rn.read_json(d / "done.json")
            rows.append(rn.BaseRow(e.name, f"앙상블(구성 {', '.join(e.extra['s4']['members'])} 평균)",
                                   rn.load_preds(d, ev.CV_FOLDS), pd.read_parquet(rn.pred_path(d, "refit")),
                                   done["selection_count"], done.get("mlflow_run_id"), thr_preds=rn.load_thr_preds(d)))
        return rows
    return report_rows


def member_notes(cfg, inputs_root: Path, built: Mapping) -> list[str]:
    """앙상블 대상: 구성별 기법 전후 CV(부록 A-12) 주석."""
    st = built["state"]
    b = st["current_best"]
    if b["kind"] != "ensemble" or not built["run"]:
        return []
    out_rel = cfg["dirs"][SUB_DIR[cfg["sub"]]]
    notes = []
    for m, sp in zip(b["members"], built["run"]):
        before = s4._cv_of(resolve(inputs_root, m["dir"]))[0]
        d = resolve(inputs_root, f"{out_rel}/{sp.name}")
        after = s4._cv_of(d)[0] if (d / "done.json").exists() else float("nan")
        notes.append(f"구성 {m['name']} → {sp.name}: CV {before:.4f} → {after:.4f}")
    return notes


NOTES = ("단계 8 시도·판 점수 = 대상 판과 같은 방식(부록 A-12) — 부스팅 = 교차 N(§3), ④ = 라이브러리 기본 반복 수",
         "트리 튜닝은 이상 점수 칼럼을 빼고 한 뒤 판에서 다시 붙임(§6 단계 8)",
         "5·6 판이 대상이면 학습 씨앗 0(MLflow param seed 는 42 로 남음 — 판 설정 extra.s4.model_seed = 0)")


# ================================================================ tune
def load_cfg(path) -> dict:
    cfg = sc.load_cfg(path, COMMON_EXTRA)
    check_cfg(cfg)
    return cfg


def tune_main(argv, ns, *, wiring_fn=wiring_for, python: str | None = None) -> dict:
    ap = sc.parser(__doc__.splitlines()[0])
    ap.add_argument("--part")
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    inputs_root = Path(args.inputs).expanduser().resolve()
    if rp.out.resolve() != resolve(inputs_root, cfg["dirs"]["8-tune"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[8-tune]")
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    state = s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))
    parts = tune_parts(state, inputs_root, log)
    rp.out.mkdir(parents=True, exist_ok=True)
    if not parts:
        cf.dump_json(rp.out / "no_tuning.json", {"target": state["current_best"]["name"],
                                                 "reason": "트리 부분이 없는 판 — 튜닝 안 함(§6 단계 8)"}, overwrite=True)
        print("단계 8: 대상에 트리 부분이 없음 — 튜닝 안 함")
        return {}
    if args.part:
        part = next((p for p in parts if p.label == args.part), None)
        if part is None:
            raise KeyError(f"구성 없음: {args.part}")
        rn.snapshot_code(CODE_ROOT, rp.out)
        samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
        w = wiring_fn(cfg, rp.root, inputs_root, rp.out)
        smoke = {k: cfg["smoke"][k] for k in rn.SMOKE_KEYS} if "smoke" in cfg else None
        res = tune_part(part, rp.out,
                        folds_fn=lambda: make_folds(w, part, samples, inputs_root, smoke=smoke,
                                                    check=lambda: check_resources(rp.out)),
                        base_cv=lambda: target_cv(inputs_root, state), n_jobs=cfg["n_jobs"])
        log.append("tuning", stage=STAGE, part=part.label, adopt=res["adopt"], best_cv=res["selection"]["best_cv"],
                   baseline_cv=res["selection"]["baseline_cv"], n_trials=res["n_trials_total"])
        print(f"{part.label}: {res['selection']['judgement']} (t000 {res['selection']['baseline_cv']:.4f}, "
              f"1위 {res['selection']['best_trial']} {res['selection']['best_cv']:.4f})")
        return res
    out = {}
    for p in parts:      # 구성마다 하위 프로세스(B-1 — 멈추면 그 구성부터, run_optuna 가 시도 이어함)
        if (rp.out / f"tune_{p.label}.json").exists():
            out[p.label] = rn.read_json(rp.out / f"tune_{p.label}.json")
            continue
        rc = subprocess.call([python or sys.executable, str(Path(__file__).resolve()), *argv, "--part", p.label])
        if rc != 0:
            log.append("stopped", stage=STAGE, name=f"tune-{p.label}", returncode=rc, reason=rn._rc_reason(rc))
            raise rn.RunStopped(f"튜닝 {p.label} 멈춤 — {rn._rc_reason(rc)}. 원인을 없앤 뒤 같은 명령을 다시(B-1)")
        out[p.label] = rn.read_json(rp.out / f"tune_{p.label}.json")
    return out


# ================================================================ judge
def _trial_runs(tr, parent_id: str, name: str, tunes: Mapping[str, dict], tune_dir: Path) -> None:
    """시도 run(부록 A-15): <판 이름>-<약어>-t000 ~ t019, 점수만."""
    for label, t in tunes.items():
        rid = tr.child_run(s4.EXPERIMENT, parent_id, f"{name}-{label}-t000", "시도", {"단계": STAGE})
        tr.log_metrics(rid, {"cv_band_weighted": t["t000"]["cv"]})
        tr.finish(rid)
        for r in tu._read_log(tune_dir / f"optuna_{label}.jsonl"):
            if r.get("event") != "trial":
                continue
            rid = tr.child_run(s4.EXPERIMENT, parent_id, f"{name}-{label}-{tu.trial_label(r['number'])}", "시도",
                               {"단계": STAGE})
            tr.log_params(rid, r["raw"])
            tr.log_metrics(rid, {"cv_band_weighted": r["value"],
                                 **{f"{f}_band_weighted": v for f, v in
                                    ((r.get("result") or {}).get("fold_scores") or {}).items() if v is not None}})
            tr.finish(rid)


def judge_T_core(cfg: Mapping, inputs_root: Path, samples: ev.SampleTable, exp_log: rn.ExperimentLog | None) -> dict:
    b = build_T(cfg, inputs_root, exp_log)
    state = b["state"]
    rec = {"step": "8-T", "status": b["status"], "target": state["current_best"]["name"], "name": b["name"],
           "parts": [], "tunes": {k: {"adopt": v["adopt"], "t000": v["t000"]["cv"],
                                      "best": v["selection"]["best_cv"], "best_trial": v["selection"]["best_trial"],
                                      "top": v["selection"]["top"]} for k, v in (b.get("tunes") or {}).items()},
           "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    if b["status"] != "판":
        new = dict(state)
        new["step"] = "8-T"
        new["history"] = list(state.get("history", [])) + [{"step": "8-T", "before": state["current_best"]["name"],
                                                            "after": state["current_best"]["name"],
                                                            "judgement": b["status"], "utc": rec["utc"]}]
        rec.update(before=state["current_best"]["name"], after=state["current_best"]["name"])
        return {"record": rec, "state": new, "built": b}
    d = resolve(inputs_root, b["best"]["dir"])
    res = s4.judge_step(step="8-T", state=state, stack=[(b["best"], d)], repl=[], samples=samples,
                        inputs_root=inputs_root)
    rec.update(res["record"])
    rec["status"] = b["status"]
    tcv = target_cv(inputs_root, state)
    cand_cv = s4._cv_of(d)[0]
    rec["same_score_comparison"] = bool(state["current_best"]["kind"] == "tree"
                                        and not state["current_best"]["members"][0].get("anomaly"))
    if rec["same_score_comparison"]:   # 구현 세부 t1: 8-T 판 CV 와 고른 시도 점수 대조
        t = next(iter(b["tunes"].values()))
        rec["trial_vs_variant_cv"] = {"trial": t["selection"]["best_cv"], "variant": cand_cv,
                                      "abs_diff": abs(t["selection"]["best_cv"] - cand_cv)}
    if state["current_best"]["kind"] == "ensemble":   # 구현 세부 t5
        rec["ensemble_tuned_vs_default"] = {"tuned": cand_cv, "default": tcv}
    return {"record": rec, "state": res["state"], "built": b}


def judge_F_core(cfg: Mapping, inputs_root: Path, samples: ev.SampleTable) -> dict:
    b = build_F(cfg, inputs_root)
    state = b["state"]
    utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    rec = {"step": "8-F", "status": b["status"], "base": b["base"], "name": b["name"], "parts": [], "utc": utc}
    if b["status"] != "판":
        new = dict(state)
        new["step"] = "8-F"
        new["history"] = list(state.get("history", [])) + [{"step": "8-F", "before": b["base"], "after": b["base"],
                                                            "judgement": b["status"], "utc": utc}]
        rec.update(before=b["base"], after=b["base"])
        return {"record": rec, "state": new, "built": b}
    res = s4.judge_step(step="8-F", state=state, stack=[(b["best"], resolve(inputs_root, b["best"]["dir"]))], repl=[],
                        samples=samples, inputs_root=inputs_root)
    rec.update(res["record"])
    rec["status"] = b["status"]
    return {"record": rec, "state": res["state"], "built": b}


def judge(argv, *, tracker_fn=sc.make_tracker) -> dict:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))
    if cfg["stage"] != STAGE or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("judge 설정: stage '8' · experiment listeria-2차")
    sub = cfg["sub"]
    if sub == "judgeF" and "chain_state_out" not in cfg:
        raise cf.ConfigError("judgeF 에는 chain_state_out")
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = rp.out
    if out.resolve() != resolve(inputs_root, cfg["dirs"]["8-judge"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[8-judge]")
    p = out / ("8T_decision.json" if sub == "judgeT" else "8_decision.json")
    if p.exists():
        print(f"판정 이미 있음: {p}")
        return rn.read_json(p)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    if sub == "judgeT":
        r = judge_T_core(cfg, inputs_root, samples, log)
        state_path = cfg["chain_state_mid"]
    else:
        if not (out / "8T_decision.json").exists():
            raise SystemExit("8-T 판정(judgeT)을 먼저")
        r = judge_F_core(cfg, inputs_root, samples)
        state_path = cfg["chain_state_out"]
    rec, new, b = r["record"], r["state"], r["built"]
    tr = tracker_fn(cfg)
    if tr is not None:
        sr = tr.stage_run(s4.EXPERIMENT, STAGE)
        if b["status"] == "판":
            s4.tag_judgements(tr, rec, lambda n: tr.find_run(s4.EXPERIMENT, n, sr))
        if sub == "judgeT" and b["status"] in ("판", "기본 유지"):
            run = tr.find_run(s4.EXPERIMENT, b["name"], sr)
            if run is None:      # 기본 유지 — 판 run 만 만듦(부록 A-15)
                rid = tr.start_variant(s4.EXPERIMENT, sr, b["name"], {"단계": STAGE, "역할": s4.ROLE_CAND},
                                       {"parts": [p.label for p in b["parts"]]})
                tr.set_judgement(rid, "기본 유지", against=rec["target"])
                tr.finish(rid)
            else:
                rid = run.info.run_id
            _trial_runs(tr, rid, b["name"], b["tunes"], resolve(inputs_root, cfg["dirs"]["8-tune"]))
    line = {"step": rec["step"], "judgement": ("채택" if rec.get("after") != rec.get("before") else
                                               (b["status"] if b["status"] != "판" else "버림")),
            "before": rec.get("before"), "after": rec.get("after"), "tree_target": new["tree_target"]["name"],
            "utc": rec["utc"]}
    cpath = s4.append_chain_log(inputs_root, cfg["chain_log"], line)
    if tr is not None:
        tr.log_artifact(tr.chain_run(), cpath)
    for part in rec.get("parts", []):
        for a in part["judge"]["alerts"]:
            log.append("alert", stage=STAGE, message=a)
            print(f"[알림] {a}")
    log.append("decision", stage=rec["step"], before=rec.get("before"), after=rec.get("after"), status=b["status"])
    s4.save_state(resolve(inputs_root, state_path), new)
    out.mkdir(parents=True, exist_ok=True)
    body = {"record": rec, "state_out": state_path}
    if sub == "judgeF":
        body["record_8T"] = rn.read_json(out / "8T_decision.json")["record"]
    cf.dump_scores_json(p, body)
    if tr is not None:
        tr.log_artifact(tr.stage_run(s4.EXPERIMENT, STAGE), p, "decision")
    print(f"{rec['step']}: {b['status']} — {rec.get('before')} → {rec.get('after')}")
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
    if sub in ("judgeT", "judgeF"):
        return judge(argv, tracker_fn=tracker_fn)
    if sub == "tune":
        return tune_main(argv, ns)
    if sub not in TRAIN_SUBS:
        raise cf.ConfigError(f"sub: {sub!r}")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    if not (ns.list or ns.verify_run):
        want = resolve(inputs_root, raw["dirs"][SUB_DIR[sub]]).resolve()
        if Path(ns.out).expanduser().resolve() != want:
            raise cf.ConfigError(f"--out 은 --inputs 아래 dirs[{SUB_DIR[sub]}] = {want}")
    specs_fn = make_specs_fn(inputs_root)
    cfg = load_cfg(ns.config)
    specs = specs_fn(cfg)
    built = specs_fn.holder["b"]
    if not specs:
        msg = f"단계 8 {sub}: 판 없음 — {built['status']}"
        print(msg)
        if not (ns.list or ns.report):
            rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])).append("note", stage=STAGE, message=msg)
        return []
    notes = NOTES
    if ns.variant:
        by = {s.name: s for s in specs}
        group = [by[n.strip()] for n in ns.variant.split(",") if n.strip()]
        s4.install_for_group(group, inputs_root, None)
    if ns.report:
        notes = NOTES + tuple(member_notes(cfg, inputs_root, built))
    res = sc.stage_main(argv, script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=COMMON_EXTRA,
                        specs_fn=specs_fn, factories_fn=factories_fn or factories, before_variant=before_variant_fn,
                        fold_recall=True, tracker_fn=tracker_fn, check_resources=check_resources_fn,
                        report_rows_fn=report_rows_for(inputs_root), report_notes=notes)
    default = not (ns.list or ns.variant or ns.verify_run or ns.report or ns.only)
    if default and built["ens"]:
        samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
        log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
        tr = tracker_fn(cfg)
        sr = tr.stage_run(cfg["experiment"], STAGE) if tr is not None else None
        outp = Path(ns.out).expanduser().resolve()
        code = rn.snapshot_code(CODE_ROOT, outp)
        for e in built["ens"]:
            d = s4.write_ensemble(outp, e, samples, log, code, tracker=tr, stage_run_id=sr)
            print(f"{e.name}: 앙상블 CV {d['cv_band_weighted']}")
    return res


if __name__ == "__main__":
    main()
