"""단계 3-0b 변수 빼기(1등) 공통 — 계획서(v78) 요약 15, §6 2차 공통(교체 후보·트리 대상 판)·단계 3-0b, §1-2(4자리·동점),
부록 A-5(그룹·중요도 — 1등이 앙상블·스태킹일 때 줄 포함)·A-9(3-0b 0~12)·A-10·A-12(3-0b 끝 판 = 1등 판과 같은 셈)·A-14(판 이름
구조 자리)·A-15(단계 run '3-0b', 끝 판 tag '원 판').

1등 = 3-0 ~ 3-3 판 전부(⑥·⑦·④ 포함, 비교만 판 제외) 중 CV 1위 = 3-3 판정 뒤 사슬 상태(logs/chain_state_3-3.json)의 현재
최고 판(동점이면 현재 최고 판). 종류(트리·딥러닝 단독·결합·앙상블·④·이상 점수 칼럼 트리·교체 후보)와 관계없이:
  1. 중요도: 1등 fold 모델(앙상블이면 구성 모델 전부 — importance.Mean, 재학습 없음)로 그룹 칼럼을 `h_block` 으로 섞음
     (M = 10, 대상 = 부록 A-5 44 그룹 중 1등 입력에 칼럼·채널이 있는 그룹 — 노드의 present_units). 이상 점수 칼럼은 3-1-VAEt
     저장 값 그대로(VAE 입력은 안 섞음 — v78 A-5), 범주 5·h·g·모델 확률 칼럼은 대상 아님. ④ = 대상 그룹 없음.
  2. 빼기: 3M 칸 평균 I ≤ 0 그룹을 한꺼번에 뺀 후보 1판 = 1등을 줄인 입력으로 3 fold + refit 다시 학습(부스팅은 반복 수 고르기
     켠 채, 씨앗 1개, k·N_ma 그대로, VAE 다시 학습 안 함). 앙상블 = 구성마다(뺄 그룹이 입력에 없는 구성은 그대로) → 다시 평균.
     평균 I ≤ 0 그룹이 없거나 빼는 대상 그룹이 없으면(④) 학습 없이 1등 유지.
  3. 채택 = CV 평균 상승(4자리) + 3 fold 중 2개 이상 상승(stage3.adopt_3_0b). 끝 판 3-0b-<구조 자리>-end = 채택한 후보 판
     (아니면 1등) = 새 현재 최고 판(교체 후보 여부 물려받음). 트리 대상 판 = 끝 판이 교체 후보가 아닌 트리·앙상블이면 끝 판.
  4. 단계 4 가 읽는 사슬 상태(lst/stage4 꼴) logs/chain_state_3-0b.json.

옛 순서 3-0b(3-0 CV 1위 구조 — 서버 code_frozen/s3_20261109)의 결과 이어받기(`reuse_check`): 1등이 그 판(3-0 판, 이상 점수·
결합 없음)과 같고, 옛 중요도의 판·M·perm_seed·대상 그룹 목록이 지금 규칙과 같고, 옛 결과 폴더 코드 사본(_code/lst)의 계산
경로 모듈이 지금 코드와 바이트 단위로 같을 때만 — 그 중요도 표·후보 판(3-0b-<구조>, 설정 md5 대조)을 그대로 씀.

구현 세부(계획서에 없음 — 담당자 확인):
 (a) 판 이름: 후보 = '3-0b-<구조 자리>', 끝 판 = '3-0b-<구조 자리>-end'(구조 자리 = 1등이 3-0 판이면 구조 약어, 그 밖이면 1등
     판 이름 — 부록 A-14), 앙상블 구성 후보 = '3-0b-<구조 자리>--<알고리즘>'(A-14 에 없음, lst/stage4 앙상블 구성 이름 꼴).
     1등이 3-0 판이면 후보 판 설정 = stage3.spec_3_0b 그대로(옛 3-0b 와 같은 설정 md5 — 이어받기 대조용, '바탕 판' tag 는 판정 때).
 (b) 그 밖 후보 판 설정 = 1등(구성) 설정 그대로 + version.exclude_groups(그 판 입력에 있는 뺄 그룹만), stage '3-0b',
     이상 점수 칸(extra.anomaly)은 그대로(같은 3-1-VAEt 값), params 의 max_iter 는 지우고 부스팅이면 iter_select 켬.
 (c) 중요도 섞기 전 점수 대조 허용 오차 = 구성이 모두 트리면 1e-9, 딥러닝이 있으면 1e-6(stage3 구현 세부 g).
 (d) 앙상블 후보 판 = 3-3 과 같은 평균(stage3_combine.ensemble_preds — CV = 교차 N 평균, threshold = 공통 N 평균), 판 폴더 꼴
     (kind 'ensemble')도 3-3-T3 와 같음(백분위 평균 대조·구성 분위수 포함). 학습 횟수 0.
 (e) 끝 판 사슬 상태 BestDef: 이름 = 끝 판 이름, dir = 예측이 있는 원 판 폴더(채택한 후보 판 또는 1등 판 폴더).
 (f) 이어받기 코드 대조 = lst/*.py 중 이 단계 뒤에 고친 모듈(stage3·stage3_combine·stage3b·stage4)을 뺀 전부 바이트 비교 +
     stage3.py 의 계산 경로 함수(중요도 노드·후보 설정) 원문 비교. 결과 바이트 비교(accept_previous_code)는 트리 묶음만 되고
     딥러닝 판을 다시 학습해야 해 쓰지 않음.
 (g) 중요도 나눠 돌리기(설정 importance_parts — scripts/s3_0b 머리): (fold, h 구간) 부분마다 importance.importance_part,
     모두 끝나면 importance.merge_parts → finish_importance(한 번에 돌린 것과 같은 records·summary·meta, seconds 만 다름).
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cf
from . import evaluate as ev
from . import registry as rg
from . import runner as rn
from . import stage3 as s3
from . import stage3_combine as c3
from .paths import resolve

STAGE = s3.STAGE_3_0B
ENS_SEP = "--"
TARGET_FILE = s3.TARGET_FILE                 # 3-0b_target.json
IMPORTANCE_DIR = s3.IMPORTANCE_DIR           # 3-0b_importance
DECISION_FILE = s3.DECISION_FILE             # 3-0b_decision.json
REUSE_FILE = "3-0b_reuse.json"
QUANTILE_FILE = "member_quantiles.csv"
PCT_FILE = "pct_cv_scores.json"
CODE_SKIP = ("stage3.py", "stage3_combine.py", "stage3b.py", "stage4.py")
STAGE3_FUNCS = ("full_version", "spec_3_0", "specs_3_0", "_check_specs", "spec_3_0b", "drop_groups_from_summary",
                "dedupe_predict_fn", "importance_nodes", "base_tol")


class StageError(RuntimeError):
    pass


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ================================================================ 1등
def member_spec(inputs_root: Path, m: Mapping) -> rn.VariantSpec:
    return rn.VariantSpec(**rn.read_json(resolve(inputs_root, m["dir"]) / "spec.json"))


def is_3_0_board(spec: rn.VariantSpec) -> bool:
    return spec.stage == "3-0" and spec.name == f"3-0-{spec.algorithm}"


def part_of(best: Mapping, inputs_root: Path) -> str:
    """부록 A-14 구조 자리: 1등이 3-0 판이면 구조 약어, 그 밖(3-1·3-2 판·앙상블·스태킹·교체 후보)이면 1등 판 이름."""
    if best["kind"] in ("tree", "dl") and not best["replacement"]:
        sp = member_spec(inputs_root, best["members"][0])
        if is_3_0_board(sp) and not (sp.extra or {}).get("anomaly"):
            return sp.algorithm
    return best["name"]


def target_info(best: Mapping, inputs_root: Path) -> dict:
    d = resolve(inputs_root, best["dir"])
    spec = rn.read_json(d / "spec.json")
    _, done, cv = rn.load_done(d.parent, d.name)
    part = part_of(best, inputs_root)
    mem = []
    if best["kind"] != "stack4":
        for m in best["members"]:
            sp = member_spec(inputs_root, m)
            mem.append({"name": m["name"], "dir": m["dir"], "algorithm": sp.algorithm, "kind": sp.kind,
                        "spec_name": sp.name, "anomaly": bool((sp.extra or {}).get("anomaly"))})
    return {"name": best["name"], "kind": best["kind"], "dir": best["dir"], "replacement": bool(best["replacement"]),
            "part": part, "candidate": f"3-0b-{part}", "end": f"3-0b-{part}{s3.END_SUFFIX}",
            "cv": float(cv["cv_band_weighted"]), "fold": dict(cv["fold_band_weighted"]),
            "train_count": int(spec["train_count"]), "many_ties": bool(cv["ties"]["many_ties"]), "members": mem,
            "best_def": copy.deepcopy(dict(best))}


def version_groups(spec: rn.VariantSpec) -> set[str]:
    """판 변수 묶음에 단위가 있는 부록 A-5 그룹 이름."""
    from . import wiring as wr

    return {rg.group(u.group_no).name for u in wr.version_spec(spec).units()}


# ================================================================ 중요도 노드(1등 전체)
def anomaly_nodes(ctx, samples: ev.SampleTable, vdir: Path, spec: rn.VariantSpec, inputs_root: Path) -> dict:
    """이상 점수 칼럼이 있는 트리 판: 조립 객체 = stage3_combine.AnomalyInputs(트리 입력 + 3-1-VAEt 저장 값 — VAE 입력은 안
    섞음), fold 모델 다시 읽음(재학습 없음), 부스팅 = 교차 N. 칼럼 소속 = 등록부(이상 점수 칼럼은 단위 없음)."""
    from . import folds as fd
    from . import importance as im
    from . import inputs as ip
    from . import models_tree as mtr
    from . import wiring as wr

    if spec.kind != "tree":
        raise StageError(f"{spec.name}: 이상 점수 칼럼은 트리 판에만")
    a = spec.extra["anomaly"]
    look = c3.AnomalyLookup(resolve(inputs_root, a["dir"]) / c3.VAE_T, samples)
    vs = wr.version_spec(spec)
    lk = im.column_lookup(rg.tree_columns(vs))
    nit = rn.load_n_iter(vdir)
    if rn.is_chain_boost(spec) and nit is None:
        raise StageError(f"{spec.name}: 부스팅 판인데 n_iter.json 없음(교차 N 필요)")
    known = samples.known_frame()
    nodes = {}
    for f in ev.CV_FOLDS:
        if spec.algorithm == "LR":
            rows = rn.add_bounds(fd.make_train_rows(known, fd.SPLITS[f]), spec.W)
            inner = ip.TreeInputs(ctx, vs, f, scale=True).fit(rows)
        else:
            inner = ip.TreeInputs(ctx, vs, f)
        asm = c3.AnomalyInputs(inner, look, f)
        saved = rn.read_json(vdir / f"columns_{f}.json")["columns"]
        if list(asm.columns) != list(saved):
            raise StageError(f"{spec.name} {f}: 저장 칼럼과 지금 조립 칼럼이 다름")
        model = mtr.TreeModel.load(vdir / f"model_{f}")
        n_iter = int(nit["selection"]["cross_n"][f]) if nit is not None else None
        fn = s3.dedupe_predict_fn(model, n_iter) if spec.algorithm in rn.ROW_DEDUPE_ALGS else None
        nodes[f] = im.Leaf(f"{spec.name}-{f}", asm, model, n_iter=n_iter, predict_fn=fn,
                           layout=im.layout_from_names(list(asm.columns), lk))
    return nodes


def member_nodes(ctx, samples: ev.SampleTable, inputs_root: Path, m: Mapping, device: str | None = None) -> dict:
    spec = member_spec(inputs_root, m)
    vdir = resolve(inputs_root, m["dir"])
    if (spec.extra or {}).get("anomaly"):
        return anomaly_nodes(ctx, samples, vdir, spec, inputs_root)
    return s3.importance_nodes(ctx, samples, vdir, spec, device=device)


def importance_graph(best: Mapping, *, inputs_root: Path, member_nodes_fn: Callable[[Mapping], dict]) -> dict | None:
    """{fold: Node} — 트리·딥러닝 단독 = 구성 노드 그대로, 앙상블 = importance.Mean(구성 평균 — 3-3 과 같은 단순 평균).
    ④(stack4) = None(구성 트리 입력이 범주 5 + h·g 뿐 — 빼는 대상 그룹 없음)."""
    from . import importance as im

    if best["kind"] == "stack4":
        return None
    if best["kind"] not in ("tree", "dl", "ensemble"):
        raise StageError(f"3-0b 1등 종류를 중요도 노드로 못 만듦: {best['kind']}")
    per = [member_nodes_fn(m) for m in best["members"]]
    if len(per) == 1:
        return per[0]
    return {f: im.Mean(f"{best['name']}-{f}", [p[f] for p in per]) for f in ev.CV_FOLDS}


def graph_tol(best: Mapping, inputs_root: Path) -> float:
    kinds = {member_spec(inputs_root, m).kind for m in best["members"]}
    return 1e-9 if kinds == {"tree"} else 1e-6


def run_importance(nodes: Mapping, samples: ev.SampleTable, *, M: int, expect_base: Mapping[str, float],
                   base_tol: float, check=None):
    from . import importance as im

    targets = importance_targets(nodes)
    if not targets:
        return None
    res = im.permutation_importance(nodes, samples, targets, M=M, check=check, expect_base=expect_base,
                                    base_tol=base_tol)
    return finish_importance(res)


def importance_targets(nodes: Mapping) -> list:
    """3-0b 대상 = 1등 입력에 있는 부록 A-5 그룹(run_importance 와 같음)."""
    from . import importance as im

    return im.targets_groups(next(iter(nodes.values())).present_units())


def finish_importance(res):
    """중요도 결과에 '뺌 여부'·빼는 그룹·규칙 기록(한 번에 돌린 것·나눠 돌려 합친 것 공통)."""
    g = res.summary["target_kind"] == "group"
    res.summary["drop"] = g & (res.summary["mean_I"].astype(float) <= 0)        # 부록 A-5 '뺌 여부'
    res.meta.update({"rule": "빼는 그룹 = 3M 칸 평균 I ≤ 0(반올림 없이, 부록 A-5)",
                     "drop_groups": s3.drop_groups_from_summary(res.summary),
                     "note": "다른 그룹의 비슷한 칼럼(기온 ↔ 지중온도, 1년·5년 검출률 ↔ WOE)은 서로 메워 중요도가 낮게 나옴"})
    return res


# ================================================================ 후보 판
def reduced_spec(src: rn.VariantSpec, *, name: str, base_name: str, groups: Sequence[str]) -> rn.VariantSpec:
    """구현 세부 (b)."""
    ver = dict(src.extra["version"])
    ex = list(ver.get("exclude_groups") or [])
    ver["exclude_groups"] = sorted(set(ex) | set(groups), key=lambda g: rg.group(g).no)
    extra = {"version": ver}
    if (src.extra or {}).get("anomaly"):
        extra["anomaly"] = dict(src.extra["anomaly"])
    p = dict(src.params)
    p.pop("max_iter", None)
    if src.kind == "tree" and src.algorithm in rn.BOOST_ALGS:
        p[rn.ITER_SELECT] = True
    sp = rn.VariantSpec(name=name, stage=STAGE, order=s3.ORDER, kind=src.kind, algorithm=src.algorithm, W=src.W,
                        mode=src.mode, input_set=name, role=s3.ROLE_CAND, train_count=4, params=p, judgement=None,
                        base_variant=base_name, extra=extra)
    s3._check_specs([sp])
    return sp


def _left_units(spec: rn.VariantSpec) -> int:
    from . import wiring as wr

    return len(wr.version_spec(spec).units())


def candidate_plan(info: Mapping, drop: Sequence[str], inputs_root: Path, out_rel: str) -> dict | None:
    """후보 판 계획(None = 학습 없음). train = runner 로 돌릴 판, ensemble = 앙상블 후보 판 설정(dict)·구성,
    candidate_dir = 후보 판 예측 폴더(--inputs 아래)."""
    if info["kind"] == "stack4" or not drop:
        return None
    best = info["best_def"]
    out_rel = out_rel.rstrip("/")
    cand = info["candidate"]
    if info["kind"] in ("tree", "dl"):
        src = member_spec(inputs_root, best["members"][0])
        groups = [g for g in drop if g in version_groups(src)]
        if not groups:
            return None
        if is_3_0_board(src) and not (src.extra or {}).get("anomaly") and info["part"] == src.algorithm:
            sp = s3.spec_3_0b(src, groups)                       # 구현 세부 (a) — 옛 3-0b 와 같은 설정
        else:
            sp = reduced_spec(src, name=cand, base_name=info["name"], groups=groups)
        if sp.name != cand:
            raise StageError(f"후보 판 이름 {sp.name} ≠ {cand}")
        if sp.kind == "dl" and _left_units(sp) == 0:
            raise StageError(f"{sp.name}: 그룹을 빼면 딥러닝 입력 단위가 하나도 안 남음 — 지휘자 결정")
        return {"train": [sp], "ensemble": None, "members": None, "groups": {sp.name: groups},
                "candidate": cand, "candidate_dir": f"{out_rel}/{cand}"}
    mem, train, gmap = [], [], {}
    for m in best["members"]:
        src = member_spec(inputs_root, m)
        groups = [g for g in drop if g in version_groups(src)]
        if groups:
            sp = reduced_spec(src, name=f"{cand}{ENS_SEP}{src.algorithm}", base_name=info["name"], groups=groups)
            train.append(sp)
            gmap[sp.name] = groups
            mem.append({"algorithm": src.algorithm, "name": sp.name, "source": None, "pred_dir": f"{out_rel}/{sp.name}",
                        "before": m["name"], "before_dir": m["dir"]})
        else:                                                    # 뺀 그룹 칼럼이 없는 구성은 그대로
            mem.append({"algorithm": src.algorithm, "name": m["name"], "source": m["name"], "pred_dir": m["dir"],
                        "before": m["name"], "before_dir": m["dir"]})
    if not train:
        return None
    ens = dict(c3.t3_spec([{k: x[k] for k in ("algorithm", "name", "source", "pred_dir")} for x in mem]))
    ens.update(name=cand, stage=STAGE, input_set="3-0b 구성 (시료, h) 확률 단순 평균", base_variant=info["name"])
    return {"train": train, "ensemble": ens, "members": mem, "groups": gmap, "candidate": cand,
            "candidate_dir": f"{out_rel}/{cand}"}


def build_ensemble_board(*, out: Path, spec: Mapping, member_dirs: Sequence[Path], samples: ev.SampleTable,
                         exp_log: rn.ExperimentLog, code_md5: str, code_root: Path, n_jobs: int, tracker=None,
                         stage_run_id: str | None = None, code_reviewed: bool = True) -> dict:
    """구현 세부 (d) — scripts/s3_3.build_t3 와 같은 계산."""
    vdir = out / spec["name"]
    if c3.begin_board(vdir, out, spec, code_md5) == "done":
        return rn.read_json(vdir / "done.json")
    smoke = any(rn.read_json(d / "done.json").get("smoke") for d in member_dirs) or None
    ens = c3.ensemble_preds(list(member_dirs))
    has_thr = any(rn.load_thr_preds(d) is not None for d in member_dirs)
    qp = vdir / QUANTILE_FILE
    ens["quantiles"].to_csv(qp, index=False, encoding="utf-8-sig", float_format="%.17g")
    pct = ev.score_cv({f: ens["pct"][f] for f in c3.CV_FOLDS}, samples,
                      thr_preds=ens["pct_thr"] if has_thr else None, require_complete=not smoke)
    pname = f"{spec['name']}-백분위"
    pct_count = exp_log.cv_scored(spec["stage"], pname)
    cf.dump_scores_json(vdir / PCT_FILE, {"name": pname, "selection_count": pct_count, "cv": pct,
                                          "rule": "부록 A-12 백분위 평균(기준 분포 = 채점하는 쌍 자신의 예측) — 대조, 기록만"})
    md5s = {f"member:{Path(d).name}:{f}": cf.file_md5(Path(d) / f)
            for d in member_dirs for f in [f"pred_{s}.parquet" for s in rn.SPLIT_ORDER]}
    mem = spec["extra"]["members"]
    done = c3.finish_board(
        vdir=vdir, spec=spec, preds=ens["pred"], thr_preds=ens["thr"] if has_thr else None, samples=samples,
        exp_log=exp_log, code_md5=code_md5, input_md5s=md5s, code_root=code_root, n_jobs=n_jobs,
        smoke={"from_members": True} if smoke else None, tracker=tracker, stage_run_id=stage_run_id,
        tags={"코드 검수 완료": "예" if code_reviewed else "아니오", "바탕 판": spec.get("base_variant") or ""},
        params={"algorithm": "T3", "members": ",".join(m["name"] for m in mem), "train_count": 0},
        artifacts=[(qp, "ensemble"), (vdir / PCT_FILE, "ensemble")],
        done_extra={"members": mem, "pct_cv_band_weighted": pct["cv_band_weighted"]})
    return done


# ================================================================ 판정·끝 판·사슬 상태
def judge(info: Mapping, cand_dir: Path | None, samples: ev.SampleTable, inputs_root: Path) -> dict:
    """채택 = CV 평균 상승(4자리) + 3 fold 중 2개 이상(stage3.adopt_3_0b). 동점 많은 판 다시 채점 기록·알림 = judge_adoption."""
    if cand_dir is None:
        return {"candidate": None, "rule": None, "judge": None, "adopted": False}
    _, cdone, ccv = rn.load_done(cand_dir.parent, cand_dir.name)
    cspec = rn.read_json(cand_dir / "spec.json")
    dir_of = {cand_dir.name: cand_dir, info["name"]: resolve(inputs_root, info["dir"])}
    with ev.selection_guard():
        j = ev.judge_adoption(
            ev.Candidate(cand_dir.name, float(ccv["cv_band_weighted"]), int(cspec["train_count"]),
                         bool(ccv["ties"]["many_ties"])),
            ev.Candidate(info["name"], float(info["cv"]), int(info["train_count"]), bool(info["many_ties"])),
            c3.rescore_from(dir_of, samples))
        rule = s3.adopt_3_0b({"cv": ccv["cv_band_weighted"], "fold": ccv["fold_band_weighted"]},
                             {"cv": info["cv"], "fold": info["fold"]})
    return {"candidate": {"name": cand_dir.name, "cv": float(ccv["cv_band_weighted"]),
                          "fold": dict(ccv["fold_band_weighted"]), "mlflow_run_id": cdone.get("mlflow_run_id")},
            "rule": rule, "judge": j, "adopted": bool(rule["adopt"])}


def end_spec(info: Mapping, src_dir: Path, src_name: str) -> dict:
    """끝 판 기록 설정(원 판 설정 꼴 + 이름·단계·역할(1등이 교체 후보면 교체 후보)·학습 횟수(1등과 같은 셈 — A-12))."""
    sd = dict(rn.read_json(src_dir / "spec.json"))
    sd.update(name=info["end"], stage=STAGE, role=s3.ROLE_REPL if info["replacement"] else s3.ROLE_CAND,
              judgement=s3.NO_JUDGE, train_count=int(info["train_count"]))
    sd["extra"] = {**dict(sd.get("extra") or {}), "source": src_name}
    return sd


def new_state(state_in: Mapping, info: Mapping, *, adopted: bool, src_rel: str, inputs_root: Path,
              line: Mapping) -> dict:
    """lst/stage4 사슬 상태 꼴 — current_best = 끝 판(구현 세부 e), 트리 대상 판 = 끝 판이 교체 후보가 아닌 트리·앙상블이면."""
    if adopted:
        nb = c3.best_def(inputs_root, {"name": info["end"], "dir": src_rel, "replacement": info["replacement"]})
    else:
        nb = copy.deepcopy(dict(info["best_def"]))
        nb["name"] = info["end"]
    st = copy.deepcopy(dict(state_in))
    st["current_best"] = nb
    if nb["kind"] in ("tree", "ensemble") and not nb["replacement"]:
        st["tree_target"] = copy.deepcopy(nb)
    st["step"] = STAGE
    st["history"] = list(st.get("history", [])) + [dict(line)]
    return st


# ================================================================ 옛 3-0b 결과 이어받기
def _func_sources(path: Path, names: Sequence[str]) -> dict[str, str | None]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {n: None for n in names}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in out:
            out[node.name] = ast.get_source_segment(src, node)
    return out


def code_same(old_code: Path, code_root: Path) -> list[str]:
    """구현 세부 (f): 다른 곳 목록(빈 목록 = 같음)."""
    diff = []
    cur = {p.name for p in (code_root / "lst").glob("*.py")} - set(CODE_SKIP)
    old = {p.name for p in (old_code / "lst").glob("*.py")} - set(CODE_SKIP)
    for n in sorted(cur | old):
        a, b = old_code / "lst" / n, code_root / "lst" / n
        if not a.exists() or not b.exists() or a.read_bytes() != b.read_bytes():
            diff.append(f"lst/{n}")
    if not (old_code / "lst" / "stage3.py").exists():
        return diff + ["lst/stage3.py"]
    fo = _func_sources(old_code / "lst" / "stage3.py", STAGE3_FUNCS)
    fn = _func_sources(code_root / "lst" / "stage3.py", STAGE3_FUNCS)
    diff += [f"lst/stage3.py:{k}" for k in STAGE3_FUNCS if fo[k] is None or fo[k] != fn[k]]
    return diff


def reuse_check(*, old_rel: str, info: Mapping, inputs_root: Path, code_root: Path, M: int,
                code_check: bool = True) -> dict:
    """{'ok': bool, 'reasons': [...], 'wait': bool, …}. ok 면 drop·importance_dir·candidate(_dir) 를 돌려줌.
    wait = 이어받을 수 있는데 옛 후보 판이 아직 안 끝남(같은 판을 두 번 학습하지 않게 멈춰야 함)."""
    from . import importance as im
    from . import wiring as wr

    old = resolve(inputs_root, old_rel)
    res = {"ok": False, "wait": False, "reasons": [], "old_dir": old_rel.rstrip("/")}
    if info["kind"] not in ("tree", "dl") or info["replacement"] or len(info["members"]) != 1:
        res["reasons"].append(f"1등 종류 {info['kind']}(교체 후보 {info['replacement']}) — 옛 3-0b 대상(3-0 판) 아님")
        return res
    m = info["best_def"]["members"][0]
    src = member_spec(inputs_root, m)
    if not is_3_0_board(src) or (src.extra or {}).get("anomaly"):
        res["reasons"].append(f"1등 {info['name']} 이 3-0 판이 아님")
        return res
    tp = old / TARGET_FILE
    if not tp.exists():
        res["reasons"].append(f"옛 대상 파일 없음: {tp}")
        return res
    t = rn.read_json(tp)
    if t.get("selected") != src.name or t.get("dir", "").rstrip("/") != m["dir"].rstrip("/"):
        res["reasons"].append(f"옛 3-0b 대상 {t.get('selected')}({t.get('dir')}) ≠ 1등 {src.name}({m['dir']})")
        return res
    imp = old / IMPORTANCE_DIR
    if not (imp / "importance_meta.json").exists():
        res["wait"] = True
        res["reasons"].append("옛 3-0b 중요도가 아직 안 끝남")
        return res
    meta = rn.read_json(imp / "importance_meta.json")
    want = [{"kind": x.kind, "no": x.no, "name": x.name, "seed_key": list(x.seed_key)}
            for x in im.targets_groups(wr.version_spec(src).units())]
    if meta.get("variant") != src.name or meta.get("M") != M or meta.get("perm_seed") != im.PERM_SEED:
        res["reasons"].append(f"옛 중요도 판·M·perm_seed 가 다름: {meta.get('variant')}·{meta.get('M')}·{meta.get('perm_seed')}")
        return res
    if meta.get("targets") != want:
        res["reasons"].append("옛 중요도 대상 그룹 목록이 지금 규칙(1등 입력에 있는 그룹)과 다름")
        return res
    if code_check:
        diff = code_same(old / "_code", code_root)
        if diff:
            res["reasons"].append(f"옛 결과 코드 사본과 지금 코드가 다름(바이트): {diff[:10]}")
            res["code_diff"] = diff
            return res
    summary = pd.read_csv(imp / "importance_summary.csv", encoding="utf-8-sig")
    drop = s3.drop_groups_from_summary(summary)
    res.update(drop=drop, importance_dir=f"{old_rel.rstrip('/')}/{IMPORTANCE_DIR}", candidate=None, candidate_dir=None)
    if drop:
        cspec = s3.spec_3_0b(src, drop)
        cdir = old / cspec.name
        if not (cdir / "done.json").exists():
            res["wait"] = True
            res["reasons"].append(f"옛 3-0b 후보 판 {cspec.name} 이 아직 안 끝남")
            return res
        if rn.read_json(cdir / "done.json")["spec_md5"] != cspec.md5():
            res["reasons"].append(f"옛 후보 판 {cspec.name} 설정 md5 가 지금 규칙의 후보 판과 다름")
            return res
        res.update(candidate=cspec.name, candidate_dir=f"{old_rel.rstrip('/')}/{cspec.name}")
    res["ok"] = True
    res["files_md5"] = {f: cf.file_md5(imp / f) for f in ("importance_summary.csv", "importance_records.csv",
                                                           "importance_meta.json") if (imp / f).exists()}
    return res


def copy_importance(src_dir: Path, dst_dir: Path) -> None:
    """이어받은 중요도 표를 이 단계 폴더에 사본으로(원본은 그대로)."""
    dst_dir.mkdir(parents=True, exist_ok=False)
    for f in ("importance_summary.csv", "importance_records.csv", "importance_meta.json"):
        if (src_dir / f).exists():
            shutil.copyfile(src_dir / f, dst_dir / f)


def drop_from_summary_file(imp_dir: Path) -> list[str]:
    s = pd.read_csv(imp_dir / "importance_summary.csv", encoding="utf-8-sig")
    return s3.drop_groups_from_summary(s)
