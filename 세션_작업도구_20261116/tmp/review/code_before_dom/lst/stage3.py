"""2차 사슬 앞부분(단계 3-0·⑥⑦·3-0b·추가 판 ⑧) 공통 — 계획서(v78) 요약 10·12·13·15·19, §6 2차 공통·단계 3-0·3-0b·추가 판,
§1-2(동점·채택), §1-3(판 이름·test 막기), 부록 A-5(3-0b 그룹·중요도)·A-8(딥러닝 입력)·A-9·A-12·A-14·A-15.

순서(v78, 2026-11-09 담당자 결정): 3-0 → ⑥·⑦ 판정 → 3-1 → 3-2(④) → 3-3 → 3-0b(1등 = 3-3 판정 뒤 현재 최고 판) → ⑧ → 단계 4.
- 앞 단계 결정 읽기: 1-3 결정(W·W_DL·트리 입력 방식·1차 베이스라인), 단계 2 준비 고르기(k·트리 N_ma·딥러닝 N_ma).
- 3-0 판 목록·순위(`three_0`·`rank_3_0`): 13 판 중 CV 1위 = 첫 현재 최고 판(출발), 트리 6종 CV 순 = 트리 대상 판(1위)·
  트리 상위 3개 — ⑥⑦ 판정(scripts/s3_extra judge)이 x67_decision.json 에 적고 3-1~3-3 이 읽음.
- 판 설정: 3-0 13 구조(변수 전체), 3-0b 후보(그룹 빼기), 추가 판 ⑥(x6-LSTMA)·⑦(x7-T·x7-D)·⑧(x8-<알고리즘>).
- 3-0b 채택 규칙(§6 3-0b): CV 평균이 오르고(4자리) 3 fold 중 2개 이상 오름(fold 마다 4자리) — `adopt_3_0b`.
- 끝 판(3-0b-<구조 자리>-end) = 원 판을 가리키는 기록 폴더(예측·점수 다시 저장 안 함 — 부록 A-15).
- 3-0b 중요도 노드(1등이 3-0 판일 때·구성 모델마다): 그 판 fold 모델(재학습 없음) + 입력 조립 객체(트리 = 다시 만듦,
  로지스틱 = fold train 으로 다시 fit, 딥러닝 = 저장 전처리기), 부스팅 = 교차 N(부록 A-5 '부스팅은 교차 N 트리 수').
  1등 전체(앙상블·이상 점수 칼럼) 노드는 lst/stage3b.

구현 세부(계획서에 없음 — 담당자 확인):
 (a) 3-0 판 입력 묶음 id = '3-0-전체'(트리·로지스틱)·'3-0-전체-DL'(딥러닝·결합) — 판 설정의 version 은 §5 ID 전부 + k·
     N_ma(트리)·N_ma(딥러닝) 셋 다(트리는 n_ma_dl 을, 딥러닝은 n_ma 를 안 씀).
 (b) 3-0 판 판정 tag = '판정 없음'(부록 A-15 '판정 없음(3-0 판·3-0b 나머지 끝 판)'), 역할 = '후보'.
 (c) 3-0b 후보 판 이름 = '3-0b-<구조>', 입력 묶음 id = '3-0b-<구조>', version.exclude_groups = 뺄 그룹 이름(그룹 번호 순).
 (d) 3-0b 의 fold 별 '오름' = fold 점수 4자리 반올림 비교(§1-2 '모든 점수 비교 = 소수 4자리 반올림 뒤 비교') — 계획서
     문구 "3 fold 중 2개 이상 오르면"(§6 3-0b).
 (e) 끝 판 폴더 = spec.json·source.json(원 판 위치·파일 md5)·cv_scores.json·h_metrics.csv 사본·done.json, 판의 학습 횟수
     4(부록 A-12 '3-0b 끝 판 = 4'), '같은 validation 으로 고른 횟수' = 원 판 값(새 CV 채점 아님).
 (f) 중요도 때 같은 입력 행은 한 번만 예측(XGB·LGB·CAT·RF·ET — runner.ROW_DEDUPE_ALGS, 값 같음): Leaf 의 predict_fn 으로.
     로지스틱·딥러닝은 모든 행 그대로.
 (g) 중요도 섞기 전 점수 대조 허용 오차: 트리 1e-9, 딥러닝 1e-6(GPU 추론 묶음 차이 — 부록 models_dl 구현 세부 8).
 (h) ⑥·⑦-D 판 설정 = §5 ID WEATHER·WX1·WX2 + n_ma_dl, dl_layout 'seq_only'(x_seq + g ÷ 50·해외 구조적 결측),
     ⑦-T = 같은 단위 + n_ma = 딥러닝 N_ma(⑦-D 의 x_seq 채널과 같은 차감), tree_layout 'seq_only'·원값(mode raw) — g 만.
 (i) 추가 판 단계 이름: ⑥⑦ = '⑥⑦⑧'(이미 끝난 x6-LSTMA·x7-T·x7-D 판 설정·MLflow 단계 run 이름 그대로 — v78 부록 A-15 는
     '⑥⑦' 이지만 끝난 판 설정 md5 가 바뀌지 않게 둠), ⑧ = '⑧'(v78 부록 A-15 "⑧ → 단계 run '⑧'(3-0b 뒤)").
     역할 ⑥⑦ = '교체 후보', ⑧ = '비교만'(판정 '해당 없음').
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import config as cf
from . import evaluate as ev
from . import metrics as mt
from . import registry as rg
from . import runner as rn
from .paths import resolve
from .windows import W_GRID

ORDER = "2차"
EXPERIMENT = "listeria-2차"
TREE6 = ("XGB", "LGB", "CAT", "RF", "ET", "LR")
DL7 = ("LSTM", "GRU", "TRF", "LSTMA", "TCN", "CMB", "CMBV")
STRUCTS13 = TREE6 + DL7
CHAIN_BOOST = ("XGB", "LGB", "CAT")
INPUT_SET_TREE = "3-0-전체"
INPUT_SET_DL = "3-0-전체-DL"
EXTRA_STAGE = "⑥⑦⑧"                          # ⑥⑦ 판(이미 끝남) 단계 이름 — 구현 세부 (i)
X8_STAGE = "⑧"
STAGE_3_0B = "3-0b"
SEQ_IDS = ("WEATHER", "WX1", "WX2")           # ⑥⑦: 기상·차감·SPI 채널 전부
ROLE_CAND, ROLE_REPL, ROLE_CMP = "후보", "교체 후보", "비교만"
NO_JUDGE, NA_JUDGE = "판정 없음", "해당 없음"
END_SUFFIX = "-end"
TARGET_FILE = "3-0b_target.json"
IMPORTANCE_DIR = "3-0b_importance"
DECISION_FILE = "3-0b_decision.json"
X67_DECISION_FILE = "x67_decision.json"
SOURCE_FILES = tuple(f"pred_{s}.parquet" for s in rn.SPLIT_ORDER) + ("cv_scores.json",)


class StageError(RuntimeError):
    pass


# ================================================================ 앞 단계 결정
def load_decision_1_3(inputs_root: Path, rel: str) -> dict:
    p = resolve(inputs_root, rel)
    if not p.exists():
        raise StageError(f"1-3 결정이 없음: {p}")
    d = cf.load_json_strict(p)
    for k in ("baseline", "W", "W_DL", "tree_mode", "A", "A_variant"):
        if k not in d:
            raise StageError(f"1-3 결정 json 에 {k} 없음")
    if d["W"] not in W_GRID or d["W_DL"] != d["W"] or d["tree_mode"] not in rn.MODES:
        raise StageError(f"1-3 W·W_DL·입력 방식 이상: {d['W']}·{d['W_DL']}·{d['tree_mode']}")
    d["_md5"] = cf.file_md5(p)
    return d


def load_selection_2(inputs_root: Path, rel: str, dec: Mapping) -> dict:
    """단계 2 준비 고르기(2_prep_selection.json): k·n_ma(트리)·n_ma_dl(딥러닝) — 1-3 결정과 같은 W 인지 확인."""
    p = resolve(inputs_root, rel)
    if not p.exists():
        raise StageError(f"단계 2 준비 고르기가 없음(s2_prep select 먼저): {p}")
    s = cf.load_json_strict(p)
    for k in ("k", "n_ma", "n_ma_dl", "W", "W_DL", "tree_mode"):
        if k not in s:
            raise StageError(f"단계 2 준비 고르기 json 에 {k} 없음")
    if (s["W"], s["W_DL"], s["tree_mode"]) != (dec["W"], dec["W_DL"], dec["tree_mode"]):
        raise StageError("단계 2 준비 고르기의 W·입력 방식이 1-3 결정과 다름")
    if s.get("decision_1_3_md5") not in (None, dec["_md5"]):
        raise StageError("단계 2 준비 고르기 때의 1-3 결정 파일과 지금 파일이 다름")
    s["_md5"] = cf.file_md5(p)
    return s


def full_version(sel: Mapping) -> dict:
    """변수 전체(§5 후보 전부 + 1차 기상) — 구현 세부 (a)."""
    return {"include_ids": list(rg.IDS), "k": int(sel["k"]), "n_ma": int(sel["n_ma"]), "n_ma_dl": int(sel["n_ma_dl"])}


# ================================================================ 판 설정
def spec_3_0(alg: str, dec: Mapping, sel: Mapping) -> rn.VariantSpec:
    if alg not in STRUCTS13:
        raise ValueError(f"3-0 구조 아님: {alg}")
    tree = alg in TREE6
    return rn.VariantSpec(
        name=f"3-0-{alg}", stage="3-0", order=ORDER, kind="tree" if tree else "dl", algorithm=alg,
        W=int(dec["W"] if tree else dec["W_DL"]), mode=dec["tree_mode"] if tree else None,
        input_set=INPUT_SET_TREE if tree else INPUT_SET_DL, role=ROLE_CAND, train_count=4,
        params={rn.ITER_SELECT: True} if alg in CHAIN_BOOST else {}, judgement=NO_JUDGE,
        extra={"version": full_version(sel)})


SUB_ALGS = {"T": TREE6, "D": DL7}


def specs_3_0(sub: str, dec: Mapping, sel: Mapping) -> list[rn.VariantSpec]:
    """3-0 판 목록: 'T' = 트리 6(XGB·LGB·CAT·RF·ET 은 같은 입력 묶음 실행, LR 혼자), 'D' = 딥러닝 5 + 결합 2(GPU)."""
    if sub not in SUB_ALGS:
        raise ValueError(f"3-0 sub 는 {tuple(SUB_ALGS)}: {sub}")
    out = [spec_3_0(a, dec, sel) for a in SUB_ALGS[sub]]
    _check_specs(out)
    return out


def _check_specs(specs: Sequence[rn.VariantSpec]) -> None:
    from . import wiring as wr

    for s in specs:                      # 등록부 규칙에 맞는지 미리(모르는 단위·k·N_ma 면 멈춤)
        wr.version_spec(s)
        rn.is_chain_boost(s)


def spec_3_0b(base: rn.VariantSpec, drop_groups: Sequence[str]) -> rn.VariantSpec:
    """3-0b 후보 판(구현 세부 c): 3-0 판 설정 그대로 + 평균 I ≤ 0 그룹을 한꺼번에 뺌."""
    if not drop_groups:
        raise ValueError("뺄 그룹 없음 — 후보 판 없음")
    groups = sorted({rg.group(g).name for g in drop_groups}, key=lambda g: rg.group(g).no)
    ver = dict(base.extra["version"])
    if ver.get("exclude_groups"):
        raise StageError("3-0 판에 이미 빠진 그룹이 있음")
    ver["exclude_groups"] = groups
    s = rn.VariantSpec(name=f"3-0b-{base.algorithm}", stage="3-0b", order=ORDER, kind=base.kind,
                       algorithm=base.algorithm, W=base.W, mode=base.mode, input_set=f"3-0b-{base.algorithm}",
                       role=ROLE_CAND, train_count=4, params=dict(base.params), judgement=None,
                       extra={**dict(base.extra), "version": ver})
    _check_specs([s])
    return s


def end_name(alg: str) -> str:
    return f"3-0b-{alg}{END_SUFFIX}"


def specs_extra(sub: str, dec: Mapping, sel: Mapping, x8_base: rn.VariantSpec | None = None) -> list[rn.VariantSpec]:
    """추가 판(§6 추가 판 표): 'G' = ⑥ x6-LSTMA·⑦-D x7-D(GPU, 같은 입력 묶음), 'C' = ⑦-T x7-T(CPU),
    'E' = ⑧ x8-<알고리즘>(CPU — 3-0b 뒤). x8_base = 3-0 트리 6종 중 CV 1위 판의 설정(⑧ 바탕, 변수 전체)."""
    W = int(dec["W_DL"])
    seq_dl = {"include_ids": list(SEQ_IDS), "n_ma_dl": int(sel["n_ma_dl"]), "dl_layout": "seq_only"}
    com = dict(stage=EXTRA_STAGE, order=ORDER, train_count=4)
    if sub == "G":
        out = [rn.VariantSpec(name="x6-LSTMA", kind="dl", algorithm="LSTMA", W=W, mode=None, input_set="x6x7D-시계열만",
                              role=ROLE_REPL, params={}, extra={"version": seq_dl}, **com),
               rn.VariantSpec(name="x7-D", kind="dl", algorithm="TRF", W=W, mode=None, input_set="x6x7D-시계열만",
                              role=ROLE_REPL, params={}, extra={"version": seq_dl}, **com)]
    elif sub == "C":
        seq_t = {"include_ids": list(SEQ_IDS), "n_ma": int(sel["n_ma_dl"]), "tree_layout": "seq_only"}
        out = [rn.VariantSpec(name="x7-T", kind="tree", algorithm="XGB", W=W, mode="raw", input_set="x7T-시계열만",
                              role=ROLE_REPL, params={rn.ITER_SELECT: True}, extra={"version": seq_t}, **com)]
    elif sub == "E":
        if x8_base is None:
            raise ValueError("⑧ 바탕 판(3-0 트리 6종 중 CV 1위) 필요")
        s8 = spec_x8(x8_base)
        out = [] if s8 is None else [s8]
    else:
        raise ValueError(f"추가 판 sub 는 G·C·E: {sub}")
    _check_specs(out)
    return out


def weather8_left(version: Mapping) -> list[str]:
    """그 판 변수 묶음에 남은 기상 8그룹(계절 제외) 이름."""
    vs = rg.load_version({"name": "_", "order": 2, "W": 10, "mode": "sum", **version})
    have = {u.group for u in vs.units()}
    return [g for g in rg.WEATHER8_GROUPS if g in have]


def spec_x8(base: rn.VariantSpec) -> rn.VariantSpec | None:
    """⑧ 정형만: 바탕 판(3-0 트리 6종 중 CV 1위)에서 기상 8그룹을 모두 뺀 판(같은 설정 — 부스팅은 반복 수만 다시).
    이미 다 빠졌으면 None(바탕이 변수 전체라 실제로는 없음)."""
    if base.kind != "tree":
        raise StageError("⑧ 바탕은 트리 판")
    ver = dict(base.extra["version"])
    left = weather8_left(ver)
    if not left:
        return None
    ex = list(ver.get("exclude_groups") or [])
    ver["exclude_groups"] = sorted(set(ex) | set(rg.WEATHER8_GROUPS), key=lambda g: rg.group(g).no)
    return rn.VariantSpec(name=f"x8-{base.algorithm}", stage=X8_STAGE, order=ORDER, kind="tree",
                          algorithm=base.algorithm, W=base.W, mode=base.mode, input_set=f"x8-{base.algorithm}-정형만",
                          role=ROLE_CMP, train_count=4, params=dict(base.params), judgement=NA_JUDGE,
                          base_variant=None, extra={**dict(base.extra), "version": ver})


# ================================================================ 3-0b 채택 규칙(§6 3-0b)
def adopt_3_0b(cand: Mapping, base: Mapping) -> dict:
    """cand·base = {'cv': CV 평균, 'fold': {fold: 점수}}. 채택 = CV 평균 상승(4자리, 같으면 상승 아님) + 3 fold 중 2개 이상
    상승(fold 마다 4자리 — 구현 세부 d)."""
    up = {f: mt.is_improvement(float(cand["fold"][f]), float(base["fold"][f])) for f in ev.CV_FOLDS}
    cv_up = mt.is_improvement(float(cand["cv"]), float(base["cv"]))
    n_up = sum(up.values())
    return {"cv_up": cv_up, "fold_up": up, "n_fold_up": n_up, "adopt": bool(cv_up and n_up >= 2),
            "cand_cv": float(cand["cv"]), "base_cv": float(base["cv"])}


def drop_groups_from_summary(summary) -> list[str]:
    """빼는 그룹 = 3M 칸 평균 I ≤ 0(반올림 없이 — 부록 A-5)."""
    s = summary[summary["target_kind"] == "group"]
    out = s.loc[s["mean_I"].astype(float) <= 0, "target"].tolist()
    return sorted(out, key=lambda g: rg.group(g).no)


# ================================================================ 3-0 판 목록·순위(⑥⑦ 판정·⑧·3-0b 재사용 확인이 씀)
def three_0(inputs_root: Path, decision_1_3: str, selection_2: str, dirs: Mapping[str, str]) -> dict[str, dict]:
    """3-0 13 판 {구조: {spec, root(판 폴더 부모), rel(--inputs 아래 판 폴더)}} — 판 목록을 결정·고르기 파일에서 다시 만들어
    끝난 판의 설정 md5 와 대조(끝나지 않았거나 설정이 다르면 멈춤). dirs = {'3-0-T': …, '3-0-D': …}."""
    dec = load_decision_1_3(inputs_root, decision_1_3)
    sel = load_selection_2(inputs_root, selection_2, dec)
    out = {}
    for sub in ("T", "D"):
        rel = dirs[f"3-0-{sub}"]
        root = resolve(inputs_root, rel)
        for sp in specs_3_0(sub, dec, sel):
            spec, done, _ = rn.load_done(root, sp.name)
            if done["spec_md5"] != sp.md5():
                raise StageError(f"{sp.name}: 끝난 3-0 판 설정이 지금 판 목록과 다름")
            out[sp.algorithm] = {"spec": sp, "root": root, "rel": f"{rel.rstrip('/')}/{sp.name}"}
    if tuple(out) != STRUCTS13:
        raise StageError(f"3-0 구조 13 이 아님: {list(out)}")
    return out


def rank_3_0(t3: Mapping[str, Mapping], samples: ev.SampleTable | None = None) -> dict:
    """3-0 순위(§1-2: 4자리 → 판의 학습 횟수 → 이름, CV 만): first = 13 판 중 CV 1위(첫 현재 최고 판 — samples 를 주면
    동점 많은 판 다시 채점 기록·알림 = evaluate.select_with_tie_check), tree_rank = 트리 6종 CV 순(1위 = 트리 대상 판,
    앞 3 = 트리 상위 3개 — 3-2·3-3), boards = 판 정보(구조·종류·폴더·CV·fold 점수·변수 묶음)."""
    cands, boards = [], {}
    for alg, v in t3.items():
        _, done, cv = rn.load_done(v["root"], v["spec"].name)
        c = ev.Candidate(v["spec"].name, float(cv["cv_band_weighted"]), int(v["spec"].train_count),
                         bool(cv["ties"]["many_ties"]), {"algorithm": alg, "kind": v["spec"].kind})
        cands.append(c)
        boards[c.name] = {"structure": alg, "kind": v["spec"].kind, "dir": v["rel"], "cv": c.cv,
                          "fold": dict(cv["fold_band_weighted"]), "version": dict(v["spec"].extra["version"]),
                          "train_count": c.train_count, "many_ties": c.many_ties}
    with ev.selection_guard():
        if samples is not None:
            dir_of = {v["spec"].name: v["root"] / v["spec"].name for v in t3.values()}

            def rescore(n):
                return float(ev.score_cv(rn.load_preds(dir_of[n], ev.CV_FOLDS), samples,
                                         rescore=True)["cv_band_weighted"])
            sel = ev.select_with_tie_check(cands, rescore)
        else:
            sel = {"selected": ev.rank_candidates(cands)[0].name, "alerts": []}
        trees = [c.name for c in ev.rank_candidates([c for c in cands if c.extra["algorithm"] in TREE6])]
    return {"first": sel["selected"], "selection": sel, "tree_rank": trees, "tree_top3": trees[:3],
            "tree_target": trees[0], "boards": boards}


# ================================================================ 판 폴더 읽기
def candidate_of(vdir_root: Path, name: str, train_count: int | None = None) -> ev.Candidate:
    spec, done, cv = rn.load_done(vdir_root, name)
    return ev.Candidate(name, float(cv["cv_band_weighted"]), int(train_count or spec["train_count"]),
                        bool(cv["ties"]["many_ties"]), {"algorithm": spec["algorithm"], "kind": spec["kind"]})


def files_md5(vdir: Path) -> dict[str, str]:
    miss = [f for f in SOURCE_FILES if not (vdir / f).exists()]
    if miss:
        raise StageError(f"원 판 파일 없음: {vdir.name} {miss}")
    thr = [f"pred_{rn.THR_PREFIX}{f}.parquet" for f in ev.CV_FOLDS]
    return {f: cf.file_md5(vdir / f) for f in SOURCE_FILES + tuple(t for t in thr if (vdir / t).exists())}


def _spec_dict_md5(spec) -> tuple[dict, str]:
    """VariantSpec 또는 판 설정 dict(앙상블·VAE 판 꼴) → (dict, md5) — md5 는 VariantSpec.md5 와 같은 식."""
    import hashlib
    import json

    d = spec.to_dict() if isinstance(spec, rn.VariantSpec) else dict(spec)
    return d, hashlib.md5(json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def write_pointer(out_dir: Path, spec, src_dir: Path, src_rel: str, code_md5: str,
                  mlflow_run_id: str | None = None) -> dict:
    """끝 판 기록 폴더(구현 세부 e) — 예측·점수는 원 판을 가리킴. 이미 끝났으면(같은 설정·같은 원 파일) 그대로.
    spec = VariantSpec 또는 판 설정 dict(3-0b 끝 판의 원 판이 앙상블일 때 — kind 'ensemble')."""
    sd, smd5 = _spec_dict_md5(spec)
    vdir = rn.variant_dir(out_dir, sd["name"])
    md5s = files_md5(src_dir)
    src_done = rn.read_json(src_dir / "done.json")
    if (vdir / "done.json").exists():
        d = rn.read_json(vdir / "done.json")
        if d["spec_md5"] != smd5 or rn.read_json(vdir / "source.json")["files_md5"] != md5s:
            raise rn.ResumeError(f"{sd['name']}: 끝난 끝 판 기록의 설정·원 파일이 지금과 다름")
        return d
    if vdir.exists():
        rn._move_partial(vdir, out_dir, reason="멈춘 끝 판 기록 다시")
    vdir.mkdir(parents=True)
    cf.dump_json(vdir / "spec.json", sd)
    cf.dump_json(vdir / "source.json", {"variant": src_dir.name, "dir": src_rel, "files_md5": md5s,
                                        "mlflow_run_id": src_done.get("mlflow_run_id"),
                                        "pred_location": {s: f"{src_rel}/pred_{s}.parquet" for s in rn.SPLIT_ORDER}})
    shutil.copyfile(src_dir / "cv_scores.json", vdir / "cv_scores.json")
    if (src_dir / rn.H_METRICS_FILE).exists():
        shutil.copyfile(src_dir / rn.H_METRICS_FILE, vdir / rn.H_METRICS_FILE)
    cv = rn._load_scores(vdir / "cv_scores.json")
    done = {"name": sd["name"], "stage": sd["stage"], "spec_md5": smd5, "code_md5": code_md5,
            "cv_band_weighted": cv["cv_band_weighted"], "many_ties": cv["ties"]["many_ties"],
            "selection_count": src_done["selection_count"], "mlflow_run_id": mlflow_run_id, "copied_from": src_rel,
            "n_iter": src_done.get("n_iter")}
    cf.dump_scores_json(vdir / "done.json", done)
    return done


def base_row_from(src_dir: Path, row_name: str, desc: str, run_id: str | None = None) -> rn.BaseRow:
    done = rn.read_json(src_dir / "done.json")
    import pandas as pd

    return rn.BaseRow(row_name, desc, rn.load_preds(src_dir, ev.CV_FOLDS),
                      pd.read_parquet(rn.pred_path(src_dir, "refit")), int(done["selection_count"]), run_id,
                      thr_preds=rn.load_thr_preds(src_dir))


def baseline_1_row(inputs_root: Path, dec: Mapping, dir_1_1: str, dir_1_2: str) -> rn.BaseRow:
    name = dec["baseline"]
    d = resolve(inputs_root, dir_1_2 if name.startswith("1-LSTM-") else dir_1_1) / name
    return base_row_from(d, name, "1차 베이스라인(1차 70 설정 중 CV 1위) — 비교 줄")


def baseline_1_cv(inputs_root: Path, dec: Mapping, dir_1_1: str, dir_1_2: str) -> float:
    name = dec["baseline"]
    d = resolve(inputs_root, dir_1_2 if name.startswith("1-LSTM-") else dir_1_1)
    return float(rn.load_done(d, name)[2]["cv_band_weighted"])


# ================================================================ 3-0b 중요도 노드(부록 A-5)
def dedupe_predict_fn(model, n_iter: int | None):
    """같은 입력 행은 한 번만 예측(구현 세부 f — ROW_DEDUPE_ALGS 만, 값 같음)."""
    def f(X):
        rep, inv = rn.unique_rows(X)
        Xu = rn.take_rows(X, rep) if rep.size < len(X) else X
        p = model.predict_proba(Xu, n_iter=n_iter) if n_iter is not None else model.predict_proba(Xu)
        p = np.asarray(p, dtype=np.float64)
        return p[inv] if rep.size < len(X) else p
    return f


def importance_nodes(ctx, samples: ev.SampleTable, vdir: Path, spec: rn.VariantSpec, *, device: str | None = None
                     ) -> dict:
    """그 판 fold 모델 3개(저장 모델 다시 읽음 — 재학습 없음) → importance.Leaf. 부스팅 n_iter = 교차 N(fold 마다)."""
    from . import folds as fd
    from . import importance as im
    from . import inputs as ip
    from . import wiring as wr

    vs = wr.version_spec(spec)
    nit = rn.load_n_iter(vdir)
    if rn.is_chain_boost(spec) and nit is None:
        raise StageError(f"{spec.name}: 부스팅 판인데 n_iter.json 없음(교차 N 필요)")
    known = samples.known_frame()
    nodes = {}
    for f in ev.CV_FOLDS:
        if spec.kind == "tree":
            from . import models_tree as mtr

            if spec.algorithm == "LR":
                rows = rn.add_bounds(fd.make_train_rows(known, fd.SPLITS[f]), spec.W)
                asm = ip.TreeInputs(ctx, vs, f, scale=True).fit(rows)   # 표준화 통계 = 그 fold train(저장 값과 같음)
            else:
                asm = ip.TreeInputs(ctx, vs, f)
            saved = rn.read_json(vdir / f"columns_{f}.json")["columns"]
            if list(asm.columns) != list(saved):
                raise StageError(f"{spec.name} {f}: 저장 칼럼과 지금 조립 칼럼이 다름")
            model = mtr.TreeModel.load(vdir / f"model_{f}")
            n_iter = int(nit["selection"]["cross_n"][f]) if nit is not None else None
            fn = dedupe_predict_fn(model, n_iter) if spec.algorithm in rn.ROW_DEDUPE_ALGS else None
            nodes[f] = im.Leaf(f"{spec.name}-{f}", asm, model, n_iter=n_iter, predict_fn=fn)
        else:
            from . import models_dl as md

            asm = ip.DLInputs(ctx, vs, f)
            asm.prep = ip.DLPreprocessor.load(vdir / f"inputs_{f}" / "dl_preprocessor.json")
            saved = rn.read_json(vdir / f"columns_{f}.json")["columns"]
            if list(asm.columns) != list(saved):
                raise StageError(f"{spec.name} {f}: 저장 칼럼과 지금 전처리기 칼럼이 다름")
            nodes[f] = im.Leaf(f"{spec.name}-{f}", asm, md.DLClassifier.load(vdir / f"model_{f}", device=device))
    return nodes


def base_tol(spec: rn.VariantSpec) -> float:
    return 1e-9 if spec.kind == "tree" else 1e-6     # 구현 세부 (g)
