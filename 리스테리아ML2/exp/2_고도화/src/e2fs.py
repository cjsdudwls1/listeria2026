# -*- coding: utf-8 -*-
"""양방향 단계 선택 (사용자 결정 2026-10-19 [종합] 전달, 100일 위 2차 실험 A단계) - run.py 단계 'fs'(전진)·'bw'(뒤로).

판단 점수 = 검증 T 통합 60일평균 PR-AUC 시드 42·43·44 평균(e2eval.val_tint_pr_auc, e2tables.tint_seed_scores).
CatBoost = E00 설정 그대로(e2models.CatRunner), k = k*(e2tables.select_k). 묶음 정의 = e2cfg.FS_BUNDLES.
전진: 기준 = E00 에서 시작. 묶음 순서대로 후보 FSnn = 지금 기준 + 묶음 하나(시드 3) -> 후보 점수 > 기준 점수면 남김(새 기준 = 후보), 아니면 뺌.
뒤로: 전진 최종 기준에서 남긴 묶음을 하나씩 뺀 후보 BWnn(차례 번호, 라운드를 넘어 이어짐, 후보 순서 = 묶음 순서) -> 기준보다 점수가 오른 후보가
      있으면 가장 많이 오른 것(같으면 앞 순서)을 빼고(새 기준) 남은 묶음으로 다시, 오른 후보가 없으면 끝.
'ae4'(AE·VAE 점수 4칸): 표 AE·VAE 입력 = FS12 시점 기준 모델 입력(숫자 칸 + 범주 임베딩, e2ae 캐시는 입력 칸 목록 md5 로 구분),
      시퀀스 AE·VAE = L일 기상 시퀀스. 뒤로 단계는 FS12 때 계산한 점수를 고정 묶음으로 씀(입력 'ae_tab' 을 그대로 두어 같은 캐시를 읽음 - 다시 계산 안 함).
결정은 저장된 결과만으로 다시 계산(재생) -> 끝난 실행은 건너뛰고 이어하기 해도 같은 결정. 결정 기록 = results/selection_FS.json.
"""
import time

import numpy as np

import e2cfg as E

ORDER = [b["code"] for b in E.FS_BUNDLES]
RULE = "T 통합 검증 60일평균 PR-AUC(시드 42·43·44 평균), 후보 점수 > 기준 점수면 남김"
AE_FIXED_NOTE = ("FS12(ae4) 의 AE·VAE 점수는 FS12 시점에 계산한 값(표 AE·VAE 입력 = FS12 기준 모델 입력)을 고정 묶음으로 취급 - "
                 "뒤로 단계에서 다른 묶음을 빼도 다시 계산하지 않음(입력 'ae_tab' 그대로, 같은 점수 캐시)")


class NotReady(Exception):
    """재생 중 결과가 아직 없는 후보를 만남 -> exp 를 실행해야 함."""

    def __init__(self, exp):
        super().__init__(exp["id"])
        self.exp = exp


def sort_codes(codes):
    return sorted(set(codes), key=ORDER.index)


def feats_of(codes, k_star, ae_tab=None):
    """E00 입력 + 묶음들(묶음 순서로). k = k*(이력·제품명·AE 묶음이 있을 때). 'ae4' 가 있으면 ae_tab(표 AE 입력, 해석된 입력 dict) 필요."""
    codes = sort_codes(codes)
    f = dict(E.EXP_BY_ID["E00"]["feats"])
    for c in codes:
        for key, val in E.FS_BY_CODE[c]["add"].items():
            f[key] = (list(f.get(key) or []) + list(val)) if isinstance(val, list) else val
    if any(c in E.FS_K_NEED for c in codes):
        f["k"] = int(k_star)
    if "ae4" in codes:
        if ae_tab is None:
            raise ValueError("ae4 묶음에 표 AE 입력(ae_tab)이 없음")
        f["ae_tab"] = dict(ae_tab)
    return f


def bundle_text(code):
    b = E.FS_BY_CODE[code]
    nm = b["name"] if b["name"].endswith(f"{b['n']}칸") else f"{b['name']} {b['n']}칸"      # '토양 중앙값 3칸' 처럼 이름에 칸 수가 있으면 다시 붙이지 않음
    return f"{nm}({b['desc']})"


def make_exp(cid, phase, base_id, codes, changed, k_star, ae_tab=None):
    """후보 실험 dict(run.py run_one 이 쓰는 칸). phase = 'forward'(더함) | 'backward'(뺌)."""
    fwd = phase == "forward"
    desc = f"{cid} = {base_id} {'+' if fwd else '−'} {bundle_text(changed)}"
    if changed == "ae4" or ("ae4" in codes and not fwd):
        desc += " [AE·VAE 점수: FS12 시점 기준 모델 입력으로 계산한 고정 묶음]" if not fwd else " [표 AE·VAE 입력 = 기준 모델 입력]"
    return dict(id=cid, stage="fs" if fwd else "bw", plan=E.FS_PLAN[phase], model="cat",
                feats=feats_of(codes, k_star, ae_tab if "ae4" in codes else None), seeds=list(E.SEEDS),
                fs_info={"phase": phase, "base_id": base_id, "bundle": changed, "action": "add" if fwd else "remove",
                         "bundles": sort_codes(codes), "bundle_name": E.FS_BY_CODE[changed]["name"], "n_cols": E.FS_BY_CODE[changed]["n"]},
                desc=desc)


def _step(phase, step, rnd, code, cid, base_id, base_score, cand_score, decision, new_base_id, by_seed=None):
    ko = {"keep": "남김" if phase == "forward" else "그대로 둠", "drop": "뺌", "remove": "뺌"}[decision]
    d = {"phase": phase, "step": step, "round": rnd, "bundle": code, "bundle_name": E.FS_BY_CODE[code]["name"],
         "n_cols": E.FS_BY_CODE[code]["n"], "candidate_id": cid, "base_id": base_id, "base_score": base_score,
         "candidate_score": cand_score, "decision": decision, "decision_ko": ko, "new_base_id": new_base_id}
    if by_seed is not None:
        d["candidate_score_by_seed"] = by_seed
    return d


def forward(score_of, k_star, steps):
    """전진 재생. score_of(exp) -> (평균 점수, 시드별) 또는 None(결과 없음 -> NotReady). steps 에 결정을 차례로 붙임.
    돌려줌: 최종 기준 {'id', 'codes', 'score', 'ae_tab'}."""
    e00 = E.EXP_BY_ID["E00"]
    sc = score_of(e00)
    if sc is None:
        raise NotReady(e00)
    base_id, codes, base_score, ae_tab = "E00", [], sc[0], None
    for i, b in enumerate(E.FS_BUNDLES, 1):
        cid = f"FS{i:02d}"
        tab = feats_of(codes, k_star) if b["code"] == "ae4" else None      # 그 시점 기준 모델 입력(AE 없음)
        exp = make_exp(cid, "forward", base_id, codes + [b["code"]], b["code"], k_star, tab)
        r = score_of(exp)
        if r is None:
            raise NotReady(exp)
        keep = r[0] > base_score
        new_id = cid if keep else base_id
        steps.append(_step("forward", i, None, b["code"], cid, base_id, base_score, r[0], "keep" if keep else "drop", new_id, r[1]))
        if keep:
            base_id, codes, base_score = cid, codes + [b["code"]], r[0]
            if b["code"] == "ae4":
                ae_tab = tab
    return {"id": base_id, "codes": sort_codes(codes), "score": base_score, "ae_tab": ae_tab}


def backward(score_of, fw, k_star, steps):
    """뒤로 재생(전진 최종 기준 fw 에서). 돌려줌: 최종 {'id', 'codes', 'score', 'ae_tab'}."""
    base_id, codes, base_score, ae_tab = fw["id"], list(fw["codes"]), fw["score"], fw.get("ae_tab")
    n, rnd = 0, 0
    while codes:
        rnd += 1
        cands = []
        for c in codes:
            n += 1
            cid = f"BW{n:02d}"
            rest = [x for x in codes if x != c]
            exp = make_exp(cid, "backward", base_id, rest, c, k_star, ae_tab)
            r = score_of(exp)
            if r is None:
                raise NotReady(exp)
            cands.append((c, cid, r[0], r[1]))
        best = None
        for c, cid, s, _ in cands:                                  # 가장 많이 오른 것(같으면 앞 순서)
            if s > base_score and (best is None or s > best[2]):
                best = (c, cid, s)
        new_id = best[1] if best else base_id
        for c, cid, s, bys in cands:
            steps.append(_step("backward", None, rnd, c, cid, base_id, base_score, s,
                               "remove" if best and cid == best[1] else "keep", new_id, bys))
        if best is None:
            break
        base_id, codes, base_score = best[1], [x for x in codes if x != best[0]], best[2]
    return {"id": base_id, "codes": sort_codes(codes), "score": base_score, "ae_tab": ae_tab if "ae4" in codes else None}


def selection_doc(k_star, steps, final, status, phase_done):
    from e2eval import TINT_RULE, tint_bounds
    vs, ve = tint_bounds()
    d = {"rule": RULE, "val_T_range": [str(vs), str(ve)], "weather_L": E.L, "k_star": int(k_star), "seeds": list(E.SEEDS),
         "start_base_id": "E00", "steps": steps,
         "final": ({"model_id": final["id"], "bundles": final["codes"], "score": final["score"]} if final else None),
         "status": status, "phases_done": phase_done, "score_detail": TINT_RULE, "ae_fixed_note": AE_FIXED_NOTE,
         "bundles_def": [{"bundle": b["code"], "bundle_name": b["name"], "n_cols": b["n"], "add": b["add"], "desc": b["desc"]}
                         for b in E.FS_BUNDLES]}
    if final and final.get("ae_tab"):
        d["final"]["ae_tab_feats"] = final["ae_tab"]
    return d


def replay(stage, score_of, k_star):
    """stage 'fs' = 전진만, 'bw' = 전진 + 뒤로. 돌려줌 (steps, final, None) 또는 결과가 모자라면 (steps, None, NotReady 후보 exp)."""
    steps = []
    try:
        fw = forward(score_of, k_star, steps)
        if stage == "fs":
            return steps, fw, None
        fin = backward(score_of, fw, k_star, steps)
        return steps, fin, None
    except NotReady as nr:
        return steps, None, nr.exp


def results_score_of(P, smoke):
    """저장된 결과로 점수: 시드 전부 완료면 (시드 평균, 시드별), 아니면 None."""
    from e2tables import tint_seed_scores
    seeds = E.SMOKE_SEEDS if smoke else E.SEEDS

    def score_of(exp):
        v = tint_seed_scores(P, exp["id"], seeds)
        return None if v is None else (float(np.mean(v)), [float(x) for x in v])
    return score_of


def run_stage(stage, P, smoke, log, run_fn, k_star):
    """run.py 에서: 결과가 모자란 후보를 차례로 실행(run_fn(exp, seed))하며 재생을 되풀이. 끝나면 selection_FS.json."""
    from e2tables import is_complete, jdump
    score_of = results_score_of(P, smoke)
    t0 = time.time()
    n_run = 0
    while True:
        steps, final, need = replay(stage, score_of, k_star)
        done = (["forward"] if final is not None or need["stage"] == "bw" else []) + (["backward"] if stage == "bw" and final else [])
        jdump(selection_doc(k_star, steps, final, "진행 중" if need else "끝", done), P["results"] / "selection_FS.json")
        if steps:
            s = steps[-1]
            log(f"  [{s['phase']}] {s['candidate_id']} {s['bundle']}: 기준 {s['base_id']} {s['base_score']:.6f} vs 후보 "
                f"{s['candidate_score']:.6f} -> {s['decision_ko']} (기준 {s['new_base_id']})")
        if need is None:
            log(f"단계 선택 {stage} 끝: 최종 {final['id']} 묶음 {final['codes']} 점수 {final['score']:.6f} "
                f"(이번에 실행 {n_run}개, {time.time() - t0:.0f}s) -> results/selection_FS.json")
            return final
        seeds = E.exp_seeds(need, smoke)
        log(f"단계 선택 {stage}: 다음 후보 {need['id']} ({need['desc']}) 시드 {seeds}")
        for s in seeds:
            if not is_complete(P["results"] / E.run_dir_name(need["id"], s)):
                run_fn(need, s)
                n_run += 1
        if not all(is_complete(P["results"] / E.run_dir_name(need["id"], s)) for s in seeds):
            raise RuntimeError(f"{need['id']}: 실행했으나 결과가 완전하지 않음")


def table_experiments(P, smoke):
    """표용: 저장된 결과로 재생해 지금까지 정해진 FS·BW 후보 실험(+ 결과가 없는 다음 후보 하나)과 결정 dict(후보 ID -> 단계 기록)."""
    try:
        from e2tables import select_k
        k_star = select_k(P, smoke)["k_star"]
    except Exception:
        return [], {}
    score_of = results_score_of(P, smoke)
    exps = []

    def so(e):
        if e["id"] != "E00":
            exps.append(e)
        return score_of(e)
    steps, _final, _need = replay("bw", so, k_star)
    seen, out = set(), []
    for e in exps:
        if e["id"] not in seen:
            seen.add(e["id"])
            out.append(e)
    return out, {st["candidate_id"]: st for st in steps}
