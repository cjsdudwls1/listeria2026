# -*- coding: utf-8 -*-
"""표: tables/실험기록.xlsx
  시트 '실험기록'       실험마다 한 줄(시드 평균, 구간 가중 PR-AUC 는 평균±표준편차)
  시트 '시드별'         실험 x 시드 한 줄
  시트 '1차_가중지표'   1차 결과 16개 metrics_by_h.csv 에서 다시 계산(재학습 없음)
  시트 '고위험품목비교' C 실험(C1~C4, C4 = 학습 구간 검출 품목 전부 다운사이징): 같은 품목 시험(·검증) 행에서 전용 모델과 전체 모델(E00, A 최고) 저장 예측 비교
                        (전체 모델은 기준값 = 자기 검증(전체 행) 줄과, 기준값 = 그 품목 검증 행에서 다시 구한 줄 두 가지, 재학습 없음)
  시트 '설정'           k*, A 최고, 고위험 품목 묶음, 규칙
선정 기준 = 검증 구간 가중 PR-AUC (시드 평균). 시험은 기록만.
  EXP2_SELECT=tint(사용자 확정 원칙 2026-10-19) 이면 k*·A 최고·F·G 조합 선정 = 검증 T 통합 60일평균 PR-AUC(e2eval.val_tint_pr_auc, 시드 평균).
"""
import json
import re

import numpy as np
import pandas as pd

import config as C1
import metrics as M1
import e2cfg as E
from e2eval import CM_KEYS, TINT_RULE, read_preds, summarize, val_tint_pr_auc
from e2smote import DIFF_FROM_STANDARD, METHOD as SMOTE_METHOD

MODEL_NAME = {"cat": "CatBoost", "lr": "로지스틱 회귀", "dl": "딥러닝", "ae": "이상 점수(AE·VAE, CatBoost 없음)",
              "stack": "스태킹(로지스틱 메타)"}                      # stack = E08 스태킹(사용자 지시 2026-10-19, e2stk.py)
XS_RULE = ("단계 'xs', 시드 42·43·44, 기존 CatBoost 설정 그대로(사용자 지시 2026-10-19). E00_R1 = E00 + 1년 스무딩 검출률 3칸(rate1: 품목군 3단계·시군구·단위, "
           f"k = k*, A3 의 rate1 칸과 같은 e2hist 계산) - 계획서 11절 3순위(기준 + 최근 1년 검출률). E00_NW = E00 에서 기상 요약 {E.NW}칸을 뺀 판(범주 4 + h), "
           f"A3_NW = A3 에서 기상 요약 {E.NW}칸을 뺀 판(범주 4 + h + 이력 9, k*) - 계획서 6절 실험 A(정형 데이터만)")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))


RUN_FILES = ["config.json", "metrics_by_h.csv", "pred_val.parquet", "pred_test.parquet", "summary.json"]


def is_complete(rdir):
    if not all((rdir / f).exists() for f in RUN_FILES):
        return False
    try:
        return bool(jload(rdir / "summary.json").get("complete"))
    except Exception:
        return False


def load_summary(P, exp_id, seed):
    rdir = P["results"] / E.run_dir_name(exp_id, seed)
    return jload(rdir / "summary.json") if is_complete(rdir) else None


# ---------------------------------------------------------------- 선정
def tint_score(P, exp_id, seed):
    """검증 T 통합 60일평균 PR-AUC (그 실행의 저장 검증 예측에서, e2eval.val_tint_pr_auc). 결과가 완전하지 않으면 None."""
    rdir = P["results"] / E.run_dir_name(exp_id, seed)
    return val_tint_pr_auc(rdir)["mean60"] if is_complete(rdir) else None


def tint_seed_scores(P, exp_id, seeds):
    """시드마다 검증 T 통합 60일평균 PR-AUC. 하나라도 결과가 없으면 None (seed_stats 와 같은 약속)."""
    vals = []
    for s in seeds:
        v = tint_score(P, exp_id, s)
        if v is None:
            return None
        vals.append(v)
    return vals


def select_k(P, smoke, log=None):
    if E.SELECT == "tint":
        return _select_k_tint(P, smoke, log)
    recs = []
    for k in E.K_GRID:
        s = load_summary(P, f"A1_k{k}", 42 if not smoke else E.SMOKE_SEEDS[0])
        if s is None:
            raise RuntimeError(f"k 비교 결과가 완전하지 않음: A1_k{k}")
        recs.append({"k": k, "run": s["run"], "val_pr_auc_w": s["val_pr_auc_w"], "test_pr_auc_w": s["test_pr_auc_w"]})
    best = max(recs, key=lambda r: r["val_pr_auc_w"])          # 같은 값이면 앞(작은 k)
    sel = {"rule": "A1_k{10,30,50,100,200} (시드 42) 중 검증 구간 가중 PR-AUC 최대. 같은 값이면 작은 k. 시험 미사용",
           "ranking": sorted(recs, key=lambda r: -r["val_pr_auc_w"]), "k_star": best["k"]}
    jdump(sel, P["results"] / "selection_k.json")
    if log:
        log(f"k* = {best['k']} (검증 구간 가중 PR-AUC {best['val_pr_auc_w']:.5f})")
    return sel


def _select_k_tint(P, smoke, log=None):
    """EXP2_SELECT=tint: A1_k* (시드 42) 중 검증 T 통합 60일평균 PR-AUC 최대(같은 값이면 작은 k). 옛 점수는 기록만."""
    seed = 42 if not smoke else E.SMOKE_SEEDS[0]
    recs = []
    for k in E.K_GRID:
        s = load_summary(P, f"A1_k{k}", seed)
        if s is None:
            raise RuntimeError(f"k 비교 결과가 완전하지 않음: A1_k{k}")
        recs.append({"k": k, "run": s["run"], "val_tint_pr_auc_mean60": tint_score(P, f"A1_k{k}", seed),
                     "val_pr_auc_w(기록만)": s["val_pr_auc_w"], "val_pr_auc_mean60(기록만)": s["val_pr_auc_mean60"],
                     "test_pr_auc_w": s["test_pr_auc_w"]})
    best = max(recs, key=lambda r: r["val_tint_pr_auc_mean60"])   # 같은 값이면 앞(작은 k)
    sel = {"rule": f"A1_k{{{','.join(map(str, E.K_GRID))}}} (시드 {seed}) 중 {TINT_RULE}이 최대. 같은 값이면 작은 k",
           "select_rule": "tint", "ranking": sorted(recs, key=lambda r: -r["val_tint_pr_auc_mean60"]), "k_star": best["k"]}
    jdump(sel, P["results"] / "selection_k.json")
    if log:
        log(f"k* = {best['k']} (검증 T 통합 60일평균 PR-AUC {best['val_tint_pr_auc_mean60']:.5f})")
    return sel


def seed_stats(P, exp_id, seeds, key):
    vals = []
    for s in seeds:
        sm = load_summary(P, exp_id, s)
        if sm is None:
            return None
        vals.append(sm[key])
    return vals


def select_A(P, smoke, log=None):
    seeds = E.SMOKE_SEEDS if smoke else E.SEEDS
    if E.SELECT == "tint":
        return _select_A_tint(P, seeds, log)
    recs = []
    for eid in ("A1", "A2", "A3"):
        v = seed_stats(P, eid, seeds, "val_pr_auc_w")
        if v is None:
            raise RuntimeError(f"A 결과가 완전하지 않음: {eid} (시드 {seeds})")
        t = seed_stats(P, eid, seeds, "test_pr_auc_w")
        recs.append({"exp": eid, "val_pr_auc_w_mean": float(np.mean(v)), "val_pr_auc_w_by_seed": v,
                     "test_pr_auc_w_mean": float(np.mean(t))})
    best = max(recs, key=lambda r: r["val_pr_auc_w_mean"])     # 같은 값이면 앞(A1)
    sel = {"rule": f"A1~A3 중 검증 구간 가중 PR-AUC 시드 평균(시드 {seeds}) 최대. 같은 값이면 앞. 시험 미사용",
           "ranking": sorted(recs, key=lambda r: -r["val_pr_auc_w_mean"]), "best": best["exp"]}
    jdump(sel, P["results"] / "selection_A.json")
    if log:
        log(f"A 최고 = {best['exp']} (검증 구간 가중 PR-AUC 시드 평균 {best['val_pr_auc_w_mean']:.5f})")
    return sel


def _select_A_tint(P, seeds, log=None):
    """EXP2_SELECT=tint: A1~A3 중 검증 T 통합 60일평균 PR-AUC 시드 평균 최대(같은 값이면 앞). 옛 점수는 기록만."""
    recs = []
    for eid in ("A1", "A2", "A3"):
        v = tint_seed_scores(P, eid, seeds)
        if v is None:
            raise RuntimeError(f"A 결과가 완전하지 않음: {eid} (시드 {seeds})")
        w = seed_stats(P, eid, seeds, "val_pr_auc_w")
        t = seed_stats(P, eid, seeds, "test_pr_auc_w")
        recs.append({"exp": eid, "val_tint_pr_auc_mean60_mean": float(np.mean(v)), "val_tint_pr_auc_mean60_by_seed": v,
                     "val_pr_auc_w_mean(기록만)": float(np.mean(w)), "test_pr_auc_w_mean": float(np.mean(t))})
    best = max(recs, key=lambda r: r["val_tint_pr_auc_mean60_mean"])
    sel = {"rule": f"A1~A3 중 {TINT_RULE}(시드 {seeds})이 최대. 같은 값이면 앞", "select_rule": "tint",
           "ranking": sorted(recs, key=lambda r: -r["val_tint_pr_auc_mean60_mean"]), "best": best["exp"]}
    jdump(sel, P["results"] / "selection_A.json")
    if log:
        log(f"A 최고 = {best['exp']} (검증 T 통합 60일평균 PR-AUC 시드 평균 {best['val_tint_pr_auc_mean60_mean']:.5f})")
    return sel


# ---------------------------------------------------------------- 1차 가중 지표
RUN1_PAT = re.compile(r"^(xgboost|lightgbm|catboost|lstm)__(summary|raw|seq)__L(\d{3})$")


def prev1_table(P, log=None):
    rows = []
    for d in sorted(E.EXP1_RESULTS.iterdir()):
        m = RUN1_PAT.match(d.name)
        if not m or not (d / "metrics_by_h.csv").exists():
            continue
        mdf = pd.read_csv(d / "metrics_by_h.csv")
        s = summarize(mdf)
        r = {"1차 실행": d.name, "모델": m.group(1), "입력": m.group(2), "L": int(m.group(3)),
             "판정 기준값": s.get("threshold")}
        for sp, spk in (("val", "검증"), ("test", "시험")):
            for mt, mk in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("recall", "Recall"),
                           ("precision", "Precision"), ("f2", "F2")):
                r[f"{spk} {mk} 60일 평균"] = s[f"{sp}_{mt}_mean60"]
                r[f"{spk} {mk} 구간 가중"] = s[f"{sp}_{mt}_w"]
        try:   # 1차 summary.json 의 60일 평균과 같은지(계산 확인용)
            s1 = jload(d / "summary.json")
            r["확인: 1차 summary 60일 평균 PR-AUC 와 차이"] = abs(s1["val_pr_auc_mean60"] - s["val_pr_auc_mean60"]) + \
                abs(s1["test_pr_auc_mean60"] - s["test_pr_auc_mean60"])
        except Exception:
            r["확인: 1차 summary 60일 평균 PR-AUC 와 차이"] = None
        rows.append(r)
    df = pd.DataFrame(rows)
    P["tables"].mkdir(parents=True, exist_ok=True)
    df.to_csv(P["tables"] / "1차_가중지표.csv", index=False, encoding="utf-8-sig")
    if log:
        log(f"1차 가중 지표: {len(df)}개 실행 -> {P['tables'] / '1차_가중지표.csv'}")
    return df


# ---------------------------------------------------------------- 실험기록
def _ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return np.nan, np.nan
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else np.nan)


def _fmt(m, s):
    if np.isnan(m):
        return ""
    return f"{m:.4f}" + (f" ± {s:.4f}" if not np.isnan(s) else " (시드 1개)")


def input_desc(cfg_feats, model):
    if cfg_feats is None:
        return ""
    f = cfg_feats
    parts = []
    if f.get("cats"):
        parts.append("범주 4(sido·sgg·p2·p3)")
    parts.append("h")
    if f.get("weather"):
        parts.append(f"기상 요약 {E.L}일({E.NW})")
    if f.get("hist"):
        lab = {"rate5": "5년 스무딩 검출률 3", "logn5": "5년 log(1+검사 수) 3", "rate1": "1년 스무딩 검출률 3",
               "logn1": "1년 log(1+검사 수) 3", "sd1": "1년 검출률 사후 표준편차 3"}      # logn1·sd1 = HX(사용자 지시 2026-10-19)
        parts.append("이력[" + ", ".join(lab[x] for x in f["hist"]) + f"] (k={f.get('k')})")
    if f.get("woe"):
        parts.append("WOE 4(범주 대신)")
    if f.get("ext"):
        import e2fg
        parts.append(e2fg.ext_desc(f["ext"]))
    if f.get("nb"):                                              # N 제품명(사용자 승인 2026-10-19)
        import e2nb
        parts.append(e2nb.nb_desc(f["nb"]))
    if f.get("ae"):                                              # AE·VAE 이상 점수(사용자 지시 2026-10-19)
        import e2ae
        parts.append(e2ae.ae_desc(f["ae"]) + (" (표 AE·VAE 입력 = 단계 선택 그 시점 기준 모델 입력)" if f.get("ae_tab") else ""))
    return " + ".join(parts)


def _pct(vals):
    v = [x for x in vals if x is not None]
    return f"{min(v):.1%}" if v else "기록 없음"


def smote_note(infos):
    """D2 비고: 완료 시드들의 summary['smotenc'] 목록 -> 한 문자열. 점검 비율은 시드 중 최소값."""
    infos = [i for i in infos if i]
    if not infos:
        return ""
    si = infos[0]
    if not si.get("method"):                                 # 이전 방식(전국 이웃 + 칸마다 이웃 최빈값)으로 만든 결과
        cc = si.get("cat_consistency") or {}
        return (f"SMOTENC 새 표본 {si.get('n_new')} - [주의] 이전 방식(전국 이웃 + 칸마다 이웃 최빈값 범주)으로 만든 결과"
                + (f": sido ≠ sgg 앞 2자리 {cc['sido_mismatch_rate']:.1%}" if "sido_mismatch_rate" in cc else ""))
    ccs = [i.get("cat_consistency") or {} for i in infos]
    fb = si.get("fallback_groups") or {}
    fb_txt = (", ".join(f"{g}(학습 검출 행 {n}개)" for g, n in fb.items())
              + f" -> 그 시도 검출 행을 기준 행으로 한 새 표본 {', '.join(str(i.get('n_new_from_fallback')) for i in infos)}개(시드별)"
              if fb else "없음")
    parts = [f"SMOTE 방식 = {si['method']}(사용자 결정 2026-10-19): 학습 검출 {si['n_minority']:,} + 새 표본 {si['n_new']:,} "
             f"= 불검출 {si['n_majority']:,}의 {si['ratio_after']:.0%}",
             f"이웃 = 기준 검출 행과 같은 시도(sgg_code 앞 2자리) 학습 검출 행 중 거리 가까운 {si['k']}개"
             f"(그 시도 학습 검출 행이 {si['group_min_rows']}개 미만이면 전국 학습 검출 행에서), 숫자 칸 = 기준 행 + U(0,1) x (이웃 - 기준 행), "
             f"범주 4칸 = 기준 검출 행 값 그대로",
             "시도별 학습 검출 행: " + ", ".join(f"{g} {n}" for g, n in si["minority_by_group"].items()),
             f"전국 대체 시도: {fb_txt}",
             f"합성 행 범주 점검(시드 {len(infos)}개 중 최소): sido = sgg 앞 2자리 {_pct([c.get('sido_eq_sgg2_rate') for c in ccs])}, "
             f"기준 검출 행(실제 검출 단위)과 같은 단위 {_pct([c.get('same_as_base_rate') for c in ccs])}, "
             f"학습 검출 단위에 있는 조합 {_pct([c.get('in_train_pos_units_rate') for c in ccs])}, "
             f"이웃이 같은 시도(전국 대체 제외) {_pct([i.get('neighbor_same_group_rate') for i in infos])}",
             "표준 SMOTENC 와 다른 점: " + " / ".join(f"({k + 1}) {d}" for k, d in enumerate(si.get("diff_from_standard_smotenc") or []))]
    return "; ".join(parts)


def c4_outside_text(G4):
    """C4 밖(학습 때 검출 없던 품목) 검증·시험 검출 = C4 전용 모델이 다루지 않는 범위(사용자 지시 2026-10-19 추가 보고). G4 = groups.json 의 C4."""
    o = (G4 or {}).get("outside")
    if not o:
        return "C4 밖 검출: groups.json 에 기록 없음"
    parts = []
    for sp, lab in (("val", "검증"), ("test", "시험")):
        r = o[sp]
        t = f"{lab} 검출 행 {r.get('pos_rows_outside', '?'):,}/{r.get('pos_rows', 0):,}" if "pos_rows" in r else f"{lab} 검출 행 기록 없음"
        if r.get("pos_rows"):
            t += f" ({r['pos_rows_outside'] / r['pos_rows']:.2%})"
        t += f"·검출 시료 {r['pos_samples_outside']:,}/{r['pos_samples']:,}"
        if "rows" in r:
            t += f", C4 밖 단위x날짜 행 {r['rows_outside']:,}/{r['rows']:,} ({r['rows_outside'] / r['rows']:.1%})"
        if r.get("pos_rows_outside_codes"):
            t += " - 검출 행 품목 " + ", ".join(f"{c} {n}" for c, n in r["pos_rows_outside_codes"].items())
        parts.append(t)
    return "C4 밖(학습 때 검출 없던 품목) 검출 = 이 모델이 다루지 않는 범위: " + "; ".join(parts)


def c4_note(G4):
    """C4 실험 비고 첫머리: C_NOTE 와 같은 뜻이나 '훨씬 높음' 대신 C4 행 검출 비율을 적음(C4 행은 전체 행의 약 69%라 비율 차이가 작음)."""
    o = (G4 or {}).get("outside") or {}
    rates = []
    for sp, lab in (("val", "검증"), ("test", "시험")):
        r = o.get(sp) or {}
        if r.get("rows") and "rows_outside" in r:
            n_in = r["rows"] - r["rows_outside"]
            y_in = r["pos_rows"] - r["pos_rows_outside"]
            rates.append(f"{lab} C4 행 {y_in / n_in:.2%} vs 전체 행 {r['pos_rows'] / r['rows']:.2%}")
    return ("그 품목 행만 평가(검출 비율이 전체보다 높음" + (": " + ", ".join(rates) if rates else "") + ") - 전체 실험과 PR-AUC 등을 직접 비교 불가, "
            "순위 제외. 전체 모델과의 비교는 '고위험품목비교' 시트")


NB_NOTE = ("N2·N0 = 목표일 그 단위 시료의 제품명으로 만든 값 -> 운영에서 예측할 때 그날 검사할 제품명을 입력받는다는 전제(보고서 한계, 설정 시트 'N 전제·한계')")
C_NOTE = "그 품목 행만 평가(검출 비율이 전체보다 훨씬 높음) - 전체 실험과 PR-AUC 등을 직접 비교 불가, 순위 제외. 전체 모델과의 비교는 '고위험품목비교' 시트"
RANK_COL = "순위(전체 행 실험, 검증 구간가중 PR-AUC 시드 평균)"


def add_rank(rows):
    """전체 행을 평가한 실험(C 제외)만 검증 구간 가중 PR-AUC 시드 평균으로 순위(1 = 최고, 같은 값은 같은 순위).
    시드 1개 실험(A1_k*, H30, smoke)은 '순위 참고'에 표시. 시험은 순위에 쓰지 않음."""
    vals = [(i, r["검증 구간가중 PR-AUC 평균"]) for i, r in enumerate(rows)
            if r.get("평가 행 범위") == "전체 행" and not _no_rank(r) and not np.isnan(r.get("검증 구간가중 PR-AUC 평균", np.nan))]
    order = sorted(vals, key=lambda x: -x[1])
    for i, r in enumerate(rows):
        r.setdefault(RANK_COL, None)
        if r.get("평가 행 범위") != "전체 행":
            r["순위 참고"] = "품목 행만 평가 - 순위 제외"
        elif np.isnan(r.get("검증 구간가중 PR-AUC 평균", np.nan)):
            r["순위 참고"] = "결과 없음"
        elif _no_rank(r):                                        # 스태킹: 검증 = 메타 학습 구간 -> 검증 기준 순위에서 뺌
            import e2stk
            r["순위 참고"] = e2stk.RANK_NOTE
    for i, v in order:
        r = rows[i]
        r[RANK_COL] = 1 + sum(1 for _, w in order if w > v)
        n_done, n_seeds = (int(x) for x in str(r["시드 수"]).split("/"))
        ref = []
        if r[RANK_COL] == 1:
            ref.append("최고")
        if n_done == 1:
            ref.append("시드 1개(평균 아님)")
        elif n_done < n_seeds:
            ref.append(f"완료 시드 {n_done}/{n_seeds}")
        r["순위 참고"] = ", ".join(ref)


def _no_rank(r):
    """검증 기준 순위에서 빼는 줄: 스태킹(메타 모델을 검증 구간으로 학습 - 검증 점수가 메타 학습 구간 점수)."""
    return r.get("모델") == MODEL_NAME["stack"]


def _stk_input_desc(exp):
    import e2stk
    return e2stk.input_desc(exp)


def _ae_only_desc(blocks):
    import e2ae
    return e2ae.score_only_desc(e2ae.parse_block(blocks[0])[0])


def ae_notes(exp, blocks, done):
    """AE·VAE 실험 비고: 점수 단독이면 p 단위·방향(검증·시험 구간가중 PR-AUC vs 기저율), A3 + 점수면 점수 공유·위약 방식."""
    import e2ae
    out = []
    if exp["model"] == "ae":
        out.append("점수 단독(CatBoost 없음): 예측값 p = 이상 점수 그대로(확률 아님), 판정 기준값도 점수 단위")
        S = [sm for _, sm in done]
        for sp, lab in (("val", "검증"), ("test", "시험")):
            v = [sm.get(f"{sp}_pr_auc_w") for sm in S]
            b = [sm[f"{sp}_n_pos"] / sm[f"{sp}_n_rows"] for sm in S if sm.get(f"{sp}_n_rows")]
            if v and b:
                m, base = float(np.mean(v)), float(np.mean(b))
                out.append(f"방향({lab}): 구간가중 PR-AUC {m:.4f} vs 기저율 {base:.4f} -> "
                           + ("'클수록 검출' 방향 맞음" if m > base else "반대 방향(PR-AUC < 기저율) - 방향을 바꾸지 않고 그대로 둠"))
    else:
        out.append("점수는 시드마다 한 번 계산해 S_·A3_·_p 가 같이 씀(results/_ae_scores/), 점수 규칙 v2: 학습 행 전부(검출 포함) 5겹 배정 -> 그 행을 안 본 겹 모델 점수, 검증·시험 = 겹 모델 5개 평균")
    if any(e2ae.parse_block(b)[1] for b in blocks):
        out.append("위약판: 점수를 분할(·h)마다 예측일 T 의 달력 연도 안에서 행끼리 섞음(설정 시트 'AE 위약판')")
    return out


def _dl_input_desc(P, exp_id, done):
    """DL 은 summary 의 feats 가 없으므로 첫 완료 시드 config.json 의 n_features_desc 를 쓴다."""
    if not done:
        return ""
    try:
        return jload(P["results"] / E.run_dir_name(exp_id, done[0][0]) / "config.json").get("n_features_desc", "")
    except Exception:
        return ""


def build_tables(P, smoke, log=None):
    tab = P["tables"]
    tab.mkdir(parents=True, exist_ok=True)
    notes = {}
    try:
        selk = select_k(P, smoke)
    except Exception as e:
        selk = None
        notes["k*"] = f"미정: {e}"
    try:
        selA = select_A(P, smoke)
    except Exception as e:
        selA = None
        notes["A 최고"] = f"미정: {e}"
    main_rows, seed_rows = [], []
    gpath = P["results"] / "groups.json"
    groups_j = jload(gpath) if gpath.exists() else {}
    import e2fg                                                  # F·G 조합: F·G 결과가 다 있으면 선정해서 줄을 더함
    try:
        combos, selFG = e2fg.combo_experiments(P, smoke, None, strict=False)
    except Exception as e:
        combos, selFG = [], {"note": f"조합 선정 오류: {e}"}
    import e2fs                                                  # 양방향 단계 선택(사용자 결정 2026-10-19): 저장 결과로 결정을 재생해 정해진 후보만 줄로
    try:
        fsx, fsrec = e2fs.table_experiments(P, smoke)
    except Exception as e:
        fsx, fsrec = [], {}
        notes["단계 선택"] = f"재생 오류: {e}"
    for exp in list(E.EXPERIMENTS) + combos + fsx:
        seeds = E.exp_seeds(exp, smoke)
        sums = [(s, load_summary(P, exp["id"], s)) for s in seeds]
        done = [(s, sm) for s, sm in sums if sm is not None]
        for s, sm in done:
            r = {"실험 ID": exp["id"], "시드": s}
            r.update({k: v for k, v in sm.items() if not isinstance(v, (dict, list))})
            seed_rows.append(r)
        note = []
        if exp["model"] == "dl" and not done:
            note.append("아직 실행 안 함")
        if len(done) < len(seeds):
            note.append(f"완료 시드 {len(done)}/{len(seeds)}")
        if done and (done[0][1].get("feats") or {}).get("k") is not None:   # DL 은 feats = None
            note.append(f"k={done[0][1]['feats']['k']}")
        if exp.get("feats") == "A*" and done:
            note.append(f"A 최고 = {done[0][1].get('feats_from')}")
        if exp.get("group"):
            note.append(C_NOTE if exp["group"] != "C4" else c4_note(groups_j.get("C4")))
            if done:
                g = done[0][1].get("group_codes")
                note.append(f"품목 {len(g)}종: {', '.join(g)}")
            if exp["group"] == "C4":
                note.append(c4_outside_text(groups_j.get("C4")))
        if exp.get("smote") and done:
            sn = smote_note([sm.get("smotenc") for _, sm in done])
            if sn:
                note.append(sn)
        if exp.get("spw") and done:
            note.append(f"scale_pos_weight={done[0][1].get('scale_pos_weight')}")
        ext = (exp.get("feats") or {}).get("ext") if isinstance(exp.get("feats"), dict) else None
        if exp.get("combo_kind"):
            note.append(f"조합(A3 + {' + '.join(exp['combo_groups'])}), 선정 = results/selection_FG.json"
                        + (f" - 대체 실행(검증에서 E00 보다 높은 F·G 묶음 없음, 완료 시드 {len(done)}개)" if exp["combo_kind"] == "fallback" else ""))
        if ext and "food" in ext:
            note.append("식품: 3단계 코드 없는 단위·표에 없는 코드는 숫자 결측, aw_label '결측' 범주. 범주 5칸(aw_label 추가)")
        if ext and any(b in ("g1", "g2", "g3") for b in ext):          # G 묶음만(지하수 'gw'·'gw_p' 는 G 가 아님)
            note.append("G 변수 = 바깥 기온·시군구 기상으로 만든 대리 변수(설정 시트 'G 한계' 참고)")
        nbb = (exp.get("feats") or {}).get("nb") if isinstance(exp.get("feats"), dict) else None
        if nbb and any(b in ("n2", "n0") for b in nbb):
            note.append(NB_NOTE)
        if nbb and "n1" in nbb:
            note.append("N1 = 과거(T 전날까지) 시료 제품명만 씀, 건수 칸 없음")
        hxx = [x for x in ((exp.get("feats") or {}).get("hist") or []) if x in ("logn1", "sd1")] if isinstance(exp.get("feats"), dict) else []
        if hxx:                                                  # HX(사용자 지시 2026-10-19)
            note.append("1년 칸 날짜 시계 확인 = logs/selftest_hx_dateclock.csv (설정 시트 'HX 날짜 시계 확인')")
        aeb = (exp.get("feats") or {}).get("ae") if isinstance(exp.get("feats"), dict) else None
        if aeb:                                                  # AE·VAE(사용자 지시 2026-10-19)
            note += ae_notes(exp, aeb, done)
        if exp["stage"] in ("p", "q") and isinstance(exp.get("feats"), dict) and exp["feats"].get("hist"):
            note.append("A3 위에 더한 판(기준이 다름)")                 # P·Q 중 A3 기준 실험(E00_P·E00_Q = 베이스라인 기준, 사용자 결정 2026-10-19)
        if exp.get("fs_info"):                                   # 양방향 단계 선택(사용자 결정 2026-10-19)
            r = fsrec.get(exp["id"])
            note.append(f"단계 선택 {'전진' if exp['fs_info']['phase'] == 'forward' else '뒤로'}: 기준 {exp['fs_info']['base_id']} "
                        + (f"{r['base_score']:.6f} vs 후보 {r['candidate_score']:.6f} -> {r['decision_ko']} (다음 기준 {r['new_base_id']})"
                           if r else "- 결과 없음") + " (점수 = 검증 T 통합 60일평균 PR-AUC 시드 평균, results/selection_FS.json)")
        if exp["model"] == "stack":                              # E08 스태킹(사용자 지시 2026-10-19)
            import e2stk
            note += e2stk.table_notes(exp, done)
        if smoke:
            note.append("시험 실행(smoke): 일부 행, CatBoost 라운드 제한(DL 은 본 설정 에폭) - 성능 해석 금지")
        S = [sm for _, sm in done]

        def col(key):
            return [sm.get(key) for sm in S]

        vm, vs = _ms(col("val_pr_auc_w"))
        tm, ts = _ms(col("test_pr_auc_w"))
        desc = exp["desc"]
        if exp.get("smote") and S and (S[0].get("smotenc") or {}).get("method"):
            desc += f" [방식: {S[0]['smotenc']['method']} - 비고 참고]"
        row = {"실험 ID": exp["id"], "계획서 대응": exp["plan"], "설명": desc,
               "모델": MODEL_NAME[exp["model"]],
               "입력": (_dl_input_desc(P, exp["id"], done) if exp["model"] == "dl"
                        else _ae_only_desc(aeb) if exp["model"] == "ae"
                        else _stk_input_desc(exp) if exp["model"] == "stack"
                        else input_desc(S[0].get("feats") if S else None, exp["model"])),
               "학습 행 수": S[0].get("n_train_rows") if S else None,
               "학습 검출 행 수": S[0].get("n_train_pos") if S else None,
               "시드 수": f"{len(done)}/{len(seeds)}",
               "평가 행 범위": (f"{exp['group']} 품목 {len(S[0]['group_codes'])}종 행만" if exp.get("group") and S
                              else (f"{exp['group']} 품목 행만" if exp.get("group") else "전체 행")),
               "검증 구간가중 PR-AUC (평균±표준편차)": _fmt(vm, vs),
               "시험 구간가중 PR-AUC (평균±표준편차)": _fmt(tm, ts),
               "검증 구간가중 PR-AUC 평균": vm, "검증 구간가중 PR-AUC 표준편차": vs,
               "시험 구간가중 PR-AUC 평균": tm, "시험 구간가중 PR-AUC 표준편차": ts}
        for sp, spk in (("val", "검증"), ("test", "시험")):
            row[f"{spk} 60일평균 PR-AUC"] = _ms(col(f"{sp}_pr_auc_mean60"))[0]
        for sp, spk in (("val", "검증"), ("test", "시험")):
            row[f"{spk} 60일평균 ROC-AUC"] = _ms(col(f"{sp}_roc_auc_mean60"))[0]
            row[f"{spk} 구간가중 ROC-AUC"] = _ms(col(f"{sp}_roc_auc_w"))[0]
        for sp, spk in (("val", "검증"), ("test", "시험")):
            row[f"{spk} 구간가중 Recall"] = _ms(col(f"{sp}_recall_w"))[0]
            row[f"{spk} 구간가중 Precision"] = _ms(col(f"{sp}_precision_w"))[0]
            row[f"{spk} 구간가중 F2"] = _ms(col(f"{sp}_f2_w"))[0]
        row["판정 기준값(시드 평균)"] = _ms(col("threshold"))[0]
        row["시험 Precision 60일평균"] = _ms(col("test_precision_mean60"))[0]
        row["시험 Recall 60일평균"] = _ms(col("test_recall_mean60"))[0]
        row["시험 F2 60일평균"] = _ms(col("test_f2_mean60"))[0]
        for c in ("tp", "fn", "fp", "tn"):
            row[f"시험 {c.upper()} (h당 평균)"] = _ms(col(f"test_{c}_mean_per_h"))[0]
        row["학습 시간 초(시드 평균)"] = _ms(col("fit_seconds"))[0]
        row["총 시간 초(시드 평균)"] = _ms(col("total_seconds"))[0]
        row["최대 메모리 GB(private)"] = max([x for x in col("peak_private_gb") if x is not None], default=None)
        row["비고"] = "; ".join(note)
        main_rows.append(row)
    add_rank(main_rows)
    for nr in E.NOT_RUN:
        main_rows.append({"실험 ID": nr["id"], "계획서 대응": nr["plan"], "설명": nr["desc"],
                          "모델": MODEL_NAME[nr["model"]], "시드 수": "0 (실행 안 함)", "평가 행 범위": "-",
                          "순위 참고": "실행 안 함", "비고": nr["reason"]})
    main = pd.DataFrame(main_rows)
    cols = [c for c in main.columns if c not in (RANK_COL, "순위 참고")]
    at = cols.index("검증 구간가중 PR-AUC (평균±표준편차)") + 1          # 순위는 그 기준 칸 바로 뒤
    main = main[cols[:at] + [RANK_COL, "순위 참고"] + cols[at:]]
    main[RANK_COL] = main[RANK_COL].astype("Int64")                        # 정수(빈칸 허용)
    seed_df = pd.DataFrame(seed_rows)
    prev1_path = tab / "1차_가중지표.csv"
    prev1 = pd.read_csv(prev1_path) if prev1_path.exists() else prev1_table(P)
    comp = high_risk_comparison(P, smoke, selA)
    setting = settings_rows(P, smoke, selk, selA, notes, selFG)
    main.to_csv(tab / "실험기록.csv", index=False, encoding="utf-8-sig")
    seed_df.to_csv(tab / "시드별.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(tab / "고위험품목비교.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(tab / "실험기록.xlsx", engine="openpyxl") as xw:
        main.to_excel(xw, sheet_name="실험기록", index=False)
        seed_df.to_excel(xw, sheet_name="시드별", index=False)
        prev1.to_excel(xw, sheet_name="1차_가중지표", index=False)
        comp.to_excel(xw, sheet_name="고위험품목비교", index=False)
        setting.to_excel(xw, sheet_name="설정", index=False)
    if log:
        log(f"표 작성 끝 -> {tab / '실험기록.xlsx'} (실험 {len(main)}개, 시드별 {len(seed_df)}줄, 고위험 비교 {len(comp)}줄)")
    return main


# ---------------------------------------------------------------- 고위험 품목 비교
def _fixed_threshold_metrics(p_val, y_val, p_test, y_test, thr):
    recs = []
    for sp, P_, y in (("val", p_val, y_val), ("test", p_test, y_test)):
        ok = 0 < int(y.sum()) < len(y)
        for h in range(1, E.H_MAX + 1):
            p = P_[h - 1]
            cm = M1.confusion(y, p, thr)
            recs.append({"split": sp, "h": h, "n": len(y), "pos": int(y.sum()),
                         "pr_auc": M1.pr_auc(y, p) if ok else np.nan, "roc_auc": M1.roc_auc(y, p) if ok else np.nan,
                         "threshold": thr, **{k: cm[k] for k in CM_KEYS}})
    return summarize(pd.DataFrame(recs))


def _comp_row(group, gdesc, model_label, exp_id, seeds_summ, extra=""):
    r = {"묶음": group, "묶음 설명": gdesc, "모델": model_label, "실험 ID": exp_id, "시드 수": len(seeds_summ)}
    if not seeds_summ:
        r["비고"] = "결과 없음 " + extra
        return r
    r["시험 행 수"] = seeds_summ[0].get("test_n_rows")
    r["시험 검출 행 수"] = seeds_summ[0].get("test_n_pos")
    r["검증 행 수"] = seeds_summ[0].get("val_n_rows")
    r["검증 검출 행 수"] = seeds_summ[0].get("val_n_pos")
    for key, lab in (("val_pr_auc_w", "검증 구간가중 PR-AUC"), ("test_pr_auc_w", "시험 구간가중 PR-AUC"),
                     ("test_pr_auc_mean60", "시험 60일평균 PR-AUC"), ("test_roc_auc_mean60", "시험 60일평균 ROC-AUC"),
                     ("test_roc_auc_w", "시험 구간가중 ROC-AUC"), ("test_recall_w", "시험 구간가중 Recall"),
                     ("test_precision_w", "시험 구간가중 Precision"), ("test_f2_w", "시험 구간가중 F2"),
                     ("test_recall_mean60", "시험 60일평균 Recall"), ("test_precision_mean60", "시험 60일평균 Precision"),
                     ("test_f2_mean60", "시험 60일평균 F2"), ("threshold", "판정 기준값")):
        m, s = _ms([x.get(key) for x in seeds_summ])
        r[lab] = m
        if key in ("val_pr_auc_w", "test_pr_auc_w"):
            r[lab + " 표준편차"] = s
    for c in ("tp", "fn", "fp", "tn"):
        r[f"시험 {c.upper()} (h당 평균)"] = _ms([x.get(f"test_{c}_mean_per_h") for x in seeds_summ])[0]
    r["비고"] = extra
    return r


def high_risk_comparison(P, smoke, selA):
    seeds = E.SMOKE_SEEDS if smoke else E.SEEDS
    gpath = P["results"] / "groups.json"
    groups = jload(gpath) if gpath.exists() else None
    rows = []
    if groups is None:
        return pd.DataFrame([{"비고": "groups.json 없음(C 실험 전)"}])
    whole = [("전체 모델 E00", "E00")]
    if selA:
        whole.append((f"전체 모델 A 최고({selA['best']})", selA["best"]))
    for g in E.GROUPS:
        if g not in groups:                                      # C4 전에 만든 groups.json
            continue
        codes = groups[g]["codes"]
        gdesc = f"{groups[g]['desc']}: {', '.join(codes)} (학습 검출 시료 {groups[g]['train_pos']})"
        if g == "C4":
            gdesc += f"; {c4_outside_text(groups[g])}"
        for suffix, lab in (("base", "전용 모델(E00 입력)"), ("hist", "전용 모델(A 최고 입력)")):
            eid = f"{g}_{suffix}"
            ss = [s for s in (load_summary(P, eid, sd) for sd in seeds) if s is not None]
            rows.append(_comp_row(g, gdesc, lab, eid, ss, "판정 기준값 = 그 품목 검증 행에서"))
        regroup = []                                             # 기준값을 그 품목 검증 행에서 다시 구한 줄(사용자 승인 2026-10-19)
        for lab, eid in whole:
            ss, ss_re, thr_re = [], [], []
            for sd in seeds:
                rdir = P["results"] / E.run_dir_name(eid, sd)
                if not is_complete(rdir):
                    continue
                thr = jload(rdir / "summary.json")["threshold"]
                pv, yv, iv = read_preds(rdir / "pred_val.parquet")
                pt, yt, it = read_preds(rdir / "pred_test.parquet")
                mv = iv["train_prdlst_code_2"].astype(str).isin(codes).to_numpy()
                mt = it["train_prdlst_code_2"].astype(str).isin(codes).to_numpy()
                s = _fixed_threshold_metrics(pv[:, mv], yv[mv], pt[:, mt], yt[mt], thr)
                s["threshold"] = thr
                ss.append(s)
                thr_g = group_threshold(pv[:, mv], yv[mv])
                s2 = _fixed_threshold_metrics(pv[:, mv], yv[mv], pt[:, mt], yt[mt], thr_g)
                s2["threshold"] = thr_g
                ss_re.append(s2)
                thr_re.append(f"시드 {sd} {thr_g:.6f}")
            rows.append(_comp_row(g, gdesc, lab, eid, ss, "전체 모델 저장 예측을 그 품목 행으로 거름, 판정 기준값 = 전체 모델 자기 검증(전체 행)에서"))
            lab_re = (lab[:-1] + ", " if lab.endswith(")") else lab + "(") + "기준값 = 그 품목 검증 행에서 다시 구함)"
            regroup.append(_comp_row(g, gdesc, lab_re, eid, ss_re,
                                     "재학습 없음. 전체 모델 저장 예측을 그 품목 행으로 거른 뒤, 판정 기준값 = 그 품목 검증 행(60개 h 합침)에서 "
                                     "F2 최대 확률값을 다시 구함(전용 모델과 같은 규칙, 1차 metrics.f2_threshold) -> 시험에 그대로 적용. "
                                     f"PR-AUC·ROC-AUC 는 같은 모델의 '{lab}' 줄(기준값 = 전체 모델 자기 검증)과 같고 "
                                     "Recall·Precision·F2·TP~TN 만 다름"
                                     + (f". 기준값 {', '.join(thr_re)}" if thr_re else "")))
        rows += regroup
    return pd.DataFrame(rows)


def group_threshold(p_val, y_val):
    """그 품목 검증 행의 예측 (60, n) 과 y (n,) -> 60개 h 를 합친 예측에서 F2 최대 확률값(1차 metrics.f2_threshold).
    검출 또는 불검출이 없으면 NaN."""
    ya = np.tile(y_val, E.H_MAX)
    if not (0 < int(ya.sum()) < len(ya)):
        return float("nan")
    return float(M1.f2_threshold(ya, p_val.reshape(-1))["threshold"])


def fg_settings_rows(selFG):
    """F·G·조합 설정 줄(사용자 승인 2026-10-19)."""
    import e2fg as G
    it = [("F·G 공통", f"모델 = E00 과 같은 CatBoost 설정, 기본 입력 = E00(범주 4 + h + 기상 요약 {E.L}일) + 묶음. 예측하는 날 T = d − h. "
                      f"창 = T−{E.L} … T−1(T 당일 제외), '최근 10일' = T−10 … T−1. 결측은 채우지 않음(특성 행렬에 넣을 때 float32)"),
          ("F1 표준강수지수", f"{G.SPI_PATH.name} (sgg_code, date, spi3, spi6) -> 기상과 같은 창 요약(10일 구간 평균 5 + 평균·표준편차·최소·최대·기울기, "
                            f"결측 무시) 2 x {E.NSPI // 2} = {E.NSPI}칸"),
          ("F2 토양", f"{G.SOIL_PATH.name} 날짜 T 행 6칸 {G.SOIL_VARS}. 생성 규칙(04_시군구_토양.py): 날짜 t 값 = exam_day ∈ [t−1095일, t−1일] "
                     "그 시군구 기록 중앙값·기록 수(당일 검정 제외) -> T 전날까지 자료. 기록 0건이면 값 결측·건수 0"),
          ("F3 식품 속성", f"{G.FOOD_PATH.name} train_prdlst_code_2 로 붙임 -> 숫자 aw·ph·na(결측 그대로) 3칸 + aw_label 범주 1칸('결측' = 한 범주). "
                         "ph_label·na_label 은 넣지 않음(식품 세션 결정). 3단계 코드 없는 단위('코드없음')는 결측"),
          ("F4", f"F1 + F2 + F3 (숫자 {E.NSPI + 9}칸 + 범주 aw_label)")]
    it += list(G.F_NOTES)
    it += [("G 한계", G.G_LIMIT)]
    it += G.g_notes()
    it += [
          ("G1 온도 생장 요인", "f(T) = (T−Tmax)(T−Tmin)² / {(Topt−Tmin)[(Topt−Tmin)(T−Topt) − (Topt−Tmax)(Topt+Tmin−2T)]} (엑셀 D6 식의 온도 항), "
                            f"Tmin={G.SILVA['Tmin']}, Topt={G.SILVA['Topt']}, Tmax={G.SILVA['Tmax']} (F6). T = 시군구 일평균기온 avg_temp(°C). "
                            "Tmin 이하·Tmax 이상은 0. 창 요약 4칸: 평균, 최근 10일 평균, 합, f>0 인 날 수"),
          ("G2 µmax", "µmax = µopt·f(T)·f(pH)·f(aw) (Silva 2020, D6·D19·D26 식) - "
                      f"pH: pHmin={G.SILVA['pHmin']}, pHopt={G.SILVA['pHopt']}, pHmax={G.SILVA['pHmax']} (F26), "
                      f"aw: awmin={G.SILVA['awmin']}, awopt={G.SILVA['awopt']}, awmax=1.00 (F19, D6 식), µopt={G.SILVA['mu_opt']} (J6 비고 식 맨 앞 계수). "
                      "pH·aw = 품목 3단계 대표값(F3 와 같은 표), 결측이면 결측. 범위 밖 0. 창 요약: 평균·최근 10일 평균·최대"),
          ("G2 생장 확률 P", "ln[P/(1−P)] = b0 + b1·ln(T−Tmin) + b2·ln²(T−Tmin) + b3·ln{1−exp[0.536(T−48)]} + b4·ln(aw−awmin) + b5·ln(1−10^(pHmin−pH)) "
                           "+ b6·ln²(1−10^(pHmin−pH)) + b7·ln(1−[LAC]/…) (Tienungoon 2000, D15 식), 젖산 [LAC]=0 -> 마지막 항 0. "
                           "Scott A·L5 두 균주 각각(F15·F21·F28 계수). 로그 안이 0 이하(T ≤ Tmin, T ≥ 48, aw ≤ awmin, pH ≤ pHmin)는 P 0. "
                           "창 요약: 균주마다 평균·최근 10일 평균·최대"),
          ("G3 환경 검출 확률", "Ivanek 2009 토양 모델(D41·F41·G41): logit(P) = −1.79 + 0.13·Precipitation.2, Precipitation.2 = 채취 2일 전 강수량(mm) "
                             f"-> 날 t 의 P 는 시군구 rain(t−2)(mm, 같은 단위). 창 T−{E.L}…T−1 의 날마다 P 요약(평균·최근 10일 평균·최대). "
                             f"쓰는 강수 = T−{E.L + 2} … T−3. Listeria spp.(L. monocytogenes 아님) 토양 검출 확률")]
    for name, cell, val in G.GROWTH_CELLS:
        it.append((f"생장식 셀: {name}", f"{cell}: {val}"))
    for name, why in G.EXCLUDED_MODELS:
        it.append((f"뺀 모델: {name}", why))
    if selFG.get("note"):
        it.append(("F·G 조합 선정", selFG["note"]))
    else:
        it.append(("F·G 조합 선정 규칙", selFG["rule"]))
        sk = "select_score" if selFG.get("select_rule") == "tint" else "val_pr_auc_w"     # tint = 검증 T 통합 60일평균 PR-AUC
        it.append(("F·G 조합 선정 E00 기준", f"{selFG[f'E00_{sk}_mean']:.6f}"))
        for r in selFG["ranking"]:
            it.append((f"F·G {r['group']}", f"검증 {r[f'{sk}_mean']:.6f} (E00 대비 {r['minus_E00']:+.6f}) -> "
                                           + ("높음(선정)" if r["above_E00"] else "높지 않음")))
        it.append(("F·G 조합 실행 목록", ", ".join(f"{p['id']}({p['kind']})" for p in selFG["plan"])))
    return it


def settings_rows(P, smoke, selk, selA, notes, selFG=None):
    items = [("선정 기준", "검증 구간 가중 PR-AUC (시드 평균). 시험은 기록만" if E.SELECT == "old" else
              f"k*·A 최고·F·G 조합 선정 = {TINT_RULE}. 이 표의 순위 칸은 검증 구간 가중 PR-AUC(기존 칸 그대로)"),
             ("구간 가중", "0.5 x 평균(H1~20) + 0.3 x 평균(H21~40) + 0.2 x 평균(H41~60)"),
             ("판정 기준값", "검증 60개 h 합친 예측에서 F2 최대 확률값(1차 metrics.f2_threshold), 시험에 그대로 적용"),
             ("시드", "smoke: " + str(E.SMOKE_SEEDS) if smoke else str(E.SEEDS)),
             ("표준편차", "시드 간 표본 표준편차(ddof=1). 시드 1개 실험은 표준편차 없음('(시드 1개)')"),
             ("60일 평균 Precision·Recall·F2·TP~TN", "h 별 값(h 마다 혼동행렬)의 평균. 1차 task2_결과표.xlsx 의 시험 Precision·Recall·F2·TP~TN 은 "
              "h 60개를 합친 혼동행렬 값이라 단위가 다름 -> 1차와 비교할 때는 '1차_가중지표' 시트 값을 쓸 것"),
             ("순위", f"'{RANK_COL}': 전체 행을 평가한 실험만(C 제외). 1 = 최고. 시드 1개 실험은 '순위 참고'에 표시"),
             ("평가 행 범위", "C 실험은 그 품목 행만으로 학습·조기 종료·기준값·평가 -> 검출 비율이 높아 전체 실험과 지표를 직접 비교할 수 없음"),
             ("고위험품목비교 판정 기준값", "전용 모델 = 그 품목 검증 행에서, 전체 모델(E00, A 최고) = 전체 모델 자기 검증(전체 행)에서(사양 '각자의 검증에서'). "
              "그래서 Recall·Precision·F2·TP~TN 차이에는 기준값 효과가 섞임. PR-AUC·ROC-AUC 는 기준값과 무관"),
             ("고위험품목비교 추가 줄(기준값 = 그 품목 검증 행에서 다시 구함)",
              "사용자 승인 2026-10-19, 재학습 없음. 묶음마다 전체 모델 E00·A 최고 저장 예측을 그 품목 행으로 거른 뒤 그 품목 검증 행(60개 h 합침)에서 "
              "F2 최대 확률값을 다시 구해 시험에 적용(전용 모델과 같은 기준값 규칙). PR-AUC·ROC-AUC 는 기존 전체 모델 줄과 같고 "
              "Recall·Precision·F2·TP~TN 만 달라짐 -> 전용 모델과 기준값 효과 없이 비교"),
             ("D2 SMOTE 방식", f"{SMOTE_METHOD}(사용자 결정 2026-10-19, 추천안). 이웃: 기준 검출 행과 같은 시도(sido = sgg_code 앞 2자리) 학습 검출 행 중 "
              f"k={E.SMOTE_K}(그 시도 학습 검출 행이 {E.SMOTE_K + 1}개 미만이면 전국 학습 검출 행에서), 거리는 같은 방식으로 그 시도 검출 행끼리만 계산. "
              f"숫자 칸(h + 기상 요약 {E.NW}) = 기준 행 + U(0,1) x (이웃 - 기준 행). 범주 4칸(sido, sgg, p2, p3) = 기준 검출 행 값 그대로. "
              f"만들 개수 = int(불검출 x {E.SMOTE_RATIO} - 검출), 거리(결측 처리)·범주 원-핫 가중·시드 처리는 이전과 같음"),
             ("이력 변수 창", "T-P .. T-1 (기준일 < T 인 시료, 사용자 승인 사양). 시료 결과가 T 에 이미 나와 있었는지(결과 확정 시차)는 보지 않음"),
             ("smoke", "예" if smoke else "아니오")]
    if E.L != E.L_DEFAULT:                                       # 기상 창이 50일이 아닐 때만(50일 표는 이전과 같은 줄)
        items.append(("기상 창", f"T−{E.L} … T−1 ({E.L}일, 기상 요약 {E.NW}칸, EXP2_L={E.L})"))
    if selk:
        tk = selk.get("select_rule") == "tint"
        items.append(("k*", selk["k_star"]))
        items += [(f"k 순위 {i + 1}", f"k={r['k']} 검증{' T 통합 60일평균' if tk else ''} "
                                     f"{r['val_tint_pr_auc_mean60' if tk else 'val_pr_auc_w']:.6f} / 시험 {r['test_pr_auc_w']:.6f}")
                  for i, r in enumerate(selk["ranking"])]
    if selA:
        ta = selA.get("select_rule") == "tint"
        items.append(("A 최고", selA["best"]))
        items += [(f"A 순위 {i + 1}", f"{r['exp']} 검증{' T 통합 60일평균' if ta else ''} "
                                     f"{r['val_tint_pr_auc_mean60_mean' if ta else 'val_pr_auc_w_mean']:.6f} / 시험 {r['test_pr_auc_w_mean']:.6f}")
                  for i, r in enumerate(selA["ranking"])]
    gpath = P["results"] / "groups.json"
    if gpath.exists():
        groups = jload(gpath)
        for g in E.GROUPS:
            if g not in groups:                                  # C4 전에 만든 groups.json
                continue
            G = groups[g]
            items.append((f"{g} {G['desc']}", ", ".join(f"{t['code']}(검사 {t['n']}, 검출 {t['s']}, 검출률 {t['rate']:.4f}, 비중 {t['share']:.4f})"
                                                       for t in G["table"])))
            items.append((f"{g} 학습 검출 시료 합", f"{G['train_pos']} / 전체 {groups['_train_total']['pos']} ({G['train_pos_share']:.4f}), 예상과 같음 {G['matches_expected']}"))
            if g == "C4":
                items.append(("C4 규칙", f"{G.get('rule', '')} -> {len(G['codes'])}개, 학습 시료 {G['train_samples']:,}건 / 전체 학습 시료 "
                                        f"{groups['_train_total']['samples']:,}건 (예상 학습 시료 수와 같음 {G.get('matches_expected_train_samples')}). "
                                        "C4_base·C4_hist = 그 품목 행만으로 학습·조기 종료·기준값·평가(C1~C3 와 같은 틀)"))
                items.append(("C4 밖 검출(이 모델이 다루지 않는 범위)", c4_outside_text(G)))
    items += [(f"D2 표준 SMOTENC 와 다른 점 {i + 1}", d) for i, d in enumerate(DIFF_FROM_STANDARD)]
    for s in (load_summary(P, "D2", sd) for sd in (E.SMOKE_SEEDS if smoke else E.SEEDS)):
        if s is None:
            continue
        si = s.get("smotenc") or {}
        if not si.get("method"):
            items.append((f"D2 시드 {s['seed']}", "[주의] 이전 방식(전국 이웃 + 칸마다 이웃 최빈값 범주)으로 만든 결과"))
            continue
        cc = si.get("cat_consistency") or {}
        nsg = si.get("neighbor_same_group_rate")
        items.append((f"D2 시드 {s['seed']} 시도별 학습 검출 행", ", ".join(f"{g} {n}" for g, n in si["minority_by_group"].items())))
        items.append((f"D2 시드 {s['seed']} 전국 대체 시도",
                      (", ".join(f"{g}(학습 검출 행 {n}개)" for g, n in si["fallback_groups"].items())
                       + f", 그 시도를 기준 행으로 한 새 표본 {si['n_new_from_fallback']}개") if si["fallback_groups"] else "없음"))
        items.append((f"D2 시드 {s['seed']} 합성 행 범주 점검",
                      f"새 표본 {cc.get('n_synthetic')}개: sido = sgg 앞 2자리 {cc.get('sido_eq_sgg2_rate', float('nan')):.1%}, "
                      f"기준 검출 행과 같은 단위 {cc.get('same_as_base_rate', float('nan')):.1%}, "
                      f"학습 검출 단위에 있는 조합 {cc.get('in_train_pos_units_rate', float('nan')):.1%}, "
                      f"이웃이 같은 시도(전국 대체 제외) " + ("해당 없음" if nsg is None else f"{nsg:.1%}")))
    items += fg_settings_rows(selFG or {"note": "F·G 결과 없음"})
    import e2fg
    items += e2fg.wxanom_settings_rows()                         # M1·A3_M1 이동평균 차감 기상(사용자 승인 2026-10-19)
    import e2nb
    items += e2nb.nb_settings_rows()                             # A3_N1·A3_N2·A3_N1N2·A3_N0 제품명 변수(사용자 승인 2026-10-19)
    items += e2fg.rg_settings_rows()                             # Q1~Q4 생장 범위(사용자 지시, [질문] 세션 전달 2026-10-19)
    items += e2fg.p_settings_rows()                              # P1·P2·P3·P1p·P2p 외부 지표(사용자 승인·확정 2026-10-19)
    from e2hist import HX_RULE, hx_dateclock_text                # A3_L1·A3_U1·A3_L1U1 과거 이력 추가 칸(사용자 지시 2026-10-19)
    items += [("HX 과거 이력 추가 칸(A3_L1·A3_U1·A3_L1U1)", HX_RULE), ("HX 날짜 시계 확인", hx_dateclock_text())]
    import e2ae                                                  # S_*·A3_AE*·A3_*_p AE·VAE 이상 탐지(사용자 지시 2026-10-19)
    items += e2ae.settings_rows()
    items += [("XS 작은 실험(E00_R1·E00_NW·A3_NW)", XS_RULE)]       # 사용자 지시 2026-10-19
    items += [("XALL 추가 변수 전부", "단계 'xall', 시드 42·43·44, CatBoost = E00 설정, k = k*(사용자 지시 2026-10-19). ALL_A = E00 + 단계 선택 12묶음 전부"
                                   "(외부 수집·파생변수·이상 점수, AE·VAE 표 입력 = ALL_A 의 나머지 입력), ALL_B = ALL_A + 그날 제품명 n2·n0(" + E.NB_DAY_PREMISE + "), "
                                   "E00_N2·E00_N0 = E00 + 그날 제품명 묶음 하나(같은 전제)")]
    import e2fs                                                  # 양방향 단계 선택(사용자 결정 2026-10-19)
    items += [("FS·BW 양방향 단계 선택", f"단계 'fs'(전진)·'bw'(뒤로), 시드 42·43·44, CatBoost = E00 설정 그대로, k = k*. 규칙: {e2fs.RULE}. "
                                     "전진 = E00 에서 시작해 묶음 순서대로 '지금 기준 + 묶음 하나'(FS01~FS12), 뒤로 = 남긴 묶음을 하나씩 뺀 후보(BW01, …) 중 "
                                     "가장 많이 오른 것을 빼고 반복. " + e2fs.AE_FIXED_NOTE + ". 묶음: "
                                     + " / ".join(f"{b['code']} = {b['name']} {b['n']}칸" for b in E.FS_BUNDLES))]
    import e2stk                                                 # STK_E00·STK_A3 E08 스태킹(사용자 지시 2026-10-19)
    items += e2stk.settings_rows()
    for k, v in notes.items():
        items.append((k, v))
    return pd.DataFrame(items, columns=["항목", "값"]).astype(str)
