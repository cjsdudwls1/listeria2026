# -*- coding: utf-8 -*-
"""E08 스태킹 (사용자 지시 2026-10-19, 계획서 7절 방식 C, 기록표 E08 '스태킹 = 정형 모델 + 시계열 모델') - 단계 'stk', 모델 종류 'stack'.

재학습 없음. 시드 s(42·43·44)마다 같은 시드 기본 모델 3개의 저장 예측을 읽는다:
  CatBoost(E00 또는 A3) + DL_lstmA(LSTM 방식 A) + DL_trfA(트랜스포머 방식 A) 의 results/<ID>__s{s}/pred_val.parquet·pred_test.parquet
  (기본 모델 결과는 다른 실험과 같은 완전 조건 - summary.json complete=True + 파일 5개, e2tables.is_complete).
행 맞추기: 세 파일의 키 5칸(sgg_code, train_prdlst_code, train_prdlst_code_2, d, h)과 y 가 같은지 확인하고 키로 맞춘다(순서만 믿지 않음).
  첫 파일(CatBoost)은 h 우선 순서(h 1..60 블록마다 같은 단위x날짜 행 순서, e2eval.write_preds 형식)인지 확인.
  행 수가 다름·키 중복·키가 다름(빠짐)·같은 키의 y 가 다름 중 하나라도 있으면 StackKeyError 로 멈춤.
  맞춘 순서 = 첫 기본 모델 파일의 행 순서.
메타 모델: 입력 = 각 기본 예측의 logit(clip(p, 1e-6, 1 − 1e-6)) 3칸(float64, 기본 모델 순서),
  sklearn LogisticRegression(기본값 = L2·C 1·lbfgs·절편 있음, max_iter = e2cfg.STK['max_iter'], random_state = 시드),
  학습 = 검증 구간 저장 예측 전체(h 1..60 합침, 행 = (h, 단위x날짜 행), y = 그 행 y) -> 검증·시험에 적용(predict_proba 검출 열, float32 저장).
판정 기준값·지표: 다른 실험과 같은 e2eval.evaluate(검증 h 1~60 합친 예측의 F2 최대 확률값 -> 시험에 그대로).
주의: 계획서는 시간 순서 out-of-fold 예측으로 메타 모델을 학습하길 권하지만, 이번엔 재학습 없이 저장 예측만 쓰므로 메타 모델을 검증 구간으로 학습함
  -> 검증 점수는 메타 학습 구간 점수. 비교에는 시험 점수만(config.json·summary.json·표 비고에 적음, 표 순위에서 뺌).
결과 폴더: config.json·summary.json·metrics_by_h.csv·pred_val.parquet·pred_test.parquet (다른 실험과 같은 형식·같은 is_complete 조건).
  예측 파일 = 첫 기본 모델 예측 파일의 칸·형식·행 순서 그대로에 p 만 메타 예측으로 바꿈(E00 예측 파일과 같은 칸·순서).
run.py 는 이 단계에서 특성 자료(Ctx2)를 만들지 않는다(저장 예측만 읽음).
"""
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import e2cfg as E
import config as C1
from e2eval import evaluate
from e2tables import is_complete, jdump, jload

KEY_COLS = ["sgg_code", "train_prdlst_code", "train_prdlst_code_2", "d", "h"]
CODE_COLS = KEY_COLS[:3]
NOTE = "메타 모델을 검증 구간으로 학습 — 검증 점수는 메타 학습 구간(비교에는 시험 점수만)"
NOTE_DETAIL = ("계획서 7절은 시간 순서 out-of-fold 예측으로 메타 모델을 학습하길 권하지만, 이번엔 기본 모델을 다시 학습하지 않고 저장 예측만 써서 "
               "메타 모델을 검증 구간 저장 예측(h 1~60 합침)으로 학습함. 그래서 검증 지표(구간가중 PR-AUC 등)는 메타 모델이 학습한 자료에서 잰 값이고, "
               "판정 기준값도 같은 검증 예측에서 정함. 다른 실험과의 비교에는 시험 지표만 쓸 것")
RANK_NOTE = "순위 제외: 검증 점수가 메타 학습 구간 점수라 검증 기준 순위에 넣지 않음(시험 점수로만 비교)"
INPUT_RULE = "각 기본 예측의 logit(clip(p, 1e-6, 1 − 1e-6)) (float64, 기본 모델 순서)"
ALIGN_RULE = ("기본 모델 저장 예측의 키 5칸(sgg_code, train_prdlst_code, train_prdlst_code_2, d, h)으로 행을 맞춤(순서만 믿지 않음). "
              "행 수·키 중복·키 빠짐·같은 키의 y 가 하나라도 다르면 멈춤. 맞춘 순서 = 첫 기본 모델(CatBoost) 파일 순서(h 우선)")


class StackKeyError(RuntimeError):
    """기본 모델 저장 예측의 키·y 가 맞지 않음 - 멈춤."""


# ---------------------------------------------------------------- 설명
def base_label(b):
    return f"{b}({E.STK_BASE_DESC.get(b, b)})"


def input_desc(exp):
    """표 '입력' 칸."""
    return "기본 모델 저장 예측 3개의 logit: " + " + ".join(base_label(b) for b in exp["bases"]) + " -> 로지스틱 메타 모델(재학습 없음)"


def n_features_desc(exp):
    return f"숫자 {len(exp['bases'])} (기본 예측 logit: " + ", ".join(exp["bases"]) + "), 범주 없음"


def feature_names(exp):
    return [f"logit_{b}" for b in exp["bases"]]


def table_notes(exp, done):
    """실험기록 비고(완료 시드 summary 목록 done = [(시드, summary)])."""
    out = [NOTE, RANK_NOTE, "기본 모델 = " + " + ".join(base_label(b) for b in exp["bases"]) + " 같은 시드 저장 예측(재학습 없음)"]
    co = [sm.get("meta_coef") for _, sm in done if sm.get("meta_coef")]
    if co:
        names = list(co[0])
        icp = [sm.get("meta_intercept") for _, sm in done if sm.get("meta_intercept") is not None]
        out.append(f"메타 계수 시드 평균({len(co)}개): " + ", ".join(f"{n} {np.mean([c[n] for c in co]):.4f}" for n in names)
                   + (f", 절편 {np.mean(icp):.4f}" if icp else ""))
    return out


def settings_rows():
    """표 '설정' 시트 줄(단계 'stk')."""
    return [("STK 스태킹(STK_E00·STK_A3)", "E08 스태킹(계획서 7절 방식 C, 기록표 E08 '스태킹 = 정형 모델 + 시계열 모델'). 재학습 없음: 시드 42·43·44 마다 "
                                         "같은 시드 기본 모델 3개 저장 예측(results/<ID>__s<시드>/pred_val·pred_test.parquet). "
                                         "STK_E00 = E00 + DL_lstmA + DL_trfA, STK_A3 = A3 + DL_lstmA + DL_trfA. 단계 'stk'(특성 자료를 만들지 않음)"),
            ("STK 메타 모델", f"입력 = {INPUT_RULE} 3칸. sklearn LogisticRegression(기본값: L2, C=1, lbfgs, 절편 있음; max_iter={E.STK['max_iter']}, "
                            "random_state=시드). 학습 = 검증 구간 저장 예측 전체(h 1~60 합침, 행 = (h, 단위x날짜 행)) -> 검증·시험에 적용. "
                            "판정 기준값·지표 = 다른 실험과 같은 e2eval(검증 h 1~60 합친 예측의 F2 최대 확률값 -> 시험에 그대로). 메타 계수 = config.json"),
            ("STK 행 맞추기", ALIGN_RULE),
            ("STK 주의(검증 점수)", NOTE + ". " + NOTE_DETAIL),
            ("STK 순위", RANK_NOTE)]


# ---------------------------------------------------------------- 읽기·행 맞추기
def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def key_frame(t: pa.Table) -> pd.DataFrame:
    """키 5칸을 비교 가능한 값으로: 코드 3칸 = 문자열, d = 날짜 번호(int32), h = int16."""
    out = {c: t.column(c).cast(pa.string()).to_numpy(zero_copy_only=False) for c in CODE_COLS}
    out["d"] = t.column("d").cast(pa.int32()).to_numpy()
    out["h"] = t.column("h").to_numpy().astype(np.int16)
    return pd.DataFrame(out)


def check_layout(t: pa.Table, k: pd.DataFrame, name: str) -> int:
    """h 우선 순서(h 블록 1..60, 블록마다 같은 단위x날짜 행·y 순서)인지. 돌려줌: 블록 행 수 n."""
    N = t.num_rows
    n = N // E.H_MAX
    if n * E.H_MAX != N or n == 0:
        raise StackKeyError(f"{name}: 행 수 {N} 가 60 의 배수가 아님")
    h = k["h"].to_numpy().reshape(E.H_MAX, n)
    if not (h == np.arange(1, E.H_MAX + 1)[:, None]).all():
        raise StackKeyError(f"{name}: h 우선 순서가 아님")
    for c in CODE_COLS + ["d"]:
        a = k[c].to_numpy().reshape(E.H_MAX, n)
        if not (a == a[:1]).all():
            raise StackKeyError(f"{name}: h 블록마다 행 순서가 다름({c})")
    y = t.column("y").to_numpy().reshape(E.H_MAX, n)
    if not (y == y[:1]).all():
        raise StackKeyError(f"{name}: h 블록마다 y 가 다름")
    return n


def align(tables, names):
    """tables[i] = 기본 모델 i 의 예측 표(pyarrow), names[i] = 실험 ID.
    돌려줌: (Pm (k, N) float32 = 첫 표 행 순서로 맞춘 p, y (N,) int8, 점검 dict). 맞지 않으면 StackKeyError."""
    k0 = key_frame(tables[0])
    N = len(k0)
    n = check_layout(tables[0], k0, names[0])
    d0 = int(k0.duplicated().sum())
    if d0:
        raise StackKeyError(f"{names[0]}: 키 중복 {d0}행")
    y0 = tables[0].column("y").to_numpy().astype(np.int8)
    Pm = np.empty((len(tables), N), dtype=np.float32)
    Pm[0] = tables[0].column("p").to_numpy()
    chk = {"n_rows": N, "n_per_h": n, "n_pos_rows": int(y0.sum()), "key_cols": KEY_COLS, "first": names[0], "others": {}}
    left = k0.assign(_i0=np.arange(N, dtype=np.int64))
    for i in range(1, len(tables)):
        nm = names[i]
        if tables[i].num_rows != N:
            raise StackKeyError(f"{nm}: 행 수 {tables[i].num_rows} != {names[0]} {N}")
        ki = key_frame(tables[i])
        di = int(ki.duplicated().sum())
        if di:
            raise StackKeyError(f"{nm}: 키 중복 {di}행")
        m = left.merge(ki.assign(_ii=np.arange(N, dtype=np.int64)), on=KEY_COLS, how="inner")
        if len(m) != N:
            raise StackKeyError(f"{nm}: 키가 {names[0]} 와 다름 - 같은 키 {len(m):,}/{N:,}행")
        perm = np.empty(N, dtype=np.int64)
        perm[m["_i0"].to_numpy()] = m["_ii"].to_numpy()
        yi = tables[i].column("y").to_numpy().astype(np.int8)[perm]
        nbad = int(np.sum(yi != y0))
        if nbad:
            raise StackKeyError(f"{nm}: 같은 키의 y 가 {names[0]} 와 다른 행 {nbad:,}개")
        Pm[i] = tables[i].column("p").to_numpy()[perm]
        chk["others"][nm] = {"key_match_rows": int(len(m)), "y_mismatch_rows": 0,
                             "same_row_order_as_first": bool(np.array_equal(perm, np.arange(N)))}
    return Pm, y0, chk


# ---------------------------------------------------------------- 메타 모델
def logit_features(Pm) -> np.ndarray:
    """(k, N) 확률 -> (N, k) logit(clip(p, c, 1 − c)), float64."""
    c = E.STK["clip"]
    p = np.clip(np.asarray(Pm, dtype=np.float64), c, 1.0 - c)
    return np.ascontiguousarray((np.log(p) - np.log1p(-p)).T)


def make_meta(seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=E.STK["max_iter"], random_state=seed)


def write_stack_preds(path, base_table: pa.Table, p) -> None:
    """첫 기본 모델 예측 표의 칸·형식·행 순서 그대로, p 만 바꿔 저장(e2eval.write_preds 와 같은 zstd)."""
    j = base_table.schema.get_field_index("p")
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    if len(p) != base_table.num_rows:
        raise ValueError("메타 예측 길이가 예측 표 행 수와 다름")
    t = base_table.set_column(j, base_table.schema.field(j), pa.array(p, type=pa.float32()))
    pq.write_table(t, path, compression="zstd")


def compute(exp, seed, base_results):
    """읽기·맞추기·메타 학습·예측. 돌려줌: dict(preds, ys, tabs, model, chk, seconds, base_dirs, ...). 파일은 쓰지 않음."""
    base_results = Path(base_results)
    bdirs = [base_results / E.run_dir_name(b, seed) for b in exp["bases"]]
    miss = [d.name for d in bdirs if not is_complete(d)]
    if miss:
        raise RuntimeError(f"{E.run_dir_name(exp['id'], seed)}: 기본 모델 결과가 완전하지 않음 {miss} ({base_results})")
    t0 = time.time()
    tabs, al = {}, {}
    for sp in ("val", "test"):
        tabs[sp] = [pq.read_table(d / f"pred_{sp}.parquet") for d in bdirs]
        try:
            al[sp] = align(tabs[sp], exp["bases"])
        except StackKeyError as e:
            raise StackKeyError(f"{E.run_dir_name(exp['id'], seed)} {sp}: {e}") from None
    read_seconds = time.time() - t0
    Xv = logit_features(al["val"][0])
    yv = al["val"][1]
    t1 = time.time()
    m = make_meta(seed).fit(Xv, yv)
    fit_seconds = time.time() - t1
    preds, ys = {}, {}
    t2 = time.time()
    for sp in ("val", "test"):
        Pm, y, chk = al[sp]
        X = Xv if sp == "val" else logit_features(Pm)
        p = m.predict_proba(X)[:, 1].astype(np.float32)
        n = chk["n_per_h"]
        preds[sp] = p.reshape(E.H_MAX, n)
        ys[sp] = y[:n]
    pred_seconds = time.time() - t2
    return {"preds": preds, "ys": ys, "tabs": tabs, "model": m, "chk": {sp: al[sp][2] for sp in al}, "bdirs": bdirs,
            "n_meta_rows": int(len(yv)), "n_meta_pos": int(yv.sum()), "read_seconds": read_seconds, "fit_seconds": fit_seconds,
            "pred_seconds": pred_seconds}


def run_one_stack(exp, seed, P, log, mem, smoke, code_md5, base_results=None):
    """run.py run_one 과 같은 결과 폴더(파일 5개, summary complete=True). base_results: 기본 모델 결과를 읽을 폴더(기본 = P['results'])."""
    name = E.run_dir_name(exp["id"], seed)
    rdir = P["results"] / name
    if is_complete(rdir):
        log(f"[건너뜀] {name}: 결과 완전")
        return jload(rdir / "summary.json")
    if rdir.exists():
        log(f"  {name}: 불완전한 이전 결과 삭제 후 다시 실행")
        shutil.rmtree(rdir)
    src = Path(base_results) if base_results is not None else P["results"]
    bnames = [E.run_dir_name(b, seed) for b in exp["bases"]]
    miss = [b for b in bnames if not is_complete(src / b)]
    if miss:
        raise RuntimeError(f"{name}: 기본 모델 결과가 완전하지 않음 {miss} ({src})")
    rdir.mkdir(parents=True)
    log(f"[시작] {name}: {exp['desc']} | 기본 모델 {bnames} (저장 예측, 재학습 없음)")
    t_all = time.time()
    mem.reset()
    r = compute(exp, seed, src)
    m = r["model"]
    t_ev = time.time()
    mdf, thr, summ = evaluate(r["preds"], r["ys"])
    eval_seconds = time.time() - t_ev + r["pred_seconds"]
    memd = mem.result()
    import sklearn
    n_iter = int(np.max(m.n_iter_))
    coef = {nm: float(c) for nm, c in zip(feature_names(exp), m.coef_[0])}
    bsum = {}
    for b, d in zip(exp["bases"], r["bdirs"]):
        s = jload(d / "summary.json")
        bsum[b] = {"run": d.name, "model": s.get("model"), "val_pr_auc_w": s.get("val_pr_auc_w"), "test_pr_auc_w": s.get("test_pr_auc_w"),
                   "pred_md5": {sp: _md5(d / f"pred_{sp}.parquet") for sp in ("val", "test")},
                   "summary_finished_at": s.get("finished_at")}
    info = {"meta_model": "sklearn LogisticRegression", "sklearn_version": sklearn.__version__,
            "meta_params": m.get_params(), "meta_coef": coef, "meta_intercept": float(m.intercept_[0]),
            "n_iter": n_iter, "converged": n_iter < E.STK["max_iter"], "meta_input": INPUT_RULE,
            "meta_train": f"검증 구간 저장 예측 전체(h 1~60 합침): {r['n_meta_rows']:,}행(검출 {r['n_meta_pos']:,})",
            "meta_apply": "검증·시험 저장 예측에 같은 메타 모델 적용(predict_proba 검출 열, float32 저장)",
            "alignment": ALIGN_RULE, "key_check": r["chk"], "bases": bsum, "note": NOTE, "note_detail": NOTE_DETAIL,
            "read_seconds": round(r["read_seconds"], 2), "pred_seconds": round(r["pred_seconds"], 2)}
    rows = {sp: {"n": int(len(r["ys"][sp])), "pos": int(r["ys"][sp].sum())} for sp in ("val", "test")}
    cfg = {"run": name, "exp": exp["id"], "plan": exp["plan"], "desc": exp["desc"], "model": exp["model"],
           "seed": seed, "smoke": smoke, "feats": None, "feats_from": None, "group": None, "group_codes": None, "options": {},
           "bases": list(exp["bases"]), "base_runs": bnames, "stack_note": NOTE,
           "alignment": ALIGN_RULE,
           "splits": {"train": f"d <= {C1.TRAIN_END} (메타 모델은 쓰지 않음 - 기본 모델 학습에만 쓰였음)",
                      "val": f"{C1.VAL_START}~{C1.VAL_END} (메타 모델 학습 + 판정 기준값)",
                      "test": f"{C1.TEST_START}~{C1.TEST_END}"},
           "rows": rows,
           "h_assignment": "기본 모델 저장 예측의 h 1~60 전부(평가). 메타 모델 학습 = 검증 h 1~60 전부 합침",
           "n_features": len(coef), "n_features_desc": n_features_desc(exp), "feature_columns": list(coef), "window_range": None,
           "n_train_rows": r["n_meta_rows"], "n_train_pos": r["n_meta_pos"],
           "n_train_rows_note": "메타 모델 학습 행 = 검증 구간 (h, 단위x날짜 행) 전부",
           **({"wx_L": E.L} if E.L != E.L_DEFAULT else {}),      # 기본 모델의 기상 창 길이(50일이면 칸 없음 - 이전과 같음)
           "code_md5": code_md5(), "model_info": info}
    jdump(cfg, rdir / "config.json")
    mdf.to_csv(rdir / "metrics_by_h.csv", index=False, encoding="utf-8-sig")
    for sp in ("val", "test"):
        write_stack_preds(rdir / f"pred_{sp}.parquet", r["tabs"][sp][0], r["preds"][sp])
    summ_out = {"run": name, "exp": exp["id"], "seed": seed, "model": exp["model"], "plan": exp["plan"],
                "smoke": smoke, "feats": None, "feats_from": None, "group": None, "group_codes": None,
                "n_train_rows": r["n_meta_rows"], "n_train_pos": r["n_meta_pos"],
                **{k: v for k, v in summ.items() if k not in ("val_overall", "test_overall")},
                "threshold_rule": "검증 예측(h 1~60 합침)에서 F2 최대 확률값, 판정 p >= 기준값", **thr,
                "val_overall": summ["val_overall"], "test_overall": summ["test_overall"],
                "best_rounds_or_epoch": n_iter, "n_rounds_trained": n_iter, "early_stop_val_ap": None,
                "fit_seconds": round(r["fit_seconds"], 2), "feature_seconds": round(r["read_seconds"], 2),
                "eval_seconds": round(eval_seconds, 2), "total_seconds": round(time.time() - t_all, 2), "n_features": len(coef), **memd,
                "smotenc": None, "scale_pos_weight": None, "h60": None,
                "bases": list(exp["bases"]), "meta_coef": coef, "meta_intercept": float(m.intercept_[0]), "stack_note": NOTE,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": True}
    jdump(summ_out, rdir / "summary.json")
    log(f"[끝] {name}: 검증 구간가중 PR-AUC {summ['val_pr_auc_w']:.5f}(메타 학습 구간) (60일 평균 {summ['val_pr_auc_mean60']:.5f}), "
        f"시험 {summ['test_pr_auc_w']:.5f}, 기준값 {thr['threshold']:.5f}, 메타 계수 "
        + ", ".join(f"{k} {v:.4f}" for k, v in coef.items()) + f", 절편 {float(m.intercept_[0]):.4f}, 반복 {n_iter}, "
        f"총 {summ_out['total_seconds']:.0f}s, 최대 메모리 private {memd['peak_private_gb']} GB / rss {memd['peak_rss_gb']} GB")
    return summ_out
