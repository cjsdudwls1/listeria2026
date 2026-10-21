# -*- coding: utf-8 -*-
"""3차 실험(시간순 교차검증) 실행. 설정 = config3.json (cfg3.py 머리말), 사양 = 사양_20261020.md.

python run3.py <명령> [공통 옵션] [명령 옵션]
공통: --root 폴더(results·tables·cache·logs·params·tune 를 그 아래에) --config 설정 파일 --L 창 길이 --device cpu|gpu(트리 3종)
      --seed --smoke(시험 실행: 설정 smoke 켬) --data-end YYYY-MM-DD(데이터 끝 날짜 덮어쓰기 - 가짜 경계 점검용)
명령
  vanilla  [--pairs xgb:summary,xgb:raw,...,bilstm:seq] [--folds 1,2,3] [--sub 하위폴더] [--drop-eval-cache]
           바닐라 1단계(기본 = 설정 vanilla_stage1: 트리 3종 x 기상 요약·일별 원값 + BiLSTM) 교차검증 -> results/<모델>__<base|raw|seq>__L<창>__s<시드>[__gpu]/
           (--models m1,m2 [--featset 묶음] 은 기상 요약/변수 묶음 입력으로 - 시간 측정 등)
  window   --pair 모델:입력 --Ls 10,20,...,100   바닐라 2단계: 1단계 1위 하나로 창 길이마다 vanilla 를 따로 실행(e2cfg 가 모듈 상수라 프로세스 분리)
  device   --models xgb,lgbm,cat --folds 3       트리 3종 CPU·GPU 바닐라를 겹 하나로 재서 results/_장치비교/
  kselect  [--model cat]                         스무딩 계수 k 를 교차검증 겹 평균 PR-AUC 로 고름(설정 smoothing_k) -> selection_k.json, tables/k선택.csv
  tune     [--model xgb] [--trials 30] [--featset all]   Optuna(목표 = 설정 optuna_objectives, 기본 겹 평균 PR-AUC 하나) -> tune/<모델>.db,
           tables/튜닝후보_<모델>.csv(상위 optuna_top_n 개 표시). --model 없으면 설정 tune_models 목록 차례로
  choose   --model xgb --trial N                 고른 후보의 파라미터 저장(get_params JSON) -> params/<모델>.json
  drop     --model xgb [--vanilla]               중요도로 묶음 빼기(같은 파라미터, 판단 규칙 = 설정 bundle_drop) -> selection_drop_<모델>.json, tables/묶음빼기_<모델>.csv
  final    --model xgb [--vanilla] [--featset ...]   최종 재학습(학습 자료 전체, 반복 = 세 겹 최적 반복 평균) + 기준값(세 겹 맞혀 보는 행 예측 합친 것에서 Recall≥목표 중 Precision 최대) -> 시험 한 번 채점
           -> results/final_<모델>/, tables/최종_시험_<모델>.csv
  tables                                         바닐라 모델 비교·창 길이 비교·장치 비교 표
  chain    [--trials 30]                         이어 돌리기: 바닐라 1단계(기상 100일) -> 1위 (모델:입력)로 2단계 창 길이 -> 고른 창으로 k 고르기 ->
           튜닝(설정 tune_models, 입력 = 설정 featset_tune) -> 모델별 상위 후보 표 -> 멈춤. 단계마다 저장, 다시 실행하면 이어서(chain_status.json)
  items                                          고위험 품목(겹마다 그 겹 학습 자료, 최종용 = 학습 자료 전체로 검출 비중 상위 top_n) 목록만 -> selection_items.json
  rescore                                        끝난 실행을 저장된 예측으로 지금 기준값 규칙(config3 threshold)에 맞춰 다시 채점(재학습 없음) -> 표 다시
시험 행은 final 에서만 만든다. 다른 명령은 끝날 때 시험 행 접근 기록(cv.TEST_ACCESS)이 비어 있는지 확인해 결과에 적는다.
메모리 보호: 시스템 사용 가능 메모리가 설정 mem_floor_gb(10GB) 아래면 이 프로세스만 멈춤.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SRC = os.path.dirname(os.path.abspath(__file__))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vanilla", "window", "device", "kselect", "tune", "choose", "drop", "final", "tables", "rescore", "chain", "items"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "gpu"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-end", default=None)
    ap.add_argument("--models", default="xgb,lgbm,cat,rf,lr")
    ap.add_argument("--model", default=None)
    ap.add_argument("--featset", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--sub", default="")
    ap.add_argument("--drop-eval-cache", action="store_true")
    ap.add_argument("--Ls", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--vanilla", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="drop: 빼 보기 시도 수 상한(시험 실행용)")
    ap.add_argument("--pairs", default=None, help="vanilla: 모델:입력 쌍(쉼표) - 입력 = summary(기상 요약)·raw(일별 원값)·seq(BiLSTM). 없으면 설정 vanilla_stage1")
    ap.add_argument("--pair", default=None, help="window: 창 길이 비교에 쓸 (모델:입력) 하나 - 1단계 표의 1위")
    return ap.parse_args()


A = parse()
if A.config:
    os.environ["EXP3_CONFIG"] = A.config
if A.L:
    os.environ["EXP2_L"] = str(A.L)
_ov = json.loads(os.environ.get("EXP3_OVERRIDE") or "{}")
if A.smoke:
    _ov.setdefault("smoke", {})["enabled"] = True
if A.data_end:
    _ov.setdefault("dates", {})["data_end"] = A.data_end
if _ov:
    os.environ["EXP3_OVERRIDE"] = json.dumps(_ov, ensure_ascii=False)
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, SRC)

import cfg3 as G     # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import psutil        # noqa: E402

import cv as CV3      # noqa: E402
import engine3 as EN  # noqa: E402
import models3 as M3  # noqa: E402
import metrics as M1  # noqa: E402
from common import Logger  # noqa: E402

SEED = A.seed if A.seed is not None else G.SEED
DEVICE = A.device or G.CFG.get("tree_device", "cpu")
R = Path(A.root).resolve() if A.root else G.EXP3_DIR
P = {k: R / k for k in ("results", "tables", "cache", "logs", "params", "tune")}
for _k in ("results", "tables", "cache", "logs", "params", "tune"):
    P[_k].mkdir(parents=True, exist_ok=True)
LOG = Logger(P["logs"] / "progress3.log")


class MemGuard(threading.Thread):
    def __init__(self, floor):
        super().__init__(daemon=True)
        self.floor = floor
        self.proc = psutil.Process()
        self.peak_rss = 0
        self.min_avail = psutil.virtual_memory().available / 1e9
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                a = psutil.virtual_memory().available / 1e9
                self.min_avail = min(self.min_avail, a)
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
                if a < self.floor:
                    LOG(f"!!!!! 사용 가능 메모리 {a:.1f} GB < {self.floor} GB - 팀 공용 서버 보호로 이 프로세스(pid {os.getpid()})만 멈춤")
                    os._exit(9)
            except Exception:
                pass
            self._halt.wait(2.0)


def folds_arg():
    return [int(x) for x in A.folds.split(",")] if A.folds else [f["fold"] for f in G.FOLDS]


def featset_label(fs):
    b = CV3.resolve_featset(fs)
    for name, lst in CV3.FEATSETS.items():
        if CV3.resolve_featset(lst) == b:
            return name
    import hashlib
    return "fs" + hashlib.md5(",".join(b).encode()).hexdigest()[:6]


def get_k(bundles, allow_missing=False):
    """스무딩 계수 k 와 출처. 설정 select=true 면 selection_k.json(교차검증으로 고른 값), 아니면 fixed."""
    sk = G.CFG["smoothing_k"]
    if not CV3.needs_k(bundles):
        return {"k": int(sk["fixed"]), "used": False, "selected_by_cv": None}
    if sk.get("select"):
        f = R / "selection_k.json"
        if f.exists():
            js = CV3.jload(f)
            return {"k": int(js["k"]), "used": True, "selected_by_cv": True, "source": str(f.name)}
        if not allow_missing:
            raise RuntimeError("설정 smoothing_k.select=true 인데 selection_k.json 이 없음 - 먼저 'run3.py kselect' 실행")
    return {"k": int(sk["fixed"]), "used": True, "selected_by_cv": False, "source": "설정 smoothing_k.fixed"}


def new_cv(log):
    t0 = time.time()
    ctx = CV3.load_ctx(log)
    cvo = CV3.CV(ctx, log, P["cache"], seed=SEED)
    log(f"자료 준비 {time.time() - t0:.1f}s | 경계 {json.dumps(G.BOUNDS, ensure_ascii=False)} | 행 범위 {cvo.keep_desc}")
    return cvo


FD_CATS = {}


def run_config(name, extra):
    """튜닝·묶음 빼기 실행 기록(results/<이름>/config.json) - 범주 입력 점검(selftest3 cat)도 이 파일을 훑음."""
    d = P["results"] / name
    d.mkdir(parents=True, exist_ok=True)
    CV3.jdump(common_cfg(extra), d / "config.json")


def build_folds(cvo, bundles, k, folds, log):
    lc = {}
    fds = {}
    for f in folds:
        t0 = time.time()
        fds[f] = EN.build(cvo, f, bundles, k, log, learn_cache=lc)
        log(f"  겹 {f} 자료: 학습 {fds[f].rs_tr.n:,}행/검출 {int(fds[f].rs_tr.y.sum())}, 맞혀 보는 (행,h) {fds[f].rs_ev.n:,}/검출 "
            f"{int(fds[f].rs_ev.y.sum())}, 숫자 {len(fds[f].names)}칸 + 범주 {len(fds[f].cat_names)}칸 ({time.time() - t0:.1f}s)")
        if fds[f].cat_names[:4] != CV3.BASE_CATS:
            raise RuntimeError(f"범주 입력이 [sido, sgg, p2, p3] 로 시작하지 않음: {fds[f].cat_names}")
        FD_CATS.update({"cats": fds[f].cat_names, "names": fds[f].names})
    return fds


def test_access_check(what):
    n = len(CV3.TEST_ACCESS)
    if n:
        raise RuntimeError(f"{what}: 고르는 단계에서 시험 행을 만든 기록이 있음 {CV3.TEST_ACCESS}")
    return {"test_rows_built": 0, "note": "이 명령은 시험 행을 만들지 않음(cv.TEST_ACCESS 빈 것 확인)"}


def run_name(model, fs_label, L, seed, device):
    return f"{model}__{fs_label}__L{L:03d}__s{seed}" + ("__gpu" if device == "gpu" and model in M3.TREE_BOOST else "")


def common_cfg(extra=None):
    return {"threshold_rule": CV3.THRESHOLD_RULE, "threshold_desc": CV3.THRESHOLD_DESC, "bounds": G.BOUNDS, "config": G.cfg_snapshot(),
            "score_rule": CV3.SCORE_RULE, "code_md5": CV3.code_md5(),
            "python": sys.version.split()[0], "library_versions": lib_versions(), "argv": sys.argv[1:], "seed": SEED,
            "smoke": dict(G.SMOKE), **(extra or {})}


def lib_versions():
    import importlib.metadata as md
    out = {}
    for p in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna", "torch", "pyarrow"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def write_run(rdir, res, fds, model, bundles, k_info, extra=None):
    rdir.mkdir(parents=True, exist_ok=True)
    for k, (p, df) in res["preds"].items():
        CV3.write_pred(rdir / f"pred_fold{k}.parquet", fds[k].rs_ev, p, k)
        df.insert(0, "fold", k)
        df.to_csv(rdir / f"metrics_fold{k}.csv", index=False, encoding="utf-8-sig")
    summ = {k: v for k, v in res.items() if k not in ("preds", "model_info")}
    summ.update({"run": rdir.name, "model_name": M3.MODEL_NAMES[model], "input": (extra or {}).get("input", "summary"), "L": G.L, "seed": SEED,
                 "device": res["model_info"][list(fds)[0]]["device"], "folds": list(fds),
                 "thresholds": {k: res["by_fold"][k]["threshold"] for k in fds}, "best_rounds_mean": float(np.mean(res["best_rounds"])),
                 "threshold_rule": CV3.THRESHOLD_RULE, "score_version": CV3.SCORE_VERSION, "k": k_info, "limitations": CV3.limitations(bundles, k_info),
                 "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "complete": True, **(extra or {})})
    cfg = common_cfg({"model": model, "model_name": M3.MODEL_NAMES[model], "bundles": bundles,
                      "bundle_names": {b: CV3.BUNDLES.get(b, {"name": {"raw": "일별 원값(창 x 19변수)", "seq": "기상 시퀀스(BiLSTM)"}.get(b, b)})["name"] for b in bundles},
                      "feature_columns": fds[list(fds)[0]].names, "cat_columns": fds[list(fds)[0]].cat_names,
                      "model_info": res["model_info"], "k": k_info, "window_range": {k: fd.window for k, fd in fds.items()},
                      "tfidf": {k: fd.tfidf for k, fd in fds.items()}, "ae": {k: fd.ae_info for k, fd in fds.items()},
                      "h_rule": G.BOUNDS["h_rule"], "vanilla": "라이브러리 기본 설정, 불균형 처리 없음(models3.py 머리말)"})
    CV3.jdump(cfg, rdir / "config.json")
    CV3.jdump(summ, rdir / "summary.json")
    return summ


# ---------------------------------------------------------------- 명령
INPUT_LABEL = {"summary": "base", "raw": "raw", "seq": "seq"}
INPUT_NAME = {"summary": "기상 요약", "raw": "일별 원값", "seq": "기상 시퀀스"}


def parse_pairs():
    if A.pairs:
        items = [x for x in A.pairs.split(",") if x]
    elif A.featset and A.featset != "base":
        items = [f"{m}:feat" for m in A.models.split(",") if m]          # 변수 묶음 입력(시간 측정 등)
    elif A.models != "xgb,lgbm,cat,rf,lr" or not G.CFG.get("vanilla_stage1"):
        items = [f"{m}:summary" for m in A.models.split(",") if m]
    else:
        items = list(G.CFG["vanilla_stage1"])
    out = []
    for it in items:
        m, inp = it.split(":")
        assert (m in M3.MODELS and inp in ("summary", "raw", "feat")) or (m == "bilstm" and inp == "seq"), it
        out.append((m, inp))
    return out


def fold_ns(rs_ev, names, cat_names, window):
    from types import SimpleNamespace
    return SimpleNamespace(rs_ev=rs_ev, names=names, cat_names=cat_names, window=window, tfidf=None, ae_info=None)


def finish_run(rdir, m, inp, per_fold, preds, infos, k_info, bundles, fds_ns, t0):
    """원값·BiLSTM 결과를 run_cv 와 같은 모양으로 묶어 저장."""
    ks = sorted(per_fold)
    res = {"model": m, "bundles": bundles, "mean": {kk: float(np.mean([per_fold[k][kk] for k in ks])) for kk in EN.MEAN_KEYS},
           "mean_folds": ks, "by_fold": per_fold, "best_rounds": [per_fold[k]["best_rounds"] for k in ks],
           "fit_seconds": round(sum(per_fold[k]["fit_seconds"] for k in ks), 2), "importance_bundle": {}, "model_info": infos, "preds": preds}
    s_ = write_run(rdir, res, fds_ns, m, bundles, k_info, {"input": inp, "total_seconds": round(time.time() - t0, 1),
                                                           **test_access_check("vanilla")})
    LOG(f"[끝] {rdir.name}: 평균 PR-AUC {s_['mean']['pr_auc']:.5f}, 최적 반복 {s_['best_rounds']}, 학습 합 {s_['fit_seconds']:.0f}s")


def run_raw(cvo, models, folds, base_dir, k_info):
    """일별 원값(19L칸) 트리 3종: 한 겹에서 같이(raw3.py 머리말 - 맞혀 보는 행 입력은 캐시·통째 메모리 없이 h 조각마다)."""
    import raw3
    t0 = time.time()
    lr = cvo.learn_rows()
    X_all = raw3.raw_X(cvo.ctx, lr, G.L)
    LOG(f"  원값 학습 자료 입력 {X_all.shape} ({X_all.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f}s, 메모리만)")
    fb = cvo.builder(["wx"], k_info["k"])
    levels = cvo.cat_levels(list(CV3.BASE_CATS), fb)
    per = {m: {} for m in models}
    preds = {m: {} for m in models}
    infos = {m: {} for m in models}
    fns = {}
    for k in folds:
        rs_tr, rs_ev = cvo.fold_train(k), cvo.eval_rows(k)
        msk = lr.d <= np.datetime64(CV3.fold_def(k)["train_end"], "D")
        assert np.array_equal(rs_tr.pos, lr.pos[msk]) and np.array_equal(rs_tr.h, lr.h[msk])
        X_tr = np.ascontiguousarray(X_all[msk])
        cats_tr = cvo.cats(rs_tr, fb, levels)
        cats_ev = cvo.cats(rs_ev, fb, levels)
        win = cvo.check_windows([rs_tr, rs_ev])
        out = raw3.run_fold(cvo.ctx, rs_tr, rs_ev, X_tr, cats_tr, cats_ev, levels, models, SEED, LOG)
        offs = rs_ev.h_offsets()
        for m in models:
            p, inf = out[m]
            sc, df = CV3.score_fold(rs_ev.y, p, offs)
            sc.update({"fold": k, "n_train_rows": int(rs_tr.n), "n_train_pos": int(rs_tr.y.sum()), "best_rounds": inf["best_rounds"],
                       "n_trained": inf["n_trained_es"], "fit_seconds": inf["fit_seconds"], "predict_seconds": inf["predict_seconds"],
                       "n_num": X_tr.shape[1], "n_cat": 4})
            inf["device"] = "cpu"
            inf["notes"] = {"rule": "최대 반복 학습 + h 조각별 단계 점수로 조기 종료(인내 100) 재현 - raw3.py"}
            per[m][k], preds[m][k], infos[m][k] = sc, (p, df), inf
            LOG(f"    겹 {k} 원값 {m}: PR-AUC {sc['pr_auc']:.5f}, 최적 반복 {inf['best_rounds']}")
        fns[k] = fold_ns(rs_ev, raw3.raw_names(G.L), list(CV3.BASE_CATS), win)
        del X_tr
    for m in models:
        finish_run(base_dir / run_name(m, "raw", G.L, SEED, "cpu"), m, "raw", per[m], preds[m], infos[m], k_info, ["raw"], fns, t0)


def run_bilstm(cvo, folds, base_dir, k_info):
    import dl3
    t0 = time.time()
    fb = cvo.builder(["wx"], k_info["k"])
    levels = cvo.cat_levels(list(CV3.BASE_CATS), fb)
    per, preds, infos, fns = {}, {}, {}, {}
    for k in folds:
        rs_tr, rs_ev = cvo.fold_train(k), cvo.eval_rows(k)
        _, cd_tr = cvo.cats(rs_tr, fb, levels)
        _, cd_ev = cvo.cats(rs_ev, fb, levels)
        win = cvo.check_windows([rs_tr, rs_ev])
        t1 = time.time()
        p, inf = dl3.run_fold(cvo.ctx, rs_tr, rs_ev, cd_tr, cd_ev, CV3.fold_def(k)["train_end"], SEED, LOG)
        sc, df = CV3.score_fold(rs_ev.y, p, rs_ev.h_offsets())
        sc.update({"fold": k, "n_train_rows": int(rs_tr.n), "n_train_pos": int(rs_tr.y.sum()), "best_rounds": inf["best_epoch"],
                   "n_trained": inf["n_epochs"], "fit_seconds": inf["fit_seconds"], "predict_seconds": 0.0, "n_num": 38, "n_cat": 4})
        inf["device"] = "cuda"
        inf["get_params"] = dl3.CFG
        per[k], preds[k], infos[k] = sc, (p, df), inf
        fns[k] = fold_ns(rs_ev, [f"seq_{G.L}x38"], list(CV3.BASE_CATS), win)
        LOG(f"    겹 {k} BiLSTM: PR-AUC {sc['pr_auc']:.5f}, 최고 에폭 {inf['best_epoch']} ({time.time() - t1:.0f}s)")
    finish_run(base_dir / run_name("bilstm", "seq", G.L, SEED, "cpu"), "bilstm", "seq", per, preds, infos, k_info, ["seq"], fns, t0)


def cmd_vanilla(res_dir=None):
    pairs = parse_pairs()
    folds = folds_arg()
    base_dir = res_dir or (P["results"] / A.sub if A.sub else P["results"])
    fs = A.featset or "base"

    def rname(m, inp):
        lab = featset_label(fs) if inp == "feat" else INPUT_LABEL[inp]
        return run_name(m, lab, G.L, SEED, DEVICE if inp in ("summary", "feat") else "cpu")

    todo = [(m, i) for m, i in pairs if not (base_dir / rname(m, i) / "summary.json").exists()]
    LOG(f"vanilla: 할 일 {todo} (건너뜀 {[p for p in pairs if p not in todo]}), 겹 {folds}, 트리 장치 {DEVICE}, 창 {G.L}일")
    if not todo:
        return
    cvo = new_cv(LOG)
    summ_feat = [(m, i) for m, i in todo if i in ("summary", "feat")]
    if summ_feat:
        bundles = CV3.resolve_featset("base" if summ_feat[0][1] == "summary" else fs)
        k_info = get_k(bundles)
        fds = build_folds(cvo, bundles, k_info["k"], folds, LOG)
        for m, i in summ_feat:
            rdir = base_dir / rname(m, i)
            LOG(f"[시작] {rdir.name}")
            t0 = time.time()
            res = EN.run_cv(m, None, fds, bundles, DEVICE, SEED, LOG, keep_preds=True)
            s_ = write_run(rdir, res, fds, m, bundles, k_info, {"input": i, "total_seconds": round(time.time() - t0, 1), **test_access_check("vanilla")})
            LOG(f"[끝] {rdir.name}: 평균 PR-AUC {s_['mean']['pr_auc']:.5f}, 최적 반복 {s_['best_rounds']}, 학습 합 {s_['fit_seconds']:.0f}s")
        del fds
    raw_m = [m for m, i in todo if i == "raw"]
    if raw_m:
        run_raw(cvo, raw_m, folds, base_dir, get_k(["wx"]))
    if ("bilstm", "seq") in todo:
        run_bilstm(cvo, folds, base_dir, get_k(["wx"]))
    if A.drop_eval_cache:
        for f in P["cache"].glob(f"*eval_f*_L{G.L:03d}.*"):
            f.unlink()
            LOG(f"  맞혀 보는 행 캐시 지움: {f.name}")


def cmd_window():
    """2단계: 1단계 1위 (모델:입력) 하나로 창 길이마다 따로 실행(e2cfg 가 모듈 상수라 프로세스 분리). 100 밖의 맞혀 보는 행 캐시는 끝나면 지움."""
    if not A.pair:
        raise RuntimeError("window: --pair 모델:입력 (1단계 표 바닐라_1단계_모델입력비교 의 1위)를 지정")
    Ls = [int(x) for x in A.Ls.split(",")]
    G_root = ["--root", str(R)] + (["--config", A.config] if A.config else []) + (["--smoke"] if A.smoke else []) \
        + (["--data-end", A.data_end] if A.data_end else []) + ["--device", DEVICE, "--seed", str(SEED)]
    for L in Ls:
        cmd = [sys.executable, os.path.join(SRC, "run3.py"), "vanilla", "--pairs", A.pair, "--L", str(L)] + G_root \
            + ([] if L == 100 else ["--drop-eval-cache"])
        LOG(f"window: L={L} 실행 {' '.join(cmd[2:])}")
        r = subprocess.run(cmd, env=dict(os.environ, EXP2_L=str(L)))
        if r.returncode != 0:
            raise RuntimeError(f"window L={L} 실패(종료 코드 {r.returncode})")
    m, inp = A.pair.split(":")
    CV3.jdump({"pair": A.pair, "model": m, "input": inp, "Ls": Ls, "device": DEVICE}, R / "window_models.json")


def cmd_device():
    global DEVICE
    for dev in ("cpu", "gpu"):
        DEVICE = dev
        cmd_vanilla(res_dir=P["results"] / "_장치비교")


def cmd_kselect():
    sk = G.CFG["smoothing_k"]
    model = A.model or sk.get("model", "cat")             # 이어 돌리기: 바닐라 2단계 1위 모델(--model)·입력(기상 요약 + 과거 이력 핵심 칸)
    bundles = CV3.resolve_featset(A.featset or sk.get("featset", ["wx", "hist_core"]))
    folds = folds_arg()
    cvo = new_cv(LOG)
    rows = []
    for k in sk["candidates"]:
        LOG(f"kselect: k={k}, 모델 {model}, 입력 {bundles}")
        fds = build_folds(cvo, bundles, k, folds, LOG)
        res = EN.run_cv(model, None, fds, bundles, DEVICE, SEED, LOG)
        rows.append({"k": k, **{f"겹{f} PR-AUC": res["by_fold"][f]["pr_auc"] for f in folds}, "평균 PR-AUC": res["mean"]["pr_auc"],
                     f"평균 P@R≥{CV3.RECALL_MIN:.2f}": res["mean"]["p_at_r"], "최적 반복": "/".join(map(str, res["best_rounds"])), "학습 시간(초)": res["fit_seconds"]})
        del fds
    df = pd.DataFrame(rows)
    judge = "평균 PR-AUC" if sk.get("judge", "pr_auc") == "pr_auc" else f"평균 P@R≥{CV3.RECALL_MIN:.2f}"
    best = df.sort_values([judge, "k"], ascending=[False, True]).iloc[0]
    write_table(df, P["tables"] / "k선택", f"스무딩 계수 k 고르기: 모델 {M3.MODEL_NAMES[model]}(바닐라), 입력 {bundles}, 창 {G.L}일, 판단 = {judge}(같으면 작은 k). "
                "BiLSTM 이면 기상 = 시퀀스, 과거 이력 핵심 칸 = 정적 입력(머리에 붙임). 시험 미사용")
    CV3.jdump({"k": int(best["k"]), "judge": judge, "table": rows, "model": model, "bundles": bundles, "L": G.L, "folds": folds,
               "rule": "바닐라 2단계 1위 (모델, 입력)에 과거 이력 핵심 칸(hist_core)을 붙여 교차검증 세 겹 평균 PR-AUC 최대(같으면 작은 k)",
               "bounds": G.BOUNDS, **test_access_check("kselect")}, R / "selection_k.json")
    LOG(f"kselect 끝: k = {int(best['k'])} ({judge} {best[judge]:.5f})")


def tune_study(model, bundles, k):
    import optuna
    obj = CV3.OBJECTIVES
    sig = featset_label(bundles) + f"_k{k}_L{G.L}_s{SEED}_obj-{'-'.join(obj)}" + ("_gpu" if DEVICE == "gpu" and model in M3.TREE_BOOST else "")
    st = optuna.create_study(directions=["maximize"] * len(obj), study_name=f"{model}__{sig}",
                             storage=f"sqlite:///{(P['tune'] / f'{model}.db').as_posix()}", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=int(G.CFG["optuna"]["sampler_seed"])))
    return st, sig


def cmd_tune():
    models = [A.model] if A.model else (G.CFG.get("tune_models") or [])
    if not models:
        raise RuntimeError("튜닝 모델 목록 미정 - config3.json tune_models 에 적거나 --model 로 지정")
    for m in models:
        tune_one(m)


def tune_one(model):
    import optuna
    bundles = CV3.resolve_featset(A.featset or "all")
    k_info = get_k(bundles)
    folds = folds_arg()
    n_trials = A.trials or int(G.CFG["optuna"]["n_trials"])
    space = G.CFG["optuna"]["spaces"][model]
    st, sig = tune_study(model, bundles, k_info["k"])
    done = len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])
    LOG(f"tune {model}: 입력 {bundles}, k {k_info}, 시도 {n_trials} (끝난 것 {done}), 겹 {folds}, 장치 {DEVICE}")
    if done < n_trials:
        cvo = new_cv(LOG)
        fds = build_folds(cvo, bundles, k_info["k"], folds, LOG)
        (P["tune"] / f"{model}.fds_ready").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")   # chain: 입력 캐시·이상치 점수 준비 끝 표시
        if model == "bilstm":                              # BiLSTM 은 기상 요약 칸을 쓰지 않음(시퀀스로) -> 메모리에서 버림
            for fd in fds.values():
                fd.drop_bundle_columns({"wx"})
            import gc
            gc.collect()
            LOG(f"  BiLSTM: 기상 요약 칸 버림 -> 숫자 {len(fds[folds[0]].names)}칸(메모리)")

        def objective(trial):
            params = M3.suggest(trial, space)
            t0 = time.time()
            res = EN.run_cv(model, params, fds, bundles, DEVICE, SEED, LOG, quiet=True)
            m = res["mean"]
            for kk, v in m.items():
                trial.set_user_attr(kk, v)
            trial.set_user_attr("best_rounds", res["best_rounds"])
            trial.set_user_attr("fit_seconds", res["fit_seconds"])
            trial.set_user_attr("by_fold", {str(f): {"pr_auc": res["by_fold"][f]["pr_auc"], "p_at_r": res["by_fold"][f]["p_at_r"], "f2": res["by_fold"][f]["f2"],
                                                     "threshold": res["by_fold"][f]["threshold"]} for f in fds})
            trial.set_user_attr("params_full", params)
            LOG(f"  시도 {trial.number}: PR-AUC {m['pr_auc']:.5f}, P@R≥{CV3.RECALL_MIN:.2f} {m['p_at_r']:.5f}, 최적 반복 {res['best_rounds']}, {time.time() - t0:.0f}s, {params}")
            vals = tuple(m[o] for o in CV3.OBJECTIVES)
            return vals[0] if len(vals) == 1 else vals

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st.optimize(objective, n_trials=n_trials - done)
    test_access_check("tune")
    tune_table(model, st, sig, bundles, k_info)
    run_config(f"tune_{model}", {"model": model, "study": sig, "bundles": bundles, "k": k_info, "space": space,
                                  "cat_columns": FD_CATS.get("cats"), "feature_columns": FD_CATS.get("names")})


def tune_table(model, st, sig, bundles, k_info):
    import optuna
    obj = CV3.OBJECTIVES
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_n = int(G.CFG.get("optuna_top_n", 5))
    if len(obj) == 1:
        mark = {t.number for t in sorted(done, key=lambda t: -t.value)[:top_n]}
        lab, how = f"PR-AUC 상위 {top_n}", f"단일 목표 {obj[0]}(겹 평균) - 상위 {top_n}개 표시"
    else:
        mark = {t.number for t in st.best_trials}
        lab, how = "파레토 앞줄", f"다목적 {obj}(겹 평균) - 파레토 앞줄 = 한 점수를 더 올릴 수 없는 후보"
    rows = []
    for t in done:
        u = t.user_attrs
        rows.append({"시도": t.number, lab: "예" if t.number in mark else "", "평균 PR-AUC": u["pr_auc"],
                     f"평균 P@R≥{CV3.RECALL_MIN:.2f}(60개 h 합친 행)": u["p_at_r"], "평균 기준값 Recall(합친 행)": u["r_at_thr"],
                     "평균 FP(h 당)": u["fp"], "평균 FN(h 당)": u["fn"], "평균 TP(h 당)": u["tp"], "평균 TN(h 당)": u["tn"], "평균 N(h 당)": u["n"],
                     "평균 검사축소율": u["inspect_reduction"], "평균 Recall(h 평균)": u["recall"], "평균 Precision(h 평균)": u["precision"],
                     "평균 F1": u["f1"], "평균 Accuracy": u["accuracy"], **{f"상위 {k}% 포착률": u[f"top{k}"] for k in CV3.TOPK},
                     "평균 ROC-AUC": u["roc_auc"], "평균 F2(최대, 참고)": u.get("f2"), "최적 반복(겹별)": "/".join(map(str, u["best_rounds"])),
                     "학습 시간(초, 세 겹 합)": u["fit_seconds"], "파라미터": json.dumps(u["params_full"], ensure_ascii=False)})
    df = pd.DataFrame(rows).sort_values([lab, "평균 PR-AUC"], ascending=[False, False])
    write_table(df, P["tables"] / f"튜닝후보_{model}",
                f"Optuna {how}. {M3.MODEL_NAMES[model]}, 연구 {sig}, 입력 {bundles}, k {k_info}. 기준값(오탐·놓침 칸) = {CV3.THRESHOLD_RULE}. "
                "사용자가 고른 뒤 'run3.py choose --model 모델 --trial 번호'. 시험 미사용")
    LOG(f"표: 튜닝후보_{model} ({len(df)}줄, {lab} {len(mark)}개)")


def cmd_choose():
    import optuna
    model = A.model
    bundles = CV3.resolve_featset(A.featset or "all")
    k_info = get_k(bundles)
    st, sig = tune_study(model, bundles, k_info["k"])
    t = [x for x in st.trials if x.number == A.trial][0]
    params = M3.suggest(optuna.trial.FixedTrial(t.params), G.CFG["optuna"]["spaces"][model])
    if model == "bilstm":
        import dl3
        obj = {"model": "bilstm", "params": {**dl3.CFG, **params}, "not_saved": [], "trial": t.number, "study": sig, "bundles": bundles,
               "k": k_info, "L": G.L, "seed": SEED, "scores": dict(t.user_attrs), "chosen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        CV3.jdump(obj, P["params"] / "bilstm.json")
        LOG(f"choose: bilstm 시도 {t.number} 파라미터 저장 -> params/bilstm.json")
        return
    m = M3.make_model(model, SEED, DEVICE, LOG, params)
    obj = M3.save_params(P["params"] / f"{model}.json", m, {"trial": t.number, "study": sig, "bundles": bundles, "k": k_info,
                                                           "L": G.L, "seed": SEED, "device": DEVICE, "suggested": params,
                                                           "scores": dict(t.user_attrs), "chosen_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    LOG(f"choose: {model} 시도 {t.number} 파라미터 저장 -> params/{model}.json ({len(obj['params'])}칸, 저장 안 한 칸 {obj['not_saved']})")


def chosen_params(model):
    if A.vanilla:
        return None, "바닐라(기본 설정)"
    f = P["params"] / f"{model}.json"
    if not f.exists():
        raise RuntimeError(f"params/{model}.json 없음 - 'run3.py choose' 먼저(또는 --vanilla)")
    return M3.load_params(f), f"params/{model}.json"


def drop_metrics():
    return list(G.CFG["bundle_drop"].get("metrics") or CV3.OBJECTIVES)


def accept(base, cand, rule, tol, metrics=None):
    metrics = metrics or drop_metrics()
    d = [cand[m] - base[m] for m in metrics]
    if rule == "no_decrease_any_increase":
        return all(x >= -tol for x in d) and any(x > tol for x in d)
    if rule == "all_increase":
        return all(x > tol for x in d)
    raise ValueError(rule)


def cmd_drop():
    model = A.model
    params, psrc = chosen_params(model)
    parent = CV3.resolve_featset(A.featset or "all")
    k_info = get_k(parent)
    folds = folds_arg()
    bd = G.CFG["bundle_drop"]
    rule, tol, max_passes = bd["rule"], float(bd.get("tol", 0.0)), int(bd.get("max_passes", 20))
    n1_folds = bd.get("n1_judge_folds")
    cvo = new_cv(LOG)
    fds = build_folds(cvo, parent, k_info["k"], folds, LOG)

    def means(res, ks):
        return {kk: float(np.mean([res["by_fold"][f][kk] for f in ks])) for kk in sorted(set(["pr_auc", "p_at_r"] + drop_metrics()))}

    imp_kind = bd.get("importance", "permutation")
    keep = model != "bilstm"

    def with_imp(res, bl):
        """회차 시작 기준 모델의 묶음 중요도: permutation = 묶음 섞기(맞혀 보는 행, PR-AUC 감소, 세 겹 평균), model = 모델 내장(기록용)."""
        if imp_kind != "permutation":
            return res, res["importance_bundle"]
        if "perm_importance" not in res:
            res = EN.perm_from_models(res, fds, bl, LOG) if keep else EN.run_cv(model, params, fds, bl, DEVICE, SEED, LOG, perm=True)
        return res, res["perm_importance"]

    cur = list(parent)
    base = EN.run_cv(model, params, fds, cur, DEVICE, SEED, LOG, keep_models=keep)
    hist = [{"단계": 0, "회차": 0, "시도한 묶음": "(시작: 전부)", "묶음 섞기 중요도": None, "기준 PR-AUC": base["mean"]["pr_auc"],
             "기준 P@R": base["mean"]["p_at_r"], "후보 PR-AUC": None, "후보 P@R": None, "결정": "시작", "남은 묶음": ",".join(cur)}]
    step = 0
    imp_log = []
    for ps in range(1, max_passes + 1):
        base, imp = with_imp(base, cur)
        imp_log.append({"회차": ps, "importance": imp, "by_fold": base.get("perm_by_fold"), "rule": base.get("perm_rule")})
        order = sorted([b for b in cur if b != "base"], key=lambda b: (imp.get(b, 0.0), b))   # 범주 4칸·h 는 후보 아님
        LOG(f"drop 회차 {ps}: 순서(중요도 낮은 것부터) {[(b, round(imp.get(b, 0.0), 5)) for b in order]}")
        acc_any = False
        for b in order:
            if b not in cur or len(cur) <= 1:
                continue
            if A.max_steps is not None and step >= A.max_steps:
                break
            step += 1
            cand = [x for x in cur if x != b]
            r = EN.run_cv(model, params, fds, cand, DEVICE, SEED, LOG, quiet=True, keep_models=keep)
            ks = n1_folds if (b == "n1" and n1_folds) else folds
            mb, mc = means(base, ks), means(r, ks)
            ok = accept(mb, mc, rule, tol)
            hist.append({"단계": step, "회차": ps, "시도한 묶음": b, "묶음 섞기 중요도": imp.get(b, 0.0), "기준 PR-AUC": mb["pr_auc"],
                         "기준 P@R": mb["p_at_r"], "후보 PR-AUC": mc["pr_auc"], "후보 P@R": mc["p_at_r"], "판단 겹": ",".join(map(str, ks)),
                         "결정": "뺌" if ok else "되돌림", "남은 묶음": ",".join(cand if ok else cur)})
            LOG(f"  단계 {step}: {b} 빼기 -> PR-AUC {mb['pr_auc']:.5f}->{mc['pr_auc']:.5f}, P@R {mb['p_at_r']:.5f}->{mc['p_at_r']:.5f}: {'뺌' if ok else '되돌림'}")
            if ok:
                cur, base, acc_any = cand, r, True
            else:
                r.pop("models", None)
        if not acc_any or (A.max_steps is not None and step >= A.max_steps):
            break
    df = pd.DataFrame(hist)
    write_table(df, P["tables"] / f"묶음빼기_{model}",
                f"중요도로 묶음 빼기: {M3.MODEL_NAMES[model]}, 파라미터 {psrc}, 시작 = {parent}, 판단 점수 = {drop_metrics()}, 규칙 = {rule}(허용 {tol}), "
                f"회차마다 그때 기준 모델의 묶음 섞기 중요도(맞혀 보는 행에서 묶음 칸을 통째로 섞었을 때 PR-AUC 감소, 세 겹 평균; 씨앗·범위 = 설정 perm_seed·perm_scope) 낮은 것부터 하나씩 빼 보고 "
                f"좋아지면 뺀 채로, 아니면 되돌림. 한 회차에서 뺀 것이 없으면 끝. 이상치 점수는 처음 계산한 점수 고정. 시험 미사용")
    CV3.jdump({"model": model, "params_source": psrc, "parent": parent, "final_bundles": cur, "rule": rule, "tol": tol, "metrics": drop_metrics(),
               "importance_kind": imp_kind, "importance_by_round": imp_log,
               "n1_judge_folds": n1_folds, "history": hist, "final_mean": base["mean"], "final_best_rounds": base["best_rounds"],
               "k": k_info, "limitations": CV3.limitations(cur, k_info), **test_access_check("drop")}, R / f"selection_drop_{model}.json")
    run_config(f"drop_{model}", {"model": model, "params_source": psrc, "parent": parent, "final_bundles": cur, "k": k_info,
                                  "cat_columns": FD_CATS.get("cats"), "feature_columns": FD_CATS.get("names")})
    LOG(f"drop 끝: 남은 묶음 {cur}")


def cmd_final():
    model = A.model
    params, psrc = chosen_params(model)
    if A.featset:
        bundles = CV3.resolve_featset(A.featset)
        bsrc = "명령 옵션 --featset"
    elif (R / f"selection_drop_{model}.json").exists():
        bundles = CV3.jload(R / f"selection_drop_{model}.json")["final_bundles"]
        bsrc = f"selection_drop_{model}.json"
    else:
        raise RuntimeError("입력 묶음을 모름 - drop 먼저 또는 --featset")
    # 부모 묶음: 묶음 빼기 결과를 쓰면 그 출발 묶음(이상치 점수의 표 입력이 묶음 빼기 때와 같게), --featset 이면 그 묶음 그대로
    parent = CV3.resolve_featset(CV3.jload(R / f"selection_drop_{model}.json")["parent"]) if bsrc.startswith("selection_drop") else bundles
    k_info = get_k(parent)
    folds = folds_arg()
    cvo = new_cv(LOG)
    fds = build_folds(cvo, parent, k_info["k"], folds, LOG)
    res = EN.run_cv(model, params, fds, bundles, DEVICE, SEED, LOG, keep_preds=True)
    assert not CV3.TEST_ACCESS
    y_all = np.concatenate([fds[f].rs_ev.y for f in folds])
    p_all = np.concatenate([res["preds"][f][0] for f in folds])
    thr = CV3.threshold_at_recall(y_all, p_all)
    rounds = int(round(float(np.mean(res["best_rounds"])))) if model in M3.TREE_BOOST + ("bilstm",) else None
    LOG(f"final: 교차검증 평균 PR-AUC {res['mean']['pr_auc']:.5f}, P@R {res['mean']['p_at_r']:.5f}, 최적 반복 {res['best_rounds']} -> 최종 반복 {rounds}, "
        f"기준값(세 겹 합친 행 {CV3.THRESHOLD_RULE}) {thr['threshold']:.6f} (합친 Precision {thr['precision']:.5f}, Recall {thr['recall']:.5f})")
    rdir = P["results"] / f"final_{model}"
    rdir.mkdir(parents=True, exist_ok=True)
    for f in folds:                                        # 세 겹 맞혀 보는 행 예측(기준값 출처) 저장 - 다시 채점(rescore)용
        CV3.write_pred(rdir / f"pred_cv_fold{f}.parquet", fds[f].rs_ev, res["preds"][f][0], f)
    del fds
    CV3.ALLOW_TEST = True                                  # 시험 행은 여기(최종)에서만 만든다
    fdF = EN.build(cvo, "final", parent, k_info["k"], LOG)
    tr, te = fdF.view(bundles)
    if model == "bilstm":                                  # 최종: 정한 에폭 수(세 겹 최고 에폭 평균)만큼, 시험 행 평가 없이
        ep = int(round(float(np.mean(res["best_rounds"]))))
        _, df_, p, inf = EN._bilstm_fold("final", fdF, bundles, {**(params or {}), "fixed_epochs": ep}, SEED, LOG)
        s, df = CV3.score_test(te.y, p, fdF.offs, thr["threshold"])
        rounds = ep
        m = None
    else:
        m = M3.make_model(model, SEED, DEVICE, LOG, params)
    if m is None:
        pass
    else:
        if model == "cat" and (params is None or "learning_rate" not in params):
            m.fixed_lr = float(np.mean([res["model_info"][f]["notes"]["learning_rate_used"] for f in folds]))
        m.fit(tr, rounds=rounds) if model in M3.TREE_BOOST else m.fit(tr)
        p = m.predict(te)
        s, df = CV3.score_test(te.y, p, fdF.offs, thr["threshold"])
    CV3.write_pred(rdir / "pred_test.parquet", fdF.rs_ev, p, 0)
    df.to_csv(rdir / "metrics_test_by_h.csv", index=False, encoding="utf-8-sig")
    summ = {"threshold_rule": CV3.THRESHOLD_RULE, "score_version": CV3.SCORE_VERSION, "model": model, "params_source": psrc, "bundles": bundles, "bundles_source": bsrc, "k": k_info, "rounds": rounds,
            "cv_best_rounds": res["best_rounds"], "cv_mean": res["mean"], "threshold": thr, "test": s,
            "cat_learning_rate_fixed": getattr(m, "fixed_lr", None) if m is not None else None, "test_access": list(CV3.TEST_ACCESS),
            "limitations": CV3.limitations(bundles, k_info), "train_rows": int(tr.n), "train_pos": int(tr.y.sum()),
            "test_pairs": int(te.n), "test_pos_pairs": int(te.y.sum()), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    CV3.jdump(summ, rdir / "summary.json")
    CV3.jdump(common_cfg({"model": model, "get_params": (m.info()["get_params"] if m is not None else params), "bundles": bundles, "feature_columns": tr.names,
                          "cat_columns": tr.cat_names, "ae": fdF.ae_info, "tfidf": fdF.tfidf, "window_range": fdF.window}),
              rdir / "config.json")
    row = {"모델": M3.MODEL_NAMES[model], "Accuracy": s["accuracy"], "Precision": s["precision"], "Recall": s["recall"],
           "F1": s["f1"], "PR-AUC": s["pr_auc"], "ROC-AUC": s["roc_auc"], "TP(h 당)": s["tp"], "FN(h 당)": s["fn"],
           "FP(h 당)": s["fp"], "TN(h 당)": s["tn"], "N(h 당)": s["n"], "검사축소율": s["inspect_reduction"],
           **{f"상위 {k}% 포착률": s[f"top{k}"] for k in CV3.TOPK}, "기준값": thr["threshold"], "반복 수": rounds,
           "입력 묶음": ",".join(bundles), "파라미터": psrc, "한계": " / ".join(CV3.limitations(bundles, {**k_info}))}
    write_table(pd.DataFrame([row]), P["tables"] / f"최종_시험_{model}",
                f"최종: 학습 자료 전체로 재학습(부스팅 반복 = 세 겹 최적 반복 평균), 기준값 = 세 겹 맞혀 보는 행 예측 합친 것에서 {CV3.THRESHOLD_RULE}, "
                "시험 한 번 채점. 점수 = h 1~60 평균, 건수 = h 당 평균. 검사축소율 = 1 − (TP+FP+FN)/N")
    LOG(f"final 끝: 시험 Precision {s['precision']:.5f}, Recall {s['recall']:.5f}, PR-AUC {s['pr_auc']:.5f}, 검사축소율 {s['inspect_reduction']:.4f}")


# ---------------------------------------------------------------- 표
NOTE = (f"기준값 규칙 = {CV3.THRESHOLD_RULE}: " + CV3.THRESHOLD_DESC + ". 점수 = 겹마다 계산해 겹 평균(표준편차 없음). "
        "PR-AUC·ROC-AUC = h 별 값의 h 1~60 평균. P@R = 그 기준값에서 60개 h 합친 Precision(고르기 목표). Accuracy·Precision·Recall·F1·검사축소율"
        "(= 1 − (TP+FP+FN)/N)·상위 k% = 그 기준값에서 h 마다 -> h 평균. TP·FP·FN·TN·N = h 당 평균. 학습 시간 = 겹 학습(fit) 합(입력 만들기·예측 제외)")


def write_table(df, path, note):
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(path.with_suffix(".xlsx")) as xw:
        df.to_excel(xw, sheet_name="표", index=False)
        pd.DataFrame({"설명": [note, NOTE]}).to_excel(xw, sheet_name="규칙", index=False)


def table_row(s):
    bf = {int(k): v for k, v in s["by_fold"].items()}
    r = {"모델": s["model_name"], "입력": INPUT_NAME.get(s.get("input", "summary"), s.get("input")), "기상 창(일)": s["L"], "계산 장치": s["device"]}
    for k in sorted(bf):
        r[f"겹{k} PR-AUC"] = bf[k]["pr_auc"]
    r["평균 PR-AUC"] = s["mean"]["pr_auc"]
    for k in sorted(bf):
        r[f"겹{k} P@R≥{CV3.RECALL_MIN:.2f}"] = bf[k]["p_at_r"]
    r.update({f"평균 P@R≥{CV3.RECALL_MIN:.2f}": s["mean"]["p_at_r"], "평균 F2(최대, 참고)": s["mean"].get("f2"), "평균 Accuracy": s["mean"]["accuracy"],
              "평균 Precision": s["mean"]["precision"], "평균 Recall": s["mean"]["recall"], "평균 F1": s["mean"]["f1"],
              **{f"평균 {c.upper()}(h 당)": s["mean"][c] for c in ("tp", "fn", "fp", "tn", "n")},
              "평균 검사축소율": s["mean"]["inspect_reduction"], "평균 ROC-AUC": s["mean"]["roc_auc"],
              **{f"상위 {k}% 포착률": s["mean"][f"top{k}"] for k in CV3.TOPK},
              **{f"겹{k} 기준값": bf[k]["threshold"] for k in sorted(bf)},
              "최적 반복(겹별)": "/".join(map(str, s["best_rounds"])) if s["model"] in M3.TREE_BOOST + ("bilstm",) else "-",
              "학습 시간(초, 겹 합)": s["fit_seconds"], "기준값 규칙": s.get("threshold_rule", "?"), "실행 이름": s["run"]})
    return r


def collect(base):
    out = []
    if base.exists():
        for d in sorted(base.iterdir()):
            f = d / "summary.json"
            if d.is_dir() and f.exists():
                s = CV3.jload(f)
                if s.get("complete"):
                    out.append(s)
    return out


def cmd_tables():
    order = {m: i for i, m in enumerate(M3.MODELS + ["bilstm"])}
    allr = [s_ for s_ in collect(P["results"]) if "by_fold" in s_ and s_.get("folds") and len(s_["folds"]) == len(G.FOLDS)]
    runs = [s_ for s_ in allr if s_.get("threshold_rule") == CV3.THRESHOLD_RULE]
    if len(runs) < len(allr):
        LOG(f"표: 기준값 규칙이 지금({CV3.THRESHOLD_RULE})과 다른 실행 {len(allr) - len(runs)}개는 뺌 - 'run3.py rescore' 로 다시 채점하면 들어감")
    stage1 = set(G.CFG.get("vanilla_stage1") or [])
    st1 = [s_ for s_ in runs if s_["L"] == 100 and s_["seed"] == SEED and f"{s_['model']}:{s_.get('input', 'summary')}" in stage1
           and s_["run"] == run_name(s_["model"], INPUT_LABEL[s_.get("input", "summary")], 100, SEED, DEVICE if s_.get("input", "summary") == "summary" else "cpu")]
    if st1:
        df = pd.DataFrame([table_row(s_) for s_ in st1]).sort_values("평균 PR-AUC", ascending=False)
        df.insert(0, "PR-AUC 순위", range(1, len(df) + 1))
        write_table(df, P["tables"] / "바닐라_1단계_모델입력비교",
                    "바닐라 1단계(기상 100일): 트리 3종 x (기상 요약·일별 원값) + BiLSTM, 기본 설정, 불균형 처리 없음. 고르기 = 교차검증 3겹 평균 PR-AUC")
        LOG(f"표: 바닐라_1단계_모델입력비교 ({len(df)}줄)")
    wf = R / "window_models.json"
    if wf.exists():
        js = CV3.jload(wf)
        m, inp = js["model"], js["input"]
        rows = [s_ for s_ in runs if s_["model"] == m and s_.get("input", "summary") == inp
                and s_["run"] == run_name(m, INPUT_LABEL[inp], s_["L"], SEED, js["device"] if inp == "summary" else "cpu")]
        if rows:
            df = pd.DataFrame([table_row(s_) for s_ in sorted(rows, key=lambda s_: s_["L"])])
            write_table(df, P["tables"] / "창길이비교", f"바닐라 2단계 창 길이 비교({M3.MODEL_NAMES[m]}, {INPUT_NAME[inp]}, 창 {js['Ls']}). 고르기 = 교차검증 3겹 평균 PR-AUC")
            LOG(f"표: 창길이비교 ({len(df)}줄)")
    dev = [s_ for s_ in collect(P["results"] / "_장치비교") if s_.get("threshold_rule") == CV3.THRESHOLD_RULE]
    if dev:
        recs = []
        for s_ in sorted(dev, key=lambda s_: (order[s_["model"]], s_["device"])):
            for k, bf in s_["by_fold"].items():
                recs.append({"모델": s_["model_name"], "계산 장치": s_["device"], "겹": int(k), "PR-AUC": bf["pr_auc"],
                             f"P@R≥{CV3.RECALL_MIN:.2f}": bf["p_at_r"], "상위 10% 포착률": bf["top10"], "최적 반복": bf["best_rounds"],
                             "학습한 반복": bf["n_trained"], "학습 시간(초)": bf["fit_seconds"], "예측 시간(초)": bf["predict_seconds"],
                             "반복당 학습 시간(초)": round(bf["fit_seconds"] / max(bf["n_trained"], 1), 3), "실행 이름": s_["run"]})
        write_table(pd.DataFrame(recs), P["tables"] / "장치비교", "트리 3종 CPU/GPU(겹 3, 바닐라, 범주 4 + h + 기상 요약 100일). 학습 시간 = fit(조기 종료 지표 계산 포함)")
        LOG(f"표: 장치비교 ({len(recs)}줄)")


def _read_pred(f):
    import pyarrow.parquet as pq
    t = pq.read_table(f, columns=["h", "y", "p"])
    return t.column("h").to_numpy().astype(np.int64), t.column("y").to_numpy().astype(np.int8), t.column("p").to_numpy()


TRAIN_KEYS = ("fold", "n_train_rows", "n_train_pos", "best_rounds", "n_trained", "fit_seconds", "predict_seconds", "n_num", "n_cat")


def cmd_rescore():
    """results 아래(하위 폴더 포함) 끝난 실행을 저장 예측(pred_fold*.parquet, final 은 pred_cv_fold*·pred_test)으로 다시 채점."""
    n = 0
    for sf in sorted(P["results"].glob("**/summary.json")):
        d = sf.parent
        s = CV3.jload(sf)
        if s.get("score_version") == CV3.SCORE_VERSION and not A.vanilla:
            continue
        old_rule = s.get("threshold_rule", "f2_max")
        preds = sorted(d.glob("pred_fold*.parquet"))
        if preds and "by_fold" in s:
            by = {}
            for f in preds:
                k = int(f.stem.replace("pred_fold", ""))
                h, y, p = _read_pred(f)
                sc, df = CV3.score_fold(y, p, CV3.offsets_from_h(h))
                old = s["by_fold"].get(str(k), s["by_fold"].get(k, {}))
                by[k] = {**{kk: old[kk] for kk in TRAIN_KEYS if kk in old}, **sc}
                df.insert(0, "fold", k)
                df.to_csv(d / f"metrics_fold{k}.csv", index=False, encoding="utf-8-sig")
            ks = sorted(by)
            CV3.jdump(s, d / f"summary_old_{old_rule.replace('>=', 'ge').replace('.', '')}.json")
            s.update({"by_fold": by, "mean": {kk: float(np.mean([by[k][kk] for k in ks])) for kk in EN.MEAN_KEYS},
                      "thresholds": {k: by[k]["threshold"] for k in ks}, "threshold_rule": CV3.THRESHOLD_RULE, "score_version": CV3.SCORE_VERSION,
                      "rescored_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rescored_from": old_rule,
                      "rescore_note": "재학습 없음 - 저장된 맞혀 보는 행 예측으로 기준값·지표만 다시 계산"})
            CV3.jdump(s, sf)
            n += 1
            LOG(f"rescore: {d.relative_to(P['results'])} ({old_rule} -> {CV3.THRESHOLD_RULE}) 평균 PR-AUC {s['mean']['pr_auc']:.5f}, P@R {s['mean']['p_at_r']:.5f}")
        elif (d / "pred_test.parquet").exists() and sorted(d.glob("pred_cv_fold*.parquet")):
            ys, ps = [], []
            for f in sorted(d.glob("pred_cv_fold*.parquet")):
                _, y, p = _read_pred(f)
                ys.append(y)
                ps.append(p)
            thr = CV3.threshold_at_recall(np.concatenate(ys), np.concatenate(ps))
            h, y, p = _read_pred(d / "pred_test.parquet")
            st, df = CV3.score_test(y, p, CV3.offsets_from_h(h), thr["threshold"])
            CV3.jdump(s, d / f"summary_old_{old_rule.replace('>=', 'ge').replace('.', '')}.json")
            s.update({"threshold": thr, "test": st, "threshold_rule": CV3.THRESHOLD_RULE, "score_version": CV3.SCORE_VERSION, "rescored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "rescored_from": old_rule})
            CV3.jdump(s, sf)
            df.to_csv(d / "metrics_test_by_h.csv", index=False, encoding="utf-8-sig")
            n += 1
            LOG(f"rescore: {d.name} 시험 다시 채점 (기준값 {thr['threshold']:.6f})")
    LOG(f"rescore: {n}개 실행 다시 채점")
    cmd_tables()


def run_sub(args, L=None, wait=True):
    cmd = [sys.executable, os.path.join(SRC, "run3.py")] + args + ["--root", str(R), "--device", DEVICE, "--seed", str(SEED)] \
        + (["--config", A.config] if A.config else []) + (["--smoke"] if A.smoke else []) + (["--data-end", A.data_end] if A.data_end else []) \
        + (["--L", str(L)] if L else [])
    env = dict(os.environ)
    if L:
        env["EXP2_L"] = str(L)
    LOG(f"chain: 실행 {' '.join(args)}" + (f" (창 {L}일)" if L else ""))
    if not wait:
        return subprocess.Popen(cmd, env=env)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"chain: {' '.join(args)} 실패(종료 코드 {r.returncode})")
    return r


def stage1_runs():
    pairs = set(G.CFG.get("vanilla_stage1") or [])
    out = []
    for s_ in collect(P["results"]):
        if s_.get("L") == 100 and f"{s_['model']}:{s_.get('input', 'summary')}" in pairs and s_.get("score_version") == CV3.SCORE_VERSION \
                and len(s_.get("folds", [])) == len(G.FOLDS) and s_["run"] == run_name(s_["model"], INPUT_LABEL[s_.get("input", "summary")], 100, SEED,
                                                                                     DEVICE if s_.get("input", "summary") == "summary" else "cpu"):
            out.append(s_)
    return out


def cmd_chain():
    st_f = R / "chain_status.json"
    status = CV3.jload(st_f) if st_f.exists() else {"steps": {}}

    def mark(step, **kw):
        status["steps"][step] = {"done_at": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
        CV3.jdump(status, st_f)

    cc = G.CFG.get("chain", {})
    t_all = time.time()
    # 1) 바닐라 1단계(기상 100일)
    t0 = time.time()
    run_sub(["vanilla"], L=100)
    run_sub(["tables"], L=100)
    s1 = stage1_runs()
    need = set(G.CFG["vanilla_stage1"])
    have = {f"{s_['model']}:{s_.get('input', 'summary')}" for s_ in s1}
    if have != need:
        raise RuntimeError(f"chain: 1단계 결과가 모자람 {sorted(need - have)}")
    best = max(s1, key=lambda s_: (s_["mean"]["pr_auc"], -sorted(need).index(f"{s_['model']}:{s_.get('input', 'summary')}")))
    pair = f"{best['model']}:{best.get('input', 'summary')}"
    CV3.jdump({"pair": pair, "rule": "교차검증 3겹 평균 PR-AUC 최대", "table": {f"{s_['model']}:{s_.get('input', 'summary')}": s_["mean"]["pr_auc"] for s_ in s1}},
              R / "selection_stage1.json")
    mark("stage1", pair=pair, seconds=round(time.time() - t0))
    LOG(f"chain: 1단계 1위 = {pair} (평균 PR-AUC {best['mean']['pr_auc']:.5f})")
    if best.get("input") == "raw" and cc.get("stop_if_raw_wins", True):
        mark("stopped", reason="1단계 1위가 일별 원값 - 창 길이 비교(원값 9개 창)·튜닝 입력을 사용자와 정해야 함(시간·메모리)")
        LOG("chain: 1위가 일별 원값이라 멈춤 - 창 길이 비교·튜닝 입력 결정 필요(config chain.stop_if_raw_wins)")
        sys.exit(10)
    # 2) 2단계 창 길이
    t0 = time.time()
    Ls = [int(x) for x in (cc.get("Ls") or [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]
    run_sub(["window", "--pair", pair, "--Ls", ",".join(map(str, Ls))])
    run_sub(["tables"], L=100)
    m_, inp = pair.split(":")
    wr = [s_ for s_ in collect(P["results"]) if s_["model"] == m_ and s_.get("input", "summary") == inp and s_.get("L") in Ls
          and s_.get("score_version") == CV3.SCORE_VERSION and s_["run"] == run_name(m_, INPUT_LABEL[inp], s_["L"], SEED, DEVICE if inp == "summary" else "cpu")]
    if {s_["L"] for s_ in wr} != set(Ls):
        raise RuntimeError(f"chain: 창 길이 결과가 모자람 {sorted(set(Ls) - {s_['L'] for s_ in wr})}")
    bw = max(wr, key=lambda s_: (s_["mean"]["pr_auc"], -s_["L"]))
    Lsel = int(bw["L"])
    CV3.jdump({"pair": pair, "L": Lsel, "rule": "교차검증 3겹 평균 PR-AUC 최대(같으면 짧은 창)", "table": {s_["L"]: s_["mean"]["pr_auc"] for s_ in wr}},
              R / "selection_window.json")
    mark("stage2", L=Lsel, seconds=round(time.time() - t0))
    LOG(f"chain: 고른 창 = {Lsel}일 (평균 PR-AUC {bw['mean']['pr_auc']:.5f})")
    # 3) 고른 창 밖의 입력 캐시 지움(디스크)
    for f in list(P["cache"].glob("*_L[0-9][0-9][0-9].npy")) + list(P["cache"].glob("*_L[0-9][0-9][0-9].json")):
        if not f.name.endswith(f"_L{Lsel:03d}{f.suffix}"):
            f.unlink()
    # 4) 스무딩 계수 k (교차검증으로, 설정 smoothing_k.select)
    if cc.get("kselect", True) and G.CFG["smoothing_k"].get("select") and not (R / "selection_k.json").exists():
        t0 = time.time()
        if inp == "raw":
            raise RuntimeError("chain: 2단계 1위가 일별 원값 - k 고르기 입력(원값 + 과거 이력)은 사용자와 정해야 함")
        run_sub(["kselect", "--model", m_, "--featset", "wx,hist_core"], L=Lsel)   # 2단계 1위 모델 + 기상(요약/시퀀스) + 과거 이력 핵심 칸, 창 = 2단계 창
        mark("kselect", k=CV3.jload(R / "selection_k.json")["k"], model=m_, featset="wx,hist_core", L=Lsel, seconds=round(time.time() - t0))
    # 5) 튜닝: 트리 모델 차례로(한 프로세스씩), BiLSTM 은 첫 트리 튜닝이 입력 캐시·이상치 점수를 만든 뒤 GPU 로 함께
    t0 = time.time()
    tms = list(G.CFG.get("tune_models") or [])
    trials = [] if A.trials is None else ["--trials", str(A.trials)]
    trees = [m for m in tms if m != "bilstm"]
    bil = None
    try:
        for i, m in enumerate(trees):
            if i == 0 and "bilstm" in tms and cc.get("parallel_bilstm", True):
                ready = P["tune"] / f"{m}.fds_ready"
                if ready.exists():
                    ready.unlink()                         # 지난 실행이 남긴 표시 지움(BiLSTM 이 캐시를 같이 만들지 않게)
                pr = run_sub(["tune", "--model", m, "--featset", "tune"] + trials, L=Lsel, wait=False)
                while pr.poll() is None and not ready.exists():
                    time.sleep(20)
                if ready.exists():
                    bil = run_sub(["tune", "--model", "bilstm", "--featset", "tune"] + trials, L=Lsel, wait=False)
                if pr.wait() != 0:
                    raise RuntimeError(f"chain: tune {m} 실패")
            else:
                run_sub(["tune", "--model", m, "--featset", "tune"] + trials, L=Lsel)
        if "bilstm" in tms:
            if bil is None:
                run_sub(["tune", "--model", "bilstm", "--featset", "tune"] + trials, L=Lsel)
            elif bil.wait() != 0:
                raise RuntimeError("chain: tune bilstm 실패")
    except BaseException:
        if bil is not None and bil.poll() is None:          # 실패하면 함께 띄운 BiLSTM 튜닝(이 chain 의 자식)도 끝냄 - 다시 실행하면 이어서
            bil.terminate()
            LOG("chain: 실패로 함께 돌던 BiLSTM 튜닝(자식 프로세스) 끝냄")
        raise
    mark("tune", models=tms, seconds=round(time.time() - t0))
    mark("done", total_seconds=round(time.time() - t_all), note="모델별 상위 후보 표 = tables/튜닝후보_<모델>.csv - 변수 선택·최종은 사용자가 후보를 고른 뒤 따로")
    LOG("chain: 튜닝까지 끝 - 멈춤(변수 선택·최종은 사용자 결정 뒤 따로 명령)")


def cmd_items():
    """고위험 품목: 겹마다 그 겹 학습 자료로, 최종용은 학습 자료 전체로 검출 비중 상위 top_n(3단계)."""
    ctx = CV3.load_ctx(LOG)
    cvo = CV3.CV(ctx, LOG, P["cache"], seed=SEED)
    out = {"rule": f"기준일 <= 학습 끝 시료의 품목군 3단계별 검출 수 내림차순 상위 {G.CFG['group_items']['top_n']} (검출 비중 = 그 품목 검출 ÷ 전체 검출)",
           "folds": {f["fold"]: cvo.top_items(f["train_end"]) for f in G.FOLDS}, "final": cvo.top_items(G.BOUNDS["learn_end"])}
    CV3.jdump(out, R / "selection_items.json")
    LOG(f"items: {json.dumps({k: v['codes'] for k, v in out['folds'].items()}, ensure_ascii=False)} / 최종 {out['final']['codes']}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    LOG(f"===== run3.py {' '.join(sys.argv[1:])} | pid {os.getpid()} | L={G.L} | 장치 {DEVICE} | 시드 {SEED} | 설정 {G.CONFIG_PATH}"
        + (" | 시험 실행(smoke)" if G.SMOKE.get("enabled") else ""))
    guard = MemGuard(float(G.CFG.get("mem_floor_gb", 10)))
    guard.start()
    t0 = time.time()
    {"vanilla": cmd_vanilla, "window": cmd_window, "device": cmd_device, "kselect": cmd_kselect, "tune": cmd_tune,
     "choose": cmd_choose, "drop": cmd_drop, "final": cmd_final, "tables": cmd_tables, "rescore": cmd_rescore, "chain": cmd_chain,
     "items": cmd_items}[A.cmd]()
    LOG(f"===== {A.cmd} 끝 {time.time() - t0:.0f}s (최소 여유 메모리 {guard.min_avail:.1f} GB, 최대 RSS {guard.peak_rss / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
