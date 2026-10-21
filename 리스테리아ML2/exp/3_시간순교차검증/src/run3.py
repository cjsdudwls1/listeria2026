# -*- coding: utf-8 -*-
"""3차 실험(시간순 교차검증) 실행. 설정 = config3.json (cfg3.py 머리말), 사양 = 사양_20261020.md.

python run3.py <명령> [공통 옵션] [명령 옵션]
공통: --root 폴더(results·tables·cache·logs·params·tune 를 그 아래에) --config 설정 파일 --L 창 길이 --device cpu|gpu(트리 3종)
      --seed --smoke(시험 실행: 설정 smoke 켬) --data-end YYYY-MM-DD(데이터 끝 날짜 덮어쓰기 - 가짜 경계 점검용)
고르기 점수 = 교차검증 세 겹 평균 구간 가중 PR-AUC(0.5·0.3·0.2, cv.SELECT_KEY) - 바닐라 순위·기상 거르기·k·튜닝 목표·변수 빼기·묶음 섞기 중요도 모두.
명령
  vanilla  [--pairs xgb:summary,xgb:raw,...,bilstm:seq] [--folds 1,2,3] [--sub 하위폴더] [--drop-eval-cache]
           바닐라(사양 4절 2번, 창 = --L 하나): 기본 = 설정 vanilla_pairs(트리 3종 x 기상 요약·일별 원값 + BiLSTM 시퀀스) 교차검증
           -> results/<모델>__<base|raw|seq>__L<창>__s<시드>/ (끝난 설정은 건너뜀 = 이어하기). 창 10~100 전부는 chain 이 창마다 따로 실행
  baseline                                       바닐라 표(설정 70개) + 1위 (모델, 입력, 창) = 1차 베이스라인 -> selection_baseline.json, tables/바닐라_비교
  kselect                                        스무딩 계수 k 를 고르기 점수로 고름(1차 베이스라인 + 과거 이력 핵심 칸) -> selection_k.json, tables/k선택
  tune     [--trials 30]                         Optuna(목표 = 고르기 점수) - 1차 베이스라인 모델 하나 -> tune/<모델>.db, tables/튜닝후보_<모델>.csv
  choose   --model xgb --trial N                 고른 후보의 파라미터 저장(get_params JSON) -> params/<모델>.json
  drop     --model xgb [--vanilla]               묶음 섞기 중요도로 묶음 빼기(같은 파라미터) -> selection_drop_<모델>.json, tables/묶음빼기_<모델>.csv
  final    --model xgb [--vanilla] [--featset ...]   최종 재학습(학습 자료 전체, 반복 = 세 겹 최적 반복 평균) + 기준값(세 겹 맞혀 보는 행 예측 합친 것에서
           Recall≥목표 중 Precision 최대) -> 시험 한 번 채점 -> results/final_<모델>/, tables/최종_시험_<모델>.csv
  post     [--model cat]                         사양 4절 8·9·10번(같은 겹 입력을 한 프로세스에서): 최종 리핏 -> 시험 한 번, 고위험 품목 전용 모델 vs 전체 모델, 이상치 점수 단독(원본·지터링)
           -> results/final_<모델>/·post_<모델>/, tables/최종_시험_<모델>·고위험품목_비교_<모델>·이상치단독
  dl11                                           사양 4절 11번: BiLSTM·트랜스포머 x 방식 A·B x 학습 행 순서(섞기·시간 순서), 교차검증만 -> results/_dl11/, tables/딥러닝단계
  tables                                         바닐라 표 다시
  chain                                          이어 돌리기: 바닐라(창 10~100, 트리 프로세스·BiLSTM 프로세스 나란히) -> 바닐라 표·1차 베이스라인 -> 기상 거르기 -> k 고르기 -> 멈춤
                                                 (튜닝은 새 뭉치 파일이 온 뒤 따로). 단계마다 저장, 다시 실행하면 이어서(chain_status.json)
  items                                          고위험 품목(겹마다 그 겹 학습 자료, 최종용 = 학습 자료 전체로 검출 비중 상위 top_n) 목록만 -> selection_items.json
  rescore                                        끝난 실행을 저장된 예측으로 지금 채점 규칙에 맞춰 다시 채점(재학습 없음) -> 표 다시
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
    ap.add_argument("cmd", choices=["vanilla", "baseline", "wxsel", "kselect", "tune", "choose", "drop", "final", "post", "dl11", "tables", "rescore", "chain", "items"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "gpu"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-end", default=None)
    ap.add_argument("--models", default="xgb,lgbm,cat")
    ap.add_argument("--model", default=None)
    ap.add_argument("--featset", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--sub", default="")
    ap.add_argument("--drop-eval-cache", action="store_true")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--vanilla", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="drop·wxsel: 빼 보기 시도 수 상한(시험 실행용)")
    ap.add_argument("--pairs", default=None, help="vanilla: 모델:입력 쌍(쉼표) - 입력 = summary(기상 요약)·raw(일별 원값)·seq(BiLSTM). 없으면 설정 vanilla_pairs")
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
    """메모리·디스크 보호: 시스템 사용 가능 메모리 < mem_floor_gb 이거나 결과 폴더 디스크 남은 공간 < disk_floor_gb 이면 이 프로세스만 멈춤."""

    def __init__(self, floor):
        super().__init__(daemon=True)
        self.floor = floor
        self.disk_floor = float(G.CFG.get("disk_floor_gb", 10))
        self.min_disk = None
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
                import shutil
                dfree = shutil.disk_usage(str(R)).free / 1e9
                self.min_disk = dfree if self.min_disk is None else min(self.min_disk, dfree)
                if dfree < self.disk_floor:
                    print(f"!!!!! 디스크 남은 공간 {dfree:.1f} GB < {self.disk_floor} GB - 팀 공용 서버 보호로 이 프로세스(pid {os.getpid()})만 멈춤", flush=True)
                    os._exit(8)
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


def build_folds(cvo, bundles, k, folds, log, wvars=None):
    lc = {}
    fds = {}
    for f in folds:
        t0 = time.time()
        fds[f] = EN.build(cvo, f, bundles, k, log, learn_cache=lc, wvars=wvars)
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
    elif A.models != "xgb,lgbm,cat":
        items = [f"{m}:summary" for m in A.models.split(",") if m]
    else:
        items = list(G.CFG["vanilla_pairs"])
    out = []
    for it in items:
        m, inp = it.split(":")
        assert (m in M3.TREE_BOOST and inp in ("summary", "raw", "feat")) or (m == "bilstm" and inp == "seq"), it
        out.append((m, inp))
    return out


def run_done(rdir):
    """끝난 실행(지금 채점 판·세 겹 모두)이면 True - 이어하기."""
    f = rdir / "summary.json"
    if not f.exists():
        return False
    s_ = CV3.jload(f)
    return bool(s_.get("complete")) and s_.get("score_version") == CV3.SCORE_VERSION and len(s_.get("folds", [])) == len(folds_arg())


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
    LOG(f"[끝] {rdir.name}: 구간 가중 PR-AUC {s_['mean']['wpr_auc']:.5f}, 60일 평균 {s_['mean']['pr_auc']:.5f}, 최적 반복 {s_['best_rounds']}, 학습 합 {s_['fit_seconds']:.0f}s")


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
            inf["notes"] = {"rule": inf.get("es_rule", "raw3.py 조기 종료 재현")}
            per[m][k], preds[m][k], infos[m][k] = sc, (p, df), inf
            LOG(f"    겹 {k} 원값 {m}: 구간 가중 PR-AUC {sc['wpr_auc']:.5f}, 60일 평균 {sc['pr_auc']:.5f}, 최적 반복 {inf['best_rounds']}")
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
        LOG(f"    겹 {k} BiLSTM: 구간 가중 PR-AUC {sc['wpr_auc']:.5f}, 60일 평균 {sc['pr_auc']:.5f}, 최고 에폭 {inf['best_epoch']} ({time.time() - t1:.0f}s)")
    finish_run(base_dir / run_name("bilstm", "seq", G.L, SEED, "cpu"), "bilstm", "seq", per, preds, infos, k_info, ["seq"], fns, t0)


def cmd_vanilla(res_dir=None):
    pairs = parse_pairs()
    folds = folds_arg()
    base_dir = res_dir or (P["results"] / A.sub if A.sub else P["results"])
    fs = A.featset or "base"

    def rname(m, inp):
        lab = featset_label(fs) if inp == "feat" else INPUT_LABEL[inp]
        return run_name(m, lab, G.L, SEED, DEVICE if inp in ("summary", "feat") else "cpu")

    todo = [(m, i) for m, i in pairs if not run_done(base_dir / rname(m, i))]
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
            LOG(f"[끝] {rdir.name}: 구간 가중 PR-AUC {s_['mean']['wpr_auc']:.5f}, 60일 평균 {s_['mean']['pr_auc']:.5f}, 최적 반복 {s_['best_rounds']}, 학습 합 {s_['fit_seconds']:.0f}s")
            M3.clear_dm_cache()                            # 메모리: XGBoost DMatrix 재사용 캐시 비움(결과와 무관)
        del fds
        import gc
        gc.collect()
    raw_m = [m for m, i in todo if i == "raw"]
    if raw_m:
        run_raw(cvo, raw_m, folds, base_dir, get_k(["wx"]))
    if ("bilstm", "seq") in todo:
        run_bilstm(cvo, folds, base_dir, get_k(["wx"]))
    if A.drop_eval_cache:
        for f in P["cache"].glob(f"*eval_f*_L{G.L:03d}.*"):
            f.unlink()
            LOG(f"  맞혀 보는 행 캐시 지움: {f.name}")


# ---------------------------------------------------------------- 1차 베이스라인(바닐라 1위)
SEL = CV3.SELECT_KEY
SEL_NAME = "구간 가중 PR-AUC"


def vanilla_runs():
    """설정 vanilla_pairs x chain.Ls 의 끝난 바닐라 실행(results/<실행 이름>/summary.json) 목록."""
    out, missing = [], []
    for L in [int(x) for x in G.CFG["chain"]["Ls"]]:
        for it in G.CFG["vanilla_pairs"]:
            m, inp = it.split(":")
            rn = run_name(m, INPUT_LABEL[inp], L, SEED, DEVICE if inp == "summary" else "cpu")
            if run_done(P["results"] / rn):
                out.append(CV3.jload(P["results"] / rn / "summary.json"))
            else:
                missing.append(rn)
    return out, missing


def metric_cols(mean, by_fold=None):
    """표 공통 칸(세 겹 평균): 고르기 점수 + 보고 지표(사양 3절)."""
    r = {f"{SEL_NAME}(세 겹 평균)": mean[SEL]}
    if by_fold:
        for k in sorted(by_fold, key=int):
            r[f"겹{k} {SEL_NAME}"] = by_fold[k][SEL]
    r.update({"60일 평균 PR-AUC": mean["pr_auc"], f"Recall≥{CV3.RECALL_MIN:.2f} 에서의 Precision(60개 h 합친 행)": mean["p_at_r"],
              "기준값 Recall(60개 h 합친 행)": mean["r_at_thr"], "Precision(h 평균)": mean["precision"], "Recall(h 평균)": mean["recall"],
              "구간 가중 Recall": mean["wrecall"], "F1(h 평균)": mean["f1"], "Accuracy(h 평균)": mean["accuracy"], "ROC-AUC(h 평균)": mean["roc_auc"],
              "TP(h 당)": mean["tp"], "FN(h 당)": mean["fn"], "FP(h 당)": mean["fp"], "TN(h 당)": mean["tn"], "N(h 당)": mean["n"],
              "검사축소율": mean["inspect_reduction"], **{f"상위 {k}% 포착률": mean[f"top{k}"] for k in CV3.TOPK}})
    return r


def vanilla_table(runs):
    rows = []
    for s_ in runs:
        r = {"모델": M3.MODEL_NAMES[s_["model"]], "입력": INPUT_NAME[s_.get("input", "summary")], "기상 창(일)": int(s_["L"])}
        r.update(metric_cols(s_["mean"], s_["by_fold"]))
        r.update({"최적 반복(겹별)": "/".join(map(str, s_["best_rounds"])), "학습 시간(초, 겹 합)": s_["fit_seconds"],
                  "실행 이름": s_["run"], "_model": s_["model"], "_input": s_.get("input", "summary")})
        rows.append(r)
    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(["xgb", "lgbm", "cat", "bilstm"])}
    df["_o"] = df["_model"].map(order)
    df = df.sort_values([f"{SEL_NAME}(세 겹 평균)", "기상 창(일)", "_o"], ascending=[False, True, True]).reset_index(drop=True)
    df.insert(0, "순위", range(1, len(df) + 1))
    return df


def cmd_baseline():
    runs, missing = vanilla_runs()
    if missing:
        raise RuntimeError(f"baseline: 바닐라 실행이 모자람 {len(missing)}개: {missing[:10]}")
    df = vanilla_table(runs)
    best = df.iloc[0]
    write_table(df.drop(columns=["_model", "_input", "_o"]), P["tables"] / "바닐라_비교",
                f"바닐라(사양 4절 2번): 트리 3종 x 기상 창 {G.CFG['chain']['Ls']}일 x (기상 요약·일별 원값) + BiLSTM x 창(일별 시퀀스) = {len(df)}개 설정, "
                f"기본 설정 + 조기 종료만, 입력 = 범주 4(sido·sgg·p2·p3) + h + 기상. 순위 = 고르기 점수({SEL_NAME}, 세 겹 평균; 같으면 짧은 창). "
                "1위 = 1차 베이스라인. 시험 미사용")
    sel = {"model": best["_model"], "input": best["_input"], "L": int(best["기상 창(일)"]), "run": best["실행 이름"],
           "score": float(best[f"{SEL_NAME}(세 겹 평균)"]), "rule": f"바닐라 {len(df)}개 설정 중 고르기 점수({SEL_NAME}, 세 겹 평균) 최대(같으면 짧은 창)",
           "top5": df.head(5)[["순위", "모델", "입력", "기상 창(일)", f"{SEL_NAME}(세 겹 평균)"]].to_dict("records"),
           "chosen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "score_version": CV3.SCORE_VERSION}
    CV3.jdump(sel, R / "selection_baseline.json")
    LOG(f"baseline: 1차 베이스라인 = {sel['model']}:{sel['input']} 창 {sel['L']}일 ({SEL_NAME} {sel['score']:.5f})")
    return sel


def baseline(check_L=True):
    f = R / "selection_baseline.json"
    if not f.exists():
        raise RuntimeError("selection_baseline.json 없음 - 'run3.py baseline' 먼저")
    b = CV3.jload(f)
    if check_L and int(b["L"]) != G.L:
        raise RuntimeError(f"1차 베이스라인 창 {b['L']}일인데 이 프로세스 창 {G.L}일 - '--L {b['L']}' 로 실행")
    return b


def baseline_wvars(required=True):
    f = R / "selection_wx.json"
    if not f.exists():
        if required:
            raise RuntimeError("selection_wx.json 없음 - 'run3.py wxsel'(기상 거르기, 사양 4절 3번) 먼저")
        return None
    return CV3.jload(f)["wvars"]


# ---------------------------------------------------------------- 실행 결과 캐시(이어하기)
def res_key(*parts):
    import hashlib
    return hashlib.md5(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def run_cached(sub, label, key, fn, need_perm=False):
    """results/<sub>/<label>__<key>/summary.json 이 있으면(지금 채점 판, 필요한 섞기 중요도 포함) 읽고, 없으면 fn() 실행 후 저장."""
    d = P["results"] / sub / f"{label}__{key}"
    f = d / "summary.json"
    if f.exists():
        js = CV3.jload(f)
        if js.get("score_version") == CV3.SCORE_VERSION and (not need_perm or js.get("perm_wvar_importance") or js.get("perm_importance")):
            js["by_fold"] = {int(k): v for k, v in js["by_fold"].items()}
            js["_cached"] = True
            return js
    res = fn()
    d.mkdir(parents=True, exist_ok=True)
    for k, (p_, df_) in (res.get("preds") or {}).items():
        CV3.write_pred(d / f"pred_fold{k}.parquet", res["fold_ns"][k].rs_ev, p_, k)
    out = {kk: v for kk, v in res.items() if kk not in ("preds", "models", "fold_ns", "model_info")}
    out.update({"label": label, "key": key, "score_version": CV3.SCORE_VERSION, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "code_md5": CV3.code_md5()})
    CV3.jdump(out, f)
    CV3.jdump({"model_info": res.get("model_info")}, d / "model_info.json")
    out["by_fold"] = {int(k): v for k, v in out["by_fold"].items()}
    return out


# ---------------------------------------------------------------- 3. 기상 변수 거르기
def cmd_wxsel():
    """사양 4절 3번: 1차 베이스라인 (모델, 입력, 창) 기본 설정 모델로 기상 19개 변수를 하나씩 거름(변수 섞기 중요도 낮은 것부터 빼 보고 고르기 점수가 오르면 뺀 채로)."""
    bl = baseline()
    model, inp = bl["model"], bl["input"]
    folds = folds_arg()
    bd = G.CFG["bundle_drop"]
    tol, max_passes = float(bd.get("tol", 0.0)), int(bd.get("max_passes", 20))
    cvo = new_cv(LOG)
    cvo.enable_mem_cache(G.CFG.get("mem_cache_gb", 16))      # 후보마다 같은 기상 칸을 다시 계산하지 않게(값 같음)
    lc = {}

    def run(wl, perm):
        key = res_key("wxsel", model, inp, G.L, SEED, sorted(wl), folds)
        return run_cached("_wxsel", f"{model}__{INPUT_LABEL[inp]}__L{G.L:03d}__n{len(wl)}", key,
                          lambda: EN.run_setting(cvo, model, inp, [], wl, G.K_SMOOTH_FIXED, None, DEVICE, SEED, LOG, folds,
                                                 perm_wvars=perm, learn_cache=lc), need_perm=perm)

    cur = list(EN.WVARS)
    base = run(cur, True)
    van = CV3.jload(P["results"] / bl["run"] / "summary.json")
    same = abs(base["mean"][SEL] - van["mean"][SEL]) <= 1e-9
    LOG(f"wxsel: 시작(19개) {SEL_NAME} {base['mean'][SEL]:.6f} - 바닐라 실행 {bl['run']} {van['mean'][SEL]:.6f} 와 같음 {same}")
    hist = [{"단계": 0, "회차": 0, "시도한 변수": "(시작: 19개 전부)", "변수 섞기 중요도": None, f"기준 {SEL_NAME}": base["mean"][SEL],
             f"후보 {SEL_NAME}": None, "결정": "시작", "남은 변수 수": len(cur), "남은 변수": ",".join(cur)}]
    imp_log, step = [], 0
    for ps in range(1, max_passes + 1):
        if not base.get("perm_wvar_importance"):
            base = run(cur, True)
        imp = base["perm_wvar_importance"]
        imp_log.append({"회차": ps, "기준 변수": list(cur), f"기준 {SEL_NAME}": base["mean"][SEL], "importance": imp,
                        "by_fold": base.get("perm_wvar_by_fold"), "rule": base.get("perm_wvar_rule")})
        order = sorted(cur, key=lambda v: (imp.get(v, 0.0), EN.WVARS.index(v)))
        LOG(f"wxsel 회차 {ps}: 순서(중요도 낮은 것부터) {[(v, round(imp.get(v, 0.0), 6)) for v in order]}")
        acc_any = False
        for v in order:
            if v not in cur or len(cur) <= 1:
                continue
            if A.max_steps is not None and step >= A.max_steps:      # 시험 실행용 상한
                break
            step += 1
            cand = [x for x in cur if x != v]
            r = run(cand, False)
            ok = r["mean"][SEL] > base["mean"][SEL] + tol
            hist.append({"단계": step, "회차": ps, "시도한 변수": v, "변수 섞기 중요도": imp.get(v, 0.0), f"기준 {SEL_NAME}": base["mean"][SEL],
                         f"후보 {SEL_NAME}": r["mean"][SEL], "결정": "뺌" if ok else "되돌림", "남은 변수 수": len(cand if ok else cur),
                         "남은 변수": ",".join(cand if ok else cur)})
            LOG(f"  단계 {step}: {v} 빼기 -> {SEL_NAME} {base['mean'][SEL]:.6f} -> {r['mean'][SEL]:.6f}: {'뺌' if ok else '되돌림'}")
            if ok:
                cur, base, acc_any = cand, r, True
        if not acc_any or (A.max_steps is not None and step >= A.max_steps):
            break
    df = pd.DataFrame(hist)
    write_table(df, P["tables"] / "기상거르기",
                f"기상 변수 거르기(사양 4절 3번): 1차 베이스라인 {M3.MODEL_NAMES[model]}·{INPUT_NAME[inp]}·창 {G.L}일 기본 설정. 변수 하나 = 그 변수 칸 전부"
                "(요약 = L/10+5칸, 원값 = L칸, BiLSTM = 값·결측 표시 채널). 회차마다 그때 기준 모델의 변수 섞기 중요도(맞혀 보는 행에서 그 변수 칸을 h 마다 행끼리 섞었을 때 "
                f"{SEL_NAME} 감소, 세 겹 평균; 씨앗 = default_rng([perm_seed, 겹, 1000 + 변수 번호])) 낮은 것부터 하나씩 빼 보고 {SEL_NAME}가 오르면 뺀 채로, 아니면 되돌림. "
                "한 회차에서 뺀 것이 없으면 끝. 범주 4칸·h 는 그대로. 시험 미사용")
    imp_rows = []
    for il in imp_log:
        for v, x in sorted(il["importance"].items(), key=lambda t: -t[1]):
            imp_rows.append({"회차": il["회차"], "변수": v, "변수 섞기 중요도(세 겹 평균)": x,
                             **{f"겹{k} 감소": il["by_fold"][k][v] if il.get("by_fold") and v in il["by_fold"].get(k, {}) else None
                                for k in (il.get("by_fold") or {})}})
    write_table(pd.DataFrame(imp_rows), P["tables"] / "기상거르기_변수중요도", "회차별 기상 변수 섞기 중요도(그 회차 기준 모델). 시험 미사용")
    pseed, pscope = EN.perm_cfg()
    CV3.jdump({"wvars": cur, "dropped": [v for v in EN.WVARS if v not in cur], "model": model, "input": inp, "L": G.L,
               "start_score": hist[0][f"기준 {SEL_NAME}"], "final_score": base["mean"][SEL], "final_mean": base["mean"],
               "start_equals_vanilla": same, "history": hist, "importance_by_round": imp_log,
               "perm": {"seed": pseed, "scope": pscope, "seed_index": "1000 + 기상 변수 번호", "rule": EN.PERM_WVAR_RULE},
               "rule": "사양 4절 3번(7번과 같은 규칙, 단위 = 기상 변수 하나)", **test_access_check("wxsel")}, R / "selection_wx.json")
    LOG(f"wxsel 끝: 남은 변수 {len(cur)}개 {cur}")


# ---------------------------------------------------------------- 4. k 고르기
def cmd_kselect():
    """스무딩 계수 k: 1차 베이스라인 (모델, 입력, 창) + 기상 거르기 변수 + 과거 이력 핵심 칸(hist_core), 후보 k 를 고르기 점수로 비교."""
    sk = G.CFG["smoothing_k"]
    bl = baseline()
    model, inp = bl["model"], bl["input"]
    wl = baseline_wvars()
    folds = folds_arg()
    cvo = new_cv(LOG)
    cvo.enable_mem_cache(G.CFG.get("mem_cache_gb", 16))      # k 마다 같은 기상 칸을 다시 계산하지 않게(값 같음)
    lc = {}
    rows, by_k = [], {}
    for k in sk["candidates"]:
        LOG(f"kselect: k={k}, {model}:{inp} 창 {G.L}일, 기상 변수 {len(wl)}개, + hist_core")
        key = res_key("kselect", model, inp, G.L, SEED, sorted(wl), ["hist_core"], int(k), folds)
        r = run_cached("_kselect", f"k{int(k):03d}__{model}__{INPUT_LABEL[inp]}__L{G.L:03d}", key,
                       lambda: EN.run_setting(cvo, model, inp, ["hist_core"], wl, int(k), None, DEVICE, SEED, LOG, folds, learn_cache=lc))
        by_k[int(k)] = r
        rows.append({"k": int(k), **metric_cols(r["mean"], r["by_fold"]), "최적 반복(겹별)": "/".join(map(str, r["best_rounds"])),
                     "학습 시간(초, 겹 합)": r["fit_seconds"]})
    df = pd.DataFrame(rows)
    col = f"{SEL_NAME}(세 겹 평균)"
    best = df.sort_values([col, "k"], ascending=[False, True]).iloc[0]
    write_table(df, P["tables"] / "k선택", f"스무딩 계수 k 고르기(사양 2절·4절 4번): 1차 베이스라인 {M3.MODEL_NAMES[model]}·{INPUT_NAME[inp]}·창 {G.L}일(바닐라 설정) "
                f"+ 기상 거르기 변수 {len(wl)}개 + 과거 이력 핵심 칸(hist_core: 3단위 x 5년·1년 스무딩 검출률, 5년 log 검사 수). 판단 = {SEL_NAME}(같으면 작은 k). "
                "과거 칸 규칙: 학습 행 = 예측일 전날까지 굴러가는 창, 맞혀 보는 행 = 그 겹 학습 끝 값 고정. 시험 미사용")
    CV3.jdump({"k": int(best["k"]), "judge": SEL, "table": rows, "model": model, "input": inp, "wvars": wl, "bundles": ["hist_core"], "L": G.L,
               "folds": folds, "rule": f"1차 베이스라인 (모델, 입력, 창) + 기상 거르기 변수 + hist_core 로 교차검증 세 겹 평균 {SEL_NAME} 최대(같으면 작은 k)",
               "past_rule": CV3.PAST_RULE_DESC, "bounds": G.BOUNDS, **test_access_check("kselect")}, R / "selection_k.json")
    LOG(f"kselect 끝: k = {int(best['k'])} ({SEL_NAME} {best[col]:.5f})")


# ---------------------------------------------------------------- 5. 튜닝(1차 베이스라인 모델 하나)
def tune_study(model, bundles, k, inp, wl):
    import optuna
    sig = featset_label(bundles) + f"_{INPUT_LABEL[inp]}_wx{len(wl)}_k{k}_L{G.L}_s{SEED}_obj-{SEL}" + ("_gpu" if DEVICE == "gpu" and model in M3.TREE_BOOST else "")
    st = optuna.create_study(direction="maximize", study_name=f"{model}__{sig}",
                             storage=f"sqlite:///{(P['tune'] / f'{model}.db').as_posix()}", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=int(G.CFG["optuna"]["sampler_seed"])))
    return st, sig


def tune_featset():
    """튜닝 입력 묶음 = 설정 featset_tune (+ 미리 뺄 단위 설정 pre_drop_units 의 칸 이름)."""
    bundles = CV3.resolve_featset(A.featset or "tune")
    pre = list(G.CFG.get("pre_drop_units") or [])
    drop_names = sorted({n for u in drop_unit_defs() if u["id"] in pre for n in u["cols"] + u.get("cat_cols", [])})
    return bundles, pre, drop_names


def nb_preflight(cvo):
    """튜닝 전 제품명 자료 점검(사양 2절): 시료별_뭉치(행 수·키 순서·칸 = 뭉치_정의.csv), 겹별 TF-IDF(겹 학습 끝 날짜 파일, 행·키·tf 칸 수, n_rows = 시료 수).
    하나라도 어긋나면 오류로 멈춤. 결과 -> nb_preflight.json."""
    import e2nb
    out = {}
    tags = [f["train_end"] for f in G.FOLDS] + [G.BOUNDS["learn_end"]]
    for tag in tags:
        tf = cvo.set_tfidf(tag)
        if not tf["fold_specific"]:
            raise RuntimeError(f"겹별 TF-IDF 파일 없음: {tag}")
        nb = e2nb.get_nb(cvo.ctx, ["n1", "n2", "n0"])
        c = nb.check
        ok = (c["nb_samples"]["lims_data_key_same_order"] and c["nb_samples"]["n_clusters"] == len(e2nb.cluster_cols())
              and c["n0"]["matched"] == nb.n_rows_full and c["n0"]["n_rows_eq_n_samples"] and c["n0"]["n_cols"] == len(e2nb.tfidf_cols()))
        out[tag] = {"tfidf": tf, "nb_samples": c["nb_samples"], "n0": c["n0"], "n2": c.get("n2"), "ok": bool(ok)}
        LOG(f"  제품명 점검 {tag}: 시료 {c['nb_samples']['rows']:,}행·키 순서 같음 {c['nb_samples']['lims_data_key_same_order']}, 뭉치 {c['nb_samples']['n_clusters']}개, "
            f"TF-IDF {tf['file']} 키 일치 {c['n0']['matched']:,}/{nb.n_rows_full:,}행, tf 칸 {c['n0']['n_cols']}, n_rows = 시료 수 {c['n0']['n_rows_eq_n_samples']} -> {'통과' if ok else '실패'}")
        if not ok:
            raise RuntimeError(f"제품명 자료 점검 실패({tag}): {out[tag]}")
    CV3.jdump({"checks": out, "n1_cols": len(e2nb.n1_names()), "n2_cols": len(e2nb.n2_names()), "n0_cols": len(e2nb.tfidf_cols()),
               "clusters": e2nb.cluster_cols(), "def_period": CV3.nb_def_period(), "at": time.strftime("%Y-%m-%d %H:%M:%S")}, R / "nb_preflight.json")
    return out


def cmd_tune():
    import optuna
    bl = baseline()
    model, inp = bl["model"], bl["input"]
    if inp == "raw":
        raise NotImplementedError("1차 베이스라인이 일별 원값 입력 - 원값 + 변수 전부 튜닝 경로는 아직 없음(raw3 에 파라미터 넣기 필요)")
    wl = baseline_wvars()
    bundles, pre, drop_names = tune_featset()
    if inp == "summary" and "wx" not in bundles:
        bundles = CV3.resolve_featset(["wx"] + bundles)
    k_info = get_k(bundles)
    folds = folds_arg()
    n_trials = A.trials or int(G.CFG["optuna"]["n_trials"])
    space = G.CFG["optuna"]["spaces"][model]
    st, sig = tune_study(model, bundles, k_info["k"], inp, wl)
    done = len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])
    LOG(f"tune {model}({inp}): 입력 {bundles}, 기상 변수 {len(wl)}개, 미리 뺀 단위 {pre}, k {k_info}, 시도 {n_trials} (끝난 것 {done}), 겹 {folds}, 장치 {DEVICE}")
    if done < n_trials:
        cvo = new_cv(LOG)
        if any(b in bundles for b in ("n0", "n1", "n2")):
            nb_preflight(cvo)
        fds = build_folds(cvo, bundles, k_info["k"], folds, LOG, wvars=wl)
        (P["tune"] / f"{model}.fds_ready").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        if model == "bilstm":                              # BiLSTM 은 기상 요약 칸을 쓰지 않음(시퀀스로) -> 메모리에서 버림
            for fd in fds.values():
                fd.drop_bundle_columns({"wx"})

        def objective(trial):
            params = M3.suggest(trial, space)
            t0 = time.time()
            res = EN.run_cv(model, params, fds, bundles, DEVICE, SEED, LOG, quiet=True, wvars=wl, drop_names=drop_names)
            m = res["mean"]
            for kk, v in m.items():
                trial.set_user_attr(kk, v)
            trial.set_user_attr("best_rounds", res["best_rounds"])
            trial.set_user_attr("fit_seconds", res["fit_seconds"])
            trial.set_user_attr("by_fold", {str(f): {SEL: res["by_fold"][f][SEL], "pr_auc": res["by_fold"][f]["pr_auc"], "p_at_r": res["by_fold"][f]["p_at_r"],
                                                     "threshold": res["by_fold"][f]["threshold"]} for f in fds})
            trial.set_user_attr("params_full", params)
            LOG(f"  시도 {trial.number}: {SEL_NAME} {m[SEL]:.5f}, 60일 평균 {m['pr_auc']:.5f}, P@R≥{CV3.RECALL_MIN:.2f} {m['p_at_r']:.5f}, 최적 반복 {res['best_rounds']}, {time.time() - t0:.0f}s, {params}")
            return m[SEL]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st.optimize(objective, n_trials=n_trials - done)
    test_access_check("tune")
    tune_table(model, st, sig, bundles, k_info)
    run_config(f"tune_{model}", {"model": model, "input": inp, "study": sig, "bundles": bundles, "wvars": wl, "pre_drop_units": pre, "k": k_info,
                                  "space": space, "cat_columns": FD_CATS.get("cats"), "feature_columns": FD_CATS.get("names")})


def tune_table(model, st, sig, bundles, k_info):
    import optuna
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_n = int(G.CFG.get("optuna_top_n", 5))
    mark = {t.number for t in sorted(done, key=lambda t: -t.value)[:top_n]}
    lab = f"{SEL_NAME} 상위 {top_n}"
    rows = []
    for t in done:
        u = t.user_attrs
        rows.append({"시도": t.number, lab: "예" if t.number in mark else "", **metric_cols(u),
                     "최적 반복(겹별)": "/".join(map(str, u["best_rounds"])), "학습 시간(초, 세 겹 합)": u["fit_seconds"],
                     "파라미터": json.dumps(u["params_full"], ensure_ascii=False)})
    df = pd.DataFrame(rows).sort_values([lab, f"{SEL_NAME}(세 겹 평균)"], ascending=[False, False])
    write_table(df, P["tables"] / f"튜닝후보_{model}",
                f"Optuna 목표 = {SEL_NAME}(세 겹 평균) - 상위 {top_n}개 표시. {M3.MODEL_NAMES[model]}, 연구 {sig}, 입력 {bundles}, k {k_info}. "
                f"기준값(오탐·놓침 칸) = {CV3.THRESHOLD_RULE}. 사용자가 고른 뒤 'run3.py choose --model 모델 --trial 번호'. 시험 미사용")
    LOG(f"표: 튜닝후보_{model} ({len(df)}줄)")


def cmd_choose():
    import optuna
    bl = baseline(check_L=False)
    model = A.model or bl["model"]
    wl = baseline_wvars()
    bundles, pre, _ = tune_featset()
    if bl["input"] == "summary" and "wx" not in bundles:
        bundles = CV3.resolve_featset(["wx"] + bundles)
    k_info = get_k(bundles)
    st, sig = tune_study(model, bundles, k_info["k"], bl["input"], wl)
    t = [x for x in st.trials if x.number == A.trial][0]
    params = M3.suggest(optuna.trial.FixedTrial(t.params), G.CFG["optuna"]["spaces"][model])
    extra = {"trial": t.number, "study": sig, "bundles": bundles, "wvars": wl, "pre_drop_units": pre, "k": k_info, "L": G.L, "seed": SEED,
             "scores": dict(t.user_attrs), "chosen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if model == "bilstm":
        import dl3
        CV3.jdump({"model": "bilstm", "params": {**dl3.CFG, **params}, "not_saved": [], **extra}, P["params"] / "bilstm.json")
        LOG(f"choose: bilstm 시도 {t.number} 파라미터 저장 -> params/bilstm.json")
        return
    m = M3.make_model(model, SEED, DEVICE, LOG, params)
    obj = M3.save_params(P["params"] / f"{model}.json", m, {**extra, "device": DEVICE, "suggested": params})
    LOG(f"choose: {model} 시도 {t.number} 파라미터 저장 -> params/{model}.json ({len(obj['params'])}칸, 저장 안 한 칸 {obj['not_saved']})")


def chosen_params(model):
    if A.vanilla:
        return None, "바닐라(기본 설정)"
    f = P["params"] / f"{model}.json"
    if not f.exists():
        raise RuntimeError(f"params/{model}.json 없음 - 'run3.py choose' 먼저(또는 --vanilla)")
    return M3.load_params(f), f"params/{model}.json"


# ---------------------------------------------------------------- 7. 변수 빼기(단위 = 묶음, 또는 설정 drop_units 의 변수 하나)
def drop_unit_defs():
    """설정 drop_units: 묶음째가 아니라 변수 하나씩 빼기 후보로 볼 단위(외부 지표·토양 중앙값·식품 속성). [{id, name, bundle, cols, cat_cols?}]."""
    return list(G.CFG.get("drop_units") or [])


def drop_units_of(bundles):
    """입력 묶음 -> 빼기 후보 단위 목록 [{id, name, bundle, cols, cat_cols}] (범주 4칸·h 제외). 설정 drop_units 에 있는 묶음은 변수 단위로 나눔."""
    defs = drop_unit_defs()
    split = {u["bundle"] for u in defs}
    out = []
    for b in bundles:
        if b in split:
            out += [u for u in defs if u["bundle"] == b]
        else:
            out.append({"id": b, "name": CV3.BUNDLES[b]["name"], "bundle": b, "cols": None})
    return out


def units_state(units, cur_ids, fd):
    """남은 단위 id -> (묶음 목록, 뺄 칸 이름, 섞기 단위 {id: (칸 이름, 씨앗 번호)})."""
    keep_b = sorted({u["bundle"] for u in units if u["id"] in cur_ids}, key=CV3.BUNDLE_ORDER.index)
    drop = set()
    groups = {}
    for i, u in enumerate(units):
        cols = (u["cols"] or []) + u.get("cat_cols", [])
        if u["cols"] is not None and u["id"] not in cur_ids:
            drop |= set(cols)
        if u["id"] in cur_ids:
            if u["cols"] is None:
                names_u = [n for n, b in zip(fd.names, fd.num_b) if b == u["bundle"]] + [c for c, b in zip(fd.cat_names, fd.cat_b) if b == u["bundle"]]
            else:
                names_u = cols
            groups[u["id"]] = (names_u, 2000 + i)
    return keep_b, sorted(drop), groups


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
    """사양 4절 7번(6번 보고 뒤 단계 - 코드만 준비): 고른 설정값으로 단위 섞기 중요도 낮은 것부터 빼 보고 고르기 점수가 오르면 뺀 채로."""
    bl = baseline()
    model = A.model or bl["model"]
    if bl["input"] == "raw":
        raise NotImplementedError("1차 베이스라인이 일별 원값 입력 - 원값 + 변수 묶음 빼기 경로는 아직 없음")
    params, psrc = chosen_params(model)
    wl = baseline_wvars()
    parent, pre, pre_drop = tune_featset()
    if bl["input"] == "summary" and "wx" not in parent:
        parent = CV3.resolve_featset(["wx"] + parent)
    k_info = get_k(parent)
    folds = folds_arg()
    bd = G.CFG["bundle_drop"]
    rule, tol, max_passes = bd["rule"], float(bd.get("tol", 0.0)), int(bd.get("max_passes", 20))
    n1_folds = bd.get("n1_judge_folds")
    cvo = new_cv(LOG)
    fds = build_folds(cvo, parent, k_info["k"], folds, LOG, wvars=wl)
    fd0 = fds[folds[0]]
    units = [u for u in drop_units_of(parent) if u["id"] not in pre]
    keep = model != "bilstm"

    def run(cur_ids, perm):
        kb, dn, groups = units_state(units, set(cur_ids), fd0)
        dn = sorted(set(dn) | set(pre_drop))
        r = EN.run_cv(model, params, fds, kb, DEVICE, SEED, LOG, quiet=True, keep_models=keep and perm, wvars=wl, drop_names=dn,
                      perm=(perm and not keep), unit_groups=groups)
        if perm and keep:
            r = EN.perm_from_models(r, fds, kb, LOG, wvars=wl, drop_names=dn, unit_groups=groups)
            r.pop("models", None)
        return r

    def means(res, ks):
        return {kk: float(np.mean([res["by_fold"][f][kk] for f in ks])) for kk in sorted(set([SEL, "pr_auc", "p_at_r"] + drop_metrics()))}

    cur = [u["id"] for u in units]
    uname = {u["id"]: u["name"] for u in units}
    base = run(cur, True)
    hist = [{"단계": 0, "회차": 0, "시도한 단위": "(시작: 전부)", "단위 섞기 중요도": None, f"기준 {SEL_NAME}": base["mean"][SEL],
             f"후보 {SEL_NAME}": None, "결정": "시작", "남은 단위": ", ".join(uname[u] for u in cur)}]
    step, imp_log = 0, []
    for ps in range(1, max_passes + 1):
        if "perm_importance" not in base:
            base = run(cur, True)
        imp = base["perm_importance"]
        imp_log.append({"회차": ps, "importance": imp, "by_fold": base.get("perm_by_fold"), "rule": base.get("perm_rule")})
        order = sorted(cur, key=lambda u: (imp.get(u, 0.0), u))
        LOG(f"drop 회차 {ps}: 순서(중요도 낮은 것부터) {[(uname[u], round(imp.get(u, 0.0), 6)) for u in order]}")
        acc_any = False
        for u in order:
            if u not in cur or len(cur) <= 1:
                continue
            if A.max_steps is not None and step >= A.max_steps:
                break
            step += 1
            cand = [x for x in cur if x != u]
            r = run(cand, False)
            ks = n1_folds if (u == "n1" and n1_folds) else folds
            mb, mc = means(base, ks), means(r, ks)
            ok = accept(mb, mc, rule, tol)
            hist.append({"단계": step, "회차": ps, "시도한 단위": uname[u], "단위 섞기 중요도": imp.get(u, 0.0), f"기준 {SEL_NAME}": mb[SEL],
                         f"후보 {SEL_NAME}": mc[SEL], "판단 겹": ",".join(map(str, ks)), "결정": "뺌" if ok else "되돌림",
                         "남은 단위": ", ".join(uname[x] for x in (cand if ok else cur))})
            LOG(f"  단계 {step}: {uname[u]} 빼기 -> {SEL_NAME} {mb[SEL]:.5f}->{mc[SEL]:.5f}: {'뺌' if ok else '되돌림'}")
            if ok:
                cur, base, acc_any = cand, r, True
        if not acc_any or (A.max_steps is not None and step >= A.max_steps):
            break
    kb, dn, _ = units_state(units, set(cur), fd0)
    write_table(pd.DataFrame(hist), P["tables"] / f"묶음빼기_{model}",
                f"변수 빼기(사양 4절 7번): {M3.MODEL_NAMES[model]}, 파라미터 {psrc}, 시작 = {parent}(미리 뺀 단위 {pre}), 기상 변수 {len(wl)}개, "
                f"판단 = {drop_metrics()}, 규칙 = {rule}(허용 {tol}). 단위 = 묶음, 단 외부 지표·토양 중앙값·식품 속성은 변수 하나씩(설정 drop_units). "
                f"회차마다 그때 기준 모델의 단위 섞기 중요도({SEL_NAME} 감소, 세 겹 평균) 낮은 것부터 빼 보고 오르면 뺀 채로. 범주 4칸·h 는 후보 아님. 시험 미사용")
    CV3.jdump({"model": model, "params_source": psrc, "parent": parent, "final_units": cur, "final_bundles": kb, "final_drop_names": dn,
               "pre_drop_units": pre, "wvars": wl, "rule": rule, "tol": tol, "metrics": drop_metrics(), "importance_by_round": imp_log,
               "n1_judge_folds": n1_folds, "history": hist, "final_mean": base["mean"], "final_best_rounds": base["best_rounds"],
               "k": k_info, "limitations": CV3.limitations(kb, k_info), **test_access_check("drop")}, R / f"selection_drop_{model}.json")
    run_config(f"drop_{model}", {"model": model, "params_source": psrc, "parent": parent, "final_bundles": kb, "final_drop_names": dn, "k": k_info,
                                  "cat_columns": FD_CATS.get("cats"), "feature_columns": FD_CATS.get("names")})
    LOG(f"drop 끝: 남은 단위 {[uname[u] for u in cur]}")


# ---------------------------------------------------------------- 8. 최종(리핏)
def cmd_final():
    bl = baseline()
    model = A.model or bl["model"]
    if bl["input"] == "raw":
        raise NotImplementedError("1차 베이스라인이 일별 원값 입력 - 원값 최종 재학습 경로는 아직 없음")
    params, psrc = chosen_params(model)
    wl = baseline_wvars()
    drop_names = []
    if A.featset:
        bundles = CV3.resolve_featset(A.featset)
        parent, bsrc = bundles, "명령 옵션 --featset"
    elif (R / f"selection_drop_{model}.json").exists():
        js = CV3.jload(R / f"selection_drop_{model}.json")
        bundles, drop_names, parent = js["final_bundles"], js.get("final_drop_names", []), CV3.resolve_featset(js["parent"])
        bsrc = f"selection_drop_{model}.json"
    else:
        raise RuntimeError("입력 묶음을 모름 - drop 먼저 또는 --featset")
    k_info = get_k(parent)
    folds = folds_arg()
    cvo = new_cv(LOG)
    fds = build_folds(cvo, parent, k_info["k"], folds, LOG, wvars=wl)
    res = EN.run_cv(model, params, fds, bundles, DEVICE, SEED, LOG, keep_preds=True, wvars=wl, drop_names=drop_names)
    assert not CV3.TEST_ACCESS
    y_all = np.concatenate([fds[f].rs_ev.y for f in folds])
    p_all = np.concatenate([res["preds"][f][0] for f in folds])
    thr = CV3.threshold_at_recall(y_all, p_all)
    rounds = int(round(float(np.mean(res["best_rounds"]))))
    LOG(f"final: 교차검증 {SEL_NAME} {res['mean'][SEL]:.5f}, P@R {res['mean']['p_at_r']:.5f}, 최적 반복 {res['best_rounds']} -> 최종 반복 {rounds}, "
        f"기준값(세 겹 합친 행 {CV3.THRESHOLD_RULE}) {thr['threshold']:.6f} (합친 Precision {thr['precision']:.5f}, Recall {thr['recall']:.5f})")
    rdir = P["results"] / f"final_{model}"
    rdir.mkdir(parents=True, exist_ok=True)
    for f in folds:                                        # 세 겹 맞혀 보는 행 예측(기준값 출처) 저장 - 다시 채점(rescore)용
        CV3.write_pred(rdir / f"pred_cv_fold{f}.parquet", fds[f].rs_ev, res["preds"][f][0], f)
    del fds
    CV3.ALLOW_TEST = True                                  # 시험 행은 여기(최종)에서만 만든다
    fdF = EN.build(cvo, "final", parent, k_info["k"], LOG, wvars=wl)
    tr, te = fdF.view(bundles, wl, drop_names)
    m = None
    if model == "bilstm":                                  # 최종: 정한 에폭 수(세 겹 최고 에폭 평균)만큼, 시험 행 평가 없이
        _, df_, p, inf = EN._bilstm_fold("final", fdF, bundles, {**(params or {}), "fixed_epochs": rounds}, SEED, LOG, wvars=wl, drop_names=drop_names)
    else:
        m = M3.make_model(model, SEED, DEVICE, LOG, params)
        if model == "cat" and (params is None or "learning_rate" not in params):
            m.fixed_lr = float(np.mean([res["model_info"][f]["notes"]["learning_rate_used"] for f in folds]))
        m.fit(tr, rounds=rounds)
        p = m.predict(te)
    s, df = CV3.score_test(te.y, p, fdF.offs, thr["threshold"])
    CV3.write_pred(rdir / "pred_test.parquet", fdF.rs_ev, p, 0)
    df.to_csv(rdir / "metrics_test_by_h.csv", index=False, encoding="utf-8-sig")
    summ = {"threshold_rule": CV3.THRESHOLD_RULE, "score_version": CV3.SCORE_VERSION, "model": model, "params_source": psrc, "bundles": bundles,
            "drop_names": drop_names, "wvars": wl, "bundles_source": bsrc, "k": k_info, "rounds": rounds,
            "cv_best_rounds": res["best_rounds"], "cv_mean": res["mean"], "threshold": thr, "test": s,
            "cat_learning_rate_fixed": getattr(m, "fixed_lr", None) if m is not None else None, "test_access": list(CV3.TEST_ACCESS),
            "limitations": CV3.limitations(bundles, k_info), "train_rows": int(tr.n), "train_pos": int(tr.y.sum()),
            "test_pairs": int(te.n), "test_pos_pairs": int(te.y.sum()), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    CV3.jdump(summ, rdir / "summary.json")
    CV3.jdump(common_cfg({"model": model, "get_params": (m.info()["get_params"] if m is not None else params), "bundles": bundles, "feature_columns": tr.names,
                          "cat_columns": tr.cat_names, "ae": fdF.ae_info, "tfidf": fdF.tfidf, "window_range": fdF.window}),
              rdir / "config.json")
    row = {"모델": M3.MODEL_NAMES[model], **metric_cols({**s, "p_at_r": s["p_pooled"], "r_at_thr": s["r_pooled"]}), "기준값": thr["threshold"], "반복 수": rounds,
           "입력 묶음": ",".join(bundles), "파라미터": psrc, "한계": " / ".join(CV3.limitations(bundles, {**k_info}))}
    write_table(pd.DataFrame([row]), P["tables"] / f"최종_시험_{model}",
                f"최종: 학습 자료 전체로 재학습(반복 = 세 겹 최적 반복 평균), 기준값 = 세 겹 맞혀 보는 행 예측 합친 것에서 {CV3.THRESHOLD_RULE}, "
                "시험 한 번 채점. 점수 = h 1~60 평균(구간 가중 포함), 건수 = h 당 평균. 검사축소율 = 1 − (TP+FP+FN)/N")
    LOG(f"final 끝: 시험 {SEL_NAME} {s[SEL]:.5f}, Precision {s['precision']:.5f}, Recall {s['recall']:.5f}, 검사축소율 {s['inspect_reduction']:.4f}")


# ---------------------------------------------------------------- 8·9·10. 최종(리핏)·고위험 품목·이상치 탐지 단독 (같은 겹 입력을 한 프로세스에서 같이 씀)
JITTER = {"sigma": 0.03, "seed": 20261021}


def sm(s):
    """score_fold / score_test 결과 -> metric_cols 용(60개 h 합친 Precision·Recall 칸 이름 맞춤)."""
    return {**s, "p_at_r": s.get("p_at_r", s.get("p_pooled")), "r_at_thr": s.get("r_at_thr", s.get("r_pooled"))}


def mean_scores(by):
    return {kk: float(np.mean([by[f][kk] for f in by])) for kk in EN.MEAN_KEYS}


def cmd_post():
    """사양 4절 8번(최종 리핏 -> 시험 한 번), 9번(고위험 품목 상위 5개 전용 모델 vs 전체 모델, 같은 품목 행), 10번(이상치 점수 단독: 원본·지터링)."""
    import ae3
    bl = baseline()
    model = A.model or bl["model"]
    if bl["input"] != "summary":
        raise NotImplementedError("post: 1차 베이스라인 입력이 기상 요약일 때만")
    params, psrc = chosen_params(model)
    wl = baseline_wvars()
    js = CV3.jload(R / f"selection_drop_{model}.json")
    bundles, drop_names = js["final_bundles"], list(js.get("final_drop_names", []))
    parent = CV3.resolve_featset(js["parent"])
    k_info = get_k(parent)
    folds = folds_arg()
    out = {"model": model, "params_source": psrc, "bundles": bundles, "drop_names": drop_names, "parent": parent, "wvars": wl, "k": k_info,
           "L": G.L, "seed": SEED, "jitter": JITTER, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    t_all = time.time()
    cvo = new_cv(LOG)
    fds = build_folds(cvo, parent, k_info["k"], folds, LOG, wvars=wl)
    # ---------------- 8. 최종(리핏)
    t8 = time.time()
    res = EN.run_cv(model, params, fds, bundles, DEVICE, SEED, LOG, keep_preds=True, wvars=wl, drop_names=drop_names)
    same = abs(res["mean"][SEL] - js["final_mean"][SEL]) <= 1e-9
    y_cv = np.concatenate([fds[f].rs_ev.y for f in folds])
    p_cv = np.concatenate([res["preds"][f][0] for f in folds])
    thr = CV3.threshold_at_recall(y_cv, p_cv)
    rounds = int(round(float(np.mean(res["best_rounds"]))))
    if CV3.TEST_ACCESS:
        raise RuntimeError(f"시험 행을 이미 만든 기록: {CV3.TEST_ACCESS}")
    LOG(f"post 8: 교차검증 {SEL_NAME} {res['mean'][SEL]:.5f}(변수 빼기 결과와 같음 {same}), 최적 반복 {res['best_rounds']} -> 최종 반복 {rounds}, "
        f"기준값 {thr['threshold']:.6f} (합친 Precision {thr['precision']:.5f}, Recall {thr['recall']:.5f})")
    rdir = P["results"] / f"final_{model}"
    rdir.mkdir(parents=True, exist_ok=True)
    for f in folds:
        CV3.write_pred(rdir / f"pred_cv_fold{f}.parquet", fds[f].rs_ev, res["preds"][f][0], f)
    CV3.ALLOW_TEST = True                                  # 시험 행은 여기서 처음 만든다(사양 4절 8번)
    fdF = EN.build(cvo, "final", parent, k_info["k"], LOG, wvars=wl)
    tr, te = fdF.view(bundles, wl, drop_names)
    mF = M3.make_model(model, SEED, DEVICE, LOG, params)
    mF.fit(tr, rounds=rounds)
    p_te = mF.predict(te)
    s_te, df_te = CV3.score_test(te.y, p_te, fdF.offs, thr["threshold"])
    CV3.write_pred(rdir / "pred_test.parquet", fdF.rs_ev, p_te, 0)
    df_te.to_csv(rdir / "metrics_test_by_h.csv", index=False, encoding="utf-8-sig")
    t_rows = [{"구분": "교차검증(세 겹 평균, 겹마다 그 겹 기준값)", **metric_cols(res["mean"], res["by_fold"]), "기준값": "겹별 " + "/".join(f"{res['by_fold'][f]['threshold']:.5f}" for f in folds),
               "반복 수": "/".join(map(str, res["best_rounds"]))},
              {"구분": "시험(학습 자료 전체로 재학습, 기준값 = 세 겹 맞혀 보는 행 예측 합친 것에서)", **metric_cols(sm(s_te)), "기준값": f"{thr['threshold']:.6f}", "반복 수": str(rounds)}]
    write_table(pd.DataFrame(t_rows), P["tables"] / f"최종_시험_{model}",
                f"최종(사양 4절 8번): {M3.MODEL_NAMES[model]} {psrc}, 입력 묶음 {bundles}, 뺀 변수 {drop_names}, 기상 변수 {len(wl)}개, k {k_info['k']}, 창 {G.L}일. "
                f"학습 자료 전체(d ≤ {G.BOUNDS['learn_end']})로 재학습(반복 = 세 겹 최적 반복 평균), 판정 기준값 = 세 겹 맞혀 보는 행 예측 합친 것에서 {CV3.THRESHOLD_RULE}, "
                "시험 1년 한 번 채점. 점수 = h 1~60 평균·구간 가중, 건수 = h 당 평균")
    out["final"] = {"cv_mean": res["mean"], "cv_by_fold": res["by_fold"], "cv_equals_drop_result": same, "threshold": thr, "rounds": rounds,
                    "test": s_te, "train_rows": int(tr.n), "train_pos": int(tr.y.sum()), "test_pairs": int(te.n), "test_pos_pairs": int(te.y.sum()),
                    "cat_learning_rate": mF.notes.get("learning_rate_used"), "seconds": round(time.time() - t8, 1)}
    CV3.jdump(common_cfg({"model": model, "get_params": mF.info()["get_params"], "bundles": bundles, "drop_names": drop_names, "feature_columns": tr.names,
                          "cat_columns": tr.cat_names, "ae": fdF.ae_info, "tfidf": fdF.tfidf, "window_range": fdF.window}), rdir / "config.json")
    CV3.jdump({**out["final"], "test_access": list(CV3.TEST_ACCESS), "limitations": CV3.limitations(bundles, k_info)}, rdir / "summary.json")
    LOG(f"post 8 끝: 시험 {SEL_NAME} {s_te[SEL]:.5f}, 60일 평균 PR-AUC {s_te['pr_auc']:.5f}, Precision {s_te['precision']:.5f}, Recall {s_te['recall']:.5f}")
    del tr, te, mF
    # ---------------- 9. 고위험 품목
    t9 = time.time()
    items = {f: cvo.top_items(CV3.fold_def(f)["train_end"]) for f in folds}
    itemsF = cvo.top_items(G.BOUNDS["learn_end"])
    subs, full_sub = {}, {}
    for f in folds:
        codes = items[f]["codes"]
        mtr, mev = cvo.item_mask(fds[f].rs_tr, codes), cvo.item_mask(fds[f].rs_ev, codes)
        subs[f] = EN.subset_fd(fds[f], mtr, mev)
        full_sub[f] = res["preds"][f][0][mev]
        LOG(f"  고위험 품목 겹 {f}: {codes} - 학습 {subs[f].rs_tr.n:,}행/검출 {int(subs[f].rs_tr.y.sum())}, 맞혀 보는 (행,h) {subs[f].rs_ev.n:,}/검출 {int(subs[f].rs_ev.y.sum())}")
    rg = EN.run_cv(model, params, subs, bundles, DEVICE, SEED, LOG, keep_preds=True, wvars=wl, drop_names=drop_names)
    sf = {f: CV3.score_fold(subs[f].rs_ev.y, full_sub[f], subs[f].offs)[0] for f in folds}
    ys = np.concatenate([subs[f].rs_ev.y for f in folds])
    thr_g = CV3.threshold_at_recall(ys, np.concatenate([rg["preds"][f][0] for f in folds]))
    thr_f = CV3.threshold_at_recall(ys, np.concatenate([full_sub[f] for f in folds]))
    mtrF, meF = cvo.item_mask(fdF.rs_tr, itemsF["codes"]), cvo.item_mask(fdF.rs_ev, itemsF["codes"])
    subF = EN.subset_fd(fdF, mtrF, meF)
    trg, teg = subF.view(bundles, wl, drop_names)
    rounds_g = int(round(float(np.mean(rg["best_rounds"]))))
    mg = M3.make_model(model, SEED, DEVICE, LOG, params)
    mg.fit(trg, rounds=rounds_g)
    p_g = mg.predict(teg)
    sg_te, _ = CV3.score_test(teg.y, p_g, subF.offs, thr_g["threshold"])
    sf_te, _ = CV3.score_test(teg.y, p_te[meF], subF.offs, thr_f["threshold"])
    i_rows = [{"구분": "교차검증 평균 - 고위험 품목 전용 모델", **metric_cols(rg["mean"], rg["by_fold"]), "기준값": "겹별", "반복 수": "/".join(map(str, rg["best_rounds"]))},
              {"구분": "교차검증 평균 - 전체 모델(같은 품목 행)", **metric_cols(mean_scores(sf), sf), "기준값": "겹별", "반복 수": "/".join(map(str, res["best_rounds"]))},
              {"구분": "시험 - 고위험 품목 전용 모델", **metric_cols(sm(sg_te)), "기준값": f"{thr_g['threshold']:.6f}", "반복 수": str(rounds_g)},
              {"구분": "시험 - 전체 모델(같은 품목 행)", **metric_cols(sm(sf_te)), "기준값": f"{thr_f['threshold']:.6f}", "반복 수": str(rounds)}]
    write_table(pd.DataFrame(i_rows), P["tables"] / f"고위험품목_비교_{model}",
                f"고위험 품목(사양 4절 9번): 학습 행 검출 비중 상위 {G.CFG['group_items']['top_n']}개 품목(3단계) - 겹마다 그 겹 학습 행으로, 최종은 학습 행 전체로 다시 고름. "
                f"품목 전용 모델 = 같은 설정값({psrc})·같은 입력 묶음으로 그 품목 행만 학습(이상치 점수 칸은 전체 겹 값 그대로). 전체 모델 = 8번 모델의 같은 품목 행 예측. "
                "기준값 = 각 모델의 세 겹 맞혀 보는 행(그 품목) 예측 합친 것에서 Recall ≥ 0.80 중 Precision 최대(교차검증 줄은 겹마다 그 겹 기준값). "
                f"겹별 품목: {json.dumps({f: items[f]['codes'] for f in folds}, ensure_ascii=False)}, 최종 품목: {itemsF['codes']}")
    out["items"] = {"folds": items, "final": itemsF, "group_cv_mean": rg["mean"], "group_cv_by_fold": rg["by_fold"], "full_cv_by_fold": sf,
                    "full_cv_mean": mean_scores(sf), "thr_group": thr_g, "thr_full": thr_f, "group_test": sg_te, "full_test": sf_te,
                    "rounds_group": rounds_g, "test_pairs": int(teg.n), "test_pos": int(teg.y.sum()), "seconds": round(time.time() - t9, 1)}
    CV3.jdump({"rule": "학습 행 검출 비중 상위 품목(3단계)", "folds": items, "final": itemsF}, R / "selection_items.json")
    LOG(f"post 9 끝: 시험 고위험 품목 전용 {SEL_NAME} {sg_te[SEL]:.5f} vs 전체 모델 {sf_te[SEL]:.5f}")
    del subs, subF, trg, teg, mg
    # ---------------- 10. 이상치 탐지 단독(원본·지터링)
    t10 = time.time()
    a_rows, aes, jit_sc = [], {}, {}
    wk = set(wl)
    for ver, jit in (("원본", None), ("지터링", JITTER)):
        for kind in ae3.KINDS:
            col = f"ae_{kind}"
            S_ev, by = {}, {}
            for f in folds:
                fd = fds[f]
                if jit is None:
                    S_ev[f] = fd.X_ev[:, fd.names.index(col)]
                else:
                    ai = [i for i, (n_, b_) in enumerate(zip(fd.names, fd.num_b)) if b_ not in ("n2", "n0", "ae4") and not (b_ == "wx" and EN.wvar_of(n_) not in wk)]
                    S_tr_j, S_ev[f], _ = ae3.scores(kind, cvo.seed, f"f{f}", fd.rs_tr, np.ascontiguousarray(fd.X_tr[:, ai]), fd.cd_tr, fd.rs_ev,
                                                    np.ascontiguousarray(fd.X_ev[:, ai]), fd.cd_ev, cvo.ctx.weather, cvo.cache_dir, LOG, wvars=wl, jitter=jit)
                    jit_sc[(f, kind)] = (S_tr_j, S_ev[f])
                by[f] = CV3.score_fold(fd.rs_ev.y, S_ev[f], fd.offs)[0]
            thr_a = CV3.threshold_at_recall(y_cv, np.concatenate([S_ev[f] for f in folds]))
            if jit is None:
                S_te = fdF.X_ev[:, fdF.names.index(col)]
            else:
                ai = [i for i, (n_, b_) in enumerate(zip(fdF.names, fdF.num_b)) if b_ not in ("n2", "n0", "ae4") and not (b_ == "wx" and EN.wvar_of(n_) not in wk)]
                S_trF_j, S_te, _ = ae3.scores(kind, cvo.seed, "final", fdF.rs_tr, np.ascontiguousarray(fdF.X_tr[:, ai]), fdF.cd_tr, fdF.rs_ev,
                                              np.ascontiguousarray(fdF.X_ev[:, ai]), fdF.cd_ev, cvo.ctx.weather, cvo.cache_dir, LOG, wvars=wl, jitter=jit)
                jit_sc[("final", kind)] = (S_trF_j, S_te)
            st, _ = CV3.score_test(fdF.rs_ev.y, S_te, fdF.offs, thr_a["threshold"])
            aes[f"{kind}_{ver}"] = {"cv_by_fold": by, "cv_mean": mean_scores(by), "threshold": thr_a, "test": st}
            a_rows.append({"점수": kind, "판": ver, "구분": "교차검증 평균", **metric_cols(mean_scores(by), by)})
            a_rows.append({"점수": kind, "판": ver, "구분": "시험", **metric_cols(sm(st))})
            LOG(f"  이상치 단독 {kind} {ver}: 교차검증 {SEL_NAME} {aes[f'{kind}_{ver}']['cv_mean'][SEL]:.5f}, 시험 {st[SEL]:.5f}")
    write_table(pd.DataFrame(a_rows), P["tables"] / "이상치단독",
                "이상치 탐지 단독(사양 4절 10번): 이상치 점수(AEt·VAEt = 표 입력, AEs·VAEs = 기상 시퀀스)를 예측값으로 바로 채점. 겹마다 그 겹 학습 행 불검출만으로 학습(5겹 교차 적합, "
                "점수 낼 행 = 겹 모델 5개 평균), 최종 = 학습 자료 전체 불검출로 학습해 시험 채점. 표 입력 = 튜닝 입력 숫자 칸(그날 제품명 n2·n0·거른 기상 변수 제외). "
                f"지터링 = 학습 행 한 벌을 더 붙이고 그 벌의 표준화 입력에 N(0, {JITTER['sigma']}²) 잡음(배치마다 새로, 시퀀스는 값 채널만, 씨앗 {JITTER['seed']}). "
                "기준값 = 세 겹 맞혀 보는 행 점수 합친 것에서 Recall ≥ 0.80 중 Precision 최대(교차검증 줄은 겹마다 그 겹 기준값)")
    out["ae_solo"] = aes
    t10a = round(time.time() - t10, 1)
    # ---------------- 10 이어서: 더 좋은 판(이상치 점수 단독 교차검증 구간 가중 PR-AUC, 점수 4개 평균)의 점수를 최종 CatBoost 입력으로 넣은 판 vs 안 넣은 판
    t10b = time.time()
    mean_ver = {ver: float(np.mean([aes[f"{kd}_{ver}"]["cv_mean"][SEL] for kd in ae3.KINDS])) for ver in ("원본", "지터링")}
    best_ver = "지터링" if mean_ver["지터링"] > mean_ver["원본"] else "원본"
    with_b = [b for b in CV3.BUNDLE_ORDER if b in set(bundles) | {"ae4"}]
    wo_b = [b for b in bundles if b != "ae4"]

    def swap(ver):
        """겹·최종 자료의 이상치 점수 칸을 그 판 점수로(제자리). 되돌리는 함수를 돌려줌."""
        if ver != "지터링":
            return lambda: None
        saved = []
        for key, fd in [(f, fds[f]) for f in folds] + [("final", fdF)]:
            for kd in ae3.KINDS:
                j = fd.names.index(f"ae_{kd}")
                str_, sev_ = jit_sc[(key, kd)]
                saved.append((fd, j, fd.X_tr[:, j].copy(), fd.X_ev[:, j].copy()))
                fd.X_tr[:, j] = str_
                fd.X_ev[:, j] = sev_

        def restore():
            for fd, j, a_, b_ in saved:
                fd.X_tr[:, j] = a_
                fd.X_ev[:, j] = b_
        return restore

    v_rows, vres = [], {}
    for name, bl_, ver in (("이상치 점수 넣은 판", with_b, best_ver), ("이상치 점수 안 넣은 판", wo_b, None)):
        same8 = bl_ == bundles and ver in (None, "원본")
        if same8:                                          # 8번 모델과 같은 입력·같은 점수 -> 8번 결과 그대로
            r_v, thr_v, rounds_v, st_v = res, thr, rounds, s_te
        else:
            restore = swap(ver)
            try:
                r_v = EN.run_cv(model, params, fds, bl_, DEVICE, SEED, LOG, keep_preds=True, wvars=wl, drop_names=drop_names)
                thr_v = CV3.threshold_at_recall(y_cv, np.concatenate([r_v["preds"][f][0] for f in folds]))
                rounds_v = int(round(float(np.mean(r_v["best_rounds"]))))
                trv, tev = fdF.view(bl_, wl, drop_names)
                mv = M3.make_model(model, SEED, DEVICE, LOG, params)
                mv.fit(trv, rounds=rounds_v)
                st_v, _ = CV3.score_test(tev.y, mv.predict(tev), fdF.offs, thr_v["threshold"])
                del trv, tev, mv
            finally:
                restore()
        vres[name] = {"bundles": bl_, "ae_version": ver, "same_as_step8": same8, "cv_mean": r_v["mean"], "cv_by_fold": r_v["by_fold"],
                      "threshold": thr_v, "rounds": rounds_v, "test": st_v}
        v_rows.append({"판": name + (f"({ver} 점수)" if ver else ""), "구분": "교차검증(세 겹 평균)", **metric_cols(r_v["mean"], r_v["by_fold"]),
                       "반복 수": "/".join(map(str, r_v["best_rounds"])), "8번과 같은 모델": "예" if same8 else ""})
        v_rows.append({"판": name + (f"({ver} 점수)" if ver else ""), "구분": "시험", **metric_cols(sm(st_v)), "반복 수": str(rounds_v),
                       "8번과 같은 모델": "예" if same8 else ""})
        LOG(f"  {name}: 교차검증 {SEL_NAME} {r_v['mean'][SEL]:.5f}, 시험 {st_v[SEL]:.5f}")
    chosen = max(vres, key=lambda n_: vres[n_]["cv_mean"][SEL])
    for r_ in v_rows:
        r_["최종으로 고름(교차검증)"] = "예" if r_["판"].startswith(chosen) else ""
    write_table(pd.DataFrame(v_rows), P["tables"] / f"이상치넣기_비교_{model}",
                f"이상치 점수를 최종 CatBoost 입력으로(사양 4절 10번 이어서): 원본·지터링 중 이상치 점수 단독 교차검증 {SEL_NAME}(점수 4개 평균)가 더 좋은 판 = {best_ver} "
                f"(원본 {mean_ver['원본']:.5f}, 지터링 {mean_ver['지터링']:.5f}). 넣은 판 = 7번 변수 + 이상치 점수 묶음(그 판 점수), 안 넣은 판 = 7번 변수에서 이상치 점수 묶음 뺌. "
                f"같은 설정값({psrc}), 교차검증 3겹 -> 최종 리핏 -> 시험 각각 한 번. 최종 = 교차검증 {SEL_NAME}가 높은 판 = {chosen}(시험으로 고르지 않음)")
    out["ae_into_model"] = {"version_cv_mean_of_4": mean_ver, "best_version": best_ver, "variants": vres, "chosen_by_cv": chosen,
                            "seconds": round(time.time() - t10b, 1)}
    out["seconds"] = {"total": round(time.time() - t_all, 1), "final": out["final"]["seconds"], "items": out["items"]["seconds"], "ae_solo": t10a,
                      "ae_into_model": out["ae_into_model"]["seconds"]}
    out["test_access"] = list(CV3.TEST_ACCESS)
    CV3.jdump(out, P["results"] / f"post_{model}" / "summary.json")
    LOG(f"post 끝: {out['seconds']}")


# ---------------------------------------------------------------- 11. 딥러닝 단계
DL11_COMBOS = [(a, m, o) for a in ("bilstm", "trf") for m in ("A", "B") for o in ("shuffle", "time")]
DL11_NAMES = {"bilstm": "BiLSTM", "trf": "트랜스포머", "A": "방식 A(h 입력, 출력 1칸)", "B": "방식 B(출력 60칸)", "shuffle": "에폭마다 섞기", "time": "시간 순서"}


def cmd_dl11():
    """사양 4절 11번: BiLSTM·트랜스포머 x 방식 A·B x 학습 행 순서(섞기·시간 순서), 창 = 1차 베이스라인 창, 기상 = 거르고 남은 변수. 교차검증만."""
    import dl11
    bl = baseline()
    wl = baseline_wvars()
    folds = folds_arg()
    cvo = new_cv(LOG)
    fb = cvo.builder(["wx"], G.K_SMOOTH_FIXED)
    levels = cvo.cat_levels(list(CV3.BASE_CATS), fb)
    rows = []
    for a, m, o in DL11_COMBOS:
        rdir = P["results"] / "_dl11" / f"{a}{m}_{o}__L{G.L:03d}__s{SEED}"
        sf = rdir / "summary.json"
        if sf.exists() and CV3.jload(sf).get("score_version") == CV3.SCORE_VERSION:
            s_ = CV3.jload(sf)
        else:
            t0 = time.time()
            per, infos = {}, {}
            rdir.mkdir(parents=True, exist_ok=True)
            for f in folds:
                rs_tr, rs_ev = cvo.fold_train(f), cvo.eval_rows(f)
                _, cd_tr = cvo.cats(rs_tr, fb, levels)
                _, cd_ev = cvo.cats(rs_ev, fb, levels)
                p, inf = dl11.run_fold(cvo.ctx, rs_tr, rs_ev, cd_tr, cd_ev, CV3.fold_def(f)["train_end"], SEED, LOG, a, m, o, wvars=wl)
                sc, df = CV3.score_fold(rs_ev.y, p, rs_ev.h_offsets())
                sc.update({"fold": f, "best_rounds": inf["best_epoch"], "n_trained": inf["n_epochs"], "fit_seconds": inf["fit_seconds"]})
                per[f], infos[f] = sc, inf
                CV3.write_pred(rdir / f"pred_fold{f}.parquet", rs_ev, p, f)
                df.insert(0, "fold", f)
                df.to_csv(rdir / f"metrics_fold{f}.csv", index=False, encoding="utf-8-sig")
                LOG(f"    dl11 {a}{m}/{o} 겹 {f}: {SEL_NAME} {sc[SEL]:.5f}, 최고 에폭 {inf['best_epoch']}, {inf['fit_seconds']:.0f}s")
            s_ = {"arch": a, "mode": m, "order": o, "by_fold": per, "mean": mean_scores(per), "best_rounds": [per[f]["best_rounds"] for f in folds],
                  "fit_seconds": round(sum(per[f]["fit_seconds"] for f in folds), 1), "total_seconds": round(time.time() - t0, 1), "L": G.L, "wvars": wl,
                  "score_version": CV3.SCORE_VERSION, "code_md5": CV3.code_md5(), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            CV3.jdump(s_, sf)
            CV3.jdump({"model_info": infos}, rdir / "model_info.json")
        s_["by_fold"] = {int(k): v for k, v in s_["by_fold"].items()}
        rows.append({"구조": DL11_NAMES[a], "방식": DL11_NAMES[m], "학습 행 순서": DL11_NAMES[o], **metric_cols(s_["mean"], s_["by_fold"]),
                     "최고 에폭(겹별)": "/".join(map(str, s_["best_rounds"])), "학습 시간(초, 겹 합)": s_["fit_seconds"]})
    write_table(pd.DataFrame(rows), P["tables"] / "딥러닝단계",
                f"딥러닝 단계(사양 4절 11번): 창 {G.L}일(1차 베이스라인), 기상 변수 {len(wl)}개(기상 거르기), 범주 4칸 임베딩, 다른 입력 칸 없음. 방식 A = h 입력·출력 1칸, "
                "방식 B = 출력 60칸(T+1..T+60, 학습 정답 빈칸·겹 학습 끝 뒤는 마스크). 학습 행 순서: 에폭마다 섞기 vs 예측일 T 시간 순서 고정. 조기 종료 = 맞혀 보는 행 h 평균 AP 인내 5. "
                f"고르기 점수 = {SEL_NAME}. 교차검증만(시험 미사용)")
    CV3.jdump({"rows": rows, **test_access_check("dl11")}, P["results"] / "_dl11" / "summary_all.json")
    LOG(f"dl11 끝: {len(rows)}개 설정")


# ---------------------------------------------------------------- 표
NOTE = (f"고르기 점수 = {SEL_NAME}({CV3.BAND_DESC}, h 마다 PR-AUC) 세 겹 평균. 기준값 규칙 = {CV3.THRESHOLD_RULE}: " + CV3.THRESHOLD_DESC
        + ". 점수 = 겹마다 계산해 겹 평균(표준편차 없음). 60일 평균 PR-AUC·ROC-AUC = h 별 값의 h 1~60 평균. Recall≥0.80 에서의 Precision = 그 기준값에서 60개 h 합친 Precision. "
          "Accuracy·Precision·Recall·F1·검사축소율(= 1 − (TP+FP+FN)/N)·상위 k% = 그 기준값에서 h 마다 -> h 평균. TP·FP·FN·TN·N = h 당 평균. "
          "학습 시간 = 겹 학습(fit, 조기 종료 지표 계산 포함) 합")


def write_table(df, path, note):
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(path.with_suffix(".xlsx")) as xw:
        df.to_excel(xw, sheet_name="표", index=False)
        pd.DataFrame({"설명": [note, NOTE]}).to_excel(xw, sheet_name="규칙", index=False)


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
    runs, missing = vanilla_runs()
    if runs:
        df = vanilla_table(runs)
        write_table(df.drop(columns=["_model", "_input", "_o"]), P["tables"] / "바닐라_비교" if not missing else P["tables"] / "바닐라_비교_중간",
                    f"바닐라(사양 4절 2번) 끝난 설정 {len(df)}개(모자란 설정 {len(missing)}개). 순위 = {SEL_NAME}(세 겹 평균). 시험 미사용")
        LOG(f"표: 바닐라 {len(df)}줄 (모자람 {len(missing)})")


def _read_pred(f):
    import pyarrow.parquet as pq
    t = pq.read_table(f, columns=["h", "y", "p"])
    return t.column("h").to_numpy().astype(np.int64), t.column("y").to_numpy().astype(np.int8), t.column("p").to_numpy()


TRAIN_KEYS = ("fold", "n_train_rows", "n_train_pos", "best_rounds", "n_trained", "fit_seconds", "predict_seconds", "n_num", "n_cat")


def cmd_rescore():
    """results 바로 아래 끝난 실행을 저장 예측(pred_fold*.parquet)으로 지금 채점 규칙에 맞춰 다시 채점(재학습 없음)."""
    n = 0
    for sf in sorted(P["results"].glob("*/summary.json")):
        d = sf.parent
        s = CV3.jload(sf)
        if s.get("score_version") == CV3.SCORE_VERSION and not A.vanilla:
            continue
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
            CV3.jdump(s, d / f"summary_old_{str(s.get('score_version', 'v0'))[:2]}.json")
            s.update({"by_fold": by, "mean": {kk: float(np.mean([by[k][kk] for k in ks])) for kk in EN.MEAN_KEYS},
                      "thresholds": {k: by[k]["threshold"] for k in ks}, "threshold_rule": CV3.THRESHOLD_RULE, "score_version": CV3.SCORE_VERSION,
                      "rescored_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rescore_note": "재학습 없음 - 저장된 맞혀 보는 행 예측으로 기준값·지표만 다시 계산"})
            CV3.jdump(s, sf)
            n += 1
            LOG(f"rescore: {d.name} {SEL_NAME} {s['mean'][SEL]:.5f}")
    LOG(f"rescore: {n}개 실행 다시 채점")
    cmd_tables()


def run_sub(args, L=None, wait=True, env_extra=None):
    cmd = [sys.executable, os.path.join(SRC, "run3.py")] + args + ["--root", str(R), "--device", DEVICE, "--seed", str(SEED)] \
        + (["--config", A.config] if A.config else []) + (["--smoke"] if A.smoke else []) + (["--data-end", A.data_end] if A.data_end else []) \
        + (["--L", str(L)] if L else [])
    env = dict(os.environ)
    if L:
        env["EXP2_L"] = str(L)
    env.update(env_extra or {})
    LOG(f"chain: 실행 {' '.join(args)}" + (f" (창 {L}일)" if L else ""))
    if not wait:
        return subprocess.Popen(cmd, env=env)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"chain: {' '.join(args)} (창 {L}) 실패(종료 코드 {r.returncode})")
    return r


def cmd_chain():
    """바닐라(창 10~100: 트리 프로세스 chain.tree_workers 개 + BiLSTM 프로세스 1개 나란히) -> 1차 베이스라인 -> 기상 거르기 -> k 고르기 -> 멈춤(튜닝 직전)."""
    st_f = R / "chain_status.json"
    status = CV3.jload(st_f) if st_f.exists() else {"steps": {}}

    def mark(step, **kw):
        status["steps"][step] = {"done_at": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
        CV3.jdump(status, st_f)

    cc = G.CFG["chain"]
    t_all = time.time()
    Ls = [int(x) for x in cc["Ls"]]
    pairs = list(G.CFG["vanilla_pairs"])
    tree_pairs = [p for p in pairs if not p.startswith("bilstm")]
    bil_pairs = [p for p in pairs if p.startswith("bilstm")]
    nw = max(1, int(cc.get("tree_workers", 1)))
    env_t = {"OMP_NUM_THREADS": str(cc["omp_threads"])} if cc.get("omp_threads") else {}
    order = sorted(Ls, reverse=True)                       # 큰 창(오래 걸림)부터
    jobs = [("트리", [order[i::nw] for i in range(nw)][w], tree_pairs) for w in range(nw)] + ([("BiLSTM", order, bil_pairs)] if bil_pairs else [])
    errs = []

    def worker(name, ls, prs):
        failed = []
        for L in ls + [None]:
            if L is None:                                  # 실패한 창은 끝에 한 번 더(이어하기 - 끝난 설정은 건너뜀)
                for L2 in failed:
                    try:
                        run_sub(["vanilla", "--pairs", ",".join(prs)], L=L2, env_extra=env_t if name == "트리" else {})
                    except Exception as e:  # 두 번째 실패는 기록하고 끝에 멈춤
                        errs.append(f"{name} 창 {L2}: {e}")
                break
            try:
                run_sub(["vanilla", "--pairs", ",".join(prs)], L=L, env_extra=env_t if name == "트리" else {})
            except Exception as e:  # 다음 창은 계속
                LOG(f"chain: {name} 창 {L} 실패 - 다음 창 먼저, 끝에 다시: {e}")
                failed.append(L)

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=j, daemon=True) for j in jobs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise RuntimeError(f"chain: 바닐라 실패 {errs}")
    mark("vanilla", Ls=Ls, pairs=pairs, seconds=round(time.time() - t0))
    run_sub(["baseline"], L=100)
    bl = baseline(check_L=False)
    mark("baseline", **{k: bl[k] for k in ("model", "input", "L", "score")})
    if cc.get("wxsel", True):
        t0 = time.time()
        run_sub(["wxsel"], L=bl["L"], env_extra=env_t)
        mark("wxsel", wvars=baseline_wvars(), seconds=round(time.time() - t0))
    if cc.get("kselect", True) and G.CFG["smoothing_k"].get("select"):
        t0 = time.time()
        run_sub(["kselect"], L=bl["L"], env_extra=env_t)
        mark("kselect", k=CV3.jload(R / "selection_k.json")["k"], seconds=round(time.time() - t0))
    if not cc.get("tune", False):
        mark("stop_before_tune", total_seconds=round(time.time() - t_all), note="튜닝은 따로(설정 chain.tune = false)")
        LOG("chain: k 고르기까지 끝 - 튜닝 직전에서 멈춤")
        return
    t0 = time.time()
    run_sub(["tune"], L=bl["L"], env_extra=env_t)               # 튜닝 앞에서 제품명 자료 점검(nb_preflight) - 실패하면 여기서 멈춤
    mark("tune", model=bl["model"], seconds=round(time.time() - t0))
    mark("stop_for_report", total_seconds=round(time.time() - t_all), note="사양 4절 6번 보고(후보 고르기) - 사람이 고름")
    LOG("chain: 튜닝까지 끝 - 6번 보고에서 멈춤")


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
        + (f" | OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}" if os.environ.get("OMP_NUM_THREADS") else "")
        + (" | 시험 실행(smoke)" if G.SMOKE.get("enabled") else ""))
    guard = MemGuard(float(G.CFG.get("mem_floor_gb", 10)))
    guard.start()
    t0 = time.time()
    {"vanilla": cmd_vanilla, "baseline": cmd_baseline, "wxsel": cmd_wxsel, "kselect": cmd_kselect, "tune": cmd_tune,
     "choose": cmd_choose, "drop": cmd_drop, "final": cmd_final, "post": cmd_post, "dl11": cmd_dl11, "tables": cmd_tables, "rescore": cmd_rescore, "chain": cmd_chain,
     "items": cmd_items}[A.cmd]()
    LOG(f"===== {A.cmd} 끝 {time.time() - t0:.0f}s (최소 여유 메모리 {guard.min_avail:.1f} GB, 최대 RSS {guard.peak_rss / 1e9:.1f} GB, "
        f"최소 디스크 남은 공간 {guard.min_disk if guard.min_disk is None else round(guard.min_disk, 1)} GB)")


if __name__ == "__main__":
    main()
