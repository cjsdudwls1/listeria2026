# -*- coding: utf-8 -*-
"""2차(고도화) 실험 실행.

python src/run.py --stage <단계|all> [--smoke] [--only 실험ID 또는 실험ID__s시드] [--console-log 파일]
단계 순서(all): e00 -> prev1(1차 가중 지표) -> a_k -> a -> b -> c -> d -> h10 -> hbig -> h60 -> dl -> f -> g -> combo -> m -> n -> q -> p -> hx -> ae -> xs -> stk -> tables
  e00   : E00 (시드 42·43·44)
  prev1 : 1차 결과 16개에 구간 가중 지표(재학습 없음) -> tables/1차_가중지표.csv
  a_k   : A1_k{10,30,50,100,200} (시드 42) -> k* (results/selection_k.json)
  a     : A1, A2, A3 (k*) -> A 최고 (results/selection_A.json)
  b     : B1, B2 (로지스틱)
  c     : C1~C4 x (E00 입력, A 최고 입력). C4 = 학습 구간 검출 1건 이상 품목 전부(다운사이징, 사용자 지시 2026-10-19) -
          끝난 C1~C3 는 건너뛰고 C4_base·C4_hist 만 실제로 돎
  d     : D1(scale_pos_weight), D2(SMOTENC), D3(WOE)
  h10   : H10, hbig: H30 (시드 42), h60: H60 (학습 행마다 h 1~60 전부 - 사용 가능 메모리 e2cfg.H60_RAM_NEED_GB 이상일 때만, 팀 공용 서버에서 실행)
  dl    : DL 4종(e2dl.DL_REGISTRY 에 없으면 '미구현'으로 건너뜀)
  f     : F1(SPI)·F2(토양)·F3(식품 속성)·F4(F1+F2+F3) - 1차 보고서 표 9 ① (e2fg.py, 사용자 승인 2026-10-19)
  g     : G1(온도 생장 요인)·G2(µmax·생장 확률)·G3(강수 검출 확률) - 1차 보고서 표 9 ② 생장식
  combo : F·G 중 검증 구간 가중 PR-AUC 시드 평균이 E00 보다 높은 묶음 -> A3 + 각각, A3 + 전부(시드 3).
          높은 묶음이 없으면 A3 + F4, A3 + G1~G3 (시드 42). 선정 = results/selection_FG.json
  m     : M1(E00 + 이동평균 차감 기상 57칸)·A3_M1(A3 + 같은 57칸, k*) 시드 42·43·44 - 1차 보고서 표 9 ② 이동평균 차감
          (사용자 승인 2026-10-19, 계산 = e2fg.py 'wxanom' 묶음, 기간 e2cfg.WXANOM_K)
  n     : A3_N1(A3 + 과거 뭉치 이력 84)·A3_N2(A3 + 그날 뭉치 멀티핫 21)·A3_N1N2·A3_N0(A3 + 제품명 TF-IDF 500) 시드 42·43·44
          - 제품명 변수(교수님 계획서 4·5절, 사용자 승인 2026-10-19, 계산 = e2nb.py)
  q     : Q1(A3 + 일별 기상 범위 12)·Q2(A3 + 토양 pH 범위 3)·Q3(A3 + 식품 범위 4)·Q4(A3 + 셋 다 19) 시드 42·43·44
          - 리서치팀 생장 최적·한계 범위(사용자 지시, [질문] 세션 전달 2026-10-19, 계산 = e2fg.py 'rgwx'·'rgsoil'·'rgfood')
          + E00_Q(E00 + 생장 범위 19, 묶음 단독 추가 - 베이스라인 기준, 사용자 결정 2026-10-19)
  p     : P1(A3 + 친환경 비율)·P2(A3 + 지하수 적합률)·P3(A3 + 둘 다)·P1p·P2p(위약판: 자료연도 안 시군구 섞기)·P4(A3 + HACCP 지정 비율)·P4p(위약판) 시드 42·43·44
          - 외부 지표(사용자 승인·확정 2026-10-19, 계산 = e2fg.py 'eco'·'gw'·'eco_p'·'gw_p')
          + E00_P(E00 + 외부 지표 3: eco·gw·haccp, 묶음 단독 추가 - 베이스라인 기준, 사용자 결정 2026-10-19)
  hx    : A3_L1(A3 + 1년 log(1+검사 수) 3)·A3_U1(A3 + 1년 스무딩 검출률 사후 표준편차 3)·A3_L1U1(둘 다 6) 시드 42·43·44
          - 과거 이력 파생변수(사용자 지시 2026-10-19, 계산 = e2hist.py 'logn'·'sd', k = k*)
  ae    : AE·VAE 이상 탐지(사용자 지시 2026-10-19, 계산 = e2ae.py) 시드 42·43·44 - 모델 AEt·VAEt(표: A3 숫자 200 + 범주 임베딩)·
          AEs·VAEs(시퀀스: DL 과 같은 50일 x 38채널). S_<모델> = 점수 단독(CatBoost 없음), A3_<모델> = A3 + 점수 1칸,
          A3_<모델>_p = A3 + 위약 점수(예측일 T 의 달력 연도 안 섞기). 점수는 (모델, 시드)마다 한 번 계산해 results/_ae_scores/ 에 저장해 같이 씀
  xs    : 작은 실험 3개(사용자 지시 2026-10-19, 계획서 11절 3순위·6절 실험 A 빈칸) 시드 42·43·44, 기존 CatBoost 설정 그대로 -
          E00_R1(E00 + 1년 스무딩 검출률 3, k*)·E00_NW(E00 에서 기상 요약 190칸 뺌: 범주 4 + h)·A3_NW(A3 에서 기상 요약 뺌: 범주 4 + h + 이력 9, k*)
  stk   : E08 스태킹(사용자 지시 2026-10-19, 계획서 7절 방식 C, 계산 = e2stk.py) 시드 42·43·44 - STK_E00(E00 + DL_lstmA + DL_trfA)·
          STK_A3(A3 + DL_lstmA + DL_trfA) 같은 시드 저장 예측의 logit 3칸 -> 로지스틱 메타 모델(검증 구간 저장 예측으로 학습, 재학습 없음).
          검증 점수는 메타 학습 구간(비교에는 시험 점수만). 이 단계는 특성 자료(Ctx2)를 만들지 않고 저장 예측만 읽음(기본 모델 결과가 results/ 에 있어야 함)
  xall  : 추가 변수 전부(사용자 지시 2026-10-19) 시드 42·43·44 - ALL_A(E00 + 단계 선택 12묶음 전부, AE 표 입력 = 나머지 입력)·
          ALL_B(ALL_A + 그날 제품명 n2·n0, 운영 전제)·E00_N2·E00_N0(E00 + 그날 제품명 묶음 하나). AE 점수가 GPU 를 씀
  fs    : 양방향 단계 선택 - 전진(사용자 결정 2026-10-19, 100일 위 2차 실험 A단계, 계산 = e2fs.py, 묶음 = e2cfg.FS_BUNDLES) 시드 42·43·44,
          CatBoost = E00 설정, k = k*. 기준 = E00 에서 시작, FS01~FS12 = 지금 기준 + 묶음 하나 -> 검증 T 통합 60일평균 PR-AUC 시드 평균이
          기준보다 높으면 남김. FS12(ae4) 의 AE 점수는 GPU 로 계산(이 단계에서 CUDA 를 먼저 쓰는 것 없음 - e2ae 주의 사항)
  bw    : 양방향 단계 선택 - 뒤로: 전진 최종 기준에서 남긴 묶음을 하나씩 뺀 후보 BW01, BW02, … -> 가장 많이 오른 것을 빼고 반복, 오른 것이 없으면 끝.
          fs·bw 는 실행 중에 후보를 정함(저장된 결과로 결정을 재생 -> 이어하기 가능). 결정 기록 = results/selection_FS.json
  tables: tables/실험기록.xlsx
환경변수: EXP2_L = 기상 창 길이(없으면 50 = 이전과 같음), EXP2_SELECT = old(없으면, 검증 구간 가중 PR-AUC) | tint(검증 T 통합 60일평균
  PR-AUC - e2eval.val_tint_pr_auc). L 이 50 이 아니면 config.json 에 창 길이 칸 'wx_L', tint 면 'select_rule' 칸, 단계 선택 후보면 'fs' 칸이
  더해짐(50·old·단계 선택 아님이면 이전과 같은 칸).
--root 폴더: results·tables·logs(progress·잠금)·cache 를 그 폴더 아래에 둔다(시험 실행용, 기본 = 이 실험 폴더).
이어하기: 결과가 완전한 실행(summary.json complete=True + 파일 5개)은 건너뜀. 불완전한 폴더는 지우고 다시 함.
  건너뛸 때 그 결과를 만든 코드(config.json 의 code_md5)와 지금 코드가 다르면 다른 파일 이름을 로그에 [주의]로 적음(건너뛰기는 그대로).
잠금: logs/run.lock (pid). 살아 있는 run.py 가 잠금을 쥐고 있으면 종료 코드 3 으로 바로 끝냄(같은 실행 두 번 띄우기 방지).
선정은 검증만 사용(구간 가중 PR-AUC, EXP2_SELECT=tint 면 검증 T 통합 60일평균 PR-AUC). 시험은 기록만.
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2cfg as E   # noqa: E402  (1차 src 를 sys.path 뒤에 추가)

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import config as C1  # noqa: E402
from common import Logger, install_tee  # noqa: E402  (1차 공통: 로그, 콘솔 복사)
import e2fg  # noqa: E402  (F·G 입력·조합 실험)
import e2nb  # noqa: E402  (N 제품명 변수)
from e2data import Ctx2  # noqa: E402
from e2eval import Mem, evaluate, write_preds  # noqa: E402
from e2models import make_runner  # noqa: E402
from e2tables import (build_tables, is_complete, jdump, jload, prev1_table, select_A,  # noqa: E402
                      select_k)


LOCK_EXIT_CODE = 3
EXP1_MODULES = ("config.py", "data.py", "features.py", "metrics.py", "models.py", "common.py")
_MD5 = None


def code_md5():
    """이 실행의 코드 지문: 계산에 쓰는 2차 모듈(run.py, e2*.py)과 import 하는 1차 모듈의 md5 (config.json 에 기록).
    점검·측정 스크립트(selftest*, timing*, h60_memtest)는 넣지 않음."""
    global _MD5
    if _MD5 is None:
        src2 = Path(__file__).resolve().parent
        files = [src2 / "run.py"] + sorted(src2.glob("e2*.py")) + [E.EXP1_SRC / f for f in EXP1_MODULES]
        _MD5 = {(f.name if f.parent == src2 else f"1차/{f.name}"): hashlib.md5(f.read_bytes()).hexdigest()
                for f in files if f.exists()}
    return _MD5


def _lock_owner_alive(pid):
    """잠금의 pid 가 살아 있는 run.py 인지. 같은 사용자가 띄운 run.py 는 명령줄을 읽을 수 있으므로
    읽을 수 없거나(AccessDenied) run.py 가 아니면 죽은 잠금으로 본다(pid 재사용)."""
    import psutil
    try:
        return any(os.path.basename(a) == "run.py" for a in psutil.Process(pid).cmdline())
    except Exception:
        return False


def acquire_lock(P):
    lock = P["logs"] / "run.lock"
    for _ in range(3):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            txt = lock.read_text(encoding="utf-8", errors="replace") if lock.exists() else ""
            kv = dict(x.split("=", 1) for x in txt.splitlines() if "=" in x)
            try:
                pid = int(kv.get("pid", ""))
            except ValueError:
                pid = None
            if pid and pid != os.getpid() and _lock_owner_alive(pid):
                print(f"[멈춤] 이미 실행 중인 run.py 가 있음(pid {pid}, 시작 {kv.get('started')}, 인자 {kv.get('args')}). "
                      f"같은 실행을 두 번 띄우지 않음. 그 실행이 끝난 뒤 다시 실행. 잠금 파일: {lock}", flush=True)
                sys.exit(LOCK_EXIT_CODE)
            try:
                lock.unlink()                          # 죽은 실행이 남긴 잠금
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\nstarted={time.strftime('%Y-%m-%d %H:%M:%S')}\nargs={' '.join(sys.argv[1:])}\n")
        return lock
    raise RuntimeError(f"잠금 파일을 만들 수 없음: {lock}")


def release_lock(lock):
    try:
        txt = lock.read_text(encoding="utf-8")
        if f"pid={os.getpid()}\n" in txt:
            lock.unlink()
    except Exception:
        pass


def check_code_changed(rdir, log, name):
    try:
        old = jload(rdir / "config.json").get("code_md5")
    except Exception:
        old = None
    if not old:
        return
    cur = code_md5()
    diff = sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k))
    if diff:
        log(f"  [주의] {name}: 이 결과를 만든 뒤 코드가 바뀜({', '.join(diff)}). 결과는 그대로 둠 - 계산에 영향 있는 변경이면 그 폴더를 지우고 다시 실행")


def resolve_feats(exp, P, smoke, log):
    f = exp["feats"]
    src = None
    if f == "A*":                                   # 매번 결과에서 다시 고름(선정 파일도 다시 씀)
        sel = select_A(P, smoke, log)
        src = sel["best"]
        f = E.EXP_BY_ID[src]["feats"]
    f = dict(f)
    if f.get("k") == "k*":
        sel = select_k(P, smoke, log)
        f["k"] = sel["k_star"]
    if isinstance(f.get("ae_tab"), dict) and f["ae_tab"].get("k") == "k*":      # 표 AE 입력(ALL_A·ALL_B)의 k* 도 같은 값으로
        f["ae_tab"] = dict(f["ae_tab"], k=select_k(P, smoke, log)["k_star"])
    if f.get("hist") and not isinstance(f.get("k"), (int, float)):
        raise ValueError(f"{exp['id']}: k 가 정해지지 않음")
    return f, src


def alignment_text():
    """config.json 'alignment' 글. L=50 이면 이전과 같은 글."""
    return f"목표일 d, 앞당김 h -> T = d - h, 기상 창 = T-{E.L} .. T-1, 이력 창 = T-P .. T-1 (T 당일 제외)"


def extra_cfg(exp=None):
    """config.json 에 더하는 칸: 창 길이 'wx_L'(L 이 50 이 아닐 때만), 'select_rule'(tint 일 때만), 'fs'(단계 선택 후보만: 기준 ID·더한/뺀 묶음).
    L=50·old·단계 선택 아님이면 빈 dict -> 이전 config 와 같은 칸."""
    d = {}
    if E.L != E.L_DEFAULT:
        d["wx_L"] = E.L
    if E.SELECT != "old":
        from e2eval import TINT_RULE
        d["select_rule"] = {"name": E.SELECT, "rule": TINT_RULE}
    if exp is not None and exp.get("fs_info"):
        d["fs"] = exp["fs_info"]
    return d


def _ae_info(blocks, tab_feats=None):
    import e2ae                                     # torch 를 쓰므로 AE 실험일 때만 import
    return e2ae.ae_info(blocks) if tab_feats is None else e2ae.ae_info(blocks, tab_feats)


def check_windows(ctx, log):
    """모든 창(학습·조기 종료 h 1..60, 평가 h 1..60)이 기상표 안인지: T = d-h, 창 T-L..T-1."""
    w = ctx.weather
    lo, hi = [], []
    for sp in C1.SPLITS:
        d = ctx.sp[sp]["d"]
        for t in (d - E.H_MAX, d - 1):
            a, b = w.check_windows(t.astype(np.int64), E.L, f"{sp} L={E.L}")
            lo.append(a)
            hi.append(b)
    r = (str(w.date0 + np.timedelta64(min(lo), "D")), str(w.date0 + np.timedelta64(max(hi), "D")))
    log(f"창 확인 통과: 모든 창이 기상표 안 ({r[0]} ~ {r[1]})")
    return r


def run_one(exp, seed, ctx, P, log, mem, smoke, win_range):
    name = E.run_dir_name(exp["id"], seed)
    rdir = P["results"] / name
    if is_complete(rdir):
        log(f"[건너뜀] {name}: 결과 완전")
        check_code_changed(rdir, log, name)
        return jload(rdir / "summary.json")
    if rdir.exists():
        log(f"  {name}: 불완전한 이전 결과 삭제 후 다시 실행")
        shutil.rmtree(rdir)
    feats, feats_from = resolve_feats(exp, P, smoke, log) if exp["model"] != "dl" else (None, None)
    rdir.mkdir(parents=True)
    log(f"[시작] {name}: {exp['desc']}" + (f" | 입력 {feats}" if feats else ""))
    t_all = time.time()
    mem.reset()
    ctx.ae_dir = P["results"] / "_ae_scores"                # AE 점수 저장 위치(이 실행의 결과 폴더 아래, e2ae - 다른 실험은 안 씀)
    runner = make_runner(ctx, exp, seed, feats, log, smoke, cache_dir=P["cache"], mem=mem)
    unit = "에폭" if exp["model"] in ("dl", "ae") else ("반복" if exp["model"] == "lr" else "라운드")
    try:
        runner.fit()
        log(f"  학습 끝: 최적 {unit} {runner.best_rounds} (학습한 수 {runner.n_rounds_trained}), "
            f"조기종료 val AP {runner.best_val_ap:.5f}, 학습 {runner.fit_seconds:.1f}s, 학습 행 {runner.n_train_rows:,}")
        t_ev = time.time()
        preds = {}
        for sp in ("val", "test"):
            n = len(runner.pos[sp])
            Pm = np.empty((E.H_MAX, n), dtype=np.float32)
            for h in range(1, E.H_MAX + 1):
                Pm[h - 1] = runner.predict(sp, h)
            preds[sp] = Pm
        eval_seconds = time.time() - t_ev
        info = runner.info()
    finally:
        runner.close()
    memd = mem.result()
    ys = {sp: ctx.sp[sp]["y"][runner.pos[sp]] for sp in ("val", "test")}
    mdf, thr, summ = evaluate(preds, ys)
    group_codes = ctx.groups[exp["group"]]["codes"] if exp.get("group") else None
    cfg = {"run": name, "exp": exp["id"], "plan": exp["plan"], "desc": exp["desc"], "model": exp["model"],
           "seed": seed, "smoke": smoke, "feats": feats, "feats_from": feats_from,
           "group": exp.get("group"), "group_codes": group_codes,
           "options": {k: exp[k] for k in ("spw", "smote", "multi_h") if k in exp},
           "alignment": alignment_text(),
           "splits": {"train": f"d <= {C1.TRAIN_END}", "val": f"{C1.VAL_START}~{C1.VAL_END}",
                      "test": f"{C1.TEST_START}~{C1.TEST_END}"},
           "rows": {sp: {"n": int(len(runner.pos[sp])), "pos": int(ctx.sp[sp]["y"][runner.pos[sp]].sum())}
                    for sp in C1.SPLITS},
           "h_assignment": f"학습 행: default_rng({seed}).integers(1,61) (전체 학습 행, 단위 키+d 정렬); "
                           f"조기 종료용 검증: default_rng({seed + 1}); 평가: h=1..60 전부"
                           + (f"; 여러 h: {exp['multi_h']}개" if exp.get("multi_h") else ""),
           "n_features": runner.n_features, "n_features_desc": runner.n_features_desc,
           "feature_columns": getattr(runner, "cols", None), "window_range": list(win_range),
           **({"ext": e2fg.ext_info(feats["ext"]),
               "cat_columns": getattr(getattr(runner, "fb", None), "cat_names", None)}
              if feats and feats.get("ext") else {}),
           **({"combo": {"groups": exp.get("combo_groups"), "kind": exp.get("combo_kind")}} if exp.get("combo_kind") else {}),
           **({"nb": e2nb.nb_info(feats["nb"])} if feats and feats.get("nb") else {}),
           **({"ae": _ae_info(feats["ae"], feats.get("ae_tab"))} if feats and feats.get("ae") else {}),
           "n_train_rows": runner.n_train_rows, "n_train_pos": getattr(runner, "n_train_pos", None),
           **extra_cfg(exp),
           "code_md5": code_md5(), "model_info": info}
    jdump(cfg, rdir / "config.json")
    mdf.to_csv(rdir / "metrics_by_h.csv", index=False, encoding="utf-8-sig")
    for sp in ("val", "test"):
        write_preds(rdir / f"pred_{sp}.parquet", ctx, sp, runner.pos[sp], preds[sp])
    extra = info.get("extra", {}) if isinstance(info, dict) else {}
    summ_out = {"run": name, "exp": exp["id"], "seed": seed, "model": exp["model"], "plan": exp["plan"],
                "smoke": smoke, "feats": feats, "feats_from": feats_from, "group": exp.get("group"),
                "group_codes": group_codes, "n_train_rows": runner.n_train_rows,
                "n_train_pos": getattr(runner, "n_train_pos", None),
                **{k: v for k, v in summ.items() if k not in ("val_overall", "test_overall")},
                "threshold_rule": "검증 예측(h 1~60 합침)에서 F2 최대 확률값, 판정 p >= 기준값", **thr,
                "val_overall": summ["val_overall"], "test_overall": summ["test_overall"],
                "best_rounds_or_epoch": runner.best_rounds, "n_rounds_trained": runner.n_rounds_trained,
                "early_stop_val_ap": runner.best_val_ap, "fit_seconds": round(runner.fit_seconds, 2),
                "feature_seconds": round(runner.feature_seconds, 2), "eval_seconds": round(eval_seconds, 2),
                "total_seconds": round(time.time() - t_all, 2), "n_features": runner.n_features, **memd,
                "smotenc": extra.get("smotenc"), "scale_pos_weight": (extra.get("scale_pos_weight") or {}).get("value"),
                "h60": extra.get("h60"), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": True}
    jdump(summ_out, rdir / "summary.json")
    log(f"[끝] {name}: 검증 구간가중 PR-AUC {summ['val_pr_auc_w']:.5f} (60일 평균 {summ['val_pr_auc_mean60']:.5f}), "
        f"시험 {summ['test_pr_auc_w']:.5f}, 기준값 {thr['threshold']:.5f}, 총 {summ_out['total_seconds']:.0f}s, "
        f"최대 메모리 private {memd['peak_private_gb']} GB / rss {memd['peak_rss_gb']} GB")
    return summ_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=E.STAGES + ["all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="실험 ID(모든 시드) 또는 실험ID__s시드")
    ap.add_argument("--console-log", default=None)
    ap.add_argument("--root", default=None, help="results·tables·logs·cache 를 이 폴더 아래에(시험 실행용)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if args.console_log:
        install_tee(args.console_log)
    P = E.paths(args.smoke)
    if args.root:
        R = Path(args.root).resolve()
        sub = "_smoke" if args.smoke else ""
        P = {"results": R / "results" / sub, "tables": R / "tables" / sub, "logs": R / "logs", "cache": R / "cache",
             "progress": R / "logs" / ("progress_smoke.log" if args.smoke else "progress.log")}
    for k in ("results", "tables", "logs", "cache"):
        P[k].mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(P)
    try:
        _main(args, P)
    finally:
        release_lock(lock)


def _main(args, P):
    log = Logger(P["progress"])
    log(f"===== run.py {' '.join(sys.argv[1:])} | python {sys.version.split()[0]} | pid {os.getpid()}")
    only_id, only_seed = None, None
    if args.only:
        if "__s" in args.only:
            only_id, s = args.only.split("__s")
            only_seed = int(s)
        else:
            only_id = args.only
        if only_id not in E.EXP_BY_ID and not e2fg.is_combo_id(only_id) and args.stage not in ("fs", "bw"):
            raise ValueError(f"--only {args.only}: 실험 ID 없음. 가능: {list(E.EXP_BY_ID)} 또는 조합 A3_<F1~F4·G1~G3 이어 붙임>")
    stages = E.STAGES if args.stage == "all" else [args.stage]
    if only_id and args.stage not in ("fs", "bw"):
        st_of = E.EXP_BY_ID[only_id]["stage"] if only_id in E.EXP_BY_ID else "combo"
        stages = [st for st in stages if st == st_of] + (["tables"] if args.stage == "all" else [])
    ctx, win, mem = None, None, Mem()
    mem.start()

    def ensure_ctx():
        nonlocal ctx, win
        if ctx is None:
            ctx = Ctx2(args.smoke, log, cache_dir=P["cache"])
            win = check_windows(ctx, log)
            jdump({k: v for k, v in ctx.groups.items()}, P["results"] / "groups.json")
            ctx.data_summary().to_csv(P["results"] / "data_summary.csv", index=False, encoding="utf-8-sig")
        return ctx

    try:
        for st in stages:
            if st == "prev1":
                prev1_table(P, log)
                continue
            if st == "tables":
                build_tables(P, args.smoke, log)
                continue
            if st in ("fs", "bw"):                  # 양방향 단계 선택: 저장된 결과로 결정을 재생하며 모자란 후보를 차례로 실행(e2fs)
                import e2fs
                if only_id:
                    log(f"  --only {args.only}: fs·bw 단계는 후보를 실행 중에 정하므로 --only 를 쓰지 않음 -> 무시")
                ensure_ctx()
                k_star = select_k(P, args.smoke, log)["k_star"]
                e2fs.run_stage(st, P, args.smoke, log, lambda e, s: run_one(e, s, ctx, P, log, mem, args.smoke, win), k_star)
                continue
            if st == "combo":                       # F·G 결과로 그때 정함(결과가 다 없으면 오류로 멈춤)
                plan, _sel = e2fg.combo_experiments(P, args.smoke, log)
                exps = [e for e in plan if only_id is None or e["id"] == only_id]
                if only_id and not exps:
                    log(f"  --only {only_id}: 지금 선정 결과의 조합 목록 {[e['id'] for e in plan]} 에 없음 -> 실행 안 함")
            else:
                exps = [e for e in E.EXPERIMENTS if e["stage"] == st and (only_id is None or e["id"] == only_id)]
            todo = [(e, s) for e in exps for s in E.exp_seeds(e, args.smoke)
                    if (only_seed is None or s == only_seed)]
            left = [(e, s) for e, s in todo if not is_complete(P["results"] / E.run_dir_name(e["id"], s))]
            log(f"--- {st}: 실행 {len(todo)}개 중 남은 것 {len(left)}개: {[E.run_dir_name(e['id'], s) for e, s in left]}")
            if st == "dl":
                import e2dl
                miss = sorted({e["id"] for e, _ in left if e["id"] not in e2dl.DL_REGISTRY})
                if miss:
                    log(f"  DL 미구현(e2dl.DL_REGISTRY 에 없음) -> 건너뜀: {miss}")
                left = [(e, s) for e, s in left if e["id"] in e2dl.DL_REGISTRY]
            if not left:
                continue
            if st == "h60" and not args.smoke:              # 이 PC 보호: H60 은 사용 가능 메모리가 충분한 곳(팀 공용 서버)에서만
                import psutil
                av = psutil.virtual_memory().available / 1e9
                if av < E.H60_RAM_NEED_GB:
                    log(f"  [건너뜀] h60: 시스템 사용 가능 메모리 {av:.1f} GB < {E.H60_RAM_NEED_GB} GB - H60 은 서버에서 실행(e2cfg.H60_RAM_NEED_GB)")
                    continue
                log(f"  시스템 사용 가능 메모리 {av:.1f} GB (h60 시작 조건 {E.H60_RAM_NEED_GB} GB 이상)")
            if st in ("h10", "hbig") and not args.smoke:
                import psutil
                av = psutil.virtual_memory().available / 1e9
                need = {"h10": 7.6, "hbig": 8.3}[st]
                log(f"  시스템 사용 가능 메모리 {av:.1f} GB (이 단계 측정 최대 private 약 {need} GB)"
                    + ("" if av >= 10 else " - [주의] 10 GB 미만: 브라우저·한글 등 큰 프로그램을 닫기 권장(멈추면 이어하기 가능)"))
            if st == "stk":                                 # E08 스태킹: 저장 예측만 읽음 - 특성 자료(Ctx2)를 만들지 않음(메모리)
                import e2stk
                for e, s in left:
                    e2stk.run_one_stack(e, s, P, log, mem, args.smoke, code_md5)
                continue
            ensure_ctx()
            for e, s in left:
                run_one(e, s, ctx, P, log, mem, args.smoke, win)
        log(f"===== 완료: {args.stage}" + (f" --only {args.only}" if args.only else ""))
    except Exception:
        log("!!!!! 오류로 멈춤:\n" + traceback.format_exc())
        raise
    finally:
        mem.stop()


if __name__ == "__main__":
    main()
