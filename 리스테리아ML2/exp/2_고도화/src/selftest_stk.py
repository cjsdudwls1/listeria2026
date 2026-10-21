# -*- coding: utf-8 -*-
"""E08 스태킹(STK_E00·STK_A3, 단계 'stk', e2stk.py, 사용자 지시 2026-10-19) 자체 점검. 결과: logs/selftest_stk.txt (UTF-8). 실패가 있으면 종료 코드 1.
특성 자료(Ctx2)를 만들지 않음. 기본 모델 결과(E00·A3·DL_lstmA·DL_trfA 시드 42·43·44)는 <root>/results 에서 읽기만 하고,
점검용 결과·표는 scratch/stk/selftest_stk/ 아래에만 씀(results/·tables/·cache/ 에 쓰지 않음).
--root: 기본 모델 결과를 읽을 곳(기본 = 이 실험 폴더). 서버에서는 --root ~/cyj_listeria/root.

(0) 설정: STK_E00·STK_A3 단계 'stk'·모델 'stack'·기본 모델·계획서 대응 'E08 스태킹'·시드 42·43·44, STAGES 에서 'stk' 가 'ae'·'xs' 뒤 'tables'(끝) 앞,
    run_exp2.ps1 ValidateSet 에 'stk', 표 모델 이름 '스태킹(로지스틱 메타)', 조합 ID 와 겹치지 않음, 설정 값(clip 1e-6, max_iter 10000)
(a) 키 맞춤: 시드 42·43·44 x 실험 2개 x 검증·시험 - e2stk.align 결과 = 따로 만든 키 사전('sgg|p2|p3|d|h' 문자열 -> 행 번호)으로 찾은 p·y (비트 단위)
(b) 대조군(시드 42, STK_E00, 검증): 기본 모델 파일의 행 순서만 섞음 -> 키로 다시 맞춰 같은 결과(순서를 믿지 않음),
    행을 섞어 키와 y·p 의 짝을 끊음(키 칸만 섞음) -> 키 검사(같은 키의 y 다름)에서 멈춤, 한 행 빠짐·키 하나 바뀜·키 중복·y 하나 바뀜·첫 파일 h 순서 깨짐 -> 멈춤
(c) 같은 입력 두 번: run_one_stack 을 두 폴더에 각각(시드 42, 실험 2개) -> 예측 파일 p·metrics_by_h.csv·summary 지표·config(시간 칸 제외) 같음, 결과 폴더 완전
(d) 메타 계수 재현(run.py 결과 6개): (a)처럼 따로 맞춘 p 로 logit 을 따로 계산해 LogisticRegression(C=1, l1_ratio=0, lbfgs, 절편, max_iter 10000,
    random_state=시드)을 다시 학습 -> config.json 메타 계수·절편과 같음, 저장 예측 = 1/(1+exp(−(X·계수 + 절편))) (float32 반올림 안), 수렴,
    summary 지표·판정 기준값 = 저장 예측으로 e2eval.evaluate 다시 계산한 값, 기준값 = 1차 metrics.f2_threshold(검증 h 1~60 합침)
(e) 예측 파일 = E00 예측 파일과 칸·스키마·행 수·행 순서·키·h·y 같음(p 만 다름), zstd, metrics_by_h.csv 칸 = E00, summary 키 ⊇ E00 summary 키,
    config.json·summary.json 에 주의 글('메타 모델을 검증 구간으로 학습 …')·메타 계수
(f) run.py 통합: 임시 --root(기본 모델 결과 폴더를 연결)에서 run.py --stage stk -> 실행 6개 모두 완전, 진행 기록에 특성 자료 준비 줄('데이터 준비') 없음,
    시드 42 예측 = (c) 결과(비트 단위), 다시 실행하면 6개 모두 건너뜀
(g) 기존 실험 그대로: 고치기 전 e2cfg 실험 목록 = 지금(새 실험 5개만 더해짐), 고치지 않은 파일 md5 = 고치기 전(scratch/stk/md5_before_stk.json),
    같은 결과 폴더(<root>/results 연결 + (f) 스태킹 결과)에서 고치기 전 e2tables(실험 목록도 고치기 전)·지금 e2tables 로 각각 만든 표:
    기존 실험 줄(실험기록·시드별·고위험품목비교) 같음, 새 줄 = xs 3 + stk 2, 스태킹 줄 모델 이름·입력·비고(주의 글)·순위 제외, 설정 시트 기존 줄 그대로 + XS·STK 줄
    (자료 파일이 없는 곳(서버)에서는 설정 시트 줄 중 자료를 읽는 고치지 않은 함수(e2fg·e2nb)가 고치기 전·지금 표 모두에 같은 대체 줄을 넣음 - 기록에 남김)
"""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import config as C1  # noqa: E402
import metrics as M1  # noqa: E402
import e2stk as S  # noqa: E402
from common import Logger  # noqa: E402
from e2eval import Mem, evaluate, read_preds  # noqa: E402
from e2tables import MODEL_NAME, is_complete, jload  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_stk.txt"
SCR = E.EXP2_DIR / "scratch" / "stk"
WORK = SCR / "selftest_stk"
OLD_E2CFG = SCR / "e2cfg_before_stk.py.bak"
OLD_E2TABLES = SCR / "e2tables_before_stk.py.bak"
MD5_BEFORE = SCR / "md5_before_stk.json"
STK_IDS = ["STK_E00", "STK_A3"]
NEW_IDS = ["E00_R1", "E00_NW", "A3_NW"] + STK_IDS
BASES = ["E00", "A3", "DL_lstmA", "DL_trfA"]
SEEDS = [42, 43, 44]
CHANGED = {"e2cfg.py", "run.py", "e2tables.py", "selftest_hx.py", "run_exp2.ps1"}
NEW_FILES = {"e2stk.py", "selftest_xs.py", "selftest_stk.py"}
VOLATILE = {"fit_seconds", "feature_seconds", "eval_seconds", "total_seconds", "peak_rss_gb", "peak_private_gb",
            "peak_gpu_used_mb_device", "system_min_available_gb", "finished_at"}
lines, fails = [], []


def say(msg):
    lines.append(msg)
    try:
        print(msg, flush=True)
    except Exception:
        pass


def check(name, ok, detail=""):
    say(f"[{'통과' if ok else '실패'}] {name} {detail}")
    if not ok:
        fails.append(name)


def load_mod(name, path):
    ld = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


def pred_path(results, eid, seed, sp):
    return results / E.run_dir_name(eid, seed) / f"pred_{sp}.parquet"


# ---------------------------------------------------------------- (0)
def t_config():
    ok = True
    for b in ("E00", "A3"):
        e = E.EXP_BY_ID[f"STK_{b}"]
        ok &= (e["stage"] == "stk" and e["model"] == "stack" and e["bases"] == [b, "DL_lstmA", "DL_trfA"] and e["plan"] == "E08 스태킹"
               and E.exp_seeds(e, False) == SEEDS and "feats" not in e)
    ids = [e["id"] for e in E.EXPERIMENTS if e["stage"] == "stk"]
    check("(0) STK_E00·STK_A3: 단계 'stk', 모델 'stack', 기본 모델 [CatBoost, DL_lstmA, DL_trfA], 계획서 대응 'E08 스태킹', 시드 42·43·44",
          ok and ids == STK_IDS, str(ids))
    st = E.STAGES
    check("(0) STAGES: 'stk' 가 'ae'·'xs' 뒤, 'tables'(끝) 앞", st[-1] == "tables" and st.index("ae") < st.index("xs") < st.index("stk") < st.index("tables"),
          str(st[-6:]))
    ps1 = (E.EXP2_DIR / "run_exp2.ps1").read_text(encoding="utf-8-sig")
    vs = [ln for ln in ps1.splitlines() if "ValidateSet" in ln]
    check("(0) run_exp2.ps1 ValidateSet 에 'stk'", len(vs) == 1 and "'stk'" in vs[0], vs[0].strip() if vs else "")
    import e2fg
    check("(0) 표 모델 이름 '스태킹(로지스틱 메타)', 조합 ID 와 겹치지 않음, 설정 clip 1e-6·max_iter 10000",
          MODEL_NAME.get("stack") == "스태킹(로지스틱 메타)" and not any(e2fg.is_combo_id(i) for i in NEW_IDS)
          and E.STK == {"clip": 1e-6, "max_iter": 10000})


# ---------------------------------------------------------------- (a) 따로 만든 키 사전
def key_str(t):
    s = None
    for c in S.CODE_COLS:
        v = pd.Series(t.column(c).cast(pa.string()).to_pylist(), dtype=object)
        s = v if s is None else s + "|" + v
    s = s + "|" + pd.Series(t.column("d").cast(pa.int32()).to_numpy()).astype(str) + "|" + pd.Series(t.column("h").to_numpy()).astype(str)
    return s.to_numpy(dtype=object)


def ref_align(tabs, keys):
    """tabs·keys = 기본 모델 순서. 첫 표 행 순서로 p·y 를 찾아 맞춤(키 사전). 돌려줌: (Pm (k, N) float32, y (N,), 문제 목록)."""
    k0 = keys[0]
    prob = []
    if len(set(k0)) != len(k0):
        prob.append("첫 표 키 중복")
    Pm = np.empty((len(tabs), len(k0)), dtype=np.float32)
    y0 = tabs[0].column("y").to_numpy().astype(np.int8)
    Pm[0] = tabs[0].column("p").to_numpy()
    for i in range(1, len(tabs)):
        idx = pd.Series(np.arange(len(keys[i])), index=keys[i])
        if not idx.index.is_unique or len(keys[i]) != len(k0):
            prob.append(f"{i}: 키 중복·행 수")
            continue
        pos = idx.reindex(k0).to_numpy()
        if np.isnan(pos.astype(np.float64)).any():
            prob.append(f"{i}: 없는 키")
            continue
        pos = pos.astype(np.int64)
        if not np.array_equal(tabs[i].column("y").to_numpy().astype(np.int8)[pos], y0):
            prob.append(f"{i}: y 다름")
        Pm[i] = tabs[i].column("p").to_numpy()[pos]
    return Pm, y0, prob


def t_align(root):
    R = root / "results"
    bad, n, same_order, ref = [], 0, [], {}
    for seed in SEEDS:
        for sp in ("val", "test"):
            tabs = {b: pq.read_table(pred_path(R, b, seed, sp)) for b in BASES}
            keys = {b: key_str(tabs[b]) for b in BASES}
            for eid in STK_IDS:
                bs = E.EXP_BY_ID[eid]["bases"]
                Pm, y, chk = S.align([tabs[b] for b in bs], bs)
                Pr, yr, prob = ref_align([tabs[b] for b in bs], [keys[b] for b in bs])
                n += 1
                ok = not prob and np.array_equal(Pm, Pr) and np.array_equal(y, yr) and chk["n_rows"] == tabs[bs[0]].num_rows
                if not ok:
                    bad.append((eid, seed, sp, prob))
                same_order.append(all(v["same_row_order_as_first"] for v in chk["others"].values()))
                ref[(eid, seed, sp)] = (Pr, yr)
            del tabs, keys
    check("(a) 키 맞춤: e2stk.align = 따로 만든 키 사전으로 찾은 p·y (비트 단위), 키 중복·빠짐 없음", not bad and n == 12,
          f"{n}개(시드 3 x 실험 2 x 분할 2), 다름 {bad}")
    say(f"    [기록] 기본 모델 파일 행 순서가 첫 파일과 같았던 경우 {sum(same_order)}/{len(same_order)} (같아도 키로 맞춤)")
    return ref


# ---------------------------------------------------------------- (b) 대조군
def _set(t, name, arr):
    j = t.schema.get_field_index(name)
    return t.set_column(j, t.schema.field(j), arr)


def _raises(fn):
    try:
        fn()
        return None
    except S.StackKeyError as e:
        return str(e)


def t_controls(root):
    R = root / "results"
    eid, seed, sp = "STK_E00", 42, "val"
    bs = E.EXP_BY_ID[eid]["bases"]
    tabs = [pq.read_table(pred_path(R, b, seed, sp)) for b in bs]
    P0, y0, _ = S.align(tabs, bs)
    N = tabs[0].num_rows
    rng = np.random.default_rng(20261019)
    perm = pa.array(rng.permutation(N))
    t1 = tabs[1].take(perm)
    P1, y1, c1 = S.align([tabs[0], t1, tabs[2]], bs)
    check("(b) 대조군: DL_lstmA 파일 행 순서만 섞음 -> 키로 다시 맞춰 같은 p·y (비트 단위, 순서를 믿지 않음)",
          np.array_equal(P0, P1) and np.array_equal(y0, y1) and not c1["others"][bs[1]]["same_row_order_as_first"])
    tk = tabs[1]
    for c in S.KEY_COLS:                                        # 키 칸만 섞음: 키와 y·p 의 짝이 끊김
        tk = _set(tk, c, tabs[1].column(c).take(perm))
    e_mix = _raises(lambda: S.align([tabs[0], tk, tabs[2]], bs))
    check("(b) 대조군: 행을 섞어 키와 y·p 의 짝을 끊으면(키 칸만 섞음) -> 키 검사에서 멈춤", e_mix is not None and "y" in e_mix, f"{e_mix}")
    j, k = 12345, 54321
    errs = {}
    errs["한 행 빠짐(DL_trfA)"] = _raises(lambda: S.align([tabs[0], tabs[1], tabs[2].slice(1)], bs))
    d = tabs[2].column("d").to_numpy().copy()
    d[j] = d[j] + np.timedelta64(400, "D")
    errs["키 하나 바뀜(d + 400일)"] = _raises(lambda: S.align([tabs[0], tabs[1], _set(tabs[2], "d", pa.array(d, type=pa.date32()))], bs))
    idx = np.arange(N)
    idx[j] = k
    td = tabs[1]
    for c in S.KEY_COLS:
        td = _set(td, c, tabs[1].column(c).take(pa.array(idx)))
    errs["키 중복(DL_lstmA)"] = _raises(lambda: S.align([tabs[0], td, tabs[2]], bs))
    yy = tabs[2].column("y").to_numpy().astype(np.int8).copy()
    yy[j] = 1 - yy[j]
    errs["y 하나 바뀜(DL_trfA)"] = _raises(lambda: S.align([tabs[0], tabs[1], _set(tabs[2], "y", pa.array(yy, type=pa.int8()))], bs))
    errs["첫 파일(E00) 행 순서 깨짐(h 우선 아님)"] = _raises(lambda: S.align([tabs[0].take(perm), tabs[1], tabs[2]], bs))
    check("(b) 대조군: 한 행 빠짐·키 하나 바뀜·키 중복·y 하나 바뀜·첫 파일 h 순서 깨짐 -> 모두 멈춤(StackKeyError)",
          all(v is not None for v in errs.values()), "; ".join(f"{a} -> {b}" for a, b in errs.items()))


# ---------------------------------------------------------------- (c) 같은 입력 두 번
def _strip_cfg(c):
    c = json.loads(json.dumps(c))
    for k in ("read_seconds", "pred_seconds"):
        c["model_info"].pop(k, None)
    return c


def t_twice(root, log, mem, code_md5):
    outs = {}
    for tag in ("runA", "runB"):
        R = WORK / tag / "results"
        if R.parent.exists():
            shutil.rmtree(R.parent)
        R.mkdir(parents=True)
        P = {"results": R}
        for eid in STK_IDS:
            S.run_one_stack(E.EXP_BY_ID[eid], 42, P, log, mem, False, code_md5, base_results=root / "results")
        outs[tag] = R
    bad = []
    for eid in STK_IDS:
        a, b = (outs[t] / E.run_dir_name(eid, 42) for t in ("runA", "runB"))
        ok = is_complete(a) and is_complete(b)
        for sp in ("val", "test"):
            pa_, pb_ = (pq.read_table(x / f"pred_{sp}.parquet") for x in (a, b))
            ok &= pa_.equals(pb_)
        ok &= (a / "metrics_by_h.csv").read_bytes() == (b / "metrics_by_h.csv").read_bytes()
        sa, sb = jload(a / "summary.json"), jload(b / "summary.json")
        ok &= {k: v for k, v in sa.items() if k not in VOLATILE} == {k: v for k, v in sb.items() if k not in VOLATILE}
        ok &= _strip_cfg(jload(a / "config.json")) == _strip_cfg(jload(b / "config.json"))
        if not ok:
            bad.append(eid)
    check("(c) 같은 입력 두 번(다른 폴더): 예측 파일·metrics_by_h.csv·summary(시간·메모리 칸 제외)·config(시간 칸 제외) 같음, 결과 폴더 완전",
          not bad, f"다름 {bad}")
    return outs["runA"]


# ---------------------------------------------------------------- (f) run.py 통합
def _link(src, dst):
    """결과 폴더는 연결(읽기만), 파일(selection_*.json 등 - 표 만들 때 다시 쓰일 수 있음)은 복사해 원본에 쓰지 않음."""
    if src.is_dir():
        try:
            os.symlink(src, dst, target_is_directory=True)
            return
        except (OSError, NotImplementedError):
            shutil.copytree(src, dst)
            return
    shutil.copy2(src, dst)


def t_runpy(root, runA):
    froot = WORK / "runroot"
    if froot.exists():
        shutil.rmtree(froot)
    R = froot / "results"
    R.mkdir(parents=True)
    n_link = 0
    for p in sorted((root / "results").iterdir()):               # 기본 모델·k 비교 등 결과 폴더·파일을 연결(읽기만)
        if p.name.startswith("_") or p.name.startswith("STK_"):
            continue
        _link(p, R / p.name)
        n_link += 1
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", str(Path(__file__).resolve().parent / "run.py"), "--stage", "stk", "--root", str(froot)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=7200)
    sec = time.time() - t0
    prog = (froot / "logs" / "progress.log").read_text(encoding="utf-8") if (froot / "logs" / "progress.log").exists() else ""
    done = [E.run_dir_name(e, s) for e in STK_IDS for s in SEEDS if is_complete(R / E.run_dir_name(e, s))]
    check("(f) run.py --stage stk(임시 --root): 종료 코드 0, 실행 6개 모두 완전, 특성 자료 준비 줄('데이터 준비') 없음",
          r.returncode == 0 and len(done) == 6 and "데이터 준비" not in prog and "[끝] STK_A3__s44" in prog,
          f"종료 코드 {r.returncode}, 완전 {len(done)}/6, {sec:.0f}s, 연결 {n_link}개" + ("" if r.returncode == 0 else f"\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"))
    same = all(pq.read_table(R / E.run_dir_name(e, 42) / f"pred_{sp}.parquet").equals(pq.read_table(runA / E.run_dir_name(e, 42) / f"pred_{sp}.parquet"))
               for e in STK_IDS for sp in ("val", "test"))
    check("(f) run.py 결과(시드 42) 예측 파일 = (c) 결과(다른 프로세스, 비트 단위)", same)
    for ln in prog.splitlines():
        if "[끝] STK_" in ln:
            say("    " + ln.strip())
    r2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=600)
    prog2 = (froot / "logs" / "progress.log").read_text(encoding="utf-8")
    check("(f) 다시 실행하면 6개 모두 건너뜀(남은 것 0개)", r2.returncode == 0 and prog2.count("--- stk: 실행 6개 중 남은 것 0개") == 1, f"종료 코드 {r2.returncode}")
    return froot


# ---------------------------------------------------------------- (d)(e)
def t_meta_and_format(root, froot, ref):
    from sklearn.linear_model import LogisticRegression
    R0, R = root / "results", froot / "results"
    c = E.STK["clip"]
    bad_coef, bad_pred, bad_eval, bad_fmt, notes = [], [], [], [], []
    for eid in STK_IDS:
        for seed in SEEDS:
            d = R / E.run_dir_name(eid, seed)
            cfg, sm = jload(d / "config.json"), jload(d / "summary.json")
            mi = cfg["model_info"]
            names = [f"logit_{b}" for b in E.EXP_BY_ID[eid]["bases"]]
            X = {}
            for sp in ("val", "test"):
                Pr, _ = ref[(eid, seed, sp)]
                p = np.clip(Pr.astype(np.float64), c, 1 - c)              # logit = log p − log(1 − p) (사양 식, 따로 계산)
                X[sp] = np.ascontiguousarray((np.log(p) - np.log1p(-p)).T)
            yv = ref[(eid, seed, "val")][1]
            m = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True, max_iter=10000, random_state=seed).fit(X["val"], yv)
            co = np.array([mi["meta_coef"][n] for n in names])
            dc = float(np.max(np.abs(m.coef_[0] - co))) if list(mi["meta_coef"]) == names else np.inf
            di = abs(float(m.intercept_[0]) - mi["meta_intercept"])
            notes.append(f"{eid}__s{seed} 계수 차 {dc:.2e} 절편 차 {di:.2e} 반복 {mi['n_iter']}")
            if not (dc <= 1e-9 * max(1.0, np.max(np.abs(co))) and di <= 1e-9 * max(1.0, abs(mi["meta_intercept"])) and mi["converged"]
                    and mi["n_iter"] < 10000 and sm["meta_coef"] == mi["meta_coef"]):
                bad_coef.append(f"{eid}__s{seed}")
            preds, ys = {}, {}
            for sp in ("val", "test"):
                p_s, y_s, _ = read_preds(d / f"pred_{sp}.parquet")
                z = X[sp] @ co + mi["meta_intercept"]
                ph = (1.0 / (1.0 + np.exp(-z))).reshape(E.H_MAX, -1)
                if np.max(np.abs(ph - p_s.astype(np.float64))) > 1e-6 * max(1.0, float(np.max(ph))) + 1e-7:
                    bad_pred.append(f"{eid}__s{seed} {sp}")
                preds[sp], ys[sp] = p_s, y_s
            mdf, thr, summ = evaluate(preds, ys)
            keys = [k for k in summ if k not in ("val_overall", "test_overall")]
            thr2 = M1.f2_threshold(np.tile(ys["val"], E.H_MAX), preds["val"].reshape(-1))["threshold"]
            if not (all(sm[k] == summ[k] for k in keys) and sm["threshold"] == thr["threshold"] == thr2
                    and sm["val_overall"] == summ["val_overall"] and sm["test_overall"] == summ["test_overall"]):
                bad_eval.append(f"{eid}__s{seed}")
            # (e) 형식 = E00 예측 파일
            e0 = R0 / E.run_dir_name("E00", seed)
            ok = True
            for sp in ("val", "test"):
                a, b = pq.read_table(e0 / f"pred_{sp}.parquet"), pq.read_table(d / f"pred_{sp}.parquet")
                ok &= (a.schema.equals(b.schema) and pq.read_schema(e0 / f"pred_{sp}.parquet").equals(pq.read_schema(d / f"pred_{sp}.parquet"))
                       and a.num_rows == b.num_rows and a.column_names == b.column_names)
                for col in S.KEY_COLS + ["y"]:
                    ok &= a.column(col).equals(b.column(col))
                ok &= b.schema.field("p").type == pa.float32() and not a.column("p").equals(b.column("p"))
                ok &= pq.ParquetFile(d / f"pred_{sp}.parquet").metadata.row_group(0).column(0).compression == "ZSTD"
            m0 = pd.read_csv(e0 / "metrics_by_h.csv", nrows=1)
            m1 = pd.read_csv(d / "metrics_by_h.csv", nrows=1)
            s0 = jload(e0 / "summary.json")
            ok &= list(m0.columns) == list(m1.columns) and set(s0) <= set(sm)
            ok &= cfg.get("stack_note") == S.NOTE == sm.get("stack_note") and mi.get("note") == S.NOTE and list(mi["meta_coef"]) == names
            ok &= cfg["model"] == "stack" and cfg["n_features"] == 3 and cfg["feature_columns"] == names and sm["complete"] is True
            if not ok:
                bad_fmt.append(f"{eid}__s{seed}")
    for t in notes:
        say("    [기록] " + t)
    check("(d) 메타 계수 재현: 따로 맞춘 p·따로 계산한 logit 으로 다시 학습한 계수·절편 = config.json (상대 1e-9), 수렴, summary 계수 = config",
          not bad_coef, f"다름 {bad_coef}")
    check("(d) 저장 예측 = 1/(1+exp(−(X·계수 + 절편))) (float32 반올림 안)", not bad_pred, f"다름 {bad_pred}")
    check("(d) summary 지표·판정 기준값 = 저장 예측으로 e2eval.evaluate 다시 계산, 기준값 = metrics.f2_threshold(검증 h 1~60 합침)", not bad_eval,
          f"다름 {bad_eval}")
    check("(e) 예측 파일 = E00 예측 파일과 칸·스키마·행 수·순서·키·h·y 같음(p 만 다름, float32), zstd, metrics 칸 = E00, summary 키 ⊇ E00, "
          "주의 글·메타 계수 기록", not bad_fmt, f"다름 {bad_fmt}")


# ---------------------------------------------------------------- (g)
def t_old_cfg_md5():
    O = load_mod("e2cfg_old_stk2", OLD_E2CFG)
    old_ids = [e["id"] for e in O.EXPERIMENTS]
    cur = [e for e in E.EXPERIMENTS if e["id"] in set(old_ids)]
    added = [e["id"] for e in E.EXPERIMENTS if e["id"] not in set(old_ids)]
    check("(g) 실험 목록: 고치기 전 실험(설정 dict 전부·순서) = 지금, 더해진 것 = xs 3 + stk 2",
          [e["id"] for e in cur] == old_ids and all(a == b for a, b in zip(O.EXPERIMENTS, cur)) and added == NEW_IDS, f"더해진 것 {added}")
    before = json.loads(MD5_BEFORE.read_text(encoding="utf-8"))
    src2 = Path(__file__).resolve().parent
    bad = []
    for k, v in before.items():
        if k in CHANGED:
            continue
        p = (E.EXP1_SRC / k.split("/", 1)[1]) if k.startswith("1차/") else src2 / k
        if not p.exists() or hashlib.md5(p.read_bytes()).hexdigest() != v:
            bad.append(k)
    extra = sorted(f.name for f in src2.glob("*.py") if f.name not in before and f.name not in NEW_FILES)
    check("(g) 고치지 않은 파일 md5 = 고치기 전(계산 모듈·1차 포함 -> 기존 실험 특성 행렬 코드 그대로)", not bad and not extra,
          f"비교 {len(before) - len(CHANGED)}개, 다름 {bad}, 목록에 없던 새 파일 {extra}")


def _stub_data_rows(stubbed):
    """자료 파일이 없는 곳(서버: 시료·기상 2개만 올림)에서 설정 시트 줄 함수 중 자료를 읽는 것(고치지 않은 e2fg·e2nb 코드, 고치기 전·지금 표가
    똑같이 부름)이 파일 없음으로 멈추면 같은 대체 줄을 돌려주게 감쌈. 돌려줌: 원래 함수(되돌리기용). stubbed: 대체된 함수 이름 기록."""
    import e2fg
    import e2nb
    orig = {}
    for mod, name in ((e2fg, "g_notes"), (e2fg, "wxanom_settings_rows"), (e2fg, "rg_settings_rows"), (e2fg, "p_settings_rows"),
                      (e2nb, "nb_settings_rows")):
        f = getattr(mod, name)
        orig[(mod, name)] = f

        def w(*a, _f=f, _n=name, **k):
            try:
                return _f(*a, **k)
            except OSError:
                stubbed.add(_n)
                return [(f"점검용 대체 줄({_n})", "자료 파일 없음 - 고치기 전·지금 표에 똑같이 들어감")]
        setattr(mod, name, w)
    return orig


def t_tables(froot):
    import e2tables as T
    O = load_mod("e2tables_old_stk", OLD_E2TABLES)
    out = WORK / "tables"
    if out.exists():
        shutil.rmtree(out)
    full = list(E.EXPERIMENTS)
    old_exps = [e for e in full if e["stage"] not in ("xs", "stk")]
    stubbed = set()
    orig = _stub_data_rows(stubbed)
    try:
        for name, mod, exps in (("old", O, old_exps), ("new", T, full)):
            P = {"results": froot / "results", "tables": out / name, "logs": out / "logs", "cache": out / "cache"}
            P["tables"].mkdir(parents=True, exist_ok=True)
            pv = P["tables"] / "1차_가중지표.csv"                 # 1차 결과가 없는 곳(서버)에서도 돌게: 1차 가중 지표 시트는 비교 대상 아님
            if not pv.exists() and not E.EXP1_RESULTS.exists():
                pv.write_text("비고\n점검용 - 1차 결과 폴더 없음\n", encoding="utf-8-sig")
            E.EXPERIMENTS = exps
            mod.build_tables(P, False, None)
    finally:
        E.EXPERIMENTS = full
        for (mod_, name_), f_ in orig.items():
            setattr(mod_, name_, f_)
    if stubbed:
        say(f"    [기록] 자료 파일이 없어 설정 시트 대체 줄로 바꾼 함수(고치지 않은 코드, 고치기 전·지금 표 같음): {sorted(stubbed)}")
    rd = lambda n, f: pd.read_csv(out / n / f, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    old, new = rd("old", "실험기록.csv"), rd("new", "실험기록.csv")
    new_old = new[new["실험 ID"].isin(old["실험 ID"])].reset_index(drop=True)
    same_cols = list(new.columns) == list(old.columns)
    diff, fmt_only = [], []
    if same_cols and len(new_old) == len(old):
        for col in old.columns:
            m = (old[col] != new_old[col]).to_numpy()
            if not m.any():
                continue
            a = pd.to_numeric(old.loc[m, col], errors="coerce").to_numpy()
            b = pd.to_numeric(new_old.loc[m, col], errors="coerce").to_numpy()
            if not np.isnan(a).any() and np.array_equal(a, b):
                fmt_only.append(col)
            else:
                diff.append((col, int(m.sum()), list(old.loc[m, "실험 ID"])[:5]))
    added = [i for i in new["실험 ID"] if i not in set(old["실험 ID"])]
    if fmt_only:
        say(f"    [기록] 표기만 다른 칸(값 같음 - 결과 없는 줄 때문에 정수가 '.0' 붙은 실수로 써짐): {fmt_only}")
    check("(g) 실험기록 표(같은 결과 폴더에서 고치기 전·지금 코드로 각각): 기존 실험 줄 칸·값·순서 같음, 새 줄 = xs 3 + stk 2",
          same_cols and len(new_old) == len(old) and list(new_old["실험 ID"]) == list(old["실험 ID"]) and not diff and added == NEW_IDS,
          f"기존 {len(old)}줄, 다른 칸 {diff}, 새 줄 {added}")
    rank_col = [c for c in new.columns if c.startswith("순위(")][0]
    ok = True
    for eid in STK_IDS:
        r = new[new["실험 ID"] == eid].iloc[0]
        ok &= (r["모델"] == "스태킹(로지스틱 메타)" and r["입력"] == S.input_desc(E.EXP_BY_ID[eid]) and S.NOTE in r["비고"]
               and r[rank_col] == "" and r["순위 참고"] == S.RANK_NOTE and r["시드 수"] == "3/3" and r["계획서 대응"] == "E08 스태킹")
        say(f"    [기록] {eid}: 검증 구간가중 PR-AUC {r['검증 구간가중 PR-AUC (평균±표준편차)']}(메타 학습 구간), 시험 {r['시험 구간가중 PR-AUC (평균±표준편차)']}")
    for eid in NEW_IDS[:3]:
        r = new[new["실험 ID"] == eid].iloc[0]
        ok &= r["모델"] == "CatBoost" and r["순위 참고"] == "결과 없음"
    check("(g) 스태킹 줄: 모델 '스태킹(로지스틱 메타)'·입력·계획서 대응·비고 주의 글·순위 제외(순위 참고 글), xs 줄: 결과 없음", ok)
    so_, sn_ = rd("old", "시드별.csv"), rd("new", "시드별.csv")
    sn_old = sn_[sn_["실험 ID"].isin(set(so_["실험 ID"]))].reset_index(drop=True)
    add_cols = [c for c in sn_.columns if c not in so_.columns]
    ok_seed = (list(sn_.columns[:len(so_.columns)]) == list(so_.columns) and so_.equals(sn_old[list(so_.columns)])
               and all((sn_old[c] == "").all() for c in add_cols))
    check("(g) 시드별 표: 기존 줄 같음(새 칸은 스태킹 줄 값만, 기존 줄 빈칸), 고위험품목비교 같음",
          ok_seed and rd("old", "고위험품목비교.csv").equals(rd("new", "고위험품목비교.csv")), f"시드별 {len(so_)}줄, 새 칸 {add_cols}")
    so = pd.read_excel(out / "old" / "실험기록.xlsx", sheet_name="설정", dtype=str).fillna("")
    sn = pd.read_excel(out / "new" / "실험기록.xlsx", sheet_name="설정", dtype=str).fillna("")
    old_pairs, new_pairs = list(zip(so["항목"], so["값"])), list(zip(sn["항목"], sn["값"]))
    oset = set(old_pairs)
    add_keys = [p[0] for p in new_pairs if p not in oset]
    check("(g) 설정 시트: 기존 줄 그대로(순서 같음) + 새 줄(XS 1 + STK 5)",
          [p for p in new_pairs if p in oset] == old_pairs and len(add_keys) == 6 and add_keys[0].startswith("XS")
          and all(k.startswith("STK") for k in add_keys[1:]), f"새 줄 {add_keys}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(E.EXP2_DIR))
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    log = Logger(WORK / "progress_selftest_stk.log")
    say(f"기본 모델 결과 읽는 곳(--root): {root.name}/results")
    miss = [E.run_dir_name(b, s) for b in BASES for s in SEEDS if not is_complete(root / "results" / E.run_dir_name(b, s))]
    check("기본 모델 결과 12개(E00·A3·DL_lstmA·DL_trfA x 시드 42·43·44) 완전", not miss, f"없음 {miss}")
    if miss:
        return
    t_config()
    ref = t_align(root)
    t_controls(root)
    from run import code_md5
    mem = Mem()
    mem.start()
    try:
        runA = t_twice(root, log, mem, code_md5)
    finally:
        mem.stop()
    froot = t_runpy(root, runA)
    t_meta_and_format(root, froot, ref)
    t_old_cfg_md5()
    t_tables(froot)
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
