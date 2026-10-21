# -*- coding: utf-8 -*-
"""XS 작은 실험 3개(E00_R1·E00_NW·A3_NW, 단계 'xs', 사용자 지시 2026-10-19) 자체 점검 (전체 자료). 결과: logs/selftest_xs.txt (UTF-8).
실패가 있으면 종료 코드 1. results/·tables/·cache/ 에는 쓰지 않는다(k* 는 <root>/results/selection_k.json 읽기만, 기상 요약 캐시는 쓰지도 읽지도 않음).
--root 폴더: 결과를 읽을 곳(results/selection_k.json, 기존 결과 config.json). 기본 = 이 실험 폴더. 서버에서는 --root ~/cyj_listeria/root.

(0) 설정: 실험 3개 단계 'xs'·모델 cat·시드 42·43·44, 입력(E00_R1 = E00 + rate1·k*, E00_NW = E00 에서 weather False, A3_NW = A3 에서 weather False),
    계획서 대응 칸, STAGES 에서 'xs'·'stk' 가 'ae' 뒤·'tables'(끝) 앞, run_exp2.ps1 ValidateSet 에 'xs'·'stk', 조합 ID 와 겹치지 않음
(a) E00_R1 특성 행렬 = E00 행렬(비트 단위) + rate1 3칸, 그 3칸 = A3 행렬의 rate1 칸(비트 단위), 칸 이름 = E00 + hist_rate1y_<p3|sgg|unit>,
    범주 = E00 범주 (분할마다 무작위 2,000행 x h 배열·정수, 학습 행 전체 x 시드 42 h 배정)
(b) E00_NW·A3_NW 특성 행렬 = E00·A3 행렬에서 기상 요약 190칸만 뺀 것(비트 단위), 칸 이름 = 같은 칸을 뺀 것, 범주 같음 (같은 행·h)
(c) 칸 수·설명: CatRunner n_features·n_features_desc(198·5·14), 표 '입력' 글(e2tables.input_desc)
(d) 기존 실험 그대로: 고치기 전 e2cfg(scratch/stk/e2cfg_before_stk.py.bak) 실험 목록·설정 값 = 지금(새 실험 5개·STAGES 의 xs·stk·STK 설정만 더해짐),
    고치기 전에 잰 md5(scratch/stk/md5_before_stk.json) 와 계산 모듈(e2models·e2hist·e2data·e2eval·e2fg·e2nb·e2ae·e2dl*·e2smote·1차 전부) md5 같음
    (-> 기존 실험 특성 행렬 계산 코드가 그대로), 기존 결과(<root>/results 의 CatBoost 결과 config.json) 입력·칸 이름·칸 수·설명 = 지금 코드로 만든 값
"""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402

import config as C1  # noqa: E402
import features as F  # noqa: E402
from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2hist import hist_names  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_xs.txt"
SCR = E.EXP2_DIR / "scratch" / "stk"
OLD_E2CFG = SCR / "e2cfg_before_stk.py.bak"
MD5_BEFORE = SCR / "md5_before_stk.json"
XS_IDS = ["E00_R1", "E00_NW", "A3_NW"]
NEW_IDS = XS_IDS + ["STK_E00", "STK_A3"]
CHANGED = {"e2cfg.py", "run.py", "e2tables.py", "selftest_hx.py", "run_exp2.ps1"}   # 이번에 고친 파일(나머지는 md5 같아야 함)
NEW_FILES = {"e2stk.py", "selftest_xs.py", "selftest_stk.py"}
CHECK_IDS = ["E00", "A3"] + [f"A1_k{_k}" for _k in E.K_GRID]                     # (d) config 대조 대상 기존 결과
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


def load_old(name, path):
    ld = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


def expected_feats():
    e00, a3 = E.EXP_BY_ID["E00"]["feats"], E.EXP_BY_ID["A3"]["feats"]
    return {"E00_R1": dict(e00, hist=["rate1"], k="k*"), "E00_NW": dict(e00, weather=False), "A3_NW": dict(a3, weather=False)}


def t_config():
    want = expected_feats()
    plans = {"E00_R1": "계획서 11절 3순위(기준 + 최근 1년 검출률)", "E00_NW": "계획서 6절 실험 A(정형 데이터만)",
             "A3_NW": "계획서 6절 실험 A(정형 데이터만)"}
    ok = all(E.EXP_BY_ID[i]["stage"] == "xs" and E.EXP_BY_ID[i]["model"] == "cat" and E.EXP_BY_ID[i]["feats"] == want[i]
             and E.EXP_BY_ID[i]["plan"] == plans[i] and E.exp_seeds(E.EXP_BY_ID[i], False) == [42, 43, 44]
             and not any(E.EXP_BY_ID[i].get(x) for x in ("group", "spw", "smote", "multi_h")) for i in XS_IDS)
    ids = [e["id"] for e in E.EXPERIMENTS if e["stage"] == "xs"]
    check("(0) E00_R1·E00_NW·A3_NW: 단계 'xs', CatBoost, 입력(E00 + rate1 k* / E00 기상 없음 / A3 기상 없음), 계획서 대응 칸, 시드 42·43·44",
          ok and ids == XS_IDS, f"{ids} | {[E.EXP_BY_ID[i]['feats'] for i in XS_IDS]}")
    st = E.STAGES
    check("(0) STAGES: 'xs'·'stk' 가 'ae' 뒤, 'tables'(끝) 앞, 'stk' 가 'xs' 뒤",
          st[-1] == "tables" and st.index("ae") < st.index("xs") < st.index("stk") < st.index("tables"), str(st[-6:]))
    ps1 = (E.EXP2_DIR / "run_exp2.ps1").read_text(encoding="utf-8-sig")
    vs = [ln for ln in ps1.splitlines() if "ValidateSet" in ln]
    check("(0) run_exp2.ps1 ValidateSet 에 'xs'·'stk'(ae 와 tables 사이)", len(vs) == 1 and "'ae', 'xs', 'stk', 'tables'" in vs[0],
          vs[0].strip() if vs else "")
    import e2fg
    check("(0) 조합 ID 정규식과 겹치지 않음", not any(e2fg.is_combo_id(i) for i in NEW_IDS))


def resolved(eid, k):
    f = dict(E.EXP_BY_ID[eid]["feats"])
    if f.get("k") == "k*":
        f["k"] = k
    return f


def t_r1(ctx, k, rng):
    from e2models import FeatureBuilder
    b0 = FeatureBuilder(ctx, resolved("E00", k), seed=42)
    b1 = FeatureBuilder(ctx, resolved("E00_R1", k), seed=42)
    b3 = FeatureBuilder(ctx, resolved("A3", k), seed=42)
    r1n = hist_names(["rate1"])
    j3 = [b3.names.index(n) for n in r1n]
    check("(a) E00_R1 칸 이름 = E00 칸 + hist_rate1y_<p3|sgg|unit>, 범주 칸 = E00",
          b1.names == b0.names + r1n and b1.cat_names == b0.cat_names and r1n == [f"hist_rate1y_{lv}" for lv in E.HIST_LEVELS],
          f"칸 {len(b1.names)} (E00 {len(b0.names)})")
    bad, tot = [], 0
    cases = []
    for sp in C1.SPLITS:
        pos = np.sort(rng.choice(len(ctx.sp[sp]["idx"]), 2000, replace=False))
        hh = rng.integers(1, E.H_MAX + 1, size=len(pos))
        cases += [(sp, pos, hh), (sp, pos, int(hh[0]))]
    trp = np.arange(len(ctx.sp["train"]["idx"]))
    cases.append(("train", trp, ctx.h_fit(42)["train"]))
    for sp, pos, h in cases:
        tot += 1
        X0, X1, X3 = b0.num(sp, pos, h), b1.num(sp, pos, h), b3.num(sp, pos, h)
        ok = (X1.shape[1] == X0.shape[1] + 3 and np.array_equal(X1[:, :X0.shape[1]], X0, equal_nan=True)
              and np.array_equal(X1[:, X0.shape[1]:], X3[:, j3], equal_nan=True)
              and np.array_equal(b1.cats(sp, pos), b0.cats(sp, pos)))
        if not ok:
            bad.append((sp, len(pos), "정수" if np.isscalar(h) else "배열"))
        if sp == "train" and len(pos) == len(trp):
            say(f"    [기록] 학습 행 전체 {len(pos):,}행 rate1 결측 비율 {np.isnan(X1[:, -3:]).mean(0).round(4).tolist()}")
    check("(a) E00_R1 행렬 = E00 행렬(비트 단위) + rate1 3칸, rate1 3칸 = A3 행렬의 rate1 칸(비트 단위), 범주 = E00 범주", not bad,
          f"경우 {tot}개(분할마다 2,000행 x h 배열·정수 + 학습 행 전체 x 시드 42 h), 다름 {bad}")


def t_nw(ctx, k, rng):
    from e2models import FeatureBuilder
    wn = set(F.feature_names("summary", E.L))
    for base, nw in (("E00", "E00_NW"), ("A3", "A3_NW")):
        b0 = FeatureBuilder(ctx, resolved(base, k), seed=42)
        b1 = FeatureBuilder(ctx, resolved(nw, k), seed=42)
        keep = [j for j, n in enumerate(b0.names) if n not in wn]
        okn = (b1.names == [b0.names[j] for j in keep] and len(b0.names) - len(keep) == 190 and b1.cat_names == b0.cat_names)
        bad, tot = [], 0
        cases = []
        for sp in C1.SPLITS:
            pos = np.sort(rng.choice(len(ctx.sp[sp]["idx"]), 2000, replace=False))
            hh = rng.integers(1, E.H_MAX + 1, size=len(pos))
            cases += [(sp, pos, hh), (sp, pos, int(hh[0]))]
        cases.append(("train", np.arange(len(ctx.sp["train"]["idx"])), ctx.h_fit(42)["train"]))
        for sp, pos, h in cases:
            tot += 1
            X0, X1 = b0.num(sp, pos, h), b1.num(sp, pos, h)
            if not (X1.shape[1] == len(keep) and np.array_equal(X1, X0[:, keep], equal_nan=True)
                    and np.array_equal(b1.cats(sp, pos), b0.cats(sp, pos))):
                bad.append((sp, len(pos), "정수" if np.isscalar(h) else "배열"))
        check(f"(b) {nw} 행렬 = {base} 행렬에서 기상 요약 190칸만 뺀 것(비트 단위), 칸 이름·범주 같음", okn and not bad,
              f"칸 {b1.names}, 경우 {tot}개, 다름 {bad}")


def t_desc(ctx, k):
    from e2models import CatRunner
    from e2tables import input_desc
    want = {"E00_R1": (198, "범주 4 + 숫자 194 (h 1 + 기상 190 + 이력 3 + WOE 0)",
                       f"범주 4(sido·sgg·p2·p3) + h + 기상 요약 50일(190) + 이력[1년 스무딩 검출률 3] (k={k})"),
            "E00_NW": (5, "범주 4 + 숫자 1 (h 1 + 기상 0 + 이력 0 + WOE 0)", "범주 4(sido·sgg·p2·p3) + h"),
            "A3_NW": (14, "범주 4 + 숫자 10 (h 1 + 기상 0 + 이력 9 + WOE 0)",
                      f"범주 4(sido·sgg·p2·p3) + h + 이력[5년 스무딩 검출률 3, 5년 log(1+검사 수) 3, 1년 스무딩 검출률 3] (k={k})")}
    for eid, (n, d, inp) in want.items():
        f = resolved(eid, k)
        r = CatRunner(ctx, E.EXP_BY_ID[eid], 42, f, lambda *_: None, 1, 1)
        got = (r.n_features, r.n_features_desc, input_desc(f, "cat"))
        check(f"(c) {eid} 칸 수·설명·표 입력 글", got == (n, d, inp), f"{got}")


def t_old_cfg():
    O = load_old("e2cfg_old_stk", OLD_E2CFG)
    old_ids = [e["id"] for e in O.EXPERIMENTS]
    cur = [e for e in E.EXPERIMENTS if e["id"] in set(old_ids)]
    added = [e["id"] for e in E.EXPERIMENTS if e["id"] not in set(old_ids)]
    check("(d) 실험 목록: 고치기 전 실험(설정 dict 전부) = 지금 같은 실험, 순서 같음, 더해진 것 = xs 3 + stk 2",
          [e["id"] for e in cur] == old_ids and all(a == b for a, b in zip(O.EXPERIMENTS, cur)) and added == NEW_IDS,
          f"고치기 전 {len(old_ids)}개, 더해진 것 {added}")
    skip = {"EXP2_DIR", "EXP1_DIR", "EXP1_SRC", "EXP1_RESULTS", "EXPERIMENTS", "EXP_BY_ID", "STAGES"}
    names = [n for n in dir(O) if n.isupper() and not n.startswith("_") and n not in skip]
    diff = [n for n in names if getattr(O, n) != getattr(E, n)]
    new_names = sorted(n for n in dir(E) if n.isupper() and not n.startswith("_") and n not in skip and not hasattr(O, n))
    st_ok = [s for s in E.STAGES if s not in ("xs", "stk")] == O.STAGES
    check("(d) 설정 값: 고치기 전 대문자 설정 전부 같음, STAGES = 고치기 전 + xs·stk, 새 설정 = STK·STK_BASE_DESC·STK_PLAN",
          not diff and st_ok and new_names == ["STK", "STK_BASE_DESC", "STK_PLAN"], f"다름 {diff}, 새 설정 {new_names}")


def t_md5():
    before = json.loads(MD5_BEFORE.read_text(encoding="utf-8"))
    src2 = Path(__file__).resolve().parent
    src1 = E.EXP1_SRC
    bad, same = [], 0
    for k, v in before.items():
        if k in CHANGED:
            continue
        p = (src1 / k.split("/", 1)[1]) if k.startswith("1차/") else src2 / k
        cur = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
        if cur != v:
            bad.append(k)
        else:
            same += 1
    calc = [k for k in before if k.startswith(("e2models", "e2hist", "e2data", "e2eval", "e2fg", "e2nb", "e2ae", "e2dl", "e2smote", "1차/"))]
    extra = sorted(f.name for f in src2.glob("*.py") if f.name not in before and f.name not in NEW_FILES)
    check("(d) 고치지 않은 파일 md5 = 고치기 전(계산 모듈 e2models·e2hist·e2data·e2eval·e2fg·e2nb·e2ae·e2dl*·e2smote·1차 포함)",
          not bad and not extra and all(k not in bad for k in calc), f"같음 {same}개(계산 모듈 {len(calc)}개), 다름 {bad}, 목록에 없던 새 파일 {extra}")


def t_existing_cfg(ctx, root):
    from e2models import CatRunner
    rows, nb = [], 0
    for d in sorted((root / "results").glob("*__s4[234]")):
        cf = d / "config.json"
        if not cf.exists():
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        if c.get("model") != "cat" or c["exp"] not in E.EXP_BY_ID:
            continue
        exp = E.EXP_BY_ID[c["exp"]]
        if c["exp"] not in CHECK_IDS:                          # 서버로 복사한 기존 결과(E00·A3·A1_k*)만
            continue
        r = CatRunner(ctx, exp, c["seed"], c["feats"], lambda *_: None, 1, 1)
        f = dict(exp["feats"]) if isinstance(exp["feats"], dict) else None
        ok = (r.cols == c["feature_columns"] and r.n_features_desc == c["n_features_desc"] and r.n_features == c["n_features"]
              and c["desc"] == exp["desc"] and c["plan"] == exp["plan"]
              and f is not None and all(c["feats"].get(kk) == vv for kk, vv in f.items() if kk != "k"))
        nb += int(not ok)
        rows.append(d.name + ("" if ok else "(다름)"))
    check("(d) 기존 결과(<root>/results CatBoost) config.json 입력·칸 이름·칸 수·설명·계획서 대응 = 지금 코드", nb == 0 and len(rows) > 0,
          f"{len(rows)}개 {rows}, 다름 {nb}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(E.EXP2_DIR))
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    log = Logger(SCR / "progress_selftest_xs.log")
    rng = np.random.default_rng(20261019)
    say(f"결과 읽는 곳(--root): {root.name}/results")
    t_config()
    t_old_cfg()
    t_md5()
    k = json.loads((root / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]
    say(f"    k* = {k} (selection_k.json)")
    ctx = Ctx2(False, log, use_cache=False, cache_dir=None)
    t_r1(ctx, k, rng)
    t_nw(ctx, k, rng)
    t_desc(ctx, k)
    t_existing_cfg(ctx, root)
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
