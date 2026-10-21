# -*- coding: utf-8 -*-
"""C4 다운사이징·N 제품명 변수 자체 점검(사용자 지시 2026-10-19, 전체 자료). 결과: logs/selftest_c4nb.txt (UTF-8). 하나라도 실패하면 종료 코드 1.
results/·tables/·cache/ 에는 쓰지 않는다(E00 재현 결과는 scratch/c4nb/selftest_e00/, 기상 요약 캐시는 메타가 같을 때만 읽기).

(0) 설정: C4_base·C4_hist(단계 c, 묶음 C4, 시드 42·43·44), A3_N1·A3_N2·A3_N1N2·A3_N0(단계 n, A3 입력 + 묶음, k = k*),
    STAGES 에서 'n' 이 tables 바로 앞, 조합 ID 정규식과 겹치지 않음, run_exp2.ps1 ValidateSet 에 'n', 칸 수 84·21·500, 이름 중복 없음
(a) C4 목록을 시료 parquet(시료_시군구.parquet)에서 반복문으로 직접 다시 세어 대조(코드·순서·학습 시료·검출 수),
    C4 밖 검증·시험 검출(단위x날짜 행·시료) 수, C4 행 수(분할별), C1~C3 = 기존 results/groups.json
(b) N1 을 무작위 (행, h) 300쌍(분할마다 100)에서 원본 parquet 두 개를 lims_data_key 로 맞춘 뒤 반복문으로 다시 계산해 대조
    (창 = 기준일 ∈ [T−P, T−1], p0_c 도 같은 창 전국). 한 쌍씩(h 정수) = 한꺼번에(h 배열). 단위 시료 수 n = A3 이력(e2hist) 단위 n.
    대조군: 창에 T 당일을 넣으면(기준일 ≤ T) 달라지는 쌍이 있음.
    앞날 자료: T 당일·이후 시료의 뭉치·검출을 무작위로 바꿔 다시 만들어도 84칸 비트 단위로 같음(대조군: T−30..T−1 을 바꾸면 달라짐)
(c) N2 를 단위x날짜 키별로 시료를 반복문으로 직접 모아(뭉치 OR) 전체 행 대조, 특성 행렬 칸 = 그 값(h 와 무관)
(d) N0: 단위날짜_tfidf.parquet 을 따로 읽어 키(train_prdlst_code 빈 값 = '코드없음')로 찾은 500칸 = 특성 칸(비트 단위, 무작위 3,000행 + 빈 2단계 44행 전부),
    n_rows 는 입력에 없음, n_rows = 그 행 시료 수, 칸 이름·순서 = parquet tf_* = 어휘 csv 열이름
(a2) A3_N* 특성 행렬 = A3 특성 행렬(비트 단위) + 묶음 칸 (h 배열·정수)
(e) E00 시드 42 를 끝까지 실행해 1차 catboost__summary__L050·2차 results/E00__s42 예측과 비트 단위로 같은지,
    고치기 전 FeatureBuilder(scratch/e2models_before_c4nb.py.bak)와 지금 FeatureBuilder 의 기존 실험(E00·A·B·D3·F·G·M·조합) 특성 행렬이 비트 단위로 같은지,
    기존 결과(시드 42) config.json 입력 칸 이름·칸 수·설명 = 지금 코드, e2fg.py·e2hist.py·e2eval.py 는 바뀌지 않음(md5)
"""
import collections
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import config as C1  # noqa: E402
import e2fg as G  # noqa: E402
import e2nb as NB  # noqa: E402
from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import Mem, read_preds  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_c4nb.txt"
SCR = E.EXP2_DIR / "scratch" / "c4nb"
OLD_E2MODELS = E.EXP2_DIR / "scratch" / "e2models_before_c4nb.py.bak"
NAN = float("nan")
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


def close(a, b, rtol, atol):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=rtol, atol=atol))


def k_star():
    return json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]


def feats_of(eid):
    f = dict(E.EXP_BY_ID[eid]["feats"]) if E.EXP_BY_ID[eid]["feats"] != "A*" else dict(E.EXP_BY_ID["A3"]["feats"])
    if f.get("k") == "k*":
        f["k"] = k_star()
    return f


# ---------------------------------------------------------------- 원본 읽기(코드와 따로)
def read_raw_samples():
    """시료_시군구.parquet 과 시료별_뭉치.parquet 을 lims_data_key 로 맞춤(행 순서에 기대지 않음)."""
    t = pq.read_table(C1.SAMPLE_PATH, columns=["lims_data_key", "sgg_code", "train_prdlst_code", "train_prdlst_code_2",
                                               "mnfctur_de", "judge_result"]).to_pandas()
    nb = pq.read_table(NB.NB_SAMPLE_PATH).to_pandas()
    m = t.merge(nb, on="lims_data_key", how="left", validate="one_to_one")
    if m[NB.cluster_cols()].isna().any().any():
        raise RuntimeError("뭉치 파일에 없는 시료")
    def code(x):
        return C1.NULL_CODE if x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA else str(x)
    df = {                                                    # 날짜는 numpy datetime64[D] 그대로(DataFrame 에 넣으면 Timestamp 로 바뀜)
        "sgg": m["sgg_code"].astype(str).to_numpy(dtype=object),
        "p2": np.array([code(x) for x in m["train_prdlst_code"].astype(object)], dtype=object),
        "p3": np.array([code(x) for x in m["train_prdlst_code_2"].astype(object)], dtype=object),
        "day": m["mnfctur_de"].to_numpy().astype("datetime64[D]"),
        "pos": (m["judge_result"].astype(str) == "검출").to_numpy().astype(np.int8),
    }
    F = m[NB.cluster_cols()].to_numpy().astype(np.int8)
    return df, F


# ---------------------------------------------------------------- (0) 설정
def t_config():
    c4 = [e for e in E.EXPERIMENTS if e.get("group") == "C4"]
    n = [e for e in E.EXPERIMENTS if e["stage"] == "n"]
    a3 = E.EXP_BY_ID["A3"]["feats"]
    ok_c4 = ([e["id"] for e in c4] == ["C4_base", "C4_hist"] and all(e["stage"] == "c" for e in c4)
             and c4[0]["feats"] == E.BASE_FEATS and c4[1]["feats"] == "A*" and all(E.exp_seeds(e, False) == [42, 43, 44] for e in c4))
    check("(0) C4_base(E00 입력)·C4_hist(A 최고 입력): 단계 c, 묶음 C4, 시드 42·43·44", ok_c4, f"{[(e['id'], e['stage'], e['plan']) for e in c4]}")
    ids = [e["id"] for e in n]
    want = {"A3_N1": ["n1"], "A3_N2": ["n2"], "A3_N1N2": ["n1", "n2"], "A3_N0": ["n0"]}
    ok_n = (ids == list(want) and all(e["feats"] == dict(a3, nb=want[e["id"]]) and e["feats"]["k"] == "k*" for e in n)
            and all(e["plan"] == "제품명 변수(교수님 계획서 4·5절)" for e in n) and all(E.exp_seeds(e, False) == [42, 43, 44] for e in n))
    check("(0) A3_N1·A3_N2·A3_N1N2·A3_N0: 단계 n, A3 입력(k = k*) + 묶음, 계획서 대응 '제품명 변수(교수님 계획서 4·5절)', 시드 42·43·44", ok_n, f"{ids}")
    st = E.STAGES
    check("(0) STAGES: 'n' 이 'tables' 바로 앞, 'm' 바로 뒤", st.index("n") == st.index("tables") - 1 and st.index("m") == st.index("n") - 1, f"{st}")
    check("(0) 실험 ID 가 조합 ID 정규식(^A3_(?:F[1-4]|G[1-3])+$)과 겹치지 않음", not any(G.is_combo_id(x) for x in ids + ["C4_base", "C4_hist"]))
    ps1 = (E.EXP2_DIR / "run_exp2.ps1").read_text(encoding="utf-8-sig")
    vs = [ln for ln in ps1.splitlines() if "ValidateSet" in ln]
    check("(0) run_exp2.ps1 ValidateSet 에 'n'(m 과 tables 사이)", len(vs) == 1 and "'m', 'n', 'tables'" in vs[0], vs[0].strip() if vs else "")
    n1, n2, n0 = NB.nb_names(["n1"]), NB.nb_names(["n2"]), NB.nb_names(["n0"])
    allnames = NB.nb_names(["n1", "n2", "n0"])
    from e2models import num_feature_names
    a3n = num_feature_names(dict(feats_of("A3_N1N2")))
    a30 = num_feature_names(dict(feats_of("A3_N0")))
    check("(0) 칸 수 N1 84 = 21 x 2 x 2, N2 21, N0 500, 이름 중복 없음(A3 칸 포함)",
          len(n1) == 84 and len(n2) == 21 and len(n0) == 500 and len(set(allnames)) == 605 and len(set(a3n)) == len(a3n) and len(set(a30)) == len(a30),
          f"{n1[:2]} … {n1[-1]}, {n2[0]}, {n0[0]} … {n0[-1]}")
    check("(0) n_rows 는 입력 칸에 없음", "n_rows" not in a30 and all(c.startswith("tf_") for c in n0))


# ---------------------------------------------------------------- (a) C4
def t_c4(ctx, raw):
    df, _ = raw
    tr_end = np.datetime64(C1.TRAIN_END)
    n_by, s_by = collections.Counter(), collections.Counter()
    for p3, day, pos in zip(df["p3"], df["day"], df["pos"]):
        if day <= tr_end:
            n_by[p3] += 1
            s_by[p3] += int(pos)
    ref = sorted([c for c in n_by if s_by[c] >= 1], key=lambda c: (-s_by[c], c))
    G4 = ctx.groups["C4"]
    check("(a) C4 코드 목록·순서(학습 검출 수 내림차순, 같으면 코드 순) = 시료 parquet 반복문 다시 셈", G4["codes"] == ref, f"{len(ref)}개")
    tr_n = sum(n_by[c] for c in ref)
    tr_s = sum(s_by[c] for c in ref)
    check("(a) C4 26개, 학습 시료 356,955건, 예상 목록과 같음", len(ref) == 26 and tr_n == 356955 == G4["train_samples"] and G4["matches_expected"]
          and G4["matches_expected_train_samples"], f"코드 {len(ref)}개, 학습 시료 {tr_n:,}, 학습 검출 시료 {tr_s:,} / 전체 학습 검출 {sum(s_by.values()):,}, 전체 학습 시료 {sum(n_by.values()):,}")
    tab_ok = all(t["n"] == n_by[t["code"]] and t["s"] == s_by[t["code"]] for t in G4["table"])
    check("(a) C4 표(코드별 검사 수·검출 수) = 다시 셈", tab_ok)
    say("    C4 목록: " + ", ".join(f"{c}(검사 {n_by[c]:,}, 검출 {s_by[c]})" for c in ref))
    # C4 밖 검증·시험 검출 (시료·단위x날짜 행) - 반복문
    c4 = set(ref)
    ranges = {"train": (None, tr_end), "val": (np.datetime64(C1.VAL_START), np.datetime64(C1.VAL_END)),
              "test": (np.datetime64(C1.TEST_START), np.datetime64(C1.TEST_END))}
    rows_all = {sp: set() for sp in ranges}
    rows_pos = {sp: set() for sp in ranges}
    smp = {sp: [0, 0, 0, 0] for sp in ranges}          # 시료, 밖 시료, 검출 시료, 밖 검출 시료
    for sgg, p2, p3, day, pos in zip(df["sgg"], df["p2"], df["p3"], df["day"], df["pos"]):
        for sp, (lo, hi) in ranges.items():
            if (lo is None or day >= lo) and day <= hi:
                key = (sgg, p2, p3, day)
                rows_all[sp].add(key)
                if pos:
                    rows_pos[sp].add(key)
                o = p3 not in c4
                smp[sp][0] += 1
                smp[sp][1] += int(o)
                smp[sp][2] += int(pos)
                smp[sp][3] += int(pos and o)
                break
    ok = True
    for sp in ("val", "test"):
        o = G4["outside"][sp]
        r_out = sum(1 for k in rows_all[sp] if k[2] not in c4)
        rp_out = sum(1 for k in rows_pos[sp] if k[2] not in c4)
        ref_o = {"rows": len(rows_all[sp]), "rows_outside": r_out, "pos_rows": len(rows_pos[sp]), "pos_rows_outside": rp_out,
                 "samples": smp[sp][0], "samples_outside": smp[sp][1], "pos_samples": smp[sp][2], "pos_samples_outside": smp[sp][3]}
        same = all(o[k] == v for k, v in ref_o.items())
        ok &= same
        say(f"    {sp}: C4 밖 검출 단위x날짜 행 {rp_out}/{len(rows_pos[sp])} ({rp_out / len(rows_pos[sp]):.2%}), 검출 시료 {smp[sp][3]}/{smp[sp][2]}, "
            f"C4 밖 행 {r_out:,}/{len(rows_all[sp]):,} ({r_out / len(rows_all[sp]):.1%}), C4 밖 시료 {smp[sp][1]:,}/{smp[sp][0]:,}")
    check("(a) C4 밖 검증·시험 검출 수(행·시료) = 다시 셈 (groups.json 'outside')", ok)
    # C4 행 수(분할별) = 전용 모델 학습·평가 행
    pos_ok = True
    for sp in C1.SPLITS:
        n_c4 = sum(1 for k in rows_all[sp] if k[2] in c4)
        got = len(ctx.positions(sp, "C4"))
        pos_ok &= got == n_c4
        say(f"    {sp}: C4 단위x날짜 행 {got:,} (다시 셈 {n_c4:,}), 그중 검출 {int(ctx.sp[sp]['y'][ctx.positions(sp, 'C4')].sum())}")
    check("(a) C4 전용 모델 행(분할별 ctx.positions) = 다시 센 C4 단위x날짜 행", pos_ok)
    old = json.loads((E.EXP2_DIR / "results" / "groups.json").read_text(encoding="utf-8"))

    def norm(o):                                          # run.py 가 groups.json 을 쓸 때와 같은 변환(e2tables.jdump)
        return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else (x.tolist() if hasattr(x, "tolist") else str(x))))
    same = all(norm(old[g]) == norm(ctx.groups[g]) for g in ("C1", "C2", "C3")) and old["_train_total"] == norm(ctx.groups["_train_total"])
    check("(a) C1~C3 묶음(코드·표·합계) = 기존 results/groups.json (바뀌지 않음)", same)


# ---------------------------------------------------------------- (b) N1
class RawIndex:
    """원본 시료(lims_data_key 로 맞춘 것)를 단위 키별 목록으로. 반복문용으로 날짜는 1970-01-01 기준 정수 일, 뭉치·검출은 파이썬 목록."""

    def __init__(self, raw):
        df, F = raw
        self.dayi = df["day"].astype(np.int64)            # datetime64[D] -> 정수 일
        self.pos = df["pos"].astype(np.int64)
        self.F = F.astype(np.int64)
        self.dayl = self.dayi.tolist()
        self.posl = self.pos.tolist()
        self.Fl = self.F.tolist()
        self.by_unit = collections.defaultdict(list)
        for i, k in enumerate(zip(df["sgg"], df["p2"], df["p3"])):
            self.by_unit[k].append(i)


def ref_n1(R, ukey, T, k, incl_T=False):
    """반복문 N1: 단위 시료(기준일 ∈ [T−P, T−1], incl_T 면 [T−P, T]) -> P 마다 [share 21, rate 21]. 전국 p0_c 도 같은 창."""
    Ti = int(np.datetime64(T, "D").astype(np.int64))
    out, ns = [], []
    C = R.F.shape[1]
    for P, _ in NB.NB_PERIODS:
        lo = Ti - P
        n = 0
        nc = [0] * C
        sc = [0] * C
        for i in R.by_unit[ukey]:
            d = R.dayl[i]
            if d >= lo and (d <= Ti if incl_T else d < Ti):
                n += 1
                fi = R.Fl[i]
                for c in range(C):
                    if fi[c]:
                        nc[c] += 1
                        sc[c] += R.posl[i]
        m = (R.dayi >= lo) & ((R.dayi <= Ti) if incl_T else (R.dayi < Ti))
        Nc = R.F[m].sum(0)
        Sc = (R.F[m] * R.pos[m][:, None]).sum(0)
        share = [nc[c] / n if n > 0 else NAN for c in range(C)]
        p0 = [Sc[c] / Nc[c] if Nc[c] > 0 else NAN for c in range(C)]
        rate = [(sc[c] + k * p0[c]) / (nc[c] + k) for c in range(C)]
        out += share + rate
        ns.append(n)
    return np.array(out, dtype=np.float64), ns[0]          # ns[0] = 5년 창 단위 시료 수


def pick_pairs(ctx, rng, n_per=100):
    picks = []
    for sp in C1.SPLITS:
        for j in rng.choice(len(ctx.sp[sp]["idx"]), n_per, replace=False):
            picks.append((sp, int(j), int(rng.integers(1, E.H_MAX + 1))))
    return picks


def t_n1(ctx, raw, R, rng):
    from e2models import FeatureBuilder
    k = float(k_star())
    f = feats_of("A3_N1")
    fb = FeatureBuilder(ctx, f)
    na3 = len(FeatureBuilder(ctx, feats_of("A3")).names)
    picks = pick_pairs(ctx, rng)
    by_sp = collections.defaultdict(list)
    for sp, j, h in picks:
        by_sp[sp].append((j, h))
    batch = {sp: fb.num(sp, np.array([j for j, _ in v]), np.array([h for _, h in v]))[:, na3:] for sp, v in by_sp.items()}
    cnt = collections.Counter()
    bad = bad_batch = bad_n = diff_inclT = 0
    n_pos5 = n_pos1 = n_nc = 0
    maxd = 0.0
    first_bad = None
    t0 = time.time()
    for sp, j, h in picks:
        row = ctx.rows.iloc[ctx.sp[sp]["idx"][j]]
        ukey = (str(row["sgg_code"]), str(row["train_prdlst_code"]), str(row["train_prdlst_code_2"]))
        T = np.datetime64(row["d"], "D") - np.timedelta64(h, "D")
        got = fb.num(sp, np.array([j]), h)[0, na3:]
        gb = batch[sp][cnt[sp]]
        cnt[sp] += 1
        bad_batch += int(not np.array_equal(got, gb, equal_nan=True))
        ref, n5 = ref_n1(R, ukey, T, k)
        n_pos5 += int(n5 > 0)
        n_pos1 += int(not np.isnan(ref[84 // 2]))          # 1년 share 첫 칸이 결측 아님 = 1년 창 단위 시료 있음
        n_nc += int(np.nansum(ref[:21]) > 0)
        if not close(got, ref, 1e-6, 1e-9):
            bad += 1
            first_bad = first_bad or (sp, j, h, [(a, b) for a, b in zip(got[:8], ref[:8])])
        mm = ~np.isnan(ref) & ~np.isnan(got)
        if mm.any():
            maxd = max(maxd, float(np.max(np.abs(got[mm] - ref[mm]))))
        # 단위 시료 수 n = A3 이력(e2hist) 단위 창 n
        g = int(ctx.sp[sp]["grp"]["unit"][j])
        t = int(ctx.sp[sp]["hd"][j]) - h
        for P, _ in NB.NB_PERIODS:
            nh, _ = ctx.H.counts("unit", np.array([g]), np.array([t]), P)
            n_nb = ctx._nb.hist.window(np.array([g]), np.array([t]), P)[0]
            bad_n += int(nh[0] != n_nb[0])
        ref_i, _ = ref_n1(R, ukey, T, k, incl_T=True)
        diff_inclT += int(not close(got, ref_i, 1e-6, 1e-9))
    n = len(picks)
    say(f"    반복문 대조 {time.time() - t0:.1f}s, 5년 창 단위 시료 있는 쌍 {n_pos5}/{n}, 1년 창 있는 쌍 {n_pos1}/{n}, 뭉치 시료가 하나라도 있는 쌍(5년) {n_nc}/{n}")
    check("(b) N1 84칸 = 원본 parquet 반복문 계산(창 기준일 ∈ [T−P, T−1], p0_c 같은 창 전국, k = k*) (float32 반올림 허용 rtol 1e-6)", bad == 0,
          f"불일치 {bad}/{n} 쌍, 최대 절대 차이 {maxd:.2e}")
    check("(b) 한 쌍씩(h 정수) = 한꺼번에(h 배열, 학습 경로) 비트 단위", bad_batch == 0, f"불일치 {bad_batch}/{n}")
    check("(b) 단위 창 시료 수 n = A3 이력(e2hist) 단위 n (같은 창·같은 자르기)", bad_n == 0, f"불일치 {bad_n}/{2 * n}")
    check("(b) 대조군: 창에 T 당일을 넣으면(기준일 ≤ T, p0_c 포함) 달라지는 쌍이 있음", diff_inclT > 0, f"{diff_inclT}/{n} 쌍이 달라짐")
    if first_bad:
        say(f"    [첫 불일치] {first_bad}")
    # 앞날 자료: T 당일·이후 시료의 뭉치·검출을 바꿔도 같음
    X = ctx._nb
    nbad = nctrl = n_elig = 0
    tests = picks[::10]
    for sp, j, h in tests:
        g = np.array([int(ctx.sp[sp]["grp"]["unit"][j])])
        t = int(ctx.sp[sp]["hd"][j]) - h
        base = X.hist.features(g, np.array([t]), k)
        fut = X.s_day >= t
        Fm = X.flags.copy()
        Pm = ctx.samples["pos"].to_numpy().copy()
        Fm[fut] = rng.integers(0, 2, size=(int(fut.sum()), Fm.shape[1]), dtype=np.int8)
        Pm[fut] = rng.integers(0, 2, size=int(fut.sum()), dtype=np.int8)
        Hm = NB.NbHist(X.s_unit, X.s_day, ctx.H.Dh, Fm, Pm)
        nbad += int(not np.array_equal(Hm.features(g, np.array([t]), k), base, equal_nan=True))
        past = (X.s_day >= t - 30) & (X.s_day < t)
        if not past.any():                                # 그 30일에 시료가 하나도 없으면 대조군 대상 아님
            continue
        n_elig += 1
        Fc = X.flags.copy()
        Pc = ctx.samples["pos"].to_numpy().copy()
        Fc[past] = 1 - Fc[past]
        Pc[past] = 1 - Pc[past]
        Hc = NB.NbHist(X.s_unit, X.s_day, ctx.H.Dh, Fc, Pc)
        nctrl += int(not np.array_equal(Hc.features(g, np.array([t]), k), base, equal_nan=True))
    check("(b) T 당일·이후 시료(전국)의 뭉치·검출을 무작위로 바꿔 다시 만들어도 N1 84칸 비트 단위로 같음", nbad == 0, f"불일치 {nbad}/{len(tests)}")
    check("(b) 대조군: T−30..T−1 시료의 뭉치·검출을 뒤집으면 달라짐", nctrl == n_elig > 0, f"{nctrl}/{n_elig} 쌍이 달라짐(그 30일에 시료 있는 쌍)")


# ---------------------------------------------------------------- (c) N2
def t_n2(ctx, raw, rng):
    from e2models import FeatureBuilder
    df, F = raw
    t0 = time.time()
    acc = {}
    for i, key in enumerate(zip(df["sgg"], df["p2"], df["p3"], df["day"])):
        a = acc.get(key)
        acc[key] = F[i].copy() if a is None else np.maximum(a, F[i])
    rf = ctx.rows_full
    ref = np.stack([acc[(s, p2, p3, d)] for s, p2, p3, d in zip(rf["sgg_code"].astype(str), rf["train_prdlst_code"].astype(str),
                                                               rf["train_prdlst_code_2"].astype(str), rf["d"].values.astype("datetime64[D]"))])
    X = ctx._nb
    check("(c) N2 전체 단위x날짜 행 = 시료를 키별로 반복문으로 모은 뭉치 OR (모든 행·21칸 같음)", len(acc) == len(rf) and np.array_equal(ref, X.N2),
          f"키 {len(acc):,} / 행 {len(rf):,}, 뭉치 하나라도 있는 행 {int(ref.any(1).sum()):,}, {time.time() - t0:.1f}s")
    fb = FeatureBuilder(ctx, feats_of("A3_N2"))
    na3 = len(FeatureBuilder(ctx, feats_of("A3")).names)
    bad = 0
    keep_idx = np.flatnonzero(ctx._keep)
    for sp in C1.SPLITS:
        pos = rng.choice(len(ctx.sp[sp]["idx"]), 2000, replace=False)
        fr = keep_idx[ctx.sp[sp]["idx"][pos]]
        for h in (1, 60, rng.integers(1, 61, size=len(pos))):
            got = fb.num(sp, pos, h)[:, na3:]
            bad += int(not np.array_equal(got, ref[fr].astype(np.float32)))
    check("(c) A3_N2 특성 행렬의 N2 칸 = 그 행의 반복문 값 (h=1·h=60·h 배열 모두 같음 = h 와 무관)", bad == 0, f"불일치 {bad}/9")


# ---------------------------------------------------------------- (d) N0
def t_n0(ctx, rng):
    from e2models import FeatureBuilder
    pf = pq.ParquetFile(NB.TFIDF_PATH)
    names = pf.schema_arrow.names
    tcols = [c for c in names if c.startswith("tf_")]
    voc = pd.read_csv(NB.TFIDF_VOCAB_PATH, encoding="utf-8-sig")
    fb = FeatureBuilder(ctx, feats_of("A3_N0"))
    na3 = len(FeatureBuilder(ctx, feats_of("A3")).names)
    check("(d) N0 칸 이름·순서 = parquet tf_* = 어휘 csv 열이름(500), n_rows 는 입력 칸에 없음",
          fb.names[na3:] == tcols == voc["열이름"].astype(str).tolist() and len(tcols) == 500 and "n_rows" not in fb.names)
    keys = pq.read_table(NB.TFIDF_PATH, columns=C1.KEY_COLS + ["d", "n_rows"]).to_pandas()
    p2f = [C1.NULL_CODE if x is None or x is pd.NA else str(x) for x in keys["train_prdlst_code"].astype(object)]
    p3f = [C1.NULL_CODE if x is None or x is pd.NA else str(x) for x in keys["train_prdlst_code_2"].astype(object)]
    dd = pd.to_datetime(keys["d"]).to_numpy().astype("datetime64[D]")
    null_idx = np.flatnonzero(keys["train_prdlst_code"].isna().to_numpy())
    rf = ctx.rows_full
    rmap = {(s, p2, p3, d): i for i, (s, p2, p3, d) in enumerate(zip(rf["sgg_code"].astype(str), rf["train_prdlst_code"].astype(str),
                                                                     rf["train_prdlst_code_2"].astype(str), rf["d"].values.astype("datetime64[D]")))}
    tf_row = np.array([rmap.get((s, p2, p3, d), -1) for s, p2, p3, d in zip(keys["sgg_code"].astype(str), p2f, p3f, dd)])
    check("(d) tfidf 키(빈 2단계 = '코드없음') 291,812행이 단위x날짜 행과 1:1", (tf_row >= 0).all() and len(np.unique(tf_row)) == len(rf) == len(keys),
          f"못 찾은 키 {int((tf_row < 0).sum())}, 빈 2단계 {len(null_idx)}행")
    nr_ok = bool(np.array_equal(keys["n_rows"].to_numpy(), rf["n_samples"].to_numpy()[tf_row]))
    check("(d) n_rows = 그 단위x날짜 행 시료 수(입력에는 넣지 않음)", nr_ok)
    sel = np.unique(np.concatenate([rng.choice(len(keys), 3000, replace=False), null_idx]))
    tb = pq.read_table(NB.TFIDF_PATH, columns=tcols).take(sel).to_pandas()
    vals = tb[tcols].to_numpy(dtype=np.float32)
    X = ctx._nb
    ok_arr = np.array_equal(X.N0[tf_row[sel]], vals)
    check("(d) N0 배열 = 원본 parquet 값(비트 단위, 무작위 3,000행 + 빈 2단계 44행)", ok_arr, f"{len(sel)}행")
    # FeatureBuilder 경로(분할 행 -> 전체 행)
    keep_idx = np.flatnonzero(ctx._keep)
    inv = np.full(len(rf), -1)
    inv[tf_row[sel]] = np.arange(len(sel))
    bad = tested = 0
    for sp in C1.SPLITS:
        fr = keep_idx[ctx.sp[sp]["idx"]]
        pos = np.flatnonzero(inv[fr] >= 0)
        for h in (1, 60, rng.integers(1, 61, size=len(pos))):
            got = fb.num(sp, pos, h)[:, na3:]
            bad += int(not np.array_equal(got, vals[inv[fr[pos]]]))
        tested += len(pos)
    check("(d) A3_N0 특성 행렬의 N0 칸 = 원본 값 (h=1·h=60·h 배열, 분할 행 → 전체 행 대응)", bad == 0, f"불일치 {bad}/9, 대조 행 {tested}")


# ---------------------------------------------------------------- (a2) A3_N* = A3 + 묶음
def t_matrix(ctx, rng):
    from e2models import FeatureBuilder
    b0 = FeatureBuilder(ctx, feats_of("A3"))
    bad = []
    for eid in ("A3_N1", "A3_N2", "A3_N1N2", "A3_N0"):
        f = feats_of(eid)
        b1 = FeatureBuilder(ctx, f)
        if b1.names != b0.names + NB.nb_names(f["nb"]) or b1.cat_names != b0.cat_names:
            bad.append((eid, "이름"))
        for sp in C1.SPLITS:
            pos = rng.choice(len(ctx.sp[sp]["idx"]), 500, replace=False)
            hh = rng.integers(1, E.H_MAX + 1, size=500)
            for h in (hh, int(hh[0])):
                X0, X1 = b0.num(sp, pos, h), b1.num(sp, pos, h)
                nb = NB.nb_features(ctx, sp, pos, h, f["nb"], f["k"])
                if not (np.array_equal(X1[:, :X0.shape[1]], X0, equal_nan=True) and np.array_equal(X1[:, X0.shape[1]:], nb, equal_nan=True)):
                    bad.append((eid, sp, "배열" if not np.isscalar(h) else "정수"))
                if not np.array_equal(b1.cats(sp, pos), b0.cats(sp, pos)):
                    bad.append((eid, sp, "범주"))
        say(f"    {eid}: 숫자 칸 {len(b1.names)} = A3 {len(b0.names)} + {len(b1.names) - len(b0.names)}, 범주 {b1.cat_names}")
    check("(a2) A3_N* 특성 행렬 = A3 특성 행렬(비트 단위) + 묶음 칸, 범주 같음 (h 배열·정수)", not bad, f"불일치 {bad}")


# ---------------------------------------------------------------- (e) 기존 실험 그대로
def _load_old_models():
    ld = importlib.machinery.SourceFileLoader("e2models_old_c4nb", str(OLD_E2MODELS))
    spec = importlib.util.spec_from_loader("e2models_old_c4nb", ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


def t_old_same(ctx, rng):
    import e2models as MN
    MO = _load_old_models()
    ids = ["E00", "A1", "A2", "A3", "B1", "B2", "D3", "F1", "F2", "F3", "F4", "F2b", "G1", "G2", "G3", "A3_F4b", "M1", "A3_M1"]
    combos = [("A3_F4", dict(feats_of("A3"), ext=G.blocks_of(["F4"]))), ("A3_G1G2G3", dict(feats_of("A3"), ext=G.blocks_of(["G1", "G2", "G3"])))]
    bad = []
    for eid, f in [(i, feats_of(i)) for i in ids] + combos:
        bo, bn = MO.FeatureBuilder(ctx, f), MN.FeatureBuilder(ctx, f)
        if bo.names != bn.names or bo.cat_names != bn.cat_names:
            bad.append((eid, "이름"))
            continue
        for sp in C1.SPLITS:
            pos = rng.choice(len(ctx.sp[sp]["idx"]), 300, replace=False)
            hh = rng.integers(1, E.H_MAX + 1, size=300)
            for h in (hh, int(hh[0])):
                if not np.array_equal(bo.num(sp, pos, h), bn.num(sp, pos, h), equal_nan=True):
                    bad.append((eid, sp, "숫자"))
            co, cn = bo.cats(sp, pos), bn.cats(sp, pos)
            if not ((co is None and cn is None) or np.array_equal(co, cn)):
                bad.append((eid, sp, "범주"))
    check("(e) 기존 실험(E00·A1~A3·B1·B2·D3·F1~F4·F2b·G1~G3·A3_F4b·M1·A3_M1·A3_F4·A3_G1G2G3) 특성 행렬 = 고치기 전 FeatureBuilder (비트 단위, h 배열·정수, 범주)",
          not bad, f"불일치 {bad}")
    rows, nb = [], 0
    for d in sorted((E.EXP2_DIR / "results").glob("*__s42")):
        cf = d / "config.json"
        if not cf.exists():
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        if c.get("model") not in ("cat", "lr"):
            continue
        exp = E.EXP_BY_ID.get(c["exp"]) or dict(id=c["exp"], model=c["model"], feats=c["feats"])
        if c["model"] == "cat":
            r = MN.CatRunner(ctx, exp, 42, c["feats"], lambda *_: None, 1, 1)
        else:
            r = MN.LRRunner(ctx, exp, 42, c["feats"], lambda *_: None)
        ok = r.cols == c["feature_columns"] and r.n_features_desc == c["n_features_desc"] and r.n_features == c["n_features"]
        if c.get("group"):
            ok &= c["group_codes"] == ctx.groups[c["group"]]["codes"] and all(
                c["rows"][sp]["n"] == len(r.pos[sp]) for sp in C1.SPLITS)
        nb += int(not ok)
        rows.append(f"{c['exp']}{'' if ok else '(다름)'}")
    check("(e) 기존 결과(시드 42, CatBoost·로지스틱) config.json 입력 칸 이름·칸 수·설명(C 는 품목 코드·행 수까지) = 지금 코드", nb == 0 and len(rows) > 0,
          f"{len(rows)}개: {', '.join(rows)}")
    for fn in ("e2fg.py", "e2hist.py", "e2eval.py"):
        a = hashlib.md5((E.EXP2_DIR / "src" / fn).read_bytes()).hexdigest()
        b = hashlib.md5((E.EXP2_DIR / "scratch" / fn.replace(".py", "_before_c4nb.py.bak")).read_bytes()).hexdigest()
        check(f"(e) {fn} 바뀌지 않음(고치기 전 사본과 md5 같음)", a == b, a)


def t_e00(ctx, log):
    import shutil
    from run import check_windows, run_one
    from selftest_fg import _cache_meta_ok
    root = SCR / "selftest_e00"
    P = {"results": root / "results", "tables": root / "tables", "logs": root / "logs", "cache": E.EXP2_DIR / "cache",
         "progress": root / "logs" / "progress.log"}
    if (P["results"] / "E00__s42").exists():
        shutil.rmtree(P["results"] / "E00__s42")
    P["results"].mkdir(parents=True, exist_ok=True)
    mem = Mem()
    mem.start()
    win = check_windows(ctx, log)
    t0 = time.time()
    s = run_one(E.EXP_BY_ID["E00"], 42, ctx, P, log, mem, False, win)
    mem.stop()
    d1 = E.EXP1_RESULTS / "catboost__summary__L050"
    d2 = P["results"] / "E00__s42"
    d3 = E.EXP2_DIR / "results" / "E00__s42"
    for sp in ("val", "test"):
        p2, y2, i2 = read_preds(d2 / f"pred_{sp}.parquet")
        for lab, dref in (("1차 catboost__summary__L050", d1), ("2차 results/E00__s42", d3)):
            p1, y1, i1 = read_preds(dref / f"pred_{sp}.parquet")
            same_rows = i1.astype(str).equals(i2.astype(str)) and np.array_equal(y1, y2)
            check(f"(e) E00 시드 42 {sp} 예측 = {lab} (행·y·p 비트 단위)", same_rows and np.array_equal(p1, p2),
                  f"{p1.shape}, 최대 차이 {float(np.max(np.abs(p1.astype(np.float64) - p2))):.1e}")
    m1 = pd.read_csv(d1 / "metrics_by_h.csv")
    m2 = pd.read_csv(d2 / "metrics_by_h.csv")
    cols = ["pr_auc", "roc_auc", "threshold", "tp", "fn", "fp", "tn", "precision", "recall", "f1", "f2"]
    check("(e) E00 시드 42 h 별 지표 = 1차", np.allclose(m1[cols].to_numpy(float), m2[cols].to_numpy(float), rtol=0, atol=1e-12))
    s1 = json.loads((d1 / "summary.json").read_text(encoding="utf-8"))
    check("(e) E00 시드 42 최적 라운드·학습 라운드 = 1차", s["best_rounds_or_epoch"] == s1["best_rounds_or_epoch"]
          and s["n_rounds_trained"] == s1["n_rounds_trained"],
          f"{s['best_rounds_or_epoch']}/{s['n_rounds_trained']} vs {s1['best_rounds_or_epoch']}/{s1['n_rounds_trained']}, 총 {time.time() - t0:.0f}s")
    c2 = json.loads((d2 / "config.json").read_text(encoding="utf-8"))
    c3 = json.loads((d3 / "config.json").read_text(encoding="utf-8"))
    check("(e) E00 config: 입력 칸 이름·개수·설명 = 기존 results/E00__s42, 제품명 기록 칸 없음",
          c2["feature_columns"] == c3["feature_columns"] and c2["n_features"] == c3["n_features"]
          and c2["n_features_desc"] == c3["n_features_desc"] and "nb" not in c2 and "ext" not in c2, f"{c2['n_features_desc']}")


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    log = Logger(SCR / "progress_selftest_c4nb.log")
    rng = np.random.default_rng(20261019)
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    from selftest_fg import _cache_meta_ok
    if not _cache_meta_ok(ctx):                              # 캐시 파일을 새로 쓰지 않도록(메타가 다르면 그 자리에서 계산)
        ctx.use_cache = False
        say("    기상 요약 캐시 메타가 다름 -> 캐시를 쓰지 않음")
    else:
        say("    기상 요약 캐시(cache/eval_wsum_L050_*.npy) 메타 같음 -> 읽기만 함")
    t_config()
    t1 = time.time()
    raw = read_raw_samples()
    say(f"    원본 시료 {len(raw[0]['day']):,}행 + 뭉치 21칸을 lims_data_key 로 맞춤 ({time.time() - t1:.1f}s)")
    t_c4(ctx, raw)
    NB.get_nb(ctx, ["n1", "n2", "n0"])
    R = RawIndex(raw)
    t_n1(ctx, raw, R, rng)
    del R
    t_n2(ctx, raw, rng)
    t_n0(ctx, rng)
    t_matrix(ctx, rng)
    t_old_same(ctx, rng)
    if "--no-e00" not in sys.argv:
        ctx._nb = ctx._wxanom = ctx._fg_ext = None          # E00 재현 전 제품명·F·G·M 자료 놓기(메모리)
        import gc
        gc.collect()
        t_e00(ctx, log)
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
