# -*- coding: utf-8 -*-
"""P 외부 지표('eco'·'gw'·'eco_p'·'gw_p', P1·P2·P3·P1p·P2p) 자체 점검 (전체 자료). 결과: logs/selftest_p.txt (UTF-8). 실패가 있으면 종료 코드 1.
results/·tables/·cache/ 에는 쓰지 않는다(기상 요약 캐시는 메타가 같을 때만 읽기).

(0) 설정: 실험 5개 단계 'p'·시드 42·43·44, 입력 = A3 + 묶음, STAGES 에서 'p' 가 tables 바로 앞, 조합 ID 정규식과 겹치지 않음,
    칸 이름 5개, run_exp2.ps1 ValidateSet 에 'p'
(a) 무작위 (행, h) 300쌍 + 예측일 T 가 사용가능일 당일인 쌍: 외부 지표 parquet 원본을 따로 읽어 반복문으로
    '그 시군구 행 중 사용가능일 ≤ T−1, 그중 사용가능일 가장 늦은(같으면 자료연도 큰) 행 값' 을 구해 대조.
    대조군: 같은 쌍을 '사용가능일 ≤ T' 로 구하면 T = 사용가능일 인 쌍에서 달라짐. 한 쌍씩(h 정수) = 한꺼번에(h 배열)
(b) 위약판: 자료연도마다 값 모음(결측 포함)이 원본과 같고, 시군구 대응은 대부분 바뀜, 두 번 만들어도 같음, 사용가능일 같음,
    위약판 칸 = 섞은 표로 같은 반복문 규칙을 적용한 값
(c) P 실험 특성 행렬 = A3 특성 행렬(비트 단위) + 묶음 칸
(d) 기존 계산 그대로: 고치기 전 e2fg(scratch/p/e2fg_before_p.py.bak)와 F·G·M·Q 칸 값·범주·이름·설명이 같음,
    기존 결과(시드 42, ext 있는 CatBoost) config.json 입력 칸·설명 = 지금 코드
(e) 기록: 분할별 결측 비율, 학습 행 연도별 결측 비율
"""
import importlib.machinery
import importlib.util
import json
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
from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_p.txt"
SCR = E.EXP2_DIR / "scratch" / "p"
OLD_E2FG = SCR / "e2fg_before_p.py.bak"
PIDS = ["P1", "P2", "P3", "P1p", "P2p"]
PBLK = {"P1": ["eco"], "P2": ["gw"], "P3": ["eco", "gw"], "P1p": ["eco_p"], "P2p": ["gw_p"]}
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


def same(a, b):
    return (np.isnan(a) and np.isnan(b)) or (not np.isnan(a) and not np.isnan(b) and abs(a - b) <= 1e-6 * max(1.0, abs(b)))


def raw_table(base):
    path, col, _ = G.EXTIDX[base]
    t = pq.read_table(path).to_pandas()
    rows = {}
    for r in t.itertuples(index=False):
        d = getattr(r, "_asdict")()
        rows.setdefault(str(d["sgg_code"]), []).append((np.datetime64(pd.Timestamp(d["사용가능일"]).date(), "D"), int(d["자료연도"]),
                                                        float("nan") if pd.isna(d[col]) else float(d[col])))
    return rows


def ref_value(rows, sgg, T, incl_today=False):
    lim = T if incl_today else T - np.timedelta64(1, "D")
    ok = [r for r in rows.get(sgg, []) if r[0] <= lim]
    if not ok:
        return float("nan")
    ok.sort(key=lambda r: (r[0], r[1]))
    return ok[-1][2]


# ---------------------------------------------------------------- (0) 설정
def t_config():
    p = [e for e in E.EXPERIMENTS if e["stage"] == "p"]
    ids = [e["id"] for e in p]
    a3 = E.EXP_BY_ID["A3"]["feats"]
    check("(0) 실험 P1·P2·P3·P1p·P2p: 단계 'p', 시드 42·43·44(smoke 42)",
          ids == PIDS and all(E.exp_seeds(e, False) == [42, 43, 44] for e in p) and E.exp_seeds(p[0], True) == [42], f"{ids}")
    check("(0) 입력 = A3(k = k*) + 외부 지표 묶음", all(E.EXP_BY_ID[i]["feats"] == dict(a3, ext=PBLK[i]) for i in PIDS),
          "; ".join(f"{i} {E.EXP_BY_ID[i]['feats'].get('ext')}" for i in PIDS))
    st = E.STAGES
    check("(0) STAGES: 'p' 가 'tables' 바로 앞", "p" in st and st.index("p") == st.index("tables") - 1 and st[-1] == "tables", f"{st}")
    check("(0) 실험 ID 가 조합 ID 정규식과 겹치지 않음", not any(G.is_combo_id(x) for x in ids))
    nm = [G.ext_names([b])[0] for b in G.P_BLOCKS]
    check("(0) 칸 이름 eco_ratio·gw_pass_rate·eco_ratio_placebo·gw_pass_rate_placebo",
          nm == ["eco_ratio", "gw_pass_rate", "eco_ratio_placebo", "gw_pass_rate_placebo"] and G.ext_names(["eco", "gw"]) == nm[:2], f"{nm}")
    ps1 = (E.EXP2_DIR / "run_exp2.ps1").read_text(encoding="utf-8-sig")
    vs = [ln for ln in ps1.splitlines() if "ValidateSet" in ln]
    check("(0) run_exp2.ps1 ValidateSet 에 'p'(q 와 tables 사이)", len(vs) == 1 and "'q', 'p', 'tables'" in vs[0], vs[0].strip() if vs else "")
    check("(0) F·G 조합 규칙(BLOCK_ORDER)에 외부 지표 묶음 없음", not any(b in G.BLOCK_ORDER for b in G.P_BLOCKS))


# ---------------------------------------------------------------- (a) 원본 반복문 대조
def pick_pairs(ctx, rng, n_per=100):
    picks = []
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        for j in rng.choice(n, n_per, replace=False):
            picks.append((sp, int(j), int(rng.integers(1, E.H_MAX + 1))))
    return picks


def edge_pairs(ctx, rng, raws, per=20):
    """T = 어떤 사용가능일 인 (분할, 행, h) 쌍을 묶음마다 per 개 고름(목표일 d = 사용가능일 + h 인 행)."""
    out = []
    d0 = ctx.weather.date0
    for base, rows in raws.items():
        avail = sorted({r[0] for v in rows.values() for r in v})
        cand = []
        for sp in C1.SPLITS:
            dd = ctx.sp[sp]["d"]
            for a in avail:
                ai = int((a - d0).astype(np.int64))
                for h in (1, 7, 30, 60):
                    js = np.flatnonzero(dd == ai + h)
                    cand += [(sp, int(j), h) for j in js[:3]]
        if cand:
            sel = rng.choice(len(cand), min(per, len(cand)), replace=False)
            out += [cand[i] for i in sel]
    return out


def t_values(ctx, rng, raws):
    picks = pick_pairs(ctx, rng) + edge_pairs(ctx, rng, raws)
    blocks = ["eco", "gw"]
    by_sp = {}
    for sp, j, h in picks:
        by_sp.setdefault(sp, []).append((j, h))
    batch = {sp: G.ext_features(ctx, sp, np.array([j for j, _ in v]), np.array([h for _, h in v]), blocks) for sp, v in by_sp.items()}
    cnt = {sp: 0 for sp in by_sp}
    bad = {"eco": 0, "gw": 0}
    bad_batch = 0
    diff_today = {"eco": 0, "gw": 0}
    on_day = {"eco": 0, "gw": 0}
    nan_ref = {"eco": 0, "gw": 0}
    first_bad = None
    for sp, j, h in picks:
        row = ctx.rows.iloc[ctx.sp[sp]["idx"][j]]
        sgg = str(row["sgg_code"])
        T = np.datetime64(row["d"], "D") - np.timedelta64(h, "D")
        got = G.ext_features(ctx, sp, np.array([j]), h, blocks)[0]
        gb = batch[sp][cnt[sp]]
        cnt[sp] += 1
        bad_batch += int(not np.array_equal(got, gb, equal_nan=True))
        for k, base in enumerate(blocks):
            ref = ref_value(raws[base], sgg, T)
            nan_ref[base] += int(np.isnan(ref))
            if not same(float(got[k]), ref):
                bad[base] += 1
                first_bad = first_bad or (base, sp, j, h, sgg, str(T), float(got[k]), ref)
            if any(r[0] == T for r in raws[base].get(sgg, [])):
                on_day[base] += 1
                diff_today[base] += int(not same(float(got[k]), ref_value(raws[base], sgg, T, incl_today=True)))
    n = len(picks)
    for base in blocks:
        check(f"(a) {base} = 원본 반복문('사용가능일 ≤ T−1' 가장 최근 행)", bad[base] == 0, f"불일치 {bad[base]}/{n}, 참조 결측 {nan_ref[base]}")
        check(f"(a) 대조군 {base}: T = 사용가능일 인 쌍에서 '≤ T' 규칙 값과 다름(당일 공개 자료 안 씀)",
              on_day[base] > 0 and diff_today[base] > 0, f"T = 사용가능일 인 쌍 {on_day[base]}개 중 {diff_today[base]}개 다름")
    check("(a) 한 쌍씩(h 정수) = 한꺼번에(h 배열) 비트 단위", bad_batch == 0, f"불일치 {bad_batch}/{n}")
    if first_bad:
        say(f"    [첫 불일치] {first_bad}")
    return picks


# ---------------------------------------------------------------- (b) 위약판
def t_placebo(ctx, rng, picks):
    for base in ("eco", "gw"):
        t0 = G.extidx_table(base)
        t1 = G.extidx_table(base, placebo=True)
        t2 = G.extidx_table(base, placebo=True)
        ok_multi, changed, expect = True, [], []
        for y in sorted(t0["자료연도"].unique()):
            a = t0.loc[t0["자료연도"] == y, "v"].to_numpy()
            b = t1.loc[t1["자료연도"] == y, "v"].to_numpy()
            ok_multi &= bool(np.array_equal(np.sort(a), np.sort(b), equal_nan=True))
            changed.append(float(np.mean(~((a == b) | (np.isnan(a) & np.isnan(b))))))
            _, cnt = np.unique(np.where(np.isnan(a), -1.0, a), return_counts=True)      # 같은 값(결측은 한 값으로)끼리 묶음
            expect.append(1.0 - float(np.sum((cnt / len(a)) ** 2)))                      # 무작위로 섞을 때 값이 바뀌는 비율(근사)
        ok_keys = (t0[["sgg_code", "자료연도", "사용가능일"]].equals(t1[["sgg_code", "자료연도", "사용가능일"]]))
        check(f"(b) {base} 위약판: 자료연도마다 값 모음 같음·키와 사용가능일 같음·두 번 만들어도 같음",
              ok_multi and ok_keys and t1["v"].equals(t2["v"]), "")
        gap = [c - e for c, e in zip(changed, expect)]
        check(f"(b) {base} 위약판: 해마다 값이 바뀐 시군구 비율이 무작위 섞기 기대치(같은 값 수 감안)에서 10%p 안",
              min(gap) > -0.10, f"바뀐 비율 최소 {min(changed):.1%}·평균 {np.mean(changed):.1%}, 기대치 최소 {min(expect):.1%}·평균 {np.mean(expect):.1%}, "
                                f"차이 최소 {min(gap):+.1%}")
        # 위약판 칸 = 섞은 표에 같은 규칙
        rows = {}
        for r in t1.itertuples(index=False):
            rows.setdefault(r.sgg_code, []).append((np.datetime64(r.사용가능일, "D"), int(r.자료연도), float(r.v)))
        bad = 0
        for sp, j, h in picks[:300]:
            row = ctx.rows.iloc[ctx.sp[sp]["idx"][j]]
            T = np.datetime64(row["d"], "D") - np.timedelta64(h, "D")
            got = float(G.ext_features(ctx, sp, np.array([j]), h, [base + "_p"])[0, 0])
            bad += int(not same(got, ref_value(rows, str(row["sgg_code"]), T)))
        check(f"(b) {base}_p 칸 = 섞은 표에 '사용가능일 ≤ T−1' 규칙", bad == 0, f"불일치 {bad}/300")


# ---------------------------------------------------------------- (c) 특성 행렬
def t_matrix(ctx, rng):
    from e2models import FeatureBuilder
    k_star = json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]
    b0 = FeatureBuilder(ctx, dict(E.EXP_BY_ID["A3"]["feats"], k=k_star))
    for pid in PIDS:
        b1 = FeatureBuilder(ctx, dict(E.EXP_BY_ID[pid]["feats"], k=k_star))
        blk = PBLK[pid]
        ok_names = b1.names == b0.names + G.ext_names(blk) and b1.cat_names == b0.cat_names
        bad = []
        for sp in C1.SPLITS:
            pos = rng.choice(len(ctx.sp[sp]["idx"]), 500, replace=False)
            hh = rng.integers(1, E.H_MAX + 1, size=500)
            for h in (hh, int(hh[0])):
                X0, X1 = b0.num(sp, pos, h), b1.num(sp, pos, h)
                ext = G.ext_features(ctx, sp, pos, h, blk)
                if not (X1.shape[1] == X0.shape[1] + len(blk) and np.array_equal(X1[:, :X0.shape[1]], X0, equal_nan=True)
                        and np.array_equal(X1[:, X0.shape[1]:], ext, equal_nan=True)):
                    bad.append((sp, "정수" if np.isscalar(h) else "배열"))
        check(f"(c) {pid} 특성 행렬 = A3 행렬(비트 단위) + {blk}", ok_names and not bad,
              f"숫자 칸 {len(b1.names)} = A3 {len(b0.names)} + {len(blk)}, 범주 {b1.cat_names}, 불일치 {bad}")


# ---------------------------------------------------------------- (d) 기존 계산 그대로
def _load_old():
    ld = importlib.machinery.SourceFileLoader("e2fg_old_p", str(OLD_E2FG))
    spec = importlib.util.spec_from_loader("e2fg_old_p", ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


def t_old_same(ctx, rng):
    O = _load_old()
    old_blocks = list(O.EXT_BLOCKS)
    sets = [[b] for b in old_blocks] + [list(O.BLOCK_ORDER), ["spi", "soilmed", "food"], list(O.RG_BLOCKS), old_blocks]
    ok_txt = all(O.ext_names(s) == G.ext_names(s) and O.ext_desc(s) == G.ext_desc(s) and O.ext_count_desc(s) == G.ext_count_desc(s)
                 and O.ext_cat_names(s) == G.ext_cat_names(s) and O.ext_info(s) == G.ext_info(s) for s in sets)
    check("(d) 기존 묶음(F·G·M·Q) 이름·설명·칸 수 설명·범주·config 기록 = 고치기 전",
          ok_txt and O.BLOCK_ORDER == G.BLOCK_ORDER and O.FG_BLOCKS == G.FG_BLOCKS)
    check("(d) 설정 시트 M·Q 줄 = 고치기 전", O.wxanom_settings_rows() == G.wxanom_settings_rows() and O.rg_settings_rows() == G.rg_settings_rows())
    picks = pick_pairs(ctx, rng)
    bad = badc = 0
    for sp in C1.SPLITS:
        pp = [(j, h) for s_, j, h in picks if s_ == sp]
        pos = np.array([j for j, _ in pp])
        hh = np.array([h for _, h in pp])
        for h in (hh, int(hh[0])):
            old = O.ext_features(ctx, sp, pos, h, old_blocks)
            new = G.ext_features(ctx, sp, pos, h, old_blocks)
            bad += int(not np.array_equal(old, new, equal_nan=True))
        badc += int(not np.array_equal(O.ext_cats(ctx, sp, pos, old_blocks), G.ext_cats(ctx, sp, pos, old_blocks)))
    check("(d) F·G·M·Q 전체 묶음 칸 값(h 배열·정수) = 고치기 전 e2fg (300쌍, 비트 단위)", bad == 0, f"불일치 {bad}/6")
    check("(d) F3 범주 aw_label = 고치기 전", badc == 0)
    from e2models import CatRunner
    rows, nb = [], 0
    for d in sorted((E.EXP2_DIR / "results").glob("*__s42")):
        cf = d / "config.json"
        if not cf.exists():
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        if not (c.get("feats") or {}).get("ext") or c.get("model") != "cat":
            continue
        exp = E.EXP_BY_ID.get(c["exp"]) or dict(id=c["exp"], model="cat", feats=c["feats"])
        r = CatRunner(ctx, exp, 42, c["feats"], lambda *_: None, 1, 1)
        ok = r.cols == c["feature_columns"] and r.n_features_desc == c["n_features_desc"] and r.n_features == c["n_features"] \
            and G.ext_info(c["feats"]["ext"])["names"] == c["ext"]["names"]
        nb += int(not ok)
        rows.append(f"{c['exp']}{'' if ok else '(다름)'}")
    check("(d) 기존 결과(시드 42, ext 있는 CatBoost) config.json 입력 칸 이름·칸 수·설명 = 지금 코드", nb == 0 and len(rows) > 0,
          f"{len(rows)}개: {', '.join(rows)}")


# ---------------------------------------------------------------- (e) 기록
def t_record(ctx, rng):
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        pos = rng.choice(n, min(n, 20000), replace=False)
        hh = rng.integers(1, E.H_MAX + 1, size=len(pos))
        X = G.ext_features(ctx, sp, pos, hh, G.P_BLOCKS)
        r = np.isnan(X).mean(0)
        say(f"    [기록] {sp} (행 x 무작위 h) {len(pos):,}쌍 결측 비율: " + ", ".join(f"{b} {v:.2%}" for b, v in zip(G.P_BLOCKS, r)))
        if sp == "train":
            T = ctx.weather.date0 + (ctx.sp[sp]["d"][pos] - hh).astype("timedelta64[D]")
            yr = T.astype("datetime64[Y]").astype(int) + 1970
            say("    [기록] train 예측일 T 연도별 eco 결측 비율: " + ", ".join(f"{y} {np.isnan(X[yr == y, 0]).mean():.0%}" for y in sorted(set(yr))))


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    log = Logger(SCR / "progress_selftest_p.log")
    rng = np.random.default_rng(20261019)
    t_config()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    from selftest_fg import _cache_meta_ok
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        say("    기상 요약 캐시 메타가 다름 -> 캐시를 쓰지 않음")
    raws = {"eco": raw_table("eco"), "gw": raw_table("gw")}
    picks = t_values(ctx, rng, raws)
    t_placebo(ctx, rng, picks)
    t_matrix(ctx, rng)
    t_old_same(ctx, rng)
    t_record(ctx, rng)
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
