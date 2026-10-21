# -*- coding: utf-8 -*-
"""P4 HACCP('haccp'·'haccp_p', 실험 P4·P4p) 자체 점검 (전체 자료). 결과: logs/selftest_p4.txt (UTF-8). 실패가 있으면 종료 코드 1.
results/·tables/·cache/ 에는 쓰지 않는다.

(0) 설정: P4·P4p 단계 'p'·시드 3, 입력 = A3 + 묶음, 칸 이름
(a) 무작위 (행, h) 300쌍 + 사용시작일 앞뒤 쌍: 구간표 parquet 원본을 따로 읽어 반복문으로
    '그 단위 2단계 코드의 업종묶음에서 시작일 ≤ T−1 ≤ 끝일 인 그 시군구 행 수 ÷ 전국 행 수' 를 구해 대조(T < 사용시작일·묶음 없음 = 결측).
    대조군: T 당일까지 센 값은 달라지는 쌍이 있음. 한 쌍씩(h 정수) = 한꺼번에(h 배열)
(b) 위약판: 연도마다 위약 격자 = 원 격자의 시군구 순서를 섞은 것(같은 해·모든 묶음 같은 섞기), 해마다 값 모음 같음, 위약 칸 = 섞은 격자 값
(c) P4·P4p 특성 행렬 = A3 특성 행렬(비트 단위) + 1칸
(d) 기존 계산 그대로: 고치기 전 e2fg(scratch/p4/e2fg_before_p4.py.bak)와 기존 묶음(F·G·M·Q·P) 칸 값·이름·설명·config 기록 같음,
    기존 결과(시드 42, ext 있는 CatBoost) config.json 입력 칸·설명 = 지금 코드
(e) 기록: 분할별 결측 비율(전체·검출 행), 학습 예측일 연도별 결측
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
OUT = E.EXP2_DIR / "logs" / "selftest_p4.txt"
SCR = E.EXP2_DIR / "scratch" / "p4"
OLD_E2FG = SCR / "e2fg_before_p4.py.bak"
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


class Raw:
    """원본 구간표(반복문 대조용)."""

    def __init__(self):
        t = pq.read_table(G.HACCP_PATH).to_pandas()
        self.use = np.datetime64(pd.Timestamp(t["사용시작일"].iloc[0]).date(), "D")
        self.code2b = {}
        for b, g in t.groupby("업종묶음"):
            if b == "대응 없음":
                continue
            for L in g["우리_2단계_코드목록"].dropna():
                for c in str(L).split(","):
                    if c.strip() and c.strip() != "None":
                        self.code2b[c.strip()] = b
        self.b = t["업종묶음"].to_numpy()
        self.sgg = t["sgg_code"].astype(str).to_numpy()
        self.st = pd.to_datetime(t["시작일"]).values.astype("datetime64[D]")
        self.en = pd.to_datetime(t["끝일"]).values.astype("datetime64[D]")

    def value(self, sgg, p2, T, lag=1):
        b = self.code2b.get(p2)
        if b is None or T < self.use:
            return float("nan")
        day = T - np.timedelta64(lag, "D")
        ok = (self.b == b) & (self.st <= day) & (day <= self.en)
        n = int(ok.sum())
        return float("nan") if n == 0 else int((ok & (self.sgg == sgg)).sum()) / n


def pick_pairs(ctx, rng, n_per=100):
    out = []
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        for j in rng.choice(n, n_per, replace=False):
            out.append((sp, int(j), int(rng.integers(1, E.H_MAX + 1))))
    return out


def t_config():
    a3 = E.EXP_BY_ID["A3"]["feats"]
    ok = (E.EXP_BY_ID["P4"]["stage"] == "p" and E.EXP_BY_ID["P4p"]["stage"] == "p"
          and E.EXP_BY_ID["P4"]["feats"] == dict(a3, ext=["haccp"]) and E.EXP_BY_ID["P4p"]["feats"] == dict(a3, ext=["haccp_p"])
          and E.exp_seeds(E.EXP_BY_ID["P4"], False) == [42, 43, 44])
    check("(0) P4·P4p: 단계 'p', 입력 = A3 + haccp / haccp_p, 시드 42·43·44", ok)
    check("(0) 칸 이름 haccp_ratio·haccp_ratio_placebo", G.ext_names(["haccp"]) == ["haccp_ratio"] and G.ext_names(["haccp_p"]) == ["haccp_ratio_placebo"])


def t_values(ctx, rng, raw):
    X = G.get_haccp(ctx)
    picks = pick_pairs(ctx, rng)
    u0 = int((raw.use - ctx.weather.date0).astype(np.int64))
    for sp in ("train",):                                        # 사용시작일 앞뒤(T = 사용시작일 −1, 0, +1)
        dd = ctx.sp[sp]["d"]
        for off in (-1, 0, 1):
            for h in (1, 30):
                js = np.flatnonzero(dd == u0 + off + h)[:5]
                picks += [(sp, int(j), h) for j in js]
    p2all = ctx.rows["train_prdlst_code"].astype(str).to_numpy(dtype=object)
    by = {}
    for sp, j, h in picks:
        by.setdefault(sp, []).append((j, h))
    batch = {sp: G.ext_features(ctx, sp, np.array([j for j, _ in v]), np.array([h for _, h in v]), ["haccp"]) for sp, v in by.items()}
    cnt = {sp: 0 for sp in by}
    bad = bad_batch = diff_today = nn = 0
    first = None
    for sp, j, h in picks:
        row = ctx.rows.iloc[ctx.sp[sp]["idx"][j]]
        T = np.datetime64(row["d"], "D") - np.timedelta64(h, "D")
        sgg, p2 = str(row["sgg_code"]), str(p2all[ctx.sp[sp]["idx"][j]])
        got = float(G.ext_features(ctx, sp, np.array([j]), h, ["haccp"])[0, 0])
        bad_batch += int(not np.array_equal(np.float32(got), batch[sp][cnt[sp]][0], equal_nan=True))
        cnt[sp] += 1
        ref = raw.value(sgg, p2, T)
        nn += int(not np.isnan(ref))
        if not same(got, ref):
            bad += 1
            first = first or (sp, j, h, sgg, p2, str(T), got, ref)
        diff_today += int(not same(got, raw.value(sgg, p2, T, lag=0)))
    n = len(picks)
    check("(a) haccp = 원본 반복문(T−1 에 유효한 그 시군구 수 ÷ 전국 수, 사용시작일 전·묶음 없음 = 결측)", bad == 0,
          f"불일치 {bad}/{n}, 값 있는 쌍 {nn}")
    check("(a) 대조군: T 당일까지 센 값과 다른 쌍이 있음", diff_today > 0, f"{diff_today}/{n}")
    check("(a) 한 쌍씩 = 한꺼번에(비트 단위)", bad_batch == 0, f"불일치 {bad_batch}/{n}")
    if first:
        say(f"    [첫 불일치] {first}")
    say(f"    업종묶음 {X.bundles}, 사용시작일 {X.use_start}, 묶음별 {X.stats}")
    return picks


def t_placebo(ctx, rng, picks):
    X = G.get_haccp(ctx)
    wx = ctx.weather
    r2 = np.random.default_rng(G.EXTIDX_PLACEBO_SEED)
    years = (wx.date0 + np.arange(wx.D).astype("timedelta64[D]")).astype("datetime64[Y]").astype(int) + 1970
    perms = {y: r2.permutation(wx.S) for y in sorted(set(years.tolist()))}
    ok_perm, ok_multi = True, True
    for bi in range(len(X.R)):
        for y, pm in perms.items():
            cols = np.flatnonzero(years == y)
            ok_perm &= bool(np.array_equal(X.Rp[bi][:, cols], X.R[bi][pm][:, cols], equal_nan=True))
            ok_multi &= bool(np.array_equal(np.sort(X.Rp[bi][:, cols], axis=0), np.sort(X.R[bi][:, cols], axis=0), equal_nan=True))
    check("(b) 위약 격자 = 원 격자의 시군구 순서를 연도마다 섞은 것(모든 묶음 같은 섞기), 날마다 값 모음 같음", ok_perm and ok_multi)
    bad = 0
    for sp, j, h in picks[:300]:
        s = int(ctx.sp[sp]["s"][j])
        t = int(ctx.sp[sp]["d"][j]) - h
        bi = int(X.ub[sp][j])
        ref = float("nan") if bi < 0 else float(X.Rp[bi][s, t])
        got = float(G.ext_features(ctx, sp, np.array([j]), h, ["haccp_p"])[0, 0])
        bad += int(not same(got, ref))
    check("(b) haccp_p 칸 = 위약 격자 값", bad == 0, f"불일치 {bad}/300")


def t_matrix(ctx, rng):
    from e2models import FeatureBuilder
    k_star = json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]
    b0 = FeatureBuilder(ctx, dict(E.EXP_BY_ID["A3"]["feats"], k=k_star))
    for pid, blk in (("P4", ["haccp"]), ("P4p", ["haccp_p"])):
        b1 = FeatureBuilder(ctx, dict(E.EXP_BY_ID[pid]["feats"], k=k_star))
        bad = []
        for sp in C1.SPLITS:
            pos = rng.choice(len(ctx.sp[sp]["idx"]), 500, replace=False)
            hh = rng.integers(1, E.H_MAX + 1, size=500)
            for h in (hh, int(hh[0])):
                X0, X1 = b0.num(sp, pos, h), b1.num(sp, pos, h)
                if not (X1.shape[1] == X0.shape[1] + 1 and np.array_equal(X1[:, :-1], X0, equal_nan=True)
                        and np.array_equal(X1[:, -1:], G.ext_features(ctx, sp, pos, h, blk), equal_nan=True)):
                    bad.append((sp, "정수" if np.isscalar(h) else "배열"))
        check(f"(c) {pid} 특성 행렬 = A3 행렬(비트 단위) + {blk}", b1.names == b0.names + G.ext_names(blk) and not bad, f"불일치 {bad}")


def _load_old():
    ld = importlib.machinery.SourceFileLoader("e2fg_old_p4", str(OLD_E2FG))
    spec = importlib.util.spec_from_loader("e2fg_old_p4", ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


def t_old_same(ctx, rng):
    O = _load_old()
    old_blocks = list(O.EXT_BLOCKS)
    sets = [[b] for b in old_blocks] + [list(O.BLOCK_ORDER), ["eco", "gw"], list(O.RG_BLOCKS), old_blocks]
    ok_txt = all(O.ext_names(s) == G.ext_names(s) and O.ext_desc(s) == G.ext_desc(s) and O.ext_count_desc(s) == G.ext_count_desc(s)
                 and O.ext_cat_names(s) == G.ext_cat_names(s) and O.ext_info(s) == G.ext_info(s) for s in sets)
    check("(d) 기존 묶음(F·G·M·Q·P) 이름·설명·칸 수 설명·범주·config 기록 = 고치기 전", ok_txt)
    picks = pick_pairs(ctx, rng)
    bad = 0
    for sp in C1.SPLITS:
        pp = [(j, h) for s_, j, h in picks if s_ == sp]
        pos = np.array([j for j, _ in pp])
        hh = np.array([h for _, h in pp])
        for h in (hh, int(hh[0])):
            bad += int(not np.array_equal(O.ext_features(ctx, sp, pos, h, old_blocks), G.ext_features(ctx, sp, pos, h, old_blocks), equal_nan=True))
    check("(d) 기존 전체 묶음 칸 값(h 배열·정수) = 고치기 전 e2fg (300쌍, 비트 단위)", bad == 0, f"불일치 {bad}/6")
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
    check("(d) 기존 결과(시드 42, ext 있는 CatBoost) config.json 입력 칸·설명 = 지금 코드", nb == 0 and len(rows) > 0, f"{len(rows)}개: {', '.join(rows)}")


def t_record(ctx, rng):
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        hh = rng.integers(1, E.H_MAX + 1, size=n)
        x = G.ext_features(ctx, sp, np.arange(n), hh, ["haccp"])[:, 0]
        y = ctx.sp[sp]["y"].astype(bool)
        say(f"    [기록] {sp} 행 {n:,}(무작위 h): 결측 {np.isnan(x).mean():.2%}, 검출 행 결측 {np.isnan(x[y]).mean():.2%}")
        if sp == "train":
            T = ctx.weather.date0 + (ctx.sp[sp]["d"] - hh).astype("timedelta64[D]")
            yr = T.astype("datetime64[Y]").astype(int) + 1970
            say("    [기록] train 예측일 연도별 결측: " + ", ".join(f"{yy} {np.isnan(x[yr == yy]).mean():.0%}" for yy in sorted(set(yr))))


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    log = Logger(SCR / "progress_selftest_p4.log")
    rng = np.random.default_rng(20261019)
    t_config()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    from selftest_fg import _cache_meta_ok
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        say("    기상 요약 캐시 메타가 다름 -> 캐시를 쓰지 않음")
    raw = Raw()
    picks = t_values(ctx, rng, raw)
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
