# -*- coding: utf-8 -*-
"""딥러닝(DL) 자체 점검. 결과: logs/selftest_dl.txt (UTF-8), 빈칸 통계 logs/dl_maskstats.json. 실패가 있으면 종료 코드 1.
results/ 에는 쓰지 않는다(예측은 메모리에서만 비교).

(1) 방식 B 정답 벡터·마스크: 단순 반복문(단위x날짜 행 사전, 날짜 직접 더하기)으로 다시 만들어 대조
    - 무작위 학습 표본 + 학습 끝(2024-06-30) 가까운 표본(검증 기간 칸이 생기는 표본)
    - 검증 기간 칸은 모두 마스크인지(그 날짜에 실제 단위x날짜 행이 있어도 마스크인지 포함), j = h 칸 = 그 행 y
(2) 방식 B 평가가 방식 A 와 같은 (행, h) 를 채점하는지
    - predict(sp, h) 를 '창을 numpy 로 직접 자르고(T-50..T-1, T = d-h) 한 번에 넣은 값' 과 대조 (A: 출력, B: 출력[h-1])
    - 조기 종료용 검증(행마다 다른 h)도 같은 방식으로 대조, 대상 행·y 가 A·B 같음
    - smoke 저장 예측(DL 4종 + E00)의 (sgg, p2, p3, d, h, y) 열이 모두 같음
(3) 1차와 같은 계산: 입력(38채널) = 1차 LSTMRunner.Wg, DL_lstmA(시드 42, L=100) = 1차 lstm__seq__L100 저장 예측·에폭별 AP,
    DL_lstmA(L=50) = 1차 LSTMRunner(L=50) 같은 프로세스 실행
(4) 재현: smoke 4종을 다시 학습해 저장된 smoke 예측과 같은지(프로세스가 달라도 같은지)
(5) 빈칸 비율·표본당 관측 칸 수: 전체 학습 표본, 시드 42·43·44
"""
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import config as C1  # noqa: E402
import models as MD1  # noqa: E402
from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import read_preds  # noqa: E402
import e2dl  # noqa: E402,F401
from e2dl_impl import DLRunner, build_seq, label_matrix, mask_stats, train_end_idx  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_dl.txt"
lines, fails = [], []
ARCH = {"DL_lstmA": ("lstm", "A"), "DL_trfA": ("trf", "A"), "DL_lstmB": ("lstm", "B"), "DL_trfB": ("trf", "B")}


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


def nolog(_msg):
    pass


# ---------------------------------------------------------------- (1)
def t_labels(ctx, rng):
    r = ctx.rows_full
    lut = dict(zip(zip(r["sgg_code"].astype(str), r["train_prdlst_code"].astype(str),
                       r["train_prdlst_code_2"].astype(str), r["d"].to_numpy().astype("datetime64[D]")),
                   r["y"].to_numpy().astype(int)))
    end = np.datetime64(C1.TRAIN_END)
    end_idx = train_end_idx(ctx)
    check("(1) 학습 끝 날짜 번호 = 2024-06-30", ctx.weather.date0 + np.timedelta64(end_idx, "D") == end)
    tr = ctx.sp["train"]
    rows_tr = ctx.rows.iloc[tr["idx"]]
    keys = np.stack([rows_tr[c].astype(str).to_numpy() for c in C1.KEY_COLS], axis=1)
    dd = rows_tr["d"].to_numpy().astype("datetime64[D]")
    n = len(dd)
    for seed in (42, 43):
        h = ctx.h_fit(seed)["train"]
        Y = label_matrix(ctx, "train", np.arange(n), h, end_idx)
        near = np.flatnonzero(dd >= end - np.timedelta64(59, "D"))
        pick = np.unique(np.concatenate([rng.choice(n, 3000, replace=False), rng.choice(near, min(1500, len(near)), replace=False)]))
        bad = 0
        n_cells = n_after = n_after_row = n_obs = n_pos = 0
        for i in pick:
            T = dd[i] - np.timedelta64(int(h[i]), "D")
            for j in range(1, 61):
                day = T + np.timedelta64(j, "D")
                k = (keys[i, 0], keys[i, 1], keys[i, 2], day)
                if day > end:
                    exp = -1
                    n_after += 1
                    if k in lut:
                        n_after_row += 1          # 검증 기간에 실제 단위x날짜 행이 있지만 마스크해야 하는 칸
                else:
                    exp = lut.get(k, -1)
                n_cells += 1
                n_obs += exp >= 0
                n_pos += exp == 1
                if Y[i, j - 1] != exp:
                    bad += 1
        check(f"(1) 시드 {seed}: 정답 벡터·마스크 = 반복문 재계산", bad == 0,
              f"(표본 {len(pick):,}개 x 60칸 = {n_cells:,}칸, 불일치 {bad}, 관측 {n_obs:,}·검출 {n_pos:,}칸, "
              f"학습 끝 이후 칸 {n_after:,})")
        check(f"(1) 시드 {seed}: 검증 기간 칸은 실제 행이 있어도 마스크", n_after_row > 0 and bad == 0,
              f"(그런 칸 {n_after_row:,}개가 모두 -1)")
        T_all = tr["d"].astype(np.int64) - h.astype(np.int64)
        after = (T_all[:, None] + np.arange(1, 61)[None, :]) > end_idx
        check(f"(1) 시드 {seed}: 전체 학습 표본에서 학습 끝 이후 칸 전부 마스크",
              bool((Y[after] == -1).all()), f"(그런 칸 {int(after.sum()):,}개)")
        check(f"(1) 시드 {seed}: 전체 학습 표본 j=h 칸 = 그 행 y (늘 관측)",
              bool(np.array_equal(Y[np.arange(n), h.astype(np.int64) - 1], tr["y"])))


# ---------------------------------------------------------------- (2)
def manual_prob(runner, seq_np, sp, rows, hvec):
    """창을 numpy 로 직접 잘라 한 번에 넣은 확률. A: sigmoid(출력), B: sigmoid(출력[h-1])."""
    v = runner.ctx.sp[sp]
    L = runner.L
    X = np.stack([seq_np[v["s"][i], v["d"][i] - hh - L: v["d"][i] - hh] for i, hh in zip(rows, hvec)])
    x = torch.from_numpy(X).to(runner.dev)
    c = torch.from_numpy(v["lstm_cats"][rows]).to(runner.dev)
    hh = torch.from_numpy(np.asarray(hvec, dtype=np.int64)).to(runner.dev)
    runner.net.eval()
    with torch.no_grad():
        if runner.mode == "A":
            o = runner.net(x, c, hh.float() / E.H_MAX)
            return torch.sigmoid(o).cpu().numpy(), None
        o = runner.net(x, c)
        right = torch.sigmoid(o[torch.arange(len(rows)), hh - 1]).cpu().numpy()
        wrong = torch.sigmoid(o[torch.arange(len(rows)), torch.clamp(hh, max=E.H_MAX - 1)]).cpu().numpy()  # 한 칸 밀린 값(점검력 확인용)
        return right, wrong


def t_alignment(ctx, rng):
    seq_np, _, _ = build_seq(ctx)
    runners = {}
    for eid, (a, m) in ARCH.items():
        rr = DLRunner(ctx, {"id": eid}, 42, nolog, False, a, m)
        rr.build_net()
        runners[eid] = rr
        for sp in ("val", "test"):
            rows = np.sort(rng.choice(len(ctx.sp[sp]["idx"]), 400, replace=False))
            worst, worst_wrong = 0.0, np.inf
            for h in (1, 2, 17, 30, 59, 60):
                p_run = rr.predict(sp, h)[rows]
                p_man, p_wrong = manual_prob(rr, seq_np, sp, rows, np.full(len(rows), h))
                worst = max(worst, float(np.max(np.abs(p_run - p_man))))
                if p_wrong is not None and h < 60:
                    worst_wrong = min(worst_wrong, float(np.max(np.abs(p_run - p_wrong))))
            check(f"(2) {eid} {sp}: predict(sp, h) = 직접 자른 창(T=d-h, T-50..T-1)" + (" 의 출력[h-1]" if m == "B" else ""),
                  worst < 1e-5, f"(행 400 x h 6개, 최대 차이 {worst:.2e}" +
                  (f"; 한 칸 밀린 출력[h] 와는 최소 최대차 {worst_wrong:.2e} 로 구별됨)" if m == "B" else ")"))
        rows = np.sort(rng.choice(len(ctx.sp["val"]["idx"]), 400, replace=False))
        hv = ctx.h_fit(42)["val"]
        p_es = rr._predict_tensor("val", rr.T["val"]["hf"])[rows]
        p_man, _ = manual_prob(rr, seq_np, "val", rows, hv[rows].astype(np.int64))
        check(f"(2) {eid}: 조기 종료용 검증(행마다 h = default_rng(43)) = 직접 계산", float(np.max(np.abs(p_es - p_man))) < 1e-5,
              f"(최대 차이 {float(np.max(np.abs(p_es - p_man))):.2e})")
    pa, pb = runners["DL_lstmA"], runners["DL_lstmB"]
    check("(2) A·B 대상 행 같음(pos = 전체 행 순서 그대로)",
          all(np.array_equal(pa.pos[sp], pb.pos[sp]) and np.array_equal(pa.pos[sp], np.arange(len(ctx.sp[sp]["idx"])))
              for sp in C1.SPLITS))
    for rr in runners.values():
        rr.close()
    del runners
    torch.cuda.empty_cache()
    # smoke 저장 예측의 행·h·y 열 비교
    sm = E.paths(True)["results"]
    ids = ["E00"] + list(ARCH)
    ok_files = all((sm / E.run_dir_name(i, 42) / "pred_val.parquet").exists() for i in ids)
    if not ok_files:
        check("(2) smoke 저장 예측 열 비교", False, "(smoke 결과 없음 - run.py --stage dl --smoke 먼저)")
        return
    for sp in ("val", "test"):
        ref = pd.read_parquet(sm / E.run_dir_name("E00", 42) / f"pred_{sp}.parquet").drop(columns=["p"])
        same = []
        for i in ARCH:
            t = pd.read_parquet(sm / E.run_dir_name(i, 42) / f"pred_{sp}.parquet").drop(columns=["p"])
            same.append(t.astype(str).equals(ref.astype(str)))
        check(f"(2) smoke {sp}: DL 4종 저장 예측의 (sgg, p2, p3, d, h, y) = E00", all(same),
              f"({len(ref):,}줄 = 60 x {len(ref) // 60:,}행, 같음 {same})")


# ---------------------------------------------------------------- (3)
class Shim:
    """1차 LSTMRunner 가 쓰는 ctx 모양(sp[sp]['h_fit'])."""

    def __init__(self, ctx, seed=42):
        self.weather = ctx.weather
        self.lstm_cat_sizes = ctx.lstm_cat_sizes
        hf = ctx.h_fit(seed)
        self.sp = {}
        for sp in C1.SPLITS:
            v = ctx.sp[sp]
            self.sp[sp] = {"s": v["s"], "d": v["d"], "lstm_cats": v["lstm_cats"], "y": v["y"],
                           "h_fit": hf.get(sp, np.zeros(len(v["s"]), dtype=np.int16))}


def run_full(runner):
    runner.fit()
    P = {sp: np.stack([runner.predict(sp, h) for h in range(1, E.H_MAX + 1)]) for sp in ("val", "test")}
    return P


def t_first(ctx):
    seq, _, _ = build_seq(ctx)
    r1 = MD1.LSTMRunner(Shim(ctx), 50, nolog)
    check("(3) 입력 38채널 = 1차 LSTMRunner.Wg (비트 단위)", bool(np.array_equal(seq, r1.Wg.cpu().numpy())), f"{seq.shape}")
    # (3a) L=100, 시드 42 -> 1차 저장 결과
    t0 = time.time()
    mine = DLRunner(ctx, {"id": "DL_lstmA"}, 42, nolog, False, "lstm", "A", L=100)
    P = run_full(mine)
    hist = [h["val_ap"] for h in mine.history]
    rd = E.EXP1_RESULTS / "lstm__seq__L100"
    c1 = json.load(open(rd / "config.json", encoding="utf-8"))["model_info"]["history"]
    h1 = [h["val_ap"] for h in c1]
    mine.close()
    for sp in ("val", "test"):
        p1, y1, _ = read_preds(rd / f"pred_{sp}.parquet")
        d = float(np.max(np.abs(p1 - P[sp])))
        check(f"(3) DL_lstmA(시드 42, L=100) {sp} 예측 = 1차 lstm__seq__L100", d == 0.0 and np.array_equal(y1, ctx.sp[sp]["y"]),
              f"({P[sp].shape}, 최대 차이 {d:.2e})")
    check("(3) DL_lstmA(L=100) 에폭별 조기종료 AP = 1차 기록", hist == h1,
          f"(에폭 {len(hist)} vs {len(h1)}, 최적 {int(np.argmax(hist)) + 1} vs {int(np.argmax(h1)) + 1}, {time.time() - t0:.0f}s)")
    # (3b) L=50: 1차 LSTMRunner 그대로 vs DL_lstmA (같은 프로세스)
    t0 = time.time()
    r1.fit()
    P1 = {sp: np.stack([r1.predict(sp, h) for h in range(1, E.H_MAX + 1)]) for sp in ("val", "test")}
    hist1 = [h["val_ap"] for h in r1.history]
    r1.close()
    del r1
    mine = DLRunner(ctx, {"id": "DL_lstmA"}, 42, nolog, False, "lstm", "A")
    P2 = run_full(mine)
    hist2 = [h["val_ap"] for h in mine.history]
    mine.close()
    d = max(float(np.max(np.abs(P1[sp] - P2[sp]))) for sp in ("val", "test"))
    check("(3) DL_lstmA(시드 42, L=50) = 1차 LSTMRunner(L=50) 예측·에폭별 AP", d == 0.0 and hist1 == hist2,
          f"(최대 차이 {d:.2e}, 에폭 {len(hist2)}, {time.time() - t0:.0f}s)")
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- (4)
def t_repro():
    log = Logger(E.EXP2_DIR / "logs" / "progress_selftest_dl.log")
    sctx = Ctx2(True, log)
    sm = E.paths(True)["results"]
    for eid, (a, m) in ARCH.items():
        rd = sm / E.run_dir_name(eid, 42)
        if not (rd / "pred_test.parquet").exists():
            check(f"(4) {eid} smoke 재현", False, "(smoke 결과 없음)")
            continue
        rr = DLRunner(sctx, {"id": eid}, 42, nolog, True, a, m)
        P = run_full(rr)
        rr.close()
        d = 0.0
        for sp in ("val", "test"):
            p0, _, _ = read_preds(rd / f"pred_{sp}.parquet")
            d = max(d, float(np.max(np.abs(p0 - P[sp]))))
        check(f"(4) {eid}: smoke 다시 학습 = 저장된 smoke 예측(다른 프로세스)", d == 0.0, f"(최대 차이 {d:.2e})")
    # 시드가 바뀌면 달라지는지(lstmA, 2에폭)
    pa = DLRunner(sctx, {"id": "x"}, 42, nolog, True, "lstm", "A", max_epochs=2)
    pb = DLRunner(sctx, {"id": "x"}, 43, nolog, True, "lstm", "A", max_epochs=2)
    pa.fit()
    pb.fit()
    qa, qb = pa.predict("val", 5), pb.predict("val", 5)
    check("(4) 시드 42·43 결과 다름(초기값·셔플·h 배정)", not np.array_equal(qa, qb),
          f"(학습 h 같은 비율 {float(np.mean(pa.hf['train'] == pb.hf['train'])):.3f})")
    pa.close()
    pb.close()


# ---------------------------------------------------------------- (5)
def t_maskstats(ctx):
    end_idx = train_end_idx(ctx)
    n = len(ctx.sp["train"]["idx"])
    out = {}
    for seed in E.SEEDS:
        h = ctx.h_fit(seed)["train"]
        Y = label_matrix(ctx, "train", np.arange(n), h, end_idx)
        st = mask_stats(ctx, "train", np.arange(n), h, Y, end_idx)
        out[f"s{seed}"] = st
        o = st["observed_per_sample"]
        say(f"    (5) 시드 {seed}: 표본 {st['n_samples']:,} x 60 = {st['cells']:,}칸, 관측 {st['observed_cells']:,} "
            f"({st['observed_rate']:.4%}), 빈칸 {st['masked_rate']:.4%} = 학습 끝 이후 {st['masked_after_train_end_rate']:.4%} "
            f"+ 행 없음 {st['masked_no_row_rate']:.4%}; 표본당 관측 칸 평균 {o['mean']:.3f}, 최소 {o['min']}, "
            f"p10 {o['p10']:.0f}, 중앙값 {o['median']:.0f}, p90 {o['p90']:.0f}, 최대 {o['max']}; "
            f"검출 칸 {st['positive_cells']:,} (관측 칸 중 {st['positive_rate_in_observed']:.4%}), 검출 칸 있는 표본 {st['samples_with_positive_cell']:,}; "
            f"학습 끝 이후 칸이 있는 표본 {st['samples_with_after_train_end_cells']:,}")
        check(f"(5) 시드 {seed}: 표본마다 관측 칸 >= 1 (j=h)", o["min"] >= 1)
    p = E.EXP2_DIR / "logs" / "dl_maskstats.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"    (5) 저장: {p}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    rng = np.random.default_rng(2026)
    log = Logger(E.EXP2_DIR / "logs" / "progress_selftest_dl.log")
    only = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    say(f"DL 자체 점검 시작 {time.strftime('%Y-%m-%d %H:%M:%S')} (torch {torch.__version__}, {torch.cuda.get_device_name(0)}), 항목 {only}")
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache") if set(only) & {"1", "2", "3", "5"} else None
    try:
        if "1" in only:
            t_labels(ctx, rng)
        if "5" in only:
            t_maskstats(ctx)
        if "2" in only:
            t_alignment(ctx, rng)
        if "3" in only:
            t_first(ctx)
        if "4" in only:
            t_repro()
    except Exception:
        import traceback
        check("예외 없이 끝남", False, traceback.format_exc())
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
