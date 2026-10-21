# -*- coding: utf-8 -*-
"""사양 4절 11번 딥러닝 단계: BiLSTM·트랜스포머 x 방식 A·B x 학습 행 순서(에폭마다 섞기·시간 순서) - 3차 겹 틀 그대로.

공통(1차 베이스라인과 같게): 기상 창 = 1차 베이스라인 창(T−L … T−1, T = d − h), 기상 변수 = 기상 거르기에서 남은 변수(값 채널 + 결측 표시 채널),
  표준화 = 그 겹 학습 끝 날짜까지 기상표 평균·표준편차(dl3.fold_stats), 범주 4칸(sido·sgg·p2·p3) 임베딩(그 겹 학습 행 어휘, 없던 범주 0),
  Adam 1e-3, 배치 1024, 최대 30에폭, 조기 종료 = 그 겹 맞혀 보는 행 h 별 AP 의 h 1..60 평균(cv.ap60_fast) 인내 5 -> 최고 에폭 복원,
  고르기·보고 점수 = cv.score_fold(구간 가중 PR-AUC, 기준값 = 60개 h 합친 예측에서 Recall ≥ 0.80 중 Precision 최대). 다른 입력 칸(이력 등)은 넣지 않음.
구조
  BiLSTM     = dl3.BiNet 과 같은 구조(양방향 LSTM 64, 마지막 은닉 정·역 128 + 임베딩 + [h/60] -> Linear(64) ReLU Dropout(0.2) Linear(출력)).
  트랜스포머 = 2차 e2dl_impl.make_net('trf') 그대로(Linear(채널->64) + 학습 위치 임베딩(L) -> TransformerEncoder 2층·4헤드·ff 128·dropout 0.1
               -> 시간 평균 풀링 + 임베딩 + [h/60] -> Linear(64) ReLU Linear(출력)).
방식 A = 입력에 h/60, 출력 1칸(그 행 d 의 검출 확률). 방식 B = h 입력 없음, 출력 60칸(T+1 … T+60 의 검출 확률):
  학습 표본 = 학습 행 (단위, d) 마다 그 행의 h 로 T = d − h, 정답 y[j] = (단위, T+j) 단위x날짜 행의 y(행이 없거나 T+j > 겹 학습 끝이면 빈칸 = 마스크,
  j = h 칸은 그 행 자신이라 늘 관측 - 점검), 손실 = 관측 칸 BCE 평균. 평가 = 맞혀 보는 (행, h) 마다 T = d − h 입력의 출력[h].
학습 행 순서: shuffle = 에폭마다 torch.randperm(Generator(seed)) (dl3·지금 방식), time = 예측일 T 순서(같으면 행 순서)로 고정, 섞지 않음.
재현성: torch.manual_seed(seed)·cuda.manual_seed_all(seed), cudnn.deterministic=True·benchmark=False, CUBLAS_WORKSPACE_CONFIG=:4096:8.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import cv as CV3  # noqa: E402
import dl3  # noqa: E402

H = CV3.H_MAX


class BiNet11(nn.Module):
    """dl3.BiNet 과 같은 구조, 방식 B 면 h 입력 없이 출력 60칸."""

    def __init__(self, n_in, cat_sizes, multi, c=dl3.CFG):
        super().__init__()
        self.multi = multi
        self.lstm = nn.LSTM(n_in, int(c["hidden"]), num_layers=int(c["layers"]), batch_first=True, bidirectional=True)
        self.embs = nn.ModuleList([nn.Embedding(n + 1, int(c["emb_dim"]), padding_idx=0) for n in cat_sizes])
        d = 2 * int(c["hidden"]) + int(c["emb_dim"]) * len(cat_sizes) + (0 if multi else 1)
        self.head = nn.Sequential(nn.Linear(d, int(c["head"])), nn.ReLU(), nn.Dropout(float(c["dropout"])), nn.Linear(int(c["head"]), H if multi else 1))

    def forward(self, x, cats, hn=None):
        _, (hl, _) = self.lstm(x)
        z = [hl[-2], hl[-1]] + [e(cats[:, j]) for j, e in enumerate(self.embs)]
        if not self.multi:
            z.append(hn[:, None])
        o = self.head(torch.cat(z, dim=1))
        return o if self.multi else o.squeeze(1)


def make_net(arch, mode, n_in, sizes, L):
    if arch == "bilstm":
        return BiNet11(n_in, sizes, mode == "B")
    import e2dl_impl
    if e2dl_impl.DLC["trf"]["pos_emb"] < L:
        raise ValueError(f"트랜스포머 위치 임베딩 {e2dl_impl.DLC['trf']['pos_emb']} < 창 {L}")
    return e2dl_impl.make_net("trf", mode, n_in, sizes, L)


def labels_B(ctx, rs, end_idx):
    """방식 B 정답 (n, 60) int8: (단위, T+j) 행 y, 행 없음·T+j > 겹 학습 끝 = −1."""
    unit = ctx.sp[CV3.SP]["unit"][rs.pos].astype(np.int64)
    wd = rs.t[:, None].astype(np.int64) + np.arange(1, H + 1, dtype=np.int64)[None, :]
    Y = np.full(wd.shape, -1, dtype=np.int8)
    ok = wd <= end_idx
    Y[ok] = ctx.lookup_y(np.broadcast_to(unit[:, None], wd.shape)[ok], wd[ok])
    own = Y[np.arange(rs.n), rs.h.astype(np.int64) - 1]
    chk = {"own_cell_equals_row_y": bool(np.array_equal(own, rs.y)), "observed_rate": float((Y >= 0).mean()),
           "after_end_masked": bool((Y[~ok] == -1).all()), "positive_cells": int((Y == 1).sum())}
    if not (chk["own_cell_equals_row_y"] and chk["after_end_masked"]):
        raise RuntimeError(f"방식 B 정답 점검 실패: {chk}")
    return Y, chk


def run_fold(ctx, rs_tr, rs_ev, codes_tr, codes_ev, train_end, seed, log, arch, mode, order, wvars=None):
    """돌려줌 (p (n_ev,) float32, info)."""
    import e2ae
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가 - 딥러닝은 GPU 로")
    dev = torch.device("cuda")
    t0 = time.time()
    c = dict(dl3.CFG)
    W = ctx.weather.W
    L = CV3.L
    end_idx = int((np.datetime64(train_end, "D") - ctx.weather.date0).astype(int))
    mu, sd = dl3.fold_stats(W, end_idx)
    chans = dl3.seq_channels(wvars)
    tb = torch.from_numpy(e2ae.seq_table(W, mu, sd)).to(dev)
    if len(chans) != tb.shape[2]:
        tb = tb[:, :, torch.tensor(chans, device=dev)].contiguous()
    ctr, sizes = dl3.vocab_codes(codes_tr, codes_tr)
    cev, _ = dl3.vocab_codes(codes_tr, codes_ev)
    ar = torch.arange(L, device=dev)
    T_tr = {"s": torch.from_numpy(rs_tr.s).to(dev), "t": torch.from_numpy(rs_tr.t).to(dev), "c": torch.from_numpy(ctr).to(dev),
            "hn": torch.from_numpy(rs_tr.h.astype(np.float32) / H).to(dev), "y": torch.from_numpy(rs_tr.y.astype(np.float32)).to(dev)}
    T_ev = {"s": torch.from_numpy(rs_ev.s).to(dev), "t": torch.from_numpy(rs_ev.t).to(dev), "c": torch.from_numpy(cev).to(dev),
            "hn": torch.from_numpy(rs_ev.h.astype(np.float32) / H).to(dev), "hi": torch.from_numpy(rs_ev.h.astype(np.int64) - 1).to(dev)}
    lab = None
    if mode == "B":
        Y, lab = labels_B(ctx, rs_tr, end_idx)
        T_tr["Y"] = torch.from_numpy((Y == 1).astype(np.float32)).to(dev)
        T_tr["M"] = torch.from_numpy((Y >= 0).astype(np.float32)).to(dev)
        del Y
    offs = rs_ev.h_offsets()
    yev = rs_ev.y

    def xb(T, idx):
        return tb[T["s"][idx][:, None], T["t"][idx][:, None] - L + ar[None, :]]

    n = rs_tr.n
    if order == "time":
        fixed = torch.from_numpy(np.argsort(rs_tr.t, kind="stable").astype(np.int64)).to(dev)
    elif order != "shuffle":
        raise ValueError(order)
    prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        net = make_net(arch, mode, tb.shape[2], sizes, L).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=float(c["lr"]))
        g = torch.Generator().manual_seed(seed)
        lossA = nn.BCEWithLogitsLoss()
        best, best_ep, wait, best_state, hist = -1.0, 0, 0, None, []

        def predict():
            net.eval()
            out = torch.empty(rs_ev.n, device=dev)
            with torch.no_grad():
                for a in range(0, rs_ev.n, c["eval_batch"]):
                    idx = torch.arange(a, min(rs_ev.n, a + c["eval_batch"]), device=dev)
                    if mode == "A":
                        o = net(xb(T_ev, idx), T_ev["c"][idx], T_ev["hn"][idx])
                    else:
                        o = net(xb(T_ev, idx), T_ev["c"][idx]).gather(1, T_ev["hi"][idx][:, None]).squeeze(1)
                    out[a:a + len(idx)] = torch.sigmoid(o)
            return out.cpu().numpy().astype(np.float32)

        for ep in range(1, int(c["max_epochs"]) + 1):
            te = time.time()
            net.train()
            perm = torch.randperm(n, generator=g).to(dev) if order == "shuffle" else fixed
            tot, nb = 0.0, 0
            bs = int(c["batch"])
            for a in range(0, n, bs):
                idx = perm[a:a + bs]
                x = xb(T_tr, idx)
                if mode == "A":
                    loss = lossA(net(x, T_tr["c"][idx], T_tr["hn"][idx]), T_tr["y"][idx])
                else:
                    o = net(x, T_tr["c"][idx])
                    m = T_tr["M"][idx]
                    loss = (nn.functional.binary_cross_entropy_with_logits(o, T_tr["Y"][idx], reduction="none") * m).sum() / m.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                nb += 1
            p = predict()
            ap = CV3.ap60_fast(yev, p, offs)
            hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "es_ap60": ap, "seconds": round(time.time() - te, 1)})
            log(f"    {arch}{mode}/{order} 에폭 {ep}: 학습 손실 {tot / max(nb, 1):.5f}, 맞혀 보는 행 PR-AUC {ap:.5f}, {time.time() - te:.1f}s")
            if ap > best:
                best, best_ep, wait = ap, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= int(c["patience"]):
                    break
        net.load_state_dict(best_state)
        p = predict()
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev
    info = {"arch": arch, "mode": mode, "order": order, "config": c, "n_seq_channels": len(chans), "wvars": wvars, "labels_B": lab,
            "best_epoch": best_ep, "n_epochs": len(hist), "es_best_value": best, "history": hist,
            "n_params": int(sum(q.numel() for q in net.parameters())), "vocab_sizes": sizes,
            "standardize": f"겹 학습 끝 {train_end} 까지 기상표 평균·표준편차", "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "fit_seconds": round(time.time() - t0, 1)}
    del tb, T_tr, T_ev, net
    torch.cuda.empty_cache()
    return p, info
