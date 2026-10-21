# -*- coding: utf-8 -*-
"""바닐라 1단계 BiLSTM(양방향 LSTM) - 2차 DL 방식 A(h 입력, 출력 1칸)와 같은 틀에서 LSTM 만 양방향으로.

입력: 기상 창 T−L … T−1 (T = d − h, 시간순) 19변수 표준화(결측 0) + 결측 표시 19 = 38채널(2차 e2dl_impl.build_seq 와 같은 식).
      표준화 통계 = 그 겹 학습 끝 날짜까지의 기상표(시군구 x 날짜) 값의 변수별 평균·표준편차(결측 무시, ddof=0) - 겹마다 다시(누수 방지).
      범주 4칸(sido·sgg·p2·p3) 임베딩 각 8차원(그 겹 학습 행에 있던 범주만 1..n, 없던 범주 = 0 = 0 벡터), h/60.
구조: nn.LSTM(38, 64, 1층, bidirectional=True) -> 마지막 은닉(정방향 마지막 + 역방향 마지막 = 128) + 임베딩 32 + h/60 [+ 표 입력]
      -> Linear(-> 64) ReLU Dropout(0.2) Linear(-> 1)   (2차 DL_lstmA 머리와 같고 LSTM 출력만 2배)
표 입력(튜닝 단계, '변수 전부'): 기상 요약(wx) 묶음·h 를 뺀 숫자 칸 전부(외부 자료·파생변수·이력·제품명·이상치 점수 등)를
      그 겹 학습 행 평균·표준편차로 표준화(결측·표준편차 0 은 0) -> 머리 입력에 이어 붙임. 기상은 시퀀스로만 들어감.
튜닝(Optuna) 파라미터 = CFG 의 hidden·layers·dropout(LSTM 층 사이·머리)·lr·batch·emb_dim·head (config3 optuna.spaces.bilstm). 최대 에폭·인내는 고정.
학습: Adam 1e-3, 배치 1024, 최대 30에폭, BCE(가중치 없음), 에폭마다 셔플(Generator(seed)), 시드 = torch.manual_seed(seed).
조기 종료: 그 겹 맞혀 보는 행 전부(h 1..60)의 PR-AUC(h 별 AP 의 평균, cv.ap60_fast) 인내 5 -> 최고 에폭 가중치 복원. GPU 필수.
재현성: cudnn.deterministic=True·benchmark=False, CUBLAS_WORKSPACE_CONFIG=:4096:8(이 모듈을 CUDA 전에 import).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import cv as CV3  # noqa: E402

CFG = {"hidden": 64, "layers": 1, "emb_dim": 8, "head": 64, "dropout": 0.2, "lr": 1e-3, "batch": 1024, "max_epochs": 30,
       "patience": 5, "eval_batch": 8192}


class BiNet(nn.Module):
    def __init__(self, n_in, cat_sizes, c=CFG, n_tab=0):
        super().__init__()
        c = {**CFG, **c}
        self.lstm = nn.LSTM(n_in, int(c["hidden"]), num_layers=int(c["layers"]), batch_first=True, bidirectional=True,
                            dropout=float(c["dropout"]) if int(c["layers"]) > 1 else 0.0)
        self.embs = nn.ModuleList([nn.Embedding(n + 1, int(c["emb_dim"]), padding_idx=0) for n in cat_sizes])
        d = 2 * int(c["hidden"]) + int(c["emb_dim"]) * len(cat_sizes) + 1 + n_tab
        self.head = nn.Sequential(nn.Linear(d, int(c["head"])), nn.ReLU(), nn.Dropout(float(c["dropout"])), nn.Linear(int(c["head"]), 1))

    def forward(self, x, cats, hn, tab=None):
        _, (hl, _) = self.lstm(x)                       # hl (layers*2, B, H): [-2] = 정방향 마지막, [-1] = 역방향 마지막
        z = [hl[-2], hl[-1]] + [e(cats[:, j]) for j, e in enumerate(self.embs)] + [hn[:, None]]
        if tab is not None:
            z.append(tab)
        return self.head(torch.cat(z, dim=1)).squeeze(1)


def tab_standardize(X_tr, X_ev):
    """그 겹 학습 행 평균·표준편차(결측 무시, ddof=0)로 표준화, 결측·표준편차 0/전부 결측은 0. float32."""
    Xd = X_tr.astype(np.float64)
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(Xd, axis=0), np.nanstd(Xd, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)

    def z(X):
        out = np.empty(X.shape, dtype=np.float32)
        for a in range(0, len(X), 200000):
            Z = (X[a:a + 200000].astype(np.float64) - mu) / sd
            Z[np.isnan(Z)] = 0.0
            out[a:a + 200000] = Z
        return out
    return z(X_tr), z(X_ev)


def fold_stats(W, end_idx):
    sub = W[:, : end_idx + 1, :].reshape(-1, W.shape[2]).astype(np.float64)
    mu, sd = np.nanmean(sub, axis=0), np.nanstd(sub, axis=0)
    if not (np.all(np.isfinite(mu)) and np.all(sd > 0)):
        raise RuntimeError("표준화 통계 이상")
    return mu.astype(np.float32), sd.astype(np.float32)


def vocab_codes(codes_tr, codes):
    """그 겹 학습 행 범주만 1..n, 그 밖 0."""
    out = np.zeros(codes.shape, dtype=np.int64)
    sizes = []
    for j in range(codes.shape[1]):
        voc = np.unique(codes_tr[:, j])
        i = np.clip(np.searchsorted(voc, codes[:, j]), 0, len(voc) - 1)
        out[:, j] = np.where(voc[i] == codes[:, j], i + 1, 0)
        sizes.append(len(voc))
    return out, sizes


def run_fold(ctx, rs_tr, rs_ev, codes_tr, codes_ev, train_end, seed, log, params=None, tab=None):
    """돌려줌 (p (n_ev,) float32, info). params = CFG 덮어쓰기(튜닝), tab = (X_tr, X_ev) 표 입력(표준화 전) 또는 None."""
    import e2ae
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가 - BiLSTM 은 GPU 로")
    dev = torch.device("cuda")
    t0 = time.time()
    W = ctx.weather.W
    L = CV3.L
    end_idx = int((np.datetime64(train_end, "D") - ctx.weather.date0).astype(int))
    mu, sd = fold_stats(W, end_idx)
    c = {**CFG, **(params or {})}
    tb = torch.from_numpy(e2ae.seq_table(W, mu, sd)).to(dev)
    n_tab = 0
    if tab is not None:
        Ztr, Zev = tab_standardize(tab[0], tab[1])
        n_tab = Ztr.shape[1]
    ctr, sizes = vocab_codes(codes_tr, codes_tr)
    cev, _ = vocab_codes(codes_tr, codes_ev)
    ar = torch.arange(L, device=dev)
    T_tr = {"s": torch.from_numpy(rs_tr.s).to(dev), "t": torch.from_numpy(rs_tr.t).to(dev), "c": torch.from_numpy(ctr).to(dev),
            "h": torch.from_numpy(rs_tr.h.astype(np.float32) / 60.0).to(dev), "y": torch.from_numpy(rs_tr.y.astype(np.float32)).to(dev)}
    T_ev = {"s": torch.from_numpy(rs_ev.s).to(dev), "t": torch.from_numpy(rs_ev.t).to(dev), "c": torch.from_numpy(cev).to(dev),
            "h": torch.from_numpy(rs_ev.h.astype(np.float32) / 60.0).to(dev)}
    if n_tab:
        T_tr["x"] = torch.from_numpy(Ztr).to(dev)
        T_ev["x"] = torch.from_numpy(Zev).to(dev)
        del Ztr, Zev
    offs = rs_ev.h_offsets()
    yev = rs_ev.y

    def xb(T, idx):
        return tb[T["s"][idx][:, None], T["t"][idx][:, None] - L + ar[None, :]]

    prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        net = BiNet(tb.shape[2], sizes, c, n_tab).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=float(c["lr"]))
        lossf = nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(seed)
        n = rs_tr.n
        best, best_ep, wait, best_state, hist = -1.0, 0, 0, None, []

        def predict():
            net.eval()
            out = torch.empty(rs_ev.n, device=dev)
            with torch.no_grad():
                for a in range(0, rs_ev.n, CFG["eval_batch"]):
                    idx = torch.arange(a, min(rs_ev.n, a + CFG["eval_batch"]), device=dev)
                    out[a:a + len(idx)] = torch.sigmoid(net(xb(T_ev, idx), T_ev["c"][idx], T_ev["h"][idx], T_ev["x"][idx] if n_tab else None))
            return out.cpu().numpy().astype(np.float32)

        for ep in range(1, int(c["max_epochs"]) + 1):
            te = time.time()
            net.train()
            perm = torch.randperm(n, generator=g).to(dev)
            tot, nb = 0.0, 0
            bs = int(c["batch"])
            for a in range(0, n, bs):
                idx = perm[a:a + bs]
                loss = lossf(net(xb(T_tr, idx), T_tr["c"][idx], T_tr["h"][idx], T_tr["x"][idx] if n_tab else None), T_tr["y"][idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                nb += 1
            if c.get("fixed_epochs"):                  # 최종 재학습: 정한 에폭 수만큼, 맞혀 보는 행(=시험) 평가 없음
                hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "seconds": round(time.time() - te, 1)})
                best_ep, best_state = ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                if ep >= int(c["fixed_epochs"]):
                    break
                continue
            p = predict()
            ap = CV3.ap60_fast(yev, p, offs)
            hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "es_ap60": ap, "seconds": round(time.time() - te, 1)})
            log(f"    BiLSTM 에폭 {ep}: 학습 손실 {tot / max(nb, 1):.5f}, 맞혀 보는 행 PR-AUC {ap:.5f}, {time.time() - te:.1f}s")
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
    info = {"model": "bilstm", "config": c, "n_tab": int(n_tab), "best_epoch": best_ep, "n_epochs": len(hist), "es_best_value": best, "history": hist,
            "standardize": f"겹 학습 끝 {train_end} 까지 기상표 평균·표준편차", "vocab_sizes": sizes,
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "fit_seconds": round(time.time() - t0, 1)}
    del tb, T_tr, T_ev, net
    torch.cuda.empty_cache()
    return p, info
