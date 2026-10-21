# -*- coding: utf-8 -*-
"""딥러닝 4종 (사양 DL). e2dl.register_dl 로 등록 -> run.py 의 dl 단계가 그대로 실행한다.

공통(사양)
  기상 창 50일(T-50 .. T-1, T = d - h, T 당일 제외), 시간순 19변수를 학습 기간(<= 2024-06-30) 기상표 평균·표준편차로
  표준화(1차 Weather.train_period_stats: NaN 무시, ddof=0) -> 결측 0 + 결측 표시 19칸 = 38채널 (1차 LSTMRunner 와 같은 계산).
  범주 4(sido, sgg, p2, p3) 임베딩 각 8차원, 학습 행에 없던 범주 = 0번 = 0 벡터(padding_idx=0, 학습 안 됨) - 1차 lstm_vocab_codes.
  Adam 1e-3, 배치 1024, 최대 30에폭, 조기 종료 = 조기 종료용 검증 PR-AUC(sklearn AP) 인내 5 -> 최고 에폭 가중치 복원,
  BCE(pos_weight 없음), 에폭마다 셔플(torch.randperm, Generator(seed)), GPU 필수(없으면 오류).
  시드 s: torch.manual_seed(s)·cuda.manual_seed_all(s)(초기값·dropout), Generator(s)(셔플), 학습 h = ctx.h_fit(s)['train']
  (default_rng(s)), 조기 종료용 검증 h = ctx.h_fit(s)['val'] (default_rng(s+1)). 시드 42 의 lstmA 를 L=100 으로 돌리면
  1차 lstm__seq__L100 과 같은 계산(selftest_dl 에서 확인).

방식 A (h 입력): 출력 1칸 = 목표일 d 의 검출 확률, 입력에 h/60.
  DL_lstmA = 1차 models.make_lstm_net 그대로(단방향 1층 은닉 64, 머리 Linear(64+32+1->64) ReLU Dropout(0.2) Linear(->1)).
  DL_trfA  = Linear(38->64) + 학습되는 위치 임베딩(50, nn.Embedding 기본 초기값) -> TransformerEncoder(2층, 4헤드, ff 128,
             dropout 0.1, post-norm·ReLU = PyTorch 기본) -> 시간 평균 풀링 + 임베딩 32 + h/60 -> Linear(->64) ReLU Linear(->1).
방식 B (다중 출력 60칸, h 입력 없음): 출력 j(1..60) = T+j 일의 검출 확률.
  학습 표본 = 학습 행 (단위, d) 마다 그 시드의 h 로 T = d - h, 입력 = T-50 .. T-1 창 + 범주,
  정답 y[j] = (단위, T+j) 단위x날짜 행의 y (행이 없거나 T+j > 2024-06-30 이면 빈칸 = 마스크). j = h 칸은 그 행 자신이라 늘 관측.
  손실 = 관측 칸만 BCE 평균(배치 안 관측 칸 수로 나눔). 조기 종료 = 조기 종료용 검증 (단위, d, h) 의 출력[h] vs y 의 AP.
  평가 = 검증·시험 행 (단위, d), h 마다 T = d - h 입력의 출력[h] (방식 A 와 같은 행·같은 h·같은 창).
  DL_lstmB / DL_trfB = A 와 같은 구조에서 h 입력을 빼고 출력층만 60칸.
"""
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

import config as C1
import models as MD1
import e2cfg as E
from e2dl import register_dl

DLC = E.DL
N_CH = 2 * len(C1.WEATHER_VARS)          # 38


# ---------------------------------------------------------------- 입력·정답
def build_seq(ctx):
    """(S, D, 38) float32: 표준화 19(결측 0) + 결측 표시 19. 1차 LSTMRunner.__init__ 과 같은 식."""
    W = ctx.weather.W
    mu, sd = ctx.weather.train_period_stats()
    if not (np.all(np.isfinite(mu)) and np.all(sd > 0)):
        raise RuntimeError("표준화 통계 이상")
    miss = np.isnan(W)
    Wn = np.where(miss, 0.0, (W - mu) / sd).astype(np.float32)
    seq = np.ascontiguousarray(np.concatenate([Wn, miss.astype(np.float32)], axis=2))
    return seq, mu, sd


def train_end_idx(ctx) -> int:
    return int(ctx.weather.d_idx(np.array([np.datetime64(C1.TRAIN_END)]))[0])


def label_matrix(ctx, sp, pos, h, end_idx):
    """방식 B 정답 (n, 60) int8: 칸 j-1 = (단위, T+j) 단위x날짜 행의 y, T = d - h. 행 없음 또는 T+j > end_idx 이면 -1."""
    v = ctx.sp[sp]
    T = v["d"][pos].astype(np.int64) - np.asarray(h, dtype=np.int64)
    wd = T[:, None] + np.arange(1, E.H_MAX + 1, dtype=np.int64)[None, :]
    Y = np.full(wd.shape, -1, dtype=np.int8)
    ok = wd <= end_idx
    unit = np.broadcast_to(v["unit"][pos].astype(np.int64)[:, None], wd.shape)
    Y[ok] = ctx.lookup_y(unit[ok], wd[ok])
    return Y


def mask_stats(ctx, sp, pos, h, Y, end_idx):
    """빈칸 비율·표본당 관측 칸 수 (보고용)."""
    v = ctx.sp[sp]
    T = v["d"][pos].astype(np.int64) - np.asarray(h, dtype=np.int64)
    wd = T[:, None] + np.arange(1, E.H_MAX + 1, dtype=np.int64)[None, :]
    obs = Y >= 0
    after = wd > end_idx
    n, cells = Y.shape[0], Y.size
    per = obs.sum(1)
    own = Y[np.arange(n), np.asarray(h, dtype=np.int64) - 1]
    q = np.percentile(per, [10, 25, 50, 75, 90])
    return {"n_samples": int(n), "cells": int(cells),
            "observed_cells": int(obs.sum()), "observed_rate": float(obs.mean()),
            "masked_rate": float(1 - obs.mean()),
            "masked_after_train_end_cells": int(after.sum()), "masked_after_train_end_rate": float(after.mean()),
            "masked_no_row_cells": int((~obs & ~after).sum()), "masked_no_row_rate": float((~obs & ~after).mean()),
            "samples_with_after_train_end_cells": int(after.any(1).sum()),
            "observed_per_sample": {"mean": float(per.mean()), "min": int(per.min()), "p10": float(q[0]),
                                    "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
                                    "p90": float(q[4]), "max": int(per.max())},
            "observed_per_sample_counts": {str(k): int(c) for k, c in zip(*np.unique(per, return_counts=True))},
            "positive_cells": int((Y == 1).sum()), "positive_rate_in_observed": float((Y == 1).sum() / obs.sum()),
            "samples_with_positive_cell": int((Y == 1).any(1).sum()),
            "own_cell_equals_row_y": bool(np.array_equal(own, v["y"][pos])),
            "after_train_end_all_masked": bool((Y[after] == -1).all())}


# ---------------------------------------------------------------- 신경망
def make_net(arch, mode, n_in, cat_sizes, L):
    """arch 'lstm'|'trf', mode 'A'(h 입력, 출력 1)|'B'(h 없음, 출력 60). forward(x, cats, hn=None)."""
    if arch == "lstm" and mode == "A":
        return MD1.make_lstm_net(n_in, cat_sizes)          # 1차와 같은 구조·같은 파라미터 생성 순서

    multi = mode == "B"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if arch == "lstm":
                self.lstm = nn.LSTM(n_in, DLC["lstm_hidden"], num_layers=DLC["lstm_layers"], batch_first=True)
                enc = DLC["lstm_hidden"]
            else:
                t = DLC["trf"]
                self.proj = nn.Linear(n_in, t["d_model"])
                self.pos_emb = nn.Embedding(t["pos_emb"], t["d_model"])
                layer = nn.TransformerEncoderLayer(t["d_model"], t["heads"], dim_feedforward=t["ff"],
                                                   dropout=t["dropout"], batch_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers=t["layers"], enable_nested_tensor=False)
                enc = t["d_model"]
            self.embs = nn.ModuleList([nn.Embedding(n + 1, DLC["emb_dim"], padding_idx=0) for n in cat_sizes])
            d = enc + DLC["emb_dim"] * len(cat_sizes) + (0 if multi else 1)
            out = E.H_MAX if multi else 1
            if arch == "lstm":       # 1차 LSTM 머리와 같음(출력층만 다름)
                self.head = nn.Sequential(nn.Linear(d, C1.LSTM_HEAD), nn.ReLU(), nn.Dropout(C1.LSTM_DROPOUT),
                                          nn.Linear(C1.LSTM_HEAD, out))
            else:
                hh = DLC["trf"]["head_hidden"]
                self.head = nn.Sequential(nn.Linear(d, hh), nn.ReLU(), nn.Linear(hh, out))

        def encode(self, x):
            if arch == "lstm":
                _, (h_last, _) = self.lstm(x)
                return h_last[-1]
            z = self.proj(x) + self.pos_emb.weight[None, : x.shape[1], :]
            return self.enc(z).mean(dim=1)

        def forward(self, x, cats, hn=None):
            z = [self.encode(x)] + [e(cats[:, j]) for j, e in enumerate(self.embs)]
            if not multi:
                z.append(hn[:, None])
            o = self.head(torch.cat(z, dim=1))
            return o if multi else o.squeeze(1)

    if arch == "trf" and L > DLC["trf"]["pos_emb"]:
        raise ValueError(f"창 {L}일 > 위치 임베딩 {DLC['trf']['pos_emb']}")
    return Net()


# ---------------------------------------------------------------- 실행기
class DLRunner:
    def __init__(self, ctx, exp, seed, log, smoke, arch, mode, L=None, max_epochs=None, patience=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 사용 불가 - 딥러닝은 GPU 필수")
        t0 = time.time()
        self.ctx, self.exp, self.seed, self.log, self.smoke = ctx, exp, seed, log, smoke
        self.arch, self.mode = arch, mode
        self.L = int(L or DLC["L"])
        self.max_epochs = int(max_epochs or DLC["max_epochs"])
        self.patience = int(patience or DLC["patience"])
        self.batch = DLC["batch"]
        self.eval_batch = C1.LSTM_EVAL_BATCH          # 1차와 같은 1024 (사양에 없음)
        self.dev = torch.device("cuda")
        self.pos = {sp: np.arange(len(ctx.sp[sp]["idx"])) for sp in C1.SPLITS}
        seq, self.mu, self.sd = build_seq(ctx)
        self.Wg = torch.from_numpy(seq).to(self.dev)
        del seq
        self.ar = torch.arange(self.L, device=self.dev)
        hf = ctx.h_fit(seed)
        self.hf = hf
        self.T = {}
        for sp in C1.SPLITS:
            r = ctx.sp[sp]
            self.T[sp] = {"s": torch.from_numpy(r["s"]).to(self.dev), "d": torch.from_numpy(r["d"]).to(self.dev),
                          "c": torch.from_numpy(r["lstm_cats"]).to(self.dev),
                          "y": torch.from_numpy(r["y"].astype(np.float32)).to(self.dev)}
        self.T["train"]["hf"] = torch.from_numpy(hf["train"].astype(np.int64)).to(self.dev)
        self.T["val"]["hf"] = torch.from_numpy(hf["val"].astype(np.int64)).to(self.dev)
        self.cat_sizes = list(ctx.lstm_cat_sizes)
        self.mask_info = None
        if mode == "B":
            self.end_idx = train_end_idx(ctx)
            ntr = len(ctx.sp["train"]["idx"])
            Y = label_matrix(ctx, "train", np.arange(ntr), hf["train"], self.end_idx)
            self.mask_info = mask_stats(ctx, "train", np.arange(ntr), hf["train"], Y, self.end_idx)
            if not (self.mask_info["own_cell_equals_row_y"] and self.mask_info["after_train_end_all_masked"]):
                raise RuntimeError(f"방식 B 정답 행렬 검사 실패: {self.mask_info}")
            self.T["train"]["Y"] = torch.from_numpy((Y == 1).astype(np.float32)).to(self.dev)
            self.T["train"]["M"] = torch.from_numpy((Y >= 0).astype(np.float32)).to(self.dev)
            mi = self.mask_info
            log(f"    방식 B 정답: 표본 {mi['n_samples']:,} x 60칸, 관측 {mi['observed_rate']:.4%} "
                f"(빈칸: 학습 끝 이후 {mi['masked_after_train_end_rate']:.4%}, 행 없음 {mi['masked_no_row_rate']:.4%}), "
                f"표본당 관측 칸 평균 {mi['observed_per_sample']['mean']:.3f} / 중앙값 {mi['observed_per_sample']['median']:.0f} "
                f"/ 최대 {mi['observed_per_sample']['max']}, 검출 칸 {mi['positive_cells']:,}")
        self.n_train_rows = int(len(ctx.sp["train"]["idx"]))
        self.n_train_pos = int(ctx.sp["train"]["y"].sum())
        self.n_features = self.L * N_CH + len(C1.CAT_COLS) + (1 if mode == "A" else 0)
        self.n_features_desc = (f"시퀀스 {self.L}일 x {N_CH}채널(표준화 19 + 결측 표시 19) + 범주 4(임베딩 각 {DLC['emb_dim']})"
                                + (" + h/60" if mode == "A" else ", h 입력 없음, 출력 60칸(T+1..T+60)"))
        self.cols = None
        self.feature_seconds = time.time() - t0
        self.n_in = N_CH

    # 창: (n, L, 38), 위치 0 = T-L, L-1 = T-1
    def _x(self, s, t):
        return self.Wg[s[:, None], t[:, None] - self.L + self.ar[None, :]]

    def _out(self, x, c, hh):
        """확률 대상 logit (n,): A = net(x, c, h/60), B = net(x, c)[:, h-1]."""
        if self.mode == "A":
            return self.net(x, c, hh.float() / E.H_MAX)
        return self.net(x, c).gather(1, (hh - 1)[:, None]).squeeze(1)

    def _predict_tensor(self, sp, h_t, batch=None):
        batch = batch or self.eval_batch
        T = self.T[sp]
        n = T["s"].shape[0]
        out = torch.empty(n, device=self.dev)
        self.net.eval()
        with torch.no_grad():
            for a in range(0, n, batch):
                b = slice(a, min(n, a + batch))
                hh = h_t[b]
                out[b] = torch.sigmoid(self._out(self._x(T["s"][b], T["d"][b] - hh), T["c"][b], hh))
        return out.cpu().numpy().astype(np.float32)

    def build_net(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        self.net = make_net(self.arch, self.mode, self.n_in, self.cat_sizes, self.L).to(self.dev)
        self.n_params = int(sum(p.numel() for p in self.net.parameters()))
        return self.net

    def fit(self, max_batches=None):
        self.build_net()
        opt = torch.optim.Adam(self.net.parameters(), lr=DLC["lr"])
        lossA = nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(self.seed)
        tr = self.T["train"]
        n = tr["s"].shape[0]
        yv = self.ctx.sp["val"]["y"]
        best_ap, best_state, best_ep, wait = -1.0, None, 0, 0
        self.history = []
        t0 = time.time()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        name = f"{self.arch}{self.mode}"
        for ep in range(1, self.max_epochs + 1):
            te = time.time()
            self.net.train()
            perm = torch.randperm(n, generator=g).to(self.dev)
            tot, nb, nobs = 0.0, 0, 0.0
            for a in range(0, n, self.batch):
                if max_batches is not None and nb >= max_batches:
                    break
                b = perm[a:a + self.batch]
                hh = tr["hf"][b]
                x = self._x(tr["s"][b], tr["d"][b] - hh)
                if self.mode == "A":
                    logit = self.net(x, tr["c"][b], hh.float() / E.H_MAX)
                    loss = lossA(logit, tr["y"][b])
                else:
                    o = self.net(x, tr["c"][b])
                    m = tr["M"][b]
                    lc = nn.functional.binary_cross_entropy_with_logits(o, tr["Y"][b], reduction="none")
                    loss = (lc * m).sum() / m.sum()
                    nobs += float(m.sum())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                nb += 1
            torch.cuda.synchronize()
            tfit = time.time() - te
            tv = time.time()
            p = self._predict_tensor("val", self.T["val"]["hf"])
            ap = float(average_precision_score(yv, p))
            tval = time.time() - tv
            rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_ap": ap, "train_seconds": tfit,
                   "val_seconds": tval, "seconds": time.time() - te, "batches": nb,
                   "gpu_peak_allocated_gb_so_far": torch.cuda.max_memory_allocated() / 1e9}
            if self.mode == "B":
                rec["observed_cells"] = nobs
            self.history.append(rec)
            self.log(f"    {name} 에폭 {ep}: train loss {rec['train_loss']:.5f}, 조기종료 val AP {ap:.5f}, "
                     f"{rec['seconds']:.1f}s (학습 {tfit:.1f}s, 검증 {tval:.1f}s, {nb} 배치)")
            if ap > best_ap:
                best_ap, best_ep, wait = ap, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.fit_seconds = time.time() - t0
        self.best_rounds = best_ep
        self.best_val_ap = best_ap
        self.n_rounds_trained = len(self.history)
        self.gpu_train_peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        self.gpu_train_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        self.padding_rows_zero = bool(all(float(e.weight[0].detach().abs().max()) == 0.0 for e in self.net.embs))
        torch.cuda.reset_peak_memory_stats()
        self._t_eval0 = time.time()

    def predict(self, sp, h):
        n = self.T[sp]["s"].shape[0]
        h_t = torch.full((n,), int(h), dtype=torch.int64, device=self.dev)
        return self._predict_tensor(sp, h_t)

    def info(self):
        t = DLC["trf"]
        if self.arch == "lstm":
            enc = (f"LSTM({N_CH}->{DLC['lstm_hidden']}, {DLC['lstm_layers']}층, 단방향), 마지막 은닉"
                   + (" = 1차 models.make_lstm_net" if self.mode == "A" else ""))
            head = f"Linear(->{C1.LSTM_HEAD}) ReLU Dropout({C1.LSTM_DROPOUT}) Linear(->{1 if self.mode == 'A' else E.H_MAX})"
        else:
            enc = (f"Linear({N_CH}->{t['d_model']}) + 학습 위치 임베딩 nn.Embedding({t['pos_emb']},{t['d_model']}) 기본 초기값 N(0,1) -> "
                   f"TransformerEncoder({t['layers']}층, {t['heads']}헤드, ff {t['ff']}, dropout {t['dropout']}, post-norm, ReLU) -> 시간 평균 풀링")
            head = f"Linear(->{t['head_hidden']}) ReLU Linear(->{1 if self.mode == 'A' else E.H_MAX}) (머리 dropout 없음)"
        hist = getattr(self, "history", None) or []
        ep_sec = [r["train_seconds"] for r in hist]
        return {"arch": self.arch, "mode": self.mode, "L": self.L, "encoder": enc, "head": head,
                "n_params": getattr(self, "n_params", None),
                "input_channels": "19 변수 표준화(학습 기간 <= 2024-06-30 기상표 평균·표준편차, NaN 무시, ddof=0) 후 NaN->0 + 결측 표시 19 = 38",
                "standardize_mean": [float(x) for x in self.mu], "standardize_std": [float(x) for x in self.sd],
                "embedding_dim_each": DLC["emb_dim"], "embedding_sizes_train_vocab_plus0": self.cat_sizes,
                "embedding_padding_rows_zero_after_fit": getattr(self, "padding_rows_zero", None),
                "h_input": "h/60" if self.mode == "A" else "없음(출력 60칸 중 h 번째 사용)",
                "loss": "BCEWithLogits (pos_weight 없음)" + ("" if self.mode == "A" else ", 관측 칸만 평균(배치 안 관측 칸 수로 나눔)"),
                "optimizer": f"Adam lr={DLC['lr']}", "batch": self.batch, "eval_batch": self.eval_batch,
                "max_epochs": self.max_epochs, "patience": self.patience,
                "early_stopping": "조기 종료용 검증(h = default_rng(seed+1)) sklearn average_precision, 최고 에폭 가중치 복원"
                                  + ("" if self.mode == "A" else ", 출력[h]"),
                "seed": self.seed, "shuffle": f"에폭마다 torch.randperm(Generator seed {self.seed})",
                "init": f"torch.manual_seed({self.seed}), cuda.manual_seed_all({self.seed}), cudnn.benchmark=False",
                "device": "cuda", "gpu_name": torch.cuda.get_device_name(0),
                "gpu_train_peak_allocated_gb": getattr(self, "gpu_train_peak_alloc_gb", None),
                "gpu_train_peak_reserved_gb": getattr(self, "gpu_train_peak_reserved_gb", None),
                "gpu_eval_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "seconds_per_epoch_train_mean": float(np.mean(ep_sec)) if ep_sec else None,
                "eval_seconds_total": (time.time() - self._t_eval0) if hasattr(self, "_t_eval0") else None,
                "labels_B": None if self.mode == "A" else {
                    "rule": "학습 표본 (단위, d), T = d - h(시드 h): y[j] = (단위, T+j) 단위x날짜 행 y, 행 없음·T+j > 2024-06-30 은 빈칸(마스크)",
                    **(self.mask_info or {})},
                "history": hist}

    def close(self):
        for k in ("Wg", "T", "net"):
            if hasattr(self, k):
                delattr(self, k)
        torch.cuda.empty_cache()


def _factory(arch, mode):
    def make(ctx, exp, seed, log, smoke):
        return DLRunner(ctx, exp, seed, log, smoke, arch, mode)
    return make


for _id, _a, _m in (("DL_lstmA", "lstm", "A"), ("DL_trfA", "trf", "A"), ("DL_lstmB", "lstm", "B"), ("DL_trfB", "trf", "B")):
    register_dl(_id)(_factory(_a, _m))
