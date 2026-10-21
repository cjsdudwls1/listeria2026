# -*- coding: utf-8 -*-
"""모델 4종. 튜닝 없음, 불균형 처리 없음(가중치·샘플링·pos_weight 없음), 시드 42.
트리 3종: 라이브러리 기본 하이퍼파라미터, 최대 2,000라운드, 조기 종료 = 조기 종료용 val 의
sklearn average_precision(PR-AUC), 인내 100라운드.

트리 fit(Xtr, ytr, make_xva, yva, ...) 의 make_xva 는 조기 종료용 val 행렬을 만드는 함수다(배열도 받음).
학습 자료 구조(DMatrix/Dataset/Pool)를 다 만든 뒤에 불러서 두 행렬이 메모리에 동시에 있는 시간을 줄인다
(예측 결과는 같음 - selftest (e1) 에서 확인).
fit_seconds = fit 안에서 걸린 시간 - 조기 종료용 val 특성 생성 시간
            (= DMatrix/Dataset/Pool 만들기 + 학습. 특성 생성·평가는 제외).
val_ap_history = 라운드(에폭)별 조기 종료용 val AP 기록(config.json 의 model_info.val_ap_by_round).
"""
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score

import config as C


def _make_val(make_xva):
    """조기 종료용 val 행렬을 만들고 (행렬, 걸린 초) 를 돌려준다. 배열이 들어오면 그대로."""
    t = time.time()
    X = make_xva() if callable(make_xva) else make_xva
    return X, time.time() - t


# ---------------------------------------------------------------- XGBoost
class XGBModel:
    with_cats = True        # 행렬 앞 4칸 = 범주 코드

    def __init__(self, log):
        self.log = log
        # disable_default_eval_metric = 기본 logloss 평가 끔(조기 종료는 AP 만). 학습 하이퍼파라미터 아님.
        self.params = {"objective": "binary:logistic", "device": "cuda", "seed": C.SEED,
                       "disable_default_eval_metric": 1}

    def fit(self, Xtr, ytr, make_xva, yva, cols, cats_tr=None, cats_va=None):
        import xgboost as xgb
        t_enter = time.time()
        if not xgb.build_info().get("USE_CUDA"):
            raise RuntimeError("xgboost 가 CUDA 없이 빌드됨 (GPU 사용 불가)")
        self.cols = cols
        self.ft = ["c"] * len(C.CAT_COLS) + ["q"] * (len(cols) - len(C.CAT_COLS))
        dtr = xgb.QuantileDMatrix(Xtr, label=ytr, feature_names=cols, feature_types=self.ft, enable_categorical=True)
        Xva, t_xva = _make_val(make_xva)
        dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr, feature_names=cols, feature_types=self.ft,
                                  enable_categorical=True)
        del Xva
        yv = np.asarray(yva)
        log = self.log

        def ap(predt, dmat):
            return "ap", float(average_precision_score(yv, predt))

        class Prog(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if (epoch + 1) % 200 == 0:
                    v = evals_log["val"]["ap"]
                    log(f"    xgboost 라운드 {epoch + 1}: val AP {v[-1]:.5f} (최고 {max(v):.5f})")
                return False

        es = xgb.callback.EarlyStopping(rounds=C.PATIENCE, metric_name="ap", data_name="val",
                                        maximize=True, save_best=False)
        evr = {}
        self.bst = xgb.train(self.params, dtr, num_boost_round=C.MAX_ROUNDS, evals=[(dva, "val")],
                             custom_metric=ap, callbacks=[es, Prog()], verbose_eval=False, evals_result=evr)
        self.fit_seconds = time.time() - t_enter - t_xva
        cfg = json.loads(self.bst.save_config())
        dev = cfg["learner"]["generic_param"].get("device", "")
        upd = [u.get("name") for u in cfg["learner"]["gradient_booster"].get("updater", [])]
        if not str(dev).startswith("cuda") or "grow_gpu_hist" not in upd:
            raise RuntimeError(f"xgboost 가 GPU 로 학습되지 않음: device={dev}, updater={upd}")
        self.best_rounds = int(self.bst.best_iteration) + 1
        self.best_val_ap = float(self.bst.best_score)
        self.n_rounds_trained = int(self.bst.num_boosted_rounds())
        self.val_ap_history = [float(x) for x in evr["val"]["ap"]]
        if self.best_val_ap != self.val_ap_history[self.best_rounds - 1]:
            raise RuntimeError("xgboost best_score 와 라운드별 기록이 다름")
        del dtr, dva

    def predict(self, X, cats_str=None):
        import xgboost as xgb
        dm = xgb.DMatrix(X, feature_names=self.cols, feature_types=self.ft, enable_categorical=True)
        return self.bst.predict(dm, iteration_range=(0, self.best_rounds)).astype(np.float32)

    def info(self):
        return {"library_params": self.params, "max_rounds": C.MAX_ROUNDS, "patience": C.PATIENCE,
                "early_stopping": "xgboost EarlyStopping(maximize) on custom_metric = sklearn average_precision(val)",
                "categorical": "QuantileDMatrix(enable_categorical=True, feature_types c x4), max_bin 기본 256",
                "device": "cuda (GPU 미사용 시 오류)",
                "val_ap_by_round": getattr(self, "val_ap_history", None)}


# ---------------------------------------------------------------- LightGBM
class LGBModel:
    with_cats = True

    def __init__(self, log):
        self.log = log
        # objective 는 기본값이 회귀라 binary 지정 필수. metric None = 기본 logloss 평가 끔(조기 종료는 AP 만).
        self.params = {"objective": "binary", "seed": C.SEED, "verbose": -1, "metric": "None"}

    def fit(self, Xtr, ytr, make_xva, yva, cols, cats_tr=None, cats_va=None):
        import lightgbm as lgb
        t_enter = time.time()
        self.cols = cols
        cat_idx = list(range(len(C.CAT_COLS)))
        # lgb.train 에 넘기는 것과 같은 params 를 Dataset 에도 넘겨(구간 경계 표본의 seed 가 같게) 미리 construct 한 뒤
        # 조기 종료용 val 행렬을 만든다. 예전(지연 construct)과 예측이 같음 - selftest (e1).
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=cols, categorical_feature=cat_idx, free_raw_data=True,
                          params=dict(self.params))
        dtr.construct()
        Xva, t_xva = _make_val(make_xva)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr, feature_name=cols, categorical_feature=cat_idx)
        yv = np.asarray(yva)
        log = self.log

        def ap(preds, ds):
            return "ap", float(average_precision_score(yv, preds)), True

        def prog(env):
            if (env.iteration + 1) % 200 == 0 and env.evaluation_result_list:
                log(f"    lightgbm 라운드 {env.iteration + 1}: val AP {env.evaluation_result_list[0][2]:.5f}")

        rec = {}
        self.bst = lgb.train(self.params, dtr, num_boost_round=C.MAX_ROUNDS, valid_sets=[dva], valid_names=["val"],
                             feval=ap, callbacks=[lgb.early_stopping(C.PATIENCE, verbose=False), prog,
                                                  lgb.record_evaluation(rec)])
        self.fit_seconds = time.time() - t_enter - t_xva
        self.best_rounds = int(self.bst.best_iteration)
        self.best_val_ap = float(self.bst.best_score["val"]["ap"])
        self.val_ap_history = [float(x) for x in rec["val"]["ap"]]
        self.n_rounds_trained = len(self.val_ap_history)   # 학습 끝난 모델은 최적 라운드까지만 남으므로 기록 길이로 셈
        if self.best_val_ap != self.val_ap_history[self.best_rounds - 1]:
            raise RuntimeError("lightgbm best_score 와 라운드별 기록이 다름")
        del dtr, dva, Xva

    def predict(self, X, cats_str=None):
        return self.bst.predict(X, num_iteration=self.best_rounds).astype(np.float32)

    def info(self):
        return {"library_params": self.params, "max_rounds": C.MAX_ROUNDS, "patience": C.PATIENCE,
                "early_stopping": "lgb.early_stopping on feval = sklearn average_precision(val)",
                "categorical": "Dataset(categorical_feature=[0,1,2,3])", "device": "CPU",
                "dataset": "학습 Dataset 은 학습 params 를 그대로 넘겨 미리 construct (지연 construct 와 결과 같음)",
                "val_ap_by_round": getattr(self, "val_ap_history", None)}


# ---------------------------------------------------------------- CatBoost
class _APMetric:
    """CatBoost 사용자 평가지표: sklearn average_precision(val). 내장 PRAUC 는 쓰지 않음."""

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        return float(average_precision_score(np.asarray(target), np.asarray(approxes[0]))), 1.0

    def get_final_error(self, error, weight):
        return error


class CatModel:
    with_cats = False       # 숫자 행렬 [h + 기상] + 범주 문자열 4칸 따로

    def __init__(self, log):
        self.log = log
        self.params = {"iterations": C.MAX_ROUNDS, "random_seed": C.SEED, "od_type": "Iter", "od_wait": C.PATIENCE,
                       "use_best_model": True, "task_type": "CPU", "verbose": 0, "allow_writing_files": False}

    def _pool(self, X, cats_str, y=None):
        from catboost import FeaturesData, Pool
        fd = FeaturesData(num_feature_data=X, cat_feature_data=cats_str,
                          num_feature_names=self.num_names, cat_feature_names=list(C.CAT_COLS))
        return Pool(fd, label=y)

    def fit(self, Xtr, ytr, make_xva, yva, cols, cats_tr=None, cats_va=None):
        from catboost import CatBoostClassifier
        t_enter = time.time()
        self.num_names = cols
        ptr = self._pool(Xtr, cats_tr, ytr)
        Xva, t_xva = _make_val(make_xva)
        pva = self._pool(Xva, cats_va, yva)
        del Xva
        metric = _APMetric()
        self.metric_key = type(metric).__name__      # CatBoost 가 평가 기록에 쓰는 이름 = 클래스 이름
        self.model = CatBoostClassifier(eval_metric=metric, **self.params)
        self.model.fit(ptr, eval_set=pva)
        self.fit_seconds = time.time() - t_enter - t_xva
        self.best_rounds = int(self.model.get_best_iteration()) + 1
        # 사전 순서(첫 값)에 기대지 않고 사용자 지표 이름으로 고른다. 순서가 바뀌면 Logloss 가 들어갈 수 있었음.
        ev = self.model.get_evals_result().get("validation", {})
        if self.metric_key not in ev:
            raise RuntimeError(f"catboost 평가 기록에 사용자 지표 {self.metric_key} 가 없음: {list(ev)}")
        self.val_ap_history = [float(x) for x in ev[self.metric_key]]
        self.n_rounds_trained = len(self.val_ap_history)
        self.best_val_ap = self.val_ap_history[self.best_rounds - 1]
        if self.best_val_ap != max(self.val_ap_history):
            raise RuntimeError(f"catboost 최적 반복의 AP {self.best_val_ap} 가 기록 최대 {max(self.val_ap_history)} 와 다름")
        if self.model.tree_count_ != self.best_rounds:
            raise RuntimeError(f"catboost tree_count {self.model.tree_count_} != best {self.best_rounds}")
        del ptr, pva

    def predict(self, X, cats_str=None):
        return self.model.predict(self._pool(X, cats_str), prediction_type="Probability")[:, 1].astype(np.float32)

    def info(self):
        return {"library_params": self.params, "eval_metric": "사용자 지표 = sklearn average_precision(val, raw score)",
                "early_stopping": "CatBoost od_type=Iter, od_wait=100, use_best_model=True 를 사용자 지표(AP)에 적용",
                "categorical": "FeaturesData(cat_feature_data = 범주 문자열 4칸)",
                "eval_metric_key": getattr(self, "metric_key", None),
                "best_val_ap_source": "get_evals_result()['validation'][사용자 지표 이름][최적 반복] (= 기록 최대값 확인)",
                "val_ap_by_round": getattr(self, "val_ap_history", None)}


TREE_CLASSES = {"xgboost": XGBModel, "lightgbm": LGBModel, "catboost": CatModel}


# ---------------------------------------------------------------- LSTM
def make_lstm_net(n_in, cat_sizes):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_in, C.LSTM_HIDDEN, num_layers=C.LSTM_LAYERS, batch_first=True)
            self.embs = nn.ModuleList([nn.Embedding(n + 1, C.LSTM_EMB_DIM, padding_idx=0) for n in cat_sizes])
            d = C.LSTM_HIDDEN + C.LSTM_EMB_DIM * len(cat_sizes) + 1
            self.head = nn.Sequential(nn.Linear(d, C.LSTM_HEAD), nn.ReLU(), nn.Dropout(C.LSTM_DROPOUT),
                                      nn.Linear(C.LSTM_HEAD, 1))

        def forward(self, x, cats, hn):
            _, (h_last, _) = self.lstm(x)
            z = [h_last[-1]] + [e(cats[:, j]) for j, e in enumerate(self.embs)] + [hn[:, None]]
            return self.head(torch.cat(z, dim=1)).squeeze(1)

    return Net()


class LSTMRunner:
    """창은 배치마다 GPU 위 W(표준화 19 + 결측 표시 19 = 38채널)에서 꺼낸다."""

    def __init__(self, ctx, L, log):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 사용 불가 - LSTM 은 GPU 필수")
        self.torch = torch
        self.dev = torch.device("cuda")
        self.L, self.log, self.ctx = L, log, ctx
        W = ctx.weather.W
        mu, sd = ctx.weather.train_period_stats()
        if not (np.all(np.isfinite(mu)) and np.all(sd > 0)):
            raise RuntimeError("표준화 통계 이상")
        self.mu, self.sd = mu, sd
        miss = np.isnan(W)
        Wn = np.where(miss, 0.0, (W - mu) / sd).astype(np.float32)
        seq = np.concatenate([Wn, miss.astype(np.float32)], axis=2)
        self.Wg = torch.from_numpy(np.ascontiguousarray(seq)).to(self.dev)
        del Wn, seq, miss
        self.ar = torch.arange(L, device=self.dev)
        self.T = {}
        for sp in C.SPLITS:
            r = ctx.sp[sp]
            self.T[sp] = {
                "s": torch.from_numpy(r["s"]).to(self.dev),
                "d": torch.from_numpy(r["d"]).to(self.dev),
                "c": torch.from_numpy(r["lstm_cats"]).to(self.dev),
                "y": torch.from_numpy(r["y"].astype(np.float32)).to(self.dev),
                "hf": torch.from_numpy(r["h_fit"].astype(np.int64)).to(self.dev),
            }
        self.cat_sizes = ctx.lstm_cat_sizes
        self.n_in = 2 * W.shape[2]

    def _x(self, s, t):
        idx = t[:, None] - self.L + self.ar[None, :]
        return self.Wg[s[:, None], idx]

    def _predict_tensor(self, sp, h_t, batch):
        torch = self.torch
        T = self.T[sp]
        n = T["s"].shape[0]
        out = torch.empty(n, device=self.dev)
        self.net.eval()
        with torch.no_grad():
            for a in range(0, n, batch):
                b = slice(a, min(n, a + batch))
                hh = h_t[b]
                out[b] = torch.sigmoid(self.net(self._x(T["s"][b], T["d"][b] - hh), T["c"][b], hh.float() / C.H_MAX))
        return out.cpu().numpy().astype(np.float32)

    def fit(self, max_epochs=None, max_batches=None):
        torch = self.torch
        torch.manual_seed(C.SEED)
        torch.cuda.manual_seed_all(C.SEED)
        torch.backends.cudnn.benchmark = False
        self.net = make_lstm_net(self.n_in, self.cat_sizes).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=C.LSTM_LR)
        lossf = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(C.SEED)
        tr = self.T["train"]
        n = tr["s"].shape[0]
        yv = self.ctx.sp["val"]["y"]
        best_ap, best_state, best_ep, wait = -1.0, None, 0, 0
        self.history = []
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        for ep in range(1, (max_epochs or C.LSTM_MAX_EPOCHS) + 1):
            te = time.time()
            self.net.train()
            perm = torch.randperm(n, generator=g).to(self.dev)
            tot, nb = 0.0, 0
            for a in range(0, n, C.LSTM_BATCH):
                if max_batches is not None and nb >= max_batches:
                    break
                b = perm[a:a + C.LSTM_BATCH]
                hh = tr["hf"][b]
                logit = self.net(self._x(tr["s"][b], tr["d"][b] - hh), tr["c"][b], hh.float() / C.H_MAX)
                loss = lossf(logit, tr["y"][b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                nb += 1
            tfit = time.time() - te
            p = self._predict_tensor("val", self.T["val"]["hf"], C.LSTM_EVAL_BATCH)
            ap = float(average_precision_score(yv, p))
            self.history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_ap": ap,
                                 "train_seconds": tfit, "seconds": time.time() - te, "batches": nb})
            self.log(f"    lstm 에폭 {ep}: train loss {tot / max(nb, 1):.5f}, val AP {ap:.5f}, "
                     f"{time.time() - te:.1f}s (학습 {tfit:.1f}s, {nb} 배치)")
            if ap > best_ap:
                best_ap, best_ep, wait = ap, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= C.LSTM_PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.fit_seconds = time.time() - t0
        self.best_rounds = best_ep
        self.best_val_ap = best_ap
        self.n_rounds_trained = len(self.history)
        self.gpu_peak_gb = torch.cuda.max_memory_allocated() / 1e9

    def predict(self, sp, h):
        n = self.T[sp]["s"].shape[0]
        h_t = self.torch.full((n,), int(h), dtype=self.torch.int64, device=self.dev)
        return self._predict_tensor(sp, h_t, C.LSTM_EVAL_BATCH)

    def close(self):
        del self.Wg, self.T
        if hasattr(self, "net"):
            del self.net
        self.torch.cuda.empty_cache()

    def info(self):
        return {"input_channels": "19 변수 표준화(학습 기간 <= 2024-06-30 기상표 평균·표준편차, NaN 무시, ddof=0) 후 NaN->0 + 결측 표시 19 = 38",
                "standardize_mean": [float(x) for x in self.mu], "standardize_std": [float(x) for x in self.sd],
                "lstm_hidden": C.LSTM_HIDDEN, "lstm_layers": C.LSTM_LAYERS, "bidirectional": False,
                "embedding_dim_each": C.LSTM_EMB_DIM, "embedding_sizes_train_vocab_plus0": self.cat_sizes,
                "head": f"Linear({C.LSTM_HIDDEN}+{C.LSTM_EMB_DIM}x4+1 -> {C.LSTM_HEAD}) ReLU Dropout({C.LSTM_DROPOUT}) Linear(->1)",
                "h_input": "h/60", "loss": "BCEWithLogits (pos_weight 없음)", "optimizer": f"Adam lr={C.LSTM_LR}",
                "batch": C.LSTM_BATCH, "max_epochs": C.LSTM_MAX_EPOCHS, "patience": C.LSTM_PATIENCE,
                "early_stopping": "조기 종료용 val sklearn average_precision, 최고 에폭 가중치 복원",
                "shuffle": "에폭마다 torch.randperm(Generator seed 42)", "device": "cuda",
                "history": getattr(self, "history", None)}
