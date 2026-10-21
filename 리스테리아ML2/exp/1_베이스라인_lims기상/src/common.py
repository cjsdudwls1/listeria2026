# -*- coding: utf-8 -*-
"""공통: 로그, 메모리 측정, 실행 문맥(데이터·기상 배열), 실행기(트리/LSTM)."""
import datetime as _dt
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

import config as C
import data as D
import features as F
import models as MD


class Logger:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"[{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except Exception:
            pass


class Tee:
    """stdout/stderr 를 콘솔과 UTF-8 파일에 같이 쓴다(run_task2.ps1 의 logs/console.log)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        self.fh.flush()


def install_tee(path):
    fh = open(path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)


class MemSampler(threading.Thread):
    """실행 중 프로세스 메모리(RSS)와 GPU 사용량(nvidia-smi, 장치 전체) 최대값을 잰다. 측정용일 뿐 계산에 영향 없음."""

    def __init__(self, interval=0.5, gpu_every=4):
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.interval, self.gpu_every = interval, gpu_every
        self.peak_rss = 0
        self.peak_gpu_mb = None
        self._halt = threading.Event()

    def _gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            return float(out[0])
        except Exception:
            return None

    def run(self):
        k = 0
        while not self._halt.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            except Exception:
                pass
            if k % self.gpu_every == 0:
                g = self._gpu()
                if g is not None:
                    self.peak_gpu_mb = g if self.peak_gpu_mb is None else max(self.peak_gpu_mb, g)
            k += 1
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=15)
        try:
            self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
        except Exception:
            pass
        return {"peak_rss_gb": round(self.peak_rss / 1e9, 3),
                "peak_gpu_used_mb_device": self.peak_gpu_mb}


class Ctx:
    """데이터 문맥: 단위x날짜 행, 범주 코드, 기상 배열 W, 분할별 인덱스."""

    def __init__(self, smoke, log):
        t0 = time.time()
        self.smoke = smoke
        self.rows, self.levels = D.load_all(smoke)
        self.weather = D.Weather()
        r = self.rows
        codes = D.cat_codes(r, self.levels)
        lcodes, self.lstm_cat_sizes = D.lstm_vocab_codes(r)
        s = self.weather.s_idx(r["sgg_code"].to_numpy())
        d = self.weather.d_idx(r["d"].to_numpy())
        cat_str = np.stack([r[c].astype(str).to_numpy(dtype=object) for c in C.CAT_COLS], axis=1)
        self.sp = {}
        for sp in C.SPLITS:
            idx = np.flatnonzero((r["split"] == sp).to_numpy())
            self.sp[sp] = {"idx": idx, "s": s[idx], "d": d[idx], "y": r["y"].to_numpy()[idx].astype(np.int8),
                           "h_fit": r["h_fit"].to_numpy()[idx].astype(np.int16), "cats": codes[idx],
                           "lstm_cats": lcodes[idx], "cat_str": np.ascontiguousarray(cat_str[idx])}
        log(f"데이터 준비 {time.time() - t0:.1f}s: " + ", ".join(
            f"{sp} {len(v['idx']):,}행/양성 {int(v['y'].sum()):,}" for sp, v in self.sp.items())
            + f" | 기상 W {self.weather.W.shape} {self.weather.date0}~{self.weather.date_last}")

    def check_all_windows(self, L):
        """모든 행의 창(학습 h, 조기 종료 h, 평가 h=1..60)이 기상표 안인지 assert."""
        w = self.weather
        lo, hi = [], []
        for sp in C.SPLITS:
            v = self.sp[sp]
            ts = [v["d"] - v["h_fit"]] if sp == "train" else [v["d"] - v["h_fit"], v["d"] - C.H_MAX, v["d"] - 1]
            if sp == "test":
                ts = [v["d"] - C.H_MAX, v["d"] - 1]
            for t in ts:
                a, b = w.check_windows(t.astype(np.int64), L, f"{sp} L={L}")
                lo.append(a); hi.append(b)
        return str(w.date0 + np.timedelta64(min(lo), "D")), str(w.date0 + np.timedelta64(max(hi), "D"))


class TreeRunner:
    def __init__(self, ctx, model, inp, L, log):
        self.ctx, self.inp, self.L, self.log = ctx, inp, L, log
        self.m = MD.TREE_CLASSES[model](log)
        self.cols = F.tree_columns(inp, L, with_cats=self.m.with_cats)
        self.n_features = len(C.CAT_COLS) + 1 + F.n_weather_features(inp, L)
        self.n_features_desc = f"범주 4 + h 1 + 기상 {F.n_weather_features(inp, L)}"
        self.feature_seconds = 0.0

    def _X(self, sp, h=None):
        v = self.ctx.sp[sp]
        n = len(v["idx"])
        hh = v["h_fit"] if h is None else np.full(n, h, dtype=np.int16)
        t = v["d"] - hh.astype(np.int64)
        t0 = time.time()
        X = F.build_tree_matrix(self.ctx.weather.W, v["s"], t, hh, self.L, self.inp,
                                cats=v["cats"] if self.m.with_cats else None)
        self.feature_seconds += time.time() - t0
        return X

    def fit(self):
        t0 = time.time()
        Xtr = self._X("train")
        self.log(f"    학습 특성 {Xtr.shape} ({Xtr.nbytes / 1e9:.2f} GB), {time.time() - t0:.1f}s")
        tr, va = self.ctx.sp["train"], self.ctx.sp["val"]

        def make_xva():
            # 모델이 학습 자료 구조를 다 만든 뒤에 부른다(메모리 동시 사용 줄임, 결과 같음)
            t1 = time.time()
            Xva = self._X("val")
            self.log(f"    조기종료 val 특성 {Xva.shape}, {time.time() - t1:.1f}s")
            return Xva

        self.m.fit(Xtr, tr["y"], make_xva, va["y"], self.cols, cats_tr=tr["cat_str"], cats_va=va["cat_str"])
        del Xtr
        self.fit_seconds = self.m.fit_seconds
        self.best_rounds = self.m.best_rounds
        self.best_val_ap = self.m.best_val_ap
        self.n_rounds_trained = self.m.n_rounds_trained

    def predict(self, sp, h):
        X = self._X(sp, h)
        return self.m.predict(X, self.ctx.sp[sp]["cat_str"])

    def info(self):
        return self.m.info()

    def close(self):
        pass


class SeqRunner:
    def __init__(self, ctx, L, log):
        self.r = MD.LSTMRunner(ctx, L, log)
        self.n_features = 2 * len(C.WEATHER_VARS) * L + len(C.CAT_COLS) + 1
        self.n_features_desc = f"시퀀스 {L}일 x 38채널(표준화 19 + 결측 표시 19) + 범주 4(임베딩) + h/60"
        self.feature_seconds = 0.0

    def fit(self):
        self.r.fit()
        self.fit_seconds = self.r.fit_seconds
        self.best_rounds = self.r.best_rounds
        self.best_val_ap = self.r.best_val_ap
        self.n_rounds_trained = self.r.n_rounds_trained

    def predict(self, sp, h):
        return self.r.predict(sp, h)

    def info(self):
        d = self.r.info()
        d["gpu_peak_allocated_gb_torch"] = getattr(self.r, "gpu_peak_gb", None)
        return d

    def close(self):
        self.r.close()


def make_runner(ctx, model, inp, L, log):
    return SeqRunner(ctx, L, log) if model == "lstm" else TreeRunner(ctx, model, inp, L, log)
