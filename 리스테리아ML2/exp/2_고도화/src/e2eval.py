# -*- coding: utf-8 -*-
"""평가: h 마다 PR-AUC·ROC-AUC, 판정 기준값(검증 60개 h 합친 예측의 F2 최대 확률값, 1차 metrics.f2_threshold 그대로)을
시험에 그대로 적용한 h 별 혼동행렬 지표, 요약(60일 평균·구간 가중), 예측 저장, 메모리 측정."""
import subprocess
import threading

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

import config as C1
import metrics as M1
import e2cfg as E

CM_KEYS = ("tp", "fn", "fp", "tn", "accuracy", "precision", "recall", "f1", "f2")


def weighted(v) -> float:
    """v[h-1] (h=1..60) -> 0.5 x 평균(H1~20) + 0.3 x 평균(H21~40) + 0.2 x 평균(H41~60)."""
    v = np.asarray(v, dtype=np.float64)
    assert len(v) == E.H_MAX
    return float(sum(w * np.mean(v[a - 1:b]) for a, b, w in E.H_BANDS))


def _both_classes(y):
    s = int(np.sum(y))
    return 0 < s < len(y)


def evaluate(preds: dict, ys: dict):
    """preds[sp] = (60, n) 확률, ys[sp] = (n,) 0/1. 돌려줌: (h 별 지표 DataFrame, 기준값 dict, 요약 dict)."""
    recs = []
    for sp in ("val", "test"):
        y = ys[sp]
        ok = _both_classes(y)
        for h in range(1, E.H_MAX + 1):
            p = preds[sp][h - 1]
            recs.append({"split": sp, "h": h, "n": len(y), "pos": int(y.sum()),
                         "pr_auc": M1.pr_auc(y, p) if ok else np.nan,
                         "roc_auc": M1.roc_auc(y, p) if ok else np.nan})
    yv_all = np.tile(ys["val"], E.H_MAX)
    if _both_classes(yv_all):
        thr = M1.f2_threshold(yv_all, preds["val"].reshape(-1))
    else:
        thr = {"threshold": np.nan, "val_f2_at_threshold": np.nan, "val_precision_at_threshold": np.nan,
               "val_recall_at_threshold": np.nan, "note": "검증에 검출 또는 불검출이 없어 기준값 없음"}
    for r in recs:
        cm = M1.confusion(ys[r["split"]], preds[r["split"]][r["h"] - 1], thr["threshold"])
        r.update({"threshold": thr["threshold"], **{k: cm[k] for k in CM_KEYS}})
    mdf = pd.DataFrame(recs)
    summ = summarize(mdf)
    summ["val_overall"] = M1.confusion(yv_all, preds["val"].reshape(-1), thr["threshold"])
    summ["test_overall"] = M1.confusion(np.tile(ys["test"], E.H_MAX), preds["test"].reshape(-1), thr["threshold"])
    return mdf, thr, summ


def summarize(mdf: pd.DataFrame) -> dict:
    """h 별 지표 표 -> 60일 평균(_mean60)·구간 가중(_w)·혼동행렬 h 당 평균(_mean_per_h)."""
    out = {}
    for sp in ("val", "test"):
        d = mdf[mdf["split"] == sp].sort_values("h")
        if len(d) != E.H_MAX or list(d["h"]) != list(range(1, E.H_MAX + 1)):
            raise ValueError(f"{sp}: h 1~60 이 다 있지 않음")
        for m in E.SUMMARY_METRICS:
            if m not in d.columns:
                continue
            v = d[m].to_numpy(dtype=np.float64)
            out[f"{sp}_{m}_mean60"] = float(np.mean(v))
            out[f"{sp}_{m}_w"] = weighted(v)
        for c in ("tp", "fn", "fp", "tn"):
            if c in d.columns:
                out[f"{sp}_{c}_mean_per_h"] = float(d[c].mean())
        out[f"{sp}_n_rows"] = int(d["n"].iloc[0])
        out[f"{sp}_n_pos"] = int(d["pos"].iloc[0])
    if "threshold" in mdf.columns:
        out["threshold"] = float(mdf["threshold"].iloc[0])
    return out


def write_preds(path, ctx, sp, pos, P):
    """1차와 같은 형식: 행 = (h, 대상 행) h 우선 순서. 열 sgg_code, train_prdlst_code, train_prdlst_code_2, d, h, y, p."""
    v = ctx.sp[sp]
    n = len(pos)
    rows = ctx.rows.iloc[v["idx"][pos]]
    cols = {}
    for name, cidx, lev in (("sgg_code", 1, "sgg"), ("train_prdlst_code", 2, "p2"), ("train_prdlst_code_2", 3, "p3")):
        codes = np.tile(v["cats"][pos, cidx], E.H_MAX).astype(np.int32)
        cols[name] = pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(ctx.levels[lev], type=pa.string()))
    cols["d"] = pa.array(np.tile(rows["d"].to_numpy().astype("datetime64[D]"), E.H_MAX), type=pa.date32())
    cols["h"] = pa.array(np.repeat(np.arange(1, E.H_MAX + 1, dtype=np.int8), n))
    cols["y"] = pa.array(np.tile(v["y"][pos].astype(np.int8), E.H_MAX))
    cols["p"] = pa.array(P.reshape(-1).astype(np.float32))
    pq.write_table(pa.table(cols), path, compression="zstd")


def read_preds(path):
    """저장 예측 -> (h 별 (n,) 배열을 쌓은 (60, n) p, (n,) y, 행 정보 DataFrame(첫 h 블록))."""
    t = pq.read_table(path).to_pandas()
    n = len(t) // E.H_MAX
    if n * E.H_MAX != len(t) or not (t["h"].to_numpy().reshape(E.H_MAX, n) == np.arange(1, E.H_MAX + 1)[:, None]).all():
        raise ValueError(f"{path}: h 우선 순서가 아님")
    p = t["p"].to_numpy().reshape(E.H_MAX, n)
    y = t["y"].to_numpy()[:n].astype(np.int8)
    info = t.iloc[:n][["sgg_code", "train_prdlst_code", "train_prdlst_code_2", "d"]].reset_index(drop=True)
    return p, y, info


# ---------------------------------------------------------------- 고르는 점수: 검증 T 통합 60일평균 PR-AUC (사용자 확정 원칙 2026-10-19, EXP2_SELECT=tint)
def tint_bounds():
    """검증 T 통합 행의 예측하는 날 T 범위: VAL_START ~ VAL_END − H_MAX (1차 config 에서 계산, 지금 2024-07-01 ~ 2025-05-01)."""
    vs = np.datetime64(C1.VAL_START, "D")
    return vs, np.datetime64(C1.VAL_END, "D") - np.timedelta64(E.H_MAX, "D")


_TS, _TE = tint_bounds()
TINT_RULE = (f"검증 T 통합 60일평균 PR-AUC: 저장 검증 예측(pred_val.parquet) 중 예측하는 날 T = d − h 가 {_TS} ~ {_TE} 인 행만"
             f"({_TE} = 검증 끝 {C1.VAL_END} − {E.H_MAX}일, 모든 h 같은 예측일 {int((_TE - _TS).astype(int)) + 1}일), h 마다 PR-AUC"
             f"(sklearn average_precision) -> h 1~{E.H_MAX} 단순 평균, 시드가 여러 개면 시드 평균. 시험 미사용")
_TINT_CACHE = {}


def val_tint_pr_auc(path) -> dict:
    """저장 검증 예측(실행 폴더 또는 pred_val.parquet 경로) -> {'mean60': h 1..60 PR-AUC 단순 평균, 'by_h': [..], 'n_by_h': [..], 'pos_by_h': [..]}.
    행 = T = d − h 가 tint_bounds() 안인 행. h 에 검출·불검출이 다 없으면 그 h 는 NaN(평균도 NaN). 파일 (경로, 크기, 수정 시각)으로 메모리 캐시."""
    from pathlib import Path
    path = Path(path)
    if path.is_dir():
        path = path / "pred_val.parquet"
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key in _TINT_CACHE:
        return _TINT_CACHE[key]
    t = pq.read_table(path, columns=["d", "h", "y", "p"])
    d = t.column("d").to_numpy().astype("datetime64[D]")
    h = t.column("h").to_numpy().astype(np.int64)
    y = t.column("y").to_numpy().astype(np.int8)
    p = t.column("p").to_numpy()
    vs, ve = tint_bounds()
    T = d - h.astype("timedelta64[D]")
    keep = (T >= vs) & (T <= ve)
    by_h, n_by_h, pos_by_h = [], [], []
    for hh in range(1, E.H_MAX + 1):
        m = keep & (h == hh)
        yy, pp = y[m], p[m]
        n_by_h.append(int(m.sum()))
        pos_by_h.append(int(yy.sum()))
        by_h.append(M1.pr_auc(yy, pp) if _both_classes(yy) else np.nan)
    out = {"mean60": float(np.mean(by_h)), "by_h": [float(x) for x in by_h], "n_by_h": n_by_h, "pos_by_h": pos_by_h,
           "T_range": [str(vs), str(ve)], "file": path.name}
    _TINT_CACHE[key] = out
    return out


class Mem(threading.Thread):
    """프로세스 메모리 최대값: rss = 작업 집합(Working Set, 메모리맵 파일 페이지 포함), private = 커밋된 개인 메모리
    (Windows Private Bytes, 파일 기반 페이지 제외). GPU 는 nvidia-smi 장치 전체 사용량."""

    def __init__(self, interval=0.5, gpu_every=6):
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.interval, self.gpu_every = interval, gpu_every
        self.peak_rss = self.peak_private = 0
        self.peak_gpu = None
        self.min_avail = None                 # 시스템 사용 가능 메모리 최소값
        self.marks = {}                       # 이름 -> 그 시점부터의 최대값(단계별 측정)
        self._halt = threading.Event()

    def _read(self):
        mi = self.proc.memory_info()
        return mi.rss, getattr(mi, "private", getattr(mi, "vms", 0))

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
                r, p = self._read()
                self._update(r, p)
            except Exception:
                pass
            if self.gpu_every and k % self.gpu_every == 0:
                g = self._gpu()
                if g is not None:
                    self.peak_gpu = g if self.peak_gpu is None else max(self.peak_gpu, g)
            k += 1
            self._halt.wait(self.interval)

    def _update(self, r, p):
        self.peak_rss, self.peak_private = max(self.peak_rss, r), max(self.peak_private, p)
        a = psutil.virtual_memory().available
        self.min_avail = a if self.min_avail is None else min(self.min_avail, a)
        for m in self.marks.values():
            m["peak_rss"], m["peak_private"] = max(m["peak_rss"], r), max(m["peak_private"], p)
            m["min_avail"] = min(m["min_avail"], a)

    def mark(self, name):
        """이 시점부터의 최대값을 따로 잰다(같은 이름이면 새로 시작)."""
        r, p = self._read()
        a = psutil.virtual_memory().available
        self.marks[name] = {"start_rss": r, "start_private": p, "peak_rss": r, "peak_private": p, "min_avail": a}

    def mark_result(self, name):
        r, p = self._read()
        self._update(r, p)
        m = self.marks.pop(name)
        g = lambda x: round(x / 1e9, 3)
        return {"start_rss_gb": g(m["start_rss"]), "start_private_gb": g(m["start_private"]),
                "peak_rss_gb": g(m["peak_rss"]), "peak_private_gb": g(m["peak_private"]),
                "end_rss_gb": g(r), "end_private_gb": g(p), "system_min_available_gb": g(m["min_avail"])}

    def snapshot(self):
        r, p = self._read()
        self._update(r, p)
        return {"rss_gb": round(r / 1e9, 3), "private_gb": round(p / 1e9, 3),
                "peak_rss_gb": round(self.peak_rss / 1e9, 3), "peak_private_gb": round(self.peak_private / 1e9, 3)}

    def reset(self):
        r, p = self._read()
        self.peak_rss, self.peak_private = r, p
        self.peak_gpu = None
        self.min_avail = psutil.virtual_memory().available

    def stop(self):
        self._halt.set()
        self.join(timeout=15)
        return self.result()

    def result(self):
        try:
            self.snapshot()
        except Exception:
            pass
        return {"peak_rss_gb": round(self.peak_rss / 1e9, 3), "peak_private_gb": round(self.peak_private / 1e9, 3),
                "peak_gpu_used_mb_device": self.peak_gpu,
                "system_min_available_gb": round(self.min_avail / 1e9, 3) if self.min_avail is not None else None}
