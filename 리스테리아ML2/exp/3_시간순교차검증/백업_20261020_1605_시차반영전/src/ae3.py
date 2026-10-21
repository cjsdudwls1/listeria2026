# -*- coding: utf-8 -*-
"""이상치 점수 4개(AEt·VAEt·AEs·VAEs) - 겹마다 그 겹 학습 행으로 다시 학습 (2차 e2ae 의 점수 규칙 v2 '고친 판'을 겹 단위로).

2차 e2ae 의 부품(신경망·학습·점수·정규화·겹 배정 함수)을 그대로 불러 쓰고, 행 범위만 3차 겹으로 바꾼다(2차 코드는 고치지 않음).
범위(scope) 하나 = (학습 행 묶음, 점수 낼 행 묶음): 겹 k = (겹 k 학습 행, 겹 k 맞혀 보는 행), 최종 = (학습 자료 전체, 시험 행).
- 학습 행 점수: 학습 행 전부(검출·불검출)를 5겹에 무작위 배정(default_rng([seed, 3]))해 '그 행을 안 본 겹 모델' 점수.
  겹 모델 m = 자기 겹 밖 학습 행 중 불검출(y=0)만으로 학습·정규화(조기 종료용 10% 포함, default_rng([seed, 2, m])).
- 점수 낼 행(맞혀 보는 행·시험 행) = 겹 모델 5개 점수 평균(float64 로 더해 5 로 나눈 뒤 float32).
- 입력: 표(AEt·VAEt) = 그 실행의 숫자 칸 전부(이상치 점수 칸 제외, 첫 칸 h) 표준화 + 범주 칸 임베딩(그 모델 정상 행 어휘),
        시퀀스(AEs·VAEs) = 기상 창 T−L … T−1 (19변수 표준화 + 결측 표시 19 = 38채널, 정규화 = 그 모델 정상 행 창 겹침 가중).
- 학습 설정·구조·점수 정의·재현성 설정 = 2차 e2cfg.AE·e2ae 그대로(GPU 필수).
그 범위 밖 행(맞혀 보는 해·시험 해)의 정답은 학습·정규화·어휘에 들어가지 않는다.
캐시: <cache>/ae_<scope>_s<seed>_<kind>_<지문>.npz (+ .json).
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import cfg3 as G  # noqa: F401  (설정·감싸기 먼저)
import e2ae as A
import torch

KINDS = list(A.KINDS)
EVAL_CHUNK = 200000


def _md5(*arrs):
    m = hashlib.md5()
    for a in arrs:
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()


def scores(kind, seed, scope, rs_tr, X_tr, codes_tr, rs_ev, X_ev, codes_ev, weather, cache_dir, log):
    """돌려줌 (S_tr (n_tr,) float32, S_ev (n_ev,) float32, info). X_ev 는 메모리맵이어도 됨(조각으로 읽음)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = _md5(rs_tr.pos, rs_tr.h, rs_tr.y, rs_ev.pos, rs_ev.h, np.array([X_tr.shape[1], codes_tr.shape[1]]))[:12]
    f = cache_dir / f"ae_{scope}_s{seed}_{kind}_{sig}.npz"
    mf = f.with_suffix(".json")
    meta = {"kind": kind, "seed": int(seed), "scope": scope, "sig": sig, "n_tr": int(rs_tr.n), "n_ev": int(rs_ev.n),
            "n_num": int(X_tr.shape[1]), "n_cat": int(codes_tr.shape[1]), "L": int(A.AEC["seq"]["L"]), "config": A.AEC,
            "score_rule": "3차: 2차 v2 를 겹 범위로(학습 행 5겹 교차 적합, 점수 낼 행 = 겹 모델 5개 평균)", "torch": torch.__version__}
    if f.exists() and mf.exists():
        try:
            if json.loads(mf.read_text(encoding="utf-8"))["meta"] == json.loads(json.dumps(meta)):
                z = np.load(f)
                return z["S_tr"], z["S_ev"], json.loads(mf.read_text(encoding="utf-8"))["info"]
        except Exception:
            pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가 - 이상치 점수는 GPU 로 계산(2차와 같은 재현성 설정)")
    t0 = time.time()
    dev = torch.device("cuda")
    tab = A.is_tab(kind)
    y = rs_tr.y
    n_tr, n_ev = rs_tr.n, rs_ev.n
    fold = A.fold_assign_all(n_tr, seed)
    neg = np.flatnonzero(y == 0)
    S_tr = np.full(n_tr, np.nan, dtype=np.float32)
    acc = np.zeros(n_ev, dtype=np.float64)
    Lq = int(A.AEC["seq"]["L"])
    W = weather.W
    models = {}
    with A.deterministic_torch():
        for mi in range(A.AEC["n_folds"]):
            rows = neg[fold[neg] != mi]
            target = np.flatnonzero(fold == mi)
            assert np.all(y[rows] == 0) and not np.intersect1d(rows, target).size
            fit_rows, es_rows = A.es_split(rows, seed, mi)
            if tab:
                prep = A.TabInput.prep(X_tr[rows], codes_tr[rows])
                dd = {"Z": torch.from_numpy(A.TabInput.z(X_tr, prep)).to(dev),
                      "C": torch.from_numpy(A.TabInput.cat_idx(codes_tr.astype(np.int64), prep)).to(dev)}
                net_args = (X_tr.shape[1], [len(v) for v in prep["vocab"]])
                batch_fn = A.TabInput.batch
            else:
                mu, sd = A.seq_stats(W, rs_tr.s[rows], rs_tr.t[rows], Lq)
                tb = torch.from_numpy(A.seq_table(W, mu.astype(np.float32), sd.astype(np.float32))).to(dev)
                dd = {"tb": tb, "s": torch.from_numpy(rs_tr.s).to(dev), "T": torch.from_numpy(rs_tr.t).to(dev),
                      "ar": torch.arange(Lq, device=dev)}
                net_args = ()
                batch_fn = A.SeqInput.batch
            net, minfo, _ = A.fit_net(kind, net_args, batch_fn, dd, fit_rows, es_rows, seed * 10 + mi, dev, log,
                                      f"{scope} {kind} 겹 모델 {mi}")
            S_tr[target] = A.score_rows(net, batch_fn, dd, target, dev)
            del dd
            if tab:
                for a in range(0, n_ev, EVAL_CHUNK):
                    b = min(n_ev, a + EVAL_CHUNK)
                    de = {"Z": torch.from_numpy(A.TabInput.z(np.asarray(X_ev[a:b]), prep)).to(dev),
                          "C": torch.from_numpy(A.TabInput.cat_idx(codes_ev[a:b].astype(np.int64), prep)).to(dev)}
                    acc[a:b] += A.score_rows(net, batch_fn, de, np.arange(b - a), dev)
                    del de
            else:
                de = {"tb": tb, "s": torch.from_numpy(rs_ev.s).to(dev), "T": torch.from_numpy(rs_ev.t).to(dev),
                      "ar": torch.arange(Lq, device=dev)}
                acc += A.score_rows(net, batch_fn, de, np.arange(n_ev), dev)
                del de, tb
            minfo.pop("history", None)
            models[str(mi)] = {**minfo, "rows": int(len(rows)), "target_rows": int(len(target))}
            del net
            torch.cuda.empty_cache()
    S_ev = (acc / A.AEC["n_folds"]).astype(np.float32)
    if np.isnan(S_tr).any() or np.isnan(S_ev).any():
        raise RuntimeError(f"이상치 점수 {kind} {scope}: 빈 점수")
    info = {"kind": kind, "seed": int(seed), "scope": scope, "n_normal_train": int(len(neg)), "n_pos_train": int((y == 1).sum()),
            "train_d_max": str(rs_tr.d.max()), "models": models, "seconds": round(time.time() - t0, 1)}
    np.savez(f, S_tr=S_tr, S_ev=S_ev)
    mf.write_text(json.dumps({"meta": meta, "info": info}, ensure_ascii=False), encoding="utf-8")
    log(f"    이상치 점수 {kind} {scope}: 학습 정상 {len(neg):,}행, {info['seconds']:.0f}s")
    return S_tr, S_ev, info
