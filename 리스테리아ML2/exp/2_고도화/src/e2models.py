# -*- coding: utf-8 -*-
"""2차 모델 실행기.

CatRunner : CatBoost (E00, A, C, D, H10, H30(H60 대체)). 1차 models.CatModel 과 같은 설정·같은 Pool 구성에 시드와
            scale_pos_weight 만 바꿀 수 있게 복사(1차 파일은 고치지 않음). 사용자 지표 = 1차 _APMetric(sklearn AP).
LRRunner  : 로지스틱 회귀 (B1, B2)
DL        : e2dl.py 의 DL_REGISTRY (실험 ID -> 실행기 생성 함수). 지금은 비어 있음.
AE        : e2ae.AEScoreRunner (점수 단독 S_*, 사용자 지시 2026-10-19). A3 + 점수 칸(A3_AE*·_p)은 CatRunner + feats['ae'].

실행기 공통 속성/메서드: fit(), predict(sp, h) -> (그 분할의 대상 행 수,) float32, info(), close(),
  best_rounds, n_rounds_trained, best_val_ap, fit_seconds, feature_seconds, n_train_rows, n_features, n_features_desc, cols
"""
import os
import time

import numpy as np

import config as C1
import features as F
import models as MD1
import e2cfg as E
from e2hist import hist_features, hist_names, woe_features, woe_names
from e2smote import smotenc


def num_feature_names(feats):
    names = ["h"]
    if feats["weather"]:
        names += F.feature_names("summary", E.L)
    names += hist_names(feats["hist"])
    if feats["woe"]:
        names += woe_names()
    if feats.get("ext"):                                   # F·G (e2fg.py) - 없으면 이전과 같음
        import e2fg
        names += e2fg.ext_names(feats["ext"])
    if feats.get("nb"):                                    # N 제품명 (e2nb.py, 사용자 승인 2026-10-19) - 없으면 이전과 같음
        import e2nb
        names += e2nb.nb_names(feats["nb"])
    if feats.get("ae"):                                    # AE·VAE 이상 점수 (e2ae.py, 사용자 지시 2026-10-19) - 없으면 이전과 같음
        import e2ae
        names += e2ae.ae_names(feats["ae"])
    return names


class FeatureBuilder:
    """한 실험의 숫자 입력 [h, (기상 요약 E.NW칸, L=50 이면 190), (이력), (WOE), (F·G·M), (N 제품명), (AE 점수)] 과 범주 문자열 4칸(+ F3 이면 aw_label)을 만든다.
    seed: AE 점수 칸(feats['ae'])에만 씀(점수가 시드마다 다름). 그 밖의 실험에는 영향 없음."""

    def __init__(self, ctx, feats, seed=None):
        self.ctx, self.feats, self.seed = ctx, feats, seed
        self.names = num_feature_names(feats)
        self.seconds = 0.0
        ext = feats.get("ext") or []
        if ext:
            import e2fg
            self._ext_cat = e2fg.ext_cat_names(ext)
        else:
            self._ext_cat = []
        self.cat_names = (list(C1.CAT_COLS) if feats["cats"] else []) + self._ext_cat

    def num(self, sp, pos, h):
        t0 = time.time()
        v = self.ctx.sp[sp]
        f = self.feats
        if f["weather"]:
            X0 = self.ctx.weather_block(sp, pos, h)
        else:
            X0 = np.empty((len(pos), 1), dtype=np.float32)
            X0[:, 0] = h
        parts = [X0]
        if f["hist"] or f["woe"]:
            grp = {lv: g[pos] for lv, g in v["grp"].items()}
            hd = v["hd"][pos]
            if f["hist"]:
                parts.append(hist_features(self.ctx.H, grp, hd, h, f["hist"], f["k"]))
            if f["woe"]:
                parts.append(woe_features(self.ctx.H, grp, hd, h))
        if f.get("ext"):
            import e2fg
            parts.append(e2fg.ext_features(self.ctx, sp, pos, h, f["ext"]))
        if f.get("nb"):
            import e2nb
            parts.append(e2nb.nb_features(self.ctx, sp, pos, h, f["nb"], f["k"]))
        if f.get("ae"):                                    # AE 점수(학습 행은 그 시드의 h 에서만 있음 - e2ae.ae_features 가 확인)
            import e2ae
            if f.get("ae_tab") is None:                    # 표 AE 입력 = A3(기존 실험)
                parts.append(e2ae.ae_features(self.ctx, sp, pos, h, f["ae"], self.seed, f["k"]))
            else:                                          # 표 AE 입력 = 주어진 입력(단계 선택 ae4: 그 시점 기준 모델, 사용자 결정 2026-10-19)
                parts.append(e2ae.ae_features(self.ctx, sp, pos, h, f["ae"], self.seed, f["k"], tab_feats=f["ae_tab"]))
        X = parts[0] if len(parts) == 1 else np.ascontiguousarray(np.hstack(parts), dtype=np.float32)
        self.seconds += time.time() - t0
        assert X.shape[1] == len(self.names)
        return X

    def cats(self, sp, pos):
        if not self._ext_cat:
            return self.ctx.sp[sp]["cat_str"][pos] if self.feats["cats"] else None
        import e2fg
        ex = e2fg.ext_cats(self.ctx, sp, pos, self.feats["ext"])
        if not self.feats["cats"]:
            return np.ascontiguousarray(ex)
        return np.ascontiguousarray(np.hstack([self.ctx.sp[sp]["cat_str"][pos], ex]))


def smote_cat_consistency(cats_new, cats_min, rows):
    """SMOTE 합성 행의 범주 조합 점검(기록용, '시도 안 이웃 + 기준 행 범주' 방식이면 아래 비율이 모두 1 = 100% 여야 함).
    cats_new (m, 4) 합성 행 범주 문자열, cats_min (n_min, 4) 학습 검출 행 범주 문자열, rows (m,) 합성 행의 기준 검출 행 번호."""
    j = {c: i for i, c in enumerate(C1.CAT_COLS)}
    cn = cats_new.astype(str)
    cm = cats_min.astype(str)
    sgg, sido = cn[:, j["sgg"]], cn[:, j["sido"]]
    seen_units = set(map(tuple, cm.tolist()))
    return {"n_synthetic": int(len(cn)),
            "sido_eq_sgg2_rate": float(np.mean([a == b[:2] for a, b in zip(sido, sgg)])),
            "same_as_base_rate": float(np.mean(np.all(cn == cm[rows], axis=1))),
            "in_train_pos_units_rate": float(np.mean([t in seen_units for t in map(tuple, cn.tolist())])),
            "n_distinct_units_synthetic": int(len(set(map(tuple, cn.tolist())))),
            "n_distinct_units_train_pos": int(len(seen_units)),
            "note": "sido_eq_sgg2_rate = sido 가 sgg 앞 2자리와 같은 비율, same_as_base_rate = 범주 4칸이 기준 검출 행과 모두 같은 비율"
                    "(= 실제 검출 단위와 같은 단위), in_train_pos_units_rate = (sido, sgg, p2, p3) 조합이 학습 검출 행에 있는 비율"}


# ---------------------------------------------------------------- CatBoost
class Cat2:
    """1차 models.CatModel 과 같은 학습(params·Pool·조기 종료·검사)에 시드와 추가 파라미터만 받는다."""

    def __init__(self, seed, log, max_rounds, patience, extra=None, mem=None):
        self.log = log
        self.mem = mem
        self.mem_train_phase = None
        self.params = {"iterations": max_rounds, "random_seed": seed, "od_type": "Iter", "od_wait": patience,
                       "use_best_model": True, "task_type": "CPU", "verbose": 0, "allow_writing_files": False}
        if extra:
            self.params.update(extra)

    def pool(self, X, cats_str, y=None):
        from catboost import FeaturesData, Pool
        if cats_str is None:
            fd = FeaturesData(num_feature_data=X, num_feature_names=self.num_names)
        else:
            fd = FeaturesData(num_feature_data=X, cat_feature_data=cats_str,
                              num_feature_names=self.num_names, cat_feature_names=list(self.cat_names))
        return Pool(fd, label=y)

    def fit(self, Xtr, ytr, make_val, yva, num_names, cats_tr=None, cat_names=None):
        """Xtr 가 catboost.Pool 이면 그대로 쓴다(H60). make_val() -> (Xva, cats_va).
        cat_names: 범주 칸 이름(없으면 1차 범주 4개 C1.CAT_COLS - 이전과 같음). F3 은 aw_label 이 5번째."""
        from catboost import CatBoostClassifier, Pool
        t_enter = time.time()
        self.num_names = num_names
        self.cat_names = list(cat_names) if cat_names is not None else list(C1.CAT_COLS)
        ptr = Xtr if isinstance(Xtr, Pool) else self.pool(Xtr, cats_tr, ytr)
        t = time.time()
        Xva, cats_va = make_val()
        t_xva = time.time() - t
        pva = self.pool(Xva, cats_va, yva)
        del Xva
        metric = MD1._APMetric()
        self.metric_key = type(metric).__name__
        self.model = CatBoostClassifier(eval_metric=metric, **self.params)
        if self.mem:
            self.mem.mark("train")
        self.model.fit(ptr, eval_set=pva)
        if self.mem:
            self.mem_train_phase = self.mem.mark_result("train")    # 학습 단계만(Pool 만든 뒤 ~ 학습 끝)
        self.fit_seconds = time.time() - t_enter - t_xva
        self.best_rounds = int(self.model.get_best_iteration()) + 1
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
        return self.model.predict(self.pool(X, cats_str), prediction_type="Probability")[:, 1].astype(np.float32)

    def info(self):
        return {"library_params": self.params, "eval_metric": "사용자 지표 = sklearn average_precision(조기 종료용 검증, raw score) - 1차 _APMetric",
                "early_stopping": f"CatBoost od_type=Iter, od_wait={self.params['od_wait']}, use_best_model=True",
                "eval_metric_key": getattr(self, "metric_key", None),
                "mem_train_phase": self.mem_train_phase,
                "val_ap_by_round": getattr(self, "val_ap_history", None)}


def build_h60_pool(ctx, fb, pos, seed, cache_dir, log, mem=None):
    """H60: 학습 행마다 h 1..60 전부."""
    Hm = np.tile(np.arange(1, E.H60_N + 1, dtype=np.int16), (len(pos), 1))
    return build_multi_h_pool(ctx, fb, pos, Hm, seed, cache_dir, log, mem)


def build_multi_h_pool(ctx, fb, pos, Hm, seed, cache_dir, log, mem=None):
    """학습 행 x h 여러 개(Hm: (n, k), 행마다 h k 개)의 학습 Pool. 디스크 파일(float32, 열 우선 F-order)에
    칸 순서로 쓰고(행 = [칸 1 의 모든 학습 행, 칸 2 ..., 칸 k], H60 이면 칸 j = h j) np.memmap 으로 열어
    FeaturesData 로 넘긴다(CatBoost 는 열 우선 float32 를 복사하지 않고 참조) -> Pool.quantize() 로 양자화.
    양자화 뒤 Pool.slice(전체 행) 로 양자화 자료만 가진 Pool 을 새로 만들고 원래 Pool·메모리맵을 놓은 뒤 파일을 지운다
    (Pool 이 메모리맵 참조를 쥐고 있어 그대로 두면 파일 페이지가 작업 집합에 남음. 전체 행 slice 로 학습해도 예측이
    같음 - scratch/slice_test.py, selftest (8)).
    돌려줌: (양자화된 pool, 정보 dict)"""
    from catboost import FeaturesData, Pool
    v = ctx.sp["train"]
    n, H = Hm.shape
    assert n == len(pos)
    N, ncol = n * H, len(fb.names)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_multih(cache_dir, log)
    path = cache_dir / f"multih{H}_train_{'smoke' if ctx.smoke else 'full'}_s{seed}_{os.getpid()}.f32"
    try:
        return _build_multi_h_pool_file(ctx, fb, pos, Hm, seed, log, mem, v, n, H, N, ncol, path)
    except BaseException as e:
        # 쓰기·양자화 중 오류나 중단(Ctrl+C 포함): 5GB 넘는 임시 파일을 남기지 않음. 메모리맵·Pool 참조는 안쪽 함수의
        # 지역 변수에만 있으므로, 예외의 traceback 이 쥐고 있는 끝난 프레임의 지역 변수를 비운 뒤 지운다.
        import gc
        import traceback
        traceback.clear_frames(e.__traceback__)
        gc.collect()
        try:
            if path.exists():
                os.remove(path)
                log(f"    여러 h 임시 파일 지움(오류로 중단): {path.name}")
        except Exception as e:
            log(f"    여러 h 임시 파일을 지우지 못함: {path.name} ({e})")
        raise


def cleanup_stale_multih(cache_dir, log):
    """죽은 프로세스가 남긴 multih*_<pid>.f32 를 지운다(살아 있는 pid 의 파일과 이 프로세스의 파일은 그대로)."""
    import psutil
    for f in cache_dir.glob("multih*_*.f32"):
        try:
            pid = int(f.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if pid == os.getpid() or psutil.pid_exists(pid):
            continue
        try:
            gb = f.stat().st_size / 1e9
            f.unlink()
            log(f"    이전 중단 실행이 남긴 여러 h 임시 파일 지움: {f.name} ({gb:.1f} GB, pid {pid} 없음)")
        except Exception as e:
            log(f"    이전 임시 파일을 지우지 못함: {f.name} ({e})")


def _build_multi_h_pool_file(ctx, fb, pos, Hm, seed, log, mem, v, n, H, N, ncol, path):
    from catboost import FeaturesData, Pool
    info = {"rows": N, "cols": ncol, "k": int(H), "memmap_file": str(path), "memmap_gb": N * ncol * 4 / 1e9,
            "layout": "행 = (칸 j, 학습 행) 칸 우선 순서: [칸 1 의 모든 학습 행, 칸 2 ..., 칸 k], 열 우선(F-order) float32"}
    t0 = time.time()
    if mem:
        mem.mark("h60_write")
    # 쓰기는 메모리맵이 아니라 보통 파일 쓰기(열 조각마다 seek+write): 메모리맵으로 쓰면 더러운 파일 페이지가
    # 프로세스 작업 집합에 쌓여 시스템 사용 가능 메모리가 0.1GB 까지 내려갔음(첫 시험). 파일 내용은 같음(F-order).
    with open(path, "wb") as f:
        f.truncate(N * ncol * 4)
        for c in range(H):
            bt = np.ascontiguousarray(fb.num("train", pos, Hm[:, c]).T)      # (ncol, n): 열 j 가 연속
            for j in range(ncol):
                f.seek((j * N + c * n) * 4)
                f.write(memoryview(bt[j]))
            if (c + 1) % 10 == 0:
                log(f"    여러 h 파일 쓰기 칸 {c + 1}/{H} ({time.time() - t0:.0f}s)")
    info["write_seconds"] = time.time() - t0
    info["write_method"] = "보통 파일 쓰기(열 우선 위치에 열 조각 seek+write), 읽기는 np.memmap(mode='r', order='F')"
    if mem:
        info["mem_write_phase"] = mem.mark_result("h60_write")
        mem.mark("h60_pool_quantize")
    X = np.memmap(path, dtype=np.float32, mode="r", shape=(N, ncol), order="F")
    y = np.tile(v["y"][pos], H)
    cats = np.tile(fb.cats("train", pos), (H, 1)) if fb.feats["cats"] else None
    t1 = time.time()
    if cats is None:
        fd = FeaturesData(num_feature_data=X, num_feature_names=fb.names)
    else:
        fd = FeaturesData(num_feature_data=X, cat_feature_data=cats, num_feature_names=fb.names,
                          cat_feature_names=list(C1.CAT_COLS))
    pool = Pool(fd, label=y)
    info["pool_seconds"] = time.time() - t1
    del cats, fd
    t2 = time.time()
    pool.quantize(random_seed=seed, task_type="CPU")
    info["quantize_seconds"] = time.time() - t2
    if mem:
        info["mem_pool_quantize_phase"] = mem.mark_result("h60_pool_quantize")
    info["quantize_params"] = {"random_seed": seed, "task_type": "CPU", "그 밖": "CatBoost 기본(border_count 254, GreedyLogSum, nan_mode Min)"}
    t3 = time.time()
    if mem:
        mem.mark("h60_slice_release")
    qpool = pool.slice(np.arange(N))
    del pool, X
    import gc
    gc.collect()
    try:
        os.remove(path)
        info["memmap_removed"] = True
    except Exception as e:
        info["memmap_removed"] = f"지우지 못함: {e}"
    info["slice_seconds"] = time.time() - t3
    if mem:
        info["mem_slice_release_phase"] = mem.mark_result("h60_slice_release")
    info["pool_shape"] = list(qpool.shape)
    log(f"    여러 h({H}) Pool: {N:,}행 x {ncol}열, 쓰기 {info['write_seconds']:.0f}s, Pool {info['pool_seconds']:.0f}s, "
        f"양자화 {info['quantize_seconds']:.0f}s, 원값 놓기 {info['slice_seconds']:.0f}s (파일 지움 {info['memmap_removed']})")
    return qpool, info


class CatRunner:
    def __init__(self, ctx, exp, seed, feats, log, rounds, patience, cache_dir=None, mem=None):
        self.ctx, self.exp, self.seed, self.feats, self.log = ctx, exp, seed, feats, log
        self.group = exp.get("group")
        self.pos = {sp: ctx.positions(sp, self.group) for sp in C1.SPLITS}
        self.fb = FeatureBuilder(ctx, feats, seed=seed)
        self.cols = self.fb.names
        self.rounds, self.patience = rounds, patience
        self.cache_dir, self.mem = cache_dir, mem
        self.extra_info = {}
        n_cat = len(self.fb.cat_names)
        self.n_features = len(self.cols) + n_cat
        self.n_features_desc = (f"범주 {n_cat}" if n_cat else "범주 없음") + f" + 숫자 {len(self.cols)}" + \
            f" (h 1 + 기상 {F.n_weather_features('summary', E.L) if feats['weather'] else 0}" + \
            f" + 이력 {len(hist_names(feats['hist']))} + WOE {4 if feats['woe'] else 0}"
        if feats.get("ext"):                                   # F·G·M 있을 때만 덧붙임(E00 등은 이전과 같은 글, F·G 만이면 '외부·생장식 N' 그대로)
            import e2fg
            self.n_features_desc += f" + {e2fg.ext_count_desc(feats['ext'])}"
            if self.fb._ext_cat:
                self.n_features_desc += f"; 범주에 {', '.join(self.fb._ext_cat)} 포함"
        if feats.get("nb"):                                    # N 제품명 있을 때만 덧붙임(이전 실험은 같은 글)
            import e2nb
            self.n_features_desc += f" + {e2nb.nb_count_desc(feats['nb'])}"
        if feats.get("ae"):                                    # AE 점수 있을 때만 덧붙임(이전 실험은 같은 글)
            import e2ae
            self.n_features_desc += f" + {e2ae.ae_count_desc(feats['ae'])}"
        self.n_features_desc += ")"

    @property
    def feature_seconds(self):
        return self.fb.seconds

    def fit(self):
        ctx, fb, seed = self.ctx, self.fb, self.seed
        hf = ctx.h_fit(seed)
        ptr_pos = self.pos["train"]
        tr = ctx.sp["train"]
        extra = {}
        mh = self.exp.get("multi_h")
        if mh and ctx.release_eval_cache():                     # 메모리용(결과와 무관): 평가 캐시 페이지를 내림
            self.log("    검증·시험 기상 요약 캐시 메모리맵을 놓음(평가 때 다시 엶)")
        t0 = time.time()
        if mh and mh >= E.MULTI_H_MEMMAP_MIN:
            # 지역 변수에 Pool·메모리맵을 따로 두지 않음(학습 뒤 del Xtr 로 참조가 다 풀려야 파일을 지울 수 있음)
            Hm = ctx.h_multi(seed, mh)[ptr_pos]
            Xtr, info = build_multi_h_pool(ctx, fb, ptr_pos, Hm, seed, self.cache_dir, self.log, self.mem)
            info["h_assignment"] = "1..60 전부" if mh == E.H_MAX else f"행마다 중복 없는 무작위 {mh}개(default_rng({seed}), 전체 학습 행 기준)"
            self.extra_info["h60"] = info
            ytr, cats_tr = None, None
            del Hm
            self.n_train_rows = info["rows"]
            self.n_train_pos = int(tr["y"][ptr_pos].sum()) * mh
        else:
            if mh:
                Hm = ctx.h_multi(seed, mh)[ptr_pos]                 # (n, mh)
                rep = np.repeat(ptr_pos, mh)
                hh = Hm.reshape(-1)
                self.extra_info["multi_h"] = {"k": mh, "order": "학습 행마다 h k 개를 붙여 놓음(행 우선)",
                                              "rows_distinct_ok": bool(all(len(set(r)) == mh for r in Hm[:1000]))}
            else:
                rep = ptr_pos
                hh = hf["train"][ptr_pos]
            Xtr = fb.num("train", rep, hh)
            ytr = tr["y"][rep]
            cats_tr = fb.cats("train", rep)
            if self.exp.get("smote"):
                codes = tr["cats"][rep]
                jsgg, jsido = C1.CAT_COLS.index("sgg"), C1.CAT_COLS.index("sido")
                sido_grp = np.array([str(s)[:2] for s in cats_tr[:, jsgg]], dtype=object)   # 이웃 찾기 묶음 = sgg_code 앞 2자리
                Xn, Cn, info, plan = smotenc(Xtr, codes, ytr, E.SMOTE_RATIO, E.SMOTE_K, seed, group=sido_grp)
                cats_new = np.empty((len(Cn), len(C1.CAT_COLS)), dtype=object)
                for j, c in enumerate(C1.CAT_COLS):
                    lev = np.array(ctx.levels[c], dtype=object)
                    cats_new[:, j] = lev[Cn[:, j]]
                cats_min = cats_tr[plan["minority_index"]]
                info["group_eq_sido_col"] = bool(np.all(sido_grp == cats_tr[:, jsido].astype(str)))
                info["cat_consistency"] = cc = smote_cat_consistency(cats_new, cats_min, plan["rows"])   # 기록용
                if not (info["group_eq_sido_col"] and cc["sido_eq_sgg2_rate"] == 1.0 and cc["same_as_base_rate"] == 1.0
                        and info["neighbor_same_group_rate"] in (None, 1.0)):
                    raise RuntimeError(f"SMOTE 점검 실패: 묶음=sido {info['group_eq_sido_col']}, 범주 점검 {cc}, "
                                       f"시도 안 이웃 비율 {info['neighbor_same_group_rate']}")
                Xtr = np.ascontiguousarray(np.vstack([Xtr, Xn]), dtype=np.float32)
                ytr = np.concatenate([ytr, np.ones(len(Xn), dtype=np.int8)])
                cats_tr = np.ascontiguousarray(np.vstack([cats_tr, cats_new]))
                info["seconds"] = time.time() - t0
                self.extra_info["smotenc"] = info
                self.log(f"    SMOTE({info['method']}): 검출 {info['n_minority']:,} + 새 표본 {info['n_new']:,} (불검출 {info['n_majority']:,}), "
                         f"중앙값 표준편차 {info['median_std']:.4f}, 결측 비율 원래 {info['nan_rate_minority']:.4f} -> 새 {info['nan_rate_synthetic']:.4f}")
                self.log(f"    SMOTE 시도별 학습 검출 행: {info['minority_by_group']} | 전국 대체 시도(검출 행 {info['group_min_rows']}개 미만): "
                         f"{info['fallback_groups'] or '없음'} (그 시도를 기준 행으로 한 새 표본 {info['n_new_from_fallback']:,})")
                self.log(f"    SMOTE 합성 행 범주 점검: sido = sgg 앞 2자리 {cc['sido_eq_sgg2_rate']:.1%}, 기준 검출 행과 같은 단위 "
                         f"{cc['same_as_base_rate']:.1%}, 학습 검출 단위에 있는 조합 {cc['in_train_pos_units_rate']:.1%}, "
                         f"시도 안 이웃(전국 대체 제외) " + ("해당 없음" if info["neighbor_same_group_rate"] is None
                                                         else f"{info['neighbor_same_group_rate']:.1%}"))
            self.n_train_rows = len(ytr)
            self.n_train_pos = int(ytr.sum())
        if self.exp.get("spw"):
            n_pos = int(tr["y"][ptr_pos].sum())
            n_neg = len(ptr_pos) - n_pos
            extra["scale_pos_weight"] = n_neg / n_pos
            self.extra_info["scale_pos_weight"] = {"value": n_neg / n_pos, "n_neg": n_neg, "n_pos": n_pos}
        if not isinstance(Xtr, np.ndarray):
            self.log(f"    학습 Pool {self.n_train_rows:,}행 ({time.time() - t0:.1f}s)")
        else:
            self.log(f"    학습 특성 {Xtr.shape} ({Xtr.nbytes / 1e9:.2f} GB), {time.time() - t0:.1f}s")
        va_pos = self.pos["val"]

        def make_val():
            return fb.num("val", va_pos, hf["val"][va_pos]), fb.cats("val", va_pos)

        self.m = Cat2(seed, self.log, self.rounds, self.patience, extra, mem=self.mem)
        self.m.fit(Xtr, ytr, make_val, ctx.sp["val"]["y"][va_pos], self.cols, cats_tr=cats_tr,
                   cat_names=fb.cat_names if fb.cat_names else None)
        del Xtr, cats_tr
        if self.mem:
            self.extra_info["mem_after_fit"] = self.mem.snapshot()
        self.fit_seconds = self.m.fit_seconds
        self.best_rounds = self.m.best_rounds
        self.best_val_ap = self.m.best_val_ap
        self.n_rounds_trained = self.m.n_rounds_trained

    def predict(self, sp, h):
        pos = self.pos[sp]
        return self.m.predict(self.fb.num(sp, pos, h), self.fb.cats(sp, pos))

    def info(self):
        d = self.m.info()
        d["extra"] = self.extra_info
        return d

    def close(self):
        pass


# ---------------------------------------------------------------- 로지스틱 회귀
class LRRunner:
    def __init__(self, ctx, exp, seed, feats, log, **_):
        self.ctx, self.exp, self.seed, self.feats, self.log = ctx, exp, seed, feats, log
        self.group = exp.get("group")
        self.pos = {sp: ctx.positions(sp, self.group) for sp in C1.SPLITS}
        self.fb = FeatureBuilder(ctx, feats)
        self.cols = self.fb.names
        self.n_features = len(self.cols)
        self.n_features_desc = f"숫자 {len(self.cols)} (h 1 + 기상 {E.NW if feats['weather'] else 0} + 이력 {len(hist_names(feats['hist']))}), 범주 없음"

    @property
    def feature_seconds(self):
        return self.fb.seconds

    def _z(self, X):
        Z = (X.astype(np.float64) - self.mu) / self.sd
        Z[np.isnan(Z)] = 0.0
        return Z

    def fit(self):
        from sklearn.linear_model import LogisticRegression
        hf = self.ctx.h_fit(self.seed)
        pos = self.pos["train"]
        X = self.fb.num("train", pos, hf["train"][pos])
        y = self.ctx.sp["train"]["y"][pos]
        t0 = time.time()
        Xd = X.astype(np.float64)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(Xd, axis=0)
            sd = np.nanstd(Xd, axis=0)
        self.n_allnan_cols = int(np.isnan(mu).sum())
        self.n_zero_sd_cols = int(((sd == 0) | np.isnan(sd)).sum())
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
        self.mu, self.sd = mu, sd
        self.nan_rate_train = float(np.isnan(Xd).mean())
        Z = self._z(X)
        self.m = LogisticRegression(**E.LR_PARAMS).fit(Z, y)
        self.fit_seconds = time.time() - t0
        self.n_iter = int(np.max(self.m.n_iter_))
        self.n_train_rows, self.n_train_pos = len(y), int(y.sum())
        va = self.pos["val"]
        pv = self.predict_X(self.fb.num("val", va, hf["val"][va]))
        from sklearn.metrics import average_precision_score
        self.best_val_ap = float(average_precision_score(self.ctx.sp["val"]["y"][va], pv))   # 기록용(조기 종료 없음)
        self.best_rounds = self.n_iter
        self.n_rounds_trained = self.n_iter
        self.log(f"    로지스틱: 반복 {self.n_iter} (최대 {E.LR_PARAMS['max_iter']}), 결측 비율 {self.nan_rate_train:.4f}, "
                 f"{self.fit_seconds:.1f}s")

    def predict_X(self, X):
        return self.m.predict_proba(self._z(X))[:, 1].astype(np.float32)

    def predict(self, sp, h):
        pos = self.pos[sp]
        return self.predict_X(self.fb.num(sp, pos, h))

    def info(self):
        return {"library_params": {**E.LR_PARAMS, "class_weight": None},
                "standardize": "학습 행 평균·표준편차(결측 무시, ddof=0)로 표준화, 표준편차 0/전부 결측 열은 1, 결측은 표준화 뒤 0(=학습 평균)",
                "n_iter": self.n_iter, "converged": self.n_iter < E.LR_PARAMS["max_iter"],
                "nan_rate_train": self.nan_rate_train, "n_allnan_cols": self.n_allnan_cols,
                "n_zero_sd_cols": self.n_zero_sd_cols,
                "coef": [float(c) for c in self.m.coef_[0]], "intercept": float(self.m.intercept_[0]),
                "early_stopping": "없음(검증은 기준값·선정에만). best_val_ap = 조기 종료용 검증 h 배정에서의 AP(기록용)"}

    def close(self):
        pass


def make_runner(ctx, exp, seed, feats, log, smoke, cache_dir=None, mem=None):
    if exp["model"] == "cat":
        rounds = E.SMOKE_MAX_ROUNDS if smoke else E.MAX_ROUNDS
        patience = E.SMOKE_PATIENCE if smoke else E.PATIENCE
        return CatRunner(ctx, exp, seed, feats, log, rounds, patience, cache_dir=cache_dir, mem=mem)
    if exp["model"] == "lr":
        return LRRunner(ctx, exp, seed, feats, log)
    if exp["model"] == "dl":
        import e2dl
        fac = e2dl.DL_REGISTRY.get(exp["id"])
        if fac is None:
            raise NotImplementedError(f"DL 실험 {exp['id']} 은 아직 구현되지 않음(e2dl.DL_REGISTRY 에 등록 필요)")
        return fac(ctx, exp, seed, log, smoke)
    if exp["model"] == "ae":                               # 점수 단독(S_*): CatBoost 없음, 이상 점수를 예측값으로
        import e2ae
        return e2ae.AEScoreRunner(ctx, exp, seed, feats, log, smoke)
    raise ValueError(exp["model"])
