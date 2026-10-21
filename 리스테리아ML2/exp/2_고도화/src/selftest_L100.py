# -*- coding: utf-8 -*-
"""기상 창 길이 설정(EXP2_L)·T 통합 선정(EXP2_SELECT=tint)·양방향 단계 선택(FS·BW)·E00_P·E00_Q 자체 점검 (사용자 지시 2026-10-19, 100일 위 2차 실험 A단계).
결과: logs/selftest_L100.txt (UTF-8). 실패가 있으면 종료 코드 1. results/·tables/·cache/ 에는 쓰지 않는다(기상 요약 캐시는 쓰지도 읽지도 않음).

부분(--parts, 쉼표로):
 a  EXP2_L·EXP2_SELECT 없음(= 50일·old): 기존 실험(E00·A3·F4·G2·M1·Q4·P3·A3_N1·A3_L1U1·A3_AEt·D3·B2·F2b·A3_F4b·E00_NW·S_AEt·S_AEs)의
    특성 칸 이름·칸 수·범주 칸·n_features_desc·config 의 ext/nb/ae 설명·표 '입력' 글, 표본 행렬(분할마다 무작위 1,500행 x 무작위 h, 학습 = 시드 42 h)과
    AE 입력(표 AE 입력 행렬 전체 md5, 시퀀스 AE 정규화 통계·창 묶음), 설정 시트 글(F·G·M·Q·P·AE·XS·설정 줄)이 고치기 전 코드(--old-src)와 비트 단위로 같음.
    고치기 전 코드 = scratch/L100/*_before_L100.py.bak 을 넣은 src 사본(서버에서는 ~/cyj_listeria/exp_before_L100/2_고도화/src). 둘은 따로 프로세스에서 계산.
    AE 점수 칸은 GPU 를 쓰지 않도록 두 쪽 모두 같은 가짜 점수(고정 난수)로 바꿔 칸 연결만 대조(실제 점수 계산 코드는 AE 입력 대조로 확인).
    run.alignment_text() = 이전 글, run.extra_cfg() = {} (config 칸이 이전과 같음)
 b  EXP2_L=100: 기상 요약 칸 = 19 x (100/10 + 5) = 285, 칸 이름 = 1차 feature_names('summary', 100), 기상 칸 = 1차 build_tree_matrix(L=100) 비트 단위,
    창 = T−100..T−1(넘파이로 직접 잰 평균·최소·최대·10일 구간 평균과 같음; 대조군 = 당일 T 를 넣은 창 T−99..T 는 다름), 이동평균 차감·SPI·생장식·생장 범위 창도 100일
    (직접 계산과 같음, 50일 창과는 다름), 시퀀스 AE 입력 (B, 38, 100) = 표 [T−100..T−1], DL 설정 L·위치 임베딩 100, 표 AE 입력 칸 295, 창 확인 통과,
    alignment 글 'T-100', config 칸 wx_L = 100
 c  T 통합 선정 함수(e2eval.val_tint_pr_auc) = analysis/T기준_통합 표의 '검증 60일평균 PR-AUC'(시드별·실험별, 차이 1e-12 이하) - A1_k10~k200·A1·A2·A3·F1~F4·G1~G3,
    50일 결과(results/) 읽기만. tint 규칙으로 고른 k*·A 최고·F·G 선정을 기록(선정 파일은 쓰지 않음)
 e  AE·VAE 점수 규칙 v2(GPU, 시험 실행 자료 AEt·AEs): 학습 행 전부(검출 포함)가 자기 겹 밖 모델 점수, 검증·시험 = 겹 모델 5개 평균(비트 단위),
    겹 x 정답 교차표·전체 모델 없음, 같은 시드 두 번 같은 값, 캐시 이름 '_v2'·메타 score_rule
 d  EXP2_L=100·tint: 단계 선택 묶음 12개·STAGES('fs'·'bw' 가 'tables' 앞, 'cum' 없음), 결정 규칙 단위 시험(가짜 점수: 전진 = 후보 > 기준이면 남김,
    뒤로 = 가장 많이 오른 후보를 빼고 반복, 이어하기가 같은 결정), 후보 행렬 = 기준 행렬 ± 묶음 칸(이름으로 맞춰 비트 단위; E00 + 묶음 11개, 중간 기준 + 묶음 9개,
    12묶음 − 묶음 12개), ae4 표 AE 입력 = 기준 모델 입력(aw_label 번호 포함)·점수 캐시 입력 구분, E00_P·E00_Q = E00 행렬 + 묶음 칸
사용: python src/selftest_L100.py --parts a,b,d --old-src <고치기 전 src 사본>   (서버, 자료가 있는 cwd 에서 PYTHONUTF8=1)
      python src/selftest_L100.py --parts c                                     (이 PC, 결과 읽기만)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", category=RuntimeWarning)
A_IDS = ["E00", "A3", "F4", "G2", "M1", "Q4", "P3", "A3_N1", "A3_L1U1", "A3_AEt", "D3", "B2", "F2b", "A3_F4b", "E00_NW",
         "S_AEt", "S_AEs"]
K_FIX = 50                                  # 점검용 k(행렬 구조만 보므로 아무 값이나 같게)
N_ROWS = 1500
C_IDS = [f"A1_k{k}" for k in (10, 30, 50, 100, 200)] + ["A1", "A2", "A3", "F1", "F2", "F3", "F4", "G1", "G2", "G3"]
NEW_IDS = {"E00_P", "E00_Q", "ALL_A", "ALL_A0", "ALL_B", "E00_N2", "E00_N0"}                                                # 이번에 더한 실험(고치기 전 코드에 없음, FS·BW 는 EXPERIMENTS 밖)
SOLO = {"E00_P": ["eco", "gw", "haccp"], "E00_Q": ["rgwx", "rgsoil", "rgfood"]}   # 묶음 단독 추가(베이스라인 기준)


def _set_src(src):
    """sys.path 에서 이 파일 폴더(새 코드)를 빼고 src 를 맨 앞에 둔다(고치기 전 코드로 계산할 때)."""
    keep = []
    for p in sys.path:
        try:
            if p and Path(p).resolve() == HERE:
                continue
        except Exception:
            pass
        keep.append(p)
    sys.path[:] = [str(src)] + keep


def _log(msg):
    try:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    except Exception:
        pass


def _jn(x):
    """json 으로 비교할 수 있게(numpy·Path 등 -> 문자열/숫자)."""
    return json.loads(json.dumps(x, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else
                                 (o.tolist() if hasattr(o, "tolist") else str(o))))


def _md5(a):
    import numpy as np
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _patch_ae(e2ae, E):
    """AE 점수를 GPU 없이 고정 난수 가짜 점수로(두 코드판 같은 값). 칸 연결 대조용."""
    import numpy as np
    cache = {}

    def fake(ctx, kind, seed, k, log=None, tab_feats=None):
        sig = int(e2ae.tab_sig(tab_feats)[:6], 16) if (tab_feats is not None and kind in ("AEt", "VAEt")) else 0
        key = (kind, int(seed), sig)
        if key not in cache:
            r = np.random.default_rng([int(seed), list(E.AE_KINDS).index(kind), 7] + ([sig] if sig else []))
            S = {"train": r.random(len(ctx.sp["train"]["idx"])).astype(np.float32),
                 "val": r.random((E.H_MAX, len(ctx.sp["val"]["idx"]))).astype(np.float32),
                 "test": r.random((E.H_MAX, len(ctx.sp["test"]["idx"]))).astype(np.float32)}
            cache[key] = (S, {"fake": True}, "가짜 점수(점검용)")
        return cache[key]
    e2ae.get_scores = fake


def _samples(ctx, E, seed=20261019, n=N_ROWS):
    """분할마다 무작위 행 n 개와 h(학습 = 시드 42 의 h 배정)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {}
    for sp in ("train", "val", "test"):
        nn_ = len(ctx.sp[sp]["idx"])
        pos = np.sort(rng.choice(nn_, min(n, nn_), replace=False))
        if sp == "train":
            hh = ctx.h_fit(42)["train"][pos].astype(np.int16)
        else:
            hh = rng.integers(1, E.H_MAX + 1, size=len(pos)).astype(np.int16)
        out[sp] = (pos, hh)
    return out


def _resolved(E, eid, k=K_FIX):
    f = E.EXP_BY_ID[eid]["feats"]
    f = dict(f)
    if f.get("k") == "k*":
        f["k"] = k
    if isinstance(f.get("ae_tab"), dict) and f["ae_tab"].get("k") == "k*":
        f["ae_tab"] = dict(f["ae_tab"], k=k)
    return f


# ================================================================ (a) 50일·old: 한 코드판의 값 덤프
def dump_a(out_path):
    import numpy as np
    import e2cfg as E
    import e2ae
    import e2fg
    import e2models
    import e2nb
    import e2tables
    import torch
    from e2data import Ctx2
    _patch_ae(e2ae, E)
    import e2hist                                            # 설정 시트 'HX 날짜 시계 확인' 이 읽는 파일 = 두 코드판 같은 파일(새 코드 폴더 기준)
    e2hist.HX_DATECLOCK_CSV = HERE.parent / "logs" / e2hist.HX_DATECLOCK_CSV.name
    ctx = Ctx2(False, _log, use_cache=False)
    smp = _samples(ctx, E)
    meta, arrs = {"L": E.L}, {}
    for eid in A_IDS:
        exp = E.EXP_BY_ID[eid]
        feats = _resolved(E, eid)
        if exp["model"] == "cat":
            r = e2models.CatRunner(ctx, exp, 42, feats, _log, 10, 5)
        elif exp["model"] == "lr":
            r = e2models.LRRunner(ctx, exp, 42, feats, _log)
        elif exp["model"] == "ae":
            r = e2ae.AEScoreRunner(ctx, exp, 42, feats, _log, False)
        else:
            raise ValueError(eid)
        m = {"n_features": r.n_features, "n_features_desc": r.n_features_desc, "cols": r.cols,
             "input_desc": e2tables.input_desc(feats, exp["model"]) if exp["model"] != "ae" else e2tables._ae_only_desc(feats["ae"])}
        if hasattr(r, "fb"):
            m["cat_names"] = r.fb.cat_names
            m["names"] = r.fb.names
            for sp, (pos, hh) in smp.items():
                X = r.fb.num(sp, pos, hh)
                arrs[f"{eid}|{sp}|X"] = X
                C = r.fb.cats(sp, pos)
                if C is not None:
                    arrs[f"{eid}|{sp}|C"] = np.asarray(C).astype(str)
                if sp == "val":                                  # 정수 h(평가 경로)도
                    arrs[f"{eid}|{sp}|Xh7"] = r.fb.num(sp, pos, 7)
        if feats.get("ext"):
            m["ext_info"] = _jn(e2fg.ext_info(feats["ext"]))
            m["ext_desc"] = e2fg.ext_desc(feats["ext"])
        if feats.get("nb"):
            m["nb_info"] = _jn(e2nb.nb_info(feats["nb"]))
        if feats.get("ae"):
            m["ae_info"] = _jn(e2ae.ae_info(feats["ae"]))
        meta[eid] = m
        _log(f"  덤프 {eid}: 칸 {len(r.cols or [])}")
    # AE 입력(점수 계산 코드가 쓰는 입력)
    htr = ctx.h_fit(42)["train"]
    ti = e2ae.TabInput(ctx, K_FIX, htr)
    meta["AE_tab_X_md5"] = _md5(ti.X)
    meta["AE_tab_X_shape"] = list(ti.X.shape)
    del ti
    si = e2ae.SeqInput(ctx, htr)
    neg = np.flatnonzero(ctx.sp["train"]["y"] == 0)
    prep = si.model_prep(neg[:30000])
    meta["AE_seq_prep"] = {"mu": prep["mu64"].tolist(), "sd": prep["sd64"].tolist(), "L": si.L}
    dd = si.device_train(prep, torch.device("cpu"))
    rows = np.sort(np.random.default_rng(5).choice(len(htr), 2000, replace=False))
    x, _ = si.batch(dd, torch.from_numpy(rows.astype(np.int64)))
    arrs["AE|seq_batch"] = x.numpy()
    meta["AE_texts"] = _jn({"input_t": e2ae.input_text("AEt"), "input_s": e2ae.input_text("AEs"), "arch": e2ae.ARCH_TEXT,
                            "rule_normal": e2ae.RULE_NORMAL, "settings": e2ae.settings_rows()})
    # 설정 시트 글
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        P = {"results": Path(td) / "results", "tables": Path(td) / "tables"}
        P["results"].mkdir()
        meta["settings_rows"] = e2tables.settings_rows(P, False, None, None, {}, {"note": "점검"}).values.tolist()
    meta["texts"] = _jn({"fg": e2tables.fg_settings_rows({"note": "점검"}), "wx": e2fg.wxanom_settings_rows(),
                         "rg": e2fg.rg_settings_rows(), "p": e2fg.p_settings_rows(), "xs": e2tables.XS_RULE,
                         "F_NOTES": e2fg.F_NOTES, "g_notes": e2fg.g_notes(), "BLOCK_DESC": e2fg.BLOCK_DESC,
                         "WXANOM_RULE": e2fg.WXANOM_RULE, "RG_JOIN": e2fg.RG_JOIN})
    meta["input_desc_all"] = {e["id"]: e2tables.input_desc(_resolved(E, e["id"]), e["model"])
                              for e in E.EXPERIMENTS if isinstance(e.get("feats"), dict) and e["model"] in ("cat", "lr")
                              and e["id"] not in NEW_IDS}
    meta["exp_defs"] = {e["id"]: _jn(e) for e in E.EXPERIMENTS if e["id"] not in NEW_IDS}
    meta["ext_info_all"] = {e["id"]: _jn(e2fg.ext_info(e["feats"]["ext"])) for e in E.EXPERIMENTS
                            if e["id"] not in NEW_IDS and isinstance(e.get("feats"), dict) and e["feats"].get("ext")}
    meta["stages"] = [x for x in E.STAGES if x not in ("cum", "fs", "bw", "xall")]
    try:
        import run as R
        meta["run_alignment"] = R.alignment_text()
        meta["run_extra_cfg"] = {eid: R.extra_cfg(E.EXP_BY_ID[eid]) for eid in A_IDS}
    except AttributeError:
        meta["run_alignment"] = None                             # 고치기 전 run.py 에는 함수가 없음
    np.savez(out_path, **arrs)
    Path(str(out_path) + ".json").write_text(json.dumps(_jn(meta), ensure_ascii=False, indent=1), encoding="utf-8")


def compare_a(new_p, old_p, check):
    import numpy as np
    mn = json.loads(Path(str(new_p) + ".json").read_text(encoding="utf-8"))
    mo = json.loads(Path(str(old_p) + ".json").read_text(encoding="utf-8"))
    an, ao = np.load(new_p), np.load(old_p)
    check("(a) 두 코드판 모두 기상 창 50", mn["L"] == 50 and mo["L"] == 50)
    def known_gw(ext_info):
        """알려진 차이(2026-10-19 고침): 지하수 'gw'·'gw_p' 가 있고 G 묶음(g1·g2·g3)이 없는 실행은 옛 config 에 생장식 설명 'growth' 가 잘못 들어 있었음."""
        b = (ext_info or {}).get("blocks") or []
        return any(x in ("gw", "gw_p") for x in b) and not any(x in ("g1", "g2", "g3") for x in b)

    def strip_growth(ei):
        ei = dict(ei)
        had = ei.pop("growth", None) is not None
        return ei, had

    AE_V2_KEYS = ("rule_rows", "cache")                       # 알려진 차이(사용자 승인 2026-10-19): AE·VAE 점수 규칙 v2 로 설명 글만 바뀜
    AE_V2_ROWS = ("AE 행과 h·교차 적합", "AE 점수 저장")

    def ae_strip(ai):
        return {kk: vv for kk, vv in ai.items() if kk not in AE_V2_KEYS}

    for eid in A_IDS:
        mo_e, mn_e, note = dict(mo[eid]), dict(mn[eid]), ""
        if "ae_info" in mo_e and "ae_info" in mn_e:
            chg = [kk for kk in AE_V2_KEYS if mo_e["ae_info"].get(kk) != mn_e["ae_info"].get(kk)]
            mo_e["ae_info"], mn_e["ae_info"] = ae_strip(mo_e["ae_info"]), ae_strip(mn_e["ae_info"])
            if chg:
                note += f" [알려진 차이: AE 점수 규칙 v2 설명 칸 {chg}]"
        if known_gw(mo_e.get("ext_info")):
            mo_e["ext_info"], had = strip_growth(mo_e["ext_info"])
            note = f" [알려진 차이: 옛 config 의 생장식 설명 'growth' 칸 빠짐(옛 판에 있었음 {had}, 새 판에 없음 {'growth' not in mn_e['ext_info']})]"
            if not had or "growth" in mn_e["ext_info"]:
                note += " - 예상과 다름"
                mo_e["ext_info"] = mo[eid]["ext_info"]
        dif = sorted(k for k in set(mn_e) | set(mo_e) if mn_e.get(k) != mo_e.get(k))
        check(f"(a) {eid}: 칸 이름·칸 수·범주 칸·n_features_desc·ext/nb/ae 설명·표 '입력' 글 같음", not dif,
              f"다른 항목 {dif} | 칸 {mn[eid]['n_features']} | {mn[eid]['n_features_desc']}{note}")
        keys = sorted(k for k in set(an.files) | set(ao.files) if k.startswith(eid + "|"))
        bad = [k for k in keys if k not in an.files or k not in ao.files or an[k].dtype != ao[k].dtype or an[k].shape != ao[k].shape
               or an[k].tobytes() != ao[k].tobytes()]
        if keys:
            check(f"(a) {eid}: 표본 행렬 비트 단위 같음(학습·검증·시험, 범주 포함)", not bad, f"배열 {len(keys)}개, 다른 것 {bad}")
    kn, bad = [], []
    for eid, oi in mo["ext_info_all"].items():
        ni = mn["ext_info_all"].get(eid)
        if known_gw(oi):
            oi2, had = strip_growth(oi)
            (kn if (had and ni == oi2) else bad).append(eid)
        elif ni != oi:
            bad.append(eid)
    check("(a) config 'ext' 설명(외부 묶음 있는 기존 실험 전부) 같음 - 지하수가 든 실험은 'growth' 칸만 빠짐(알려진 차이)",
          not bad and set(mn["ext_info_all"]) == set(mo["ext_info_all"]), f"알려진 차이 {kn}, 그 밖 다른 것 {bad}")
    check("(a) 표 AE 입력 행렬(학습 행 전체, 시드 42 h) md5 같음", mn["AE_tab_X_md5"] == mo["AE_tab_X_md5"] and mn["AE_tab_X_shape"] == mo["AE_tab_X_shape"],
          f"{mn['AE_tab_X_shape']}")
    check("(a) 시퀀스 AE 정규화 통계·창 묶음 같음", mn["AE_seq_prep"] == mo["AE_seq_prep"]
          and an["AE|seq_batch"].tobytes() == ao["AE|seq_batch"].tobytes(), f"묶음 {an['AE|seq_batch'].shape}")
    def ae_txt(t):
        t = dict(t)
        t["settings"] = [r for r in t["settings"] if r[0] not in AE_V2_ROWS]
        return t
    chg_rows = [r[0] for r in mo["AE_texts"]["settings"] if r[0] in AE_V2_ROWS and r not in mn["AE_texts"]["settings"]]
    check("(a) AE 설명 글(입력·구조·정규화·설정 시트) 같음 - 점수 규칙 v2 로 바뀐 줄만 다름(알려진 차이)", ae_txt(mn["AE_texts"]) == ae_txt(mo["AE_texts"]),
          f"바뀐 줄 {chg_rows}")
    check("(a) 설정 시트 글(F·G·M·Q·P·XS, F_NOTES, g_notes, BLOCK_DESC) 같음", mn["texts"] == mo["texts"],
          str([k for k in mn["texts"] if mn["texts"][k] != mo["texts"].get(k)]))
    dif = [k for k in mo["exp_defs"] if mn["exp_defs"].get(k) != mo["exp_defs"][k]]
    check("(a) 기존 실험 정의(EXPERIMENTS 전부: 단계·입력·설명·시드 등) 같음, 단계 목록 = 이전 + 'xall'·'fs'·'bw'",
          not dif and set(mn["exp_defs"]) == set(mo["exp_defs"]) and mn["stages"] == mo["stages"], f"기존 실험 {len(mo['exp_defs'])}개, 다른 것 {dif}")
    dif = [k for k in mo["input_desc_all"] if mn["input_desc_all"].get(k) != mo["input_desc_all"][k]]
    check("(a) 표 '입력' 글(기존 실험 전부) 같음", not dif and set(mn["input_desc_all"]) == set(mo["input_desc_all"]), str(dif))
    def v2rows(rows):
        return [[r[0], "(AE 점수 규칙 v2 로 바뀐 줄)"] if r[0] in AE_V2_ROWS else r for r in rows]
    sn, so = v2rows(mn["settings_rows"]), v2rows(mo["settings_rows"])
    extra = [r for r in sn if r not in so]
    new_rows = ["XALL 추가 변수 전부", "FS·BW 양방향 단계 선택"]
    check("(a) 설정 시트 줄 = 이전 줄 + 'XALL 추가 변수 전부'·'FS·BW 양방향 단계 선택' 두 줄(순서 그대로, AE 점수 규칙 v2 두 줄은 글만 바뀜 - 알려진 차이)",
          [r for r in sn if r[0] not in new_rows] == so and [r[0] for r in extra] == new_rows, f"더해진 줄 {[r[0] for r in extra]}")
    old_align = "목표일 d, 앞당김 h -> T = d - h, 기상 창 = T-50 .. T-1, 이력 창 = T-P .. T-1 (T 당일 제외)"
    check("(a) run.alignment_text() = 이전 config 글, run.extra_cfg() = {} (config 칸 이전과 같음)",
          mn["run_alignment"] == old_align and all(v == {} for v in mn["run_extra_cfg"].values()), str(mn["run_extra_cfg"]))


# ================================================================ (b) L=100
def run_b():
    import numpy as np
    import e2cfg as E
    import e2ae
    import e2fg
    import features as F
    import torch
    from e2data import Ctx2
    from e2models import FeatureBuilder
    import run as R
    res = []

    def check(n, ok, d=""):
        res.append((n, bool(ok), str(d)))

    check("(b) EXP2_L=100 -> e2cfg.L 100, 기상 요약 칸 NW = 19 x (100/10 + 5) = 285", E.L == 100 and E.NW == 285 == F.n_weather_features("summary", 100),
          f"L {E.L}, NW {E.NW}")
    check("(b) DL 설정 L 100·트랜스포머 위치 임베딩 100, AE 시퀀스 L 100", E.DL["L"] == 100 and E.DL["trf"]["pos_emb"] == 100 and E.AE["seq"]["L"] == 100)
    from e2dl_impl import make_net
    net = make_net("trf", "A", 38, [17, 250, 40, 300], E.DL["L"])
    check("(b) 트랜스포머 위치 임베딩 크기 (100, 64) (CPU 에서 만들기만)", tuple(net.pos_emb.weight.shape) == (100, 64), str(tuple(net.pos_emb.weight.shape)))
    check("(b) 표 AE 입력 칸 N_TAB = 1 + 285 + 9 = 295, 시퀀스 길이 글 100 -> 50 -> 25 -> 50 -> 100",
          e2ae.N_TAB == 295 and e2ae.SEQ_LENS == "100 -> 50 -> 25 -> 50 -> 100", f"{e2ae.N_TAB} | {e2ae.SEQ_LENS}")
    check("(b) alignment 글 'T-100', config 칸 wx_L = 100", "기상 창 = T-100 .. T-1" in R.alignment_text() and R.extra_cfg().get("wx_L") == 100,
          f"{R.alignment_text()} | {list(R.extra_cfg())}")
    ctx = Ctx2(False, _log, use_cache=False)
    try:
        win = R.check_windows(ctx, _log)
        check("(b) 창 확인(모든 분할·h 1..60, 창 T−100..T−1) 기상표 안", True, str(win))
    except AssertionError as e:
        check("(b) 창 확인(모든 분할·h 1..60, 창 T−100..T−1) 기상표 안", False, str(e))
    W = ctx.weather.W
    smp = _samples(ctx, E, seed=11, n=1000)
    fb = FeatureBuilder(ctx, _resolved(E, "E00"), seed=42)
    check("(b) E00 칸 = ['h'] + 1차 feature_names('summary', 100) (286칸)", fb.names == ["h"] + F.feature_names("summary", 100) and len(fb.names) == 286,
          f"{len(fb.names)}칸")
    ok_bits, ok_manual, ctrl_diff = True, True, []
    vi = {v: i for i, v in enumerate(F.C.WEATHER_VARS)}
    nm = {n: j for j, n in enumerate(fb.names)}
    for sp, (pos, hh) in smp.items():
        X = fb.num(sp, pos, hh)
        s = ctx.sp[sp]["s"][pos]
        T = ctx.sp[sp]["d"][pos] - hh.astype(np.int64)
        ref = F.build_tree_matrix(W, s, T, hh, 100, "summary")
        ok_bits &= X.tobytes() == ref.tobytes()
        with np.errstate(all="ignore"):
            for v in ("avg_temp", "rain", "soil_moist"):
                c = vi[v]
                w100 = np.stack([W[s[i], T[i] - 100:T[i], c] for i in range(len(s))])
                man = {"mean": np.nanmean(w100, 1), "min": np.nanmin(w100, 1), "max": np.nanmax(w100, 1),
                       "b01": np.nanmean(w100[:, -10:], 1), "b10": np.nanmean(w100[:, :10], 1)}
                for st, mv in man.items():
                    got = X[:, nm[f"{v}_{st}"]].astype(np.float64)
                    ok_manual &= bool(np.allclose(got, mv, rtol=2e-6, atol=1e-6, equal_nan=True))
                wT = np.stack([W[s[i], T[i] - 99:T[i] + 1, c] for i in range(len(s))])      # 대조군: 당일 T 포함
                ctrl = np.nanmean(wT, 1)
                got = X[:, nm[f"{v}_mean"]].astype(np.float64)
                fin = np.isfinite(ctrl) & np.isfinite(got)
                ctrl_diff.append(float(np.mean(~np.isclose(got[fin], ctrl[fin], rtol=2e-6, atol=1e-6))))
    check("(b) E00 기상 칸 = 1차 build_tree_matrix(L=100, 'summary') 비트 단위(학습·검증·시험 표본)", ok_bits)
    check("(b) 창 = T−100..T−1: 평균·최소·최대·b01(T−10..T−1)·b10(T−100..T−91) = 넘파이 직접 계산(avg_temp·rain·soil_moist)", ok_manual)
    check("(b) 대조군: 당일 T 를 넣은 창(T−99..T) 평균은 다름(행 대부분)", min(ctrl_diff) > 0.5, f"다른 행 비율 {[round(x, 3) for x in ctrl_diff]}")
    # 외부·생장 창(100일)
    blocks = ["spi", "g1", "g3", "wxanom", "rgwx"]
    fbx = FeatureBuilder(ctx, dict(_resolved(E, "E00"), ext=blocks), seed=42)
    X_ = e2fg.get_ext(ctx)
    AN = e2fg.get_wxanom(ctx)
    RG = e2fg.get_rg(ctx)
    nmx = {n: j for j, n in enumerate(fbx.names)}
    ok = {"spi": True, "g1": True, "g3": True, "wxanom": True, "rgwx": True}
    diff50 = {}
    for sp, (pos, hh) in smp.items():
        X = fbx.num(sp, pos, hh)
        s = ctx.sp[sp]["s"][pos]
        T = ctx.sp[sp]["d"][pos] - hh.astype(np.int64)
        idx = T[:, None] - 100 + np.arange(100)[None, :]
        with np.errstate(all="ignore"):
            spi = X_.SPI[s[:, None], idx]
            ok["spi"] &= bool(np.allclose(X[:, nmx["spi3_mean"]], np.nanmean(spi[:, :, 0], 1), rtol=2e-6, atol=1e-6, equal_nan=True)
                              and np.allclose(X[:, nmx["spi6_b10"]], np.nanmean(spi[:, :10, 1], 1), rtol=2e-6, atol=1e-6, equal_nan=True))
            fT = X_.fT[s[:, None], idx]
            ok["g1"] &= bool(np.allclose(X[:, nmx["g1_fT_mean"]], np.nanmean(fT, 1), rtol=2e-6, atol=1e-6, equal_nan=True)
                             and np.allclose(X[:, nmx["g1_fT_sum"]], np.where(np.isnan(fT).all(1), np.nan, np.nansum(fT, 1)), rtol=2e-6, atol=1e-5, equal_nan=True))
            Q = X_.Q[s[:, None], idx]
            ok["g3"] &= bool(np.allclose(X[:, nmx["g3_P_ivanek_max"]], np.nanmax(Q, 1), rtol=2e-6, atol=1e-6, equal_nan=True))
            a = AN[s[:, None], idx]
            ok["wxanom"] &= bool(np.allclose(X[:, nmx[f"avg_temp_anom{E.WXANOM_K}_mean"]], np.nanmean(a[:, :, 0], 1), rtol=2e-5, atol=1e-5, equal_nan=True)
                                 and np.allclose(X[:, nmx[f"rain_anom{E.WXANOM_K}_max"]], np.nanmax(a[:, :, 3], 1), rtol=2e-6, atol=1e-5, equal_nan=True))
            rg = RG.WX[s[:, None], idx]
            ok["rgwx"] &= bool(np.allclose(X[:, nmx["rg_tmax_opt_mean"]], np.nanmean(rg[:, :, 0], 1), rtol=2e-6, atol=1e-6, equal_nan=True))
            idx50 = T[:, None] - 50 + np.arange(50)[None, :]
            m50 = np.nanmean(AN[s[:, None], idx50][:, :, 0], 1)
            g = X[:, nmx[f"avg_temp_anom{E.WXANOM_K}_mean"]]
            fin = np.isfinite(m50) & np.isfinite(g)
            diff50[sp] = float(np.mean(~np.isclose(g[fin], m50[fin], rtol=2e-5, atol=1e-5)))
    for b, v in ok.items():
        check(f"(b) {b} 창 = T−100..T−1 (넘파이 직접 계산과 같음)", v)
    check("(b) 이동평균 차감 평균은 50일 창 값과 다름(행 대부분)", min(diff50.values()) > 0.5, str({k: round(v, 3) for k, v in diff50.items()}))
    check("(b) SPI 창 요약 칸 30 = 2 x (100/10 + 5)", len(e2fg.ext_names(["spi"])) == 30 == E.NSPI)
    # 시퀀스 AE 입력
    htr = ctx.h_fit(42)["train"]
    si = e2ae.SeqInput(ctx, htr)
    neg = np.flatnonzero(ctx.sp["train"]["y"] == 0)
    prep = si.model_prep(neg[:30000])
    dd = si.device_train(prep, torch.device("cpu"))
    rows = np.sort(np.random.default_rng(3).choice(len(htr), 500, replace=False))
    x, _ = si.batch(dd, torch.from_numpy(rows.astype(np.int64)))
    tb = dd["tb"].numpy()
    s = ctx.sp["train"]["s"][rows]
    T = ctx.sp["train"]["d"][rows].astype(np.int64) - htr[rows].astype(np.int64)
    exp_x = np.stack([tb[s[i], T[i] - 100:T[i], :].T for i in range(len(rows))])
    check("(b) 시퀀스 AE 입력 (B, 38, 100) = 표 [T−100..T−1] (위치 0 = T−100, 99 = T−1)", si.L == 100 and tuple(x.shape) == (500, 38, 100)
          and np.array_equal(x.numpy(), exp_x), f"{tuple(x.shape)}")
    ti_names = FeatureBuilder(ctx, e2ae.a3_feats(K_FIX)).names
    check("(b) 표 AE 입력 칸 = A3 FeatureBuilder 칸 수 295", len(ti_names) == e2ae.N_TAB == 295, str(len(ti_names)))
    return res


# ================================================================ (d) 단계 선택 FS·BW (L=100, tint)
def fs_unit_tests(check):
    """단계 선택 결정 규칙 단위 시험(자료 없음): 가짜 점수로 전진·뒤로 결정이 점수 비교와 맞는지, 이어하기(일부 결과만 있을 때)가 같은 결정을 내는지."""
    import e2cfg as E
    import e2fs
    w = {"spi": 0.01, "soilmed": -0.002, "food": 0.02, "eco_gw_haccp": 0.0, "rg": 0.003, "g123": -0.001, "wxanom": 0.004,
         "hist_core": 0.03, "hist_1y": 0.001, "woe": -0.0005, "n1": 0.002, "ae4": 0.0015}

    def fake_score(exp):
        codes = set((exp.get("fs_info") or {}).get("bundles") or [])
        v = 0.05 + sum(w[c] for c in e2fs.sort_codes(codes))
        if "spi" in codes and "hist_core" in codes:                 # 상호작용: 이력이 들어가면 SPI 를 빼는 편이 나음
            v -= 0.012
        return (v, [v, v, v])

    seen = []

    def so_all(exp):
        seen.append(exp["id"])
        return fake_score(exp)
    steps, fin, need = e2fs.replay("bw", so_all, 10)
    fw = [s for s in steps if s["phase"] == "forward"]
    bw = [s for s in steps if s["phase"] == "backward"]
    want_fw = ["keep", "drop", "keep", "drop", "keep", "drop", "keep", "keep", "keep", "drop", "keep", "keep"]
    ok = need is None and [s["decision"] for s in fw] == want_fw and [s["candidate_id"] for s in fw] == [f"FS{i:02d}" for i in range(1, 13)]
    # 결정 = 점수 비교(후보 > 기준), 같으면 뺌(FS04 는 가중치 0 이라 기준과 같은 점수)
    ok &= all((s["candidate_score"] > s["base_score"]) == (s["decision"] == "keep") for s in fw)
    ok &= fw[3]["candidate_score"] == fw[3]["base_score"] and fw[3]["decision"] == "drop"
    ok &= all(fw[i]["base_id"] == (fw[i - 1]["new_base_id"] if i else "E00") for i in range(12))
    check("(d) 단위 시험: 전진 결정 = 점수 비교(후보 > 기준이면 남김, 같으면 뺌), 기준 이어짐", ok,
          f"{[(s['candidate_id'], s['decision']) for s in fw]}")
    kept = [s["bundle"] for s in fw if s["decision"] == "keep"]
    r1 = [s for s in bw if s["round"] == 1]
    r2 = [s for s in bw if s["round"] == 2]
    ok = (len(r1) == len(kept) and [s["bundle"] for s in r1] == kept and [s["candidate_id"] for s in r1] == [f"BW{i:02d}" for i in range(1, len(kept) + 1)]
          and [s["decision"] for s in r1] == ["remove"] + ["keep"] * (len(kept) - 1) and r1[0]["bundle"] == "spi"
          and all(s["new_base_id"] == "BW01" for s in r1)
          and len(r2) == len(kept) - 1 and all(s["decision"] == "keep" and s["new_base_id"] == "BW01" for s in r2)
          and r2[0]["candidate_id"] == f"BW{len(kept) + 1:02d}" and not any(s["round"] == 3 for s in bw)
          and fin["id"] == "BW01" and fin["codes"] == [c for c in kept if c != "spi"])
    best = max(r1, key=lambda s: s["candidate_score"])
    ok &= best["decision"] == "remove" and best["candidate_score"] > r1[0]["base_score"]
    ok &= all(s["candidate_score"] <= s["base_score"] for s in r2)
    check("(d) 단위 시험: 뒤로 = 오른 후보 중 가장 많이 오른 것(spi)을 빼고 다시, 오른 것 없으면 끝(라운드 2 모두 그대로 둠), 차례 번호 BWnn",
          ok, f"라운드1 {[(s['candidate_id'], s['bundle'], s['decision']) for s in r1]} | 라운드2 {len(r2)}개 | 최종 {fin['id']} {fin['codes']}")
    # 이어하기: 앞 m 개 후보 결과만 있을 때 -> 다음 후보가 전체 재생의 m+1 번째와 같고, 그때까지 결정이 같음
    order = [x for x in seen if x != "E00"]
    ok, bad = True, []
    for m in (0, 3, 11, 12, len(order) - 1):
        avail = set(order[:m])
        st2, fin2, need2 = e2fs.replay("bw", lambda e: fake_score(e) if (e["id"] == "E00" or e["id"] in avail) else None, 10)
        good = need2 is not None and need2["id"] == order[m] and st2 == steps[:len(st2)]
        ok &= good
        if not good:
            bad.append(m)
    st3, fin3, need3 = e2fs.replay("bw", so_all, 10)
    ok &= need3 is None and st3 == steps and fin3 == fin
    check("(d) 단위 시험: 이어하기(일부 결과만 있음) -> 다음 후보·그때까지 결정이 전체 재생과 같음, 다시 재생해도 같은 결정", ok,
          f"후보 순서 {len(order)}개, 실패 위치 {bad}")
    # ae4 표 입력 = 그 시점 기준 모델 입력, 뒤로 단계에서 고정
    fs12 = [s for s in fw if s["bundle"] == "ae4"][0]
    base_codes = [s["bundle"] for s in fw[:11] if s["decision"] == "keep"]
    e12 = e2fs.make_exp("FS12", "forward", fs12["base_id"], base_codes + ["ae4"], "ae4", 10, e2fs.feats_of(base_codes, 10))
    ok = e12["feats"]["ae_tab"] == e2fs.feats_of(base_codes, 10) and "ae_tab" not in e12["feats"]["ae_tab"]
    ebw = e2fs.make_exp("BW09", "backward", "BW01", [c for c in kept if c not in ("spi", "food")], "food", 10, fin["ae_tab"])
    ok &= ebw["feats"]["ae_tab"] == e2fs.feats_of(base_codes, 10)
    doc = e2fs.selection_doc(10, steps, fin, "끝", ["forward", "backward"])
    keys = ["rule", "val_T_range", "weather_L", "k_star", "seeds", "start_base_id", "steps", "final"]
    skeys = ["phase", "step", "round", "bundle", "bundle_name", "n_cols", "candidate_id", "base_id", "base_score", "candidate_score",
             "decision", "decision_ko", "new_base_id"]
    ok &= all(k in doc for k in keys) and all(all(k in s for k in skeys) for s in doc["steps"]) and set(doc["final"]) >= {"model_id", "bundles", "score"}
    ok &= doc["val_T_range"] == ["2024-07-01", "2025-05-01"] and doc["start_base_id"] == "E00"
    e02 = e2fs.make_exp("FS02", "forward", "E00", ["soilmed"], "soilmed", 10)
    ok &= "3칸 3칸" not in e02["desc"] and e02["desc"].startswith("FS02 = E00 + 토양 중앙값 3칸(")
    check("(d) ae4 표 AE 입력 = FS12 시점 기준 입력(뒤로 단계에서도 같은 입력 = 고정), selection_FS.json 칸(약속한 키 이름), FS02 설명 칸 수 중복 없음", ok,
          f"묶음 코드 {[b['code'] for b in E.FS_BUNDLES]}")


def run_d():
    import numpy as np
    import e2cfg as E
    import e2ae
    import e2fs
    from e2data import Ctx2
    from e2models import FeatureBuilder
    res = []

    def check(n, ok, d=""):
        res.append((n, bool(ok), str(d)))

    codes = [b["code"] for b in E.FS_BUNDLES]
    check("(d) 단계 선택 묶음 12개 = spi·soilmed·food·eco_gw_haccp·rg·g123·wxanom·hist_core·hist_1y·woe·n1·ae4 (칸 수 n)",
          codes == ["spi", "soilmed", "food", "eco_gw_haccp", "rg", "g123", "wxanom", "hist_core", "hist_1y", "woe", "n1", "ae4"],
          str([(b["code"], b["n"]) for b in E.FS_BUNDLES]))
    check("(d) STAGES: 'fs'·'bw' 가 'tables'(끝) 바로 앞, 'cum' 없음, CUM 실험 정의 없음",
          E.STAGES[-3:] == ["fs", "bw", "tables"] and "cum" not in E.STAGES and not any(e["id"].startswith("CUM") for e in E.EXPERIMENTS),
          str(E.STAGES[-4:]))
    fs_unit_tests(check)
    _patch_ae(e2ae, E)
    ctx = Ctx2(False, _log, use_cache=False)
    smp = _samples(ctx, E, seed=29, n=1000)
    cache = {}

    def mats(feats):
        key = json.dumps(feats, sort_keys=True, ensure_ascii=False, default=str)
        if key not in cache:
            fb = FeatureBuilder(ctx, feats, seed=42)
            cache[key] = (fb.names, list(fb.cat_names), {sp: (fb.num(sp, pos, hh), fb.cats(sp, pos)) for sp, (pos, hh) in smp.items()})
        return cache[key]

    def compare(tag, big_f, small_f, n_expect):
        """big = small + 묶음: small 칸은 이름으로 맞춰 big 과 비트 단위 같고, 새 칸 수 = n_expect, 이름 중복 없음."""
        bn, bc, bm = mats(big_f)
        sn, sc, sm = mats(small_f)
        new_n = [x for x in bn if x not in sn]
        new_c = [x for x in bc if x not in sc]
        ok = (len(set(bn)) == len(bn) and set(sn) <= set(bn) and set(sc) <= set(bc) and len(new_n) + len(new_c) == n_expect)
        for sp in smp:
            Xb, Cb = bm[sp]
            Xs, Cs = sm[sp]
            ok &= Xb[:, [bn.index(x) for x in sn]].tobytes() == Xs.tobytes()
            if Cs is not None:
                ok &= np.array_equal(np.asarray(Cb)[:, [bc.index(x) for x in sc]].astype(str), np.asarray(Cs).astype(str))
        check(tag, ok, f"칸 {len(bn)}+{len(bc)} vs {len(sn)}+{len(sc)}, 새 칸 {len(new_n) + len(new_c)} (기대 {n_expect}) {(new_n or new_c)[:2]}")
        return new_n, bm, bn

    k = K_FIX
    # 전진: E00 + 묶음 하나(ae4 제외 - 표 입력이 필요해 아래에서)
    for b in E.FS_BUNDLES:
        if b["code"] == "ae4":
            continue
        compare(f"(d) 전진 후보 E00 + {b['code']} = E00 행렬 + 묶음 칸(이름으로 맞춰 비트 단위)", e2fs.feats_of([b["code"]], k), e2fs.feats_of([], k), b["n"])
    # 중간 기준 + 묶음(ae4 포함: 표 입력 = 그 기준 입력)
    B1 = ["spi", "food", "hist_core"]
    for b in E.FS_BUNDLES:
        if b["code"] in B1:
            continue
        tab = e2fs.feats_of(B1, k) if b["code"] == "ae4" else None
        compare(f"(d) 전진 후보 [{'+'.join(B1)}] + {b['code']} = 기준 행렬 + 묶음 칸", e2fs.feats_of(B1 + [b["code"]], k, tab), e2fs.feats_of(B1, k), b["n"])
    # 뒤로: 12묶음 전부에서 하나씩 뺌(ae4 표 입력 = 나머지 11묶음 기준 입력으로 고정)
    tab_all = e2fs.feats_of([c for c in codes if c != "ae4"], k)
    full = e2fs.feats_of(codes, k, tab_all)
    for b in E.FS_BUNDLES:
        rest = [c for c in codes if c != b["code"]]
        compare(f"(d) 뒤로 후보 (12묶음 − {b['code']}) = 기준 행렬 − 묶음 칸", full, e2fs.feats_of(rest, k, tab_all if "ae4" in rest else None), b["n"])
    # ae4 표 입력: 기준 모델 입력(추가 범주 aw_label 포함), 캐시 구분
    htr = ctx.h_fit(42)["train"]
    tf = e2fs.feats_of(["food", "hist_core"], k)
    ti = e2ae.TabInput(ctx, k, htr, feats=tf)
    fbt = FeatureBuilder(ctx, tf)
    ok = (ti.n_num == len(fbt.names) and ti.extra_cats == ["aw_label"] and ti.codes("train").shape[1] == 5
          and np.array_equal(ti.codes("val")[:, :4], ctx.sp["val"]["cats"]) and ti.X.shape == (len(htr), len(fbt.names)))
    lab = np.asarray(fbt.cats("test", np.arange(len(ctx.sp["test"]["idx"]))))[:, 4].astype(str)
    lv = np.unique(np.concatenate([np.asarray(fbt.cats(sp, np.arange(len(ctx.sp[sp]["idx"]))))[:, 4].astype(str) for sp in ("train", "val", "test")]))
    ok &= np.array_equal(lv[ti.codes("test")[:, 4]], lab)
    ctx.ae_dir = Path("/tmp/selftest_L100_never_written")
    f_def, _ = e2ae._cache_files(ctx, "AEt", 42)
    f_tab, _ = e2ae._cache_files(ctx, "AEt", 42, tf)
    f_seq, _ = e2ae._cache_files(ctx, "AEs", 42, tf)
    f_seq0, _ = e2ae._cache_files(ctx, "AEs", 42)
    ok &= f_def != f_tab and f_seq == f_seq0 and e2ae.tab_sig(tf) == e2ae.tab_sig(dict(tf)) and e2ae.tab_sig(tf) != e2ae.tab_sig(e2fs.feats_of(["food"], k))
    m_tab = e2ae.meta(ctx, "AEt", 42, k, tf)
    m_def = e2ae.meta(ctx, "AEt", 42, k)
    ok &= m_tab["a3_feats"] is None and m_tab["tab_feats"] == tf and "tab_feats" not in m_def and m_def["a3_feats"] == e2ae.a3_feats(k)
    check("(d) ae4 표 AE 입력 = 기준 모델 입력(숫자 칸 + 범주 4 + aw_label 번호), 점수 캐시 = 입력 구분값으로 따로(시퀀스는 공유), 메타 구분",
          ok, f"숫자 {ti.n_num}, 범주 {ti.codes('train').shape[1]} ({ti.extra_cats}), 파일 {f_tab.name} / {f_def.name}")
    del ti
    # 묶음 단독 추가(베이스라인 기준): E00_P·E00_Q = E00 행렬 + 묶음 칸(비트 단위)
    import e2fg
    e00 = dict(E.EXP_BY_ID["E00"]["feats"])
    f0 = FeatureBuilder(ctx, _resolved(E, "E00"), seed=42)
    for sid, blocks in SOLO.items():
        e = E.EXP_BY_ID[sid]
        okd = (e["stage"] == ("p" if sid == "E00_P" else "q") and e["model"] == "cat" and e["plan"] == "묶음 단독 추가(베이스라인 기준)"
               and E.exp_seeds(e, False) == [42, 43, 44] and e["feats"] == dict(e00, ext=blocks))
        check(f"(d) {sid}: 정의(단계·CatBoost·시드 3·계획서 대응·입력 = E00 + {blocks})", okd, e["desc"])
        fs = FeatureBuilder(ctx, _resolved(E, sid), seed=42)
        nb_ = e2fg.ext_names(blocks)
        ok = fs.names == f0.names + nb_ and fs.cat_names == f0.cat_names and len(nb_) == (3 if sid == "E00_P" else 19)
        fr = FeatureBuilder(ctx, dict(_resolved(E, "E00"), ext=blocks), seed=42)
        for sp, (pos, hh) in smp.items():
            Xs, X0 = fs.num(sp, pos, hh), f0.num(sp, pos, hh)
            ok &= Xs[:, :len(f0.names)].tobytes() == X0.tobytes()
            ok &= np.array_equal(np.asarray(fs.cats(sp, pos)).astype(str), np.asarray(f0.cats(sp, pos)).astype(str))
            ok &= Xs[:, len(f0.names):].tobytes() == fr.num(sp, pos, hh)[:, len(f0.names):].tobytes()
        check(f"(d) {sid}: 행렬 = E00 행렬 + 묶음 칸 {len(nb_)}개(비트 단위, 범주 같음)", ok, f"칸 {len(fs.names)} = {len(f0.names)} + {nb_[:3]}..")
    # 추가 변수 전부(xall): ALL_A·ALL_B·E00_N2·E00_N0
    fa, fbb = _resolved(E, "ALL_A"), _resolved(E, "ALL_B")
    want_a = e2fs.feats_of(codes, k, e2fs.feats_of([c for c in codes if c != "ae4"], k))
    okd = (fa == want_a and fbb == dict(want_a, nb=["n1", "n2", "n0"]) and fbb["ae_tab"] == fa["ae_tab"] == want_a["ae_tab"]
           and "n2" not in fa["ae_tab"].get("nb", []) and "n0" not in fa["ae_tab"].get("nb", [])
           and _resolved(E, "E00_N2") == dict(e00, nb=["n2"], k=k) and _resolved(E, "E00_N0") == dict(e00, nb=["n0"], k=k)
           and all(E.EXP_BY_ID[i]["stage"] == "xall" and E.EXP_BY_ID[i]["model"] == "cat" and E.exp_seeds(E.EXP_BY_ID[i], False) == [42, 43, 44]
                   for i in ("ALL_A", "ALL_B", "E00_N2", "E00_N0"))
           and E.STAGES.index("xall") < E.STAGES.index("fs"))
    check("(d) xall 정의: ALL_A = E00 + 단계 선택 12묶음 전부(표 AE 입력 = 나머지 11묶음 입력), ALL_B = ALL_A + n2·n0(AE 입력 같음), "
          "E00_N2·E00_N0 = E00 + 묶음 하나, 단계 'xall'·CatBoost·시드 3", okd,
          f"ALL_A 묶음 ext {fa['ext']} hist {fa['hist']} woe {fa['woe']} nb {fa['nb']} ae {fa['ae']}")
    n_all = sum(b["n"] for b in E.FS_BUNDLES)
    compare(f"(d) ALL_A = E00 행렬 + 12묶음 칸 {n_all}개(이름으로 맞춰 비트 단위)", fa, _resolved(E, "E00"), n_all)
    compare("(d) ALL_B = ALL_A 행렬 + 그날 제품명 칸 521개(n2 21 + n0 500)", fbb, fa, 21 + 500)
    fa0 = _resolved(E, "ALL_A0")
    check("(d) ALL_A0 정의 = ALL_A 에서 AE·VAE 점수 4칸(ae·ae_tab)만 뺀 입력, 단계 'xall'·시드 3",
          fa0 == {kk: vv for kk, vv in fa.items() if kk not in ("ae", "ae_tab")} and E.EXP_BY_ID["ALL_A0"]["stage"] == "xall"
          and E.exp_seeds(E.EXP_BY_ID["ALL_A0"], False) == [42, 43, 44], json.dumps(fa0, ensure_ascii=False)[:200])
    compare("(d) ALL_A = ALL_A0 행렬 + AE·VAE 점수 4칸(= ALL_A0 = ALL_A − AE 4칸, 이름으로 맞춰 비트 단위)", fa, fa0, 4)
    compare("(d) E00_N2 = E00 행렬 + n2 21칸", _resolved(E, "E00_N2"), _resolved(E, "E00"), 21)
    compare("(d) E00_N0 = E00 행렬 + n0 500칸", _resolved(E, "E00_N0"), _resolved(E, "E00"), 500)
    return res


# ================================================================ (e) AE·VAE 점수 규칙 v2 (GPU, 시험 실행 자료)
def run_e():
    """점수 규칙 v2(사용자 승인 2026-10-19) 점검 - 시험 실행(smoke) 자료(학습 불검출 2만 + 검출 전부)로 AEt·AEs."""
    import numpy as np
    import torch
    import e2cfg as E
    import e2ae
    from e2data import Ctx2
    res = []

    def check(n, ok, d=""):
        res.append((n, bool(ok), str(d)))

    ctx = Ctx2(True, _log)
    k = K_FIX
    seed = 42
    htr = ctx.h_fit(seed)["train"]
    y = ctx.sp["train"]["y"]
    dev = torch.device("cuda")
    import tempfile
    for kind in ("AEt", "AEs"):
        tab = e2ae.is_tab(kind)
        S, info, dbg = e2ae.compute_scores(ctx, kind, seed, k, _log, debug=True)
        fold = dbg["fold_all"]
        nf = E.AE["n_folds"]
        ok = np.array_equal(fold, e2ae.fold_assign_all(len(y), seed))
        tg = np.concatenate([dbg["models"][m]["target"] for m in range(nf)])
        ok &= np.array_equal(np.sort(tg), np.arange(len(y)))
        for m in range(nf):
            d = dbg["models"][m]
            ok &= bool(np.all(fold[d["target"]] == m) and np.all(fold[d["rows"]] != m) and np.all(y[d["rows"]] == 0)
                       and np.intersect1d(d["rows"], d["target"]).size == 0
                       and np.array_equal(np.sort(d["rows"]), np.flatnonzero((y == 0) & (fold != m))))
        check(f"(e) {kind}: 학습 행 전부(검출 포함)가 정확히 한 겹의 점수 대상, 그 겹 모델의 학습 행 = 겹 밖 불검출 전부(자기 행 안 봄), "
              "겹 배정 = fold_assign_all(정답 안 씀)", ok, f"학습 행 {len(y):,}, 검출 {int(y.sum())}, 겹 {nf}")
        inp = e2ae.TabInput(ctx, k, htr) if tab else e2ae.SeqInput(ctx, htr)
        nets, ok = [], True
        with e2ae.deterministic_torch():
            for m in range(nf):
                d = dbg["models"][m]
                net = e2ae.make_net(kind, *d["net_args"]).to(dev)
                net.load_state_dict(d["state"])
                nets.append(net)
                dd = inp.device_train(d["prep"], dev)
                sc = e2ae.score_rows(net, inp.batch, dd, d["target"], dev)
                ok &= bool(np.array_equal(sc, S["train"][d["target"]]))
                del dd
            check(f"(e) {kind}: 학습 행 점수 = 자기 겹 모델(저장 가중치)로 다시 계산한 값과 비트 같음(검출 행 포함)", ok)
            ok2, cnt = True, 0
            for sp in ("val", "test"):
                n = len(ctx.sp[sp]["idx"])
                for h in (1, 30, 60):
                    acc = np.zeros(n, dtype=np.float64)
                    for m in range(nf):
                        d = dbg["models"][m]
                        if tab:
                            Xh = inp.fb.num(sp, np.arange(n), h)
                            de = {"Z": torch.from_numpy(inp.z(Xh, d["prep"])).to(dev),
                                  "C": torch.from_numpy(inp.cat_idx(inp.codes(sp), d["prep"])).to(dev)}
                        else:
                            de = inp.device_eval(inp.device_train(d["prep"], dev), sp, h)
                        acc += e2ae.score_rows(nets[m], inp.batch, de, np.arange(n), dev)
                    ok2 &= bool(np.array_equal((acc / nf).astype(np.float32), S[sp][h - 1]))
                    cnt += 1
            check(f"(e) {kind}: 검증·시험 점수 = 겹 모델 {nf}개 점수 평균(h 1·30·60, 비트 단위)", ok2, f"{cnt}개 비교")
        tbl = info["fold_label_table"]
        pos_f = [r[1] for r in tbl]
        exp_pos = y.sum() / nf
        ok = (set(info["models"]) == {str(m) for m in range(nf)} and "full" not in info["models"] and info["score_rule"] == e2ae.SCORE_RULE
              and sum(pos_f) == int(y.sum()) and sum(r[0] for r in tbl) == int((y == 0).sum()))
        check(f"(e) {kind}: 점수 출처와 정답 무관 - 학습 구간 점수의 겹 모델 x 정답 교차표(검출 행도 겹 모델 5개로 고루), 전체 모델 없음, 규칙 v2 기록",
              ok, f"교차표 [불검출, 검출] 겹별 {tbl} (검출 기대 {exp_pos:.1f}/겹)")
        S2, _ = e2ae.compute_scores(ctx, kind, seed, k, _log)
        check(f"(e) {kind}: 같은 시드 두 번 계산 = 같은 값(학습·검증·시험 비트 단위)", all(np.array_equal(S[sp], S2[sp]) for sp in S))
        with tempfile.TemporaryDirectory() as td:
            ctx.ae_dir = Path(td)
            f1, _ = e2ae._cache_files(ctx, kind, seed)
            m1 = e2ae.meta(ctx, kind, seed, k)
            ok = f1.name.endswith("_v2.npz") and m1.get("score_rule") == e2ae.SCORE_RULE
            check(f"(e) {kind}: 캐시 파일 이름 '_v2'·메타 score_rule(옛 v1 점수와 섞이지 않음)", ok, f1.name)
            ctx.ae_dir = None
        del nets
        torch.cuda.empty_cache()
    return res


# ================================================================ (c) T 통합 선정 함수 (이 PC, 결과 읽기만)
def run_c(check):
    import numpy as np
    import pandas as pd
    import e2cfg as E
    import e2tables
    import e2fg
    from e2eval import val_tint_pr_auc
    res_dir = E.EXP2_DIR / "results"
    an = E.EXP2_DIR / "analysis" / "T기준_통합"
    sd = pd.read_csv(an / "시드별_T통합.csv", encoding="utf-8-sig")
    ex = pd.read_csv(an / "실험기록_T통합.csv", encoding="utf-8-sig")
    sd = sd[sd["구분"] == "2차"]
    ex = ex[ex["구분"] == "2차"]
    col = "검증 60일평균 PR-AUC"
    worst, n_run, mine = 0.0, 0, {}
    for eid in C_IDS:
        rows = sd[sd["실험 ID"] == eid]
        vals = []
        for _, r in rows.iterrows():
            v = val_tint_pr_auc(res_dir / f"{eid}__s{int(r['시드'])}")["mean60"]
            vals.append(v)
            worst = max(worst, abs(v - float(r[col])))
            n_run += 1
        mine[eid] = float(np.mean(vals))
        er = ex[ex["실험 ID"] == eid]
        d = abs(mine[eid] - float(er[col].iloc[0]))
        check(f"(c) {eid}: T 통합 60일평균 PR-AUC 시드 평균 = 실험기록_T통합.csv (시드 {len(vals)}개)", len(er) == 1 and d <= 1e-12 and len(vals) > 0,
              f"{mine[eid]:.12f} vs {float(er[col].iloc[0]):.12f} (차이 {d:.2e})")
    check("(c) 실행별 값 = 시드별_T통합.csv (최대 차이 1e-12 이하)", worst <= 1e-12 and n_run == sum((sd["실험 ID"] == i).sum() for i in C_IDS),
          f"실행 {n_run}개, 최대 차이 {worst:.2e}")
    # tint 규칙으로 고른 결과(기록만, 선정 파일은 쓰지 않음)
    saved = (E.SELECT, e2tables.jdump)
    E.SELECT = "tint"
    e2tables.jdump = lambda *a, **k: None
    try:
        P = {"results": res_dir}
        sk = e2tables.select_k(P, False)
        sa = e2tables.select_A(P, False)
        _, sf = e2fg.combo_experiments(P, False, None, strict=False)
    finally:
        E.SELECT, e2tables.jdump = saved
    check("(c) tint 선정(50일 결과, 기록만): k*·A 최고·F·G 높은 묶음", True,
          f"k* = {sk['k_star']} ({[(r['k'], round(r['val_tint_pr_auc_mean60'], 6)) for r in sk['ranking']]}), "
          f"A 최고 = {sa['best']} ({[(r['exp'], round(r['val_tint_pr_auc_mean60_mean'], 6)) for r in sa['ranking']]}), "
          f"F·G E00 {sf.get('E00_select_score_mean')} 높은 묶음 {sf.get('selected')}")


# ================================================================ 실행
def _sub(args_list, env_extra, src=None):
    env = dict(os.environ)
    for k in ("EXP2_L", "EXP2_SELECT"):
        env.pop(k, None)
    env.update(env_extra)
    env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve())] + args_list + (["--src", str(src)] if src else [])
    _log(f"하위 실행: {' '.join(args_list)} env {env_extra} src {src or HERE}")
    r = subprocess.run(cmd, env=env)
    if r.returncode:
        raise RuntimeError(f"하위 실행 실패({r.returncode}): {args_list}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b,d")
    ap.add_argument("--old-src", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--dump", default=None, choices=["a", "b", "d", "e"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.dump:                                            # 하위 실행: 지정한 코드판으로 계산해 파일로
        _set_src(args.src or HERE)
        import e2cfg  # noqa: F401  (1차 src 를 sys.path 에 추가)
        _log(f"코드판 {Path(sys.modules['e2cfg'].__file__).parent} L={sys.modules['e2cfg'].L} SELECT={getattr(sys.modules['e2cfg'], 'SELECT', '(없음)')}")
        if args.dump == "a":
            dump_a(args.out)
        else:
            res = run_b() if args.dump == "b" else (run_d() if args.dump == "d" else run_e())
            Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    sys.path.insert(0, str(HERE))
    import e2cfg as E
    out_txt = E.EXP2_DIR / "logs" / "selftest_L100.txt"
    tmp = E.EXP2_DIR / "logs" / "selftest_L100_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    lines, fails = [], []

    def check(name, ok, detail=""):
        line = f"[{'통과' if ok else '실패'}] {name} {detail}"
        lines.append(line)
        _log(line)
        if not ok:
            fails.append(name)

    t0 = time.time()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    lines.append(f"selftest_L100 {time.strftime('%Y-%m-%d %H:%M:%S')} parts={parts} python {sys.version.split()[0]}")
    if "a" in parts:
        if not args.old_src:
            raise SystemExit("--parts a 에는 --old-src(고치기 전 src 사본) 필요")
        _sub(["--dump", "a", "--out", str(tmp / "a_new.npz")], {}, HERE)
        _sub(["--dump", "a", "--out", str(tmp / "a_old.npz")], {}, Path(args.old_src))
        compare_a(tmp / "a_new.npz", tmp / "a_old.npz", check)
    if "b" in parts:
        _sub(["--dump", "b", "--out", str(tmp / "b.json")], {"EXP2_L": "100"}, HERE)
        for n, ok, d in json.loads((tmp / "b.json").read_text(encoding="utf-8")):
            check(n, ok, d)
    if "d" in parts:
        _sub(["--dump", "d", "--out", str(tmp / "d.json")], {"EXP2_L": "100", "EXP2_SELECT": "tint"}, HERE)
        for n, ok, d in json.loads((tmp / "d.json").read_text(encoding="utf-8")):
            check(n, ok, d)
    if "e" in parts:                                          # GPU 필요(CUDA_VISIBLE_DEVICES 로 GPU 를 보이게)
        _sub(["--dump", "e", "--out", str(tmp / "e.json")], {"EXP2_L": "100"}, HERE)
        for n, ok, d in json.loads((tmp / "e.json").read_text(encoding="utf-8")):
            check(n, ok, d)
    if "c" in parts:
        run_c(check)
    lines.append(f"끝: 점검 {len([x for x in lines if x.startswith('[')])}개, 실패 {len(fails)}개, {time.time() - t0:.0f}s")
    _log(lines[-1])
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    for p_ in tmp.glob("*"):
        try:
            p_.unlink()
        except Exception:
            pass
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
