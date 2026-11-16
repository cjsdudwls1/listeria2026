# -*- coding: utf-8 -*-
"""s_final.py — 최종 모델 고르기·최종 보고 — 계획서 요약 20·24, §6 '최종 모델 고르기', §1-3 마지막 줄, 부록 A-10(리서치팀 보고),
A-12(최종 모델 보조 표·좌표 차이 민감도), A-15(MLflow '최종' tag·'사슬' run artifact), B-5(섞기 씨앗).

모드(설정 configs/s_final.json 하나):
  (기본)        고르기: 사슬 상태(단계 8 판정 뒤 마지막 state json — stage4.load_state 꼴)의 현재 최고 판과 1차 베이스라인
                (1-3 결정) 중 CV 1위(evaluate.select_with_tie_check — 동점 §1-2, test 안 읽음) → final_decision.json
  --importance  최종 모델 안 순열 중요도(부록 A-10·A-5 — 학습 없음, fold validation 쌍만, M = 10, h_block,
                씨앗 [perm_seed, f, 20000, 칼럼 번호, r]·[.., 20000, 0, r]·[.., 21000, 그룹 번호, r]) → importance/
  --report --reason "…"  보고(test 정답은 여기서만 — evaluate.report_permit): report/ 아래
                change_table.csv(1차 베이스라인 → 3-0 1위 → 채택한 기법마다 → 튜닝 → 최종 모델, val·test·변화·fold 1 알림)
                final_test_scores.json·final_h_metrics.csv(최종 모델 test 전체 지표)
                x1_compare.csv(① 당일 예측 대 최종 모델 — h = 1 CV·test PR-AUC, threshold 지표 — 설정 x1_dir 이 있을 때)
                aux_product_route.csv(품목 묶음 × 국내·해외)·aux_unit_tertile.csv(단위 시료 수 3분위)
                research_table.csv·research_notes.json(리서치팀 보고 — 부록 A-10)
                input_variables.csv(최종 모델 입력 변수 표)·coord_sensitivity.json('하지 않음')
                + MLflow: '사슬' run 에 변화 표, 최종 판 run 에 tag '최종' = '예'

실행(서버 — 검수 뒤 지휘자, 단계 8 판정 뒤):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s_final.py $A --out ~/cyj_listeria_final/results/final --config configs/s_final.json
  $P scripts/s_final.py $A --out ~/cyj_listeria_final/results/final --config configs/s_final.json --importance
  $P scripts/s_final.py $A --out ~/cyj_listeria_final/results/final --config configs/s_final.json --report --reason "최종 보고"
  (설정 importance_parts 가 있으면 --importance 한 줄이 부분 하위 프로세스 `--importance --part <부분>` 를 한꺼번에 띄우고
   모두 끝나면 합침 — 구현 세부 (9))

구현 세부(계획서에 없음 — 담당자 확인):
 (1) 판 폴더 찾기 순서: 설정 variant_dirs[이름] → 사슬 기록 줄의 *_dir 칸 → 사슬 상태 BestDef·구성의 dir → 설정 search_dirs
     (단계 결과 폴더들) 아래 <이름>/done.json. 예측이 없는 기록 판(3-0b 끝 판·2-0)은 source.json 'dir'·spec extra.source.dir
     의 원 판 예측을 씀.
 (2) 변화 표 줄 = 사슬 기록(jsonl) 차례: '출발' 줄(step '3-0' = 3-0 1위 — v78, 옛 순서 3-0b 의 '출발' 줄은 안 씀), judgement '채택' 줄, judgement 칸이 없는 줄(stage4.judge_step 꼴)은
     parts 의 채택된 후보마다(없으면 after ≠ before 일 때 after). 단계가 '8' 로 시작하면 이름표 '튜닝'. 마지막에 '최종 모델' 줄.
     뒤에 덧붙인 judgement '채택 취소' 줄(담당자 결정)이 있으면 그 단계의 앞선 채택 줄을 변화 표에서 뺌(기록은 그대로).
     fold 1 알림 = 채택 줄에서 fold 1 단독 점수가 직전 줄보다 4자리 반올림으로 오르지 않으면 '채택했는데 fold 1 단독으로
     안 오름'(§6 최종 모델 고르기 — 참고용, 규칙은 그대로).
 (3) 품목 묶음: p2 를 t_prdlst_cl.hrnk_prdlst_cd 로 올라가 최상위(1단계) 코드 → 식물성 = 농산물, 동물성 = 2단계(최상위 바로
     아래 조상)가 축산물·수산물이면 그 이름, 그 밖 동물성(기타동물) = 그 밖, 가공식품 = 가공식품, 기준규격외 = 기준규격외,
     나머지 1단계(첨가물·건강기능식품·기구 등)·분류표에 없는 p2 = 그 밖, p2 '코드없음' = 코드없음. 이름은 t_prdlst_cl.kor_nm.
 (4) 단위 3분위: 단위 = (sgg, p2, p3), 시료 수 = d ≤ 2025-06-30 시료, 경계 = test 쌍에 나오는 단위들(단위 하나 = 1번)의 시료 수
     1/3·2/3 분위(numpy 선형), 하위 ≤ q1 < 중간 ≤ q2 < 상위.
 (5) 보조 표 점수 = metrics.score_report + threshold_metrics(최종 모델 CV threshold), 덜 찬 달 쌍은 뺌(본 점수와 같음).
 (6) 순열 중요도 노드: 트리·딥러닝 단독 = stage3.importance_nodes 방식(저장 모델·교차 N), 단계 4~6 판(extra.s4) 트리 =
     Stage4Wiring.make_inputs → fit(그 fold train 행 = stage4.plan_train_rows) → 저장 칼럼과 대조, 칼럼 소속 = 등록부 +
     'catenc_<칼럼>' 은 원 칼럼 단위, 단계 4~6 딥러닝 판(② 제외 — Stage4Wiring 이 wiring 의 DLInputs 를 그대로 씀) =
     stage3.importance_nodes(저장 전처리기·모델 — 섞기 전 점수가 판 fold 점수와 1e-6 안인지 대조), 앙상블·씨앗 평균 =
     importance.Mean. 값 모델(VAE 이상 점수·④ 6트리)이 조립 객체 안에서 값을 만드는 판·
     ② 60칸·3-1 판(사슬 상태 anomaly)은 지원 안 함 → 멈추지 않고 importance/unsupported.json 에 이유를 남김.
 (8) ① 비교(§6 추가 판 ①): h = 1 CV PR-AUC = fold 마다 그 판 fold 예측의 h == 1 쌍만으로 metrics.pr_auc(u_tie) → 평균,
     4자리 반올림으로 ① 이 오르면 '①_better' = True(최종 모델 고르기와 무관 — 비교만). test = refit 예측 h == 1 쌍(덜 찬 달 뺌),
     threshold 지표 = 각 판 자기 CV threshold.
 (7) 입력 변수 표 '정의' = '§5 표 <ID>'(등록부에 문장 정의 없음), 원천·갱신 주기 = ID 별 고정 문구 + 칼럼 window 끝.
 (9) 순열 중요도 나눠 돌리기·이어 하기(설정 importance_parts — 없으면 지금처럼 한 프로세스): 부분 = 'fold1' 또는
     'fold3:1-30'(lst/imp_parts — 세 fold, fold 마다 h 1~60 을 겹침·빈틈 없이). 본 프로세스(--importance)는 남은 h 가 있는
     부분만 `--importance --part` 하위 프로세스로 한꺼번에 띄우고(출력 = importance_logs/<부분>.log) 모두 끝나면 합쳐 같은
     importance/(records·summary·meta — seconds 만 = h 계산 초의 합)를 쓰고 상태 폴더 importance_state/ 를 지움. 부분은
     h 하나 끝날 때마다 importance_state/<fold>/hNN.npz(계산 규칙·서명 = lst/importance 구현 세부 h, 씨앗은 위 그대로),
     부분 정보(섞기 전 점수 대조 값·허용 오차 또는 '지원 안 함' 이유) = importance_state/parts/<부분>.json. 서명 tag =
     최종 판 이름·폴더·final_decision.json md5·시료 파일 md5·코드 md5(바뀌면 이어 하지 않고 멈춤). 부분은 h 마다 자원
     점검(check_resources — 기본 한 프로세스 길은 예전처럼 점검 없음), GPU 상한은 기본 길과 같은 gpu_memory_fraction.
     멈춘 부분이 있으면 실험 로그 'stopped' 뒤 멈춤 — 같은 명령을 다시 하면 남은 h 만.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import imp_parts as ipt  # noqa: E402
from lst import scope as lst_scope  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import metrics as mt  # noqa: E402
from lst import registry as rg  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage3 as s3  # noqa: E402
from lst import stage4 as s4  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = "최종"
EXPERIMENT = "listeria-2차"
DECISION_FILE = "final_decision.json"
REPORT_DIR = "report"
IMPORTANCE_DIR = "importance"
SCHEMA = {
    "plan": cf.Field(str),
    "stage": cf.Field(str, choices=(STAGE,)),
    "order": cf.Field(str, choices=("2차",)),
    "experiment": cf.Field(str, choices=(EXPERIMENT,)),
    "samples": cf.Field(str),
    "incomplete_months": cf.Field(str),
    "experiment_log": cf.Field(str),
    "use_mlflow": cf.Field(bool),
    "mlflow_uri": cf.Field(str),
    "code_reviewed": cf.Field(bool),
    "n_jobs": cf.Field(int),
    "seed": cf.Field(int),
    "decision_1_3": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "chain_state": cf.Field(str),            # --inputs 아래 마지막 사슬 상태(단계 8 판정 뒤)
    "chain_log": cf.Field(str),
    "decision_3_0b": cf.Field(str),          # --inputs 아래 3-0b_decision.json(v78 — 1등 변수 빼기, s3/3-0b-best)
    "dir_2_0": cf.Field(str),                # --inputs 아래 2-0 판 폴더(기준 = 파생변수 없는 판)
    "dir_2_R": cf.Field(str),                # --inputs 아래 2-R 결과 폴더(2-R1 ~ 2-R10·2-R1_importance)
    "search_dirs": cf.Field(list, item_types=str),
    "variant_dirs": cf.Field(dict, required=False),
    "prdlst_cl": cf.Field(str, required=False),   # --root 아래 t_prdlst_cl parquet
    "theil_targets": cf.Field(str, required=False),
    "importance_M": cf.Field(int),
    "x1_dir": cf.Field(str, required=False),   # --inputs 아래 ① 당일 예측 판 폴더(h = 1 쌍 예측) — 비교만
    "gpu_memory_fraction": cf.Field(float, required=False),
    "importance_parts": cf.Field(list, required=False, item_types=str),   # 구현 세부 (9) — 없으면 한 프로세스
    "context": cf.Field(dict),
}
SCHEMA = lst_scope.with_field(SCHEMA)   # 국내만 버전 scope 칸(없으면 지금 판)
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")

# 부록 A-10 대상(리서치 PDF p12 표 1~7번 + 축사)
RESEARCH_REF = {
    "t3": ("Murphy 2023", "p12 표 1번", "기준점 R 앞 3일(R−3 ~ R−1) 평균 기온"),
    "rain_lag2": ("Ivanek 2009", "p12 표 2번", "기준점 R 2일 전 강수량"),
    "wind_lag2": ("Pang 2017", "p12 표 3번", "기준점 R 2일 전 평균 풍속"),
    "rain_mean25": ("Pang 2017", "p12 표 4번", "기준점 R 앞 25일 평균 강수량"),
    "solar_23": ("Weller 2020", "p12 표 5번", "기준점 R 2·3일 전 일사량 평균"),
    "trange7": ("Chandler-Khayd 2023", "p12 표 6번", "기준점 R 앞 7일 일교차(최고 − 최저 기온) 평균"),
    "rh1": ("Murphy 2023", "p12 표 7번", "기준점 R 하루 전 습도"),
    "td1": ("Murphy 2023", "p12 표 7번", "기준점 R 하루 전 이슬점"),
    "L1": ("Weller 2020", "p8·p9", "시군구(해외는 나라) 전체 농장 수 10km²당"),
}
PDF = "리스테리아 모델적용방안.pdf"

ID_SOURCE = {   # 구현 세부 (7)
    "CAT5": ("시료(LIMS) 칼럼", "시료마다"), "TIME": ("계산(h·g)", "쌍마다"),
    "WEATHER": ("식정원 DB 기상(국내 시군구 일별·해외 나라 일별)", "매일"),
    "WX1": ("SPI(기상청 일자료·강수로 계산)", "매일"), "WX2": ("식정원 DB 기상 이동평균 차감", "매일"),
    "WX3": ("기상 + 식품 고정표(생장식)", "매일"), "WX4": ("기상·토양·식품 고정표(생장 범위)", "매일"),
    "WX5": ("식정원 DB 기상(단기 파생)", "매일"), "WX6": ("ASOS·NAS 습도·이슬점", "매일"), "WX7": ("d 의 달", "달마다"),
    "HIS1": ("LIMS 검출 이력", "연 1회(T 직전 06-30)"), "HIS2": ("LIMS 검사 이력", "연 1회(T 직전 06-30)"),
    "HIS3": ("LIMS 검출 이력", "연 1회(T 직전 06-30)"), "HIS4": ("LIMS 검출 이력", "연 1회(T 직전 06-30)"),
    "FD1": ("식품 고정표(t_aw_cl)", "고정"), "SO1": ("api_data_info.soil_ph_info·global_soil_ph", "매일(T−1)"),
    "X1": ("친환경 인증 면적 비율", "연 1회"), "X2": ("지하수 정기수질검사 적합률", "연 1회"), "L1": ("축사 농장 수", "고정"),
}


class FinalError(RuntimeError):
    pass


# ================================================================ 설정·판 폴더
def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--reason"):
        ap.add_argument(a)
    for a in ("--report", "--importance"):
        ap.add_argument(a, action="store_true")
    ns = ap.parse_known_args(list(argv))[0]
    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)
    return ns


def load_cfg(path) -> dict:
    cfg = dict(cf.load_config(path, SCHEMA))
    if cfg["seed"] != 42:
        raise cf.ConfigError("씨앗은 42")
    if "importance_parts" in cfg:
        ipt.parse_parts(cfg["importance_parts"])
    return cfg


def _done_dir(d: Path) -> Path | None:
    return d if (d / "done.json").exists() else None


def find_dir(name: str, *, cfg: Mapping, inputs_root: Path, state: Mapping | None = None,
             line: Mapping | None = None) -> Path:
    """구현 세부 (1)."""
    vd = cfg.get("variant_dirs") or {}
    if name in vd:
        d = resolve(inputs_root, vd[name])
        if _done_dir(d):
            return d
        raise FinalError(f"variant_dirs[{name}] 에 done.json 없음: {d}")
    for k in ("after_dir", "current_best_after_dir"):
        if line and line.get(k) and line.get("after") == name:
            d = _done_dir(resolve(inputs_root, line[k]))
            if d:
                return d
    if state:
        for key in ("current_best", "tree_target"):
            b = state.get(key) or {}
            if b.get("name") == name and b.get("dir"):
                d = _done_dir(resolve(inputs_root, b["dir"]))
                if d:
                    return d
            for m in b.get("members", []):
                if m.get("name") == name and m.get("dir"):
                    d = _done_dir(resolve(inputs_root, m["dir"]))
                    if d:
                        return d
    for sd in cfg["search_dirs"]:
        d = _done_dir(resolve(inputs_root, sd) / name)
        if d:
            return d
    raise FinalError(f"판 폴더를 못 찾음: {name}(variant_dirs·search_dirs 확인)")


def pred_dir(inputs_root: Path, vdir: Path) -> Path:
    """예측이 있는 폴더(기록 판 = 원 판) — 구현 세부 (1)."""
    if rn.pred_path(vdir, "refit").exists():
        return vdir
    if (vdir / "source.json").exists():
        return resolve(inputs_root, rn.read_json(vdir / "source.json")["dir"])
    spec = rn.read_json(vdir / "spec.json")
    src = (spec.get("extra") or {}).get("source")
    if isinstance(src, Mapping) and src.get("dir"):
        return resolve(inputs_root, src["dir"])
    raise FinalError(f"{vdir.name}: 예측 위치를 못 찾음")


def cv_of(vdir: Path) -> dict:
    return rn._load_scores(vdir / "cv_scores.json")


def candidate(name: str, vdir: Path) -> ev.Candidate:
    cv = cv_of(vdir)
    spec = rn.read_json(vdir / "spec.json")
    return ev.Candidate(name, float(cv["cv_band_weighted"]), int(spec["train_count"]), bool(cv["ties"]["many_ties"]))


def baseline_dir(cfg: Mapping, inputs_root: Path, name: str) -> Path:
    return resolve(inputs_root, cfg["dir_1_2"] if name.startswith("1-LSTM-") else cfg["dir_1_1"]) / name


# ================================================================ 고르기(CV 만)
def select_final(cfg: Mapping, inputs_root: Path, out: Path) -> dict:
    p = out / DECISION_FILE
    if p.exists():
        print(f"최종 고르기 이미 있음: {p}")
        return rn.read_json(p)
    spath = resolve(inputs_root, cfg["chain_state"])
    state = s4.load_state(spath)
    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    best = state["current_best"]
    bdir = resolve(inputs_root, best["dir"])
    base_name = dec["baseline"]
    base_d = baseline_dir(cfg, inputs_root, base_name)
    if best["name"] == base_name:
        raise FinalError("사슬 현재 최고 판 이름이 1차 베이스라인과 같음")
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    dirs = {best["name"]: bdir, base_name: base_d}

    def rescore(n):
        return float(ev.score_cv(rn.load_preds(pred_dir(inputs_root, dirs[n]), ev.CV_FOLDS), samples,
                                 rescore=True)["cv_band_weighted"])

    with ev.selection_guard():
        sel = ev.select_with_tie_check([candidate(best["name"], bdir), candidate(base_name, base_d)], rescore)
    final = sel["selected"]
    if final == best["name"]:
        fdef = best
        source = "사슬 마지막 현재 최고 판"
    else:
        spec = rn.read_json(base_d / "spec.json")
        rel = str(Path(cfg["dir_1_2"] if base_name.startswith("1-LSTM-") else cfg["dir_1_1"]) / base_name).replace("\\", "/")
        fdef = {"name": base_name, "kind": spec["kind"], "dir": rel, "replacement": False,
                "members": [{"name": base_name, "dir": rel, "anomaly": None}]}
        source = "1차 베이스라인"
    x1 = x1_compare_cv(cfg, inputs_root, samples, fdef) if cfg.get("x1_dir") else None
    decision = {"stage": STAGE, "final": final, "final_source": source, "final_def": fdef,
                "chain_last": best["name"], "baseline_1": base_name, "selection": sel,
                "x1": x1, "①_better": None if x1 is None else x1["①_better"],
                "chain_state_md5": cf.file_md5(spath), "alerts": list(sel["alerts"]),
                "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    out.mkdir(parents=True, exist_ok=True)
    cf.dump_scores_json(p, decision)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    log.append("decision", stage=STAGE, final=final, source=source, chain_last=best["name"], baseline=base_name)
    for a in sel["alerts"]:
        log.append("alert", stage=STAGE, message=a)
        print(f"[알림] {a}")
    print(f"최종 모델 = {final}({source})")
    return decision


# ================================================================ ① 당일 예측 비교(구현 세부 8)
def h1_cv(pdir: Path, samples) -> dict:
    fold = {}
    for f in ev.CV_FOLDS:
        pr = pd.read_parquet(rn.pred_path(pdir, f))
        pr = pr[pr["h"].to_numpy() == 1]
        P = ev.prepare_pairs(pr, samples, f, require_complete=False)
        fold[f] = float(mt.pr_auc(samples.cv_y(P.pos), P.pred, P.u()))
    return {"cv": float(mt.cv_mean(fold)), "fold": fold}


def x1_compare_cv(cfg: Mapping, inputs_root: Path, samples, fdef: Mapping) -> dict:
    xdir = resolve(inputs_root, cfg["x1_dir"])
    fdir = resolve(inputs_root, fdef["dir"])
    a = h1_cv(pred_dir(inputs_root, fdir), samples)
    b = h1_cv(pred_dir(inputs_root, xdir), samples)
    return {"final": fdef["name"], "x1": xdir.name, "x1_dir": cfg["x1_dir"], "final_h1": a, "x1_h1": b,
            "①_better": bool(mt.is_improvement(b["cv"], a["cv"]))}


def x1_compare_test(cfg, inputs_root: Path, decision: Mapping, fdir: Path, samples, truth, inc) -> pd.DataFrame:
    x = decision["x1"]
    rows = []
    for label, d, h1 in (("최종 모델", fdir, x["final_h1"]), ("① 당일 예측", resolve(inputs_root, x["x1_dir"]),
                                                              x["x1_h1"])):
        pdir = pred_dir(inputs_root, d)
        thr = float(cv_of(d)["threshold"]["threshold"])
        pr = pd.read_parquet(rn.pred_path(pdir, "refit"))
        pr = pr[pr["h"].to_numpy() == 1]
        P = ev.prepare_pairs(pr, samples, ev.TEST_FOLD, require_complete=False)
        y = truth.y_for(P.keys)
        if inc:
            keep = ~fd.incomplete_pair_mask(P.d_days, P.src_route, set(inc))
            P, y = P.take(keep), y[keep]
        tm = mt.threshold_metrics(y, P.pred, P.h, thr)
        rows.append({"구분": label, "판": d.name, "CV h = 1 PR-AUC": h1["cv"],
                     **{f"{f} h = 1": h1["fold"][f] for f in ev.CV_FOLDS},
                     "test h = 1 PR-AUC": float(mt.pr_auc(y, P.pred, P.u())), "threshold(자기 CV)": thr,
                     "Recall": tm["recall"], "Precision": tm["precision"], "F1": tm["f1"], "F2": tm["f2"],
                     "TP": tm["tp"], "FN": tm["fn"], "FP": tm["fp"], "TN": tm["tn"],
                     "①_better(CV)": decision["①_better"]})
    return pd.DataFrame(rows)


# ================================================================ 변화 표 줄(구현 세부 2)
def read_chain_log(path: Path) -> list[dict]:
    if not path.exists():
        raise FinalError(f"사슬 기록 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def chain_steps(lines: Sequence[Mapping]) -> list[dict]:
    """사슬 기록 → [{label, step, name, adopted, line}] (출발 → 채택 차례)."""
    out = []
    for ln in lines:
        step = str(ln.get("step", ""))
        j = ln.get("judgement")
        if j == "채택 취소":                         # 담당자 결정으로 그 단계 채택을 되돌림(뒤에 덧붙인 줄) — 변화 표에서 뺌
            out = [r for r in out if not (r["adopted"] and r["step"] == step)]
            continue
        if j == "출발":
            if step != "3-0":                       # 옛 순서 3-0b 가 쓴 출발 줄 — 새 순서에선 안 씀(v78)
                continue
            out.append({"label": "3-0 1위(첫 현재 최고 판)", "step": step, "name": ln["after"], "adopted": False,
                        "line": ln})
            continue
        names = []
        if j is not None:
            if j == "채택":
                names = [ln.get("after") or ln.get("candidate")]
        elif ln.get("parts"):
            names = [p["top"] for p in ln["parts"] if p.get("adopted")]
        elif ln.get("after") and ln.get("after") != ln.get("before"):
            names = [ln["after"]]
        for n in names:
            lab = "튜닝" if step.startswith("8") else "채택한 기법"
            out.append({"label": f"{lab}({step})", "step": step, "name": n, "adopted": True, "line": ln})
    if not out or out[0]["label"].startswith("3-0 1위") is False:
        raise FinalError("사슬 기록 첫 줄이 3-0 '출발' 이 아님")
    return out


def change_rows(cfg: Mapping, inputs_root: Path, decision: Mapping, state: Mapping) -> list[dict]:
    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    rows = [{"label": "1차 베이스라인", "step": "1", "name": dec["baseline"], "adopted": False,
             "dir": baseline_dir(cfg, inputs_root, dec["baseline"])}]
    for st in chain_steps(read_chain_log(resolve(inputs_root, cfg["chain_log"]))):
        st["dir"] = find_dir(st["name"], cfg=cfg, inputs_root=inputs_root, state=state, line=st["line"])
        rows.append(st)
    fin = decision["final"]
    fdir = (baseline_dir(cfg, inputs_root, fin) if decision["final_source"] == "1차 베이스라인"
            else resolve(inputs_root, decision["final_def"]["dir"]))
    rows.append({"label": "최종 모델", "step": STAGE, "name": fin, "adopted": False, "dir": fdir})
    return rows


def change_table(rows: Sequence[Mapping], inputs_root: Path, samples, truth, inc) -> pd.DataFrame:
    out, prev = [], None
    for r in rows:
        cv = cv_of(r["dir"])
        done = rn.read_json(r["dir"] / "done.json")
        pdir = pred_dir(inputs_root, r["dir"])
        t = ev.score_test(pd.read_parquet(rn.pred_path(pdir, "refit")), samples, truth,
                          float(cv["threshold"]["threshold"]), inc)
        row = {"구분": r["label"], "판": r["name"], "단계": r["step"],
               "val(CV 구간 가중)": float(cv["cv_band_weighted"]),
               "fold 1": float(cv["fold_band_weighted"]["fold1"]),
               "fold 2": float(cv["fold_band_weighted"]["fold2"]),
               "fold 3": float(cv["fold_band_weighted"]["fold3"]),
               "test 구간 가중": float(t["report"]["pr_auc_band_weighted"]),
               "같은 validation 으로 고른 횟수": done.get("selection_count"),
               "val 변화(직전 대비)": math.nan, "test 변화(직전 대비)": math.nan, "fold 1 알림": ""}
        if prev is not None:
            row["val 변화(직전 대비)"] = row["val(CV 구간 가중)"] - prev["val(CV 구간 가중)"]
            row["test 변화(직전 대비)"] = row["test 구간 가중"] - prev["test 구간 가중"]
            if r["adopted"] and not mt.is_improvement(row["fold 1"], prev["fold 1"]):
                row["fold 1 알림"] = "채택했는데 fold 1 단독으로 안 오름"
        out.append(row)
        prev = row
    return pd.DataFrame(out)


# ================================================================ 보조 표(구현 세부 3~5)
def product_bundles(p2: np.ndarray, cl: pd.DataFrame | None) -> np.ndarray:
    p2 = np.asarray(p2, dtype=object)
    out = np.full(p2.shape, "그 밖", dtype=object)
    out[p2 == s4.hi.NO_CODE] = s4.hi.NO_CODE
    if cl is None:
        return out
    par = dict(zip(cl["prdlst_cd"].astype(str), cl["hrnk_prdlst_cd"].fillna("").astype(str)))
    nm = dict(zip(cl["prdlst_cd"].astype(str), cl["kor_nm"].astype(str)))
    memo: dict[str, str] = {}

    def bundle(c: str) -> str:
        if c in memo:
            return memo[c]
        if c not in par:
            memo[c] = "그 밖"
            return memo[c]
        chain, x = [c], c
        for _ in range(20):
            up = par.get(x, "")
            if not up or up == x or up not in par:
                break
            chain.append(up)
            x = up
        top = nm.get(chain[-1], "")
        lv2 = nm.get(chain[-2], "") if len(chain) >= 2 else ""
        b = {"식물성": "농산물", "가공식품": "가공식품", "기준규격외": "기준규격외"}.get(top)
        if b is None and top == "동물성":
            b = lv2 if lv2 in ("축산물", "수산물") else "그 밖"
        memo[c] = b or "그 밖"
        return memo[c]

    for i, c in enumerate(p2):
        if c != s4.hi.NO_CODE:
            out[i] = bundle(str(c))
    return out


def _subset_scores(y, pred, h, u, thr) -> dict:
    if y.size == 0:
        return {"n_pairs": 0, "n_pos": 0}
    r = mt.score_report(y, pred, h, u)
    tm = mt.threshold_metrics(y, pred, h, thr)
    return {"n_pairs": int(y.size), "n_pos": int(y.sum()), "pr_auc_band_weighted": r["pr_auc_band_weighted"],
            "pr_auc_pooled": r["pr_auc_pooled"], "lift_pooled": r["lift_pooled"], "recall": tm["recall"],
            "precision": tm["precision"], "f2": tm["f2"]}


def test_pairs(pdir: Path, samples, truth, inc):
    P = ev.prepare_pairs(pd.read_parquet(rn.pred_path(pdir, "refit")), samples, ev.TEST_FOLD)
    y = truth.y_for(P.keys)
    if inc:
        keep = ~fd.incomplete_pair_mask(P.d_days, P.src_route, set(inc))
        P, y = P.take(keep), y[keep]
    return P, y


def aux_tables(P, y, samples, thr: float, cl: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = P.u()
    df = samples.df.iloc[P.pos]
    bundles = product_bundles(df["p2"].astype(str).to_numpy(dtype=object), cl)
    rows = []
    for b in ("농산물", "축산물", "수산물", "가공식품", "기준규격외", "그 밖", s4.hi.NO_CODE):
        for r in ("domestic", "overseas", "전체"):
            m = (bundles == b) & ((P.src_route == r) if r != "전체" else True)
            m = np.asarray(m, dtype=bool)
            rows.append({"품목 묶음": b, "국내·해외": r, **_subset_scores(y[m], P.pred[m], P.h[m], u[m], thr)})
    prod = pd.DataFrame(rows)
    if cl is None:
        prod["주석"] = "t_prdlst_cl 자료 없음 — 코드없음 밖은 모두 '그 밖'"
    # 단위 시료 수 3분위(구현 세부 4)
    k = samples.known_frame()
    cnt = k.groupby(["sgg", "p2", "p3"]).size()
    key = pd.MultiIndex.from_arrays([df["sgg"].astype(str), df["p2"].astype(str), df["p3"].astype(str)])
    n_unit = cnt.reindex(key).fillna(0).to_numpy(dtype=np.int64)
    uniq = pd.Series(n_unit, index=key)
    uniq = uniq[~uniq.index.duplicated()].to_numpy(dtype=float)
    q1, q2 = np.quantile(uniq, [1 / 3, 2 / 3]) if uniq.size else (0.0, 0.0)
    tier = np.where(n_unit <= q1, "하위", np.where(n_unit <= q2, "중간", "상위"))
    rows = [{"단위 시료 수 구간": t, "경계(q1·q2)": f"{q1:g}·{q2:g}", "단위 수": int(len(set(key[tier == t]))),
             **_subset_scores(y[tier == t], P.pred[tier == t], P.h[tier == t], u[tier == t], thr)}
            for t in ("하위", "중간", "상위")]
    return prod, pd.DataFrame(rows)


# ================================================================ 최종 모델 구성·입력 칼럼
def final_members(decision: Mapping, inputs_root: Path) -> list[tuple[str, Path, rn.VariantSpec]]:
    out = []
    for m in decision["final_def"]["members"]:
        d = resolve(inputs_root, m["dir"])
        out.append((m["name"], d, rn.VariantSpec(**rn.read_json(d / "spec.json"))))
    return out


def member_columns(vdir: Path) -> list[str]:
    for s in ("refit", "fold1"):
        p = vdir / f"columns_{s}.json"
        if p.exists():
            return list(rn.read_json(p)["columns"])
    return []


def column_lookup(spec: rn.VariantSpec) -> dict[str, rg.Column]:
    vs = s4.version_of(spec)
    cols = rg.tree_columns(vs) if spec.kind == "tree" else (rg.dl_seq_channels(vs) + rg.dl_static_columns(vs)
                                                             + rg.dl_fixed_static(vs))
    lk = {c.name: c for c in cols}
    for c in s4.CATENC_COLS:
        if c in lk:
            lk[s4.CATENC_PREFIX + c] = lk[c]
    return lk


def _strip(name: str) -> str:
    for p in ("x_seq:", "x_static:", "x_cat:"):
        if name.startswith(p):
            return name[len(p):]
    return name


def input_variable_table(members) -> pd.DataFrame:
    rows = []
    for mname, d, spec in members:
        try:
            lk = column_lookup(spec)
        except Exception as e:   # 판 설정을 등록부로 못 읽는 판(②·④ 메타 등) — 칼럼 이름만
            lk, note = {}, f"등록부로 못 읽음: {type(e).__name__}"
        else:
            note = ""
        for c in member_columns(d):
            n = _strip(c)
            base = n[: -len(rg.NA_SUFFIX)] if n.endswith(rg.NA_SUFFIX) else n
            col = lk.get(base)
            cid = col.id if col is not None and col.id else ("CAT5" if base in rg.CAT5 else
                                                              "TIME" if base in ("h", "g") else "")
            src, per = ID_SOURCE.get(cid, ("", ""))
            rows.append({"구성 모델": mname, "칼럼": c, "단위": getattr(col, "unit", None), "그룹": getattr(col, "group", None),
                         "ID": cid, "정의": f"§5 표 {cid}" if cid else "", "원천": src, "갱신 주기": per,
                         "window 끝": getattr(col, "window_end", None),
                         "결측 표시": n.endswith(rg.NA_SUFFIX), "주석": note})
    return pd.DataFrame(rows)


# ================================================================ 순열 중요도(부록 A-10)
class Unsupported(FinalError):
    pass


def default_nodes(cfg: Mapping, root: Path, inputs_root: Path, decision: Mapping, samples) -> dict:
    """구현 세부 (6): fold 마다 importance Node."""
    from lst import importance as im
    from lst import inputs as ip
    from lst import models_tree as mtr
    from lst import wiring as wr

    fdef = decision["final_def"]
    if fdef["kind"] in ("stack4", "nn60"):
        raise Unsupported(f"{fdef['kind']} 판은 importance 노드로 못 만듦(값 모델·60칸 출력)")
    members = final_members(decision, inputs_root)
    for m in fdef["members"]:
        if m.get("anomaly"):
            raise Unsupported(f"{m['name']}: 이상 점수 칼럼 판(값 모델이 입력 안에서 다시 추론돼야 함)")
    ctx = wr.load(cfg["context"], root, inputs_root, COMPONENTS).ctx
    w4 = None
    per_member = []
    for mname, d, spec in members:
        if "s4" not in spec.extra:
            per_member.append(s3.importance_nodes(ctx, samples, d, spec))
            continue
        if s4.s4_of(spec).get("provider"):
            raise Unsupported(f"{mname}: 값 모델(provider) 판")
        if spec.kind != "tree":
            if spec.algorithm == "NN60":
                raise Unsupported(f"{mname}: ② 60칸 출력 판")
            # 단계 4~6 딥러닝 판(② 제외) 입력 = Stage4Wiring → wiring 그대로(같은 DLInputs) → 저장 전처리기·모델 다시 읽기
            per_member.append(s3.importance_nodes(ctx, samples, d, spec))
            continue
        if w4 is None:
            th = resolve(inputs_root, cfg["theil_targets"]) if cfg.get("theil_targets") else None
            w4 = s4.Stage4Wiring(ctx, samples, inputs_root, data_root=root, theil_targets=th, n_jobs=cfg["n_jobs"])
        plan = s4.RowPlan.from_dict(s4.s4_of(spec).get("row_plan"))
        nit = rn.load_n_iter(d)
        lk = {k: v.unit for k, v in column_lookup(spec).items()}
        nodes = {}
        known = samples.known_frame()
        for f in ev.CV_FOLDS:
            asm = w4.make_inputs(spec, f)
            rows = rn.add_bounds(s4.plan_train_rows(known, f, plan, inputs_root), spec.W)
            asm.fit(rows)
            saved = rn.read_json(d / f"columns_{f}.json")["columns"]
            if list(asm.columns) != list(saved):
                raise FinalError(f"{mname} {f}: 저장 칼럼과 지금 조립 칼럼이 다름")
            model = mtr.TreeModel.load(d / f"model_{f}")
            n_iter = int(nit["selection"]["cross_n"][f]) if nit is not None else None
            nodes[f] = im.Leaf(f"{mname}-{f}", asm, model, n_iter=n_iter,
                               layout=im.layout_from_names(list(asm.columns), lk))
        per_member.append(nodes)
    if len(per_member) == 1:
        return per_member[0]
    return {f: im.Mean(f"{fdef['name']}-{f}", [pm[f] for pm in per_member]) for f in ev.CV_FOLDS}


def _expect(cfg: Mapping, inputs_root: Path, decision: Mapping, nodes: Mapping,
            base_tol: float | None) -> tuple[dict, float]:
    """섞기 전 점수 대조 값(판의 fold 점수)·허용 오차(딥러닝 모델이 있으면 1e-6, 아니면 1e-9)."""
    fdir = resolve(inputs_root, decision["final_def"]["dir"]) if decision["final_source"] != "1차 베이스라인" else \
        baseline_dir(cfg, inputs_root, decision["final"])
    cv = cv_of(fdir)
    kinds = {lf.model.__class__.__name__ for n in nodes.values() for lf in n.leaves()}
    tol = base_tol if base_tol is not None else (1e-6 if any("DL" in k for k in kinds) else 1e-9)
    return dict(cv["fold_band_weighted"]), tol


def _final_targets(nodes: Mapping) -> list:
    from lst import importance as im

    any_node = next(iter(nodes.values()))
    return im.targets_final(any_node.present_units(), any_node.present_names())


def run_importance(cfg: Mapping, root: Path, inputs_root: Path, out: Path, *, nodes_fn=None, M: int | None = None,
                   base_tol: float | None = None) -> Path:
    from lst import importance as im

    dst = out / IMPORTANCE_DIR
    if (dst / "importance_meta.json").exists() or (dst / "unsupported.json").exists():
        print(f"최종 모델 안 순열 중요도 이미 있음: {dst}")
        return dst
    decision = rn.read_json(out / DECISION_FILE)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    try:
        nodes = (nodes_fn or default_nodes)(cfg, root, inputs_root, decision, samples)
    except Unsupported as e:
        return _write_unsupported(dst, decision, str(e))
    targets = _final_targets(nodes)
    expect, tol = _expect(cfg, inputs_root, decision, nodes, base_tol)
    res = im.permutation_importance(nodes, samples, targets, M=M or cfg["importance_M"],
                                    expect_base=expect, base_tol=tol)
    return _save_importance(cfg, inputs_root, dst, decision, res)


def _write_unsupported(dst: Path, decision: Mapping, reason: str) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    cf.dump_json(dst / "unsupported.json", {"final": decision["final"], "reason": reason})
    print(f"[알림] 최종 모델 안 순열 중요도 지원 안 함: {reason}")
    return dst


def _save_importance(cfg: Mapping, inputs_root: Path, dst: Path, decision: Mapping, res) -> Path:
    res.save(dst)
    rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])).append(
        "importance", stage=STAGE, name=decision["final"], M=res.M, n_targets=len(res.summary))
    print(f"최종 모델 안 순열 중요도: {dst}")
    return dst


# ---------------------------------------------------------------- 나눠 돌리기(구현 세부 9)
STATE_DIR = "importance_state"
PART_LOG_DIR = "importance_logs"
PART_INFO_DIR = "parts"


def importance_done(out: Path) -> bool:
    dst = out / IMPORTANCE_DIR
    return (dst / "importance_meta.json").exists() or (dst / "unsupported.json").exists()


def state_tag(cfg: Mapping, inputs_root: Path, out: Path, decision: Mapping) -> str:
    """이어 하기 설정 서명 tag(최종 판·final_decision.json·시료 파일·코드 md5 — 바뀌면 이어 하지 않고 멈춤)."""
    return json.dumps({"final": decision["final"], "dir": decision["final_def"]["dir"],
                       "decision_md5": cf.file_md5(out / DECISION_FILE),
                       "samples_md5": cf.file_md5(resolve(inputs_root, cfg["samples"])),
                       "code_md5": cf.code_md5(CODE_ROOT)}, ensure_ascii=False, sort_keys=True)


def _info_path(out: Path, part: str) -> Path:
    return out / STATE_DIR / PART_INFO_DIR / f"{part.replace(':', '_h')}.json"


def _part_infos(out: Path) -> list[dict]:
    d = out / STATE_DIR / PART_INFO_DIR
    return [rn.read_json(p) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def run_importance_part(cfg: Mapping, root: Path, inputs_root: Path, out: Path, part: str, *, nodes_fn=None,
                        M: int | None = None, base_tol: float | None = None, check=None) -> dict:
    """부분 하나(--importance --part): 끝난 h 는 건너뛰고 남은 h 마다 상태 파일."""
    from lst import importance as im
    from lst.resources import check_resources

    if "importance_parts" not in cfg:
        raise cf.ConfigError("--part 는 설정 importance_parts 가 있을 때만")
    p = ipt.check_part_arg(part, cfg["importance_parts"])
    name = ipt.part_name(p)
    if importance_done(out):
        print(f"최종 모델 안 순열 중요도 이미 있음 — 부분 {name} 할 일 없음")
        return {"skipped": True}
    decision = rn.read_json(out / DECISION_FILE)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    sd = out / STATE_DIR
    f, lo, hi = p
    if not im.pending_h(samples, f, lo, hi, sd):       # 다 끝난 부분 — 모델을 읽지 않음
        print(f"최종 모델 안 순열 중요도 부분 {name}: 남은 h 없음")
        return {"skipped": True}
    tag = state_tag(cfg, inputs_root, out, decision)
    try:
        nodes = (nodes_fn or default_nodes)(cfg, root, inputs_root, decision, samples)
    except Unsupported as e:                           # 합치기에서 기본 길과 같은 unsupported.json
        cf.dump_json(_info_path(out, name), {"part": name, "tag": tag, "unsupported": str(e)}, overwrite=True)
        print(f"[알림] 최종 모델 안 순열 중요도 지원 안 함(부분 {name}): {e}")
        return {"unsupported": str(e)}
    targets = _final_targets(nodes)
    expect, tol = _expect(cfg, inputs_root, decision, nodes, base_tol)
    cf.dump_json(_info_path(out, name), {"part": name, "tag": tag, "unsupported": None, "expect_base": expect,
                                         "base_tol": tol}, overwrite=True)
    r = im.importance_part(nodes, samples, targets, fold=f, h_from=lo, h_to=hi, M=M or cfg["importance_M"],
                           state_dir=sd, check=check or (lambda: check_resources(out)), tag=tag,
                           progress=lambda msg: print(f"[{name}] {msg}", flush=True))
    print(f"최종 모델 안 순열 중요도 부분 {name}: 계산 h {len(r['computed'])}개·이어받은 h {len(r['resumed'])}개, "
          f"{r['seconds']:.0f}초")
    return r


def pending_parts(cfg: Mapping, inputs_root: Path, out: Path) -> list[str]:
    """남은 h 가 있는 부분(상태 파일 기준) — '지원 안 함' 부분 정보가 있으면 없음."""
    from lst import importance as im

    if any(i.get("unsupported") for i in _part_infos(out)):
        return []
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    return [ipt.part_name(p) for p in ipt.parse_parts(cfg["importance_parts"])
            if im.pending_h(samples, p[0], p[1], p[2], out / STATE_DIR)]


def merge_importance(cfg: Mapping, inputs_root: Path, out: Path, *, M: int | None = None) -> Path:
    """모든 부분이 끝난 뒤: 상태 파일 → importance/(한 번에 돌린 것과 같은 records·summary·meta) → 상태 폴더 지움."""
    import shutil

    from lst import importance as im

    dst = out / IMPORTANCE_DIR
    if importance_done(out):
        return dst
    if dst.exists():                                  # 쓰다 멈춘 결과 폴더 — 옮기고 다시 씀(상태 파일은 그대로)
        rn._move_partial(dst, out, reason="멈춘 최종 중요도 합치기 다시")
    decision = rn.read_json(out / DECISION_FILE)
    sd = out / STATE_DIR
    tag = state_tag(cfg, inputs_root, out, decision)
    infos = _part_infos(out)
    if not infos:
        raise FinalError(f"부분 정보 없음: {sd / PART_INFO_DIR}")
    if any(i["tag"] != tag for i in infos):
        raise FinalError("부분 정보의 서명 tag 가 지금 판·코드와 다름 — 지휘자 확인")
    uns = [i for i in infos if i.get("unsupported")]
    if uns:
        _write_unsupported(dst, decision, uns[0]["unsupported"])
    else:
        keys = {json.dumps([i["expect_base"], i["base_tol"]], sort_keys=True) for i in infos}
        if len(keys) != 1:
            raise FinalError("부분마다 섞기 전 점수 대조 값·허용 오차가 다름")
        expect, tol = infos[0]["expect_base"], infos[0]["base_tol"]
        samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
        targets = [im.target_from_dict(d) for d in im.state_targets(sd)]
        res = im.merge_parts(samples, targets, state_dir=sd, M=M or cfg["importance_M"], expect_base=expect,
                             base_tol=tol, tag=tag)
        _save_importance(cfg, inputs_root, dst, decision, res)
    shutil.rmtree(sd)
    return dst


def run_importance_parts(cfg: Mapping, argv, root: Path, inputs_root: Path, out: Path, *, launch=None, nodes_fn=None,
                         M: int | None = None, base_tol: float | None = None, check=None,
                         in_process: bool = False) -> Path:
    """본 프로세스: 남은 부분을 한꺼번에(lst/imp_parts.run_parts) → 모두 끝나면 합치기. in_process = 시험."""
    if importance_done(out):
        print(f"최종 모델 안 순열 중요도 이미 있음: {out / IMPORTANCE_DIR}")
        return out / IMPORTANCE_DIR
    ipt.run_parts(
        label="최종 중요도", stage=STAGE, parts=[ipt.part_name(p) for p in ipt.parse_parts(cfg["importance_parts"])],
        pending_fn=lambda: pending_parts(cfg, inputs_root, out), script=Path(__file__), argv=argv,
        log_dir=out / PART_LOG_DIR, exp_log=rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])),
        launch=launch or ipt.launch_parts,
        run_in_process=(lambda part: run_importance_part(cfg, root, inputs_root, out, part, nodes_fn=nodes_fn, M=M,
                                                         base_tol=base_tol, check=check)) if in_process else None)
    return merge_importance(cfg, inputs_root, out, M=M)


def _limit_gpu(cfg: Mapping) -> None:
    if cfg.get("gpu_memory_fraction"):
        from lst import models_dl as md
        from lst.resources import limit_gpu_memory

        limit_gpu_memory(cfg["gpu_memory_fraction"])
        md.set_gpu_fraction(cfg["gpu_memory_fraction"])   # 딥러닝 모델 불러올 때 0.4 로 다시 올라가지 않게(models_dl 17)


# ================================================================ 리서치팀 표(부록 A-10)
def _scores_of(vdir: Path, inputs_root: Path, samples, truth, inc) -> tuple[float, float]:
    cv = cv_of(vdir)
    tp = vdir / "test_scores.json"
    if tp.exists():
        t = rn._load_scores(tp)["report"]["pr_auc_band_weighted"]
    else:
        t = ev.score_test(pd.read_parquet(rn.pred_path(pred_dir(inputs_root, vdir), "refit")), samples, truth,
                          float(cv["threshold"]["threshold"]), inc)["report"]["pr_auc_band_weighted"]
    return float(cv["cv_band_weighted"]), float(t)


def research_table(cfg: Mapping, inputs_root: Path, out: Path, decision: Mapping, samples, truth, inc
                   ) -> tuple[pd.DataFrame, list[str]]:
    b_dir = resolve(inputs_root, cfg["dir_2_0"])
    if not (b_dir / "done.json").exists():
        raise FinalError(f"2-0 판 폴더 없음: {b_dir}")
    b_cv, b_t = _scores_of(b_dir, inputs_root, samples, truth, inc)
    r_root = resolve(inputs_root, cfg["dir_2_R"])
    imp2r = r_root / "2-R1_importance" / "importance_summary.csv"
    imp2r = pd.read_csv(imp2r, encoding="utf-8-sig") if imp2r.exists() else None
    impf_p = out / IMPORTANCE_DIR / "importance_summary.csv"
    impf = pd.read_csv(impf_p, encoding="utf-8-sig") if impf_p.exists() else None
    unsup = out / IMPORTANCE_DIR / "unsupported.json"
    unsup_reason = rn.read_json(unsup)["reason"] if unsup.exists() else None
    d3 = rn.read_json(resolve(inputs_root, cfg["decision_3_0b"]))
    members = final_members(decision, inputs_root)
    final_cols = {_strip(c) for _, d, _ in members for c in member_columns(d)}
    structs = [sp.algorithm for _, _, sp in members]
    fkind = decision["final_def"]["kind"]

    def drop_result(col: str) -> str:
        """부록 A-10(v78): 최종 모델이 3-0b 끝 판에서 이어진 판이면 그 결과(1등이 앙상블이면 구성마다 같은 결과), 1차
        베이스라인이면 '3-0b 를 안 거친 판', 그 판 입력에 없는 칼럼이면 해당 없음. 구현 세부: '이어진 판' = 사슬 기록에서
        3-0b 판정 뒤에 최종 모델이 정해짐(최종 = 2차 사슬 끝 판) — ② 가 최종이면 ② 입력을 준 트리 대상 판 기준은 판정 파일
        에서 따로 못 가려 '3-0b 를 거친 사슬(② 입력 판 확인 필요)' 로 적음."""
        if decision["final_source"] == "1차 베이스라인":
            return "3-0b 를 안 거친 판(1차 베이스라인)"
        if fkind == "nn60":
            return "3-0b 를 거친 사슬(② 입력 판 확인 필요)"
        g = rg.group(rg.unit(col).group_no).name
        if d3.get("adopted") and g in set(d3.get("drop_groups") or []):
            return "그룹째 빠짐"
        if col not in final_cols:
            return "해당 없음(최종 모델 입력에 없음)"
        return "남음" if d3.get("adopted") else "남음(3-0b 후보 버림·학습 없음)"

    def imp_of(tab, kind, name):
        if tab is None:
            return math.nan, math.nan
        r = tab[(tab["target_kind"] == kind) & (tab["target"] == name)]
        return (float(r["mean_I"].iloc[0]), float(r["pi"].iloc[0])) if len(r) else (math.nan, math.nan)

    rows, notes = [], []
    for i, col in enumerate(rg.RESEARCH_UNITS):
        lit, page, desc = RESEARCH_REF[col]
        vd = r_root / f"2-R{i + 2}"
        cv_, t_ = _scores_of(vd, inputs_root, samples, truth, inc) if (vd / "done.json").exists() else (math.nan,) * 2
        mi2, _ = imp_of(imp2r, "column", col)
        mif, pif = imp_of(impf, "column", col)
        kept = col in final_cols
        rows.append({"구분": "칼럼", "칼럼": col, "문헌": lit, "쪽": f"{PDF} {page}", "변수 정의": desc,
                     "하나씩 추가 판": f"2-R{i + 2}", "CV 구간 가중": cv_, "CV 변화(기준 대비)": cv_ - b_cv,
                     "test 구간 가중": t_, "test 변화(기준 대비)": t_ - b_t, "변수별 중요도(모든 리서치 칼럼 넣은 판)": mi2,
                     "최종 모델 구조의 변수 빼기 결과": drop_result(col), "최종 모델에 남음": "예" if kept else "아니오",
                     "최종 모델 안 순열 중요도(평균 하락)": mif, "최종 모델 안 순열 중요도(하락 비율)": pif})
        imp_txt = (f"최종 모델 안에서 이 칼럼을 섞으면 검증 성능(구간 가중 PR-AUC)이 평균 {mif:.4f} 떨어진다"
                   if kept and not math.isnan(mif) else
                   ("최종 모델 입력에 없다" if not kept else
                    f"최종 모델 안 순열 중요도는 계산하지 못했다({unsup_reason or '계산 전'})"))
        notes.append(f"{PDF} {page}({lit})에 따라 '{desc}'({col})를 넣었고, 파생변수 없는 기준 판보다 검증 점수가 "
                     f"{cv_ - b_cv:+.4f}, test 점수가 {t_ - b_t:+.4f} 바뀌었으며, {imp_txt}.")
    vd1 = r_root / "2-R1"
    cv1, t1 = _scores_of(vd1, inputs_root, samples, truth, inc) if (vd1 / "done.json").exists() else (math.nan,) * 2
    mi2, _ = imp_of(imp2r, "bundle", "리서치 변수 전체")
    mif, pif = imp_of(impf, "bundle", "리서치 변수 전체")
    rows.append({"구분": "리서치 변수 전체", "칼럼": "·".join(rg.RESEARCH_UNITS), "하나씩 추가 판": "2-R1(9개 모두)",
                 "CV 구간 가중": cv1, "CV 변화(기준 대비)": cv1 - b_cv, "test 구간 가중": t1, "test 변화(기준 대비)": t1 - b_t,
                 "변수별 중요도(모든 리서치 칼럼 넣은 판)": mi2, "최종 모델 안 순열 중요도(평균 하락)": mif,
                 "최종 모델 안 순열 중요도(하락 비율)": pif})
    if impf is not None:
        for _, r in impf[impf["target_kind"] == "group"].iterrows():
            rows.append({"구분": "그룹", "칼럼": r["target"], "최종 모델 안 순열 중요도(평균 하락)": float(r["mean_I"]),
                         "최종 모델 안 순열 중요도(하락 비율)": float(r["pi"])})
    t = pd.DataFrame(rows)
    t.attrs["header"] = f"기준 = 파생변수 없는 판(2-0: {b_dir.name}), CV {b_cv:.4f}·test {b_t:.4f}"
    return t, notes


# ================================================================ 보고
def _write_csv(df: pd.DataFrame, p: Path) -> Path:
    with open(p, "x", newline="", encoding="utf-8-sig") as fh:
        df.to_csv(fh, index=False, float_format="%.17g")
    return p


def report(cfg: Mapping, root: Path, inputs_root: Path, out: Path, reason: str, tracker=None) -> Path:
    decision = rn.read_json(out / DECISION_FILE)
    rdir = out / REPORT_DIR
    if rdir.exists():
        raise FinalError(f"보고 폴더가 이미 있음(덮어쓰지 않음): {rdir}")
    permit = ev.report_permit(STAGE, reason)
    samples_path = resolve(inputs_root, cfg["samples"])
    samples = ev.SampleTable.read(samples_path)
    truth = ev.TestTruth.load(samples_path, permit)
    inc = ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"]))
    state = s4.load_state(resolve(inputs_root, cfg["chain_state"]))
    rdir.mkdir(parents=True)
    rows = change_rows(cfg, inputs_root, decision, state)
    ch = change_table(rows, inputs_root, samples, truth, inc)
    p_ch = _write_csv(ch, rdir / "change_table.csv")
    fdir = rows[-1]["dir"]
    pdir = pred_dir(inputs_root, fdir)
    cv = cv_of(fdir)
    thr = float(cv["threshold"]["threshold"])
    test = ev.score_test(pd.read_parquet(rn.pred_path(pdir, "refit")), samples, truth, thr, inc)
    cf.dump_scores_json(rdir / "final_test_scores.json", {"final": decision["final"], "cv": cv, "test": test})
    _write_csv(ev.h_metrics_test(pd.read_parquet(rn.pred_path(pdir, "refit")), samples, truth, thr, inc),
               rdir / "final_h_metrics.csv")
    if decision.get("x1"):
        _write_csv(x1_compare_test(cfg, inputs_root, decision, fdir, samples, truth, inc), rdir / "x1_compare.csv")
    cl = None
    if cfg.get("prdlst_cl"):
        cp = resolve(root, cfg["prdlst_cl"])
        if cp.exists():
            cl = pd.read_parquet(cp, columns=["prdlst_cd", "hrnk_prdlst_cd", "kor_nm"])
    P, y = test_pairs(pdir, samples, truth, inc)
    prod, tier = aux_tables(P, y, samples, thr, cl)
    _write_csv(prod, rdir / "aux_product_route.csv")
    _write_csv(tier, rdir / "aux_unit_tertile.csv")
    rt, notes = research_table(cfg, inputs_root, out, decision, samples, truth, inc)
    _write_csv(rt, rdir / "research_table.csv")
    cf.dump_json(rdir / "research_notes.json", {"header": rt.attrs.get("header"), "sentences": notes})
    _write_csv(input_variable_table(final_members(decision, inputs_root)), rdir / "input_variables.csv")
    cf.dump_json(rdir / "coord_sensitivity.json", {"status": "하지 않음(0-3 예보 좌표 차이 측정 필요 — 부록 A-12)"})
    cf.save_run_meta(rdir / "run_meta.json", dict(cfg), cf.inputs_md5(inputs_root, [cfg["samples"], cfg["chain_state"],
                                                                                   cfg["chain_log"]]),
                     CODE_ROOT, n_jobs=cfg["n_jobs"], extra={"mode": "report", "reason": reason})
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    log.append("test_reported", stage=STAGE, names=[r["name"] for r in rows], permit_reason=reason)
    if tracker is not None:
        tracker.log_artifact(tracker.chain_run(), p_ch, "final")
        rid = rn.read_json(fdir / "done.json").get("mlflow_run_id")
        if rid:
            tracker.set_tags(rid, {"최종": "예"})
    print(f"최종 보고: {rdir}")
    return rdir


# ================================================================ 진입점
def main(argv=None, *, tracker_fn=sc.make_tracker, nodes_fn=None, importance_M: int | None = None,
         check_resources_fn=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--report", action="store_true")
    g.add_argument("--importance", action="store_true")
    ap.add_argument("--reason")
    ap.add_argument("--part")                 # --importance 부분 하나(구현 세부 9 — 보통은 본 프로세스가 띄움)
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = rp.out
    if args.report:
        if not args.reason:
            raise SystemExit("--report 에는 --reason 이 필요")
        tr = tracker_fn(cfg) if cfg["use_mlflow"] else None
        return report(cfg, rp.root, inputs_root, out, args.reason, tr)
    if args.part and not args.importance:
        raise SystemExit("--part 는 --importance 와 같이")
    if args.importance:
        if not (out / DECISION_FILE).exists():
            raise SystemExit("최종 고르기(기본 모드)를 먼저")
        if args.part:
            _limit_gpu(cfg)
            return run_importance_part(cfg, rp.root, inputs_root, out, args.part, nodes_fn=nodes_fn, M=importance_M,
                                       check=check_resources_fn)
        if cfg.get("importance_parts") and not importance_done(out):
            return run_importance_parts(cfg, [a for a in argv if a != "--importance"], rp.root, inputs_root, out,
                                        M=importance_M)
        if cfg.get("gpu_memory_fraction"):
            from lst.resources import limit_gpu_memory

            limit_gpu_memory(cfg["gpu_memory_fraction"])
        return run_importance(cfg, rp.root, inputs_root, out, nodes_fn=nodes_fn, M=importance_M)
    d = select_final(cfg, inputs_root, out)
    if not (out / "run_meta.json").exists():
        cf.save_run_meta(out / "run_meta.json", dict(cfg), cf.inputs_md5(inputs_root, [cfg["samples"],
                                                                                         cfg["chain_state"]]),
                         CODE_ROOT, n_jobs=cfg["n_jobs"], extra={"mode": "select"})
    return d


if __name__ == "__main__":
    main()
