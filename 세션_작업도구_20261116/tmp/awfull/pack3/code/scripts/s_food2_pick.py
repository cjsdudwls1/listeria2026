# -*- coding: utf-8 -*-
"""s_food2_pick.py — 식품 갱신 최종 고르기(사용자 결정 2026-11-12: 'Koutsoumanis 생장식 6칼럼 추가해서 점수 더 좋으면 그게 최종').

5-LSTM-food2(식품 고정표 v2 로 5-LSTM 다시 학습)를 현재 판, 5-LSTM-KO(5-LSTM-food2 + KO 6칼럼)를 후보로 두고 2차 사슬 채택
규칙(lst.evaluate.judge_adoption — val = CV 구간 가중 PR-AUC 를 소수 4자리 반올림한 뒤 오를 때만 채택)으로 판정해, 최종
설정 파일 둘(--config-food2 / --config-ko) 중 하나를 고름. test 는 읽지 않음(고르기는 CV 만 — §1-3).
출력: <inputs>/logs/food2_final_pick.json(이미 있으면 멈춤) · 실험 로그 decision 한 줄 · 표준 출력 마지막 줄 = 고른 설정 경로.

실행(서버 대기열 B):
  $P -I scripts/s_food2_pick.py --inputs $R --config-food2 configs/s_final_food2.json --config-ko configs/s_final_food2_ko.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import metrics as mt  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage3 as st3  # noqa: E402

PICK_FILE = "logs/food2_final_pick.json"
STAGE = "최종(식품 갱신)"


class PickError(RuntimeError):
    pass


def _override(cfg_path: Path) -> dict:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    ov = cfg.get("final_override")
    if not ov or not ov.get("name") or not ov.get("dir"):
        raise PickError(f"{cfg_path}: final_override(name·dir) 없음")
    return ov


def pick(inputs_root: Path, cfg_food2: Path, cfg_ko: Path) -> dict:
    """두 설정의 final_override 판(food2 = 현재 판, KO = 후보)을 채택 규칙으로 판정 → 기록 dict(고른 설정 포함)."""
    ov_f, ov_k = _override(cfg_food2), _override(cfg_ko)
    if ov_f["replaces"] != ov_k["replaces"]:
        raise PickError("두 설정의 final_override.replaces 가 다름")
    if ov_f["name"] == ov_k["name"]:
        raise PickError("두 설정의 final_override.name 이 같음")
    cands = {}
    for key, ov in (("food2", ov_f), ("ko", ov_k)):
        d = Path(inputs_root) / ov["dir"]
        cands[key] = st3.candidate_of(d.parent, ov["name"])
    rec = ev.judge_adoption(cands["ko"], cands["food2"], None)   # 동점 많아 다시 채점이 필요하면 ValueError(대기열 멈춤)
    chosen = "ko" if rec["adopt"] else "food2"
    return {
        "stage": STAGE,
        "rule": "2차 사슬 채택 규칙 — val(CV 구간 가중 PR-AUC) 소수 4자리 반올림 뒤 후보가 오를 때만 채택(judge_adoption)",
        "user_decision": "Koutsoumanis 생장식 6칼럼 추가해서 점수 더 좋으면 그게 최종(2026-11-12)",
        "current": {"name": cands["food2"].name, "cv": cands["food2"].cv, "cv_round4": mt.round4(cands["food2"].cv),
                    "dir": ov_f["dir"], "config": str(cfg_food2)},
        "candidate": {"name": cands["ko"].name, "cv": cands["ko"].cv, "cv_round4": mt.round4(cands["ko"].cv),
                      "dir": ov_k["dir"], "config": str(cfg_ko)},
        "judge": rec,
        "chosen": cands[chosen].name,
        "chosen_config": str(cfg_ko if chosen == "ko" else cfg_food2),
    }


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="결과 뿌리(results)")
    ap.add_argument("--config-food2", required=True)
    ap.add_argument("--config-ko", required=True)
    a = ap.parse_args(argv)
    inputs_root = Path(a.inputs)
    out = inputs_root / PICK_FILE
    if out.exists():
        raise PickError(f"이미 있음(덮지 않음): {out}")
    res = pick(inputs_root, Path(a.config_food2), Path(a.config_ko))
    cf.dump_json(out, res)
    rn.ExperimentLog(inputs_root / "logs" / "experiment_log.jsonl").append(
        "decision", stage=STAGE, name=res["chosen"], decision=f"최종 = {res['chosen']}",
        candidate=res["candidate"]["name"], current=res["current"]["name"],
        cv={res["current"]["name"]: res["current"]["cv"], res["candidate"]["name"]: res["candidate"]["cv"]},
        adopt=res["judge"]["adopt"], rule=res["rule"], reason=res["user_decision"])
    print(f"{res['current']['name']} CV {res['current']['cv']:.6f}({res['current']['cv_round4']:.4f}) · "
          f"{res['candidate']['name']} CV {res['candidate']['cv']:.6f}({res['candidate']['cv_round4']:.4f}) → 최종 {res['chosen']}",
          file=sys.stderr)
    print(res["chosen_config"])
    return res["chosen_config"]


if __name__ == "__main__":
    main()
