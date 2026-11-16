# -*- coding: utf-8 -*-
"""make_configs_dom.py — '국내만 버전' 설정 묶음 configs_dom/ 을 configs/ 에서 만듦(손으로 베끼지 않음 — configs/ 가
바뀌면 다시 돌려 따라가게). lst/scope.py 머리 참고.

바꾸는 것(그 밖 칸은 configs/ 그대로 — 설정 md5·판 이름 규칙도 같은 방식):
  - 맨 앞(plan 다음)에 "scope": "domestic"
  - plan 끝에 " — 국내만 버전"
  - code_reviewed → false(지휘자가 검수 뒤 바꿈 — 바꾼 파일은 --check 에서 그 칸만 차이로 허용). 다시 만들 때 그 칸 말고는
    내용이 같은 기존 파일은 기존 code_reviewed 값을 그대로 둠(지휘자가 바꾼 값을 되돌리지 않게 — 새·바뀐 파일만 false)
  - accept_previous_code·reuse_3_0b_old 칸 뺌(지금 판 실행 이력 — 국내판은 처음부터)
  - 문자열 값 'configs/…' → 'configs_dom/…'(다른 설정을 가리키는 칸)
  - s_final search_dirs 에서 's3/3-0b'(지금 판 옛 순서 3-0b 폴더) 뺌
  - Theil's U 대상 목록 경로는 그대로(s0/0-1/apply/14_theil_target_columns.json — 국내판 뿌리의 파일은 s0_dom_prep 가
    src_route 를 뺀 목록으로 씀)
더 만드는 것: configs_dom/s0_dom_prep.json(단계 0 국내 준비)·configs_dom/s_dom_rescore.json(두 버전 비교).
빼는 설정(국내판에서 안 돌림 — 시료와 무관한 단계 0 결과는 지금 판 것을 읽기 전용으로 씀, 운영 준비·시간 측정·DB 추출):
  EXCLUDE 목록.

실행(로컬, 코드 폴더에서):  python scripts/make_configs_dom.py            # configs_dom/ 을 새로 씀(있던 파일 교체)
                           python scripts/make_configs_dom.py --check    # configs_dom/ 이 configs/ 와 맞는지만(다르면 1)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC = "configs"
DST = "configs_dom"
SCOPE = "domestic"
EXCLUDE = (
    "s0_6_static_food2.json", "s5_6_T_food2.json", "s5_6_T_food2_ko.json", "s_final_food2.json",
    "s_ko_dist.json",             # 식품 고정표 v2 다시 학습·KO 비교(11-12 — 지금 판만, 국내판 폐기 뒤)
    "s0_1_samples.json",          # 시료 표 = s0_dom_prep 가 지금 판 0-1 결과에서 국내 행만
    "s0_3_forecast_prep.json",    # 운영 준비(실험 입력 아님)
    "s0_4_spi.json", "s0_4_weather_domestic.json", "s0_4_weather_overseas.json",   # 시료 무관 — 지금 판 것 공유
    "s0_5_window_check.json",     # window 생성기 대조(부록 B-7 예시 — 시료 무관)
    "s0_6_static.json",           # 식품 고정표·축사 — 시료 무관(식품 2단계 짝은 s0_dom_prep 가 국내 행으로 대조)
    "s0_6_wxfeat.json",           # 기상 파생 공식 점검(다음 단계 입력 아님)
    "s0_9_timing.json",           # 시간 측정(계획서: 예상 끝 날짜 안 잼)
    "s0_db_extract.json",         # DB 추출(로컬)
)
DROP_KEYS = ("accept_previous_code", "reuse_3_0b_old")
OLD_3_0B_DIR = "s3/3-0b"
PLAN_TAIL = " — 국내만 버전"

PREP_CFG = {
    "plan": "국내만 버전 단계 0 준비: 결과 뿌리 표지·지금 판 단계 0 중 시료와 무관한 결과 읽기 전용 연결·국내 행 시료 표"
            "(지금 판 0-1 시료 표에서 src_route == 'domestic' 만)·덜 찬 달·Theil's U 대상(src_route 뺌)·fold 별 셈" + PLAN_TAIL,
    "scope": SCOPE,
    "source_samples": "s0/0-1/apply/samples.parquet",
    "source_theil_targets": "s0/0-1/apply/14_theil_target_columns.json",
    "source_incomplete_months": "s0/0-1/apply/12_incomplete_months.json",
    "shared_dirs": ["s0/0-4_v2", "s0/0-4_spi", "s0/0-4_weather_overseas", "s0/0-6_static"],
    "food_table": "s0/0-6_static/식품_고정표.parquet",
    "expected": {"rows": 628586, "pos": 1844},
    "drop_theil_columns": ["src_route"],
}
RESCORE_CFG = {
    "plan": "두 버전 비교: 지금 판(국내 + 해외) 판들을 학습 없이 국내 시료로만 다시 채점(lst.evaluate 그대로 — CV 는 늘, "
            "test 는 --report 보고 허가로만)" + PLAN_TAIL,
    "scope": SCOPE,
    "samples": "s0/0-1/apply/samples.parquet",
    "incomplete_months": "s0/0-1/apply/12_incomplete_months.json",
    "source_samples": "s0/0-1/apply/samples.parquet",
    "source_te7a_dir": "s0/0-10",
    "source_final_decision": "final/final_decision.json",
    "stage_dirs": ["s1/1-1", "s1/1-2", "s1/x3", "s2/2-0", "s2/2-1", "s2/2-1b-T", "s2/2-1b-L", "s2/2-R",
                   "s3/3-0-T", "s3/3-0-D", "s3/x678-G", "s3/x678-C", "s3/3-1", "s3/3-2", "s3/3-3", "s3/3-0b-best",
                   "s3/x8", "s4/4-pre", "s4/4-T", "s4/4-X1", "s4/4-NN", "s4/4-C", "s5_6/5-6-T", "s8/8-T", "s8/8-F",
                   "s8/8-D"],
}


def transform(name: str, cfg: dict) -> dict:
    if "scope" in cfg:
        raise ValueError(f"{name}: configs/ 설정에 scope 칸이 있음")
    out = {}
    for k, v in cfg.items():
        if k in DROP_KEYS:
            continue
        out[k] = v
        if k == "plan":
            out[k] = v + PLAN_TAIL
            out["scope"] = SCOPE
    if "scope" not in out:
        out = {"scope": SCOPE, **out}
    if "code_reviewed" in out:
        out["code_reviewed"] = False
    out = _rewrite(out)
    if name == "s_final.json" and "search_dirs" in out:
        out["search_dirs"] = [d for d in out["search_dirs"] if d != OLD_3_0B_DIR]
    return out


def _rewrite(o):
    if isinstance(o, str):
        return DST + o[len(SRC):] if o.startswith(SRC + "/") else o
    if isinstance(o, dict):
        return {k: _rewrite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_rewrite(v) for v in o]
    return o


def build(code_root: Path = CODE_ROOT) -> dict[str, dict]:
    src = code_root / SRC
    out = {}
    for p in sorted(src.glob("*.json")):
        if p.name in EXCLUDE:
            continue
        with open(p, "r", encoding="utf-8") as f:
            out[p.name] = transform(p.name, json.load(f))
    out["s0_dom_prep.json"] = PREP_CFG
    out["s_dom_rescore.json"] = RESCORE_CFG
    return out


def dumps(cfg: dict) -> str:
    return json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"


def check(code_root: Path = CODE_ROOT) -> list[str]:
    """configs_dom/ 이 지금 configs/ 에서 만든 것과 다른 파일 목록(code_reviewed 칸만 다른 것은 허용)."""
    want = build(code_root)
    dst = code_root / DST
    bad = []
    have = {p.name for p in dst.glob("*.json")} if dst.is_dir() else set()
    for n in sorted(set(want) | have):
        p = dst / n
        if n not in want or not p.exists():
            bad.append(n)
            continue
        with open(p, "r", encoding="utf-8") as f:
            got = json.load(f)
        a, b = dict(got), dict(want[n])
        if "code_reviewed" in a and "code_reviewed" in b:
            a.pop("code_reviewed")
            b.pop("code_reviewed")
        if a != b:
            bad.append(n)
    return bad


def write(code_root: Path = CODE_ROOT) -> list[Path]:
    dst = code_root / DST
    dst.mkdir(exist_ok=True)
    want = build(code_root)
    stale = [p for p in dst.glob("*.json") if p.name not in want]
    for p in stale:
        p.unlink()
    out = []
    for n, cfg in want.items():
        p = dst / n
        if p.exists() and "code_reviewed" in cfg:
            with open(p, "r", encoding="utf-8") as f:
                old = json.load(f)
            a, b = dict(old), dict(cfg)
            a.pop("code_reviewed", None)
            b.pop("code_reviewed", None)
            if a == b and "code_reviewed" in old:
                cfg = dict(cfg)
                cfg["code_reviewed"] = old["code_reviewed"]
        p.write_text(dumps(cfg), encoding="utf-8", newline="\n")
        out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        bad = check()
        print("configs_dom 맞음" if not bad else f"configs_dom 다름(다시 만들 것): {bad}")
        return 1 if bad else 0
    ps = write()
    print(f"configs_dom/ {len(ps)}개 씀")
    return 0


if __name__ == "__main__":
    sys.exit(main())
