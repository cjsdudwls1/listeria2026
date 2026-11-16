# -*- coding: utf-8 -*-
"""dom_final_cfg.py — 국내만 버전 최종 단계 설정 채우기(lst/scope.py 머리 참고).

configs_dom/s_final.json 의 두 칸은 그 판의 앞 단계 결과에 따라 정해지는 판 폴더라(지금 판 configs/s_final.json 은
지휘자가 지금 판 결과를 보고 적은 값: dir_2_0 = s2/2-0/2-0-CAT, x1_dir = s4/4-X1/x1-CAT) 국내판 결과로 다시 정함:
  - dir_2_0 = s2/2-0/2-0-<A>(A = 국내판 s1/1-3/1-3_decision.json 의 A — s2_prep.name_base 규칙)
  - x1_dir  = 국내판 s4/4-X1 아래 끝난 판(done.json) 폴더 — 정확히 1개일 때만, 0개면 칸을 뺌(선택 칸), 2개 이상이면 멈춤
나머지 칸은 configs_dom/s_final.json 그대로. 결과 = --inputs 아래 --write 경로(이미 있으면 내용이 같을 때만 그대로).

실행(서버 — run_dom_cpu.sh 가 단계 8 판정 뒤에 부름):
  $P -I scripts/dom_final_cfg.py --inputs $RD --config configs_dom/s_final.json --write logs/s_final_dom.json
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
from lst import scope as lst_scope  # noqa: E402
from lst.paths import resolve, to_rel  # noqa: E402

DECISION_1_3 = "s1/1-3/1-3_decision.json"
X1_ROOT = "s4/4-X1"


def fill(cfg: dict, inputs_root: Path) -> dict:
    if cfg.get("scope") != lst_scope.DOMESTIC:
        raise cf.ConfigError("국내판 설정(scope = domestic)만")
    out = dict(cfg)
    dec = cf.load_json_strict(resolve(inputs_root, DECISION_1_3))
    rel = f"s2/2-0/2-0-{dec['A']}"
    if not (resolve(inputs_root, rel) / "done.json").exists():
        raise FileNotFoundError(f"2-0 판 폴더 없음: {rel}")
    out["dir_2_0"] = rel
    x1 = resolve(inputs_root, X1_ROOT)
    done = sorted(p.name for p in x1.iterdir() if p.is_dir() and (p / "done.json").exists()) if x1.is_dir() else []
    if len(done) > 1:
        raise RuntimeError(f"① 판이 여럿 — 지휘자가 x1_dir 를 정할 것: {done}")
    if done:
        out["x1_dir"] = f"{X1_ROOT}/{done[0]}"
    else:
        out.pop("x1_dir", None)
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--write", required=True, help="--inputs 아래 상대 경로")
    a = ap.parse_args(argv)
    inputs_root = Path(a.inputs).expanduser().resolve()
    cfg = dict(cf.load_json_strict(a.config))
    lst_scope.note_roots(inputs_root, None)
    lst_scope.activate(cfg.get("scope"))
    out = fill(cfg, inputs_root)
    dst = resolve(inputs_root, to_rel(a.write))
    if dst.exists():
        if cf.load_json_strict(dst) != out:
            raise FileExistsError(f"이미 있고 내용이 다름: {dst}")
        return dst
    cf.dump_json(dst, out)
    print(f"최종 설정: dir_2_0 = {out['dir_2_0']}, x1_dir = {out.get('x1_dir')} → {dst}")
    return dst


if __name__ == "__main__":
    main()
