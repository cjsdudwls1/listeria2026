# -*- coding: utf-8 -*-
"""s0_dom_prep.py — '국내만 버전' 단계 0 준비(lst/scope.py 머리 참고).

하는 일(--inputs = 국내판 결과 뿌리, --source = 지금 판 결과 뿌리 — 읽기만):
  1. 국내판 결과 뿌리 표지 `SCOPE.json`(scope = domestic) — 이 뒤 국내판 설정만 이 뿌리에서 돎.
  2. 시료와 무관한 지금 판 단계 0 결과를 읽기 전용으로 연결: <inputs>/<shared_dirs> → <source>/<같은 경로> 심볼릭 링크
     (있으면 같은 곳을 가리키는지만 확인). 국내판의 --out 은 링크를 따라간 실제 경로가 국내판 뿌리 안이어야 하므로
     (lst.scope.check_roots) 국내판 단계가 공유 폴더에 쓰는 일은 없음.
  3. 시료 표: 지금 판 0-1 시료 표(행 처리 RR1~RR8 뒤 — RR1 로 국내로 옮긴 78행 포함)에서 src_route == 'domestic' 행만,
     칼럼·형·행 순서 그대로 → <out>/apply/samples.parquet. 행 수·검출 수가 설정 expected(단계 0-1 기록)와 다르면 멈춤.
  4. 덜 찬 달(§3 규칙 — 국내·해외 따로라 국내 몫이 지금 판과 같아야 함, 다르면 멈춤)·적용 후 수·fold 별 셈.
  5. Theil's U 대상 = 지금 판 목록에서 drop_theil_columns(src_route — 해외 정보 칼럼)만 뺀 목록.
  6. 식품 고정표 공유 점검: '2단계값' 규칙의 3단계 → 2단계 짝(static_feats.pair_p2 — 시료에서 짝지어진 2단계 중 시료
     수 최다)을 국내 행으로 다시 셈 → 식품 고정표 p2_used 와 같아야 함(다르면 멈춤 — 그때는 식품 고정표를 국내 행으로
     다시 만들어야 함). 국내 시료 3단계 코드가 모두 표에 있는지도.
출력(<out>/apply/, 이미 있으면 멈춤): samples.parquet·11_counts_after_rules.csv·12_incomplete_months.csv·
  12_incomplete_months.json·14_theil_target_columns.json·15_domestic_split_counts.csv·16_shared_dirs.json·
  17_food_pairing_check.json / <out>/run_meta.json

실행(서버 — 검수 뒤 지휘자):
  B=~/cyj_listeria_final; $B/venv/bin/python -I scripts/s0_dom_prep.py --root $B/data --inputs $B/results_dom \
      --source $B/results --out $B/results_dom/s0/0-1 --config configs_dom/s0_dom_prep.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import samples as S  # noqa: E402
from lst import scope as sp  # noqa: E402
from lst import static_feats as sf  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve, to_rel  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = sp.with_field({
    "plan": cf.Field(str),
    "source_samples": cf.Field(str),            # --source 아래
    "source_theil_targets": cf.Field(str),
    "source_incomplete_months": cf.Field(str),
    "shared_dirs": cf.Field(list, item_types=str),
    "food_table": cf.Field(str),                # --inputs 아래(공유 폴더 연결 뒤)
    "expected": {"rows": cf.Field(int), "pos": cf.Field(int)},
    "drop_theil_columns": cf.Field(list, item_types=str),
})
APPLY = "apply"


class PrepStop(RuntimeError):
    pass


def load_cfg(path) -> dict:
    cfg = dict(cf.load_config(path, SCHEMA))
    if cfg.get("scope") != sp.DOMESTIC:
        raise cf.ConfigError("s0_dom_prep 는 scope = domestic 설정만")
    for d in cfg["shared_dirs"]:
        to_rel(d)
        if not d.startswith("s0/") or d.startswith("s0/0-1"):
            raise cf.ConfigError(f"공유 폴더는 시료와 무관한 단계 0 결과만: {d}")
    bad = [c for c in cfg["drop_theil_columns"] if c != sp.ROUTE_COLUMN]
    if bad:
        raise cf.ConfigError(f"Theil's U 대상에서 뺄 수 있는 칼럼은 {sp.ROUTE_COLUMN} 뿐: {bad}")
    return cfg


def _make_link(src: Path, dst: Path) -> None:
    os.symlink(src, dst, target_is_directory=True)


def _is_link(p: Path) -> bool:
    return p.is_symlink()


def link_shared(inputs_root: Path, source_root: Path, rels) -> list[dict]:
    rec = []
    for rel in rels:
        dst = resolve(inputs_root, rel)
        src = resolve(source_root, rel)
        if not src.is_dir():
            raise PrepStop(f"지금 판 단계 0 폴더 없음: {src}")
        if _is_link(dst) or dst.exists():
            if not (_is_link(dst) and dst.resolve() == src.resolve()):
                raise PrepStop(f"이미 있는데 지금 판 폴더 링크가 아님: {dst}")
            state = "있음"
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            _make_link(src.resolve(), dst)
            state = "만듦"
        rec.append({"dir": rel, "target": str(src.resolve()), "state": state})
    return rec


def domestic_table(src_path: Path):
    t = pq.read_table(src_path)
    if t.schema.names != list(S.SAMPLE_COLUMNS):
        raise PrepStop("지금 판 시료 표 칼럼이 계획 목록과 다름")
    dom = t.filter(pc.equal(t["src_route"], "domestic"))
    return t, dom


def food_pairing(dom: pd.DataFrame, food: pd.DataFrame) -> dict:
    p3 = dom["p3"].to_numpy(dtype=object)
    p2 = dom["p2"].to_numpy(dtype=object)
    real = p3 != S.NO_CODE
    pair = sf.pair_p2(p3[real], p2[real])
    tab = food.set_index("p3")["p2_used"]
    missing = sorted(set(pair) - set(tab.index))
    diff = sorted(k for k in pair if k in tab.index and tab[k] != pair[k])
    return {"domestic_p3_codes": len(pair), "codes_not_in_food_table": missing, "p2_used_differs": diff}


def main(argv=None) -> Path:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="국내판 결과 뿌리(새로 만듦)")
    ap.add_argument("--source", required=True, help="지금 판 결과 뿌리(읽기만)")
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    inputs_root = Path(args.inputs).expanduser().resolve()
    source_root = Path(args.source).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"--source 폴더 없음: {source_root}")
    if inputs_root == source_root or source_root in inputs_root.parents or inputs_root in source_root.parents:
        raise PrepStop("국내판 결과 뿌리와 지금 판 결과 뿌리는 서로 안에 있으면 안 됨")
    if sp.read_marker(source_root) is not None:
        raise PrepStop("--source 가 국내판 뿌리(SCOPE.json 있음)")
    sp.write_marker(inputs_root, cfg["scope"])
    sp.apply(cfg, inputs_root, rp.out)                 # 표지·--out 대조(--out 은 국내판 뿌리 안)
    res0 = check_resources(rp.out)      # 가용 RAM 15GB·디스크 10GB 바닥값(부록 B-1)
    out = rp.out / APPLY
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"결과 폴더가 비어 있지 않음(덮어쓰기 금지): {out}")
    out.mkdir(parents=True, exist_ok=True)

    shared = link_shared(inputs_root, source_root, cfg["shared_dirs"])
    cf.dump_json(out / "16_shared_dirs.json", {"rule": "시료와 무관한 지금 판 단계 0 결과 — 읽기 전용 심볼릭 링크",
                                               "dirs": shared})

    src_samples = resolve(source_root, cfg["source_samples"])
    full, dom_t = domestic_table(src_samples)
    dom = S._to_plain(dom_t.to_pandas())
    n, pos = len(dom), int(dom["y"].sum())
    if (n, pos) != (cfg["expected"]["rows"], cfg["expected"]["pos"]):
        raise PrepStop(f"국내 시료 {n}행·검출 {pos} ≠ 단계 0-1 기록 {cfg['expected']}")
    if dom["lims_data_key"].duplicated().any() or (dom["src_route"] != "domestic").any():
        raise PrepStop("국내 시료 표 키 중복·src_route 이상")
    pq.write_table(dom_t, out / "samples.parquet")

    S.post_counts(dom).to_csv(out / "11_counts_after_rules.csv", index=False, encoding="utf-8-sig")
    inc_t, inc = S.incomplete_months(dom)
    with open(resolve(source_root, cfg["source_incomplete_months"]), "r", encoding="utf-8") as f:
        src_inc = json.load(f)["incomplete"]
    if sorted(inc) != sorted([r, m] for r, m in src_inc if r == "domestic"):
        raise PrepStop(f"국내 덜 찬 달 {inc} ≠ 지금 판 국내 몫 {src_inc}")
    inc_t.to_csv(out / "12_incomplete_months.csv", index=False, encoding="utf-8-sig")
    cf.dump_json(out / "12_incomplete_months.json",
                 {"rule": "§3 덜 찬 달: test 기간 끝 3개월, 국내·해외 따로, 이전 3년 같은 달 평균의 70% 미만 — 국내만 버전",
                  "incomplete": inc})

    with open(resolve(source_root, cfg["source_theil_targets"]), "r", encoding="utf-8") as f:
        th = json.load(f)
    cols = [c for c in th["columns"] if c not in cfg["drop_theil_columns"]]
    cf.dump_json(out / "14_theil_target_columns.json",
                 {"columns": cols, "note": th["note"] + f" / 국내만 버전: {cfg['drop_theil_columns']} 뺌(해외 정보 칼럼)"})

    days = fd.to_days(dom["d"])
    rows = [{"split": name, **fd.count_split(days, dom["y"].to_numpy(), dom["src_route"].to_numpy(), s)}
            for name, s in fd.SPLITS.items()]
    pd.DataFrame(rows).to_csv(out / "15_domestic_split_counts.csv", index=False, encoding="utf-8-sig")

    food_p = resolve(inputs_root, cfg["food_table"])
    fchk = food_pairing(dom, pd.read_parquet(food_p, columns=["p3", "p2_used"]))
    cf.dump_json(out / "17_food_pairing_check.json", fchk)
    if fchk["codes_not_in_food_table"] or fchk["p2_used_differs"]:
        raise PrepStop(f"식품 고정표를 국내 행으로 다시 만들어야 함: {fchk}")

    md5 = {"source:" + cfg["source_samples"]: cf.file_md5(src_samples),
           "source:" + cfg["source_theil_targets"]: cf.file_md5(resolve(source_root, cfg["source_theil_targets"])),
           "source:" + cfg["source_incomplete_months"]: cf.file_md5(resolve(source_root, cfg["source_incomplete_months"])),
           "inputs:" + cfg["food_table"]: cf.file_md5(food_p)}
    cf.save_run_meta(rp.out / "run_meta.json", cfg, md5, CODE_ROOT, n_jobs=None,
                     extra={"source_root": str(source_root), "inputs_root": str(inputs_root), "resources_at_start": res0,
                            "rows": {"source": full.num_rows, "domestic": n, "domestic_pos": pos,
                                     "overseas_dropped": full.num_rows - n},
                            "incomplete_domestic": inc, "theil_columns": cols})
    print(f"[국내 준비] 시료 {n:,}행·검출 {pos:,}, 덜 찬 달 {inc}, 공유 폴더 {len(shared)}개")
    return out


if __name__ == "__main__":
    main()
