# -*- coding: utf-8 -*-
"""s0_1_samples.py — 단계 0-1·0-2 시료 표(계획서 §2·§3·§6 단계 0-1·0-2·부록 A-1·A-2·A-6·B-6).

두 단계로 나눠 돎('담당자 확인' 을 스크립트가 정하지 않게 — 계획서 §6 0-1 "원천은 부록 B-6 대조·담당자 확인 뒤",
"Theil's U 대상 칼럼 이름 목록 고정(부록 B-6 대조·담당자 확인 뒤)"):

  --stage check : §2-1 다시 셈·§3 기록값 대조(다르면 멈춤)·RR1 좌표 점검표·부록 B-6 확인용 표
                  (d ≤ 2021-06-30 수치표 + d ≤ 2025-06-30 은 이름만 목록, 수치는 봉인 폴더)·
                  A-2 원천 전부를 쓴 잠정 RR 표. 시료 표는 만들지 않음.
  --stage apply --config <설정> : 담당자가 확인한 나라 코드 원천·Theil's U 대상 칼럼(설정 JSON, confirmed=true)
                  으로 RR1~RR8·운영 단위 칼럼을 적용해 시료 표·적용 후 수·덜 찬 달 점검표·0-2 나라별 표를 만듦.

실행(서버, 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_1_samples.py --stage check \
      --root ~/cyj_listeria_final/data --out ~/cyj_listeria_final/results/s0/0-1
  ~/cyj_listeria_final/venv/bin/python scripts/s0_1_samples.py --stage apply --config configs/s0_1_samples.json \
      --root ~/cyj_listeria_final/data --out ~/cyj_listeria_final/results/s0/0-1
결과 = <out>/check/ · <out>/apply/ (폴더가 이미 있고 비어 있지 않으면 멈춤 — 덮어쓰기 금지).
--config 의 상대 경로는 코드 폴더(이 파일의 위 폴더) 기준.

입력(자료 뿌리 아래) = lst.samples.INPUTS. `02_시군구_배정.py`(openpyxl 필요)·법정동 표·대응표도 뿌리에 있어야 함.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.paths import build_parser, parse_paths  # noqa: E402  (표준 라이브러리만 씀)
from lst.resources import check_resources, set_thread_env  # noqa: E402

set_thread_env()  # numpy 를 불러오기 전에(부록 B-1 스레드 8)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lst import config as C  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import samples as S  # noqa: E402
from lst.geo import CountryPolygons  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEALED_DIR = "sealed_until_final_report"  # 부록 B-6: d ≤ 2025-06-30 수치 — 최종 보고 전까지 안 보여 줌

CONFIG_SCHEMA = {
    "nation_sources": C.Field(list, item_types=str),   # 부록 A-2 순서 안에서 담당자가 확인한 원천
    "theil_columns": C.Field(list, item_types=str),    # 부록 A-6 후보 안에서 담당자가 확인한 칼럼
    "confirmed": C.Field(bool),
    "confirmation_note": C.Field(str),                 # 확인한 사람·시점·본 자료(§1-3 기록)
}


def load_apply_config(path: Path) -> dict:
    cfg = dict(C.load_config(path, CONFIG_SCHEMA))
    if cfg["confirmed"] is not True or not cfg["confirmation_note"].strip():
        raise C.ConfigError("담당자 확인 전 설정(confirmed=true·confirmation_note 필요) — apply 안 함")
    ns, tc = cfg["nation_sources"], cfg["theil_columns"]
    C.check_columns(ns, S.NATION_SOURCES, "나라 코드 원천")
    C.check_columns(tc, S.THEIL_CANDIDATES, "Theil's U 대상 칼럼")
    for name, v, order in (("nation_sources", ns, S.NATION_SOURCES), ("theil_columns", tc, S.THEIL_CANDIDATES)):
        if len(set(v)) != len(v):
            raise C.ConfigError(f"{name} 중복")
        if [x for x in order if x in v] != list(v):
            raise C.ConfigError(f"{name} 순서가 계획서(부록 A-2·A-6) 순서와 다름")
    if not tc:
        raise C.ConfigError("theil_columns 비어 있음")
    return cfg


class Writer:
    def __init__(self, d: Path):
        self.dir = d
        self.files: list[str] = []

    def _p(self, name: str) -> Path:
        p = self.dir / name
        if p.exists():
            raise FileExistsError(f"덮어쓰기 금지: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        self.files.append(name)
        return p

    def csv(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self._p(name), index=False, encoding="utf-8-sig")

    def json(self, name: str, obj) -> None:
        C.dump_json(self._p(name), obj)

    def parquet(self, name: str, df: pd.DataFrame) -> None:
        t = pa.Table.from_pandas(df, preserve_index=False)
        if "d" in df.columns:
            t = t.set_column(t.schema.get_field_index("d"), "d", pa.array(df["d"].to_numpy().astype("datetime64[D]")))
        pq.write_table(t, self._p(name))


def stage_dir(out: Path, stage: str) -> Path:
    d = out / stage
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"결과 폴더가 비어 있지 않음(덮어쓰기 금지): {d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(argv=None, recorded_split=None) -> Path:
    ap = build_parser("단계 0-1·0-2 시료 표")
    ap.add_argument("--stage", required=True, choices=("check", "apply"))
    ap.add_argument("--config", help="apply 단계 설정 JSON(상대 경로 = 코드 폴더 기준)")
    paths, args = parse_paths(argv, ap)
    recorded_split = fd.RECORDED_COUNTS if recorded_split is None else recorded_split
    cfg = None
    if args.stage == "apply":
        if not args.config:
            raise SystemExit("--stage apply 는 --config 필요")
        cp = Path(args.config)
        cfg = load_apply_config(cp if cp.is_absolute() else CODE_ROOT / cp)
    elif args.config:
        raise SystemExit("--stage check 는 --config 를 받지 않음")
    res = check_resources(paths.out)  # 가용 RAM 15GB·디스크 10GB 바닥값(부록 B-1) — 모자라면 아무것도 안 만들고 멈춤
    out = stage_dir(paths.out, args.stage)
    W = Writer(out)
    status, err = "ok", None
    try:
        _run(paths, args.stage, cfg, W, recorded_split)
    except BaseException as e:  # 멈춤도 메타에 남김
        status, err = "stopped", f"{type(e).__name__}: {e}"
        raise
    finally:
        try:
            meta_extra = {"stage": args.stage, "argv": list(sys.argv if argv is None else argv), "status": status,
                          "error": err, "resources_at_start": res, "outputs": list(W.files),
                          "traceback": traceback.format_exc() if err else None}
            C.save_run_meta(out / "run_meta.json", cfg, C.inputs_md5(paths.root, S.INPUTS.values()), CODE_ROOT,
                            n_jobs=None, extra=meta_extra)
        except Exception as e2:  # 메타 저장 실패는 원 오류를 가리지 않게 알리기만
            print(f"[경고] run_meta 저장 실패: {e2}", file=sys.stderr)
    return out


def _run(paths, stage: str, cfg, W: Writer, recorded_split) -> None:
    P = {k: paths.data(v) for k, v in S.INPUTS.items()}
    # ---- 읽기
    raw = S.read_parquet(P["sample"])
    n_cols = raw.shape[1]
    S.check_no_empty_strings(raw, [c for c in raw.columns if raw[c].dtype == object])
    s = S.prepare_basic(raw)
    del raw
    sgg_t = S.read_parquet(P["sgg"], columns=["lims_data_key", "sgg_code", "lat", "lon"])
    if sgg_t["lims_data_key"].duplicated().any() or sgg_t["sgg_code"].isna().any():
        raise ValueError("시료_시군구 키 중복·sgg_code 결측")
    if not sgg_t["lims_data_key"].isin(s.loc[s["src_route"] == "domestic", "lims_data_key"]).all():
        raise ValueError("시료_시군구에 시료 파일 국내 행이 아닌 키")
    sgg_map = sgg_t.set_index("lims_data_key")["sgg_code"]
    rr2_sum = pd.read_csv(P["rr2_summary"], encoding="utf-8-sig", dtype={"원래_시군구코드": str})
    nc = S.read_parquet(P["nation_code"])
    a2_to_a3, a3_set = S.nation_maps(nc)
    nsrc = S.load_nation_sources(s["lims_data_key"], S.read_parquet(P["hazard"]), S.read_parquet(P["s_org"]))
    polys = CountryPolygons.from_geojson(P["ne"])
    sgg02 = S.load_sgg02(P["sgg02"])
    region = sgg02.Region(str(P["krc"]))
    mapping = sgg02.load_mapping(str(P["sgg_xlsx"]))
    krc = S.read_parquet(P["krc"], columns=["legal_dong_code", "legal_dong_nm", "molit_use_yn", "level", "lat", "lon"])

    # ---- §2-1 다시 셈(다르면 보고)
    W.csv("01_basic_counts.csv", S.basic_counts(s, n_cols, len(sgg_t)))

    # ---- RR2 대조(요약 CSV 와 다르면 멈춤)
    rr2 = S.rr2_mask(s, set(sgg_t["lims_data_key"]))
    rr2_t = S.rr2_check(s, rr2, rr2_sum)
    W.csv("02_rr2_vs_summary.csv", rr2_t.drop(columns=["pos", "검출수"]).assign(
        pos_match=(rr2_t["pos"] == rr2_t["검출수"])))
    if not rr2_t["match"].all():
        raise S.PlanCountMismatch("RR2 행이 시군구_미배정_삭제행_요약.csv 와 다름 — 멈춤")

    # ---- §3 기록값 대조(시료 파일 − RR2, 그 밖 규칙 전, 원 src_route) — 다르면 멈춤
    keep0 = ~rr2
    t3 = S.split_count_table(s.loc[keep0, "d"], s.loc[keep0, "y"], s.loc[keep0, "src_route"], recorded_split)
    W.csv("03_split_counts_vs_plan.csv", t3)
    if not t3["match"].all():
        bad = t3.loc[~t3["match"], ["split", "item", "recorded", "observed"]].to_dict("records")
        raise S.PlanCountMismatch(f"§3 기록값과 다름(멈춤): {bad}")

    # ---- 좌표 나라(해외 행)·RR1
    ov0 = (s["src_route"] == "overseas").to_numpy()
    cc = S.coord_countries_for(s, ov0, polys)
    rr1 = S.apply_rr1(s, cc, sgg02, region, mapping, krc, sgg_t)
    W.csv("04_rr1_coords.csv", rr1.coord_table)
    route = np.where(rr1.moved, "domestic", s["src_route"].to_numpy()).astype(object)

    if stage == "check":
        _check(s, route, rr2, nsrc, a2_to_a3, a3_set, cc, rr1, sgg_map, nc, W)
    else:
        _apply(s, cfg, rr2, nsrc, cc, rr1, sgg_map, nc, W)


def _check(s, route, rr2, nsrc, a2_to_a3, a3_set, cc, rr1, sgg_map, nc, W: Writer) -> None:
    rows = S.check_rows_mask(s, route, rr2)
    c21, s21 = S.b6_tables(s, route, rows, nsrc, a2_to_a3, a3_set, S.B6_JUDGE_CUT)
    c25, s25 = S.b6_tables(s, route, rows, nsrc, a2_to_a3, a3_set, S.B6_SEALED_CUT)
    W.csv(f"05_b6_columns_d_le_{S.B6_JUDGE_CUT}.csv", c21)
    W.csv(f"05_b6_nation_sources_d_le_{S.B6_JUDGE_CUT}.csv", s21)
    W.csv(f"{SEALED_DIR}/b6_columns_d_le_{S.B6_SEALED_CUT}.csv", c25)
    W.csv(f"{SEALED_DIR}/b6_nation_sources_d_le_{S.B6_SEALED_CUT}.csv", s25)
    names = {
        "rule": "부록 B-6: d ≤ 2025-06-30 의 |검출 빈 칸 비율 − 불검출 빈 칸 비율| 이 d ≤ 2021-06-30 때보다 큰 이름(수치 없음)",
        "columns": S.gap_increased_names(c21, c25, "column"),
        "nation_sources": S.gap_increased_names(s21, s25, "source"),
        "sealed_dir_note": f"{SEALED_DIR}/ 의 수치는 판단 자료로 안 쓰고 최종 보고 전까지 열지 않음",
    }
    W.json("05_b6_names_gap_increased_d_le_2025-06-30.json", names)
    # 잠정(A-2 원천 전부) — 담당자 확인용 셈, 시료 표는 안 만듦
    b = S.build_samples(s, cc, rr1, rr2, sgg_map, nsrc, S.NATION_SOURCES, nc)
    W.csv("06_rr_table_provisional_all_sources.csv", b.rr_table)
    W.csv("06_nation_source_usage_provisional.csv", b.nation_usage)
    W.csv("06_rr4_rows_provisional.csv", b.rr4_rows)
    W.json("07_handoff_to_conductor.json", {
        "담당자 확인 필요": [
            "나라 코드 원천(부록 A-2 순서): 05_b6_nation_sources_* 와 이름 목록을 보고 쓸 원천을 정함 → 설정 nation_sources",
            "Theil's U 대상 칼럼(부록 A-6): 05_b6_columns_* 와 이름 목록을 보고 '검사 결과 전에 정해지는 칼럼' 만 → 설정 theil_columns",
            "RR1 좌표 → 시군구(구현 세부 a): 04_rr1_coords.csv",
            "RR4 좌표·주소로 정한 나라(구현 세부 c): 06_rr4_rows_provisional.csv",
            "계획서 기록값과 다른 셈: 01_basic_counts.csv·06_rr_table_provisional_all_sources.csv 의 match=False",
        ],
        "candidates": {"nation_sources": list(S.NATION_SOURCES), "theil_columns": list(S.THEIL_CANDIDATES)},
    })
    print("[check] 끝 — 담당자 확인 뒤 --stage apply --config 로")


def _apply(s, cfg, rr2, nsrc, cc, rr1, sgg_map, nc, W: Writer) -> None:
    b = S.build_samples(s, cc, rr1, rr2, sgg_map, nsrc, cfg["nation_sources"], nc)
    W.parquet("samples.parquet", b.table)
    W.parquet("excluded_rows.parquet", b.excluded)
    W.csv("10_rr_table.csv", b.rr_table)
    W.csv("10_nation_source_usage.csv", b.nation_usage)
    W.csv("10_rr4_rows.csv", b.rr4_rows)
    W.csv("10_rr7_by_year.csv", b.rr7_by_year)
    W.csv("11_counts_after_rules.csv", S.post_counts(b.table))
    inc_t, inc = S.incomplete_months(b.table)
    W.csv("12_incomplete_months.csv", inc_t)
    W.json("12_incomplete_months.json", {"rule": "§3 덜 찬 달: test 기간 끝 3개월, 국내·해외 따로, 이전 3년 같은 달 평균의 70% 미만",
                                          "incomplete": inc})
    nt, ninfo = S.nation_table(b.table, nc)
    W.csv(f"13_nation_table_d_le_{S.B6_JUDGE_CUT}.csv", nt)
    W.json("13_nation_count.json", ninfo)
    W.json("14_theil_target_columns.json", {"columns": cfg["theil_columns"], "note": cfg["confirmation_note"]})
    W.json("14_nation_sources.json", {"sources": cfg["nation_sources"], "note": cfg["confirmation_note"]})
    print(f"[apply] 시료 표 {len(b.table):,}행, 제외 {len(b.excluded):,}행, 덜 찬 달 {inc}")


if __name__ == "__main__":
    run()
