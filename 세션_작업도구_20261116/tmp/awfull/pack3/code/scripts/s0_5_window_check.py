# -*- coding: utf-8 -*-
"""s0_5_window_check.py — 단계 0-5 window 생성기 대조(부록 B-7 예시, 실제 국내·해외 일자료).

계획서: §1-1(T·E·S·R·g), §4-1(window·관측 일수), §4-2(요약·원값·시퀀스·이동평균 차감), §6 단계 0-5, 부록 B-7.

하는 일:
1. 경계: d = 2025-11-01 에서 부록 B-7 7줄(W 60 h 1·10·11·12·30·60, W 100 h 13)의 T·S·E·관측·예보·g 를
   `lst.windows.window_bounds` 값과 대조. 설정의 extra_cases(W 100 의 다른 h)는 §1-1 식을 손으로 풀어 대조.
2. 값: 국내·해외 몇 단위(설정)마다 같은 (W, h) 로 생성기(window_features 요약·원값, sequences, anomaly_cube →
   차감 요약·원값, N_ma 4값)를 돌리고, 같은 값을 긴 표(일자료 parquet)에서 날짜를 하나씩 찾아 직접 계산한 값과 대조
   (결측 위치는 똑같아야, 값은 허용오차 안).
   직접 계산: 요약 = numpy nan 함수(평균·표준편차 ddof 0·최소·최대)·np.polyfit(값 있는 날, 날짜 일수) 기울기,
   10일 구간 = E 부터 거꾸로, 차감 = 그날 값 − [t − N_ma, t − 1] 의 값 있는 날 평균(표에 없는 날·NaN 빼고).

만드는 것(--out, 기존 파일 안 덮음, --dry-run 이면 아무것도 안 만들고 화면에만):
- 0-5_경계_대조.csv: (W, h)마다 생성기·B-7(또는 식) T·S·E·R·g·관측·예보와 일치 여부
- 0-5_값_대조.csv: (단위, W, h, 항목)마다 값 수·결측 수(생성기·직접)·결측 위치 일치·최대 절대 차이·통과
- 0-5_요약.json, run_meta.json(code_md5·입력 md5·스레드·설정, B-2)
하나라도 다르면 파일을 쓴 뒤 종료 코드 1.

실행(서버, 검수 뒤 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_5_window_check.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s0/0-5
로컬 시험(파일 안 만듦):
  <venv python> scripts/s0_5_window_check.py --root <천영진> --inputs <results_mirror> --out <임시> --dry-run

구현 세부(계획서에 없음 — 담당자 확인 필요):
- 대조 단위·extra_cases·허용오차(rtol·atol 1e-9)는 설정 값. 결측 위치가 다르면 허용오차와 무관하게 불일치.
- 대조 변수 = 기상 19(습도·이슬점은 window 요약·원값으로 안 넣음 — §5 WX6). 표는 국내 합본 + 해외 1차(= 2차) 표,
  해외 습도·이슬점 이름 맞추기는 lst.wxfeat.combined_cube.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst import resources as rs  # noqa: E402

rs.set_thread_env(rs.THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import paths as ph  # noqa: E402
from lst import windows as wn  # noqa: E402
from lst import wxfeat as wf  # noqa: E402
from lst.folds import to_days  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CONFIG = "configs/s0_5_window_check.json"
B7_SCHEMA = {"W": cf.Field(int), "h": cf.Field(int), "T": cf.Field(str), "S": cf.Field(str), "E": cf.Field(str),
             "n_obs": cf.Field(int), "n_fc": cf.Field(int), "g": cf.Field(int)}
CASE_SCHEMA = {"W": cf.Field(int), "h": cf.Field(int)}
SCHEMA = {
    "plan": cf.Field(str),
    "inputs": {"domestic_weather": cf.Field(str), "overseas_weather": cf.Field(str)},
    "d": cf.Field(str),
    "b7": cf.Field(list, item_types=dict),
    "extra_cases": cf.Field(list, item_types=dict),
    "units_domestic": cf.Field(list, item_types=str),
    "units_overseas": cf.Field(list, item_types=str),
    "variables": cf.Field(list, item_types=str),
    "n_ma": cf.Field(list, item_types=int),
    "tolerance": {"rtol": cf.Field(float), "atol": cf.Field(float)},
}


def build_parser():
    ap = ph.build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="단계 0 결과 뿌리(서버 ~/cyj_listeria_final/results)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true", help="파일을 만들지 않고 화면에만")
    return ap


def D(s) -> int:
    return int(to_days(s)[0])


# ---------------------------------------------------------------- 읽기
def load_long(inputs_root: Path, cfg, units_dom, units_ov, variables):
    dom_p = ph.resolve(inputs_root, cfg["inputs"]["domestic_weather"])
    ov_p = ph.resolve(inputs_root, cfg["inputs"]["overseas_weather"])
    dom = pq.read_table(dom_p, columns=["sgg_code", "date"] + list(wf.CUBE_VARS),
                        filters=[("sgg_code", "in", list(units_dom))]).to_pandas()
    ov = pq.read_table(ov_p, columns=["alpha_3", "sgg", "date"] + list(wf.WEATHER19) + list(wf.OVERSEAS_RENAME),
                       filters=[("sgg", "in", list(units_ov))]).to_pandas()
    for name, df, col, want in (("국내", dom, "sgg_code", units_dom), ("해외", ov, "sgg", units_ov)):
        miss = sorted(set(want) - set(df[col].astype(str)))
        if miss:
            raise KeyError(f"{name} 일자료 표에 없는 단위: {miss}")
    cube = wf.combined_cube(dom, ov, units=list(units_dom) + list(units_ov))
    cube = cube.select(variables)
    # 직접 계산용 긴 표: 단위 → (날짜 일수 → 값) 표(국내·해외 같은 꼴)
    long = pd.concat([dom.rename(columns={"sgg_code": "unit"})[["unit", "date"] + list(variables)],
                      ov.rename(columns={"sgg": "unit"})[["unit", "date"] + list(variables)]], ignore_index=True)
    long["unit"] = long["unit"].astype(str)
    long["day"] = to_days(long["date"])
    ref = {u: g.set_index("day")[list(variables)].astype(np.float64) for u, g in long.groupby("unit")}
    return cube, ref, [dom_p, ov_p]


# ---------------------------------------------------------------- 직접 계산(표에서 날짜를 찾아)
def _span(tab: pd.DataFrame, var: str, lo: int, hi: int) -> np.ndarray:
    """lo ~ hi(포함) 날짜 값(표에 없는 날 = NaN)."""
    return tab[var].reindex(range(lo, hi + 1)).to_numpy(dtype=np.float64)


def _stat(x, f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(f(x)) if (~np.isnan(x)).any() else np.nan


def ref_summary(x: np.ndarray, days: np.ndarray, W: int) -> list[float]:
    """x = S ~ E 시간순."""
    rev = x[::-1]
    out = [_stat(rev[k * 10:(k + 1) * 10], np.nanmean) for k in range(W // 10)]
    out += [_stat(x, np.nanmean), _stat(x, lambda v: np.nanstd(v, ddof=0)), _stat(x, np.nanmin), _stat(x, np.nanmax)]
    ok = ~np.isnan(x)
    out.append(float(np.polyfit(days[ok].astype(np.float64), x[ok], 1)[0]) if ok.sum() >= 2 else np.nan)
    return out


def ref_anom(vals: dict, S: int, E: int, n_ma: int) -> np.ndarray:
    """vals = 날짜 일수 → 값(그 단위·변수, 표에 있는 날만). 날마다 앞 n_ma 일을 하나씩 찾아 평균."""
    a = []
    for t in range(S, E + 1):
        xt = vals.get(t, np.nan)
        prev = [vals.get(t - k, np.nan) for k in range(1, n_ma + 1)]
        prev = [p for p in prev if not np.isnan(p)]
        a.append(xt - sum(prev) / len(prev) if (prev and not np.isnan(xt)) else np.nan)
    return np.array(a, dtype=np.float64)


def compare(a, b, tol) -> dict:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.isnan(a), np.isnan(b)
    same_nan = a.shape == b.shape and bool(np.array_equal(na, nb))
    ok = ~(na | nb)
    diff = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
    close = bool(np.allclose(a[ok], b[ok], rtol=tol["rtol"], atol=tol["atol"])) if ok.any() else True
    return {"n_values": int(a.size), "n_nan_gen": int(na.sum()), "n_nan_ref": int(nb.sum()),
            "nan_same": same_nan, "max_abs_diff": diff, "pass": same_nan and close}


# ---------------------------------------------------------------- 경계
def bounds_rows(cfg) -> pd.DataFrame:
    d = D(cfg["d"])
    rows = []
    for e in cfg["b7"]:
        cf.validate(e, B7_SCHEMA, where="b7[].")
        rows.append(dict(e, src="부록 B-7", T=D(e["T"]), S=D(e["S"]), E=D(e["E"])))
    for e in cfg["extra_cases"]:
        cf.validate(e, CASE_SCHEMA, where="extra_cases[].")
        W, h = e["W"], e["h"]
        T = d - h
        E = min(d, T + 10)          # §1-1 식을 손으로
        rows.append({"W": W, "h": h, "src": "§1-1 식", "T": T, "S": E - W + 1, "E": E,
                     "n_obs": max(0, W - min(h + 1, 11)), "n_fc": W - max(0, W - min(h + 1, 11)), "g": max(0, h - 10)})
    out = []
    for r in rows:
        b = wn.window_bounds(d, r["h"], r["W"])
        fc = wn.forecast_day_mask(b)[0]
        R_exp = min(d, r["E"] + 1)
        gen = {"T": int(b.T[0]), "S": int(b.S[0]), "E": int(b.E[0]), "R": int(b.R[0]), "g": int(b.g[0]),
               "n_obs": int(b.n_obs[0]), "n_fc": int(b.n_fc[0])}
        exp = {"T": r["T"], "S": r["S"], "E": r["E"], "R": R_exp, "g": r["g"], "n_obs": r["n_obs"], "n_fc": r["n_fc"]}
        ok = gen == exp and int(fc.sum()) == r["n_fc"] and int((~fc).sum()) == r["n_obs"]
        rec = {"W": r["W"], "h": r["h"], "기준": r["src"]}
        for k in exp:
            v_g, v_e = gen[k], exp[k]
            if k in ("T", "S", "E", "R"):
                v_g, v_e = str(np.datetime64(v_g, "D")), str(np.datetime64(v_e, "D"))
            rec[f"{k}_생성기"], rec[f"{k}_기준"] = v_g, v_e
        rec["예보칸표시_수"] = int(fc.sum())
        rec["일치"] = bool(ok)
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 값
def value_rows(cube, ref, cfg, cases, variables, tol) -> pd.DataFrame:
    d = D(cfg["d"])
    units = list(cfg["units_domestic"]) + list(cfg["units_overseas"])
    recs = []
    vdict = {u: {v: ref[u][v].to_dict() for v in variables} for u in units}
    for W in sorted({c["W"] for c in cases}):
        hs = [c["h"] for c in cases if c["W"] == W]
        codes = np.array([u for u in units for _ in hs])
        hh = np.array([h for _ in units for h in hs])
        dd = np.full(hh.shape, d)
        b = wn.window_bounds(dd, hh, W)
        gen = {"요약": wn.window_features(cube, codes, dd, hh, W, "sum"),
               "원값": wn.window_features(cube, codes, dd, hh, W, "raw")}
        seq = wn.sequences(cube, codes, dd, hh, W)
        anom = {}
        for n in cfg["n_ma"]:
            ac = wn.anomaly_cube(cube, n)
            anom[n] = (wn.window_features(ac, codes, dd, hh, W, "sum", anomaly=True),
                       wn.window_features(ac, codes, dd, hh, W, "raw", anomaly=True))
        for i, (u, h) in enumerate(zip(codes, hh)):
            S, E = int(b.S[i]), int(b.E[i])
            days = np.arange(S, E + 1)
            tab = ref[u]
            xs = {v: _span(tab, v, S, E) for v in variables}
            items = {
                "요약": (gen["요약"].iloc[i].to_numpy(), sum((ref_summary(xs[v], days, W) for v in variables), [])),
                "원값": (gen["원값"].iloc[i].to_numpy(), np.concatenate([xs[v][::-1] for v in variables])),
                "시퀀스": (seq[i], np.stack([xs[v] for v in variables], axis=1)),
            }
            vals = vdict[u]
            for n in cfg["n_ma"]:
                a = {v: ref_anom(vals[v], S, E, n) for v in variables}
                last = days >= E - 9
                ref_sum = []
                for v in variables:
                    ref_sum += [_stat(a[v], np.nanmean), _stat(a[v][last], np.nanmean), _stat(a[v], np.nanmax)]
                items[f"차감{n}_요약"] = (anom[n][0].iloc[i].to_numpy(), ref_sum)
                items[f"차감{n}_원값"] = (anom[n][1].iloc[i].to_numpy(), np.concatenate([a[v][::-1] for v in variables]))
            for k, (g, r) in items.items():
                recs.append({"unit": u, "W": W, "h": int(h), "S": str(np.datetime64(S, "D")),
                             "E": str(np.datetime64(E, "D")), "항목": k, **compare(g, r, tol)})
    return pd.DataFrame(recs)


def run(argv=None) -> int:
    rp, args = ph.parse_paths(argv, build_parser())
    inputs_root = Path(args.inputs).expanduser().resolve()
    if not inputs_root.is_dir():
        raise FileNotFoundError(f"--inputs 폴더 없음: {inputs_root}")
    cfg = cf.load_config(CODE_ROOT / args.config if not Path(args.config).is_absolute() else args.config, SCHEMA)
    variables = list(cfg["variables"])
    cf.check_columns(variables, wf.WEATHER19, "기상 변수")
    for n in cfg["n_ma"]:
        if n not in (5, 7, 15, 30):
            raise cf.ConfigError(f"N_ma 는 2-1b 후보 5·7·15·30 만(§6): {n}")
    cube, ref, in_paths = load_long(inputs_root, cfg, cfg["units_domestic"], cfg["units_overseas"], variables)
    if not args.dry_run:
        rs.check_resources(rp.out)
    bt = bounds_rows(cfg)
    cases = [{"W": r["W"], "h": r["h"]} for r in cfg["b7"]] + list(cfg["extra_cases"])
    vt = value_rows(cube, ref, cfg, cases, variables, cfg["tolerance"])
    n_bad_b = int((~bt["일치"]).sum())
    n_bad_v = int((~vt["pass"]).sum())
    summary = {"경계_사례": len(bt), "경계_불일치": n_bad_b, "값_대조": len(vt), "값_불일치": n_bad_v,
               "단위": list(cfg["units_domestic"]) + list(cfg["units_overseas"]),
               "최대_절대_차이": float(vt["max_abs_diff"].max()),
               "항목별_결측_생성기": vt.groupby("항목")["n_nan_gen"].sum().to_dict()}
    print(bt.to_string(index=False))
    print(vt.groupby(["W", "항목"])[["n_values", "n_nan_gen", "max_abs_diff", "pass"]]
          .agg({"n_values": "sum", "n_nan_gen": "sum", "max_abs_diff": "max", "pass": "all"}).to_string())
    print(summary)
    if not args.dry_run:
        out = rp.out
        for name, df in (("0-5_경계_대조.csv", bt), ("0-5_값_대조.csv", vt)):
            p = out / name
            if p.exists():
                raise FileExistsError(f"덮어쓰기 금지: {p}")
            df.to_csv(p, index=False, encoding="utf-8-sig")
        cf.dump_json(out / "0-5_요약.json", cf.nan_to_null(summary))
        md5 = {f"inputs/{k}": cf.file_md5(p) for k, p in zip(cfg["inputs"], in_paths)}
        cf.save_run_meta(out / "run_meta.json", cfg, md5, CODE_ROOT, n_jobs=None,
                         extra={"inputs_root": "--inputs", "dry_run": False})
    if n_bad_b or n_bad_v:
        print(f"[0-5] 불일치: 경계 {n_bad_b}, 값 {n_bad_v}")
        return 1
    print("[0-5] 전부 일치")
    return 0


if __name__ == "__main__":
    sys.exit(run())
