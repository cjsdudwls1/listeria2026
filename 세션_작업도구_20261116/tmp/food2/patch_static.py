from pathlib import Path
p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\scripts\s0_6_static.py")
s = p.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old, s.count(old))
    s = s.replace(old, new)


rep('''- rg 품목 4칼럼 float32(build_food_db_table.py 와 같음). 표에 없는 코드('코드없음') → FD1·rg 결측, aw_label null.
''', '''- rg 품목 4칼럼 float32(build_food_db_table.py 와 같음). 표에 없는 코드('코드없음') → FD1·rg 결측, aw_label null.
- 설정 aw_source_filter(없으면 'original' = 옛 부록 A-4 'aw 는 aw_source = original 만'): 'all' = aw·ph·na 모두 출처와
  상관없이 표 값 그대로(사용자 결정 2026-11-12 — 식품 고정표 v2, configs/s0_6_static_food2.json). 보충 규칙(하위 중앙값 →
  2단계 값 → 결측)·범위·부등호·음수 원문 처리는 그대로.
- 식품_값출처_세기.csv: 항목(aw·ph·na) × 값 출처(자기값·하위중앙·2단계값·결측) 3단계 코드 수(+ 자기값의 aw_source 분포는
  점검.json FD1.self_value_aw_source, 이전 0-6 고정표와 코드별 비교는 FD1.vs_prev_static — 설정 results_inputs.prev_food_static).
- 설정 missing_check(선택): 이 고정표로 d ≤ y_cut 시료의 식품 칼럼(food_aw·ph·na, rg 품목 4)에 결측이 남는지 보고 기존
  결측 표시 목록(missing_lists.json order2)과 대조(gr2 9칼럼은 aw·pH 결측 행이 남으면 목록에 그대로 — 결측 규칙 부록 A-3).
  달라지면 결과 파일을 다 쓴 뒤 멈춤(0-6_missing 을 새 고정표로 다시 만들어야 함 — 대기열이 멈춤).
''')
rep('''    "inputs": {k: cf.Field(str) for k in INPUT_KEYS},
    "results_inputs": {"samples": cf.Field(str)},
''', '''    "inputs": {k: cf.Field(str) for k in INPUT_KEYS},
    "results_inputs": {"samples": cf.Field(str), "prev_food_static": cf.Field(str, required=False)},
    "aw_source_filter": cf.Field(str, required=False, choices=sf.AW_SOURCE_FILTERS),
    "missing_check": {"missing_lists": cf.Field(str), "y_cut": cf.Field(str)},
''')
rep('''       "area": "면적_출처.json", "check": "점검.json", "meta": "run_meta.json"}''',
    '''       "area": "면적_출처.json", "check": "점검.json", "meta": "run_meta.json", "food_src": "식품_값출처_세기.csv"}
OPTIONAL = ("missing_check",)
# 결측 표시 목록 대조 대상(식품 고정표에서 바로 나오는 표 칼럼) — gr2 는 aw·pH 결측이면 결측(부록 A-3)
MISSING_FOOD_VARS = sf.FD1_COLUMNS[:3] + sf.RG_ITEM_COLUMNS
GR2_VARS = tuple(f"gr2_{p}_{s}" for p in ("mu", "pscotta", "pl5") for s in ("mean", "mean10", "max"))''')
rep('''def load_cfg(path):
    cfg = cf.load_config(path, SCHEMA)
''', '''def load_cfg(path):
    raw = cf.load_json_strict(path)
    cfg = cf.validate(raw, {k: v for k, v in SCHEMA.items() if k not in OPTIONAL or k in raw})
''')
rep('''def run_food(P, samples, chk):''', '''def run_food(P, samples, chk, cfg=None, prev_static=None):''')
rep('''    ft = sf.build_food_table(cl, S._to_plain(awt.to_pandas()), codes3, p2_of)
    t = ft.table
''', '''    awf = (cfg or {}).get("aw_source_filter", "original")
    aw_df = S._to_plain(awt.to_pandas())
    ft = sf.build_food_table(cl, aw_df, codes3, p2_of, aw_source_filter=awf)
    t = ft.table
''')
rep('''        "vs_prev_db_table_20261027": {"common_codes": len(common), **cmp},
    }
    return t
''', '''        "vs_prev_db_table_20261027": {"common_codes": len(common), **cmp},
        "aw_source_filter": awf,
        "codes_not_in_t_aw_cl": sorted(set(codes3) - set(aw_df["prdlst_cd"])),
        "self_value_aw_source": self_value_sources(t, aw_df),
    }
    if prev_static is not None:
        chk["FD1"]["vs_prev_static"] = compare_static(t, prev_static)
    return t


def source_counts(t: pd.DataFrame) -> pd.DataFrame:
    """항목 × 값 출처 → 3단계 코드 수(출처 4종 전부 — 0 포함)."""
    rows = [{"항목": c, "값 출처": r, "3단계 코드 수": int((t[f"{c}_rule"] == r).sum())}
            for c in sf.FOOD_COLS for r in sf.RULES]
    out = pd.DataFrame(rows)
    tot = out.groupby("항목")["3단계 코드 수"].sum()
    if (tot != len(t)).any():
        raise BuildStop(f"값 출처 합이 코드 수와 다름: {tot.to_dict()}")
    return out


def self_value_sources(t: pd.DataFrame, aw_df: pd.DataFrame) -> dict:
    """자기값을 쓴 3단계 코드의 t_aw_cl.aw_source 분포(항목마다)."""
    src = dict(zip(aw_df["prdlst_cd"], aw_df["aw_source"]))
    return {c: {str(k): int(v) for k, v in Counter(src.get(k3) for k3 in t.loc[t[f"{c}_rule"] == "자기값", "p3"]).items()}
            for c in sf.FOOD_COLS}


def compare_static(t: pd.DataFrame, prev: pd.DataFrame) -> dict:
    """이전 단계 0-6 고정표(결과 폴더)와 코드마다 값·출처 비교(점검용)."""
    pv = prev.set_index("p3")
    tt = t.set_index("p3")
    if set(pv.index) != set(tt.index):
        return {"same_codes": False, "only_new": sorted(set(tt.index) - set(pv.index)),
                "only_prev": sorted(set(pv.index) - set(tt.index))}
    out = {"same_codes": True}
    for c in sf.FOOD_COLS:
        a = tt[f"food_{c}"].to_numpy(dtype=float)
        b = pv.loc[tt.index, f"food_{c}"].to_numpy(dtype=float)
        same = (np.isnan(a) & np.isnan(b)) | (a == b)
        out[c] = {"값_다른_코드_수": int((~same).sum()),
                  "결측→값": int((np.isnan(b) & ~np.isnan(a)).sum()), "값→결측": int((~np.isnan(b) & np.isnan(a)).sum()),
                  "이전_출처": {str(k): int(v) for k, v in Counter(pv.loc[tt.index, f"{c}_rule"]).items()},
                  "지금_출처": {str(k): int(v) for k, v in Counter(tt[f"{c}_rule"]).items()}}
    return out


def missing_list_check(food: pd.DataFrame, samples: pd.DataFrame, lists: dict, y_cut: str) -> dict:
    """d ≤ y_cut 시료에서 식품 칼럼 결측이 남는지 → 기존 2차 결측 표시 목록과 같은지(설명은 머리 구현 세부)."""
    cut = int(np.datetime64(y_cut, "D").astype(np.int64))
    m = day_ints(samples["d"]) <= cut
    sub = samples.loc[m]
    f = sf.food_features(food, sub["p3"].to_numpy(dtype=object))
    old = set(lists["order2"]["variables"])
    rec, diff = {}, []
    for c in MISSING_FOOD_VARS:
        n = int(np.isnan(f[c].to_numpy(dtype=np.float64)).sum())
        now, before = n > 0, c in old
        rec[c] = {"nan_rows_d_le_cut": n, "in_list_now": now, "in_list_before": before}
        if now != before:
            diff.append(c)
    aw_ph_nan = bool(np.isnan(f["food_aw"].to_numpy(dtype=np.float64)).any()
                     or np.isnan(f["food_ph"].to_numpy(dtype=np.float64)).any())
    for c in GR2_VARS:
        before = c in old
        if aw_ph_nan:   # aw·pH 결측 행이 있으면 gr2 결측 → 목록에 있음
            now = True
        elif before:    # 식품 결측이 없어졌으면 기상 결측으로만 정해짐 — 여기서 못 가림
            now = None
        else:
            now = False
        rec[c] = {"in_list_now": now, "in_list_before": before}
        if now is None or now != before:
            diff.append(c)
    return {"y_cut": y_cut, "rows_d_le_cut": int(m.sum()), "variables": rec, "changed": diff, "same": not diff}
''')
rep('''    chk = {"samples": {"rows": len(samples), "pos": int(samples["y"].sum())}}
    food = run_food(P, samples, chk)
''', '''    chk = {"samples": {"rows": len(samples), "pos": int(samples["y"].sum())}}
    prev_rel = cfg["results_inputs"].get("prev_food_static")
    prev_static = S.read_parquet(ph.resolve(inputs_root, prev_rel)) if prev_rel else None
    food = run_food(P, samples, chk, cfg, prev_static)
    mchk = None
    if "missing_check" in cfg:
        lists = json.loads(ph.resolve(inputs_root, cfg["missing_check"]["missing_lists"]).read_text(encoding="utf-8"))
        mchk = chk["missing_list_check"] = missing_list_check(food, samples, lists, cfg["missing_check"]["y_cut"])
''')
rep('''    pq.write_table(pa.Table.from_pandas(food, preserve_index=False), rp.result(OUT["food"]))
''', '''    pq.write_table(pa.Table.from_pandas(food, preserve_index=False), rp.result(OUT["food"]))
    with open(rp.result(OUT["food_src"]), "x", newline="", encoding="utf-8-sig") as fh:
        source_counts(food).to_csv(fh, index=False)
''')
rep('''    md5.update({f"results/{v}": cf.file_md5(ph.resolve(inputs_root, v)) for v in cfg["results_inputs"].values()})
    cf.save_run_meta(rp.result(OUT["meta"]), cfg, md5, CODE_ROOT, n_jobs=None)
    print(f"끝: {rp.out}")
    return rp.out
''', '''    md5.update({f"results/{v}": cf.file_md5(ph.resolve(inputs_root, v)) for v in cfg["results_inputs"].values()})
    if "missing_check" in cfg:
        v = cfg["missing_check"]["missing_lists"]
        md5[f"results/{v}"] = cf.file_md5(ph.resolve(inputs_root, v))
    cf.save_run_meta(rp.result(OUT["meta"]), cfg, md5, CODE_ROOT, n_jobs=None)
    print("식품 값 출처(3단계 코드 수):", {c: dict(Counter(food[f"{c}_rule"])) for c in sf.FOOD_COLS})
    if mchk is not None and not mchk["same"]:
        raise BuildStop(f"결측 표시 목록이 바뀜(식품 칼럼 {mchk['changed']}) — 결과는 썼음. 0-6_missing 을 새 고정표로 "
                        f"다시 만들어야 함(지휘자 확인): {rp.result(OUT['check'])}")
    if mchk is not None:
        print("결측 표시 목록 대조: 같음 — 기존 0-6_missing 그대로 씀")
    print(f"끝: {rp.out}")
    return rp.out
''')
p.write_text(s, encoding="utf-8")
print("ok")
