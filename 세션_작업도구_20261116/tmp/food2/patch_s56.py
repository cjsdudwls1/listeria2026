from pathlib import Path
p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\scripts\s5_6.py")
s = p.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:80], s.count(old))
    s = s.replace(old, new)


rep(''' (u2) 쌓기 바탕 = 단계 4 판정 뒤 현재 최고 판이 4-a·4-a0·x2-NN60 중 하나일 때(사슬 상태 stage4.adopted).
"""''', ''' (u2) 쌓기 바탕 = 단계 4 판정 뒤 현재 최고 판이 4-a·4-a0·x2-NN60 중 하나일 때(사슬 상태 stage4.adopted).
 (u3) 다시 학습(T 설정 선택 칸 rerun — 사용자 결정 2026-11-12, 식품 고정표 v2): 같은 사슬 상태·같은 뽑은 키(dirs 의
      sel-*)로 판 설정을 만든 뒤 rerun.of 판 하나만 골라, 서버의 원 판 spec.json(rerun.of_dir)과 칸 하나하나 같은지
      확인(다르면 멈춤 — 같은 설정 보증) → 이름·단계·입력 묶음 id 만 바꿈(rerun.name·rerun.stage·'<이름>-입력',
      extra.rerun 에 원 판·이유 기록). rerun.add_ids 가 있으면 판 설정(version)의 include_ids 뒤에 그 ID 를 더함
      (KO1 추가 판 — registry 구현 세부 (i)). MLflow 단계 run = rerun.stage(새 이름) 아래 새 판 run — 원 판 run 과 안 겹침.
      보고(--report) 비교 줄 = 1차 베이스라인·단계 5·6 시작 때 현재 최고 판 + rerun.compare(원 판·다른 다시 학습 판).
      판정(judge)은 하지 않음(test 는 원 단계 판정 파일이 있을 때만 — dirs[5-6-judge] = 원 판정 폴더).
"""''')
rep('''import argparse
import sys
''', '''import argparse
import dataclasses
import sys
''')
rep('''    "gpu_memory_fraction": cf.Field(float, required=False),
    "device": cf.Field(str, required=False),
}''', '''    "gpu_memory_fraction": cf.Field(float, required=False),
    "device": cf.Field(str, required=False),
    "rerun": cf.Field(dict, required=False),   # 구현 세부 (u3)
}
RERUN_SCHEMA = {"of": cf.Field(str), "of_dir": cf.Field(str), "name": cf.Field(str), "stage": cf.Field(str),
                "reason": cf.Field(str), "add_ids": cf.Field(list, item_types=str, required=False),
                "compare": cf.Field(list, required=False)}
COMPARE_SCHEMA = {"name": cf.Field(str), "dir": cf.Field(str), "desc": cf.Field(str)}''')
rep('''    return {"run": run, "ens": ens, "bests": bests, "state": state}


def check_cfg(cfg) -> None:
    if cfg["stage"] != STAGE or cfg["order"] != s4.ORDER or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("stage '5-6' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
''', '''    b = {"run": run, "ens": ens, "bests": bests, "state": state}
    return apply_rerun(b, cfg["rerun"], inputs_root) if cfg.get("rerun") else b


def check_rerun(rr: dict) -> None:
    cf.validate(rr, RERUN_SCHEMA, where="rerun.")
    for i, c in enumerate(rr.get("compare", [])):
        cf.validate(c, COMPARE_SCHEMA, where=f"rerun.compare[{i}].")
    if rr["name"] == rr["of"]:
        raise cf.ConfigError("rerun.name 은 원 판 이름과 달라야 함(MLflow·폴더 안 겹치게)")
    if rr["stage"] == STAGE or not rr["stage"].startswith(STAGE + "-"):
        raise cf.ConfigError(f"rerun.stage 는 '{STAGE}-<꼬리>'(원 단계 run 과 안 겹치게)")
    from lst import registry as rg

    bad = [i for i in rr.get("add_ids", []) if i not in rg.EXTRA_IDS]
    if bad:
        raise cf.ConfigError(f"rerun.add_ids 는 추가 ID({rg.EXTRA_IDS})만: {bad}")


def apply_rerun(b: dict, rr: Mapping, inputs_root: Path) -> dict:
    """구현 세부 (u3)."""
    if b["ens"]:
        raise s4.StageError("다시 학습은 앙상블 판 없는 단계에서만")
    by = {sp.name: sp for sp in b["run"]}
    if rr["of"] not in by:
        raise s4.StageError(f"rerun.of {rr['of']} 가 이 단계 판에 없음: {sorted(by)}")
    sp = by[rr["of"]]
    old = rn.read_json(resolve(inputs_root, rr["of_dir"]) / "spec.json")
    if old != sp.to_dict():
        diff = sorted(k for k in set(old) | set(sp.to_dict()) if old.get(k) != sp.to_dict().get(k))
        raise s4.StageError(f"다시 만든 판 설정이 원 판 spec.json 과 다름: {diff}")
    extra = dict(sp.extra)
    if rr.get("add_ids"):
        v = dict(extra["version"])
        dup = [i for i in rr["add_ids"] if i in v["include_ids"]]
        if dup:
            raise s4.StageError(f"이미 있는 ID: {dup}")
        v["include_ids"] = list(v["include_ids"]) + list(rr["add_ids"])
        extra["version"] = v
    extra["rerun"] = {"of": rr["of"], "of_dir": rr["of_dir"], "reason": rr["reason"],
                      "add_ids": list(rr.get("add_ids", []))}
    new = dataclasses.replace(sp, name=rr["name"], stage=rr["stage"], input_set=f"{rr['name']}-입력", extra=extra)
    best = dict(b["bests"][rr["of"]])
    best["name"] = rr["name"]
    best["members"] = [{**m, "name": rr["name"]} for m in best["members"]]
    if len(best["members"]) != 1:
        raise s4.StageError("다시 학습 판은 구성 하나")
    out_rel = best["dir"].rsplit("/", 1)[0]
    best = s4.fill_dirs(best, out_rel)
    return {"run": [new], "ens": [], "bests": {rr["name"]: best}, "state": b["state"]}


def check_cfg(cfg) -> None:
    if cfg.get("rerun"):
        check_rerun(cfg["rerun"])
        if cfg["stage"] != cfg["rerun"]["stage"]:
            raise cf.ConfigError("다시 학습 설정: stage = rerun.stage")
        if cfg["order"] != s4.ORDER or cfg["experiment"] != s4.EXPERIMENT:
            raise cf.ConfigError("order '2차' · experiment 'listeria-2차'(부록 A-15)")
    elif cfg["stage"] != STAGE or cfg["order"] != s4.ORDER or cfg["experiment"] != s4.EXPERIMENT:
        raise cf.ConfigError("stage '5-6' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
''')
rep('''        if st["stage4"]["start_best"]["name"] != st["current_best"]["name"]:
            rows.append(s4.best_row(inputs_root, st["stage4"]["start_best"], "단계 4 시작 때 현재 최고 판(따로 판 바탕)"))
''', '''        if st["stage4"]["start_best"]["name"] != st["current_best"]["name"]:
            rows.append(s4.best_row(inputs_root, st["stage4"]["start_best"], "단계 4 시작 때 현재 최고 판(따로 판 바탕)"))
        for c in (cfg.get("rerun") or {}).get("compare", []):   # 구현 세부 (u3)
            rows.append(s4.best_row(inputs_root, {"name": c["name"], "dir": c["dir"]}, c["desc"]))
''')
rep('''    specs_fn = make_specs_fn(inputs_root)
    if ns.variant:''', '''    specs_fn = make_specs_fn(inputs_root)
    notes = NOTES
    if raw.get("rerun"):
        rr = raw["rerun"]
        notes = NOTES + (f"다시 학습 판 {rr['name']} = {rr['of']} 와 같은 설정·같은 뽑은 키·씨앗 0"
                         + (f" + {', '.join(rr['add_ids'])}" if rr.get("add_ids") else "")
                         + f" — {rr['reason']}",)
    if ns.variant:''')
rep('''                        report_rows_fn=report_rows_for(inputs_root), report_notes=NOTES)''',
    '''                        report_rows_fn=report_rows_for(inputs_root), report_notes=notes)''')
p.write_text(s, encoding="utf-8")
print("ok")
