from pathlib import Path
p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\scripts\s_final.py")
s = p.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''     멈춘 부분이 있으면 실험 로그 'stopped' 뒤 멈춤 — 같은 명령을 다시 하면 남은 h 만.
"""''', '''     멈춘 부분이 있으면 실험 로그 'stopped' 뒤 멈춤 — 같은 명령을 다시 하면 남은 h 만.
 (10) 최종 판 바꿔 끼우기(설정 final_override — 사용자 결정 2026-11-12, 식품 고정표 v2 다시 학습): 사슬 상태의 현재 최고
     판(이름 = final_override.replaces 여야 함 — 아니면 멈춤)을 같은 BestDef 꼴로 이름·폴더만 final_override.name·dir 로
     바꿔 고르기에 넣음(1차 베이스라인과 CV 비교 규칙 그대로). 고른 결과가 바꿔 끼운 판이 아니면 멈춤(지휘자 확인 —
     결정 파일 안 씀). 변화 표 끝 = '최종 모델(식품 갱신 전)'(원 판) → final_override.label(새 판) 두 줄, 리서치 표 머리
     주석에 기준·하나씩 추가 판(2-0·2-R)은 식품 갱신 전 판임을 적음. 설정 context 의 food_table 은 새 고정표여야 함
     (중요도 입력 — 바꿔 끼운 판 spec 의 학습 입력과 같아야 함, run_meta 입력 md5 로 대조).
"""''')
rep('''    "speed": cf.Field(dict, required=False),   # 속도 개선 스위치(lst/speed — 없으면 전부 꺼짐, 결과 불변)
    "context": cf.Field(dict),
}''', '''    "speed": cf.Field(dict, required=False),   # 속도 개선 스위치(lst/speed — 없으면 전부 꺼짐, 결과 불변)
    "final_override": cf.Field(dict, required=False),   # 구현 세부 (10)
    "context": cf.Field(dict),
}
OVERRIDE_SCHEMA = {"name": cf.Field(str), "dir": cf.Field(str), "replaces": cf.Field(str), "label": cf.Field(str),
                   "reason": cf.Field(str)}''')
rep('''    if "importance_parts" in cfg:
        ipt.parse_parts(cfg["importance_parts"])
    return cfg
''', '''    if "importance_parts" in cfg:
        ipt.parse_parts(cfg["importance_parts"])
    if "final_override" in cfg:
        cf.validate(cfg["final_override"], OVERRIDE_SCHEMA, where="final_override.")
        if cfg["final_override"]["name"] == cfg["final_override"]["replaces"]:
            raise cf.ConfigError("final_override.name 은 replaces 와 달라야 함")
    return cfg


def override_best(cfg: Mapping, state: Mapping, inputs_root: Path) -> dict:
    """구현 세부 (10): 사슬 현재 최고 판 BestDef → 이름·폴더만 바꾼 BestDef."""
    import copy

    ov = cfg["final_override"]
    best = state["current_best"]
    if best["name"] != ov["replaces"]:
        raise FinalError(f"사슬 현재 최고 판 {best['name']} ≠ final_override.replaces {ov['replaces']}")
    if len(best["members"]) != 1 or best["members"][0]["name"] != best["name"]:
        raise FinalError("final_override 는 구성 하나(이름 = 판 이름)인 판만")
    d = resolve(inputs_root, ov["dir"])
    if not (d / "done.json").exists():
        raise FinalError(f"final_override.dir 에 done.json 없음: {d}")
    if rn.read_json(d / "spec.json")["name"] != ov["name"]:
        raise FinalError("final_override.dir 의 spec 이름이 final_override.name 과 다름")
    rel = cfg["context"]["results"]["food_table"]
    got = rn.read_json(d / "run_meta.json")["inputs_md5"].get(f"inputs:{rel}")
    if got is None or got != cf.file_md5(resolve(inputs_root, rel)):
        raise FinalError(f"설정 context food_table({rel})이 바꿔 끼운 판 학습 입력과 다름(run_meta 입력 md5)")
    nb = copy.deepcopy(best)
    nb["name"] = ov["name"]
    nb["dir"] = ov["dir"]
    nb["members"][0].update(name=ov["name"], dir=ov["dir"])
    return nb
''')
rep('''    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    best = state["current_best"]
    bdir = resolve(inputs_root, best["dir"])''', '''    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    chain_last = state["current_best"]["name"]
    best = override_best(cfg, state, inputs_root) if cfg.get("final_override") else state["current_best"]
    bdir = resolve(inputs_root, best["dir"])''')
rep('''    final = sel["selected"]
    if final == best["name"]:
        fdef = best
        source = "사슬 마지막 현재 최고 판"''', '''    final = sel["selected"]
    if cfg.get("final_override") and final != best["name"]:
        raise FinalError(f"고르기 결과 {final} 가 바꿔 끼운 판 {best['name']} 이 아님 — 지휘자 확인(결정 파일 안 씀): "
                         f"{sel['cv']}")
    if final == best["name"]:
        fdef = best
        source = ("사슬 마지막 현재 최고 판" if not cfg.get("final_override")
                  else f"사슬 마지막 현재 최고 판({chain_last})을 바꿔 끼운 판 — {cfg['final_override']['reason']}")''')
rep('''    decision = {"stage": STAGE, "final": final, "final_source": source, "final_def": fdef,
                "chain_last": best["name"], "baseline_1": base_name, "selection": sel,''',
    '''    decision = {"stage": STAGE, "final": final, "final_source": source, "final_def": fdef,
                "chain_last": chain_last, "baseline_1": base_name, "selection": sel,
                "final_override": cfg.get("final_override"),''')
rep('''    log.append("decision", stage=STAGE, final=final, source=source, chain_last=best["name"], baseline=base_name)''',
    '''    log.append("decision", stage=STAGE, final=final, source=source, chain_last=chain_last, baseline=base_name)''')
rep('''    fin = decision["final"]
    fdir = (baseline_dir(cfg, inputs_root, fin) if decision["final_source"] == "1차 베이스라인"
            else resolve(inputs_root, decision["final_def"]["dir"]))
    rows.append({"label": "최종 모델", "step": STAGE, "name": fin, "adopted": False, "dir": fdir})
    return rows''', '''    fin = decision["final"]
    fdir = (baseline_dir(cfg, inputs_root, fin) if decision["final_source"] == "1차 베이스라인"
            else resolve(inputs_root, decision["final_def"]["dir"]))
    ov = decision.get("final_override")
    if ov:   # 구현 세부 (10)
        prev = state["current_best"]
        rows.append({"label": "최종 모델(식품 갱신 전)", "step": STAGE, "name": prev["name"], "adopted": False,
                     "dir": resolve(inputs_root, prev["dir"])})
        rows.append({"label": ov["label"], "step": STAGE, "name": fin, "adopted": False, "dir": fdir})
        return rows
    rows.append({"label": "최종 모델", "step": STAGE, "name": fin, "adopted": False, "dir": fdir})
    return rows''')
rep('''    t.attrs["header"] = f"기준 = 파생변수 없는 판(2-0: {b_dir.name}), CV {b_cv:.4f}·test {b_t:.4f}"''',
    '''    t.attrs["header"] = f"기준 = 파생변수 없는 판(2-0: {b_dir.name}), CV {b_cv:.4f}·test {b_t:.4f}"
    if decision.get("final_override"):   # 구현 세부 (10)
        t.attrs["header"] += (f" — 기준 판·하나씩 추가 판(2-0·2-R)은 식품 고정표 갱신 전 판, 최종 모델 안 순열 중요도는 "
                              f"{decision['final']}(식품 고정표 갱신 판)")''')
p.write_text(s, encoding="utf-8")
print("ok")
