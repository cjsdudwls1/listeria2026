import json
from pathlib import Path

C = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\configs")


def load(n):
    return json.loads((C / n).read_text(encoding="utf-8"))


def save(n, obj):
    p = C / n
    assert not p.exists(), p
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


FOOD2 = "s0/0-6_static_v2/식품_고정표.parquet"
SPEED = {"dl_keyed_inputs": True, "fast_predict": True, "gpu_importance": True, "gpu_fraction_cap": 0.25}

# 1. 식품 고정표 v2
s = load("s0_6_static.json")
s["inputs"]["aw_cl"] = "리스테리아ML2/data/추출_20261112/t_aw_cl.parquet"
s["results_inputs"]["prev_food_static"] = "s0/0-6_static/식품_고정표.parquet"
s = {"aw_source_filter": "all", **s,
     "missing_check": {"missing_lists": "s0/0-6_missing/missing_lists.json", "y_cut": "2025-06-30"}}
save("s0_6_static_food2.json", s)

# 2. 5-LSTM 다시 학습(식품 v2)
t = load("s5_6_T.json")
t["plan"] = "단계 5·6 5-LSTM 다시 학습(식품 고정표 v2 — 사용자 결정 2026-11-12: aw 출처 거르기 없앰) — §6 단계 5·6, 부록 A-4"
t["stage"] = "5-6-food2"
t["dirs"]["5-6-T"] = "s5_6/5-6-T-food2"
t["gpu_memory_fraction"] = 0.25
t["context"]["results"]["food_table"] = FOOD2
t["speed"] = dict(SPEED)
t["rerun"] = {"of": "5-LSTM", "of_dir": "s5_6/5-6-T/5-LSTM", "name": "5-LSTM-food2", "stage": "5-6-food2",
              "reason": "식품 고정표 v2(t_aw_cl 2026-11-12 추출, aw 출처 거르기 없앰 — 사용자 결정 2026-11-12)",
              "compare": [{"name": "5-LSTM", "dir": "s5_6/5-6-T/5-LSTM", "desc": "식품 갱신 전 같은 판(원 판)"}]}
save("s5_6_T_food2.json", t)

# 3. 5-LSTM-KO(식품 v2 + Koutsoumanis 6칼럼 — 비교만)
k = json.loads(json.dumps(t))
k["plan"] = ("단계 5·6 5-LSTM(식품 v2) + KO1 Koutsoumanis 생육확률 6칼럼 비교 판(사용자 결정 2026-11-12, 계획서 밖 — "
             "식품 연구원 식 그대로)")
k["stage"] = "5-6-food2-KO"
k["dirs"]["5-6-T"] = "s5_6/5-6-T-food2-KO"
k["rerun"] = {"of": "5-LSTM", "of_dir": "s5_6/5-6-T/5-LSTM", "name": "5-LSTM-KO", "stage": "5-6-food2-KO",
              "reason": "5-LSTM(식품 v2)에 Koutsoumanis et al. (2004) 생육확률 6칼럼만 더한 비교 판(사용자 결정 2026-11-12)",
              "add_ids": ["KO1"],
              "compare": [{"name": "5-LSTM", "dir": "s5_6/5-6-T/5-LSTM", "desc": "식품 갱신 전 같은 판(원 판)"},
                          {"name": "5-LSTM-food2", "dir": "s5_6/5-6-T-food2/5-LSTM-food2",
                           "desc": "식품 v2 다시 학습 판(KO 칼럼 없음 — 비교 기준)"}]}
save("s5_6_T_food2_ko.json", k)

# 4. KO 분포
ctx = {"results": {kk: t["context"]["results"][kk] for kk in ("samples", "domestic_weather", "overseas_weather",
                                                               "food_table")},
       "data": {}, "y_cut": t["context"]["y_cut"]}
save("s_ko_dist.json", {"plan": "KO1 Koutsoumanis 생육확률 6칼럼 시료 기준 분포(자기 h·W 60 = 5-LSTM W) — 보고용",
                        "context": ctx, "W": 60, "chunk_rows": 50000})

# 5. 최종(식품 갱신)
f = load("s_final.json")
f["plan"] = ("최종 보고(식품 고정표 v2 다시 학습 판을 최종으로 — 사용자 결정 2026-11-12) — §6 최종 모델 고르기, §1-3, "
             "부록 A-10·A-12·A-15")
f["chain_state"] = "logs/chain_state_8.json"
f["context"]["results"]["food_table"] = FOOD2
f["speed"] = {**SPEED, "cpu_cores": 12}
f["final_override"] = {"name": "5-LSTM-food2", "dir": "s5_6/5-6-T-food2/5-LSTM-food2", "replaces": "5-LSTM",
                       "label": "최종(식품 갱신)",
                       "reason": "식품 고정표 v2 로 같은 설정 다시 학습(사용자 결정 2026-11-12)"}
save("s_final_food2.json", f)
print("ok")
