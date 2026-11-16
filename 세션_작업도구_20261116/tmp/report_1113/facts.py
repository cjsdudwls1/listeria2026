# -*- coding: utf-8 -*-
"""보고서에 쓰는 숫자를 원 파일에서만 읽어 facts.json 으로 모음(손으로 옮겨 적지 않기 위함)."""
import csv, json, os
import openpyxl

W = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final_results"
XLSX = os.path.join(BASE, "exports", "단계별_val_test_지표_20261112.xlsx")
F2 = os.path.join(BASE, "final_food2")

wb = openpyxl.load_workbook(XLSX, data_only=True)

def sheet_rows(name):
    ws = wb[name]
    rows = list(ws.iter_rows(values_only=True))
    return rows[0], rows[1:]

facts = {"runs": {}, "summary": {}, "steady": []}

# 전체 판: 판 이름 -> 모든 칸
hdr, rows = sheet_rows("전체 판")
for r in rows:
    if r[1] is None:
        continue
    facts["runs"][r[1]] = {h: v for h, v in zip(hdr, r)}
facts["runs_header"] = list(hdr)

# 단계별 요약: 단계 이름 -> 칸
hdr2, rows2 = sheet_rows("단계별 요약")
for r in rows2:
    if r[0] is None:
        continue
    facts["summary"][r[0]] = {h: v for h, v in zip(hdr2, r)}

# 점진적 상승
hdr3, rows3 = sheet_rows("점진적 상승")
for r in rows3:
    if r[0] is None:
        continue
    facts["steady"].append({h: v for h, v in zip(hdr3, r)})

# 최종 모델 상세
facts["final_scores"] = json.load(open(os.path.join(F2, "report", "final_test_scores.json"), encoding="utf-8"))
facts["final_decision"] = json.load(open(os.path.join(F2, "final_decision.json"), encoding="utf-8"))
with open(os.path.join(F2, "report", "final_h_metrics.csv"), encoding="utf-8-sig") as f:
    facts["final_h"] = [dict(r) for r in csv.DictReader(f)]
with open(os.path.join(F2, "report", "change_table.csv"), encoding="utf-8-sig") as f:
    facts["change_table"] = [dict(r) for r in csv.DictReader(f)]
with open(os.path.join(F2, "report", "research_table.csv"), encoding="utf-8-sig") as f:
    facts["research_table"] = [dict(r) for r in csv.DictReader(f)]
with open(os.path.join(F2, "importance", "importance_summary.csv"), encoding="utf-8-sig") as f:
    facts["importance"] = [dict(r) for r in csv.DictReader(f)]
with open(os.path.join(F2, "report", "input_variables.csv"), encoding="utf-8-sig") as f:
    facts["input_variables"] = [dict(r) for r in csv.DictReader(f)]

json.dump(facts, open(os.path.join(W, "facts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("runs", len(facts["runs"]), "summary", len(facts["summary"]), "imp", len(facts["importance"]))
