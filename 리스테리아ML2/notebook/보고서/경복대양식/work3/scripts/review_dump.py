# -*- coding: utf-8 -*-
"""검수용(읽기 전용): 3차 결과표에서 원고 숫자 대조에 필요한 칸만 짧게 찍음."""
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\3_시간순교차검증\tables"


def r(path):
    return pd.read_csv(T + "\\" + path, encoding="utf-8-sig")


d = r("튜닝후보_cat.csv")
sc = "구간 가중 PR-AUC(세 겹 평균)"
d = d.sort_values(sc, ascending=False)
cols = ["시도", sc, "60일 평균 PR-AUC", "Recall≥0.80 에서의 Precision(60개 h 합친 행)", "기준값 Recall(60개 h 합친 행)",
        "FP(h 당)", "검사축소율", "상위 10% 포착률", "최적 반복(겹별)"]
print("== 튜닝 상위 6 + 16·29")
print(d[cols].head(6).to_string(index=False))
print(d[d["시도"].isin([16, 29])][cols].to_string(index=False))
print("범위 점수", d[sc].min(), d[sc].max(), "FP", d["FP(h 당)"].min(), d["FP(h 당)"].max(), "n", len(d))
print("FP 오름차순 앞 4:")
print(d.sort_values("FP(h 당)")[cols].head(4).to_string(index=False))

v = r("바닐라_비교.csv")
print("\n== 바닐라 칼럼", list(v.columns)[:12])
sc2 = [c for c in v.columns if c.startswith("구간 가중 PR-AUC")][0]
v = v.sort_values(sc2, ascending=False)
keep = [c for c in v.columns if c in ("모델", "알고리즘", "입력", "창(일)", "창", "L")] + [sc2, "FP(h 당)", "TP(h 당)", "FN(h 당)", "검사축소율",
                                                                                "상위 10% 포착률"]
print(v[keep].head(6).to_string(index=False))
print("TP 범위(상위5)", v["TP(h 당)"].head(5).min(), v["TP(h 당)"].head(5).max(), "FN", v["FN(h 당)"].head(5).min(), v["FN(h 당)"].head(5).max())
print("범위 전체", v[sc2].min(), v[sc2].max(), len(v))
grp = [c for c in v.columns if c in ("모델", "알고리즘")][0], [c for c in v.columns if c == "입력"][0]
best = v.groupby(list(grp)).head(1)
print(best[keep].to_string(index=False))
bl = v[(v[grp[0]].astype(str).str.contains("BiLSTM"))]
print(bl[keep].to_string(index=False))

w = r("기상거르기.csv")
print("\n== 기상거르기 칼럼", list(w.columns))
print(w.to_string(index=False, max_colwidth=18))
