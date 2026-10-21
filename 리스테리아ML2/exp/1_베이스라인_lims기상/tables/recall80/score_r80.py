# 1차 실험(10-16) 저장 예측을 새 기준으로 다시 채점
# 기준값 = T 통합 검증 행(T 2024-07-01~2025-05-01, 60개 h 합침)에서 Recall >= 0.80 인 기준값 중 Precision 최대
# 시험 = T 2025-07-01~2026-05-01 행, h 마다 채점 후 60개 평균. 자체 점검: F2 최대 기준값으로 기존 T 통합 표 값 재현.
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

RES = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\1_베이스라인_lims기상\results"
OUT = r"C:\Users\NT751\.claude\jobs\79b84d9c\tmp\r80"
VAL = (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-05-01"))
TST = (pd.Timestamp("2025-07-01"), pd.Timestamp("2026-05-01"))
RUNS = ["catboost__summary__L%03d" % L for L in range(10, 101, 10)] + [
    "catboost__raw__L100", "xgboost__summary__L100", "xgboost__raw__L100",
    "lightgbm__summary__L100", "lightgbm__raw__L100", "lstm__seq__L100"]

def load(run, split, lo_hi):
    d = pd.read_parquet(f"{RES}\\{run}\\pred_{split}.parquet", columns=["d", "h", "y", "p"])
    T = pd.to_datetime(d["d"]) - pd.to_timedelta(d["h"].astype(int), unit="D")
    m = (T >= lo_hi[0]) & (T <= lo_hi[1])
    return d.loc[m, ["h", "y", "p"]].reset_index(drop=True)

def thr_f2(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = 5 * pr * rc / (4 * pr + rc)
    f2 = np.nan_to_num(f2[:-1])
    i = int(np.argmax(f2))
    return float(th[i])

def thr_r80(y, p, target=0.80):
    pr, rc, th = precision_recall_curve(y, p)
    pr, rc = pr[:-1], rc[:-1]
    ok = np.where(rc >= target)[0]
    best = ok[np.argmax(pr[ok])]
    # 같은 Precision 이면 더 높은 기준값(검사 덜 함)
    cand = ok[pr[ok] == pr[best]]
    j = cand.max()
    return float(th[j]), float(pr[j]), float(rc[j])

def score_test(t, thr):
    rows = []
    for h, g in t.groupby("h"):
        y = g["y"].to_numpy(); p = g["p"].to_numpy()
        pred = p >= thr
        TP = int((pred & (y == 1)).sum()); FP = int((pred & (y == 0)).sum())
        FN = int((~pred & (y == 1)).sum()); TN = int((~pred & (y == 0)).sum()); N = len(y)
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
        rows.append(dict(h=h, Accuracy=(TP + TN) / N, Precision=prec, Recall=rec, F1=f1, F2=f2,
                         ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                         TP=TP, FN=FN, FP=FP, TN=TN, N=N, 검사축소율=1 - (TP + FP + FN) / N))
    return pd.DataFrame(rows).drop(columns="h").mean()

res = []
for run in RUNS:
    v = load(run, "val", VAL); t = load(run, "test", TST)
    tf2 = thr_f2(v["y"].to_numpy(), v["p"].to_numpy())
    tr, vp, vr = thr_r80(v["y"].to_numpy(), v["p"].to_numpy())
    s_f2 = score_test(t, tf2); s_r = score_test(t, tr)
    res.append(dict(run=run, 기준값_F2=tf2, **{"F2판_" + k: s_f2[k] for k in ["F2", "Recall", "Precision", "TP"]},
                    기준값_R80=tr, 검증_Precision_at_R80=vp, 검증_Recall_at_R80=vr,
                    검증_PR_AUC_60일평균=float(v.groupby("h").apply(lambda g: average_precision_score(g["y"], g["p"])).mean()),
                    **{k: s_r[k] for k in s_r.index}))
    print(run, "done", flush=True)
df = pd.DataFrame(res)
df.to_csv(OUT + r"\r80_1차.csv", index=False, encoding="utf-8-sig")
with open(OUT + r"\r80_1차.txt", "w", encoding="utf-8") as f:
    pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40)
    f.write(df.round(4).to_string(index=False))
