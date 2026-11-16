import json
P = {"1단계 1차 베이스라인": "s1/1-1/1-CAT-W060-sum", "2단계 3-0 1위 LSTM": "s3/3-0-D/3-0-LSTM", "3단계 최종 5-LSTM": "s5_6/5-6-T/5-LSTM"}
rows = {}
for lab, d in P.items():
    c = json.load(open(d + "/cv_scores.json")); t = json.load(open(d + "/test_scores.json"))["report"]
    th = c["threshold"]; npos, npairs = c["n_pos"], c["n_pairs"]
    R, Pr = th["recall"], th["precision"]
    f1 = 2 * Pr * R / (Pr + R) if Pr + R else 0.0
    f2 = 5 * Pr * R / (4 * Pr + R) if Pr + R else 0.0
    fq = lambda q: sum(c["folds"][f]["top_q_capture"][str(q)] for f in ("fold1", "fold2", "fold3")) / 3
    tt = t["threshold_metrics"]
    rows[lab] = {
        "PR-AUC 구간 가중": (c["cv_band_weighted"], t["pr_auc_band_weighted"]),
        "PR-AUC h 1~20": (c["cv_h_bands"]["h01_20"], t["pr_auc_h01_20"]),
        "PR-AUC h 21~40": (c["cv_h_bands"]["h21_40"], t["pr_auc_h21_40"]),
        "PR-AUC h 41~60": (c["cv_h_bands"]["h41_60"], t["pr_auc_h41_60"]),
        "PR-AUC 60일 통": (c["cv_pooled"], t["pr_auc_pooled"]),
        "배수(구간 가중)": (c["cv_lift_band_weighted"], t["lift_band_weighted"]),
        "Recall": (R, tt["recall"]), "Precision": (Pr, tt["precision"]), "F1": (f1, tt["f1"]), "F2": (f2, tt["f2"]),
        "TP": (th["tp"], tt["tp"]), "FN": (npos - th["tp"], tt["fn"]), "FP": (th["fp"], tt["fp"]), "TN": (npairs - npos - th["fp"], tt["tn"]),
        "상위 1% 포착": (fq(1), t["top_q_capture"]["1"]), "상위 5% 포착": (fq(5), t["top_q_capture"]["5"]),
        "상위 10% 포착": (fq(10), t["top_q_capture"]["10"]), "상위 20% 포착": (fq(20), t["top_q_capture"]["20"]),
        "쌍 수": (npairs, t["n_pairs"]), "검출 쌍": (npos, t["n_pos"]),
    }
labs = list(P)
print("지표|" + "|".join(f"{l} val|{l} test" for l in labs))
for m in rows[labs[0]]:
    vals = []
    for l in labs:
        for v in rows[l][m]:
            vals.append(f"{v:,.0f}" if m in ("TP", "FN", "FP", "TN", "쌍 수", "검출 쌍") else f"{v:.4f}")
    print(m + "|" + "|".join(vals))
