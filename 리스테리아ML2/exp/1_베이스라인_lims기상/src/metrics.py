# -*- coding: utf-8 -*-
"""지표: PR-AUC(average precision), ROC-AUC, F2 최대 기준값, 혼동행렬 지표."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def roc_auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def f2_threshold(y, p) -> dict:
    """val 예측에서 F2 가 최대가 되는 확률값. 판정 = p >= 기준값 (precision_recall_curve 와 같은 규칙).
    같은 F2 가 여러 개면 가장 낮은 기준값(np.argmax 의 첫 번째)."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]             # 마지막 점(precision=1, recall=0)은 기준값이 없음
    with np.errstate(invalid="ignore", divide="ignore"):
        f2 = 5 * prec * rec / (4 * prec + rec)
    f2 = np.nan_to_num(f2, nan=0.0)
    i = int(np.argmax(f2))
    return {"threshold": float(thr[i]), "val_f2_at_threshold": float(f2[i]),
            "val_precision_at_threshold": float(prec[i]), "val_recall_at_threshold": float(rec[i])}


def confusion(y, p, thr) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= thr
    tp = int((pred & y).sum()); fn = int((~pred & y).sum())
    fp = int((pred & ~y).sum()); tn = int((~pred & ~y).sum())
    n = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "n": n, "accuracy": (tp + tn) / n if n else 0.0,
            "precision": precision, "recall": recall, "f1": f1, "f2": f2}
