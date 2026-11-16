# 결과 폴더의 판마다 cv_scores.json·test_scores.json 에서 지표를 모아 CSV 1개로(읽기만).
import json, os, sys, csv
R = sys.argv[1]; OUT = sys.argv[2]
def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d
def f12(p, r):
    if p is None or r is None: return (None, None)
    return ((2*p*r/(p+r)) if p+r else 0.0, (5*p*r/(4*p+r)) if p+r else 0.0)
rows = []
for root, dirs, files in os.walk(R):
    dirs.sort()
    if "cv_scores.json" not in files: continue
    rel = os.path.relpath(root, R)
    c = json.load(open(os.path.join(root, "cv_scores.json")))
    row = {"path": rel, "stage_dir": os.path.dirname(rel), "run": os.path.basename(rel)}
    th = c.get("threshold") or {}
    rr, pp = th.get("recall"), th.get("precision")
    npairs, npos = c.get("n_pairs"), c.get("n_pos")
    tp, fp = th.get("tp"), th.get("fp")
    f1, f2 = f12(pp, rr)
    tq = {}
    for q in ("1", "5", "10", "20"):
        vals = [g(c, "folds", f, "top_q_capture", q) for f in ("fold1", "fold2", "fold3")]
        tq[q] = sum(vals)/3 if all(v is not None for v in vals) else None
    row.update({
        "val_prauc_bw": c.get("cv_band_weighted"),
        "val_fold1": g(c, "fold_band_weighted", "fold1"), "val_fold2": g(c, "fold_band_weighted", "fold2"), "val_fold3": g(c, "fold_band_weighted", "fold3"),
        "val_prauc_h01_20": g(c, "cv_h_bands", "h01_20"), "val_prauc_h21_40": g(c, "cv_h_bands", "h21_40"), "val_prauc_h41_60": g(c, "cv_h_bands", "h41_60"),
        "val_prauc_pooled": c.get("cv_pooled"), "val_lift_bw": c.get("cv_lift_band_weighted"),
        "val_threshold": th.get("threshold"), "val_recall": rr, "val_precision": pp, "val_f1": f1, "val_f2": f2,
        "val_tp": tp, "val_fn": (npos - tp) if (npos is not None and tp is not None) else None, "val_fp": fp,
        "val_tn": (npairs - npos - fp) if (npairs is not None and npos is not None and fp is not None) else None,
        "val_top1": tq["1"], "val_top5": tq["5"], "val_top10": tq["10"], "val_top20": tq["20"],
        "val_n_pairs": npairs, "val_n_pos": npos,
    })
    tpath = os.path.join(root, "test_scores.json")
    t = json.load(open(tpath)).get("report") if os.path.exists(tpath) else None
    if t:
        tt = t.get("threshold_metrics") or {}
        row.update({
            "test_prauc_bw": t.get("pr_auc_band_weighted"),
            "test_prauc_h01_20": t.get("pr_auc_h01_20"), "test_prauc_h21_40": t.get("pr_auc_h21_40"), "test_prauc_h41_60": t.get("pr_auc_h41_60"),
            "test_prauc_pooled": t.get("pr_auc_pooled"), "test_lift_bw": t.get("lift_band_weighted"),
            "test_threshold": tt.get("threshold"), "test_recall": tt.get("recall"), "test_precision": tt.get("precision"),
            "test_f1": tt.get("f1"), "test_f2": tt.get("f2"), "test_tp": tt.get("tp"), "test_fn": tt.get("fn"), "test_fp": tt.get("fp"), "test_tn": tt.get("tn"),
            "test_top1": g(t, "top_q_capture", "1"), "test_top5": g(t, "top_q_capture", "5"), "test_top10": g(t, "top_q_capture", "10"), "test_top20": g(t, "top_q_capture", "20"),
            "test_n_pairs": t.get("n_pairs"), "test_n_pos": t.get("n_pos"),
        })
    rows.append(row)
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(len(rows), "판", sum(1 for r in rows if r.get("test_prauc_bw") is not None), "test 있음")
