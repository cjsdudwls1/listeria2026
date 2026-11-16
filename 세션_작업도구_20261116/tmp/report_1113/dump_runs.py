# -*- coding: utf-8 -*-
import sys, openpyxl
path, out = sys.argv[1], sys.argv[2]
wb = openpyxl.load_workbook(path, data_only=True)
L = []
for name in wb.sheetnames:
    ws = wb[name]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        continue
    hdr = rows[0]
    L.append(f"===== {name} ({len(rows)-1} rows)")
    if "PR-AUC 구간 가중 val" in hdr:
        iv = hdr.index("PR-AUC 구간 가중 val"); it = hdr.index("PR-AUC 구간 가중 test")
        keys = ["Recall test", "Precision test", "F2 test", "threshold val"]
        idx = [hdr.index(k) for k in keys]
        for r in rows[1:]:
            def f(v):
                return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            L.append(" | ".join([f(r[0]), f(r[1]), f(r[2]), "val=" + f(r[iv]), "test=" + f(r[it])] + [k + "=" + f(r[j]) for k, j in zip(keys, idx)]))
    else:
        for r in rows:
            L.append(" | ".join("" if v is None else str(v) for v in r))
open(out, "w", encoding="utf-8").write("\n".join(L))
print("ok")
