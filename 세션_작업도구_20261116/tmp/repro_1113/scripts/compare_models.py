# -*- coding: utf-8 -*-
"""저장된 모델 파일(패키지)과 다시 학습한 모델 파일의 학습 기록(fit_info)·가중치를 비교한다(검증관 도구, 패키지 밖).
사용: python -I compare_models.py <패키지 02_학습모델파일> <다시 학습한 3-0-LSTM 폴더> <출력 csv>
모델 파일은 torch.save(파이썬 기본형 + 텐서) → torch.load(weights_only=True) 로 읽음.
"""
import json
import sys
from pathlib import Path

import torch

saved_root, new_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
record = json.loads((saved_root / "3-0-LSTM_모델파일_학습기록.json").read_text(encoding="utf-8"))["파일"]
SCALARS = ("device", "n_jobs", "n_rows", "n_train", "n_es", "n_pos_train", "n_pos_es", "es_rows_md5",
           "best_epoch", "epochs_run", "final_splits", "best_es_loss")
rows = []


def add(split, item, a, b):
    if isinstance(a, float) and isinstance(b, float):
        diff = abs(a - b)
        same = a == b
    else:
        diff = ""
        same = a == b
    rows.append({"판": split, "항목": item, "저장_모델": a, "다시_학습": b, "같음": same, "차이": diff})


for split in ("fold1", "fold2", "fold3", "refit"):
    a = torch.load(saved_root / "3-0-LSTM" / f"model_{split}", map_location="cpu", weights_only=True)
    b = torch.load(new_dir / f"model_{split}", map_location="cpu", weights_only=True)
    fa, fb = a["fit_info"], b["fit_info"]
    rec = record[f"model_{split}"]["fit_info"]
    # 패키지의 학습기록 json 이 저장 모델 파일 내용과 같은지(문서 대조)
    rows.append({"판": split, "항목": "학습기록.json = 저장 모델 fit_info", "저장_모델": "", "다시_학습": "",
                 "같음": json.dumps(rec, sort_keys=True, default=str) == json.dumps(fa, sort_keys=True, default=str),
                 "차이": ""})
    for k in ("arch", "cat_sizes", "dims", "max_epochs", "batch_size"):
        add(split, k, a.get(k), b.get(k))
    for k in SCALARS:
        add(split, f"fit_info.{k}", fa.get(k), fb.get(k))
    add(split, "fit_info.determinism", json.dumps(fa.get("determinism"), sort_keys=True),
        json.dumps(fb.get("determinism"), sort_keys=True))
    add(split, "fit_info.oom_events", json.dumps(fa.get("oom_events")), json.dumps(fb.get("oom_events")))
    for k in ("train_loss", "es_loss"):
        la, lb = list(fa.get(k) or []), list(fb.get(k) or [])
        add(split, f"fit_info.{k} 에폭 수", len(la), len(lb))
        for i, (x, y) in enumerate(zip(la, lb), 1):
            add(split, f"fit_info.{k}[에폭 {i}]", float(x), float(y))
    sa, sb = a["state_dict"], b["state_dict"]
    add(split, "state_dict 키", sorted(sa), sorted(sb))
    mx, n_equal_t, n_t, n_el, n_el_eq = 0.0, 0, 0, 0, 0
    for k in sa:
        if k not in sb:
            continue
        ta, tb = sa[k].double(), sb[k].double()
        n_t += 1
        n_el += ta.numel()
        if torch.equal(sa[k], sb[k]):
            n_equal_t += 1
        n_el_eq += int((ta == tb).sum())
        if ta.numel():
            mx = max(mx, float((ta - tb).abs().max()))
    rows.append({"판": split, "항목": "가중치(state_dict) 비교", "저장_모델": f"텐서 {n_t}개 · 원소 {n_el:,}",
                 "다시_학습": f"바이트 같은 텐서 {n_equal_t}/{n_t} · 같은 원소 {n_el_eq:,}/{n_el:,}",
                 "같음": n_equal_t == n_t, "차이": mx})

import csv  # noqa: E402

with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["판", "항목", "저장_모델", "다시_학습", "같음", "차이"])
    w.writeheader()
    w.writerows(rows)
for r in rows:
    if r["항목"].startswith("fit_info.train_loss[") or r["항목"].startswith("fit_info.es_loss["):
        continue
    print(f"{r['판']:6s} {r['항목']:36s} 저장={str(r['저장_모델'])[:60]:60s} 새={str(r['다시_학습'])[:60]:60s} 같음={r['같음']} 차이={r['차이']}")
print("에폭별 손실(저장 → 새, 차이):")
for r in rows:
    if r["항목"].startswith("fit_info.train_loss[") or r["항목"].startswith("fit_info.es_loss["):
        print(f"  {r['판']:6s} {r['항목']:28s} {r['저장_모델']:.10f} → {r['다시_학습']:.10f}  차이 {r['차이']:.3e}")
print("결과:", out_csv)
