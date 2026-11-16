# -*- coding: utf-8 -*-
"""두 예측 결과를 (lims_data_key, h) 로 맞춰 비교한다(검증관 도구, 패키지 밖).
사용: python -I compare_preds.py <이름> <A 파일> <A 칼럼> <B 파일> <B 칼럼> <출력 csv(한 줄 추가)>
파일은 .csv(utf-8-sig) 또는 .parquet. 칼럼 예: 다시_낸_예측 · 저장된_예측 · pred
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read(p: str, col: str) -> pd.DataFrame:
    p = Path(p)
    # csv 는 float 를 글자로 적은 것 → round_trip 으로 읽어야 바이트 같음 셈이 정확함
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, encoding="utf-8-sig",
                                                                       float_precision="round_trip")
    out = df[["lims_data_key", "h"]].copy()
    out["h"] = out["h"].astype(np.int64)
    out["v"] = df[col].astype(np.float64).to_numpy()
    return out


name, fa, ca, fb, cb, out_csv = sys.argv[1:7]
a, b = read(fa, ca), read(fb, cb)
m = a.merge(b, on=["lims_data_key", "h"], suffixes=("_a", "_b"), validate="one_to_one")
d = (m["v_a"] - m["v_b"]).abs()
rel = d / m["v_b"].abs()
ra, rb = m["v_a"].rank().to_numpy(), m["v_b"].rank().to_numpy()
row = {"비교": name, "A": f"{Path(fa).name}:{ca}", "B": f"{Path(fb).name}:{cb}", "A_행": len(a), "B_행": len(b),
       "맞춘_쌍": len(m), "바이트_같은_쌍": int((d == 0).sum()), "최대_절대_차이": float(d.max()),
       "절대_차이_중앙": float(d.median()), "최대_상대_차이": float(rel.max()), "상대_차이_중앙": float(rel.median()),
       "순위_상관": float(np.corrcoef(ra, rb)[0, 1])}
print(" · ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
outp = Path(out_csv)
pd.DataFrame([row]).to_csv(outp, mode="a", header=not outp.exists(), index=False, encoding="utf-8-sig")
