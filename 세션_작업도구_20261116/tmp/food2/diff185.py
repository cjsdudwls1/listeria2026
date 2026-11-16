import sys, collections
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from lst import static_feats as sf, samples as S
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data"
M = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\results_mirror"
s = pd.read_parquet(M + r"\s0\0-1\apply\samples.parquet", columns=["p3"])
codes3 = sorted(set(s["p3"]) - {S.NO_CODE})
a = S._to_plain(pq.read_table(D + r"\추출_20261112\t_aw_cl.parquet").to_pandas()).set_index("prdlst_cd")
for c in ("aw", "ph", "na"):
    sub = a.loc[[k for k in codes3 if k in a.index], c]
    nonnull = sub[sub.notna() & (sub.astype(str).str.strip() != "")]
    bad = {k: v for k, v in nonnull.items() if np.isnan(sf.parse_db(v)[0])}
    print(c, "원문 있음", len(nonnull), "해석 못 함", bad, "원문 값 종류(숫자 아닌 것)",
          collections.Counter(v for v in nonnull if not str(v).replace('.', '', 1).isdigit()).most_common(8))
print(collections.Counter(a.loc[[k for k in codes3 if k in a.index], "aw_source"]))
