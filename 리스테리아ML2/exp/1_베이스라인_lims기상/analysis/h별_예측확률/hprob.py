import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr
R = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\1_베이스라인_lims기상\results"
runs = ["catboost__summary__L050","catboost__summary__L020","catboost__summary__L030","catboost__summary__L060","catboost__summary__L100","catboost__raw__L100","lstm__seq__L100","xgboost__summary__L100","lightgbm__summary__L100"]
out = open(r"C:\Users\NT751\.claude\jobs\2af334fa\tmp\hprob\out.txt","w",encoding="utf-8")
curves = {}
for r in runs:
    t = pq.read_table(f"{R}\{r}\pred_test.parquet").to_pandas()
    t["key"] = t["sgg_code"].astype(str)+"|"+t["train_prdlst_code"].astype(str)+"|"+t["train_prdlst_code_2"].astype(str)+"|"+t["d"].astype(str)
    P = t.pivot(index="key", columns="h", values="p")          # rows x 60
    y = t.drop_duplicates("key").set_index("key")["y"].reindex(P.index)
    A = P.to_numpy(dtype=np.float64)
    mean_h = A.mean(axis=0); curves[r] = mean_h
    rng = A.max(1)-A.min(1); rel = rng/A.mean(1)
    rho = spearmanr(A[:,0], A[:,59]).correlation
    pear = np.corrcoef(A[:,0], A[:,59])[0,1]
    # 행 안 변동(h 사이) vs 행 사이 변동
    within = A.std(axis=1).mean(); between = A.mean(axis=1).std()
    # 상위 1% 겹침
    k = max(1,int(len(A)*0.01))
    top1 = set(np.argsort(-A[:,0])[:k]); top60 = set(np.argsort(-A[:,59])[:k])
    pos = y.to_numpy()==1
    out.write(f"=== {r}  (시험 {len(A):,}행, 검출 {pos.sum()})\n")
    out.write(f"  h별 평균 확률: h1 {mean_h[0]:.5f}, h10 {mean_h[9]:.5f}, h20 {mean_h[19]:.5f}, h30 {mean_h[29]:.5f}, h40 {mean_h[39]:.5f}, h50 {mean_h[49]:.5f}, h60 {mean_h[59]:.5f} | 최소 {mean_h.min():.5f}(h{mean_h.argmin()+1}) 최대 {mean_h.max():.5f}(h{mean_h.argmax()+1})\n")
    out.write(f"  검출 행 평균 확률 h1 {A[pos,0].mean():.5f} h60 {A[pos,59].mean():.5f} | 불검출 행 h1 {A[~pos,0].mean():.5f} h60 {A[~pos,59].mean():.5f}\n")
    out.write(f"  행마다 60개 h 확률의 (최대-최소): 중앙값 {np.median(rng):.5f}, 90% {np.quantile(rng,0.9):.5f} | 행 평균 대비 비율 중앙값 {np.median(rel):.3f}, 90% {np.quantile(rel,0.9):.3f}\n")
    out.write(f"  h 사이 변동(행 안 표준편차 평균) {within:.5f} vs 행 사이 변동(행 평균의 표준편차) {between:.5f} → 비 {within/between:.3f}\n")
    out.write(f"  h1 과 h60 확률의 상관: 스피어만 {rho:.3f}, 피어슨 {pear:.3f} | 상위 1%({k}행) 겹침 {len(top1&top60)/k*100:.1f}%\n")
    # 예시: h1 확률 상위 3행의 h별 확률
    if r == "catboost__summary__L050":
        idx = np.argsort(-A.mean(1))[:3]
        for i in idx:
            v = A[i]
            out.write(f"  예시 {P.index[i]} y={int(y.iloc[i])}: h1 {v[0]:.4f} h15 {v[14]:.4f} h30 {v[29]:.4f} h45 {v[44]:.4f} h60 {v[59]:.4f} (최소 {v.min():.4f} 최대 {v.max():.4f})\n")
        mid = np.argsort(A.mean(1))[len(A)//2]
        v = A[mid]; out.write(f"  예시(중간 행) {P.index[mid]}: h1 {v[0]:.4f} h30 {v[29]:.4f} h60 {v[59]:.4f} (최소 {v.min():.4f} 최대 {v.max():.4f})\n")
pd.DataFrame(curves, index=range(1,61)).to_csv(r"C:\Users\NT751\.claude\jobs\2af334fa\tmp\hprob\mean_curves.csv", encoding="utf-8-sig")
out.close()
