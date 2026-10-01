# -*- coding: utf-8 -*-
"""
성능 향상 후보 피처 빠른 점검 (2026-09-29). MLflow 에 기록하지 않는 탐색용 스크립트.
- 입력: data/listeria_body.parquet, data/collected/probe/morg_listeria_codes.parquet
- 분할: EXP/1/readme.md 와 같음 (mnfctur_de 기준 train ~2024-06-30 / val 2024-07-01~2025-06-30 / test 2025-07-01~)
- 모델: LightGBM 한 가지. 기본값(2_바로학습과 같음)과 정규화+조기종료(val average_precision) 두 설정
- 결과 해석은 docs/04_실험계획/성능향상_방안.md 에 있다.

실행: py 리스테리아ML/scripts/probe_feature_candidates.py   (약 3분, CPU)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

ROOT = r"C:\Users\NT751\Desktop\천영진\리스테리아ML"
BODY = os.path.join(ROOT, "data", "listeria_body.parquet")
MORG = os.path.join(ROOT, "data", "collected", "probe", "morg_listeria_codes.parquet")

# ---------------------------------------------------------------- 데이터
df = pd.read_parquet(BODY)
df["y"] = (df.judge_result.astype(str) == "검출").astype("int8")
df["mnfctur_de"] = df["mnfctur_de"].astype("datetime64[ns]")
df = df.sort_values(["mnfctur_de", "lims_data_key"]).reset_index(drop=True)
D = df.mnfctur_de
BASE = ["src_route", "inspct_instt_code", "prdlst_code", "train_prdlst_code", "train_prdlst_nm",
        "date_basis", "addr_source", "legal_dong_code", "lat", "lon"]          # 2_바로학습 피처 10개
GEO = ["lat", "lon", "legal_dong_code"]                                         # 장소를 그대로 외우는 칼럼
NOGEO = [c for c in BASE if c not in GEO]

# ---------------------------------------------------------------- 이력 피처 (기준일 - LAG 일 이전 행만 집계)
# 검사 결과가 나오기까지 의뢰일 기준 중앙값 9일·95% 20일·99% 37일(m_org inspct_compt_de - reqest_de) → LAG 30일
LAG = 30


def history(keys, name):
    g = df.groupby(keys + ["mnfctur_de"], observed=True)["y"].agg(n="size", k="sum").reset_index()
    for k in keys:
        g[k] = g[k].astype(str)
    g = g.sort_values(keys + ["mnfctur_de"])
    g["cn"] = g.groupby(keys)["n"].cumsum()
    g["ck"] = g.groupby(keys)["k"].cumsum()
    q = df[keys + ["mnfctur_de"]].copy()
    for k in keys:
        q[k] = q[k].astype(str)
    q["qd"] = (q.mnfctur_de - pd.Timedelta(days=LAG)).astype("datetime64[ns]")
    q["_i"] = np.arange(len(q))
    q = q.sort_values("qd")
    gg = g.rename(columns={"mnfctur_de": "qd"})
    gg["qd"] = gg["qd"].astype("datetime64[ns]")
    gg = gg.sort_values("qd")
    m = pd.merge_asof(q, gg[keys + ["qd", "cn", "ck"]], on="qd", by=keys, direction="backward").sort_values("_i")
    n = m["cn"].fillna(0).to_numpy()
    k = m["ck"].fillna(0).to_numpy()
    df[f"h_{name}_n"] = n.astype("float32")          # 과거 검사 건수
    df[f"h_{name}_k"] = k.astype("float32")          # 과거 검출 건수
    df[f"h_{name}_rate"] = np.where(n > 0, k / np.maximum(n, 1), np.nan).astype("float32")   # 과거 검출 비율


history(["addr"], "addr")
history(["prdlst_code"], "prd")
history(["inspct_instt_code"], "instt")
history(["addr", "prdlst_code"], "addrprd")
HIST = [c for c in df.columns if c.startswith("h_")]

# ---------------------------------------------------------------- 제품명 키워드 (도메인: 비가열 RTE·원물·환경검체)
pn = df.prduct_nm.fillna("").str.lower()
KW = {
    "kw_enoki": r"팽이|enoki", "kw_mushroom": r"버섯", "kw_salmon": r"연어|salmon",
    "kw_raw_fish": r"회$|회\(|횟감|필렛|초밥|사시미",
    "kw_raw_meat": r"육회|포장육|원료육|정육|생고기|다짐육|절단|지육|한우|돈육|우육|계육|닭고기|돼지고기|쇠고기|소고기|양고기",
    "kw_frozen": r"냉동", "kw_chilled": r"냉장", "kw_ham_sausage": r"햄|소시지|베이컨|델리", "kw_cheese": r"치즈|cheese",
    "kw_smoked": r"훈제",
    "kw_env": r"바닥|배수구|작업대|장갑|칼날|도마|컨베이어|콘베어|슬라이서|표면|공정|설비|기구|세척수|qc|앞치마|팔레트|트레이|신발|권지",
    "kw_heated": r"가열|살균|멸균|익힌|구이|볶음|튀김|조림|찜", "kw_pre_heat": r"가열전|가열 전|살균전",
    "kw_wash": r"세척전|세척 전|세척후|세척 후", "kw_dairy_past": r"우유|요구르트|요거트|아이스크림|발효유",
    "kw_rte": r"샐러드|샌드위치|김밥|도시락|새싹|나물|무침", "kw_export": r"수출",
}
for k, pat in KW.items():
    df[k] = pn.str.contains(pat, regex=True).astype("int8")
KWC = list(KW) + ["pn_len", "pn_has_num"]
df["pn_len"] = pn.str.len().astype("int16")
df["pn_has_num"] = pn.str.contains(r"\d").astype("int8")
df["month"] = D.dt.month.astype("int8")

# ---------------------------------------------------------------- LIMS 원본 코드 (ba_data.m_org_lims_data_info)
mo = pd.read_parquet(MORG)
df = df.merge(mo, on="lims_data_key", how="left")
LIMS = ["inspct_purps_code", "inspct_knd_code", "reqest_sys_code", "cnvync_mth_code", "splore_sttus_code"]
for c in LIMS + ["jdgmnt_fom_code"]:
    df[c] = df[c].astype("category")

masks = {"train": D <= "2024-06-30", "val": D.between("2024-07-01", "2025-06-30"), "test": D >= "2025-07-01"}
scope = ~(df.kw_env.eq(1) | df.kw_export.eq(1) | df.train_prdlst_nm.astype(str).eq("정액,수정란,종란"))

# ---------------------------------------------------------------- 학습·평가
DEFAULT = dict(random_state=42, verbose=-1)     # 2_바로학습 LightGBM 과 같은 기본값
REG = dict(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.7, reg_lambda=10, cat_smooth=100, min_data_per_group=300, cat_l2=50, max_cat_to_onehot=4,
           random_state=42, verbose=-1)


def run(feats, label, reg=True, spw=1.0):
    Xtr, Xva, Xte = df.loc[masks["train"], feats], df.loc[masks["val"], feats], df.loc[masks["test"], feats]
    ytr, yva, yte = df.y[masks["train"]], df.y[masks["val"]], df.y[masks["test"]]
    t = time.time()
    if reg:
        m = lgb.LGBMClassifier(scale_pos_weight=spw, **REG)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        trees = m.best_iteration_
    else:
        m = lgb.LGBMClassifier(scale_pos_weight=spw, **DEFAULT)
        m.fit(Xtr, ytr)
        trees = m.n_estimators
    pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    imp = pd.Series(m.booster_.feature_importance("gain"), index=feats)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False).head(5).round(1)
    print(f"{label:48s} trees {trees:4d} | val roc {roc_auc_score(yva, pv):.3f} ap {average_precision_score(yva, pv):.4f}"
          f" | test roc {roc_auc_score(yte, pt):.3f} ap {average_precision_score(yte, pt):.4f} | {time.time() - t:.0f}s"
          f" | top {imp.to_dict()}")
    return pt


print("== 결과 나오기까지 일수 (inspct_compt_de - reqest_de)")
print((pd.to_datetime(df.inspct_compt_de) - pd.to_datetime(df.reqest_de)).dt.days.describe(percentiles=[.5, .75, .9, .95, .99]).round(1).to_dict())

print("\n== 단순 점수 (ML 없음) — test")
T, yt = masks["test"], df.y[masks["test"]]
prior = df.y[masks["train"]].mean()
for a in [5, 20, 100]:
    s = (df.loc[T, "h_addr_k"] + a * prior) / (df.loc[T, "h_addr_n"] + a)
    print(f"주소 과거 검출 비율(평활 a={a:3d})  roc {roc_auc_score(yt, s):.3f} ap {average_precision_score(yt, s):.4f}")
seen = set(df.loc[masks["train"] | masks["val"], "addr"])
cold = ~df.loc[T, "addr"].isin(seen)
print(f"test 행 중 train·val 에 없던 주소 {cold.mean() * 100:.1f}% (검출 {int(yt[cold].sum())}/{int(yt.sum())})")

print("\n== 기본값 LightGBM (2_바로학습 설정) — 피처를 더해도 좋아지지 않음")
run(BASE, "A. 기본 10칼럼", reg=False)
run(BASE + HIST, "C. 기본 + 이력", reg=False)
run(BASE + HIST + KWC + ["month"], "E. 기본 + 이력 + 키워드 + 월", reg=False)

print("\n== 정규화 + 조기종료 LightGBM")
run(BASE, "A. 기본 10칼럼")
run(BASE, "A. 기본 10칼럼 + scale_pos_weight=30", spw=30)
run(NOGEO, "A2. 기본 − 좌표·법정동")
run(NOGEO + LIMS, "H. 기본 − 좌표 + LIMS 코드(검사목적 등)")
run(NOGEO + HIST, "C2. 기본 − 좌표 + 이력")
run(NOGEO + HIST + KWC, "E2. 기본 − 좌표 + 이력 + 키워드")
pt = run(NOGEO + HIST + KWC + LIMS, "I. E2 + LIMS 코드")
run(NOGEO + HIST + KWC + LIMS + ["jdgmnt_fom_code"], "I+. I + 판정형식코드(누수 의심)")

print("\n== I 모델 test 운영점 (recall 목표별 precision)")
p, r, th = precision_recall_curve(yt, pt)
for rr in [0.2, 0.3, 0.5]:
    i = np.where(r[:-1] >= rr)[0]
    j = i[np.argmax(p[:-1][i])]
    print(f"recall≥{rr:.0%}: precision {p[j]:.3f}, 임계값 {th[j]:.4f}, 예측양성 {(pt >= th[j]).sum()}건 / 실제 검출 {int(yt.sum())}")

print("\n== 범위 축소 (환경검체·수출·'정액,수정란,종란' 제외)")
print(f"행 {int(scope.sum()):,} 검출 {int(df.y[scope].sum()):,} | test 행 {int((scope & T).sum()):,} 검출 {int(df.y[scope & T].sum())}")
for s in masks:
    masks[s] = masks[s] & scope
run(BASE, "F. 범위축소 기본")
run(NOGEO + HIST + KWC + LIMS, "G. 범위축소 I")
