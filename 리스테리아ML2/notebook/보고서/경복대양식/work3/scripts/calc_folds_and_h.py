# -*- coding: utf-8 -*-
"""II.2·II.7 보조 계산(읽기 전용 원자료 -> work3/calc/ 에만 씀).

1) 겹 구성 수치: 시료 파일(리스테리아ML2/data/데이터셋/시료_시군구.parquet)을 1차 data.build_unit_rows 와 같은 규칙
   (단위 = 시군구 x 품목 2단계 x 3단계, 코드 빈 값 = '코드없음', 기준일 d = mnfctur_de, y = 그 단위·날짜 시료 중 검출 1건 이상)으로
   단위x날짜 행을 만들어 겹별 학습 행·검출, 맞혀 보는 해(07-01~06-30)의 행·검출, 시료 수·검출 시료 수를 센다.
   서버 기록(final_cat/summary.json 의 n_train_rows·n_train_pos, train_rows·train_pos)과 같아야 함 -> 같은지 적음.
2) 최종 모델 h 별 점수: 서버 final_cat 의 pred_cv_fold1~3.parquet(세 겹 맞혀 보는 행 예측)으로 h 마다 PR-AUC(sklearn average_precision),
   그 겹 기준값(summary.json cv_by_fold.threshold)에서 Precision·Recall -> 세 겹 평균. 시험은 metrics_test_by_h.csv 그대로.
   겹별 60일 평균이 summary.json 의 cv_by_fold.pr_auc 와 같은지 적음.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

BASE = 'C:/Users/NT751/Desktop/천영진'
SAMPLE = os.path.join(BASE, '리스테리아ML2', 'data', '데이터셋', '시료_시군구.parquet')
W3 = os.path.join(BASE, '리스테리아ML2', 'notebook', '보고서', '경복대양식', 'work3')
SRV = os.path.join(W3, 'server', 'final_cat')
OUT = os.path.join(W3, 'calc')
os.makedirs(OUT, exist_ok=True)

FOLDS = [(1, '2022-06-30', '2022-07-01', '2023-05-01'), (2, '2023-06-30', '2023-07-01', '2024-05-01'),
         (3, '2024-06-30', '2024-07-01', '2025-05-01')]
FINAL = ('최종', '2025-06-30', '2025-07-01', '2026-05-01')
H_MAX = 60
NULL = '코드없음'

# ---------------------------------------------------------------- 1) 겹 구성
cols = ['sgg_code', 'train_prdlst_code', 'train_prdlst_code_2', 'mnfctur_de', 'judge_result']
s = pq.read_table(SAMPLE, columns=cols).to_pandas()
assert set(s['judge_result'].astype(str).unique()) <= {'검출', '불검출'}
s['p2'] = s['train_prdlst_code'].astype(object).where(s['train_prdlst_code'].notna(), NULL)
s['p3'] = s['train_prdlst_code_2'].astype(object).where(s['train_prdlst_code_2'].notna(), NULL)
s['d'] = pd.to_datetime(s['mnfctur_de']).values.astype('datetime64[D]')
s['pos'] = (s['judge_result'].astype(str) == '검출').astype(np.int8)
u = (s.groupby(['sgg_code', 'p2', 'p3', 'd'], observed=True, sort=False)
     .agg(n=('pos', 'size'), k=('pos', 'sum')).reset_index())
u['y'] = (u['k'] > 0).astype(np.int8)
ud = u['d'].values.astype('datetime64[D]')
sd = s['d'].values.astype('datetime64[D]')

summ = json.load(open(os.path.join(SRV, 'summary.json'), encoding='utf-8'))
res = {'samples': int(len(s)), 'samples_pos': int(s['pos'].sum()), 'unit_days': int(len(u)), 'unit_days_pos': int(u['y'].sum()),
       'd_min': str(sd.min()), 'd_max': str(sd.max()), 'n_sgg': int(s['sgg_code'].nunique()),
       'n_p2': int(s['p2'].nunique()), 'n_p3': int(s['p3'].nunique()), 'n_units': int(u.groupby(['sgg_code', 'p2', 'p3']).ngroups)}

rows = []
for name, te, ts, tend in FOLDS + [FINAL]:
    te_, ts_, tend_ = np.datetime64(te), np.datetime64(ts), np.datetime64(tend)
    m_tr = ud <= te_
    # 맞혀 보는 해 = 학습 끝 다음 날 ~ 1년 (07-01 ~ 06-30)
    ye = np.datetime64(str(int(te[:4]) + 1) + '-06-30')
    m_y = (ud > te_) & (ud <= ye)
    ms_y = (sd > te_) & (sd <= ye)
    # 맞혀 보는 (행, h) 쌍에 한 번이라도 들어가는 행: d - h ∈ [ts, tend] 인 h 가 1..60 중 있음 <=> d ∈ [ts+1, tend+60]
    m_ev = (ud >= ts_ + np.timedelta64(1, 'D')) & (ud <= tend_ + np.timedelta64(H_MAX, 'D'))
    # 쌍 수: 행마다 T = d - h ∈ [ts, tend] 인 h 의 개수
    dd = ud[m_ev]
    lo = np.maximum(1, (dd - tend_).astype(np.int64))
    hi = np.minimum(H_MAX, (dd - ts_).astype(np.int64))
    npair = np.clip(hi - lo + 1, 0, None)
    yy = u['y'].values[m_ev]
    r = {'겹': name, '학습 끝(d)': te, '맞혀 보는 T 시작': ts, '맞혀 보는 T 끝': tend,
         'T 일 수': int((tend_ - ts_).astype(int)) + 1,
         '학습 행(단위x날짜)': int(m_tr.sum()), '학습 행 검출': int(u['y'].values[m_tr].sum()),
         '학습 시료': int((sd <= te_).sum()), '학습 시료 검출': int(s['pos'].values[sd <= te_].sum()),
         '그해 행(d 07-01~06-30)': int(m_y.sum()), '그해 행 검출': int(u['y'].values[m_y].sum()),
         '그해 시료': int(ms_y.sum()), '그해 시료 검출': int(s['pos'].values[ms_y].sum()),
         '맞혀 보는 쌍에 들어간 행': int(m_ev.sum()), '그중 검출 행': int(yy.sum()),
         '(행,h) 쌍': int(npair.sum()), '(행,h) 쌍 검출': int((npair * yy).sum())}
    rows.append(r)
fold_df = pd.DataFrame(rows)

# 서버 기록과 대조
chk = {}
for k in ('1', '2', '3'):
    b = summ['cv_by_fold'][k]
    r = fold_df[fold_df['겹'] == int(k)].iloc[0]
    chk[f'겹{k}'] = {'학습 행': [int(r['학습 행(단위x날짜)']), b['n_train_rows']], '학습 검출': [int(r['학습 행 검출']), b['n_train_pos']],
                    '(행,h) 쌍': [int(r['(행,h) 쌍']), b['n_rows']], '쌍 검출': [int(r['(행,h) 쌍 검출']), b['n_pos']]}
rF = fold_df[fold_df['겹'] == '최종'].iloc[0]
chk['최종'] = {'학습 행': [int(rF['학습 행(단위x날짜)']), summ['train_rows']], '학습 검출': [int(rF['학습 행 검출']), summ['train_pos']],
              '(행,h) 쌍': [int(rF['(행,h) 쌍']), summ['test_pairs']], '쌍 검출': [int(rF['(행,h) 쌍 검출']), summ['test_pos_pairs']]}
res['server_match'] = {k: {kk: (v[0] == v[1]) for kk, v in d.items()} for k, d in chk.items()}
res['server_values'] = chk
fold_df.to_csv(os.path.join(OUT, '겹구성_계산.csv'), index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 2) 최종 모델 h 별(교차검증 세 겹 + 시험)
per = []
fold_check = {}
for k in (1, 2, 3):
    t = pq.read_table(os.path.join(SRV, f'pred_cv_fold{k}.parquet'), columns=['h', 'y', 'p']).to_pandas()
    thr = summ['cv_by_fold'][str(k)]['threshold']
    aps = []
    for hh, g in t.groupby('h', sort=True):
        y, p = g['y'].values, g['p'].values
        ap = average_precision_score(y, p)
        pred = p >= thr
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        per.append({'겹': k, 'h': int(hh), 'n': len(y), 'pos': int(y.sum()), 'pr_auc': ap,
                    'precision': tp / (tp + fp) if tp + fp else np.nan, 'recall': tp / (tp + fn) if tp + fn else np.nan,
                    'inspect_reduction': 1 - (tp + fp + fn) / len(y)})
        aps.append(ap)
    aps = np.array(aps)
    w = 0.5 * aps[0:20].mean() + 0.3 * aps[20:40].mean() + 0.2 * aps[40:60].mean()
    fold_check[k] = {'pr_auc_60': [float(aps.mean()), summ['cv_by_fold'][str(k)]['pr_auc']],
                     'wpr_auc': [float(w), summ['cv_by_fold'][str(k)]['wpr_auc']]}
per_df = pd.DataFrame(per)
cv_h = per_df.groupby('h')[['pr_auc', 'precision', 'recall', 'inspect_reduction']].mean().reset_index()
te_h = pd.read_csv(os.path.join(SRV, 'metrics_test_by_h.csv'), encoding='utf-8-sig')
both = cv_h.merge(te_h[['h', 'pr_auc', 'precision', 'recall', 'inspect_reduction', 'pos', 'n']], on='h', suffixes=('_cv', '_test'))
both.to_csv(os.path.join(OUT, '최종_h별_교차검증_시험.csv'), index=False, encoding='utf-8-sig')
per_df.to_csv(os.path.join(OUT, '최종_h별_겹별.csv'), index=False, encoding='utf-8-sig')


def band(x):
    x = np.asarray(x)
    return {'H1~20': float(x[0:20].mean()), 'H21~40': float(x[20:40].mean()), 'H41~60': float(x[40:60].mean())}


res['fold_check_h'] = {str(k): {kk: [float(v[0]), float(v[1]), abs(v[0] - v[1]) < 1e-9] for kk, v in d.items()} for k, d in fold_check.items()}
res['band_test_pr_auc'] = band(te_h.sort_values('h')['pr_auc'])
res['band_cv_pr_auc'] = band(cv_h.sort_values('h')['pr_auc'])
res['band_test_recall'] = band(te_h.sort_values('h')['recall'])
res['band_cv_recall'] = band(cv_h.sort_values('h')['recall'])
res['test_pr_auc_h1_h60'] = [float(te_h['pr_auc'].iloc[0]), float(te_h['pr_auc'].iloc[-1])]
res['test_pr_auc_max'] = [int(te_h.loc[te_h['pr_auc'].idxmax(), 'h']), float(te_h['pr_auc'].max())]
res['test_pr_auc_min'] = [int(te_h.loc[te_h['pr_auc'].idxmin(), 'h']), float(te_h['pr_auc'].min())]
res['test_recall_range'] = [float(te_h['recall'].min()), float(te_h['recall'].max())]
res['test_precision_range'] = [float(te_h['precision'].min()), float(te_h['precision'].max())]
res['test_pos_per_h_range'] = [int(te_h['pos'].min()), int(te_h['pos'].max())]
json.dump(res, open(os.path.join(OUT, '계산_요약.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print(fold_df.to_string())
