# -*- coding: utf-8 -*-
"""검수용 독립 대조(읽기 전용) — build/verify 스크립트와 따로, 원자료에서 표 값을 다시 셈.

- 문서 표를 머리글로 찾아(표 순서에 기대지 않음) 원자료 값과 칸마다 견줌.
- 원자료: 바닐라_비교.csv, 서버 겹별 지표 사본(work/server), selection_wx.json, 튜닝후보_cat.csv, 최종_시험_cat.csv,
  final_cat/metrics_test_by_h.csv, facts_data.md 4절 표(겹 학습 행·검출).
- 반올림: 사사오입(점수 4자리, 건수 1자리, % 는 표마다).
사용: review_check.py 문서.docx
"""
import csv
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

from docx import Document

DOCX = sys.argv[1]
B = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2'
TAB = os.path.join(B, 'exp', '3_시간순교차검증', 'tables')
RES = os.path.join(B, 'exp', '3_시간순교차검증', 'results')
SRV = os.path.join(B, 'notebook', '보고서', '모델성과', 'work', 'server')
FACTS = os.path.join(B, 'notebook', '보고서', '경복대양식', 'work3', 'facts_data.md')
K = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'tp', 'fn', 'fp', 'tn', 'n']
VC = {'accuracy': 'Accuracy(h 평균)', 'precision': 'Precision(h 평균)', 'recall': 'Recall(h 평균)', 'f1': 'F1(h 평균)',
      'roc_auc': 'ROC-AUC(h 평균)', 'pr_auc': '60일 평균 PR-AUC', 'tp': 'TP(h 당)', 'fn': 'FN(h 당)', 'fp': 'FP(h 당)',
      'tn': 'TN(h 당)', 'n': 'N(h 당)'}
WPR = '구간 가중 PR-AUC(세 겹 평균)'


def q(x, nd):
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)


def f4(x):
    return f'{q(x, 4):.4f}'


def f1c(x):
    return f'{q(x, 1):,.1f}'


def fk(k, x):
    return f4(x) if k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc') else f1c(x)


def pct1(x):
    return f'{q(float(x) * 100, 1):.1f}%'


def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


doc = Document(DOCX)
tabs = [[[c.text.replace('\n', ' ').strip() for c in r.cells] for r in t.rows] for t in doc.tables]
err, n_ok = [], 0


def cmp(name, got, want):
    global n_ok
    if got != want:
        err.append((name, got, want))
    else:
        n_ok += 1


def find(first_head, ncol=None, nth=0):
    hits = [t for t in tabs if t[0][0] == first_head and (ncol is None or len(t[0]) == ncol)]
    return hits[nth]


# ---------------- 데이터 기간(facts_data.md 4절 표에서 직접 읽음)
md = open(FACTS, encoding='utf-8').read()
fold = {}
for m in re.finditer(r'^\| (겹 \d|최종·시험) \| ~(\d{4}-\d\d-\d\d) \| \**([\d,]+)\** \| \**([\d,]+)\** \|', md, re.M):
    fold[m.group(1)] = (m.group(2), int(m.group(3).replace(',', '')), int(m.group(4).replace(',', '')))
assert len(fold) == 4, fold
mt = re.search(r'시험 기간 d\(2025-07-01~2026-06-30\) = ([\d,]+)행·검출 행 (\d+)', md)
trow, tpos = int(mt.group(1).replace(',', '')), int(mt.group(2))
dp = find('구간')
for i, k in enumerate(['겹 1', '겹 2', '겹 3']):
    end, n, p = fold[k]
    y = int(end[:4])
    cmp(f'데이터 기간 {k} 기간', dp[i + 1][1], f'train 2015-04-07~{end} → validation {y}-07-01~{y + 1}-06-30')
    cmp(f'데이터 기간 {k} 수', dp[i + 1][2:4], [f'{n:,}/{p:,}', f'{q(p / n * 100, 3):.3f}%'])
cmp('데이터 기간 시험', dp[4][1:4], ['2025-07-01~2026-06-30', f'{trow:,}/{tpos:,}', f'{q(tpos / trow * 100, 3):.3f}%'])
an, ap = fold['최종·시험'][1] + trow, fold['최종·시험'][2] + tpos
cmp('데이터 기간 전체', dp[5][1:4], ['2015-04-07~2026-06-30', f'{an:,}/{ap:,}', f'{q(ap / an * 100, 3):.3f}%'])

# ---------------- 바닐라 70개
van = rows(os.path.join(TAB, '바닐라_비교.csv'))
assert len(van) == 70


def vrow(model, inp, L):
    return next(r for r in van if r['모델'] == model and r['입력'] == inp and int(float(r['기상 창(일)'])) == L)


s1 = find('기상 window')
for i, L in enumerate(range(10, 101, 10)):
    r = vrow('CatBoost', '기상 요약', L)
    cmp(f'1부 {L}일', s1[i + 1][1:], [fk(k, r[VC[k]]) for k in K] + [f4(r[WPR])])


def hband(run):
    per = {}
    for f in (1, 2, 3):
        for r in rows(os.path.join(SRV, run, f'metrics_fold{f}.csv')):
            assert int(r['fold']) == f
            per.setdefault(int(r['h']), {})[f] = r
    assert sorted(per) == list(range(1, 61)) and all(len(v) == 3 for v in per.values())
    out, allm = [], {}
    for k in K:
        allm[k] = sum(sum(float(per[h][f][k]) for f in (1, 2, 3)) / 3 for h in range(1, 61)) / 60
    for a in range(1, 61, 10):
        out.append([f'{a}~{a + 9}일'] + [fk(k, sum(sum(float(per[h][f][k]) for f in (1, 2, 3)) / 3
                                                  for h in range(a, a + 10)) / 10) for k in K])
    return out, allm


hts = [t for t in tabs if t[0][0] == '예측 거리']
assert len(hts) == 6, len(hts)
o, allm = hband('cat__base__L080__s42')
cmp('1부 h 별 80일', hts[0][1:], o)
r80 = vrow('CatBoost', '기상 요약', 80)
for k in K:  # 서버 사본 60개 h 평균 = 결과 표
    cmp(f'서버 사본 대 결과 표 cat80 {k}', abs(allm[k] - float(r80[VC[k]])) < 1e-9, True)

s2 = find('모델', 13)
for i, a in enumerate(['XGBoost', 'CatBoost', 'LightGBM', 'BiLSTM']):
    top = max((r for r in van if r['모델'] == a), key=lambda r: float(r[WPR]))
    L = int(float(top['기상 창(일)']))
    cmp(f'2부 {a} 이름', s2[i + 1][0], f"{a} ({top['입력']} {L}일)")
    cmp(f'2부 {a}', s2[i + 1][1:], [fk(k, top[VC[k]]) for k in K] + [f4(top[WPR])])
    o, allm = hband(top['실행 이름'])
    cmp(f'2부 {a} h 별', hts[1 + i][1:], o)
    for k in K:
        cmp(f'서버 사본 대 결과 표 {a} {k}', abs(allm[k] - float(top[VC[k]])) < 1e-9, True)

# ---------------- 3부
sel = json.load(open(os.path.join(RES, 'selection_wx.json'), encoding='utf-8'))
assert sel['dropped'] == ['co'] and len(sel['wvars']) == 18
fm = sel['final_mean']
t18 = next(r for r in rows(os.path.join(TAB, '튜닝후보_cat.csv')) if r['시도'] == '18')
fin = rows(os.path.join(TAB, '최종_시험_cat.csv'))
summ = json.load(open(os.path.join(RES, 'final_cat', 'summary.json'), encoding='utf-8'))
s3 = find('모델', 14)
want3 = [[fk(k, fm[k]) for k in K] + [f4(fm['wpr_auc']), pct1(fm['inspect_reduction'])]]
for r in (t18, fin[0], fin[1]):
    want3.append([fk(k, r[VC[k]]) for k in K] + [f4(r[WPR]), pct1(r['검사축소율'])])
for i, w in enumerate(want3):
    cmp(f'3부 요약 {i + 1}줄', s3[i + 1][1:], w)
# 최종 교차검증 줄 = summary.json cv_mean 과도 같은지
cv = summ['cv_mean']
cmp('3부 최종 교차검증 = summary cv_mean', s3[3][1:], [fk(k, cv[k]) for k in K] + [f4(cv['wpr_auc']), pct1(cv['inspect_reduction'])])

th = {int(r['h']): r for r in rows(os.path.join(RES, 'final_cat', 'metrics_test_by_h.csv'))}
assert sorted(th) == list(range(1, 61))
cmp('3부 시험 h 별', hts[5][1:], [[f'{a}~{a + 9}일'] + [fk(k, sum(float(th[h][k]) for h in range(a, a + 10)) / 10) for k in K]
                                for a in range(1, 61, 10)])
for k in K:
    cmp(f'시험 h 평균 대 최종 표 {k}', abs(sum(float(th[h][k]) for h in th) / 60 - float(fin[1][VC[k]])) < 1e-9, True)

# 본문 숫자(문단): train 행·검출, threshold, 입력 컬럼 수
txt = '\n'.join(p.text for p in doc.paragraphs)
for s in ('257,042', '973', '0.004895', '239개', '1,005개', '1,001개'):
    cmp(f'본문에 {s}', s in txt, True)
cmp('기준값 = 최종 표', fin[1]['기준값'], '0.004895')
cmp('최종 학습 행', (summ.get('train_rows'), summ.get('train_pos')), (257042, 973))
cmp('최종 입력 칸', (summ['cv_by_fold']['1']['n_num'], summ['cv_by_fold']['1']['n_cat']), (996, 5))

print('대조한 항목', n_ok + len(err), '같음', n_ok, '다름', len(err))
for e in err:
    print('  다름:', e)
