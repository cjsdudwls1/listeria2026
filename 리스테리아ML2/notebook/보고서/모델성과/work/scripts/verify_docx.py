# -*- coding: utf-8 -*-
"""완성 docx 의 표 숫자를 원자료에서 따로 다시 계산해 칸마다 대조(읽기 전용).

- tables.json 을 거치지 않고 결과 표 CSV·selection_wx.json·서버 겹별 지표 사본·최종 시험 h 별 CSV·겹구성_계산.csv 에서 바로 셈.
- 쓰면 안 되는 말('묶음'·'섞기') 이 본문에 없는지도 봄.
사용: verify_docx.py 문서.docx
"""
import csv
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

from docx import Document

DOCX = sys.argv[1]
BASE = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2'
TAB = os.path.join(BASE, 'exp', '3_시간순교차검증', 'tables')
RES = os.path.join(BASE, 'exp', '3_시간순교차검증', 'results')
SRV = os.path.join(BASE, 'notebook', '보고서', '모델성과', 'work', 'server')
FOLDS = os.path.join(BASE, 'notebook', '보고서', '경복대양식', 'work3', 'calc', '겹구성_계산.csv')
K = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'tp', 'fn', 'fp', 'tn', 'n']


def r4(x):
    return str(Decimal(repr(float(x))).quantize(Decimal('0.0001'), ROUND_HALF_UP))


def r1c(x):
    return f"{Decimal(repr(float(x))).quantize(Decimal('0.1'), ROUND_HALF_UP):,}"


def fmt(k, x):
    return r4(x) if k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'wpr') else r1c(x)


def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


COLS = {'accuracy': 'Accuracy(h 평균)', 'precision': 'Precision(h 평균)', 'recall': 'Recall(h 평균)', 'f1': 'F1(h 평균)',
        'roc_auc': 'ROC-AUC(h 평균)', 'pr_auc': '60일 평균 PR-AUC', 'tp': 'TP(h 당)', 'fn': 'FN(h 당)', 'fp': 'FP(h 당)',
        'tn': 'TN(h 당)', 'n': 'N(h 당)', 'wpr': '구간 가중 PR-AUC(세 겹 평균)'}

doc = Document(DOCX)
tabs = [[[c.text for c in r.cells] for r in t.rows] for t in doc.tables]
print('표 수', len(tabs))
assert len(tabs) == 11
errors = []


def cmp(name, got, want):
    if got != want:
        errors.append((name, got, want))


# ---- 데이터 기간
fr = {r['겹']: r for r in rows(FOLDS)}
want = []
for k in ('1', '2', '3'):
    r = fr[k]
    n, p = int(r['학습 행(단위x날짜)']), int(r['학습 행 검출'])
    want.append([f'{n:,}/{p:,}', f"{Decimal(p / n * 100).quantize(Decimal('0.001'), ROUND_HALF_UP)}%"])
r = fr['최종']
n, p = int(r['그해 행(d 07-01~06-30)']), int(r['그해 행 검출'])
want.append([f'{n:,}/{p:,}', f"{Decimal(p / n * 100).quantize(Decimal('0.001'), ROUND_HALF_UP)}%"])
n2, p2 = n + int(r['학습 행(단위x날짜)']), p + int(r['학습 행 검출'])
want.append([f'{n2:,}/{p2:,}', f"{Decimal(p2 / n2 * 100).quantize(Decimal('0.001'), ROUND_HALF_UP)}%"])
for i, w in enumerate(want):
    cmp(f'데이터 기간 {i}', tabs[0][i + 1][2:4], w)
cmp('데이터 기간 fold 1 기간', tabs[0][1][1], 'train 2015-04-07~2022-06-30 → validation 2022-07-01~2023-06-30')
cmp('데이터 기간 구간 이름', [r[0] for r in tabs[0][1:]], ['fold 1', 'fold 2', 'fold 3', 'test 1년', '전체'])

# ---- 1부 요약(CatBoost 기상 요약 창 10~100)
van = rows(os.path.join(TAB, '바닐라_비교.csv'))
for i, L in enumerate(range(10, 101, 10)):
    row = next(x for x in van if x['모델'] == 'CatBoost' and x['입력'] == '기상 요약' and int(float(x['기상 창(일)'])) == L)
    want = [fmt(k, row[COLS[k]]) for k in K] + [fmt('wpr', row[COLS['wpr']])]
    got = tabs[1][i + 1]
    cmp(f'1부 요약 {L}일 이름', got[0], f'{L}일' + ('(채택)' if L == 80 else ''))
    cmp(f'1부 요약 {L}일', got[1:], want)
best_L = max(range(10, 101, 10), key=lambda L: float(next(
    x for x in van if x['모델'] == 'CatBoost' and x['입력'] == '기상 요약' and int(float(x['기상 창(일)'])) == L)[COLS['wpr']]))
cmp('1부 채택 창', best_L, 80)


# ---- h 별(서버 겹별 지표 → 세 겹 평균 → 10일 구간 평균)
def hband(run):
    per = {}
    for f in (1, 2, 3):
        for r in rows(os.path.join(SRV, run, f'metrics_fold{f}.csv')):
            per.setdefault(int(r['h']), []).append(r)
    out = []
    for a in range(1, 61, 10):
        vals = []
        for k in K:
            v = sum(sum(float(r[k]) for r in per[h]) / 3 for h in range(a, a + 10)) / 10
            vals.append(fmt(k, v))
        out.append([f'{a}~{a + 9}일'] + vals)
    return out


cmp('1부 h 별(80일)', tabs[2][1:], hband('cat__base__L080__s42'))

# ---- 2부
algos = ['XGBoost', 'CatBoost', 'LightGBM', 'BiLSTM']
for i, a in enumerate(algos):
    cand = [x for x in van if x['모델'] == a]
    top = max(cand, key=lambda x: float(x[COLS['wpr']]))
    want = [fmt(k, top[COLS[k]]) for k in K] + [fmt('wpr', top[COLS['wpr']])]
    got = tabs[3][i + 1]
    cmp(f'2부 요약 {a} 이름', got[0].replace('\n', ' '), f"{a} ({top['입력']} {int(float(top['기상 창(일)']))}일)")
    cmp(f'2부 요약 {a}', got[1:], want)
    cmp(f'2부 h 별 {a}', tabs[4 + i][1:], hband(top['실행 이름']))
    print(f'  2부 {a}: {top["실행 이름"]}')

# ---- 3부 요약
sel = json.load(open(os.path.join(RES, 'selection_wx.json'), encoding='utf-8'))['final_mean']
t18 = next(x for x in rows(os.path.join(TAB, '튜닝후보_cat.csv')) if x['시도'] == '18')
fin = rows(os.path.join(TAB, '최종_시험_cat.csv'))
srcs = [
    ([fmt(k, sel[k]) for k in K] + [r4(sel['wpr_auc']),
                                    f"{Decimal(repr(sel['inspect_reduction'] * 100)).quantize(Decimal('0.1'), ROUND_HALF_UP)}%"]),
]
for row in (t18, fin[0], fin[1]):
    srcs.append([fmt(k, row[COLS[k]]) for k in K] + [fmt('wpr', row[COLS['wpr']]),
                 f"{Decimal(repr(float(row['검사축소율']) * 100)).quantize(Decimal('0.1'), ROUND_HALF_UP)}%"])
for i, w in enumerate(srcs):
    cmp(f'3부 요약 {i}', tabs[9][i + 1][1:], w)

# ---- 3부 시험 h 별
th = {int(r['h']): r for r in rows(os.path.join(RES, 'final_cat', 'metrics_test_by_h.csv'))}
want = []
for a in range(1, 61, 10):
    want.append([f'{a}~{a + 9}일'] + [fmt(k, sum(float(th[h][k]) for h in range(a, a + 10)) / 10) for k in K])
cmp('3부 시험 h 별', tabs[10][1:], want)

# ---- 변수 표 칸 수 합
var = tabs[8]
num = 0
cat = 0
for r in var[1:-1]:
    c = r[3]
    if c.startswith('범주'):
        cat += int(c.split()[1])
    elif '범주' in c:
        a, b = c.split('+')
        num += int(a)
        cat += int(b.split()[1])
    elif c.strip().isdigit():
        num += int(c)
cmp('변수 표 숫자 칸 합', num, 1000)
cmp('변수 표 범주 칸 합', cat, 5)

# ---- 금지 말
alltext = '\n'.join(p.text for p in doc.paragraphs) + '\n' + '\n'.join('\n'.join('\t'.join(r) for r in t) for t in tabs)
for bad in ('묶음', '섞기'):
    if bad in alltext:
        errors.append(('금지 말', bad, '없어야 함'))
# 2026-10-22 용어 규칙(억지 번역 대신 train·test·validation·fold·컬럼·window·트리·하이퍼파라미터·threshold, 제외·추가)
import re  # noqa: E402
OLD_TERMS = ['겹', '칸', '칼럼', '나무', '갈림', '설정값', '기준값', '시험', '맞혀 보', '맞혀 본', '맞혀 봄',
             '학습 자료', '학습 데이터', '학습 기간', '학습 끝', '학습 행', '뺌', '뺀', '빼 ', '빼고', '빼면', '빼도', '빼니',
             '빼기', '빠짐', '빠진', '더함', '만드는 법', '로그 눈금', '순서열', '씨앗', '갈래', '섞음']
for bad in OLD_TERMS:
    if bad in alltext:
        errors.append(('옛 용어', bad, '없어야 함'))
for m in re.finditer(r'창(?![원녕출])', alltext):
    errors.append(('옛 용어', alltext[max(0, m.start() - 8):m.end() + 8], '창 → window'))
for m in re.finditer(r'\((?:Ⅰ부 |Ⅱ부 )?(?:[1-9](?:\.\d{1,2}){1,2}|\d{1,2}장)\)|\(\d부[) ]|, \d부\)', alltext):
    errors.append(('절·장 번호 참조', m.group(0), '없어야 함'))

print('대조 결과:', '모두 같음' if not errors else f'다름 {len(errors)}건')
for e in errors:
    print('  ', e)
