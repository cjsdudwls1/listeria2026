# -*- coding: utf-8 -*-
"""리스테리아 모델 성과 문서(3차)의 표 수치 계산.

읽는 것(읽기만):
- exp/3_시간순교차검증/tables/바닐라_비교.csv·튜닝후보_cat.csv·최종_시험_cat.csv
- exp/3_시간순교차검증/results/selection_wx.json(1차 베이스라인 18개 기상 final_mean)
- exp/3_시간순교차검증/results/final_cat/metrics_test_by_h.csv(최종 모델 시험 h 별)
- work/server/<실행 이름>/metrics_fold1~3.csv(서버 ~/cyj_listeria/root3/results/<실행 이름>/ 사본)
- 경복대양식/work3/calc/겹구성_계산.csv(겹 학습 행·검출, 시험 해 행·검출)

쓰는 것: work/tables.json(문서 생성기가 읽음), 화면에 대조 결과.

계산 규칙:
- 교차검증 h 별 표 = 겹마다 h 별 지표 → 같은 h 의 세 겹 평균 → h 10일 구간(1~10, …, 51~60) 평균.
- 요약 표 = 결과 표의 'h 평균' 값(세 겹 평균) 그대로. PR-AUC 칸 = 60일 평균 PR-AUC.
- 서버 사본의 h·겹 평균이 결과 표 값과 같은지 대조(차이 1e-9 이하).
- 반올림 = 사사오입(ROUND_HALF_UP). 점수 소수 넷째 자리, 건수 소수 첫째 자리, 검사축소율·검출률은 %.
"""
import csv
import json
import os
from decimal import Decimal, ROUND_HALF_UP

BASE = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2'
EXP = os.path.join(BASE, 'exp', '3_시간순교차검증')
TAB = os.path.join(EXP, 'tables')
RES = os.path.join(EXP, 'results')
WORK = os.path.join(BASE, 'notebook', '보고서', '모델성과', 'work')
SRV = os.path.join(WORK, 'server')
FOLDS_CSV = os.path.join(BASE, 'notebook', '보고서', '경복대양식', 'work3', 'calc', '겹구성_계산.csv')

METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'tp', 'fn', 'fp', 'tn', 'n']
SCORE = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'wpr_auc'}
BANDS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)]

# 결과 표 칸 이름 ↔ 지표 키
VCOL = {
    'accuracy': 'Accuracy(h 평균)', 'precision': 'Precision(h 평균)', 'recall': 'Recall(h 평균)',
    'f1': 'F1(h 평균)', 'roc_auc': 'ROC-AUC(h 평균)', 'pr_auc': '60일 평균 PR-AUC',
    'tp': 'TP(h 당)', 'fn': 'FN(h 당)', 'fp': 'FP(h 당)', 'tn': 'TN(h 당)', 'n': 'N(h 당)',
    'wpr_auc': '구간 가중 PR-AUC(세 겹 평균)', 'inspect_reduction': '검사축소율',
}


def rnd(x, nd):
    q = Decimal(1).scaleb(-nd)
    return Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)


def fmt(key, x):
    if key in SCORE:
        return f'{rnd(x, 4):.4f}'
    if key == 'inspect_reduction':
        return f'{rnd(float(x) * 100, 1):.1f}%'
    # 건수
    v = rnd(x, 1)
    return f'{v:,.1f}'


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def summary_from_row(row):
    out = {}
    for k, c in VCOL.items():
        out[k] = float(row[c])
    return out


def fold_h_means(run):
    """세 겹 h 별 지표 → 같은 h 세 겹 평균 {h: {metric: value}}."""
    per = {}
    for fold in (1, 2, 3):
        for r in read_csv(os.path.join(SRV, run, f'metrics_fold{fold}.csv')):
            h = int(r['h'])
            assert int(r['fold']) == fold
            per.setdefault(h, []).append({m: float(r[m]) for m in METRICS})
    assert sorted(per) == list(range(1, 61)), run
    out = {}
    for h, lst in per.items():
        assert len(lst) == 3, (run, h)
        out[h] = {m: sum(d[m] for d in lst) / 3 for m in METRICS}
    return out


def band_rows(hm):
    rows = []
    for a, b in BANDS:
        vals = {m: sum(hm[h][m] for h in range(a, b + 1)) / (b - a + 1) for m in METRICS}
        rows.append({'label': f'{a}~{b}일', **{m: fmt(m, vals[m]) for m in METRICS}, '_raw': vals})
    return rows


def check_overall(hm, ref, run, tol=1e-9):
    bad = []
    for m in METRICS:
        allmean = sum(hm[h][m] for h in range(1, 61)) / 60
        if abs(allmean - ref[m]) > tol * max(1.0, abs(ref[m])):
            bad.append((m, allmean, ref[m]))
    print(f'  대조 {run}: ' + ('결과 표와 같음(11개 지표)' if not bad else f'다름 {bad}'))
    return not bad


def main():
    out = {'checks': {}}

    # ---------- 데이터 기간 ----------
    fr = {r['겹']: r for r in read_csv(FOLDS_CSV)}
    dp = []
    for k, lab in (('1', '겹 1'), ('2', '겹 2'), ('3', '겹 3')):
        r = fr[k]
        rows_, pos = int(r['학습 행(단위x날짜)']), int(r['학습 행 검출'])
        end = r['학습 끝(d)']
        y0 = int(end[:4])
        dp.append({'구간': lab,
                   '기간': f'학습 2015-04-07~{end} → 맞혀 봄 {y0}-07-01~{y0 + 1}-06-30',
                   '전체/검출': f'{rows_:,}/{pos:,}',
                   '검출률': f'{rnd(pos / rows_ * 100, 3):.3f}%'})
    r = fr['최종']
    trows, tpos = int(r['그해 행(d 07-01~06-30)']), int(r['그해 행 검출'])
    dp.append({'구간': '시험 1년', '기간': '2025-07-01~2026-06-30', '전체/검출': f'{trows:,}/{tpos:,}',
               '검출률': f'{rnd(tpos / trows * 100, 3):.3f}%'})
    ftrain, ftpos = int(r['학습 행(단위x날짜)']), int(r['학습 행 검출'])
    allr, allp = ftrain + trows, ftpos + tpos
    dp.append({'구간': '전체', '기간': '2015-04-07~2026-06-30', '전체/검출': f'{allr:,}/{allp:,}',
               '검출률': f'{rnd(allp / allr * 100, 3):.3f}%', 'bold': True})
    out['data_period'] = dp
    out['final_train'] = {'rows': ftrain, 'pos': ftpos}
    print('데이터 기간:', json.dumps(dp, ensure_ascii=False))
    assert (allr, allp) == (291812, 1141), (allr, allp)

    # ---------- 바닐라 70개 ----------
    van = read_csv(os.path.join(TAB, '바닐라_비교.csv'))
    assert len(van) == 70
    byrun = {r['실행 이름']: r for r in van}

    # 1부: CatBoost 기상 요약 창 10~100
    p1 = []
    for L in range(10, 101, 10):
        run = f'cat__base__L{L:03d}__s42'
        row = byrun[run]
        assert row['모델'] == 'CatBoost' and row['입력'] == '기상 요약' and int(float(row['기상 창(일)'])) == L
        s = summary_from_row(row)
        d = {'label': f'{L}일' + ('(채택)' if L == 80 else ''), 'chosen': L == 80, 'run': run,
             'rank70': int(float(row['순위']))}
        d.update({k: fmt(k, v) for k, v in s.items()})
        p1.append(d)
        hm = fold_h_means(run)
        out['checks'][run] = check_overall(hm, s, run)
    out['part1_summary'] = p1
    best = max(p1, key=lambda d: float(d['wpr_auc']))
    assert best['chosen'], best
    out['part1_h'] = band_rows(fold_h_means('cat__base__L080__s42'))

    # 2부: 알고리즘별 최고(구간 가중 PR-AUC) 입력·창
    algos = ['XGBoost', 'CatBoost', 'LightGBM', 'BiLSTM']
    p2, p2h = [], {}
    for a in algos:
        cand = [r for r in van if r['모델'] == a]
        top = max(cand, key=lambda r: float(r['구간 가중 PR-AUC(세 겹 평균)']))
        L = int(float(top['기상 창(일)']))
        name = f"{a}({top['입력']} {L}일)"
        s = summary_from_row(top)
        d = {'label': name, 'run': top['실행 이름'], 'rank70': int(float(top['순위'])), 'n_cand': len(cand)}
        d.update({k: fmt(k, v) for k, v in s.items()})
        p2.append(d)
        hm = fold_h_means(top['실행 이름'])
        out['checks'][top['실행 이름']] = check_overall(hm, s, top['실행 이름'])
        p2h[name] = band_rows(hm)
        print(f'  2부 {a}: 후보 {len(cand)}개 중 최고 = {top["실행 이름"]} 구간 가중 {d["wpr_auc"]} (70개 중 {d["rank70"]}위)')
    out['part2_summary'] = p2
    out['part2_h'] = p2h

    # ---------- 3부 ----------
    selwx = json.load(open(os.path.join(RES, 'selection_wx.json'), encoding='utf-8'))
    assert selwx['dropped'] == ['co'] and len(selwx['wvars']) == 18 and selwx['L'] == 80 and selwx['model'] == 'cat'
    fm = selwx['final_mean']
    base = {k: fm[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'tp', 'fn', 'fp',
                               'tn', 'n', 'wpr_auc', 'inspect_reduction']}
    tune = {r['시도']: r for r in read_csv(os.path.join(TAB, '튜닝후보_cat.csv'))}
    t18 = summary_from_row(tune['18'])
    fin = read_csv(os.path.join(TAB, '최종_시험_cat.csv'))
    assert fin[0]['구분'].startswith('교차검증') and fin[1]['구분'].startswith('시험')
    fcv, fte = summary_from_row(fin[0]), summary_from_row(fin[1])
    p3 = []
    for lab, s, kind in (
            ('1차 베이스라인', base, '교차검증'),
            ('변수 전부 + 튜닝(18번 후보)', t18, '교차검증'),
            ('최종 모델(WOE 제외)', fcv, '교차검증'),
            ('최종 모델(WOE 제외)', fte, '시험 1년')):
        d = {'label': lab, 'kind': kind}
        d.update({k: fmt(k, v) for k, v in s.items()})
        p3.append(d)
    out['part3_summary'] = p3
    out['final_threshold'] = fin[1]['기준값']
    out['final_rounds'] = fin[1]['반복 수']
    out['fold_thresholds'] = fin[0]['기준값']

    # 최종 모델 시험 h 별(10일 단위)
    th = read_csv(os.path.join(RES, 'final_cat', 'metrics_test_by_h.csv'))
    assert [int(r['h']) for r in th] == list(range(1, 61))
    hm = {int(r['h']): {m: float(r[m]) for m in METRICS} for r in th}
    ok = check_overall(hm, fte, 'final_cat 시험 h 별')
    out['checks']['final_test_by_h'] = ok
    out['part3_h_test'] = band_rows(hm)

    for k in ('part1_h', 'part3_h_test'):
        for r in out[k]:
            r.pop('_raw', None)
    for k, rows in out['part2_h'].items():
        for r in rows:
            r.pop('_raw', None)

    with open(os.path.join(WORK, 'tables.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('대조 모두 같음:', all(out['checks'].values()))
    print('1부 요약:')
    for d in p1:
        print('  ', d['label'], d['accuracy'], d['precision'], d['recall'], d['f1'], d['roc_auc'], d['pr_auc'],
              d['tp'], d['fn'], d['fp'], d['tn'], d['n'], d['wpr_auc'], d['inspect_reduction'])
    print('1부 80일 h 10일 단위:')
    for d in out['part1_h']:
        print('  ', d)
    print('2부 요약:')
    for d in p2:
        print('  ', d['label'], d['accuracy'], d['precision'], d['recall'], d['f1'], d['roc_auc'], d['pr_auc'],
              d['tp'], d['fn'], d['fp'], d['tn'], d['n'], d['wpr_auc'])
    print('3부 요약:')
    for d in p3:
        print('  ', d['label'], d['kind'], d['accuracy'], d['precision'], d['recall'], d['f1'], d['roc_auc'],
              d['pr_auc'], d['tp'], d['fn'], d['fp'], d['tn'], d['n'], d['wpr_auc'], d['inspect_reduction'])
    print('3부 시험 h 10일 단위:')
    for d in out['part3_h_test']:
        print('  ', d)
    print('기준값', out['final_threshold'], '반복', out['final_rounds'], '겹 기준값', out['fold_thresholds'])


if __name__ == '__main__':
    main()
