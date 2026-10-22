# -*- coding: utf-8 -*-
"""리스테리아 3차 보고서 II.2~II.8 수치 모음 -> work3/facts_results.md (원고 아님, 표에 넣을 수치·출처만).

읽는 것(모두 읽기 전용):
  exp/3_시간순교차검증/tables/*.csv, results/selection_*.json·params_cat.json·post_cat/summary.json
  work3/server/(서버 사본: final_cat/summary.json·config.json, baseline_wx18/summary.json)
  work3/calc/(calc_folds_and_h.py 결과)
  data/데이터셋/시료_시군구.parquet(품목 이름), data/데이터셋/제품명/뭉치_정의.csv, data/중간/시군구_기상_점검.csv
"""
import json
import os
import re

import pandas as pd
import pyarrow.parquet as pq

BASE = 'C:/Users/NT751/Desktop/천영진/리스테리아ML2'
EXP = os.path.join(BASE, 'exp', '3_시간순교차검증')
TAB = os.path.join(EXP, 'tables')
RES = os.path.join(EXP, 'results')
W3 = os.path.join(BASE, 'notebook', '보고서', '경복대양식', 'work3')
SRV = os.path.join(W3, 'server')
CALC = os.path.join(W3, 'calc')
SCRIPTS = 'C:\\Users\\NT751\\Desktop\\천영진\\리스테리아ML2\\notebook\\보고서\\경복대양식\\work3\\scripts'
FIGS = 'C:\\Users\\NT751\\Desktop\\천영진\\리스테리아ML2\\notebook\\보고서\\경복대양식\\figs3'
SEL = '구간 가중 PR-AUC(세 겹 평균)'


def tab(n):
    return pd.read_csv(os.path.join(TAB, n), encoding='utf-8-sig')


def jl(p):
    return json.load(open(p, encoding='utf-8'))


def f4(x):
    return '' if pd.isna(x) else f'{float(x):.4f}'


def pc(x, d=1):
    return '' if pd.isna(x) else f'{float(x) * 100:.{d}f}%'


def c1(x):
    return '' if pd.isna(x) else f'{float(x):,.1f}'


def c0(x):
    return '' if pd.isna(x) else f'{int(round(float(x))):,}'


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(v) for v in r) + ' |')
    return '\n'.join(out)


L = []


def w(s=''):
    L.append(s)


# ---------------------------------------------------------------- 자료 읽기
vb = tab('바닐라_비교.csv')
wxh = tab('기상거르기.csv')
wxi = tab('기상거르기_변수중요도.csv')
ks = tab('k선택.csv')
tc = tab('튜닝후보_cat.csv')
dr = tab('묶음빼기_cat.csv')
ft = tab('최종_시험_cat.csv')
it = tab('고위험품목_비교_cat.csv')
ae1 = tab('이상치단독.csv')
ae2 = tab('이상치넣기_비교_cat.csv')
dl = tab('딥러닝단계.csv')
summ = jl(os.path.join(SRV, 'final_cat', 'summary.json'))
cfg = jl(os.path.join(SRV, 'final_cat', 'config.json'))
b18 = jl(os.path.join(SRV, 'baseline_wx18', 'summary.json'))
selb = jl(os.path.join(RES, 'selection_baseline.json'))
selwx = jl(os.path.join(RES, 'selection_wx.json'))
selk = jl(os.path.join(RES, 'selection_k.json'))
seld = jl(os.path.join(RES, 'selection_drop_cat.json'))
seli = jl(os.path.join(RES, 'selection_items.json'))
pcat = jl(os.path.join(RES, 'params_cat.json'))
post = jl(os.path.join(RES, 'post_cat', 'summary.json'))
fold = pd.read_csv(os.path.join(CALC, '겹구성_계산.csv'), encoding='utf-8-sig')
calc = jl(os.path.join(CALC, '계산_요약.json'))
wn = pd.read_csv(os.path.join(BASE, 'data', '중간', '시군구_기상_점검.csv'), encoding='utf-8-sig')
WNAME = dict(zip(wn['칼럼'], wn['변수']))
nbdef = pd.read_csv(os.path.join(BASE, 'data', '데이터셋', '제품명', '뭉치_정의.csv'), encoding='utf-8-sig')
pn = pq.read_table(os.path.join(BASE, 'data', '데이터셋', '시료_시군구.parquet'),
                   columns=['train_prdlst_code', 'train_prdlst_nm', 'train_prdlst_code_2', 'train_prdlst_nm_2']).to_pandas()
PNAME = (pn.dropna(subset=['train_prdlst_code_2']).drop_duplicates('train_prdlst_code_2')
         .set_index('train_prdlst_code_2')[['train_prdlst_nm', 'train_prdlst_nm_2']].to_dict('index'))
dup_names = pn.dropna(subset=['train_prdlst_code_2']).astype(str).groupby('train_prdlst_code_2')['train_prdlst_nm_2'].nunique()
assert int(dup_names.max()) == 1, '3단계 코드 하나에 이름이 여러 개'


def wname(v):
    return f"{WNAME.get(v, v)}({v})"


def mrow(r, extra=()):
    """표 공통 지표 칸."""
    return [f4(r[SEL]), f4(r['60일 평균 PR-AUC']), f4(r['Recall≥0.80 에서의 Precision(60개 h 합친 행)']), c1(r['TP(h 당)']),
            c1(r['FN(h 당)']), c1(r['FP(h 당)']), pc(r['검사축소율']), pc(r['상위 10% 포착률'])] + list(extra)


MH = ['구간 가중 PR-AUC', '60일 평균 PR-AUC', 'Recall 0.8 에서 Precision', 'TP(h당)', 'FN(h당)', 'FP(h당)', '검사축소율', '상위 10% 포착률']

# ---------------------------------------------------------------- 머리말
w('# 리스테리아 3차 연구 보고서 II.2~II.8 수치 모음 (facts_results.md)')
w()
w('- 원고가 아니다. 절별 표·문장에 넣을 수치와 그 출처만 모았다. 이 파일은 `work3\\scripts\\build_facts.py` 가 만든다(다시 돌리면 같은 내용).')
w('- 출처 표기: **[표: 이름]** = `리스테리아ML2\\exp\\3_시간순교차검증\\tables\\이름.csv`(utf-8-sig), **[결과: 이름]** = `exp\\3_시간순교차검증\\results\\이름`, '
  '**[서버: 경로]** = 팀 서버 `~/cyj_listeria/root3/경로` 를 `work3\\server\\` 로 복사한 사본(계정 이름은 `<user>` 로 가림), '
  '**[계산: 이름]** = 이 작업에서 저장된 자료로 다시 센 값(`work3\\calc\\이름`, 스크립트 `work3\\scripts\\calc_folds_and_h.py`) — 서버 기록과 맞는지 대조함, '
  '**[코드: 파일]** = 실험 코드(읽기만 함), **[사양]** = `exp\\3_시간순교차검증\\사양_20261020.md`.')
w('- 숫자 규칙: 점수(PR-AUC·Precision 등)는 소수 넷째 자리, 비율은 %(소수 첫째 자리), 건수(TP·FN·FP·TN·N)는 h 하나당 평균(소수 첫째 자리). '
  '교차검증 점수는 세 겹 평균. 점수만 적는다(표준편차·신뢰구간 없음).')
w('- 1·2차 실험 수치는 넣지 않았다(3차 결과만).')
w()

# ---------------------------------------------------------------- 0. 흐름 요약
w('## 0. 3차 흐름 한눈에')
w()
r_van = vb.iloc[0]
d_wo = dr[dr['결정'] == '뺌'].iloc[0]
rows0 = [
    ['① 바닐라 70개', 'XGBoost·LightGBM·CatBoost × 기상 창 10~100일 × (기상 요약·일별 원값) 60개 + BiLSTM × 창 10~100일 10개, 기본 설정 + 조기 종료',
     f"1위 CatBoost·기상 요약·{int(r_van['기상 창(일)'])}일 {f4(r_van[SEL])}", '[표: 바닐라_비교], [결과: selection_baseline.json]'],
    ['② 기상 변수 거르기', '1위 설정으로 기상 19변수를 섞기 중요도 낮은 것부터 하나씩 빼 봄', f"일산화탄소(co)만 뺌 {f4(selwx['start_score'])} → {f4(selwx['final_score'])} = 1차 베이스라인 (18변수)",
     '[표: 기상거르기], [결과: selection_wx.json]'],
    ['③ 스무딩 k 고르기', '1차 베이스라인 + 과거 이력 핵심 칸, k 후보 10·30·50·100·200', f"k = {selk['k']} ({f4(ks.loc[ks['k'] == selk['k'], SEL].iloc[0])})", '[표: k선택], [결과: selection_k.json]'],
    ['④ 튜닝', 'CatBoost Optuna(TPE) 30회, 변수 전부(16묶음)', f"사용자가 시도 18 고름 ({f4(tc.loc[tc['시도'] == 18, SEL].iloc[0])})", '[표: 튜닝후보_cat], [결과: params_cat.json]'],
    ['⑤ 변수 빼기', '22단위 묶음 섞기 중요도 낮은 것부터 하나씩 빼 봄', f"WOE 만 뺌 {f4(d_wo['기준 구간 가중 PR-AUC'])} → {f4(d_wo['후보 구간 가중 PR-AUC'])} (21단위)",
     '[표: 묶음빼기_cat], [결과: selection_drop_cat.json]'],
    ['⑥ 최종 재학습·시험', f"학습 자료 전체(d ≤ 2025-06-30), 반복 {summ['rounds']}, 기준값 {summ['threshold']['threshold']:.6f}",
     f"시험 구간 가중 PR-AUC {f4(summ['test']['wpr_auc'])} (교차검증 {f4(summ['cv_mean']['wpr_auc'])})", '[표: 최종_시험_cat], [서버: results/final_cat/summary.json]'],
    ['⑦ 추가 실험', '고위험 5품목 전용 모델, 이상치 탐지(AE·VAE) 단독·넣기, 딥러닝 8개', '8절', '[표: 고위험품목_비교_cat·이상치단독·이상치넣기_비교_cat·딥러닝단계]'],
]
w(md_table(['단계', '한 것', '결과', '출처'], rows0))
w()

# ---------------------------------------------------------------- II.2
w('## II.2 실험 설계')
w()
w('### 2-1. 날짜 말 [사양 0절]')
w('- T = 예측하는 날, h = 며칠 뒤를 맞히는지(1~60), d = 맞히려는 날(시료 기준일 `mnfctur_de`) = T + h. 예: T = 2025-07-01, h = 60 → d = 2025-08-30.')
w('- 맞히는 단위(행) = 시군구 × 품목 2단계 × 품목 3단계 × 기준일 d. 그 단위·날짜 시료 중 검출이 1건 이상이면 정답 1 [코드: 1차 data.py build_unit_rows].')
w()
w('### 2-2. 자료 규모 [계산: 계산_요약.json, 서버 로그 progress3.log 의 "데이터 준비" 줄과 같음]')
rows = [['시료(시료_시군구.parquet, src_route 전부 domestic — check_nb_counts_src_route.py)', f"{calc['samples']:,}", f"{calc['samples_pos']:,}"],
        ['단위×날짜 행', f"{calc['unit_days']:,}", f"{calc['unit_days_pos']:,}"]]
w(md_table(['구분', '행 수', '검출'], rows))
w(f"- 기준일 범위 {calc['d_min']} ~ {calc['d_max']}(자료 끝 = 시료 파일 mnfctur_de 최댓값). 시군구 {calc['n_sgg']}개, 품목 2단계 {calc['n_p2']}개·3단계 {calc['n_p3']}개('코드없음' 한 값 포함), 단위(시군구×2단계×3단계) {calc['n_units']:,}개.")
w('- 기상표 = 시군구 256개 × 4,573일 × 19변수(2014-04-07 ~ 2026-10-13) [서버: logs/progress3.log "기상 W (256, 4573, 19)"].')
w()
w('### 2-3. 겹 구성 [계산: 겹구성_계산.csv — 학습 행·검출, (행,h) 쌍·검출 모두 서버 summary.json 과 같음]')
rows = []
for _, r in fold.iterrows():
    nm = '최종 재학습 + 시험' if str(r['겹']) == '최종' else f"겹 {r['겹']}"
    rows.append([nm, f"2015-04-07 ~ {r['학습 끝(d)']}", f"{int(r['학습 행(단위x날짜)']):,}", f"{int(r['학습 행 검출']):,}",
                 f"{int(r['학습 시료']):,} ({int(r['학습 시료 검출']):,})",
                 f"{r['맞혀 보는 T 시작']} ~ {r['맞혀 보는 T 끝']} ({int(r['T 일 수'])}일)", f"{int(r['(행,h) 쌍']):,}", f"{int(r['(행,h) 쌍 검출']):,}",
                 c1(r['(행,h) 쌍'] / 60), c1(r['(행,h) 쌍 검출'] / 60)])
w(md_table(['구분', '학습 행 기준일 d', '학습 단위×날짜 행', '그중 검출', '학습 시료(검출)', '맞혀 보는 예측일 T', '(행,h) 쌍', '그중 검출', 'h 당 평균 행', 'h 당 평균 검출'], rows))
w()
w('맞혀 보는 해(기준일 d 07-01 ~ 다음 해 06-30)의 행·검출 [계산: 겹구성_계산.csv]:')
rows = []
for _, r in fold.iterrows():
    nm = '시험 해' if str(r['겹']) == '최종' else f"겹 {r['겹']}"
    rows.append([nm, f"{int(r['그해 행(d 07-01~06-30)']):,}", f"{int(r['그해 행 검출']):,}", f"{int(r['그해 시료']):,}", f"{int(r['그해 시료 검출']):,}",
                 f"{int(r['맞혀 보는 쌍에 들어간 행']):,}", f"{int(r['그중 검출 행']):,}"])
w(md_table(['구분', '단위×날짜 행', '그중 검출', '시료', '그중 검출', '(행,h) 쌍에 한 번이라도 들어간 행', '그중 검출'], rows))
w('- 겹 2 의 예측일 T 는 2024-02-29 가 있어 306일(나머지 305일). 사양 1절의 "모든 h 같은 305일"은 시험·겹 1·3 기준.')
w('- (행,h) 쌍 = 그 겹의 예측일 범위 안에 T = d − h 가 들어가는 (행, h) 조합 전부. 그래서 기준일 07-02 ~ 06-30 행이 h 마다 다르게 들어간다(07-01 행은 어느 h 로도 T 범위에 안 들어감).')
w('- 학습 행 = 정답 날 d ≤ 학습 끝인 단위×날짜 행. 행마다 h 하나를 무작위로 붙임(d ≤ 2024-06-30 행 = default_rng(42), 2024-07-01 ~ 2025-06-30 행 = default_rng(43), 1~60 균등) [코드: cv.py _h_assign, 서버 로그 h_rule].')
w('- 예측일 05-02 ~ 06-30 을 채점하지 않는 이유: 자료 끝이 06-30 이라 60일 뒤 정답이 있는 마지막 예측일이 05-01 → 모든 h 를 같은 예측일 범위로 채점하려고(겹도 같은 규칙) [사양 1절].')
w('- 결과 시차(검사 결과가 분기마다 늦게 DB 에 반영)는 학습 범위에 반영하지 않음 — 학습 끝을 앞당기지 않고 과거 검사 결과 칸을 학습 끝 값으로 고정 [사양 1·2절].')
w()
w('### 2-4. 점수 정의 [사양 3절, 서버: results/final_cat/config.json score_rule·threshold_desc]')
w('- **고르기 점수** = 구간 가중 PR-AUC = 0.5 × 평균(H1~20) + 0.3 × 평균(H21~40) + 0.2 × 평균(H41~60). h 마다 PR-AUC(sklearn average_precision) → 겹마다 계산 → 세 겹 평균. 바닐라 순위·기상 거르기·k·튜닝 목표·변수 빼기·섞기 중요도에 모두 이 점수를 씀.')
w('- **60일 평균 PR-AUC** = h 별 PR-AUC 의 h 1~60 단순 평균(겹 평균).')
w('- **판정 기준값** = 60개 h 를 합친 예측에서 Recall ≥ 0.80 를 지키는 기준값 가운데 Precision 이 가장 큰 값(같으면 Recall 큰 쪽), 판정 = 예측 확률 ≥ 기준값. '
  '교차검증 줄은 겹마다 그 겹 맞혀 보는 행으로 정한 기준값, 시험은 세 겹 맞혀 보는 행 예측을 합친 것에서 같은 규칙으로 정한 값을 최종 모델에 그대로 씀.')
w('- **Recall 0.8 에서 Precision** = 그 기준값에서 60개 h 합친 행의 Precision. Accuracy·Precision·Recall·F1·ROC-AUC·검사축소율·상위 k% = 그 기준값에서 h 마다 계산 → h 평균. TP·FN·FP·TN·N = h 하나당 평균 건수.')
w('- **검사축소율** = 1 − (TP + FP + FN) / N (= TN / N).')
w('- **상위 k% 포착률** = h 마다 예측 확률 상위 k% 행이 잡은 검출 수 ÷ 그 h 의 검출 수 → h 평균(k = 1·5·10·20).')
w('- **구간 가중 Recall** = 그 기준값에서 h 별 Recall 을 같은 가중치(0.5·0.3·0.2)로.')
w()
w('### 2-5. 지키는 것·기록 [사양 5절, 서버: results/final_cat/summary.json test_access]')
w('- 시험 행은 최종 단계(8번)에서만 만들 수 있게 코드가 막음(고르는 단계에서 만들면 오류로 멈춤). 시험 행을 처음 만든 기록 = 2026-10-22 01:28:00, 이유 "final 시험 입력" 한 번.')
w('- 나누기는 시간순만. 트리는 행 순서와 무관, BiLSTM·딥러닝은 에폭마다 학습 행을 섞음(11번에서 시간 순서 판과 비교).')
w('- 변수 빼기·이상치 넣기 판단은 교차검증으로만(시험으로 고르지 않음).')
w()
w('### 2-6. 알고 있는 한계(보고 때 적기로 한 것) [사양 7절, 서버: results/final_cat/summary.json limitations]')
w('- 부스팅 조기 종료를 그 겹의 맞혀 보는 행으로 함(지표 = 맞혀 보는 행 h 별 PR-AUC 평균, 최대 2,000회·인내 100) → 교차검증 점수가 실제보다 조금 높게 나올 수 있음. 진짜 점수는 시험 1년 [코드: models3.py 머리말, config3.json boost].')
w('- 운영 때 그날 검사할 제품명을 입력하는지는 확인되지 않음(그날 제품명 유형·TF-IDF 칸의 전제).')
w('- 과거 검사 결과 칸(과거 이력·WOE·제품명 과거 유형)은 학습 행에서는 예측일 전날까지 굴러가는 창, 맞혀 보는 행·시험 행에서는 그 겹(최종) 학습 끝 날 값으로 고정 → 그 해 뒤쪽 예측일일수록 오래된 값(운영과 같은 조건, 1년마다 다시 학습 전제).')
w('- 제품명 뭉치 정의(23개)는 학습 자료 전체(mnfctur_de ≤ 2025-06-30)의 검출 수로 고른 것 → 겹 1·2·3 에는 그 겹이 맞혀 보는 해의 검출 정보가 정의에 섞임(겹 점수에서 제품명 칸이 조금 좋게 나올 수 있음, 시험은 해당 없음).')
w('- 식품 속성 3단계 대표값 = 학습 기간 시료 최빈값(기간 고정 표).')
w('- 이상치 점수는 겹마다 그 겹 학습 정상 행으로 다시 학습. 변수 빼기 동안에는 처음 계산한 점수를 고정 묶음으로 씀.')
w()

# ---------------------------------------------------------------- II.3
w('## II.3 1차 베이스라인(바닐라 70개 → 기상 변수 거르기)')
w()
w('### 3-1. 바닐라 설정 [사양 4절 2번, 결과: chain_status.json, 코드: models3.py·dl3.py]')
w('- 입력 = 범주 4칸(시도·시군구·품목 2단계·3단계) + h + 기상(19변수). 기상 요약 = 변수마다 최근부터 10일 구간 평균 L/10개 + 창 평균·표준편차·최소·최대·기울기 5개(창 80일이면 13칸/변수), 일별 원값 = 변수마다 L칸, BiLSTM = 일별 시퀀스(값 19 + 결측 표시 19 = 38채널) [코드: 1차 features.py, dl3.py].')
w('- 트리 3종 = 라이브러리 기본 설정 + 조기 종료(최대 2,000회, 인내 100), 불균형 처리 없음. BiLSTM = 양방향 LSTM 64(1층) + 범주 임베딩 8차원×4 + h/60 → Linear 64 → 출력 1, Adam 1e-3, 배치 1,024, 최대 30에폭, 인내 5.')
w('- 설정 수 = 트리 3 × 창 10 × 입력 2 = 60 + BiLSTM 창 10 = 70(서버에서 트리 2프로세스 + BiLSTM 1프로세스를 나란히 실행).')
w(f"- 1위 고르는 규칙 = 고르기 점수 최대(같으면 짧은 창) → {selb['run']}, 고른 때 {selb['chosen_at']} [결과: selection_baseline.json].")
w()
w('### 3-2. 상위 10 [표: 바닐라_비교]')
rows = []
for _, r in vb.head(10).iterrows():
    rows.append([int(r['순위']), r['모델'], r['입력'], int(r['기상 창(일)'])] + [f4(r['겹1 구간 가중 PR-AUC']), f4(r['겹2 구간 가중 PR-AUC']), f4(r['겹3 구간 가중 PR-AUC'])]
                + mrow(r, [pc(r['상위 20% 포착률'])]))
w(md_table(['순위', '모델', '입력', '창(일)', '겹1', '겹2', '겹3'] + MH + ['상위 20%'], rows))
w()
w('### 3-3. 알고리즘·입력별 최고 창 [표: 바닐라_비교]')
rows = []
for m, inp in [('CatBoost', '기상 요약'), ('CatBoost', '일별 원값'), ('BiLSTM', '기상 시퀀스'), ('XGBoost', '기상 요약'), ('XGBoost', '일별 원값'),
               ('LightGBM', '기상 요약'), ('LightGBM', '일별 원값')]:
    d = vb[(vb['모델'] == m) & (vb['입력'] == inp)]
    b = d.loc[d[SEL].idxmax()]
    rows.append([m, inp, int(b['순위']), int(b['기상 창(일)']), f4(b[SEL]), f"{f4(d[SEL].min())} ~ {f4(d[SEL].max())}",
                 f4(b['Recall≥0.80 에서의 Precision(60개 h 합친 행)']), c1(b['FP(h 당)']), pc(b['검사축소율']), pc(b['상위 10% 포착률'])])
w(md_table(['모델', '입력', '전체 순위', '최고 창(일)', '구간 가중 PR-AUC', '창 10~100 범위', 'Recall 0.8 에서 Precision', 'FP(h당)', '검사축소율', '상위 10%'], rows))
w(f"- 70개 전체 범위: {f4(vb[SEL].min())}(LightGBM 일별 원값 90일) ~ {f4(vb[SEL].max())}(CatBoost 기상 요약 80일).")
w()
w('### 3-4. 기상 변수 거르기 [표: 기상거르기·기상거르기_변수중요도, 결과: selection_wx.json]')
w('- 규칙: 1위 설정(CatBoost 기상 요약 80일, 기본 설정)으로 회차마다 그때 기준 모델의 변수 섞기 중요도(맞혀 보는 행에서 그 변수 칸 전부를 h 마다 행끼리 섞었을 때 고르기 점수 감소, 세 겹 평균; 씨앗 = default_rng([20261020, 겹, 1000 + 변수 번호]))를 재고, '
  '낮은 것부터 하나씩 빼서 교차검증 → 고르기 점수가 오르면 뺀 채로, 아니면 되돌림 → 한 회차에서 뺀 것이 없으면 끝. 이동평균 차감 기상(19변수)은 이 단계와 별개 묶음이라 그대로 둠.')
w(f"- 시작 점수 {f4(selwx['start_score'])}{' = 바닐라 1위 실행 점수와 같음' if selwx['start_equals_vanilla'] else ' (바닐라 1위 실행 점수와 다름 — 확인 필요)'}. 결과: 일산화탄소(co) 하나만 빠짐 → {f4(selwx['final_score'])}, 남은 18변수 = 1차 베이스라인 기상 입력(4번부터 '기상'은 이 18변수).")
w()
w('1회차(19변수, 중요도 낮은 것부터 시도):')
rows = []
for _, r in wxh[wxh['회차'] == 1].iterrows():
    rows.append([int(r['단계']), wname(r['시도한 변수']), f"{r['변수 섞기 중요도']:.6f}", f4(r['기준 구간 가중 PR-AUC']), f4(r['후보 구간 가중 PR-AUC']), r['결정']])
w(md_table(['단계', '변수', '섞기 중요도', '기준 점수', '빼 본 점수', '결정'], rows))
w()
r2 = wxh[wxh['회차'] == 2]
w(f"2회차(18변수): {len(r2)}개 모두 되돌림(빼 본 점수 {f4(r2['후보 구간 가중 PR-AUC'].min())} ~ {f4(r2['후보 구간 가중 PR-AUC'].max())}, 기준 {f4(r2['기준 구간 가중 PR-AUC'].iloc[0])}) → 끝. "
  f"2회차 중요도 1위 {wname(wxi[wxi['회차'] == 2].iloc[0]['변수'])} {wxi[wxi['회차'] == 2].iloc[0]['변수 섞기 중요도(세 겹 평균)']:.6f}.")
w(f"- 기상 거르기 실행 시간 {int(jl(os.path.join(RES, 'chain_status.json'))['steps']['wxsel']['seconds']):,}초 [결과: chain_status.json].")
w('- 기상 변수 이름 출처 = `data\\중간\\시군구_기상_점검.csv` 의 \'변수\' 칸(soil_t_nas 는 원문 "지중온도(NAS, 깊이 설명 없음)").')
w()
w('### 3-5. 1차 베이스라인 지표(바닐라 1위 19변수 vs 기상 거르기 뒤 18변수) [표: 바닐라_비교 1위, 서버: results/_wxsel/cat__base__L080__n18__5a595bd76924/summary.json]')
mb = b18['mean']
rows = [['바닐라 1위(19변수)', f4(r_van[SEL]), f4(r_van['겹1 구간 가중 PR-AUC']), f4(r_van['겹2 구간 가중 PR-AUC']), f4(r_van['겹3 구간 가중 PR-AUC']),
         f4(r_van['60일 평균 PR-AUC']), f4(r_van['ROC-AUC(h 평균)']), f4(r_van['Recall≥0.80 에서의 Precision(60개 h 합친 행)']), c1(r_van['TP(h 당)']),
         c1(r_van['FN(h 당)']), c1(r_van['FP(h 당)']), pc(r_van['검사축소율']), pc(r_van['상위 10% 포착률']), pc(r_van['상위 20% 포착률']), r_van['최적 반복(겹별)']],
        ['1차 베이스라인(18변수, co 뺌)', f4(mb['wpr_auc']), f4(b18['by_fold']['1']['wpr_auc']), f4(b18['by_fold']['2']['wpr_auc']), f4(b18['by_fold']['3']['wpr_auc']),
         f4(mb['pr_auc']), f4(mb['roc_auc']), f4(mb['p_at_r']), c1(mb['tp']), c1(mb['fn']), c1(mb['fp']), pc(mb['inspect_reduction']), pc(mb['top10']), pc(mb['top20']),
         '/'.join(map(str, b18['best_rounds']))]]
w(md_table(['구분', '구간 가중 PR-AUC', '겹1', '겹2', '겹3', '60일 평균 PR-AUC', 'ROC-AUC', 'Recall 0.8 에서 Precision', 'TP(h당)', 'FN(h당)', 'FP(h당)', '검사축소율', '상위 10%', '상위 20%', '최적 반복(겹별)'], rows))
w()

# ---------------------------------------------------------------- II.4
w('## II.4 추가 데이터·특성공학(입력 묶음) + 스무딩 k')
w()
w('### 4-1. 입력 묶음과 칸 수 [코드: 3차 cv.py BUNDLES, 2차 e2fg.py·e2hist.py·e2nb.py·e2ae.py 규칙 문장, 서버: results/final_cat/config.json feature_columns]')
fc = cfg['feature_columns']


def cnt(pat):
    return sum(1 for n in fc if re.match(pat, n))


wv18 = selwx['wvars']
n_wx = sum(1 for n in fc if any(re.match(rf'^{re.escape(v)}_(b\d\d|mean|std|min|max|slope)$', n) for v in wv18))
counts = {
    'base': 1, 'wx': n_wx, 'spi': cnt(r'^spi[36]_'), 'soilmed': cnt(r'^soil_(ph|om|vldpha)$'), 'food': cnt(r'^food_'), 'extidx': cnt(r'^(eco_ratio|gw_pass_rate|haccp_ratio)$'),
    'rg': cnt(r'^rg_'), 'g123': cnt(r'^g[123]_'), 'wxanom': cnt(r'^.+_anom\d+_'), 'hist_core': cnt(r'^hist_(rate5y|rate1y|logn5y)_'), 'logn1': cnt(r'^hist_logn1y_'),
    'sd1': cnt(r'^hist_sd1y_'), 'n1': cnt(r'^n1_'), 'ae4': cnt(r'^ae_'), 'n2': cnt(r'^n2_'), 'n0': cnt(r'^tf_[cw]_'),
}
assert sum(counts.values()) == len(fc) == 996, (counts, len(fc))
BD = [
    ('wx', '기상 요약', f"18변수 × (최근부터 10일 구간 평균 8 + 창 평균·표준편차·최소·최대·기울기 5) = 13칸/변수, 창 T−80 … T−1(T 당일 제외)", '묶음째', '1차 features.py'),
    ('spi', '표준강수지수(SPI3·SPI6)', '시군구_일별_SPI 의 spi3·spi6 를 기상과 같은 창 요약(13칸씩)', '묶음째', 'e2fg.py F1'),
    ('soilmed', '토양 중앙값', '시군구_일별_토양 날짜 T 행의 토양 pH·유기물·유효인산(값 = 검정일 ∈ [T−1095, T−1] 기록의 중앙값, 검정 건수 칸 제외)', '변수 하나씩(3단위)', 'e2fg.py F2·soilmed'),
    ('food', '식품 속성', '품목 3단계 코드로 aw·pH·염도(na) 3칸 + aw 구간 범주(aw_label, 범주 칸) — 3단계 대표값 = 학습 기간(mnfctur_de ≤ 2024-06-30) 시료 값의 최빈값(검출 정답 안 씀), '
     '값이 없으면 원자료 3단계 코드 값 → 2단계 코드 값 → 결측(원자료 = 엑셀 우선, 없으면 ba_data.t_aw_cl), 겹마다 다시 만들지 않는 고정 표. 코드없음·표에 없는 코드는 결측',
     '변수 하나씩(3단위, aw 는 aw_label 과 한 단위)', 'e2fg.py F3, 품목3단계_식품속성.parquet 머리 기록(06_품목_식품속성.py)'),
    ('extidx', '외부 지표', '친환경 인증 비율((유기+무농약 인증 면적) ÷ 경지면적, 자료연도 2016~2025, 사용가능일 = 자료연도+2년 1월 1일) · 지하수 정기수질검사 적합률(적합 ÷ 검사, 자료연도 2013~2024) · '
     'HACCP 지정 비율(T−1 에 유효한 그 시군구·업종묶음 지정 수 ÷ 전국 같은 업종묶음 지정 수, 2017-08-03 전·대응 업종 없는 품목은 결측) 각 1칸 — 예측일 T 에 사용가능일 ≤ T−1 인 가장 최근 값',
     '변수 하나씩(3단위)', 'e2fg.py P_RULE·HACCP_RULE'),
    ('rg', '생장 범위', "예측인자 정리 PPT 표 '리스테리아 모노사이토제네스 예측인자 생장 최적·한계 범위'(기온 최적 30~37°C·한계 −2~45°C, 5cm 지중온도 최적 8~15°C·한계 8~30°C, 토양 pH 최적 6.5~7.0·한계 5.0~8.0, "
     '식품 aw 한계 0.90~0.997·최적 0.97, 식품 pH 한계 4.0~9.6·최적 7.0)로 0/1·거리 칸: 기상 6칸 × (창 평균·최근 10일 평균) 12 + 토양 pH 3 + 식품 4', '묶음째', 'e2fg.py RG_RULE'),
    ('g123', '생장식 지수', "'리스테리아 생장식.xlsx' 계수로 G1 온도 생장 요인 f(T)(Silva 2020; 평균·최근 10일 평균·합·f>0 날 수 4), G2 µmax·생장 확률 P(Tienungoon 2000, Scott A·L5 두 균주; 3×3=9), G3 강수 검출 확률(Ivanek 2009; 3)", '묶음째', 'e2fg.py G1~G3'),
    ('wxanom', '이동평균 차감 기상', 'a_t = x_t − 앞 30일 평균(당일 제외), 19변수(co 포함 그대로) × (창 평균·최근 10일 평균·최대)', '묶음째', 'e2fg.py WXANOM_RULE'),
    ('hist_core', '과거 검출 이력 핵심', '3수준(품목 3단계 전국·시군구 전 품목·단위) × {5년 스무딩 검출률, 1년 스무딩 검출률, 5년 log(1+검사 수)}; 스무딩 검출률 = (s + k·p0)/(n + k), p0 = 같은 창 전국 검출률, k = 200', '묶음째', 'e2hist.py'),
    ('logn1', '1년 로그 검사 수', '3수준 × log(1 + 1년 검사 수)', '묶음째', 'e2hist.py HX_RULE'),
    ('sd1', '1년 검출률 불확실성', '3수준 × sqrt(m(1−m)/(n+k+1)), m = 1년 스무딩 검출률(베타 사후 표준편차)', '묶음째', 'e2hist.py HX_RULE'),
    ('woe', 'WOE', '시도·시군구·2단계·3단계 × ln((s+0.5)/(S+0.5)) − ln((n−s+0.5)/(N−S+0.5)), 5년 창 — 변수 빼기에서 빠짐(최종 모델엔 없음)', '묶음째', 'e2hist.py woe_features'),
    ('n1', '제품명 과거 유형', '단위의 과거 시료 중 뭉치 c 비율(share)·뭉치 c 스무딩 검출률(rate) × 5년·1년, 뭉치 23개', '묶음째', 'e2nb.py N1_RULE'),
    ('ae4', '이상치 점수(AE·VAE)', 'AEt·VAEt(표 입력)·AEs·VAEs(기상 시퀀스) 점수 4칸, 겹마다 그 겹 학습 행 불검출만으로 학습(학습 행 = 5겹 교차 적합, 점수 낼 행 = 겹 모델 5개 평균)', '묶음째', 'ae3.py·e2ae.py'),
    ('n2', '그날 제품명 유형(운영 전제)', '그날(목표일 d) 그 단위 시료의 뭉치 멀티핫(23칸, h 와 무관)', '묶음째', 'e2nb.py N2_RULE'),
    ('n0', '그날 제품명 TF-IDF(운영 전제)', '글자 2~4그램 300 + 낱말 200 = 500칸, 어휘·IDF = 겹마다 그 겹 학습 기간 파일(단위날짜_tfidf_train<학습 끝>.parquet)', '묶음째', 'e2nb.py N0_RULE·cv.py'),
]
rows = []
for code, nm, how, unit, src in BD:
    n = counts.get(code, 4 if code == 'woe' else '')
    if code == 'food':
        n = f'{n} + 범주 1'
    rows.append([code, nm, n, how, unit, src])
w(md_table(['코드', '묶음', '칸 수', '만드는 법(코드 규칙 요약)', '변수 빼기 단위', '출처 코드'], rows))
w(f"- 늘 들어가는 칸(빼기 후보 아님) = h 1칸 + 범주 4칸(시도·시군구·2단계·3단계). 최종 모델 입력 = 숫자 {len(fc)}칸 + 범주 {len(cfg['cat_columns'])}칸({', '.join(cfg['cat_columns'])}). 튜닝(변수 전부) 입력 = 숫자 {len(fc) + 4}칸(WOE 4칸 포함) + 범주 5칸.")
w('- 과거 검사 결과로 만드는 칸(hist_core·logn1·sd1·woe·n1, 그 안의 전국 p0): 학습 행 = 예측일 T 전날까지 굴러가는 창(1년 = 365일, 5년 = 1,826일), 맞혀 보는 행·시험 행 = 그 겹(최종) 학습 끝 날까지 시료로 만든 값을 그 해 내내 고정 [코드: cv.py PAST_RULE_DESC].')
w('- 외부 지표·토양 중앙값·식품 속성만 변수 하나씩 빼기 후보, 나머지는 묶음째 → 빼기 후보 22단위(사용자 확정 10-20) [사양 4절 7번].')
w('- 생장 범위 자료 이름: 코드 문구는 "리서치팀 예측인자 정리 ppt". 사용자 답(10-19 메모 기록)은 "식정원이 애초에 준 자료" → 보고서 표기는 확인 필요.')
w()
w('### 4-2. 제품명 뭉치 23개 [data/데이터셋/제품명/뭉치_정의.csv, 선정 기간 = 학습 자료 전체(mnfctur_de ≤ 2025-06-30)]')
rows = []
for _, r in nbdef.iterrows():
    rows.append([r['칼럼명'], r['한글 이름'], r['방향'], f"{int(r['학습 행']):,}", int(r['학습 검출']), f"{r['학습 검출률(%)']:.2f}%"])
w(md_table(['칼럼', '한글 이름', '방향', '학습 시료', '학습 검출 시료', '학습 검출률'], rows))
w('- 정의 파일의 \'학습 행\'·\'학습 검출\' 칸 = 시료 수(시료별_뭉치.parquet 와 시료 파일로 기준일 ≤ 2025-06-30 시료를 다시 세어 nb_enoki 6,805·499, nb_raw_meat 12,817·231, nb_preheat 36,080·97, nb_extract 13,128·1 로 같음 — `work3\\scripts\\check_nb_counts_src_route.py`). '
  '\'방향\'(위험 높음·기준 수준·위험 낮음)도 정의 파일 값 그대로. 한 시료가 여러 뭉치에 들 수 있음 [코드: e2nb.py N1_RULE].')
w()
w('### 4-3. 스무딩 계수 k 고르기 [표: k선택, 결과: selection_k.json]')
w(f"- 모델·입력 = 1차 베이스라인(CatBoost 기상 요약 80일, 기본 설정) + 기상 18변수 + 과거 이력 핵심 칸(hist_core 9칸). 후보 10·30·50·100·200(교수님 readme 4.1절과 같음), 고르기 점수 최대(같으면 작은 k) → k = {selk['k']}.")
rows = []
for _, r in ks.iterrows():
    rows.append([int(r['k']), f4(r['겹1 구간 가중 PR-AUC']), f4(r['겹2 구간 가중 PR-AUC']), f4(r['겹3 구간 가중 PR-AUC'])] + mrow(r, [r['최적 반복(겹별)']]))
w(md_table(['k', '겹1', '겹2', '겹3'] + MH + ['최적 반복(겹별)'], rows))
w(f"- 다섯 후보 모두 1차 베이스라인(이력 칸 없음) {f4(mb['wpr_auc'])} 보다 낮음(최고 k=200 {f4(ks[SEL].max())}).")
w()

# ---------------------------------------------------------------- II.5
w('## II.5 튜닝(CatBoost Optuna 30회)')
w()
w('### 5-1. 설정 [결과: params_cat.json·tune_cat/config.json, config3.json optuna]')
sp = jl(os.path.join(RES, 'tune_cat', 'config.json'))['space']
w(f"- 모델 = 1차 베이스라인 모델 하나(CatBoost, 기상 요약 80일). 입력 = 변수 전부 16묶음({', '.join(pcat['bundles'])}), 기상 18변수, k = 200. 목표 = 고르기 점수 하나(구간 가중 PR-AUC), 시도 30회, TPE(씨앗 42), 부스팅 최대 2,000회·인내 100(맞혀 보는 행 조기 종료).")
w('- 탐색 범위: ' + ', '.join(f"{k} {v[1]}~{v[2]}({'로그' if 'log' in v[0] else '정수' if v[0] == 'int' else '균등'})" for k, v in sp.items()) + '.')
w(f"- 튜닝 실행 시간 {int(jl(os.path.join(RES, 'chain_status.json'))['steps']['tune']['seconds']):,}초 [결과: chain_status.json].")
w()
w('### 5-2. 상위 5 후보(구간 가중 PR-AUC 기준) [표: 튜닝후보_cat]')
rows = []
for _, r in tc[tc['구간 가중 PR-AUC 상위 5'].astype(str).eq('예')].iterrows():
    p = json.loads(r['파라미터'])
    rows.append([int(r['시도'])] + mrow(r) + [pc(r['상위 1% 포착률']), pc(r['상위 20% 포착률']), f4(r['ROC-AUC(h 평균)']), p['depth'], f"{p['learning_rate']:.4f}",
                                           f"{p['l2_leaf_reg']:.3f}", f"{p['random_strength']:.3f}", f"{p['bagging_temperature']:.3f}", r['최적 반복(겹별)']])
w(md_table(['시도'] + MH + ['상위 1%', '상위 20%', 'ROC-AUC', '깊이', '학습률', 'l2_leaf_reg', 'random_strength', 'bagging_temperature', '최적 반복(겹별)'], rows))
w()
w(f"- 30회 전체 범위: 구간 가중 PR-AUC {f4(tc[SEL].min())} ~ {f4(tc[SEL].max())}, FP(h당) {c1(tc['FP(h 당)'].min())} ~ {c1(tc['FP(h 당)'].max())}, "
  f"Recall 0.8 에서 Precision {f4(tc['Recall≥0.80 에서의 Precision(60개 h 합친 행)'].min())} ~ {f4(tc['Recall≥0.80 에서의 Precision(60개 h 합친 행)'].max())}.")
w()
w('### 5-3. 시도 18 을 고른 근거(수치) [표: 튜닝후보_cat, 결과: params_cat.json chosen_at 2026-10-21 20:12:59]')
t5 = tc[tc['구간 가중 PR-AUC 상위 5'].astype(str).eq('예')]
r18 = tc[tc['시도'] == 18].iloc[0]
w('- 상위 5개는 판정 기준값을 모두 Recall ≈ 0.80 에 맞췄으므로(기준값 Recall 0.8001~0.8012) 같은 Recall 에서 오탐(FP)·Precision 을 비교할 수 있음.')
w(f"- 상위 5개 안에서 시도 18 이 FP 가장 적음({c1(r18['FP(h 당)'])}; 나머지 {', '.join(c1(x) for x in t5[t5['시도'] != 18]['FP(h 당)'])}), "
  f"Recall 0.8 에서 Precision 가장 큼({f4(r18['Recall≥0.80 에서의 Precision(60개 h 합친 행)'])}), 검사축소율 가장 큼({pc(r18['검사축소율'])}), "
  f"상위 1%·10%·20% 포착률 가장 큼({pc(r18['상위 1% 포착률'])}·{pc(r18['상위 10% 포착률'])}·{pc(r18['상위 20% 포착률'])}).")
r5 = tc[tc['시도'] == 5].iloc[0]
w(f"- 고르기 점수 1위 시도 5({f4(r5[SEL])})와 비교하면 시도 18 은 고르기 점수가 {f4(r5[SEL] - r18[SEL])} 낮고, FP 는 h 당 {c1(r5['FP(h 당)'] - r18['FP(h 당)'])}건 적음.")
fewer = tc[tc['FP(h 당)'] < r18['FP(h 당)']].sort_values('FP(h 당)')
w(f"- 30회 전체로 넓히면 시도 18 보다 FP 가 적은 시도가 {len(fewer)}개 있음: " + ', '.join(
    f"시도 {int(r['시도'])}(FP {c1(r['FP(h 당)'])}, 고르기 점수 {f4(r[SEL])}, 상위 5 밖)" for _, r in fewer.iterrows())
  + ' → "오탐 가장 적음"은 상위 5 후보 안에서의 말.')
w('- 사용자가 고른 사유 문장은 결과 파일에 없음(params_cat.json 에는 고른 시각만) — 보고서 문구는 확인 필요.')
w()
w('### 5-4. 시도 18 파라미터 [결과: params_cat.json]')
pp = pcat['params']
w(md_table(['learning_rate', 'depth', 'l2_leaf_reg', 'random_strength', 'bagging_temperature', 'iterations(상한)', 'random_seed', '튜닝 때 최적 반복(겹별)'],
           [[f"{pp['learning_rate']:.6f}", pp['depth'], f"{pp['l2_leaf_reg']:.6f}", f"{pp['random_strength']:.6f}", f"{pp['bagging_temperature']:.6f}", pp['iterations'],
             pp['random_seed'], '/'.join(map(str, pcat['scores']['best_rounds']))]]))
w()

# ---------------------------------------------------------------- II.6
w('## II.6 변수 선택(묶음 빼기)')
w()
w('### 6-1. 규칙 [사양 4절 7번, 결과: selection_drop_cat.json, 코드: run3.py cmd_drop]')
w('- 시도 18 설정값을 그대로 쓰고(get_params → set_params), 회차마다 그때 기준 모델의 묶음 섞기 중요도(겹마다 맞혀 보는 행에서 그 단위의 칸만 h 마다 행끼리 섞어 다시 예측 → 고르기 점수 감소 → 세 겹 평균, 다시 학습 안 함; 씨앗 default_rng([20261020, 겹, 단위 번호]))를 잼.')
w('- 중요도 낮은 단위부터 하나씩 빼고 교차검증 → 고르기 점수가 오르면(허용 0) 뺀 채로, 아니면 되돌림 → 한 회차에서 뺀 것이 없을 때까지. 범주 4칸·h 는 후보 아님. 자동 조합 탐색 안 함.')
w()
w('### 6-2. 단계 요약 [표: 묶음빼기_cat]')
s0 = dr[dr['회차'] == 0].iloc[0]
rr1, rr2 = dr[dr['회차'] == 1], dr[dr['회차'] == 2]
w(f"- 시작 = 22단위, 기준 {f4(s0['기준 구간 가중 PR-AUC'])}(= 튜닝 시도 18 점수).")
w(f"- 1회차: {len(rr1)}단위를 중요도 낮은 것부터 시도. {int(d_wo['단계'])}번째 시도 WOE(중요도 {d_wo['단위 섞기 중요도']:.6f}) 빼니 {f4(d_wo['기준 구간 가중 PR-AUC'])} → {f4(d_wo['후보 구간 가중 PR-AUC'])} 로 올라 뺌. 나머지 {len(rr1) - 1}단위는 모두 되돌림.")
w(f"- 2회차: 남은 21단위 모두 되돌림(빼 본 점수 {f4(rr2['후보 구간 가중 PR-AUC'].min())} ~ {f4(rr2['후보 구간 가중 PR-AUC'].max())}, 기준 {f4(rr2['기준 구간 가중 PR-AUC'].iloc[0])}) → 끝. 최종 입력 = 21단위.")
w(f"- 최종 21단위 교차검증 점수 {f4(seld['final_mean']['wpr_auc'])}, 겹별 최적 반복 {'/'.join(map(str, seld['final_best_rounds']))}.")
w()
UN = {u: n for u, n in [('wx', '기상 요약'), ('spi', '표준강수지수(SPI3·SPI6)')]}
for rnd, d in ((1, rr1), (2, rr2)):
    w(f'{rnd}회차 표:')
    rows = []
    for _, r in d.iterrows():
        rows.append([int(r['단계']), r['시도한 단위'], f"{r['단위 섞기 중요도']:.6f}", f4(r['기준 구간 가중 PR-AUC']), f4(r['후보 구간 가중 PR-AUC']), r['결정']])
    w(md_table(['단계', '단위', '섞기 중요도', '기준 점수', '빼 본 점수', '결정'], rows))
    w()
w('- 섞기 중요도의 겹별 값은 [결과: selection_drop_cat.json importance_by_round.by_fold].')
w()

# ---------------------------------------------------------------- II.7
w('## II.7 최종 모델·시험 1년')
w()
w('### 7-1. 최종 설정 [서버: results/final_cat/summary.json·config.json]')
cvm, tsm = summ['cv_mean'], summ['test']
w(f"- CatBoost 시도 18 파라미터, 입력 21단위(숫자 {len(fc)}칸 + 범주 5칸), 기상 18변수 창 80일, k = 200, 씨앗 42.")
w(f"- 학습 자료 전체(d ≤ 2025-06-30) {summ['train_rows']:,}행(검출 {summ['train_pos']:,})로 다시 학습. 반복 수 = 세 겹 최적 반복({'/'.join(str(summ['cv_by_fold'][k]['best_rounds']) for k in ('1', '2', '3'))}) 평균 → {summ['rounds']}. "
  f"CatBoost 학습률(실제 쓴 값) {summ['cat_learning_rate']:.6f}.")
w(f"- 판정 기준값 = 세 겹 맞혀 보는 행 예측 합친 것(후보 {summ['threshold']['n_candidates']:,}개)에서 Recall ≥ 0.80 중 Precision 최대 → {summ['threshold']['threshold']:.6f}(그 합친 행에서 Precision {summ['threshold']['precision']:.4f}, Recall {summ['threshold']['recall']:.4f}).")
w(f"- 시험 = 예측일 T 2025-07-01 ~ 2026-05-01(305일), (행,h) {summ['test_pairs']:,}쌍(검출 {summ['test_pos_pairs']:,}) — 한 번 채점. 최종 단계 실행 {summ['seconds']:,}초.")
w('- 최종 단계에서 다시 돌린 교차검증 점수가 변수 빼기 결과(0.0927)와 ' + ('같음.' if summ['cv_equals_drop_result'] else '다름 — 확인 필요.'))
w()
w('### 7-2. 교차검증 vs 시험 [표: 최종_시험_cat, 서버: results/final_cat/summary.json]')
w('교차검증 = 세 겹 평균(겹마다 그 겹 기준값 0.00491/0.00513/0.00447), 시험 = 고정 기준값 0.004895.')
cvr, ter = ft.iloc[0], ft.iloc[1]
items = [('구간 가중 PR-AUC', SEL, f4), ('60일 평균 PR-AUC', '60일 평균 PR-AUC', f4), ('ROC-AUC(h 평균)', 'ROC-AUC(h 평균)', f4),
         ('Recall 0.8 에서 Precision(60개 h 합친 행)', 'Recall≥0.80 에서의 Precision(60개 h 합친 행)', f4), ('기준값에서 Recall(60개 h 합친 행)', '기준값 Recall(60개 h 합친 행)', f4),
         ('Precision(h 평균)', 'Precision(h 평균)', f4), ('Recall(h 평균)', 'Recall(h 평균)', f4), ('구간 가중 Recall', '구간 가중 Recall', f4), ('F1(h 평균)', 'F1(h 평균)', f4),
         ('Accuracy(h 평균)', 'Accuracy(h 평균)', f4), ('TP(h 당)', 'TP(h 당)', c1), ('FN(h 당)', 'FN(h 당)', c1), ('FP(h 당)', 'FP(h 당)', c1), ('TN(h 당)', 'TN(h 당)', c1),
         ('N(h 당)', 'N(h 당)', c1), ('검사축소율', '검사축소율', pc), ('상위 1% 포착률', '상위 1% 포착률', pc), ('상위 5% 포착률', '상위 5% 포착률', pc),
         ('상위 10% 포착률', '상위 10% 포착률', pc), ('상위 20% 포착률', '상위 20% 포착률', pc)]
rows = [[a, fn(cvr[b]), fn(ter[b])] for a, b, fn in items]
rows.append(['기준값', cvr['기준값'], ter['기준값']])
rows.append(['반복 수', cvr['반복 수'], ter['반복 수']])
w(md_table(['지표', '교차검증(세 겹 평균)', '시험 1년'], rows))
w()
w('### 7-3. 혼동행렬(h 하나당 평균 건수) [표: 최종_시험_cat]')
for nm, r in (('교차검증(세 겹 평균)', cvr), ('시험 1년', ter)):
    w(f'{nm}:')
    w(md_table(['', '예측 검출(판정 ≥ 기준값)', '예측 불검출'], [['실제 검출', c1(r['TP(h 당)']), c1(r['FN(h 당)'])], ['실제 불검출', c1(r['FP(h 당)']), c1(r['TN(h 당)'])]]))
    w(f"- N = {c1(r['N(h 당)'])}, 검사축소율 = 1 − (TP+FP+FN)/N = {pc(r['검사축소율'])}.")
    w()
w('### 7-4. h 구간별(최종 모델) [계산: 최종_h별_교차검증_시험.csv — 겹별 60일 평균·구간 가중 값이 서버 summary.json 과 같음]')
bt, bc = calc['band_test_pr_auc'], calc['band_cv_pr_auc']
rt, rc = calc['band_test_recall'], calc['band_cv_recall']
rows = [[b, f4(bc[b]), f4(bt[b]), f4(rc[b]), f4(rt[b])] for b in ('H1~20', 'H21~40', 'H41~60')]
w(md_table(['h 구간(가중치)', 'PR-AUC 교차검증', 'PR-AUC 시험', 'Recall 교차검증', 'Recall 시험'], rows))
w(f"- 시험 h 별 PR-AUC: h=1 {f4(calc['test_pr_auc_h1_h60'][0])}, h=60 {f4(calc['test_pr_auc_h1_h60'][1])}, 최고 h={calc['test_pr_auc_max'][0]} {f4(calc['test_pr_auc_max'][1])}, 최저 h={calc['test_pr_auc_min'][0]} {f4(calc['test_pr_auc_min'][1])}.")
w(f"- 시험 h 별 Recall {f4(calc['test_recall_range'][0])} ~ {f4(calc['test_recall_range'][1])}, Precision {f4(calc['test_precision_range'][0])} ~ {f4(calc['test_precision_range'][1])}, h 별 시험 검출 수 {calc['test_pos_per_h_range'][0]} ~ {calc['test_pos_per_h_range'][1]}.")
hh = pd.read_csv(os.path.join(CALC, '최종_h별_교차검증_시험.csv'), encoding='utf-8-sig').sort_values('h')
ge = hh[hh['pr_auc_test'] >= hh['pr_auc_cv']]
w(f"- h 별 PR-AUC 가 시험 ≥ 교차검증인 h: {', '.join(f'h={int(r.h)}(시험 {r.pr_auc_test:.4f} vs 교차검증 {r.pr_auc_cv:.4f})' for r in ge.itertuples()) or '없음'}; 나머지 {60 - len(ge)}개 h 는 시험이 낮음. "
  f"Precision(기준값에서)은 {int((hh['precision_test'] > hh['precision_cv']).sum())}개 h 에서 시험이 높음.")
w('- 교차검증 h 별 Precision·Recall 은 겹마다 그 겹 기준값으로 계산해 세 겹 평균(시험은 고정 기준값).')
H_GE = ge
w()
w('### 7-5. 겹별 교차검증(최종 21단위 모델) [서버: results/final_cat/summary.json cv_by_fold]')
rows = []
for k in ('1', '2', '3'):
    b = summ['cv_by_fold'][k]
    rows.append([f'겹 {k}', f4(b['wpr_auc']), f4(b['pr_auc']), f4(b['roc_auc']), f4(b['p_at_r']), c1(b['tp']), c1(b['fn']), c1(b['fp']), pc(b['inspect_reduction']),
                 pc(b['top10']), f"{b['threshold']:.6f}", b['best_rounds'], f"{b['n_train_rows']:,}", f"{b['n_train_pos']:,}"])
w(md_table(['겹', '구간 가중 PR-AUC', '60일 평균 PR-AUC', 'ROC-AUC', 'Recall 0.8 에서 Precision', 'TP(h당)', 'FN(h당)', 'FP(h당)', '검사축소율', '상위 10%', '기준값', '최적 반복', '학습 행', '학습 검출'], rows))
w()
w('### 7-6. 단계별 교차검증 점수 흐름(같은 고르기 점수)')
rows = [['바닐라 1위(CatBoost 요약 80일, 19변수)', f4(r_van[SEL])], ['1차 베이스라인(co 뺌, 18변수)', f4(mb['wpr_auc'])],
        ['k 고르기(+ 과거 이력 핵심, k=200)', f4(ks[SEL].max())], ['튜닝 시도 18(변수 전부 22단위)', f4(r18[SEL])], ['변수 빼기 뒤(21단위, WOE 뺌)', f4(seld['final_mean']['wpr_auc'])],
        ['최종 모델 시험 1년', f4(tsm['wpr_auc'])]]
w(md_table(['단계', '구간 가중 PR-AUC'], rows))
w()

# ---------------------------------------------------------------- II.8
w('## II.8 추가 실험')
w()
w('### 8-1. 고위험 5품목 [결과: selection_items.json, 품목 이름 = 시료_시군구.parquet 의 train_prdlst_nm·train_prdlst_nm_2]')
w('- 고르는 규칙: 기준일 ≤ 학습 끝 시료의 품목 3단계별 검출 수 내림차순(같으면 코드 순) 상위 5. 겹마다 그 겹 학습 자료로, 최종은 학습 자료 전체로 다시 고름 [코드: cv.py top_items].')
fi = seli['final']
tot = sum(fi['pos']) / fi['share_of_pos']
rows = []
for j, (code, pos, n) in enumerate(zip(fi['codes'], fi['pos'], fi['n'])):
    nm = PNAME.get(code, {})
    rows.append([j + 1, code, nm.get('train_prdlst_nm', '확인 필요'), nm.get('train_prdlst_nm_2', '확인 필요'), f'{n:,}', f'{pos:,}', f'{pos / n * 100:.2f}%', f'{pos / tot * 100:.1f}%'])
w('최종(학습 자료 전체 ≤ 2025-06-30) 5품목:')
w(md_table(['순위', '3단계 코드', '2단계 이름', '3단계 이름', '시료 수', '검출 시료', '검출률', '전체 검출 중 비중'], rows))
w(f"- 5품목 합계 검출 비중 {fi['share_of_pos'] * 100:.1f}%(전체 학습 검출 시료 {tot:,.0f}건 중 {sum(fi['pos']):,}건).")
rows = []
for k in ('1', '2', '3'):
    f_ = seli['folds'][k]
    rows.append([f'겹 {k}(≤ {f_["train_end"]})', ', '.join(f"{c}({PNAME.get(c, {}).get('train_prdlst_nm_2', '확인 필요')}) {p}" for c, p in zip(f_['codes'], f_['pos'])), f"{f_['share_of_pos'] * 100:.1f}%"])
w('겹별 5품목(코드(3단계 이름) 검출 시료 수):')
w(md_table(['겹', '5품목', '검출 비중'], rows))
w('- 겹마다 4·5위 품목이 다름: 겹 1 = 즉석섭취·편의식품류·햄류, 겹 2 = 포장육·즉석섭취·편의식품류, 겹 3 = 포장육·햄류, 최종 = 포장육·즉석섭취·편의식품류.')
w()
w('### 8-2. 고위험 품목 전용 모델 vs 전체 모델(같은 품목 행) [표: 고위험품목_비교_cat, 결과: post_cat/summary.json items]')
w('- 전용 모델 = 같은 설정값(시도 18)·같은 입력 21단위로 그 품목 행만 학습(이상치 점수 칸은 전체 겹 값 그대로). 전체 모델 = 7절 최종 모델의 같은 품목 행 예측. 기준값 = 각 모델의 세 겹 맞혀 보는 행(그 품목) 예측 합친 것에서 Recall ≥ 0.80 중 Precision 최대.')
rows = []
for _, r in it.iterrows():
    rows.append([r['구분'], f4(r[SEL]), f4(r['60일 평균 PR-AUC']), f4(r['Recall≥0.80 에서의 Precision(60개 h 합친 행)']), f4(r['기준값 Recall(60개 h 합친 행)']),
                 c1(r['TP(h 당)']), c1(r['FN(h 당)']), c1(r['FP(h 당)']), c1(r['N(h 당)']), pc(r['검사축소율']), pc(r['상위 10% 포착률']), r['기준값'], r['반복 수']])
w(md_table(['구분', '구간 가중 PR-AUC', '60일 평균 PR-AUC', 'Recall 0.8 에서 Precision', '기준값 Recall(합친 행)', 'TP(h당)', 'FN(h당)', 'FP(h당)', 'N(h당)', '검사축소율', '상위 10%', '기준값', '반복 수'], rows))
pi = post['items']
w(f"- 시험 해 5품목 행: (행,h) {pi['test_pairs']:,}쌍(검출 {pi['test_pos']:,}) = 시험 전체 쌍의 {pi['test_pairs'] / summ['test_pairs'] * 100:.1f}%, 검출 쌍의 {pi['test_pos'] / summ['test_pos_pairs'] * 100:.1f}%.")
w(f"- 전용 모델 시험 기준값 {pi['thr_group']['threshold']:.6f} 에서 시험 Recall(합친 행) {f4(it.iloc[2]['기준값 Recall(60개 h 합친 행)'])}(교차검증으로 정한 기준값을 그대로 써서 0.80 아래).")
w()
w('### 8-3. 이상치 탐지 단독(점수를 예측값으로 바로 채점) [표: 이상치단독, 결과: post_cat/summary.json ae_solo]')
w('- AEt·VAEt = 표 입력(튜닝 입력 숫자 칸에서 그날 제품명 n2·n0·거른 기상 변수 제외) 오토인코더·VAE, AEs·VAEs = 기상 시퀀스(80일) 오토인코더·VAE. 겹마다 그 겹 학습 행 불검출만으로 학습(5겹 교차 적합, 점수 낼 행 = 겹 모델 5개 평균), 최종 = 학습 자료 전체 불검출로 학습해 시험 채점. 점수가 클수록 검출 쪽.')
w(f"- 지터링 판 = 겹 모델 학습 행을 한 벌 더 붙이고, 붙인 벌의 표준화 입력에 N(0, {post['jitter']['sigma']}²) 잡음(배치마다 새로, 시퀀스는 값 채널만, 씨앗 {post['jitter']['seed']}). 조기 종료·점수 계산에는 잡음 없음 [코드: ae3.py 머리말].")
rows = []
for _, r in ae1.iterrows():
    rows.append([r['점수'], r['판'], r['구분'], f4(r[SEL]), f4(r['60일 평균 PR-AUC']), f4(r['ROC-AUC(h 평균)']), f4(r['Recall≥0.80 에서의 Precision(60개 h 합친 행)']),
                 f4(r['기준값 Recall(60개 h 합친 행)']), c1(r['FP(h 당)']), pc(r['검사축소율']), pc(r['상위 10% 포착률'])])
w(md_table(['점수', '판', '구분', '구간 가중 PR-AUC', '60일 평균 PR-AUC', 'ROC-AUC', 'Recall 0.8 에서 Precision', '기준값 Recall(합친 행)', 'FP(h당)', '검사축소율', '상위 10%'], rows))
vm = post['ae_into_model']['version_cv_mean_of_4']
w(f"- 판 고르기(점수 4개의 교차검증 구간 가중 PR-AUC 평균): 원본 {vm['원본']:.6f} vs 지터링 {vm['지터링']:.6f} → {post['ae_into_model']['best_version']} 판.")
w()
w('### 8-4. 이상치 점수를 최종 CatBoost 입력으로 넣은 판 vs 안 넣은 판 [표: 이상치넣기_비교_cat]')
rows = []
for _, r in ae2.iterrows():
    rows.append([r['판'], r['구분'], f4(r[SEL]), f4(r['60일 평균 PR-AUC']), f4(r['Recall≥0.80 에서의 Precision(60개 h 합친 행)']), f4(r['기준값 Recall(60개 h 합친 행)']),
                 c1(r['FP(h 당)']), c1(r['FN(h 당)']), pc(r['검사축소율']), pc(r['상위 10% 포착률']), r['반복 수'],
                 r['8번과 같은 모델'] if not pd.isna(r['8번과 같은 모델']) else '', r['최종으로 고름(교차검증)'] if not pd.isna(r['최종으로 고름(교차검증)']) else ''])
w(md_table(['판', '구분', '구간 가중 PR-AUC', '60일 평균 PR-AUC', 'Recall 0.8 에서 Precision', '기준값 Recall(합친 행)', 'FP(h당)', 'FN(h당)', '검사축소율', '상위 10%', '반복 수', '8번과 같은 모델', '최종으로 고름'], rows))
vno = post['ae_into_model']['variants']['이상치 점수 안 넣은 판']
w(f"- 안 넣은 판의 시험 기준값 {vno['threshold']['threshold']:.6f}(세 겹 합친 행에서 Recall {vno['threshold']['recall']:.4f}, Precision {vno['threshold']['precision']:.4f}) → 시험에서 Recall(합친 행) {f4(ae2.iloc[3]['기준값 Recall(60개 h 합친 행)'])}.")
w('- 넣은 판 = 7절 최종 모델과 같음(원본 점수가 이미 21단위 안에 있음). 안 넣은 판 = 21단위에서 이상치 점수 묶음만 뺌(교차검증 점수는 묶음 빼기 2회차의 같은 후보 점수와 같음). 최종 = 교차검증 점수가 높은 넣은 판.')
w()
w('### 8-5. 딥러닝 8개 [표: 딥러닝단계, 코드: dl11.py 머리말]')
w('- 공통: 창 80일, 기상 18변수(값 + 결측 표시 채널), 범주 4칸 임베딩, 다른 입력 칸 없음. Adam 1e-3, 배치 1,024, 최대 30에폭, 조기 종료 = 맞혀 보는 행 h 별 PR-AUC 평균 인내 5. 교차검증만(시험 안 함).')
w('- BiLSTM = 양방향 LSTM 64 + 임베딩 + [h/60] → Linear 64 → 출력. 트랜스포머 = Linear(채널→64) + 학습 위치 임베딩 → TransformerEncoder 2층·4헤드·ff 128·dropout 0.1 → 시간 평균 풀링 + 임베딩 + [h/60] → Linear 64 → 출력.')
w('- 방식 A = 입력에 h/60, 출력 1칸. 방식 B = h 입력 없이 출력 60칸(T+1 … T+60), 관측 없는 칸은 손실에서 가림. 학습 행 순서: 에폭마다 섞기 vs 예측일 T 시간 순서 고정.')
rows = []
for _, r in dl.iterrows():
    rows.append([r['구조'], r['방식'], r['학습 행 순서'], f4(r[SEL]), f4(r['겹1 구간 가중 PR-AUC']), f4(r['겹2 구간 가중 PR-AUC']), f4(r['겹3 구간 가중 PR-AUC']),
                 f4(r['60일 평균 PR-AUC']), f4(r['Recall≥0.80 에서의 Precision(60개 h 합친 행)']), c1(r['FP(h 당)']), pc(r['검사축소율']), pc(r['상위 10% 포착률']),
                 r['최고 에폭(겹별)'], c0(r['학습 시간(초, 겹 합)'])])
w(md_table(['구조', '방식', '학습 행 순서', '구간 가중 PR-AUC', '겹1', '겹2', '겹3', '60일 평균 PR-AUC', 'Recall 0.8 에서 Precision', 'FP(h당)', '검사축소율', '상위 10%', '최고 에폭(겹별)', '학습 시간(초)'], rows))
best_dl = dl.loc[dl[SEL].idxmax()]
w(f"- 최고 = {best_dl['구조']} {best_dl['방식']} {best_dl['학습 행 순서']} {f4(best_dl[SEL])}. 비교 기준: 1차 베이스라인 CatBoost(같은 기상 18변수 + 범주, 기본 설정) {f4(mb['wpr_auc'])}, 최종 CatBoost(변수 전부 21단위) 교차검증 {f4(cvm['wpr_auc'])}.")
for arch in ('BiLSTM', '트랜스포머'):
    for mode in dl['방식'].unique():
        a = dl[(dl['구조'] == arch) & (dl['방식'] == mode)]
        sh = float(a[a['학습 행 순서'] == '에폭마다 섞기'][SEL].iloc[0])
        tm = float(a[a['학습 행 순서'] == '시간 순서'][SEL].iloc[0])
        w(f"  - {arch} {mode}: 섞기 {sh:.4f} vs 시간 순서 {tm:.4f} (차이 {sh - tm:+.4f})")
w()

# ---------------------------------------------------------------- 그림
IMP2 = seld['importance_by_round'][1]['importance']
big4 = [IMP2[u] for u in ('n2', 'n0', 'hist_core', 'n1')]
SMALL4 = f"{min(big4):.4f}~{max(big4):.4f}"
s9 = [IMP2[u] for u in ('eco', 'gw', 'haccp', 'soil_ph', 'soil_om', 'soil_vldpha', 'food_aw', 'food_ph', 'food_na')]
SMALL9 = f"{min(s9):.4f} ~ {max(s9):.4f}"
assert seld['importance_by_round'][1]['회차'] == 2
w('## 그림 목록(PNG 150dpi, 맑은 고딕) — 스크립트 `' + SCRIPTS + '\\make_figs_II.py`(그림 II-7-1 의 교차검증 h 별 값은 `calc_folds_and_h.py` 결과)')
w()
FIGL = [
    ('II_2-1_겹구성_시간표.png', '그림 II-2-1. 시간순 교차검증 3겹과 시험(홀드아웃) 1년',
     '학습 구간을 한 해씩 늘리며 다음 해 예측일(07-01~05-01)을 맞혀 보고, 마지막 1년(2025-07-01~2026-05-01)은 맨 끝에 한 번만 채점함 — 학습 행 157,671→257,042, 맞혀 보는 쌍 160만~174만.'),
    ('II_3-1_바닐라_창길이별.png', '그림 II-3-1. 바닐라 70개 설정의 기상 창 길이별 고르기 점수',
     f"기상 요약 CatBoost 80일이 {f4(r_van[SEL])}로 1위, BiLSTM 은 창 40·90일에서 0.067 대, 일별 원값 입력은 세 알고리즘 모두 요약 입력 최고점보다 낮고 LightGBM 은 0.02~0.04 대."),
    ('II_3-2_기상변수_섞기중요도.png', '그림 II-3-2. 기상 변수 거르기 1회차 변수 섞기 중요도',
     f"이산화질소 섞기 중요도가 가장 크고, 일산화탄소(co)는 10번째로 빼 본 변수로 빼니 점수가 {f4(selwx['start_score'])}→{f4(selwx['final_score'])} 로 올라 유일하게 빠짐."),
    ('II_5-1_튜닝후보_산점도.png', '그림 II-5-1. CatBoost Optuna 30회 시도: 오탐 수와 고르기 점수',
     f"상위 5개 후보 중 시도 18 이 FP {c1(r18['FP(h 당)'])}로 가장 적고(고르기 점수 {f4(r18[SEL])}, 1위 시도 5 는 {f4(r5[SEL])}·FP {c1(r5['FP(h 당)'])}) 사용자가 시도 18 을 고름."),
    ('II_6-1_최종묶음_섞기중요도.png', '그림 II-6-1. 최종 입력 21단위의 묶음 섞기 중요도(변수 빼기 2회차)',
     f"그날 제품명 유형·그날 제품명 TF-IDF·과거 검출 이력 핵심·제품명 과거 유형 4단위가 {SMALL4} 로 크고, 기상 요약 {f4(IMP2['wx'])}, 외부 지표·토양 중앙값·식품 속성 9단위는 {SMALL9}."),
    ('II_7-1_최종_h별점수.png', '그림 II-7-1. 최종 모델의 h 별 점수 — 시험과 교차검증',
     f"시험 PR-AUC 는 H1~20 평균 {f4(bt['H1~20'])}, H21~40 {f4(bt['H21~40'])}, H41~60 {f4(bt['H41~60'])}로 h=60 을 빼고 모두 교차검증보다 낮고(교차검증 {f4(bc['H1~20'])}·{f4(bc['H21~40'])}·{f4(bc['H41~60'])}), "
     f"고정 기준값에서 시험 Recall 은 h 마다 {f4(calc['test_recall_range'][0])}~{f4(calc['test_recall_range'][1])}."),
    ('II_7-2_상위k_포착률.png', '그림 II-7-2. 최종 모델의 상위 k% 포착률(교차검증 vs 시험)',
     f"예측 확률 상위 10% 행이 시험 검출의 {pc(ter['상위 10% 포착률'])}, 상위 20% 가 {pc(ter['상위 20% 포착률'])}를 잡음(교차검증 {pc(cvr['상위 10% 포착률'])}·{pc(cvr['상위 20% 포착률'])})."),
    ('II_8-1_딥러닝8개.png', '그림 II-8-1. 딥러닝 8개 설정의 고르기 점수',
     f"트랜스포머 방식 A·섞기가 {f4(best_dl[SEL])}로 가장 높지만 최종 CatBoost 교차검증 {f4(cvm['wpr_auc'])} 보다 낮음(1차 베이스라인 CatBoost {f4(mb['wpr_auc'])} 보다는 높음). "
     "구조·방식 4조합 중 3개에서 섞기 판이 시간 순서 판보다 높고(트랜스포머 방식 B 만 시간 순서 판이 0.0007 높음), 트랜스포머 방식 B 는 0.047 대."),
]
FIG_SCRIPT = f'`{SCRIPTS}\\make_figs_II.py`'
rows = [[f'`{FIGS}\\{fn}`', t, s, FIG_SCRIPT + (f' (교차검증 h 별 값: `{SCRIPTS}\\calc_folds_and_h.py`)' if fn.startswith('II_7-1') else
                                                (f' (겹 수치: `{SCRIPTS}\\calc_folds_and_h.py`)' if fn.startswith('II_2-1') else ''))] for fn, t, s in FIGL]
w(md_table(['파일', '제목', '한 줄 해석', '스크립트'], rows))
w()
w('- 색: dataviz 기본 범주 팔레트(파랑 #2a78d6·주황 #eb6834·청록 #1baf7a·노랑 #eda100, 밝은 면 #fcfcfb). 팔레트 검증 스크립트는 이 PC 에 Node.js 가 없어 못 돌림 — 문서에 검증된 값·순서를 그대로 씀, 같은 그림 안 계열은 선 모양·표식·직접 이름표를 함께 씀.')
w()

# ---------------------------------------------------------------- 확인 필요
w('## 확인 필요')
w()
w('1. 시도 18 선택 사유 문구: 결과 파일에는 고른 시각(params_cat.json chosen_at)만 있고 사유가 없음. "같은 Recall 에서 오탐 가장 적음"은 구간 가중 PR-AUC 상위 5 후보 안에서만 맞음(30회 전체로는 시도 16·29 가 FP 더 적음, 5-3절).')
w('2. 생장 범위 자료의 출처 표기: 코드 문구 "리서치팀 예측인자 정리 ppt" vs 메모 기록(10-19 사용자 답 "식정원이 애초에 준 자료") — 보고서에 쓸 이름.')
w('3. 기상 변수 soil_t_nas(지중온도, NAS 자료)의 깊이: 원 점검표에 "깊이 설명 없음".')
w(f"4. 딥러닝 단계 BiLSTM 방식 A·섞기(80일, 18변수) {f4(dl.iloc[0][SEL])} 와 바닐라 BiLSTM 80일(19변수) {f4(vb[(vb['모델'] == 'BiLSTM') & (vb['기상 창(일)'] == 80)][SEL].iloc[0])} 의 차이 원인: 코드 설명상 다른 점은 기상 변수 수(co 유무)뿐이고 원인은 이 작업에서 따로 확인하지 않음.")
w('5. 식품 속성 표의 학습 기간(mnfctur_de ≤ 2024-06-30)은 3차 겹 1·2 학습 끝(2022·2023-06-30)보다 늦음 — 검출 정답은 쓰지 않은 표라는 기록만 있음, 보고서에 한계로 적을지.')
w('6. 서버에서 최종 모델 파일 저장(run3.py savefinal, 2026-10-22 06:45 시작)이 이 작업 중에 돌고 있었음 — 끝나면 final_cat/ 에 CatBoost 자체 중요도(변수중요도_묶음별.csv)가 생길 수 있으나 이 수치 모음에는 쓰지 않음(그림 II-6-1 은 묶음 섞기 중요도).')

out = os.path.join(W3, 'facts_results.md')
open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('저장', out, len(L), '줄')
