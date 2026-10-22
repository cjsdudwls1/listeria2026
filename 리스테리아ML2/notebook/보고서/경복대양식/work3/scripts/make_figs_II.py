# -*- coding: utf-8 -*-
"""리스테리아 3차 보고서 II.2~II.8 그림 -> notebook/보고서/경복대양식/figs3/II_*.png

rev1(2026-10-22, [종합] 수정 5): 모든 그림을 본문 폭(161mm = 6.35인치)보다 좁은 폭 6.3인치로 그리고 글자를 9.5pt 이상으로 —
  조립기는 그림을 줄이지 않고 넣으므로(폭이 본문보다 넓을 때만 줄임) 인쇄 글자도 9.5pt(빈 자리 메우기로 95% 까지 줄여도 9pt).
  그림 안 제목은 빼고(그림 아래 '[그림 n] …' 제목이 대신함) 200dpi 로 저장, bbox_inches 는 쓰지 않음(그림 크기 = figsize 그대로).
  예전 임시 조치(PNG 해상도 표시만 바꾼 set_png_dpi.py)는 쓰지 않음 — 이 스크립트를 다시 돌리면 해상도 표시도 200dpi 로 돌아감.
  새 그림 II_7-3: 최종 모델의 CatBoost 내장 중요도(PredictionValuesChange) 묶음별 합(results/final_cat/변수중요도_묶음별.csv).
rev2(2026-10-22, 사용자 지시 — 억지 번역 대신 일반 용어): 본문 그림(II_2-1·II_3-1·II_7-1·II_7-2·II_7-3)의 그림 안 글자만 바꿈.
  학습 → train, 시험 → test, 교차검증 맞혀 보기 → 교차검증 validation, 겹 → fold(세 겹 → 3개 fold), 창 → window, 칸 → 컬럼
  (9칸 → 9개 컬럼, 범주 1칸 → 범주형 1개 컬럼). 자료·숫자·색·크기·dpi·파일 이름은 그대로. 읽는 표의 칸 이름(SEL 등)도 그대로.
  본문에 안 쓰는 그림(II_3-2·II_5-1·II_6-1·II_8-1)의 글자는 바꾸지 않음.

읽는 것(읽기 전용):
  exp/3_시간순교차검증/tables/*.csv (utf-8-sig) - 바닐라_비교, 기상거르기_변수중요도, 기상거르기, 튜닝후보_cat, 묶음빼기_cat,
    최종_시험_cat, 딥러닝단계
  exp/3_시간순교차검증/results/final_cat/변수중요도_묶음별.csv
  work3/calc/겹구성_계산.csv, work3/calc/최종_h별_교차검증_시험.csv (calc_folds_and_h.py 결과)
  data/중간/시군구_기상_점검.csv ('변수' 칸 = 기상 변수 한글 이름)
본문에 쓰는 그림: II_2-1, II_3-1, II_7-1, II_7-2, II_7-3. 나머지(II_3-2, II_5-1, II_6-1, II_8-1)는 같은 규칙으로 다시 그려 두기만 함.
색: dataviz 기본 범주 팔레트(밝은 면 #fcfcfb 기준 검증된 순서) 앞 슬롯부터 고정 순서로(이 PC 에 Node.js 가 없어 검사 스크립트는 못 돌림).
"""
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE = 'C:/Users/NT751/Desktop/천영진/리스테리아ML2'
TAB = os.path.join(BASE, 'exp', '3_시간순교차검증', 'tables')
RES = os.path.join(BASE, 'exp', '3_시간순교차검증', 'results')
W3 = os.path.join(BASE, 'notebook', '보고서', '경복대양식', 'work3')
CALC = os.path.join(W3, 'calc')
FIG = os.path.join(BASE, 'notebook', '보고서', '경복대양식', 'figs3')
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- 스타일(팔레트·잉크·글자)
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'      # 범주 슬롯 1~4(파랑·주황·청록·노랑)
SURF = '#fcfcfb'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#6f6e69'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
GRAY_MARK = '#b9b7ae'
FS = 9.5          # 모든 글자의 최소 크기(pt) = 인쇄 크기
W = 6.3           # 그림 폭(인치)
DPI = 200

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': FS,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': FS + 0.5, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'axes.labelsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS,
    'legend.fontsize': FS, 'legend.frameon': False, 'lines.linewidth': 1.8, 'axes.axisbelow': True,
})


def tab(name):
    return pd.read_csv(os.path.join(TAB, name), encoding='utf-8-sig')


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print('저장', p)
    return p


SEL = '구간 가중 PR-AUC(세 겹 평균)'
OUT = []

# ---------------------------------------------------------------- II_2-1 겹 구성 시간표
fd = pd.read_csv(os.path.join(CALC, '겹구성_계산.csv'), encoding='utf-8-sig')
start = pd.Timestamp('2015-04-07')
fig, ax = plt.subplots(figsize=(W, 2.55))
labels = []
for i, r in fd.iterrows():
    y = len(fd) - 1 - i
    te = pd.Timestamp(r['학습 끝(d)'])
    ts, tend = pd.Timestamp(r['맞혀 보는 T 시작']), pd.Timestamp(r['맞혀 보는 T 끝'])
    is_test = str(r['겹']) == '최종'
    ax.barh(y, (te - start).days, left=mdates.date2num(start), height=0.46, color=S1, edgecolor=SURF, linewidth=1.2)
    ax.barh(y, (tend - ts).days + 1, left=mdates.date2num(ts), height=0.46, color=S3 if is_test else S2, edgecolor=SURF, linewidth=1.2)
    ax.text(mdates.date2num(start) + 30, y, f"train {int(r['학습 행(단위x날짜)']):,}행 · 검출 {int(r['학습 행 검출']):,}",
            va='center', ha='left', fontsize=FS, color='white')
    ax.text(mdates.date2num(tend) + 2, y - 0.43,
            f"{'test' if is_test else 'validation'} {int(r['(행,h) 쌍']):,}쌍 · 검출 {int(r['(행,h) 쌍 검출']):,}",
            va='center', ha='right', fontsize=FS, color=INK2)
    labels.append('최종·test' if is_test else f"fold {r['겹']}")
ax.set_yticks(range(len(fd)))
ax.set_yticklabels(labels[::-1], fontsize=FS, color=INK)
yrs = list(range(2015, 2027))
ax.set_xticks([mdates.date2num(pd.Timestamp(f'{y}-01-01')) for y in yrs])
ax.set_xticklabels([str(y) for y in yrs])
ax.set_xlim(mdates.date2num(pd.Timestamp('2015-01-01')), mdates.date2num(pd.Timestamp('2026-08-31')))
ax.set_ylim(-0.75, len(fd) - 0.45)
ax.grid(axis='y', visible=False)
ax.tick_params(axis='y', length=0)
ax.legend(handles=[Patch(color=S1, label='train(기준일 d)'), Patch(color=S2, label='교차검증 validation(T)'),
                   Patch(color=S3, label='test 1년(T)')],
          loc='lower left', bbox_to_anchor=(-0.01, 1.0), ncol=3, handlelength=1.1, columnspacing=1.2, borderaxespad=0.1)
fig.subplots_adjust(left=0.125, right=0.99, top=0.88, bottom=0.11)
OUT.append(save(fig, 'II_2-1_겹구성_시간표.png'))

# ---------------------------------------------------------------- II_3-1 바닐라 창 길이별 점수
vb = tab('바닐라_비교.csv')
col_alg = {'CatBoost': S1, 'XGBoost': S2, 'LightGBM': S3, 'BiLSTM': S4}
mk = {'CatBoost': 'o', 'XGBoost': 's', 'LightGBM': '^', 'BiLSTM': 'D'}
fig, axes = plt.subplots(1, 2, figsize=(W, 2.95), sharey=True)
panels = [('(가) 기상 요약(트리)·기상 시퀀스(BiLSTM)', [('CatBoost', '기상 요약'), ('XGBoost', '기상 요약'), ('LightGBM', '기상 요약'),
                                                     ('BiLSTM', '기상 시퀀스')]),
          ('(나) 기상 일별 원값(트리)', [('CatBoost', '일별 원값'), ('XGBoost', '일별 원값'), ('LightGBM', '일별 원값')])]
for ax, (ttl, sers) in zip(axes, panels):
    for m, inp in sers:
        d = vb[(vb['모델'] == m) & (vb['입력'] == inp)].sort_values('기상 창(일)')
        ax.plot(d['기상 창(일)'], d[SEL], color=col_alg[m], marker=mk[m], markersize=4.5, markeredgecolor=SURF,
                markeredgewidth=0.8, lw=1.6, label=m)
    ax.set_title(ttl, loc='left', fontsize=FS, pad=4)
    ax.set_xticks([10, 40, 70, 100])
    ax.set_xlim(5, 105)
    ax.set_xlabel('기상 window 길이(일)')
    ax.grid(axis='x', visible=False)
best = vb.iloc[vb[SEL].idxmax()]
axes[0].scatter([best['기상 창(일)']], [best[SEL]], s=110, facecolor='none', edgecolor=INK, linewidth=1.2, zorder=5)
axes[0].annotate(f"1위 {best[SEL]:.4f}", (best['기상 창(일)'], best[SEL]), xytext=(-62, 2), textcoords='offset points',
                 fontsize=FS, color=INK, va='center')
axes[0].set_ylabel('구간 가중 PR-AUC')
axes[0].set_ylim(0.02, 0.08)
axes[0].set_yticks([0.02, 0.04, 0.06, 0.08])
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='upper left', bbox_to_anchor=(0.10, 1.0), ncol=4, handlelength=1.6, columnspacing=1.2)
fig.subplots_adjust(left=0.115, right=0.99, top=0.80, bottom=0.15, wspace=0.08)
OUT.append(save(fig, 'II_3-1_바닐라_창길이별.png'))

# ---------------------------------------------------------------- II_3-2 기상 변수 섞기 중요도(1회차) — 본문 미사용
wn = pd.read_csv(os.path.join(BASE, 'data', '중간', '시군구_기상_점검.csv'), encoding='utf-8-sig')
kname = dict(zip(wn['칼럼'], wn['변수']))
kname['soil_t_nas'] = '지중온도(NAS)'
wi = tab('기상거르기_변수중요도.csv')
w1 = wi[wi['회차'] == 1].sort_values('변수 섞기 중요도(세 겹 평균)')
fig, ax = plt.subplots(figsize=(W, 4.4))
cols = [S2 if v == 'co' else S1 for v in w1['변수']]
yy = np.arange(len(w1))
ax.barh(yy, w1['변수 섞기 중요도(세 겹 평균)'], color=cols, height=0.66, edgecolor=SURF, linewidth=1.0)
ax.set_yticks(yy)
ax.set_yticklabels([f"{kname.get(v, v)}({v})" for v in w1['변수']], fontsize=FS)
ax.axvline(0, color=AXIS, lw=1.0)
ax.set_xlabel('섞기 중요도(구간 가중 PR-AUC 감소, 세 겹 평균)')
ax.grid(axis='y', visible=False)
ax.legend(handles=[Patch(color=S1, label='남긴 변수'), Patch(color=S2, label='뺀 변수')], loc='lower right')
fig.subplots_adjust(left=0.40, right=0.98, top=0.98, bottom=0.11)
OUT.append(save(fig, 'II_3-2_기상변수_섞기중요도.png'))

# ---------------------------------------------------------------- II_5-1 튜닝 후보 산점도 — 본문 미사용
tc = tab('튜닝후보_cat.csv')
top5 = tc['구간 가중 PR-AUC 상위 5'].astype(str).eq('예')
fig, ax = plt.subplots(figsize=(W, 3.4))
ax.scatter(tc.loc[~top5, 'FP(h 당)'], tc.loc[~top5, SEL], s=36, color=GRAY_MARK, edgecolor=SURF, linewidth=1.0, label='나머지 25개 후보', zorder=3)
ax.scatter(tc.loc[top5, 'FP(h 당)'], tc.loc[top5, SEL], s=48, color=S1, edgecolor=SURF, linewidth=1.0, label='고르기 점수 상위 5개', zorder=4)
r18 = tc[tc['시도'] == 18].iloc[0]
ax.scatter([r18['FP(h 당)']], [r18[SEL]], s=170, facecolor='none', edgecolor=S2, linewidth=1.8, zorder=5, label='고른 후보(18번)')
for _, r in tc[top5].iterrows():
    ax.annotate(f"{int(r['시도'])}번", (r['FP(h 당)'], r[SEL]), xytext=(6, 3), textcoords='offset points', fontsize=FS, color=INK)
ax.set_xlabel('FP(h 하나당 평균 건수, Recall 0.80 판정 기준값)')
ax.set_ylabel('구간 가중 PR-AUC')
ax.legend(loc='lower right')
fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.15)
OUT.append(save(fig, 'II_5-1_튜닝후보_산점도.png'))

# ---------------------------------------------------------------- II_6-1 최종 묶음 섞기 중요도(묶음빼기 2회차) — 본문 미사용(부록 표로)
dr = tab('묶음빼기_cat.csv')
SHORT = {
    '식품 수분활성도 aw(aw 구간 범주 포함)': '식품 수분활성도(표기 범주 포함)',
    '이상치 점수(AE·VAE 4개, 겹마다 재학습)': '이상치 점수(AE·VAE)',
    '과거 검출 이력(5년·1년 스무딩 검출률, 5년 로그 검사 수)': '과거 검출 이력',
}
r2 = dr[dr['회차'] == 2].copy().sort_values('단위 섞기 중요도')
fig, ax = plt.subplots(figsize=(W, 4.6))
yy = np.arange(len(r2))
ax.barh(yy, r2['단위 섞기 중요도'], color=S1, height=0.66, edgecolor=SURF, linewidth=1.0)
ax.set_yticks(yy)
ax.set_yticklabels([SHORT.get(n, n) for n in r2['시도한 단위']], fontsize=FS)
ax.axvline(0, color=AXIS, lw=1.0)
ax.set_xlabel('섞기 중요도(구간 가중 PR-AUC 감소, 세 겹 평균)')
ax.grid(axis='y', visible=False)
fig.subplots_adjust(left=0.42, right=0.98, top=0.98, bottom=0.10)
OUT.append(save(fig, 'II_6-1_최종묶음_섞기중요도.png'))

# ---------------------------------------------------------------- II_7-1 최종 모델 h 별 점수
hb = pd.read_csv(os.path.join(CALC, '최종_h별_교차검증_시험.csv'), encoding='utf-8-sig').sort_values('h')
fig, axes = plt.subplots(3, 1, figsize=(W, 3.9), sharex=True)
specs = [('pr_auc', 'PR-AUC'), ('recall', 'Recall'), ('precision', 'Precision')]
for ax, (k, lab) in zip(axes, specs):
    for a, b in ((0.5, 20.5), (40.5, 60.5)):
        ax.axvspan(a, b, color='#f0efec', zorder=0, lw=0)
    ax.plot(hb['h'], hb[f'{k}_test'], color=S1, label='test 1년(최종 모델)', lw=1.6)
    ax.plot(hb['h'], hb[f'{k}_cv'], color=S2, label='교차검증(3개 fold 평균)', lw=1.6)
    ax.set_ylabel(lab)
    ax.grid(axis='x', visible=False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
axes[0].set_ylim(0.06, 0.114)
ymax = axes[0].get_ylim()[1]
for (a, b), w in zip(((1, 20), (21, 40), (41, 60)), (0.5, 0.3, 0.2)):
    axes[0].text((a + b) / 2, ymax * 0.995, f'h {a}~{b}(가중치 {w})', ha='center', va='top', fontsize=FS, color=INK2)
axes[1].axhline(0.8, color=MUTED, lw=0.9, ls=(0, (3, 2)))
axes[2].set_xlabel('h(며칠 뒤를 맞히는지)')
axes[2].set_xticks([1, 10, 20, 30, 40, 50, 60])
axes[2].set_xlim(0.5, 60.5)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='upper left', bbox_to_anchor=(0.11, 1.0), ncol=2, handlelength=1.8)
fig.align_ylabels(axes)
fig.subplots_adjust(left=0.13, right=0.99, top=0.925, bottom=0.11, hspace=0.12)
OUT.append(save(fig, 'II_7-1_최종_h별점수.png'))

# ---------------------------------------------------------------- II_7-2 상위 k% 포착률
ft = tab('최종_시험_cat.csv')
cvr, ter = ft.iloc[0], ft.iloc[1]
ks = ['상위 1% 포착률', '상위 5% 포착률', '상위 10% 포착률', '상위 20% 포착률']
x = np.arange(len(ks))
wdt = 0.36
fig, ax = plt.subplots(figsize=(W, 2.35))
b1 = ax.bar(x - wdt / 2 - 0.01, [cvr[k] * 100 for k in ks], width=wdt, color=S2, label='교차검증(3개 fold 평균)', edgecolor=SURF, linewidth=1.0)
b2 = ax.bar(x + wdt / 2 + 0.01, [ter[k] * 100 for k in ks], width=wdt, color=S1, label='test 1년', edgecolor=SURF, linewidth=1.0)
for bars in (b1, b2):
    for bb in bars:
        ax.annotate(f'{bb.get_height():.1f}', (bb.get_x() + bb.get_width() / 2, bb.get_height()), xytext=(0, 2), textcoords='offset points',
                    ha='center', fontsize=FS, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(['확률 상위 1%', '상위 5%', '상위 10%', '상위 20%'])
ax.set_ylabel('포착률(%)')
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis='x', visible=False)
ax.legend(loc='upper left', ncol=2, handlelength=1.2)
fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.12)
OUT.append(save(fig, 'II_7-2_상위k_포착률.png'))

# ---------------------------------------------------------------- II_7-3 최종 모델 CatBoost 내장 중요도(묶음별 합)
im = pd.read_csv(os.path.join(RES, 'final_cat', '변수중요도_묶음별.csv'), encoding='utf-8-sig')
assert len(im) == 26 and abs(im['비중 %'].sum() - 100) < 1e-6, (len(im), im['비중 %'].sum())
NAME = {
    'hist_core': '과거 검출 이력(9개 컬럼)', 'n1': '제품명 과거 유형(92개 컬럼)', 'n2': '그날 제품명 유형(23개 컬럼, 운영 전제)',
    'n0': '그날 제품명 TF-IDF(500개 컬럼, 운영 전제)', 'wx': '기상 요약(234개 컬럼)', 'g123': '생장식 지수(16개 컬럼)',
    'wxanom': '이동평균 차감 기상(57개 컬럼)', 'sd1': '1년 검출률 불확실성(3개 컬럼)', 'spi': '표준강수지수(26개 컬럼)',
    'logn1': '1년 로그 검사 수(3개 컬럼)', 'food_ph': '식품 pH(1개 컬럼)', 'rg': '생장 범위(19개 컬럼)', 'ae4': '이상치 점수(4개 컬럼)',
    'soil_vldpha': '토양 유효인산(1개 컬럼)', 'soil_om': '토양 유기물(1개 컬럼)', 'gw': '지하수 정기수질검사 적합률(1개 컬럼)',
    'eco': '친환경 인증 비율(1개 컬럼)', 'p3': '품목 3단계(범주형 1개 컬럼)', 'h': 'h(1개 컬럼)', 'sido': '시도(범주형 1개 컬럼)',
    'sgg': '시군구(범주형 1개 컬럼)', 'p2': '품목 2단계(범주형 1개 컬럼)', 'food_aw': '식품 수분활성도(숫자형 1개 + 범주형 1개 컬럼)',
    'food_na': '식품 염도(1개 컬럼)', 'soil_ph': '토양 pH(1개 컬럼)', 'haccp': 'HACCP 지정 비율(1개 컬럼)',
}
assert set(NAME) == set(im['묶음 ID']), set(im['묶음 ID']) ^ set(NAME)
ZERO_ORDER = ['h', 'sido', 'sgg', 'p2', 'food_aw', 'food_na', 'soil_ph', 'haccp']
im['zo'] = im['묶음 ID'].map({k: i for i, k in enumerate(ZERO_ORDER)}).fillna(-1)
im = im.sort_values(['비중 %', 'zo'], ascending=[False, True]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(W, 4.25))
yy = np.arange(len(im))[::-1]
ax.barh(yy, im['비중 %'], color=S1, height=0.62, edgecolor=SURF, linewidth=0.6, zorder=3)
for yi, v in zip(yy, im['비중 %']):
    txt = '0' if v == 0 else (f'{v:.1f}%' if v >= 1 else f'{v:.2f}%')
    ax.text(v + 0.4, yi, txt, va='center', ha='left', fontsize=FS, color=INK if v >= 1 else INK2)
ax.set_yticks(yy)
ax.set_yticklabels([NAME[k] for k in im['묶음 ID']], fontsize=FS)
ax.set_xlim(0, 37)
ax.set_xlabel('CatBoost 중요도 비중(%, 컬럼 1,001개 합 = 100)')
ax.grid(axis='y', visible=False)
ax.tick_params(axis='y', length=0)
ax.set_ylim(-0.6, len(im) - 0.4)
fig.subplots_adjust(left=0.475, right=0.99, top=0.995, bottom=0.11)
OUT.append(save(fig, 'II_7-3_최종_CatBoost중요도.png'))

# ---------------------------------------------------------------- II_8-1 딥러닝 8개 — 본문 미사용
dl = tab('딥러닝단계.csv')
base_wx18 = 0.07397412719334831        # 1차 베이스라인(CatBoost 기상 요약 80일, 기상 18변수, 기본 설정) - 기상거르기.csv CO 뺀 후보 점수
final_cv = float(ft.iloc[0][SEL])       # 최종 CatBoost 교차검증
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True)
for ax, arch in zip(axes, ['BiLSTM', '트랜스포머']):
    d = dl[dl['구조'] == arch]
    modes = list(dict.fromkeys(d['방식']))
    xx = np.arange(len(modes))
    for j, (order, col) in enumerate((('에폭마다 섞기', S1), ('시간 순서', S2))):
        vals = [float(d[(d['방식'] == m) & (d['학습 행 순서'] == order)][SEL].iloc[0]) for m in modes]
        ax.bar(xx + (j - 0.5) * 0.36, vals, width=0.34, color=col, label=order, edgecolor=SURF, linewidth=1.0)
    ax.axhline(final_cv, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.axhline(base_wx18, color=MUTED, lw=1.0, ls=(0, (1, 2)))
    ax.set_xticks(xx)
    ax.set_xticklabels(['방식 A', '방식 B'])
    ax.set_title(arch, loc='left', fontsize=FS)
    ax.grid(axis='x', visible=False)
axes[0].set_ylabel('구간 가중 PR-AUC')
axes[0].set_ylim(0, 0.105)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='upper left', bbox_to_anchor=(0.10, 1.0), ncol=2)
fig.subplots_adjust(left=0.12, right=0.99, top=0.84, bottom=0.10, wspace=0.08)
OUT.append(save(fig, 'II_8-1_딥러닝8개.png'))

print('\n'.join(OUT))
