# -*- coding: utf-8 -*-
src = r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v67.md'
dst = r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v68.md'
text = open(src, 'rb').read().decode('utf-8')
crlf0 = text.count('\r\n'); lf0 = text.count('\n')

def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, (c, old)
    text = text.replace(old, new)

# 1. 단계 8 고른 설정 메모리 오류
rep(r"하고 다음 시도로 — float32·디스크 기반·m 은 바꾸지 않음",
    r"하고 다음 시도로 — float32·디스크 기반·m 은 바꾸지 않음. 고른 설정이 시도 밖 학습(8-T refit·딥러닝 칼럼 다시 붙인 판·Focal 판·④ 의 OOF·메타 다시 만들기)에서 메모리 오류면 그 시도를 실패로 바꾸고 CV 다음 순위 시도로, 남은 시도가 없으면 기본 유지(Focal 이면 Focal 판 없음으로 기록)")

# 2. AutoKeras 시도 메모리 오류 (A-7 딥러닝 줄 끝)
rep(r"고른 설정 다시 학습(단독 = 3 fold + refit 4회, 값 모델 = 전역·엄격 OOF 전부)은 §6 2차 공통 규칙)" + "\r\n",
    r"고른 설정 다시 학습(단독 = 3 fold + refit 4회, 값 모델 = 전역·엄격 OOF 전부)은 §6 2차 공통 규칙), 시도가 메모리(GPU) 오류면 실패로 기록하고 계속(`max_consecutive_failed_trials` = 19), 성공한 시도 + 기본 설정 중에서 고름, 시도가 전부 실패면 그 모델은 기본 유지로 기록" + "\r\n")

# 3. ⑥·⑦-D 해외 구조적 결측 칼럼
rep(r"  - ⑥·⑦-D = x_seq(2차 딥러닝, 기상·차감·SPI 채널 전부) + g 만.",
    r"  - ⑥·⑦-D = x_seq(2차 딥러닝, 기상·차감·SPI 채널 전부) + g·해외 구조적 결측 1칼럼(§4-2).")
rep(r"sigmoid(⑦-D 는 x_static = g 만, x_cat 없음) |",
    r"sigmoid(⑦-D 는 x_static = g·해외 구조적 결측 1칼럼, x_cat 없음) |")
rep(r"Dense 1(sigmoid)(⑥ 는 x_static = g 만, x_cat 없음) |",
    r"Dense 1(sigmoid)(⑥ 는 x_static = g·해외 구조적 결측 1칼럼, x_cat 없음) |")
rep(r"입력 = 기상 시퀀스 + g 만(지역·범주·계절 없음)",
    r"입력 = 기상 시퀀스 + g·해외 구조적 결측 1칼럼(지역·범주·계절 없음)")
rep(r"입력 = ⑥ 과 같음(기상 시퀀스 + g)",
    r"입력 = ⑥ 과 같음(기상 시퀀스 + g·해외 구조적 결측 1칼럼)")
rep(r"- ⑥·⑦ 의 '시계열만' 입력에도 결측 표시 채널(해외 구조적 결측 포함)·NaN 패턴으로 국내/해외 정보가 들어감",
    r"- ⑥·⑦ 의 '시계열만' 입력에도 결측 표시 채널·해외 구조적 결측 칼럼(⑥·⑦-D)·NaN 패턴(⑦-T)으로 국내/해외 정보가 들어감")

# 4. A-9 미포함 T3D3
rep(r"+12(딥러닝 확률 칼럼 3개·⑥ 3종)), AutoKeras 대상과",
    r"+12(딥러닝 확률 칼럼 3개·⑥ 3종)), (d) 방식 트리 앙상블(T3D3)이 단계 8 대상이면 딥러닝 튜닝 = 상위 3개 + ⑥ 3종 6 모델 → +111, AutoKeras 대상과")

# 5. A-12 단계 8 점수 씨앗
rep(r"대상이 씨앗 평균 판이면 시도는 씨앗 0, 채택 직전",
    r"대상이 씨앗 평균 판이면 시도는 씨앗 0 자료(뽑기)로(AutoKeras 씨앗은 부록 A-7), 채택 직전")

# 6. §8 일정
rep(r"단계 8 시도는 Optuna 1회 측정값을 대상 판 행 수 비례로(4-a 면 h 1~60 복제 행 수)",
    r"단계 8 시도는 Optuna 1회 측정값을 대상 판 행 수 비례로(4-a 면 h 1~60 복제 행 수, 최대 반복을 늘린 알고리즘은 반복 수 비례도)")

# 7. A-15 구성 약어
rep(r"구성 약어 = 튜닝하는 구성 모델의 구조 약어 — ",
    r"구성 약어 = 튜닝하는 구성 모델의 구조 약어(⑥ 구성은 x6-LSTMA 처럼 추가 판 이름) — ")

# 8. 0-2 판단용 검출 집계
rep(r"판단용 검출 집계 = fold 1 train(d ≤ 2022-06-30) 만",
    r"판단용 검출 집계 = d ≤ 2021-06-30 시료만")

# 9. B-6 확인만
rep(r"d ≤ 2025-06-30 은 확인만 — 그 결과로 목록을 바꾸지 않고 기록만",
    r"d ≤ 2025-06-30 은 확인만 — 그 결과로 목록을 바꾸지 않고 기록만(판단 자료로 안 쓰고 최종 보고 전까지 담당자에게 보여 주지 않음)")

# 10. 요약 14
rep(r"딥러닝 3종(⑥ — LSTM·GRU + attention·TCN)",
    r"딥러닝 3종(⑥ — LSTM + attention·GRU + attention·TCN)")

# 11. A-11 표시
rep(r"- 확률 보정(calibration)은 안 함 — 지표가 순위 기반 PR-AUC·threshold 규칙(§1-2)" + "\r\n",
    r"- 확률 보정(calibration)은 안 함 — 지표가 순위 기반 PR-AUC·threshold 규칙(§1-2)(담당자 결정)" + "\r\n")
rep(r"딥러닝 튜닝은 손실 이진 교차엔트로피 고정(부록 A-7)" + "\r\n",
    r"딥러닝 튜닝은 손실 이진 교차엔트로피 고정(부록 A-7)(담당자 결정)" + "\r\n")
rep(r"두 예측을 메타 모델로 합치는 판은 ④ 뿐 — 고도화 계획서 방식 C 와 다름",
    r"정형·시계열 두 예측을 메타 모델로 합치는 판은 없음(④ 는 LIMS 트리 6종의 메타) — 고도화 계획서 방식 C 와 다름")
rep(r"(담당자 결정 — 11-02 통화 예시의 'd 전날까지' 와 다름)",
    r"(담당자 결정 — 11-02 통화 예시(H5 = 11-07 에 11-02 ~ 11-06 예보)와 다름)")

assert text.count('\r\n') == crlf0 and text.count('\n') == lf0
open(dst, 'wb').write(text.encode('utf-8'))
print('ok', crlf0)
