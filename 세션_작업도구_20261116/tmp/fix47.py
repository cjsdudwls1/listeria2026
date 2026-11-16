import re
SRC = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v46.md"
DST = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v47.md"
raw = open(SRC, 'rb').read().decode('utf-8')
assert '\r\n' in raw
t = raw.replace('\r\n', '\n')
lines = t.split('\n')


def find_line(pred, desc):
    idx = [i for i, l in enumerate(lines) if pred(l)]
    assert len(idx) == 1, (desc, idx)
    return idx[0]


def rep(old, new, desc, count=1):
    global lines
    s = '\n'.join(lines)
    assert s.count(old) == count, (desc, s.count(old))
    s = s.replace(old, new)
    lines = s.split('\n')


def append_end(pred, add, desc):
    i = find_line(pred, desc)
    lines[i] = lines[i].rstrip() + add


def insert_after(pred, newline, desc):
    i = find_line(pred, desc)
    lines.insert(i + 1, newline)


# 1. §3 학습률
rep("(단계 8 의 시도와 8-T·Focal 판은 Optuna 가 뽑은 learning_rate 그대로 — 첫 시도와 기본 유지로 끝난 판은 대상 판의 학습률 그대로)",
    "(단계 8 의 시도·8-T 판과 8-T 채택 뒤의 Focal 판은 Optuna 가 뽑은 learning_rate 그대로 — 첫 시도·기본 유지로 끝난 판·8-T 미채택 때의 Focal 판은 대상 판의 학습률 그대로)", '1')

# 2. ④ 단계 8
append_end(lambda l: l.startswith('- 단계 8 점수(§6 단계 8):'),
           ". ④ 가 대상이면 시도 고르기 = 기본 트리마다 그 트리 단독 fold validation CV(반복 수 = 라이브러리 기본값 그대로 — ④ 규칙, §3 교차 N 안 씀), 판 점수 = 고른 설정으로 OOF 블록(§7-3)·메타 로지스틱을 다시 만든 ④ CV, Focal 도 ④ 의 CatBoost 에 같은 방식(OOF 다시)", '2a')
rep("시도 점수·최종 비교 모두 §3 교차 N 규칙.", "시도 점수·최종 비교 모두 §3 교차 N 규칙(④ 는 부록 A-12).", '2b')
rep("④ 면 기본 트리 6종마다(× 6, OOF 블록 다시 — §7-3)",
    "④ 면 기본 트리 6종마다 시도(× 6) + 고른 설정으로 OOF 블록·메타 다시(6 × 19 + 4, Focal 도 OOF 19 + 메타 4)", '2c')

# 3.
rep("설정 부록 A-7): 대상 = 대상의 딥러닝 —", "설정 부록 A-7): 대상의 딥러닝 —", '3')

# 4. 요약
rep("다운사이징 4-a 판과 그 위에 쌓은 판만", "다운사이징 4-a 판·미리보기와 그 위에 쌓은 판만", '4-2')
rep("이력 k·차감 길이(트리·LSTM 따로)를 고르고", "이력 k(트리 A 로)·차감 길이(트리·LSTM 따로)를 고르고", '4-11')
i13 = find_line(lambda l: l.startswith('13. 3차 방식'), '4-13')
lines[i13] = "13. 3차 방식(14~20 을 차례로): 단계마다 후보를 만들어 val 이 현재 최고 판보다 오르면 채택해 새 현재 최고 판으로 삼고, 안 오르면 버린다(test 는 기록만). 후보는 트리 대상 판에 기법을 더한 판, 3-0 상위 트리를 현재 입력으로 다시 학습해 묶은 판(스태킹·앙상블), 따로 만든 교체 후보(⑥·⑦·②·④·결합 모델 — 이기면 현재 최고 판이 되지만 그 위에 쌓지 않음)다. 트리 대상 판은 채택된 판 가운데 교체 후보가 아닌 트리·스태킹·앙상블 판 중 가장 최근 것이고(앙상블이면 구성 트리마다), 단계 8 튜닝만은 현재 최고 판에 한다. 한 단계 안에서는 쌓는 후보를 먼저, 교체 후보를 나중에 판정한다."

# 5. 표 3-2 가운데 칸
rep("| 3-2(④ 같이) | 3-0 트리 상위 3개를 현재 입력(채택한 칼럼 포함)으로 다시 학습한 트리마다 (a)~(e) + ④(교체 후보) — 같은 판이 되면 학습 없이 다시 씀(예: 3-1 채택 뒤의 (a)·(b), 3-1 판과 같은 (c)) | 현재 입력 |",
    "| 3-2(④ 같이) | 3-0 트리 상위 3개를 현재 입력(채택한 칼럼 포함)으로 다시 학습한 트리마다 (a)~(e) + ④(교체 후보) | 현재 입력 |", '5')

# 6. A-5 새 줄 (빼기 단위 줄 뒤)
insert_after(lambda l: l.startswith('- 빼기 단위: 원 변수 하나의 window 칼럼 묶음'),
             "- 결측 표시 칼럼(로지스틱·딥러닝)은 원 칼럼의 단위에 묶여 같이 섞고 뺌", '6')

# 7. A-12 동점 많은 판
rep("모든 판에서 세 fold validation 예측을 합쳐", "모든 판에서 세 fold validation 예측(3차 부스팅은 CV 점수와 같은 교차 N 예측)을 합쳐", '7a')
rep("1·2차 고르기에선 동점 많은 판이 CV 1위와 0.001 안일 때(1위면 2위와) 같게 하고",
    "1·2차 고르기에서도 동점 많은 판이 CV 1위와 0.001 안이면(1위면 2위와) u_tie 를 바꿔 다시 채점해 기록(규칙은 그대로)하고", '7b')

# 8. 설정 덤프
rep("모든 판(튜닝 시도 포함)의 실제 적용값을 `get_all_params()` 로 기록",
    "모든 판(튜닝 시도 포함)의 실제 적용값 기록 — CatBoost `get_all_params()`·XGBoost `save_config()`·LightGBM `params`·sklearn `get_params()`·딥러닝(AutoKeras 포함) = 구조·학습 설정 json", '8a')
rep("설정 덤프(`get_all_params()`)", "설정 덤프(B-4)", '8b')
assert '\n'.join(lines).count('get_all_params()') == 1

# 9. A-9 0-9 예상 줄 끝
append_end(lambda l: l.startswith('- 0-9 예상 = 1,484'),
           " — 미포함을 뺀 값이라 흔히 붙는 미포함(3-3 트리 8·Focal 비교 4·8-TD 4·딥러닝 칼럼 +6)만큼 실제는 조금 더 많을 수 있음", '9')

# 10. A-15
append_end(lambda l: l.startswith('- 저장소 = `리스테리아ML2\\exp\\mlflow\\mlflow.db`'),
           ". 기록은 모두 서버의 `mlflow server` 를 거침(1-1·1-2 를 나란히 돌릴 때 SQLite 동시 쓰기를 피함)", '10a')
rep("자식 run: 2-3·3-0b = 바퀴마다 1개", "자식 run: 2-3·3-0b = 바퀴마다 1개(부모 = 그 단계 끝 판 run)", '10b')
rep("·트리 대상 판(3차, 판 이름)·", "·바탕 트리 대상 판(3차 — 그 판을 만들 때 바탕이 된 트리 대상 판 이름)·", '10c')
rep("·현재 입력 묶음 id·코드 검수 완료 여부", "·코드 검수 완료 여부", '10d')
rep("변수 묶음 id(칼럼 목록은 artifact)", "입력 묶음 id(그 판 입력 칼럼 묶음 — 칼럼 목록은 artifact)·k·N_ma", '10e')

# 11. B-9
i = find_line(lambda l: l.startswith('- 같은 3 fold 로 여러 번 고름 → CV 가 높게 나오는 쪽:'), '11a')
lines[i] = "- 같은 3 fold 로 여러 번 고르고 기법마다 채택을 정함 → CV 가 높게 나오는 쪽: 채택 수·'같은 validation 으로 고른 횟수'(§1-3)·채택한 판의 fold 1 단독 상승 여부(§6 최종 모델)·단계별 test(§1-3)"
j = find_line(lambda l: l.startswith('- 같은 3 fold val 로 기법마다 채택을 정해'), '11b')
del lines[j]
k = find_line(lambda l: l.startswith('- Focal 비교는 Logloss 로 고른 설정'), '11c')
lines[k] = "- Focal 비교는 그 앞까지 채택한 판 설정(Logloss 기준, 8-T 채택 때 scale_pos_weight 포함)에 손실만 바꿔 Focal 에 불리할 수 있음(§6 단계 8)"
# AutoKeras 줄: Focal 줄(단계 8) 뒤
lines.insert(k + 1, "- 단계 8 AutoKeras 후보 19 개는 Keras, 기본 설정은 torch 모델이라 '기본 유지냐' 판정에 구조·학습률 말고 프레임워크 차이도 섞임(부록 A-7)")
# 4-a 줄: (3차) refit·4-a 줄 뒤
insert_after(lambda l: l.startswith('- (3차) refit·4-a 는 train 행이 늘어도'),
             "- 4-a 는 시료를 h 60개로 복제해 같은 정답 행이 60배 — 행 수·헤시안 기준 정규화(min_child_samples·min_child_weight·l2_leaf_reg 등)가 약해져 4-a 대 4-a0 차이에 이 효과가 섞임(§6 단계 4)", '11d')
# 해외 기상 줄: 1차 해외 기상 줄 뒤
insert_after(lambda l: l.startswith('- 1차 해외 기상은 DB 값만이라'),
             "- 해외 기상은 1차 = DB 값만(2015-01-01 전 NaN), 2차부터 2014-04-07 ~ 12-31 을 Open-Meteo 로 채움 — 2015 년 초 d 해외 행의 window·차감 앞쪽 날 처리가 국내(2014 년에 닿으면 RR8 로 제외)·1차와 다름(§4-4)", '11e')

# 요약 번호 확인
s0 = lines.index('## 요약(순서대로 읽기)')
s1 = lines.index('## 0. 고정 조건')
nums = [int(re.match(r'(\d+)\. ', l).group(1)) for l in lines[s0 + 1:s1] if re.match(r'\d+\. ', l)]
assert nums == list(range(1, 27)), nums
assert all((l.strip() == '' or re.match(r'\d+\. ', l)) for l in lines[s0 + 1:s1])
a = [i for i, l in enumerate(lines) if l.startswith('## 부록 A')][0]
print('body lines', a - s1)

open(DST, 'wb').write('\r\n'.join(lines).encode('utf-8'))
print('ok')
