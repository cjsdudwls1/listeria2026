import re
SRC = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v43.md"
DST = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v44.md"
raw = open(SRC, 'rb').read().decode('utf-8')
assert '\r\n' in raw
lines = raw.split('\r\n')


def find_line(prefix):
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx) == 1, (prefix, idx)
    return idx[0]


def set_line(prefix, new):
    i = find_line(prefix)
    lines[i] = new


def sub_once(old, new, prefix=None):
    hits = [i for i, l in enumerate(lines) if old in l and (prefix is None or l.startswith(prefix))]
    assert len(hits) == 1, (old, hits)
    i = hits[0]
    assert lines[i].count(old) == 1, old
    lines[i] = lines[i].replace(old, new)

# 1
set_line('- 교체 후보 = ', "- 교체 후보 = ⑥·⑦·②·④·3-4 — 채택되면 현재 최고 판(채택 판정 상대)만 바뀌고 그 위에 쌓지 않음(쌓는 단계는 트리 대상 판·현재 입력 그대로). 예외: 단계 8 의 8-TD 는 현재 최고 판이 교체 후보 딥러닝(⑥·⑦-D·②·3-4)이면 그 모델을 튜닝")
# 2
set_line('- 트리 대상 판 = ', "- 트리 대상 판 = 3차에서 채택된 판(현재 최고 판이 됐던 판) 가운데 교체 후보가 아닌 가장 최근의 트리·스태킹·앙상블 판(앙상블이면 구성 트리 모두 — 기법을 구성 트리마다 적용해 다시 평균, 딥러닝 구성은 그대로, 판정은 단순 평균(대조 기록은 부록 A-12)). 그런 판이 없으면(채택된 판이 딥러닝 단독·교체 후보뿐) 3-0b 묶음이 채택됐으면 3-0b 끝 판, 아니면 3-0 트리 6종 중 CV 1위. 채택 안 된 판은 트리 대상 판이 되지 않음. 트리 대상 판이 현재 최고 판이 아니면 그 판의 기법 전후 val 도 기록")
# 3a 표 단계 8 줄
i = find_line('| 단계 8 | ')
cells = lines[i].split(' | ')
assert len(cells) == 3, cells
cells[1] = "트리 대상 판에 8-T → Focal(8-T 튜닝 설정 위에), 현재 최고 판의 딥러닝에 8-TD(스태킹·앙상블이면 그 앞까지 채택한 판의 딥러닝 부분, 딥러닝 단독·결합 모델이면 교체 후보라도 그 모델)"
lines[i] = ' | '.join(cells)
# 3b
sub_once("이 순서로, 8-T·8-TD 는 그 앞까지 채택한 판 위에, Focal 은 8-T 튜닝 설정(기본 유지면 트리 대상 판 설정) 위에(8-T 채택 여부와 무관) 만들어",
         "이 순서로, 8-T 는 트리 대상 판 위에, Focal 은 8-T 튜닝 설정(기본 유지면 트리 대상 판 설정) 위에(8-T 채택 여부와 무관), 8-TD 는 3차 공통 표대로 만들어")
# 3c
sub_once("(CatBoost·지원될 때, 앙상블이면 구성 트리마다)", "(CatBoost·지원될 때, 앙상블이면 구성 트리 중 CatBoost 에만)")
sub_once("대상이 딥러닝 단독·결합 모델이면 튜닝한 설정의 그 모델", "대상이 딥러닝 단독·결합 모델(교체 후보 포함)이면 튜닝한 설정의 그 모델")
# 3d
sub_once("단계 8 대상이 앙상블이면 8-T·Focal 도 구성 트리마다(× 3)", "단계 8 대상이 앙상블이면 8-T 도 구성 트리마다(× 3, Focal 은 CatBoost 구성만)")
# 4
sub_once("7-a 는 모든 표의 비교 기준 줄로만(담당자 결정)", "7-a 는 모든 표의 기준 줄로만(담당자 결정)")

# 5 요약 13~21, 25
S = {
13: "13. 3차 방식(14~20 을 차례로): 단계마다 후보를 만들어 val 이 현재 최고 판보다 오르면 채택해 새 현재 최고 판으로 삼고, 안 오르면 버린다(test 는 기록만). 후보는 세 가지다 — 트리 대상 판(채택된 트리·스태킹 판 중 가장 최근 것, 앙상블이면 그 구성 트리마다)에 기법을 더한 판, 3-0 상위 트리를 현재 입력으로 다시 학습해 묶은 판(스태킹·앙상블), 따로 만든 교체 후보(⑥·⑦·②·④·결합 모델 — 이기면 현재 최고 판이 되지만 그 위에 쌓지 않음).",
14: "14. 3-0 알고리즘 넓히기: RF·ExtraTrees·로지스틱·1차 4 알고리즘·GRU·트랜스포머·LSTM + attention 10종을 1차 window·변수 전체로 비교한다(부스팅 반복 수 고르기 시작). 1위가 첫 현재 최고 판이다.",
15: "15. 3-0b 변수 다시 고르기: 1위 알고리즘으로 반복 수 고르기를 켠 채 변수 빼기를 다시 해 val 이 오르면 그 묶음을 채택한다.",
16: "16. ⑥·⑦ 시계열만: 기상 시퀀스만 넣은 딥러닝 2종(⑥)과, 같은 시계열 정보를 넣은 트리·트랜스포머(⑦)를 교체 후보로 본다.",
17: "17. 3-1~3-4 결합: 트리 대상 판에 AE·VAE 이상 점수 더하기(3-1), 상위 트리 스태킹과 LIMS 만 스태킹 ④(3-2), 앙상블(3-3), 결합 모델(3-4)을 차례로 본다.",
18: "18. 단계 4 다운사이징: Theil's U 1위 칼럼에서 검출 없는 값의 행을 빼고 4-a(h 1~60)·4-a0(h 1개)로 학습한다(원 예측 그대로 — SMOTENC 와, 학습 기간에 검출 없는 품목 0·검출만 있는 품목 1 로 점수를 덮는 판은 뺌). ① 당일 예측(h = 1 비교만)·② 60칸 출력(교체 후보)도 이때 본다.",
19: "19. 단계 5·6 손 교수 방식: 해마다 양성 전부와 음성 4배를 뽑는 2종이고, 씨앗 5번의 예측 평균을 쓴다.",
20: "20. 단계 8 튜닝·불균형(맨 끝): 트리 대상 판에 Optuna 100회(검출 행 가중치 포함)와 Focal 손실(CatBoost), 현재 최고 판의 딥러닝(딥러닝 단독·결합 모델이면 그 모델, 스태킹·앙상블이면 그 딥러닝 부분)에 AutoKeras 튜닝을 한다. val 이 오를 때만 채택한다.",
21: "21. 비교만(채택 대상 아님): ③ 시료 무작위 나누기(1차 뒤), 고위험 품목 모델(단계 4), 7-a 타겟 인코딩 단독(단계 7 — 학습 없이 과거 검출률(품목 3단계 × 지역, test 는 train+val 기간)을 그대로 확률로 쓰는 계산, 모든 표의 기준 줄로 두어 머신러닝 판이 이보다 나은지 본다).",
25: "25. 실험 관리: MLflow — 판마다 run, 차수·단계·채택 여부(1·2차는 고른 것)를 tag 로 남겨 로컬 MLflow 화면에서 확인한다.",
}
for n, txt in S.items():
    set_line(f"{n}. ", txt)
# 요약 번호 확인
s0 = find_line('## 요약')
s1 = find_line('## 0. 고정 조건')
nums = [int(m.group(1)) for l in lines[s0+1:s1] if (m := re.match(r'(\d+)\. ', l))]
assert nums == list(range(1, 27)), nums
assert all(l == '' or re.match(r'\d+\. ', l) for l in lines[s0+1:s1])

# 6
sub_once("— 2-1·2-1b·2-2 의 출발 판", "— 2-1·2-1b·2-2 트리 쪽의 출발 판")
# 7
sub_once("현재 최고 판이 앙상블일 때(§6 3차 공통 트리 대상 판): 단계 4~8 표", "현재 최고 판이 앙상블일 때(§6 3차 공통 트리 대상 판): 단계 4~6·8 표")
sub_once("현재 최고 판이 앙상블이면 단계 4~8 기법이", "현재 최고 판이 앙상블이면 단계 4~6·8 기법이")
sub_once("반복 수 56 시점은 안 셈)", "반복 수 56 시점은 안 셈 — 표 주석에 \"반복 수 고르기(판마다 56 시점)는 셈에서 빠짐\" 표시)")
# 8
sub_once("(1-1·1-2 의 같은 설정 refit 예측 그대로, 추가 학습 없음)", "(1-1·1-2 의 같은 설정 refit 예측 그대로, 추가 학습 없음, test threshold = 무작위 겹 validation 예측을 합쳐 §1-2 규칙(시간순 threshold 값도 같이 적음))", prefix='| ③ |')
i = find_line('- 1·2차는 라이브러리 기본 반복 수 — 3차 판과의 차이에')
lines.insert(i + 1, "- ④ 는 기본 반복 수(1·2차 규칙)라 3차 판과 반복 수 규칙이 달라 채택 비교에 그 차이가 섞임")
# 9
sub_once("서버에서 돌린 run 은 단계가 끝날 때마다 이 로컬 저장소로 옮김(run id 그대로)",
         "run 은 서버의 한 저장소에서 만들고 단계가 끝날 때마다 로컬로 통째로 옮김(옮기는 방법은 단계 0 에서 시험해 정함 — 예: 서버에서 `mlflow server --artifacts-destination` 로 artifact 주소를 `mlflow-artifacts:/` 로 두고 mlflow.db·artifacts 를 함께 복사), tag 는 run id 대신 판 이름으로 가리킴")
set_line('- 확인 = 로컬에서 `mlflow ui', "- 확인 = 로컬에서 `mlflow ui`(저장소·artifact 경로는 단계 0 에서 정한 방법대로) → 브라우저 [IP 가림]:5000")
sub_once("(③ = 1차, 미리보기 4-a = 3차)", "(③ = 1차, 미리보기 4-a = 3차, 7-a = 준비)")
set_line('- tag: 차수·단계·판 이름', "- tag: 차수·단계·판 이름·역할(후보/교체 후보/비교만/기준 줄)·판정(채택/버림 — 판정 때 한 번 쓰고 안 고침)·판정 상대(판정 때의 현재 최고 판 이름)·트리 대상 판(3차, 판 이름)·선택(1·2차 — 1차 베이스라인·트리 window·LSTM window·A·k·N_ma(트리·딥러닝)·2차 베이스라인 중 해당하는 것)·현재 입력 묶음 id·코드 검수 완료 여부")
sub_once("사슬 변화 표(`listeria-3차` 의 run \"사슬\" 에 판정마다 그때의 현재 최고 판·트리 대상 판·현재 입력 묶음을 한 줄씩 더함, 1차 → 2차 → 3차 채택마다 → 최종)",
         "사슬 기록(`listeria-3차` 의 run \"사슬\" 에 판정마다 한 줄 — 그때의 현재 최고 판·트리 대상 판·현재 입력 묶음)·변화 표(§1-3, 같은 run)")

out = '\r\n'.join(lines)
open(DST, 'wb').write(out.encode('utf-8'))

# 점검
b0 = find_line('## 0. 고정 조건'); b1 = find_line('## 부록 A')
print('body lines', b1 - b0)
for i, l in enumerate(lines):
    for kw in ('비교 기준', '단계 4~8'):
        if kw in l:
            k = l.find(kw)
            print(kw, i + 1, l[max(0, k-60):k+40])
