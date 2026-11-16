import re

SRC = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v55.md"
DST = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v56.md"

with open(SRC, "rb") as f:
    raw = f.read()
assert b"\r\n" in raw
text = raw.decode("utf-8")
assert not text.startswith("\ufeff")
text = text.replace("\r\n", "\n")


def rep(old, new, count=1):
    global text
    n = text.count(old)
    assert n == count, (old, n)
    text = text.replace(old, new)


# 1. 3-0 알고리즘 10종 -> 11종
rep("·GRU·트랜스포머 인코더·LSTM + attention(1차 W_DL, 기본 설정, §7)",
    "·GRU·트랜스포머 인코더·LSTM + attention·TCN(1차 W_DL, 기본 설정, §7)")
rep("/ 딥러닝 4구조 = LSTM(2-2 LSTM 판 재사용", "/ 딥러닝 5구조 = LSTM(2-2 LSTM 판 재사용")
rep("| 3-0 | 10 알고리즘 × 변수 전체 |", "| 3-0 | 11 알고리즘 × 변수 전체 |")
rep("·GRU·트랜스포머·LSTM + attention 10종을", "·GRU·트랜스포머·LSTM + attention·TCN 11종을")

# 2. 구조 표·§7-2
rep("| 트랜스포머 인코더 | 선형 투영 d_model 64 + 사인 위치 인코딩 → 인코더 2층(헤드 4·FF 128·dropout 0.1) → 평균 풀링 → concat → Dense 64(relu) → sigmoid(⑦-D 는 x_static = g 만, x_cat 없음) |\n",
    "| 트랜스포머 인코더 | 선형 투영 d_model 64 + 사인 위치 인코딩 → 인코더 2층(헤드 4·FF 128·dropout 0.1) → 평균 풀링 → concat → Dense 64(relu) → sigmoid(⑦-D 는 x_static = g 만, x_cat 없음) |\n"
    "| TCN | 인과 합성곱 잔차 블록 5개(블록마다 커널 3 인과 합성곱 2층·ReLU·dropout 0.1·잔차 연결, 채널 64, dilation 1·2·4·8·16 — 받는 범위 125일로 W_DL 최대 100 을 덮음) → 마지막 시점 출력 → 정적·임베딩 concat → Dense 64(relu) → Dense 1(sigmoid) |\n")
rep("·트랜스포머 인코더(3-0·⑦-D)·GRU + attention(⑥)", "·트랜스포머 인코더(3-0·⑦-D)·TCN(3-0)·GRU + attention(⑥)")

# 3. 부록 A-9·요약 23·§8
rep("| 3-0 | 81 | 트리 6 × 4 + 딥러닝 3(GRU·트랜스포머·LSTM + attention) × 4 + 상위 3 딥러닝 × 15(",
    "| 3-0 | 85 | 트리 6 × 4 + 딥러닝 4(GRU·트랜스포머·LSTM + attention·TCN) × 4 + 상위 3 딥러닝 × 15(")
rep("| 합계(2-3·3-0b 제외) | 1,075~1,541 | 단계 8 대상에 따라 — 트리만 1,373, 트리 + 딥러닝 확률 칼럼 3개 1,484(⑥ 2종이면 1,447), 딥러닝 단독·결합 모델 1,098(1,373 − 298 + 23), ② 1,075, 3-0b 묶음이 채택되면 + 57. 예상에는 상한 1,541 |",
    "| 합계(2-3·3-0b 제외) | 1,079~1,545 | 단계 8 대상에 따라 — 트리만 1,377, 트리 + 딥러닝 확률 칼럼 3개 1,488(⑥ 2종이면 1,451), 딥러닝 단독·결합 모델 1,102(1,377 − 298 + 23), ② 1,079, 3-0b 묶음이 채택되면 + 57. 예상에는 상한 1,545 |")
rep("- 0-9 예상 = 1,541 + 2 × 1,068 = 3,677, 최악(2-3·3-0b 상한, 미포함 항목 제외) = 1,541 + 2 × 11,413 = 24,367",
    "- 0-9 예상 = 1,545 + 2 × 1,068 = 3,681, 최악(2-3·3-0b 상한, 미포함 항목 제외) = 1,545 + 2 × 11,413 = 24,371")
rep("→ 합계 1,083~1,549, 최악 = 1,549 + 2 × 11,413 = 24,375", "→ 합계 1,087~1,553, 최악 = 1,553 + 2 × 11,413 = 24,379")
rep("23. 규모: 예상 3,677회(2-3·3-0b 각 3바퀴 가정), 최악 24,367회(양방향 추가 시 24,375) 학습이다",
    "23. 규모: 예상 3,681회(2-3·3-0b 각 3바퀴 가정), 최악 24,371회(양방향 추가 시 24,379) 학습이다")
rep("예상 = 부록 A-9 의 0-9 예상(3,677 —", "예상 = 부록 A-9 의 0-9 예상(3,681 —")

# 4. A-11 TCN 줄 삭제
rep("- 교수님 고도화 계획서의 TCN 은 안 함 — 시계열 딥러닝은 이후 통화의 LSTM·GRU·트랜스포머 구조로(담당자 결정)\n", "")

# ---- 확인 ----
lines = text.split("\n")
# A-9 표 다시 더하기
s = lines.index("### A-9. 학습 횟수(3 fold + refit = 4)")
fixed = 0
for ln in lines[s + 3:]:
    if not ln.startswith("|"):
        break
    cells = [c.strip() for c in ln.strip("|").split("|")]
    name, cnt = cells[0], cells[1]
    if name.startswith("합계"):
        break
    if "~" in cnt:
        print("  범위 줄:", name, cnt)
        continue
    fixed += int(cnt.replace(",", ""))
print("고정 합(단계 8·3-0b 제외) =", fixed)
assert fixed == 1079
assert fixed + 298 == 1377
assert fixed + 298 + 111 == 1488
assert fixed + 298 + 74 == 1451
assert fixed + 23 == 1102 == 1377 - 298 + 23
assert fixed + 298 + 111 + 57 == 1545
assert 1545 + 2 * 1068 == 3681 and 1545 + 2 * 11413 == 24371
assert 1079 + 8 == 1087 and 1545 + 8 == 1553 and 1553 + 2 * 11413 == 24379
assert 24 + 16 + 45 == 85

for old in ["81", "1,075", "1,373", "1,484", "1,447", "1,098", "1,541", "3,677",
            "24,367", "1,083", "1,549", "24,375"]:
    hits = [i + 1 for i, l in enumerate(lines) if re.search(r"(?<![\d,])" + re.escape(old) + r"(?![\d,])", l)]
    print("옛 숫자", old, "->", hits)
    if old != "81":
        assert not hits, old
for new in ["3,681", "24,371", "24,379"]:
    print("새 숫자", new, [i + 1 for i, l in enumerate(lines) if new in l])

# 요약 번호
a = lines.index("## 요약(순서대로 읽기)")
b = lines.index("## 0. 고정 조건")
nums = [int(m.group(1)) for l in lines[a + 1:b] if (m := re.match(r"^(\d+)\. ", l))]
assert nums == list(range(1, 27)), nums
print("요약 번호 1~26 연속")

# 본문 줄 수
c = lines.index("## 부록 A. 구현 사양")
print("본문 줄 수 =", c - b)
assert c - b <= 300

# 문서 전체 '10' 검색(3-0 뜻 확인용)
for i, l in enumerate(lines):
    for m in re.finditer(r"(?<![\d.,~\-])10(?![\d,.])", l):
        ctx = l[max(0, m.start() - 25): m.end() + 15]
        print(f"  10 @{i+1}: …{ctx}…")

out = text.replace("\n", "\r\n")
with open(DST, "wb") as f:
    f.write(out.encode("utf-8"))
print("저장:", DST)
