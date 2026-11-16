import io
src = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v42.md"
dst = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v43.md"
t = open(src, encoding="utf-8", newline="").read()
assert "\r\n" in t
def rep(old, new, n=1):
    global t
    c = t.count(old)
    assert c == n, (old, c)
    t = t.replace(old, new)
rep("- SPI 예보 칸 = 늘 우리 계산값, 감마 적합 2015-01-01 ~ 2022-06-30 고정(§4-5)\r\n",
    "- SPI 예보 칸 = 늘 우리 계산값, 감마 적합 2015-01-01 ~ 2025-06-30 고정(§4-5) — 기상 쪽 고정값(SPI 감마 적합·DB 에 없는 나라 좌표)은 fold validation 기간이 들어가도 누수로 보지 않음(기상은 정답이 아님)\r\n")
rep("감마 적합 기간 = 2015-01-01 ~ 2022-06-30 하나로 고정(모든 fold·모델·refit·test·운영 같은 값 — 어느 fold validation 보다 앞)",
    "감마 적합 기간 = 2015-01-01 ~ 2025-06-30 하나로 고정(모든 fold·모델·refit·test·운영 같은 값)")
rep("d ≤ 2022-06-30 시료 좌표 중앙값", "d ≤ 2025-06-30 시료 좌표 중앙값")
rep("SPI 감마 적합 기간이 7.5년으로 짧음(§4-5)", "SPI 감마 적합 기간이 10.5년으로 짧음(§4-5)")
rep("- 기상청 SPI(관측 날)의 적합 기준 기간에 T 뒤 해가 들어 있을 수 있음(0-3 에서 확인)\r\n", "")
rep(", 기상청 SPI 일자료의 감마 적합 기준 기간 확인·기록. 측정 정의", ". 측정 정의")
old = "모든 표의 기준 줄) — 단계마다 학습하는 타겟 인코딩 판은 두지 않음"
i = t.index(old); assert t.count(old) == 1
j = t.index("\r\n", i) + 2
t = t[:j] + "- 최종 모델은 머신러닝 판(1·2차 베이스라인·3차 사슬)에서만 고름, 7-a 는 모든 표의 비교 기준 줄로만(담당자 결정)\r\n" + t[j:]
assert "\n" not in t.replace("\r\n", "")
open(dst, "w", encoding="utf-8", newline="").write(t)
print("ok")
