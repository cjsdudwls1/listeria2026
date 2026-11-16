import io
p = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v59.md"
t = io.open(p, encoding="utf-8", newline="").read()
a = "| 3-4 | 결합 모델 2종(정해 둔 구성) | 결합 모델 시계열 갈래 = LSTM 구조의 3-0b 변수 묶음(시퀀스 채널), 3-1 시퀀스 AE·VAE = 트리 대상 판 변수 묶음의 시퀀스 채널 — 위에 쌓지 않음(교체 후보) |"
b = "| 3-4 | 결합 모델 2종(정해 둔 구성) | 시계열 갈래 = LSTM 구조의 3-0b 변수 묶음(시퀀스 채널) — 위에 쌓지 않음(교체 후보) |"
assert t.count(a) == 1
t = t.replace(a, b)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ok")
