import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:80])
    s = s.replace(old, new)
def line(prefix, new):
    global s
    L = s.split("\n"); idx = [i for i, x in enumerate(L) if x.startswith(prefix)]
    assert len(idx) == 1, (len(idx), prefix)
    if new is None: del L[idx[0]]
    else: L[idx[0]] = new
    s = "\n".join(L)

# 요약 11
rep("(해외 기상은 Open-Meteo 로 채운 표, 축사는 전체 농장 수 — 자료 없는 곳은 빈칸)", "(축사는 전체 농장 수 — 자료 없는 곳은 빈칸)")
rep("(W, 입력 방식) 판을 2차 기상으로 다시 학습한 2-0 판을 만들고,", "(W, 입력 방식) 1차 판을 그대로 2-0 판으로 두고(기상이 1차와 같아 다시 학습 안 함),")
# 요약 21 규모
rep("예상 15,419회(", "예상 15,415회(")
rep("최악 149,904회 학습", "최악 149,900회 학습")
# §0
rep("기상 쪽 고정값(SPI 감마 적합·DB 에 없는 나라 좌표·옹진군 대표 좌표)", "기상 쪽 고정값(SPI 감마 적합·옹진군 대표 좌표)")
# §4-2
rep("soil_t_nas·no2·co·so2 는 해외 행에서 늘 결측 — 그 결측 표시는 '해외 구조적 결측' 1칼럼으로 합침(해외 행의 soil_t_nas·no2·co·so2 칼럼별 결측 표시 = 0, 합친 '해외 구조적 결측' 칼럼만 1, 딥러닝은 x_static 1칼럼), 1차는 soil_t5~t30·풍향 sin·cos 도 해외 구조적 결측에 넣음(DB 값만이라), 국내 행과 해외 행의 그 밖 결측(DB 에 없는 나라·기간 등)은 칼럼마다 표시",
    "soil_t5~t30·soil_t_nas·no2·co·so2·풍향 sin·cos(1차·2차)와 이슬점 td1(2차)은 해외 행에서 늘 결측(해외 기상 = DB 값만, §4-4) — 그 결측 표시는 '해외 구조적 결측' 1칼럼으로 합침(해외 행의 그 변수들에서 나온 칼럼별 결측 표시 = 0, 합친 '해외 구조적 결측' 칼럼만 1, 딥러닝은 x_static 1칼럼), 국내 행과 해외 행의 그 밖 결측(DB 에 없는 날짜 등)은 칼럼마다 표시")
# §4-4
line("- DB `api_data_info.global_weather_info` 나라 값(나라마다 좌표 1곳) 먼저,",
     "- DB `api_data_info.global_weather_info` 나라 값만(나라마다 좌표 1곳, 보정 없음 — 다른 원천으로 채우지 않음), 운영 예보는 지금 수집 안 함(해외 운영 입력은 운영 준비 때 정함). 기간·칼럼 대응·단위는 부록 A-2. DB 에 없는 변수(soil_t5~t30·soil_t_nas·no2·co·so2·풍향 sin·cos·이슬점)·날짜 = NaN")
line("- 1차 기상 = DB `global_weather_info` 값만(", "- 1차 기상 = 2차 기상(같은 표)")
# §4-5
line("- 해외: 관측·예보 모두 우리 식, 나라 강수로 나라마다 감마 적합(", "- 해외: 관측·예보 모두 우리 식, 나라 강수(DB, 2015-01-01 부터)로 나라마다 감마 적합 — SPI3 2015-03-31·SPI6 2015-06-30 부터(그 전 NaN)")
# 0-4
rep("해외 나라 표·DB 대조·보정·해외 칸 원천 표시(1차 = DB 값만 표)", "해외 나라 표(DB 값만, 1차 = 2차)")
# 단계 2 준비
rep("기상 = 2차 기상(해외 = Open-Meteo 로 채운 표), 알고리즘 = A", "기상 = 1차와 같은 표, 알고리즘 = A")
line("- 2-0 기준 판: A 의 (W, 입력 방식) 판을 2차 기상으로 다시 학습(파생변수 없음)",
     "- 2-0 기준 판: 1차 A 의 (W, 입력 방식) 판 그대로(기상이 1차와 같고 파생변수 없음 — 다시 학습 안 함, 3 fold·refit·test 점수도 1차 것) — 2-1·2-1b(트리 쪽)·2-R 의 바탕(2-R 기준)")
# A-2
line("- 해외 기간: DB = 2015-01-01 ~ 2026-07-07,", "- 해외 기간: DB `global_weather_info` 2015-01-01 ~ 2026-06-30(그 밖 날짜 NaN)")
line("- DB 에 없는 나라: `global_weather_info` 에 없는 나라는 Open-Meteo,", "- DB 에 없는 나라: 기상 NaN(0-1 적용 뒤 해외 시료 나라 95곳 모두 DB 에 있음). `global_soil_ph` 에 없으면 soil_ph NaN")
rep("- 나라 경계 파일(DB 에 없는 나라 좌표) = ", "- 나라 경계 파일(RR1·RR4·RR5 좌표 나라) = ")
line("- 해외 보정: 대상 = DB 에도 있는 변수", None)
rep("- 칼럼 설명에 '근사' 표기: max_wind·sunshine·soil_t5~t30·soil_moist·풍향 sin·cos·rg 지중온도 6·soil_ph·rg 토양 pH 3, SPI 칼럼 설명에 감마 적합 기간",
    "- 칼럼 설명에 '근사' 표기: max_wind·sunshine·soil_moist·soil_ph·rg 토양 pH 3, SPI 칼럼 설명에 감마 적합 기간")
rep("soil_moisture_mean → soil_moist(단위는 겹치는 날 Open-Meteo 값으로 확인해 맞춤)",
    "soil_moisture_mean → soil_moist — 단위 변환 = wind_speed_mean·max ÷ 3.6(km/h → m/s), sunshine_duration ÷ 3600(초 → 시간), soil_moisture_mean × 100(m³/m³ → %), 나머지 그대로")
line("- Open-Meteo: 기온 `temperature_2m_mean/max/min`", None)
# A-9
rep("| 2-0 | 4 | 1판 × 4 |", "| 2-0 | 0 | 1차 A 판 그대로(다시 학습 안 함) |")
rep("| 합계(3-0b 제외) | 1,126~1,535 | 단계 8 대상에 따라 — 트리만 1,424, 트리 + 딥러닝 확률 칼럼 3개(또는 ⑥ 3종) 1,535, 딥러닝 단독·결합 모델 1,149(1,424 − 298 + 23), ② 1,126(단계 4 채택이 없으면 모두 −40). 예상에는 상한 1,535 |",
    "| 합계(3-0b 제외) | 1,122~1,531 | 단계 8 대상에 따라 — 트리만 1,420, 트리 + 딥러닝 확률 칼럼 3개(또는 ⑥ 3종) 1,531, 딥러닝 단독·결합 모델 1,145(1,420 − 298 + 23), ② 1,122(단계 4 채택이 없으면 모두 −40). 예상에는 상한 1,531 |")
rep("- 0-9 예상 = 1,535 + 13 × 1,068 = 15,419, 최악(3-0b 상한, 미포함 항목 제외) = 1,535 + 13 × 11,413 = 149,904",
    "- 0-9 예상 = 1,531 + 13 × 1,068 = 15,415, 최악(3-0b 상한, 미포함 항목 제외) = 1,531 + 13 × 11,413 = 149,900")
# A-11
line("- 해외 지중온도 4칸·풍향·이슬점 = 2차부터 Open-Meteo 근사값으로 채움",
     "- 해외 지중온도 4칸·soil_t_nas·no2·co·so2·풍향 sin·cos·이슬점 = 해외 행 모두 결측(DB 에 없는 변수), 결측 표시는 '해외 구조적 결측' 1칼럼(§4-4·§4-2) — 트리는 NaN 그대로, 로지스틱·딥러닝은 표준화 뒤 결측 0 이라 해외 행 모두 같은 값")
# A-13
rep("`global_weather_info` 좌표), Open-Meteo 수집, 습도·이슬점 계산,", "`global_weather_info` 좌표), 습도·이슬점 계산,")
# B-9
line("- 1차 해외 기상은 DB 값만(2015-01-01 전 NaN)이라",
     "- 해외 기상 = DB 나라 값만(나라마다 좌표 1곳, 2015-01-01 부터) — 지중온도·풍향·이슬점·soil_t_nas·no2·co·so2 는 해외 행에 없고(해외 구조적 결측, §4-2), 해외 SPI 는 2015 년 초 NaN(§4-5)")
assert "Open-Meteo" not in s, [l[:80] for l in s.split("\n") if "Open-Meteo" in l]
p.write_text(s, encoding="utf-8"); print("ok")
