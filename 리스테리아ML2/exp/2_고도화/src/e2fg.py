# -*- coding: utf-8 -*-
"""2차 추가분(사용자 승인 2026-10-19 "싹다 ㄱ"): F = 외부 자료(1차 보고서 표 9 ①), G = 리스테리아 생장식 변수(표 9 ②),
combo = A3 + (검증에서 E00 보다 높았던 F·G 묶음).

정렬: 목표일 d, 앞당김 h -> 예측하는 날 T = d - h. 창 = T-L .. T-1 (L = e2cfg.L 기상 창 길이, 기본 50, T 당일 제외, 기상 요약과 같음).
  F1 SPI   : 시군구_일별_SPI.parquet spi3·spi6 -> 창 요약(10일 구간 평균 L/10 + 평균·표준편차·최소·최대·기울기) 2 x (L/10 + 5)칸(L=50 이면 20)
  F2 토양  : 시군구_일별_토양.parquet 날짜 T 행 6칸 (그 행 값 = exam_day ∈ [T-1095일, T-1일] 기록 - 04_시군구_토양.py, 당일 제외)
  F3 식품  : 품목3단계_식품속성.parquet (train_prdlst_code_2) -> 숫자 aw·ph·na 3칸(결측 그대로) + 범주 aw_label 1칸('결측' = 한 범주)
             ph_label·na_label 은 넣지 않음(식품 세션 결정). 3단계 코드 없는 단위('코드없음')와 표에 없는 코드는 결측.
  G1       : Silva et al. 2020 카디널 모델 온도 항 f(T) 를 날마다 시군구 일평균기온(avg_temp)으로 -> 평균·최근 10일 평균·합·f>0 인 날 수 4칸
  G2       : µmax = µopt·f(T)·f(pH)·f(aw) (Silva 2020) 와 생장 확률 P (Tienungoon et al. 2000, Scott A·L5 두 균주, 젖산 항 0)
             날마다 기온 x 3단계 대표 aw·ph -> 각각 평균·최근 10일 평균·최대 = 3 x 3 = 9칸
  G3       : Ivanek et al. 2009 강수 모델 logit(P) = −1.79 + 0.13·Precipitation.2 (채취 2일 전 강수량 mm) -> 날 t 의 P 는 rain(t-2)
             -> 평균·최근 10일 평균·최대 3칸
계수는 모두 '리스테리아 생장식.xlsx' 시트 '(완료) 리스테리아 모노사이토제네스' 에 적힌 값만 쓴다(셀 위치는 GROWTH_CELLS).
정의역 밖(로그·제곱근 안이 0 이하, 카디널 범위 밖): 생장 없음 = f 0, µ 0, P 0. 입력(기온·aw·pH·강수)이 결측이면 결측.
원자료는 읽기만 한다. 값 채움·단위 변환 없음(특성 행렬에 넣을 때 float32 로 바뀌는 것만).

M 'wxanom' (사용자 승인 2026-10-19, 1차 보고서 표 9 ② 이동평균 차감 - 10-12 교수님 통화의 이동평균 차감 피처, 실험 M1·A3_M1):
  기상 19개 변수마다 시군구 x 날짜 기상표(1차 W) 위에서 a_t = x_t − mean(x_{t−K} … x_{t−1}), K = e2cfg.WXANOM_K(30)
  (앞 K 일 평균, 당일 제외, 결측은 빼고 평균, 앞 K 일이 전부 결측이거나 x_t 결측이면 a_t 결측. 기상표 시작 전 날은 없는 날 = 결측처럼 뺌 -
   실험 창의 가장 이른 날 2014-12-18 이 기상표 시작 2014-04-07 뒤 255일이라 실제로는 해당 없음)
  -> 창 T−L … T−1 의 a_t 로 변수마다 평균·최근 10일(T−10 … T−1) 평균·최대 = 19 x 3 = 57칸(결측 빼고 계산, 전부 결측이면 결측).
  창 끝이 T−1 이고 a_{T−1} 의 앞 K 일도 T 이전이라 T 당일·이후 기상은 쓰지 않음. F·G 조합 규칙(BLOCK_ORDER·combo)에는 들어가지 않음.

Q 생장 범위 (리서치팀 '리스테리아 모노사이토제네스 예측인자 생장 최적·한계 범위' 표, 파일은 [질문] 세션이 data/데이터셋/생장범위/ 에 만듦,
  정의 = 생장범위_정의.csv, 실험 Q1~Q4 = A3 + 묶음). F·G 조합 규칙(BLOCK_ORDER·combo)에는 들어가지 않음.
  'rgwx'   : 생장범위_일별_기상.parquet 날마다 6칸(0/1 또는 거리) -> 창 T−L … T−1 에서 칸마다 평균(0/1 칸은 그 범위 날 비율)·
             최근 10일(T−10 … T−1) 평균 = 6 x 2 = 12칸(결측 날 빼고 계산, 전부 결측이면 결측)
  'rgsoil' : 생장범위_일별_토양pH.parquet 날짜 T 행 3칸(soil_ph = exam_day ∈ [T-1095, T-1] 기록 중앙값으로 만든 값 - F2 와 같은 행)
  'rgfood' : 생장범위_품목3단계.parquet (train_prdlst_code_2) -> 4칸(결측 그대로, 코드없음·표에 없는 코드는 결측 - F3 와 같은 규칙)

P 외부 지표 (사용자 승인·확정 2026-10-19, 파일은 [질문] 세션이 data/데이터셋/외부_지표/ 에 만듦, 실험 P1·P2·P3·P1p·P2p = A3 + 묶음).
  F·G 조합 규칙(BLOCK_ORDER·combo)에는 들어가지 않음. 시군구 x 자료연도 표 -> 예측하는 날 T 에 사용가능일 ≤ T−1 인 가장 최근 자료연도 행 1칸.
  'eco'   : 친환경_인증비율.parquet 친환경_비율 ((유기+무농약 인증 면적) ÷ 경지면적)
  'gw'    : 지하수_정기수질검사.parquet 적합률 (적합 ÷ 검사)
  'eco_p'·'gw_p' : 위약판 - 자료연도마다 시군구끼리 값을 섞음(고정 난수 EXTIDX_PLACEBO_SEED, 결측도 같이 섞음), 사용가능일은 그대로
"""
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C1
import e2cfg as E

DATA_DIR = C1.DATA_DIR
SPI_PATH = DATA_DIR / "외부_수집" / "시군구_일별_SPI.parquet"
SOIL_PATH = DATA_DIR / "식정원_db_토양" / "시군구_일별_토양.parquet"
FOOD_PATH = DATA_DIR / "식정원_db_식품" / "품목3단계_식품속성.parquet"
GROWTH_XLSX = DATA_DIR / "리스테리아 생장식.xlsx"
GROWTH_SHEET = "(완료) 리스테리아 모노사이토제네스"
RG_DIR = DATA_DIR / "생장범위"
RG_WX_PATH = RG_DIR / "생장범위_일별_기상.parquet"
RG_SOIL_PATH = RG_DIR / "생장범위_일별_토양pH.parquet"
RG_FOOD_PATH = RG_DIR / "생장범위_품목3단계.parquet"
RG_WX_VARS = ["rg_tmax_opt", "rg_tavg_opt_dist", "rg_t_out_lim", "rg_soilt_opt", "rg_soilt_lim", "rg_soilt_opt_dist"]
RG_SOIL_VARS = ["rg_soilph_opt", "rg_soilph_lim", "rg_soilph_opt_dist"]
RG_FOOD_VARS = ["rg_aw_lim", "rg_aw_opt_dist", "rg_ph_lim", "rg_ph_opt_dist"]
EXTIDX_DIR = DATA_DIR / "외부_지표"
EXTIDX = {"eco": (EXTIDX_DIR / "친환경_인증비율.parquet", "친환경_비율", "eco_ratio"),      # 묶음: (파일, 값 칸, 특성 이름)
          "gw": (EXTIDX_DIR / "지하수_정기수질검사.parquet", "적합률", "gw_pass_rate")}
EXTIDX_PLACEBO_SEED = 20261019
HACCP_PATH = EXTIDX_DIR / "haccp_유효구간.parquet"        # HACCP 지정 유효 구간표([질문] 세션, 처리 = data/중간/외부지표/haccp_I0580_보고.md 9절)

SPI_VARS = ["spi3", "spi6"]
SOIL_VARS = ["soil_ph", "soil_ph_n", "soil_om", "soil_om_n", "soil_vldpha", "soil_vldpha_n"]
SOIL_MED_IDX = [0, 2, 4]                                  # 'soilmed' 묶음: 중앙값 3칸만(검정 건수 *_n 제외, 사용자 승인 10-19)
SOIL_MED_VARS = [SOIL_VARS[i] for i in SOIL_MED_IDX]
FOOD_NUM = ["aw", "ph", "na"]
FOOD_CAT = "aw_label"
FOOD_MISSING = "결측"
LAST = 10                                      # '최근 10일' = T-10 .. T-1
CHUNK = 16384

# ---------------------------------------------------------------- 생장식 계수 (엑셀 값 그대로)
SILVA = {"Tmin": -1.273, "Topt": 37.26, "Tmax": 45.12,          # F6
         "pHmin": 4.303, "pHopt": 7.085, "pHmax": 9.483,         # F26
         "awmin": 0.894, "awopt": 0.995,                          # F19
         "awmax": 1.00,                                           # D6 식 안의 (aw−1.00), (awopt−1.00)
         "mu_opt": 0.409}                                         # J6(비고) 식 맨 앞 계수 - D6 식의 '√µopt' 자리
TIEN = {"scottA": {"b0": -6.023, "b1": 19.00, "b2": -3.049, "b3": 7514.0, "Tmin": 0.4164,      # F15
                   "b4": 4.635, "awmin": 0.9142,                                                # F21
                   "b5": 141.0, "b6": 240.2, "pHmin": 3.350},                                   # F28
        "L5": {"b0": -25.36, "b1": 44.12, "b2": -7.022, "b3": 10257.0, "Tmin": -1.623,         # F15
               "b4": 8.951, "awmin": 0.9152,                                                    # F21
               "b5": 291.8, "b6": 704.1, "pHmin": 3.350}}                                       # F28
TIEN_FIXED = {"d": 0.536, "Tmax": 48.0}        # D15 식 'ln{1−exp[0.536(T−48)]}', J15 'Tmax=48℃ ... d=0.536 고정'
TIEN_STRAINS = ["scottA", "L5"]
IVANEK = {"b0": -1.79, "b1": 0.13, "lag": 2}   # D41 식, F41 계수, G41 'Precipitation.2: 채취 2일 전 강수량(mm)'

# 셀 위치 기록(표·보고용). selftest 가 엑셀을 다시 읽어 이 셀들에 이 숫자 글자가 있는지 확인한다.
GROWTH_CELLS = [
    ("G1·G2 Silva 2020 온도 항", "D6(식), F6(값)", "Tmin=−1.273, Topt=37.26, Tmax=45.12 (°C)"),
    ("G2 Silva 2020 pH 항", "D26(식), F26(값)", "pHmin=4.303, pHopt=7.085, pHmax=9.483"),
    ("G2 Silva 2020 aw 항", "D19(식), F19(값), D6 식 안 1.00", "awmin=0.894, awopt=0.995, awmax=1.00"),
    ("G2 Silva 2020 µopt", "J6(비고 식 맨 앞)", "µopt=0.409 (D6 식에서는 '√µopt' 자리 - 단위 적혀 있지 않음)"),
    ("G2 Tienungoon 2000 P 온도", "D15(식), F15(값), J15(고정값)",
     "Scott A: b0=−6.023, b1=19.00, b2=−3.049, b3=7514, Tmin=0.4164 / L5: b0=−25.36, b1=44.12, b2=−7.022, b3=10257, Tmin=−1.623; d=0.536, Tmax=48"),
    ("G2 Tienungoon 2000 P aw", "D21(식), F21(값)", "Scott A: b4=4.635, awmin=0.9142 / L5: b4=8.951, awmin=0.9152"),
    ("G2 Tienungoon 2000 P pH", "D28(식), F28(값)", "Scott A: b5=141.0, b6=240.2, pHmin=3.350 / L5: b5=291.8, b6=704.1, pHmin=3.350"),
    ("G2 Tienungoon 2000 P 젖산", "D15 식 b7 항", "b7·UMIC·pKa 는 엑셀에 없음 -> [LAC]=0 이면 ln(1−0)=0 이라 항 전체 0 (사양 '젖산 항은 0')"),
    ("G3 Ivanek 2009 강수", "D41(식), F41(값), G41(변수 정의)", "intercept=−1.79, β(Precipitation.2)=0.13, Precipitation.2 = 채취 2일 전 강수량(mm)"),
]
# 엑셀 셀 글자 확인용(셀 -> 있어야 하는 숫자 글자). 유니코드 빼기(−)는 '-' 로 바꿔 비교.
GROWTH_CELL_TEXT = {
    "F6": ["-1.273", "37.26", "45.12"], "F26": ["4.303", "7.085", "9.483"], "F19": ["0.894", "0.995"],
    "D6": ["1.00"], "J6": ["0.409"],
    "F15": ["-6.023", "19.00", "-3.049", "7514", "0.4164", "-25.36", "44.12", "-7.022", "10257", "-1.623"],
    "D15": ["0.536", "48"], "J15": ["48", "0.536"],
    "F21": ["4.635", "0.9142", "8.951", "0.9152"], "F28": ["141.0", "240.2", "3.350", "291.8", "704.1"],
    "D41": ["Precipitation.2"], "F41": ["-1.79", "0.13"], "G41": ["2일 전 강수량(mm)"],
}
# 쓰지 않은 모델(엑셀 행)과 이유 - 표 '설정' 시트·보고용
EXCLUDED_MODELS = [
    ("Wijtes 1993 Modified Ratkowsky (D5·D18·D23)", "c, cpH = 'not mentioned'(F5·F18·F23) -> 사양대로 제외"),
    ("Koutsoumanis 2004 로지스틱 (D16·D22·D29)", "Tienungoon 2000 계수가 엑셀에 다 있어 사양대로 쓰지 않음(사양: Tienungoon 이 불완전할 때만 대체). "
     "broth 는 a₀~a₈ 이 모두 있음(F16 a₀=−605.935·a₁=1.272·a₇=−0.065, F22 a₃=−809.232·a₅=−4.411·a₆=−183.674, F29 a₂=180.260·a₄=0.507·a₈=−11.460), "
     "agar 는 a₄(T·pH)='NS'(F29). 참고: broth 를 엑셀 값 그대로 계산하면 logit 이 25℃·aw 0.993·pH 7 에서 −10.3, aw 0.98 에서 −137.6, "
     "aw 0.95 에서 −318.9(P≈0)라 대체 모델로도 쓰기 어려움"),
    ("Chandler-Khayd 2023 토양 온도·토양 수분·토양 pH·강수 (B34·B36·B39·B40)",
     "오즈비(OR)만 있고 절편 없음, HOBO 센서·강수는 '표준화 변수(mean-centered, SD 단위)'인데 평균·표준편차가 엑셀에 없음 -> 확률 계산 불가. "
     "시료 토양수분 OR(10% 단위)도 절편 없음. 변수 하나의 로지스틱 변환은 단조라 기존 기상 요약(soil_moist·rain·soil_t5 평균 등)에 대해 트리 입력으로 새 정보 거의 없음"),
    ("Ivanek 2009 식생층 온도 모델 (D35)", "Freeze.thaw.3 정의 없음, Loam.soil(양토 여부) 자료 없음, Northing(뉴욕 UTM 북향 좌표, 평균 4,707,076m) 을 한국에 쓸 수 없음"),
    ("Ivanek 2009 위치 모델 토양 수분 (D37)", "Water.storage.50 = 50cm 유효수분 저장량(cm) - 우리 soil_moist(%) 와 단위·정의 다름, Loam.soil·Freeze.thaw.3 없음"),
    ("Ivanek 2009 수계 모델 (D43)", "Northing·Freeze.thaw.3 (위와 같은 이유), 수계 대상"),
    ("Weller 2020 CTree 일사 (D44)", "분기값 0.4 MJ/m² 하나만 있고 분기 방향·잎 확률(검출 비율)·다른 분기가 없음(트리 규칙 불완전). "
     "'2–3일 평균 일사량'의 시간 단위도 적혀 있지 않음 - 우리 solar_rad(MJ/m², 일합)는 결측 아닌 칸의 99.99% 가 0.4 초과라 규칙이 거의 상수"),
    ("Devlieghere 1999, Hwang 2007, Nerbrink 1999, Shrestha 2022, Yoon 2009, Huang 2022 등 (D7·D8·D24·D25·D17·D30 …)",
     "CO2·Na-lactate·sodium diacetate·CDVB·침지시간 등 우리에게 없는 변수 필요, 또는 일부 계수만 적힘(F7·F8 등) - 사양 G 대상 아님"),
    ("Xanthiakos 2006, Bolivar 2023, Cho 2011, Thomas 2019, Lobacz 2013, Rosshaug 2012 (D9~D14·D27)",
     "특정 식품(우유·훈제연어파테·김밥·치즈) 전용 온도 모델 - 사양 G 는 Silva 2020(G1·G2)·Tienungoon 2000(G2)만 지정"),
]
G_LIMIT = ("모든 G 변수는 바깥 기온·시군구 기상으로 만든 대리 변수: 식품 내부·보관·유통 온도가 아니라 시군구 대표 좌표의 일평균기온(avg_temp)·강수(rain)를 생장식에 그대로 넣은 값. "
           "식품 aw·pH 는 품목 3단계 대표값(품목3단계_식품속성.parquet, 시료 실측 아님). Silva 2020 은 배지(BHI·TSB·TPB) 메타분석, Tienungoon 2000 은 TSB-YE 배지 "
           "Scott A·L5 균주 실험식(적용범위 3~35℃, aw 0.92~0.997, pH 3.4~7.4 - J15), Ivanek 2009 는 미국 뉴욕주 자연환경 토양의 Listeria spp. 검출 확률. "
           "적용범위 밖 값도 식 그대로 계산(잘라내지 않음)")

# F 자료 한계(검증 발견 2026-10-19, 수치는 그날 잰 값 - scratch/fg/verify_findings.py). 계산은 바꾸지 않고 설정 시트에 적기만 한다.
# 처리 방법(결측 채움·칸 추가·칸 빼기·지연일)은 사용자 결정 필요.
F_NOTES = [
    ("F1 한계: SPI 결측 구간",
     "시군구_일별_SPI.parquet 에서 모든 시군구가 빈 날이 2025-05-08~05-21(14일), 2025-05-27~06-02(7일), 2026-05-09(1일) 등에 있음"
     "(원 CSV 에 그 지수가 전 지점 빈칸인 날 - 05_시군구_SPI.py 규칙으로 결측). 그래서 창 안 10일 구간 평균 칸(spi*_b01~b05) 중 하나 이상이 결측인 "
     "(행 x h) 비율이 학습 0%, 검증 2.49%, 시험 4.69%(spi3·spi6 같음, 창 50일 전체가 결측인 경우 0). 결측은 채우지 않음. "
     "CatBoost 기본 nan_mode=Min 은 결측을 그 칸의 모든 값보다 작게 다루므로, 학습에서 본 적 없는 이 결측이 검증·시험에서 '매우 낮은 SPI' 쪽으로 "
     "읽힐 수 있음 -> F1·F4 검증 점수·조합 선정 해석 주의(누설은 아님). 처리 방법은 사용자 결정 필요"),
    ("F1 한계: SPI 기준 기간",
     "기상청 SPI 를 계산할 때의 분포 적합 기준 기간이 원 CSV·05_시군구_SPI.py 어디에도 적혀 있지 않음(설명 없음 - 확인 필요). "
     "기준 기간에 학습 기간 뒤의 해가 들어 있다면 학습 기간 SPI 값에 그 뒤 기후 분포가 조금 반영됨(라벨 정보는 아님)"),
    ("F2 한계: 기록 수(*_n)는 날짜 대리 변수일 수 있음",
     "soil_*_n 은 그 시군구 지난 3년 검정 기록 수라 토양 성질보다 검정 사업량·날짜를 담음: soil_ph_n 중앙값((행 x h) 전체) "
     "학습 5,558 / 검증 8,364 / 시험 8,836, 학습 T 연도별 2015년 1,280 · 2016년 3,775 · 2017년 5,537 · 2019년 4,061 · 2024년 8,167. "
     "검증·시험 값이 학습 분포 위쪽에 몰려 트리가 날짜 대리 변수로 쓸 수 있음(해석 주의). 6칸 사용은 사양대로"),
    ("F2 한계: 2018년 전 창이 3년보다 짧음",
     "토양 원자료(raw_soil_ph_info) exam_day 가 2015-01-01 부터라 T < 2018-01-01 이면 창 [T−1095, T−1] 이 실제로 3년보다 짧음 "
     "(학습 (행 x h) 의 12.43%)"),
    ("F2 한계: DB 등록 시점",
     "토양 원자료 exam_day 월별 기록 수가 2026-04 21,381 · 05 19,454 · 06 16,082 · 07 6,796 으로 2025년 같은 달 76,157 · 61,467 · 51,541 · 57,483 보다 "
     "적음 -> 받은 시점(2026-10) DB 에 최근 기록이 덜 들어온 것으로 보임(확인 필요). 원자료에 등록일 칼럼이 없어(raw_soil_ph_info 칼럼에 없음) "
     "학습 기간 창에 T 뒤에 등록된 기록이 'T 전날까지 자료'로 들어 있을 수 있음 - 라벨 누설은 아니고 운영 때 쓸 수 있는 자료와 다를 수 있는 문제. "
     "SPI 공표 시점도 자료에 적혀 있지 않음. 등록 지연을 반영할지(날짜 T−지연일 행 사용 등)는 사용자 결정 필요"),
]

if E.L != E.L_DEFAULT:                         # 수치는 2026-10-19 에 50일 창으로 잰 값 - 다른 창 길이면 그 사실을 앞에 적음(글 내용은 그대로)
    F_NOTES = [(k, f"[이 줄의 수치는 기상 창 50일로 잰 값 - 지금 창 {E.L}일] " + v) if k.startswith("F1 한계: SPI 결측") or k.startswith("F2 한계: 기록 수")
               else (k, v) for k, v in F_NOTES]

AW_INEQ_RE = "초과|이상|미만|이하"
# g_notes 글에 적는 행 수(학습·검증·시험 단위x날짜 행, 2026-10-19 잰 값). selftest_fg (f) 가 다시 세어 같은지 확인한다.
NOTE_ROWS = {"rising": (4228, 264, 308), "rising_pos": (0, 0, 0), "ineq_with_ph": (29650, 3715, 3941)}


def tien_ph_min():
    """Tienungoon 식품 부분의 pH 항 b5·u + b6·u² (u = ln(1−10^(pHmin−pH)))이 가장 작아지는 pH (균주마다). b6 > 0 이라 그보다 낮은 pH 에서 다시 커짐."""
    out = {}
    for st in TIEN_STRAINS:
        c = TIEN[st]
        out[st] = float(c["pHmin"] - np.log10(1.0 - np.exp(-c["b5"] / (2.0 * c["b6"]))))
    return out


def food_domain_facts(food=None):
    """품목 3단계 표에서 G2 해석에 필요한 코드 목록(설정 시트·selftest 용).
    rising = 어느 균주든 pH < 그 균주 pH 항 최소 pH 이면서 식품 부분이 유한(aw > awmin, pH > pHmin) -> P 가 pH 항 때문에 커질 수 있음
    rising_zero = 같은 pH 쪽이지만 aw ≤ awmin 또는 pH ≤ pHmin 이라 정의역 규칙으로 P 0
    ineq = aw_label 이 부등호(초과·이상·미만·이하) / aw1 = aw = 1.00"""
    if food is None:
        food = load_food()
    aw = food["aw"].to_numpy(dtype=float)
    ph = food["ph"].to_numpy(dtype=float)
    pmin = tien_ph_min()
    rise = np.zeros(len(food), dtype=bool)
    side = np.zeros(len(food), dtype=bool)
    for st in TIEN_STRAINS:
        B = tien_B(aw, ph, st)
        rise |= (ph < pmin[st]) & np.isfinite(B)
        side |= ph < pmin[st]
    ineq = food["aw_label"].astype(str).str.contains(AW_INEQ_RE, regex=True, na=False).to_numpy()
    aw1 = aw >= 1.0
    return {"pmin": pmin, "rising": food[rise], "rising_zero": food[side & ~rise], "ineq": food[ineq],
            "ineq_with_ph": food[ineq & ~np.isnan(ph)], "aw1": food[aw1], "aw1_with_ph": food[aw1 & ~np.isnan(ph)]}


def _ph_term(pH, st):
    """Tienungoon pH 항 b5·u + b6·u² (u = ln(1−10^(pHmin−pH)))."""
    c = TIEN[st]
    u = float(np.log(1.0 - 10.0 ** (c["pHmin"] - pH)))
    return c["b5"] * u + c["b6"] * u * u


def _nm(df, with_vals=True):
    if with_vals:
        return ", ".join(f"{r.train_prdlst_nm_2}(aw {r.aw:g}{'' if r.aw_label == '단일값' else ' ' + str(r.aw_label)}, pH {r.ph:g})"
                         for r in df.itertuples(index=False))
    return ", ".join(str(r.train_prdlst_nm_2) for r in df.itertuples(index=False))


def g_notes(food=None):
    """G2 한계 줄(검증 발견 2026-10-19). 코드 목록은 표에서 그때 계산, 행 수는 2026-10-19 잰 값(scratch/fg/verify_findings.py)."""
    f = food_domain_facts(food)
    pm = f["pmin"]
    return [
        ("G2 한계: Tienungoon pH 항(F28 b6 양수)",
         f"F28 의 b6 = {TIEN['scottA']['b6']}(Scott A)·{TIEN['L5']['b6']}(L5)가 양수라 pH 항 b5·u + b6·u²(u = ln(1−10^(pHmin−pH)))은 "
         f"pH {pm['scottA']:.3f}(Scott A)·{pm['L5']:.3f}(L5)에서 가장 작고, 그보다 낮은 pH(pHmin 3.35 까지)에서 다시 커짐"
         f"(예: pH 3.4 에서 pH 항 Scott A {_ph_term(3.4, 'scottA'):+,.1f}, L5 {_ph_term(3.4, 'L5'):+,.1f}). J15 적용범위(pH 3.4~7.4) 안. 엑셀 값 그대로 계산함(원 논문과 부호·값 대조는 안 함 - 확인 필요). "
         f"이 때문에 P 가 커질 수 있는 품목 3단계 코드 {len(f['rising'])}개: {_nm(f['rising'])}. "
         "학습·검증·시험 단위x날짜 행 " + " · ".join(f"{x:,}" for x in NOTE_ROWS["rising"])
         + ", 검출 " + " · ".join(str(x) for x in NOTE_ROWS["rising_pos"]) + "(2026-10-19 잰 값). 예(h=1 창 평균): 과일·채소류음료는 두 균주 모두 P 약 0.88~0.94, "
         "소스류는 L5 약 0.63~0.69 인데 Scott A 약 1e-5. "
         f"같은 pH 쪽이지만 aw ≤ awmin 또는 pH ≤ pHmin 이라 정의역 규칙으로 P 0 인 코드 {len(f['rising_zero'])}개: {_nm(f['rising_zero'], False)}"
         "(그래서 pH 3.35 경계에서 값이 끊김). 이 코드들은 Silva pH 항(pHmin 4.303)으로 µmax 는 0"),
        ("G2 한계: aw 부등호 표기를 점값으로 씀",
         f"aw_label 이 부등호(초과·이상·미만·이하)인 품목 3단계 코드 {len(f['ineq'])}개는 aw 칸의 경계값을 그대로 점값으로 씀(F3 숫자 aw 와 G2 모두). "
         f"이 가운데 ph 가 있어 G2 가 계산되는 코드 {len(f['ineq_with_ph'])}개(학습·검증·시험 단위x날짜 행 " + " · ".join(f"{x:,}" for x in NOTE_ROWS["ineq_with_ph"]) + " - 2026-10-19 잰 값): "
         f"{_nm(f['ineq_with_ph'])}. Silva f(aw) 는 aw 0.98 에서 {float(silva_faw(0.98)):.3f}, 0.99 에서 {float(silva_faw(0.99)):.3f}, "
         "0.995 에서 1 이라 실제 aw 가 경계보다 높으면 µmax 가 최대 약 1.3배 달라질 수 있음(Tienungoon P 는 이 구간에서 거의 1 이라 차이가 작음). "
         "처리를 바꿀지(예: 부등호 aw 는 G2 결측)는 사용자 결정 필요"),
        ("G2 한계: aw = 1.00 코드",
         f"aw = 1.00 인 코드 {len(f['aw1'])}개({_nm(f['aw1'], False)}): "
         + ("모두 ph 가 결측이라 G2(µmax·P)는 0 이 아니라 결측(지금 자료에는 Silva aw 항이 aw = 1.00 때문에 0 이 되는 행이 없음)"
            if len(f["aw1_with_ph"]) == 0 else
            f"ph 가 있는 {len(f['aw1_with_ph'])}개({_nm(f['aw1_with_ph'], False)})는 Silva aw 항이 0 이라 µmax 항상 0, 나머지는 ph 결측이라 G2 결측")),
    ]

# ---------------------------------------------------------------- 묶음
FG_GROUPS = ["F1", "F2", "F3", "F4", "G1", "G2", "G3"]
FG_BLOCKS = {"F1": ["spi"], "F2": ["soil"], "F3": ["food"], "F4": ["spi", "soil", "food"],
             "G1": ["g1"], "G2": ["g2"], "G3": ["g3"]}
BLOCK_ORDER = ["spi", "soil", "soilmed", "food", "g1", "g2", "g3"]
BLOCK_DESC = {"spi": f"SPI3·SPI6 창 요약 {E.NSPI}", "soil": "토양 날짜 T 6", "soilmed": "토양 중앙값 날짜 T 3(검정 건수 제외)",
              "food": "식품 aw·ph·na 3 + aw_label 범주 1",
              "g1": "G1 온도 생장 요인 f(T) 요약 4", "g2": "G2 µmax·생장확률 P(2균주) 요약 9", "g3": "G3 강수 검출확률(Ivanek 2009) 요약 3"}
G_STATS = {"g1": ("mean", "mean10", "sum", "npos"), "g2": ("mean", "mean10", "max"), "g3": ("mean", "mean10", "max")}

# ---------------------------------------------------------------- M. 이동평균 차감 기상 ('wxanom', 사용자 승인 2026-10-19)
WXANOM = "wxanom"
WXANOM_STATS = ("mean", "mean10", "max")
BLOCK_DESC[WXANOM] = f"기상 19변수 x (평균·최근 10일 평균·최대) = 57, a_t = x_t − 앞 {E.WXANOM_K}일 평균"
# ---------------------------------------------------------------- Q. 생장 범위 ('rgwx'·'rgsoil'·'rgfood', 실험 Q1~Q4)
RGWX, RGSOIL, RGFOOD = "rgwx", "rgsoil", "rgfood"
RG_BLOCKS = [RGWX, RGSOIL, RGFOOD]
RG_STATS = ("mean", "mean10")
# ---------------------------------------------------------------- P. 외부 지표 ('eco'·'gw' + 위약판 'eco_p'·'gw_p', 실험 P1·P2·P3·P1p·P2p)
HACCP_BLOCKS = ["haccp", "haccp_p"]           # P4·P4p (HACCP 은 시군구 x 업종묶음 x 날짜 - 연도표가 아니라 따로 계산)
P_BLOCKS = ["eco", "gw", "eco_p", "gw_p"] + HACCP_BLOCKS
EXT_BLOCKS = BLOCK_ORDER + [WXANOM] + RG_BLOCKS + P_BLOCKS   # ext_* 함수가 아는 묶음 전부. F·G 조합(blocks_of·combo_plan)은 BLOCK_ORDER 만 씀
BLOCK_DESC[RGWX] = "일별 기상 범위 6칸 x (평균·최근 10일 평균) = 12"
BLOCK_DESC[RGSOIL] = "토양 pH 범위 날짜 T 3"
BLOCK_DESC[RGFOOD] = "식품 aw·pH 범위 3단계 4"
RG_RULE = ("리서치팀 예측인자 정리 ppt 표 '리스테리아 모노사이토제네스 예측인자 생장 최적·한계 범위' 로 [질문] 세션이 만든 칸(정의·계산·근거 = "
           "data/데이터셋/생장범위/생장범위_정의.csv). 기온 최적 30~37°C(최고기온 0/1, 평균기온 거리)·한계 −2~45°C(밖으로 나간 날 1), "
           "5cm 지중온도 최적 8~15°C·한계 8~30°C(0/1, 거리), 토양 pH 최적 6.5~7.0·한계 5.0~8.0(soil_ph, 0/1, 거리), "
           "식품 aw 한계 0.90~0.997(0/1)·최적 0.97(거리), 식품 pH 한계 4.0~9.6(0/1)·최적 7.0(거리). 염도(≤16%)는 na 단위가 문서로 확인되지 않아 없음. "
           "입력이 결측이면 결측(0 으로 채우지 않음)")
RG_JOIN = (f"rgwx = 날마다 6칸을 창 T−{E.L} … T−1 에서 칸마다 평균(0/1 칸은 범위 안 날 비율)·최근 10일(T−10 … T−1) 평균 = 12칸(결측 날 빼고, "
           "전부 결측이면 결측; 결측 날은 2014-04-07~12-31 뿐이라 실험 창에는 거의 없음). rgsoil = 날짜 T 행 3칸(F2 와 같은 행, 검정 건수 안 씀; "
           "41115 수원시 팔달구는 토양 기록이 없어 전 기간 결측). rgfood = 단위의 3단계 코드(train_prdlst_code_2)로 4칸(코드없음·표에 없는 코드는 결측, "
           "aw 결측 49·ph 결측 120 품목; aw 부등호 표기 22개 품목은 숫자 그대로 - 생장범위_부등호표기.csv). Q1 = A3 + rgwx, Q2 = A3 + rgsoil, "
           "Q3 = A3 + rgfood, Q4 = A3 + 셋 다(A3 의 k* 그대로)")
BLOCK_DESC["eco"] = "친환경 인증 면적 비율 1"
BLOCK_DESC["gw"] = "지하수 정기 수질검사 적합률 1"
BLOCK_DESC["eco_p"] = "친환경 비율 위약판 1"
BLOCK_DESC["gw_p"] = "지하수 적합률 위약판 1"
P_RULE = ("[질문] 세션이 만든 시군구 x 자료연도 표(data/데이터셋/외부_지표/, 근거·처리 = data/중간/외부지표/연도표_작성보고.md '사용자 확정(10-19)' 절). "
          "eco = 친환경_인증비율.parquet 친환경_비율((유기+무농약 인증 면적) ÷ 경지면적, 자료연도 2016~2025, 사용가능일 = 자료연도+2년 1월 1일), "
          "gw = 지하수_정기수질검사.parquet 적합률(적합 ÷ 검사, 자료연도 2013~2024, 사용가능일 = max(판권면 Y-12-31, PDF 만든 날)). "
          "면적·건수 원값 칸은 넣지 않음(지역 크기·해마다 늘어나는 건수 = 날짜 시계 위험)")
P_JOIN = ("예측하는 날 T 에 그 시군구의 '사용가능일 ≤ T−1' 인 가장 최근 자료연도 행 값 1칸(없으면 결측, 값 빈칸은 결측 그대로). "
          "친환경은 첫 사용가능일이 2018-01-01 이라 T ≤ 2018-01-01 인 학습 행은 결측(검증·시험 행에는 없음 - 학습에서 결측이 '2018년 이전' 표시가 됨). "
          f"위약판(eco_p·gw_p) = 자료연도마다 256 시군구의 값(결측 포함)을 고정 난수(seed {EXTIDX_PLACEBO_SEED})로 섞은 표, 사용가능일·붙이는 규칙은 같음. "
          "P1 = A3 + eco, P2 = A3 + gw, P3 = A3 + eco + gw, P1p = A3 + eco_p, P2p = A3 + gw_p (A3 의 k* 그대로)")
BLOCK_DESC["haccp"] = "HACCP 지정 비율 1"
BLOCK_DESC["haccp_p"] = "HACCP 지정 비율 위약판 1"
HACCP_RULE = ("[질문] 세션이 만든 HACCP 지정 유효 구간표(data/데이터셋/외부_지표/haccp_유효구간.parquet, 처리 = data/중간/외부지표/haccp_I0580_보고.md 9절). "
              "예측하는 날 T 의 값 = T−1 에 유효한(시작일 ≤ T−1 ≤ 끝일) 그 시군구·업종묶음 지정번호 수 ÷ 같은 날 전국 같은 업종묶음 지정번호 수"
              "(시군구가 비어 있는 행은 전국 수에만 셈). 단위의 2단계 코드(train_prdlst_code)가 업종묶음의 '우리_2단계_코드목록' 어디에도 없으면 결측"
              "('대응 없음'·코드목록 없는 업종묶음은 쓰지 않음, 축산물가공 HACCP 은 끝 정보가 없어 구간표에 없음 - 식육가공품 C0316 등 결측). "
              "T < 사용시작일(2017-08-03, 그 전에 끝난 지정이 자료에 거의 없어 과소 집계) 은 결측. "
              f"위약판(haccp_p) = 달력 연도마다 시군구 순서를 고정 난수(seed {EXTIDX_PLACEBO_SEED})로 섞은 값(같은 해·모든 업종묶음에 같은 섞기). "
              "P4 = A3 + haccp, P4p = A3 + haccp_p")
# 실험 창의 가장 이른 날: 50일 창에서 2014-12-18(기상표 시작 2014-04-07 뒤 255일, 2026-10-19 잰 값) -> 창이 L 이면 (L − 50)일 당겨짐
_WIN_FIRST = np.datetime64("2014-12-18") - np.timedelta64(E.L - E.L_DEFAULT, "D")
_WIN_FIRST_GAP = 255 - (E.L - E.L_DEFAULT)
WXANOM_RULE = (f"기상 19개 변수(1차 기상 변수 그대로)마다 시군구 x 날짜 기상표(시군구_일별_기상.parquet, 1차 W) 위에서 "
               f"a_t = x_t − mean(x_(t−{E.WXANOM_K}) … x_(t−1)) - 앞 {E.WXANOM_K}일 평균(당일 제외, 결측은 빼고 평균). "
               f"앞 {E.WXANOM_K}일이 전부 결측이거나 x_t 가 결측이면 a_t 결측(채우지 않음). 기상표 시작 전 날은 없는 날로 봄 - 실험 창의 가장 이른 날 "
               f"{_WIN_FIRST} 이 기상표 시작 2014-04-07 뒤 {_WIN_FIRST_GAP}일이라 해당 없음. 요약: 예측하는 날 T 의 창 T−{E.L} … T−1 의 a_t 로 변수마다 평균, "
               "최근 10일(T−10 … T−1) 평균, 최대 = 19 x 3 = 57칸(결측 빼고 계산, 전부 결측이면 결측). 창 끝이 T−1 이고 a_(T−1) 의 앞 "
               f"{E.WXANOM_K}일도 T 이전이라 T 당일·이후 기상은 쓰지 않음. 계산 float64, 특성 행렬에 넣을 때 float32. "
               f"기간 {E.WXANOM_K}일은 제안 기본값(사용자 승인 흐름 2026-10-19)")


def blocks_of(groups):
    s = set()
    for g in groups:
        s.update(FG_BLOCKS[g])
    return [b for b in BLOCK_ORDER if b in s]


def plan_label(groups):
    f = any(g.startswith("F") for g in groups)
    g_ = any(g.startswith("G") for g in groups)
    return "1차 보고서 표 9 " + ("①+② 생장식" if f and g_ else ("①" if f else "② 생장식"))


# ---------------------------------------------------------------- 이름
def summary_names(vars_, L=E.L):
    nb = L // C1.BLOCK
    names = []
    for v in vars_:
        names += [f"{v}_b{k:02d}" for k in range(1, nb + 1)]
        names += [f"{v}_mean", f"{v}_std", f"{v}_min", f"{v}_max", f"{v}_slope"]
    return names


def wxanom_names():
    """<변수>_anom30_<mean|mean10|max>, 변수(1차 WEATHER_VARS 순서)마다 3칸씩."""
    return [f"{v}_anom{E.WXANOM_K}_{s}" for v in C1.WEATHER_VARS for s in WXANOM_STATS]


def ext_names(blocks):
    names = []
    for b in EXT_BLOCKS:
        if b not in blocks:
            continue
        if b == "spi":
            names += summary_names(SPI_VARS)
        elif b == "soil":
            names += list(SOIL_VARS)
        elif b == "soilmed":
            names += list(SOIL_MED_VARS)
        elif b == "food":
            names += [f"food_{c}" for c in FOOD_NUM]
        elif b == "g1":
            names += [f"g1_fT_{s}" for s in G_STATS["g1"]]
        elif b == "g2":
            names += [f"g2_{k}_{s}" for k in ["mumax"] + [f"P_{st}" for st in TIEN_STRAINS] for s in G_STATS["g2"]]
        elif b == "g3":
            names += [f"g3_P_ivanek_{s}" for s in G_STATS["g3"]]
        elif b == WXANOM:
            names += wxanom_names()
        elif b == RGWX:
            names += [f"{v}_{s}" for v in RG_WX_VARS for s in RG_STATS]
        elif b == RGSOIL:
            names += list(RG_SOIL_VARS)
        elif b == RGFOOD:
            names += list(RG_FOOD_VARS)
        elif b in HACCP_BLOCKS:
            names.append("haccp_ratio" + ("_placebo" if b.endswith("_p") else ""))
        elif b in P_BLOCKS:
            base = b[:-2] if b.endswith("_p") else b
            names.append(EXTIDX[base][2] + ("_placebo" if b.endswith("_p") else ""))
    bad = set(blocks) - set(EXT_BLOCKS)
    if bad:
        raise ValueError(f"모르는 묶음: {bad}")
    return names


def ext_cat_names(blocks):
    return [FOOD_CAT] if "food" in blocks else []


def ext_desc(blocks):
    """표 '입력' 칸. F·G 묶음만 있으면 이전과 같은 글."""
    parts = []
    fg = [b for b in BLOCK_ORDER if b in blocks]
    if fg:
        parts.append("외부·생장식[" + ", ".join(BLOCK_DESC[b] for b in fg) + "]")
    if WXANOM in blocks:
        parts.append(f"이동평균 차감 기상[{BLOCK_DESC[WXANOM]}]")
    rg = [b for b in RG_BLOCKS if b in blocks]
    if rg:
        parts.append("생장 범위[" + ", ".join(BLOCK_DESC[b] for b in rg) + "]")
    pb = [b for b in P_BLOCKS if b in blocks]
    if pb:
        parts.append("외부 지표[" + ", ".join(BLOCK_DESC[b] for b in pb) + "]")
    return " + ".join(parts)


def ext_count_desc(blocks):
    """config.json n_features_desc 용 칸 수 글. F·G 묶음만 있으면 이전과 같은 글('외부·생장식 N')."""
    parts = []
    fg = [b for b in blocks if b != WXANOM and b not in RG_BLOCKS and b not in P_BLOCKS]
    if fg:
        parts.append(f"외부·생장식 {len(ext_names(fg))}")
    if WXANOM in blocks:
        parts.append(f"이동평균 차감 기상 {len(wxanom_names())}")
    rg = [b for b in RG_BLOCKS if b in blocks]
    if rg:
        parts.append(f"생장 범위 {len(ext_names(rg))}")
    pb = [b for b in P_BLOCKS if b in blocks]
    if pb:
        parts.append(f"외부 지표 {len(ext_names(pb))}")
    return " + ".join(parts)


# ---------------------------------------------------------------- 날마다 계산하는 식 (float64, 벡터)
def silva_fT(T):
    """Silva 2020 온도 항(D6): (T−Tmax)(T−Tmin)² / {(Topt−Tmin)[(Topt−Tmin)(T−Topt) − (Topt−Tmax)(Topt+Tmin−2T)]},
    Tmin < T < Tmax 밖은 0, T 결측은 결측."""
    T = np.asarray(T, dtype=np.float64)
    a, o, b = SILVA["Tmin"], SILVA["Topt"], SILVA["Tmax"]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        f = (T - b) * (T - a) ** 2 / ((o - a) * ((o - a) * (T - o) - (o - b) * (o + a - 2.0 * T)))
    f = np.where((T > a) & (T < b), f, 0.0)
    return np.where(np.isnan(T), np.nan, f)


def silva_fpH(pH):
    """Silva 2020 pH 항(D26): (pH−pHmax)(pH−pHmin) / [(pHopt−pHmin)(pH−pHopt) − (pHopt−pHmax)(pHmin−pH)], 범위 밖 0."""
    x = np.asarray(pH, dtype=np.float64)
    a, o, b = SILVA["pHmin"], SILVA["pHopt"], SILVA["pHmax"]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        f = (x - b) * (x - a) / ((o - a) * (x - o) - (o - b) * (a - x))
    f = np.where((x > a) & (x < b), f, 0.0)
    return np.where(np.isnan(x), np.nan, f)


def silva_faw(aw):
    """Silva 2020 aw 항(D19): (aw−1.00)(aw−awmin)² / {(awopt−awmin)[(awopt−awmin)(aw−awopt) − (awopt−1.00)(awopt+awmin−2aw)]}, 범위 밖 0."""
    x = np.asarray(aw, dtype=np.float64)
    a, o, b = SILVA["awmin"], SILVA["awopt"], SILVA["awmax"]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        f = (x - b) * (x - a) ** 2 / ((o - a) * ((o - a) * (x - o) - (o - b) * (o + a - 2.0 * x)))
    f = np.where((x > a) & (x < b), f, 0.0)
    return np.where(np.isnan(x), np.nan, f)


def tien_A(T, st):
    """Tienungoon 2000 logit 의 온도 부분(D15): b0 + b1·ln(T−Tmin) + b2·ln²(T−Tmin) + b3·ln{1−exp[0.536(T−48)]}.
    T−Tmin ≤ 0 또는 1−exp[...] ≤ 0(T ≥ 48) -> −inf(= P 0, 생장 없음). T 결측 -> 결측."""
    c = TIEN[st]
    T = np.asarray(T, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x = T - c["Tmin"]
        lx = np.log(x)
        hi = 1.0 - np.exp(TIEN_FIXED["d"] * (T - TIEN_FIXED["Tmax"]))
        A = c["b0"] + c["b1"] * lx + c["b2"] * lx ** 2 + c["b3"] * np.log(hi)
    A = np.where((x > 0) & (hi > 0), A, -np.inf)
    return np.where(np.isnan(T), np.nan, A)


def tien_B(aw, pH, st):
    """Tienungoon 2000 logit 의 식품 부분(D15): b4·ln(aw−awmin) + b5·ln(1−10^(pHmin−pH)) + b6·ln²(1−10^(pHmin−pH)) (+ 젖산 항 0).
    aw−awmin ≤ 0 또는 1−10^(pHmin−pH) ≤ 0 -> −inf. aw·pH 중 하나라도 결측 -> 결측."""
    c = TIEN[st]
    aw = np.asarray(aw, dtype=np.float64)
    x = np.asarray(pH, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        y = aw - c["awmin"]
        u_in = 1.0 - 10.0 ** (c["pHmin"] - x)
        u = np.log(u_in)
        B = c["b4"] * np.log(y) + c["b5"] * u + c["b6"] * u ** 2
    B = np.where((y > 0) & (u_in > 0), B, -np.inf)
    return np.where(np.isnan(aw) | np.isnan(x), np.nan, B)


def sigmoid(z):
    """P = 1/(1+e^(−logit)) (J15). logit −inf -> 0, 결측 -> 결측."""
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def ivanek_P(rain_lag2):
    """Ivanek 2009(D41): logit(P) = −1.79 + 0.13·Precipitation.2. 입력 = 그날 기준 2일 전 강수량(mm). 결측 -> 결측."""
    return sigmoid(IVANEK["b0"] + IVANEK["b1"] * np.asarray(rain_lag2, dtype=np.float64))


# ---------------------------------------------------------------- 창 요약
def summary_any(win, L):
    """1차 features.summary_from_window 와 같은 계산(변수 수 V 만 창에서 읽음):
    변수마다 [최근부터 10일 구간 평균 L/10개, 평균, 표준편차(ddof=0), 최소, 최대, 기울기]. 결측 무시."""
    n, _, V = win.shape
    nb = L // C1.BLOCK
    m = ~np.isnan(win)
    v0 = np.where(m, win, 0.0).astype(np.float64)
    mf = m.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cnt_b = mf.reshape(n, nb, C1.BLOCK, V).sum(2)
        sum_b = v0.reshape(n, nb, C1.BLOCK, V).sum(2)
        bmean = (sum_b / cnt_b)[:, ::-1, :]
        cnt = cnt_b.sum(1)
        s1 = sum_b.sum(1)
        mean = s1 / cnt
        dev = np.where(m, win - mean[:, None, :], 0.0)
        std = np.sqrt((dev * dev).sum(1) / cnt)
        mn = np.where(m, win, np.inf).min(1).astype(np.float64)
        mx = np.where(m, win, -np.inf).max(1).astype(np.float64)
        mn[cnt == 0] = np.nan
        mx[cnt == 0] = np.nan
        x = np.arange(L, dtype=np.float64)[None, :, None]
        sx = (mf * x).sum(1)
        sxx = (mf * x * x).sum(1)
        sxy = (v0 * x).sum(1)
        slope = (cnt * sxy - sx * s1) / (cnt * sxx - sx * sx)
        slope[cnt < 2] = np.nan
    parts = np.concatenate(
        [bmean.transpose(0, 2, 1), mean[..., None], std[..., None], mn[..., None], mx[..., None], slope[..., None]],
        axis=2)
    return parts.reshape(n, V * (nb + 5)).astype(np.float32)


def g_stats(win, stats):
    """(n, L) 날마다 값 -> 요약. 결측 무시: mean = 관측일 평균, mean10 = 마지막 10일(T-10..T-1) 관측일 평균,
    sum = 관측일 합, npos = 관측일 중 값 > 0 인 날 수, max = 관측일 최대. 관측일이 0 이면 결측."""
    m = ~np.isnan(win)
    v0 = np.where(m, win, 0.0)
    cnt = m.sum(1)
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in stats:
            if s == "mean":
                r = v0.sum(1) / cnt
            elif s == "mean10":
                r = v0[:, -LAST:].sum(1) / m[:, -LAST:].sum(1)
            elif s == "sum":
                r = np.where(cnt > 0, v0.sum(1), np.nan)
            elif s == "npos":
                r = np.where(cnt > 0, ((win > 0) & m).sum(1).astype(np.float64), np.nan)
            elif s == "max":
                r = np.where(cnt > 0, np.where(m, win, -np.inf).max(1), np.nan)
            else:
                raise ValueError(s)
            out.append(r)
    return np.stack(out, axis=1)


def wx_anomaly(W, K=None):
    """기상 격자 W (S, D, V) -> a (S, D, V) float64: a[:, t] = W[:, t] − (W[:, t−K .. t−1] 중 결측 아닌 값의 평균).
    당일 제외. 앞 K 일이 전부 결측이거나 W[:, t] 가 결측이면 결측. t−k < 0(기상표 시작 전)인 날은 없는 날(결측처럼 뺌).
    더하는 순서: k = 1(전날) .. K. 변수마다 따로(메모리용)."""
    K = E.WXANOM_K if K is None else int(K)
    S, D, V = W.shape
    out = np.empty((S, D, V), dtype=np.float64)
    for v in range(V):
        x = W[:, :, v].astype(np.float64)
        ok = ~np.isnan(x)
        x0 = np.where(ok, x, 0.0)
        s = np.zeros((S, D), dtype=np.float64)
        c = np.zeros((S, D), dtype=np.float64)
        for k in range(1, min(K, D - 1) + 1):
            s[:, k:] += x0[:, :-k]
            c[:, k:] += ok[:, :-k]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, :, v] = x - s / c                     # c = 0 -> 0/0 = 결측, x 결측 -> 결측
    return out


def wx_anom_stats(win):
    """(n, L, V) 날마다 a_t (위치 0 = T-L, 위치 L-1 = T-1) -> (n, V*3) float64, 변수마다 [평균, 최근 10일(T-10..T-1) 평균, 최대].
    결측 빼고 계산, 관측일이 0 이면 결측(g_stats 의 mean·mean10·max 와 같은 정의)."""
    n, _, V = win.shape
    m = ~np.isnan(win)
    v0 = np.where(m, win, 0.0)
    cnt = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = v0.sum(1) / cnt
        mean10 = v0[:, -LAST:, :].sum(1) / m[:, -LAST:, :].sum(1)
        mx = np.where(m, win, -np.inf).max(1)
    mx = np.where(cnt > 0, mx, np.nan)
    return np.stack([mean, mean10, mx], axis=2).reshape(n, V * len(WXANOM_STATS))


def rg_stats(win):
    """(n, L, V) 날마다 생장 범위 칸 (위치 0 = T-L, 위치 L-1 = T-1) -> (n, V*2) float64, 칸마다 [평균, 최근 10일(T-10..T-1) 평균].
    결측 빼고 계산, 관측일이 0 이면 결측(g_stats 의 mean·mean10 과 같은 정의). 0/1 칸의 평균 = 범위 안(또는 밖) 날 비율."""
    n, _, V = win.shape
    m = ~np.isnan(win)
    v0 = np.where(m, win, 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = v0.sum(1) / m.sum(1)
        mean10 = v0[:, -LAST:, :].sum(1) / m[:, -LAST:, :].sum(1)
    return np.stack([mean, mean10], axis=2).reshape(n, V * len(RG_STATS))


# ---------------------------------------------------------------- 자료
def _grid(path, cols, wx):
    """시군구 x 날짜 표 -> (S, D, V) float32. 기상표(wx)와 시군구 목록·날짜(연속, 같은 시작·끝)가 같은지 확인. 결측(null)은 NaN."""
    t = pq.read_table(path, columns=["sgg_code", "date"] + cols).to_pandas()
    t["sgg_code"] = t["sgg_code"].astype(str)
    t["date"] = pd.to_datetime(t["date"]).values.astype("datetime64[D]")
    sggs = np.array(sorted(t["sgg_code"].unique()))
    dates = np.array(sorted(t["date"].unique())).astype("datetime64[D]")
    if not np.array_equal(sggs, wx.sggs):
        raise ValueError(f"{path.name}: 시군구 목록이 기상표와 다름")
    if len(dates) != wx.D or dates[0] != wx.date0 or dates[-1] != wx.date_last:
        raise ValueError(f"{path.name}: 날짜가 기상표({wx.date0}~{wx.date_last}, {wx.D}일)와 다름: {dates[0]}~{dates[-1]}, {len(dates)}일")
    if len(t) != wx.S * wx.D or t.duplicated(["sgg_code", "date"]).any():
        raise ValueError(f"{path.name}: 시군구x날짜 완전 격자가 아님")
    t = t.sort_values(["sgg_code", "date"], kind="mergesort")
    return np.ascontiguousarray(t[cols].to_numpy(dtype=np.float32).reshape(wx.S, wx.D, len(cols)))


def load_food():
    f = pq.read_table(FOOD_PATH).to_pandas()
    if f["train_prdlst_code_2"].duplicated().any():
        raise ValueError("품목3단계_식품속성: 3단계 코드 중복")
    return f


class ExtData:
    """F·G 입력용 배열. ctx(Ctx2)의 기상표 시군구·날짜 번호와 같은 번호를 쓴다."""

    def __init__(self, ctx, log=None):
        t0 = time.time()
        wx = ctx.weather
        self.SPI = _grid(SPI_PATH, SPI_VARS, wx)
        self.SOIL = _grid(SOIL_PATH, SOIL_VARS, wx)
        iT = C1.WEATHER_VARS.index("avg_temp")
        iR = C1.WEATHER_VARS.index("rain")
        T = wx.W[:, :, iT].astype(np.float64)
        R = wx.W[:, :, iR].astype(np.float64)
        self.fT = silva_fT(T)
        self.A = {st: tien_A(T, st) for st in TIEN_STRAINS}
        Rlag = np.full_like(R, np.nan)
        Rlag[:, IVANEK["lag"]:] = R[:, :-IVANEK["lag"]]          # 날 t 의 Precipitation.2 = rain(t-2)
        self.Q = ivanek_P(Rlag)
        food = load_food()
        self.food_table = food
        mp = {r.train_prdlst_code_2: (r.aw, r.ph, r.na, r.aw_label) for r in food.itertuples(index=False)}
        self.food_num, self.food_cat, self.gfood, self.B, self.food_stats = {}, {}, {}, {}, {}
        for sp in C1.SPLITS:
            p3 = ctx.sp[sp]["p3"]
            n = len(p3)
            num = np.full((n, 3), np.nan, dtype=np.float64)
            cat = np.full(n, FOOD_MISSING, dtype=object)
            unknown = 0
            for i, c in enumerate(p3):
                r = mp.get(c)
                if r is None:
                    unknown += int(c != C1.NULL_CODE)
                    continue
                num[i] = [np.nan if v is None or pd.isna(v) else float(v) for v in r[:3]]
                if r[3] is not None and not pd.isna(r[3]):
                    cat[i] = str(r[3])
            self.food_num[sp] = num
            self.food_cat[sp] = cat
            self.gfood[sp] = silva_fpH(num[:, 1]) * silva_faw(num[:, 0])
            self.B[sp] = {st: tien_B(num[:, 0], num[:, 1], st) for st in TIEN_STRAINS}
            self.food_stats[sp] = {"rows": int(n), "no_code_rows": int(np.sum(p3 == C1.NULL_CODE)),
                                   "unknown_code_rows": int(unknown),
                                   "aw_nan_rows": int(np.isnan(num[:, 0]).sum()), "ph_nan_rows": int(np.isnan(num[:, 1]).sum()),
                                   "na_nan_rows": int(np.isnan(num[:, 2]).sum()),
                                   "aw_label_missing_rows": int(np.sum(cat == FOOD_MISSING))}
        self.seconds = time.time() - t0
        if log:
            log(f"  F·G 자료 준비 {self.seconds:.1f}s: SPI {self.SPI.shape}, 토양 {self.SOIL.shape}, 식품 3단계 {len(food)}개 | "
                + "; ".join(f"{sp} 행 {v['rows']:,}(코드없음 {v['no_code_rows']}, 표에 없는 코드 {v['unknown_code_rows']}, "
                            f"aw 결측 {v['aw_nan_rows']:,}, ph 결측 {v['ph_nan_rows']:,}, na 결측 {v['na_nan_rows']:,})"
                            for sp, v in self.food_stats.items()))


def get_ext(ctx):
    X = getattr(ctx, "_fg_ext", None)
    if X is None:
        X = ExtData(ctx, getattr(ctx, "log", None))
        ctx._fg_ext = X
    return X


def get_wxanom(ctx):
    """이동평균 차감 기상 격자 (S, D, 19) float64 - ctx 의 기상표(1차 W)와 같은 시군구·날짜 번호. 처음 한 번 계산해 ctx 에 둠."""
    A = getattr(ctx, "_wxanom", None)
    if A is None:
        t0 = time.time()
        W = ctx.weather.W
        A = wx_anomaly(W, E.WXANOM_K)
        ctx._wxanom = A
        log = getattr(ctx, "log", None)
        if log:
            log(f"  이동평균 차감 기상 격자 {A.shape} (앞 {E.WXANOM_K}일, float64 {A.nbytes / 1e9:.2f} GB) {time.time() - t0:.1f}s, "
                f"결측 비율 {float(np.isnan(A).mean()):.4f} (원 기상 {float(np.isnan(W).mean()):.4f})")
    return A


class RgData:
    """Q 생장 범위 배열. ctx 의 기상표 시군구·날짜 번호와 같은 번호(_grid 가 확인). F·G 자료(ExtData)와 따로 읽음."""

    def __init__(self, ctx, log=None):
        t0 = time.time()
        wx = ctx.weather
        self.WX = _grid(RG_WX_PATH, RG_WX_VARS, wx)
        self.SOIL = _grid(RG_SOIL_PATH, RG_SOIL_VARS, wx)
        f = pq.read_table(RG_FOOD_PATH, columns=["train_prdlst_code_2"] + RG_FOOD_VARS).to_pandas()
        if f["train_prdlst_code_2"].duplicated().any():
            raise ValueError(f"{RG_FOOD_PATH.name}: 3단계 코드 중복")
        mp = {r[0]: [np.nan if v is None or pd.isna(v) else float(v) for v in r[1:]]
              for r in f.itertuples(index=False, name=None)}
        self.food, self.stats = {}, {}
        for sp in C1.SPLITS:
            p3 = ctx.sp[sp]["p3"]
            num = np.full((len(p3), len(RG_FOOD_VARS)), np.nan, dtype=np.float64)
            unknown = 0
            for i, c in enumerate(p3):
                r = mp.get(c)
                if r is None:
                    unknown += int(c != C1.NULL_CODE)
                    continue
                num[i] = r
            self.food[sp] = num
            self.stats[sp] = {"rows": int(len(p3)), "no_code_rows": int(np.sum(p3 == C1.NULL_CODE)), "unknown_code_rows": int(unknown),
                              "aw_nan_rows": int(np.isnan(num[:, 0]).sum()), "ph_nan_rows": int(np.isnan(num[:, 2]).sum())}
        self.seconds = time.time() - t0
        if log:
            log(f"  생장 범위 자료 준비 {self.seconds:.1f}s: 기상 범위 {self.WX.shape}, 토양 pH 범위 {self.SOIL.shape}, 품목 {len(f)}개 | "
                + "; ".join(f"{sp} 행 {v['rows']:,}(코드없음 {v['no_code_rows']}, 표에 없는 코드 {v['unknown_code_rows']}, "
                            f"aw 결측 {v['aw_nan_rows']:,}, ph 결측 {v['ph_nan_rows']:,})" for sp, v in self.stats.items()))


def get_rg(ctx):
    R = getattr(ctx, "_rg", None)
    if R is None:
        R = RgData(ctx, getattr(ctx, "log", None))
        ctx._rg = R
    return R


def extidx_table(base, placebo=False):
    """외부 지표 표 -> DataFrame [sgg_code(str), 자료연도, 사용가능일(datetime64[D]), v]. placebo 면 자료연도마다 시군구끼리 v 를 섞음
    (자료연도 오름차순으로 그 해 행을 sgg_code 순서로 놓고 rng.permutation - 고정 난수, 결측도 같이 섞음)."""
    path, col, _ = EXTIDX[base]
    t = pq.read_table(path, columns=["sgg_code", "자료연도", "사용가능일", col]).to_pandas()
    t["sgg_code"] = t["sgg_code"].astype(str)
    t["사용가능일"] = pd.to_datetime(t["사용가능일"]).values.astype("datetime64[D]")
    t = t.rename(columns={col: "v"}).sort_values(["자료연도", "sgg_code"], kind="mergesort").reset_index(drop=True)
    if t.duplicated(["sgg_code", "자료연도"]).any():
        raise ValueError(f"{path.name}: 시군구 x 자료연도 중복")
    t["v"] = t["v"].astype(np.float64)
    if placebo:
        rng = np.random.default_rng(EXTIDX_PLACEBO_SEED)
        for y in sorted(t["자료연도"].unique()):
            m = (t["자료연도"] == y).to_numpy()
            t.loc[m, "v"] = rng.permutation(t.loc[m, "v"].to_numpy())
    return t


def extidx_grid(t, wx):
    """(S, D) float32: G[s, t] = 그 시군구의 '사용가능일 ≤ 날짜 t − 1' 인 행 중 사용가능일이 가장 늦은 행(같으면 자료연도 큰 쪽)의 값, 없으면 결측.
    시군구 목록은 기상표와 같아야 함."""
    if set(t["sgg_code"]) != set(wx.sggs.tolist()):
        raise ValueError("외부 지표: 시군구 목록이 기상표와 다름")
    G = np.full((wx.S, wx.D), np.nan, dtype=np.float32)
    si = {s: i for i, s in enumerate(wx.sggs.tolist())}
    days = np.arange(wx.D)
    for sgg, g in t.groupby("sgg_code", sort=False):
        g = g.sort_values(["사용가능일", "자료연도"], kind="mergesort")
        a = (g["사용가능일"].to_numpy().astype("datetime64[D]") - wx.date0).astype(np.int64)    # 사용가능일의 기상표 날짜 번호
        k = np.searchsorted(a, days - 1, side="right") - 1                                       # a[k] ≤ t − 1 인 마지막 행
        v = g["v"].to_numpy()
        G[si[sgg]] = np.where(k >= 0, v[np.maximum(k, 0)], np.nan)
    return G


class PData:
    """P 외부 지표 격자(묶음마다 (S, D)). 쓰는 묶음만 읽음."""

    def __init__(self, ctx, blocks, log=None):
        t0 = time.time()
        wx = ctx.weather
        self.G = {}
        for b in blocks:
            base = b[:-2] if b.endswith("_p") else b
            self.G[b] = extidx_grid(extidx_table(base, placebo=b.endswith("_p")), wx)
        self.seconds = time.time() - t0
        if log:
            log(f"  외부 지표 격자 준비 {self.seconds:.1f}s: " + ", ".join(f"{b} {g.shape} 결측 {float(np.isnan(g).mean()):.3f}" for b, g in self.G.items()))


class HaccpData:
    """HACCP 지정 비율 격자: 업종묶음마다 R[b] (S, D) = T−1 에 유효한 그 시군구 지정 수 ÷ 전국 수, T < 사용시작일 은 결측.
    위약판 격자 Rp[b] = 달력 연도마다 시군구 순서를 섞은 R[b]. 단위별 업종묶음 번호 ub[sp] (없으면 −1)."""

    def __init__(self, ctx, log=None):
        t0 = time.time()
        wx = ctx.weather
        t = pq.read_table(HACCP_PATH, columns=["sgg_code", "업종묶음", "우리_2단계_코드목록", "시작일", "끝일", "사용시작일"]).to_pandas()
        use0 = sorted(set(pd.to_datetime(t["사용시작일"]).values.astype("datetime64[D]")))
        if len(use0) != 1:
            raise ValueError(f"{HACCP_PATH.name}: 사용시작일이 하나가 아님 {use0}")
        self.use_start = use0[0]
        code2b, bundles = {}, []
        for b, g in t.groupby("업종묶음", sort=True):
            codes = {c.strip() for L in g["우리_2단계_코드목록"].dropna() for c in str(L).split(",") if c.strip() and c.strip() != "None"}
            if b == "대응 없음" or not codes:
                continue
            bundles.append(b)
            for c in codes:
                if c in code2b:
                    raise ValueError(f"{HACCP_PATH.name}: 2단계 코드 {c} 가 업종묶음 두 곳에 있음")
                code2b[c] = len(bundles) - 1
        self.bundles, self.code2b = bundles, code2b
        si = {s: i for i, s in enumerate(wx.sggs.tolist())}
        S, Dn = wx.S, wx.D
        st = (pd.to_datetime(t["시작일"]).values.astype("datetime64[D]") - wx.date0).astype(np.int64)
        en = (pd.to_datetime(t["끝일"]).values.astype("datetime64[D]") - wx.date0).astype(np.int64)
        lo = np.clip(st + 1, 0, Dn)                    # 날짜 t 에 셈: 시작일 ≤ t−1 ≤ 끝일  <=>  t ∈ [시작일+1, 끝일+1]
        hi = np.clip(en + 2, 0, Dn)                    # 끝 다음 칸(배제)
        sidx = np.array([si.get(str(x), -1) for x in t["sgg_code"]], dtype=np.int64)
        u0 = int((self.use_start - wx.date0).astype(np.int64))
        self.R, self.Rp, self.stats = [], [], {}
        rng = np.random.default_rng(EXTIDX_PLACEBO_SEED)
        years = (wx.date0 + np.arange(Dn).astype("timedelta64[D]")).astype("datetime64[Y]").astype(int) + 1970
        perms = {y: rng.permutation(S) for y in sorted(set(years.tolist()))}
        bmask_all = t["업종묶음"].to_numpy()
        for bi, b in enumerate(bundles):
            m = (bmask_all == b) & (lo < hi)
            dC = np.zeros((S + 1, Dn + 1), dtype=np.int64)           # 마지막 행 = 시군구 없음(전국 수에만)
            rows_s = np.where(sidx[m] >= 0, sidx[m], S)
            np.add.at(dC, (rows_s, lo[m]), 1)
            np.add.at(dC, (rows_s, hi[m]), -1)
            C = np.cumsum(dC[:, :Dn], axis=1)
            N = C.sum(0).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                R = np.where(N > 0, C[:S] / N, np.nan)
            R[:, :u0] = np.nan
            Rp = np.empty_like(R)
            for y, pm in perms.items():
                cols = np.flatnonzero(years == y)
                Rp[:, cols] = R[pm][:, cols]
            self.R.append(R.astype(np.float32))
            self.Rp.append(Rp.astype(np.float32))
            self.stats[b] = {"rows": int(m.sum()), "no_sgg_rows": int((sidx[m] < 0).sum()), "codes": sorted(c for c, v in code2b.items() if v == bi)}
        self.ub = {}
        for sp in C1.SPLITS:
            p2 = ctx.rows["train_prdlst_code"].astype(str).to_numpy(dtype=object)[ctx.sp[sp]["idx"]]
            self.ub[sp] = np.array([code2b.get(c, -1) for c in p2], dtype=np.int64)
        self.seconds = time.time() - t0
        if log:
            log(f"  HACCP 격자 준비 {self.seconds:.1f}s: 업종묶음 {bundles}, 사용시작일 {self.use_start} | "
                + "; ".join(f"{sp} 묶음 없는 행 {float(np.mean(v < 0)):.1%}" for sp, v in self.ub.items()))

    def features(self, sp, p, s, t, placebo=False):
        G = self.Rp if placebo else self.R
        ub = self.ub[sp][p]
        out = np.full(len(p), np.nan, dtype=np.float32)
        for bi in range(len(G)):
            m = ub == bi
            if m.any():
                out[m] = G[bi][s[m], t[m]]
        return out


def get_haccp(ctx):
    X = getattr(ctx, "_haccp", None)
    if X is None:
        X = HaccpData(ctx, getattr(ctx, "log", None))
        ctx._haccp = X
    return X


def get_p(ctx, blocks):
    P = getattr(ctx, "_pidx", None)
    need = [b for b in blocks if b in P_BLOCKS and b not in HACCP_BLOCKS and (P is None or b not in P.G)]
    if need:
        new = PData(ctx, need, getattr(ctx, "log", None))
        if P is None:
            P = new
        else:
            P.G.update(new.G)
        ctx._pidx = P
    return P


def _win(grid, s, t, L):
    idx = t[:, None] - L + np.arange(L)[None, :]           # 시간순: 위치 0 = T-L, 위치 L-1 = T-1
    return grid[s[:, None], idx]


def ext_features(ctx, sp, pos, h, blocks, L=E.L):
    """[F·G·M 숫자 칸] float32 (n, len(ext_names(blocks)))."""
    X = get_ext(ctx) if any(b in BLOCK_ORDER for b in blocks) else None   # F·G 자료(이동평균 차감·생장 범위만이면 읽지 않음)
    AN = get_wxanom(ctx) if WXANOM in blocks else None
    RG = get_rg(ctx) if any(b in RG_BLOCKS for b in blocks) else None
    PX = get_p(ctx, blocks) if any(b in P_BLOCKS and b not in HACCP_BLOCKS for b in blocks) else None
    HX = get_haccp(ctx) if any(b in HACCP_BLOCKS for b in blocks) else None
    v = ctx.sp[sp]
    pos = np.asarray(pos)
    n = len(pos)
    hh = np.full(n, int(h), dtype=np.int64) if np.isscalar(h) else np.asarray(h, dtype=np.int64)
    out = np.empty((n, len(ext_names(blocks))), dtype=np.float32)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        p = pos[a:b]
        s = v["s"][p]
        t = v["d"][p] - hh[a:b]                              # T 의 기상표 날짜 번호
        parts = []
        for blk in EXT_BLOCKS:
            if blk not in blocks:
                continue
            if blk == "spi":
                parts.append(summary_any(_win(X.SPI, s, t, L), L))
            elif blk == "soil":
                parts.append(X.SOIL[s, t])
            elif blk == "soilmed":
                parts.append(X.SOIL[s, t][:, SOIL_MED_IDX])
            elif blk == "food":
                parts.append(X.food_num[sp][p])
            elif blk == "g1":
                parts.append(g_stats(_win(X.fT, s, t, L), G_STATS["g1"]))
            elif blk == "g2":
                fw = _win(X.fT, s, t, L)
                parts.append(g_stats(SILVA["mu_opt"] * X.gfood[sp][p][:, None] * fw, G_STATS["g2"]))
                for st in TIEN_STRAINS:
                    parts.append(g_stats(sigmoid(_win(X.A[st], s, t, L) + X.B[sp][st][p][:, None]), G_STATS["g2"]))
            elif blk == "g3":
                parts.append(g_stats(_win(X.Q, s, t, L), G_STATS["g3"]))
            elif blk == WXANOM:
                parts.append(wx_anom_stats(_win(AN, s, t, L)))
            elif blk == RGWX:
                parts.append(rg_stats(_win(RG.WX, s, t, L)))
            elif blk == RGSOIL:
                parts.append(RG.SOIL[s, t])
            elif blk == RGFOOD:
                parts.append(RG.food[sp][p])
            elif blk in HACCP_BLOCKS:
                parts.append(HX.features(sp, p, s, t, placebo=blk.endswith("_p"))[:, None])
            elif blk in P_BLOCKS:
                parts.append(PX.G[blk][s, t][:, None])
        out[a:b] = np.hstack(parts)                          # float64·float32 섞임 -> float64 로 합친 뒤 float32 칸에 넣음
    return out


def ext_cats(ctx, sp, pos, blocks):
    if "food" not in blocks:
        return None
    return get_ext(ctx).food_cat[sp][np.asarray(pos)].reshape(-1, 1)


def ext_info(blocks):
    """config.json 기록용."""
    d = {"blocks": list(blocks), "names": ext_names(blocks), "cat_names": ext_cat_names(blocks)}
    if "spi" in blocks:
        d["spi"] = f"{SPI_PATH.name} {SPI_VARS}, 창 T-{E.L}..T-1 (T 당일 제외), 기상과 같은 요약"
    if "soil" in blocks:
        d["soil"] = f"{SOIL_PATH.name} 날짜 T 행 {SOIL_VARS} (값 = exam_day ∈ [T-1095, T-1] 기록, 04_시군구_토양.py)"
    if "soilmed" in blocks:
        d["soilmed"] = f"{SOIL_PATH.name} 날짜 T 행 {SOIL_MED_VARS} (중앙값만, 검정 건수 제외; 값 = exam_day ∈ [T-1095, T-1] 기록)"
    if "food" in blocks:
        d["food"] = f"{FOOD_PATH.name} train_prdlst_code_2 -> {FOOD_NUM} 숫자(결측 그대로) + {FOOD_CAT} 범주('{FOOD_MISSING}' 포함)"
    if any(b in ("g1", "g2", "g3") for b in blocks):            # G 묶음만(지하수 'gw'·'gw_p' 는 생장식 아님 - 2026-10-19 고침, 그 전 gw 실행 config 에는 이 칸이 잘못 들어 있음)
        d["growth"] = {"silva": SILVA, "tienungoon": TIEN, "tienungoon_fixed": TIEN_FIXED, "ivanek": IVANEK,
                       "cells": GROWTH_CELLS, "domain_rule": "정의역 밖 = 생장 없음(f 0, µ 0, P 0), 입력 결측 = 결측",
                       "window": f"T-{E.L}..T-1, 최근 10일 = T-10..T-1", "limit": G_LIMIT}
    if WXANOM in blocks:
        d["wxanom"] = {"vars": list(C1.WEATHER_VARS), "K": E.WXANOM_K, "stats": list(WXANOM_STATS),
                       "window": f"T-{E.L}..T-1, 최근 10일 = T-10..T-1", "rule": WXANOM_RULE}
    rg = [b for b in RG_BLOCKS if b in blocks]
    if rg:
        d["growth_range"] = {"blocks": rg, "wx": f"{RG_WX_PATH.name} {RG_WX_VARS}" if RGWX in rg else None,
                             "soil": f"{RG_SOIL_PATH.name} {RG_SOIL_VARS} 날짜 T 행" if RGSOIL in rg else None,
                             "food": f"{RG_FOOD_PATH.name} train_prdlst_code_2 -> {RG_FOOD_VARS}" if RGFOOD in rg else None,
                             "stats": list(RG_STATS), "window": f"T-{E.L}..T-1, 최근 10일 = T-10..T-1",
                             "rule": RG_RULE, "join": RG_JOIN}
    hb = [b for b in HACCP_BLOCKS if b in blocks]
    if hb:
        d["haccp"] = {"blocks": hb, "file": HACCP_PATH.name, "placebo_seed": EXTIDX_PLACEBO_SEED, "rule": HACCP_RULE}
    pb = [b for b in P_BLOCKS if b in blocks and b not in HACCP_BLOCKS]
    if pb:
        d["external_index"] = {"blocks": pb, "files": {b: f"{EXTIDX[b.replace('_p', '')][0].name} {EXTIDX[b.replace('_p', '')][1]}" for b in pb},
                               "placebo_seed": EXTIDX_PLACEBO_SEED, "rule": P_RULE, "join": P_JOIN}
    return d


def wxanom_settings_rows():
    """표 '설정' 시트 줄(M1·A3_M1)."""
    return [("M 이동평균 차감 기상(M1·A3_M1)", WXANOM_RULE),
            ("M 칸", f"<변수>_anom{E.WXANOM_K}_<mean|mean10|max> - 1차 기상 변수 19개 순서대로 3칸씩 = {len(wxanom_names())}칸. "
                    "M1 = E00 + 57칸, A3_M1 = A3(이력 9칸, k*) + 57칸")]


def rg_settings_rows():
    """표 '설정' 시트 줄(Q1~Q4)."""
    return [("Q 생장 범위(Q1~Q4) 칸 정의", RG_RULE),
            ("Q 붙이는 방식", RG_JOIN),
            ("Q 칸 이름", f"rgwx {len(ext_names([RGWX]))}칸 <칸>_<mean|mean10>, rgsoil {len(RG_SOIL_VARS)}칸, rgfood {len(RG_FOOD_VARS)}칸, "
                       f"Q4 합 {len(ext_names(RG_BLOCKS))}칸")]


def p_settings_rows():
    """표 '설정' 시트 줄(P1·P2·P3·P1p·P2p, P4·P4p)."""
    return [("P 외부 지표(P1·P2·P3·P1p·P2p) 칸 정의", P_RULE),
            ("P 붙이는 방식·위약판", P_JOIN),
            ("P 칸 이름", ", ".join(f"{b} -> {ext_names([b])[0]}" for b in P_BLOCKS if b not in HACCP_BLOCKS)),
            ("P4 HACCP(P4·P4p)", HACCP_RULE)]


# ---------------------------------------------------------------- 조합(combo)
def combo_plan(selected):
    """selected = 검증에서 E00 보다 높았던 묶음(FG_GROUPS 순서). 돌려줌 [(실험 ID, 묶음 목록, 종류)].
    종류: each = A3 + 그 묶음(시드 3), all = A3 + 높았던 묶음 전부(시드 3), fallback = 높은 묶음 없을 때(시드 1개).
    'all' 의 입력이 어느 each 와 같으면(높은 묶음 1개, 또는 F1·F4 처럼 합집합이 한 묶음과 같음) 'all' 은 만들지 않음."""
    if not selected:
        return [("A3_F4", ["F4"], "fallback"), ("A3_G1G2G3", ["G1", "G2", "G3"], "fallback")]
    plan = [(f"A3_{g}", [g], "each") for g in selected]
    if len(selected) >= 2:
        bs = blocks_of(selected)
        if not any(blocks_of(gs) == bs for _, gs, _ in plan):
            plan.append(("A3_" + "".join(selected), list(selected), "all"))
    return plan


COMBO_RE = re.compile(r"^A3_(?:F[1-4]|G[1-3])+$")


def is_combo_id(x):
    return bool(x) and bool(COMBO_RE.match(x))


def combo_experiments(P, smoke, log=None, strict=True):
    """F1~F4·G1~G3 결과(시드 전부)로 선정 -> 조합 실험 목록. 결과가 다 없으면 strict 면 오류, 아니면 ([], 설명).
    선정: 검증 구간 가중 PR-AUC 시드 평균 > E00 시드 평균(같은 시드). 시험은 쓰지 않음. results/selection_FG.json 에 기록.
    EXP2_SELECT=tint 이면 점수 = 검증 T 통합 60일평균 PR-AUC 시드 평균(e2eval.val_tint_pr_auc, 그 밖은 같음)."""
    from e2tables import jdump, seed_stats
    seeds = E.SMOKE_SEEDS if smoke else E.SEEDS
    if E.SELECT == "tint":
        return _combo_experiments_tint(P, smoke, seeds, log, strict)
    base = seed_stats(P, "E00", seeds, "val_pr_auc_w")
    vals, missing = {}, []
    for g in FG_GROUPS:
        v = seed_stats(P, g, seeds, "val_pr_auc_w")
        if v is None:
            missing.append(g)
        else:
            vals[g] = v
    if base is None or missing:
        msg = f"조합 선정 불가: 결과가 완전하지 않음 - " + ", ".join((["E00"] if base is None else []) + missing) + f" (시드 {seeds})"
        if strict:
            raise RuntimeError(msg)
        return [], {"note": msg}
    e00m = float(np.mean(base))
    ranking = []
    for g in FG_GROUPS:
        t = seed_stats(P, g, seeds, "test_pr_auc_w")
        ranking.append({"group": g, "val_pr_auc_w_mean": float(np.mean(vals[g])), "val_pr_auc_w_by_seed": vals[g],
                        "minus_E00": float(np.mean(vals[g])) - e00m, "above_E00": bool(np.mean(vals[g]) > e00m),
                        "test_pr_auc_w_mean(기록만)": float(np.mean(t)) if t else None})
    selected = [r["group"] for r in ranking if r["above_E00"]]
    exps, plan = _combo_exps(selected)
    sel = {"rule": f"F1~F4·G1~G3 각각 검증 구간 가중 PR-AUC 시드 평균(시드 {seeds}) > E00 시드 평균 이면 '높은 묶음'. "
                   "(i) A3 + 높은 묶음 각각, (ii) A3 + 높은 묶음 전부(입력이 (i) 하나와 같으면 생략) - 시드 3개. "
                   "높은 묶음이 없으면 A3 + F4, A3 + G1~G3 전부를 시드 3개(처음엔 시드 42 하나였으나 10-19 확인을 위해 43·44 추가). 시험 미사용",
           "E00_val_pr_auc_w_mean": e00m, "E00_val_pr_auc_w_by_seed": base, "ranking": ranking, "selected": selected,
           "plan": [{"id": c, "groups": gs, "kind": k} for c, gs, k in plan]}
    jdump(sel, P["results"] / "selection_FG.json")
    if log:
        log(f"F·G 선정: E00 검증 {e00m:.5f}, 높은 묶음 {selected or '없음'} -> 조합 {[c for c, _, _ in plan]}")
    return exps, sel


def _combo_exps(selected):
    """선정된 묶음 -> (조합 실험 목록, combo_plan)."""
    plan = combo_plan(selected)
    a3 = E.EXP_BY_ID["A3"]["feats"]
    exps = []
    for cid, gs, kind in plan:
        feats = dict(a3)
        feats["ext"] = blocks_of(gs)
        what = " + ".join(gs)
        desc = f"A3 + {what} ({', '.join(BLOCK_DESC[b] for b in feats['ext'])})"
        if kind == "fallback":
            desc += (" - 대체 실행: 검증에서 E00 보다 높은 F·G 묶음 없음(시드 3개 — 10-19 시드 42 결과가 높아 43·44 추가)"
                     if E.L == E.L_DEFAULT and E.SELECT == "old" else " - 대체 실행: 검증에서 E00 보다 높은 F·G 묶음 없음(시드 3개)")
        elif kind == "all":
            desc += " - 높았던 묶음 전부"
        exps.append(dict(id=cid, stage="combo", plan=plan_label(gs), model="cat", feats=feats,
                         seeds=list(E.SEEDS), combo_groups=list(gs), combo_kind=kind, desc=desc))
    return exps, plan


def _combo_experiments_tint(P, smoke, seeds, log, strict):
    """EXP2_SELECT=tint: 점수 = 검증 T 통합 60일평균 PR-AUC 시드 평균. 높은 묶음·조합 만드는 규칙은 combo_experiments 와 같음."""
    from e2eval import TINT_RULE
    from e2tables import jdump, seed_stats, tint_seed_scores
    base = tint_seed_scores(P, "E00", seeds)
    vals, missing = {}, []
    for g in FG_GROUPS:
        v = tint_seed_scores(P, g, seeds)
        if v is None:
            missing.append(g)
        else:
            vals[g] = v
    if base is None or missing:
        msg = f"조합 선정 불가: 결과가 완전하지 않음 - " + ", ".join((["E00"] if base is None else []) + missing) + f" (시드 {seeds})"
        if strict:
            raise RuntimeError(msg)
        return [], {"note": msg}
    e00m = float(np.mean(base))
    ranking = []
    for g in FG_GROUPS:
        t = seed_stats(P, g, seeds, "test_pr_auc_w")
        w = seed_stats(P, g, seeds, "val_pr_auc_w")
        ranking.append({"group": g, "select_score_mean": float(np.mean(vals[g])), "select_score_by_seed": vals[g],
                        "minus_E00": float(np.mean(vals[g])) - e00m, "above_E00": bool(np.mean(vals[g]) > e00m),
                        "val_pr_auc_w_mean(기록만)": float(np.mean(w)) if w else None,
                        "test_pr_auc_w_mean(기록만)": float(np.mean(t)) if t else None})
    selected = [r["group"] for r in ranking if r["above_E00"]]
    exps, plan = _combo_exps(selected)
    sel = {"rule": f"F1~F4·G1~G3 각각 {TINT_RULE}(시드 {seeds})이 E00 보다 높으면 '높은 묶음'. "
                   "(i) A3 + 높은 묶음 각각, (ii) A3 + 높은 묶음 전부(입력이 (i) 하나와 같으면 생략) - 시드 3개. "
                   "높은 묶음이 없으면 A3 + F4, A3 + G1~G3 전부를 시드 3개",
           "select_rule": "tint", "select_score": "검증 T 통합 60일평균 PR-AUC",
           "E00_select_score_mean": e00m, "E00_select_score_by_seed": base, "ranking": ranking, "selected": selected,
           "plan": [{"id": c, "groups": gs, "kind": k} for c, gs, k in plan]}
    jdump(sel, P["results"] / "selection_FG.json")
    if log:
        log(f"F·G 선정(T 통합): E00 검증 {e00m:.5f}, 높은 묶음 {selected or '없음'} -> 조합 {[c for c, _, _ in plan]}")
    return exps, sel
