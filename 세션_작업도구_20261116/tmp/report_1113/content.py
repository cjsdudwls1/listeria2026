# -*- coding: utf-8 -*-
"""보고서 본문(블록 목록). 숫자는 facts.json(원 파일에서 읽은 값)으로만 채움.

블록: ("part"|"h1"|"h2"|"h3"|"o"|"d"|"b", 글, 옵션) / ("table", dict) / ("figure", dict)
글 안 {표:키}·{그림:키} 는 조립 때 번호로 바뀜, **굵게**.
"""
import json
import math
import os

W = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(W, "facts.json"), encoding="utf-8"))
R = F["runs"]
FS = F["final_scores"]


def v(run, key="PR-AUC 구간 가중 val"):
    return R[run][key]


def f4(x):
    return f"{x:.4f}"


def V(run):
    return f4(v(run))


def T(run):
    return f4(v(run, "PR-AUC 구간 가중 test"))


def n(x):
    return f"{int(x):,}"


def ro(numstr):
    """숫자 뒤 '로/으로'(끝자리 읽기: 0 영·3 삼·6 육 = 받침(ㄹ 아님) → 으로)."""
    return "으로" if numstr.rstrip("%")[-1] in "036" else "로"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


BLOCKS = []


def P(kind, text, **kw):
    BLOCKS.append((kind, text, kw))


def TBL(key, caption, header, rows, widths, aligns, pb="CELL", inline=None):
    if inline is None:
        inline = len(rows) <= 7
    BLOCKS.append(("table", dict(key=key, caption=caption, header=header, rows=rows, widths=widths,
                                 aligns=aligns, pb=("NONE" if inline else "TABLE"), inline=inline), {}))


def FIG(key, caption, png, width=45360):
    BLOCKS.append(("figure", dict(key=key, caption=caption, png=os.path.join(W, "figs", png), width=width), {}))


# ------------------------------------------------------------------ 자주 쓰는 값
N_ALL, N_POS = 676520, 1969          # 계획서 §2-1 시료 파일 기록값(단계 0 대조)
N_DOM, N_DOM_POS = 628878, 1844
N_OV, N_OV_POS = 47642, 125
N_USE = N_ALL - 361 - 9              # RR2·RR8 제외(둘 다 검출 0)
N_USE_DOM = N_DOM + 78 - 361 - 9
N_USE_OV = N_OV - 78
cv = FS["cv"]
te = FS["test"]["report"]
tm = te["threshold_metrics"]
S_final = F["summary"]["최종: 식품 갱신 + 생장식 KO"]
S_base = F["summary"]["1차: 기상+LIMS 70판"]
S_7a = F["summary"]["기준 줄: 과거 검출률만"]
val_pairs, val_pos = cv["n_pairs"], cv["n_pos"]
test_pairs, test_pos = te["n_pairs"], te["n_pos"]
flag_test = tm["tp"] + tm["fp"]
hm = {int(r["h"]): r for r in F["final_h"] if r["split"] == "test"}
rec_h = [float(hm[h]["recall"]) for h in range(1, 61)]
imp = {r["target"]: float(r["mean_I"]) for r in F["importance"]}
imp_groups = [r for r in F["importance"] if r["target_kind"] == "group"]
n_pos_groups = sum(1 for r in imp_groups if float(r["mean_I"]) > 0)
n_groups = len(imp_groups)

# ================================================================== I. 데이터 전처리 부문
P("part", "I. 데이터 전처리 부문", first=True)
P("h1", "1. 데이터 전처리")
P("h2", "1.1 원본 데이터")
P("o", "식품안전정보원 제공 시료 자료(식품의약품안전처 실험실 정보관리 시스템 검사 시료, 이하 LIMS 자료)에 "
       "국내·해외 기상, 토양, 식품 특성, 지역 지표 자료를 붙여 구성(아래 {표:src} 참고)")
P("d", "목표 변수는 시료 검사 결과의 리스테리아 검출(1)·불검출(0) 이진 분류")
P("d", "국내 시료와 해외(수입) 시료를 한 모델로 학습하고, 지역은 국내는 시군구·해외는 나라 단위로 맞춤")
P("o", f"시료 {n(N_ALL)}건 중 검출은 {n(N_POS)}건({pct(N_POS / N_ALL, 2)})으로, 1,000건에 약 3건만 검출되는 "
       f"불균형 자료임(국내 {n(N_DOM)}건·검출 {n(N_DOM_POS)}건, 해외 {n(N_OV)}건·검출 {n(N_OV_POS)}건)")
TBL("src", "수집 자료 구성",
    ["자료", "원천", "공간·시간 단위"],
    [["LIMS 자료", "식품안전정보원 제공 시료 자료\n(검사 결과·품목 2·3단계 코드·지역·기준일)",
      f"시료 {n(N_ALL)}건\n기준일 2015-04-07 ~ 2026-06-30"],
     ["국내 기상 19변수", "기상청 종관기상관측(ASOS)·농촌진흥청 농업기상·한국환경공단 에어코리아 일자료",
      "시군구 256곳 × 일\n2015-01-01 ~ 2026-10-13"],
     ["습도·이슬점", "기상청 ASOS 일자료(습도가 빈 날은 농업기상 습도)", "시군구 × 일"],
     ["해외 기상", "식품안전정보원 DB 나라별 일 기상\n(api_data_info.global_weather_info)",
      "나라(좌표 1곳) × 일\n2015-01-01 ~ 2026-06-30"],
     ["표준강수지수", "기상청 표준강수지수(SPI) 일자료 66지점, 예보 구간·해외는 일강수로 계산", "시군구·나라 × 일"],
     ["토양", "토양검정 pH·유기물·유효인산(api_data_info.soil_ph_info), 해외 토양 pH(api_data_info.global_soil_ph)",
      "시군구·나라"],
     ["식품 특성", "품목별 수분활성도·pH·염도(ba_data.t_aw_cl)", "품목 3단계"],
     ["지역 지표", "친환경 인증 면적 비율·지하수 정기수질검사 적합률", "시군구 × 연"],
     ["축사", "전체 농장 수 목록(한 시점)", "국내 시군구 33곳·해외 나라"]],
    [9000, 24000, 14411], ["C", "L", "L"], pb="NONE")

P("h2", "1.2 데이터 병합과 예측 구조")
P("o", "시료마다 지역·품목 코드와 기준일(d)을 정하고, 같은 지역의 일별 기상·토양·지역 지표와 품목의 식품 특성을 붙임")
P("d", "국내 지역은 주소로 정한 현재 시군구(256곳), 해외 지역은 제조국 등 나라 코드로 정한 나라(코드가 없으면 시료 좌표)")
P("d", "국내 기상은 날마다 그날 값이 있는 관측소 중 시군구 대표 좌표(시군구청, 섬으로 흩어진 옹진군은 시료 좌표 "
       "중앙값)에서 가까운 3곳 평균으로 만들어 한 관측소가 빈 날에도 값이 비지 않게 하고, 해외 기상은 DB 의 나라 값을 그대로 씀")
P("o", "예측 단위는 국내 시군구 × 품목 2단계 × 3단계, 해외 나라 × 품목 2단계 × 3단계이고, 모델은 검사한 시료가 "
       "검출될 확률을 냄")
P("o", "T = 예측하는 날, h = 며칠 뒤(1 ~ 60), d = T + h = 맞히려는 날(시료 기준일)로 두고, 예측 시차를 입력으로 넣은 "
       "한 모델(트리는 h·g, 딥러닝은 g)로 1 ~ 60일 뒤를 모두 예측함(아래 {그림:window} 참고)")
P("d", "일 단위 변수(기상 등)는 발표 당일 포함 11일 예보가 있다고 보고 window 끝을 E = min(d, T + 10)으로 두며, "
       "학습·검증·평가에서는 예보 칸에 실제 관측값을 넣음(완전 예보 가정)")
P("d", "h 가 11 이상이면 window 끝 다음 날부터 d 까지의 g 일(g = d - E)은 기상 입력이 없으므로 g 를 입력으로 넣음")
P("d", "window 길이는 1차 실험에서 60일로 정해 이후 모든 모델에 같게 씀")
FIG("window", "예측하는 날(T)·며칠 뒤(h)·맞히려는 날(d)과 기상 window(보기: d = 2025-11-01, window 60일)",
    "fig_window.png")
P("o", "학습 행은 시료 1건에 h 1개(시료 고유 키로 정한 무작위 값)이고, 검증·평가는 채점 기간에 드는 (시료, h) 쌍 전부임")
P("d", "시료마다 h 60개를 모두 붙이면 학습 행이 60배가 되어 메모리 한계로 h 1개만 붙임")

P("h2", "1.3 데이터 정제")
P("o", "결측 처리: 트리 모델은 결측을 채우지 않고 그대로 분할 조건에 쓰며(채운 값이 실제 관측값으로 오해되지 않게), "
       "로지스틱 회귀와 딥러닝은 학습 자료 평균·표준편차로 표준화한 뒤 결측을 0 으로 두고 결측 표시 컬럼을 함께 넣음")
P("d", "해외 시료에 없는 변수(지중온도·대기질·풍향·이슬점 등)는 '해외 구조적 결측' 컬럼 하나로 표시")
P("o", f"시료 {n(N_ALL)}건에 아래 처리를 차례로 적용해 {n(N_USE)}건(국내 {n(N_USE_DOM)}건·해외 {n(N_USE_OV)}건, "
       f"검출 {n(N_POS)}건)을 분석 대상으로 함(아래 {{표:clean}} 참고)")
TBL("clean", "시료 처리",
    ["처리", "대상", "시료 수", "이유"],
    [["국내로 옮김", "해외 시료 중 좌표가 국내 시군구에 배정되는 시료(검출 0건)", "78", "주소가 국내"],
     ["제외", "시군구를 정할 수 없는 국내 시료(검출 0건)", "361", "예측 단위를 정할 수 없음"],
     ["'코드없음' 범주", "품목 2·3단계 코드가 빈 시료", "130", "시료는 남기고 빈 코드만 한 범주로 둠"],
     ["좌표로 나라 정함", "국내로 옮긴 시료를 뺀 해외 시료 중 나라 코드가 빈 시료", "144",
      "시료 좌표가 든 나라로 정함(못 정한 시료 0건)"],
     ["제조국으로 나라 정함", "해외 좌표의 나라와 제조국이 다른 시료", "4,381", "나라는 제조국 코드를 따름"],
     ["제외", "기상 window 가 기상 자료가 없는 2014년에 닿는 국내 시료(기준일 2015-04-07 ~ 05-27, 검출 0건)", "9",
      "window 를 채울 수 없음"]],
    [8200, 19611, 5600, 14000], ["C", "L", "R", "L"], pb="NONE")

# ------------------------------------------------------------------ 2. 특성 공학
P("h1", "2. 특성 공학")
P("h2", "2.1 1차 입력 변수")
P("o", "1차 입력은 LIMS 기본 범주 5개(시도·시군구(해외는 나라)·품목 2단계·품목 3단계·국내/해외 구분), 예측 시차(트리는 "
       "h·g, 딥러닝은 g), 기상 19변수임")
P("d", "기상 19변수: 평균·최고·최저기온, 강수량, 평균·최대풍속, 풍향(sin·cos), 일조시간, 일사량, 지중온도(5·10·20·30cm·"
       "농업기상), 토양수분, 이산화질소·일산화탄소·아황산가스")
P("d", "트리 모델은 기상 window 를 요약값(10일 구간 평균과 window 평균·표준편차·최소·최대·기울기) 또는 일별 원값으로, "
       "딥러닝은 일별 시퀀스로 넣음")
P("h2", "2.2 2차 추가 변수")
P("o", "2차 입력은 1차 입력에 아래 기상 파생 변수와 그 밖의 변수를 모두 더한 것임(아래 {표:vars}·{표:vars2} 참고)")
TBL("vars", "2차 입력에 더한 기상 파생 변수(맨 아래 변수는 최종 실험에서만 추가)",
    ["변수", "컬럼 수", "내용", "넣은 이유·근거"],
    [["이동평균 차감", "19 × 3", "기상 변수마다 그날 값 - 앞 7일 평균(트리는 window 평균·최근 10일 평균·최대, 딥러닝은 일별 채널)",
      "계절 차이를 빼고 최근 변화를 보려고"],
     ["표준강수지수", "2", "SPI3·SPI6: 최근 3·6개월 누적 강수를 평년과 비교해 표준화한 값", "강수의 중장기 건습 상태"],
     ["생장식 지수", "16", "일평균기온·식품 수분활성도·pH 로 계산한 생장 적합도·성장 확률과 2일 전 강수로 계산한 검출 확률의 "
                         "window 요약", "Silva et al. (2020)\nTienungoon et al. (2000)\nIvanek et al. (2009)"],
     ["생장 범위", "19", "기온·지중온도(12)·토양 pH(3)·식품 수분활성도·pH(4)가 생장 최적·한계 범위 안인지와 최적 범위까지의 거리",
      "리스테리아 생장 조건"],
     ["단기 기상", "6", "앞 3일 평균기온, 2일 전 강수량, 2일 전 평균풍속, 앞 25일 평균 강수량, 2·3일 전 일사량 평균, "
                      "앞 7일 일교차 평균",
      "Murphy et al. (2023)\nIvanek et al. (2009)\nPang (2017)\nWeller et al. (2020)\nChandler-Khayd et al. (2023)"],
     ["습도·이슬점", "2", "하루 전 상대습도·이슬점", "Murphy et al. (2023)"],
     ["계절", "1", "d 의 달로 봄(3 ~ 5월)·여름(6 ~ 8월)·가을(9 ~ 11월)·겨울(12 ~ 2월)", "계절성"],
     ["Koutsoumanis 생장식", "6", "생육 확률(액체배지·고체배지 계수)의 window 평균·최근 10일 평균·최대", "Koutsoumanis et al. (2004)"]],
    [8000, 5400, 20611, 13400], ["C", "C", "L", "L"], inline=True)
TBL("vars2", "2차 입력에 더한 그 밖의 변수",
    ["변수", "컬럼 수", "내용", "넣은 이유·근거"],
    [["검출 이력", "19", "5년·1년 스무딩 검출률(6), 5년·1년 로그 검사 수(6), 1년 검출률 불확실성(3), WOE(4)",
      "같은 품목·지역의 과거 검출"],
     ["식품 특성", "4", "품목 3단계 대표 수분활성도(값 표기 범주 포함)·pH·염도", "품목의 물리·화학 특성"],
     ["토양", "3", "예측하는 날 전 3년 토양검정 pH·유기물·유효인산의 시군구 중앙값(해외는 pH 만)", "생산 환경"],
     ["지역 지표", "2", "친환경 인증 면적 비율·지하수 정기수질검사 적합률(최근 5개 자료연도 평균)",
      "과거 식품안전정보원 과제의 식품안전지수 지표"],
     ["축사 밀도", "1", "10km²당 전체 농장 수(국내는 자료가 있는 33개 시군구, 해외는 나라, 그 밖은 결측)",
      "Weller et al. (2020)"]],
    [8000, 5400, 20611, 13400], ["C", "C", "L", "L"], inline=True)
P("d", "검출 이력의 세 수준은 품목 3단계 전국, 지역(시군구, 해외는 나라) 전 품목, 단위(품목 3단계 × 지역)임")
P("d", "스무딩 검출률 = (검출 수 + k × 같은 기간 전체 검출률) ÷ (검사 수 + k), k = 100 으로, 검사가 적은 단위의 "
       "검출률이 0% 나 100% 로 튀지 않게 전체 검출률 쪽으로 당김")
P("d", "로그 검사 수 = log(1 + 검사 수), 1년 검출률 불확실성 = √(검출률 × (1 - 검출률) ÷ (검사 수 + k + 1)), "
       "WOE(Weight of Evidence) = 그 범주의 검출군 비율과 불검출군 비율의 로그 차이(5년, 범주 = 시도·지역·품목 "
       "2단계·품목 3단계)")
P("d", "검출 이력은 예측하는 날 직전 6월 30일까지의 시료로 1년에 한 번 계산해, 미래 검사 결과가 입력에 섞이지 않게 함")
P("d", "단기 기상·습도·이슬점의 기준일은 d(예보 범위를 넘으면 예보 마지막 날의 다음 날)임")

# ================================================================== II. 예측모델 개발 부문
P("part", "II. 예측모델 개발 부문", pagebreak=True)
P("h1", "1. 서론")
P("h2", "1.1 연구 배경")
P("o", f"리스테리아 검출은 시료의 {pct(N_POS / N_ALL, 2)}({n(N_ALL)}건 중 {n(N_POS)}건)로 드물어, 검사할 품목·지역과 "
       f"날을 미리 고르려면 단위·날짜별 검출 가능성을 예측할 필요가 있음")
P("h2", "1.2 연구 목적")
P("o", "LIMS 자료와 기상·토양·식품 특성·지역 지표로 국내 시군구·해외 나라 × 품목 2·3단계 단위마다 1 ~ 60일 뒤 "
       "각 날의 검출 확률을 예측함")
P("o", "1차 베이스라인(기상 + LIMS)에서 시작해 변수·모델 구조·결합·변수 빼기·다운사이징·식품 특성 갱신과 생장식 "
       "추가를 차례로 시험하고, 검증 점수가 오른 것만 채택해 최종 모델을 정함")

P("h1", "2. 실험 설계")
P("h2", "2.1 학습·검증·평가 나누기")
P("o", "마지막 1년(시료 기준일 2025-07-01 ~ 2026-06-30)을 평가(test) 자료로 떼고, 그 전 자료는 시간순 교차검증 "
       "3 fold 로 나눔(아래 {그림:split} 참고)")
P("d", "fold 마다 그해 6월 30일까지의 시료로 학습하고, 이어지는 7월 1일 ~ 다음 해 5월 1일의 예측하는 날(T)에 드는 "
       "(시료, h) 쌍으로 채점함")
P("d", "채점하는 T 를 5월 1일까지로 둔 것은 h = 60 이어도 맞히려는 날(d)이 다음 학습 끝(6월 30일)을 넘지 않게 하려는 것임")
inc = FS["test"]["incomplete"]
assert inc["months"] == [["overseas", "2026-06"]] and inc["n_pos"] == 0, inc
P("d", f"평가도 같은 규칙으로 예측하는 날(T) 2025-07-01 ~ 2026-05-01 의 (시료, h) 쌍을 채점하고, 자료가 덜 들어온 "
       f"해외 2026년 6월 시료의 {n(inc['n_pairs'])}쌍(검출 {inc['n_pos']})은 채점에서 뺌")
P("d", "최종 모델은 2025-06-30 까지의 자료 전체로 다시 학습해 평가 자료를 한 번 예측하고, 모든 실험 모델의 평가 점수는 "
       "기록만 하고 고르기에는 쓰지 않음")
FIG("split", "시간순 교차검증 3 fold 와 평가 자료(쌍 = (시료, h) 조합, 검출 쌍 = 검출 시료의 쌍)", "fig_split.png")

P("h2", "2.2 평가 지표")
P("o", "검출이 드문 자료라 순위 성능은 PR-AUC 로 보고, threshold 를 정한 뒤 판정 성능은 F2 를 중심으로 Recall·Precision·"
       "혼동행렬로 봄(아래 {표:metric} 참고)")
TBL("metric", "평가 지표",
    ["지표", "뜻·계산", "쓰는 곳"],
    [["PR-AUC", "확률이 높은 순으로 검출이라 판정해 갈 때의 Precision-Recall 곡선 아래 넓이(사다리꼴 적분), "
                "무작위로 고르면 검출 비율 정도", "h 마다 계산"],
     ["구간 가중 PR-AUC", "0.5 × 평균(h 1 ~ 20) + 0.3 × 평균(h 21 ~ 40) + 0.2 × 평균(h 41 ~ 60), 검증은 3 fold 평균",
      "모든 고르기·채택(주 지표)"],
     ["검출 비율 대비 배수", "구간 가중 PR-AUC ÷ 같은 가중의 검출 비율", "무작위보다 몇 배인지"],
     ["threshold", "3 fold 검증 예측을 합쳐 Recall 0.8 이상인 값 중 Precision 이 가장 큰 값", "검출 판정(평가에 그대로 씀)"],
     ["Recall", "실제 검출 중 검출로 판정한 비율", "판정 성능"],
     ["Precision", "검출로 판정한 것 중 실제 검출 비율", "판정 성능"],
     ["F2", "Recall 을 Precision 보다 중시하는 F 점수\n= 5 × Precision × Recall ÷ (4 × Precision + Recall)",
      "판정 성능(주 지표)"],
     ["TP·FN·FP·TN", "혼동행렬: 맞힌 검출·놓친 검출·잘못 판정한 불검출·맞힌 불검출", "(시료, h) 쌍 수"],
     ["상위 q% 포착률", "확률 상위 q% 쌍에 든 검출 ÷ 전체 검출", "검사량을 정했을 때"]],
    [9200, 24211, 14000], ["C", "L", "L"], pb="NONE")
P("d", "구간 가중치(0.5·0.3·0.2)는 가까운 날의 예측일수록 검사 계획에 바로 쓰여 더 중요하다고 본 것임")

P("h2", "2.3 채택 규칙과 운영 가정")
P("o", "모든 고르기·채택은 검증 구간 가중 PR-AUC 로 하고, 소수 4자리로 반올림해 그때까지의 최고 점수보다 오를 때만 채택함")
P("o", "운영 가정: 일 단위 변수는 예보가 있다고 보고(발표 당일 포함 11일), 검출 이력은 1년에 한 번 갱신하며, 모델은 "
       "해마다 6월 30일까지의 자료로 다시 학습함")

# ------------------------------------------------------------------ 3. 1차
best_lstm = max(((v(f"1-LSTM-W{w:03d}"), w) for w in range(10, 101, 10)))
P("h1", "3. 1차 실험: 기상 + LIMS 베이스라인")
P("o", "LIMS 기본 범주·예측 시차·기상 19변수만으로 window 길이와 알고리즘을 비교해, 이후 모든 실험의 비교 기준인 "
       "1차 베이스라인을 정함")
P("d", "트리 3종(CatBoost·XGBoost·LightGBM) × window 10 ~ 100일(10일 간격) × 기상 입력 2가지(요약값·일별 원값) "
       "60개와 LSTM × window 10개를 더한 70개 설정을 모두 라이브러리 기본 설정으로 학습함(아래 {그림:stage1} 참고)")
P("d", "LSTM(Long Short-Term Memory)은 기상 일별 값을 시간 순서대로 읽는 순환 신경망임")
FIG("stage1", "window 길이별 검증 구간 가중 PR-AUC(70개 설정)", "fig_stage1.png")
P("o", f"1위는 CatBoost·기상 요약값·window 60일(검증 {V('1-CAT-W060-sum')}, 평가 {T('1-CAT-W060-sum')})로 "
       f"1차 베이스라인이 되고, window 는 이후 모든 모델에 60일로 고정함")
P("d", f"LSTM 은 window {best_lstm[1]}일이 가장 높았으나({f4(best_lstm[0])}) 1위보다 낮음")

# ------------------------------------------------------------------ 4. 2차
P("h1", "4. 2차 실험: 변수 추가와 13개 모델 구조")
P("h2", "4.1 변수 준비")
P("o", "2차 입력에 쓸 검출 이력 스무딩 상수 k 와 이동평균 차감 길이 N 을 1차와 같은 기본 설정으로 골랐음")
ks = [10, 30, 50, 100, 200]
P("d", "k: 1차 베이스라인에 검출 이력만 더해 k = 10·30·50·100·200 을 비교한 검증 점수는 "
       + "·".join(V(f"2-1-k{k:03d}") for k in ks) + f"로 k = 100({V('2-1-k100')})을 고름")
P("d", "N: 이동평균 차감만 더해 N = 5·7·15·30일을 비교해 트리(CatBoost)·LSTM 모두 가장 높은 7일을 고름"
       f"(트리 {V('2-1b-T-N07')}·LSTM {V('2-1b-L-N07')})")
rt = {r["칼럼"]: r for r in F["research_table"] if r["구분"] == "칼럼"}
r_all = [r for r in F["research_table"] if r["구분"] == "리서치 변수 전체"][0]
base = v("2-0-CAT")
assert abs(base - v("1-CAT-W060-sum")) < 1e-12
up = [c for c in rt if float(rt[c]["CV 구간 가중"]) > base]
assert up == ["trange7"], up
assert [rt[c]["문헌"] for c in ("t3", "rain_lag2", "wind_lag2", "rain_mean25", "solar_23", "trange7", "rh1", "td1", "L1")] == \
    ["Murphy 2023", "Ivanek 2009", "Pang 2017", "Pang 2017", "Weller 2020", "Chandler-Khayd 2023", "Murphy 2023",
     "Murphy 2023", "Weller 2020"]
P("o", f"문헌 근거로 만든 변수 9개(단기 기상 6개·습도·이슬점·축사 밀도)는 1차 베이스라인({f4(base)})에 하나씩·함께 "
       f"넣어 비교했고, 하나씩 넣었을 때는 앞 7일 일교차 평균만 올랐으며({f4(float(rt['trange7']['CV 구간 가중']))}) "
       f"9개를 함께 넣으면 {f4(float(r_all['CV 구간 가중']))}{ro(f4(float(r_all['CV 구간 가중'])))} 낮아졌음")
P("d", "2차는 후보 변수를 모두 넣고 3차에서 변수 단위로 빼는 설계라 9개 모두 2차 입력에 포함함")

P("h2", "4.2 13개 모델 구조 비교")
P("o", "window 60일·2차 입력(변수 전부)으로 13개 구조를 비교하고 검증 1위를 2차 모델로 정함(아래 {표:struct} 참고)")
P("d", "정형 입력 6종: XGBoost·LightGBM·CatBoost·랜덤포레스트·ExtraTrees·로지스틱 회귀(기상은 window 요약값)")
P("d", "딥러닝 5종: LSTM·GRU(Gated Recurrent Unit)·트랜스포머 인코더·LSTM + attention·TCN(시간 합성곱 신경망), "
       "입력은 기상 시퀀스(60일)·"
       "정적 변수·지역·품목 임베딩")
P("d", "결합 2종: 정형 변수 Dense 갈래와 기상 시계열 갈래(LSTM + 트랜스포머 인코더)를 합친 모델과, 여기에 시계열 "
       "VAE 갈래를 더한 모델")
P("d", "부스팅 3종(XGBoost·LightGBM·CatBoost)은 이 단계부터 반복 수를 검증으로 고름: 각 fold 점수는 나머지 두 "
       "fold 의 검증 점수로 고른 반복 수로 내고(자기 fold 정답으로 고르지 않음), threshold 와 최종 재학습에는 세 fold "
       "평균으로 고른 반복 수를 씀")
STRUCT = [("정형 입력", "XGBoost", "3-0-XGB"), ("정형 입력", "LightGBM", "3-0-LGB"), ("정형 입력", "CatBoost", "3-0-CAT"),
          ("정형 입력", "랜덤포레스트", "3-0-RF"), ("정형 입력", "ExtraTrees", "3-0-ET"),
          ("정형 입력", "로지스틱 회귀", "3-0-LR"),
          ("딥러닝", "LSTM", "3-0-LSTM"), ("딥러닝", "GRU", "3-0-GRU"), ("딥러닝", "트랜스포머 인코더", "3-0-TRF"),
          ("딥러닝", "LSTM + attention", "3-0-LSTMA"), ("딥러닝", "TCN", "3-0-TCN"),
          ("결합", "결합(정형 + 시계열)", "3-0-CMB"), ("결합", "결합(정형 + 시계열 + VAE)", "3-0-CMBV")]
top = max(STRUCT, key=lambda t: v(t[2]))
assert top[2] == "3-0-LSTM"
TBL("struct", "13개 구조의 검증 구간 가중 PR-AUC(window 60일·변수 전부, 굵게 = 1위)",
    ["갈래", "구조", "검증 PR-AUC"],
    [[(g if i == 0 or STRUCT[i - 1][0] != g else None), (f"**{name}**" if run == top[2] else name),
      (f"**{V(run)}**" if run == top[2] else V(run))] for i, (g, name, run) in enumerate(STRUCT)],
    [12000, 22000, 13411], ["C", "L", "R"], inline=True)
P("o", f"1위는 LSTM(검증 {V('3-0-LSTM')}, 평가 {T('3-0-LSTM')})으로 1차 베이스라인({V('1-CAT-W060-sum')})보다 높아 "
       f"2차 모델로 채택하고 이후 실험의 비교 기준으로 삼음")
P("d", f"LSTM 은 같은 window 60일에서 1차 입력({V('1-LSTM-W060')})보다 변수 전부를 넣었을 때({V('3-0-LSTM')}) 높음")
P("d", f"딥러닝 상위 3종(LSTM {V('3-0-LSTM')}·GRU {V('3-0-GRU')}·트랜스포머 인코더 {V('3-0-TRF')})이 정형 입력 1위"
       f"(랜덤포레스트 {V('3-0-RF')})보다 높음")

# ------------------------------------------------------------------ 5. 2차 결합
P("h1", "5. 2차 결합 실험")
P("o", f"2차 모델(LSTM {V('3-0-LSTM')})을 넘는지 보려고 시계열만 쓰는 모델, 이상 점수 추가, 스태킹, 앙상블을 시험했으나 "
       f"모두 낮아 채택하지 않음(아래 {{표:comb}} 참고)")
TBL("comb", f"2차 결합 실험의 검증 구간 가중 PR-AUC(2차 모델 {V('3-0-LSTM')})",
    ["시도", "내용", "검증 PR-AUC"],
    [["시계열만 LSTM + attention", "지역·품목 입력 없이 기상 시퀀스만 넣은 LSTM + attention", V("x6-LSTMA")],
     ["시계열만 트랜스포머", "같은 입력의 트랜스포머 인코더", V("x7-D")],
     ["랜덤포레스트 + 이상 점수", "정형 입력 1위(랜덤포레스트)에 VAE 이상 점수 1컬럼 추가", V("3-1-RF-VAE")],
     ["CatBoost + 이상 점수", "정형 입력 상위 3종(랜덤포레스트·ExtraTrees·CatBoost)에 각각 이상 점수를 더한 것 중 1위",
      V("3-2-CAT")],
     ["LIMS 만 스태킹", "LIMS 범주와 예측 시차로 학습한 정형 6종의 확률을 로지스틱 회귀로 다시 합침", V("x4-L6")],
     ["트리 앙상블", "정형 입력 상위 3종의 확률 평균", V("3-3-T3")]],
    [12500, 25111, 9800], ["L", "L", "R"], pb="NONE")
assert max(v(r) for r in ("x6-LSTMA", "x7-D", "3-1-RF-VAE", "3-2-CAT", "x4-L6", "3-3-T3")) < v("3-0-LSTM")
P("d", "이상 점수는 불검출 시료만으로 학습한 변분 오토인코더(VAE, 입력을 압축했다가 다시 복원하도록 학습하는 신경망)가 "
       "입력을 잘 복원하지 못하는 정도로, 평소와 다른 조건을 점수로 나타낸 것임")
lift6 = R["x6-LSTMA"]["배수(구간 가중) val"]
lift7 = R["x7-D"]["배수(구간 가중) val"]
P("d", f"기상 시퀀스만 넣은 두 모델은 검출 비율 대비 {lift6:.1f}·{lift7:.1f}배로, 지역·품목 정보 없이 기상만으로는 "
       f"검출을 거의 가려내지 못함")
P("d", f"비교로, 이상 점수 자체가 검출을 가려내는지 보려고 이상 점수만 검출 점수로 쓴 결과 검증 {V('3-1-VAEt')}로 "
       f"2차 모델({V('3-0-LSTM')})보다 크게 낮았음(비교만)")

# ------------------------------------------------------------------ 6. 3차
P("h1", "6. 3차 실험: 변수 빼기")
P("o", f"2차 모델에서 도움이 안 되는 변수를 빼면 점수가 오르는지 permutation importance 로 확인했으나, 뺀 모델이 "
       f"검증 {V('3-0b-LSTM')}로 2차 모델({V('3-0-LSTM')})보다 낮아 채택하지 않음")
P("d", "permutation importance(Breiman, 2001)는 한 변수의 값만 행끼리 뒤섞었을 때 검증 점수가 얼마나 떨어지는지로 "
       "그 변수의 쓸모를 재는 방법이며(여러 컬럼으로 된 변수는 컬럼 전체를 함께 섞음), 트리 모델의 내장 중요도는 값의 "
       "종류가 많은 변수 쪽으로 치우칠 수 있다는 scikit-learn 안내에 따라 이 방법을 씀")
P("d", "변수 44개(기상 9개: 기온·강수·바람·햇빛·지중온도·토양수분·습도·대기질·계절, 비기상 35개: 검출 이력 19·"
       "식품 7·토양 6·지역 지표 2·축사 1)를 3 fold × 10회 섞어 평균 하락을 냈고, 평균 하락이 0 이하인 변수 4개를 "
       "한꺼번에 빼고 다시 학습함")
P("o", f"비교로, 정형 입력 1위(랜덤포레스트 {V('3-0-RF')})에서 기상 변수 8개(계절 제외)를 모두 빼니 {V('x8-RF')}로 "
       f"낮아져 기상이 예측에 기여함을 확인함")

# ------------------------------------------------------------------ 7. 다운사이징
P("h1", "7. 다운사이징")
P("o", f"검증 쌍의 검출 비율이 {pct(val_pos / val_pairs, 2)}({n(val_pairs)}쌍 중 {n(val_pos)}쌍)로 음성이 "
       f"압도적이어서, 학습 자료의 음성을 줄이면 검출 신호를 더 잘 배우는지 네 가지 방법으로 검토함(아래 {{표:down}} 참고)")
P("d", "모든 방법은 학습 자료만 줄이고 검증·평가 자료는 전부 그대로 채점함")
TBL("down", "다운사이징 방법별 적용 이유·결과·결론",
    ["방법", "적용한 이유", "검증 PR-AUC", "결론"],
    [["Theil's U 기반 다운사이징", "검출과 가장 관련이 큰 컬럼(학습 자료의 Theil's U 1위)에서 학습 기간에 검출이 한 번도 "
                                 "없던 값의 시료는 검출 가능성이 낮다고 보고 학습에서 뺌", V("4-a0-LSTM"),
      f"2차 모델({V('3-0-LSTM')})보다 크게 낮아 채택하지 않음"],
     ["60칸 출력 신경망", "h 마다 따로 맞히는 대신 예측하는 날의 입력 하나로 1 ~ 60일 뒤를 함께 내면, Theil's U 기반 "
                         "다운사이징으로 줄인 자료에서도 h 사이 정보를 나눠 배워 나을 수 있다고 봄",
      V("x2-NN60"), f"2차 모델({V('3-0-LSTM')})보다 크게 낮아 채택하지 않음"],
     ["검출 칸 음성 샘플링", "해마다 양성은 모두 두고, 음성은 그해 검출이 나온 품목·지역 칸(품목 3단계 × 지역)의 "
                            "음성에서만 양성 수의 4배를 뽑아 검출이 날 만한 조건 안에서 검출과 불검출을 가르도록 함",
      V("6-LSTM"), f"2차 모델({V('3-0-LSTM')})보다 크게 낮아 채택하지 않음"],
     ["연도별 무작위 언더샘플링", "해마다 양성은 모두 두고, 그해 음성을 양성 수의 4배만큼 무작위로 뽑아 학습 자료의 "
                                "검출 비율을 20%로 맞추면서 해마다의 검출과 여러 조건의 음성을 고르게 남김",
      f"{V('5-LSTM-KO')}\n(최종 실험)", "채택(최종 모델)"]],
    [9000, 21211, 6600, 10600], ["L", "L", "C", "L"], pb="NONE")
P("d", "Theil's U 는 한 컬럼이 검출 여부를 얼마나 설명하는지를 0 ~ 1 로 나타낸 값임")
P("o", "Theil's U 기반 다운사이징과 그 자료로 학습한 60칸 출력 신경망, 검출 칸 음성 샘플링(그해 검출이 나온 품목·지역 "
       "칸의 음성에서만 뽑음)은 검증 점수가 크게 낮아졌고, 연도별 무작위 언더샘플링(해마다 양성은 모두 두고, 그해 음성을 "
       "양성 수의 4배만큼 무작위로 뽑음)은 최종 실험에서 갱신된 식품 특성 데이터·Koutsoumanis 생장식과 함께 적용해 "
       "2차 모델을 넘었음")
h1_final = sum(FS["cv"]["folds"][f]["pr_auc_by_h"][0] for f in ("fold1", "fold2", "fold3")) / 3
x1c = [r for r in F["runs"].values() if r["판"] == "x1-LSTM"][0]
P("o", "다운사이징과 함께 비교만 한 실험(채택 대상 아님)")
P("d", f"본 실험 전에 줄이기 효과를 미리 보려고 1차 베이스라인 설정(CatBoost)에 Theil's U 기반 다운사이징을 해 본 결과 "
       f"{V('4-a-pre-CAT')}로 1차 베이스라인({V('1-CAT-W060-sum')})보다 낮았음")
P("d", f"검출이 몰린 품목만 따로 배우면 나은지 보려고 학습 기간 검출 비중 상위 5개 품목(품목 3단계)의 시료만으로 "
       f"LSTM 을 학습한 결과 {V('4-c-LSTM')}(그 5개 품목 시료로만 채점해 다른 모델과 바로 비교하지 않음)")
P("d", f"같은 5개 품목에서 기상 등 없이 LIMS 입력(범주·예측 시차)만 쓴 랜덤포레스트는 {V('4-c-L-RF')}로 같은 품목의 "
       f"LSTM({V('4-c-LSTM')})보다 낮았음")
P("d", f"1일 뒤(h = 1)만 맞히는 모델이 나은지 보려고 Theil's U 기반 다운사이징으로 줄인 자료로 h = 1 만 학습한 결과 "
       f"{f4(x1c['PR-AUC 구간 가중 val'])}{ro(f4(x1c['PR-AUC 구간 가중 val']))} 최종 모델의 {f4(h1_final)}보다 낮아 "
       f"따로 두지 않음(두 값은 h = 1 PR-AUC 라 구간 가중 값과 비교하지 않음)")

# ------------------------------------------------------------------ 8. 최종
P("h1", "8. 최종 실험: 식품 특성 갱신과 Koutsoumanis 생장식")
P("h2", "8.1 최종 모델 구성")
P("o", "최종 모델은 2차 모델(LSTM, 2차 입력 전부)을 연도별 무작위 언더샘플링으로 뽑은 학습 자료로 학습하고, 갱신된 "
       "식품 특성 데이터와 Koutsoumanis 생장식 6컬럼을 더한 것임")
P("d", "식품 특성 DB 가 갱신되어 최종 실험에서만 식품 특성 데이터를 바꾸고 Koutsoumanis 생장식 6컬럼을 더했으며, "
       "시간 관계상 앞 단계 실험은 갱신 전 데이터로 함")
P("d", "Koutsoumanis 생장식은 Koutsoumanis et al. (2004)의 리스테리아 생육 확률 식 Logit(P) = a0 + a1·t + a2·pH + "
       "a3·bw + a4·t·pH + a5·t·bw + a6·pH·bw + a7·t² + a8·pH²(t = 일평균기온, bw = √(1 - 수분활성도), 액체배지·고체배지 "
       "계수 두 벌)에 품목의 수분활성도·pH 를 넣어 날마다 계산하고 window 평균·최근 10일 평균·최대로 요약한 6컬럼임")
P("o", f"Koutsoumanis 생장식 없이 연도별 무작위 언더샘플링과 식품 특성 갱신만 적용한 모델은 검증 {V('5-LSTM-food2')}{ro(V('5-LSTM-food2'))} 2차 모델"
       f"({V('3-0-LSTM')})보다 낮았고, 생장식 6컬럼을 더한 모델이 {V('5-LSTM-KO')}로 높아 최종 모델로 채택함")

P("h2", "8.2 성능")
P("o", f"검증으로 정한 threshold({f4(tm['threshold'])})로 평가 자료를 판정한 F2 는 {f4(tm['f2'])}(Recall "
       f"{f4(tm['recall'])}·Precision {f4(tm['precision'])})이고, 구간 가중 PR-AUC 는 검증 {f4(cv['cv_band_weighted'])}"
       f"(검출 비율 대비 {cv['cv_lift_band_weighted']:.1f}배)·평가 {f4(te['pr_auc_band_weighted'])}"
       f"({te['lift_band_weighted']:.1f}배)임(아래 {{표:final}} 참고)")
P("d", "모델 출력값은 학습 자료의 검출 비율을 맞춰(연도별 무작위 언더샘플링) 학습해 실제 검출 확률보다 크게 나오므로 순위 점수로 "
       "쓰고, 검출 판정은 검증으로 정한 threshold 로 함")
P("d", f"평가 자료 {n(test_pairs)}쌍(검출 {n(test_pos)}쌍) 중 검출로 판정한 쌍은 {n(flag_test)}쌍"
       f"({pct(flag_test / test_pairs)})이고, 그 안에 검출의 {pct(tm['recall'])}({n(tm['tp'])}쌍)가 듦")
P("d", f"확률 상위 10%·20% 쌍에 평가 검출의 {pct(te['top_q_capture']['10'])}·{pct(te['top_q_capture']['20'])}가 듦")
P("d", f"평가 F2 는 1차 베이스라인({f4(S_base['F2 test'])})보다, 평가 구간 가중 PR-AUC 는 1차 베이스라인"
       f"({f4(S_base['PR-AUC 구간 가중 test'])})의 {te['pr_auc_band_weighted'] / S_base['PR-AUC 구간 가중 test']:.1f}배로 높음")
assert abs(S_final["PR-AUC 구간 가중 test"] - te["pr_auc_band_weighted"]) < 1e-12
assert abs(S_final["F2 test"] - tm["f2"]) < 1e-12
assert S_final["TP val"] == cv["threshold"]["tp"] and S_final["FP val"] == cv["threshold"]["fp"]
sv = S_final
TBL("final", "최종 모델의 검증·평가 점수(threshold 는 검증 예측으로 정해 평가에 그대로 씀)",
    ["지표", "검증(3 fold)", "평가"],
    [["채점 쌍(검출 쌍)", f"{n(val_pairs)}({n(val_pos)})", f"{n(test_pairs)}({n(test_pos)})"],
     ["구간 가중 PR-AUC", f4(sv["PR-AUC 구간 가중 val"]), f4(sv["PR-AUC 구간 가중 test"])],
     ["PR-AUC h 1 ~ 20 평균", f4(sv["PR-AUC h 1~20 val"]), f4(sv["PR-AUC h 1~20 test"])],
     ["PR-AUC h 21 ~ 40 평균", f4(sv["PR-AUC h 21~40 val"]), f4(sv["PR-AUC h 21~40 test"])],
     ["PR-AUC h 41 ~ 60 평균", f4(sv["PR-AUC h 41~60 val"]), f4(sv["PR-AUC h 41~60 test"])],
     ["검출 비율 대비 배수", f"{sv['배수(구간 가중) val']:.1f}배", f"{sv['배수(구간 가중) test']:.1f}배"],
     ["threshold", f4(sv["threshold val"]), f4(sv["threshold test"])],
     ["**F2**", f"**{f4(sv['F2 val'])}**", f"**{f4(sv['F2 test'])}**"],
     ["Recall", f4(sv["Recall val"]), f4(sv["Recall test"])],
     ["Precision", f4(sv["Precision val"]), f4(sv["Precision test"])],
     ["F1", f4(sv["F1 val"]), f4(sv["F1 test"])],
     ["TP(맞힌 검출)", n(sv["TP val"]), n(sv["TP test"])],
     ["FN(놓친 검출)", n(sv["FN val"]), n(sv["FN test"])],
     ["FP(잘못 판정한 불검출)", n(sv["FP val"]), n(sv["FP test"])],
     ["TN(맞힌 불검출)", n(sv["TN val"]), n(sv["TN test"])],
     ["상위 10% 포착률", pct(sv["상위 10% 포착 val"]), pct(sv["상위 10% 포착 test"])],
     ["상위 20% 포착률", pct(sv["상위 20% 포착 val"]), pct(sv["상위 20% 포착 test"])]],
    [17000, 15205, 15206], ["L", "R", "R"], inline=True)
assert sv["TP val"] + sv["FN val"] == val_pos and sv["TP test"] + sv["FN test"] == test_pos
assert sv["TP val"] + sv["FN val"] + sv["FP val"] + sv["TN val"] == val_pairs
assert sv["TP test"] + sv["FN test"] + sv["FP test"] + sv["TN test"] == test_pairs
P("o", f"h 별 평가 PR-AUC 는 h 1 ~ 20 평균 {f4(te['pr_auc_h01_20'])}·21 ~ 40 평균 {f4(te['pr_auc_h21_40'])}·41 ~ 60 평균 "
       f"{f4(te['pr_auc_h41_60'])}이고, h 별 평가 Recall 은 {min(rec_h):.2f} ~ {max(rec_h):.2f} 임(아래 {{그림:finalh}} 참고)")
FIG("finalh", "최종 모델의 h 별 PR-AUC(검증·평가)와 평가 Recall·Precision(threshold 는 검증으로 정한 값)", "fig_final_h.png")

P("h2", "8.3 변수 기여")
top4 = sorted(imp_groups, key=lambda r: float(r["mean_I"]), reverse=True)[:4]
NAMES = {"기온": ("기온", ""), "hist_unc1y_u": ("1년 검출률 불확실성", "단위, "),
         "hist_unc1y_p3": ("1년 검출률 불확실성", "품목, "), "hist_rate5y_p3": ("5년 검출률", "품목, ")}
assert [r["target"] for r in top4] == ["기온", "hist_unc1y_u", "hist_unc1y_p3", "hist_rate5y_p3"], [r["target"] for r in top4]


def nm(r):
    a, b = NAMES[r["target"]]
    return f"{a}({b}{f4(float(r['mean_I']))})"


P("o", "최종 모델 안에서 변수별 permutation importance(여러 컬럼으로 된 변수는 컬럼 전체를 함께 섞음, 검증 점수 평균 "
       "하락)를 보면 " + f"{nm(top4[0])}이 가장 크고, " + "·".join(nm(r) for r in top4[1:]) + " 순임(아래 {그림:imp} 참고)")
P("d", f"{n_groups}개 변수 중 {n_pos_groups}개는 섞으면 검증 점수가 떨어졌고(평균 하락이 0 보다 큼), "
       f"{n_groups - n_pos_groups}개는 0 이하임")
P("d", f"Koutsoumanis 생장식은 섞었을 때 검증 점수가 평균 {f4(imp['생육확률(Koutsoumanis)'])} 떨어짐")
FIG("imp", f"최종 모델의 변수별 permutation importance 상위 20개(전체 {n_groups}개, 품목 = 품목 3단계 전국, 지역 = 시군구"
           f"(해외는 나라) 전 품목, 단위 = 품목 3단계 × 지역)", "fig_importance.png")

# ------------------------------------------------------------------ 10. 소결
P("h1", "9. 소결")
P("h2", "9.1 요약")
P("o", f"검증 구간 가중 PR-AUC 는 1차 베이스라인 {V('1-CAT-W060-sum')} → 2차 모델 {V('3-0-LSTM')} → 최종 "
       f"{V('5-LSTM-KO')}로 올랐고, 평가 구간 가중 PR-AUC 는 {T('1-CAT-W060-sum')} → {T('3-0-LSTM')} → "
       f"{T('5-LSTM-KO')} 임(아래 {{표:sum}}·{{그림:overview}} 참고)")
TBL("sum", "단계별 점수(구간 가중 PR-AUC, 고르기는 검증으로만)",
    ["단계", "한 것", "검증", "평가"],
    [["1차 베이스라인", "LIMS 기본 범주 + 기상 19변수, CatBoost·기상 요약값·window 60일", V("1-CAT-W060-sum"),
      T("1-CAT-W060-sum")],
     ["2차 모델", "변수 전부 추가, 13개 구조 중 1위 LSTM", V("3-0-LSTM"), T("3-0-LSTM")],
     ["최종 모델", "2차 모델 + 연도별 무작위 언더샘플링 + 식품 특성 갱신 + Koutsoumanis 생장식", V("5-LSTM-KO"),
      T("5-LSTM-KO")],
     ["참고: 과거 검출률만", "학습 없이 단위의 과거 스무딩 검출률을 그대로 확률로 씀", f4(S_7a["PR-AUC 구간 가중 val"]),
      f4(S_7a["PR-AUC 구간 가중 test"])]],
    [9800, 24611, 6500, 6500], ["L", "L", "R", "R"], pb="NONE")
P("o", "2차 모델 뒤의 결합·변수 빼기 시도와 Theil's U 기반 다운사이징·60칸 출력 신경망·검출 칸 음성 샘플링은 모두 "
       "2차 모델보다 낮았음")
P("o", f"학습 없이 과거 검출률만 쓴 기준(검증 {f4(S_7a['PR-AUC 구간 가중 val'])}·평가 {f4(S_7a['PR-AUC 구간 가중 test'])})보다 "
       f"최종 모델이 검증·평가 모두 높음")
FIG("overview", "실험 순서별 검증 구간 가중 PR-AUC(채택 여부)", "fig_overview.png")

P("h2", "9.2 한계")
P("o", "학습·검증·평가 모두 예보 칸에 실제 관측값을 넣은 점수(완전 예보 가정)라 운영 때는 예보 오차만큼 달라질 수 있고, "
       "실제 예보에 없는 변수는 운영 때 따로 채워야 함")
P("o", "해외 기상은 나라마다 좌표 1곳의 DB 값만 써서 지중온도·대기질·풍향·이슬점은 해외 시료에 없음")
P("o", "window 길이(60일)는 1차 결과로 하나만 정해 모든 구조에 같게 씀")
P("o", "시료 자료에 검사·결과 등록일이 없어 결과가 늦게 등록되는 시차는 반영하지 않음")
P("h2", "9.3 운영")
P("o", "해마다 6월 30일까지의 자료로 교차검증·threshold 결정·전체 재학습을 되풀이하고, 검출 이력 변수도 1년에 한 번 다시 계산함")
P("o", "운영 전에 국내 기상·표준강수지수의 11일 예보 입력과 해외 기상의 운영 입력을 연결해야 함")

# ------------------------------------------------------------------ 참고문헌·부록
P("h1", "참고문헌")
for ref in ["Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.",
            "Chandler-Khayd et al. (2023). Frontiers in Sustainable Food Systems. doi:10.3389/fsufs.2023.1222192",
            "Ivanek et al. (2009). Applied and Environmental Microbiology. doi:10.1128/AEM.02757-08",
            "Koutsoumanis et al. (2004). doi:10.1016/j.fm.2003.11.003",
            "Murphy et al. (2023).",
            "Pang (2017).",
            "scikit-learn User Guide, 'Permutation feature importance'.",
            "Silva et al. (2020). Food Research International. doi:10.1016/j.foodres.2020.109476",
            "Tienungoon et al. (2000). doi:10.1128/AEM.66.11.4979-4987.2000",
            "Weller et al. (2020). Frontiers in Sustainable Food Systems. doi:10.3389/fsufs.2019.00124"]:
    P("o", ref)

IV = F["input_variables"]
seq = [r for r in IV if r["칼럼"].startswith("x_seq:")]
sta = [r for r in IV if r["칼럼"].startswith("x_static:")]
cat = [r for r in IV if r["칼럼"].startswith("x_cat:")]
seq_na = sum(1 for r in seq if r["결측 표시"] == "True")
sta_na = sum(1 for r in sta if r["결측 표시"] == "True")
assert len(seq) == 60 and len(sta) == 142 and len(cat) == 4 and seq_na == 20 and sta_na == 50, (len(seq), len(sta), seq_na, sta_na)


def cnt(pred, rows):
    return sum(1 for r in rows if pred(r) and r["결측 표시"] == "False")


c_wx = cnt(lambda r: r["ID"] == "WEATHER", seq)
c_anom = cnt(lambda r: r["ID"] == "WX2", seq)
c_spi = cnt(lambda r: r["ID"] == "WX1", seq)
c_base = cnt(lambda r: r["ID"] == "", sta)
c_short = cnt(lambda r: r["ID"] in ("WX5", "WX6"), sta)
c_gr = cnt(lambda r: r["ID"] == "WX3", sta)
c_rg = cnt(lambda r: r["ID"] == "WX4", sta)
c_season = cnt(lambda r: r["ID"] == "WX7", sta)
c_his = cnt(lambda r: r["ID"].startswith("HIS"), sta)
c_food = cnt(lambda r: r["ID"] == "FD1", sta)
c_soil = cnt(lambda r: r["ID"] == "SO1", sta)
c_x = cnt(lambda r: r["ID"] in ("X1", "X2"), sta)
c_l1 = cnt(lambda r: r["ID"] == "L1", sta)
c_ko = cnt(lambda r: r["ID"] == "KO1", sta)
assert c_wx + c_anom + c_spi + seq_na == 60
assert c_base + c_short + c_gr + c_rg + c_season + c_his + c_food + c_soil + c_x + c_l1 + c_ko + sta_na == 142

P("h1", "부록", pagebreak=True)
P("o", "최종 모델(LSTM)의 입력은 기상 시퀀스(60일), 정적 변수, 지역·품목 임베딩 세 갈래임(아래 {표:inputs} 참고)")
P("d", "결측 표시 컬럼은 학습 기간 자료에 결측이 있는 변수에만 붙이고, 해외 시료에만 없는 변수는 '해외 구조적 결측' "
       "컬럼 하나로 표시함")
TBL("inputs", "최종 모델 입력 변수",
    ["갈래", "변수", "컬럼 수", "원천", "갱신"],
    [["기상 시퀀스\n(60일 × 60채널)", "기상 19변수", str(c_wx), "국내 시군구·해외 나라 일 기상", "매일(window 끝까지)"],
     [None, "이동평균 차감(7일)", str(c_anom), "기상", "매일"],
     [None, "표준강수지수(SPI3·SPI6)", str(c_spi), "기상청 SPI 일자료·일강수 계산", "매일"],
     [None, "결측 표시", str(seq_na), "-", "-"],
     ["정적 변수: 기상·생장", "g ÷ 50·국내/해외·해외 구조적 결측", str(c_base), "LIMS·기상", "시료마다"],
     [None, "단기 기상·습도·이슬점", str(c_short), "기상", "매일(기준일 전날까지)"],
     [None, "생장식 지수", str(c_gr), "기상 + 식품 특성", "매일"],
     [None, "생장 범위", str(c_rg), "기상·토양·식품 특성", "매일·고정"],
     [None, "계절(봄·여름·가을·겨울)", str(c_season), "d 의 달", "매일"],
     [None, "Koutsoumanis 생장식", str(c_ko), "기상 + 식품 특성", "매일"],
     ["정적 변수: 이력·식품·환경", "검출 이력", str(c_his), "LIMS", "연 1회(T 직전 6월 30일)"],
     [None, "식품 특성(수분활성도 표기 범주 8 포함)", str(c_food), "식품 특성 표(갱신본)", "고정"],
     [None, "토양", str(c_soil), "토양검정·해외 토양 pH", "매일(T 전날까지)"],
     [None, "지역 지표", str(c_x), "친환경 인증·지하수 수질검사", "연 1회"],
     [None, "축사 밀도", str(c_l1), "축사 목록", "고정"],
     [None, "결측 표시(정적 변수 전체)", str(sta_na), "-", "-"],
     ["임베딩", "시도·지역·품목 2단계·품목 3단계", str(len(cat)), "LIMS", "시료마다"]],
    [8800, 14411, 5000, 10800, 8400], ["C", "L", "R", "L", "L"])

if __name__ == "__main__":
    print(len(BLOCKS), "blocks")
