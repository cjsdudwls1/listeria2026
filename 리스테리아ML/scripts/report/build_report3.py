# -*- coding: utf-8 -*-
"""리스테리아 위해인자 DB 수집 결과 보고 (엑셀) v3 — 사용자 수정 요청 반영
- 우리 시료(listeria_body.parquet)에 이미 있는 컬럼(조인 후 같은 값이 되는 키 포함)·아플라톡신 전용 자료 삭제
- 0_문자보고: 한글명만 표기(컬럼명 병기 요청은 사용자가 취소), 관측소 선택 방법·한계, 토양 pH 필지 예시, 컬럼별 빈 이유, 강수 빈 값 근거, 관측소 판단 근거, t_aw_cl 서사
- 1_수집가능여부: 시트명 변경, 근거 = 데이터출처 엑셀 빨간 글자(데이터명)·예측인자 PPT 순위, 판정 용어 변경
수치 = rep_stats·rep_extra·v3_evidence·v3_decomp·v3_alt·v3_check5·v3_check6·v3_rain0 (모두 원자료 측정값)"""
import sys, json, openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
sys.stdout.reconfigure(encoding='utf-8')
OUT = sys.argv[1]
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
def J(name): return json.load(open(T + '\\' + name, encoding='utf-8'))
S, E, EV, D, ALT, C5, C6, RZ = J('rep_stats.json'), J('rep_extra.json'), J('v3_evidence.json'), J('v3_decomp.json'), J('v3_alt.json'), J('v3_check5.json'), J('v3_check6.json'), J('v3_rain0.json')
src_wb = openpyxl.load_workbook(r"C:\Users\NT751\Desktop\천영진\리서치팀문서\위해요소별_중요인자_데이터출처.xlsx", data_only=True)
SRC = {r[0].row: [c.value for c in r] for r in src_wb.worksheets[0].iter_rows(min_row=2)}
def dname(rn): return ' '.join(str(SRC[rn][1]).split())
def red(*rows): return '데이터출처 엑셀 빨간 글자: ' + ' · '.join(dname(x) for x in rows)
def notred(*rows): return '데이터출처 엑셀(빨간 글자 아님): ' + ' · '.join(dname(x) for x in rows)

V = {(x['table'], x['column']): x for g in ['asos', 'nas', 'air', 'global_weather_info'] for x in S[g]}
COV = {(c['source'], c['column']): c for c in S['coverage_weather']}
LIM = {(x['table'], x['column']): x for x in S['lims_coverage']}
smp = S['sample']; so = S['coverage_overseas']; ss = S['coverage_soil']; aw = S['t_aw_cl']
wp = E['water_pollution']; fmi = E['food_moisture_info']; dse = E['distb_step_exam']; db_ = E['date_basis']
cmp_ = EV['asos_nas_rain_compare']; mon = EV['asos_rn_monthly']['rn_null_pct']; gaps = EV['station_gaps']
ex19 = EV['soil_examples_19']; ex10 = EV['soil_examples_10'][0]
A, N, K = 'kma_asos_daily_info', 'nas_agri_weather_daily_info', 'keco_airkorea_fine_dust_info'
def v(t, c): return V[(t, c)]
def cov(s, c): return COV[(s, c)]
def L(t, c): return LIM[(t, c)]
def lp(t, c, k): return L(t, c)[k]
def dc(src, c): return D[f'{src}.{c}']
def pct(x): return f"{x:.1f}%"
def n(x): return f"{int(x):,}"
def fmtd(d): return ' · '.join(f"{k} {int(x):,}" for k, x in d.items())
def top(net, sid, why=None):
    for rec in C6[f'{net}_norow_top']:
        if rec['sid'] == sid and (why is None or rec['why'] == why): return rec
    raise KeyError((net, sid, why))
def alt(net, c): return ALT[net][c]['alt_row_that_day']
WHY = {k: ALT[k]['why_pct'] for k in ['ASOS', 'NAS', 'AIR']}
FK = next(x for x in C5['nas_kind_counts_study'] if x['kind'] == 'F')
TST = ALT['nas_test_stations'][0]
nrows = smp['rows']
soil_ex = ' · '.join(x['pnu_nm'] for x in ex19)

wb = openpyxl.Workbook()
thin = Side(style='thin', color='BFBFBF'); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
H_FILL = PatternFill('solid', fgColor='DDEBF7'); T_FILL = PatternFill('solid', fgColor='1F4E78')
G_FILL = {'수집 가능': 'E2EFDA', '일부만 있음': 'FFF2CC', '대체 값': 'FFF2CC', '사용 여부 결정 필요': 'FCE4D6', '우리 시료 값 없음': 'F8CBAD', '참고': 'EDEDED'}
WRAP = Alignment(wrap_text=True, vertical='top')

def title(ws, r, text, sub=None):
    ws.cell(r, 1, text).font = Font(bold=True, size=14, color='FFFFFF')
    for c in range(1, 12): ws.cell(r, c).fill = T_FILL
    if sub:
        ws.cell(r + 1, 1, sub).font = Font(italic=True, size=9, color='595959'); return r + 3
    return r + 2
def section(ws, r, text):
    ws.cell(r, 1, text).font = Font(bold=True, size=12, color='1F4E78'); return r + 1
def bullets(ws, r, lines):
    for ln in lines:
        c = ws.cell(r, 1, ln); c.alignment = Alignment(vertical='top')
        if ln.startswith('■'): c.font = Font(bold=True, size=11, color='1F4E78')
        elif ln.startswith(' - '): c.font = Font(bold=True)
        r += 1
    return r + 1
def table(ws, r, header, rows, widths=None, judge_col=None):
    for j, h in enumerate(header, 1):
        c = ws.cell(r, j, h); c.font = Font(bold=True); c.fill = H_FILL; c.border = BORDER
        c.alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row, 1):
            c = ws.cell(r + i, j, val); c.border = BORDER; c.alignment = WRAP
            if judge_col is not None and j == judge_col + 1 and isinstance(val, str):
                for k, col in G_FILL.items():
                    if val.startswith(k): c.fill = PatternFill('solid', fgColor=col)
    if widths:
        for j, w in enumerate(widths, 1):
            cur = ws.column_dimensions[get_column_letter(j)].width or 0
            ws.column_dimensions[get_column_letter(j)].width = max(cur, w)
    return r + len(rows) + 2

STUDY = '연구기간 = 우리 시료 기준일 범위 2015-04-07 ~ 2026-06-30'
RAW_NOTE = '수치는 DB 원자료를 그대로 센 값(빈 값 채우기·단위 바꾸기·평균 내기 없음)'
NOROW = {'ASOS': '관측소 폐쇄 후·개설 전 날짜', 'NAS': '대부분 관측 시작 전 날짜', 'AIR': '측정소 설치 전 날짜'}
def rng(vals, unit='%'):
    lo, hi = min(vals), max(vals)
    return f"{lo:.1f}{unit}" if lo == hi else f"{lo:.1f}~{hi:.1f}{unit}"
def reason(src, cols, never_label='그 항목을 기록하지 않는 관측소', norow_detail=True):
    parts = []
    for key, label in [('no_row', '그날 자료 행 없음'), ('station_never_records', never_label), ('null_on_that_day', '그날 그 항목만 빈 값')]:
        vals = [dc(src, c)[key] for c in cols]
        if max(vals) >= 0.05:
            parts.append(f"{label} {rng(vals)}" + (f"({NOROW[src]})" if key == 'no_row' and norow_detail else ''))
    return ' · '.join(parts)
def present(src, cols): return rng([dc(src, c)['present'] for c in cols])

# 관측소 사례(원자료 확인값)
a176, a175 = top('ASOS', '176'), top('ASOS', '175')
a181, a296, a239 = top('ASOS', '181'), top('ASOS', '296'), top('ASOS', '239')
n031 = top('NAS', '031028F034'); k_ds, k_ss = top('AIR', '덕산읍'), top('AIR', '수신면')

# ================================================================== 0. 문자 보고 (한글명만 표기)
ws = wb.active; ws.title = '0_문자보고'
ws.column_dimensions['A'].width = 175
r = title(ws, 1, '[보고] 리스테리아 위해인자 데이터 — 식정원 DB 수집 가능 여부 · 데이터 상태 점검 결과', f'{RAW_NOTE} / {STUDY} / 영문 컬럼명은 1~3번 시트')
AS95 = ['avg_ta', 'max_ta', 'min_ta', 'avg_ws', 'max_ws', 'max_ins_ws', 'sum_sshr', 'avg_ts', 'min_tg']
NT3, NW3, CM3 = ['temp', 'hghst_artmp', 'lowst_artmp'], ['wind', 'max_wind', 'widdir'], ['avg_cm5_te', 'avg_cm10_te', 'avg_cm20_tes']
AIR3 = ['no2_value', 'co_value', 'so2_value']
gwi_no_code = 100 - so['nation_code_present_pct']; gwi_no_map = so['nation_code_present_pct'] - so['alpha3_mapped_pct']; gwi_no_day = so['alpha3_mapped_pct'] - so['global_weather_info_sameday_pct']
rz_null, rz_zero = RZ['빈 값'], RZ['0.0']
lines = [
    '■ 점검 범위',
    ' - 대상 위해인자: 「예측인자 정리 PPT」 우선순위 9개(수분활성도·온도 공동 1위, pH 3위, 염도 4위, 토양 온도 5위, 토양 수분 6위, 대기조성 7위, 토양 pH 8위, 강수 9위) + 추가 인자 5개(식품 유형·저장 및 유통 온도·포장 유형·일조·일사·수질) + 「위해요소별_중요인자_데이터출처」 엑셀의 빨간 글자 항목',
    ' - 조회 DB: 식정원 TOXIN_MODEL_PRED(조회·내려받기만 함, DB 수정 없음)',
    f" - 우리 시료: listeria_body.parquet {n(nrows)}행(국내 {n(smp['domestic'])} · 수입 {n(smp['overseas'])} · 검출 {n(smp['detected'])}). 이 파일에 이미 있는 항목(품목코드·학습용 품목명·위도/경도·법정동코드·기준일)과 아플라톡신 전용 자료는 보고에서 뺌",
    ' - 관측망 이름: 종관기상 = 기상청 종관기상관측(ASOS, 99개 관측소) / 농업기상 = 농촌진흥청 농업기상관측(219개 관측소) / 에어코리아 = 한국환경공단 대기측정소',
    '',
    '■ 1. DB에서 수집 가능한 위해인자',
    ' - 수집 가능',
    '   · 기온: 종관기상 평균·최고·최저기온, 농업기상 기온·최고·최저기온',
    '   · 강수: 종관기상 일강수량, 농업기상 일강수량',
    '   · 바람: 종관기상 평균·최대·최대순간풍속과 풍향, 농업기상 평균·최대풍속과 풍향',
    '   · 일조: 종관기상 합계일조시간(단위 시간), 농업기상 일조시간(단위 분)',
    '   · 일사: 종관기상 합계일사량, 농업기상 일사량',
    '   · 토양 온도: 종관기상 5·10·20cm 지중온도, 농업기상 지중온도',
    f"   · 토양 pH·유효인산·유기물: 토양검정 표(soil_ph_info)의 토양산도·유효인산·유기물 함량 — 필지 단위({soil_ex})",
    '   · 식품 유형: 식약처 검사 원본의 품목명·상위 품목명, 식품공전 분류표의 한글명·분류 단계·상위 품목코드, 수입식품 신고제품구분·HSK번호',
    '   · 수입 시료 기상: 해외 제조소 위치 일별 기상표(global_factory_weather)의 평균기온·강수량·풍속·일조시간(단위 초)·일사량·토양수분',
    ' - 일부만 있음',
    '   · 토양 수분: 농업기상 토양수분보정값 한 가지만 있음(빨간 글자의 10·20·30cm 깊이별 값은 없음)',
    ' - 같은 항목은 없고 대체 값만 있음',
    '   · 대기조성: 이산화탄소 농도 없음 → 이산화질소·일산화탄소·아황산가스 농도',
    '   · 저장·유통 온도: 시료의 실제 보관·유통 온도 없음 → 시료 운반 온도·시료 상태(실온/냉장/냉동)·유통기한 문구',
    '   · 포장 유형: 포장 방식(진공·가스치환) 없음 → 포장 재질',
    f"   · 수질: 수질 측정값 없음 → 수질오염원 설치시설 목록(water_pollution: 사업장명·주소·좌표, {n(wp['n'])}개소)",
    '   · 수입 시료 국가 단위 값: 나라마다 대표 지점 1곳의 기상(global_weather_info)·국가 평균 토양 pH(global_soil_ph)',
    ' - DB에 있으나 사용 여부 결정 필요',
    '   · 식품 수분활성도·pH·염도: 품목별 문헌값 표(t_aw_cl) — 값 문제는 3번 항목',
    f"   · 수분활성도·수분함량 문헌값: food_moisture_info({n(fmi['n'])}품목, 품목명 단위라 품목코드 없음)",
    ' - DB에 없음',
    '   · 수질 측정값(농업저수지·관개수), 이산화탄소 농도, 해수 염분·해상풍, 지하수 오염예측 등급·오염취약성(DRASTIC) 점수',
    '   · 깊이별 토양수분(10·20·30cm)·높이별 기온/풍향/풍속, 유역별·시군별 토양수분, 월별 미래 기후, 기상청 방재기상관측(AWS)·농업기상관측(AAOS) 자료',
    '   · 시료와 연결되는 실제 보관·유통 온도와 포장 방식, 섭취 형태(그대로 먹는 식품인지)',
    '',
    '■ 2. 우리 시료와 조인 방법',
    ' - 공통 연결',
    f"   · 시료 고유키로 DB 전처리 시료표(hazard_preprocessed_data, 리스테리아 {n(smp['db_body_rows'])}행)에 연결 → 우리 {n(nrows)}행 100% 존재",
    '   · 이 표에서 품목제조번호·영업허가번호·수입접수번호·국가코드·위도/경도를 받아 각 표로 연결',
    f"   · 기준일: 우리 기준일 = DB 기준일 100% 일치. 이 날짜는 실제 제조일 {pct(db_['제조일']/nrows*100)} · 검사 의뢰일 {pct(db_['의뢰일']/nrows*100)} · 수입 선적일 {pct(db_['선적일']/nrows*100)}로 구성",
    ' - 국내 기상·대기: 가장 가까운 관측소의 기준일 당일 값',
    '   · ① 위치: 시료 = 전처리 시료표의 위도·경도 / 관측소 = 각 관측소표의 위도·경도(종관기상은 이전 이력으로 한 관측소에 행이 여러 개면 최신 수정일 행)',
    '   · ② 후보: DB에 일자료가 한 건이라도 있는 관측소 전부(종관기상 99곳 · 농업기상 219곳 · 에어코리아 좌표가 있는 672곳). 날짜별로 운영 여부는 따지지 않음',
    f"   · ③ 선택: 시료 위치마다 모든 후보와의 지표면 거리(하버사인 공식 = 두 지점의 위도·경도로 지구 표면 최단거리 계산)를 구해 가장 가까운 1곳으로 고정 (거리 중앙값: 종관기상 {cov('ASOS','avg_ta')['dist_p50']}km · 농업기상 {cov('NAS','temp')['dist_p50']}km · 에어코리아 {cov('에어코리아','no2_value')['dist_p50']}km)",
    '   · ④ 연결: 그 관측소에서 날짜가 기준일과 같은 행의 값 1개만 붙임. 그날 값이 비어 있으면 빈 값 그대로 둠(다른 날·다른 관측소 값으로 채우지 않음)',
    f"   · 이 방식의 한계: 날짜와 무관하게 골라서, 기준일에 아직 개설 전이거나 이미 폐쇄된 관측소가 잡힌 시료가 있음(종관기상 {pct(dc('ASOS','avg_ta')['no_row'])} · 농업기상 {pct(dc('NAS','temp')['no_row'])} · 에어코리아 {pct(dc('AIR','no2_value')['no_row'])}, 3번 항목에 사례·참고 수치)",
    "   · 기간 집계 여부(결정 필요): 지금은 기준일 '하루' 값만 붙임. 기준일 이전 7일·30일 평균기온·누적강수량처럼 여러 날을 요약한 값을 만들어 붙일지는 가공이라 아직 정하지 않음",
    f" - 토양 pH: 필지 단위({soil_ex})",
    f"   · 한 행 = 한 필지의 토양검정 결과. 예) 필지고유번호 {ex19[0]['pnu_cd']} · 필지 명칭 '{ex19[0]['pnu_nm']}' · 법정동코드 {ex19[0]['legal_dong_code']} · 조사 연도 {ex19[0]['any_year']} · 토양산도 {float(ex19[0]['acid']):g}",
    f"   · 2023년 이후 621,359행은 필지고유번호 자리에 10자리 법정동코드가 들어 있음. 예) 필지고유번호 {ex10['pnu_cd']} · 필지 명칭 '{ex10['pnu_nm']}'",
    f"   · 연결: 우리 시료 법정동코드(예 4113310500) + 기준일 연도 = 토양 표 법정동코드 + 조사 연도. 같은 동·연도에 필지가 여러 개(중앙값 {ss['parcels_per_dong_year_p50']:.0f}개)라 한 값으로 줄이는 방법 결정 필요. 같은 동·연도 필지가 있는 국내 시료 {pct(ss['dong_year_pct'])}",
    ' - 수입 시료 기상: 제조소 위치 기상표의 위도·경도가 시료 위도·경도와 같은 값이라, 좌표 + 날짜 = 기준일로 정확히 연결(수입 시료 100%)',
    '',
    '■ 3. 데이터 상태(원자료): 국내 시료 중 기준일 당일 값이 있는 비율과 빈 이유',
    ' - 빈 이유는 3가지로 나눔',
    '   · 그날 자료 행 없음: 가장 가까운 관측소에 그 날짜 행 자체가 없음 → 대부분 그 날짜에 관측소가 개설 전이거나 폐쇄 후(2번의 관측소 선택 방식 때문)',
    '   · 그 항목을 기록하지 않는 관측소: 그 관측소는 연구기간 내내 그 항목이 0건(판단 방법은 아래)',
    '   · 그날 그 항목만 빈 값: 행은 있는데 그 항목만 비어 있음. DB에 결측 사유 칸이 없어 원인은 확인 불가(종관기상 강수량은 예외, 아래 근거)',
    ' - 95% 이상',
    f"   · 종관기상 평균·최고·최저기온 · 평균·최대·최대순간풍속 · 합계일조시간 · 평균 지면온도 · 최저 초상온도: {present('ASOS', AS95)}",
    f"     → 빈 이유: {reason('ASOS', AS95, norow_detail=False)}",
    f"     → 행 없음 내용(종관기상 전 항목 공통): 폐쇄된 관측소가 잡힘 {pct(WHY['ASOS']['after'])}(대구(기) 관측소는 {a176['oper_end_dt'][:10]} 운영 종료인데 대구 인근 국내 시료 {n(a176['samples'])}행의 가장 가까운 관측소 · 진도(첨찰산) {a175['oper_end_dt'][:10]} 종료) + 개설 전 {pct(WHY['ASOS']['before'])}(서청주 {a181['db_first'][:10]} · 북부산 {a296['db_first'][:10]} · 세종 {a239['db_first'][:10]}부터 자료) · 운영 중 빠진 날 {pct(WHY['ASOS']['gap'])}",
    f"   · 농업기상 일강수량 {present('NAS', ['rn'])} → 빈 이유: {reason('NAS', ['rn'], never_label='강수량을 기록하지 않는 관측소', norow_detail=False)}",
    f"     → 행 없음 내용(농업기상 전 항목 공통): 관측 시작 전 날짜 {pct(WHY['NAS']['before'])}(예: 천안시 직산읍 관측소는 {n031['db_first'][:10]}부터 자료 → 국내 시료 {n(n031['samples'])}행) · 운영 중 빠진 날 {pct(WHY['NAS']['gap'])}(절반 이상이 아래 시험용 관측소) · 폐쇄 후 {WHY['NAS']['after']:.2f}%",
    f"   · 수입 시료: 제조소 위치 기상 {pct(so['global_factory_sameday_pct'])} · 국가 단위 기상 {pct(so['global_weather_info_sameday_pct'])} → 빈 이유: 시료에 국가코드 없음 {pct(gwi_no_code)} · 3자리 국가코드로 안 바뀜 {pct(gwi_no_map)} · 그 나라 그날 값 없음 {pct(gwi_no_day)}",
    ' - 70~95%',
    f"   · 농업기상 기온·최고·최저기온 {present('NAS', NT3)} → 빈 이유: {reason('NAS', NT3, never_label='기온을 기록하지 않는 관측소', norow_detail=False)}",
    f"   · 에어코리아 이산화질소·일산화탄소·아황산가스 {present('AIR', AIR3)} → 빈 이유: {reason('AIR', AIR3, norow_detail=False)}",
    f"     → 행 없음 내용: 측정소 설치 전 날짜 {pct(WHY['AIR']['before'])}(2019~2025년에 새로 생긴 측정소가 가장 가까운 곳으로 잡힘. 예: 덕산읍 측정소 {k_ds['year']}년 설치·{k_ds['db_first'][:10]}부터 자료 → 국내 시료 {n(k_ds['samples'])}행, 수신면 {k_ss['year']}년 설치 → {n(k_ss['samples'])}행) · 운영 중 빠진 날 {pct(WHY['AIR']['gap'])}",
    f"   · 농업기상 평균·최대풍속·풍향 {present('NAS', NW3)} → 빈 이유: {reason('NAS', NW3, never_label='풍속·풍향을 기록하지 않는 관측소', norow_detail=False)}",
    f"   · 농업기상 일사량 {present('NAS', ['srqty'])} → 빈 이유: {reason('NAS', ['srqty'], never_label='일사량을 기록하지 않는 관측소', norow_detail=False)}",
    ' - 70% 미만',
    f"   · 종관기상 일강수량 {present('ASOS', ['sum_rn'])} — 0mm도 값으로 셈: 0.0mm {pct(EV['sample_asos']['sum_rn_zero_pct_of_all'])} + 0.1mm 이상 {pct(EV['sample_asos']['sum_rn_pos_pct_of_all'])}",
    f"     → 빈 이유: 그날 자료 행 없음 {pct(dc('ASOS','sum_rn')['no_row'])} · 그날 행은 있는데 강수량만 빈 값 {pct(dc('ASOS','sum_rn')['null_on_that_day'])} → 비가 오지 않은 날로 판단(근거 아래)",
    f"   · 종관기상 합계일사량 {present('ASOS', ['sum_gsr'])} → 빈 이유: {reason('ASOS', ['sum_gsr'], never_label='일사량을 기록하지 않는 관측소', norow_detail=False)}",
    f"   · 농업기상 토양수분보정값 {present('NAS', ['soil_wt'])} → 빈 이유: {reason('NAS', ['soil_wt'], never_label='토양수분을 기록하지 않는 관측소', norow_detail=False)} (2017·2018년은 DB 전체로도 값 있는 비율 30%대)",
    f"   · 농업기상 초상온도 {present('NAS', ['gr_temp'])} → 빈 이유: {reason('NAS', ['gr_temp'], never_label='초상온도를 기록하지 않는 관측소', norow_detail=False)}",
    f"   · 농업기상 일조시간 {present('NAS', ['sun_time'])} → 빈 이유: {reason('NAS', ['sun_time'], never_label='일조시간을 기록하지 않는 관측소', norow_detail=False)}",
    f"   · 종관기상 5·10·20cm 지중온도 {present('ASOS', CM3)} → 빈 이유: {reason('ASOS', CM3, never_label='지중온도를 기록하지 않는 관측소', norow_detail=False)}",
    f"   · 농업기상 지중온도 {present('NAS', ['soil_temp'])} → 빈 이유: {reason('NAS', ['soil_temp'], never_label='지중온도를 기록하지 않는 관측소', norow_detail=False)}",
    ' - (참고) 관측소를 다르게 고르면 — 둘 다 선택 방식을 바꾸는 것이라 결정 필요',
    f"   · 기준일에 자료가 있는 관측소 중 가장 가까운 곳: 종관기상 평균기온 {pct(dc('ASOS','avg_ta')['present'])}→{pct(alt('ASOS','avg_ta'))} · 일강수량 {pct(dc('ASOS','sum_rn')['present'])}→{pct(alt('ASOS','sum_rn'))} · 농업기상 기온 {pct(dc('NAS','temp')['present'])}→{pct(alt('NAS','temp'))} · 에어코리아 이산화질소 {pct(dc('AIR','no2_value')['present'])}→{pct(alt('AIR','no2_value'))}",
    f"   · 그 항목을 연구기간에 기록한 관측소 중 가장 가까운 곳: 종관기상 합계일사량 {pct(dc('ASOS','sum_gsr')['present'])}→{pct(cov('ASOS','sum_gsr')['var_sameday_pct'])}(거리 중앙값 {cov('ASOS','avg_ta')['dist_p50']}→{cov('ASOS','sum_gsr')['var_dist_p50']}km) · 농업기상 일조시간 {pct(dc('NAS','sun_time')['present'])}→{pct(cov('NAS','sun_time')['var_sameday_pct'])}({cov('NAS','temp')['dist_p50']}→{cov('NAS','sun_time')['var_dist_p50']}km) · 토양수분보정값 {pct(dc('NAS','soil_wt')['present'])}→{pct(cov('NAS','soil_wt')['var_sameday_pct'])} · 지중온도 {pct(dc('NAS','soil_temp')['present'])}→{pct(cov('NAS','soil_temp')['var_sameday_pct'])}",
    ' - 종관기상 일강수량 빈 값을 "비가 오지 않은 날"로 판단한 근거',
    f"   · 같은 날 다른 비 기록: 강수량이 빈 날({n(rz_null['days'])}일)은 1시간 최다강수량·10분 최다강수량·강수 계속시간이 각 {EV['rn_null_and_hr_1_max_rn']['hr_1_max_rn_notnull_when_rn_null']}일·{EV['rn_null_and_mi_10_max_rn']['mi_10_max_rn_notnull_when_rn_null']}일·{EV['rn_null_and_sum_rn_dur']['sum_rn_dur_notnull_when_rn_null']}일에만 있음",
    f"   · 반면 0.0mm인 날({n(rz_zero['days'])}일)은 1시간 최다강수량이 {pct(rz_zero['hr_1_max_rn_pct'])}, 강수 계속시간이 {pct(rz_zero['sum_rn_dur_pct'])} 기록돼 있음(예: 0.17시간). 0.1~1mm인 날({pct(RZ['0.1~1 미만']['hr_1_max_rn_pct'])} · {pct(RZ['0.1~1 미만']['sum_rn_dur_pct'])})과 비슷 → 0.0mm = 비가 왔지만 0.1mm 미만",
    f"   · 가까운 농업기상 관측소와 비교: 5km 이내 {cmp_['pairs']}쌍 · {n(cmp_['days'])}일에서 종관기상 강수량이 빈 날 농업기상 강수량은 0mm {cmp_['asos_null__nas_rn0_pct']}% · 1mm 이상 {cmp_['asos_null__nas_rn_ge1_pct']}%",
    f"   · 계절: 빈 날 비율이 장마철 7월 {mon['7']}%로 가장 낮고 1~3월 {min(mon['1'], mon['2'], mon['3'])}~{max(mon['1'], mon['2'], mon['3'])}%로 가장 높음",
    '   · 빈 값을 0mm로 바꿔 쓸지는 값을 채우는 가공이라 결정 필요(기상청 공식 설명 문서는 찾지 못함, 확인 권장)',
    ' - "그 항목을 기록하지 않는 관측소"로 판단한 방법',
    f"   · 종관기상: 연구기간 내내 합계일사량이 0건인 관측소 {gaps['ASOS.sum_gsr']['stations_without_value']}곳, 지중온도가 0건인 관측소 {gaps['ASOS.avg_cm5_te']['stations_without_value']}곳(99곳 중). 이 관측소들도 같은 기간 평균기온은 {gaps['ASOS.sum_gsr']['avg_ta_present_pct_in_those_rows']}~{gaps['ASOS.avg_cm5_te']['avg_ta_present_pct_in_those_rows']}% 기록 → 관측소가 멈춘 것이 아니라 그 항목만 관측하지 않는 곳으로 판단",
    f"   · 농업기상: 연구기간 내내 0건인 관측소(219곳 중) 일조시간 {gaps['NAS.sun_time']['stations_without_value']}곳 · 지중온도 {gaps['NAS.soil_temp']['stations_without_value']}곳 · 초상온도 {gaps['NAS.gr_temp']['stations_without_value']}곳 · 토양수분 {gaps['NAS.soil_wt']['stations_without_value']}곳 · 일사량 {gaps['NAS.srqty']['stations_without_value']}곳 · 풍속 {gaps['NAS.wind']['stations_without_value']}곳 · 기온 {gaps['NAS.temp']['stations_without_value']}곳",
    f"   · 예) 농업기상 관측소 코드 7번째 글자가 F인 {FK['stations']}곳(예: 031028F034 천안시 직산읍)은 기온은 {FK['temp_n']/FK['rows']*100:.1f}% 기록하지만 일조시간·토양수분·지중온도·초상온도는 DB 전체 기간 0건 (코드 글자의 뜻은 DB에 설명 없음)",
    '   · 관측소별 관측 장비 목록은 DB에 없어 기관 확인 필요',
    ' - DB 관측소 표에서 확인된 문제',
    f"   · 시험용 관측소: 농업기상 관측소표에 '{TST['name']}'(주소 정읍시 북면 (유)동방이노베이션)가 있고 일자료 {TST['daily_rows']}일치({TST['daily_first']}~{TST['daily_last']})가 들어 있음 → 국내 시료 {n(TST['domestic_samples_nearest'])}행이 이곳을 가장 가까운 농업기상 관측소로 잡음(그중 기온 값 있음 {TST['of_which_temp_present']}행). 후보에서 뺄지 결정 필요",
    '   · 종관기상 관측소표는 한 관측소에 이력 행이 여러 개(243행/105곳)이고, 이전으로 좌표가 1km 넘게 달라진 곳 15곳',
    ' - 원자료에서 이상하게 보이는 값(처리하지 않고 개수만 셈)',
    f"   · 농업기상 토양수분보정값: 50.0이 24,565건, 5.0이 11,270건 반복 · 음수 {n(v(N,'soil_wt')['rules']['음수'])}건",
    f"   · 농업기상 일사량 39.5 이상 {n(v(N,'srqty')['rules']['39.5이상'])}건(40 근처 몰림) · 농업기상 최고·최저기온 ±50 경계값 {v(N,'hghst_artmp')['rules']['±50경계값'] + v(N,'lowst_artmp')['rules']['±50경계값']}건",
    '   · 토양산도: 2021년부터 최대 9.5, 2022년부터 최소 3.5로 잘린 모양 · 2015~2020년에는 pH 14 초과 99건(최대 5,396)',
    ' - 식품 수분활성도·pH·염도 표(t_aw_cl)의 문제',
    '   · 무엇: 품목코드별 식품 수분활성도·식품 pH·염도·수분함량 문헌값 표. 예측인자 PPT 1·3·4위 인자',
    f"   · 출처: 리서치팀 「통합_수분활성도_수분함량_식품pH_염도_최종.xlsx」({aw['file_codes']}품목)를 DB에 올리면서 {aw['db_only_codes']}품목을 더함(총 {n(aw['rows'])}품목)",
    f"   · 문제 1: 파일에서 수분활성도가 빈칸이던 {aw['filled_codes']}품목에 DB에서 값이 들어감(예: 호밀 빈칸 → 0.561, 곡류 중앙값과 같음). 값 출처 표시는 대부분 'original'이라 채운 값인지 구분 안 됨",
    f"   · 문제 2: 파일과 값이 다른 {aw['changed_codes']}품목(예: 건면 0.51 → 0.15) · 우유 등 6품목은 부등호가 반대(파일 '0.98<' → DB '<0.98') · 건조 품목 등 {aw['aw_015_rows']}품목이 0.15로 같음",
    f"   · 영향: DB 값을 그대로 붙이면 국내 시료 {pct(aw['aw']['domestic'])}에 수분활성도가 생기지만 리서치팀 파일 원래 값 기준은 32.0% · 부등호가 뒤집힌 우유류 품목의 시료 11,219행",
    '   · 결정 요청: 쓴다면 리서치팀 파일 값을 기준으로 할지',
    ' - 식약처 검사 기록(LIMS)의 대체 변수는 거의 비어 있음(국내 시료 기준, 표 연결은 되지만 값이 입력되지 않음)',
    f"   · 운반 온도 {pct(lp('m_org_lims_data_info','cnvync_sttus_tp','value_domestic'))} · 시료 상태 {pct(lp('m_org_lims_data_info','splore_sttus_code','value_domestic'))} · 유통기한 문구 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','value_domestic'))}(국내 검출 시료 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','value_detected_domestic'))}) · 포장 재질 {pct(lp('t_prdlst_mnf_ledg','frmlc_mtrqlt','value_domestic'))}",
    f"   · 유통기한·포장 재질은 품목제조대장에서 오는데, 국내 시료의 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','key_match_domestic'))}만 품목제조번호로 연결됨",
    ' - 문서와 DB가 다른 곳',
    "   · 코드 분류명: 테이블정의서 'ZIM08·ZIM11·ZIM40' → DB 실제 'IM08·IM11·IM40'(운반방법·시료상태·온도단위 코드)",
    "   · 에어코리아 측정소 좌표: 정의서 'TM 좌표' → 실제 값은 위도·경도",
    '   · 테이블목록에 없는 표: 해외 제조소 기상(global_factory_weather)·수질오염원 시설(water_pollution)·식품 수분 문헌값(food_moisture_info)',
    '',
    '■ 4. DB에 없는 인자의 수집 방법',
    ' - 공공데이터포털·기상청 API허브에서 API로 받기: 농촌진흥청 농업기상 상세관측(깊이별 토양수분·높이별 기온/풍향/풍속) · 농정원 팜맵 농업기상 · 기상청 방재기상관측(AWS) · 국립수산과학원 해수 염분·해상풍 · 한국수자원조사기술원 유역별 토양수분 · 한국농어촌공사 농업저수지 수질',
    ' - 파일 내려받기: 한국농어촌공사 지하수 오염예측도·오염취약성도(SHP 지도 → 시료 좌표로 위치 조회) · 기상청 이산화탄소 농도(CSV, 월·년) · 기상청 농업기상관측(AAOS, CSV) · 한국수자원공사 토양수분(CSV, 용담 시험유역 6개 지점만)',
    ' - 사이트 직접 조회·자료 요청: 흙토람 시군별 토양유효수분율(API·파일 없음)',
    ' - 학습 변수로 부적합: 농업용 미래상세 기후도(PDF, 미래 기후변화 시나리오)',
    ' - 출처 없음(리서치팀 확인 필요): 시료의 실제 보관·유통 온도 · 포장 방식(진공·가스치환) · 섭취 형태',
    '',
    '■ 결정 요청',
    ' - 수분활성도·pH·염도 표(t_aw_cl) 사용 여부와 기준(리서치팀 파일 값 / 범위 텍스트 해석 / 상위 품목으로 찾기) · 문헌값 표(food_moisture_info) 품목명 매칭 방법',
    ' - 수입 기상 원천: 제조소 위치(global_factory_weather) vs 나라별 대표 지점(global_weather_info)',
    ' - 토양 pH: 같은 동·연도의 여러 필지를 한 값으로 줄이는 방법',
    ' - 국내 기상: ① 종관기상 일강수량 빈 값을 0mm로 볼지 ② 당일 값만 쓸지, 기간 요약할지 ③ 관측소 고르는 방식(날짜 무관 가장 가까운 곳 / 기준일에 자료가 있는 곳 중 가장 가까운 곳 / 그 항목을 기록하는 곳 중 가장 가까운 곳) ④ 시험용 관측소(동방로거테스트) 제외 여부',
    ' - 수질오염원 시설(water_pollution) 연결 방법(좌표계 확인 후 거리 계산)',
    '',
    '※ 상세: 1~4번 시트 / 원자료 파일: 리스테리아ML/data/collected/raw/',
    '※ 기존 조인 파일(listeria_joined.parquet)은 강수 빈 값 0 채움·단위 변환·기간 요약이 들어간 가공본이라 이 보고의 근거가 아님',
]
r = bullets(ws, r, lines)

# ---- 5. (추가 요청) 20260901.테이블정의서 색 표시 변수 — 기존 행 번호가 바뀌지 않도록 마지막 행 뒤에 추가
TD, TDC = J('td_status.json'), J('td_colored_rows.json')
def at(table, *cols):
    xs = [next(y for y in TDC if y['table'] == table and y['column'] == c) for c in cols]
    return f"[{xs[0]['sheet']} 시트 {'·'.join(str(x['row']) for x in xs)}행]"
ns, pr = TD['nas_station'], TD['predict']
n_yel = sum(1 for y in TDC if '노랑' in y['colors']); n_grn = sum(1 for y in TDC if '초록' in y['colors']); n_crm = sum(1 for y in TDC if '연노랑(크림)' in y['colors'])
def st(t, c): return f"{v(t, c)['stations_with_value_study']}곳"
lines_td = [
    '■ 5. (추가) 「20260901.테이블정의서」에 색으로 표시된 변수의 수집 가능 여부',
    f" - 표시 현황: 노란색 {n_yel}개 · 초록색 {n_grn}개(모두 설명 칸에 표시). 그 밖에 빨간색 1개(수분함량), ba_data 뷰 시트의 연한 노란색 {n_crm}행(판정 결과)",
    ' - 원자료: 표시된 DB 변수는 모두 원래 값 그대로 내려받아 둠(리스테리아ML/data/collected/raw/, 수분활성도 표는 data/collected/80_t_aw_cl.parquet)',
    ' - 한눈에 보기(% = 국내 시료 중 기준일 당일 값이 있는 비율)',
    f"   · 수집 가능, 95% 이상: 종관기상 합계일조시간 {pct(dc('ASOS','sum_sshr')['present'])} · 농업기상 일강수량 {pct(dc('NAS','rn')['present'])}",
    f"   · 수집 가능, 70~95%: 농업기상 평균기온 {pct(dc('NAS','temp')['present'])} · 농업기상 일사량 {pct(dc('NAS','srqty')['present'])}",
    f"   · 수집 가능하지만 70% 미만: 종관기상 합계일사량 {pct(dc('ASOS','sum_gsr')['present'])} · 농업기상 일조시간 {pct(dc('NAS','sun_time')['present'])}",
    '   · 수집 가능(위치 정보라 비율 대신 관측소 219곳 모두 값 있음): 농업기상 관측소 설치 위도·경도',
    f"   · 수집 가능하지만 리스테리아 위해인자 목록 밖: 에어코리아 미세먼지농도 {pct(dc('AIR','pm10_value')['present'])} · 농업기상 평균습도 {pct(dc('NAS','hum')['present'])}",
    '   · DB에 있으나 사용 여부 결정 필요: 수분활성도 · 식품 pH · 염도(빨간색 수분함량 포함)',
    '   · DB에 있으나 우리 시료에 쓸 수 없음: 기상청 예보 강수량',
    ' - 노란색',
    f"   · 에어코리아 미세먼지농도 {at('keco_airkorea_fine_dust_info', 'pm10_value')}: 수집 가능하지만 리스테리아 위해인자 목록 밖(데이터출처 엑셀의 같은 행에서 이산화질소·일산화탄소·아황산가스만 빨간 글자). 연구기간 값 있음 {pct(v(K,'pm10_value')['nonnull_study'])}(측정소 {v(K,'pm10_value')['stations_total']}곳 중 {st(K,'pm10_value')}), 국내 시료 {pct(dc('AIR','pm10_value')['present'])} → 빈 이유: {reason('AIR', ['pm10_value'])}",
    f"   · 종관기상 합계일조시간 {at('kma_asos_daily_info', 'sum_sshr')}: 수집 가능. 관측소 {st(A,'sum_sshr')} 모두 기록, 연구기간 {pct(v(A,'sum_sshr')['nonnull_study'])}, 국내 시료 {pct(dc('ASOS','sum_sshr')['present'])} → 빈 이유: {reason('ASOS', ['sum_sshr'])}. 문자 타입이지만 값은 모두 숫자(단위 시간, 값의 {v(A,'sum_sshr')['zero_pct_of_values']}%가 0시간)",
    f"   · 종관기상 합계일사량 {at('kma_asos_daily_info', 'sum_gsr')}: 수집 가능하지만 관측소 99곳 중 {st(A,'sum_gsr')}만 기록. 연구기간 {pct(v(A,'sum_gsr')['nonnull_study'])}, 국내 시료 {pct(dc('ASOS','sum_gsr')['present'])} → 빈 이유: {reason('ASOS', ['sum_gsr'], never_label='일사량을 기록하지 않는 관측소')}. (참고) 일사량을 기록한 관측소 중 가장 가까운 곳이면 {pct(cov('ASOS','sum_gsr')['var_sameday_pct'])}(거리 중앙값 {cov('ASOS','avg_ta')['dist_p50']}→{cov('ASOS','sum_gsr')['var_dist_p50']}km). 정의서 설명의 '일사랑'은 오타",
    f"   · 농업기상 평균기온 {at('nas_agri_weather_daily_info', 'temp')}: 수집 가능. 관측소 219곳 중 {st(N,'temp')} 기록, 연구기간 {pct(v(N,'temp')['nonnull_study'])}, 국내 시료 {pct(dc('NAS','temp')['present'])} → 빈 이유: {reason('NAS', ['temp'], never_label='기온을 기록하지 않는 관측소')}. 범위(-35~45℃) 밖 {v(N,'temp')['rules']['범위밖(-35~45)']}건(최저 {v(N,'temp')['min']:g} · 최고 {v(N,'temp')['max']:g})",
    f"   · 농업기상 평균습도 {at('nas_agri_weather_daily_info', 'hum')}: 수집 가능하지만 리스테리아 위해인자 목록 밖(습도는 예측인자 PPT·데이터출처 엑셀 빨간 글자 어디에도 없음). 관측소 219곳 중 {st(N,'hum')} 기록, 연구기간 {pct(v(N,'hum')['nonnull_study'])}, 국내 시료 {pct(dc('NAS','hum')['present'])} → 빈 이유: {reason('NAS', ['hum'], never_label='습도를 기록하지 않는 관측소')}. 습도 0% 값 {v(N,'hum')['rules']['값0']}건",
    f"   · 농업기상 일강수량 {at('nas_agri_weather_daily_info', 'rn')}: 수집 가능. 관측소 219곳 중 {st(N,'rn')} 기록, 연구기간 {pct(v(N,'rn')['nonnull_study'])}, 국내 시료 {pct(dc('NAS','rn')['present'])} → 빈 이유: {reason('NAS', ['rn'], never_label='강수량을 기록하지 않는 관측소')}. 비가 안 온 날도 0mm로 기록(값의 {v(N,'rn')['zero_pct_of_values']}%)해 정의서 설명(0을 기록하므로 빈 값은 진짜 결측)과 맞음. 490mm 이상 {v(N,'rn')['rules']['490mm이상']}건",
    f"   · 농업기상 관측소 설치 위도·경도 {at('nas_agri_weather_obs_station', 'instl_la', 'instl_lo')}: 수집 가능. 관측소 {ns['rows']}곳 모두 값 있음 · 모두 국내 범위(위도 {ns['la_min']:.1f}~{ns['la_max']:.1f}, 경도 {ns['lo_min']:.1f}~{ns['lo_max']:.1f}) · 좌표가 겹치는 관측소 없음. 기상값이 아니라 시료와 가장 가까운 관측소를 찾는 위치 정보(2번 항목). 사용 안 함으로 표시된 {ns['use_yn'].get('0', 0)}곳은 시험용 관측소(동방로거테스트)",
    f"   · 수분활성도 · 식품 pH · 염도 {at('t_aw_cl', 'aw', 'ph', 'na')}: DB에 있으나 사용 여부 결정 필요(3번 '식품 수분활성도·pH·염도 표의 문제')",
    f"     → 값 있는 품목(전체 {n(aw['rows'])}): 수분활성도 {n(aw['aw']['table_nonblank'])} · pH {n(aw['ph']['table_nonblank'])} · 염도 {n(aw['na']['table_nonblank'])} / 숫자가 아닌 범위·부등호 글자('0.91~0.98', '<0.6' 등)인 행: {aw['aw']['range_text_rows']} · {aw['ph']['range_text_rows']} · {aw['na']['range_text_rows']}",
    f"     → 국내 시료 연결(품목코드 정확 일치): 수분활성도 {pct(aw['aw']['domestic'])}(리서치팀 파일 원래 값 기준 32.0%) · pH {pct(aw['ph']['domestic'])} · 염도 {pct(aw['na']['domestic'])} / 국내 검출 시료 {pct(aw['aw']['detected_domestic'])} · {pct(aw['ph']['detected_domestic'])} · {pct(aw['na']['detected_domestic'])}",
    f"     → 수분활성도에만 DB에서 채운 값 {aw['filled_codes']}품목 · 바뀐 값 {aw['changed_codes']}품목 · 부등호가 반대인 6품목이 섞임. 정의서에는 이 표가 8열·1,262행인데 이번에 내려받은 표는 값 출처 표시 열이 더해진 9열·{n(aw['rows'])}행 → 정의서 작성(2026-09-01) 뒤 표가 바뀐 것으로 보임",
    ' - 초록색',
    f"   · 기상청 예보 강수량 {at('kma_weather_predict_info', 'precip_mm')}: DB에 있으나 우리 시료에 쓸 수 없음. {pr['date_ranges']['meas_date'][0][:10]} 발표 예보 {n(pr['rows'])}행(예보 날짜 {pr['date_ranges']['forecast_date'][0][:10]}~{pr['date_ranges']['forecast_date'][1][:10]} {pr['forecast_days']}일, {pr['grids']}개 격자)뿐이라 시료 기준일(2015-04-07~2026-06-30)과 겹치는 날이 없음",
    f"   · 농업기상 일조시간 {at('nas_agri_weather_daily_info', 'sun_time')}: 수집 가능하지만 관측소 219곳 중 {st(N,'sun_time')}만 기록(정의서 설명 '91/219'와 같음). 연구기간 {pct(v(N,'sun_time')['nonnull_study'])}, 국내 시료 {pct(dc('NAS','sun_time')['present'])} → 빈 이유: {reason('NAS', ['sun_time'], never_label='일조시간을 기록하지 않는 관측소')}. 문자 타입, 단위 분(중앙값 {v(N,'sun_time')['p50']:.0f}분). (참고) 일조시간을 기록한 관측소 중 가장 가까운 곳이면 {pct(cov('NAS','sun_time')['var_sameday_pct'])}(거리 중앙값 {cov('NAS','temp')['dist_p50']}→{cov('NAS','sun_time')['var_dist_p50']}km)",
    f"   · 농업기상 일사량 {at('nas_agri_weather_daily_info', 'srqty')}: 수집 가능. 관측소 219곳 중 {st(N,'srqty')} 기록(정의서 설명 '180/219'와 같음). 연구기간 {pct(v(N,'srqty')['nonnull_study'])}, 국내 시료 {pct(dc('NAS','srqty')['present'])} → 빈 이유: {reason('NAS', ['srqty'], never_label='일사량을 기록하지 않는 관측소')}. 39.5 이상 {n(v(N,'srqty')['rules']['39.5이상'])}건(40 근처 몰림). (참고) 일사량을 기록한 관측소 중 가장 가까운 곳이면 {pct(cov('NAS','srqty')['var_sameday_pct'])}(거리 중앙값 {cov('NAS','temp')['dist_p50']}→{cov('NAS','srqty')['var_dist_p50']}km)",
    ' - 그 밖의 색',
    f"   · 빨간색 수분함량 {at('t_aw_cl', 'mc')}: 수분활성도 표의 수분함량. 값 있는 품목 {n(aw['mc']['table_nonblank'])}, 범위·부등호 글자 {aw['mc']['range_text_rows']}행, 국내 시료 연결 {pct(aw['mc']['domestic'])}. 사용 여부는 수분활성도 표와 함께 결정",
    f"   · 연한 노란색 [ba_data 뷰 시트 {n_crm}행]: 위해요소별 판정 뷰의 '★ 검출/불검출 판정' 칸(예측 대상). 리스테리아 판정 결과는 우리 시료 파일에 이미 있어 수집 대상 아님",
]
r = bullets(ws, r, lines_td)

# ================================================================== 1. 수집 가능 여부
ws = wb.create_sheet('1_수집가능여부')
title(ws, 1, '1. 리스테리아 위해인자 — 식정원 DB 수집 가능 여부',
      '근거: "예측인자 PPT" = 예측인자 정리 ppt.jpg / "데이터출처 엑셀 빨간 글자" = 위해요소별_중요인자_데이터출처.xlsx에서 빨간 글자로 표시된 항목(뒤는 그 행의 데이터명) / 한글 컬럼명 = 테이블정의서(없으면 DB 컬럼 설명)')
ws.cell(3, 1, '판정: 수집 가능 = 위해인자와 같은 항목이 DB에 있음 / 일부만 있음 = 같은 항목이지만 깊이 등 일부만 있음 / 대체 값 = 같은 항목은 없고 성격이 가까운 다른 값 / 사용 여부 결정 필요 = DB에 있으나 값에 문제가 있어 정해야 함 / 우리 시료 값 없음 = 컬럼은 있으나 우리 시료에 연결되는 값이 없음 / 참고 = 위해인자 범위 밖 보조 컬럼').font = Font(size=9, color='595959')
r = section(ws, 5, '1-1. DB에서 수집 가능한 데이터 (영문 + 한글 컬럼명)  ※ 우리 시료 파일에 이미 있는 컬럼(조인하면 시료 값과 같아지는 좌표·날짜·법정동 키 포함)과 아플라톡신 전용 자료는 제외')
H = ['구분', '위해인자', '근거', 'DB 테이블 (스키마.테이블)', '컬럼(영문)', '컬럼(한글)', '판정', '비고']
R1 = []
def add(grp, factor, basis, table_, cols, judge, note=''):
    for i, item in enumerate(cols):
        c, k = item[0], item[1]
        j = item[2] if len(item) > 2 else judge
        R1.append([grp, factor, basis, table_, c, k, j, (item[3] if len(item) > 3 else (note if i == 0 else ''))])
AT, NT = 'api_data_info.kma_asos_daily_info', 'api_data_info.nas_agri_weather_daily_info'
OK, PART, SUB, DEC, AUX, NOV = '수집 가능', '일부만 있음', '대체 값', '사용 여부 결정 필요', '참고', '우리 시료 값 없음'
add('국내 기상(종관기상)', '기온', '예측인자 PPT 우선순위 1위(온도) · ' + red(2, 3, 4, 5, 6, 9), AT, [('avg_ta', '평균기온'), ('max_ta', '최고기온'), ('min_ta', '최저기온')], OK, '기상청 종관기상관측(ASOS) 일자료 99개 관측소. 빨간 글자의 방재기상관측(AWS)·농업기상 관측 조회(AAOS) 자료 자체는 DB에 없어 종관기상·농업기상으로 대신함')
add('국내 기상(농업기상)', '기온', '예측인자 PPT 우선순위 1위(온도) · ' + red(5, 6, 9), NT, [('temp', '기온(℃)'), ('hghst_artmp', '최고기온'), ('lowst_artmp', '최저기온')], OK, '농촌진흥청 농업기상 일자료 219개 관측소. 측정 높이 표기 없음')
add('국내 기상(종관기상)', '강수', '예측인자 PPT 우선순위 9위(강수) · ' + red(2, 3, 5, 6, 9), AT, [('sum_rn', '일강수량')], OK, '빈 값은 비가 오지 않은 날로 판단(3번 시트), 0.0은 0.1mm 미만 강수')
add('국내 기상(농업기상)', '강수', '예측인자 PPT 우선순위 9위(강수) · ' + red(5, 6, 9), NT, [('rn', '일강수량')], OK, '비가 안 온 날도 0으로 기록')
add('국내 기상(종관기상)', '바람(풍속·풍향)', red(2, 3, 4, 5, 6, 9), AT, [('avg_ws', '평균풍속'), ('max_ws', '최대풍속'), ('max_ins_ws', '최대순간풍속'), ('max_ws_wd', '최대풍속 풍향'), ('max_ins_ws_wd', '최대순간풍속 풍향'), ('max_ws1', '최다풍향')], OK, 'PPT 우선순위표에는 없음(빨간 글자 항목)')
add('국내 기상(농업기상)', '바람(풍속·풍향)', red(5, 6, 9), NT, [('wind', '평균풍속'), ('max_wind', '최대풍속'), ('widdir', '풍향')], OK, 'widdir: 정의서에 "하류 미사용" 표기. 측정 높이 표기 없음')
add('국내 기상(종관기상)', '일조', '예측인자 PPT 추가 인자(일조·일사) · ' + red(2, 4, 5, 6, 9), AT, [('sum_sshr', '합계일조시간(시간)')], OK, 'DB 타입이 문자(text)지만 값은 모두 숫자 형식')
add('국내 기상(농업기상)', '일조', '예측인자 PPT 추가 인자(일조·일사) · ' + red(5, 6, 9), NT, [('sun_time', '일조시간(분)')], OK, 'DB 타입 문자(varchar), DB 설명 "일조시간(MM)"')
add('국내 기상(종관기상)', '일사', '예측인자 PPT 추가 인자(일조·일사) · ' + red(5, 6, 9), AT, [('sum_gsr', '합계일사량')], OK, '정의서 표기 "합계 일사랑"(오타)')
add('국내 기상(농업기상)', '일사', '예측인자 PPT 추가 인자(일조·일사) · ' + red(5, 6, 9), NT, [('srqty', '일사량')], OK, '')
add('국내 기상(종관기상)', '토양 온도', '예측인자 PPT 우선순위 5위(토양 온도) · ' + red(6) + '(지중온도 5·10·20cm)', AT, [('avg_cm5_te', '평균 5cm 지중온도'), ('avg_cm10_te', '평균 10cm 지중온도'), ('avg_cm20_tes', '평균 20cm 지중온도'),
    ('avg_cm30_te', '평균 30cm 지중온도', AUX, '30cm는 빨간 글자 범위(5·10·20cm) 밖'), ('avg_ts', '평균 지면온도', AUX, '지면온도는 빨간 글자 아님'), ('min_tg', '최저 초상온도', AUX, '초상온도는 빨간 글자 아님')], OK, '빨간 글자 출처는 농업기상 상세관측이나 깊이별 값은 DB에서 종관기상 표에만 있음(29개 관측소)')
add('국내 기상(농업기상)', '토양 온도', '예측인자 PPT 우선순위 5위(토양 온도) · ' + red(6), NT, [('soil_temp', '지중온도', OK, '깊이 표기 없음'), ('gr_temp', '초상온도(℃)', AUX, 'DB 설명 "초상온도(℃)", 정의서 표기 "지면온도". 초상온도는 빨간 글자 아님')], OK)
add('국내 기상(농업기상)', '토양 수분', '예측인자 PPT 우선순위 6위(토양 수분) · ' + red(4, 6, 9) + '(토양수분·10-20-30cm)', NT, [('soil_wt', '토양수분보정값(%)')], PART, 'DB 설명 "토양수분보정값(%)". 깊이 표기 없는 한 가지 값이고 10·20·30cm 깊이별 값은 없음')
add('토양', '토양 pH·유효인산·유기물', '예측인자 PPT 우선순위 8위(토양 pH) · ' + red(7, 10), 'api_data_info.soil_ph_info', [('acid', '토양산도'), ('vldpha', '유효인산'), ('om', '유기물 함량'), ('pnu_cd', '필지고유번호'), ('pnu_nm', '필지 명칭'), ('exam_day', '조사일')], OK,
    f"필지 단위 원자료 6,280,656행. 예) pnu_cd {ex19[0]['pnu_cd']} = {ex19[0]['pnu_nm']}. 2023년 이후 621,359행은 pnu_cd에 필지번호 대신 법정동코드(10자리)가 들어 있음. 연결 키인 법정동코드(legal_dong_code)·조사 연도(any_year)는 우리 시료 값과 같아져 목록에서 뺌")
add('대기(에어코리아)', '대기조성(대체)', '예측인자 PPT 우선순위 7위(대기조성) · ' + red(14), 'api_data_info.keco_airkorea_fine_dust_info', [('no2_value', '이산화질소 농도'), ('co_value', '일산화탄소 농도'), ('so2_value', '아황산가스 농도')], SUB, '빨간 글자의 이산화질소·일산화탄소·아황산가스는 그대로 있음. 대기조성 이산화탄소(CO2)는 DB에 없음(co_value는 일산화탄소)')
add('환경', '수질(대체: 수질오염원 설치시설)', '예측인자 PPT 추가 인자(수질) · ' + red(21) + '(빨간 글자: "수질오염물질을 배출하는 시설 데이")', 'api_data_info.water_pollution',
    [('bplc_nm', '사업장명'), ('lcpmt_ymd', '인허가일자'), ('sals_stts_nm', '영업상태명'), ('clsbiz_ymd', '폐업일자'), ('road_nm_addr', '도로명주소'), ('lotno_addr', '지번주소'), ('crd_info_x', '좌표 X(TM 계열 추정)'), ('crd_info_y', '좌표 Y(TM 계열 추정)')],
    SUB, f"행정안전부 인허가 자료 중 기타수질오염원 설치시설 {n(wp['n'])}개소(수질 측정값 아님). 좌표 있는 곳 {n(wp['n_xy'])}개소. {wp['col'][:10]} 적재, 테이블목록·정의서에 없음, DB 컬럼 설명 없음(한글명은 인허가 표준 컬럼명 기준)")
add('해외 기상(수입 시료)', '기온·강수·바람·일조·일사·토양수분', '매핑 엑셀 4번 시트(데이터출처 엑셀 빨간 글자의 기상 항목을 제조국에 적용)', 'api_data_info.global_weather_info',
    [('temperature_mean', '평균기온'), ('temperature_max', '최고기온'), ('temperature_min', '최저기온'), ('precipitation_sum', '일 강수량 합계'), ('wind_speed_mean', '평균 풍속(km/h)'), ('wind_speed_max', '최대 풍속(km/h)'),
     ('sunshine_duration', '일조시간(초)'), ('shortwave_radiation_sum', '일사 합계(MJ/m²)'), ('soil_moisture_mean', '토양수분 평균(m³/m³)'), ('relative_humidity_mean', '평균 상대습도', AUX, '습도는 PPT·빨간 글자 아님'), ('alpha_3', '국가 ISO 3자리 코드')],
    SUB, '나라마다 대표 지점 1곳(수도 부근) 값이라 제조소 위치 값이 아님. 테이블목록 "재수집 대상, 현재 미사용". 날짜(date)는 조인하면 시료 기준일과 같아져 목록에서 뺌')
add('해외 기상(수입 시료)', '기온·강수·바람·일조·일사·토양수분', '매핑 엑셀 외 추가 확인', 'api_data_info.global_factory_weather',
    [('temperature_mean', '일평균 지상 2m 기온(°C)'), ('temperature_max', '일최고 기온'), ('temperature_min', '일최저 기온'), ('precipitation_sum', '일강수량 합계(mm)'), ('wind_speed_mean', '일평균 10m 풍속(km/h)'), ('wind_speed_max', '일최대 10m 풍속(km/h)'),
     ('sunshine_duration', '일조시간(초)'), ('radiation_sum', '일 단파복사량 합계(MJ/m²)'), ('soil_moisture_mean', '표층 0~7cm 토양수분(m³/m³)'), ('relative_humidity_mean', '일평균 상대습도(%)', AUX, '습도는 PPT·빨간 글자 아님')],
    OK, 'ERA5 재분석 자료, 해외 제조소 좌표 × 일. 위도·경도(lat_g·lon_g)와 날짜(obsr_de, UTC 기준)는 조인하면 시료 값과 같아져 목록에서 뺌. 테이블목록·정의서에 없음(DB 설명으로 확인)')
add('해외 토양(수입 시료)', '토양 pH', '데이터출처 엑셀 빨간 글자의 토양검정 항목을 제조국에 적용', 'api_data_info.global_soil_ph', [('ph_mean', '국가 평균 pH'), ('ph_std', 'pH 표준편차'), ('iso3', '국가 ISO 3자리 코드')], SUB, '나라당 1행, 연도 구분 없음')
add('식품 정보(LIMS·대장)', '식품 유형', '예측인자 PPT 추가 인자(식품 유형)', 'ba_data.m_org_lims_data_info', [('prdlst_nm', '품목 명'), ('upper_prdlst_code', '상위 품목 코드'), ('upper_prdlst_nm', '상위 품목 명')], OK, '식약처 검사 원본(LIMS). 품목코드는 우리 시료 파일에 있어 뺌')
add('식품 정보(LIMS·대장)', '식품 유형', '예측인자 PPT 추가 인자(식품 유형)', 'ba_data.t_prdlst_cl', [('kor_nm', '한글 명'), ('lv', '분류 단계'), ('hrnk_prdlst_cd', '상위 품목 코드'), ('htrk_prdlst_cd', '최상위 품목 코드'), ('fdgrp_yn', '식품군 여부'), ('artcl_gun_code', '식품군코드'), ('food_hf_ls_cl_cd', '식품·건기·축산 분류코드'), ('dfn', '정의')], OK, '식품공전 품목분류(8,864행). 품목코드로 연결')
add('식품 정보(LIMS·대장)', '식품 유형(수입)', '예측인자 PPT 추가 인자(식품 유형)', 'ba_data.s_org_lims_data_info', [('prdlst_nm', '품명'), ('prduct_dvs_cd', '제품 구분 코드'), ('dclr_prduct_dvs', '신고제품구분'), ('hsk_no', 'HSK번호')], OK, '수입식품 검사 원본(수입 시료 전용)')
add('식품 정보(LIMS·대장)', '식품 유형(수입)', '예측인자 PPT 추가 인자(식품 유형)', 'ba_data.s_co_ifd_ovsmnfst_regstr_m', [('sfrprp_food_se_cd', '동물성 식품 식품종류')], NOV, '우리 시료에 연결되는 값이 없음')
add('식품 정보(LIMS·대장)', '식품 유형(가공방법)', '예측인자 PPT 추가 인자(식품 유형: 가공방법·제조공정)', 'ba_data.t_prdlst_mnf_ledg', [('mnf_cnd_cn', '제조 조건 내용')], SUB, '가공방법·섭취형태 전용 컬럼 없음. 값도 거의 비어 있음')
add('식품 정보(LIMS·대장)', '식품 유형(참고: 전처리 컬럼)', '예측인자 PPT 추가 인자(식품 유형)', 'ready_data.hazard_preprocessed_data',
    [('mapped_prdlst_code', '옛 품목코드를 새 코드로 바꾼 값', AUX, '전처리 컬럼. 식품공전 분류표·수분활성도 표 연결 키로만 사용'), ('statistic_prdlst_nm_1~4', '통계용 품목분류 1~4단계', AUX, '전처리 컬럼(일부 분류 통합, 없으면 "기타")')], AUX)
add('식품 정보(LIMS·대장)', '저장 및 유통 온도(대체)', '예측인자 PPT 추가 인자(저장 및 유통 온도)', 'ba_data.m_org_lims_data_info', [('cnvync_sttus_tp', '운반 상태 온도'), ('cnvync_sttus_tp_unit_code', '운반 상태 온도 단위 코드'), ('cnvync_mth_code', '운반 방법 코드'), ('splore_sttus_code', '시료 상태 코드'), ('distb_de', '유통 일'), ('valid_tmlmt', '유효 기한')], SUB, '시료와 연결되는 실제 보관·유통 온도 없음. 시료 운반 온도·시료 상태(실온/냉장/냉동)로 대신함. 코드 뜻은 공통코드 IM40·IM08·IM11')
add('식품 정보(LIMS·대장)', '저장 및 유통 온도(대체)', '예측인자 PPT 추가 인자(저장 및 유통 온도)', 'ba_data.t_prdlst_mnf_ledg', [('prsrv_distb_stdr_cn', '보존·유통 기준 내용'), ('pog_daycnt', '유통기한'), ('qlity_mntnc_tmlmt_daycnt', '품질 유지 기한 일수'), ('prdlst_report_no', '품목 보고 번호(연결 키)')], SUB, '텍스트 문구(예: "제조일로부터 12개월")')
add('식품 정보(LIMS·대장)', '저장 및 유통 온도(대체, 수입)', '예측인자 PPT 추가 인자(저장 및 유통 온도)', 'ba_data.s_org_lims_data_info', [('distb_bgn_de', '유통시작일'), ('distb_end_de', '유통종료일')], SUB, '수입 시료 전용')
add('식품 정보(LIMS·대장)', '포장 유형(대체)', '예측인자 PPT 추가 인자(포장 유형)', 'ba_data.t_prdlst_mnf_ledg', [('frmlc_mtrqlt', '포장 재질')], SUB, '시료와 연결되는 포장 방식(진공·가스치환) 없음. 포장 재질 텍스트로 대신함')
add('식품 내재요인', '수분활성도 · 식품 pH · 염도', '예측인자 PPT 우선순위 1위(수분활성도)·3위(pH)·4위(염도)', 'ba_data.t_aw_cl', [('aw', '수분활성도'), ('ph', '식품 pH'), ('na', '염도'), ('mc', '수분함량'),
    ('aw_source', '수분활성도 값 출처 표시', DEC, "채운 값도 'original'로 표시돼 있어 원래 값 구분에 못 씀")], DEC,
    f"DB {n(aw['rows'])}품목 = 리서치팀 통합_수분활성도…최종.xlsx {aw['file_codes']}품목 + DB에서 더한 {aw['db_only_codes']}품목. 파일 그대로가 아님: 파일 빈칸 {aw['filled_codes']}품목에 값 채움, 값 다른 {aw['changed_codes']}품목, 우유류 6품목 부등호 반대. 매핑 때 사용자 결정으로 제외했던 표")
add('식품 내재요인', '수분활성도 · 수분함량', '예측인자 PPT 우선순위 1위(수분활성도) · 매핑 엑셀 외 추가 확인', 'ba_data_info.food_moisture_info', [('water_activate', '수분활성도'), ('water_activate_source', '수분활성도 출처'), ('moisture_content', '수분함량'), ('prdlst_nm', '품목명(연결 키, 이름 매칭)'), ('food_type', '식품종')], DEC,
    f"DB 설명 '식품_수분_정보'. {n(fmi['n'])}품목 중 수분활성도 값 {n(fmi['n_aw'])} · 수분함량 {n(fmi['n_mc'])}. 품목코드·pH·염도 없음. 테이블목록·정의서에 없음")
r = table(ws, r, H, R1, widths=[18, 24, 50, 38, 24, 30, 18, 62], judge_col=6)

r = section(ws, r, '1-2. DB에 없는 위해인자 (수집 방법은 4번 시트)')
R2 = [
    ['수질 측정값(관개수·농업용수 위생)', '예측인자 PPT 추가 인자(수질) · ' + notred(17), 'DB 없음(저수지·관개수 수질 측정 표 없음)', '수질오염원 설치시설 목록(1-1의 water_pollution)'],
    ['대기조성 — 이산화탄소(CO2) 농도', '예측인자 PPT 우선순위 7위(대기조성) · ' + red(20), 'DB 없음', '이산화질소·일산화탄소·아황산가스(에어코리아)'],
    ['해수 염분', red(15), 'DB 없음(수분활성도 표의 na는 식품 염도)', '없음'],
    ['해상풍 풍향·풍속', red(16), 'DB 없음(육상 관측만 있음)', '종관기상·농업기상 풍속(육상)'],
    ['지하수 오염예측 등급', red(18), 'DB 없음', '없음'],
    ['지하수 오염취약성(DRASTIC) 점수', red(19), 'DB 없음', '없음'],
    ['깊이별 토양수분(10·20·30cm) · 높이별 기온(40·50·150·400cm)·풍향(150·300·400·1000cm)·풍속·최대풍속(150·300·1000cm)', red(6, 9), 'DB 없음(농업기상 표의 기온·토양수분·풍속은 높이·깊이 표기 없는 한 가지 값)', '농업기상 기온(temp)·토양수분보정값(soil_wt)·풍속(wind)'],
    ['방재기상관측(AWS)·농업기상 관측 조회(AAOS) 관측망 자료', red(3, 4), 'DB 없음(해당 관측망 표 없음)', '종관기상·농업기상의 같은 항목'],
    ['토양 수분량(용담 시험유역 6개 지점) · 유역별 토양수분량 · 시군별 토양유효수분율', red(11, 12, 13), 'DB 없음', '농업기상 토양수분보정값(soil_wt)'],
    ['월별 미래 기후(월강수량·월평균/최고/최저기온 예측)', red(8), 'DB 없음(기후변화 시나리오 예측 자료)', '종관기상·농업기상 과거 관측값'],
    ['시료의 실제 보관·유통 온도', '예측인자 PPT 추가 인자(저장 및 유통 온도)', f"시료와 연결되는 실측값 없음. 참고: 유통단계조사 표(ba_data_info.distb_step_exam)에 보관_온도(cstdy_tp) 자유 기재 {n(dse['n_temp'])}건(그중 온도 표기 5,577건)이 있으나 이물 신고 조사 문서라 시료 키가 없음", '운반 상태 온도·시료 상태·유통기한 문구'],
    ['포장 방식(진공·가스치환 등)', '예측인자 PPT 추가 인자(포장 유형)', f"시료와 연결되는 값 없음. 참고: 같은 유통단계조사 표의 포장_형태(frmlc_shap) {n(dse['n_pack'])}건('진공' {dse['n_vac']}건 포함), 시료 키 없음", '포장 재질 텍스트'],
    ['섭취 형태(그대로 먹는 식품인지)', '예측인자 PPT 추가 인자(식품 유형)', 'DB 없음', '품목 정의(dfn)·제조조건 텍스트'],
]
r = table(ws, r, ['위해인자', '근거', 'DB 상태', '가장 가까운 DB 대체 값'], R2)

r = section(ws, r, '1-3. (참고) DB에 있으나 리스테리아 시료와 연결할 수 없는 표')
R3 = [
    ['api_data_info.kma_weather_predict_info', '기온(temp) · 강수량(precip_mm) · 습도(hum) · 일조시간(sun_time) · 토양수분(soil_wt)', '2026-07-01 발표 12일치 예보 1,068행(89개 격자)뿐 → 과거 시료 기준일과 겹치는 날이 없음'],
    ['ba_data_info.distb_step_exam', '보관_온도(cstdy_tp) · 포장_형태(frmlc_shap) · 보관_여건(cstdy_cndtn) · 포장_상태(frmlc_stats)', '유통단계 조사 문서(doc_no) 단위라 시료 키가 없음'],
]
r = table(ws, r, ['표', '관련 컬럼', '연결할 수 없는 이유'], R3)
ws.freeze_panes = 'A4'

# ================================================================== 2. 조인 방법
ws = wb.create_sheet('2_조인방법')
r = title(ws, 1, '2. 수집 데이터 ↔ 리스테리아 시료(listeria_body.parquet) 조인 방법', f'{RAW_NOTE} / 연결 결과 % = 우리 시료 중 연결된 원래 값이 있는 비율')
r = section(ws, r, '2-1. 원칙')
lm = S['ledg_key_forms']
r = bullets(ws, r, [
    f"   · 기준 행: listeria_body.parquet {n(nrows)}행 전부 유지(left join). 국내 {n(smp['domestic'])} · 수입 {n(smp['overseas'])}",
    f"   · 기준일: 우리 기준일(mnfctur_de) = DB 기준일(mnfctur_de_or_fallback) 100% 일치. DB가 제조일 → 선적일 → 의뢰일 순으로 채운 날짜이며 제조일 {n(db_['제조일'])}행 · 의뢰일 {n(db_['의뢰일'])}행 · 선적일 {n(db_['선적일'])}행",
    f"   · DB 전처리 시료표(hazard_preprocessed_data, 리스테리아 {n(smp['db_body_rows'])}행)에 우리 시료키 100% 존재 → 이 표에서 품목제조번호·영업허가번호·수입접수번호·국가코드·좌표를 받아 각 표로 연결",
    '   · 기상·대기는 기준일 당일 값만 연결. 빈 값은 빈 값 그대로 둠(다른 날·다른 관측소 값으로 채우지 않음). 기간(7일·30일 등) 요약 여부는 미정',
    f"   · 좌표: 전처리 시료표의 위도·경도(lat·lon) 사용. DB 좌표가 없는 {smp['lat_db_null']}행만 우리 파일 좌표 사용",
    '   · 기존 조인 파일(notebook/3_DB수집_조인.ipynb → data/collected/listeria_joined.parquet)은 강수 빈 값 0 채움·범위 밖 값 제거·일조 단위 변환·토양 pH 평균·7/30일 요약이 들어간 가공본이며 이 보고 수치의 근거가 아님',
])
r = section(ws, r, '2-2. 표별 조인 키와 연결 결과')
JR = [
    ['① 전처리 시료표', 'ready_data.hazard_preprocessed_data', 'lims_data_key', 'lims_data_key', '1:1', f"100% ({n(nrows)}행)", 'DB에만 있는 89,354행(주소 출처 대부분 검사기관)은 우리 파일에 없음'],
    ['② 식약처 검사 원본', 'ba_data.m_org_lims_data_info', 'lims_data_key', 'lims_data_key', '1:1', f"키 매칭 국내 {pct(lp('m_org_lims_data_info','distb_de','key_match_domestic'))} · 수입 {pct(lp('m_org_lims_data_info','distb_de','key_match_overseas'))}", '품목명·운반온도·시료상태·유통일 등'],
    ['③ 품목제조대장', 'ba_data.t_prdlst_mnf_ledg', '전처리 시료표.prdlst_mnfctur_sttemnt_no', 'prdlst_report_no', 'N:1 (같은 번호 여러 행 → 삭제 안 된 행 중 최신 수정일 행. 삭제 행만 있는 21개 번호는 삭제 행 중 최신)', f"키 매칭 국내 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','key_match_domestic'))} · 수입 0%", f"번호 원문 일치 {n(lm['raw_match_rows'])}행 + 하이픈 등 기호를 뺀 번호로 추가 {n(lm['digits_only_extra_rows'])}행. prdlst_report_ledg_no로는 0건"],
    ['④ 영업허가대장', 'ba_data.t_bsn_lcns_ledg', '전처리 시료표.bsn_lcns_ledg_no', 'bsn_lcns_ledg_no', 'N:1', f"키 매칭 국내 {pct(lp('t_bsn_lcns_ledg','wrkshp_pkrm_area','key_match_domestic'))} · 수입 0%", ''],
    ['⑤ 식품공전 분류', 'ba_data.t_prdlst_cl', '전처리 시료표.mapped_prdlst_code(우리 시료 100% 있음)', 'prdlst_cd', 'N:1', f"키 매칭 국내 {pct(lp('t_prdlst_cl','lv','key_match_domestic'))} · 수입 {pct(lp('t_prdlst_cl','lv','key_match_overseas'))}", '20자리 옛 코드 33,992행과 14자리 코드 46행은 DB에서 새 코드로 바뀐 값'],
    ['⑥ 수입식품 검사 원본', 'ba_data.s_org_lims_data_info', '전처리 시료표.cntc_sys_key2', 'rcept_no', 'N:1 (접수번호당 시험항목별 여러 행 → 리스테리아 행 우선)', f"키 매칭 수입 {pct(lp('s_org_lims_data_info','hsk_no','key_match_overseas'))}", 'lims_data_key 두 번째 토큰으로는 고유 키 40,050개 중 7,101개(17.7%)만 매칭'],
    ['⑦ 공통코드', 'ba_data.t_cm_cmmn_code', '검사 원본 코드값', 'cmmn_cl_code + cmmn_code', 'N:1', '-', '분류코드는 IM08(운반방법)·IM11(시료상태)·IM40(온도단위). 정의서의 ZIM08/11/40은 DB에 0행'],
    ['⑧ 종관기상(ASOS)', 'api_data_info.kma_ground_obs_station_info → kma_asos_daily_info', '시료 lat·lon + 기준일', '가장 가까운 관측소 stn_id + meas_date',
     '① 관측소 좌표 = 관측소표 lat·lon(이력 여러 행이면 최신 수정일 행) ② 후보 = DB에 일자료가 있는 99곳(날짜 무관) ③ 시료 위치별 하버사인 거리 최소 1곳 고정 ④ 기준일과 같은 날짜 행의 값 1개', f"거리 중앙값 {cov('ASOS','avg_ta')['dist_p50']}km(최대 {cov('ASOS','avg_ta')['dist_max']}km). 연결률은 3번 시트 3-2",
     f"날짜와 무관하게 골라 국내 시료 {pct(dc('ASOS','avg_ta')['no_row'])}는 그날 행 없음(폐쇄된 대구(기) 176번 {a176['oper_end_dt'][:10]} 종료 → {n(a176['samples'])}행 등). (참고) 기준일에 자료가 있는 관측소 중 가장 가까운 곳이면 평균기온 {pct(alt('ASOS','avg_ta'))}. 관측소표는 이력 행 중복(243행/105곳, 이전으로 좌표가 1km 넘게 다른 곳 15곳)"],
    ['⑨ 농업기상(NAS)', 'api_data_info.nas_agri_weather_obs_station → nas_agri_weather_daily_info', '시료 lat·lon + 기준일', '가장 가까운 obsr_spot_code + meas_date',
     '① 관측소 좌표 = instl_la·instl_lo ② 후보 = 일자료가 있는 219곳(날짜 무관, 시험용 관측소 동방로거테스트 포함) ③ 거리 최소 1곳 고정 ④ 같은 날짜 값 1개', f"거리 중앙값 {cov('NAS','temp')['dist_p50']}km(최대 {cov('NAS','temp')['dist_max']}km)",
     f"관측 시작 전 날짜 등으로 국내 시료 {pct(dc('NAS','temp')['no_row'])}는 그날 행 없음((참고) 기준일에 자료가 있는 곳 중이면 기온 {pct(alt('NAS','temp'))}). 관측소마다 기록 항목이 달라 항목별로 값 없는 관측소가 있음(3번 시트)"],
    ['⑩ 대기(에어코리아)', 'api_data_info.keco_airkorea_obs_station → keco_airkorea_fine_dust_info', '시료 lat·lon + 기준일', '가장 가까운 station_nm + msur_dt',
     '① 측정소 좌표 = dm_x(위도)·dm_y(경도), 같은 이름 여러 행이면 운영 중 표시 행 우선 ② 후보 = 좌표 있는 672곳(날짜 무관) ③ 거리 최소 1곳 고정 ④ 같은 날짜 값 1개', f"거리 중앙값 {cov('에어코리아','no2_value')['dist_p50']}km(최대 {cov('에어코리아','no2_value')['dist_max']}km)",
     f"2019~2025년 새로 설치된 측정소가 설치 전 날짜에 잡혀 국내 시료 {pct(dc('AIR','no2_value')['no_row'])}는 그날 행 없음((참고) 기준일에 자료가 있는 곳 중이면 이산화질소 {pct(alt('AIR','no2_value'))}). 정의서의 'TM 좌표'는 오기(값이 위경도). 측정소표에 이름이 없는 일자료 측정소 {S['air_station']['daily_not_in_station_table']}곳(폐쇄 측정소 등)은 좌표가 없어 후보에서 빠짐"],
    ['⑪ 토양 pH', 'api_data_info.soil_ph_info', '우리 시료 legal_dong_code(10자리) + 기준일 연도', 'legal_dong_code + any_year', '같은 동·연도에 필지 여러 행 → 1:1 원값 연결 안 됨',
     f"기준일 연도에 같은 법정동 pH 행이 있는 국내 시료 {pct(ss['dong_year_pct'])}(검출 {pct(ss['dong_year_pct_detected'])}) · 연도 무관 {pct(ss['dong_anyyear_pct'])}. 국내 시료 {pct(ss['sgg_level_code_pct'])}({n(ss['sgg_level_code_rows'])}행)는 법정동코드가 시군구 단위(끝 00000)",
     f"필지 단위({soil_ex}). 연결된 동·연도당 pH 행 중앙값 {ss['parcels_per_dong_year_p50']:.0f}(최대 {ss['parcels_per_dong_year_max']:.0f}). (참고) 시군구·연도로 넓히면 {pct(ss['sgg_year_pct'])}이나 여러 동 필지를 묶어야 함. 한 값으로 줄이는 방법 미정. 법정동코드는 우리 파일 기준(DB와 296행 다름)"],
    ['⑫ 수입 기상(제조소 좌표)', 'api_data_info.global_factory_weather', '시료 lat·lon + 기준일', 'lat_g·lon_g + obsr_de', '좌표·날짜 정확 일치 1:1', f"수입 {pct(so['global_factory_sameday_pct'])} (좌표 {n(so['ovs_coords'])}개 전부 일치, 검출 {pct(so['global_factory_sameday_pct_detected'])})", 'obsr_de는 UTC 기준 일자. 테이블목록에 없음'],
    ['⑬ 수입 기상(국가 단위)', 'api_data_info.global_weather_info (+ ba_data_info.nation_code)', '전처리 시료표 prdctn→orgplce→import_nation_code(2자리) + 기준일', 'alpha_3 + date', '2자리 국가코드를 3자리로 바꾼 뒤 N:1', f"국가코드 있음 {pct(so['nation_code_present_pct'])} → 변환 {pct(so['alpha3_mapped_pct'])} → 당일 값 {pct(so['global_weather_info_sameday_pct'])} ({so['global_weather_info_countries']}개국)", '같은 나라 시료는 모두 같은 값. 생산국이 없는 수입 행의 import_nation_code는 수입식품 검사 원본에서 제조국과 같음. 원산지 코드에 KOREA·U.S.A·국내산 같은 글자 존재'],
    ['⑭ 수입 토양 pH', 'api_data_info.global_soil_ph', '3자리 국가코드', 'iso3', 'N:1', f"국가코드 변환 {pct(so['alpha3_mapped_pct'])}", '연도 없음'],
    ['⑮ 수분활성도·pH·염도', 'ba_data.t_aw_cl', '전처리 시료표.mapped_prdlst_code', 'prdlst_cd', 'N:1 (코드 정확 일치)', f"코드 일치 {pct(aw['code_match_pct'])} · 수분활성도 값 있음 국내 {pct(aw['aw']['domestic'])} / 수입 {pct(aw['aw']['overseas'])}(DB가 채운 값 포함) → 리서치팀 파일 원래 값 기준 국내 32.0% / 수입 42.3%", '상위 품목으로 올라가 찾을지는 결정 필요(현재 정확 일치만)'],
    ['⑯ 수분활성도 문헌값', 'ba_data_info.food_moisture_info', '전처리 시료표.prdlst_nm(품목명)', 'prdlst_nm', '이름 매칭(코드 없음)', '미계산(이름 매칭 규칙 결정 필요)', '품목명 정확 일치 국내 43.6% · 수입 54.1%(수분활성도 값 있는 품목만 14.8% · 31.8%)'],
    ['⑰ 수질오염원 설치시설', 'api_data_info.water_pollution', '시료 lat·lon 또는 주소', 'crd_info_x·crd_info_y(TM 계열, 좌표계 미확인) 또는 주소', '공간 연결(거리·반경 등) — 방식 결정 필요', '미계산', f"좌표 없는 시설 {n(wp['n'] - wp['n_xy'])}개소. 좌표계 확인 후 위경도 변환 필요. 시설 수·거리 같은 값은 새로 만드는 변수라 산정 방식 결정 필요"],
]
r = table(ws, r, ['단계', 'DB 표', '우리 데이터 쪽 키', 'DB 쪽 키', '관계·방식', '연결 결과', '주의'], JR, widths=[20, 42, 34, 34, 44, 46, 70])
ws.freeze_panes = 'A4'

# ================================================================== 3. 데이터 상태
ws = wb.create_sheet('3_데이터상태(raw)')
r = title(ws, 1, '3. 수집 데이터 상태 — DB 원자료 기준', f'{RAW_NOTE} / {STUDY} / 비결측 = NULL·빈 문자열 아님(0은 값으로 셈), 분모 = DB에 있는 행')
r = section(ws, r, '3-1. DB 원자료 핵심 이슈')
issues = [
    f"   · ① 종관기상 강수량(sum_rn) 빈 값 = 비가 오지 않은 날로 판단: 빈 날({n(rz_null['days'])}일)은 1시간 최다강수량·10분 최다강수량·강수 계속시간이 각 {EV['rn_null_and_hr_1_max_rn']['hr_1_max_rn_notnull_when_rn_null']}·{EV['rn_null_and_mi_10_max_rn']['mi_10_max_rn_notnull_when_rn_null']}·{EV['rn_null_and_sum_rn_dur']['sum_rn_dur_notnull_when_rn_null']}일에만 있음 / 0.0mm인 날({n(rz_zero['days'])}일)은 1시간 최다강수량 {pct(rz_zero['hr_1_max_rn_pct'])}·강수 계속시간 {pct(rz_zero['sum_rn_dur_pct'])} 기록(0.1mm 미만 강수) / 5km 이내 농업기상 {cmp_['pairs']}쌍 비교에서 빈 날 농업기상 0mm {cmp_['asos_null__nas_rn0_pct']}% / 빈 날 비율 7월 {mon['7']}%로 최저. 빈 값을 0mm로 바꿀지는 결정 필요",
    f"   · ② 관측소마다 기록하는 항목이 다름: 종관기상 일사량 {v(A,'sum_gsr')['stations_with_value_study']}/99곳 · 지중온도 29/99곳, 농업기상 일조시간 {v(N,'sun_time')['stations_with_value_study']}/219곳 · 지중온도 {v(N,'soil_temp')['stations_with_value_study']}/219곳 · 초상온도 {v(N,'gr_temp')['stations_with_value_study']}/219곳 · 토양수분 {v(N,'soil_wt')['stations_with_value_study']}/219곳. 판단 근거: 그 항목은 연구기간 내내 0건인데 같은 관측소의 기온은 기록됨 / 농업기상 관측소 코드 7번째 글자가 F인 {FK['stations']}곳은 기온 {FK['temp_n']/FK['rows']*100:.1f}% 기록, 일조·토양수분·지중온도·초상온도 DB 전체 0건",
    f"   · ③ 관측소 개설·폐쇄: 가장 가까운 관측소를 날짜와 무관하게 골라 그날 행이 없는 국내 시료 — 종관기상 {pct(dc('ASOS','avg_ta')['no_row'])}(폐쇄 후 {pct(WHY['ASOS']['after'])}: 대구(기) 176번 {a176['oper_end_dt'][:10]} 종료 → {n(a176['samples'])}행 / 개설 전 {pct(WHY['ASOS']['before'])}) · 농업기상 {pct(dc('NAS','temp')['no_row'])}(관측 시작 전 {pct(WHY['NAS']['before'])}) · 에어코리아 {pct(dc('AIR','no2_value')['no_row'])}(설치 전 {pct(WHY['AIR']['before'])}) → 기준일에 자료가 있는 관측소 중에서 고르면 행 없음이 거의 0%(3-2 참고 A 칸)",
    f"   · ④ 시험용 관측소: 농업기상 관측소표에 '{TST['name']}'({TST['code']}, 정읍시 북면 (유)동방이노베이션) · 일자료 {TST['daily_rows']}일({TST['daily_first']}~{TST['daily_last']}) · 국내 시료 {n(TST['domestic_samples_nearest'])}행이 가장 가까운 관측소로 잡음(기온 값 있음 {TST['of_which_temp_present']}행)",
    f"   · ⑤ 연도별 관측망 변화: 종관기상 일사량 연도별 비결측 {v(A,'sum_gsr')['yearly_min']}→{v(A,'sum_gsr')['yearly_max']}%, 농업기상 토양수분 2017·2018년 {v(N,'soil_wt')['yearly']['2017']}·{v(N,'soil_wt')['yearly']['2018']}%(그 외 42.6~61.8%), 에어코리아 일자료 측정소 2015년 319곳 → 2026년 674곳(좌표 있는 곳 672)",
    f"   · ⑥ 반복값·경계값: 농업기상 토양수분보정값 50.0 24,565건(31곳)·5.0 11,270건(88곳)·음수 {n(v(N,'soil_wt')['rules']['음수'])}건(43곳) / 일사량 39.5 이상 {n(v(N,'srqty')['rules']['39.5이상'])}건·0.0 {n(v(N,'srqty')['rules']['값0'])}건 / 최고·최저기온 ±50 {v(N,'hghst_artmp')['rules']['±50경계값']}·{v(N,'lowst_artmp')['rules']['±50경계값']}건 / 최대풍속 60 초과 {v(N,'max_wind')['rules']['음수·60초과']}건",
    '   · ⑦ 단위·타입: 종관기상 일조(sum_sshr) = 문자 타입(값은 전부 숫자, 단위 시간) / 농업기상 일조(sun_time) = 문자 타입, 단위 분(DB 설명 "일조시간(MM)", 중앙값 421) / 해외 일조 = 초, 풍속 = km/h',
    '   · ⑧ 토양 pH(soil_ph_info): 6,280,656행, 고유 pnu_cd 2,323,417 = 19자리 필지번호 2,309,571 + 10자리 법정동코드 13,846(2023년 이후 621,359행은 필지가 아님, DB 설명은 "2026 일부데이터부터 법정동코드") / 2015~2020 pH 14 초과 99건(최대 5,396) / pH 최대 9.5(2021년~)·최소 대부분 3.5(2022년~)로 잘린 모양 / 조사일이 조사 연도보다 뒤인 행 있음',
    '   · ⑨ 문서와 DB 불일치: 공통코드 ZIM08·11·40(정의서) → 실제 IM08·11·40 / 에어코리아 dm_x·dm_y 정의서 "TM 좌표" → 실제 위도·경도 / 농업기상 gr_temp 정의서 "지면온도" ↔ DB 설명 "초상온도(℃)" / soil_wt ↔ DB 설명 "토양수분보정값(%)" / 테이블목록에 없음: global_factory_weather·water_pollution·food_moisture_info',
    f"   · ⑩ 식약처 검사 기록 대체 변수 거의 비어 있음(국내): 운반 상태 온도 {pct(lp('m_org_lims_data_info','cnvync_sttus_tp','value_domestic'))} · 온도 단위 코드 0% · 시료 상태 {pct(lp('m_org_lims_data_info','splore_sttus_code','value_domestic'))} · 채취일 {n(E['pick_de_domestic_rows'])}행 · 보존유통기준 {pct(lp('t_prdlst_mnf_ledg','prsrv_distb_stdr_cn','value_domestic'))} · 포장실 면적 대부분 0.0(5,044/5,068 허가번호)",
    f"   · ⑪ 검출 시료에서 연결률이 낮음(국내끼리 비교): 유통기한 문구 국내 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','value_domestic'))} vs 국내 검출 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','value_detected_domestic'))}, 수분활성도(리서치팀 파일 기준) 국내 32.0% vs 국내 검출 12.0% → 값이 비었는지 여부 자체가 판정과 얽혀 있음",
    f"   · ⑫ 수분활성도 표(t_aw_cl): 리서치팀 파일 {aw['file_codes']}품목 + DB 추가 {aw['db_only_codes']}품목 / 파일 빈칸 {aw['file_aw_blank_codes']}품목 중 {aw['filled_codes']}품목에 값 채움(출처 표시 {fmtd(aw['filled_codes_source'])}, 예: 호밀 빈칸→0.561 = 곡류 중앙값) / 값 다른 {aw['changed_codes']}품목(건면 0.51→0.15 등) · 표기만 다른 54품목(≥→> 등) · 부등호 반대 6품목(우유·환원유·강화우유·유산균첨가우유·유당분해우유·산양유: 파일 0.98< → DB <0.98, 시료 11,219행) / 0.15 같은 값 {aw['aw_015_rows']}품목 / 값이 범위 텍스트('0.91~0.98', '>0.98', '-')",
]
r = bullets(ws, r, issues)

r = section(ws, r, '3-2. 국내 기상·대기 변수 (DB 원자료 + 우리 국내 시료 연결률과 빈 이유)')
NOTE = {
    (A, 'avg_ta'): '범위 밖 값 0', (A, 'max_ta'): '범위 밖 값 0', (A, 'min_ta'): '범위 밖 값 0',
    (A, 'sum_rn'): f"빈 값 = 비가 오지 않은 날로 판단(3-1 ①), 0.0 = 0.1mm 미만 강수. 300mm 초과 {v(A,'sum_rn')['rules']['300mm초과']}건",
    (A, 'avg_ws'): '이상값 없음', (A, 'max_ws'): '이상값 없음', (A, 'max_ins_ws'): '이상값 없음',
    (A, 'sum_sshr'): '문자 타입, 숫자 아닌 값 0. 0 = 흐린 날', (A, 'sum_gsr'): f"일사량을 기록하지 않는 관측소 26곳, 중간에 관측 시작한 곳 다수. 0 {v(A,'sum_gsr')['rules']['값0']}건·35 초과 {v(A,'sum_gsr')['rules']['35초과']}건(최대 {v(A,'sum_gsr')['max']})",
    (A, 'avg_ts'): '이상값 없음', (A, 'min_tg'): '이상값 없음',
    (A, 'avg_cm5_te'): '지중온도는 29개 관측소만 기록', (A, 'avg_cm10_te'): '29개 관측소만', (A, 'avg_cm20_tes'): '29개 관측소만(컬럼명 원문 그대로)', (A, 'avg_cm30_te'): '29개 관측소만',
    (N, 'temp'): '기온을 기록하지 않는 관측소 24곳. 범위(-35~45) 밖 25건(-49.4·-46.5·49.5 각 1건, -36~-40 저온 22건). 2014년 행 없음',
    (N, 'hghst_artmp'): f"±50 경계값 {v(N,'hghst_artmp')['rules']['±50경계값']}건, 범위 밖 {v(N,'hghst_artmp')['rules']['범위밖(-35~45)']}건",
    (N, 'lowst_artmp'): f"±50 경계값 {v(N,'lowst_artmp')['rules']['±50경계값']}건, 범위 밖 {v(N,'lowst_artmp')['rules']['범위밖(-35~45)']}건",
    (N, 'rn'): f"비가 안 온 날도 0으로 기록(값의 {v(N,'rn')['zero_pct_of_values']}%) → 빈 값은 진짜 결측. 490~500mm 구간 {v(N,'rn')['rules']['490mm이상']}건(500.0은 1건)",
    (N, 'wind'): '풍속을 기록하지 않는 관측소 37곳', (N, 'max_wind'): f"60 초과 {v(N,'max_wind')['rules']['음수·60초과']}건(최대 {v(N,'max_wind')['max']})",
    (N, 'widdir'): '정의서 "하류 미사용"', (N, 'sun_time'): f"문자 타입, 단위 분(중앙값 {v(N,'sun_time')['p50']:.0f}). 870분 초과 {v(N,'sun_time')['rules']['870분초과']}건",
    (N, 'srqty'): f"39.5 이상 {n(v(N,'srqty')['rules']['39.5이상'])}건(40 부근 몰림), 0.0 {n(v(N,'srqty')['rules']['값0'])}건",
    (N, 'gr_temp'): f"DB 설명 초상온도(정의서 지면온도). 범위(-35~70) 밖 {v(N,'gr_temp')['rules']['범위밖(-35~70)']}건(최대 {v(N,'gr_temp')['max']})",
    (N, 'soil_temp'): f"깊이 표기 없음. 범위 밖 {v(N,'soil_temp')['rules']['범위밖(-30~45)']}건",
    (N, 'soil_wt'): f"DB 설명 토양수분보정값(%). 50.0 24,565건(31곳), 5.0 11,270건(88곳), 음수 {n(v(N,'soil_wt')['rules']['음수'])}건(43곳). 연도별 비결측 2017·2018년만 30%대",
    (K, 'no2_value'): '음수 0', (K, 'co_value'): '음수 0. 일산화탄소(CO)이며 이산화탄소 아님', (K, 'so2_value'): f"0.000 값 {n(v(K,'so2_value')['rules']['값0'])}건",
}
FACT = [(A, 'avg_ta', '기온'), (A, 'max_ta', '기온'), (A, 'min_ta', '기온'), (N, 'temp', '기온'), (N, 'hghst_artmp', '기온'), (N, 'lowst_artmp', '기온'),
        (A, 'sum_rn', '강수'), (N, 'rn', '강수'), (A, 'avg_ws', '바람'), (A, 'max_ws', '바람'), (A, 'max_ins_ws', '바람'), (N, 'wind', '바람'), (N, 'max_wind', '바람'), (N, 'widdir', '바람'),
        (A, 'sum_sshr', '일조'), (N, 'sun_time', '일조'), (A, 'sum_gsr', '일사'), (N, 'srqty', '일사'),
        (A, 'avg_cm5_te', '토양 온도'), (A, 'avg_cm10_te', '토양 온도'), (A, 'avg_cm20_tes', '토양 온도'), (A, 'avg_cm30_te', '토양 온도(참고)'), (A, 'avg_ts', '토양 온도(참고)'), (A, 'min_tg', '토양 온도(참고)'),
        (N, 'soil_temp', '토양 온도'), (N, 'gr_temp', '토양 온도(참고)'), (N, 'soil_wt', '토양 수분'), (K, 'no2_value', '대기조성(대체)'), (K, 'co_value', '대기조성(대체)'), (K, 'so2_value', '대기조성(대체)')]
KOR = {(A, 'sum_gsr'): '합계일사량', (A, 'sum_sshr'): '합계일조시간(시간)', (N, 'temp'): '기온(℃)', (N, 'rn'): '일강수량', (N, 'widdir'): '풍향', (N, 'sun_time'): '일조시간(분)', (N, 'srqty'): '일사량', (N, 'gr_temp'): '초상온도(℃)', (N, 'soil_wt'): '토양수분보정값(%)'}
SRCN = {A: 'ASOS', N: 'NAS', K: '에어코리아'}; SRCK = {A: '종관기상', N: '농업기상', K: '에어코리아'}; DKEY = {A: 'ASOS', N: 'NAS', K: 'AIR'}
R32 = []
for t, c, f in FACT:
    x = v(t, c); cv = cov(SRCN[t], c); dk = DKEY[t]
    why = reason(dk, [c])
    if (t, c) == (A, 'sum_rn'): why = why.replace('그날 그 항목만 빈 값', '그날 강수량만 빈 값') + ' → 비가 오지 않은 날로 판단'
    R32.append([f, SRCK[t], c, KOR.get((t, c), x['korean']), x['db_type'], x['raw_period'], f"{x['stations_with_value_study']}/{x['stations_total']}", x['nonnull_study'], f"{x['yearly_min']}~{x['yearly_max']}",
                f"{x['min']:g} / {x['p50']:g} / {x['max']:g}", cv['sameday_pct'], why, cv['sameday_pct_detected'], alt(dk, c), f"{cv['var_sameday_pct']} ({cv['var_stations']}곳, {cv['var_dist_p50']}km)", NOTE[(t, c)]])
H3 = ['위해인자', '관측망', '컬럼(영문)', '컬럼(한글)', 'DB 타입', 'DB 원자료 기간', '값 있는 관측소/전체(연구기간)', '연구기간 비결측%', '연도별 비결측%(최저~최고)', '최소 / 중앙 / 최대',
      '국내 시료 기준일 당일 값 있음%(현재 방식: 날짜 무관 가장 가까운 관측소)', '국내 시료 빈 이유(합계 = 100 - 앞 칸)', '국내 검출 시료(1,844건) 당일 값 있음%',
      '(참고 A) 기준일에 자료가 있는 관측소 중 가장 가까운 곳을 쓰면 %', '(참고 B) 연구기간에 그 항목을 기록한 관측소 중 가장 가까운 곳을 쓰면 %(관측소 수, 거리 중앙값)', '원자료 특이사항']
r = table(ws, r, H3, R32, widths=[14, 10, 16, 18, 16, 22, 12, 10, 14, 18, 14, 44, 13, 14, 26, 62])
r = bullets(ws, r, [f"   · 빈 이유 용어: 그날 자료 행 없음 = 가장 가까운 관측소에 그 날짜 행 자체가 없음(종관기상: 폐쇄 후 {pct(WHY['ASOS']['after'])}·개설 전 {pct(WHY['ASOS']['before'])} / 농업기상: 관측 시작 전 {pct(WHY['NAS']['before'])}·운영 중 빠진 날 {pct(WHY['NAS']['gap'])} / 에어코리아: 설치 전 {pct(WHY['AIR']['before'])}·운영 중 빠진 날 {pct(WHY['AIR']['gap'])}) · 그 항목을 기록하지 않는 관측소 = 연구기간 내내 그 항목 0건 · 그날 그 항목만 빈 값 = 행은 있으나 그 항목만 비어 있음(DB에 사유 칸 없음)",
                     '   · 에어코리아 일자료 691곳 중 19곳은 측정소표에 이름이 없어 좌표가 없음 → 가장 가까운 측정소 후보는 672곳(연구기간 일자료의 1.7%가 후보에서 빠짐)',
                     '   · 참고 A·B는 관측소 고르는 방식을 바꿨을 때의 수치(값을 채운 것이 아님). 어느 방식을 쓸지는 결정 필요'])

r = section(ws, r, '3-3. 토양 pH (api_data_info.soil_ph_info, 필지 단위)')
r = table(ws, r, ['컬럼(영문)', '컬럼(한글)', '원자료 규모·예시', '비결측', '특이사항'], [
    ['acid', '토양산도', f"6,280,656행(고유 pnu_cd 2,323,417 = 19자리 필지번호 2,309,571 + 10자리 법정동코드 13,846) · 법정동 17,425 · 조사 연도 2015~2026. 필지 단위({soil_ex})", '6,184,118행 (98.5%)', 'pH 3 미만 51건(2021년 1건 외 2015~2020) · 10 초과 138건(모두 2015~2020, 14 초과 99, 최대 5,396) / 2021년 이후 최대 9.5, 2022년 이후 최소 대부분 3.5(2023년 3.19) → 잘린 모양(2021~2026 값 3.5: 1,120건, 9.5: 500건)'],
    ['vldpha', '유효인산', '〃', '6,239,520행 (99.3%)', '2015~2020 3,000 초과 2,657건(최대 37,647.6) / 2022년 이후 최소 1, 최대 5,000(2022)·9,999(2023~) (2021~2026 값 1: 11,333건, 9,999: 19건, 3,000 초과 3,843건)'],
    ['om', '유기물 함량', '〃', '6,263,331행 (99.7%)', '2015~2020 300 초과 184건(최대 1,936) / 2021년 이후 최대 300, 2022년 이후 최소 1 (2021~2026 값 300: 1,372건, 1: 8,932건)'],
    ['pnu_cd / pnu_nm / exam_day', '필지고유번호 / 필지 명칭 / 조사일', f"연도별 행 176,476(2018)~833,198(2023). 2026년은 부분 연도 168,089행(조사일 최대 2026-07-20). 2023년 이후 법정동코드 행 예) pnu_cd {ex10['pnu_cd']} · {ex10['pnu_nm']}", '100%', '조사일이 조사 연도보다 뒤인 행 있음(예: 2015년 조사분 조사일 최대 2020-12-23)'],
    ['(시료 연결)', '', f"국내 시료 동·연도 일치 {pct(ss['dong_year_pct'])} / 동(연도 무관) {pct(ss['dong_anyyear_pct'])} / 시군구 단위 법정동코드 {pct(ss['sgg_level_code_pct'])}", '', f"연결된 동·연도당 pH 행 중앙값 {ss['parcels_per_dong_year_p50']:.0f}개 → 한 값으로 줄이는 방법 결정 필요(임의 집계 안 함). 연결 키 legal_dong_code·any_year(조사 연도, 문자 4자리)"],
], widths=[16, 16, 44, 18, 90])

r = section(ws, r, '3-4. 해외 기상 (수입 시료)')
gwr = {x['column']: x for x in S['global_weather_info']}
gf = S['global_factory_sameday_cols']; gbc = so['global_weather_info_sameday_by_col']
RG = []
for c, k in [('temperature_mean', '평균기온'), ('temperature_max', '최고기온'), ('temperature_min', '최저기온'), ('precipitation_sum', '일 강수량 합계'), ('wind_speed_mean', '평균 풍속'), ('wind_speed_max', '최대 풍속'),
             ('sunshine_duration', '일조시간'), ('shortwave_radiation_sum', '일사 합계'), ('soil_moisture_mean', '토양수분 평균'), ('relative_humidity_mean', '평균 상대습도(참고)')]:
    x = gwr[c]
    note = {'sunshine_duration': '초 단위(최대 86,392)', 'wind_speed_max': 'km/h(최대 189.1), NULL 1건', 'wind_speed_mean': 'km/h', 'soil_moisture_mean': f"음수 {x['rules'].get('음수', 0)}건", 'temperature_min': '남극 등 극지 포함(-69.5)'}.get(c, '')
    RG.append(['global_weather_info(나라당 대표 1지점)', c, k, x['raw_period'], x['nonnull_study'], round(gbc[c], 1), f"{x['min']:g} / {x['max']:g}", note])
for c, k in [('temperature_mean', '일평균 기온(°C)'), ('temperature_max', '일최고 기온'), ('temperature_min', '일최저 기온'), ('precipitation_sum', '일강수량 합계(mm)'), ('wind_speed_mean', '일평균 풍속(km/h)'), ('wind_speed_max', '일최대 풍속(km/h)'),
             ('sunshine_duration', '일조시간(초)'), ('radiation_sum', '일 단파복사량(MJ/m²)'), ('soil_moisture_mean', '표층 토양수분(m³/m³)'), ('relative_humidity_mean', '일평균 상대습도(%)(참고)')]:
    x = gf[c]
    RG.append(['global_factory_weather(제조소 좌표)', c, k, '2015-01-01 ~ 2026-08-08', 'pg_stats null_frac 0', x['nonnull_pct'], f"{x['min']:g} / {x['max']:g}",
               {'sunshine_duration': '초 단위', 'soil_moisture_mean': 'DB 설명: 해안·도서 결측 가능(시료 당일 행은 결측 0)', 'temperature_mean': '날짜는 UTC 기준'}.get(c, '')])
r = table(ws, r, ['표', '컬럼(영문)', '컬럼(한글)', 'DB 원자료 기간', '표 비결측%', '수입 시료 당일 값 있음%', '최소 / 최대', '특이사항'], RG, widths=[30, 22, 22, 22, 16, 14, 18, 44])
r = bullets(ws, r, [f"   · global_weather_info: 249개국 × 4,206일, 테이블목록 \"재수집 대상, 현재 미사용\", 같은 나라 시료는 모두 같은 값. 빈 이유: 국가코드 없음 {pct(gwi_no_code)} · 3자리 코드로 안 바뀜 {pct(gwi_no_map)} · 그 나라 그날 값 없음 {pct(gwi_no_day)}",
                     f"   · global_factory_weather: 약 2.3억 행(해외 제조소 좌표 54,346개 × 일), 리스테리아 수입 시료 좌표 {n(so['ovs_coords'])}개 전부 일치. weather_data.overseas_weather는 이 표 일부의 단위 변환본",
                     '   · 해외 PM10은 global_factory_fine_dust(같은 좌표)·global_fine_dust_info(국가)에만 있음 — 리스테리아 위해인자 아님(참고)'])

r = section(ws, r, '3-5. 식약처 검사 기록·대장 변수 (키로 연결했을 때 우리 시료의 값 있음%; "-"·"해당없음" 같은 글자도 값으로 셈)')
RL = []
LOrder = [('m_org_lims_data_info', 'cnvync_sttus_tp', '저장·유통 온도(대체)', '값 예: 2·0·10·5 (numeric(10,0))'), ('m_org_lims_data_info', 'cnvync_sttus_tp_unit_code', '저장·유통 온도(대체)', '값 없음'),
          ('m_org_lims_data_info', 'cnvync_mth_code', '저장·유통 온도(대체)', 'IM08: 방문·택배·기타·퀵·우편'), ('m_org_lims_data_info', 'splore_sttus_code', '저장·유통 온도(대체)', 'IM11: 냉장·냉동·실온 (수입 대부분)'),
          ('m_org_lims_data_info', 'distb_de', '저장·유통 온도(대체)', '날짜. 대장 유통기한과 상관 0.80(2.ipynb) → 유통기한 날짜로 보임'), ('m_org_lims_data_info', 'valid_tmlmt', '저장·유통 온도(대체)', '날짜 아닌 작은 숫자(1·40·2·6 등)'),
          ('m_org_lims_data_info', 'pick_de', '(참고) 채취일', f"국내 {E['pick_de_domestic_rows']}행 외 빈 값"), ('m_org_lims_data_info', 'upper_prdlst_nm', '식품 유형', '치즈류·조리식품 등·발효유류'),
          ('t_prdlst_mnf_ledg', 'pog_daycnt', '저장·유통 온도(대체)', '문구: "제조일로부터 12개월" 등'), ('t_prdlst_mnf_ledg', 'prsrv_distb_stdr_cn', '저장·유통 온도(대체)', '거의 빈 값(실온보관·냉장보관 소수)'),
          ('t_prdlst_mnf_ledg', 'qlity_mntnc_tmlmt_daycnt', '저장·유통 온도(대체)', '"해당없음"·"-" 다수'), ('t_prdlst_mnf_ledg', 'frmlc_mtrqlt', '포장 유형(대체)', '자유 텍스트: 폴리에틸렌(내면)·PE·폴리에틸렌(PE) 등 표기 제각각'),
          ('t_prdlst_mnf_ledg', 'mnf_cnd_cn', '식품 유형', '거의 빈 값(레토르트식품·실온제조 소수)'), ('t_bsn_lcns_ledg', 'wrkshp_pkrm_area', '포장 유형(참고)', '값 대부분 0.0(5,044/5,068 허가번호)'),
          ('t_prdlst_cl', 'lv', '식품 유형', ''), ('t_prdlst_cl', 'food_hf_ls_cl_cd', '식품 유형', '999(미분류) 다수'), ('t_prdlst_cl', 'artcl_gun_code', '식품 유형', '빈 문자열 다수'), ('t_prdlst_cl', 'dfn', '식품 유형', '빈 문자열 다수'),
          ('s_org_lims_data_info', 'dclr_prduct_dvs', '식품 유형(수입)', '축산물·수산물·가공식품'), ('s_org_lims_data_info', 'hsk_no', '식품 유형(수입)', ''),
          ('s_org_lims_data_info', 'distb_bgn_de', '저장·유통 온도(대체, 수입)', '선적 후 중앙값 약 180일(2.ipynb) → 유통기한 성격으로 보임'), ('s_org_lims_data_info', 'packng_plc_cd', '포장 유형(참고)', '값 없음')]
for t, c, f, note in LOrder:
    x = L(t, c)
    RL.append([f, t, c, x['korean'].replace('ZIM40 ', '').replace('ZIM08 ', '').replace('ZIM11 ', ''), f"국내 {x['key_match_domestic']}% / 수입 {x['key_match_overseas']}%", x['value_domestic'], x['value_overseas'], x['value_detected_domestic'], note])
for key, tt, cc, kk in [('m_org.prdlst_nm', 'm_org_lims_data_info', 'prdlst_nm', '품목 명'), ('s_org.has_nm', 's_org_lims_data_info', 'prdlst_nm', '품명'), ('t_prdlst_cl.kor_nm', 't_prdlst_cl', 'kor_nm', '한글 명')]:
    x = E[key]
    RL.append(['식품 유형', tt, cc, kk, '위 같은 표 행과 동일', x['domestic'], x['overseas'], '-', '식품 유형 원자료(국내 검출은 미계산)'])
r = table(ws, r, ['위해인자', '표', '컬럼(영문)', '컬럼(한글)', '키 매칭률', '국내 값 있음%', '수입 값 있음%', '국내 검출 시료 값 있음%', '원자료 특이사항'], RL, widths=[22, 26, 24, 22, 20, 10, 10, 12, 52])

r = section(ws, r, '3-6. 수분활성도 · 식품 pH · 염도 (ba_data.t_aw_cl, 품목코드별 문헌값)')
RA = []
for c, k in [('aw', '수분활성도'), ('ph', '식품 pH'), ('na', '염도'), ('mc', '수분함량')]:
    x = aw[c]
    RA.append([c, k, f"{n(x['table_nonblank'])} / {n(aw['rows'])}", n(x['range_text_rows']), x['domestic'], x['overseas'], x['detected_domestic']])
r = table(ws, r, ['컬럼(영문)', '컬럼(한글)', "값 있는 품목('-' 제외)/전체", '범위·부등호 텍스트 행(~·-·< 등)', '국내 시료 DB 값 있음%(코드 정확 일치, DB가 채운 값 포함)', '수입 시료 DB 값 있음%', '국내 검출 시료 DB 값 있음%'], RA, widths=[12, 14, 22, 20, 22, 14, 14])
r = bullets(ws, r, ["   · 값이 숫자가 아닌 텍스트: '0.91~0.98', '>0.98', '<0.6', '-' 등 → 숫자로 쓰려면 범위 해석 규칙 결정 필요",
                     f"   · 리서치팀 파일과 비교: 수분활성도 빈칸 {aw['file_aw_blank_codes']}품목 중 {aw['filled_codes']}품목에 DB가 값 채움(출처 표시 {fmtd(aw['filled_codes_source'])}), 값 다름 {aw['changed_codes']}품목, 표기만 다름 54품목, 부등호 반대 6품목(우유류, 시료 11,219행) → 출처 표시(aw_source)만으로는 원래 값 구분 불가, 원래 값은 리서치팀 파일 기준",
                     '   · 수분활성도 원래 값 기준(DB가 채운 128품목 제외) 값 있음%: 국내 32.0% · 수입 42.3% · 국내 검출 12.0% (위 표는 DB 값 기준 38.6% · 49.6% · 13.0%). pH·염도·수분함량은 채운·바뀐 품목이 시료에 없어 차이 없음',
                     f"   · 우리 시료 영향: 채운 품목의 시료 {n(aw['sample_rows_filled'])}행 · 값이 바뀐 품목의 시료 {n(aw['sample_rows_changed'])}행 / 우리 시료 중 품목코드가 이 표에 그대로 있는 행 {pct(aw['code_match_pct'])}(우리 시료의 47%가 Z계열 품목코드)",
                     f"   · 0.15 같은 값 {aw['aw_015_rows']}품목(DB가 더한 품목 {aw['aw_015_db_only']} + 파일 값 0.48~0.93이던 5품목(건면 0.51·건조저장육류 0.85-0.93·기타 건포류 0.48~0.66 등)도 0.15로 올라감)",
                     f"   · 비교용: ba_data_info.food_moisture_info {n(fmi['n'])}품목(수분활성도 {n(fmi['n_aw'])}, 문헌 출처 {n(fmi['n_aw_src'])}) — 같은 품목명인데 값이 다른 사례(건조저장육류 0.65~0.92 vs t_aw_cl 0.15)"])

r = section(ws, r, '3-7. (참고) 위해인자는 아니지만 확인 요청된 변수 · 수질오염원 시설')
r = table(ws, r, ['원천', '컬럼(영문)', '컬럼(한글)', '연구기간 비결측%', '값 있는 관측소/전체', '국내 시료 당일 값 있음%', '비고'], [
    ['에어코리아', 'pm10_value', '미세먼지(PM10) 농도', v(K, 'pm10_value')['nonnull_study'], f"{v(K,'pm10_value')['stations_with_value_study']}/{v(K,'pm10_value')['stations_total']}", cov('에어코리아', 'pm10_value')['sameday_pct'], f"리스테리아 시트 빨간 글자 아님. 빈 이유: {reason('AIR', ['pm10_value'])}. (참고 A) {pct(alt('AIR','pm10_value'))}"],
    ['농업기상', 'hum', '평균습도', v(N, 'hum')['nonnull_study'], f"{v(N,'hum')['stations_with_value_study']}/{v(N,'hum')['stations_total']}", cov('NAS', 'hum')['sameday_pct'], f"습도는 빨간 글자 아님. 0.0 {v(N,'hum')['rules']['값0']}건. 빈 이유: {reason('NAS', ['hum'])}. 종관기상 평균습도(avg_rhm)는 99곳 99.8%"],
    ['water_pollution', 'sals_stts_nm 등', '수질오염원 설치시설', '-', f"{n(wp['n'])}개소(좌표 {n(wp['n_xy'])})", '미계산', f"영업상태 {fmtd(E['water_pollution_status'])} · 폐업일자 기재 {n(wp['n_close'])} · 인허가일 {wp['lc_min']}~{wp['lc_max']} · 인허가일 2026-06-30 이전이고 폐업일자가 비었거나 2015-04-07 이후인 시설 {n(wp['n_study'])}개소(상태가 폐업·취소인데 폐업일자 없는 시설 포함)"],
], widths=[14, 16, 18, 14, 16, 16, 80])
ws.freeze_panes = 'A4'

# ================================================================== 4. 수집 불가 인자 수집 방법
ws = wb.create_sheet('4_수집불가_수집방법')
r = title(ws, 1, '4. DB에 없는 위해인자 — 수집 방법', '출처 정보(기관·데이터명·형식·주기·URL)는 위해요소별_중요인자_데이터출처.xlsx 원문 / 수집·연결 방법은 제안 / 요청 기간 = 2015-04 ~ 2026-06 / 받은 원자료는 그대로 저장하고 요약·파생은 별도 결정')
def s(rn, i): return ('' if SRC[rn][i] is None else str(SRC[rn][i]).replace('\n', ' ').strip())
def srow(rn): return [s(rn, 2), s(rn, 1), f"{s(rn, 5)} / {s(rn, 6)}", s(rn, 7)]
R4 = [
    ['깊이별 토양수분(10·20·30cm) · 높이별 기온(40·50·150cm)·풍향(150·300·400·1000cm)·최대풍속(150·300·1000cm)', red(6), *srow(6), '공공데이터포털 인증키 발급 → 관측지점·기간별 API 반복 호출 → 원래 값 저장', '관측지점 좌표 중 가장 가까운 곳 + 기준일(농업기상 관측소와 같은 지점이면 지점코드 대응 확인)', '매핑 엑셀 3번 시트에 있음'],
    ['높이별 기온(50·150·400cm)·풍향·풍속(150·300·1000cm) · 깊이별 토양수분값(10·20·30cm)', red(9), *srow(9), '공공데이터포털 API(팜맵 지번코드 단위 조회)', '시료 좌표·주소 → 팜맵 필지(지번코드) 매칭 방법 검토 필요', '매핑 엑셀 3번 시트에 빠져 있던 행'],
    ['기온·강수·바람(방재기상관측망)', red(3), *srow(3), '기상청 API허브 API 호출', '관측지점 좌표 중 가장 가까운 곳 + 기준일', 'DB는 종관기상·농업기상으로 대신하는 중'],
    ['기온·풍속·토양수분·일조(농업기상관측망)', red(4), *srow(4), 'CSV 파일 내려받기', '관측지점 좌표 중 가장 가까운 곳 + 기준일', 'DB 토양수분이 일부만 있어 보완 가치 있음'],
    ['토양 수분량(관측단위 %)', red(11), *srow(11), 'CSV 파일 내려받기', '용담 시험유역 6개 지점뿐 → 전국 시료 연결은 거의 안 됨(참고용)', ''],
    ['유역별 토양수분량(m³/m³, 표준유역 850개)', red(12), *srow(12), 'API 또는 CSV 내려받기', '표준유역 경계 지도(SHP, 따로 확보) × 시료 좌표로 어느 유역인지 찾기 + 기준일', ''],
    ['시군별 토양유효수분율(%)', red(13), *srow(13), '사이트에서 지점·시간을 직접 넣어 조회(API·파일 없음) → 수동 수집 또는 농진청 자료 요청', '시군 + 날짜', '자동 수집 경로 없음'],
    ['월별 미래 기후(월강수량·월평균/최고/최저기온)', red(8), *srow(8), 'PDF 기후도면 내려받기', '미래 기후변화 시나리오 예측이라 과거 시료 학습 변수로 부적합(향후 예측 시나리오용)', ''],
    ['해수 염분', red(15), *srow(15), '공공데이터포털 API(격자·3분/15분) → 원래 시계열 그대로 저장(하루 단위 요약 여부는 별도 결정)', '시료에 생산 해역 정보가 없음(좌표 = 제조소 주소) → ① 바닷가 제조소 좌표에서 가장 가까운 격자 + 기준일로 대신하거나(생산 해역과 다를 수 있음) ② 수산물 어획·양식 해역 정보를 따로 확보. 내륙·해외 제조소 시료는 해당 없음', ''],
    ['해상풍 풍향·풍속', red(16), *srow(16), '공공데이터포털 API(격자·15분) → 원래 시계열 그대로 저장', '해수 염분과 같음(생산 해역 정보 없음)', ''],
    ['지하수 오염예측 등급', red(18), *srow(18), 'SHP 지도 파일 내려받기', '시료 좌표가 속한 구역(폴리곤)의 등급 찾기(연 단위)', ''],
    ['지하수 오염취약성(DRASTIC) 점수', red(19), *srow(19), 'SHP 지도 파일 내려받기', '시료 좌표가 속한 구역의 점수 찾기', ''],
    ['대기조성 — 이산화탄소 농도', '예측인자 PPT 우선순위 7위(대기조성) · ' + red(20), *srow(20), '기후정보포털 CSV 내려받기(국내 월평균·지점별)', '기준일 연월(+ 가까운 관측지점)', '전국 공통값에 가까워 구분력 확인 필요'],
    ['수질 — 농업저수지 수질정보', '예측인자 PPT 추가 인자(수질) · ' + notred(17), *srow(17), 'API 또는 CSV(시설별 연 1회 조사값)', '원문 제공 항목에 주소·좌표가 없음 → 시설코드·시설명으로 농어촌공사 저수지 위치 자료를 따로 확보한 뒤 시료 좌표와 가까운 저수지 + 조사연도', '빨간 글자 아님: 근거는 PPT'],
    ['시료의 실제 보관·유통 온도', '예측인자 PPT 추가 인자(저장 및 유통 온도)', '-', '-', '-', '-', '데이터출처 엑셀에 출처 없음 → 리서치팀·식약처 확인 필요', '-', 'DB 대체: 운반 상태 온도·시료 상태·유통기한 문구 / 유통단계조사(distb_step_exam)는 시료 키 없음'],
    ['포장 방식(진공·가스치환)', '예측인자 PPT 추가 인자(포장 유형)', '-', '-', '-', '-', '데이터출처 엑셀에 출처 없음 → 리서치팀 확인 필요', '-', 'DB 대체: 품목제조대장 포장 재질 텍스트'],
    ['섭취 형태(그대로 먹는 식품인지)', '예측인자 PPT 추가 인자(식품 유형)', '-', '-', '-', '-', '데이터출처 엑셀에 출처 없음 → 리서치팀 확인 필요', '-', 'DB 대체: 품목 정의(dfn)·제조조건 텍스트'],
    ['(참고·DB에 있음) 수질 — 수질오염물질 배출시설', '예측인자 PPT 추가 인자(수질) · ' + red(21) + '(일부 글자만 빨간색)', *srow(21), f"추가 수집 불필요: DB api_data_info.water_pollution에 들어 있음({n(wp['n'])}개소)", '좌표(TM 계열, 좌표계 확인 후 변환)·주소로 시료-시설 거리 계산(반경 내 시설 수 등 새 변수는 별도 결정)', '행 대부분이 파란 글자(표시 오류 가능) → 대상 여부 리서치팀 확인'],
    ['(참고·DB에 있음) 수분활성도 · 식품 pH · 염도', '예측인자 PPT 우선순위 1·3·4위', '식품안전정보원 리서치팀', '통합_수분활성도_수분함량_식품pH_염도_최종.xlsx', 'XLSX / -', '-', '추가 수집 불필요: DB ba_data.t_aw_cl에 들어 있음(채운 값 섞임 → 원래 값은 리서치팀 파일 기준). 수분활성도·수분함량은 ba_data_info.food_moisture_info에도 문헌 출처와 함께 있음', '품목코드 일치(2번 시트 ⑮)', '사용 여부 결정 필요(1번 시트)'],
]
r = table(ws, r, ['위해인자', '근거', '제공기관', '데이터명', '현재상태·형식 / 생산 주기', 'URL', '수집 방법', '시료 연결 방법(제안)', '비고'], R4, widths=[30, 40, 22, 26, 30, 44, 44, 52, 34])
ws.freeze_panes = 'A4'

for w in wb.worksheets:
    w.sheet_view.zoomScale = 90
wb.save(OUT)
print('saved', OUT)
