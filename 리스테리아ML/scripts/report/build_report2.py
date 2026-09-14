# -*- coding: utf-8 -*-
"""리스테리아 위해인자 DB 수집 결과 보고 (엑셀) v2 — 독립 검증 반영. 수치는 rep_stats.json·rep_extra.json(원자료 측정값)."""
import sys, json, openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
sys.stdout.reconfigure(encoding='utf-8')
OUT = sys.argv[1]
T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
S = json.load(open(T + r"\rep_stats.json", encoding='utf-8'))
E = json.load(open(T + r"\rep_extra.json", encoding='utf-8'))
src_wb = openpyxl.load_workbook(r"C:\Users\NT751\Desktop\천영진\리서치팀문서\위해요소별_중요인자_데이터출처.xlsx", data_only=True)
SRC = {r[0].row: [c.value for c in r] for r in src_wb.worksheets[0].iter_rows(min_row=2)}

V = {(x['table'], x['column']): x for g in ['asos', 'nas', 'air', 'global_weather_info'] for x in S[g]}
COV = {(c['source'], c['column']): c for c in S['coverage_weather']}
LIM = {(x['table'], x['column']): x for x in S['lims_coverage']}
smp = S['sample']; so = S['coverage_overseas']; ss = S['coverage_soil']; aw = S['t_aw_cl']; sn = S['asos_rn_null_semantics']
wp = E['water_pollution']; fmi = E['food_moisture_info']; dse = E['distb_step_exam']; db_ = E['date_basis']
A, N, K = 'kma_asos_daily_info', 'nas_agri_weather_daily_info', 'keco_airkorea_fine_dust_info'
def v(t, c): return V[(t, c)]
def cov(s, c): return COV[(s, c)]
def L(t, c): return LIM[(t, c)]
def pct(x): return f"{x:.1f}%"
def n(x): return f"{int(x):,}"
def lp(t, c, k): return L(t, c)[k]
def fmtd(d): return " · ".join(f"{k} {int(x):,}" for k, x in d.items())
nrows = smp['rows']

wb = openpyxl.Workbook()
thin = Side(style='thin', color='BFBFBF'); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
H_FILL = PatternFill('solid', fgColor='DDEBF7'); T_FILL = PatternFill('solid', fgColor='1F4E78')
G_FILL = {'가능': 'E2EFDA', '부분·대체': 'FFF2CC', '결정 필요': 'FCE4D6', '불가': 'F8CBAD', '참고': 'EDEDED'}
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
        if ln.startswith('■'): c.font = Font(bold=True)
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

STUDY = '연구기간 = 리스테리아 시료 기준일 범위 2015-04-07 ~ 2026-06-30'
RAW_NOTE = '모든 수치는 DB 원자료 그대로 측정(결측 대체·평균/중앙값 채움·단위 변환·이상값 제거·기간 집계 없음)'

# ================================================================== 0. 문자 보고
ws = wb.active; ws.title = '0_문자보고'
ws.column_dimensions['A'].width = 160
r = title(ws, 1, '[보고] 리스테리아 위해인자 식정원 DB 수집 가능성 · 데이터 상태 점검 결과', f'{RAW_NOTE} / {STUDY}')
lines = [
    '■ 점검 범위',
    ' - 위해인자: 예측인자 정리 PPT(우선순위 9 + 추가 5) + 위해요소별_중요인자_데이터출처.xlsx 빨간 글자 항목(19개 출처 행)',
    ' - 매핑 근거: 리스테리아_수집변수_매핑.xlsx / DB: TOXIN_MODEL_PRED(읽기 전용 SELECT) / 매핑 이후 DB 전 스키마 재탐색 결과 포함',
    f" - 조인 대상: listeria_body.parquet {n(nrows)}행(국내 {n(smp['domestic'])} · 수입 {n(smp['overseas'])} · 검출 {n(smp['detected'])})",
    '',
    '■ 1. DB 수집 가능 / 불가',
    ' - 가능: 기온 · 강수 · 바람 · 일조 · 일사 · 토양온도(지중) · 토양 pH·유효인산·유기물 · 식품 유형(LIMS 원본 품목코드·품목명, 식품공전 분류)',
    ' - 부분·대체: 토양수분(NAS 토양수분보정값, 깊이 표기 없음) · 대기조성(NO2·CO·SO2만, CO2 없음) · 저장·유통 온도(운반 온도·시료 상태·유통기한 문구) · 포장 유형(포장재질 텍스트)',
    f" - 부분·대체(매핑 이후 발견): 수질 → 수질오염원 설치시설 목록 api_data_info.water_pollution({n(wp['n'])}개소, 수질 측정값 아님, 테이블목록 미기재)",
    ' - DB에 있으나 사용 결정 필요: 수분활성도·식품 pH·염도 → ba_data.t_aw_cl(매핑 때 사용자 결정으로 제외, 리서치팀 파일 적재 후 빈칸 일부 채움) / 수분활성도·수분함량 문헌값 ba_data_info.food_moisture_info(품목명 단위)',
    ' - 수입 시료: 가능 = 기상 global_factory_weather(해외 제조소 좌표×일, 테이블목록 미기재) / 부분·대체 = 기상 global_weather_info(국가당 1점) · 토양 pH global_soil_ph(국가당 1행)',
    ' - DB 없음: 수질 실측(저수지·관개수) · CO2 · 해수 염분·해상풍 · 지하수 오염예측등급·DRASTIC · 깊이별 토양수분(20·30cm)·높이별 기온/풍향/풍속 · 기상청 AWS·AAOS 관측망(ASOS·NAS로 대체)',
    ' - DB 없음(계속): 용담·유역별·흙토람 토양수분 · 미래 기후도 · LIMS와 연결되는 실측 유통온도·포장방식(유통단계조사 텍스트는 있으나 연결 키 없음) · 섭취형태',
    '',
    '■ 2. 조인 방법',
    f" - 공통 키 lims_data_key: 우리 {n(nrows)}행 전부 DB 몸통(ready_data.hazard_preprocessed_data 중 리스테리아 {n(smp['db_body_rows'])}행)에 존재",
    f" - 기준일 = DB mnfctur_de_or_fallback(100% 일치): 제조일 {n(db_['제조일'])}(43.8%) · 의뢰일 {n(db_['의뢰일'])}(49.3%) · 선적일 {n(db_['선적일'])}(7.0%)",
    ' - LIMS·대장: lims_data_key / 품목제조번호 / 영업허가번호 / 품목코드 / 수입 접수번호(cntc_sys_key2)',
    f" - 국내 기상·대기: 시료 좌표 → 최근접 관측소 + 기준일 당일 원값(거리 중앙값 ASOS {cov('ASOS','avg_ta')['dist_p50']}km · NAS {cov('NAS','temp')['dist_p50']}km · 에어코리아 {cov('에어코리아','no2_value')['dist_p50']}km), 기간 집계 여부는 미정",
    f" - 토양 pH: 필지 단위라 1:1 연결 불가. 같은 동·연도 pH가 있는 국내 시료 {pct(ss['dong_year_pct'])} → 대표값 산정 결정 필요",
    f" - 수입 기상: 제조소 좌표 + 기준일 정확 일치 {pct(so['global_factory_sameday_pct'])} / 국가코드(alpha-2→alpha-3) + 기준일 {pct(so['global_weather_info_sameday_pct'])}",
    '',
    '■ 3. 데이터 상태 (원자료)',
    f" - 국내 시료 당일 원값 95% 이상: ASOS 기온·풍속·일조·지면온도 · NAS 강수({pct(cov('NAS','rn')['sameday_pct'])}) · 수입 제조소 좌표 기상(100%) · 수입 국가단위 기상({pct(so['global_weather_info_sameday_pct'])}, 국가당 1점)",
    f" - 70~95%: NAS 기온 {pct(cov('NAS','temp')['sameday_pct'])} · 에어코리아 NO2·CO·SO2 {pct(cov('에어코리아','co_value')['sameday_pct'])} · NAS 풍속 {pct(cov('NAS','wind')['sameday_pct'])} · NAS 일사 {pct(cov('NAS','srqty')['sameday_pct'])}",
    f" - 70% 미만(최근접 관측소 기준): ASOS 강수 {pct(cov('ASOS','sum_rn')['sameday_pct'])}(NULL·0 혼재) · ASOS 일사 {pct(cov('ASOS','sum_gsr')['sameday_pct'])} · NAS 토양수분 {pct(cov('NAS','soil_wt')['sameday_pct'])} · NAS 초상온도 {pct(cov('NAS','gr_temp')['sameday_pct'])} · NAS 일조 {pct(cov('NAS','sun_time')['sameday_pct'])} · ASOS 지중온도 {pct(cov('ASOS','avg_cm10_te')['sameday_pct'])} · NAS 지중온도 {pct(cov('NAS','soil_temp')['sameday_pct'])}",
    f" - 원인: 관측소별 센서 차이(ASOS 일사 {v(A,'sum_gsr')['stations_with_value_study']}/99 · 지중온도 29/99 · NAS 일조 {v(N,'sun_time')['stations_with_value_study']}/219 · 지중온도 {v(N,'soil_temp')['stations_with_value_study']}/219곳) → 해당 변수 관측 지점으로 넓히면 ASOS 일사 {pct(cov('ASOS','sum_gsr')['var_sameday_pct'])} · NAS 일조 {pct(cov('NAS','sun_time')['var_sameday_pct'])}",
    f" - 이상값: NAS 토양수분 50.0 동일값 24,565건·5.0 동일값 11,270건·음수 {n(v(N,'soil_wt')['rules']['음수'])}건 · NAS 일사 39.5 이상 {n(v(N,'srqty')['rules']['39.5이상'])}건 · NAS 최고/최저기온 ±50 경계값",
    ' - 이상값(계속): 토양 pH 상한 9.5(2021년~)·하한 3.5(2022년~) 절단 의심, 2015~2020 불가능 값(pH 최대 5,396)',
    f" - t_aw_cl: 리서치팀 파일에서 aw 빈칸이던 {aw['filled_codes']}코드가 DB에서 채워짐(aw_source 대부분 'original') · 값이 바뀐 {aw['changed_codes']}코드 · 우유류 6코드 부등호 방향 반전(파일 0.98< → DB <0.98, 시료 11,219행) · aw 0.15 동일값 {aw['aw_015_rows']}행",
    f" - LIMS 대체변수 희소(국내): 운반 온도 {pct(lp('m_org_lims_data_info','cnvync_sttus_tp','value_domestic'))} · 시료 상태 {pct(lp('m_org_lims_data_info','splore_sttus_code','value_domestic'))} · 유통기한 문구 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','value_domestic'))}(국내 검출 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','value_detected_domestic'))}) · 포장재질 {pct(lp('t_prdlst_mnf_ledg','frmlc_mtrqlt','value_domestic'))}(수입 0%)",
    ' - 문서와 DB 불일치: 공통코드 ZIM→IM · 에어코리아 좌표 TM 아닌 위경도 · 테이블목록 미기재 테이블 3개(global_factory_weather·water_pollution·food_moisture_info) (컬럼명 불일치는 3번 시트)',
    '',
    '■ 4. 수집 불가 인자 수집 방법',
    ' - 공공데이터포털·API허브 API: 농진청 상세관측 · 농정원 팜맵 농업기상(깊이·높이별) · 수과원 해수 염분·해상풍 · 수자원조사기술원 유역 토양수분 · 농어촌공사 저수지 수질 · 기상청 AWS',
    ' - 파일 다운로드: 농어촌공사 지하수 오염예측도·취약성도(SHP → 시료 좌표 공간조인) · 기상청 CO2(CSV, 월·년) · 기상청 AAOS(CSV) · 수자원공사 토양수분(CSV, 용담 6개 지점만)',
    ' - 직접 조회·요청: 흙토람 시군별 토양유효수분율(API·파일 없음)',
    ' - 학습용 부적합: 농업용 미래상세 기후도(PDF, SSP/RCP 예측 시나리오)',
    ' - 출처 없음(리서치팀 확인 필요): LIMS 시료 단위 실측 저장·유통 온도 · 포장 방식(진공·가스치환) · 섭취 형태(RTE)',
    '',
    '■ 결정 요청',
    ' - t_aw_cl 사용 여부(채움값·범위 텍스트 해석·상위 품목 탐색) / 문헌 aw(food_moisture_info) 품목명 매칭 규칙 / 수입 기상 원천(제조소 좌표 vs 국가)',
    ' - 토양 pH 대표값 산정 / ASOS 강수 NULL 해석(기상청 정의) / 기상 기간 집계 여부 / 수질오염원 시설 연결 방식(좌표계 확인)',
    '',
    '※ 상세: 1~4 시트 / 원자료: 리스테리아ML/data/collected/raw/',
    '※ 기존 조인본 data/collected/listeria_joined.parquet(3_DB수집_조인.ipynb)은 강수 NULL→0·단위 변환·이상값 제거·토양 pH 평균·7/30일 창 집계가 들어간 가공본으로, 본 보고 수치의 근거가 아님',
]
r = bullets(ws, r, lines)

# ================================================================== 1. 수집 가능/불가
ws = wb.create_sheet('1_수집가능·불가')
r = title(ws, 1, '1. 리스테리아 위해인자 — 식정원 DB 수집 가능 데이터 / 불가 데이터', '근거: PPT = 예측인자 정리 ppt.jpg / 출처 rN = 위해요소별_중요인자_데이터출처.xlsx N행 빨간 글자 / 한글 컬럼명 = 테이블정의서(없으면 DB 컬럼 코멘트) / 판정: 가능 · 부분·대체 · 결정 필요 · 참고')
r = section(ws, r, '1-1. DB에서 수집 가능한 데이터 (영문 + 한글 컬럼명)')
H = ['구분', '위해인자', '근거', 'DB 테이블 (스키마.테이블)', '컬럼(영문)', '컬럼(한글)', '판정', '비고']
R1 = []
def add(grp, factor, basis, table_, cols, judge, note=''):
    for i, item in enumerate(cols):
        c, k = item[0], item[1]
        j = item[2] if len(item) > 2 else judge
        R1.append([grp, factor, basis, table_, c, k, j, (item[3] if len(item) > 3 else (note if i == 0 else ''))])
AT, NT = 'api_data_info.kma_asos_daily_info', 'api_data_info.nas_agri_weather_daily_info'
add('국내 기상', '기온', 'PPT 1순위 · 출처 r2·r3·r4·r5·r6·r9', AT, [('avg_ta', '평균기온'), ('max_ta', '최고기온'), ('min_ta', '최저기온')], '가능', '기상청 ASOS 일자료 99개 지점. 출처 r3(AWS)·r4(AAOS) 관측망 자체는 DB 없음 → ASOS·NAS로 대체. 조인 키는 2번 시트')
add('국내 기상', '기온', 'PPT 1순위 · 출처 r5·r6·r9', NT, [('temp', '기온(℃)'), ('hghst_artmp', '최고기온'), ('lowst_artmp', '최저기온')], '가능', '농진청 농업기상 일자료 219개 지점. 높이 표기 없음')
add('국내 기상', '강수', 'PPT 9순위 · 출처 r2·r3·r5·r6·r9', AT, [('sum_rn', '일강수량')], '가능', 'NULL·0 혼재(3번 시트)')
add('국내 기상', '강수', 'PPT 9순위 · 출처 r5·r6·r9', NT, [('rn', '일강수량')], '가능', '0 명시 기록')
add('국내 기상', '바람(풍속·풍향)', '출처 r2·r3·r4·r5·r6·r9', AT, [('avg_ws', '평균풍속'), ('max_ws', '최대풍속'), ('max_ins_ws', '최대순간풍속'), ('max_ws_wd', '최대풍속 풍향'), ('max_ins_ws_wd', '최대순간풍속 풍향'), ('max_ws1', '최다풍향')], '가능', 'PPT 순위표에는 없음(빨간 글자 항목)')
add('국내 기상', '바람(풍속·풍향)', '출처 r5·r6·r9', NT, [('wind', '평균풍속'), ('max_wind', '최대풍속'), ('widdir', '풍향')], '가능', 'widdir: 정의서 "하류 미사용". 높이 표기 없음')
add('국내 기상', '일조', 'PPT 추가(일조·일사) · 출처 r2·r4·r5·r6·r9', AT, [('sum_sshr', '합계일조시간')], '가능', 'DB 타입 text')
add('국내 기상', '일조', 'PPT 추가(일조·일사) · 출처 r5·r6·r9', NT, [('sun_time', '일조시간(분)')], '가능', 'DB 타입 varchar(4), DB 코멘트 "일조시간(MM)"')
add('국내 기상', '일사', 'PPT 추가(일조·일사) · 출처 r5·r6·r9', AT, [('sum_gsr', '합계일사량')], '가능', '정의서 표기 "합계 일사랑"(오타)')
add('국내 기상', '일사', 'PPT 추가(일조·일사) · 출처 r5·r6·r9', NT, [('srqty', '일사량')], '가능', '')
add('국내 기상', '토양 온도', 'PPT 5순위 · 출처 r6(빨간: 지중온도 5·10·20cm)', AT, [('avg_cm5_te', '평균 5cm 지중온도'), ('avg_cm10_te', '평균 10cm 지중온도'), ('avg_cm20_tes', '평균 20cm 지중온도'),
    ('avg_cm30_te', '평균 30cm 지중온도', '가능(보조)', '30cm는 빨간 범위(5·10·20cm) 밖'), ('avg_ts', '평균 지면온도', '가능(보조)', '지면온도는 빨간 글자 아님'), ('min_tg', '최저 초상온도', '가능(보조)', '초상온도는 빨간 글자 아님')], '가능', '출처는 농진청 상세관측이나 깊이별 값은 DB에서 ASOS 테이블에만 있음(29개 지점)')
add('국내 기상', '토양 온도', 'PPT 5순위 · 출처 r6', NT, [('soil_temp', '지중온도', '가능', '깊이 표기 없음'), ('gr_temp', '초상온도(℃)', '가능(보조)', 'DB 코멘트 "초상온도(℃)", 정의서 표기 "지면온도". 초상온도는 빨간 글자 아님')], '가능')
add('국내 기상', '토양 수분', 'PPT 6순위 · 출처 r4·r6·r9(빨간: 토양수분(%)·10-20-30cm)', NT, [('soil_wt', '토양수분보정값(%)')], '부분·대체', 'DB 코멘트 "토양수분보정값(%)"(출처 r5·r6에서는 검정 항목). 깊이 표기 없음, 값이 상세관측 API 10cm 필드와 같아 10cm로 추정. 20·30cm 없음')
add('토양', '토양 pH·유효인산·유기물', 'PPT 8순위 · 출처 r7·r10', 'api_data_info.soil_ph_info', [('acid', '토양산도(pH)'), ('vldpha', '유효인산'), ('om', '유기물 함량'), ('legal_dong_code', '법정동 코드(10자리)'), ('any_year', '조사 연도'), ('pnu_cd', '필지고유번호'), ('exam_day', '조사일')], '가능', '필지 단위 원자료(행 6,280,656 / 고유 pnu_cd 2,323,417 = 19자리 필지코드 2,309,571 + 10자리 법정동코드 13,846). 2023년 이후 621,359행은 필지가 아닌 법정동코드. 뒤 4개는 연결 키')
add('대기', '대기조성(대체)', 'PPT 7순위 · 출처 r14(빨간: 이산화질소·일산화탄소·아황산가스)', 'api_data_info.keco_airkorea_fine_dust_info', [('no2_value', '이산화질소 농도'), ('co_value', '일산화탄소 농도'), ('so2_value', '아황산가스 농도')], '부분·대체', 'CO2(출처 r20)는 DB 없음. co_value는 CO. 측정소 좌표는 keco_airkorea_obs_station.dm_x(위도)·dm_y(경도)')
add('환경', '수질(대체: 수질오염원 설치시설)', 'PPT 추가(수질) · 출처 r21(빨간: 수질오염물질을 배출하는 시설 데이)', 'api_data_info.water_pollution',
    [('bplc_nm', '사업장명'), ('lcpmt_ymd', '인허가일자'), ('sals_stts_nm', '영업상태명'), ('clsbiz_ymd', '폐업일자'), ('road_nm_addr', '도로명주소'), ('lotno_addr', '지번주소'), ('crd_info_x', '좌표 X(TM 계열 추정)'), ('crd_info_y', '좌표 Y(TM 계열 추정)')],
    '부분·대체', f"행안부 인허가 자료 중 기타수질오염원 설치시설 {n(wp['n'])}개소(수질 측정값 아님). 좌표 {n(wp['n_xy'])}개소. {wp['col'][:10]} 적재, 테이블목록·정의서 미기재, DB 코멘트 없음(한글명은 인허가 표준 컬럼명 기준). 뷰 water_pollution_alive는 가공 뷰라 제외")
add('해외 기상(수입 시료)', '기온·강수·바람·일조·일사·토양수분', '매핑 시트4(출처 r2~r6 항목을 제조국에 적용)', 'api_data_info.global_weather_info',
    [('temperature_mean', '평균기온'), ('temperature_max', '최고기온'), ('temperature_min', '최저기온'), ('precipitation_sum', '일 강수량 합계'), ('wind_speed_mean', '평균 풍속(km/h)'), ('wind_speed_max', '최대 풍속(km/h)'),
     ('sunshine_duration', '일조시간(초)'), ('shortwave_radiation_sum', '일사 합계(MJ/m²)'), ('soil_moisture_mean', '토양수분 평균(m³/m³)'), ('relative_humidity_mean', '평균 상대습도', '참고', '습도는 PPT·빨간 글자 아님'), ('alpha_3', '국가 ISO 3자리 코드'), ('date', '일자')],
    '부분·대체', '국가당 대표점 1개(수도 인근). 테이블목록 "재수집 대상, 현재 미사용"')
add('해외 기상(수입 시료)', '기온·강수·바람·일조·일사·토양수분', '매핑 외 추가 확인', 'api_data_info.global_factory_weather',
    [('temperature_mean', '일평균 지상 2m 기온(°C)'), ('temperature_max', '일최고 기온'), ('temperature_min', '일최저 기온'), ('precipitation_sum', '일강수량 합계(mm)'), ('wind_speed_mean', '일평균 10m 풍속(km/h)'), ('wind_speed_max', '일최대 10m 풍속(km/h)'),
     ('sunshine_duration', '일조시간(초)'), ('radiation_sum', '일 단파복사량 합계(MJ/m²)'), ('soil_moisture_mean', '표층 0~7cm 토양수분(m³/m³)'), ('relative_humidity_mean', '일평균 상대습도(%)', '참고', '습도는 PPT·빨간 글자 아님'),
     ('lat_g', '위도(시료 lat과 동일값, 조인 키)'), ('lon_g', '경도(시료 lon과 동일값, 조인 키)'), ('obsr_de', '관측 일자(UTC 기준)')], '가능', 'ERA5 해외 제조소 좌표 × 일. 테이블목록·정의서 미기재(DB 코멘트로 확인)')
add('해외 토양', '토양 pH', '출처 r7·r10 항목을 제조국에 적용', 'api_data_info.global_soil_ph', [('ph_mean', '국가 평균 pH'), ('ph_std', 'pH 표준편차'), ('iso3', '국가 ISO 3자리 코드')], '부분·대체', '국가당 1행, 연도 구분 없음')
add('LIMS·대장', '식품 유형', 'PPT 추가(식품 유형)', 'ba_data.m_org_lims_data_info', [('prdlst_code', '품목 코드'), ('prdlst_nm', '품목 명'), ('upper_prdlst_code', '상위 품목 코드'), ('upper_prdlst_nm', '상위 품목 명')], '가능', '식약처 LIMS 원본')
add('LIMS·대장', '식품 유형', 'PPT 추가(식품 유형)', 'ba_data.t_prdlst_cl', [('prdlst_cd', '품목 코드'), ('kor_nm', '한글 명'), ('lv', '레벨'), ('hrnk_prdlst_cd', '상위 품목 코드'), ('htrk_prdlst_cd', '최상위 품목 코드'), ('fdgrp_yn', '식품군 여부'), ('artcl_gun_code', '식품군코드'), ('food_hf_ls_cl_cd', '식품·건기·축산 분류코드'), ('dfn', '정의')], '가능', '식품공전 품목분류(8,864행)')
add('LIMS·대장', '식품 유형(수입)', 'PPT 추가(식품 유형)', 'ba_data.s_org_lims_data_info', [('prdlst_cd', '품목코드'), ('prdlst_nm', '품명'), ('prduct_dvs_cd', '제품 구분 코드'), ('dclr_prduct_dvs', '신고제품구분'), ('hsk_no', 'HSK번호')], '가능', '수입식품 LIMS 원본(수입 시료 전용)')
add('LIMS·대장', '식품 유형(수입)', 'PPT 추가(식품 유형)', 'ba_data.s_co_ifd_ovsmnfst_regstr_m', [('sfrprp_food_se_cd', '동물성 식품 식품종류')], '부분·대체', '우리 시료에 값 없음')
add('LIMS·대장', '식품 유형(가공방법)', 'PPT 추가(가공방법·제조공정)', 'ba_data.t_prdlst_mnf_ledg', [('mnf_cnd_cn', '제조 조건 내용')], '부분·대체', '가공방법·섭취형태 전용 컬럼 없음')
add('LIMS·대장', '식품 유형(참고: 가공 컬럼)', 'PPT 추가(식품 유형)', 'ready_data.hazard_preprocessed_data',
    [('mapped_prdlst_code', '구→신 코드 치환 결과', '참고', '가공 컬럼. 식품공전 분류·t_aw_cl 조인 키로만 사용'), ('train_prdlst_code', '학습용 품목코드', '참고', '가공 컬럼(C계열 lv3·그 외 lv2로 재조정)'),
     ('train_prdlst_nm', '학습용 품목명', '참고', '가공 컬럼(미분류는 "기타"로 채움)'), ('statistic_prdlst_nm_1~4', '통계용 품목분류 lv1~4', '참고', '가공 컬럼(C01·C02→C03 통합, Z·G·E 계열 lv4 "기타")')], '참고')
add('LIMS·대장', '저장 및 유통 온도(대체)', 'PPT 추가(저장 및 유통 온도)', 'ba_data.m_org_lims_data_info', [('cnvync_sttus_tp', '운반 상태 온도'), ('cnvync_sttus_tp_unit_code', '운반 상태 온도 단위 코드'), ('cnvync_mth_code', '운반 방법 코드'), ('splore_sttus_code', '시료 상태 코드'), ('distb_de', '유통 일'), ('valid_tmlmt', '유효 기한')], '부분·대체', 'LIMS 시료와 연결되는 실측 저장·유통 온도 없음(유통단계조사 보관_온도는 연결 키 없음). 시료 운반 시 온도·상태로 대체. 코드 해석은 공통코드 IM40·IM08·IM11')
add('LIMS·대장', '저장 및 유통 온도(대체)', 'PPT 추가(저장 및 유통 온도)', 'ba_data.t_prdlst_mnf_ledg', [('prsrv_distb_stdr_cn', '보존·유통 기준 내용'), ('pog_daycnt', '유통기한'), ('qlity_mntnc_tmlmt_daycnt', '품질 유지 기한 일수'), ('prdlst_report_no', '품목 보고 번호(조인 키)')], '부분·대체', '텍스트 문구')
add('LIMS·대장', '저장 및 유통 온도(대체, 수입)', 'PPT 추가(저장 및 유통 온도)', 'ba_data.s_org_lims_data_info', [('distb_bgn_de', '유통시작일'), ('distb_end_de', '유통종료일')], '부분·대체', '수입 시료 전용')
add('LIMS·대장', '포장 유형(대체)', 'PPT 추가(포장 유형)', 'ba_data.t_prdlst_mnf_ledg', [('frmlc_mtrqlt', '포장 재질')], '부분·대체', 'LIMS 시료와 연결되는 포장 방식(진공·가스치환) 없음(유통단계조사 포장_형태는 연결 키 없음)')
add('식품 내재요인', '수분활성도 · 식품 pH · 염도', 'PPT 1·3·4순위', 'ba_data.t_aw_cl', [('aw', '수분활성도'), ('ph', '식품 pH'), ('na', '염도'), ('mc', '수분함량'), ('prdlst_cd', '식품공전 품목코드(14자리)'),
    ('aw_source', '수분활성도 산출 출처 표기', '결정 필요', "가공 여부 표기가 불완전('original' 표기 안에 채움값 포함)")], '결정 필요',
    f"DB {n(aw['rows'])}코드 = 리서치팀 통합_수분활성도…최종.xlsx {aw['file_codes']}코드 + DB 추가 {aw['db_only_codes']}코드. 그대로 적재한 것 아님: 파일 aw 빈칸 {aw['file_aw_blank_codes']}코드 중 {aw['filled_codes']}코드가 DB에서 채워짐, 값이 바뀐 {aw['changed_codes']}코드. 매핑 때 사용자 결정으로 제외(테이블목록 '곰팡이 독소 전용 컬럼'은 aflatoxin_aw 한 컬럼)")
add('식품 내재요인', '수분활성도 · 수분함량', 'PPT 1순위 · 매핑 외 추가 확인', 'ba_data_info.food_moisture_info', [('water_activate', '수분활성도'), ('water_activate_source', '수분활성도 출처'), ('moisture_content', '수분함량'), ('prdlst_nm', '품목명(조인 키, 이름 매칭)'), ('food_type', '식품종')], '결정 필요',
    f"DB 코멘트 '식품_수분_정보'. {n(fmi['n'])}품목 중 aw 값 {n(fmi['n_aw'])} · 수분함량 {n(fmi['n_mc'])}. 품목코드·pH·염도 없음. 테이블목록·정의서 미기재")
r = table(ws, r, H, R1, widths=[14, 24, 34, 38, 24, 28, 12, 60], judge_col=6)

r = section(ws, r, '1-2. DB에 없는 위해인자 (수집 방법은 4번 시트)')
R2 = [
    ['수질 실측(관개수·농업용수 위생)', 'PPT 추가(수질) · 출처 r17(빨간 아님)', 'DB 없음(저수지·관개수 수질 측정 테이블 없음)', '없음(수질오염원 설치시설 목록은 1-1 water_pollution)'],
    ['대기조성 — 이산화탄소(CO2)', 'PPT 7순위 · 출처 r20', 'DB 없음', 'NO2·CO·SO2(에어코리아)만 있음'],
    ['해수 염분', '출처 r15', 'DB 없음(t_aw_cl.na는 식품 염도)', '없음'],
    ['해상풍 풍향·풍속', '출처 r16', 'DB 없음(육상 관측만)', 'ASOS·NAS 풍속(육상)'],
    ['지하수 오염예측 등급', '출처 r18', 'DB 없음', '없음'],
    ['지하수 오염취약성 DRASTIC 점수', '출처 r19', 'DB 없음', '없음'],
    ['깊이별 토양수분(20·30cm) · 높이별 기온(40·50·400cm)·풍향·풍속·최대풍속', '출처 r6·r9', 'DB 없음. NAS temp는 150cm 기온, soil_wt는 10cm 토양수분에 해당하는 것으로 추정(train_data 아플라톡신 산출물의 상세관측 API 필드 tmprt_150·soil_Mitr_10과 값 일치)', 'NAS temp(150cm 추정)·soil_wt(10cm 추정)·wind(높이 미상)'],
    ['기상청 AWS(r3)·AAOS(r4) 관측망 자료', '출처 r3·r4', 'DB 없음(관측망 테이블 없음)', 'ASOS·NAS 같은 항목'],
    ['토양수분량(용담 시험유역 6개 지점) · 유역별 토양수분량 · 시군별 토양유효수분율', '출처 r11·r12·r13', 'DB 없음', 'NAS soil_wt'],
    ['월별 미래 기후(SSP/RCP 월강수량·월평균/최고/최저기온)', '출처 r8', 'DB 없음(예측 시나리오)', 'ASOS·NAS 과거 관측값'],
    ['실측 저장·유통 온도', 'PPT 추가(저장 및 유통 온도)', f"LIMS 시료와 연결되는 실측값 없음. 참고: ba_data_info.distb_step_exam(유통단계조사 {n(dse['n'])}건, 등록일시 {dse['mn']}~{dse['mx']}, 2008~2014년 문서 약 5,670건 이관 포함) cstdy_tp '보관_온도' 비어있지 않은 {n(dse['n_temp'])}건(그중 숫자·냉장/냉동/실온 등 온도 표기 5,577건, '-'·'.' 등 자리표시 다수) — LIMS 연결 키 없음", '운반 상태 온도·시료 상태·유통기한 문구'],
    ['포장 방식(진공·가스치환 등)', 'PPT 추가(포장 유형)', f"LIMS 시료와 연결되는 값 없음. 참고: distb_step_exam.frmlc_shap '포장_형태' {n(dse['n_pack'])}건('진공' {dse['n_vac']}건 포함), LIMS 연결 키 없음", '포장 재질 텍스트'],
    ['섭취 형태(그대로 섭취 여부)', 'PPT 추가(식품 유형)', 'DB 없음', '품목 정의(dfn)·제조조건 텍스트'],
]
r = table(ws, r, ['위해인자', '근거', 'DB 상태', '가장 가까운 DB 대체'], R2)

r = section(ws, r, '1-3. (참고) DB에 있으나 리스테리아 시료와 연결할 수 없는 테이블')
R3 = [
    ['api_data_info.kma_weather_predict_info', 'temp(기온) · precip_mm(강수량) · hum(습도) · sun_time(일조시간) · soil_wt(토양수분)', '2026-07-01 발표 12일 예보 1,068행(89개 격자)뿐 → 과거 시료 기준일과 겹치는 행 0'],
    ['ba_data_info.distb_step_exam', 'cstdy_tp(보관_온도) · frmlc_shap(포장_형태) · cstdy_cndtn(보관_여건) · frmlc_stats(포장_상태)', '유통단계 조사 문서(doc_no) 단위, LIMS 시료 키 없음'],
    ['train_data.wx160_aflatoxin_long 등', 'tmprt_150 · soil_Mitr_10 · hd_150 등', '아플라톡신 시료 전용 가공 산출물(NAS 결측을 채움). 리스테리아 키 교집합 0 → 원자료로 쓰지 않음'],
]
r = table(ws, r, ['테이블', '관련 컬럼', '연결 불가 사유'], R3)
ws.freeze_panes = 'A4'

# ================================================================== 2. 조인 방법
ws = wb.create_sheet('2_조인방법')
r = title(ws, 1, '2. 수집 데이터 ↔ 리스테리아 LIMS 데이터(listeria_body.parquet) 조인 방법', f'{RAW_NOTE} / 연결 결과 % = 우리 시료 중 연결된 원값이 있는 비율')
r = section(ws, r, '2-1. 원칙')
lm = S['ledg_key_forms']
r = bullets(ws, r, [
    f" - 기준 행: listeria_body.parquet {n(nrows)}행 전부 유지(left join). 국내 {n(smp['domestic'])} · 수입 {n(smp['overseas'])}",
    f" - 기준일: parquet mnfctur_de = DB mnfctur_de_or_fallback(일치율 100%). DB가 제조일 → 선적일 → 의뢰일 순으로 대체한 값이며 제조일 {n(db_['제조일'])}행(43.8%) · 의뢰일 {n(db_['의뢰일'])}행(49.3%) · 선적일 {n(db_['선적일'])}행(7.0%). 기준일의 56.2%는 제조일이 아님",
    f" - DB 몸통 ready_data.hazard_preprocessed_data(리스테리아 {n(smp['db_body_rows'])}행)에 우리 키 100% 존재 → 몸통에서 품목제조번호·영업허가번호·수입접수번호·국가코드·좌표를 가져와 각 테이블로 연결",
    ' - 기상·대기는 기준일 당일 원값만 연결. 결측은 결측 그대로 둠(인접일·주변 관측소 값으로 채우지 않음). 기간(7일·30일 등) 집계 여부는 미정',
    f" - 좌표: 몸통(DB) lat·lon 사용. DB 좌표가 없는 {smp['lat_db_null']}행만 parquet lat·lon 사용",
    ' - ※ 기존 조인본(notebook/3_DB수집_조인.ipynb → data/collected/listeria_joined.parquet)은 강수 NULL→0, 범위 밖 값 결측 처리, 일조 단위 변환, 토양 pH 평균·단계별 대체, 7·30일 창 집계가 들어간 가공본이며 본 보고 수치의 근거가 아님',
])
r = section(ws, r, '2-2. 테이블별 조인 키와 연결 결과')
J = [
    ['① 몸통', 'ready_data.hazard_preprocessed_data', 'lims_data_key', 'lims_data_key', '1:1', f"100% ({n(nrows)}행)", 'DB에만 있는 89,354행(주소 출처 대부분 검사기관)은 우리 데이터에 없음'],
    ['② 식약처 LIMS 원본', 'ba_data.m_org_lims_data_info', 'lims_data_key', 'lims_data_key', '1:1', f"키 매칭 국내 {pct(lp('m_org_lims_data_info','distb_de','key_match_domestic'))} · 수입 {pct(lp('m_org_lims_data_info','distb_de','key_match_overseas'))}", '품목코드·품목명·운반온도·시료상태·유통일 등'],
    ['③ 품목제조대장', 'ba_data.t_prdlst_mnf_ledg', '몸통.prdlst_mnfctur_sttemnt_no', 'prdlst_report_no', 'N:1 (같은 번호 여러 행 → 삭제 안 된 행 우선, 그중 최신 last_updt_dtm. 삭제 행만 있는 21개 번호는 삭제 행 중 최신)', f"키 매칭 국내 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','key_match_domestic'))} · 수입 0%", f"번호 원문 일치 {n(lm['raw_match_rows'])}행 + 하이픈 등 제거 시 추가 {n(lm['digits_only_extra_rows'])}행. prdlst_report_ledg_no 로는 0건"],
    ['④ 영업허가대장', 'ba_data.t_bsn_lcns_ledg', '몸통.bsn_lcns_ledg_no', 'bsn_lcns_ledg_no', 'N:1', f"키 매칭 국내 {pct(lp('t_bsn_lcns_ledg','wrkshp_pkrm_area','key_match_domestic'))} · 수입 0%", ''],
    ['⑤ 식품공전 분류', 'ba_data.t_prdlst_cl', '몸통.mapped_prdlst_code(우리 시료 100% 존재)', 'prdlst_cd', 'N:1', f"키 매칭 국내 {pct(lp('t_prdlst_cl','lv','key_match_domestic'))} · 수입 {pct(lp('t_prdlst_cl','lv','key_match_overseas'))}", '20자리 구코드 33,992행과 14자리 코드 46행은 DB에서 다른 코드로 매핑된 값'],
    ['⑥ 수입식품 LIMS', 'ba_data.s_org_lims_data_info', '몸통.cntc_sys_key2', 'rcept_no', 'N:1 (접수번호당 시험항목별 여러 행 → 리스테리아 행 우선)', f"키 매칭 수입 {pct(lp('s_org_lims_data_info','hsk_no','key_match_overseas'))}", 'lims_data_key 두 번째 토큰으로는 고유 키 40,050개 중 7,101개(17.7%)만 매칭'],
    ['⑦ 공통코드', 'ba_data.t_cm_cmmn_code', 'm_org 코드값', 'cmmn_cl_code + cmmn_code', 'N:1', '-', '분류코드는 IM08(운반방법)·IM11(시료상태)·IM40(온도단위). 정의서의 ZIM08/11/40은 DB에 0행'],
    ['⑧ 국내 기상 ASOS', 'api_data_info.kma_ground_obs_station_info → kma_asos_daily_info', '시료 lat·lon + 기준일', '최근접 관측소 stn_id + meas_date', '좌표 최근접(하버사인) → 당일 1행', f"거리 중앙값 {cov('ASOS','avg_ta')['dist_p50']}km(최대 {cov('ASOS','avg_ta')['dist_max']}km). 연결 결과는 3번 시트 3-2", '일자료 보유 99개 지점 중 선택. 관측소표는 이력 행 중복(243행/105지점, 이전으로 좌표 1km 초과 차이 15곳) → 최신 행 좌표 사용'],
    ['⑨ 국내 기상 NAS', 'api_data_info.nas_agri_weather_obs_station → nas_agri_weather_daily_info', '시료 lat·lon + 기준일', '최근접 obsr_spot_code(instl_la·instl_lo) + meas_date', '좌표 최근접 → 당일 1행', f"거리 중앙값 {cov('NAS','temp')['dist_p50']}km(최대 {cov('NAS','temp')['dist_max']}km)", '관측소마다 센서 구성이 달라 변수별로 값 없는 지점 있음(3번 시트)'],
    ['⑩ 대기 에어코리아', 'api_data_info.keco_airkorea_obs_station → keco_airkorea_fine_dust_info', '시료 lat·lon + 기준일', '최근접 station_nm(dm_x=위도, dm_y=경도) + msur_dt', '좌표 최근접 → 당일 1행', f"거리 중앙값 {cov('에어코리아','no2_value')['dist_p50']}km(최대 {cov('에어코리아','no2_value')['dist_max']}km)", f"정의서의 'TM 좌표'는 오기(값이 위경도). 측정소표에 이름이 없는 일자료 측정소 {S['air_station']['daily_not_in_station_table']}곳(폐쇄 측정소 등)은 좌표가 없어 제외 → 672곳 중 선택"],
    ['⑪ 토양 pH', 'api_data_info.soil_ph_info', '시료 legal_dong_code(10자리) + 기준일 연도', 'legal_dong_code + any_year', 'N:M (동·연도당 필지 여러 행) → 1:1 원값 연결 불가',
     f"기준일 연도에 같은 법정동 pH 행이 있는 국내 시료 {pct(ss['dong_year_pct'])}(검출 {pct(ss['dong_year_pct_detected'])}) · 연도 무관 {pct(ss['dong_anyyear_pct'])}. 국내 시료 {pct(ss['sgg_level_code_pct'])}({n(ss['sgg_level_code_rows'])}행)는 법정동 코드가 시군구 단위(끝 00000)",
     f"연결된 동·연도당 pH 행 중앙값 {ss['parcels_per_dong_year_p50']:.0f}(최대 {ss['parcels_per_dong_year_max']:.0f}). (참고) 시군구·연도로 넓히면 {pct(ss['sgg_year_pct'])}이나 여러 동 필지 집계가 전제됨. 집계 여부·방식 미정. any_year는 character(4). 법정동 코드는 parquet 기준(DB 몸통과 296행 상이, DB 기준 시군구 단위 56,594행, 비율 동일)"],
    ['⑫ 수입 기상(제조소 좌표)', 'api_data_info.global_factory_weather', '시료 lat·lon + 기준일', 'lat_g·lon_g + obsr_de', '좌표·날짜 정확 일치 1:1', f"수입 {pct(so['global_factory_sameday_pct'])} (좌표 {n(so['ovs_coords'])}개 전부 일치, 검출 {pct(so['global_factory_sameday_pct_detected'])})", 'obsr_de는 UTC 기준 일자. 테이블목록 미기재'],
    ['⑬ 수입 기상(국가 단위)', 'api_data_info.global_weather_info (+ ba_data_info.nation_code)', '몸통 prdctn→orgplce→import_nation_code(alpha-2) + 기준일', 'alpha_3 + date', 'alpha-2 → alpha-3 변환 후 N:1', f"국가코드 보유 {pct(so['nation_code_present_pct'])} → 변환 {pct(so['alpha3_mapped_pct'])} → 당일 값 {pct(so['global_weather_info_sameday_pct'])} ({so['global_weather_info_countries']}개국)", '같은 나라 시료는 모두 같은 값. 생산국이 없는 수입 행의 import_nation_code는 수입 LIMS 전용 행에서 제조국(mnf_nation_cd)과 같음. orgplce에 비ISO 텍스트(KOREA·U.S.A·국내산 등) 존재'],
    ['⑭ 수입 토양 pH', 'api_data_info.global_soil_ph', 'alpha-3 국가코드', 'iso3', 'N:1', f"국가코드 변환 {pct(so['alpha3_mapped_pct'])}", '연도 없음'],
    ['⑮ 수분활성도·pH·염도', 'ba_data.t_aw_cl', '몸통.mapped_prdlst_code', 'prdlst_cd', 'N:1 (코드 정확 일치)', f"코드 일치 {pct(aw['code_match_pct'])} · aw 값 있음 국내 {pct(aw['aw']['domestic'])} / 수입 {pct(aw['aw']['overseas'])}(DB가 채운 값 포함) → 리서치팀 파일 기준 원값 국내 32.0% / 수입 42.3%", '상위 품목으로 올라가 찾을지는 결정 필요(현재 정확 일치만)'],
    ['⑯ 수분활성도 문헌값', 'ba_data_info.food_moisture_info', '품목명', 'prdlst_nm', '이름 매칭(코드 없음)', '미계산(이름 매칭 규칙 결정 필요)', '학습용 품목명(train_prdlst_nm)과 일치 0%. DB 몸통 원 품목명(prdlst_nm) 정확 일치 국내 43.6% · 수입 54.1%(aw 값 있는 품목만 14.8% · 31.8%) → 매칭 규칙 결정 필요'],
    ['⑰ 수질오염원 설치시설', 'api_data_info.water_pollution', '시료 lat·lon 또는 주소', 'crd_info_x·crd_info_y(TM 계열, 좌표계 미상) 또는 주소', '공간 연결(거리·반경 등) — 방식 결정 필요', '미계산', f"좌표 없는 시설 {n(wp['n'] - wp['n_xy'])}개소. 좌표계 확인 후 위경도 변환 필요. 시설 수·거리 같은 값은 파생변수라 산정 방식 결정 필요"],
]
r = table(ws, r, ['단계', 'DB 테이블', '우리 데이터 쪽 키', 'DB 쪽 키', '관계·방식', '연결 결과', '주의'], J, widths=[20, 42, 34, 34, 34, 46, 64])
ws.freeze_panes = 'A4'

# ================================================================== 3. 데이터 상태
ws = wb.create_sheet('3_데이터상태(raw)')
r = title(ws, 1, '3. 수집 데이터 상태 — DB 원자료 기준', f'{RAW_NOTE} / {STUDY} / 비결측 = NULL·빈 문자열 아님, 분모 = DB에 존재하는 행(지점·일 행 자체가 없는 경우는 포함 안 됨)')
r = section(ws, r, '3-1. DB 원자료 핵심 이슈')
issues = [
    f" 1) ASOS 강수(sum_rn) NULL 의미 불명: 기온 관측일 중 NULL {sn['rn_null_on_obs_days_pct']}%, 값 중 0 {sn['rn_zero_pct_of_values']}% → NULL=무강수인지 결측인지 DB만으로 판단 불가(기상청 정의 확인 필요). NAS 강수(rn)는 0을 명시 기록(값의 {v(N,'rn')['zero_pct_of_values']}%)해 NULL=결측",
    f" 2) 관측소별 센서 구성 차이: ASOS 일사 {v(A,'sum_gsr')['stations_with_value_study']}/99곳, ASOS 지중온도 {v(A,'avg_cm10_te')['stations_with_value_study']}/99곳, NAS 일조 {v(N,'sun_time')['stations_with_value_study']}/219곳, NAS 지중온도 {v(N,'soil_temp')['stations_with_value_study']}/219곳, NAS 초상온도 {v(N,'gr_temp')['stations_with_value_study']}/219곳, NAS 토양수분 {v(N,'soil_wt')['stations_with_value_study']}/219곳 → 최근접 관측소에 센서가 없으면 결측",
    f" 3) 연도별 관측망 변화: ASOS 일사 연도별 비결측 {v(A,'sum_gsr')['yearly_min']}→{v(A,'sum_gsr')['yearly_max']}%, NAS 토양수분 2017·2018년 {v(N,'soil_wt')['yearly']['2017']}·{v(N,'soil_wt')['yearly']['2018']}%(그 외 42.6~61.8%), 에어코리아 일자료 측정소 2015년 319곳 → 2026년 674곳(좌표 있는 곳 672)",
    f" 4) 동일값·경계값 의심: NAS 토양수분 50.0 동일값 24,565건(31곳)·5.0 동일값 11,270건(88곳)·음수 {n(v(N,'soil_wt')['rules']['음수'])}건(43곳) / NAS 일사 39.5 이상 {n(v(N,'srqty')['rules']['39.5이상'])}건·0.0 {n(v(N,'srqty')['rules']['값0'])}건 / NAS 최고·최저기온 ±50 {v(N,'hghst_artmp')['rules']['±50경계값']}·{v(N,'lowst_artmp')['rules']['±50경계값']}건 / NAS 최대풍속 60 초과 {v(N,'max_wind')['rules']['음수·60초과']}건",
    ' 5) 단위·타입: ASOS 일조 sum_sshr = text(값은 전부 숫자 형식) / NAS 일조 sun_time = varchar(4), 분 단위(DB 코멘트 "일조시간(MM)", 중앙값 421) / 해외 일조 = 초, 풍속 = km/h',
    ' 6) 토양 pH(soil_ph_info): 6,280,656행, 고유 pnu_cd 2,323,417 = 19자리 필지코드 2,309,571 + 10자리 법정동코드 13,846(2023년 이후 621,359행은 필지가 아님, 예: 2023년 필지 590,158행/485,324필지 + 법정동코드 243,040행/7,920개; DB 코멘트는 "2026 일부부터 법정동코드") / 2015~2020 불가능 값(pH 14 초과 99건, 최대 5,396) / pH 상한 9.5(2021년~)·하한 대부분 3.5(2022년~) 절단 의심 / 조사일이 조사연도보다 뒤인 행 존재',
    ' 7) 문서와 DB 불일치: 공통코드 ZIM08·11·40(정의서) → 실제 IM08·11·40 / 에어코리아 dm_x·dm_y 정의서 "TM 좌표" → 실제 위도·경도 / NAS gr_temp 정의서 "지면온도" ↔ DB 코멘트 "초상온도(℃)" / NAS soil_wt ↔ DB 코멘트 "토양수분보정값(%)" / 테이블목록·정의서 미기재: global_factory_weather·water_pollution·food_moisture_info',
    f" 8) LIMS 대체변수 희소(국내): 운반 상태 온도 {pct(lp('m_org_lims_data_info','cnvync_sttus_tp','value_domestic'))} · 온도 단위 코드 0% · 시료 상태 {pct(lp('m_org_lims_data_info','splore_sttus_code','value_domestic'))} · 채취일 {n(E['pick_de_domestic_rows'])}행 · 보존유통기준 {pct(lp('t_prdlst_mnf_ledg','prsrv_distb_stdr_cn','value_domestic'))} · 포장실 면적 값 대부분 0.0(5,044/5,068 허가번호)",
    f" 9) 검출 시료에서 연결률 낮음(국내끼리 비교): 유통기한 문구 국내 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','value_domestic'))} vs 국내 검출 {pct(lp('t_prdlst_mnf_ledg','pog_daycnt','value_detected_domestic'))}, 수분활성도(리서치팀 파일 기준) 국내 32.0% vs 국내 검출 12.0% → 결측 여부 자체가 판정과 얽힘",
    f" 10) t_aw_cl: 리서치팀 파일 {aw['file_codes']}코드 + DB 추가 {aw['db_only_codes']}코드 / 파일 aw 빈칸 {aw['file_aw_blank_codes']}코드 중 {aw['filled_codes']}코드가 DB에서 채워짐(aw_source {fmtd(aw['filled_codes_source'])}, 예: 호밀 '-'→0.561 = 곡류 중앙값) / 값이 바뀐 {aw['changed_codes']}코드(건면 0.51→0.15 등), 표기만 바뀐 54코드(≥→> 등) + 부등호 방향이 반대로 바뀐 6코드(우유·환원유·강화우유·유산균첨가우유·유당분해우유·산양유: 파일 0.98< → DB <0.98, 시료 11,219행) / aw 0.15 동일값 {aw['aw_015_rows']}행(DB 추가 {aw['aw_015_db_only']}) / 값이 텍스트 범위('0.91~0.98', '>0.98', '-')",
]
r = bullets(ws, r, issues)

r = section(ws, r, '3-2. 국내 기상·대기 변수 (DB 테이블 원자료 + 우리 국내 시료 연결률)')
NOTE = {
    (A, 'avg_ta'): '범위 밖 값 0', (A, 'max_ta'): '범위 밖 값 0', (A, 'min_ta'): '범위 밖 값 0',
    (A, 'sum_rn'): f"NULL·0 혼재(3-1 ①). 300mm 초과 {v(A,'sum_rn')['rules']['300mm초과']}건",
    (A, 'avg_ws'): '이상값 없음', (A, 'max_ws'): '이상값 없음', (A, 'max_ins_ws'): '이상값 없음',
    (A, 'sum_sshr'): 'text 타입, 숫자 아닌 값 0. 0 = 흐린 날', (A, 'sum_gsr'): f"일사계 없는 지점 26곳(연구기간 전부 NULL), 중간 관측 시작 다수. 값0 {v(A,'sum_gsr')['rules']['값0']}건·35 초과 {v(A,'sum_gsr')['rules']['35초과']}건(최대 {v(A,'sum_gsr')['max']})",
    (A, 'avg_ts'): '이상값 없음', (A, 'min_tg'): '이상값 없음',
    (A, 'avg_cm5_te'): '지중온도 관측 29개 지점만', (A, 'avg_cm10_te'): '29개 지점만', (A, 'avg_cm20_tes'): '29개 지점만(컬럼명 원문 그대로)', (A, 'avg_cm30_te'): '29개 지점만',
    (N, 'temp'): '기온 센서 없는 지점 24곳. 범위(-35~45) 밖 25건(-49.4·-46.5·49.5 각 1건, -36~-40 저온 22건). 2014년 행 없음',
    (N, 'hghst_artmp'): f"±50 경계값 {v(N,'hghst_artmp')['rules']['±50경계값']}건, 범위 밖 {v(N,'hghst_artmp')['rules']['범위밖(-35~45)']}건",
    (N, 'lowst_artmp'): f"±50 경계값 {v(N,'lowst_artmp')['rules']['±50경계값']}건, 범위 밖 {v(N,'lowst_artmp')['rules']['범위밖(-35~45)']}건",
    (N, 'rn'): f"0 명시 기록(값의 {v(N,'rn')['zero_pct_of_values']}%) → NULL=결측. 490~500mm 구간 {v(N,'rn')['rules']['490mm이상']}건(500.0은 1건)",
    (N, 'wind'): '풍속 센서 보유 지점만', (N, 'max_wind'): f"60 초과 {v(N,'max_wind')['rules']['음수·60초과']}건(최대 {v(N,'max_wind')['max']})",
    (N, 'widdir'): '정의서 "하류 미사용"', (N, 'sun_time'): f"varchar(4). 분 단위(중앙값 {v(N,'sun_time')['p50']:.0f}). 870분 초과 {v(N,'sun_time')['rules']['870분초과']}건",
    (N, 'srqty'): f"39.5 이상 {n(v(N,'srqty')['rules']['39.5이상'])}건(40 상한 부근 몰림), 0.0 {n(v(N,'srqty')['rules']['값0'])}건",
    (N, 'gr_temp'): f"DB 코멘트 초상온도(정의서 지면온도). 범위(-35~70) 밖 {v(N,'gr_temp')['rules']['범위밖(-35~70)']}건(최대 {v(N,'gr_temp')['max']})",
    (N, 'soil_temp'): f"깊이 표기 없음. 범위 밖 {v(N,'soil_temp')['rules']['범위밖(-30~45)']}건",
    (N, 'soil_wt'): f"DB 코멘트 토양수분보정값(%). 50.0 동일값 24,565건(31곳), 5.0 동일값 11,270건(88곳), 음수 {n(v(N,'soil_wt')['rules']['음수'])}건(43곳). 연도별 비결측 2017·2018년만 30%대",
    (K, 'no2_value'): '음수 0', (K, 'co_value'): '음수 0. CO(일산화탄소)이며 CO2 아님', (K, 'so2_value'): f"0.000 값 {n(v(K,'so2_value')['rules']['값0'])}건",
}
FACT = [(A, 'avg_ta', '기온'), (A, 'max_ta', '기온'), (A, 'min_ta', '기온'), (N, 'temp', '기온'), (N, 'hghst_artmp', '기온'), (N, 'lowst_artmp', '기온'),
        (A, 'sum_rn', '강수'), (N, 'rn', '강수'), (A, 'avg_ws', '바람'), (A, 'max_ws', '바람'), (A, 'max_ins_ws', '바람'), (N, 'wind', '바람'), (N, 'max_wind', '바람'), (N, 'widdir', '바람'),
        (A, 'sum_sshr', '일조'), (N, 'sun_time', '일조'), (A, 'sum_gsr', '일사'), (N, 'srqty', '일사'),
        (A, 'avg_cm5_te', '토양 온도'), (A, 'avg_cm10_te', '토양 온도'), (A, 'avg_cm20_tes', '토양 온도'), (A, 'avg_cm30_te', '토양 온도(보조)'), (A, 'avg_ts', '토양 온도(보조)'), (A, 'min_tg', '토양 온도(보조)'),
        (N, 'soil_temp', '토양 온도'), (N, 'gr_temp', '토양 온도(보조)'), (N, 'soil_wt', '토양 수분'), (K, 'no2_value', '대기조성(대체)'), (K, 'co_value', '대기조성(대체)'), (K, 'so2_value', '대기조성(대체)')]
KOR = {(A, 'sum_gsr'): '합계일사량', (N, 'temp'): '기온(℃)', (N, 'rn'): '일강수량', (N, 'widdir'): '풍향', (N, 'sun_time'): '일조시간(분)', (N, 'srqty'): '일사량', (N, 'gr_temp'): '초상온도(℃)', (N, 'soil_wt'): '토양수분보정값(%)'}
SRCN = {A: 'ASOS', N: 'NAS', K: '에어코리아'}
R3 = []
for t, c, f in FACT:
    x = v(t, c); cv = cov(SRCN[t], c)
    R3.append([f, SRCN[t], c, KOR.get((t, c), x['korean']), x['db_type'], x['raw_period'], f"{x['stations_with_value_study']}/{x['stations_total']}", x['nonnull_study'], f"{x['yearly_min']}~{x['yearly_max']}",
               f"{x['min']:g} / {x['p50']:g} / {x['max']:g}", cv['sameday_pct'], cv['sameday_pct_detected'], f"{cv['var_sameday_pct']} ({cv['var_stations']}곳, {cv['var_dist_p50']}km)", NOTE[(t, c)]])
H3 = ['위해인자', '원천', '컬럼(영문)', '컬럼(한글)', 'DB 타입', 'DB 원자료 기간', '값 있는 지점/전체(연구기간)', '연구기간 비결측%', '연도별 비결측%(최저~최고)', '최소 / 중앙 / 최대',
      '국내 시료 기준일 당일 원값%(최근접 관측소)', '국내 검출 시료(1,844건) 당일 원값%', '(참고) 연구기간에 해당 변수 값이 있는 지점 중 최근접 시 %(지점 수, 거리 중앙값)', '원자료 특이사항']
r = table(ws, r, H3, R3, widths=[14, 10, 16, 18, 16, 22, 12, 10, 14, 18, 14, 13, 24, 62])
r = bullets(ws, r, [' - 에어코리아 일자료 691곳 중 19곳은 측정소표에 이름이 없어 좌표 없음 → 최근접 연결은 672곳 기준(연구기간 일자료의 1.7%가 연결 대상에서 빠짐)'])

r = section(ws, r, '3-3. 토양 pH (api_data_info.soil_ph_info, 필지 단위)')
r = table(ws, r, ['컬럼(영문)', '컬럼(한글)', '원자료 규모', '비결측', '특이사항'], [
    ['acid', '토양산도(pH)', '6,280,656행(고유 pnu_cd 2,323,417 = 19자리 필지코드 2,309,571 + 10자리 법정동코드 13,846) · 법정동 17,425 · 조사연도 2015~2026', '6,184,118행 (98.5%)', 'pH 3 미만 51건(2021년 1건 외 2015~2020) · 10 초과 138건(모두 2015~2020, 14 초과 99, 최대 5,396) / 2021년 이후 최대 9.5, 2022년 이후 최소 대부분 3.5(2023년 3.19) → 절단 의심(2021~2026 값 3.5: 1,120건, 9.5: 500건)'],
    ['vldpha', '유효인산', '〃', '6,239,520행 (99.3%)', '2015~2020 3,000 초과 2,657건(최대 37,647.6) / 2022년 이후 최소 1, 최대 5,000(2022)·9,999(2023~) (2021~2026 값 1: 11,333건, 9,999: 19건, 3,000 초과 3,843건)'],
    ['om', '유기물 함량', '〃', '6,263,331행 (99.7%)', '2015~2020 300 초과 184건(최대 1,936) / 2021년 이후 최대 300, 2022년 이후 최소 1 (2021~2026 값 300: 1,372건, 1: 8,932건)'],
    ['any_year / exam_day', '조사 연도 / 조사일', '연도별 행 176,476(2018)~833,198(2023). 2026년은 부분 연도 168,089행(조사일 최대 2026-07-20)', '100%', 'any_year는 character(4). 2023년 이후 pnu_cd에 10자리 법정동코드 행 섞임(621,359행). 조사일이 조사연도보다 뒤인 행 존재(예: 2015년 조사분 조사일 최대 2020-12-23)'],
    ['(시료 연결)', '', f"국내 시료 동·연도 일치 {pct(ss['dong_year_pct'])} / 동(연도 무관) {pct(ss['dong_anyyear_pct'])} / 시군구 단위 법정동 코드 {pct(ss['sgg_level_code_pct'])}", '', f"연결된 동·연도당 pH 행 중앙값 {ss['parcels_per_dong_year_p50']:.0f}개 → 대표값 산정 필요(임의 집계 안 함)"],
], widths=[16, 16, 44, 18, 90])

r = section(ws, r, '3-4. 해외 기상 (수입 시료)')
gwr = {x['column']: x for x in S['global_weather_info']}
gf = S['global_factory_sameday_cols']; gbc = so['global_weather_info_sameday_by_col']
RG = []
for c, k in [('temperature_mean', '평균기온'), ('temperature_max', '최고기온'), ('temperature_min', '최저기온'), ('precipitation_sum', '일 강수량 합계'), ('wind_speed_mean', '평균 풍속'), ('wind_speed_max', '최대 풍속'),
             ('sunshine_duration', '일조시간'), ('shortwave_radiation_sum', '일사 합계'), ('soil_moisture_mean', '토양수분 평균'), ('relative_humidity_mean', '평균 상대습도(참고)')]:
    x = gwr[c]
    note = {'sunshine_duration': '초 단위(최대 86,392)', 'wind_speed_max': 'km/h(최대 189.1), NULL 1건', 'wind_speed_mean': 'km/h', 'soil_moisture_mean': f"음수 {x['rules'].get('음수', 0)}건", 'temperature_min': '남극 등 극지 포함(-69.5)'}.get(c, '')
    RG.append(['global_weather_info(국가당 1점)', c, k, x['raw_period'], x['nonnull_study'], round(gbc[c], 1), f"{x['min']:g} / {x['max']:g}", note])
for c, k in [('temperature_mean', '일평균 기온(°C)'), ('temperature_max', '일최고 기온'), ('temperature_min', '일최저 기온'), ('precipitation_sum', '일강수량 합계(mm)'), ('wind_speed_mean', '일평균 풍속(km/h)'), ('wind_speed_max', '일최대 풍속(km/h)'),
             ('sunshine_duration', '일조시간(초)'), ('radiation_sum', '일 단파복사량(MJ/m²)'), ('soil_moisture_mean', '표층 토양수분(m³/m³)'), ('relative_humidity_mean', '일평균 상대습도(%)(참고)')]:
    x = gf[c]
    RG.append(['global_factory_weather(제조소 좌표)', c, k, '2015-01-01 ~ 2026-08-08', 'pg_stats null_frac 0', x['nonnull_pct'], f"{x['min']:g} / {x['max']:g}",
               {'sunshine_duration': '초 단위', 'soil_moisture_mean': 'DB 코멘트: 해안·도서 결측 가능(시료 당일 행은 결측 0)', 'temperature_mean': '날짜는 UTC 기준'}.get(c, '')])
r = table(ws, r, ['테이블', '컬럼(영문)', '컬럼(한글)', 'DB 원자료 기간', '테이블 비결측%', '수입 시료 당일 원값%', '최소 / 최대', '특이사항'], RG, widths=[30, 22, 22, 22, 16, 14, 18, 44])
r = bullets(ws, r, [' - global_weather_info: 249개국 × 4,206일, 테이블목록 "재수집 대상, 현재 미사용", 같은 나라 시료는 모두 같은 값',
                     f" - global_factory_weather: 약 2.3억 행(해외 제조소 좌표 54,346개 × 일), 리스테리아 수입 시료 좌표 {n(so['ovs_coords'])}개 전부 일치. weather_data.overseas_weather는 이 테이블 일부의 단위 변환본",
                     ' - 해외 PM10은 global_factory_fine_dust(같은 좌표)·global_fine_dust_info(국가)에만 있음 — 리스테리아 위해인자 아님(참고)'])

r = section(ws, r, '3-5. LIMS·대장 변수 (키로 연결했을 때 우리 시료의 원값 존재율; 원값에는 "-"·"해당없음" 같은 텍스트도 포함)')
RL = []
LOrder = [('m_org_lims_data_info', 'cnvync_sttus_tp', '저장·유통 온도(대체)', '값 예: 2·0·10·5 (numeric(10,0))'), ('m_org_lims_data_info', 'cnvync_sttus_tp_unit_code', '저장·유통 온도(대체)', '값 없음'),
          ('m_org_lims_data_info', 'cnvync_mth_code', '저장·유통 온도(대체)', 'IM08: 방문·택배·기타·퀵·우편'), ('m_org_lims_data_info', 'splore_sttus_code', '저장·유통 온도(대체)', 'IM11: 냉장·냉동·실온 (수입 대부분)'),
          ('m_org_lims_data_info', 'distb_de', '저장·유통 온도(대체)', '날짜. 대장 유통기한과 상관 0.80(2.ipynb) → 유통기한 날짜로 보임'), ('m_org_lims_data_info', 'valid_tmlmt', '저장·유통 온도(대체)', '날짜 아닌 작은 숫자(1·40·2·6 등)'),
          ('m_org_lims_data_info', 'pick_de', '(참고) 채취일', f"국내 {E['pick_de_domestic_rows']}행 외 NULL"), ('m_org_lims_data_info', 'upper_prdlst_nm', '식품 유형', '치즈류·조리식품 등·발효유류'),
          ('t_prdlst_mnf_ledg', 'pog_daycnt', '저장·유통 온도(대체)', '문구: "제조일로부터 12개월" 등'), ('t_prdlst_mnf_ledg', 'prsrv_distb_stdr_cn', '저장·유통 온도(대체)', '거의 빈 값(실온보관·냉장보관 소수)'),
          ('t_prdlst_mnf_ledg', 'qlity_mntnc_tmlmt_daycnt', '저장·유통 온도(대체)', '"해당없음"·"-" 다수'), ('t_prdlst_mnf_ledg', 'frmlc_mtrqlt', '포장 유형(대체)', '자유 텍스트: 폴리에틸렌(내면)·PE·폴리에틸렌(PE) 등 표기 제각각'),
          ('t_prdlst_mnf_ledg', 'mnf_cnd_cn', '식품 유형', '거의 빈 값(레토르트식품·실온제조 소수)'), ('t_bsn_lcns_ledg', 'wrkshp_pkrm_area', '포장 유형(참고)', '값 대부분 0.0(5,044/5,068 허가번호)'),
          ('t_prdlst_cl', 'lv', '식품 유형', ''), ('t_prdlst_cl', 'food_hf_ls_cl_cd', '식품 유형', '999(미분류) 다수'), ('t_prdlst_cl', 'artcl_gun_code', '식품 유형', '빈 문자열 다수'), ('t_prdlst_cl', 'dfn', '식품 유형', '빈 문자열 다수'),
          ('s_org_lims_data_info', 'dclr_prduct_dvs', '식품 유형(수입)', '축산물·수산물·가공식품'), ('s_org_lims_data_info', 'hsk_no', '식품 유형(수입)', ''),
          ('s_org_lims_data_info', 'distb_bgn_de', '저장·유통 온도(대체, 수입)', '선적 후 중앙값 약 180일(2.ipynb) → 유통기한 성격으로 보임'), ('s_org_lims_data_info', 'packng_plc_cd', '포장 유형(참고)', '값 없음')]
for t, c, f, note in LOrder:
    x = L(t, c)
    RL.append([f, t, c, x['korean'].replace('ZIM40 ', '').replace('ZIM08 ', '').replace('ZIM11 ', ''), f"국내 {x['key_match_domestic']}% / 수입 {x['key_match_overseas']}%", x['value_domestic'], x['value_overseas'], x['value_detected_domestic'], note])
for key, tt, cc, kk in [('m_org.prdlst_code_m', 'm_org_lims_data_info', 'prdlst_code', '품목 코드'), ('m_org.prdlst_nm', 'm_org_lims_data_info', 'prdlst_nm', '품목 명'),
                        ('s_org.has_cd', 's_org_lims_data_info', 'prdlst_cd', '품목코드'), ('s_org.has_nm', 's_org_lims_data_info', 'prdlst_nm', '품명'), ('t_prdlst_cl.kor_nm', 't_prdlst_cl', 'kor_nm', '한글 명')]:
    x = E[key]
    RL.append(['식품 유형', tt, cc, kk, '위 같은 테이블 행과 동일', x['domestic'], x['overseas'], '-', '원자료 품목코드·품목명(국내 검출 미계산)'])
r = table(ws, r, ['위해인자', '테이블', '컬럼(영문)', '컬럼(한글)', '키 매칭률', '국내 원값%', '수입 원값%', '국내 검출 시료 원값%', '원자료 특이사항'], RL, widths=[22, 26, 24, 22, 20, 10, 10, 12, 52])

r = section(ws, r, '3-6. 수분활성도 · 식품 pH · 염도 (ba_data.t_aw_cl, 품목코드 단위 문헌값)')
RA = []
for c, k in [('aw', '수분활성도'), ('ph', '식품 pH'), ('na', '염도'), ('mc', '수분함량')]:
    x = aw[c]
    RA.append([c, k, f"{n(x['table_nonblank'])} / {n(aw['rows'])}", n(x['range_text_rows']), x['domestic'], x['overseas'], x['detected_domestic']])
r = table(ws, r, ['컬럼(영문)', '컬럼(한글)', "값 있는 품목('-' 제외)/전체", '범위·부등호 텍스트 행(~·-·< 등)', '국내 시료 DB 값 있음%(코드 정확 일치, DB 채움 포함)', '수입 시료 DB 값 있음%', '국내 검출 시료 DB 값 있음%'], RA, widths=[12, 14, 22, 20, 20, 14, 14])
r = bullets(ws, r, [" - 값이 숫자가 아닌 텍스트: '0.91~0.98', '>0.98', '<0.6', '-' 등 → 수치로 쓰려면 범위 해석 규칙 결정 필요",
                     f" - 리서치팀 파일과 비교: aw 빈칸 {aw['file_aw_blank_codes']}코드 중 {aw['filled_codes']}코드가 DB에서 채워짐(aw_source {fmtd(aw['filled_codes_source'])}), 값 변경 {aw['changed_codes']}코드, 표기만 변경 54코드, 부등호 방향 반전 6코드(우유류, 시료 11,219행) → aw_source 표기만으로는 원값 구분 불가, 원값은 리서치팀 xlsx 기준",
                     ' - aw 원값%(리서치팀 파일 기준 = DB가 채운 128코드 제외): 국내 32.0% · 수입 42.3% · 국내 검출 12.0% (위 표는 DB 값 기준 38.6% · 49.6% · 13.0%). pH·염도·수분함량은 채움·변경 코드가 시료에 없어 차이 없음',
                     f" - 우리 시료 영향: 채워진 코드에 해당하는 행 {n(aw['sample_rows_filled'])} · 값이 바뀐 코드 행 {n(aw['sample_rows_changed'])} / 우리 시료 중 품목코드가 t_aw_cl에 정확히 있는 행 {pct(aw['code_match_pct'])}(우리 시료의 47%가 Z계열 품목코드)",
                     f" - aw 0.15 동일값 {aw['aw_015_rows']}행(DB 추가 코드 {aw['aw_015_db_only']}행 + 파일 값 0.48~0.93이던 5코드(건면 0.51·건조저장육류 0.85-0.93·기타 건포류 0.48~0.66 등)도 0.15로 적재) / aflatoxin_aw(0·1·2 등급)가 아플라톡신 전용 컬럼",
                     f" - 교차검증용: ba_data_info.food_moisture_info {n(fmi['n'])}품목(aw {n(fmi['n_aw'])}, 문헌 출처 {n(fmi['n_aw_src'])}) — 같은 품목명인데 값이 다른 사례(건조저장육류 0.65~0.92 vs t_aw_cl 0.15)"])

r = section(ws, r, '3-7. (참고) 위해인자는 아니지만 확인 요청된 변수 · 수질오염원 시설')
r = table(ws, r, ['원천', '컬럼(영문)', '컬럼(한글)', '연구기간 비결측%', '값 있는 지점/전체', '국내 시료 당일 원값%', '비고'], [
    ['에어코리아', 'pm10_value', '미세먼지(PM10) 농도', v(K, 'pm10_value')['nonnull_study'], f"{v(K,'pm10_value')['stations_with_value_study']}/{v(K,'pm10_value')['stations_total']}", cov('에어코리아', 'pm10_value')['sameday_pct'], '리스테리아 시트 빨간 글자 아님. 2015년 비결측 88.2%'],
    ['NAS', 'hum', '평균습도', v(N, 'hum')['nonnull_study'], f"{v(N,'hum')['stations_with_value_study']}/{v(N,'hum')['stations_total']}", cov('NAS', 'hum')['sameday_pct'], f"습도는 빨간 글자 아님. 0.0 {v(N,'hum')['rules']['값0']}건. ASOS avg_rhm은 99곳 99.8%"],
    ['water_pollution', 'sals_stts_nm 등', '수질오염원 설치시설', '-', f"{n(wp['n'])}개소(좌표 {n(wp['n_xy'])})", '미계산', f"영업상태 {fmtd(E['water_pollution_status'])} · 폐업일자 기재 {n(wp['n_close'])} · 인허가일 {wp['lc_min']}~{wp['lc_max']} · 인허가일 2026-06-30 이전이고 폐업일자가 비었거나 2015-04-07 이후인 시설 {n(wp['n_study'])}개소(상태가 폐업·취소인데 폐업일자 없는 시설 포함)"],
], widths=[14, 16, 18, 14, 16, 16, 80])
ws.freeze_panes = 'A4'

# ================================================================== 4. 수집 불가 인자 수집 방법
ws = wb.create_sheet('4_수집불가_수집방법')
r = title(ws, 1, '4. DB 수집 불가 위해인자 — 수집 방법', '출처 정보(기관·데이터명·형식·주기·URL)는 위해요소별_중요인자_데이터출처.xlsx 원문 / 수집·연결 방법은 제안 / 요청 기간 = 2015-04 ~ 2026-06 / 원자료는 받은 그대로 저장하고 집계·파생은 별도 결정')
def s(rn, i): return ('' if SRC[rn][i] is None else str(SRC[rn][i]).replace('\n', ' ').strip())
def srow(rn): return [s(rn, 2), s(rn, 1), f"{s(rn, 5)} / {s(rn, 6)}", s(rn, 7)]
R4 = [
    ['깊이별 토양수분(20·30cm) · 높이별 기온(40·50·400cm)·풍향·풍속·최대풍속', '출처 r6 (빨간)', *srow(6), '공공데이터포털 인증키 발급 → 관측지점·기간별 API 반복 호출 → 원값 저장', '관측지점 좌표 최근접 + 기준일(NAS와 같은 관측망이면 지점코드 대응 확인)', '매핑 시트3에 있음'],
    ['높이별 기온·풍향·풍속 · 깊이별 토양수분값(10·20·30cm)', '출처 r9 (빨간)', *srow(9), '공공데이터포털 API(팜맵 지번코드 단위 조회)', '시료 좌표·주소 → 팜맵 필지(지번코드) 매칭 방법 검토 필요', '매핑 시트3 누락 행'],
    ['기온·강수·바람(방재기상관측망)', '출처 r3 (빨간)', *srow(3), '기상청 API허브 API 호출', '관측지점 좌표 최근접 + 기준일', 'DB는 ASOS·NAS로 대체 중'],
    ['기온·풍속·토양수분·일조(농업기상관측망)', '출처 r4 (빨간)', *srow(4), 'CSV 파일 다운로드', '관측지점 좌표 최근접 + 기준일', '토양수분이 부분·대체라 보완 가치 있음'],
    ['토양수분량(관측단위 %)', '출처 r11 (빨간)', *srow(11), 'CSV 파일 다운로드', '용담 시험유역 6개 지점뿐 → 전국 시료 연결 거의 불가(참고용)', ''],
    ['유역별 토양수분량(m³/m³, 표준유역 850개)', '출처 r12 (빨간)', *srow(12), 'API 또는 CSV 다운로드', '표준유역 경계(SHP, 별도 확보) × 시료 좌표 공간조인 + 기준일', ''],
    ['시군별 토양유효수분율(%)', '출처 r13 (빨간)', *srow(13), '사이트에서 지점·시간 직접 입력 조회(API·파일 없음) → 수동 수집 또는 농진청 자료 요청', '시군 + 날짜', '자동 수집 경로 없음'],
    ['월별 미래 기후(월강수량·월평균/최고/최저기온)', '출처 r8 (빨간)', *srow(8), 'PDF 기후도면 다운로드', 'SSP/RCP 미래 예측 시나리오라 과거 시료 학습 변수로 부적합(향후 예측 시나리오용)', ''],
    ['해수 염분', '출처 r15 (빨간)', *srow(15), '공공데이터포털 API(격자·3분/15분) → 원 시계열 그대로 저장(일 단위 집계 여부·방식은 별도 결정)', '시료에 생산 해역 정보 없음(좌표 = 제조소 주소) → ① 연안 제조소 좌표 최근접 격자 + 기준일로 대체(생산 해역과 다를 수 있음) 또는 ② 수산물 어획·양식 해역 정보 별도 확보. 내륙·해외 제조소 시료 해당 없음', ''],
    ['해상풍 풍향·풍속', '출처 r16 (빨간)', *srow(16), '공공데이터포털 API(격자·15분) → 원 시계열 그대로 저장', '해수 염분과 동일(생산 해역 정보 없음)', ''],
    ['지하수 오염예측 등급', '출처 r18 (빨간)', *srow(18), 'SHP 파일 다운로드', '시료 좌표 point-in-polygon 공간조인(연 단위)', ''],
    ['지하수 오염취약성 DRASTIC 점수', '출처 r19 (빨간)', *srow(19), 'SHP 파일 다운로드', '시료 좌표 공간조인', ''],
    ['대기조성 — 이산화탄소 농도', '출처 r20 (빨간) · PPT 7순위', *srow(20), '기후정보포털 CSV 다운로드(국내 월평균·지점별)', '기준일 연월(+ 가까운 관측지점)', '전국 공통값에 가까워 변별력 확인 필요'],
    ['수질 — 농업저수지 수질정보', 'PPT 추가(수질) · 출처 r17 (빨간 아님)', *srow(17), 'API 또는 CSV(시설별 연 1회 조사값)', '원문 제공 항목에 주소·좌표 없음 → 시설코드·시설명으로 농어촌공사 저수지 제원(위치) 자료 별도 확보 후 시료 좌표 최근접·반경 + 조사연도', '빨간 글자 아님: 근거는 PPT'],
    ['LIMS 시료 단위 실측 저장·유통 온도', 'PPT 추가(저장 및 유통 온도)', '-', '-', '-', '-', '데이터출처에 출처 없음 → 리서치팀·식약처 확인 필요', '-', 'DB 대체: 운반 상태 온도·시료 상태·유통기한 문구 / 유통단계조사(distb_step_exam)는 연결 키 없음'],
    ['포장 방식(진공·가스치환)', 'PPT 추가(포장 유형)', '-', '-', '-', '-', '데이터출처에 출처 없음 → 리서치팀 확인 필요', '-', 'DB 대체: 품목제조대장 포장 재질 텍스트'],
    ['섭취 형태(그대로 섭취 여부)', 'PPT 추가(식품 유형)', '-', '-', '-', '-', '데이터출처에 출처 없음 → 리서치팀 확인 필요', '-', 'DB 대체: 품목 정의(dfn)·제조조건 텍스트'],
    ['(참고·DB 있음) 수질 — 수질오염물질 배출시설', 'PPT 추가(수질) · 출처 r21 (일부만 빨간)', *srow(21), f"추가 수집 불필요: DB api_data_info.water_pollution에 적재({n(wp['n'])}개소)", '좌표(TM 계열, 좌표계 확인 후 변환)·주소 기준 시료-시설 거리 원값 저장(반경 내 시설 수 등 파생은 별도 결정)', '행 대부분 파란 글자(표시 오류 가능) → 대상 여부 리서치팀 확인'],
    ['(참고·DB 있음) 수분활성도 · 식품 pH · 염도', 'PPT 1·3·4순위', '식품안전정보원 리서치팀', '통합_수분활성도_수분함량_식품pH_염도_최종.xlsx', 'XLSX / -', '-', '추가 수집 불필요: DB ba_data.t_aw_cl에 적재(채움값 섞임 → 원값은 xlsx 기준). 수분활성도·수분함량은 ba_data_info.food_moisture_info에도 문헌 출처와 함께 있음', '품목코드 일치(2번 시트 ⑮)', '사용 여부 결정 필요(1번 시트)'],
]
r = table(ws, r, ['위해인자', '근거', '제공기관', '데이터명', '현재상태·형식 / 생산 주기', 'URL', '수집 방법', '시료 연결 방법(제안)', '비고'], R4, widths=[30, 26, 22, 26, 30, 44, 44, 52, 34])
ws.freeze_panes = 'A4'

for w in wb.worksheets:
    w.sheet_view.zoomScale = 90
wb.save(OUT)
print('saved', OUT)
