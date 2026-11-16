import csv, os, re, collections
S = os.sep
rows = [r for r in csv.DictReader(open('files_repo.csv', encoding='utf-8-sig'))]


def agg(pred):
    s = c = 0
    m = ''
    for r in rows:
        if pred(r['rel']):
            s += int(r['size']); c += 1; m = max(m, r['mtime'])
    return round(s / 1e6, 1), c, m


def P(*prefixes, excl=(), rx=None, nrx=None):
    pf = [p.replace('/', S) for p in prefixes]
    ex = [e.replace('/', S) for e in excl]

    def f(rel):
        if not any(rel == p or rel.startswith(p) for p in pf):
            return False
        if any(rel.startswith(e) for e in ex):
            return False
        if rx and not re.search(rx, rel):
            return False
        if nrx and re.search(nrx, rel):
            return False
        return True
    return f


def J(*parts):
    return S.join(parts)


FR = '리스테리아ML2/exp/final_results/'
PRED = r'(pred(_thr)?_(fold\d|refit)|oof_(fold\d|refit))\.parquet$'
CACHE = r'__pycache__'
out = []


def add(loc, path, pred, content, cls, reason, dest='', flag='', size=None, cnt=None, mt=None):
    if pred is not None:
        size, cnt, mt = agg(pred)
    out.append([loc, path, size, cnt, mt, content, cls, reason, dest, flag])


L = '로컬(천영진 저장소)'
for p, d in [('리스테리아ML/.env', 'DB 접속 정보 파일'), ('보관/gpu_server.pem', '서버 키'),
             ('GPU 서버 사용법.md', '서버 접속 안내'), ('문서/테이블명세서(모델팀)/database계정.txt', 'DB 계정 문서')]:
    add(L, p, P(p), d, '제외(비밀)', '계정·키·내부 IP 포함 가능 - 내용 열람 안 함', '패키지 금지', '비밀')
add('로컬(.claude 작업 폴더)', 'C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg', None, 'ssh 설정', '제외(비밀)',
    '서버 접속 설정', '패키지 금지', '비밀', 0.0, 1, '2026-11-12')
for p in ['리스테리아ML2/data/외부_수집/식품안전지수/HACCP/출처.md',
          '리스테리아ML2/data/외부_수집/식품안전지수/scripts/dg_api_call.py',
          '리스테리아ML2/data/외부_수집/식품안전지수/scripts/haccp_fetch.py']:
    add(L, p, P(p), '공공데이터 API 호출·출처 기록', '확인 필요',
        '인증키 관련 단어 검색에 걸림(값 열람 안 함, 키 값이 박힌 패턴은 0건)', '넣는다면 키 제거 확인 뒤', '비밀 의심')

# 코드
add(L, '리스테리아ML2/exp/final (configs·lst·scripts·tests·pytest.ini·run_food2)',
    P('리스테리아ML2/exp/final/', excl=('리스테리아ML2/exp/final/configs_dom', '리스테리아ML2/exp/final/run_dom'), nrx=CACHE),
    '최종 실험 코드(라이브러리 lst·단계 스크립트·설정 json·테스트)', '가', '최종 모델을 만든 코드',
    '01_소스코드/실험코드', '최종 실행 사용 중')
add(L, '리스테리아ML2/exp/final/configs_dom·run_dom',
    P('리스테리아ML2/exp/final/configs_dom', '리스테리아ML2/exp/final/run_dom', nrx=CACHE),
    '국내만 버전 설정·실행 기록', '다', '국내만 버전 11-12 폐기 결정', '-', '최종 실행 사용 중(폴더 안) - 실행 뒤 처리')
add(L, '리스테리아ML2/exp/final/**/__pycache__', P('리스테리아ML2/exp/final/', rx=CACHE), '파이썬 캐시', '다', '캐시', '-',
    '최종 실행 사용 중(폴더 안) - 실행 뒤 처리')
add(L, '리스테리아ML2/exp/최종실험_계획_20261102.md', P('리스테리아ML2/exp/최종실험_계획_20261102.md'), '최종 실험 계획서', '가',
    '실험 설계·결정 근거', '09_검증대상버전 또는 별도 문서 폴더', '최종 실행 사용 중')
add(L, '리스테리아ML2/exp/교수님_지시_모음.md', P('리스테리아ML2/exp/교수님_지시_모음.md'), '교수님 통화·카톡 지시 원문', '확인 필요',
    '내부 회의 기록 - 외부 검증자에게 줄지 판단 필요(기본은 나)', '나(보관) 권장', '')
add(L, '리스테리아ML2/notebook/01~07_*.py·db_conn.py', P('리스테리아ML2/notebook/', nrx=CACHE),
    '원자료 받기·시군구 배정·기상·토양·SPI·식품속성·축사 집계 스크립트', '가', '데이터셋/ 가공 입력을 만든 코드(참고용)',
    '01_소스코드/참고용_입력데이터생성', 'db_conn.py 접속 방식 확인 필요(내용 미열람)')
add(L, '리스테리아ML2/notebook·data 아래 __pycache__',
    lambda r: (r.startswith(J('리스테리아ML2', 'notebook')) or r.startswith(J('리스테리아ML2', 'data'))) and CACHE in r,
    '파이썬 캐시', '다', '캐시', '-', '')
add(L, '리스테리아ML2/data/데이터셋/build_food_db_table.py', P('리스테리아ML2/data/데이터셋/build_food_db_table.py'),
    '식품속성 DB 고정표 생성 코드', '가', '최종 입력 생성 코드', '01_소스코드/참고용_입력데이터생성', '')

# 데이터
add(L, '리스테리아ML2/data/추출_20261107', P('리스테리아ML2/data/추출_20261107'),
    'DB 추출본 8표(+json 메타): 해외 기상·해외 토양pH·국가코드·시군구 좌표·예보 2종·LIMS 리스테리아·nation', '가',
    '최종 실험 입력 원천', '03_데이터/원천_DB추출', '')
add(L, '리스테리아ML2/data/추출_20261112', P('리스테리아ML2/data/추출_20261112'), 't_aw_cl 추출본(food2 판)', '가',
    '최종 실험 입력 원천', '03_데이터/원천_DB추출', '최종 실행 사용 중')
used_raw = ['raw_keco_airkorea_fine_dust_info', 'raw_keco_airkorea_obs_station', 'raw_kma_asos_daily_info',
            'raw_kma_ground_obs_station_info', 'raw_korea_region_code', 'raw_nas_agri_weather_daily_info',
            'raw_nas_agri_weather_obs_station', 'raw_soil_ph_info', 'raw_t_prdlst_cl']
RAW = J('리스테리아ML2', 'data', 'raw') + S
add(L, '리스테리아ML2/data/raw (최종 설정이 읽는 9개: ' + ' · '.join(used_raw) + ')',
    lambda r: r.startswith(RAW) and any(u + '.parquet' in r for u in used_raw),
    'DB 원자료(대기·ASOS·농업기상·행정코드·토양pH·품목분류)', '가', '최종 설정 파일이 직접 읽음', '03_데이터/원천_DB추출', '')
add(L, '리스테리아ML2/data/raw/raw_t_aw_cl*.parquet·출처_t_prdlst_cl_t_aw_cl_20261026.md',
    lambda r: r.startswith(RAW) and 't_aw_cl' in r, '옛 t_aw_cl 추출(10-16·10-26)과 출처 메모', '확인 필요',
    's0_6_static(옛 판) 설정만 읽음, food2 판은 추출_20261112 사용 - 최종이 어느 판인지 확인', '가 또는 나', '')
add(L, '리스테리아ML2/data/raw/해외_일별_기상_원본.parquet·해외_칼럼정보.csv',
    lambda r: r.startswith(RAW) and '해외_' in r, '해외 일별 기상 원본(10-29)', '나',
    '최종 코드에서 참조 없음(DB 값 global_weather_info 사용)', '보관', '')
DS = '리스테리아ML2/data/데이터셋/'
for p, d in [('시군구_대표좌표.parquet', '시군구 대표 좌표'), ('식정원_db_기상기후', '시군구 일별 기상'),
             ('식정원_db_토양', '시군구 일별 토양'), ('외부_지표/친환경_인증비율.parquet', '친환경 인증 비율'),
             ('외부_지표/지하수_정기수질검사.parquet', '지하수 수질'),
             ('품목3단계_식품속성_DB고정표_20261027', '품목 3단계 식품속성 고정표(+설명 md)'),
             ('리스테리아 생장식.xlsx', '생장식(전문가 제공)')]:
    add(L, DS + p, P(DS + p), d, '가', '최종 설정이 읽는 가공 입력', '03_데이터/가공입력', '')
add(L, DS + '시료_시군구.parquet', P(DS + '시료_시군구.parquet'),
    '시료 628,517행 + sgg_code (업체명 mnfctur_cmpny_nm·주소 addr·제품명 prduct_nm·위경도 포함)', '확인 필요',
    '최종 입력이지만 업체·주소 등 개인/영업 정보 포함', '03_데이터 (민감 열 제거·가명 처리 여부 결정)', '개인정보')
add(L, '리스테리아ML/data/listeria_data.parquet', P('리스테리아ML/data/listeria_data.parquet'),
    '시료 원자료 676,520×25 (업체명·주소·제품명·위경도 포함)', '확인 필요', '최종 코드가 읽는 시료 파일이지만 개인/영업 정보 포함',
    '03_데이터 (민감 열 처리 결정)', '개인정보')
DSR = J('리스테리아ML2', 'data', '데이터셋') + S
add(L, DS + '생장범위/·생장범위_품목3단계_DB고정표·외부_수집/시군구_일별_SPI·외부_지표/haccp_*',
    lambda r: r.startswith(DSR) and any(k in r for k in ('생장범위', '시군구_일별_SPI', 'haccp_')),
    '생장범위·HACCP·옛 SPI 가공본', '나', '최종 설정에서 참조 없음(과거 실험용)', '보관', '')
add(L, '리스테리아ML2/data/외부_데이터 (SPI csv·관측지점)', P('리스테리아ML2/data/외부_데이터'), '표준강수지수 원자료', '가',
    's0_3·s0_4_spi 설정이 읽음', '03_데이터/원천_외부', '')
CS = J('리스테리아ML2', 'data', '외부_수집', '축사')
add(L, '리스테리아ML2/data/외부_수집/축사 (시군구별_농장수·국가리스트·국경 geojson)',
    lambda r: r.startswith(CS) and '현황 데이터' not in r, '축사 집계·국가 목록·국경 경계', '가', '최종 설정이 읽음',
    '03_데이터/원천_외부', '')
add(L, '리스테리아ML2/data/외부_수집/축사/축사 현황 데이터(8.24) 2판 xlsx',
    lambda r: r.startswith(CS) and '현황 데이터' in r, '축사 현황 원본 2판(데이터팀 제공)', '확인 필요',
    '집계표의 원천. 농장 주소 등 개인정보 포함 여부 확인 필요(내용 미열람)', '가(원천) 또는 나', '개인정보 가능')
add(L, '리스테리아ML2/data/외부_수집/면적_20261107', P('리스테리아ML2/data/외부_수집/면적_20261107'), '지적통계·FAO 토지 면적', '가',
    's0_6_static 설정이 읽음', '03_데이터/원천_외부', '')
add(L, '리스테리아ML2/data/외부_수집/식품안전지수 (GAP·HACCP·음용지하수·친환경 원자료·웹조사·스크립트)',
    P('리스테리아ML2/data/외부_수집/식품안전지수', nrx=CACHE + r'|HACCP.출처\.md|dg_api_call|haccp_fetch'),
    '외부 지표 수집 원자료(친환경·지하수 가공본의 원천 포함)', '나',
    '최종은 데이터셋/외부_지표 가공본만 읽음. 원천은 원자료 목록으로 대체 권장', '보관(원천 목록만 패키지)', '')
add(L, '리스테리아ML2/data/시군구_코드_대응표.xlsx·중간/시군구_미배정_삭제행_요약.csv',
    lambda r: (r.startswith('리스테리아ML2') and r.endswith('시군구_코드_대응표.xlsx')) or r.endswith('시군구_미배정_삭제행_요약.csv'),
    '시군구 코드 대응·미배정 요약', '가', 'lst/samples.py 가 읽음', '03_데이터/가공입력', '')
add(L, '리스테리아ML2/data/중간 (나머지 점검표·외부지표 중간본)',
    lambda r: r.startswith(J('리스테리아ML2', 'data', '중간')) and not r.endswith('시군구_미배정_삭제행_요약.csv'),
    '시군구 배정·기상·토양 점검표, HACCP 등 외부지표 중간 산출물', '나', '최종 입력 아님. 처리 점검 근거로 일부 쓸 수 있음', '보관',
    '시군구_미배정_삭제행.csv 는 시료 행 정보 - 개인정보 확인')
add(L, '리스테리아ML2/data/.gitignore', P('리스테리아ML2/data/.gitignore'), 'git 설정', '나', '-', '-', '')

# 결과
add(L, FR + 'final (final_decision·importance·report)', P(FR + 'final/'),
    '최종 판정·순열 중요도·최종 보고 표(h별 지표·시험 점수·입력 변수 등)', '가', '최종 결과·보고', '05_성능검증 / 06_입출력정의',
    '최종 실행 사용 중')
add(L, FR + 's1/1-1/1-CAT-W060-sum', P(FR + 's1/1-1/1-CAT-W060-sum'), '1차 베이스라인(비교 기준 모델)', '가',
    '최종 판정의 비교 대상(baseline_1)', '02_학습모델파일/비교모델 (예측 parquet 는 05)', '최종 실행 사용 중')
add(L, FR + 's5_6/5-6-T/6-LSTM', P(FR + 's5_6/5-6-T/6-LSTM'), '6단계 LSTM(최종 아님)', '나', '비교 판', '보관', '최종 실행 사용 중')
add(L, FR + 'exports', P(FR + 'exports'), 'h별 지표 엑셀', '나', '중간 보고용', '보관', '최종 실행 사용 중')
for st in ['s0', 's1', 's2', 's3', 's4']:
    base = FR + st + '/'
    ex = (FR + 's1/1-1/1-CAT-W060-sum',) if st == 's1' else ()
    add(L, base + ' 예측 parquet(pred_*·oof_*)', P(base, excl=ex, rx=PRED), '판별 예측값(행×h) - 단계 ' + st, '다(확인 필요)',
        '큰 중간 산출물. 재학습으로만 다시 만들 수 있음(서버에서는 지워 이 로컬 사본이 유일) - 표·h_metrics 로 충분하면 삭제', '-',
        '최종 실행 사용 중')
    add(L, base + ' 나머지(모델·설정·h_metrics·점수·_code)', P(base, excl=ex, nrx=PRED), '단계 ' + st + ' 과거 판 모델·점수', '나',
        '과거 실험 기록', '보관', '최종 실행 사용 중')

# 기타 저장소
add(L, '리스테리아ML/.venv-exp2', P('리스테리아ML/.venv-exp2'), '옛 실험용 파이썬 가상환경', '다', '재설치 가능', '-', '')
add(L, 'Listeria/ (교수님 레포 사본, 자체 .git)', P('Listeria/'), '교수님 노트북 01~08·요약표', '나', '참고 코드(교수님 쪽)', '보관', '')
add(L, 'toxin_2026/ (팀 공용 레포 사본, 자체 .git)', P('toxin_2026/'), '위해요소 11종 공용 틀(아플라톡신 train·납 등)', '확인 필요',
    '리스테리아 전용 아님 - 다른 업무 자료. 04_리스테리아는 빈 틀', '패키지 제외', '')
add(L, '.git (저장소 기록)', P('.git/'),
    'git 기록 - 작업 폴더에서 지워진 옛 실험 2,334파일(ML2/exp/2_고도화 1,788·3_시간순교차검증 262·notebook/보고서 142·1_베이스라인 75·리스테리아ML/EXP·docs 등)이 여기에만 남음',
    '나', '과거 실험 원본 기록(GitHub 비공개 원격에도 있음)', '보관', '')
add(L, '.claude/ (설정)', P('.claude/'), 'Claude 설정', '제외', '작업 도구 설정', '-', '')
add(L, 'grep.exe.stackdump', P('grep.exe.stackdump'), '프로그램 오류 덤프', '다', '임시', '-', '')
for p, d, c, why in [('LLM_WIKI.MD', '작업 메모', '확인 필요', '내용 확인 후 나'), ('모델 학습시 유의사항.md', '유의사항 메모', '나', '참고'),
                     ('모델 성과 발표 예시.docx', '발표 예시', '나', '참고 양식')]:
    add(L, p, P(p), d, c, why, '보관', '')
add(L, '문서/개별보고서_양식', P('문서/개별보고서_양식'), '개별 보고서 리스테리아 v1·패류독소 v1.2 양식', '나', '보고서 양식', '보관', '')
add(L, '문서/리서치팀문서', P('문서/리서치팀문서'), '모델적용방안 PDF 2판·위해요소 데이터 현황·중요인자 출처·수분활성도 표', '나', '참고 문헌·회의 자료',
    '보관', '')
add(L, '문서/센터_발표자료_사업보고서', P('문서/센터_발표자료_사업보고서'), '통합본 예측모델 pptx·사업보고서 양식', '나', '센터 공용 자료', '보관', '')
add(L, '문서/영진문서', P('문서/영진문서'), '경복대 2차 보고서·식품안전지수 길라잡이·protocol', '나', '참고 문헌', '보관', '')
add(L, '문서/테이블명세서(모델팀) (계정 파일 제외)',
    P('문서/테이블명세서(모델팀)', excl=('문서/테이블명세서(모델팀)/database계정.txt',)), '테이블 목록·정의서', '나',
    'DB 구조 참고(입력 정의서 작성 근거)', '보관', '')
add(L, '문서/가이드.txt', P('문서/가이드.txt'), '가이드 메모', '확인 필요', '내용 미확인', '-', '')
add(L, '보관/개별보고서_원본·.gitignore', P('보관/개별보고서_원본', '보관/.gitignore'), '개별 보고서 원본 양식(09-01)', '나', '과거 양식',
    '보관', '')

# 그 밖의 위치
X = [
    ('Desktop', 'C:/Users/NT751/Desktop/11.09~11.13 리스테리아 추진계획.txt', 0.0, 1, '2026-11-11', '주간 추진계획 메모', '나', '회의·계획 자료', '보관', ''),
    ('Amaranth10', 'C:/Users/NT751/Documents/Amaranth10/리스테리아 모델적용방안 실험용.pdf', 0.1, 1, '2026-11-12',
     '리서치팀 적용방안(실험용 판, 문서/리서치팀문서의 수정중 판과 MD5 다름)', '나', '참고 문헌', '보관', ''),
    ('Amaranth10', 'C:/Users/NT751/Documents/Amaranth10/04_개별 보고서_리스테리아_제4절_양식.hwp', 1.6, 1, '2026-11-11', '개별 보고서 4절 양식',
     '나', '보고서 양식', '보관', ''),
    ('Amaranth10', 'C:/Users/NT751/Documents/Amaranth10/축사 현황 데이터(8.24).xlsx · _최종v2.xlsx · 국가리스트_1_농장전체.csv', 2.8, 3,
     '2026-11-06', '외부_수집/축사 와 MD5 동일 사본', '다', '중복(저장소에 같은 파일)', '-', '개인정보 가능'),
    ('Amaranth10', 'C:/Users/NT751/Documents/Amaranth10/축사조인_domestic_결과.parquet · 축사조인_overseas_결과.parquet', 13.6, 2,
     '2026-11-04', '데이터팀 축사 반경 조인 결과(lims_data_key·위경도·반경별 농장 수)', '나', '최종 미사용, 시료 좌표 포함', '보관', '개인정보 가능(좌표)'),
    ('Amaranth10', 'C:/Users/NT751/Documents/Amaranth10/경복대학교 2차 보고서(전처리 수정 후).hwp', 3.1, 1, '2026-10-19',
     '문서/영진문서 와 MD5 동일 사본', '다', '중복', '-', ''),
    ('Amaranth10', 'C:/Users/NT751/Documents/Amaranth10/정보보안 점검 관련 세부내용 안내.hwp', 0.05, 1, '2026-11-12', '보안 점검 안내', '제외',
     '리스테리아 무관', '-', ''),
    ('Downloads', 'C:/Users/NT751/Downloads/패류독소_기술검증_제출패키지', 388.5, 106, '2026-11-12', '양식(타 과제 제출본)', '제외', '양식 참고용', '-', ''),
    ('Documents', "C:/Users/NT751/Documents/%I9F9'T 폴더 · %I9f9't.jpg", 1.9, 10, '2026-11-12',
     '같은 크기의 bmp·doc·hwp·xls 등 9종 - 보안 프로그램 감시용 파일로 보임(추정)', '제외', '리스테리아 무관, 건드리지 말 것', '-', ''),
    ('Documents', 'C:/Users/NT751/Documents/Windows.iso', 6527.0, 1, '2026-08-28', '윈도우 설치 이미지', '제외', '리스테리아 무관', '-', ''),
    ('.claude 작업 폴더', 'C:/Users/NT751/.claude/jobs/a19471fc/tmp (dedup·results_mirror·smoke_s1·review 등)', 988.6, 2372, '2026-11-12',
     '이 작업 세션의 임시 산출물(중복 제거 작업본·결과 미러·스모크 테스트)', '다(확인 필요)', '임시. 데이터 사본이 섞여 있을 수 있어 보안 감사 전 정리 대상', '-',
     'ssh_gpu.cfg 포함'),
]
for x in X:
    out.append(list(x))

SV = '서버 ~/cyj_listeria_final'
SX = [
    ('results/s5_6/5-6-T/5-LSTM', 306, 33, '2026-11-11',
     '최종 모델(model_fold1~3·model_refit, inputs_*/dl_preprocessor.json, columns·applied_params, 예측 parquet 4개 약 312MB, h_metrics, cv_scores, test_scores)',
     '가', '최종 모델 - 로컬에 아직 없음(로컬 s5_6 은 6-LSTM 만). 내려받아 MD5 대조 필요', '02_학습모델파일/최종모델 (예측 parquet 는 05)', '최종 실행 사용 중'),
    ('results/final', 0.3, '', '2026-11-12', 'final_decision·importance·importance_logs·report·run_meta', '가',
     '최종 보고 (로컬 final_results/final 과 대조 필요)', '05_성능검증', '최종 실행 사용 중'),
    ('results/s5_6 (stage_table·sel-hon5·sel-hon6·5-6-judge·_code)', 3, '', '2026-11-11', '5·6단계 비교표·행 선택 정보·판정', '가',
     'sel-hon5 는 최종 모델 row_plan 이 참조', '02·05', '최종 실행 사용 중'),
    ('results/s1/1-1/1-CAT-W060-sum', 325, '', '2026-11-08', '베이스라인 비교 모델', '가', '최종 판정 baseline_1', '02 비교모델', '로컬 사본 있음'),
    ('results/logs', 0.3, '', '2026-11-12', 'experiment_log·chain_state·all_stage_test_table·로컬 이동 기록', '가',
     '전 단계 실험 이력(검증 대상 버전·재현 근거)', '09_검증대상버전', ''),
    ('results/s0~s4 나머지 (s0 197M·s1 1.6G·s2 465M·s3 2.5G·s4 149M, 베이스라인 제외)', 4600, '', '2026-11-11',
     '단계 0~4 판들(대부분 예측 parquet 는 로컬로 옮김)', '나', '과거 단계 결과', '보관(로컬과 합쳐 하나만)', ''),
    ('results/s8 (8-D-LSTM·8-D-tune·judge)', 309, '', '2026-11-11', '8-D 딥러닝 튜닝 - 최종 선택 안 됨', '나', '과거 실험, 로컬에 없음',
     '보관(필요 시 내려받기)', ''),
    ('results/_superseded·_speed_check·exports', 0.5, '', '2026-11-11', '대체된 최종 판정·속도 점검·엑셀', '다', '임시·중복', '-', ''),
    ('code_frozen (11개 사본 s0_9~s3g, 각 2.4~4.4MB)', 38, '', '2026-11-12', '단계별 실행 당시 얼린 코드', '나',
     '실행 시점 코드 기록(최종 모델을 만든 사본이 어느 것인지 확인 필요)', '보관(최종 모델 만든 사본은 09 에 해시 기록)', '최종 실행 사용 중일 수 있음(s3g 11-12)'),
    ('code', 4.1, '', '2026-11-12', '로컬 exp/final 과 같은 구성의 작업 사본', '다', '로컬 원본과 중복', '-', '최종 실행 사용 중'),
    ('data (리스테리아ML/data 49M·리스테리아ML2/data 1.8G)', 1850, '', '2026-11-07', '로컬 입력 데이터 미러', '다',
     '로컬과 중복(시료 원자료 포함 - 실행 끝나면 지울 대상)', '-', '최종 실행 사용 중·개인정보'),
    ('mlflow (artifacts 262M·mlflow.db 21M·서버 로그)', 284, '', '2026-11-12', '실험 추적 기록', '나', '실험 이력', '보관', ''),
    ('logs (run_*.sh/log·pip_freeze*.txt·patch 스크립트)', 0.4, '', '2026-11-12', '실행 스크립트·로그·패키지 목록', '나(pip_freeze 는 가)',
     'pip_freeze_20261107.txt 는 실행환경 근거', '04_실행환경 (pip_freeze)', ''),
    ('venv', 9400, '', '2026-11-07', '서버 파이썬 환경', '다', '재설치 가능', '-', '최종 실행 사용 중'),
    ('tmp', 1.7, '', '2026-11-12', '코드 테스트·환경 점검 임시', '다', '임시', '-', ''),
]
for p, s, c, m, d, cl, why, dest, fl in SX:
    out.append([SV, p, s, c, m, d, cl, why, dest, fl])
out.append(['서버 홈', '~/norovirus 6.1G · ~/patulin_optuna 9.0G · ~/*.tar.gz', 15100, '', '2026-11-12', '다른 과제 자료', '제외',
            '리스테리아 무관 - 건드리지 않음', '-', ''])

with open('C:/Users/NT751/.claude/jobs/a19471fc/tmp/inventory.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh)
    w.writerow(['위치', '경로(묶음)', '크기MB', '파일수', '최근수정일', '내용', '분류', '이유', '패키지 대상 폴더', '표시(비밀·사용중·개인정보)'])
    w.writerows(out)

t = collections.defaultdict(float)
for r in out:
    loc = '서버' if r[0].startswith('서버') else '로컬'
    t[(loc, r[6])] += float(r[2] or 0)
for k, v in sorted(t.items()):
    print(k, round(v, 1))
# 저장소 전체 대비 누락 점검
tot = sum(int(r['size']) for r in rows) / 1e6
print('repo total MB', round(tot, 1), 'inventory local-repo MB', round(sum(float(r[2] or 0) for r in out if r[0] == L), 1))
