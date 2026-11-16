M = r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----\memory'

# pending research report: rewrite body
p = M + r'\pending-research-variables-report.md'
t = open(p, encoding='utf-8').read()
head, _, _ = t.partition('\n---\n')
fm_end = t.index('\n---\n', 4) + 5
front = t[:fm_end]
front = front.replace('PDF p12 변수 1~7번·낙농장 밀도·Koutsoumanis 2식', 'PDF(14쪽 판) 일사량·축사·p12 표 변수')
body = '''
2026-11-04 사용자 지시: 실험이 끝나면 식품 선임연구원께 리서치팀 자료 변수의 효과를 "이 pdf의 이 문헌에 따라 이러한 변수를 넣었습니다 그 결과들은 이렇습니다" 형식으로 정리 — 사용자가 그대로 복붙해 보냄. "내가 까먹을 수도 있으니 잘 기억하셈".

- **보고 범위(11-04 밤 5 바뀜)**: `문서\\리서치팀문서\\리스테리아 모델적용방안.pdf`(14쪽 판)의 "일사량이랑 축사 변수랑 12페이지 표 정리 된 예측 변수 후보 변수들" — p12 표 1~7번(8번 관개·경운 뺌; 5번이 일사량): Murphy 2023 72시간 평균기온(t3)·Ivanek 2009 2일 전 강수(rain_lag2)·Pang 2017 2일 전 평균 풍속(wind_lag2)·Pang 2017 25일 평균 강수(rain_mean25)·Weller 2020 2~3일 전 평균 일사(solar_23)·Chandler-Khayd 2023 7일 기온 변동폭(trange7)·Murphy 2023 24시간 평균 습도·이슬점(rh1·td1) + Weller 2020 축사 밀도(L1 4칼럼: 낙농장(젖소)·소·돼지·가금, 10km²당 농장 수 — 국내 시군구·해외 나라).
- **수정중 판 PDF(Koutsoumanis 2식) 폐기**(사용자 11-04 밤 5 "전부 무시하셈") — KO1·KO2 실험에서 제외, 연구원께 보낼 식 오류 문자도 필요 없음. 보고에 Koutsoumanis 안 넣음.
- 정의 근거(11-04 원문 확인): Weller 방법 "0–1, 1–2, 2–3 … days before sample collection" → 2~3일 전 = 48~72시간 전 → 일자료 R−3·R−2 평균. Chandler-Khayd "average temperature range in the seven days prior to sampling" → 7일 일교차 평균. 습도·이슬점 = 받아 둔 ASOS `avg_rhm`·`avg_td`(빈 날 NAS `hum`) 가까운 3곳 평균, 전날 값.
- 비교 판(계획 §7-4): 기준 = 단계 2 최종 판, 3-R1 기준 + p12 8칼럼 + 축사 4칼럼, 3-R2~3-R10 하나씩(t3·rain_lag2·wind_lag2·rain_mean25·solar_23·trange7·rh1·td1·축사 4칼럼 묶음), 학습 40회, 칼럼별 중요도·단계 3 빼기에서 남는지.

**Why:** 사용자가 외부 연구원에게 그대로 보낼 결과라 빠뜨리면 안 되고, 사용자가 잊을 수 있다고 명시함.
**How to apply:** 실험 결과 보고 때(단계 3 이후·최종 보고) 반드시 이 보고를 먼저 꺼내 표 1개 + 짧은 문장, 내부 약어 없이 작성. 보고 전까지 평소엔 언급 불필요. 관련 [[listeria-1102-final-integrated-plan]], [[listeria-livestock-join-1103]], [[feedback-report-plain-language]].
'''
open(p, 'w', encoding='utf-8').write(front + body)

# plan memory
p = M + r'\listeria-1102-final-integrated-plan.md'
t = open(p, encoding='utf-8').read()
lines = t.split('\n')
idx = [i for i, l in enumerate(lines) if l.startswith('- **11-04 밤 4(')]
assert len(idx) == 1
lines.insert(idx[0] + 1,
             '- **11-04 밤 5(계획서 1,791줄, 표 어긋남 0)**: 수정중 판 PDF 폐기("전부 무시하셈") → KO1·KO2 제외, 리서치 보고 = 14쪽 판 일사량·축사·p12 1~7번(§7-4 3-R1~3-R10, 40회). '
             'SPI 결정 17 확정 = 과거 강수 + 11일 예보 강수로 우리 식(국내·해외 같은 식), 적합 기간 2015-01-01 ~ 2025-06-30(DB ASOS 일자료는 2015-01-01 ~ 2026-10-13·99지점뿐 — 11-04 밤 조회, 1991~ 없음), 기상청 SPI 파일 안 씀, 해외 SPI 용 1991~ 강수 수집 뺌. '
             '계절 = 4개 범주(봄 3~5·여름 6~8·가을 9~11·겨울 12~2, d 의 달) 단계 3 파생변수(결정 7). 축사 단위 = 국내·해외 10km²당 농장 수(결정 35), 데이터 팀 요청(해외 좌표 원본·출처 범위·국내 전국 목록) §7-3. G = 47·C 154·단위 149.')
open(p, 'w', encoding='utf-8').write('\n'.join(lines))

# index
p = M + r'\MEMORY.md'
t = open(p, encoding='utf-8').read()
a = '밤 4 SPI 예보 11일 = 예보 강수로 식 계산·17 다시 열림·축사 4칼럼, 남은 것 17·축사 단위·해외 축사 좌표'
assert t.count(a) == 1
t = t.replace(a, '밤 5 SPI 우리 식(2015~ 적합)·계절 4범주·축사 10km²당 4칼럼·KO 제외(수정중 PDF 폐기), 남은 것 해외 축사 좌표(데이터 팀)')
a2 = '— 실험 끝나면 꼭: PDF p12 변수 1~7·낙농장 밀도·Koutsoumanis 2식의 기존 모델 대비 향상·어떤 모델·변수가 나은지, 연구원께 복붙용 정리 (11-04)'
if t.count(a2) == 1:
    t = t.replace(a2, '— 실험 끝나면 꼭: 14쪽 PDF 일사량·축사·p12 변수의 기존 모델 대비 향상·어떤 변수가 나은지, 연구원께 복붙용 정리(수정중 PDF·Koutsoumanis 폐기) (11-04)')
else:
    print('index pending line not found')
open(p, 'w', encoding='utf-8').write(t)

# SPI memory
p = M + r'\spi-drought-index-data.md'
t = open(p, encoding='utf-8').read().rstrip('\n')
t += ('\n- **11-04 밤 5 확정(결정 17)**: SPI = 과거 강수 + 11일 예보 강수로 우리 식 계산(사용자 "11일치 강수 예보와 과거 강수로 가뭄지수도 계산한다 라는 식으로 우리는 그냥 이렇게 했다 하면됨"), 국내·해외 같은 식, 기상청 SPI 파일 안 씀. '
      '적합 기간 = 2015-01-01 ~ 2025-06-30(사용자 "받지말고 그냥 지금 있는데이터로만 계산 ㄱ"; DB `api_data_info.kma_asos_daily_info` 는 2015-01-01 ~ 2026-10-13·99지점뿐 — 11-04 밤 조회). 시군구 = 가까운 ASOS 3곳 일강수 평균 권고.\n')
open(p, 'w', encoding='utf-8').write(t)

# livestock memory
p = M + r'\listeria-livestock-join-1103.md'
t = open(p, encoding='utf-8').read()
anchor = '\n관련 [[listeria-1102-final-integrated-plan]], [[listeria-production-unit-inputs]].'
assert t.count(anchor) == 1
t = t.replace(anchor, '\n- **11-04 밤 5**: 단위 = 국내·해외 모두 10km²당 농장 수 확정(사용자). 4칼럼 모두 리서치 보고 변수. 데이터 팀 요청 = 줄 것: 해외 시료 제조국 목록(나라 코드·영문명, 시료 행·검출 X) / 받을 것: ① 한국 밖 축사 좌표 원본(위도·경도·좌표계·가축종·출처·기준일) ② 출처가 나라마다 전체를 담는지 ③ 국내 전국 목록 있으면 같은 형식. 조인·나라 배정·면적은 우리가.\n' + anchor)
open(p, 'w', encoding='utf-8').write(t)
print('ok')
