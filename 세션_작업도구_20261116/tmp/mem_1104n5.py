M = r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----\memory'

# plan memory
p = M + r'\listeria-1102-final-integrated-plan.md'
t = open(p, encoding='utf-8').read()
lines = t.split('\n')
idx = [i for i, l in enumerate(lines) if l.startswith('- **11-04 밤 3(')]
assert len(idx) == 1
lines.insert(idx[0] + 1,
             '- **11-04 밤 4(계획서 1,784줄, 표 어긋남 0)**: SPI 도 window E 끝 — 예보 11일은 예보 강수로 SPI 식 직접 계산(사용자 "예보 할 수 없으면 예보된 강수량으로 직접 가뭄지수 계산 식으로 계산해서 11일 치 넣는 식으로 하자"), '
             '결정 17 다시 열림: 관측 날도 같은 식으로 다시 계산 권고(기상청 값과 이어 붙이면 T 에서 값이 튐 — 11-03 확인 r 0.94~0.96·MAE 0.22~0.26, 이음매 위치가 h 마다 달라 딥러닝 입력이 h 에 따라 달라짐), 국내 1991~2020 ASOS 일강수 필요(수집은 지시 뒤). '
             '축사 L1 = 낙농장(젖소)·소·돼지·가금 4칼럼 확정(사용자 "ㅇㅇ") → 비기상 38·G = 46·C 155. 해외 축사: 데이터 팀에 한국 밖 축사 x·y 좌표 전체가 있음(사용자) → 좌표 목록을 받아 나라 경계로 우리가 나라별로 셈 권고, 원천이 나라 전체를 덮는지 확인, 단위는 국내·해외 모두 밀도 권고(사용자 답 대기).')
open(p, 'w', encoding='utf-8').write('\n'.join(lines))

# index
p = M + r'\MEMORY.md'
t = open(p, encoding='utf-8').read()
a = "29 = 교수님 방식(Theil's U 1위 = 제품명, train 행만), 남은 것"
assert t.count(a) == 1
t = t.replace(a, "29 = 교수님 방식(Theil's U 1위 = 제품명, train 행만), 밤 4 SPI 예보 11일 = 예보 강수로 식 계산·17 다시 열림·축사 4칼럼, 남은 것 17·축사 단위·해외 축사 좌표")
open(p, 'w', encoding='utf-8').write(t)

# livestock memory
p = M + r'\listeria-livestock-join-1103.md'
t = open(p, encoding='utf-8').read()
anchor = '\n관련 [[listeria-1102-final-integrated-plan]], [[listeria-production-unit-inputs]].'
assert t.count(anchor) == 1
t = t.replace(anchor, '\n- **11-04 밤 4**: 사용자 "축사 변수에 소·돼지·가금 전체 농장 수도 넣을지->ㅇㅇ" → L1 = 낙농장(젖소)·소·돼지·가금 4칼럼(가축종 글자로 나눔, 둘 이상이면 각각 셈). '
                  '사용자 "해외 축사는 나라별이 아니라 그냥 한국제외 모든 축사들의 x,y좌표가 있대" → 권고: 좌표 목록 파일(x·y·축종·원천·기준일)을 받아 국가 경계(예: Natural Earth admin-0)로 우리가 나라별로 셈, 원천이 나라 전체를 덮는지 확인(지난 결과 값은 북이탈리아·남독일·오스트리아·덴마크·뉴질랜드 쪽에만), 단위는 국내·해외 모두 밀도(개/10km²) 권고 — 사용자 답 대기.\n' + anchor)
open(p, 'w', encoding='utf-8').write(t)

# SPI memory
p = M + r'\spi-drought-index-data.md'
t = open(p, encoding='utf-8').read().rstrip('\n')
t += ('\n\n- **11-04 밤 4(최종 계획 결정 5·17)**: SPI 도 window 예보 끝(E)까지 — 예보 11일은 예보 강수로 SPI 식 직접 계산(사용자 "예보 할 수 없으면 예보된 강수량으로 직접 가뭄지수 계산 식으로 계산해서 11일 치 넣는 식으로 하자"). '
      '결정 17(국내 기상청 값 그대로)은 다시 열림 — 관측 날도 같은 식으로 다시 계산 권고(기상청 값과 이음매·h 의존 문제), 국내 1991~2020 ASOS 일강수 필요. [[listeria-1102-final-integrated-plan]]\n')
open(p, 'w', encoding='utf-8').write(t)
print('ok')
