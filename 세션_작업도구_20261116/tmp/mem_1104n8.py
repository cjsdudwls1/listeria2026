M = r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----\memory'

p = M + r'\listeria-1102-final-integrated-plan.md'
t = open(p, encoding='utf-8').read()
lines = t.split('\n')
idx = [i for i, l in enumerate(lines) if l.startswith('- **11-04 밤 5(')]
assert len(idx) == 1
lines.insert(idx[0] + 1,
             '- **11-04 밤 6(계획서 1,797줄, 표 어긋남 0)**: SPI 결정 17 = 국내 과거 기상청 값 + 예보 11일만 우리 식, 해외 과거·예보 모두 우리 식(사용자 "우리가 계산해서 넣는값은 국내 예보 11일, 해외는 과거와 예보 전부야"). '
             '국내 예보 11일은 T−1 차이만큼 옮겨 잇기 권고 — 시험: 그냥 이으면 T 에서 0.28·0.27 튐(기상청 하루 변화 0.077·0.058), 옮겨 이으면 예보 날 차이 SPI3 0.034→0.127·SPI6 0.026→0.103(10일째). '
             '해외 습도 = DB global_weather_info.relative_humidity_mean 있음, 이슬점은 DB 없고 Open-Meteo(ERA5) 있음 → 두 변수 유지. 계절 원핫 4칸을 딥러닝 정적 입력에도(x_static (N, 6)). '
             '해외 축사: 데이터 팀 원본 공유 불가·나라 칸 없음 → 나라 경계 다각형(Natural Earth admin-0, ISO3 = DB alpha_3)을 주고 나라별·축종별 수를 받는 안 권고(경계 자료 받기는 지시 뒤).')
open(p, 'w', encoding='utf-8').write('\n'.join(lines))

p = M + r'\MEMORY.md'
t = open(p, encoding='utf-8').read()
a = '밤 5 SPI 우리 식(2015~ 적합)·계절 4범주·축사 10km²당 4칼럼·KO 제외(수정중 PDF 폐기), 남은 것 해외 축사 좌표(데이터 팀)'
assert t.count(a) == 1
t = t.replace(a, '밤 6 SPI 국내=기상청 과거+예보 11일 우리 식(옮겨 잇기)·해외 전부 우리 식, 계절 4범주(딥러닝 원핫도)·축사 10km²당 4칼럼·KO 제외, 남은 것 해외 축사(나라 경계 다각형 주고 데이터 팀 조인)')
open(p, 'w', encoding='utf-8').write(t)

p = M + r'\spi-drought-index-data.md'
t = open(p, encoding='utf-8').read().rstrip('\n')
t += ('\n- **11-04 밤 6 바뀜(결정 17)**: 국내 = 과거 기상청 SPI 파일 값 + 예보 11일만 우리 식, 해외 = 과거·예보 모두 우리 식(사용자 "우리가 계산해서 넣는값은 국내 예보 11일, 해외는 과거와 예보 전부야. 알지?"). '
      '국내 예보 11일은 T−1 의 차이(기상청 − 우리 식)만큼 옮겨 이음 권고 — 시험(job tmp `spi_anchor.py`): 그냥 이으면 T 에서 평균 0.28(SPI3)·0.27(SPI6) 튐, 옮겨 이으면 예보 날 차이 0.03~0.13. 운영 때 기상청 SPI 를 날마다 받는 방법은 운영 준비 때 확인.\n')
open(p, 'w', encoding='utf-8').write(t)

p = M + r'\listeria-livestock-join-1103.md'
t = open(p, encoding='utf-8').read()
anchor = '\n관련 [[listeria-1102-final-integrated-plan]], [[listeria-production-unit-inputs]].'
assert t.count(anchor) == 1
t = t.replace(anchor, '\n- **11-04 밤 6**: 데이터 팀 = 해외 축사 원본은 너무 커서 공유 불가, 우리가 데이터를 주면 조인해 줌, 원본에 나라 칸이 없어(좌표만) 나라 목록으로는 조인 불가. 권고 = 나라 경계 다각형(Natural Earth admin-0, 공공 영역, ISO3 = DB `alpha_3`, 해외 시료 제조국만)을 주고 점이 든 나라별·축종별 농장 수 + 출처·기준일·수록 범위를 받음, 면적은 같은 경계로 우리가 계산. 경계 자료 받기는 지시 뒤.\n' + anchor)
open(p, 'w', encoding='utf-8').write(t)
print('ok')
