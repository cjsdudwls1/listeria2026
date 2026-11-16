p = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'
b = open(p, 'rb').read()
crlf = b'\r\n' in b
t = b.decode('utf-8').replace('\r\n', '\n')


def rep(old, new, cnt=1):
    global t
    n = t.count(old)
    assert n == cnt, (n, old[:90])
    t = t.replace(old, new)


FILE = '`리스테리아ML2\\data\\외부_수집\\축사\\나라경계_NaturalEarth_v5.1.2_10m.geojson`(+ 같은 이름 .zip)'
rep('(나라 코드 붙은 다각형)를 주고 점이 든 나라로 세어 달라고 요청 [권고](§7-3)\n',
    '(나라 코드 붙은 다각형)를 주고 점이 든 나라로 세어 달라고 요청 [권고](§7-3)\n'
    f'- 11-04 밤 7 반영: 나라 경계 파일 만듦(사용자 "ㅇㅇ 만들어") — {FILE}. 결과는 나라마다 축종별 1개 값(§7-3)\n')
rep('국내 전국 목록이 있으면 같은 형식으로 요청(지금 엑셀은 모집 지역 28곳뿐).',
    '국내 전국 목록이 있으면 같은 형식으로 요청(지금 엑셀은 모집 지역 28곳뿐).\n'
    f'    - 경계 파일(11-04 밤 7): {FILE} — Natural Earth v5.1.2 1:10m admin-0(공공 영역, GitHub `nvkelso/natural-earth-vector` 태그 v5.1.2 의 geojson), 나라·지역 258개, 12.4MB(zip 4.5MB), WGS84 경위도, 칸 = iso3(`ISO_A3_EH`, 없으면 `ADM0_A3`)·adm0_a3·name_en·name_ko·sovereignt.\n'
    '      - DB `api_data_info.global_weather_info` 나라 249개 중 11개(보네르·부베·코코스·크리스마스섬·과들루프·프랑스령 기아나·마르티니크·마요트·레위니옹·스발바르·토켈라우 — 작은 해외 영토)는 경계 파일에서 본국 다각형에 들어 있음.\n'
    '      - 같은 iso3 가 여러 조각인 것: AUS·BRA·FRA·KAZ → iso3 로 합쳐 세 달라고 요청. 어느 경계에도 안 드는 좌표 수도 받음.\n'
    '      - 결과 = 나라마다 축종별(낙농장·소·돼지·가금) 1개 값, 날짜와 무관(한 시점 목록) → 해외 행은 그 나라 값, 목록이 안 덮는 나라는 NaN. 10km²당 = 수 ÷ 같은 경계의 나라 면적(우리가 계산).')

out = t.replace('\n', '\r\n') if crlf else t
open(p, 'wb').write(out.encode('utf-8'))

M = r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----\memory\listeria-livestock-join-1103.md'
m = open(M, encoding='utf-8').read()
anchor = '\n관련 [[listeria-1102-final-integrated-plan]], [[listeria-production-unit-inputs]].'
assert m.count(anchor) == 1
m = m.replace(anchor, '\n- **11-04 밤 7**: 나라 경계 파일 만듦(사용자 "ㅇㅇ 만들어") — `리스테리아ML2\\data\\외부_수집\\축사\\나라경계_NaturalEarth_v5.1.2_10m.geojson`(+ .zip 4.5MB), Natural Earth v5.1.2 1:10m admin-0, 258개, 칸 iso3·adm0_a3·name_en·name_ko·sovereignt. DB 나라 249 중 작은 해외 영토 11개는 본국에 포함, iso3 겹침 AUS·BRA·FRA·KAZ → 합쳐 세 달라고. 결과 = 나라마다 축종별 1개 값(날짜 무관).\n' + anchor)
open(M, 'w', encoding='utf-8').write(m)
print('ok')
