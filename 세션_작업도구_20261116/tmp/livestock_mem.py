p = r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----\memory\listeria-livestock-join-1103.md'
t = open(p, encoding='utf-8').read()
anchor = '\n관련 [[listeria-1102-final-integrated-plan]], [[listeria-production-unit-inputs]].'
assert t.count(anchor) == 1
add = ('\n- **11-04 밤 방향 바뀜(사용자 "국내는 시군구 갯수로 다시 조인하면 될거같고 해외는 좌표 단위라 … 데이터 팀한테 우리 해외좌표데이터에서 국가 단위 좌표로 만들어서 전달 해줘야할듯")**: '
       '`Documents\\Amaranth10\\` 에 11-04 14:15·14:24 새로 받은 `축사 현황 데이터(8.24).xlsx`(19,098곳 — 소재지 주소·위도·경도·조사연도·데이터기준일·축사 개수·가축종·사육두수 등) '
       '와 옛 구성과 같은 `축사조인_domestic_결과.parquet`(628,878행)·`축사조인_overseas_결과.parquet`(47,642행, 소 5km > 0 1,741행·돼지 765·가금 72, `farm_unmapped` 전부 0) 있음. '
       '엑셀은 주소 앞 두 단어 40가지(이름 변형·세종 읍면동 포함) → 모집 지역 약 28 시군구뿐, 젖소 포함 831곳, 데이터기준일 2022-04 ~ 2026-07. '
       '계획 권고: 국내 = 우리가 엑셀로 시군구별 낙농장(젖소) 수(소·돼지·가금 수는 선택), 모집 지역 밖 NaN; 해외 = 데이터 팀에 해외 축사 원천(이름·기준일·수록 나라) 먼저 묻고 나라 전체면 나라별 수, 아니면 NaN. '
       '나라 대표 좌표 한 점 반경 개수·받은 해외 결과의 나라 평균은 안 씀.')
t = t.replace(anchor, add + '\n' + anchor)
open(p, 'w', encoding='utf-8').write(t)
print('ok')
