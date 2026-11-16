---
name: listeria-global-country-tables-join
description: "2026-09-17 해외 표 4개(global_weather_info·global_soil_ph·global_fine_dust_info·global_factory_fine_dust) 수입 시료 조인 연결률·탈락 원인·나라 대표점 거리, 질문개개정판 1번 삽입 대기"
metadata: 
  node_type: memory
  type: project
  originSessionId: d4e71e82-3176-414c-86cf-127d35e207ad
  modified: 2026-09-17T07:24:10.520Z
---

2026-09-17 사용자 요청: 시료 파일 수입 47,642행(검출 125)에 해외 표 4개를 붙인 완성도를 `공유용/질문/질문개개정판.md` 2번째 줄(1번 국내/해외 범위) 뒤에 추가. DB 불통이라 로컬 추출본 + 09-14 DB 조회 출력(대화 기록) 사용.
- 규칙: 나라 = prdctn → orgplce → import_nation_code 첫 값 → ba_data_info.nation_code alpha-3 (노트북 3과 같음), 날짜 = 시료 파일 mnfctur_de.
- global_weather_info 45,536(95.6%, 검출 113) / global_soil_ph 45,531(몰디브 4·마셜 1 pH 빈 값). 탈락 = 코드 3종 없음 1,964 + orgplce 칸 글자값('KOREA' 67·'U.S.A' 64 등) 142; 날짜 탈락 0. 1,964 중 1,895는 s_org mnf_nation_cd 있음.
- 날씨 표 나라 지점은 수도 부근(미국 38.91,-77.07), 시료 좌표까지 중앙값 431km·>500km 41.5%. global_fine_dust_info 지점은 정수 좌표(미국 38,-97)로 다름.
- global_fine_dust_info: 1,047,045행·249개국·2015-01-01~2026-07-06 전부, NULL 0. global_factory_fine_dust: 좌표 54,346(=global_factory_weather 좌표), 수입 좌표 3,072 모두 있음, 당일 전수 미확인.
- 결과: 워크트리 브랜치 worktree-listeria-global-join-coverage 의 `리스테리아ML/scripts/share/global_join_coverage.py`·`results/`(삽입 문안·작업기록 초안). 워크트리 격리로 메인 체크아웃 문서 편집이 막혀 사용자 확인 대기.

**Why:** 국내/해외 범위 결정과 해외 기상 연결 방식(나라 한 점 vs 제조소 좌표) 질문의 근거.
**How to apply:** 해외 연결 설계 시 이 수치 사용, DB 복구 후 미세먼지 표 전수 확인. 관련 [[listeria-overseas-weather-lifecycle-brief]], [[listeria-share-dataset]], [[feedback-question-list-format]].

- **11-04 밤 9 다시 셈**(최종 계획 규칙 10-01, 현재 시료 파일 `리스테리아ML/data/listeria_data.parquet` 해외 47,642행, 읽기 전용 SELECT: `ready_data.hazard_preprocessed_data` ⋈ `ba_data.s_org_lims_data_info`(`cntc_sys_key2 = rcept_no`, `label`='리스테리아') → `ba_data_info.nation_code` alpha_2→alpha_3): 코드로 정한 47,431행 = 95개국(`mnf_nation_cd` 47,421·`import_nation_code` 10, 검출 125), 못 정한 211행 = 좌표·주소로 미국·중국·태국·이탈리아·뉴질랜드·덴마크·스페인·한국 — 새 나라 없음. 행 많은 순 미국 8,641·이탈리아 7,619·프랑스 6,809·중국 2,430·스페인 2,257, 검출 이탈리아 31·스페인 18·중국 12·대만 12·일본 11. 한국(KOR) 123행은 제조국 대한민국·주소 일본·중국(결정 13-c). 데이터 팀(해외 축사)엔 한국 뺀 94개국 전달.
