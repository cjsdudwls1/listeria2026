---
name: feedback-no-background-knowledge-labels
description: "배경지식으로 표현·설명을 붙이지 말 것 — DB 코멘트·정의서·실제 값에 적힌 문구만 쓰기 (2026-10-01 지적, CLAUDE.md 데이터 원칙에 기록)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ee1aa279-bc7d-4be9-8c8f-4ad2326e327d
  modified: 2026-10-16T08:52:41.019Z
---

DB 코멘트·테이블정의서·실제 값·폴더 안 문서에 없는 말을 Claude 의 배경지식으로 덧붙여 설명하지 않는다. 출처 문구를 그대로 쓰고, 그 이상은 "설명 없음 — 확인 필요"로 둔다. 값에서 넘겨짚은 해석("외국에서 만든 제품")도 쓰지 않고 확인된 사실만 적는다.

**Why:** 2026-10-01 사용자 지적. `api_data_info.global_factory_weather`(DB 코멘트 "Open-Meteo Archive(ERA5) 해외 제조업소 좌표 일별 기상")를 DB 에 없는 표현 "재분석 격자 값"으로 불렀고, 법정동 코드 10자리 구조를 기억으로 설명했고, `import_nation_code` ≠ KR 행을 "외국에서 만든 제품"이라 단정했다. 교수님·식정원 보고에 들어가면 책임 문제가 된다.

2026-10-16 같은 실수(데이터 밖에서도): Windows 절전 기록의 "절전 모드 전환 이유: Application API"를 "사용자가 직접 절전"이라고 단정했는데, 사용자는 직접 절전한 적이 없다고 했다. 시스템 로그·프로그램 출력도 원문 그대로 옮기고, 누가·왜 했는지는 기록에 없으면 "미확인"으로 둔다.

**How to apply:** 답·문서에 쓰는 설명마다 출처(DB 코멘트 원문, 정의서 시트·문구, 집계 값)를 댈 수 있는지 본다. 댈 수 없으면 쓰지 않거나 "설명 없음 — 확인 필요"로 표시한다. 관련: [[feedback-cite-sources-exactly]], [[feedback-exact-table-names]].
