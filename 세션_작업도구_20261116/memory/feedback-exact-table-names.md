---
name: feedback-exact-table-names
description: "사용자에게 설명할 때 \"몸통·표\" 같은 별칭 대신 스키마.테이블·컬럼 정확한 이름을 쓴다 (2026-09-22 지적)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: db2ab3f1-2156-4a4e-99b8-c02f701c28d2
  modified: 2026-09-22T02:17:20.924Z
---

DB 객체를 설명할 때 "몸통", "표", "계층표" 같은 내 별칭을 쓰지 말고 `ready_data.hazard_preprocessed_data`, `ba_data.t_prdlst_cl`, `staging_data.prdlst_hierarchy` 처럼 스키마.테이블 이름과 컬럼 이름으로 말한다. 파일은 경로로 말한다(`리스테리아ML/data/listeria_body.parquet`).

**Why:** 2026-09-22 사용자가 "너가 자꾸 몸통, 표 이렇게 말하니까 이해가 안돼 정확한 명칭을 얘기해"라고 지적. 별칭은 대화가 길어지면 무엇을 가리키는지 흐려지고, 보고서에 옮길 때 다시 찾아야 한다.

**How to apply:** 첫 언급에서 "ready_data.hazard_preprocessed_data(위해요소 통합 전처리 결과)"처럼 이름 + 한 줄 뜻을 붙이고, 이후에도 이름을 반복한다. 단계·레벨을 말할 때는 어느 테이블의 어느 컬럼 값인지 함께 적는다. 관련: [[feedback-report-plain-language]], [[listeria-item-code-tables]].
