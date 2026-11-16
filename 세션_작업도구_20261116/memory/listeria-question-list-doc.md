---
name: listeria-question-list-doc
description: "리스테리아 질문 리스트 워드 현재 v0.3(15문항, 타겟 범위 A1~A10·기상 연결 B1~B5) 위치·생성 방식·상태 — 2026-09-15 사용자 검수 전"
metadata:
  node_type: memory
  type: project
  originSessionId: a1448cd6-c4bb-4222-9534-942acfa01019
  modified: 2026-09-15T03:27:54.327Z
---

현재판 **v0.3** (2026-09-15): `리스테리아ML/docs/질문리스트/리스테리아_질문리스트_v0.3.docx` (22쪽, 메인 체크아웃에도 복사), 캡처 `docs/질문리스트/캡처_v0.3/`. 브랜치 worktree-listeria-nb2-questions 커밋 a3d5e55.
- 구성: 첫 쪽 → A 타겟 범위 10건(A1 검사목적 '기타' 82.2% · A2 기준규격외 팽이·공정·환경 검체 · A3 원물 vs 비가열 식육 · A4 학습용 parquet 에서 이미 빠진 '검사기관' 주소 행 · A5 수입 선적일 · A6 검출 이력 없는 품목 · A7 라벨·단위 · A8 연속 검출 · A9 수입 원료 이름 · A10 결과 원문 누수) → B 기상 연결 5건(B1 기온 신호가 시도 보정 뒤 약해짐 · B2 계절 · B3 기준 날짜별 연결 · B4 기관 주소 좌표 · B5 기상 원자료·관측소) → 부록(SQL·방법·임의 규칙) → 답변 기록지. 질문 1건 1쪽 (v0.2 형식 유지).
- 생성: `scripts/make_question_docx_v03.py <노트북> [spec] [out]` 가 4_기상_가공식품_타겟범위.ipynb 14절의 `# [Q-태그]` 셀 표를 헤드리스 Edge 로 캡처하고 `# [Q-NUMS]` JSON 으로 `scripts/question_spec_v03.py` 문장의 {이름}을 채움. 수치를 spec 에 직접 쓰지 않음.
- 레이아웃 교훈: 넓은 표는 워드에서 글자가 작아짐 → 열 머리 줄바꿈(span max-width), 행 이름 짧게, 그림 높이 상한 6.8cm. 첫 쪽 질문 목록이 길면 2쪽으로 넘침.
- v0.1(예시 3건)·v0.2(단순화판)은 비교용으로 남김. 검수 의견 1("구분·표·하위 행이 너무 많아 안 읽힘")은 v0.2 부터 반영.

관련: [[feedback-question-list-format]], [[listeria-nb4-weather-scope-findings]], [[listeria-db-structure-findings]]
