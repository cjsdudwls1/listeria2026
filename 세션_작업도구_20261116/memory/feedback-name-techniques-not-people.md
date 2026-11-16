---
name: feedback-name-techniques-not-people
description: "보고서·패키지·표에 '손 교수 방식' 같은 사람 이름·내부 번호로 된 방법 이름을 쓰지 말고 기법 자체로 쓸 것(연도별 무작위 언더샘플링 등) (2026-11-13 지적)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: a19471fc-8fc4-4365-8d63-c14cd9e46718
  modified: 2026-11-13T08:05:07.907Z
---

'손 교수 방식'이라는 말은 쓰지 않는다 — 기법을 말한다: '연도별 무작위 언더샘플링(해마다 양성은 모두 두고, 그해 음성을 양성 수의 4배만큼 무작위로 뽑음)'. 같은 원칙으로 '6번 방식' → '검출 칸 음성 샘플링(그해 검출이 나온 품목·지역 칸의 음성에서만 뽑음)', 'Theil's U 다운사이징' 처럼 무엇을 하는지가 드러나는 이름.

**Why:** 2026-11-13 교수님 보고서 검토 중 사용자: "손교수방식 이란말도 쓰면안되고 그 기법을 말해야지". 보고서는 지시자가 아니라 기법·논리로 읽혀야 함(같은 날 "지시받아서 했다가 아니라 이런 논리로…" 지시와 같은 맥락).

**How to apply:** 보고서·제출 패키지 문서·결과 엑셀의 단계 이름·대화 설명 모두. 코드 식별자(hon5 등)는 바꾸지 않고 주석·문서만. 관련 [[feedback-use-stage-labels-not-run-ids]], [[feedback-report-reader-first]], [[feedback-no-professor-attribution]].
