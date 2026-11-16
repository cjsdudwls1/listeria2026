---
name: listeria-nb2-main-checkout-user-edits
description: 사용자는 메인 체크아웃의 리스테리아ML/notebook/2.ipynb 를 Jupyter 에서 직접 고치고 실행함 — 덮어쓰기 전에 커밋본과 셀 단위 비교 필수
metadata: 
  node_type: memory
  type: project
  originSessionId: a1448cd6-c4bb-4222-9534-942acfa01019
  modified: 2026-09-14T07:41:17.933Z
---

2026-09-14 워크트리에서 만든 2.ipynb 를 메인 체크아웃(`C:\Users\NT751\Desktop\천영진\리스테리아ML\notebook\2.ipynb`)에 복사해 두었더니, 사용자가 Jupyter 에서 다시 실행하고 '0. 한눈에 보는 결론'의 핵심 10가지 순서를 4·5·7·9·1·2·3·6·8·10 으로 바꿔 저장했다 (크기·mtime·줄바꿈 LF 로 알아챔).

**Why:** 빌드본으로 그냥 덮으면 사용자의 손 수정이 사라진다. 사용자는 노트북을 읽으면서 직접 편집한다.

**How to apply:** 메인 체크아웃 노트북을 갱신하기 전 (1) 마지막 커밋본과 셀 소스를 difflib 로 비교해 사용자 수정을 찾고, (2) 그 수정을 빌드 원본(파트 파일·결론 요약)에 반영한 뒤, (3) 크기·mtime 이 확인한 그대로일 때만 복사한다. 출력(실행 결과)만 바뀐 경우는 덮어도 된다. 관련: [[listeria-db-structure-findings]]
