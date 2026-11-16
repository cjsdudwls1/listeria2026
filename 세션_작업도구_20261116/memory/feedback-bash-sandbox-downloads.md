---
name: feedback-bash-sandbox-downloads
description: "Bash 도구로 Downloads 에 cp 하면 Bash 안에서는 보이지만 실제 디스크엔 안 생김(샌드박스) — Downloads 등 작업 폴더 밖으로 내보낼 땐 PowerShell Copy-Item 후 PowerShell 로 확인 (2026-10-27 \"없는데?\")"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5652bce3-5487-428a-a2f6-80963c63ee92
  modified: 2026-10-27T10:24:56.658Z
---

2026-10-27 교수님 보고서 v3 를 Bash `cp` 로 `C:\Users\NT751\Downloads` 에 넣고 Bash `ls`·md5 로 확인했는데, 사용자 탐색기에는 없었음("없는데?"). PowerShell `Get-ChildItem` 으로도 없음 → Bash 샌드박스 쓰기가 실제 디스크에 안 닿음. 같은 때 jobs 임시 폴더·리스테리아ML2 안 쓰기는 실제로 생김.

**Why:** 사용자에게 "저장했다"고 했는데 파일이 없어 신뢰가 깎임.
**How to apply:** Downloads·Documents 등 작업 폴더 밖에 결과물을 둘 때는 PowerShell `Copy-Item -LiteralPath` 로 복사하고, 완료 보고 전 PowerShell 로 존재·해시를 확인. 관련 [[feedback-midcheck-report-rules]].
