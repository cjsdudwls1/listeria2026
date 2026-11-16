---
name: feedback-notepad-not-sticky-notes
description: "'메모장' = 윈도우 메모장(Notepad), 스티커 메모 아님 — 저장 안 된 탭은 TabState\\*.bin(UTF-16LE)에서 읽기 (2026-11-11 지적)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3ab33f0f-8080-4b4d-ba97-a25bff4e960b
  modified: 2026-11-11T07:21:34.086Z
---

사용자가 "메모장 키면 뭐가 보여?"라고 했는데 스티커 메모를 열어서 "왜 스티커 메모를 쳐 여냐"고 지적함 (2026-11-11).

**Why:** 메모장과 스티커 메모는 다른 앱이다. [[sticky-notes-prompt-channel]]은 사용자가 "스티커 메모"라고 말했을 때만 쓴다.

**How to apply:** "메모장"이라고 하면 Notepad를 본다. `Get-Process notepad`로 창 제목을 확인하고, `%LOCALAPPDATA%\Packages\Microsoft.WindowsNotepad_8wekyb3d8bbwe\LocalState\TabState\<guid>.bin` 중 가장 최근에 수정되고 큰 파일을 scratchpad로 복사한다. 그다음 `py -I` + `sys.stdout.reconfigure(encoding="utf-8")`로 UTF-16LE 디코딩해서 읽는다(-I는 PYTHONIOENCODING을 무시함).
