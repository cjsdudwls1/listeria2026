---
name: feedback-kill-only-own-processes
description: 프로세스를 종료할 때는 내가 띄운 것만(PID·시작 시각·내 임시 경로로 확인) — 명령줄 패턴(ipykernel_launcher 등)만으로 종료하면 사용자의 주피터 커널까지 죽는다 (2026-10-06 실수)
metadata:
  node_type: memory
  type: feedback
  originSessionId: 07789c82-b348-4b84-b58d-a5d4f0f8ce29
  modified: 2026-10-06T08:34:38.111Z
---

프로세스를 종료할 때는 내가 이번 세션에서 띄운 프로세스만 종료한다. `ipykernel_launcher`·`python.exe` 같은 명령줄 패턴만으로 골라 `Stop-Process` 하지 않는다.

**Why:** 2026-10-06 데이터셋 조인을 다시 돌리려고 남은 커널을 정리하면서 `CommandLine -match 'ipykernel_launcher|run_nb.py'` 로 종료했더니, 내 커널 1개와 함께 사용자가 아침(09:00)·오후(13:14)에 띄워 둔 주피터 커널 2개가 같이 종료됐다. 사용자가 시키지 않은 종료였고, 열려 있던 노트북의 메모리 상태가 사라진다.

**How to apply:** 백그라운드로 띄운 작업은 TaskStop 으로 멈춘다. 그래도 자식 커널이 남으면 (1) 시작 시각이 내 실행 시각 이후인지, (2) 명령줄의 connection file·작업 폴더가 내 임시 폴더(`.claude/jobs/<id>/tmp`)를 가리키는지, (3) 부모 PID 가 내 실행인지 확인한 PID 만 종료한다. 확인이 안 되면 종료하지 말고 사용자에게 묻는다. 이 PC 에서는 사용자가 VS Code/Jupyter 로 노트북을 열어 두고 일한다([[listeria-nb2-main-checkout-user-edits]]).
