---
name: feedback-keep-server-disk-light
description: 공용 GPU 서버 디스크는 다른 사람도 쓰므로 가볍게 — 끝난 결과는 바로 로컬로 받고(md5 대조) 서버에서 지움 (2026-11-07)
metadata:
  node_type: memory
  type: feedback
  originSessionId: a19471fc-8fc4-4365-8d63-c14cd9e46718
  modified: 2026-11-10T21:20:49.255Z
---

서버 디스크는 다른 사람도 써야 하니 최대한 바로바로 지운다. 필요한 것은 그때그때 로컬로 받아 두고 서버 디스크는 가볍게 유지 — 사용자 "디스크는 다른사람도 써야하니까 최대한 바로바로 지워. 필요한건 로컬로 그때그때 받아놓고 디스크는 가볍게 유지 ㄱㄱ" (11-07, 1-1 이 디스크 바닥 10GB 로 멈춘 뒤).

**Why:** 공용 서버(96GB)에서 우리 예측 파일이 쌓여 바닥에 닿았고 다른 사용자(파튤린 담당)도 같은 디스크를 씀.
**How to apply:** 판이 끝나는 대로 판 폴더를 로컬 `리스테리아ML2\exp\final_results\…` 로 받고 큰 파일(예측 parquet 등) md5 대조 뒤 서버에서 지움(작은 상태 파일 done.json·cv_scores.json·spec·h_metrics 는 이어하기용으로 남김), 기록은 서버 results/logs/moved_to_local_*.jsonl. 보고·다시 채점 때 필요한 예측만 잠깐 다시 올리고 끝나면 지움. MLflow 엔 예측 사본을 두지 않음. 관련 [[listeria-final-stage0-run]].
**11-10 사고**: 국내판 병행으로 디스크가 하한 10GB 에 닿아 국내판 3-0 과 지금 판 옛 3-0b 중요도(24시간째, 중간 저장 없음)가 함께 멈춰 계산을 통째로 잃음. 이후: 판정이 끝난 단계의 예측은 다음 단계 시작 전에 바로 옮기고(scripts: job tmp `sync_now.sh <results|results_dom> <단계 폴더…>`), 여유 14GB 아래면 알림 받는 감시를 둠, 긴 단일 계산에는 중간 저장(이어 하기)을 넣음. **값 모델 판(3-1-VAEt)의 pred_* 는 이상 점수 칼럼 입력이라 현재 최고 판에 이상 점수가 있으면 옮기지 말 것**(11-11 국내판에서 잘못 옮겼다가 3분 만에 md5 대조로 되돌림) — 옮기기 전 현재 최고 판·트리 대상 판·그 판이 읽는 판을 EXCLUDE 로 남김.
