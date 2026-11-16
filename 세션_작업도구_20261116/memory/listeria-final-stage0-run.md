---
name: listeria-final-stage0-run
description: 최종 실험 단계 0 실행 상태(2026-11-07) — 코드 위치·서버 폴더·결과 경로·MLflow 서버·얼린 코드 사본 방식·7-a 결과·0-9 진행
metadata:
  type: project
---

최종 실험 코드 = `리스테리아ML2\exp\final\`(git 미추적, lst/·scripts/·configs/·tests/ — 441 시험), 서버 사본 `~/cyj_listeria_final/code/`, 결과 `~/cyj_listeria_final/results/s0/<단계>`(0-1·0-3·0-4_v2·0-4_weather_overseas·0-4_spi·0-5·0-6_static·0-6_history·0-6_wxfeat_check·0-6_missing·0-8_groups·0-10·0-10_mlflow·0-9), 로컬 사본 일부 = job tmp `results_mirror\s0`. 실험 로그 `results/logs/experiment_log.jsonl`(7-a 75 = '고른 횟수' 시작).

- 7-a(0-10): 5년·k_u 10·k_p3 10, CV 구간 가중 0.0333(fold 0.0441·0.0456·0.0101), test 는 1-1 보고 때.
- MLflow 서버: `~/cyj_listeria_final/mlflow`(mlflow.db + artifacts), [IP 가림]:5055, pid 파일 `server_5055.pid.json`(lst.tracking.start_server/stop_server — 내가 띄운 것만 끔), mlflow 3.16.1 로컬·서버 같은 판.
- 긴 실행은 얼린 코드 사본에서: `~/cyj_listeria_final/code_frozen/<단계>_<날짜>/`(code/ 에 새 코드를 올려도 도는 단계가 안 바뀜 — runner 는 단계 중 code_md5 바뀌면 멈춤). 0-9 시간 측정은 사용자 결정(11-07 "ㅇㅇ")으로 멈추고 없앰(PR #14 — 예상 끝 날짜 안 함, 부스팅 최대 반복 판정은 알고리즘마다 첫 2차 사슬 부스팅 판 때 fold 3), 멈추기 전 잰 fold 3 학습 1회: 트리 W10 110~320초·W100 810~1,170초, LSTM W10 221·W100 836초(items 는 results/s0/0-9). 0-9 = AutoKeras 확인만(LSTM·GRU·LSTMA·GRUA·TCN·CMB ok, results/s0/0-9_ak_check). 단계 1 = 1-2 는 `code_frozen/s1_20261107`(n_jobs 4, GPU, 로그 run_s1_2.log), 1-1 은 11-07 16:3x 묶음 실행(같은 W·입력 방식 XGB·LGB·CAT 입력 공유) 새 코드 `code_frozen/s1b_20261107`(로그 run_s1_1b.log, TMPDIR = 내 tmp)로 다시 시작 — 옛 코드로 끝난 1-XGB-W010-sum 을 새 코드로 다시 돌려 예측·CV 바이트 같음 확인 → 끝난 5판 인정(실험 로그 code_change_same_results), 결과 results/s1/1-1·1-2, 판마다 h_metrics.csv·내보내기 scripts/export_h_metrics.py.
- 배포: 로컬 tar(--force-local, job tmp) → scp → 서버 tar -x, md5 대조. 서버 nohup 은 `setsid nohup … < /dev/null &`(안 그러면 ssh 가 안 끝남).

- 서버 디스크(공용 96GB)가 빠듯함: 11-07 21:4x 1-1 이 여유 10GB 바닥에 걸려 멈춤 → MLflow 예측 사본(결과 폴더와 바이트 같음 9.1GB) 지움·`log_pred_artifacts` false(예측은 결과 폴더에만, MLflow 엔 지표·h_metrics·설정), 1-2 는 보고(test 채점)·h 별 표 내보내기 뒤 예측 파일을 로컬 `리스테리아ML2\expinal_results\s1-2` 로 옮기고 서버에서 지움(md5 40개 대조, 기록 results/logs/moved_to_local_*.json). 규칙: 단계 보고 뒤 그 단계 예측을 로컬로 옮김 — 다시 채점이 필요하면 로컬에서 올림. 1-1 은 run_s1_1c.log 로 이어 돎.

**Why:** 다음 세션이 같은 서버·결과 위치·실행 방식으로 이어서 단계 1(1-1 CPU·1-2 GPU 나란히, 0-9 뒤)을 돌리게.
**How to apply:** 단계 1 이 끝나면 `s1_3_decide.py`(CV 만) → `--report --reason`(test 채점, 1-1 보고 때 7-a test 도) → 사용자 보고. 관련 [[listeria-1102-final-integrated-plan]], [[gpu-server-usage]].
