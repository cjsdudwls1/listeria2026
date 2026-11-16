#!/bin/bash
# 사용: sync_after.sh <로그 이름> <기다릴 표시> <단계/판 ...>
# 로그에 표시가 나오면(오류 표시가 먼저 나오면 아무것도 안 하고 끝) 판마다 로컬로 받고 pred_*.parquet md5 대조 뒤 서버에서 지움.
LOG="$1"; MARK="$2"; shift 2
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"
LBASE="C:/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final_results"
until ssh -F "$CFG" gpu "grep -q '$MARK' ~/cyj_listeria_final/logs/$LOG" < /dev/null; do
  if ssh -F "$CFG" gpu "grep -qE 'Traceback|RunStopped|TrainingHalted' ~/cyj_listeria_final/logs/$LOG" < /dev/null; then
    echo "$(date +%T) 대기열 오류 — 옮기지 않음"; exit 1
  fi
  sleep 600
done
for SP in "$@"; do
  ST=$(dirname "$SP"); P=$(basename "$SP")
  SM=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$SP && [ -f done.json ] && [ ! -f .synced_local ] && md5sum pred_*.parquet 2>/dev/null | sort -k2" < /dev/null)
  [ -z "$SM" ] && { echo "$P 건너뜀(끝 표시 없음·이미 옮김·예측 없음)"; continue; }
  mkdir -p "$LBASE/$ST"
  scp -q -F "$CFG" -r "gpu:cyj_listeria_final/results/$SP" "$LBASE/$ST/" || { echo "$P 받기 실패"; continue; }
  LM=$(cd "$LBASE/$SP" && md5sum pred_*.parquet | sed 's/ \*/  /' | sort -k2)
  if [ "$(echo "$SM" | awk '{print $1,$2}')" = "$(echo "$LM" | awk '{print $1,$2}')" ]; then
    ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$SP && rm -f pred_*.parquet && date -u +%FT%TZ > .synced_local && echo '{\"stage\": \"$ST\", \"variant\": \"$P\", \"local\": \"리스테리아ML2/exp/final_results/$SP\", \"utc\": \"'\$(date -u +%FT%TZ)'\", \"md5_checked\": true}' >> ~/cyj_listeria_final/results/logs/moved_to_local.jsonl" < /dev/null
    echo "$(date +%T) $P 옮김"
  else
    echo "$(date +%T) $P md5 다름 — 안 지움"
  fi
done
ssh -F "$CFG" gpu "df -h ~ | tail -1" < /dev/null
