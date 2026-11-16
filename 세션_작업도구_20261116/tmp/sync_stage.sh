#!/bin/bash
# 사용: sync_stage.sh <단계 상대경로 예: s1/1-1> <끝 표시 로그 이름 예: run_s1_1c.log>
# 끝난 판(done.json 있고 .synced_local 없음)을 로컬로 받고 pred_*.parquet md5 대조 뒤 서버에서 지움.
ST="$1"; LOG="$2"
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"
LROOT="C:/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final_results/$ST"
mkdir -p "$LROOT"
while true; do
  LIST=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$ST && for d in 1-*; do [ -f \$d/done.json ] && [ ! -f \$d/.synced_local ] && echo \$d; done" < /dev/null)
  for P in $LIST; do
    SM=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$ST/$P && md5sum pred_*.parquet 2>/dev/null | sort -k2" < /dev/null)
    [ -z "$SM" ] && continue
    scp -q -F "$CFG" -r "gpu:cyj_listeria_final/results/$ST/$P" "$LROOT/" || continue
    LM=$(cd "$LROOT/$P" && md5sum pred_*.parquet | sed 's/ \*/  /' | sort -k2)
    if [ "$(echo "$SM" | awk '{print $1,$2}')" = "$(echo "$LM" | awk '{print $1,$2}')" ]; then
      ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$ST/$P && rm -f pred_*.parquet && date -u +%FT%TZ > .synced_local && echo '{\"stage\": \"$ST\", \"variant\": \"$P\", \"local\": \"리스테리아ML2/exp/final_results/$ST/$P\", \"utc\": \"'\$(date -u +%FT%TZ)'\", \"md5_checked\": true}' >> ~/cyj_listeria_final/results/logs/moved_to_local.jsonl" < /dev/null
      echo "$(date +%T) $P 옮김"
    else
      echo "$(date +%T) $P md5 다름 — 안 지움"
    fi
  done
  if ssh -F "$CFG" gpu "grep -qE 'ALL_DONE|Traceback|RunStopped' ~/cyj_listeria_final/logs/$LOG" < /dev/null; then
    REM=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$ST && for d in 1-*; do [ -f \$d/done.json ] && [ ! -f \$d/.synced_local ] && echo \$d; done | wc -l" < /dev/null)
    [ "$REM" = "0" ] && { echo "끝: $ST"; ssh -F "$CFG" gpu "df -h ~ | tail -1" < /dev/null; exit 0; }
  fi
  sleep 600
done
