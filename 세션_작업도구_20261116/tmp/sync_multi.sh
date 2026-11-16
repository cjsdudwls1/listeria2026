#!/bin/bash
# 사용: sync_multi.sh "<단계 상대경로들 공백 구분>" "<끝 표시 로그들 공백 구분>"
# 끝난 판(done.json 있고 .synced_local 없음, '_' 로 시작하는 폴더 제외)을 로컬로 받고 pred_*.parquet md5 대조 뒤 서버에서 지움.
STS="$1"; LOGS="$2"
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"
while true; do
  for ST in $STS; do
    LROOT="C:/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final_results/$ST"; mkdir -p "$LROOT"
    LIST=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$ST 2>/dev/null && for d in */; do d=\${d%/}; case \$d in _*) continue;; esac; [ -f \$d/done.json ] && [ ! -f \$d/.synced_local ] && echo \$d; done" < /dev/null)
    for P in $LIST; do
      SM=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$ST/$P && md5sum pred_*.parquet 2>/dev/null | sort -k2" < /dev/null)
      scp -q -F "$CFG" -r "gpu:cyj_listeria_final/results/$ST/$P" "$LROOT/" || continue
      if [ -n "$SM" ]; then
        LM=$(cd "$LROOT/$P" && md5sum pred_*.parquet | sed 's/ \*/  /' | sort -k2)
        [ "$(echo "$SM" | awk '{print $1,$2}')" = "$(echo "$LM" | awk '{print $1,$2}')" ] || { echo "$(date +%T) $ST/$P md5 다름 — 안 지움"; continue; }
      fi
      ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$ST/$P && rm -f pred_*.parquet && date -u +%FT%TZ > .synced_local && echo '{\"stage\": \"$ST\", \"variant\": \"$P\", \"local\": \"리스테리아ML2/exp/final_results/$ST/$P\", \"utc\": \"'\$(date -u +%FT%TZ)'\", \"md5_checked\": true}' >> ~/cyj_listeria_final/results/logs/moved_to_local.jsonl" < /dev/null
      echo "$(date +%T) $ST/$P 옮김"
    done
  done
  ALLDONE=1
  for L in $LOGS; do ssh -F "$CFG" gpu "grep -qE 'ALL_DONE|Traceback|RunStopped' ~/cyj_listeria_final/logs/$L" < /dev/null || ALLDONE=0; done
  if [ $ALLDONE = 1 ]; then
    REM=0
    for ST in $STS; do R=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/results/$ST 2>/dev/null && for d in */; do d=\${d%/}; case \$d in _*) continue;; esac; [ -f \$d/done.json ] && [ ! -f \$d/.synced_local ] && echo \$d; done | wc -l" < /dev/null); REM=$((REM + R)); done
    [ "$REM" = "0" ] && { echo "끝: $STS"; ssh -F "$CFG" gpu "df -h ~ | tail -1" < /dev/null; exit 0; }
  fi
  sleep 600
done
