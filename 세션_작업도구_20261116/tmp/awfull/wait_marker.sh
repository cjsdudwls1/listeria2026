#!/bin/bash
# 사용: wait_marker.sh <표지 글자> <최대 분> — 서버 run_food2.log 에 표지가 나오거나 대기열 프로세스가 끝나거나 시간이 지나면 끝
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"; M="$1"; MAXM="${2:-600}"; S=$(date +%s)
while :; do
  OUT=$(ssh -F "$CFG" gpu "grep -c '$M' ~/cyj_listeria_final/logs/run_food2.log; pgrep -f 'run_food2/run_food2.sh' >/dev/null && echo RUN || echo END" < /dev/null 2>/dev/null)
  N=$(echo "$OUT" | head -1); ST=$(echo "$OUT" | tail -1)
  if [ "${N:-0}" -ge 1 ] 2>/dev/null; then echo "표지 $M 나옴"; break; fi
  if [ "$ST" = END ]; then echo "대기열 끝남(표지 $M 없음)"; break; fi
  if [ $(( ($(date +%s) - S) / 60 )) -ge "$MAXM" ]; then echo "시간 초과 ${MAXM}분"; break; fi
  sleep 60
done
ssh -F "$CFG" gpu "tail -n 25 ~/cyj_listeria_final/logs/run_food2.log | cut -c1-300; df -h ~ | tail -1" < /dev/null
