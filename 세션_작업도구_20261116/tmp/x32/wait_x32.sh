#!/bin/bash
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"; S=$(date +%s)
while :; do
  OUT=$(ssh -F "$CFG" gpu "grep -cx 'X32_DONE' ~/cyj_listeria_final/logs/run_x32.log; ps -eo args | grep -qE '^bash logs/run_x32.sh' && echo RUN || echo END" < /dev/null 2>/dev/null)
  N=$(echo "$OUT" | head -1); ST=$(echo "$OUT" | tail -1)
  if [ "${N:-0}" -ge 1 ] 2>/dev/null; then echo "X32_DONE"; break; fi
  if [ "$ST" = END ]; then echo "대기열 끝남(X32_DONE 없음)"; break; fi
  if [ $(( ($(date +%s) - S) / 60 )) -ge 240 ]; then echo "시간 초과"; break; fi
  sleep 90
done
echo "걸린 시간(분): $(( ($(date +%s) - S) / 60 ))"
ssh -F "$CFG" gpu "tail -n 30 ~/cyj_listeria_final/logs/run_x32.log | cut -c1-300" < /dev/null
