#!/bin/bash
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"; S=$(date +%s)
while :; do
  OUT=$(ssh -F "$CFG" gpu "grep -cx 'X3B_DONE' ~/cyj_listeria_final/logs/run_food2x.log; ps -eo args | grep -qE '^bash logs/run_food2x.sh' && echo RUN || echo END" < /dev/null 2>/dev/null)
  N=$(echo "$OUT" | head -1); ST=$(echo "$OUT" | tail -1)
  if [ "${N:-0}" -ge 1 ] 2>/dev/null; then echo "X3B_DONE"; break; fi
  if [ "$ST" = END ]; then echo "대기열 끝남(X3B_DONE 없음)"; break; fi
  if [ $(( ($(date +%s) - S) / 60 )) -ge 600 ]; then echo "시간 초과"; break; fi
  sleep 300
done
echo "걸린 시간(분): $(( ($(date +%s) - S) / 60 ))"
ssh -F "$CFG" gpu "grep -E '^=== |_DONE|CV|채택|버림|판정|멈춤|Error|error' ~/cyj_listeria_final/logs/run_food2x.log | tail -n 40 | cut -c1-300" < /dev/null
