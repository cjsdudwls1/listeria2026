#!/bin/bash
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"; S=$(date +%s)
while :; do
  OUT=$(ssh -F "$CFG" gpu "grep -cx 'X30_DONE' ~/cyj_listeria_final/logs/run_x30.log; ps -eo args | grep -qE '^bash logs/run_x30.sh' && echo RUN || echo END" < /dev/null 2>/dev/null)
  N=$(echo "$OUT" | head -1); ST=$(echo "$OUT" | tail -1)
  if [ "${N:-0}" -ge 1 ] 2>/dev/null; then echo "X30_DONE"; break; fi
  if [ "$ST" = END ]; then echo "대기열 끝남(X30_DONE 없음)"; break; fi
  if [ $(( ($(date +%s) - S) / 60 )) -ge 300 ]; then echo "시간 초과"; break; fi
  sleep 120
done
echo "걸린 시간(분): $(( ($(date +%s) - S) / 60 ))"
ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/logs; cut -c1-250 run_x30.log | tail -12; echo '--- ①'; grep -E 'CV|Error|error|멈춤' run_x30_food2.log | tail -5 | cut -c1-250; echo '--- ②'; grep -E 'CV|Error|error|멈춤' run_x30_ko.log | tail -5 | cut -c1-250" < /dev/null
