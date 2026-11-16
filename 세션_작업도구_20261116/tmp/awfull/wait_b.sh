#!/bin/bash
# 대기열 B(pid 4048080)가 끝나거나 ALL_DONE 이 나올 때까지(최대 분) — 끝나면 A·B 로그 꼬리
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"; MAXM="${1:-900}"; S=$(date +%s)
while :; do
  OUT=$(ssh -F "$CFG" gpu "grep -cx 'ALL_DONE' ~/cyj_listeria_final/logs/run_food2_b.log; kill -0 4048080 2>/dev/null && echo RUN || echo END" < /dev/null 2>/dev/null)
  N=$(echo "$OUT" | head -1); ST=$(echo "$OUT" | tail -1)
  if [ "${N:-0}" -ge 1 ] 2>/dev/null; then echo "ALL_DONE"; break; fi
  if [ "$ST" = END ]; then echo "대기열 B 끝남(ALL_DONE 없음)"; break; fi
  if [ $(( ($(date +%s) - S) / 60 )) -ge "$MAXM" ]; then echo "시간 초과 ${MAXM}분"; break; fi
  sleep 120
done
ssh -F "$CFG" gpu "echo '--- A'; tail -n 8 ~/cyj_listeria_final/logs/run_food2_a.log | cut -c1-300; echo '--- B'; tail -n 25 ~/cyj_listeria_final/logs/run_food2_b.log | cut -c1-300; df -h ~ | tail -1" < /dev/null
