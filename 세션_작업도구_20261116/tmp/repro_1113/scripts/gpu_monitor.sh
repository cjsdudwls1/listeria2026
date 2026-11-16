#!/usr/bin/env bash
# GPU 메모리 기록(60초마다, 최대 3시간) — 내 작업 동안 40% 이하인지 확인용
end=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "$(date '+%F %T') $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tr -d '\n') | apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  sleep 60
done
