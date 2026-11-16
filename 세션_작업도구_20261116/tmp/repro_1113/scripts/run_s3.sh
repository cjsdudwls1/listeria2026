#!/usr/bin/env bash
# 재학습: 04_실행환경_실행방법/실행방법.md §4 그대로 — bash run_pipeline.sh s3 (venv 켠 상태, python = venv python)
set -u
source "$HOME/cyj_listeria_repro_1113/my_scripts/env.sh"
source "$R/venv/bin/activate"
cd "$PKG/01_소스코드"
echo "[$(date '+%F %T')] 시작: bash run_pipeline.sh s3 (python = $(command -v python | sed "s#$HOME#~#"))"
t0=$(date +%s)
bash run_pipeline.sh s3
rc=$?
echo "[$(date '+%F %T')] 끝 rc=$rc · 걸린 시간 $(( $(date +%s) - t0 ))초"
