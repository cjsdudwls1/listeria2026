#!/usr/bin/env bash
# 04_실행환경_실행방법/실행방법.md §2 그대로(venv 위치만 패키지 밖 $R/venv)
set -u
source "$HOME/cyj_listeria_repro_1113/my_scripts/env.sh"
cd "$R"
echo "[$(date '+%F %T')] venv 만들기"
python3.12 -m venv venv && source venv/bin/activate
python --version; pip --version | cut -d' ' -f1-2
echo "[$(date '+%F %T')] pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126"
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
rc1=$?
echo "[$(date '+%F %T')] 종료 코드 $rc1"
echo "[$(date '+%F %T')] pip install -r 04_실행환경_실행방법/requirements.txt"
pip install -r "$PKG/04_실행환경_실행방법/requirements.txt"
rc2=$?
echo "[$(date '+%F %T')] 종료 코드 $rc2"
pip freeze > "$R/logs/pip_freeze_repro.txt"
echo "[$(date '+%F %T')] 끝 rc1=$rc1 rc2=$rc2"
