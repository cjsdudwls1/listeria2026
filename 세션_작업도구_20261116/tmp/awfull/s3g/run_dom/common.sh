#!/bin/bash
# 국내만 버전 대기열 공통(run_dom_cpu.sh·run_dom_gpu.sh 가 source) — 지금 판 대기열(logs/run_s3_new.sh)과 같은 꼴.
# 지금 판과 나누는 것: 결과 뿌리 results_dom, 로그 logs_dom, 얼린 코드 code_frozen/dom_20261110, 설정 configs_dom,
# MLflow experiment 이름 '-국내'(같은 서버 [IP 가림]:5055). ~/cyj_listeria_final/results 에는 쓰지 않음
# (lst.scope: 국내판 설정은 결과 뿌리 표지 SCOPE.json 이 있는 뿌리에서만 돌고, --out 은 그 뿌리 안이어야 함).
set -e
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/dom_20261110
RD=$B/results_dom
L=$B/logs_dom
P=$B/venv/bin/python
CFG=configs_dom
A="--root $B/data --inputs $RD"
export TMPDIR=$B/tmp KERAS_BACKEND=torch
cd $C

# 결정 파일(국내판 뿌리 아래)이 생길 때까지 기다림 — $2(로그)에 멈춤 문구가 보이면 끝냄
wait_file() {
  until [ -s "$RD/$1" ]; do
    if [ -n "$2" ] && grep -qE "Traceback|RunStopped|ScopeError|PrepStop" "$2"; then echo "앞 단계 멈춤 — $1 기다림 끝냄"; exit 1; fi
    sleep 120
  done
}
# 다른 줄 로그의 표시 줄을 기다림(지금 판 run_s3_cpu2.sh 와 같은 꼴)
wait_mark() {
  until grep -q "^$1\$" "$2" 2>/dev/null; do
    if grep -qE "Traceback|RunStopped|ScopeError|PrepStop" "$2" 2>/dev/null; then echo "$2 멈춤 — $1 기다림 끝냄"; exit 1; fi
    sleep 120
  done
}
# 이어 돌리기: 그 단계 산출 파일($1, 국내판 뿌리 아래)이 이미 있으면 건너뜀(단계 0·판정 스크립트는 '이미 있으면 멈춤'
# 이라서 — 판 학습 스크립트는 끝난 판을 스스로 건너뛰므로 그냥 다시 부름)
run_unless() {
  local f=$1; shift
  if [ -s "$RD/$f" ]; then echo "건너뜀(있음): $f"; else "$@"; fi
}
# 앞 단계가 끝났는데 결정 파일이 없으면 멈춤(다음 단계는 결정 파일이 생긴 뒤에만)
need() {
  for f in "$@"; do
    if [ ! -s "$RD/$f" ]; then echo "결정 파일 없음 — 멈춤: $f"; exit 1; fi
  done
}
