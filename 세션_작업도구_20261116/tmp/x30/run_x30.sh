#!/bin/bash
# x30(2026-11-13 사용자 요청): '2차: 13구조 1위'(3-0-LSTM, 전체 자료) 다시 학습 — ① 식품 고정표 v2 ② 식품 v2 + KO1. 둘 나란히(GPU 0.4씩),
# 끝나면 test 채점. 새 폴더 s3/3-0-D-food2·3-0-D-food2KO, MLflow 끔. 코드 = food2c 사본(s3_0.py add_ids 더함).
set -uo pipefail
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/food2c_20261113
R=$B/results
L=$B/logs
P=$B/venv/bin/python
A="--root $B/data --inputs $R"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
cd $C
step() { echo "=== $(date '+%F %T') $1"; }

step "0 점검"
for d in s3/3-0-D-food2 s3/3-0-D-food2KO; do test ! -e "$R/$d" || { echo "이미 있음(덮지 않음): $R/$d"; exit 1; }; done
test -s "$R/s0/0-6_static_v2/식품_고정표.parquet" || { echo "식품 v2 없음"; exit 1; }

step "1 학습 둘 나란히(① 식품 v2 · ② 식품 v2 + KO1)"
$P -I scripts/s3_0.py $A --out $R/s3/3-0-D-food2 --config $L/s3_0_D_food2x.json --only 3-0-LSTM > $L/run_x30_food2.log 2>&1 &
P1=$!
sleep 20
$P -I scripts/s3_0.py $A --out $R/s3/3-0-D-food2KO --config $L/s3_0_D_food2ko.json --only 3-0-LSTM-KO1 > $L/run_x30_ko.log 2>&1 &
P2=$!
wait $P1; RC1=$?
wait $P2; RC2=$?
echo "학습 끝: ① rc=$RC1 · ② rc=$RC2"
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ] || { echo "학습 멈춤 — 로그 run_x30_food2.log·run_x30_ko.log 확인"; exit 1; }
echo TRAIN_DONE

step "2 test 채점"
set -e
$P -I scripts/s3_0.py $A --out $R/s3/3-0-D-food2 --config $L/s3_0_D_food2x.json --only 3-0-LSTM \
  --report --reason "'2차: 13구조 1위' 식품 v2 다시 학습(사용자 요청 2026-11-13)"
$P -I scripts/s3_0.py $A --out $R/s3/3-0-D-food2KO --config $L/s3_0_D_food2ko.json --only 3-0-LSTM-KO1 \
  --report --reason "'2차: 13구조 1위' 식품 v2 + KO 다시 학습(사용자 요청 2026-11-13)"
echo X30_DONE
