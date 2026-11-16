#!/bin/bash
# 지금 판(국내+해외) 단계 4 ② 60칸 신경망(4-NN) — GPU 줄(스레드 4, 설정 gpu_memory_fraction). 3-0b 판정 뒤 시작.
set -e
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/s3d_20261110
R=$B/results
L=$B/logs
P=$B/venv/bin/python
A="--root $B/data --inputs $R"
export TMPDIR=$B/tmp KERAS_BACKEND=torch OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd $C
until [ -s "$R/logs/chain_state_3-0b.json" ]; do
  if grep -qE "Traceback|RunStopped" $L/run_s3_0b_main.log 2>/dev/null; then echo "3-0b 앞 단계 멈춤 — 기다림 끝냄"; exit 1; fi
  sleep 120
done
$P -I scripts/s4.py $A --out $R/s4/4-NN --config configs/s4_NN.json
echo MAIN_G_4_NN_DONE
echo ALL_DONE
