#!/bin/bash
# 지금 판 3-0b(1등 = 3-0-LSTM) — 옛 3-0b 가 디스크 하한으로 멈춰 이어받기 없이 새로 계산(2026-11-10).
# 중요도는 fold 3개를 동시에(설정 importance_parts), h 마다 중간 저장 — 멈추면 같은 명령으로 이어 감.
# 코드 = code_frozen/s3d_20261110(로컬 코드 그대로: 국내만 모드 분기(기본 꺼짐)·x7-T 제외·중요도 나눠 돌리기 포함).
set -e
B=/home/[계정 가림]/cyj_listeria_final
cd $B/code_frozen/s3d_20261110
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
P=$B/venv/bin/python
A="--root $B/data --inputs $B/results"
$P -I scripts/s3_0b.py $A --out $B/results/s3/3-0b-best --config configs/s3_0b.json
echo S3_0B_DONE
