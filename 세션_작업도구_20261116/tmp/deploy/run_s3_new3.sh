#!/bin/bash
# 2차 새 순서(계획서 v78) — ⑦-T(x7-T) 제외(담당자 결정 2026-11-10) 뒤 다시 시작:
# 3-0 순위·⑥⑦ 판정 → 3-1 → 3-2 → 3-3 → (옛 3-0b 끝난 뒤) 3-0b 1등 → ⑧
# 코드 = code_frozen/s3c_20261110(= s3b_20261109 + scripts/s3_extra.py 의 x7-T 제외 세 곳, lst 는 그대로)
set -e
B=/home/[계정 가림]/cyj_listeria_final
cd $B/code_frozen/s3c_20261110
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
P=$B/venv/bin/python
A="--root $B/data --inputs $B/results"
R=$B/results/s3
$P -I scripts/s3_extra.py $A --out $R/x67-judge --config configs/s3_extra_judge.json
echo X67_DONE
$P -I scripts/s3_1.py $A --out $R/3-1 --config configs/s3_1.json
echo S3_1_DONE
$P -I scripts/s3_2.py $A --out $R/3-2 --config configs/s3_2.json
echo S3_2_DONE
$P -I scripts/s3_3.py $A --out $R/3-3 --config configs/s3_3.json
echo S3_3_DONE
# 옛 순서 3-0b(본 프로세스 2835860, 3-0-LSTM 대상)가 끝날 때까지 기다림 — 이어받기는 옛 작업이 다 끝나야 판정됨
while kill -0 2835860 2>/dev/null; do sleep 120; done
echo OLD_3_0B_EXITED
$P -I $B/logs/rename_old_3_0b_stage.py
$P -I scripts/s3_0b.py $A --out $R/3-0b-best --config configs/s3_0b.json
echo S3_0B_DONE
$P -I scripts/s3_extra.py $A --out $R/x8 --config configs/s3_extra_E.json
echo ALL_DONE
