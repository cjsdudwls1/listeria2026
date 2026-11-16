#!/bin/bash
# 1묶음 다시: ⑥⑦(새 순서 사본)·3-1·3-3·⑧ — 3-0 트리 판 예측을 잠깐 올린 뒤.
B=/home/[계정 가림]/cyj_listeria_final
P=$B/venv/bin/python
A="--root $B/data --inputs $B/results"
R=$B/results
WHY="단계별 test 상승·하락 보고(사용자 요청 2026-11-11) — 고르기 모두 끝난 뒤(8-D 는 CV 로만 고름)"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
rep() {
  echo "== $3"
  ( cd $B/code_frozen/$1 && $P -I scripts/$2 $A --out $R/$3 --config configs/$4 --report --reason "$WHY" ) && echo "OK $3" || echo "FAIL $3"
}
rep s3c_20261110 s3_extra.py s3/x678-G s3_extra_G.json
rep s3c_20261110 s3_1.py s3/3-1 s3_1.json
rep s3c_20261110 s3_3.py s3/3-3 s3_3.json
rep s3c_20261110 s3_extra.py s3/x8 s3_extra_E.json
echo REPORTS_B1B_DONE
