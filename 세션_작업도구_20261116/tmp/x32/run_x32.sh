#!/bin/bash
# x32(2026-11-13 사용자 요청): '결합 스태킹'(3-2-CAT = CatBoost + VAE 이상 점수) + 손 교수 방식(5) + 식품 고정표 v2 + KO 6칼럼.
# val·test PR-AUC 만(최종 중요도·최종 보고 안 함). 코드 = food2b 사본 그대로(새 코드 없음), 설정·준비 스크립트는 logs/ 에.
set -euo pipefail
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/food2b_20261112
R=$B/results
P=$B/venv/bin/python
A="--root $B/data --inputs $R"
CFG=$B/logs/s5_6_T_x32_ko.json
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 TMPDIR=$B/tmp KERAS_BACKEND=torch
cd $C
step() { echo "=== $(date '+%F %T') $1"; }

step "0 점검"
for d in s5_6/5-6-T-x32KO s5_6/x32_spec logs/chain_state_x32.json; do
  test ! -e "$R/$d" || { echo "이미 있음(덮지 않음): $R/$d"; exit 1; }
done
for f in s0/0-6_static_v2/식품_고정표.parquet s5_6/sel-hon5/kept_keys_refit.parquet s5_6/5-6-judge/5-6_decision.json \
         s3/3-2/3-2-CAT/spec.json s3/3-2/3-2-CAT/pred_refit.parquet s3/3-1/3-1-VAEt/spec.json logs/chain_state_4.json; do
  test -s "$R/$f" || { echo "없음: $R/$f"; exit 1; }
done
$P -I -c "import urllib.request; urllib.request.urlopen('http://[IP 가림]:5055/health', timeout=10)"

step "1 준비(사슬 상태 사본·5-CAT 판 설정)"
$P -I $B/logs/x32_prep.py $C $R $CFG
echo PREP_DONE

step "2 학습(5-3-2-CAT-KO — VAE 이상 점수 다시 학습 + CatBoost, 검증·test 예측)"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-x32KO --config $CFG
echo TRAIN_DONE

step "3 test 채점(보고 표)"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-x32KO --config $CFG \
  --report --reason "추가 비교 판: 결합 스태킹 + 손 교수 방식 + 식품 갱신 + KO(사용자 요청 2026-11-13)"
echo X32_DONE
