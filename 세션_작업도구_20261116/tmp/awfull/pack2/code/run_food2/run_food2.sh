#!/bin/bash
# 식품 고정표 v2 다시 학습 대기열(사용자 결정 2026-11-12) — 한 줄로 차례대로, 멈추면 거기서 끝(set -e).
#   1 식품 고정표 v2(aw 출처 거르기 없앰) + 결측 표시 목록 대조(같으면 기존 0-6_missing 그대로 — 다르면 여기서 멈춤)
#   2 5-LSTM-food2 학습(5-LSTM 과 같은 설정·같은 연도별 뽑기 s5_6/sel-hon5·씨앗 0, 단계 run '5-6-food2')
#   3 5-LSTM-KO 학습(2 + Koutsoumanis 생육확률 6칼럼 — 비교만, 단계 run '5-6-food2-KO')
#   4 두 판 test 보고(단계 표 — KO 표에 5-LSTM·5-LSTM-food2 줄 나란히)  5 KO 6칼럼 시료 기준 분포
#   6 최종 고르기(5-LSTM-food2 를 사슬 끝 5-LSTM 자리에) → 7 최종 중요도(6부분 동시, speed 켬) → 8 최종 보고(final_food2)
# 기존 결과(s0/0-6_static·0-6_missing·s5_6/5-6-T·final)는 읽기만 하고 바꾸지 않음.
#
# 띄우기(지휘자 — 코드 사본·자료를 올린 뒤):
#   cd ~/cyj_listeria_final && setsid nohup nice -n 10 bash code_frozen/food2_20261112/run_food2/run_food2.sh \
#       > logs/run_food2.log 2>&1 < /dev/null &
set -euo pipefail
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/food2_20261112
R=$B/results
P=$B/venv/bin/python
A="--root $B/data --inputs $R"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
cd $C
step() { echo "=== $(date '+%F %T') $1"; }

step "0 점검: 새 추출 파일 md5·MLflow 서버·기존 결과"
AW=$B/data/리스테리아ML2/data/추출_20261112/t_aw_cl.parquet
echo "f45e7468aa70afda70cb8634338aacee  $AW" | md5sum -c -
$P -I -c "import urllib.request; urllib.request.urlopen('http://[IP 가림]:5055/health', timeout=10)"
for f in s0/0-6_static/식품_고정표.parquet s0/0-6_missing/missing_lists.json s5_6/5-6-T/5-LSTM/done.json \
         s5_6/sel-hon5/kept_keys_refit.parquet s5_6/5-6-judge/5-6_decision.json logs/chain_state_8.json; do
  test -s "$R/$f" || { echo "없음: $R/$f"; exit 1; }
done
for d in s0/0-6_static_v2 s5_6/5-6-T-food2 s5_6/5-6-T-food2-KO s5_6/ko_dist final_food2; do
  test ! -e "$R/$d" || { echo "이미 있음(덮지 않음): $R/$d"; exit 1; }
done

step "1 식품 고정표 v2"
$P -I scripts/s0_6_static.py $A --out $R/s0/0-6_static_v2 --config configs/s0_6_static_food2.json
echo STATIC_V2_DONE

step "2 5-LSTM-food2 학습"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-food2 --config configs/s5_6_T_food2.json
echo TRAIN_FOOD2_DONE

step "3 5-LSTM-KO 학습(비교 판)"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-food2-KO --config configs/s5_6_T_food2_ko.json
echo TRAIN_KO_DONE

step "4 test 보고(두 판)"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-food2 --config configs/s5_6_T_food2.json \
  --report --reason "5-LSTM 식품 고정표 v2 다시 학습 보고(사용자 결정 2026-11-12)"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-food2-KO --config configs/s5_6_T_food2_ko.json \
  --report --reason "5-LSTM-KO(Koutsoumanis 6칼럼) 비교 보고(사용자 결정 2026-11-12)"
echo REPORTS_DONE

step "5 KO 6칼럼 분포"
$P -I scripts/s_ko_dist.py $A --out $R/s5_6/ko_dist --config configs/s_ko_dist.json
echo KO_DIST_DONE

step "6 최종 고르기(식품 갱신 판)"
$P -I scripts/s_final.py $A --out $R/final_food2 --config configs/s_final_food2.json
echo FINAL_DECIDED

step "7 최종 중요도(6부분)"
$P -I scripts/s_final.py $A --out $R/final_food2 --config configs/s_final_food2.json --importance
echo IMPORTANCE_DONE

step "8 최종 보고"
$P -I scripts/s_final.py $A --out $R/final_food2 --config configs/s_final_food2.json \
  --report --reason "최종 보고(식품 고정표 v2 다시 학습 판 — 사용자 결정 2026-11-12)"
echo ALL_DONE
