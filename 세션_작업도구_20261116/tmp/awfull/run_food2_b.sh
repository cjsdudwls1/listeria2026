#!/bin/bash
# 대기열 B(2026-11-12, 사용자 "Koutsoumanis 생장식 6칼럼 추가해서 점수 더 좋으면 그게 최종"): 대기열 A(KO 학습·두 판 보고·KO 분포)가
# 끝나면 val 로 최종을 고르고(s_food2_pick.py — 2차 채택 규칙) 고른 설정으로 원 대기열 6~8단계(최종 고르기·중요도·보고).
# 코드 사본 food2b_20261112 = food2_20261112 + s_food2_pick.py·s_final_food2_ko.json·최종 중요도 추가 그룹·MLflow 최종 기록 경로.
set -euo pipefail
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/food2b_20261112
R=$B/results
P=$B/venv/bin/python
A="--root $B/data --inputs $R"
QA=4043962          # 대기열 A(bash logs/run_food2_a.sh)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
cd $C
step() { echo "=== $(date '+%F %T') $1"; }

step "A 기다림: 대기열 A(pid $QA) 끝"
while kill -0 $QA 2>/dev/null; do sleep 60; done
grep -qx 'A_DONE' $B/logs/run_food2_a.log || { echo "대기열 A 가 A_DONE 없이 끝남 — 멈춤"; exit 1; }
for d in s5_6/5-6-T-food2/5-LSTM-food2 s5_6/5-6-T-food2-KO/5-LSTM-KO; do
  test -s "$R/$d/done.json" || { echo "없음: $R/$d/done.json"; exit 1; }
done
PICK=$R/logs/food2_final_pick.json

step "6a 최종 고르기(val — 5-LSTM-food2 현재 판 vs 5-LSTM-KO 후보)"
if [ -e "$PICK" ]; then      # 이어 하기(검수 2026-11-12): 고른 기록이 있으면 그 설정을 다시 씀 — 다시 고르지 않음
  CFG=$($P -I -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["chosen_config"])' "$PICK")
  echo "고르기 기록 있음 — 그 설정으로 이어 함: $CFG"
else
  test ! -e "$R/final_food2" || { echo "고르기 기록 없이 final_food2 가 있음 — 멈춤"; exit 1; }
  CFG=$($P -I scripts/s_food2_pick.py --inputs $R --config-food2 configs/s_final_food2.json \
          --config-ko configs/s_final_food2_ko.json | tail -n 1)
fi
test -s "$CFG" || { echo "고른 설정 파일 없음: $CFG"; exit 1; }
if [ -e "$R/final_food2/final_decision.json" ]; then   # 이어 하기: 이미 고른 최종이 설정과 같은지
  $P -I -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); n=json.load(open(sys.argv[2], encoding="utf-8"))["final_override"]["name"]; assert d["final"] == n, (d["final"], n); print("이어 하기: final_decision 최종 =", n)' \
    "$R/final_food2/final_decision.json" "$CFG"
fi
echo "고른 설정: $CFG"
echo PICKED

step "6 최종 고르기(식품 갱신 판)"
$P -I scripts/s_final.py $A --out $R/final_food2 --config "$CFG"
echo FINAL_DECIDED

step "7 최종 중요도(6부분 — 끝났으면 s_final 이 건너뜀)"
$P -I scripts/s_final.py $A --out $R/final_food2 --config "$CFG" --importance
echo IMPORTANCE_DONE

step "8 최종 보고"
if [ -s "$R/final_food2/report/run_meta.json" ]; then   # 끝 표지 = 보고 끝에 쓰는 run_meta.json(검수 2026-11-12)
  echo "보고 이미 있음 — 건너뜀"
elif [ -e "$R/final_food2/report" ]; then
  echo "보고 폴더가 반쯤 있음 — 멈춤(확인 뒤 지휘자가 정리)"; exit 1
else
  $P -I scripts/s_final.py $A --out $R/final_food2 --config "$CFG" \
    --report --reason "최종 보고(식품 갱신 — s_food2_pick 이 val 로 고른 판, 사용자 결정 2026-11-12)"
fi
echo ALL_DONE
