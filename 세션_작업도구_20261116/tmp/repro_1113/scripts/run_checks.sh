#!/usr/bin/env bash
# 검증 점검 차례(재학습 말고 전부) — 문서 명령 그대로 + 결과 사본을 $R/results 로(다음 실행이 덮어쓰기 전에)
set -u
source "$HOME/cyj_listeria_repro_1113/my_scripts/env.sh"
source "$R/venv/bin/activate"
cd "$PKG/01_소스코드"
L=$R/logs; O=$R/results; mkdir -p "$O" "$R/repred_gpu"
D03=$PKG/03_학습검증평가데이터_분할정보; D05=$PKG/05_성능검증_상세자료; D06=$PKG/06_입출력정의_샘플
step() {
  local name=$1; shift
  echo "[$(date '+%F %T')] 시작 $name: $*" | sed "s#$HOME#~#g" | tee -a "$L/steps.log"
  local t0=$(date +%s)
  "$@" > "$L/$name.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] 끝 $name rc=$rc $(( $(date +%s) - t0 ))초" | tee -a "$L/steps.log"
  return $rc
}
GPU="$R/my_scripts/gpu_cap_run.py"

# ---- (1. 환경 check_env 는 재학습 전에 따로 돌림) 2. 단위 시험 · 3. 저장 예측 → 점수
step 03_pytest python -X utf8 -m pytest -q -rs tests
step 04_reproduce_metrics python -X utf8 검증용/reproduce_metrics.py
cp -p "$D05/성능재현_요약.csv" "$O/성능재현_요약.csv"; cp -p "$D05/성능재현_h별.csv" "$O/성능재현_h별.csv"

# ---- 4. 저장 모델 → 표본 3,000쌍(CPU · GPU)
step 05_predict_cpu_3000 python -X utf8 검증용/predict_sample.py
cp -p "$D06/샘플출력/refit_예측대조.csv" "$O/refit_예측대조_cpu_3000.csv"
cp -p "$D06/샘플입력/refit_입력예시.npz" "$O/refit_입력예시_cpu.npz"; cp -p "$D06/샘플입력/refit_입력예시_행.csv" "$O/refit_입력예시_행_cpu.csv"
step 06_predict_gpu_3000 python -X utf8 "$GPU" 검증용/predict_sample.py --device cuda
cp -p "$D06/샘플출력/refit_예측대조.csv" "$O/refit_예측대조_gpu_3000.csv"
cp -p "$D06/샘플입력/refit_입력예시.npz" "$O/refit_입력예시_gpu.npz"

# ---- 5. 평가 쌍 전부(GPU) + 채점
step 07_predict_gpu_all_refit python -X utf8 "$GPU" 검증용/predict_sample.py --per-h 0 --device cuda
cp -p "$D06/샘플출력/refit_재예측_전체.parquet" "$R/repred_gpu/pred_refit.parquet"
cp -p "$D06/샘플출력/refit_예측대조_요약.csv" "$O/refit_예측대조_요약_gpu.csv"
step 08_rescore_gpu_all python -X utf8 검증용/reproduce_metrics.py --test-pred ../06_입출력정의_샘플/샘플출력/refit_재예측_전체.parquet
cp -p "$D05/성능재현_요약_재예측.csv" "$O/성능재현_요약_재예측_gpu.csv"

# ---- (추가) 교차검증 fold 1~3 저장 모델도 GPU 로 전부 다시 예측 → 교차검증 점수까지 다시 계산
for f in fold1 fold2 fold3; do
  step 09_predict_gpu_all_$f python -X utf8 "$GPU" 검증용/predict_sample.py --split $f --per-h 0 --device cuda
  cp -p "$D06/샘플출력/${f}_재예측_전체.parquet" "$R/repred_gpu/pred_${f}.parquet"
  cp -p "$D06/샘플출력/${f}_예측대조_요약.csv" "$O/${f}_예측대조_요약_gpu.csv"
done
# reproduce_metrics --pred-dir 는 결과를 05 의 '_다시학습' 파일에 씀 → 재학습 결과와 섞이지 않게 패키지 밖 사본 뿌리에서 실행
A=$R/pkg_alt; mkdir -p "$A/01_소스코드/검증용" "$A/05_성능검증_상세자료"
cp -p "$PKG/01_소스코드/검증용/reproduce_metrics.py" "$A/01_소스코드/검증용/"
ln -sfn "$PKG/02_학습모델파일" "$A/02_학습모델파일"; ln -sfn "$D03" "$A/03_학습검증평가데이터_분할정보"
ln -sfn "$D05/예측값" "$A/05_성능검증_상세자료/예측값"
step 10_score_repred_gpu python -X utf8 "$A/01_소스코드/검증용/reproduce_metrics.py" --pred-dir "$R/repred_gpu"
cp -p "$A/05_성능검증_상세자료/성능재현_요약_다시학습.csv" "$O/성능재현_요약_저장모델재예측_gpu.csv"
cp -p "$A/05_성능검증_상세자료/예측대조_다시학습.csv" "$O/예측대조_저장모델재예측_gpu.csv"

# ---- 5'. 평가 쌍 전부(CPU) + 채점
step 11_predict_cpu_all_refit python -X utf8 검증용/predict_sample.py --per-h 0
cp -p "$D06/샘플출력/refit_재예측_전체.parquet" "$O/refit_재예측_전체_cpu.parquet"
cp -p "$D06/샘플출력/refit_예측대조_요약.csv" "$O/refit_예측대조_요약_cpu.csv"
step 12_rescore_cpu_all python -X utf8 검증용/reproduce_metrics.py --test-pred ../06_입출력정의_샘플/샘플출력/refit_재예측_전체.parquet
cp -p "$D05/성능재현_요약_재예측.csv" "$O/성능재현_요약_재예측_cpu.csv"

# ---- 장치끼리 예측 비교
C="$R/my_scripts/compare_preds.py"; CP="$O/예측비교_장치.csv"; rm -f "$CP"
{
python -I "$C" "표본3000 CPU vs GPU(A100)" "$O/refit_예측대조_cpu_3000.csv" 다시_낸_예측 "$O/refit_예측대조_gpu_3000.csv" 다시_낸_예측 "$CP"
python -I "$C" "표본3000 GPU(A100) vs 저장" "$O/refit_예측대조_gpu_3000.csv" 다시_낸_예측 "$D05/예측값/pred_refit.parquet" pred "$CP"
python -I "$C" "표본3000 CPU vs 저장" "$O/refit_예측대조_cpu_3000.csv" 다시_낸_예측 "$D05/예측값/pred_refit.parquet" pred "$CP"
python -I "$C" "평가 전부 CPU vs GPU(A100)" "$O/refit_재예측_전체_cpu.parquet" pred "$R/repred_gpu/pred_refit.parquet" pred "$CP"
python -I "$C" "평가 전부 GPU(A100) vs 저장" "$R/repred_gpu/pred_refit.parquet" pred "$D05/예측값/pred_refit.parquet" pred "$CP"
python -I "$C" "평가 전부 CPU vs 저장" "$O/refit_재예측_전체_cpu.parquet" pred "$D05/예측값/pred_refit.parquet" pred "$CP"
for f in fold1 fold2 fold3; do
  python -I "$C" "$f 전부 GPU(A100) vs 저장" "$R/repred_gpu/pred_$f.parquet" pred "$D05/예측값/pred_$f.parquet" pred "$CP"
done
} > "$L/13_compare_preds.log" 2>&1
# 입력 예시 배열(장치 무관해야 함): 패키지 원본 vs 다시 만든 것
python -I -c "
import sys, numpy as np
o, c, g = (np.load(p) for p in sys.argv[1:4])
for k in sorted(o.files):
    print(k, '원본=CPU', np.array_equal(o[k], c[k]), '원본=GPU', np.array_equal(o[k], g[k]), o[k].shape)
" "$R/orig/06/샘플입력/refit_입력예시.npz" "$O/refit_입력예시_cpu.npz" "$O/refit_입력예시_gpu.npz" > "$L/13_compare_input_arrays.log" 2>&1

# ---- 분할표 다시 만들기
step 14_make_split_table python -X utf8 검증용/make_split_table.py
md5sum "$D03/라벨_분할표.csv" "$R/orig/03/라벨_분할표.csv" "$D03/분할_요약.csv" "$R/orig/03/분할_요약.csv" | sed "s#$HOME#~#g" >> "$L/14_make_split_table.log"

# ---- 단계 0 다시 만들기(패키지 입력으로 되는 범위) — run_pipeline.sh s0 그대로, 결과 뿌리만 고정 이름
export R0="$R/재생성_단계0"
step 15_s0 bash run_pipeline.sh s0
step 16_s0_gamma python -X utf8 검증용/compare_tables.py "$R0/s0/0-4_spi/SPI_감마모수_0-4.parquet" "$D03/inputs_root/s0/0-4_spi/SPI_감마모수_0-4.parquet"
unset R0
echo "[$(date '+%F %T')] run_checks 끝" | tee -a "$L/steps.log"
