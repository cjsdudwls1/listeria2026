#!/usr/bin/env bash
# 결과·로그를 한 묶음(tar)으로 — 서버 홈 경로는 '~' 로 가림. 큰 예측 parquet 은 넣지 않음(요약 csv 만)
set -u
source "$HOME/cyj_listeria_repro_1113/my_scripts/env.sh"
C=$R/collect; rm -rf "$C"; mkdir -p "$C/logs" "$C/results" "$C/retrain" "$C/package_written"
cp -p $R/logs/*.log $R/logs/pip_freeze_repro.txt "$C/logs/" 2>/dev/null
cp -p $R/results/*.csv "$C/results/" 2>/dev/null
cp -p $R/results/*.npz "$C/results/" 2>/dev/null
OUT=$(ls -d $R/재학습_3-0-LSTM_* | head -1)
if [ -n "$OUT" ]; then
  (cd "$OUT/3-0-LSTM" && find . -type f ! -name "pred_*.parquet" -print0 | xargs -0 -I{} cp --parents -p {} "$C/retrain/")
  cp -p "$OUT/_verify_experiment_log.jsonl" "$C/retrain/" 2>/dev/null
  md5sum "$OUT"/3-0-LSTM/pred_*.parquet > "$C/retrain/pred_md5.txt" 2>/dev/null
fi
# 패키지 사본 안에 실행이 쓴 파일(05 '_다시학습'·점검 run_pipeline 로그)
cp -p "$PKG/05_성능검증_상세자료/성능재현_요약_다시학습.csv" "$PKG/05_성능검증_상세자료/예측대조_다시학습.csv" "$C/package_written/" 2>/dev/null
cp -p $PKG/점검/run_pipeline_*.log "$C/package_written/" 2>/dev/null
cp -p "$R/재생성_0-6_missing/s0/0-6_missing/missing_lists.json" "$C/results/missing_lists_재생성.json" 2>/dev/null
# 실행 뒤 패키지 사본의 해시 대조(어떤 파일을 덮어썼는지 기록)
(cd "$PKG/01_소스코드" && python3.12 -X utf8 검증용/make_file_list.py --check) > "$C/logs/99_hash_after_runs.log" 2>&1
# 홈 경로 가리기
grep -rlI --null "$HOME" "$C" 2>/dev/null | xargs -0 -r sed -i "s#$HOME#~#g"
grep -rlI "$(basename "$HOME")" "$C" 2>/dev/null | sed "s#$HOME#~#g" | sed 's/^/[계정 이름 남은 파일] /'
cd "$R" && tar -cf collect.tar collect && md5sum collect.tar | cut -d' ' -f1 && du -sh collect | cut -f1
