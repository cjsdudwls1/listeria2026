#!/bin/bash
# 사용: restore_preds.sh <up|down> <results|results_dom> <단계/판 ...>
#  up   : 로컬(final_results*/<단계/판>/pred_*.parquet)을 서버 판 폴더로 올리고 md5 대조 → .restored_for_report 표지
#  down : 표지가 있는 판의 서버 pred_* 를 로컬 md5 와 다시 대조한 뒤 지우고 표지 제거(로컬 사본은 그대로)
MODE="$1"; ROOT="$2"; shift 2
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"
case "$ROOT" in
  results) LBASE="C:/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final_results" ;;
  results_dom) LBASE="C:/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final_results_dom" ;;
  *) echo "뿌리는 results 또는 results_dom"; exit 1 ;;
esac
for SP in "$@"; do
  LM=$(cd "$LBASE/$SP" 2>/dev/null && md5sum pred_*.parquet | sed 's/ \*/  /' | sort -k2)
  [ -z "$LM" ] && { echo "$SP 로컬 예측 없음 — 건너뜀"; continue; }
  if [ "$MODE" = up ]; then
    scp -q -F "$CFG" "$LBASE/$SP"/pred_*.parquet "gpu:cyj_listeria_final/$ROOT/$SP/" < /dev/null || { echo "$SP 올리기 실패"; continue; }
    SM=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/$ROOT/$SP && md5sum pred_*.parquet | sort -k2" < /dev/null)
    if [ "$(echo "$SM" | awk '{print $1,$2}')" = "$(echo "$LM" | awk '{print $1,$2}')" ]; then
      ssh -F "$CFG" gpu "date -u +%FT%TZ > ~/cyj_listeria_final/$ROOT/$SP/.restored_for_report" < /dev/null; echo "$SP 올림"
    else echo "$SP md5 다름 — 확인 필요"; fi
  else
    SM=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/$ROOT/$SP && [ -f .restored_for_report ] && md5sum pred_*.parquet | sort -k2" < /dev/null)
    if [ -n "$SM" ] && [ "$(echo "$SM" | awk '{print $1,$2}')" = "$(echo "$LM" | awk '{print $1,$2}')" ]; then
      ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/$ROOT/$SP && rm -f pred_*.parquet .restored_for_report" < /dev/null; echo "$SP 다시 지움"
    else echo "$SP 표지 없음 또는 md5 다름 — 안 지움"; fi
  fi
done
ssh -F "$CFG" gpu "df -h ~ | tail -1" < /dev/null
