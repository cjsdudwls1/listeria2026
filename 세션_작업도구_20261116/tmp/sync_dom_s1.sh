#!/bin/bash
# 국내판 단계 1(1-1·1-2) 끝난 판 예측을 로컬로 옮김(md5 대조 뒤 서버 pred_*.parquet 삭제).
# 1-3 결정 전: 끝난 판 중 CV 상위 3개는 남김(1차 베이스라인 = 70판 CV 1위가 항상 남도록).
# 1-3 결정 뒤: 결정 파일이 가리키는 판(baseline·A_variant·tree_mode_from·1-LSTM-W<W_DL>)만 남기고 나머지 옮김 → 다 옮기면 끝.
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"
R="cyj_listeria_final/results_dom"
LBASE="C:/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final_results_dom"
while true; do
  # 서버에서: 끝났고 안 옮긴 판 목록과 남길 판 목록을 한 번에 계산
  OUT=$(ssh -F "$CFG" gpu "~/cyj_listeria_final/venv/bin/python -I - <<'PY'
import json, pathlib
R = pathlib.Path.home() / '$R' / 's1'
fin = {}
for st in ('1-1', '1-2'):
    d = R / st
    if not d.is_dir():
        continue
    for p in sorted(d.iterdir()):
        if p.is_dir() and (p / 'done.json').exists() and (p / 'cv_scores.json').exists():
            fin[st + '/' + p.name] = json.loads((p / 'cv_scores.json').read_text(encoding='utf-8'))['cv_band_weighted']
dec = R / '1-3' / '1-3_decision.json'
if dec.exists():
    D = json.loads(dec.read_text(encoding='utf-8'))
    keepn = {D.get('baseline'), D.get('A_variant'), D.get('tree_mode_from')}
    if D.get('W_DL') is not None:
        keepn.add('1-LSTM-W%03d' % int(D['W_DL']))
    keep = {k for k in fin if k.split('/')[1] in keepn}
    print('DECIDED')
else:
    keep = set(sorted(fin, key=lambda k: -fin[k])[:3])
for k in fin:
    if k not in keep and not (R / k / '.synced_local').exists() and list((R / k).glob('pred_*.parquet')):
        print('MOVE', k)
PY" < /dev/null)
  for K in $(echo "$OUT" | awk '$1=="MOVE"{print $2}'); do
    SP="s1/$K"; ST=$(dirname "$SP"); P=$(basename "$SP")
    SM=$(ssh -F "$CFG" gpu "cd ~/$R/$SP && md5sum pred_*.parquet | sort -k2" < /dev/null)
    mkdir -p "$LBASE/$ST"
    scp -q -F "$CFG" -r "gpu:$R/$SP" "$LBASE/$ST/" || { echo "$(date +%T) $P 받기 실패"; continue; }
    LM=$(cd "$LBASE/$SP" && md5sum pred_*.parquet | sed 's/ \*/  /' | sort -k2)
    if [ -n "$SM" ] && [ "$(echo "$SM" | awk '{print $1,$2}')" = "$(echo "$LM" | awk '{print $1,$2}')" ]; then
      ssh -F "$CFG" gpu "cd ~/$R/$SP && rm -f pred_*.parquet && date -u +%FT%TZ > .synced_local && mkdir -p ~/$R/logs && echo '{\"stage\": \"$ST\", \"variant\": \"$P\", \"local\": \"리스테리아ML2/exp/final_results_dom/$SP\", \"utc\": \"'\$(date -u +%FT%TZ)'\", \"md5_checked\": true}' >> ~/$R/logs/moved_to_local.jsonl" < /dev/null
      echo "$(date +%T) $P 옮김"
    else
      echo "$(date +%T) $P md5 다름 — 안 지움"
    fi
  done
  if echo "$OUT" | grep -q '^DECIDED' && ! echo "$OUT" | grep -q '^MOVE'; then
    echo "끝: 국내판 단계 1 정리"; ssh -F "$CFG" gpu "df -h ~ | tail -1" < /dev/null; exit 0
  fi
  sleep 600
done
