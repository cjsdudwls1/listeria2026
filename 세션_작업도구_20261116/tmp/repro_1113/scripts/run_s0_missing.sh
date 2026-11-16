#!/usr/bin/env bash
# 단계 0-6 결측 표시 목록 다시 만들기 — 01_소스코드/README.md §5 표 그대로(입력 = 패키지 단계 0 산출물을 새 뿌리에 연결)
set -u
source "$HOME/cyj_listeria_repro_1113/my_scripts/env.sh"
source "$R/venv/bin/activate"
IR=$PKG/03_학습검증평가데이터_분할정보/inputs_root
R1=$R/재생성_0-6_missing
mkdir -p "$R1/s0"
for d in 0-1 0-4_v2 0-4_weather_overseas 0-4_spi 0-6_static; do ln -sfn "$IR/s0/$d" "$R1/s0/$d"; done
cd "$PKG/01_소스코드/단계0_코드사본/0-6_missing"
echo "[$(date '+%F %T')] 시작: python -I scripts/s0_6_missing.py --root ../../../03_학습검증평가데이터_분할정보/data_root --inputs <R1> --out <R1>/s0/0-6_missing"
t0=$(date +%s)
python -I scripts/s0_6_missing.py --root ../../../03_학습검증평가데이터_분할정보/data_root --inputs "$R1" --out "$R1/s0/0-6_missing"
rc=$?
echo "[$(date '+%F %T')] 끝 rc=$rc $(( $(date +%s) - t0 ))초"
python -I -c "
import json, sys, hashlib
a = json.load(open(sys.argv[1], encoding='utf-8')); b = json.load(open(sys.argv[2], encoding='utf-8'))
sa, sb = a.pop('seconds', None), b.pop('seconds', None)
print('missing_lists.json 내용(seconds 칸 빼고) 같음:', a == b, '· 걸린 초 새', sa, '/ 원본', sb)
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print('  다른 칸:', k)
md = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
print('missing_detail.csv md5', md(sys.argv[3]), '/', md(sys.argv[4]), '같음' if md(sys.argv[3]) == md(sys.argv[4]) else '다름')
" "$R1/s0/0-6_missing/missing_lists.json" "$IR/s0/0-6_missing/missing_lists.json" "$R1/s0/0-6_missing/missing_detail.csv" "$IR/s0/0-6_missing/missing_detail.csv"
