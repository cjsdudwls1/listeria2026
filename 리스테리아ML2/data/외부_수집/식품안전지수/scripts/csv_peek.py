# 받은 CSV 들의 인코딩·머리줄·앞 몇 줄·줄 수를 출력(읽기만, 내용 변경 없음).
# 사용: python -I csv_peek.py <파일1> [파일2 ...]  (앞줄 수는 환경변수 대신 고정 3줄)
import sys
sys.stdout.reconfigure(encoding="utf-8")
for p in sys.argv[1:]:
    b = open(p, 'rb').read()
    enc = None
    for e in ('utf-8-sig', 'cp949'):
        try:
            s = b.decode(e)
            enc = e
            break
        except UnicodeDecodeError:
            continue
    if enc is None:
        s = b.decode('cp949', errors='replace'); enc = 'cp949(깨짐 있음)'
    lines = s.splitlines()
    print(f'== {p.split(chr(92))[-1].split("/")[-1]} | 인코딩={enc} | 줄수(머리줄 포함)={len(lines)}')
    for ln in lines[:4]:
        print('   ', ln[:400])
