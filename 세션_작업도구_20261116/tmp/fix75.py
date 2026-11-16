import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
def rep(old, new, n=1):
    global s
    c = s.count(old); assert c == n, (c, old[:80]); s = s.replace(old, new)
rep("섞거나 빼지 않고, soil_t_nas·no2·co·so2 가 모두 빠지면 같이 뺌", "섞거나 빼지 않고, soil_t5~t30·soil_t_nas·no2·co·so2·풍향 sin·cos·td1 이 모두 빠지면 같이 뺌")
rep("해외 SPI 는 2015 년 초 NaN(§4-5)", "해외 SPI 는 2015 년 초 NaN(§4-5), 2015 년 초 d 해외 행은 window·차감 앞쪽 날이 2014 년에 닿아 NaN(칼럼마다 결측 표시) — 국내는 같은 경우 RR8 로 제외")
rep("(둘 다 0-1·0-3 의 공개 지연·0-4·", "(둘 다 0-1·0-4·")
p.write_text(s, encoding="utf-8"); print("ok")
