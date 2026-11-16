# 그림 PDF 한 쪽을 가로 띠 여러 장으로 크게 저장(읽기만). py -3.14 -I 로 실행.
# 사용: py -3.14 -I gw_render_strips.py <pdf> <쪽> <저장폴더> <배율> <띠수> [위여백비율] [아래여백비율] [겹침비율]
import sys, os, fitz
pdf, pno, outdir, z, k = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
top = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
bot = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
ov = float(sys.argv[8]) if len(sys.argv) > 8 else 0.02
doc = fitz.open(pdf)
pg = doc[pno - 1]
r = pg.rect
y0 = r.y0 + r.height * top
y1 = r.y1 - r.height * bot
h = (y1 - y0) / k
os.makedirs(outdir, exist_ok=True)
base = os.path.splitext(os.path.basename(pdf))[0]
for i in range(k):
    a = max(y0, y0 + i * h - r.height * ov)
    b = min(y1, y0 + (i + 1) * h + r.height * ov)
    clip = fitz.Rect(r.x0, a, r.x1, b)
    pm = pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip)
    out = os.path.join(outdir, f'p{pno}_띠{i + 1}of{k}.png')
    pm.save(out)
    print(out, pm.width, pm.height)
print('page rect', r)
