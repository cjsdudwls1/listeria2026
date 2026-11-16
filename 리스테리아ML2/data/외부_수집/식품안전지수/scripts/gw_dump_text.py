# 지하수조사연보 PDF 의 지정 쪽 글자를 파일로 저장(읽기만). py -3.14 -I 로 실행(fitz 필요).
# 사용: py -3.14 -I gw_dump_text.py <pdf> <시작쪽> <끝쪽> <출력txt> [words]
#   words 를 주면 낱말 단위 좌표(x0,y0,x1,y1,글자)도 함께 저장
import sys, fitz
pdf, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
mode = sys.argv[5] if len(sys.argv) > 5 else ''
doc = fitz.open(pdf)
with open(out, 'w', encoding='utf-8') as f:
    f.write(f'# {pdf} 쪽수={doc.page_count}\n')
    for i in range(a - 1, min(b, doc.page_count)):
        pg = doc[i]
        f.write(f'---- p{i + 1}\n')
        if mode == 'words':
            ws = pg.get_text('words')
            for w in ws:
                f.write(f'{w[0]:.1f}\t{w[1]:.1f}\t{w[2]:.1f}\t{w[3]:.1f}\t{w[4]}\n')
        else:
            f.write(pg.get_text())
print('ok', out)
