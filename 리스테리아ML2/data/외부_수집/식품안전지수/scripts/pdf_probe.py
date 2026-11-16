# PDF 쪽수·목차·글자를 확인(읽기만). fitz(PyMuPDF) 필요 → py -3.14 -I 로 실행.
# 사용:
#   python -I pdf_probe.py info <pdf>                      쪽수·목차(북마크)
#   python -I pdf_probe.py text <pdf> <시작쪽> <끝쪽>       글자 출력(1부터 셈)
#   python -I pdf_probe.py find <pdf> <키워드>              키워드가 있는 쪽 번호
#   python -I pdf_probe.py png <pdf> <쪽> <저장경로> [배율]  한 쪽을 그림으로 저장
import sys, fitz
sys.stdout.reconfigure(encoding="utf-8")
cmd, path = sys.argv[1], sys.argv[2]
doc = fitz.open(path)
if cmd == 'info':
    print('쪽수', doc.page_count)
    for lvl, title, page in doc.get_toc():
        print('  ' * (lvl - 1) + f'{title} ... p{page}')
elif cmd == 'text':
    a, b = int(sys.argv[3]), int(sys.argv[4])
    for i in range(a - 1, min(b, doc.page_count)):
        print(f'---- p{i + 1}')
        print(doc[i].get_text())
elif cmd == 'find':
    kw = sys.argv[3]
    hits = [i + 1 for i in range(doc.page_count) if kw in doc[i].get_text()]
    print(len(hits), hits[:300])
elif cmd == 'png':
    p, out = int(sys.argv[3]), sys.argv[4]
    z = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5
    doc[p - 1].get_pixmap(matrix=fitz.Matrix(z, z)).save(out)
    print('저장', out)
