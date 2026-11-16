# 지하수조사연보 PDF 12개의 파일 속 날짜(creationDate·modDate)를 CSV 로 저장(읽기만). py -3.14 -I 로 실행(fitz 필요).
# 사용: py -3.14 -I gw_pdf_dates.py <_압축풀기 폴더> <출력 csv>
import sys, os, csv, fitz
base, out = sys.argv[1], sys.argv[2]
rows = []
for y in range(2014, 2026):
    p = os.path.join(base, str(y), f'지하수조사연보_{y}.pdf')
    m = fitz.open(p).metadata
    c = m.get('creationDate') or ''; d = m.get('modDate') or ''
    f = lambda s: f'{s[2:6]}-{s[6:8]}-{s[8:10]}' if s.startswith('D:') and len(s) >= 10 else ''
    rows.append({'판': y, 'creationDate_원문': c, '만든날짜': f(c), 'modDate_원문': d, '수정날짜': f(d)})
with open(out, 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print('ok', len(rows))
