# pptx 를 zip 으로 열어 슬라이드별 글자·이미지 목록을 뽑고, 지정한 슬라이드의 이미지를 꺼낸다(읽기만).
# 사용: python -I pptx_extract.py <pptx> <out_dir> [슬라이드번호들 쉼표]
import sys, zipfile, re, os, posixpath
sys.stdout.reconfigure(encoding="utf-8")

pptx, out = sys.argv[1], sys.argv[2]
want = set(int(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 and sys.argv[3] else set()
z = zipfile.ZipFile(pptx)
names = z.namelist()
slides = sorted([n for n in names if re.match(r'ppt/slides/slide\d+\.xml$', n)],
                key=lambda s: int(re.findall(r'\d+', s)[0]))
lines = []
for s in slides:
    no = int(re.findall(r'\d+', s)[0])
    xml = z.read(s).decode('utf8')
    texts = re.findall(r'<a:t>([^<]*)</a:t>', xml)
    rel = s.replace('slides/', 'slides/_rels/') + '.rels'
    imgs, links = [], []
    if rel in names:
        rx = z.read(rel).decode('utf8')
        for m in re.finditer(r'<Relationship ([^>]*)/>', rx):
            a = m.group(1)
            tgt = re.search(r'Target="([^"]*)"', a).group(1)
            typ = re.search(r'Type="([^"]*)"', a).group(1)
            if typ.endswith('/image'):
                imgs.append(posixpath.normpath(posixpath.join('ppt/slides', tgt)))
            elif typ.endswith('/hyperlink'):
                links.append(tgt)
    lines.append(f'### slide {no}\nTEXT: ' + ' | '.join(t for t in texts if t.strip()) +
                 f'\nIMAGES: {imgs}\nLINKS: {links}\n')
    if no in want:
        for i, im in enumerate(imgs):
            ext = os.path.splitext(im)[1]
            with open(os.path.join(out, f'slide{no:02d}_{i}{ext}'), 'wb') as f:
                f.write(z.read(im))
with open(os.path.join(out, 'slides_text.txt'), 'w', encoding='utf8') as f:
    f.write('\n'.join(lines))
print(len(slides), 'slides')
