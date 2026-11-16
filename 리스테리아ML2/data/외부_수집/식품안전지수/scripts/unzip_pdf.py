# 연보 zip 안의 pdf 하나를 안전한 이름으로 풀기(경로 조작 방지: 파일 이름을 새로 지정).
# 사용: python -I unzip_pdf.py <zip> <풀 폴더> <새 파일이름.pdf>
import sys, zipfile, os, shutil
sys.stdout.reconfigure(encoding="utf-8")
zp, outdir, newname = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
with zipfile.ZipFile(zp) as z:
    pdfs = [i for i in z.infolist() if not i.is_dir()]
    if len(pdfs) != 1:
        print('파일이 1개가 아님:', [i.filename for i in pdfs]); sys.exit(1)
    with z.open(pdfs[0]) as src, open(os.path.join(outdir, newname), 'wb') as dst:
        shutil.copyfileobj(src, dst)
    print('풀림', newname, pdfs[0].file_size)
