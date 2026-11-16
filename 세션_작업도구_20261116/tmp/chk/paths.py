import re, os, sys
t = open(sys.argv[1], encoding='utf-8').read()
base = r'C:\Users\NT751\Desktop\천영진'
found = sorted(set(re.findall(r'`([^`]*\\[^`]*)`', t)))
for m in found:
    p = os.path.join(base, m)
    print(os.path.exists(p), m)
extra = [r'리스테리아ML2\exp\2_고도화\src\e2fg.py', r'리스테리아ML2\data\데이터셋\생장범위\생장범위_품목3단계.parquet',
         r'리스테리아ML2\exp', r'리스테리아ML2\exp\mlflow']
for m in extra:
    print('extra', os.path.exists(os.path.join(base, m)), m)
