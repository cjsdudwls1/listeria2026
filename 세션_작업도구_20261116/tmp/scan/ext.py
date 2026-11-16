import csv, sys, os
f=sys.argv[1]; prefix=sys.argv[2].replace('/',os.sep)
agg={}
for r in csv.DictReader(open(f,encoding='utf-8-sig')):
    if not r['rel'].startswith(prefix): continue
    b=os.path.basename(r['rel']); e=os.path.splitext(b)[1].lower() or b
    a=agg.setdefault(e,[0,0]); a[0]+=int(r['size']); a[1]+=1
for k,(s,c) in sorted(agg.items(),key=lambda x:-x[1][0])[:25]: print(f"{s/1e6:10.1f}MB\t{c}\t{k}")
