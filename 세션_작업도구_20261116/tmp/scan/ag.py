import csv, sys, os
S=os.sep
f=sys.argv[1]; prefix=sys.argv[2].replace('/',S); depth=int(sys.argv[3])
agg={}
for r in csv.DictReader(open(f,encoding='utf-8-sig')):
    rel=r['rel']
    if not rel.startswith(prefix): continue
    parts=rel.split(S)
    k=S.join(parts[:depth]) + (S if len(parts)>depth else '')
    a=agg.setdefault(k,[0,0,'']); a[0]+=int(r['size']); a[1]+=1; a[2]=max(a[2],r['mtime'])
for k in sorted(agg): s,c,m=agg[k]; print(f"{s/1e6:10.1f}MB\t{c}\t{m}\t{k}")
