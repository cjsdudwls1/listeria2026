import os, sys, time
root=sys.argv[1]; depth=int(sys.argv[2])
agg={}
for dp,dn,fn in os.walk(root):
    rel=os.path.relpath(dp,root)
    parts=[] if rel=='.' else rel.split(os.sep)
    for f in fn:
        p=os.path.join(dp,f)
        try: st=os.stat(p)
        except: continue
        key=os.sep.join((parts+[f])[:depth]) if len(parts)+1>depth else os.sep.join(parts+[f])
        a=agg.setdefault(key,[0,0,0])
        a[0]+=st.st_size; a[1]+=1; a[2]=max(a[2],st.st_mtime)
for k in sorted(agg):
    s,c,m=agg[k]
    print(f"{s/1e6:10.1f}MB\t{c}\t{time.strftime('%Y-%m-%d',time.localtime(m))}\t{k}")
