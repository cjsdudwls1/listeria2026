import os, sys, csv, time
out=sys.argv[1]; roots=sys.argv[2:]
with open(out,'w',newline='',encoding='utf-8-sig') as fh:
    w=csv.writer(fh); w.writerow(['root','rel','size','mtime'])
    for root in roots:
        for dp,dn,fn in os.walk(root):
            if os.path.basename(dp)=='.git' : dn[:]=[]; 
            for f in fn:
                p=os.path.join(dp,f)
                try: st=os.stat(p)
                except: continue
                w.writerow([root,os.path.relpath(p,root),st.st_size,time.strftime('%Y-%m-%d',time.localtime(st.st_mtime))])
