import re,sys
t=open(sys.argv[1],encoding='utf-8').read().split('\n')
pats=sys.argv[2:]
for i,l in enumerate(t,1):
    for p in pats:
        for m in re.finditer(p,l):
            s=max(0,m.start()-35); e=min(len(l),m.end()+35)
            print(i, p, '|', l[s:e].strip())
