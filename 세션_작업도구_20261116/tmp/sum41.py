import re
L=open(r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\plan_v41.md',encoding='utf-8',newline='').read().split('\r\n')
i=next(k for k,l in enumerate(L) if l.startswith('### A-9'))
tot=0;out=[]
for l in L[i:i+40]:
    if l.startswith('| ') and not l.startswith('| 단계') and not l.startswith('| 합계'):
        c=l.split('|')[2].strip().replace(',','')
        if re.fullmatch(r'\d+',c): tot+=int(c); out.append(c)
        else: out.append('skip '+c)
out.append('sum '+str(tot))
open(r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\sum41.txt','w',encoding='utf-8').write('\n'.join(out))
