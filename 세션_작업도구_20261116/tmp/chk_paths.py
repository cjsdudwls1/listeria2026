import re,os,io
B=chr(92)
s=io.open(r'C:/Users/NT751/.claude/jobs/a19471fc/tmp/review/plan_v40.md',encoding='utf-8').read()
ps=sorted(set(re.findall('`([^`]*'+B+B+'[^`]*)`',s)))
out=io.open(r'C:/Users/NT751/.claude/jobs/a19471fc/tmp/paths.txt','w',encoding='utf-8')
for p in ps:
    q=p.replace(B,'/')
    out.write(('OK ' if os.path.exists(q) else 'NO ')+p+'\n')
out.close()
