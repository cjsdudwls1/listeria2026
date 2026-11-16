import re, sys
p = sys.argv[1]
L = open(p, encoding='utf-8').read().split('\n')
PIPE = re.compile(r'(?<!\\)\|')
SEP = re.compile(r'^\|[-:| ]+\|\s*$')


def n(l):
    return len(PIPE.findall(l))


bad = 0
i = 0
while i < len(L):
    if L[i].startswith('|') and i + 1 < len(L) and SEP.match(L[i + 1].strip()):
        h = n(L[i])
        j = i + 2
        while j < len(L) and L[j].startswith('|'):
            if n(L[j]) != h:
                print('mismatch line', j + 1, 'expected', h, 'got', n(L[j]))
                bad += 1
            j += 1
        i = j
    else:
        i += 1
print('mismatches', bad)
