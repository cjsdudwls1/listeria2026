D = r'C:\Users\NT751\.claude\jobs\a19471fc\tmp\concise'
P = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'


def trail(lines):
    n = 0
    for l in reversed(lines):
        if l.strip() == '':
            n += 1
        else:
            break
    return n


merged = []
for k in range(1, 9):
    a = open(f'{D}\\in_{k}.md', encoding='utf-8').read().replace('\r\n', '\n').split('\n')
    b = open(f'{D}\\out_{k}.md', encoding='utf-8').read().replace('\r\n', '\n').split('\n')
    ta, tb = trail(a), trail(b)
    while tb > ta:
        b.pop(); tb -= 1
    while tb < ta:
        b.append(''); tb += 1
    merged += b

text = '\n'.join(merged)
old = '3단계 판·운영 칼럼 판은 안 돌림. 낙농장(L1): 국내는 받은 축사 엑셀로 우리가 시군구별 농장 수를 셈, 해외는 데이터 팀 원천 확인 뒤(사용자 11-04 밤, §7-3).'
cnt = text.count(old)
print('header anchor count', cnt)
if cnt == 1:
    text = text.replace(old, old + ' 계획서 전체 말투를 간결체로 다시 씀(사용자 "계획서 말투 간결체로 수정" — 내용·인용·숫자는 그대로).')
open(P, 'wb').write(text.replace('\n', '\r\n').encode('utf-8'))
print('merged lines', len(merged))
