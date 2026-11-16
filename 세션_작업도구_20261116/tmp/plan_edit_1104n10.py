p = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'
b = open(p, 'rb').read()
crlf = b'\r\n' in b
t = b.decode('utf-8').replace('\r\n', '\n')


def rep(old, new, cnt=1):
    global t
    n = t.count(old)
    assert n == cnt, (n, old[:90])
    t = t.replace(old, new)


Q = '"아 너무 어려워 그냥 단순하게 계산 ㄱㄱ 성능안 좋으면 빼면됨"'
rep('결과는 나라마다 축종별 1개 값(§7-3)\n',
    f'결과는 나라마다 축종별 1개 값(§7-3). SPI 국내 예보 11일은 그냥 이어 붙임 — 옮겨 잇기 안 함(사용자 {Q})\n')
rep('국내 예보 11일은 T−1 차이만큼 옮겨 이음 [권고] — ', '국내 예보 11일 이음 방식은 밤 7 에 \'그냥 이음\'으로 정함 — 시험: ')
rep('국내 예보 11일은 T−1 의 기상청·우리 식 차이만큼 옮겨 이음 [권고].',
    f'국내 예보 11일은 그냥 이어 붙임(옮겨 잇기 안 함 — 사용자 11-04 밤 7 {Q}).')
rep('예보 11일(T ~ E)만 예보 강수로 우리 식 + T−1 차이만큼 옮겨 이음 [옮겨 잇기 권고].',
    '예보 11일(T ~ E)만 예보 강수로 우리 식, 그냥 이어 붙임(사용자 11-04 밤 7).')
rep('국내 예보 11일은 T−1 의 차이(기상청 − 우리 식)만큼 옮겨 이음 [권고].',
    f'국내 예보 11일은 그냥 이어 붙임 [확정, 사용자 11-04 밤 7 {Q}] — 옮겨 잇기(T−1 차이만큼 옮김)는 안 함, 성능이 나쁘면 단계 3 빼기에서 빠짐.')
rep('  - 국내는 기상청 값(과거)과 우리 식(예보 11일)이 T 에서 이어짐(결정 17) — ',
    '  - 국내는 기상청 값(과거)과 우리 식(예보 11일)을 T 에서 그냥 이어 붙임(결정 17, 11-04 밤 7 — 성능이 나쁘면 단계 3 에서 빠짐) — ')

out = t.replace('\n', '\r\n') if crlf else t
open(p, 'wb').write(out.encode('utf-8'))

M = r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----\memory'
s = M + r'\spi-drought-index-data.md'
m = open(s, encoding='utf-8').read().rstrip('\n')
m += f'\n- **11-04 밤 7**: 국내 예보 11일은 그냥 이어 붙임 — 옮겨 잇기 안 함(사용자 {Q}). 비전공 설명이 어렵다는 반응 → 이런 보정 안은 길게 설명하지 말고 단순안부터.\n'
open(s, 'w', encoding='utf-8').write(m)
pm = M + r'\listeria-1102-final-integrated-plan.md'
m = open(pm, encoding='utf-8').read()
lines = m.split('\n')
idx = [i for i, l in enumerate(lines) if l.startswith('- **11-04 밤 6(')]
assert len(idx) == 1
lines.insert(idx[0] + 1, f'- **11-04 밤 7**: 나라 경계 파일 만듦(`리스테리아ML2\\data\\외부_수집\\축사\\나라경계_NaturalEarth_v5.1.2_10m.geojson` + .zip). SPI 국내 예보 11일 = 그냥 이어 붙임(옮겨 잇기 권고 철회, 사용자 {Q}).')
open(pm, 'w', encoding='utf-8').write('\n'.join(lines))
print('ok')
