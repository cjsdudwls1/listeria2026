p = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md'
b = open(p, 'rb').read()
crlf = b'\r\n' in b
t = b.decode('utf-8').replace('\r\n', '\n')


def rep(old, new, cnt=1):
    global t
    n = t.count(old)
    assert n == cnt, (n, old[:90])
    t = t.replace(old, new)


QD = '"우유나오는 동물들 다 해야지. 낙농장 수로 ㅇㅇ"'
QS = '"미래 예보값은 다 계산한걸로 들어가는거 맞지? 실제 값이 있어도?"'

lines = t.split('\n')
i = [k for k, l in enumerate(lines) if l.startswith('- 11-04 밤 7 반영:')]
assert len(i) == 1
lines.insert(i[0] + 1, f'- 11-04 밤 8 반영: SPI 예보 칸(T ~ E)은 학습 때도 늘 우리 계산값 — 그 날짜의 기상청 값이 있어도 안 씀(사용자 {QS} → 맞음). 낙농장 = 젖 짜는 가축(젖소·산양·염소·면양)이 들어간 농장 수(사용자 {QD}, 국내 1,360곳). 해외 축사는 데이터 팀 국경선으로 셈(§7-3)')
t = '\n'.join(lines)

rep('학습 땐 관측 강수로 같은 식 [확정,', f'학습 땐 관측 강수로 같은 식 — 그 날짜의 기상청 값이 있어도 예보 칸엔 안 씀(운영과 같게, 사용자 11-04 밤 8 {QS} → 맞음) [확정,')
rep('예보 11일(T ~ E)만 예보 강수로 우리 식, 그냥 이어 붙임(사용자 11-04 밤 7).',
    '예보 11일(T ~ E)만 예보 강수로 우리 식(학습 땐 관측 강수 — 그날 기상청 값이 있어도 안 씀), 그냥 이어 붙임(사용자 11-04 밤 7·8).')
rep("낙농장(가축종에 '젖소'가 들어간 농장)·소·돼지·가금 농장 수 4칼럼",
    '낙농장(젖 짜는 가축 — 젖소·산양·염소·면양이 들어간 농장, 사용자 11-04 밤 8)·소·돼지·가금 농장 수 4칼럼')
rep("시군구별 낙농장(젖소가 들어간 농장) 수 = Weller(2020) '낙농장 밀도'에 맞춤.",
    f"시군구별 낙농장(젖소·산양·염소·면양이 들어간 농장 — 1,360곳: 젖소 831·염소 351·산양 172·면양 9, 겹침 있음) 수 = Weller(2020) '낙농장 밀도'에 맞춤 [확정, 사용자 11-04 밤 8 {QD}].")
rep("낙농장 = '젖소' 포함. 둘 이상 적힌 농장은 각 축종에 셈. 면양·산양·염소·사슴·'기타'는 안 셈.",
    "낙농장 = '젖소'·'산양'·'염소'·'면양' 중 하나라도 포함(한우·육우는 고기소라 뺌). 둘 이상 적힌 농장은 각 축종에 셈. 사슴·'기타'는 안 셈(면양·산양·염소는 낙농장에만 셈).")
rep('    - 낙농장(젖소) 칼럼: 해외 목록에 젖소 구분이 있는지 확인 — 없으면 해외 NaN.',
    '    - 낙농장 칼럼: 해외 목록에서 젖소·젖염소·젖양(우유용)을 가를 수 있는지 확인 — 데이터 팀 결과의 가축종 대응표로 맞추고, 못 가르면 해외 NaN.')
rep('→ 해외 행은 그 나라 값, 목록이 안 덮는 나라는 NaN.', '→ 해외 행은 그 나라 값, 목록이 안 덮는 나라와 0개인 나라는 NaN [권고].')

out = t.replace('\n', '\r\n') if crlf else t
open(p, 'wb').write(out.encode('utf-8'))

M = r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----\memory'
r = M + r'\pending-research-variables-report.md'
m = open(r, encoding='utf-8').read()
a = 'Weller 2020 축사 밀도(L1 4칼럼: 낙농장(젖소)·소·돼지·가금, 10km²당 농장 수 — 국내 시군구·해외 나라).'
assert m.count(a) == 1
m = m.replace(a, 'Weller 2020 축사 밀도(L1 4칼럼: 낙농장(젖 짜는 가축 — 젖소·산양·염소·면양, 11-04 밤 8)·소·돼지·가금, 10km²당 농장 수 — 국내 시군구·해외 나라).')
open(r, 'w', encoding='utf-8').write(m)
lv = M + r'\listeria-livestock-join-1103.md'
m = open(lv, encoding='utf-8').read()
anchor = '\n관련 [[listeria-1102-final-integrated-plan]], [[listeria-production-unit-inputs]].'
assert m.count(anchor) == 1
m = m.replace(anchor, f'\n- **11-04 밤 8**: 낙농장 = 젖 짜는 가축(젖소·산양·염소·면양)이 들어간 농장 수(사용자 {QD}) — 국내 1,360곳(젖소 831·염소 351·산양 172·면양 9, 겹침 있음), 한우·육우는 뺌. 해외는 데이터 팀 국경선으로 나라별 수(다른 세션이 데이터 팀용 프롬프트 줌 — 결과 `해외_나라별_축사수.csv`), 0개·안 덮는 나라 NaN 권고.\n' + anchor)
open(lv, 'w', encoding='utf-8').write(m)
print('ok')
