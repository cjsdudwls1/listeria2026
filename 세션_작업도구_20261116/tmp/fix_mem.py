p = r'C:\Users\NT751\.claude\projects\C--Users-NT751-Desktop----\memory\listeria-1102-final-integrated-plan.md'
t = open(p, encoding='utf-8').read()
lines = t.split('\n')
idx = [i for i, l in enumerate(lines) if l.startswith('- **11-04 밤 3(')]
assert len(idx) == 1
lines[idx[0]] = ('- **11-04 밤 3(계획서 1,481줄 → 간결체로 다시 씀, 백업 job tmp `plan_before_concise.md`)**: '
                 '결정 29 확정 = 교수님 방식(사용자 "어차피 많은 다운사이징 방식 중 하나니까 교수님 방식으로 ㄱㄱ 하자") — '
                 "fold 마다 원 칼럼(더미 전) Theil's U 계산·기록 → 1위 칼럼(교수님 자료 = 제품명)에서 검출 0건 값의 행을 train 에서만 뺌, "
                 '검출 1건 이상은 검출만 있어도 남김(교수님 코드), validation·test 전부, 1위가 다른 fold 는 그 칼럼 판도'
                 '(사용자 "그 방식으로도 실험 해봐야함"), 단계 4 train 행 h 1~60 전부, 3단계·운영 칼럼 판 안 돌림. '
                 '교수님 원문 줄: 153·171·179(섞인 것만)·181·227·1360("그런 건 없을 거라고"). '
                 '낙농장 L1: 국내 = 새 엑셀 `Documents\\Amaranth10\\축사 현황 데이터(8.24).xlsx`(19,098곳, 모집 지역 약 28 시군구뿐, '
                 '젖소 포함 831곳)로 우리가 시군구별 수 셈(낙농장 = 젖소 농장 권고), 해외 = 데이터 팀에 원천(이름·기준일·수록 나라) 먼저 묻고 '
                 '나라 전체면 나라별 수, 아니면 NaN. SPI 를 예보 강수로 계산할지는 사용자 질문 — 권고 = T−1 유지.')
open(p, 'w', encoding='utf-8').write('\n'.join(lines))
print('fixed')
