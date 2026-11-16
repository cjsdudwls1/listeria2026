# HACCP 구간표를 실험에 붙일 때 결측이 되는 단위 × 날짜 행 비율(실험 세션 요청 2026-10-19).
#  단위 × 날짜 = 시료_시군구.parquet 의 (sgg_code, train_prdlst_code('코드없음' 채움), train_prdlst_code_2('코드없음' 채움), mnfctur_de 날짜) 고유 행
#  (실험 코드 exp\1_베이스라인_lims기상\src\data.py build_unit_rows 와 같은 키). 검출 = 그 행 시료 중 judge_result '검출' 1건 이상.
#  기간: 학습 d ≤ 2024-06-30, 검증 2024-07-01 ~ 2025-06-30, 시험 2025-07-01 ~ 2026-06-30.
#  시험 기간 judge_result 는 쓰지 않음: 읽은 직후 시험 기간(과 기간 밖) 행의 judge_result 를 지우고 행 수만 셈.
#  (1) 예측일 T = d − h < 사용시작일 2017-08-03 → 결측. h=1: d < 2017-08-04, h=60: d < 2017-10-02.
#  (2) train_prdlst_code 가 업종묶음(축산물가공 뺀 판)의 코드 목록 어디에도 없음 → 결측.
# 출력: 중간\외부지표\_haccp_결측비율.txt. 실행: <리스테리아ML\.venv-exp2>\python.exe -I haccp_missing_rates.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, pyarrow.parquet as pq
import haccp_induty_groups as IG

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
SAMPLE = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋\시료_시군구.parquet'
OUT = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\중간\외부지표\_haccp_결측비율.txt'
L = []
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); L.append(s)
def pct(a, b):
    return f'{a:,}/{b:,} ({(a / b * 100 if b else 0):.2f}%)'

t = pq.read_table(SAMPLE, columns=['sgg_code', 'train_prdlst_code', 'train_prdlst_code_2', 'mnfctur_de', 'judge_result']).to_pandas()
d = t.mnfctur_de.values.astype('datetime64[D]')
split = np.full(len(t), '기간 밖', dtype=object)
split[d <= np.datetime64('2024-06-30')] = '학습'
split[(d >= np.datetime64('2024-07-01')) & (d <= np.datetime64('2025-06-30'))] = '검증'
split[(d >= np.datetime64('2025-07-01')) & (d <= np.datetime64('2026-06-30'))] = '시험'
jr = t.judge_result.astype(object).to_numpy()
jr[~np.isin(split, ['학습', '검증'])] = None          # 시험·기간 밖 judge_result 는 여기서 지움(쓰지 않음)
del t['judge_result']
s = pd.DataFrame({
    'sgg_code': t.sgg_code.astype(str),
    'p2': t.train_prdlst_code.astype(object).where(t.train_prdlst_code.notna(), '코드없음'),
    'p3': t.train_prdlst_code_2.astype(object).where(t.train_prdlst_code_2.notna(), '코드없음'),
    'd': d, 'split': split,
    'pos': pd.array([None if x is None else (x == '검출') for x in jr], dtype='boolean'),
})
P('# HACCP 구간표 결측 비율 (단위 × 날짜 행)')
P('시료 행', f'{len(s):,}', '/ mnfctur_de 빈 값', int(pd.isna(s.d).sum()))
u = s.groupby(['sgg_code', 'p2', 'p3', 'd'], sort=False, observed=True).agg(split=('split', 'first'), pos=('pos', 'max')).reset_index()
P('단위 × 날짜 행', f'{len(u):,}', '/ 기간별:', u.split.value_counts().to_dict())
for sp in ['학습', '검증']:
    v = u[u.split == sp]
    P(f'  {sp}: 행 {len(v):,}, 검출 행 {int(v.pos.sum()):,}')
P('  시험: 행', f'{int((u.split == "시험").sum()):,}', '(judge_result 안 씀)')
P('  (대조) 실험 config.json: train 222,793 / 양성 851, val 34,249 / 양성 122, test 34,770')

P('\n## (1) 예측일 T = d − h 가 사용시작일 2017-08-03 전이라 결측')
for h, cut in [(1, '2017-08-04'), (60, '2017-10-02')]:
    m = u.d < np.datetime64(cut)
    P(f'h={h} (d < {cut}):')
    for sp in ['학습', '검증', '시험']:
        v = u[u.split == sp]; mm = m[u.split == sp]
        line = f'  {sp}: {pct(int(mm.sum()), len(v))}'
        if sp != '시험':
            line += f' / 검출 행 중 {pct(int((mm & v.pos.fillna(False)).sum()), int(v.pos.sum()))}'
        P(line)
v = u[u.split == '학습']
P('학습 행 d 범위:', v.d.min(), '~', v.d.max())
P('학습 행 연도별(참고):', pd.Series(pd.to_datetime(v.d).dt.year).value_counts().sort_index().to_dict())

P('\n## (2) 2단계 코드가 업종묶음(축산물가공 뺀 판) 코드 목록에 없어 결측')
gc = IG.group_codes(IG.our_pairs())
keep = {g: c for g, c in gc.items() if g != IG.G_LIVE}
ok_codes = {c for g, v in keep.items() for c, _ in v}
P('코드 있는 묶음(축산물가공 뺌):', {g: len(v) for g, v in keep.items() if v}, '/ 코드 수', len(ok_codes))
u['nocode'] = ~u.p2.isin(ok_codes)
for sp in ['학습', '검증', '시험']:
    v = u[u.split == sp]
    line = f'  {sp}: {pct(int(v.nocode.sum()), len(v))}'
    if sp != '시험':
        line += f' / 검출 행 중 {pct(int((v.nocode & v.pos.fillna(False)).sum()), int(v.pos.sum()))}'
    P(line)
nm = IG.our_pairs().set_index('train_prdlst_code').train_prdlst_nm.to_dict()
for sp in ['학습', '검증', '시험']:
    v = u[(u.split == sp) & u.nocode]
    top = v.p2.value_counts().head(10)
    P(f'  {sp} 결측 행의 2단계 코드 상위 10:', {f'{k}({nm.get(k, "")})': int(n) for k, n in top.items()})
for sp in ['학습', '검증']:
    v = u[(u.split == sp) & u.nocode & u.pos.fillna(False)]
    P(f'  {sp} 결측 검출 행의 2단계 코드:', {f'{k}({nm.get(k, "")})': int(n) for k, n in v.p2.value_counts().items()})

P('\n## (1)+(2) 둘 중 하나로 결측(참고)')
for h, cut in [(1, '2017-08-04'), (60, '2017-10-02')]:
    m = (u.d < np.datetime64(cut)) | u.nocode
    for sp in ['학습', '검증', '시험']:
        v = u[u.split == sp]; mm = m[u.split == sp]
        line = f'  h={h} {sp}: {pct(int(mm.sum()), len(v))}'
        if sp != '시험':
            line += f' / 검출 행 중 {pct(int((mm & v.pos.fillna(False)).sum()), int(v.pos.sum()))}'
        P(line)
open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('저장:', OUT)
