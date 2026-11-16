# 지하수조사연보(글자 층 있는 2014~2023년판) '시군구별 지하수 수질검사 현황' 표를 글자 그대로 옮김(읽기만).
# 입력: gw_dump_text.py 가 만든 <연보YYYY_글자.txt>
# 사용: python -I gw_parse_text.py <글자txt폴더> <출력csv> <판1> [판2 ...]
# 출력 칸: 판, 자료연도, PDF쪽, 인쇄쪽, 순번, 원문_시도, 원문_시군구, 칸이름들(원문 글자 그대로), 칸수확인
import sys, os, re, csv

src, out = sys.argv[1], sys.argv[2]
eds = [int(x) for x in sys.argv[3:]]
SIDO = {'전국', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남',
        '전북', '전남', '경북', '경남', '제주'}
NUM = re.compile(r'^-?[\d,]+(\.\d+)?$|^-$')

# 판별 칸 이름(표 머리글 그대로)
COLS4 = {
    2014: ['합계', '적합', '부적합', '미기재'], 2015: ['합계', '적합', '부적합', '미기재'],
    2016: ['합계', '적합', '부적합', '미기재'], 2017: ['합계', '적합', '부적합', '적합률(%)'],
    2018: ['합계', '적합', '부적합', '적합률(%)'], 2019: ['합계', '적합', '부적합', '적합률(%)'],
    2020: ['합계', '적합', '부적합', '적합률(%)'], 2021: ['합계', '적합', '부적합', '적합률(%)'],
    2022: ['합계', '적합', '부적합', '적합률(%)'],
}
COLS12 = ['검사건수_총계', '검사건수_음용', '검사건수_비음용', '적합_총계', '적합_음용', '적합_비음용',
          '부적합_총계', '부적합_음용', '부적합_비음용', '적합율(%)_총계', '적합율(%)_음용', '적합율(%)_비음용']

rows_out = []
maxc = 12
for ed in eds:
    cols = COLS12 if ed >= 2023 else COLS4[ed]
    n = len(cols)
    lines = open(os.path.join(src, f'연보{ed}_글자.txt'), encoding='utf-8').read().split('\n')
    toks = []  # (pdf쪽, 인쇄쪽, 글자)
    page = None; printed = None
    for ln in lines:
        if ln.startswith('---- p'):
            page = int(ln[6:]); printed = None
            continue
        s = ln.strip()
        if not s or s == '　':
            toks.append((page, printed, '|SEP|'))
            continue
        m = re.match(r'^-\s*(\d+)\s*-$', s) or re.match(r'^(\d+)\s*/\s*제\s*4\s*장', s) or re.match(r'^4\.1 .*/\s*(\d+)$', s) \
            or re.match(r'^(\d+)\s*/\s*제\s*3\s*장', s) or re.match(r'^3\.1.*/\s*(\d+)$', s)
        if m:
            printed = int(m.group(1))
            # 같은 쪽 앞 토큰에도 인쇄쪽 채움
            toks = [(p, (printed if (p == page and q is None) else q), t) for (p, q, t) in toks]
            continue
        toks.append((page, printed, s))
    i = 0; seq = 0
    prev_sido = None
    while i < len(toks):
        p, q, t = toks[i]
        # 시도 칸이 비어 있는 행: 빈칸 다음에 이름 1개 + 숫자 n개가 정확히 이어질 때만
        if t == '|SEP|' and prev_sido and i + 1 < len(toks) and toks[i + 1][2] not in SIDO \
                and toks[i + 1][2] != '|SEP|' and not NUM.match(toks[i + 1][2]):
            j = i + 2; vals = []
            while j < len(toks) and NUM.match(toks[j][2]) and len(vals) < n:
                vals.append(toks[j][2]); j += 1
            nxt = toks[j][2] if j < len(toks) else '|SEP|'
            if len(vals) == n and (nxt in SIDO or nxt == '|SEP|'):
                seq += 1
                r = {'판': ed, '자료연도': ed - 1, 'PDF쪽': toks[i + 1][0], '인쇄쪽': toks[i + 1][1], '순번': seq,
                     '원문_시도': '', '원문_시군구': toks[i + 1][2]}
                for c, v in zip(cols, vals):
                    r[c] = v
                r['칸수확인'] = f'ok(시도 칸 비어 있음, 앞 행 시도={prev_sido})'
                rows_out.append(r)
                i = j
                continue
        if t in SIDO and i + 1 < len(toks) and not NUM.match(toks[i + 1][2]) and toks[i + 1][2] not in SIDO \
                and toks[i + 1][2] != '|SEP|':
            prev_sido = t
            name = toks[i + 1][2]
            vals = []
            j = i + 2
            while j < len(toks) and NUM.match(toks[j][2]) and len(vals) < n:
                vals.append(toks[j][2]); j += 1
            if len(vals) == n:
                seq += 1
                r = {'판': ed, '자료연도': ed - 1, 'PDF쪽': p, '인쇄쪽': q, '순번': seq, '원문_시도': t, '원문_시군구': name}
                for c, v in zip(cols, vals):
                    r[c] = v
                r['칸수확인'] = 'ok'
                rows_out.append(r)
                i = j
                continue
            elif len(vals) > 0:
                seq += 1
                r = {'판': ed, '자료연도': ed - 1, 'PDF쪽': p, '인쇄쪽': q, '순번': seq, '원문_시도': t, '원문_시군구': name,
                     '칸수확인': f'칸 {len(vals)}개만 읽힘: ' + ' '.join(vals)}
                rows_out.append(r)
                i = j
                continue
        i += 1

allcols = ['판', '자료연도', 'PDF쪽', '인쇄쪽', '순번', '원문_시도', '원문_시군구']
for ed in eds:
    for c in (COLS12 if ed >= 2023 else COLS4[ed]):
        if c not in allcols:
            allcols.append(c)
allcols.append('칸수확인')
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=allcols)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)
print('rows', len(rows_out), out)
