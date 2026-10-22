# -*- coding: utf-8 -*-
"""리스테리아 모델 성과 Word 문서(3차) 만들기.

- 양식 '모델 성과 발표 예시.docx' 를 열어 본문만 비우고(스타일·글머리표 정의·설정은 그대로) 새 내용을 넣는다.
- 양식 서식: 굴림, 머리 줄 굵게·가운데, 값은 왼쪽 정렬,
  '[ 데이터 기간 ]'·'[전체 모델 요약 성능]'·'[각 모델별 자세한 성능]' 꼬리표, 글머리표(numId 1·2).
- 표 테두리: 회색 격자(808080, sz 4)·머리 줄 음영(E7E6E6)·칸 여백 위아래 20·좌우 30, 칸 간격 없음
  (2026-10-22 14:42 저장본에 넣은 서식을 옮김. 양식 원래 서식은 테두리 없음·칸 간격 15·칸 여백 15).
- 수치는 work/tables.json(compute_tables.py 결과)만 쓴다. 변수 표 문구는 경복대양식 sections4 4.3 표·부록 기록표·표[11]
  과 work3/facts_features.md·facts_data.md 에서 옮겼다(아래 VAR_ROWS 주석).
- 칸 폭은 굴림 글꼴(C:\\Windows\\Fonts\\gulim.ttc)로 글자 폭을 재서 줄바꿈 없이 들어가게 정한다.
사용: build_docx.py [설정 json(글자 크기·여백·출력 경로 등, 없으면 기본)]
"""
import json
import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm
from PIL import ImageFont

TEMPLATE = r'C:\Users\NT751\Desktop\천영진\모델 성과 발표 예시.docx'
OUTDIR = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook\보고서\모델성과'
WORK = os.path.join(OUTDIR, 'work')
OUT = os.path.join(OUTDIR, '리스테리아_모델성과_3차.docx')
FONT = '굴림'

CFG = {
    'margin_lr_cm': 1.0, 'margin_top_cm': 1.6, 'margin_bottom_cm': 1.4,
    'title_pt': 14, 'part_pt': 12.5, 'label_pt': 11, 'text_pt': 9.5, 'table_pt': None, 'var_table_pt': 8.5,
    'table_pt_try': [9.0, 8.5, 8.0],
    'pad_pt': 4.0, 'first_col_extra_pt': 2.0,
    'label_before_pt': 6, 'label_after_pt': 2, 'text_after_pt': 2, 'var_row_gap_pt': 1.5,
    'cell_mar_tb_dxa': 20, 'cell_mar_lr_dxa': 30,
    'border_sz': 4, 'border_color': '808080', 'header_fill': 'E7E6E6',
    'out': OUT,
}
if len(sys.argv) > 1:
    CFG.update(json.load(open(sys.argv[1], encoding='utf-8-sig')))

T = json.load(open(os.path.join(WORK, 'tables.json'), encoding='utf-8'))
PAGE_W_CM = 21.0  # A4(양식과 같음)
BODY_W_CM = PAGE_W_CM - 2 * CFG['margin_lr_cm']

# ---------------------------------------------------------------- 글자 폭 재기(굴림 실제 글꼴)
_GULIM = ImageFont.truetype(r'C:\Windows\Fonts\gulim.ttc', 1000, index=0)
assert _GULIM.getname()[0] == 'Gulim'


def text_pt(s, size, bold=False):
    # Word 실측(calibrate_width.py, 8.5pt): 굴림 글자 폭은 PIL 과 같고(굵게도 같음), 빈칸만 0.333em 이 아니라 0.5em 으로 놓임
    s = str(s)
    w = _GULIM.getlength(s) / 1000 * size + s.count(' ') * (0.5 - 0.333) * size
    return w * 1.01


def cell_lines(v):
    return v if isinstance(v, list) else [v]


def auto_widths_cm(tables, size, pad_pt=None):
    """여러 표(같은 칸 구성)의 내용으로 칸마다 줄바꿈 없이 들어갈 폭(cm). tables = [(header, rows, bold_rows), ...]"""
    pad_pt = CFG['pad_pt'] if pad_pt is None else pad_pt
    ncol = len(tables[0][0])
    best = [0.0] * ncol
    for header, rows, bold_rows in tables:
        for ci in range(ncol):
            for line in cell_lines(header[ci]):
                best[ci] = max(best[ci], text_pt(line, size, True))
            for ri, r in enumerate(rows):
                for line in cell_lines(r[ci]):
                    best[ci] = max(best[ci], text_pt(line, size, ri in bold_rows))
    return [(w + pad_pt) / 72 * 2.54 for w in best]


# ---------------------------------------------------------------- 글자·문단 도우미
def font_run(run, size, bold=False):
    rPr = run._element.get_or_add_rPr()
    rf = rPr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rPr.insert(0, rf)
    for a in ('w:ascii', 'w:eastAsia', 'w:hAnsi', 'w:cs'):
        rf.set(qn(a), FONT)
    run.font.size = Pt(size)
    run.font.bold = bool(bold)
    run.font.cs_bold = bool(bold)
    kern = rPr.find(qn('w:kern'))
    if kern is None:
        kern = OxmlElement('w:kern')
        rPr.append(kern)
    kern.set(qn('w:val'), '0')
    szcs = rPr.find(qn('w:szCs'))
    if szcs is None:
        szcs = OxmlElement('w:szCs')
        rPr.append(szcs)
    szcs.set(qn('w:val'), str(int(round(size * 2))))


NBH = '\u2011'  # 글 안의 이 글자는 Word 의 줄이 바뀌지 않는 하이픈(w:noBreakHyphen)으로 넣음(예: 'PR-' / 'AUC' 끊김 막기)


def add_run_text(p, t):
    if NBH not in t:
        return p.add_run(t)
    r = p.add_run()
    for k, part in enumerate(t.split(NBH)):
        if k:
            r._r.append(OxmlElement('w:noBreakHyphen'))
        if part:
            e = OxmlElement('w:t')
            e.text = part
            if part != part.strip():
                e.set(qn('xml:space'), 'preserve')
            r._r.append(e)
    return r


def para(doc, text='', size=None, bold=False, before=0, after=None, align='left', numid=None,
         page_break_before=False, keep_next=False, runs=None):
    size = size or CFG['text_pt']
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(CFG['text_after_pt'] if after is None else after)
    pf.line_spacing = 1.0
    pf.alignment = {'left': WD_ALIGN_PARAGRAPH.LEFT, 'center': WD_ALIGN_PARAGRAPH.CENTER,
                    'right': WD_ALIGN_PARAGRAPH.RIGHT}[align]
    pf.keep_with_next = keep_next
    if page_break_before:
        pf.page_break_before = True
    if numid is not None:
        numPr = p._p.get_or_add_pPr().get_or_add_numPr()
        numPr.get_or_add_ilvl().val = 0
        numPr.get_or_add_numId().val = numid
    for t, b in (runs or [(text, bold)]):
        r = add_run_text(p, t)
        font_run(r, size, b)
    return p


def part_title(doc, text, page_break_before=False):
    return para(doc, text, size=CFG['part_pt'], bold=True, before=0 if page_break_before else 4, after=2,
                keep_next=True, page_break_before=page_break_before)


def label(doc, text):
    """양식의 '[ 데이터 기간 ]' 같은 꼬리표 문단."""
    return para(doc, text, size=CFG['label_pt'], before=CFG['label_before_pt'], after=CFG['label_after_pt'],
                keep_next=True)


def bullet(doc, text, numid):
    return para(doc, text, size=CFG['label_pt'], before=CFG['label_before_pt'] / 2, after=CFG['label_after_pt'],
                numid=numid, keep_next=True)


def note(doc, text):
    return para(doc, text, size=CFG['text_pt'], after=CFG['text_after_pt'], keep_next=True)


# ---------------------------------------------------------------- 표 도우미(양식 표 모양)
def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


def make_table(doc, header, rows, widths_cm, size=None, bold_rows=(), row_gap_pt=0.0):
    size = size or CFG['table_pt']
    ncol = len(header)
    assert len(widths_cm) == ncol
    assert sum(widths_cm) <= BODY_W_CM + 1e-6, (sum(widths_cm), BODY_W_CM, header)
    tw = [int(round(w / 2.54 * 1440)) for w in widths_cm]
    table = doc.add_table(rows=1 + len(rows), cols=ncol)
    tbl = table._tbl
    # 표 속성: 회색 격자 테두리, 칸 간격 없음, 칸 여백 위아래 20·좌우 30 + 고정 폭
    tblPr = tbl.tblPr
    for ch in list(tblPr):
        tblPr.remove(ch)
    tblPr.append(_el('w:tblW', **{'w:w': sum(tw), 'w:type': 'dxa'}))
    borders = _el('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        borders.append(_el(f'w:{side}', **{'w:val': 'single', 'w:sz': CFG['border_sz'], 'w:space': 0,
                                           'w:color': CFG['border_color']}))
    tblPr.append(borders)
    tblPr.append(_el('w:tblLayout', **{'w:type': 'fixed'}))
    mar = _el('w:tblCellMar')
    for side in ('top', 'left', 'bottom', 'right'):
        mw = CFG['cell_mar_tb_dxa'] if side in ('top', 'bottom') else CFG['cell_mar_lr_dxa']
        mar.append(_el(f'w:{side}', **{'w:w': mw, 'w:type': 'dxa'}))
    tblPr.append(mar)
    tblPr.append(_el('w:tblLook', **{'w:val': '04A0', 'w:firstRow': 1, 'w:lastRow': 0, 'w:firstColumn': 1,
                                     'w:lastColumn': 0, 'w:noHBand': 0, 'w:noVBand': 1}))
    for gc, w in zip(tbl.tblGrid.findall(qn('w:gridCol')), tw):
        gc.set(qn('w:w'), str(w))

    for ri, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(_el('w:cantSplit'))
        if ri == 0:
            trPr.append(_el('w:tblHeader'))
        vals = header if ri == 0 else rows[ri - 1]
        is_bold_row = ri == 0 or (ri - 1) in bold_rows
        for ci, cell in enumerate(row.cells):
            tcPr = cell._tc.get_or_add_tcPr()
            for ch in list(tcPr):
                tcPr.remove(ch)
            tcPr.append(_el('w:tcW', **{'w:w': tw[ci], 'w:type': 'dxa'}))
            if ri == 0:  # 머리 줄 음영
                tcPr.append(_el('w:shd', **{'w:val': 'clear', 'w:color': 'auto', 'w:fill': CFG['header_fill']}))
            tcPr.append(_el('w:vAlign', **{'w:val': 'center'}))
            p = cell.paragraphs[0]
            pf = p.paragraph_format
            pf.space_before = Pt(row_gap_pt if ri else 0)
            pf.space_after = Pt(row_gap_pt if ri else 0)
            pf.line_spacing = 1.0
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else WD_ALIGN_PARAGRAPH.LEFT
            if ri < len(table.rows) - 1:
                pf.keep_with_next = True
            for k, part in enumerate(cell_lines(vals[ci])):
                if k:
                    r = p.add_run()
                    r.add_break()
                    font_run(r, size, is_bold_row)
                r = p.add_run(str(part))
                font_run(r, size, is_bold_row)
    return table


# ---------------------------------------------------------------- 표 내용
SUM_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'tp', 'fn', 'fp', 'tn', 'n']
SUM_HEAD = ['Accuracy', 'Precision', 'Recall', 'F1', ['ROC-', 'AUC'], ['PR-', 'AUC'], 'TP', 'FN', 'FP', 'TN', 'N']
WPR_HEAD = ['구간 가중', 'PR-AUC']
H_HEAD = ['예측 거리', 'accuracy', 'precision', 'recall', 'f1', 'roc-auc', 'pr-auc', 'tp', 'fn', 'fp', 'tn', 'n']


def sum_rows(lst, extra_keys=()):
    return [[d['label']] + [d[k] for k in SUM_KEYS] + [d[k] for k in extra_keys] for d in lst]


def h_rows(lst):
    return [[d['label']] + [d[k] for k in SUM_KEYS] for d in lst]


# 3부 변수 표(구분 | 변수 | 출처 데이터 파일 | 컬럼 수 | 쓸 수 있는 시점)
# 근거: sections4/14_II4_특성공학.md 4.3 정리 표(칸 수)·표[11](외부 지표 원자료 파일), sections4/30_부록_입력변수기록표.md
# (쓸 수 있는 시점), work3/facts_features.md 14절(원자료·DB 표 이름), work3/facts_data.md 1.1·1.2절(시료 파일·기상 DB 표).
# 리스트 값 = 칸 안 줄바꿈.
VAR_ROWS = [
    ['기본', '시도·시군구·품목 2단계·품목 3단계(범주형)',
     ['식품안전정보원 제공 시료 자료', '(listeria_data.parquet)'], '범주형 4', '고정'],
    ['', 'h(며칠 뒤, 1~60)', '-', '1', '예측할 때'],
    ['', '기상 요약(18개 변수마다 10일 구간 평균 8개 컬럼 + window 평균·표준편차·최소·최대·기울기 5개 컬럼)',
     ['DB 표 api_data_info 의 kma_asos_daily_info(ASOS)·',
      'nas_agri_weather_daily_info(농업기상)·',
      'keco_airkorea_fine_dust_info(에어코리아)'],
     '234', 'T−80 ~ T−1(매일)'],
    ['외부 수집', '표준강수지수 SPI3·SPI6(최근 3·6개월 강수가 평년보다 많은지, window 요약)',
     ['CLM_SPI_DD_합본_20140101_20261006.csv', '(기상청 기상자료개방포털)'], '26', 'T−80 ~ T−1(매일)'],
    ['', '토양 pH·유기물·유효인산(지난 3년 검정 중앙값)', 'DB 표 api_data_info.soil_ph_info', '3',
     '검정일 T−1,095 ~ T−1'],
    ['', '식품 수분활성도·pH·염도(품목 3단계 대표값) + 수분활성도 표기 종류(범주형)',
     # 컬럼 수 '3 + 범주형 1' 은 두 줄로(한 줄이면 컬럼 수 칸이 넓어져 변수 칸 줄바꿈이 늘고 3부 ① 이 한 쪽을 넘음)
     ['DB 표 ba_data.t_aw_cl +', '통합_수분활성도_수분함량_식품pH_염도_최종.xlsx'], ['3 +', '범주형 1'], '고정표'],
    ['', '친환경 인증 비율',
     ['친환경_시도별통계_농산물_20251231.csv +', 'KOSIS_시군구_논밭면적_2008_2025_20251231.csv'], '1',
     '자료연도 + 2년 1월 1일'],
    ['', '지하수 정기수질검사 적합률', ['지하수조사연보_2014.zip ~', '지하수조사연보_2025.zip'], '1', '연보 발행일'],
    ['', 'HACCP 지정 비율',
     # 가운데 줄이 칸 수 '1' 과 같은 높이에 놓이므로 짧은 줄을 가운데에 둠(긴 줄 끝 '+' 가 '+ 1' 로 읽히지 않게)
     ['15058303_식약처_HACCP지정현황_20261019 폴더', '+ I0580 폴더 JSON 40개(식품안전나라)',
      '+ 15033311_인증원_HACCP인증업체_20261019 폴더'],
     '1', 'T−1(2017-08-03부터)'],
    ['', '생장 범위(최적·한계 범위 안 여부·거리)',
     ['예측인자 정리 ppt.jpg(식품안전정보원 제공)', '+ 위 기상·토양·식품 값'], '19', '기상·토양·식품과 같음'],
    ['파생', '이동평균 차감 기상(그날 값 − 앞 30일 평균, 19개 변수)', '기상 요약과 같음', '57', 'T−110 ~ T−1(매일)'],
    ['', '생장식 지수(온도 생장 요인·최대 생장 속도·생장 확률·강수 검출 확률)', '리스테리아 생장식.xlsx + 기상·식품 값', '16',
     # 한 줄이면 칸 폭을 넘어 범위 중간('T−82' / '~ T−3')에서 끊기므로 두 줄로
     ['T−80 ~ T−1', '(강수 T−82 ~ T−3)']],
    ['', '과거 검출 이력(5년·1년 스무딩 검출률, 5년 로그 검사 수)', '식품안전정보원 제공 시료 자료', '9', 'train 끝 날 값'],
    ['', '1년 로그 검사 수', '식품안전정보원 제공 시료 자료', '3', 'train 끝 날 값'],
    ['', '1년 검출률 불확실성', '식품안전정보원 제공 시료 자료', '3', 'train 끝 날 값'],
    ['', 'WOE(시도·시군구·품목 2·3단계별 검출·불검출 분포 차이, 최종 모델에서 제외)', '식품안전정보원 제공 시료 자료', '4',
     'train 끝 날 값'],
    ['', '제품명 과거 유형(같은 뜻 단어 23종의 과거 시료 비율·검출률)', '시료 자료의 제품명(prduct_nm)', '92',
     'train 끝 날 값'],
    ['', '그날 제품명 유형(23종 있음 여부, 운영 때 그날 제품명 입력 전제)', '시료 자료의 제품명(prduct_nm)', '23', 'd'],
    ['', '그날 제품명 TF-IDF(문자·단어 n-gram, 운영 때 그날 제품명 입력 전제)', '시료 자료의 제품명(prduct_nm)', '500', 'd'],
    ['', '이상치 점수(AE·VAE, 불검출만 학습한 모델의 복원 오차)', '위 입력(정형 입력·기상 시퀀스 80일)', '4', '입력과 같음'],
    ['합계', '기본 숫자형 235 + 범주형 4, 외부 수집 숫자형 54 + 범주형 1, 파생 숫자형 711 → 숫자형 1,000 + 범주형 5'
             '(최종 모델은 WOE 4개 컬럼을 제외한 숫자형 996 + 범주형 5)', '', '', ''],
]
VAR_HEAD = ['구분', '변수', '출처 데이터 파일', '컬럼 수', '쓸 수 있는 시점']

# 데이터 기간 표: tables.json(compute_tables.py)의 구간·기간 글자를 문서 용어로 바꿔 씀(숫자·날짜는 그대로)
DP_NAME = {'겹 1': 'fold 1', '겹 2': 'fold 2', '겹 3': 'fold 3', '시험 1년': 'test 1년', '전체': '전체'}


def dp_period(s):
    """'학습 A~B → 맞혀 봄 C~D' → 'train A~B → validation C~D'(다른 꼴은 그대로)."""
    if not s.startswith('학습 '):
        return s
    tr, va = s[len('학습 '):].split(' → 맞혀 봄 ')
    return f'train {tr} → validation {va}'


def build():
    doc = Document(TEMPLATE)
    body = doc.element.body
    for ch in list(body):
        if ch.tag != qn('w:sectPr'):
            body.remove(ch)
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(CFG['margin_lr_cm'])
    sec.top_margin = Cm(CFG['margin_top_cm'])
    sec.bottom_margin = Cm(CFG['margin_bottom_cm'])

    dp = T['data_period']
    p1, p2, p3 = T['part1_summary'], T['part2_summary'], T['part3_summary']

    # ----- 표 내용 미리 만들기(칸 폭을 함께 재려고)
    dp_head = ['구간', '기간', '전체 / 검출', '검출률']
    dp_rows = [[DP_NAME[d['구간']], dp_period(d['기간']), d['전체/검출'], d['검출률']] for d in dp]
    dp_bold = (len(dp) - 1,)

    s1_head = ['기상 window'] + SUM_HEAD + [WPR_HEAD]
    s1_rows = sum_rows(p1, ('wpr_auc',))
    s1_bold = tuple(i for i, d in enumerate(p1) if d['chosen'])
    s2_head = ['모델'] + SUM_HEAD + [WPR_HEAD]
    s2_rows = sum_rows(p2, ('wpr_auc',))
    for r, d in zip(s2_rows, p2):
        a, b = d['label'].split('(', 1)
        r[0] = [a, '(' + b]  # 예: ['BiLSTM', '(기상 시퀀스 90일)'] — 이름 칸 = 모델(입력 window)
    s3_head = ['모델'] + SUM_HEAD + [['구간', '가중', 'PR-AUC'], ['검사', '축소율']]
    s3_rows = sum_rows(p3, ('wpr_auc', 'inspect_reduction'))
    names3 = [['1차 베이스라인', '(기상 18개)', '교차검증'], ['변수 전부 +', '튜닝 18번 후보', '교차검증'],
              ['최종 모델', '(WOE 제외)', '교차검증'], ['최종 모델', '(WOE 제외)', 'test 1년']]
    for r, nm in zip(s3_rows, names3):
        r[0] = nm
    assert [d['kind'] for d in p3] == ['교차검증', '교차검증', '교차검증', '시험 1년']

    h_tables = [h_rows(T['part1_h'])] + [h_rows(T['part2_h'][d['label']]) for d in p2] + [h_rows(T['part3_h_test'])]

    # ----- 칸 폭: 1·2부 요약 표는 같은 폭(첫 칸만 각자), h 별 표 6개도 같은 폭.
    #       표 글자 크기는 후보(9 → 8.5 → 8pt) 가운데 모든 표가 본문 폭에 들어가는 가장 큰 값(모든 수치 표 같은 크기)
    def widths(tsz):
        w12 = auto_widths_cm([(s1_head, s1_rows, s1_bold), (s2_head, s2_rows, ())], tsz)
        xf = CFG.get('first_col_extra_pt', 0) / 72 * 2.54  # 여러 줄 이름 칸 덧폭(실측 보정)
        w3 = auto_widths_cm([(s3_head, s3_rows, ())], tsz)
        return {
            '1부 요약': [auto_widths_cm([(s1_head, s1_rows, s1_bold)], tsz)[0]] + w12[1:],
            '2부 요약': [auto_widths_cm([(s2_head, s2_rows, ())], tsz)[0] + xf] + w12[1:],
            '3부 요약': [w3[0] + xf] + w3[1:],
            'h 별': auto_widths_cm([(H_HEAD, r, ()) for r in h_tables], tsz),
            '데이터 기간': auto_widths_cm([(dp_head, dp_rows, dp_bold)], tsz),
        }

    tsz = CFG['table_pt']
    if tsz is None:
        for cand in CFG['table_pt_try']:
            if all(sum(w) <= BODY_W_CM for w in widths(cand).values()):
                tsz = cand
                break
        assert tsz is not None, '표가 본문 폭에 들어가지 않음'
    CFG['table_pt'] = tsz
    W = widths(tsz)
    print(f'  표 글자 {tsz}pt')
    for nm, w in W.items():
        print(f'  칸 폭 {nm}: 합 {sum(w):.2f}cm / 본문 {BODY_W_CM:.2f}cm  {[round(x, 2) for x in w]}')
    w1, w2, w3, wh, wdp = W['1부 요약'], W['2부 요약'], W['3부 요약'], W['h 별'], W['데이터 기간']

    # ---------------- 제목
    para(doc, '리스테리아 검출 예측 모델 성과(3차 실험)', size=CFG['title_pt'], bold=True, after=4)

    # ================= 1부
    part_title(doc, '1부. 1차 베이스라인 window 비교(LIMS + 기상, CatBoost 기상 요약)')
    note(doc, '입력 = 식품안전정보원 제공 시료 자료(식품의약품안전처 LIMS 검사 시료)의 시도·시군구·품목 2·3단계 + '
              'h(며칠 뒤를 맞히는지, 1~60일) + 시군구 기상 19개 변수의 예측하는 날 전 10~100일(window) 요약, '
              '하이퍼파라미터는 기본값. 80일을 채택한 뒤 일산화탄소를 제외해 기상 18개로 1차 베이스라인을 확정함.')

    label(doc, '[ 데이터 기간 ]')
    note(doc, '교차검증 3개 fold = train 기간을 1년씩 늘려 가며 다음 1년을 validation 으로 쓴 세 번의 평균'
              '(test 1년은 최종 모델만 평가). 행 = 시군구 × 품목 2·3단계 × 날짜, fold 의 ‘전체 / 검출’은 train 행. '
              'validation 1년·test 1년은 예측하는 날이 7월 1일~다음 해 5월 1일인 경우만 평가(60일 뒤가 6월 30일 안에 들게).')
    make_table(doc, dp_head, dp_rows, wdp, bold_rows=dp_bold)

    label(doc, '[전체 모델 요약 성능]')
    bullet(doc, '모델 60일간 예측 평균 값(교차검증 3개 fold 평균)', numid=1)
    note(doc, '값은 h 1~60 평균이고 TP(검출을 검출로 판정)·FN(놓친 검출)·FP(오탐, 불검출을 검출로 판정)·'
              'TN(불검출을 불검출로 판정)·N(validation 행)은 h 하나당 평균 건수. Accuracy·Precision·Recall·F1·TP·FN·FP·TN 은 '
              'fold 마다 정한 threshold(Recall 0.80 이상인 값 중 Precision 이 가장 큰 값)에서 계산함.')
    note(doc, '구간 가중 PR-AUC = h 1~20·21~40·41~60 PR-AUC 평균에 0.5·0.3·0.2 를 곱해 더한 값(window·모델을 고른 기준).')
    make_table(doc, s1_head, s1_rows, w1, bold_rows=s1_bold)

    label(doc, '[각 모델별 자세한 성능]')
    bullet(doc, '모델명 : CatBoost(기상 요약 80일, 채택) — h 10일 단위 평균(교차검증 3개 fold)', numid=2)
    make_table(doc, H_HEAD, h_tables[0], wh)

    # ================= 2부
    part_title(doc, '2부. 모델별 베이스라인 비교', page_break_before=True)
    note(doc, '70개 설정(트리 3종 × window 10개 × 기상 입력 2가지(요약·일별 원값) + BiLSTM × window 10개) 중 '
              '알고리즘마다 구간 가중 PR-AUC 가 가장 높았던 입력·window(데이터 기간·점수 계산은 위와 같음). '
              'BiLSTM = 기상 일별 값(시퀀스)을 앞뒤로 읽는 신경망, 나머지 셋은 부스팅 트리 모델.')
    label(doc, '[전체 모델 요약 성능]')
    bullet(doc, '모델 60일간 예측 평균 값(교차검증 3개 fold 평균)', numid=1)
    make_table(doc, s2_head, s2_rows, w2)

    label(doc, '[각 모델별 자세한 성능]')
    for d, rows in zip(p2, h_tables[1:5]):
        bullet(doc, f"모델명 : {d['label']} — h 10일 단위 평균(교차검증 3개 fold)", numid=2)
        make_table(doc, H_HEAD, rows, wh)

    # ================= 3부 ① 데이터·변수
    part_title(doc, '3부. 추가 데이터·파생변수 모델 성능 ① 사용한 데이터 및 변수', page_break_before=True)
    note(doc, '1차 베이스라인 입력(기본)에 외부 수집 자료와 파생변수를 추가함. T = 예측하는 날, '
              'd = 맞히려는 날(T + h). ‘train 끝 날 값’ = train 끝(6월 30일)까지 자료로 만든 값을 다음 1년 동안 씀(train 행은 '
              'T 전날까지 자료). 결측은 채우지 않음(친환경 통계에 행이 없는 시군구만 0).')
    vsz = CFG['var_table_pt']
    wv = auto_widths_cm([(VAR_HEAD, VAR_ROWS[:-1], ())], vsz)
    # '변수'·'쓸 수 있는 시점' 칸은 줄바꿈 허용: 출처·칸 수·구분은 잰 폭, 나머지를 두 칸에 나눔
    rest = BODY_W_CM - (wv[0] + wv[2] + wv[3])
    w_when = min(wv[4], max(text_pt('검정일 T−1,095 ~ T−1', vsz) + CFG['pad_pt'], 0) / 72 * 2.54)
    wv = [wv[0], rest - w_when, wv[2], wv[3], w_when]
    print(f'  칸 폭 변수 표: {[round(x, 2) for x in wv]} 합 {sum(wv):.2f}cm')
    vt = make_table(doc, VAR_HEAD, VAR_ROWS, wv, size=vsz, bold_rows=(len(VAR_ROWS) - 1,),
                    row_gap_pt=CFG['var_row_gap_pt'])
    # 합계 줄: 변수~쓸 수 있는 시점 칸을 하나로 합침
    last = vt.rows[-1]
    merged = last.cells[1].merge(last.cells[4])
    tcW = merged._tc.get_or_add_tcPr().find(qn('w:tcW'))
    tcW.set(qn('w:w'), str(sum(int(round(w / 2.54 * 1440)) for w in wv[1:])))
    tcW.set(qn('w:type'), 'dxa')
    for p in merged.paragraphs[1:]:
        if not p.text:
            p._p.getparent().remove(p._p)

    # ================= 3부 ② 모델·성능
    part_title(doc, '3부. 추가 데이터·파생변수 모델 성능 ② 모델 및 성능', page_break_before=True)
    ft = T['final_train']
    note(doc, '모두 CatBoost·기상 요약 80일. 변수 전부를 넣어 Optuna(하이퍼파라미터 자동 탐색)로 30회 튜닝 → 구간 가중 '
              f'PR{NBH}AUC 상위 5개 중 같은 Recall 에서 오탐(FP)이 가장 적은 18번 후보 → 그 하이퍼파라미터로 변수를 하나씩 제외해 보고 '
              '점수가 오른 WOE 만 제외 → '
              f"train 전체(2015-04-07~2025-06-30, {ft['rows']:,}행·검출 {ft['pos']:,})로 다시 학습해 test 1년으로 한 번 평가.")
    note(doc, '입력 컬럼 수: 1차 베이스라인 239개, 변수 전부 1,005개, 최종 1,001개(WOE 4개 제외). '
              f"검사축소율 = 1 − (TP + FP + FN) ÷ N(검사에서 제외해도 되는 비율). test threshold = 교차검증 3개 fold 의 "
              f"예측을 합쳐 정한 {T['final_threshold']} 그대로.")
    label(doc, '[전체 모델 요약 성능]')
    bullet(doc, '모델 60일간 예측 평균 값', numid=1)
    make_table(doc, s3_head, s3_rows, w3)

    label(doc, '[각 모델별 자세한 성능]')
    bullet(doc, '모델명 : 최종 모델(WOE 제외) — test 1년, h 10일 단위 평균', numid=2)
    make_table(doc, H_HEAD, h_tables[5], wh)

    doc.core_properties.title = '리스테리아 모델 성과(3차)'
    doc.core_properties.author = ''
    doc.core_properties.last_modified_by = ''
    doc.save(CFG['out'])
    print('saved', CFG['out'])


if __name__ == '__main__':
    build()
