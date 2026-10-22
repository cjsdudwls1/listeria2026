# -*- coding: utf-8 -*-
"""
경복대 양식 HWP 조립기 — 리스테리아 위해요소의 검출 예측모델 개발 연구 보고서
=============================================================================

쓰는 법('py' = 파이썬 3.14, pyhwpx·pymupdf 설치됨)
  py build_kb_hwp.py                                   sections2/ 원고(순서.txt 순서)로 out/ 에 만든다
  py build_kb_hwp.py --sections <원고 폴더> --out <결과 폴더>
  py build_kb_hwp.py --no-com                          한글(COM) 없이 HWPX 까지만 만든다(서식 점검용)
  py build_kb_hwp.py --gap 0.25 --fig-min 0.7          빈 자리 메우기를 더 세게(쪽 수 줄이기, 기본 0.40·0.75)
  시험: py build_kb_hwp.py --sections work/시험원고 --out out/시험
  결과: <결과 폴더>/리스테리아_예측모델_개발_연구보고서.hwp, 같은 이름 .pdf, png/p01.png …(쪽마다 110dpi),
        _build/리스테리아_예측모델_개발_연구보고서.hwpx(중간 파일), 점검 결과는 화면에 찍음(점검 1~8)
  결과 hwp·pdf 가 한글·PDF 보기에 열려 있으면 덮어쓸 수 없어 시작 전에 멈춤(닫고 다시).

입력(원고)
  <원고 폴더>/순서.txt 에 적힌 순서대로, 그 폴더에 실제로 있는 md 만 붙인다(없는 파일은 건너뛰고 알림).
  순서.txt 가 없으면 폴더의 *.md 를 이름 순으로 붙인다.
  줄 규칙(양식_가이드.md):
    '# Ⅰ. …'  부문 제목: 앞에서 쪽 나눔, 부문 제목만 한 쪽(다음 요소도 새 쪽에서 시작) — 원본 p3·p27 모양
    '## 1. …'  대제목 / '### 1.1 …' 중제목('####' 이하는 중제목으로 처리하고 알림)
    '□ ' 소제목 줄 / '○ ' 1단계 / '- ' 2단계 / '• ' 3단계 (기호는 문단 글머리표로 찍히므로 글에서 뗌)
    그 밖 줄 = 일반 문단, 빈 줄 = 문단 구분(빈 문단을 만들지 않음 — 원본도 간격은 문단 모양으로 띄움)
    파이프 표 + 바로 아래 '[표] 제목' -> 표 아래 가운데 '[표 n] 제목'(n 은 한글 자동 번호)
      정렬 줄 '---:' = 오른쪽(숫자), ':---' = 왼쪽, ':---:'·'---' = 가운데(원본 표 관례 — 여러 줄로 접히는 칸도
      가운데, 원본 [표 5]·[표 21]). 머리 행은 늘 가운데·굵게
      열 폭: 낱말이 글자 사이에서 끊기지 않게 하면서 표 어림 높이가 가장 낮아지게 자동으로 나눔(col_widths)
      칸 안 '<br>' = 같은 칸 안 줄 나눔(문단), '\\|' = 글자 '|'
      칸 글의 '~' 는 '∼'(U+223C)로 바꿔 넣음 — 표 글꼴의 '~' 가 위로 뜬 '˜' 처럼 찍힘(cell_text). 머리 행 칸 글이
      6자 이하(띄어쓰기 빼고)면 그 칸 폭은 머리 글이 한 줄에 들어가는 폭 이상('칸 수' 세로 접힘 막기)
    '![제목](figs/파일.png)' -> 그림(쪽 폭 안에 맞춤) + 그림 아래 가운데 '[그림 n] 제목'
      그림 경로는 md 파일 폴더, 그 위 폴더, 이 스크립트 폴더 순으로 찾는다(png 만)
      파일이 아직 없으면 멈추지 않고 빨간 점선 상자 '{{그림 준비 중}} + 경로' 자리 그림을 넣고 알림(점검 8 에 목록)
    '{{…}}' 자리표시는 글자 그대로 빨간 글자로 넣는다(본문·표 칸·캡션 어디서나)
    '`…`' 의 역따옴표와 '**…**' 의 별표는 떼고 글만 넣는다, '<!-- … -->' 주석 줄은 넣지 않는다
    '<!-- 쪽 나눔 -->' 한 줄 = 다음 요소를 새 쪽에서 시작(필요할 때만)

서식(원본 template/kb.hwpx = 경복대학교 2차 보고서(전처리 수정 후).hwp 를 HWPX 로 바꾼 것)
  Contents/header.xml(글꼴·글자 모양·문단 모양·테두리)·쪽 설정(secPr)·표지 구역(section0)은 원본 그대로 쓰고
  본문 구역(Contents/section1.xml)만 새로 만든다. 요소별 ID 는 work/kb_스타일표.md 0절 표 그대로(STYLE·TABLE·PICTURE).
  시험(out/시험/)에서 원본 kb.pdf 와 좌표를 맞춰 봄: 제목·□○-• 글머리·둘째 줄 들여쓰기·줄 간격·쪽 번호·표지·부문 제목 위치,
  표 행 높이·칸 여백, 표·그림 제목 간격이 원본과 같은 값으로 찍힘.
  원본과 다른 점(보이는 서식은 같고 쪽 나눔 동작·색만 다름 — header.xml 끝에 복제본을 새 ID 로 붙임)
    - 제목('1.'·'1.1'·□) 문단 모양 복제본에 '다음 문단과 함께'(keepWithNext) 켬: 쪽 끝에 제목만 남지 않게
    - 본문(○·-·•·일반) 문단 모양 복제본에 '외톨이줄 보호'(widowOrphan) 켬
    - 표 칸 왼쪽 정렬(원고 ':---')용으로 pp6 을 복제해 정렬만 LEFT
    - 빨간 자리표시 글자 모양: 쓰는 글자 모양(c1·c7·c9·c10·c15·c25·c31·c60)을 복제해 글자색만 #FF0000
    - 한 쪽의 60% 보다 길어 보이는 표는 '글자처럼 취급'을 끄고(원본 [표 15]·[표 18] 모양: 놓는 문단 pp63 + 표 뒤 빈 문단)
      행 경계에서 쪽을 넘기며 머리 행을 반복함(원본처럼 모두 글자처럼 취급하면 한 쪽보다 긴 표가 잘려 찍힘)
    - 부문 제목의 'Ⅰ.'·'Ⅱ.'(로마 숫자 글자)는 원본처럼 라틴 대문자 'I.'·'II.' 로 바꿔 넣음(PART_LATIN_NUMERAL)
    - 원본의 표지 뒤 빈 쪽(p2)·'(작성 중)' 쪽(p4)은 만들지 않음 — 부문 제목 쪽이 2쪽부터
    - 표·그림이 글 없이 바로 이어지면(그림 아래 바로 표 등) 사이에 빈 문단 하나(쪽을 넘기는 표 뒤 빈 문단과 같은 모양)
    - 표·그림이 통째로 다음 쪽으로 밀려 앞 쪽 아래가 본문 높이의 40%(--gap) 넘게 비면 다시 변환하면서
      표는 쪽 경계에서 나누는 표로, 그림은 75%(--fig-min)까지 줄여 앞 쪽에 넣음(find_gaps, 최대 MAX_CONVERT 번 변환)
  줄 나눔(시험으로 확인, work/scripts/builder_test/exp_break.py): 이 한글(2020)에서 breakNonLatinWord 는
    "BREAK_WORD" = 한글 어절 단위, "KEEP_WORD" = 글자 단위로 동작함. 원본 본문(○·-·• = KEEP_WORD)은 글자 단위로
    줄이 바뀌고(원본 PDF 도 '반/응하는' 처럼 나뉨), 표 칸(pp6·pp7 = BREAK_WORD)은 어절 단위 — 둘 다 원본 그대로 둠.

방법(검증된 절차, 메모리 hwp-generation-via-pyhwpx-hwpx)
  1) section1.xml 생성(hp:linesegarray 넣지 않음 — 넣으면 여러 줄 문단이 한 줄로 겹쳐 찍힘), 표지는 원본 section0 에서
     제목 글만 바꾸고 linesegarray 를 뺌
  2) 그림은 BinData/imageN.png + content.hpf 목록 + hp:pic(원본 [그림 13] 틀)으로 HWPX 에 직접 넣음
  3) zip: mimetype 첫 항목·무압축, 원본 항목 순서 유지(원본 그림 21개는 뺌)
  4) pyhwpx(COM, 한컴오피스 2020)로 열어 SaveAs "HWP"·"PDF", pymupdf 로 쪽별 PNG(110dpi)
  5) 한글 COM 잠금: 보고서/hwp_build.lock 이 있으면 30초 간격으로 기다리고(최대 20분) 없으면 만들어 쓰고 끝나면 지움(finally).
     내용 = owner=[종합] 경복대 조립기 / started=시각 / pid=PID (build_report_hwp.py 와 같은 꼴). 이 조립기가 남긴 잠금인데
     그 PID 가 이미 끝났으면(강제 종료) 지우고 바로 잡음 — 다른 세션 잠금은 건드리지 않음(work/scripts/builder_test/test_lock.py)
     Quit 뒤에도 남은 한글은 이번에 띄운 PID 중 창 제목 없는 것만 종료(사용자 한글 창은 건드리지 않음)
  6) 자동 점검: 1 md 글이 PDF 에 다 들어갔는지, 2 표·그림 번호 차례, 3 표 칸 글이 다 있고 낱말이 줄 경계에서 끊기지
     않았는지(칸 잘림), 4 그림 수, 5 쪽 번호, 6 본문 영역 밖 글, 7 쪽 아래 큰 빈 자리, 8 남은 '{{…}}' 자리표시·없는 그림
"""
import argparse
import csv
import functools
import io
import os
import re
import struct
import subprocess
import sys
import time
import zipfile
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_HWPX = os.path.join(HERE, "template", "kb.hwpx")
SECTIONS_DIR = os.path.join(HERE, "sections2")                                 # 줄인 판 원고(10-19 17:40 규칙)
OUT_DIR = os.path.join(HERE, "out")
OUT_NAME = "리스테리아_예측모델_개발_연구보고서"
REPORT_TITLE = "리스테리아 위해요소 예측모델 개발 연구 보고서"
TEMPLATE_TITLE = "아폴라톡신 위해요소의 하이브리드 예측모델 개발 연구 보고서"   # 원본 표지 제목(원문 오타 그대로)
HWP_LOCK = os.path.join(os.path.dirname(HERE), "hwp_build.lock")              # 리스테리아ML2/notebook/보고서/hwp_build.lock
LOCK_OWNER = "[종합] 경복대 조립기"
LOCK_POLL_S = 30
LOCK_MAX_S = 20 * 60
PNG_DPI = 110

# ---------------------------------------------------------------------------
# 요소별 서식 ID (원본 header.xml 기준, work/kb_스타일표.md 0절)
#   pp = paraPrIDRef, st = styleIDRef, cp = charPrIDRef
# ---------------------------------------------------------------------------
STYLE = {
    "part": dict(pp=36, st=0, cp=31),    # 부문 제목: 가운데, HY견고딕 26pt (section1 #1)
    "h1": dict(pp=16, st=2, cp=15),      # '1.' 대제목: 휴먼명조 15pt 굵게 (#20)
    "h2": dict(pp=16, st=3, cp=15),      # '1.1' 중제목 (#21)
    "box": dict(pp=10, st=6, cp=9),      # □ 소제목 줄: 휴먼명조 13pt 굵게, 글머리표 3 (#22)
    "b1": dict(pp=9, st=7, cp=1),        # ○ 1단계: 휴먼명조 12pt, 글머리표 2 (#23)
    "b2": dict(pp=8, st=8, cp=7),        # - 2단계: 글머리표 1 (#24)
    "b3": dict(pp=12, st=9, cp=1),       # • 3단계: 글머리표 4 (#76)
    "plain": dict(pp=53, st=0, cp=1),    # 글머리 없는 일반 문단: ○ 와 같은 들여쓰기·간격 (#169 모양, 질문 3)
}
TABLE = dict(
    anchor_pp=6, anchor_st=0, anchor_cp=7,        # 표를 놓는 문단(가운데, 간격 0) — 원본 #79·#113
    split_anchor_pp=63, split_anchor_cp=65,       # 쪽을 넘기는 표를 놓는 문단 + 표 뒤 빈 문단 — 원본 [표 15] #155·#156
    tbl_bf=9, cell_bf=12, cell_st=11,             # 표 테두리 0.12mm / 칸 0.25mm 검정·흰 바탕 / 스타일 '표내용'
    head_cp=60, body_cp=25,                       # 머리 행 한양중고딕 10pt 굵게 / 본문 10pt
    pp_center=6, pp_right=7, pp_left=4,           # 칸 문단: 가운데 / 오른쪽(숫자) / 양쪽(긴 글, 원본 [표 5])
    cap_pp=6, cap_st=10, cap_cp=10,               # 표 제목 '[표 n] …'(표 아래): 한양중고딕 12pt (질문 1)
    cell_margin=(900, 900, 600, 600),             # 칸 안 여백 좌·우·위·아래 (원본 bf12 표)
    in_margin=(510, 510, 141, 141), out_margin=141,
    cap_gap=850, cap_width=8503,
    max_width=47411,                              # 원본 꽉 찬 표 폭([표 9]·[표 11])
    min_ratio=0.6,                                # 내용이 짧은 표도 이 비율보다 좁히지 않음(원본 가장 좁은 표 [표 8] 65%)
    row_h=1200,                                   # cellSz height(최소값, 한글이 글에 맞춰 늘림)
    split_ratio=0.6,                              # 어림 높이가 본문 높이의 이 비율보다 크면 쪽 경계에서 나누는 표로
)
PICTURE = dict(
    pp=6, st=0, cp=7,                             # 그림 놓는 문단(가운데) — 원본 [그림 4](#205)
    cap_pp=6, cap_st=10, cap_cp=10,               # '[그림 n] …'(그림 아래)
    max_w=45700,                                  # 원본 넓은 그림 폭(161mm)
    max_h=42000,                                  # 원본에서 가장 높은 그림(41,194 = 145mm) 정도 — 더 높으면 앞 쪽이 크게 빔
    default_dpi=96,
)
BODY_W = 48360                                    # 본문 폭(59528-5499-5669)
BODY_H = 71435                                    # 본문 높이(84188-2834-1417-4251-4251)
RED = "#FF0000"
RED_BASE_CPS = (1, 7, 9, 10, 15, 25, 31, 60)
PART_LATIN_NUMERAL = True     # 부문 제목 'Ⅰ.'·'Ⅱ.'(원고) -> 'I.'·'II.'(원본 글자, 질문 4)
PART_OWN_PAGE = False         # 사용자 10-19: 부문 제목만 있는 쪽을 없애고 부문 제목을 첫 절 위에 둠(원본은 한 쪽 통째)

# 표 칸 글 폭 어림용(한양중고딕 10pt, HWPUNIT). 이 조립기로 만든 PDF(Hancom PDF) 의 글자 위치로 잼
CELL_PAD_LR = 1800 + 150                          # 칸 좌우 여백 900+900 + 여유

# header.xml 확인 값: charPr id -> (크기pt, 굵게, 한글 글꼴)
EXPECT_CHAR = {
    1: (12.0, False, "휴먼명조"), 7: (12.0, False, "휴먼명조"), 9: (13.0, True, "휴먼명조"),
    10: (12.0, False, "한양중고딕"), 15: (15.0, True, "휴먼명조"), 25: (10.0, False, "한양중고딕"),
    31: (26.0, False, "HY견고딕"), 41: (21.0, True, "HY견고딕"), 60: (10.0, True, "한양중고딕"),
    65: (12.0, True, "함초롬돋움"),
}
EXPECT_PARA = {4: "JUSTIFY", 6: "CENTER", 7: "RIGHT", 8: "JUSTIFY", 9: "JUSTIFY", 10: "JUSTIFY",
               12: "JUSTIFY", 16: "JUSTIFY", 36: "CENTER", 53: "JUSTIFY", 63: "LEFT"}
EXPECT_STYLE = {0: "바탕글", 2: "2: 1. (대제목)", 3: "3: 1.1 (중제목)", 6: "6: □ (소제목)", 7: "7: ○ (내용)",
                8: "8: - (내용)", 9: "9: ∙ (내용)", 10: "그림,표 제목", 11: "표내용"}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 원고 읽기
# ---------------------------------------------------------------------------
class Block(object):
    def __init__(self, kind, src, text=None, table=None, fig=None):
        self.kind, self.src, self.text, self.table, self.fig = kind, src, text, table, fig


RE_HEAD = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
RE_IMG = re.compile(r"^!\[(.*?)\]\(\s*(.+?)\s*\)\s*$")
RE_TCAP = re.compile(r"^\[표\]\s*(.*)$")
RE_ALIGN_CELL = re.compile(r"^:?-{3,}:?$")
RE_PH = re.compile(r"\{\{[^{}]*\}\}")
RE_PAGEBREAK = re.compile(r"^<!--\s*(쪽\s*나눔|pagebreak)\s*-->$", re.I)
BULLETS = (("□", "box"), ("○", "b1"), ("- ", "b2"), ("* ", "b2"), ("•", "b3"))
WARN = []


def warn(msg):
    WARN.append(msg)
    log("  [알림] " + msg)


def read_order(sections_dir):
    """순서.txt 순서대로 있는 md 경로, 순서.txt 에 있으나 없는 파일 이름"""
    order = os.path.join(sections_dir, "순서.txt")
    files, missing = [], []
    if os.path.isfile(order):
        with open(order, encoding="utf-8-sig") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                m = re.match(r"^(.+?\.md)(\s|$)", s)
                if not m:
                    continue
                p = os.path.join(sections_dir, m.group(1))
                if os.path.isfile(p):
                    files.append(p)
                else:
                    missing.append(m.group(1))
    else:
        files = sorted(os.path.join(sections_dir, f) for f in os.listdir(sections_dir) if f.lower().endswith(".md"))
    return files, missing


def clean_inline(s):
    """역따옴표·굵게 표시를 떼고 탭은 띄어쓰기로"""
    s = s.replace("\t", " ").replace("`", "")
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    return s


def segments(s):
    """글 -> [(조각, 자리표시인가)]"""
    out, pos = [], 0
    for m in RE_PH.finditer(s):
        if m.start() > pos:
            out.append((s[pos:m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(s):
        out.append((s[pos:], False))
    return out


ROMAN = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V", "Ⅵ": "VI"}


def latin_numeral(text):
    """부문 제목 앞 로마 숫자 글자(Ⅰ U+2160 …)를 원본처럼 라틴 대문자('I.' 'II.')로.
    HY견고딕의 'Ⅰ' 글자는 가는 세로 막대로 찍혀 원본 'I.' 과 모양이 다름(시험 PDF 2쪽 확인)"""
    m = re.match(r"^\s*([ⅠⅡⅢⅣⅤⅥ])(\s*\.)", text)
    if m:
        return ROMAN[m.group(1)] + m.group(2) + text[m.end():]
    return text


def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.strip().replace("\\|", "|") for c in cells]


def cell_parts(text):
    """칸 글 -> 칸 안 문단 목록('<br>' 로 나눔)"""
    return [p.strip() for p in re.split(r"<br\s*/?>", text, flags=re.I)]


class Table(object):
    def __init__(self, header, rows, aligns, caption, src):
        self.header, self.rows, self.aligns, self.caption, self.src = header, rows, aligns, caption, src
        self.number = None


def make_table(tlines, caption, src):
    rows = [split_row(t) for _, t in tlines]
    header = rows[0]
    ncol = len(header)
    aligns = [None] * ncol
    body = rows[1:]
    if len(rows) > 1 and all(RE_ALIGN_CELL.match(c.replace(" ", "")) for c in rows[1] if c != "") and rows[1]:
        al = rows[1]
        for i in range(min(ncol, len(al))):
            c = al[i].replace(" ", "")
            if c.startswith(":") and c.endswith(":"):
                aligns[i] = "center"
            elif c.endswith(":"):
                aligns[i] = "right"
            elif c.startswith(":"):
                aligns[i] = "left"
        body = rows[2:]
    else:
        warn("%s:%d 표에 정렬 줄(|---|) 이 없음 — 첫 줄을 머리 행으로 씀" % src)
    fixed = []
    for k, r in enumerate(body):
        if len(r) < ncol:
            warn("%s:%d 표 %d번째 행 칸 수 %d < 머리 행 %d — 빈 칸으로 채움" % (src[0], src[1], k + 1, len(r), ncol))
            r = r + [""] * (ncol - len(r))
        elif len(r) > ncol:
            raise SystemExit("%s:%d 표 %d번째 행 칸 수 %d > 머리 행 %d: %r" % (src[0], src[1], k + 1, len(r), ncol, r))
        fixed.append(r)
    if ncol > 7:
        warn("%s:%d 표 열 %d개(가이드: 7개 이하)" % (src[0], src[1], ncol))
    if caption is None:
        warn("%s:%d 표 바로 아래 '[표] 제목' 줄이 없음 — 제목 없이 넣음" % src)
    header = [cell_text(c) for c in header]
    fixed = [[cell_text(c) for c in r] for r in fixed]
    return Table(header, fixed, aligns, clean_inline(caption) if caption else None, src)


TILDE_CELL = chr(0x223C)          # '∼'(KS X 1001 물결) — 표 칸 글꼴(한양중고딕)의 '~' 는 위로 뜬 '˜' 처럼 찍힘(눈 검수 10-19)
HEAD_NOWRAP_CHARS = 6             # 머리 행 칸 글(띄어쓰기 빼고)이 이 글자 수 이하면 한 줄에 다 들어가게('칸 수'·'누적 단계' 세로 접힘 막기)


def cell_text(c):
    """표 칸 글: clean_inline + '~' -> '∼'(자리표시 '{{…}}' 안은 그대로)"""
    c = clean_inline(c)
    return "".join(t if ph else t.replace("~", TILDE_CELL) for t, ph in segments(c))


def parse_md(path, base_dirs):
    name = os.path.basename(path)
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    blocks = []
    i, in_comment = 0, False
    while i < len(lines):
        s = lines[i].strip()
        src = (name, i + 1)
        if in_comment:
            if "-->" in s:
                in_comment = False
            i += 1
            continue
        if RE_PAGEBREAK.match(s):
            blocks.append(Block("pagebreak", src))
            i += 1
            continue
        if s.startswith("<!--"):
            if "-->" not in s:
                in_comment = True
            i += 1
            continue
        if not s:
            i += 1
            continue
        if s.startswith("|"):
            tl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tl.append((i + 1, lines[i].strip()))
                i += 1
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            cap = None
            if j < len(lines):
                m = RE_TCAP.match(lines[j].strip())
                if m:
                    cap = m.group(1).strip()
                    i = j + 1
            blocks.append(Block("table", src, table=make_table(tl, cap, src)))
            continue
        i += 1
        m = RE_HEAD.match(s)
        if m:
            lv = len(m.group(1))
            kind = {1: "part", 2: "h1", 3: "h2"}.get(lv, "h2")
            if lv > 3:
                warn("%s:%d '%s' 제목은 중제목(###) 모양으로 넣음" % (src[0], src[1], m.group(1)))
            text = clean_inline(m.group(2))
            if kind == "part" and PART_LATIN_NUMERAL:
                text = latin_numeral(text)
            blocks.append(Block(kind, src, text=text))
            continue
        m = RE_IMG.match(s)
        if m:
            title, rel = clean_inline(m.group(1).strip()), m.group(2)
            cands = [os.path.join(d, rel) for d in base_dirs(path)]
            hit = next((c for c in cands if os.path.isfile(c)), None)
            if not hit:     # 아직 안 그린 그림(결과 대기 등): 멈추지 않고 빨간 자리 그림으로 넣고 알림
                warn("%s:%d 그림 파일 없음: %s — 빨간 자리 그림('{{그림 준비 중}}')으로 넣음" % (src[0], src[1], rel))
                blocks.append(Block("fig", src, fig=dict(title=title, path=None, rel=rel)))
                continue
            if not hit.lower().endswith(".png"):
                raise SystemExit("%s:%d 그림은 png 만 넣을 수 있음: %s" % (src[0], src[1], rel))
            blocks.append(Block("fig", src, fig=dict(title=title, path=os.path.abspath(hit), rel=rel)))
            continue
        if RE_TCAP.match(s):
            warn("%s:%d 표 아래가 아닌 '[표]' 줄 — 일반 문단으로 넣음" % src)
            blocks.append(Block("plain", src, text=clean_inline(s)))
            continue
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", s):
            warn("%s:%d 가로줄 '%s' 는 넣지 않음" % (src[0], src[1], s))
            continue
        kind, text = "plain", s
        for mark, k in BULLETS:
            if s.startswith(mark):
                kind, text = k, s[len(mark):].strip()
                break
        if s.startswith(">"):
            text = s.lstrip(">").strip()
        blocks.append(Block(kind, src, text=clean_inline(text)))
    return blocks


def default_base_dirs(md_path):
    d = os.path.dirname(os.path.abspath(md_path))
    return [d, os.path.dirname(d), HERE]


# ---------------------------------------------------------------------------
# header.xml 확인·복제
# ---------------------------------------------------------------------------
def parse_header(xml):
    fonts = {}
    for fm in re.finditer(r'<hh:fontface lang="(\w+)"[^>]*>(.*?)</hh:fontface>', xml, re.S):
        fonts[fm.group(1)] = {int(m.group(1)): m.group(2) for m in re.finditer(r'<hh:font id="(\d+)" face="([^"]+)"', fm.group(2))}
    chars = {}
    for m in re.finditer(r'<hh:charPr id="(\d+)"(.*?)</hh:charPr>', xml, re.S):
        b = m.group(2)
        fr = re.search(r'<hh:fontRef hangul="(\d+)" latin="(\d+)"', b)
        chars[int(m.group(1))] = dict(size=int(re.search(r'height="(\d+)"', b).group(1)) / 100.0,
                                      bold="<hh:bold/>" in b, hangul=fonts["HANGUL"][int(fr.group(1))],
                                      color=re.search(r'textColor="([^"]+)"', b).group(1))
    paras = {}
    for m in re.finditer(r'<hh:paraPr id="(\d+)"(.*?)</hh:paraPr>', xml, re.S):
        paras[int(m.group(1))] = dict(align=re.search(r'<hh:align horizontal="(\w+)"', m.group(2)).group(1))
    fills = set(int(x) for x in re.findall(r'<hh:borderFill id="(\d+)"', xml))
    styles = {int(m.group(1)): m.group(2) for m in re.finditer(r'<hh:style id="(\d+)" type="\w+" name="([^"]*)"', xml)}
    return dict(chars=chars, paras=paras, fills=fills, styles=styles)


def check_header(h):
    bad = []
    for cid, (size, bold, face) in EXPECT_CHAR.items():
        c = h["chars"].get(cid)
        if not c or (c["size"], c["bold"], c["hangul"]) != (size, bold, face):
            bad.append("charPr %d: 기대 %s, 실제 %s" % (cid, (size, bold, face), c))
    for pid, al in EXPECT_PARA.items():
        p = h["paras"].get(pid)
        if not p or p["align"] != al:
            bad.append("paraPr %d: 기대 %s, 실제 %s" % (pid, al, p))
    for sid, nm in EXPECT_STYLE.items():
        if h["styles"].get(sid) != nm:
            bad.append("style %d: 기대 %r, 실제 %r" % (sid, nm, h["styles"].get(sid)))
    for bf in (TABLE["tbl_bf"], TABLE["cell_bf"]):
        if bf not in h["fills"]:
            bad.append("borderFill %d 없음" % bf)
    if bad:
        raise SystemExit("template/kb.hwpx 의 header.xml 이 예상과 다름:\n  " + "\n  ".join(bad))


class HeaderPatch(object):
    """header.xml 끝에 문단 모양·글자 모양 복제본을 새 ID 로 붙인다(원래 ID 의 다른 값은 그대로)"""

    def __init__(self, xml):
        self.xml = xml
        self.next_pp = max(int(x) for x in re.findall(r'<hh:paraPr id="(\d+)"', xml)) + 1
        self.next_cp = max(int(x) for x in re.findall(r'<hh:charPr id="(\d+)"', xml)) + 1
        self.new_pp, self.new_cp = [], []
        self.pp_map, self.cp_red = {}, {}

    def clone_para(self, pid, changes, tag):
        key = (pid, tag)
        if key in self.pp_map:
            return self.pp_map[key]
        blk = re.search(r'<hh:paraPr id="%d" .*?</hh:paraPr>' % pid, self.xml, re.S).group(0)
        blk = blk.replace('<hh:paraPr id="%d" ' % pid, '<hh:paraPr id="%d" ' % self.next_pp, 1)
        for attr, val in changes.items():
            blk, k = re.subn(r'\b%s="[^"]*"' % attr, '%s="%s"' % (attr, val), blk, count=1)
            if k != 1:
                raise SystemExit("paraPr %d 에 %s 설정이 없음" % (pid, attr))
        self.new_pp.append(blk)
        self.pp_map[key] = self.next_pp
        self.next_pp += 1
        return self.pp_map[key]

    def red(self, cid):
        if cid in self.cp_red:
            return self.cp_red[cid]
        blk = re.search(r'<hh:charPr id="%d" .*?</hh:charPr>' % cid, self.xml, re.S).group(0)
        blk = blk.replace('<hh:charPr id="%d" ' % cid, '<hh:charPr id="%d" ' % self.next_cp, 1)
        blk, k = re.subn(r'textColor="[^"]*"', 'textColor="%s"' % RED, blk, count=1)
        if k != 1:
            raise SystemExit("charPr %d 에 textColor 없음" % cid)
        self.new_cp.append(blk)
        self.cp_red[cid] = self.next_cp
        self.next_cp += 1
        return self.cp_red[cid]

    def result(self):
        x = self.xml
        if self.new_pp:
            x = x.replace("</hh:paraProperties>", "".join(self.new_pp) + "</hh:paraProperties>", 1)
            x = re.sub(r'<hh:paraProperties itemCnt="(\d+)"',
                       lambda m: '<hh:paraProperties itemCnt="%d"' % (int(m.group(1)) + len(self.new_pp)), x, count=1)
        if self.new_cp:
            x = x.replace("</hh:charProperties>", "".join(self.new_cp) + "</hh:charProperties>", 1)
            x = re.sub(r'<hh:charProperties itemCnt="(\d+)"',
                       lambda m: '<hh:charProperties itemCnt="%d"' % (int(m.group(1)) + len(self.new_cp)), x, count=1)
        return x


# ---------------------------------------------------------------------------
# 글 폭 어림(표 열 폭·표 높이 어림용) — 한양중고딕 10pt, HWPUNIT
# ---------------------------------------------------------------------------
# 이 조립기 PDF(실제 원고 204쪽)에서 같은 줄 이웃 글자 기준점 차이의 중앙값으로 잰 값(보통 글자 = 한글 PDF 글꼴 T9).
# 굵은 머리 행 글자는 숫자 552·한글 1056·'-' 576 으로 약 10% 넓음(work/scripts/builder_test/calib_width.py)
CHAR_W = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                  (636, 624, 696, 696, 648, 600, 756, 686, 252, 468, 636, 540, 804, 696, 744, 636, 756, 696, 636, 588,
                   684, 636, 900, 624, 612, 588)))
CHAR_W.update(zip("abcdefghijklmnopqrstuvwxyz",
                  (540, 528, 480, 528, 528, 252, 528, 528, 192, 192, 468, 192, 804, 528, 540, 540, 528, 312, 480, 252,
                   528, 480, 696, 468, 468, 483)))
CHAR_W.update({" ": 504, ",": 245, ".": 240, ":": 246, ";": 246, "'": 240, '"': 420, "(": 288, ")": 280, "[": 288,
               "]": 280, "{": 312, "}": 312, "%": 876, "-": 528, "_": 504, "~": 498, "/": 262, "\\": 288, "+": 516,
               "=": 517, "<": 588, ">": 588, "*": 372, "·": 384, "×": 618, "÷": 619})
BOLD_FACTOR = 1.10


def text_width(s, bold=False):
    w = 0
    for ch in s:
        if ch in CHAR_W:
            w += CHAR_W[ch]
        elif "0" <= ch <= "9":
            w += 504
        elif ch == ZWSP:
            continue
        elif ord(ch) < 0x2000:
            w += 600                                      # 그 밖 라틴 기호(잰 값 없음)
        else:
            w += 1000                                     # 한글(실측 996~1008)·한자·전각 기호·화살표
    return int(w * (BOLD_FACTOR if bold else 1.0))


LINE_H = 1600        # 표 칸 한 줄(한양중고딕 10pt x 줄 간격 160%)
ROW_PAD = 600        # 한 줄 행 높이 22pt(원본 PDF 실측) = 600 + 1600 x 1
WRAP_SAFETY = 1.03   # 글 폭 어림 여유


def word_pieces(word, bold=False):
    """낱말을 줄바꿈 자리(break_points)에서 나눈 조각들 — 칸보다 긴 낱말은 soft_breaks 가 이 자리에 폭 없는 빈칸을 넣음"""
    pts = sorted(break_points(word))
    out, prev = [], 0
    for p in pts:
        out.append(word[prev:p])
        prev = p
    out.append(word[prev:])
    return [x for x in out if x]


@functools.lru_cache(maxsize=None)
def para_lines(text, inner, bold):
    """칸 안 문단 하나가 칸 안 폭 inner 에서 몇 줄로 찍히는지 어림(한글 표 칸 = 어절 단위 줄 나눔 흉내).
    칸보다 긴 낱말은 줄바꿈 자리(조각)에서, 조각도 칸보다 길면 글자 사이에서 나뉨"""
    space = text_width(" ", bold)
    lines, cur = 1, 0.0
    for word in text.split():
        long_word = text_width(word, bold) * WRAP_SAFETY > inner
        pieces = word_pieces(word, bold) if long_word else [word]
        for k, piece in enumerate(pieces):
            w = text_width(piece, bold) * WRAP_SAFETY
            gap = space if (k == 0 and cur > 0) else 0.0
            if cur > 0 and cur + gap + w > inner:
                lines += 1
                cur, gap = 0.0, 0.0
            if w > inner:                                  # 조각도 칸보다 김: 글자 사이에서 잘림
                extra = int(w // inner)
                lines += extra
                cur = w - extra * inner
            else:
                cur += gap + w
    return lines


def cell_lines(text, width, bold=False):
    inner = max(1000, int(width) - CELL_PAD_LR)
    return sum(para_lines(part, inner, bold) for part in cell_parts(text))


def table_height(rows, widths):
    """표 몸통 어림 높이(HWPUNIT): 행마다 600 + 1600 x (그 행에서 가장 많은 줄 수)"""
    return sum(ROW_PAD + LINE_H * max(cell_lines(t, widths[c], ri == 0) for c, t in enumerate(row))
               for ri, row in enumerate(rows))


def col_widths(tb):
    """열 폭(합 = 본문 꽉 찬 표 폭 47,411, 짧은 표는 더 좁게).
    최소 폭 = 낱말이 글자 사이에서 끊기지 않는 폭(칸보다 긴 낱말은 줄바꿈 자리 조각 기준), 최대 폭 = 가장 긴 줄.
    - 최대 폭 합이 표 폭 안: 최대 폭 그대로(본문 폭의 min_ratio 보다 좁으면 그만큼 넓힘)
    - 최소 폭 합도 표 폭보다 큼: 넓은 열부터 같은 높이로 깎음(글자 사이에서 끊기는 열 수를 줄임)
    - 그 밖: 표 어림 높이(행마다 가장 많은 줄 수의 합)가 가장 낮아지게 — (가) 최소 폭에서 시작해 줄 수를 가장 많이
      줄이는 열에 차례로 폭을 더한 것, (나) 최소~최대 폭 사이를 비례로 나눈 것 둘 다 두 열 사이 폭 옮기기로 다듬고 낮은 쪽"""
    T = TABLE
    ncol = len(tb.header)
    rows = [tb.header] + tb.rows
    nrow = len(rows)
    W = T["max_width"]
    minw, maxw = [0.0] * ncol, [0.0] * ncol
    for ri, row in enumerate(rows):
        bold = ri == 0
        for c, txt in enumerate(row):
            for part in cell_parts(txt):
                maxw[c] = max(maxw[c], text_width(part, bold) * WRAP_SAFETY)
                for word in part.split():
                    minw[c] = max(minw[c], max(text_width(p, bold) for p in word_pieces(word, bold)) * WRAP_SAFETY)
                if ri == 0 and len(part.replace(" ", "")) <= HEAD_NOWRAP_CHARS:
                    minw[c] = max(minw[c], text_width(part, bold) * WRAP_SAFETY)
    minw = [int(m) + CELL_PAD_LR + 20 for m in minw]
    maxw = [max(int(m) + CELL_PAD_LR + 20, mn) for m, mn in zip(maxw, minw)]
    if sum(maxw) <= W:
        total = max(sum(maxw), int(W * T["min_ratio"]))
        extra = total - sum(maxw)
        w = [m + int(extra * m / float(sum(maxw))) for m in maxw]
        w[max(range(ncol), key=lambda i: w[i])] += total - sum(w)
        return w
    if sum(minw) >= W:
        lo, hi = 0.0, float(max(minw))
        for _ in range(60):
            mid = (lo + hi) / 2
            if sum(min(m, mid) for m in minw) > W:
                hi = mid
            else:
                lo = mid
        w = [int(min(m, lo)) for m in minw]
        w[max(range(ncol), key=lambda i: w[i])] += W - sum(w)
        return w

    cache = {}

    def col_lines(c, width):
        v = cache.get((c, width))
        if v is None:
            v = cache[(c, width)] = [cell_lines(rows[r][c], width, r == 0) for r in range(nrow)]
        return v

    def height(ws):
        cols = [col_lines(c, ws[c]) for c in range(ncol)]
        return sum(max(col[r] for col in cols) for r in range(nrow))

    step = max(100, W // 100)

    def greedy():
        w = list(minw)
        while W - sum(w) >= step:
            rest = W - sum(w)
            cols = [col_lines(c, w[c]) for c in range(ncol)]
            cur = sum(max(col[r] for col in cols) for r in range(nrow))
            best = None
            for c in range(ncol):
                if w[c] >= maxw[c]:
                    continue
                others = [max([cols[c2][r] for c2 in range(ncol) if c2 != c] or [1]) for r in range(nrow)]
                d = step
                while d <= rest:
                    nl = col_lines(c, w[c] + d)
                    gain = cur - sum(max(o, x) for o, x in zip(others, nl))
                    if gain > 0:
                        if best is None or gain / float(d) > best[0]:
                            best = (gain / float(d), c, d)
                        break
                    if w[c] + d >= maxw[c]:
                        break
                    d *= 2
            if best is None:
                break
            w[best[1]] += best[2]
        return spread(w)

    def proportional():
        rest = W - sum(minw)
        span = float(sum(maxw) - sum(minw))
        return spread([mn + int(rest * (mx - mn) / span) for mn, mx in zip(minw, maxw)])

    def spread(w):
        """남은 폭: 아직 한 줄에 다 안 들어가는 열에 먼저(모자란 만큼 비례), 없으면 폭 비율로"""
        rest = W - sum(w)
        if rest > 0:
            need = [max(0, mx - x) for mx, x in zip(maxw, w)]
            base = need if sum(need) > 0 else w
            w = [x + int(rest * b / float(sum(base))) for x, b in zip(w, base)]
        w = [int(x) for x in w]
        w[max(range(ncol), key=lambda i: w[i])] += W - sum(w)
        return w

    def refine(w):
        """두 열 사이에서 폭을 옮겨 어림 높이가 낮아지면 받아들임(최소 폭 아래로는 줄이지 않음)"""
        cur = height(w)
        for _ in range(12):
            better = False
            for i in range(ncol):
                for j in range(ncol):
                    if i == j:
                        continue
                    for d in (step, 2 * step, 4 * step, 8 * step):
                        if w[i] - d < minw[i]:
                            break
                        t = list(w)
                        t[i] -= d
                        t[j] += d
                        h = height(t)
                        if h < cur:
                            w, cur, better = t, h, True
                            break
            if not better:
                break
        return w, cur

    cands = [refine(greedy()), refine(proportional())]
    return min(cands, key=lambda x: x[1])[0]


ZWSP = chr(0x200B)                 # 폭 없는 빈칸(소스에 보이지 않는 글자를 두지 않으려고 chr 로 씀)
SEP_AFTER = "._/\\·,-~)]:;" + chr(0x223C)   # '∼'(표 칸 물결) 뒤도 줄바꿈 자리


def soft_breaks(text, inner, bold=False):
    """칸 안 폭(inner)보다 긴 낱말에만 폭 없는 빈칸(U+200B)을 넣어 한글이 그 자리에서 줄을 바꿀 수 있게 함:
    구분 글자('.' '_' '/' '\\' '·' ',' '-' '~' ')' ']' ':' ';') 뒤와 '(' '[' 앞(break_points). 보이지 않음(시험: exp_zwsp.py).
    넣지 않으면 한글이 긴 낱말을 아무 글자에서나 자름(예: 'staging_data.exp|ort_body_listeria')"""
    out = []
    for tok in re.split(r"(\s+)", text):
        if tok and not tok.isspace() and text_width(tok, bold) * WRAP_SAFETY > inner:
            pts = break_points(tok)
            tok = "".join((ZWSP if k in pts else "") + ch for k, ch in enumerate(tok))
        out.append(tok)
    return "".join(out)


def break_points(word):
    """soft_breaks 가 넣는 줄바꿈 자리(낱말 안 글자 위치 k = k 번째 글자 앞).
    숫자 사이의 구분 글자(1,234,567 · 2015-04-07 · 0.0356 · 10~100)와 파일 확장자 앞 '.'(…표.csv)에서는 나누지 않음"""
    pts = set()
    n = len(word)
    for k, ch in enumerate(word):
        if ch in SEP_AFTER and k < n - 1 and not (k > 0 and word[k - 1].isdigit() and word[k + 1].isdigit()):
            if ch == "." and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,7}[)\]]?", word[k + 1:]):
                continue
            pts.add(k + 1)
        if ch in "([" and k > 0:
            pts.add(k)
    return pts


# ---------------------------------------------------------------------------
# section XML 조각
# ---------------------------------------------------------------------------
def run_xml(cp, text):
    if text == "":
        return '<hp:run charPrIDRef="%d"/>' % cp
    return '<hp:run charPrIDRef="%d"><hp:t>%s</hp:t></hp:run>' % (cp, escape(text))


def runs_xml(text, cp, hp):
    segs = segments(text)
    out = [run_xml(hp.red(cp) if ph else cp, t) for t, ph in segs]
    if segs and segs[-1][1]:
        # 자리표시로 끝나는 문단: 글머리표(□○•)는 문단 끝 글자 모양을 따라 빨갛게 찍힘(시험 확인) -> 끝에 보통 글자 모양 빈 run
        out.append(run_xml(cp, ""))
    return "".join(out) if out else run_xml(cp, "")


def para_open(pp, st, page_break):
    return '<hp:p id="0" paraPrIDRef="%d" styleIDRef="%d" pageBreak="%d" columnBreak="0" merged="0">' % (
        pp, st, 1 if page_break else 0)


SUBLIST = ('<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="%s" linkListIDRef="0" '
           'linkListNextIDRef="0" textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">')


def caption_xml(kind, num, title, width, hp, pp, st, cp):
    """표·그림 아래 제목: '[표 ' + 자동 번호 + '] 제목' (원본 [표 21]·[그림 13] 모양)"""
    word, ntype = ("표", "TABLE") if kind == "table" else ("그림", "PICTURE")
    autonum = ('<hp:ctrl><hp:autoNum num="%d" numType="%s"><hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" '
               'suffixChar="" supscript="0"/></hp:autoNum></hp:ctrl>' % (num, ntype))
    body = '<hp:run charPrIDRef="%d"><hp:t>[%s </hp:t>%s<hp:t>] </hp:t></hp:run>' % (cp, word, autonum)
    body += runs_xml(title, cp, hp) if title else ""
    return ('<hp:caption side="BOTTOM" fullSz="0" width="%d" gap="%d" lastWidth="%d">' % (TABLE["cap_width"], TABLE["cap_gap"], width)
            + SUBLIST % "TOP" + para_open(pp, st, False) + body + "</hp:p></hp:subList></hp:caption>")


class Builder(object):
    def __init__(self, hp):
        self.hp = hp
        self.n_tbl = self.n_fig = 0
        self.z = 0
        self.images = []          # (binary id, 경로, 바이트)
        self.tables = []          # Table (번호 붙은 것)
        self.figs = []
        self.log_tables = []      # (번호, 제목, 열 폭 %, 어림 높이 비율, 방식)
        # 쪽 나눔·줄 나눔 동작만 다른 문단 모양 복제본
        self.pp_keep = {}
        for k in ("h1", "h2", "box"):
            pid = STYLE[k]["pp"]
            self.pp_keep[k] = hp.clone_para(pid, {"keepWithNext": "1"}, "keepWithNext")
        for k in ("b1", "b2", "b3", "plain"):
            pid = STYLE[k]["pp"]
            self.pp_keep[k] = hp.clone_para(pid, {"widowOrphan": "1"}, "widowOrphan")
        self.pp_keep["part"] = STYLE["part"]["pp"]
        # 표 칸: 원본 pp6(가운데)·pp7(오른쪽) 그대로 — 한글 어절 단위 줄 나눔(breakNonLatinWord="BREAK_WORD").
        # 왼쪽 정렬(원고 ':---')만 pp6 을 복제해 정렬만 LEFT 로(어절 단위 그대로).
        #   주의: 이 한글(2020)에서 breakNonLatinWord 는 BREAK_WORD = 어절 단위, KEEP_WORD = 글자 단위로 동작함
        #   (work/scripts/builder_test/exp_break.py 시험). 원본 본문 문단(pp8·9·12·53)은 KEEP_WORD = 글자 단위라 그대로 둠.
        self.cell_pp = {
            "center": TABLE["pp_center"],
            "right": TABLE["pp_right"],
            "left": hp.clone_para(TABLE["pp_center"], {"horizontal": "LEFT"}, "left"),
        }

    def next_z(self):
        self.z += 1
        return self.z

    # -- 문단
    def para(self, kind, text, page_break, first_runs=""):
        s = STYLE[kind]
        pp = self.pp_keep.get(kind, s["pp"])
        return para_open(pp, s["st"], page_break) + first_runs + runs_xml(text, s["cp"], self.hp) + "</hp:p>"

    # -- 표
    def table(self, tb, page_break, first_runs=""):
        T, hp = TABLE, self.hp
        if tb.caption is not None:
            self.n_tbl += 1
            tb.number = self.n_tbl
        self.tables.append(tb)
        ncol = len(tb.header)
        widths = col_widths(tb)
        width = sum(widths)
        # 열 정렬: 원고 정렬 줄을 따르고, 지정 없으면 원본 표 관례대로 가운데
        aligns = [a or "center" for a in tb.aligns]
        # 어림 높이: 행 = 600 + 1600 x 줄 수(한 줄 22pt, 원본 PDF 실측), 제목 = 간격 850 + 12pt x 160%
        est = 850 + 1920 + 2 * T["out_margin"] + table_height([tb.header] + tb.rows, widths)
        split = est > T["split_ratio"] * BODY_H or getattr(tb, "force_split", False)   # force_split: find_gaps
        trs = []
        for ri, row in enumerate([tb.header] + tb.rows):
            head = ri == 0
            tcs = []
            for c, txt in enumerate(row):
                if head:
                    pp, cp = self.cell_pp["center"], T["head_cp"]
                else:
                    pp, cp = self.cell_pp[aligns[c]], T["body_cp"]
                inner = max(1000, widths[c] - CELL_PAD_LR)
                paras = "".join(para_open(pp, T["cell_st"], False) + runs_xml(soft_breaks(part, inner, head), cp, hp)
                                + "</hp:p>" for part in cell_parts(txt))
                ml, mr, mt, mb = T["cell_margin"]
                tcs.append('<hp:tc name="" header="%d" hasMargin="1" protect="0" editable="0" dirty="0" borderFillIDRef="%d">'
                           % (1 if (head and split) else 0, T["cell_bf"])
                           + SUBLIST % "CENTER" + paras + "</hp:subList>"
                           + '<hp:cellAddr colAddr="%d" rowAddr="%d"/><hp:cellSpan colSpan="1" rowSpan="1"/>' % (c, ri)
                           + '<hp:cellSz width="%d" height="%d"/>' % (widths[c], T["row_h"])
                           + '<hp:cellMargin left="%d" right="%d" top="%d" bottom="%d"/></hp:tc>' % (ml, mr, mt, mb))
            trs.append("<hp:tr>%s</hp:tr>" % "".join(tcs))
        nrow = len(tb.rows) + 1
        om = T["out_margin"]
        il, ir, it, ib = T["in_margin"]
        if split:   # 글자처럼 취급하지 않음(원본 [표 15]·[표 18] 모양) + 행 경계에서 쪽 나눔 + 머리 행 반복, 단 가운데
            pos = ('<hp:pos treatAsChar="0" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
                   'vertRelTo="PARA" horzRelTo="COLUMN" vertAlign="TOP" horzAlign="CENTER" vertOffset="0" horzOffset="0"/>')
            pbreak = "TABLE"
            anchor_pp, anchor_cp = T["split_anchor_pp"], T["split_anchor_cp"]
        else:       # 원본처럼 글자처럼 취급(놓는 문단 가운데 정렬을 따름), 나누지 않음
            pos = ('<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
                   'vertRelTo="PARA" horzRelTo="PARA" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>')
            pbreak = "CELL"
            anchor_pp, anchor_cp = T["anchor_pp"], T["anchor_cp"]
        cap = caption_xml("table", tb.number, tb.caption, width, hp, T["cap_pp"], T["cap_st"], T["cap_cp"]) \
            if tb.caption is not None else ""
        tbl = ('<hp:tbl id="%d" zOrder="%d" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" lock="0" '
               'dropcapstyle="None" pageBreak="%s" repeatHeader="1" rowCnt="%d" colCnt="%d" cellSpacing="0" borderFillIDRef="%d" noAdjust="0">'
               % (1900000000 + len(self.tables), self.next_z(), pbreak, nrow, ncol, T["tbl_bf"])
               + '<hp:sz width="%d" widthRelTo="ABSOLUTE" height="%d" heightRelTo="ABSOLUTE" protect="0"/>' % (width, T["row_h"] * nrow)
               + pos
               + '<hp:outMargin left="%d" right="%d" top="%d" bottom="%d"/>' % (om, om, om, om)
               + cap
               + '<hp:inMargin left="%d" right="%d" top="%d" bottom="%d"/>' % (il, ir, it, ib)
               + "".join(trs) + "</hp:tbl>")
        tb.widths = widths
        tb.split = split
        self.log_tables.append((tb.number, tb.caption, [round(100.0 * x / T["max_width"], 1) for x in widths],
                                est / float(BODY_H), "쪽 경계에서 나눔" if split else "글자처럼 취급", aligns))
        xml = (para_open(anchor_pp, T["anchor_st"], page_break) + first_runs
               + '<hp:run charPrIDRef="%d">%s<hp:t/></hp:run></hp:p>' % (anchor_cp, tbl))
        if split:   # 원본처럼 글자처럼 취급 안 한 표 뒤에 빈 문단 하나(#156·#166): 표 제목과 다음 문단 사이 간격
            xml += para_open(anchor_pp, T["anchor_st"], False) + run_xml(anchor_cp, "") + "</hp:p>"
        return xml

    # -- 그림
    def figure(self, fig, page_break, first_runs=""):
        P, hp = PICTURE, self.hp
        self.n_fig += 1
        self.figs.append(fig)
        if fig.get("path"):
            data = open(fig["path"], "rb").read()
        else:                                              # 파일 없는 그림: 빨간 점선 상자 + '{{그림 준비 중}}' + 경로
            data = placeholder_png(fig["rel"])
        pw, ph, dpi = png_info(data, fig.get("path") or fig["rel"])
        bid = "image%d" % (len(self.images) + 1)
        self.images.append((bid, fig.get("path") or "(자리 그림) " + fig["rel"], data))
        ow, oh = pw * 75, ph * 75                         # 원본 규칙: orgSz = imgDim = 픽셀 x 75
        nat_w = int(round(pw * 7200.0 / (dpi or P["default_dpi"])))
        w = min(nat_w, P["max_w"])
        h = int(round(w * ph / float(pw)))
        if h > P["max_h"]:
            h = P["max_h"]
            w = int(round(h * pw / float(ph)))
        s = fig.get("scale", 1.0)                          # 앞 쪽 빈 자리에 넣으려고 줄인 비율(find_gaps)
        if s < 1.0:
            w, h = int(round(w * s)), int(round(h * s))
        fig["size_mm"] = (round(w / 7200.0 * 25.4, 1), round(h / 7200.0 * 25.4, 1))
        pid = 2000000000 + self.n_fig
        cap = caption_xml("fig", self.n_fig, fig["title"], w, hp, P["cap_pp"], P["cap_st"], P["cap_cp"])
        pic = ('<hp:pic id="%d" zOrder="%d" numberingType="PICTURE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" lock="0" '
               'dropcapstyle="None" href="" groupLevel="0" instid="%d" reverse="0">' % (pid, self.next_z(), pid - 1073741823)
               + '<hp:offset x="0" y="0"/><hp:orgSz width="%d" height="%d"/><hp:curSz width="%d" height="%d"/>' % (ow, oh, w, h)
               + '<hp:flip horizontal="0" vertical="0"/>'
               + '<hp:rotationInfo angle="0" centerX="%d" centerY="%d" rotateimage="1"/>' % (w // 2, h // 2)
               + '<hp:renderingInfo><hc:transMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
               + '<hc:scaMatrix e1="%.6f" e2="0" e3="0" e4="0" e5="%.6f" e6="0"/>' % (w / float(ow), h / float(oh))
               + '<hc:rotMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/></hp:renderingInfo>'
               + '<hp:imgRect><hc:pt0 x="0" y="0"/><hc:pt1 x="%d" y="0"/><hc:pt2 x="%d" y="%d"/><hc:pt3 x="0" y="%d"/></hp:imgRect>'
               % (ow, ow, oh, oh)
               + '<hp:imgClip left="0" right="%d" top="0" bottom="%d"/>' % (ow, oh)
               + '<hp:inMargin left="0" right="0" top="0" bottom="0"/>'
               + '<hp:imgDim dimwidth="%d" dimheight="%d"/>' % (ow, oh)
               + '<hc:img binaryItemIDRef="%s" bright="0" contrast="0" effect="REAL_PIC" alpha="0"/><hp:effects/>' % bid
               + '<hp:sz width="%d" widthRelTo="ABSOLUTE" height="%d" heightRelTo="ABSOLUTE" protect="0"/>' % (w, h)
               + '<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
                 'vertRelTo="PARA" horzRelTo="PARA" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
               + '<hp:outMargin left="0" right="0" top="0" bottom="0"/>'
               + cap + "</hp:pic>")
        return (para_open(P["pp"], P["st"], page_break) + first_runs
                + '<hp:run charPrIDRef="%d">%s<hp:t/></hp:run></hp:p>' % (P["cp"], pic))


PLACEHOLDER_FONTS = (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\gulim.ttc")


def placeholder_png(rel, w_mm=150.0, h_mm=45.0, dpi=150):
    """아직 없는 그림 자리: 빨간 점선 상자 안에 빨간 글 '{{그림 준비 중}}' + 원고에 적힌 경로(pymupdf 로 그림)"""
    import pymupdf
    w, h = w_mm / 25.4 * 72, h_mm / 25.4 * 72
    doc = pymupdf.open()
    pg = doc.new_page(width=w, height=h)
    pg.draw_rect(pymupdf.Rect(3, 3, w - 3, h - 3), color=(1, 0, 0), width=1.5, dashes="[6 4] 0")
    font = dict(fontname="korea")                                     # pymupdf 내장 한글 글꼴(맑은 고딕이 없을 때)
    for f in PLACEHOLDER_FONTS:
        if os.path.isfile(f):
            font = dict(fontname="kbph", fontfile=f)
            break
    msg = "{{그림 준비 중}}\n" + rel
    pg.insert_textbox(pymupdf.Rect(12, h / 2 - 24, w - 12, h - 8), msg, fontsize=15, color=(1, 0, 0), align=1, **font)
    pix = pg.get_pixmap(dpi=dpi)
    pix.set_dpi(dpi, dpi)
    data = pix.tobytes("png")
    doc.close()
    return data


def png_info(data, path):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("PNG 가 아님(그림은 png 만): %s" % path)
    w, h = struct.unpack(">II", data[16:24])
    dpi = None
    i = data.find(b"pHYs")
    if 0 < i < 4096:
        x, y, unit = struct.unpack(">IIB", data[i + 4:i + 13])
        if unit == 1 and x > 0:
            dpi = x * 0.0254
    return w, h, dpi


def build_section1(blocks, s1_orig, hp, forced=frozenset()):
    """forced: 새 쪽에서 시작할 블록 번호(쪽 끝에 머리 행만 남은 표를 다음 변환에서 넘길 때)"""
    head = re.search(r"^(.*?<hs:sec [^>]*>)", s1_orig, re.S).group(1)
    m = re.search(r'<hp:run charPrIDRef="\d+"><hp:ctrl><hp:colPr [^>]*/></hp:ctrl><hp:secPr .*?</hp:secPr></hp:run>'
                  r'<hp:run charPrIDRef="\d+"><hp:ctrl><hp:pageNum [^>]*/></hp:ctrl><hp:t/></hp:run>', s1_orig, re.S)
    if not m:
        raise SystemExit("원본 section1.xml 첫 문단(쪽 설정·쪽 번호) 모양이 예상과 다름")
    first_runs = m.group(0)
    b = Builder(hp)
    out = [head]
    pending_break = False
    k = -1
    prev = None                              # 바로 앞에 넣은 요소: (종류, 쪽을 넘기는 표인가)
    for bi, blk in enumerate(blocks):
        if blk.kind == "pagebreak":          # '<!-- 쪽 나눔 -->' 줄: 다음 요소를 새 쪽에서
            pending_break = k >= 0
            continue
        k += 1
        fr = first_runs if k == 0 else ""
        pb = pending_break or (bi in forced and k > 0)
        pending_break = False
        if blk.kind == "table":
            blk.table.block_index = bi
        if (blk.kind in ("table", "fig") and prev is not None and prev[0] in ("table", "fig") and not prev[1]
                and not pb):
            # 표·그림이 글 없이 바로 이어지면(그림 제목 바로 밑에 표 등) 사이에 빈 문단 하나 — 쪽을 넘기는 표 뒤
            # 빈 문단(원본 [표 15] #156)과 같은 모양·간격. 쪽을 넘기는 표는 이미 뒤에 빈 문단이 있어 넣지 않음
            out.append(para_open(TABLE["split_anchor_pp"], TABLE["anchor_st"], False)
                       + run_xml(TABLE["split_anchor_cp"], "") + "</hp:p>")
        if blk.kind == "part":
            pb = k > 0                       # 부문 제목은 새 쪽에서(구역 첫 문단이면 이미 새 쪽)
            pending_break = PART_OWN_PAGE    # True 면 다음 요소도 새 쪽에서 -> 부문 제목만 한 쪽
            out.append(b.para("part", blk.text, pb, fr))
        elif blk.kind in STYLE:
            out.append(b.para(blk.kind, blk.text, pb, fr))
        elif blk.kind == "table":
            out.append(b.table(blk.table, pb, fr))
        elif blk.kind == "fig":
            out.append(b.figure(blk.fig, pb, fr))
        else:
            raise SystemExit("모르는 블록 %r" % blk.kind)
        prev = (blk.kind, bool(blk.kind == "table" and blk.table.split))
    if k < 0:
        out.append(para_open(4, 0, False) + first_runs + "</hp:p>")
    out.append("</hs:sec>")
    xml = "".join(out)
    assert "linesegarray" not in xml
    return xml, b


def build_cover(s0_orig, hp):
    s0 = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", s0_orig, flags=re.S)
    if s0.count(TEMPLATE_TITLE) != 1:
        raise SystemExit("원본 표지 제목을 찾지 못함: %s" % TEMPLATE_TITLE)
    s0 = s0.replace(TEMPLATE_TITLE, escape(REPORT_TITLE))
    return s0   # 표지 제목 문단 pp36 은 원래 어절 단위 줄 나눔(BREAK_WORD)


def preview_text(blocks):
    lines = [REPORT_TITLE, ""]
    for blk in blocks:
        if blk.kind in STYLE:
            lines.append(blk.text)
        elif blk.kind == "table" and blk.table.caption:
            lines.append("[표] " + blk.table.caption)
        elif blk.kind == "fig":
            lines.append("[그림] " + blk.fig["title"])
    return "\r\n".join(lines)[:1000]


def blank_png(w, h):
    import pymupdf
    pm = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), 0)
    pm.clear_with(255)
    return pm.tobytes("png")


def build_hwpx(blocks, out_path, forced=frozenset()):
    with zipfile.ZipFile(TEMPLATE_HWPX) as z:
        entries = [(i, z.read(i.filename)) for i in z.infolist()]
    get = dict((i.filename, d) for i, d in entries)
    header = get["Contents/header.xml"].decode("utf-8")
    check_header(parse_header(header))
    hp = HeaderPatch(header)
    for cp in RED_BASE_CPS:
        hp.red(cp)
    section1, b = build_section1(blocks, get["Contents/section1.xml"].decode("utf-8"), hp, forced)
    section0 = build_cover(get["Contents/section0.xml"].decode("utf-8"), hp)
    header = hp.result()
    # content.hpf: 문서 정보 제목, 지은이 비움, 그림 목록(원본 21개 빼고 새 그림)
    hpf = get["Contents/content.hpf"].decode("utf-8")
    hpf, k = re.subn(r"<opf:title>.*?</opf:title>|<opf:title/>", "<opf:title>%s</opf:title>" % escape(REPORT_TITLE), hpf, count=1)
    if k != 1:
        raise SystemExit("content.hpf 에 opf:title 이 없음")
    hpf = re.sub(r'<opf:meta name="creator" content="text">[^<]*</opf:meta>', '<opf:meta name="creator" content="text"/>', hpf)
    hpf = re.sub(r'<opf:item id="image\d+" href="BinData/[^"]+" media-type="[^"]+" isEmbeded="1"/>', "", hpf)
    items = "".join('<opf:item id="%s" href="BinData/%s.png" media-type="image/png" isEmbeded="1"/>' % (bid, bid)
                    for bid, _, _ in b.images)
    anchor = '<opf:item id="section0" href="Contents/section0.xml" media-type="application/xml"/>'
    if anchor not in hpf:
        raise SystemExit("content.hpf 에 section0 항목이 없음")
    hpf = hpf.replace(anchor, anchor + items, 1)
    settings = get["settings.xml"].decode("utf-8")
    settings = re.sub(r"<ha:CaretPosition [^>]*/>", '<ha:CaretPosition listIDRef="0" paraIDRef="0" pos="0"/>', settings)
    prv = get["Preview/PrvImage.png"]
    pw, ph, _ = png_info(prv, "PrvImage.png")
    replace = {
        "Contents/header.xml": header.encode("utf-8"),
        "Contents/section0.xml": section0.encode("utf-8"),
        "Contents/section1.xml": section1.encode("utf-8"),
        "Contents/content.hpf": hpf.encode("utf-8"),
        "settings.xml": settings.encode("utf-8"),
        "Preview/PrvText.txt": preview_text(blocks).encode("utf-8"),
        "Preview/PrvImage.png": blank_png(pw, ph),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img_attr = next((i.external_attr for i, _ in entries if i.filename.startswith("BinData/")), 0)
    images_done = False
    with zipfile.ZipFile(out_path, "w") as z:
        assert entries[0][0].filename == "mimetype"
        for info, data in entries:
            name = info.filename
            if name.startswith("BinData/"):
                if not images_done:
                    for bid, _, img in b.images:
                        zi = zipfile.ZipInfo("BinData/%s.png" % bid, date_time=(1980, 1, 1, 0, 0, 0))
                        zi.compress_type = zipfile.ZIP_STORED
                        zi.external_attr = img_attr
                        z.writestr(zi, img)
                    images_done = True
                continue
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED if name in ("mimetype", "version.xml", "Preview/PrvImage.png") else zipfile.ZIP_DEFLATED
            zi.external_attr = info.external_attr
            z.writestr(zi, replace.get(name, data))
        if not images_done and b.images:
            raise SystemExit("원본에 BinData 항목이 없어 그림 자리를 정하지 못함")
    return b, hp, section1


# ---------------------------------------------------------------------------
# 한글(COM) 변환 — 잠금 파일·자기가 띄운 숨은 한글만 정리
# ---------------------------------------------------------------------------
def hwp_processes():
    """실행 중인 Hwp.exe: {PID: 창 제목}. 창 제목이 없으면 'N/A'"""
    r = subprocess.run(["tasklist", "/V", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq Hwp.exe"],
                       capture_output=True, encoding="cp949", errors="replace")
    out = {}
    for row in csv.reader(io.StringIO(r.stdout)):
        if len(row) >= 9 and row[0].lower() == "hwp.exe":
            out[int(row[1])] = row[8]
    return out


def pid_alive(pid):
    """그 PID 의 프로세스가 아직 살아 있는가(Windows: OpenProcess + GetExitCodeProcess — 종료 신호를 보내지 않음.
    os.kill(pid, 0) 은 Windows 에서 그 프로세스를 끝내 버리므로 쓰지 않음). 알 수 없으면 살아 있다고 봄(기다리는 쪽으로)"""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return True
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5             # ERROR_ACCESS_DENIED = 있지만 열 권한 없음
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == 259                        # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    except Exception:
        return True


def lock_text():
    try:
        with open(HWP_LOCK, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def lock_acquire():
    """보고서/hwp_build.lock: 있으면 30초 간격으로 기다림(최대 20분), 없으면 만들고 씀
    (다른 조립기 build_report_hwp.py 와 같은 꼴: owner= / started= / pid=).
    이 조립기가 남긴 잠금인데 그 PID 가 이미 끝난 것(강제 종료로 finally 를 못 거친 것)만 지우고 바로 다시 시도함 —
    다른 세션의 잠금은 건드리지 않음"""
    t0 = time.time()
    while True:
        try:
            fd = os.open(HWP_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            txt = lock_text()
            if txt is None:                                  # 그 사이 지워짐
                continue
            m = re.search(r"pid=(\d+)", txt)
            if LOCK_OWNER in txt and m and int(m.group(1)) != os.getpid() and not pid_alive(int(m.group(1))):
                try:
                    os.remove(HWP_LOCK)
                    log("이 조립기가 남긴 잠금(끝난 PID %s)을 지움: %s" % (m.group(1), txt.strip().replace("\n", " | ")))
                except OSError:
                    pass
                continue
            if time.time() - t0 > LOCK_MAX_S:
                raise SystemExit("한글 잠금 파일이 %d분 넘게 남아 있어 멈춤: %s (%s)"
                                 % (LOCK_MAX_S // 60, HWP_LOCK, txt.strip().replace("\n", " | ")))
            log("한글 잠금 대기(%d초 간격, 최대 %d분): %s" % (LOCK_POLL_S, LOCK_MAX_S // 60, txt.strip().replace("\n", " | ")))
            time.sleep(LOCK_POLL_S)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("owner=%s\nstarted=%s\npid=%d\n" % (LOCK_OWNER, time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid()))
        return


def lock_release():
    """내가 만든 잠금(이 조립기 이름 + 내 PID)일 때만 지움"""
    txt = lock_text()
    if txt and LOCK_OWNER in txt and ("pid=%d" % os.getpid()) in txt:
        try:
            os.remove(HWP_LOCK)
        except OSError:
            pass


def check_writable(paths):
    """결과 파일이 다른 프로그램(한글·PDF 보기)에 열려 있으면 덮어쓸 수 없음 -> 잠금을 잡기 전에 알림"""
    busy = []
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r+b"):
                    pass
            except OSError:
                busy.append(p)
    if busy:
        raise SystemExit("결과 파일이 다른 프로그램에 열려 있어 덮어쓸 수 없음(닫고 다시): " + ", ".join(busy))


def convert_with_hangul(hwpx_path, hwp_path, pdf_path):
    from pyhwpx import Hwp
    before = set(hwp_processes())
    hwp, com, mine = None, None, []
    try:
        hwp = Hwp(new=True, visible=False)
        mine = [p for p in hwp_processes() if p not in before]   # 이번에 띄운 숨은 한글
        com = hwp.hwp
        if not com.Open(hwpx_path, "HWPX", "forceopen:true"):
            raise RuntimeError("HWPX 열기 실패: " + hwpx_path)
        pages = com.PageCount
        for p in (hwp_path, pdf_path):
            if os.path.exists(p):
                os.remove(p)
        if not com.SaveAs(hwp_path, "HWP", ""):
            raise RuntimeError("HWP 저장 실패")
        if not com.SaveAs(pdf_path, "PDF", ""):
            raise RuntimeError("PDF 저장 실패")
        return pages
    finally:
        if com is not None:
            try:
                com.Clear(1)
            except Exception:
                pass
            try:
                com.Quit()
            except Exception:
                pass
        if not mine:
            mine = [p for p in hwp_processes() if p not in before]
        time.sleep(2)
        now = hwp_processes()
        for p in mine:
            if p in now and now[p] in ("N/A", ""):
                subprocess.run(["taskkill", "/PID", str(p), "/F"], capture_output=True)
                log("Quit 뒤 남은 숨은 한글(이번에 띄운 것) 종료: PID", p)


# ---------------------------------------------------------------------------
# PNG·점검
# ---------------------------------------------------------------------------
FOOT_Y = 795.0      # PDF 좌표에서 이보다 아래는 꼬리말 영역(쪽 번호 '- n -' 기준선 약 811.9pt)


def render_png(pdf_path, png_dir, dpi=PNG_DPI):
    """쪽별 PNG: p01.png …(100쪽 이상이면 p001.png …)"""
    import pymupdf
    os.makedirs(png_dir, exist_ok=True)
    for f in os.listdir(png_dir):
        if re.fullmatch(r"p\d+\.png", f):
            os.remove(os.path.join(png_dir, f))
    doc = pymupdf.open(pdf_path)
    n = doc.page_count
    width = max(2, len(str(n)))
    for i, pg in enumerate(doc):
        pg.get_pixmap(dpi=dpi).save(os.path.join(png_dir, "p%0*d.png" % (width, i + 1)))
    doc.close()
    return n, width


def norm(s):
    return re.sub(r"\s+", "", s.replace(ZWSP, ""))


def page_blocks(pg, by_position=False):
    """쪽의 글 묶음(꼬리말 영역 빼고). 기본은 PDF 글 흐름 순서(표 칸 하나의 여러 줄이 이어서 나옴),
    by_position=True 면 위에서 아래(같은 높이면 왼쪽부터) 순서 — 쪽을 넘기는 표의 제목은 흐름 순서로는 뒤에 나옴"""
    bl = [b for b in pg.get_text("blocks") if b[1] < FOOT_Y]
    return sorted(bl, key=lambda b: (round(b[1], 1), b[0])) if by_position else bl


def pdf_pages_text(pdf_path):
    """쪽마다 본문 글(꼬리말 쪽 번호 빼고 — 쪽을 넘긴 문단 가운데에 '- n -' 이 끼지 않게).
    (흐름 순서 글, 위치 순서 글) 두 벌: 흐름 순서는 표 칸 하나의 여러 줄이 이어서 나오고, 위치 순서는 쪽을 넘긴 표가
    쪽 위에 있고 그 아래 문단이 다음 쪽으로 이어질 때(흐름 순서로는 표 글이 문단 가운데에 낌) 문단이 이어짐"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    flow = ["".join(b[4] for b in page_blocks(pg)) for pg in doc]
    pos = ["".join(b[4] for b in page_blocks(pg, by_position=True)) for pg in doc]
    doc.close()
    return flow, pos


def check_text(blocks, pages_text):
    """원고 글이 PDF 에 다 들어갔는지(공백 무시, 쪽 사이에 걸친 문단도 이어 붙여 봄, 흐름·위치 순서 중 한 곳에만 있어도 됨)"""
    hays = [norm("".join(t)) for t in pages_text]

    def found(t):
        k = norm(t)
        return any(k in h for h in hays)
    missing = []
    for blk in blocks:
        where = "%s:%d" % blk.src
        if blk.kind in STYLE:
            if not found(blk.text):
                missing.append((where, blk.text[:50]))
        elif blk.kind == "table":
            tb = blk.table
            if tb.caption and not found(tb.caption):
                missing.append((where, "표 제목: " + tb.caption[:50]))
            for row in [tb.header] + tb.rows:
                for cell in row:
                    for part in cell_parts(cell):
                        if part and not found(part):
                            missing.append((where, "표 칸: " + part[:50]))
        elif blk.kind == "fig":
            if not found(blk.fig["title"]):
                missing.append((where, "그림 제목: " + blk.fig["title"][:50]))
    return missing


def check_numbers(pdf_path, n_tbl, n_fig):
    """'[표 n]'·'[그림 n]' 번호가 쪽·위치 순서로 1, 2, … 인지"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    t, f = [], []
    for pg in doc:
        for b in page_blocks(pg, by_position=True):
            s = norm(b[4])
            t += [int(x) for x in re.findall(r"\[표(\d+)\]", s)]
            f += [int(x) for x in re.findall(r"\[그림(\d+)\]", s)]
    doc.close()
    return t, f, t == list(range(1, n_tbl + 1)), f == list(range(1, n_fig + 1))


def table_lines(pdf_path):
    """표 칸 글(10pt) 줄: [(쪽, y, 글(공백·폭 없는 빈칸 없앰))], PDF 글 흐름 순서(칸마다 줄이 이어서 나옴)"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    lines = []
    for pno, pg in enumerate(doc):
        for b in pg.get_text("rawdict")["blocks"]:
            for ln in b.get("lines", []):
                chars = [c for s in ln["spans"] if abs(s["size"] - 10.0) < 0.6 for c in s["chars"]]
                if not chars or ln["bbox"][1] > FOOT_Y:
                    continue
                t = norm("".join(c["c"] for c in chars))
                if t:
                    lines.append((pno + 1, ln["bbox"][1], t))
    doc.close()
    return lines


def check_cells(tables, pdf_path):
    """표 칸 글이 PDF 에 있고, 줄이 바뀐 곳이 띄어쓰기 자리(또는 긴 낱말에 넣은 줄바꿈 자리)인지 확인.
    돌려주는 값: (문제 [(표, 칸 글, 내용)], 칸 폭보다 긴 낱말이 글자 사이에서 나뉜 것 [(표, 칸 글, 내용)])"""
    lines = table_lines(pdf_path)
    joined = "\n".join(t for _, _, t in lines)
    probs, long_words = [], []
    for tb in tables:
        name = "[표 %s] %s" % (tb.number, (tb.caption or "")[:20]) if tb.number else "제목 없는 표(%s:%d)" % tb.src
        widths = getattr(tb, "widths", None) or [10 ** 9] * len(tb.header)
        for ri, row in enumerate([tb.header] + tb.rows):
            for ci, cell in enumerate(row):
                inner = max(1000, widths[ci] - CELL_PAD_LR)
                for part in cell_parts(cell):
                    words = part.split()
                    flat = "".join(words)
                    if len(flat) < 2:
                        continue
                    ok_at, acc = set(), 0
                    for wi, w in enumerate(words):
                        if text_width(w, ri == 0) * WRAP_SAFETY > inner:      # soft_breaks 가 줄바꿈 자리를 넣은 낱말
                            ok_at.update(acc + x for x in break_points(w))
                        acc += len(w)
                        if wi < len(words) - 1:
                            ok_at.add(acc)
                    best = None                    # 같은 글이 여러 곳에 있으면 낱말 중간 줄바꿈이 가장 적은 곳
                    for m in re.finditer("\n?".join(re.escape(ch) for ch in flat), joined):
                        idx, bad = 0, []
                        for ch in m.group(0):
                            if ch == "\n":
                                if idx not in ok_at:
                                    bad.append(idx)
                            else:
                                idx += 1
                        if best is None or len(bad) < len(best):
                            best = bad
                        if not bad:
                            break
                    if best is None:
                        probs.append((name, part, "PDF 표 글에서 찾지 못함(잘림 의심)"))
                        continue
                    cuts, forced = [], []
                    for at in best:
                        pos = 0
                        for w in words:
                            if pos < at < pos + len(w):
                                cut = "%s|%s" % (w[:at - pos], w[at - pos:])
                                (forced if text_width(w, ri == 0) * WRAP_SAFETY > inner else cuts).append(cut)
                                break
                            pos += len(w)
                    if cuts:
                        probs.append((name, part, "낱말 중간 줄바꿈: " + ", ".join(cuts)))
                    if forced:
                        long_words.append((name, part, "칸보다 긴 낱말이 글자 사이에서 나뉨: " + ", ".join(forced)))
    return probs, long_words


def find_orphan_headers(blocks, pdf_path):
    """쪽을 넘기는 표 가운데 첫 쪽에 머리 행만 찍히고 첫 행은 다음 쪽으로 넘어간 표의 블록 번호"""
    lines = table_lines(pdf_path)
    out = []
    for bi, (p, y) in sorted(table_positions(blocks, pdf_path, lines).items()):
        tb = blocks[bi].table
        if not getattr(tb, "split", False) or not tb.rows:
            continue
        row1 = max((norm(c) for c in tb.rows[0]), key=len)
        if row1 and row1[:12] not in "".join(t for pp, yy, t in lines if pp == p and yy > y + 1):
            out.append(bi)
    return out


def break_target(blocks, bi):
    """머리 행만 남은 표를 새 쪽에서 시작할 때: 표 바로 앞이 제목('1.'·'1.1'·□) 줄이면 그 제목부터"""
    j = bi
    while j > 0 and blocks[j - 1].kind in ("h1", "h2", "box"):
        j -= 1
    return j


# 앞 쪽 빈 자리 메우기(find_gaps): 표·그림이 통째로 다음 쪽으로 밀려 앞 쪽 아래가 크게 비면 다음 변환에서
#   표 = 쪽 경계에서 나누는 표로(원본 [표 15]·[표 18] 모양, 머리 행 반복), 그림 = FIG_MIN_SCALE 까지 줄여 앞 쪽에 넣음
PDF_SHIFT = 14.45                                                 # 한글 PDF 가 모든 요소를 위·왼쪽으로 옮겨 찍는 양(pt)
BODY_TOP_PT = (2834 + 4251) / 100.0 - PDF_SHIFT                   # 본문 위 끝(PDF 좌표) 56.4
BODY_BOTTOM_PT = 842 - (1417 + 4251) / 100.0 - PDF_SHIFT          # 본문 아래 끝 770.9
GAP_RATIO = 0.40          # 앞 쪽 아래 빈 높이가 본문 높이의 이 비율보다 크면 고침(원본 kb.pdf 본문 쪽 최대 42%)
GAP_TOP = 110.0           # 다음 쪽 본문 위 끝에서 이 높이(pt) 안에서 시작한 표·그림만(앞에 따라 넘어간 제목 줄 허용)
GAP_MIN_ROWS = 3          # 나누는 표로 바꿀 표의 최소 본문 행 수
SPACER_PT = 16.0          # 표·그림 사이 빈 문단 높이(pp63 함초롬돋움 12pt x 120% = 14.4pt) + 여유


def prev_is_object(blocks, bi):
    """bi 바로 앞 요소(쪽 나눔 표시 빼고)가 표·그림인가 — 그러면 둘 사이에 빈 문단이 있음(build_section1·쪽을 넘기는 표 뒤)"""
    j = bi - 1
    while j >= 0 and blocks[j].kind == "pagebreak":
        j -= 1
    return j >= 0 and blocks[j].kind in ("table", "fig")
FIG_MIN_SCALE = 0.75      # 그림은 원래 크기의 이 비율까지만 줄임
MAX_CONVERT = 4           # 한글 변환 최대 횟수(머리 행만 남은 표·빈 자리 고치기 포함)


def content_bottom(pg):
    """쪽 본문(꼬리말 빼고)에서 가장 아래 요소(글·그림·선)의 아래 끝(PDF 좌표)"""
    ys = [b[3] for b in pg.get_text("blocks") if b[1] < FOOT_Y]
    ys += [i["bbox"][3] for i in pg.get_image_info() if i["bbox"][1] < FOOT_Y]
    ys += [d["rect"].y1 for d in pg.get_drawings() if d["rect"].y0 < FOOT_Y]
    return max(ys) if ys else BODY_TOP_PT


def table_positions(blocks, pdf_path, lines=None):
    """표 블록 번호 -> (쪽, 머리 행 첫 줄 y). 표 칸 글(10pt) 흐름에서 머리 행 글을 찾되, 앞 표보다 (쪽, y) 가 뒤이고
    바로 뒤에 첫 행 글이 이어지는 곳만 받음 — 쪽을 넘긴 표의 반복 머리 행(머리 행이 같은 표가 이어질 때)과,
    쪽을 넘긴 표의 다음 쪽 부분이 흐름 순서로는 그 쪽의 다른 글 뒤에 나오는 것에 걸리지 않게"""
    if lines is None:
        lines = table_lines(pdf_path)
    where, parts = [], []
    for p, y, t in lines:
        parts.append(t)
        where.extend([(p, y)] * len(t))
    s = "".join(parts)
    out, prev = {}, (0, -1.0)
    for bi, blk in enumerate(blocks):
        if blk.kind != "table":
            continue
        tb = blk.table
        head = norm("".join(tb.header))
        if not head:
            continue
        first = next((norm(c) for c in (tb.rows[0] if tb.rows else []) if norm(c)), "")
        i = s.find(head)
        while i >= 0:
            if where[i] > prev and (not first or first[:10] in s[i + len(head): i + len(head) + 600]):
                out[bi] = prev = where[i]
                break
            i = s.find(head, i + 1)
    return out


def after_manual_break(blocks, bi, forced):
    """표·그림(앞 제목 줄 포함)이 일부러 새 쪽에서 시작한 것인지: 부문 제목 다음, '<!-- 쪽 나눔 -->' 다음, 머리 행 처리로 넘긴 것"""
    j = break_target(blocks, bi)
    if j in forced:
        return True
    return j > 0 and blocks[j - 1].kind in ("pagebreak", "part")


def find_gaps(blocks, pdf_path, forced):
    """앞 쪽 아래를 크게 비우고 다음 쪽 맨 위로 밀려난 표·그림.
    돌려주는 값: (나누는 표로 바꿀 표 블록 번호 [(번호, 앞 쪽, 빈 비율)], 줄일 그림 {블록 번호: (비율, 앞 쪽, 빈 비율)})"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    bottoms = [content_bottom(pg) for pg in doc]
    imgs = []
    for pno, pg in enumerate(doc):
        for info in sorted(pg.get_image_info(), key=lambda x: (x["bbox"][1], x["bbox"][0])):
            bb = info["bbox"]
            imgs.append((pno + 1, bb[1], bb[3], bb[2] - bb[0]))
    doc.close()
    tpos = table_positions(blocks, pdf_path)
    figs = [bi for bi, blk in enumerate(blocks) if blk.kind == "fig"]
    fpos = dict(zip(figs, imgs)) if len(figs) == len(imgs) else {}
    H = BODY_BOTTOM_PT - BODY_TOP_PT
    split_new, shrink = [], {}
    for bi, blk in enumerate(blocks):
        if blk.kind == "table" and bi in tpos:
            p, y = tpos[bi]
        elif blk.kind == "fig" and bi in fpos:
            p, y = fpos[bi][0], fpos[bi][1]
        else:
            continue
        if p < 3 or y - BODY_TOP_PT > GAP_TOP or after_manual_break(blocks, bi, forced):
            continue
        blank = BODY_BOTTOM_PT - bottoms[p - 2]
        if blank / H <= GAP_RATIO:
            continue
        if blk.kind == "table":
            tb = blk.table
            if not getattr(tb, "split", False) and len(tb.rows) >= GAP_MIN_ROWS:
                split_new.append((bi, p - 1, blank / H))
        else:
            _, y0, y1, wpt = fpos[bi]
            fig_h = y1 - y0
            avail = blank - (y0 - BODY_TOP_PT) - 12.0          # 따라 넘어간 제목 줄 높이, 앞 문단 아래 간격 여유
            if prev_is_object(blocks, bi):
                # 바로 앞이 표·그림이면 그 사이 빈 문단(pp63, 12pt x 120%)이 앞 쪽 끝에 남음 — 글이 없어 PDF 로는 안 보임
                avail -= SPACER_PT
            s = 1.0
            for _ in range(3):                                  # 그림이 좁아지면 제목 줄 수가 늘 수 있어 몇 번 다시 셈
                cap_w = text_width("[그림 %d] %s" % (figs.index(bi) + 1, blk.fig["title"])) * 1.2 / 100.0
                cap_h = 8.5 + 19.2 * max(1, -(-int(cap_w) // max(1, int(wpt * s))))
                s = min(1.0, (avail - cap_h) / fig_h)
            if FIG_MIN_SCALE <= s < 0.995:
                shrink[bi] = (s, p - 1, blank / H)
    return split_new, shrink


def gap_pages(pdf_path):
    """아래가 본문 높이의 GAP_RATIO 보다 많이 빈 쪽: [(쪽, 빈 비율, 다음 쪽 첫 요소)] — 부문 제목 쪽(26pt 글만 있는 쪽)·
    표지·마지막 쪽 빼고. 다음 쪽 첫 요소가 표·그림이면 통째로 밀려난 것(find_gaps 가 못 고친 것), 제목이면 쪽 나눔"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    H = BODY_BOTTOM_PT - BODY_TOP_PT
    out = []
    for i in range(1, doc.page_count - 1):
        pg = doc[i]
        spans = [s for b in pg.get_text("dict")["blocks"] if b["bbox"][1] < FOOT_Y
                 for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
        if spans and all(s["size"] >= 25 for s in spans):
            continue
        r = (BODY_BOTTOM_PT - content_bottom(pg)) / H
        if r <= GAP_RATIO:
            continue
        nxt = doc[i + 1]
        first = []
        for b in nxt.get_text("dict")["blocks"]:
            if b["bbox"][1] >= FOOT_Y:
                continue
            if b.get("type") == 1:
                first.append((b["bbox"][1], "그림"))
            else:
                sp = [s for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
                if sp:
                    t = "".join(s["text"] for s in sp)[:30]
                    if sp[0]["size"] >= 25:
                        t = "\x00part"                      # 다음 쪽이 부문 제목 쪽: 일부러 쪽을 나눈 곳
                    first.append((b["bbox"][1], ("표 '%s'" % t) if sp[0]["size"] < 10.6 else ("'%s'" % t)))
        for d in nxt.get_drawings():
            if d["rect"].y0 < FOOT_Y:
                first.append((d["rect"].y0, "표"))
        what = min(first)[1] if first else "?"
        if "\x00part" in what:
            continue
        out.append((i + 1, r, what))
    doc.close()
    return out


def check_layout(pdf_path):
    """그림 수·크기, 쪽 번호, 본문 영역 아래로 넘친 글"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    imgs, nums_bad, overflow = [], [], []
    # 본문 아래 끝(PDF 좌표): 용지 높이 - 아래 여백 - 꼬리말 - 한글 PDF 가 모든 요소를 위·왼쪽으로 14.45pt 옮겨 찍는 양
    # (원본 kb.pdf 와 이 조립기 PDF 모두 같음, work/kb_스타일표.md 1절) + 여유 2pt
    body_bottom = 842 - (1417 + 4251) / 100.0 - 14.45 + 2.0
    for pno, pg in enumerate(doc):
        for info in pg.get_image_info():
            bb = info["bbox"]
            imgs.append((pno + 1, round(bb[2] - bb[0], 1), round(bb[3] - bb[1], 1)))
        blocks = pg.get_text("blocks")
        if pno >= 1:
            foot = norm("".join(b[4] for b in blocks if b[1] >= FOOT_Y))
            if ("-%d-" % (pno + 1)) not in foot:
                nums_bad.append(pno + 1)
        for b in blocks:
            if b[3] > body_bottom and b[1] < FOOT_Y:
                overflow.append((pno + 1, round(b[3], 1), b[4].strip()[:30]))
    doc.close()
    return imgs, nums_bad, overflow


def heading_pages(blocks, pdf_path):
    """부문·'1.'·'1.1' 제목이 찍힌 쪽(14.5pt 이상 글 줄에서 차례대로 찾음)"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    big = []
    for pno, pg in enumerate(doc):
        for b in pg.get_text("rawdict")["blocks"]:
            for ln in b.get("lines", []):
                chars = [c for s in ln["spans"] if s["size"] >= 14.5 for c in s["chars"]]
                if chars:
                    big.append((pno + 1, norm("".join(c["c"] for c in chars))))
    doc.close()
    out, k = [], 0
    for blk in blocks:
        if blk.kind not in ("part", "h1", "h2"):
            continue
        nh = norm(blk.text)
        p = None
        for j in range(k, len(big)):
            line = big[j][1]
            if line.startswith(nh[:12]) or (len(line) >= 6 and nh.startswith(line)):
                p, k = big[j][0], j + 1
                break
        out.append((blk.kind, blk.text, p))
    return out


def log_build(hwpx, b, hp, section1):
    log("HWPX:", hwpx, "(section1.xml %d bytes, 표 %d, 그림 %d, 새 문단 모양 %d, 새 글자 모양 %d)"
        % (len(section1.encode("utf-8")), len(b.tables), len(b.figs), len(hp.new_pp), len(hp.new_cp)))
    for num, cap, ws, r, how, al in b.log_tables:
        log("  표 %-3s 열 폭 %s%% (합 %.1f%%) 정렬 %s, 어림 높이 본문의 %.0f%% -> %s  %s"
            % (num if num else "-", ws, sum(ws), "/".join({"center": "가", "right": "오", "left": "왼"}[x] for x in al),
               100 * r, how, (cap or "")[:30]))
    for i, f in enumerate(b.figs):
        name = os.path.basename(f["path"]) if f.get("path") else "(파일 없음 — 빨간 자리 그림) " + f["rel"]
        log("  그림 %d %s -> %s x %s mm" % (i + 1, name, f["size_mm"][0], f["size_mm"][1]))


def count_placeholders(blocks):
    """남은 '{{…}}' 자리표시 수(본문·표 칸·표 제목·그림 제목)와 파일 없는 그림 수"""
    n = 0
    for blk in blocks:
        if blk.kind in STYLE:
            n += len(RE_PH.findall(blk.text))
        elif blk.kind == "table":
            tb = blk.table
            n += len(RE_PH.findall(tb.caption or ""))
            n += sum(len(RE_PH.findall(c)) for row in [tb.header] + tb.rows for c in row)
        elif blk.kind == "fig":
            n += len(RE_PH.findall(blk.fig["title"]))
    nofile = [blk.fig["rel"] for blk in blocks if blk.kind == "fig" and not blk.fig.get("path")]
    return n, nofile


# ---------------------------------------------------------------------------
def main():
    global GAP_RATIO, FIG_MIN_SCALE
    ap = argparse.ArgumentParser(description="경복대 양식 HWP 조립기")
    ap.add_argument("--sections", default=SECTIONS_DIR, help="원고 md 폴더(기본 sections2/, 순서.txt 순서)")
    ap.add_argument("--out", default=OUT_DIR, help="결과 폴더(기본 out/)")
    ap.add_argument("--no-com", action="store_true", help="한글(COM) 없이 HWPX 까지만")
    ap.add_argument("--gap", type=float, default=GAP_RATIO,
                    help="앞 쪽 아래가 본문 높이의 이 비율보다 비면 밀려난 표를 쪽 경계에서 나누고 그림을 줄임"
                         "(기본 %(default)s = 원본 kb.pdf 본문 쪽 최대 빈 자리 42%% 수준, 쪽 수를 줄이려면 0.25 등)")
    ap.add_argument("--fig-min", type=float, default=FIG_MIN_SCALE,
                    help="빈 자리를 메우려고 그림을 줄일 때 최소 비율(기본 %(default)s)")
    a = ap.parse_args()
    GAP_RATIO, FIG_MIN_SCALE = a.gap, a.fig_min
    sys.stdout.reconfigure(encoding="utf-8")
    sections_dir, out_dir = os.path.abspath(a.sections), os.path.abspath(a.out)
    log("원고 폴더:", sections_dir)
    log("결과 폴더:", out_dir)

    files, missing = read_order(sections_dir)
    if missing:
        log("순서.txt 에 있으나 아직 없는 파일(건너뜀): " + ", ".join(missing))
    if not files:
        raise SystemExit("붙일 md 가 없음")
    blocks = []
    for p in files:
        bl = parse_md(p, default_base_dirs)
        log("  %-46s 요소 %3d개 (표 %d, 그림 %d)" % (os.path.basename(p), len(bl),
            sum(1 for x in bl if x.kind == "table"), sum(1 for x in bl if x.kind == "fig")))
        blocks.extend(bl)

    hwpx = os.path.join(out_dir, "_build", OUT_NAME + ".hwpx")
    if a.no_com:
        b, hp, section1 = build_hwpx(blocks, hwpx)
        log_build(hwpx, b, hp, section1)
        return

    hwp_path = os.path.join(out_dir, OUT_NAME + ".hwp")
    pdf_path = os.path.join(out_dir, OUT_NAME + ".pdf")
    check_writable([hwp_path, pdf_path])
    forced, handled, left = set(), [], []
    gap_tables, gap_figs = [], {}        # 앞 쪽 빈 자리 메우기: 나누는 표로 바꾼 표 [(번호, 앞 쪽, 빈 비율)], 줄인 그림
    fig_no = dict((bi, k + 1) for k, bi in enumerate(i for i, x in enumerate(blocks) if x.kind == "fig"))
    lock_acquire()
    try:
        # 변환 뒤 (1) 쪽 끝에 머리 행만 남은 표 -> 그 표(앞 제목)를 새 쪽에서, (2) 앞 쪽 아래를 크게 비우고 다음 쪽 맨 위로
        # 밀려난 표 -> 쪽 경계에서 나누는 표로, 그림 -> 조금 줄여 앞 쪽에 넣고 다시 변환(최대 MAX_CONVERT 번)
        for attempt in range(1, MAX_CONVERT + 1):
            b, hp, section1 = build_hwpx(blocks, hwpx, forced)
            pages = convert_with_hangul(hwpx, hwp_path, pdf_path)
            orphans = [bi for bi in find_orphan_headers(blocks, pdf_path) if bi not in handled]
            g_tbl, g_fig = find_gaps(blocks, pdf_path, forced)
            g_tbl = [x for x in g_tbl if x[0] not in handled and not getattr(blocks[x[0]].table, "force_split", False)]
            g_fig = dict((bi, v) for bi, v in g_fig.items()
                         if blocks[bi].fig.get("scale", 1.0) * v[0] >= FIG_MIN_SCALE)
            if not (orphans or g_tbl or g_fig):
                break
            if attempt == MAX_CONVERT:
                left = orphans
                break
            msg = []
            if orphans:
                msg.append("쪽 끝에 머리 행만 남은 표 %d개 -> 새 쪽에서 시작" % len(orphans))
            if g_tbl:
                msg.append("앞 쪽을 크게 비우고 밀려난 표 %d개 -> 쪽 경계에서 나누는 표로" % len(g_tbl))
            if g_fig:
                msg.append("앞 쪽을 크게 비우고 밀려난 그림 %d개 -> 줄여서 앞 쪽에" % len(g_fig))
            log("%d차 변환: %s — 다시 변환" % (attempt, "; ".join(msg)))
            for bi in orphans:
                handled.append(bi)
                forced.add(break_target(blocks, bi))
            for bi, pg, r in g_tbl:
                blocks[bi].table.force_split = True
                gap_tables.append((bi, pg, r))
            for bi, (s, pg, r) in g_fig.items():
                f = blocks[bi].fig
                f["scale"] = f.get("scale", 1.0) * s
                gap_figs[bi] = (f["scale"], pg, r)
    finally:
        lock_release()
    log_build(hwpx, b, hp, section1)
    n, width = render_png(pdf_path, os.path.join(out_dir, "png"))
    log("HWP:", hwp_path, "(한글 쪽 수 %s)" % pages)
    log("PDF:", pdf_path, "(%d쪽)" % n)
    log("PNG:", os.path.join(out_dir, "png"), "p%0*d.png ~ p%0*d.png" % (width, 1, width, n))
    if handled:
        log("쪽 끝 머리 행만 남은 표를 새 쪽으로 넘김: " + ", ".join(
            "[표 %s] %s" % (blocks[bi].table.number, (blocks[bi].table.caption or "")[:20]) for bi in handled))
    if gap_tables:
        log("앞 쪽 빈 자리 메움 — 쪽 경계에서 나누는 표로 바꾼 표: " + ", ".join(
            "[표 %s](%d쪽 아래 %.0f%% 비었음)" % (blocks[bi].table.number, pg, 100 * r) for bi, pg, r in gap_tables))
    if gap_figs:
        log("앞 쪽 빈 자리 메움 — 줄인 그림: " + ", ".join(
            "[그림 %d] %.0f%% 크기(%d쪽 아래 %.0f%% 비었음)" % (fig_no[bi], 100 * s, pg, 100 * r)
            for bi, (s, pg, r) in sorted(gap_figs.items())))

    pages_text = pdf_pages_text(pdf_path)
    missing = check_text(blocks, pages_text)
    log("점검 1 글 대조:", "빠진 글 없음" if not missing else "빠진 글 %d건" % len(missing))
    for where, t in missing[:30]:
        log("    %s  %s" % (where, t))
    t_nums, f_nums, t_ok, f_ok = check_numbers(pdf_path, b.n_tbl, b.n_fig)
    log("점검 2 번호(쪽·위치 순서): 표 %d개 %s / 그림 %d개 %s" % (
        len(t_nums), "1부터 차례 맞음" if t_ok else "차례 다름 %s" % t_nums,
        len(f_nums), "1부터 차례 맞음" if f_ok else "차례 다름 %s" % f_nums))
    probs, long_words = check_cells(b.tables, pdf_path)
    log("점검 3 표 칸:", "모든 칸 글이 있고 줄이 띄어쓰기·구분 글자 자리에서만 바뀜" if not probs else "문제 %d건" % len(probs))
    for name, part, why in probs[:40]:
        log("    %s  %r  %s" % (name, part[:40], why))
    if long_words:
        log("  참고: 칸 폭보다 긴 낱말이 글자 사이에서 나뉜 곳 %d건(원고에서 줄이거나 '<br>' 로 나누면 없어짐):" % len(long_words))
        for name, part, why in long_words[:20]:
            log("    %s  %r  %s" % (name, part[:40], why))
    if left:
        log("  쪽 끝에 머리 행만 남은 표(%d차 변환 뒤에도): " % MAX_CONVERT + ", ".join(
            "[표 %s]" % blocks[bi].table.number for bi in left))
    imgs, nums_bad, overflow = check_layout(pdf_path)
    small = [x for x in imgs if x[1] < 50 or x[2] < 30]
    log("점검 4 그림: PDF 그림 %d개 / 원고 그림 %d개%s" % (len(imgs), b.n_fig,
        "" if not small else " — 너무 작게 찍힌 그림 %s" % small))
    log("점검 5 쪽 번호:", "2쪽부터 모두 '- n -' 있음" if not nums_bad else "쪽 번호 없는 쪽 %s" % nums_bad)
    log("점검 6 본문 영역 밖 글:", "없음" if not overflow else overflow[:10])
    gaps = gap_pages(pdf_path)
    log("점검 7 쪽 아래 빈 자리(본문 높이의 %.0f%% 넘음, 부문 제목 쪽과 그 앞 쪽·마지막 쪽 빼고): %s" % (
        100 * GAP_RATIO, "없음" if not gaps else "%d쪽" % len(gaps)))
    for pno, r, what in gaps:
        log("    p%-4d %3.0f%% 빔 — 다음 쪽 첫 요소: %s" % (pno, 100 * r, what))
    n_ph, nofile = count_placeholders(blocks)
    log("점검 8 남은 자리표시: '{{…}}' %d곳(빨간 글자), 파일 없는 그림 %d개%s" % (
        n_ph, len(nofile), "" if not nofile else "(빨간 자리 그림: %s)" % ", ".join(nofile)))
    log("== 제목 쪽")
    for kind, text, p in heading_pages(blocks, pdf_path):
        log("  p%-3s %s%s" % (p, {"part": "", "h1": "  ", "h2": "    "}[kind], text[:50]))
    if WARN:
        log("== 알림 %d건(위에 찍음)" % len(WARN))


if __name__ == "__main__":
    main()
