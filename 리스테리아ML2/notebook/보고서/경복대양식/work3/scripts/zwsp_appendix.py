# -*- coding: utf-8 -*-
"""부록 원고(sections3/30_부록_입력변수기록표.md) 끝 글머리 3줄('칸 이름'·'출처 원표'·'쓸 수 있는 시점')의 긴 영문 이름에
폭 없는 빈칸(U+200B)을 넣어 한글이 그 자리에서 줄을 바꿀 수 있게 함 — 보이는 글은 그대로.

왜: 한글은 띄어쓰기 없는 긴 영문 덩어리(예: avg_temp·max_temp·…·wind_dir_cos, 117글자)를 한 낱말로 보고
  (1) 줄 남은 자리에 안 들어가면 통째로 다음 줄로 넘겨 앞 줄 낱말 사이가 크게 벌어지고('기상     18변수     =')
  (2) 한 줄보다 길면 아무 글자에서나 자름('soil|_t5', 'fine|_dust_info') — 1차 조립 PDF 47쪽.
  조립기(build_kb_hwp.py)는 표 칸에만 같은 빈칸을 넣고(soft_breaks) 본문 문단에는 넣지 않으므로 원고에 직접 넣음.
넣는 자리: 영문·숫자 쪽에 붙은 '·' 뒤, 영문 이름 안 '.' 뒤(api_data_info.|kma_…). 이미 들어간 빈칸은 다시 넣지 않음.
  그래도 빈칸 사이 조각이 20글자 이상이면(keco_airkorea_fine_dust_info. 등) 그 조각 안 '_' 뒤에도 넣음 — 2차 조립에서
  이 조각이 줄 끝에 안 들어가 앞 줄('=   api_data_info…')이 크게 벌어진 것(표 칸도 조립기가 '_' 뒤에서 나눔).
글 대조(조립기 점검 1)는 이 글자를 빼고 비교하므로 영향 없음.
"""
import io
import os
import re
import sys

ZWSP = chr(0x200B)
HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "sections3", "30_부록_입력변수기록표.md")
TARGET_PREFIX = ("- 칸 이름:", "- 출처 원표:", "- 쓸 수 있는 시점:")


def is_ascii_word(ch):
    return ch.isascii() and (ch.isalnum() or ch in "_()")


def add_breaks(line):
    out = []
    n = len(line)
    for k, ch in enumerate(line):
        out.append(ch)
        if k + 1 < n and line[k + 1] == ZWSP:
            continue
        prev = line[k - 1] if k > 0 else " "
        nxt = line[k + 1] if k + 1 < n else " "
        if ch == "·" and not prev.isspace() and not nxt.isspace() and (is_ascii_word(prev) or is_ascii_word(nxt)):
            out.append(ZWSP)
        elif ch == "." and re.match(r"[A-Za-z_]", prev) and re.match(r"[A-Za-z]", nxt):
            out.append(ZWSP)
    return split_long("".join(out))


LONG_PIECE = 20


def split_long(line):
    """빈칸(공백·U+200B) 사이 조각이 LONG_PIECE 글자 이상이면 그 안의 '_' 뒤(조각 끝 빼고)에 U+200B"""
    parts = re.split("([\\s%s])" % ZWSP, line)
    out = []
    for p in parts:
        if len(p) >= LONG_PIECE and "_" in p:
            p = "".join(ch + (ZWSP if ch == "_" and k < len(p) - 1 else "") for k, ch in enumerate(p))
        out.append(p)
    return "".join(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with io.open(MD, encoding="utf-8") as f:
        lines = f.read().split("\n")
    done = 0
    for i, s in enumerate(lines):
        if s.startswith(TARGET_PREFIX):
            new = add_breaks(s)
            added = new.count(ZWSP) - s.count(ZWSP)
            if added:
                lines[i] = new
                done += 1
            print("%d줄 %-14s 빈칸 %d개 추가(글 그대로: %s)" % (i + 1, s[:12], added, new.replace(ZWSP, "") == s))
    with io.open(MD, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    print("고친 줄", done)


if __name__ == "__main__":
    main()
