# -*- coding: utf-8 -*-
"""그림 PNG 의 해상도 표시(pHYs)만 바꿔 조립기(build_kb_hwp.py)가 넣는 그림 폭을 정함 — 그림 픽셀은 그대로.

조립기는 그림 폭 = 픽셀 폭 x 7200 / dpi(HWPUNIT)로 정하고 본문 폭 161.2mm 를 넘으면 161.2mm 로 줄임.
150dpi 로 저장된 figs3 그림은 모두 161.2mm(본문 꽉 찬 폭)로 들어가므로, 더 작게 넣을 그림만 dpi 표시를 올림.
쓰는 법: python set_png_dpi.py <png> <넣을 폭 mm>      (원본은 work3/figs3_원본_1022/ 에 먼저 복사해 둠)
그림 스크립트(02_eda_figs.py·make_figs_II.py)를 다시 돌리면 150dpi 로 돌아가 본문 꽉 찬 폭으로 들어감(내용은 그대로).

2026-10-22 3차 보고서 조립(배치 점검)에서 바꾼 것:
  I_2_item3_share_top10.png   137mm(원래 161.2mm의 85%) — 같은 쪽의 [그림 4] 시도별 그림이 72%까지 줄어 글자가 너무
                              작아지지 않게, 두 그림을 비슷한 크기로(조립기가 [그림 4]를 남은 자리에 맞춰 줄임)
  II_7-2_상위k_포착률.png      110mm — [그림 9] 아래 남은 자리(84mm)에 들어가게(161.2mm 면 다음 쪽으로 밀려 앞 쪽 아래 33% 빔)
"""
import struct
import sys
import zlib


def set_dpi(path, width_mm):
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("PNG 가 아님: " + path)
    w_px = struct.unpack(">I", data[16:20])[0]
    dpi = w_px / (width_mm / 25.4)
    ppm = int(round(dpi / 0.0254))
    chunk = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    new_chunk = struct.pack(">I", 9) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    i = data.find(b"pHYs")
    if 0 < i < 4096:
        start = i - 4
        end = i + 4 + 9 + 4
        out = data[:start] + new_chunk + data[end:]
    else:                                   # pHYs 가 없으면 IHDR 바로 뒤에 넣음
        ihdr_end = 8 + 4 + 4 + 13 + 4
        out = data[:ihdr_end] + new_chunk + data[ihdr_end:]
    with open(path, "wb") as f:
        f.write(out)
    print("%s: %d px -> %.1f mm (%.1f dpi 표시)" % (path, w_px, width_mm, ppm * 0.0254))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    set_dpi(sys.argv[1], float(sys.argv[2]))
