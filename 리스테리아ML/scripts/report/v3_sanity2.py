# -*- coding: utf-8 -*-
"""v3 보고서 1~3번 시트 점검: 삭제 대상 컬럼 잔존, 3-2 새 열, 2번 시트 기상 행"""
import sys, openpyxl
sys.stdout.reconfigure(encoding='utf-8')
wb = openpyxl.load_workbook(sys.argv[1])
ws1 = wb['1_수집가능여부']
DEL = {'prdlst_code', 'prdlst_cd', 'train_prdlst_code', 'train_prdlst_nm', 'lat_g', 'lon_g', 'obsr_de', 'date', 'any_year', 'legal_dong_code', 'lat', 'lon', 'mnfctur_de', 'aflatoxin_aw'}
left = [(r, ws1.cell(r, 4).value, ws1.cell(r, 5).value) for r in range(1, ws1.max_row + 1) if ws1.cell(r, 5).value in DEL]
print('1번 시트 삭제 대상 컬럼 잔존:', left)
print('1번 시트 1-1 행 수:', sum(1 for r in range(7, ws1.max_row + 1) if ws1.cell(r, 7).value in ('수집 가능', '일부만 있음', '대체 값', '사용 여부 결정 필요', '우리 시료 값 없음', '참고')))
ws3 = wb['3_데이터상태(raw)']
hdr_row = next(r for r in range(1, ws3.max_row + 1) if ws3.cell(r, 1).value == '위해인자' and ws3.cell(r, 2).value == '관측망')
print('3-2 헤더:', [ws3.cell(hdr_row, c).value for c in range(1, 17)])
for r in range(hdr_row + 1, hdr_row + 31):
    if ws3.cell(r, 3).value in ('avg_ta', 'sum_rn', 'temp', 'widdir', 'sun_time', 'soil_wt', 'no2_value', 'sum_gsr'):
        print(' ', ws3.cell(r, 2).value, ws3.cell(r, 3).value, '| 현재', ws3.cell(r, 11).value, '| 빈 이유:', ws3.cell(r, 12).value, '| 검출', ws3.cell(r, 13).value, '| 참고A', ws3.cell(r, 14).value, '| 참고B', ws3.cell(r, 15).value)
ws2 = wb['2_조인방법']
for r in range(1, ws2.max_row + 1):
    if isinstance(ws2.cell(r, 1).value, str) and ws2.cell(r, 1).value[:1] in '⑧⑨⑩':
        print('2번', ws2.cell(r, 1).value, '| 방식:', ws2.cell(r, 5).value, '| 주의:', ws2.cell(r, 7).value)
