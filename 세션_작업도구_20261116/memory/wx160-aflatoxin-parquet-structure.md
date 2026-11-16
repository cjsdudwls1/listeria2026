---
name: wx160-aflatoxin-parquet-structure
description: "리서치팀문서/wx160_aflatoxin_final.parquet 구조 (샘플별 wx_key + 160일 기상 시계열, _00=기준일 당일, _159=159일 전) 및 리스테리아 df와의 결합 가능성 수치"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 16777e8f-d758-4494-bdc6-0f1d57b5de7b
  modified: 2026-09-10T07:24:15.429Z
---

파일: `C:\Users\NT751\Desktop\천영진\리서치팀문서\wx160_aflatoxin_final.parquet` (1.1GB, 324,729행 × 2,152열, 아플라톡신 샘플 전용). 생성 스크립트는 폴더에 없음 (리서치팀 산출물).

- 키: `lims_data_key` (리스테리아 df와 교집합 0), `wx_key` (domestic = `336812A001` 형태 지점코드 206개, overseas = `"lat_lon"` 문자열 14,083개), `wx_dist_km`, 기준일 `mnfctur_de_or_fallback`.
- 시계열 13종 × 160열: tmprt_150 / tmprt_150Top / tmprt_150Lwet(기온·최고·최저), hd_150(습도), arvlty_300 / Top(풍속), afp(강수), 일조, solrad_Qy(일사), soil_Mitr_10(토양수분), 미세먼지, 곰팡이생장률, 독소생성률(아플라톡신 전용 파생).
- **`_00` = 기준일 당일, `_159` = 159일 전** (검증 완료). 같은 (wx_key, 기준일)이면 시계열 100% 동일 → (wx_key, 날짜) 롱테이블로 복원 가능.
- 지점 좌표는 파일에 없음. wx_key별 최소 wx_dist_km 샘플 좌표로 근사 가능(domestic 오차 median 2.25km).
- 리스테리아 df(listeria_body.parquet)에 최근접 지점+날짜창으로 붙일 때 160일 창 100% 커버율: domestic 92.9% (2015년 4%, 2016년 62%, 2018년 이후 90%+), overseas 8.5%.
- 아플라톡신 전용 컬럼(곰팡이생장률, 독소생성률, 독소_생성_최적_온도_일수, 저습도_일수, aw_grade, 토양산도 등)은 리스테리아에 그대로 가져오면 안 됨.

관련: [[listeria-variable-mapping]]
