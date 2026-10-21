# H60 서버 실행 근거 (2026-10-19)

H60(학습 행마다 h 1~60 전부, 학습 행 13,367,580)은 이 PC 메모리(15.9GB)로 안 돼 팀 공용 서버(RAM 94GB, CPU 16코어, GPU 안 씀)에서 돌렸다.
서버 주소·계정은 적지 않는다(천영진/GPU 서버 사용법.md).

- `E00_s42_재현_md5.txt`: 같은 코드·자료로 서버에서 E00 시드 42 를 먼저 돌려 이 PC 결과와 비교 — 예측 파일(pred_val·pred_test) md5 가 같고(예측값 비트 단위 동일), 최적/학습 라운드 38/138 도 같음.
- `code_md5.json`: 서버에 복사한 코드 35개 파일의 md5(복사 시점 = 이 PC 와 같음). H60 결과 config.json 의 code_md5 는 이 판 기준.
- `data_md5.json`: 서버에 올린 자료 2개(시료_시군구.parquet, 시군구_일별_기상.parquet)의 md5 = 이 PC 와 같음. 실행 뒤 서버에서 지움.
- `h60_manifest.json`: H60 시드 42·43·44 결과 파일 md5(서버 원본·가져온 판), 계정 경로를 가린 칸(config·summary 의 memmap_file 1곳씩), 5초 간격 메모리 측정(최대 익명 메모리 22.4~24.5GB, 스왑 0).
- 주의: 리눅스에서 돌아 summary.json 의 peak_private_gb 는 가상 메모리 크기(vms)라 이 PC 값과 비교 불가. metrics_by_h.csv 줄바꿈은 LF.
