---
name: listeria-sample-file-changed-1012
description: "기준 데이터(시료 파일)는 리스테리아ML/data/listeria_data.parquet (2026-10-12 11:40 부터, 676,520행×25칼럼) — 같은 날 listeria_body → listeria_dataset → listeria_data 로 두 번 바뀜"
metadata:
  node_type: memory
  type: project
  originSessionId: f2f9a184-5cc7-40fb-897e-a268413655de
  modified: 2026-10-12T02:45:55.486Z
---

**지금 기준 데이터 = `C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_data.parquet`** (사용자 지시 2026-10-12 11:40 "기준데이터 이걸로 변경"). 다른 메모리의 "시료 파일 `listeria_body.parquet`"·"`listeria_dataset.parquet`" 은 이 파일로 읽는다 — 좌표 규칙 [[feedback-sample-file-coords]] 도 이 파일의 `lat`·`lon` 기준.

- 676,520행 × 25칼럼. `src_route` domestic 628,878 / overseas 47,642. 검출 1,969(국내 1,844·해외 125). 기준일 `mnfctur_de` 2015-04-07 ~ 2026-06-30. `train_prdlst_code`·`train_prdlst_code_2` 결측 130행.
- 칼럼은 `listeria_body.parquet` 23개 + `train_prdlst_code_2`·`train_prdlst_nm_2`(교수님 메모: 하위 품목군 코드 = 기존 품목군보다 한 단계 아래, 하위 단계가 없으면 기존 값 그대로).
- `listeria_body.parquet`(676,537행, 같은 폴더에 남아 있음) 대비 17행 빠짐, 새 키 0, 행 순서 다름.

**같은 날의 변천**: 09:05 `listeria_dataset.parquet`(676,531행 — 교수님 Listeria_05 노트북이 Colab 에서 읽던 파일과 같은 행 수) → 11:40 `listeria_data.parquet`(11행 적음)로 교체, `listeria_dataset.parquet` 은 폴더에서 사라짐(사본 없음 — 두 파일의 값 차이는 확인 못 함). listeria_dataset 기준으로 잰 옛 수치(body 대비 train 라벨 10,070행·좌표 290행 변경, 필터 뒤 499,303행 등)는 참고만.

**How to apply:** 사용자가 작업 도중 기준 파일을 바꾸고 옛 파일을 지우므로, 에이전트를 띄우기 전에 `리스테리아ML/data/` 목록을 다시 확인한다. 데이터셋 구축([[listeria-ml2-dataset]])과 교수님 노트북 실행 모두 이 파일을 쓴다 — [[listeria-1012-decisions]].
