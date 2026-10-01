# 실험 2 — 다운사이징·변수 정리 데이터로 AutoGluon(탐색·배깅·스태킹) + 논문 딥러닝 구조

작성 2026-09-29. 실험 1(`EXP/1`, MLflow `리스테리아_EXP1`)에 이어, 사용자 지시(2026-09-29)대로 데이터를 줄이고 변수를 정리한 뒤 튜닝까지 한 성능을 잰다. MLflow 실험 이름 `리스테리아_EXP2`.

## 환경

- **Python 3.12 가상환경 `리스테리아ML/.venv-exp2`** (AutoGluon 이 Python 3.14 를 지원하지 않아 따로 만듦, `uv` 로 설치). autogluon.tabular 1.6.3 · torch 2.13.0+cu126 · optuna 5.0.0 · mlflow 3.16.1 · lightgbm 4.7.0 · catboost 1.2.10 · xgboost 3.3.0(GPU 빌드로 교체, AutoGluon 기본은 xgboost-cpu) · scikit-learn 1.9.1 · pandas 2.3.3.
- Jupyter 커널 이름 `listeria-exp2` ("listeria-exp2 (py3.12)"). EXP/2 노트북은 이 커널로 연다. 실험 1 노트북은 그대로 Python 3.14 커널.
- GPU RTX 2060 6GB: AutoGluon 의 CatBoost·XGBoost, 딥러닝에 사용. LightGBM·RF·XT 는 CPU. (CatBoost 를 CPU 로 하면 360여 피처 × 44만 행에서 설정 하나에 수십 분이 걸려 GPU 로 함 — GPU 학습은 같은 seed 로 다시 돌리면 조금 달라질 수 있음)
- 실행 전 메모리: PC RAM 15.9GB 중 다른 프로그램(DataGrip·VS Code·브라우저 등)이 약 11GB 를 쓰고 있어, 본 실행 전에 안 쓰는 프로그램을 닫아 두는 것이 안전하다.

## 데이터 (`1_데이터준비.ipynb` → `data/exp2/`, git 제외)

| 단계 | 결과 |
|---|---|
| 출발 | 실험 1 입력: `data/X.parquet`(676,537행) + `exp1_wx/`(기상·대기 17변수 × D-1~D-100, 식품 3) |
| 다운사이징 | **전체 기간(train·val·test)에 검출 0 인 품목명(`train_prdlst_nm`) 38종의 행 삭제**(76,497행) → 600,040행. 2026-10-01 사용자 지시로 바꿈 — 이전에는 train 만 보고 정한 `train_prdlst_code` 37종(76,462행) → 600,075행이었고, 차이는 train 에 행이 없는 `환경검체`(`Z4100000000000`, val 21·test 14행, 검출 0) 35행. 새 기준은 품목 목록을 정할 때 val·test 의 검출 여부도 본다 |
| 변수 제거 | `lat`·`lon`·`legal_dong_code`·`train_prdlst_nm`(성능향상_방안 §3-1·3-3); `jdgmnt_fom_code` 는 입력에 없어 해당 없음(§3-4); 기상 하루값 1,700칼럼 → 창 집계(§3-2) |
| 기상 창 집계 | 창 7·14·30·100일 × 평균·최고·최저·표준편차 + 강수 합·강수일·최장 무강수 + 생육 범위 일수(기온 ≥30℃·30~37℃·최저 ≥−2℃, 토양온도 8~15·8~30℃) + 결측 일수 = **353칼럼** |
| 최종 피처 | **364개** = 범주형 8(LIMS 5 `src_route`·`prdlst_code`·`train_prdlst_code`·`date_basis`·`addr_source` + 식품 값 표기 3 `aw_label`·`ph_label`·`na_label`) + 식품 3(`aw`·`ph`·`na`) + 기상 집계 353. 2026-10-01 재생성: 검사기관 코드 `inspct_instt_code` 제거(예측 시점에 알 수 없는 값), 식품 값 표기 추가, 로마자로 적힌 한국 주소 78행에 국내 기상 반영. 2026-09-29 본 실행(362피처, 검사기관 포함)의 MLflow 기록은 `리스테리아ML/mlflow_backup_20261001/` |
| 딥러닝 시계열 | `seq_std.npy` (600,040 × 100일 × 17변수, float16, train 통계 표준화, 결측 0, 시간 순서 D-100→D-1) |
| 분할 | train 440,116(검출 1,491) · val 78,420(191) · test 81,504(287) — 날짜 경계는 실험 1과 같음 |

## 노트북

| 파일 | 내용 | 예상 시간 |
|---|---|---|
| `1_데이터준비.ipynb` | 위 표. 실행 완료(2026-09-29) | 1분 |
| `2_autogluon.ipynb` | 가족 5종(LightGBM·CatBoost·XGBoost·RandomForest·ExtraTrees)을 각각 **Optuna(TPE, 베이지안)** 로 탐색 — 0회차 기본 설정 → 가족별 시간 예산(LightGBM 1.5h·CatBoost 1h·XGBoost 1.5h·RF 1h·XT 1h, 가족당 최대 80회) → 가족별 최적 5개 + **가중 앙상블**(가중치는 val 로). CatBoost·XGBoost GPU, 나머지 CPU. (AutoGluon 내장 HPO 는 Ray 가 이 PC 에서 안 떠서 못 씀 → Optuna 가 설정을 제안하고 AutoGluon `fit_extra` 가 학습·채점.) **배깅·스태킹은 하지 않음** — out-of-fold 분할이 시간 순서를 어기므로(2026-09-29 사용자 결정) | 약 7시간 |
| `3_딥러닝.ipynb` | 논문(`0808Manuscript.2.docx` Table 2) Conv1D–BiLSTM–Attention + 표 형식 가지 → `DL_논문설정`(논문 값 그대로) + `DL_튜닝`(Optuna 최대 25회 / 6시간, 범위 넓힘: Conv 층 수·필터·커널·LSTM·Dense·배치·증강·창 길이) | 약 4~7시간 |
| `run_all.py` | 위 두 노트북을 순서대로 실행하고 출력을 노트북에 저장 | |

실행: `cd 리스테리아ML` → `.venv-exp2\Scripts\python.exe EXP\2\run_all.py`. 진행 로그 `output/exp2/run_all_*.log`.

## 공통 규칙

- test(2025-07-01~2026-06-30)는 마지막 평가에만. val 은 조기 종료·탐색 채점·앙상블 가중치·임계값 선택에 쓴다(논문과 같은 방식; val 지표는 낙관적일 수 있음).
- 임계값 규칙 0.5 / val F1최대 / val F2최대, 지표는 실험 1과 같은 이름(`test_pr_auc` 등, 비율은 0~1 소수).
- 불균형: AutoGluon 은 재표집·가중 없음(§5-1), 딥러닝은 논문대로 가중 BCE + 양성 3배 증강(탐색에서는 선택).
- MLflow: 부모 run = 노트북 이름, 자식 run = `노트북 / 단계-모델 / 규칙`, param `input` 에 입력 설명.

## 이번 실험에 넣지 않은 것 (다음 단계 후보, 성능향상_방안 §7)

업소·품목·기관 이력 피처, 제품명 키워드, LIMS 검사목적 코드(§2-1~2-5 — 탐색 점검에서 가장 큰 이득), 묶음 단위 평가, 환경검체·수출 팽이버섯 범위 결정.
