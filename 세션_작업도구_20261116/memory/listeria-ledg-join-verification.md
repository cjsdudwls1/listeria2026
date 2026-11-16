---
name: listeria-ledg-join-verification
description: "공유용 데이터셋 ledg_*(품목제조대장) 조인 검증 결과(2026-09-17) — 키·규칙은 맞음, 번호 재사용·대장번호 중복·이력 표 없음·컬럼사전 문구 누락"
metadata: 
  node_type: memory
  type: project
  originSessionId: e6768a4a-5d06-4ed7-9bf0-88f31983aa0a
  modified: 2026-09-28T04:29:47.610Z
---

2026-09-17 사용자가 "보고서에 틀린 걸 쓸 수 없다"며 ledg_* 9개 컬럼 조인 검증을 요청. DB 접속 불가라 09-14 추출본(02b_ledg = DB수집/t_prdlst_mnf_ledg.parquet)으로 검증.
- 경로: 시료 lims_data_key → ready_data.hazard_preprocessed_data.prdlst_mnfctur_sttemnt_no → t_prdlst_mnf_ledg.prdlst_report_no (원문, 없으면 숫자 아닌 글자 전부 제거). prdlst_report_ledg_no 와는 0개 일치라 키 선택 맞음.
- 독립 재구현 = 공유 parquet 9열 불일치 0. 번호 있는 국내 행 99.70% 연결(원문 202,926 + 숫자만 570). 제품명 같거나 포함 95%(숫자만 93.7%).
- 한계: 같은 번호 여러 행 689개(시료 2,771행), 번호 재사용 사례 있음(예 2013044800111), prdlst_report_ledg_no 까지 중복 6개 번호·완전 동점 2개(참치마요 유부초밥 유통기한 9/6개월 → 파일 순서), 대장 이력 표 없음 → 값은 추출 시점 상태, last_updt_dtm 은 일괄 수정일(2024-01-17) 많음. 날짜 규칙(prms_dt≤기준일)은 제품명 일치율이 현재 규칙보다 낮음(95.8 vs 96.7%).
- '숫자만 일치' 변환이 식정원 번호 체계인지는 미검증 → 최종 데이터셋 확정 시 알림 [[pending-ledg-digit-number-verification]].
- 컬럼사전 '연결 방법' 문구에 번호 출처 표·완전 동점·시점·빈 값 이유가 빠져 있음(수정은 사용자 확인 후).
- 기록: 브랜치 worktree-listeria-ledg-join-verify 의 docs/00_보고서/작업기록.md + scripts/share/verify_ledg_join_1·2.py (메인 체크아웃 작업기록.md 는 비어 있어 사용자가 옮겨야 함).

**2026-09-28 pog_daycnt 조인(노트북 6-2, 현재 DB, 사용자 승인 규칙)**: 입력 `data/listeria_df_joined_full.parquet`. 사용자 선택 = **번호 원문 일치만**(숫자만·변경전 번호 안 씀) + **대장에 같은 번호 여러 행이면 연결 안 함(결측)** + 노트북·보고만(파일 저장 없음). 사실: ready_data 번호 = m_org_lims_data_info 번호(669,334행 전부 동일, 나머지 7,203 = 수입), 번호 있음 204,115행(국내만, 고유 35,504), 대장 1행 34,401·여러 행 688(시료 2,770)·없음 415(시료 1,189, 그중 844행은 하이픈·'즉석판매제조가공업'·'-'). pog_daycnt 결측 71.67%(① 번호 없음 472,422 · ② 1,189 · ③ 2,770 · ④ NULL 8,525), 값 있음 191,631 중 99.2% 가공식품, 검출 결측 88.78% vs 불검출 71.62%. 값은 문장(고유 6,984, '제조일로부터 12개월' 등). 상세 작업기록 2026-09-28.

**Why:** 보고서 제1절 데이터 구축·품질 점검에 이 조인 설명이 들어감.
**How to apply:** DB 복구 후 m_org 번호 동일성·org_prdlst_report_no 로 미연결 494행 확인. 규칙 변경(제품명으로 행 고르기 등)은 사용자 결정. 관련 [[listeria-share-dataset]], [[listeria-db-structure-findings]].
