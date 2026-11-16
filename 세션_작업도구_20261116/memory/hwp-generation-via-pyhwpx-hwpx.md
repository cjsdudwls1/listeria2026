---
name: hwp-generation-via-pyhwpx-hwpx
description: "한컴 .hwp 첨부 문서를 참고 파일 서식 그대로 만드는 검증된 방법 — pyhwpx(COM)로 hwp↔hwpx 변환, section0.xml 만 생성, linesegarray 는 넣지 말 것, PDF→PNG 로 검수"
metadata: 
  node_type: memory
  type: reference
  originSessionId: f89b6aa8-85da-451c-98f6-ac55fb436f27
  modified: 2026-09-22T06:14:04.801Z
---

2026-09-22 모델팀 월간보고 첨부용 `모델팀_공유문서/리스테리아_9월보고서_첨부용.hwp` 를 만들 때 확립한 절차.

- 환경: 한컴오피스 2020(`HWPFrame.HwpObject` COM) 설치됨. `py -m pip install pyhwpx pywin32` 로 pyhwpx 를 넣으면
  `FilePathCheckerModule.dll` 이 같이 들어와 파일 접근 승인 대화상자 없이 `Open/SaveAs` 가 된다
  ("RegisterModule 액션을 실행할 수 없음" 경고가 찍혀도 동작함). `hwp = Hwp(new=True, visible=False); com = hwp.hwp`,
  `com.Open(path, "HWP", "forceopen:true")`, `com.SaveAs(path, "HWPX"|"HWP"|"PDF", "")`, `com.GetTextFile("TEXT", "")`, `com.Quit()`.
- 절차: 참고 .hwp → SaveAs HWPX → zip 풀어 `Contents/header.xml`(글자·문단·테두리/음영 스타일)은 그대로 두고
  `Contents/section0.xml` 만 파이썬 f-string 으로 생성(같은 charPrIDRef/paraPrIDRef/borderFillIDRef 재사용) → zip(mimetype 을 첫 항목·무압축) →
  COM 으로 열어 SaveAs "HWP". 생성기: 작업 임시 폴더 `build_listeria_hwp.py`(job 삭제 시 사라짐; 구조는 이 메모대로 재작성 가능).
- **`hp:linesegarray` 는 넣지 않는다.** 넣으면 한컴이 그 줄배치 캐시를 믿어 여러 줄 문단이 한 줄로 겹쳐 인쇄된다. 빼면 열 때 직접 계산한다.
- 검수: SaveAs "PDF" 뒤 `pymupdf`(`py -m pip install pymupdf`)로 `page.get_pixmap(dpi=110).save(png)` → Read 로 눈으로 확인.
  숨은 COM 인스턴스가 멈추면 `Get-Process Hwp | where MainWindowTitle -eq ''` 만 골라 종료(사용자 창은 제목이 있음).
- 참고 파일(황색포도상구균 첨부용) 스타일 ID: 제목 □ charPr 8 + 제목글 charPr 9(빨강 16pt) paraPr 20 / 빈 줄 paraPr 21 charPr 10 /
  ㅇ 문장 paraPr 22 (charPr 11 " ㅇ" + 12 본문 15pt) / - 문장 paraPr 23 charPr 13 / 표 셀 문단 paraPr 24(가운데) 본문 charPr 1(10pt)·굵게 14 /
  캡션 charPr 15(12pt, side=TOP) / 표1 머리 borderFill 7(#D9D9D9)·본문 4, 표1 문단 paraPr 26 styleIDRef 22 /
  표2 머리 borderFill 5(#D9D9D9)·합계 행 6(#F2F2F2), 표2 문단 paraPr 23. 표 폭 46841(표1)·47621(표2) HWPUNIT, 셀 여백 510/141.

- 본문 문단이 쪽 경계에서 갈라지면(한 줄만 앞 쪽에 남는 현상) header.xml 의 해당 `hh:paraPr` 블록을 복제해 새 id 를 주고
  `keepLines="1"`(문단 보호)만 켠 뒤 `hh:paraProperties itemCnt` 를 늘리면 된다(2026-09-22 문단 모양 27·28 로 적용, 글꼴·간격 동일).
- Bash 도구의 heredoc 은 이중 백슬래시를 한 개로 줄여, 파이썬 일반 문자열 안의 백슬래시-b·백슬래시-d 가 깨진다(백스페이스 문자가 들어감). 정규식이 있는 패치는 Write 도구로 파일을 쓰거나 raw 문자열로 둘 것.

**How to apply:** 다음 달 보고서·다른 위해요소 첨부용을 만들 때 같은 생성기 구조를 쓰고, 내용만 바꾼다. 참고 파일 2쪽 캡션 "<예비용>" 은 원저자 잔재라 복제하지 않았다.
