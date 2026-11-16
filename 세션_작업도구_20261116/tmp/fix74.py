import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
def rep(old, new, n=1):
    global s
    c = s.count(old); assert c == n, (c, old[:80]); s = s.replace(old, new)
def line(prefix, new):
    global s
    L = s.split("\n"); idx = [i for i, x in enumerate(L) if x.startswith(prefix)]
    assert len(idx) == 1, (len(idx), prefix); L[idx[0]] = new; s = "\n".join(L)
# §1-2 배수·동점
rep("| 부적합률 대비 배수 | 구간 가중 PR-AUC ÷ (h 별 검출 비율을 같은 가중으로 합친 값), 60일 통",
    "| 부적합률 대비 배수 | 구간 가중 PR-AUC ÷ (h 별 검출 비율을 같은 가중으로 합친 값 — PR-AUC 가 NaN 인 h 는 분모에서도 뺌), 60일 통")
rep("threshold = Precision → Recall → 큰 값, 판·설정",
    "threshold = Precision → Recall → 큰 값(Recall ≥ 0.8 판정은 반올림 없이), 판·설정")
# 0-3 공개 지연
rep("- 0-3 (공개 지연은 0-4 전에 정함,", "- 0-3 (공개 지연은 반영 안 함(부록 A-13),")
line("- 원천별 공개 지연(단계 0-3):",
     "- 원천별 공개 지연(단계 0-3 에서 확인만): 반영 안 함 — 관측·예보 경계·SPI 이음 위치는 §4-1·§4-5 그대로, 운영 입력 정의는 운영 준비 때")
# A-2 이슬점 한 칸
rep("의 `hum`)를 §4-3 의 3곳 평균 규칙으로",
    "의 `hum`)를 §4-3 의 3곳 평균 규칙으로 — ASOS 지점 295 의 2025-06-14 `avg_td`(79.7, 그날 `avg_ta` 21.0 보다 큼)는 결측으로 바꿔 3곳 평균에서 뺌")
# A-6 Theil's U
rep("`prduct_nm`·`mnfctur_cmpny_nm`·`date_basis`", "`prduct_nm`·`date_basis`")
rep("(검사 결과 전에 정해지는지는 부록 B-6 점검·담당자 확인으로 정함, 시료 파일 25칼럼 중",
    "(`mnfctur_cmpny_nm` 은 검사 결과에 따라 채워지는 칼럼이라 뺌 — 부록 B-6 점검, 시료 파일 25칼럼 중")
p.write_text(s, encoding="utf-8"); print("ok")
