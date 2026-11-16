import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
def rep(old, new, n=1):
    global s
    c = s.count(old); assert c == n, (c, old[:70]); s = s.replace(old, new)
rep("15,419 — 실제 바퀴 수는", "15,415 — 실제 바퀴 수는")
rep("와 이슬점 td1(2차)은 해외 행에서 늘 결측", "와 이슬점 td1·rg 지중온도 6(2차)은 해외 행에서 늘 결측")
rep("섞거나 빼지 않고, soil_t5~t30·soil_t_nas·no2·co·so2·풍향 sin·cos·td1 이 모두 빠지면 같이 뺌",
    "섞거나 빼지 않고, 그 묶음(§4-2 — rg 지중온도 6 포함)의 칼럼이 모두 빠지면 같이 뺌")
rep("- 설치·지원 확인: SPI 계산 패키지, CatBoost Focal, AutoKeras(Keras 3·백엔드 포함), mlflow(로컬·서버 같은 판 — 판은 단계 0 에서 위 서버 venv 목록에 적음). CatBoost `has_time` 기본값(False)",
    "- 설치·지원 확인(단계 0): SPI = 자체 구현(scipy 1.18.1 `scipy.special`), CatBoost Focal 동작, AutoKeras 3.0.0(keras 3.15.1·백엔드 torch, tensorflow 2.21.0 은 import 용) — Keras Dense 에 units 가 numpy 정수로 넘어와 멈추므로 쓰기 전 int 로 바꾸는 패치(원 서명 유지), 확률은 `export_model().predict`, mlflow 3.16.1(로컬·서버 같은 판). CatBoost `has_time` 기본값(False)")
rep("(복사 방법은 단계 0 에서 시험해 정함)", "(서버 `mlflow server` = [IP 가림]:5055, 복사 = sqlite3 backup 으로 뜬 mlflow.db 사본 + artifacts 폴더)")
rep("(저장소·artifact 경로는 단계 0 에서 정한 방법대로)", "(`--backend-store-uri sqlite:///<복사한 mlflow.db> --artifacts-destination <복사한 artifacts>`)")
rep("블록 대응·범위는 단계 0-9 AutoKeras 확인 때 단계 8 대상이 될 수 있는 딥러닝·결합 구조 전부(LSTM·GRU·트랜스포머·LSTM + attention·TCN·결합-기본·결합-VAE·⑥ 3종·⑦-D)마다 정해 실험 로그에 고정(단계 1 전에), 대응 블록을 못 만드는 구조는 기록하고 튜닝 안 함",
    "블록 대응(실험 로그에 고정) = LSTM·GRU·LSTM + attention·GRU + attention·결합-기본 → RNNBlock, TCN → ConvBlock(attention·인과 dilation·트랜스포머 1블록은 AutoKeras 블록에 없어 가까운 블록으로), 트랜스포머·결합-VAE·⑦-D = 대응 블록 없음 → 튜닝 안 함, 범위는 단계 0-9 AutoKeras 확인 때 실험 로그에 고정")
rep("÷ 시군구 면적(공개 통계, 구 단위 면적은 단계 0-6 에서 확인) × 10",
    "÷ 시군구 면적(국토교통부 지적통계 2025 행정구역별 국토이용현황 — 2026-02 신설 화성시 만세·효행·병점구는 통계에 없어 경기일보 2026-01-20 보도 면적 466.5·153.5·24.3 km²) × 10")
rep("farm_total, 우리 해외 행의 나라 94개 전부)", "farm_total, 우리 해외 행 나라 95곳 중 KOR 뺀 94곳 — OVKOR = NaN)")
rep("÷ 나라 면적(km², 공개 통계 한 출처로 94개 나라 — 예: World Bank 국토 면적, 출처·기준 연도 기록, 단계 0-6) × 10",
    "÷ 나라 면적(km², FAOSTAT Land area 2024 한 출처 — World Bank 국토 면적엔 대만·쿡 제도가 없음) × 10")
p.write_text(s, encoding="utf-8"); print("ok")
