## 바뀐 것
`리스테리아ML2/exp/최종실험_계획_20261102.md`

- **해외 기상**: Open-Meteo 로 채우지 않고 DB `global_weather_info` 값만 씁니다. DB 에 없는 변수·날짜는 NaN 이고, 1차 기상 = 2차 기상입니다.
- **해외 구조적 결측**: 해외 행 전부에서 결측인 변수(지중온도 4·soil_t_nas·no2·co·so2·풍향 sin·cos·이슬점)는 빼지 않고 '해외 구조적 결측' 1칼럼으로 표시합니다.
- **2-0 판**: 기상이 1차와 같아 1차 A 판을 그대로 씁니다(다시 학습 안 함). 학습 합계 1,122~1,531, 예상 15,415회·최악 149,900회.
- **단계 0 확인 사항**
  - 국내 ASOS 295 의 2025-06-14 avg_td(79.7, 그날 avg_ta 21.0 보다 큼) 한 칸 결측
  - Theil's U 대상에서 `mnfctur_cmpny_nm` 뺌(검사 결과에 따라 채워지는 칼럼)
  - 공개 지연 반영 안 함
  - threshold 의 Recall ≥ 0.8 판정은 반올림 없이, 배수 분모는 PR-AUC 가 NaN 인 h 뺌

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01NGsi4zpqEU3pbmvLcD9eWn
