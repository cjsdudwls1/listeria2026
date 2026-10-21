# -*- coding: utf-8 -*-
"""딥러닝(DL) 실험 등록 지점. 구현은 e2dl_impl.py (DL_lstmA, DL_trfA, DL_lstmB, DL_trfB 등록). 그 파일이 없으면
레지스트리가 비어 run.py 가 DL 실험을 '미구현'으로 건너뛴다.

등록 방법
---------
    from e2dl import register_dl

    @register_dl("DL_lstmA")
    def make_lstmA(ctx, exp, seed, log, smoke):
        return MyRunner(...)

실행기(runner)가 갖춰야 할 것 (e2models.CatRunner 와 같은 약속, run.py 가 이것만 씀)
  fit()                     학습(조기 종료 포함). 시드 = seed (초기값·셔플·h 배정 모두)
  predict(sp, h) -> ndarray 분할 sp('val'|'test')의 대상 행 전부(ctx.sp[sp] 순서, 묶음 필터 없음)에 대해
                            앞당김 h 의 검출 확률 (n,) float32. 방식 B 는 T = d - h 입력의 출력[h] 를 돌려줌
  info() -> dict            config.json 의 model_info 로 저장 (구조·표준화 통계·에폭 기록 등)
  close()                   GPU 메모리 해제
  속성: pos (dict: 분할 -> ctx.sp[sp] 안 위치 배열, 묶음 필터가 없으면 np.arange(n)),
        best_rounds(최적 에폭), n_rounds_trained, best_val_ap(조기 종료용 검증 PR-AUC), fit_seconds,
        feature_seconds, n_train_rows, n_train_pos, n_features, n_features_desc, cols(없으면 None)

쓸 수 있는 문맥(ctx = e2data.Ctx2)
  ctx.weather.W (시군구, 날짜, 19) float32 원값, ctx.weather.train_period_stats() (1차 LSTM 과 같은 표준화 통계)
  ctx.sp[sp]: s(시군구 번호), d(기상 날짜 번호), y, lstm_cats(학습에 없던 범주 = 0), unit(단위 번호) ...
  ctx.lstm_cat_sizes, ctx.h_fit(seed) -> {'train': h, 'val': 조기 종료용 h}
  ctx.lookup_y(unit, wday) -> (단위, 날짜) 단위x날짜 행의 y (없으면 -1). 방식 B 정답 벡터용
  1차 LSTM 구현: 1_베이스라인_lims기상/src/models.py LSTMRunner, make_lstm_net (import models as MD1)
설정값: e2cfg.DL
"""

DL_REGISTRY = {}


def register_dl(exp_id):
    def deco(factory):
        DL_REGISTRY[exp_id] = factory
        return factory
    return deco


# 다음 에이전트: 구현 파일 e2dl_impl.py (같은 src 폴더)를 만들고 그 안에서 @register_dl 로 등록하면
# 아래 import 로 자동 등록된다. 파일이 없으면 아무것도 등록되지 않음(run.py 가 DL 을 '미구현'으로 건너뜀).
try:
    import e2dl_impl  # noqa: F401,E402
except ModuleNotFoundError as _e:
    if _e.name != "e2dl_impl":
        raise
