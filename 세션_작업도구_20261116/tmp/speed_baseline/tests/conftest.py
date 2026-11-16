"""시험 공통: 실행 범위(lst.scope — 국내만 버전)는 프로세스 전역 상태라 시험마다 기본(지금 판)으로 되돌림."""
import pytest


@pytest.fixture(autouse=True)
def _reset_scope():
    from lst import scope

    scope._reset_for_tests()
    yield
    scope._reset_for_tests()
