"""downsize 메모리 판정 시험용 하위 프로세스 함수(spawn 이 import 함)."""


def memory_until_m20(tag, setting):
    """m 이 20 이하가 될 때까지 메모리 부족 흉내."""
    if setting.m is None or setting.m > 20:
        raise MemoryError("Unable to allocate 40.0 GiB")
    return {"tag": tag, "level": setting.level, "m": setting.m, "disk": setting.disk}


def ok_first(tag, setting):
    return {"tag": tag, "level": setting.level}


def value_error(tag, setting):
    raise ValueError("코드 문제")
