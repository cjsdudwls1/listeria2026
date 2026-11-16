import sys

p = sys.argv[1]
s = open(p, encoding='utf-8').read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''17. GPU 상한: `set_gpu_fraction(f)` 로 낮추면 학습·불러오기 때 그 값(없으면 공통 0.4) — 스크립트의 gpu_memory_fraction 이
    모델 만들 때 0.4 로 다시 올라가지 않게.
"""''', '''17. GPU 상한: `set_gpu_fraction(f)` 로 낮추면 학습·불러오기 때 그 값(없으면 공통 0.4) — 스크립트의 gpu_memory_fraction 이
    모델 만들 때 0.4 로 다시 올라가지 않게.
18. 속도 개선(결과 불변 — 사용자 승인 11-12, 설정 speed.fast_predict — lst/speed, 꺼져 있으면 예전 길 그대로): 예측
    (predict_proba)·이상 점수(raw_score)의 묶음을 a[np.arange(s, e)] 복사 대신 연속 구간 a[s:e] 를 고정 메모리(pinned)
    완충 배열에 옮겨 GPU 로(비동기 — 다음 묶음 전에 그 복사가 끝났는지 기다림). 묶음 경계·크기(8192, 메모리 오류 때 절반씩)
    ·순서는 그대로라 GPU 에 올라간 값·계산이 같음. CPU 장치면 그 구간의 사본(예전처럼 새 배열).
    순열 중요도용 `predict_proba_source(n, fetch)`·`score_source(n, fetch)` = 묶음을 fetch(s, e)(장치 텐서 dict)로 받는
    같은 예측 고리(lst/importance 구현 세부 (i) — 입력 점검은 부르는 쪽이 원 입력으로 `check_source_inputs`).
"""''')

rep('''def _state_md5(idx: np.ndarray) -> str:''', '''def _speed_on(flag: str) -> bool:
    from . import speed

    return speed.on(flag)


class _SliceSource:
    """구현 세부 18: 연속 구간 묶음 → 장치 텐서 dict. CUDA 면 키마다 고정 메모리 완충 배열(PRED_BATCH 행) 하나를 돌려 씀."""

    def __init__(self, inputs: Mapping[str, np.ndarray], dev: torch.device):
        self.inputs = {k: np.asarray(v) for k, v in inputs.items()}
        self.dev = dev
        self.cuda = dev.type == "cuda"
        self.buf: dict[str, torch.Tensor] = {}
        self.event = None

    def __call__(self, s: int, e: int) -> dict:
        m = e - s
        if not self.cuda:
            return {k: torch.from_numpy(np.array(a[s:e], copy=True)) for k, a in self.inputs.items()}
        if self.event is not None:
            self.event.synchronize()            # 앞 묶음의 비동기 복사가 끝난 뒤에 완충 배열을 다시 씀
        out = {}
        for k, a in self.inputs.items():
            b = self.buf.get(k)
            if b is None or b.shape[0] < m:
                b = self.buf[k] = torch.empty((max(m, PRED_BATCH),) + tuple(a.shape[1:]),
                                              dtype=torch.from_numpy(a[:0]).dtype, pin_memory=True)
            bv = b[:m]
            bv.numpy()[...] = a[s:e]
            out[k] = bv.to(self.dev, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record()
        return out


def _state_md5(idx: np.ndarray) -> str:''')

rep('''    @torch.no_grad()
    def _eval_rows(self, data: Mapping[str, np.ndarray], y: np.ndarray | None, idx: np.ndarray,
                   which: int) -> np.ndarray:
        """eval 모드 행별 값(which 0 = 학습 손실 행, 1 = 감시 손실 행)."""
        self.net.eval()
        dev = self.device
        out = np.empty(len(idx), dtype=np.float64)
        step = PRED_BATCH
        s = 0
        while s < len(idx):
            c = idx[s:s + step]
            try:
                batch = _to_dev(data, c, dev)
''', '''    @torch.no_grad()
    def _eval_rows(self, data: Mapping[str, np.ndarray] | None, y: np.ndarray | None, idx: np.ndarray,
                   which: int, fetch=None) -> np.ndarray:
        """eval 모드 행별 값(which 0 = 학습 손실 행, 1 = 감시 손실 행). fetch(s, e) 를 주면 idx = arange(n) 이고
        묶음을 그것으로 받음(구현 세부 18 — 묶음 경계 같음)."""
        self.net.eval()
        dev = self.device
        out = np.empty(len(idx), dtype=np.float64)
        step = PRED_BATCH
        s = 0
        while s < len(idx):
            c = idx[s:s + step]
            try:
                batch = fetch(s, s + len(c)) if fetch is not None else _to_dev(data, c, dev)
''')

rep('''        n = self._check(inputs)
        if self._dims_of(inputs) != self.dims:
            raise ValueError(f"입력 모양 {self._dims_of(inputs)} ≠ 학습 때 {self.dims}")
        self.net.eval()
        dev = self.device
        out = np.empty(n, dtype=np.float64)
        step, s = PRED_BATCH, 0
        while s < n:
            c = np.arange(s, min(n, s + step))
            try:
                b = _to_dev(inputs, c, dev)
                logit = self.net(b["x_seq"], b["x_static"], b.get("x_cat"))["logit"]
''', '''        n = self.check_source_inputs(inputs)
        dev = self.device
        if _speed_on("fast_predict"):
            return self._predict_loop(n, _SliceSource(inputs, dev))
        return self._predict_loop(n, lambda s, e: _to_dev(inputs, np.arange(s, e), dev))

    def check_source_inputs(self, inputs: Mapping[str, np.ndarray]) -> int:
        """predict_proba 의 입력 점검(행 수 돌려줌) — 중요도 GPU 길은 원 입력으로 한 번 부름(섞은 행 = 원 행의 사본)."""
        if self.net is None:
            raise RuntimeError("학습 전")
        n = self._check(inputs)
        if self._dims_of(inputs) != self.dims:
            raise ValueError(f"입력 모양 {self._dims_of(inputs)} ≠ 학습 때 {self.dims}")
        return n

    def source_keys(self) -> list[str]:
        return self._keys[0]

    @torch.no_grad()
    def predict_proba_source(self, n: int, fetch) -> np.ndarray:
        """구현 세부 18: 묶음 fetch(s, e) → 장치 텐서 dict 로 예측(predict_proba 와 같은 고리·묶음 경계)."""
        if self.net is None:
            raise RuntimeError("학습 전")
        return self._predict_loop(int(n), fetch)

    def _predict_loop(self, n: int, fetch) -> np.ndarray:
        self.net.eval()
        dev = self.device
        out = np.empty(n, dtype=np.float64)
        step, s = PRED_BATCH, 0
        while s < n:
            c = np.arange(s, min(n, s + step))
            try:
                b = fetch(s, s + len(c))
                logit = self.net(b["x_seq"], b["x_static"], b.get("x_cat"))["logit"]
''')

rep('''        if self._dims_of(inputs) != self.dims:
            raise ValueError(f"입력 모양 {self._dims_of(inputs)} ≠ 학습 때 {self.dims}")
        return self._eval_rows({self.key: inputs[self.key]}, None, np.arange(n), which=0)

    def score(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        return robust_z(self.raw_score(inputs), self.z_params)
''', '''        if self._dims_of(inputs) != self.dims:
            raise ValueError(f"입력 모양 {self._dims_of(inputs)} ≠ 학습 때 {self.dims}")
        data = {self.key: inputs[self.key]}
        if _speed_on("fast_predict"):
            return self._eval_rows(None, None, np.arange(n), which=0, fetch=_SliceSource(data, self.device))
        return self._eval_rows(data, None, np.arange(n), which=0)

    def score(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        return robust_z(self.raw_score(inputs), self.z_params)

    def check_source_inputs(self, inputs: Mapping[str, np.ndarray]) -> int:
        """raw_score 의 입력 점검(행 수) — 중요도 GPU 길(구현 세부 18)."""
        if self.net is None:
            raise RuntimeError("학습 전")
        n = _check_inputs(inputs, [self.key], [k for k in ("x_seq", "x_static", "x_cat", "x_tab") if k != self.key])
        if self._dims_of(inputs) != self.dims:
            raise ValueError(f"입력 모양 {self._dims_of(inputs)} ≠ 학습 때 {self.dims}")
        return n

    def source_keys(self) -> list[str]:
        return [self.key]

    def score_source(self, n: int, fetch) -> np.ndarray:
        """구현 세부 18: 묶음 fetch(s, e) 로 이상 점수(score 와 같은 고리·묶음 경계)."""
        if self.net is None:
            raise RuntimeError("학습 전")
        return robust_z(self._eval_rows(None, None, np.arange(int(n)), which=0, fetch=fetch), self.z_params)
''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print("ok")
