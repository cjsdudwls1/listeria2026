import sys

p = sys.argv[1]
s = open(p, encoding='utf-8').read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''     fold 범위·M·대상·시료(키·d·src_route)·tag(부른 쪽 — 판·코드 md5 등)의 md5, 다르면 멈춤.
"""''', '''     fold 범위·M·대상·시료(키·d·src_route)·tag(부른 쪽 — 판·코드 md5 등)의 md5, 다르면 멈춤.
 (i) 속도 개선(결과 불변 — 사용자 승인 11-12, 설정 speed.gpu_importance — lst/speed, 꺼져 있으면 예전 길 그대로):
     딥러닝 잎(DLClassifier 확률·AnomalyModel 이상 점수, predict_fn 없음)은 h 마다 그 h 의 입력(그 모델이 쓰는 키)을 모델
     장치(GPU)에 한 번 올리고, 섞은 입력을 R 회 이어 붙여 CPU 에 만드는 대신 예측 묶음(8192 행, 같은 경계)마다 GPU 색인으로
     만듦: 묶음 행 = 원 입력[base_take 구간], 대상 칼럼·채널만 원 입력[sel_take 구간] 값으로(행 사본 — 값 같음). 섞기
     순열(rng·칸·차례)·base_take·sel_take·묶음 순서·크기는 예전과 같음. 입력 점검(NaN·dtype·모양·범주 번호 범위)은 원 입력으로
     한 번(섞은 행은 원 행의 사본). GPU 에 못 올리면(메모리 오류) 그 잎은 예전 길(CPU 이어 붙이기). 트리·그 밖 잎은 CPU 그대로.
"""''')

rep('''class _HEval:
    """h 하나의 평가 쌍 위에서 그래프 예측(섞기 전·섞은 뒤). 조립 객체 출력·섞기 전 예측은 h 마다 한 번."""

    def __init__(self, meta: pd.DataFrame):
        self.meta = meta
        self.n = len(meta)
        self._X: dict[int, Any] = {}
        self._pack: dict[int, _FramePack] = {}
        self._base: dict[int, np.ndarray] = {}
''', '''def _speed_on(flag: str) -> bool:
    from . import speed

    return speed.on(flag)


def _gpu_capable(leaf: "Leaf") -> bool:
    """구현 세부 (i): GPU 색인 길을 쓸 수 있는 잎(배열 dict 입력의 딥러닝 확률·이상 점수, predict_fn 없음)."""
    if leaf.predict_fn is not None or leaf.get_layout().kind != "dict":
        return False
    m = leaf.model
    if leaf.output == "score":
        return hasattr(m, "score_source") and hasattr(m, "source_keys")
    return leaf.n_iter is None and hasattr(m, "predict_proba_source") and hasattr(m, "source_keys")


class _GPUInput:
    """구현 세부 (i): h 하나·잎 하나의 입력(모델이 쓰는 키)을 모델 장치에 한 번 올려 둠."""

    def __init__(self, leaf: "Leaf", X: Mapping[str, np.ndarray]):
        import torch

        m = leaf.model
        self.n = int(m.check_source_inputs(X))
        self.dev = m.device
        self.torch = torch
        self.t = {}
        for k in m.source_keys():
            self.t[k] = torch.from_numpy(np.ascontiguousarray(X[k])).to(self.dev)

    def fetch_fn(self, sel: Mapping[str, np.ndarray], base_take: np.ndarray, sel_take: np.ndarray):
        torch = self.torch
        bt = torch.from_numpy(np.ascontiguousarray(base_take, dtype=np.int64)).to(self.dev)
        st = torch.from_numpy(np.ascontiguousarray(sel_take, dtype=np.int64)).to(self.dev)
        si = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.int64)).to(self.dev)
              for k, v in sel.items() if k in self.t}
        for k in sel:
            if k in self.t and self.t[k].dim() not in (2, 3):
                raise ImportanceError(f"{k}: 섞을 수 없는 차원 {tuple(self.t[k].shape)}")

        def fetch(s: int, e: int) -> dict:
            out = {}
            for k, a in self.t.items():
                b = a.index_select(0, bt[s:e])                  # 새 텐서(예전 .to(dev) 처럼 새 할당)
                if k in si:
                    d = a.dim() - 1
                    b.index_copy_(d, si[k], a.index_select(0, st[s:e]).index_select(d, si[k]))
                out[k] = b
            return out
        return fetch

    def predict(self, leaf: "Leaf", sel, base_take, sel_take) -> np.ndarray:
        fetch = self.fetch_fn(sel, base_take, sel_take)
        n = int(base_take.shape[0])
        if leaf.output == "score":
            p = leaf.model.score_source(n, fetch)
        else:
            p = leaf.model.predict_proba_source(n, fetch)
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if not np.isfinite(p).all():
            raise ImportanceError(f"{leaf.name}: 예측에 NaN·inf")
        return p


class _HEval:
    """h 하나의 평가 쌍 위에서 그래프 예측(섞기 전·섞은 뒤). 조립 객체 출력·섞기 전 예측은 h 마다 한 번."""

    def __init__(self, meta: pd.DataFrame):
        self.meta = meta
        self.n = len(meta)
        self._X: dict[int, Any] = {}
        self._pack: dict[int, _FramePack] = {}
        self._base: dict[int, np.ndarray] = {}
        self._gpu: dict[int, _GPUInput | None] = {}     # 구현 세부 (i) — None = 못 올림(예전 길)
        self.gpu_fallbacks: list[str] = []

    def gpu_input(self, leaf: "Leaf") -> _GPUInput | None:
        k = id(leaf)
        if k not in self._gpu:
            try:
                self._gpu[k] = _GPUInput(leaf, self.X(leaf))
            except Exception as e:  # torch.cuda.OutOfMemoryError 등 — 예전 길로(값 같음)
                if type(e).__name__ != "OutOfMemoryError":
                    raise
                self._gpu[k] = None
                self.gpu_fallbacks.append(leaf.name)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # pragma: no cover
                    pass
        return self._gpu[k]

    def release(self) -> None:
        self._gpu.clear()
''')

rep('''        if isinstance(node, Leaf):
            sel = node.get_layout().select(t)
            if not sel:
                out = (np.tile(self.base(node), R), False)
            else:
                out = (node.predict(self._take(node, sel, base_take, sel_take)), True)
''', '''        if isinstance(node, Leaf):
            sel = node.get_layout().select(t)
            if not sel:
                out = (np.tile(self.base(node), R), False)
            else:
                g = self.gpu_input(node) if _speed_on("gpu_importance") and _gpu_capable(node) else None
                if g is not None:
                    out = (g.predict(node, sel, base_take, sel_take), True)
                else:
                    out = (node.predict(self._take(node, sel, base_take, sel_take)), True)
''')

rep('''                for j, r in enumerate(rs):
                    ph[ti, r - 1, h - 1] = mt.pr_auc(y, pred[j * n:(j + 1) * n], u)
        del E
''', '''                for j, r in enumerate(rs):
                    ph[ti, r - 1, h - 1] = mt.pr_auc(y, pred[j * n:(j + 1) * n], u)
        if E.gpu_fallbacks and progress is not None:
            progress(f"h {h}: GPU 에 못 올린 잎(예전 길로) {sorted(set(E.gpu_fallbacks))}")
        E.release()
        del E
''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print("ok")
