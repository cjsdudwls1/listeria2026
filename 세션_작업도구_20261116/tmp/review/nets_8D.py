class _RNNNet(nn.Module):
    """LSTM·GRU: RNN(64) 마지막 은닉 → 정적·임베딩 concat → Dense 64(relu) → Dense 1.
    hp(구현 세부 16): RNN 폭·층, dropout = 마지막 은닉 뒤·Dense 64 뒤(기본 0 = 없음)."""

    def __init__(self, cell: str, C: int, W: int, tail: _Tail, hp: Mapping | None = None):
        super().__init__()
        hp = _hp(hp, cell)
        H, d = hp["hidden"], hp["dropout"]
        self.rnn = _rnn(cell, C, H, hp["layers"], d)
        self.tail = tail
        self.head = nn.Sequential(nn.Linear(H + tail.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.drop = _drop(d)

    def forward(self, x_seq, x_static, x_cat):
        _, hn = self.rnn(x_seq)
        h = (hn[0] if isinstance(hn, tuple) else hn)[-1]
        z = _cat(self.drop(h), self.tail(x_static, x_cat))
        z = self.drop(self.head[1](self.head[0](z)))
        return {"logit": self.head[2](z).squeeze(1)}


class _TRFNet(nn.Module):
    """트랜스포머 인코더: 선형 투영 64 + 사인 위치 인코딩 → 인코더 2층(헤드 4·FF 128·dropout 0.1)
    → 평균 풀링 → concat → Dense 64(relu) → Dense 1. hp: d_model·층 수·인코더 dropout."""

    def __init__(self, C: int, W: int, tail: _Tail, hp: Mapping | None = None):
        super().__init__()
        hp = _hp(hp, "TRF")
        H = hp["hidden"]
        self.proj = nn.Linear(C, H)
        self.register_buffer("pe", sinusoidal_encoding(W, H), persistent=False)
        self.enc = nn.TransformerEncoder(_encoder_layer(H, hp["dropout"]), num_layers=hp["layers"],
                                         enable_nested_tensor=False)
        self.tail = tail
        self.head = nn.Sequential(nn.Linear(H + tail.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_seq, x_static, x_cat):
        h = self.enc(self.proj(x_seq) + self.pe[: x_seq.shape[1]]).mean(dim=1)
        return {"logit": self.head(_cat(h, self.tail(x_static, x_cat))).squeeze(1)}


class _TCNNet(nn.Module):
    """TCN: 잔차 블록 5개(커널 3·채널 64·dilation 1~16·dropout 0.1) → 마지막 시점 → concat
    → Dense 64(relu) → Dense 1. hp: 채널 폭·블록 dropout(블록 수 고정)."""

    def __init__(self, C: int, W: int, tail: _Tail, hp: Mapping | None = None):
        super().__init__()
        hp = _hp(hp, "TCN")
        H = hp["hidden"]
        blocks, cin = [], C
        for d in TCN_DILATIONS:
            blocks.append(_TCNBlock(cin, H, d, hp["dropout"]))
            cin = H
        self.tcn = nn.Sequential(*blocks)
        self.tail = tail
        self.head = nn.Sequential(nn.Linear(H + tail.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_seq, x_static, x_cat):
        h = self.tcn(x_seq.transpose(1, 2))[:, :, -1]
        return {"logit": self.head(_cat(h, self.tail(x_static, x_cat))).squeeze(1)}


class _AttnRNNNet(nn.Module):
    """LSTM·GRU + attention: RNN(64, 시퀀스) → Dropout 0.3 → MHA(헤드 2, key 32) + 잔차 + LayerNorm
    → Dense 64(relu) → Dropout 0.3 → 잔차 + LayerNorm → 평균 풀링 → Dropout 0.4 → concat → Dense 1.
    hp: 폭(RNN·어텐션·Dense — 잔차라 같은 폭)·RNN 층 수·dropout 한 값(세 자리 — None 이면 0.3·0.3·0.4)."""

    def __init__(self, cell: str, C: int, W: int, tail: _Tail, hp: Mapping | None = None):
        super().__init__()
        hp = _hp(hp, cell + "A")
        H = hp["hidden"]
        d1, d2, d3 = ATTN_DROPOUTS if hp["dropout"] is None else (hp["dropout"],) * 3
        self.rnn = _rnn(cell, C, H, hp["layers"], d1)
        self.drop1 = nn.Dropout(d1)
        self.mha = nn.MultiheadAttention(H, num_heads=2, batch_first=True)
        self.ln1 = nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Dropout(d2))
        self.ln2 = nn.LayerNorm(H)
        self.drop2 = nn.Dropout(d3)
        self.tail = tail
        self.out = nn.Linear(H + tail.out_dim, 1)

    def forward(self, x_seq, x_static, x_cat):
        a, _ = self.rnn(x_seq)
        a = self.drop1(a)
        z = self.ln1(a + self.mha(a, a, a, need_weights=False)[0])
        z = self.ln2(z + self.ff(z))
        h = self.drop2(z.mean(dim=1))
        return {"logit": self.out(_cat(h, self.tail(x_static, x_cat))).squeeze(1)}


class _CombinedNet(nn.Module):
    """결합-기본: 정형 Dense 128→64(x_static + 임베딩) + 시계열 LSTM(64) → 트랜스포머 인코더 1블록
    (헤드 4·FF 128·dropout 0.1) → 평균 풀링 → Dense 64, concat → Dense 64 → Dense 1.
    결합-VAE(vae=True): + 시퀀스 VAE 갈래, 머리 concat 에 μ 16·복원 오차 1칼럼(기울기 막음).
    hp: 시계열 갈래 폭(LSTM·d_model)·LSTM 층 수·트랜스포머 dropout(LSTM 층 사이도)."""

    def __init__(self, C: int, W: int, tail: _Tail, vae: bool, hp: Mapping | None = None):
        super().__init__()
        if tail.out_dim == 0:
            raise ValueError("결합 모델 정형 갈래에 넣을 x_static·x_cat 이 없음")
        hp = _hp(hp, "CMBV" if vae else "CMB")
        H, d = hp["hidden"], hp["dropout"]
        self.tail = tail
        self.struct = nn.Sequential(nn.Linear(tail.out_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.rnn = _rnn("LSTM", C, H, hp["layers"], d)
        self.enc = nn.TransformerEncoder(_encoder_layer(H, d), num_layers=1, enable_nested_tensor=False)
        self.ts_out = nn.Sequential(nn.Linear(H, 64), nn.ReLU())
        self.vae = SeqVAECore(C, W, variational=True) if vae else None
        head_in = 64 + 64 + ((LATENT + 1) if vae else 0)
        self.head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_seq, x_static, x_cat):
        s = self.struct(self.tail(x_static, x_cat))
        a, _ = self.rnn(x_seq)
        t = self.ts_out(self.enc(a).mean(dim=1))
        parts = [s, t]
        out = {}
        if self.vae is not None:
            vae_rows, mu, _ = _vae_rows(self.vae, x_seq, sample=self.training)
            with torch.no_grad():
                rec_err = _mse_rows(self.vae.decode(mu), x_seq)
            parts += [mu, rec_err.detach().unsqueeze(1)]
            out["vae_rows"] = vae_rows
        out["logit"] = self.head(torch.cat(parts, dim=1)).squeeze(1)
        return out


def _build_classifier(arch: str, C: int, W: int, tail: _Tail, hp: Mapping | None = None) -> nn.Module:
    if arch in ("LSTM", "GRU"):
        return _RNNNet(arch, C, W, tail, hp)
    if arch == "TRF":
        return _TRFNet(C, W, tail, hp)
    if arch == "TCN":
        return _TCNNet(C, W, tail, hp)
    if arch in ("LSTMA", "GRUA"):
        return _AttnRNNNet(arch[:-1], C, W, tail, hp)
    if arch in ("CMB", "CMBV"):
        return _CombinedNet(C, W, tail, vae=(arch == "CMBV"), hp=hp)
    raise ValueError(f"모르는 구조: {arch}")


