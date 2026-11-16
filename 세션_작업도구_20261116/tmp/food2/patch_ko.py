from pathlib import Path
R = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")


class P:
    def __init__(self, rel):
        self.p = R / rel
        self.s = self.p.read_text(encoding="utf-8")

    def rep(self, old, new, n=1):
        assert self.s.count(old) == n, (self.p.name, old[:80], self.s.count(old))
        self.s = self.s.replace(old, new)

    def save(self):
        self.p.write_text(self.s, encoding="utf-8")


# ---------------------------------------------------------------- wxfeat
w = P("lst/wxfeat.py")
w.rep('''- WX5 단기 기상 6칼럼·WX6 rh1·td1: 기준점 R = min(d, E + 1), R − 1 끝, W 와 별개로 자기 길이.
''', '''- WX5 단기 기상 6칼럼·WX6 rh1·td1: 기준점 R = min(d, E + 1), R − 1 끝, W 와 별개로 자기 길이.
- (추가 판 — 사용자 결정 2026-11-12, 계획서 밖) KO1 Koutsoumanis et al. (2004) 생육확률 6칼럼(`ko_features`):
    Logit(P) = a0 + a1·T + a2·pH + a3·bw + a4·T·pH + a5·T·bw + a6·pH·bw + a7·T² + a8·pH², bw = sqrt(1 − aw),
    P = 1/(1 + exp(−Logit)), 액체배지(broth)·고체배지(agar) 계수 = 식품 연구원 자료(리스테리아 모델적용방안 실험용.pdf
    1쪽) 그대로. T = 날마다 avg_temp(θ, gr 과 같은 window [S, E]), aw·pH = 식품 고정표. 날마다 P → 요약 3(mean·mean10·
    max). 결측 = gr2 와 같음(그날 θ·aw·pH 중 하나라도 결측 → 그날 결측, 값 있는 날 0 → 결측). 구현 세부(계획서에
    없음): exp 넘침만 막음(z ≥ 0 이면 1/(1 + e^(−z)), z < 0 이면 e^z/(1 + e^z) — 같은 값), aw > 1 이면 bw = sqrt(음수)
    = NaN 그대로(값을 바꾸지 않음), 칼럼 이름 gr4_ko_<broth|agar>_<mean|mean10|max>.
''')
w.rep('''GROUP_NAMES = {"WX3": WX3_NAMES,''', '''# KO1(추가 판): Koutsoumanis et al. (2004) a0~a8 — 연구원 자료 그대로(고치지 않음)
KOUTSOUMANIS = {"broth": (-605.935, 1.272, 180.260, -809.232, 0.507, -4.411, -183.674, -0.065, -11.460),
                "agar": (-516.908, 3.218, 130.741, -957.558, 0.0, -7.574, -186.298, -0.031, -6.687)}
KO_MEDIA = ("broth", "agar")
KO_NAMES = [f"gr4_ko_{m}_{s}" for m in KO_MEDIA for s in GR_STATS]
GROUP_NAMES = {"WX3": WX3_NAMES,''')
w.rep('''# ---------------------------------------------------------------- WX4
def _in(x, lo, hi):''', '''# ---------------------------------------------------------------- KO1(추가 판 — 머리 설명)
def _stable_sigmoid(z) -> np.ndarray:
    """1/(1 + e^(−z)) 를 넘침 없이(z ≥ 0: 1/(1 + e^(−z)), z < 0: e^z/(1 + e^z)) — NaN 은 NaN."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def ko_P(theta, aw, ph, medium: str) -> np.ndarray:
    """Koutsoumanis et al. (2004) 생육확률 P(브로드캐스트). θ·aw·pH 중 하나라도 결측 → 결측(NaN 이 그대로 퍼짐)."""
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = KOUTSOUMANIS[medium]
    T, aw, ph = np.broadcast_arrays(np.asarray(theta, np.float64), np.asarray(aw, np.float64),
                                    np.asarray(ph, np.float64))
    with np.errstate(invalid="ignore"):
        bw = np.sqrt(1.0 - aw)
    z = (a0 + a1 * T + a2 * ph + a3 * bw + a4 * T * ph + a5 * T * bw + a6 * ph * bw + a7 * T ** 2 + a8 * ph ** 2)
    return _stable_sigmoid(z)


def ko_block(temp_win: np.ndarray, aw: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """temp_win (행, W) = window [S, E] 의 avg_temp, aw·ph (행,) → (행, 6) KO_NAMES 순."""
    a, p = aw[:, None], ph[:, None]
    daily = np.stack([ko_P(temp_win, a, p, m) for m in KO_MEDIA], axis=2)     # (n, W, 2)
    st = _mean_mean10_max(daily)                                               # (n, 2, 3)
    return st.reshape(st.shape[0], -1)


def ko_features(cube: wn.DailyCube, codes, d, h, W: int, aw, ph, *, forecast_cube: wn.DailyCube | None = None,
                allow_missing_units: bool = False, chunk_rows: int = 10_000) -> pd.DataFrame:
    """KO1 6칼럼 — window [S, E](E 끝, W 일)·θ = avg_temp(WX3 gr2 와 같은 날짜 규칙)."""
    W = _check_w10(W)
    b = wn.window_bounds(d, h, W)
    n = b.d.shape[0]
    aw = _row_array(aw, n, "aw")
    ph = _row_array(ph, n, "pH")
    ct = _need(cube, ["avg_temp"])
    ft = None if forecast_cube is None else _need(forecast_cube, ["avg_temp"])
    out = np.empty((n, len(KO_NAMES)))
    for lo in range(0, n, chunk_rows):
        sl = slice(lo, min(n, lo + chunk_rows))
        t = wn.gather_window(ct, codes, b, ft, allow_missing_units, rows=sl)[:, :, 0]
        out[sl] = ko_block(t, aw[sl], ph[sl])
    return pd.DataFrame(out, columns=KO_NAMES)


# ---------------------------------------------------------------- WX4
def _in(x, lo, hi):''')
w.save()

# ---------------------------------------------------------------- registry
r = P("lst/registry.py")
r.rep(''' (h) 국내만 버전(lst/scope.py''', ''' (i) 추가 판 KO1(사용자 결정 2026-11-12 — 계획서 밖, 5-LSTM-KO 비교 판만): Koutsoumanis 생육확률 6칼럼
     (gr4_ko_<broth|agar>_<mean|mean10|max>, lst/wxfeat.ko_features) = 새 그룹 하나 '생육확률(Koutsoumanis)'(번호 45,
     기상 그룹 — gr2 처럼 기온에서 나옴), 단위 = 칼럼 하나씩(111~116), ID 'KO1'. 기존 UNITS(110)·GROUPS(44)·IDS·
     ALL_IDS 는 그대로 두고 EXTRA_UNITS·EXTRA_GROUPS·EXTRA_IDS 로 따로 둠 → 'KO1' 을 include_ids 에 넣은 판에서만 칼럼이
     생기고(단위 번호가 가장 커서 표 칼럼 맨 뒤), 기존 판의 칼럼·순서·그룹 번호·섞기 씨앗은 그대로. 결측 표시: KO 칼럼은
     입력(θ·aw·pH)·결측 규칙이 gr2 µmax 와 같아 결측 행이 같음 → 결측 표시 목록 판정 = 같은 요약의 gr2_mu_<요약>
     (MISSING_PROXY — 단계 0-6 목록을 다시 만들지 않음).
 (h) 국내만 버전(lst/scope.py''')
r.rep('''ORDER1_IDS = ("WEATHER",)
ALL_IDS = IDS
''', '''ORDER1_IDS = ("WEATHER",)
ALL_IDS = IDS
EXTRA_IDS = ("KO1",)                                                      # 구현 세부 (i) — 추가 판만
''')
r.rep('''HIS4_STEMS = tuple(f"hist_unc1y_{lv}" for lv in HIS_LEVELS)
''', '''HIS4_STEMS = tuple(f"hist_unc1y_{lv}" for lv in HIS_LEVELS)
KO = tuple(f"gr4_ko_{m}_{s}" for m in ("broth", "agar") for s in ("mean", "mean10", "max"))   # 구현 세부 (i)
MISSING_PROXY = {f"gr4_ko_{m}_{s}": f"gr2_mu_{s}" for m in ("broth", "agar") for s in ("mean", "mean10", "max")}
KO_GROUP = "생육확률(Koutsoumanis)"
''')
r.rep('''UNITS, GROUPS = _build()
_UNIT_BY_NAME = {u.name: u for u in UNITS}
_GROUP_BY_NAME = {g.name: g for g in GROUPS}
_GROUP_BY_NO = {g.no: g for g in GROUPS}
''', '''UNITS, GROUPS = _build()


def _build_extra():
    """구현 세부 (i): 기존 번호 뒤에 붙는 추가 그룹·단위(UNITS·GROUPS 에는 안 넣음)."""
    no, gi = len(UNITS), len(GROUPS) + 1
    units = tuple(Unit(no + i + 1, c, "KO1", "table", (c,), gi, KO_GROUP, "E") for i, c in enumerate(KO))
    return units, (Group(gi, KO_GROUP, True, tuple(u.name for u in units)),)


EXTRA_UNITS, EXTRA_GROUPS = _build_extra()
_UNIT_BY_NAME = {u.name: u for u in UNITS + EXTRA_UNITS}
_GROUP_BY_NAME = {g.name: g for g in GROUPS + EXTRA_GROUPS}
_GROUP_BY_NO = {g.no: g for g in GROUPS + EXTRA_GROUPS}
''')
r.rep('''        bad = [i for i in self.include_ids if i not in IDS]''',
      '''        bad = [i for i in self.include_ids if i not in IDS + EXTRA_IDS]''')
r.rep('''        keep = {u.name for u in UNITS if u.id in self.include_ids} | set(self.include_units)
        keep -= set(self.exclude_units)
        return [u for u in UNITS if u.name in keep and u.group_no not in drop_g]''',
      '''        keep = {u.name for u in UNITS + EXTRA_UNITS if u.id in self.include_ids} | set(self.include_units)
        keep -= set(self.exclude_units)
        return [u for u in UNITS + EXTRA_UNITS if u.name in keep and u.group_no not in drop_g]''')
r.rep('''        if col.kind not in ("numeric",) or col.source is None:
            return False
        return col.source in self.lists[order]''', '''        if col.kind not in ("numeric",) or col.source is None:
            return False
        return MISSING_PROXY.get(col.source, col.source) in self.lists[order]   # 구현 세부 (i)''')
r.save()

# ---------------------------------------------------------------- inputs
i = P("lst/inputs.py")
i.rep('''    need_food = bool(ids & {"FD1"}) or bool(stems & set(rg.RG_ITEM)) or "WX3" in groups   # WX3 은 aw·pH 필요''',
      '''    need_food = (bool(ids & {"FD1"}) or bool(stems & set(rg.RG_ITEM)) or "WX3" in groups   # WX3 은 aw·pH 필요
                 or "KO1" in ids)                                                         # KO1(추가 판)도 aw·pH''')
i.rep('''    if "WX7" in ids:
        out.update(_keyed(memo, ("season",)''', '''    if "KO1" in ids:   # 추가 판(registry 구현 세부 (i)) — WX3 과 같은 키 (sgg, E, p3)
        cube_k = _need(ctx.wx, "기상 일자료")
        aw_k, ph_k = out["food_aw"], out["food_ph"]

        def _ko(pos):
            r = _take(rows, pos)
            df = wf.ko_features(cube_k, r.sgg, r.d, r.h, W, aw_k[pos], ph_k[pos])
            return {c: df[c].to_numpy(dtype=np.float64) for c in df.columns}
        out.update(_keyed(memo, ("KO1", W), [sgg, E, p3], n, _ko))
    if "WX7" in ids:
        out.update(_keyed(memo, ("season",)''')
i.save()
print("ok")
