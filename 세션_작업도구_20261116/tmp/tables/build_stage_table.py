# 단계별 val(CV)·test × 지표 표 만들기(읽기만): val = 판 폴더 cv_scores.json 모음(all_scores.csv), test = 단계 보고 표(stage_table_report.csv)
import csv, glob, os, sys, json
T = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\tables"
OUT_XLSX = sys.argv[1] if len(sys.argv) > 1 else None

val = {}
for r in csv.DictReader(open(os.path.join(T, "all_scores.csv"), encoding="utf-8-sig")):
    val.setdefault(r["run"], r)  # 같은 이름 판은 첫 폴더(정렬 순) — 이름 겹침 점검은 아래

test, test_src = {}, {}
for p in sorted(glob.glob(os.path.join(T, "st", "**", "stage_table_report.csv"), recursive=True)):
    stage = os.path.relpath(os.path.dirname(p), os.path.join(T, "st")).replace("\\", "/")
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        name = r["단계·판"].strip()
        if "같은 쌍" in name: continue
        own = val.get(name, {}).get("stage_dir") == stage
        if name not in test or own:
            test[name] = r; test_src[name] = stage

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

METRICS = [  # (표 이름, val 칼럼, test 칼럼, 형식)
    ("PR-AUC 구간 가중", "val_prauc_bw", "test 구간 가중", "f4"),
    ("PR-AUC h 1~20", "val_prauc_h01_20", "test h 1~20 평균", "f4"),
    ("PR-AUC h 21~40", "val_prauc_h21_40", "test h 21~40 평균", "f4"),
    ("PR-AUC h 41~60", "val_prauc_h41_60", "test h 41~60 평균", "f4"),
    ("PR-AUC 60일 통", "val_prauc_pooled", "test 60일 통", "f4"),
    ("배수(구간 가중)", "val_lift_bw", "배수(구간 가중)", "f2"),
    ("threshold", "val_threshold", "threshold", "f4"),
    ("Recall", "val_recall", "Recall(test 쌍 전체)", "f3"),
    ("Precision", "val_precision", "Precision", "f4"),
    ("F1", "val_f1", "F1", "f4"),
    ("F2", "val_f2", "F2", "f4"),
    ("TP", "val_tp", "TP", "i"), ("FN", "val_fn", "FN", "i"), ("FP", "val_fp", "FP", "i"), ("TN", "val_tn", "TN", "i"),
    ("상위 1% 포착", "val_top1", "상위 1% 포착률", "f3"), ("상위 5% 포착", "val_top5", "상위 5% 포착률", "f3"),
    ("상위 10% 포착", "val_top10", "상위 10% 포착률", "f3"), ("상위 20% 포착", "val_top20", "상위 20% 포착률", "f3"),
]
VAL_FROM_TABLE = {"val_prauc_bw": "CV 구간 가중(평균)"}  # 판 폴더 없는 기준 줄(7-a)


def scores(name):
    v, t = val.get(name, {}), test.get(name, {})
    out = {}
    for lab, vc, tc, fmt in METRICS:
        vv = fnum(v.get(vc)) if v else None
        if vv is None and vc in VAL_FROM_TABLE and t:
            vv = fnum(t.get(VAL_FROM_TABLE[vc]))
        out[lab] = (vv, fnum(t.get(tc)) if t else None, fmt)
    return out


STAGES = [  # (차수·단계, 판, 판정)
    ("기준 줄: 과거 검출률만", "7-a-5y-ku010-kp010", "타겟 인코딩 단독(학습 없음) — 비교 기준"),
    ("1차: 기상+LIMS 70판", "1-CAT-W060-sum", "val 1위 → 1차 베이스라인"),
    ("2차 준비: 이력 스무딩 k", "2-1-k100", "k 100 고름"),
    ("2차 준비: 이동평균 차감 N(트리)", "2-1b-T-N07", "N 7 고름"),
    ("2차 준비: 이동평균 차감 N(LSTM)", "2-1b-L-N07", "N 7 고름"),
    ("2차: 변수 전부 13 구조", "3-0-LSTM", "val 1위 → 첫 1등"),
    ("2차 결합: ⑥ 시계열만 LSTM+attention", "x6-LSTMA", "버림"),
    ("2차 결합: ⑦ 시계열만 딥러닝", "x7-D", "버림"),
    ("2차 결합: 랜덤포레스트 + 이상 점수(VAE)", "3-1-RF-VAE", "버림"),
    ("2차 결합: CatBoost + 이상 점수(VAE)", "3-2-CAT", "버림 — 트리 상위 3개 + 이상 점수 중 1위"),
    ("2차 결합: ④ LIMS만 스태킹", "x4-L6", "버림"),
    ("2차 결합: 트리 앙상블", "3-3-T3", "버림"),
    ("3차: 1등 변수 빼기", "3-0b-LSTM", "버림"),
    ("⑧ 정형만(기상 뺌, 비교)", "x8-RF", "비교만"),
    ("다운사이징(Theil's U)", "4-a0-LSTM", "버림"),
    ("② 60칸 신경망(줄인 자료)", "x2-NN60", "버림"),
    ("연도별 무작위 언더샘플링(음성 4배)", "5-LSTM", "채택"),
    ("검출 칸 음성 샘플링", "6-LSTM", "버림"),
    ("딥러닝 튜닝(8-D)", "8-D-LSTM", "규칙상 채택 → 담당자 결정으로 취소"),
    ("이전 최종(식품 갱신 전)", "5-LSTM", "이전 최종"),
    ("식품 갱신(생장식 KO 없음)", "5-LSTM-food2", "식품 고정표 v2 로 다시 학습 — val 로 KO 판에 짐"),
    ("최종: 식품 갱신 + 생장식 KO", "5-LSTM-KO", "최종 — val 로 고름(KO 6칼럼 추가)"),
]


def fmt(x, f):
    if x is None: return "-"
    if f == "i": return f"{x:,.0f}"
    return f"{x:.{int(f[1])}f}"


def md(kind, labs):
    i = 0 if kind == "val" else 1
    head = "| 단계 | 판 | " + " | ".join(labs) + " |"
    sep = "|" + "---|" * (len(labs) + 2)
    lines = [head, sep]
    for st, nm, jd in STAGES:
        if nm not in val and nm not in test: continue
        s = scores(nm)
        lines.append(f"| {st} | {nm} | " + " | ".join(fmt(s[l][i], s[l][2]) for l in labs) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    dup = {}
    for r in csv.DictReader(open(os.path.join(T, "all_scores.csv"), encoding="utf-8-sig")):
        dup.setdefault(r["run"], []).append(r["path"])
    print("이름 겹침:", {k: v for k, v in dup.items() if len(v) > 1})
    print("요약에 없는 판(아직 없음):", [nm for _, nm, _ in STAGES if nm not in val and nm not in test])
    print("test 출처:", {nm: test_src.get(nm) for _, nm, _ in STAGES})
    KEY = ["PR-AUC 구간 가중", "PR-AUC h 1~20", "PR-AUC h 21~40", "PR-AUC h 41~60", "PR-AUC 60일 통", "배수(구간 가중)",
           "Recall", "Precision", "F1", "F2"]
    print("\n## val\n" + md("val", KEY)); print("\n## test\n" + md("test", KEY))
    if OUT_XLSX:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill
        wb = Workbook()
        def sheet(ws, names):
            labs = [m[0] for m in METRICS]
            ws.append(["차수·단계", "판", "판정"] + [f"{l} val" for l in labs] + [f"{l} test" for l in labs])
            for st, nm, jd in names:
                if nm not in val and nm not in test: continue
                s = scores(nm)
                ws.append([st, nm, jd] + [s[l][0] for l in labs] + [s[l][1] for l in labs])
            for c in ws[1]:
                c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="center")
            for row in ws.iter_rows(min_row=2):
                for c, (lab, *_r) in zip(row[3:], METRICS + METRICS):
                    f = dict((m[0], m[3]) for m in METRICS)[lab]
                    c.number_format = "#,##0" if f == "i" else ("0." + "0" * int(f[1]))
            ws.freeze_panes = "D2"
            ws.column_dimensions["A"].width = 26; ws.column_dimensions["B"].width = 22; ws.column_dimensions["C"].width = 28
        ws0 = wb.active; ws0.title = "점진적 상승"
        PATH = [("1차 베이스라인(기상+LIMS)", "1-CAT-W060-sum"), ("2차: 변수 전부, 13구조 1위", "3-0-LSTM"),
                ("연도별 무작위 언더샘플링(음성 4배)", "5-LSTM"), ("식품 특성 갱신", "5-LSTM-food2"),
                ("생장식 KO 6칼럼 추가 = 최종", "5-LSTM-KO")]
        ws0.append(["단계", "판", "val PR-AUC", "val 변화", "test PR-AUC", "test 변화"])
        pv = pt = None
        for st, nm in PATH:
            sc = scores(nm)
            v, t = sc["PR-AUC 구간 가중"][0], sc["PR-AUC 구간 가중"][1]
            ws0.append([st, nm, v, None if pv is None else v - pv, t, None if pt is None else t - pt])
            pv, pt = v, t
        for row in ws0.iter_rows(min_row=2):
            for c in row[2:]:
                c.number_format = "+0.0000;-0.0000" if c.column in (4, 6) else "0.0000"
        for c in ws0[1]:
            c.font = Font(bold=True)
        ws0.column_dimensions["A"].width = 30; ws0.column_dimensions["B"].width = 18
        ws0.append([]); ws0.append(["참고: 과거 검출률만(학습 없음)", "7-a", scores("7-a-5y-ku010-kp010")["PR-AUC 구간 가중"][0], None,
                                    scores("7-a-5y-ku010-kp010")["PR-AUC 구간 가중"][1], None])
        ws = wb.create_sheet("단계별 요약"); sheet(ws, STAGES)
        ws2 = wb.create_sheet("전체 판")
        allnames = []
        seen = set()
        for r in csv.DictReader(open(os.path.join(T, "all_scores.csv"), encoding="utf-8-sig")):
            if r["run"] in seen: continue
            seen.add(r["run"]); allnames.append((r["stage_dir"], r["run"], ""))
        if "7-a-5y-ku010-kp010" not in seen: allnames.insert(0, ("s0/0-10", "7-a-5y-ku010-kp010", "기준 줄"))
        sheet(ws2, allnames)
        ws3 = wb.create_sheet("읽는 법")
        for line in [
            "val = 시간순 교차검증 3 fold 평균(PR-AUC·배수·상위 q% 포착), Recall·Precision·F1·F2·TP·FN·FP·TN 은 3 fold 예측을 합쳐 판정 기준값에서 셈",
            "test = 2025-07-01 ~ 2026-06-30 홀드아웃(예측하는 날 기준), 판정 기준값은 val 에서 정한 값 그대로",
            "PR-AUC = trapz(precision_recall_curve), 구간 가중 = h 1~20·21~40·41~60 구간 평균을 가중",
            "판정 기준값 = val 에서 Recall 0.8 이상 중 Precision 최대",
            "고르기·채택은 val 로만, test 는 기록",
        ]:
            ws3.append([line])
        wb.save(OUT_XLSX); print("저장:", OUT_XLSX)
