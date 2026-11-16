# -*- coding: utf-8 -*-
"""s3_0b.py — 단계 3-0b 변수 빼기(1등 = 3-3 판정 뒤 현재 최고 판) — 계획서(v78) 요약 15, §6 2차 공통·단계 3-0b, §1-2,
§1-3(test 막기), 부록 A-5·A-9·A-10·A-12·A-14·A-15. 공통 = lst/stage3b(머리 설명·구현 세부 (a)~(f)).

차례(기본 실행 한 번 — 끝난 차례는 건너뜀, 3-3 판정 뒤):
  1. 대상: 사슬 상태 chain_state_in(logs/chain_state_3-3.json)의 현재 최고 판 = 1등 → 3-0b_target.json
     (구조 자리·후보 판·끝 판 이름 — 부록 A-14)
  2. 옛 순서 3-0b 결과 이어받기(설정 reuse_3_0b_old 가 있을 때): 1등이 옛 3-0b 대상(3-0 판)과 같고 중요도 규칙·코드가 바이트
     단위로 같으면 그 중요도 표·후보 판을 씀 → 3-0b_reuse.json(이어받을 수 있는데 옛 판이 아직 안 끝났으면 멈춤)
  3. 중요도(하위 프로세스 `--importance`): 1등 fold 모델(앙상블이면 구성 전부 — 평균), 그룹 = 1등 입력에 있는 부록 A-5 그룹,
     M = 10(`h_block`), 이상 점수 칼럼은 저장 값 그대로 → 3-0b_importance/(④ 는 대상 그룹 없음 — 계산 없이 기록)
     설정 importance_parts 가 있으면(구현 세부 — 아래) 부분 하위 프로세스 `--importance --part <부분>` 를 한꺼번에 띄우고,
     모두 끝나면 본 프로세스가 합쳐 같은 3-0b_importance/ 를 씀.
  4. 후보 판: 3M 칸 평균 I ≤ 0 그룹을 한꺼번에 뺀 1등 다시 학습(3 fold + refit, 부스팅 반복 수 고르기 켠 채) — 앙상블이면
     구성마다(하위 프로세스 `--variant`) 뒤 평균 판 3-0b-<구조 자리>
  5. 판정·끝 판: 채택 = CV 평균 상승(4자리) + 3 fold 중 2개 이상. 끝 판 3-0b-<구조 자리>-end(원 판을 가리키는 기록) = 새 현재
     최고 판 → 3-0b_decision.json + 사슬 기록 한 줄 + 단계 4 가 읽는 사슬 상태 chain_state_out(logs/chain_state_3-0b.json)
  보고(--report, 판정 뒤): 후보 판 test + 1등·끝 판·7-a·1차 베이스라인 줄 표(부록 A-12 — 앙상블이면 구성별 전후 CV 도 판정 파일에)

실행(서버 — 검수 뒤 지휘자, 3-3 판정 뒤):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s3_0b.py $A --out ~/cyj_listeria_final/results/s3/3-0b-best --config configs/s3_0b.json
  $P scripts/s3_0b.py $A --out ~/cyj_listeria_final/results/s3/3-0b-best --config configs/s3_0b.json --report --reason "단계 3-0b 보고"
  (부분 하나만 손으로: 위 첫 줄 뒤에 --importance --part fold3:1-30 — 보통은 본 프로세스가 띄움)

구현 세부(계획서에 없음 — 담당자 확인): lst/stage3b 머리 (a)~(f). 결과 폴더 = s3/3-0b-best(옛 순서 3-0b 폴더 s3/3-0b 와
나눔 — 옛 폴더의 코드 사본·판정 파일과 섞이지 않게). 중요도는 기본(importance_parts 없음) = 하위 프로세스 하나에서 한 번
(멈추면 처음부터).
중요도 나눠 돌리기·이어 하기(설정 importance_parts — 구현 세부, 계산 규칙은 lst/importance 머리 (h)):
  - 값 = 부분 목록, 부분 = 'fold1'(그 fold h 1~60) 또는 'fold3:1-30'(그 fold h 구간). 세 fold 모두, fold 마다 h 1~60 을
    겹침·빈틈 없이 덮어야 함(예: ["fold1", "fold2", "fold3"] = 3 프로세스, ["fold1", "fold2", "fold3:1-30",
    "fold3:31-60"] = 4 프로세스). 나누는 방법은 결과에 영향 없음(바꿔도 끝난 h 는 그대로 씀).
  - 부분 프로세스 = 지금 하위 프로세스와 같은 준비(자원 점검 h 마다·딥러닝이면 gpu_memory_fraction 상한), h 하나 끝날
    때마다 3-0b_importance_state/<fold>/hNN.npz. 출력은 3-0b_importance_logs/<부분>.log(이어 씀).
  - 본 프로세스: 남은 h 가 있는 부분만 한꺼번에 띄우고 모두 기다림(하나가 멈춰도 나머지는 끝까지) → 멈춘 부분이 있으면
    실험 로그에 남기고 멈춤(같은 명령을 다시 하면 남은 h 만) → 없으면 합쳐 3-0b_importance/ 를 쓰고 상태 폴더를 지움.
  - ④(대상 그룹 없음)는 나누지 않고 기본 길로.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

from lst import config as cf  # noqa: E402
from lst import imp_parts as ipt  # noqa: E402
from lst import scope as lst_scope  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage3 as s3  # noqa: E402
from lst import stage3_combine as c3  # noqa: E402
from lst import stage3b as sb  # noqa: E402
from lst import stage4 as s4  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.paths import resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, check_resources, limit_gpu_memory  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = sb.STAGE
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
SCHEMA = {
    "decision_1_3": cf.Field(str),
    "selection_2": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "dirs": {"3-0-T": cf.Field(str), "3-0-D": cf.Field(str), "3-0b": cf.Field(str)},
    "reuse_3_0b_old": cf.Field(str, required=False),   # --inputs 아래 옛 순서 3-0b 결과 폴더(이어받기)
    "importance_M": cf.Field(int),
    "importance_parts": cf.Field(list, required=False, item_types=str),   # 중요도 나눠 돌리기(머리 설명) — 없으면 하나
    "gpu_memory_fraction": cf.Field(float),
    "chain_log": cf.Field(str),
    "chain_state_in": cf.Field(str),                   # 3-3 판정 뒤 사슬 상태(1등 = 현재 최고 판)
    "chain_state_out": cf.Field(str),                  # 단계 4 가 읽는 사슬 상태
}


def _pre(argv):
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason", "--part"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run", "--importance"):
        ap.add_argument(a, action="store_true")
    ns = ap.parse_known_args(list(argv))[0]
    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)
    return ns


def load_cfg(path) -> dict:
    cfg = sc.load_cfg(path, SCHEMA)
    if cfg["stage"] != STAGE or cfg["order"] != s3.ORDER or cfg["experiment"] != s3.EXPERIMENT:
        raise cf.ConfigError(f"stage '{STAGE}' · order '2차' · experiment 'listeria-2차'")
    if cfg["importance_M"] != 10 and "smoke" not in cfg:
        raise cf.ConfigError("섞기 횟수 M = 10(부록 A-5)")
    if not (0 < cfg["gpu_memory_fraction"] <= GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    if cfg["n_jobs"] != 8:
        raise cf.ConfigError("n_jobs 8")
    if "importance_parts" in cfg:
        parse_parts(cfg["importance_parts"])
    return cfg


# ================================================================ 중요도 나눠 돌리기(설정 importance_parts)
STATE_DIR = "3-0b_importance_state"
PART_LOG_DIR = "3-0b_importance_logs"
parse_part, parse_parts, part_name = ipt.parse_part, ipt.parse_parts, ipt.part_name     # lst/imp_parts
launch_parts = ipt.launch_parts


# ================================================================ 1등
def state_in(cfg, inputs_root: Path) -> dict:
    return s4.load_state(resolve(inputs_root, cfg["chain_state_in"]))


def choose_target(cfg, inputs_root: Path, out: Path) -> dict:
    p = out / sb.TARGET_FILE
    sp = resolve(inputs_root, cfg["chain_state_in"])
    st = s4.load_state(sp)
    info = sb.target_info(st["current_best"], inputs_root)
    info["chain_state_in_md5"] = cf.file_md5(sp)
    if p.exists():
        old = rn.read_json(p)
        if old["name"] != info["name"] or old["dir"] != info["dir"] or old["chain_state_in_md5"] != info["chain_state_in_md5"]:
            raise sb.StageError("3-0b_target.json 이 지금 사슬 상태의 1등과 다름")
        return old
    out.mkdir(parents=True, exist_ok=True)
    info["utc"] = sb._utc()
    cf.dump_scores_json(p, info)
    rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])).append(
        "decision", stage=STAGE, what="3-0b 대상(1등 = 3-3 판정 뒤 현재 최고 판)", selected=info["name"],
        kind=info["kind"], candidate=info["candidate"], end=info["end"])
    print(f"3-0b 1등 = {info['name']}({info['kind']}, CV {info['cv']}) → 후보 {info['candidate']}·끝 판 {info['end']}")
    return info


def target(out: Path) -> dict:
    return rn.read_json(out / sb.TARGET_FILE)


# ================================================================ 이어받기
def reuse(cfg, inputs_root: Path, out: Path) -> dict | None:
    """옛 순서 3-0b 결과(설정 reuse_3_0b_old) — 이어받으면 3-0b_reuse.json 과 중요도 사본, 아니면 None."""
    p = out / sb.REUSE_FILE
    if p.exists():
        r = rn.read_json(p)
        return r if r["ok"] else None
    if not cfg.get("reuse_3_0b_old"):
        return None
    info = target(out)
    r = sb.reuse_check(old_rel=cfg["reuse_3_0b_old"], info=info, inputs_root=inputs_root, code_root=CODE_ROOT,
                       M=cfg["importance_M"])
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    if r["wait"]:
        raise rn.RunStopped(f"옛 3-0b 결과를 이어받을 판인데 아직 안 끝남 — {r['reasons']}. 끝난 뒤 같은 명령을 다시")
    log.append("note", stage=STAGE, message=f"옛 3-0b 결과 이어받기 {'함' if r['ok'] else '안 함'}", reasons=r["reasons"])
    print(f"옛 3-0b 결과 이어받기: {'함' if r['ok'] else '안 함 — ' + '; '.join(r['reasons'])}")
    if r["ok"]:
        sb.copy_importance(resolve(inputs_root, r["importance_dir"]), out / sb.IMPORTANCE_DIR)
    r["utc"] = sb._utc()
    cf.dump_scores_json(p, r)
    return r if r["ok"] else None


# ================================================================ 중요도(하위 프로세스)
def importance_done(out: Path) -> bool:
    return (out / sb.IMPORTANCE_DIR / "importance_meta.json").exists()


def _nodes(cfg, root: Path, inputs_root: Path, info, samples, member_nodes_fn=None):
    """1등 중요도 노드({fold: Node} — ④ 면 None). 딥러닝 구성이 있으면 GPU 메모리 상한부터."""
    if member_nodes_fn is None:
        from lst import inputs as ip

        if any(m["kind"] == "dl" for m in info["members"]):
            limit_gpu_memory(cfg["gpu_memory_fraction"])
        ctx = ip.load_context(cfg["context"], root, inputs_root, components=list(COMPONENTS))

        def member_nodes_fn(m):
            return sb.member_nodes(ctx, samples, inputs_root, m)
    return sb.importance_graph(info["best_def"], inputs_root=inputs_root, member_nodes_fn=member_nodes_fn)


def _write_no_targets(dst: Path, info) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    cf.dump_scores_json(dst / "importance_meta.json", {
        "variant": info["name"], "no_targets": True, "drop_groups": [],
        "rule": "빼는 대상 그룹 없음(1등 입력이 범주 5·h·g·이상 점수·모델 확률 칼럼뿐) — 학습 없이 1등 유지"})
    print(f"3-0b 중요도: {info['name']} 에 빼는 대상 그룹 없음")
    return dst


def _save_result(cfg, inputs_root: Path, dst: Path, info, res) -> Path:
    res.meta.update({"variant": info["name"], "kind": info["kind"],
                     "members": [m["name"] for m in info["members"]]})
    res.save(dst)
    rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])).append(
        "importance", stage=STAGE, name=info["name"], M=cfg["importance_M"], n_targets=len(res.summary),
        drop_groups=res.meta["drop_groups"], seconds=round(sum(res.seconds.values()), 2))
    print(f"3-0b 중요도: {dst} — 뺄 그룹 {res.meta['drop_groups']}")
    return dst


def run_importance(cfg, root: Path, inputs_root: Path, out: Path, *, member_nodes_fn=None, check=None) -> Path:
    dst = out / sb.IMPORTANCE_DIR
    if importance_done(out):
        return dst
    if dst.exists():                                  # 멈춘 중요도 — 옮기고 처음부터(B-1)
        rn._move_partial(dst, out, reason="멈춘 3-0b 중요도 다시")
    info = target(out)
    best = info["best_def"]
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    nodes = _nodes(cfg, root, inputs_root, info, samples, member_nodes_fn)
    res = None
    if nodes is not None:
        res = sb.run_importance(nodes, samples, M=cfg["importance_M"], expect_base=info["fold"],
                                base_tol=sb.graph_tol(best, inputs_root),
                                check=check or (lambda: check_resources(out)))
    if res is None:                                   # ④·대상 그룹 없음 — 계산 없이 기록(§6 3-0b)
        return _write_no_targets(dst, info)
    return _save_result(cfg, inputs_root, dst, info, res)


# ---------------------------------------------------------------- 나눠 돌리기(importance_parts)
def state_tag(cfg, inputs_root: Path, info) -> str:
    """이어 하기 설정 서명에 넣는 값(판·사슬 상태·시료 파일·코드 md5 — 바뀌면 이어 하지 않고 멈춤)."""
    return json.dumps({"variant": info["name"], "dir": info["dir"], "chain_state_in_md5": info["chain_state_in_md5"],
                       "samples_md5": cf.file_md5(resolve(inputs_root, cfg["samples"])),
                       "code_md5": cf.code_md5(CODE_ROOT)}, ensure_ascii=False, sort_keys=True)


def run_importance_part(cfg, root: Path, inputs_root: Path, out: Path, part: str, *, member_nodes_fn=None,
                        check=None) -> dict:
    """부분 하나(--importance --part): 끝난 h 는 건너뛰고 남은 h 마다 상태 파일."""
    from lst import importance as im

    if "importance_parts" not in cfg:
        raise cf.ConfigError("--part 는 설정 importance_parts 가 있을 때만")
    p = ipt.check_part_arg(part, cfg["importance_parts"])
    if importance_done(out):
        print(f"3-0b 중요도 이미 있음 — 부분 {part} 할 일 없음")
        return {"skipped": True}
    info = target(out)
    sd = out / STATE_DIR
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    tag = state_tag(cfg, inputs_root, info)
    f, lo, hi = p
    if not im.pending_h(samples, f, lo, hi, sd):       # 다 끝난 부분 — 모델을 읽지 않음
        print(f"3-0b 중요도 부분 {part}: 남은 h 없음")
        return {"skipped": True}
    nodes = _nodes(cfg, root, inputs_root, info, samples, member_nodes_fn)
    targets = sb.importance_targets(nodes) if nodes is not None else []
    if not targets:                                    # ④·대상 그룹 없음 — 합치기에서 기본 길과 같은 기록
        sd.mkdir(parents=True, exist_ok=True)
        cf.dump_scores_json(sd / "no_targets.json", {"variant": info["name"], "tag": tag}, overwrite=True)
        print(f"3-0b 중요도 부분 {part}: 대상 그룹 없음")
        return {"no_targets": True}
    r = im.importance_part(nodes, samples, targets, fold=f, h_from=lo, h_to=hi, M=cfg["importance_M"], state_dir=sd,
                           check=check or (lambda: check_resources(out)), tag=tag,
                           progress=lambda msg: print(f"[{part}] {msg}", flush=True))
    print(f"3-0b 중요도 부분 {part}: 계산 h {len(r['computed'])}개·이어받은 h {len(r['resumed'])}개, "
          f"{r['seconds']:.0f}초")
    return r


def pending_parts(cfg, inputs_root: Path, out: Path) -> list[str]:
    """남은 h 가 있는 부분(상태 파일 기준)."""
    from lst import importance as im

    sd = out / STATE_DIR
    if (sd / "no_targets.json").exists():
        return []
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    return [part_name(p) for p in parse_parts(cfg["importance_parts"])
            if im.pending_h(samples, p[0], p[1], p[2], sd)]


def merge_importance(cfg, inputs_root: Path, out: Path) -> Path:
    """모든 부분이 끝난 뒤: 상태 파일 → 3-0b_importance/(한 번에 돌린 것과 같은 records·summary·meta) → 상태 폴더 지움."""
    from lst import importance as im

    dst = out / sb.IMPORTANCE_DIR
    if importance_done(out):
        return dst
    if dst.exists():                                  # 쓰다 멈춘 결과 폴더 — 옮기고 다시 씀(상태 파일은 그대로)
        rn._move_partial(dst, out, reason="멈춘 3-0b 중요도 합치기 다시")
    info = target(out)
    sd = out / STATE_DIR
    tag = state_tag(cfg, inputs_root, info)
    if (sd / "no_targets.json").exists():
        if rn.read_json(sd / "no_targets.json").get("tag") != tag:
            raise sb.StageError("대상 없음 기록이 지금 판·코드와 다름 — 지휘자 확인")
        _write_no_targets(dst, info)
    else:
        samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
        targets = [im.target_from_dict(d) for d in im.state_targets(sd)]
        if any(t.kind != "group" for t in targets):
            raise sb.StageError("3-0b 상태 파일 대상이 그룹이 아님")
        res = im.merge_parts(samples, targets, state_dir=sd, M=cfg["importance_M"], expect_base=info["fold"],
                             base_tol=sb.graph_tol(info["best_def"], inputs_root), tag=tag)
        _save_result(cfg, inputs_root, dst, info, sb.finish_importance(res))
    shutil.rmtree(sd)
    return dst


def run_importance_parts(cfg, argv, root: Path, inputs_root: Path, out: Path, *, launch=None,
                         member_nodes_fn=None, check=None, in_process: bool = False) -> Path:
    """본 프로세스: 남은 부분을 한꺼번에(lst/imp_parts.run_parts) → 모두 끝나면 합치기. in_process = 시험(같은 프로세스에서
    차례로)."""
    if importance_done(out):
        return out / sb.IMPORTANCE_DIR
    ipt.run_parts(
        label="3-0b 중요도", stage=STAGE, parts=[part_name(p) for p in parse_parts(cfg["importance_parts"])],
        pending_fn=lambda: pending_parts(cfg, inputs_root, out), script=Path(__file__), argv=argv,
        log_dir=out / PART_LOG_DIR, exp_log=rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])),
        launch=launch or launch_parts,
        run_in_process=(lambda part: run_importance_part(cfg, root, inputs_root, out, part,
                                                         member_nodes_fn=member_nodes_fn, check=check))
        if in_process else None)
    return merge_importance(cfg, inputs_root, out)


def drop_groups(out: Path) -> list[str]:
    meta = rn.read_json(out / sb.IMPORTANCE_DIR / "importance_meta.json")
    if meta.get("no_targets"):
        return []
    return sb.drop_from_summary_file(out / sb.IMPORTANCE_DIR)


# ================================================================ 후보 판
def plan(cfg, inputs_root: Path, out: Path) -> dict | None:
    if not (out / sb.TARGET_FILE).exists() or not importance_done(out):
        raise sb.StageError("대상·중요도가 먼저(기본 실행)")
    info = target(out)
    r = reuse(cfg, inputs_root, out) if (out / sb.REUSE_FILE).exists() else None
    if r is not None:                                 # 이어받기 — 학습 없음(옛 후보 판 그대로)
        if not r["candidate"]:
            return None
        return {"train": [], "ensemble": None, "members": None, "groups": {r["candidate"]: r["drop"]},
                "candidate": r["candidate"], "candidate_dir": r["candidate_dir"], "reused_from": r["old_dir"]}
    return sb.candidate_plan(info, drop_groups(out), inputs_root, cfg["dirs"]["3-0b"])


def train_specs(cfg, inputs_root: Path, out: Path) -> list[rn.VariantSpec]:
    p = plan(cfg, inputs_root, out)
    return list(p["train"]) if p else []


def build_ensemble(cfg, inputs_root: Path, out: Path, code_md5: str, tracker=None) -> dict | None:
    p = plan(cfg, inputs_root, out)
    if not p or not p["ensemble"]:
        return None
    sr = tracker.stage_run(s3.EXPERIMENT, STAGE) if tracker is not None else None
    d = sb.build_ensemble_board(
        out=out, spec=p["ensemble"], member_dirs=[resolve(inputs_root, m["pred_dir"]) for m in p["members"]],
        samples=ev.SampleTable.read(resolve(inputs_root, cfg["samples"])),
        exp_log=rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])), code_md5=code_md5,
        code_root=CODE_ROOT, n_jobs=cfg["n_jobs"], tracker=tracker, stage_run_id=sr, code_reviewed=cfg["code_reviewed"])
    print(f"{p['ensemble']['name']}: 앙상블 후보 CV {d['cv_band_weighted']}")
    return d


# ================================================================ 판정·끝 판·사슬 상태
def finalize(cfg, inputs_root: Path, out: Path, code_md5: str, tracker=None) -> dict:
    dpath = out / sb.DECISION_FILE
    if dpath.exists():
        print(f"3-0b 판정 이미 있음: {dpath}")
        d = rn.read_json(dpath)
        _write_state(cfg, inputs_root, d)
        return d
    info = target(out)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    p = plan(cfg, inputs_root, out)
    cand_dir = resolve(inputs_root, p["candidate_dir"]) if p else None
    if cand_dir is not None and p["train"] and not p["ensemble"]:
        _, cdone, _ = rn.load_done(cand_dir.parent, cand_dir.name)
        if cdone["spec_md5"] != p["train"][0].md5():
            raise sb.StageError("끝난 후보 판 설정이 지금과 다름")
    j = sb.judge(info, cand_dir, samples, inputs_root)
    adopted = j["adopted"]
    if adopted:
        src_rel, src_name, src_dir = p["candidate_dir"], p["candidate"], cand_dir
    else:
        src_rel, src_name, src_dir = info["dir"], info["name"], resolve(inputs_root, info["dir"])
    espec = sb.end_spec(info, src_dir, src_name)
    edone = s3.write_pointer(out, espec, src_dir, src_rel, code_md5)
    members_cv = None
    if p and p.get("members"):                        # 부록 A-12: 앙상블이면 구성 모델별 기법 전후 CV
        members_cv = {m["algorithm"]: {"before": m["before"], "before_cv": _cv(inputs_root, m["before_dir"]),
                                       "after": m["name"], "after_cv": _cv(inputs_root, m["pred_dir"])}
                      for m in p["members"]}
    judgement = "채택" if adopted else ("버림" if p else "학습 없음")
    utc = sb._utc()
    line = {"step": STAGE, "judgement": judgement, "candidate": p["candidate"] if p else None,
            "before": info["name"], "after": info["end"], "after_dir": src_rel, "after_source": src_name, "utc": utc}
    st_in = state_in(cfg, inputs_root)
    st_out = sb.new_state(st_in, info, adopted=adopted, src_rel=src_rel, inputs_root=inputs_root, line=line)
    line["tree_target"] = st_out["tree_target"]["name"]
    st_out["history"][-1] = dict(line)
    alerts = list((j["judge"] or {}).get("alerts", []))
    meta = rn.read_json(out / sb.IMPORTANCE_DIR / "importance_meta.json")
    decision = {
        "stage": STAGE, "first": info["name"], "first_kind": info["kind"], "first_dir": info["dir"],
        "first_scores": {"cv": info["cv"], "fold": info["fold"]}, "replacement": info["replacement"],
        "part": info["part"], "importance_targets": [t["name"] for t in meta.get("targets", [])],
        "drop_groups": drop_groups(out), "member_groups": (p or {}).get("groups"),
        "candidate": j["candidate"], "candidate_dir": p["candidate_dir"] if p else None,
        "reused_from": (p or {}).get("reused_from"), "adoption_rule": j["rule"], "adoption_judge": j["judge"],
        "adopted": adopted, "members_cv": members_cv,
        "end": {"name": info["end"], "source": src_name, "source_dir": src_rel, "cv": edone["cv_band_weighted"]},
        "current_best_after": st_out["current_best"], "tree_target_after": st_out["tree_target"]["name"],
        "alerts": alerts, "code_md5": code_md5, "utc": utc,
    }
    if tracker is not None:
        _log_mlflow(tracker, cfg, out, decision, p, espec, edone, src_dir)
    c3.append_chain(inputs_root, cfg["chain_log"], line, tracker)
    for a in alerts:
        log.append("alert", stage=STAGE, message=a)
        print(f"[알림] {a}")
    log.append("decision", stage=STAGE, first=info["name"], adopted=adopted, drop_groups=decision["drop_groups"],
               end=info["end"], end_source=src_name)
    cf.dump_scores_json(dpath, decision)
    s4.save_state(resolve(inputs_root, cfg["chain_state_out"]), st_out)
    if tracker is not None:
        tracker.log_artifact(tracker.stage_run(s3.EXPERIMENT, STAGE), dpath, "decision")
    print(f"3-0b: 1등 {info['name']} — 후보 {judgement}, 끝 판 {info['end']}(원 판 {src_name}) = 현재 최고 판, "
          f"트리 대상 판 {st_out['tree_target']['name']}")
    return decision


def _cv(inputs_root: Path, rel: str) -> float:
    d = resolve(inputs_root, rel)
    return float(rn.load_done(d.parent, d.name)[2]["cv_band_weighted"])


def _write_state(cfg, inputs_root: Path, d: Mapping) -> None:
    p = resolve(inputs_root, cfg["chain_state_out"])
    if not p.exists():
        raise sb.StageError(f"판정 파일은 있는데 사슬 상태가 없음: {p} — 지휘자 확인")


def _log_mlflow(tr, cfg, out: Path, decision, p, espec, edone, src_dir: Path) -> None:
    sr = tr.stage_run(s3.EXPERIMENT, STAGE)
    imp = out / sb.IMPORTANCE_DIR
    for f in sorted(imp.glob("importance_*")):
        tr.log_artifact(sr, f, "importance")
    c = decision["candidate"]
    if c is not None and c.get("mlflow_run_id"):
        tags = tr.client.get_run(c["mlflow_run_id"]).data.tags
        if "바탕 판" not in tags or not tags.get("바탕 판"):
            tr.set_tags(c["mlflow_run_id"], {"바탕 판": decision["first"]})
        if "판정" not in tags:
            tr.set_judgement(c["mlflow_run_id"], "채택" if decision["adopted"] else "버림", against=decision["first"])
    r = tr.find_run(s3.EXPERIMENT, espec["name"], sr)
    if r is not None and r.info.status == "FINISHED":
        return
    tags = {"단계": STAGE, "역할": espec["role"], "원 판": decision["end"]["source"],
            "코드 검수 완료": "예" if cfg["code_reviewed"] else "아니오", "설정 md5": edone["spec_md5"],
            "code_md5": edone["code_md5"]}
    rid = tr.start_variant(s3.EXPERIMENT, sr, espec["name"], tags,
                           {"source_variant": decision["end"]["source"],
                            "pred_location": f"--inputs/{decision['end']['source_dir']}"})
    tr.set_judgement(rid, s3.NO_JUDGE)
    tr.log_cv(rid, rn._load_scores(out / espec["name"] / "cv_scores.json"), selection_count=edone["selection_count"])
    tr.finish(rid)
    d = rn.read_json(out / espec["name"] / "done.json")
    d["mlflow_run_id"] = rid
    cf.dump_scores_json(out / espec["name"] / "done.json", d, overwrite=True)


# ================================================================ 보고(test — 3-0b 판정 뒤)
def report(cfg, inputs_root: Path, out: Path, reason: str, tracker=None) -> Path:
    dpath = out / sb.DECISION_FILE
    if not dpath.exists():
        raise SystemExit("3-0b 판정을 먼저(후보 판 test 는 3-0b 끝난 뒤 — §1-3)")
    d = rn.read_json(dpath)
    permit = ev.report_permit(STAGE, reason)
    samples_path = resolve(inputs_root, cfg["samples"])
    samples = ev.SampleTable.read(samples_path)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    rows = []
    if cfg.get("te7a_dir"):
        rows.append(rn.te7a_baseline(resolve(inputs_root, cfg["te7a_dir"]), samples, log))
    rows.append(s3.baseline_1_row(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"]))
    rows.append(s3.base_row_from(resolve(inputs_root, d["first_dir"]), d["first"], "1등(3-3 판정 뒤 현재 최고 판)"))
    names = []
    p = plan(cfg, inputs_root, out)
    if p:
        for sp in p["train"]:
            names.append(sp.name)
        if p["ensemble"] or p.get("reused_from"):
            cdir = resolve(inputs_root, p["candidate_dir"])
            rows.append(s3.base_row_from(cdir, p["candidate"], "후보 판(앙상블 평균)" if p["ensemble"]
                                         else f"후보 판(옛 3-0b {p['reused_from']} 이어받음)"))
    e = d["end"]
    rows.append(s3.base_row_from(resolve(inputs_root, e["source_dir"]), e["name"], f"3-0b 끝 판 = {e['source']}"))
    sr = tracker.stage_run(s3.EXPERIMENT, STAGE) if tracker is not None else None
    notes = ["끝 판 줄 = 원 판(채택한 후보 판 또는 1등) 예측 그대로",
             "부스팅 판: CV = 교차 N 예측, threshold = 공통 N 예측, test = refit 공통 N(§3·§1-2)"]
    return rn.report_stage(out_dir=out, stage=STAGE, names=names, samples=samples, samples_path=samples_path,
                           permit=permit, incomplete=ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"])),
                           exp_log=log, tracker=tracker, stage_run_id=sr, base_rows=rows, extra_notes=notes)


# ================================================================ 진입점
def factories(cfg, rp, inputs_root: Path) -> sc.Factories:
    from lst import wiring as wr

    dl_kw = {"max_epochs": cfg["smoke"]["max_epochs"]} if "smoke" in cfg and "max_epochs" in cfg["smoke"] else {}
    w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS, dl_kwargs=dl_kw)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    md5s = dict(wr.input_md5s(cfg["context"], rp.root, inputs_root))
    out = Path(rp.out)
    for sp in train_specs(cfg, inputs_root, out):
        a = (sp.extra or {}).get("anomaly")
        if a:
            md5s.update(c3.vae_input_md5s(inputs_root, a["dir"]))
    return sc.Factories(make_inputs=c3.anomaly_make_inputs(w.make_inputs, samples, inputs_root),
                        make_model=w.make_model, input_md5s=md5s, fit_kwargs=w.fit_kwargs)


def before_variant(cfg) -> None:
    limit_gpu_memory(cfg["gpu_memory_fraction"])     # 대상이 딥러닝·결합일 때(트리면 GPU 안 씀 — 영향 없음)


def main(argv=None, *, factories_fn=None, tracker_fn=sc.make_tracker, check_resources_fn=None,
         before_variant_fn=before_variant, member_nodes_fn=None, subprocess_importance: bool = True):
    argv = list(sys.argv[1:] if argv is None else argv)
    ns = _pre(argv)
    if not ns.inputs or not ns.config or not ns.out:
        raise SystemExit("--inputs·--config·--out 필요")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    out = Path(ns.out).expanduser().resolve()
    cfg = load_cfg(ns.config)
    if not ns.verify_run and out != resolve(inputs_root, cfg["dirs"]["3-0b"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[3-0b]")
    root = Path(ns.root).expanduser().resolve()
    if ns.part and not ns.importance:
        raise SystemExit("--part 는 --importance 와 같이")
    if ns.importance:
        if ns.part:
            return run_importance_part(cfg, root, inputs_root, out, ns.part, member_nodes_fn=member_nodes_fn,
                                       check=check_resources_fn)
        return run_importance(cfg, root, inputs_root, out, member_nodes_fn=member_nodes_fn, check=check_resources_fn)
    if ns.report:
        if not ns.reason:
            raise SystemExit("--report 에는 --reason 이 필요(실험 로그)")
        p = report(cfg, inputs_root, out, ns.reason, tracker_fn(cfg))
        print(f"보고 표: {p}")
        return p

    def specs_fn(c):
        return train_specs(c, inputs_root, out)

    kw = dict(script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=SCHEMA, specs_fn=specs_fn,
              factories_fn=factories_fn or factories, before_variant=before_variant_fn, tracker_fn=tracker_fn,
              check_resources=check_resources_fn)
    if ns.variant or ns.list or ns.verify_run:
        return sc.stage_main(argv, **kw)
    # 기본: 대상 → 이어받기 → 중요도 → 후보 판 → (앙상블 평균) → 판정
    choose_target(cfg, inputs_root, out)
    reuse(cfg, inputs_root, out)
    if not importance_done(out) and cfg.get("importance_parts") and target(out)["kind"] != "stack4":
        run_importance_parts(cfg, argv, root, inputs_root, out, member_nodes_fn=member_nodes_fn,
                             check=check_resources_fn, in_process=not subprocess_importance)
    if not importance_done(out):
        if subprocess_importance:
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), *argv, "--importance"])
            if rc != 0:
                rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"])).append(
                    "stopped", stage=STAGE, name="3-0b 중요도", returncode=rc, reason=rn._rc_reason(rc))
                raise rn.RunStopped(f"3-0b 중요도 멈춤 — {rn._rc_reason(rc)}. 원인을 없앤 뒤 같은 명령을 다시(B-1)")
        else:
            run_importance(cfg, root, inputs_root, out, member_nodes_fn=member_nodes_fn, check=check_resources_fn)
    if train_specs(cfg, inputs_root, out):
        sc.stage_main(argv, **kw)
    code_md5 = rn.snapshot_code(CODE_ROOT, out)
    build_ensemble(cfg, inputs_root, out, code_md5, tracker_fn(cfg))
    return finalize(cfg, inputs_root, out, code_md5, tracker_fn(cfg))


if __name__ == "__main__":
    main()
