"""practice2 노트북용 헬퍼.

반복적인 것만 모아 둔다. PLUTO 를 조립하는 과정과 시뮬레이션 루프처럼
프레임워크를 이해하는 데 필요한 코드는 노트북 본문에 직접 쓴다.

⚠ 파일 이름은 저장소 안에서 유일해야 한다. 노트북이
  `REPO_ROOT.glob("practice/**/_practice2_helper.py")` 로 찾기 때문에
  같은 이름이 둘이면 엉뚱한 파일을 집는다.

부트스트랩·시나리오 함수는 `_practice1_helper.py` 에서 복사했다. 교차 import 하면
실습 1 을 손볼 때 실습 2 가 깨진다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# 부트스트랩 — devkit 을 import 하기 전에 반드시 먼저 부른다
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    """출력 구분선. 한 셀에서 여러 결과를 이어 출력할 때 무엇이 무엇인지 나뉘도록 쓴다.

    실습 0·1 의 `section` 과 같은 형식이다.
    """
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)


def find_repo_root() -> Path:
    """run_simulation.py 가 있는 상위 디렉토리를 저장소 루트로 본다."""
    for d in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
        if (d / "run_simulation.py").exists():
            return d
    raise RuntimeError("저장소 루트를 찾지 못했습니다.")


def setup_korean_font() -> None:
    """차트 라벨에 한글이 있어 미리 폰트를 등록해 둔다.

    기본 DejaVu Sans 는 한글 글리프가 없어 빈 네모(tofu)로 나온다. WSL 이면 마운트된
    윈도우 폰트를 그대로 쓴다. 후보가 하나도 없으면 조용히 넘어간다(그림 자체는 그려짐).
    """
    import matplotlib
    import matplotlib.font_manager as fm

    # ① 파일 경로로 먼저 찾는다.
    for path in ("/mnt/c/Windows/Fonts/malgun.ttf",                  # WSL 의 윈도우 폰트
                 "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):  # apt install fonts-nanum
        if Path(path).exists():
            fm.fontManager.addfont(path)
            family = fm.FontProperties(fname=path).get_name()
            break
    else:
        # ② 없으면 이미 설치된 폰트 중 한글이 되는 것을 쓴다. 경로 목록만 두면
        #    Noto CJK 만 깔린 서버에서 아무것도 못 찾고 조용히 tofu 가 된다.
        installed = {f.name for f in fm.fontManager.ttflist}
        family = next((n for n in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic",
                                   "Noto Sans CJK KR", "Noto Sans CJK JP",
                                   "Noto Sans CJK SC", "AppleGothic", "UnDotum")
                       if n in installed), None)
        if family is None:
            return

    matplotlib.rcParams["font.family"] = family
    matplotlib.rcParams["axes.unicode_minus"] = False


def bootstrap(verbose: bool = True) -> Path:
    """script/nuplan_env.sh 와 같은 환경변수를 이 파이썬 프로세스에 넣는다.

    setdefault 를 쓰면 안 된다. 셸에 다른 nuPlan 데이터셋 경로가 이미 export 되어
    있으면 그쪽이 이겨서, 노트북이 저장소의 data/ 가 아닌 곳을 본다.
    """
    repo = find_repo_root()
    os.environ["NUPLAN_DATA_ROOT"] = str(repo / "data")
    os.environ["NUPLAN_MAPS_ROOT"] = str(repo / "data/maps")
    os.environ["NUPLAN_EXP_ROOT"] = str(repo / "data")
    os.environ["PRACTICE_MODEL_ROOT"] = str(repo / "data/model")
    os.environ["HYDRA_FULL_ERROR"] = "1"

    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))

    setup_korean_font()

    if verbose:
        print("REPO_ROOT        :", repo)
        print("NUPLAN_DATA_ROOT :", os.environ["NUPLAN_DATA_ROOT"])
    return repo


def run(cmd: str, cwd=None) -> int:
    """bash 로 실행하고 출력을 실시간으로 흘려보낸다."""
    print(f"$ {cmd}\n" + "-" * 78, flush=True)
    proc = subprocess.Popen(
        ["bash", "-lc", cmd], cwd=str(cwd or find_repo_root()),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    print("-" * 78 + f"\n{'✅' if rc == 0 else '❌'} exit={rc}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# 시나리오
# ---------------------------------------------------------------------------


def load_scenario_mapping(repo: Path):
    """devkit 의 nuplan_scenario_mapping.yaml 을 그대로 읽어 ScenarioMapping 을 만든다.

    이것이 없으면 devkit 은 로그를 자르지도 솎아내지도 않아 20 s @ 0.05 s(400 스텝)가
    나오고, 있으면 15 s @ 0.1 s(150 스텝) — 공식 챌린지와 같은 조건이 된다.
    """
    from omegaconf import OmegaConf

    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

    cfg = OmegaConf.load(
        repo / "nuplan/planning/script/config/common/scenario_builder"
        "/scenario_mapping/nuplan_scenario_mapping.yaml"
    )
    return ScenarioMapping(
        scenario_map=OmegaConf.to_container(cfg.scenario_map),
        subsample_ratio_override=cfg.subsample_ratio_override,
    )


def make_builder(repo: Path, data_root: Optional[str] = None, scenario_mapping=None):
    """NuPlanScenarioBuilder 를 만든다. data_root 기본값은 data/db/test."""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )

    return NuPlanScenarioBuilder(
        data_root=data_root or str(repo / "data/db/test"),
        map_root=str(repo / "data/maps"),
        sensor_root=str(repo / "data/sensor_blobs"),
        db_files=None,
        map_version="nuplan-maps-v1.0",
        scenario_mapping=scenario_mapping if scenario_mapping is not None
        else load_scenario_mapping(repo),
    )


def make_filter(**overrides):
    """ScenarioFilter 를 만든다. 지정하지 않은 필드는 '끄기'(None) 가 기본이다."""
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    base = dict(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None, limit_total_scenarios=None,
        timestamp_threshold_s=None, ego_displacement_minimum_m=None,
        expand_scenarios=False, remove_invalid_goals=True, shuffle=False,
        ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    base.update(overrides)
    return ScenarioFilter(**base)


def scenario_table(scenarios: list):
    """시나리오 목록을 한눈에 보는 표."""
    import pandas as pd

    return pd.DataFrame([
        {"idx": i, "token": s.token, "scenario_type": s.scenario_type,
         "log_name": s.log_name[:28], "iterations": s.get_number_of_iterations(),
         "dt": s.database_interval}
        for i, s in enumerate(scenarios)
    ])


LQR_DEFAULT = dict(
    q_longitudinal=[10.0], r_longitudinal=[1.0],
    q_lateral=[1.0, 10.0, 0.0], r_lateral=[1.0],
    discretization_time=0.1, tracking_horizon=10,
    jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
    stopping_proportional_gain=0.5, stopping_velocity=0.2,
)


def make_sim(scenario, ego_controller=None, observations=None):
    """4축을 채워 Simulation 을 만든다 (실습 1 §5 에서 손으로 쓴 것과 같은 코드)."""
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
        KinematicBicycleModel,
    )
    from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
    from nuplan.planning.simulation.simulation import Simulation
    from nuplan.planning.simulation.simulation_setup import SimulationSetup
    from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
        StepSimulationTimeController,
    )

    if ego_controller is None:
        ego_controller = TwoStageController(
            scenario, LQRTracker(**LQR_DEFAULT),
            KinematicBicycleModel(scenario.ego_vehicle_parameters),
        )
    setup = SimulationSetup(
        time_controller=StepSimulationTimeController(scenario),
        observations=observations or TracksObservation(scenario),
        ego_controller=ego_controller,
        scenario=scenario,
    )
    return Simulation(setup)


# ---------------------------------------------------------------------------
# PLUTO 구성 상수 — config/planner/model_adapter/pluto.yaml 을 그대로 옮긴 것이다.
# 값을 바꿔야 하면 그 yaml 이 기준이다.
# ---------------------------------------------------------------------------

PLUTO_CKPT = "data/model/pluto_planner.ckpt"

PLUTO_FEATURE_KWARGS = dict(
    radius=120, history_horizon=2, future_horizon=8,
    sample_interval=0.1, max_agents=48, build_reference_line=True,
)

PLUTO_MODEL_KWARGS = dict(
    dim=128, state_channel=6, polygon_channel=6, history_channel=9,
    history_steps=21, future_steps=80, encoder_depth=4, decoder_depth=4,
    drop_path=0.2, dropout=0.1, num_heads=4, num_modes=12,
    use_ego_history=False, state_attn_encoder=True, state_dropout=0.75,
    use_hidden_proj=True, cat_x=True, ref_free_traj=True,
)


# ---------------------------------------------------------------------------
# 시뮬레이션 실행
# ---------------------------------------------------------------------------

#: 검증용. 켜면 시나리오 1개·워커 1개로 낮춘다. 노트북 본문에는 분기를 두지 않는다.
FAST = bool(os.environ.get("PRACTICE2_FAST"))


def exp_dir(repo: Path, challenge: str, uid: str) -> Path:
    """Hydra 가 결과를 쌓는 디렉토리. default_experiment.yaml 의 규칙 그대로다."""
    return repo / "data/exp/simulation" / challenge / uid


def run_sim(
    challenge: str,
    uid: str,
    overrides: Sequence[str] = (),
    adapter: str = "pluto",
    scenario_filter: str = "practice_scenarios",
    limit: Optional[int] = None,
    n_workers: int = 6,
    video_dir: str = "videos",
    mode: str = "reuse",
) -> Path:
    """run_simulation.py 를 실행하고 결과 디렉토리를 돌려준다.

    override 순서를 강제한다. `+simulation=<프리셋>` 이 planner 그룹을 덮어쓰므로
    `planner=refinement_planner` 는 반드시 프리셋 뒤에 와야 한다.

    mode="reuse"  집계 parquet 이 이미 있으면 실행하지 않는다(기본).
    mode="rerun"  결과 디렉토리를 지우고 다시 실행한다. 같은 uid 로 재실행할 때
                  스텝별 CSV 가 누적되는 것을 막는다.

    Open-loop 프리셋이면 스텝별 Open-loop 채점을 켜고 영상 하단을 Open-loop 표로
    바꾼다. 그 런에서 실제로 채점되는 지표가 화면에 오도록 맞추는 것이다.
    """
    repo = find_repo_root()
    out = exp_dir(repo, challenge, uid)

    if mode == "rerun" and out.exists():
        shutil.rmtree(out)
    elif list(out.glob("aggregator_metric/*.parquet")):
        print(f"기존 결과를 재사용합니다: {out}")
        print('  다시 돌리려면 mode="rerun" 을 주십시오.')
        return out

    if FAST:
        limit, n_workers = 1, 1

    if n_workers and n_workers > 0:
        worker = [
            "worker=ray_distributed",
            f"worker.threads_per_node={n_workers}",
            "distributed_mode=SINGLE_NODE",
            "number_of_gpus_allocated_per_simulation=0.15",
        ]
    else:
        worker = ["worker=sequential"]

    # Open-loop 런에서만 계획 궤적 대 로그 ego 채점을 켠다.
    if challenge.startswith("open_loop"):
        open_loop = ["planner.refinement_planner.score_open_loop=true",
                     "planner.refinement_planner.render_mode=open"]
    else:
        open_loop = []

    args = [
        f"+simulation={challenge}",          # ① 프리셋이 planner 를 덮는다
        "planner=refinement_planner",        # ② 그래서 그 뒤에서 되돌린다
        f"planner/model_adapter={adapter}",
        "scenario_builder=nuplan",
        f"scenario_filter={scenario_filter}",
        *( [f"scenario_filter.limit_total_scenarios={limit}"] if limit else [] ),
        *worker,
        f"experiment_uid={uid}",
        "planner.refinement_planner.use_refinement=false",   # 실습 2 는 MPC 를 쓰지 않는다
        "planner.refinement_planner.render=true",
        "planner.refinement_planner.log_csv=true",
        *open_loop,
        f"+planner.refinement_planner.save_dir={video_dir}",
        *overrides,
    ]
    cmd = ("source script/nuplan_env.sh && python run_simulation.py \\\n    "
           + " \\\n    ".join(args))

    t0 = time.time()
    rc = run(cmd, cwd=repo)
    print(f"소요 {time.time() - t0:.0f}s")
    if rc != 0:
        raise RuntimeError(f"시뮬레이션 실패 (exit={rc}). 위 로그를 확인하십시오.")
    return out


# ---------------------------------------------------------------------------
# 결과 읽기 — tool/compare_runs.py 의 로더를 옮겨 왔다.
# ---------------------------------------------------------------------------

#: Open-loop 집계표의 지표 열. 앞 네 개는 max(0, 1 - 오차/임계) 연속값이고 miss_rate 만 0/1 게이트다.
OPEN_BREAKDOWN = [
    "planner_expert_average_l2_error_within_bound",
    "planner_expert_final_l2_error_within_bound",
    "planner_expert_average_heading_error_within_bound",
    "planner_expert_final_heading_error_within_bound",
    "planner_miss_rate_within_bound",
]

#: Closed-loop 집계표에서 점수에 기여하는 8개 열. 앞 4개가 곱셈 항이다.
CLOSED_BREAKDOWN = [
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "ego_is_comfortable",
]


def load_aggregate(run_dir: Path):
    """런 디렉토리에서 가장 최근 집계 parquet 을 읽는다."""
    import pandas as pd

    files = sorted(Path(run_dir).rglob("aggregator_metric/*.parquet"),
                   key=lambda p: p.stat().st_ctime)
    if not files:
        raise FileNotFoundError(f"집계 parquet 이 없습니다: {run_dir}")
    return pd.read_parquet(files[-1])


def final_score(run_dir: Path) -> float:
    """nuPlan 공식 최종 점수."""
    df = load_aggregate(run_dir)
    return float(df[df["scenario"] == "final_score"]["score"].iloc[0])


def per_scenario_scores(run_dir: Path, columns: Optional[Sequence[str]] = None):
    """시나리오별 점수 표.

    집계 parquet 에는 세 종류의 행이 섞여 있다 — 시나리오별, 시나리오 유형별 집계,
    그리고 final_score. 유형별 집계 행은 log_name 이 비어 있으므로 그것으로 거른다.
    거르지 않으면 같은 시나리오가 두 번 세어진다.
    """
    df = load_aggregate(run_dir)
    df = df[(df["scenario"] != "final_score") & (df["log_name"].notna())].copy()

    cols = ["scenario", "log_name", "scenario_type", "score"]
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"집계표에 없는 열입니다: {missing}\n"
                f"  Open-loop 와 Closed-loop 는 지표 체계가 다릅니다. "
                f"OPEN_BREAKDOWN / CLOSED_BREAKDOWN 을 구분해 쓰십시오."
            )
        cols += list(columns)

    out = df[cols].rename(columns={"scenario": "token"})
    out.insert(0, "video", out["log_name"] + "_" + out["token"] + ".mp4")
    return out.sort_values("score").reset_index(drop=True)


def openloop_horizon_table(run_dir: Path):
    """Open-loop 의 horizon 별 원시값(ADE·FDE·AHE·FHE·miss rate).

    집계표의 5개 열은 임계로 정규화된 점수라, 실제로 몇 미터
    틀렸는지는 metrics/*.parquet 안에만 있다.
    """
    import pandas as pd

    rows = []
    for p in sorted(Path(run_dir).rglob("metrics/planner_expert_*.parquet")):
        df = pd.read_parquet(p)
        stat_cols = [c for c in df.columns if c.endswith("_stat_value") and "horizon" in c]
        if not stat_cols:
            continue
        for _, r in df.iterrows():
            for c in stat_cols:
                rows.append({"token": r["scenario_name"],
                             "stat": c[: -len("_stat_value")],
                             "value": r[c]})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).pivot_table(index="token", columns="stat", values="value")
    return out.round(3)


def verify_open_loop(run_dir: Path):
    """스텝별 Open-loop 지표가 공식 최종 점수와 맞는지 대조한다.

    영상 좌하단 표의 `mean(1Hz)` 열은 공식 격자 프레임의 누적 평균이고, `SCORE` 는
    그 평균으로 만든 집계 점수다. 시나리오 마지막 프레임에서 공식 parquet 과 같아야
    한다. 시나리오마다 한 행으로 비교 결과를 돌려준다.
    """
    import numpy as np
    import pandas as pd

    df = load_step_csv(run_dir)
    if df is None or "ml_ol_score" not in df.columns:
        print("스텝 CSV 에 ol_* 열이 없습니다. score_open_loop=true 로 다시 실행하십시오.")
        return pd.DataFrame()

    agg = load_aggregate(run_dir)
    agg = agg[agg["scenario"] != "final_score"].set_index("scenario")["score"]

    rows = []
    for token, g in df.groupby("token"):
        g = g.sort_values("iteration")
        last = g.iloc[-1]
        official = float(agg.get(token, np.nan))
        rows.append({
            "token": token,
            "frames(1Hz)": int(last["ml_ol_frames"]),
            "ADE": round(float(last["ml_ol_ade_mean"]), 3),
            "FDE": round(float(last["ml_ol_fde_mean"]), 3),
            "MR": round(float(last["ml_ol_miss_rate"]), 3),
            "SCORE": round(float(last["ml_ol_score"]), 3),
            "공식 SCORE": round(official, 3),
            "Δ": round(abs(float(last["ml_ol_score"]) - official), 4),
        })
    return pd.DataFrame(rows)


def load_step_csv(run_dir: Path):
    """planner 가 매 스텝 남긴 채점 CSV 를 모두 합친다 (워커 PID 마다 한 파일)."""
    import pandas as pd

    files = sorted(Path(run_dir).rglob("trajectory_evaluator_results/*.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def runner_status(run_dir: Path):
    """시나리오별 성공 여부와 스텝 소요 시간."""
    import pandas as pd

    files = list(Path(run_dir).rglob("runner_report.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.read_parquet(files[0])
    cols = [c for c in ["scenario_name", "log_name", "succeeded", "error_message",
                        "duration", "compute_trajectory_runtimes_mean"] if c in df.columns]
    return df[cols].rename(columns={"scenario_name": "token"})


# ---------------------------------------------------------------------------
# 영상 — 노트북 디렉토리 아래로 옮겨 놓아야 브라우저가 읽을 수 있다.
# ---------------------------------------------------------------------------

#: practice/videos/<tag>/ 에 모은다. .gitignore 의 videos/ 규칙에 이미 걸린다.
VIDEO_STAGE = "videos"


def collect_videos(run_dir: Path, tag: str) -> Dict[str, Optional[Path]]:
    """런의 mp4 를 practice/videos/<tag>/ 로 하드링크한다(안 되면 복사).

    JupyterLab 은 출력 HTML 의 상대경로를 **노트북 위치 기준**으로 푼다. 영상은
    data/exp/ 아래에 있어 노트북에서 `..` 없이 가리킬 수 없으므로 옮겨 온다.
    같은 파일시스템이면 하드링크라 용량도 시간도 들지 않는다.

    반환: {"<log>_<token>.mp4": 경로 또는 None}
    """
    stage = Path(__file__).resolve().parents[2] / VIDEO_STAGE / tag
    stage.mkdir(parents=True, exist_ok=True)

    found = {p.name: p for p in Path(run_dir).rglob("*.mp4")}
    out: Dict[str, Optional[Path]] = {}
    for name in sorted(per_scenario_scores(run_dir)["video"]):
        src = found.get(name)
        if src is None:
            out[name] = None
            continue
        dst = stage / name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        out[name] = dst
    return out


def video_html(rel_path: str, width: int = 760) -> str:
    """<video> 태그 하나. rel_path 는 노트북 디렉토리 기준 상대경로여야 한다."""
    return (f'<video src="{rel_path}" width="{width}" controls loop '
            f'style="border:1px solid #ccc"></video>')


def scenario_video_browser(run_dir: Path, tag: str,
                           columns: Optional[Sequence[str]] = None,
                           width: int = 760):
    """시나리오 목록 + 선택 시 해당 mp4 재생 + 그 시나리오의 세부지표.

    위젯 상태는 노트북을 정적으로 변환하면 사라진다. 표는 별도 셀에서 한 번 더
    출력하고, 여기서는 영상 확인만 담당한다.
    """
    import ipywidgets as widgets
    from IPython.display import HTML, display

    scores = per_scenario_scores(run_dir, columns=columns)
    videos = collect_videos(run_dir, tag)
    status = runner_status(run_dir)

    options = [
        (f"{r.score:6.3f}  {r.token[:8]}  {r.scenario_type}", r.token)
        for r in scores.itertuples()
    ]
    select = widgets.Select(options=options, rows=min(10, max(3, len(options))),
                            layout=widgets.Layout(width="330px"))
    out = widgets.Output()

    def show(token: str) -> None:
        row = scores[scores["token"] == token].iloc[0]
        with out:
            out.clear_output(wait=True)
            path = videos.get(row["video"])
            if path is None:
                print(f"영상이 없습니다: {row['video']}")
                print("  · 해당 시나리오가 실패했을 수 있습니다 (아래 실행 상태 참조)")
                print("  · render=false 로 실행되었을 수 있습니다")
                if len(status):
                    display(status[status["token"] == token])
            else:
                rel = f"{VIDEO_STAGE}/{tag}/{path.name}"
                display(HTML(video_html(rel, width=width)))
            display(row.to_frame(name=token))

    select.observe(lambda ch: show(ch["new"]) if ch["name"] == "value" else None,
                   names="value")
    show(options[0][1])          # 처음부터 비어 있지 않도록 한 번 그린다
    return widgets.HBox([select, out])


# ---------------------------------------------------------------------------
# 모델 간 비교 — 같은 시나리오에서 두 런을 나란히 놓는다
# ---------------------------------------------------------------------------


def _aligned_runs(runs: Mapping[str, Path], columns: Sequence[str]):
    """런들을 공통 시나리오로 맞춘다.

    두 모델의 점수는 **같은 시나리오**에서만 비교가 성립한다. limit 이 달랐거나 한쪽에서
    시뮬레이션이 실패해 시나리오 수가 다르면, 평균이 서로 다른 표본의 평균이 되어
    "어느 모델이 낫다" 는 말 자체가 성립하지 않는다. Closed-loop 은 곱셈 항 하나로
    시나리오 점수가 0 이 되므로 표본이 하나만 달라도 평균이 크게 흔들린다.

    그래서 `(log_name, token)` 이 모든 런에 다 있는 시나리오만 남기고, 무엇이 빠졌는지
    출력한다. 조용히 버리면 표가 정상으로 보인다.

    :return: {이름: 공통 시나리오만 남은 표}. index 는 (log_name, token).
    """
    tables = {}
    for name, run_dir in runs.items():
        table = per_scenario_scores(run_dir, columns=columns)
        tables[name] = table.set_index(["log_name", "token"]).sort_index()

    common = None
    for table in tables.values():
        common = table.index if common is None else common.intersection(table.index)
    if len(common) == 0:
        raise ValueError(
            "공통 시나리오가 없습니다 — 두 런이 같은 scenario_filter·limit 으로 "
            "돌았는지 확인하십시오.")

    dropped = {n: len(t) - len(common) for n, t in tables.items() if len(t) != len(common)}
    if dropped:
        print(f"공통 시나리오 {len(common)}건만 비교합니다 "
              f"(제외: {', '.join(f'{n} {v}건' for n, v in dropped.items())}).")
    return {n: t.loc[common] for n, t in tables.items()}


def compare_scenario_scores(runs: Mapping[str, Path],
                            columns: Optional[Sequence[str]] = None):
    """공통 시나리오의 총점을 런별로 나란히 놓는다.

    런이 둘이면 `차이` 열(뒤 - 앞)이 붙는다. 평균은 여기 넣지 않는다 — 세부 지표와
    함께 `compare_breakdown()` 의 `score` 행에서 본다.

    :param runs: {표시 이름: 런 디렉토리}. 파이썬 dict 는 넣은 순서를 지키므로
                 표의 열 순서는 여기 적은 순서 그대로다.
    """
    import pandas as pd

    columns = list(columns or CLOSED_BREAKDOWN)
    tables = _aligned_runs(runs, columns)
    names = list(runs)

    out = pd.DataFrame({name: table["score"] for name, table in tables.items()})
    out.insert(0, "scenario_type", next(iter(tables.values()))["scenario_type"])
    if len(names) == 2:
        out["차이"] = out[names[1]] - out[names[0]]
    return out.reset_index().sort_values(names[0]).reset_index(drop=True)


def compare_breakdown(runs: Mapping[str, Path],
                      columns: Optional[Sequence[str]] = None):
    """공통 시나리오 평균으로 세부 지표와 총점을 나란히 놓는다.

    Closed-loop 지표는 대부분 시나리오마다 0/1 로 갈려서, 시나리오별 표만 봐서는
    두 모델의 차이가 어느 항에서 왔는지 읽히지 않는다. 평균을 내야 드러난다.

    :return: index 가 지표 이름이고 마지막 행이 `score`(총점 평균)인 표.
    """
    import pandas as pd

    columns = list(columns or CLOSED_BREAKDOWN)
    tables = _aligned_runs(runs, columns)
    names = list(runs)

    out = pd.DataFrame({name: table[[*columns, "score"]].mean()
                        for name, table in tables.items()})
    if len(names) == 2:
        out["차이"] = out[names[1]] - out[names[0]]
    return out


def plot_breakdown_comparison(breakdown, title: Optional[str] = None,
                              figsize=(10, 5.5)):
    """`compare_breakdown()` 표를 가로 막대그래프로 그린다.

    지표 이름이 길어 가로로 눕힌다. 곱셈 항(하나라도 0 이면 총점이 0)은 이름 앞에
    `×` 를 붙여 가중합 항과 구분하고, 총점 행은 구분선 아래로 뗀다 — 같은 축에
    나란히 두면 총점이 지표 하나처럼 보인다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = [c for c in breakdown.columns if c != "차이"]
    rows = list(breakdown.index)
    labels = [("× " if r in CLOSED_MULTIPLICATIVE else "   ") + r for r in rows]

    y = np.arange(len(rows))
    height = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * height
        bars = ax.barh(y + offset, breakdown[name].to_numpy(float),
                       height=height * 0.9, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    if "score" in rows:
        ax.axhline(rows.index("score") - 0.5, color="0.5", lw=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("공통 시나리오 평균 점수  (× = 곱셈 항)")
    ax.axvline(1.0, color="0.7", lw=0.8, ls="--")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 빈칸 실습 채점
# ---------------------------------------------------------------------------

#: Closed-loop 집계 규칙 — closed_loop_nonreactive_agents_weighted_average.yaml 그대로.
CLOSED_MULTIPLICATIVE = [
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
]
CLOSED_WEIGHTS = {
    "ego_progress_along_expert_route": 5.0,
    "time_to_collision_within_bound": 5.0,
    "speed_limit_compliance": 4.0,
    "ego_is_comfortable": 2.0,
}


def _safe_call(fn, *args):
    """학습자 함수를 부른다. TODO 가 남아 있으면 예외 대신 문자열을 돌려준다."""
    try:
        return fn(*args)
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def _report(name: str, cases, hint: str) -> bool:
    """cases = [(설명, 기대값, 실제값), ...]. 전부 근사 일치하면 통과."""
    import numpy as np

    bad = []
    for desc, want, got in cases:
        try:
            ok = np.allclose(np.asarray(got, dtype=float),
                             np.asarray(want, dtype=float), atol=1e-6)
        except Exception:
            ok = False
        if not ok:
            bad.append((desc, want, got))

    if not bad:
        print(f"✅ {name} — {len(cases)}개 검사 통과")
        return True

    print(f"❌ {name} — {len(bad)}/{len(cases)}개 실패")
    for desc, want, got in bad[:4]:
        print(f"   · {desc}")
        print(f"       기대 {np.round(np.asarray(want, dtype=float), 4)}")
        try:
            print(f"       실제 {np.round(np.asarray(got, dtype=float), 4)}")
        except Exception:
            print(f"       실제 {got!r}   ← TODO 를 채우지 않았습니다")
    print(f"   힌트: {hint}")
    return False


def _closed_case(**over):
    """8지표가 모두 만점인 dict 에서 일부만 바꾼 입력을 만든다."""
    base = {m: 1.0 for m in CLOSED_MULTIPLICATIVE}
    base.update({k: 1.0 for k in CLOSED_WEIGHTS})
    base.update(over)
    return base


def check_closed_loop_score(fn) -> bool:
    """Closed-loop 최종 점수 함수를 검사한다. fn(metrics: dict) -> float"""
    cases = [
        ("8지표 만점 → 1.0", _closed_case(), 1.0),
        ("충돌 0 → 곱셈 항이 0 이므로 최종 0",
         _closed_case(no_ego_at_fault_collisions=0.0), 0.0),
        ("주행영역 이탈 → 0", _closed_case(drivable_area_compliance=0.0), 0.0),
        ("충돌 0.5 (물체 충돌) → 절반",
         _closed_case(no_ego_at_fault_collisions=0.5), 0.5),
        ("progress 0 → 5/16 만큼 감점",
         _closed_case(ego_progress_along_expert_route=0.0), 11 / 16),
        ("comfort 0 → 2/16 만큼 감점 (가중치가 가장 작다)",
         _closed_case(ego_is_comfortable=0.0), 14 / 16),
        ("speed_limit 0.5 → 4·0.5 반영",
         _closed_case(speed_limit_compliance=0.5), 14 / 16),
        ("가중 항 전부 0 → 0", _closed_case(**{k: 0.0 for k in CLOSED_WEIGHTS}), 0.0),
    ]
    return _report(
        "closed_loop_score",
        [(d, want, _safe_call(fn, m)) for d, m, want in cases],
        "곱셈 항 4개를 모두 곱한 값에, 가중 항 4개의 가중평균(5·5·4·2, 합 16)을 곱한다.",
    )


def closed_loop_inputs(run_dir: Path):
    """런의 집계 parquet 에서 시나리오별 8지표와 공식 점수를 뽑는다."""
    import pandas as pd

    files = sorted(Path(run_dir).rglob("aggregator_metric/*.parquet"))
    if not files:
        raise FileNotFoundError(f"aggregator_metric/*.parquet 이 없습니다: {run_dir}")
    df = pd.read_parquet(files[-1])
    df = df[df["num_scenarios"].isna()]          # 시나리오 행만 (타입 집계·final 제외)
    cols = ["scenario", "scenario_type", *CLOSED_MULTIPLICATIVE, *CLOSED_WEIGHTS]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["official"] = df["score"].values
    return out.reset_index(drop=True)


def check_local_to_global(fn) -> bool:
    """ego-local (T,3) 을 global 로 옮기는 함수를 검사한다.

    fn(local, ego_x, ego_y, ego_h) -> (T,3) ndarray  [x, y, heading]
    """
    import numpy as np

    # 헤딩 합이 정확히 ±pi 가 되면 두 표현이 같은 각이라 되감기를 판정할 수 없다.
    # 그래서 경계를 피한 값을 쓴다.
    L = np.array([[1.0, 0.0, 0.0],
                  [0.0, 2.0, np.pi / 4]])
    W = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, np.pi / 2]])          # 되감기 확인용 (pi + pi/2 = 3pi/2)

    cases = [
        ("ego 가 원점·heading 0 → local 그대로",
         L, _safe_call(fn, L, 0.0, 0.0, 0.0)),
        ("ego 평행이동 (10, -5) → 위치만 더해진다",
         L + np.array([10.0, -5.0, 0.0]), _safe_call(fn, L, 10.0, -5.0, 0.0)),
        ("ego heading 90° → local 의 전방이 global +y 로 간다",
         np.array([[0.0, 1.0, np.pi / 2],
                   [-2.0, 0.0, 3 * np.pi / 4]]), _safe_call(fn, L, 0.0, 0.0, np.pi / 2)),
        ("헤딩 되감기 — pi + pi/2 는 +3pi/2 가 아니라 -pi/2 다",
         np.array([[0.0, 0.0, np.pi],
                   [0.0, 0.0, -np.pi / 2]]), _safe_call(fn, W, 0.0, 0.0, np.pi)),
    ]
    return _report("local_to_global", cases,
                   "2차원 회전 후 평행이동이다. heading 은 더한 뒤 [-pi, pi) 로 감아야 한다 "
                   "(np.arctan2(np.sin(a), np.cos(a)) 를 쓰면 간단하다).")


def plot_trajectory_points(local, ego_x, ego_y, ego_h, log_xy=None, axes=None):
    """모델이 낸 궤적을 ego-local·global 두 좌표계에서 점별로 그린다.

    표와 영상은 '점수'만 보여 준다. 실제로 어떤 점열이 나왔는지, 그리고 그것이 어디에
    놓이는지는 이 그림에서만 보인다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    local = np.asarray(local, dtype=float)
    g = np.asarray(local_to_global_reference(local, ego_x, ego_y, ego_h), dtype=float)

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax0, ax1, ax2 = axes
    t = np.arange(len(local)) * 0.1 + 0.1

    ax0.scatter(local[:, 0], local[:, 1], c=t, cmap="viridis", s=14)
    ax0.plot(0, 0, "r*", ms=14, label="ego (원점)")
    ax0.set_xlabel("전방 [m]"); ax0.set_ylabel("좌측 [m]")
    ax0.set_title("① 모델 출력 — ego-local"); ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3); ax0.axis("equal")

    sc = ax1.scatter(g[:, 0], g[:, 1], c=t, cmap="viridis", s=14, label="계획 (변환됨)")
    if log_xy is not None:
        log_xy = np.asarray(log_xy, dtype=float)
        ax1.plot(log_xy[:, 0], log_xy[:, 1], "k--", lw=1.4, label="로그 ego")
    ax1.plot(ego_x, ego_y, "r*", ms=14, label="ego")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_title("② 같은 궤적 — global"); ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3); ax1.axis("equal")
    plt.colorbar(sc, ax=ax1, label="시간 [s]")

    step = np.hypot(np.diff(g[:, 0]), np.diff(g[:, 1]))
    ax2.plot(t[1:], step / 0.1, lw=1.8)
    ax2.set_xlabel("시간 [s]"); ax2.set_ylabel("속도 [m/s]")
    ax2.set_title("③ 점 간격에서 나온 속도"); ax2.grid(alpha=0.3)
    return axes


def local_to_global_reference(local, ego_x, ego_y, ego_h):
    """시각화가 쓰는 기준 변환 — 학습자 함수와 무관하게 그림이 나오도록 한다."""
    import numpy as np

    local = np.asarray(local, dtype=float)
    c, s = np.cos(ego_h), np.sin(ego_h)
    x = ego_x + local[:, 0] * c - local[:, 1] * s
    y = ego_y + local[:, 0] * s + local[:, 1] * c
    h = np.arctan2(np.sin(local[:, 2] + ego_h), np.cos(local[:, 2] + ego_h))
    return np.stack([x, y, h], axis=1)


#: 개별 지표의 파라미터 — devkit yaml 기본값 그대로.
#:   ego_progress_along_expert_route_statistics.yaml : score_progress_threshold 2 [m]
#:   speed_limit_compliance_statistics.yaml          : max_overspeed_value_threshold 2.23 [m/s]
PROGRESS_THRESHOLD_M = 2.0
MAX_OVERSPEED_THRESHOLD_MPS = 2.23


def check_progress_score(fn) -> bool:
    """진행률 지표를 검사한다. fn(ego_progress, expert_progress) -> float"""
    cases = [
        ("ego 50 m / expert 100 m → 0.5", (50.0, 100.0), 0.5),
        ("ego 가 expert 보다 더 감 → 1 로 saturate", (150.0, 100.0), 1.0),
        ("ego 0 m / expert 100 m → 임계 2 m 가 하한", (0.0, 100.0), 0.02),
        ("expert 가 거의 정지(1 m) → 하한 2 m 로 나눠 1.0", (5.0, 1.0), 1.0),
        ("둘 다 정지 → 1.0 (감점하지 않는다)", (0.0, 0.0), 1.0),
        ("임계보다 더 뒤로 감 (-3 m) → 0", (-3.0, 100.0), 0.0),
        ("살짝 뒤로 감 (-1 m) → 0 이 아니라 하한 적용", (-1.0, 100.0), 0.02),
    ]
    return _report(
        "progress_score",
        [(d, want, _safe_call(fn, *args)) for d, args, want in cases],
        "min(1, max(ego, 2) / max(expert, 2)) 이다. 단 ego 가 -2 m 보다 더 뒤로 가면 0.",
    )


def check_speed_limit_score(fn) -> bool:
    """제한속도 준수 지표를 검사한다. fn(overspeed, dt, duration) -> float

    overspeed 는 스텝별 초과 속도[m/s] 목록(초과가 없으면 0).
    """
    n = 150
    cases = [
        ("초과 없음 → 1.0", ([0.0] * n, 0.1, 15.0), 1.0),
        ("15 초 내내 2.23 m/s 초과 → 손실 1.0 → 0 점",
         ([2.23] * n, 0.1, 15.0), 0.0),
        ("15 초 내내 1.115 m/s 초과 → 손실 0.5 → 0.5 점",
         ([1.115] * n, 0.1, 15.0), 0.5),
        ("절반 구간만 2.23 초과 → 손실 0.5",
         ([2.23] * (n // 2) + [0.0] * (n // 2), 0.1, 15.0), 0.5),
        ("크게 초과해도 음수가 되지 않는다",
         ([10.0] * n, 0.1, 15.0), 0.0),
    ]
    return _report(
        "speed_limit_score",
        [(d, want, _safe_call(fn, *args)) for d, args, want in cases],
        "loss = sum(overspeed) * dt / (2.23 * duration) 이고 score = max(0, 1 - loss) 이다.",
    )


def progress_inputs(run_dir: Path):
    """런에서 시나리오별 ego/expert 진행량과 공식 비율을 뽑는다."""
    import pandas as pd

    hits = list(Path(run_dir).rglob("metrics/ego_progress_along_expert_route.parquet"))
    if not hits:
        raise FileNotFoundError(f"ego_progress_along_expert_route.parquet 이 없습니다: {run_dir}")
    df = pd.read_parquet(hits[0])
    return pd.DataFrame({
        "scenario": df["scenario_name"],
        "ego_progress": df["ego_total_progress_along_route_stat_value"],
        "expert_progress": df["expert_total_progress_along_route_stat_value"],
        "official": df["ego_expert_progress_along_route_ratio_stat_value"],
    }).reset_index(drop=True)
