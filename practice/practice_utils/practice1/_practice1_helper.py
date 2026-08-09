"""practice1 노트북용 헬퍼.

지루하고 반복되는 것만 모아 둔다. 프레임워크를 이해하는 데 중요한 코드
(SimulationSetup 조립, 시뮬레이션 루프)는 노트북 본문에 직접 쓴다.

⚠ 파일 이름을 `_setup_helper.py` 로 하면 안 된다.
   practice0 노트북이 `practice/**/_setup_helper.py` 를 glob 으로 찾기 때문에
   같은 이름이 둘이면 practice0 이 엉뚱한 파일을 집는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 부트스트랩 — devkit 을 import 하기 전에 반드시 먼저 부른다
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    """출력 구분선. 한 셀에서 여러 결과를 이어 출력할 때 무엇이 무엇인지 나뉘도록 쓴다.

    실습 0 의 `_setup_helper.section` 과 같은 형식이다.
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

    노트북에서는 `source nuplan_env.sh` 를 할 수 없다. 그런데 devkit 의 yaml 은
    ${oc.env:NUPLAN_DATA_ROOT} 처럼 환경변수를 읽으므로, devkit 을 import 하기 전에
    여기서 직접 넣어 줘야 한다.
    """
    repo = find_repo_root()
    # setdefault 를 쓰면 안 된다. 셸(~/.bashrc 등)에 다른 nuPlan 데이터셋 경로가 이미
    # export 되어 있으면 그쪽이 이겨서, 노트북이 저장소의 data/ 가 아닌 엉뚱한 곳을 본다.
    # nuplan_env.sh 도 무조건 덮어쓰므로 여기서도 똑같이 덮어쓴다.
    os.environ["NUPLAN_DATA_ROOT"] = str(repo / "data")
    os.environ["NUPLAN_MAPS_ROOT"] = str(repo / "data/maps")
    os.environ["NUPLAN_EXP_ROOT"] = str(repo / "data")
    os.environ["PRACTICE_MODEL_ROOT"] = str(repo / "data/model")
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # 벤더링된 nuplan/ 이 pip 설치본을 가리도록 저장소 루트를 맨 앞에 둔다.
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))

    setup_korean_font()

    if verbose:
        print("REPO_ROOT        :", repo)
        print("NUPLAN_DATA_ROOT :", os.environ["NUPLAN_DATA_ROOT"])
    return repo


# ---------------------------------------------------------------------------
# 시나리오
# ---------------------------------------------------------------------------

_SCENARIO_CACHE: Dict[tuple, list] = {}


def load_scenario_mapping(repo: Path):
    """devkit 의 nuplan_scenario_mapping.yaml 을 그대로 읽어 ScenarioMapping 을 만든다.

    이걸 빼먹으면 시나리오가 CLI 실행과 달라진다. mapping 이 없으면 devkit 은 로그를
    자르지도 솎아내지도 않아 20 s @ 0.05 s (400 스텝) 짜리가 나오고,
    mapping 이 있으면 15 s @ 0.1 s (150 스텝) — 공식 챌린지와 같은 조건이 된다.
    """
    from omegaconf import OmegaConf

    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

    yaml_path = (
        repo / "nuplan/planning/script/config/common/scenario_builder"
        "/scenario_mapping/nuplan_scenario_mapping.yaml"
    )
    cfg = OmegaConf.load(yaml_path)
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
        scenario_mapping=scenario_mapping if scenario_mapping is not None else load_scenario_mapping(repo),
    )


def make_filter(**overrides):
    """ScenarioFilter 를 만든다. 지정하지 않은 필드는 '끄기'(None) 가 기본이다."""
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    base = dict(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    base.update(overrides)
    return ScenarioFilter(**base)


def get_scenarios(builder, cache_key: Optional[str] = None, **filter_overrides) -> list:
    """빌더 + 필터로 시나리오를 만든다. 같은 조건은 캐시해서 재빌드를 피한다."""
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    key = (cache_key or repr(sorted(filter_overrides.items())),)
    if key in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[key]
    scenarios = builder.get_scenarios(make_filter(**filter_overrides), Sequential())
    _SCENARIO_CACHE[key] = scenarios
    return scenarios


def scenario_table(scenarios: list):
    """시나리오 목록을 한눈에 보는 표."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "idx": i,
                "token": s.token,
                "scenario_type": s.scenario_type,
                "log_name": s.log_name[:28],
                "iterations": s.get_number_of_iterations(),
                "dt": s.database_interval,
            }
            for i, s in enumerate(scenarios)
        ]
    )


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------

LQR_DEFAULT = dict(
    q_longitudinal=[10.0],
    r_longitudinal=[1.0],
    q_lateral=[1.0, 10.0, 0.0],
    r_lateral=[1.0],
    discretization_time=0.1,
    tracking_horizon=10,
    jerk_penalty=1e-4,
    curvature_rate_penalty=1e-2,
    stopping_proportional_gain=0.5,
    stopping_velocity=0.2,
)


def make_sim(scenario, ego_controller=None, observations=None):
    """4축을 채워 Simulation 을 만든다 (노트북 4절에서 손으로 쓴 것과 같은 코드)."""
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
            scenario,
            LQRTracker(**LQR_DEFAULT),
            KinematicBicycleModel(scenario.ego_vehicle_parameters),
        )
    setup = SimulationSetup(
        time_controller=StepSimulationTimeController(scenario),
        observations=observations or TracksObservation(scenario),
        ego_controller=ego_controller,
        scenario=scenario,
    )
    return Simulation(setup)


def run_steps(sim, planner, n_steps: Optional[int] = None, verbose: bool = False):
    """시뮬레이션 루프. SimulationRunner.run() 에서 콜백만 뺀 것이다."""
    from nuplan.planning.simulation.simulation_setup import validate_planner_setup

    validate_planner_setup(sim.setup, planner)
    planner.initialize(sim.initialize())  # initialize 가 reset 을 부르므로 재실행 안전

    t0, i = time.time(), 0
    while sim.is_simulation_running() and (n_steps is None or i < n_steps):
        planner_input = sim.get_planner_input()
        trajectory = planner.compute_trajectory(planner_input)
        sim.propagate(trajectory)
        i += 1
    if verbose:
        print(f"  {i} 스텝, {time.time() - t0:.1f}s")
    return sim.history


def ego_xy(history) -> "np.ndarray":
    """history 에서 ego 의 (N, 2) 위치 배열을 뽑는다."""
    import numpy as np

    return np.array([[s.rear_axle.x, s.rear_axle.y] for s in history.extract_ego_state])


def ego_speed(history) -> "np.ndarray":
    import numpy as np

    return np.array([s.dynamic_car_state.speed for s in history.extract_ego_state])


def expert_xy(scenario, n: Optional[int] = None) -> "np.ndarray":
    """비교 기준이 되는 로그(전문가) 궤적."""
    import numpy as np

    states = list(scenario.get_expert_ego_trajectory())
    if n is not None:
        states = states[: n + 1]
    return np.array([[s.rear_axle.x, s.rear_axle.y] for s in states])


# ---------------------------------------------------------------------------
# 시각화 — 여러 런을 겹쳐 그리려면 global frame 이어야 한다.
# (렌더러는 매 호출 ego 중심으로 회전시키므로 두 런을 한 축에 못 겹친다)
# ---------------------------------------------------------------------------


def plot_map_background(ax, map_api, center, radius: float = 80.0) -> None:
    """주변 차선을 회색으로 깔아 배경을 만든다."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    objs = map_api.get_proximal_map_objects(Point2D(*center), radius, layers)
    for layer in layers:
        for obj in objs.get(layer, []):
            try:
                x, y = obj.polygon.exterior.xy
                ax.fill(x, y, color="0.90", zorder=0)
                ax.plot(x, y, color="0.75", lw=0.5, zorder=1)
            except Exception:
                pass


def plot_paths(paths: Dict[str, "np.ndarray"], map_api=None, margin: float = 25.0,
               title: str = "", ax=None):
    """여러 궤적을 global frame 한 축에 겹쳐 그린다.

    축 범위는 궤적 전체를 감싸도록 잡고 margin[m] 만큼 여유를 준다 —
    고정 반경으로 두면 3 초짜리 궤적이 지도 한복판에서 점처럼 보인다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    allp = np.concatenate(list(paths.values()))
    cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
    half = max(allp[:, 0].ptp(), allp[:, 1].ptp()) / 2 + margin

    if map_api is not None:
        plot_map_background(ax, map_api, (cx, cy), half * 1.5)
    for name, p in paths.items():
        style = "--" if name == "expert" else "-"
        ax.plot(p[:, 0], p[:, 1], style, lw=2, label=f"{name} ({len(p)})")
        ax.plot(p[0, 0], p[0, 1], "o", ms=6, color=ax.lines[-1].get_color())
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# 셸 실행 (7절 CLI 용) — practice0 헬퍼와 같은 방식
# ---------------------------------------------------------------------------


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


def plot_open_loop_scores(scores, ax=None):
    """planner 별 Open-loop 지표를 막대로 그린다.

    표만 보면 다섯 지표가 전부 0~1 에 몰려 있어 어느 지표가 점수를 끌어내렸는지
    한눈에 들어오지 않는다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cols = [c for c in OPEN_METRICS if c in scores.columns]
    labels = ["ADE", "FDE", "AHE", "FHE", "MR"][: len(cols)]
    runs = list(scores["run"])
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(labels) + 1)          # 지표 5개 + final_score
    width = 0.8 / max(len(runs), 1)
    for k, run in enumerate(runs):
        row = scores[scores["run"] == run].iloc[0]
        vals = [float(row[c]) if row[c] is not None else np.nan for c in cols]
        vals.append(float(row["final_score"]))
        ax.bar(x + k * width - 0.4 + width / 2, vals, width,
               label=f"{run} ({row['final_score']:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels + ["최종"])
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_ylabel("점수 (0~1)")
    ax.set_title("Open-loop 지표별 점수")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_open_loop_horizons(table, ax=None):
    """지평(3·5·8 초)이 늘어날 때 원시 오차가 어떻게 커지는지 그린다."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    horizons = [3, 5, 8]
    for run in table.index:
        for kind, style in (("ADE", "-o"), ("FDE", "--s")):
            cols = [f"planner_expert_{kind}_horizon_{h}" for h in horizons]
            if not all(c in table.columns for c in cols):
                continue
            ax.plot(horizons, [float(table.loc[run, c]) for c in cols], style,
                    lw=1.6, ms=5, label=f"{run} {kind}")
    ax.set_xticks(horizons)
    ax.set_xlabel("지평 [s]")
    ax.set_ylabel("원시 오차 [m]")
    ax.set_title("지평별 원시 오차 — 멀리 볼수록 벌어진다")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)
    return ax


def plan_vs_log(scenario, planners, iteration: int = 0, horizon_s: float = 8.0):
    """각 planner 의 계획 궤적과 같은 구간의 로그 궤적을 함께 돌려준다.

    Open-loop 지표가 재는 것이 바로 이 둘의 차이다. planner 를 한 스텝만 호출한다.

    planner 마다 궤적의 점 수와 간격이 다르므로(devkit yaml 기본값: simple 0.25 s,
    idm 0.5 s, log_future 0.5 s) 좌표와 함께 **시작 시점 기준 상대 시각**을 돌려준다.

    :param planners: {이름: AbstractPlanner}
    :return: (plans, log) — plans[이름] = dict(xy=(N,2), t=(N,)), log = dict(xy, t)
    """
    import numpy as np

    from nuplan.planning.simulation.history.simulation_history_buffer import (
        SimulationHistoryBuffer,
    )
    from nuplan.planning.simulation.planner.abstract_planner import (
        PlannerInitialization, PlannerInput,
    )
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration,
    )

    dt = scenario.database_interval
    n = int(round(horizon_s / dt))
    log_states = [scenario.get_ego_state_at_iteration(iteration)] + list(
        scenario.get_ego_future_trajectory(iteration, horizon_s, n))
    t0 = log_states[0].time_point.time_us
    log = dict(xy=np.array([[s.rear_axle.x, s.rear_axle.y] for s in log_states]),
               t=np.array([(s.time_point.time_us - t0) / 1e6 for s in log_states]))

    init = PlannerInitialization(
        route_roadblock_ids=scenario.get_route_roadblock_ids(),
        mission_goal=scenario.get_mission_goal(),
        map_api=scenario.map_api,
    )
    plans = {}
    for name, planner in planners.items():
        buf = SimulationHistoryBuffer.initialize_from_scenario(
            2, scenario, planner.observation_type())
        planner.initialize(init)
        traj = planner.compute_trajectory(PlannerInput(
            iteration=SimulationIteration(scenario.get_time_point(iteration), iteration),
            history=buf,
            traffic_light_data=list(
                scenario.get_traffic_light_status_at_iteration(iteration))))
        st = traj.get_sampled_trajectory()
        plans[name] = dict(
            xy=np.array([[x.rear_axle.x, x.rear_axle.y] for x in st]),
            t=np.array([(x.time_point.time_us - t0) / 1e6 for x in st]))
    return plans, log


def _second_marks(t):
    """정수 초에 가장 가까운 표본의 인덱스. planner 마다 간격이 달라 시각으로 찾는다."""
    import numpy as np

    marks = []
    for sec in range(1, int(np.floor(t.max())) + 1):
        k = int(np.argmin(np.abs(t - sec)))
        if abs(t[k] - sec) < 0.3 and k not in marks:
            marks.append(k)
    return marks


def plot_plan_vs_log(plans, log, map_api=None, ax=None):
    """계획 궤적과 로그 궤적을 점으로 겹쳐 그린다.

    Open-loop 는 이 두 점열을 1 초 간격으로 짝지어 거리를 재는 것이다. 선으로만 그리면
    점 간격(=속도)과 어느 시점부터 벌어지는지가 보이지 않는다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    allp = np.concatenate([log["xy"]] + [p["xy"] for p in plans.values()])
    cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
    half = max(allp[:, 0].ptp(), allp[:, 1].ptp()) / 2 + 15.0
    if map_api is not None:
        plot_map_background(ax, map_api, (cx, cy), half * 1.5)

    ax.plot(log["xy"][:, 0], log["xy"][:, 1], "k--", lw=1.5, zorder=3, label="로그 ego (정답)")
    m = _second_marks(log["t"])
    ax.scatter(log["xy"][m, 0], log["xy"][m, 1], s=80, facecolors="none",
               edgecolors="black", lw=1.6, zorder=5)
    for name, p in plans.items():
        line, = ax.plot(p["xy"][:, 0], p["xy"][:, 1], "-", lw=1.4, alpha=0.85, zorder=3,
                        label=f"{name} ({len(p['xy'])}점)")
        ax.scatter(p["xy"][:, 0], p["xy"][:, 1], s=8, color=line.get_color(), zorder=3)
        mk = _second_marks(p["t"])
        ax.scatter(p["xy"][mk, 0], p["xy"][mk, 1], s=50, color=line.get_color(), zorder=4)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title("계획 궤적 대 로그 궤적 — 큰 표식이 1 초 간격")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_plan_error(plans, log, ax=None):
    """1 초 표본마다 계획과 로그의 거리 오차를 그린다 — ADE/FDE 의 재료다."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for name, p in plans.items():
        secs, errs = [], []
        for sec in range(1, 9):
            k = int(np.argmin(np.abs(p["t"] - sec)))
            j = int(np.argmin(np.abs(log["t"] - sec)))
            if abs(p["t"][k] - sec) > 0.3:
                continue
            secs.append(sec)
            errs.append(float(np.hypot(*(p["xy"][k] - log["xy"][j]))))
        if secs:
            ax.plot(secs, errs, "-o", lw=1.6, ms=5,
                    label=f"{name}  (ADE {np.mean(errs):.2f} m, FDE {errs[-1]:.2f} m)")
    ax.axhline(8.0, color="crimson", ls="--", lw=1.0, label="ADE/FDE 임계 8 m")
    ax.set_xlabel("계획 시점으로부터의 시간 [s]")
    ax.set_ylabel("로그와의 거리 오차 [m]")
    ax.set_title("1 초 표본별 거리 오차 — 이 값들이 ADE·FDE 가 된다")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


#: Open-loop 집계표의 지표 열. 앞 네 개는 max(0, 1 - 오차/임계) 인 [0,1] 연속값이고,
#: miss_rate 만 0/1 게이트이며 곱셈 항이다.
OPEN_METRICS = [
    "planner_expert_average_l2_error_within_bound",
    "planner_expert_final_l2_error_within_bound",
    "planner_expert_average_heading_error_within_bound",
    "planner_expert_final_heading_error_within_bound",
    "planner_miss_rate_within_bound",
]


#: Open-loop 점수의 임계값 — nuplan/.../simulation_metric/high_level/*.yaml 그대로.
OPEN_LOOP_THRESHOLDS = {
    "ADE": (8.0, "m"), "FDE": (8.0, "m"),
    "AHE": (0.8, "rad"), "FHE": (0.8, "rad"),
}
OPEN_LOOP_WEIGHTS = {"ADE": 1, "FDE": 1, "AHE": 2, "FHE": 2}
OPEN_LOOP_MISS_RATE_THRESHOLD = 0.3
#: miss_rate 판정용 지평별 허용 변위 (3·5·8 초).
OPEN_LOOP_DISPLACEMENT_THRESHOLDS = [6.0, 8.0, 16.0]


def plot_open_loop_score_formula(axes=None):
    """Open-loop 점수 산식을 한 눈에 보이게 그린다.

    왼쪽 둘  지표 점수 s = max(0, 1 - 오차/임계). 임계에서 0 이 되는 직선이며,
             거리(8 m)와 헤딩(0.8 rad)은 임계가 다르므로 실제 단위로 나눠 그린다.
    오른쪽   miss_rate 게이트. 0.3 을 넘는 순간 최종 점수가 0 이 된다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ax_l2, ax_hd, ax_mr = axes

    panels = [
        (ax_l2, "ADE", "FDE", "거리 오차 [m]", "#1F5FB4"),
        (ax_hd, "AHE", "FHE", "헤딩 오차 [rad]", "#D6249F"),
    ]
    for ax, avg_key, fin_key, xlabel, colour in panels:
        th, unit = OPEN_LOOP_THRESHOLDS[avg_key]
        e = np.linspace(0, th * 1.6, 200)
        ax.plot(e, np.maximum(0.0, 1.0 - e / th), color=colour, lw=2.4)
        ax.axvline(th, color="0.4", ls="--", lw=1.2)
        ax.annotate(f"임계 {th:g} {unit}\n이후 0 점", xy=(th, 0.0),
                    xytext=(th * 1.03, 0.42), fontsize=8.5, color="0.25")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("지표 점수 s")
        w = OPEN_LOOP_WEIGHTS[avg_key]
        ax.set_title(f"{avg_key} / {fin_key}   (가중치 {w})", fontsize=10.5)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.3)

    mr = np.linspace(0, 0.6, 400)
    ax_mr.step(mr, (mr <= OPEN_LOOP_MISS_RATE_THRESHOLD).astype(float),
               where="post", color="#B4321F", lw=2.4)
    ax_mr.axvline(OPEN_LOOP_MISS_RATE_THRESHOLD, color="0.4", ls="--", lw=1.2)
    ax_mr.annotate("0.3 초과 →\n최종 0 점",
                   xy=(OPEN_LOOP_MISS_RATE_THRESHOLD, 0.0),
                   xytext=(OPEN_LOOP_MISS_RATE_THRESHOLD + 0.03, 0.45),
                   fontsize=8.5, color="0.25")
    ax_mr.set_xlabel("miss_rate")
    ax_mr.set_ylabel("게이트")
    ax_mr.set_title("MR   (곱셈 항)", fontsize=10.5)
    ax_mr.set_ylim(-0.05, 1.08)
    ax_mr.grid(alpha=0.3)

    plt.gcf().suptitle(
        "최종 점수 = (1·s_ADE + 1·s_FDE + 2·s_AHE + 2·s_FHE) / 6  x  MR 게이트",
        fontsize=11, y=1.05)
    return axes


def openloop_horizon_table(run_root: Path):
    """Open-loop 의 지평(3·5·8초)별 원시 오차값.

    집계표의 다섯 열은 임계로 정규화된 점수라, 실제로 몇 미터 틀렸는지는
    metrics/*.parquet 안에만 있다.
    """
    import pandas as pd

    rows = []
    for path in sorted(Path(run_root).rglob("metrics/planner_expert_*.parquet")):
        run = path.parents[1].name
        df = pd.read_parquet(path)
        for c in [c for c in df.columns if c.endswith("_stat_value") and "horizon" in c]:
            rows.append({"run": run, "stat": c[: -len("_stat_value")],
                         "value": float(df[c].mean())})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).pivot_table(index="run", columns="stat", values="value")
    return out[[c for c in out.columns if "ADE" in c or "FDE" in c]].round(3)


def final_scores(exp_root: Path, columns: Optional[List[str]] = None,
                 pattern: str = "**/aggregator_metric/*.parquet"):
    """런 디렉토리에서 nuPlan 공식 최종 점수를 읽는다.

    columns 를 주면 그 세부 지표도 같이 뽑는다 (컬럼 이름은 parquet 그대로).
    None 인 값은 그 챌린지에서 집계하지 않는 지표라는 뜻이다.
    """
    import pandas as pd

    # 같은 experiment_uid 로 다시 돌리면 Hydra 가 디렉토리를 재사용해 parquet 이 쌓인다.
    # 런마다 가장 최근 것 하나만 쓴다 — 전부 읽으면 같은 planner 가 여러 줄로 나온다.
    latest: Dict[str, Path] = {}
    for f in Path(exp_root).glob(pattern):
        run = f.parents[1].name
        if run not in latest or f.stat().st_ctime > latest[run].stat().st_ctime:
            latest[run] = f

    rows = []
    for _, p in sorted(latest.items()):
        df = pd.read_parquet(p)
        fs = df[df["scenario"] == "final_score"]
        if not len(fs):
            continue
        row = {"run": p.parents[1].name, "final_score": float(fs["score"].iloc[0])}
        for c in columns or []:
            row[c] = fs[c].iloc[0] if c in fs.columns else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 빈칸 실습 채점
#
# 정답 코드를 노트북에 두지 않고, 학습자가 채운 함수를 여기서 직접 호출해 확인한다.
# 실패하면 입력·기대값·힌트만 알려 주고 정답 수식은 출력하지 않는다.
# ---------------------------------------------------------------------------


def _safe_call(fn, *args):
    """학습자 함수를 부른다. TODO 가 남아 있으면 예외가 나므로 문자열로 바꿔 돌려준다.

    예외를 그대로 올리면 트레이스백이 이 헬퍼 내부를 가리켜 원인이 가려진다.
    """
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
            note = "   ← TODO 를 채우지 않았습니다"
            print(f"       실제 {got!r}{note}")
    print(f"   힌트: {hint}")
    return False


def check_decompose_error(fn) -> bool:
    """오차를 reference 진행 방향 기준 종·횡으로 분해하는 함수를 검사한다.

    fn(dx, dy, h) -> (lon, lat, total)
    """
    import numpy as np

    r2 = np.sqrt(0.5)
    cases = [
        # 헤딩 0 — x 가 그대로 종방향, y 가 횡방향이다.
        ("dx=3, dy=0, h=0        (정면으로 3 m 앞섬)",   (3.0, 0.0, 3.0), _safe_call(fn, 3.0, 0.0, 0.0)),
        ("dx=0, dy=2, h=0        (왼쪽으로 2 m 벗어남)", (0.0, 2.0, 2.0), _safe_call(fn, 0.0, 2.0, 0.0)),
        # 헤딩 90° — 축이 뒤바뀐다. 부호를 틀리면 여기서 걸린다.
        ("dx=0, dy=3, h=pi/2",   (3.0, 0.0, 3.0), _safe_call(fn, 0.0, 3.0, np.pi / 2)),
        ("dx=2, dy=0, h=pi/2",   (0.0, -2.0, 2.0), _safe_call(fn, 2.0, 0.0, np.pi / 2)),
        # 대각 — total 이 종·횡과 무관하게 원래 거리인지 본다.
        ("dx=1, dy=1, h=pi/4",   (2 * r2, 0.0, np.sqrt(2)), _safe_call(fn, 1.0, 1.0, np.pi / 4)),
    ]
    return _report("decompose_error", cases,
                   "회전 변환이다. lon 은 진행 방향 성분, lat 은 그 왼쪽 성분이며 "
                   "total 은 분해와 무관한 원래 거리다.")


def check_open_loop_score(fn) -> bool:
    """Open-loop 최종 점수 함수를 검사한다.

    fn(ade, fde, ahe, fhe, miss_rate) -> float
    임계 ADE/FDE 8 m, AHE/FHE 0.8 rad, 가중치 1·1·2·2.
    miss_rate 는 지평별 miss rate 의 최댓값이며 0.3 초과면 게이트가 0 이다.
    """
    cases = [
        ("오차 0, MR 0        → 만점",                (0, 0, 0, 0, 0.0), 1.0),
        ("전부 임계값, MR 0   → 0 점",                (8, 8, 0.8, 0.8, 0.0), 0.0),
        ("전부 임계 절반      → 0.5",                 (4, 4, 0.4, 0.4, 0.0), 0.5),
        ("임계 초과는 음수가 아니라 0 으로 잘린다",     (20, 0, 0, 0, 0.0), 5 / 6),
        ("MR 0.3 은 통과(이하)",                      (0, 0, 0, 0, 0.3), 1.0),
        ("MR 0.31 은 게이트 차단 → 0",                (0, 0, 0, 0, 0.31), 0.0),
        ("게이트는 곱해진다 — 좋은 오차라도 MR 초과면 0", (2, 2, 0.2, 0.2, 0.5), 0.0),
        ("헤딩 가중치가 2 인지 — ADE 만 임계",          (8, 0, 0, 0, 0.0), 5 / 6),
        ("헤딩 가중치가 2 인지 — AHE 만 임계",          (0, 0, 0.8, 0, 0.0), 4 / 6),
    ]
    return _report(
        "open_loop_score",
        [(d, want, _safe_call(fn, *args)) for d, args, want in cases],
        "s = max(0, 1 - 오차/임계) 를 네 지표에 적용하고, 가중치 ADE 1 · FDE 1 · "
        "AHE 2 · FHE 2 로 가중평균한 뒤 miss_rate 게이트를 곱한다.",
    )


#: Open-loop 점수의 입력이 되는 원시 오차 — metric parquet 의 어느 열에서 오는지.
_OPEN_LOOP_INPUT_COLUMNS = {
    "ade": ("planner_expert_average_l2_error_within_bound",
            "avg_planner_expert_ADE_over_all_horizons_stat_value"),
    "fde": ("planner_expert_final_l2_error_within_bound",
            "avg_planner_expert_FDE_over_all_horizons_stat_value"),
    "ahe": ("planner_expert_average_heading_error_within_bound",
            "avg_planner_expert_AHE_over_all_horizons_stat_value"),
    "fhe": ("planner_expert_final_heading_error_within_bound",
            "avg_planner_expert_FHE_over_all_horizons_stat_value"),
}


def open_loop_inputs(run_root: Path):
    """런에서 시나리오별 원시 오차와 miss_rate 를 뽑는다 — 점수 함수의 입력이다.

    반환 열: planner, scenario, ade, fde, ahe, fhe, miss_rate, official
      * ade/fde/ahe/fhe 는 3·5·8 초 지평의 평균값(공식 산식이 그대로 쓰는 값)
      * miss_rate 는 지평별 miss rate 의 **최댓값**. 공식 게이트가
        `all(지평별 miss rate <= 0.3)` 이라 최댓값 하나로 같은 판정이 된다
      * official 은 nuPlan 이 계산한 그 시나리오의 점수 (정답)
    """
    import pandas as pd

    root = Path(run_root)
    rows = []
    for run_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        cols = {}
        for key, (metric, col) in _OPEN_LOOP_INPUT_COLUMNS.items():
            hits = list(run_dir.rglob(f"metrics/{metric}.parquet"))
            if not hits:
                break
            df = pd.read_parquet(hits[0])
            cols[key] = df.set_index("scenario_name")[col]
        else:
            hits = list(run_dir.rglob("metrics/planner_miss_rate_within_bound.parquet"))
            if hits:
                df = pd.read_parquet(hits[0])
                horizon_cols = [c for c in df.columns
                                if c.startswith("planner_miss_rate_horizon_")
                                and c.endswith("_stat_value")]
                cols["miss_rate"] = df.set_index("scenario_name")[horizon_cols].max(axis=1)
            agg = sorted(run_dir.rglob("aggregator_metric/*.parquet"))
            if agg:
                off = pd.read_parquet(agg[-1])
                off = off[off["num_scenarios"].isna()]
                cols["official"] = off.set_index("scenario")["score"]
            out = pd.DataFrame(cols)
            out.insert(0, "planner", run_dir.name)
            rows.append(out.reset_index().rename(columns={"index": "scenario",
                                                          "scenario_name": "scenario"}))
    import pandas as pd
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
