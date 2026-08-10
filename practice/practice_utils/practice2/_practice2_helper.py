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


class RealtimeNuPlanMetricTracker:
    """실제 Closed-loop history에 같은 nuPlan metric을 매 step 적용한다.

    ``update``는 두 dict를 반환한다.

    - ``step``: 이번 planning timestep에서 계산된 값
    - ``cumulative``: 현재까지 실행된 history prefix의 metric 값

    노트북은 계산 순서와 시각화에 집중하고, 지표별 입력 window를 맞추는 반복 코드는
    이 helper에 둔다. 공식 metric의 점수 함수와 threshold는 바꾸지 않는다.
    """

    def __init__(self, scenario):
        self.scenario = scenario
        self.ego_states = []
        self.observations = []
        self.expert_ego_states = []
        self.baseline_path = None
        self._cumulative_collision_score = 1.0

    @staticmethod
    def _metric_functions():
        from src.planners.evaluator.evaluate_functions import (
            score_drivable_area_compliance,
            score_driving_direction_compliance,
            score_ego_is_comfortable,
            score_ego_progress,
            score_no_ego_at_fault_collisions,
            score_speed_limit_compliance,
            score_time_to_collision,
        )
        return {
            "collision": score_no_ego_at_fault_collisions,
            "drivable": score_drivable_area_compliance,
            "direction": score_driving_direction_compliance,
            "ttc": score_time_to_collision,
            "speed": score_speed_limit_compliance,
            "comfort": score_ego_is_comfortable,
            "progress": score_ego_progress,
        }

    def _values(self, *, current_step: bool):
        fn = self._metric_functions()
        dt = float(self.scenario.database_interval)

        if current_step:
            current_states = self.ego_states[-1:]
            current_observations = self.observations[-1:]
            direction_window = max(2, int(round(1.0 / dt)) + 1)
            direction_states = self.ego_states[-direction_window:]
            comfort_window = max(5, int(round(1.5 / dt)) + 1)
            comfort_states = self.ego_states[-comfort_window:]
        else:
            current_states = self.ego_states
            current_observations = self.observations
            direction_states = self.ego_states
            comfort_states = self.ego_states

        collision, _ = fn["collision"](
            self.scenario, current_states, current_observations)
        drivable, _ = fn["drivable"](self.scenario, current_states)
        direction, _ = fn["direction"](self.scenario, direction_states)
        ttc, _ = fn["ttc"](self.scenario, current_states, current_observations)
        speed, _ = fn["speed"](self.scenario, current_states)

        comfort = 1.0
        if len(comfort_states) >= 5:
            comfort, _ = fn["comfort"](comfort_states, dt=dt)

        # progress는 nuPlan 정의상 pose 하나의 값이 아니라 시작~현재 history의 진행량이다.
        progress, making_progress, _ = fn["progress"](
            self.ego_states, self.expert_ego_states, self.baseline_path)

        return {
            "no_ego_at_fault_collisions": float(collision),
            "drivable_area_compliance": float(drivable),
            "driving_direction_compliance": float(direction),
            "ego_is_making_progress": float(making_progress),
            "ego_progress_along_expert_route": float(progress),
            "time_to_collision_within_bound": float(ttc),
            "speed_limit_compliance": float(speed),
            "ego_is_comfortable": float(comfort),
        }

    def update(self, ego_state, observation, expert_ego_state, baseline_path=None):
        """실제 상태 하나를 추가하고 ``(step, cumulative)``을 반환한다."""
        self.ego_states.append(ego_state)
        self.observations.append(observation)
        self.expert_ego_states.append(expert_ego_state)
        if self.baseline_path is None and baseline_path is not None:
            self.baseline_path = baseline_path
        step = self._values(current_step=True)
        cumulative = self._values(current_step=False)

        # Collision은 최초 접촉과 과실 분류를 기억하는 이벤트 지표다. 단일 snapshot의
        # overlap을 매번 새 충돌로 보지 않고, prefix gate가 처음 1→0이 된 step만 표시한다.
        collision_name = "no_ego_at_fault_collisions"
        cumulative_collision = min(
            self._cumulative_collision_score,
            float(cumulative[collision_name]),
        )
        step[collision_name] = float(
            not (self._cumulative_collision_score > 0.0 and cumulative_collision == 0.0)
        )
        cumulative[collision_name] = cumulative_collision
        self._cumulative_collision_score = cumulative_collision
        return step, cumulative


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
    """planner 가 매 스텝 남긴 PID별 CSV를 합쳐 한 실행 timeline으로 만든다.

    같은 uid로 중단 후 재실행한 폴더에는 예전 PID CSV가 남아 같은
    ``(log_name, token, iteration)``이 중복될 수 있다. 시나리오별로 iteration이
    가장 많이 기록된 CSV 하나를 선택하고, 길이가 같으면 수정 시각이 최신인 파일을
    선택한다. 중복 행을 그대로 그리면 한 모델의 시간축이 앞뒤로 되감겨 다른
    모델과 비교할 수 없다.
    """
    import pandas as pd

    files = sorted(Path(run_dir).rglob("trajectory_evaluator_results/*.csv"))
    frames = []
    for path in files:
        if path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        frame["_csv_source"] = str(path)
        frame["_csv_mtime_ns"] = path.stat().st_mtime_ns
        frames.append(frame)
    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)
    identity = [
        column for column in ("log_name", "token", "iteration")
        if column in result.columns
    ]
    if len(identity) == 3:
        scenario_key = ["log_name", "token"]
        source_rank = (
            result.groupby([*scenario_key, "_csv_source"], as_index=False)
            .agg(
                _iteration_count=("iteration", "nunique"),
                _source_mtime_ns=("_csv_mtime_ns", "max"),
            )
            .sort_values(["_iteration_count", "_source_mtime_ns"])
            .drop_duplicates(scenario_key, keep="last")
        )
        chosen_sources = source_rank[[*scenario_key, "_csv_source"]]
        result = (
            result.merge(
                chosen_sources,
                on=[*scenario_key, "_csv_source"],
                how="inner",
                validate="many_to_one",
            )
            .sort_values("_csv_mtime_ns")
            .drop_duplicates(identity, keep="last")
            .sort_values(["token", "iteration"])
            .reset_index(drop=True)
        )
    return result.drop(columns=["_csv_source", "_csv_mtime_ns"])


def load_executed_metric_runs(
    model_runs: Mapping[str, Path],
    expected_adapters: Mapping[str, str],
    metrics: Optional[Sequence[str]] = None,
):
    """Chapter 10의 executed-history CSV와 official 표를 model·token으로 맞춘다."""
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    metric_names = list(metrics or CLOSED_BREAKDOWN)
    step_columns = [f"step_{name}" for name in metric_names]
    cumulative_columns = [f"cumulative_{name}" for name in metric_names]
    required = ["step_final", "cumulative_final", *step_columns, *cumulative_columns]
    official_tables = {
        model: per_scenario_scores(run_dir, columns=metric_names).set_index("token")
        for model, run_dir in model_runs.items()
    }

    step_runs = {}
    for model, run_dir in model_runs.items():
        cfg = OmegaConf.load(Path(run_dir) / "code/hydra/config.yaml")
        adapter = str(cfg.planner.refinement_planner.model_adapter._target_)
        expected = expected_adapters[model]
        if not adapter.endswith(expected):
            raise AssertionError(f"{model} adapter가 아닙니다: {adapter}")

        frame = load_step_csv(run_dir)
        if frame is None:
            raise FileNotFoundError(f"{model} timestep CSV가 없습니다.")
        missing = [column for column in required if column not in frame]
        if missing:
            raise KeyError(
                f"{model} 결과는 이전 CSV 스키마입니다. 10.1을 다시 실행하십시오. "
                f"누락 열 예: {missing[:3]}"
            )
        empty = [column for column in required if frame[column].isna().all()]
        if empty:
            raise ValueError(
                f"{model} executed-history 계산이 실패했습니다. 빈 열: {empty[:3]}"
            )

        frame = frame.drop_duplicates(
            ["log_name", "token", "iteration"], keep="last"
        ).copy()
        frame.insert(0, "model", model)
        official = official_tables[model]
        unknown = sorted(set(frame["token"]) - set(official.index))
        if unknown:
            raise KeyError(f"공식 결과가 없는 token: {unknown}")
        frame["official_final"] = frame["token"].map(official["score"])
        frame["official_collision"] = frame["token"].map(
            official["no_ego_at_fault_collisions"]
        )
        # v2 CSV의 step collision은 매 snapshot에서 collision history를 초기화해
        # 지속 overlap을 새 충돌로 오인할 수 있었다. 두 모델 모두 official과 같은
        # history-prefix 판정을 기준으로 삼고, 최초 1→0 전환만 step event로 복원한다.
        original_step_collision = frame["step_no_ego_at_fault_collisions"].copy()
        corrected_groups = []
        for _, group in frame.groupby(["log_name", "token"], sort=False):
            group = group.sort_values("iteration").copy()
            cumulative_name = "cumulative_no_ego_at_fault_collisions"
            cumulative_collision = group[cumulative_name].astype(float).cummin()
            previous_collision = cumulative_collision.shift(fill_value=1.0)
            new_at_fault_event = (previous_collision > 0.0) & (cumulative_collision <= 0.0)
            group[cumulative_name] = cumulative_collision
            group["step_no_ego_at_fault_collisions"] = np.where(
                new_at_fault_event, 0.0, 1.0
            )
            corrected_groups.append(group)
        frame = pd.concat(corrected_groups, ignore_index=True)

        # collision gate가 바뀐 만큼 step final도 같은 곱셈·가중평균 식으로 다시 계산한다.
        total_weight = float(sum(CLOSED_WEIGHTS.values()))
        frame["step_final"] = frame.apply(
            lambda row: float(np.prod([
                row[f"step_{name}"] for name in CLOSED_MULTIPLICATIVE
            ])) * float(sum(
                weight * row[f"step_{name}"]
                for name, weight in CLOSED_WEIGHTS.items()
            )) / total_weight,
            axis=1,
        )
        corrected_count = int(np.sum(~np.isclose(
            original_step_collision.to_numpy(dtype=float),
            frame["step_no_ego_at_fault_collisions"].to_numpy(dtype=float),
        )))
        if corrected_count:
            print(f"{model:9s} | legacy step collision {corrected_count}개 자동 보정")
        invalid = np.isclose(frame["official_collision"], 0.0) & ~np.isclose(
            frame["official_final"], 0.0
        )
        if invalid.any():
            raise AssertionError(
                "ego-at-fault collision=0인데 official final이 0이 아닙니다."
            )

        step_runs[model] = frame
        print(
            f"{model:9s} | {adapter.rsplit('.', 1)[-1]:21s} | "
            f"{frame['token'].nunique()} scenarios, {len(frame)} executed timesteps"
        )
    return step_runs, official_tables


def summarize_executed_metric_runs(
    step_runs: Mapping[str, object],
    metrics: Optional[Sequence[str]] = None,
    dt_s: float = 0.1,
):
    """모델·시나리오별 대표 저하 시점과 공통 비교 token을 고른다."""
    import pandas as pd

    metric_names = list(metrics or CLOSED_BREAKDOWN)
    gate_names = list(CLOSED_MULTIPLICATIVE)
    alert_rows = []
    for model, frame in step_runs.items():
        for token, group in frame.groupby("token"):
            rows = (
                group.sort_values("iteration")
                .drop_duplicates("iteration", keep="last")
                .reset_index(drop=True)
            )
            first_iteration = int(rows["iteration"].iloc[0])
            failed_gate = rows[
                [f"cumulative_{name}" for name in gate_names]
            ].lt(1.0 - 1e-9).any(axis=1)
            if failed_gate.any():
                row = rows.loc[failed_gate].iloc[0]
                selection = "first executed-history gate failure"
            else:
                drops = rows["cumulative_final"].diff()
                if drops.notna().any() and float(drops.min()) < -1e-9:
                    row = rows.loc[drops.idxmin()]
                    selection = f"largest cumulative drop ({float(drops.min()):.3f})"
                else:
                    row = rows.loc[rows["cumulative_final"].idxmin()]
                    selection = "minimum cumulative score"

            step_degraded = [
                name for name in metric_names
                if float(row[f"step_{name}"]) < 1.0 - 1e-9
            ]
            cumulative_degraded = [
                name for name in metric_names
                if float(row[f"cumulative_{name}"]) < 1.0 - 1e-9
            ]
            alert_rows.append({
                "model": model,
                "token": token,
                "iteration": int(row["iteration"]),
                "time_s": (int(row["iteration"]) - first_iteration) * dt_s,
                "online_cumulative": float(row["cumulative_final"]),
                "official_collision": float(row["official_collision"]),
                "official_final": float(row["official_final"]),
                "selection": selection,
                "step_degraded": " | ".join(step_degraded),
                "cumulative_degraded": " | ".join(cumulative_degraded),
            })

    alerts = pd.DataFrame(alert_rows).sort_values(
        ["official_final", "time_s", "model"]
    ).reset_index(drop=True)
    common_tokens = sorted(set.intersection(
        *[set(frame["token"]) for frame in step_runs.values()]
    ))
    if not common_tokens:
        raise ValueError("두 모델의 CSV에 공통 token이 없습니다.")
    token_rank = (
        alerts[alerts["token"].isin(common_tokens)]
        .groupby("token")["official_final"].min().sort_values()
    )
    return alerts, common_tokens, str(token_rank.index[0])


def plot_executed_metric_timeline(
    rows,
    title: str,
    metrics: Optional[Sequence[str]] = None,
    dt_s: float = 0.1,
):
    """누적 final과 실제 step·prefix 지표를 같은 iteration 축에 그린다."""
    import matplotlib.pyplot as plt
    import numpy as np

    metric_names = list(metrics or CLOSED_BREAKDOWN)
    step_columns = [f"step_{name}" for name in metric_names]
    cumulative_columns = [f"cumulative_{name}" for name in metric_names]
    rows = rows.sort_values("iteration").reset_index(drop=True)
    iterations = rows["iteration"].to_numpy(dtype=float)
    edges = np.concatenate([iterations, [iterations[-1] + 1]])

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.1, 2.1]},
    )
    axes[0].plot(iterations, rows["cumulative_final"], "o-", ms=3, lw=1.8,
                 label="online cumulative final")
    official_final = float(rows["official_final"].iloc[0])
    axes[0].axhline(official_final, color="black", ls=":", alpha=.7,
                    label=f"official scenario final = {official_final:.3f}")
    axes[0].set(ylabel="score", ylim=(-0.05, 1.05),
                title=f"{title} — actual executed-history metrics")
    axes[0].grid(alpha=.3); axes[0].legend(loc="lower left", fontsize=8)

    for axis, columns, panel_title in (
        (axes[1], step_columns, "Per-timestep metrics (step_*)"),
        (axes[2], cumulative_columns, "History-prefix metrics (cumulative_*)"),
    ):
        image = axis.pcolormesh(
            edges, np.arange(len(columns) + 1),
            rows[columns].to_numpy(dtype=float).T,
            shading="flat", vmin=0, vmax=1, cmap="RdYlGn",
        )
        axis.set_ylim(len(columns), 0)
        axis.set_yticks(np.arange(len(columns)) + .5)
        axis.set_yticklabels(metric_names, fontsize=8)
        axis.set_title(panel_title)
        fig.colorbar(image, ax=axis, label="metric value", pad=.01)
    axes[2].set_xlabel(f"simulation iteration  (1 step = {dt_s:g} s)")
    fig.tight_layout()
    return fig


def show_executed_metric_analysis(
    step_runs: Mapping[str, object],
    alerts,
    token: str,
    metrics: Optional[Sequence[str]] = None,
    dt_s: float = 0.1,
) -> None:
    """대표 시점 전후 표와 전체 executed-history timeline을 모델별로 출력한다."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    metric_names = list(metrics or CLOSED_BREAKDOWN)
    for model, frame in step_runs.items():
        selected = frame[frame["token"] == token].sort_values("iteration").copy()
        selected["time_s"] = (
            selected["iteration"] - selected["iteration"].iloc[0]
        ) * dt_s
        alert = alerts[
            (alerts["model"] == model) & (alerts["token"] == token)
        ].iloc[0]
        center = int(alert["iteration"])
        window = selected[selected["iteration"].between(center - 3, center + 3)]
        degraded = {
            name
            for column in ("step_degraded", "cumulative_degraded")
            for name in str(alert[column]).split(" | ")
            if name
        }
        detail_names = [name for name in metric_names if name in degraded]
        detail_columns = [
            column
            for name in detail_names
            for column in (f"step_{name}", f"cumulative_{name}")
        ]
        show_columns = [
            "model", "token", "iteration", "time_s", "step_final",
            "cumulative_final", "official_collision", "official_final",
            *detail_columns,
        ]

        section(
            f"{model} | {alert['selection']} | "
            f"iter={center}, t={alert['time_s']:.1f}s"
        )
        print("현재 timestep 저하:", alert["step_degraded"] or "없음")
        print("누적 history 저하:", alert["cumulative_degraded"] or "없음")
        print(
            f"official collision={alert['official_collision']:.3f} | "
            f"official final={alert['official_final']:.3f}"
        )
        display(window[show_columns].round(3))
        fig = plot_executed_metric_timeline(
            selected, f"{model} — {token[:8]}", metric_names, dt_s
        )
        for axis in fig.axes[:3]:
            axis.axvline(center, color="red", ls="--", alpha=.7)
        plt.show()


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
# PLUTO — 후보 궤적과 학습 점수
# ---------------------------------------------------------------------------


def capture_pluto_candidates(adapter, planner_input, initialization, top_k: int = 10):
    """PLUTO 가 한 번의 forward 로 내놓는 **후보 궤적과 점수**를 상위 top_k 개만 꺼낸다.

    어댑터는 `out["output_trajectory"]` 하나만 꺼내고 후보와 점수는 버린다
    (`pluto_adapter.py` 의 `build_and_forward`). 후보를 보려면 모델 forward 를 직접 부른다 —
    Diffusion 과 달리 monkey-patch 는 필요 없다. 이미 반환 dict 에 들어 있다.

    `probability` 는 **reference line 이 패딩된 자리를 `-1e6` 으로 채워** 둔다
    (`pluto_model.py` 의 `masked_fill_`). 거르지 않으면 유효한 후보들이 색 하나로 뭉갠다.

    :return: `(candidates, scores, data)` — 점수 내림차순. `candidates` 는 `(K, 80, 3)`
             `[x, y, yaw]` ego-local 이고 `candidates[0]` 이 곧 `AdapterOutput.ml_local` 이다.
             `data` 는 지도를 그릴 때 쓰는 모델 입력 dict 다.
    """
    import torch

    feature = adapter._feature_builder.get_features_from_simulation(
        planner_input, initialization)
    data = feature.collate([feature.to_feature_tensor()]).to_device(adapter._device).data
    with torch.no_grad():
        out = adapter._planner.forward(data)

    candidates = out["candidate_trajectories"][0]        # (R, M, T, 3)
    scores = out["probability"][0]                       # (R, M)
    if candidates.numel() == 0:
        raise RuntimeError(
            "후보가 없습니다 — reference line 이 만들어지지 않았습니다. "
            "build_reference_line=True 인지 확인하십시오.")

    flat_traj = candidates.reshape(-1, candidates.shape[-2], candidates.shape[-1])
    flat_score = scores.reshape(-1)
    keep = flat_score > -1e5                             # 패딩 자리를 뗀다
    flat_traj, flat_score = flat_traj[keep], flat_score[keep]

    order = torch.argsort(flat_score, descending=True)[:top_k]
    return (flat_traj[order].cpu().numpy(), flat_score[order].cpu().numpy(), data)


def _plot_pluto_map(ax, data, alpha: float = 1.0):
    """PLUTO 피처의 지도를 배경으로 깐다. `point_position` 은
    `[구간, {중심, 좌경계, 우경계}, 20, xy]` 구조다."""
    import numpy as np

    def numpy_of(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    points = numpy_of(data["map"]["point_position"])[0]          # (M, 3, P, 2)
    on_route = numpy_of(data["map"].get("polygon_on_route", np.zeros(len(points))))
    on_route = on_route[0] if on_route.ndim > 1 else on_route

    for idx, segment in enumerate(points):
        routed = bool(on_route[idx]) if idx < len(on_route) else False
        ax.plot(*segment[0].T, color=_SCENE_COLOURS["route" if routed else "lane"],
                lw=1.6 if routed else 0.9, ls="-" if routed else (0, (4, 3)),
                alpha=(0.9 if routed else 1.0) * alpha, zorder=1)
        for boundary in segment[1:]:
            ax.plot(*boundary.T, color=_SCENE_COLOURS["boundary"], lw=0.8, alpha=alpha,
                    zorder=1)

    ax.plot(*_rotated_box(0, 0, 0, 2.297, 5.176).T,
            color=_SCENE_COLOURS["ego"], lw=2.0, alpha=alpha, zorder=5)
    return ax


def plot_pluto_candidates(candidates, scores, data=None, axes=None, figsize=(13, 4.6),
                          cmap: str = "Blues", dt: float = 0.1):
    """후보 궤적을 **학습 점수로 칠하고 점수 순으로 겹쳐** 그린다.

    색과 zorder 가 같은 것(점수)을 나타내므로, 최우선 후보가 가장 진하면서 맨 위에 온다.
    점수는 순서가 있는 양이라 한 색상의 밝기 단계만 쓰고 색막대를 붙인다.

    패널이 둘인 이유 — 후보는 `Q_lat × Q_lon` 격자라 **횡방향으로도 종방향으로도** 갈리는데,
    조감도는 앞의 것만 보여 준다. 직진 장면에서는 후보가 전부 같은 차선 위에 겹쳐 한 줄로
    보이며, 실제 차이는 "얼마나 멀리 가는가"에 있다. 오른쪽 패널이 그것을 편다.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=figsize,
                               gridspec_kw={"width_ratios": [1.55, 1]})
    ax_map, ax_lon = axes
    if data is not None:
        _plot_pluto_map(ax_map, data)

    lo, hi = float(np.min(scores)), float(np.max(scores))
    norm = Normalize(lo, hi if hi > lo else lo + 1e-6)
    colours = plt.get_cmap(cmap)(np.linspace(0.28, 1.0, len(scores)))

    travel = np.concatenate(
        [np.zeros((len(candidates), 1)),
         np.cumsum(np.linalg.norm(np.diff(candidates[:, :, :2], axis=1), axis=-1), axis=1)],
        axis=1)
    t = np.arange(candidates.shape[1]) * dt

    # 점수가 낮은 것부터 그려 높은 것이 위로 오게 한다.
    for rank in range(len(candidates) - 1, -1, -1):
        colour = colours[len(scores) - 1 - rank]
        width = 2.8 if rank == 0 else 1.4
        order = 10 + (len(scores) - rank)
        ax_map.plot(candidates[rank][:, 0], candidates[rank][:, 1], color=colour,
                    lw=width, zorder=order, solid_capstyle="round")
        ax_lon.plot(t, travel[rank], color=colour, lw=width, zorder=order)

    ax_map.annotate(f"1위 = ml_local  (점수 {scores[0]:.2f})", candidates[0][-1, :2],
                    fontsize=9, fontweight="bold", xytext=(-6, 8), ha="right",
                    textcoords="offset points", color=colours[-1], zorder=30)
    ax_lon.annotate(f"1위  {travel[0][-1]:.0f} m", (t[-1], travel[0][-1]),
                    fontsize=9, fontweight="bold", xytext=(-4, 6),
                    textcoords="offset points", ha="right", color=colours[-1], zorder=30)

    # 지도는 반경 120 m 라 그대로 두면 후보가 화면 한가운데 뭉갠다. 후보 범위로 맞춘다.
    points = candidates[:, :, :2].reshape(-1, 2)
    margin = 12.0
    ax_map.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    ax_map.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x [m]  (자차 진행 방향)")
    ax_map.set_ylabel("y [m]")
    ax_map.set_title("경로 — 횡방향 후보", fontsize=10)
    ax_map.grid(alpha=0.15)

    ax_lon.set_xlabel("t [s]")
    ax_lon.set_ylabel("주행 거리 [m]")
    ax_lon.set_title("진행 거리 — 종방향 후보", fontsize=10)
    ax_lon.grid(alpha=0.25)

    bar = ax_lon.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_lon,
                                 fraction=0.04, pad=0.02)
    bar.set_label("학습 점수 (logit)")
    ax_lon.figure.tight_layout()
    return axes


# ---------------------------------------------------------------------------
# Diffusion — 입력 씬과 denoising 과정
# ---------------------------------------------------------------------------


def diffusion_scene_inputs(adapter, planner_input) -> Dict[str, "np.ndarray"]:
    """Diffusion 이 **실제로 보는** 입력을 정규화 전 상태로 꺼낸다.

    `DiffusionModelAdapter.build_and_forward` 는 이 dict 를 만든 직후
    `observation_normalizer` 를 통과시키므로, 모델에 들어가는 값은 무차원이다.
    그림으로 확인하려면 정규화 전 값이 필요해서 같은 호출을 한 번 더 한다.

    :return: {키: (배치 차원 제거된) numpy}. 좌표는 **ego rear-axle 기준 미터**이며,
             `AdapterOutput.ml_local` 및 denoising 중간 궤적과 같은 좌표계다.

    채널 배치는 실제 텐서로 확인한 것이다.

        neighbor_agents_past  (32, 21, 11)  [x, y, cos, sin, vx, vy, 폭, 길이, onehot(차·보행자·자전거)]
        static_objects        (5, 10)       [x, y, cos, sin, 폭, 길이, onehot(4종)]
        lanes                 (70, 20, 12)  [x, y, dx, dy, 좌경계offset(2), 우경계offset(2), 신호onehot(4)]
        route_lanes           (25, 20, 12)  lanes 와 같은 채널, 주행 경로에 속한 것만

    유효/패딩 마스크는 따로 없다 — **전 채널이 0 인 행이 패딩**이며 모델도 같은 규칙을 쓴다.
    """
    inputs = adapter._data_processor.observation_adapter(
        planner_input.history,
        list(planner_input.traffic_light_data),
        adapter._map_api,
        adapter._route_roadblock_ids,
        adapter._device_str,
    )
    return {k: v.detach().cpu().numpy()[0] for k, v in inputs.items()}


def capture_denoising_steps(adapter, planner_input, initialization, seed: Optional[int] = 0):
    """denoising 매 단계의 중간 궤적을 모은다.

    Diffusion 은 잡음에서 시작해 여러 단계를 거쳐 궤적을 복원한다. 그 중간값은 평소
    밖으로 나오지 않지만, 샘플러가 이미 돌려줄 수 있게 되어 있다 —
    `DPM_Solver.sample(..., return_intermediate=True)` 가 단계마다 `x_t` 를 모아 준다.
    `dpm_sampler` 는 그 인자를 `sample_params` 로 그대로 통과시키는데, 호출부인
    `decoder.py` 가 넘기지 않을 뿐이다.

    그래서 **모델 코드를 고치지 않고** `decoder` 모듈에 바인딩된 `dpm_sampler` 이름만
    잠시 감싼다(`from ... import dpm_sampler` 로 모듈 전역에 붙어 있어 속성 교체가 먹는다).
    감싼 함수는 반드시 `x0` **하나만** 돌려줘야 한다 — 호출부가 튜플을 받을 준비가 없다.

    :param seed: `DIFFUSION_EVAL_SEED`. 초기 잡음이 고정되어 다시 돌려도 같은 그림이 나온다.
                 None 이면 건드리지 않는다.
    :return: `(steps, out)`. `steps` 는 `(K, 80, 3)` = `[x, y, yaw]` ego-local numpy 이고
             마지막 원소가 `out.ml_local` 과 같다.
    """
    import numpy as np
    import torch

    import diffusion_planner.model.module.decoder as decoder_module

    original = decoder_module.dpm_sampler
    captured: List["torch.Tensor"] = []

    def capturing_sampler(model, x_T, **kwargs):
        kwargs["sample_params"] = {**kwargs.get("sample_params", {}),
                                   "return_intermediate": True}
        x0, intermediates = original(model, x_T, **kwargs)
        # correcting_xt_fn(initial_state_constraint) 이 xt 를 in-place 로 고친다.
        # 복사하지 않으면 뒤 단계가 앞 단계 텐서를 덮어쓴다.
        captured.extend(t.detach().clone() for t in intermediates)
        return x0

    previous = os.environ.get("DIFFUSION_EVAL_SEED")
    decoder_module.dpm_sampler = capturing_sampler
    if seed is not None:
        os.environ["DIFFUSION_EVAL_SEED"] = str(seed)
    try:
        out = adapter.build_and_forward(planner_input, initialization)
    finally:
        decoder_module.dpm_sampler = original
        if seed is not None:
            if previous is None:
                os.environ.pop("DIFFUSION_EVAL_SEED", None)
            else:
                os.environ["DIFFUSION_EVAL_SEED"] = previous

    if not captured:
        raise RuntimeError(
            "중간값을 하나도 잡지 못했습니다 — 어댑터가 Diffusion 이 맞는지 확인하십시오.")

    # 모델이 내놓는 것은 정규화된 값이다. decoder 가 마지막에 하는 것과 똑같이 되돌린다.
    normalizer = adapter._config.state_normalizer
    steps = []
    for x_t in captured:
        batch, agents = x_t.shape[0], x_t.shape[1]
        denorm = normalizer.inverse(x_t.reshape(batch, agents, -1, 4))
        ego = denorm[0, 0, 1:]                      # 0번 칸은 현재 상태 앵커라 뗀다
        yaw = torch.atan2(ego[:, 3], ego[:, 2])
        steps.append(torch.stack([ego[:, 0], ego[:, 1], yaw], dim=-1).cpu().numpy())
    return np.stack(steps), out


#: 씬 배경은 궤적을 읽는 데 방해가 되면 안 된다. 전부 무채색으로 뒤로 물린다.
_SCENE_COLOURS = dict(lane="0.82", boundary="0.90", route="0.55",
                      agent="0.62", static="0.72", ego="#d95f02")


def _rotated_box(x, y, yaw, width, length):
    """중심·헤딩·크기로 사각형 네 꼭짓점을 만든다 (닫힌 5점)."""
    import numpy as np

    half = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                     [-length / 2, -width / 2], [-length / 2, width / 2],
                     [length / 2, width / 2]])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return half @ rot.T + np.array([x, y])


def plot_diffusion_scene(scene, ax=None, figsize=(11, 6), show_boundary: bool = True,
                         alpha: float = 1.0, labels: bool = True):
    """Diffusion 입력 네 갈래를 ego-local 좌표에 그린다.

    노트북 4 절 머리말 그림의 *Scenario Inputs* 네 갈래와 1:1 로 대응한다 —
    Lanes · Navigation(route) · Neighbors · Static Obj. 궤적을 겹쳐 그릴 배경이므로
    전부 무채색으로 뒤로 물리고, 자차만 색을 준다.

    :param alpha: 배경으로 깔 때 더 흐리게 하려면 낮춘다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for lane in scene["lanes"]:
        pts = lane[np.abs(lane[:, :2]).sum(-1) > 0]      # 0 패딩 제거
        if len(pts) < 2:
            continue
        if show_boundary:
            ax.plot(*(pts[:, :2] + pts[:, 4:6]).T, color=_SCENE_COLOURS["boundary"],
                    lw=0.8, alpha=alpha)
            ax.plot(*(pts[:, :2] + pts[:, 6:8]).T, color=_SCENE_COLOURS["boundary"],
                    lw=0.8, alpha=alpha)
        ax.plot(*pts[:, :2].T, color=_SCENE_COLOURS["lane"], lw=0.9,
                ls=(0, (4, 3)), alpha=alpha)

    for lane in scene["route_lanes"]:
        pts = lane[np.abs(lane[:, :2]).sum(-1) > 0]
        if len(pts) >= 2:
            ax.plot(*pts[:, :2].T, color=_SCENE_COLOURS["route"], lw=2.2, alpha=0.9 * alpha)

    agents = scene["neighbor_agents_past"][:, -1, :]      # 현재 프레임만
    for a in agents[np.abs(agents[:, :4]).sum(-1) > 0]:
        box = _rotated_box(a[0], a[1], np.arctan2(a[3], a[2]), a[6], a[7])
        ax.plot(*box.T, color=_SCENE_COLOURS["agent"], lw=1.1, alpha=alpha)

    static = scene["static_objects"]
    for s in static[np.abs(static[:, :4]).sum(-1) > 0]:
        box = _rotated_box(s[0], s[1], np.arctan2(s[3], s[2]), s[4], s[5])
        ax.fill(*box.T, color=_SCENE_COLOURS["static"], lw=0, alpha=alpha)

    ax.plot(*_rotated_box(0, 0, 0, 2.297, 5.176).T,
            color=_SCENE_COLOURS["ego"], lw=2.0, alpha=alpha, zorder=5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15 * alpha)
    if labels:
        ax.set_xlabel("x [m]  (자차 진행 방향)")
        ax.set_ylabel("y [m]")
    return ax


def plot_denoising_steps(steps, scene=None, ncols: int = 4, panel: float = 3.0,
                         cmap: str = "Blues"):
    """denoising 단계별 중간 궤적을 **단계마다 한 칸씩** 그린다.

    12 개를 한 축에 겹치면 초반 잡음이 화면을 덮어 아무것도 읽히지 않는다. 칸을 나누면
    "잡음이 접혀 들어가 궤적이 된다" 는 과정이 순서대로 보인다. 모든 칸이 같은 축 범위를
    쓰므로 칸끼리 크기를 그대로 비교할 수 있다.

    색은 단계 **순서**를 나타내는 양이라 한 색상의 밝기 단계(연함 → 진함)만 쓴다.
    순서를 읽는 것은 칸 제목이므로 색에만 기대지 않는다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(steps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows * 0.78),
                             squeeze=False)
    shades = plt.get_cmap(cmap)(np.linspace(0.35, 1.0, n))

    pts = steps[:, :, :2].reshape(-1, 2)
    pad = 8.0
    xlim = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ylim = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)

    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis("off")
            continue
        if scene is not None:
            plot_diffusion_scene(scene, ax=ax, show_boundary=False,
                                 alpha=0.55, labels=False)
        ax.plot(steps[k][:, 0], steps[k][:, 1], color=shades[k],
                lw=2.4 if k == n - 1 else 1.6, solid_capstyle="round", zorder=10)
        note = "  ← 잡음" if k == 0 else ("  ← ml_local" if k == n - 1 else "")
        ax.set_title(f"step {k}{note}", fontsize=9,
                     fontweight="bold" if k in (0, n - 1) else "normal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, axes


def plot_denoising_convergence(steps, axes=None, figsize=(11, 3.6)):
    """단계가 진행되며 궤적이 어떻게 정리되는지 두 값으로 본다.

    두 값은 단위와 크기가 달라 한 축에 겹치지 않고 패널을 나눈다.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=figsize)

    length = np.linalg.norm(np.diff(steps[:, :, :2], axis=1), axis=-1).sum(-1)
    shift = np.r_[np.nan, np.linalg.norm(np.diff(steps[:, :, :2], axis=0),
                                         axis=-1).mean(-1)]
    k = np.arange(len(steps))

    for ax, value, title, unit in (
            (axes[0], length, "궤적의 경로 길이", "m"),
            (axes[1], shift, "직전 단계 대비 평균 이동량", "m")):
        ax.plot(k, value, color="#1f6fb4", lw=2.0, marker="o", ms=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("denoising 단계")
        ax.set_ylabel(f"[{unit}]")
        ax.grid(alpha=0.25)

    # 로그축의 기본 눈금 라벨은 mathtext(10^{-1})라 유니코드 마이너스를 쓴다. 한글 폰트에는
    # 그 글리프가 없어 네모로 나오므로, 평범한 숫자로 찍는다.
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    return axes


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

    지표 이름이 길어 가로로 눕힌다. Closed-loop 곱셈 항이 포함된 표에서는 이름 앞에
    `×` 를 붙여 가중합 항과 구분한다. Open-loop 표에는 해당 표시를 붙이지 않는다.
    총점 행은 구분선 아래로 뗀다 — 같은 축에 나란히 두면 총점이 지표 하나처럼 보인다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = [c for c in breakdown.columns if c != "차이"]
    rows = list(breakdown.index)
    has_multiplicative = any(r in CLOSED_MULTIPLICATIVE for r in rows)
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
    xlabel = "공통 시나리오 평균 점수"
    if has_multiplicative:
        xlabel += "  (× = 곱셈 항)"
    ax.set_xlabel(xlabel)
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
