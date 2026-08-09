"""ML 추론 + MPC refinement 실습 planner.

한 스텝의 흐름:
    [1] ML 추론      어댑터가 ego-local 궤적 하나를 만든다
    [2] 채점 world   로그에서 주변 agent 의 미래를 한 번 조회한다
    [3] Refinement   [1] 을 reference 로 MPC 를 풀어 개선 궤적을 만든다
    [4] 채점         두 궤적을 같은 조건에서 nuPlan 공식 metric 으로 매긴다 (표시용)
    [5] 궤적 선택    driving_policy 에 따라 둘 중 하나를 주행 궤적으로 정한다
    [6] 기록 / 렌더  CSV 한 줄과 프레임 한 장을 남긴다

acados 가 없으면 경고만 남기고 ML-only 로 동작한다.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from src.feature_builders.eval_scene_render import EvalSceneRender
from src.feature_builders.openloop_scene_render import OpenLoopSceneRender
from src.planners.evaluator import (
    NuPlanTrajectoryScorer,
    TrajectoryScore,
    build_score_panel,
    highlight_tokens,
)

from ..scenario_manager.scenario_manager import ScenarioManager
from .ml_planner_utils import global_trajectory_to_states
from .model_adapters import PlannerModelAdapter, PlutoModelAdapter
from .utils.agent_world import agents_to_local_info

logger = logging.getLogger(__name__)

DRIVING_POLICIES = ("ml", "refine")
#: 화면 하단에 그릴 지표. open 은 Open-loop 런 전용 렌더러를 쓴다.
RENDER_MODES = ("closed", "open")

# MPC 해의 주행 가능 판정 문턱값.
_START_BEHIND_M = -0.5       # 궤적이 ego 뒤에서 시작하면 기각 (속도와 무관한 이상)
_REVERSE_STEP_M = -0.05      # 스텝당 종방향 변화가 이보다 작으면 "뒤로 감"
_REVERSE_RATIO = 0.10        # 뒤로 가는 스텝 비율이 이보다 크면 접힌 해로 본다

# 주행에 허용하는 solver status. 0=SUCCESS, 2=MAXITER(궤적 자체는 나온다).
_ACCEPTABLE_SOLVER_STATUS = (0, 2)


class RefinementPlanner(AbstractPlanner):
    """ML 궤적과 MPC 개선 궤적을 매 스텝 채점·시각화하고, 둘 중 하나로 주행한다."""

    requires_scenario: bool = True

    def __init__(
        self,
        # ── 모델 ────────────────────────────────────────────────────────
        planner: TorchModuleWrapper = None,
        model_adapter: PlannerModelAdapter = None,
        planner_ckpt: str = None,
        scenario: AbstractScenario = None,
        use_gpu: bool = True,
        # ── Refinement (실습 2) ─────────────────────────────────────────
        use_refinement: bool = True,
        # ── 채점 (실습 3) ───────────────────────────────────────────────
        eval_dt: float = 0.1,
        eval_num_frames: int = 80,
        score_safety_horizon_steps: Optional[int] = 10,  # collision/TTC 만 1.0 s 창
        score_open_loop: bool = False,         # 계획 궤적 vs 로그 ego (ADE/FDE/AHE/FHE/MR)
        # ── 주행 정책 (실습 4) ──────────────────────────────────────────
        driving_policy: str = "ml",
        refine_fallback_to_ml: bool = True,    # MPC 해가 못 쓸 상태면 ML 로 주행
        # ── 출력 ────────────────────────────────────────────────────────
        render: bool = True,
        render_mode: str = "closed",           # closed | open — 화면 하단에 그릴 지표
        save_dir: str = None,
        log_csv: bool = True,
    ) -> None:
        if driving_policy not in DRIVING_POLICIES:
            raise ValueError(
                f"driving_policy must be one of {DRIVING_POLICIES}, got {driving_policy!r}"
            )
        if render_mode not in RENDER_MODES:
            raise ValueError(
                f"render_mode must be one of {RENDER_MODES}, got {render_mode!r}"
            )

        self._scenario = scenario
        self._render = bool(render)
        self._render_mode = render_mode
        self._log_csv = bool(log_csv)
        self._use_refinement = bool(use_refinement)
        self._driving_policy = driving_policy
        self._refine_fallback_to_ml = bool(refine_fallback_to_ml)

        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # 모델 접근은 어댑터를 통한다. 맨 TorchModuleWrapper 가 오면 여기서 감싼다.
        if model_adapter is None:
            assert planner is not None, "planner 또는 model_adapter 중 하나는 반드시 필요합니다"
            model_adapter = PlutoModelAdapter(planner, planner_ckpt=planner_ckpt)
        self._adapter = model_adapter
        self._planner_feature_builder = model_adapter.feature_builder

        self._initialization: Optional[PlannerInitialization] = None
        self._scenario_manager: Optional[ScenarioManager] = None
        self._mpc = None
        self._prev_mpc_solution: Optional[np.ndarray] = None

        self._step_interval = float(eval_dt)
        self._eval_dt = float(eval_dt)
        self._eval_num_frames = int(eval_num_frames)

        self._imgs: List[np.ndarray] = []

        self._scorer = NuPlanTrajectoryScorer(
            scenario,
            dt=self._step_interval,
            horizon_steps=self._eval_num_frames,
            safety_horizon_steps=score_safety_horizon_steps,
            score_open_loop=score_open_loop,
        )

        self._save_dir = save_dir
        self._csv_path: Optional[str] = None   # PID 가 필요해 initialize() 에서 정한다

    # ------------------------------------------------------------------
    # nuPlan AbstractPlanner 인터페이스
    # ------------------------------------------------------------------

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        # ⚠ 이 메서드에 @torch.no_grad() 를 붙이면 아래 한 줄이 무효가 된다.
        torch.set_grad_enabled(False)

        self._adapter.initialize(initialization, self.device)
        self._initialization = initialization

        self._scenario_manager = ScenarioManager(
            map_api=initialization.map_api,
            ego_state=None,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )
        # feature builder 가 route/drivable area 를 planner 와 공유하게 한다.
        self._planner_feature_builder.scenario_manager = self._scenario_manager

        self._mpc = self._build_mpc()
        self._prev_mpc_solution = None

        if self._log_csv and self._csv_path is None:
            out = Path(os.getcwd()) / "trajectory_evaluator_results"
            out.mkdir(parents=True, exist_ok=True)
            self._csv_path = str(out / f"{os.getpid()}.csv")

        # 시나리오마다 planner 가 새로 만들어지는 것이 정상이지만, 노트북에서 손으로
        # 루프를 돌리면 재사용되므로 Open-loop 누적기를 여기서 명시적으로 비운다.
        self._scorer.reset_open_loop()

        self._scene_render = (
            OpenLoopSceneRender() if self._render_mode == "open" else EvalSceneRender()
        )
        self._scene_render.scenario_manager = self._scenario_manager
        self.video_dir = Path(self._save_dir) if self._save_dir else Path(os.getcwd())
        self.video_dir.mkdir(exist_ok=True, parents=True)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    @torch.no_grad()
    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """시뮬레이터가 매 스텝 부르는 함수. 이 실습의 본체다.

        아래 [1]~[6] 순서로 한 스텝을 처리하고, 시뮬레이터에 넘길 궤적 하나를 돌려준다.
        시뮬레이터는 이 궤적을 LQR 로 추종해 다음 ego 상태를 만든다.
        """
        ego_state = current_input.history.ego_states[-1]
        iteration = current_input.iteration.index

        self._scenario_manager.update_ego_state(ego_state)
        self._scenario_manager.update_drivable_area_map()

        # [1] ML 추론 — 어댑터가 ego-local 궤적 하나를 만들고 global 로 변환한다.
        ao = self._adapter.build_and_forward(current_input, self._initialization)
        ml_local = ao.ml_local.cpu().numpy()
        ml_global = self._to_global_trajectory(ml_local, ego_state)

        # [2] 채점 world — 주변 agent 의 미래를 한 번만 조회해 채점·MPC·렌더가 공유한다.
        #     비싼 호출이다 (0.64 s/step).
        tracked = self._scenario.get_tracked_objects_at_iteration(
            iteration,
            future_trajectory_sampling=TrajectorySampling(
                num_poses=self._eval_num_frames, interval_length=self._step_interval
            ),
        )
        agents_local = agents_to_local_info(
            tracked.tracked_objects.get_agents(), ego_state, self._eval_num_frames + 1
        )

        # [3] Refinement — ML 궤적을 reference 로 MPC 를 풀어 개선 궤적을 만든다.
        refine_local, refine_global, refine_ok, refine_reason = self._refine(
            ao, agents_local, ego_state, iteration
        )

        # [4] 채점 — 두 궤적을 같은 조건에서 공식 metric 으로 매긴다 (표시·비교용).
        scores = self._score_step(iteration, ego_state, ml_global, refine_global, tracked)

        # [5] 궤적 선택 — driving_policy 에 따라 실제로 주행할 궤적을 정한다.
        executed, trajectory = self._select_trajectory(
            ml_global, refine_global, refine_ok, refine_reason, iteration
        )

        # [6] 기록 / 렌더 — CSV 한 줄과 프레임 한 장을 남긴다.
        if self._log_csv and scores:
            self._append_csv(iteration, scores, executed, refine_reason)
        if self._render and scores:
            self._render_step(
                current_input, ml_local, refine_local, scores,
                agents_local["predictions"], executed, refine_reason,
            )

        trajectory = np.asarray(trajectory, dtype=np.float64)
        return InterpolatedTrajectory(
            global_trajectory_to_states(
                global_trajectory=trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=len(trajectory) * self._step_interval,
                step_interval=self._step_interval,
                include_ego_state=True,
            )
        )

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """시나리오가 끝날 때 한 번 불린다. 모아 둔 프레임을 mp4 로 저장한다.

        스텝마다 저장하면 매번 재인코딩이 일어나 크게 느려지므로 여기서 한 번에 쓴다.
        """
        report = PlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes
        )

        # 영상 저장은 시나리오가 끝날 때 한 번만 한다.
        if self._render and self._imgs:
            import imageio

            path = (
                self.video_dir
                / f"{self._scenario.log_name}_{self._scenario.token}.mp4"
            )
            imageio.mimsave(path, self._imgs, fps=10)
            logger.info("video saved to %s", path)

        if clear_stats:
            # 시나리오 간 누적 방지 — worker 가 여러 시나리오를 순차 처리한다.
            self._compute_trajectory_runtimes: List[float] = []
            self._imgs.clear()

        return report

    # ------------------------------------------------------------------
    # [3] Refinement — MPC 풀이와 해의 주행 가능 판정
    # ------------------------------------------------------------------

    def _build_mpc(self):
        """MPC 를 준비한다. acados 가 없으면 None 을 돌려주고 ML-only 로 계속 돈다.

        acados 설치는 실습에서 가장 실패하기 쉬운 단계다. 그래서 MPC 를 쓰는 코드를
        mpc_interface 한 파일에 몰아두고, 여기서 그 import 하나만 try 로 감쌌다.
        덕분에 acados 가 없어도 ML 추론 / 채점 / 시각화 실습은 그대로 진행된다.
        """
        if not self._use_refinement:
            return None
        try:
            from .utils.mpc_interface import RefinementMpcInterface

            return RefinementMpcInterface(
                horizon=self._eval_num_frames, dt=self._step_interval
            )
        except Exception as e:
            logger.warning(
                "MPC 를 쓸 수 없어 ML-only 모드로 계속합니다: %s\n"
                "  acados 설치 : bash script/install_acados.sh\n"
                "  솔버 빌드   : bash script/build_mpc.sh",
                e,
            )
            return None

    def _refine(
        self, ao, agents_local: Dict, ego_state: EgoState, iteration: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[str]]:
        """[3단계] ML 궤적을 기준선(reference)으로 삼아 MPC 를 풀어 개선 궤적을 만든다.

        MPC 는 지도를 직접 보지 않는다. 우리가 넣어 주는 기준선과 제약(차선 경계,
        주변 차량)만 보고 그 근처에서 더 부드럽고 안전한 궤적을 찾는다.

        :return: (개선궤적 ego-local, 개선궤적 global, 주행에 써도 되는지, 사유)
                 MPC 가 없거나 풀다가 예외가 나면 (None, None, False, 사유).
        """
        if self._mpc is None:
            return None, None, False, "mpc_unavailable"

        try:
            x_pred, status = self._mpc.solve(
                current_state=ao.data["current_state"],
                ml_local=ao.ml_local,
                agents_local=agents_local,
                map_feature=ao.data["map"],
                prev_solution=self._prev_mpc_solution,
            )
        except Exception as e:
            logger.warning(
                "[%s] iter=%d MPC 예외 → ML 로 계속: %s",
                getattr(self._scenario, "token", "?"), iteration, e,
            )
            return None, None, False, f"exception:{type(e).__name__}"

        # 해가 나빠도 다음 스텝의 warm start 로는 쓴다.
        self._prev_mpc_solution = x_pred

        refine_local = np.asarray(x_pred, dtype=np.float64)[:, :3]
        refine_global = self._to_global_trajectory(refine_local, ego_state)
        ok, reason = self._mpc_solution_is_usable(x_pred, status)
        return refine_local, refine_global, ok, reason

    @staticmethod
    def _mpc_solution_is_usable(
        x_pred: np.ndarray, status: int
    ) -> Tuple[bool, Optional[str]]:
        """MPC 가 내놓은 해를 실제 주행에 써도 되는지 검사한다.

        driving_policy=refine 이면 이 궤적이 그대로 차량에 들어간다. 최적화가 실패해
        발산했거나 뒤로 접힌 궤적을 그대로 실행하면 안 되므로 여기서 걸러 낸다.
        점수와는 무관한 수치 검사라는 점이 중요하다 — 어느 쪽이 더 좋은 점수인지는
        보지 않는다.

        :return: (주행 가능 여부, 사유). 주행 가능해도 참고 사유가 남을 수 있고
                 (예: 반복 한도 도달), 그 값은 CSV 와 화면에 그대로 표시된다.
        """
        x = np.asarray(x_pred, dtype=np.float64)

        if status not in _ACCEPTABLE_SOLVER_STATUS:
            return False, f"solver_status={status}"
        if not np.all(np.isfinite(x[:, :3])):
            return False, "non_finite"
        # 궤적이 ego 뒤에서 시작하는지
        if float(x[0, 0]) < _START_BEHIND_M:
            return False, "start_behind(%.2fm)" % float(x[0, 0])
        # 뒤로 가는 스텝이 많은지 (접힌 해)
        dx = np.diff(x[:, 0])
        if dx.size and float(np.mean(dx < _REVERSE_STEP_M)) > _REVERSE_RATIO:
            return False, "reverse_kink"

        # 주행 가능. status 2 는 참고용으로 사유에 남긴다.
        return True, (f"solver_status={status}" if status != 0 else None)

    # ------------------------------------------------------------------
    # [4] 채점 — 두 궤적을 같은 context 로 채점한다
    # ------------------------------------------------------------------

    def _score_step(
        self,
        iteration: int,
        ego_state: EgoState,
        ml_global: np.ndarray,
        refine_global: Optional[np.ndarray],
        tracked,
    ) -> Dict[str, TrajectoryScore]:
        """[4단계] 두 궤적을 nuPlan 공식 지표로 채점한다. 실습 3 이 보는 숫자다.

        핵심은 두 궤적을 완전히 같은 조건에서 채점한다는 점이다. 같은 ego 상태,
        같은 주변 차량, 같은 시간 창을 쓰므로 점수 차이는 오직 궤적 차이에서 온다.

        채점에 실패해도 빈 dict 만 돌려주고 주행은 계속한다. 채점은 보여주기 위한
        것이라, 여기서 예외를 터뜨려 시나리오를 통째로 잃을 이유가 없다.
        """
        try:
            ctx = self._scorer.build_context(
                iteration=iteration,
                init_ego_state=ego_state,
                # progress 사영 기준선. progress 계열은 PROGRESS_METRICS 로 만점
                # 고정이라 최종 점수에 기여하지 않으므로 구하지 않는다.
                baseline_path=None,
                tracked_objects=tracked,
            )
            # 후보마다 따로 감싼다. rf 채점이 실패했다고 ml 행까지 사라지면, Open-loop
            # 누적 프레임 수가 후보 사이에서 어긋난 채로 표에만 안 보이게 된다.
            scores: Dict[str, TrajectoryScore] = {}
            for name, traj in (("ml", ml_global), ("rf", refine_global)):
                if traj is None:
                    continue
                try:
                    scores[name] = self._scorer.score(traj, ctx, candidate=name)
                except Exception as e:
                    logger.warning("[%s] iter=%d: %s 채점 실패: %s",
                                   getattr(self._scenario, "token", "?"), iteration, name, e)
            return scores
        except Exception as e:
            logger.warning(
                "[%s] iter=%d: 채점 실패 → 렌더/CSV 생략, 주행만 계속: %s",
                getattr(self._scenario, "token", "?"), iteration, e,
            )
            return {}

    # ------------------------------------------------------------------
    # [5] 궤적 선택 — driving_policy 와 MPC 해의 유효성만 본다 (점수는 보지 않는다)
    # ------------------------------------------------------------------

    def _select_trajectory(
        self,
        ml_global: np.ndarray,
        refine_global: Optional[np.ndarray],
        refine_ok: bool,
        refine_reason: Optional[str],
        iteration: int,
    ) -> Tuple[str, np.ndarray]:
        """[5단계] 실제로 주행할 궤적을 고른다. 점수는 보지 않는다.

        어느 쪽으로 주행할지는 driving_policy 로 실행 전에 미리 정해 두고, 매 스텝
        점수가 좋은 쪽으로 갈아타지 않는다. 그래야 "ML 로 끝까지 주행한 결과" 와
        "refine 으로 끝까지 주행한 결과" 를 공정하게 비교할 수 있다 (실습 4).

        예외는 하나뿐이다. MPC 해가 수치적으로 못 쓸 상태이면 ML 로 되돌린다.
        이것도 점수 비교가 아니라 안전 장치다.

        :return: (실제로 주행한 궤적 이름 "ml" 또는 "rf", 그 궤적)
        """
        if self._driving_policy != "refine" or refine_global is None:
            return "ml", ml_global

        if refine_ok or not self._refine_fallback_to_ml:
            return "rf", refine_global

        logger.warning(
            "[%s] iter=%d MPC 해 사용 불가(%s) → ML 로 대체 주행",
            getattr(self._scenario, "token", "?"), iteration, refine_reason,
        )
        return "ml", ml_global

    # ------------------------------------------------------------------
    # [6] 기록 / 렌더 — 스텝별 CSV 와 프레임
    # ------------------------------------------------------------------

    def _append_csv(
        self,
        iteration: int,
        scores: Dict[str, TrajectoryScore],
        executed: str,
        refine_reason: Optional[str],
    ) -> None:
        """[6단계] 스텝마다 CSV 한 줄을 남긴다. 나중에 두 런을 정량 비교할 근거다.

        한 행에 ml_ 과 rf_ 접두사로 두 궤적의 지표를 나란히 적는다. 영상은 눈으로
        보는 용도이고, 숫자로 따지려면 이 CSV 가 필요하다.
        """
        ml_row = scores["ml"].as_row()
        metric_keys = [
            k for k in ml_row if k not in ("final_score", "neutralized_metrics")
        ]
        rf = scores.get("rf")
        rf_row = rf.as_row() if rf is not None else {}

        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(
                    ["log_name", "token", "iteration",
                     "driving_policy", "executed", "refine_reason",
                     "ml_final", "rf_final", "delta"]
                    + [f"ml_{k}" for k in metric_keys]
                    + [f"rf_{k}" for k in metric_keys]
                )
            delta = (rf.final - scores["ml"].final) if rf is not None else ""
            writer.writerow(
                [self._scenario.log_name, self._scenario.token, iteration,
                 self._driving_policy, executed, refine_reason or "",
                 scores["ml"].final, rf.final if rf is not None else "", delta]
                + [ml_row[k] for k in metric_keys]
                + [rf_row.get(k, "") for k in metric_keys]
            )

    def _render_step(
        self,
        current_input: PlannerInput,
        ml_local: np.ndarray,
        refine_local: Optional[np.ndarray],
        scores: Dict[str, TrajectoryScore],
        agent_future: np.ndarray,
        executed: str,
        refine_reason: Optional[str],
    ) -> None:
        """[6단계] 이번 스텝을 그림 한 장으로 그린다. 실습 3 의 화면이다.

        ML 궤적은 실선, MPC 개선 궤적은 점선으로 겹쳐 그리고, 왼쪽 아래에 두 궤적의
        점수표를 붙인다. MPC 가 없으면 표가 한 열이 되고, 있으면 ml/rf 비교 열이 된다.
        """
        # 화면 하단 한 줄. 길어지면 오른쪽 comfort 패널에 가려진다.
        summary = {"driving": "ML" if executed == "ml" else "REFINE"}
        if refine_reason:
            summary["mpc"] = refine_reason
        panel = build_score_panel(scores, summary=summary)

        img = self._scene_render.render_dagger_scene_from_simulation(
            current_input=current_input,
            initialization=self._initialization,
            route_roadblock_ids=self._scenario_manager.get_route_roadblock_ids(),
            scenario=self._scenario,
            iteration=current_input.iteration.index,
            planning_trajectory=ml_local,
            refined_trajectory=refine_local,
            # 궤적은 ML 하나와 MPC 하나뿐이다.
            candidate_trajectories=None,
            # agent 미래는 채점에 쓴 GT 로그를 그대로 그린다.
            predictions=None,
            agent_future_log=agent_future,
            evaluator_result=panel,
            reason_detail={
                "comfort": panel,
                "ttc": {name: panel.ttc.get(name) for name in scores},
                "collision": {"ml": list(highlight_tokens(panel))},
            },
            return_img=True,
        )
        self._imgs.append(img)

    # ------------------------------------------------------------------
    # 좌표 유틸 — ego-local (T,3) 을 global 로 옮긴다
    # ------------------------------------------------------------------

    @staticmethod
    def _to_global_trajectory(
        trajectory: np.ndarray, ego_state: EgoState
    ) -> np.ndarray:
        """ego 기준 상대좌표 궤적을 지도 위 절대좌표로 옮긴다.

        모델과 MPC 는 모두 ego 를 원점으로 보는 좌표계에서 동작하지만, 시뮬레이터와
        채점기는 절대좌표를 쓴다. 이 함수가 그 경계다.
        """
        traj = np.asarray(trajectory, dtype=np.float64).copy()
        if traj.ndim != 2 or traj.shape[1] < 3:
            raise ValueError("trajectory must have shape (T, >=3)")
        traj = traj[:, :3]

        origin = ego_state.rear_axle.array.astype(np.float64)
        theta0 = float(ego_state.rear_axle.heading)
        c, s = np.cos(theta0), np.sin(theta0)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)

        traj[:, :2] = traj[:, :2] @ R.T + origin
        traj[:, 2] = (traj[:, 2] + theta0 + np.pi) % (2 * np.pi) - np.pi
        return traj
