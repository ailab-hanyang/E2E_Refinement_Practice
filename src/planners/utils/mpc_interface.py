"""MPC refinement 전처리 + solve 를 한 곳에 모은 어댑터.

Trajectory_refinement 와 acados 를 import 하는 유일한 파일이다.
RefinementPlanner 는 이 모듈을 지연 import 하므로, acados 가 없는 환경에서는
이 파일만 로드에 실패하고 나머지 실습(ML 추론 · 채점 · 시각화)은 그대로 돌아간다.

한 스텝의 흐름:

    ML 궤적 (ego-local (80,3))
      → TrajectoryVelocityInfoGeneratorTorch   속도를 붙여 (80,4) [x,y,yaw,v] reference 생성
      → FindNearestAgentsInfo / FindNearestPedestriansInfo
                                               로그 GT 미래에서 가장 가까운 차량 10 / 보행자 5 선별
      → GetLeftRightBoundaryPointsUsingSpace   지도 feature 에서 좌/우 차선 경계점 추출
      → GetBoundaryFrontRearPointsByReferenceTrajectory
                                               경계를 차량 전/후축 기준 y 상한으로 변환
      → RefinementMPC.Solve                    (80,6) [x,y,yaw,v,ax,delta] 해

MPC 는 지도를 직접 보지 않는다 — 위에서 넣어 주는 reference 와 제약만 본다.

제약의 한계:

    - 제한속도 제약이 없다. 과속은 채점에서만 평가되고 refined 궤적이 고치지 못한다.
    - 충돌 회피 제약은 앞 1.5 초 구간에만 걸린다.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: MPC 가 기대하는 계획 구간. Solve() 는 (80,4) reference 를 전제로 인덱싱한다.
MPC_HORIZON = 80
MPC_DT = 0.1

_SO_NAME = "libacados_ocp_solver_kinematic_model.so"
#: src/planners/utils/mpc_interface.py → parents[3] 이 저장소 루트다.
_SO_PATH = (
    Path(__file__).resolve().parents[3]
    / "Trajectory_refinement" / "refinementMPC" / "c_generated_code" / _SO_NAME
)


class MpcUnavailable(RuntimeError):
    """MPC 를 초기화할 수 없을 때. planner 는 이걸 잡아 ML-only 로 강등한다."""


class RefinementMpcInterface:
    """RefinementMPC 를 planner 가 쓰기 좋은 형태로 감싼다."""

    def __init__(self, horizon: int = MPC_HORIZON, dt: float = MPC_DT) -> None:
        if horizon != MPC_HORIZON or abs(dt - MPC_DT) > 1e-9:
            raise MpcUnavailable(
                f"MPC 솔버는 horizon={MPC_HORIZON}, dt={MPC_DT} 로 컴파일되어 있습니다 "
                f"(요청: horizon={horizon}, dt={dt}). "
                f"eval_num_frames / eval_dt 를 맞추거나 솔버를 다시 빌드하세요."
            )

        # 값싼 선행 검사 — 여기서 걸리면 acados import 비용도 들이지 않는다.
        if not _SO_PATH.exists():
            raise MpcUnavailable(
                f"MPC 솔버가 빌드되지 않았습니다 ({_SO_PATH}). "
                f"먼저 실행하세요: bash script/build_mpc.sh"
            )

        from Trajectory_refinement.pluto.interface.interface_pluto import (
            ConvertPrevSolutionToCurrentTrajectory,
            FindNearestAgentsInfo,
            FindNearestPedestriansInfo,
            GetBoundaryFrontRearPointsByReferenceTrajectory,
            GetLeftRightBoundaryPointsUsingSpace,
            TrajectoryVelocityInfoGeneratorTorch,
        )
        from Trajectory_refinement.refinementMPC.refinement_mpc_solver import RefinementMPC

        self._add_velocity = TrajectoryVelocityInfoGeneratorTorch
        self._nearest_agents = FindNearestAgentsInfo
        self._nearest_pedestrians = FindNearestPedestriansInfo
        self._lane_boundaries = GetLeftRightBoundaryPointsUsingSpace
        self._boundary_limits = GetBoundaryFrontRearPointsByReferenceTrajectory
        self._prev_to_reference = ConvertPrevSolutionToCurrentTrajectory

        self._solver = RefinementMPC()
        if not getattr(self._solver, "initialized", False):
            raise MpcUnavailable("RefinementMPC 초기화에 실패했습니다.")

    # ------------------------------------------------------------------

    def solve(
        self,
        current_state: torch.Tensor,
        ml_local: torch.Tensor,
        agents_local: Dict,
        map_feature,
        prev_solution: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """이 클래스의 본체. ML 궤적을 기준선 삼아 MPC 를 한 번 푼다.

        MPC 에 넣을 재료를 세 가지 만들어 넘긴다.
          1) 기준선  - ML 궤적에 속도를 붙여 (80, 4) 로
          2) 장애물  - 주변 차량 10대 / 보행자 5명
          3) 차선 경계 - 좌우 경계를 횡방향 상한값으로
        MPC 는 지도를 직접 보지 않으므로, 이 세 가지가 MPC 가 아는 세상의 전부다.


        :param current_state: 어댑터의 data["current_state"] (1, >=6) 텐서.
            [x, y, yaw, v, ax, delta] — ego-local 이라 앞 3개는 0 이다.
        :param ml_local: 어댑터의 ml_local (80, 3) 텐서, ego-local [x, y, yaw].
        :param agents_local: agent_world.agents_to_local_info() 결과.
        :param map_feature: 어댑터의 data["map"] — 차선 경계 추출용.
        :param prev_solution: 직전 스텝의 x_pred (warm start). 없으면 None.
        :return: (x_pred (80, 6) [x, y, yaw, v, ax, delta], solver status)
                 status == 0 이 정상 수렴이다.
        """
        x0 = current_state.cpu().numpy()[0, :6].astype(np.float64)

        # ML 궤적에 속도 프로파일을 붙여 (80, 4) reference 로 만든다.
        # Solve() 가 이 배열을 in-place 로 손대므로 사본을 넘긴다.
        reference = np.asarray(self._add_velocity(ml_local), dtype=np.float64)
        if reference.shape[0] != MPC_HORIZON:
            raise ValueError(
                f"MPC reference must have {MPC_HORIZON} steps, got {reference.shape[0]}"
            )

        agent_pred, agent_width, pedestrian = self._build_obstacles(agents_local, reference, x0)
        left_lim, right_lim = self._build_boundaries(map_feature, reference, prev_solution)

        _, x_pred, status, _ = self._solver.Solve(
            x0,
            reference.copy(),
            agent_pred,
            pedestrian,
            agent_width,
            left_lim[0], left_lim[1],      # front / rear (좌)
            right_lim[0], right_lim[1],    # front / rear (우)
        )
        return np.asarray(x_pred, dtype=np.float64), int(status)

    # ------------------------------------------------------------------

    def _build_obstacles(
        self, agents_local: Dict, reference: np.ndarray, x0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """충돌 회피 제약에 넣을 주변 차량 10대와 보행자 5명을 고른다.

        주변의 모든 물체를 제약으로 넣으면 문제가 너무 커져 실시간에 못 푼다.
        그래서 기준선(ML 궤적) 근처에 있는 것만 가까운 순으로 추린다.
        """
        # (N, 80, 3) — solver 는 t=0 을 뺀 80 스텝만 본다.
        predictions = torch.as_tensor(
            agents_local["predictions"][:, :MPC_HORIZON, :], dtype=torch.float32
        )
        category = torch.as_tensor(agents_local["category"], dtype=torch.long)
        shape = torch.as_tensor(agents_local["shape"], dtype=torch.float32)

        agent_pred, agent_width = self._nearest_agents(
            predictions, category, shape, reference, x0
        )
        pedestrian = self._nearest_pedestrians(predictions, category)
        return agent_pred, agent_width, pedestrian

    def _build_boundaries(
        self, map_feature, reference: np.ndarray, prev_solution: Optional[np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """차선 밖으로 나가지 않도록, 좌우 경계를 MPC 가 쓰는 형태로 바꾼다.

        MPC 는 차선을 선으로 보지 않고 "이 지점에서 좌우로 얼마까지 허용" 이라는
        상한값으로 받는다. 그래서 경계선을 차량 앞축/뒷축 기준 횡방향 한계로 변환한다.

        변환 기준선은 직전 스텝의 MPC 해가 있으면 그것을 쓴다. 매 스텝 ML 궤적으로
        새로 잡으면 경계가 흔들려 해가 스텝마다 튄다.
        """
        left_points, right_points = self._lane_boundaries(map_feature)[:2]

        baseline = reference
        if prev_solution is not None:
            baseline = self._prev_to_reference(prev_solution)

        front_left, rear_left = self._boundary_limits(left_points, baseline, "Left")
        front_right, rear_right = self._boundary_limits(right_points, baseline, "Right")

        # solver 는 각 경계의 y 성분만 상한으로 받는다.
        return (
            (front_left[:, 1], rear_left[:, 1]),
            (front_right[:, 1], rear_right[:, 1]),
        )
