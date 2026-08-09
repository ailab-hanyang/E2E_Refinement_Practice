"""주변 agent 의 로그(GT) 미래를 ego-local 배열로 만드는 변환기.

nuPlan 이 로그에서 뽑아 준 각 agent 의 8 초 미래를 ego rear-axle 기준 상대좌표로 바꾼다.
이것이 채점 world 이고, 세 곳(planner 의 렌더 · mpc_interface 의 충돌 제약 · 채점기)이
같은 조회 결과를 나눠 쓴다.

numpy 외에 의존이 없다 — acados 가 없는 환경에서도 그대로 쓸 수 있다.
"""

from typing import Dict, List

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

#: MPC 의 TTC/충돌 제약이 기대하는 카테고리 코드.
CATEGORY_VEHICLE = 3
CATEGORY_PEDESTRIAN = 1

_CATEGORY_BY_TYPE = {
    TrackedObjectType.VEHICLE: CATEGORY_VEHICLE,
    TrackedObjectType.BICYCLE: CATEGORY_VEHICLE,  # 자전거는 차량으로 취급한다
    TrackedObjectType.PEDESTRIAN: CATEGORY_PEDESTRIAN,
}


def agents_to_local_info(agents, ego_state: EgoState, num_steps: int = 81) -> Dict:
    """주변 차량들의 로그상 미래 8초를 ego 기준 상대좌표 배열로 바꾼다.

    이 결과가 "채점이 보는 세상" 이다. 화면에 그리는 회색 궤적, MPC 의 충돌 제약,
    채점기가 쓰는 주변 차량이 모두 여기서 나온 같은 데이터다. 셋이 서로 다른 세상을
    보면 화면의 충돌 표시와 실제 감점이 어긋나 실습에서 설명할 수 없게 된다.


    :param agents: tracked_objects.get_agents() 결과. 각 agent 는 로그 GT 미래를
        agent.predictions[0].waypoints 로 들고 있다.
    :param ego_state: 변환 기준이 되는 현재 ego state (rear axle).
    :param num_steps: t=0 을 포함한 스텝 수. 기본 81 = 현재 + 8 초(0.1 s × 80).
    :return: dict
        tokens       agent 토큰 리스트 (len N)
        shape        (N, 2) [width, length]
        category     길이 N 의 카테고리 코드 리스트
        velocity     (N, num_steps) 속력 [m/s]
        predictions  (N, num_steps, 3) ego-local [x, y, yaw]
    """
    if len(agents) == 0:
        return {
            "tokens": [],
            "shape": np.zeros((0, 2), dtype=np.float64),
            "category": [],
            "velocity": np.zeros((0, num_steps), dtype=np.float64),
            "predictions": np.zeros((0, num_steps, 3), dtype=np.float64),
        }

    origin = ego_state.rear_axle.array
    angle = ego_state.rear_axle.heading
    # global → local 회전 (전치 행렬을 오른쪽에서 곱하는 형태)
    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    tokens: List[str] = []
    shapes: List[List[float]] = []
    categories: List[int] = []
    velocities: List[np.ndarray] = []
    predictions: List[np.ndarray] = []

    for agent in agents:
        tokens.append(agent.metadata.token)
        shapes.append([agent.box.width, agent.box.length])
        categories.append(_CATEGORY_BY_TYPE.get(agent.tracked_object_type, CATEGORY_VEHICLE))

        # t=0 은 항상 현재 상태
        poses = [[agent.center.x, agent.center.y, agent.center.heading]]
        speeds = [float(np.hypot(agent.velocity.x, agent.velocity.y))]

        waypoints = agent.predictions[0].waypoints if agent.predictions else []
        for wp in waypoints[: num_steps - 1]:
            if wp is None or wp._oriented_box is None:
                # 결측 waypoint 는 마지막 유효 상태를 유지한다 (정지로 간주하지 않는다)
                poses.append(poses[-1])
                speeds.append(speeds[-1])
                continue
            box = wp._oriented_box.center
            poses.append([box.x, box.y, box.heading])
            speeds.append(float(np.hypot(wp.velocity.x, wp.velocity.y)))

        # 길이가 모자라면 마지막 상태로 패딩
        while len(poses) < num_steps:
            poses.append(poses[-1])
            speeds.append(speeds[-1])

        arr = np.asarray(poses[:num_steps], dtype=np.float64)
        local = np.concatenate(
            [
                np.matmul(arr[:, :2] - origin, rot_mat),
                (arr[:, 2] - angle)[:, None],
            ],
            axis=-1,
        )
        predictions.append(local)
        velocities.append(np.asarray(speeds[:num_steps], dtype=np.float64))

    return {
        "tokens": tokens,
        "shape": np.asarray(shapes, dtype=np.float64),
        "category": categories,
        "velocity": np.asarray(velocities, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.float64),
    }
