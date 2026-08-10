"""Closed-loop에서 실제로 실행된 history의 step·누적 지표 추적기."""

from typing import Dict, Tuple


CLOSED_BREAKDOWN = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "ego_is_comfortable",
)

CLOSED_MULTIPLICATIVE = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
)

CLOSED_WEIGHTS = {
    "ego_progress_along_expert_route": 5.0,
    "time_to_collision_within_bound": 5.0,
    "speed_limit_compliance": 4.0,
    "ego_is_comfortable": 2.0,
}


def aggregate_closed_loop_metrics(metrics: Dict[str, float]) -> float:
    """nuPlan Closed-loop의 곱셈 항 × 가중평균 식을 적용한다."""
    gate = 1.0
    for name in CLOSED_MULTIPLICATIVE:
        gate *= float(metrics[name])

    total_weight = sum(CLOSED_WEIGHTS.values())
    weighted = sum(
        weight * float(metrics[name])
        for name, weight in CLOSED_WEIGHTS.items()
    ) / total_weight
    return float(gate * weighted)


class RealtimeNuPlanMetricTracker:
    """실제 Closed-loop state·observation을 모아 step·prefix 지표를 계산한다."""

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

    def _values(self, *, current_step: bool) -> Dict[str, float]:
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
            self.scenario, current_states, current_observations
        )
        drivable, _ = fn["drivable"](self.scenario, current_states)
        direction, _ = fn["direction"](self.scenario, direction_states)
        ttc, _ = fn["ttc"](
            self.scenario, current_states, current_observations
        )
        speed, _ = fn["speed"](self.scenario, current_states)

        comfort = 1.0
        if len(comfort_states) >= 5:
            comfort, _ = fn["comfort"](comfort_states, dt=dt)

        progress, making_progress, _ = fn["progress"](
            self.ego_states, self.expert_ego_states, self.baseline_path
        )

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

    def update(
        self,
        ego_state,
        observation,
        expert_ego_state,
        baseline_path=None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """현재 실행 상태를 추가하고 ``(step, cumulative)``을 반환한다."""
        self.ego_states.append(ego_state)
        self.observations.append(observation)
        self.expert_ego_states.append(expert_ego_state)
        if self.baseline_path is None and baseline_path is not None:
            self.baseline_path = baseline_path
        step = self._values(current_step=True)
        cumulative = self._values(current_step=False)

        # Collision은 독립 pose 검사가 아니라 최초 접촉부터 track id를 기억하는 이벤트 지표다.
        # 단일 snapshot을 새 충돌로 재분류하면 step=0, prefix=1이라는 모순이 생길 수 있다.
        # 따라서 step은 official history gate가 이번 update에서 처음 1→0으로 바뀌었는지를 뜻한다.
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


def score_executed_history_step(
    tracker: RealtimeNuPlanMetricTracker,
    ego_state,
    observation,
    expert_ego_state,
    baseline_path=None,
) -> Dict[str, object]:
    """CSV 한 행에 바로 펼칠 실제 실행 step·누적 지표를 만든다."""
    step, cumulative = tracker.update(
        ego_state=ego_state,
        observation=observation,
        expert_ego_state=expert_ego_state,
        baseline_path=baseline_path,
    )
    return {
        "step": step,
        "cumulative": cumulative,
        "step_final": aggregate_closed_loop_metrics(step),
        "cumulative_final": aggregate_closed_loop_metrics(cumulative),
    }
