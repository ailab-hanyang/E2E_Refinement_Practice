#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
점수 종합.

``evaluate_functions`` 가 낸 개별 지표를 nuPlan 공식
``WeightedAverageMetricAggregator`` 와 동일한 식으로 하나의 스칼라로 합친다.

    final = (∏ multiplicative) × (Σ wᵢ·mᵢ / Σ wᵢ)

개별 지표 계산은 ``evaluate_functions.py``, 궤적/world 변환과 결과 출력은
``utils.py`` 에 있다. 이 파일은 그 위에 얹히는 얇은 층이다.

구조::

    utils.py               후보 궤적(numpy) → EgoState 시퀀스, 로그 GT → world
        ↓
    evaluate_functions.py  공식 metric 호출 → check_* (bool) / score_* (스칼라)
        ↓
    nuplan_evaluator.py    집계 → TrajectoryScore
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from .evaluate_functions import (
    score_drivable_area_compliance,
    score_driving_direction_compliance,
    score_ego_is_comfortable,
    score_ego_progress,
    score_no_ego_at_fault_collisions,
    score_speed_limit_compliance,
    score_time_to_collision,
    OPEN_LOOP_COMPARISON_HZ,
    OPEN_LOOP_HORIZONS,
    OPEN_LOOP_MAX_HEADING_ERROR,
    OPEN_LOOP_MAX_L2_ERROR,
    OPEN_LOOP_MAX_MISS_RATE,
    OPEN_LOOP_MISS_THRESHOLDS,
    score_open_loop_vs_expert,
)
from .utils import (
    build_ego_states_from_rollout,
    build_ego_states_from_trajectory,
    build_future_observations,
)

logger = logging.getLogger(__name__)


# --- 공식 집계 파라미터 (nuplan/planning/script/config/simulation/metric_aggregator/
#     closed_loop_nonreactive_agents_weighted_average.yaml) ---
OFFICIAL_METRIC_WEIGHTS: Dict[str, float] = {
    "ego_progress_along_expert_route": 5.0,
    "time_to_collision_within_bound": 5.0,
    "speed_limit_compliance": 4.0,
    "ego_is_comfortable": 2.0,
}
DEFAULT_METRIC_WEIGHT = 1.0

OFFICIAL_MULTIPLICATIVE_METRICS = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
)

#: 진행률 계열 metric. **기본적으로 만점(1.0)으로 고정한다.**
#
# 이유: closed-loop 시뮬에서 시뮬 ego 는 로그 ego 와 발산한다. 그런데 공식
# progress 는 분모가 "같은 시점부터의 로그(expert) 진행량"이다. 시뮬 ego 가
# 앞차에 막혀 있고 로그 ego 는 지나갔다면 분모만 커져 플래너 탓이 아닌 벌점이
# 붙는다. 발산은 에피소드가 갈수록 커지므로 **scene 마다 편향의 크기가 달라
# 공정한 절대 비교가 불가능하다.**
#
# 특히 ego_is_making_progress 는 곱셈항이라, 8 s 창에서 시뮬 ego 가 신호 대기로
# 정지했는데 로그 ego 는 진행한 구간이면 ratio < 0.2 → **점수 전체가 0** 이
# 된다. ML·refined 양쪽이 0 이 되어 δ = 0 → 그 scene 이 영구 기각되고 정지
# 구간 데이터가 조용히 사라진다.
#
# **집계에서 빼지 않고 1.0 으로 고정하는 이유**: 빼면 가중치 합이 16 → 11,
# 곱셈항이 4 → 3 으로 줄어 공식과 스케일이 달라진다. 그러면 τ 문턱을 새로
# 잡아야 하고 공식 score 와의 대조도 어려워진다. 만점 고정은 스케일(합 16,
# 곱셈 4항)을 그대로 두면서 불공정한 신호만 무력화한다.
#
# 실측값은 계속 계산해 TrajectoryScore.diagnostics 에 남긴다. 나중에 공정한
# 진행률 기준(예: 제한속도 기반 자기참조)이 마련되면 다시 켤 수 있다.
PROGRESS_METRICS = (
    "ego_progress_along_expert_route",
    "ego_is_making_progress",
)

#: Open-loop 집계 가중치 — open_loop_boxes_weighted_average.yaml 그대로.
OPEN_LOOP_METRIC_WEIGHTS = {
    "avg_l2": 1.0,
    "final_l2": 1.0,
    "avg_heading": 2.0,
    "final_heading": 2.0,
}


def aggregate_open_loop_score(means: Dict[str, float], miss_rate: float) -> float:
    """Open-loop 최종 점수. open_loop_boxes_weighted_average.yaml 과 같은 구조다.

    aggregate_official_score 와 일부러 분리했다 — 같은 함수를 재사용하면 Closed-loop 과
    Open-loop 이름공간이 섞일 길이 생긴다.

    :param means: 누적 평균 {"ade", "fde", "ahe", "fhe"} (m, m, rad, rad)
    :param miss_rate: 누적 miss rate. 0.3 을 넘으면 곱셈항이 0 이 되어 전체가 0 이다.
    """
    scores = {
        "avg_l2": max(0.0, 1.0 - means["ade"] / OPEN_LOOP_MAX_L2_ERROR),
        "final_l2": max(0.0, 1.0 - means["fde"] / OPEN_LOOP_MAX_L2_ERROR),
        "avg_heading": max(0.0, 1.0 - means["ahe"] / OPEN_LOOP_MAX_HEADING_ERROR),
        "final_heading": max(0.0, 1.0 - means["fhe"] / OPEN_LOOP_MAX_HEADING_ERROR),
    }
    total = sum(OPEN_LOOP_METRIC_WEIGHTS.values())
    weighted = sum(OPEN_LOOP_METRIC_WEIGHTS[k] * v for k, v in scores.items()) / total
    gate = 1.0 if miss_rate <= OPEN_LOOP_MAX_MISS_RATE else 0.0
    return float(gate * weighted)


# -----------------------------------------------------------------------------
# 결과 자료형
# -----------------------------------------------------------------------------

@dataclass
class TrajectoryScore:
    """한 후보 궤적의 nuPlan 정렬 채점 결과."""

    final: float
    multiplicative: Dict[str, float] = field(default_factory=dict)
    weighted: Dict[str, float] = field(default_factory=dict)
    #: Open-loop(로그 ego 대비) 지표. **집계와 분리된 이름공간**이다 —
    #: aggregate_official_score 는 multiplicative/weighted 만 읽으므로 여기에
    #: 무엇을 넣어도 Closed-loop `final` 은 변하지 않는다.
    #: 켜져 있는 동안 키 집합이 고정되어야 한다(없으면 nan). CSV 헤더가
    #: 파일당 한 번만 쓰이므로 행마다 키 수가 달라지면 열이 어긋난다.
    open_loop: Dict[str, float] = field(default_factory=dict)
    #: 시각화·오탐 진단용 per-step 신호 (위반 여부, 최초 위반 시각 등)
    diagnostics: Dict[str, object] = field(default_factory=dict)
    #: final 계산에서 1.0 으로 고정한 metric 이름.
    #: 조용히 다른 점수가 되는 것을 막기 위해 결과에 함께 남긴다.
    neutralized: Tuple[str, ...] = ()

    def as_row(self) -> Dict[str, float]:
        """CSV 로깅용 평탄화. 고정된 metric 도 **실측값**을 남긴다(집계만 1.0)."""
        row: Dict[str, float] = {"final_score": self.final}
        row.update(self.multiplicative)
        row.update(self.weighted)
        row.update(self.open_loop)
        row["neutralized_metrics"] = "|".join(self.neutralized)
        return row


# -----------------------------------------------------------------------------
# 집계
# -----------------------------------------------------------------------------

def aggregate_official_score(
    multiplicative: Dict[str, float],
    weighted: Dict[str, float],
    neutralize: Tuple[str, ...] = PROGRESS_METRICS,
) -> float:
    """
    nuPlan WeightedAverageMetricAggregator 와 동일한 집계.

        final = (∏ multiplicative) × (Σ wᵢ·mᵢ / Σ wᵢ)

    현행 TrajectoryEvaluator 와 다른 점:
      - 곱셈 항이 공식대로 4개다 (ego_is_making_progress 포함)
      - speed_limit 가중치가 4.0 이다 (현행 2.0)
      - 공식에 없는 ×0.5 페널티를 넣지 않는다
      - progress 를 후보 집합 내 상대 정규화하지 않는다

    :param neutralize: 실측값 대신 **만점(1.0)으로 고정할** metric 이름.
        기본값은 PROGRESS_METRICS — closed-loop 발산 때문에 진행률 계열은
        scene 간 공정 비교가 불가능하다(PROGRESS_METRICS 주석 참조).

        집계에서 **빼는 게 아니라 1.0 으로 두는 것**이므로 가중치 합 16 과
        곱셈 4항이 그대로 유지된다. 공식 score 와 스케일이 같아 τ 문턱을
        공식 감각(≈0.9)으로 잡을 수 있고, 공식 결과와의 대조도 가능하다.
        다만 진행률이 상수이므로 **공식 score 보다 항상 크거나 같다.**

        neutralize=() 를 넘기면 공식과 완전히 동일한 [EXACT] 집계가 된다.
    """
    pinned = set(neutralize)

    factor = 1.0
    for name in OFFICIAL_MULTIPLICATIVE_METRICS:
        value = 1.0 if name in pinned else float(multiplicative.get(name, 1.0))
        factor *= value

    weight_sum = 0.0
    weighted_sum = 0.0
    for name, raw in weighted.items():
        w = OFFICIAL_METRIC_WEIGHTS.get(name, DEFAULT_METRIC_WEIGHT)
        value = 1.0 if name in pinned else float(raw)
        weighted_sum += w * value
        weight_sum += w

    if weight_sum == 0.0:
        return 0.0
    return factor * (weighted_sum / weight_sum)


class NuPlanTrajectoryScorer:
    """
    모델 무관 궤적 채점기.

    입력이 numpy 궤적 + scenario 컨텍스트뿐이라 PLUTO / Diffusion / Flow
    어느 planner 에서도 동일하게 쓸 수 있다. 모델 feature 레이아웃에 의존하지
    않으므로 어댑터가 채점용으로 PlutoFeatureBuilder 를 다시 돌릴 필요가 없다.

    사용 예::

        scorer = NuPlanTrajectoryScorer(scenario)
        ctx = scorer.build_context(iteration, init_ego_state, baseline_path)
        s_ml     = scorer.score(ml_global_traj, ctx, candidate="ml")
        s_refine = scorer.score(refined_global_traj, ctx, candidate="rf")

    두 후보를 **같은 ctx 로** 채점해야 world 가 동일해져 점수 차이가 궤적
    차이에서만 온다.
    """

    def __init__(
        self,
        scenario: AbstractScenario,
        dt: float = 0.1,
        horizon_steps: int = 80,
        safety_horizon_steps: Optional[int] = 10,
        neutralize_metrics: Tuple[str, ...] = PROGRESS_METRICS,
        forward_simulate: bool = True,
        score_open_loop: bool = False,
        open_loop_horizons: Tuple[int, ...] = OPEN_LOOP_HORIZONS,
    ):
        """
        :param horizon_steps: 궤적 rollout·채점 창의 전체 길이. planner 출력 전체(8 s).

        :param safety_horizon_steps: **상호작용 의존 2항**(collision, TTC)만 짧은 창으로
            채점한다. None 이면 horizon_steps 전체를 쓴다.

            drivable_area / driving_direction 은 여기 **포함되지 않는다** — 지도와
            ego 궤적만으로 결정되어 꼬리가 노이즈가 아니고, 짧은 창으로 보면
            도로를 벗어나는 궤적이 라벨로 새어 들어간다(score() 주석의 실측 참조).

            근거: LQR tracker 는 궤적을 t+1.0 s 까지만 읽는다.
            lqr_tracker.yaml 의 discretization_time=0.1, tracking_horizon=10 이고
            lqr.py 에서 reference_velocity 는 t+1.0 s 단일 점, curvature_profile 은
            t+[0.0 … 0.9] 10 점으로 보간한다. 그 뒤 구간은 제어 입력에 직접 들어가지
            않으며, 0.1 s 뒤 재계획으로 폐기된다.

            이 2항은 채점 창 전체에 대한 all-or-nothing 이라(한 스텝만 위반해도 0)
            창이 길수록 꼬리 노이즈가 그대로 점수를 지배한다. 81 스텝이면 위반 기회가
            11 스텝의 7 배다. t14h 실측에서 TTC 가 Δmean −0.0246 으로 최대 감점 항이었다.

            TTC 자체는 각 스텝에서 여전히 3 s 등속 투영을 하므로 창을 줄여도
            선견성은 유지된다 — ego rollout 이 1 s 까지만 유효하면 충분하다.

            comfort 는 제외한다: nuPlan jerk 추출이 Savitzky-Golay window 15 라
            11 샘플에서는 window_length > x.size 로 실패한다. 기존대로 전체 창을 쓴다.
            speed_limit 은 지속시간 비례 연속값이라 전체 창이 자연스럽다.

        :param neutralize_metrics: 최종 점수에서 만점(1.0)으로 고정할 metric.
            기본값은 진행률 계열(PROGRESS_METRICS) — closed-loop 발산으로
            scene 간 공정 비교가 불가능하기 때문이다. 빼지 않고 고정하므로
            가중치 합 16·곱셈 4항 스케일이 유지된다. 실측값은 계속 계산되어
            multiplicative/weighted 와 diagnostics 에 남는다.

        :param forward_simulate: True 면 채점 **전에** 궤적을 LQR tracker +
            kinematic bicycle model 로 rollout 해 "실제로 주행될 궤적"을 채점한다.

            근거: 계획 궤적을 그대로 채점하면 실행 결과와 크게 어긋난다.
            val_demo 실측에서 nuPlan 최종 점수 대비 시나리오별 격차가
            −0.59 ~ +0.55 였고(Pearson 0.475), 두 방향으로 다 틀렸다 —
            LQR 이 고쳐주는 것을 벌점으로 세거나(1863deb8: 우리 0.41 ↔ 공식 1.00),
            추종 오차가 만드는 위험을 놓쳤다(15320d97: 우리 0.55 ↔ 공식 0.00).

            rollout 은 jerk 를 절반으로(1.47→0.73), 감속 피크를 −1.11→−0.23 으로
            줄인다(실측 80개 궤적). ML·refined **양쪽에 동일 적용**해야 δ 가 성립한다.
        """
        self._scenario = scenario
        self._dt = dt
        self._horizon_steps = horizon_steps
        self._safety_horizon_steps = safety_horizon_steps
        self._neutralize = tuple(neutralize_metrics)
        self._forward_simulate = bool(forward_simulate)
        self._forward_simulator = None   # 최초 사용 시 생성 (직렬화 비용 회피)
        self._last_track_err = None      # (mean, max) — 직전 rollout 의 추종 오차

        # Open-loop 누적. MR 과 누적 평균은 프레임 하나로 정의되지 않으므로 상태가 필요하다.
        # 후보 이름 → {n, sum_ade, sum_fde, sum_ahe, sum_fhe, miss[지평별 카운트]}
        self._score_open_loop = bool(score_open_loop)
        self._open_loop_horizons = tuple(open_loop_horizons)
        self._ol_acc: Dict[str, Dict] = {}

    def reset_open_loop(self) -> None:
        """Open-loop 누적을 비운다. 시나리오가 바뀔 때 planner 가 부른다.

        planner 는 보통 시나리오마다 새로 만들어지지만, 노트북의 손 루프는 planner 를
        재사용하므로 명시적으로 초기화해야 앞 시나리오의 프레임이 섞이지 않는다.
        """
        self._ol_acc.clear()

    def _score_open_loop_block(
        self, trajectory: np.ndarray, context: Dict, candidate: str
    ) -> Tuple[Dict[str, float], Dict]:
        """프레임 지표를 계산하고 공식 격자 프레임이면 누적한다.

        누적은 **공식 1 Hz 격자 프레임만** 대상으로 한다. 10 Hz 전부를 누적하면 공식값과
        다른 수로 수렴한다. 격자가 아닌 프레임에서는 직전 누적값을 그대로 내보내므로
        화면의 누적 열은 1 초에 한 번 갱신된다.
        """
        init_ego_state = context["init_ego_state"]
        r2c = init_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center

        frame, diag = score_open_loop_vs_expert(
            trajectory,
            context["expert_ego_states"],
            dt=self._dt,
            rear_to_center=r2c,
            horizons=self._open_loop_horizons,
            expert_offsets=context["expert_official_offsets"],
        )

        acc = self._ol_acc.setdefault(candidate, {
            "n": 0, "ade": 0.0, "fde": 0.0, "ahe": 0.0, "fhe": 0.0,
            "miss": [0] * len(self._open_loop_horizons),
        })

        on_grid = bool(context["on_official_grid"])
        maxde = (diag.get("open_loop_per_horizon") or {}).get("maxde", [])
        # 지평을 하나라도 못 채운 프레임은 ol_ade_m 이 NaN 이라 여기서 통째로 빠진다.
        # 남은 지평만으로 평균해 넣으면 공식보다 크게 낮은 값이 누적에 섞인다.
        if on_grid and np.isfinite(frame["ol_ade_m"]):
            acc["n"] += 1
            acc.setdefault("iters", []).append(int(context["iteration"]))
            for key in ("ade", "fde", "ahe", "fhe"):
                acc[key] += float(frame[f"ol_{key}_m" if key in ("ade", "fde")
                                        else f"ol_{key}_rad"])
            for i, thr in enumerate(OPEN_LOOP_MISS_THRESHOLDS[:len(maxde)]):
                if maxde[i] > thr:
                    acc["miss"][i] += 1
        elif on_grid:
            # 공식은 모든 격자 프레임을 채점한다. 하나라도 빠지면 누적 평균이 공식과
            # 달라지므로 조용히 넘기지 않는다.
            logger.warning("[open-loop] iter=%s(%s): 격자 프레임을 채점하지 못해 누적에서 "
                           "제외한다 — 공식 점수와 어긋난다",
                           context.get("iteration"), candidate)

        n = acc["n"]
        if n:
            means = {k: acc[k] / n for k in ("ade", "fde", "ahe", "fhe")}
            rates = [c / n for c in acc["miss"]]
            miss_rate = float(max(rates))
            score = aggregate_open_loop_score(means, miss_rate)
        else:
            means = {k: float("nan") for k in ("ade", "fde", "ahe", "fhe")}
            rates = [float("nan")] * len(acc["miss"])
            miss_rate = float("nan")
            score = float("nan")

        frame.update({
            "ol_frames": float(n),
            "ol_ade_mean": means["ade"],
            "ol_fde_mean": means["fde"],
            "ol_ahe_mean": means["ahe"],
            "ol_fhe_mean": means["fhe"],
            "ol_miss_rate": miss_rate,
            "ol_score": score,
        })
        diag.update({
            "open_loop_miss_rate_per_horizon": rates,
            "open_loop_on_grid": on_grid,
            "expert_drift_m": context.get("expert_drift_m", float("nan")),
        })
        return frame, diag

    def _rollout(self, trajectory: np.ndarray, init_ego_state) -> Optional[List]:
        """계획 궤적을 LQR+bicycle 로 rollout 해 EgoState 시퀀스를 만든다.

        실패하면 None 을 돌려주고, 호출부는 raw 채점으로 fallback 한다.
        (추종이 불가능한 궤적이 있어도 시나리오를 잃지 않도록)
        """
        from .forward_simulation.forward_simulator import (
            ForwardSimulator,
        )

        traj = np.asarray(trajectory, dtype=np.float64)[:, :3]
        n = traj.shape[0]
        if self._forward_simulator is None or self._forward_simulator.num_frames != n:
            self._forward_simulator = ForwardSimulator(dt=self._dt, num_frames=n)

        # ForwardSimulator 는 (N, T+1, S) 를 받는다 — 현재 상태를 앞에 붙인다.
        current = np.array(
            [[init_ego_state.rear_axle.x,
              init_ego_state.rear_axle.y,
              init_ego_state.rear_axle.heading]],
            dtype=np.float64,
        )
        reference = np.vstack([current, traj])[None, ...]
        try:
            rollout = self._forward_simulator.forward(reference, init_ego_state)[0]
        except Exception as e:
            logger.warning("forward simulation 실패 → raw 궤적으로 채점한다: %s", e)
            self._last_track_err = None
            return None

        # 추종 오차 — 계획 궤적과 실제 추종 결과의 위치 차이.
        # MPC 해가 충돌 제약 경계에 붙어 있으면(여유 0) 이 오차가 곧바로 위반이 된다.
        # 그 가설을 재려면 스텝별로 기록해야 한다.
        from .common.enum import StateIndex
        err = np.linalg.norm(
            rollout[1:, [StateIndex.X, StateIndex.Y]] - traj[:, :2], axis=1
        )
        self._last_track_err = (float(err.mean()), float(err.max()))

        return build_ego_states_from_rollout(rollout, init_ego_state, dt=self._dt)

    def build_context(self, iteration: int, init_ego_state, baseline_path,
                      tracked_objects=None) -> Dict:
        """
        채점 컨텍스트를 만든다. **두 후보에 재사용**해야 world 가 동일해진다.

        :param iteration: 현재 시뮬 iteration
        :param init_ego_state: t=0 ego state
        :param baseline_path: shapely LineString (progress 사영 기준)
        :param tracked_objects: 호출자가 이미 조회한
            `scenario.get_tracked_objects_at_iteration(iteration,
             future_trajectory_sampling=TrajectorySampling(horizon_steps, dt))` 결과.
            넘기면 재조회하지 않는다 — 인자가 다르면 채점 world 가 달라지므로
            **반드시 같은 sampling 으로 얻은 것**이어야 한다.
        """
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )

        # 이 쿼리는 로그 DB 에서 전 agent 의 8 초 미래를 끌어오는 비싼 호출이다
        # (실측 0.64 s/step, 40 worker 경합 하에서). planner 가 MPC preprocess 용으로
        # 이미 **같은 인자로** 한 번 부르므로, 넘겨받으면 스텝당 한 번을 아낀다.
        if tracked_objects is not None:
            tracked = tracked_objects
        else:
            tracked = self._scenario.get_tracked_objects_at_iteration(
                iteration,
                future_trajectory_sampling=TrajectorySampling(
                    num_poses=self._horizon_steps, interval_length=self._dt
                ),
            )
        num_states = self._horizon_steps + 1  # prepend_init 로 1 늘어난다
        observations = build_future_observations(
            tracked.tracked_objects,
            num_steps=num_states,
            init_time_us=init_ego_state.time_point.time_us,
            dt=self._dt,
        )

        expert_states = list(
            self._scenario.get_ego_future_trajectory(
                iteration=iteration,
                time_horizon=self._horizon_steps * self._dt,
                num_samples=self._horizon_steps,
            )
        )
        expert_states = [init_ego_state] + expert_states

        # Open-loop 지표는 로그 ego 를 기준으로 삼는다. Closed-loop 에서는 시뮬 ego 가
        # 이미 로그에서 벗어나 있고 그 이탈량이 오차에 그대로 더해지므로, 얼마나
        # 벗어났는지를 함께 남겨 화면에서 확인할 수 있게 한다.
        # Open-loop 런(log_play_back_controller)에서는 0 에 가깝다.
        try:
            log_ego = self._scenario.get_ego_state_at_iteration(iteration)
            drift = float(np.hypot(init_ego_state.center.x - log_ego.center.x,
                                   init_ego_state.center.y - log_ego.center.y))
        except Exception:
            drift = float("nan")

        # 공식 지표는 1 Hz 격자 프레임에서만 계산된다. 그 프레임인지 표시해 둔다.
        # 두 stride 의 단위가 다르다. 공식은 격자를 시나리오의 database_interval 로 정하고
        # (iteration 단위), 우리 expert_ego_states 는 eval_dt 간격으로 뽑는다(표본 단위).
        # CLI 경로에서는 둘 다 0.1 s 라 같은 값이지만, scenario_mapping 없이 만든
        # 시나리오는 0.05 s 라 갈린다.
        stride = max(1, int(round(1.0 / (self._dt * OPEN_LOOP_COMPARISON_HZ))))
        grid_stride = max(1, int(round(
            1.0 / (self._scenario.database_interval * OPEN_LOOP_COMPARISON_HZ))))

        return {
            "init_ego_state": init_ego_state,
            "observations": observations,
            "expert_ego_states": expert_states,
            "baseline_path": baseline_path,
            "iteration": int(iteration),
            "expert_drift_m": drift,
            "on_official_grid": (int(iteration) % grid_stride == 0),
            # Open-loop 를 켠 런에서만 만든다. 여기서 예외가 나도 Closed-loop 채점은
            # 멀쩡해야 하기 때문이다.
            "expert_official_offsets": (self._official_expert_offsets(int(iteration), stride)
                                        if self._score_open_loop else None),
        }

    def _official_expert_offsets(self, iteration: int, stride: int) -> List[int]:
        """1 초 표본 k(=1..max horizon) 가 볼 expert_ego_states 인덱스.

        보통은 `k*stride`(= +k 초)다. 다만 devkit 의 planner_expert_* 지표는 로그 표본을
        이렇게 잇는다::

            시나리오 구간 [0, 10, …, N]  +  연장 구간 get_ego_future_trajectory(N-10, 8s, 8)

        연장 구간의 첫 표본이 시나리오 구간의 마지막 표본과 같은 시각이라 표본 하나가
        중복되고, 그 뒤 표본은 전부 1 초씩 밀린다. 그래서 시나리오 후반부 프레임에서는
        공식 지표가 "계획 +k 초 vs 로그 +(k-1) 초" 를 비교한다. 우리 표의 숫자가 공식
        parquet 과 같아야 하므로 같은 짝을 만든다.

        (`iteration + k*stride` 가 시나리오 마지막 인덱스를 넘어가는 k 부터 밀린다.)

        offsets 는 **expert_ego_states 인덱스**(eval_dt 간격)이고 iteration 은 **시나리오
        iteration**(database_interval 간격)이라, 경계를 볼 때는 단위를 맞춰야 한다.
        """
        k_max = max(self._open_loop_horizons)
        offsets = [k * stride for k in range(1, k_max + 1)]
        last = self._scenario.get_number_of_iterations() - 1
        ratio = max(1, int(round(self._dt / self._scenario.database_interval)))
        return [off if iteration + off * ratio <= last else max(off - stride, stride)
                for off in offsets]

    def score(
        self,
        trajectory: np.ndarray,
        context: Dict,
        forward_simulate: Optional[bool] = None,
        candidate: str = "ml",
    ) -> TrajectoryScore:
        """
        후보 궤적 하나를 채점한다.

        :param trajectory: (T, 3) global rear-axle [x, y, heading]
        :param context: build_context() 결과
        :param forward_simulate: None 이면 생성자 설정을 따른다.
            False 를 명시하면 raw 계획 궤적을 채점한다 — 두 방식을 나란히
            로깅해 forward simulation 의 효과를 재려면 이렇게 쓴다.
        :param candidate: Open-loop 누적을 후보별로 분리하는 키. 스텝당 ml·rf
            두 번 불리므로 구분하지 않으면 한 후보에 두 번 누적된다.
        """
        init_ego_state = context["init_ego_state"]
        use_fs = self._forward_simulate if forward_simulate is None else bool(forward_simulate)

        ego_states = None
        if use_fs:
            ego_states = self._rollout(trajectory, init_ego_state)
        rolled_out = ego_states is not None
        if ego_states is None:
            ego_states = build_ego_states_from_trajectory(
                trajectory, init_ego_state, dt=self._dt, prepend_init=True
            )
        observations = context["observations"]
        if observations is not None and len(observations) != len(ego_states):
            # 길이 불일치는 조용히 넘기면 프레임이 어긋난 채로 채점된다. 잘라 맞춘다.
            n = min(len(observations), len(ego_states))
            ego_states = ego_states[:n]
            observations = observations[:n]

        diagnostics: Dict[str, object] = {"forward_simulated": rolled_out}
        if rolled_out and self._last_track_err is not None:
            diagnostics["track_err_mean"] = self._last_track_err[0]
            diagnostics["track_err_max"] = self._last_track_err[1]

        # Open-loop 를 Closed-loop 지표보다 먼저 계산한다.
        # ⚠ ego_states(rollout 결과)가 아니라 **계획 궤적 원본**을 쓴다 — 공식 지표는
        #   planner 가 내놓은 궤적을 그대로 보기 때문이다. 그래서 아래 지표들과 의존관계가
        #   없고, 순서를 앞에 두면 Closed-loop 지표가 하나 실패해도 격자 프레임을 잃지
        #   않는다. 프레임 하나가 빠지면 누적 평균이 공식과 어긋난다(실측 SCORE 1.7e-3).
        open_loop: Dict[str, float] = {}
        if self._score_open_loop:
            open_loop, ol_diag = self._score_open_loop_block(
                trajectory, context, candidate)
            diagnostics.update(ol_diag)

        # 위반형 4항은 LQR 이 실제로 소비하는 창(기본 10 스텝 = 1.0 s)까지만 본다.
        # 나머지(speed_limit, comfort, progress)는 전체 창을 쓴다 — 자세한 근거는
        # __init__ 의 safety_horizon_steps 참조.
        n_safe = len(ego_states)
        if self._safety_horizon_steps is not None:
            n_safe = min(n_safe, self._safety_horizon_steps + 1)  # init state 포함
        safe_states = ego_states[:n_safe]
        safe_obs = observations[:n_safe] if observations is not None else None
        diagnostics["safety_horizon_states"] = n_safe

        # drivable / direction 은 **전체 창**을 본다. 짧은 창을 쓰면 안 되는 이유:
        #   이 둘은 지도 + ego 궤적만으로 결정된다. 두 입력 다 어느 시점에서든
        #   정확히 알 수 있으므로 꼬리가 노이즈가 아니다 — 도로를 벗어나는 계획은
        #   재계획 여부와 무관하게 벗어나는 계획이다.
        #   반면 TTC/collision 은 주변 agent 의 미래에 의존하고 그 world 는
        #   시간이 갈수록 실제와 벌어지므로, 거기서만 꼬리가 노이즈다.
        #
        #   실측(b703d99845f45666 iter96): refined 가 t>4 s 에서 도로를 이탈하는데
        #   1.0/2.0/4.0 s 창에서는 dac=ddc=1.00 → PASS, 8.0 s 에서만 0.00 이었다.
        #   저장되는 라벨(ego_agent_future)은 8 s 전체이므로, 짧은 창으로 판정하면
        #   **도로를 벗어나는 궤적이 그대로 학습 라벨로 들어간다.**
        dac, d = score_drivable_area_compliance(self._scenario, ego_states)
        diagnostics.update(d)
        ddc, d = score_driving_direction_compliance(self._scenario, ego_states)
        diagnostics.update(d)
        collision, d = score_no_ego_at_fault_collisions(
            self._scenario, safe_states, safe_obs
        )
        diagnostics.update(d)
        ttc, d = score_time_to_collision(self._scenario, safe_states, safe_obs)
        diagnostics.update(d)
        speed_limit, d = score_speed_limit_compliance(self._scenario, ego_states)
        diagnostics.update(d)
        comfort, d = score_ego_is_comfortable(ego_states, dt=self._dt)
        diagnostics.update(d)
        # 진행률은 기본적으로 집계에서 1.0 으로 고정되지만 실측값은 계산해 남긴다.
        # (나중에 공정한 기준이 마련되면 neutralize_metrics=() 로 되살릴 수 있다)
        progress, making_progress, d = score_ego_progress(
            ego_states, context["expert_ego_states"], context["baseline_path"]
        )
        diagnostics.update(d)

        multiplicative = {
            "no_ego_at_fault_collisions": collision,
            "drivable_area_compliance": dac,
            "ego_is_making_progress": making_progress,
            "driving_direction_compliance": ddc,
        }
        weighted = {
            "ego_progress_along_expert_route": progress,
            "time_to_collision_within_bound": ttc,
            "speed_limit_compliance": speed_limit,
            "ego_is_comfortable": comfort,
        }

        return TrajectoryScore(
            final=aggregate_official_score(
                multiplicative, weighted, neutralize=self._neutralize
            ),
            multiplicative=multiplicative,
            weighted=weighted,
            open_loop=open_loop,
            diagnostics=diagnostics,
            neutralized=self._neutralize,
        )
