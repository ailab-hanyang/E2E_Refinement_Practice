#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator utilities — 채점 자체가 아닌 보조 기능.

세 종류가 들어있다:

1. **EgoState 시퀀스 추출** (`get_ego_states_*`)
   scenario 로부터 로그 ego 궤적을 뽑는다. 사후 분석용.

2. **채점 입력 구성** (`build_ego_states_from_trajectory`, `build_future_observations`)
   임의의 후보 궤적(numpy)과 로그 GT 미래를 nuPlan 공식 metric 이 요구하는
   형태로 변환한다. **모델 무관 경계**가 여기다 — 이 위로는 numpy 만 오간다.

3. **결과 출력** (`write_violation_csv`, `generate_*_chart`)
   위반 통계를 CSV·차트로 떨어뜨린다. matplotlib 의존은 이 파일에만 있다.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import contextlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _headless_chart():
    """차트를 파일로만 저장할 때만 Agg 를 쓴다. 대화형 세션(노트북 등)의 backend 는 건드리지 않는다."""
    prev = matplotlib.get_backend()
    matplotlib.use('Agg')
    try:
        yield
    finally:
        matplotlib.use(prev)


# -----------------------------------------------------------------------------
# 로그 ego 궤적 추출
# -----------------------------------------------------------------------------

def get_ego_states_from_trajectory(
    scenario: AbstractScenario,
    iteration: int = 0,
    history_horizon: float = 2.0,
    future_horizon: float = 8.0,
    sample_interval: float = 0.1,
) -> List:
    """
    FeatureBuilder와 동일한 방식으로 ego trajectory를 추출합니다.
    history 2초 + current + future 8초의 ego state list를 반환합니다.

    :param scenario: AbstractScenario 인스턴스
    :param iteration: 시작 iteration (default: 0)
    :param history_horizon: 과거 시간 (초, default: 2.0)
    :param future_horizon: 미래 시간 (초, default: 8.0)
    :param sample_interval: 샘플링 간격 (초, default: 0.1)
    :return: ego state list (past + current + future)
    """
    # Calculate number of samples
    history_samples = int(history_horizon / sample_interval)
    future_samples = int(future_horizon / sample_interval)

    # Get current ego state
    ego_cur_state = scenario.get_ego_state_at_iteration(iteration)

    # Get past trajectory
    past_ego_trajectory = scenario.get_ego_past_trajectory(
        iteration=iteration,
        time_horizon=history_horizon,
        num_samples=history_samples,
    )

    # Get future trajectory
    future_ego_trajectory = scenario.get_ego_future_trajectory(
        iteration=iteration,
        time_horizon=future_horizon,
        num_samples=future_samples,
    )

    # Combine: past + current + future
    ego_state_list = (
        list(past_ego_trajectory) + [ego_cur_state] + list(future_ego_trajectory)
    )

    return ego_state_list


def get_ego_states_at_all_iterations(scenario: AbstractScenario, use_full_scenario: bool = True) -> List:
    """
    시나리오의 모든 iteration에서 ego vehicle의 상태를 추출합니다.
    :param scenario: AbstractScenario 인스턴스.
    :param use_full_scenario: True면 전체 시나리오, False면 trajectory sampling 범위만.
    :return: ego state list.
    """
    if use_full_scenario:
        # Use all iterations (full scenario)
        num_iterations = scenario.get_number_of_iterations()
        ego_states = list(scenario.get_ego_trajectory_slice(0, num_iterations))
    else:
        # Use Trajectory sampling (past 2s + current + future 8s)
        ego_states = get_ego_states_from_trajectory(scenario)

    return ego_states


# -----------------------------------------------------------------------------
# 후보 궤적 → 채점 입력
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 궤적 → EgoState 시퀀스
# -----------------------------------------------------------------------------

def build_ego_states_from_trajectory(
    trajectory: np.ndarray,
    init_ego_state,
    dt: float = 0.1,
    prepend_init: bool = True,
) -> List:
    """
    임의의 후보 궤적을 nuPlan EgoState 시퀀스로 변환한다.

    공식 metric 클래스들은 전부 EgoState 시퀀스를 입력으로 받으므로, 이 함수가
    "모델 출력 → 공식 채점" 사이의 유일한 어댑터다. 입력이 numpy 이므로
    PLUTO / Diffusion / Flow 어느 모델이든 동일하게 쓸 수 있다.

    :param trajectory: (T, 3) global [x, y, heading] — **rear axle 기준**.
        planner 의 _to_global_trajectory() 출력 규약과 같다.
    :param init_ego_state: 궤적 시작 시점의 실제 EgoState (차량 제원·시각·초기 속도 제공)
    :param dt: 궤적 샘플 간격 [s]
    :param prepend_init: True 면 init_ego_state 의 pose 를 궤적 앞에 붙인다.
        속도/가속도를 차분으로 얻으려면 시작점이 필요하다.
    :return: EgoState 리스트 (길이 T 또는 T+1)
    """
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateSE2,
        StateVector2D,
        TimePoint,
    )

    poses = np.asarray(trajectory, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"trajectory must be (T,3); got {poses.shape}")
    poses = poses[:, :3].copy()

    if prepend_init:
        init_pose = np.array(
            [[init_ego_state.rear_axle.x,
              init_ego_state.rear_axle.y,
              init_ego_state.rear_axle.heading]],
            dtype=np.float64,
        )
        poses = np.concatenate([init_pose, poses], axis=0)

    n = len(poses)
    if n < 2:
        raise ValueError("need at least 2 poses to derive velocity")

    # heading 은 연속화해야 각속도 차분이 ±π 에서 튀지 않는다
    poses[:, 2] = np.unwrap(poses[:, 2])

    # 전역 속도 → body frame (전진 x, 횡 y)
    d_xy = np.diff(poses[:, :2], axis=0) / dt                      # (n-1, 2)
    hd = poses[:-1, 2]
    v_lon = d_xy[:, 0] * np.cos(hd) + d_xy[:, 1] * np.sin(hd)
    v_lat = -d_xy[:, 0] * np.sin(hd) + d_xy[:, 1] * np.cos(hd)
    # 마지막 스텝은 직전 값 유지 (차분이 정의되지 않음)
    v_lon = np.append(v_lon, v_lon[-1])
    v_lat = np.append(v_lat, v_lat[-1])

    a_lon = np.append(np.diff(v_lon) / dt, 0.0)
    a_lat = np.append(np.diff(v_lat) / dt, 0.0)

    yaw_rate = np.append(np.diff(poses[:, 2]) / dt, 0.0)
    yaw_accel = np.append(np.diff(yaw_rate) / dt, 0.0)

    # 시작 상태는 실제 관측값을 그대로 쓴다 (차분 추정보다 정확하다)
    if prepend_init:
        v_lon[0] = init_ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        v_lat[0] = init_ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        a_lon[0] = init_ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        a_lat[0] = init_ego_state.dynamic_car_state.rear_axle_acceleration_2d.y

    vehicle = init_ego_state.car_footprint.vehicle_parameters
    t0_us = init_ego_state.time_point.time_us
    dt_us = int(round(dt * 1e6))
    wheel_base = float(vehicle.wheel_base)

    ego_states: List = []
    for i in range(n):
        # 자전거 모델 역산: delta = atan(L * yaw_rate / v). 저속에서는 0 으로 둔다.
        speed = float(v_lon[i])
        if abs(speed) > 0.1:
            steering = float(np.arctan(wheel_base * yaw_rate[i] / speed))
        else:
            steering = 0.0

        ego_states.append(
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(poses[i, 0], poses[i, 1], poses[i, 2]),
                rear_axle_velocity_2d=StateVector2D(float(v_lon[i]), float(v_lat[i])),
                rear_axle_acceleration_2d=StateVector2D(float(a_lon[i]), float(a_lat[i])),
                tire_steering_angle=steering,
                time_point=TimePoint(t0_us + i * dt_us),
                vehicle_parameters=vehicle,
                is_in_auto_mode=True,
                angular_vel=float(yaw_rate[i]),
                angular_accel=float(yaw_accel[i]),
            )
        )

    return ego_states


def build_future_observations(
    tracked_objects,
    num_steps: int,
    init_time_us: int,
    dt: float = 0.1,
) -> List:
    """
    로그의 GT 미래 궤적으로 스텝별 world 를 만든다.

    데이터 취득은 **non-reactive** 로 진행하므로 주변 agent 는 로그 재생이다.
    따라서 scenario.get_tracked_objects_at_iteration(k, future_trajectory_sampling=...)
    한 번 호출로 얻은 GT 미래를 그대로 펼치면 채점 구간의 world 가 된다.
    scenario iteration 을 k, k+1, … 로 재조회하지 않으므로 database_interval 과
    시뮬 dt 의 불일치 문제가 생기지 않는다.

    ML 과 refined 에 **같은 world** 가 적용되는 것이 중요하다. 절대 점수는
    낙관적이지만 게이트가 보는 것은 두 점수의 차이(δ)와 문턱(τ)이다.

    :param tracked_objects: TrackedObjects (agent 의 predictions 에 GT 미래 포함)
    :param num_steps: 만들 스텝 수 (ego_states 길이와 같아야 함)
    :param init_time_us: t=0 의 timestamp [us]
    :param dt: 스텝 간격 [s]
    :return: DetectionsTracks 리스트 (길이 num_steps)
    """
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

    agents = tracked_objects.get_agents()
    static_objects = [
        obj for obj in tracked_objects.tracked_objects if obj not in agents
    ]

    from nuplan.common.actor_state.state_representation import StateVector2D

    # 미래가 끊긴 뒤 고정된 agent 에 쓸 속도. pose 가 안 움직이는데 속도만 살아 있으면
    # TTC 의 등속 투영이 실재하지 않는 상자를 밀어내 위반을 만든다.
    _FROZEN_VELOCITY = StateVector2D(0.0, 0.0)

    # agent 별 GT 미래 (pose, velocity) 시퀀스를 미리 뽑아둔다
    agent_futures = []
    for agent in agents:
        waypoints = None
        predictions = getattr(agent, "predictions", None)
        if predictions:
            # GT 재생이므로 모드가 하나뿐이다
            waypoints = predictions[0].valid_waypoints
        agent_futures.append((agent, waypoints))

    observations: List = []

    for step in range(num_steps):
        objects_at_step: List = list(static_objects)

        for agent, waypoints in agent_futures:
            if waypoints and step < len(waypoints):
                # GT 미래가 있는 구간: pose 와 velocity 를 **둘 다** 그 시점 값으로 쓴다.
                # velocity 를 t=0 값으로 고정하면 감속·정지하는 차량이 80 스텝 내내
                # 초기 속도로 투영돼 TTC 가 허위 위반을 낸다.
                wp = waypoints[step]
                pose = wp.center
                velocity = wp.velocity if wp.velocity is not None else agent.velocity
            elif step == 0:
                # 미래가 아예 없어도 t=0 은 실제 관측이다
                pose = agent.center
                velocity = agent.velocity
            else:
                # 미래가 끊긴 구간: 마지막 pose 로 고정한다(사라지게 두면 충돌 판정이
                # 낙관적으로 치우친다). 고정된 이상 속도도 0 이어야 자기모순이 없다.
                pose = waypoints[-1].center if waypoints else agent.center
                velocity = _FROZEN_VELOCITY

            objects_at_step.append(
                Agent(
                    tracked_object_type=agent.tracked_object_type,
                    oriented_box=OrientedBox.from_new_pose(
                        agent.box, StateSE2(pose.x, pose.y, pose.heading)
                    ),
                    velocity=velocity,
                    metadata=agent.metadata,
                )
            )

        observations.append(DetectionsTracks(TrackedObjects(objects_at_step)))

    return observations


# -----------------------------------------------------------------------------
# 개별 metric → 스칼라

def build_ego_states_from_rollout(rollout_states, init_ego_state, dt: float = 0.1) -> List:
    """
    ForwardSimulator 의 rollout 배열을 nuPlan EgoState 시퀀스로 변환한다.

    `ForwardSimulator.forward()` 는 `(N, T+1, StateIndex.size())` 배열을 돌려준다.
    그 안의 속도·가속도·조향각은 **kinematic bicycle model 이 적분해 만든 값**이므로
    위치를 차분해 유도할 필요가 없다(그게 이 함수가 build_ego_states_from_trajectory
    와 다른 점이다).

    :param rollout_states: (T+1, StateIndex.size()) — 배치 차원을 이미 벗긴 것
    :param init_ego_state: 차량 제원·시작 시각 제공
    :param dt: 스텝 간격 [s]
    """
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateSE2,
        StateVector2D,
        TimePoint,
    )

    from .common.enum import StateIndex

    r = np.asarray(rollout_states, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"rollout_states must be (T+1, S); got {r.shape}")

    vehicle = init_ego_state.car_footprint.vehicle_parameters
    t0_us = init_ego_state.time_point.time_us
    dt_us = int(round(dt * 1e6))

    out: List = []
    for i in range(r.shape[0]):
        s = r[i]
        out.append(
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(
                    s[StateIndex.X], s[StateIndex.Y], s[StateIndex.HEADING]
                ),
                rear_axle_velocity_2d=StateVector2D(
                    s[StateIndex.VELOCITY_X], s[StateIndex.VELOCITY_Y]
                ),
                rear_axle_acceleration_2d=StateVector2D(
                    s[StateIndex.ACCELERATION_X], s[StateIndex.ACCELERATION_Y]
                ),
                tire_steering_angle=float(s[StateIndex.STEERING_ANGLE]),
                time_point=TimePoint(t0_us + i * dt_us),
                vehicle_parameters=vehicle,
                is_in_auto_mode=True,
                angular_vel=float(s[StateIndex.ANGULAR_VELOCITY]),
                angular_accel=float(s[StateIndex.ANGULAR_ACCELERATION]),
            )
        )
    return out


# -----------------------------------------------------------------------------
# 결과 출력 (CSV / 차트)
# -----------------------------------------------------------------------------

def write_violation_csv(violation_results: List[Dict], output_dir: Path) -> None:
    """
    Write violation results to CSV files (5 files: per-violation-type + overall + all results).
    :param violation_results: List of violation result dictionaries
    :param output_dir: Directory to save CSV files
    """
    if not violation_results:
        logger.warning("No violation results to write to CSV")
        return

    fieldnames = [
        'scenario_type', 'log_name', 'token',
        'speed_limit_violated', 'speed_limit_first_time',
        'drivable_area_violated', 'drivable_area_first_time',
        'driving_direction_violated', 'driving_direction_first_time',
        'ego_at_fault_collision_violated', 'ego_at_fault_collision_first_time',
        'time_to_collision_violated', 'time_to_collision_first_time'
    ]

    # Helper function to write CSV
    def write_csv(filepath: Path, data: List[Dict], filter_key: str = None):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in data:
                writer.writerow({
                    'scenario_type': result['scenario_type'],
                    'log_name': result['log_name'],
                    'token': result['token'],
                    'speed_limit_violated': result.get('speed_limit_violated', False),
                    'speed_limit_first_time': f"{result['speed_limit_first_time']:.1f}s" if result.get('speed_limit_first_time') is not None else 'N/A',
                    'drivable_area_violated': result.get('drivable_area_violated', False),
                    'drivable_area_first_time': f"{result['drivable_area_first_time']:.1f}s" if result.get('drivable_area_first_time') is not None else 'N/A',
                    'driving_direction_violated': result.get('driving_direction_violated', False),
                    'driving_direction_first_time': f"{result['driving_direction_first_time']:.1f}s" if result.get('driving_direction_first_time') is not None else 'N/A',
                    'ego_at_fault_collision_violated': result.get('ego_at_fault_collision_violated', False),
                    'ego_at_fault_collision_first_time': f"{result['ego_at_fault_collision_first_time']:.1f}s" if result.get('ego_at_fault_collision_first_time') is not None else 'N/A',
                    'time_to_collision_violated': result.get('time_to_collision_violated', False),
                    'time_to_collision_first_time': f"{result['time_to_collision_first_time']:.1f}s" if result.get('time_to_collision_first_time') is not None else 'N/A'
                })

    # Filter by violation type
    speed_violations = [r for r in violation_results if r.get('speed_limit_violated', False)]
    drivable_violations = [r for r in violation_results if r.get('drivable_area_violated', False)]
    direction_violations = [r for r in violation_results if r.get('driving_direction_violated', False)]
    collision_violations = [r for r in violation_results if r.get('ego_at_fault_collision_violated', False)]
    ttc_violations = [r for r in violation_results if r.get('time_to_collision_violated', False)]
    overall_violations = [
        r for r in violation_results
        if r.get('speed_limit_violated', False) or r.get('drivable_area_violated', False) or
           r.get('driving_direction_violated', False) or r.get('ego_at_fault_collision_violated', False) or
           r.get('time_to_collision_violated', False)
    ]

    # Write 7 CSV files (5 violation types + overall + all results)
    write_csv(output_dir / "speed_limit_violations.csv", speed_violations)
    write_csv(output_dir / "drivable_area_violations.csv", drivable_violations)
    write_csv(output_dir / "driving_direction_violations.csv", direction_violations)
    write_csv(output_dir / "ego_at_fault_collision_violations.csv", collision_violations)
    write_csv(output_dir / "time_to_collision_violations.csv", ttc_violations)
    write_csv(output_dir / "overall_violations.csv", overall_violations)
    write_csv(output_dir / "results.csv", violation_results)  # ALL scenarios (for paper figures)

    logger.info(f"Saved violation CSVs: speed={len(speed_violations)}, drivable={len(drivable_violations)}, "
                f"direction={len(direction_violations)}, collision={len(collision_violations)}, "
                f"ttc={len(ttc_violations)}, overall={len(overall_violations)}, total={len(violation_results)}")


def generate_violation_summary_chart(
    stats_by_type: Dict[str, Dict],
    output_dir: Path,
    violation_name: str,
    chart_filename: str
) -> None:
    """
    Generate horizontal stacked bar chart for violation summary.
    :param stats_by_type: Statistics by scenario type
    :param output_dir: Output directory
    :param violation_name: Name of violation type
    :param chart_filename: Output filename
    """
    sorted_types = sorted(stats_by_type.items(), key=lambda x: x[1]['total'], reverse=False)

    scenario_types = [item[0] for item in sorted_types]
    totals = [item[1]['total'] for item in sorted_types]
    violated_counts = [item[1]['violated'] for item in sorted_types]
    clean_counts = [total - violated for total, violated in zip(totals, violated_counts)]

    with _headless_chart():
        fig, ax = plt.subplots(figsize=(12, max(6, len(scenario_types) * 0.5)))

        y_pos = np.arange(len(scenario_types))
        bar_height = 0.6

        ax.barh(y_pos, clean_counts, bar_height, label='No Violation', color='#4CAF50', alpha=0.8)
        ax.barh(y_pos, violated_counts, bar_height, left=clean_counts, label=f'{violation_name} Violated', color='#F44336', alpha=0.8)

        for i, (clean, violated, total) in enumerate(zip(clean_counts, violated_counts, totals)):
            if clean > 0:
                ax.text(clean/2, i, str(clean), ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            if violated > 0:
                ax.text(clean + violated/2, i, str(violated), ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            ax.text(total + max(totals)*0.02, i, f'Total: {total}', ha='left', va='center', fontsize=9, fontweight='bold')

        ax.set_ylabel('Scenario Type', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Scenarios', fontsize=12, fontweight='bold')
        ax.set_title(f'{violation_name} Violation Summary by Scenario Type', fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(scenario_types)
        ax.legend(loc='lower right', frameon=True, fancybox=True)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        total_scenarios = sum(totals)
        total_violations = sum(violated_counts)
        violation_rate = (total_violations / total_scenarios * 100) if total_scenarios > 0 else 0

        summary_text = f'Total Scenarios: {total_scenarios} | Violations: {total_violations} ({violation_rate:.1f}%)'
        fig.text(0.5, 0.95, summary_text, ha='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        output_path = output_dir / chart_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved {violation_name} summary chart to: {output_path}")


def generate_overall_violation_summary_chart(
    stats_by_type: Dict[str, Dict],
    output_dir: Path
) -> None:
    """
    Generate overall violation summary chart (any violation vs all clean).
    :param stats_by_type: Statistics by scenario type
    :param output_dir: Output directory
    """
    sorted_types = sorted(stats_by_type.items(), key=lambda x: x[1]['total'], reverse=False)

    scenario_types = [item[0] for item in sorted_types]
    totals = [item[1]['total'] for item in sorted_types]
    any_violated_counts = [item[1]['any_violated'] for item in sorted_types]
    clean_counts = [total - any_violated for total, any_violated in zip(totals, any_violated_counts)]

    with _headless_chart():
        fig, ax = plt.subplots(figsize=(12, max(6, len(scenario_types) * 0.5)))

        y_pos = np.arange(len(scenario_types))
        bar_height = 0.6

        ax.barh(y_pos, clean_counts, bar_height, label='All Clean', color='#4CAF50', alpha=0.8)
        ax.barh(y_pos, any_violated_counts, bar_height, left=clean_counts, label='Any Violation', color='#F44336', alpha=0.8)

        for i, (clean, any_violated, total) in enumerate(zip(clean_counts, any_violated_counts, totals)):
            if clean > 0:
                ax.text(clean/2, i, str(clean), ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            if any_violated > 0:
                ax.text(clean + any_violated/2, i, str(any_violated), ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            ax.text(total + max(totals)*0.02, i, f'Total: {total}', ha='left', va='center', fontsize=9, fontweight='bold')

        ax.set_ylabel('Scenario Type', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Scenarios', fontsize=12, fontweight='bold')
        ax.set_title('Overall Violation Summary by Scenario Type (Any Violation vs All Clean)', fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(scenario_types)
        ax.legend(loc='lower right', frameon=True, fancybox=True)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        total_scenarios = sum(totals)
        total_any_violations = sum(any_violated_counts)
        violation_rate = (total_any_violations / total_scenarios * 100) if total_scenarios > 0 else 0

        summary_text = f'Total Scenarios: {total_scenarios} | Any Violation: {total_any_violations} ({violation_rate:.1f}%)'
        fig.text(0.5, 0.95, summary_text, ha='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        output_path = output_dir / "overall_violation_summary.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved overall violation summary chart to: {output_path}")


def generate_all_summary_charts(violation_results: List[Dict], output_dir: Path) -> None:
    """
    Generate all summary charts from violation results.
    :param violation_results: List of violation result dictionaries
    :param output_dir: Output directory
    """
    if not violation_results:
        logger.warning("No violation results to generate summary charts")
        return

    # Collect statistics
    speed_stats = defaultdict(lambda: {'total': 0, 'violated': 0})
    drivable_stats = defaultdict(lambda: {'total': 0, 'violated': 0})
    direction_stats = defaultdict(lambda: {'total': 0, 'violated': 0})
    collision_stats = defaultdict(lambda: {'total': 0, 'violated': 0})
    ttc_stats = defaultdict(lambda: {'total': 0, 'violated': 0})
    overall_stats = defaultdict(lambda: {'total': 0, 'any_violated': 0})

    for result in violation_results:
        scenario_type = result['scenario_type']

        speed_stats[scenario_type]['total'] += 1
        drivable_stats[scenario_type]['total'] += 1
        direction_stats[scenario_type]['total'] += 1
        collision_stats[scenario_type]['total'] += 1
        ttc_stats[scenario_type]['total'] += 1
        overall_stats[scenario_type]['total'] += 1

        if result['speed_limit_violated']:
            speed_stats[scenario_type]['violated'] += 1

        if result['drivable_area_violated']:
            drivable_stats[scenario_type]['violated'] += 1

        if result['driving_direction_violated']:
            direction_stats[scenario_type]['violated'] += 1
            
        if result['ego_at_fault_collision_violated']:
            collision_stats[scenario_type]['violated'] += 1

        if result['time_to_collision_violated']:
            ttc_stats[scenario_type]['violated'] += 1

        any_violation = (
            result['speed_limit_violated'] or
            result['drivable_area_violated'] or
            result['driving_direction_violated'] or
            result['ego_at_fault_collision_violated'] or
            result['time_to_collision_violated']
        )
        if any_violation:
            overall_stats[scenario_type]['any_violated'] += 1

    # Generate charts
    generate_violation_summary_chart(speed_stats, output_dir, 'Speed Limit', 'speed_limit_violation_summary.png')
    generate_violation_summary_chart(drivable_stats, output_dir, 'Drivable Area', 'drivable_area_violation_summary.png')
    generate_violation_summary_chart(direction_stats, output_dir, 'Driving Direction', 'driving_direction_violation_summary.png')
    generate_violation_summary_chart(collision_stats, output_dir, 'Ego At Fault Collision', 'ego_at_fault_collision_violation_summary.png')
    generate_violation_summary_chart(ttc_stats, output_dir, 'Time To Collision', 'time_to_collision_violation_summary.png')
    generate_overall_violation_summary_chart(overall_stats, output_dir)

