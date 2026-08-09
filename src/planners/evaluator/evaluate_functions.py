#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
평가 core — nuPlan 공식 metric 을 직접 호출해 개별 지표를 계산한다.

두 계층이 있다:

* ``check_*`` — **위반 여부(bool) + per-step 신호 + 최초 위반 시각**.
  nuPlan 공식 metric 클래스를 그대로 호출한다. 사후 분석·오탐 시각화용.

* ``score_*`` — **[0,1] 스칼라**. 같은 공식 수식을 점수로 환산한다.
  게이트가 두 후보 궤적(ML planner 출력 / MPC refined)을 비교하려면
  bool 이 아니라 스칼라가 필요하기 때문이다.

공식과의 대응 관계 (각 함수 docstring 에 표기):

* ``[EXACT]``  공식 metric 클래스를 그대로 호출하거나 공식 수식을 그대로 옮긴 것
* ``[WINDOW]`` 공식은 에피소드 전체(~15 s)를 보지만 게이트는 계획 궤적(8 s)만
  볼 수 있어, 같은 수식을 8 s 창에 적용한 것
* ``[OPEN]``   Open-loop 지표 — 계획 궤적을 **로그 ego(human)** 와 비교한다.
  Closed-loop 런에서는 시뮬 ego 가 이미 로그에서 벗어나 있어 그 이탈량(drift)이
  오차에 그대로 더해진다. 절대값을 공식 Open-loop 점수와 동일시할 수 있는 것은
  ego 가 로그를 재생하는 ``open_loop_boxes`` 런뿐이다.

창(window) 차이는 원리적 한계다. 절대값을 공식 score 와 동일시하면 안 되고,
두 후보에 **동일한 창·동일한 world** 를 적용해 비교하는 용도로만 쓴다.

집계(∏ multiplicative × Σwm/Σw)와 게이트 판정은 이 파일이 아니라
``nuplan_evaluator.py`` 가 담당한다.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import DrivableAreaComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_compliance import DrivingDirectionComplianceStatistics
from nuplan.planning.metrics.utils.route_extractor import extract_corners_route, get_route
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center, extract_ego_corners
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 개별 metric 파라미터 — 각 metric 의 공식 statistics yaml 기본값
# -----------------------------------------------------------------------------

# 개별 metric 파라미터 (각 metric 의 statistics yaml 기본값)
PROGRESS_SCORE_THRESHOLD = 2.0        # ego_progress_along_expert_route.score_progress_threshold
MIN_PROGRESS_THRESHOLD = 0.2          # ego_is_making_progress.min_progress_threshold
MAX_OVERSPEED_VALUE_THRESHOLD = 2.23  # speed_limit_compliance.max_overspeed_value_threshold
TTC_LEAST_MIN_TTC = 0.95              # time_to_collision_within_bound.least_min_ttc
TTC_TIME_HORIZON = 3.0
TTC_TIME_STEP_SIZE = 0.1
DRIVABLE_AREA_MAX_VIOLATION = 0.3     # drivable_area_compliance.max_violation_threshold
# driving_direction_compliance_statistics.yaml
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0  # [m] 이 이하 역주행은 위반으로 보지 않음
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0   # [m] 이 이상 역주행은 용인하지 않음 (score 0)
DRIVING_DIRECTION_TIME_HORIZON = 1.0          # [s] 역주행 진행량을 재는 창

# --- Open-loop (planner_expert_*) -------------------------------------------
# 값을 바꾸면 공식 점수와 비교할 수 없게 된다. 노브가 아니라 규약이다.
OPEN_LOOP_HORIZONS = (3, 5, 8)                # [s] comparison_horizon
OPEN_LOOP_COMPARISON_HZ = 1                   # comparison_frequency — 공식 표본은 1 Hz 다
OPEN_LOOP_MAX_L2_ERROR = 8.0                  # [m]   max_average/final_l2_error_threshold
OPEN_LOOP_MAX_HEADING_ERROR = 0.8             # [rad] max_average/final_heading_error_threshold
OPEN_LOOP_MISS_THRESHOLDS = (6.0, 8.0, 16.0)  # [m]   max_displacement_threshold, 지평과 인덱스 정렬
OPEN_LOOP_MAX_MISS_RATE = 0.3                 # max_miss_rate_threshold

# -----------------------------------------------------------------------------
# check_* — 위반 여부 (bool) + per-step 신호
# -----------------------------------------------------------------------------

def check_speed_limit_violation(
    scenario: AbstractScenario,
    ego_states: List
) -> Tuple[bool, Optional[float], List[bool]]:
    """
    Check speed limit violation.
    :param scenario: AbstractScenario instance
    :param ego_states: List of EgoState objects
    :return: (is_violated, first_violation_time, violation_per_step)
    """
    map_api = scenario.map_api
    speeds = np.array([state.dynamic_car_state.speed for state in ego_states])

    # Calculate speed limits
    speed_limits = []
    for ego_state in ego_states:
        speed_limit = None
        try:
            point = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            all_lanes = map_api.get_all_map_objects(point, SemanticMapLayer.LANE)
            all_lane_connectors = map_api.get_all_map_objects(point, SemanticMapLayer.LANE_CONNECTOR)

            local_speed_limits = []
            for lane in all_lanes:
                if hasattr(lane, 'speed_limit_mps') and lane.speed_limit_mps is not None:
                    local_speed_limits.append(lane.speed_limit_mps)

            for lane_connector in all_lane_connectors:
                if hasattr(lane_connector, 'speed_limit_mps') and lane_connector.speed_limit_mps is not None:
                    local_speed_limits.append(lane_connector.speed_limit_mps)

            if local_speed_limits:
                speed_limit = max(local_speed_limits)

            speed_limits.append(speed_limit if speed_limit is not None else np.nan)
        except Exception:
            speed_limits.append(np.nan)

    speed_limits = np.array(speed_limits)

    # Check violations
    is_violated = False
    first_violation_time = None
    violation_per_step = []
    time_steps = np.arange(len(speeds)) * scenario.database_interval

    for i in range(len(speeds)):
        is_violating = False
        if not np.isnan(speed_limits[i]) and speeds[i] > speed_limits[i]:
            is_violating = True
            if not is_violated:
                is_violated = True
                first_violation_time = time_steps[i]
        violation_per_step.append(is_violating)

    return is_violated, first_violation_time, violation_per_step


def check_drivable_area_violation(
    scenario: AbstractScenario,
    ego_states: List,
    max_violation_threshold: float = 0.3
) -> Tuple[bool, Optional[float], List[bool]]:
    """
    Check drivable area violation using NuPlan's DrivableAreaComplianceStatistics logic directly.
    :param scenario: AbstractScenario instance
    :param ego_states: List of EgoState objects
    :param max_violation_threshold: Distance threshold in meters (NuPlan default: 0.3m)
    :return: (is_violated, first_violation_time, violation_per_step)
    """
    map_api = scenario.map_api

    # Use NuPlan's helper functions to extract data
    all_ego_corners = extract_ego_corners(ego_states)  # FL, RL, RR, FR
    ego_poses = extract_ego_center(ego_states)
    ego_footprint_list = [ego_state.car_footprint for ego_state in ego_states]

    # Get route objects using NuPlan's functions
    center_route = get_route(map_api, ego_poses)
    corners_route = extract_corners_route(map_api, ego_footprint_list)

    # Create instance to use static/helper methods
    drivable_metric = DrivableAreaComplianceStatistics(
        name="drivable_area_compliance",
        category="dynamics",
        lane_change_metric=None,
        max_violation_threshold=max_violation_threshold
    )

    violation_per_step = []
    far_from_drivable_area = False
    first_violation_time = None

    for step_idx, (ego_corners, corners_lane_lane_connector, center_lane_lane_connector) in enumerate(
        zip(all_ego_corners, corners_route, center_route)
    ):
        # Use NuPlan's compute_violation_for_iteration method
        not_in_drivable_area, far_from_drivable_area = drivable_metric.compute_violation_for_iteration(
            map_api, ego_corners, corners_lane_lane_connector, center_lane_lane_connector, far_from_drivable_area
        )

        # Record first violation time when far_from_drivable_area first becomes True
        if far_from_drivable_area and first_violation_time is None:
            first_violation_time = step_idx * scenario.database_interval

        # For per-step visualization, use not_in_drivable_area
        violation_per_step.append(not_in_drivable_area)

    # Overall violation is based on far_from_drivable_area (NuPlan's official metric)
    return far_from_drivable_area, first_violation_time, violation_per_step


def check_driving_direction_violation(
    scenario: AbstractScenario,
    ego_states: List,
    compliance_threshold: float = 2.0,
    time_horizon: float = 1.0
) -> Tuple[bool, Optional[float], List[bool]]:
    """
    Check driving direction violation using NuPlan's DrivingDirectionComplianceStatistics logic directly.
    :param scenario: AbstractScenario instance
    :param ego_states: List of EgoState objects
    :param compliance_threshold: Threshold for reverse driving (meters, negative progress)
    :param time_horizon: Time window to check progress (seconds)
    :return: (is_violated, first_violation_time, violation_per_step)
    """
    map_api = scenario.map_api

    # Use NuPlan's helper functions to extract data
    ego_poses = extract_ego_center(ego_states)
    ego_driven_route = get_route(map_api, ego_poses)

    # Calculate n_horizon based on database interval
    n_horizon = max(1, int(time_horizon / scenario.database_interval))

    # Use NuPlan's static method to extract progress
    progress_over_interval = DrivingDirectionComplianceStatistics._extract_metric(
        ego_poses, ego_driven_route, n_horizon
    )

    # Check for violations
    is_violated = False
    first_violation_time = None
    violation_per_step = []

    for step_idx, progress in enumerate(progress_over_interval):
        # Negative progress beyond compliance threshold means violation
        step_violated = progress < -compliance_threshold

        if step_violated and not is_violated:
            is_violated = True
            first_violation_time = step_idx * scenario.database_interval

        violation_per_step.append(step_violated)

    return is_violated, first_violation_time, violation_per_step


def check_ego_at_fault_collision(
    scenario: AbstractScenario,
    ego_states: List,
    stopped_speed_threshold: float = 5e-02,
    observations: Optional[List] = None,
) -> Tuple[bool, Optional[float], List[bool], List]:
    """
    Check if ego has at-fault collisions using NuPlan's exact logic.
    This uses the same logic as EgoAtFaultCollisionStatistics.
    :param scenario: AbstractScenario instance
    :param ego_states: List of EgoState objects
    :param stopped_speed_threshold: Threshold for considering stopped (m/s)
    :param observations: Optional per-step DetectionsTracks aligned with ego_states.
        기본값(None)이면 기존 동작대로 scenario 의 iteration=step_idx 에서 로그를 읽는다.
        후보 궤적(ML / MPC refined)을 채점할 때는 build_future_observations() 로 만든
        GT-future world 를 넘긴다. 이때 ego_states 와 길이가 같아야 한다.
    :return: (has_collision, first_collision_time, collision_per_step, all_collisions)
    """
    from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
        find_new_collisions,
        classify_at_fault_collisions,
        Collisions
    )
    from nuplan.planning.metrics.utils.route_extractor import (
        extract_corners_route,
        get_common_or_connected_route_objs_of_corners,
        get_timestamps_in_common_or_connected_route_objs
    )
    from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point

    # Collect all collisions using NuPlan's find_new_collisions
    all_collisions: List[Collisions] = []
    collided_track_ids = set()

    for step_idx, ego_state in enumerate(ego_states):
        try:
            if observations is not None:
                observation = observations[step_idx]
            else:
                observation = scenario.get_tracked_objects_at_iteration(step_idx)
            timestamp = ego_state.time_point.time_us

            # Use NuPlan's exact collision detection
            collided_track_ids, collisions_id_data = find_new_collisions(
                ego_state, observation, collided_track_ids
            )

            if len(collisions_id_data):
                all_collisions.append(Collisions(timestamp, collisions_id_data))

        except Exception as e:
            logger.debug(f"Error checking collision at step {step_idx}: {e}")

    # Determine timestamps where ego is in common or connected route objects
    # This is needed for at-fault classification (using NuPlan's exact method)
    ego_timestamps = extract_ego_time_point(ego_states)
    ego_footprint_list = [ego_state.car_footprint for ego_state in ego_states]
    corners_route = extract_corners_route(scenario.map_api, ego_footprint_list)
    common_or_connected_route_objs = get_common_or_connected_route_objs_of_corners(corners_route)
    timestamps_in_common_or_connected_route_objs = get_timestamps_in_common_or_connected_route_objs(
        common_or_connected_route_objs, ego_timestamps
    )

    # Use NuPlan's exact at-fault classification
    timestamps_at_fault_collisions, at_fault_collisions = classify_at_fault_collisions(
        all_collisions, timestamps_in_common_or_connected_route_objs
    )

    # Convert timestamps to per-step boolean list
    collision_per_step = []
    has_collision = len(timestamps_at_fault_collisions) > 0
    first_collision_time = None

    for step_idx, ego_state in enumerate(ego_states):
        timestamp = ego_state.time_point.time_us
        step_has_collision = timestamp in timestamps_at_fault_collisions
        collision_per_step.append(step_has_collision)

        if step_has_collision and first_collision_time is None:
            first_collision_time = step_idx * scenario.database_interval

    # Return collision data for visualization
    return has_collision, first_collision_time, collision_per_step, all_collisions


def _compute_per_object_ttc_at_step(
    ego_state,
    observation,
    timestamp: int,
    timestamps_at_fault_collisions: List[int],
    time_step_size: float,
    time_horizon: float,
    stopped_speed_threshold: float = 5e-03
) -> Dict[str, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Compute TTC for each object at a single timestep using NuPlan's logic.
    Returns dict of {track_token: (ttc, projected_ego_pose, projected_track_pose)}
    """
    from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import (
        _get_ego_tracks_displacement_info,
        _get_relevant_tracks
    )
    from nuplan.common.actor_state.oriented_box import OrientedBox, in_collision
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.agent import Agent

    ego_speed = ego_state.dynamic_car_state.speed

    # Check if ego is in at-fault collision
    if timestamp in timestamps_at_fault_collisions:
        return {}

    # Skip if ego is stopped or no objects
    if ego_speed <= stopped_speed_threshold or len(observation.tracked_objects) == 0:
        return {}

    # Extract track info
    tracks_poses = np.array([np.array([*tracked_object.center], dtype=np.float64)
                             for tracked_object in observation.tracked_objects])
    tracks_speed = np.array([np.array(tracked_object.velocity.magnitude(), dtype=np.float64)
                             if isinstance(tracked_object, Agent) else 0
                             for tracked_object in observation.tracked_objects])
    tracks_boxes = [tracked_object.box for tracked_object in observation.tracked_objects]
    track_tokens = [tracked_object.track_token for tracked_object in observation.tracked_objects]

    if len(tracks_poses) == 0:
        return {}

    # Get displacement info
    displacement_info = _get_ego_tracks_displacement_info(
        ego_state, ego_speed, tracks_poses, tracks_speed, time_step_size
    )

    # Get relevant tracks
    relevant_tracks_mask = _get_relevant_tracks(
        displacement_info.ego_pose,
        displacement_info.ego_box,
        displacement_info.ego_dx,
        displacement_info.ego_dy,
        tracks_poses,
        tracks_boxes,
        displacement_info.tracks_dxy,
        time_step_size,
        time_horizon,
    )

    if len(relevant_tracks_mask) == 0:
        return {}

    # Compute TTC for each relevant track
    per_object_ttc = {}

    # Make copies for projection
    ego_pose_copy = displacement_info.ego_pose.copy()
    tracks_poses_copy = tracks_poses.copy()

    for time_to_collision in np.arange(time_step_size, time_horizon, time_step_size):
        # Project ego
        ego_pose_copy[:2] += (displacement_info.ego_dx, displacement_info.ego_dy)
        projected_ego_box = OrientedBox.from_new_pose(
            displacement_info.ego_box, StateSE2(*ego_pose_copy)
        )

        # Project tracks
        tracks_poses_copy[:, :2] += displacement_info.tracks_dxy

        for idx in relevant_tracks_mask:
            track_token = track_tokens[idx]
            if track_token in per_object_ttc:
                continue

            track_box = tracks_boxes[idx]
            track_pose = tracks_poses_copy[idx]
            projected_track_box = OrientedBox.from_new_pose(track_box, StateSE2(*track_pose))

            if in_collision(projected_ego_box, projected_track_box):
                per_object_ttc[track_token] = (
                    float(time_to_collision),
                    ego_pose_copy.copy(),
                    track_pose.copy()
                )

    return per_object_ttc


def check_time_to_collision(
    scenario: AbstractScenario,
    ego_states: List,
    ttc_threshold: float = 0.95,
    time_step_size: float = 0.1,
    time_horizon: float = 3.0,
    stopped_speed_threshold: float = 5e-03,
    observations_override: Optional[List] = None,
) -> Tuple[bool, Optional[float], List[bool], np.ndarray, List[Dict]]:
    """
    Check if time-to-collision (TTC) goes below threshold using NuPlan's exact logic.
    This uses the same logic as TimeToCollisionStatistics.
    Uses NuPlan's default values: ttc_threshold=0.95s, time_step_size=0.1s, time_horizon=3.0s
    :param scenario: AbstractScenario instance
    :param ego_states: List of EgoState objects
    :param ttc_threshold: TTC threshold in seconds (default: 0.95s, from NuPlan config)
    :param time_step_size: Step size for projection (default: 0.1s, from NuPlan config)
    :param time_horizon: Time horizon for TTC calculation (default: 3.0s, from NuPlan config)
    :param stopped_speed_threshold: Threshold for considering stopped (default: 5e-03 m/s, from NuPlan)
    :return: (has_violation, first_violation_time, violation_per_step, time_to_collision_array)
    """
    from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import (
        compute_time_to_collision
    )
    from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
        find_new_collisions,
        classify_at_fault_collisions,
        Collisions
    )
    from nuplan.planning.metrics.utils.route_extractor import (
        extract_corners_route,
        get_common_or_connected_route_objs_of_corners,
        get_timestamps_in_common_or_connected_route_objs
    )
    from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point

    # First, compute all collisions (needed for TTC calculation)
    all_collisions: List[Collisions] = []
    collided_track_ids = set()
    observations = []

    for step_idx, ego_state in enumerate(ego_states):
        try:
            if observations_override is not None:
                observation = observations_override[step_idx]
            else:
                observation = scenario.get_tracked_objects_at_iteration(step_idx)
            observations.append(observation)
            timestamp = ego_state.time_point.time_us

            collided_track_ids, collisions_id_data = find_new_collisions(
                ego_state, observation, collided_track_ids
            )

            if len(collisions_id_data):
                all_collisions.append(Collisions(timestamp, collisions_id_data))
        except Exception as e:
            logger.debug(f"Error collecting collision data at step {step_idx}: {e}")

    # Extract ego timestamps (needed for NuPlan's route extraction)
    ego_timestamps = extract_ego_time_point(ego_states)

    # Determine timestamps where ego is in common or connected route objects (using NuPlan's exact method)
    ego_footprint_list = [ego_state.car_footprint for ego_state in ego_states]
    corners_route = extract_corners_route(scenario.map_api, ego_footprint_list)
    common_or_connected_route_objs = get_common_or_connected_route_objs_of_corners(corners_route)
    timestamps_in_common_or_connected_route_objs = get_timestamps_in_common_or_connected_route_objs(
        common_or_connected_route_objs, ego_timestamps
    )

    # Classify at-fault collisions
    timestamps_at_fault_collisions, _ = classify_at_fault_collisions(
        all_collisions, timestamps_in_common_or_connected_route_objs
    )

    # Use NuPlan's exact TTC computation
    time_to_collision = compute_time_to_collision(
        ego_states,
        ego_timestamps,
        observations,
        timestamps_in_common_or_connected_route_objs,
        all_collisions,
        timestamps_at_fault_collisions,
        scenario.map_api,
        time_step_size,
        time_horizon,
        stopped_speed_threshold
    )

    # Compute per-object TTC for visualization
    per_object_ttc_list = []
    for step_idx, ego_state in enumerate(ego_states):
        try:
            timestamp = ego_state.time_point.time_us
            observation = observations[step_idx]
            per_obj_ttc = _compute_per_object_ttc_at_step(
                ego_state,
                observation,
                timestamp,
                timestamps_at_fault_collisions,
                time_step_size,
                time_horizon,
                stopped_speed_threshold
            )
            per_object_ttc_list.append(per_obj_ttc)
        except Exception as e:
            logger.debug(f"Error computing per-object TTC at step {step_idx}: {e}")
            per_object_ttc_list.append({})

    # Check for violations (TTC < threshold)
    violation_per_step = []
    has_violation = False
    first_violation_time = None

    for step_idx, ttc_value in enumerate(time_to_collision):
        step_violated = ttc_value < ttc_threshold
        violation_per_step.append(step_violated)

        if step_violated and not has_violation:
            has_violation = True
            first_violation_time = step_idx * scenario.database_interval

    return has_violation, first_violation_time, violation_per_step, time_to_collision, per_object_ttc_list

# -----------------------------------------------------------------------------
# 개별 metric → 스칼라
# -----------------------------------------------------------------------------

def score_drivable_area_compliance(
    scenario: AbstractScenario, ego_states: List
) -> Tuple[float, Dict]:
    """[EXACT] 공식 DrivableAreaComplianceStatistics. 위반이면 0, 아니면 1."""
    violated, first_time, per_step = check_drivable_area_violation(
        scenario, ego_states, max_violation_threshold=DRIVABLE_AREA_MAX_VIOLATION
    )
    return (0.0 if violated else 1.0), {
        "drivable_area_first_time": first_time,
        "drivable_area_per_step": per_step,
    }


def score_driving_direction_compliance(
    scenario: AbstractScenario,
    ego_states: List,
    compliance_threshold: float = DRIVING_DIRECTION_COMPLIANCE_THRESHOLD,
    violation_threshold: float = DRIVING_DIRECTION_VIOLATION_THRESHOLD,
    time_horizon: float = DRIVING_DIRECTION_TIME_HORIZON,
) -> Tuple[float, Dict]:
    """
    [EXACT] 공식 DrivingDirectionComplianceStatistics 의 3단 점수.

    1 s 창의 역방향 진행량이
      - violation_threshold 초과      → 0.0
      - compliance_threshold 초과     → 0.5
      - 그 외                          → 1.0
    """
    from nuplan.planning.metrics.utils.state_extractors import extract_ego_center

    map_api = scenario.map_api
    ego_poses = extract_ego_center(ego_states)
    ego_driven_route = get_route(map_api, ego_poses)
    n_horizon = max(1, int(time_horizon / scenario.database_interval))

    progress_over_interval = DrivingDirectionComplianceStatistics._extract_metric(
        ego_poses, ego_driven_route, n_horizon
    )
    if len(progress_over_interval) == 0:
        return 1.0, {"driving_direction_min_progress": None}

    min_progress = float(np.min(progress_over_interval))
    if min_progress < -violation_threshold:
        score = 0.0
    elif min_progress < -compliance_threshold:
        score = 0.5
    else:
        score = 1.0
    return score, {"driving_direction_min_progress": min_progress}


def score_no_ego_at_fault_collisions(
    scenario: AbstractScenario,
    ego_states: List,
    observations: Optional[List] = None,
) -> Tuple[float, Dict]:
    """
    [EXACT] 공식 no_ego_at_fault_collisions.

    at-fault 충돌이 하나라도 있으면 0, 없으면 1.
    (공식은 VRU/차량 구분으로 0.5 를 주기도 하나, 게이트 용도에서는
     '충돌했는가'만 필요하므로 0/1 로 둔다 — 보수적인 쪽이다.)
    """
    has_collision, first_time, per_step, all_collisions = check_ego_at_fault_collision(
        scenario, ego_states, observations=observations
    )
    # 부딪힌 상대 토큰 — 렌더러가 해당 객체를 빨갛게 칠하는 데 쓴다.
    # Collisions.collisions_id_data 는 {track_token: CollisionData} 다.
    hit_tokens = []
    for c in (all_collisions or []):
        hit_tokens.extend(getattr(c, "collisions_id_data", {}) or {})

    # **물리 충돌이 일어난 스텝 인덱스**. 점수(no_ego_at_fault_collisions)는
    # at-fault 만 벌하므로, 물리적으로 부딪혔는데 만점이 나오는 경우가 있다
    # (정차 중 추돌, 상대가 ego 차선으로 침범 등). 그런 스텝을 수집에서
    # 걸러내려면 at-fault 여부와 무관한 이 신호가 필요하다.
    ts_to_step = {st.time_point.time_us: i for i, st in enumerate(ego_states)}
    collision_steps = sorted(
        {ts_to_step[c.timestamp] for c in (all_collisions or [])
         if c.timestamp in ts_to_step}
    )

    return (0.0 if has_collision else 1.0), {
        "collision_first_time": first_time,
        "collision_per_step": per_step,
        "collision_tokens": hit_tokens,
        "collision_steps": collision_steps,
    }


def score_time_to_collision(
    scenario: AbstractScenario,
    ego_states: List,
    observations: Optional[List] = None,
) -> Tuple[float, Dict]:
    """
    [EXACT] 공식 time_to_collision_within_bound.

    horizon 3.0 s, step 0.1 s, least_min_ttc 0.95 — 전부 공식 yaml 값이다.
    (현행 TrajectoryEvaluator 는 horizon 이 0.8 s 라 여기서 크게 달라진다.)
    """
    violated, first_time, per_step, ttc_array, per_object = check_time_to_collision(
        scenario,
        ego_states,
        ttc_threshold=TTC_LEAST_MIN_TTC,
        time_step_size=TTC_TIME_STEP_SIZE,
        time_horizon=TTC_TIME_HORIZON,
        observations_override=observations,
    )
    # 위반을 만든 **첫 스텝**에서 누가 범인이고 어디서 부딪히는지 뽑아둔다.
    # 렌더러가 충돌 객체를 빨간 상자로, 투영 충돌 지점을 빨간 x 로 그리는 데 쓴다.
    # per_object[step] = {track_token: (ttc, ego_pose_global, track_pose_global)}
    violation = None
    if violated:
        for step_idx, bad in enumerate(per_step):
            if not bad or step_idx >= len(per_object):
                continue
            hits = [
                (tok, float(v[0]), np.asarray(v[1], dtype=float), np.asarray(v[2], dtype=float))
                for tok, v in per_object[step_idx].items()
                if float(v[0]) <= TTC_LEAST_MIN_TTC
            ]
            if not hits:
                continue
            tok, ttc_v, ego_pose, track_pose = min(hits, key=lambda h: h[1])  # 가장 임박한 것
            violation = {
                "step": int(step_idx),
                "track_token": tok,
                "ttc": ttc_v,
                # global [x, y, heading] — 렌더러가 ego-local 로 변환해 쓴다
                "ego_pose": ego_pose.tolist(),
                "track_pose": track_pose.tolist(),
            }
            break

        if violation is None:
            # per-object 가 비는 경우가 있다: at-fault 충돌 시점이면
            # _compute_time_to_collision_at_timestamp 가 투영 없이 곧장 0 을 주고
            # _compute_per_object_ttc_at_step 은 {} 를 돌려주기 때문이다.
            # 그때는 투영 지점이 아니라 **ego 의 실제 위치**가 곧 충돌 지점이다.
            bad_steps = [i for i, b in enumerate(per_step) if b]
            if bad_steps:
                i = bad_steps[0]
                if i < len(ego_states):
                    c = ego_states[i].rear_axle
                    violation = {
                        "step": int(i),
                        "track_token": None,
                        "ttc": float(ttc_array[i]) if i < len(ttc_array) else 0.0,
                        "ego_pose": [float(c.x), float(c.y), float(c.heading)],
                        "track_pose": None,
                    }

    return (0.0 if violated else 1.0), {
        "ttc_first_time": first_time,
        "ttc_per_step": per_step,
        "ttc_min": float(np.min(ttc_array)) if len(ttc_array) else None,
        "ttc_per_object": per_object,
        "ttc_violation": violation,
    }


def score_speed_limit_compliance(
    scenario: AbstractScenario, ego_states: List
) -> Tuple[float, Dict]:
    """
    [EXACT] 공식 SpeedLimitComplianceStatistics._compute_violation_metric_score.

        violation_loss = Σ(overspeed) · dt / (max_overspeed_threshold · duration)
        score          = max(0, 1 − violation_loss)

    현행 TrajectoryEvaluator 와 달리 **연속값**이다. 위반 여부(bool)가 아니라
    얼마나 오래·얼마나 크게 넘었는지가 점수에 들어간다.
    """
    from nuplan.common.maps.abstract_map_objects import Lane

    map_api = scenario.map_api

    # 공식과 동일하게 **주행 경로(get_route)** 기준으로 제한속도를 얻는다.
    # 이전 구현은 ego 위치의 모든 lane/lane_connector 중 max(speed_limit) 를 썼는데,
    # 그러면 옆 차선·교차 차선의 제한속도가 섞여 들어가 점수가 어긋난다
    # (val_demo 0adcd0e658d359a7 에서 공식 0.936 vs 우리 0.892 로 불일치).
    ego_poses = extract_ego_center(ego_states)
    ego_route = get_route(map_api, ego_poses)

    overspeed: List[float] = []
    for ego_state, curr_route in zip(ego_states, ego_route):
        # 주행가능영역 밖이면 공식과 동일하게 위반으로 세지 않는다
        if not curr_route:
            overspeed.append(0.0)
            continue
        try:
            # 공식 SpeedLimitComplianceStatistics._get_speed_limit_violation 과 동일:
            #  · Lane 이면 그 lane 의 제한속도 하나만
            #  · LaneConnector 면 in/out 으로 연결된 lane 들의 제한속도
            if isinstance(curr_route[0], Lane):
                limits = [curr_route[0].speed_limit_mps]
            else:
                limits = []
                for obj in curr_route:
                    for lane in obj.outgoing_edges + obj.incoming_edges:
                        limits.append(lane.speed_limit_mps)
        except Exception:
            limits = []

        # 공식은 `all(speed_limits)` — 하나라도 None/0 이면 판정하지 않는다
        if limits and all(limits):
            over = float(ego_state.dynamic_car_state.speed) - max(limits)
            overspeed.append(max(0.0, over))
        else:
            overspeed.append(0.0)

    dt = float(scenario.database_interval)
    duration = dt * max(1, len(ego_states) - 1)
    threshold = max(MAX_OVERSPEED_VALUE_THRESHOLD, 1e-3)
    violation_loss = float(np.sum(overspeed)) * dt / (threshold * duration)
    score = float(max(0.0, 1.0 - violation_loss))

    return score, {
        "speed_limit_max_overspeed": float(np.max(overspeed)) if overspeed else 0.0,
        "speed_limit_overspeed_per_step": overspeed,
    }


# --- comfort bound (nuPlan statistics yaml 기본값) ---------------------------
# nuplan/planning/script/config/common/simulation_metric/low_level/*.yaml
COMFORT_MIN_LON_ACCEL = -4.05    # ego_lon_acceleration (비대칭!)
COMFORT_MAX_LON_ACCEL = 2.40
COMFORT_MAX_ABS_LAT_ACCEL = 4.89
COMFORT_MAX_ABS_MAG_JERK = 8.37
COMFORT_MAX_ABS_LON_JERK = 4.13
COMFORT_MAX_ABS_YAW_ACCEL = 1.93
COMFORT_MAX_ABS_YAW_RATE = 0.95


def score_ego_is_comfortable(ego_states: List, dt: float = 0.1) -> Tuple[float, Dict]:
    """
    [EXACT] 공식 ego_is_comfortable — 6개 물리량이 전부 bound 안이면 1, 아니면 0.

    **nuPlan 의 추출 함수(`state_extractors`)를 직접 호출한다.** 이전에는
    `src/post_processing/evaluation/comfort_metrics.py`(PDM 이식본)를 썼는데
    두 가지가 공식과 달라 위반을 놓쳤다:

    1. **savgol window_length** — 이식본의 `_compute_*` 는 `window_length=n_time`
       (시계열 전체 길이)을 넘긴다. 공식은 지표별로 accel 8 / jerk 15 /
       yaw_rate 5 다. 150 프레임 창으로 평활하면 시계열이 거의 2차 다항식이 되어
       **피크가 사라진다.** val_demo 검증에서 lon_accel 4.99 (상한 2.40)를
       "통과"로 판정하는 사고가 실제로 발생했다.
    2. **가속도 기준점** — 이식본은 `StateIndex.ACCELERATION_X`(프로젝트 규약상
       **후축** 가속도)를 쓴다. 공식은 `center_acceleration_2d`(차량 중심)다.
       회전 중에는 원심 성분 때문에 둘이 다르다.

    이식본은 4 s rollout 채점용으로 만들어진 것이라 그 스케일에서는 문제가 덜했다.
    구 파이프라인(`post_processing/trajectory_evaluator.py`)이 같은 모듈을 공유하므로
    **이식본은 건드리지 않고** 여기서 공식 함수를 직접 쓴다.

    bound (공식 yaml):
      lon accel [-4.05, 2.40] · lat accel ±4.89 · jerk(magnitude) ±8.37
      lon jerk ±4.13 · yaw accel ±1.93 · yaw rate ±0.95

    :param dt: 미사용. 공식 추출기가 ego_state 의 time_point 로 직접 dt 를 얻는다.
        (시그니처 호환을 위해 남겨둔다)
    """
    from nuplan.planning.metrics.utils.state_extractors import (
        extract_ego_acceleration,
        extract_ego_jerk,
        extract_ego_yaw_rate,
    )

    def within(values, lo, hi) -> bool:
        """공식 WithinBoundMetricBase._compute_within_bound 과 동일 (엄격 부등호)."""
        v = np.asarray(values, dtype=np.float64)
        return bool(np.all((v > lo) & (v < hi)))

    # 각 지표의 추출 파라미터는 공식 metric 클래스의 compute() 호출과 일치시킨다.
    #   ego_lon_acceleration : extract_ego_acceleration(coordinate='x')
    #   ego_lat_acceleration : extract_ego_acceleration(coordinate='y')
    #   ego_jerk             : extract_ego_jerk(coordinate='magnitude')
    #   ego_lon_jerk         : extract_ego_jerk(coordinate='x')
    #   ego_yaw_rate         : extract_ego_yaw_rate()                    deriv 1, poly 2
    #   ego_yaw_acceleration : extract_ego_yaw_rate(deriv_order=2, poly_order=3)
    lon_accel = extract_ego_acceleration(ego_states, acceleration_coordinate="x")
    lat_accel = extract_ego_acceleration(ego_states, acceleration_coordinate="y")
    mag_jerk = extract_ego_jerk(ego_states, acceleration_coordinate="magnitude")
    lon_jerk = extract_ego_jerk(ego_states, acceleration_coordinate="x")
    yaw_rate = extract_ego_yaw_rate(ego_states)
    yaw_accel = extract_ego_yaw_rate(ego_states, deriv_order=2, poly_order=3)

    checks = {
        "lon_accel": within(lon_accel, COMFORT_MIN_LON_ACCEL, COMFORT_MAX_LON_ACCEL),
        "lat_accel": within(lat_accel, -COMFORT_MAX_ABS_LAT_ACCEL, COMFORT_MAX_ABS_LAT_ACCEL),
        "jerk": within(mag_jerk, -COMFORT_MAX_ABS_MAG_JERK, COMFORT_MAX_ABS_MAG_JERK),
        "lon_jerk": within(lon_jerk, -COMFORT_MAX_ABS_LON_JERK, COMFORT_MAX_ABS_LON_JERK),
        "yaw_accel": within(yaw_accel, -COMFORT_MAX_ABS_YAW_ACCEL, COMFORT_MAX_ABS_YAW_ACCEL),
        "yaw_rate": within(yaw_rate, -COMFORT_MAX_ABS_YAW_RATE, COMFORT_MAX_ABS_YAW_RATE),
    }
    score = 1.0 if all(checks.values()) else 0.0

    return score, {
        "comfort_per_metric": checks,
        # 시각화용 원계열. ML/refined 를 나란히 그려 "어느 구간이 한계를 넘었나" 를
        # 눈으로 보려면 극값만으로는 부족하다. 길이는 len(ego_states)-1 (미분 결과).
        #
        # lat_accel 은 **일부러 빼놨다**. nuPlan 에서 이 항은 구조적으로 항상 0 이다:
        #   ① kinematic bicycle 이 rear_axle_acceleration_2d.y 를 0 으로 고정(no-slip)
        #   ② comfort 가 읽는 center_acceleration_2d.y 를 만드는 get_acceleration_shifted
        #      가 변위 (d,0) 에 외적이 아니라 **원소별 곱**을 써서 회전항이 x 로만 들어간다
        # 따라서 ego_lat_acceleration(±4.89) 검사는 closed-loop 에서 발동하지 않는다.
        # 대신 같은 comfort 6항 중 실제로 변하는 yaw_rate(±0.95)를 그린다.
        "comfort_series": {
            "lon_accel": np.asarray(lon_accel, dtype=np.float32).tolist(),
            "lon_jerk": np.asarray(lon_jerk, dtype=np.float32).tolist(),
            "yaw_rate": np.asarray(yaw_rate, dtype=np.float32).tolist(),
        },
        "comfort_bounds": {
            "lon_accel": [COMFORT_MIN_LON_ACCEL, COMFORT_MAX_LON_ACCEL],
            "lon_jerk": [-COMFORT_MAX_ABS_LON_JERK, COMFORT_MAX_ABS_LON_JERK],
            "yaw_rate": [-COMFORT_MAX_ABS_YAW_RATE, COMFORT_MAX_ABS_YAW_RATE],
        },
        # 어느 항이 얼마나 넘었는지 — 오탐 진단용
        "comfort_extremes": {
            "lon_accel": [float(np.min(lon_accel)), float(np.max(lon_accel))],
            "lat_accel": [float(np.min(lat_accel)), float(np.max(lat_accel))],
            "jerk": [float(np.min(mag_jerk)), float(np.max(mag_jerk))],
            "lon_jerk": [float(np.min(lon_jerk)), float(np.max(lon_jerk))],
            "yaw_accel": [float(np.min(yaw_accel)), float(np.max(yaw_accel))],
            "yaw_rate": [float(np.min(yaw_rate)), float(np.max(yaw_rate))],
        },
    }


def score_ego_progress(
    ego_states: List,
    expert_ego_states: List,
    baseline_path,
) -> Tuple[float, float, Dict]:
    """
    [WINDOW] 공식 ego_progress_along_expert_route 의 수식을 8 s 창에 적용.

        ratio = min(1, max(ego, 2) / max(expert, 2)),  ego < −2 이면 0

    공식은 route roadblock 을 따라 에피소드 전체 진행량을 재지만, 계획 궤적은
    아직 실행되지 않았으므로 roadblock 통과 이력이 없다. 대신 **동일한
    baseline path 위로 사영한 종방향 진행량**을 ego / expert 양쪽에 같은 방식으로
    적용한다. expert 는 같은 창의 로그 ego(=human) 궤적이다.

    :param ego_states: 후보 궤적의 EgoState 시퀀스
    :param expert_ego_states: 같은 구간의 로그 ego EgoState 시퀀스
    :param baseline_path: shapely LineString — 사영 기준선
    :return: (progress_score, making_progress_score, diagnostics)
    """
    from shapely.geometry import Point as ShapelyPoint

    def _progress_along(states) -> float:
        if baseline_path is None or len(states) < 2:
            return 0.0
        s0 = baseline_path.project(ShapelyPoint(states[0].rear_axle.x, states[0].rear_axle.y))
        s1 = baseline_path.project(ShapelyPoint(states[-1].rear_axle.x, states[-1].rear_axle.y))
        return float(s1 - s0)

    ego_progress = _progress_along(ego_states)
    expert_progress = _progress_along(expert_ego_states) if expert_ego_states else 0.0

    if ego_progress < -PROGRESS_SCORE_THRESHOLD:
        ratio = 0.0
    else:
        ratio = min(
            1.0,
            max(ego_progress, PROGRESS_SCORE_THRESHOLD)
            / max(expert_progress, PROGRESS_SCORE_THRESHOLD),
        )

    making_progress = 1.0 if ratio >= MIN_PROGRESS_THRESHOLD else 0.0

    return ratio, making_progress, {
        "ego_progress_m": ego_progress,
        "expert_progress_m": expert_progress,
    }



# -----------------------------------------------------------------------------
# Open-loop — 계획 궤적 대 로그 ego
# -----------------------------------------------------------------------------

def score_open_loop_vs_expert(
    trajectory: np.ndarray,
    expert_states: List,
    *,
    dt: float = 0.1,
    rear_to_center: float = 1.461,
    horizons: Tuple[int, ...] = OPEN_LOOP_HORIZONS,
    expert_offsets: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, float], Dict]:
    """[OPEN] 계획 궤적과 로그 ego 의 변위·헤딩 오차. 공식 planner_expert_* 와 같은 수식이다.

    ⚠ 입력은 **계획 궤적 원본** `(T,3)` 이어야 한다. forward-simulate 한 ego_states 를 넣으면
      안 된다 — 공식 지표는 planner 가 내놓은 궤적을 그대로 보기 때문이다.

    :param trajectory: (T,3) global **rear-axle** [x, y, heading], dt 간격, 초기 상태 미포함
    :param expert_states: build_context 의 expert_ego_states. 인덱스 0 이 **초기 상태**다
    :param rear_to_center: 차량의 rear axle → box center 거리
    :param expert_offsets: 1 초 표본 k(=1..max(horizons)) 가 볼 expert_states 인덱스.
        None 이면 `k*stride`. 공식 지표에 맞추려면 호출부가 지정한다("공식 표본" 주석 참조)
    :return: (프레임 지표 dict, diagnostics)
    """
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.planning.metrics.utils.expert_comparisons import (
        compute_traj_errors,
        compute_traj_heading_errors,
    )

    from .common.geometry import ego_rear_to_center

    traj = np.asarray(trajectory, dtype=float)[:, :3]
    n_plan = len(traj)

    # 계획 궤적을 center 기준 StateSE2 로. 공식은 ego center 를 비교한다.
    centers = ego_rear_to_center(traj[:, :2], traj[:, 2], rear_to_center=rear_to_center)
    plan_poses = [StateSE2(float(c[0]), float(c[1]), float(h))
                  for c, h in zip(centers, traj[:, 2])]

    # 인덱스 규약: traj 행 i 는 +(i+1)*dt 초, expert_states[j] 는 +j*dt 초.
    # 따라서 +k*dt 초를 보려면 traj[k-1] 과 expert_states[k] 를 짝지어야 한다.
    n_pair = min(n_plan, len(expert_states) - 1)
    expert_poses = [expert_states[j + 1].center for j in range(n_pair)]

    if n_pair <= 0:
        nan = float("nan")
        empty = {k: nan for k in ("ol_ade_m", "ol_fde_m", "ol_ahe_rad", "ol_fhe_rad")}
        empty.update({f"ol_ade_h{h}": nan for h in horizons})
        return empty, {"open_loop_de_per_step": [], "open_loop_he_per_step": []}

    de = compute_traj_errors(plan_poses[:n_pair], expert_poses, heading_diff_weight=0)
    he = compute_traj_heading_errors(plan_poses[:n_pair], expert_poses)

    # ── 공식 표본 — 1 초 간격 k=1..max(horizons) ────────────────────────────
    # 표본 k 는 "offsets[k] 번째 로그 표본의 시각" 에서 계획과 로그를 비교한다.
    #
    # 로그 타임스탬프는 정확히 1 초 간격이 아니라 ±2 ms 흔들린다. 공식 지표는 계획 궤적을
    # 그 실제 시각에서 보간하므로, 0.1 s 격자 값을 그대로 쓰면 프레임당 1 cm 남짓 벌어진다.
    # 같은 시각에서 보간해 그 차이를 없앤다.
    #
    # 어느 표본을 볼지는 offsets 가 정한다. 보통 offsets[k] = k*stride(= +k 초)지만,
    # devkit 의 planner_expert_* 지표는 로그 표본을 [시나리오 구간] + [그 뒤 연장 구간] 으로
    # 잇다가 경계에서 표본 하나를 중복시킨다. 그래서 시나리오 후반부 프레임의 마지막 표본들은
    # +8 초가 아니라 앞 시각을 다시 본다. 화면의 숫자가 공식 parquet 과 같아야 하므로
    # 그 표본을 그대로 재현한다.
    stride = int(round(1.0 / (dt * OPEN_LOOP_COMPARISON_HZ)))
    k_max = max(horizons)
    offsets = (list(expert_offsets)[:k_max] if expert_offsets is not None
               else [k * stride for k in range(1, k_max + 1)])

    t_us = np.array([s.time_point.time_us for s in expert_states], dtype=float)
    # 계획 궤적 노드의 시각. devkit 은 이 값을 **초 도메인에서 만들어 마이크로초로 절삭**한다
    # (transform_utils._get_fixed_timesteps → TimePoint(int(t * 1e6))). 마이크로초 격자로
    # 곧바로 계산하면 노드가 최대 1 µs 어긋나고, 그만큼(≈1.5e-5 m) 보간 결과가 밀린다.
    t0_s = expert_states[0].time_point.time_s
    plan_t = np.floor((t0_s + (np.arange(n_plan) + 1) * dt) * 1e6)
    plan_end = float(plan_t[-1])
    # heading 은 ±π 를 넘나들 수 있어 펼친 값으로 보간한다. cos/sin 과 각도차 계산 모두
    # 2π 이동에 불변이므로 뒤쪽 계산에는 영향이 없다.
    plan_heading = np.unwrap(traj[:, 2])

    def _center_pose(x: float, y: float, heading: float,
                     d: float = rear_to_center) -> "StateSE2":
        # devkit 의 AngularInterpolator 는 펼친 각을 보간한 뒤 principal_value 로 다시
        # [-π, π) 에 넣는다. 각도차 계산은 2π 이동에 불변이라 값은 같지만, 같은 대표값을
        # 쓰도록 맞춰 둔다.
        heading = float((heading + np.pi) % (2 * np.pi) - np.pi)
        c = ego_rear_to_center(np.array([[x, y]]), np.array([heading]), rear_to_center=d)[0]
        return StateSE2(float(c[0]), float(c[1]), heading)

    def _plan_pose_at(t: float) -> "StateSE2":
        """rear-axle 궤적을 시각 t 에서 보간한 뒤 center 로 옮긴다.

        공식이 InterpolatedTrajectory 로 하는 것과 같은 순서다(보간 → center).
        """
        return _center_pose(float(np.interp(t, plan_t, traj[:, 0])),
                            float(np.interp(t, plan_t, traj[:, 1])),
                            float(np.interp(t, plan_t, plan_heading)))

    def _expert_pose_at(t: float) -> "StateSE2":
        """로그를 시각 t 에서 보간한다. 공식과 같게 **1 Hz 로 솎은** 표본 위에서 한다."""
        g = np.arange(0, len(expert_states), stride)
        gt = t_us[g]
        gx = np.array([expert_states[i].rear_axle.x for i in g])
        gy = np.array([expert_states[i].rear_axle.y for i in g])
        gh = np.unwrap(np.array([expert_states[i].rear_axle.heading for i in g]))
        # 로그 쪽 center 변환은 로그 상태 자신의 제원을 쓴다. expert_states[0] 은 시뮬 ego
        # 이므로 [1] 을 본다. nuPlan 로그는 항상 pacifica 라 보통 같은 값이다.
        d = expert_states[1].car_footprint.vehicle_parameters.rear_axle_to_center \
            if len(expert_states) > 1 else rear_to_center
        return _center_pose(float(np.interp(t, gt, gx)), float(np.interp(t, gt, gy)),
                            float(np.interp(t, gt, gh)), d=d)

    plan_1hz, expert_1hz = [], []
    for k in range(1, k_max + 1):
        j_expert = offsets[k - 1]
        if j_expert >= len(expert_states):
            break
        if t_us[j_expert] <= plan_end:
            plan_1hz.append(_plan_pose_at(float(t_us[j_expert])))
            expert_1hz.append(expert_states[j_expert].center)
        else:
            # 로그 표본이 계획 끝을 넘어간다(타임스탬프 지터 때문에 마지막 표본에서 생긴다).
            # 공식은 계획의 마지막 상태를 쓰고 로그 쪽을 그 시각에서 보간한다.
            plan_1hz.append(plan_poses[n_plan - 1])
            expert_1hz.append(_expert_pose_at(plan_end))

    if plan_1hz:
        d1 = compute_traj_errors(plan_1hz, expert_1hz, heading_diff_weight=0)
        a1 = compute_traj_heading_errors(plan_1hz, expert_1hz)
    else:
        d1 = a1 = np.zeros(0)

    per_h: Dict[str, List[float]] = {k: [] for k in ("ade", "fde", "maxde", "ahe", "fhe")}
    for h in horizons:
        if len(d1) < h:                        # 궤적이 짧아 그 지평을 채우지 못한다
            for key in per_h:
                per_h[key].append(float("nan"))
            continue
        d, a = d1[:h], a1[:h]
        per_h["ade"].append(float(np.mean(d)))
        per_h["fde"].append(float(d[-1]))
        per_h["maxde"].append(float(np.max(d)))
        per_h["ahe"].append(float(np.mean(a)))
        per_h["fhe"].append(float(a[-1]))

    # nanmean 이 아니라 mean 이다. 공식은 언제나 세 지평의 평균을 내므로, 지평 하나를
    # 못 채운 프레임을 남은 지평만으로 평균하면 값이 크게 낮아진 채(실측 36%) 누적에
    # 섞인다. NaN 으로 두면 호출부의 유한성 검사가 그 프레임을 통째로 제외한다.
    frame = {
        "ol_ade_m": float(np.mean(per_h["ade"])),
        "ol_fde_m": float(np.mean(per_h["fde"])),
        "ol_ahe_rad": float(np.mean(per_h["ahe"])),
        "ol_fhe_rad": float(np.mean(per_h["fhe"])),
    }
    frame.update({f"ol_ade_h{h}": per_h["ade"][i] for i, h in enumerate(horizons)})

    diagnostics = {
        # 렌더러가 그리는 곡선 — 1 초 표본이 아니라 스텝 전부(+dt … +T*dt)다.
        "open_loop_de_per_step": de.tolist(),
        "open_loop_he_per_step": he.tolist(),
        "open_loop_per_horizon": {k: list(v) for k, v in per_h.items()},
        "open_loop_horizons": list(horizons),
        # 공식 표본이 실제로 본 시각(스텝 단위). 곡선 위에 이 점들을 강조한다.
        "open_loop_sample_steps": offsets[:len(d1)],
    }
    return frame, diagnostics
