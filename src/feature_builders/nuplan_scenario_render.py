from typing import Dict, List, Set

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.patches import Polygon
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)

from src.scenario_manager.scenario_manager import ScenarioManager
from ..utils.vis import *

AGENT_COLOR_MAPPING = {
    TrackedObjectType.VEHICLE: "#001eff",
    TrackedObjectType.PEDESTRIAN: "#9500ff",
    TrackedObjectType.BICYCLE: "#ff0059",
}

TRAFFIC_LIGHT_COLOR_MAPPING = {
    TrafficLightStatusType.GREEN: "#2ca02c",
    TrafficLightStatusType.YELLOW: "#ff7f0e",
    TrafficLightStatusType.RED: "#d62728",
}

from src.planners.evaluator.common.enum import WeightedMetricIndex, MultiMetricIndex


class NuplanScenarioRender:
    def __init__(
        self,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
        bounds=60,
        offset=20,
        disable_agent=False,
    ) -> None:
        super().__init__()

        self.future_horizon = future_horizon
        self.future_samples = int(self.future_horizon / sample_interval)
        self.sample_interval = sample_interval
        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width
        self.bounds = bounds
        self.offset = offset
        self.disable_agent = disable_agent
        self.initialize = False
        self.scenario_manager = None
        self.need_update = False
        self.candidate_index = None
        self._history_trajectory = []
        self._expert_history_trajectory = []

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.static_objects_types = [
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.BARRIER,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        self.road_elements = [
            # SemanticMapLayer.ROADBLOCK,
            # SemanticMapLayer.ROADBLOCK_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            # SemanticMapLayer.CROSSWALKh
        ]

    def render_from_simulation(
        self,
        current_input: PlannerInput = None,
        initialization: PlannerInitialization = None,
        route_roadblock_ids: List[str] = None,
        scenario=None,
        iteration=None,
        planning_trajectory=None,
        candidate_trajectories=None,
        predictions=None,
        rollout_trajectories=None,
        agent_attn_weights=None,
        candidate_index=None,
        return_img=True,
        agent_future_log=None,  # (N, T, >=2) ego-local — 주변 agent 의 GT(로그) 미래
    ):
        ego_state = current_input.history.ego_states[-1]
        map_api = initialization.map_api
        tracked_objects = current_input.history.observations[-1]
        traffic_light_status = current_input.traffic_light_data
        mission_goal = initialization.mission_goal
        if route_roadblock_ids is None:
            route_roadblock_ids = initialization.route_roadblock_ids

        self.candidate_index = candidate_index

        if scenario is not None:
            gt_state = scenario.get_ego_state_at_iteration(iteration)
            gt_trajectory = scenario.get_ego_future_trajectory(
                iteration=iteration,
                time_horizon=self.future_horizon,
                num_samples=self.future_samples,
            )
        else:
            gt_state, gt_trajectory = None, None

        return self.render(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids,
            tracked_objects=tracked_objects,
            traffic_light_status=traffic_light_status,
            mission_goal=mission_goal,
            gt_state=gt_state,
            gt_trajectory=gt_trajectory,
            planning_trajectory=planning_trajectory,
            candidate_trajectories=candidate_trajectories,
            rollout_trajectories=rollout_trajectories,
            predictions=predictions,
            agent_attn_weights=agent_attn_weights,
            agent_future_log=agent_future_log,
            return_img=return_img,
        )
    
    def render_dagger_scene_from_simulation(
        self,
        current_input: PlannerInput = None,
        initialization: PlannerInitialization = None,
        route_roadblock_ids: List[str] = None,
        scenario=None,
        iteration=None,
        planning_trajectory=None,
        candidate_trajectories=None,
        predictions=None,
        rollout_trajectories=None,
        agent_attn_weights=None,
        candidate_index=None,
        return_img=True,
        smoothed_trajectory=None,
        refined_trajectory=None,
        dagger_information=None,
        evaluator_result=None, # {"ml": {metric: v}, "rf": {metric: v}}  (구 형식도 허용)
        agent_future_log=None,  # (N, T, >=2) ego-local — 주변 agent 의 GT(로그) 미래
        #: 취득 이유별 상세 시각화 재료.
        #:   {"comfort": {"ml":{lon_accel,lat_accel}, "rf":{...}, "bounds":{...}},
        #:    "ttc":     {"ml": violation|None, "rf": violation|None}}
        #: comfort/TTC 는 0/1 이라 표만으로는 취득 사유를 판단할 수 없어 별도로 그린다.
        reason_detail=None,
    ):
        ego_state = current_input.history.ego_states[-1]
        map_api = initialization.map_api
        tracked_objects = current_input.history.observations[-1]
        traffic_light_status = current_input.traffic_light_data
        mission_goal = initialization.mission_goal
        if route_roadblock_ids is None:
            route_roadblock_ids = initialization.route_roadblock_ids

        self.candidate_index = candidate_index

        if scenario is not None:
            gt_state = scenario.get_ego_state_at_iteration(iteration)
            gt_trajectory = scenario.get_ego_future_trajectory(
                iteration=iteration,
                time_horizon=self.future_horizon,
                num_samples=self.future_samples,
            )
        else:
            gt_state, gt_trajectory = None, None

        return self.render_dagger_scene(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids,
            tracked_objects=tracked_objects,
            traffic_light_status=traffic_light_status,
            mission_goal=mission_goal,
            gt_state=gt_state,
            gt_trajectory=gt_trajectory,
            planning_trajectory=planning_trajectory,
            candidate_trajectories=candidate_trajectories,
            rollout_trajectories=rollout_trajectories,
            predictions=predictions,
            agent_attn_weights=agent_attn_weights,
            return_img=return_img,
            smoothed_trajectory=smoothed_trajectory,
            refined_trajectory=refined_trajectory,
            dagger_information=dagger_information,
            evaluator_result=evaluator_result,
            agent_future_log=agent_future_log,
            reason_detail=reason_detail,
        )

    def render_from_scenario(
        self,
        scenario: AbstractScenario,
        ego_state: EgoState = None,
        iteration=0,
        planning_trajectory=None,
        candidate_trajectories=None,
        rollout_trajectories=None,
        predictions=None,
        return_image=True,
    ):
        if ego_state is None:
            ego_state = scenario.get_ego_state_at_iteration(iteration)
        map_api = scenario.map_api
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        tracked_objects = scenario.get_tracked_objects_at_iteration(iteration)
        traffic_light_status = scenario.get_traffic_light_status_at_iteration(iteration)
        mission_goal = scenario.get_mission_goal()
        gt_state = scenario.get_ego_state_at_iteration(iteration)
        gt_trajectory = scenario.get_ego_future_trajectory(
            iteration=iteration,
            time_horizon=self.future_horizon,
            num_samples=self.future_samples,
        )

        return self.render(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids,
            tracked_objects=tracked_objects,
            traffic_light_status=traffic_light_status,
            mission_goal=mission_goal,
            gt_state=gt_state,
            gt_trajectory=gt_trajectory,
            planning_trajectory=planning_trajectory,
            candidate_trajectories=candidate_trajectories,
            rollout_trajectories=rollout_trajectories,
            predictions=predictions,
            return_img=return_image,
        )

    def render(
        self,
        map_api: AbstractMap,
        ego_state: EgoState,
        route_roadblock_ids: List[str],
        tracked_objects: TrackedObjects,
        traffic_light_status: Dict[int, TrafficLightStatusData],
        mission_goal: StateSE2,
        gt_state=None,
        gt_trajectory=None,
        planning_trajectory=None,
        candidate_trajectories=None,
        rollout_trajectories=None,
        predictions=None,
        agent_attn_weights=None,
        agent_future_log=None,  # (N, T, >=2) ego-local — 주변 agent 의 GT(로그) 미래
        return_img=False,
    ):
        fig, ax = plt.subplots(figsize=(10, 10))

        self._history_trajectory.append(ego_state.rear_axle.array)
        if gt_state is not None:
            self._expert_history_trajectory.append(gt_state.rear_axle.array)

        if self.scenario_manager is None:
            self.scenario_manager = ScenarioManager(
                map_api, ego_state, route_roadblock_ids
            )
            self.scenario_manager.get_route_roadblock_ids()
            self.need_update = True

        if self.need_update:
            self.scenario_manager.update_ego_state(ego_state)
            self.scenario_manager.update_drivable_area_map()
            self.scenario_manager.update_ego_path()

        self.origin = ego_state.rear_axle.array
        self.angle = ego_state.rear_axle.heading
        self.rot_mat = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ],
            dtype=np.float64,
        )

        self._plot_map(
            ax,
            map_api,
            ego_state.center.point,
            traffic_light_status,
            set(route_roadblock_ids),
        )

        self._plot_reference_lines(ax, self.scenario_manager.get_reference_lines())

        self._plot_ego(ax, ego_state)

        if gt_state is not None:
            self._plot_ego(ax, gt_state, gt=True)
            gt_trajectory = np.array([state.rear_axle.array for state in gt_trajectory])
            gt_trajectory = np.matmul(gt_trajectory - self.origin, self.rot_mat)
            ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], color="blue", alpha=0.5)

        if not self.disable_agent:
            for track in tracked_objects.tracked_objects:
                self._plot_tracked_object(ax, track, agent_attn_weights)

        if planning_trajectory is not None:
            self._plot_planning(ax, planning_trajectory)
     
        if candidate_trajectories is not None:
            self._plot_candidate_trajectories(ax, candidate_trajectories)

        if rollout_trajectories is not None:
            self._plot_rollout_trajectories(ax, rollout_trajectories)

        if predictions is not None:
            self._plot_prediction(ax, predictions)

        # 주변 agent 의 GT(로그) 미래. 채점 world 가 바로 이것이므로,
        # "왜 여기서 충돌/TTC 판정이 났는지" 를 화면에서 검증하려면 반드시 필요하다.
        # (정지 화면에는 t=0 위치만 나오는데 궤적은 8 초치라 눈으로 판단할 수 없었다)
        if agent_future_log is not None:
            self._plot_agent_future_log(ax, agent_future_log)

        self._plot_mission_goal(ax, mission_goal)
        self._plot_history(ax)

        ax.axis("equal")
        ax.set_xlim(xmin=-self.bounds + self.offset, xmax=self.bounds + self.offset)
        ax.set_ylim(ymin=-self.bounds, ymax=self.bounds)
        ax.axis("off")
        plt.tight_layout(pad=0)

        if return_img:
            fig.canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                int(height), int(width), 3
            )
            plt.close(fig)
            return img
        else:
            plt.show()

    def render_dagger_scene(
        self,
        map_api: AbstractMap,
        ego_state: EgoState,
        route_roadblock_ids: List[str],
        tracked_objects: TrackedObjects,
        traffic_light_status: Dict[int, TrafficLightStatusData],
        mission_goal: StateSE2,
        gt_state=None,
        gt_trajectory=None,
        planning_trajectory=None,
        candidate_trajectories=None,
        rollout_trajectories=None,
        predictions=None,
        agent_attn_weights=None,
        return_img=False,
        smoothed_trajectory=None,
        refined_trajectory=None,
        dagger_information=None,
        evaluator_result=None, # {"ml": {metric: v}, "rf": {metric: v}}  (구 형식도 허용)
        agent_future_log=None,  # (N, T, >=2) ego-local — 주변 agent 의 GT(로그) 미래
        #: 취득 이유별 상세 시각화 재료.
        #:   {"comfort": {"ml":{lon_accel,lat_accel}, "rf":{...}, "bounds":{...}},
        #:    "ttc":     {"ml": violation|None, "rf": violation|None}}
        #: comfort/TTC 는 0/1 이라 표만으로는 취득 사유를 판단할 수 없어 별도로 그린다.
        reason_detail=None,
    ):
        fig, ax = plt.subplots(figsize=(10, 10))

        self._history_trajectory.append(ego_state.rear_axle.array)
        if gt_state is not None:
            self._expert_history_trajectory.append(gt_state.rear_axle.array)

        if self.scenario_manager is None:
            self.scenario_manager = ScenarioManager(
                map_api, ego_state, route_roadblock_ids
            )
            self.scenario_manager.get_route_roadblock_ids()
            self.need_update = True

        if self.need_update:
            self.scenario_manager.update_ego_state(ego_state)
            self.scenario_manager.update_drivable_area_map()
            self.scenario_manager.update_ego_path()

        self.origin = ego_state.rear_axle.array
        self.angle = ego_state.rear_axle.heading
        self.rot_mat = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ],
            dtype=np.float64,
        )

        self._plot_map(
            ax,
            map_api,
            ego_state.center.point,
            traffic_light_status,
            set(route_roadblock_ids),
        )

        self._plot_reference_lines(ax, self.scenario_manager.get_reference_lines())

        self._plot_ego(ax, ego_state)


        if gt_state is not None:
            self._plot_ego(ax, gt_state, gt=True)
            gt_trajectory = np.array([state.rear_axle.array for state in gt_trajectory])
            gt_trajectory = np.matmul(gt_trajectory - self.origin, self.rot_mat)
            ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], color="blue", alpha=0.5)

        # TTC 위반을 만든 객체 토큰 — 어느 후보(ml/sm/rf)에서든 걸린 객체를 빨갛게 칠한다.
        # 후보 이름을 열거하지 않고 넘어온 dict 의 키를 그대로 순회한다.
        ttc_detail = (reason_detail or {}).get("ttc") or {}
        collision_detail = (reason_detail or {}).get("collision") or {}
        hit_tokens = set()
        for side in set(ttc_detail) | set(collision_detail):
            v = ttc_detail.get(side)
            if v and v.get("track_token"):
                hit_tokens.add(v["track_token"])
            # 실제 at-fault 충돌 상대도 같은 빨간색으로 (원인이 TTC 와 다를 수 있다)
            hit_tokens.update(collision_detail.get(side) or [])

        if not self.disable_agent:
            for track in tracked_objects.tracked_objects:
                self._plot_tracked_object(ax, track, agent_attn_weights,
                                          highlight_tokens=hit_tokens)

        if planning_trajectory is not None:
            self._plot_ml_planning(ax, planning_trajectory)

        if smoothed_trajectory is not None:
            self._plot_smoothed_planning(ax, smoothed_trajectory)

        if refined_trajectory is not None:
            self._plot_refined_planning(ax, refined_trajectory)

        if candidate_trajectories is not None:
            self._plot_candidate_trajectories(ax, candidate_trajectories)

        if rollout_trajectories is not None:
            self._plot_rollout_trajectories(ax, rollout_trajectories)

        if predictions is not None:
            self._plot_prediction(ax, predictions)

        # 주변 agent 의 GT(로그) 미래. 채점 world 가 바로 이것이므로,
        # "왜 여기서 충돌/TTC 판정이 났는지" 를 화면에서 검증하려면 반드시 필요하다.
        # (정지 화면에는 t=0 위치만 나오는데 궤적은 8 초치라 눈으로 판단할 수 없었다)
        if agent_future_log is not None:
            self._plot_agent_future_log(ax, agent_future_log)

        # 최종 점수 비교는 별도 박스가 아니라 좌하단 표의 최상단 행으로 통합한다.

        # 좌하단 단일 표: 최상단 SCORE 행 + 세부지표 8행.
        # refine 우세=파랑 / 열세=빨강 / 동일=검정
        # 취득 사유 상세: TTC 충돌 지점(궤적 위) + comfort a_x/a_y 플롯(표 오른쪽)
        if reason_detail:
            for side in ("rf", "ml"):   # refined 를 위에 그린다
                self._plot_ttc_violation(ax, (reason_detail.get("ttc") or {}).get(side))

        if evaluator_result is not None:
            self._plot_evaluator_result(ax, evaluator_result, summary=dagger_information)
        elif dagger_information is not None:
            self._plot_dagger_information(ax, dagger_information)

        # comfort a_x / a_y 플롯은 점수표 **오른쪽**에 붙는다. 표 폭을 실측한 뒤에
        # 그려야 겹치지 않으므로 _plot_evaluator_result 다음에 둔다.
        if reason_detail:
            self._plot_comfort_panel(ax, reason_detail.get("comfort"))

        self._plot_mission_goal(ax, mission_goal)
        self._plot_history(ax)

        ax.axis("equal")
        ax.set_xlim(xmin=-self.bounds + self.offset, xmax=self.bounds + self.offset)
        ax.set_ylim(ymin=-self.bounds, ymax=self.bounds)
        ax.axis("off")
        plt.tight_layout(pad=0)

        if return_img:
            fig.canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                int(height), int(width), 3
            )
            plt.close(fig)
            return img
        else:
            plt.show()

    def _plot_map(
        self,
        ax,
        map_api: AbstractMap,
        query_point: Point2D,
        traffic_light_status: Dict[int, TrafficLightStatusData],
        route_roadblock_ids: Set[str],
    ):
        road_objects = map_api.get_proximal_map_objects(
            query_point, self.bounds + self.offset, self.road_elements
        )
        road_objects = (
            road_objects[SemanticMapLayer.LANE]
            + road_objects[SemanticMapLayer.LANE_CONNECTOR]
        )
        tls = {tl.lane_connector_id: tl.status for tl in traffic_light_status}

        for obj in road_objects:
            obj_id = int(obj.id)
            kwargs = {"color": "lightgray", "alpha": 0.4, "ec": None, "zorder": 0}
            if obj.get_roadblock_id() in route_roadblock_ids:
                kwargs["color"] = "dodgerblue"
                kwargs["alpha"] = 0.1
                kwargs["zorder"] = 1
            ax.add_artist(self._polygon_to_patch(obj.polygon, **kwargs))

            # for stopline in obj.stop_lines:
            #     if stopline.id in plotted_stopline:
            #         continue
            #     kwargs = {"color": "k", "alpha": 0.3, "ec": None, "zorder": 1}
            #     ax.add_artist(self._polygon_to_patch(stopline.polygon, **kwargs))
            #     plotted_stopline.add(stopline.id)

            cl_color, linewidth = "gray", 1.0
            if obj_id in tls:
                cl_color = TRAFFIC_LIGHT_COLOR_MAPPING.get(tls[obj_id], "gray")
                linewidth = 1
            cl = np.array([[s.x, s.y] for s in obj.baseline_path.discrete_path])
            cl = np.matmul(cl - self.origin, self.rot_mat)
            ax.plot(
                cl[:, 0],
                cl[:, 1],
                color=cl_color,
                alpha=0.5,
                linestyle="--",
                zorder=1,
                linewidth=linewidth,
            )

        crosswalks = map_api.get_proximal_map_objects(
            query_point, self.bounds + self.offset, [SemanticMapLayer.CROSSWALK]
        )
        for obj in crosswalks[SemanticMapLayer.CROSSWALK]:
            xys = np.array(obj.polygon.exterior.coords.xy).T
            xys = np.matmul(xys - self.origin, self.rot_mat)
            polygon = Polygon(
                xys, color="gray", alpha=0.4, ec=None, zorder=3, hatch="///"
            )
            ax.add_patch(polygon)

    def _plot_ego(self, ax, ego_state: EgoState, gt=False):
        kwargs = {"lw": 1.5}
        if gt:
            ax.add_patch(
                self._polygon_to_patch(
                    ego_state.car_footprint.geometry,
                    color="gray",
                    alpha=0.3,
                    zorder=9,
                    **kwargs,
                )
            )
        else:
            ax.add_patch(
                self._polygon_to_patch(
                    ego_state.car_footprint.geometry,
                    ec="#ff7f0e",
                    fill=False,
                    zorder=10,
                    **kwargs,
                )
            )

        ax.plot(
            [1.69, 1.69 + self.length * 0.75],
            [0, 0],
            color="#ff7f0e",
            linewidth=1.5,
            zorder=11,
        )

    def _plot_tracked_object(self, ax, track: TrackedObject, agent_attn_weights=None,
                             highlight_tokens=None):
        center, angle = track.center.array, track.center.heading
        center = np.matmul(center - self.origin, self.rot_mat)
        angle = angle - self.angle

        direct = np.array([np.cos(angle), np.sin(angle)]) * track.box.length / 1.5
        direct = np.stack([center, center + direct], axis=0)

        # TTC 위반을 만든 객체는 빨간 굵은 상자로 — "누구 때문에 0 점인가" 를 즉시 본다
        hit = bool(highlight_tokens) and track.track_token in highlight_tokens
        color = "red" if hit else AGENT_COLOR_MAPPING.get(track.tracked_object_type, "k")
        ax.add_patch(
            self._polygon_to_patch(
                track.box.geometry, ec=color, fill=False, alpha=1.0,
                zorder=12 if hit else 4, lw=2.6 if hit else 1.5,
            )
        )

        if color != "k":
            ax.plot(direct[:, 0], direct[:, 1], color=color, linewidth=1, zorder=4)
        if agent_attn_weights is not None and track.track_token in agent_attn_weights:
            weight = agent_attn_weights[track.track_token]
            ax.text(
                center[0],
                center[1] + 0.5,
                f"{weight:.2f}",
                color="red",
                zorder=5,
                fontsize=7,
            )

    def _polygon_to_patch(self, polygon: shapely.geometry.Polygon, **kwargs):
        polygon = np.array(polygon.exterior.xy).T
        polygon = np.matmul(polygon - self.origin, self.rot_mat)
        return patches.Polygon(polygon, **kwargs)

    def _plot_refined_planning(self, ax, mpc_refined_trajectory: np.ndarray):
        # MPC 개선 궤적 = 파랑 **점선**, ML 위에 그린다.
        # 두 궤적은 대개 거의 겹치므로, 실선끼리면 위에 그려진 쪽이 아래를 완전히 덮어
        # "refinement 가 무엇을 바꿨는지" 가 화면에서 보이지 않는다.
        plot_polyline(
            ax,
            [mpc_refined_trajectory],
            linewidth=2.5,
            arrow=False,
            zorder=8,
            alpha=0.95,
            color="#1F5FB4",
            color_change=False,
            linestyle="--",
        )

    def _plot_smoothed_planning(self, ax, smoothed_trajectory: np.ndarray):
        # 저역통과 필터를 건 궤적 = 초록 **점선**. refined 와 같은 이유로 점선이다
        # (ML 과 거의 겹치므로 실선이면 무엇이 바뀌었는지 화면에서 보이지 않는다).
        # 색은 점수표·comfort 패널의 "sm" 계열색과 맞춘다 (evaluator/viz.py).
        plot_polyline(
            ax,
            [smoothed_trajectory],
            linewidth=2.5,
            arrow=False,
            zorder=8,
            alpha=0.95,
            color="#2E8B57",
            color_change=False,
            linestyle=":",
        )

    def _plot_ml_planning(self, ax, planning_trajectory: np.ndarray):
        # ML 궤적 = 자홍 실선
        plot_polyline(
            ax,
            [planning_trajectory],
            linewidth=2.5,
            arrow=False,
            zorder=7,
            alpha=0.85,
            color="#D6249F",
            color_change=False,
        )

    def _plot_dagger_information(self, ax, dagger_information):
        info_text = ""   # [논문용] "Dagger Info" 헤더 제거, 수치만 표시
        for key in dagger_information:
            value = dagger_information[key]
            if isinstance(value, float):
                value = f"{value:.3f}"
            info_text += f"{key}: {value}\n"

        # Remove trailing newline
        info_text = info_text.rstrip('\n')

        # Count number of lines for box sizing
        num_lines = len(info_text.split('\n'))

        # [논문용] 좌하단 배치 + 흰 배경(불투명)
        ax.text(
            0.05, 0.05, info_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='black',
                alpha=1.0,
                linewidth=1.5
            )
        )

    def _plot_planning(self, ax, planning_trajectory: np.ndarray):
        plot_polyline(
            ax,
            [planning_trajectory],
            linewidth=3,
            arrow=False,
            zorder=6,
            alpha=0.7,
            cmap="spring",
        )

    def _plot_candidate_trajectories(self, ax, candidate_trajectories: np.ndarray):
        for traj in candidate_trajectories:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="gray",
                alpha=0.5,
                zorder=5,
                linewidth=2,
            )
            ax.scatter(traj[-1, 0], traj[-1, 1], color="gray", zorder=5, s=10)

    def _plot_rollout_trajectories(self, ax, candidate_trajectories: np.ndarray):
        for i, traj in enumerate(candidate_trajectories):
            kwargs = {"lw": 1.5, "zorder": 5, "color": "cyan"}
            if self.candidate_index is not None and i == self.candidate_index:
                kwargs = {"lw": 5, "zorder": 6, "color": "red"}
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, **kwargs)
            ax.scatter(traj[-1, 0], traj[-1, 1], color="cyan", zorder=5, s=10)

    def _plot_prediction(self, ax, predictions: np.ndarray):
        kwargs = {"lw": 3}
        for pred in predictions:
            pred = pred[:40, ..., :2]
            self._plot_polyline(ax, pred, cmap="Greys_r", **kwargs)

    def _plot_agent_future_log(self, ax, agent_future_log) -> None:
        """주변 agent 의 GT(로그) 미래 궤적을 black->white 로 그린다.

        시간이 갈수록 밝아지므로 어느 차량이 **언제** 다가오는지 읽을 수 있다.
        비반응 로그 재생이라 이 궤적이 곧 채점에 쓰인 world 다.
        """
        arr = np.asarray(agent_future_log)
        if arr.ndim != 3 or arr.shape[0] == 0:
            return
        for pred in arr:
            pts = np.asarray(pred[:, :2], dtype=np.float64)
            if len(pts) < 2 or not np.all(np.isfinite(pts)):
                continue
            # 제자리 정지 궤적은 arc length 가 0 이라 정규화가 깨진다
            if np.linalg.norm(pts[-1] - pts[0]) < 1e-3:
                continue
            # zorder 3.5: lane fill(0~1) 과 crosswalk hatch(3) 위, agent box(4) 아래.
            # 1 로 두면 lane 폴리곤에 덮여 사실상 안 보인다.
            self._plot_polyline(ax, pts, cmap="gray", lw=1.6, alpha=0.75, zorder=3.5)

    #: comfort / TTC 패널 공통 색. ML=회색 계열, refined=파랑, 위반=빨강.
    _PANEL_ML = "#666666"
    _PANEL_RF = "#1F5FB4"
    _PANEL_BAD = "#C0392B"

    def _plot_ttc_violation(self, ax, ttc_violation) -> None:
        """TTC 위반 지점을 궤적 위에 빨간 x 로 찍고 라벨을 단다.

        `ttc_violation` 은 위반을 만든 첫 스텝의
        {step, track_token, ttc, ego_pose(global), track_pose(global)} 이다.
        ego_pose 는 등속 투영으로 얻은 **예상 충돌 시점의 ego 위치**라
        정지 화면에서 "어디서 부딪히는가" 를 그대로 가리킨다.
        """
        if not ttc_violation:
            return
        ego_p = np.asarray(ttc_violation.get("ego_pose", []), dtype=float)
        if ego_p.size < 2:
            return
        p = np.matmul(ego_p[:2] - self.origin, self.rot_mat)

        trk = np.asarray(ttc_violation.get("track_pose", []), dtype=float)
        if trk.size >= 2:
            q = np.matmul(trk[:2] - self.origin, self.rot_mat)
            # 예상 충돌 상대 위치까지 잇는 얇은 선 — 어느 객체와인지 연결해 보여준다
            ax.plot([p[0], q[0]], [p[1], q[1]], color=self._PANEL_BAD,
                    lw=1.0, ls=":", alpha=0.9, zorder=13)
            ax.scatter(q[0], q[1], marker="x", s=70, linewidths=2.0,
                       color=self._PANEL_BAD, zorder=13)

        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3.0,
                   color=self._PANEL_BAD, zorder=14)
        # 라벨에는 **위반 스텝의 시각**을 반드시 넣는다. TTC 값만 적으면
        # "TTC 0.00s" 를 t=0 으로 오해한다(실제로 오해가 있었다).
        #   t=+X.Xs : 궤적 시작으로부터의 경과 시간. step 0 은 두 후보가 공유하는
        #             초기 상태라 여기서 갈릴 수 없으므로 위반은 항상 t>=+0.1s 다.
        #   TTC     : 그 시점에서 등속 투영으로 본 충돌까지 남은 시간(0 이면 충돌 중).
        step = ttc_violation.get("step")
        t_off = "t=+%.1fs  " % (float(step) * 0.1) if step is not None else ""
        ax.annotate(
            "collision  %sTTC %.2fs" % (t_off, float(ttc_violation.get("ttc", float("nan")))),
            xy=(p[0], p[1]), xytext=(8, 8), textcoords="offset points",
            fontsize=8, family="monospace", color="white", zorder=15,
            bbox=dict(boxstyle="round,pad=0.28", fc=self._PANEL_BAD, ec="none", alpha=0.92),
        )

    def _plot_comfort_panel(self, ax, comfort) -> None:
        """좌하단 점수표 **오른쪽**에 a_x / a_y 2D 플롯 두 개를 붙인다.

        comfort 는 0/1 이라 표만 봐서는 "왜 불편으로 찍혔는지" 를 알 수 없다.
        ML 과 refined 를 겹쳐 그리고 nuPlan 상·하한을 점선으로 표시하면
        어느 구간이 어느 정도로 넘었는지가 바로 보인다.
        한계를 넘은 구간은 빨간 점으로 덧찍고, 각 플롯에 최대 절대값을 적는다.

        고르는 3항(a_x / j_x / yaw rate)은 comfort 6항 중 실제로 comfort=0 을
        만드는 것들이다. 나머지 3항을 뺀 이유:
          - ego_lat_acceleration : 구조적으로 항상 0 이다(no-slip +
            get_acceleration_shifted 가 회전항을 x 로만 넣는다) → 그려도 평평하다
          - ego_jerk(magnitude)  : a_y=0 이라 |a_x| 의 미분과 사실상 같다 → j_x 와 중복
          - ego_yaw_acceleration : yaw_rate 의 미분이라 같은 축을 본다

        :param comfort: {"ml": {"lon_accel":[..],"lon_jerk":[..],"yaw_rate":[..]},
                         "rf": {...}, "bounds": {키마다 [lo,hi]}}
        """
        if not comfort:
            return
        ml = comfort.get("ml") or {}
        rf = comfort.get("rf") or {}
        bounds = comfort.get("bounds") or {}
        specs = [("lon_accel", "a_x  [m/s²]"),
                 ("lon_jerk",  "j_x  [m/s³]"),
                 ("yaw_rate",  "yaw rate  [rad/s]")]

        # 점수표 오른쪽에 3열. 표는 x0=0.012 에서 폭 ~0.38 을 쓰므로 0.44 부터 시작해
        # 폭 0.165 + 간격 0.03 으로 세 개를 0.995 까지 채운다.
        boxes = [[0.440, 0.030, 0.165, 0.185],
                 [0.635, 0.030, 0.165, 0.185],
                 [0.830, 0.030, 0.165, 0.185]]

        # 패널은 6항 중 2개만 그린다. jerk·lon_jerk·yaw_accel 이 comfort=0 을 만들면
        # 그림만 봐서는 설명이 안 되므로 실패한 항 이름을 패널 위에 한 줄로 적는다.
        failed = comfort.get("failed") or {}
        fm, fr = failed.get("ml") or [], failed.get("rf") or []
        if fm or fr:
            ax.text(
                0.44, 0.250,
                "comfort ✗   ML: %s   RF: %s" % (", ".join(fm) or "-", ", ".join(fr) or "-"),
                transform=ax.transAxes, fontsize=7.0, family="monospace",
                color=self._PANEL_BAD, va="bottom", ha="left", zorder=53,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", alpha=0.9),
            )

        for (key, title), box in zip(specs, boxes):
            a = np.asarray(ml.get(key, []), dtype=float)
            b = np.asarray(rf.get(key, []), dtype=float)
            if a.size == 0 and b.size == 0:
                continue
            sub = ax.inset_axes(box, transform=ax.transAxes, zorder=52)
            sub.patch.set_facecolor("white")
            sub.patch.set_alpha(0.94)

            lo, hi = (bounds.get(key) or [None, None])[:2]
            for series, colour, label in ((a, self._PANEL_ML, "ML"),
                                          (b, self._PANEL_RF, "RF")):
                if series.size == 0:
                    continue
                t = np.arange(series.size) * 0.1
                sub.plot(t, series, color=colour, lw=1.2, label=label, zorder=3)
                if lo is not None:
                    bad = (series <= lo) | (series >= hi)
                    if bad.any():
                        sub.scatter(t[bad], series[bad], s=7, color=self._PANEL_BAD,
                                    zorder=4, linewidths=0)

            if lo is not None:
                for y in (lo, hi):
                    sub.axhline(y, color=self._PANEL_BAD, ls="--", lw=0.9, alpha=0.8, zorder=2)

            # 각 플롯에서 max |a| 를 바로 읽을 수 있게 한다
            def peak(s):
                return float(np.abs(s).max()) if s.size else float("nan")
            sub.text(0.98, 0.96, "max|ML| %.2f\nmax|RF| %.2f" % (peak(a), peak(b)),
                     transform=sub.transAxes, ha="right", va="top",
                     fontsize=5.6, family="monospace", color="#222222", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BBBBBB", alpha=0.85))

            sub.set_title(title, fontsize=6.5, pad=2.0)
            sub.tick_params(labelsize=5.0, length=2, pad=1)
            sub.set_xlabel("t [s]", fontsize=6, labelpad=1)
            sub.grid(alpha=0.25, lw=0.4)
            for s in sub.spines.values():
                s.set_edgecolor("#444444")
            if key == "lon_accel":
                sub.legend(fontsize=5.5, loc="lower right", framealpha=0.85,
                           handlelength=1.2, borderpad=0.25)

    def _plot_polyline(self, ax, polyline, cmap="spring", **kwargs) -> None:
        arc = get_polyline_arc_length(polyline)
        polyline = polyline.reshape(-1, 1, 2)
        segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
        norm = plt.Normalize(arc.min(), arc.max())
        lc = LineCollection(
            segment,
            cmap=cmap,
            norm=norm,
            array=arc,
            **kwargs,
        )
        ax.add_collection(lc)

    def _plot_reference_lines(self, ax, ref_lines):
        for ref_line in ref_lines:
            ref_line_pos = np.matmul(ref_line[::20, :2] - self.origin, self.rot_mat)
            ref_line_angle = ref_line[::20, 2] - self.angle
            for p, angle in zip(ref_line_pos, ref_line_angle):
                ax.arrow(
                    p[0],
                    p[1],
                    np.cos(angle) * 1.5,
                    np.sin(angle) * 1.5,
                    color="magenta",
                    width=0.2,
                    head_width=0.8,
                    zorder=6,
                    alpha=0.2,
                )

    def _plot_mission_goal(self, ax, mission_goal: StateSE2):
        point = np.matmul(mission_goal.point.array - self.origin, self.rot_mat)
        ax.plot(point[0], point[1], marker="*", markersize=5, color="gold", zorder=6)

    def _plot_history(self, ax):
        history = np.array(self._history_trajectory)
        history = np.matmul(history - self.origin, self.rot_mat)
        ax.plot(
            history[:, 0],
            history[:, 1],
            color="#ff7f0e",
            alpha=0.5,
            zorder=6,
            linewidth=2,
        )

        if len(self._expert_history_trajectory) > 0:
            expert_history = np.array(self._expert_history_trajectory)
            expert_history = np.matmul(expert_history - self.origin, self.rot_mat)
            ax.plot(
                expert_history[:, 0],
                expert_history[:, 1],
                color="blue",
                alpha=0.5,
                zorder=6,
            )

    #: 세부지표 표시 순서. (키, 짧은 이름, 종류) — 곱셈항(x) 먼저, 그다음 가중항(w+가중치).
    METRIC_PATCH_ORDER = [
        ("no_ego_at_fault_collisions", "collision", "x"),
        ("drivable_area_compliance", "drivable", "x"),
        ("driving_direction_compliance", "direction", "x"),
        ("ego_is_making_progress", "making_prog", "x"),
        ("ego_progress_along_expert_route", "progress", "w5"),
        ("time_to_collision_within_bound", "TTC", "w5"),
        ("speed_limit_compliance", "speed_limit", "w4"),
        ("ego_is_comfortable", "comfort", "w2"),
    ]

    #: refine 이 더 나으면 파랑, 더 나쁘면 빨강, 같으면 검정
    _CMP_BETTER = "#1F5FB4"
    _CMP_WORSE = "#C0392B"
    _CMP_SAME = "#1A1A1A"
    _CMP_EPS = 1e-6

    #: 표 행 순서. (키, 표시명, 종류) — 곱셈항(x) 먼저, 그다음 가중항(w+가중치).
    #
    # 진행률 계열(ego_progress_along_expert_route, ego_is_making_progress)은 뺐다.
    # closed-loop 에서 시뮬 ego 가 로그 ego 와 발산하는데 분모가 "같은 시점부터의
    # 로그 진행량"이라 scene 마다 편향 크기가 달라 공정 비교가 불가능하고,
    # 채점에서도 1.0 으로 고정(PROGRESS_METRICS)돼 최종 점수에 기여하지 않는다.
    # 실측값은 CSV(gate_stats/all_steps.csv)와 TrajectoryScore.diagnostics 에 남는다.
    METRIC_PATCH_ORDER = [
        ("no_ego_at_fault_collisions", "collision", "x"),
        ("drivable_area_compliance", "drivable", "x"),
        ("driving_direction_compliance", "direction", "x"),
        ("time_to_collision_within_bound", "TTC", "w5"),
        ("speed_limit_compliance", "speed_limit", "w4"),
        ("ego_is_comfortable", "comfort", "w2"),
    ]

    #: refine 이 더 나으면 파랑, 더 나쁘면 빨강, 같으면 검정
    _CMP_BETTER = "#1F5FB4"
    _CMP_WORSE = "#C0392B"
    _CMP_SAME = "#1A1A1A"
    _CMP_EPS = 1e-6

    def _plot_evaluator_result(self, ax, evaluator_result, summary=None):
        """ML vs refined 평가결과를 **좌하단 단일 표**로 그린다.

        최상단 행이 최종 점수 비교(SCORE = S_ML -> S_refine, delta)이고,
        그 아래로 세부지표 8행이 같은 열에 정렬된다. 행마다
        **refine 이 더 나으면 파랑 / 더 나쁘면 빨강 / 같으면 검정**이다.

        열: metric | 종류 | ML | -> | RF | delta
        종류: x = 곱셈항, w5/w4/w2 = 가중항과 가중치(합 16).

        배경은 Rectangle 하나로 깔고 행은 개별 text 로 그린다 — 여러 줄 문자열
        위에 색을 덧그리면 linespacing 이 어긋나 정렬이 깨지기 때문이다.

        :param evaluator_result: {"ml": {metric: v}, "rf": {metric: v}} 또는 구 형식
        :param summary: {"S_ML":float,"S_refine":float,"decision":str,"guard":str}
        """
        ml, rf = self._normalize_evaluator_result(evaluator_result)
        if not ml or not rf:
            return

        def colour(d):
            if abs(d) <= self._CMP_EPS:
                return self._CMP_SAME
            return self._CMP_BETTER if d > 0 else self._CMP_WORSE

        rows = []  # (label, colour, bold)
        if summary and summary.get("S_ML") is not None:
            a, b = float(summary["S_ML"]), float(summary["S_refine"])
            rows.append(("%-12s%-3s %5.3f -> %5.3f  %+7.3f" % ("SCORE", "", a, b, b - a),
                         colour(b - a), True))
        for key, short, kind in self.METRIC_PATCH_ORDER:
            if key not in ml or key not in rf:
                continue
            a, b = float(ml[key]), float(rf[key])
            d = b - a
            diff = "" if abs(d) <= self._CMP_EPS else "%+.3f" % d
            rows.append(("%-12s%-3s %5.3f -> %5.3f  %7s" % (short, kind, a, b, diff),
                         colour(d), abs(d) > self._CMP_EPS))
        if not rows:
            return

        header = "%-12s%-3s %5s    %5s  %7s" % ("metric", "", "ML", "RF", "delta")
        ncol = len(header)

        fs = 8.5
        dy = 0.0205                      # 행 간격 (axes 좌표)
        x0, y0 = 0.012, 0.012            # 표 좌하단
        pad_x, pad_y = 0.010, 0.010
        n_extra = 2 if summary else 1    # header(+구분선) / decision 줄
        n_line = len(rows) + 1 + n_extra

        # 폭은 monospace 문자폭을 실측해 잡는다. 상수로 잡으면 dpi·figsize 가
        # 바뀔 때 배경이 글자를 자른다(실제로 한 번 잘렸다).
        fig = ax.get_figure()
        try:
            probe = ax.text(0, 0, "0" * ncol, transform=ax.transAxes,
                            fontsize=fs, family="monospace", alpha=0.0)
            fig.canvas.draw()
            bb = probe.get_window_extent(renderer=fig.canvas.get_renderer())
            width = bb.transformed(ax.transAxes.inverted()).width + 2 * pad_x
            probe.remove()
        except Exception:
            width = 0.0075 * ncol + 2 * pad_x

        ax.add_patch(plt.Rectangle(
            (x0, y0), width, dy * n_line + 2 * pad_y,
            transform=ax.transAxes, facecolor="white", edgecolor="#444444",
            linewidth=1.2, alpha=0.94, zorder=50,
        ))

        tx = x0 + pad_x
        y = y0 + pad_y                    # 아래에서 위로 쌓는다
        if summary:
            ax.text(tx, y, "decision: %s   guard: %s"
                    % (summary.get("decision", "-"), summary.get("guard", "-")),
                    transform=ax.transAxes, fontsize=fs - 0.5, family="monospace",
                    color="#555555", va="bottom", ha="left", zorder=51)
            y += dy

        for label, col, bold in reversed(rows):
            ax.text(tx, y, label, transform=ax.transAxes, fontsize=fs,
                    family="monospace", color=col,
                    fontweight="bold" if bold else "normal",
                    va="bottom", ha="left", zorder=51)
            y += dy

        ax.text(tx, y, "-" * ncol, transform=ax.transAxes, fontsize=fs,
                family="monospace", color="#999999", va="bottom", ha="left", zorder=51)
        y += dy * 0.55
        ax.text(tx, y, header, transform=ax.transAxes, fontsize=fs,
                family="monospace", color="#555555", va="bottom", ha="left", zorder=51)

    def _normalize_evaluator_result(self, evaluator_result):
        """구/신 두 형식을 {metric: value} 두 개(ml, rf)로 통일한다."""
        if not isinstance(evaluator_result, dict) or not evaluator_result:
            return None, None

        def pick(d):
            ml = rf = None
            for k, v in d.items():
                kl = str(k).strip().lower()
                if kl.startswith("ml") or "model" in kl:
                    ml = v
                elif kl.startswith("rf") or "refine" in kl:
                    rf = v
            return ml, rf

        if all(isinstance(v, dict) for v in evaluator_result.values()):
            return pick(evaluator_result)

        conv = {}
        for k, v in evaluator_result.items():
            try:
                multi, weight = v
            except Exception:
                continue
            d = {}
            for idx, val in enumerate(multi):
                try:
                    d[MultiMetricIndex(idx).name.lower()] = float(val)
                except Exception:
                    pass
            for idx, val in enumerate(weight):
                try:
                    d[WeightedMetricIndex(idx).name.lower()] = float(val)
                except Exception:
                    pass
            conv[k] = d
        return pick(conv)
