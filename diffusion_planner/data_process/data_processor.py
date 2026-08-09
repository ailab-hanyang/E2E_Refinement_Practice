import numpy as np
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
agent_past_process, 
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list,
agent_future_process
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states, sampled_past_ego_states_to_array
from diffusion_planner.data_process.utils import convert_to_model_inputs

from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses


class DataProcessor(object):
    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None) 

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len} # maximum number of points per feature to extract per feature layer.

    # Use for inference
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_agent_past = None # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

        
        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": np.array([0., 0., 1. ,0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), # ego centric x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate, we only use x, y, cos, sin during inference
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data
    
    # Use for data preprocess
    def work(self, scenarios):

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api        

            '''
            ego & agents past
            '''
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses, self.past_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]
            neighbor_agents_past, neighbor_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)
            
            static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
            
            '''
            Map
            '''
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids
                )

            coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
                map_api, self._map_features, ego_coords, self._radius, traffic_light_data
            )

            vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

            '''
            ego & agents future
            '''
            ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]

            sampled_future_observations = [present_tracked_objects] + future_tracked_objects
            future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
            neighbor_agents_future = agent_future_process(anchor_ego_state, future_tracked_objects_array_list, self.num_agents, neighbor_indices)


            '''
            ego current
            '''
            ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

            # gather data
            data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
            data.update(vector_map)

            self.save_to_disk(self._save_dir, data)

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def build_dagger_sample(self, scenario, iteration, past_ego_states, future_ego_states,
                            ml_trajectory=None):
        """Build ONE diffusion-format training sample for a DAgger cache hit.

        Mirrors ``work()`` but for the closed-loop DAgger context:
          - ego PAST comes from the rollout (``past_ego_states``, the planner's
            ego_state_history), not the logged scenario;
          - everything is anchored at the CURRENT iteration, not iteration 0;
          - ego FUTURE is the MPC-refined pseudo-GT (``future_ego_states`` =
            planner.refined_trajectory, global EgoStates), not the scenario GT;
          - neighbor FUTURE is the scenario GT at the current iteration (same as
            PLUTO's dagger feature).

        Returns a dict of numpy arrays (un-normalized, no batch dim) matching the
        diffusion ``.npz`` contract, or None if there is insufficient history/future.
        """
        if past_ego_states is None or len(past_ego_states) < self.num_past_poses + 1:
            return None
        if future_ego_states is None or len(future_ego_states) < self.num_future_poses:
            return None

        map_name = scenario._map_name
        token = scenario.token
        map_api = scenario.map_api

        # ── ego past (rollout) ──
        past = list(past_ego_states)[-(self.num_past_poses + 1):]  # 21 states
        current_ego_state = past[-1]
        ego_coords = Point2D(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)
        anchor_ego_state = np.array(
            [current_ego_state.rear_axle.x, current_ego_state.rear_axle.y, current_ego_state.rear_axle.heading],
            dtype=np.float64,
        )
        ego_agent_past = sampled_past_ego_states_to_array(past)
        time_stamps_past = np.array([s.time_point.time_us for s in past], dtype=np.int64)

        # ── neighbor & static past (scenario @ current iteration) ──
        present_tracked_objects = scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=iteration, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        neighbor_agents_past, neighbor_agents_types = \
            sampled_tracked_objects_to_array_list(sampled_past_observations)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = agent_past_process(
            ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents,
            static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state,
        )

        # ── map ──
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(iteration))
        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(current_ego_state, map_api, route_roadblock_ids)
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit,
                                 lane_route, self._map_features, self._max_elements, self._max_points)

        # ── ego current (real 10-dim; NOT the inference placeholder) ──
        ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

        # ── ego future: MPC-refined pseudo-GT, relative to current ego ──
        future = list(future_ego_states)[:self.num_future_poses]
        ego_agent_future = convert_absolute_to_relative_poses(
            current_ego_state.rear_axle, [s.rear_axle for s in future]
        )

        # ── neighbor future: scenario GT @ current iteration ──
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=iteration, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]
        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
        neighbor_agents_future = agent_future_process(
            anchor_ego_state, future_tracked_objects_array_list, self.num_agents, neighbor_indices
        )

        data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state,
                "ego_agent_future": ego_agent_future, "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}

        # ── ML 계획 궤적도 함께 남긴다 (라벨이 아니라 **진단용**) ──────────────
        # 게이트는 S_ML 과 S_refine 을 비교해 판정하는데, 지금까지 refined 만
        # 저장돼 "ML 이 왜 0 점인가" 를 사후에 확인할 수 없었다. (80,3) float32 =
        # 1 KB 로 비용이 없으니 함께 남긴다. 학습은 ego_agent_future 만 쓰므로
        # 이 키가 추가돼도 기존 데이터 로더는 영향받지 않는다.
        if ml_trajectory is not None:
            ml = np.asarray(ml_trajectory, dtype=np.float32)
            if ml.ndim == 2 and ml.shape[0] >= self.num_future_poses:
                data["ml_agent_future"] = ml[:self.num_future_poses, :3]

        data.update(vector_map)
        return data