# post_processor/interface/interface_pluto.py

import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

@torch.no_grad()
def FindNearestAgentsInfo(
    predictions: torch.Tensor,             # (N_agents, T, 5)  columns: [x_center, y_center, yaw, vx, vy] (ego/vehicle frame)
    agent_categories: torch.Tensor,        # (N_agents,)       categorical labels per agent
    agent_shapes: torch.Tensor,            # (N_agents, 2)     [width, length] per agent
    reference_trajectory: np.ndarray,      # (T, 4)            [x, y, yaw, speed] of ego (may be 0 in ego frame)
    initial_state: np.ndarray,             # (4,)              [x_rear, y_rear, yaw, speed] of ego at t=0
    filtering_step: int = 20,              # cutoff steps for rear agents
    low_speed_filtering_step: int = 5,    # cutoff steps when ego is moving slowly
    normal_speed_filtering_step: int = 20, # cutoff steps for front agents at normal speed (also used for selection)
    num_agents: int = 10,                  # how many nearest vehicles to keep
    large_value: float = 1e3,              # value used for padding or masking
    small_value: float = 0.1,
    low_speed_threshold: float = 1.0,      # [m/s] below this is considered low speed
    stationary_threshold: float = 0.8,     # [m] displacement below this is stationary
    # --- NEW: rear-collision exclusion window in ego frame (default enabled) ---
    exclude_xmin: float = -20.01,           # [m] exclude if xmin < x < xmax                                  # -30
    exclude_xmax: float = 7.0,             # [m] wheelbase 3.089 + agent half-length 4.0 + 4.0               # 7.5
    exclude_ymin: float = -2.1,            # [m] exclude if ymin < y < ymax                                  # -2.5
    exclude_ymax: float = 2.1,             # [m]                                                             # 2.5
    apply_exclusion: bool = True,          # apply the rectangular exclusion filter
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        predictions           (torch.Tensor): (N_agents, T, 5) predicted states in ego frame.
        agent_categories      (torch.Tensor): (N_agents,) integer codes (vehicles: 1 or 3).
        agent_shapes          (torch.Tensor): (N_agents, 2) [width, length].
        reference_trajectory  (np.ndarray):   (T, 4) ego reference [x, y, yaw, speed].
        initial_state         (np.ndarray):   (4,) ego initial [x_rear, y_rear, yaw, speed].
        filtering_step        (int):          Cutoff horizon for rear agents.
        low_speed_filtering_step (int):       Cutoff for moving front agents when ego is slow.
        normal_speed_filtering_step (int):    Selection/cutoff step for front agents.
        num_agents            (int):          Number of nearest agents to keep.
        large_value           (float):        Large number for masking/padding.
        low_speed_threshold   (float):        Ego speed below which is considered low-speed.
        stationary_threshold  (float):        Movement below which agent is stationary.
        exclude_*             (float):        Rectangular exclusion window in ego frame (x forward, y left).
        apply_exclusion       (bool):         If True, agents within the window are excluded.

    Outputs:
        positions (np.ndarray): (T, num_agents * 6)
            Stacked per-agent over time: [x_center, y_center, x_rear, y_rear, x_front, y_front] for each agent.
            If fewer than num_agents are available, remaining columns are filled with `large_value`.

        agent_width_info (np.ndarray): (T, num_agents)
            Per-agent width aligned with `positions` order (agent1..agentN). Width is constant over time.
            For moving agents, rows beyond the masking horizon are set to `large_value` to match `positions`.
            If fewer than num_agents are available, remaining columns are filled with `large_value`.
    """
    # 0) Low-speed flag
    is_ego_speed_low = abs(float(initial_state[3])) < low_speed_threshold

    # Basic dims
    num_time_steps = predictions.shape[1]
    T = num_time_steps

    # 1) Select only vehicle agents
    # [DEBUG] category 체계 확인
    # unique_cats, counts = torch.unique(agent_categories, return_counts=True)
    # cat_dict = {int(c): int(n) for c, n in zip(unique_cats, counts)}
    # print(f"[DEBUG FindNearestAgentsInfo] 수신된 agent_categories unique={unique_cats.tolist()}, counts={counts.tolist()}, total={len(agent_categories)}")
    # print(f"[DEBUG FindNearestAgentsInfo] vehicle_mask(==1) sum={int((agent_categories==1).sum())}  ← feature_builder VEHICLE")
    # print(f"[DEBUG FindNearestAgentsInfo] vehicle_mask(==3) sum={int((agent_categories==3).sum())}  ← planner VEHICLE")
    # print(f"[DEBUG FindNearestAgentsInfo] ped_mask(==1)     sum={int((agent_categories==1).sum())}  ← planner PEDESTRIAN")
    # print(f"[DEBUG FindNearestAgentsInfo] ped_mask(==2)     sum={int((agent_categories==2).sum())}  ← feature_builder PEDESTRIAN")
    # print(f"[DEBUG FindNearestAgentsInfo] 현재 코드 vehicle_mask(==3) sum={vehicle_mask.sum().item()} → 0이면 vehicle이 안잡힘!")
    
    vehicle_mask = (agent_categories == 3)  # 3=VEHICLE (1=PEDESTRIAN, per pluto_planner convention)
    if vehicle_mask.sum() == 0:
        positions_empty = np.full((T, num_agents * 6), large_value, dtype=np.float32)
        widths_empty = np.full((T, num_agents), small_value, dtype=np.float32)
        return positions_empty, widths_empty

    selected_indices    = vehicle_mask.nonzero(as_tuple=True)[0]
    vehicle_predictions = predictions[selected_indices].cpu().numpy()     # (N_v, T, 5) in ego frame
    vehicle_shapes      = agent_shapes[selected_indices].cpu().numpy()    # (N_v, 2) [width, length]
    vehicle_categories  = agent_categories[selected_indices].cpu().numpy()# (N_v,)

    # 1-1) Exclude rear vehicles (x < 0, all y values)
    cutoff_step = int(np.clip(normal_speed_filtering_step, 0, T - 1))  # safe clamp
    cutoff_step_excl = 0  # use the FIRST step for exclusion window
    # COMMENTED OUT: Rear vehicle filtering disabled to consider all vehicles in MPC
    # if apply_exclusion and vehicle_predictions.shape[0] > 0:
    #     x_cut = vehicle_predictions[:, cutoff_step_excl, 0]

    #     # Rear vehicle filtering: x < 0 (behind ego) regardless of y position
    #     is_rear = (x_cut < 0.0)

    #     # Keep agents that are NOT rear vehicles
    #     exclusion_mask = ~is_rear

    #     # Apply exclusion
    #     vehicle_predictions = vehicle_predictions[exclusion_mask]
    #     vehicle_shapes      = vehicle_shapes[exclusion_mask]
    #     vehicle_categories  = vehicle_categories[exclusion_mask]

    #     if vehicle_predictions.shape[0] == 0:
    #         positions_empty = np.full((T, num_agents * 6), large_value, dtype=np.float32)
    #         widths_empty = np.full((T, num_agents), small_value, dtype=np.float32)
    #         return positions_empty, widths_empty
    
    ## Seungwoo Changed (DISABLED: AND inside_y 조건이 교차로/다차선에서 필터링 실패함)
    # if apply_exclusion and vehicle_predictions.shape[0] > 0:
    #     x_cut = vehicle_predictions[:, cutoff_step_excl, 0]
    #     y_cut = vehicle_predictions[:, cutoff_step_excl, 1]
    #     inside_x = (x_cut > exclude_xmin) & (x_cut < exclude_xmax)
    #     inside_y = (y_cut > exclude_ymin) & (y_cut < exclude_ymax)
    #     exclusion_mask = ~(inside_x & inside_y)  # keep agents NOT inside the window
    #     vehicle_predictions = vehicle_predictions[exclusion_mask]
    #     vehicle_shapes      = vehicle_shapes[exclusion_mask]
    #     vehicle_categories  = vehicle_categories[exclusion_mask]

    # 뒷차 필터링: x < 0이면 y 위치 무관하게 제외
    if apply_exclusion and vehicle_predictions.shape[0] > 0:
        x_cut = vehicle_predictions[:, cutoff_step_excl, 0]
        is_rear = (x_cut < 1.5)
        exclusion_mask = ~is_rear

        vehicle_predictions = vehicle_predictions[exclusion_mask]
        vehicle_shapes      = vehicle_shapes[exclusion_mask]
        vehicle_categories  = vehicle_categories[exclusion_mask]

        if vehicle_predictions.shape[0] == 0:
            positions_empty = np.full((T, num_agents * 6), large_value, dtype=np.float32)
            widths_empty = np.full((T, num_agents), small_value, dtype=np.float32)
            return positions_empty, widths_empty

    # 2) Compute distances at the selection step and pick nearest K agents
    ref_x, ref_y = reference_trajectory[cutoff_step, :2]
    distances = np.hypot(
        vehicle_predictions[:, cutoff_step, 0] - ref_x,
        vehicle_predictions[:, cutoff_step, 1] - ref_y
    )  # (N_v_filtered,)

    nearest_order = np.argsort(distances)[:num_agents]

    sel_pred   = vehicle_predictions[nearest_order]   # (K, T, 5)
    sel_shapes = vehicle_shapes[nearest_order]        # (K, 2) [width, length]
    sel_cats   = vehicle_categories[nearest_order]    # (K,)
    K = sel_pred.shape[0]

    # 3) Initial relative x to decide front/rear horizon
    ego_initial_x = initial_state[0]
    initial_rel_x = sel_pred[:, 0, 0] - ego_initial_x  # (K,)

    # 4) Compute center, rear, and front positions for each agent over time
    yaw_angles = sel_pred[:, :, 2]    # (K, T)
    center_x   = sel_pred[:, :, 0]    # (K, T)
    center_y   = sel_pred[:, :, 1]    # (K, T)

    half_length = (sel_shapes[:, 1] * 0.5)[:, None]   # (K,1) length -> half-length
    offset_x = half_length * np.cos(yaw_angles)
    offset_y = half_length * np.sin(yaw_angles)

    rear_x  = center_x - offset_x
    rear_y  = center_y - offset_y
    front_x = center_x + offset_x
    front_y = center_y + offset_y

    # Stack into shape (K, T, 6) → (T, K*6)
    stacked_positions = np.stack(
        [center_x, center_y, rear_x, rear_y, front_x, front_y],
        axis=2
    )  # (K, T, 6)
    positions = stacked_positions.transpose(1, 0, 2).reshape(T, -1)  # (T, K*6)

    # --- NEW: build agent_width_info aligned with selection order ---
    # sel_shapes[:, 0] is width per selected agent
    sel_widths = sel_shapes[:, 0].astype(np.float32)  # (K,)
    agent_width_info = np.full((T, num_agents), small_value, dtype=np.float32)

    if K > 0:
        # Fill first K columns with constant widths across time
        agent_width_info[:, :K] = sel_widths.reshape(1, K).repeat(T, axis=0)

    # 5) Pad if fewer than num_agents
    if K < num_agents:
        pad_width = (num_agents - K) * 6
        padding_array = np.full((T, pad_width), large_value, dtype=positions.dtype)
        positions = np.hstack([positions, padding_array])  # (T, num_agents*6)

    # 6) Apply per-agent cutoff and stationary masking (positions + widths together)
    for idx in range(K):
        # Determine cutoff horizon: rear vs front
        horizon = filtering_step if initial_rel_x[idx] < 0 else normal_speed_filtering_step
        horizon = int(np.clip(horizon, 1, T))  # ensure at least 1 and at most T

        # Compute displacement over [0, horizon)
        dx = sel_pred[idx, horizon - 1, 0] - sel_pred[idx, 0, 0]
        dy = sel_pred[idx, horizon - 1, 1] - sel_pred[idx, 0, 1]
        displacement = np.hypot(dx, dy)

        # For special category agents (e.g., buses/trucks?), further limit horizon
        if sel_cats[idx] == 3:
            horizon = min(horizon, 5)

        # If ego is slow and agent is moving, shorten horizon
        if is_ego_speed_low and displacement > stationary_threshold:
            horizon = min(horizon, low_speed_filtering_step)

        col_start_idx = idx * 6
        col_end_idx   = col_start_idx + 6

        if displacement > stationary_threshold:
            # Mask all times beyond horizon for positions and widths
            positions[horizon:, col_start_idx:col_end_idx] = large_value
            # agent_width_info[horizon:, idx] = large_value
        else:
            # Stationary agent: keep initial pose for all times; width stays constant
            initial_slice = positions[0, col_start_idx:col_end_idx]
            positions[:, col_start_idx:col_end_idx] = initial_slice
            # agent_width_info[:, idx] already constant; no change needed

    return positions.astype(np.float32), agent_width_info.astype(np.float32)


@torch.no_grad()
def FindNearestStaticObjects(
    object_positions: torch.Tensor,
    horizon: int = 80,
    k: int = 5,
    fill_value: float = 1e6
) -> torch.Tensor:
    """
    Select K nearest static objects in front of the ego, pad missing entries,
    and repeat their coordinates over a time horizon.

    Inputs:
        object_positions (torch.Tensor): shape (N, 2), each row is [x, y] in world frame.
        horizon          (int):          Number of timesteps to repeat the positions.
        k                (int):          Number of nearest objects to select.
        fill_value       (float):        Value to use for padding when fewer than K objects exist.

    Outputs:
        torch.Tensor of shape (horizon, k * 2):
            Each row is [x1, y1, x2, y2, ..., xK, yK]. Missing entries filled with `fill_value`.
    """
    # 1) Filter objects located ahead (x > 0)
    forward_mask = object_positions[:, 0] > 0
    filtered_positions = object_positions[forward_mask]  # (M, 2)

    # 2) Compute squared distances to origin for sorting
    distances = (filtered_positions ** 2).sum(dim=1)     # (M,)

    # 3) Pad with fill_value and infinite distance if fewer than K objects
    num_filtered = filtered_positions.size(0)
    if num_filtered < k:
        pad_count = k - num_filtered
        pad_positions = torch.full(
            (pad_count, 2),
            fill_value,
            device=filtered_positions.device,
            dtype=filtered_positions.dtype
        )
        filtered_positions = torch.cat([filtered_positions, pad_positions], dim=0)  # (K, 2)

        pad_distances = torch.full(
            (pad_count,),
            float('inf'),
            device=distances.device,
            dtype=distances.dtype
        )
        distances = torch.cat([distances, pad_distances], dim=0)  # (K,)

    # 4) Select indices of the K smallest distances
    nearest_indices = torch.argsort(distances)[:k]        # (K,)

    # 5) Gather the K nearest positions
    nearest_positions = filtered_positions[nearest_indices]  # (K, 2)

    # 6) Flatten and replicate over the time horizon
    flattened_positions = nearest_positions.reshape(-1)    # (K*2,)
    static_trajectory = flattened_positions.unsqueeze(0).repeat(horizon, 1)  # (horizon, K*2)

    return static_trajectory

@torch.no_grad()
def FindNearestPedestriansInfo(
    predictions: torch.Tensor,             # (N_agents, T, 5)  columns: [x_center, y_center, yaw, vx, vy] in ego/vehicle frame
    agent_categories: torch.Tensor,        # (N_agents,)       integer categories; pedestrians == 2
    normal_speed_filtering_step: int = 20, # time index used for selection
    num_pedistrians: int = 5,              # desired number of pedestrians (legacy name preserved)
    large_value: float = 1e3,              # padding value for missing pedestrians
    filtering_step: int = 30,              # rear cutoff step (not used here)
) -> np.ndarray:
    """
    Input:
        predictions (torch.Tensor): (N_agents, T, 5) predicted states in ego/vehicle frame.
        agent_categories (torch.Tensor): (N_agents,) integer codes; pedestrians are category==2.
        agent_shapes (torch.Tensor): (N_agents, 2) sizes; not used.
        reference_trajectory (np.ndarray): (T, 4); not used (frame is already ego-centric).
        initial_state (np.ndarray): (4,); not used.
        normal_speed_filtering_step (int): selection time index.
        num_pedistrians (int): number of pedestrians to output (x,y per pedestrian).
        large_value (float): padding value when fewer pedestrians are available.

    Output:
        np.ndarray: shape (T, 2 * num_pedestrians) with columns
                    [ped1_x, ped1_y, ped2_x, ped2_y, ..., pedK_x, pedK_y].
                    When fewer than K are found, remaining columns are filled with `large_value`.

    Behavior:
        - Select only pedestrians (category==2).
        - Consider only those **in front** at the selection time, defined by x > 0
          in the ego/vehicle coordinate frame.
        - Among those, pick the nearest K by Euclidean distance at the selection time
          (distance = sqrt(x^2 + y^2), since ego is at origin in this frame).
    """
    # Resolve legacy typo for consistency in code
    num_pedestrians = int(num_pedistrians)

    # Convert tensors to numpy
    preds_np = predictions.detach().cpu().numpy()      # (N, T, 5)
    cats_np  = agent_categories.detach().cpu().numpy() # (N,)

    # Basic dims
    if preds_np.ndim != 3 or preds_np.shape[2] < 2:
        raise ValueError(f"`predictions` must be (N, T, 5) with at least x,y columns; got {preds_np.shape}")

    num_agents, num_time_steps, _ = preds_np.shape
    T = num_time_steps

    # Clamp the selection index to valid range
    cutoff_step = int(np.clip(normal_speed_filtering_step, 0, T - 1))

    # # [DEBUG] FindNearestPedestriansInfo category 확인
    # unique_p, counts_p = np.unique(cats_np, return_counts=True)
    # print(f"[DEBUG FindNearestPedestriansInfo] 수신된 cats_np unique={unique_p.tolist()}, counts={counts_p.tolist()}, total={len(cats_np)}")
    # print(f"[DEBUG FindNearestPedestriansInfo] ped_mask(==1) sum={int((cats_np==1).sum())}  ← planner PEDESTRIAN")
    # print(f"[DEBUG FindNearestPedestriansInfo] ped_mask(==2) sum={int((cats_np==2).sum())}  ← feature_builder PEDESTRIAN")
    # print(f"[DEBUG FindNearestPedestriansInfo] 현재 코드 pedestrian_mask(==1) → 맞으면 pedestrian이 잡힘")

    # 1) Select pedestrians only
    # class_pedestrian = 1 (assigned in pluto_planner_post_process_trajectory_evaluator.py)
    pedestrian_mask = (cats_np == 1)
    if pedestrian_mask.sum() == 0:
        # No pedestrians at all → return padding
        return np.full((T, 2 * num_pedestrians), large_value, dtype=np.float32)

    ped_preds = preds_np[pedestrian_mask]  # (N_ped, T, 5)

    # 2) Front selection by x > 0 at cutoff_step (ego/vehicle frame)
    x_cut = ped_preds[:, cutoff_step, 0]   # (N_ped,)
    y_cut = ped_preds[:, cutoff_step, 1]   # (N_ped,)
    front_mask = (x_cut > 0.0)

    if front_mask.sum() == 0:
        # No pedestrians in front → padding
        return np.full((T, 2 * num_pedestrians), large_value, dtype=np.float32)

    ped_front = ped_preds[front_mask]      # (N_front, T, 5)
    x_cut_f = ped_front[:, cutoff_step, 0]
    y_cut_f = ped_front[:, cutoff_step, 1]

    # 3) Nearest K by Euclidean distance at cutoff_step (ego at origin)
    dists = np.hypot(x_cut_f, y_cut_f)     # (N_front,)
    order = np.argsort(dists)[:num_pedestrians]
    sel = ped_front[order]                 # (K, T, 5)

    K = sel.shape[0]

    # 4) Build output (T, 2K) as [x1, y1, x2, y2, ...]
    xs = sel[:, :, 0]  # (K, T)
    ys = sel[:, :, 1]  # (K, T)

    out_cols = []
    for k in range(K):
        out_cols.append(xs[k])  # (T,)
        out_cols.append(ys[k])  # (T,)

    if len(out_cols) > 0:
        out_array = np.stack(out_cols, axis=1)  # (T, 2K)
    else:
        out_array = np.zeros((T, 0), dtype=np.float32)

    # 5) Pad to exactly (T, 2 * num_pedestrians)
    required_cols = 2 * num_pedestrians
    current_cols = out_array.shape[1]
    if current_cols < required_cols:
        pad_cols = required_cols - current_cols
        pad_array = np.full((T, pad_cols), large_value, dtype=np.float32)
        if current_cols == 0:
            out_array = pad_array
        else:
            out_array = np.hstack([out_array.astype(np.float32), pad_array])
    else:
        out_array = out_array.astype(np.float32)

    out_array[filtering_step:, :] = large_value  # Mask after filtering step

    return out_array


def GetLeftRightBoundaryPointsUsingSpace(map_data: Dict,
                                        centerline_idx = 0,
                                        leftline_idx   = 1,
                                        rightline_idx  = 2,
                                        x_point_idx = 0,
                                        y_point_idx = 1,
                                        x_point_upper_threshold = 40,
                                        x_point_lower_threshold = -10,
                                        y_point_lower_threshold = -20,
                                        y_point_upper_threshold = 20,
                                        x_interval = 2.0,
                                        use_smoothing = True,
                                        smoothing_window_size = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find left and right boundary points by scanning through x intervals.

    Inputs:
        use_smoothing (bool): Apply moving average smoothing to y coordinates (default=True).
        smoothing_window_size (int): Window size for moving average (default=3).

    Returns:
        left_boundary_points (np.ndarray): (K, 2) concatenated left boundary points
        right_boundary_points (np.ndarray): (K, 2) concatenated right boundary points
    """
     # Convert torch tensors to numpy arrays if needed
    point_position   = map_data["point_position"][0]   # (N, 3, 20, 2)
    polygon_on_route = map_data["polygon_on_route"][0] # (N,)
    polygon_type     = map_data["polygon_type"][0]     # (N,)
    polygon_position = map_data["polygon_position"][0] # (N, 2)

    # Ensure numpy arrays
    if hasattr(point_position, 'cpu'):
        point_position = point_position.cpu().numpy()
    if hasattr(polygon_on_route, 'cpu'):
        polygon_on_route = polygon_on_route.cpu().numpy()
    if hasattr(polygon_type, 'cpu'):
        polygon_type = polygon_type.cpu().numpy()
    if hasattr(polygon_position, 'cpu'):
        polygon_position = polygon_position.cpu().numpy()

    # (1) Filter by polygon_type (0 or 1) and on_route
    mask_polygon_type = (polygon_type == 0) | (polygon_type == 1)  # element-wise OR
    mask_polygon_on_route = (polygon_on_route == True)
    candidate_lane_mask = mask_polygon_type & mask_polygon_on_route  # (N,)

    if not np.any(candidate_lane_mask):
        # No candidate lanes found, return empty arrays
        return np.array([]), np.array([]), np.array([]), False, False, False, np.array([])

    candidates_point_position = point_position[candidate_lane_mask]  # (M, 3, 20, 2)
    candidates_polygon_position = polygon_position[candidate_lane_mask, :]  # (M, 2)
    total_centerline_points = candidates_point_position[:, centerline_idx, :, :]  # (M, 20, 2)
    total_leftline_points   = candidates_point_position[:, leftline_idx, :, :]      # (M, 20, 2)
    total_rightline_points  = candidates_point_position[:, rightline_idx, :, :]     # (M, 20, 2)

    # Flatten all points to (M*20, 2) for easier processing
    total_leftline_points_flat = total_leftline_points.reshape(-1, 2)   # (M*20, 2)
    total_rightline_points_flat = total_rightline_points.reshape(-1, 2)  # (M*20, 2)

    # Lists to collect boundary points
    left_boundary_points_list = []
    right_boundary_points_list = []

    # Scan through x intervals from x_point_lower_threshold to x_point_upper_threshold
    x_start = x_point_lower_threshold
    while x_start < x_point_upper_threshold:
        x_end = x_start + x_interval

        # --- Find LEFT boundary point in this x interval ---
        # Filter left points in current x range
        left_x = total_leftline_points_flat[:, x_point_idx]
        left_y = total_leftline_points_flat[:, y_point_idx]

        left_mask = (left_x >= x_start) & (left_x < x_end) & (left_y < y_point_upper_threshold) & (left_y >= 0)

        if np.any(left_mask):
            # Find point with maximum y value (leftmost)
            left_candidates = total_leftline_points_flat[left_mask]  # (K, 2)
            left_y_candidates = left_candidates[:, y_point_idx]
            leftmost_idx = np.argmax(left_y_candidates)
            leftmost_point = left_candidates[leftmost_idx]  # (2,)
            left_boundary_points_list.append(leftmost_point)

        # --- Find RIGHT boundary point in this x interval ---
        # Filter right points in current x range
        right_x = total_rightline_points_flat[:, x_point_idx]
        right_y = total_rightline_points_flat[:, y_point_idx]

        right_mask = (right_x >= x_start) & (right_x < x_end) & (right_y > y_point_lower_threshold) & (right_y <= 0)

        if np.any(right_mask):
            # Find point with minimum y value (rightmost)
            right_candidates = total_rightline_points_flat[right_mask]  # (K, 2)
            right_y_candidates = right_candidates[:, y_point_idx]
            rightmost_idx = np.argmin(right_y_candidates)
            rightmost_point = right_candidates[rightmost_idx]  # (2,)
            right_boundary_points_list.append(rightmost_point)

        # Move to next interval
        x_start += x_interval

    # Convert lists to numpy arrays
    if len(left_boundary_points_list) > 0:
        left_boundary_points = np.array(left_boundary_points_list)  # (K, 2)
    else:
        left_boundary_points = np.array([]).reshape(0, 2)

    if len(right_boundary_points_list) > 0:
        right_boundary_points = np.array(right_boundary_points_list)  # (K, 2)
    else:
        right_boundary_points = np.array([]).reshape(0, 2)

    # Apply moving average smoothing to y coordinates if requested
    if use_smoothing:
        from scipy.ndimage import uniform_filter1d

        # Smooth left boundary y coordinates
        if len(left_boundary_points) >= smoothing_window_size:
            left_boundary_points[:, y_point_idx] = uniform_filter1d(
                left_boundary_points[:, y_point_idx],
                size=smoothing_window_size,
                mode='nearest'
            )

        # Smooth right boundary y coordinates
        if len(right_boundary_points) >= smoothing_window_size:
            right_boundary_points[:, y_point_idx] = uniform_filter1d(
                right_boundary_points[:, y_point_idx],
                size=smoothing_window_size,
                mode='nearest'
            )

    # Default values for unused return fields
    is_have_to_use_front_lanes = False
    is_have_to_use_behind_lanes = False
    is_disconnected = False

    return left_boundary_points, right_boundary_points, candidates_point_position, is_have_to_use_front_lanes, is_have_to_use_behind_lanes, is_disconnected, candidates_polygon_position


def GetBoundaryFrontRearPointsByReferenceTrajectory(boundary_points: np.ndarray,
                                           reference_trajectory: np.ndarray,
                                           LeftorRight: str,
                                           rear_axle_to_front: float = 4.049,
                                           rear_axle_to_rear: float = 1.147,
                                           width: float = 2.297,
                                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get boundary y values at vehicle front and rear corner positions for each reference trajectory step.

    Inputs:
        boundary_points (np.ndarray): (M, 2) boundary points [x, y].
        reference_trajectory (np.ndarray): (N, 4) reference trajectory [x, y, yaw, v].
        LeftorRight (str): "Left" or "Right" to specify which side corners.
        rear_axle_to_front (float): Distance from rear axle to front of vehicle.
        rear_axle_to_rear (float): Distance from rear axle to rear of vehicle.
        width (float): Vehicle width.

    Returns:
        front_points (np.ndarray): (N, 2) array of [corner_x, y_lim] for front corners.
        rear_points (np.ndarray): (N, 2) array of [corner_x, y_lim] for rear corners.
    """
    from scipy.interpolate import interp1d

    if boundary_points.size == 0 or reference_trajectory.shape[0] == 0:
        N = reference_trajectory.shape[0]
        empty = np.zeros((N, 2), dtype=np.float64)
        return empty, empty

    # Extract reference trajectory components
    ref_x = reference_trajectory[:, 0]  # (N,) rear axle x
    ref_y = reference_trajectory[:, 1]  # (N,) rear axle y
    ref_yaw_orig = reference_trajectory[:, 2]  # (N,) original yaw (may be inaccurate)
    N = len(ref_x)

    # Recalculate yaw from x, y positions for better accuracy
    ref_yaw = np.zeros(N, dtype=np.float64)
    for i in range(N - 1):
        dx = ref_x[i + 1] - ref_x[i]
        dy = ref_y[i + 1] - ref_y[i]
        ref_yaw[i] = np.arctan2(dy, dx)

    # Last step: use previous yaw
    ref_yaw[-1] = ref_yaw[-2] if N > 1 else ref_yaw_orig[-1]

    # Calculate front and rear axle positions
    front_x = ref_x + rear_axle_to_front * np.cos(ref_yaw)  # (N,)
    front_y = ref_y + rear_axle_to_front * np.sin(ref_yaw)  # (N,)

    rear_x = ref_x - rear_axle_to_rear * np.cos(ref_yaw)  # (N,)
    rear_y = ref_y - rear_axle_to_rear * np.sin(ref_yaw)  # (N,)

    # Calculate corner positions based on Left or Right
    half_width = width / 2.0

    if LeftorRight == "Left":
        # Left corners: perpendicular offset in positive y direction (left)
        front_corner_x = front_x - half_width * np.sin(ref_yaw)  # (N,)
        front_corner_y = front_y + half_width * np.cos(ref_yaw)  # (N,)

        rear_corner_x = rear_x - half_width * np.sin(ref_yaw)  # (N,)
        rear_corner_y = rear_y + half_width * np.cos(ref_yaw)  # (N,)

    elif LeftorRight == "Right":
        # Right corners: perpendicular offset in negative y direction (right)
        front_corner_x = front_x + half_width * np.sin(ref_yaw)  # (N,)
        front_corner_y = front_y - half_width * np.cos(ref_yaw)  # (N,)

        rear_corner_x = rear_x + half_width * np.sin(ref_yaw)  # (N,)
        rear_corner_y = rear_y - half_width * np.cos(ref_yaw)  # (N,)

    else:
        raise ValueError(f"LeftorRight must be 'Left' or 'Right', got '{LeftorRight}'")

    # Sort boundary points by x for interpolation
    boundary_x = boundary_points[:, 0]  # (M,)
    boundary_y = boundary_points[:, 1]  # (M,)

    # Filter boundary points with x > 0 and sort
    valid_mask = boundary_x > -10
    if not np.any(valid_mask):
        # No valid boundary points, return geometric corners as fallback
        front_points = np.column_stack([front_corner_x, front_corner_y]).astype(np.float64)
        rear_points = np.column_stack([rear_corner_x, rear_corner_y]).astype(np.float64)
        return front_points, rear_points

    boundary_x_valid = boundary_x[valid_mask]
    boundary_y_valid = boundary_y[valid_mask]

    # Before interpolation, remove duplicate x values
    sort_indices = np.argsort(boundary_x_valid)
    boundary_x_sorted = boundary_x_valid[sort_indices]
    boundary_y_sorted = boundary_y_valid[sort_indices]

    # Remove duplicates: keep first occurrence of each unique x
    unique_mask = np.concatenate(([True], np.diff(boundary_x_sorted) > 1e-6))
    boundary_x_unique = boundary_x_sorted[unique_mask]
    boundary_y_unique = boundary_y_sorted[unique_mask]

    # Need at least 2 points for interpolation
    if len(boundary_x_unique) < 2:
        # Fallback: return geometric corners
        front_points = np.column_stack([front_corner_x, front_corner_y]).astype(np.float64)
        rear_points = np.column_stack([rear_corner_x, rear_corner_y]).astype(np.float64)
        return front_points, rear_points

    # Create interpolation function
    interp_func = interp1d(
        boundary_x_unique,
        boundary_y_unique,
        kind='linear',
        fill_value='extrapolate',
        bounds_error=False
    )

    # Interpolate boundary y values at front and rear corner x positions
    front_y_lim = interp_func(front_corner_x)  # (N,)
    rear_y_lim = interp_func(rear_corner_x)  # (N,)

    # Return as two (N, 2) arrays: [corner_x, y_lim] for front and rear
    front_points = np.column_stack([front_corner_x, front_y_lim]).astype(np.float64)  # (N, 2)
    rear_points = np.column_stack([rear_corner_x, rear_y_lim]).astype(np.float64)  # (N, 2)

    return front_points, rear_points


@torch.no_grad()
def TrajectoryVelocityInfoGeneratorTorch(
    traj_tensor: torch.Tensor,
    dt: float = 0.1,
    init_state: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """
    Compute [x, y, yaw, v] using forward differences with last step backward difference.
    init_state is retained but unused in velocity calculation.
    """
    # Copy tensor to numpy once
    traj_array = traj_tensor.detach().cpu().numpy()  # shape (T,3)

    # Compute forward differences for indices 0 to T-2
    deltas = traj_array[1:, :2] - traj_array[:-1, :2]
    velocities = np.empty(traj_array.shape[0], dtype=np.float32)
    velocities[:-1] = np.linalg.norm(deltas, axis=1) / dt

    # Compute backward difference for last index
    last_delta = traj_array[-1, :2] - traj_array[-2, :2]
    velocities[-1] = np.linalg.norm(last_delta) / dt

    # Assemble output [x, y, yaw, v]
    return np.hstack((traj_array, velocities[:, None])).astype(np.float32)

def TrajectoryVelocityInfoGeneratorNumpy(
    traj_array: np.ndarray,
    dt: float = 0.1,
    init_state: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """
    Compute [x, y, yaw, v] using forward differences with last step backward difference.
    init_state is retained but unused in velocity calculation.
    """
    # Forward differences for velocities
    deltas = traj_array[1:, :2] - traj_array[:-1, :2]
    velocities = np.empty(traj_array.shape[0], dtype=np.float32)
    velocities[:-1] = np.linalg.norm(deltas, axis=1) / dt

    # Backward difference for last velocity
    last_delta = traj_array[-1, :2] - traj_array[-2, :2]
    velocities[-1] = np.linalg.norm(last_delta) / dt

    # Assemble output [x, y, yaw, v]
    return np.hstack((traj_array, velocities[:, None])).astype(np.float32)


def TrajectoryChecker(
    reference_trajectory: np.ndarray,
    initial_state: np.ndarray,
    x_threshold: float = 0.005,
    vel_threshold: float = 0.001
) -> np.ndarray:
    """
    Zero out positions and velocity where x-position is below the x_threshold.
    """
    # Copy input trajectory
    modified_traj = reference_trajectory.copy()

    # Identify points where x ≤ threshold
    low_x_mask = modified_traj[:, 0] <= x_threshold

    # Zero out x, y, and v for those points
    modified_traj[low_x_mask, 0] = 0.0  # x position
    modified_traj[low_x_mask, 1] = 0.0  # y position
    # modified_traj[low_x_mask, 3] = 0.0  # velocity

    return modified_traj


def ConvertPrevSolutionToCurrentTrajectory(
    prev_mpc_solution: np.ndarray,
) -> np.ndarray:
    """
    Convert previous MPC solution to current reference frame.

    Process:
    1. Take first point of prev_mpc_solution as new origin
    2. Transform all points to be relative to this new origin (translation + rotation)
    3. Remove first point (now at origin)
    4. Duplicate last point to maintain 80 steps

    Inputs:
        prev_mpc_solution (np.ndarray): (80, 6) [x, y, yaw, v, ax, delta]

    Returns:
        converted_trajectory (np.ndarray): (80, 4) [x, y, yaw, v]
    """
    if prev_mpc_solution.shape[0] < 2:
        raise ValueError("prev_mpc_solution must have at least 2 points")

    # Extract first point as new origin
    x0 = prev_mpc_solution[0, 0]  # x of first point
    y0 = prev_mpc_solution[0, 1]  # y of first point
    yaw0 = prev_mpc_solution[0, 2]  # yaw of first point

    # Extract all points
    x = prev_mpc_solution[:, 0]  # (80,)
    y = prev_mpc_solution[:, 1]  # (80,)
    yaw = prev_mpc_solution[:, 2]  # (80,)
    v = prev_mpc_solution[:, 3]  # (80,)

    # Translation: move origin to first point
    dx = x - x0  # (80,)
    dy = y - y0  # (80,)

    # Rotation: rotate by -yaw0 to align first point's heading with x-axis
    cos_yaw0 = np.cos(-yaw0)
    sin_yaw0 = np.sin(-yaw0)

    x_transformed = dx * cos_yaw0 - dy * sin_yaw0  # (80,)
    y_transformed = dx * sin_yaw0 + dy * cos_yaw0  # (80,)
    yaw_transformed = yaw - yaw0  # (80,)

    # Normalize yaw to [-pi, pi]
    yaw_transformed = (yaw_transformed + np.pi) % (2 * np.pi) - np.pi

    # Stack into (80, 4) array
    transformed_traj = np.column_stack([
        x_transformed,
        y_transformed,
        yaw_transformed,
        v
    ])  # (80, 4)

    # Remove first point (now at origin)
    trajectory_shifted = transformed_traj[1:, :]  # (79, 4)

    # Duplicate last point to maintain 80 steps
    last_point = trajectory_shifted[-1:, :]  # (1, 4)
    converted_trajectory = np.vstack([trajectory_shifted, last_point])  # (80, 4)

    return converted_trajectory.astype(np.float32)


def TrajectoryVelocityInfoGeneratorNumpy(traj_t: np.ndarray, dt: float = 0.1,
                                         init_state=(0.,0.,0.)) -> np.ndarray:
    """
    Convert trajectory tensor/array to numpy with velocities appended.

    This function accepts a trajectory array (or tensor converted to numpy)
    of shape (T, 3) representing [x, y, yaw] and returns a numpy array
    of shape (T, 4) with velocities computed by forward differences
    (last step uses a backward/last-difference approximation).

    Performance notes:
    - Copies data only once when converting from a GPU tensor to CPU numpy.
    - Uses NumPy vectorized operations for speed.
    - Typical timing: ~0.25–0.30 ms per call (device-to-CPU copy ~0.2–0.25 ms; NumPy ops ~0.05 ms).
    """
    # 1) Use input as-is (if input is a tensor, caller should convert to numpy beforehand)
    traj = traj_t

    # 2) Vectorized NumPy operations for velocity calculation
    # total has one extra row (init_state) so that diff yields T rows for T timesteps
    total = np.vstack((init_state, traj))      # (T+1, 3)
    dxdy  = np.diff(total[:, :2], axis=0)      # (T, 2)
    v     = np.linalg.norm(dxdy, axis=1) / dt  # (T,)

    # Avoid a zero/undefined first velocity by copying the second sample
    v[0] = v[1]

    out = np.hstack((traj, v[:, None])).astype(np.float32)
    return out
