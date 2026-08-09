import math
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Polygon, Rectangle, Circle
from matplotlib.colors import ListedColormap


def get_polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    """Get the arc length of each point in a polyline"""
    diff = xy[1:] - xy[:-1]
    displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    arc_length = np.cumsum(displacement)
    return np.concatenate((np.zeros(1), arc_length), axis=0)


def interpolate_centerline(xy: np.ndarray, n_points: int):
    arc_length = get_polyline_arc_length(xy)
    steps = np.linspace(0, arc_length[-1], n_points)
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: np.ndarray,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
    alpha=1.0,
    label=None,
    zorder=50,
    fill=True,
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        xy=(pivot_x, pivot_y),
        width=bbox_length,
        height=bbox_width,
        angle=np.degrees(heading),
        fc=color if fill else "none",
        ec=color,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(vehicle_bounding_box)

    if bbox_length > 1.0:
        direction = (
            0.25 * bbox_size[0] * np.array([math.cos(heading), math.sin(heading)])
        )
        ax.arrow(
            cur_location[0],
            cur_location[1],
            direction[0],
            direction[1],
            color="white",
            zorder=zorder + 1,
            head_width=0.5,
        )


def plot_box(
    ax: plt.Axes,
    cur_location: np.ndarray,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
    alpha=1.0,
    label=None,
    zorder=50,
    fill=True,
    **kwargs,
) -> None:
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        xy=(pivot_x, pivot_y),
        width=bbox_length,
        height=bbox_width,
        angle=np.degrees(heading),
        fc=color if fill else "none",
        ec="dimgrey",
        alpha=alpha,
        label=label,
        zorder=zorder,
        **kwargs,
    )
    ax.add_patch(vehicle_bounding_box)


def plot_polygon(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    alpha=1.0,
    zorder=50,
    label=None,
) -> None:
    ax.add_patch(
        Polygon(
            np.stack([x, y], axis=1),
            closed=True,
            fc=color,
            ec="dimgrey",
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
    )


def plot_polyline(
    ax,
    polylines: List[np.ndarray],
    cmap="spring",
    linewidth=3,
    arrow: bool = True,
    reverse: bool = False,
    alpha=0.5,
    zorder=100,
    color_change: bool = True,
    color=None,
    linestyle="-",
    label=None,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    if isinstance(color, str):
        color = [color] * len(polylines)

    for i, polyline in enumerate(polylines):
        inter_poly = interpolate_centerline(polyline, 50)
        # inter_poly = polyline[...,:2]

        if arrow:
            point = inter_poly[-1]
            diff = inter_poly[-1] - inter_poly[-2]
            diff = diff / np.linalg.norm(diff)
            if color_change:
                c = plt.cm.get_cmap(cmap)(0)
            else:
                c = color[i]
            arrow = ax.quiver(
                point[0],
                point[1],
                diff[0],
                diff[1],
                alpha=alpha,
                scale_units="xy",
                scale=0.25,
                minlength=0.5,
                zorder=zorder - 1,
                color=c,
            )

        if color_change:
            arc = get_polyline_arc_length(inter_poly)
            polyline = inter_poly.reshape(-1, 1, 2)
            segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
            norm = plt.Normalize(arc.min(), arc.max())
            lc = LineCollection(
                segment, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha, label=label
            )
            lc.set_array(arc if not reverse else arc[::-1])
            lc.set_linewidth(linewidth)
            ax.add_collection(lc)
        else:
            ax.plot(
                inter_poly[:, 0],
                inter_poly[:, 1],
                color=color[i],
                linewidth=linewidth,
                zorder=zorder,
                alpha=alpha,
                linestyle=linestyle,
                label=label,
            )


def plot_direction(ax, anchors, dir_vecs, zorder=1):
    for anchor, dir_vec in zip(anchors, dir_vecs):
        if np.linalg.norm(dir_vec) == 0:
            continue
        vec = dir_vec / np.linalg.norm(dir_vec)
        ax.arrow(
            anchor[0],
            anchor[1],
            vec[0],
            vec[1],
            color="black",
            zorder=zorder,
            head_width=0.2,
            head_length=0.2,
        )


def plot_trajectory_with_angle(ax, traj):
    if traj.shape[-1] > 3:
        angle_phase_num = traj.shape[-1] - 2
        phase = 2 * np.pi * np.arange(angle_phase_num) / angle_phase_num
        xn = traj[..., -3:]  # (N, 3)
        angles = -np.arctan2(
            np.sum(np.sin(phase) * xn, axis=-1), np.sum(np.cos(phase) * xn, axis=-1)
        )
    else:
        angles = traj[..., -1]

    ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=2)
    for p, angle in zip(traj, angles):
        ax.arrow(
            p[0],
            p[1],
            np.cos(angle) * 0.5,
            np.sin(angle) * 0.5,
            color="black",
            zorder=1,
            head_width=0.3,
            head_length=0.2,
        )
    ax.axis("equal")


def plot_crosswalk(ax, edge1, edge2):
    polygon = np.concatenate([edge1, edge2[::-1]])
    ax.add_patch(
        Polygon(
            polygon, closed=True, fc="k", alpha=0.3, hatch="///", ec="w", linewidth=2
        )
    )


def plot_sdc(
    ax,
    center,
    heading,
    width,
    length,
    steer=0.0,
    color="pink",
    fill=True,
    wheel=True,
    **kwargs,
):
    vec_heading = np.array([np.cos(heading), np.sin(heading)])
    vec_tan = np.array([np.sin(heading), -np.cos(heading)])

    front_left_wheel = center + 1.419 * vec_heading + 0.35 * width * vec_tan
    front_right_wheel = center + 1.419 * vec_heading - 0.35 * width * vec_tan
    wheel_heading = heading + steer
    wheel_size = (0.8, 0.3)

    plot_box(
        ax, center, heading, color=color, fill=fill, bbox_size=(length, width), **kwargs
    )

    if wheel:
        plot_box(
            ax,
            front_left_wheel,
            wheel_heading,
            color="k",
            fill=True,
            bbox_size=wheel_size,
            **kwargs,
        )
        plot_box(
            ax,
            front_right_wheel,
            wheel_heading,
            color="k",
            fill=True,
            bbox_size=wheel_size,
            **kwargs,
        )


def plot_lane_area(ax, left_bound, right_bound, fc="silver", alpha=1.0, ec=None):
    polygon = np.concatenate([left_bound, right_bound[::-1]])
    ax.add_patch(
        Polygon(polygon, closed=True, fc=fc, alpha=alpha, ec=None, linewidth=2)
    )


def plot_cov_ellipse(logstd, rho, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    # compute covariance matrix from logstd and rho
    std = np.exp(logstd)
    cov = np.array(
        [[std[0] ** 2, rho * std[0] * std[1]], [rho * std[0] * std[1], std[1] ** 2]]
    )

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def visualize_pluto_scene(data: Dict[str, Any], batch_idx: int = 0, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None):
    """
    Visualize a scene from Pluto model's forward input data at specified batch index.
    
    Args:
        data: Dictionary containing the input data to pluto_model.forward()
              Expected keys: 'agent', 'map', 'static_objects', 'reference_line'
        batch_idx: Index of the batch to visualize (default: 0)
        figsize: Figure size tuple (width, height)
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure object
    """
    # Check if batch_idx is valid
    batch_size = data['agent']['position'].shape[0]
    if batch_idx >= batch_size:
        raise ValueError(f"batch_idx {batch_idx} is out of range. Batch size: {batch_size}")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Extract data for the specified batch
    agent_data = {k: v[batch_idx] for k, v in data['agent'].items()}
    map_data = {k: v[batch_idx] for k, v in data['map'].items()}
    static_data = {k: v[batch_idx] for k, v in data['static_objects'].items()}
    ref_data = {k: v[batch_idx] for k, v in data['reference_line'].items()}
    
    # Convert to numpy if tensors
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x
    
    agent_data = {k: to_numpy(v) for k, v in agent_data.items()}
    map_data = {k: to_numpy(v) for k, v in map_data.items()}
    static_data = {k: to_numpy(v) for k, v in static_data.items()}
    ref_data = {k: to_numpy(v) for k, v in ref_data.items()}
    
    # 1. Draw map elements (from point_position with 4D structure)
    if 'point_position' in map_data and map_data['point_position'].shape[0] > 0:
        point_pos = map_data['point_position']  # [segment, [center, left, right], points(20), xy]
        point_mask = map_data.get('point_valid_mask', None)  # Same structure as point_pos or simplified
        
        num_segments, num_lanes, num_points, xy = point_pos.shape
        
        # Colors and styles for different lane elements
        # 0=center (driving path), 1=left boundary, 2=right boundary
        lane_colors = ['blue', 'gray', 'gray']  # center=blue (driving path), boundaries=gray
        lane_labels = ['Driving path', 'Lane boundary', None]  # Only label first boundary
        lane_linewidths = [0.5, 1, 1]  # Make center path even thinner (half of current)
        lane_linestyles = ['-', '-', '-']  # All solid lines
        lane_alphas = [0.5, 0.5, 0.5]  # Lower transparency for all lanes
        
        for segment_idx in range(num_segments):
            for lane_idx in range(num_lanes):  # 0=center, 1=left, 2=right
                lane_points = point_pos[segment_idx, lane_idx]  # [points(20), xy]
                
                # Check mask if available
                if point_mask is not None:
                    if len(point_mask.shape) == 4:  # Same structure as point_pos
                        if not point_mask[segment_idx, lane_idx].any():
                            continue
                        valid_points = lane_points[point_mask[segment_idx, lane_idx]]
                    elif len(point_mask.shape) == 2:  # [segment, lane] level mask
                        if not point_mask[segment_idx, lane_idx]:
                            continue
                        valid_points = lane_points
                    else:
                        valid_points = lane_points
                else:
                    valid_points = lane_points
                
                # Filter out invalid coordinates (NaN, inf, or very large values)
                if len(valid_points) > 0:
                    finite_mask = np.isfinite(valid_points).all(axis=1)
                    valid_points = valid_points[finite_mask]
                
                # Draw lane element if we have enough valid points
                if len(valid_points) > 1:
                    color = lane_colors[lane_idx] if lane_idx < len(lane_colors) else 'gray'
                    linewidth = lane_linewidths[lane_idx] if lane_idx < len(lane_linewidths) else 1
                    linestyle = lane_linestyles[lane_idx] if lane_idx < len(lane_linestyles) else '-'
                    alpha = lane_alphas[lane_idx] if lane_idx < len(lane_alphas) else 0.6
                    
                    # Add label only for the first occurrence of each type
                    label = None
                    if segment_idx == 0:
                        if lane_idx == 0:  # Center (driving path)
                            label = lane_labels[0]
                        elif lane_idx == 1:  # First boundary
                            label = lane_labels[1]
                    
                    ax.plot(valid_points[:, 0], valid_points[:, 1], 
                           color=color, linewidth=linewidth, linestyle=linestyle, 
                           alpha=alpha, label=label)

    # # 2. Draw map polygon elements (if available)
    # if 'polygon_position' in map_data and map_data['polygon_position'].shape[0] > 0:
    #     polygon_pos = map_data['polygon_position']  # Original polygon data
    #     polygon_mask = map_data.get('valid_mask', None)
        
    #     # Handle polygon data as before (for road surfaces, etc.)
    #     for i, poly_points in enumerate(polygon_pos):
    #         # Check if this polygon should be skipped
    #         if polygon_mask is not None:
    #             if len(polygon_mask.shape) == 1:  # [num_polygons] - polygon level mask
    #                 if not polygon_mask[i]:
    #                     continue
    #                 valid_points = poly_points
    #             elif len(polygon_mask.shape) == 2:  # [num_polygons, num_points] - point level mask
    #                 if not polygon_mask[i].any():
    #                     continue
    #                 valid_points = poly_points[polygon_mask[i]]
    #             else:
    #                 valid_points = poly_points
    #         else:
    #             valid_points = poly_points
                
    #         # Filter out invalid coordinates
    #         if len(valid_points) > 0:
    #             finite_mask = np.isfinite(valid_points).all(axis=1)
    #             valid_points = valid_points[finite_mask]
                
    #         if len(valid_points) > 2:  # Need at least 3 points for polygon
    #             polygon = Polygon(valid_points, fill=False, edgecolor='lightgray', alpha=0.4, linewidth=0.5)
    #             ax.add_patch(polygon)
    
    # 3. Draw static objects
    if 'position' in static_data and static_data['position'].shape[0] > 0:
        static_pos = static_data['position']  # [num_static, 2]
        static_heading = static_data.get('heading', None)
        static_mask = static_data.get('valid_mask', None)
        
        for i, pos in enumerate(static_pos):
            if static_mask is not None and not static_mask[i]:
                continue
                
            # Check if position is valid
            if not np.isfinite(pos).all():
                continue
                
            # Draw as small rectangles or circles
            if static_heading is not None and i < len(static_heading):
                heading = static_heading[i]
                if np.isfinite(heading):
                    # Draw as oriented rectangle
                    cos_h, sin_h = np.cos(heading), np.sin(heading)
                    width, height = 0.5, 0.2  # Approximate vehicle size in normalized coords
                    
                    # Rectangle corners relative to center
                    corners = np.array([[-width/2, -height/2], [width/2, -height/2], 
                                      [width/2, height/2], [-width/2, height/2]])
                    
                    # Rotate and translate
                    rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
                    rotated_corners = corners @ rot_matrix.T + pos
                    
                    rect = Polygon(rotated_corners, fill=True, facecolor='orange', 
                                 edgecolor='darkorange', alpha=0.8)
                    ax.add_patch(rect)
                else:
                    # Draw as circle if heading is invalid
                    circle = Circle(pos, 0.3, fill=True, facecolor='orange', 
                                  edgecolor='darkorange', alpha=0.8)
                    ax.add_patch(circle)
            else:
                # Draw as circle
                circle = Circle(pos, 0.3, fill=True, facecolor='orange', 
                              edgecolor='darkorange', alpha=0.8)
                ax.add_patch(circle)
    
    # 4. Draw agents with past/present/future distinction
    if 'position' in agent_data and agent_data['position'].shape[0] > 0:
        agent_pos = agent_data['position']  # [num_agents, 101_steps, 2]
        agent_heading = agent_data.get('heading', None)  # [num_agents, 101_steps]
        agent_shape = agent_data.get('shape', None)  # [num_agents, 2] (length, width)
        agent_mask = agent_data.get('valid_mask', None)  # [num_agents, 101_steps]
        
        # Time step definitions: 0-19(past), 20(present), 21-100(future)
        past_steps = slice(0, 20)
        present_step = 20
        future_steps = slice(21, 101)
        
        for i in range(agent_pos.shape[0]):
            traj = agent_pos[i]  # [101_steps, 2]
            
            # Check if agent is valid
            if agent_mask is not None:
                if len(agent_mask.shape) == 2:  # [num_agents, steps]
                    if not agent_mask[i].any():
                        continue
                    valid_mask = agent_mask[i]
                elif len(agent_mask.shape) == 1:  # [num_agents]
                    if not agent_mask[i]:
                        continue
                    valid_mask = np.ones(traj.shape[0], dtype=bool)
                else:
                    valid_mask = np.ones(traj.shape[0], dtype=bool)
            else:
                valid_mask = np.ones(traj.shape[0], dtype=bool)
            
            # Filter out invalid coordinates
            finite_mask = np.isfinite(traj).all(axis=1)
            combined_mask = valid_mask & finite_mask
            
            if not combined_mask.any():
                continue
            
            # Define colors for ego vs other agents
            if i == 0:  # Ego vehicle
                past_color = 'darkgreen'
                present_color = 'green' 
                future_color = 'lightgreen'
                box_color = 'green'
                traj_alpha = 0.9
                linewidth = 2
                agent_label = 'Ego'
            else:  # Other agents
                past_color = 'darkred'
                present_color = 'red'
                future_color = 'lightcoral'
                box_color = 'red'
                traj_alpha = 0.7
                linewidth = 1
                agent_label = 'Agent' if i == 1 else None  # Label only first other agent
            
            # Draw past trajectory (0-19)
            past_mask = combined_mask[past_steps]
            if past_mask.any():
                past_traj = traj[past_steps][past_mask]
                if len(past_traj) > 1:
                    ax.plot(past_traj[:, 0], past_traj[:, 1], 
                           color=past_color, linewidth=linewidth, alpha=traj_alpha,
                           label=f'{agent_label} (past)' if agent_label and i <= 1 else None)
            
            # Draw future trajectory (21-100)
            future_mask = combined_mask[future_steps]
            if future_mask.any():
                future_traj = traj[future_steps][future_mask]
                if len(future_traj) > 1:
                    ax.plot(future_traj[:, 0], future_traj[:, 1], 
                           color=future_color, linewidth=linewidth, alpha=traj_alpha, linestyle='--',
                           label=f'{agent_label} (future)' if agent_label and i <= 1 else None)
            
            # Draw present position and bounding box (step 20)
            if combined_mask[present_step]:
                present_pos = traj[present_step]
                
                # Draw present position
                ax.scatter(present_pos[0], present_pos[1], 
                          c=present_color, s=13, marker='o', alpha=0.9, zorder=10,
                          label=f'{agent_label} (present)' if agent_label and i <= 1 else None)
                
                # Draw heading arrow
                if agent_heading is not None and agent_shape is not None:
                    if i < len(agent_shape) and np.isfinite(agent_heading[i, present_step]):
                        heading = agent_heading[i, present_step]
                        width, length = agent_shape[i, 20, :]
                        
                        # Draw arrow in heading direction (0.65 * length)
                        arrow_length = length * 0.65
                        dx = arrow_length * np.cos(heading)
                        dy = arrow_length * np.sin(heading)
                        
                        ax.arrow(present_pos[0], present_pos[1], dx, dy,
                               head_width=width*0.3, head_length=length*0.2, 
                               fc=present_color, ec=present_color, alpha=0.8, zorder=11)
                
                # Draw bounding box at present position
                if agent_heading is not None and agent_shape is not None:
                    if i < len(agent_shape) and np.isfinite(agent_heading[i, present_step]):
                        heading = agent_heading[i, present_step]
                        width, length = agent_shape[i, 20, :]
                        
                        # Create bounding box corners (relative to center)
                        half_length, half_width = length / 2, width / 2
                        corners = np.array([
                            [-half_length, -half_width],  # rear left
                            [half_length, -half_width],   # front left  
                            [half_length, half_width],    # front right
                            [-half_length, half_width]    # rear right
                        ])
                        
                        # Rotate corners according to heading
                        cos_h, sin_h = np.cos(heading), np.sin(heading)
                        rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
                        rotated_corners = corners @ rot_matrix.T
                        
                        # Translate to agent position
                        box_corners = rotated_corners + present_pos
                        
                        # Draw bounding box
                        bbox = Polygon(box_corners, fill=False, edgecolor=box_color, 
                                     linewidth=1, alpha=0.9, zorder=9)
                        ax.add_patch(bbox)
                else:
                    # Fallback: draw simple circle if no heading/shape info
                    circle = Circle(present_pos, 0.5, fill=False, edgecolor=box_color, 
                                  linewidth=1, alpha=0.9, zorder=9)
                    ax.add_patch(circle)
    
    # 5. Draw reference line
    if 'position' in ref_data and ref_data['position'].shape[0] > 0:
        ref_pos = ref_data['position']  # [num_ref_points, 2]
        ref_mask = ref_data.get('valid_mask', None)
        
        if ref_mask is not None:
            valid_ref_points = ref_pos[ref_mask]
        else:
            valid_ref_points = ref_pos
            
        # Filter out invalid coordinates
        if len(valid_ref_points) > 0:
            finite_mask = np.isfinite(valid_ref_points).all(axis=1)
            valid_ref_points = valid_ref_points[finite_mask]
            
        if len(valid_ref_points) > 1:
            ax.plot(valid_ref_points[:, 0], valid_ref_points[:, 1], 
                   'purple', linewidth=1, linestyle='--', alpha=0.8, label='Reference line')
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (normalized local coordinates)')
    ax.set_ylabel('Y (normalized local coordinates)')
    ax.set_title(f'Pluto Scene Visualization (Batch {batch_idx})')
    
    # Add legend
    ax.legend()
    
    # Set reasonable axis limits based on data
    all_positions = []
    if 'position' in agent_data and agent_data['position'].shape[0] > 0:
        all_positions.append(agent_data['position'].reshape(-1, 2))
    if 'polygon_position' in map_data and map_data['polygon_position'].shape[0] > 0:
        all_positions.append(map_data['polygon_position'].reshape(-1, 2))
    if 'point_position' in map_data and map_data['point_position'].shape[0] > 0:
        all_positions.append(map_data['point_position'].reshape(-1, 2))
    
    if all_positions:
        all_pos = np.concatenate(all_positions, axis=0)
        # Filter out invalid positions (e.g., very large values)
        valid_mask = np.all(np.abs(all_pos) < 1000, axis=1)
        if valid_mask.any():
            valid_pos = all_pos[valid_mask]
            margin = 2.0
            ax.set_xlim(valid_pos[:, 0].min() - margin, valid_pos[:, 0].max() + margin)
            ax.set_ylim(valid_pos[:, 1].min() - margin, valid_pos[:, 1].max() + margin)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    data = torch.load("src/utils/pluto_input.pt")
    visualize_pluto_scene(data, batch_idx=0, figsize=(12, 8), save_path="src/utils/pluto_input.png")