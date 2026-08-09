# module_safetycritical_acados/algorithm/mpc_ego_solver.py

import numpy as np
import os
import time
import yaml
from dataclasses import dataclass
from typing import Tuple

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import SX, vertcat, cos, sin, tan, fmax, sqrt, tanh

from ..interface.python import (
    AUTO_VehicleState, AUTO_Trajectory, AUTO_TrajectoryPoint,
    AUTO_Objects, AUTO_Object, AUTO_Pose, AUTO_Motion, AUTO_Gnss,
    AUTO_Lanes
)


@dataclass
class VehiclePositions:
    """Vehicle position points (rear axle, center, front axle)"""
    rear_x: SX
    rear_y: SX
    center_x: SX
    center_y: SX
    front_x: SX
    front_y: SX


@dataclass
class AgentTrajectory:
    """Agent trajectory information with 3 trajectory points and width"""
    x1: SX
    y1: SX
    x2: SX
    y2: SX
    x3: SX
    y3: SX
    width: SX


@dataclass
class PedestrianPosition:
    """Pedestrian position information"""
    x: SX
    y: SX


class MPCConstants:
    """MPC related all constants"""
    # Environment related constants
    AGENT_COUNT = 10
    TRAJECTORY_POINTS_PER_AGENT = 3
    PEDESTRIAN_COUNT = 5
    VEHICLE_COLLISION_POINTS = 3  # rear, center, front

    # Number of constraints
    COMFORT_CONSTRAINTS = 4
    ACCELERATION_CONSTRAINTS = 12
    AGENT_CONSTRAINTS = 90  # 10 agents * 9 constraints each
    PEDESTRIAN_CONSTRAINTS = 15  # 5 pedestrians * 3 constraints each
    TOTAL_NONLINEAR_CONSTRAINTS = COMFORT_CONSTRAINTS + AGENT_CONSTRAINTS + PEDESTRIAN_CONSTRAINTS  # 109
    BOUNDARY_CONSTRAINTS = 4  # left boundary (front, rear, mid) + right boundary (front, rear, mid)

    # Parameter index definition
    class ParameterIndices:
        # Reference trajectory (4 values: x, y, yaw, vel)
        REF_TRAJECTORY_START = 0
        REF_TRAJECTORY_END = 3

        # Agent trajectory data (60 values: 10 agents * 6 values each)
        AGENT_DATA_START = 4
        AGENT_DATA_END = 63

        # Pedestrian positions (10 values: 5 pedestrians * 2 values each)
        PEDESTRIAN_START = 64
        PEDESTRIAN_END = 73

        # Agent widths (10 values)
        AGENT_WIDTHS_START = 74
        AGENT_WIDTHS_END = 83

        # Boundary limits (4 values: left front, left rear, right front, right rear)
        BOUNDARY_LIMITS_START = 84
        BOUNDARY_LIMITS_END = 88

    # Vehicle physical constants
    class VehicleDefaults:
        WHEELBASE = 3.089
        WIDTH = 2.297
        LENGTH = 5.176
        REAR_AXLE_TO_FRONT = 4.049
        REAR_AXLE_TO_REAR  = 1.127

    # Soft constraint weights
    class DefaultWeights:
        LONGITUDINAL_JERK = 3.5
        LONGITUDINAL_ACCEL = 3.5
        COMFORT = 3.5
        COLLISION = 4.0

    # Numerical constants
    class Numerical:
        LARGE_POSITIVE = 1e4    # value <= 1e10
        LARGE_NEGATIVE = -1e4   # value >= -1e10
        SMALL_EPSILON = 1e-1
        LARGE_POSITIVE_FOR_CALC = 1e3
        LARGE_NEGATIVE_FOR_CALC = -1e3

@dataclass
class VehicleAcceleration:
    ax: SX
    ay: SX

class RefinementMPC:
    def __init__(self, yaml_path=None):
        """
        Initialize RefinementMPC
        :param T: Number of prediction steps
        :param dt: Time interval between prediction steps [sec]
        :param wheelbase: Wheel base of vehicle [m]
        """
        self.initialized = False

        # Define the YAML file path relative to the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if yaml_path == None:
            yaml_path   = os.path.join(current_dir, 'refinement_mpc_params.yaml')

        # Defensive check: crash if YAML config is missing
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")

        # YAML Config File Load
        with open(yaml_path, 'r') as file_handle:
            params = yaml.safe_load(file_handle)

        # Get Vehicle Parameters
        self.width                       = params["width"]
        self.width_half                  = self.width / 2.0
        self.wheel_base                  = params["wheel_base"]
        self.rear_axle_to_front          = params["rear_axle_to_front"]
        self.rear_axle_to_rear           = params["rear_axle_to_rear"]
        self.max_delta                   = params["max_delta"]
        self.max_ddelta                  = params["max_ddelta"]

        self.dt                          = params["dt"]
        self.T                           = params["T"]
        self.horizon                     = self.T * self.dt

        # Get constraint data
        self.safety_margin               = params["safety_margin"]
        self.boundary_margin             = params["boundary_margin"]
        self.boundary_lim_time           = params["boundary_lim_time"]
        self.safety_distance_pedestrian  = params["safety_distance_pedestrian"]

        # Nuplan Comfort Metric
        self.nuplan_max_abs_mag_jerk            = params["nuplan_max_abs_mag_jerk"]
        self.nuplan_max_abs_lat_accel           = params["nuplan_max_abs_lat_accel"]
        self.nuplan_max_lon_accel               = params["nuplan_max_lon_accel"]
        self.nuplan_min_lon_accel               = params["nuplan_min_lon_accel"]
        self.nuplan_max_abs_yaw_accel           = params["nuplan_max_abs_yaw_accel"]
        self.nuplan_max_abs_lon_jerk            = params["nuplan_max_abs_lon_jerk"]
        self.nuplan_max_abs_yaw_rate            = params["nuplan_max_abs_yaw_rate"]

        # tanh gain
        self.tanh_gain                   = params["tanh_gain"]

        # MinMax Constraints Info
        self.max_lon_accel               = params["max_lon_accel"]
        self.min_lon_accel               = params["min_lon_accel"]
        self.max_abs_lat_accel           = params["max_abs_lat_accel"]
        self.max_abs_lon_jerk            = params["max_abs_lon_jerk"]

        # Normalize term
        self.N_x                         = params["N_x"]
        self.N_y                         = params["N_y"]
        self.N_yaw                       = params["N_yaw"]
        self.N_vel                       = params["N_vel"]

        self.N_jerk                      = params["N_jerk"]
        self.N_ddelta                    = params["N_ddelta"]

        # Tracking Cost Weights
        self.Q_x                         = params["Q_x"]
        self.Q_y                         = params["Q_y"]
        self.Q_yaw                       = params["Q_yaw"]
        self.Q_vel                       = params["Q_vel"]

        # Input Cost Weights
        self.R_jerk                      = params["R_jerk"]
        self.R_ddelta                    = params["R_ddelta"]

        # Tracking cost for the endpoint
        self.Qe_x                        = params["Qe_x"]
        self.Qe_y                        = params["Qe_y"]
        self.Qe_yaw                      = params["Qe_yaw"]
        self.Qe_vel                      = params["Qe_vel"]

        # Kernel-based Cost Parameters
        # Kernel scale parameters (steepness 'a')
        self.kernel_scale_x              = params["kernel_scale_x"]
        self.kernel_scale_y              = params["kernel_scale_y"]
        self.kernel_scale_yaw            = params["kernel_scale_yaw"]
        self.kernel_scale_vel            = params["kernel_scale_vel"]
        self.kernel_scale_jerk           = params["kernel_scale_jerk"]
        self.kernel_scale_ddelta         = params["kernel_scale_ddelta"]

        # Kernel shift parameters (shift 'b')
        self.kernel_shift_x              = params["kernel_shift_x"]
        self.kernel_shift_y              = params["kernel_shift_y"]
        self.kernel_shift_yaw            = params["kernel_shift_yaw"]
        self.kernel_shift_vel            = params["kernel_shift_vel"]
        self.kernel_shift_jerk           = params["kernel_shift_jerk"]
        self.kernel_shift_ddelta         = params["kernel_shift_ddelta"]
        
        # Use kernel cost flags
        self.use_kernel_cost             = params["use_kernel_cost"]
        self.use_kernel_cost_e           = params["use_kernel_cost_e"]

        # Soft Constraints Cost
        self.collision_check_time_s      = params["collision_check_time_s"]
        self.s_lon_jerk                  = params["s_lon_jerk"]
        self.s_lon_accel                 = params["s_lon_accel"]
        self.s_comfort                   = params["s_comfort"]
        self.s_accel                     = params["s_accel"]
        self.s_col                       = params["s_col"]
        self.s_boundary                  = params["s_boundary"]

        self.model = self.CreateModel()
        self.ocp   = self.SetupOCP(self.model)

        # Acados Solver Build or Just Use
        # true: save new changes to the Acados solver
        # false: build the Acados solver from the existing JSON file
        b_acados_generation = params["b_acados_generation"]

        # Save current directory to restore later
        original_dir = os.getcwd()
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            self.ocp.json_file = "acados_ocp.json"
            path = os.path.abspath(self.ocp.json_file)
            print("🔍 solver will load JSON:", path)
            print("📝 json will be written to:", os.path.abspath(self.ocp.json_file))
            if b_acados_generation:
                if os.path.exists('acados_ocp.json'):
                    os.remove('acados_ocp.json')
                self.ocp.dump_to_json()
                self.solver = AcadosOcpSolver(self.ocp,
                                              json_file=os.path.abspath(self.ocp.json_file))
            else:
                self.solver = AcadosOcpSolver(self.ocp, json_file=os.path.abspath(self.ocp.json_file),
                                              build=False,
                                              generate=False)
            self.initialized = True
        finally:
            # Restore original directory
            os.chdir(original_dir)

    # ------------------------------------------------------------------ #
    #                           Model definition                         #
    # ------------------------------------------------------------------ #
    def CreateModel(self) -> AcadosModel:
        # Total States (Rear Axle)
        pos_x   = SX.sym('pos_x')
        pos_y   = SX.sym('pos_y')
        yaw     = SX.sym('yaw')
        v       = SX.sym('v')
        ax      = SX.sym('ax')
        delta   = SX.sym('delta')

        # Input States (Front Axle)
        jerk   = SX.sym('jerk')
        ddelta = SX.sym('ddelta')

        x   = vertcat(pos_x, pos_y, yaw, v, ax, delta)
        u   = vertcat(jerk, ddelta)

        yaw_dot = v * SX.tan(delta) / self.wheel_base

        f_expl = vertcat(
            v * cos(yaw),   # pos_x_dot
            v * sin(yaw),   # pos_y_dot
            yaw_dot,        # yaw_dot
            ax,             # v_dot
            jerk,           # ax_dot
            ddelta,         # delta_dot
        )

        xdot  = SX.sym('xdot', x.shape[0])

        z = vertcat([])

        #  Calculation for Collision Area
        x_c = SX.sym('x_center')
        y_c = SX.sym('y_center')
        x_f = SX.sym('x_front')
        y_f = SX.sym('y_front')

        x_ref      = SX.sym("x_ref")
        y_ref      = SX.sym("y_ref")
        yaw_ref    = SX.sym("yaw_ref")
        vel_ref    = SX.sym("vel_ref")

        agent1_x1 = SX.sym("agent1_x1")
        agent1_y1 = SX.sym("agent1_y1")
        agent1_x2 = SX.sym("agent1_x2")
        agent1_y2 = SX.sym("agent1_y2")
        agent1_x3 = SX.sym("agent1_x3")
        agent1_y3 = SX.sym("agent1_y3")

        agent2_x1 = SX.sym("agent2_x1")
        agent2_y1 = SX.sym("agent2_y1")
        agent2_x2 = SX.sym("agent2_x2")
        agent2_y2 = SX.sym("agent2_y2")
        agent2_x3 = SX.sym("agent2_x3")
        agent2_y3 = SX.sym("agent2_y3")

        agent3_x1 = SX.sym("agent3_x1")
        agent3_y1 = SX.sym("agent3_y1")
        agent3_x2 = SX.sym("agent3_x2")
        agent3_y2 = SX.sym("agent3_y2")
        agent3_x3 = SX.sym("agent3_x3")
        agent3_y3 = SX.sym("agent3_y3")

        agent4_x1 = SX.sym("agent4_x1")
        agent4_y1 = SX.sym("agent4_y1")
        agent4_x2 = SX.sym("agent4_x2")
        agent4_y2 = SX.sym("agent4_y2")
        agent4_x3 = SX.sym("agent4_x3")
        agent4_y3 = SX.sym("agent4_y3")

        agent5_x1 = SX.sym("agent5_x1")
        agent5_y1 = SX.sym("agent5_y1")
        agent5_x2 = SX.sym("agent5_x2")
        agent5_y2 = SX.sym("agent5_y2")
        agent5_x3 = SX.sym("agent5_x3")
        agent5_y3 = SX.sym("agent5_y3")

        agent6_x1 = SX.sym("agent6_x1")
        agent6_y1 = SX.sym("agent6_y1")
        agent6_x2 = SX.sym("agent6_x2")
        agent6_y2 = SX.sym("agent6_y2")
        agent6_x3 = SX.sym("agent6_x3")
        agent6_y3 = SX.sym("agent6_y3")

        agent7_x1 = SX.sym("agent7_x1")
        agent7_y1 = SX.sym("agent7_y1")
        agent7_x2 = SX.sym("agent7_x2")
        agent7_y2 = SX.sym("agent7_y2")
        agent7_x3 = SX.sym("agent7_x3")
        agent7_y3 = SX.sym("agent7_y3")

        agent8_x1 = SX.sym("agent8_x1")
        agent8_y1 = SX.sym("agent8_y1")
        agent8_x2 = SX.sym("agent8_x2")
        agent8_y2 = SX.sym("agent8_y2")
        agent8_x3 = SX.sym("agent8_x3")
        agent8_y3 = SX.sym("agent8_y3")

        agent9_x1 = SX.sym("agent9_x1")
        agent9_y1 = SX.sym("agent9_y1")
        agent9_x2 = SX.sym("agent9_x2")
        agent9_y2 = SX.sym("agent9_y2")
        agent9_x3 = SX.sym("agent9_x3")
        agent9_y3 = SX.sym("agent9_y3")

        agent10_x1 = SX.sym("agent10_x1")
        agent10_y1 = SX.sym("agent10_y1")
        agent10_x2 = SX.sym("agent10_x2")
        agent10_y2 = SX.sym("agent10_y2")
        agent10_x3 = SX.sym("agent10_x3")
        agent10_y3 = SX.sym("agent10_y3")

        pedestrian_x1        = SX.sym("pedestrian_x1")
        pedestrian_y1        = SX.sym("pedestrian_y1")

        pedestrian_x2        = SX.sym("pedestrian_x2")
        pedestrian_y2        = SX.sym("pedestrian_y2")

        pedestrian_x3        = SX.sym("pedestrian_x3")
        pedestrian_y3        = SX.sym("pedestrian_y3")

        pedestrian_x4        = SX.sym("pedestrian_x4")
        pedestrian_y4        = SX.sym("pedestrian_y4")

        pedestrian_x5        = SX.sym("pedestrian_x5")
        pedestrian_y5        = SX.sym("pedestrian_y5")

        agent1_width          = SX.sym("agent1_width")
        agent2_width          = SX.sym("agent2_width")
        agent3_width          = SX.sym("agent3_width")
        agent4_width          = SX.sym("agent4_width")
        agent5_width          = SX.sym("agent5_width")
        agent6_width          = SX.sym("agent6_width")
        agent7_width          = SX.sym("agent7_width")
        agent8_width          = SX.sym("agent8_width")
        agent9_width          = SX.sym("agent9_width")
        agent10_width         = SX.sym("agent10_width")

        left_front_y_lim = SX.sym("left_front_y_lim")
        left_rear_y_lim  = SX.sym("left_rear_y_lim")
        right_front_y_lim= SX.sym("right_front_y_lim")
        right_rear_y_lim = SX.sym("right_rear_y_lim")

        p = vertcat(x_ref, y_ref, yaw_ref, vel_ref,   # 4
                    agent1_x1, agent1_y1, agent1_x2, agent1_y2, agent1_x3, agent1_y3, # 10
                    agent2_x1, agent2_y1, agent2_x2, agent2_y2, agent2_x3, agent2_y3, # 16
                    agent3_x1, agent3_y1, agent3_x2, agent3_y2, agent3_x3, agent3_y3, # 22
                    agent4_x1, agent4_y1, agent4_x2, agent4_y2, agent4_x3, agent4_y3, # 28
                    agent5_x1, agent5_y1, agent5_x2, agent5_y2, agent5_x3, agent5_y3, # 34
                    agent6_x1, agent6_y1, agent6_x2, agent6_y2, agent6_x3, agent6_y3, # 40
                    agent7_x1, agent7_y1, agent7_x2, agent7_y2, agent7_x3, agent7_y3, # 46
                    agent8_x1, agent8_y1, agent8_x2, agent8_y2, agent8_x3, agent8_y3, # 52
                    agent9_x1, agent9_y1, agent9_x2, agent9_y2, agent9_x3, agent9_y3, # 58
                    agent10_x1, agent10_y1, agent10_x2, agent10_y2, agent10_x3, agent10_y3, # 64
                    pedestrian_x1, pedestrian_y1, # 66
                    pedestrian_x2, pedestrian_y2, # 68
                    pedestrian_x3, pedestrian_y3, # 70
                    pedestrian_x4, pedestrian_y4, # 72
                    pedestrian_x5, pedestrian_y5, # 74
                    agent1_width, agent2_width, agent3_width, agent4_width, agent5_width, # 79
                    agent6_width, agent7_width, agent8_width, agent9_width, agent10_width, # 84
                    left_front_y_lim, left_rear_y_lim,
                    right_front_y_lim, right_rear_y_lim)

        x_r       = pos_x - MPCConstants.VehicleDefaults.REAR_AXLE_TO_REAR * cos(yaw)
        y_r       = pos_y - MPCConstants.VehicleDefaults.REAR_AXLE_TO_REAR * sin(yaw)

        rear_to_center = MPCConstants.VehicleDefaults.REAR_AXLE_TO_FRONT - (MPCConstants.VehicleDefaults.LENGTH)/2
        x_c       = pos_x + rear_to_center * cos(yaw)
        y_c       = pos_y + rear_to_center * sin(yaw)

        x_f       = pos_x + MPCConstants.VehicleDefaults.REAR_AXLE_TO_FRONT * cos(yaw)
        y_f       = pos_y + MPCConstants.VehicleDefaults.REAR_AXLE_TO_FRONT * sin(yaw)

        # Create vehicle position object for collision constraints
        vehicle_positions = VehiclePositions(
            rear_x=x_r, rear_y=y_r,
            center_x=x_c, center_y=y_c,
            front_x=x_f, front_y=y_f
        )

        # Additional States for Adapting Comfort Metric
        ay        = (v**2) * tan(delta) / self.wheel_base
        jerk_mag    = jerk**2 + ((2 * v * ax * delta + (v**2) * ddelta) / self.wheel_base) ** 2
        yawrate   = v * tan(delta) / self.wheel_base
        yaw_accel = (ax * delta + v * ddelta) / self.wheel_base

        # Acceleration constraints
        vehicle_acceleration = VehicleAcceleration(
            ax = ax,
            ay = ay
        )
        acceleration_constraints = self.CreateLinearAccelConstraints(vehicle_acceleration)
        (accel_con1, accel_con2, accel_con3,
         accel_con4, accel_con5, accel_con6,
         accel_con7, accel_con8, accel_con9,
         accel_con10, accel_con11, accel_con12) = acceleration_constraints

        # Collision Constraints
        # Agent1
        agent1_trajectory = AgentTrajectory(x1=agent1_x1, y1=agent1_y1,x2=agent1_x2, y2=agent1_y2,x3=agent1_x3, y3=agent1_y3,width=agent1_width)
        agent1_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent1_trajectory)
        agent1_con1, agent1_con2, agent1_con3, agent1_con4, agent1_con5, agent1_con6, agent1_con7, agent1_con8, agent1_con9 = agent1_constraints
        # Agent2
        agent2_trajectory = AgentTrajectory(x1=agent2_x1, y1=agent2_y1, x2=agent2_x2, y2=agent2_y2,x3=agent2_x3, y3=agent2_y3, width=agent2_width)
        agent2_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent2_trajectory)
        agent2_con1, agent2_con2, agent2_con3, agent2_con4, agent2_con5, agent2_con6, agent2_con7, agent2_con8, agent2_con9 = agent2_constraints
        # Agent3
        agent3_trajectory = AgentTrajectory(x1=agent3_x1, y1=agent3_y1, x2=agent3_x2, y2=agent3_y2,x3=agent3_x3, y3=agent3_y3, width=agent3_width)
        agent3_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent3_trajectory)
        agent3_con1, agent3_con2, agent3_con3, agent3_con4, agent3_con5, agent3_con6, agent3_con7, agent3_con8, agent3_con9 = agent3_constraints
        # Agent4
        agent4_trajectory = AgentTrajectory(x1=agent4_x1, y1=agent4_y1, x2=agent4_x2, y2=agent4_y2,x3=agent4_x3, y3=agent4_y3, width=agent4_width)
        agent4_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent4_trajectory)
        agent4_con1, agent4_con2, agent4_con3, agent4_con4, agent4_con5, agent4_con6, agent4_con7, agent4_con8, agent4_con9 = agent4_constraints
        # Agent5
        agent5_trajectory = AgentTrajectory(x1=agent5_x1, y1=agent5_y1, x2=agent5_x2, y2=agent5_y2,x3=agent5_x3, y3=agent5_y3, width=agent5_width)
        agent5_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent5_trajectory)
        agent5_con1, agent5_con2, agent5_con3, agent5_con4, agent5_con5, agent5_con6, agent5_con7, agent5_con8, agent5_con9 = agent5_constraints
        # Agent 6
        agent6_trajectory = AgentTrajectory(x1=agent6_x1, y1=agent6_y1, x2=agent6_x2, y2=agent6_y2,x3=agent6_x3, y3=agent6_y3, width=agent6_width)
        agent6_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent6_trajectory)
        agent6_con1, agent6_con2, agent6_con3, agent6_con4, agent6_con5, agent6_con6, agent6_con7, agent6_con8, agent6_con9 = agent6_constraints
        # Agent7
        agent7_trajectory = AgentTrajectory(x1=agent7_x1, y1=agent7_y1, x2=agent7_x2, y2=agent7_y2,x3=agent7_x3, y3=agent7_y3, width=agent7_width)
        agent7_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent7_trajectory)
        agent7_con1, agent7_con2, agent7_con3, agent7_con4, agent7_con5, agent7_con6, agent7_con7, agent7_con8, agent7_con9 = agent7_constraints
        # Agent8
        agent8_trajectory = AgentTrajectory(x1=agent8_x1, y1=agent8_y1, x2=agent8_x2, y2=agent8_y2,x3=agent8_x3, y3=agent8_y3, width=agent8_width)
        agent8_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent8_trajectory)
        agent8_con1, agent8_con2, agent8_con3, agent8_con4, agent8_con5, agent8_con6, agent8_con7, agent8_con8, agent8_con9 = agent8_constraints
        # Agent9
        agent9_trajectory = AgentTrajectory(x1=agent9_x1, y1=agent9_y1, x2=agent9_x2, y2=agent9_y2,x3=agent9_x3, y3=agent9_y3, width=agent9_width)
        agent9_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent9_trajectory)
        agent9_con1, agent9_con2, agent9_con3, agent9_con4, agent9_con5, agent9_con6, agent9_con7, agent9_con8, agent9_con9 = agent9_constraints
        # Agent10
        agent10_trajectory = AgentTrajectory(x1=agent10_x1, y1=agent10_y1, x2=agent10_x2, y2=agent10_y2,x3=agent10_x3, y3=agent10_y3, width=agent10_width)
        agent10_constraints = self.CreateAgentCollisionConstraints(vehicle_positions, agent10_trajectory)
        agent10_con1, agent10_con2, agent10_con3, agent10_con4, agent10_con5, agent10_con6, agent10_con7, agent10_con8, agent10_con9 = agent10_constraints

        # Pedestrian Collision Constraints
        pedestrian1 = PedestrianPosition(x=pedestrian_x1, y=pedestrian_y1)
        pedestrian1_constraints = self.CreatePedestrianCollisionConstraints(vehicle_positions, pedestrian1)
        pedestrain_1_con1, pedestrain_1_con2, pedestrain_1_con3 = pedestrian1_constraints

        pedestrian2 = PedestrianPosition(x=pedestrian_x2, y=pedestrian_y2)
        pedestrian2_constraints = self.CreatePedestrianCollisionConstraints(vehicle_positions, pedestrian2)
        pedestrain_2_con1, pedestrain_2_con2, pedestrain_2_con3 = pedestrian2_constraints

        pedestrian3 = PedestrianPosition(x=pedestrian_x3, y=pedestrian_y3)
        pedestrian3_constraints = self.CreatePedestrianCollisionConstraints(vehicle_positions, pedestrian3)
        pedestrain_3_con1, pedestrain_3_con2, pedestrain_3_con3 = pedestrian3_constraints

        pedestrian4 = PedestrianPosition(x=pedestrian_x4, y=pedestrian_y4)
        pedestrian4_constraints = self.CreatePedestrianCollisionConstraints(vehicle_positions, pedestrian4)
        pedestrain_4_con1, pedestrain_4_con2, pedestrain_4_con3 = pedestrian4_constraints

        pedestrian5 = PedestrianPosition(x=pedestrian_x5, y=pedestrian_y5)
        pedestrian5_constraints = self.CreatePedestrianCollisionConstraints(vehicle_positions, pedestrian5)
        pedestrain_5_con1, pedestrain_5_con2, pedestrain_5_con3 = pedestrian5_constraints

        # Calculate vehicle corner positions
        # Vehicle dimensions: width = self.width, wheelbase = self.wheel_base
        # Rear axle is at (pos_x, pos_y), heading is yaw

        # Front axle position (already defined as x_f, y_f)
        # Rear axle position is (pos_x, pos_y)

        # Left side offset (perpendicular to heading, positive y direction)
        left_offset_x = -self.width_half * sin(yaw)
        left_offset_y = self.width_half * cos(yaw)

        # Right side offset (perpendicular to heading, negative y direction)
        right_offset_x = self.width_half * sin(yaw)
        right_offset_y = -self.width_half * cos(yaw)

        # Four corner positions
        # Front left corner
        front_left_x = x_f + left_offset_x
        front_left_y = y_f + left_offset_y

        # Front right corner
        front_right_x = x_f + right_offset_x
        front_right_y = y_f + right_offset_y

        # Rear left corner
        rear_left_x = x_r + left_offset_x
        rear_left_y = y_r + left_offset_y

        # Rear right corner
        rear_right_x = x_r + right_offset_x
        rear_right_y = y_r + right_offset_y

        # Boundary constraints
        # Left boundary: vehicle left corners should be less than left_y_lim
        # Constraint: left_y_lim - corner_y > 0  (corner_y < left_y_lim)
        left_boundary_con1 = left_front_y_lim - front_left_y - self.boundary_margin   # Front left corner < left boundary
        left_boundary_con2 = left_rear_y_lim - rear_left_y - self.boundary_margin     # Rear left corner < left boundary

        right_boundary_con1 = front_right_y - right_front_y_lim - self.boundary_margin # Front right corner > right boundary
        right_boundary_con2 = rear_right_y - right_rear_y_lim - self.boundary_margin  # Rear right corner > right boundary

        con_h_expr = vertcat(ay, jerk_mag, yawrate, yaw_accel,        # 4
                             accel_con1, accel_con2, accel_con3,
                             accel_con4, accel_con5, accel_con6,
                             accel_con7, accel_con8, accel_con9,
                             accel_con10, accel_con11, accel_con12,  # 12, 16
                             agent1_con1,  agent1_con2,  agent1_con3,  agent1_con4,  agent1_con5,  agent1_con6,  agent1_con7,  agent1_con8,  agent1_con9,
                             agent2_con1,  agent2_con2,  agent2_con3,  agent2_con4,  agent2_con5,  agent2_con6,  agent2_con7,  agent2_con8,  agent2_con9,
                             agent3_con1,  agent3_con2,  agent3_con3,  agent3_con4,  agent3_con5,  agent3_con6,  agent3_con7,  agent3_con8,  agent3_con9,
                             agent4_con1,  agent4_con2,  agent4_con3,  agent4_con4,  agent4_con5,  agent4_con6,  agent4_con7,  agent4_con8,  agent4_con9,
                             agent5_con1,  agent5_con2,  agent5_con3,  agent5_con4,  agent5_con5,  agent5_con6,  agent5_con7,  agent5_con8,  agent5_con9,
                             agent6_con1,  agent6_con2,  agent6_con3,  agent6_con4,  agent6_con5,  agent6_con6,  agent6_con7,  agent6_con8,  agent6_con9,
                             agent7_con1,  agent7_con2,  agent7_con3,  agent7_con4,  agent7_con5,  agent7_con6,  agent7_con7,  agent7_con8,  agent7_con9,
                             agent8_con1,  agent8_con2,  agent8_con3,  agent8_con4,  agent8_con5,  agent8_con6,  agent8_con7,  agent8_con8,  agent8_con9,
                             agent9_con1,  agent9_con2,  agent9_con3,  agent9_con4,  agent9_con5,  agent9_con6,  agent9_con7,  agent9_con8,  agent9_con9,
                             agent10_con1, agent10_con2, agent10_con3, agent10_con4, agent10_con5, agent10_con6, agent10_con7, agent10_con8, agent10_con9,   # 90, 106
                             pedestrain_1_con1, pedestrain_1_con2, pedestrain_1_con3,
                             pedestrain_2_con1, pedestrain_2_con2, pedestrain_2_con3,
                             pedestrain_3_con1, pedestrain_3_con2, pedestrain_3_con3,
                             pedestrain_4_con1, pedestrain_4_con2, pedestrain_4_con3,
                             pedestrain_5_con1, pedestrain_5_con2, pedestrain_5_con3,   # 15, 121
                             left_boundary_con1, left_boundary_con2,
                             right_boundary_con1, right_boundary_con2)                  # 4, 125

        model = AcadosModel()
        model.p = p
        model.con_h_expr    = con_h_expr
        model.f_expl_expr   = f_expl
        model.xdot          = xdot
        model.x             = x
        model.u             = u
        model.z             = z
        model.name          = "kinematic_model"
        return model

    # ------------------------------------------------------------------ #
    #                         OCP Setting                                #
    # ------------------------------------------------------------------ #
    def SetupOCP(self, model):
        """
        Main OCP setup function - orchestrates all setup steps
        """
        ocp = self.InitializeOcpBase(model)

        # Setup cost expressions (includes risk calculations)
        self.SetupCostExpressions(ocp)

        # Setup cost function configuration
        self.SetupCostFunction(ocp)

        # Setup constraints
        self.SetupConstraints(ocp)

        # Setup solver options
        self.SetupSolverOptions(ocp)

        return ocp

    def InitializeOcpBase(self, model):
        """
        Initialize basic OCP structure and parameters
        """
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.T
        ocp.dims.N = self.T

        # Initialize parameter values
        np_param = ocp.model.p.rows()
        ocp.parameter_values = np.zeros(np_param)
        ocp.dims.np = np_param

        # Parameters structure (using MPCConstants for indices):
            # Reference trajectory: x_ref, y_ref, yaw_ref, vel_ref (0-3)
            # Agent trajectories: 10 agents * 6 values each (4-63)
            # Pedestrian positions: 5 pedestrians * 2 values each (64-73)
            # Agent widths: 10 values (74-83)
            # Road boundary limits left (84-86) and right (87-89)

        return ocp

    def SetupCostExpressions(self, ocp):
        """
        Setup cost expressions including risk calculations and reference tracking
        """
        x = ocp.model.x  # [x, y, yaw, v, ax, delta]
        u = ocp.model.u  # [jerk, ddelta]
        p = ocp.model.p  # [x, y, yaw, v, agent_data, pedestrian_data, ...]

        # Reference trajectory parameter indices
        x_ref_idx = MPCConstants.ParameterIndices.REF_TRAJECTORY_START
        y_ref_idx = MPCConstants.ParameterIndices.REF_TRAJECTORY_START + 1
        yaw_ref_idx = MPCConstants.ParameterIndices.REF_TRAJECTORY_START + 2

        # ============================================================
        # ORIGINAL COST FORMULATION (Quadratic with Normalization)
        # ============================================================
        # # Stage cost expression (tracking + input costs)
        # cost_expr = (self.Q_x     * ((p[x_ref_idx]-x[0])**2)   / (self.N_x ** 2)    +
        #              self.Q_y      * ((p[y_ref_idx]-x[1])**2)   / (self.N_y ** 2)    +
        #             #  self.Q_yaw    * ((p[yaw_ref_idx]-x[2])**2) / (self.N_yaw ** 2)  +
        #              self.Q_vel    * ((p[vel_ref_idx]-x[3])**2) / (self.N_vel ** 2)+
        #              self.R_jerk   * ((u[0]) ** 2)              / (self.N_jerk ** 2) +
        #              self.R_ddelta * ((u[1]) ** 2)              / (self.N_ddelta ** 2))
        #
        # # Terminal cost expression (only tracking costs)
        # cost_expr_ext = (self.Qe_x   * ((p[x_ref_idx]-x[0])**2)   / (self.N_x ** 2)   +
        #                  self.Qe_y   * ((p[y_ref_idx]-x[1])**2)   / (self.N_y ** 2)   +
        #                 #  self.Qe_yaw * ((p[yaw_ref_idx]-x[2])**2) / (self.N_yaw ** 2) +
        #                  self.Qe_vel * ((p[vel_ref_idx]-x[3])**2) / (self.N_vel ** 2))


        # ============================================================
        # KERNEL-BASED COST FORMULATION
        # kernel = (tanh(a*error² + b) + 1) / 2, range: [0, 1]
        # ============================================================
        error_x_square = (p[x_ref_idx] - x[0])**2
        error_y_square = (p[y_ref_idx] - x[1])**2
        error_yaw_square = (p[yaw_ref_idx] - x[2])**2
        error_jerk_square = (u[0])**2
        error_ddelta_square = (u[1])**2

        kernel_x = (tanh(self.kernel_scale_x * error_x_square + self.kernel_shift_x) + 1.0) / 2.0
        kernel_y = (tanh(self.kernel_scale_y * error_y_square + self.kernel_shift_y) + 1.0) / 2.0
        kernel_yaw = (tanh(self.kernel_scale_yaw * error_yaw_square + self.kernel_shift_yaw) + 1.0) / 2.0
        kernel_jerk = (tanh(self.kernel_scale_jerk * error_jerk_square + self.kernel_shift_jerk) + 1.0) / 2.0
        kernel_ddelta = (tanh(self.kernel_scale_ddelta * error_ddelta_square + self.kernel_shift_ddelta) + 1.0) / 2.0

        # Stage cost
        # Q_yaw 항은 heading 감쇠용이다. 위치 오차(비례항)만 있고 heading 이 없으면
        # 계단형 reference 에서 부족감쇠가 나 오버슈트한다 — 실측 |e0|=3.5 m 에서
        # 1.08 m 넘어간 뒤 진동. Q_yaw=0 이면 이 항은 상수라 종전과 동일하게 동작한다.
        cost_expr = self.use_kernel_cost * (self.Q_x * kernel_x + self.Q_y * kernel_y +
                                            self.Q_yaw * kernel_yaw +
                                            self.R_jerk * kernel_jerk + self.R_ddelta * kernel_ddelta)

        # Terminal cost
        cost_expr_ext = self.use_kernel_cost_e * (self.Qe_x * kernel_x + self.Qe_y * kernel_y)

        # Store cost expressions in the OCP model
        ocp.model.cost_expr_ext_cost = cost_expr
        ocp.model.cost_expr_ext_cost_e = cost_expr_ext

    def SetupCostFunction(self, ocp):
        """
        Setup cost function configuration
        """
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

    def SetupConstraints(self, ocp):
        """
        Setup all constraint definitions and soft constraint weights
        """
        nx = self.model.x.rows()

        # ------------------------------------------------------------------ #
        #        Constraint Definition
        #        States:(x,y,yaw,v,ax,delta)
        #        Inputs: (jerk, ddelta)
        # ------------------------------------------------------------------ #

        # 1) State Constraints
        ocp.constraints.lbx = np.array([-0.001, self.nuplan_min_lon_accel, -np.deg2rad(self.max_delta)])
        ocp.constraints.ubx = np.array([MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC , self.nuplan_max_lon_accel, np.deg2rad(self.max_delta)])
        ocp.constraints.idxbx = np.array([3, 4, 5])  # v, ax, delta
        ocp.dims.nbx = ocp.constraints.idxbx.shape[0]
        ocp.constraints.idxsbx = np.array([1])

        # 2) Initial State Constraints
        ocp.constraints.idxbx_0 = np.arange(nx).astype(int)
        ocp.constraints.lbx_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ocp.constraints.ubx_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ocp.dims.nbx_0 = ocp.constraints.idxbx_0.shape[0]

        # 3) Terminal State Constraints
        ocp.constraints.lbx_e = np.array([-0.001, self.nuplan_min_lon_accel, -np.deg2rad(self.max_delta)])
        ocp.constraints.ubx_e = np.array([MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC, self.nuplan_max_lon_accel, np.deg2rad(self.max_delta)])
        ocp.constraints.idxbx_e = np.array([3, 4, 5])  # ax, delta
        ocp.dims.nbx_e = ocp.constraints.idxbx_e.shape[0]
        ocp.constraints.idxsbx_e = np.array([1])

        # 4) Input Constraints
        ocp.constraints.lbu = np.array([-self.max_abs_lon_jerk, -np.deg2rad(self.max_ddelta)])
        ocp.constraints.ubu = np.array([self.max_abs_lon_jerk, np.deg2rad(self.max_ddelta)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.dims.nbu = ocp.constraints.idxbu.shape[0]
        ocp.constraints.idxsbu = np.array([0])

        # 5) Nonlinear Constraints (Comfort + Collision)
        self.SetupNonlinearConstraints(ocp)

        # 6) Soft Constraint Weights
        self.SetupSoftConstraintWeights(ocp)

    def SetupNonlinearConstraints(self, ocp):
        """
        Setup nonlinear constraints (comfort and collision)
        """
        # Comfort constraints
        comfort_lh = np.array([-self.nuplan_max_abs_lat_accel, -0.001, -self.nuplan_max_abs_yaw_rate, -self.nuplan_max_abs_yaw_accel])
        comfort_uh = np.array([self.nuplan_max_abs_lat_accel, self.nuplan_max_abs_mag_jerk ** 2, self.nuplan_max_abs_yaw_rate, self.nuplan_max_abs_yaw_accel])

        # Acceleration constraints
        acceleration_constraint_count = MPCConstants.ACCELERATION_CONSTRAINTS
        acceleration_lh = np.zeros(acceleration_constraint_count)
        acceleration_uh = np.full(acceleration_constraint_count, MPCConstants.Numerical.LARGE_POSITIVE)

        # Collision constraints
        collision_constraint_count = MPCConstants.AGENT_CONSTRAINTS + MPCConstants.PEDESTRIAN_CONSTRAINTS
        collision_lh = np.zeros(collision_constraint_count)  # All collision constraints >= 0
        collision_uh = np.full(collision_constraint_count, MPCConstants.Numerical.LARGE_POSITIVE)

        boundary_lh = np.zeros(MPCConstants.BOUNDARY_CONSTRAINTS)
        boundary_uh = np.full(MPCConstants.BOUNDARY_CONSTRAINTS, MPCConstants.Numerical.LARGE_POSITIVE)

        ocp.constraints.lh = np.concatenate([comfort_lh, acceleration_lh, collision_lh, boundary_lh])
        ocp.constraints.uh = np.concatenate([comfort_uh, acceleration_uh, collision_uh, boundary_uh])
        ocp.dims.nh = ocp.constraints.lh.shape[0]
        ocp.constraints.idxsh = np.arange(ocp.constraints.lh.shape[0], dtype=int)

    def SetupSoftConstraintWeights(self, ocp):
        """
        Setup soft constraint weights for all constraint types
        """
        # Soft constraint weights: input(1) + state(1) + nonlinear constraints
        comfort_weights      = [self.s_comfort] * MPCConstants.COMFORT_CONSTRAINTS
        acceleration_weights = [self.s_accel] * MPCConstants.ACCELERATION_CONSTRAINTS
        agent_weights        = [self.s_col] * MPCConstants.AGENT_CONSTRAINTS
        pedestrian_weights   = [self.s_col] * MPCConstants.PEDESTRIAN_CONSTRAINTS
        boundary_weights     = [self.s_boundary] * MPCConstants.BOUNDARY_CONSTRAINTS

        soft_weights = ([self.s_lon_jerk] + [self.s_lon_accel] +
                       comfort_weights + acceleration_weights + agent_weights + pedestrian_weights + boundary_weights)

        # Stage constraints soft weights
        ocp.cost.Zl = np.array(soft_weights, dtype=float)
        ocp.cost.Zu = np.array(soft_weights, dtype=float)
        ocp.cost.zl = np.array(soft_weights, dtype=float)
        ocp.cost.zu = np.array(soft_weights, dtype=float)

        # Terminal state constraint soft weights
        ocp.cost.Zl_e = np.array([self.s_lon_accel], dtype=float)
        ocp.cost.Zu_e = np.array([self.s_lon_accel], dtype=float)
        ocp.cost.zl_e = np.array([self.s_lon_accel], dtype=float)
        ocp.cost.zu_e = np.array([self.s_lon_accel], dtype=float)

        # Initial input constraint soft weights
        ocp.cost.Zl_0 = self.s_lon_jerk * np.ones(1)
        ocp.cost.Zu_0 = self.s_lon_jerk * np.ones(1)
        ocp.cost.zl_0 = self.s_lon_jerk * np.ones(1)
        ocp.cost.zu_0 = self.s_lon_jerk * np.ones(1)

    def SetupSolverOptions(self, ocp):
        """
        Setup solver options and parameters
        """
        ocp.solver_options.tf = self.horizon
        ocp.solver_options.tol = 5e-4
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.hpipm_mode = "SPEED"
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.regularize_method = 'CONVEXIFY'
        # 30 → 50 (2026-07-26). acados 기본값이 50 인데 30 으로 낮춰져 있었다.
        # SQP 는 이중 루프다: 바깥(nlp_solver_max_iter)이 선형화·이동을 반복하고,
        # 안쪽(여기)이 각 스텝의 QP 를 interior-point 로 푼다. 안쪽이 소진되면
        # **그 스텝의 방향 자체가 부정확**해져 바깥을 아무리 늘려도 수렴하지 않는다.
        # maxiter 30→100 으로 status=2 가 81%→71% 밖에 안 떨어진 것이 이 가설의 근거.
        # 50 → 100 (2026-07-26, 2차). 30→50 에서 status=2 비율은 71.1%→71.4% 로
        # 그대로였지만(= QP 반복 부족이 미수렴 원인은 아님) **궤적 품질은 계속 좋아졌다**:
        # 취득 99→111, refine 악화 97→82, 충돌항 Δ −0.018→−0.010.
        # 같은 추세가 100 에서도 이어지는지 확인한다.
        # qp100 은 qp50 보다 나아지지 않았다(취득 111→104, 충돌항 −0.010→−0.028).
        # qp50 으로 되돌려 고정한다.
        ocp.solver_options.qp_solver_iter_max = 50         # acados 기본값
        # 30 → 100 (2026-07-26). val_demo 수집 런에서 solve status=2(ACADOS_MAXITER)가
        # 전체 스텝의 81% 였다. tol 5e-4 + step_length 0.7(감쇠) 조합에서 30회는
        # 부족했을 가능성이 높다. 시뮬 병목이 MPC 가 아니므로 반복을 늘려
        # "덜 돌린 것"인지 "수렴 자체가 안 되는 것"인지 가른다.
        ocp.solver_options.nlp_solver_max_iter = 100       # was 30
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.levenberg_marquardt = 1e-3
        ocp.solver_options.line_search_use_sufficient_descent = 1
        ocp.solver_options.nlp_solver_step_length = 0.7
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    # ------------------------------------------------------------------ #
    #                           MPC Solve                                #
    # ------------------------------------------------------------------ #
    def Solve(self,
              x0: np.ndarray, # (6,) [x, y, yaw, v, ax, delta]
              x_ref: np.ndarray, # (80, 4) [x, y, yaw, v]
              agent_pred_info: np.ndarray, # (80, 60) [x_c, y_c, x_r, y_r, x_f, y_f], 10 agents × 6 pts, each agent: (x_c, y_c, x_r, y_r, x_f, y_f)
              pedestrian_pred_info: np.ndarray, # (80, 10) [x_c, y_c], 5 pedestrians
              agent_width_info: np.ndarray, # (80, 10) [width], 10 agents
              left_front_y_lim: np.ndarray,  # (80,) left front boundary y limit
              left_rear_y_lim: np.ndarray,   # (80,) left rear boundary y
              right_front_y_lim: np.ndarray, # (80,) right front boundary y limit
              right_rear_y_lim: np.ndarray,  # (80,) right rear boundary y limit
              ): # (80, 84)
        """
        Solve MPC and return first control + predicted trajectory.

        Parameters
        ----------
        x0 : (6,) ndarray [x, y, yaw, v, ax, delta]
            Current state.
        x_ref : (N, 107) ndarray
            Reference trajectory; N can be different from N+1.

        Returns
        -------
        u0 : (2,) ndarray
        x_pred : (N, 6) ndarray [x, y, yaw, v, ax, delta]
        """

        if not self.initialized:
            raise RuntimeError("PostProcessMPC is not initialized")

        ## Clip velocity to be non-negative
        x_ref[:, 3] = np.maximum(x_ref[:, 3], 0.0)

        ## Calculate acceleration
        acceleration = (x_ref[1:, 3] - x_ref[:-1, 3]) / self.dt
        acceleration = np.append(acceleration[0], acceleration)
        acceleration = np.clip(acceleration, self.min_lon_accel, self.max_lon_accel)

        ## Calculate jerk
        jerk = (acceleration[1:] - acceleration[:-1]) / self.dt
        jerk = np.clip(jerk, -self.max_abs_lon_jerk, self.max_abs_lon_jerk)

        ## Calculate yaw rate
        yaw_rate = (x_ref[1:, 2] - x_ref[:-1, 2]) / self.dt
        yaw_rate = np.append(yaw_rate[0], yaw_rate)
        yaw_rate = np.clip(yaw_rate, -self.nuplan_max_abs_yaw_rate, self.nuplan_max_abs_yaw_rate)

        ## Calculate steering angle
        # Prevent division by zero when velocity is near zero
        velocity = np.maximum(np.abs(x_ref[:, 3]), 1e-3)  # Use minimum velocity threshold for each trajectory point
        # From bicycle model: yaw_rate = v * tan(delta) / L → delta = arctan(yaw_rate * L / v)
        steering_angle = np.arctan(yaw_rate * self.wheel_base / velocity + 1e-6)
        steering_angle = np.clip(steering_angle, -np.deg2rad(self.max_delta), np.deg2rad(self.max_delta))

        ## Calculate delta steering angle
        delta_steering_angle = (steering_angle[1:] - steering_angle[:-1]) / self.dt
        delta_steering_angle = np.clip(delta_steering_angle, -np.deg2rad(self.max_ddelta), np.deg2rad(self.max_ddelta))

        ## Calculate total state parameters
        total_state_parameters = np.column_stack([x_ref, acceleration, steering_angle])
        total_control_input_parameters = np.column_stack([jerk, delta_steering_angle])

        # Reference stages
        N_stage = x_ref.shape[0]
        if N_stage < 1:
            raise ValueError("x_ref must have at least two states.")

        # when use just norml controller
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)

        # Set Parameters
        for i in range(0, self.T+1):
            if i == 0:
                self.solver.set(i, "p", np.hstack([x0[:4],
                                                   agent_pred_info[0],
                                                   pedestrian_pred_info[0],
                                                   agent_width_info[0],
                                                   left_front_y_lim[0],
                                                   left_rear_y_lim[0],
                                                   right_front_y_lim[0],
                                                   right_rear_y_lim[0]]))
            else:
                self.solver.constraints_set(i, "lbx", np.array([-0.001, self.nuplan_min_lon_accel, -np.deg2rad(self.max_delta)]))
                self.solver.constraints_set(i, "ubx", np.array([MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC, self.nuplan_max_lon_accel, np.deg2rad(self.max_delta)]))

                # Collision time horizon
                if i*self.dt < self.collision_check_time_s:
                    agent_prediction = agent_pred_info[i-1]
                    pedestrian_prediction = pedestrian_pred_info[i-1]
                    agent_width      = agent_width_info[i-1]
                else:
                    agent_prediction        = np.array([MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC,
                                                        MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC,
                                                        MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC,
                                                        MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC,
                                                        MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC,
                                                        MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC]*MPCConstants.AGENT_COUNT)
                    pedestrian_prediction   = np.array([MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC,
                                                        MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC]*MPCConstants.PEDESTRIAN_COUNT)

                    agent_width             = np.array([MPCConstants.Numerical.SMALL_EPSILON]*MPCConstants.AGENT_COUNT)

                if i*self.dt > self.boundary_lim_time:
                    left_front_y_lim[i-1]  =  MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC
                    left_rear_y_lim[i-1]   =  MPCConstants.Numerical.LARGE_POSITIVE_FOR_CALC
                    right_front_y_lim[i-1] =  MPCConstants.Numerical.LARGE_NEGATIVE_FOR_CALC
                    right_rear_y_lim[i-1]  =  MPCConstants.Numerical.LARGE_NEGATIVE_FOR_CALC

                self.solver.set(i, "p", np.hstack([x_ref[i-1],
                                                   agent_prediction,
                                                   pedestrian_prediction,
                                                   agent_width,
                                                   left_front_y_lim[i-1],
                                                   left_rear_y_lim[i-1],
                                                   right_front_y_lim[i-1],
                                                   right_rear_y_lim[i-1]]))

                # Set Warm Start
                # self.solver.set(i, "x", total_state_parameters[i-1])
                # if i != self.T:
                #     self.solver.set(i, "u", total_control_input_parameters[i-1])

        # SQP solve
        start_time = time.time()
        status = self.solver.solve()
        total_cost = self.solver.get_cost()
        print("[Acados Pure Solving Time] ", time.time() - start_time, " [sec]")
        print("[Acados Result] status: ", status)
        print("[Acados Result] cost: ", total_cost)

        # Get Result
        u_pred = np.vstack([self.solver.get(k, "u") for k in range(N_stage)]) # [T, 2], [jerk, ddelta]
        x_pred = np.vstack([self.solver.get(k, "x") for k in range(1, N_stage + 1)]) # [T, 6], [x, y, yaw, v, ax, delta]

        return u_pred, x_pred, status, total_cost

    def WorldToLocal(self, xw, yw, x_ref, y_ref, cos_y, sin_y, T = 80):                             # Transform function
        dx = xw - x_ref.reshape(T, 1)
        dy = yw - y_ref.reshape(T, 1)
        x_local = dx * cos_y + dy * sin_y
        y_local = -dx * sin_y + dy * cos_y
        return x_local, y_local

    def GetInputParameters(
            self,
            current_state: np.ndarray,
            reference_traj: np.ndarray,          # (T,4)  rear  [x, y, yaw, v]
            agent_pred_info: np.ndarray,         # (T,60)  10 agents × 6 pts, each agent: (x_c, y_c, x_r, y_r, x_f, y_f)
            pedestrian_pred_info: np.ndarray,
            agent_width_info: np.ndarray,
            yaw_cost_default:float = 0.1,
            speed_threshold: float = 5.0,
            ) -> np.ndarray:
        """
        Fast input packaging using a rectangular ROI for neighbor selection.
        """

        if not self.initialized:
            raise RuntimeError("PostProcessMPC is not initialized")

        # 0. Prepare reference trajectory and dimensions
        reference_traj = np.asarray(reference_traj, dtype=float)  # Ensure float array
        T = reference_traj.shape[0]                               # Time horizon length

        # 2. Reshape agent predictions into (T, n_agents, 6)
        n_agents = agent_pred_info.shape[1] // 6            # Compute number of agents
        pos_all  = agent_pred_info.reshape(T, n_agents, 6)  # Split into coordinate groups

        current_speed = current_state[3]
        yaw_cost = np.zeros((reference_traj.shape[0], 1), dtype=np.float32)
        if current_speed >= speed_threshold:
            yaw_cost[:] = yaw_cost_default

        out = np.hstack([reference_traj,
                         pos_all.reshape(T, -1),
                         pedestrian_pred_info.reshape(T,-1),
                         agent_width_info.reshape(T, -1),
                         ]).astype(np.float32)
        return out

    def BuildAUTOFromArrays(
            self,
            current_state_np: np.ndarray,
            reference_traj: np.ndarray,           # (T,4) [x,y,yaw,v]
            agent_pred_info: np.ndarray,          # (T, 6*max_agents)
            pedestrian_info: np.ndarray,          # (T, 2*max_pedestrians_or_10)
            agent_width_info: np.ndarray,         # (T, max_agents)
            time_step: int = 0,
            max_agents: int = 10,
            max_pedestrians: int = 5,
            speed_limit: float = None,
        ) -> tuple:
        """
        Convert existing numpy arrays (reference_traj, agent_pred_info, pedestrian_info, agent_width_info)
        into AUTO types: (AUTO_VehicleState, AUTO_Trajectory, AUTO_Objects for agents, AUTO_Objects for pedestrians).

        Returns:
            AUTO_VehicleState, AUTO_Trajectory, AUTO_Objects (agents), AUTO_Objects (pedestrians)
        """
        from ..interface.python import (
            AUTO_VehicleState, AUTO_Trajectory, AUTO_TrajectoryPoint,
            AUTO_Objects, AUTO_Object, AUTO_Pose, AUTO_Motion, AUTO_Gnss
        )

        # Clamp time_step
        T = int(reference_traj.shape[0])
        if T <= 0:
            raise ValueError("reference_traj must have length > 0")
        ts = int(np.clip(time_step, 0, T - 1))

        # 1) Vehicle state from current_state_np = [x, y, yaw, v, ax?, delta?]
        veh = AUTO_VehicleState()
        # Set nested fields in-place to avoid ctypes type mismatch
        veh.pose.x = float(current_state_np[0])
        veh.pose.y = float(current_state_np[1])
        veh.pose.yaw = float(current_state_np[2])
        veh.motion.speed_x = float(current_state_np[3]) if current_state_np.shape[0] > 3 else 0.0
        veh.motion.acceleration_x = float(current_state_np[4]) if current_state_np.shape[0] > 4 else 0.0
        veh.steering_angle = float(current_state_np[5]) if current_state_np.shape[0] > 5 else 0.0
        # Calculate yaw rate
        speed = current_state_np[3]
        steering_angle = current_state_np[5]
        yaw_rate = np.tan(steering_angle) * speed / self.wheel_base
        veh.motion.yaw_rate = float(yaw_rate)

        # 2) AUTO_Trajectory from RELATIVE reference_traj
        traj = AUTO_Trajectory()
        traj.number_of_trajectory_points = int(min(T, 500))
        # For relative trajectory, base pose can be zero
        traj.pose.x = 0.0
        traj.pose.y = 0.0
        traj.pose.yaw = 0.0
        for i in range(traj.number_of_trajectory_points):
            tp = traj.trajectory_point[i]
            x_rel, y_rel, yaw_rel, v = reference_traj[i]
            tp.x_rel = float(x_rel)
            tp.y_rel = float(y_rel)
            tp.yaw_rel = float(yaw_rel)
            tp.speed = float(v)
            tp.time = float(i * self.dt)

        # 3) Unified Objects: agents (center/rear/front) + pedestrians (points)
        objects = AUTO_Objects()
        objects.primary_object_index = 0
        total_idx = 0

        # Already ego-local (relative)
        vyaw = veh.pose.yaw

        # Prepare rows
        row = agent_pred_info[ts] if agent_pred_info is not None and agent_pred_info.size > 0 else None
        widths_row = agent_width_info[ts] if agent_width_info is not None and agent_width_info.size > 0 else None
        has_next = ts + 1 < T and agent_pred_info is not None and agent_pred_info.size > 0
        next_row = agent_pred_info[ts + 1] if has_next else None
        dt = self.dt if self.dt > 0 else 0.1

        # Agents first (inputs are local already)
        n_agents = int(agent_pred_info.shape[1] // 6) if row is not None else 0
        n_agents = min(n_agents, max_agents)
        for i in range(n_agents):
            obj = objects.object[total_idx]
            b = i * 6
            x_c, y_c, x_r, y_r, x_f, y_f = row[b:b+6]
            # center in local frame
            obj.x_rel = float(x_c)
            obj.y_rel = float(y_c)
            # yaw from rear->front vector
            yawg = float(np.arctan2(y_f - y_r, x_f - x_r))
            obj.yaw_rel = float(yawg)
            # speed estimate
            if has_next:
                x_c2, y_c2 = next_row[b], next_row[b+1]
                vx_l = (x_c2 - x_c) / dt
                vy_l = (y_c2 - y_c) / dt
                obj.speed_x_rel = float(vx_l)
                obj.speed_y_rel = float(vy_l)
                obj.speed = float(np.hypot(vx_l, vy_l))
            else:
                obj.speed_x_rel = 0.0
                obj.speed_y_rel = 0.0
                obj.speed = 0.0
            # dimensions
            if widths_row is not None and i < widths_row.shape[0]:
                obj.width = float(widths_row[i])
            else:
                obj.width = float(self.width)
            obj.length = float(max(np.hypot(x_f - x_r, y_f - y_r), 1.0))
            obj.object_class = 3 # 3: Car
            total_idx += 1

        # Pedestrians appended (inputs are local already)
        ped_pairs = int(pedestrian_info.shape[1] // 2) if pedestrian_info is not None and pedestrian_info.size > 0 else 0
        n_peds = min(ped_pairs, max_pedestrians)
        if n_peds > 0:
            prow = pedestrian_info[ts]
            for i in range(n_peds):
                obj = objects.object[total_idx]
                x_l = float(prow[i * 2])
                y_l = float(prow[i * 2 + 1])
                obj.x_rel = x_l
                obj.y_rel = y_l
                obj.width = 0.8
                obj.length = 0.8
                obj.object_class = 1 # 1: Pedestrian
                total_idx += 1

        objects.number_of_objects = total_idx

        return veh, traj, objects

    def ConvertAUTOToMPCParameters(
            self,
            current_state: "AUTO_VehicleState",
            reference_trajectory: "AUTO_Trajectory",
            objects: "AUTO_Objects",
            time_step: int = 0,
            large_number: float = 1e6,
            gain_parameter: float = 1.0,
            jerk_cost: float = 1.0,
            x_cost: float = 1.0,
            max_agents: int = 10,
            max_pedestrians: int = 5
            ) -> np.ndarray:
        """
        Convert AUTO interface data to MPC solver parameter format.

        This function converts unified AUTO data (cars+pedestrians in one container, distinguished by object_class)
        into the specific parameter layout expected
        by the casadi MPC solver (107 parameters total).

        Ego State: x, y, yaw, v, ax, delta
        Parameter Layout:
        - Reference: x_ref, y_ref, yaw_ref, vel_ref (4)
        - Agents: 10 agents × 6 positions each = 60 parameters
        - Front agent: x_rear, y_rear, vx, vy (4)
        - Left agent: x_center, y_center, x_rear, y_rear, x_front, y_front, vx, vy (8)
        - Right agent: x_center, y_center, x_rear, y_rear, x_front, y_front, vx, vy (8)
        - Pedestrians: 5 pedestrians × 2 positions each = 10 parameters
        - Agent widths: 10 agent widths (10)
        - Cost parameters: gain_parameter, jerk_cost, x_cost (3)
        Total: 107 parameters

        Args:
            current_state: Current vehicle state
            reference_trajectory: Reference trajectory
            objects: Unified objects (object_class: car=3, pedestrian=1)
            time_step: Time step to extract (default: 0)
            large_number: Large value for missing data (default: 1e6)
            gain_parameter: Gain parameter (default: 1.0)
            jerk_cost: Jerk cost weight (default: 1.0)
            x_cost: X cost weight (default: 1.0)
            max_agents: Maximum number of agents (default: 10)
            max_pedestrians: Maximum number of pedestrians (default: 5)

        Returns:
            np.ndarray: Ego state of shape (6,) [x, y, yaw, v, ax, delta]
            np.ndarray: Parameter vector of shape (107,) for MPC solver
        """

        # Ego State
        ego_state = np.array([
            float(current_state.pose.x),
            float(current_state.pose.y),
            float(current_state.pose.yaw),
            float(current_state.motion.speed_x),
            float(current_state.motion.acceleration_x),
            float(current_state.steering_angle),
        ], dtype=np.float32)

        # Reference Trajectory
        # Build arrays from unified objects (inline, RELATIVE→RELATIVE) and call GetInputParameters
        T = min(int(reference_trajectory.number_of_trajectory_points), self.T)
        # Reference traj (T,4) already local
        ref = np.zeros((T, 4), dtype=np.float32)
        for i in range(T):
            tp = reference_trajectory.trajectory_point[i]
            ref[i] = [tp.x_rel, tp.y_rel, tp.yaw_rel, tp.speed]

        # Agent/ped arrays
        agent_pred = np.full((T, max_agents * 6), large_number, dtype=np.float32)
        agent_widths = np.full((T, max_agents), 2.0, dtype=np.float32)
        ped_pred = np.full((T, max_pedestrians * 2), large_number, dtype=np.float32)

        agent_cursor = 0
        ped_cursor = 0
        n_objects_total = int(getattr(objects, 'number_of_objects', 0))
        for i in range(n_objects_total):
            if agent_cursor >= max_agents and ped_cursor >= max_pedestrians:
                break
            obj = objects.object[i]
            obj_class = int(getattr(obj, 'object_class', 0))
            # local base and local velocities
            x_base = float(obj.x_rel)
            y_base = float(obj.y_rel)
            vx_abs = float(getattr(obj, 'speed_x_rel', 0.0))
            vy_abs = float(getattr(obj, 'speed_y_rel', 0.0))
            if obj_class == 3 and agent_cursor < max_agents: # 3: Car
                yaw_rel = float(getattr(obj, 'yaw_rel', 0.0))
                cy, sy = np.cos(yaw_rel), np.sin(yaw_rel)
                half_len = max(float(getattr(obj, 'length', 4.0)) * 0.5, 0.5)
                # Agent prediction with Constant Velocity Model
                for t in range(T):
                    x_center = x_base + vx_abs * self.dt * t
                    y_center = y_base + vy_abs * self.dt * t
                    x_rear = x_center - half_len * cy
                    y_rear = y_center - half_len * sy
                    x_front = x_center + half_len * cy
                    y_front = y_center + half_len * sy
                    b = agent_cursor * 6
                    agent_pred[t, b:b+6] = [x_center, y_center, x_rear, y_rear, x_front, y_front]
                agent_widths[:, agent_cursor] = float(getattr(obj, 'width', self.width))
                agent_cursor += 1
            elif obj_class == 1 and ped_cursor < max_pedestrians: # 1: Pedestrian
                for t in range(T):
                    x_rel = x_base + vx_abs * self.dt * t
                    y_rel = y_base + vy_abs * self.dt * t
                    b = ped_cursor * 2
                    ped_pred[t, b:b+2] = [x_rel, y_rel]
                ped_cursor += 1

        vehicle_state_np = np.array([
            float(current_state.pose.x),
            float(current_state.pose.y),
            float(current_state.pose.yaw),
            float(current_state.motion.speed_x),
        ], dtype=np.float32)

        raw_output = self.GetInputParameters(
            current_state=vehicle_state_np,
            reference_traj=ref,
            agent_pred_info=agent_pred,
            pedestrian_pred_info=ped_pred,
            agent_width_info=agent_widths,
        )

        return ego_state, ref, agent_pred, ped_pred, agent_widths

    def ConvertMpcToAutoTrajectory(self, x_pred: np.ndarray) -> AUTO_Trajectory:
        """
        Convert MPC trajectory to AUTO trajectory
        Args:
            x_pred: MPC trajectory [T, 6], [x, y, yaw, v, ax, delta]

        Returns:
            AUTO_Trajectory: AUTO trajectory
        """
        T = x_pred.shape[0]
        trajectory = AUTO_Trajectory()
        trajectory.number_of_trajectory_points = T
        delta_time = self.dt

        # Calculate yaw rate (angular velocity) by numerical differentiation
        yaw_rate = np.zeros(T)
        for i in range(T):
            if i == 0:
                # Forward difference for first point
                if T > 1:
                    yaw_rate[i] = (x_pred[1, 2] - x_pred[0, 2]) / delta_time
                else:
                    yaw_rate[i] = 0.0
            elif i == T - 1:
                # Backward difference for last point
                yaw_rate[i] = (x_pred[i, 2] - x_pred[i-1, 2]) / delta_time
            else:
                # Central difference for middle points
                yaw_rate[i] = (x_pred[i+1, 2] - x_pred[i-1, 2]) / (2.0 * delta_time)

        for i in range(T):
            # Calculate curvature using speed and yaw rate (angular velocity)
            speed = max(abs(x_pred[i, 3]), 1e-6)  # Prevent division by zero
            curvature = yaw_rate[i] / speed

            trajectory.trajectory_point[i].x_rel = x_pred[i, 0]
            trajectory.trajectory_point[i].y_rel = x_pred[i, 1]
            trajectory.trajectory_point[i].z_rel = 0.0
            trajectory.trajectory_point[i].yaw_rel = x_pred[i, 2]
            trajectory.trajectory_point[i].yaw_sigma = 0.0
            trajectory.trajectory_point[i].speed = x_pred[i, 3]
            trajectory.trajectory_point[i].acceleration = x_pred[i, 4]
            trajectory.trajectory_point[i].width = 0.0
            trajectory.trajectory_point[i].width_sigma = 0.0
            trajectory.trajectory_point[i].curvature = curvature
            trajectory.trajectory_point[i].lateral_offset = 0.0
            trajectory.trajectory_point[i].decision_reason = 0
            trajectory.trajectory_point[i].time = i * delta_time

        return trajectory

    # Customize pickling to save only essential attributes and reinitialize the object on unpickle
    def __getstate__(self) -> dict:
        return {}

    def __setstate__(self, state: dict) -> None:
        # Reinitialize with default parameters (will reload from YAML)
        self.__init__()

    # ------------------------------------------------------------------ #
    #                           Helper Functions                         #
    # ------------------------------------------------------------------ #
    def CreateAgentCollisionConstraints(self, vehicle: VehiclePositions, agent: AgentTrajectory):
        """
        Create 9 collision constraints for a single agent using parameter objects

        Args:
            vehicle: Vehicle position points (rear, center, front)
            agent: Agent trajectory information (3 points + width)

        Returns:
            List of 9 constraint expressions
        """
        # Distance threshold for collision
        # collision_threshold = (agent.width/2 + self.width_half + self.safety_margin) ** 2
        collision_threshold = agent.width/2 + self.width_half + self.safety_margin
        # collision_threshold = 0.0
        eps = 0.01
        # Constraints 1-3: Rear axle vs agent trajectory points
        con1 = sqrt((vehicle.rear_x - agent.x1)**2 + (vehicle.rear_y - agent.y1)**2 + eps**2) - collision_threshold
        con2 = sqrt((vehicle.rear_x - agent.x2)**2 + (vehicle.rear_y - agent.y2)**2 + eps**2) - collision_threshold
        con3 = sqrt((vehicle.rear_x - agent.x3)**2 + (vehicle.rear_y - agent.y3)**2 + eps**2) - collision_threshold

        # Constraints 4-6: Center vs agent trajectory points
        # con4 = (vehicle.center_x - agent.x1)**2 + (vehicle.center_y - agent.y1)**2 - collision_threshold
        # con5 = (vehicle.center_x - agent.x2)**2 + (vehicle.center_y - agent.y2)**2 - collision_threshold
        # con6 = (vehicle.center_x - agent.x3)**2 + (vehicle.center_y - agent.y3)**2 - collision_threshold

        con4 = sqrt((vehicle.center_x - agent.x1)**2 + (vehicle.center_y - agent.y1)**2 + eps**2) - collision_threshold
        con5 = sqrt((vehicle.center_x - agent.x2)**2 + (vehicle.center_y - agent.y2)**2 + eps**2) - collision_threshold
        con6 = sqrt((vehicle.center_x - agent.x3)**2 + (vehicle.center_y - agent.y3)**2 + eps**2) - collision_threshold

        # Constraints 7-9: Front vs agent trajectory points
        # con7 = (vehicle.front_x - agent.x1)**2 + (vehicle.front_y - agent.y1)**2 - collision_threshold
        # con8 = (vehicle.front_x - agent.x2)**2 + (vehicle.front_y - agent.y2)**2 - collision_threshold
        # con9 = (vehicle.front_x - agent.x3)**2 + (vehicle.front_y - agent.y3)**2 - collision_threshold

        con7 = sqrt((vehicle.front_x - agent.x1)**2 + (vehicle.front_y - agent.y1)**2 + eps**2) - collision_threshold
        con8 = sqrt((vehicle.front_x - agent.x2)**2 + (vehicle.front_y - agent.y2)**2 + eps**2) - collision_threshold
        con9 = sqrt((vehicle.front_x - agent.x3)**2 + (vehicle.front_y - agent.y3)**2 + eps**2) - collision_threshold

        return [con1, con2, con3, con4, con5, con6, con7, con8, con9]


    def CreatePedestrianCollisionConstraints(self, vehicle: VehiclePositions, pedestrian: PedestrianPosition):
        """
        Create 3 collision constraints for a single pedestrian

        Args:
            vehicle: Vehicle position points (rear, center, front)
            pedestrian: Pedestrian position

        Returns:
            List of 3 constraint expressions
        """
        # Distance threshold for pedestrian collision
        # collision_threshold = self.safety_distance_pedestrian ** 2
        collision_threshold = self.safety_distance_pedestrian
        # collision_threshold = 0.0

        # Constraints: Rear, Center, Front vs pedestrian
        # con1 = (vehicle.rear_x - pedestrian.x)**2 + (vehicle.rear_y - pedestrian.y)**2 - collision_threshold
        # con2 = (vehicle.center_x - pedestrian.x)**2 + (vehicle.center_y - pedestrian.y)**2 - collision_threshold
        # con3 = (vehicle.front_x - pedestrian.x)**2 + (vehicle.front_y - pedestrian.y)**2 - collision_threshold
        eps = 0.01
        con1 = sqrt((vehicle.rear_x - pedestrian.x)**2 + (vehicle.rear_y - pedestrian.y)**2 + eps**2)     - collision_threshold
        con2 = sqrt((vehicle.center_x - pedestrian.x)**2 + (vehicle.center_y - pedestrian.y)**2 + eps**2) - collision_threshold
        con3 = sqrt((vehicle.front_x - pedestrian.x)**2 + (vehicle.front_y - pedestrian.y)**2 + eps**2)   - collision_threshold

        return [con1, con2, con3]

    def CreateLinearAccelConstraints(self, vehicle_acceleration: VehicleAcceleration):
        """
        Create linear acceleration constraints for comfort

        Args:
            vehicle_acceleration: Vehicle acceleration components
        """
        L_x_min_1 = 6/5 * SX.cos(7 * np.pi / 12)  / abs(self.min_lon_accel)
        L_x_min_2 = 6/5 * SX.cos(9 * np.pi / 12)  / abs(self.min_lon_accel)
        L_x_min_3 = 6/5 * SX.cos(11 * np.pi / 12) / abs(self.min_lon_accel)

        L_x_max_1 = 6/5 * SX.cos(7 * np.pi / 12) / abs(self.max_lon_accel)
        L_x_max_2 = 6/5 * SX.cos(9 * np.pi / 12) / abs(self.max_lon_accel)
        L_x_max_3 = 6/5 * SX.cos(11 * np.pi / 12) / abs(self.max_lon_accel)

        L_y_1     = SX.sin(7 * np.pi / 12)  / abs(self.max_abs_lat_accel)
        L_y_2     = SX.sin(9 * np.pi / 12)  / abs(self.max_abs_lat_accel)
        L_y_3     = SX.sin(11 * np.pi / 12) / abs(self.max_abs_lat_accel)

        b_xy      = SX.sin((5*np.pi)/12)

        acceleration_con_1 = b_xy - (L_x_min_1 * vehicle_acceleration.ax + L_y_1 * vehicle_acceleration.ay)
        acceleration_con_2 = b_xy - (L_x_min_2 * vehicle_acceleration.ax + L_y_2 * vehicle_acceleration.ay)
        acceleration_con_3 = b_xy - (L_x_min_3 * vehicle_acceleration.ax + L_y_3 * vehicle_acceleration.ay)

        acceleration_con_4 = b_xy - (L_x_min_1 * vehicle_acceleration.ax - L_y_1 * vehicle_acceleration.ay)
        acceleration_con_5 = b_xy - (L_x_min_2 * vehicle_acceleration.ax - L_y_2 * vehicle_acceleration.ay)
        acceleration_con_6 = b_xy - (L_x_min_3 * vehicle_acceleration.ax - L_y_3 * vehicle_acceleration.ay)

        acceleration_con_7 = b_xy - (- L_x_max_1 * vehicle_acceleration.ax + L_y_1 * vehicle_acceleration.ay)
        acceleration_con_8 = b_xy - (- L_x_max_2 * vehicle_acceleration.ax + L_y_2 * vehicle_acceleration.ay)
        acceleration_con_9 = b_xy - (- L_x_max_3 * vehicle_acceleration.ax + L_y_3 * vehicle_acceleration.ay)

        acceleration_con_10 = b_xy - (- L_x_max_1 * vehicle_acceleration.ax - L_y_1 * vehicle_acceleration.ay)
        acceleration_con_11 = b_xy - (- L_x_max_2 * vehicle_acceleration.ax - L_y_2 * vehicle_acceleration.ay)
        acceleration_con_12 = b_xy - (- L_x_max_3 * vehicle_acceleration.ax - L_y_3 * vehicle_acceleration.ay)

        return [acceleration_con_1, acceleration_con_2, acceleration_con_3,
                acceleration_con_4, acceleration_con_5, acceleration_con_6,
                acceleration_con_7, acceleration_con_8, acceleration_con_9,
                acceleration_con_10, acceleration_con_11, acceleration_con_12]

    def GetSolverStats(self) -> dict:
        """
        Returns MPC solver statistics information

        Returns
        -------
        dict
            Solver status, performance and statistics information
        """
        if not self.initialized:
            return {"status": "not_initialized", "error": "Solver not initialized"}

        try:
            # Collect Acados solver statistics information
            stats = {}

            # Basic status information
            if hasattr(self.solver, 'get_status'):
                status_code = self.solver.get_status()
                # Interpret Acados status codes
                status_map = {
                    0: "SUCCESS",
                    1: "FAILURE",
                    2: "MAXITER",
                    3: "MIN_STEP_LENGTH",
                    4: "QP_FAILURE"
                }
                stats["status"] = status_map.get(status_code, f"UNKNOWN_{status_code}")
                stats["status_code"] = status_code
            else:
                stats["status"] = "UNKNOWN"
                stats["status_code"] = None

            # Cost function value
            if hasattr(self.solver, 'get_cost'):
                stats["cost"] = float(self.solver.get_cost())
            else:
                stats["cost"] = None

            # Number of iterations (Acados SQP iterations)
            if hasattr(self.solver, 'get_stats'):
                solver_stats = self.solver.get_stats("sqp_iter")
                if solver_stats is not None:
                    stats["sqp_iterations"] = int(solver_stats)
                else:
                    stats["sqp_iterations"] = None
            else:
                stats["sqp_iterations"] = None

            # Computation time information (if available)
            if hasattr(self.solver, 'get_stats'):
                try:
                    solve_time = self.solver.get_stats("time_tot")
                    if solve_time is not None:
                        stats["solve_time"] = float(solve_time)
                    else:
                        stats["solve_time"] = None
                except:
                    stats["solve_time"] = None
            else:
                stats["solve_time"] = None

            # Residual information
            if hasattr(self.solver, 'get_stats'):
                try:
                    residual = self.solver.get_stats("res_g")
                    if residual is not None:
                        stats["residual"] = float(residual)
                    else:
                        stats["residual"] = None
                except:
                    stats["residual"] = None
            else:
                stats["residual"] = None

            # Determine convergence status
            if stats["status"] == "SUCCESS":
                stats["converged"] = True
            else:
                stats["converged"] = False

            # Solver configuration information
            stats["solver_config"] = {
                "horizon": self.T,
                "dt": self.dt,
                "max_iter": 10,  # Value set in SetupOCP
                "tolerance": 1e-3  # Value set in SetupOCP
            }

            return stats

        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "converged": False
            }
