#DART 4.2
from ctypes import *

class AUTO_LanePoint(Structure):
    _pack_ = 1
    _fields_ = [
        ("tunnel", c_bool),
        ("multiple_carriageway", c_bool),
        ("road_type", c_uint),
        ("surface", c_uint),
        ("lane_use", c_uint),
        ("lane_marking_class_left", c_uint),
        ("lane_marking_colour_left", c_uint),
        ("lane_marking_width_left", c_float),
        ("lane_marking_class_right", c_uint),
        ("lane_marking_colour_right", c_uint),
        ("lane_marking_width_right", c_float),
        ("lane_marking_direction", c_uint),
        ("lane_marking_class_center", c_uint),
        ("lane_marking_colour_center", c_uint),
        ("lane_marking_width_center", c_float),
        ("lane_direction", c_uint),
        ("x_rel", c_float),
        ("y_rel", c_float),
        ("z_rel", c_float),
        ("yaw_rel", c_float),
        ("width", c_float),
        ("curvature", c_float),
        ("slope", c_float),
        ("speed_limit", c_float),
        ("a_sigma", c_float),
        ("b_sigma", c_float),
        ("theta_angle", c_float),
        ("z_sigma", c_float),
        ("width_sigma", c_float),
        ("yaw_sigma", c_float),
        ("curvature_sigma", c_float),
    ]