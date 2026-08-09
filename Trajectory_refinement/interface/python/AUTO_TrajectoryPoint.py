#DART 4.2
from ctypes import *

class AUTO_TrajectoryPoint(Structure):
    _pack_ = 1
    _fields_ = [
        ("x_rel", c_float),
        ("y_rel", c_float),
        ("z_rel", c_float),
        ("yaw_rel", c_float),
        ("yaw_sigma", c_float),
        ("speed", c_float),
        ("acceleration", c_float),
        ("width", c_float),
        ("width_sigma", c_float),
        ("curvature", c_float),
        ("lateral_offset", c_float),
        ("decision_reason", c_uint),
        ("time", c_float),
    ]