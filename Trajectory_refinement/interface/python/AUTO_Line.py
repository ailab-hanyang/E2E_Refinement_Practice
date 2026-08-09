#DART 4.2
from ctypes import *

class AUTO_Line(Structure):
    _pack_ = 1
    _fields_ = [
        ("available", c_bool),
        ("x_rel", c_float),
        ("y_rel", c_float),
        ("z_rel", c_float),
        ("yaw_rel", c_float),
        ("width", c_float),
        ("curvature", c_float),
        ("slope", c_float),
        ("a_sigma", c_float),
        ("b_sigma", c_float),
        ("theta_angle", c_float),
        ("z_sigma", c_float),
        ("width_sigma", c_float),
        ("yaw_sigma", c_float),
        ("curvature_sigma", c_float),
    ]