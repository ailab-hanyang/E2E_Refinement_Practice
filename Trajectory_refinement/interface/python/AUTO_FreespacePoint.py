#DART 4.2
from ctypes import *

class AUTO_FreespacePoint(Structure):
    _pack_ = 1
    _fields_ = [
        ("freespace_point_class", c_uint),
        ("x_rel", c_float),
        ("y_rel", c_float),
        ("z_rel", c_float),
        ("a_sigma", c_float),
        ("b_sigma", c_float),
        ("theta_angle", c_float),
        ("z_sigma", c_float),
    ]