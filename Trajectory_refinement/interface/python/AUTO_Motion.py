#DART 4.2
from ctypes import *

class AUTO_Motion(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("speed_x", c_float),
        ("speed_y", c_float),
        ("speed_z", c_float),
        ("acceleration_x", c_float),
        ("acceleration_y", c_float),
        ("acceleration_z", c_float),
        ("roll_rate", c_float),
        ("pitch_rate", c_float),
        ("yaw_rate", c_float),
        ("speed_x_sigma", c_float),
        ("speed_y_sigma", c_float),
        ("speed_z_sigma", c_float),
        ("acceleration_x_sigma", c_float),
        ("acceleration_y_sigma", c_float),
        ("acceleration_z_sigma", c_float),
        ("roll_rate_sigma", c_float),
        ("pitch_rate_sigma", c_float),
        ("yaw_rate_sigma", c_float),
    ]