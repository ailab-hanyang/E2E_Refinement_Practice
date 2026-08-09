#DART 4.2
from ctypes import *

class AUTO_Pose(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("roll", c_float),
        ("pitch", c_float),
        ("yaw", c_float),
    ]