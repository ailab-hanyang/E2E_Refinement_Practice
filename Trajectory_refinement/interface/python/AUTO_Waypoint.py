#DART 4.2
from AUTO_LLA import AUTO_LLA
from ctypes import *

class AUTO_Waypoint(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("lla", AUTO_LLA),
        ("cost_criterion", c_uint),
    ]
