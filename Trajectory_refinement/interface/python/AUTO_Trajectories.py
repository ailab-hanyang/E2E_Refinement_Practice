#DART 4.2
from AUTO_Trajectory import AUTO_Trajectory
from ctypes import *

AUTO_MAX_NUM_TRAJECTORIES = 9

class AUTO_Trajectories(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("number_of_trajectories", c_uint16),
        ("trajectory", AUTO_Trajectory * AUTO_MAX_NUM_TRAJECTORIES),
    ]
