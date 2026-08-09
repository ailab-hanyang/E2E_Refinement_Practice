#DART 4.2
from AUTO_TrajectoryPoint import AUTO_TrajectoryPoint
from AUTO_Manoeuvre import AUTO_Manoeuvre
from AUTO_Gnss import AUTO_Gnss
from AUTO_Pose import AUTO_Pose
from ctypes import *

AUTO_MAX_NUM_TRAJECTORY_POINTS = 500

class AUTO_Trajectory(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("id", c_uint16),
        ("pose", AUTO_Pose),
        ("gnss", AUTO_Gnss),
        ("valential", c_float),
        ("manoeuvre", AUTO_Manoeuvre),
        ("number_of_trajectory_points", c_uint16),
        ("trajectory_point", AUTO_TrajectoryPoint * AUTO_MAX_NUM_TRAJECTORY_POINTS),
    ]
