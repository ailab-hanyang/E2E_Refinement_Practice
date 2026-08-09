#DART 4.2
from AUTO_LanePoint import AUTO_LanePoint
from AUTO_Gnss import AUTO_Gnss
from AUTO_Pose import AUTO_Pose
from ctypes import *

AUTO_MAX_NUM_LANE_POINTS = 1000

class AUTO_Lane(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("id", c_uint16),
        ("pose", AUTO_Pose),
        ("gnss", AUTO_Gnss),
        ("number_of_lane_points", c_uint16),
        ("lane_point", AUTO_LanePoint * AUTO_MAX_NUM_LANE_POINTS),
    ]
