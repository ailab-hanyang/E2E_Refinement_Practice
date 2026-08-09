#DART 4.2
from AUTO_Lane import AUTO_Lane
from ctypes import *

AUTO_MAX_NUM_LANES = 5

class AUTO_Lanes(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("driving_on_left", c_bool),
        ("road_speed_limit", c_float),
        ("lane_change", c_uint16),
        ("primary_lane_index", c_uint16),
        ("number_of_lanes", c_uint16),
        ("lane", AUTO_Lane * AUTO_MAX_NUM_LANES),
    ]
