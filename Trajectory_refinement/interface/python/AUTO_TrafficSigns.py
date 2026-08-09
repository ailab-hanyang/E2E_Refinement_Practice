#DART 4.2
from AUTO_TrafficSign import AUTO_TrafficSign
from ctypes import *

AUTO_MAX_NUM_TRAFFIC_SIGNS = 20

class AUTO_TrafficSigns(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("number_of_traffic_signs", c_uint16),
        ("traffic_sign", AUTO_TrafficSign * AUTO_MAX_NUM_TRAFFIC_SIGNS),
    ]
