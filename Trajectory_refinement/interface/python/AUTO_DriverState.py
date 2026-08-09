#DART 4.2
from ctypes import *

class AUTO_DriverState(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("id", c_char * 255),
        ("driver_present", c_bool),
        ("seatbelt_on", c_bool),
        ("hands_on_wheel", c_uint),
        ("feet_on_pedal", c_uint),
        ("drowsiness_level", c_uint),
        ("drowsiness_level_confidence", c_float),
        ("inattention_level", c_uint),
        ("inattention_level_confidence", c_float),
    ]