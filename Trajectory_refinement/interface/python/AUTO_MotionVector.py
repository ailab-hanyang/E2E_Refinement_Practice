#DART 4.2
from AUTO_Command import AUTO_Command
from ctypes import *

class AUTO_MotionVector(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("automation_level", c_uint),
        ("acceleration", AUTO_Command),
        ("speed_reference", c_float),
        ("steering", AUTO_Command),
        ("lateral_offset", c_float),
        ("manoeuvre_type", c_uint),
    ]
