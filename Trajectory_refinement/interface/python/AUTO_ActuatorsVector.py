#DART 4.2
from AUTO_Command import AUTO_Command
from ctypes import *

class AUTO_ActuatorsVector(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("brake_pedal", AUTO_Command),
        ("throttle_pedal", AUTO_Command),
        ("steering_wheel", AUTO_Command),
    ]
