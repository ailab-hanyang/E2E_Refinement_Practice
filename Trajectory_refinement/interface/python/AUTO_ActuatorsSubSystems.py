#DART 4.2
from ctypes import *

AUTO_NUMBER_OF_WIPERS = 2
AUTO_NUMBER_OF_DOORS = 5

class AUTO_ActuatorsSubSystems(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("automation_level", c_uint),
        ("steering_clutch", c_uint),
        ("transmission_clutch", c_uint),
        ("transmission_gear", c_uint),
        ("ignition_on", c_uint),
        ("engine_on", c_uint),
        ("door_open", c_uint * AUTO_NUMBER_OF_DOORS),
        ("wipers", c_uint * AUTO_NUMBER_OF_WIPERS),
        ("epb", c_uint),
    ]