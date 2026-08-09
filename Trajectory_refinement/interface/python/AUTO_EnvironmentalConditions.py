#DART 4.2
from AUTO_SystemTime import AUTO_SystemTime
from ctypes import *

class AUTO_EnvironmentalConditions(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("weather", c_uint),
        ("illuminance", c_float),
        ("temperature", c_float),
        ("relative_humidity", c_float),
        ("air_quality", c_uint),
        ("system_time", AUTO_SystemTime),
    ]
