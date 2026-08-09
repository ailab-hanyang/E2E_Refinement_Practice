#DART 4.2
from ctypes import *

class AUTO_Country(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("country_code", c_uint32),
        ("region_code", c_uint32),
        ("driving_on_left", c_bool),
        ("speed_mph", c_bool),
    ]