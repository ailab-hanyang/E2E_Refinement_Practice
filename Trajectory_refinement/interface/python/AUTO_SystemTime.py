#DART 4.2
from ctypes import *

class AUTO_SystemTime(Structure):
    _pack_ = 1
    _fields_ = [
        ("year", c_uint32),
        ("month", c_uint32),
        ("day_of_week", c_uint32),
        ("day", c_uint32),
        ("hour", c_uint32),
        ("minute", c_uint32),
        ("second", c_uint32),
        ("millisecond", c_uint32),
    ]