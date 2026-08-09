#DART 4.2
from ctypes import *

class AUTO_LLA(Structure):
    _pack_ = 1
    _fields_ = [
        ("latitude", c_double),
        ("longitude", c_double),
        ("altitude", c_float),
    ]