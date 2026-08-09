#DART 4.2
from AUTO_LLA import AUTO_LLA
from ctypes import *

class AUTO_RoutePoint(Structure):
    _pack_ = 1
    _fields_ = [
        ("lla", AUTO_LLA),
        ("event", c_uint),
        ("distance_cost", c_float),
        ("time_cost", c_float),
        ("fare_cost", c_float),
        ("energy_cost", c_float),
    ]
