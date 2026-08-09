#DART 4.2
from AUTO_RoutePoint import AUTO_RoutePoint
from ctypes import *

AUTO_MAX_NUM_ROUTE_POINTS = 10000

class AUTO_Route(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("cost_criterion", c_uint),
        ("number_of_route_points", c_uint16),
        ("route_point", AUTO_RoutePoint * AUTO_MAX_NUM_ROUTE_POINTS),
    ]
