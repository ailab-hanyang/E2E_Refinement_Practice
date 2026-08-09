#DART 4.2
from ctypes import *

AUTO_MAX_NUM_TRAFFIC_LIGHT_ARROWS = 3

class AUTO_TrafficLightPhase(Structure):
    _pack_ = 1
    _fields_ = [
        ("flashing_on", c_bool),
        ("light_shape", c_uint),
        ("light_colour", c_uint),
        ("light_arrow", c_uint * AUTO_MAX_NUM_TRAFFIC_LIGHT_ARROWS),
        ("time_to_next_phase", c_uint32),
    ]