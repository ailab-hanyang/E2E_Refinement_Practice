#DART 4.2
from AUTO_TrafficLight import AUTO_TrafficLight
from ctypes import *

AUTO_MAX_NUM_TRAFFIC_LIGHTS = 20

class AUTO_TrafficLights(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("primary_traffic_light_index", c_uint16),
        ("number_of_traffic_lights", c_uint16),
        ("traffic_light", AUTO_TrafficLight * AUTO_MAX_NUM_TRAFFIC_LIGHTS),
    ]
