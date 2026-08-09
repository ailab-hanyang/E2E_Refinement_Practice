#DART 4.2
from AUTO_Freespace import AUTO_Freespace
from AUTO_Lanes import AUTO_Lanes
from AUTO_Lane import AUTO_Lane
from AUTO_TrafficSigns import AUTO_TrafficSigns
from AUTO_TrafficSign import AUTO_TrafficSign
from AUTO_TrafficLights import AUTO_TrafficLights
from AUTO_TrafficLight import AUTO_TrafficLight
from AUTO_Objects import AUTO_Objects
from AUTO_Object import AUTO_Object
from AUTO_EnvironmentalConditions import AUTO_EnvironmentalConditions
from AUTO_Country import AUTO_Country
from ctypes import *

class AUTO_EnvironmentState(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("country", AUTO_Country),
        ("environmental_conditions", AUTO_EnvironmentalConditions),
        ("objects", AUTO_Objects),
        ("traffic_lights", AUTO_TrafficLights),
        ("traffic_signs", AUTO_TrafficSigns),
        ("lanes", AUTO_Lanes),
        ("freespace", AUTO_Freespace),
    ]
