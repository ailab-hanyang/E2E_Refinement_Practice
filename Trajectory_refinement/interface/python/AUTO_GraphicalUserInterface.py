#DART 4.2
from AUTO_Motion import AUTO_Motion
from AUTO_MotionVector import AUTO_MotionVector
from AUTO_AutomationLevelsState import AUTO_AutomationLevelsState
from AUTO_Trajectories import AUTO_Trajectories
from AUTO_Manoeuvres import AUTO_Manoeuvres
from AUTO_Manoeuvre import AUTO_Manoeuvre
from AUTO_Route import AUTO_Route
from AUTO_DriverRequests import AUTO_DriverRequests
from AUTO_EnvironmentState import AUTO_EnvironmentState
from AUTO_VehicleState import AUTO_VehicleState
from AUTO_DriverState import AUTO_DriverState
from ctypes import *

class AUTO_GraphicalUserInterface(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("driver_state", AUTO_DriverState),
        ("vehicle_state", AUTO_VehicleState),
        ("environment_state", AUTO_EnvironmentState),
        ("driver_requests", AUTO_DriverRequests),
        ("route", AUTO_Route),
        ("manoeuvres", AUTO_Manoeuvres),
        ("trajectories", AUTO_Trajectories),
        ("automation_levels_state", AUTO_AutomationLevelsState),
        ("motion_vector", AUTO_MotionVector),
        ("warning", c_uint16),
    ]
