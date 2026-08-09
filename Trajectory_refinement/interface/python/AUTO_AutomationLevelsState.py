#DART 4.2
from ctypes import *

AUTO_MAX_NUM_AUTOMATION_LEVELS = 8

class AUTO_AutomationLevelsState(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("previous_automation_level", c_uint),
        ("current_automation_level", c_uint),
        ("requested_automation_level_by_driver", c_uint),
        ("requested_automation_level_by_automation", c_uint),
        ("arbitrated_automation_level_request", c_uint),
        ("previous_transition", c_uint),
        ("current_transition", c_uint),
        ("previous_requested_transition", c_uint),
        ("current_requested_transition", c_uint),
        ("time_to_transition", c_uint32),
        ("automation_levels_availability", c_uint32 * AUTO_MAX_NUM_AUTOMATION_LEVELS),
    ]