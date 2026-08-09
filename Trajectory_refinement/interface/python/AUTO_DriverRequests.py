#DART 4.2
from ctypes import *

AUTO_NUMBER_OF_WIPERS = 2
AUTO_MAX_NUM_AUTOMATION_LEVELS = 8

class AUTO_DriverRequests(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("selected_automation_level", c_bool * AUTO_MAX_NUM_AUTOMATION_LEVELS),
        ("set_automation_level", c_uint),
        ("set_driving_style", c_uint),
        ("set_speed", c_float),
        ("set_time_gap", c_float),
        ("set_lateral_offset", c_float),
        ("steering_torque", c_float),
        ("throttle_pedal_force", c_float),
        ("throttle_pedal_position", c_float),
        ("brake_pedal_force", c_float),
        ("brake_pedal_position", c_float),
        ("transmission_clutch", c_uint),
        ("transmission_gear", c_uint),
        ("ignition_on", c_bool),
        ("engine_on", c_bool),
        ("wipers", c_uint * AUTO_NUMBER_OF_WIPERS),
        ("horn_on", c_bool),
        ("headlamps", c_uint),
        ("turn_signal", c_uint),
    ]