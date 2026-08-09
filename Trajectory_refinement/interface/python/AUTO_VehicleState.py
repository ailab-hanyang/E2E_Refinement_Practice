#DART 4.2
from AUTO_Motion import AUTO_Motion
from AUTO_Gnss import AUTO_Gnss
from AUTO_Pose import AUTO_Pose
from ctypes import *

AUTO_NUMBER_OF_WIPERS = 2
AUTO_NUMBER_OF_DOORS = 5

class AUTO_VehicleState(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("id", c_char * 255),
        ("pose", AUTO_Pose),
        ("gnss", AUTO_Gnss),
        ("motion", AUTO_Motion),
        ("wheel_ticks_front_left", c_uint16),
        ("wheel_ticks_front_right", c_uint16),
        ("wheel_ticks_rear_left", c_uint16),
        ("wheel_ticks_rear_right", c_uint16),
        ("wheel_direction_front_left", c_uint),
        ("wheel_direction_front_right", c_uint),
        ("wheel_direction_rear_left", c_uint),
        ("wheel_direction_rear_right", c_uint),
        ("distance", c_float),
        ("front_wheels_angle", c_float),
        ("steering_angle", c_float),
        ("steering_torque", c_float),
        ("throttle_pedal_position", c_float),
        ("brake_pedal_position", c_float),
        ("brake_pedal_pressure", c_float),
        ("driving_style", c_uint),
        ("steering_clutch", c_uint),
        ("transmission_clutch", c_uint),
        ("transmission_gear", c_uint),
        ("transmission_gear_number", c_uint16),
        ("ignition_on", c_uint),
        ("engine_on", c_uint),
        ("engine_rpm", c_uint16),
        ("engine_torque", c_float),
        ("engine_temperature", c_float),
        ("energy_level", c_float),
        ("standstill", c_uint),
        ("door_open", c_uint * AUTO_NUMBER_OF_DOORS),
        ("wipers", c_uint * AUTO_NUMBER_OF_WIPERS),
        ("horn_on", c_uint),
        ("headlamps", c_uint),
        ("turn_signal", c_uint),
        ("brake_signal", c_uint),
        ("abs_status", c_uint),
        ("esp_status", c_uint),
        ("epb_status", c_uint),
        ("static_friction_coefficient", c_float),
    ]
