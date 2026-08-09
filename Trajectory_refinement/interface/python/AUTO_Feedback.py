#DART 4.2
from ctypes import *

class AUTO_Feedback(Structure):
    _pack_ = 1
    _fields_ = [
        ("feedback_type", c_uint),
        ("intensity", c_float),
        ("frequency", c_float),
        ("pwm_frequency", c_float),
        ("pwm_duty_cycle", c_float),
        ("feedback_position", c_uint),
        ("x_rel", c_float),
        ("y_rel", c_float),
    ]