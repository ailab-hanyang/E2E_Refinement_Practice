#DART 4.2
from AUTO_HapticFeedback import AUTO_HapticFeedback
from ctypes import *

AUTO_MAX_NUM_HAPTIC_DEVICES = 7

class AUTO_HapticUserInterface(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("number_of_haptic_feedback", c_uint16),
        ("haptic_feedback", AUTO_HapticFeedback * AUTO_MAX_NUM_HAPTIC_DEVICES),
    ]
