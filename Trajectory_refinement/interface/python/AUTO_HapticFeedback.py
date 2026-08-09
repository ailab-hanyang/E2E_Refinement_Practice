#DART 4.2
from AUTO_Feedback import AUTO_Feedback
from ctypes import *

class AUTO_HapticFeedback(Structure):
    _pack_ = 1
    _fields_ = [
        ("haptic_device", c_uint),
        ("haptic_feedback", AUTO_Feedback),
    ]
