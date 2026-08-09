#DART 4.2
from AUTO_Feedback import AUTO_Feedback
from ctypes import *

class AUTO_AudioFeedback(Structure):
    _pack_ = 1
    _fields_ = [
        ("audio_device", c_uint),
        ("audio_feedback", AUTO_Feedback),
    ]
