#DART 4.2
from AUTO_AudioFeedback import AUTO_AudioFeedback
from ctypes import *

AUTO_MAX_NUM_AUDIO_DEVICES = 5

class AUTO_AudioUserInterface(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("number_of_interior_audio_feedback", c_uint16),
        ("number_of_exterior_audio_feedback", c_uint16),
        ("interior_audio_feedback", AUTO_AudioFeedback * AUTO_MAX_NUM_AUDIO_DEVICES),
        ("exterior_audio_feedback", AUTO_AudioFeedback * AUTO_MAX_NUM_AUDIO_DEVICES),
    ]
