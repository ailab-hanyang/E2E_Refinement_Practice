#DART 4.2
from AUTO_Manoeuvre import AUTO_Manoeuvre
from ctypes import *

AUTO_MAX_NUM_MANOEUVRES = 9

class AUTO_Manoeuvres(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("number_of_manoeuvres", c_uint16),
        ("manoeuvre", AUTO_Manoeuvre * AUTO_MAX_NUM_MANOEUVRES),
    ]
