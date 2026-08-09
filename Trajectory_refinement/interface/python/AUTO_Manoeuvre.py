#DART 4.2
from ctypes import *

class AUTO_Manoeuvre(Structure):
    _pack_ = 1
    _fields_ = [
        ("manoeuvre_type", c_uint),
        ("valential", c_float),
    ]