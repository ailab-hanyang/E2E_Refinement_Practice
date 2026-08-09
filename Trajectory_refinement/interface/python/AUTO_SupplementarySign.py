#DART 4.2
from ctypes import *

class AUTO_SupplementarySign(Structure):
    _pack_ = 1
    _fields_ = [
        ("id", c_uint16),
        ("value", c_char * 100),
        ("confidence", c_float),
    ]