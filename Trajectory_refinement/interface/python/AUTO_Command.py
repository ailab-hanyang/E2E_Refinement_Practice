#DART 4.2
from ctypes import *

class AUTO_Command(Structure):
    _pack_ = 1
    _fields_ = [
        ("command", c_float),
        ("status", c_uint),
    ]