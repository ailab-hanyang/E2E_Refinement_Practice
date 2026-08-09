#DART 4.2
from ctypes import *

class AUTO_Gnss(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("latitude", c_double),
        ("longitude", c_double),
        ("altitude", c_float),
        ("roll", c_float),
        ("pitch", c_float),
        ("heading", c_float),
        ("latitude_sigma", c_float),
        ("longitude_sigma", c_float),
        ("altitude_sigma", c_float),
        ("roll_sigma", c_float),
        ("pitch_sigma", c_float),
        ("heading_sigma", c_float),
        ("a_sigma", c_float),
        ("b_sigma", c_float),
        ("c_sigma", c_float),
        ("quality", c_uint),
        ("number_of_satellites", c_uint32),
        ("hdop", c_float),
        ("vdop", c_float),
        ("hpl", c_float),
        ("vpl", c_float),
    ]