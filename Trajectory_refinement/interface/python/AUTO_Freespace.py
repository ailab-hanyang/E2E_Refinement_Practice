#DART 4.2
from AUTO_FreespacePoint import AUTO_FreespacePoint
from AUTO_Gnss import AUTO_Gnss
from AUTO_Pose import AUTO_Pose
from ctypes import *

AUTO_MAX_NUM_FREESPACE_POINTS = 1024

class AUTO_Freespace(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("id", c_uint16),
        ("pose", AUTO_Pose),
        ("gnss", AUTO_Gnss),
        ("surface_class", c_uint),
        ("surface_class_confidence", c_float),
        ("number_of_freespace_points", c_uint16),
        ("freespace_point", AUTO_FreespacePoint * AUTO_MAX_NUM_FREESPACE_POINTS),
    ]
