#DART 4.2
from AUTO_Object import AUTO_Object
from ctypes import *

AUTO_MAX_NUM_OBJECTS = 255

class AUTO_Objects(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("primary_object_index", c_uint16),
        ("number_of_objects", c_uint16),
        ("object", AUTO_Object * AUTO_MAX_NUM_OBJECTS),
    ]
