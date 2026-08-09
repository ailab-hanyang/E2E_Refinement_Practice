#DART 4.2
from AUTO_Line import AUTO_Line
from ctypes import *

class AUTO_Lines(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("lane_change", c_uint16),
        ("primary_lane_index", c_uint16),
        ("left_lane_center_line", AUTO_Line * 2),
        ("current_lane_left_line", AUTO_Line),
        ("current_lane_center_line", AUTO_Line * 2),
        ("current_lane_right_line", AUTO_Line),
        ("right_lane_center_line", AUTO_Line * 2),
    ]
