#DART 4.2
from AUTO_LLA import AUTO_LLA
from AUTO_SupplementarySign import AUTO_SupplementarySign
from AUTO_Gnss import AUTO_Gnss
from AUTO_Pose import AUTO_Pose
from ctypes import *

AUTO_MAX_NUM_LANES = 5

class AUTO_TrafficSign(Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", c_uint8),
        ("version_minor", c_uint8),
        ("status", c_uint),
        ("timestamp", c_uint64),
        ("id", c_uint16),
        ("pose", AUTO_Pose),
        ("gnss", AUTO_Gnss),
        ("value", c_char * 100),
        ("supplementary_sign", AUTO_SupplementarySign),
        ("sign_shape", c_uint),
        ("ground_colour", c_uint),
        ("border_colour", c_uint),
        ("sign_class", c_uint),
        ("class_age", c_uint16),
        ("class_confidence", c_float),
        ("track_age", c_uint16),
        ("track_confidence", c_float),
        ("x_rel", c_float),
        ("y_rel", c_float),
        ("z_rel", c_float),
        ("width", c_float),
        ("height", c_float),
        ("distance_to_ground", c_float),
        ("distance", c_float),
        ("a_sigma", c_float),
        ("b_sigma", c_float),
        ("theta_angle", c_float),
        ("z_sigma", c_float),
        ("width_sigma", c_float),
        ("height_sigma", c_float),
        ("distance_to_ground_sigma", c_float),
        ("lane_point_lla", AUTO_LLA * AUTO_MAX_NUM_LANES),
    ]
