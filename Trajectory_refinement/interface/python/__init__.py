import sys
import os

# 현재 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Level 0: 의존성 없는 기본 클래스들
from .AUTO_ActuatorsSubSystems import AUTO_ActuatorsSubSystems
from .AUTO_AutomationLevelsState import AUTO_AutomationLevelsState
from .AUTO_Command import AUTO_Command
from .AUTO_Country import AUTO_Country
from .AUTO_DriverRequests import AUTO_DriverRequests
from .AUTO_DriverState import AUTO_DriverState
from .AUTO_Feedback import AUTO_Feedback
from .AUTO_FreespacePoint import AUTO_FreespacePoint
from .AUTO_Gnss import AUTO_Gnss
from .AUTO_LanePoint import AUTO_LanePoint
from .AUTO_Line import AUTO_Line
from .AUTO_LLA import AUTO_LLA
from .AUTO_Manoeuvre import AUTO_Manoeuvre
from .AUTO_Motion import AUTO_Motion
from .AUTO_Pose import AUTO_Pose
from .AUTO_SupplementarySign import AUTO_SupplementarySign
from .AUTO_SystemTime import AUTO_SystemTime
from .AUTO_TrafficLightPhase import AUTO_TrafficLightPhase
from .AUTO_TrajectoryPoint import AUTO_TrajectoryPoint

# Level 1: Level 0에 의존하는 클래스들
from .AUTO_ActuatorsVector import AUTO_ActuatorsVector
from .AUTO_AudioFeedback import AUTO_AudioFeedback
from .AUTO_EnvironmentalConditions import AUTO_EnvironmentalConditions
from .AUTO_HapticFeedback import AUTO_HapticFeedback
from .AUTO_Lines import AUTO_Lines
from .AUTO_Manoeuvres import AUTO_Manoeuvres
from .AUTO_MotionVector import AUTO_MotionVector
from .AUTO_Object import AUTO_Object
from .AUTO_RoutePoint import AUTO_RoutePoint
from .AUTO_VehicleState import AUTO_VehicleState
from .AUTO_Waypoint import AUTO_Waypoint

# Level 2: Level 1에 의존하는 클래스들
from .AUTO_AudioUserInterface import AUTO_AudioUserInterface
from .AUTO_Freespace import AUTO_Freespace
from .AUTO_HapticUserInterface import AUTO_HapticUserInterface
from .AUTO_Lane import AUTO_Lane
from .AUTO_Objects import AUTO_Objects
from .AUTO_Route import AUTO_Route
from .AUTO_TrafficLight import AUTO_TrafficLight
from .AUTO_TrafficSign import AUTO_TrafficSign
from .AUTO_Trajectory import AUTO_Trajectory
from .AUTO_Trajectories import AUTO_Trajectories

# Level 3: Level 2에 의존하는 클래스들
from .AUTO_Lanes import AUTO_Lanes
from .AUTO_TrafficLights import AUTO_TrafficLights
from .AUTO_TrafficSigns import AUTO_TrafficSigns

# Level 4: 최상위 클래스들
from .AUTO_EnvironmentState import AUTO_EnvironmentState
from .AUTO_GraphicalUserInterface import AUTO_GraphicalUserInterface
