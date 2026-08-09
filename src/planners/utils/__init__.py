"""RefinementPlanner 가 쓰는 보조 모듈.

- agent_world  — 로그(GT) agent 미래를 ego-local 배열로 변환. numpy 만 의존한다.
- mpc_interface — acados MPC 전처리 + solve. 여기만 Trajectory_refinement 를 import 한다.

mpc_interface 는 여기서 import 하지 않는다. acados 가 없는 환경을 위해
planner 가 try 안에서 직접 지연 import 한다.
"""

from .agent_world import agents_to_local_info

__all__ = ["agents_to_local_info"]
