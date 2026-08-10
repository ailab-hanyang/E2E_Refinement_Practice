"""ML 궤적 후처리 모듈.

후처리 분류 지도(실습 3 §1)에서 가장 왼쪽 계열 — **Smoothing** 만 여기에 있다.
Projection 계열(MPC)은 `src/planners/utils/mpc_interface.py` 가 담당한다.

두 계열을 굳이 다른 패키지에 둔 이유는 의존성이다. 이쪽은 numpy 와 scipy 만 있으면
동작하므로 acados 없이도, nuPlan devkit 없이도 import 된다.
"""

from .lowpass import (  # noqa: F401
    SMOOTHING_MODES,
    LowPassTrajectoryFilter,
    curvature,
    heading_rate,
    make_smoother,
)

__all__ = [
    "LowPassTrajectoryFilter",
    "make_smoother",
    "SMOOTHING_MODES",
    "curvature",
    "heading_rate",
]
