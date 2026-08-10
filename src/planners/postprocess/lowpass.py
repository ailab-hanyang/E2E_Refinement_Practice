"""궤적 Smoothing 후처리 — 저역통과 필터.

후처리 분류 지도에서 **개입이 가장 약한 계열**이다. 궤적을 시계열로 보고 고주파
성분만 깎는다. 지도도 주변 차량도 보지 않으므로 충돌 회피나 주행가능영역 준수는
**보장하지 못한다** — 그것은 Projection 계열(MPC)의 몫이다.

무엇을 고치는가:

    ML planner 의 출력에는 waypoint 단위의 미세한 떨림이 남는다. 궤적을 그대로
    추종하면 곡률이 스텝마다 튀고, 그 곡률이 조향으로 들어가 lateral jerk 와
    yaw rate 를 흔든다. nuPlan 의 comfort 항이 잡아내는 것이 정확히 이 부분이다.

무엇을 못 고치는가:

    - 궤적이 차선 밖을 향하고 있으면 필터를 걸어도 여전히 밖을 향한다
    - 앞차와 겹치는 궤적은 부드러운 채로 여전히 겹친다
    - 프레임 간 불일치(매 스텝 다른 궤적을 내는 것)는 한 궤적 안의 문제가 아니라
      스텝 사이의 문제라 여기서 손대지 않는다 (MPC 의 warm start 가 다루는 영역)

구현상의 선택 두 가지:

1. **위치가 아니라 증분(diff)을 필터링한다.**
   위치 x 는 시간에 대해 거의 직선으로 증가하는 램프 신호다. 램프를 인과 필터에
   통과시키면 정상상태 지연이 그대로 남아 궤적 전체가 뒤로 밀리고, 결과적으로
   차가 느려진다. 증분은 거의 상수(≈ v·dt)라 이 문제가 없다 — 필터의 DC 이득이
   1 이므로 증분의 평균이 보존되고, 따라서 **총 이동거리가 보존된다.**
   덤으로 시작점이 정확히 유지된다(원점에서 cumsum 하므로).

2. **yaw 를 x, y 와 독립으로 필터링한다.**
   부드러워진 경로에서 yaw = atan2(dy, dx) 로 다시 만드는 방법도 있지만, 저속에서
   dx, dy 가 0 에 가까워지면 yaw 가 발산한다. 채점기와 시뮬레이터 모두 궤적을
   LQR + 자전거 모델로 추종하므로 약간의 불일치는 추종 단계에서 흡수된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: 지원하는 필터 종류. "none" 은 후처리를 끄는 값이다(분기를 밖에 두지 않기 위해).
SMOOTHING_MODES = ("none", "zerophase", "causal", "savgol")

#: 필터를 걸기에 표본이 너무 적으면 원본을 그대로 돌려준다.
_MIN_SAMPLES = 8


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """각도를 (-π, π] 로 접는다."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class LowPassTrajectoryFilter:
    """궤적 하나를 받아 고주파 성분을 깎아 돌려준다.

    상태를 갖지 않는다. 같은 입력에는 항상 같은 출력이 나오므로 스텝마다 새로
    만들 필요도, 시나리오 사이에 초기화할 필요도 없다.

    :param dt: 웨이포인트 간격 [s]. 차단주파수의 기준이다.
    :param cutoff_hz: 차단주파수 [Hz]. 낮출수록 부드러워지고, 낮출수록 급격한
        회피 기동도 함께 뭉개진다. dt=0.1 이면 Nyquist 가 5 Hz 이므로 그보다
        작아야 한다.
    :param mode: SMOOTHING_MODES 중 하나.
        ``zerophase`` filtfilt — 정방향·역방향 두 번 통과. 위상 지연이 없다.
                                 계획 궤적은 전 구간이 이미 손에 있으므로
                                 (실시간 신호와 달리) 이것이 가능하다.
        ``causal``    lfilter  — 정방향 한 번. 위상 지연이 남는다. 실시간
                                 제어 신호에 필터를 걸면 무슨 일이 생기는지
                                 보이기 위한 비교용이다.
        ``savgol``    Savitzky-Golay — 국소 다항식 적합. 첨두값을 덜 깎는다.
                                 nuPlan 이 jerk 를 뽑을 때 쓰는 필터이기도 하다.
    :param order: Butterworth 차수 (zerophase / causal).
    :param savgol_window: Savitzky-Golay 창 길이. 홀수여야 한다.
    :param savgol_polyorder: Savitzky-Golay 다항식 차수.
    """

    dt: float = 0.1
    cutoff_hz: float = 1.0
    mode: str = "zerophase"
    order: int = 2
    savgol_window: int = 11
    savgol_polyorder: int = 3

    def __post_init__(self) -> None:
        if self.mode not in SMOOTHING_MODES:
            raise ValueError(
                f"mode must be one of {SMOOTHING_MODES}, got {self.mode!r}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        if self.mode in ("zerophase", "causal"):
            nyquist_hz = 0.5 / self.dt
            if not 0.0 < self.cutoff_hz < nyquist_hz:
                raise ValueError(
                    f"cutoff_hz 는 0 과 Nyquist({nyquist_hz:.2f} Hz) 사이여야 합니다. "
                    f"받은 값: {self.cutoff_hz}"
                )
        if self.mode == "savgol":
            if self.savgol_window % 2 == 0:
                raise ValueError(f"savgol_window 는 홀수여야 합니다: {self.savgol_window}")
            if self.savgol_polyorder >= self.savgol_window:
                raise ValueError("savgol_polyorder < savgol_window 여야 합니다")

    # ------------------------------------------------------------------

    def __call__(self, trajectory: np.ndarray, origin=None) -> np.ndarray:
        """`smooth()` 의 별칭. planner 가 필터를 콜러블로만 다루게 한다."""
        return self.smooth(trajectory, origin=origin)

    def smooth(self, trajectory: np.ndarray, origin=None) -> np.ndarray:
        """궤적을 평활화한다.

        :param trajectory: (T, >=3) [x, y, yaw]. ego-local 이든 global 이든
            상관없다 — 이 함수는 좌표계를 모른다.
        :param origin: 궤적이 출발하는 지점 (3,) [x, y, yaw]. 기본값은 원점이며,
            ego-local 궤적에서는 그것이 곧 현재 자차 위치다. 첫 웨이포인트까지의
            증분도 필터에 넣기 위해 필요하다.
        :return: (T, trajectory.shape[1]) — 3열 뒤의 열은 원본 그대로 둔다.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 3:
            raise ValueError(f"trajectory must have shape (T, >=3), got {traj.shape}")

        out = traj.copy()
        if self.mode == "none" or len(traj) < _MIN_SAMPLES:
            return out

        start = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)[:3]

        # 원점을 앞에 붙여 (T+1, 3) 으로 만든 뒤 증분을 낸다. yaw 는 ±π 경계에서
        # 증분이 2π 로 튀지 않도록 먼저 펼친다.
        seq = np.vstack([start[None, :], traj[:, :3]])
        seq[:, 2] = np.unwrap(seq[:, 2])
        delta = np.diff(seq, axis=0)                    # (T, 3)

        smoothed = seq[0, :3] + np.cumsum(self._filter(delta), axis=0)
        smoothed[:, 2] = _wrap_to_pi(smoothed[:, 2])

        out[:, :3] = smoothed
        return out

    # ------------------------------------------------------------------

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        """(T, C) 신호를 열마다 필터링한다."""
        from scipy.signal import butter, filtfilt, lfilter, lfilter_zi, savgol_filter

        if self.mode == "savgol":
            window = min(self.savgol_window, len(signal) - (len(signal) + 1) % 2)
            if window <= self.savgol_polyorder:
                return signal.copy()
            return savgol_filter(
                signal, window, self.savgol_polyorder, axis=0, mode="interp"
            )

        b, a = butter(self.order, self.cutoff_hz / (0.5 / self.dt), btype="low")

        if self.mode == "zerophase":
            # padlen 기본값은 3·max(len(a), len(b)). 짧은 궤적에서 넘치면 줄인다.
            padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
            return filtfilt(b, a, signal, axis=0, padlen=padlen)

        # causal — 첫 표본으로 정상상태를 잡아 시작 과도응답을 없앤다.
        # 이걸 안 하면 필터가 0 에서 출발해 궤적 앞부분이 원점 쪽으로 끌려간다.
        zi = lfilter_zi(b, a)[:, None] * signal[0][None, :]
        filtered, _ = lfilter(b, a, signal, axis=0, zi=zi)
        return filtered


def make_smoother(
    mode: str = "none",
    dt: float = 0.1,
    cutoff_hz: float = 1.0,
    order: int = 2,
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
) -> Optional[LowPassTrajectoryFilter]:
    """설정값에서 필터를 만든다. ``mode="none"`` 이면 None 을 돌려준다.

    planner 가 "필터가 있으면 통과, 없으면 그대로" 한 줄로 쓸 수 있게 하기 위한
    얇은 생성자다.
    """
    if mode == "none":
        return None
    return LowPassTrajectoryFilter(
        dt=dt,
        cutoff_hz=cutoff_hz,
        mode=mode,
        order=order,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )


# ---------------------------------------------------------------------------
# 진단용 — 필터가 무엇을 바꿨는지 보기 위한 것이다. 채점에는 쓰이지 않는다.
# ---------------------------------------------------------------------------


def curvature(trajectory: np.ndarray) -> np.ndarray:
    """경로 곡률 κ [1/m] 를 웨이포인트마다 계산한다.

    κ = (x'y'' − y'x'') / (x'² + y'²)^(3/2) 을 중심차분으로 근사한다.
    곡률은 조향각과 직결되므로(δ ≈ atan(κ·L)) 떨림이 가장 잘 보이는 신호다.

    :param trajectory: (T, >=2)
    :return: (T,) — 분모가 0 에 가까운 정지 구간은 0 으로 둔다.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    dx = np.gradient(traj[:, 0])
    dy = np.gradient(traj[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2) ** 1.5
    kappa = np.zeros(len(traj))
    ok = denom > 1e-6
    kappa[ok] = (dx[ok] * ddy[ok] - dy[ok] * ddx[ok]) / denom[ok]
    return kappa


def heading_rate(trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """yaw rate [rad/s]. nuPlan comfort 의 yaw rate 항과 같은 축을 본다.

    :param trajectory: (T, >=3) — 3열이 yaw 여야 한다.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    return np.gradient(np.unwrap(traj[:, 2])) / dt
