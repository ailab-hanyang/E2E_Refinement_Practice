#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
채점 결과 시각화 — 지도 렌더러와 분리된 이식 가능 층.

`nuplan_evaluator` 가 낸 `TrajectoryScore` 를 **matplotlib Axes 하나에** 그린다.
지도·agent·경로 렌더링은 건드리지 않으므로, `evaluator/` 패키지만 복사하면
"채점 + 점수 시각화" 가 통째로 따라온다.

왜 여기 있는가::

    이전  점수표·comfort 패널·TTC 마커가 NuplanScenarioRender 의 메서드였다.
          → 떼어내려면 ScenarioManager·utils.vis·지도 렌더가 전부 따라오고,
            ml/rf **2개가 하드코딩**되어 단일 평가를 표현할 수 없었다.
    이후  후보를 dict 로 받는다. {"ml":…} 단일 / {"ml":…,"rf":…} 비교 / N개가
          같은 코드 경로를 쓴다. 좌표 변환은 콜러블로 주입한다.

사용 예::

    from src.planners.evaluator import build_score_panel, draw_score_table

    panel = build_score_panel({"ml": ml_score})                 # 단일 평가
    panel = build_score_panel({"ml": ml_score, "rf": rf_score},  # 비교 평가
                              summary={"decision": "PASS"})
    draw_score_table(ax, panel)
    draw_comfort_panels(ax, panel)
    draw_collision_markers(ax, panel, to_local)   # to_local: (2,)->(2,) 콜러블

레이어::

    nuplan_evaluator.TrajectoryScore
        ↓  build_score_panel   (diagnostics → 그리기 재료)
    ScorePanel
        ↓  draw_*              (Axes 에 그린다)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ScorePanel",
    "build_score_panel",
    "draw_score_table",
    "draw_violation_reasons",
    "draw_open_loop_table",
    "draw_open_loop_error_panels",
    "OPEN_LOOP_ROWS",
    "DEFAULT_OPEN_LOOP_BOXES",
    "draw_comfort_panels",
    "draw_collision_markers",
    "candidate_colour",
    "highlight_tokens",
    "METRIC_ROWS",
    "COMFORT_SPECS",
    "DEFAULT_COMFORT_BOXES",
]


# -----------------------------------------------------------------------------
# 표시 규약 (렌더러에서 옮겨온 상수 — 여기가 유일한 정의다)
# -----------------------------------------------------------------------------

#: 표 행 순서. (키, 표시명, 종류) — 곱셈항(x) 먼저, 그다음 가중항(w+가중치).
#
# 진행률 계열(ego_progress_along_expert_route, ego_is_making_progress)은 뺐다.
# closed-loop 에서 시뮬 ego 가 로그 ego 와 발산하는데 분모가 "같은 시점부터의
# 로그 진행량"이라 scene 마다 편향 크기가 달라 공정 비교가 불가능하고,
# 채점에서도 1.0 으로 고정(PROGRESS_METRICS)돼 최종 점수에 기여하지 않는다.
# 실측값은 CSV 와 TrajectoryScore.diagnostics 에 남는다.
METRIC_ROWS: List[Tuple[str, str, str]] = [
    ("no_ego_at_fault_collisions", "collision", "x"),
    ("drivable_area_compliance", "drivable", "x"),
    ("driving_direction_compliance", "direction", "x"),
    ("time_to_collision_within_bound", "TTC", "w5"),
    ("speed_limit_compliance", "speed_limit", "w4"),
    ("ego_is_comfortable", "comfort", "w2"),
]

#: 뒤 후보가 더 나으면 파랑, 더 나쁘면 빨강, 같으면 검정.
#: 단일 후보에서는 위반(값 < 1)이 빨강, 만점이 검정이다.
CMP_BETTER = "#1F5FB4"
CMP_WORSE = "#C0392B"
CMP_SAME = "#1A1A1A"
CMP_EPS = 1e-6

#: 후보별 계열 색. ML=회색, refined=파랑, 그 외는 순환.
_CANDIDATE_COLOURS = {
    "ml": "#666666",
    "rf": "#1F5FB4",
}
_FALLBACK_COLOURS = ("#2E8B57", "#B8860B", "#8E44AD", "#16A085")

#: 위반 표시 색 (comfort 초과 구간, TTC 충돌 지점, at-fault 충돌 상대)
BAD = "#C0392B"

#: comfort 패널에 그릴 3항. comfort 6항 중 실제로 comfort=0 을 만드는 것들이다.
# 나머지 3항을 뺀 이유:
#   - ego_lat_acceleration : 구조적으로 항상 0 이다(no-slip +
#     get_acceleration_shifted 가 회전항을 x 로만 넣는다) → 그려도 평평하다
#   - ego_jerk(magnitude)  : a_y=0 이라 |a_x| 의 미분과 사실상 같다 → j_x 와 중복
#   - ego_yaw_acceleration : yaw_rate 의 미분이라 같은 축을 본다
COMFORT_SPECS: List[Tuple[str, str]] = [
    ("lon_accel", "a_x  [m/s²]"),
    ("lon_jerk", "j_x  [m/s³]"),
    ("yaw_rate", "yaw rate  [rad/s]"),
]

#: 점수표 오른쪽 3열. 표는 x0=0.012 에서 폭 ~0.38 을 쓰므로 0.44 부터 시작해
#: 폭 0.165 + 간격 0.03 으로 세 개를 0.995 까지 채운다.
DEFAULT_COMFORT_BOXES = [
    [0.440, 0.030, 0.165, 0.185],
    [0.635, 0.030, 0.165, 0.185],
    [0.830, 0.030, 0.165, 0.185],
]

#: Open-loop 표의 행. (프레임값 키, 누적값 키, 표시명, higher_is_better).
#: MR·SCORE 는 프레임 하나로 정의되지 않으므로 프레임값 키가 None 이다.
OPEN_LOOP_ROWS: List[Tuple[Optional[str], str, str, bool]] = [
    ("ol_ade_m", "ol_ade_mean", "ADE   [m]", False),
    ("ol_fde_m", "ol_fde_mean", "FDE   [m]", False),
    ("ol_ahe_rad", "ol_ahe_mean", "AHE [rad]", False),
    ("ol_fhe_rad", "ol_fhe_mean", "FHE [rad]", False),
    (None, "ol_miss_rate", "MR", False),
    (None, "ol_score", "SCORE", True),
]

#: Open-loop 오차 곡선 2칸. comfort 패널과 같은 y 범위를 쓴다(둘은 함께 그리지 않는다).
DEFAULT_OPEN_LOOP_BOXES = [
    [0.440, 0.030, 0.255, 0.185],
    [0.740, 0.030, 0.255, 0.185],
]

#: MR 이 이 값을 넘으면 공식 게이트가 0 이 된다.
OPEN_LOOP_MISS_RATE_GATE = 0.3

#: 오차 곡선에 그리는 공식 임계선. viz 는 numpy 만 쓰는 층이라 상수를 여기에도 둔다.
OPEN_LOOP_MAX_L2_ERROR_LINE = 8.0     # [m]
OPEN_LOOP_MAX_HEADING_LINE = 0.8      # [rad]


def candidate_colour(name: str, index: int = 0) -> str:
    """후보 이름 → 계열 색. 알려진 이름(ml/rf)은 고정, 나머지는 순환한다."""
    key = str(name).strip().lower()
    if key in _CANDIDATE_COLOURS:
        return _CANDIDATE_COLOURS[key]
    return _FALLBACK_COLOURS[index % len(_FALLBACK_COLOURS)]


# -----------------------------------------------------------------------------
# 그리기 재료
# -----------------------------------------------------------------------------

@dataclass
class ScorePanel:
    """한 스텝의 채점 결과를 그리기 직전 형태로 정리한 것.

    후보 수에 제약이 없다 — 1개면 단일 평가, 2개면 비교 평가다. 값 dict 는
    삽입 순서를 유지하므로 표의 열 순서 = `build_score_panel` 에 넘긴 순서다.
    """

    #: 후보 이름 → {metric: value}. multiplicative + weighted 를 합친 것.
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: 후보 이름 → 최종 스칼라
    finals: Dict[str, float] = field(default_factory=dict)
    #: {"<name>": {"lon_accel":[..], "lon_jerk":[..], "yaw_rate":[..]}, ...,
    #:  "bounds": {키마다 [lo,hi]}, "failed": {"<name>": [항 이름...]}}
    comfort: Optional[Dict] = None
    #: 후보 이름 → TTC 위반 dict(없으면 None)
    ttc: Dict[str, Optional[Dict]] = field(default_factory=dict)
    #: 후보 이름 → at-fault 충돌 상대 토큰 리스트
    collision: Dict[str, List[str]] = field(default_factory=dict)
    #: 후보 이름 → 위반 사유 문자열 목록. 표의 빨간 행을 실측값으로 푼 것이다.
    reasons: Dict[str, List[str]] = field(default_factory=dict)
    #: 후보 이름 → Open-loop 지표 dict (TrajectoryScore.open_loop 그대로)
    open_loop: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Open-loop 부가 정보 — {"de": {name: [...]}, "he": {...},
    #: "drift": float, "on_grid": bool}
    open_loop_detail: Optional[Dict] = None
    #: 표 최하단에 붙는 부가 정보. 예: {"decision": "PASS", "guard": "ok"}
    summary: Optional[Dict] = None

    @property
    def names(self) -> List[str]:
        return list(self.metrics.keys())

    def has_detail(self) -> bool:
        """comfort/TTC/충돌 중 하나라도 그릴 것이 있는가."""
        return bool(
            self.comfort
            or any(self.ttc.values())
            or any(self.collision.values())
        )


def _violation_reasons(score: "object") -> List[str]:
    """1.0 미만인 지표를 diagnostics 의 실측값과 묶어 한 줄씩 만든다.

    점수표는 "어느 규칙이 깨졌는가" 까지만 답한다. "얼마나 깨졌는가" 는 이미
    채점 과정에서 계산되어 `TrajectoryScore.diagnostics` 에 들어 있는데 지금까지
    아무도 읽지 않았다. 여기서 그것을 꺼내 쓴다 — 추가 연산은 없다.
    """
    metrics = {**(getattr(score, "multiplicative", None) or {}),
               **(getattr(score, "weighted", None) or {})}
    diag = getattr(score, "diagnostics", None) or {}
    out: List[str] = []

    def bad(key: str) -> bool:
        v = metrics.get(key)
        return v is not None and float(v) < 1.0 - CMP_EPS

    def at(key: str) -> str:
        t = diag.get(key)
        return "" if t is None else "  first t=+%.1fs" % float(t)

    if bad("no_ego_at_fault_collisions"):
        n = len(diag.get("collision_tokens") or [])
        out.append("collision    %s%s" % (
            "n=%d" % n if n else "at-fault", at("collision_first_time")))

    if bad("drivable_area_compliance"):
        t = diag.get("drivable_area_first_time")
        out.append("drivable     %s" % (
            "violated" if t is None else "first t=+%.1fs" % float(t)))

    if bad("driving_direction_compliance"):
        p = diag.get("driving_direction_min_progress")
        out.append("direction    %s" % (
            "wrong-way" if p is None else "min_progress %+.1f m" % float(p)))

    if bad("time_to_collision_within_bound"):
        v = diag.get("ttc_violation") or {}
        ttc = v.get("ttc", diag.get("ttc_min"))
        out.append("TTC          %s%s" % (
            "violated" if ttc is None else "%.2f s" % float(ttc), at("ttc_first_time")))

    if bad("speed_limit_compliance"):
        o = diag.get("speed_limit_max_overspeed")
        out.append("speed_limit  %s" % (
            "over limit" if o is None else "over %.2f m/s" % float(o)))

    if bad("ego_is_comfortable"):
        failed = [k for k, ok in (diag.get("comfort_per_metric") or {}).items() if not ok]
        out.append("comfort      %s" % (", ".join(failed) if failed else "violated"))

    return out


def build_score_panel(
    candidates: Dict[str, "object"],
    summary: Optional[Dict] = None,
) -> ScorePanel:
    """`TrajectoryScore` 들을 `ScorePanel` 로 모은다.

    이전에는 이 변환이 `IncrementalPlanner._build_reason_detail` 안에 묻혀 있어
    다른 planner 가 재사용할 수 없었다. 채점기와 같은 패키지로 끌어올린다.

    :param candidates: 이름 → TrajectoryScore. 순서가 표의 열 순서가 된다.
        관례상 첫 후보가 기준(비교의 좌변)이다.
    :param summary: 표 하단에 한 줄로 찍을 부가 정보(decision/guard 등).
    """
    metrics: Dict[str, Dict[str, float]] = {}
    finals: Dict[str, float] = {}
    ttc: Dict[str, Optional[Dict]] = {}
    collision: Dict[str, List[str]] = {}
    reasons: Dict[str, List[str]] = {}
    open_loop: Dict[str, Dict[str, float]] = {}
    ol_de: Dict[str, List[float]] = {}
    ol_he: Dict[str, List[float]] = {}
    ol_grid: Dict[str, List[int]] = {}
    ol_detail: Optional[Dict] = None

    comfort_series: Dict[str, Dict] = {}
    comfort_failed: Dict[str, List[str]] = {}
    comfort_bounds = None

    for name, score in candidates.items():
        if score is None:
            continue
        metrics[name] = {**(score.multiplicative or {}), **(score.weighted or {})}
        finals[name] = float(score.final)

        diag = score.diagnostics or {}
        series = diag.get("comfort_series")
        if series:
            comfort_series[name] = series
            comfort_bounds = comfort_bounds or diag.get("comfort_bounds")
        # comfort 는 6항 AND 인데 패널은 3항만 그린다. jerk·yaw_accel 이 원인일 때
        # 화면만으로는 comfort=0 을 설명할 수 없으므로 실패한 항 이름을 같이 남긴다.
        comfort_failed[name] = [
            k for k, ok in (diag.get("comfort_per_metric") or {}).items() if not ok
        ]
        ttc[name] = diag.get("ttc_violation")
        collision[name] = list(diag.get("collision_tokens") or [])
        reasons[name] = _violation_reasons(score)

        if getattr(score, "open_loop", None):
            open_loop[name] = dict(score.open_loop)
            ol_de[name] = list(diag.get("open_loop_de_per_step") or [])
            ol_he[name] = list(diag.get("open_loop_he_per_step") or [])
            ol_grid[name] = list(diag.get("open_loop_sample_steps") or [])
            # drift·격자 여부는 후보와 무관한 컨텍스트 값이라 하나만 남긴다
            ol_detail = {"drift": diag.get("expert_drift_m", float("nan")),
                         "on_grid": bool(diag.get("open_loop_on_grid", False))}

    comfort = None
    if comfort_series:
        comfort = dict(comfort_series)
        comfort["bounds"] = comfort_bounds or {}
        comfort["failed"] = comfort_failed

    detail = None
    if open_loop:
        detail = dict(ol_detail or {})
        detail["de"] = ol_de
        detail["he"] = ol_he
        detail["grid"] = ol_grid

    return ScorePanel(
        metrics=metrics,
        finals=finals,
        comfort=comfort,
        ttc=ttc,
        collision=collision,
        reasons=reasons,
        open_loop=open_loop,
        open_loop_detail=detail,
        summary=summary,
    )


def highlight_tokens(panel: ScorePanel) -> set:
    """빨갛게 칠할 상대 객체 토큰 — TTC 위반 상대 ∪ at-fault 충돌 상대.

    원인이 서로 다를 수 있으므로 둘을 합친다(TTC 는 예측, 충돌은 실제).
    """
    tokens = set()
    for v in panel.ttc.values():
        if v and v.get("track_token"):
            tokens.add(v["track_token"])
    for toks in panel.collision.values():
        tokens.update(toks or [])
    return tokens


# -----------------------------------------------------------------------------
# 점수표
# -----------------------------------------------------------------------------

def _row_colour(
    values: Sequence[float],
    *,
    higher_is_better: bool = True,
    flag_below: Optional[float] = 1.0,
) -> Tuple[str, bool]:
    """행 색과 굵기. 반환 (colour, bold).

    후보 2개 이상 → 마지막이 첫째보다 나으면 파랑 / 나쁘면 빨강 / 같으면 검정.
    후보 1개     → flag_below 미만이면 빨강(그 행이 감점 원인이다), 아니면 검정.

    :param higher_is_better: False 면 값이 작을수록 좋다(오차 지표). 비교 부호를 뒤집는다.
    :param flag_below: 단일 후보에서 빨강으로 칠할 기준. None 이면 칠하지 않는다 —
        미터·라디안처럼 "만점" 이 정의되지 않는 값에 쓴다.
    """
    vals = [float(v) for v in values]
    if any(v != v for v in vals):          # nan
        return "#999999", False
    if len(vals) >= 2:
        d = vals[-1] - vals[0]
        if not higher_is_better:
            d = -d
        if abs(d) <= CMP_EPS:
            return CMP_SAME, False
        return (CMP_BETTER if d > 0 else CMP_WORSE), True
    if flag_below is not None and vals[0] < flag_below - CMP_EPS:
        return CMP_WORSE, True
    return CMP_SAME, False


def _format_rows(panel: ScorePanel) -> Tuple[List[Tuple[str, str, bool]], str]:
    """(label, colour, bold) 리스트와 헤더 문자열을 만든다.

    비교(2개)일 때의 서식은 이전 구현과 동일하다 — 기존 수집 런 시각화와
    글자 폭·정렬이 어긋나면 안 되기 때문이다.
    """
    names = panel.names
    if not names:
        return [], ""

    if len(names) == 2:
        a_name, b_name = names
        header = "%-12s%-3s %5s    %5s  %7s" % (
            "metric", "", a_name.upper()[:5], b_name.upper()[:5], "delta"
        )

        def fmt(short, kind, vals):
            a, b = float(vals[0]), float(vals[1])
            d = b - a
            diff = "" if abs(d) <= CMP_EPS else "%+.3f" % d
            return "%-12s%-3s %5.3f -> %5.3f  %7s" % (short, kind, a, b, diff)

        def fmt_score(vals):
            a, b = float(vals[0]), float(vals[1])
            return "%-12s%-3s %5.3f -> %5.3f  %+7.3f" % ("SCORE", "", a, b, b - a)
    else:
        cols = "".join("%8s" % n.upper()[:7] for n in names)
        header = "%-12s%-3s%s" % ("metric", "", cols)

        def fmt(short, kind, vals):
            return "%-12s%-3s%s" % (
                short, kind, "".join("%8.3f" % float(v) for v in vals)
            )

        def fmt_score(vals):
            return "%-12s%-3s%s" % (
                "SCORE", "", "".join("%8.3f" % float(v) for v in vals)
            )

    rows: List[Tuple[str, str, bool]] = []

    finals = [panel.finals[n] for n in names if n in panel.finals]
    if len(finals) == len(names) and finals:
        colour, _ = _row_colour(finals)
        rows.append((fmt_score(finals), colour, True))

    for key, short, kind in METRIC_ROWS:
        if any(key not in panel.metrics[n] for n in names):
            continue
        vals = [float(panel.metrics[n][key]) for n in names]
        colour, bold = _row_colour(vals)
        rows.append((fmt(short, kind, vals), colour, bold))

    return rows, header


def draw_score_table(
    ax,
    panel: ScorePanel,
    *,
    origin: Tuple[float, float] = (0.012, 0.012),
    fontsize: float = 8.5,
    row_height: float = 0.0205,
) -> Optional[float]:
    """세부지표 표를 Axes 좌하단(기본)에 그린다. 반환값은 표의 폭(axes 좌표).

    최상단 행이 최종 점수(SCORE)이고 그 아래로 세부지표가 같은 열에 정렬된다.
    행 색은 `_row_colour` 규칙을 따른다.

    배경은 Rectangle 하나로 깔고 행은 개별 text 로 그린다 — 여러 줄 문자열
    위에 색을 덧그리면 linespacing 이 어긋나 정렬이 깨지기 때문이다.

    반환한 폭은 호출자가 오른쪽에 다른 것(comfort 패널)을 붙일 때 겹침을
    피하는 데 쓴다.
    """
    import matplotlib.pyplot as plt

    rows, header = _format_rows(panel)
    if not rows:
        return None

    ncol = len(header)
    fs = fontsize
    dy = row_height
    x0, y0 = origin
    pad_x, pad_y = 0.010, 0.010
    n_extra = 2 if panel.summary else 1    # header(+구분선) / summary 줄
    n_line = len(rows) + 1 + n_extra

    # 폭은 monospace 문자폭을 실측해 잡는다. 상수로 잡으면 dpi·figsize 가
    # 바뀔 때 배경이 글자를 자른다(실제로 한 번 잘렸다).
    fig = ax.get_figure()
    try:
        probe = ax.text(0, 0, "0" * ncol, transform=ax.transAxes,
                        fontsize=fs, family="monospace", alpha=0.0)
        fig.canvas.draw()
        bb = probe.get_window_extent(renderer=fig.canvas.get_renderer())
        width = bb.transformed(ax.transAxes.inverted()).width + 2 * pad_x
        probe.remove()
    except Exception:
        width = 0.0075 * ncol + 2 * pad_x

    ax.add_patch(plt.Rectangle(
        (x0, y0), width, dy * n_line + 2 * pad_y,
        transform=ax.transAxes, facecolor="white", edgecolor="#444444",
        linewidth=1.2, alpha=0.94, zorder=50,
    ))

    tx = x0 + pad_x
    y = y0 + pad_y                    # 아래에서 위로 쌓는다
    if panel.summary:
        ax.text(tx, y, "   ".join(
                    "%s: %s" % (k, v) for k, v in panel.summary.items()),
                transform=ax.transAxes, fontsize=fs - 0.5, family="monospace",
                color="#555555", va="bottom", ha="left", zorder=51)
        y += dy

    for label, col, bold in reversed(rows):
        ax.text(tx, y, label, transform=ax.transAxes, fontsize=fs,
                family="monospace", color=col,
                fontweight="bold" if bold else "normal",
                va="bottom", ha="left", zorder=51)
        y += dy

    ax.text(tx, y, "-" * ncol, transform=ax.transAxes, fontsize=fs,
            family="monospace", color="#999999", va="bottom", ha="left", zorder=51)
    y += dy * 0.55
    ax.text(tx, y, header, transform=ax.transAxes, fontsize=fs,
            family="monospace", color="#555555", va="bottom", ha="left", zorder=51)

    return width


# -----------------------------------------------------------------------------
# 위반 사유
# -----------------------------------------------------------------------------

def score_table_top(
    panel: ScorePanel,
    *,
    origin: Tuple[float, float] = (0.012, 0.012),
    row_height: float = 0.0205,
    pad_y: float = 0.010,
) -> float:
    """`draw_score_table` 이 같은 인자로 그렸을 때 표의 윗변 y(axes 좌표).

    사유 상자를 표 바로 위에 올리려면 표 높이를 알아야 하는데, 높이는 후보 수가
    아니라 **행 수**로 정해진다(지표가 빠지면 행이 준다). 그래서 실제 행 생성기를
    한 번 더 돌려 센다.
    """
    rows, _ = _format_rows(panel)
    if not rows:
        return origin[1]
    n_extra = 2 if panel.summary else 1
    return origin[1] + row_height * (len(rows) + 1 + n_extra) + 2 * pad_y


def draw_violation_reasons(
    ax,
    panel: ScorePanel,
    *,
    origin: Optional[Tuple[float, float]] = None,
    fontsize: float = 8.5,
    row_height: float = 0.0205,
) -> Optional[float]:
    """점수표 위에 위반 사유를 한 줄씩 그린다. 반환값은 상자의 폭(axes 좌표).

    점수표의 빨간 숫자는 **어느** 규칙이 깨졌는지를, 이 상자는 **얼마나** 깨졌는지를
    답한다. 두 가지를 한 화면에 두어야 영상만 보고도 감점 원인을 말할 수 있다.

    후보가 여럿이면 줄 앞에 후보 이름을 붙인다. 위반이 하나도 없으면 회색으로
    `VIOLATED  none` 한 줄만 그린다 — 상자가 사라졌다 나타나면 프레임마다 레이아웃이
    흔들려 영상에서 읽기 어렵다.
    """
    import matplotlib.pyplot as plt

    if not panel.names:
        return None

    multi = len(panel.names) > 1
    lines: List[str] = []
    for name in panel.names:
        for r in panel.reasons.get(name, []):
            lines.append(("%-3s " % name.upper()[:3] if multi else "") + r)

    ok = not lines
    body = lines or ["none"]
    ncol = max(len("VIOLATED  "), max(len(s) for s in body) + 2)

    fs = fontsize
    dy = row_height
    x0, y0 = origin if origin is not None else (0.012, score_table_top(panel) + 0.008)
    pad_x, pad_y = 0.010, 0.008

    fig = ax.get_figure()
    try:
        probe = ax.text(0, 0, "0" * ncol, transform=ax.transAxes,
                        fontsize=fs, family="monospace", alpha=0.0)
        fig.canvas.draw()
        bb = probe.get_window_extent(renderer=fig.canvas.get_renderer())
        width = bb.transformed(ax.transAxes.inverted()).width + 2 * pad_x
        probe.remove()
    except Exception:
        width = 0.0075 * ncol + 2 * pad_x

    ax.add_patch(plt.Rectangle(
        (x0, y0), width, dy * (len(body) + 1) + 2 * pad_y,
        transform=ax.transAxes, facecolor="white",
        edgecolor="#999999" if ok else BAD,
        linewidth=1.2, alpha=0.94, zorder=50,
    ))

    tx = x0 + pad_x
    y = y0 + pad_y
    colour = "#999999" if ok else BAD
    for label in reversed(body):
        ax.text(tx, y, label, transform=ax.transAxes, fontsize=fs,
                family="monospace", color=colour, va="bottom", ha="left", zorder=51)
        y += dy
    ax.text(tx, y, "VIOLATED", transform=ax.transAxes, fontsize=fs,
            family="monospace", color=colour, fontweight="bold",
            va="bottom", ha="left", zorder=51)

    return width


# -----------------------------------------------------------------------------
# Open-loop — 표와 오차 곡선
# -----------------------------------------------------------------------------

def _fmt_ol(v: Optional[float]) -> str:
    """미터·라디안은 10 을 넘길 수 있어 %5.3f 로는 정렬이 깨진다."""
    if v is None or v != v:            # None / nan
        return "    n/a"
    return "%7.3f" % float(v)


def draw_open_loop_table(
    ax,
    panel: ScorePanel,
    *,
    origin: Tuple[float, float] = (0.012, 0.012),
    fontsize: float = 8.5,
    row_height: float = 0.0205,
) -> Optional[float]:
    """Open-loop 지표표를 Axes 좌하단에 그린다. 반환값은 표의 폭(axes 좌표).

    지표마다 두 값을 보인다.

    * ``now``       — 이 프레임의 계획 궤적만으로 정해지는 값
    * ``mean(1Hz)`` — 공식 격자 프레임에 대한 누적 평균. 시나리오 끝에서 공식값에 수렴한다

    MR 과 SCORE 는 프레임 하나로 정의되지 않으므로 누적값만 있다.
    """
    import matplotlib.pyplot as plt

    names = [n for n in panel.names if panel.open_loop.get(n)]
    if not names:
        return None

    detail = panel.open_loop_detail or {}
    multi = len(names) > 1

    # 헤더 — 후보마다 now/mean 두 열
    head = "%-10s" % "open-loop"
    for n in names:
        head += ("%9s%11s" % (n.upper()[:5] + " now", "mean(1Hz)")) if multi \
            else ("%9s%11s" % ("now", "mean(1Hz)"))

    # 표 위 한 줄 — 무엇과 비교한 값인지, 공식 격자인지, 몇 프레임이 누적됐는지.
    # 헤더보다 길 수 있으므로 표 폭 계산에 함께 넣는다.
    drift = detail.get("drift", float("nan"))
    n_acc = panel.open_loop[names[0]].get("ol_frames", float("nan"))
    off_log = drift == drift and drift > 0.5
    note = "vs log ego%s   n=%d   drift %s m%s" % (
        "  [1Hz]" if detail.get("on_grid") else "",
        int(n_acc) if n_acc == n_acc else 0,
        ("%.2f" % drift) if drift == drift else "n/a",
        "  off-log" if off_log else "",
    )
    ncol = max(len(head), len(note))

    rows: List[Tuple[str, str, bool]] = []
    for frame_key, mean_key, label, higher in OPEN_LOOP_ROWS:
        cells = ""
        for n in names:
            ol = panel.open_loop[n]
            cells += ("%9s" % "--") if frame_key is None else ("%9s" % _fmt_ol(ol.get(frame_key)))
            cells += "%11s" % _fmt_ol(ol.get(mean_key))
        vals = [panel.open_loop[n].get(mean_key, float("nan")) for n in names]
        flag = OPEN_LOOP_MISS_RATE_GATE if mean_key == "ol_miss_rate" else None
        # MR 은 게이트가 있어 절대 기준이 성립한다. 나머지 오차 지표는 "만점" 이 없으므로
        # 단일 후보에서 빨강으로 칠하지 않는다.
        if mean_key == "ol_miss_rate":
            colour, bold = ((CMP_WORSE, True)
                            if (vals[0] == vals[0] and vals[0] > flag) else (CMP_SAME, False))
            if multi:
                colour, bold = _row_colour(vals, higher_is_better=False, flag_below=None)
        else:
            colour, bold = _row_colour(vals, higher_is_better=higher, flag_below=None)
        rows.append(("%-10s%s" % (label, cells), colour, bold))

    fs, dy = fontsize, row_height
    x0, y0 = origin
    pad_x, pad_y = 0.010, 0.010
    n_line = len(rows) + 2                     # 구분선 + 헤더

    fig = ax.get_figure()
    try:
        probe = ax.text(0, 0, "0" * ncol, transform=ax.transAxes,
                        fontsize=fs, family="monospace", alpha=0.0)
        fig.canvas.draw()
        bb = probe.get_window_extent(renderer=fig.canvas.get_renderer())
        width = bb.transformed(ax.transAxes.inverted()).width + 2 * pad_x
        probe.remove()
    except Exception:
        width = 0.0075 * ncol + 2 * pad_x

    ax.add_patch(plt.Rectangle(
        (x0, y0), width, dy * n_line + 2 * pad_y,
        transform=ax.transAxes, facecolor="white", edgecolor="#444444",
        linewidth=1.2, alpha=0.94, zorder=50,
    ))

    tx, y = x0 + pad_x, y0 + pad_y
    for label, col, bold in reversed(rows):
        ax.text(tx, y, label, transform=ax.transAxes, fontsize=fs,
                family="monospace", color=col,
                fontweight="bold" if bold else "normal",
                va="bottom", ha="left", zorder=51)
        y += dy
    ax.text(tx, y, "-" * ncol, transform=ax.transAxes, fontsize=fs,
            family="monospace", color="#999999", va="bottom", ha="left", zorder=51)
    y += dy * 0.55
    ax.text(tx, y, head, transform=ax.transAxes, fontsize=fs,
            family="monospace", color="#555555", va="bottom", ha="left", zorder=51)

    ax.text(tx, y + dy * 1.15, note, transform=ax.transAxes, fontsize=fs - 0.5,
            family="monospace", color=BAD if off_log else "#555555",
            va="bottom", ha="left", zorder=51)
    return width


def draw_open_loop_error_panels(
    ax,
    panel: ScorePanel,
    *,
    boxes: Optional[Sequence[Sequence[float]]] = None,
    horizons: Sequence[int] = (3, 5, 8),
) -> None:
    """계획 궤적 80 스텝의 변위·헤딩 오차를 오른쪽 아래 두 칸에 그린다.

    표는 1 초 표본만 요약하므로 "8 초 중 어디서 벌어졌는가" 를 알 수 없다.
    스텝 전부를 점으로 찍고 공식 표본 8개를 크게 강조하면, 표의 한 숫자가 어느
    구간에서 만들어졌는지 바로 보인다.
    """
    detail = panel.open_loop_detail or {}
    boxes = boxes or DEFAULT_OPEN_LOOP_BOXES

    specs = [("de", "displacement error  [m]", OPEN_LOOP_MAX_L2_ERROR_LINE),
             ("he", "heading error  [rad]", OPEN_LOOP_MAX_HEADING_LINE)]

    for (key, title, limit), box in zip(specs, boxes):
        series_by_name = detail.get(key) or {}
        arrays = {n: np.asarray(v, dtype=float) for n, v in series_by_name.items()}
        if not arrays or all(a.size == 0 for a in arrays.values()):
            continue

        sub = ax.inset_axes(box, transform=ax.transAxes, zorder=52)
        sub.patch.set_facecolor("white")
        # comfort 패널과 달리 완전 불투명하게 둔다. 오차 곡선은 점으로 찍히므로 뒤의
        # agent 상자가 비쳐 보이면 데이터 점과 헷갈린다.
        sub.patch.set_alpha(1.0)

        for idx, (name, series) in enumerate(arrays.items()):
            if series.size == 0:
                continue
            colour = candidate_colour(name, idx)
            t = (np.arange(series.size) + 1) * 0.1        # 행 0 은 +0.1 s 다
            sub.scatter(t, series, s=3, color=colour, alpha=0.55,
                        zorder=3, linewidths=0, label=name.upper())
            # 표의 숫자를 만든 공식 표본만 크게 찍는다. 시나리오 후반부에서는 표본
            # 하나가 앞 시각을 다시 보므로 점 두 개가 겹친다 — 공식 계산 그대로다.
            steps = (detail.get("grid") or {}).get(name)
            grid = (np.asarray(steps, dtype=int) - 1 if steps
                    else np.arange(9, series.size, 10))
            grid = grid[(grid >= 0) & (grid < series.size)]
            if grid.size:
                sub.scatter(t[grid], series[grid], s=22, facecolors="none",
                            edgecolors=colour, linewidths=1.1, zorder=4)

        for h in horizons:
            sub.axvline(h, color="#888888", ls=":", lw=0.8, alpha=0.8, zorder=2)
        sub.axhline(limit, color=BAD, ls="--", lw=0.9, alpha=0.8, zorder=2)

        sub.set_title(title, fontsize=6.5, pad=2.0)
        sub.tick_params(labelsize=5.0, length=2, pad=1)
        sub.set_xlabel("t [s]", fontsize=6, labelpad=1)
        sub.set_xlim(0, (max(a.size for a in arrays.values()) + 1) * 0.1)
        # y 범위는 데이터가 정한다. axhline 은 범위를 늘리므로 뒤에서 다시 고정하지 않으면
        # 오차가 작을 때 곡선이 바닥에 눌려 형태가 보이지 않는다. 임계선은 사거리에
        # 들어올 때만 보인다.
        peak = max((float(np.nanmax(a)) for a in arrays.values() if a.size), default=limit)
        sub.set_ylim(0, max(peak * 1.15, limit * 0.05))
        sub.grid(alpha=0.25, lw=0.4)
        for sp in sub.spines.values():
            sp.set_edgecolor("#444444")
        if key == "de" and len(arrays) > 1:
            sub.legend(fontsize=5.5, loc="upper left", framealpha=0.85,
                       handlelength=1.2, borderpad=0.25)


# -----------------------------------------------------------------------------
# comfort 패널
# -----------------------------------------------------------------------------

def draw_comfort_panels(
    ax,
    panel: ScorePanel,
    *,
    boxes: Optional[Sequence[Sequence[float]]] = None,
) -> None:
    """점수표 오른쪽에 a_x / j_x / yaw rate 3개 2D 플롯을 붙인다.

    comfort 는 0/1 이라 표만 봐서는 "왜 불편으로 찍혔는지" 를 알 수 없다.
    후보를 겹쳐 그리고 nuPlan 상·하한을 점선으로 표시하면 어느 구간이 어느
    정도로 넘었는지가 바로 보인다. 한계를 넘은 구간은 빨간 점으로 덧찍고,
    각 플롯에 후보별 최대 절대값을 적는다.
    """
    comfort = panel.comfort
    if not comfort:
        return

    bounds = comfort.get("bounds") or {}
    failed = comfort.get("failed") or {}
    series_by_name = {
        n: comfort[n] for n in panel.names if isinstance(comfort.get(n), dict)
    }
    if not series_by_name:
        return

    boxes = boxes or DEFAULT_COMFORT_BOXES

    # 패널은 6항 중 3개만 그린다. jerk·yaw_accel 이 comfort=0 을 만들면 그림만
    # 봐서는 설명이 안 되므로 실패한 항 이름을 패널 위에 한 줄로 적는다.
    failed_txt = "   ".join(
        "%s: %s" % (n.upper(), ", ".join(failed.get(n) or []) or "-")
        for n in series_by_name
    )
    if any(failed.get(n) for n in series_by_name):
        ax.text(
            boxes[0][0], boxes[0][1] + boxes[0][3] + 0.035,
            "comfort ✗   " + failed_txt,
            transform=ax.transAxes, fontsize=7.0, family="monospace",
            color=BAD, va="bottom", ha="left", zorder=53,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", alpha=0.9),
        )

    for (key, title), box in zip(COMFORT_SPECS, boxes):
        arrays = {
            n: np.asarray(s.get(key, []), dtype=float)
            for n, s in series_by_name.items()
        }
        if all(a.size == 0 for a in arrays.values()):
            continue

        sub = ax.inset_axes(box, transform=ax.transAxes, zorder=52)
        sub.patch.set_facecolor("white")
        sub.patch.set_alpha(0.94)

        lo, hi = (bounds.get(key) or [None, None])[:2]
        for idx, (name, series) in enumerate(arrays.items()):
            if series.size == 0:
                continue
            colour = candidate_colour(name, idx)
            t = np.arange(series.size) * 0.1
            sub.plot(t, series, color=colour, lw=1.2, label=name.upper(), zorder=3)
            if lo is not None:
                bad = (series <= lo) | (series >= hi)
                if bad.any():
                    sub.scatter(t[bad], series[bad], s=7, color=BAD,
                                zorder=4, linewidths=0)

        if lo is not None:
            for y in (lo, hi):
                sub.axhline(y, color=BAD, ls="--", lw=0.9, alpha=0.8, zorder=2)

        # 각 플롯에서 max |·| 를 바로 읽을 수 있게 한다
        peak_txt = "\n".join(
            "max|%s| %.2f" % (
                n.upper(),
                float(np.abs(a).max()) if a.size else float("nan"),
            )
            for n, a in arrays.items()
        )
        sub.text(0.98, 0.96, peak_txt, transform=sub.transAxes,
                 ha="right", va="top", fontsize=5.6, family="monospace",
                 color="#222222", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BBBBBB", alpha=0.85))

        sub.set_title(title, fontsize=6.5, pad=2.0)
        sub.tick_params(labelsize=5.0, length=2, pad=1)
        sub.set_xlabel("t [s]", fontsize=6, labelpad=1)
        sub.grid(alpha=0.25, lw=0.4)
        for s in sub.spines.values():
            s.set_edgecolor("#444444")
        if key == COMFORT_SPECS[0][0] and len(arrays) > 1:
            sub.legend(fontsize=5.5, loc="lower right", framealpha=0.85,
                       handlelength=1.2, borderpad=0.25)


# -----------------------------------------------------------------------------
# TTC / 충돌 마커
# -----------------------------------------------------------------------------

def draw_collision_markers(
    ax,
    panel: ScorePanel,
    to_local: Callable[[np.ndarray], np.ndarray],
) -> None:
    """TTC 위반 지점을 궤적 위에 빨간 x 로 찍고 라벨을 단다.

    `to_local` 은 global (2,) 좌표를 그림 좌표로 옮기는 콜러블이다. 렌더러의
    `self.origin`/`self.rot_mat` 에 의존하지 않기 위해 주입받는다 — 지도 없이
    임의의 Axes 에 그릴 때도 그대로 동작한다.

    각 후보의 위반은 `{step, track_token, ttc, ego_pose(global), track_pose(global)}`
    이다. ego_pose 는 등속 투영으로 얻은 **예상 충돌 시점의 ego 위치**라
    정지 화면에서 "어디서 부딪히는가" 를 그대로 가리킨다.
    """
    # 뒤 후보를 먼저 그려 앞 후보(기준)가 위에 오게 한다.
    for name in reversed(panel.names):
        _draw_one_ttc(ax, panel.ttc.get(name), to_local)


def _draw_one_ttc(ax, ttc_violation, to_local) -> None:
    if not ttc_violation:
        return
    ego_p = np.asarray(ttc_violation.get("ego_pose", []), dtype=float)
    if ego_p.size < 2:
        return
    p = np.asarray(to_local(ego_p[:2]), dtype=float)

    trk = np.asarray(ttc_violation.get("track_pose", []), dtype=float)
    if trk.size >= 2:
        q = np.asarray(to_local(trk[:2]), dtype=float)
        # 예상 충돌 상대 위치까지 잇는 얇은 선 — 어느 객체와인지 연결해 보여준다
        ax.plot([p[0], q[0]], [p[1], q[1]], color=BAD,
                lw=1.0, ls=":", alpha=0.9, zorder=13)
        ax.scatter(q[0], q[1], marker="x", s=70, linewidths=2.0,
                   color=BAD, zorder=13)

    ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3.0,
               color=BAD, zorder=14)
    # 라벨에는 **위반 스텝의 시각**을 반드시 넣는다. TTC 값만 적으면
    # "TTC 0.00s" 를 t=0 으로 오해한다(실제로 오해가 있었다).
    #   t=+X.Xs : 궤적 시작으로부터의 경과 시간. step 0 은 두 후보가 공유하는
    #             초기 상태라 거기서 갈릴 수 없으므로 위반은 항상 t>=+0.1s 다.
    #   TTC     : 그 시점에서 등속 투영으로 본 충돌까지 남은 시간(0 이면 충돌 중).
    step = ttc_violation.get("step")
    t_off = "t=+%.1fs  " % (float(step) * 0.1) if step is not None else ""
    ax.annotate(
        "collision  %sTTC %.2fs" % (
            t_off, float(ttc_violation.get("ttc", float("nan")))),
        xy=(p[0], p[1]), xytext=(8, 8), textcoords="offset points",
        fontsize=8, family="monospace", color="white", zorder=15,
        bbox=dict(boxstyle="round,pad=0.28", fc=BAD, ec="none", alpha=0.92),
    )
