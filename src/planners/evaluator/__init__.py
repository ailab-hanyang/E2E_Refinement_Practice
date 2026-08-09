#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nuPlan 정렬 궤적 평가기.

레이어:

    utils.py               후보 궤적(numpy) → EgoState 시퀀스, 로그 GT → world,
                           CSV·차트 출력
    evaluate_functions.py  공식 metric 호출 → check_* (bool) / score_* (스칼라)
    nuplan_evaluator.py    집계 → TrajectoryScore
    viz.py                 TrajectoryScore → matplotlib Axes (점수표·comfort·마커)

바깥에서는 보통 nuplan_evaluator 의 것만 쓴다:

    from src.planners.evaluator import NuPlanTrajectoryScorer

시각화까지 필요하면 viz 의 것을 같이 쓴다. 지도 렌더러와 독립이라
이 패키지만 복사하면 채점 + 점수 시각화가 통째로 따라온다:

    panel = build_score_panel({"ml": ml_score})   # 단일 평가 / N개 비교 모두
    draw_score_table(ax, panel); draw_comfort_panels(ax, panel)
"""

from .nuplan_evaluator import (  # noqa: F401
    DEFAULT_METRIC_WEIGHT,
    OFFICIAL_METRIC_WEIGHTS,
    OFFICIAL_MULTIPLICATIVE_METRICS,
    PROGRESS_METRICS,
    NuPlanTrajectoryScorer,
    TrajectoryScore,
    aggregate_official_score,
    aggregate_open_loop_score,
)
from .viz import (  # noqa: F401
    COMFORT_SPECS,
    METRIC_ROWS,
    ScorePanel,
    build_score_panel,
    candidate_colour,
    draw_collision_markers,
    draw_comfort_panels,
    draw_open_loop_error_panels,
    draw_open_loop_table,
    draw_score_table,
    draw_violation_reasons,
    highlight_tokens,
    score_table_top,
)

__all__ = [
    "NuPlanTrajectoryScorer",
    "TrajectoryScore",
    "aggregate_official_score",
    "aggregate_open_loop_score",
    "OFFICIAL_METRIC_WEIGHTS",
    "OFFICIAL_MULTIPLICATIVE_METRICS",
    "PROGRESS_METRICS",
    "DEFAULT_METRIC_WEIGHT",
    # viz — 지도 렌더러와 독립. matplotlib Axes 하나만 있으면 동작한다.
    "ScorePanel",
    "build_score_panel",
    "draw_score_table",
    "draw_violation_reasons",
    "draw_open_loop_table",
    "draw_open_loop_error_panels",
    "score_table_top",
    "draw_comfort_panels",
    "draw_collision_markers",
    "highlight_tokens",
    "candidate_colour",
    "METRIC_ROWS",
    "COMFORT_SPECS",
]
