"""
점수 시각화를 evaluator.viz 에 위임하는 렌더러.

NuplanScenarioRender 의 지도·agent·궤적 렌더는 그대로 쓰고 점수 관련 3개 메서드만
덮어쓴다. 원본은 ml/rf 2개를 하드코딩해 단일 평가를 그리지 못한다
(_plot_evaluator_result 가 if not ml or not rf: return).
여기서는 ScorePanel 하나를 받아 후보 개수와 무관하게 그린다.

_plot_evaluator_result 는 점수표에 더해 위반 사유 상자도 그린다 — 표의 빨간 숫자는
어느 규칙이 깨졌는지만 알려주고, 얼마나 깨졌는지는 diagnostics 에만 있기 때문이다.

사용:

    render = EvalSceneRender()
    render.render_dagger_scene_from_simulation(
        ...,
        evaluator_result=panel,                       # ScorePanel
        reason_detail={"comfort": panel,              # 같은 panel 을 재사용
                       "ttc": {"ml": panel.ttc.get("ml")},
                       "collision": {"ml": panel.collision.get("ml")}},
        dagger_information={"score": "0.812"},
    )
"""

import numpy as np

from src.feature_builders.nuplan_scenario_render import NuplanScenarioRender
from src.planners.evaluator.viz import (
    ScorePanel,
    _draw_one_ttc,
    draw_comfort_panels,
    draw_score_table,
    draw_violation_reasons,
)


class EvalSceneRender(NuplanScenarioRender):
    """점수표·comfort 패널·TTC 마커를 evaluator.viz 로 그리는 렌더러."""

    def _to_local(self, point) -> np.ndarray:
        """global (2,) → 그림 좌표. render_dagger_scene 이 세운 origin/rot_mat 을 쓴다."""
        return np.matmul(np.asarray(point, dtype=float)[:2] - self.origin, self.rot_mat)

    # ── 덮어쓰는 3개 — ScorePanel 이 오면 viz 로 위임, 아니면 원본 동작 ──────

    def _plot_evaluator_result(self, ax, evaluator_result, summary=None):
        if isinstance(evaluator_result, ScorePanel):
            if summary and not evaluator_result.summary:
                evaluator_result.summary = summary
            draw_score_table(ax, evaluator_result)
            # 표는 어느 규칙이 깨졌는지까지만 보여준다. 얼마나 깨졌는지는 바로 위 상자에.
            draw_violation_reasons(ax, evaluator_result)
            return
        return super()._plot_evaluator_result(ax, evaluator_result, summary=summary)

    def _plot_comfort_panel(self, ax, comfort):
        if isinstance(comfort, ScorePanel):
            draw_comfort_panels(ax, comfort)
            return
        return super()._plot_comfort_panel(ax, comfort)

    def _plot_ttc_violation(self, ax, ttc_violation):
        # 원본과 같은 그림이지만 좌표 변환을 콜러블로 주입한다.
        _draw_one_ttc(ax, ttc_violation, self._to_local)
