"""
Open-loop 런 전용 렌더러.

Open-loop 런에서 채점되는 것은 "계획 궤적이 로그 ego 와 얼마나 어긋났는가" 인데,
`EvalSceneRender` 는 Closed-loop 지표(충돌·주행가능영역·comfort)를 그린다.
그 런에서 쓰이지 않는 숫자가 화면을 채우는 셈이라, 점수 관련 두 자리만 바꾼다.

    좌하단  Closed-loop 점수표      → Open-loop 지표표 (ADE/FDE/AHE/FHE/MR/SCORE)
    우하단  comfort 패널            → 계획 궤적 80 스텝의 변위·헤딩 오차 곡선

지도·agent·궤적 렌더는 `EvalSceneRender` 와 완전히 같다.

사용:

    render = OpenLoopSceneRender()
    render.render_dagger_scene_from_simulation(
        ...,
        evaluator_result=panel,      # ScorePanel — open_loop 이 채워져 있어야 한다
        reason_detail={"comfort": panel, "ttc": ..., "collision": ...},
    )
"""

from src.feature_builders.eval_scene_render import EvalSceneRender
from src.planners.evaluator.viz import (
    ScorePanel,
    draw_open_loop_error_panels,
    draw_open_loop_table,
)


class OpenLoopSceneRender(EvalSceneRender):
    """점수 관련 두 자리를 Open-loop 지표로 대체한 렌더러."""

    def _plot_evaluator_result(self, ax, evaluator_result, summary=None):
        if isinstance(evaluator_result, ScorePanel) and evaluator_result.open_loop:
            if summary and not evaluator_result.summary:
                evaluator_result.summary = summary
            draw_open_loop_table(ax, evaluator_result)
            # 위반 사유 상자는 Closed-loop 규칙(충돌·주행가능영역 등) 전용이라 생략한다.
            return
        # open_loop 이 비어 있으면(채점기 옵션이 꺼진 런) 기존 화면 그대로 둔다.
        return super()._plot_evaluator_result(ax, evaluator_result, summary=summary)

    def _plot_comfort_panel(self, ax, comfort):
        if isinstance(comfort, ScorePanel) and comfort.open_loop:
            draw_open_loop_error_panels(ax, comfort)
            return
        return super()._plot_comfort_panel(ax, comfort)
