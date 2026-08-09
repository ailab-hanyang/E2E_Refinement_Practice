#!/usr/bin/env bash
# 실습 4 — 같은 시나리오를 ML 주행 / Refine 주행으로 각각 돌린 뒤 비교한다.
#
# 사용법: bash script/compare_ml_vs_refine.sh [ADAPTER] [FILTER] [NUM_WORKER]
#
# 두 번 돌리는 이유: 한 스텝에 실행되는 궤적은 하나뿐이라, "refine 으로 계속 주행하면
# 최종 점수가 오르는가" 는 실제로 그렇게 주행해 봐야 알 수 있다.
set -uo pipefail

# 아래에서 두 런의 출력 디렉토리를 찾으려면 NUPLAN_EXP_ROOT 가 이 셸에 있어야 한다.
source "$(dirname "$0")/nuplan_env.sh"

ADAPTER=${1:-pluto}
FILTER=${2:-practice_scenarios}
NUM_WORKER=${3:-8}

# 두 런이 같은 스탬프를 공유해야 짝을 찾을 수 있다.
export EXP_STAMP=$(date +%m%d_%H%M%S)

echo "############ [1/2] driving_policy=ml ############"
bash "$(dirname "$0")/run_refinement_planner.sh" "$ADAPTER" ml "$FILTER" "$NUM_WORKER"

echo "############ [2/2] driving_policy=refine ############"
bash "$(dirname "$0")/run_refinement_planner.sh" "$ADAPTER" refine "$FILTER" "$NUM_WORKER"

RUN_ROOT="${NUPLAN_EXP_ROOT:-$(pwd)/data/exp}"
BASE="${RUN_ROOT}/exp/simulation/closed_loop_nonreactive_agents/refinement_planner/${FILTER}"
echo ""
echo "############ 비교 ############"
python tool/compare_runs.py \
    "${BASE}/${EXP_STAMP}_${ADAPTER}_ml" \
    "${BASE}/${EXP_STAMP}_${ADAPTER}_refine" \
    --videos "videos/compare_${EXP_STAMP}_${ADAPTER}"
