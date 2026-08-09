#!/usr/bin/env bash
# ML 추론 + MPC refinement 실습 런.
#
# 사용법: bash script/run_refinement_planner.sh [ADAPTER] [DRIVING_POLICY] [FILTER] [NUM_WORKER]
#   ADAPTER        pluto | diffusion               (기본 pluto)
#   DRIVING_POLICY ml | refine                     (기본 ml)
#   FILTER         config/scenario_filter/ 의 이름 (기본 practice_scenarios)
#   NUM_WORKER     0 이면 sequential, 1 이상이면 ray_distributed 워커 수 (기본 8)
#
# 스텝당 ~4 초(MPC + 채점 2회 + 렌더). 디버깅할 때만 NUM_WORKER=0 으로 둔다.
source "$(dirname "$0")/nuplan_env.sh"

ADAPTER=${1:-pluto}
DRIVING_POLICY=${2:-ml}
FILTER=${3:-practice_scenarios}
NUM_WORKER=${4:-8}
VIDEO_SAVE_DIR=${5:-videos/${ADAPTER}_${DRIVING_POLICY}}

# 런마다 출력 디렉토리 분리. 같은 uid 면 Hydra 가 디렉토리를 재사용해 CSV 가 누적된다.
STAMP=${EXP_STAMP:-$(date +%m%d_%H%M%S)}

CHALLENGE="closed_loop_nonreactive_agents"

if [ "$NUM_WORKER" -le 0 ]; then
    WORKER_ARGS=(worker=sequential)
else
    # ⚠ ray 워커는 각자 RefinementMPC 를 새로 초기화한다.
    #   refinement_mpc_params.yaml 의 b_acados_generation 이 반드시 false 여야 한다.
    WORKER_ARGS=(
        worker=ray_distributed
        worker.threads_per_node="$NUM_WORKER"
        distributed_mode='SINGLE_NODE'
        number_of_gpus_allocated_per_simulation=0.15
        enable_simulation_progress_bar=true
    )
fi

python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=refinement_planner \
    planner/model_adapter=$ADAPTER \
    scenario_builder=nuplan \
    scenario_filter=$FILTER \
    "${WORKER_ARGS[@]}" \
    verbose=true \
    experiment_uid="refinement_planner/${FILTER}/${STAMP}_${ADAPTER}_${DRIVING_POLICY}" \
    planner.refinement_planner.driving_policy="$DRIVING_POLICY" \
    planner.refinement_planner.render=true \
    +planner.refinement_planner.save_dir="$VIDEO_SAVE_DIR"
