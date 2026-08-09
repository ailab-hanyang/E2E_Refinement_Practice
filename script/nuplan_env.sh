# 실습 환경변수. 저장소 루트에서 `source script/nuplan_env.sh`.
CUR_DIR=$(pwd)

export NUPLAN_DATA_ROOT="${CUR_DIR}/data"
export NUPLAN_MAPS_ROOT="${CUR_DIR}/data/maps"
export NUPLAN_EXP_ROOT="${CUR_DIR}/data"

# 모델 경로는 DATA_ROOT 를 외부 데이터셋으로 덮어써도 따라가면 안 되므로 별도 변수.
export PRACTICE_MODEL_ROOT="${CUR_DIR}/data/model"

# 벤더링된 nuplan/ devkit 과 Trajectory_refinement/ 를 저장소 루트에서 찾게 한다.
export PYTHONPATH="${CUR_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# acados 는 실습 2(MPC)에만 필요하다. 없으면 planner 가 ML-only 로 강등된다.
export ACADOS_ROOT="${ACADOS_ROOT:-${CUR_DIR}/Trajectory_refinement/acados}"
if [ -f "${ACADOS_ROOT}/lib/libacados.so" ]; then
    export ACADOS_SOURCE_DIR="${ACADOS_ROOT}"
    export LD_LIBRARY_PATH="${ACADOS_ROOT}/lib:${LD_LIBRARY_PATH:-}"
    # PYTHONPATH 가 site-packages 보다 먼저 검색되는 것을 이용해, 다른 프로젝트의
    # acados_template editable 설치를 이 셸에서만 덮어쓴다 (C 라이브러리와 버전 일치).
    export PYTHONPATH="${ACADOS_ROOT}/interfaces/acados_template:${PYTHONPATH:-}"
else
    echo "[nuplan_env] acados 미발견 (${ACADOS_ROOT}) — refinement 없이 동작합니다."
    echo "             설치: bash script/install_acados.sh"
fi
