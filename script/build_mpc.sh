#!/usr/bin/env bash
# acados MPC 코드 생성 + 빌드. refinement 실습 전에 한 번 실행한다.
#
# 별도 스크립트인 이유: yaml 의 b_acados_generation 을 true 로 둔 채 시뮬을 돌리면
# RefinementMPC.__setstate__ 가 워커마다 __init__() 을 다시 불러 병렬 워커들이 같은
# 디렉토리에 동시 코드 생성을 시도한다. 빌드는 여기서 한 번, 시뮬은 false 로 돌린다.
set -uo pipefail

source "$(dirname "$0")/nuplan_env.sh"

if [ -z "${ACADOS_SOURCE_DIR:-}" ]; then
    echo "❌ acados 를 찾지 못했습니다. ACADOS_ROOT 를 확인하거나 다음을 실행하세요:"
    echo "     bash script/install_acados.sh"
    exit 1
fi

MPC_DIR="${PWD}/Trajectory_refinement/refinementMPC"
SO="${MPC_DIR}/c_generated_code/libacados_ocp_solver_kinematic_model.so"

echo "=================================================="
echo "| acados MPC build"
echo "| workspace : ${PWD}"
echo "| ACADOS    : ${ACADOS_SOURCE_DIR}"
echo "| python    : $(which python)  ($(python -V 2>&1))"
echo "=================================================="

if [ -f "$SO" ]; then
    echo "[before] $(stat -c '%y  %s bytes' "$SO")"
else
    echo "[before] .so 없음 (최초 빌드)"
fi

# 추적 중인 yaml 은 false 로 두고, true 로 덮어쓴 임시 사본을 넘긴다.
TMP_YAML="$(mktemp /tmp/refinement_mpc_params_build_XXXXXX.yaml)"
python - "$MPC_DIR/refinement_mpc_params.yaml" "$TMP_YAML" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
new, n = re.subn(r'(?m)^(b_acados_generation:)\s*\S+', r'\1 true', text)
assert n == 1, f"b_acados_generation 치환 실패 (n={n})"
open(dst, 'w').write(new)
print(f"[yaml] 임시 설정 생성: {dst}  (b_acados_generation: true)")
PY
[ $? -eq 0 ] || { echo "임시 yaml 생성 실패"; exit 1; }

echo ""
echo "--- 코드 생성 + 컴파일 ---"
python - "$TMP_YAML" <<'PY'
import sys, time
from Trajectory_refinement.refinementMPC.refinement_mpc_solver import RefinementMPC

t0 = time.time()
mpc = RefinementMPC(yaml_path=sys.argv[1])   # __init__ 안에서 codegen + build 수행
assert mpc.initialized, "RefinementMPC 초기화 실패"

opt = mpc.ocp.solver_options
print(f"[opts] nlp_solver_type        = {opt.nlp_solver_type}")
print(f"[opts] nlp_solver_max_iter    = {opt.nlp_solver_max_iter}")
print(f"[opts] qp_solver_iter_max     = {opt.qp_solver_iter_max}")
print(f"[opts] tol                    = {opt.tol}")
print(f"[opts] nlp_solver_step_length = {opt.nlp_solver_step_length}")
print(f"[build] {time.time() - t0:.1f} s")
PY
RC=$?
rm -f "$TMP_YAML"

if [ $RC -ne 0 ]; then
    echo ""
    echo "❌ 빌드 실패 (rc=$RC). 기존 .so 는 그대로 남아 있습니다."
    exit $RC
fi

echo ""
if [ -f "$SO" ]; then
    echo "[after ] $(stat -c '%y  %s bytes' "$SO")"
else
    echo "❌ .so 가 생성되지 않았습니다: $SO"
    exit 1
fi

echo ""
echo "✅ 빌드 완료. refinement_mpc_params.yaml 의 b_acados_generation 은 false 그대로입니다"
echo "   (시뮬 워커는 이 .so 를 로드만 합니다)"
