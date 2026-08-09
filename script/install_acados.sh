#!/usr/bin/env bash
# acados 설치 (실습 2 MPC 전용). 없으면 planner 가 ML-only 로 강등되므로 실습 1·3·4 는 진행된다.
#
#   bash script/install_acados.sh                          # <repo>/Trajectory_refinement/acados
#   ACADOS_ROOT=$HOME/acados bash script/install_acados.sh  # 다른 위치
#
# 설치 후: source script/nuplan_env.sh && bash script/build_mpc.sh
# 빌드 산출물은 설치한 OS 에 묶인다 — 다른 배포판으로 옮기면 다시 빌드해야 한다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACADOS_ROOT="${ACADOS_ROOT:-${REPO_ROOT}/Trajectory_refinement/acados}"
ACADOS_VERSION="${ACADOS_VERSION:-v0.5.4}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "=================================================="
echo "| acados 설치"
echo "| install : ${ACADOS_ROOT}"
echo "| version : ${ACADOS_VERSION}   jobs: ${JOBS}"
echo "=================================================="

# 1) 빌드 도구 — 이미 있으면 sudo 를 요구하지 않는다.
missing=""
for c in cmake git make gcc; do
    command -v "$c" >/dev/null 2>&1 || missing="${missing} ${c}"
done
if [ -n "${missing}" ]; then
    echo "=== 사전 패키지 설치 (sudo 필요):${missing} ==="
    if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y build-essential cmake git pkg-config libblas-dev liblapack-dev
    else
        echo "❌ sudo/apt-get 이 없습니다 —${missing} 를 직접 설치한 뒤 다시 실행하세요."
        exit 1
    fi
else
    echo "[1/5] 빌드 도구 확인 완료"
fi

# 2) 소스 — blasfeo/hpipm/qpOASES 가 서브모듈이라 --recursive 가 빠지면 링크 단계에서 터진다.
if [ ! -d "${ACADOS_ROOT}/.git" ]; then
    echo "[2/5] clone ${ACADOS_VERSION}"
    git clone --branch "${ACADOS_VERSION}" --depth 1 --recursive \
        https://github.com/acados/acados.git "${ACADOS_ROOT}"
else
    echo "[2/5] 이미 존재 — 서브모듈만 동기화"
    git -C "${ACADOS_ROOT}" submodule update --recursive --init
fi

# 3) C 라이브러리 — ACADOS_ROOT 안에 설치해서 sudo 를 피한다.
echo "[3/5] cmake + make install -j${JOBS}"
mkdir -p "${ACADOS_ROOT}/build"
cmake -S "${ACADOS_ROOT}" -B "${ACADOS_ROOT}/build" \
      -DACADOS_WITH_QPOASES=ON -DCMAKE_INSTALL_PREFIX="${ACADOS_ROOT}"
make -C "${ACADOS_ROOT}/build" install "-j${JOBS}"
[ -f "${ACADOS_ROOT}/lib/libacados.so" ] || {
    echo "❌ libacados.so 생성 실패: ${ACADOS_ROOT}/lib"; exit 1; }

# 4) 파이썬 인터페이스는 pip install -e 하지 않는다 (conda 환경 전역을 오염시킨다).
#    nuplan_env.sh 가 PYTHONPATH 로 붙이므로 여기서는 의존 패키지만 확인한다.
echo "[4/5] acados_template 의존 패키지 확인"
PYTHONPATH="${ACADOS_ROOT}/interfaces/acados_template:${PYTHONPATH:-}" \
ACADOS_SOURCE_DIR="${ACADOS_ROOT}" \
python - <<'PY'
import importlib, subprocess, sys
# key=import 이름, value=pip 이름. wrapt 로 Deprecated 를 검사하면 안 된다 —
# 의존성일 뿐이라 검사만 통과하고 acados_template 이 'No module named deprecated' 로 죽는다.
need = {"numpy": "numpy", "scipy": "scipy", "casadi": "casadi",
        "matplotlib": "matplotlib", "Cython": "cython", "deprecated": "Deprecated"}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("  설치: " + " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
import acados_template
print("  acados_template :", acados_template.__file__)
PY

# 5) t_renderer 를 미리 받아 둔다. 없으면 codegen 중 input() 으로 멈추고,
#    기본 v0.2.0 은 GLIBC 2.34 를 요구해 구버전 OS 에서 "Rendering ... failed" 로 죽는다.
echo "[5/5] t_renderer 준비"
PYTHONPATH="${ACADOS_ROOT}/interfaces/acados_template:${PYTHONPATH:-}" \
ACADOS_SOURCE_DIR="${ACADOS_ROOT}" \
python - <<'PY'
import subprocess
from acados_template.utils import get_tera

def usable(path):
    # 인자 없이 실행하면 tera 는 panic 한다(정상). GLIBC 오류만 걸러낸다.
    p = subprocess.run([path], capture_output=True, text=True)
    err = (p.stderr or "") + (p.stdout or "")
    return "GLIBC" not in err and "cannot execute" not in err

path = get_tera(force_download=True)
if not usable(path):
    print("  기본 t_renderer 실행 불가 (glibc) — v0.0.34 로 재시도")
    path = get_tera(tera_version="0.0.34", force_download=True)
    if not usable(path):
        raise SystemExit("❌ t_renderer 를 실행할 수 없습니다: " + path)
print("  t_renderer :", path)
PY

echo ""
echo "✅ acados 설치 완료: ${ACADOS_ROOT}"
if [ "${ACADOS_ROOT}" != "${REPO_ROOT}/Trajectory_refinement/acados" ]; then
    echo "    export ACADOS_ROOT=${ACADOS_ROOT}   # 기본 경로가 아니므로 필요"
fi
echo "    source script/nuplan_env.sh && bash script/build_mpc.sh"
