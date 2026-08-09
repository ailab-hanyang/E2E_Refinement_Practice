#!/usr/bin/env bash
# 실습 환경 구축. 저장소 루트에서 한 번만 실행한다.
#
#   bash script/setup_env.sh
#   ENV_NAME=my_env bash script/setup_env.sh
#
# conda 환경 → 파이썬 패키지 → natten → acados → MPC 솔버 순서로 진행한다.
# 데이터(data/db, data/maps, data/model)는 별도로 넣어야 한다.
# 단계별로 확인하며 진행하려면 practice/practice0_environment_setting.ipynb 를 쓴다.
set -euo pipefail

ENV_NAME="${ENV_NAME:-e2e_refinement}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 0) conda — 없을 때만 설치한다. -b -p 가 없으면 라이선스 동의에서 멈춘다.
if ! command -v conda >/dev/null 2>&1; then
    echo "=== miniconda 설치 (${HOME}/miniconda3) ==="
    INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    curl -fsSL -o "${INSTALLER}" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "${INSTALLER}" -b -p "${HOME}/miniconda3"
    rm -f "${INSTALLER}"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
fi

# conda activate 는 셸 함수라 이 줄이 있어야 스크립트 안에서 동작한다.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1) conda 환경.
#    `python==3.9` 로 쓰면 3.9.0 이 잡히는데, 3.9.0 의 typing 버그(PEP 585 제네릭 해시)로
#    aiohttp import 가 깨져 시뮬이 시작조차 못 한다. 반드시 `=` 하나로 최신 3.9.x 를 받는다.
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "=== conda 환경 '${ENV_NAME}' 재사용 ==="
else
    echo "=== conda 환경 '${ENV_NAME}' 생성 ==="
    conda create -n "${ENV_NAME}" python=3.9 pip=24.0 -y
fi
conda activate "${ENV_NAME}"
echo "python : $(python -V 2>&1)  ($(which python))"

# 2) 파이썬 패키지. natten 은 전용 인덱스에서 받으며 torch 설치 후여야 한다.
echo "=== requirements.txt 설치 ==="
pip install -r ./requirements.txt
echo "=== natten 설치 ==="
pip install natten==0.14.6+torch200cu118 -f https://whl.natten.org/old

# 3) acados + MPC 솔버. 실패해도 ML-only 로는 돌아가므로 죽이지 않는다.
echo "=== acados 설치 ==="
if bash ./script/install_acados.sh; then
    # shellcheck disable=SC1091
    source ./script/nuplan_env.sh
    echo "=== MPC 솔버 빌드 ==="
    bash ./script/build_mpc.sh || echo "⚠ MPC 솔버 빌드 실패 — ML-only 로는 동작합니다."
else
    echo "⚠ acados 설치 실패 — refinement 없이(ML-only) 동작합니다."
fi

echo ""
echo "=== 설치 확인 ==="
python - <<'PY'
import importlib
mods = ["torch", "torchvision", "pytorch_lightning", "torchmetrics", "timm", "natten",
        "numba", "imageio", "imageio_ffmpeg", "cv2", "hydra", "ray", "shapely",
        "geopandas", "casadi", "numpy", "pandas"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("  OK   %-20s %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:
        bad.append(m); print("  FAIL %-20s %s" % (m, e))
import torch
print("  torch.cuda.is_available() =", torch.cuda.is_available())
raise SystemExit(1 if bad else 0)
PY

cat <<EOF

✅ 환경 구축 완료.

    conda activate ${ENV_NAME}
    source script/nuplan_env.sh
    bash script/run_refinement_planner.sh pluto ml

데이터는 따로 넣어야 합니다:
    data/db/<이름>/*.db   (넣은 뒤 devkit 의 nuplan.yaml 에 경로 추가)
    data/maps/<지도>/
    data/model/pluto_planner.ckpt
EOF
