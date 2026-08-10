"""practice3 노트북용 헬퍼.

반복적인 것만 모아 둔다. 필터를 거는 코드나 MPC 를 들여다보는 코드처럼
후처리를 이해하는 데 필요한 것은 노트북 본문에 직접 쓴다.

⚠ 파일 이름은 저장소 안에서 유일해야 한다. 노트북이
  `REPO_ROOT.glob("practice/**/_practice3_helper.py")` 로 찾기 때문에
  같은 이름이 둘이면 엉뚱한 파일을 집는다.

부트스트랩·시나리오·결과 로더는 `_practice2_helper.py` 에서 복사했다. 교차 import
하면 실습 2 를 손볼 때 실습 3 이 깨진다.

실습 2 와 다른 점은 두 가지다.

    run_sim()       후처리 스위치(smoothing / use_refinement / driving_policy)를
                    인자로 받는다. 실습 3 은 이 스위치만 바꿔 가며 같은 시나리오를
                    여러 번 돌리는 실습이다.
    compare_runs()  여러 런의 결과를 한 표로 모은다. 없어진 tool/compare_runs.py
                    자리를 대신한다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# 부트스트랩 — devkit 을 import 하기 전에 반드시 먼저 부른다
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
    """run_simulation.py 가 있는 상위 디렉토리를 저장소 루트로 본다."""
    for d in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
        if (d / "run_simulation.py").exists():
            return d
    raise RuntimeError("저장소 루트를 찾지 못했습니다.")


def bootstrap(verbose: bool = True) -> Path:
    """script/nuplan_env.sh 와 같은 환경변수를 이 파이썬 프로세스에 넣는다.

    setdefault 를 쓰면 안 된다. 셸에 다른 nuPlan 데이터셋 경로가 이미 export 되어
    있으면 그쪽이 이겨서, 노트북이 저장소의 data/ 가 아닌 곳을 본다.
    """
    repo = find_repo_root()
    os.environ["NUPLAN_DATA_ROOT"] = str(repo / "data")
    os.environ["NUPLAN_MAPS_ROOT"] = str(repo / "data/maps")
    os.environ["NUPLAN_EXP_ROOT"] = str(repo / "data")
    os.environ["PRACTICE_MODEL_ROOT"] = str(repo / "data/model")
    os.environ["HYDRA_FULL_ERROR"] = "1"

    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))

    # §0에서 설치한 acados를 커널 재시작 뒤 바로 찾게 한다.
    acados = repo / "Trajectory_refinement/acados"
    if (acados / "lib/libacados.so").exists():
        os.environ["ACADOS_ROOT"] = str(acados)
        os.environ["ACADOS_SOURCE_DIR"] = str(acados)
        lib = str(acados / "lib")
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = lib + (f":{old_ld}" if old_ld else "")
        template = str(acados / "interfaces/acados_template")
        if template not in sys.path:
            sys.path.insert(0, template)

        # Linux의 dlopen 은 프로세스 시작 뒤에 바꾼 LD_LIBRARY_PATH 를 항상 따라주지 않는다.
        # acados 가 쓰는 의존 라이브러리를 절대 경로로 먼저 올려 두면, 이후 AcadosOcpSolver 로드가 안정적이다.
        import ctypes

        for so_name in ("libblasfeo.so", "libblasfeo.so.0", "libblasfeo.so.0.1.4.2",
                        "libhpipm.so", "libqpOASES_e.so", "libacados.so"):
            so_path = acados / "lib" / so_name
            if so_path.exists():
                ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)

    if verbose:
        print("REPO_ROOT        :", repo)
        print("NUPLAN_DATA_ROOT :", os.environ["NUPLAN_DATA_ROOT"])
    return repo


def run(cmd: str, cwd=None) -> int:
    """bash 로 실행하고 출력을 실시간으로 흘려보낸다."""
    print(f"$ {cmd}\n" + "-" * 78, flush=True)
    proc = subprocess.Popen(
        ["bash", "-lc", cmd], cwd=str(cwd or find_repo_root()),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    print("-" * 78 + f"\n{'✅' if rc == 0 else '❌'} exit={rc}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# 그림 — 이 실습은 축 라벨에 한글을 많이 쓴다
# ---------------------------------------------------------------------------

#: 흔한 한글 폰트. 앞에서부터 찾아 처음 걸리는 것을 쓴다.
#: `Noto Sans CJK KR` 은 일부러 빼 두었다 — matplotlib 은 .ttc 컬렉션의 첫 face 만
#: 등록하므로 NotoSansCJK-Regular.ttc 는 리눅스에서 늘 `Noto Sans CJK JP` 라는 이름으로
#: 잡힌다. 같은 pan-CJK 글리프 세트라 그 이름으로 써도 한글은 그대로 나온다.
_KOREAN_FONTS = ("NanumGothic", "NanumBarunGothic", "Malgun Gothic", "Noto Sans CJK JP",
                 "Noto Sans KR", "AppleGothic", "UnDotum", "Baekmuk Gulim", "D2Coding")

#: 위 폰트들의 파일 이름 조각. 캐시에 없는 폰트를 직접 등록할 때 쓴다.
_KOREAN_FONT_FILES = ("nanum", "notosanscjk", "notoserifcjk", "malgun", "notosanskr",
                      "gulim", "batang", "d2coding")


def setup_korean_font(verbose: bool = True) -> Optional[str]:
    """matplotlib 에 한글 폰트를 물린다. 없으면 조용히 넘어간다.

    설정하지 않으면 축 라벨의 한글이 네모(□)로 나온다. 그림 자체는 정상이므로
    치명적이지는 않지만, 실습 중에 "그림이 깨졌다" 는 질문이 반복해서 나온다.

    ⚠ `fontManager.ttflist` 는 `~/.cache/matplotlib/fontlist-*.json` 캐시를 그대로 읽은
      것이다. 캐시를 만든 뒤에 폰트를 설치했다면 목록에 영영 나타나지 않아, 폰트가
      깔려 있는데도 tofu 가 난다. 그래서 디스크를 먼저 훑어 등록한 다음 고른다.

    :return: 사용하기로 한 폰트 이름. 못 찾았으면 None.
    """
    import matplotlib
    from matplotlib import font_manager

    # WSL 의 윈도우 폰트는 findSystemFonts() 가 훑지 않는 위치라 따로 붙인다.
    known = {f.fname for f in font_manager.fontManager.ttflist}
    for path in (*font_manager.findSystemFonts(), "/mnt/c/Windows/Fonts/malgun.ttf"):
        name = Path(path).name.lower().replace(" ", "").replace("-", "")
        if path in known or not any(k in name for k in _KOREAN_FONT_FILES):
            continue
        try:
            font_manager.fontManager.addfont(path)
        except Exception:  # 깨진 폰트 파일 하나 때문에 실습이 멈추면 안 된다.
            pass

    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = next((f for f in _KOREAN_FONTS if f in available), None)

    if picked:
        matplotlib.rcParams["font.family"] = picked
        # 한글 폰트는 유니코드 마이너스(−)를 대개 갖고 있지 않다. ASCII 하이픈을 쓴다.
        matplotlib.rcParams["axes.unicode_minus"] = False
        if verbose:
            print("matplotlib 한글 폰트:", picked)
    elif verbose:
        print("matplotlib 한글 폰트를 찾지 못했습니다 — 그림의 한글이 □ 로 나옵니다.")
        print("  설치 예: sudo apt install fonts-nanum  (설치 후 커널 재시작)")
    return picked


# ---------------------------------------------------------------------------
# MPC 사용 가능 여부 — 실습 3 은 여기서 갈린다
# ---------------------------------------------------------------------------

#: build_mpc.sh 가 만드는 공유 라이브러리. 이것이 없으면 MPC 절을 돌릴 수 없다.
MPC_SO_REL = (
    "Trajectory_refinement/refinementMPC/c_generated_code"
    "/libacados_ocp_solver_kinematic_model.so"
)


def mpc_status(verbose: bool = True) -> bool:
    """MPC 를 쓸 수 있는 상태인지 확인한다.

    §2~§4 (smoothing filter) 는 acados 없이도 전부 돌아간다. 여기서 False 가 나와도
    노트북 절반은 그대로 진행할 수 있으므로, 예외를 던지지 않고 안내만 한다.
    """
    repo = find_repo_root()
    so = repo / MPC_SO_REL

    try:
        import acados_template  # noqa: F401
        has_template = True
    except Exception:
        has_template = False

    ok = so.exists() and has_template
    if verbose:
        print("acados_template :", "OK" if has_template else "없음")
        print("MPC 솔버(.so)   :",
              f"OK ({so.stat().st_size / 1e6:.1f} MB)" if so.exists() else "없음")
        if not ok:
            print()
            print("  MPC 절(§5~§8)을 돌리려면 아래를 먼저 실행하십시오:")
            if not has_template:
                print("    bash script/install_acados.sh")
            print("    bash script/build_mpc.sh")
            print()
            print("  §2~§4(smoothing filter)는 acados 없이 그대로 진행됩니다.")
    return ok


# ---------------------------------------------------------------------------
# 시나리오
# ---------------------------------------------------------------------------


def load_scenario_mapping(repo: Path):
    """devkit 의 nuplan_scenario_mapping.yaml 을 그대로 읽어 ScenarioMapping 을 만든다.

    이것이 없으면 devkit 은 로그를 자르지도 솎아내지도 않아 20 s @ 0.05 s(400 스텝)가
    나오고, 있으면 15 s @ 0.1 s(150 스텝) — 공식 챌린지와 같은 조건이 된다.
    """
    from omegaconf import OmegaConf

    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

    cfg = OmegaConf.load(
        repo / "nuplan/planning/script/config/common/scenario_builder"
        "/scenario_mapping/nuplan_scenario_mapping.yaml"
    )
    return ScenarioMapping(
        scenario_map=OmegaConf.to_container(cfg.scenario_map),
        subsample_ratio_override=cfg.subsample_ratio_override,
    )


def make_builder(repo: Path, data_root: Optional[str] = None, scenario_mapping=None):
    """NuPlanScenarioBuilder 를 만든다. data_root 기본값은 data/db/test."""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )

    return NuPlanScenarioBuilder(
        data_root=data_root or str(repo / "data/db/test"),
        map_root=str(repo / "data/maps"),
        sensor_root=str(repo / "data/sensor_blobs"),
        db_files=None,
        map_version="nuplan-maps-v1.0",
        scenario_mapping=scenario_mapping if scenario_mapping is not None
        else load_scenario_mapping(repo),
    )


def make_filter(**overrides):
    """ScenarioFilter 를 만든다. 지정하지 않은 필드는 '끄기'(None) 가 기본이다."""
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    base = dict(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None, limit_total_scenarios=None,
        timestamp_threshold_s=None, ego_displacement_minimum_m=None,
        expand_scenarios=False, remove_invalid_goals=True, shuffle=False,
        ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    base.update(overrides)
    return ScenarioFilter(**base)


def scenario_table(scenarios: list):
    """시나리오 목록을 한눈에 보는 표."""
    import pandas as pd

    return pd.DataFrame([
        {"idx": i, "token": s.token, "scenario_type": s.scenario_type,
         "log_name": s.log_name[:28], "iterations": s.get_number_of_iterations(),
         "dt": s.database_interval}
        for i, s in enumerate(scenarios)
    ])


LQR_DEFAULT = dict(
    q_longitudinal=[10.0], r_longitudinal=[1.0],
    q_lateral=[1.0, 10.0, 0.0], r_lateral=[1.0],
    discretization_time=0.1, tracking_horizon=10,
    jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
    stopping_proportional_gain=0.5, stopping_velocity=0.2,
)


def make_sim(scenario, ego_controller=None, observations=None):
    """4축을 채워 Simulation 을 만든다 (실습 1 §5 에서 손으로 쓴 것과 같은 코드)."""
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
        KinematicBicycleModel,
    )
    from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
    from nuplan.planning.simulation.simulation import Simulation
    from nuplan.planning.simulation.simulation_setup import SimulationSetup
    from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
        StepSimulationTimeController,
    )

    if ego_controller is None:
        ego_controller = TwoStageController(
            scenario, LQRTracker(**LQR_DEFAULT),
            KinematicBicycleModel(scenario.ego_vehicle_parameters),
        )
    setup = SimulationSetup(
        time_controller=StepSimulationTimeController(scenario),
        observations=observations or TracksObservation(scenario),
        ego_controller=ego_controller,
        scenario=scenario,
    )
    return Simulation(setup)


# ---------------------------------------------------------------------------
# PLUTO 구성 상수 — config/planner/model_adapter/pluto.yaml 을 그대로 옮긴 것이다.
# 값을 바꿔야 하면 그 yaml 이 기준이다.
# ---------------------------------------------------------------------------

PLUTO_CKPT = "data/model/pluto_planner.ckpt"

PLUTO_FEATURE_KWARGS = dict(
    radius=120, history_horizon=2, future_horizon=8,
    sample_interval=0.1, max_agents=48, build_reference_line=True,
)

PLUTO_MODEL_KWARGS = dict(
    dim=128, state_channel=6, polygon_channel=6, history_channel=9,
    history_steps=21, future_steps=80, encoder_depth=4, decoder_depth=4,
    drop_path=0.2, dropout=0.1, num_heads=4, num_modes=12,
    use_ego_history=False, state_attn_encoder=True, state_dropout=0.75,
    use_hidden_proj=True, cat_x=True, ref_free_traj=True,
)


# ---------------------------------------------------------------------------
# 시뮬레이션 실행
# ---------------------------------------------------------------------------

#: 검증용. 켜면 시나리오 1개·워커 1개로 낮춘다. 노트북 본문에는 분기를 두지 않는다.
FAST = bool(os.environ.get("PRACTICE3_FAST"))


def exp_dir(repo: Path, challenge: str, uid: str) -> Path:
    """Hydra 가 결과를 쌓는 디렉토리. default_experiment.yaml 의 규칙 그대로다."""
    return repo / "data/exp/simulation" / challenge / uid


def run_sim(
    uid: str,
    *,
    driving_policy: str = "ml",
    smoothing: str = "none",
    smoothing_cutoff_hz: float = 1.0,
    smoothing_savgol_window: int = 11,
    smoothing_savgol_polyorder: int = 3,
    use_refinement: bool = False,
    challenge: str = "closed_loop_nonreactive_agents",
    overrides: Sequence[str] = (),
    adapter: str = "pluto",
    scenario_filter: str = "practice_scenarios",
    limit: Optional[int] = None,
    n_workers: int = 6,
    video_dir: str = "videos",
    mode: str = "reuse",
) -> Path:
    """run_simulation.py 를 실행하고 결과 디렉토리를 돌려준다.

    실습 3 의 런은 후처리 스위치 세 개로 구분된다.

        driving_policy   실제로 **주행할** 궤적. 최종 점수를 바꾸는 것은 이것뿐이다.
        smoothing        평활 궤적을 만들지 여부. "none" 이 아니면 채점표에 sm 열이
                         생긴다. driving_policy="smooth" 면 이 궤적으로 주행한다.
        use_refinement   MPC 궤적을 만들지 여부. driving_policy="refine" 이면 이
                         궤적으로 주행한다.

    후처리를 켜되 주행은 ML 로 하는 조합(예: smoothing 켜고 driving_policy="ml")이
    쓸모 있다. 같은 주행 궤적 위에서 세 궤적을 **매 스텝 같은 조건으로** 채점한
    CSV 가 나오기 때문이다. 주행까지 바꾸면 스텝마다 자차 위치가 달라져 스텝 단위
    비교가 성립하지 않는다.

    override 순서를 강제한다. `+simulation=<프리셋>` 이 planner 그룹을 덮어쓰므로
    `planner=refinement_planner` 는 반드시 프리셋 뒤에 와야 한다.

    mode="reuse"  집계 parquet 이 이미 있으면 실행하지 않는다(기본).
    mode="rerun"  결과 디렉토리를 지우고 다시 실행한다. 같은 uid 로 재실행할 때
                  스텝별 CSV 가 누적되는 것을 막는다.
    """
    repo = find_repo_root()
    out = exp_dir(repo, challenge, uid)

    if mode == "rerun" and out.exists():
        shutil.rmtree(out)
    elif list(out.glob("aggregator_metric/*.parquet")):
        print(f"기존 결과를 재사용합니다: {out}")
        print('  다시 돌리려면 mode="rerun" 을 주십시오.')
        return out

    if FAST:
        limit, n_workers = 1, 1

    if n_workers and n_workers > 0:
        worker = [
            "worker=ray_distributed",
            f"worker.threads_per_node={n_workers}",
            "distributed_mode=SINGLE_NODE",
            "number_of_gpus_allocated_per_simulation=0.15",
        ]
    else:
        worker = ["worker=sequential"]

    planner_key = "planner.refinement_planner"
    args = [
        f"+simulation={challenge}",          # ① 프리셋이 planner 를 덮는다
        "planner=refinement_planner",        # ② 그래서 그 뒤에서 되돌린다
        f"planner/model_adapter={adapter}",
        "scenario_builder=nuplan",
        f"scenario_filter={scenario_filter}",
        *([f"scenario_filter.limit_total_scenarios={limit}"] if limit else []),
        *worker,
        f"experiment_uid={uid}",
        f"{planner_key}.driving_policy={driving_policy}",
        f"{planner_key}.smoothing={smoothing}",
        f"{planner_key}.smoothing_cutoff_hz={smoothing_cutoff_hz}",
        f"{planner_key}.smoothing_savgol_window={smoothing_savgol_window}",
        f"{planner_key}.smoothing_savgol_polyorder={smoothing_savgol_polyorder}",
        f"{planner_key}.use_refinement={str(use_refinement).lower()}",
        f"{planner_key}.render=true",
        f"{planner_key}.log_csv=true",
        f"+{planner_key}.save_dir={video_dir}",
        *overrides,
    ]
    cmd = ("source script/nuplan_env.sh && python run_simulation.py \\\n    "
           + " \\\n    ".join(args))

    t0 = time.time()
    rc = run(cmd, cwd=repo)
    print(f"소요 {time.time() - t0:.0f}s")
    if rc != 0:
        raise RuntimeError(f"시뮬레이션 실패 (exit={rc}). 위 로그를 확인하십시오.")
    return out


# ---------------------------------------------------------------------------
# 결과 읽기
# ---------------------------------------------------------------------------

#: 폐루프 집계표에서 점수에 기여하는 8개 열. 앞 4개가 곱셈 항이다.
CLOSED_BREAKDOWN = [
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "ego_is_comfortable",
]

#: 스텝별 CSV 의 후보 접두사 → 표시 이름.
CANDIDATES = {"ml": "ML", "sm": "Smoothed", "rf": "Refined(MPC)"}


def load_aggregate(run_dir: Path):
    """런 디렉토리에서 가장 최근 집계 parquet 을 읽는다."""
    import pandas as pd

    files = sorted(Path(run_dir).rglob("aggregator_metric/*.parquet"),
                   key=lambda p: p.stat().st_ctime)
    if not files:
        raise FileNotFoundError(f"집계 parquet 이 없습니다: {run_dir}")
    return pd.read_parquet(files[-1])


def final_score(run_dir: Path) -> float:
    """nuPlan 공식 최종 점수."""
    df = load_aggregate(run_dir)
    return float(df[df["scenario"] == "final_score"]["score"].iloc[0])


def per_scenario_scores(run_dir: Path, columns: Optional[Sequence[str]] = None):
    """시나리오별 점수 표.

    집계 parquet 에는 세 종류의 행이 섞여 있다 — 시나리오별, 시나리오 유형별 집계,
    그리고 final_score. 유형별 집계 행은 log_name 이 비어 있으므로 그것으로 거른다.
    거르지 않으면 같은 시나리오가 두 번 세어진다.
    """
    df = load_aggregate(run_dir)
    df = df[(df["scenario"] != "final_score") & (df["log_name"].notna())].copy()

    cols = ["scenario", "log_name", "scenario_type", "score"]
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"집계표에 없는 열입니다: {missing}")
        cols += list(columns)

    out = df[cols].rename(columns={"scenario": "token"})
    out.insert(0, "video", out["log_name"] + "_" + out["token"] + ".mp4")
    return out.sort_values("score").reset_index(drop=True)


def load_step_csv(run_dir: Path):
    """워커별 스텝 CSV를 합치고 재시도·재실행으로 생긴 중복을 제거한다."""
    import pandas as pd

    files = sorted(Path(run_dir).rglob("trajectory_evaluator_results/*.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("token", "iteration") if c in out.columns]
    if len(keys) == 2:
        n_before = len(out)
        out = out.drop_duplicates(keys, keep="last").reset_index(drop=True)
        out.attrs["duplicates_removed"] = n_before - len(out)
    return out


def runner_status(run_dir: Path):
    """시나리오별 성공 여부와 스텝 소요 시간."""
    import pandas as pd

    files = list(Path(run_dir).rglob("runner_report.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.read_parquet(files[0])
    cols = [c for c in ["scenario_name", "log_name", "succeeded", "error_message",
                        "duration", "compute_trajectory_runtimes_mean"] if c in df.columns]
    return df[cols].rename(columns={"scenario_name": "token"})


# ---------------------------------------------------------------------------
# 비교 — 없어진 tool/compare_runs.py 자리를 대신한다
# ---------------------------------------------------------------------------


def compare_runs(runs: Mapping[str, Path], columns: Optional[Sequence[str]] = None):
    """여러 런의 최종 점수와 지표 평균을 한 표로 모은다.

    각 런은 **끝까지 주행한 결과**다. 따라서 이 표가 답하는 질문은
    "이 후처리로 계속 주행하면 최종 점수가 오르는가" 이다.

    :param runs: {표시이름: 런 디렉토리}. dict 순서가 행 순서가 된다.
    """
    import pandas as pd

    cols = list(columns) if columns else CLOSED_BREAKDOWN
    rows = []
    for name, run_dir in runs.items():
        scores = per_scenario_scores(run_dir, columns=cols)
        row = {"run": name, "final_score": final_score(run_dir),
               "n_scenarios": len(scores)}
        row.update({c: scores[c].mean() for c in cols})
        rows.append(row)
    return pd.DataFrame(rows).set_index("run").round(4)


def compare_scenarios(runs: Mapping[str, Path]):
    """시나리오별 점수를 런끼리 나란히 놓는다. 어느 장면이 달라졌는지 본다.

    최종 점수는 시나리오 평균이라 한두 장면의 큰 변화를 가린다. 후처리가 무엇을
    고쳤는지(혹은 망가뜨렸는지) 는 이 표에서 읽어야 한다.
    """
    import pandas as pd

    out = None
    for name, run_dir in runs.items():
        s = per_scenario_scores(run_dir)[["token", "scenario_type", "score"]]
        s = s.rename(columns={"score": name})
        out = s if out is None else pd.merge(out, s, on=["token", "scenario_type"])

    names = list(runs)
    if len(names) > 1:
        out["Δ(last-first)"] = out[names[-1]] - out[names[0]]
    return out.sort_values(names[0]).reset_index(drop=True).round(4)


def step_comparison(run_dir: Path):
    """한 런 안에서 세 궤적을 **매 스텝 같은 조건으로** 견준 결과.

    compare_runs 와 답하는 질문이 다르다. 저쪽은 "그 궤적으로 주행하면" 이고,
    이쪽은 "같은 상황에서 그 궤적이 더 나은가" 다. 자차 위치가 모든 후보에 대해
    동일하므로 후처리 자체의 효과만 분리해서 볼 수 있다 — 표본도 스텝 수만큼
    많아서(시나리오 6개면 900 스텝 남짓) 시나리오 6개보다 훨씬 안정적이다.

    :return: 후보별 평균 점수, ML 대비 개선/악화 스텝 수.
    """
    import pandas as pd

    steps = load_step_csv(run_dir)
    if steps is None:
        return pd.DataFrame()

    rows = []
    ml = steps["ml_final"]
    for key, label in CANDIDATES.items():
        col = f"{key}_final"
        if col not in steps.columns:
            continue
        vals = pd.to_numeric(steps[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        delta = vals - ml
        rows.append({
            "candidate": label,
            "n_steps": int(vals.notna().sum()),
            "mean_score": vals.mean(),
            "mean_delta_vs_ml": delta.mean(),
            "better": int((delta > 1e-9).sum()),
            "worse": int((delta < -1e-9).sum()),
            "same": int((delta.abs() <= 1e-9).sum()),
        })
    return pd.DataFrame(rows).set_index("candidate").round(4)


def step_metric_deltas(run_dir: Path, candidate: str = "sm"):
    """지표별로 ML 대비 얼마나 달라졌는지 (스텝 평균).

    최종 점수 하나만 보면 "왜" 가 빠진다. comfort 는 올랐는데 collision 이 떨어져
    상쇄되는 경우가 실제로 나오므로, 항목별로 갈라서 본다.
    """
    import pandas as pd

    steps = load_step_csv(run_dir)
    if steps is None:
        return pd.DataFrame()

    rows = []
    for metric in CLOSED_BREAKDOWN:
        ml_col, cand_col = f"ml_{metric}", f"{candidate}_{metric}"
        if ml_col not in steps.columns or cand_col not in steps.columns:
            continue
        a = pd.to_numeric(steps[ml_col], errors="coerce")
        b = pd.to_numeric(steps[cand_col], errors="coerce")
        both = a.notna() & b.notna()
        if both.sum() == 0:
            continue
        rows.append({"metric": metric, "ml": a[both].mean(),
                     candidate: b[both].mean(), "delta": (b - a)[both].mean()})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("metric").sort_values("delta").round(4)


# ---------------------------------------------------------------------------
# 궤적 신호 시각화 — §2 와 §6 에서 같은 그림을 쓴다
# ---------------------------------------------------------------------------


def plot_trajectory_signals(
    trajectories: Mapping[str, "object"],
    dt: float = 0.1,
    title: str = "",
    colours: Optional[Mapping[str, str]] = None,
):
    """궤적들을 경로 · 곡률 · yaw rate 세 패널로 겹쳐 그린다.

    후처리가 무엇을 바꿨는지는 경로 그림만으로는 거의 보이지 않는다(두 궤적이
    수 cm 차이라 겹쳐 보인다). 곡률과 yaw rate 로 미분해야 떨림이 드러나며,
    그 떨림이 곧 comfort 점수를 깎는 신호다.

    :param trajectories: {이름: (T, >=3) ego-local 궤적}
    :param colours: {이름: 색}. 없으면 evaluator 의 후보 색을 따른다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from src.planners.postprocess import curvature, heading_rate

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (name, traj) in enumerate(trajectories.items()):
        traj = np.asarray(traj, dtype=float)
        colour = (colours or {}).get(name) or _candidate_colour(name, i)
        t = np.arange(len(traj)) * dt
        style = dict(color=colour, lw=1.6, label=name)

        axes[0].plot(traj[:, 0], traj[:, 1], **style)
        kappa = curvature(traj)
        kappa[:2] = np.nan; kappa[-2:] = np.nan  # 수치미분 경계 오차는 표시하지 않는다.
        axes[1].plot(t, kappa, **style)
        axes[2].plot(t, heading_rate(traj, dt), **style)

    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")
    axes[0].set_title("경로"); axes[0].axis("equal")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("κ [1/m]")
    axes[1].set_title("곡률 — 조향각과 직결된다")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("yaw rate [rad/s]")
    axes[2].set_title("yaw rate — comfort 항이 보는 값")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93) if title else None)
    return fig, axes


def _candidate_colour(name: str, index: int) -> str:
    """evaluator 의 후보 색을 쓰되, 그쪽을 import 할 수 없으면 순환색으로 떨어진다."""
    try:
        from src.planners.evaluator import candidate_colour

        return candidate_colour(name.split()[0].lower(), index)
    except Exception:
        return ("#666666", "#2E8B57", "#1F5FB4", "#B8860B")[index % 4]


# ---------------------------------------------------------------------------
# 영상 — 노트북 디렉토리 아래로 옮겨 놓아야 브라우저가 읽을 수 있다.
# ---------------------------------------------------------------------------

#: practice/videos/<tag>/ 에 모은다. .gitignore 의 videos/ 규칙에 이미 걸린다.
VIDEO_STAGE = "videos"


def collect_videos(run_dir: Path, tag: str) -> Dict[str, Optional[Path]]:
    """런의 mp4 를 practice/videos/<tag>/ 로 하드링크한다(안 되면 복사).

    JupyterLab 은 출력 HTML 의 상대경로를 **노트북 위치 기준**으로 푼다. 영상은
    data/exp/ 아래에 있어 노트북에서 `..` 없이 가리킬 수 없으므로 옮겨 온다.
    같은 파일시스템이면 하드링크라 용량도 시간도 들지 않는다.
    """
    stage = Path(__file__).resolve().parents[2] / VIDEO_STAGE / tag
    stage.mkdir(parents=True, exist_ok=True)

    found = {p.name: p for p in Path(run_dir).rglob("*.mp4")}
    out: Dict[str, Optional[Path]] = {}
    for name in sorted(per_scenario_scores(run_dir)["video"]):
        src = found.get(name)
        if src is None:
            out[name] = None
            continue
        dst = stage / name
        if dst.exists():
            same = os.path.samefile(src, dst)
            if not same:
                a, b = src.stat(), dst.stat()
                same = (a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns)
            if not same:
                dst.unlink()  # 같은 tag로 런을 다시 만들었으면 오래된 staging 영상을 교체한다.
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        out[name] = dst
    return out


def video_html(rel_path: str, width: int = 760) -> str:
    """<video> 태그 하나. rel_path 는 노트북 디렉토리 기준 상대경로여야 한다."""
    return (f'<video src="{rel_path}" width="{width}" controls loop '
            f'style="border:1px solid #ccc"></video>')


def scenario_video_browser(run_dir: Path, tag: str,
                           columns: Optional[Sequence[str]] = None,
                           width: int = 760):
    """시나리오 목록 + 선택 시 해당 mp4 재생 + 그 시나리오의 세부지표.

    위젯 상태는 노트북을 정적으로 변환하면 사라진다. 표는 별도 셀에서 한 번 더
    출력하고, 여기서는 영상 확인만 담당한다.
    """
    import ipywidgets as widgets
    from IPython.display import HTML, display

    scores = per_scenario_scores(run_dir, columns=columns)
    videos = collect_videos(run_dir, tag)
    status = runner_status(run_dir)

    options = [
        (f"{r.score:6.3f}  {r.token[:8]}  {r.scenario_type}", r.token)
        for r in scores.itertuples()
    ]
    select = widgets.Select(options=options, rows=min(10, max(3, len(options))),
                            layout=widgets.Layout(width="330px"))
    out = widgets.Output()

    def show(token: str) -> None:
        row = scores[scores["token"] == token].iloc[0]
        with out:
            out.clear_output(wait=True)
            path = videos.get(row["video"])
            if path is None:
                print(f"영상이 없습니다: {row['video']}")
                print("  · 해당 시나리오가 실패했을 수 있습니다 (아래 실행 상태 참조)")
                print("  · render=false 로 실행되었을 수 있습니다")
                if len(status):
                    display(status[status["token"] == token])
            else:
                rel = f"{VIDEO_STAGE}/{tag}/{path.name}"
                display(HTML(video_html(rel, width=width)))
            display(row.to_frame(name=token))

    select.observe(lambda ch: show(ch["new"]) if ch["name"] == "value" else None,
                   names="value")
    show(options[0][1])          # 처음부터 비어 있지 않도록 한 번 그린다
    return widgets.HBox([select, out])


def video_side_by_side(pairs: List[tuple], width: int = 470):
    """여러 런의 같은 시나리오 영상을 가로로 나란히 띄운다.

    :param pairs: [(제목, 노트북 기준 상대경로), ...]
    """
    from IPython.display import HTML

    cells = "".join(
        f'<div style="display:inline-block;margin:4px;vertical-align:top">'
        f'<div style="font:600 13px sans-serif;padding:2px">{title}</div>'
        f'{video_html(rel, width=width)}</div>'
        for title, rel in pairs
    )
    return HTML(cells)
