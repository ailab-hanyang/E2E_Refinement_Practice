"""practice0 노트북용 헬퍼. 셸 명령을 실행하면서 출력을 그대로 흘려보낸다.

노트북 셀에서 `!cmd` 를 쓰면 명령이 끝날 때까지 출력이 안 보인다. 환경 구축은
한 단계가 수 분씩 걸리므로 진행 상황이 실시간으로 보여야 한다. 그래서 subprocess 를
line-buffered 로 읽어 즉시 출력한다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _find_repo_root() -> Path:
    """run_simulation.py 가 있는 상위 디렉토리를 저장소 루트로 본다.

    깊이로 세면(parents[N]) 이 파일을 다른 폴더로 옮길 때 조용히 틀린 곳을 가리킨다.
    """
    for d in Path(__file__).resolve().parents:
        if (d / "run_simulation.py").exists():
            return d
    raise RuntimeError(f"저장소 루트를 찾지 못했습니다: {Path(__file__).resolve()}")


REPO_ROOT = _find_repo_root()


def run(cmd: str, cwd: Path | str | None = None, env: dict | None = None,
        check: bool = False, quiet_ok: bool = False) -> int:
    """bash 로 cmd 를 실행하고 stdout/stderr 를 실시간으로 흘려보낸다. 반환값은 exit code."""
    merged = os.environ.copy()
    if env:
        merged.update(env)

    t0 = time.time()
    print(f"$ {cmd}", flush=True)
    print("-" * 78, flush=True)
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(cwd or REPO_ROOT),
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:            # type: ignore[union-attr]
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()

    print("-" * 78, flush=True)
    mark = "✅" if rc == 0 else "❌"
    print(f"{mark} exit={rc}   ({time.time() - t0:.1f}s)", flush=True)
    if check and rc != 0:
        raise RuntimeError(f"명령 실패 (exit={rc}): {cmd}")
    if rc != 0 and not quiet_ok:
        print("   위 로그의 마지막 에러 줄을 확인하세요.", flush=True)
    return rc


def conda_run(env_name: str, cmd: str, **kw) -> int:
    """지정한 conda 환경을 activate 한 뒤 cmd 를 실행한다."""
    base = conda_base()
    if base is None:
        print("❌ conda 를 찾지 못했습니다. README 「준비」 1 절을 먼저 수행하십시오.")
        return 127
    # PYTHONNOUSERSITE=1 — ~/.local/lib/pythonX.Y/site-packages 를 sys.path 에서 뺀다.
    # 과거에 pip install --user 로 깔아 둔 패키지가 있으면 환경의 버전 핀을 덮어쓴다.
    full = (f'source "{base}/etc/profile.d/conda.sh" && conda activate {env_name} && '
            f'export PYTHONNOUSERSITE=1 && {cmd}')
    return run(full, **kw)


def conda_python(env_name: str, code: str, setup_env: bool = False, **kw) -> int:
    """파이썬 코드를 지정 conda 환경에서 실행한다.

    `python -c "<코드>"` 로 넘기면 여러 줄 코드가 셸 따옴표를 거치며 깨지므로 파일로 넘긴다.
    파일은 반드시 저장소 루트에 둔다 — hydra 의 config_path 가 **호출한 파일의 위치** 기준이라
    /tmp 에 두면 /tmp/config 를 찾다가 실패한다.
    """
    tmp = REPO_ROOT / f"._practice0_{os.getpid()}.py"
    tmp.write_text(code)
    try:
        prefix = "source script/nuplan_env.sh >/dev/null && " if setup_env else ""
        return conda_run(env_name, f'cd "{REPO_ROOT}" && {prefix}python -u "{tmp}"', **kw)
    finally:
        tmp.unlink(missing_ok=True)


def conda_base() -> str | None:
    """conda 설치 경로. PATH 에 없으면 흔한 위치를 뒤진다."""
    exe = shutil.which("conda")
    if exe:
        out = subprocess.run(["bash", "-lc", "conda info --base"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout.strip()
    for c in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path("/opt/conda")):
        if (c / "etc/profile.d/conda.sh").exists():
            return str(c)
    return None


def env_exists(env_name: str) -> bool:
    base = conda_base()
    if base is None:
        return False
    out = subprocess.run(
        ["bash", "-lc", f'source "{base}/etc/profile.d/conda.sh" && conda env list'],
        capture_output=True, text=True)
    return any(line.split()[0:1] == [env_name]
               for line in out.stdout.splitlines() if line and not line.startswith("#"))


def section(title: str) -> None:
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)
