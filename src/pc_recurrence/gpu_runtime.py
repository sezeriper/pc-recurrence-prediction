from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

GPU_ENV_MARKER = "PC_RECURRENCE_GPU_CONFIGURED"


def same_executable(first: Path, second: Path) -> bool:
    try:
        return first.resolve().samefile(second.resolve())
    except (FileNotFoundError, OSError):
        return first.resolve() == second.resolve()


def _probe_gpu_names(runtime_python: Path) -> list[str]:
    environment = os.environ.copy()
    environment.pop("HIP_VISIBLE_DEVICES", None)
    script = (
        "import json, torch; "
        "print(json.dumps([torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]))"
    )
    completed = subprocess.run(
        [str(runtime_python), "-c", script],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"unable to enumerate ROCm devices: {detail}")
    try:
        names = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("ROCm device probe returned invalid output") from exc
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise RuntimeError("ROCm device probe returned invalid device names")
    return names


def dispatch_to_gpu_runtime(
    runtime_python: Path,
    *,
    module: str,
    expected_gpu_name: str,
) -> None:
    """Relaunch once with the requested physical GPU exposed as logical cuda:0."""
    if os.environ.get(GPU_ENV_MARKER) == "1":
        if not same_executable(Path(sys.executable), runtime_python):
            raise RuntimeError("GPU runtime marker is set under the wrong Python interpreter")
        return
    if not runtime_python.is_file():
        raise ValueError(f"runtime Python does not exist: {runtime_python}")
    names = _probe_gpu_names(runtime_python)
    matches = [index for index, name in enumerate(names) if name == expected_gpu_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {expected_gpu_name}; detected {names or 'no ROCm devices'}"
        )
    environment = os.environ.copy()
    environment["HIP_VISIBLE_DEVICES"] = str(matches[0])
    environment[GPU_ENV_MARKER] = "1"
    source_root = Path(__file__).resolve().parents[1]
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root) + (
        os.pathsep + existing if existing else ""
    )
    completed = subprocess.run(
        [str(runtime_python), "-m", module, *sys.argv[1:]],
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)
