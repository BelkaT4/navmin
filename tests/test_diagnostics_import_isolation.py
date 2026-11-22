from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else os.pathsep.join((src_path, existing))
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_launch_runtime_import_does_not_eager_load_diagnostic_runners_or_vt11() -> None:
    result = _run_python(
        """
import sys
import navmin.launch_runtime  # noqa: F401

forbidden = {
    "navmin.diagnostics.synthetic_tracking_closed_loop",
    "navmin.diagnostics.synthetic_tracking_stereo",
}
loaded_forbidden = sorted(forbidden.intersection(sys.modules))
loaded_vt11 = sorted(
    name for name in sys.modules if name.startswith("navmin.vision.processors.vt11")
)
assert not loaded_forbidden, loaded_forbidden
assert not loaded_vt11, loaded_vt11
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_diagnostics_package_root_reexports_remain_lazy_and_compatible() -> None:
    result = _run_python(
        """
import sys
import navmin.diagnostics as diagnostics

assert "navmin.diagnostics.synthetic_tracking_turret" not in sys.modules
plant_type = diagnostics.SyntheticTurretPlant
assert plant_type.__name__ == "SyntheticTurretPlant"
assert "navmin.diagnostics.synthetic_tracking_turret" in sys.modules
assert "navmin.diagnostics.synthetic_tracking_closed_loop" not in sys.modules
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_all_historical_root_reexports_still_resolve() -> None:
    result = _run_python(
        """
import navmin.diagnostics as diagnostics

for name in diagnostics.__all__:
    getattr(diagnostics, name)
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_diagnostics_submodule_from_import_still_works() -> None:
    result = _run_python(
        """
from navmin.diagnostics import tracking_time_integration

assert tracking_time_integration.__name__ == "navmin.diagnostics.tracking_time_integration"
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout
