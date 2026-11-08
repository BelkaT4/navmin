"""Preview or execute one camera-independent production MOVE_RELATIVE request."""

from __future__ import annotations

import argparse
from pathlib import Path

from navmin.config import load_config
from navmin.diagnostics.relative_move import (
    RelativeMoveDiagnosticError,
    build_relative_move_diagnostic_plan,
    execute_relative_move_diagnostic,
    format_relative_move_plan,
)
from navmin.turret.hal import HalError
from navmin.turret.protocol import ProtocolValueError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or execute one raw NavMin relative turret move without cameras, "
            "aiming, tracking, or motion-calibration correction."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/navmin.json"),
        help="root NavMin config (default: config/navmin.json)",
    )
    parser.add_argument("--axis", choices=("x", "y"), required=True)
    parser.add_argument("--degrees", type=float, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open the configured real STM32 transport and allow one confirmed move",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="timeout for READY/MOTOR state transitions (default: 5)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        plan = build_relative_move_diagnostic_plan(
            config.turret,
            axis=args.axis,
            degrees=args.degrees,
        )
    except (OSError, ValueError, TypeError, HalError, ProtocolValueError) as exc:
        print(f"PREVIEW FAIL: {exc}")
        return 2

    print("NAVMIN RELATIVE MOVE DIAGNOSTIC")
    print("No cameras, aiming, tracking, or motion-calibration correction are used.")
    for line in format_relative_move_plan(plan):
        print(f"  {line}")

    if not args.execute:
        print("PREVIEW ONLY: no serial device was opened and no motion command was sent.")
        return 0

    print()
    print("WARNING: --execute controls real production equipment.")
    print("Confirm a clear mechanical path and an independent way to measure angle.")
    try:
        execute_relative_move_diagnostic(
            config.turret,
            config.tracking_control,
            plan,
            timeout_s=args.timeout_seconds,
        )
    except (RelativeMoveDiagnosticError, OSError, RuntimeError) as exc:
        print(f"DIAGNOSTIC FAIL: {exc}")
        return 2

    print("DIAGNOSTIC COMPLETE: cleanup requested MOTOR_OFF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
