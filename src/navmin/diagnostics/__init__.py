"""Reusable hardware-independent diagnostic boundaries.

The package root intentionally keeps its historical re-export API lazy.  Normal
application imports should not pull diagnostic runners (or their heavy optional
dependencies) into ``sys.modules`` merely because a diagnostics submodule is
referenced elsewhere.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # localhost_rtp
    "GST_RECEIVER_ELEMENTS": (".localhost_rtp", "GST_RECEIVER_ELEMENTS"),
    "GST_SENDER_ELEMENTS": (".localhost_rtp", "GST_SENDER_ELEMENTS"),
    "LocalhostRtpJpegSender": (".localhost_rtp", "LocalhostRtpJpegSender"),
    "PreflightCheck": (".localhost_rtp", "PreflightCheck"),
    "PreflightResult": (".localhost_rtp", "PreflightResult"),
    "RtpJpegSenderConfig": (".localhost_rtp", "RtpJpegSenderConfig"),
    "SenderProcessError": (".localhost_rtp", "SenderProcessError"),
    "SyntheticDiagnosticFrameGenerator": (
        ".localhost_rtp",
        "SyntheticDiagnosticFrameGenerator",
    ),
    "build_sender_command": (".localhost_rtp", "build_sender_command"),
    "check_gstreamer_runtime": (".localhost_rtp", "check_gstreamer_runtime"),
    "check_gstreamer_sender_runtime": (
        ".localhost_rtp",
        "check_gstreamer_sender_runtime",
    ),
    "diagnostic_camera_config": (".localhost_rtp", "diagnostic_camera_config"),
    "diagnostic_overview_calibration": (
        ".localhost_rtp",
        "diagnostic_overview_calibration",
    ),
    "diagnostic_stereo_calibration": (
        ".localhost_rtp",
        "diagnostic_stereo_calibration",
    ),
    # pty_stm32
    "EndpointSnapshot": (".pty_stm32", "EndpointSnapshot"),
    "PtyDiagnosticError": (".pty_stm32", "PtyDiagnosticError"),
    "PtyEmulatorStats": (".pty_stm32", "PtyEmulatorStats"),
    "PtyFault": (".pty_stm32", "PtyFault"),
    "PtyFaultKind": (".pty_stm32", "PtyFaultKind"),
    "PtyFaultQueue": (".pty_stm32", "PtyFaultQueue"),
    "PtyPreflightResult": (".pty_stm32", "PtyPreflightResult"),
    "PtyReplacement": (".pty_stm32", "PtyReplacement"),
    "PtyRequestRecord": (".pty_stm32", "PtyRequestRecord"),
    "PtyStm32Emulator": (".pty_stm32", "PtyStm32Emulator"),
    "RequestFrameAssembler": (".pty_stm32", "RequestFrameAssembler"),
    "StableDiagnosticPort": (".pty_stm32", "StableDiagnosticPort"),
    "check_pty_preflight": (".pty_stm32", "check_pty_preflight"),
    # synthetic_tracking_closed_loop
    "SyntheticAngularCameraModel": (
        ".synthetic_tracking_closed_loop",
        "SyntheticAngularCameraModel",
    ),
    "SyntheticClosedLoopFrame": (
        ".synthetic_tracking_closed_loop",
        "SyntheticClosedLoopFrame",
    ),
    "SyntheticClosedLoopPhaseSummary": (
        ".synthetic_tracking_closed_loop",
        "SyntheticClosedLoopPhaseSummary",
    ),
    "SyntheticClosedLoopResult": (
        ".synthetic_tracking_closed_loop",
        "SyntheticClosedLoopResult",
    ),
    "run_synthetic_tracking_closed_loop": (
        ".synthetic_tracking_closed_loop",
        "run_synthetic_tracking_closed_loop",
    ),
    "write_synthetic_tracking_closed_loop_json": (
        ".synthetic_tracking_closed_loop",
        "write_synthetic_tracking_closed_loop_json",
    ),
    # synthetic_tracking_stereo
    "SyntheticTrackingStereoDistanceSample": (
        ".synthetic_tracking_stereo",
        "SyntheticTrackingStereoDistanceSample",
    ),
    "SyntheticTrackingStereoResult": (
        ".synthetic_tracking_stereo",
        "SyntheticTrackingStereoResult",
    ),
    "run_synthetic_tracking_stereo_diagnostic": (
        ".synthetic_tracking_stereo",
        "run_synthetic_tracking_stereo_diagnostic",
    ),
    "write_synthetic_tracking_stereo_json": (
        ".synthetic_tracking_stereo",
        "write_synthetic_tracking_stereo_json",
    ),
    # synthetic_tracking_turret
    "RealtimeSyntheticTrackingRig": (
        ".synthetic_tracking_turret",
        "RealtimeSyntheticTrackingRig",
    ),
    "SyntheticTrackingFrameSource": (
        ".synthetic_tracking_turret",
        "SyntheticTrackingFrameSource",
    ),
    "SyntheticTrackingSimulation": (
        ".synthetic_tracking_turret",
        "SyntheticTrackingSimulation",
    ),
    "SyntheticTrackingSimulationSnapshot": (
        ".synthetic_tracking_turret",
        "SyntheticTrackingSimulationSnapshot",
    ),
    "SyntheticTrackingStm32Endpoint": (
        ".synthetic_tracking_turret",
        "SyntheticTrackingStm32Endpoint",
    ),
    "SyntheticTurretPlant": (".synthetic_tracking_turret", "SyntheticTurretPlant"),
    "SyntheticTurretPlantState": (
        ".synthetic_tracking_turret",
        "SyntheticTurretPlantState",
    ),
}

__all__ = (
    'GST_RECEIVER_ELEMENTS',
    'GST_SENDER_ELEMENTS',
    'EndpointSnapshot',
    'LocalhostRtpJpegSender',
    'PreflightCheck',
    'PreflightResult',
    'PtyDiagnosticError',
    'PtyEmulatorStats',
    'PtyFault',
    'PtyFaultKind',
    'PtyFaultQueue',
    'PtyPreflightResult',
    'PtyReplacement',
    'PtyRequestRecord',
    'PtyStm32Emulator',
    'RealtimeSyntheticTrackingRig',
    'RequestFrameAssembler',
    'RtpJpegSenderConfig',
    'SenderProcessError',
    'StableDiagnosticPort',
    'SyntheticAngularCameraModel',
    'SyntheticClosedLoopFrame',
    'SyntheticClosedLoopPhaseSummary',
    'SyntheticClosedLoopResult',
    'SyntheticDiagnosticFrameGenerator',
    'SyntheticTrackingFrameSource',
    'SyntheticTrackingSimulation',
    'SyntheticTrackingSimulationSnapshot',
    'SyntheticTrackingStereoDistanceSample',
    'SyntheticTrackingStereoResult',
    'SyntheticTrackingStm32Endpoint',
    'SyntheticTurretPlant',
    'SyntheticTurretPlantState',
    'build_sender_command',
    'check_gstreamer_runtime',
    'check_gstreamer_sender_runtime',
    'check_pty_preflight',
    'diagnostic_camera_config',
    'diagnostic_overview_calibration',
    'diagnostic_stereo_calibration',
    'run_synthetic_tracking_closed_loop',
    'run_synthetic_tracking_stereo_diagnostic',
    'write_synthetic_tracking_closed_loop_json',
    'write_synthetic_tracking_stereo_json',
)


def __getattr__(name: str) -> Any:
    """Resolve historical package-root re-exports without eager imports."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy re-exports to interactive tooling and introspection."""

    return sorted(set(globals()) | set(__all__))
