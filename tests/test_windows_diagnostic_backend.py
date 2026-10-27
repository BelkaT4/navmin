"""Regression tests for Windows diagnostics without physical peripherals."""

from navmin.diagnostic_launcher import (
    CameraEndpoint,
    DiagnosticSelection,
    TurretEndpoint,
    _parser,
    _planned_preflight_inputs,
    _synthetic_diagnostic_inputs,
)
from navmin.preflight import StartupBackendSelection


def test_fake_endpoint_cli_accepts_synthetic_inputs():
    args = _parser().parse_args(["--overview", "localhost", "--stereo-left", "localhost", "--turret", "fake", "--synthetic-inputs"])
    assert args.turret == "fake" and args.synthetic_inputs
    StartupBackendSelection(overview="localhost", stereo_left="localhost", turret="fake")


def test_fake_endpoint_uses_existing_software_transport():
    selection = DiagnosticSelection(CameraEndpoint.LOCALHOST, CameraEndpoint.LOCALHOST, TurretEndpoint.FAKE)
    inputs = _synthetic_diagnostic_inputs()
    prepared = _planned_preflight_inputs(selection, inputs)
    assert prepared.config.turret.emulate_stm32 is True
    assert prepared.config.turret.serial.port == inputs.config.turret.serial.port


def test_real_or_pty_preflight_never_switches_to_fake():
    inputs = _synthetic_diagnostic_inputs()
    for backend in (TurretEndpoint.REAL, TurretEndpoint.PTY):
        selection = DiagnosticSelection(CameraEndpoint.LOCALHOST, CameraEndpoint.LOCALHOST, backend)
        prepared = _planned_preflight_inputs(selection, inputs)
        assert prepared.config.turret.emulate_stm32 is False
