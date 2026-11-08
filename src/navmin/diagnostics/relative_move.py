"""Hardware-isolated relative-move diagnostic over the production Turret stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from navmin.config.models import TrackingControlConfig, TurretConfig
from navmin.contracts import (
    MotorState,
    MoveRelativeCommand,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
)
from navmin.turret.hal import RelativeMovePreview, preview_move_relative
from navmin.turret.worker import TurretWorker

AxisName = Literal["x", "y"]
WorkerFactory = Callable[[TurretConfig, TrackingControlConfig], TurretWorker]
Prompt = Callable[[str], str]
Output = Callable[[str], None]


class RelativeMoveDiagnosticError(RuntimeError):
    """The diagnostic cannot safely proceed or confirm a requested transition."""


@dataclass(frozen=True, slots=True)
class RelativeMoveDiagnosticPlan:
    axis: AxisName
    requested_degrees: float
    command: MoveRelativeCommand
    preview: RelativeMovePreview
    full_steps_per_revolution: int
    microstep_divider: int
    invert: bool
    max_relative_move_deg: float

    @property
    def selected_axis_steps(self) -> int:
        if self.axis == "x":
            return self.preview.payload.delta_x_steps
        return self.preview.payload.delta_y_steps

    @property
    def selected_axis_effective_steps_per_revolution(self) -> int:
        if self.axis == "x":
            return self.preview.effective_steps_per_revolution_x
        return self.preview.effective_steps_per_revolution_y

    @property
    def selected_axis_quantized_degrees(self) -> float:
        if self.axis == "x":
            return self.preview.quantized_delta_x_deg
        return self.preview.quantized_delta_y_deg


def build_relative_move_diagnostic_plan(
    config: TurretConfig,
    *,
    axis: AxisName,
    degrees: float,
) -> RelativeMoveDiagnosticPlan:
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    if isinstance(degrees, bool) or not isinstance(degrees, int | float):
        raise TypeError("degrees must be a number")
    degrees = float(degrees)
    if degrees == 0.0:
        raise ValueError("degrees must be non-zero")

    command = (
        MoveRelativeCommand(degrees, 0.0)
        if axis == "x"
        else MoveRelativeCommand(0.0, degrees)
    )
    preview = preview_move_relative(config, command)
    mechanics = config.axes.x if axis == "x" else config.axes.y
    return RelativeMoveDiagnosticPlan(
        axis=axis,
        requested_degrees=degrees,
        command=command,
        preview=preview,
        full_steps_per_revolution=mechanics.full_steps_per_revolution,
        microstep_divider=mechanics.microstep_divider,
        invert=mechanics.invert,
        max_relative_move_deg=mechanics.max_relative_move_deg,
    )


def format_relative_move_plan(plan: RelativeMoveDiagnosticPlan) -> tuple[str, ...]:
    sign = "+" if plan.requested_degrees > 0 else ""
    quantized_sign = "+" if plan.selected_axis_quantized_degrees > 0 else ""
    return (
        f"Axis: {plan.axis.upper()}",
        f"Requested logical angle: {sign}{plan.requested_degrees:g} deg",
        f"full-steps-per-revolution: {plan.full_steps_per_revolution}",
        f"microstep-divider: {plan.microstep_divider}",
        (
            "effective STEP pulses/revolution: "
            f"{plan.selected_axis_effective_steps_per_revolution}"
        ),
        f"invert: {str(plan.invert).lower()}",
        f"max-relative-move-deg: {plan.max_relative_move_deg:g}",
        f"MOVE_RELATIVE payload STEP pulses: {plan.selected_axis_steps}",
        (
            "Quantized logical angle represented by payload: "
            f"{quantized_sign}{plan.selected_axis_quantized_degrees:.6f} deg"
        ),
        "Protocol contract: one step = one physical STEP pulse sent to the driver.",
    )


def execute_relative_move_diagnostic(
    config: TurretConfig,
    tracking_control_config: TrackingControlConfig,
    plan: RelativeMoveDiagnosticPlan,
    *,
    worker_factory: WorkerFactory | None = None,
    prompt: Prompt = input,
    output: Output = print,
    timeout_s: float = 5.0,
) -> None:
    """Execute exactly one relative move after two explicit operator confirmations.

    Cameras, aiming and motion-calibration correction are deliberately absent.
    The production ``TurretWorker -> Controller -> HAL -> Session`` path remains
    authoritative for the actual request.
    """
    if config.emulate_stm32:
        raise RelativeMoveDiagnosticError(
            "real execution requires turret.emulate-stm32=false"
        )
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    factory = worker_factory or _default_worker_factory
    worker = factory(config, tracking_control_config)
    started = False
    motors_enabled = False
    try:
        worker.start()
        started = True
        state = _wait_for_state(
            worker,
            lambda current: current.connection_state is TurretConnectionState.READY,
            timeout_s=timeout_s,
            failure="Turret did not reach READY",
        )
        _require_safe_initial_state(state)
        output(
            "Turret READY: recovery confirmed MOTOR_OFF and RELATIVE mode. "
            "No move has been sent."
        )

        if prompt("Type MOTOR ON to enable the drivers, or anything else to abort: ").strip() != "MOTOR ON":
            raise RelativeMoveDiagnosticError("operator aborted before MOTOR_ON")
        if not worker.motor_on():
            raise RelativeMoveDiagnosticError("MOTOR_ON ingress was rejected")
        state = _wait_for_state(
            worker,
            lambda current: (
                current.connection_state is TurretConnectionState.READY
                and current.motor_state is MotorState.ON
            ),
            timeout_s=timeout_s,
            failure="MOTOR_ON was not confirmed",
        )
        if state.control_mode is not TurretControlMode.RELATIVE:
            raise RelativeMoveDiagnosticError(
                "control mode changed away from RELATIVE before the move"
            )
        motors_enabled = True

        if prompt("Type MOVE to send this one MOVE_RELATIVE request: ").strip() != "MOVE":
            raise RelativeMoveDiagnosticError("operator aborted before MOVE_RELATIVE")

        before_revision = worker.state_updates.snapshot().revision
        _require_ready_for_move(worker.current_state)
        worker.submit_move_relative(plan.command)
        _wait_for_revision(
            worker,
            after_revision=before_revision,
            timeout_s=timeout_s,
            failure="Turret worker did not process the MOVE_RELATIVE ingress",
        )
        output(
            "MOVE_RELATIVE was processed by the production Turret worker. "
            "The protocol has no MOVE_COMPLETED event; wait for physical motion "
            "to stop and measure the angle independently from the camera image."
        )
        prompt("Press Enter after measurement to MOTOR_OFF and exit: ")
    except KeyboardInterrupt:
        if started:
            output("Keyboard interrupt: requesting EMERGENCY_STOP.")
            worker.request_emergency()
            _wait_for_revision_best_effort(worker, timeout_s=timeout_s)
        raise RelativeMoveDiagnosticError("diagnostic interrupted by operator") from None
    finally:
        if started:
            if motors_enabled or worker.current_state.motor_state is MotorState.ON:
                motor_off_accepted = worker.motor_off()
                if motor_off_accepted:
                    try:
                        _wait_for_state(
                            worker,
                            lambda current: (
                                current.connection_state is TurretConnectionState.READY
                                and current.motor_state is MotorState.OFF
                            ),
                            timeout_s=timeout_s,
                            failure="MOTOR_OFF was not confirmed during cleanup",
                        )
                    except RelativeMoveDiagnosticError:
                        worker.request_emergency()
                        _wait_for_revision_best_effort(worker, timeout_s=timeout_s)
                else:
                    worker.request_emergency()
                    _wait_for_revision_best_effort(worker, timeout_s=timeout_s)
            worker.shutdown(timeout_s)


def _default_worker_factory(
    config: TurretConfig,
    tracking_control_config: TrackingControlConfig,
) -> TurretWorker:
    return TurretWorker(config, tracking_control_config)


def _require_safe_initial_state(state: TurretState) -> None:
    if state.connection_state is not TurretConnectionState.READY:
        raise RelativeMoveDiagnosticError("Turret is not READY")
    if state.motor_state is not MotorState.OFF:
        raise RelativeMoveDiagnosticError(
            f"diagnostic startup requires MOTOR_OFF, got {state.motor_state.value}"
        )
    if state.control_mode is not TurretControlMode.RELATIVE:
        raise RelativeMoveDiagnosticError(
            "diagnostic startup requires RELATIVE control mode"
        )


def _require_ready_for_move(state: TurretState) -> None:
    if state.connection_state is not TurretConnectionState.READY:
        raise RelativeMoveDiagnosticError("connection is not READY before move")
    if state.motor_state is not MotorState.ON:
        raise RelativeMoveDiagnosticError("motors are not confirmed ON before move")
    if state.control_mode is not TurretControlMode.RELATIVE:
        raise RelativeMoveDiagnosticError("control mode is not RELATIVE before move")


def _wait_for_state(
    worker: TurretWorker,
    predicate: Callable[[TurretState], bool],
    *,
    timeout_s: float,
    failure: str,
) -> TurretState:
    snapshot = worker.state_updates.snapshot()
    deadline = monotonic() + timeout_s
    while True:
        state = snapshot.value
        if state is not None and predicate(state):
            return state
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RelativeMoveDiagnosticError(f"{failure}; last state={state}")
        changed = worker.state_updates.wait_for_revision(snapshot.revision, remaining)
        if changed is None:
            raise RelativeMoveDiagnosticError(f"{failure}; last state={state}")
        snapshot = changed


def _wait_for_revision(
    worker: TurretWorker,
    *,
    after_revision: int,
    timeout_s: float,
    failure: str,
) -> TurretState:
    deadline = monotonic() + timeout_s
    snapshot = worker.state_updates.snapshot()
    while snapshot.revision <= after_revision:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RelativeMoveDiagnosticError(failure)
        changed = worker.state_updates.wait_for_revision(snapshot.revision, remaining)
        if changed is None:
            raise RelativeMoveDiagnosticError(failure)
        snapshot = changed
    state = snapshot.value
    if state is None:
        raise RelativeMoveDiagnosticError(failure)
    return state


def _wait_for_revision_best_effort(worker: TurretWorker, *, timeout_s: float) -> None:
    snapshot = worker.state_updates.snapshot()
    worker.state_updates.wait_for_revision(snapshot.revision, min(timeout_s, 1.0))


__all__ = [
    "RelativeMoveDiagnosticError",
    "RelativeMoveDiagnosticPlan",
    "build_relative_move_diagnostic_plan",
    "execute_relative_move_diagnostic",
    "format_relative_move_plan",
]
