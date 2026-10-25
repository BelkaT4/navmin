"""Motion demo checks use a deterministic clock, no cameras or serial ports."""

import pytest

from navmin.contracts import CameraRole
from navmin.diagnostic_launcher import _synthetic_diagnostic_inputs
from navmin.diagnostics.motion_demo import DemoPlatform, DemoScene
from navmin.turret.protocol import (
    CommandCode,
    MoveRelativePayload,
    ProtocolRequest,
    SetVelocityPayload,
    encode_request,
)


def test_motion_pose_obeys_retry_motor_stop_and_velocity_watchdog():
    now = [0.0]
    platform = DemoPlatform(
        _synthetic_diagnostic_inputs().config.turret.axes, clock=lambda: now[0]
    )

    def send(index, command, payload=None):
        platform.handle_request(
            encode_request(ProtocolRequest(index, command, payload))
        )

    send(0, CommandCode.MOTOR_ON)
    send(1, CommandCode.MOVE_RELATIVE, MoveRelativePayload(800, 0))
    now[0] = 0.2
    assert platform.pose()[0] == pytest.approx(4)
    send(1, CommandCode.MOVE_RELATIVE, MoveRelativePayload(800, 0))
    now[0] = 1
    assert platform.pose()[0] == pytest.approx(9)
    send(2, CommandCode.SET_VELOCITY, SetVelocityPayload(0, 800))
    now[0] = 2
    assert platform.pose()[1] == pytest.approx(1.8)
    send(3, CommandCode.MOVE_RELATIVE, MoveRelativePayload(800, 0))
    send(4, CommandCode.SET_VELOCITY, SetVelocityPayload(0, 0))
    now[0] = 3
    assert platform.pose()[0] == pytest.approx(9)
    send(5, CommandCode.MOTOR_OFF)
    send(6, CommandCode.MOVE_RELATIVE, MoveRelativePayload(800, 0))
    now[0] = 4
    assert platform.pose()[0] == pytest.approx(9)
    send(7, CommandCode.MOTOR_ON)
    send(8, CommandCode.SET_VELOCITY, SetVelocityPayload(800, 800))
    send(9, CommandCode.EMERGENCY_STOP)
    now[0] = 5
    assert platform.pose() == pytest.approx((9, 1.8))


@pytest.mark.parametrize("camera", [CameraRole.OVERVIEW, CameraRole.STEREO_LEFT])
def test_scene_moves_against_camera_rotation_and_renders(camera):
    inputs = _synthetic_diagnostic_inputs()
    calibration = (
        inputs.overview_calibration
        if camera is CameraRole.OVERVIEW
        else inputs.stereo_calibration
    )
    scene = DemoScene(DemoPlatform(inputs.config.turret.axes), camera, calibration)
    before = scene.project(4, 0, (0, 0))
    after = scene.project(4, 0, (4, 0))
    assert after[0] < before[0]
    assert after[0] == 160
    assert scene.project(0, 0, (0, 4))[1] > 120
    image = scene.next_frame()
    assert image.shape == (240, 320, 3)
    assert image.max() == 245
