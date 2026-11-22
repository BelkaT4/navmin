"""Production-like localhost RTP/JPEG transport for normal tracking SIM cameras."""

from __future__ import annotations

from navmin.config.models import RtpJpegSourceConfig, RtpJpegTimestampMode
from navmin.contracts import CameraRole
from navmin.vision.gstreamer_source import CameraSourceError, GStreamerRtpJpegSource

from .rtp_jpeg_appsrc import (
    AppsrcRtpJpegFrame,
    AppsrcRtpJpegSender,
    AppsrcRtpJpegSenderConfig,
)
from .synthetic_tracking_turret import RealtimeSyntheticTrackingRig
from .synthetic_tracking_world import SYNTHETIC_TRACKING_FPS

SYNTHETIC_TRACKING_RTP_PORTS = {
    CameraRole.OVERVIEW: 18_888,
    CameraRole.STEREO_LEFT: 18_889,
    CameraRole.STEREO_RIGHT: 18_890,
}


def synthetic_tracking_rtp_source_config(camera: CameraRole) -> RtpJpegSourceConfig:
    """Return the fixed localhost receiver endpoint for one normal SIM camera."""
    try:
        port = SYNTHETIC_TRACKING_RTP_PORTS[camera]
    except KeyError as exc:
        raise ValueError(f"unsupported synthetic tracking camera: {camera}") from exc
    return RtpJpegSourceConfig(
        bind_address="127.0.0.1",
        port=port,
        buffer_size=1,
        # Match the current production hardware profile: NavMin timestamps frames
        # on receive while RTP still crosses the real production decoder boundary.
        timestamp_mode=RtpJpegTimestampMode.RECEIVE_TIME,
    )


class SyntheticTrackingRtpJpegSource(GStreamerRtpJpegSource):
    """Normal operator SIM source using synthetic JPEG/RTP and production decode."""

    def __init__(
        self,
        rig: RealtimeSyntheticTrackingRig,
        camera: CameraRole,
        config: RtpJpegSourceConfig,
    ) -> None:
        if not isinstance(rig, RealtimeSyntheticTrackingRig):
            raise TypeError("rig must be RealtimeSyntheticTrackingRig")
        expected = synthetic_tracking_rtp_source_config(camera)
        if config != expected:
            raise ValueError(
                "normal synthetic camera requires canonical localhost RTP/JPEG config"
            )
        self._rig = rig
        self._camera = camera
        self._sender = AppsrcRtpJpegSender(
            AppsrcRtpJpegSenderConfig(
                host="127.0.0.1",
                port=config.port,
                fps=SYNTHETIC_TRACKING_FPS,
            ),
            self._next_sender_frame,
        )
        super().__init__(config)

    @property
    def sender(self) -> AppsrcRtpJpegSender:
        return self._sender

    @property
    def rig(self) -> RealtimeSyntheticTrackingRig:
        return self._rig

    @property
    def camera(self) -> CameraRole:
        return self._camera

    @property
    def failure(self) -> BaseException | None:
        sender_failure = self._sender.failure
        return sender_failure if sender_failure is not None else super().failure

    def start(self) -> None:
        if self.started:
            return
        # Receiver first: no deterministic first synthetic datagram is lost before
        # the production udpsrc/appsink boundary is ready.
        super().start()
        try:
            self._sender.start()
        except Exception as exc:
            super().stop()
            if isinstance(exc, CameraSourceError):
                raise
            raise CameraSourceError(
                f"cannot start synthetic RTP/JPEG sender: {exc}"
            ) from exc

    def stop(self) -> None:
        sender_error: Exception | None = None
        try:
            self._sender.stop()
        except Exception as exc:  # noqa: BLE001
            # Cleanup boundary: the receiver must still be stopped before the
            # sender failure is re-raised to the caller.
            sender_error = exc
        super().stop()
        if sender_error is not None:
            if isinstance(sender_error, CameraSourceError):
                raise sender_error
            raise CameraSourceError(
                f"cannot stop synthetic RTP/JPEG sender: {sender_error}"
            ) from sender_error

    def _next_sender_frame(self) -> AppsrcRtpJpegFrame:
        frame_set, timestamp_ns = self._rig.capture(self._camera)
        return AppsrcRtpJpegFrame(
            image=frame_set.image(self._camera),
            pts_ns=timestamp_ns,
            capture_id=frame_set.capture_id,
        )


__all__ = [
    "SYNTHETIC_TRACKING_RTP_PORTS",
    "SyntheticTrackingRtpJpegSource",
    "synthetic_tracking_rtp_source_config",
]
