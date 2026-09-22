"""Main-thread camera session acceptance gate shared by Core and later UI."""

from __future__ import annotations

from navmin.contracts import CameraModel, CameraRole, CameraSessionStarted


class CameraSessionGate:
    """Bind each accepted camera generation to its immutable CameraModel."""

    def __init__(self) -> None:
        self._accepted_generations: dict[CameraRole, int] = {}
        self._camera_models: dict[CameraRole, CameraModel] = {}

    def accept(self, session: CameraSessionStarted) -> bool:
        """Accept only a strictly newer barrier for the given camera."""
        if not isinstance(session, CameraSessionStarted):
            raise TypeError("session must be CameraSessionStarted")
        current = self._accepted_generations.get(session.camera)
        if current is not None and session.generation <= current:
            return False
        self._accepted_generations[session.camera] = session.generation
        self._camera_models[session.camera] = session.camera_model
        return True

    def accepted_generation(self, camera: CameraRole) -> int | None:
        return self._accepted_generations.get(camera)

    def accepts(self, camera: CameraRole, generation: int) -> bool:
        return self._accepted_generations.get(camera) == generation

    def camera_model(
        self,
        camera: CameraRole,
        generation: int,
    ) -> CameraModel | None:
        if not self.accepts(camera, generation):
            return None
        return self._camera_models.get(camera)


__all__ = ["CameraSessionGate"]
