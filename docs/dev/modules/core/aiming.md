# Core: Aiming

`Aiming` — компонент Core, который преобразует точки на working frame в логическое геометрическое требование к Turret.

Aiming выполняется в main thread и вызывается Mediator синхронно.

## Границы ответственности

Aiming не:

- выбирает authoritative target;
- выполняет VisionProcessor;
- управляет моторами;
- выполняет PID;
- отправляет UART;
- исправляет distortion/rectification.

Geometric correction уже выполнена в Vision до публикации `FramePacket`.

## Working frame и `CameraModel`

Все входные pixel coordinates относятся к working frame.

`CameraModel` immutable и приходит вместе с `CameraSessionStarted(camera, generation, camera_model)`. Core/Aiming используют модель только для принятой `generation` из `CameraSessionGate`.

Публичный геометрический контракт:

```text
CameraModel.pixel_to_ray(x_px, y_px) → normalized CameraRay
```

Система CameraRay:

```text
+x → вправо
+y → вниз
+z → вперёд от камеры
```

Aiming не зависит напрямую от OpenCV `K`, `D`, `P1`, `P2`.

## Логическая система Turret

На выходе Aiming:

```text
+X → турели нужно вправо
+Y → турели нужно вверх
```

Следовательно, image/camera `+Y вниз` явно преобразуется внутри Aiming в turret `+Y вверх`.

Например вертикальный угол луча может вычисляться со знаком, соответствующим Turret:

```text
vertical_deg = -atan2(ray.y, sqrt(ray.x² + ray.z²))
```

HAL `invert-y` не используется для этого преобразования: он отвечает только за физическое направление двигателя/проводки.

## Aim point

Для Overview и Stereo Left задаётся aim point в пикселях working frame:

```text
aim-x-px
aim-y-px
```

Если не задан — используется center working frame.

В первой реализации предполагается, что camera axes достаточно близки к turret axes. Постоянное boresight-смещение компенсируется aim point. Полный `R_camera_to_turret` откладывается до механических испытаний.

## Поворот по клику

```text
click pixel → click ray
aim point → aim ray
→ angular difference в логической системе Turret
→ delta_x_deg / delta_y_deg
```

Постоянный `degrees_per_pixel` и простая линейная разница pixels не используются.

Click-to-move вызывается только после подтверждённого `TurretState.control_mode == RELATIVE`.

## Сопровождение и lead

Для актуального `selected_target` в подтверждённом `TRACKING`:

```text
lead_point =
    bbox_center +
    velocity_px_s * lead_time_s
```

Lead рассчитывается только при наличии актуального `TrackedObject`.

Далее:

```text
lead pixel → target ray
aim point → aim ray
→ angular error
→ TrackingError
```

Если selected track временно отсутствует, Aiming не продолжает самостоятельно экстраполировать старую цель.

## `TrackingError`

`TrackingError` создаётся только если:

- latest `TurretState.control_mode == TRACKING`;
- существует `selected_target`;
- target относится к `main_camera`;
- generation принята `CameraSessionGate`;
- track присутствует в актуальном `VisionResult`.

На межпоточной границе `TrackingError` хранится в latest-state container с monotonic `revision`. Turret PID обрабатывает каждое revision не более одного раза.

При уходе из TRACKING, final target loss/deselect, camera-generation invalidation, StopMotion или Emergency latest-state явно `clear()/invalidate()`. Core не посылает отдельный PID-reset: Turret Controller сам реагирует на domain/control boundaries.

## Calibration

Calibration хранится отдельно от `config.json`:

```text
calibration/overview.json
calibration/stereo.json
```

Vision загружает calibration и строит working-frame `CameraModel`. При calibration mismatch/error pipeline не публикует raw frame как замену working frame.

## Конфигурация Aiming

Минимально:

```text
lead-time-ms
target-lost-timeout-ms
aim point Overview
aim point Stereo Left
```

`target-lost-timeout-ms` используется Core, но относится к поведению tracking.

Подробно: [Конфигурация](../../architecture/configuration.md).
