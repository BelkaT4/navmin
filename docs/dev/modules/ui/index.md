# UI

`UI` написан на PyQt6 и выполняется в главном потоке приложения вместе с Core и `CameraSessionGate`.

![Диаграмма модуля UI](../../diagrams/ui-diagram.png)

## Ответственность

UI:

- показывает Overview и Stereo Left как main + preview;
- рисует overlays поверх working frame;
- показывает camera/Turret/distance state и diagnostics;
- принимает target selection только на main image;
- принимает click-to-move только в RELATIVE;
- предоставляет отдельный control `RELATIVE / TRACKING`;
- предоставляет settings и motor/stop controls;
- передаёт actions в Core.

UI не выполняет VisionProcessor, calibration math, Aiming, PID или UART.

## `main_camera` и preview

Одна из двух камер является main:

```text
OVERVIEW
STEREO_LEFT
```

Вторая автоматически является preview.

Startup `main_camera` берётся из `ui.default-camera`. `STEREO_RIGHT` обычной main camera не является.

Пользователь может явно swap main/preview. Swap проходит через Core, потому что он влияет на selection и processing scope.

При отказе main camera UI **не переключается автоматически** на preview. Текущий `main_camera` сохраняется, UI показывает stale/error/«Нет видеосигнала», а swap остаётся явным действием пользователя.

## Данные и `CameraSessionGate`

UI получает логически:

- `VisionResult`;
- `main_camera`;
- `selected_target`;
- `DistanceResult`;
- `TurretState`;
- camera state/freshness;
- UiConfig.

VisionResult может приходить прямо из Vision, но UI принимает его только если:

```text
result.frame.generation == CameraSessionGate.accepted_generation[result.frame.camera]
```

Новая generation не может отображаться раньше обработки `CameraSessionStarted(camera, generation, camera_model)`.

Эта проверка применяется ко **всем** camera-derived путям UI, включая main/preview, diagnostics и clean recording. Diagnostics/recording не имеют bypass мимо `CameraSessionGate`.

## Кадр и overlays

UI отображает:

```text
VisionResult.frame
+
VisionResult.tracked_objects
```

`FramePacket.image` уже corrected working frame:

- Overview undistorted;
- Stereo Left rectified.

Все UI clicks находятся в coordinates того же working frame.

Overlays не изменяют сам `FramePacket.image`.

## Запись видео

Основная camera recording сохраняет чистый working frame без UI overlays.

Не записываются:

- bbox;
- reticle;
- aim point;
- lead;
- status text;
- FPS.

Это нужно для повторного запуска detector/tracker на записи.

## Vision processing main / preview

UI setting:

```text
main-only
main-and-preview
```

При `main-only` bbox формируются только для main camera; preview остаётся live corrected image.

При `main-and-preview` bbox могут рисоваться на обеих картинках.

Target selection на preview запрещён независимо от processing scope. Selection вообще доступен только после подтверждённого `TurretState.control_mode == TRACKING`.

## Выбор цели

UI передаёт Core identity объекта из отображаемого `VisionResult`:

```text
camera + generation + track_id
```

Core остаётся authoritative source и принимает selection только если:

- latest `TurretState.control_mode == TRACKING`;
- camera == current `main_camera`;
- generation принята `CameraSessionGate`;
- track_id всё ещё существует в latest `VisionResult` этой camera/generation.

Если объект уже исчез, новый selection не применяется.

При swap current selection сбрасывается только если она существует, то есть в TRACKING. В RELATIVE уже сформированный manual `MOVE_RELATIVE` swap не отменяет.

## RELATIVE / TRACKING

UI имеет отдельный control для режима:

```text
RELATIVE
TRACKING
```

Mode меняется только по явному действию пользователя. UI считает изменение подтверждённым после получения `TurretState.control_mode`.

Допустимые состояния:

```text
RELATIVE + no target
TRACKING + no target
TRACKING + selected target
```

`RELATIVE + selected target` недопустимо. При `TRACKING → RELATIVE` selection очищается. Потеря/deselect цели или swap не переключают applied mode автоматически.

## Click-to-move

Click-to-move разрешён только после подтверждённого `TurretState.control_mode == RELATIVE`:

```text
UI click on main working frame
→ Core/Mediator
→ Aiming CameraModel
→ relative angle
→ MoveRelativeCommand
```

В `TRACKING` click-to-move disabled/ignored в первой реализации.

## Stop / Emergency / Motor controls

UI может инициировать:

- switch `RELATIVE / TRACKING`;
- `StopMotion`;
- `EMERGENCY_STOP`;
- `MOTOR_ON`;
- `MOTOR_OFF`.

`StopMotion` — штатная остановка с acceleration limit.

В TRACKING StopMotion снимает selected target и оставляет mode TRACKING. В RELATIVE StopMotion отменяет неотправленный manual intent и через Turret приводит firmware к zero velocity. Уже физически отправленный ordinary request считается committed и заканчивает normal retry cycle; для немедленного прерывания используется Emergency.

Emergency — отдельный priority action без acceleration limit. Persistent emergency latch / Resume state в первой реализации нет.

## Настройки Vision

UI может редактировать:

- source/network settings;
- `processing-scope`;
- optional per-camera `processing-enabled` master switch;
- `vision-processor-class`;
- processor-specific settings после определения schema.

Calibration data не является обычной UI-настройкой `config.json`; она хранится отдельными calibration files.

## Настройки Aiming

Минимально:

- `lead-time-ms`;
- aim point Overview;
- aim point Stereo Left;
- `target-lost-timeout-ms`, если открыт пользователю.

Aim point — pixels working frame.

## Настройки Turret

UI может редактировать:

- PID `Kp/Ki/Kd`;
- max speed;
- acceleration;
- velocity watchdog;
- axis inversion (restart-only);
- full steps per revolution (restart-only);
- microstep divider (restart-only);
- max relative move per axis;
- desired serial baudrate/port;
- simulation mode.

Все изменения проходят через Core/Config Manager.

## Потеря видеосигнала

UI хранит camera connection state отдельно от freshness.

```text
ONLINE + fresh → обычный кадр
ONLINE + stale → freeze last frame + «Нет видеосигнала»
RECONNECTING / ERROR → соответствующий status
```

Freshness вычисляется в main thread по `last_receive_timestamp_ns`, а не только по bool от producer.

## Производительность

Latest-only VisionResult не должен превращаться в backlog Qt notifications. Concrete coalescing определяется при реализации.

`CameraSessionStarted` — barrier event и не может быть потерян/coalesced как обычный latest notification.

## Границы

UI не:

- хранит authoritative `selected_target` или applied `control_mode`;
- автоматически меняет `main_camera` при отказе;
- выполняет stereo distance;
- вычисляет lead/ray/angles;
- выполняет PID;
- отправляет serial packets.

## Что ещё не определено

- точные mouse/key gestures selection/deselect;
- overlap/empty-click UX;
- состав object overlay;
- Stereo Right diagnostic layout;
- Qt notification coalescing primitives;
- future latest-live-frame mode.

Полный список: [Открытые вопросы](../../architecture/problems.md).
