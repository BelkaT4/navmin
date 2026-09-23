# UI

`UI` написан на PyQt6 и выполняется в главном потоке приложения вместе с Core и `CameraSessionGate`.

![Диаграмма модуля UI](../../diagrams/ui-diagram.png)

## Ответственность

UI:

- запускает основной интерфейс fullscreen;
- показывает Overview и Stereo Left как main + preview;
- рисует overlays поверх working frame;
- показывает camera/Turret/distance state и diagnostics;
- принимает target selection только на main image;
- принимает click-to-move только в RELATIVE;
- предоставляет отдельный control `RELATIVE / TRACKING`;
- предоставляет motor/Emergency controls и settings/recording menu actions;
- предоставляет modeless Operator Window с runtime status и редкими административными действиями;
- передаёт actions в Core.

UI не выполняет VisionProcessor, calibration math, Aiming, PID или UART.

Отдельный startup input-recovery dialog находится в `navmin.input_recovery_dialog`, но не является частью `MainWindow`/operational UI. Его вызывает только normal launcher **до** preflight/workers после strict input error. Он показывает точную причину и предлагает закрыть программу либо явно восстановить local config/calibration set; после recovery текущий startup завершается. `--preflight-only` и diagnostic/headless paths этот dialog не используют.

## `main_camera` и preview

Одна из двух камер является main:

```text
OVERVIEW
STEREO_LEFT
```

Вторая автоматически является preview.

Startup `main_camera` берётся из `ui.default-camera`. `STEREO_RIGHT` обычной main camera не является.

Preview отображается небольшим отдельным окном в правом нижнем углу области main video, выше нижней operational bar. Оно не участвует в target selection/click-to-move.

Пользователь может явно swap main/preview click'ом по preview либо кликабельным swap-icon внутри preview. Swap проходит через Core, потому что он влияет на selection и processing scope. Название камеры отображается прямо в соответствующем view; swap-icon означает действие, а не отдельный state indicator.

При отказе main camera UI **не переключается автоматически** на preview. Текущий `main_camera` сохраняется, UI показывает реальный camera status и presentation-признак `НЕТ НОВЫХ КАДРОВ`, а swap остаётся явным действием пользователя.

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

Baseline object overlay минимален: bbox всех текущих `TrackedObject` и явное выделение selected target. Постоянные `track_id`, distance, velocity и `age_frames` рядом с каждым bbox не показываются; расширенный diagnostic overlay может быть добавлен отдельно без изменения общего `TrackedObject` contract.

## Основной layout и menu bar

Основное окно стартует fullscreen. `F11` неограниченно переключает fullscreen/windowed state. Main video занимает основную область, preview находится в её правом нижнем углу, а компактная нижняя operational bar остаётся отдельной постоянной полосой.

Diagnostic launcher при наличии localhost camera явно включает в operational bar
`NOW HH:MM:SS.mmm`; normal production launcher этот debug clock не показывает.
Clock не заменяет mode/motor/connection controls, а Emergency остаётся самым
крупным и постоянно доступным элементом bar. В VIRTUAL это локальный companion
для sender-side `SOURCE` pixels, не real-camera latency telemetry.

`Esc` открывает или скрывает единственный `Operator Window`; fullscreen state при этом не меняется. Это отдельное modeless movable tool-window, принадлежащее `MainWindow`: оно остаётся поверх основного окна NavMin, но не использует global always-on-top и уходит назад вместе с приложением при переключении на другое приложение. Закрытие крестиком скрывает только Operator Window и не останавливает main window или workers.

Operator Window показывает только уже существующее runtime state:

- фактический `CameraStatus.state` для Overview и Stereo Left без смешивания с presentation freshness;
- `TurretState.connection_state`;
- applied и, при наличии, pending control mode по той же semantics, что operational bar;
- подтверждённый `TurretState.motor_state`.

Кнопка `Полноэкранный режим: ВКЛ/ВЫКЛ` выполняет тот же toggle, что `F11`, и синхронно отражает состояние main window. Кнопка `Выход` после стандартного confirmation `Выйти из NavMin?` закрывает MainWindow штатным UI lifecycle path. Operator Window не дублирует motor control или Emergency и не является полным Settings UI.

Верхний menu bar содержит редко используемые действия, в том числе recording, view, diagnostics и settings. Motor control в menu bar не дублируется, потому что motor state/control постоянно доступен в нижней bar.

Stereo Right не входит в normal main/preview pair. Он открывается через Diagnostics как отдельный diagnostic view и не меняет `main_camera`, selection или обычный swap contract.

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

Start/stop записи доступен из верхнего menu bar. Пока запись активна, в левом верхнем углу main view показывается индикатор фиксированной геометрии: мигающий красный круг и немигающая белая надпись `Запись`. При выключенной записи индикатор полностью скрыт.

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

Baseline gestures:

- в подтверждённом TRACKING левый click внутри bbox запрашивает selection этого объекта;
- левый click по пустому месту выполняет explicit deselect;
- если точку click содержат несколько bbox, выбирается bbox, центр которого ближе к click;
- bbox, не содержащие click, не участвуют в выборе; отдельный nearest-object helper в v1 не используется;
- preview click всегда означает swap и не является selection gesture.

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

В `TRACKING` click-to-move disabled/ignored в первой реализации. В `RELATIVE` левый click по main working frame является click-to-move; preview click не создаёт manual motion intent.

## Operational bar / Emergency / Motor controls

Нижняя operational bar всегда видима и использует компактные fixed-size controls: изменение текста/состояния не меняет их ширину/высоту и не сдвигает соседние элементы. Emergency остаётся визуально крупнейшим действием. Минимальный набор:

- switch `RELATIVE / TRACKING`;
- кликабельный motor state/control;
- connection state;
- крупный `EMERGENCY` в правом нижнем углу.

Motor control выполняет `MOTOR_ON` / `MOTOR_OFF` одним click без confirmation dialog. Отображаемое состояние меняется по подтверждённому `TurretState.motor_state`, а не optimistic UI state.

Dedicated ordinary `StopMotion` button в первом prototype отсутствует. Сам `StopMotion` остаётся control operation Core/Turret и используется автоматическими control boundaries, где это требует архитектура.

`StopMotion` — штатная остановка с acceleration limit.

В TRACKING StopMotion снимает selected target и оставляет mode TRACKING. В RELATIVE StopMotion отменяет неотправленный manual intent и через Turret приводит firmware к zero velocity. Уже физически отправленный ordinary request считается committed и заканчивает normal retry cycle; для немедленного прерывания используется Emergency.

Emergency — отдельный priority action без acceleration limit. Persistent emergency latch / Resume state в первой реализации нет.

## Настройки Vision

UI может редактировать:

- source/network settings;
- `processing-scope`;
- optional per-camera `processing-enabled` master switch;
- `vision-processor-class`.

В v1 внутренние detector/tracker tuning constants конкретного `VisionProcessor` в UI не редактируются и не являются полями `config.json`.

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
- max relative move per axis (restart-only);
- desired serial baudrate/port;
- simulation mode.

Все изменения проходят через Core/Config Manager.

## Потеря видеосигнала

UI хранит camera connection state отдельно от freshness.

```text
frame ещё не принят → neutral placeholder + camera name + CameraStatus
accepted frame + fresh → обычный кадр
accepted frame + stale → frozen last frame + dim + «НЕТ НОВЫХ КАДРОВ»
RECONNECTING / ERROR / STOPPED → реальный status поверх placeholder/frozen frame
```

Freshness вычисляется в main thread по monotonic `CameraStatus.last_receive_timestamp_ns`, а не по bool от producer. Threshold поступает в UI boundary из authoritative `vision.camera-stale-timeout-ms`: меньше configured timeout считается fresh, на границе и выше — stale. Это presentation state, а не новый `CameraState` и не новое config field.

Frozen frame остаётся полезным для ориентации, но не является актуальной aiming surface. RELATIVE/TRACKING interaction разрешена только для accepted generation, fresh frame и `CameraStatus.state == ONLINE`. Preview swap, operational bar и Emergency остаются доступны независимо от camera freshness/state.

Новый accepted `CameraSessionStarted` немедленно очищает displayed result прежней generation до принятия первого результата новой generation. Временное отсутствие кадров внутри той же accepted generation последний кадр не очищает.

## Производительность

UI использует main-thread `QTimer` с interval `16 ms` (примерно `60 Hz`) как revision-aware pump. Это polling frequency, а не video FPS: если revision не изменилась, image/result повторно не обрабатывается и `QImage/QPixmap` не создаётся. При нескольких публикациях между ticks читается только freshest latest payload.

В начале каждого tick UI сначала полностью draining'ит FIFO `CameraSessionStarted` каждой камеры, затем читает latest Vision/CameraStatus/TurretState snapshots. Успешно accepted revision помечается consumed; rejected result новой generation остаётся retryable, чтобы race `payload опубликован после drain, barrier будет принят на следующем tick` не терял кадр. Already accepted Vision/Turret revision повторно в Mediator не передаётся.

Per-frame queued Qt signals не используются: они могли бы превратить latest-state в event backlog и накапливать video latency. Более сложный coalesced event-driven wakeup откладывается до измеренной необходимости.

`CameraSessionStarted` — barrier event и не может быть потерян/coalesced как обычный latest notification.

Все QWidget/QPixmap/QPainter operations выполняются только в Qt main thread.

Открытый Operator Window не является pause state: main-thread pump, camera presentation, Turret state и уже активный tracking продолжают обновляться. Поскольку это отдельное окно, оно не меняет geometry или image-coordinate mapping main video.

## Границы

UI не:

- хранит authoritative `selected_target` или applied `control_mode`;
- автоматически меняет `main_camera` при отказе;
- выполняет stereo distance;
- вычисляет lead/ray/angles;
- выполняет PID;
- отправляет serial packets.

## Что ещё не определено

- детальный partial-failure UX для всех комбинаций availability;
- future latest-live-frame mode.
- FPS display: до реализации нужно разделить receive, accepted/displayed `VisionResult` и display FPS; предпочтительный обычный operator metric — accepted/displayed `VisionResult` FPS, а diagnostics сможет показывать несколько метрик. Существующий `ui.show-fps` не получает новую semantics в этом checkpoint.

Полный список: [Открытые вопросы](../../architecture/problems.md).
