# Открытые вопросы архитектуры

Этот файл содержит только вопросы, которые пока не имеют окончательного решения. Уже принятые решения сюда не включаются.

На момент синхронизации текущий набор **не блокирует начало первой реализации**; для каждого вопроса указан этап, к которому его нужно закрыть.

## До интеграции с реальным RS485-железом

### 1. Физическое управление полудуплексным RS485

Логическая request/response-модель, framing, CRC, `REQUEST_ID`, Emergency-resync, retry и baudrate уже определены. Остаются hardware-dependent детали:

- управляет ли STM32 `DE/RE` или transceiver/adapter делает это автоматически;
- момент освобождения шины после последнего TX byte;
- нужна ли turnaround delay перед response;
- реальные timing по измерениям;
- общий reference/GND для неизолированных устройств или galvanic isolation;
- termination/biasing для реальной линии и большой длины кабеля.

## До включения полноценного Stereo distance

### 2. Механизм `capture_id` и stereo pairing

Generation уже не позволяет смешивать разные camera sessions, но нужно определить:

- кто создаёт `capture_id` и как обе Raspberry Pi получают одно значение;
- старт/перезапуск нумерации;
- обнаружение и восстановление рассинхронизации;
- wraparound;
- duplicate/missing/out-of-order capture IDs;
- `pair-timeout-ms`;
- очистку bounded Stereo Right pairing buffer.

### 3. Lifecycle / staleness `DistanceResult`

Остаётся определить:

- owner `distance-stale-timeout-ms`;
- как latest `DistanceResult` явно инвалидируется (`None`) после staleness;
- поведение при `stereo ↔ manual`;
- защита от late result старого `TargetRef`/generation после длительного stereo computation.

## Во время первой реализации


### 5. Qt notification coalescing

Latest payload не должен создавать backlog queued Qt signals.

Нужно определить единый notification/coalescing pattern для Vision, Turret и ConfigUpdate. `CameraSessionStarted` является barrier и не может быть потерян/coalesced как обычный latest notification.

### 6. Processor-specific configuration

Нужно определить:

- формат settings конкретного `VisionProcessor`;
- schema validation;
- какие поля доступны UI;
- persistence;
- какие changes dynamic, а какие требуют pipeline restart/new generation.

### 7. Edge cases runtime config apply

Базовая classification уже определена в `configuration.md`. Turret-specific PID apply semantics закрыты и перенесены в `configuration.md`, `modules/turret/index.md` и `decisions.md`.

Остаются детали:

- изменение `target-lost-timeout-ms` для уже временно потерянной цели;
- processor-specific dynamic/restart policy.

STM32 max speed / acceleration / velocity watchdog уже применяются dynamic полным атомарным `SET_CONFIG` snapshot. Mechanical conversion (`invert`, steps/rev, microstep) — restart-only и safe-point semantics для них не нужна.

### 8. Camera reconnect transitions / backoff

Enum уже определён:

```text
STARTING
ONLINE
RECONNECTING
ERROR
STOPPED
```

Freshness вычисляется отдельно по timestamp.

Остаётся определить:

- точные state transitions;
- reconnect/backoff;
- критерий устойчивого ERROR;
- restart/reset `VisionProcessor`;
- reconnect history/logging.

При каждом новом pipeline start создаётся новая `generation`.

### 10. Worker lifecycle

Foundation уже определил общий cooperative `StopToken` и минимальную `request_stop() / join() / is_alive()` boundary. Turret-specific lifecycle закрыт: Turret worker создаёт/останавливает application orchestration, serial waits bounded/cancellable, reconnect backoff использует `StopToken`, join bounded, а незавершившийся worker считается явной shutdown error. Numeric join timeout остаётся implementation tuning, а не config field.

Для будущих owner/integration stages остаётся определить только:

- как Vision owner прерывает blocking GStreamer/video wait;
- детали общего startup/shutdown orchestration нескольких workers и main-thread компонентов.

Reconnect/recovery принадлежит owner-модулям; отдельный orchestration component без concrete v1 responsibility не вводится.

### 11. Частичные отказы

Нужно определить доступность функций/UI в сценариях:

- Overview работает, Stereo Left unavailable;
- Stereo Left работает, Stereo Right unavailable;
- distance unavailable;
- STM32 unavailable;
- одна camera в RECONNECTING;
- PC config изменён, но `SET_CONFIG` STM32 ещё не подтверждён.

`main_camera` автоматически на другую камеру не переключается.


### 13. Logging

Foundation уже имеет единый idempotent bootstrap стандартного Python logging для `navmin` logger tree. Runtime state передаётся typed contracts, transient diagnostics — logging; generic event bus не вводится.

Turret-specific policy закрыта: в v1 остаётся обычный Python logging без `QueueHandler/QueueListener`; INFO содержит lifecycle/connection/recovery/baud boundaries и значимые failures, transaction/retry details доступны для diagnostics, а high-rate PID/setpoint traffic не логируется на INFO. Queue-based logging возвращается только при измеренной contention/blocking problem.

Остаются общие integration/UI details, которые нужно решать только при реальной потребности:

- rotation / file limits;
- production levels/config source;
- как UI показывает последние важные ошибки без превращения logging в machine-readable state.

### 14. Будущие STM32 hardware events

Wire response сохраняет reserved `EVENTS` section, но в v1 она всегда пустая и event queue заранее не проектируется.

При появлении первого реального hardware event нужно определить вместе:

- concrete event code/payload schema;
- нужен ли STM32 pending-event buffer;
- capacity / overflow policy;
- delivery/retry semantics;
- typed PC contract вместо generic event, если это состояние лучше выразить отдельным типом.

## После первых измерений и базового прототипа

### 15. Timing budget TRACKING и точные timeout

Роли timeout уже определены:

```text
velocity-watchdog-timeout-ms < target-lost-timeout-ms
```

Serial defaults:

```text
response-timeout-ms = 100
max-retries = 2
inter-request-delay-ms = 2
```

После измерений нужно подобрать camera FPS, Vision frequency, latency/jitter, watchdog, target-lost timeout и запас на config/serial exchanges.

### 16. Backlash compensation

После mechanical tests решить:

- нужна ли software compensation;
- только MoveRelative или также TRACKING;
- Controller/HAL/STM32;
- взаимодействие с PID и acceleration limiter.

### 17. Camera-to-turret rotational extrinsic

Первая версия предполагает близкую параллельность axes и компенсирует constant boresight через aim point.

После tests решить необходимость `R_camera_to_turret`, calibration procedure и storage.

### 18. Target handoff Overview ↔ Stereo Left

Сейчас одна selected target существует только в TRACKING на текущей `main_camera`, а swap её сбрасывает.

Будущий Target Handoff/Reacquisition должен сопоставлять один физический объект между независимыми VisionProcessors и учитывать latency streams.

### 19. UI gestures selection / deselect

Уже определено:

- target выбирается только на main image и только в confirmed TRACKING;
- в RELATIVE selection отсутствует;
- click-to-move разрешён только в confirmed RELATIVE;
- invalid/stale selection не создаёт target;
- swap в TRACKING сбрасывает selection.

Остаётся UX:

- mouse/key gesture выбора bbox;
- explicit deselect;
- click empty area;
- overlapping bbox;
- нужен ли nearest-object helper.

### 20. Object overlay content

Определить, какие данные показывать рядом с bbox: ID, distance, velocity, selection status, diagnostics. `confidence` не является обязательным полем общего `TrackedObject`.

### 21. Stereo Right diagnostics

Определить layout/activation diagnostic view и processor-specific overlays. Stereo Right не становится обычной `main_camera`.

### 22. Future latest-live-frame UI

Сейчас UI показывает `VisionResult.frame`, поэтому frame+bbox синхронизированы.

Если позже понадобится separate freshest live frame, потребуется отдельный overlay synchronization contract.

### 23. D-filter и PID tuning

После measurements проверить D noise, derivative kick и sufficiency текущего anti-windup.

### 24. Per-camera Aiming parameters

Aim point уже per-camera. После prototype проверить, достаточно ли общих `lead-time-ms` и `target-lost-timeout-ms` при разных FPS/latency Overview и Stereo Left.

### 25. Diagnostics / profiling

Минимальные candidates:

- camera FPS;
- Vision processing time;
- frame age/latency;
- dropped frames;
- Stereo time;
- reconnect count;
- UART timeout/error count;
- command latency;
- TrackingError/SET_VELOCITY frequency;
- watchdog stops.

### 26. Режим «Самая быстрая»

Определить судьбу старого automatic target-selection mode: basic product, postponed feature или удаление. Режим «Ближайшая» уже отложен.

### 27. Vision load adaptation

Latest-frame model уже исключает processing backlog. После profiling решить, нужны ли intentional frame skipping, processor optimization/replacement или снижение Stereo load.
