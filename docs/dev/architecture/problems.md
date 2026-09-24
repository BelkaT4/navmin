# Открытые вопросы архитектуры

Этот файл содержит только вопросы, которые пока не имеют окончательного решения. Уже принятые решения сюда не включаются.

На момент синхронизации текущий набор **не блокирует начало первой реализации**; для каждого вопроса указан этап, к которому его нужно закрыть.

## До интеграции с реальным RS485-железом

### 1. Физическое управление полудуплексным RS485

RS485 перенесён за пределы обязательной первой реализации: Stage 4 и v1 используют обычный UART. Этот вопрос **не блокирует** UART-first firmware или завершение v1 и возвращается при отдельной RS485 migration.

Логическая request/response-модель, framing, CRC, `REQUEST_ID`, Emergency-resync, retry и baudrate уже определены и должны сохраниться без protocol fork. Для будущей RS485 physical layer остаются hardware-dependent детали:

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


### 7. Edge cases runtime config apply

Базовая classification уже определена в `configuration.md`. Turret-specific PID apply semantics закрыты и перенесены в `configuration.md`, `modules/turret/index.md` и `decisions.md`.

Остаются детали:

- изменение `target-lost-timeout-ms` для уже временно потерянной цели;

STM32 max speed / acceleration / velocity watchdog уже применяются dynamic полным атомарным `SET_CONFIG` snapshot. Axis mechanics (`invert`, steps/rev, microstep, `max-relative-move-deg`) — restart-only и safe-point semantics для них не нужна.

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

Для подготовки автономного hardware day launcher-side session evidence уже имеет:

- отдельный per-run session directory для normal и diagnostic;
- bounded rotating file logging (normal INFO, diagnostic DEBUG);
- `manifest.json` с lifecycle outcome, backend/input mode и allowlisted runtime/Git metadata;
- effective config/calibration copies и file-backed SHA-256 либо явную synthetic provenance;
- manual cross-session retention без automatic deletion.

Logging остаётся diagnostics output, а typed runtime state по-прежнему не заменяется
парсингом логов.

Backend-aware startup preflight, `preflight.json`, `preflight-failed` и
`--preflight-only` закрыты launcher-side: mandatory FAIL происходит до external
endpoints/workers/UI, а static checks выбираются по real/localhost и real/PTY
backend.

Остаются следующие integration details:

- production level/config source beyond fixed normal-vs-diagnostic baseline;
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

### 22. Future latest-live-frame UI

Сейчас UI показывает `VisionResult.frame`, поэтому frame+bbox синхронизированы.

Если позже понадобится separate freshest live frame, потребуется отдельный overlay synchronization contract.

### 23. D-filter и PID tuning

После measurements проверить D noise, derivative kick и sufficiency текущего anti-windup.

### 24. Per-camera Aiming parameters

Aim point уже per-camera. После prototype проверить, достаточно ли общих `lead-time-ms` и `target-lost-timeout-ms` при разных FPS/latency Overview и Stereo Left.

### 25. Diagnostics / profiling

Software soak уже получил bounded observability для command/transport histories, reconnect count, thread count, RSS и cycle timing. Targeted memory diagnosis на synthetic `InMemoryFrameSource + FakeTransport` path не нашёл unbounded Python-level retention: nominal, generation-restart, reconnect и stale/resume runs показали ранний RSS allocation step с последующим plateau, а Legacy14/OpenCV microbench подтвердил bounded native allocator/cache behaviour. Это не закрывает profiling на production RTP/SerialTransport и real hardware.

Для localhost/VIRTUAL добавлена только визуальная наблюдаемость camera latency:
sender наносит `SOURCE HH:MM:SS.mmm` и `FRAME n` до JPEG/RTP/UDP, а diagnostic
UI показывает `NOW HH:MM:SS.mmm` в том же clock domain одного PC. Это позволяет
заметить frozen frame и визуально проверить, что gap не растёт, но не является
калиброванным benchmark. Real Raspberry Pi camera end-to-end latency остаётся
открытой: отдельные clocks, их synchronization, network path и sender scheduling
этим механизмом не измеряются.

Минимальные candidates для следующих transport/hardware checkpoints:

- camera FPS;
- Vision processing time;
- frame age/latency;
- dropped frames;
- Stereo time;
- reconnect count;
- UART timeout/error count;
- command latency;
- TrackingError/SET_VELOCITY frequency;
- watchdog stops;
- RSS/allocator behaviour на production RTP/GStreamer и SerialTransport paths.

### 26. Режим «Самая быстрая»

Определить судьбу старого automatic target-selection mode: basic product, postponed feature или удаление. Режим «Ближайшая» уже отложен.

### 27. Vision load adaptation

Latest-frame model уже исключает processing backlog. После profiling решить, нужны ли intentional frame skipping, processor optimization/replacement или снижение Stereo load.
