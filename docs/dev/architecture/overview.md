# Общая архитектура системы

Этот документ описывает устойчивую верхнеуровневую структуру системы. Детали вынесены в документы модулей и общих контрактов.

Причины ключевых решений и существенные отвергнутые альтернативы собраны в [архитектурных решениях](./decisions.md).

![Общая архитектура системы](../diagrams/overview-diagram.png)

## Основные модули

Система разделена на четыре верхнеуровневых программных модуля:

- `Vision` — получает кадры трёх камер, исправляет их геометрию, формирует tracked objects и определяет дальность;
- `Core` — координирует систему и содержит `Mediator`, `Aiming`, `Config Manager`;
- `Turret` — преобразует требования управления в движение приводов и обменивается со STM32;
- `UI` — отображает working frames и состояние системы, принимает действия пользователя.

Core и UI работают в главном Qt-потоке. Vision использует три camera worker, Turret — отдельный worker.

## Vision и working frame

Фиксированные роли:

```text
overview
stereo-left
stereo-right
```

Публичные кадры всегда имеют исправленную геометрию:

```text
Overview:
RTP/JPEG UDP → GStreamer decode → fisheye undistort → FramePacket → VisionProcessor → VisionResult

Stereo Left / Right:
RTP/JPEG UDP → GStreamer decode → rectify → FramePacket → VisionProcessor → VisionResult
```

`FramePacket.image` — working frame. Raw image наружу как обычный `FramePacket` не публикуется.

Все публичные pixel coordinates (`BBox`, velocity, aim point, lead, UI click) относятся к working frame.

Каждый camera pipeline имеет `generation`. Перед данными новой generation Vision публикует ordered/barrier:

```text
CameraSessionStarted(camera, generation, camera_model)
```

В main thread общий `CameraSessionGate` атомарно принимает `generation + CameraModel`. Core и UI отбрасывают данные generation, которая не принята gate. Никакой consumer не должен принять `VisionResult generation=N` раньше session event `generation=N`.

`VisionProcessor` — заменяемый компонент и обязан выдавать `TrackedObject` со стабильными ID внутри одной generation.

Подробно: [Vision](../modules/vision/index.md).

## Calibration и геометрия

Calibration хранится отдельно от `config.json`:

```text
calibration/
  overview.json
  stereo.json
```

Overview использует OpenCV fisheye calibration (`K`, ровно 4 коэффициента `D`, `new_camera_matrix`) и fisheye-undistortion. Геометрия исправленного working frame строится по `new_camera_matrix`. Stereo Left/Right сохраняют отдельную pinhole/stereo OpenCV calibration и rectification.

Публичный геометрический контракт:

```text
CameraModel.pixel_to_ray(x, y) → normalized CameraRay
```

`CameraRay` использует `+Y вниз`, как image/OpenCV geometry. Aiming преобразует результат в логическую систему Turret, где `+Y вверх`.

При несовпадении calibration image size и фактического decoded camera image size pipeline не считается ready. Размер берётся из decoded sample, а не из requested sender resolution. Автоматический crop/resize/scaling calibration в первой реализации не выполняется.

## UI: main / preview

В обычном интерфейсе одновременно отображаются Overview и Stereo Left:

- одна камера — `main_camera`, занимает основную область;
- вторая — preview;
- пользователь может явно выполнить swap.

Начальная `main_camera` берётся из `ui.default-camera`; допустимы только `overview` и `stereo-left`.

Автоматического переключения `main_camera` при отказе камеры нет. UI показывает stale/error для текущей main camera, а swap остаётся явным действием пользователя.

Vision processing имеет scope:

```text
main-only
main-and-preview
```

BBox могут отображаться на preview, если processing для неё включён, но выбирать цель разрешено только на main image.

## Selection и TRACKING

Core хранит одну authoritative selection, но только для подтверждённого `TRACKING`:

```text
RELATIVE:
    selected_target = None

TRACKING:
    selected_target: TargetRef | None
```

`TargetRef = camera + generation + track_id`. Selection разрешён только на `main_camera`, после того как `TurretState.control_mode == TRACKING`. Неуспешная попытка выбора не создаёт target.

При swap в TRACKING selection очищается, `TrackingError`/lead/DistanceResult инвалидируются и Turret выполняет normal StopMotion. Сам applied mode остаётся TRACKING. В RELATIVE swap не отменяет уже успешно сформированный manual `MoveRelativeCommand`.

При временной потере selected track `TargetRef` сохраняется до `target-lost-timeout-ms`; новые TrackingError/setpoints не создаются. STM32 velocity watchdog безопасно ведёт velocity target к zero, если updates не возобновились.

При final loss/deselect в TRACKING:

```text
selected_target → None
lead / TrackingError → clear/invalidate
DistanceResult старой цели → invalidate
Turret normal StopMotion
```

PID принадлежит Turret Controller и reset'ится там по control boundaries; Core не посылает отдельный `PID_RESET`.

При смене цели A → B старый ещё не отправленный tracking motion intent инвалидируется, Turret reset'ит PID и рассчитывает новый setpoint без обязательного промежуточного zero.

При `TRACKING → RELATIVE` selection очищается сразу. Если physical mode transition не завершится из-за UART failure, безопасное состояние остаётся `TRACKING + no target`.

## Aiming

Aiming преобразует target/lead pixel и aim point в `CameraRay`, затем в логическую угловую ошибку Turret.

Постоянный `degrees_per_pixel` не используется.

Логический результат Aiming:

```text
+X → турели нужно вправо
+Y → турели нужно вверх
```

`invert-x/y` в HAL отвечают только за физическое направление двигателя.

## Stereo / Distance Provider

Distance Provider выполняется в потоке Stereo Left.

Аппаратно синхронизированные Stereo Left / Right сопоставляются по `capture_id`, но pairing также обязан учитывать `generation` обеих camera sessions. Кадры разных generations нельзя объединять даже при одинаковом `capture_id`.

Поддерживаются источники:

```text
stereo
manual
```

Первая реализация может использовать manual distance без готового stereo pairing.

## Turret

Turret состоит из `Turret Controller` и `Turret HAL`. Applied `control_mode` принадлежит Turret Controller и публикуется через `TurretState`; Core/UI не держат вторую authoritative копию.

Motion transport latest-only. Вместо двух независимых slots используется один:

```text
pending_motion =
    MoveRelativeCommand
    | AxisVelocitySetpoint
    | None
```

В подтверждённом `RELATIVE` допустим только `MoveRelativeCommand`; в `TRACKING` — только velocity setpoint. Новый неотправленный intent заменяет старый. Normal control boundary инвалидирует только неотправленный pending intent. Уже физически отправленный ordinary request считается committed и завершает обычный response/retry cycle перед следующей normal control operation.

Одновременно существует максимум один physical request in flight. Для немедленного прерывания normal retry cycle используется `EMERGENCY_STOP`.

### StopMotion и mode transition

`StopMotion` — штатная acceleration-limited остановка. `SET_VELOCITY(0,0)` — обычный нулевой velocity setpoint и не имеет скрытой application-семантики.

В TRACKING StopMotion очищает selection/TrackingError и оставляет applied mode TRACKING. В RELATIVE он отменяет неотправленный manual intent и затем переводит firmware в velocity-control с target zero.

При `RELATIVE → TRACKING` при motors ON всегда выполняется:

```text
invalidate unsent pending_motion
→ SET_VELOCITY(0,0) → OK
→ apply TRACKING
→ TurretState(control_mode=TRACKING)
```

Selection разрешается только после этого подтверждения. При motors OFF физическое движение невозможно, поэтому zero handshake не требуется.

### Emergency Stop

`EMERGENCY_STOP` немедленно прекращает STEP generation без acceleration limit и является одновременно safety + request-sequence resync boundary. Persistent latch отсутствует.

Если ordinary attempt уже физически отправлен, Emergency не передаётся параллельно: дальнейшие ordinary retries прекращаются, ожидается только response или timeout текущей попытки, затем отправляется Emergency с новым общим `REQUEST_ID`.

После successful Emergency STM32 устанавливает `expected_request_id = emergency_id + 1`, поэтому дополнительная transport-reset команда не нужна.

### Recovery

После transport loss старая session не продолжается. После обнаружения физической связи/baud:

```text
EMERGENCY_STOP → OK
→ MOTOR_OFF → OK
→ SET_BAUDRATE при необходимости
→ SET_CONFIG(full snapshot) → OK
→ READY
```

Auto `MOTOR_ON` отсутствует. Уже принятый ограниченный `MOVE_RELATIVE` при внезапной потере связи может закончиться; velocity control защищён watchdog.

## Relative move без completion lifecycle

`MOVE_RELATIVE → OK` означает только принятие новой relative target. `MOVE_COMPLETED`, `command_id`, terminal lifecycle и completion polling отсутствуют.

После успешной camera/session validation и вычисления relative angle последующий camera swap/restart/stale не отменяет уже сформированный manual intent.

PC ограничивает move в градусах; STM32 дополнительно применяет простой static sanity bound по `abs(delta_steps)`.

## PID

PID работает с угловой ошибкой:

- отдельные `Kp/Ki/Kd` по X/Y;
- шаг ровно один раз на новый `TrackingError` revision;
- `dt` по monotonic timestamp;
- conditional anti-windup;
- I-term ограничен `±max_speed`;
- reset при входе/выходе TRACKING, смене цели, final loss и при gap больше watchdog timeout.

Первый sample после reset: P-only, `I=0`, `D=0`.

## STM32 / serial transport

Первая реализация использует обычный full-duplex UART. Production RS485 half-duplex отложен и позднее должен заменить только physical byte transport без изменения binary protocol или PC-side semantics.

Обмен на уровне protocol строго последовательный:

```text
1 request → 1 response
```

STM32 не отправляет асинхронные packets.

Физические команды:

```text
PING
SET_CONFIG
SET_BAUDRATE
MOVE_RELATIVE
SET_VELOCITY
EMERGENCY_STOP
MOTOR_ON
MOTOR_OFF
```

Framing, CRC, retry, error codes и reconnect описаны в [Протоколе STM32](./serial-protocol.md).

## Ограничения механики первой реализации

В конструкции пока нет:

- limit switches;
- encoders;
- достоверного absolute position feedback;
- надёжного homing.

Физические упоры существуют, но попадание в них считается нештатной ситуацией. STM32 знает число выданных STEP pulses, а не гарантированное фактическое перемещение.

PC ограничивает величину одной relative move через `max-relative-move-*-deg` до перевода в steps. STM32 дополнительно отвергает аномально большой `delta_steps` по compile-time/static firmware bound; это не механический absolute limit.

## UI и запись

UI overlays рисуются только для отображения. Основная запись видео сохраняет чистый `FramePacket.image` без bbox, reticle, aim point, lead и status overlays, чтобы запись можно было повторно использовать для тестов Vision.

## Модель выполнения

Пять прикладных потоков:

1. main thread: UI + Core + `CameraSessionGate`;
2. Overview pipeline;
3. Stereo Left pipeline + Distance Provider;
4. Stereo Right pipeline;
5. Turret Controller + HAL.

Внутренние потоки Qt/GStreamer/OpenCV сюда не входят.

## Shared application composition

Production composition имеет один owner: `ApplicationRuntime`, создаваемый
`build_application_runtime(...)` из уже загруженных typed `AppConfig`,
`OverviewCalibration` и `StereoCalibration`. Загрузка файлов, разбор CLI и
создание `QApplication` этой boundary не принадлежат.

Текущий prototype runtime собирает ровно:

```text
Overview CameraWorker ─┐
Stereo Left CameraWorker ─┼→ один Mediator / один CameraSessionGate
TurretWorker ───────────┘
```

Оба camera worker создаются существующим `build_camera_worker(...)` с реальными
corrector из переданных calibration. Production defaults используют
`GStreamerRtpJpegSource`, а Turret создаётся через существующий
`TurretWorker(config.turret)` и его production transport policy. Узкие factories
для двух camera sources и physical Turret transport существуют только как
explicit dependency-injection seams; silent fallback на InMemory/Fake отсутствует.

Runtime сразу предоставляет полную UI dependency surface: один `Mediator`,
bindings pipeline-owned barrier/result/status каналов Overview и Stereo Left,
`TurretWorker.state_updates` и `camera-stale-timeout-ms`. `Stereo Right` в текущий
normal UI/application flow не входит.

Normal и diagnostic launchers передают сюда effective typed inputs и вызывают
одну и ту же composition. Normal entrypoint — `python -m navmin` (или root
`main.py`): он принимает пути к `config.json`, `calibration/overview.json` и
`calibration/stereo.json`, использует только production RTP receivers и запрещает
`turret.emulate-stm32=true`, чтобы normal запуск не мог молча перейти на
`FakeTransport`.

Эти file-backed inputs являются локальными site-specific файлами и не входят в repository baseline. Strict loaders не создают и не чинят их автоматически. Если обычный GUI startup получает missing/invalid input, launcher до preflight/workers может предложить оператору закрыть программу либо выполнить явный recovery: timestamped backup всего существующего input set → safe non-hardware-ready defaults → завершение текущего startup для ручной проверки параметров. `--preflight-only` и diagnostic/headless paths такого диалога не открывают.

Diagnostic entrypoint — `python tools/run_diagnostic_app.py`. Он требует явного
выбора `--overview real|localhost`, `--stereo-left real|localhost` и
`--turret real|pty`. Localhost RTP senders и PTY STM32 emulator остаются внешними
diagnostic endpoints и не принадлежат `ApplicationRuntime`; внутренними
production boundaries остаются `GStreamerRtpJpegSource` и `SerialTransport`.
Localhost sender использует resolution загруженной calibration соответствующей
camera role, а PTY selection формирует effective Turret config со stable PTY path
и `emulate-stm32=false`. Для полностью software-only сочетания
`localhost + localhost + pty` diagnostic launcher также поддерживает явный
`--synthetic-inputs`: launcher-owned 320×240 diagnostic config/calibrations
создаются в памяти и не требуют локальных files. Этот режим запрещён для любых
mixed/real endpoint selections; normal launcher и mixed/real diagnostics
по-прежнему используют file-backed typed inputs.

Оба launcher создают отдельный UTC-timestamped session directory под `logs/`
(или под parent directory из `--log-dir`) до загрузки config/calibration. Session
owner остаётся launcher-side и хранит bounded rotating `runtime.log`, `manifest.json`,
`preflight.json` и `inputs/` с effective config/calibration и provenance/hash evidence. Normal file
log имеет INFO level, diagnostic — DEBUG; console logging сохраняется. Один runtime
log ограничен 10 MiB с пятью backup files, а старые session directories автоматически
не удаляются: cross-session retention остаётся ручной ответственностью оператора.
Backend-aware static preflight также остаётся launcher-side и не принадлежит
`ApplicationRuntime`. После typed input loading normal launcher проверяет real
receiver + real serial prerequisites; diagnostic launcher проверяет именно выбранные
`real|localhost` camera и `real|pty` Turret backends до запуска внешних endpoints.
`FAIL` записывает `preflight.json`, завершает manifest как `preflight-failed` и не
создаёт `ApplicationRuntime`, `QApplication`, camera/turret workers, localhost sender
или PTY service. `WARN` startup не блокирует. `--preflight-only` выполняет ту же
static boundary и завершает session без runtime-specific effective PTY path.
`SoftwareSmokeRuntime` также делегирует общую worker/Mediator/lifecycle wiring
`ApplicationRuntime`, но сохраняет ownership своих `InMemoryFrameSource` и
synthetic producers.

## Startup / shutdown order

После того как launcher загрузил/проверил inputs и построил runtime, deterministic
startup order текущей composition:

```text
1. TurretWorker
2. Overview CameraWorker
3. Stereo Left CameraWorker
```

Критический invariant: до запуска Vision workers уже существуют единственный
`CameraSessionGate` и pipeline-owned lossless barrier channels, связанные с
готовыми UI bindings. `run_ui()` вызывается только после `runtime.start()` и
сначала drain'ит накопленные barriers, поэтому данные generation не принимаются
раньше соответствующего `CameraSessionStarted`.

Успешный запуск потока не означает готовность hardware: camera ERROR после
успешного `CameraWorker.start()` и Turret reconnecting остаются обычными worker
states и не завершают весь runtime. Если синхронный `start()` очередного worker
падает, runtime автоматически останавливает уже запущенные components в обратном
порядке и повторно поднимает исходную startup exception.

Bounded application shutdown сначала сохраняет safety boundary Turret:

```text
если Turret READY и Motor ON:
STOP_MOTION
→ MOTOR_OFF
→ дождаться подтверждённого MotorState.OFF в пределах shutdown timeout
```

Если Turret уже недоступен, runtime не зависает на недостижимом подтверждении и
переходит к bounded cleanup. После safety boundary owned workers останавливаются:

```text
1. Stereo Left CameraWorker
2. Overview CameraWorker
3. TurretWorker
```

Shutdown idempotent и безопасен после полного start, partial-start rollback и
самостоятельного завершения worker. Ошибка остановки одного component не
прерывает cleanup остальных; после всех попыток runtime сообщает aggregate
failure. Timeout применяется к каждой из трёх bounded worker boundaries, поэтому
общая верхняя граница также конечна. Runtime one-shot и не перезапускается;
camera/Turret reconnect остаётся responsibility соответствующего worker.

## Межпотоковая семантика

```text
VisionResult        → latest-only per camera
DistanceResult      → latest-only / invalidatable
TrackingError       → latest-only + monotonic revision
TurretState          → latest-only
CameraStatus         → latest-only per camera
ConfigUpdate         → latest-only
CameraSessionStarted → ordered/barrier
pending_motion       → one latest unsent motion intent
Emergency Stop       → dedicated priority operation
```

Типизированные состояния покрывают runtime state, диагностические сообщения идут в logging, а reconnect/recovery принадлежит owner-модулям Vision/Turret. UI coalesce'ит latest-state через revision-aware main-thread QTimer pump; `CameraSessionStarted` остаётся отдельным lossless barrier.

## Конфигурация

Config Manager загружает `config.json`, валидирует его и публикует типизированные snapshots. Calibration хранится отдельно.

`config.json` имеет `schema-version = 1`; сохранение выполняется через temporary file + atomic replace. Подробно: [Конфигурация](./configuration.md).

## Контракты и открытые вопросы

- [Общие контракты данных](./contracts.md)
- [Открытые вопросы](./problems.md)
