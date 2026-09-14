# Общие контракты данных

Этот документ описывает публичные типы данных и устойчивую семантику обмена между основными частями приложения.

Контракты находятся в общем Python-пакете, например `contracts/`, а не принадлежат `Core`, `Vision`, `UI` или `Turret`. Для внутренних контрактов одного процесса используются типизированные `dataclass`, `Enum` и небольшие immutable objects.

## `CameraRole`

```python
class CameraRole(Enum):
    OVERVIEW = "overview"
    STEREO_LEFT = "stereo_left"
    STEREO_RIGHT = "stereo_right"
```

`CameraRole` — общий enum для Vision/Core/UI/Aiming. В `config.json` человекочитаемые ключи используют `overview`, `stereo-left`, `stereo-right`, а Python enum остаётся единым типизированным контрактом.

## `FramePacket`

```python
@dataclass(frozen=True)
class FramePacket:
    camera: CameraRole
    generation: int
    frame_id: int
    capture_id: int | None
    receive_timestamp_ns: int
    image: np.ndarray
```

Правила:

- `camera` — фиксированная роль камеры;
- `generation` — номер текущего запуска camera pipeline внутри процесса;
- `frame_id` увеличивается внутри одной `generation` и при новом запуске pipeline начинается заново;
- обычная идентичность кадра — `camera + generation + frame_id`;
- `capture_id` нужен только для Stereo Left / Stereo Right; для Overview он равен `None`;
- `receive_timestamp_ns` фиксируется через `time.monotonic_ns()` после полного decode исходного кадра;
- `image` — публичный **working frame**, `np.ndarray`, `uint8`, BGR, форма `(height, width, 3)`;
- для Overview working frame уже undistorted;
- для Stereo Left / Stereo Right working frame уже rectified;
- raw frame наружу как обычный `FramePacket` не публикуется;
- после публикации `image.flags.writeable = False`.

Video Source / camera pipeline должен гарантировать стабильность памяти опубликованного `image`. Если backend может повторно использовать буфер, перед публикацией выполняется один `image.copy()`.

Точный механизм создания и синхронизации `capture_id` остаётся открытым вопросом до включения полноценного Stereo distance.

## `CameraRay` и `CameraModel`

```python
@dataclass(frozen=True)
class CameraRay:
    x: float
    y: float
    z: float
```

`CameraModel` — immutable модель геометрии **working frame** конкретной camera session. Публичный контракт:

```text
CameraModel.pixel_to_ray(x_px, y_px) → normalized CameraRay
```

Система координат `CameraRay`:

```text
+x → вправо на изображении
+y → вниз на изображении
+z → вперёд от камеры
```

`CameraModel` скрывает детали OpenCV `K/D/P/R`. Aiming не должен напрямую зависеть от этих матриц.

## `CameraSessionStarted`

Каждая новая camera session публикуется ordered/barrier event до любых camera-derived данных этой `generation`.

```python
@dataclass(frozen=True)
class CameraSessionStarted:
    camera: CameraRole
    generation: int
    camera_model: CameraModel
    timestamp_ns: int
```

Обязательная гарантия порядка **для каждого consumer**:

```text
increment generation
→ CameraSessionStarted(camera, generation, camera_model)
→ consumer принимает session и обновляет CameraSessionGate
→ только затем consumer может принять VisionResult / другие camera-derived данные generation=N
```

`CameraSessionStarted` нельзя drop/coalesce как обычное latest-state notification.

В main thread существует общий `CameraSessionGate`, который хранит минимум:

```text
accepted_generation[camera]
camera_model[camera]
```

Core и UI используют один gate. Данные неизвестной или старой `generation` отбрасываются.

## `CameraState` и freshness

Connection/lifecycle state камеры:

```python
class CameraState(Enum):
    STARTING = "starting"
    ONLINE = "online"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"
```

Latest status камеры передаётся отдельным typed state:

```python
@dataclass(frozen=True)
class CameraStatus:
    camera: CameraRole
    state: CameraState
    generation: int | None
    last_receive_timestamp_ns: int | None
    error_code: str | None = None
    message: str | None = None
```

`CameraStatus` — latest-only per camera. `generation` относится к текущей/последней известной session, если она существует.

`STALE` не является отдельным взаимоисключающим `CameraState`. Свежесть вычисляется consumer по `last_receive_timestamp_ns`:

```text
is_stale = now_monotonic_ns - last_receive_timestamp_ns > stale_timeout_ns
```

Это позволяет обнаружить stale даже если producer полностью перестал публиковать новые сообщения. `message` предназначен для UI/диагностики и не используется как machine-readable state.

## `BBox`

```python
@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    width: int
    height: int
```

Публичная рамка:

- задаётся в пикселях working frame;
- использует `xywh`;
- ограничена границами изображения;
- не содержит отрицательных или выходящих за кадр координат.

Внутри `VisionProcessor` разрешены float/subpixel координаты. Округление выполняется при формировании публичного `BBox`.

## `TrackedObject`

```python
@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    bbox: BBox
    velocity_x_px_s: float
    velocity_y_px_s: float
    age_frames: int
```

Правила:

- `track_id` устойчиво связывает объект между обработанными кадрами;
- ID уникален внутри одной camera `generation`;
- ранее выданный ID не переиспользуется для другого объекта в той же generation;
- `bbox` и velocity относятся к working frame;
- velocity — скорость центра bbox в пикселях в секунду;
- `age_frames` — возраст трека в обработанных кадрах;
- history, confidence и внутреннее состояние алгоритма остаются внутри `VisionProcessor`.

## `VisionResult`

```python
@dataclass(frozen=True)
class VisionResult:
    frame: FramePacket
    tracked_objects: tuple[TrackedObject, ...]
    processing_time_ns: int
```

`VisionResult` содержит ровно тот `FramePacket`, который был обработан. `generation` доступна через `frame.generation`.

UI рисует bbox поверх `VisionResult.frame`, поэтому отдельная синхронизация кадра и рамок не нужна.

На границе worker → main thread `VisionResult` имеет latest-only семантику per camera: старые результаты можно пропускать, но нельзя принимать result до barrier `CameraSessionStarted` той же generation.

## `TargetRef`

```python
@dataclass(frozen=True)
class TargetRef:
    camera: CameraRole
    generation: int
    track_id: int
```

`TargetRef` однозначно ссылается на track внутри конкретной camera session.

Core хранит **один** authoritative `selected_target`, но только для режима `TRACKING`:

```text
RELATIVE:
    selected_target = None

TRACKING:
    selected_target: TargetRef | None
```

Цель разрешено выбирать только на `main_camera` и только после подтверждённого `TurretState.control_mode == TRACKING`. `Stereo Right` пользовательского selection не имеет.

Совпадение цели требует `camera + generation + track_id`. При новом запуске camera pipeline старый `TargetRef` немедленно становится недействительным.

При временном отсутствии выбранного track Core сохраняет `TargetRef` до `target-lost-timeout-ms`. После timeout selection очищается, но applied `TurretControlMode` автоматически не меняется. При переходе в `RELATIVE` selection всегда очищается.

## `DistanceResult`

```python
class DistanceSource(Enum):
    STEREO = "stereo"
    MANUAL = "manual"


@dataclass(frozen=True)
class DistanceResult:
    target: TargetRef
    distance_m: float
    source: DistanceSource
    source_frame_id: int | None
    capture_id: int | None
    measured_timestamp_ns: int
```

Правила:

- `distance_m` — метры;
- Stereo-result содержит `source_frame_id` левого кадра и `capture_id` пары;
- Manual-result имеет `source_frame_id = None`, `capture_id = None`;
- `generation` отдельно не дублируется, поскольку входит в `target`;
- Stereo-result используется не дольше `distance-stale-timeout-ms`;
- Manual-result действителен, пока источник manual и `TargetRef` актуален;
- при смене/снятии цели или camera generation результат старой цели немедленно инвалидируется;
- межпотоковая семантика — latest state, который должен уметь явно переходить в `None`/invalid.

## Motion intent и команда относительного движения

```python
@dataclass(frozen=True)
class MoveRelativeCommand:
    delta_x_deg: float
    delta_y_deg: float
```

`MoveRelativeCommand` — одноразовый manual motion intent, который создаётся только в подтверждённом `RELATIVE`. После успешной camera/session validation и расчёта угла дальнейший camera swap/restart/stale не отменяет уже сформированный intent.

Межмодульная граница Core → Turret различается по режимам:

```text
RELATIVE:
    Core → MoveRelativeCommand → Turret Controller

TRACKING:
    Core → TrackingError → Turret Controller → PID → AxisVelocitySetpoint
```

`AxisVelocitySetpoint` не является Core → Turret контрактом: он создаётся только внутри Turret после PID.

Внутри Turret существует один latest-only slot для ещё не отправленного физического motion intent:

```text
pending_motion:
    MoveRelativeCommand
    | AxisVelocitySetpoint
    | None
```

Тип допустимого значения определяется applied `TurretState.control_mode`:

```text
RELATIVE → только MoveRelativeCommand
TRACKING → только AxisVelocitySetpoint
```

Новый ещё не отправленный motion intent заменяет предыдущий. На control boundaries старый **неотправленный** `pending_motion` инвалидируется. Уже физически отправленный ordinary request считается committed и завершает обычный response/retry cycle до следующей normal control operation.

Для `MOVE_RELATIVE`:

- после `MOVE_RELATIVE → OK` PC знает только, что STM32 приняла новую relative target;
- PC не отслеживает естественный момент окончания;
- новый успешно принятый `MOVE_RELATIVE` заменяет previous relative target STM32;
- `SET_VELOCITY`, `MOTOR_OFF` и `EMERGENCY_STOP` очищают relative target по своим контрактам;
- `command_id`, `MOVE_COMPLETED` и lifecycle `Completed / Cancelled / Failed` отсутствуют.

Причина: без encoder/limit feedback окончание запланированной STEP sequence не подтверждает фактическое достижение положения и не нужно control logic.

## Режим управления Turret

```python
class TurretControlMode(Enum):
    RELATIVE = "relative"
    TRACKING = "tracking"
```

Startup mode — `RELATIVE`.

Applied mode принадлежит Turret Controller и публикуется только через `TurretState.control_mode`. Core/UI могут иметь pending request на смену режима, но не держат вторую независимую authoritative копию applied mode.

Допустимо:

```text
RELATIVE + selected_target = None
TRACKING + selected_target = None
TRACKING + selected_target = TargetRef
```

`RELATIVE + selected_target` недопустимо. Только пользователь запрашивает `RELATIVE ↔ TRACKING`. `MOVE_RELATIVE`/click-to-move разрешён только в подтверждённом `RELATIVE`; target selection — только в подтверждённом `TRACKING`.

## `TrackingError`

```python
@dataclass(frozen=True)
class TrackingError:
    target: TargetRef
    error_x_deg: float
    error_y_deg: float
    timestamp_ns: int
```

На выходе Aiming используется **логическая система Turret**:

```text
+X → вправо
+Y → вверх
```

То есть Aiming явно преобразует image/camera `+Y вниз` в turret `+Y вверх`. HAL `invert-y` применяется только к физическому направлению двигателя и не исправляет математическую систему координат.

`TrackingError` передаётся как latest state с monotonic `revision`/sequence у container-а. Turret Controller выполняет PID step ровно один раз на новое revision и хранит `last_processed_revision`.

Latest-state abstraction должна поддерживать `clear()/invalidate()`. Invalidation тоже создаёт новое monotonic revision, чтобы consumer гарантированно увидел границу даже без нового `TrackingError` value.

## `AxisVelocitySetpoint`

```python
@dataclass(frozen=True)
class AxisVelocitySetpoint:
    velocity_x_deg_s: float
    velocity_y_deg_s: float
    timestamp_ns: int
```

Setpoint не имеет `command_id`.

- новое значение полностью заменяет старое;
- промежуточные значения можно пропускать;
- Controller clamp'ит PID output по `±max_speed`;
- HAL применяет hardware `invert`, переводит градусы в STEP-единицы и передаёт `SET_VELOCITY`.

## Состояние Turret

```python
class TurretConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    ERROR = "error"


class MotorState(Enum):
    UNKNOWN = "unknown"
    OFF = "off"
    ON = "on"
```

Концептуально `TurretState` содержит минимум:

```python
@dataclass(frozen=True)
class TurretState:
    connection_state: TurretConnectionState
    motor_state: MotorState
    control_mode: TurretControlMode
    max_speed_x_deg_s: float
    max_speed_y_deg_s: float
    acceleration_x_deg_s2: float
    acceleration_y_deg_s2: float
```

Правила:

- startup `control_mode = RELATIVE`;
- `control_mode` — authoritative applied mode Turret Controller;
- `READY` возможен только после успешной reconnect/init sequence и подтверждённого `SET_CONFIG`;
- `CONNECTING` используется во время initial connect/automatic reconnect/recovery; отсутствие STM32/serial device не переводит Turret в `ERROR`, пока owner может продолжать automatic reconnect;
- `ERROR` зарезервирован для действительно невосстановимой локальной ошибки/invariant failure, при которой automatic reconnect нельзя корректно продолжить;
- после потери связи `motor_state = UNKNOWN`;
- speed/acceleration — последние подтверждённо применённые пределы STM32;
- достоверного absolute position и признака естественного завершения relative move в state нет;
- `TurretState` — latest-only;
- смена mode считается применённой после публикации нового `TurretState` с новым `control_mode`.

Turret Controller может сохранить applied mode через UART reconnect, но selection, PID input и старое физическое движение не восстанавливаются. После recovery motors остаются OFF до нового явного `MOTOR_ON`.

## `StopMotion` и `EMERGENCY_STOP`

`StopMotion` — normal control operation с acceleration-limited остановкой. Значение `SET_VELOCITY(0,0)` само по себе не имеет специальной application-семантики: это обычный нулевой velocity setpoint.

В `TRACKING` Core очищает:

```text
selected_target → None
lead / TrackingError → clear/invalidate
DistanceResult старой цели → invalidate
```

Turret Controller, владеющий PID, видит control boundary, инвалидирует старый неотправленный `pending_motion`, reset'ит PID и при motors ON выполняет `SET_VELOCITY(0,0)`. Applied mode остаётся TRACKING.

В `RELATIVE` StopMotion инвалидирует старый неотправленный manual `pending_motion`, затем при motors ON выполняется `SET_VELOCITY(0,0)`, что очищает firmware relative target и плавно ведёт velocity target к zero.

Если ordinary motion request уже физически in flight, он считается committed и сначала завершает обычный response/retry cycle. Если retries исчерпаны, normal traffic не продолжается и начинается recovery.

Mode transition использует те же правила. В частности `RELATIVE → TRACKING` при motors ON всегда проходит через:

```text
invalidate unsent pending_motion
→ SET_VELOCITY(0,0) → OK
→ Turret Controller applies TRACKING
→ TurretState(control_mode=TRACKING)
```

Selection становится доступна только после этого подтверждённого `TurretState`. При `TRACKING → RELATIVE` Core сразу очищает selection/TrackingError; если UART transition затем не завершится, безопасное состояние остаётся `TRACKING + no target`.

`EMERGENCY_STOP` — единственное исключение из полного ordinary retry cycle:

- old unsent `pending_motion` инвалидируется;
- новые normal motion actions блокируются на время операции/recovery;
- если ordinary attempt уже физически отправлен, дальнейшие retries этой ordinary transaction не выполняются: ждём только response либо timeout текущей попытки;
- затем Emergency отправляется с новым общим `REQUEST_ID`;
- STM32 немедленно прекращает STEP generation без acceleration limit, очищает relative/velocity target и сохраняет drivers enabled;
- successful Emergency одновременно является request-sequence resync boundary; persistent emergency latch отсутствует.

Отдельный публичный boolean состояния Emergency не является частью архитектурного контракта: достаточно внутреннего состояния текущей Turret transport operation.

## Логирование и будущие hardware events

В v1 межмодульные runtime состояния публикуются типизированно (`CameraStatus`, `TurretState`, `DistanceResult` и т. п.), а диагностические сообщения идут в logging. Отдельная generic event-bus infrastructure заранее не вводится.

STM32 response сохраняет reserved `EVENTS` section как wire-format extension, но в v1 она пустая. Event types, buffering и overflow будут определены вместе с первым реальным hardware event.

## `ConfigUpdate`

```python
@dataclass(frozen=True)
class ConfigUpdate[T]:
    revision: int
    config: T
```

Правила:

- Config Manager — единственный источник актуальной общей конфигурации;
- один глобальный `revision` увеличивается после каждого принятого изменения;
- компонент получает полный immutable snapshot;
- межпотоковый `ConfigUpdate` — latest-only;
- устаревший revision игнорируется;
- компонент сам применяет утверждённую для поля policy: dynamic / pipeline restart / reconnect / app restart.

Конкретный thread-safe primitive остаётся вопросом реализации.
