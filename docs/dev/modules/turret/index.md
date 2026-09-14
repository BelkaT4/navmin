# Turret

`Turret` преобразует application motion intent в команды STM32 и владеет PID, serial transport, motor state и recovery.

![Диаграмма модуля Turret](../../diagrams/turret-diagram.png)

## Состав

```text
Turret worker
├── Turret Controller
│   ├── applied control_mode
│   ├── PID
│   └── pending_motion latest-only
└── Turret HAL
    ├── unit conversion / invert
    ├── STM32 config sync
    ├── motor state
    ├── TX arbiter
    └── Serial Protocol / recovery
```

Turret worker — единственный owner UART/RS485.

Cross-thread ingress не выполняет UART I/O. В частности Emergency request из Core/UI/application thread является только thread-safe signal: если ordinary physical attempt уже идёт, signal подавляет его будущие retries, но actual `EMERGENCY_STOP` frame передаёт только Turret worker после response/timeout текущей попытки; при idle UART worker делает Emergency следующей physical operation.

## Authoritative applied mode

Turret Controller владеет applied:

```text
RELATIVE
TRACKING
```

Startup mode — `RELATIVE`.

Authoritative applied value публикуется через:

```text
TurretState.control_mode
```

Core/UI могут иметь pending user request на смену режима, но не держат отдельную applied-копию.

## Один latest-only `pending_motion`

Вместо двух независимых slots используется один:

```text
pending_motion:
    MoveRelativeCommand
    | AxisVelocitySetpoint
    | None
```

Допустимый тип определяется applied mode:

```text
RELATIVE → MoveRelativeCommand
TRACKING → AxisVelocitySetpoint
```

Новый ещё не отправленный motion intent заменяет старый pending intent.

Control boundaries инвалидируют только **неотправленный** `pending_motion`. Уже физически отправленный ordinary request считается committed и завершает обычный response/retry cycle перед следующей normal control operation.

`motion_generation` не используется: один Turret worker, один UART owner, один in-flight request и один latest slot уже задают достаточную сериализацию.

## Mode transition

### `RELATIVE → TRACKING`

Переход начинается с invalidation старого unsent `pending_motion`.

Если motors ON:

```text
wait committed ordinary transaction if any
→ SET_VELOCITY(0,0) → OK
→ PID reset
→ apply TRACKING
→ publish TurretState(TRACKING)
```

`SET_VELOCITY(0,0)` здесь используется как ownership boundary firmware: после `OK` STM32 больше не следует старой relative target. `OK` не означает, что физическая скорость уже равна zero.

Если motors OFF, physical motion невозможен и zero handshake не нужен; Controller reset'ит PID, применяет TRACKING и публикует `TurretState`.

Пока transition не завершён, несовместимые новые normal motion actions не принимаются. Отдельный публичный `mode_transition_pending` не нужен: достаточно внутреннего состояния текущей Controller operation.

### `TRACKING → RELATIVE`

Core очищает target/TrackingError сразу до request mode change.

Controller:

```text
invalidate unsent pending_motion
→ PID reset
→ if motors ON: SET_VELOCITY(0,0) → OK
→ apply RELATIVE
→ publish TurretState(RELATIVE)
```

Если transaction/recovery не завершилась, applied mode может остаться TRACKING, но target уже отсутствует и automatic tracking не возобновляется.

## RELATIVE

Core передаёт:

```text
MoveRelativeCommand(delta_x_deg, delta_y_deg)
```

Click-to-move разрешён только при confirmed `TurretState.control_mode == RELATIVE`.

После успешной camera/session validation и вычисления угла camera swap/restart/stale не отменяет уже сформированный command.

PC проверяет:

```text
abs(delta_x_deg) <= max-relative-move-x-deg
abs(delta_y_deg) <= max-relative-move-y-deg
```

до degrees → steps conversion.

STM32 дополнительно применяет static firmware sanity bound по `abs(delta_steps)`; это не mechanical absolute limit и не runtime config.

После `MOVE_RELATIVE → OK` PC знает только, что STM32 приняла relative target. Естественное завершение не отслеживается.

При внезапной потере связи уже принятый ограниченный `MOVE_RELATIVE` может завершиться полностью. Отдельного communication watchdog для relative move нет.

## TRACKING

Controller получает `TrackingError` через latest-state с monotonic revision:

```text
new TrackingError revision
→ PID exactly once
→ clamp ±max_speed
→ AxisVelocitySetpoint
→ pending_motion
→ HAL
→ SET_VELOCITY
```

Controller хранит `last_processed_revision` и не обрабатывает sample повторно.

`MOVE_RELATIVE` в TRACKING отклоняется выше Turret и не является manual override в v1.

При временной потере selected target новых TrackingError не появляется. STM32 velocity watchdog ведёт target velocity к zero, если новые velocity setpoints не возобновились.

## PID ownership

PID полностью принадлежит Turret Controller. Core не посылает отдельный `PID_RESET`.

Controller reset'ит PID при domain/control boundaries, которые видны на Turret side:

- вход/выход TRACKING;
- смена `TargetRef` в новом `TrackingError`;
- final loss/deselect/StopMotion;
- Emergency Stop;
- `MOTOR_OFF`;
- UART reconnect/recovery;
- gap между TrackingError больше `velocity-watchdog-timeout-ms` перед следующим sample.

Первый sample после reset: P-only, `I=0`, `D=0`.

PID работает в градусах по X/Y, отдельные `Kp/Ki/Kd`, real monotonic `dt`, conditional anti-windup и I-term clamp по `±max_speed`.

Runtime PID config применяется по осям независимо:

- если во время TRACKING меняется любой `Kp/Ki/Kd` оси, Controller полностью reset'ит PID state этой оси перед следующим новым `TrackingError`;
- первый sample после такого reset использует новые gains и остаётся P-only (`I=0`, `D=0`);
- gain change вне TRACKING не требует отдельного действия: следующий вход в TRACKING уже является reset boundary;
- уменьшение `max_speed` оси не делает full PID reset, но сразу clamp'ит сохранённый I-term в новый `±max_speed`;
- увеличение `max_speed` сохраняет текущий I-term без масштабирования;
- при одновременном изменении gains и `max_speed` gain-change reset имеет приоритет;
- config update не создаёт motion command и не меняет `control_mode` сам по себе.

Внешнего `PID_RESET` contract нет: это внутреннее состояние Turret Controller.

D-filter добавляется только после измерений, если нужен.

## `SET_VELOCITY(0,0)`

Zero setpoint не является специальной application-командой.

```text
SET_VELOCITY(0,0)
```

означает только target velocity `(0,0)`. PID может штатно выдавать его, когда target находится в aim point.

StopMotion, PID reset, target clear и mode change определяются явным control state/action, а не численным значением setpoint.

## StopMotion

Старой serial-команды `STOP` нет.

### TRACKING

Core предварительно очищает selection/TrackingError. Turret Controller:

```text
invalidate unsent pending_motion
PID reset
wait committed ordinary request if any
if motors ON: SET_VELOCITY(0,0)
```

Applied mode остаётся TRACKING.

### RELATIVE

```text
invalidate unsent pending_motion
wait committed ordinary request if any
if motors ON: SET_VELOCITY(0,0)
```

Успешный zero setpoint очищает firmware relative target и переводит STM32 в velocity-control behaviour; скорость плавно приходит к zero по acceleration limiter.

Normal Stop может ждать ordinary retry cycle. Для срочной остановки существует Emergency.

## Emergency Stop

`EMERGENCY_STOP`:

- имеет отдельный highest-priority transport path;
- немедленно прекращает STEP generation без acceleration limit;
- очищает STM32 relative target и velocity target;
- drivers остаются enabled;
- persistent emergency latch отсутствует;
- является request-sequence resync boundary.

При user Emergency:

```text
invalidate unsent pending_motion
block new normal motion actions
PID reset
```

Если UART idle, Emergency отправляется сразу.

Если ordinary request уже физически отправлен:

```text
не планировать дальнейшие ordinary retries
wait response OR timeout current physical attempt
→ send EMERGENCY_STOP with next global REQUEST_ID
```

Matching successful Emergency `ID=N` заставляет STM32 установить:

```text
expected_request_id = N + 1 mod 65536
```

и PC продолжает с тем же `N+1`. Поэтому отдельная transport-reset transaction после successful Emergency не нужна.

Emergency exact retries используют тот же `REQUEST_ID`. Если retries исчерпаны, motion/control остаётся blocked, transport переходит в LOST/RECOVERING.

Отдельный публичный Emergency flag не является архитектурным state; internal in-flight operation Turret transport достаточно.

## Turret HAL и TX arbiter

HAL отвечает за:

- UART/RS485 lifecycle;
- binary framing/CRC;
- one physical request in flight;
- общий cyclic `REQUEST_ID` для ordinary и Emergency transactions;
- response correlation по `REQUEST_ID + COMMAND_CODE`;
- exact retry/cache contract;
- Emergency preemption;
- baudrate switching/recovery;
- unit conversion;
- axis inversion;
- STM32 config sync;
- motor state;
- reconnect/recovery.

Нет общей FIFO физических operations.

Концептуально arbiter имеет:

```text
current in-flight transaction | None
pending normal control operation | None
pending motor state | None
latest pending STM32 config | None
pending_motion | None
```

Emergency может preempt ordinary **retry plan**, но не передаёт второй packet поверх уже незавершённой half-duplex physical attempt.

## Normal committed request

Если normal control boundary (`StopMotion`, mode transition, `MOTOR_OFF`) возникает, когда ordinary request уже физически in flight, этот request считается committed:

```text
finish normal response/retry cycle
→ then execute boundary operation
```

Старый unsent `pending_motion` при этом инвалидируется сразу.

Если committed transaction исчерпала retries, normal traffic не продолжается; запускается transport recovery.

## Преобразование единиц

PC-side для каждой оси:

```text
invert
full_steps_per_revolution
microstep_divider
max-relative-move-deg
```

HAL вычисляет:

```text
effective_steps_per_revolution =
    full_steps_per_revolution * microstep_divider
```

и преобразует degrees / deg/s / deg/s² в steps / steps/s / steps/s².

Один protocol `step` = один STEP pulse driver.

`invert`, steps/rev и microstep divider не передаются STM32.

В v1 `invert`, steps/rev, microstep divider и `max-relative-move-deg` restart-only: сохранённое изменение применяется только после Turret/application restart, не посреди active motion. Уже работающий HAL продолжает использовать startup snapshot, включая прежний relative-move safety envelope.

## STM32 config

Полный `SET_CONFIG`:

```text
max_speed_x_steps_s
max_speed_y_steps_s
acceleration_x_steps_s2
acceleration_y_steps_s2
velocity_watchdog_timeout_ms
```

HAL переводит application values в STEP units.

STM32 проверяет hardware-supported ranges и применяет весь snapshot атомарно. Эти пять параметров dynamic даже во время motion.

- снижение max speed отрабатывается через acceleration limiter;
- новое acceleration используется со следующего control update;
- relative planner использует актуальный snapshot;
- изменение watchdog timeout не refresh'ит watchdog.

HAL хранит applied/pending STM32 config state. Pending config — latest-only. Если во время in-flight exchange появился более новый snapshot, после завершения transaction отправляется freshest pending snapshot.

Serial `response-timeout-ms`, `max-retries` и `inter-request-delay-ms` принадлежат concrete `TurretSession`; runtime изменение требует Turret reconnect и построения новой session boundary с freshest values. `emulate-stm32` в v1 application-restart-only и reconnect не переключает real ↔ fake transport.

Desired baud latest-only: config update не выполняет hidden `MOTOR_OFF`. При confirmed motors OFF worker может выполнить controlled `SET_BAUDRATE`; при motors ON/UNKNOWN переход откладывается. После confirmed `MOTOR_OFF` pending desired baud применяется, а перед новым `MOTOR_ON` при motors OFF baud transition завершается первым.

## Config exchange во время TRACKING

Во время физического non-motion request новые velocity packets не могут передаваться из-за one-in-flight rule. Controller сохраняет только freshest `pending_motion`; после response отправляется актуальный setpoint.

Если пауза достаточно длинная, firmware watchdog безопасно ведёт velocity target к zero.

## Motor state

```python
class MotorState(Enum):
    UNKNOWN = "unknown"
    OFF = "off"
    ON = "on"
```

`motor_state` — последнее подтверждённое фактическое состояние STM32 drivers.

После disconnect → `UNKNOWN`.

`MOTOR_OFF` — normal control boundary:

- unsent `pending_motion` invalidated;
- PID reset;
- уже in-flight ordinary request committed и заканчивает normal retry cycle;
- затем `MOTOR_OFF` немедленно прекращает STEP generation, очищает firmware motion и disables drivers.

`MOTOR_ON` включает drivers, но не восстанавливает старое motion и не выполняет hidden emergency re-arm.

Auto `MOTOR_ON` после reconnect отсутствует.

## Serial recovery

Подробный transport/wire contract: [Протокол STM32](../../architecture/serial-protocol.md).

Auto-reconnect запускается после исчерпания ordinary retries, physical serial I/O/disconnect failure, исчерпания Emergency retries или неуспешного bounded baud-recovery attempt. CRC-valid matching command-level error сам по себе не означает transport loss; `INVALID_REQUEST_ID` требует Emergency-based sequence resync.

При признанной потере transport session на PC:

```text
pending_motion → None
selected tracking input уже invalidated Core при соответствующем failure handling
PID reset
MotorState → UNKNOWN
applied STM32 config → unknown
new normal motion blocked
```

Уже принятый limited `MOVE_RELATIVE` может закончиться; velocity control останавливается по watchdog.

Отсутствие STM32/serial device не считается fatal `ERROR`: worker продолжает reconnect cycles без конечного лимита попыток. Между cycles используется interruptible capped backoff `0.25 → 0.5 → 1 → 2 → 2 ... s`, который сбрасывается после `READY`.

Обычный baud search проверяет без дубликатов:

```text
last-known → desired → 9600
```

После обнаружения physical connection/baud:

```text
EMERGENCY_STOP(ID=N) → OK
→ sequence synced to N+1
→ MOTOR_OFF → OK
→ SET_BAUDRATE(desired) при необходимости
→ SET_CONFIG(full snapshot) → OK
→ READY
```

После recovery motors OFF. `MOTOR_ON` — только новым user action.

Applied `control_mode` можно сохранить через reconnect, но selected target/PID input/physical motion не replay'ятся.

### `SET_BAUDRATE`

Runtime command сохраняется, потому что требуется тестировать разные UART baudrate без перепрошивки STM32.

STM32 разрешает `SET_BAUDRATE` только при фактическом motors OFF.

При lost/uncertain response рабочий baud ищется без дубликатов в порядке `new → old → 9600`, после чего запускается обычный Emergency-based recovery. Не нужны отдельные baud generation/ID state.

## Worker lifecycle

Turret worker создаётся и останавливается application orchestration; отдельный Supervisor для него не вводится.

Все ожидания serial transport должны быть bounded или cooperative-cancellable. Reconnect/backoff ожидается через общий `StopToken.wait(timeout)`, а не через неконтролируемый `sleep`, чтобы `request_stop()` быстро прерывал ожидание. Concrete serial adapter не должен держать worker в бесконечном blocking read.

Shutdown boundary:

```text
request_stop()
→ interrupt/cancel bounded serial wait or backoff
→ join(bounded timeout)
→ verify !is_alive()
```

Точный numeric join timeout остаётся внутренним implementation tuning, а не новым `config.json` field. Если worker не завершился в bounded timeout, это явная shutdown error: её нужно залогировать и нельзя молча считать shutdown успешным.

## Ограничения механики

Первая конструкция не имеет:

- limit switches;
- encoders;
- absolute position feedback;
- надёжного homing.

Физические упоры существуют, но попадание в них считается нештатным. STEP count не подтверждает фактическое перемещение.

Нет достоверных software absolute limits или public absolute position. Blind hard-stop homing не используется.

PC ограничивает одну relative move в градусах. STM32 имеет только консервативный static bound по `abs(delta_steps)` как sanity check payload.

## Публикуемое состояние и logging

Turret публикует latest-only `TurretState` с:

- connection state;
- confirmed `MotorState`;
- authoritative applied `control_mode`;
- подтверждёнными speed/acceleration limits.

Если `HAL.applied_stm32_config is None`, четыре поля speed/acceleration в `TurretState` равны `None`; desired config не подменяет подтверждённое applied state. После successful `SET_CONFIG` публикуется именно подтверждённый snapshot, а после transport loss поля снова становятся `None`.

Transient diagnostics/errors в v1 идут в стандартный Python logging; обязательные runtime состояния имеют typed contracts. Generic event bus заранее не вводится.

Для Turret в INFO достаточно lifecycle/connection/recovery/baud-transition boundaries и значимых failures. `REQUEST_ID`, command code, retry/attempt/candidate details должны быть доступны для диагностики на более подробном уровне, но high-rate PID samples и обычные velocity setpoints не логируются на INFO.

`QueueHandler/QueueListener` в v1 заранее не вводятся: сначала используется существующий logging bootstrap, а queue-based logging добавляется только при измеренной проблеме blocking/contention.

Wire `EVENTS` section STM32 зарезервирована, но пуста в v1. Hardware event queue проектируется только вместе с первым реальным hardware event.

## Эмуляция

Физический STM32 должен быть заменяем эмулятором через тот же HAL interface.

## Что ещё не определено

- Qt notification/coalescing details для доставки latest-state в main thread;
- hardware upper limits step rate/acceleration/watchdog/static relative delta;
- D-filter после измерений;
- future position feedback/homing.

Полный список: [Открытые вопросы](../../architecture/problems.md).
