# Протокол STM32

Этот документ определяет бинарный протокол `Turret HAL ↔ STM32` поверх последовательного byte stream.

Первая реализация использует обычный full-duplex UART. Будущий production RS485 half-duplex меняет только physical byte transport: framing, request/response, `REQUEST_ID`, retry, Emergency, baud и command semantics остаются теми же.

## Физический интерфейс

Стартовые UART settings после hardware reset STM32:

```text
baudrate = 9600
data bits = 8
parity = none
stop bits = 1
```

Желаемый рабочий baudrate можно менять runtime-командой `SET_BAUDRATE`, чтобы тестировать разные скорости без перепрошивки STM32.

Для первой реализации STM32 использует USART3 как обычный full-duplex UART; `DE/RE` и turnaround отсутствуют.

Управление `DE/RE` или auto-direction, termination/bias, общий reference/GND или galvanic isolation и точный turnaround delay относятся к будущей физической реализации RS485. Эти детали намеренно deferred за пределы обязательной первой реализации и не создают отдельную версию wire protocol.

## Модель обмена

Строго:

```text
1 request → 1 response
```

- один владелец UART в Turret HAL;
- одновременно максимум одна физическая transaction in flight;
- ordinary transaction завершается response либо исчерпанием обычного retry cycle до начала следующей ordinary transaction;
- STM32 не отправляет асинхронные packets;
- response содержит echo `REQUEST_ID + COMMAND_CODE` для однозначной correlation;
- wire-format сохраняет reserved `EVENTS` section для будущего расширения, но в v1 она всегда пустая.

HAL использует TX arbiter/latest state, а не общую FIFO motion-команд.

## Общий frame

Все многобайтовые значения — little-endian.

### Request

```text
Offset  Size  Field
0       2     START = AA 55
2       1     LENGTH
3       2     REQUEST_ID
5       1     COMMAND_CODE
6       N     PAYLOAD
6+N     2     CRC16
```

Минимальный request = 8 bytes.

### Response

```text
Offset  Size  Field
0       2     START = AA 55
2       1     LENGTH
3       2     REQUEST_ID
5       1     COMMAND_CODE
6       1     RESULT_CODE
7       N     EVENTS      # N=0 в v1
7+N     2     CRC16
```

Минимальный response = 9 bytes.

PC принимает response как ответ текущей transaction только если одновременно:

```text
response.REQUEST_ID == request.REQUEST_ID
AND
response.COMMAND_CODE == request.COMMAND_CODE
```

CRC-valid response с другой парой `REQUEST_ID + COMMAND_CODE` считается late/stale, текущую transaction не завершает, логируется и игнорируется.

`LENGTH` — полный размер frame, включая START, LENGTH и CRC; maximum = 255 bytes.

## CRC

Используется `CRC-16/MODBUS`:

```text
poly              = 0x8005
reflected poly    = 0xA001
init              = 0xFFFF
refin             = true
refout            = true
xorout            = 0x0000
```

CRC считается от `LENGTH` до последнего payload/event byte включительно; `AA 55` в CRC не входит.

CRC передаётся `CRC_LO`, затем `CRC_HI`.

При неверном CRC response не формируется.

## Command codes

| Code | Command |
|---:|---|
| `0x02` | `PING` |
| `0x10` | `SET_CONFIG` |
| `0x11` | `SET_BAUDRATE` |
| `0x20` | `MOVE_RELATIVE` |
| `0x21` | `SET_VELOCITY` |
| `0x30` | `EMERGENCY_STOP` |
| `0x31` | `MOTOR_ON` |
| `0x32` | `MOTOR_OFF` |

Диапазоны:

```text
0x0x protocol/service
0x1x configuration
0x2x normal motion
0x3x motor/safety
```

Отдельной transport-reset команды, `SET_MODE`, обычной `STOP` и `MOVE_COMPLETED` в v1 нет.

## Result codes

| Code | Result |
|---:|---|
| `0x00` | `OK` |
| `0x01` | `UNKNOWN_COMMAND` |
| `0x02` | `PARSE_ERROR` |
| `0x03` | `INVALID_ARGUMENT` |
| `0x04` | `INVALID_REQUEST_ID` |
| `0x05` | `MOTORS_OFF` |
| `0x06` | `NOT_CONFIGURED` |
| `0x07` | `INVALID_STATE` |
| `0x08` | `INTERNAL_ERROR` |

`RESULT_CODE` сообщает только результат обработки. Identity transaction определяется `REQUEST_ID + COMMAND_CODE`.

`PARSE_ERROR` означает: framing и CRC корректны, но payload не соответствует формату команды.

Framing/CRC/inter-byte errors не получают response.

## Reserved `EVENTS` section

В v1 `EVENTS` section всегда пустая. Обязательных STM32 asynchronous/hardware events сейчас нет, поэтому заранее не определяются event codes, FIFO capacity, overflow или backpressure.

Формат оставляет место для будущих событий вроде limit switch, homing, driver fault или encoder fault. При появлении первого реального producer/consumer requirement event schema и buffering будут спроектированы отдельно.

`PING` пока используется только как service/health request.

## Один общий `REQUEST_ID`

PC использует один циклический counter для **всех новых transactions**, ordinary и special:

```text
uint16
0000 ... FFFF → 0000
```

`0xFFFF` не зарезервирован и не имеет специальной семантики.

Каждая новая transaction получает следующий ID. Retry той же transaction использует тот же:

```text
REQUEST_ID
COMMAND_CODE
PAYLOAD
```

После начала новой transaction PC больше не возвращается к старой transaction. При одном UART owner и одной физической transaction in flight это делает поздние responses безопасно отбрасываемыми по `REQUEST_ID + COMMAND_CODE`.

### Exact retry cache STM32

STM32 хранит signature последней обработанной transaction и exact response:

```text
REQUEST_ID + COMMAND_CODE + PAYLOAD
```

При exact retry команда повторно не выполняется; возвращается cached response.

Если ordinary request использует тот же ID, что cached transaction, но command/payload отличаются, возвращается `INVALID_REQUEST_ID` и normal sequence не изменяется.

## Ordinary request sequence

STM32 хранит:

```text
expected_request_id: uint16
```

Для новой ordinary команды требуется:

```text
request_id == expected_request_id
```

Если ID другой — `INVALID_REQUEST_ID`, command не выполняется, sequence не меняется.

После корректно принятого нового ordinary request ID считается использованным даже если дальнейшая command validation вернула `UNKNOWN_COMMAND`, `PARSE_ERROR`, `INVALID_ARGUMENT`, `MOTORS_OFF`, `NOT_CONFIGURED`, `INVALID_STATE` или `INTERNAL_ERROR`:

```text
expected_request_id = request_id + 1 mod 65536
```

Response cache'ится для exact retry.

## `EMERGENCY_STOP` — special command и resync boundary

`EMERGENCY_STOP` является special **по `COMMAND_CODE`**, а не по значению `REQUEST_ID`.

Новая Emergency transaction получает обычный следующий PC `REQUEST_ID = N` и может быть обработана STM32 независимо от текущего `expected_request_id`.

При CRC-valid `EMERGENCY_STOP(ID=N)` STM32 в первую очередь приводит motion к безопасному состоянию:

```text
stop STEP generation immediately
clear relative target
clear velocity target
no acceleration limit
drivers remain enabled if they were enabled
```

Persistent emergency latch не используется.

После обработки Emergency normal sequence пересинхронизируется:

```text
expected_request_id = N + 1 mod 65536
```

Matching `OK` подтверждает PC ту же границу, поэтому его следующий новый request также использует `N+1`.

Payload в v1 должен быть пустым. Если frame CRC-valid и command распознан как Emergency, stop выполняется до payload validation; malformed payload может вернуть `PARSE_ERROR`, при этом PC остаётся в blocked/recovery state до успешной подтверждённой Emergency transaction.

### Retry Emergency

Emergency idempotent. При timeout выполняется bounded exact retry с теми же `REQUEST_ID`, `COMMAND_CODE` и payload.

Если retries исчерпаны:

```text
motion/control remains blocked
Turret transport → LOST / RECOVERING
```

PC не считает Emergency успешно выполненной и не разрешает обычное управление.

Дополнительная transport-reset transaction после successful Emergency не нужна: sequence уже синхронизирована на `N+1`.

## Порядок обработки request STM32

После framing/CRC:

```text
1. decode REQUEST_ID + COMMAND_CODE
2. exact retry check
3. if EMERGENCY_STOP → special safety/resync handling
4. ordinary REQUEST_ID validation
5. known COMMAND_CODE
6. parse payload
7. validate arguments
8. validate config / motor state
9. execute
10. advance ordinary sequence if applicable
11. cache and return response with echoed COMMAND_CODE
```

Если одновременно нарушено несколько ordinary условий, этот порядок определяет возвращаемую ошибку.

## `SET_CONFIG`

Payload:

```text
Offset  Type    Field
0       uint32  max_speed_x_steps_s
4       uint32  max_speed_y_steps_s
8       uint32  acceleration_x_steps_s2
12      uint32  acceleration_y_steps_s2
16      uint32  velocity_watchdog_timeout_ms
```

Payload = 20 bytes, full request = 28 bytes.

Минимальная validation:

```text
max_speed_x > 0
max_speed_y > 0
acceleration_x > 0
acceleration_y > 0
velocity_watchdog_timeout_ms > 0
```

Также проверяются firmware/hardware-supported ranges.

Возможные results:

```text
OK
INVALID_REQUEST_ID
PARSE_ERROR
INVALID_ARGUMENT
INTERNAL_ERROR
```

Эти пять полей полностью dynamic и могут применяться при relative motion, velocity control, остановке, motors ON или OFF.

Firmware атомарно заменяет **весь config snapshot**: control loop/ISR видит либо весь предыдущий, либо весь новый snapshot, но не смесь полей.

- уменьшение `max_speed` не создаёт мгновенный скачок: acceleration limiter плавно приводит скорость к новому limit;
- новое `acceleration` используется со следующего control update;
- relative planner использует актуальный snapshot на control updates, а не фиксированный профиль на весь move;
- изменение `velocity_watchdog_timeout_ms` не refresh'ит watchdog.

Watchdog продолжает отсчёт от последнего успешно принятого `SET_VELOCITY`. Если после уменьшения timeout уже прошедшее время больше нового значения, target velocity становится `(0,0)` при ближайшей watchdog check.

`OK` означает, что snapshot принят и применяется.

## `SET_BAUDRATE`

Payload:

```text
baudrate uint32
```

Whitelist v1:

```text
9600
19200
38400
57600
115200
```

Возможные results:

```text
OK
INVALID_REQUEST_ID
PARSE_ERROR
INVALID_ARGUMENT
INVALID_STATE
INTERNAL_ERROR
```

`INVALID_STATE` возвращается, если **сама STM32** видит motors не OFF. Runtime baud transition никогда не разрешён при включённых drivers.

Повторное задание текущего baudrate возвращает `OK`.

### Нормальная смена

```text
PC --old baud--> SET_BAUDRATE(new)
STM32 --old baud--> response OK
STM32 ждёт physical TX complete
STM32 переключается на new baud
PC после response + inter-request delay переключается на new baud
```

Первичный response передаётся на old baud.

### Неопределённый результат

Если response потерян, STM32 может быть уже на old или new baud. Exact retry использует тот же request/ID; PC проверяет кандидаты без дубликатов в порядке `new → old → 9600`, где startup `9600` добавляется только если ещё не проверялась. Cached response exact retry может прийти уже на текущем baud STM32.

Поскольку motors гарантированно OFF, потеря UART во время baud transition не создаёт продолжающегося физического движения.

Если обычный baud retry/recovery не восстановил однозначную transaction, transport переходит в LOST/RECOVERING. После обнаружения фактического baud обычный recovery начинается с `EMERGENCY_STOP`, который одновременно подтверждает связь и пересинхронизирует request sequence.

## `MOVE_RELATIVE`

Payload:

```text
Offset  Type   Field
0       int32  delta_x_steps
4       int32  delta_y_steps
```

Payload = 8 bytes, request = 16 bytes.

Semantics:

- `OK` означает только: новая relative target принята STM32;
- PC не отслеживает естественный момент завершения;
- новый `MOVE_RELATIVE` заменяет предыдущую relative target;
- delta отсчитывается от управляемого текущего положения при принятии команды;
- replacement не требует промежуточной остановки;
- planner использует current velocity, актуальные config limits и acceleration limiter;
- reversal проходит через zero согласно acceleration limiter;
- `(0,0)` — valid no-op.

STM32 дополнительно проверяет простой static firmware sanity bound:

```text
abs(delta_x_steps) <= MAX_RELATIVE_DELTA_X_STEPS
abs(delta_y_steps) <= MAX_RELATIVE_DELTA_Y_STEPS
```

Это защита от аномального payload, а не mechanical absolute limit и не runtime config. Превышение → `INVALID_ARGUMENT`.

Возможные results:

```text
OK
INVALID_REQUEST_ID
PARSE_ERROR
INVALID_ARGUMENT
NOT_CONFIGURED
MOTORS_OFF
INTERNAL_ERROR
```

`command_id`, `MOVE_COMPLETED` и lifecycle `Completed/Cancelled/Failed` отсутствуют.

При внезапной потере связи уже принятый и заранее ограниченный `MOVE_RELATIVE` разрешено завершить полностью. Отдельного communication watchdog для relative move нет.

## `SET_VELOCITY`

Payload:

```text
Offset  Type   Field
0       int32  velocity_x_steps_s
4       int32  velocity_y_steps_s
```

Любой успешно принятый `SET_VELOCITY(vx, vy)`:

- очищает current relative target;
- переводит firmware в velocity-control behaviour;
- заменяет velocity target;
- clamp'ится по configured max speed;
- меняет физическую скорость через acceleration limiter;
- refresh'ит velocity watchdog.

Watchdog expiry устанавливает target velocity `(0,0)`; замедление остаётся acceleration-limited.

`SET_VELOCITY(0,0)` — обычный velocity setpoint с нулевым target. Он **не** означает сам по себе StopMotion, PID reset, target clear или mode change.

Возможные results:

```text
OK
INVALID_REQUEST_ID
PARSE_ERROR
NOT_CONFIGURED
MOTORS_OFF
INTERNAL_ERROR
```

## `MOTOR_ON`

Payload отсутствует.

Возможные results:

```text
OK
INVALID_REQUEST_ID
PARSE_ERROR
NOT_CONFIGURED
INTERNAL_ERROR
```

Если drivers уже ON → `OK`.

`MOTOR_ON` только включает drivers. Старое relative/velocity motion не восстанавливается; hidden emergency re-arm отсутствует.

## `MOTOR_OFF`

Payload отсутствует.

Возможные results:

```text
OK
INVALID_REQUEST_ID
PARSE_ERROR
INTERNAL_ERROR
```

Если уже OFF → `OK`.

При выполнении:

```text
stop STEP generation immediately
disable drivers
clear velocity target
clear relative target
```

Acceleration limit не применяется. Последующий `MOTOR_ON` старое движение не восстанавливает.

## `PING`

Payload отсутствует.

`PING` — ordinary request и участвует в normal request sequence.

Возможные results:

```text
OK
INVALID_REQUEST_ID
PARSE_ERROR
```

В v1 `PING` используется для service/health проверки. Completion polling для `MOVE_RELATIVE` отсутствует.

## Unknown command

При valid expected ordinary ID и неизвестном `COMMAND_CODE` возвращается:

```text
UNKNOWN_COMMAND
```

Response echo'ит неизвестный command code. ID считается использованным, response cache'ится.

Если ordinary ID одновременно неправильный, приоритет имеет `INVALID_REQUEST_ID`.

## Parser и resynchronization STM32

Request parser:

```text
1. искать AA 55
2. мусор до START отбрасывать
3. прочитать LENGTH
4. LENGTH < 8 → invalid candidate
5. иначе ждать ровно LENGTH bytes
6. проверить CRC
7. valid → protocol parser
8. invalid → response отсутствует, resync
```

При invalid candidate/CRC failure отбрасывается только первый byte текущего candidate, затем поиск `AA 55` продолжается по оставшемуся buffer.

`AA 55` внутри payload не считается START, пока текущий frame собирается по валидному LENGTH.

PC response parser требует minimum response length 9 и проверяет `REQUEST_ID + COMMAND_CODE`.

## Inter-byte timeout STM32

```text
rx-inter-byte-timeout-ms = 20
```

Если во время незавершённого frame между bytes прошло больше 20 ms, candidate invalidируется без response, а накопленные bytes снова участвуют в поиске START.

Это inter-byte timeout, не total-frame timeout.

## PC timing / ordinary retry

Стартовые значения:

```text
serial-response-timeout-ms     = 100
serial-max-retries             = 2
serial-inter-request-delay-ms  = 2
```

`max-retries = 2` означает:

```text
1 initial + 2 exact retries = 3 transmissions
```

Для ordinary transaction timeout/invalid CRC приводит к exact retry того же request, кроме Emergency preemption ниже.

После matching response PC выдерживает inter-request delay перед следующей physical transaction.

Если ordinary retries исчерпаны, старая transport session больше не считается надёжной:

```text
connection → LOST / RECOVERING
ordinary traffic blocked
```

## Normal control boundary и committed in-flight request

`StopMotion`, normal mode transition и `MOTOR_OFF` очищают старый **неотправленный** `pending_motion`.

Если ordinary request уже физически отправлен и ждёт response, он считается committed:

```text
already-sent ordinary request
→ completes normal response/retry cycle
→ only then next normal control operation is transmitted
```

Это сохраняет однозначную sequence и не требует resync для обычного Stop/mode change. Цена — bounded transport latency normal Stop. Для срочной остановки используется `EMERGENCY_STOP`.

Если committed transaction исчерпала retries, ordinary traffic не продолжается; начинается общий recovery.

## Emergency preemption

Emergency — единственное исключение из полного ordinary retry cycle.

Если UART idle, `EMERGENCY_STOP` отправляется сразу.

Если ordinary request уже **физически передан**:

```text
invalidate unsent pending_motion
block new normal motion
не планировать новые retries ordinary transaction
wait current physical attempt response OR current timeout
then send EMERGENCY_STOP with next global REQUEST_ID
```

Второй request не передаётся до завершения текущей physical attempt. Это правило действует и на full-duplex UART первой реализации, и на будущий half-duplex RS485: protocol сохраняет one-request-in-flight boundary независимо от возможностей физического канала.

Если old ordinary response успел прийти — он обрабатывается штатно. Если текущая attempt timeout — её outcome остаётся неизвестным, но следующая подтверждённая Emergency уничтожает любое motion state и пересинхронизирует sequence независимо от старого expected ID.

## Connection loss / recovery

Transport session считается потерянной и Turret входит в auto-reconnect, если происходит хотя бы одно из событий:

- ordinary request исчерпал configured exact retries после timeout/invalid CRC/no valid matching response;
- serial I/O сообщает disconnect/device disappearance или другой I/O failure, из-за которого текущий physical exchange нельзя продолжить;
- Emergency transaction исчерпала bounded exact retries;
- bounded baud-detection/`SET_BAUDRATE` recovery attempt не смог подтвердить рабочую transport session.

CRC-valid matching response с command-level result сам по себе не означает потерю physical transport и не запускает reconnect только из-за result code. `INVALID_REQUEST_ID` является отдельной sequence-loss границей: normal traffic блокируется и выполняется Emergency-based resync/recovery.

После признанной потери связи Turret:

```text
block new motion actions
clear unsent pending_motion
Turret Controller resets PID
MotorState → UNKNOWN
applied STM32 config → unknown
```

Turret публикует connection loss через `TurretState`; Core в ответ очищает `selected_target` и инвалидирует `TrackingError`/target-dependent state. HAL сам application selection не хранит.

Уже принятый ограниченный `MOVE_RELATIVE` может закончиться во время разрыва связи. Velocity control остановится по `velocity_watchdog_timeout_ms`.

Recovery не пытается продолжить старую session с середины. Отсутствие STM32/serial device не является само по себе fatal `ERROR`: auto-reconnect продолжается без конечного лимита попыток, пока Turret worker не остановлен. Между неуспешными reconnect cycles используется capped exponential backoff:

```text
0.25 s → 0.5 s → 1 s → 2 s → 2 s → ...
```

Backoff сбрасывается после полного успешного выхода в `READY`. Ожидание backoff должно быть interruptible через cooperative stop, а не через неконтролируемый `sleep`.

### Обычный reconnect

После hardware reset STM32 baud = 9600. PC знает desired configured baud и last-known baud. Baud candidates проверяются без дубликатов с сохранением порядка:

```text
last-known baud → desired configured baud → 9600
```

После обнаружения физической связи recovery выполняется через Emergency как safety + sequence boundary:

```text
find physical baud / connection
→ EMERGENCY_STOP(ID=N) → OK
→ expected/next request ID = N+1
→ MOTOR_OFF → OK
→ SET_BAUDRATE(desired) при необходимости
→ SET_CONFIG(full current snapshot) → OK
→ READY
```

Если baud transition выполняется после `MOTOR_OFF`, motors остаются OFF, как требует firmware.

Автоматического `MOTOR_ON` после reconnect/recovery нет. Пользователь включает motors заново явным действием.

PC-side applied `control_mode` Controller может сохранить, но selection, PID input и physical motion не replay'ятся. Поэтому возможен безопасный `TRACKING + no target + motors OFF`.

### Recovery после неопределённого `SET_BAUDRATE`

Так как `SET_BAUDRATE` разрешён только при фактическом motors OFF, после lost/uncertain response сначала определяется фактический baud. Candidates проверяются без дубликатов в порядке:

```text
new baud → old baud → 9600
```

`9600` добавляется только если он ещё не был проверен как `new` или `old`. После нахождения рабочей скорости выполняется та же sequence:

```text
EMERGENCY_STOP → MOTOR_OFF → SET_CONFIG → READY
```

Если desired baud отличается от найденного, controlled `SET_BAUDRATE` выполняется снова при motors OFF.

## Motor state на PC

```python
class MotorState(Enum):
    UNKNOWN = "unknown"
    OFF = "off"
    ON = "on"
```

`MotorState` означает последнее подтверждённое фактическое состояние drivers.

После disconnect → `UNKNOWN`.

Желаемое действие `MOTOR_ON/OFF` может быть pending, но после полноценного reconnect/recovery baseline всегда OFF; старое `ON` автоматически не replay'ится.

## Единицы

На верхних уровнях:

```text
degrees
deg/s
deg/s²
```

HAL переводит в:

```text
steps
steps/s
steps/s²
```

Один protocol `step` = один фактический STEP pulse driver.

На PC:

```text
effective_steps_per_revolution =
    full_steps_per_revolution * microstep_divider
```

`invert`, `full_steps_per_revolution` и `microstep_divider` — PC-side mechanical conversion settings и в STM32 `SET_CONFIG` не входят.

## Firmware bounds, определяемые на реализации

Compile-time/firmware constants:

```text
MAX_SUPPORTED_STEP_RATE
MAX_SUPPORTED_ACCELERATION
MAX_SUPPORTED_WATCHDOG_TIMEOUT
MAX_RELATIVE_DELTA_X_STEPS
MAX_RELATIVE_DELTA_Y_STEPS
```

выбираются после реализации STEP generator и проверки timer/driver/mechanics.

Требуемый turnaround delay будущего RS485 transceiver определяется только при отдельной RS485 hardware migration и не меняет binary framing. Для первой UART реализации такого delay нет.
