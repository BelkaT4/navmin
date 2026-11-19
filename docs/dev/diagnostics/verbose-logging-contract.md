# VERBOSE diagnostic logging contract v1

Статус: **согласованный контракт Stage 1; реализация следует отдельными этапами**.

Этот документ определяет подробное диагностическое логирование NavMin для поиска
ошибок по всей цепочке

```text
camera → Vision/VT11 → Aiming → estimator → PID/FF/prediction
       → Turret → UART → STM32 → STEP/DIR → physical motion / IMU
```

и для последующего offline replay/сопоставления с моделью. Контракт опирается на
существующий launcher/session-artifact path и не вводит новый runtime state или
универсальную event-bus инфраструктуру.

## 1. Режимы логирования

В launcher остаются **ровно два** пользовательских режима:

```text
NORMAL
VERBOSE
```

Третий режим не добавляется.

### NORMAL

- сохраняет текущее лёгкое поведение;
- console и session file используют обычные lifecycle/connection/error сообщения;
- новые high-rate события из этого документа **не пишутся**;
- существующий `runtime.log` остаётся bounded rotating file с текущим лимитом
  `10 MiB` и `5` backup-файлами.

### VERBOSE

- включает все обычные сообщения NORMAL;
- session file принимает DEBUG;
- включает все подробные события из этого контракта;
- console остаётся на INFO, чтобы high-rate DEBUG не заполнял терминал;
- используется один логический поток `runtime.log` + его rotated backup-файлы;
- лимит одного файла: `100 MiB`;
- число backup-файлов: `10`;
- суммарный верхний предел одного session log stream — примерно `1.1 GiB`
  (активный файл + десять backup-файлов).

## 2. Один `runtime.log`, а не набор event-файлов

Все PC-side диагностические события пишутся в один логический `runtime.log`.
Отдельные `vision.log`, `tracking.log`, `uart.log`, `imu.log` не создаются.

Существующие session artifacts сохраняются отдельно по своему назначению:

```text
session/
├── runtime.log
├── runtime.log.1
├── ...
├── runtime.log.10
├── manifest.json
├── preflight.json
└── inputs/
```

Видео и `frames.jsonl`, когда пользователь включает запись видео, остаются
артефактами recording subsystem и не копируются в `runtime.log`.

`manifest.json` сохраняет существующее поле `mode` (`normal | diagnostic`) как
режим приложения и отдельно поле `logging_mode` (`normal | verbose`). После
штатного VERBOSE shutdown в manifest добавляется `logging_integrity` с теми же
ключевыми counters, что и итоговый `RUN_INTEGRITY`.

Для offline-восстановления chronological stream rotated-файлы читаются от самого
старого backup к активному файлу:

```text
runtime.log.10 → ... → runtime.log.2 → runtime.log.1 → runtime.log
```

## 3. Формат строки

Существующий префикс Python logging сохраняется:

```text
%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s
```

Он уже даёт:

```text
wall-clock timestamp | level | thread | logger/module | message
```

Новый подробный message имеет стабильный формат:

```text
EVENT_NAME {compact JSON object}
```

Пример:

```text
2026-11-17 15:30:12,345 DEBUG turret-worker navmin.turret.controller: TRACK_CONTROL {"schema_version":1,"run_id":"navmin-normal-...","control_id":812,...}
```

Правила:

- `EVENT_NAME` — ASCII `UPPER_SNAKE_CASE` и находится сразу после `:`;
- JSON всегда занимает ровно одну физическую строку;
- JSON кодируется UTF-8;
- `NaN`, `Infinity` и `-Infinity` запрещены;
- недоступное значение обозначается JSON `null`, а не магическим `0`, `-1` или
  пустой строкой;
- имена полей с физическими величинами содержат единицу в suffix, например
  `_ns`, `_ms`, `_px`, `_deg`, `_deg_s`, `_steps`, `_hz`, `_bytes`, `_ppm`;
- enum/state/reason значения — стабильные lowercase strings;
- произвольный человеческий текст не используется вместо machine-readable
  `reason`/`state`, когда такое поле определено контрактом.

Logger name (`%(name)s`) остаётся owner/source module. `EVENT_NAME` определяет
конкретную границу/событие. Поэтому источник можно определить уже по началу строки,
а отдельный event легко выбирать обычным `grep`.

## 4. Версионирование

Каждое detailed event содержит:

```text
schema_version = 1
```

Совместимые изменения в рамках v1:

- добавление нового event type;
- добавление нового nullable/optional поля;
- добавление нового enum value, если consumer обязан переносить неизвестное
  значение как unknown и не делать safety-решений по логам.

Несовместимые изменения требуют повышения `schema_version`:

- смена смысла существующего поля;
- смена единицы;
- смена системы координат;
- удаление обязательного поля;
- изменение identity/time semantics.

Offline reader должен игнорировать неизвестные поля, но не должен молча угадывать
смысл неизвестной версии схемы.

## 5. Общие идентификаторы

### `run_id`

Новый отдельный UUID не вводится. `run_id` равен существующему
`SessionArtifacts.session_id` и `manifest.json.session_id`.

Он обязателен во всех новых detailed events.

### Common payload

Каждое новое detailed event содержит как минимум:

```text
schema_version
run_id
source_instance
sequence
event_monotonic_ns
```

`source_instance` — стабильная строка экземпляра producer внутри run, например
`camera:stereo_left:generation:3`, `uart:connection_epoch:2`, `turret:controller`,
`imu:<sensor_id>`.

### `sequence`

`sequence` — монотонный integer внутри одного `source_instance`.

Он используется для обнаружения пропусков и локального порядка одного producer,
но не заменяет timestamps и не объявляется глобальным causal order.

### Camera frame key

Кадр однозначно идентифицируется тройкой:

```text
camera + generation + frame_id
```

где `camera` принимает:

```text
overview | stereo_left | stereo_right
```

Не создаётся параллельный глобальный frame ID.

### `control_id`

Монотонный ID одного PC-side control decision. Назначается при формировании
решения, которое может быть отправлено, заменено, отменено или отброшено.

Один и тот же `control_id` должен проходить через:

```text
TRACK_CONTROL
→ CONTROL_DECISION
→ UART_TX / UART_RX
→ STM32_COMMAND_*   (когда MCU telemetry станет доступна)
```

### UART identity

UART transaction связывается полями:

```text
connection_epoch
request_id
attempt
control_id (nullable для service/non-control requests)
```

`connection_epoch` увеличивается после каждого нового serial connection/recovery.
`request_id` сохраняется ровно в wire-значении протокола и может оборачиваться.
Retry того же logical request не получает новый `control_id`.

### MCU/IMU identity

Когда firmware/IMU transport предоставляют соответствующие идентификаторы:

```text
boot_id
imu_boot_id
sensor_id
```

Если целевая версия их ещё не предоставляет, поле равно `null`; PC не синтезирует
фиктивный boot ID и не выдаёт его за аппаратный факт.

## 6. Время

Для каждого PC-side detailed event обязательно:

```text
event_monotonic_ns
```

Это время фиксации события по monotonic clock PC.

Boundary-specific timestamps сохраняются отдельно, например:

```text
capture_timestamp_ns
receive_timestamp_ns
processing_start_ns
processing_end_ns
control_timestamp_ns
tx_monotonic_ns
rx_monotonic_ns
source_timestamp_ns
```

Каждая внешняя отметка имеет явный `time_domain`/`time_basis`. Допустимые значения
расширяются по мере появления реальных источников, например:

```text
pc_monotonic
receive_time
camera_pts
camera_capture
stm32_clock
imu_sample_clock
external_reference_clock
```

Правила:

- порядок строк в файле **не считается** причинным порядком;
- latency вычисляется только между отметками в общей шкале либо после валидного
  clock mapping;
- при неизвестной асимметрии PC↔STM32 RTT не превращается в «точную»
  одностороннюю задержку;
- reset/wrap/reconnect времени создаёт явную clock/session boundary;
- повторное использование старого кадра не создаёт новый frame key: новый
  `control_id` ссылается на тот же camera/generation/frame_id.

## 7. Системы координат

### Image coordinates

```text
+X → вправо
+Y → вниз
unit → px
```

### Logical Turret coordinates

После Aiming:

```text
+X / AZ → вправо
+Y / EL → вверх
angle → deg
angular rate → deg/s
```

Эта система находится **до** hardware `axis.invert`.

### Physical/protocol direction

Payload после `axis.invert` логируется отдельно от logical setpoint. Нельзя
подменять logical и physical значения одним полем.

STEP counters STM32, когда появятся, должны иметь явно документированный знак по
physical axis/direction и сопоставление с logical NavMin axis.

## 8. Политика отсутствующих/невалидных данных

Для diagnostic analysis различаются как минимум:

```text
no_frame
frame_without_detection
no_selected_target
insufficient_samples
span_too_short
fit_rms_exceeded
rate_implausible
platform_rate_unavailable
stale_observation
session_mismatch
transport_failure
unknown
```

Nullable value и validity/reason — разные понятия. Если вычисление было попытано и
отклонено, сохраняется `valid=false` + конкретный `reason`. Если величина к этому
событию неприменима, сохраняется `null` без придумывания failure reason.

## 9. Async file logging в VERBOSE

High-rate VERBOSE session-file logging не должно выполнять disk I/O из Vision,
Core, Turret, UART или IMU producer thread.

Для VERBOSE используется:

```text
producer threads
    ↓ non-blocking enqueue
bounded logging queue
    ↓
dedicated logging writer/listener thread
    ↓
one RotatingFileHandler → runtime.log
```

Это transport только для logging и **не** generic runtime event bus.

Начальная ёмкость очереди:

```text
8192 LogRecord
```

Это internal implementation constant, а не новое поле пользовательского config.
Фактическая capacity записывается в `LOGGING_SESSION_START`.

### Overflow

Producer никогда не ждёт свободного места в logging queue.

При переполнении:

1. diagnostic record разрешено потерять;
2. управление/vision/UART не блокируются;
3. счётчики потерь увеличиваются как минимум по `level` и `event_name`;
4. после восстановления writer публикует агрегированный
   `LOGGING_DROP_SUMMARY`, если это возможно;
5. итоговые counters сохраняются в session manifest/integrity summary;
6. потерянные записи не маскируются как успешная полная трасса.

### Shutdown

При штатном завершении writer получает bounded drain window `2 s`.
Если очередь не успела полностью выгрузиться, число оставшихся/потерянных records
фиксируется в итоговой диагностике. Shutdown не зависает бессрочно из-за логов.

## 10. Безопасность и чувствительные данные

`runtime.log` не должен копировать credentials из camera URI или query string.

Для source/config evidence используются:

- redacted endpoint identity в логе;
- существующие effective inputs/session artifacts;
- hashes/IDs calibration/config profile там, где это достаточно.

Detailed logging не меняет runtime ownership и не является источником команд.
Логи никогда не парсятся обратно для принятия safety/control решений online.

# 11. Event catalog

Ниже определён полный каталог v1. В Stage 2 реализуется PC-side часть; MCU/IMU и
external-reference events реализуются последующими этапами, но их identity и поля
фиксируются здесь заранее, чтобы не менять формат задним числом.

## 11.1 `LOGGING_SESSION_START`

**Когда:** один раз после создания session logging transport в VERBOSE.

**Owner/logger:** launcher / `navmin.logging_setup`.

**Поля:**

| Поле | Тип | Единица | Nullable | Смысл |
|---|---|---:|:---:|---|
| `schema_version` | int | — | no | `1` |
| `run_id` | string | — | no | session ID |
| `sequence` | int | — | no | sequence logging owner |
| `event_monotonic_ns` | int | ns | no | boundary time |
| `logging_mode` | string | — | no | `verbose` |
| `file_level` | string | — | no | `debug` |
| `console_level` | string | — | no | `info` |
| `queue_capacity_records` | int | records | no | initial `8192` |
| `rotation_max_bytes` | int | bytes | no | `104857600` |
| `rotation_backup_count` | int | files | no | `10` |
| `format_version` | int | — | no | line-format version `1` |

## 11.2 `LOGGING_DROP_SUMMARY`

**Когда:** после overflow recovery и обязательно при завершении, если были drops.

| Поле | Тип | Единица | Nullable | Смысл |
|---|---|---:|:---:|---|
| common identity/time | — | — | no | общие поля |
| `dropped_total` | int | records | no | суммарно |
| `dropped_by_level` | object | records | no | DEBUG/INFO/... |
| `dropped_by_event` | object | records | no | event name → count |
| `queue_high_watermark` | int | records | no | максимум очереди |
| `queue_capacity_records` | int | records | no | capacity |
| `undrained_on_shutdown` | int | records | no | остаток при timeout |

## 11.3 `CONFIG_SNAPSHOT`

**Когда:** один раз после effective input loading; повторно только после restart новой
application runtime внутри того же процесса, если такой path появится.

**Не дублирует secrets/full URI.** Полный effective config остаётся в `inputs/`.

Минимальные поля:

```text
config_schema_version
config_sha256                     nullable for synthetic/in-memory config
overview_calibration_sha256       nullable for synthetic/in-memory calibration
stereo_calibration_sha256         nullable for synthetic/in-memory calibration
cameras                           object keyed by camera role
tracking                          object
turret                            object
serial                            object
imu                               object nullable
```

`cameras[role]` содержит как минимум backend, working resolution,
`processing_enabled`, processor class/profile и redacted endpoint identity.

`tracking` содержит как минимум PID `kp/ki/kd` per logical axis, FF enabled/gains,
prediction enabled/horizon, estimator window/min-span/min/max samples/RMS/rate
limit, platform-motion source и nominal/effective Stereo Left HFOV/fx/fy.

`turret` содержит max velocity, acceleration, watchdog timeout и axis mechanical
conversion/invert snapshot. `serial` содержит redacted port identity и effective
baud. `imu` содержит source/profile/sensor identity, когда они доступны.

Все numerical fields используют единицы в имени.

## 11.4 `CONFIG_CHANGE`

**Когда:** на каждое runtime изменение tracking/control/diagnostic параметра.

| Поле | Тип | Nullable | Смысл |
|---|---|:---:|---|
| common fields | — | no | identity/time |
| `revision` | int | no | monotonic runtime tuning revision |
| `owner` | string | no | subsystem |
| `parameter` | string | no | stable path/name |
| `old_value` | JSON scalar/object | yes | предыдущий effective value |
| `new_value` | JSON scalar/object | yes | новый effective value |
| `apply_result` | string | no | `applied/queued/rejected/restart_required` |
| `reason` | string | yes | reject/defer reason |

HFOV override обязательно попадает сюда с nominal/effective `fx/fy/HFOV`.

## 11.5 `OPERATOR_ACTION`

**Когда:** действие оператора изменяет tracking/control intent.

Примеры `action`:

```text
target_select
target_clear
camera_swap
control_mode_request
motor_on
motor_off
stop_motion
emergency
move_relative
tracking_tuning_apply
hfov_override_apply
```

Поля:

```text
action
action_id
camera                         nullable
generation                     nullable
frame_id                       nullable
image_x_px                     nullable
image_y_px                     nullable
requested                      object nullable
accepted                       bool
reason                         nullable
resulting_target_id             nullable
resulting_control_mode          nullable
```

Новый event не заменяет существующие operational INFO messages; он включается как
structured DEBUG trace только в VERBOSE.

# 12. Video / camera path

## 12.1 `FRAME_INGRESS`

**Когда:** каждый decoded frame, до correction/vision.

Поля:

```text
camera
generation
frame_id
source_frame_id                nullable
capture_timestamp_ns           nullable
capture_time_domain            nullable
pts_ns                         nullable
receive_timestamp_ns
source_resolution_width_px
source_resolution_height_px
payload_bytes                  nullable
source_drop_counter            nullable
transport_state
```

Сам BGR/JPEG payload в log не помещается.

## 12.2 `FRAME_PROCESS`

**Когда:** каждый кадр после завершения соответствующего processing boundary.

Поля:

```text
camera/generation/frame_id
processing_start_ns
processing_end_ns
processing_duration_ms
correction_applied
processor_class
processor_profile
processing_enabled
result_delivered
result_drop_reason             nullable
output_resolution_width_px
output_resolution_height_px
```

Если pipeline имеет несколько измеримых stages, дополнительно разрешены:

```text
correction_duration_ms
vision_duration_ms
publish_duration_ms
```

## 12.3 `FRAME_DROP`

**Когда:** наблюдаемая очередь/consumer/source явно обнаружил потерю или reject.

Поля:

```text
camera/generation/frame_id      nullable, если неизвестен конкретный frame
boundary
reason
count
queue_depth                    nullable
queue_capacity                 nullable
last_good_frame_id             nullable
next_good_frame_id             nullable
```

Нельзя приписывать source loss, если известен только gap downstream frame IDs.

# 13. Detector / tracker

## 13.1 `VISION_TRACKING_FRAME`

**Когда:** каждый обработанный кадр камеры с включённым tracking processor.

Поля:

```text
camera/generation/frame_id
track_state
track_id                       nullable
selected_target_id              nullable
candidate_count
candidates                     array
selected_center_x_px           nullable
selected_center_y_px           nullable
selected_bbox                  nullable
search_region                  nullable
full_frame_search
best_candidate                 nullable
confidence/quality fields      nullable when backend has no such metric
transition                     nullable
transition_reason              nullable
no_detection_reason            nullable
```

Каждый candidate, если backend предоставляет данные, хранит как минимум:

```text
bbox
center_x_px
center_y_px
score/quality                  nullable
accepted                       bool
reject_reason                  nullable
```

Кадр без обнаружения отличается от отсутствующего кадра: первый создаёт
`VISION_TRACKING_FRAME` с `candidate_count=0`, второй — нет и при известной потере
отражается через `FRAME_DROP`.

# 14. Aiming

## 14.1 `AIMING_SAMPLE`

**Когда:** каждое преобразование выбранного observation в logical angular error.

Поля:

```text
camera/generation/frame_id
target_id
observation_monotonic_ns
observation_age_ms
camera_model_source            calibration | tracking_hfov_override
nominal_hfov_deg
effective_hfov_deg
fx_px
fy_px
cx_px
cy_px
aim_x_px
aim_y_px
target_x_px
target_y_px
error_x_px
error_y_px
error_x_deg                    logical +X/right
error_y_deg                    logical +Y/up
valid
invalid_reason                 nullable
```

# 15. Tracking estimator / platform model

## 15.1 `ESTIMATOR_SAMPLE`

**Когда:** каждый новый selected-target observation, независимо от `rate_valid`.

Общие поля:

```text
camera/generation/frame_id
target_id
estimator_revision
window_ms
min_span_ms
min_samples
max_samples
max_fit_rms_deg
max_target_rate_deg_s
platform_motion_source
```

По каждой logical axis `x/y`:

```text
sample_count
actual_span_ms
relative_rate_deg_s            nullable
platform_rate_deg_s            nullable
target_rate_deg_s              nullable
fit_rms_deg                    nullable
rate_valid
invalid_reason                 nullable
prediction_offset_deg
```

`invalid_reason` использует стабильный enum и обязательно отличает как минимум:

```text
insufficient_samples
span_too_short
fit_rms_exceeded
relative_rate_unavailable
platform_rate_unavailable
target_rate_implausible
observation_stale
target_changed
estimator_reset
unknown
```

## 15.2 `PLATFORM_MODEL_SAMPLE`

**Когда:** platform-motion model изменяет/оценивает effective motion, а также на
каждый control sample, если это нужно для однозначного replay.

По каждой оси:

```text
logical_command_target_deg_s
logical_effective_velocity_deg_s
physical_payload_velocity_units
configured_acceleration_deg_s2
effective_acceleration_deg_s2
model_dt_ms
anchor_request_id              nullable
anchor_tx_monotonic_ns         nullable
anchor_ack_monotonic_ns        nullable
valid
invalid_reason                 nullable
```

Это событие должно позволять проверить, не расходится ли software command model с
фактическим исполнением STM32/IMU.

# 16. PC control loop

## 16.1 `TRACK_CONTROL`

**Когда:** каждый PID/control step, включая step, где команда не будет отправлена.

Identity/time:

```text
control_id
camera/generation/frame_id
target_id
control_timestamp_ns
dt_ms
observation_age_ms
rate_valid_x
rate_valid_y
```

По каждой logical axis:

```text
real_error_deg
prediction_offset_deg
p_error_deg                    real + prediction
p_term_deg_s
i_term_deg_s
d_term_deg_s
ff_term_deg_s
ff_gain
target_rate_deg_s              nullable
i_state_deg_s
integrator_updated             bool
integrator_reset               bool
integrator_reset_reason        nullable
```

Сумма/ограничение:

```text
raw_velocity_x_deg_s
raw_velocity_y_deg_s
limited_velocity_x_deg_s
limited_velocity_y_deg_s
clamp_x_active
clamp_y_active
anti_windup_x_active
anti_windup_y_active
max_velocity_deg_s
```

## 16.2 `CONTROL_DECISION`

**Когда:** после control step, когда решение отправляется, заменяется, отменяется
или отклоняется до transport.

Поля:

```text
control_id
decision                    send | replace | cancel | reject | no_command
command_type                nullable
logical_velocity_x_deg_s    nullable
logical_velocity_y_deg_s    nullable
replaced_control_id         nullable
reason                      nullable
pending_revision            nullable
```

Это событие обязательно отличает «контроллер рассчитал setpoint» от «этот setpoint
действительно дошёл до UART boundary».

# 17. UART PC boundary

## 17.1 `UART_CONNECTION`

**Когда:** open/ready/lost/reconnect/baud transition/close.

```text
connection_epoch
state
port_redacted
baud
reason                       nullable
retry_index                  nullable
```

## 17.2 `UART_TX`

**Когда:** непосредственно на transport boundary фактической попытки записи.

```text
connection_epoch
request_id
attempt
control_id                   nullable
command_type
logical_command              nullable
protocol_payload
payload_bytes
tx_monotonic_ns
```

`protocol_payload` — значения после unit conversion / integer quantization /
`axis.invert`, то есть ровно то, что сериализуется в wire request, без секретов.

## 17.3 `UART_RX`

**Когда:** полный response принят/валидирован либо transaction завершилась ошибкой.

```text
connection_epoch
request_id
attempt
control_id                   nullable
command_type
rx_monotonic_ns
response_code                nullable
response_payload             nullable
crc_valid                    nullable
transaction_result
round_trip_ms                nullable
error_reason                 nullable
```

`transaction_result` различает:

```text
ok
protocol_error
crc_error
timeout
io_error
request_mismatch
connection_lost
```

## 17.4 `UART_RETRY`

**Когда:** принято решение повторить тот же logical request.

```text
connection_epoch
request_id
control_id                   nullable
completed_attempt
next_attempt
reason
backoff_ms                   nullable
```

# 18. STM32 execution events

Эти события являются контрактом будущей telemetry реализации. Они не считаются
доступными, пока целевая firmware реально их не публикует.

## 18.1 `STM32_COMMAND_RECEIVED`

```text
boot_id
mcu_sequence
mcu_timestamp                raw MCU clock value
mcu_time_unit
request_id
command_type
payload
```

## 18.2 `STM32_COMMAND_APPLIED`

```text
boot_id
mcu_sequence
mcu_timestamp
request_id
control_id                   nullable/correlated PC-side
result                       applied | rejected
reject_reason                nullable
target_velocity_x_protocol
target_velocity_y_protocol
motor_enabled
control_mode
watchdog_state
```

## 18.3 `STM32_MOTION_SNAPSHOT`

**Частота:** только измеренно допустимая для целевого UART/MCU budget; не из STEP ISR.

```text
boot_id
mcu_sequence
mcu_timestamp
request_id_anchor            nullable
velocity_generator_x
velocity_generator_y
dir_x
dir_y
step_count_x_signed
step_count_y_signed
motor_enabled
control_mode
watchdog_age
limit_x_state                nullable
limit_y_state                nullable
faults
```

## 18.4 `STM32_EVENT`

Для редких boundaries:

```text
motor_on
motor_off
watchdog_stop
emergency
limit
fault
reset
boot
config_applied
```

Поля содержат `boot_id`, MCU timestamp, event code, axis/value/reason при наличии.

# 19. IMU

## 19.1 `IMU_SAMPLE`

**Когда:** каждый реально полученный sample в VERBOSE, если measured load позволяет.
Если transport присылает batch, сохраняется individual sample time либо явный
метод reconstruction.

```text
sensor_id
imu_boot_id
imu_sequence
sample_timestamp
sample_time_domain
receive_monotonic_ns
gyro_raw_x
gyro_raw_y
gyro_raw_z
gyro_range_dps
accel_raw_x
accel_raw_y
accel_raw_z
accel_range_g
temperature_c                 nullable
fifo_state                    nullable
saturated_axes
corrected_gyro_x_deg_s
corrected_gyro_y_deg_s
corrected_gyro_z_deg_s
orientation_roll_deg          nullable
orientation_pitch_deg         nullable
orientation_yaw_deg           nullable
profile_id                    nullable
valid
invalid_reason                nullable
```

## 19.2 `IMU_STATE`

**Когда:** startup/reconnect/profile change/drop/error/state transition.

```text
sensor_id
imu_boot_id                   nullable
state
configured_rate_hz
filter_profile
mounting_profile_id           nullable
dropped_samples_total
last_error                    nullable
```

# 20. External reference

## 20.1 `REFERENCE_SAMPLE`

Опциональное событие для output encoder, external camera или calibrated landmark.

```text
reference_id
source_type
source_sequence
source_timestamp
source_time_domain
receive_monotonic_ns
az_deg                        nullable
el_deg                        nullable
az_uncertainty_deg            nullable
el_uncertainty_deg            nullable
line_of_sight_az_deg          nullable
line_of_sight_el_deg          nullable
voltage_v                     nullable
current_a                     nullable
motor_angle_deg               nullable
```

Отсутствие внешнего эталона не заполняется данными IMU «как будто это reference».

# 21. Clock quality

## 21.1 `CLOCK_SNAPSHOT`

**Когда:** на session/time-basis transition, при state change и примерно `1 Hz` для
активных mapping-ов.

```text
time_source
source_instance
state
sample_count
span_ns
offset_ns                     nullable
drift_ppm                     nullable
rms_ns                        nullable
valid_from_source_timestamp   nullable
valid_to_source_timestamp     nullable
reset_count
wrap_count
uncertainty_ns                nullable
reason                        nullable
```

# 22. Performance / recording health

## 22.1 `PERF_SNAPSHOT`

**Когда:** примерно `1 Hz` в VERBOSE.

```text
process_cpu_percent           nullable
process_rss_bytes             nullable
thread_timing_summary         nullable
camera_fps                    object
vision_processing_p50_ms      nullable
vision_processing_p95_ms      nullable
vision_processing_p99_ms      nullable
control_dt_p50_ms             nullable
control_dt_p95_ms             nullable
control_dt_p99_ms             nullable
uart_rtt_p50_ms               nullable
uart_rtt_p95_ms               nullable
logging_queue_depth
logging_queue_high_watermark
recording_drop_count
pipeline_drop_counters        object
```

Если библиотека/платформа не предоставляет некоторую process metric без нового
тяжёлого dependency, поле остаётся `null`; это не причина добавлять dependency
только ради Stage 2.

## 22.2 `RUN_INTEGRITY`

**Когда:** один раз при orderly shutdown; best-effort при runtime failure.

```text
run_id
completed
logging_dropped_total
logging_undrained_total
frame_drop_counters
recording_drop_counters
uart_error_counters
uart_retry_counters
imu_drop_counters
imu_error_counters
max_observed_queue_depths
last frame keys per camera
last control_id
last request_id per connection epoch
notes/reasons for unavailable evidence
```

Итоговые ключевые counters также переносятся в `manifest.json`, чтобы integrity
можно было увидеть без парсинга `runtime.log`.

# 23. Частота событий

| Event | VERBOSE cadence |
|---|---|
| `LOGGING_SESSION_START` | 1/run |
| `CONFIG_SNAPSHOT` | 1 effective runtime start |
| `CONFIG_CHANGE` | every change |
| `OPERATOR_ACTION` | every relevant action |
| `FRAME_INGRESS` | every decoded frame |
| `FRAME_PROCESS` | every processed frame |
| `FRAME_DROP` | every observed loss/reject boundary, aggregatable only when individual IDs unavailable |
| `VISION_TRACKING_FRAME` | every vision-processed frame |
| `AIMING_SAMPLE` | every selected-target aiming calculation |
| `ESTIMATOR_SAMPLE` | every selected-target observation |
| `PLATFORM_MODEL_SAMPLE` | every control-relevant model update/sample |
| `TRACK_CONTROL` | every control step, no sampling/throttling |
| `CONTROL_DECISION` | every control decision boundary |
| `UART_TX` | every actual attempt |
| `UART_RX` | every terminal receive/error |
| `UART_RETRY` | every retry decision |
| `UART_CONNECTION` | every connection/baud boundary |
| `STM32_COMMAND_*` | every event when telemetry exists |
| `STM32_MOTION_SNAPSHOT` | measured feasible periodic rate |
| `IMU_SAMPLE` | every received sample if load permits; otherwise transport-level bounded policy must be explicit |
| `IMU_STATE` | every state/error/drop boundary |
| `REFERENCE_SAMPLE` | every received reference sample |
| `CLOCK_SNAPSHOT` | state transition + ~1 Hz |
| `PERF_SNAPSHOT` | ~1 Hz |
| `LOGGING_DROP_SUMMARY` | after recovery + end if needed |
| `RUN_INTEGRITY` | 1/run |

# 24. What VERBOSE must let us answer

После реализации PC-side событий один runtime trace должен позволять без догадок
ответить:

1. Какой конкретный кадр вызвал данный control step?
2. Сколько лет было observation в момент control и UART TX?
3. Какой pixel error превратился в какой angular error и с каким effective HFOV?
4. Почему `rate_valid` true/false?
5. Из каких `relative_rate + platform_rate` получен `target_rate`?
6. Какие P/I/D/FF/prediction составляющие дали raw setpoint?
7. Где и почему сработал clamp/anti-windup/reset?
8. Был ли setpoint отправлен, заменён или отменён?
9. Какой integer/inverted payload реально ушёл в UART?
10. Когда и с каким результатом STM32 ответила?
11. После MCU telemetry: когда команда была применена и сколько STEP реально выдано?
12. После IMU/reference instrumentation: совпадает ли command model с фактическим движением?
13. Есть ли gap в evidence из-за logging/recording/queue drop?

# 25. Реализационные этапы после этого контракта

Этот документ фиксирует схему целиком, но реализация делится на независимые
checkpoint-ы:

```text
Stage 2  PC-side VERBOSE infrastructure + camera/vision/aiming/estimator/control/UART
Stage 3  STM32 execution telemetry + PC correlation
Stage 4  IMU + clocks/external-reference integration
Stage 5  integrity/replay/export/performance validation
```

До прохождения hardware validation нельзя утверждать, что MCU/IMU timestamps,
STEP counters или physical motion подтверждены на аппаратуре.
