# Конфигурация системы

Этот документ описывает общие правила `config.json` и основные параметры модулей. Calibration data хранится отдельно и не является обычной пользовательской конфигурацией.

## Общие правила

- `Config Manager` в Core — единственный источник актуальной общей конфигурации;
- файл хранения — `config.json`;
- первая версия schema: `schema-version = 1`;
- после загрузки настройки валидируются и преобразуются в typed config objects;
- изменения UI проходят через Core/Config Manager;
- рабочие модули не записывают общий файл напрямую;
- внутренние обновления передаются полным `ConfigUpdate[T]` с глобальным `revision`;
- `ConfigUpdate` — latest-only;
- владелец ресурса применяет заранее определённую policy: dynamic / pipeline restart / reconnect / application restart.

Контракт: [ConfigUpdate](./contracts.md#configupdate).

### Сохранение

Запись `config.json` должна быть атомарной:

```text
write temporary file
→ flush/close (и fsync там, где применимо)
→ atomic replace config.json
```

Повреждённый существующий JSON не должен молча перезаписываться defaults: Config Manager сообщает startup/config error и сохраняет исходный файл для диагностики.

Полностью отсутствующий `config.json` — startup/config error. Config Manager не создаёт файл автоматически и не запускает normal runtime на неявных defaults.

### Строгая validation policy v1

`config.json` schema v1 валидируется строго и атомарно до публикации typed snapshots.

```text
parse JSON
→ schema-version check
→ required/unknown-field check
→ type/value validation
→ cross-field validation
→ typed immutable snapshot
→ publish
```

Правила:

- `schema-version` обязателен, имеет тип integer (не `bool`) и в v1 равен ровно `1`;
- отсутствующий или неверного типа `schema-version` — invalid config;
- любое другое значение `schema-version` — unsupported schema error;
- автоматической migration для неизвестной schema нет; migration проектируется только после появления реальной schema v2;
- неизвестное поле на любом уровне schema — validation error, а не warning/ignore;
- все поля v1 обязательны, кроме тех, которые ниже явно отмечены optional;
- отсутствующий required field — validation error;
- optional field получает только документированный in-memory default; Config Manager не дописывает его в файл только из-за загрузки;
- неверный тип, значение вне допустимой области, `NaN`, `+Inf` или `-Inf` — validation error;
- `bool` не принимается как integer/number;
- invalid config не исправляется молча и не заменяется defaults;
- ошибка по возможности содержит точный path поля и причину;
- при invalid runtime update остаётся активным последний полностью валидный snapshot; partial apply/publication запрещены.

После того как schema v1 реализована и считается опубликованной, добавление/удаление/переименование persisted fields требует нового `schema-version`, если старый v1 parser не сможет строго принять новый файл.

## Calibration отдельно от `config.json`

Для первой реализации:

```text
calibration/
  overview.json
  stereo.json
```

Calibration содержит measured camera/stereo geometry и собственный `schema_version`. Для первой реализации `schema_version = 1`.

Calibration files также валидируются строго: malformed JSON, неизвестное поле, неверный тип/shape, non-finite matrix value или несовпадение `image_width/image_height` с source resolution дают calibration error. Config Manager/Vision не исправляют и не перезаписывают calibration автоматически.

Отсутствующая/невалидная calibration не обязана останавливать всё приложение, но соответствующий camera pipeline не считается ready и не публикует raw frame как fallback.

`config.json` не дублирует `K/D/R/T/P/Q`.

### Минимальная validation boundary calibration v1

Overview `calibration/overview.json`:

```text
schema_version = 1
image_width / image_height: integer > 0
K: 3x3 finite matrix
D: finite coefficient vector, non-empty
new_camera_matrix: 3x3 finite matrix
```

Stereo `calibration/stereo.json`:

```text
schema_version = 1
image_width / image_height: integer > 0
K_left / K_right: 3x3 finite matrices
D_left / D_right: finite coefficient vectors, non-empty
R / R1 / R2: 3x3 finite matrices
T: finite vector length 3
P1 / P2: 3x4 finite matrices
Q: 4x4 finite matrix
```

Точный допустимый размер distortion vectors должен соответствовать реально используемой OpenCV camera model и проверяется loader'ом Stage 2; он не должен молча truncate/pad coefficients. Maps в JSON не сохраняются.

## Структура верхнего уровня

```text
config
├── schema-version
├── vision
├── aiming
├── turret
└── ui
```

## Required и optional fields schema v1

Базовое правило v1: **все перечисленные ниже поля required**, если явно не указано обратное. Это сделано намеренно: hardware/control параметры не получают скрытых аппаратных defaults.

Единственные optional fields базовой schema v1:

```text
aiming.aim-points.overview.x-px
aiming.aim-points.overview.y-px
aiming.aim-points.stereo-left.x-px
aiming.aim-points.stereo-left.y-px
```

Для каждого из них отсутствие поля или JSON `null` означает center соответствующего working frame. Если значение задано, это integer `>= 0`; проверка попадания в фактическое разрешение выполняется относительно принятого `CameraModel`/working frame.

Все остальные поля в описанной ниже v1-схеме, включая booleans simulation/emulation и camera processing switches, должны присутствовать явно.

### Общие ограничения значений

- `address`, serial `port` path и `vision-processor-class` — непустые строки;
- UDP/TCP camera `port` — integer `1..65535`;
- buffer sizes — integer `>= 1`;
- timeout/delay fields — integer `> 0`, кроме `lead-time-ms >= 0` и `max-retries >= 0`;
- `manual-distance-m` — finite number `> 0`;
- PID `Kp/Ki/Kd` — finite number `>= 0`;
- `full-steps-per-revolution`, `microstep-divider` — integer `> 0`;
- `max-relative-move-deg`, max speed и acceleration — finite number `> 0`;
- `baudrate` — только значение из зафиксированного whitelist;
- enum-like strings принимают только явно перечисленные значения;
- booleans принимают только JSON `true/false`.

Hardware-specific upper bounds, которых пока нет в архитектуре, не придумываются Config Manager: их owner проверяет на своей границе (например STM32 проверяет свои допустимые пределы `SET_CONFIG`).

## Vision

### Cameras

Для:

```text
overview
stereo-left
stereo-right
```

основные поля:

```text
enabled: bool
address: str
port: int
rtp-enabled: bool
buffer-size: int
processing-enabled: bool          # per-camera master switch
vision-processor-class: str
```

`processing-enabled = false` полностью запрещает VisionProcessor для этой камеры, но pipeline продолжает публиковать corrected working frame через `VisionResult` с пустым `tracked_objects`.

### Processing scope main / preview

Для Overview + Stereo Left:

```text
vision.processing-scope = main-only | main-and-preview
```

Effective processing:

```text
processing-enabled(camera)
AND
(
    processing-scope == main-and-preview
    OR camera == main_camera
)
```

`Stereo Right` не подчиняется main/preview selection и используется по diagnostic/stereo policy.

В v1 processor-specific tuning не входит в `config.json`: detector/tracker constants принадлежат реализации выбранного `VisionProcessor`, не persistятся Config Manager и не показываются UI. Публичным config contract остаётся выбор `vision-processor-class`. Если позднее конкретный processor parameter потребуется менять per-camera/runtime, он добавляется в schema только вместе с явной validation/apply policy.

### Distance

```text
vision.distance.source = stereo | manual
vision.distance.manual-distance-m
vision.distance.distance-stale-timeout-ms
```

Для Stereo:

```text
right-frame-buffer-size
pair-timeout-ms
stereo-enabled
```

Точный `capture_id`/resync определяется перед полноценным Stereo distance.

### Camera stale / simulation

```text
vision.camera-stale-timeout-ms
vision.simulation-mode
```

Camera connection state и freshness разделены. Stale вычисляется consumer по `last_receive_timestamp_ns`.

Конкретная reconnect/backoff policy остаётся открытой.

## Aiming

Минимально:

```text
aiming.lead-time-ms
aiming.target-lost-timeout-ms
```

Aim points — pixels working frame:

```text
aiming.aim-points.overview.x-px
aiming.aim-points.overview.y-px
aiming.aim-points.stereo-left.x-px
aiming.aim-points.stereo-left.y-px
```

Если X/Y отсутствуют, используется center working frame.

`R_camera_to_turret` в первой реализации не используется.

## Turret

### Serial / HAL

```text
turret.serial.port
turret.serial.baudrate
turret.serial.response-timeout-ms
turret.serial.max-retries
turret.serial.inter-request-delay-ms
```

Рекомендованные начальные protocol values (поля остаются required в `config.json`):

```text
STM32 startup baudrate               = 9600
response-timeout-ms                  = 100
max-retries                          = 2
inter-request-delay-ms               = 2
```

`max-retries = 2` означает две повторные передачи после initial request, то есть максимум три attempts.

`baudrate` в `config.json` — желаемая рабочая скорость ПК. STM32 после hardware reset всегда стартует на 9600 и при необходимости переключается через `SET_BAUDRATE`.

Runtime update desired baud не выполняет hidden `MOTOR_OFF`: при confirmed motors OFF controlled transition может выполняться сразу; при motors ON/UNKNOWN worker сохраняет только freshest desired baud и применяет его после последующего confirmed `MOTOR_OFF` либо в safe recovery. Если затем запрошен `MOTOR_ON`, pending desired baud при confirmed OFF применяется первым.

Поддерживаемый первый whitelist:

```text
9600
19200
38400
57600
115200
```

### Axis mechanics on PC

Для каждой оси:

```text
invert
full-steps-per-revolution
microstep-divider
max-relative-move-deg
```

HAL вычисляет:

```text
effective_steps_per_revolution =
    full_steps_per_revolution * microstep_divider
```

`microstep-divider` и mechanical limit одной relative move на STM32 не передаются.

`max-relative-move-deg` проверяется на ПК до перевода degrees → steps.

`invert`, `full-steps-per-revolution`, `microstep-divider` и `max-relative-move-deg` в v1 считаются restart-only: изменение сохраняется в config, но не меняет уже работающую mechanical conversion или relative-move safety envelope до Turret/application restart. Это исключает safe-point semantics посреди active motion.

STM32 имеет отдельный compile-time/static sanity bound по `abs(delta_steps)` для защиты от аномального payload; он не является пользовательской настройкой и не дублирует `max-relative-move-deg`.

### PID Controller

Для каждой оси:

```text
pid-kp
pid-ki
pid-kd
```

I-term ограничивается `±max_speed` соответствующей оси. Отдельного `integral-limit` в прототипе нет.

Runtime apply policy принадлежит Turret Controller и применяется по осям независимо:

- изменение любого из `Kp/Ki/Kd` оси в TRACKING полностью сбрасывает PID state этой оси перед обработкой следующего нового `TrackingError`;
- первый sample после такого reset использует новые gains и остаётся P-only: `I=0`, `D=0`;
- изменение gains вне TRACKING не создаёт дополнительного действия: при следующем входе в TRACKING действует обычный reset boundary;
- уменьшение соответствующего `max-speed-*-deg-s` не сбрасывает PID целиком, но сразу clamp'ит сохранённый I-term в новый диапазон `±max_speed`;
- увеличение `max-speed-*-deg-s` сохраняет накопленный I-term без масштабирования;
- если один config revision меняет и gains, и output limit одной оси, gain-change reset имеет приоритет, поэтому I-term становится zero уже под новым limit;
- config update сам по себе не создаёт motion command и не меняет `control_mode`.

Core/Config Manager не посылает отдельный `PID_RESET`: внутреннее PID state и его reset/apply semantics остаются ответственностью Turret Controller.

### STM32 config

На уровне `config.json` параметры задаются в физических единицах ПК:

```text
max-speed-x-deg-s
max-speed-y-deg-s
acceleration-x-deg-s2
acceleration-y-deg-s2
velocity-watchdog-timeout-ms
```

HAL переводит их в `steps/s`, `steps/s²`, `ms` и передаёт одним `SET_CONFIG`.

В `SET_CONFIG` не входят:

```text
PID Kp/Ki/Kd
target-lost-timeout-ms
lead-time-ms
aim points
invert
full-steps-per-revolution
microstep-divider
max-relative-move-deg
serial port
desired baudrate
```

Желательный invariant:

```text
velocity-watchdog-timeout-ms < aiming.target-lost-timeout-ms
```

Конкретные timeout после первых измерений могут быть скорректированы.

Эти пять STM32-параметров применяются dynamic через атомарный полный `SET_CONFIG` snapshot даже во время движения. Control loop/ISR не должен видеть смесь полей разных snapshots.

Изменение `velocity-watchdog-timeout-ms` не считается новым `SET_VELOCITY` и не refresh'ит watchdog. Отсчёт сохраняется от последнего успешно принятого velocity setpoint; если новый timeout уже истёк, target velocity становится zero при ближайшей watchdog check.

### Simulation

```text
turret.emulate-stm32: bool
```

В v1 `emulate-stm32` — application-restart setting: runtime сохранение допустимо, но уже работающий Turret worker не переключает real ↔ fake transport до application restart.

### Backlash

Backlash compensation пока не входит в обязательную конфигурацию. Добавлять параметры следует только после механических измерений и отдельного решения о месте компенсации.

## UI

```text
ui.default-camera = overview | stereo-left
ui.show-fps
ui.show-stereo-right-diagnostics
```

`ui.default-camera` задаёт startup `main_camera`. `stereo-right` здесь недопустима.

При недоступности main camera автоматического switch на preview нет.

UI overlays не являются частью recorded working frame.

## Классификация применения runtime config

Базовая policy первой реализации:

| Настройка | Policy |
|---|---|
| PID `Kp/Ki/Kd` | dynamic; изменение gains reset'ит PID state только соответствующей оси перед следующим новым sample в TRACKING |
| `lead-time-ms` | dynamic |
| aim point | dynamic |
| `target-lost-timeout-ms` | dynamic, точная семантика для уже потерянной цели уточняется при реализации |
| `processing-scope` | dynamic |
| `processing-enabled` | dynamic |
| `vision-processor-class` | camera pipeline restart / new generation |
| camera source/address/port/GStreamer settings | camera pipeline restart / new generation |
| calibration file/content | camera pipeline restart / new generation |
| serial port | Turret reconnect |
| `response-timeout-ms`, `max-retries`, `inter-request-delay-ms` | Turret reconnect; новая session использует freshest accepted values, active in-flight transaction не перенастраивается |
| desired serial baudrate | controlled `SET_BAUDRATE` / reconnect path; при motors ON/UNKNOWN сохраняется latest desired baud и physical transition откладывается до confirmed motors OFF или recovery |
| max speed / acceleration / velocity watchdog | dynamic full STM32 `SET_CONFIG` snapshot; уменьшение max speed также clamp'ит application-side PID I-term соответствующей оси без полного reset |
| `invert`, steps/rev, microstep, `max-relative-move-deg` | restart-only; применяются только после Turret/application restart, не dynamic |
| `emulate-stm32` | application restart; runtime real ↔ fake transport switching в v1 отсутствует |
| UI-only display settings | dynamic |

Processor-specific tuning в v1 не является частью config schema и поэтому не имеет runtime apply policy. Смена `vision-processor-class` остаётся restart/new-generation boundary.

## Пример `config.json`

Все числа ниже демонстрационные и не являются аппаратными пределами. Пример показывает полную required schema v1; optional aim-point coordinates приведены как `null` для явности.

```json
{
  "schema-version": 1,
  "vision": {
    "processing-scope": "main-only",
    "cameras": {
      "overview": {
        "enabled": true,
        "address": "192.168.1.101",
        "port": 5001,
        "rtp-enabled": false,
        "buffer-size": 1,
        "processing-enabled": true,
        "vision-processor-class": "Legacy14VisionProcessor"
      },
      "stereo-left": {
        "enabled": true,
        "address": "192.168.1.102",
        "port": 5002,
        "rtp-enabled": false,
        "buffer-size": 1,
        "processing-enabled": true,
        "vision-processor-class": "Legacy14VisionProcessor"
      },
      "stereo-right": {
        "enabled": true,
        "address": "192.168.1.103",
        "port": 5003,
        "rtp-enabled": false,
        "buffer-size": 1,
        "processing-enabled": false,
        "vision-processor-class": "Legacy14VisionProcessor"
      }
    },
    "distance": {
      "source": "manual",
      "manual-distance-m": 100.0,
      "distance-stale-timeout-ms": 300,
      "stereo": {
        "stereo-enabled": false,
        "right-frame-buffer-size": 4,
        "pair-timeout-ms": 100
      }
    },
    "camera-stale-timeout-ms": 500,
    "simulation-mode": false
  },
  "aiming": {
    "lead-time-ms": 150,
    "target-lost-timeout-ms": 500,
    "aim-points": {
      "overview": {"x-px": null, "y-px": null},
      "stereo-left": {"x-px": null, "y-px": null}
    }
  },
  "turret": {
    "serial": {
      "port": "/dev/ttyUSB0",
      "baudrate": 115200,
      "response-timeout-ms": 100,
      "max-retries": 2,
      "inter-request-delay-ms": 2
    },
    "axes": {
      "x": {
        "invert": false,
        "full-steps-per-revolution": 2000,
        "microstep-divider": 16,
        "max-relative-move-deg": 45.0
      },
      "y": {
        "invert": false,
        "full-steps-per-revolution": 2000,
        "microstep-divider": 16,
        "max-relative-move-deg": 45.0
      }
    },
    "controller": {
      "pid-kp-x": 1.0,
      "pid-ki-x": 0.0,
      "pid-kd-x": 0.0,
      "pid-kp-y": 1.0,
      "pid-ki-y": 0.0,
      "pid-kd-y": 0.0
    },
    "stm32": {
      "max-speed-x-deg-s": 50.0,
      "max-speed-y-deg-s": 50.0,
      "acceleration-x-deg-s2": 100.0,
      "acceleration-y-deg-s2": 100.0,
      "velocity-watchdog-timeout-ms": 200
    },
    "emulate-stm32": false
  },
  "ui": {
    "default-camera": "overview",
    "show-fps": true,
    "show-stereo-right-diagnostics": false
  }
}
```

## Применение изменений

```text
UI
→ Core / Config Manager
→ validate
→ new typed snapshot
→ revision + 1
→ ConfigUpdate
→ component
```

STM32-dependent config синхронизируется Turret HAL по правилам [Turret](../modules/turret/index.md) и [Serial Protocol](./serial-protocol.md).

Concrete foundation primitives уже реализованы. Внутренний tuning `VisionProcessor` в v1 не является config schema; остальные runtime edge cases из `problems.md` остаются owner-specific/open. Базовая persistence/validation policy schema v1 закрыта.
