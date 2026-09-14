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

Поведение при полностью отсутствующем `config.json` допускается определить при реализации Config Manager (создание defaults либо явная startup error).

## Calibration отдельно от `config.json`

Для первой реализации:

```text
calibration/
  overview.json
  stereo.json
```

Calibration содержит measured camera/stereo geometry и собственный `schema_version`.

`config.json` не дублирует `K/D/R/T/P/Q`.

## Структура верхнего уровня

```text
config
├── schema-version
├── vision
├── aiming
├── turret
└── ui
```

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
processing-enabled: bool          # optional per-camera master switch
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

Processor-specific settings остаются отдельным открытым вопросом.

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

Стартовые protocol defaults:

```text
STM32 startup baudrate               = 9600
response-timeout-ms                  = 100
max-retries                          = 2
inter-request-delay-ms               = 2
```

`max-retries = 2` означает две повторные передачи после initial request, то есть максимум три attempts.

`baudrate` в `config.json` — желаемая рабочая скорость ПК. STM32 после hardware reset всегда стартует на 9600 и при необходимости переключается через `SET_BAUDRATE`.

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

`invert`, `full-steps-per-revolution` и `microstep-divider` в v1 считаются restart-only: изменение сохраняется в config, но не меняет уже работающую mechanical conversion до Turret/application restart. Это исключает safe-point semantics посреди active motion.

STM32 имеет отдельный compile-time/static sanity bound по `abs(delta_steps)` для защиты от аномального payload; он не является пользовательской настройкой и не дублирует `max-relative-move-deg`.

### PID Controller

Для каждой оси:

```text
pid-kp
pid-ki
pid-kd
```

I-term ограничивается `±max_speed` соответствующей оси. Отдельного `integral-limit` в прототипе нет.

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
| PID `Kp/Ki/Kd` | dynamic |
| `lead-time-ms` | dynamic |
| aim point | dynamic |
| `target-lost-timeout-ms` | dynamic, точная семантика для уже потерянной цели уточняется при реализации |
| `processing-scope` | dynamic |
| `processing-enabled` | dynamic |
| `vision-processor-class` | camera pipeline restart / new generation |
| camera source/address/port/GStreamer settings | camera pipeline restart / new generation |
| calibration file/content | camera pipeline restart / new generation |
| serial port | Turret reconnect |
| desired serial baudrate | controlled `SET_BAUDRATE` / reconnect path |
| max speed / acceleration / velocity watchdog | dynamic full STM32 `SET_CONFIG` snapshot |
| `invert`, steps/rev, microstep | restart-only; применяются только после Turret/application restart, не dynamic |
| UI-only display settings | dynamic |

Processor-specific settings классифицируются вместе со схемой конкретного `VisionProcessor`.

## Пример `config.json`

Все числа ниже демонстрационные и не являются аппаратными пределами.

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
        "vision-processor-class": "DefaultVisionProcessor"
      },
      "stereo-left": {
        "enabled": true,
        "address": "192.168.1.102",
        "port": 5002,
        "rtp-enabled": false,
        "buffer-size": 1,
        "processing-enabled": true,
        "vision-processor-class": "DefaultVisionProcessor"
      },
      "stereo-right": {
        "enabled": true,
        "address": "192.168.1.103",
        "port": 5003,
        "rtp-enabled": false,
        "buffer-size": 1,
        "processing-enabled": false,
        "vision-processor-class": "DefaultVisionProcessor"
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

Concrete thread-safe primitive, processor-specific schemas и часть edge-case semantics runtime changes остаются открытыми.
