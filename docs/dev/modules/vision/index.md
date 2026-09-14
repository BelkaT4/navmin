# Vision

`Vision` отвечает за получение видеопотоков трёх камер, исправление геометрии, формирование tracked objects и определение дальности до выбранного объекта Stereo Left.

![Диаграмма модуля Vision](../../diagrams/vision-diagram.png)

## Камеры и потоки

Фиксированные роли:

```text
overview
stereo-left
stereo-right
```

В Python:

```text
overview
stereo_left
stereo_right
```

Каждый camera pipeline работает в отдельном прикладном потоке. Отдельного `Camera Manager` нет.

```text
Overview worker
Stereo Left worker + Stereo / Distance Provider
Stereo Right worker
```

Vision хранит постоянный camera registry с `current_generation[camera]`.

## Working frame

Публичный `FramePacket.image` всегда содержит кадр с исправленной геометрией.

```text
Overview:
receive/decode
→ undistort
→ FramePacket
→ VisionProcessor
→ VisionResult

Stereo Left / Right:
receive/decode
→ rectify
→ FramePacket
→ VisionProcessor
→ VisionResult
```

Raw frame может существовать внутри pipeline, но наружу как обычный `FramePacket` не публикуется.

Все публичные pixel coordinates относятся к working frame:

- `BBox`;
- velocity tracked objects;
- aim point;
- lead point;
- UI click.

Geometric correction выполняется до `VisionProcessor`, потому что одна и та же geometry нужна Vision, UI, Aiming, stereo и recording.

## Calibration и `CameraModel`

Calibration хранится отдельно от `config.json`:

```text
calibration/
  overview.json
  stereo.json
```

### Overview

Минимально:

```text
schema_version
image_width
image_height
K
D
new_camera_matrix
```

Working frame сохраняет исходное разрешение. Автоматического crop/resize в первой реализации нет.

### Stereo

`stereo.json` — единый атомарный файл пары:

```text
schema_version
image_width
image_height
K_left / D_left
K_right / D_right
R / T
R1 / R2
P1 / P2
Q
```

Stereo Left working frame соответствует `P1`, Stereo Right — `P2`.

Rectification/undistortion maps строятся при старте и живут в памяти; в calibration JSON они не хранятся.

Если calibration отсутствует, повреждена или image size не совпадает, pipeline не считается ready и не публикует raw image как fallback.

На основе calibration создаётся immutable `CameraModel` working frame.

## Camera generation и barrier

При каждом новом запуске/restart pipeline:

```text
current_generation[camera] += 1
frame_id = 0
```

До публикации любых camera-derived данных новой generation Vision публикует:

```text
CameraSessionStarted(camera, generation, camera_model)
```

Это ordered/barrier event, а не latest-state notification. Для каждого consumer гарантируется:

```text
CameraSessionStarted generation=N
→ consumer принимает session
→ только затем VisionResult generation=N может считаться валидным
```

Generation защищает от запоздалых результатов старого worker и от повторного использования `track_id` после restart.

При restart одновременно очищаются локальные latest/buffer state и stereo pairing state соответствующей camera.

## `CameraSessionGate`

В main thread Core и UI используют общий `CameraSessionGate` с:

```text
accepted_generation[camera]
camera_model[camera]
```

Vision может доставлять `VisionResult` напрямую UI для низкой задержки, но UI обязан проверять generation через тот же gate, что и Core.

## `FramePacket`

Video Source / camera pipeline:

- получает и декодирует stream;
- выполняет geometric correction;
- назначает `generation` и `frame_id`;
- фиксирует `receive_timestamp_ns`;
- получает/формирует `capture_id` для stereo;
- гарантирует безопасное владение памятью;
- публикует read-only `FramePacket`.

Полный контракт: [FramePacket](../../architecture/contracts.md#framepacket).

## `VisionProcessor`

Выбирается настройкой:

```text
vision-processor-class
```

Внутренняя реализация не фиксируется:

```text
Detector → Tracker
integrated tracker
другой алгоритм
```

Публичный результат обязан содержать `TrackedObject`:

- устойчивый `track_id` внутри generation;
- `bbox`;
- velocity центра bbox;
- `age_frames`.

Если processing для камеры выключен effective policy, pipeline всё равно публикует working `VisionResult` с пустым `tracked_objects`.

Если processing медленнее camera source, backlog старых кадров не создаётся: после текущей обработки берётся freshest available frame.

## Processing scope main / preview

Обычный UI показывает Overview + Stereo Left. Processing scope:

```text
main-only
main-and-preview
```

При `main-only` VisionProcessor активен для текущей `main_camera`; preview продолжает публиковать working frames без tracked objects.

При `main-and-preview` processing работает для обеих отображаемых камер. BBox могут отображаться на preview, но target selection всё равно разрешён только на main image.

Per-camera `processing-enabled`, если сохраняется, является дополнительным master switch; effective processing определяется совместно с `processing-scope`.

Stereo Right processing используется только для diagnostics и не участвует в user selection.

## `VisionResult`

```text
FramePacket
+ tuple[TrackedObject]
+ processing_time_ns
```

UI отображает именно `VisionResult.frame` вместе с его bbox.

Межпотоковая семантика `VisionResult` — latest-only per camera. Qt notification не должна превращать latest-state в backlog.

## Overview

Overview используется:

- для main/preview UI;
- VisionProcessor согласно processing scope;
- user selection, только когда Overview является main camera;
- Aiming.

Working frame — undistorted.

## Stereo Left

Stereo Left используется:

- для main/preview UI;
- VisionProcessor согласно processing scope;
- user selection, только когда Stereo Left является main camera;
- Aiming;
- как ведущий stream Distance Provider.

Working frame — rectified.

## Stereo Right

Stereo Right прежде всего нужен для Stereo Distance. Он может иметь VisionProcessor и diagnostic display.

Пользовательского `TargetRef` для Stereo Right нет.

## Stereo pairing

Stereo Left/Right аппаратно синхронизированы.

Pairing должен учитывать:

```text
capture_id
left generation
right generation
```

Кадры разных generations нельзя объединять даже при одинаковом `capture_id`.

Stereo Right использует небольшой bounded buffer по `capture_id`; это pairing buffer, а не очередь последовательной обработки старых кадров.

Точный механизм генерации `capture_id`, resync, wraparound и pair timeout остаётся открытым до включения полноценного Stereo distance.

## Stereo / Distance Provider

Distance Provider работает в Stereo Left thread, но остаётся отдельным class.

Источники:

```text
stereo
manual
```

Distance вычисляется только в TRACKING, если текущий `selected_target` относится к Stereo Left и остаётся валиден по `camera + generation + track_id`. В RELATIVE selection отсутствует.

### Stereo

Успешный result содержит:

- `source = STEREO`;
- `source_frame_id`;
- `capture_id`;
- `measured_timestamp_ns`.

Неуспешный расчёт не публикует invalid-object. Последний успешный result живёт до `distance-stale-timeout-ms`; точный owner invalidation остаётся открытым до включения Stereo distance.

### Manual

Manual source публикует постоянное значение для актуального selected target. Временного stale timeout нет.

При смене target/generation старый result инвалидируется.

## Camera state и freshness

Connection/lifecycle enum:

```text
STARTING
ONLINE
RECONNECTING
ERROR
STOPPED
```

Vision публикует latest-only `CameraStatus` per camera с `camera`, `state`, `generation`, `last_receive_timestamp_ns` и optional diagnostic error fields.

Freshness не является отдельным state. UI/Core вычисляют stale по `CameraStatus.last_receive_timestamp_ns` и `camera-stale-timeout-ms`.

Конкретная reconnect/backoff policy остаётся открытым вопросом реализации.

## Запись видео

Основная запись сохраняет чистый:

```text
FramePacket.image
```

без UI overlays.

В файл не добавляются bbox, reticle, aim point, lead, FPS и status text. Это позволяет использовать запись повторно для тестирования detector/tracker.

## Обмен с Core/UI

Vision логически публикует:

- ordered/barrier `CameraSessionStarted`;
- latest-only `VisionResult` per camera;
- latest/invalidate-able `DistanceResult`;
- latest-only `CameraStatus` per camera.

Diagnostics/errors идут в logging, runtime state — через typed contracts. Generic event bus заранее не вводится. Конкретный thread-safe primitive и Qt notification coalescing остаются техническими вопросами реализации.

## Что ещё не определено

- механизм `capture_id` и stereo resync;
- owner staleness `DistanceResult`;
- processor-specific config;
- camera reconnect/backoff;
- concrete thread-safe primitives / notification coalescing;
- адаптация нагрузки после profiling.

Полный список: [Открытые вопросы](../../architecture/problems.md).
