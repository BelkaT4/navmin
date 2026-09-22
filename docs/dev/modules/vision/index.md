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

В current accepted minimum отдельного постоянного production `Camera Registry` нет. Каждый `VisionPipeline` является фактическим owner своей generation в пределах lifetime экземпляра и предоставляет собственные `session_barriers`, `latest_result` и `status`. E2E-1 использует эти pipeline-owned boundaries напрямую.

Единственный `CameraSessionGate` находится в main thread и принадлежит `Mediator`; второго generation gate/counter нет. Ownership при будущем camera reconnect или replacement экземпляра pipeline остаётся открытой частью полного reconnect work: monotonic generation semantics должны сохраниться, а единственный production owner будет выбран до/в E2E-4. См. [открытый вопрос о camera reconnect](../../architecture/problems.md#8-camera-reconnect-transitions-backoff).

### Production camera transport v1

Текущий production source — `GStreamerRtpJpegSource`: RTP/JPEG (MJPEG) over UDP → `rtpjpegdepay` → `jpegdec` → `videoconvert` → BGR `appsink`. Source владеет Gst pipeline, получает фактические width/height из sample caps и копирует Gst buffer в независимый NumPy frame до `unmap()`.

Low-latency boundary:

```text
appsink emit-signals=true max-buffers=<CameraConfig.buffer-size> drop=true sync=false
```

Для prototype `buffer-size = 1`; source и `VisionPipeline` оба latest-only и не образуют processing FIFO. `receive_timestamp_ns` ставится monotonic clock в момент application-side получения decoded frame. Текущий RTP transport не несёт согласованный cross-camera `capture_id`, поэтому Overview/Stereo Left/Stereo Right source публикуют `capture_id = None` до отдельного stereo-pairing решения.

`CameraConfig.address` — local bind address PC receiver, `port` — local listen UDP port, `rtp-enabled=true` обязателен. Рабочая mapping: Overview `8888`, Stereo Left `8889`, Stereo Right `8890`.

Appsink callback выполняется GStreamer streaming thread; process-global `GLib.MainLoop` для source не требуется. Один application `CameraWorker` на camera соединяет source с существующим `VisionPipeline.submit_decoded_frame() → process_latest()`, использует cooperative stop и bounded join. Reconnect/backoff policy в этом checkpoint не вводится.

### Localhost RTP/JPEG diagnostic boundary

`tools/run_localhost_rtp_diagnostic.py` поднимает diagnostics-only внешние
GStreamer senders для Overview и Stereo Left:

```text
videotestsrc is-live=true
→ raw video caps 320x240 @ 20 FPS
→ videoconvert
→ jpegenc
→ rtpjpegpay payload=26
→ udpsink 127.0.0.1:8888/8889
→ существующий production GStreamerRtpJpegSource
→ CameraWorker
→ VisionPipeline
```

Sender ownership и preflight реализованы в `navmin.diagnostics.localhost_rtp`,
чтобы будущий diagnostic launcher мог переиспользовать boundary без импорта из
`tools/`. Это не второй receiver и не альтернативный camera source: UDP/RTP/JPEG
всегда принимает существующий production `GStreamerRtpJpegSource`, а correction
выполняется существующим `VisionPipeline` через tool-local exact-size fisheye и
stereo calibration.

Diagnostic runner проверяет оба startup ordering, одновременный progress двух
потоков, silence/resume одного UDP sender на том же работающем receiver,
отклонение wrong-resolution кадра без raw fallback и bounded cleanup. Он не
является общей application composition, не запускает UI/Turret и не заменяет
быстрые `InMemoryFrameSource` tests. Silence/resume сохраняет текущую generation,
но не закрывает production camera reconnect/backoff: source recreation, hard
GStreamer ERROR/EOS recovery и ownership новой generation остаются вопросом #8.

Ручной transport check запускается одной командой из project root:

```bash
python tools/run_localhost_rtp_diagnostic.py --duration-seconds 30
```

## Working frame

Публичный `FramePacket.image` всегда содержит кадр с исправленной геометрией.

```text
Overview:
RTP/JPEG over UDP
→ GStreamer decode to raw BGR
→ OpenCV fisheye undistort
→ FramePacket
→ VisionProcessor
→ VisionResult

Stereo Left / Right:
RTP/JPEG over UDP
→ GStreamer decode to raw BGR
→ pinhole/stereo rectify
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

Geometric correction выполняется ровно один раз в `VisionPipeline` до `VisionProcessor`, потому что одна и та же geometry нужна Vision, UI, Aiming, stereo и recording. Camera source публикует только raw decoded BGR и не выполняет undistort/rectify.

## Calibration и `CameraModel`

Calibration хранится отдельно от `config.json`:

```text
calibration/
  overview.json
  stereo.json
```

### Overview

Overview calibration использует именно OpenCV fisheye model:

```text
schema_version
image_width
image_height
K: 3x3 finite
D: ровно 4 finite fisheye coefficients
new_camera_matrix: 3x3 finite
```

Maps строятся через `cv2.fisheye.initUndistortRectifyMap(K, D, I, new_camera_matrix, ...)`. Исправленный working-frame `CameraModel` использует `new_camera_matrix`.

Working frame сохраняет точное calibration resolution. Автоматического crop/resize или scaling `K/new_camera_matrix` в NavMin нет. Фактически наблюдавшийся Overview sender request `1296x972` может декодироваться как `1296x976`; source обязан брать размер из GStreamer sample caps, и calibration должна совпасть именно с фактическим размером.

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

В current accepted minimum каждый вызов `VisionPipeline.start()` для существующего экземпляра pipeline:

```text
pipeline-owned generation += 1
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

`VisionPipeline.start()` инвалидирует предыдущий latest result и сбрасывает processor state до публикации barrier. Полный reconnect/replacement lifecycle, включая сохранение monotonic generation при замене экземпляра pipeline, остаётся частью открытого reconnect work.

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

`VisionProcessor` является единственной архитектурной boundary обработки изображения. Его внутренняя реализация не фиксируется:

```text
Detector → Tracker
integrated tracker
другой алгоритм
```

Внутренние detector/tracker/components не являются публичными модулями NavMin и не должны импортироваться Core/UI как отдельные архитектурные зависимости.

Публичный результат обязан содержать `TrackedObject`:

- устойчивый `track_id` внутри generation;
- `bbox`;
- velocity центра bbox;
- `age_frames`.

### Processor implementations v1

Первая реализация содержит два взаимозаменяемых processor class:

```text
Legacy14VisionProcessor   # default
Legacy11VisionProcessor
```

Оба используют чисто перенесённую detector-логику соответствующих legacy Variant 1.4 / Variant 1.1, без старой UI/application-обвязки и без переноса legacy tracker как публичной архитектуры. Оба processor используют один общий внутренний `SimpleTracker`, поэтому сравнение 1.1 и 1.4 не смешивает качество detector с разными tracker algorithms.

`AdvancedVisionProcessor` на базе VT11/identity/reacquisition рассматривается как отдельная будущая реализация той же `VisionProcessor` boundary и не блокирует baseline Stage 5.

### `SimpleTracker` v1

`SimpleTracker` — внутренняя переиспользуемая часть первых processor implementations, а не отдельный межмодульный контракт. Для v1 фиксируются следующие semantics:

- tentative track публикуется как `TrackedObject` после двух последовательных matched observations;
- track после публикации хранится внутренне при кратком пропуске, но predicted bbox наружу не публикуется без нового detection;
- после трёх последовательных misses track удаляется;
- последние пять matched observations используются для motion history;
- velocity центра оценивается по реальным monotonic timestamps как robust median последовательных `dx/dt`, `dy/dt`, а не из предполагаемого FPS;
- association использует predicted center, жёсткий distance gate, consistency размера bbox и IoU, затем one-to-one matching;
- prediction используется только для внутренней association; Aiming/lead остаётся ответственностью Core;
- полноценный appearance identity/reacquisition в baseline tracker отсутствует; после окончательного удаления повторно найденный объект получает новый `track_id`;
- `track_id` монотонно выделяется и не переиспользуется для другого объекта внутри одной camera generation; processor state очищается при новой generation.

Точная cost formula, gates и numeric tuning являются внутренней настройкой processor/tracker и могут уточняться по измерениям без изменения публичного `VisionProcessor` contract.

### Внутренние настройки processor/tracker

В v1 algorithm tuning не входит в пользовательский `config.json` и не показывается в UI. Настройки хранятся рядом с owner-кодом как module-level constants:

```text
src/navmin/vision/processors/legacy_11/settings.py
src/navmin/vision/processors/legacy_14/settings.py
src/navmin/vision/tracking/settings.py
```

У processor-specific `settings.py` нет общей обязательной schema: Variant 1.1, Variant 1.4 и будущие processor implementations могут иметь разные поля. Общие настройки `SimpleTracker` не дублируются в processor directories. Runtime-derived/adaptive state хранится в экземпляре processor/tracker и не мутирует module constants.

Если позже конкретный tuning-параметр действительно потребуется менять пользователю, per-camera или runtime, он переносится в основной config отдельным архитектурным решением с явной validation/apply policy; заранее такой compatibility/config layer не создаётся.

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

Межпотоковая семантика `VisionResult` — latest-only per camera. UI читает revision-aware freshest snapshots через main-thread QTimer pump и не создаёт per-frame queued Qt callbacks.

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

Freshness не является отдельным state. UI вычисляет prototype presentation stale по `CameraStatus.last_receive_timestamp_ns` и authoritative `vision.camera-stale-timeout-ms`; это не меняет camera lifecycle state.

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

Diagnostics/errors идут в logging, runtime state — через typed contracts. Generic event bus заранее не вводится. UI notification coalescing реализован revision-aware QTimer pump поверх существующих thread-safe containers; ordered session barriers обрабатываются до latest payloads.

## Что ещё не определено

- механизм `capture_id` и stereo resync;
- owner staleness `DistanceResult`;
- camera reconnect/backoff;
- адаптация нагрузки после profiling.

Полный список: [Открытые вопросы](../../architecture/problems.md).
