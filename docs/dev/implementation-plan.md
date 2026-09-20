# План реализации BelkaT4 / NavMin v1

Этот документ задаёт **порядок реализации, зависимости между этапами и критерии приёмки**. Он не заменяет архитектурную документацию и не является источником нормативных runtime-контрактов.

Нормативное поведение определяется документами в `docs/dev/architecture/`, документацией модулей в `docs/dev/modules/` и исходными `.mmd` в `docs/dev/diagrams/`.

## 1. Как использовать план

План ведётся управляющим чатом проекта. Реализация отдельных этапов может выполняться в отдельных рабочих чатах.

Статусы этапов:

```text
not-started  работа не начата
in-progress  этап выполняется
blocked      есть блокирующий вопрос или зависимость
review       рабочий чат закончил реализацию, требуется приёмка управляющим чатом
done         этап принят управляющим чатом
```

Рабочий чат не должен самостоятельно менять утверждённую архитектуру ради удобства реализации. Если обнаружено настоящее противоречие или отсутствующий архитектурный выбор, нужно:

1. локализовать проблему;
2. показать, какой контракт/инвариант она затрагивает;
3. не вводить временный параллельный механизм без необходимости;
4. вернуть вопрос в управляющий чат;
5. после решения синхронизировать архитектурную документацию по правилам `docs/dev/architecture/AGENTS.md`.

Implementation detail, который не меняет responsibility, public contract, state machine, threading/transport semantics или observable behavior, можно решать внутри рабочего этапа.

## 2. Общие правила реализации

1. Выполнять этапы в порядке зависимостей, а не по удобству отдельных файлов.
2. Сначала реализовывать минимальный vertical foundation, затем consumers.
3. Не вводить заранее generic event bus, Supervisor, command lifecycle, motion generation, отдельный RESET_ID и другие механизмы, сознательно исключённые из v1.
4. Для транспорта состояния использовать определённую архитектурой семантику: latest-only, invalidatable latest-state или ordered barrier; не заменять её FIFO «для надёжности».
5. Hardware-dependent поведение должно иметь simulator/fake boundary там, где это возможно без искажения архитектуры.
6. Каждый этап обязан сохранять возможность запуска normal tests без физического оборудования.
7. Изменение production-кода само по себе не требует нового теста. Перед созданием тестов действует coverage-first policy из корневого `AGENTS.md`.
8. Не создавать test file, механически зеркалящий каждый production file. Test ownership определяется поведением/invariant, который защищается.
9. Документацию менять только при реальном изменении контракта/поведения/архитектурного решения либо когда реализация выявила уже существующую ошибку документации.

## 3. Начальная структура репозитория

Это implementation layout, а не архитектурный контракт. Leaf-файлы можно уточнять в соответствующем этапе без архитектурного решения, если границы модулей не меняются.

```text
src/
  navmin/
    app/
    common/
    runtime/
    config/
    turret/
    vision/
    core/
    ui/

firmware/
  stm32/

tests/
  unit/
  integration/
  hardware/
  support/        # только реально общие helpers/fakes

config/
calibration/
```

Для v1 базовый Python package name: `navmin`. Это implementation-level convention master plan, а не runtime-архитектурный контракт. Если имя потребуется изменить, это нужно сделать до появления устойчивых imports либо явно обновить master plan.

`tests/support/` не должен становиться свалкой. Helper переносится туда только если используется несколькими тематическими test groups и имеет устойчивую ответственность.

## 4. Стратегия тестирования

### 4.1. Уровни

```text
unit         один owner / state machine / parser / math / policy
integration несколько реальных project components через их публичные boundaries
hardware     физические камеры, Raspberry Pi, RS485/UART, STM32, турель
```

Normal test run не должен требовать hardware.

### 4.2. Защита от дублирования

Перед добавлением test/test file рабочий чат обязан:

1. сформулировать защищаемый behavior/invariant;
2. найти coverage по symbol/class/function;
3. найти coverage по поведению и failure mode;
4. проверить существующие fixtures/fakes/helpers;
5. расширить существующее тематическое покрытие, если это остаётся читаемо;
6. создать новый test file только для новой самостоятельной ответственности или отдельного класса отказов.

При приёмке этапа должно быть понятно не «сколько добавлено тестов», а **какие новые gaps были закрыты**. Если новый test file создан, в handoff кратко указывается, почему существующие test owners для него не подходили.

### 4.3. Документация тестов

На этапе 1 создать короткий `tests/README.md`. В нём фиксируются только устойчивые правила:

- структура suite;
- markers и команды запуска;
- расположение shared fakes/helpers;
- граница normal/integration/hardware;
- coverage-first / anti-duplication policy.

Не вести вручную каталог всех test cases: источником фактического покрытия остаётся сам test suite.

## 5. Сводка этапов

| Этап | Статус | Основной результат | Зависит от |
|---|---|---|---|
| 1. Foundation | done | package/test skeleton, contracts, runtime primitives | — |
| 2. Config + Calibration foundation | done | typed config, persistence, calibration loading/model boundary | 1 |
| 3. Turret PC stack | done | PC protocol/transport/controller/HAL + simulator | 1, 2 |
| 4. STM32 firmware + UART | done | firmware protocol/control + UART integration boundary | 3 |
| 5. Vision | in-progress — prototype minimum accepted | camera pipelines, generations, working frame, manual distance | 1, 2 |
| 6. Core + Aiming | in-progress — prototype minimum accepted | mediator/state transitions/aiming/tracking error | 2, 3, 5 |
| 7. UI | in-progress — prototype minimum accepted | PyQt6 main/preview, controls, overlays, settings | 2, 3, 5, 6 |
| 8. System integration and v1 hardening | not-started | startup/shutdown, failure paths, E2E, hardware smoke/performance | 4, 5, 6, 7 |

Полноценный stereo distance, camera-to-turret rotational extrinsic, target handoff и другие явно отложенные возможности не являются условиями завершения v1, если архитектурные документы не будут изменены отдельным решением.

### Prototype-first checkpoint для этапов 5–7

После Stage 4 первая цель — как можно раньше получить запускаемый end-to-end prototype и проверить реальные относительные перемещения и TRACKING. Для этапов 5–7 разрешён один последовательный vertical-slice checkpoint до полного завершения каждого этапа:

```text
Stage 5 minimum: Legacy14VisionProcessor + SimpleTracker + рабочие Overview/Stereo Left frames
→ Stage 6 minimum: RELATIVE click-to-move + TRACKING selection/error path
→ Stage 7 minimum: fullscreen main/preview + mode/motor/link/Emergency + bbox/selection
→ первый runnable relative/tracking smoke
→ затем завершение оставшегося scope Stage 5/6/7
```

Это не создаёт временную параллельную архитектуру: prototype использует те же `VisionProcessor`, Core/Aiming, Turret и UI contracts, которые остаются production path. Упрощается только набор реализованных функций. Recording, расширенная диагностика, Stereo Right diagnostic view, второй baseline processor 1.1, полный settings UI и presentation polish не должны блокировать первый runnable prototype, если их отсутствие не нарушает используемый contract.

Рабочие чаты по-прежнему выполняются последовательно на одной ветке; этот checkpoint не разрешает параллельные конкурирующие реализации. Статус полного Stage меняется на `done` только после выполнения его полного критерия завершения.

---

## 6. Этап 1 — Foundation

**Статус:** `done`

### Цель

Создать минимальную исполняемую и тестируемую основу, на которой последующие модули смогут использовать общие типы и одинаковые concurrency semantics, не изобретая их независимо.

### Читать перед началом

- `AGENTS.md`
- `docs/dev/architecture/overview.md`
- `docs/dev/architecture/contracts.md`
- `docs/dev/architecture/decisions.md`
- `docs/dev/architecture/problems.md`, пункты про latest-state, worker lifecycle и logging
- `docs/dev/diagrams/overview-diagram.mmd`

### Реализовать

- базовый `src/navmin/` package;
- минимальный `pyproject.toml`/test configuration для воспроизводимого импорта и запуска suite;
- общие dataclass/enum/value types из `contracts.md`, без добавления новых domain states;
- concrete thread-safe primitives для:
  - latest-only value;
  - invalidatable latest-state;
  - monotonic revision там, где она нормативно требуется;
  - ordered/barrier delivery для `CameraSessionStarted`;
- минимальный application lifecycle boundary / stop token abstraction без отдельного Supervisor;
- logging bootstrap, достаточный для последующих модулей;
- test infrastructure и `tests/README.md`;
- минимальный executable/import smoke path без UI/hardware.

### Не делать

- generic SystemEvent bus;
- generic FIFO для latest-state contracts;
- Qt-specific notification bridge;
- camera/UART implementation;
- полноценный worker orchestrator сверх конкретных требований startup/shutdown.

### Тестовый фокус

- atomicity/thread safety primitives;
- invalidation/revision semantics;
- barrier ordering;
- отсутствие FIFO backlog у latest store;
- basic lifecycle cancellation semantics.

Не дублировать тест каждого concrete store для каждого будущего domain type, если один generic primitive уже покрыт и domain type не добавляет своей логики.

### Закрытые prerequisites и открытые вопросы этапа

- concrete latest-state / thread-safe primitives закрыты в Foundation;
- минимальная часть #10: общий stop/join contract;
- минимальная часть #13: logging transport/setup.

Qt-specific notification bridge остаётся до UI-этапа.

### Критерий завершения

- package импортируется в clean environment;
- common contracts соответствуют архитектуре;
- concurrency primitives имеют focused tests;
- normal tests не требуют hardware/Qt display;
- будущие модули могут использовать foundation без создания параллельных state primitives.

---

## 7. Этап 2 — Config + Calibration foundation

**Статус:** `done`

### Цель

Реализовать единый типизированный источник конфигурации и базовые calibration/model boundaries до появления модулей-consumers.

### Читать перед началом

- `docs/dev/architecture/configuration.md`
- `docs/dev/architecture/contracts.md`
- `docs/dev/architecture/decisions.md`
- `docs/dev/modules/core/config-manager.md`
- `docs/dev/modules/core/aiming.md`
- `docs/dev/modules/vision/index.md`
- `docs/dev/modules/turret/index.md`
- релевантный пункт `problems.md` #7; базовая v1 persistence/schema policy уже закрыта в `configuration.md`

### Реализовать

- typed immutable config snapshots;
- `schema-version = 1` validation;
- load/save;
- temporary file + fsync/replace policy, насколько это поддерживается целевой платформой;
- ошибку corrupted existing JSON без silent overwrite;
- module-oriented `ConfigUpdate` publication через foundation latest-state;
- общие config types/snapshots и минимальные diff/metadata средства, необходимые owner-модулям для уже принятой classification `dynamic / pipeline restart / app restart / controlled serial transition`;
- owner-specific решение `apply / restart / reconnect` остаётся в соответствующих Stage 3/5/6/7; Config Manager не получает централизованный `requires_restart`/resource policy;
- загрузку calibration files отдельно от `config.json`;
- immutable `CameraModel`/calibration data boundary, достаточный для Aiming/Vision;
- validation errors, пригодные для UI/logging без generic event bus.

### Закрытый decision checkpoint перед Stage 2

До начала реализации зафиксирована v1-policy:

- полностью отсутствующий `config.json` — startup/config error; файл автоматически не создаётся;
- malformed/corrupted JSON — startup/config error; исходный файл не изменяется;
- отсутствующий/неверного типа `schema-version` — invalid config;
- `schema-version != 1` — unsupported schema error без автоматической migration;
- неизвестные поля — validation error;
- отсутствующие required fields — validation error;
- optional fields получают только явно документированные in-memory defaults;
- invalid type/range/non-finite number — validation error без silent fallback;
- невалидный runtime update не заменяет последний валидный snapshot и не записывается как частично применённая конфигурация.

Автоматический migration mechanism проектируется только после появления реальной schema v2. Базовая схема required/optional и validation rules закреплены в `configuration.md`; рабочий чат Stage 2 их реализует, а не выбирает заново.

### Тестовый фокус

- valid/invalid schema;
- corrupted JSON preservation;
- atomic save behavior на test filesystem;
- config snapshot immutability;
- Config Manager публикует snapshots/updates, но не принимает owner-specific решение restart/reconnect/apply;
- exact calibration resolution/validation boundaries;
- отсутствие непреднамеренной записи реального пользовательского config во время тестов.

### Критерий завершения

- Config Manager можно использовать без UI;
- consumers получают typed snapshots;
- calibration invalid/mismatch даёт явную ошибку, а не raw fallback;
- persistence tests изолированы temp directories;
- архитектурный config contract не дублируется в UI/Turret/Vision.

---

## 8. Этап 3 — Turret PC stack

**Статус:** `done`

### Цель

Реализовать PC-side Turret полностью поверх fake/simulated transport до зависимости от реального STM32.

### Читать перед началом

- `docs/dev/architecture/serial-protocol.md`
- `docs/dev/architecture/contracts.md`
- `docs/dev/architecture/configuration.md`
- `docs/dev/architecture/decisions.md`
- `docs/dev/modules/turret/index.md`
- `docs/dev/diagrams/turret-diagram.mmd`

### Реализовать

- frame codec + CRC-16/MODBUS;
- request/response parser;
- response correlation по `REQUEST_ID + COMMAND_CODE`;
- один общий `next_request_id:uint16`;
- ordinary exact retry semantics;
- Emergency special transaction / sequence resync semantics;
- physical transport abstraction, fake transport и concrete serial-port adapter за тем же interface;
- PC-side `SET_BAUDRATE` transaction/state machine;
- uncertain old/new baud recovery logic, полностью проверяемую на fake transport до реального RS485;
- Turret HAL:
  - degrees↔steps;
  - axis inversion;
  - mechanical conversion restart-only policy;
- Controller:
  - authoritative applied mode;
  - один latest-only `pending_motion`;
  - PID ownership/reset rules;
  - mode transitions;
  - StopMotion;
  - motor state;
  - config application;
- reconnect/recovery state machine с agreed baud-candidate order и capped interruptible backoff;
- cooperative Turret worker lifecycle поверх Foundation `StopToken`;
- Turret diagnostics/logging policy без high-rate INFO spam;
- Turret simulator/fake endpoint для integration tests без STM32.

### Не делать

- `MOVE_COMPLETED`;
- command lifecycle/command_id;
- `RESET_ID`;
- persistent emergency latch;
- parallel pending slots для relative/velocity;
- автоматический MOTOR_ON после recovery;
- absolute position/soft limits, которых нет в hardware model.

### Закрытый decision checkpoint перед config/PID runtime-веткой

До начала Stage 3 управляющий чат зафиксировал Turret-семантику runtime PID config:

- изменение `Kp/Ki/Kd` конкретной оси в TRACKING полностью reset'ит PID state только этой оси перед обработкой следующего нового `TrackingError`;
- первый sample этой оси после reset использует новые gains и остаётся P-only: `I=0`, `D=0`;
- если gains меняются вне TRACKING, отдельный runtime reset не нужен: при следующем входе в TRACKING срабатывает уже существующий reset boundary;
- уменьшение application-side output limit (`max-speed-*-deg-s`) не reset'ит PID целиком, но немедленно clamp'ит сохранённый I-term соответствующей оси в новый диапазон `±max_speed`;
- увеличение output limit сохраняет текущий I-term без искусственного масштабирования;
- если в одном config revision меняются и gains, и output limit одной оси, gain-change reset имеет приоритет, поэтому новый I-term начинается с zero уже под новым limit;
- config change сам по себе не создаёт motion command, не меняет `control_mode` и не вводит внешний `PID_RESET` contract.

Processor-specific runtime policy для Vision закрыта decision checkpoint перед Stage 5: внутренний tuning не входит в `config.json` v1. `target-lost-timeout-ms` остаётся checkpoint Stage 6.

### Закрытый decision checkpoint перед transport/recovery веткой

До начала Stage 3 управляющий чат также зафиксировал Turret transport/lifecycle/logging semantics:

- auto-reconnect запускается после retry exhaustion, physical serial I/O/disconnect failure, exhausted Emergency retries или неуспешного bounded baud recovery;
- matching command-level result сам по себе не означает transport loss; `INVALID_REQUEST_ID` требует Emergency-based resync;
- обычный baud search: `last-known → desired → 9600`, без дубликатов; uncertain `SET_BAUDRATE`: `new → old → 9600`, без дубликатов;
- reconnect attempts не имеют конечного лимита; backoff `0.25 → 0.5 → 1 → 2 → 2 ... s`, reset после `READY`;
- отсутствие STM32/device не является fatal `ERROR`, пока automatic reconnect может продолжаться;
- Turret worker создаёт/останавливает application orchestration, serial waits bounded/cancellable, backoff interruptible через `StopToken`, shutdown использует bounded join;
- numeric join timeout — implementation tuning, не config field; незавершившийся worker является явной shutdown error;
- Turret использует обычный Python logging; INFO — lifecycle/connection/recovery/baud/failures, high-rate PID/setpoint traffic на INFO не идёт; `QueueHandler/QueueListener` не вводится без измеренной необходимости.

### Тестовый фокус

Предпочитать protocol/state-machine matrices и parameterization вместо нового файла на каждую command.

Нужны отдельные failure modes, а не дубли одной happy path:

- wrap request ID;
- retry exact same transaction;
- stale/mismatched response ignored;
- normal committed in-flight request;
- emergency preemption boundary;
- recovery sequence;
- mode transition zero handshake;
- PID reset boundaries;
- runtime gain change reset по оси и P-only first sample;
- output-limit decrease clamp I-term без полного PID reset, increase сохраняет I-term;
- HAL conversion/inversion;
- dynamic STM32 config snapshot semantics на PC-side;
- `MOTOR_OFF` invalidates unsent `pending_motion`;
- `MOTOR_ON` не replay'ит старый motion intent;
- StopMotion invalidates несовместимый unsent pending intent;
- wire conversion/encoding явно отклоняет значения вне encodable `int32`/`uint32` range;
- `SET_BAUDRATE` success/lost-response/old-new uncertainty/recovery на fake transport;
- ordinary reconnect baud candidates `last-known → desired → 9600` с deduplication;
- uncertain baud candidates `new → old → 9600` с deduplication;
- reconnect backoff progression/cap/reset и отсутствие finite attempt limit;
- `request_stop()` прерывает reconnect/backoff и worker завершается через bounded join;
- matching command-level errors не ошибочно классифицируются как physical transport loss, а `INVALID_REQUEST_ID` ведёт в Emergency resync;
- lifecycle/recovery logging не требует generic event bus и не пишет high-rate PID/setpoint traffic на INFO.

Эти cases расширяют существующие protocol/state-machine matrices; отдельный test file для каждого case не требуется.

### Открытые вопросы этапа

Turret-specific архитектурные вопросы reconnect/backoff, worker lifecycle и logging закрыты до начала реализации. Stage 3 не должен заново выбирать эти semantics.

Точный numeric join timeout и другие чисто внутренние bounded timing constants можно выбрать как implementation tuning при условии сохранения зафиксированных cancellation/shutdown invariants; они не становятся новым architecture state или `config.json` field без отдельной причины.

### Критерий завершения

- весь Turret PC stack проходит normal tests с fake transport;
- simulator способен пройти startup/recovery и принять реальные protocol frames;
- PC-side `SET_BAUDRATE`, uncertain old/new recovery и concrete serial adapter реализованы за transport interface и проверены без реального STM32;
- никаких hardware tests не требуется для завершения PC-side логики;
- serial-protocol.md остаётся единственным wire-format source of truth.

---

## 9. Этап 4 — STM32 firmware + UART

**Статус:** `done`

### Цель

Реализовать firmware counterpart существующего protocol contract на STM32F103C8T6. Первая реализация использует обычный full-duplex UART как development/integration physical transport. Binary protocol, request/response, retry, Emergency, request-sequence и baud semantics остаются окончательными и не зависят от будущего перехода на RS485.

Production RS485 half-duplex переносится за пределы обязательной первой реализации и должен позднее заменить только physical byte transport, не создавая вторую версию протокола.

### Читать перед началом

- `docs/dev/architecture/serial-protocol.md`
- `docs/dev/architecture/configuration.md`
- `docs/dev/modules/turret/index.md`
- `docs/dev/architecture/decisions.md`
- `problems.md` #1 — только как явно deferred RS485 hardware question; он не блокирует UART-first Stage 4

### Реализовать firmware

- byte parser/resynchronization/inter-byte timeout;
- CRC;
- expected request ID + exact retry cache;
- все v1 commands из `serial-protocol.md`;
- `EMERGENCY_STOP` special/resync behavior;
- relative planner без completion event;
- velocity target + acceleration limiter;
- velocity watchdog;
- motor enable/disable semantics;
- atomic full `SET_CONFIG` snapshot;
- bounded firmware sanity limits;
- runtime `SET_BAUDRATE` semantics;
- serial byte I/O boundary поверх USART3 full-duplex UART;
- firmware target и pin mapping, зафиксированные в Turret module docs.

### UART integration boundary

- production binary frames должны передаваться через обычный USART3 TX/RX без protocol fork;
- firmware build и host-side tests не требуют физической STM32-платы;
- реальный PC↔USB-UART↔STM32 smoke переносится в Stage 8 user-side hardware checkpoint;
- будущий RS485 transport не должен менять framing, request IDs, retry, Emergency или command semantics.

### Durable checkpoints

- **4A — Firmware foundation:** project skeleton, byte parser/resynchronization/inter-byte timeout, CRC, request sequence, exact retry cache, command/result framing, Emergency resync.
- **4B — Motor/control:** STEP generation, relative planner, velocity target + acceleration limiter, watchdog, motor state, atomic `SET_CONFIG`, firmware bounds.
- **4C — UART integration:** USART3 byte transport, runtime `SET_BAUDRATE`, reproducible firmware build, host-side integration tests and final Stage 4 verification.

RS485 migration не входит в 4A–4C и возвращается отдельным post-v1 hardware change.

### Открытые вопросы, которые должен закрыть этап

- фактические firmware bounds из `serial-protocol.md`;
- software timing limits parser/control loop, если они влияют на protocol tuning.

`problems.md` #1 про RS485 half-duplex остаётся открытым, но намеренно deferred и не блокирует завершение UART-first Stage 4.

### Тестовый фокус

Host-side protocol tests из этапа 3 не дублировать в firmware как копию тех же Python cases. Firmware tests должны защищать firmware parser/control implementation; hardware suite — реальный boundary и timing/fault modes.

Отложенный P2 из независимого аудита Stage 3: перед firmware/hardware integration повысить fidelity `FakeStm32Endpoint` для ordinary request sequence — `expected REQUEST_ID` и exact retry cache. Это simulator hardening, а не незакрытый production blocker Stage 3.

### Критерий завершения

- PC simulator tests всё ещё проходят;
- firmware build воспроизводим;
- firmware parser/protocol/control имеют hardware-independent tests;
- USART3 transport реализует тот же production protocol без отдельной UART-версии;
- normal Python suite остаётся hardware-independent;
- отсутствие физической STM32-платы не блокирует Stage 4 acceptance; реальный UART hardware smoke остаётся обязательным Stage 8 checkpoint;
- RS485-specific код/DE-RE/turnaround не добавляются до отдельного решения.

---

## 10. Этап 5 — Vision

**Статус:** `in-progress` — prototype minimum accepted

### Цель

Реализовать камеры и Vision pipelines с корректной session/generation семантикой, working frame и manual distance, не блокируя v1 полноценным stereo distance.

### Читать перед началом

- `docs/dev/architecture/overview.md`
- `docs/dev/architecture/contracts.md`
- `docs/dev/architecture/configuration.md`
- `docs/dev/architecture/decisions.md`
- `docs/dev/modules/vision/index.md`
- `docs/dev/diagrams/vision-diagram.mmd`
- `problems.md` #8, Vision часть #10/#13

### Реализовать

- camera pipelines по `CameraRole` без отдельного production `Camera Registry`;
- по одному pipeline worker на роль;
- в current accepted minimum каждый `VisionPipeline` владеет своей monotonic generation в пределах lifetime экземпляра, а `VisionPipeline.start()` публикует `CameraSessionStarted` до данных новой generation;
- E2E-1 использует напрямую pipeline-owned `VisionPipeline.session_barriers`, `VisionPipeline.latest_result` и `VisionPipeline.status`;
- `FramePacket` stamping;
- receive timestamp/freshness;
- Overview undistort → working frame;
- Stereo Left/Right rectify → working frame;
- immutable `CameraModel` binding к generation;
- `VisionProcessor` interface как единственную внешнюю boundary processing;
- `Legacy14VisionProcessor` как default и `Legacy11VisionProcessor` как альтернативную baseline implementation;
- чистый перенос detector-алгоритмов legacy Variant 1.4 / 1.1 без старой UI/application обвязки;
- общий внутренний `SimpleTracker` для обоих processors: 2-hit confirmation, delete после 3 consecutive misses, 5 matched observations history, real-time robust median velocity, predicted-center/distance/size/IoU one-to-one association, без публикации predicted bbox и без полноценного reacquisition;
- processor/tracker internal tuning через owner-local `settings.py` module constants, без processor-specific fields в `config.json`/UI v1;
- per-camera processing config + processing scope;
- `VisionResult` latest-only per camera;
- camera reconnect/state transitions/backoff;
- published `FramePacket.image` остаётся clean corrected working frame без overlays и пригоден для последующего UI-side recording;
- manual `DistanceResult` source как полноценный v1 path;
- camera/pipeline simulation suitable for tests and UI development.

Для первого runnable vertical slice достаточно сначала реализовать `Legacy14VisionProcessor` (default) + общий `SimpleTracker` и working-frame path Overview/Stereo Left. `Legacy11VisionProcessor` остаётся обязательным для полного завершения Stage 5, но не блокирует первый end-to-end smoke.

Этот current minimum не принимает окончательного решения об ownership при будущем camera reconnect или replacement экземпляра pipeline. До/в E2E-4 и полном Stage 5 reconnect work нужно сохранить monotonic generation semantics и выбрать одного production owner; отдельный registry или второй generation counter не являются требованием E2E-1.

### Отложить

- полноценный `capture_id` pairing algorithm;
- production stereo depth;
- target handoff;
- load adaptation;
- future latest-live-frame optimization;
- `AdvancedVisionProcessor` на базе VT11 identity/reference/reacquisition до получения baseline measurements.

Stereo Right pipeline должен существовать настолько, насколько это требуется rectification/session/diagnostics architecture, но отсутствие production stereo distance не блокирует v1.

### Закрытый decision checkpoint перед Vision implementation

До начала Stage 5 управляющий чат зафиксировал baseline processing semantics:

- `VisionProcessor` остаётся единственной архитектурной processing boundary; detector/tracker decomposition является private implementation detail;
- baseline processors: `Legacy14VisionProcessor` (default) и `Legacy11VisionProcessor`, оба используют один `SimpleTracker`;
- legacy detector algorithms переносятся чисто, без legacy UI/application glue и без копирования legacy tracker как есть;
- tracker публикует track после 2 confirmations, удаляет после 3 consecutive misses, использует history=5 matched observations и real-time robust median velocity; prediction используется только внутри association, predicted bbox без detection наружу не публикуется;
- internal detector/tracker tuning хранится рядом с owner code в `settings.py` как module-level constants и не входит в `config.json`/UI v1;
- full VT11 identity/reacquisition остаётся будущим `AdvancedVisionProcessor`, не blocker baseline Stage 5.

### Открытые вопросы, которые должен закрыть этап

- #8 camera reconnect transitions/backoff;
- Vision часть #10 worker lifecycle;
- Vision часть #13 logging.

`problems.md` #2/#3 остаются deferred до полноценного stereo distance, если manual source достаточен для v1.

### Тестовый фокус

- generation increments;
- producer ordering: `CameraSessionStarted` публикуется до любых data новой generation;
- exact resolution/calibration behavior;
- corrected clean working frame contract без overlays;
- reconnect state transitions;
- processing scope;
- both baseline processor classes produce the same public `TrackedObject` contract;
- SimpleTracker confirmation/miss/ID non-reuse/real-time velocity/one-to-one association behavior;
- no predicted `TrackedObject` publication on detector miss;
- default processor selection is `Legacy14VisionProcessor`;
- manual distance target binding/invalidation where defined.

Rejection stale/unaccepted generation проверяется на реальных consumer boundaries в Stage 6/7, а не искусственным consumer внутри Vision.

Не копировать один и тот же generation test отдельно для каждой camera role, если role не меняет behavior; parameterize role cases.

### Критерий завершения

- simulated pipelines дают реальные `FramePacket`/`VisionResult` contracts;
- reconnect не смешивает generations;
- calibration mismatch не имеет raw fallback;
- Overview + Stereo Left могут независимо работать/падать;
- manual distance usable downstream;
- published working frame пригоден для clean recording, но recording lifecycle/file I/O не принадлежат Vision;
- normal tests не требуют камер/Raspberry Pi.

---

## 11. Этап 6 — Core + Aiming

**Статус:** `in-progress` — prototype minimum accepted

### Цель

Реализовать orchestration application state и математический путь от working-frame данных к `MoveRelativeCommand`/`TrackingError`, не перенося PID или transport ownership в Core.

### Читать перед началом

- `docs/dev/architecture/overview.md`
- `docs/dev/architecture/contracts.md`
- `docs/dev/architecture/decisions.md`
- `docs/dev/modules/core/index.md`
- `docs/dev/modules/core/mediator.md`
- `docs/dev/modules/core/aiming.md`
- `docs/dev/diagrams/core-diagram.mmd`
- `docs/dev/architecture/problems.md`, в том числе Core-related часть #7
- Turret/Vision contracts как consumers/producers

### Реализовать Mediator

- main/preview state;
- `CameraSessionGate` consumer behavior;
- selection только в TRACKING;
- `TargetRef` validation;
- target temporary/final loss;
- target A→B behavior;
- RELATIVE↔TRACKING request/applied-mode boundaries;
- click-to-move RELATIVE path;
- StopMotion/Emergency user intents;
- Turret connection-loss response;
- typed state для UI.

### Реализовать Aiming

- pixel→ray через generation-bound `CameraModel`;
- aim point;
- image Y-down → turret Y-up mapping;
- relative angle delta;
- tracking error;
- lead computation согласно текущим contracts/config;
- Core хранит/передаёт только актуальный target-bound `DistanceResult` по существующему контракту; Stage 6 не придумывает новое влияние distance на Aiming/ballistics.

Для первого runnable vertical slice приоритетны только подтверждённый `RELATIVE` click-to-move и минимальный `TRACKING` path `selection → TrackingError → Turret`. Остальные loss/config/diagnostic edge cases закрываются до полного `done` Stage 6.

### Decision checkpoint до target-loss runtime-ветки

До реализации поведения временно потерянной цели управляющий чат должен закрыть Core-часть `problems.md` #7:

- как изменение `target-lost-timeout-ms` применяется к цели, которая уже находится в состоянии temporary loss.

Это observable state-machine semantics; рабочий чат не выбирает её самостоятельно.

### Не делать

- PID в Core;
- `PID_RESET` command из Core;
- `selected_target` в RELATIVE;
- автоматический main-camera switch;
- скрытый absolute turret position;
- новые target lifecycle states без concrete requirement.

### Тестовый фокус

Тестировать state transitions как behavior matrix, а не по одному файлу на каждое user action:

- stale/unaccepted generation rejected на Core `CameraSessionGate` boundary;
- stale generation click/selection rejected;
- selection only main camera + TRACKING;
- applied mode authority from TurretState;
- target temporary vs final loss;
- A→B switch;
- swap в TRACKING invalidates target-dependent state и запрашивает StopMotion согласно текущему contract;
- swap в RELATIVE не отменяет уже сформированный `MoveRelativeCommand`;
- StopMotion/Emergency очищают соответствующий Core target-dependent state;
- Turret connection loss очищает target-dependent state;
- RELATIVE click survives later camera session change после успешной validation;
- coordinate sign/aim point/ray math;
- no TrackingError when preconditions invalid.

Эти cases добавляются в существующие behavior matrices, а не превращаются в отдельные test suites без нового failure-mode owner.

### Критерий завершения

- Core/Aiming полностью тестируются с simulated Vision/Turret boundaries;
- mode/selection invariants соответствуют architecture baseline;
- Core не знает serial protocol и не владеет PID;
- UI ещё не нужен для проверки business behavior.

---

## 12. Этап 7 — UI

**Статус:** `in-progress` — prototype minimum accepted

### Цель

Подключить PyQt6 как display/input layer поверх готовых Core/Vision/Turret contracts, не перенося в окна business logic.

### Читать перед началом

- `docs/dev/modules/ui/index.md`
- `docs/dev/architecture/overview.md`
- `docs/dev/architecture/contracts.md`
- `docs/dev/architecture/configuration.md`
- `docs/dev/architecture/decisions.md`
- `docs/dev/diagrams/ui-diagram.mmd`
- `problems.md` #11, #22

### Реализовать

Для первого runnable vertical slice:

- приложение стартует fullscreen;
- Overview / Stereo Left отображаются как large main + небольшой preview в правом нижнем углу области видео; preview не перекрывает нижнюю operational bar;
- click по preview и отдельный кликабельный swap-icon в preview выполняют explicit swap;
- нижняя operational bar имеет fixed-size controls и не меняет геометрию из-за текста/состояния;
- нижняя bar содержит `RELATIVE / TRACKING`, кликабельный confirmed motor state, connection state и крупный `EMERGENCY` в правом нижнем углу; dedicated ordinary Stop button в первом prototype отсутствует;
- в TRACKING левый click по bbox выбирает target, empty click выполняет deselect; при overlapping bbox выбирается содержащий click bbox с ближайшим центром; nearest-object helper вне bbox не используется;
- в RELATIVE левый click по main working frame означает click-to-move; preview click не выполняет selection/click-to-move;
- baseline overlay показывает bbox всех текущих объектов и явно выделяет selected target, без постоянных ID/distance/velocity/age labels;
- camera/turret status presentation достаточна для запуска prototype.

До полного завершения Stage 7 также реализовать:

- stale/error overlay с сохранением последнего кадра;
- top menu bar для recording/view/diagnostics/settings;
- clean recording implementation + controls в main-thread/UI-side path: только accepted через `CameraSessionGate` `FramePacket.image`, без overlays; при активной записи в левом верхнем углу main view мигает красный круг и постоянно отображается белая надпись `Запись`;
- settings UI поверх Config Manager, без прямой записи module internals;
- Stereo Right открывается только из Diagnostics как отдельный diagnostic view и не участвует в main/preview swap;
- recording lifecycle/file I/O остаются вне Vision worker;
- Qt notification bridge/coalescing для latest-state;
- ordered handling `CameraSessionStarted` отдельно от coalesced latest payloads;
- partial-failure UX для доступных v1 сценариев.

Motor control выполняется одним click без confirmation dialog. Текст/индикатор motor control отражает подтверждённый `TurretState.motor_state`, а не optimistic UI state.

### Открытые вопросы, которые должен закрыть этап

- UI часть #11 partial failures.

Future latest-live-frame mode (#22) остаётся deferred. Gesture/overlay/Stereo Right baseline UX уже закрыт управляющим чатом и не должен проектироваться заново внутри Stage 7.

### Тестовый фокус

- UI adapter/view-model logic без display там, где возможно;
- focused Qt tests для signal/coalescing/barrier ordering;
- action enable/disable по current state;
- stale overlay;
- stale/unaccepted generation rejected на UI display/diagnostics/recording boundaries;
- main/preview selection rules;
- clean recorder получает только accepted corrected `FramePacket.image` и не записывает overlays;
- settings round-trip через Config Manager.

Не дублировать Core state-machine tests через клики UI, если UI test не проверяет отдельную wiring/presentation failure mode.

### Критерий завершения

- UI не владеет control state;
- queued latest notifications не образуют backlog;
- barrier event нельзя потерять/coalesce;
- UI можно запускать с симулированными Turret/Vision;
- hardware absence отображается как state, а не приводит к падению приложения.

---

## 13. Этап 8 — System integration and v1 hardening

**Статус:** `not-started`

### Цель

Собрать приложение в целостный v1, проверить startup/shutdown, recovery/failure paths, hardware boundary и timing, не добавляя новые features ради «завершённости».

### Читать перед началом

- весь `overview.md`;
- relevant module docs;
- `serial-protocol.md` для hardware scenarios;
- оставшиеся `problems.md` категории «во время первой реализации» и «после первых измерений»;
- все принятые за реализацию записи `decisions.md`.

### Реализовать/проверить

- application composition root;
- startup order;
- shutdown order;
- prohibition of new motion during shutdown;
- final MOTOR_OFF path;
- worker stop/join behavior;
- config propagation end-to-end;
- camera partial failures;
- Turret unavailable/reconnect;
- PC config changed while STM32 update pending;
- recovery with real STM32;
- minimum real Raspberry Pi/camera smoke:
  - Overview real stream -> decode -> undistort -> session/result;
  - Stereo Left real stream -> decode -> rectify -> session/result;
  - Stereo Right -> decode/rectify/diagnostic pipeline minimum;
  - physical/network stream disconnect + reconnect -> new generation;
  - real stale/freeze path;
  - resolution/calibration mismatch rejection;
  - basic real-stream frame latency/FPS measurement;
- simulated end-to-end tracking/relative flows;
- recording/logging/diagnostics sufficient for field debugging;
- performance measurement of tracking pipeline;
- reproducible run instructions/dependencies/packaging required by deployment target.

### Timing and tuning

На этом этапе измеряются, а не угадываются:

- frame latency;
- Vision processing time;
- Core→Turret latency;
- UART request/response latency;
- effective tracking update rate;
- watchdog/retry/reconnect tuning ranges.

`problems.md` #15 закрывается только после измерений. PID numerical tuning может начаться здесь, но изменение архитектуры PID не требуется без evidence.

### Тестовый фокус

Integration tests должны проверять boundaries, которых нет в unit tests:

- Vision generation → Core gate → UI;
- Core mode transition → Turret zero handshake;
- tracking error → Turret PID → fake/real endpoint;
- transport loss/recovery;
- config change propagation;
- startup/shutdown with simulated components.

Hardware tests остаются отдельным suite.

### Критерий завершения v1

- все этапы 1–7 приняты;
- normal test suite green без hardware;
- обязательные STM32/UART и Raspberry Pi/camera hardware smoke scenarios выполнены и результаты зафиксированы;
- нет известных P0/P1 противоречий между реализацией и архитектурой;
- открытые deferred вопросы явно остаются в `problems.md`, а не скрываются временным кодом;
- приложение проходит startup/shutdown/recovery paths;
- документация запуска/конфигурации соответствует фактической реализации;
- управляющий чат принимает stage как `done`.

---

## 14. Явно отложено за пределы обязательного v1

Если не появится новое требование, следующие темы не должны незаметно расширять scope этапов 1–8:

- production stereo distance и полноценный `capture_id` pairing;
- camera-to-turret rotational extrinsic;
- backlash compensation;
- automatic Overview↔Stereo target handoff;
- D-filter как обязательная часть PID;
- per-camera Aiming parameter expansion;
- future latest-live-frame UI path;
- режим «Самая быстрая»;
- automatic Vision load adaptation;
- production RS485 half-duplex migration (transceiver, DE/RE или auto-direction, termination/biasing, isolation/reference и turnaround measurements);
- generic STM32 hardware event queue до появления конкретного event use case;
- автоматический config migration mechanism за пределами `schema-version = 1` до появления реальной schema v2.

Они возвращаются в план только отдельным решением после v1 measurements/requirements.

## 15. Карта открытых вопросов `problems.md` → этапы

| Вопрос | Этап |
|---|---|
| #1 RS485 half-duplex | deferred post-v1 / отдельная RS485 migration |
| #2 capture_id / stereo pairing | deferred |
| #3 DistanceResult stereo lifecycle | deferred / 5 только manual path |
| #7 runtime config edge cases | 2 config infrastructure; processor-specific Vision part closed before 5; 6 target-loss-timeout checkpoint |
| #8 camera reconnect | 5 |
| #10 worker lifecycle | 1 foundation + Vision/integration stages + 8 final; Turret part closed before 3 |
| #11 partial failures | 7 + 8 |
| #13 logging | 1 foundation + UI/integration + 8 final; Turret part closed before 3 |
| #14 future STM32 events | deferred |
| #15 tracking timing budget | 8 после измерений |
| #16 backlash | deferred |
| #17 camera-to-turret extrinsic | deferred |
| #18 target handoff | deferred |
| #22 latest-live-frame UI | deferred |
| #23 D-filter / PID tuning | 8 tuning, D-filter deferred unless evidence |
| #24 per-camera Aiming parameters | deferred |
| #25 diagnostics/profiling | 8 minimal field diagnostics |
| #26 «Самая быстрая» | deferred |
| #27 Vision load adaptation | deferred |

## 16. Handoff рабочего этапа в управляющий чат

Не нужен старый большой stage report. Для приёмки достаточно короткого handoff:

```text
Stage:
Proposed status: review
Implemented:
Tests/checks actually run:
Hardware/user checks still pending:
Architecture deviations/conflicts: none | ...
Open issues/deferred items:
New test files created and why existing owners were insufficient: none | ...
```

Плюс обычный delivery block из корневого `AGENTS.md` с update ZIP и действиями пользователя.

Управляющий чат после проверки:

- принимает этап → `done`;
- возвращает на доработку → `in-progress`;
- фиксирует зависимость/решение → `blocked` до устранения;
- обновляет этот master plan только по фактическому состоянию.

## 17. Пакет для отдельного рабочего чата

Обычно достаточно:

1. актуального project/update snapshot, из которого можно восстановить текущий working tree;
2. корневого `AGENTS.md` и локальных `AGENTS.md`, уже находящихся в project tree;
3. этого `implementation-plan.md` или явно выделенного раздела текущего этапа;
4. relevant architecture/module docs из того же текущего tree.

Не использовать старые stage reports вместо актуальных project files и нормативной документации.
