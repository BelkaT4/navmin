# Архитектурные решения и отвергнутые альтернативы

Этот документ фиксирует **почему** архитектура v1 устроена именно так. Нормативные контракты остаются в `overview.md`, `contracts.md`, `configuration.md`, `serial-protocol.md` и документах модулей; здесь не дублируются все поля и wire-format, а сохраняются причины ключевых решений и существенные отвергнутые альтернативы.

Если этот документ расходится с нормативным контрактом, источником истины является соответствующий постоянный документ архитектуры или `.mmd`-диаграмма. Расхождение нужно исправить, а не трактовать как второй вариант контракта.

## 1. `.mmd` — источник истины для архитектурных диаграмм

### Решение

Mermaid-файлы в `dev/diagrams/*.mmd` являются исходниками архитектурных диаграмм. PNG/SVG — только экспортированные представления.

### Почему

Текстовый источник можно проверять diff'ом, синхронизировать с контрактами и автоматически валидировать. Экспортированное изображение может быть устаревшим относительно исходника.

### Отвергнутая альтернатива

**Использовать PNG как равноправный источник архитектуры.** Отклонено, потому что это создаёт два источника истины и позволяет незаметно расходиться диаграмме и её исходнику.

---

## 2. Working frame всегда имеет исправленную геометрию

### Решение

`FramePacket.image` всегда содержит working frame:

```text
Overview: raw → undistort → working frame
Stereo Left/Right: raw → rectify → working frame
```

Все публичные pixel coordinates относятся именно к working frame. При отсутствующей/несовместимой calibration pipeline не считается ready; raw frame не публикуется как прозрачный fallback.

### Почему

Один публичный геометрический контекст исключает ситуацию, когда bbox, click, aim point и lead относятся к разным изображениям или calibration model.

### Отвергнутые альтернативы

- **Публиковать raw при ошибке calibration.** Отклонено: downstream не сможет безопасно понять, в какой системе координат находится кадр.
- **Автоматически масштабировать/crop calibration под другое разрешение в v1.** Отклонено: добавляет скрытое геометрическое преобразование и усложняет проверку точности.

---

## 3. Camera generation + `CameraSessionStarted` barrier + общий `CameraSessionGate`

### Решение

Каждый новый запуск camera pipeline получает новую `generation`. Перед любыми данными этой generation публикуется ordered/barrier:

```text
CameraSessionStarted(camera, generation, camera_model)
```

Core и UI используют один main-thread `CameraSessionGate` и не принимают camera-derived данные новой generation до принятия barrier.

### Почему

Queued callbacks и latest-state могут пережить restart worker. Один `track_id` или `frame_id` недостаточен, чтобы отличить старую session от новой. `CameraModel` привязывается к той же generation, чтобы кадр и геометрия менялись атомарно.

### Отвергнутые альтернативы

- **Только очищать очереди/state при restart.** Отклонено: late queued message всё ещё может прийти после очистки.
- **Проверять generation только в Core.** Отклонено: UI получает high-bandwidth camera data напрямую и тоже должен быть защищён.
- **Вводить отдельный `geometry_revision`.** Не нужен в v1: `CameraModel` меняется вместе с camera session и уже защищён `generation`.

---

## 4. UI использует `main_camera + preview`, а target selection существует только в TRACKING

### Решение

Overview и Stereo Left отображаются одновременно: одна камера main, вторая preview. Selection разрешён только на main image и только после подтверждённого `TRACKING`.

Допустимые состояния:

```text
RELATIVE:
    selected_target = None

TRACKING:
    selected_target = None | TargetRef
```

При `TRACKING → RELATIVE` selection очищается; при возврате в TRACKING цель выбирается заново.

### Почему

В RELATIVE target identity не участвует в управлении: пользователь кликает в точку, Aiming один раз вычисляет относительное перемещение. Сохранение target в RELATIVE создаёт двусмысленное состояние «цель выбрана, но не управляет турелью» и позволяет target loss/deselect случайно влиять на manual motion.

### Отвергнутые альтернативы

- **`selected_targets` отдельно для Overview и Stereo Left.** Отклонено: selection относится к текущему пользовательскому tracking context, а не к каждой камере независимо.
- **`RELATIVE + selected_target`.** Отклонено: не даёт обязательной функции и усложняет state machine.
- **Автоматически переносить selection при swap.** Отклонено: разные VisionProcessors имеют независимые `generation + track_id`; handoff — отдельная будущая задача.

---

## 5. Applied `control_mode` принадлежит Turret, PID reset — тоже Turret responsibility

### Решение

Authoritative применённый `control_mode` хранит Turret Controller и публикует через `TurretState`. Core может иметь request на смену режима, но не вторую authoritative applied-копию.

PID также полностью принадлежит Turret Controller. Core не посылает отдельный `PID_RESET`; Turret сам сбрасывает PID по наблюдаемым control boundaries: вход/выход TRACKING, смена `TargetRef`, invalidation `TrackingError`, длинный gap, Emergency, MOTOR_OFF, reconnect/recovery.

### Почему

Компонент, который владеет состоянием, должен владеть и правилами его reset/apply. Иначе один факт приходится синхронизировать между несколькими модулями и появляются гонки «Core уже считает mode новым, а Turret ещё нет».

### Отвергнутые альтернативы

- **Хранить applied mode одновременно в Core и Turret.** Отклонено из-за двух источников истины.
- **Отдельная команда/канал `PID_RESET` из Core.** Отклонено: протекает внутренняя реализация PID через межмодульный контракт.

---

## 6. `SET_VELOCITY(0,0)` — обычный velocity setpoint

### Решение

Нулевой setpoint не несёт скрытой application-семантики. `SET_VELOCITY(vx, vy)` всегда означает установку target velocity; `(0,0)` — просто target velocity zero.

### Почему

В TRACKING PID штатно может выдать `(0,0)`, если ошибка уже нулевая. StopMotion, смена режима, target clear и PID reset должны определяться явными state transitions, а не конкретными числовыми значениями команды.

### Отвергнутая альтернатива

**Трактовать `(0,0)` как специальный Stop.** Отклонено: обычный PID output стал бы неотличим от control operation высокого уровня.

---

## 7. `MOVE_RELATIVE` не имеет completion lifecycle

### Решение

`MOVE_RELATIVE → OK` означает только, что STM32 приняла новую relative target. В v1 отсутствуют:

```text
MOVE_COMPLETED
command_id у MOVE_RELATIVE
Completed / Cancelled / Failed lifecycle
PING polling ради completion
```

### Почему

Без encoder/limit feedback окончание генерации STEP не доказывает фактического достижения требуемого положения. Для текущей логики момент естественного завершения не нужен, а lifecycle создавал большое количество races: completion vs cancellation, event buffering, Emergency recovery и command-id reuse.

### Отвергнутые альтернативы

- **`MOVE_COMPLETED(command_id)`.** Отклонено: дорого по сложности и не подтверждает физическую позицию.
- **`GET_MOTION_STATE`/polling.** Отклонено: возвращает ту же сложность в другой форме без обязательной функции.

---

## 8. Один latest-only `pending_motion`; already-sent ordinary request считается committed

### Решение

В Turret есть один внутренний slot:

```text
pending_motion = MoveRelativeCommand | AxisVelocitySetpoint | None
```

В confirmed RELATIVE допустим только `MoveRelativeCommand`, в TRACKING — только velocity setpoint. Новый ещё не отправленный intent заменяет старый.

Control boundaries инвалидируют только **неотправленный** `pending_motion`. Если ordinary request уже физически отправлен, он считается committed и завершает свой обычный response/retry cycle; следующий normal control action выполняется после него.

### Почему

Один slot структурно исключает невозможную комбинацию «одновременно ждут relative и velocity». Правило committed сохраняет строгую one-in-flight transport model и не требует resync для normal Stop/mode transition/MOTOR_OFF.

### Отвергнутые альтернативы

- **Два независимых motion slots.** Отклонено: позволяют выразить недопустимое состояние и требуют дополнительной очистки между режимами.
- **FIFO motion-команд.** Отклонено: старые motion intents создают latency и не должны воспроизводиться позже.
- **`motion_generation`.** Отклонено как дублирование защиты при одном Turret worker, одном UART owner и одном latest-only slot.
- **Прерывать retries любого already-sent request при normal Stop.** Отклонено: делает normal control boundary причиной transport uncertainty/resync. Для срочного прерывания есть Emergency.

---

## 9. Normal Stop и Emergency разделены

### Решение

Normal StopMotion использует acceleration-limited velocity control. Emergency Stop немедленно прекращает STEP generation без acceleration limit.

Persistent emergency latch отсутствует. Emergency — действие, а не долгоживущий режим; старое motion state уничтожается и не восстанавливается автоматически.

### Почему

Обычная остановка должна сохранять механику и плавность; аварийная — минимизировать время до прекращения импульсов. Persistent latch был бы оправдан только при отдельном требовании «никакого движения до явного re-arm», которого у v1 нет.

### Отвергнутые альтернативы

- **Один и тот же Stop для normal и emergency.** Отклонено из-за разных требований к acceleration limit.
- **PC+STM32 persistent emergency latch.** Отклонено: добавляет состояние, re-arm и recovery cases без существующего safety requirement.
- **Использовать `MOTOR_ON` как reset emergency.** Отклонено: смешивает enable driver и safety-state semantics.

---

## 10. Один общий `REQUEST_ID`, response correlation по `REQUEST_ID + COMMAND_CODE`

### Решение

Все новые ordinary и special transactions используют один cyclic `uint16 REQUEST_ID`. Exact retry использует тот же ID. Response echo'ит `REQUEST_ID + COMMAND_CODE`, и PC принимает response только при совпадении обоих полей.

`EMERGENCY_STOP` special по `COMMAND_CODE`, а не по конкретному значению ID.

### Почему

Один счётчик проще двух independent counters и не требует резервировать `FFFF`. `COMMAND_CODE` в response устраняет неоднозначность между late responses разных command types с одинаковым ID.

### Отвергнутые альтернативы

- **Фиксированный `FFFF` для special commands.** Отклонено: не различает разные transactions одной special-команды и создаёт ненужное исключение из общего ID-space.
- **Отдельные normal/special counters.** Отклонено: два механизма корреляции вместо одного.
- **Special result codes (`EMERGENCY_OK`, `RESET_OK`, ...).** Отклонено: смешивает identity команды и результат выполнения.

---

## 11. `EMERGENCY_STOP` является sequence-resync boundary; отдельный `RESET_ID` удалён

### Решение

`EMERGENCY_STOP(ID=N)` может быть обработан независимо от текущего ordinary `expected_request_id`. После успешной обработки STM32 очищает motion state и устанавливает:

```text
expected_request_id = N + 1 mod 65536
```

PC после подтверждённого response использует тот же следующий ID. Exact retries Emergency повторяют тот же `N`.

Отдельной `RESET_ID` command в v1 нет.

### Почему

При transport uncertainty безопасное восстановление всё равно должно уничтожать старое motion state. Emergency одновременно решает обе нужные задачи: приводит motion к известному состоянию и пересинхронизирует sequence. Дополнительный reset после успешного Emergency ничего не добавляет.

### Отвергнутые альтернативы

- **`RESET_ID` как обязательный шаг recovery.** Отклонено: дублирует resync и создаёт отдельные retry/response/event semantics.
- **Resync без очистки motion state.** Не найден обязательный v1 use case; при неизвестном transport state сохранение старого motion скорее нежелательно.

---

## 12. Recovery после transport loss возвращает систему в безопасный baseline

### Решение

После признанной потери transport старая session не продолжается. После восстановления физической связи/baud выполняется безопасная последовательность, начинающаяся с Emergency resync, затем `MOTOR_OFF` и replay полного STM32 config snapshot. Auto `MOTOR_ON` отсутствует.

Уже принятый заранее ограниченный `MOVE_RELATIVE` при внезапной потере связи может закончиться. Velocity control останавливается своим watchdog.

### Почему

Полный безопасный resync проще и надёжнее попыток угадать, какие последние commands STM32 успела применить. Отдельный communication watchdog для relative move не нужен, потому что размер одной команды уже ограничен и её завершение после обрыва считается допустимым.

### Отвергнутые альтернативы

- **Продолжать старую session после исчерпания retries.** Отклонено: фактическое состояние STM32 может быть неизвестно.
- **Общий communication watchdog, отменяющий `MOVE_RELATIVE`.** Отклонено для v1 как лишний механизм при допустимом завершении ограниченного move.
- **Автоматически восстанавливать MOTOR_ON.** Отклонено: после recovery движение не должно возобновляться без нового явного действия пользователя.

---

## 13. `SET_CONFIG` — dynamic atomic full snapshot

### Решение

STM32-параметры max speed, acceleration и velocity watchdog применяются полным атомарным `SET_CONFIG` snapshot даже во время движения. Control loop/ISR видит либо старый snapshot целиком, либо новый целиком.

Изменение watchdog timeout не refresh'ит watchdog: отсчёт остаётся от последнего успешно принятого `SET_VELOCITY`.

### Почему

Для этих параметров нет аппаратной необходимости ждать окончания relative move. Dynamic snapshot устраняет safe-point, который после отказа от `MOVE_COMPLETED` пришлось бы определять отдельным механизмом.

### Отвергнутые альтернативы

- **Применять `SET_CONFIG` только в motion safe-point.** Отклонено: требует знать/опрашивать окончание relative motion.
- **Считать изменение watchdog timeout новым velocity activity.** Отклонено: config update не является управляющим setpoint и не должен искусственно продлевать движение.

---

## 14. Runtime `SET_BAUDRATE` сохраняется, но только при motors OFF

### Решение

STM32 стартует на фиксированном startup baud, а рабочую скорость можно менять runtime через `SET_BAUDRATE`. Firmware принимает команду только при фактическом motors OFF. При неопределённом результате PC ищет фактический baud по known candidates и затем выполняет обычный safe recovery.

### Почему

Есть реальное требование тестировать разные скорости UART без перепрошивки STM32. Motors OFF делают потерю связи во время baud transition безопасной относительно движения.

### Отвергнутая альтернатива

**Убрать runtime baud switching и менять baud только прошивкой.** Отклонено, потому что это мешает испытаниям физического RS485-канала.

---

## 15. Mechanical conversion settings — restart-only

### Решение

`invert`, `full_steps_per_revolution`, `microstep_divider` и `max-relative-move-deg` в v1 не меняют active Turret mechanical conversion или relative-move safety envelope на лету. Сохранённое изменение требует Turret/application restart.

### Почему

Runtime изменение преобразования degrees ↔ steps или relative-move safety envelope посреди active motion создаёт safe-point и consistency cases, которых текущие требования не требуют.

### Отвергнутая альтернатива

**Dynamic apply mechanical conversion или `max-relative-move-deg`.** Отклонено как ненужная сложность v1.

---

## 16. Supervisor не входит в v1

### Решение

Отдельный Supervisor не реализуется, пока нет конкретной уникальной обязанности, которую нельзя корректно обработать owner-модулем.

Camera reconnect принадлежит Vision, UART recovery — Turret, UI получает typed states, diagnostics идут через logging.

### Почему

Абстрактный Supervisor без собственного domain responsibility превращается в второй центр orchestration и дублирует lifecycle logic модулей.

### Отвергнутая альтернатива

**Создать Supervisor заранее для heartbeat/restart policy.** Отклонено до появления конкретного failure mode, требующего межмодульной координации.

---

## 17. Generic `SystemEvent` bus и STM32 event FIFO не проектируются заранее

### Решение

В v1 runtime machine-readable состояние передаётся typed contracts (`CameraStatus`, `TurretState`, etc.), а диагностика — logging. Generic `SystemEvent/SystemEventType/FIFO` не вводятся заранее.

Wire-format может сохранять reserved `EVENTS` section, но пока обязательных hardware events нет, event queue/capacity/overflow policy не проектируются.

### Почему

Инфраструктура без реального producer/consumer requirement обслуживала бы только саму себя. Конкретный event contract лучше проектировать вместе с первым настоящим hardware event, например limit switch или driver fault.

### Отвергнутые альтернативы

- **Generic event bus с самого начала.** Отклонено как premature abstraction.
- **Заранее определить STM32 FIFO capacity/overflow.** Отклонено: требования к доставке неизвестны до появления реального event type.

---

## 18. Stereo distance можно отложить; manual distance остаётся полноценным источником v1

### Решение

Distance Provider поддерживает `manual` и `stereo`. Полная `capture_id` pairing/resync логика не блокирует первую реализацию: manual distance остаётся допустимым источником.

### Почему

Stereo pairing — отдельная сложная задача синхронизации двух camera sessions. Она не должна блокировать проверку остальной цепочки Vision → Aiming → Turret.

### Отвергнутая альтернатива

**Сделать готовый stereo pairing обязательным до первой интеграции.** Отклонено: связывает независимые этапы разработки и увеличивает стоимость раннего прототипа.

---

## 19. Camera-to-turret rotational extrinsic отложен до mechanical tests

### Решение

В v1 предполагается достаточная параллельность camera/turret axes; постоянный boresight компенсируется aim point. Полный `R_camera_to_turret` вводится только если mechanical tests покажут необходимость.

### Почему

Без измеренной ошибки дополнительная extrinsic calibration создаёт процедуру и storage, которые могут оказаться ненужными.

### Отвергнутая альтернатива

**Обязательная extrinsic calibration с первой версии.** Отклонено до появления измеренной потребности.

---

## 20. `config.json` v1 валидируется строго и не чинится молча

### Решение

`config.json` обязателен для normal startup. Отсутствующий файл, malformed JSON, unknown fields, missing required fields, invalid values и unsupported `schema-version` дают явную config/startup error. Только заранее документированные optional fields получают in-memory defaults; loader не дописывает их в файл автоматически. Invalid runtime update не заменяет последний полностью валидный snapshot.

Schema v1 использует `schema-version = 1`; automatic migration не проектируется до появления реальной schema v2.

### Почему

Конфигурация содержит camera sources и параметры, влияющие на физическое движение Turret. Silent fallback/default repair может скрыть опечатку или аппаратно неверное значение и запустить систему не с теми настройками, которые считает активными пользователь. Строгая schema также делает typo в имени поля наблюдаемой ошибкой вместо молчаливого ignore.

### Отвергнутые альтернативы

- **Автоматически создавать полный default config при отсутствующем файле.** Отклонено: безопасные универсальные hardware defaults неизвестны, а примерные числа в документации не являются аппаратными пределами.
- **Игнорировать unknown fields.** Отклонено: опечатка превращается в скрытый fallback/неприменённую настройку.
- **Исправлять invalid values defaults и продолжать startup.** Отклонено: скрывает проблему и создаёт неочевидный effective config.
- **Проектировать migration framework заранее.** Отклонено до появления второй реальной schema; требования migration пока неизвестны.

---

## 21. Runtime PID config reset'ит changed gains по оси, а уменьшение output limit только clamp'ит I-term

### Решение

PID config остаётся dynamic и полностью принадлежит Turret Controller. Если во время TRACKING меняется любой `Kp/Ki/Kd` конкретной оси, Controller полностью сбрасывает PID state только этой оси перед обработкой следующего нового `TrackingError`. Первый sample после reset использует новые gains и остаётся P-only: `I=0`, `D=0`.

Изменение только application-side output limit (`max-speed-*-deg-s`) full PID reset не вызывает. При уменьшении limit сохранённый I-term соответствующей оси сразу clamp'ится в новый диапазон `±max_speed`; при увеличении limit текущий I-term сохраняется без масштабирования. Если gains и output limit одной оси меняются в одном config revision, gain-change reset имеет приоритет. Config update сам по себе не создаёт motion command и не меняет `control_mode`.

### Почему

Сохранение integral/derivative history после замены gains связывает новый controller tuning со state, накопленным при другой динамике, и делает переход плохо предсказуемым. Полный per-axis reset при смене gains даёт простой детерминированный boundary и согласуется с уже существующими Turret-owned PID reset rules.

Для одного лишь уменьшения output limit полный reset избыточен: проблема состоит только в том, что накопленный I-term может оказаться вне нового допустимого диапазона. Clamp устраняет это состояние, сохраняя полезную историю controller там, где сами gains не менялись.

### Отвергнутые альтернативы

- **Сохранять PID state без изменений при смене `Kp/Ki/Kd`.** Отклонено: history была накоплена при других gains и может дать неочевидный transient на следующем sample.
- **Масштабировать integral state при смене `Ki`, чтобы сохранить прежний I-output.** Отклонено для v1: это скрытая трансформация внутреннего state, усложняет сочетание с изменениями `Kp/Kd` и не даёт преимущества перед явным reset boundary.
- **Полностью reset'ить PID при любом изменении max speed/output limit.** Отклонено: при неизменных gains достаточно clamp I-term к новому limit; потеря всей controller history не нужна.
- **Посылать внешний `PID_RESET` из Core/Config Manager.** Отклонено по уже принятому ownership: PID state и его apply/reset policy принадлежат Turret Controller.

---

## 22. Turret reconnect бесконечный с capped backoff; отсутствие STM32 не является fatal `ERROR`

### Решение

Turret сам владеет automatic serial reconnect. Transport loss признаётся после исчерпания ordinary/Emergency retries, physical I/O/disconnect failure или неуспешного bounded baud-recovery attempt. Matching command-level error сам по себе не означает physical transport loss; `INVALID_REQUEST_ID` переводит normal traffic в Emergency-based sequence resync.

Reconnect cycles не имеют конечного лимита попыток. Между ними используется interruptible capped exponential backoff `0.25 → 0.5 → 1 → 2 → 2 ... s`, который сбрасывается после полного `READY`. Обычный baud search проверяет без дубликатов `last-known → desired → 9600`; после uncertain `SET_BAUDRATE` — `new → old → 9600`.

Пока owner способен продолжать reconnect, отсутствие STM32/serial device отражается как `DISCONNECTED/CONNECTING`, а не `ERROR`. `ERROR` зарезервирован для действительно невосстановимой локальной ошибки/invariant failure, при которой automatic recovery нельзя корректно продолжить.

### Почему

Физическое отсутствие устройства — ожидаемый recoverable condition, а не причина завершать приложение. Бесконечный reconnect делает unplug/replug штатным сценарием, а capped backoff не создаёт busy loop. Явный порядок baud candidates делает recovery детерминированным и покрывает hardware reset на 9600 и lost response после `SET_BAUDRATE`.

### Отвергнутые альтернативы

- **Конечное число reconnect attempts с переходом в ERROR.** Отклонено: временно отсутствующий STM32 не должен требовать restart приложения.
- **Постоянный короткий polling без backoff.** Отклонено: создаёт лишнюю нагрузку и log spam.
- **Случайный/неопределённый порядок baud candidates.** Отклонено: усложняет тестирование и диагностику uncertain baud transition.

---

## 23. Turret worker останавливается cooperative через `StopToken`; отдельный Supervisor не нужен

### Решение

Turret worker создаётся/останавливается application orchestration. Serial waits должны быть bounded или cancellable, reconnect/backoff ожидается через `StopToken`, а shutdown выполняет `request_stop() → bounded join() → is_alive()` check. Numeric join timeout остаётся внутренним implementation tuning и не добавляется в `config.json`. Незавершившийся после bounded join worker является явной shutdown error и логируется; успешный shutdown в таком случае не объявляется.

### Почему

Это использует уже существующий Foundation lifecycle contract и не создаёт второй orchestration owner. Interruptible waits гарантируют, что бесконечный auto-reconnect не мешает остановке приложения.

### Отвергнутые альтернативы

- **Supervisor только ради остановки/restart Turret worker.** Отклонено как дублирование owner/application orchestration.
- **Unbounded blocking UART read или `sleep()` в backoff.** Отклонено: `request_stop()` не смог бы гарантированно прервать ожидание.
- **Новый user-configurable join timeout.** Отклонено: это implementation tuning без пользовательской domain-семантики.

---

## 24. Turret diagnostics остаются logging, без queue/event infrastructure в v1

### Решение

Turret использует существующий стандартный Python logging bootstrap. `QueueHandler/QueueListener` и generic diagnostic event bus в v1 не добавляются. INFO предназначен для lifecycle/connection/recovery/baud boundaries и значимых failures; `REQUEST_ID`, command/retry/candidate details доступны для более подробной диагностики. High-rate PID samples и обычные velocity setpoints не логируются на INFO.

### Почему

Typed state уже несёт machine-readable runtime состояние, а logging нужен для transient diagnostics. Queue-based logging без измеренной blocking/contention проблемы добавляет отдельную concurrency infrastructure, а INFO на частоте tracking loop создавал бы шум и потенциально влиял на timing.

### Отвергнутые альтернативы

- **`QueueHandler/QueueListener` с первой реализации.** Отклонено до profiling/измеренной contention problem.
- **Логировать каждый PID/setpoint на INFO.** Отклонено из-за высокой частоты, шума и риска влияния на timing.
- **Generic machine-readable diagnostic event bus.** Отклонено: обязательное runtime состояние уже выражено typed contracts.

---


## 25. Serial timing reconnect, deferred desired baud и restart-only emulation

### Решение

`response-timeout-ms`, `max-retries` и `inter-request-delay-ms` применяются через новую Turret session boundary после reconnect. `emulate-stm32` не переключается runtime и требует application restart. Desired baud остаётся latest-only: при confirmed motors OFF переход допустим, при motors ON/UNKNOWN он откладывается без hidden `MOTOR_OFF`; перед последующим `MOTOR_ON` pending baud применяется первым.

### Почему

Serial timing является свойством bounded transaction/session и не должен меняться посреди committed in-flight exchange. Runtime real ↔ fake switching создало бы второй transport lifecycle внутри одного процесса без v1-потребности. Отложенный baud сохраняет safety ownership: config change не должен сам выключать motors.

### Отвергнутые альтернативы

- **Mutate timing активной session.** Отклонено из-за неоднозначной семантики текущей transaction.
- **Config baud update автоматически делает MOTOR_OFF.** Отклонено: motor state меняется только явной control operation/recovery.
- **Runtime real ↔ fake switch.** Отклонено как лишний lifecycle mechanism для v1.

---

## 26. Первая STM32 реализация использует UART; RS485 переносится в отдельную post-v1 migration

### Решение

Первый firmware target — `STM32F103C8T6` с STM32 HAL/CubeIDE-compatible project. Physical transport первой реализации — обычный full-duplex `USART3` (`PB10 TX`, `PB11 RX`, startup `9600 8N1`).

Текущий binary protocol считается окончательным относительно physical transport: UART и будущий RS485 не получают разных frame/request/retry/state semantics. RS485 half-duplex, выбор transceiver, `DE/RE` или auto-direction, termination/biasing, electrical reference/isolation и turnaround measurements переносятся в отдельную migration после обязательной первой реализации.

Старый firmware-проект используется только как hardware reference для MCU/pin mapping/polarity/timer baseline. Его старый serial protocol не является compatibility target и не переносится.

### Почему

Старая турель уже работала через обычный UART, а доступа к физической STM32/RS485 сборке сейчас нет. Protocol/control firmware можно реализовать и полноценно тестировать без смешивания новых RS485 electrical/timing unknowns с parser, request sequence, Emergency, motor planner и watchdog. Изоляция physical byte transport делает последующий переход на RS485 ограниченным hardware change.

### Отвергнутые альтернативы

- **Блокировать Stage 4 до выбора и проверки RS485 transceiver.** Отклонено: это связывает независимые protocol/control задачи с недоступным сейчас железом.
- **Сделать отдельные UART и RS485 версии протокола.** Отклонено: physical medium не требует второго framing/request/state contract.
- **Перенести старый UART protocol из legacy firmware.** Отклонено: authoritative v1 protocol уже определён в `serial-protocol.md`, а legacy firmware нужен только как hardware reference.

## 27. `VisionProcessor` остаётся единственной processing boundary; baseline v1 использует legacy 1.4/1.1 + общий `SimpleTracker`

### Решение

Публичная архитектура не разбивает Vision processing на обязательные `Detector` и `Tracker` модули. Конкретный `VisionProcessor` может быть detector+tracker, integrated algorithm или другой схемой, пока он соблюдает `TrackedObject` contract.

Для baseline v1 реализуются:

```text
Legacy14VisionProcessor   # default
Legacy11VisionProcessor
```

Они используют detector-алгоритмы legacy Variant 1.4 / Variant 1.1 и один общий внутренний `SimpleTracker`. Legacy tracker целиком не переносится. `SimpleTracker` использует короткую robust motion history, внутреннее prediction и one-to-one association; полноценный appearance identity/reacquisition остаётся будущей реализацией `AdvancedVisionProcessor`.

Processor-specific tuning в v1 является implementation detail: owner-local `settings.py` с module-level constants. Эти параметры не входят в `config.json`, не persistятся Config Manager и не показываются UI.

### Почему

Такой boundary позволяет сначала получить простой измеримый baseline на двух detector variants, сравнивая их на одном tracker, а затем заменить внутренний algorithm на более сложный VT11-подобный processor без изменения Core, UI, Aiming или публичных Vision contracts.

Локальные tuning constants не являются пользовательской политикой системы. Если преждевременно включить их в строгий `config.json`, экспериментальные thresholds становятся долгоживущим публичным schema/API и требуют validation, persistence и runtime apply semantics без доказанной необходимости.

### Отвергнутые альтернативы

- **Сделать `Detector` и `Tracker` отдельными архитектурными модулями.** Отклонено: это преждевременно фиксирует внутреннюю структуру всех будущих `VisionProcessor` implementations.
- **Перенести legacy tracker как есть.** Отклонено: он несёт историческую application/reacquisition complexity, которая не нужна baseline multi-object tracking и мешает чисто сравнивать detector 1.1/1.4.
- **Сразу использовать полный VT11 pipeline как обязательный Stage 5 processor.** Отклонено как blocker первой реализации; advanced identity/reacquisition должен сравниваться с простым baseline после первых измерений.
- **Хранить все detector/tracker thresholds в общем `config.json` или UI.** Отклонено: внутренний tuning не должен становиться публичным config contract без реальной operator/runtime потребности.

## 28. UI v1 prioritizes a fixed operational bar and minimal main/preview interaction

### Решение

Основной UI стартует fullscreen. Overview и Stereo Left образуют pair `main + preview`; preview — небольшое окно в правом нижнем углу области видео, swap выполняется click'ом по preview или его action-icon.

Постоянная нижняя operational bar содержит fixed-size controls для `RELATIVE / TRACKING`, подтверждённого motor state/control, connection state и крупного `EMERGENCY` в правом нижнем углу. Motor toggle выполняется одним click без confirmation dialog. Dedicated ordinary Stop button в первом prototype не показывается.

В TRACKING левый click по bbox выбирает target, empty click снимает selection, overlap разрешается ближайшим к click центром среди bbox, содержащих точку. В RELATIVE левый click по main image выполняет click-to-move. Preview click только выполняет swap.

Baseline overlay показывает bbox и selection highlight без постоянных ID/distance/velocity/age labels. Stereo Right доступен только как отдельный diagnostic view. Recording запускается из верхнего menu bar; активная запись обозначается в левом верхнем углу main view мигающим красным кругом и статической белой надписью `Запись`.

### Почему

Первая цель проекта — быстро получить runnable prototype для относительного движения и TRACKING. Поэтому постоянно видимыми остаются только действия и states, нужные оператору во время управления, а редко используемые функции уходят в menu/diagnostics. Fixed geometry исключает смещение кнопок при изменении текста состояния, а Emergency остаётся быстро достижимым.

### Отвергнутые альтернативы

- **Постоянные Motor/Recording/Settings buttons в отдельной панели.** Отклонено как лишнее загромождение; motor остаётся в operational bar, recording/settings доступны из menu.
- **Dedicated ordinary Stop button в первом prototype.** Отложен: архитектурный `StopMotion` остаётся, но отдельная пользовательская кнопка пока не нужна.
- **Выбор nearest object вне bbox.** Отклонён для v1 как неочевидное действие; click должен попадать в отображаемый bbox.
- **Постоянные diagnostic labels возле каждого bbox.** Отклонены ради читаемого основного изображения; расширенная диагностика может включаться отдельно.
- **Stereo Right как обычная main/preview camera.** Отклонено: её роль остаётся diagnostic/stereo и не должна менять обычный two-camera interaction contract.

## 29. Camera transport v1 — RTP/JPEG через GStreamer; Overview correction — OpenCV fisheye без silent scaling

### Решение

Production camera source v1 использует фактически подтверждённый sender/receiver path RTP/JPEG over UDP. PC receiver декодирует через GStreamer в raw BGR и хранит только freshest frame; geometric correction остаётся единственной responsibility `VisionPipeline`. `appsink` работает bounded/latest-oriented (`drop=true`, `sync=false`, prototype `max-buffers=1`).

Overview calibration schema v1 трактуется как OpenCV fisheye: `D` содержит ровно четыре коэффициента, maps строятся через `cv2.fisheye.initUndistortRectifyMap`, а исправленная geometry использует `new_camera_matrix`. Source size обязан точно совпадать с calibration size; автоматическое scaling `K/new_camera_matrix` не выполняется.

### Почему

Этот RTP/JPEG sender/receiver path уже реально работает на camera setup и GStreamer appsink даёт прямой low-latency latest-frame boundary без отдельной display loop architecture. Fisheye API обязателен, потому что фактическая Overview calibration была получена через OpenCV fisheye model; обычный pinhole `initUndistortRectifyMap` описывает другую camera model. Exact-size validation сохраняет явную геометрическую ошибку вместо скрытого изменения calibration.

### Отвергнутые альтернативы

- **Выполнять fisheye correction внутри camera source.** Отклонено: это дублировало бы correction path и смешало transport с geometry.
- **Автоматически масштабировать `K`/`new_camera_matrix` под любой decoded size.** Отклонено: silent scaling скрывает geometry mismatch; NavMin требует exact calibration resolution.
- **Использовать Raspberry Pi IP как `CameraConfig.address` receiver-side correlation.** Отклонено: UDP sender уже направляет stream на PC; receiver bind address и local listen port являются достаточной source boundary.
- **Копировать process-global `GLib.MainLoop` из reference viewer.** Отклонено: appsink callback и polling bus достаточны owner-local source и не создают новый application-global manager.

---

## 30. UI использует revision-aware QTimer pump примерно 60 Hz

### Решение

Prototype UI в Qt main thread каждые `16 ms` сначала полностью обрабатывает доступные lossless `CameraSessionStarted` barriers, затем читает freshest revisions существующих latest-state containers Vision, CameraStatus и TurretState. Успешно принятые revisions больше не обрабатываются; result, опередивший свой barrier в пределах tick, остаётся retryable до следующего tick.

Per-frame queued Qt signals не используются.

### Почему

Это минимальная concurrency model поверх уже существующих `LatestValue`, `InvalidatableLatest` и `CameraSessionBarrierChannel`. Pump добавляет не более примерно `16 ms` polling latency, но не превращает camera rate `15–20 FPS` в `60 FPS` image conversions: без новой revision tick почти ничего не делает. Lossless barriers сохраняют session ordering, а latest payloads естественно coalesce'ятся до freshest value без Qt event backlog.

### Отвергнутые альтернативы

- **Queued Qt signal на каждый frame/state update.** Отклонено: producer может накопить queued callbacks и тем самым вернуть video latency backlog поверх latest-only Vision boundary.
- **Frame FIFO в UI.** Отклонено: UI нужен freshest accepted state, а не последовательное воспроизведение устаревших кадров.
- **Сразу вводить coalesced event-driven wakeup.** Отложено до измеренной необходимости: оно сложнее, а `16 ms` pump уже даёт малую bounded polling latency при простой проверяемой ordering model.

---

## Как использовать этот документ при реализации

При разработке нового модуля сначала нужно следовать нормативным контрактам соответствующего документа. Если возникает желание вернуть ранее удалённый механизм, полезно проверить этот журнал: часто механизм был удалён не случайно, а потому что более простой инвариант закрывает тот же failure case.

Новая запись здесь нужна, если решение:

- существенно меняет границу ответственности модулей;
- выбирает один из нескольких реалистичных архитектурных вариантов;
- удаляет ранее существовавший механизм;
- вводит ограничение, которое будущий разработчик может попытаться «исправить», не зная причины.

Мелкие implementation details сюда не переносятся.
