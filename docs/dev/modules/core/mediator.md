# Core: Mediator

`Mediator` — центральный координирующий компонент Core. Он связывает UI, Vision, Aiming и Turret, но не выполняет их специализированные вычисления.

## Основное состояние

Mediator хранит минимум:

```text
main_camera: CameraRole                 # OVERVIEW | STEREO_LEFT
selected_target: TargetRef | None       # только в TRACKING
latest VisionResult per displayed camera
latest DistanceResult | None
latest TurretState
```

`main_camera` по умолчанию берётся из `ui.default-camera`. Автоматического переключения main camera при отказе камеры нет.

Applied `control_mode` читается из `latest TurretState`. Core не хранит вторую independent authoritative копию mode; pending user request на смену mode — это operation intent, а не applied state.

## Camera sessions и `CameraSessionGate`

Vision перед любыми данными новой generation публикует ordered/barrier:

```text
CameraSessionStarted(camera, generation, camera_model)
```

В main thread общий `CameraSessionGate` хранит:

```text
accepted_generation[camera]
camera_model[camera]
```

Core и UI используют один gate. Session event должен быть принят consumer раньше любого camera-derived data той же generation.

После новой session Core очищает camera-derived state предыдущей generation. Если invalid target существовал в TRACKING, Core очищает `selected_target`, lead/DistanceResult и `TrackingError`, после чего запрашивает normal StopMotion у Turret. Core не вызывает PID reset напрямую.

В RELATIVE target отсутствует, поэтому camera restart/swap сам по себе не отменяет уже сформированный `MoveRelativeCommand`.

## Selection

Инвариант:

```text
RELATIVE:
    selected_target = None

TRACKING:
    selected_target = None | TargetRef
```

Selection разрешён только если latest confirmed:

```text
TurretState.control_mode == TRACKING
```

и одновременно:

```text
requested_target.camera == main_camera
requested_target.generation == CameraSessionGate.accepted_generation[main_camera]
requested_target.track_id существует в latest VisionResult main_camera
```

Если объект уже отсутствует, selection request отклоняется. В RELATIVE любой selection request отклоняется независимо от bbox на экране.

## Swap main / preview

Swap меняет `main_camera` только явным user action.

Если applied mode TRACKING:

```text
selected_target → None
lead / TrackingError → clear/invalidate
related DistanceResult → invalidate
request normal StopMotion
```

Applied mode остаётся TRACKING.

Если applied mode RELATIVE, swap не отменяет уже успешно сформированный manual `MoveRelativeCommand`. Новый click после swap должен пройти обычную generation/session validation новой main camera.

## Временная и окончательная потеря target

Если selected track временно отсутствует, но `target-lost-timeout-ms` не истёк:

- `TargetRef` сохраняется;
- Aiming не создаёт новый `TrackingError`;
- новый lead не создаётся;
- новый tracking motion intent не формируется;
- STM32 velocity watchdog независимо ведёт target velocity к zero, если updates не возобновились.

При final loss или explicit deselect:

```text
selected_target → None
lead / TrackingError → clear/invalidate
DistanceResult старой цели → invalidate
request normal StopMotion
```

Applied mode остаётся TRACKING.

PID принадлежит Turret Controller. Core сообщает только domain boundary через state/action; отдельного `PID_RESET` нет.

## Смена цели A → B

Новая цель принимается только если B существует в latest `VisionResult`.

После accepted switch:

```text
selected_target = B
old lead / DistanceResult invalidate
new TrackingError(B) computed from fresh data
```

Turret Controller обнаруживает смену `TargetRef`, инвалидирует старый неотправленный tracking motion, reset'ит PID и рассчитывает новый setpoint. Обязательный промежуточный `SET_VELOCITY(0,0)` не вставляется.

Если B невалидна, A остаётся selected.

## `RELATIVE → TRACKING`

User request передаётся Turret Controller. До подтверждения mode:

- selection запрещён;
- click-to-move больше не принимается как новая normal action для переходящей operation;
- Core ждёт `TurretState.control_mode == TRACKING`.

При motors ON Turret выполняет обязательную ownership boundary:

```text
invalidate old unsent pending_motion
→ SET_VELOCITY(0,0) → OK
→ apply TRACKING
→ publish TurretState(TRACKING)
```

При motors OFF physical motion невозможен, поэтому Turret может применить mode без zero handshake.

После подтверждения получаем:

```text
TRACKING + selected_target = None
```

Пользователь выбирает target заново.

## `TRACKING → RELATIVE`

Core сразу очищает:

```text
selected_target → None
lead / TrackingError → clear/invalidate
DistanceResult → invalidate
```

и передаёт mode-change intent Turret Controller.

При motors ON Turret сначала завершает уже committed ordinary transaction, инвалидирует unsent pending motion, выполняет `SET_VELOCITY(0,0) → OK`, reset'ит PID и только затем публикует `TurretState.control_mode = RELATIVE`.

Если UART transition не завершился, applied mode может остаться TRACKING, но target уже отсутствует: это безопасное `TRACKING + no target`.

## Click-to-move в RELATIVE

Click-to-move разрешён только при confirmed:

```text
TurretState.control_mode == RELATIVE
```

Порядок:

```text
UI click on main VisionResult
→ validate camera + generation + frame context
→ CameraModel / Aiming
→ MoveRelativeCommand(delta_x_deg, delta_y_deg)
→ Turret
```

Если camera session стала неактуальной **до** validation/расчёта, click отклоняется.

После того как `MoveRelativeCommand` успешно сформирован, последующие camera swap/restart/stale/new generation его не отменяют: он уже является самостоятельным manual motion intent.

## StopMotion

### TRACKING

Core сначала очищает target-dependent state:

```text
selected_target → None
lead / TrackingError → clear/invalidate
DistanceResult → invalidate
```

затем запрашивает Turret `StopMotion`.

### RELATIVE

Core только передаёт `StopMotion` Turret. Selection в RELATIVE отсутствует.

Turret Controller отвечает за invalidation неотправленного `pending_motion`, PID state и physical zero-setpoint semantics.

## Emergency Stop

При user Emergency Core немедленно очищает application tracking state:

```text
selected_target → None
lead / TrackingError → clear/invalidate
DistanceResult → invalidate
```

и передаёт `EMERGENCY_STOP` в Turret.

Transport blocking/retry/resync — ответственность Turret HAL. Core не хранит отдельный публичный Emergency flag и не управляет request IDs.

После successful Emergency applied mode автоматически не меняется; старое motion state не восстанавливается.


## Потеря связи с Turret

Если latest `TurretState.connection_state` перестаёт быть READY, Core не пытается сохранять active tracking intent:

```text
selected_target → None
lead / TrackingError → clear/invalidate
DistanceResult target-dependent → invalidate
```

Applied mode принадлежит Turret Controller и может сохраниться через reconnect, но после recovery tracking не возобновляется автоматически: при TRACKING пользователь выбирает target заново, а motors остаются OFF до нового явного `MOTOR_ON`.

## Состояние для UI

Core предоставляет UI coordinated application state:

- `main_camera`;
- `selected_target`;
- latest valid distance;
- camera-derived state, прошедший `CameraSessionGate`;
- latest `TurretState` как authoritative applied motor/mode/connection state.

Диагностические сообщения идут в logging, а runtime state передаётся typed contracts. Generic event bus заранее не вводится.
