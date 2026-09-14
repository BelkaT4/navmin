# Core

`Core` — координирующий модуль приложения. Он связывает `Vision`, `Turret` и `UI`, хранит application state и прикладную coordination logic.

![Диаграмма модуля Core](../../diagrams/core-diagram.png)

## Состав

```text
Core
├── Mediator
├── Aiming
└── Config Manager
```

Все компоненты Core выполняются в главном Qt-потоке. Общий `CameraSessionGate` также находится в main thread и используется Core и UI.

- [Mediator](./mediator.md) — `main_camera`, selection, camera sessions, mode/control intents и coordination;
- [Aiming](./aiming.md) — `CameraModel`, aim point, lead и угловая ошибка;
- [Config Manager](./config-manager.md) — typed config snapshots и revision.

Camera reconnect принадлежит Vision, UART recovery — Turret, runtime состояние передаётся typed state, диагностика — logging. Отдельный orchestration-компонент без собственной ответственности заранее не вводится.

## Роль Core

Core — Mediator, а не Broker. Он понимает семантику `main_camera`, выбранной цели и camera session, но не выполняет Vision, PID или UART.

```text
CameraSessionStarted / VisionResult / DistanceResult / TurretState
                              ↓
                    CameraSessionGate + Core
                    ├─ main_camera
                    ├─ selected_target (TRACKING only)
                    ├─ Aiming
                    └─ coordination state
                              ↓
                           Turret / UI
```

Applied `TurretControlMode` не дублируется в Core: authoritative значение приходит через `TurretState.control_mode`. Core может иметь только pending user request на смену mode.

## Границы

Core не должен:

- получать/декодировать видеопоток;
- выполнять undistort/rectify или `VisionProcessor`;
- вычислять stereo distance;
- выполнять или reset'ить PID напрямую;
- управлять UART/RS485;
- формировать binary STM32 packets;
- реализовывать camera/UART reconnect owner logic.

Core должен:

- хранить/координировать `main_camera`;
- хранить одну `selected_target`, но только в подтверждённом TRACKING;
- принимать camera sessions через `CameraSessionGate`;
- использовать `TurretState` как источник applied mode/motor/connection state;
- вызывать Aiming;
- создавать `MoveRelativeCommand` после validation click/session;
- координировать `StopMotion`, Emergency и mode-change intents;
- управлять общей конфигурацией и startup/shutdown приложения.

## Связанные документы

- [Общая архитектура](../../architecture/overview.md)
- [Контракты](../../architecture/contracts.md)
- [Конфигурация](../../architecture/configuration.md)
- [Открытые вопросы](../../architecture/problems.md)
