# Уточнения архитектуры

Этот файл используется как **временный журнал новых архитектурных решений**, которые уже согласованы, но ещё не перенесены в постоянные документы.

Правило работы:

1. после обсуждения решение добавляется сюда вместе с причиной выбора и отклонёнными существенными альтернативами;
2. периодически нормативная часть переносится в постоянные документы архитектуры/модулей, а устойчивое обоснование — в [`decisions.md`](./decisions.md);
3. после проверки перенесённые записи удаляются отсюда;
4. `.mmd` остаются источником истины для архитектурных диаграмм; PNG/SVG — только экспорт.

## Текущие уточнения

На момент синхронизации 2026-09-15 все согласованные решения перенесены в постоянную документацию и `.mmd`; rationale ключевых решений собран в [`decisions.md`](./decisions.md).

Неперенесённых архитектурных решений сейчас нет.

Новые утверждённые изменения следует добавлять ниже до следующего цикла синхронизации обязательно с кратким обоснованием выбора и указанием существенных отвергнутых альтернатив.

## 2026-11-17 — VERBOSE high-rate diagnostic logging использует один async session log

### Решение

Существующие пользовательские режимы launcher `normal | verbose` сохраняются;
третий logging mode не добавляется. Все новые high-rate diagnostic events пишутся
только при `verbose`. Console остаётся INFO, а один session `runtime.log` принимает
обычные сообщения и structured DEBUG events согласно
[`../diagnostics/verbose-logging-contract.md`](../diagnostics/verbose-logging-contract.md).

В VERBOSE session-file path вводится bounded non-blocking logging queue и отдельный
writer/listener thread. Producer threads Vision/Core/Turret/UART/IMU не выполняют
disk I/O и не блокируются при переполнении: diagnostic record разрешено потерять,
но потеря считается и явно отражается в trace/manifest integrity. Initial queue
capacity — 8192 records. VERBOSE rotation — 100 MiB на файл и 10 backup-файлов;
NORMAL сохраняет текущие 10 MiB и 5 backup-файлов.

Это уточнение **частично заменяет решение 24** из `decisions.md` только в части
VERBOSE session-file transport. Отказ от generic diagnostic event bus и запрет
high-rate INFO остаются в силе. Queue infrastructure теперь имеет конкретную
необходимость: согласованный per-frame/per-control/per-UART detailed trace не должен
искажать измеряемый control timing синхронной записью на диск.

### Почему

Для диагностики расходящейся tracking oscillation и построения измеренной модели
нужно loss-visible связать frame → Vision → Aiming → estimator → PID/FF/prediction
→ UART, а далее STM32/IMU. При частоте камер порядка 20 FPS, control traffic и
будущем IMU/MCU telemetry объём DEBUG событий становится существенно выше того,
для чего принималось старое решение «без QueueHandler/QueueListener до измеренной
необходимости». Синхронный `RotatingFileHandler` в producer thread сам мог бы стать
источником latency/jitter и испортить эксперимент.

Один logical `runtime.log` сохраняет удобный общий временной поток и текущий формат
`timestamp level thread logger: message`; источник определяется logger name, а тип
нового события — стабильным `EVENT_NAME` в начале message. Разделение на несколько
runtime event-файлов не даёт доказанного performance выигрыша и усложняет
сопоставление одного прогона.

### Отклонённые существенные альтернативы

- **Третий «trace» mode.** Отклонён: существующий VERBOSE является явным операторским
  выбором подробной диагностики.
- **Отдельные `vision.log`, `tracking.log`, `uart.log`, `imu.log`.** Отклонены:
  дополнительный routing/file ownership не нужен для текущего анализа; один
  session stream проще сопоставлять и фильтровать по logger/event.
- **Синхронно писать каждый detailed event существующим file handler.** Отклонено:
  disk I/O/contention не должны попадать в измеряемый control path.
- **Писать high-rate trace на INFO.** Отклонено: NORMAL и console должны остаться
  лёгкими.
- **Generic machine-readable runtime event bus.** Отклонён: detailed trace остаётся
  diagnostics output; typed runtime state и module ownership не меняются.


## 2026-11-17 — Stage 2 PC-side VERBOSE evidence реализован

Реализация следует контракту `verbose-logging-contract.md`: frame/vision, Aiming,
estimator, command-model, PID/FF/prediction, control decisions и фактические UART
TX/RX/retry boundaries связаны через frame key, `control_id`, `request_id` и
`connection_epoch`. Camera clock дополнительно публикует structured
`CLOCK_SNAPSHOT`; примерно раз в секунду публикуется `PERF_SNAPSHOT`.

При завершении VERBOSE run logging writer после bounded drain напрямую фиксирует
`RUN_INTEGRITY`, поэтому итоговая сводка не зависит от наличия места в самой
logging queue. Ключевые integrity counters также атомарно копируются в
`manifest.json.logging_integrity`. `manifest.mode` сохраняет прежний смысл режима
приложения; новый `manifest.logging_mode` отдельно показывает `normal | verbose`.

PC instrumentation не утверждает фактическое исполнение механики: STM32 STEP/DIR,
IMU и внешний эталон остаются следующими отдельными этапами.
