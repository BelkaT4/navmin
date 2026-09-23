# Быстрый старт

## Обычный запуск

По умолчанию NavMin читает:

```text
config.json
calibration/overview.json
calibration/stereo.json
```

и запускает только production RTP cameras + production serial path к STM32:

```bash
python -m navmin
```

Эквивалентно можно использовать root entrypoint:

```bash
python main.py
```

Пути можно задать явно:

```bash
python -m navmin \
  --config config.json \
  --overview-calibration calibration/overview.json \
  --stereo-calibration calibration/stereo.json
```

Normal launcher не делает fallback на software STM32. Если в config установлен
`turret.emulate-stm32=true`, запуск завершается с явной ошибкой. До создания
`ApplicationRuntime` launcher выполняет backend-aware static preflight. Mandatory
`FAIL` завершает запуск с code `2` до UI/workers; `WARN` запуск не блокирует.

### Локальные настройки и восстановление

`config.json`, `calibration/overview.json` и `calibration/stereo.json` — локальные файлы конкретной установки; они не должны коммититься и не входят в project snapshot.

Если при обычном `python -m navmin` один из этих файлов отсутствует или не проходит strict validation, до запуска workers показывается точная причина и предлагается:

```text
Закрыть программу
Восстановить настройки по умолчанию
```

При восстановлении существующие файлы сначала сохраняются с timestamp **местного системного времени**, например:

```text
config.json-20260923-101530.bak
overview.json-20260923-101530.bak
stereo.json-20260923-101530.bak
```

При коллизии используется общий suffix `-01`, `-02`, ...; backup никогда не перезаписывается. Если backup хотя бы одного существующего файла не удалось создать, восстановление не заменяет исходный набор.

Созданные defaults — только безопасная точка восстановления: serial path специально требует ручной настройки, PID = 0, motion limits низкие, calibration нейтральная 320×240. После восстановления NavMin **не продолжает запуск**. Проверьте serial port, steps/rev, направления/limits и реальные calibration files, затем запустите программу снова.

`--preflight-only` и diagnostic launcher диалог восстановления не показывают: они печатают точную input error и завершаются.

## Diagnostic launcher

Diagnostic launcher использует тот же `ApplicationRuntime`, но позволяет явно
заменить только внешние hardware endpoints. Все три выбора обязательны:

```bash
python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret pty \
  --synthetic-inputs
```

`--synthetic-inputs` — явный полностью software-only профиль для проверки launcher/UI
без локальных `config.json` и calibration files. Он создаёт встроенные 320×240
diagnostic config/calibrations в памяти и разрешён только для сочетания
`localhost + localhost + pty`. Для любых mixed/real сценариев config и calibration
files по-прежнему обязательны.

Допустимые значения:

```text
--overview     real | localhost
--stereo-left  real | localhost
--turret       real | pty
```

Например, реальные камеры вместе с PTY controller:

```bash
python tools/run_diagnostic_app.py \
  --overview real \
  --stereo-left real \
  --turret pty
```

`localhost` запускает внешний synthetic RTP/JPEG sender, но receiver приложения
остаётся production `GStreamerRtpJpegSource`. `pty` запускает внешний software
STM32 endpoint через Linux PTY, но приложение продолжает использовать production
`SerialTransport`/pyserial. Diagnostic preflight выполняется до запуска localhost
sender и PTY service, поэтому failed preflight не оставляет внешние diagnostic
endpoints.

Оба режима создают отдельный session directory в `logs/` (или в parent directory,
заданном через `--log-dir`):

```text
logs/
└── navmin-<mode>-YYYYMMDD-HHMMSS-ffffff/
    ├── runtime.log
    ├── manifest.json
    ├── preflight.json
    └── inputs/
        ├── effective-config.json
        ├── overview-calibration.json
        ├── stereo-calibration.json
        └── source-hashes.json
```

Normal пишет в session file на уровне INFO, diagnostic — DEBUG. `runtime.log`
rotates при 10 MiB и хранит до пяти backup files; старые session directories
автоматически не удаляются. Manifest содержит outcome запуска, выбранные backends и
ограниченный набор runtime/platform evidence. Environment variables, credentials и
полный filesystem inventory не собираются. Effective inputs всё равно могут содержать
операционные параметры, поэтому session directory не следует считать автоматически
безопасным для публичной публикации.

Для проверки окружения без запуска UI/workers/endpoints оба launcher поддерживают:

```bash
python -m navmin --preflight-only

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret pty \
  --synthetic-inputs \
  --preflight-only
```

`--preflight-only` всё равно создаёт session directory, `runtime.log`,
`manifest.json` и `preflight.json`. `PASS`/`WARN` возвращают `0`; `FAIL` возвращает
`2` и manifest status `preflight-failed`. Static preflight проверяет только
prerequisites выбранных backends: он не ждёт RTP packets, не ping'ует Raspberry Pi,
не открывает real serial device и не делает STM32 protocol probe.

Для автономной репетиции, четырёх рекомендуемых operator profiles и controlled hardware-day gates см. [Offline Hardware Runbook](./offline-hardware-runbook.md).
