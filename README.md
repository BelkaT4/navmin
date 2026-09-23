# NavMin

NavMin — программно-аппаратный проект BelkaT4 для приёма видеопотоков, Vision/Aiming и управления турелью через STM32.

## Текущий статус

Текущий baseline подготовлен к pre-hardware этапу на уровне software integration: production camera/serial transport paths имеют software diagnostics, backend-aware preflight и session evidence, а полностью виртуальная offline rehearsal принята.

При этом реальное hardware validation ещё не выполнено: Stage 5/6/7 остаются на уровне принятого prototype minimum, Stage 8 не завершён. Фактические параметры механики, DM860/STEP-DIR, physical limits и real-camera end-to-end latency должны подтверждаться на последующих hardware checkpoints.

## Обычный запуск

Production entrypoint использует real RTP cameras и real STM32 serial path:

```bash
python -m navmin --preflight-only
python -m navmin
```

`--preflight-only` проверяет статические prerequisites выбранных real backends и создаёт session evidence, но не запускает workers/UI и не выполняет hardware protocol probe.

`config.json` и `calibration/*.json` являются локальными site-specific inputs и не входят в repository baseline. Обычный GUI startup при missing/invalid input может по явному выбору оператора сохранить существующие файлы в timestamped `.bak`, создать safe recovery defaults и завершиться для ручной проверки параметров перед повторным запуском.

## Полностью виртуальный diagnostic

Для software-only проверки production transport boundaries без камер и STM32:

```bash
python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret pty \
  --synthetic-inputs \
  --preflight-only
```

Полный VIRTUAL запуск:

```bash
python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret pty \
  --synthetic-inputs
```

Подробный порядок offline rehearsal, STM32-only, cameras-only и eventual REAL hardware run описан в [Offline Hardware Runbook](docs/user/offline-hardware-runbook.md).

## Документация

- [Быстрый старт](docs/user/quick-start.md)
- [Offline Hardware Runbook](docs/user/offline-hardware-runbook.md)
- [Как начать работу над проектом](docs/dev/getting-started.md)
- [Архитектура](docs/dev/architecture/overview.md)
- [Master implementation plan](docs/dev/implementation-plan.md)
- [Опубликованная документация](https://belkat4.github.io/navmin/)

## Разработка и проверки

Перед изменениями прочитайте [AGENTS.md](AGENTS.md) и релевантные module/architecture docs. Основное Python-окружение проекта — `uv`, Python 3.13.

Базовая regression-проверка:

```bash
uv run --extra dev ruff check \
  src \
  tests \
  main.py \
  tools/run_software_smoke.py \
  tools/run_software_soak.py \
  tools/run_localhost_rtp_diagnostic.py \
  tools/run_pty_stm32_diagnostic.py \
  tools/run_diagnostic_app.py

uv run --extra dev pytest
uv run --extra dev python -m compileall -q src main.py tests tools
```

Полные hardware-dependent проверки выполняются отдельно и не входят в обычный test suite.
