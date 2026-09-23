# Offline Hardware Runbook

Этот документ — operator-facing procedure для **BelkaT4 / NavMin** перед автономной
репетицией и последующим controlled hardware day. Он связывает уже существующие
launchers, preflight и session artifacts в воспроизводимый порядок действий.

Runbook **не добавляет новый runtime mode** и не заменяет архитектурные документы.
Технически diagnostic launcher по-прежнему допускает независимый выбор backend для
Overview, Stereo Left и Turret. Для обычной работы ниже выделены только четыре
рекомендуемых профиля; остальные combinations предназначены для узкой диагностики.

## 1. Что считается evidence

В этом runbook строго различаются:

- **implemented in software** — capability присутствует в коде;
- **verified in software tests** — behavior доказан автоматическими tests;
- **checked by static preflight** — prerequisite проверен без запуска runtime;
- **observed in software diagnostic** — behavior наблюдался через diagnostic path;
- **measured on hardware** — значение измерено на физическом стенде;
- **candidate** — значение/идея предложены для будущей проверки;
- **assumed** — предположение, не являющееся evidence;
- **not yet validated** — проверка ещё не выполнена.

Software capability или `PREFLIGHT PASS` не превращаются автоматически в hardware
proof.

## 2. Offline rule

Offline rehearsal и hardware day должны быть выполнимы при отключённой сети.
Машина, Python environment, system runtime и документация должны быть подготовлены
**до** начала rehearsal.

Во время rehearsal/hardware day не использовать как способ исправления среды:

```text
apt install
pip install
uv sync, если он требует network
curl
wget
git pull
```

Если обязательная dependency отсутствует при отключённой сети:

```text
REHEARSAL FAIL
```

Нужные datasheets/manuals, включая exact DM860 documentation для установленной
ревизии, следует скачать или распечатать заранее.

## 3. Preflight простыми словами

`--preflight-only` означает:

> Проверить, готов ли компьютер к выбранному запуску, но не запускать саму систему.

В зависимости от выбранных backends static preflight может проверять:

- Python/runtime dependencies;
- PyQt6;
- production GStreamer receiver и required elements;
- localhost sender prerequisites;
- config/calibration inputs;
- UDP bind availability;
- `pyserial`;
- существование и read/write access real serial path;
- Linux PTY capability.

Preflight **не доказывает**, что:

- real cameras уже передают RTP;
- STM32 отвечает по NavMin protocol;
- motors/mechanics безопасны;
- camera image quality или latency приемлемы;
- mechanical scale откалиброван;
- hardware test пройден.

Для real Turret preflight может проверить `pyserial`, наличие serial path и права,
но не открывает real serial port, не переключает DTR/RTS, не отправляет STM32 `PING`
и не двигает motors.

Для real cameras preflight проверяет static receiver readiness, config/calibration и
UDP endpoint availability, но не ждёт RTP traffic, не ping'ует Raspberry Pi, не
подключается по SSH и не оценивает изображение.

Technical exit semantics:

```text
PASS → exit 0
WARN → exit 0
FAIL → exit 2
```

Operator policy для rehearsal/hardware day строже:

```text
PASS
→ можно переходить к следующему gate

WARN
→ остановиться и понять причину;
  необъяснённый WARN не считать readiness evidence

FAIL
→ STOP
```

Mandatory preflight failure происходит до запуска:

```text
ApplicationRuntime
QApplication
CameraWorker
TurretWorker
localhost RTP sender
PTY emulator runtime
```

## 4. Официальные operator profiles

Diagnostic launcher технически поддерживает восемь комбинаций:

```text
--overview     real | localhost
--stereo-left  real | localhost
--turret       real | pty
```

Но основными operator profiles считаются ровно четыре:

| Profile | Overview | Stereo Left | Turret | Inputs | Physical boundary | Назначение |
| --- | --- | --- | --- | --- | --- | --- |
| **VIRTUAL** | localhost | localhost | pty | synthetic | none | Offline rehearsal и software-only readiness |
| **STM32** | localhost | localhost | real | files | STM32/turret only | Controlled physical STM32 preparation |
| **CAMERAS** | real | real | pty | files | cameras only | Real RTP camera bring-up без real Turret |
| **REAL** | real | real | real | files | cameras + STM32/turret | Eventual combined hardware run |

`--synthetic-inputs` используется только в **VIRTUAL**. Current launcher запрещает
его для mixed/real profiles.

### 4.1 VIRTUAL

Boundary:

```text
Overview       localhost
Stereo Left    localhost
Turret         pty
inputs         synthetic
physical HW    none
```

Preflight:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret pty \
  --synthetic-inputs \
  --preflight-only
```

Launch:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret pty \
  --synthetic-inputs
```

Это основной profile для offline rehearsal. Localhost cameras проходят через
production `GStreamerRtpJpegSource`, а PTY Turret — через production
`SerialTransport`/pyserial.

При visual smoke обе synthetic camera показывают `SOURCE HH:MM:SS.mmm` и
`FRAME n`, а нижняя operational bar — `NOW HH:MM:SS.mmm`:

- сравнить `SOURCE` и `NOW` как приблизительный текущий visual lag;
- подтвердить, что `FRAME` постоянно увеличивается;
- в течение короткого наблюдения убедиться, что gap `SOURCE` → `NOW` визуально
  не растёт и bbox движущейся цели продолжает обновляться.

Это localhost-only проверка в clock domain одного PC, а не измерение latency
реальных Raspberry Pi cameras.

### 4.2 STM32

Boundary:

```text
Overview       localhost
Stereo Left    localhost
Turret         real
physical HW    STM32 / turret only
```

Термин **STM32-only physical boundary** не означает запуск приложения без cameras
или UI. Единственный физический endpoint здесь — STM32/turret; camera inputs
заменены localhost RTP/JPEG diagnostics.

Используются current file-backed inputs:

```text
config.json
calibration/overview.json
calibration/stereo.json
```

Это локальные site-specific inputs, они не входят в repository baseline. Если обычный GUI startup обнаруживает missing/invalid input, оператор может закрыть программу либо явно восстановить safe defaults; существующие файлы получают backup вида `<filename>-YYYYMMDD-HHMMSS[-NN].bak` по местному системному времени. Recovery не продолжает запуск автоматически и не заменяет on-site настройку serial/mechanics/calibration. Preflight/diagnostic paths остаются неинтерактивными.

Preflight:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret real \
  --preflight-only
```

Launch:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret real
```

Отдельный Turret-only production runner для этого не нужен.

### 4.3 CAMERAS

Boundary:

```text
Overview       real
Stereo Left    real
Turret         pty
physical HW    cameras only
```

Используются file-backed config/calibrations.

Preflight:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview real \
  --stereo-left real \
  --turret pty \
  --preflight-only
```

Launch:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview real \
  --stereo-left real \
  --turret pty
```

Это проверяет physical cameras, production RTP receiver, real Vision/UI/Core и PTY
Turret boundary без физической турели.

### 4.4 REAL

Boundary:

```text
Overview       real
Stereo Left    real
Turret         real
physical HW    all
```

Production preflight:

```bash
cd ~/NAV\ MIN

python -m navmin --preflight-only
```

Простая интерпретация для оператора:

> Проверь, готов ли компьютер запустить реальную систему, но ничего физически не
> запускай.

Production launch:

```bash
cd ~/NAV\ MIN

python -m navmin
```

Explicit diagnostic equivalent остаётся допустимым, когда полезно явно видеть
выбранные backends:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview real \
  --stereo-left real \
  --turret real \
  --preflight-only
```

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview real \
  --stereo-left real \
  --turret real
```

`python -m navmin` — короткий production entrypoint. Explicit diagnostic command
полезен для operator diagnostics, но не является отдельной архитектурой.

### 4.5 On-site configuration and calibration changes

Эта секция относится к file-backed profiles **STM32**, **CAMERAS** и **REAL**, а
также к advanced mixed combinations. Официальный **VIRTUAL** profile намеренно
использует built-in `--synthetic-inputs`; не превращать его в hardware-day config
profile.

Current normal и diagnostic launchers оба принимают одни и те же source-input
options:

```text
--config <path>
--overview-calibration <path>
--stereo-calibration <path>
```

По умолчанию это:

```text
config.json
calibration/overview.json
calibration/stereo.json
```

На hardware day предпочтительно не перезаписывать единственную рабочую baseline
копию. Можно заранее сделать отдельные файлы, например:

```bash
cd ~/NAV\ MIN

cp config.json config-hardware-day.json
cp calibration/overview.json calibration/overview-hardware-day.json
cp calibration/stereo.json calibration/stereo-hardware-day.json
```

Имена здесь только operator recommendation, а не новый project naming contract.

#### Безопасный workflow изменения inputs

1. Полностью закрыть NavMin.
2. Сохранить предыдущие рабочие `config`/calibration files или работать с их копиями.
3. Изменить только нужные **source input files**.
4. Не редактировать `logs/navmin-.../inputs/*.json`: это generated evidence, а не
   source of truth для следующего запуска.
5. Запустить тот же нужный profile с `--preflight-only` и теми же `--config`,
   `--overview-calibration`, `--stereo-calibration` paths, которые планируются для
   real launch.
6. При `PASS` проверить выбранные source paths в `runtime.log`, calibration sizes в
   `preflight.json` и при необходимости сами source JSON через `python -m json.tool`.
   `preflight-only` намеренно не создаёт `inputs/`, потому что runtime ещё не
   запускался.
7. Запустить тот же profile с теми же input paths без `--preflight-only`.
8. После clean shutdown сохранить весь session directory как evidence.
9. Hardware-critical изменения дополнительно перенести в [hardware measurement
   worksheet](#7-hardware-measurement-worksheet).

`PASS` здесь означает только static readiness выбранных inputs/backends. Для real
camera preflight не видит фактические RTP pixels, а для real Turret не открывает
serial port и не проверяет protocol response.

#### Что именно менять в `config.json`

Точные persisted JSON paths для наиболее частых on-site изменений:

| Задача | Current field | Operational meaning |
| --- | --- | --- |
| Overview UDP port | `vision.cameras.overview.port` | local UDP listen port на PC; для localhost diagnostics тот же port используется diagnostic sender/receiver pair |
| Stereo Left UDP port | `vision.cameras.stereo-left.port` | local UDP listen port на PC; для localhost diagnostics тот же port используется diagnostic sender/receiver pair |
| Overview local bind address | `vision.cameras.overview.address` | local PC address для production receiver при backend `real` |
| Stereo Left local bind address | `vision.cameras.stereo-left.address` | local PC address для production receiver при backend `real` |
| Real serial device | `turret.serial.port` | например фактически обнаруженный `/dev/ttyUSB0` или `/dev/ttyACM0` |
| Desired Turret baudrate | `turret.serial.baudrate` | желаемая рабочая скорость serial link |

Полная current schema и apply policy описаны в
[Configuration](../dev/architecture/configuration.md). Не добавлять неизвестные
JSON keys: parser schema-v1 строгий.

**Camera `address` не является IP Raspberry Pi.** Это адрес, на котором PC receiver
делает local UDP bind/listen. Обычный portable listen может использовать
`0.0.0.0`; конкретный address, если выбран, должен принадлежать PC. IP назначения,
куда Raspberry Pi/source sender отправляет RTP, задаётся **на стороне sender** и
должен указывать на подходящий адрес PC. В current NavMin config отдельного поля
`Raspberry Pi source IP` нет.

Для diagnostic backend `localhost` launcher намеренно заменяет camera `address` на
`127.0.0.1` и включает RTP receiver. Поэтому изменение
`vision.cameras.<camera>.address` не меняет localhost diagnostic bind address;
`port` при этом остаётся взят из выбранного config file.

#### Изменение real stream resolution

В current `config.json` нет отдельного camera width/height field. Размер, для
которого calibration действительна, хранится в calibration JSON как:

```text
image_width
image_height
```

Для **real camera** недостаточно поменять только Raspberry Pi/source sender:

```text
new sender resolution
→ calibration для точно этого resolution
→ preflight с выбранной calibration
→ launch
```

NavMin не масштабирует calibration автоматически. Для real RTP preflight может
подтвердить, что typed calibration загружена, но не получает реальные кадры и
поэтому не может заранее проверить фактический stream resolution. Если decoded
frame имеет другой размер, production correction path выдаёт hard
`WorkingFrameError` вида `source resolution ... does not match calibration ...`;
такой mismatch нельзя принимать как рабочий режим.

Для localhost diagnostic sender размер, наоборот, строится из выбранной calibration.
Официальный VIRTUAL profile использует built-in synthetic calibration/stream
`320x240`; не менять его как способ имитировать новую physical calibration.

#### Overview distortion и Stereo calibration

Overview source file выбирается через:

```text
--overview-calibration <path>
```

Stereo calibration source file выбирается через:

```text
--stereo-calibration <path>
```

На hardware day предпочтительно использовать заранее подготовленный calibration
file, полученный calibration procedure, а не вручную "подкручивать" матрицы или
коэффициенты до визуально удобного результата. В частности, при новом real stream
resolution нужна calibration, реально рассчитанная для этого exact resolution.

#### Custom file-backed inputs в официальных profiles

Ниже примеры с условными hardware-day filenames. Использовать один и тот же набор
paths для preflight и последующего launch.

**STM32 — localhost cameras + real Turret:**

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret real \
  --config config-hardware-day.json \
  --overview-calibration calibration/overview-hardware-day.json \
  --stereo-calibration calibration/stereo-hardware-day.json \
  --preflight-only
```

После `PASS` тот же запуск без последней строки:

```bash
python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret real \
  --config config-hardware-day.json \
  --overview-calibration calibration/overview-hardware-day.json \
  --stereo-calibration calibration/stereo-hardware-day.json
```

**CAMERAS — real cameras + PTY Turret:**

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview real \
  --stereo-left real \
  --turret pty \
  --config config-hardware-day.json \
  --overview-calibration calibration/overview-hardware-day.json \
  --stereo-calibration calibration/stereo-hardware-day.json \
  --preflight-only
```

После `PASS` повторить ту же команду без `--preflight-only`.

**REAL — production entrypoint:**

```bash
cd ~/NAV\ MIN

python -m navmin \
  --config config-hardware-day.json \
  --overview-calibration calibration/overview-hardware-day.json \
  --stereo-calibration calibration/stereo-hardware-day.json \
  --preflight-only
```

После `PASS`:

```bash
python -m navmin \
  --config config-hardware-day.json \
  --overview-calibration calibration/overview-hardware-day.json \
  --stereo-calibration calibration/stereo-hardware-day.json
```

Те же three file options поддерживаются explicit diagnostic REAL command и
advanced mixed combinations. Новый `--profile` flag для этого не нужен.

#### Serial device и baudrate

Для real Turret изменить:

```text
turret.serial.port
```

на фактический device path, например `/dev/ttyUSB0` или `/dev/ttyACM0`. Real-Turret
preflight проверит наличие path и read/write access, но **не** откроет serial port,
не переключит DTR/RTS и не докажет, что STM32 отвечает.

Желаемая рабочая скорость задаётся:

```text
turret.serial.baudrate
```

Current schema разрешает только:

```text
9600
19200
38400
57600
115200
```

Это desired working baud; current STM32 startup baseline остаётся `9600`, после
чего production Turret path при необходимости выполняет controlled
`SET_BAUDRATE`. Static preflight не является доказательством успешного baud
transition на конкретном physical STM32.

#### Другие existing config settings и hardware-critical граница

Другие уже существующие settings также можно менять в отдельной копии `config`
согласно current schema и затем проверять тем же `--preflight-only` flow. Однако
сам факт, что JSON field редактируем, **не** означает, что значение допустимо
выбирать "на глаз".

Hardware-critical минимум:

```text
turret.axes.x.full-steps-per-revolution
turret.axes.y.full-steps-per-revolution
turret.axes.x.microstep-divider
turret.axes.y.microstep-divider
turret.axes.x.invert
turret.axes.y.invert
turret.axes.x.max-relative-move-deg
turret.axes.y.max-relative-move-deg
turret.stm32.max-speed-x-deg-s
turret.stm32.max-speed-y-deg-s
turret.stm32.acceleration-x-deg-s2
turret.stm32.acceleration-y-deg-s2
physical-limit assumptions
```

`full-steps-per-revolution * microstep-divider` участвует в PC mechanical
conversion. `invert`, mechanics scale и `max-relative-move-deg` являются
restart-only settings в current v1. Speed/acceleration участвуют в STM32 config.
Ни один из этих facts не заменяет measurement/safety gates раздела B.

Physical-limit assumptions особенно не являются config permission: current
pre-hardware baseline всё ещё не имеет accepted active physical-limit enforcement
для unrestricted movement.

#### Evidence фактически использованных inputs

После **successful full launcher session** generated evidence содержит именно
фактически использованный typed snapshot:

```text
inputs/effective-config.json
inputs/overview-calibration.json
inputs/stereo-calibration.json
inputs/source-hashes.json
```

`source-hashes.json` сохраняет source path и SHA-256 для file-backed config и обеих
calibrations. Поэтому после run можно установить, какие source files были выбраны и
какие effective values реально дошли до application composition. Для diagnostic
profile `effective-config.json` также отражает runtime endpoint overrides, например
PTY serial path или localhost bind address.

Не использовать generated `inputs/*.json` как следующий editable source: сохранять
весь session directory как evidence, а изменения вносить в исходные files,
указанные launcher flags.

## 5. Advanced troubleshooting combinations

Оставшиеся mixed combinations доступны, но не считаются отдельными user modes.
Использовать их только для локализации конкретного subsystem, например одной camera:

```text
real Overview + localhost Stereo Left + PTY
localhost Overview + real Stereo Left + PTY

real Overview + localhost Stereo Left + real STM32
localhost Overview + real Stereo Left + real STM32
```

Не добавлять для этого новые profile flags или abstraction layer: current backend
flags уже выражают нужную combination.

---

# A. Offline software rehearsal — VIRTUAL

## A1. Цель

Offline software rehearsal выполняется:

```text
без интернета
без physical cameras
без physical STM32
```

Цель — доказать, что подготовленная машина самодостаточна offline и может поднять
accepted software/transport/runtime boundaries без загрузки dependencies.

Это **не hardware validation**.

## A2. Что должно быть готово заранее

Минимум:

```text
Python environment already exists
dependencies доступны без download
PyQt6 доступен
production GStreamer receiver доступен
localhost sender tooling доступен
pyserial доступен
PTY доступен
launchers доступны
```

Вместо ручной установки during rehearsal эти prerequisites проверяются VIRTUAL
preflight. Missing dependency при отключённой сети означает `REHEARSAL FAIL`.

## A3. Gate A — VIRTUAL preflight

Выполнить exact command:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret pty \
  --synthetic-inputs \
  --preflight-only
```

Дальше:

1. сохранить console result `PASS/WARN/FAIL`;
2. сохранить launcher exit code;
3. при `WARN` понять причину до продолжения;
4. при `FAIL` остановить rehearsal;
5. найти созданный `logs/navmin-diagnostic-.../` session directory;
6. проверить `manifest.json`, `preflight.json`, `runtime.log`.

Удобная offline inspection без дополнительной dependency:

```bash
python -m json.tool logs/navmin-diagnostic-.../manifest.json
python -m json.tool logs/navmin-diagnostic-.../preflight.json
```

Не редактировать generated JSON.

## A4. Gate B — VIRTUAL full run

После объяснённого `PREFLIGHT PASS`:

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret pty \
  --synthetic-inputs
```

Короткая functional observation:

- application/UI стартует без network access;
- Overview и Stereo Left показывают diagnostic streams;
- labels соответствуют cameras;
- main/preview swap работает;
- Turret достигает рабочего diagnostic state через PTY/production serial path;
- нет очевидного accumulating visual latency за короткий run;
- normal operator shutdown возвращает terminal без зависания.

Не превращать этот step в soak/stress test.

## A5. Gate C — clean shutdown и session evidence

Для successful full runtime session ожидается:

```text
logs/
└── navmin-diagnostic-.../
    ├── runtime.log
    ├── manifest.json
    ├── preflight.json
    └── inputs/
        ├── effective-config.json
        ├── overview-calibration.json
        ├── stereo-calibration.json
        └── source-hashes.json
```

Проверить:

```bash
python -m json.tool logs/navmin-diagnostic-.../manifest.json
python -m json.tool logs/navmin-diagnostic-.../preflight.json
python -m json.tool logs/navmin-diagnostic-.../inputs/source-hashes.json
```

`inputs/*` в VIRTUAL будет иметь explicit synthetic provenance; fake file hashes для
synthetic inputs не создаются.

## A6. Optional isolated diagnostics

Эти tools полезны только если VIRTUAL profile не прошёл и нужно изолировать transport.
Они **не являются application profiles**.

### Localhost RTP/JPEG transport

Current CLI:

```bash
cd ~/NAV\ MIN

python tools/run_localhost_rtp_diagnostic.py \
  --duration-seconds 30 \
  --overview-port 8888 \
  --stereo-left-port 8889
```

Он проверяет detector-friendly localhost scene через production RTP/JPEG camera
receiver path, включая стабильный движущийся Legacy14 track на обеих cameras.
Это не real camera acceptance и не real-camera latency measurement.

### PTY STM32 through production SerialTransport

Current CLI:

```bash
cd ~/NAV\ MIN

python tools/run_pty_stm32_diagnostic.py \
  --timeout-seconds 5
```

Он проверяет software STM32 endpoint через Linux PTY и production
`SerialTransport`. Это не physical STM32 acceptance.

---

# B. Controlled STM32-only hardware preparation — STM32

## B1. Граница

Используется официальный **STM32** profile:

```text
localhost Overview
localhost Stereo Left
real Turret / STM32
```

Это controlled physical boundary для подготовки и измерений. Это **не unrestricted
motion procedure** и не доказательство механической безопасности.

```text
pre-hardware software readiness
!=
proof that mechanics are safe for full-range movement
```

Prompt/runbook ничего физически не выполняет сам: реальные действия делаются
оператором позже, только после review и отдельного разрешения.

## B2. Preflight

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview localhost \
  --stereo-left localhost \
  --turret real \
  --preflight-only
```

`PREFLIGHT PASS` здесь подтверждает static PC readiness выбранных backends. Он не
подтверждает protocol response STM32 и не разрешает motion автоматически.

## B3. Safety gate: факты, которые нельзя предполагать

До unrestricted/high-rate movement считать **not yet validated**, пока они не
измерены для конкретной установки:

```text
actual STEP pulses / output revolution AZ
actual STEP pulses / output revolution EL
exact DM860 model/variant
DM860 DIP/current/microstep settings
AZ direction sign
EL direction sign
STEP high requirement
STEP low requirement
DIR setup requirement
physical limit polarity
physical limit bounce behavior
safe recovery behavior after limit activation
```

Если эти факты не подтверждены:

```text
STOP — DO NOT PROCEED TO UNRESTRICTED MOTION
```

Не считать автоматически доказанными operational values:

```text
2000 STEP/output-rev
20000 STEP/output-rev
5000 STEP/s
```

Authoritative hardware fact — **actual driver STEP pulses / one output revolution**,
измеренный отдельно для AZ и EL.

## B4. STEP/DIR future validation plan

Future controlled sequence:

1. заранее идентифицировать exact DM860 variant и иметь его manual offline;
2. проверить wiring и driver settings;
3. оставить механику в безопасном диапазоне;
4. использовать deliberately low movement/rate;
5. измерить STEP/DIR на driver connector oscilloscope/logic analyzer;
6. измерить STEP high;
7. измерить STEP low;
8. измерить DIR setup before STEP;
9. проверить AZ/EL direction signs;
10. записать результат в worksheet и связать с session directory.

Current NavMin firmware implementation использует:

```text
DIR setup: 1 us
STEP high: 2 us
```

Это **current implementation values**, а не certified requirements конкретного DM860.
Они должны быть сверены с manual установленной ревизии и hardware measurement.

## B5. Physical limits

Current pre-hardware NavMin baseline физически конфигурирует limit input pins, но
**не имеет принятой active physical-limit enforcement policy** для unrestricted/full-
range movement.

Нельзя путать:

```text
limit pins physically exist
```

с:

```text
accepted physical safety policy exists
```

До отдельного STM32 safety checkpoint:

```text
NO unattended travel
NO full-range motion
NO high-rate escalation based on assumed limits
```

Для controlled test later:

```text
small travel
low rate
known clearance
one variable at a time
operator present
power removal accessible
```

Runbook не изобретает recovery policy после limit activation; это отдельный future
STM32 safety decision/checkpoint.

## B6. IWDG и velocity watchdog

IWDG — полезное future hardening, но current baseline не должен утверждать, что оно
реализовано.

```text
velocity watchdog
!=
MCU hardware lockup watchdog
```

Current velocity watchdog защищает velocity-control path при прекращении refresh от
PC. IWDG решал бы другой failure class — зависание MCU/main loop. Prompt 3 его не
реализует.

## B7. Emergency, Motor state и shutdown

Не считать Emergency синонимом `MOTOR_OFF`.

Current semantics:

```text
Emergency
→ hard stop motion
→ does NOT inherently mean MOTOR_OFF
```

Application normal shutdown при доступном Turret и confirmed Motor ON сохраняет
safety boundary:

```text
STOP_MOTION
→ MOTOR_OFF
→ дождаться confirmed MotorState.OFF в bounded timeout
→ camera cleanup
→ TurretWorker shutdown
```

Если Turret недоступен, application не ждёт недостижимого confirmation бесконечно и
переходит к bounded cleanup.

---

# C. Cameras-only smoke — CAMERAS

## C1. Граница

Profile:

```text
Overview       real
Stereo Left    real
Turret         pty
```

Physical Turret здесь не участвует. Acceptance path cameras идёт через production
RTP receiver; `FakeTransport` не заменяет camera acceptance.

## C2. Preflight

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview real \
  --stereo-left real \
  --turret pty \
  --preflight-only
```

`PREFLIGHT PASS` подтверждает static receiver/config/calibration/UDP readiness, но
не означает, что реальные RTP packets уже приходят.

## C3. Launch

```bash
cd ~/NAV\ MIN

python tools/run_diagnostic_app.py \
  --overview real \
  --stereo-left real \
  --turret pty
```

После launch проверить:

```text
Overview ONLINE
Stereo Left ONLINE
actual images visible
camera labels correct
main/preview swap works
no visible accumulating latency
stale/error presentation safe
clean shutdown
```

Сохранять session evidence так же, как для VIRTUAL full run.

## C4. Что этот run не закрывает

CAMERAS-only smoke не закрывает:

```text
camera reconnect #8
automatic production reconnect/backoff
Stereo Right production path
production stereo distance
full Stage 5
full Stage 7
full Stage 8
```

---

# D. Eventual combined real-hardware run — REAL

Этот раздел задаёт conservative future order. **В рамках Prompt 3 REAL run не
выполняется.**

Первый combined physical proof должен оставаться на **Legacy14 vision backend**, если
engineering manager отдельно не изменит решение. Разработка **VT11 vision backend**
после stable tag идёт отдельными checkpoints и не смешивается с первым hardware
bring-up.

## D1. Future order

1. проверить exact project commit/tag/revision и firmware revision;
2. выполнить REAL preflight;
3. inspect `preflight.json`, `manifest.json`, `runtime.log` и session directory;
4. запустить application;
5. подтвердить обе real cameras;
6. подтвердить Turret connection;
7. подтвердить safe motor state;
8. выполнить deliberate Motor ON только когда площадка готова;
9. сделать tiny controlled RELATIVE movement;
10. сверить direction/scale evidence;
11. вернуться в safe stop / `MOTOR_OFF`;
12. только после принятого controlled RELATIVE перейти к controlled TRACKING;
13. выполнить Emergency check в безопасном low-energy condition;
14. выполнить selected failure checks по одному;
15. normal shutdown;
16. inspect session evidence и заполнить worksheet.

Это не stress test.

## D2. Gate before real RELATIVE motion

Требуется:

```text
known clearance
known safe low speed/acceleration
known direction signs
credible STEP/output-rev
Emergency available
external power removal accessible
no person/object in travel
```

Если physical-limit safety ещё не принят, движение должно оставаться deliberately
small.

## D3. Gate before real TRACKING

TRACKING разрешается только после successful controlled RELATIVE validation.

До TRACKING требуется:

```text
camera orientation understood
direction signs verified
small RELATIVE commands correct
scale credible
Emergency verified
operator selection semantics understood
```

Target selection в current NavMin остаётся explicit user click по published bbox.

```text
vision backend proposal identity
!=
automatic turret selection
```

Runbook не добавляет VT11 seed callback и не меняет Vision public contract.

## D4. Failure checks

Future combined run может по отдельности проверять:

```text
camera traffic loss
Turret disconnect
Emergency
normal shutdown
```

Каждый failure test:

```text
one failure at a time
low-energy state
operator ready to remove power
```

Не использовать небезопасное отключение wiring во время движения.

## D5. Hardware roadmap boundary

Stable pre-hardware tag не означает готовность к unrestricted movement. Future work
логически разделяется:

```text
H1
identify / measure / very-low-energy boundary
→ actual hardware facts
→ STM32-V1/V2 implementation if required
→ regression/rehearsal

H2
validate safety boundary
→ cameras-only
→ combined RELATIVE
→ TRACKING
→ failures
```

H1/H2 здесь только фиксируют порядок evidence; этот checkpoint их не выполняет.

---

# 6. Session evidence workflow

Каждый normal/diagnostic launcher run создаёт отдельный UTC-timestamped session
под parent directory `logs/` по умолчанию:

```text
logs/
└── navmin-<mode>-YYYYMMDD-HHMMSS-ffffff/
```

## Preflight-only session

Ожидаются:

```text
runtime.log
manifest.json
preflight.json
```

При successful `--preflight-only` manifest завершается `completed` / exit `0`.
Mandatory preflight failure сохраняет evidence и завершается `preflight-failed` /
exit `2`.

## Full successful runtime session

Ожидаются:

```text
runtime.log
manifest.json
preflight.json
inputs/
  effective-config.json
  overview-calibration.json
  stereo-calibration.json
  source-hashes.json
```

Session schemas не редактировать вручную. Для offline inspection достаточно Python:

```bash
python -m json.tool logs/navmin-<mode>-.../manifest.json
python -m json.tool logs/navmin-<mode>-.../preflight.json
python -m json.tool logs/navmin-<mode>-.../inputs/effective-config.json
python -m json.tool logs/navmin-<mode>-.../inputs/source-hashes.json
```

`runtime.log` bounded: normal session пишет INFO, diagnostic — DEBUG; один runtime
log rotates при 10 MiB и сохраняет до пяти backup files. Старые session directories
автоматически не удаляются.

Session directory может содержать operational details и не считается автоматически
безопасным для публичной публикации.

## Evidence checklist

Для каждого rehearsal/hardware run записать минимум:

```text
date/time
Git commit/tag if available
branch if applicable
profile
preflight result
launcher exit code
session directory
manifest status
manifest exit_code
runtime.log presence
preflight.json presence
inputs/* presence for full runtime
clean shutdown result
operator notes
```

Полезные offline команды, если checkout содержит Git metadata:

```bash
date --iso-8601=seconds
git rev-parse HEAD
git branch --show-current
git describe --tags --always --dirty
```

Если Git metadata нет, это не причина подменять evidence предположением: записать
`UNKNOWN` / `NOT AVAILABLE`.

---

# 7. Hardware measurement worksheet

Не заполнять значениями по памяти или из candidate documents. Использовать
`UNKNOWN` / `NOT MEASURED`, пока нет соответствующего evidence.

```text
project commit/tag:              UNKNOWN
firmware revision:               UNKNOWN
STM32 board:                     UNKNOWN
DM860 AZ exact model:            UNKNOWN
DM860 EL exact model:            UNKNOWN
AZ switch settings:              NOT MEASURED
EL switch settings:              NOT MEASURED
AZ STEP/output-rev:              NOT MEASURED
EL STEP/output-rev:              NOT MEASURED
AZ direction sign:               NOT MEASURED
EL direction sign:               NOT MEASURED
STEP high measured:              NOT MEASURED
STEP low measured:               NOT MEASURED
DIR setup measured:              NOT MEASURED
limit polarity:                  NOT MEASURED
observed switch bounce:          NOT MEASURED
safe recovery notes:             UNKNOWN
verified low-speed ceiling:      NOT MEASURED
verified acceleration:           NOT MEASURED
UART/reset notes:                UNKNOWN
operator/date:                   UNKNOWN
related session directory:       UNKNOWN
```

---

# 8. Deliberately open boundaries

Наличие этого runbook не закрывает и не реализует:

```text
offline rehearsal itself
camera reconnect/backoff #8
UI error surfacing
production stereo distance
physical limit enforcement
IWDG
real STM32 validation
real camera validation
mechanical scale/timing/tuning
Stage 5/6/7/8 completion
VT11 vision backend integration
protocol 4.0 / SERVO_BOTH lineage
Stage8.2 firmware integration
```

После принятия этого documentation checkpoint следующий operational step — **offline
rehearsal**. README cleanup, merge, stable tag, selected VT11/STM32 implementation
checkpoints и real hardware work выполняются только отдельными последующими
решениями.
