# Как начать работу над проектом

Здесь дана инструкция, как быстро настроить окружение и начать работу над проектом

Проверено на Debian 13 с KDE. Хоть часть инструкций применима и к Windows, настоятельно рекомендуется установить любой linux-дистрибутив по вашему вкусу (Debian, Ubuntu LTS, Mint)

!!! note "Dual-boot"
    Рекомендую ставить **Debian** второй системой (рядом с **Windows**) c окружением рабочего стола *KDE*.

# Используемые инструменты
Краткий перечень используемых инструментов

1. `VSCodium` - IDE (VSCode без телеметрии, можно использовать любой, они совместимы)
2. `MkDocs` - для оформления документации
3. `Ruff` - **линтинг** (проверка кода на ошибки и соответствие стандартам) и **форматирование** (приведение кода к единому стилю)

# Установка и настройка

## 1. VSCodium

!!! warning "Важно"
    Установка через `flatpak` не рекомендуется, в ней сложно настроить все инструменты из-за ограничений песочницы.

!!! note "Готовые установщики"
    Для Windows и Linux есть готовые установщики на официальном GitHub (.deb и .exe), но тогда не будет автоматических обновлений.

Рекомендуемый способ установки (с автоматическими обновлениями):
```bash
# 1. Подключаем репозиторий VSCodium
sudo wget https://gitlab.com/paulcarroty/vscodium-deb-rpm-repo/raw/master/pub.gpg -O /usr/share/keyrings/vscodium-archive-keyring.asc
echo 'deb [ signed-by=/usr/share/keyrings/vscodium-archive-keyring.asc ] https://paulcarroty.gitlab.io/vscodium-deb-rpm-repo/debs vscodium main' | sudo tee /etc/apt/sources.list.d/vscodium.list

# 2. Устанавливаем VSCodium
sudo apt update
sudo apt install codium
```

Теперь VSCodium доступен в списке приложений. Установите необходимые расширения и настройки.

Команды для автоматической установки через терминал:

```bash

```

## 2. Настройка GitHub и скачивание файлов проекта

### 1. Зарегистрируйтесь на GitHub и настройте доступ к репозиторию

***TODO: Зарегать оргу, Пройти этот этап и заполнить***

### 2. Сгенерируйте SSH-ключ (если ещё нет):

```bash
ssh-keygen -t ed25519 -C "ваша_почта@example.com"
```

### 3. Добавьте публичный ключ в GitHub

Скопируйте ключ:
```bash
cat ~/.ssh/id_ed25519.pub
```

Перейдите в `Settings` → `SSH and GPG keys` → `New SSH Key`

Вставьте ключ и сохраните

### 4. Склонируйте через SSH

На странице репозитория нажмите `Code` → выберите вкладку `SSH`

Скопируйте ссылку вида `git@github.com:username/repository.git`

Выполните (вставьте правильную ссылку на репозиторий):

```bash
git clone git@github.com:username/repository.git
```


## 3. Виртуальное окружение (venv)

Для проекта требуется **Python 3.13**. Для установки всех зависимостей используется `uv`.

Рекомендуемый способ - автономный установщик:

=== Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

=== Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
После установки перезапустите терминал

Далее переместитесь в директорию с проектом

Создание окружения и установка зависимостей:

!!! warning "Для Windows"
    Ниже замените `.venv/bin/activate` на `.venv\Scripts\activate`

```bash
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .[dev,docs]
```
Откройте новый терминал в IDE. Если окружение не активируеся автоматически, проверьте, есть ли в проекте диреткория `.vscode/`, в ней настройки для автоматического запуска окружения. Без нее вы сможете активировать окружение только вручную.

## 4. Git hooks проекта

В репозитории есть локальные Git hooks, которые автоматически запускают обязательные проверки перед `commit` и `push`. Файлы hooks хранятся в `.githooks/`, но Git не включает их автоматически после `clone` или `pull`. Для каждой новой рабочей копии один раз включите их:

```bash
git config core.hooksPath .githooks
```

Проверить текущую настройку:

```bash
git config --get core.hooksPath
```

Ожидаемый вывод:

```text
.githooks
```

`pre-commit` выполняет быстрые проверки:

```text
git diff --cached --check
Ruff
compileall
```

`pre-push` запускает полный Python test suite:

```text
pytest
```

Если проверка завершается ошибкой, `commit` или `push` блокируется. Сначала исправьте причину ошибки и повторите команду Git. Hooks не изменяют код автоматически и не выполняют `ruff --fix`. Проверки запускаются через `uv --offline`: они не скачивают зависимости во время `commit` или `push`, поэтому dev-окружение должно быть подготовлено заранее по инструкции выше.

Чтобы отключить проектные hooks в текущей рабочей копии:

```bash
git config --unset core.hooksPath
```

Эта команда только отключает использование `.githooks/` для текущей рабочей копии; файлы hooks из репозитория не удаляются.

Ручные эквиваленты проверок:

```bash
git --no-pager diff --cached --check

uv run --offline --extra dev ruff check \
  src \
  tests \
  main.py \
  tools/run_software_smoke.py \
  tools/run_software_soak.py \
  tools/run_localhost_rtp_diagnostic.py \
  tools/run_pty_stm32_diagnostic.py \
  tools/run_diagnostic_app.py \
  tools/run_rtsp_camera_diagnostic.py

uv run --offline --extra dev python -m compileall -q src main.py tests tools
uv run --offline --extra dev pytest
```

## 5. Camera runtime: GStreamer

`PyGObject` является Python dependency проекта и устанавливается через `uv`. Сам GStreamer и typelibs/plugins являются **system runtime dependencies Debian**, а не Python packages.

Для текущего RTP/JPEG camera source нужны:

```text
GStreamer runtime/tools
GStreamer GstApp typelib
base plugins
good plugins
```

На Debian соответствующий минимальный runtime набор:

```bash
sudo apt install \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-libav \
  gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0
```

`plugins-bad` нужен для `h264parse`, а `gstreamer1.0-libav` — для программного `avdec_h264` в RTSP/H.264 path. Проверка production camera elements:

```bash
gst-inspect-1.0 \
  udpsrc rtpjpegdepay jpegdec \
  rtspsrc rtph264depay h264parse avdec_h264 \
  videoconvert appsink
```

Production camera source выбирается через `vision.cameras.<camera>.source.type = rtp-jpeg | rtsp`. Для RTP/JPEG обычный local bind — `0.0.0.0`, принятые порты Overview/Stereo Left/Stereo Right — `8888/8889/8890`, `buffer-size=1`. Для RTSP endpoint задаётся в `source.uri`; первая реализация поддерживает H.264, `protocol = tcp | udp` и `decoder-mode = software`.

## 6. Подготовка Linux-хоста для реального оборудования

Системные настройки последовательного порта и сети вынесены из Python-кода NavMin в отдельные Linux-скрипты. Параметры конкретного компьютера хранятся в локальном файле `config/host.local.env`: он не коммитится и служит единым источником настроек для setup/runtime-скриптов.

Сначала один раз создайте локальный файл из шаблона:

```bash
cp config/host.example.env config/host.local.env
```

Откройте `config/host.local.env` и задайте значения для текущего ПК:

```text
SERIAL_SETUP_DEVICE=/dev/ttyUSB0
SERIAL_ALIAS=navmin-turret
CAMERA_INTERFACE=enp3s0
CAMERA_HOST_ADDRESS=192.168.42.2/24
CAMERA_PROFILE_NAME=NavMin Cameras
```

`/dev/ttyUSB0`, `enp3s0` и `192.168.42.2/24` здесь только примеры. Значения с пробелами поддерживаются без shell-кавычек. Если файл отсутствует или обязательное поле задано дважды, setup/runtime-скрипты завершаются ошибкой вместо применения догадок.

### Serial-адаптер

Ручной `chmod 666/777 /dev/ttyUSB*` использовать не нужно: такой режим сбрасывается при каждом переподключении устройства и даёт избыточные права.

`SERIAL_SETUP_DEVICE` — это только текущий путь USB-UART во время одноразовой настройки. На одном ПК это может быть `/dev/ttyUSB0`, на другом — `/dev/ttyUSB1` или `/dev/ttyACM0`. После настройки runtime использует стабильное имя из `SERIAL_ALIAS`.

При подключённом адаптере один раз выполните:

```bash
sudo tools/platform/linux/setup_serial_access.sh
```

Скрипт:

- читает `SERIAL_SETUP_DEVICE` и `SERIAL_ALIAS` из `config/host.local.env`;
- определяет USB VID/PID и, если доступен, серийный номер адаптера;
- создаёт `/etc/udev/rules.d/99-navmin-turret.rules`;
- задаёт `GROUP="dialout"` и `MODE="0660"`;
- создаёт стабильное имя `/dev/<SERIAL_ALIAS>`;
- при необходимости добавляет пользователя, запустившего `sudo`, в группу `dialout`.

Если пользователь был добавлен в `dialout`, выйдите из пользовательской сессии и войдите снова. При значении `SERIAL_ALIAS=navmin-turret` в локальном `config.json` рекомендуется использовать:

```text
turret.serial.port = /dev/navmin-turret
```

Если на другом ПК адаптер получил другое временное имя, достаточно изменить `SERIAL_SETUP_DEVICE` в `host.local.env` и повторить setup. Повторный запуск безопасен: совпадающее правило не создаётся заново.

### Сетевой профиль камер

NavMin не хранит Wi-Fi пароль и не создаёт сетевое подключение с нуля. Сначала обычными средствами NetworkManager подключите нужный интерфейс к сети камеры: выберите Wi-Fi камеры или подключите нужный Ethernet-интерфейс. Затем узнайте имя интерфейса:

```bash
nmcli device status
```

Запишите его в `CAMERA_INTERFACE`. В `CAMERA_HOST_ADDRESS` укажите **адрес ПК, на который настроены отправители видеопотока**, включая CIDR-префикс. Этот адрес не должен совпадать с IP самой камеры. Имя создаваемого профиля задаётся в `CAMERA_PROFILE_NAME`.

После редактирования `host.local.env` один раз выполните:

```bash
sudo tools/platform/linux/setup_camera_network.sh
```

Если профиль из `CAMERA_PROFILE_NAME` ещё не существует, скрипт клонирует текущее активное подключение выбранного интерфейса. Поэтому для Wi-Fi сохраняются SSID и параметры безопасности уже созданного подключения NetworkManager, а секреты не попадают в репозиторий. Затем профиль получает параметры из локального setup-файла:

```text
CAMERA_INTERFACE       → connection.interface-name
CAMERA_HOST_ADDRESS    → статический IPv4
connection.autoconnect = no
ipv4.never-default     = yes
IPv6                   = disabled
```

`config/host.local.env` является источником параметров хоста, а NetworkManager-профиль — применённым системным состоянием. Если позже изменить интерфейс, адрес или имя профиля в setup-файле, повторно запустите `setup_camera_network.sh`. Runtime-проверка не позволит молча использовать профиль, который больше не соответствует `host.local.env`.

### Ручное включение и восстановление сети

Для диагностики профиль можно включить вручную:

```bash
tools/platform/linux/camera_network_up.sh
```

Скрипт читает `host.local.env`, проверяет соответствие подготовленного NetworkManager-профиля, запоминает UUID подключения, которое было активно на том же интерфейсе, активирует профиль камер и проверяет ожидаемый статический IPv4.

Вернуть прежнее состояние:

```bash
tools/platform/linux/camera_network_down.sh
```

`camera_network_down.sh` восстанавливает состояние из runtime state-файла и намеренно не зависит от наличия `host.local.env`: cleanup должен оставаться возможным даже если локальный setup-файл был удалён или изменён во время сессии.

Если до запуска профиль NavMin уже был активен, `down` оставит его активным. Если до запуска на интерфейсе не было подключения, `down` просто отключит профиль камер.

### Рекомендуемый запуск NavMin на Linux

Обычный запуск с реальным оборудованием выполняйте из корня репозитория:

```bash
./run_navmin.sh --preflight-only
./run_navmin.sh
```

`run_navmin.sh` выполняет последовательность:

```text
camera_network_up.sh
→ uv run --offline python -m navmin ...
→ camera_network_down.sh
```

Аргументы после `run_navmin.sh` передаются обычному `python -m navmin`. Сам launcher не содержит IP, имя интерфейса или профиль: их читает `camera_network_up.sh` из локального setup-файла.

Восстановление сети выполняется при обычном завершении, ошибке приложения и штатно обрабатываемом завершении процесса, включая обычный `Ctrl+C`. Если процесс был принудительно завершён через `SIGKILL` или компьютер потерял питание, выполнить восстановление невозможно. В пределах той же пользовательской сессии оставшийся файл состояния будет обнаружен следующим `camera_network_up.sh`, и новый запуск будет остановлен до явного восстановления через `camera_network_down.sh`.

`run_navmin.sh` не нужно запускать через `sudo`. Повышенные права нужны только для одноразовых `setup_serial_access.sh` и `setup_camera_network.sh`.

# Offline rehearsal и hardware day

Этот документ описывает первоначальную установку окружения и поэтому содержит online setup steps. Для автономной проверки уже подготовленной машины network не должен требоваться. Operator procedure, четыре рекомендуемых profiles, preflight gates и hardware measurement boundaries описаны в [Offline Hardware Runbook](../user/offline-hardware-runbook.md).
