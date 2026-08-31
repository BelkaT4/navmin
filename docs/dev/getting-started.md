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
