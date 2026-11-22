# AGENTS.md — BelkaT4 / NavMin

Правила работы AI-агента с проектом BelkaT4 / NavMin.

Перед началом работы прочитай этот файл полностью. Явные инструкции пользователя для конкретной задачи имеют приоритет над этими правилами.

## 1. Источники истины

Для текущего состояния проекта приоритет такой:

```text
1. текущий код authoritative working tree / snapshot;
2. текущие тесты;
3. runtime evidence: логи, трассировки, результаты запуска;
4. docs/ только как исторический материал.
```

Главный источник истины — код.

`docs/` значительно устарел. Его разрешено читать для понимания истории, старых решений и терминологии, но запрещено изменять, дополнять, удалять или синхронизировать с кодом. Вложенные `docs/**/AGENTS.md` также являются только историческими документами.

Расхождение кода и `docs/` само по себе не является ошибкой кода.

## 2. Authoritative baseline

Если пользователь указал authoritative snapshot и expected SHA-256:

1. проверить SHA-256;
2. проверить ZIP integrity;
3. распаковать отдельную рабочую копию;
4. работать только от этого baseline.

Не смешивать разные snapshot. Старые архивы, overlays, отчёты и сведения из предыдущих чатов не имеют приоритета над текущим baseline.

## 3. Обязательный workflow

По умолчанию:

```text
READ-ONLY аудит
→ отчёт по аудиту
→ пошаговый план
→ STOP
→ согласование плана пользователем
→ STEP 1
→ agent-side проверки
→ overlay ZIP + SHA-256 + команды
→ STOP
→ user-side validation
→ следующий STEP
→ ...
→ финальная проверка
```

Не начинать STEP 1 до явного согласования плана.

После каждого шага останавливаться и ждать результатов пользовательской проверки. Если checkpoint не прошёл — исправлять его, а не переходить дальше.

Пользователь может явно разрешить другой workflow.

## 4. Аудит, план и реализация

До изменения файлов провести READ-ONLY аудит и установить:

- фактическое поведение и execution path;
- владельца поведения (`responsibility owner`);
- связанные API/state/boundaries;
- существующие тесты и runtime evidence;
- для bugfix — root cause;
- минимальный scope, риски и invariants.

Для bugfix:

```text
симптом → evidence → execution path → root cause → минимальный безопасный fix
```

Не менять production-код по одной неподтверждённой гипотезе.

После аудита составить пошаговый план. Каждый шаг должен иметь цель, предполагаемые изменяемые файлы, ожидаемое изменение, сохраняемые invariants и способ проверки.

После представления плана — `STOP`.

После согласования выполнять только текущий шаг. Предпочитать минимальный diff, не делать unrelated refactor и не добавлять параллельные API/state/flags/queues/watchdogs/compatibility layers без реальной необходимости.

## 5. Тестирование

Использовать принцип **coverage first**:

- сначала искать существующие тесты, fixtures и helpers;
- по возможности расширять существующий test/parameterized case;
- новый test file создавать только для отдельного непокрытого failure mode.

На промежуточном checkpoint запускать применимые targeted проверки: pytest, Ruff, compileall, `git diff --check`, task-specific validation.

Полный regression обычно выполнять на финальном integration checkpoint либо раньше, если этого требует scope.

Никогда не писать `PASS`, если проверка реально не запускалась.

Различать:

```text
agent-side validation
user-side validation
hardware validation
```

Mocks, fake, PTY, localhost, preflight и software simulation не являются hardware validation.

Если обязательная проверка недоступна агенту, установить причину, отметить `NOT RUN` и передать точную локальную команду пользователю. Не повторять заведомо бесполезные попытки.

Основной Python workflow проекта — `uv`; по умолчанию пользователь работает на Linux, если задача не говорит обратного.

## 6. Checkpoint и overlay

После каждого шага с изменениями создать уникальный overlay ZIP.

По умолчанию overlay накопительный относительно authoritative baseline:

```text
STEP 1: baseline → STEP 1
STEP 2: baseline → STEP 2
STEP 3: baseline → STEP 3
```

Overlay содержит только changed/new project files. Не включать caches, `.venv`, `__pycache__`, временные и unrelated files.

Пути внутри ZIP — относительно project root. Удаляемые файлы перечислять отдельно: overlay сам их не удаляет.

Перед передачей проверить ZIP integrity, состав, пути, SHA-256 и отсутствие unrelated files. По возможности проверить:

```text
baseline + overlay = фактически протестированное result tree
```

Checkpoint report должен содержать:

- статус шага;
- кратко что сделано;
- changed / added / deleted files;
- unrelated local modifications;
- выполненные проверки;
- непроведённые проверки и причины;
- overlay ZIP, SHA-256 и ссылку;
- один готовый блок команд;
- что пользователь должен прислать обратно.

## 7. Обязательный блок команд

Предполагается, что терминал уже открыт в корне проекта.

Архив по умолчанию находится в `$HOME/Downloads/`.

```bash
# Проверить SHA-256 overlay
echo "<SHA256>  $HOME/Downloads/<ARCHIVE>.zip" | sha256sum -c -

# Проверить целостность ZIP
unzip -t "$HOME/Downloads/<ARCHIVE>.zip"

# Применить overlay
unzip -o "$HOME/Downloads/<ARCHIVE>.zip" -d .

# Если checkpoint удаляет файлы:
# rm path/to/obsolete_file.py

# Targeted tests
uv run --offline --extra dev pytest -q \
  <TARGETED_TESTS>

# Ruff
uv run --offline --extra dev ruff check \
  <RELEVANT_PATHS>

# Компиляция
uv run --offline python -m compileall -q \
  src tests tools

# Проверка diff
git --no-pager diff --check

# Состояние working tree
git --no-pager status --short
```

Перед выдачей пользователю обязательно заменить все placeholders реальными значениями, удалить неприменимые команды и добавить нужные task-specific проверки.

Если нужен полный regression, добавить его явно:

```bash
uv run --offline --extra dev pytest -ra
```

Ручные или hardware-проверки давать отдельно после автоматического блока.

В конце явно указать, что прислать обратно, например:

```text
Пришли полный вывод блока команд.
```

или:

```text
Пришли полный вывод блока команд и runtime.log этого запуска.
```

## 8. Git и local state

Для Git-команд, вывод которых анализируется или передаётся в чат, использовать `git --no-pager`, чтобы Git не запускал интерактивный `less`, не скрывал часть stdout и не ожидал пользовательского ввода.

Примеры:

```bash
git --no-pager status --short
git --no-pager diff
git --no-pager diff --check
git --no-pager show --stat
git --no-pager log -n 10 --oneline
```

Без явного поручения пользователя не выполнять commit, push, merge, tag, release или Pull Request.

GitHub/PR не являются обязательной частью workflow.

Не использовать `git add .`, если scope можно перечислить явно.

Локальные файлы вроде `config/navmin.json`, если они уже modified и не относятся к задаче, не восстанавливать, не перезаписывать и не включать в overlay. Указывать их как unrelated local modifications.

После tools/tests/build проверять появившиеся файлы. Не добавлять неизвестные generated artifacts автоматически в Git или `.gitignore`. `uv.lock` не считать cache автоматически.

## 9. Safety-critical изменения

Для Turret, STM32, Emergency, MOTOR_OFF, motion, recovery и других safety-critical частей:

- сначала определить гарантию существующей boundary;
- не ослаблять её только ради latency или удобства;
- не считать неизвестное physical state известным;
- использовать conservative assumptions;
- сохранять безопасный fallback;
- отдельно проверять failure/recovery path;
- не заявлять hardware safety только по software tests.

Если более точное безопасное состояние доказать нельзя — сохранять прежнее консервативное поведение.

## 10. Язык и ключевые запреты

Рабочие отчёты и этот файл писать преимущественно на русском. Английские термины использовать там, где они нужны для точности или сопоставления с кодом.

Без прямого указания пользователя не:

- начинать реализацию до аудита и согласования плана;
- переходить к следующему шагу до проверки текущего checkpoint;
- делать speculative fix без подтверждённой причины;
- изменять `docs/`;
- менять unrelated files;
- заявлять `PASS` для незапущенной проверки;
- выдавать simulation за hardware validation;
- ослаблять safety boundary без доказательства;
- перезаписывать unrelated local config;
- выполнять commit/push/merge/tag/PR автоматически.