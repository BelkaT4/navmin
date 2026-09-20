# AGENTS.md — архитектурная документация

Правила сопровождения `docs/dev/architecture/`. Этот файл описывает процесс; нормативные решения находятся в самих `.md`/`.mmd`.

## 1. Роли документов

- `overview.md` — устойчивая верхнеуровневая архитектура и system invariants.
- `contracts.md` — общие межмодульные types/state/API boundaries.
- `configuration.md` — config schema, ownership и apply policy.
- `serial-protocol.md` — единственный нормативный источник PC ↔ STM32 wire/transport semantics.
- `decisions.md` — **почему** выбраны существенные решения и какие альтернативы отвергнуты.
- `problems.md` — только реально открытые вопросы.
- `architecture-updates.md` — временные принятые/обсуждаемые изменения, которые ещё не полностью перенесены в постоянную документацию.
- `docs/dev/modules/*` — ответственность и контракт конкретных модулей.
- `docs/dev/diagrams/*.mmd` — source of truth диаграмм; PNG/SVG — экспорт.

Не превращай `decisions.md` в копию нормативных контрактов, `problems.md` — в историю, а `architecture-updates.md` — в вечный журнал.

## 2. Новое решение: lifecycle документации

Если появился настоящий открытый вопрос:

```text
неясность / противоречие
→ problems.md
```

Если решение обсуждено/принято, но ещё не синхронизировано полностью:

```text
architecture-updates.md
```

После утверждения:

```text
1. обновить authoritative contract document;
2. обновить owner module doc;
3. обновить consumer module docs, если изменился их контракт;
4. обновить .mmd, если изменились structure/data/control flow;
5. добавить rationale в decisions.md для существенного решения;
6. удалить закрытый вопрос из problems.md;
7. убрать/актуализировать временную запись в architecture-updates.md;
8. найти по docs/ остатки старой модели/терминов.
```

Не оставляй старую и новую модель одновременно как актуальные.

## 3. Что проверять по типу изменения

### Межмодульный контракт

```text
contracts.md
owner module
consumer module(s)
соответствующие .mmd
```

### Serial protocol

```text
serial-protocol.md
Turret module
contracts.md при изменении PC-side types
configuration.md при изменении config ownership
соответствующие diagrams
```

### Camera/session/generation

```text
overview.md
contracts.md
Vision
Core
UI
session/gate paths в .mmd
```

### Config

```text
configuration.md
owner module
consumers
serial-protocol.md, если поле передаётся STM32
```

### Control mode / target / aiming / motion

```text
overview.md
contracts.md
Core
Turret
UI при user-facing behavior
соответствующие .mmd
```

## 4. `decisions.md`: rationale, а не второй контракт

Добавляй rationale, когда решение:

- меняет state machine;
- удаляет существовавший механизм;
- выбирает ownership между модулями;
- определяет recovery/safety semantics;
- сознательно откладывает механизм/feature;
- имеет несколько разумных альтернатив.

Минимальная структура записи:

```text
Решение
Почему
Отклонённые существенные альтернативы и причины
```

Не придумывай фиктивные альтернативы и не фиксируй там каждый rename/helper.

Если `decisions.md` расходится с нормативным контрактом, исправь рассинхронизацию; нормативное поведение определяется соответствующим contract/module document и `.mmd` для структуры.

## 5. Диаграммы

```text
*.mmd = source of truth
PNG/SVG = export only
```

Меняется ownership, module boundary, data/control flow или transport path → проверь `.mmd`.

Не редактируй PNG/SVG вместо `.mmd`. Если для изменённого `.mmd` в Git хранится PNG/SVG export, перед commit обнови export после окончательной версии source — в том числе после ручных изменений размеров, positions, waypoints или arrangement.

Сохраняй используемые проектом metadata/position comments. Не выполняй automatic relayout существующей диаграммы без необходимости и не удаляй layout metadata только потому, что Mermaid умеет отрисовать diagram без неё. При rename node ID согласованно обновляй связанные `graph:positions`, waypoints, edge metadata и другие project-specific layout references.

## 6. Не дублировать контракты

Предпочитай:

```text
authoritative definition
+ короткое summary в consumer doc
+ ссылка
```

В частности:

- wire-format → `serial-protocol.md`;
- rationale → `decisions.md`;
- open question → `problems.md`;
- module responsibility → module docs;
- structure/data flow → `.mmd`.

## 7. Правило упрощения

При архитектурной проблеме сначала спроси:

> Можно ли удалить состояние/механизм, создающий проблему, вместо добавления ещё одного flag/queue/ID/watchdog?

Предпочитай меньшее число authoritative states и ownership boundaries.

Но не удаляй механизм, который закрывает конкретный воспроизводимый race/failure case. Не возвращай ранее отвергнутый механизм без нового требования.

## 8. Проверки после изменений

Перед завершением:

1. Проверь относительные Markdown-ссылки.
2. Проверь изменённые `.mmd` и их metadata доступными средствами.
3. Проверь metadata/edges на ссылки к удалённым nodes, если такие metadata используются.
4. Выполни поиск по удалённым/переименованным архитектурным терминам во всём `docs/`.
5. Удали уже закрытые вопросы из `problems.md`.
6. Проверь корректность статуса `architecture-updates.md`.
7. Для нового существенного решения проверь rationale в `decisions.md`.
8. Не заявляй полную синхронизацию, если известное противоречие осталось.

## 9. Не делать

Не:

1. редактировать PNG/SVG вместо `.mmd`;
2. использовать `decisions.md` вместо нормативного контракта;
3. оставлять закрытые вопросы в `problems.md`;
4. превращать `architecture-updates.md` в архив истории;
5. переписывать архитектуру под implementation bug без решения;
6. добавлять механизм только «на будущее»;
7. дублировать полный wire-format/state machine в нескольких файлах;
8. объявлять документы синхронизированными без проверки связанных module docs и `.mmd`.
