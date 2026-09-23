# Core: Config Manager

`Config Manager` — единственный источник актуальной общей конфигурации приложения.

## Ответственность

Config Manager:

- загружает `config.json` при запуске;
- валидирует настройки;
- хранит актуальные типизированные конфигурации;
- принимает изменения от UI через Core;
- сохраняет пользовательские изменения атомарно через temporary file + replace;
- публикует новые immutable snapshots заинтересованным компонентам.

Рабочие модули не изменяют общий `config.json` напрямую. Первая версия файла имеет `schema-version = 1`.

Startup/persistence policy v1 строгая:

- отсутствующий `config.json` — startup/config error; strict loader сам файл не создаёт;
- malformed JSON, missing/invalid `schema-version`, unknown field, missing required field, invalid type/range/non-finite value — validation error;
- `schema-version != 1` — unsupported schema error без automatic migration;
- только явно документированные optional fields получают in-memory defaults;
- invalid runtime update не заменяет последний валидный snapshot и не публикуется частично;
- загрузка/validation сама по себе не переписывает пользовательский файл.

`config.json` и calibration — локальные site-specific inputs и не входят в repository baseline. Явный startup recovery принадлежит launcher/UI boundary, а не Config Manager: после input error оператор может закрыть программу или восстановить весь local input set из safe defaults с timestamped backups. Даже после успешного recovery текущий normal startup не продолжается. `--preflight-only` и diagnostic/headless paths остаются неинтерактивными.

Полный required/optional contract и value constraints находятся в [Конфигурации системы](../../architecture/configuration.md).

## Типизированные снимки

Внутри приложения используются конфигурационные модели, например:

```text
VisionConfig
TurretConfig
AimingConfig
UiConfig
```

После успешного изменения Config Manager увеличивает один глобальный `revision` и создаёт:

```python
ConfigUpdate(revision=..., config=...)
```

Полный контракт описан в [общих контрактах](../../architecture/contracts.md#configupdate).

## Latest-only

Межпотоковый `ConfigUpdate` — состояние, а не история действий.

Если компонент ещё не применил несколько промежуточных revision, он может получить только самый новый snapshot. Устаревший update с меньшим или уже применённым revision игнорируется.

## Динамическое применение

Config Manager не содержит `requires_restart` и не знает внутреннюю механику ресурсов.

Компонент сам сравнивает старую и новую конфигурацию и определяет:

- применить поле сразу;
- перезапустить camera pipeline / resource;
- выполнить Turret reconnect;
- потребовать restart, если runtime-применение не является частью v1 contract.

Базовая классификация concrete fields зафиксирована в [Конфигурации системы](../../architecture/configuration.md); открыты только edge cases и processor-specific settings.

## STM32-конфигурация

Config Manager хранит параметры Turret, но физический `SET_CONFIG` создаёт и отправляет Turret HAL.

`ConfigUpdate` и `SET_CONFIG` — разные уровни:

```text
ConfigUpdate   = внутренняя конфигурация приложения
SET_CONFIG     = физическая команда STM32
```

## Связанный документ

Полная структура пользовательского `config.json` пока описана централизованно в [Конфигурации системы](../../architecture/configuration.md). Позже конкретные параметры будут разнесены по документации модулей, а архитектурный файл останется описанием общих правил.
