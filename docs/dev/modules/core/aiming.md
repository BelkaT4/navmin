# Core: Aiming

`Aiming` — компонент Core, который преобразует точки на working frame в логическое геометрическое требование к Turret.

Aiming выполняется в main thread и вызывается Mediator синхронно.

## Границы ответственности

Aiming не:

- выбирает authoritative target;
- выполняет VisionProcessor;
- управляет моторами;
- выполняет PID;
- отправляет UART;
- исправляет distortion/rectification.

Geometric correction уже выполнена в Vision до публикации `FramePacket`.

## Working frame и `CameraModel`

Все входные pixel coordinates относятся к working frame.

`CameraModel` immutable и приходит вместе с `CameraSessionStarted(camera, generation, camera_model)`. Core/Aiming используют модель только для принятой `generation` из `CameraSessionGate`.

Публичный геометрический контракт:

```text
CameraModel.pixel_to_ray(x_px, y_px) → normalized CameraRay
```

Система CameraRay:

```text
+x → вправо
+y → вниз
+z → вперёд от камеры
```

Aiming не зависит напрямую от OpenCV `K`, `D`, `P1`, `P2`.

## Логическая система Turret

На выходе Aiming:

```text
+X → турели нужно вправо
+Y → турели нужно вверх
```

Следовательно, image/camera `+Y вниз` явно преобразуется внутри Aiming в turret `+Y вверх`.

Например вертикальный угол луча может вычисляться со знаком, соответствующим Turret:

```text
vertical_deg = -atan2(ray.y, sqrt(ray.x² + ray.z²))
```

HAL `invert-y` не используется для этого преобразования: он отвечает только за физическое направление двигателя/проводки.

## Aim point

Для Overview и Stereo Left задаётся aim point в пикселях working frame:

```text
aim-x-px
aim-y-px
```

Если не задан — используется center working frame.

В первой реализации предполагается, что camera axes достаточно близки к turret axes. Постоянное boresight-смещение компенсируется aim point. Полный `R_camera_to_turret` откладывается до механических испытаний.

## Поворот по клику

```text
click pixel → click ray
aim point → aim ray
→ angular difference в логической системе Turret
→ delta_x_deg / delta_y_deg
```

Постоянный `degrees_per_pixel` и простая линейная разница pixels не используются.

Click-to-move вызывается только после подтверждённого `TurretState.control_mode == RELATIVE`.

## Сопровождение и lead

Для актуального `selected_target` в подтверждённом `TRACKING`:

```text
lead_point =
    bbox_center +
    velocity_px_s * lead_time_s
```

Lead рассчитывается только при наличии актуального `TrackedObject`.

Далее:

```text
lead pixel → target ray
aim point → aim ray
→ angular error
→ TrackingError
```

Если selected track временно отсутствует, Aiming не продолжает самостоятельно экстраполировать старую цель.

## `TrackingError`

`TrackingError` создаётся только если:

- latest `TurretState.control_mode == TRACKING`;
- существует `selected_target`;
- target относится к `main_camera`;
- generation принята `CameraSessionGate`;
- track присутствует в актуальном `VisionResult`.

На межпоточной границе `TrackingError` хранится в latest-state container с monotonic `revision`. Turret PID обрабатывает каждое revision не более одного раза.

При уходе из TRACKING, final target loss/deselect, camera-generation invalidation, StopMotion или Emergency latest-state явно `clear()/invalidate()`. Core не посылает отдельный PID-reset: Turret Controller сам реагирует на domain/control boundaries.

## Calibration

Calibration хранится отдельно от `config.json`:

```text
calibration/overview.json
calibration/stereo.json
```

Vision загружает calibration и строит working-frame `CameraModel`. При calibration mismatch/error pipeline не публикует raw frame как замену working frame.

## Конфигурация Aiming

Минимально:

```text
lead-time-ms
target-lost-timeout-ms
aim point Overview
aim point Stereo Left
```

`target-lost-timeout-ms` используется Core, но относится к поведению tracking.

Подробно: [Конфигурация](../../architecture/configuration.md).


## Ручная калибровка движения

`Mediator` хранит один незавершённый замер и результаты ручной серии;
`core.motion_calibration` рассчитывает кривую движения и оценку люфта. UI передаёт
координаты одной неподвижной точки сцены до и после поворота камеры.
Измерения выполняются по одной оси: угол луча `atan2(x,z)` для X и
`-atan2(y,z)` для Y, в градусах. Разность углов до/после даёт измеренный
поворот при принятом предположении о параллельности осей камеры и платформы.
Постоянный коэффициент «градусы на пиксель» не используется.

Начало требует `RELATIVE`, `READY`, `ON`, отсутствия pending mode transition,
принятой generation и свежего кадра main camera. Командный угол ограничен
действующим `max-relative-move-deg` и снизу 0.1°. Команда проходит обычный
Turret ingress; второй клик команды не создаёт. Оператор подтверждает остановку,
после чего принимается только более новый кадр той же камеры и generation.
Подтверждение не является телеметрией STM32. Для новой серии активная
`relative-motion-curve` выбранной оси должна быть пустой: повторное обучение
поверх неизвестной поправки не допускается.

Swap, новая сессия основной камеры, потеря её связи/свежести, смена режима,
Stop, Emergency, выключение моторов или потеря связи Turret отменяют незавершённый
замер и знание предыдущего направления. Обычное движение также сбрасывает знание
направления серии. Завершённые замеры сохраняются до конца запуска либо явного
сброса оператором. Первое движение с неизвестным преднатягом не используется
при построении кривой. После нулевого/обратного измеренного движения следующее
направление также считается неизвестным.

Для кривой нужны минимум три различных командных угла без реверса в каждом
направлении; максимальная амплитуда должна быть хотя бы вдвое больше минимальной.
Повторные замеры одной команды объединяются медианой измеренного угла.
Точки `(измеренный угол, командный угол)` с якорем `(0,0)` должны образовать
строго монотонную зависимость. Немонотонные данные, отсутствие движения или
неверное направление не принимаются. Линейный `gain = sum(command * measured) /
sum(command²)` и RMS его остатка выводятся только для диагностики. Gain вне
диапазона 0.25–4 требует проверки исходной механики. Большой RMS не запрещает
монотонную нелинейную кривую и не считается доказательством её точности.

Для каждого направления с минимум двумя реверсами оценка люфта — медиана
разности ожидаемого по прямой кривой и измеренного движения, взятой со знаком
направления. Реверсы вне диапазона команд кривой не используются. Отрицательное
значение сохраняется как признак шума/несоответствия модели. Малые движения,
не выбирающие люфт полностью, не подходят для этой модели.

Сохранение сверяет механическую конфигурацию выбранной оси с snapshot текущего
запуска и записывает только `relative-motion-curve`; количество шагов на оборот
не изменяется. Launcher callback использует существующий атомарный `save_config`,
затем application публикует ConfigUpdate и обновляет Core snapshot осей.
HAL применяет кривую к следующим командам без перезапуска — см.
[конфигурацию](../../architecture/configuration.md#кривая-калибровки-relative).
Кривая зависит от величины и знака relative move, но не моделирует абсолютное
положение, нагрузку, скорость, температуру или гистерезис. Эти зависимости
нужно проверять аппаратно; автоматической компенсации люфта нет.

Пользовательская процедура: [Калибровка движения](../../../user/motion-calibration.md).
