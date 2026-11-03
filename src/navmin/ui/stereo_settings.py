"""Runtime stereo tuning without modifying calibration or camera geometry."""

from dataclasses import replace

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
)

from navmin.vision.stereo_distance_worker import StereoDistanceTuning


class StereoSettingsWindow(QDialog):
    def __init__(self, owner, get_tuning, apply_tuning):
        super().__init__(owner, Qt.WindowType.Tool)
        self.setWindowTitle("Параметры стерео")
        self._get_tuning = get_tuning
        self._apply_tuning = apply_tuning
        self._baseline = get_tuning()
        self.editors = {}
        layout = QVBoxLayout(self)
        info = QLabel(
            "Дальность — медиана N успешных стереопар. До накопления N пар показания нет. "
            "Больше кадров — устойчивее результат, но дольше ожидание.\n"
            "Параметры применяются только к текущему запуску. Калибровка камер не меняется."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        tabs = QTabWidget()

        def group(title, *, tab=False):
            box = QGroupBox(title)
            if tab:
                tabs.addTab(box, title)
            else:
                layout.addWidget(box)
            return QFormLayout(box)

        def field(form, key, label, value, minimum, maximum, step=1, decimals=None):
            editor = QSpinBox() if decimals is None else QDoubleSpinBox()
            if decimals is not None:
                editor.setDecimals(decimals)
            editor.setRange(minimum, max(maximum, value))
            editor.setSingleStep(step)
            editor.setValue(value)
            self.editors[key] = editor
            form.addRow(label, editor)

        tuning = self._baseline
        common = group("Оба метода")
        field(
            common,
            "aggregation_frame_count",
            "Число пар кадров для решения",
            tuning.aggregation_frame_count,
            1,
            1000,
        )
        layout.addWidget(tabs)
        roi = group("ROI / SGBM", tab=True)
        field(
            roi,
            "min_valid_pixel_count",
            "Минимум достоверных пикселей ROI",
            tuning.min_valid_pixel_count,
            1,
            100000,
        )
        field(
            roi,
            "num_disparities",
            "Диапазон поиска, пиксели (кратно 16)",
            tuning.sgbm.num_disparities,
            16,
            4096,
            16,
        )
        field(
            roi,
            "block_size",
            "Размер блока (нечётный)",
            tuning.sgbm.block_size,
            3,
            31,
            2,
        )
        field(
            roi,
            "uniqueness_ratio",
            "Порог уникальности, %",
            tuning.sgbm.uniqueness_ratio,
            0,
            100,
        )
        field(
            roi,
            "speckle_window_size",
            "Фильтр мелких областей, пиксели",
            tuning.sgbm.speckle_window_size,
            0,
            10000,
        )
        field(
            roi,
            "processing_scale",
            "Масштаб обработки (меньше — быстрее)",
            tuning.sgbm.processing_scale,
            0.25,
            1,
            0.05,
            2,
        )
        detector = group("Два детектора", tab=True)
        field(
            detector,
            "vertical_tolerance_px",
            "Допуск по вертикали, пиксели",
            tuning.detector.vertical_tolerance_px,
            0,
            100,
            1,
            1,
        )
        field(
            detector,
            "vertical_height_fraction",
            "Допуск по вертикали, доля высоты",
            tuning.detector.vertical_height_fraction,
            0,
            2,
            0.05,
            2,
        )
        field(
            detector,
            "min_height_ratio",
            "Минимальное отношение высот R/L",
            tuning.detector.min_height_ratio,
            0.01,
            1,
            0.05,
            2,
        )
        field(
            detector,
            "max_height_ratio",
            "Максимальное отношение высот R/L",
            tuning.detector.max_height_ratio,
            1,
            10,
            0.1,
            2,
        )
        field(
            detector,
            "ambiguity_margin",
            "Запас однозначности сопоставления",
            tuning.detector.ambiguity_margin,
            0,
            1,
            0.01,
            3,
        )
        self.apply_button = QPushButton("Применить и начать новое накопление")
        self.apply_button.clicked.connect(self._apply)
        layout.addWidget(self.apply_button)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _apply(self):
        try:
            values = {key: editor.value() for key, editor in self.editors.items()}
            sgbm_keys = (
                "num_disparities",
                "block_size",
                "uniqueness_ratio",
                "speckle_window_size",
                "processing_scale",
            )
            detector_keys = (
                "vertical_tolerance_px",
                "vertical_height_fraction",
                "min_height_ratio",
                "max_height_ratio",
                "ambiguity_margin",
            )
            tuning = StereoDistanceTuning(
                values["aggregation_frame_count"],
                values["min_valid_pixel_count"],
                replace(self._baseline.sgbm, **{key: values[key] for key in sgbm_keys}),
                replace(
                    self._baseline.detector,
                    **{key: values[key] for key in detector_keys},
                ),
            )
            self._apply_tuning(tuning)
            self._baseline = self._get_tuning()
            self.status.setText(
                f"Применено. Ожидание {tuning.aggregation_frame_count} успешных пар кадров; перезапуск не нужен."
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))
