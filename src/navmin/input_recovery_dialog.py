"""Qt dialogs for explicit startup input recovery."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtWidgets import QApplication, QMessageBox

from navmin.input_recovery import InputRecoveryResult


def _application(argv: Sequence[str] = ()) -> QApplication:
    application = QApplication.instance()
    if application is None:
        application = QApplication(list(argv))
    return application


def prompt_input_recovery(error_text: str, *, argv: Sequence[str] = ()) -> bool:
    """Return True only when the operator explicitly chooses recovery."""
    _application(argv)
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Ошибка входных данных NavMin")
    box.setText("Не удалось загрузить настройки или калибровки.")
    box.setInformativeText(error_text)
    box.setDetailedText(
        "Восстановление сохранит существующие локальные файлы как timestamped .bak "
        "и создаст безопасные настройки по умолчанию. После восстановления NavMin "
        "завершится: перед следующим запуском необходимо проверить параметры железа."
    )
    close_button = box.addButton(
        "Закрыть программу", QMessageBox.ButtonRole.RejectRole
    )
    restore_button = box.addButton(
        "Восстановить настройки по умолчанию",
        QMessageBox.ButtonRole.AcceptRole,
    )
    box.setDefaultButton(close_button)
    box.exec()
    return box.clickedButton() is restore_button


def show_input_recovery_result(
    result: InputRecoveryResult,
    *,
    argv: Sequence[str] = (),
) -> None:
    _application(argv)
    backup_lines = [f"{source} → {backup}" for source, backup in result.backups]
    backup_text = "\n".join(backup_lines) if backup_lines else "Backup не требовался."
    QMessageBox.information(
        None,
        "Настройки NavMin восстановлены",
        "Безопасные настройки по умолчанию созданы.\n\n"
        "NavMin сейчас завершит запуск. Перед повторным запуском обязательно "
        "проверьте serial port, механику осей, ограничения движения и реальные "
        "калибровки камер.\n\n"
        f"Backup:\n{backup_text}",
    )


def show_input_recovery_failure(message: str, *, argv: Sequence[str] = ()) -> None:
    _application(argv)
    QMessageBox.critical(
        None,
        "Не удалось восстановить настройки NavMin",
        message,
    )


__all__ = [
    "prompt_input_recovery",
    "show_input_recovery_failure",
    "show_input_recovery_result",
]
