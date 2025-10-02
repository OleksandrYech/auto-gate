# core/settings_manager.py
import json
import logging
import os
from threading import Lock

logger = logging.getLogger(__name__)

class SettingsManager:
    """
    Клас для керування налаштуваннями, що зберігаються у файлі.
    Забезпечує потокобезпечний доступ до налаштувань.
    """
    def __init__(self, settings_file_path: str = "config/bot_settings.json"):
        self.settings_path = settings_file_path
        self._lock = Lock()  # Замок для потокобезпечного доступу
        self.settings = self._load_settings()

    def _load_settings(self) -> dict:
        """Завантажує налаштування з JSON-файлу."""
        with self._lock:
            try:
                if os.path.exists(self.settings_path):
                    with open(self.settings_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    logger.warning(f"Файл налаштувань '{self.settings_path}' не знайдено. Буде створено новий зі значеннями за замовчуванням.")
                    # Повертаємо налаштування за замовчуванням
                    return self._get_default_settings()
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Помилка завантаження налаштувань з '{self.settings_path}': {e}. Використовуються значення за замовчуванням.")
                return self._get_default_settings()

    def _save_settings(self):
        """Зберігає поточні налаштування у JSON-файл."""
        with self._lock:
            try:
                # Переконуємося, що директорія існує
                os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
                with open(self.settings_path, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, indent=4)
            except IOError as e:
                logger.error(f"Не вдалося зберегти налаштування у файл '{self.settings_path}': {e}")

    def _get_default_settings(self) -> dict:
        """Повертає словник з налаштуваннями за замовчуванням."""
        return {
            "notifications_enabled": True
        }

    def are_notifications_enabled(self) -> bool:
        """Перевіряє, чи увімкнені сповіщення."""
        with self._lock:
            return self.settings.get("notifications_enabled", True)

    def set_notifications_enabled(self, status: bool):
        """Встановлює стан сповіщень (True або False)."""
        with self._lock:
            if self.settings.get("notifications_enabled") == status:
                logger.debug(f"Статус сповіщень вже встановлено як {status}. Зміни не потрібні.")
                return

            self.settings["notifications_enabled"] = status
            logger.info(f"Статус сповіщень змінено на: {'УВІМКНЕНО' if status else 'ВИМКНЕНО'}.")
        self._save_settings()
