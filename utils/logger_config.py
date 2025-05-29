# utils/logger_config.py
import logging
import logging.handlers  # Для RotatingFileHandler
import os
import sys

# --- Константи для логування ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LOG_FILENAME = "gate_system.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

# Максимальний розмір файлу логів та кількість резервних копій
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Формати логів
# Детальний формат для файлу
FILE_LOG_FORMAT = '%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
# Простіший формат для консолі
CONSOLE_LOG_FORMAT = '[%(levelname)s] %(name)s: %(message)s'

# Рівні логування
# Глобальний рівень для вашого застосунку (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# Для розробки DEBUG корисний, для продакшену INFO або WARNING
APP_LOG_LEVEL = logging.DEBUG
CONSOLE_LOG_LEVEL = logging.INFO  # Можна зробити консоль менш детальною

# Рівні для сторонніх бібліотек (щоб зменшити шум)
THIRD_PARTY_LOG_LEVELS = {
    "picamera2": logging.WARNING,
    "gspread": logging.INFO,
    "oauth2client": logging.WARNING,
    "onnxruntime": logging.WARNING,  # Може бути дуже детальним
    "ultralytics": logging.INFO,  # Або WARNING, якщо забагато логів
    "PIL": logging.INFO,  # Pillow (часто використовується з зображеннями)
}


def setup_global_logging(app_log_level: int = APP_LOG_LEVEL,
                         console_log_level: int = CONSOLE_LOG_LEVEL):
    """
    Налаштовує глобальну систему логування для всього застосунку.
    Цю функцію слід викликати один раз на самому початку роботи main.py.
    """
    try:
        # Створюємо директорію для логів, якщо її немає
        os.makedirs(LOG_DIR, exist_ok=True)

        # Отримуємо кореневий логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(app_log_level)  # Встановлюємо мінімальний рівень для обробки

        # --- Обробник для файлу з ротацією ---
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'  # Важливо для кирилиці та інших символів
        )
        file_formatter = logging.Formatter(FILE_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(app_log_level)  # Рівень для файлу такий же, як глобальний
        root_logger.addHandler(file_handler)

        # --- Обробник для консолі ---
        console_handler = logging.StreamHandler(sys.stdout)  # Вивід у стандартний потік
        console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(console_log_level)  # Можна встановити інший рівень для консолі
        root_logger.addHandler(console_handler)

        # --- Налаштування рівнів для сторонніх бібліотек ---
        for lib_name, level in THIRD_PARTY_LOG_LEVELS.items():
            logging.getLogger(lib_name).setLevel(level)

        # Повідомлення про успішне налаштування (після додавання обробників)
        initial_logger = logging.getLogger(__name__)  # Логгер для цього модуля
        initial_logger.info(
            f"Глобальну систему логування налаштовано. Рівень застосунку: {logging.getLevelName(app_log_level)}.")
        initial_logger.info(f"Логи зберігаються у: {LOG_FILE_PATH}")
        initial_logger.info(f"Рівень логування консолі: {logging.getLevelName(console_log_level)}")

    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Критична помилка під час налаштування системи логування!", exc_info=e)


if __name__ == '__main__':
    # Цей блок для тестування самого logger_config.py
    print("Тестування конфігурації логування...")

    # Налаштовуємо з рівнями за замовчуванням
    setup_global_logging()

    # Тестові повідомлення з різних логгерів
    main_test_logger = logging.getLogger("MyMainAppTest")
    module_test_logger = logging.getLogger("MyModuleTest.SubModule")

    main_test_logger.debug(
        "Це тестове DEBUG повідомлення від MyMainAppTest.")  # Не з'явиться в консолі, якщо CONSOLE_LOG_LEVEL > DEBUG
    main_test_logger.info("Це тестове INFO повідомлення від MyMainAppTest.")
    main_test_logger.warning("Це тестове WARNING повідомлення від MyMainAppTest.")

    module_test_logger.error("Це тестове ERROR повідомлення від MyModuleTest.SubModule.")

    # Тест логування від сторонньої бібліотеки (імітація)
    # Якщо picamera2 встановлено на WARNING, це повідомлення не з'явиться
    logging.getLogger("picamera2").info("Тестове INFO повідомлення від picamera2 (має бути приглушене).")
    logging.getLogger("picamera2").warning("Тестове WARNING повідомлення від picamera2 (має з'явитися).")

    print(f"Тестування завершено. Перевірте файл логів: {LOG_FILE_PATH} та вивід у консолі.")