# utils/logger_config.py
import logging
import logging.handlers  # Для RotatingFileHandler
import os
import sys

# --- Константи для логування ---

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_FROM_UTILS = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, ".."))

LOG_DIR = os.path.join(PROJECT_ROOT_FROM_UTILS, "logs")
LOG_FILENAME = "gate_system.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

# Максимальний розмір файлу логів та кількість резервних копій
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Формати логів
FILE_LOG_FORMAT = '%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
CONSOLE_LOG_FORMAT = '[%(levelname)8s] %(name)s: %(message)s'  # Вирівнювання для levelname

# Рівні логування
APP_LOG_LEVEL = logging.DEBUG  # Головний рівень для вашого застосунку
CONSOLE_LOG_LEVEL = logging.INFO  # Рівень для виводу в консоль

# Рівні для сторонніх бібліотек, щоб зменшити "шум"
THIRD_PARTY_LOG_LEVELS = {
    "picamera2": logging.WARNING,
    "gspread": logging.INFO,  # gspread може бути досить балакучим на DEBUG
    "oauth2client": logging.WARNING,  # Стара бібліотека, зазвичай INFO або WARNING достатньо
    "googleapiclient": logging.WARNING,  # Частина Google API клієнта
    "onnxruntime": logging.WARNING,  # Може бути дуже детальним, особливо з попередженнями про CUDA
    "ultralytics": logging.INFO,  # Або WARNING, якщо забагато логів
    "PIL.PngImagePlugin": logging.INFO,  # Pillow може видавати багато DEBUG повідомлень
}


def setup_global_logging(app_log_level: int = APP_LOG_LEVEL,
                         console_log_level: int = CONSOLE_LOG_LEVEL,
                         log_file_path: str = LOG_FILE_PATH):
    """
    Налаштовує глобальну систему логування для всього застосунку.
    Цю функцію слід викликати один раз на самому початку роботи main.py.

    Args:
        app_log_level (int): Мінімальний рівень логування для файлу та кореневого логгера.
        console_log_level (int): Мінімальний рівень логування для консолі.
        log_file_path (str): Повний шлях до файлу логів.
    """
    # Перевіряємо, чи логування вже налаштовано (наприклад, у тестах)
    # Простий спосіб - перевірити, чи є обробники у кореневого логгера
    # if logging.getLogger().hasHandlers() and not os.environ.get("PYTEST_CURRENT_TEST"):
    #     logging.getLogger(__name__).info("Логування вже було налаштовано раніше.")
    #     return

    try:
        log_directory = os.path.dirname(log_file_path)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory, exist_ok=True)

        root_logger = logging.getLogger()

        root_logger.setLevel(min(app_log_level, console_log_level))  # Встановлюємо найнижчий з потрібних рівнів

        # --- Обробник для файлу з ротацією ---
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(FILE_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(app_log_level)
        root_logger.addHandler(file_handler)

        # --- Обробник для консолі ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(console_log_level)
        root_logger.addHandler(console_handler)

        # --- Налаштування рівнів для сторонніх бібліотек ---
        for lib_name, level in THIRD_PARTY_LOG_LEVELS.items():
            logging.getLogger(lib_name).setLevel(level)

        initial_logger = logging.getLogger(__name__)  # Логгер для цього модуля
        initial_logger.info(
            f"Глобальну систему логування налаштовано. Рівень застосунку (файл): {logging.getLevelName(app_log_level)}.")
        initial_logger.info(f"Логи зберігаються у: {log_file_path}")
        initial_logger.info(f"Рівень логування консолі: {logging.getLevelName(console_log_level)}")

    except Exception as e:
        logging.basicConfig(level=logging.ERROR)  # Fallback, якщо щось пішло не так
        logging.exception("Критична помилка під час налаштування системи логування!", exc_info=True)


if __name__ == '__main__':
    print(f"Тестування конфігурації логування. Файл логів буде: {LOG_FILE_PATH}")

    setup_global_logging()

    test_logger_main = logging.getLogger("MyMainAppTest")  # Імітація логгера з main.py
    module_test_logger = logging.getLogger("MyModuleTest.SubModule")  # Імітація логгера з іншого модуля

    test_logger_main.debug("Це тестове DEBUG повідомлення від MyMainAppTest.")
    test_logger_main.info("Це тестове INFO повідомлення від MyMainAppTest.")
    test_logger_main.warning("Це тестове WARNING повідомлення від MyMainAppTest.")

    module_test_logger.error("Це тестове ERROR повідомлення від MyModuleTest.SubModule.")
    module_test_logger.critical("Це тестове CRITICAL повідомлення від MyModuleTest.SubModule.")

    try:
        1 / 0
    except ZeroDivisionError:
        module_test_logger.exception("Тестовий виняток з трасуванням стеку.")

    # Тест логування від сторонньої бібліотеки (імітація)
    logging.getLogger("picamera2").debug("Тестове DEBUG повідомлення від picamera2 (не має з'явитися).")
    logging.getLogger("picamera2").info(
        "Тестове INFO повідомлення від picamera2 (не має з'явитися, якщо рівень WARNING).")
    logging.getLogger("picamera2").warning("Тестове WARNING повідомлення від picamera2 (має з'явитися).")
    logging.getLogger("ultralytics").debug("Тестове DEBUG повідомлення від ultralytics (не має з'явитися).")
    logging.getLogger("ultralytics").info("Тестове INFO повідомлення від ultralytics (має з'явитися).")

    print(f"Тестування завершено. Перевірте файл логів: {os.path.abspath(LOG_FILE_PATH)} та вивід у консолі.")
    print(f"Рівень APP_LOG_LEVEL (для файлу): {logging.getLevelName(APP_LOG_LEVEL)}")
    print(f"Рівень CONSOLE_LOG_LEVEL (для консолі): {logging.getLevelName(CONSOLE_LOG_LEVEL)}")