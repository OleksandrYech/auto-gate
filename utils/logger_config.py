# utils/logger_config.py
import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
LOG_FILE = "gate_system.log"

def setup_global_logging():
    """
    Налаштовує глобальну систему логування з перевіркою,
    щоб уникнути дублювання обробників.
    """
    root_logger = logging.getLogger()
    # Якщо логер вже має обробники, нічого не робимо
    if root_logger.hasHandlers():
        logging.debug("Логер вже налаштовано. Пропускаємо повторну ініціализацію.")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_FILE)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    console_formatter = logging.Formatter('[ %(levelname)s] %(name)s: %(message)s')

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    logging.info("Глобальну систему логування налаштовано.")