# main.py
import logging
import os
import signal
import sys
import threading
from typing import Dict, Any, Optional

# Імпортуємо всі необхідні модулі проєкту
from utils.logger_config import setup_global_logging
from core.camera_manager import CameraManager
from core.sensors_manager import SensorManager
from core.sheet_handler import SheetHandler
from core.gate_controller import GateController
from core.cv_processor import CVProcessor
from core.vehicle_event_handler import VehicleEventHandler

# Опціональний імпорт нотифікатора
try:
    from bot.telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None

# --- КОНФІГУРАЦІЯ ОБРОБНИКА ПОДІЙ ---
VEH_CONFIG: Dict[str, Any] = {
    "sheets_antiduplicate_delay_s": 60,
    "passage_timeout_s": 20,
    "interrupted_passage_timeout_s": 30,
    "gate_travel_time_s": 15,
    "reed_open_timeout_s": 15,
    "poll_interval_idle_s": 1.0
}

# --- АПАРАТНА КОНФІГУРАЦІЯ ---
REED_SWITCH_PIN: int = 22
OPEN_RELAY_PIN: int = 17
CLOSE_RELAY_PIN: int = 27

# --- КОНФІГУРАЦІЯ TELEGRAM ---
# Рекомендується зберігати ці дані у змінних середовища для безпеки
TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("8392356130:AAHlCj5LFqKizWp17KKPTXetvjQiPq30i6U")
TELEGRAM_CHAT_ID: Optional[str] = os.getenv("591969753")

# --- ШЛЯХИ ДО ФАЙЛІВ ТА URL ---
CONFIG_DIR = "config"
MODELS_DIR = "models"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0"
CAPTURES_DIR = "captures" # Папка для збереження зображень з розпізнаними номерами
ROI_CONFIG_PATH = os.path.join(CONFIG_DIR, "roi_config.json")
SHEETS_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")
MOBILENET_SSD_PATH = os.path.join(MODELS_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license.onnx")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "ocr.pt")

# --- КОНФІГУРАЦІЯ КАМЕР ---
CAMERA_ENTRY_CONFIG: Dict[str, Any] = {"name": "EntryCamera", "resolution": (1280, 720)}
CAMERA_EXIT_CONFIG: Dict[str, Any] = {"name": "ExitCamera", "resolution": (1280, 720)}


def main_application():
    """Головна функція запуску та керування системою."""
    logger = logging.getLogger(__name__)
    logger.info("Запуск автоматизованої системи керування воротами...")

    # Створюємо папку для зображень, якщо її немає
    os.makedirs(CAPTURES_DIR, exist_ok=True)

    # Ініціалізуємо змінні як None на випадок помилки під час запуску
    cam_manager = None
    sensor_mgr = None
    gate_ctrl = None
    vehicle_event_hndl = None

    try:
        # 1. Ініціалізація камер
        logger.info("Ініціалізація CameraManager...")
        cam_manager = CameraManager(
            entry_cam_model_sub='imx219',
            exit_cam_model_sub='imx219',
            entry_cam_config=CAMERA_ENTRY_CONFIG,
            exit_cam_config=CAMERA_EXIT_CONFIG
        )

        # 2. Ініціалізація Google Sheets
        logger.info("Ініціалізація SheetHandler...")
        if not SPREADSHEET_URL:
            logger.critical("URL Google Таблиці не вказано! Встановіть змінну середовища SPREADSHEET_URL.")
            return
        sheet_hndl = SheetHandler(
            credentials_file_path=SHEETS_CREDENTIALS_PATH,
            spreadsheet_url=SPREADSHEET_URL
        )

        # 3. Ініціалізація комп'ютерного зору
        logger.info("Ініціалізація CVProcessor...")
        cv_proc = CVProcessor(
            mobilenet_ssd_path=MOBILENET_SSD_PATH,
            license_model_path=LICENSE_PLATE_MODEL_PATH,
            ocr_model_path=OCR_MODEL_PATH,
            roi_config_path=ROI_CONFIG_PATH
        )

        # 4. Ініціалізація сенсорів
        logger.info("Ініціалізація SensorManager...")
        sensor_mgr = SensorManager(reed_pin=REED_SWITCH_PIN)

        # 5. Ініціалізація контролера воріт
        logger.info("Ініціалізація GateController...")
        gate_ctrl = GateController(
            open_relay_pin=OPEN_RELAY_PIN,
            close_relay_pin=CLOSE_RELAY_PIN
        )

        # 6. Ініціалізація Telegram нотифікатора
        notifier = None
        if TelegramNotifier and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            logger.info("Ініціалізація TelegramNotifier...")
            notifier = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, chat_id=int(TELEGRAM_CHAT_ID))
        else:
            logger.warning("Токен/ID чату для Telegram не вказано або модуль не знайдено. Сповіщення вимкнено.")

        # 7. Ініціалізація головного обробника логіки
        logger.info("Ініціалізація VehicleEventHandler...")
        vehicle_event_hndl = VehicleEventHandler(
            camera_entry=cam_manager.get_entry_camera(),
            camera_exit=cam_manager.get_exit_camera(),
            sensor_manager=sensor_mgr,
            sheet_handler=sheet_hndl,
            cv_processor=cv_proc,
            gate_controller=gate_ctrl,
            config=VEH_CONFIG,
            notifier=notifier  # Передаємо нотифікатор
        )

        # 8. Налаштування коректного завершення роботи
        shutdown_event = threading.Event()
        def signal_handler(sig, frame):
            logger.warning(f"Отримано сигнал {signal.Signals(sig).name}. Завершення роботи...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 9. Запуск системи
        vehicle_event_hndl.start(shutdown_event)
        logger.info("Система запущена. Натисніть Ctrl+C для завершення.")
        shutdown_event.wait()  # Очікуємо на сигнал Ctrl+C

    except Exception as e:
        logger.critical(f"Неперехоплена помилка в main_application: {e}", exc_info=True)
    finally:
        # 10. Коректне звільнення ресурсів
        logger.info("Початок процедури коректного завершення роботи...")
        if vehicle_event_hndl:
            vehicle_event_hndl.stop()
        if gate_ctrl:
            gate_ctrl.cleanup()
        if sensor_mgr:
            sensor_mgr.cleanup()
        if cam_manager:
            cam_manager.close_all_cameras()
        logger.info("Система завершила роботу.")


if __name__ == "__main__":
    # Налаштовуємо логування один раз при старті
    setup_global_logging()

    # Рекомендація: перед запуском виконайте в терміналі `sudo pigpiod` для стабільної роботи GPIO
    main_application()
