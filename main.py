# main.py
import os
import logging
import threading
import time
from typing import Dict, Any, Optional

# --- ІМПОРТИ МОДУЛІВ ПРОЄКТУ ---
from utils.logger_config import setup_global_logging
from core.camera_manager import CameraManager
from core.gate_controller import GateController
from core.sheet_handler import SheetHandler
from core.cv_processor import CVProcessor
from core.vehicle_event_handler import VehicleEventHandler
from core.settings_manager import SettingsManager
from core.sensors_manager import SensorManager

try:
    from bot.telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None
    logging.warning("Не вдалося імпортувати TelegramNotifier.")

# --- ОСНОВНІ НАЛАШТУВАННЯ ---
# Ці налаштування беруться зі змінних середовища або використовуються значення за замовчуванням
TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN", "")
SPREADSHEET_URL: Optional[str] = os.getenv("SPREADSHEET_URL", "https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0")

# --- Конфігурації компонентів ---
# Шляхи до файлів конфігурації та моделей
CONFIG_DIR = "config"
MODELS_DIR = "models"
ROI_CONFIG_PATH = os.path.join(CONFIG_DIR, "roi_config.json")
SHEETS_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")
MOBILENET_SSD_PATH = os.path.join(MODELS_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license.onnx")

# Піни для реле (зміни на свої, якщо потрібно)
OPEN_RELAY_PIN: int = 17
CLOSE_RELAY_PIN: int = 27

# Конфігурація камер (RTSP-потоки)
# Заміни 'YOUR_RTSP_STREAM_URL' на реальні посилання з твоїх камер
CAMERA_ENTRY_CONFIG: Dict[str, Any] = {
    "name": "EntryCamera",
    "url": os.getenv("ENTRY_CAM_URL", "YOUR_RTSP_STREAM_URL_1"),
    "resolution": (1280, 720)
}
CAMERA_EXIT_CONFIG: Dict[str, Any] = {
    "name": "ExitCamera",
    "url": os.getenv("EXIT_CAM_URL", "YOUR_RTSP_STREAM_URL_2"),
    "resolution": (1280, 720)
}

# Конфігурація обробника подій
VEHICLE_EVENT_CONFIG: Dict[str, Any] = {
    "sheets_antiduplicate_delay_s": 60,
    "passage_timeout_s": 5,
    "gate_travel_time_s": 15,
    "poll_interval_idle_s": 0.5  # Частіше опитування для реального часу
}


def main():
    """Головна функція для запуску системи."""
    setup_global_logging()
    logger = logging.getLogger(__name__)
    logger.info("--- ЗАПУСК СИСТЕМИ AUTO-GATE ---")

    # Створюємо подію для коректної зупинки всіх потоків
    shutdown_event = threading.Event()

    # Ініціалізація всіх компонентів системи
    try:
        logger.info("Ініціалізація компонентів...")

        settings_mgr = SettingsManager()
        cam_manager = CameraManager(
            entry_cam_config=CAMERA_ENTRY_CONFIG,
            exit_cam_config=CAMERA_EXIT_CONFIG
        )
        sensor_mgr = SensorManager()
        gate_ctrl = GateController(
            open_relay_pin=OPEN_RELAY_PIN,
            close_relay_pin=CLOSE_RELAY_PIN
        )
        sheet_hndl = SheetHandler(
            credentials_file_path=SHEETS_CREDENTIALS_PATH,
            spreadsheet_url=SPREADSHEET_URL
        )
        cv_proc = CVProcessor(
            mobilenet_ssd_path=MOBILENET_SSD_PATH,
            license_model_path=LICENSE_PLATE_MODEL_PATH,
            roi_config_path=ROI_CONFIG_PATH
        )

        # Ініціалізація сповіщувача (він сам читатиме файл з користувачами)
        notifier = None
        if TelegramNotifier and TELEGRAM_BOT_TOKEN:
            notifier = TelegramNotifier(token=TELEGRAM_BOT_TOKEN)
            logger.info("TelegramNotifier налаштовано.")
        else:
            logger.warning("Токен TELEGRAM_BOT_TOKEN не вказано, сповіщення вимкнено.")

        # Ініціалізація головного обробника логіки
        vehicle_event_hndl = VehicleEventHandler(
            camera_entry=cam_manager.get_entry_camera(),
            camera_exit=cam_manager.get_exit_camera(),
            sensor_manager=sensor_mgr,
            sheet_handler=sheet_hndl,
            cv_processor=cv_proc,
            gate_controller=gate_ctrl,
            config=VEHICLE_EVENT_CONFIG,
            notifier=notifier,
            settings_manager=settings_mgr
        )

        logger.info("Усі компоненти успішно ініціалізовано.")

        # Запуск основного циклу обробки в окремому потоці
        vehicle_event_hndl.start(shutdown_event)

        # Головний потік чекає на сигнал зупинки (наприклад, Ctrl+C)
        while not shutdown_event.is_set():
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Отримано сигнал переривання (Ctrl+C). Завершення роботи...")
    except Exception as e:
        logger.critical(f"Критична помилка під час ініціалізації або роботи: {e}", exc_info=True)
    finally:
        logger.info("Початок процедури коректної зупинки...")
        shutdown_event.set()

        # Зупиняємо всі компоненти у зворотному порядку
        if 'vehicle_event_hndl' in locals():
            vehicle_event_hndl.stop()
        if 'gate_ctrl' in locals():
            gate_ctrl.cleanup()
        if 'sensor_mgr' in locals():
            sensor_mgr.cleanup()
        if 'cam_manager' in locals():
            cam_manager.close_all_cameras()

        logger.info("--- СИСТЕМА AUTO-GATE ЗУПИНЕНА ---")


if __name__ == "__main__":
    main()