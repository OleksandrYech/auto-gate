# main.py
import logging
import os
import signal
import sys
import threading
from typing import Dict, Any

# Припускаємо, що ці файли існують у відповідних директоріях
from utils.logger_config import setup_global_logging
from core.camera_manager import CameraManager
from core.sensors_manager import SensorManager
from core.sheet_handler import SheetHandler
from core.gate_controller import GateController
from core.cv_processor import CVProcessor
from core.vehicle_event_handler import VehicleEventHandler

# --- Конфігурація обробника подій ---
VEH_CONFIG: Dict[str, Any] = {
    "sheets_antiduplicate_delay_s": 60,
    "passage_timeout_s": 20,
    "interrupted_passage_timeout_s": 30,
    "gate_travel_time_s": 15,
    "reed_open_timeout_s": 15,
    "poll_interval_idle_s": 1.0
}

# --- Апаратна конфігурація ---
REED_SWITCH_PIN: int = 22
OPEN_RELAY_PIN: int = 17
CLOSE_RELAY_PIN: int = 27

# --- Шляхи до файлів ---
CONFIG_DIR = "config"
MODELS_DIR = "models"
ROI_CONFIG_PATH = os.path.join(CONFIG_DIR, "roi_config.json")
SHEETS_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")
MOBILENET_SSD_PATH = os.path.join(MODELS_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license.onnx")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "ocr.pt")

# --- Конфігурація камер ---
CAMERA_ENTRY_CONFIG: Dict[str, Any] = {"name": "EntryCamera", "resolution": (1280, 720)}
CAMERA_EXIT_CONFIG: Dict[str, Any] = {"name": "ExitCamera", "resolution": (1280, 720)}

# --- Конфігурація Sheets --
SHEETS_URL = "https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0"


def main_application():
    """Головна функція запуску та керування системою."""
    logger = logging.getLogger(__name__)
    logger.info("Запуск автоматизованої системи керування воротами...")

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
        sheet_hndl = SheetHandler(credentials_file_path=SHEETS_CREDENTIALS_PATH,
        spreadsheet_url=SHEETS_URL)
        if not sheet_hndl._client:
            logger.critical("SheetHandler не зміг підключитися. Перевірте файл credentials.json.")
            return

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
        if not sensor_mgr.reed_switch:
            logger.critical("Не вдалося ініціалізувати геркон. Завершення.")
            return

        # 5. Ініціалізація контролера воріт
        logger.info("Ініціалізація GateController...")
        gate_ctrl = GateController(
            open_relay_pin=OPEN_RELAY_PIN,
            close_relay_pin=CLOSE_RELAY_PIN,
            relay_pulse_duration_s=0.5
        )
        if not gate_ctrl.relays_initialized:
            logger.critical("Реле в GateController не ініціалізовано. Завершення.")
            return

        # 6. Ініціалізація головного обробника логіки
        logger.info("Ініціалізація VehicleEventHandler...")
        vehicle_event_hndl = VehicleEventHandler(
            camera_entry=cam_manager.get_entry_camera(),
            camera_exit=cam_manager.get_exit_camera(),
            sensor_manager=sensor_mgr,
            sheet_handler=sheet_hndl,
            cv_processor=cv_proc,
            gate_controller=gate_ctrl,
            config=VEH_CONFIG
        )

        # 7. Налаштування коректного завершення роботи
        shutdown_event = threading.Event()
        def signal_handler(sig, frame):
            logger.warning(f"Отримано сигнал {signal.Signals(sig).name}. Завершення роботи...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 8. Запуск системи
        vehicle_event_hndl.start(shutdown_event)
        logger.info("Система запущена. Натисніть Ctrl+C для завершення.")
        shutdown_event.wait()  # Очікуємо на сигнал Ctrl+C

    except Exception as e:
        logger.critical(f"Неперехоплена помилка в main_application: {e}", exc_info=True)
    finally:
        # 9. Коректне звільнення ресурсів
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
    setup_global_logging()
    # Рекомендація: перед запуском виконайте в терміналі `sudo pigpiod`
    main_application()
