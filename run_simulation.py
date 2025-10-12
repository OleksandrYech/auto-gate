# run_simulation.py
import os
import logging
import cv2
import time
import threading
from typing import Dict, Any, Optional

os.environ['SIMULATION_MODE'] = '1'

from utils.logger_config import setup_global_logging
from core.camera_manager import CameraController
from core.gate_controller import GateController
from core.sheet_handler import SheetHandler
from core.cv_processor import CVProcessor
from core.vehicle_event_handler import VehicleEventHandler
from core.settings_manager import SettingsManager

try:
    from bot.telegram_notifier import TelegramNotifier
    from bot.bot_main import AUTHORIZED_USERS
except ImportError as e:
    logging.critical(f"Не вдалося імпортувати компоненти бота. Помилка: {e}")
    TelegramNotifier = None
    AUTHORIZED_USERS = set()

class MockSensorManager:
    def __init__(self):
        self._logger = logging.getLogger("SIMULATOR.MockSensorManager")
        self._logger.info("Модуль сенсорів (MockSensorManager) ініціалізовано.")
    def cleanup(self):
        self._logger.info("Очищення ресурсів MockSensorManager.")

class MockCameraController(CameraController):
    def __init__(self, config: Dict[str, Any], image_path: str):
        self.name = config.get("name", "MockCamera")
        self.resolution = config.get("resolution", (1280, 720))
        self.is_initialized_successfully = False
        self._logger = logging.getLogger(f"SIMULATOR.{self.name}")
        self._image = cv2.imread(image_path)
        if self._image is not None:
            self._image = cv2.resize(self._image, self.resolution)
            self.is_initialized_successfully = True
            self._logger.info(f"Імітаційна камера '{self.name}' завантажила зображення: {image_path}")
        else:
            self._logger.error(f"Не вдалося завантажити тестове зображення: {image_path}")
    def capture_array(self) -> Optional[cv2.typing.MatLike]:
        return self._image.copy() if self.is_initialized_successfully else None
    def close(self):
        self._logger.info(f"Імітаційна камера '{self.name}' закрито.")

TEST_IMAGE_PATH = "test_car.jpg"
DEBUG_SAVE_DIR = "simulation_output"
SIMULATION_DURATION_S = 15
TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN", "8392356130:AAHlCj5LFqKizWp17KKPTXetvjQiPq30i6U")
VEH_CONFIG: Dict[str, Any] = {
    "sheets_antiduplicate_delay_s": 10, "passage_timeout_s": 5,
    "gate_travel_time_s": 4, "poll_interval_idle_s": 1.0
}
CONFIG_DIR = "config"
MODELS_DIR = "models"
ROI_CONFIG_PATH = os.path.join(CONFIG_DIR, "roi_config.json")
SHEETS_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit#gid=0")
MOBILENET_SSD_PATH = os.path.join(MODELS_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license.onnx")
CAMERA_ENTRY_CONFIG: Dict[str, Any] = {"name": "EntryCamera", "resolution": (1280, 720)}

def run_simulation():
    logger = logging.getLogger("SIMULATOR")
    logger.info("--- ПОЧАТОК СИМУЛЯЦІЇ ---")

    if not os.path.exists(TEST_IMAGE_PATH):
        logger.error(f"Тестове зображення не знайдено: {TEST_IMAGE_PATH}")
        return

    logger.info("Ініціалізація компонентів...")
    entry_camera = MockCameraController(CAMERA_ENTRY_CONFIG, TEST_IMAGE_PATH)
    sensor_mgr = MockSensorManager()
    gate_ctrl = GateController(open_relay_pin=17, close_relay_pin=27)
    settings_mgr = SettingsManager()
    sheet_hndl = SheetHandler(
        credentials_file_path=SHEETS_CREDENTIALS_PATH,
        spreadsheet_url=SPREADSHEET_URL
    )
    cv_proc = CVProcessor(
        mobilenet_ssd_path=MOBILENET_SSD_PATH,
        license_model_path=LICENSE_PLATE_MODEL_PATH,
        roi_config_path=ROI_CONFIG_PATH
    )

    notifier = None
    if TelegramNotifier and TELEGRAM_BOT_TOKEN:
        notifier = TelegramNotifier(token=TELEGRAM_BOT_TOKEN)
        logger.info("TelegramNotifier налаштовано для розсилки авторизованим користувачам.")
    else:
        logger.warning("Токен TELEGRAM_BOT_TOKEN не вказано. Сповіщення вимкнено.")

    vehicle_event_hndl = VehicleEventHandler(
        camera_entry=entry_camera, camera_exit=None, sensor_manager=sensor_mgr,
        sheet_handler=sheet_hndl, cv_processor=cv_proc, gate_controller=gate_ctrl,
        config=VEH_CONFIG, notifier=notifier, settings_manager=settings_mgr
    )
    logger.info("Усі компоненти успішно ініціалізовано.")

    shutdown_event = threading.Event()
    try:
        vehicle_event_hndl.start(shutdown_event)
        logger.info(f"Симуляція працюватиме {SIMULATION_DURATION_S} секунд...")
        time.sleep(SIMULATION_DURATION_S)
        logger.info("Час симуляції вичерпано. Зупиняємо систему...")
    except KeyboardInterrupt:
        logger.info("Отримано сигнал переривання (Ctrl+C). Завершення роботи...")
    finally:
        shutdown_event.set()
        vehicle_event_hndl.stop()
        gate_ctrl.cleanup()
        sensor_mgr.cleanup()
        entry_camera.close()
    logger.info("--- СИМУЛЯЦІЯ ЗАВЕРШЕНА ---")

if __name__ == "__main__":
    setup_global_logging()
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
    if not os.path.exists(ROI_CONFIG_PATH):
        with open(ROI_CONFIG_PATH, 'w') as f:
            f.write('{"entry_camera_roi": {"x1": 0, "y1": 0, "x2": 1280, "y2": 720, "enabled": true}}')
    run_simulation()