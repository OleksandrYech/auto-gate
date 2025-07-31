# run_simulation.py
import os
import logging
import cv2
import time
from typing import Dict, Any, Optional

# Встановлюємо змінну середовища ДО імпорту модулів проєкту
os.environ['SIMULATION_MODE'] = '0'

# Імпортуємо всі необхідні компоненти
from utils.logger_config import setup_global_logging
from core.camera_manager import CameraManager
from core.gate_controller import GateController
from core.sheet_handler import SheetHandler
from core.cv_processor import CVProcessor
from core.vehicle_event_handler import VehicleEventHandler
try:
    from bot.telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None

# --- КЛАСИ-ІМІТАТОРИ ДЛЯ ГЕРКОНУ ---
class MockReedSwitch:
    def __init__(self):
        self._logger = logging.getLogger("SIMULATOR.MockReedSwitch")
        self.is_gate_open = False
    def wait_for_open(self, timeout: float = 5.0) -> bool:
        self._logger.info("[ІМІТАЦІЯ] Геркон: Очікування сигналу про відкриття воріт...")
        time.sleep(2.5)
        self.is_gate_open = True
        self._logger.info("[ІМІТАЦІЯ] Геркон: Сигнал отримано! Ворота вважаються відкритими.")
        return True
    @property
    def are_gates_open(self) -> bool: return self.is_gate_open
    def cleanup(self): self._logger.info("[ІМІТАЦІЯ] Геркон: Ресурси очищено.")

class MockSensorManager:
    def __init__(self):
        self._logger = logging.getLogger("SIMULATOR.MockSensorManager")
        self.reed_switch = MockReedSwitch()
        self._logger.info("Модуль сенсорів (MockSensorManager) ініціалізовано.")
    def cleanup(self): self.reed_switch.cleanup()
# -------------------------------------

# --- НАЛАШТУВАННЯ СИМУЛЯЦІЇ ---
TEST_IMAGE_PATH = "test_car.jpg"
DEBUG_SAVE_DIR = "simulation_output"

# Додаємо конфігурацію Telegram, як у main.py
TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN", "8392356130:AAHlCj5LFqKizWp17KKPTXetvjQiPq30i6U")
TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID", "591969753")

VEH_CONFIG: Dict[str, Any] = {
    "sheets_antiduplicate_delay_s": 60, "passage_timeout_s": 3,
    "interrupted_passage_timeout_s": 5, "gate_travel_time_s": 4,
    "reed_open_timeout_s": 5, "poll_interval_idle_s": 1.0
}
OPEN_RELAY_PIN: int = 17
CLOSE_RELAY_PIN: int = 27
CONFIG_DIR = "config"
MODELS_DIR = "models"
ROI_CONFIG_PATH = os.path.join(CONFIG_DIR, "roi_config.json")
SHEETS_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0")
MOBILENET_SSD_PATH = os.path.join(MODELS_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license.onnx")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "ocr.pt")
CAMERA_ENTRY_CONFIG: Dict[str, Any] = {"name": "EntryCamera", "resolution": (1280, 720)}

def run_single_test():
    """Запускає один повний цикл імітації."""
    logger = logging.getLogger("SIMULATOR")
    logger.info("--- ПОЧАТОК ПОВНОЇ СИМУЛЯЦІЇ ПРОЄКТУ ---")

    if not os.path.exists(TEST_IMAGE_PATH):
        logger.error(f"Тестове зображення не знайдено: {TEST_IMAGE_PATH}")
        return

    logger.info("Ініціалізація компонентів...")
    cam_manager = CameraManager(entry_cam_config=CAMERA_ENTRY_CONFIG)
    sensor_mgr = MockSensorManager()
    gate_ctrl = GateController(open_relay_pin=OPEN_RELAY_PIN, close_relay_pin=CLOSE_RELAY_PIN)
    sheet_hndl = SheetHandler(credentials_file_path=SHEETS_CREDENTIALS_PATH, spreadsheet_url=SPREADSHEET_URL)
    cv_proc = CVProcessor(
        mobilenet_ssd_path=MOBILENET_SSD_PATH, license_model_path=LICENSE_PLATE_MODEL_PATH,
        ocr_model_path=OCR_MODEL_PATH, roi_config_path=ROI_CONFIG_PATH
    )

    # --- ОСЬ КЛЮЧОВЕ ДОДАВАННЯ ---
    logger.info("Ініціалізація TelegramNotifier...")
    notifier = None
    if TelegramNotifier and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        notifier = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, chat_id=int(TELEGRAM_CHAT_ID))
    else:
        logger.warning("Токен/ID чату для Telegram не вказано. Сповіщення вимкнено.")
    # -----------------------------

    vehicle_event_hndl = VehicleEventHandler(
        camera_entry=cam_manager.get_entry_camera(),
        camera_exit=None,
        sensor_manager=sensor_mgr,
        sheet_handler=sheet_hndl,
        cv_processor=cv_proc,
        gate_controller=gate_ctrl,
        config=VEH_CONFIG,
        notifier=notifier  # <-- Передаємо створений нотифікатор
    )
    logger.info("Усі компоненти успішно ініціалізовано.")

    logger.info(f"Обробка зображення: {TEST_IMAGE_PATH}")
    image = cv2.imread(TEST_IMAGE_PATH)

    plate_str, photo_path = cv_proc.get_plate_number_from_image(
        image, 'entry', save_intermediate_steps=True, save_path_prefix=DEBUG_SAVE_DIR
    )

    if not plate_str:
        logger.error("Не вдалося розпізнати номер на зображенні. Симуляцію зупинено.")
    else:
        logger.info(f"Розпізнано номер: {plate_str}. Шлях до фото: {photo_path}. Передача в обробник подій...")
        vehicle_event_hndl.handle_request(
            cam_type='entry',
            plate_text=plate_str,
            photo_path=photo_path
        )

    # Очищення ресурсів
    gate_ctrl.cleanup()
    sensor_mgr.cleanup()
    cam_manager.close_all_cameras()

    logger.info("--- СИМУЛЯЦІЯ ЗАВЕРШЕНА ---")

if __name__ == "__main__":
    setup_global_logging()
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(CONFIG_DIR, 'roi_config.json')):
        with open(os.path.join(CONFIG_DIR, 'roi_config.json'), 'w') as f:
            f.write('{}')

    run_single_test()
