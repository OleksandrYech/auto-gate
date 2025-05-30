# main.py
import logging
import os
import signal
import sys
import threading
import time
from typing import Dict, Any, Optional

# --- Налаштування шляхів для імпорту ---

try:
    from utils.logger_config import setup_global_logging
    from core.camera_manager import CameraManager, DEFAULT_ENTRY_CAM_MODEL_SUBSTRING, DEFAULT_EXIT_CAM_MODEL_SUBSTRING
    from core.sensors_manager import SensorManager
    from core.sheet_handler import SheetHandler
    from core.gate_controller import GateController
    from core.cv_processor import CVProcessor, ULTRALYTICS_AVAILABLE
    from core.vehicle_event_handler import VehicleEventHandler
except ImportError as e:
    print(f"Критична помилка імпорту модулів: {e}")
    print("Переконайтеся, що всі необхідні файли (.py) знаходяться у правильних директоріях (core/, utils/)")
    print(f"Поточний робочий каталог: {os.getcwd()}")
    print(f"Шляхи пошуку Python: {sys.path}")
    sys.exit(1)

# --- Глобальні константи та конфігурація ---
CONFIG_DIR = "config"
MODELS_DIR = "models"
LOGS_DIR = "logs"
CAPTURED_IMAGES_BASE_PATH = "captured_images"

ROI_CONFIG_PATH = os.path.join(CONFIG_DIR, "roi_config.json")
SHEETS_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")
# SPREADSHEET_URL = "YOUR_GOOGLE_SHEET_URL_HERE" # Краще залишити в sheet_handler.py або передати в конструктор

MOBILENET_SSD_PATH = os.path.join(MODELS_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license.onnx")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "ocr.pt")

CAMERA_ENTRY_CONFIG: Dict[str, Any] = {
    "name": "EntryCamera", "resolution": (1920, 1080), "hflip": False, "vflip": True
}
CAMERA_EXIT_CONFIG: Dict[str, Any] = {
    "name": "ExitCamera", "resolution": (1280, 720), "hflip": False, "vflip": True
}
CAM_ENTRY_MODEL_SUB = DEFAULT_ENTRY_CAM_MODEL_SUBSTRING
CAM_EXIT_MODEL_SUB = DEFAULT_EXIT_CAM_MODEL_SUBSTRING

REED_SWITCH_PIN: int = 22
ULTRASONIC_ENTRY_TRIGGER_PIN: int = 23
ULTRASONIC_ENTRY_ECHO_PIN: int = 24
ULTRASONIC_EXIT_TRIGGER_PIN: Optional[int] = None
ULTRASONIC_EXIT_ECHO_PIN: Optional[int] = None

OPEN_RELAY_PIN: int = 17
CLOSE_RELAY_PIN: int = 27

CV_VEHICLE_MODEL_INPUT_SIZE: tuple = (300, 300)
CV_OCR_IMG_SIZE_ULTRALYTICS: int = 320

# Конфігурація для VehicleEventHandler
VEH_CONFIG: Dict[str, Any] = {
    "sheets_antiduplicate_delay_s": 60,
    "reed_open_timeout_s": 15,
    "reed_open_retries": 1,
    "passage_confirmation_timeout_s": 20,
    "ultrasonic_passage_threshold": 0.3,
    "auto_close_timer_duration_s": 4,
    "reed_close_timeout_s": 5,
    "reed_close_retries": 1,
    "gate_finish_closing_delay_s": 10,
    "poll_interval_idle_s": 1.0,
    "poll_interval_gate_closing_s": 0.3
}

# Конфігурація для GateController
GATE_CTRL_CONFIG: Dict[str, Any] = {
    "relay_pulse_duration_s": 0.5,  # Тривалість імпульсу реле
    "auto_close_timeout_s": 30,  # Стандартний таймаут GC, якщо VEH не передає свій
    "closing_obstruction_threshold_m": VEH_CONFIG["ultrasonic_passage_threshold"],
    "reed_confirmation_timeout_s": 5
}

shutdown_event = threading.Event()


def signal_handler(sig, frame):
    logger_sh = logging.getLogger(__name__)  # Отримуємо логгер тут, бо глобальний може ще не бути
    logger_sh.warning(f"Отримано сигнал {signal.Signals(sig).name}. Завершення роботи...")
    shutdown_event.set()


def main_application():
    logger = logging.getLogger(__name__)
    logger.info("Запуск автоматизованої системи керування воротами...")

    cam_manager = None
    sensor_mgr = None
    sheet_hndl = None  # Додано для перевірки в finally
    gate_ctrl = None
    vehicle_event_hndl = None

    try:
        os.makedirs(CAPTURED_IMAGES_BASE_PATH, exist_ok=True)
        # Директорії entry/exit/cv_debug створюються в VehicleEventHandler

        logger.info("Ініціалізація CameraManager...")
        cam_manager = CameraManager(
            entry_cam_model_sub=CAM_ENTRY_MODEL_SUB,
            exit_cam_model_sub=CAM_EXIT_MODEL_SUB,
            entry_cam_config=CAMERA_ENTRY_CONFIG,
            exit_cam_config=CAMERA_EXIT_CONFIG,
            image_base_path=CAPTURED_IMAGES_BASE_PATH
        )
        if not cam_manager.get_entry_camera() and not cam_manager.get_exit_camera():  # Перевірка обох
            logger.critical("ЖОДНА з камер не ініціалізована. Система не може працювати. Завершення.")
            return
        if not cam_manager.get_entry_camera():  # Для в'їзду камера критична
            logger.critical("Камера В'ЇЗДУ не ініціалізована. Робота неможлива. Завершення.")
            return
        if not cam_manager.get_exit_camera():  # Для виїзду може бути менш критично, але логуємо
            logger.warning("Камера ВИЇЗДУ не ініціалізована. Функціонал виїзду з розпізнаванням НЗ буде обмежений.")

        logger.info("Ініціалізація SensorManager...")
        sensor_mgr = SensorManager(
            reed_pin=REED_SWITCH_PIN,
            ultrasonic_entry_trigger_pin=ULTRASONIC_ENTRY_TRIGGER_PIN,
            ultrasonic_entry_echo_pin=ULTRASONIC_ENTRY_ECHO_PIN,
            ultrasonic_exit_trigger_pin=ULTRASONIC_EXIT_TRIGGER_PIN,
            ultrasonic_exit_echo_pin=ULTRASONIC_EXIT_ECHO_PIN
        )
        if not sensor_mgr.reed_switch or not sensor_mgr.ultrasonic_sensor_entry:
            logger.critical("Не вдалося ініціалізувати основні датчики (геркон або УЗД в'їзду). Завершення.")
            return

        logger.info("Ініціалізація SheetHandler...")
        sheet_hndl = SheetHandler(credentials_file_path=SHEETS_CREDENTIALS_PATH)
        if not hasattr(sheet_hndl, '_client') or sheet_hndl._client is None:
            logger.critical(
                "SheetHandler не вдалося ініціалізувати клієнта Google Sheets. Перевірте credentials.json та URL. Завершення.")
            return

        logger.info("Ініціалізація GateController...")
        gate_ctrl = GateController(
            sensor_manager_instance=sensor_mgr,
            open_relay_pin=OPEN_RELAY_PIN,
            close_relay_pin=CLOSE_RELAY_PIN,
            relay_pulse_duration_s=GATE_CTRL_CONFIG["relay_pulse_duration_s"],
            auto_close_timeout_s=GATE_CTRL_CONFIG["auto_close_timeout_s"],
            closing_obstruction_threshold_m=GATE_CTRL_CONFIG["closing_obstruction_threshold_m"],
            reed_confirmation_timeout_s=GATE_CTRL_CONFIG["reed_confirmation_timeout_s"]
        )
        if not gate_ctrl.relays_initialized:
            logger.critical("Реле в GateController не ініціалізовано. Завершення.")
            return

        logger.info("Ініціалізація CVProcessor...")
        for model_p in [MOBILENET_SSD_PATH, LICENSE_PLATE_MODEL_PATH, OCR_MODEL_PATH]:
            if not os.path.exists(model_p):
                logger.critical(f"Файл моделі не знайдено: {model_p}. Завершення роботи.")
                return
        if not os.path.exists(ROI_CONFIG_PATH):
            logger.warning(
                f"Файл конфігурації ROI не знайдено: {ROI_CONFIG_PATH}. Буде використано обробку повного кадру.")

        cv_proc = CVProcessor(
            mobilenet_ssd_path=MOBILENET_SSD_PATH,
            license_model_path=LICENSE_PLATE_MODEL_PATH,
            ocr_model_path=OCR_MODEL_PATH,
            roi_config_path=ROI_CONFIG_PATH,
            vehicle_input_target_size=CV_VEHICLE_MODEL_INPUT_SIZE,
            ocr_input_target_size_for_ultralytics=CV_OCR_IMG_SIZE_ULTRALYTICS
        )
        # Перевірка, чи завантажились моделі в CVProcessor
        if not cv_proc.vehicle_session or not cv_proc.plate_session or \
                (ULTRALYTICS_AVAILABLE and not cv_proc.ocr_model_ultralytics and not OCR_MODEL_PATH.endswith(
                    ".onnx")) or \
                (
                        not ULTRALYTICS_AVAILABLE and not cv_proc.ocr_session):  # Якщо ультралітікс недоступний, має бути ocr_session
            logger.critical("Одна або декілька моделей CV не завантажилися. Завершення.")
            return

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

        logger.info("Запуск основних потоків обробки VehicleEventHandler...")
        vehicle_event_hndl.start(shutdown_event)

        logger.info("Система запущена. Натисніть Ctrl+C для завершення.")
        while not shutdown_event.is_set():
            if vehicle_event_hndl.entry_thread and not vehicle_event_hndl.entry_thread.is_alive() and vehicle_event_hndl.is_running:
                logger.error("Потік обробки В'ЇЗДУ несподівано завершився! Система може працювати некоректно.")
            if vehicle_event_hndl.exit_thread and not vehicle_event_hndl.exit_thread.is_alive() and vehicle_event_hndl.is_running:
                logger.error("Потік обробки ВИЇЗДУ несподівано завершився! Система може працювати некоректно.")
            time.sleep(1)

    except Exception as e:
        logger.critical(f"Неперехоплена помилка в main_application: {e}", exc_info=True)
        shutdown_event.set()
    finally:
        logger.info("Початок процедури коректного завершення роботи системи...")

        if vehicle_event_hndl:  # Перевіряємо, чи було створено об'єкт
            logger.info("Зупинка VehicleEventHandler...")
            vehicle_event_hndl.stop()
            if hasattr(vehicle_event_hndl, 'entry_thread') and vehicle_event_hndl.entry_thread.is_alive():
                logger.info("Очікування завершення потоку В'ЇЗДУ...")
                vehicle_event_hndl.entry_thread.join(timeout=5)
            if hasattr(vehicle_event_hndl, 'exit_thread') and vehicle_event_hndl.exit_thread.is_alive():
                logger.info("Очікування завершення потоку ВИЇЗДУ...")
                vehicle_event_hndl.exit_thread.join(timeout=5)
            logger.info("Потоки VehicleEventHandler завершено.")

        if cam_manager:
            logger.info("Закриття камер...")
            cam_manager.close_all_cameras()
        if gate_ctrl:
            logger.info("Очищення GateController...")
            gate_ctrl.cleanup()
        if sensor_mgr:
            logger.info("Очищення SensorManager...")
            sensor_mgr.cleanup()

        logger.info("Система керування воротами завершила роботу.")


if __name__ == "__main__":
    setup_global_logging()
    main_logger = logging.getLogger(__name__)  # Тепер отримуємо логгер після налаштування

    # 2. Налаштування обробників сигналів
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main_logger.info(f"PID процесу: {os.getpid()}")

    try:
        main_application()
    except SystemExit:
        main_logger.info("Програма завершена через SystemExit.")
    except Exception as e_global:
        main_logger.critical(f"Неперехоплений виняток на глобальному рівні (if __name__ == '__main__'): {e_global}",
                             exc_info=True)
    finally:
        logging.shutdown()
