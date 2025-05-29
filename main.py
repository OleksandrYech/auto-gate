# main.py
import logging
import os
import signal
import sys
import threading
import time
from typing import Dict, Any  # Додано для type hinting конфігурацій

# --- Налаштування шляхів для імпорту ---
try:
    from utils.logger_config import setup_global_logging
    from core.camera_manager import CameraManager, DEFAULT_ENTRY_CAM_MODEL_SUBSTRING, DEFAULT_EXIT_CAM_MODEL_SUBSTRING
    from core.sensors_manager import SensorManager
    from core.sheet_handler import SheetHandler
    from core.gate_controller import GateController
    from core.cv_processor import CVProcessor
    from core.vehicle_event_handler import VehicleEventHandler
except ImportError as e:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_guess = os.path.dirname(current_script_dir)
    if project_root_guess not in sys.path:
        sys.path.insert(0, project_root_guess)

    # Повторна спроба імпорту
    try:
        from utils.logger_config import setup_global_logging
        from core.camera_manager import CameraManager, DEFAULT_ENTRY_CAM_MODEL_SUBSTRING, \
            DEFAULT_EXIT_CAM_MODEL_SUBSTRING
        from core.sensors_manager import SensorManager
        from core.sheet_handler import SheetHandler
        from core.gate_controller import GateController
        from core.cv_processor import CVProcessor
        from core.vehicle_event_handler import VehicleEventHandler
    except ImportError as final_e:
        print(f"Критична помилка імпорту модулів: {final_e}")
        print("Переконайтеся, що всі необхідні файли (.py) знаходяться у правильних директоріях (core/, utils/)")
        print(f"Поточний робочий каталог: {os.getcwd()}")
        print(f"Шляхи пошуку Python: {sys.path}")
        sys.exit(1)

# --- Глобальні константи та конфігурація ---

# Шляхи (відносно кореня проекту)
CONFIG_DIR = "config"
MODELS_DIR = "models"
LOGS_DIR = "logs"
CAPTURED_IMAGES_BASE_PATH = "captured_images"

ROI_CONFIG_PATH = os.path.join(CONFIG_DIR, "roi_config.json")
SHEETS_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")

MOBILENET_SSD_PATH = os.path.join(MODELS_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license.onnx")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "ocr.pt")

# Налаштування камер
CAMERA_ENTRY_CONFIG: Dict[str, Any] = {
    "name": "EntryCamera", "resolution": (1920, 1080), "hflip": False, "vflip": True
}
CAMERA_EXIT_CONFIG: Dict[str, Any] = {
    "name": "ExitCamera", "resolution": (1280, 720), "hflip": False, "vflip": True
}
# Підрядки для пошуку моделей камер (можна змінити, якщо ваші камери інші)
CAM_ENTRY_MODEL_SUB = DEFAULT_ENTRY_CAM_MODEL_SUBSTRING
CAM_EXIT_MODEL_SUB = DEFAULT_EXIT_CAM_MODEL_SUBSTRING

# Піни GPIO (BCM нумерація)
REED_SWITCH_PIN: int = 22
ULTRASONIC_ENTRY_TRIGGER_PIN: int = 23
ULTRASONIC_ENTRY_ECHO_PIN: int = 24

OPEN_RELAY_PIN: int = 17
CLOSE_RELAY_PIN: int = 27
RELAY_PULSE_DURATION_S: float = 0.5  # Для GateController

# Параметри для моделей CV
CV_VEHICLE_MODEL_INPUT_SIZE: tuple = (300, 300)
CV_OCR_IMG_SIZE_ULTRALYTICS: int = 320

# Таймери та налаштування для VehicleEventHandler (з ваших уточнених сценаріїв)
VEH_CONFIG: Dict[str, Any] = {
    "sheets_antiduplicate_delay_s": 60,
    "reed_open_timeout_s": 15,  # Таймаут очікування відкриття герконом (сценарій VEH)
    "reed_open_retries": 1,  # Кількість повторів відкриття (сценарій VEH)
    "passage_confirmation_timeout_s": 20,  # Таймаут очікування проїзду через УЗД (після появи в зоні)
    "ultrasonic_passage_threshold": 0.3,  # Поріг УЗД для "в проїзді" / "проїзд вільний"
    "auto_close_timer_duration_s": 4,  # 4-секундний таймер на закриття (сценарій VEH)
    "reed_close_timeout_s": 5,  # Таймаут очікування закриття герконом (сценарій VEH)
    "reed_close_retries": 1,  # Кількість повторів закриття (сценарій VEH)
    "gate_finish_closing_delay_s": 10,  # Фінальна затримка після закриття (сценарій VEH)
    "poll_interval_idle_s": 1.0,
    "poll_interval_gate_closing_s": 0.3
}

# Налаштування для GateController
GATE_CTRL_CONFIG: Dict[str, Any] = {
    "relay_pulse_duration_s": RELAY_PULSE_DURATION_S,
    "auto_close_timeout_s": VEH_CONFIG["auto_close_timer_duration_s"],
    # Внутрішній таймер GC тепер синхронізований з VEH
    "closing_obstruction_threshold_m": VEH_CONFIG["ultrasonic_passage_threshold"],  # Використовуємо той самий поріг
    "reed_confirmation_timeout_s": 5  # Короткий таймаут для внутрішньої перевірки GC (можна 0, якщо VEH все контролює)
}

# Подія для коректного завершення роботи
shutdown_event = threading.Event()


# --- Обробник сигналів ---
def signal_handler(sig, frame):
    logger = logging.getLogger(__name__)
    logger.warning(f"Отримано сигнал {signal.Signals(sig).name}. Завершення роботи...")
    shutdown_event.set()


# --- Головна функція програми ---
def main_application():
    logger = logging.getLogger(__name__)
    logger.info("Запуск автоматизованої системи керування воротами...")

    cam_manager = None
    sensor_mgr = None
    gate_ctrl = None
    vehicle_event_hndl = None

    try:
        # 1. Створення необхідних директорій
        os.makedirs(CAPTURED_IMAGES_BASE_PATH, exist_ok=True)  # Головна папка

        # 2. Ініціалізація компонентів
        logger.info("Ініціалізація CameraManager...")
        cam_manager = CameraManager(
            entry_cam_model_sub=CAM_ENTRY_MODEL_SUB,
            exit_cam_model_sub=CAM_EXIT_MODEL_SUB,
            entry_cam_config=CAMERA_ENTRY_CONFIG,
            exit_cam_config=CAMERA_EXIT_CONFIG,
            image_base_path=CAPTURED_IMAGES_BASE_PATH
        )
        if not cam_manager.get_entry_camera() and not cam_manager.get_exit_camera():
            logger.critical("ЖОДНА з камер не ініціалізована. Система не може працювати. Завершення.")
            return
        # Подальша логіка може адаптуватися, якщо одна з камер відсутня (всередині VehicleEventHandler)

        logger.info("Ініціалізація SensorManager...")
        sensor_mgr = SensorManager(
            reed_pin=REED_SWITCH_PIN,
            ultrasonic_entry_trigger_pin=ULTRASONIC_ENTRY_TRIGGER_PIN,
            ultrasonic_entry_echo_pin=ULTRASONIC_ENTRY_ECHO_PIN,
        )

        logger.info("Ініціалізація SheetHandler...")
        sheet_hndl = SheetHandler(credentials_file=SHEETS_CREDENTIALS_PATH)
        if not sheet_hndl._client:  # Проста перевірка успішності ініціалізації клієнта
            logger.critical("SheetHandler не вдалося ініціалізувати клієнта Google Sheets. Завершення.")
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
            ocr_input_target_size_for_ultralytics=CV_OCR_IMG_SIZE_ULTRALYTICS,
            # Можна також передати інші пороги з CVProcessor, якщо потрібно
            vehicle_confidence_thresh=0.5,  # Приклад
            plate_confidence_thresh=0.5,  # Приклад
            ocr_confidence_thresh=0.3,  # Приклад
        )

        logger.info("Ініціалізація VehicleEventHandler...")
        vehicle_event_hndl = VehicleEventHandler(
            camera_entry=cam_manager.get_entry_camera(),
            camera_exit=cam_manager.get_exit_camera(),
            sensor_manager=sensor_mgr,
            sheet_handler=sheet_hndl,
            cv_processor=cv_proc,
            gate_controller=gate_ctrl,
            config=VEH_CONFIG  # Передаємо конфігурацію для VEH
        )

        # 3. Запуск VehicleEventHandler
        logger.info("Запуск основних потоків обробки VehicleEventHandler...")
        vehicle_event_hndl.start(shutdown_event)

        # 4. Головний цикл очікування
        logger.info("Система запущена. Натисніть Ctrl+C для завершення.")
        while not shutdown_event.is_set():
            if vehicle_event_hndl.entry_thread and not vehicle_event_hndl.entry_thread.is_alive() and vehicle_event_hndl.is_running:
                logger.error("Потік обробки В'ЇЗДУ несподівано завершився! Система може працювати некоректно.")
                # Тут можна додати логіку зупинки всієї системи або спроби перезапуску
                # shutdown_event.set()
            if vehicle_event_hndl.exit_thread and not vehicle_event_hndl.exit_thread.is_alive() and vehicle_event_hndl.is_running:
                logger.error("Потік обробки ВИЇЗДУ несподівано завершився! Система може працювати некоректно.")
                # shutdown_event.set()

            time.sleep(1)

    except Exception as e:
        logger.critical(f"Неперехоплена помилка в main_application: {e}", exc_info=True)
        shutdown_event.set()
    finally:
        logger.info("Початок процедури коректного завершення роботи системи...")

        if vehicle_event_hndl:
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


# --- Точка входу програми ---
if __name__ == "__main__":
    setup_global_logging()
    main_logger = logging.getLogger(__name__)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main_logger.info(f"PID процесу: {os.getpid()}")

    try:
        main_application()
    except SystemExit:  # Дозволяємо SystemExit (наприклад, від sys.exit(1)) пройти
        main_logger.info("Програма завершена через SystemExit.")
    except Exception as e_global:
        main_logger.critical(f"Неперехоплений виняток на глобальному рівні (if __name__ == '__main__'): {e_global}",
                             exc_info=True)
    finally:
        logging.shutdown()