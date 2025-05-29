# main.py
import logging
import os
import signal
import sys
import threading
import time

# --- Налаштування шляхів для імпорту ---
current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from utils.logger_config import setup_global_logging
    from core.camera_manager import CameraManager, DEFAULT_ENTRY_CAM_MODEL_SUBSTRING, DEFAULT_EXIT_CAM_MODEL_SUBSTRING
    from core.sensors_manager import SensorManager
    from core.sheet_handler import SheetHandler
    from core.gate_controller import GateController
    from core.cv_processor import CVProcessor
    from core.vehicle_event_handler import VehicleEventHandler
except ImportError as e:
    print(f"Критична помилка імпорту модулів: {e}")
    print("Переконайтеся, що всі необхідні файли (.py) знаходяться у правильних директоріях (core/, utils/)")
    print(f"Поточний робочий каталог: {os.getcwd()}")
    print(f"Шляхи пошуку Python: {sys.path}")
    sys.exit(1)

# --- Глобальні константи та конфігурація ---

# Шляхи (відносно кореня проекту, де знаходиться main.py)
CONFIG_DIR = "config"
MODELS_DIR = "models"
LOGS_DIR = "logs"  # Використовується logger_config
CAPTURED_IMAGES_BASE_PATH = "captured_images"  # Для VehicleEventHandler та CVProcessor

ROI_CONFIG_PATH = os.path.join(CONFIG_DIR, "roi_config.json")
MOBILENET_SSD_PATH = os.path.join(MODELS_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license.onnx")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "ocr.pt")  # Використовуємо .pt для OCR

# Налаштування камер (приклади, можна винести в конфігураційний файл)
CAMERA_ENTRY_CONFIG = {
    "name": "EntryCamera", "resolution": (1920, 1080), "hflip": False, "vflip": True
}
CAMERA_EXIT_CONFIG = {
    "name": "ExitCamera", "resolution": (1280, 720), "hflip": False, "vflip": True
}

# Піни GPIO (BCM нумерація)
REED_SWITCH_PIN = 22
ULTRASONIC_ENTRY_TRIGGER_PIN = 23
ULTRASONIC_ENTRY_ECHO_PIN = 24
# ULTRASONIC_EXIT_TRIGGER_PIN = ... # Якщо є окремий датчик
# ULTRASONIC_EXIT_ECHO_PIN = ...  # Якщо є окремий

OPEN_RELAY_PIN = 17
CLOSE_RELAY_PIN = 27

# Параметри для моделей CV
CV_VEHICLE_MODEL_INPUT_SIZE = (300, 300)  # (width, height) для MobileNet SSD
CV_OCR_IMG_SIZE_ULTRALYTICS = 320  # Розмір для YOLO().predict() для OCR

# Таймери та налаштування для VehicleEventHandler
VEH_SHEETS_ANTIDUPLICATE_DELAY_S = 60
VEH_PASSAGE_CONFIRMATION_TIMEOUT_S = 20
VEH_POLL_INTERVAL_IDLE_S = 1.0  # Зменшено для швидшої реакції
VEH_POLL_INTERVAL_GATE_CLOSING_S = 0.3

# Таймер для GateController
GATE_AUTO_CLOSE_TIMEOUT_S = 30

# Подія для коректного завершення роботи
shutdown_event = threading.Event()


# --- Налаштування логування ---
# setup_global_logging() викликається в блоці if __name__ == "__main__"

# --- Обробник сигналів ---
def signal_handler(sig, frame):
    logger = logging.getLogger(__name__)  # Отримуємо логгер тут
    logger.warning(f"Отримано сигнал {signal.Signals(sig).name}. Завершення роботи...")
    shutdown_event.set()


# --- Головна функція програми ---
def main_application():
    logger = logging.getLogger(__name__)  # Логгер для цієї функції
    logger.info("Запуск автоматизованої системи керування воротами...")

    cam_manager = None
    sensor_mgr = None
    gate_ctrl = None
    vehicle_event_hndl = None  # Визначаємо тут, щоб був доступний у finally

    try:
        # 1. Створення необхідних директорій
        os.makedirs(CAPTURED_IMAGES_BASE_PATH, exist_ok=True)
        os.makedirs(os.path.join(CAPTURED_IMAGES_BASE_PATH, "entry"), exist_ok=True)
        os.makedirs(os.path.join(CAPTURED_IMAGES_BASE_PATH, "exit"), exist_ok=True)
        os.makedirs(os.path.join(CAPTURED_IMAGES_BASE_PATH, "cv_debug", "entry"), exist_ok=True)
        os.makedirs(os.path.join(CAPTURED_IMAGES_BASE_PATH, "cv_debug", "exit"), exist_ok=True)
        # Директорія для логів створюється в logger_config.py

        # 2. Ініціалізація компонентів
        logger.info("Ініціалізація CameraManager...")
        cam_manager = CameraManager(
            entry_cam_model_sub=DEFAULT_ENTRY_CAM_MODEL_SUBSTRING,
            exit_cam_model_sub=DEFAULT_EXIT_CAM_MODEL_SUBSTRING,
            entry_cam_config=CAMERA_ENTRY_CONFIG,
            exit_cam_config=CAMERA_EXIT_CONFIG,
            image_base_path=CAPTURED_IMAGES_BASE_PATH
        )
        # Перевірка наявності камер (принаймні в'їзної)
        if not cam_manager.get_entry_camera() and not cam_manager.get_exit_camera():
            logger.critical("ЖОДНА з камер не ініціалізована. Система не може працювати. Завершення.")
            return
        if not cam_manager.get_entry_camera():
            logger.warning("Камера В'ЇЗДУ не ініціалізована. Функціонал в'їзду буде обмежений.")
            # Можна або завершити роботу, або продовжити з обмеженим функціоналом
        if not cam_manager.get_exit_camera():
            logger.warning("Камера ВИЇЗДУ не ініціалізована. Функціонал виїзду буде обмежений.")

        logger.info("Ініціалізація SensorManager...")
        sensor_mgr = SensorManager(
            reed_pin=REED_SWITCH_PIN,
            ultrasonic_entry_trigger_pin=ULTRASONIC_ENTRY_TRIGGER_PIN,
            ultrasonic_entry_echo_pin=ULTRASONIC_ENTRY_ECHO_PIN,
            # Додайте сюди піни для УЗД виїзду, якщо вони є:
            # ultrasonic_exit_trigger_pin=ULTRASONIC_EXIT_TRIGGER_PIN,
            # ultrasonic_exit_echo_pin=ULTRASONIC_EXIT_ECHO_PIN
        )

        logger.info("Ініціалізація SheetHandler...")
        sheet_hndl = SheetHandler()  # Використовує константи з модуля sheet_handler.py

        logger.info("Ініціалізація GateController...")
        gate_ctrl = GateController(
            sensor_manager_instance=sensor_mgr,
            open_relay_pin=OPEN_RELAY_PIN,
            close_relay_pin=CLOSE_RELAY_PIN,
            auto_close_timeout_s=GATE_AUTO_CLOSE_TIMEOUT_S
        )

        logger.info("Ініціалізація CVProcessor...")
        # Перевірка існування файлів моделей
        for model_p in [MOBILENET_SSD_PATH, LICENSE_PLATE_MODEL_PATH, OCR_MODEL_PATH]:
            if not os.path.exists(model_p):
                logger.critical(f"Файл моделі не знайдено: {model_p}. Завершення роботи.")
                return
        # Перевірка файлу ROI (він може бути створений roi_create.py, тому лише попередження)
        if not os.path.exists(ROI_CONFIG_PATH):
            logger.warning(f"Файл конфігурації ROI не знайдено: {ROI_CONFIG_PATH}. "
                           f"Буде використано обробку повного кадру, якщо ROI не буде створено.")

        cv_proc = CVProcessor(
            mobilenet_ssd_path=MOBILENET_SSD_PATH,
            license_model_path=LICENSE_PLATE_MODEL_PATH,
            ocr_model_path=OCR_MODEL_PATH,  # Шлях до ocr.pt
            roi_config_path=ROI_CONFIG_PATH,
            vehicle_input_target_size=CV_VEHICLE_MODEL_INPUT_SIZE,
            ocr_input_target_size_for_ultralytics=CV_OCR_IMG_SIZE_ULTRALYTICS
        )

        logger.info("Ініціалізація VehicleEventHandler...")
        event_handler_config = {
            "sheets_antiduplicate_delay_s": VEH_SHEETS_ANTIDUPLICATE_DELAY_S,
            "passage_confirmation_timeout_s": VEH_PASSAGE_CONFIRMATION_TIMEOUT_S,
            "poll_interval_idle_s": VEH_POLL_INTERVAL_IDLE_S,
            "poll_interval_gate_closing_s": VEH_POLL_INTERVAL_GATE_CLOSING_S,
        }
        vehicle_event_hndl = VehicleEventHandler(
            camera_entry=cam_manager.get_entry_camera(),  # Може бути None
            camera_exit=cam_manager.get_exit_camera(),  # Може бути None
            sensor_manager=sensor_mgr,
            sheet_handler=sheet_hndl,
            cv_processor=cv_proc,
            gate_controller=gate_ctrl,
            config=event_handler_config
        )

        # 3. Запуск VehicleEventHandler
        logger.info("Запуск основних потоків обробки VehicleEventHandler...")
        vehicle_event_hndl.start(shutdown_event)

        # 4. Головний цикл очікування
        logger.info("Система запущена. Натисніть Ctrl+C для завершення.")
        while not shutdown_event.is_set():
            # Перевірка життєздатності потоків обробника (опціонально)
            if vehicle_event_hndl.entry_thread and not vehicle_event_hndl.entry_thread.is_alive() \
                    and vehicle_event_hndl.is_running:  # Перевіряємо is_running, щоб не логувати після штатної зупинки
                logger.error("Потік обробки В'ЇЗДУ несподівано завершився!")
                # Тут можна додати логіку перезапуску або аварійного завершення
                # shutdown_event.set() # Наприклад, зупинити все
            if vehicle_event_hndl.exit_thread and not vehicle_event_hndl.exit_thread.is_alive() \
                    and vehicle_event_hndl.is_running:
                logger.error("Потік обробки ВИЇЗДУ несподівано завершився!")
                # shutdown_event.set()

            time.sleep(1)  # Головний потік може просто чекати

    except Exception as e:
        logger.critical(f"Неперехоплена помилка в main_application: {e}", exc_info=True)
    finally:
        logger.info("Початок процедури коректного завершення роботи системи...")

        # Сигналізуємо VehicleEventHandler про зупинку (якщо він був створений)
        if vehicle_event_hndl:
            logger.info("Зупинка VehicleEventHandler...")
            vehicle_event_hndl.stop()  # Встановлює is_running = False

            # Очікуємо завершення його потоків
            if hasattr(vehicle_event_hndl, 'entry_thread') and vehicle_event_hndl.entry_thread.is_alive():
                logger.info("Очікування завершення потоку В'ЇЗДУ...")
                vehicle_event_hndl.entry_thread.join(timeout=5)
            if hasattr(vehicle_event_hndl, 'exit_thread') and vehicle_event_hndl.exit_thread.is_alive():
                logger.info("Очікування завершення потоку ВИЇЗДУ...")
                vehicle_event_hndl.exit_thread.join(timeout=5)
            logger.info("Потоки VehicleEventHandler завершено.")

        # Очищення ресурсів інших компонентів
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
    # 1. Налаштування логування - має бути першим!
    # Рівні можна передати як аргументи, якщо потрібно змінити стандартні з logger_config.py
    setup_global_logging()

    # Отримуємо головний логгер для main.py (після налаштування)
    main_logger = logging.getLogger(__name__)

    # 2. Налаштування обробників сигналів
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main_logger.info(f"PID процесу: {os.getpid()}")

    try:
        main_application()
    except Exception as e_global:
        main_logger.critical(f"Неперехоплений виняток на глобальному рівні: {e_global}", exc_info=True)
    finally:
        logging.shutdown()  # Закриваємо всі файлові дескриптори логування
