# main.py
import time
import logging
import threading
import signal
import os

# --- Імпорт ваших модулів ---
# Припускаємо, що вони знаходяться в директорії core/ або ви налаштували PYTHONPATH
from core.camera_manager import CameraController, get_camera_ids, DEFAULT_ENTRY_CAM_MODEL_SUBSTRING, \
    DEFAULT_EXIT_CAM_MODEL_SUBSTRING
from core.sensors_manager import SensorManager, ReedSwitch, UltrasonicSensor  # Класи для type hinting або конфігурації
from core.sheet_handler import SheetHandler  # Перейменований sheets.py
from core.gate_controller import GateController

# from core.cv_processor import CVProcessor # Буде створено пізніше
# from core.vehicle_event_handler import VehicleEventHandler # Буде створено пізніше

# --- Глобальні константи та конфігурація (приклади) ---
# Шляхи до моделей ONNX (будуть використовуватися CVProcessor)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MOBILENET_SSD_PATH = os.path.join(MODEL_DIR, "ssd_mobilenetv1.onnx")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODEL_DIR, "license.onnx")
OCR_MODEL_PATH = os.path.join(MODEL_DIR, "ocr.onnx")

# Конфігурація ROI (шлях до файлу, який читатиме CVProcessor)
ROI_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "roi_config.json")  #

# Налаштування логування
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "logs", "gate_system.log")
LOG_LEVEL = logging.DEBUG  # Або logging.INFO для менш детальних логів

# Піни для датчиків та реле (якщо не використовуються значення за замовчуванням з модулів)
# Ці значення можуть бути передані конструкторам відповідних класів.
# Наприклад, для SensorManager:
REED_SWITCH_PIN = 22  # З вашого sensors.py (приклад)
ULTRASONIC_ENTRY_TRIGGER_PIN = 23
ULTRASONIC_ENTRY_ECHO_PIN = 24
# ULTRASONIC_EXIT_TRIGGER_PIN = ... (якщо є окремий датчик для виїзду)
# ULTRASONIC_EXIT_ECHO_PIN = ...

# Для GateController:
OPEN_RELAY_PIN = 17  # З gate_controller.py (приклад)
CLOSE_RELAY_PIN = 27

# Таймери для VehicleEventHandler (приклади)
SHEETS_ANTIDUPLICATE_DELAY_S = 60  # 60 секунд для уникнення дублів у Sheets
GATE_AUTO_CLOSE_TIMEOUT_S = 30  # Використовується GateController, але може бути передано

# Прапорець для коректного завершення роботи
shutdown_event = threading.Event()


# --- Заглушки для ще не створених модулів/класів ---

class CVProcessor:
    """Заглушка для обробника комп'ютерного зору."""

    def __init__(self, mobilenet_path, license_model_path, ocr_model_path, roi_config_path):
        self.logger = logging.getLogger(f"{__name__}.CVProcessor")
        self.logger.info(
            f"Ініціалізація CVProcessor з моделями: \nSSD: {mobilenet_path}\nLP: {license_model_path}\nOCR: {ocr_model_path}")
        self.logger.info(f"Конфігурація ROI: {roi_config_path}")
        # Тут буде завантаження моделей ONNX та конфігурації ROI
        time.sleep(0.1)  # Імітація завантаження
        self.logger.info("CVProcessor (заглушка) ініціалізовано.")

    def detect_vehicle_in_roi(self, image_array, camera_type="entry"):
        """Імітує детекцію автомобіля."""
        self.logger.debug(f"CV: Детекція автомобіля на камері {camera_type} (заглушка)...")
        # Повертає True, якщо авто знайдено в ROI, інакше False
        return True  # Імітація, що авто завжди є для тестування циклів

    def get_license_plate_info(self, image_array):
        """Імітує розпізнавання номерного знаку."""
        self.logger.debug("CV: Розпізнавання номерного знаку (заглушка)...")
        # Повертає розпізнаний номер або None
        recognized_plate = f"AA{time.time() % 1000:.0f}AA"  # Імітація
        self.logger.info(f"CV: Розпізнано номер (заглушка): {recognized_plate}")
        return recognized_plate


class VehicleEventHandler:
    """
    Заглушка для обробника подій автомобілів.
    Координує взаємодію між камерами, CV, датчиками, Google Sheets та воротами.
    """

    def __init__(self,
                 camera_entry,  # CameraController instance
                 camera_exit,  # CameraController instance
                 sensor_manager,
                 sheet_handler,
                 cv_processor,
                 gate_controller,
                 config):  # Словник з таймерами та іншими налаштуваннями
        self.logger = logging.getLogger(f"{__name__}.VehicleEventHandler")
        self.camera_entry = camera_entry
        self.camera_exit = camera_exit
        self.sensor_manager = sensor_manager
        self.sheet_handler = sheet_handler
        self.cv_processor = cv_processor
        self.gate_controller = gate_controller
        self.config = config
        self.is_running = True
        self.recently_logged_plates = {}  # Для анти-дублювання: {'plate': timestamp}

        self.logger.info("VehicleEventHandler (заглушка) ініціалізовано.")

    def _is_duplicate_log(self, plate_number):
        """Перевіряє, чи не був цей номер залогований нещодавно."""
        now = time.time()
        if plate_number in self.recently_logged_plates:
            last_log_time = self.recently_logged_plates[plate_number]
            if (now - last_log_time) < self.config.get("sheets_antiduplicate_delay_s", 60):
                self.logger.info(f"Номер '{plate_number}' вже був залогований менше хвилини тому. Пропуск.")
                return True
        self.recently_logged_plates[plate_number] = now
        # Очищення старого кешу (опціонально)
        # self.recently_logged_plates = {p: t for p, t in self.recently_logged_plates.items() if (now - t) < max_cache_time}
        return False

    def _handle_vehicle_passage_and_autoclose(self, gate_side_name="в'їзду"):
        """Очікує проїзду авто та запускає таймер автозакриття."""
        self.logger.info(f"Очікування повного проїзду автомобіля через ворота з боку {gate_side_name}...")
        # Використовуємо ультразвуковий датчик для фіксації проїзду
        # Це спрощена логіка; sensors_manager може мати більш спеціалізований метод
        # Наприклад, чекаємо, поки зона стане вільною після того, як була зайнята
        # Припустимо, що ультразвуковий датчик в'їзду використовується для цього
        if self.sensor_manager.ultrasonic_sensor_entry.wait_for_clear_after_pass(timeout=20):  # З sensors.py
            self.logger.info(f"Автомобіль проїхав зону {gate_side_name}. Запуск таймера авто-закриття.")
            self.gate_controller.start_auto_close_timer()
        else:
            self.logger.warning(
                f"Тайм-аут очікування проїзду автомобіля через зону {gate_side_name}. Таймер авто-закриття не запущено.")

    def entry_scenario_loop(self):
        """Основний цикл для обробки сценарію В'ЇЗДУ."""
        self.logger.info("Запуск циклу обробки В'ЇЗДУ...")
        while self.is_running and not shutdown_event.is_set():
            self.logger.debug("В'їзд: очікування автомобіля (заглушка)...")
            # 1. Детекція автомобіля (УЗД або CV)
            # Припустимо, CVProcessor або SensorManager сигналізує про наближення
            # Для заглушки, просто робимо перевірку раз на кілька секунд
            time.sleep(5)  # Імітація очікування

            # Імітація детекції УЗД датчиком наближення (з вашого sensors.py)
            if self.sensor_manager.ultrasonic_sensor_entry.wait_for_approach(timeout=0.1):  # Не блокувати надовго
                self.logger.info("В'їзд: УЗД зафіксував наближення.")
                # Додаткова перевірка CV, якщо потрібно
                # if not self.cv_processor.detect_vehicle_in_roi(None, "entry"): continue

                self.logger.info("В'їзд: Автомобіль виявлено. Робимо фото...")
                image_path = self.camera_entry.capture_image(
                    f"captured_images/entry/entry_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                if not image_path:
                    self.logger.error("В'їзд: Не вдалося зробити фото.")
                    continue

                # Тут можна завантажити image_array, якщо capture_image не повертає його
                # image_array = self.camera_entry.capture_array()
                plate_number = self.cv_processor.get_license_plate_info(None)  # Передаємо None замість image_array

                if plate_number:
                    self.logger.info(f"В'їзд: Розпізнано номер: {plate_number}")
                    if self._is_duplicate_log(plate_number):
                        continue

                    if self.sheet_handler.find_vehicle_and_update_entry_time(plate_number):
                        self.logger.info(f"В'їзд: Номер '{plate_number}' АВТОРИЗОВАНО. Відкриття воріт.")
                        self.gate_controller.open_gate()
                        self._handle_vehicle_passage_and_autoclose("в'їзду")
                    else:
                        self.logger.info(f"В'їзд: Номер '{plate_number}' НЕ АВТОРИЗОВАНО. Логування спроби.")
                        self.sheet_handler.add_unauthorized_attempt(plate_number)
                else:
                    self.logger.warning("В'їзд: Номерний знак не розпізнано.")

            # Перевірка, чи потрібно перервати закриття воріт (якщо таймер активний)
            if self.gate_controller._auto_close_timer and self.gate_controller._auto_close_timer.is_alive():
                # Імітуємо, що CVProcessor постійно моніторить
                # У реальній системі це може бути складніше (окремий потік для CV моніторингу)
                # if self.cv_processor.detect_vehicle_in_roi(None, "entry_monitoring"): # Окремий ROI або камера
                # Для заглушки, припустимо, що якщо УЗД знову спрацював, це нове авто
                if self.sensor_manager.ultrasonic_sensor_entry.is_vehicle_approaching():
                    self.logger.info("В'їзд: ВИЯВЛЕНО АВТО під час таймера закриття! Переривання закриття.")
                    self.gate_controller.interrupt_closing_procedure()
                    # Потрібно негайно перейти до обробки цього нового авто (почати цикл знову)
                    continue  # Продовжити цикл для обробки нового авто

        self.logger.info("Цикл обробки В'ЇЗДУ завершено.")

    def exit_scenario_loop(self):
        """Основний цикл для обробки сценарію ВИЇЗДУ."""
        self.logger.info("Запуск циклу обробки ВИЇЗДУ...")
        # Припустимо, що для виїзду є свій ультразвуковий датчик ultrasonic_sensor_exit
        # Якщо ні, потрібно адаптувати логіку або використовувати CV для детекції на виїзд
        # Для заглушки, припустимо, що він є:
        # ultrasonic_exit = self.sensor_manager.ultrasonic_sensor_exit
        ultrasonic_exit = self.sensor_manager.ultrasonic_sensor_entry  # Тимчасово використовуємо той самий

        while self.is_running and not shutdown_event.is_set():
            self.logger.debug("Виїзд: очікування автомобіля (заглушка)...")
            time.sleep(6)  # Імітація очікування, трохи рідше ніж на в'їзд

            if ultrasonic_exit.wait_for_approach(timeout=0.1):
                self.logger.info("Виїзд: УЗД зафіксував наближення. Негайне відкриття воріт.")
                self.gate_controller.open_gate()

                self.logger.info("Виїзд: Робимо фото...")
                image_path = self.camera_exit.capture_image(
                    f"captured_images/exit/exit_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                if not image_path:
                    self.logger.error("Виїзд: Не вдалося зробити фото.")
                    # Ворота вже відкриті, потрібно все одно обробити проїзд
                    self._handle_vehicle_passage_and_autoclose("виїзду")
                    continue

                plate_number = self.cv_processor.get_license_plate_info(None)
                if plate_number:
                    self.logger.info(f"Виїзд: Розпізнано номер: {plate_number}")
                    if not self._is_duplicate_log(plate_number):
                        self.sheet_handler.log_vehicle_exit(plate_number)
                else:
                    self.logger.warning("Виїзд: Номерний знак не розпізнано. Виїзд все одно дозволено.")

                self._handle_vehicle_passage_and_autoclose("виїзду")

            # Перевірка, чи потрібно перервати закриття воріт (якщо таймер активний)
            if self.gate_controller._auto_close_timer and self.gate_controller._auto_close_timer.is_alive():
                if ultrasonic_exit.is_vehicle_approaching():  # Або CV детекція на виїзді
                    self.logger.info("Виїзд: ВИЯВЛЕНО АВТО під час таймера закриття! Переривання закриття.")
                    self.gate_controller.interrupt_closing_procedure()
                    continue  # Продовжити цикл для обробки нового авто

        self.logger.info("Цикл обробки ВИЇЗДУ завершено.")

    def stop(self):
        self.logger.info("Зупинка VehicleEventHandler...")
        self.is_running = False


# --- Функції ---

def setup_logging():
    """Налаштовує систему логування."""
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("picamera2").setLevel(logging.WARNING)  # Зменшуємо детальність логів від picamera2
    logging.getLogger("gspread").setLevel(logging.INFO)
    logging.getLogger("oauth2client").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info("Систему логування налаштовано.")
    return logger


def signal_handler(sig, frame):
    """Обробник сигналів для коректного завершення."""
    logger = logging.getLogger(__name__)
    logger.warning(f"Отримано сигнал {signal.Signals(sig).name}. Завершення роботи...")
    shutdown_event.set()


# --- Головна функція програми ---
def main_application():
    logger = logging.getLogger(__name__)
    logger.info("Запуск автоматизованої системи керування воротами...")

    # 1. Ініціалізація компонентів
    logger.info("Ініціалізація компонентів системи...")
    try:
        # --- Датчики ---
        # Потрібно передати конкретні піни, якщо вони відрізняються від значень за замовчуванням в SensorManager
        # або якщо SensorManager не має конструктора за замовчуванням.
        # Припустимо, SensorManager ініціалізує датчики всередині.
        sensor_mgr = SensorManager(
            reed_pin=REED_SWITCH_PIN,
            # Припускаємо, що SensorManager приймає піни для УЗД в'їзду
            # та потенційно для УЗД виїзду, якщо вони є.
            # Назви параметрів залежать від реалізації SensorManager
            ultrasonic_entry_trigger_pin=ULTRASONIC_ENTRY_TRIGGER_PIN,
            ultrasonic_entry_echo_pin=ULTRASONIC_ENTRY_ECHO_PIN
            # ultrasonic_exit_trigger_pin=ULTRASONIC_EXIT_TRIGGER_PIN, # Якщо є
            # ultrasonic_exit_echo_pin=ULTRASONIC_EXIT_ECHO_PIN      # Якщо є
        )
        logger.info("SensorManager ініціалізовано.")

        # --- Google Sheets ---
        sheet_hndl = SheetHandler()  # Використовує CREDENTIALS_FILE та YOUR_SPREADSHEET_URL з модуля
        logger.info("SheetHandler ініціалізовано.")

        # --- Керування воротами ---
        gate_ctrl = GateController(
            sensor_manager_instance=sensor_mgr,
            open_relay_pin=OPEN_RELAY_PIN,  # Можна взяти з констант main.py
            close_relay_pin=CLOSE_RELAY_PIN,
            auto_close_timeout_s=GATE_AUTO_CLOSE_TIMEOUT_S
        )
        logger.info("GateController ініціалізовано.")

        # --- Камери ---
        # Спочатку отримуємо ID камер
        # Можна передати специфічні підрядки моделей, якщо DEFAULT_ не підходять
        camera_ids = get_camera_ids()
        if camera_ids["entry"] is None or camera_ids["exit"] is None:
            logger.critical("Не вдалося знайти одну або обидві камери. Перевірте підключення та модельні імена.")
            # Тут можна або завершити роботу, або спробувати працювати з однією камерою,
            # або перейти в режим без камер, якщо це передбачено.
            # Для прикладу, завершуємо роботу:
            if camera_ids["entry"] is None: logger.error("Камера В'ЇЗДУ не знайдена.")
            if camera_ids["exit"] is None: logger.error("Камера ВИЇЗДУ не знайдена.")
            return  # Завершити main_application

        cam_entry = CameraController(camera_id=camera_ids["entry"], camera_name="EntryCam")
        cam_exit = CameraController(camera_id=camera_ids["exit"], camera_name="ExitCam",
                                    capture_resolution=(1280, 720))  # Приклад іншої роздільної здатності
        logger.info(f"Камера В'ЇЗДУ (ID: {camera_ids['entry']}) та ВИЇЗДУ (ID: {camera_ids['exit']}) ініціалізовано.")

        # --- Обробка зображень (CV) ---
        cv_proc = CVProcessor(
            mobilenet_path=MOBILENET_SSD_PATH,
            license_model_path=LICENSE_PLATE_MODEL_PATH,
            ocr_model_path=OCR_MODEL_PATH,
            roi_config_path=ROI_CONFIG_PATH
        )
        logger.info("CVProcessor ініціалізовано.")

        # --- Обробник подій автомобілів ---
        event_handler_config = {
            "sheets_antiduplicate_delay_s": SHEETS_ANTIDUPLICATE_DELAY_S
            # ... інші конфігурації для event_handler ...
        }
        vehicle_event_hndl = VehicleEventHandler(
            camera_entry=cam_entry,
            camera_exit=cam_exit,
            sensor_manager=sensor_mgr,
            sheet_handler=sheet_hndl,
            cv_processor=cv_proc,
            gate_controller=gate_ctrl,
            config=event_handler_config
        )
        logger.info("VehicleEventHandler ініціалізовано.")

    except Exception as e:
        logger.critical(f"Критична помилка під час ініціалізації системи: {e}", exc_info=True)
        return  # Не запускати потоки, якщо ініціалізація не вдалася

    # 2. Запуск потоків для сценаріїв в'їзду та виїзду
    logger.info("Запуск основних потоків обробки...")
    entry_thread = threading.Thread(target=vehicle_event_hndl.entry_scenario_loop, name="EntryThread")
    exit_thread = threading.Thread(target=vehicle_event_hndl.exit_scenario_loop, name="ExitThread")

    entry_thread.daemon = True  # Дозволити програмі завершитися, навіть якщо потоки ще працюють (хоча ми будемо чекати)
    exit_thread.daemon = True

    entry_thread.start()
    exit_thread.start()

    # 3. Головний цикл очікування або інші фонові задачі
    try:
        while not shutdown_event.is_set():
            # Тут можна виконувати періодичні перевірки стану, логування статистики тощо.
            # Наприклад, перевіряти стан воріт і логувати його.
            current_gate_status = gate_ctrl.get_current_gate_state()
            logger.debug(
                f"Поточний стан воріт: {current_gate_status}. Потоки активні: В'їзд - {entry_thread.is_alive()}, Виїзд - {exit_thread.is_alive()}")

            # Якщо один з потоків несподівано завершився, можна спробувати його перезапустити або залогувати критичну помилку.
            if not entry_thread.is_alive() and vehicle_event_hndl.is_running:
                logger.error("Потік В'ЇЗДУ несподівано завершився! Спроба перезапуску (не реалізовано в заглушці).")
                # vehicle_event_hndl.is_running = False # Або інша логіка
            if not exit_thread.is_alive() and vehicle_event_hndl.is_running:
                logger.error("Потік ВИЇЗДУ несподівано завершився! Спроба перезапуску (не реалізовано в заглушці).")
                # vehicle_event_hndl.is_running = False # Або інша логіка

            if not vehicle_event_hndl.is_running:  # Якщо обробник сам попросив зупинку
                shutdown_event.set()

            time.sleep(10)  # Періодичність перевірки

    except KeyboardInterrupt:  # Обробка Ctrl+C, хоча signal_handler має спрацювати першим
        logger.info("Отримано KeyboardInterrupt. Завершення роботи...")
        shutdown_event.set()
    finally:
        logger.info("Початок процедури завершення роботи...")
        if 'vehicle_event_hndl' in locals() and vehicle_event_hndl:
            vehicle_event_hndl.stop()  # Сигналізуємо обробникам подій про зупинку

        if entry_thread.is_alive():
            logger.info("Очікування завершення потоку В'ЇЗДУ...")
            entry_thread.join(timeout=5)
        if exit_thread.is_alive():
            logger.info("Очікування завершення потоку ВИЇЗДУ...")
            exit_thread.join(timeout=5)

        logger.info("Потоки обробки завершено або вийшов час очікування.")

        # 4. Очищення ресурсів
        logger.info("Очищення ресурсів...")
        if 'cam_entry' in locals() and cam_entry:
            cam_entry.close()
            logger.info("Камера В'ЇЗДУ закрита.")
        if 'cam_exit' in locals() and cam_exit:
            cam_exit.close()
            logger.info("Камера ВИЇЗДУ закрита.")

        # GateController має власний метод cleanup, який викликається в __del__
        # Але для певності можна викликати його явно, якщо об'єкт ще існує
        if 'gate_ctrl' in locals() and gate_ctrl:
            gate_ctrl.cleanup()  # Це викличе cleanup реле та таймерів
            logger.info("GateController cleanup викликано.")

        # SensorManager також може потребувати cleanup, якщо він керує GPIO напряму
        if 'sensor_mgr' in locals() and sensor_mgr:
            if hasattr(sensor_mgr, 'cleanup'):
                sensor_mgr.cleanup()
            logger.info("SensorManager cleanup (якщо є) викликано.")

        logger.info("Система керування воротами завершила роботу.")


# --- Точка входу ---
if __name__ == "__main__":
    main_logger = setup_logging()  # Ініціалізуємо логер і отримуємо головний екземпляр

    # Налаштування обробників сигналів для коректного завершення
    signal.signal(signal.SIGINT, signal_handler)  # Обробка Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Обробка сигналу завершення (напр., від systemd)

    # Перевірка наявності директорій для зображень
    os.makedirs("captured_images/entry", exist_ok=True)
    os.makedirs("captured_images/exit", exist_ok=True)

    try:
        main_application()
    except Exception as e:
        main_logger.critical(f"Неперехоплений виняток на глобальному рівні: {e}", exc_info=True)
    finally:
        logging.shutdown()  # Закриваємо всі хендлери логування
        logging.shutdown()  # Закриваємо всі хендлери логування