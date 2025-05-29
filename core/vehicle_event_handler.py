# core/vehicle_event_handler.py
import logging
import time
import threading
import os
import numpy as np
import cv2  # Для збереження зображення, якщо CVProcessor не зміг (хоча він тепер зберігає сам)

# Імпорт з image_utils
from utils.image_utils import save_image

# ... (решта імпортів та класу VehicleEventHandler) ...

logger = logging.getLogger(__name__)

DEFAULT_SHEETS_ANTIDUPLICATE_DELAY_S = 60
DEFAULT_PASSAGE_CONFIRMATION_TIMEOUT_S = 20
DEFAULT_POLL_INTERVAL_IDLE_S = 1.0
DEFAULT_POLL_INTERVAL_GATE_CLOSING_S = 0.3

CAPTURED_IMAGES_BASE_PATH = "captured_images"  # Має бути узгоджено з main.py
ENTRY_IMAGES_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "entry")
EXIT_IMAGES_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "exit")
CV_DEBUG_SAVE_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "cv_debug")  # Для CVProcessor


class VehicleEventHandler:
    def __init__(self,
                 camera_entry,
                 camera_exit,
                 sensor_manager,
                 sheet_handler,
                 cv_processor,
                 gate_controller,
                 config: dict = None):

        self._logger = logging.getLogger(f"{__name__}.VehicleEventHandler")
        self.camera_entry = camera_entry
        self.camera_exit = camera_exit
        self.sensor_manager = sensor_manager
        self.sheet_handler = sheet_handler
        self.cv_processor = cv_processor
        self.gate_controller = gate_controller

        self.config = config if config else {}
        self.sheets_antiduplicate_delay_s = self.config.get(
            "sheets_antiduplicate_delay_s", DEFAULT_SHEETS_ANTIDUPLICATE_DELAY_S
        )
        self.passage_confirmation_timeout_s = self.config.get(
            "passage_confirmation_timeout_s", DEFAULT_PASSAGE_CONFIRMATION_TIMEOUT_S
        )
        self.poll_interval_idle_s = self.config.get(
            "poll_interval_idle_s", DEFAULT_POLL_INTERVAL_IDLE_S
        )
        self.poll_interval_gate_closing_s = self.config.get(
            "poll_interval_gate_closing_s", DEFAULT_POLL_INTERVAL_GATE_CLOSING_S
        )

        self.is_running = False
        self.shutdown_event = None

        self.recently_logged_plates = {}
        self._plate_cache_lock = threading.Lock()

        # Створюємо директорії, якщо їх немає (хоча main.py теж може це робити)
        os.makedirs(ENTRY_IMAGES_PATH, exist_ok=True)
        os.makedirs(EXIT_IMAGES_PATH, exist_ok=True)
        os.makedirs(os.path.join(CV_DEBUG_SAVE_PATH, "entry"), exist_ok=True)
        os.makedirs(os.path.join(CV_DEBUG_SAVE_PATH, "exit"), exist_ok=True)

        self._logger.info("VehicleEventHandler ініціалізовано.")
        self._logger.info(f"  Анти-дублікат для Sheets: {self.sheets_antiduplicate_delay_s} с")
        self._logger.info(f"  Таймаут підтвердження проїзду УЗД: {self.passage_confirmation_timeout_s} с")

    def _is_duplicate_log(self, plate_number: str) -> bool:
        # ... (без змін) ...
        with self._plate_cache_lock:
            now = time.time()
            if plate_number in self.recently_logged_plates:
                last_log_time = self.recently_logged_plates[plate_number]
                if (now - last_log_time) < self.sheets_antiduplicate_delay_s:
                    self._logger.info(
                        f"Дублікатний запис для НЗ '{plate_number}'. Минуло менше {self.sheets_antiduplicate_delay_s} с.")
                    return True
            self.recently_logged_plates[plate_number] = now
            return False

    def _wait_for_vehicle_to_pass(self, gate_side_name: str) -> bool:
        # ... (без змін) ...
        self._logger.info(f"Очікування проїзду автомобіля через ворота ({gate_side_name})...")
        if self.sensor_manager and self.sensor_manager.ultrasonic_sensor_entry:
            if self.sensor_manager.ultrasonic_sensor_entry.wait_for_clear_after_pass(
                    timeout=self.passage_confirmation_timeout_s
            ):
                self._logger.info(f"Автомобіль проїхав зону воріт ({gate_side_name}).")
                return True
            else:
                self._logger.warning(f"Тайм-аут очікування проїзду автомобіля ({gate_side_name}). "
                                     "Можливо, автомобіль зупинився у проїзді.")
                return False
        else:
            self._logger.error("Ультразвуковий датчик для контролю проїзду недоступний.")
            return False

    def _handle_gate_closing_interruption(self, camera_type_for_check: str):
        # ... (без змін) ...
        is_timer_active = False
        if hasattr(self.gate_controller, '_auto_close_timer') and \
                self.gate_controller._auto_close_timer is not None and \
                self.gate_controller._auto_close_timer.is_alive():
            is_timer_active = True

        if is_timer_active:
            vehicle_detected_by_cv = False
            if self.cv_processor:
                cam_to_check = self.camera_entry if camera_type_for_check == "entry" else self.camera_exit
                if cam_to_check and cam_to_check.is_initialized_successfully:
                    current_frame = cam_to_check.capture_array()
                    if current_frame is not None:
                        detections = self.cv_processor.detect_vehicle_in_frame(current_frame, camera_type_for_check)
                        if detections:
                            vehicle_detected_by_cv = True
                            self._logger.info(
                                f"CV ВИЯВИВ НОВЕ АВТО ({camera_type_for_check}) під час таймера закриття!")

            vehicle_detected_by_ultrasonic = False
            ultrasonic_to_check = None
            if camera_type_for_check == "entry" and self.sensor_manager.ultrasonic_sensor_entry:
                ultrasonic_to_check = self.sensor_manager.ultrasonic_sensor_entry
            elif camera_type_for_check == "exit":
                ultrasonic_to_check = getattr(self.sensor_manager, 'ultrasonic_sensor_exit',
                                              self.sensor_manager.ultrasonic_sensor_entry)

            if ultrasonic_to_check and ultrasonic_to_check.is_vehicle_approaching():
                vehicle_detected_by_ultrasonic = True
                self._logger.info(f"УЗД ВИЯВИВ НОВЕ АВТО ({camera_type_for_check}) під час таймера закриття!")

            if vehicle_detected_by_cv or vehicle_detected_by_ultrasonic:
                self._logger.warning(
                    f"НОВИЙ АВТОМОБІЛЬ ({camera_type_for_check}) ВИЯВЛЕНО ПІД ЧАС АКТИВНОГО ТАЙМЕРА ЗАКРИТТЯ!")
                self.gate_controller.interrupt_closing_procedure()
                return True
        return False

    def entry_scenario_loop(self):
        self._logger.info("Запуск циклу обробки В'ЇЗДУ...")
        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            current_poll_interval = self.poll_interval_idle_s
            if hasattr(self.gate_controller, '_auto_close_timer') and \
                    self.gate_controller._auto_close_timer is not None and \
                    self.gate_controller._auto_close_timer.is_alive():
                current_poll_interval = self.poll_interval_gate_closing_s
                if self._handle_gate_closing_interruption("entry"):
                    self._logger.info("В'їзд: Закриття перервано новим авто. Повторний цикл детекції.")
                    time.sleep(0.1)
                    continue

            self._logger.debug("В'їзд: Очікування автомобіля...")

            vehicle_approaching = False
            initial_detection_frame = None

            if self.camera_entry and self.camera_entry.is_initialized_successfully and self.cv_processor:
                frame_for_detection = self.camera_entry.capture_array()
                if frame_for_detection is not None:
                    vehicle_detections = self.cv_processor.detect_vehicle_in_frame(frame_for_detection,
                                                                                   camera_type="entry")
                    if vehicle_detections:
                        self._logger.info("В'їзд: CV зафіксував автомобіль.")
                        vehicle_approaching = True
                        initial_detection_frame = frame_for_detection

            if not vehicle_approaching and self.sensor_manager.ultrasonic_sensor_entry:
                if self.sensor_manager.ultrasonic_sensor_entry.is_vehicle_approaching():
                    self._logger.info("В'їзд: УЗД зафіксував наближення.")
                    vehicle_approaching = True

            if vehicle_approaching:
                self._logger.info("В'їзд: Автомобіль виявлено. Початок обробки.")

                if initial_detection_frame is None and self.camera_entry and self.camera_entry.is_initialized_successfully:
                    initial_detection_frame = self.camera_entry.capture_array()

                if initial_detection_frame is None:
                    self._logger.error("В'їзд: Не вдалося отримати кадр для розпізнавання НЗ.")
                    time.sleep(current_poll_interval)
                    continue

                timestamp_str = time.strftime('%Y%m%d_%H%M%S')  # Для імені файлу
                plate_text = self.cv_processor.get_plate_number_from_image(
                    initial_detection_frame, camera_type="entry",
                    save_intermediate_steps=True,
                    save_path_prefix=os.path.join(CV_DEBUG_SAVE_PATH, "entry")
                )

                if plate_text:
                    if not self._is_duplicate_log(plate_text):
                        if self.sheet_handler.find_vehicle_and_update_entry_time(plate_text):
                            self._logger.info(f"В'їзд: Автомобіль '{plate_text}' АВТОРИЗОВАНО. Відкриття воріт.")
                            if self.gate_controller.open_gate():
                                if self._wait_for_vehicle_to_pass("в'їзду"):
                                    self.gate_controller.start_auto_close_timer()
                                else:
                                    self._logger.warning(
                                        "В'їзд: Авто не підтвердило проїзд. Таймер закриття не запущено.")
                        else:
                            self._logger.info(f"В'їзд: Автомобіль '{plate_text}' НЕ АВТОРИЗОВАНО.")
                            self.sheet_handler.add_unauthorized_attempt(plate_text)
                else:
                    self._logger.warning("В'їзд: Номерний знак не розпізнано.")
                    # Зберігаємо зображення з невдалим розпізнаванням
                    failed_filename = f"entry_ocr_failed_{timestamp_str}.jpg"
                    save_image(initial_detection_frame, ENTRY_IMAGES_PATH, failed_filename)
                    self._logger.info(
                        f"Збережено зображення з невдалим OCR (в'їзд): {os.path.join(ENTRY_IMAGES_PATH, failed_filename)}")

            time.sleep(current_poll_interval)
        self._logger.info("Цикл обробки В'ЇЗДУ завершено.")

    def exit_scenario_loop(self):
        # ... (Аналогічні зміни для збереження зображень та використання image_utils) ...
        self._logger.info("Запуск циклу обробки ВИЇЗДУ...")

        ultrasonic_exit_sensor = getattr(self.sensor_manager, 'ultrasonic_sensor_exit', None) \
                                 or self.sensor_manager.ultrasonic_sensor_entry

        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            current_poll_interval = self.poll_interval_idle_s
            if hasattr(self.gate_controller, '_auto_close_timer') and \
                    self.gate_controller._auto_close_timer is not None and \
                    self.gate_controller._auto_close_timer.is_alive():
                current_poll_interval = self.poll_interval_gate_closing_s
                if self._handle_gate_closing_interruption("exit"):
                    self._logger.info("Виїзд: Закриття перервано новим авто. Повторний цикл детекції.")
                    time.sleep(0.1)
                    continue

            self._logger.debug("Виїзд: Очікування автомобіля...")

            vehicle_approaching_exit = False
            initial_detection_frame_exit = None

            if self.camera_exit and self.camera_exit.is_initialized_successfully and self.cv_processor:
                frame_for_detection = self.camera_exit.capture_array()
                if frame_for_detection is not None:
                    vehicle_detections = self.cv_processor.detect_vehicle_in_frame(frame_for_detection,
                                                                                   camera_type="exit")
                    if vehicle_detections:
                        self._logger.info("Виїзд: CV зафіксував автомобіль.")
                        vehicle_approaching_exit = True
                        initial_detection_frame_exit = frame_for_detection

            if not vehicle_approaching_exit and ultrasonic_exit_sensor:
                if ultrasonic_exit_sensor.is_vehicle_approaching():
                    self._logger.info("Виїзд: УЗД зафіксував наближення.")
                    vehicle_approaching_exit = True

            if vehicle_approaching_exit:
                self._logger.info("Виїзд: Автомобіль виявлено. Негайне відкриття воріт.")
                if self.gate_controller.open_gate():
                    if initial_detection_frame_exit is None and self.camera_exit and self.camera_exit.is_initialized_successfully:
                        initial_detection_frame_exit = self.camera_exit.capture_array()

                    if initial_detection_frame_exit is None:
                        self._logger.error(
                            "Виїзд: Не вдалося отримати кадр для розпізнавання НЗ після відкриття воріт.")
                    else:
                        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                        plate_text = self.cv_processor.get_plate_number_from_image(
                            initial_detection_frame_exit, camera_type="exit",
                            save_intermediate_steps=True,
                            save_path_prefix=os.path.join(CV_DEBUG_SAVE_PATH, "exit")
                        )
                        if plate_text:
                            if not self._is_duplicate_log(plate_text):
                                self.sheet_handler.log_vehicle_exit(plate_text)
                        else:
                            self._logger.warning("Виїзд: Номерний знак не розпізнано, але виїзд дозволено.")
                            failed_filename = f"exit_ocr_failed_{timestamp_str}.jpg"
                            save_image(initial_detection_frame_exit, EXIT_IMAGES_PATH, failed_filename)
                            self._logger.info(
                                f"Збережено зображення з невдалим OCR (виїзд): {os.path.join(EXIT_IMAGES_PATH, failed_filename)}")

                    if self._wait_for_vehicle_to_pass("виїзду"):
                        self.gate_controller.start_auto_close_timer()
                    else:
                        self._logger.warning(
                            "Виїзд: Авто не підтвердило проїзд після відкриття. Таймер закриття не запущено.")

            time.sleep(current_poll_interval)
        self._logger.info("Цикл обробки ВИЇЗДУ завершено.")

    def start(self, shutdown_event_main):
        # ... (без змін) ...
        if self.is_running:
            self._logger.warning("VehicleEventHandler вже запущено.")
            return

        self._logger.info("Запуск VehicleEventHandler...")
        self.is_running = True
        self.shutdown_event = shutdown_event_main

        self.entry_thread = threading.Thread(target=self.entry_scenario_loop, name="EntryScenarioThread")
        self.exit_thread = threading.Thread(target=self.exit_scenario_loop, name="ExitScenarioThread")

        self.entry_thread.daemon = True
        self.exit_thread.daemon = True

        self.entry_thread.start()
        self.exit_thread.start()
        self._logger.info("Потоки обробки в'їзду та виїзду запущено.")

    def stop(self):
        # ... (без змін) ...
        if not self.is_running:
            self._logger.info("VehicleEventHandler вже зупинено або не було запущено.")
            return

        self._logger.info("Зупинка VehicleEventHandler...")
        self.is_running = False
        self._logger.info("VehicleEventHandler отримав сигнал на зупинку.")


# --- Блок для тестування ---
if __name__ == '__main__':
    # ... (Мок-класи та логіка тестування залишаються такими ж, як у попередній версії) ...
    # ... (Але тепер VehicleEventHandler використовує image_utils.save_image) ...
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    from unittest.mock import MagicMock

    logger_main_test = logging.getLogger("EventHandlerTest")


    class MockCamera:
        def __init__(self, name="mock_cam"):
            self.name = name
            self.is_initialized_successfully = True
            self._logger = logging.getLogger(f"MockCamera.{name}")
            self._logger.info(f"Мок-камера '{name}' створена.")
            self.image_counter = 0

        def capture_array(self):
            self.image_counter += 1
            self._logger.info(f"[{self.name}] capture_array() викликано (кадр {self.image_counter})")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"Frame {self.image_counter} {self.name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            return dummy_frame

        def capture_image(self, filename):
            self._logger.info(f"[{self.name}] capture_image({filename}) викликано")
            return filename

        def close(self): self._logger.info(f"[{self.name}] close() викликано")


    class MockUltrasonicSensor:
        def __init__(self, name="mock_us"):
            self.name = name
            self._logger = logging.getLogger(f"MockUltrasonic.{name}")
            self.should_detect_approach = False
            self.should_confirm_pass = True

        def is_vehicle_approaching(self, threshold_m=None):
            time.sleep(0.01)
            return self.should_detect_approach

        def wait_for_clear_after_pass(self, timeout=None):
            self._logger.info(
                f"[{self.name}] wait_for_clear_after_pass(timeout={timeout}) -> {self.should_confirm_pass}")
            if self.should_confirm_pass:
                time.sleep(0.1)
            return self.should_confirm_pass

        def cleanup(self): self._logger.info(f"[{self.name}] cleanup() викликано")


    class MockSensorManager:
        def __init__(self):
            self.ultrasonic_sensor_entry = MockUltrasonicSensor("US_Entry")
            self.ultrasonic_sensor_exit = MockUltrasonicSensor("US_Exit")
            self._logger = logging.getLogger("MockSensorManager")
            self._logger.info("Мок-менеджер датчиків створено.")

        def cleanup(self): self._logger.info("MockSensorManager cleanup() викликано.")


    class MockSheetHandler:
        def __init__(self): self._logger = logging.getLogger("MockSheetHandler")

        def find_vehicle_and_update_entry_time(self, plate_number):
            self._logger.info(f"SHEETS: find_vehicle_and_update_entry_time для '{plate_number}'")
            if "AUTH" in plate_number: return True
            return False

        def add_unauthorized_attempt(self, plate_number):
            self._logger.info(f"SHEETS: add_unauthorized_attempt для '{plate_number}'")

        def log_vehicle_exit(self, plate_number):
            self._logger.info(f"SHEETS: log_vehicle_exit для '{plate_number}'")


    class MockCVProcessor:
        def __init__(self):
            self._logger = logging.getLogger("MockCVProcessor")
            self.simulated_plate = "AUTH123AA"
            self.should_detect_vehicle_map = {"entry": False, "exit": False}
            self.should_recognize_plate = True

        def detect_vehicle_in_frame(self, image_bgr, camera_type, **kwargs):  # Додав **kwargs для сумісності
            detect = self.should_detect_vehicle_map.get(camera_type, False)
            self._logger.info(f"CV: detect_vehicle_in_frame (камера: {camera_type}) -> {detect}")
            if detect:
                return [(10, 10, 100, 100, 0.9, "car")]
            return None

        def get_plate_number_from_image(self, image_bgr, camera_type, save_intermediate_steps=False,
                                        save_path_prefix=""):
            self._logger.info(f"CV: get_plate_number_from_image (камера: {camera_type})")
            if self.should_recognize_plate:
                self._logger.info(f"CV: Імітація розпізнавання -> '{self.simulated_plate}'")
                return self.simulated_plate
            self._logger.info("CV: Імітація НЕ розпізнавання НЗ")
            return None


    class MockGateController:
        def __init__(self):
            self._logger = logging.getLogger("MockGateController")
            self._auto_close_timer_obj = None
            self._lock = threading.Lock()

        def open_gate(self):
            self._logger.info("GATE: open_gate() викликано")
            if self._auto_close_timer_obj and self._auto_close_timer_obj.is_alive():
                self._logger.info("GATE: open_gate() перериває активний таймер закриття.")
                self._auto_close_timer_obj.cancel()
                self._auto_close_timer_obj = None
            return True

        def close_gate(self):
            self._logger.info("GATE: close_gate() викликано")
            return True

        def _timer_callback_mock(self):
            with self._lock:
                self._logger.info("GATE: (Мок) Таймер авто-закриття спрацював.")
                self._auto_close_timer_obj = None

        def start_auto_close_timer(self, timeout_s=None):
            with self._lock:
                effective_timeout = timeout_s if timeout_s is not None else 1
                self._logger.info(f"GATE: start_auto_close_timer(timeout={effective_timeout}) викликано")
                if self._auto_close_timer_obj and self._auto_close_timer_obj.is_alive():
                    self._auto_close_timer_obj.cancel()

                self._auto_close_timer_obj = threading.Timer(effective_timeout, self._timer_callback_mock)
                self._auto_close_timer_obj.daemon = True
                self._auto_close_timer_obj.start()
                self._logger.info("GATE: (Мок) Таймер авто-закриття запущено.")

        def interrupt_closing_procedure(self):
            with self._lock:
                self._logger.info("GATE: interrupt_closing_procedure() викликано")
                if self._auto_close_timer_obj and self._auto_close_timer_obj.is_alive():
                    self._auto_close_timer_obj.cancel()
                    self._logger.info("GATE: (Мок) Активний таймер закриття скасовано.")
                self._auto_close_timer_obj = None

        def get_current_gate_state(self):
            return "UNKNOWN"

        def cleanup(self):
            self._logger.info("GATE: cleanup() викликано")


    logger_main_test = logging.getLogger("EventHandlerTest")
    logger_main_test.info("--- Початок тестування VehicleEventHandler ---")

    mock_cam_entry = MockCamera("EntryCam")
    mock_cam_exit = MockCamera("ExitCam")
    mock_sensors = MockSensorManager()
    mock_sheets = MockSheetHandler()
    mock_cv = MockCVProcessor()
    mock_gate = MockGateController()

    test_config = {
        "sheets_antiduplicate_delay_s": 2,
        "passage_confirmation_timeout_s": 1.5,
        "poll_interval_idle_s": 0.2,
        "poll_interval_gate_closing_s": 0.1
    }

    event_handler = VehicleEventHandler(
        camera_entry=mock_cam_entry, camera_exit=mock_cam_exit,
        sensor_manager=mock_sensors, sheet_handler=mock_sheets,
        cv_processor=mock_cv, gate_controller=mock_gate, config=test_config
    )

    test_shutdown_event = threading.Event()
    event_handler.start(test_shutdown_event)

    try:
        logger_main_test.info("\n>>> СЦЕНАРІЙ 1: Авторизований в'їзд <<<")
        mock_cv.should_detect_vehicle_map["entry"] = True
        mock_cv.simulated_plate = "AUTH123AA"
        mock_cv.should_recognize_plate = True
        mock_sensors.ultrasonic_sensor_entry.should_detect_approach = True

        time.sleep(2.5)  # Даємо час на повний цикл + спрацювання таймера (1.5+1)

        mock_cv.should_detect_vehicle_map["entry"] = False
        mock_sensors.ultrasonic_sensor_entry.should_detect_approach = False
        logger_main_test.info(">>> Сценарій 1 завершено <<<")

        logger_main_test.info("\n>>> СЦЕНАРІЙ 2: Переривання закриття новим авто на В'ЇЗДІ <<<")
        # Спочатку запускаємо нормальний проїзд, щоб запустився таймер
        mock_cv.should_detect_vehicle_map["entry"] = True
        mock_cv.simulated_plate = "AUTH_TIMER_START"
        mock_sensors.ultrasonic_sensor_entry.should_detect_approach = True
        mock_sensors.ultrasonic_sensor_entry.should_confirm_pass = True
        time.sleep(0.5)  # Час на детекцію та відкриття
        mock_sensors.ultrasonic_sensor_entry.should_detect_approach = False  # Авто "проїжджає" повз датчик наближення
        time.sleep(2)  # Час на підтвердження проїзду (1.5с) та запуск таймера (1с)
        # Тепер таймер mock_gate має бути активним

        if hasattr(mock_gate, '_auto_close_timer_obj') and \
                mock_gate._auto_close_timer_obj is not None and \
                mock_gate._auto_close_timer_obj.is_alive():
            logger_main_test.info("Таймер закриття активний. Симулюємо нове авто для переривання...")
            mock_cv.simulated_plate = "INTERRUPT_CAR"
            mock_cv.should_detect_vehicle_map["entry"] = True  # Нове авто з'явилося
            mock_sensors.ultrasonic_sensor_entry.should_detect_approach = True
            time.sleep(1)  # Дати час на реакцію і переривання
        else:
            logger_main_test.info("Таймер закриття НЕ активний для тесту переривання. Перевірте логіку мок-таймера.")

        mock_cv.should_detect_vehicle_map["entry"] = False
        mock_sensors.ultrasonic_sensor_entry.should_detect_approach = False
        logger_main_test.info(">>> Сценарій 2 завершено <<<")

        time.sleep(3)  # Загальний час роботи для інших тестів

    except KeyboardInterrupt:
        logger_main_test.info("Тест перервано користувачем.")
    finally:
        logger_main_test.info("Завершення тестів VehicleEventHandler...")
        test_shutdown_event.set()
        event_handler.stop()

        if hasattr(event_handler, 'entry_thread') and event_handler.entry_thread.is_alive():
            event_handler.entry_thread.join(timeout=2)
        if hasattr(event_handler, 'exit_thread') and event_handler.exit_thread.is_alive():
            event_handler.exit_thread.join(timeout=2)

        logger_main_test.info("Тестування завершено.")